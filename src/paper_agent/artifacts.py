"""Content-addressed artifacts and coordinator-owned shard result merging."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import mimetypes
import os
from pathlib import Path, PurePath, PureWindowsPath
import tempfile
from typing import Iterable, Mapping

from paper_agent.canonical import content_hash
from paper_agent.identity import manual_queue_id
from paper_agent.sharding import (
    ArtifactManifest,
    ArtifactReceipt,
    MergePlan,
    MergeRejected,
    MergedArtifact,
    ShardManifest,
    WorkerResultManifest,
    plan_merge,
)
from paper_agent.storage import Database


class ArtifactMetadataConflict(ValueError):
    """One content digest was associated with incompatible metadata."""


@dataclass(frozen=True)
class StoredArtifact:
    artifact_hash: str
    mime_type: str
    size_bytes: int
    relative_path: str
    path: Path


@dataclass(frozen=True)
class ReceivedArtifact:
    manifest: ArtifactManifest
    receipt: ArtifactReceipt
    payload: bytes


@dataclass(frozen=True)
class MergeApplyResult:
    plan: MergePlan | None
    applied: bool
    manual_queue_ids: tuple[str, ...] = ()


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _mime_type(payload: bytes, suffix: str = "") -> str:
    if payload.startswith(b"%PDF-"):
        return "application/pdf"
    guessed, _ = mimetypes.guess_type(f"artifact{suffix}")
    if guessed:
        return guessed
    stripped = payload.lstrip()
    if stripped.startswith((b"{", b"[")):
        return "application/json"
    try:
        payload.decode("utf-8")
    except UnicodeDecodeError:
        return "application/octet-stream"
    return "text/plain"


def _relative_path(path: str) -> None:
    if not path or PurePath(path).is_absolute() or PureWindowsPath(path).is_absolute():
        raise MergeRejected("paths must be non-empty and relative")
    if ".." in PurePath(path).parts or ".." in PureWindowsPath(path).parts:
        raise MergeRejected("paths must not escape their output root")


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class ArtifactStore:
    """A local immutable store rooted at ``artifacts/<prefix>/<sha256>``."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.artifacts_root = self.root / "artifacts"

    def relative_path(self, artifact_hash: str) -> str:
        return f"artifacts/{artifact_hash[:2]}/{artifact_hash}"

    def path_for(self, artifact_hash: str) -> Path:
        if len(artifact_hash) != 64 or any(char not in "0123456789abcdef" for char in artifact_hash):
            raise ValueError("artifact_hash must be a lowercase SHA-256 digest")
        return self.root / self.relative_path(artifact_hash)

    def put_bytes(
        self,
        payload: bytes,
        *,
        mime_type: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> StoredArtifact:
        artifact_hash = _digest(payload)
        resolved_mime_type = mime_type or _mime_type(payload)
        record = {
            "mime_type": resolved_mime_type,
            "size_bytes": len(payload),
            "metadata": dict(metadata or {}),
        }
        path = self.path_for(artifact_hash)
        metadata_path = path.with_name(f"{path.name}.metadata.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if _digest(path.read_bytes()) != artifact_hash:
                raise ValueError("artifact store contains corrupted content")
        else:
            self._write_atomically(path, payload)
        if metadata_path.exists():
            existing = json.loads(metadata_path.read_text(encoding="utf-8"))
            if existing != record:
                raise ArtifactMetadataConflict("artifact content already exists with different metadata")
        else:
            self._write_atomically(metadata_path, _json(record).encode("utf-8"))
        return StoredArtifact(
            artifact_hash=artifact_hash,
            mime_type=resolved_mime_type,
            size_bytes=len(payload),
            relative_path=self.relative_path(artifact_hash),
            path=path,
        )

    def put_file(
        self,
        path: str | Path,
        *,
        mime_type: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> StoredArtifact:
        source = Path(path)
        payload = source.read_bytes()
        return self.put_bytes(
            payload,
            mime_type=mime_type or _mime_type(payload, source.suffix),
            metadata=metadata,
        )

    def read_bytes(self, artifact_hash: str) -> bytes:
        payload = self.path_for(artifact_hash).read_bytes()
        if _digest(payload) != artifact_hash:
            raise ValueError("artifact store contains corrupted content")
        return payload

    @staticmethod
    def _write_atomically(path: Path, payload: bytes) -> None:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, path)


class CoordinatorArtifactMerger:
    """Receive worker bundles and atomically apply a complete merge plan to SQLite."""

    def __init__(self, database: Database, artifact_store: ArtifactStore) -> None:
        self.database = database
        self.artifact_store = artifact_store

    def receive_bundle(
        self, result: WorkerResultManifest, worker_root: str | Path
    ) -> tuple[ReceivedArtifact, ...]:
        root = Path(worker_root)
        resolved_root = root.resolve()
        received: list[ReceivedArtifact] = []
        for manifest in result.artifacts:
            _relative_path(manifest.relative_path)
            source = (root / manifest.relative_path).resolve()
            if not source.is_relative_to(resolved_root) or not source.is_file():
                raise MergeRejected("worker artifact is outside its bundle root or missing")
            payload = source.read_bytes()
            received.append(
                ReceivedArtifact(
                    manifest=manifest,
                    receipt=ArtifactReceipt(
                        artifact_hash=_digest(payload),
                        mime_type=_mime_type(payload, PurePath(manifest.relative_path).suffix),
                        size_bytes=len(payload),
                    ),
                    payload=payload,
                )
            )
        return tuple(received)

    def apply(
        self,
        shards: Iterable[ShardManifest],
        results: Iterable[WorkerResultManifest],
        worker_roots: Mapping[str, str | Path] | str | Path,
        *,
        stage: str,
        output_kinds: Iterable[str],
    ) -> MergeApplyResult:
        shard_list = tuple(shards)
        result_list = tuple(results)
        try:
            received = self._receive_all(result_list, worker_roots)
            receipts = {item.manifest.artifact_hash: item.receipt for item in received}
            received_by_hash = {item.manifest.artifact_hash: item for item in received}
            plan = plan_merge(
                shard_list,
                result_list,
                receipts,
                stage=stage,
                output_kinds=output_kinds,
                existing_outputs=self._existing_outputs(stage),
            )
        except MergeRejected as error:
            queue_id = self._enqueue_rejection(result_list, stage, str(error))
            return MergeApplyResult(plan=None, applied=False, manual_queue_ids=(queue_id,))

        if plan.conflicts:
            queue_ids = self._enqueue_conflicts(plan)
            return MergeApplyResult(plan=plan, applied=False, manual_queue_ids=queue_ids)
        if not plan.complete:
            return MergeApplyResult(plan=plan, applied=False)

        stored = {
            output.artifact_hash: self.artifact_store.put_bytes(
                received_by_hash[output.artifact_hash].payload,
                mime_type=output.mime_type,
            )
            for output in plan.new_outputs
        }
        with self.database.transaction() as connection:
            for output in plan.new_outputs:
                artifact_id = self._save_artifact(connection, output, stored[output.artifact_hash])
                connection.execute(
                    """INSERT INTO run_outputs(
                        run_output_id, run_id, stage, paper_id, output_kind, content_hash, artifact_id, shard_epoch
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        self._run_output_id(output),
                        output.run_id,
                        output.stage,
                        output.paper_id,
                        output.output_kind,
                        output.artifact_hash,
                        artifact_id,
                        self._epoch_for(output, result_list),
                    ),
                )
        return MergeApplyResult(plan=plan, applied=True)

    def _receive_all(
        self,
        results: tuple[WorkerResultManifest, ...],
        worker_roots: Mapping[str, str | Path] | str | Path,
    ) -> tuple[ReceivedArtifact, ...]:
        received: list[ReceivedArtifact] = []
        for result in results:
            if isinstance(worker_roots, Mapping):
                if result.shard_id not in worker_roots:
                    raise MergeRejected("worker bundle root is missing")
                root = worker_roots[result.shard_id]
            else:
                root = worker_roots
            received.extend(self.receive_bundle(result, root))
        return tuple(received)

    def _existing_outputs(self, stage: str) -> dict[tuple[str, str, str, str], MergedArtifact]:
        rows = self.database.connection.execute(
            """SELECT ro.run_id, ro.stage, ro.paper_id, ro.output_kind,
                      a.sha256, a.mime_type, a.byte_size, a.relative_path
               FROM run_outputs ro JOIN artifacts a ON a.artifact_id = ro.artifact_id
               WHERE ro.stage = ?""",
            (stage,),
        ).fetchall()
        return {
            (row["run_id"], row["stage"], row["paper_id"], row["output_kind"]): MergedArtifact(
                row["run_id"], row["stage"], row["paper_id"], row["output_kind"], row["sha256"],
                row["mime_type"], row["byte_size"], row["relative_path"],
            )
            for row in rows
        }

    def _save_artifact(self, connection, output: MergedArtifact, stored: StoredArtifact) -> str:
        existing = connection.execute(
            "SELECT artifact_id, mime_type, byte_size, relative_path FROM artifacts WHERE sha256 = ?",
            (stored.artifact_hash,),
        ).fetchone()
        if existing:
            if (
                existing["mime_type"] != stored.mime_type
                or existing["byte_size"] != stored.size_bytes
                or existing["relative_path"] != stored.relative_path
            ):
                raise ArtifactMetadataConflict("artifact database metadata conflicts with content digest")
            return str(existing["artifact_id"])
        artifact_id = f"artifact-{stored.artifact_hash}"
        kind = output.output_kind if output.output_kind in {"pdf", "supplement", "text", "analysis", "report", "manifest"} else "other"
        connection.execute(
            """INSERT INTO artifacts(
                artifact_id, paper_id, artifact_kind, relative_path, mime_type, byte_size, sha256, provenance_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                artifact_id,
                output.paper_id,
                kind,
                stored.relative_path,
                stored.mime_type,
                stored.size_bytes,
                stored.artifact_hash,
                _json({"run_id": output.run_id, "stage": output.stage, "output_kind": output.output_kind}),
            ),
        )
        return artifact_id

    @staticmethod
    def _run_output_id(output: MergedArtifact) -> str:
        return "run-output-" + content_hash(output.key)

    @staticmethod
    def _epoch_for(output: MergedArtifact, results: tuple[WorkerResultManifest, ...]) -> int:
        for result in results:
            if result.run_id == output.run_id and output.paper_id in result.paper_ids:
                return result.epoch
        raise ValueError("merged output has no worker result")

    def _enqueue_rejection(
        self, results: tuple[WorkerResultManifest, ...], stage: str, reason: str
    ) -> str:
        result = results[0] if results else None
        run_id = result.run_id if result else None
        key = f"rejected:{run_id}:{stage}:{result.shard_id if result else 'unknown'}:{result.epoch if result else 'unknown'}"
        queue_id = manual_queue_id("merge_conflict", key)
        with self.database.transaction() as connection:
            connection.execute(
                """INSERT INTO manual_queue(manual_queue_id, queue_type, dedup_key, paper_id, run_id, reason_json, status)
                   VALUES (?, 'merge_conflict', ?, NULL, ?, ?, 'pending')
                   ON CONFLICT(queue_type, dedup_key) DO NOTHING""",
                (queue_id, key, run_id, _json({"stage": stage, "reason": reason})),
            )
        return queue_id

    def _enqueue_conflicts(self, plan: MergePlan) -> tuple[str, ...]:
        queue_ids: list[str] = []
        with self.database.transaction() as connection:
            for conflict in plan.conflicts:
                run_id, stage, paper_id, output_kind = conflict.key
                key = f"{run_id}:{stage}:{paper_id}:{output_kind}:{conflict.existing_hash}:{conflict.incoming_hash}"
                queue_id = manual_queue_id("merge_conflict", key)
                connection.execute(
                    """INSERT INTO manual_queue(manual_queue_id, queue_type, dedup_key, paper_id, run_id, reason_json, status)
                       VALUES (?, 'merge_conflict', ?, ?, ?, ?, 'pending')
                       ON CONFLICT(queue_type, dedup_key) DO NOTHING""",
                    (
                        queue_id,
                        key,
                        paper_id,
                        run_id,
                        _json({"stage": stage, "output_kind": output_kind, "existing_hash": conflict.existing_hash,
                               "incoming_hash": conflict.incoming_hash}),
                    ),
                )
                queue_ids.append(queue_id)
        return tuple(queue_ids)
