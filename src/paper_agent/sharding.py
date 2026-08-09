"""Deterministic, coordinator-owned multi-machine shard manifests."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import PurePath, PureWindowsPath
import re
from typing import Any, Iterable, Mapping

from paper_agent.canonical import content_hash


OutputKey = tuple[str, str, str, str]
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class MergeRejected(ValueError):
    """A worker result cannot be considered by the coordinator."""


def _relative_path(path: str) -> None:
    if not path or PurePath(path).is_absolute() or PureWindowsPath(path).is_absolute():
        raise MergeRejected("paths must be non-empty and relative")
    if ".." in PurePath(path).parts or ".." in PureWindowsPath(path).parts:
        raise MergeRejected("paths must not escape their output root")


def _sha256(value: str, field: str = "artifact_hash") -> None:
    if not _SHA256.fullmatch(value):
        raise MergeRejected(f"{field} must be a lowercase SHA-256 digest")


def _fencing_token(snapshot_hash: str, shard_id: str, epoch: int) -> str:
    return content_hash({"snapshot_hash": snapshot_hash, "shard_id": shard_id, "epoch": epoch})


@dataclass(frozen=True)
class SnapshotManifest:
    run_id: str
    paper_ids: tuple[str, ...]
    config_hash: str
    model_profile: Any
    model_profile_hash: str
    input_artifact_hash: str
    snapshot_hash: str


@dataclass(frozen=True)
class ShardManifest:
    run_id: str
    snapshot_hash: str
    config_hash: str
    model_profile: Any
    model_profile_hash: str
    input_artifact_hash: str
    shard_id: str
    epoch: int
    fencing_token: str
    output_root: str
    paper_ids: tuple[str, ...]


@dataclass(frozen=True)
class ArtifactManifest:
    paper_id: str
    output_kind: str
    artifact_hash: str
    mime_type: str
    size_bytes: int
    relative_path: str


@dataclass(frozen=True)
class ArtifactReceipt:
    """Metadata measured by the coordinator while receiving an artifact bundle."""

    artifact_hash: str
    mime_type: str
    size_bytes: int


@dataclass(frozen=True)
class WorkerResultManifest:
    run_id: str
    snapshot_hash: str
    shard_id: str
    epoch: int
    fencing_token: str
    stage: str
    paper_ids: tuple[str, ...]
    artifacts: tuple[ArtifactManifest, ...]


@dataclass(frozen=True)
class MergedArtifact:
    run_id: str
    stage: str
    paper_id: str
    output_kind: str
    artifact_hash: str
    mime_type: str
    size_bytes: int
    relative_path: str

    @property
    def key(self) -> OutputKey:
        return (self.run_id, self.stage, self.paper_id, self.output_kind)


@dataclass(frozen=True)
class MergeConflict:
    key: OutputKey
    existing_hash: str
    incoming_hash: str


@dataclass(frozen=True)
class MergePlan:
    new_outputs: tuple[MergedArtifact, ...]
    idempotent_keys: tuple[OutputKey, ...]
    conflicts: tuple[MergeConflict, ...]
    missing_coverage: tuple[OutputKey, ...]

    @property
    def complete(self) -> bool:
        return not self.conflicts and not self.missing_coverage


def freeze_snapshot(
    run_id: str,
    paper_ids: Iterable[str],
    config_hash: str,
    model_profile: Any,
    input_artifact_hash: str,
) -> SnapshotManifest:
    """Freeze sorted canonical paper IDs before dispatching any workers."""
    if not run_id:
        raise ValueError("run_id is required")
    ids = tuple(sorted(paper_ids))
    if not ids or any(not paper_id for paper_id in ids):
        raise ValueError("paper_ids must be non-empty")
    if len(ids) != len(set(ids)):
        raise ValueError("paper_ids must be unique")
    model_profile_hash = content_hash(model_profile)
    snapshot_hash = content_hash(
        {
            "run_id": run_id,
            "paper_ids": ids,
            "config_hash": config_hash,
            "model_profile": model_profile,
            "model_profile_hash": model_profile_hash,
            "input_artifact_hash": input_artifact_hash,
        }
    )
    return SnapshotManifest(
        run_id=run_id,
        paper_ids=ids,
        config_hash=config_hash,
        model_profile=model_profile,
        model_profile_hash=model_profile_hash,
        input_artifact_hash=input_artifact_hash,
        snapshot_hash=snapshot_hash,
    )


def build_shards(
    snapshot: SnapshotManifest,
    shard_count: int,
    output_root: str,
    *,
    epoch: int = 1,
) -> tuple[ShardManifest, ...]:
    """Split a frozen snapshot into contiguous, balanced, mutually exclusive shards."""
    _relative_path(output_root)
    if epoch < 1:
        raise ValueError("epoch must be positive")
    if shard_count < 1 or shard_count > len(snapshot.paper_ids):
        raise ValueError("shard_count must be between 1 and the paper count")

    base, extra = divmod(len(snapshot.paper_ids), shard_count)
    offset = 0
    shards = []
    for index in range(shard_count):
        size = base + (index < extra)
        paper_ids = snapshot.paper_ids[offset : offset + size]
        offset += size
        shard_id = f"shard-{index + 1:04d}-of-{shard_count:04d}"
        shards.append(
            ShardManifest(
                run_id=snapshot.run_id,
                snapshot_hash=snapshot.snapshot_hash,
                config_hash=snapshot.config_hash,
                model_profile=snapshot.model_profile,
                model_profile_hash=snapshot.model_profile_hash,
                input_artifact_hash=snapshot.input_artifact_hash,
                shard_id=shard_id,
                epoch=epoch,
                fencing_token=_fencing_token(snapshot.snapshot_hash, shard_id, epoch),
                output_root=output_root,
                paper_ids=paper_ids,
            )
        )
    return tuple(shards)


def bump_epoch(shard: ShardManifest) -> ShardManifest:
    """Create a fencing-safe manifest when a shard is precisely reassigned."""
    epoch = shard.epoch + 1
    return replace(shard, epoch=epoch, fencing_token=_fencing_token(shard.snapshot_hash, shard.shard_id, epoch))


def _validate_artifact(artifact: ArtifactManifest, receipt: ArtifactReceipt | None) -> None:
    _sha256(artifact.artifact_hash)
    if not artifact.mime_type or artifact.size_bytes < 0:
        raise MergeRejected("artifact MIME type and size are required")
    _relative_path(artifact.relative_path)
    if receipt is None:
        raise MergeRejected("artifact receipt is missing")
    _sha256(receipt.artifact_hash, "received artifact_hash")
    if (
        receipt.artifact_hash != artifact.artifact_hash
        or receipt.mime_type != artifact.mime_type
        or receipt.size_bytes != artifact.size_bytes
    ):
        raise MergeRejected("received artifact metadata does not match its manifest")


def _validate_result(result: WorkerResultManifest, shard: ShardManifest, stage: str) -> None:
    if (
        result.run_id != shard.run_id
        or result.snapshot_hash != shard.snapshot_hash
        or result.epoch != shard.epoch
        or result.fencing_token != shard.fencing_token
    ):
        raise MergeRejected("worker result has an old or foreign fencing token")
    if result.stage != stage:
        raise MergeRejected("worker result stage does not match merge stage")
    result_ids = tuple(sorted(result.paper_ids))
    if len(result_ids) != len(set(result_ids)) or result_ids != shard.paper_ids:
        raise MergeRejected("worker result paper_ids do not exactly match its shard")


def plan_merge(
    shards: Iterable[ShardManifest],
    results: Iterable[WorkerResultManifest],
    receipts: Mapping[str, ArtifactReceipt],
    *,
    stage: str,
    output_kinds: Iterable[str],
    coordinator_output_root: str = "artifacts",
    existing_outputs: Mapping[OutputKey, MergedArtifact] | None = None,
) -> MergePlan:
    """Return the coordinator's deterministic merge decision without performing I/O."""
    _relative_path(coordinator_output_root)
    kinds = tuple(sorted(set(output_kinds)))
    if not stage or not kinds or any(not kind for kind in kinds):
        raise ValueError("stage and output_kinds are required")

    shard_list = tuple(shards)
    current = {shard.shard_id: shard for shard in shard_list}
    if not current or len(current) != len(shard_list):
        raise ValueError("shard IDs must be unique and non-empty")
    expected = {
        (shard.run_id, stage, paper_id, kind)
        for shard in current.values()
        for paper_id in shard.paper_ids
        for kind in kinds
    }
    existing = existing_outputs or {}
    incoming: dict[OutputKey, list[ArtifactManifest]] = {}
    for result in sorted(results, key=lambda item: (item.shard_id, item.epoch, item.fencing_token)):
        shard = current.get(result.shard_id)
        if shard is None:
            raise MergeRejected("worker result shard is not current")
        _validate_result(result, shard, stage)
        for artifact in result.artifacts:
            key = (result.run_id, result.stage, artifact.paper_id, artifact.output_kind)
            if key not in expected:
                raise MergeRejected("worker artifact is outside its assigned coverage")
            _validate_artifact(artifact, receipts.get(artifact.artifact_hash))
            incoming.setdefault(key, []).append(artifact)

    idempotent: set[OutputKey] = set()
    conflicts: list[MergeConflict] = []
    proposed: list[MergedArtifact] = []
    for key in sorted(incoming):
        artifacts = incoming[key]
        hashes = sorted({artifact.artifact_hash for artifact in artifacts})
        saved = existing.get(key)
        if saved is not None:
            for artifact_hash in hashes:
                if artifact_hash == saved.artifact_hash:
                    idempotent.add(key)
                else:
                    conflicts.append(MergeConflict(key, saved.artifact_hash, artifact_hash))
            continue
        if len(hashes) > 1:
            for artifact_hash in hashes[1:]:
                conflicts.append(MergeConflict(key, hashes[0], artifact_hash))
            continue
        artifact = artifacts[0]
        proposed.append(
            MergedArtifact(
                run_id=key[0],
                stage=key[1],
                paper_id=key[2],
                output_kind=key[3],
                artifact_hash=artifact.artifact_hash,
                mime_type=artifact.mime_type,
                size_bytes=artifact.size_bytes,
                relative_path=f"{coordinator_output_root}/{artifact.artifact_hash}",
            )
        )
        if len(artifacts) > 1:
            idempotent.add(key)

    covered = set(existing) | {output.key for output in proposed} | idempotent
    missing = tuple(sorted(expected - covered))
    conflicts = sorted(set(conflicts), key=lambda item: (item.key, item.existing_hash, item.incoming_hash))
    if conflicts or missing:
        proposed = []
    return MergePlan(tuple(proposed), tuple(sorted(idempotent)), tuple(conflicts), missing)
