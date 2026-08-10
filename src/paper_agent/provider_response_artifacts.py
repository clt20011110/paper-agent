"""Durable content-addressed replay for Stage 1 provider responses."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from sqlite3 import Connection
from typing import Any

from .artifacts import ArtifactMetadataConflict, ArtifactStore, StoredArtifact
from .domain import SourceBatch
from .storage import Database


_RAW_RESPONSE_MIME_TYPE = "application/octet-stream"


@dataclass(frozen=True, slots=True)
class ProviderResponseKey:
    replay_scope: str
    provider: str
    query_hash: str
    cursor: str | None
    api_version: str

    @property
    def cursor_json(self) -> str:
        return json.dumps(self.cursor, ensure_ascii=False, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class ReplayedProviderResponse:
    body: bytes
    artifact_hash: str
    content_type: str
    etag: str | None
    last_modified: str | None


class ProviderResponseArtifactService:
    """Write response bytes to CAS and load indexed bytes without provider I/O.

    Worker threads only write immutable files and open short read-only SQLite
    connections. The search coordinator remains the sole SQLite writer.
    """

    def __init__(self, database_path: str | Path, artifact_store: ArtifactStore) -> None:
        self.database_path = Path(database_path)
        self.artifact_store = artifact_store

    def capture(self, body: bytes) -> StoredArtifact:
        return self.artifact_store.put_bytes(body, mime_type=_RAW_RESPONSE_MIME_TYPE)

    def replay(self, key: ProviderResponseKey) -> ReplayedProviderResponse | None:
        with Database(self.database_path, read_only=True) as database:
            row = database.connection.execute(
                """SELECT a.sha256, c.content_type, c.etag, c.last_modified
                   FROM provider_response_cache c
                   JOIN artifacts a ON a.artifact_id = c.artifact_id
                   WHERE c.replay_scope = ? AND c.provider = ? AND c.query_hash = ?
                     AND c.cursor_json = ? AND c.api_version = ?""",
                (
                    key.replay_scope,
                    key.provider,
                    key.query_hash,
                    key.cursor_json,
                    key.api_version,
                ),
            ).fetchone()
        if row is None:
            return None
        artifact_hash = str(row["sha256"])
        return ReplayedProviderResponse(
            body=self.artifact_store.read_bytes(artifact_hash),
            artifact_hash=artifact_hash,
            content_type=str(row["content_type"]),
            etag=_optional_text(row["etag"]),
            last_modified=_optional_text(row["last_modified"]),
        )


class ProviderResponseArtifactRepository:
    """Coordinator-owned SQLite mappings from requests and runs to CAS bytes."""

    def __init__(self, database: Database, artifact_store: ArtifactStore | None = None) -> None:
        self.database = database
        self.artifact_store = artifact_store or ArtifactStore(database.path.parent)

    def record_batch(
        self,
        connection: Connection,
        batch: SourceBatch,
        *,
        replay_scope: str,
        recorded_at: str,
    ) -> str | None:
        records = tuple(
            record
            for record in batch.request_audit
            if record.get("response_artifact_hash") is not None
        )
        if not records:
            return None

        aggregate_artifact_id = None
        if batch.raw_response_artifact_hash is not None:
            aggregate_artifact_id = self._register_artifact(
                connection, batch.raw_response_artifact_hash
            )

        for record in records:
            if record.get("replay_scope") != replay_scope:
                raise ValueError("provider response replay scope does not match crawl run")
            artifact_hash = str(record["response_artifact_hash"])
            artifact_id = self._register_artifact(connection, artifact_hash)
            key = ProviderResponseKey(
                replay_scope=str(record["replay_scope"]),
                provider=str(record["provider"]),
                query_hash=str(record["query_hash"]),
                cursor=_optional_text(record.get("cursor")),
                api_version=str(record["api_version"]),
            )
            values = (
                key.replay_scope,
                key.provider,
                key.query_hash,
                key.cursor_json,
                key.api_version,
                artifact_id,
                str(record.get("content_type") or ""),
                _optional_text(record.get("etag")),
                _optional_text(record.get("last_modified")),
                recorded_at,
            )
            connection.execute(
                """INSERT INTO provider_response_cache(
                       replay_scope, provider, query_hash, cursor_json, api_version, artifact_id,
                       content_type, etag, last_modified, recorded_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(
                       replay_scope, provider, query_hash, cursor_json, api_version
                   ) DO NOTHING""",
                values,
            )
            existing = connection.execute(
                """SELECT artifact_id, content_type, etag, last_modified
                   FROM provider_response_cache
                   WHERE replay_scope = ? AND provider = ? AND query_hash = ?
                     AND cursor_json = ? AND api_version = ?""",
                (
                    key.replay_scope,
                    key.provider,
                    key.query_hash,
                    key.cursor_json,
                    key.api_version,
                ),
            ).fetchone()
            expected = (artifact_id, values[6], values[7], values[8])
            if existing is None or tuple(existing) != expected:
                raise ValueError("provider response cache key has different immutable response metadata")
        return aggregate_artifact_id

    def _register_artifact(self, connection: Connection, artifact_hash: str) -> str:
        payload = self.artifact_store.read_bytes(artifact_hash)
        relative_path = self.artifact_store.relative_path(artifact_hash)
        existing = connection.execute(
            """SELECT artifact_id, byte_size, relative_path
               FROM artifacts WHERE sha256 = ?""",
            (artifact_hash,),
        ).fetchone()
        if existing is not None:
            if existing["byte_size"] != len(payload) or existing["relative_path"] != relative_path:
                raise ArtifactMetadataConflict(
                    "artifact database metadata conflicts with content digest"
                )
            return str(existing["artifact_id"])

        artifact_id = f"artifact-{artifact_hash}"
        connection.execute(
            """INSERT INTO artifacts(
                   artifact_id, artifact_kind, relative_path, mime_type,
                   byte_size, sha256, provenance_json
               ) VALUES (?, 'other', ?, ?, ?, ?, ?)""",
            (
                artifact_id,
                relative_path,
                _RAW_RESPONSE_MIME_TYPE,
                len(payload),
                artifact_hash,
                _json({"kind": "provider_raw_response"}),
            ),
        )
        return artifact_id


def _optional_text(value: Any) -> str | None:
    return None if value is None else str(value)


def _json(value: Mapping[str, object]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
