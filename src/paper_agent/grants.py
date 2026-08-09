"""Immutable, SQLite-backed authorization grants."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from typing import Any, Mapping
from uuid import uuid4

from .approval import ApprovalError, approve, approved_content_hash, require_valid_approval
from .schema import validate
from .storage import Database


class GrantError(ValueError):
    pass


_KINDS = {
    "download": "download",
    "browser_data_sharing": "browser_data_sharing",
    "remote_model_processing": "remote_model_processing",
}


@dataclass(frozen=True)
class ActiveGrant:
    document: dict[str, Any]
    kind: str
    dependency_digest: str | None
    lineage_hash: str | None

    @property
    def grant_id(self) -> str:
        return str(self.document["grant_id"])

    @property
    def content_hash(self) -> str:
        return str(self.document["content_hash"])


class GrantStore:
    """Creates drafts and keeps approved grant content immutable in SQLite."""

    def __init__(self, database: Database) -> None:
        self.database = database

    def create_draft(
        self,
        *,
        kind: str,
        actions: list[str],
        purpose: str,
        mode: str,
        scope: Mapping[str, Any],
        max_papers: int,
        expires_at: str,
        skill_digest: str | None = None,
        dependency_digest: str | None = None,
        lineage_hash: str | None = None,
        grant_id: str | None = None,
    ) -> dict[str, Any]:
        if kind not in _KINDS:
            raise GrantError(f"unknown grant kind: {kind}")
        if _KINDS[kind] not in actions:
            raise GrantError(f"{kind} grants require the {kind} action")
        if not _has_selection_scope(scope):
            raise GrantError("grant scope requires a paper, collection, selection snapshot, or artifact")

        draft: dict[str, Any] = {
            "schema_version": "1",
            "grant_id": grant_id or str(uuid4()),
            "content_hash": "0" * 64,
            "status": "draft",
            "kind": kind,
            "actions": list(actions),
            "purpose": purpose,
            "mode": mode,
            "scope": deepcopy(dict(scope)),
            "max_papers": max_papers,
            "expires_at": expires_at,
            "skill_digest": skill_digest,
            "dependency_digest": dependency_digest,
            "lineage_hash": lineage_hash,
            "approval": None,
        }
        draft["content_hash"] = approved_content_hash(draft)
        validate(draft, "authorization-grant.schema.json")
        return draft

    def approve(
        self,
        draft: Mapping[str, Any],
        expected_hash: str,
        *,
        approved_by: str,
        approved_at: str,
    ) -> dict[str, Any]:
        approved = approve(
            draft,
            expected_hash,
            approved_by=approved_by,
            approved_at=approved_at,
            hash_field="content_hash",
        )
        validate(approved, "authorization-grant.schema.json")
        grant_kind = approved["kind"]
        if grant_kind not in _KINDS:
            raise GrantError(f"unknown grant kind: {grant_kind}")
        if _KINDS[grant_kind] not in approved["actions"]:
            raise GrantError(f"{grant_kind} grants require the {grant_kind} action")

        scope = approved["scope"]
        assert isinstance(scope, dict)
        with self.database.transaction() as connection:
            connection.execute(
                """
                INSERT INTO authorization_grants(
                    grant_id, content_hash, approval_json, grant_kind, actions_json, purpose,
                    mode, scope_json, selection_snapshot_hash, max_papers, artifact_hash,
                    lineage_hash, provider, model_id, skill_digest, dependency_digest, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    approved["grant_id"],
                    approved["content_hash"],
                    _json(approved["approval"]),
                    grant_kind,
                    _json(approved["actions"]),
                    approved["purpose"],
                    approved["mode"],
                    _json(scope),
                    scope["selection_snapshot_hash"],
                    approved["max_papers"],
                    _single_artifact_hash(scope),
                    approved["lineage_hash"],
                    scope["provider"],
                    scope["model"],
                    approved["skill_digest"],
                    approved["dependency_digest"],
                    approved["expires_at"],
                ),
            )
            connection.execute(
                """
                INSERT INTO authorization_grant_events(grant_event_id, grant_id, event_type, actor, event_at, event_json)
                VALUES (?, ?, 'approved', ?, ?, ?)
                """,
                (
                    str(uuid4()),
                    approved["grant_id"],
                    approved_by,
                    approved_at,
                    _json({"content_hash": approved["content_hash"]}),
                ),
            )
        return approved

    def revoke(self, grant_id: str, *, actor: str, event_at: str) -> None:
        if self._row(grant_id) is None:
            raise GrantError(f"grant not found: {grant_id}")
        with self.database.transaction() as connection:
            existing = connection.execute(
                "SELECT 1 FROM authorization_grant_events WHERE grant_id = ? AND event_type = 'revoked'",
                (grant_id,),
            ).fetchone()
            if existing is not None:
                raise GrantError(f"grant is already revoked: {grant_id}")
            connection.execute(
                """
                INSERT INTO authorization_grant_events(grant_event_id, grant_id, event_type, actor, event_at, event_json)
                VALUES (?, ?, 'revoked', ?, ?, ?)
                """,
                (str(uuid4()), grant_id, actor, event_at, _json({"reason": "revoked_by_user"})),
            )

    def load(
        self,
        grant_id: str,
        *,
        kind: str | None = None,
        now: datetime | str | None = None,
    ) -> ActiveGrant:
        row = self._row(grant_id)
        if row is None:
            raise GrantError(f"grant not found: {grant_id}")
        if kind is not None and row["grant_kind"] != kind:
            raise GrantError(f"grant kind does not match: expected {kind}")

        document = {
            "schema_version": "1",
            "grant_id": row["grant_id"],
            "content_hash": row["content_hash"],
            "status": "approved",
            "kind": row["grant_kind"],
            "actions": json.loads(row["actions_json"]),
            "purpose": row["purpose"],
            "mode": row["mode"],
            "scope": json.loads(row["scope_json"]),
            "max_papers": row["max_papers"],
            "expires_at": row["expires_at"],
            "skill_digest": row["skill_digest"],
            "dependency_digest": row["dependency_digest"],
            "lineage_hash": row["lineage_hash"],
            "approval": json.loads(row["approval_json"]),
        }
        validate(document, "authorization-grant.schema.json")
        try:
            require_valid_approval(document, "content_hash")
        except ApprovalError as error:
            raise GrantError(str(error)) from error
        grant = ActiveGrant(
            document=document,
            kind=row["grant_kind"],
            dependency_digest=row["dependency_digest"],
            lineage_hash=row["lineage_hash"],
        )
        if now is not None:
            if self._is_revoked(grant_id):
                raise GrantError(f"grant is revoked: {grant_id}")
            if _as_datetime(now) >= _as_datetime(str(document["expires_at"])):
                raise GrantError(f"grant is expired: {grant_id}")
        return grant

    def require_active(
        self,
        grant_id: str,
        *,
        kind: str,
        action: str,
        purpose: str,
        mode: str,
        now: datetime | str,
        paper_id: str | None = None,
        artifact_hash: str | None = None,
        collection_id: str | None = None,
        collection_snapshot_hash: str | None = None,
        selection_snapshot_hash: str | None = None,
        domain: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        data_category: str | None = None,
        skill_digest: str | None = None,
        dependency_digest: str | None = None,
        lineage_hash: str | None = None,
        paper_count: int = 1,
    ) -> ActiveGrant:
        grant = self.load(grant_id, kind=kind, now=now)
        document = grant.document
        scope = document["scope"]
        assert isinstance(scope, dict)
        if action not in document["actions"]:
            raise GrantError(f"grant does not allow action: {action}")
        if purpose != document["purpose"]:
            raise GrantError("grant purpose does not match")
        if mode != document["mode"]:
            raise GrantError("grant mode does not match")
        if paper_count < 1 or paper_count > document["max_papers"]:
            raise GrantError("grant max_papers does not cover this request")
        _require_scope_item(scope["paper_ids"], paper_id, "paper")
        _require_scope_item(scope["artifact_hashes"], artifact_hash, "artifact")
        _require_scope_item(scope["collection_ids"], collection_id, "collection")
        _require_scope_value(scope["collection_snapshot_hash"], collection_snapshot_hash, "collection snapshot")
        _require_scope_value(scope["selection_snapshot_hash"], selection_snapshot_hash, "selection snapshot")
        _require_scope_item(scope["domains"], domain, "domain")
        _require_scope_value(scope["provider"], provider, "provider")
        _require_scope_value(scope["model"], model, "model")
        _require_scope_item(scope["data_categories"], data_category, "data category")
        _require_scope_value(document["skill_digest"], skill_digest, "skill digest")
        _require_scope_value(grant.dependency_digest, dependency_digest, "dependency digest")
        _require_scope_value(grant.lineage_hash, lineage_hash, "lineage hash")
        return grant

    def _row(self, grant_id: str):
        return self.database.connection.execute(
            "SELECT * FROM authorization_grants WHERE grant_id = ?", (grant_id,)
        ).fetchone()

    def _is_revoked(self, grant_id: str) -> bool:
        return self.database.connection.execute(
            "SELECT 1 FROM authorization_grant_events WHERE grant_id = ? AND event_type = 'revoked'",
            (grant_id,),
        ).fetchone() is not None


def _has_selection_scope(scope: Mapping[str, Any]) -> bool:
    return bool(
        scope.get("paper_ids")
        or scope.get("collection_ids")
        or scope.get("collection_snapshot_hash")
        or scope.get("selection_snapshot_hash")
        or scope.get("artifact_hashes")
    )


def _single_artifact_hash(scope: Mapping[str, Any]) -> str | None:
    artifacts = scope["artifact_hashes"]
    return artifacts[0] if len(artifacts) == 1 else None


def _require_scope_item(allowed: list[str], requested: str | None, label: str) -> None:
    if (allowed and requested not in allowed) or (not allowed and requested is not None):
        raise GrantError(f"grant does not cover {label}: {requested}")


def _require_scope_value(allowed: str | None, requested: str | None, label: str) -> None:
    if allowed is not None and allowed != requested:
        raise GrantError(f"grant does not cover {label}")
    if allowed is None and requested is not None:
        raise GrantError(f"grant does not cover {label}")


def _as_datetime(value: datetime | str) -> datetime:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc)
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))
