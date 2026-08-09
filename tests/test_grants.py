from __future__ import annotations

import json

import pytest

from paper_agent.grants import GrantError, GrantStore
from paper_agent.storage import Database


NOW = "2026-08-09T12:00:00Z"
FUTURE = "2026-08-10T00:00:00Z"
HASH = "a" * 64
OTHER_HASH = "b" * 64


def scope(**changes: object) -> dict[str, object]:
    value: dict[str, object] = {
        "paper_ids": ["paper-1"],
        "artifact_hashes": [HASH],
        "collection_ids": ["collection-1"],
        "collection_snapshot_hash": HASH,
        "selection_snapshot_hash": OTHER_HASH,
        "domains": ["publisher.example"],
        "provider": "codex_cli",
        "model": "gpt-5.6-luna",
        "data_categories": ["full_text"],
    }
    value.update(changes)
    return value


@pytest.fixture
def grants(tmp_path) -> GrantStore:
    database = Database(tmp_path / "papers.sqlite3")
    database.migrate()
    yield GrantStore(database)
    database.close()


def approved(grants: GrantStore, **changes: object) -> dict[str, object]:
    draft = grants.create_draft(
        grant_id="grant-1",
        kind="remote_model_processing",
        actions=["remote_model_processing"],
        purpose="internal_analysis",
        mode="attended",
        scope=scope(**changes),
        max_papers=2,
        expires_at=FUTURE,
        skill_digest=HASH,
        dependency_digest=OTHER_HASH,
        lineage_hash=HASH,
    )
    return grants.approve(
        draft,
        draft["content_hash"],
        approved_by="owner",
        approved_at=NOW,
    )


def test_approved_grant_is_schema_valid_and_immutable(grants: GrantStore) -> None:
    document = approved(grants)
    loaded = grants.load("grant-1", kind="remote_model_processing")

    assert loaded.document == document
    events = grants.database.connection.execute(
        "SELECT event_type, actor FROM authorization_grant_events WHERE grant_id = 'grant-1'"
    ).fetchall()
    assert [tuple(event) for event in events] == [("approved", "owner")]


def test_tampering_content_or_approval_is_rejected(grants: GrantStore) -> None:
    approved(grants)
    grants.database.connection.execute(
        "UPDATE authorization_grants SET scope_json = ? WHERE grant_id = 'grant-1'",
        (json.dumps(scope(paper_ids=["paper-2"])),),
    )

    with pytest.raises(GrantError, match="drifted"):
        grants.load("grant-1")


def test_expired_and_revoked_grants_cannot_be_used(grants: GrantStore) -> None:
    approved(grants)
    with pytest.raises(GrantError, match="expired"):
        grants.load("grant-1", now="2026-08-11T00:00:00Z")
    with pytest.raises(GrantError, match="expired"):
        grants.require_active(
            "grant-1", kind="remote_model_processing", action="remote_model_processing",
            purpose="internal_analysis", mode="attended", now="2026-08-11T00:00:00Z",
        )

    grants.revoke("grant-1", actor="owner", event_at=NOW)
    with pytest.raises(GrantError, match="revoked"):
        grants.load("grant-1", now=NOW)
    with pytest.raises(GrantError, match="revoked"):
        grants.require_active(
            "grant-1", kind="remote_model_processing", action="remote_model_processing",
            purpose="internal_analysis", mode="attended", now=NOW,
        )
    assert grants.database.connection.execute(
        "SELECT purpose FROM authorization_grants WHERE grant_id = 'grant-1'"
    ).fetchone()[0] == "internal_analysis"


def test_active_scope_and_digest_must_match(grants: GrantStore) -> None:
    approved(grants)
    active = grants.require_active(
        "grant-1", kind="remote_model_processing", action="remote_model_processing",
        purpose="internal_analysis", mode="attended", now=NOW, paper_id="paper-1",
        artifact_hash=HASH, collection_id="collection-1", collection_snapshot_hash=HASH,
        selection_snapshot_hash=OTHER_HASH, domain="publisher.example", provider="codex_cli",
        model="gpt-5.6-luna", data_category="full_text", skill_digest=HASH, dependency_digest=OTHER_HASH,
        lineage_hash=HASH, paper_count=2,
    )
    assert active.grant_id == "grant-1"
    with pytest.raises(GrantError, match="paper"):
        grants.require_active(
            "grant-1", kind="remote_model_processing", action="remote_model_processing",
            purpose="internal_analysis", mode="attended", now=NOW, paper_id="paper-2",
        )
    with pytest.raises(GrantError, match="dependency digest"):
        grants.require_active(
            "grant-1", kind="remote_model_processing", action="remote_model_processing",
            purpose="internal_analysis", mode="attended", now=NOW, paper_id="paper-1",
            artifact_hash=HASH, collection_id="collection-1", collection_snapshot_hash=HASH,
            selection_snapshot_hash=OTHER_HASH, domain="publisher.example", provider="codex_cli",
            model="gpt-5.6-luna", data_category="full_text", skill_digest=HASH,
            dependency_digest=HASH, lineage_hash=HASH,
        )
    with pytest.raises(GrantError, match="provider"):
        grants.require_active(
            "grant-1", kind="remote_model_processing", action="remote_model_processing",
            purpose="internal_analysis", mode="attended", now=NOW, paper_id="paper-1",
            artifact_hash=HASH, collection_id="collection-1", collection_snapshot_hash=HASH,
            selection_snapshot_hash=OTHER_HASH, domain="publisher.example", model="gpt-5.6-luna",
            skill_digest=HASH, dependency_digest=OTHER_HASH, lineage_hash=HASH,
        )
    with pytest.raises(GrantError, match="data category"):
        grants.require_active(
            "grant-1", kind="remote_model_processing", action="remote_model_processing",
            purpose="internal_analysis", mode="attended", now=NOW, paper_id="paper-1",
            artifact_hash=HASH, collection_id="collection-1", collection_snapshot_hash=HASH,
            selection_snapshot_hash=OTHER_HASH, domain="publisher.example", provider="codex_cli",
            model="gpt-5.6-luna", data_category="abstract", skill_digest=HASH,
            dependency_digest=OTHER_HASH, lineage_hash=HASH,
        )


def test_attended_grant_does_not_authorize_unattended_execution(grants: GrantStore) -> None:
    approved(grants)
    with pytest.raises(GrantError, match="mode"):
        grants.require_active(
            "grant-1", kind="remote_model_processing", action="remote_model_processing",
            purpose="internal_analysis", mode="unattended", now=NOW,
        )


def test_yaml_defaults_cannot_expand_approved_scope(grants: GrantStore) -> None:
    approved(grants)
    yaml_defaults = {"paper_ids": ["paper-1", "paper-2"], "max_papers": 999}

    with pytest.raises(GrantError, match="paper"):
        grants.require_active(
            "grant-1", kind="remote_model_processing", action="remote_model_processing",
            purpose="internal_analysis", mode="attended", now=NOW,
            paper_id=yaml_defaults["paper_ids"][1], paper_count=yaml_defaults["max_papers"],
        )
