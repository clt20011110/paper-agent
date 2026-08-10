from __future__ import annotations

import json

import pytest

from paper_agent.approval import approved_content_hash
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


def test_unattended_mode_requires_an_explicit_frozen_allowance(grants: GrantStore) -> None:
    with pytest.raises(GrantError, match="allow_unattended"):
        grants.create_draft(
            kind="download",
            actions=["download", "store"],
            purpose="personal_research",
            mode="unattended",
            scope=scope(provider=None, model=None, artifact_hashes=[]),
            max_papers=1,
            expires_at=FUTURE,
        )

    draft = grants.create_draft(
        grant_id="grant-unattended",
        kind="download",
        actions=["download", "store"],
        purpose="personal_research",
        mode="unattended",
        allow_unattended=True,
        scope=scope(provider=None, model=None, artifact_hashes=[]),
        max_papers=1,
        expires_at=FUTURE,
    )
    approved_document = grants.approve(
        draft, draft["content_hash"], approved_by="owner", approved_at=NOW
    )

    assert approved_document["allow_unattended"] is True
    assert grants.load("grant-unattended").document["allow_unattended"] is True


def test_yaml_defaults_cannot_expand_approved_scope(grants: GrantStore) -> None:
    approved(grants)
    yaml_defaults = {"paper_ids": ["paper-1", "paper-2"], "max_papers": 999}

    with pytest.raises(GrantError, match="paper"):
        grants.require_active(
            "grant-1", kind="remote_model_processing", action="remote_model_processing",
            purpose="internal_analysis", mode="attended", now=NOW,
            paper_id=yaml_defaults["paper_ids"][1], paper_count=yaml_defaults["max_papers"],
        )


@pytest.mark.parametrize(
    ("scope_changes", "max_papers", "message"),
    [
        ({"paper_ids": ["paper-1", "paper-1"]}, 2, "unique"),
        ({"paper_ids": ["paper-1", "paper-2"]}, 1, "max_papers"),
        ({"paper_ids": []}, 1, "exact paper IDs"),
    ],
)
def test_create_draft_enforces_exact_remote_paper_scope(
    grants: GrantStore,
    scope_changes: dict[str, object],
    max_papers: int,
    message: str,
) -> None:
    with pytest.raises(GrantError, match=message):
        grants.create_draft(
            kind="remote_model_processing",
            actions=["remote_model_processing"],
            purpose="internal_analysis",
            mode="attended",
            scope=scope(**scope_changes),
            max_papers=max_papers,
            expires_at=FUTURE,
        )


@pytest.mark.parametrize("max_papers", [0, True])
def test_create_draft_rejects_invalid_max_papers(
    grants: GrantStore, max_papers: object
) -> None:
    with pytest.raises(GrantError, match="positive integer"):
        grants.create_draft(
            kind="download",
            actions=["download"],
            purpose="personal_research",
            mode="attended",
            scope=scope(provider=None, model=None, artifact_hashes=[]),
            max_papers=max_papers,  # type: ignore[arg-type]
            expires_at=FUTURE,
        )


def test_approve_rechecks_semantics_of_hand_written_draft(grants: GrantStore) -> None:
    draft = grants.create_draft(
        kind="remote_model_processing",
        actions=["remote_model_processing"],
        purpose="internal_analysis",
        mode="attended",
        scope=scope(),
        max_papers=1,
        expires_at=FUTURE,
    )
    draft["scope"]["paper_ids"] = ["paper-1", "paper-2"]
    draft["content_hash"] = approved_content_hash(draft)

    with pytest.raises(GrantError, match="max_papers"):
        grants.approve(
            draft,
            draft["content_hash"],
            approved_by="owner",
            approved_at=NOW,
        )


def test_load_rechecks_semantics_before_a_drifted_grant_can_run(grants: GrantStore) -> None:
    approved(grants)
    drifted_scope = scope(paper_ids=["paper-1", "paper-1"])
    grants.database.connection.execute(
        "UPDATE authorization_grants SET scope_json = ? WHERE grant_id = 'grant-1'",
        (json.dumps(drifted_scope),),
    )

    with pytest.raises(GrantError, match="unique"):
        grants.load("grant-1")


def test_active_grant_checks_the_actual_requested_paper_set(grants: GrantStore) -> None:
    draft = grants.create_draft(
        grant_id="multi-paper-grant",
        kind="remote_model_processing",
        actions=["remote_model_processing"],
        purpose="research_synthesis",
        mode="attended",
        scope=scope(
            paper_ids=["paper-1", "paper-2"],
            artifact_hashes=[],
            collection_ids=[],
            collection_snapshot_hash=None,
            selection_snapshot_hash=None,
            domains=[],
            provider=None,
            model=None,
            data_categories=[],
        ),
        max_papers=2,
        expires_at=FUTURE,
    )
    grants.approve(draft, draft["content_hash"], approved_by="owner", approved_at=NOW)

    active = grants.require_active(
        "multi-paper-grant",
        kind="remote_model_processing",
        action="remote_model_processing",
        purpose="research_synthesis",
        mode="attended",
        now=NOW,
        paper_ids=("paper-2", "paper-1", "paper-1"),
    )
    assert active.grant_id == "multi-paper-grant"

    with pytest.raises(GrantError, match="paper"):
        grants.require_active(
            "multi-paper-grant",
            kind="remote_model_processing",
            action="remote_model_processing",
            purpose="research_synthesis",
            mode="attended",
            now=NOW,
            paper_ids=("paper-1", "paper-3"),
        )
    with pytest.raises(GrantError, match="max_papers"):
        grants.require_active(
            "multi-paper-grant",
            kind="remote_model_processing",
            action="remote_model_processing",
            purpose="research_synthesis",
            mode="attended",
            now=NOW,
            paper_ids=("paper-1", "paper-2", "paper-3"),
        )
