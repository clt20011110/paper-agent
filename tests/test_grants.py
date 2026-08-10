from __future__ import annotations

import json

import pytest

from paper_agent.approval import approved_content_hash
from paper_agent.grants import (
    GrantError,
    GrantStore,
    create_grant_draft,
    validate_grant_approval,
    validate_grant_revocation,
)
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


def test_pure_grant_validation_apis_need_no_database() -> None:
    draft = create_grant_draft(
        grant_id="pure-grant",
        kind="remote_model_processing",
        actions=["remote_model_processing"],
        purpose="internal_analysis",
        mode="attended",
        scope=scope(
            collection_ids=[],
            collection_snapshot_hash=None,
            selection_snapshot_hash=None,
            domains=[],
        ),
        max_papers=1,
        expires_at=FUTURE,
    )

    approved_document = validate_grant_approval(
        draft,
        draft["content_hash"],
        approved_by="owner",
        approved_at=NOW,
    )
    event = validate_grant_revocation(
        approved_document,
        actor="owner",
        event_at=NOW,
    )

    assert approved_document["status"] == "approved"
    assert event == {
        "actor": "owner",
        "event_at": NOW,
        "event_json": {"reason": "revoked_by_user"},
    }


def test_approved_grant_is_schema_valid_and_immutable(grants: GrantStore) -> None:
    document = approved(grants)
    loaded = grants.load("grant-1", kind="remote_model_processing")

    assert loaded.document == document
    events = grants.database.connection.execute(
        "SELECT event_type, actor FROM authorization_grant_events WHERE grant_id = 'grant-1'"
    ).fetchall()
    assert [tuple(event) for event in events] == [("approved", "owner")]


def test_identical_approval_retry_is_idempotent_but_changed_approval_conflicts(
    grants: GrantStore,
) -> None:
    draft = grants.create_draft(
        grant_id="approval-retry",
        kind="remote_model_processing",
        actions=["remote_model_processing"],
        purpose="internal_analysis",
        mode="attended",
        scope=scope(
            collection_ids=[],
            collection_snapshot_hash=None,
            selection_snapshot_hash=None,
            domains=[],
        ),
        max_papers=1,
        expires_at=FUTURE,
    )
    first = grants.approve(
        draft, draft["content_hash"], approved_by="owner", approved_at=NOW
    )
    second = grants.approve(
        draft, draft["content_hash"], approved_by="owner", approved_at=NOW
    )

    assert second == first
    assert grants.database.connection.execute(
        "SELECT COUNT(*) FROM authorization_grant_events "
        "WHERE grant_id = 'approval-retry' AND event_type = 'approved'"
    ).fetchone()[0] == 1
    with pytest.raises(GrantError, match="conflicts with existing grant"):
        grants.approve(
            draft,
            draft["content_hash"],
            approved_by="different-owner",
            approved_at=NOW,
        )


def test_identical_revocation_retry_is_idempotent_but_changed_event_conflicts(
    grants: GrantStore,
) -> None:
    approved(grants)
    grants.revoke("grant-1", actor="owner", event_at=NOW)
    grants.revoke("grant-1", actor="owner", event_at=NOW)

    preview = grants.validate_revoke("grant-1", actor="owner", event_at=NOW)
    assert preview["already_applied"] is True
    assert grants.database.connection.execute(
        "SELECT COUNT(*) FROM authorization_grant_events "
        "WHERE grant_id = 'grant-1' AND event_type = 'revoked'"
    ).fetchone()[0] == 1
    with pytest.raises(GrantError, match="conflicts with existing event"):
        grants.revoke("grant-1", actor="different-owner", event_at=NOW)


@pytest.mark.parametrize(
    ("approved_by", "approved_at", "message"),
    [
        ("", NOW, "approved_by"),
        (" owner ", NOW, "approved_by"),
        ("owner", "not-a-time", "approved_at"),
        ("owner", "2026-08-09T20:00:00+08:00", "approved_at"),
    ],
)
def test_approval_requires_actor_and_rfc3339_utc(
    grants: GrantStore, approved_by: str, approved_at: str, message: str
) -> None:
    draft = grants.create_draft(
        kind="remote_model_processing",
        actions=["remote_model_processing"],
        purpose="internal_analysis",
        mode="attended",
        scope=scope(
            collection_ids=[],
            collection_snapshot_hash=None,
            selection_snapshot_hash=None,
            domains=[],
        ),
        max_papers=1,
        expires_at=FUTURE,
    )

    with pytest.raises(GrantError, match=message):
        validate_grant_approval(
            draft,
            draft["content_hash"],
            approved_by=approved_by,
            approved_at=approved_at,
        )


def test_approval_must_precede_expiry_and_revocation_cannot_precede_approval(
    grants: GrantStore,
) -> None:
    draft = grants.create_draft(
        kind="remote_model_processing",
        actions=["remote_model_processing"],
        purpose="internal_analysis",
        mode="attended",
        scope=scope(
            collection_ids=[],
            collection_snapshot_hash=None,
            selection_snapshot_hash=None,
            domains=[],
        ),
        max_papers=1,
        expires_at=FUTURE,
    )
    with pytest.raises(GrantError, match="expires_at must be after approved_at"):
        validate_grant_approval(
            draft,
            draft["content_hash"],
            approved_by="owner",
            approved_at=FUTURE,
        )

    approved_document = validate_grant_approval(
        draft,
        draft["content_hash"],
        approved_by="owner",
        approved_at=NOW,
    )
    with pytest.raises(GrantError, match="cannot precede"):
        validate_grant_revocation(
            approved_document,
            actor="owner",
            event_at="2026-08-09T11:59:59Z",
        )


@pytest.mark.parametrize(
    ("actor", "event_at", "message"),
    [
        ("", NOW, "actor"),
        (" owner ", NOW, "actor"),
        ("owner", "not-a-time", "event_at"),
        ("owner", "2026-08-09T20:00:00+08:00", "event_at"),
    ],
)
def test_revocation_requires_actor_and_rfc3339_utc(
    grants: GrantStore, actor: str, event_at: str, message: str
) -> None:
    approved(grants)

    with pytest.raises(GrantError, match=message):
        grants.validate_revoke("grant-1", actor=actor, event_at=event_at)


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
        skill_digest=HASH,
        dependency_digest=OTHER_HASH,
    )
    approved_document = grants.approve(
        draft, draft["content_hash"], approved_by="owner", approved_at=NOW
    )

    assert approved_document["allow_unattended"] is True
    assert grants.load("grant-unattended").document["allow_unattended"] is True


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"actions": ["download"]}, "download and store"),
        ({"scope_changes": {"domains": []}}, "domain scope"),
        (
            {
                "scope_changes": {"provider": None},
                "skill_digest": None,
                "dependency_digest": None,
            },
            "skill and dependency digests",
        ),
    ],
)
def test_download_draft_requires_an_operationally_exact_scope(
    changes: dict[str, object], message: str
) -> None:
    changes = dict(changes)
    scope_changes = changes.pop("scope_changes", {})
    scope_values: dict[str, object] = {
        "artifact_hashes": [],
        "provider": "public_http",
        "model": None,
    }
    scope_values.update(scope_changes)  # type: ignore[arg-type]
    values: dict[str, object] = {
        "kind": "download",
        "actions": ["download", "store"],
        "purpose": "personal_research",
        "mode": "attended",
        "scope": scope(**scope_values),
        "max_papers": 1,
        "expires_at": FUTURE,
    }
    values.update(changes)

    with pytest.raises(GrantError, match=message):
        create_grant_draft(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("scope_changes", "lineage_hash", "message"),
    [
        ({"artifact_hashes": []}, None, "artifact hashes"),
        ({"provider": None}, None, "provider codex_cli"),
        ({"model": None}, None, "frozen Luna or Sol"),
        ({"data_categories": []}, None, "data categories"),
        (
            {"model": "gpt-5.6-sol", "data_categories": ["analysis"]},
            None,
            "lineage hash",
        ),
    ],
)
def test_remote_draft_requires_exact_artifact_target_and_lineage(
    scope_changes: dict[str, object], lineage_hash: str | None, message: str
) -> None:
    scope_values: dict[str, object] = {
        "collection_ids": [],
        "collection_snapshot_hash": None,
        "selection_snapshot_hash": None,
        "domains": [],
    }
    scope_values.update(scope_changes)
    with pytest.raises(GrantError, match=message):
        create_grant_draft(
            kind="remote_model_processing",
            actions=["remote_model_processing"],
            purpose="internal_analysis",
            mode="attended",
            scope=scope(**scope_values),
            max_papers=1,
            expires_at=FUTURE,
            lineage_hash=lineage_hash,
        )


def test_browser_data_sharing_binds_target_category_and_skill_digests() -> None:
    draft = create_grant_draft(
        kind="browser_data_sharing",
        actions=["browser_data_sharing"],
        purpose="personal_research",
        mode="attended",
        scope=scope(
            artifact_hashes=[],
            provider="codex_cli",
            model="gpt-5.6-luna",
            data_categories=["full_text"],
        ),
        max_papers=1,
        expires_at=FUTURE,
        skill_digest=HASH,
        dependency_digest=OTHER_HASH,
    )

    assert draft["kind"] == "browser_data_sharing"


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
            artifact_hashes=[HASH],
            collection_ids=[],
            collection_snapshot_hash=None,
            selection_snapshot_hash=None,
            domains=[],
            provider="codex_cli",
            model="gpt-5.6-luna",
            data_categories=["full_text"],
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
        artifact_hash=HASH,
        provider="codex_cli",
        model="gpt-5.6-luna",
        data_category="full_text",
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
