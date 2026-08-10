import json

import pytest

from paper_agent.query_plan import (
    QueryPlanDriftError,
    QueryPlanError,
    QueryPlanStore,
    approve_query_plan,
    assert_runtime_matches,
    compile_query_plan,
)
from paper_agent.schema import validate


def _digest(character: str) -> str:
    return character * 64


def provider(name: str = "openalex", *, enabled: bool = True) -> dict[str, object]:
    return {
        "provider": name,
        "distribution": f"paper-agent-{name}",
        "version": "1.0.0",
        "artifact_sha256": _digest("a"),
        "manifest_hash": _digest("b"),
        "roles": ["search"],
        "capabilities": ["stable_id", "metadata", "date_filter"],
        "enabled": enabled,
        "mode": "api",
        "credentials_present": True,
    }


def draft() -> dict[str, object]:
    return {
        "created_at": "2026-08-09T00:00:00Z",
        "research": {
            "objective": "map graph learning",
            "audience": "researchers",
            "primary_question": "What methods work?",
            "subquestions": [{"id": "sq1", "question": "Which methods?"}],
        },
        "scope": {
            "date_from": "2020-01-01",
            "date_to": "2024-12-31",
            "venues": [],
            "fields": ["computer science"],
            "languages": ["en"],
            "document_types": ["article"],
            "user_seeds": [],
        },
        "inclusion": {"criteria": ["empirical"], "exclusion_criteria": ["unrelated"]},
        "query_variants": [
            {"id": "q1", "subquestion_id": "sq1", "alias_group": "graph", "raw_query": "graph learning", "synonyms": ["GNN"]}
        ],
        "filter": {
            "profile": "fake",
            "config_hash": _digest("c"),
            "thresholds_hash": _digest("d"),
            "seed_selector_version": "1",
            "seed_selector_config_hash": _digest("e"),
            "round_state_machine_version": "1",
        },
        "citation_snowball": {
            "enabled": True,
            "directions": ["references", "citations"],
            "max_depth": 2,
            "max_rounds": 3,
            "max_per_seed_per_source": 20,
        },
        "budgets": {
            "max_requests": 100,
            "max_candidates": 1000,
            "max_seconds": 300,
            "saturation": {"min_unique_included_yield": 0.05, "consecutive_low_yield_rounds": 2},
        },
        "provider_policy": "all_resolved",
        "required_roles": ["search"],
        "required_providers": ["openalex"],
    }


def test_compiled_plan_replays_and_approval_is_detached() -> None:
    first = compile_query_plan(draft(), providers=[provider()])
    second = compile_query_plan(draft(), providers=[provider()])

    assert first["plan_hash"] == second["plan_hash"]
    assert first["plan_id"] == second["plan_id"]
    approved = approve_query_plan(
        first,
        first["plan_hash"],
        approved_by="owner",
        approved_at="2026-08-09T01:00:00Z",
    )
    assert approved["plan_hash"] == first["plan_hash"]
    validate(approved, "query-plan.schema.json")


def test_approved_plan_is_immutable_and_latest_is_atomic(tmp_path) -> None:
    plan = compile_query_plan(draft(), providers=[provider()])
    store = QueryPlanStore(tmp_path)
    store.save_draft(plan)
    approved = store.approve_and_save(
        plan, plan["plan_hash"], approved_by="owner", approved_at="2026-08-09T01:00:00Z"
    )

    assert store.approved_path(approved["plan_id"]).exists()
    assert json.loads(store.latest_path.read_text()) == {
        "plan_id": approved["plan_id"],
        "plan_hash": approved["plan_hash"],
    }
    changed = dict(approved)
    changed["research"] = dict(changed["research"], objective="other")
    with pytest.raises(QueryPlanError, match="drifted"):
        store.save_approved(changed)


def test_required_provider_and_role_cannot_be_silently_unavailable() -> None:
    with pytest.raises(QueryPlanError, match="explicit required"):
        compile_query_plan(draft(), providers=[provider(enabled=False)])


def test_exact_primary_and_domain_auto_resolution_are_frozen() -> None:
    document = draft()
    document["required_providers"] = []
    exact = provider("openreview")
    exact["roles"] = ["venue_primary", "search"]
    exact["exact_required"] = True
    conditional = provider("pubmed")
    conditional["enabled"] = "auto_for_biomed"
    plan = compile_query_plan(document, providers=[exact, conditional])

    assert plan["execution"]["required_providers"] == ["openreview"]
    assert {item["provider"]: item["resolved"] for item in plan["providers"]} == {
        "openreview": True,
        "pubmed": False,
    }


def test_uninstalled_optional_plugin_keeps_an_honest_null_digest() -> None:
    optional = provider("exa", enabled=False)
    optional["artifact_sha256"] = None
    optional["manifest_trusted"] = False
    plan = compile_query_plan(draft(), providers=[provider(), optional])

    assert plan["providers"][1]["artifact_sha256"] is None
    assert plan["providers"][1]["resolved"] is False
    validate(plan, "query-plan.schema.json")


def test_runtime_drift_is_rejected_for_frozen_provider_and_budget() -> None:
    plan = compile_query_plan(draft(), providers=[provider()])
    approved = approve_query_plan(plan, plan["plan_hash"], approved_by="owner", approved_at="2026-08-09T01:00:00Z")
    runtime = approved["providers"]
    assert_runtime_matches(approved, runtime, budgets=approved["budgets"])

    changed = dict(runtime[0], version="2.0.0")
    with pytest.raises(QueryPlanDriftError, match="version"):
        assert_runtime_matches(approved, [changed])
    with pytest.raises(QueryPlanDriftError, match="budgets"):
        assert_runtime_matches(approved, runtime, budgets={"max_requests": 1})

    changed_policy = dict(runtime[0], rate_limit={**runtime[0]["rate_limit"], "global_qps": 2})
    with pytest.raises(QueryPlanDriftError, match="rate_limit"):
        assert_runtime_matches(approved, [changed_policy])
