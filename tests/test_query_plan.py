from copy import deepcopy
import json

import pytest

from paper_agent.approval import approved_content_hash
from paper_agent.query_plan import (
    QueryPlanDriftError,
    QueryPlanError,
    QueryPlanStore,
    approve_query_plan,
    assert_runtime_matches,
    compile_query_plan,
)
from paper_agent.schema import validate
from paper_agent.scope_filter import screening_scope_hash


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


def venue_spec(*, fallback_role: str = "search") -> dict[str, object]:
    return {
        "descriptor": {
            "schema_version": "1",
            "venue_id": "testconf",
            "name": "TestConf",
            "venue_type": "conference",
            "primary_provider": "pmlr",
            "provider_params": {"volume_id": "v1"},
        },
        "acceptance": {
            "schema_version": "2",
            "venue_id": "testconf",
            "primary_provider": "pmlr",
            "fallbacks": [{"provider": "openalex", "role": fallback_role}],
        },
    }


def test_compiled_plan_replays_and_approval_is_detached() -> None:
    first = compile_query_plan(draft(), providers=[provider()])
    second = compile_query_plan(draft(), providers=[provider()])

    assert first["plan_hash"] == second["plan_hash"]
    assert first["plan_id"] == second["plan_id"]
    assert first["scope"]["include_arxiv_candidates"] is False
    assert first["schema_version"] == "2"
    approved = approve_query_plan(
        first,
        first["plan_hash"],
        approved_by="owner",
        approved_at="2026-08-09T01:00:00Z",
    )
    assert approved["plan_hash"] == first["plan_hash"]
    validate(approved, "query-plan.schema.json")


def test_legacy_v1_plan_requires_explicit_recompile_before_approval_or_runtime() -> None:
    plan = compile_query_plan(draft(), providers=[provider()])
    legacy = deepcopy(plan)
    legacy["schema_version"] = "1"
    legacy["plan_hash"] = approved_content_hash(legacy)

    with pytest.raises(QueryPlanError, match="recompile.*version 2"):
        approve_query_plan(
            legacy,
            legacy["plan_hash"],
            approved_by="owner",
            approved_at="2026-08-09T01:00:00Z",
        )
    with pytest.raises(QueryPlanDriftError, match="recompile.*version 2"):
        assert_runtime_matches(legacy, legacy["providers"])


def test_venue_operation_graph_is_frozen_into_approval_identity() -> None:
    document = draft()
    document["scope"]["venues"] = ["testconf"]
    document["required_roles"] = ["venue_primary"]
    document["required_providers"] = ["pmlr"]
    primary = provider("pmlr")
    primary["roles"] = ["venue_primary"]
    fallback = provider("openalex")
    fallback["roles"] = ["search", "metadata_enricher"]

    first = compile_query_plan(
        document,
        providers=[primary, fallback],
        venue_specs=[venue_spec()],
    )
    changed = compile_query_plan(
        document,
        providers=[primary, fallback],
        venue_specs=[venue_spec(fallback_role="metadata_enricher")],
    )

    operation = first["venue_operations"][0]
    assert operation["venue_id"] == "testconf"
    assert operation["fallbacks"][0]["native_query_hashes"]
    assert first["plan_hash"] != changed["plan_hash"]
    validate(first, "query-plan.schema.json")


def test_venue_scope_requires_exact_descriptor_and_acceptance_snapshots() -> None:
    document = draft()
    document["scope"]["venues"] = ["testconf"]

    with pytest.raises(QueryPlanError, match="venue specifications do not match scope"):
        compile_query_plan(document, providers=[provider()])


def test_screening_scope_hash_is_derived_after_scope_defaults_and_overwrites_input() -> None:
    document = draft()
    document["filter"]["screening_scope_hash"] = "f" * 64
    implicit = compile_query_plan(document, providers=[provider()])
    explicit_document = draft()
    explicit_document["scope"]["include_arxiv_candidates"] = False
    explicit = compile_query_plan(explicit_document, providers=[provider()])

    assert implicit["filter"]["screening_scope_hash"] == screening_scope_hash(implicit)
    assert implicit["filter"]["screening_scope_hash"] != "f" * 64
    assert implicit["filter"]["screening_scope_hash"] == explicit["filter"][
        "screening_scope_hash"
    ]


@pytest.mark.parametrize(
    ("section", "field", "value"),
    (
        ("research", "objective", "changed objective"),
        ("inclusion", "criteria", ["replicated evidence"]),
        ("scope", "languages", ["zh"]),
    ),
)
def test_screening_scope_hash_changes_only_with_screening_scope(
    section: str,
    field: str,
    value: object,
) -> None:
    baseline = compile_query_plan(draft(), providers=[provider()])
    changed_document = draft()
    changed_document[section][field] = value
    changed = compile_query_plan(changed_document, providers=[provider()])
    budget_document = draft()
    budget_document["budgets"]["max_requests"] = 101
    budget_changed = compile_query_plan(budget_document, providers=[provider()])

    assert changed["filter"]["screening_scope_hash"] != baseline["filter"][
        "screening_scope_hash"
    ]
    assert budget_changed["filter"]["screening_scope_hash"] == baseline["filter"][
        "screening_scope_hash"
    ]


def test_forged_screening_scope_hash_is_rejected_on_approval_and_load(tmp_path) -> None:
    plan = compile_query_plan(draft(), providers=[provider()])
    plan["filter"]["screening_scope_hash"] = "f" * 64
    plan["plan_hash"] = approved_content_hash(plan)

    with pytest.raises(QueryPlanError, match="screening scope hash does not match"):
        approve_query_plan(
            plan,
            plan["plan_hash"],
            approved_by="owner",
            approved_at="2026-08-09T01:00:00Z",
        )

    forged = deepcopy(plan)
    forged["status"] = "approved"
    forged["approval"] = {
        "approved_hash": forged["plan_hash"],
        "approved_by": "owner",
        "approved_at": "2026-08-09T01:00:00Z",
        "approval_method": "cli_hash",
    }
    store = QueryPlanStore(tmp_path)
    path = store.approved_path(str(forged["plan_id"]))
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(forged), encoding="utf-8")

    with pytest.raises(QueryPlanError, match="screening scope hash does not match"):
        store.load_approved(str(forged["plan_id"]))


def test_arxiv_candidate_policy_is_frozen_and_runtime_drift_is_rejected() -> None:
    default_plan = compile_query_plan(draft(), providers=[provider()])
    included_draft = draft()
    included_draft["scope"]["include_arxiv_candidates"] = True
    included_plan = compile_query_plan(included_draft, providers=[provider()])

    assert default_plan["scope"]["include_arxiv_candidates"] is False
    assert included_plan["scope"]["include_arxiv_candidates"] is True
    assert default_plan["plan_hash"] != included_plan["plan_hash"]
    approved = approve_query_plan(
        included_plan,
        included_plan["plan_hash"],
        approved_by="owner",
        approved_at="2026-08-09T01:00:00Z",
    )
    with pytest.raises(QueryPlanDriftError, match="include_arxiv_candidates"):
        assert_runtime_matches(
            approved,
            approved["providers"],
            include_arxiv_candidates=False,
        )


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


def test_restricted_provider_terms_approval_is_frozen_into_plan() -> None:
    document = draft()
    document["terms_approvals"] = [
        {"provider": "openalex", "terms_url": "https://example.test/terms"}
    ]
    specification = {
        **provider(),
        "data_use": "restricted",
        "terms_url": "https://example.test/terms",
    }

    plan = compile_query_plan(document, providers=[specification])

    assert plan["execution"]["terms_approvals"] == document["terms_approvals"]
    validate(plan, "query-plan.schema.json")


def test_terms_approval_rejects_manifest_url_or_upstream_drift() -> None:
    upstream_policy = {
        "authentication": {"required": False},
        "rate_limit": {"global_qps": 1, "max_concurrency": 1, "cache_ttl_seconds": 60},
        "terms": {"data_use": "restricted", "url": "https://example.test/acm-terms"},
    }
    document = draft()
    document["terms_approvals"] = [
        {"provider": "openalex:acm_dl", "terms_url": "https://example.test/acm-terms"}
    ]
    specification = {**provider(), "upstream_policies": {"acm_dl": upstream_policy}}
    assert compile_query_plan(document, providers=[specification])["execution"]["terms_approvals"]

    document["terms_approvals"][0]["terms_url"] = "https://example.test/wrong"
    with pytest.raises(QueryPlanError, match="does not match"):
        compile_query_plan(document, providers=[specification])
