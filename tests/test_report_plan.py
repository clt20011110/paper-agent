from __future__ import annotations

from copy import deepcopy
import json

import pytest

from paper_agent.canonical import canonical_json, content_hash
from paper_agent.report_plan import (
    CLASSIFICATION_AXES,
    REPORT_SECTION_IDS,
    CorpusPaper,
    ReportPlanDriftError,
    ReportPlanError,
    ReportPlanStore,
    approve_report_plan,
    assert_report_runtime_matches,
    build_corpus_snapshot,
    build_search_audit_pack,
    compile_report_plan,
    persist_approved_report_plan,
)
from paper_agent.storage import Database


HASH = "a" * 64


def _raw_search_audit() -> dict:
    return {
        "schema_version": "1",
        "crawl_run_id": "crawl-1",
        "run_id": "search-run-1",
        "search_plan_id": "search-plan-1",
        "plan_hash": HASH,
        "status": "incomplete",
        "incomplete_sources": ["source-semantic-1"],
        "sources": [
            {"source_run_id": "source-openalex-1", "provider": "openalex", "status": "complete"},
            {"source_run_id": "source-semantic-1", "provider": "semantic_scholar", "status": "failed"},
        ],
        "queries": [
            {"query_id": "query-1", "provider": "openalex", "native_query": "graph learning"},
            {"query_id": "query-2", "provider": "semantic_scholar", "native_query": "GNN"},
        ],
        "rounds": [{"round_index": 1, "stop_reason": "budget_exhausted"}],
        "totals": {"sources": {"raw_discovered": 8, "unique_after_dedup": 6}},
    }


def _papers() -> tuple[CorpusPaper, ...]:
    return (
        CorpusPaper(
            "p2",
            "analysis-p2",
            "2" * 64,
            ("b" * 64,),
            "newly_discovered",
            "preprint",
            "simulation",
            "abstract_only",
            "abstract_direct",
            False,
            True,
            analysis_input_tokens=20,
            analysis_pipeline_input_hash="8" * 64,
            analysis_config_hash="9" * 64,
            analysis_implementation_version="stage4-v1",
            analysis_prompt_input_hash="e" * 64,
            analysis_rendered_prompt_hash="f" * 64,
            analysis_invocation_id="invocation-p2",
            analysis_policy_facts_hash="0" * 64,
            publication_date="2025-05-01",
            publication_year=2025,
            venue_id="venue-p2",
            venue_name="Venue Two",
        ),
        CorpusPaper(
            "p1",
            "analysis-p1",
            "1" * 64,
            ("d" * 64, "c" * 64),
            "user_library",
            "peer_reviewed",
            "real",
            "full_pdf",
            "full_text_direct",
            True,
            False,
            analysis_input_tokens=10,
            analysis_pipeline_input_hash="6" * 64,
            analysis_config_hash="7" * 64,
            analysis_implementation_version="stage4-v1",
            analysis_prompt_input_hash="a" * 64,
            analysis_rendered_prompt_hash="b" * 64,
            analysis_invocation_id="invocation-p1",
            analysis_policy_facts_hash="0" * 64,
            publication_date="2023-01-15",
            publication_year=2023,
            venue_id="venue-p1",
            venue_name="Venue One",
        ),
        CorpusPaper(
            "p3",
            None,
            None,
            (),
            "citation_snowball",
            "workshop",
            "theory",
            "missing",
            "metadata_only",
            False,
            True,
            "download not available",
        ),
    )


def _inputs() -> tuple[dict, dict]:
    raw = _raw_search_audit()
    corpus = build_corpus_snapshot(
        _papers(),
        query_plan_hash=HASH,
        search_audit=raw,
        created_at="2026-08-10T00:00:00Z",
    )
    audit = build_search_audit_pack(
        raw,
        corpus,
        screening_flow={
            "raw_discovered": 8,
            "unique_after_dedup": 6,
            "stage2_screened": 5,
            "included": 3,
        },
        exclusion_reasons={"off_topic": 1, "duplicate_version": 1},
        required_providers=("openalex", "semantic_scholar"),
        search_limitations=("One venue API was unavailable",),
        created_at="2026-08-10T00:01:00Z",
    )
    return corpus, audit


def _draft() -> dict:
    sections = [
        {
            "id": section_id,
            "title": section_id.replace("_", " "),
            "subquestion_ids": ["rq1"] if section_id in {"field_taxonomy", "evidence_synthesis"} else [],
            "target_words": 300,
            "evidence_requirements": ["Every substantive fact has evidence"],
            "allowed_evidence_levels": [
                "full_text_direct",
                "full_text_inferred",
                "abstract_direct",
                "metadata_only",
                "corpus_stat",
            ],
        }
        for section_id in REPORT_SECTION_IDS
    ]
    all_sections = list(REPORT_SECTION_IDS)
    return {
        "objective": "Map the evidence without inventing results.",
        "audience": "Researchers",
        "primary_question": "What methods are supported?",
        "subquestions": [{"id": "rq1", "question": "Which methods have direct evidence?"}],
        "synthesis_question": "Under which frozen conditions do methods differ?",
        "scope": {
            "date_from": "2020-01-01",
            "date_to": "2026-08-10",
            "venues": ["ICML"],
            "document_types": ["article", "preprint"],
            "languages": ["en"],
            "inclusion_criteria": ["Relevant to rq1"],
            "exclusion_criteria": ["Off topic"],
        },
        "stage4b_config_hash": "f" * 64,
        "stage4b_audit_config_hash": "e" * 64,
        "aggregation": {
            "max_chunk_input_tokens": 1_000,
            "reduce_output_tokens": 5_000,
        },
        "sections": sections,
        "classification_axes": list(CLASSIFICATION_AXES),
        "cohort_rules": {
            "recent_cutoff": "2024-01-01",
            "foundational_rule": "Explicit user seed or frozen citation threshold",
            "peer_review_rule": "Canonical publication status only",
            "study_setting_rule": "Registry-backed real/simulation/theory labels",
        },
        "paper_memberships": [
            {"paper_id": "p1", "section_ids": all_sections, "primary_section_id": "evidence_synthesis", "coverage_disposition": "evidence", "coverage_reason": None, "resource_table_ids": []},
            {"paper_id": "p2", "section_ids": ["field_taxonomy", "evidence_synthesis"], "primary_section_id": "evidence_synthesis", "coverage_disposition": "evidence", "coverage_reason": None, "resource_table_ids": []},
            {"paper_id": "p3", "section_ids": ["report_limitations", "references_and_appendices"], "primary_section_id": "report_limitations", "coverage_disposition": "evidence", "coverage_reason": None, "resource_table_ids": []},
        ],
        "artifacts": {
            "comparison_tables": ["methods"],
            "trend_statistics": ["publication years"],
            "resource_tables": ["code and data"],
            "appendices": ["query manifest", "coverage ledger"],
        },
        "budget": {
            "max_sol_calls": 30,
            "max_input_tokens": 100_000,
            "max_retries": 1,
            "audit_calls": 2,
            "repair_calls": 1,
        },
    }


def _compiled(*, created_at: str = "2026-08-10T00:02:00Z") -> tuple[dict, dict, dict]:
    corpus, audit = _inputs()
    plan = compile_report_plan(
        _draft(),
        corpus_snapshot=corpus,
        search_audit_pack=audit,
        created_at=created_at,
    )
    return plan, corpus, audit


def test_compiled_report_plan_binds_frozen_inputs_prompts_and_required_contract() -> None:
    plan, corpus, audit = _compiled()
    second, _, _ = _compiled(created_at="2026-08-10T02:00:00Z")

    assert plan["plan_hash"] == second["plan_hash"]
    assert plan["plan_id"] == second["plan_id"]
    assert plan["query_plan_hash"] == HASH
    assert plan["corpus_snapshot_hash"] == corpus["snapshot_hash"]
    assert plan["search_audit_pack_hash"] == audit["pack_hash"]
    assert set(plan["prompt_hashes"]) == {
        "planning_assist",
        "section_reduce",
        "cross_section_reduce",
        "final_reduce",
        "quality_audit",
        "repair",
    }
    assert set(REPORT_SECTION_IDS).issubset(section["id"] for section in plan["sections"])
    assert audit["source_audit_hash"] == corpus["search_audit_source_hash"]


def test_approval_is_detached_and_business_drift_is_rejected() -> None:
    plan, corpus, audit = _compiled()
    approved = approve_report_plan(
        plan,
        plan["plan_hash"],
        approved_by="owner",
        approved_at="2026-08-10T00:03:00Z",
    )
    runtime, _, _ = _compiled(created_at="2026-08-11T00:00:00Z")

    assert_report_runtime_matches(
        approved,
        runtime,
        corpus_snapshot=corpus,
        search_audit_pack=audit,
    )
    runtime["budget"] = {**runtime["budget"], "max_sol_calls": 29}
    with pytest.raises(ReportPlanDriftError, match="has drifted"):
        assert_report_runtime_matches(
            approved,
            runtime,
            corpus_snapshot=corpus,
            search_audit_pack=audit,
        )


def test_runtime_rejects_corpus_query_and_approval_drift() -> None:
    plan, corpus, audit = _compiled()
    approved = approve_report_plan(
        plan,
        plan["plan_hash"],
        approved_by="owner",
        approved_at="2026-08-10T00:03:00Z",
    )
    runtime, _, _ = _compiled()
    changed = deepcopy(corpus)
    changed["papers"][0]["publication_status"] = "workshop"
    with pytest.raises(ReportPlanDriftError, match="hash has drifted"):
        assert_report_runtime_matches(
            approved,
            runtime,
            corpus_snapshot=changed,
            search_audit_pack=audit,
        )

    changed_audit = deepcopy(audit)
    changed_audit["limitations"].append("New frozen limitation")
    audit_core = {
        key: value
        for key, value in changed_audit.items()
        if key not in {"pack_id", "pack_hash", "created_at"}
    }
    changed_audit["pack_hash"] = content_hash(audit_core)
    changed_audit["pack_id"] = f"search-audit-{changed_audit['pack_hash'][:12]}"
    with pytest.raises(ReportPlanDriftError, match="search audit pack has drifted"):
        assert_report_runtime_matches(
            approved,
            runtime,
            corpus_snapshot=corpus,
            search_audit_pack=changed_audit,
        )

    approved["objective"] = "Changed after approval"
    with pytest.raises(ReportPlanDriftError, match="approved document content has drifted"):
        assert_report_runtime_matches(
            approved,
            runtime,
            corpus_snapshot=corpus,
            search_audit_pack=audit,
        )


def test_plan_rejects_silent_corpus_omission_and_section_contract_drift() -> None:
    corpus, audit = _inputs()
    draft = _draft()
    draft["paper_memberships"] = draft["paper_memberships"][:-1]
    with pytest.raises(ReportPlanError, match="does not exactly match"):
        compile_report_plan(draft, corpus_snapshot=corpus, search_audit_pack=audit, created_at="2026-08-10T00:02:00Z")

    draft = _draft()
    draft["budget"]["audit_calls"] = 3
    with pytest.raises(ReportPlanError, match="audit_calls"):
        compile_report_plan(
            draft,
            corpus_snapshot=corpus,
            search_audit_pack=audit,
            created_at="2026-08-10T00:02:00Z",
        )

    draft = _draft()
    draft["sections"] = draft["sections"][:-1]
    with pytest.raises(ReportPlanError, match="missing required report sections"):
        compile_report_plan(draft, corpus_snapshot=corpus, search_audit_pack=audit, created_at="2026-08-10T00:02:00Z")


@pytest.mark.parametrize(
    ("disposition", "reason", "table_ids"),
    [
        ("background_only", None, ()),
        ("resource_or_background_table", None, ()),
        ("resource_or_background_table", None, ("unknown-table",)),
        ("evidence", "not allowed", ()),
    ],
)
def test_plan_rejects_invalid_frozen_coverage_dispositions(
    disposition: str,
    reason: str | None,
    table_ids: tuple[str, ...],
) -> None:
    corpus, audit = _inputs()
    draft = _draft()
    draft["paper_memberships"][0].update({
        "coverage_disposition": disposition,
        "coverage_reason": reason,
        "resource_table_ids": list(table_ids),
    })

    with pytest.raises(ReportPlanError, match="coverage disposition"):
        compile_report_plan(
            draft,
            corpus_snapshot=corpus,
            search_audit_pack=audit,
            created_at="2026-08-10T00:02:00Z",
        )

def test_corpus_snapshot_is_stable_sorted_and_binds_the_raw_search_audit() -> None:
    raw = _raw_search_audit()
    first = build_corpus_snapshot(
        _papers(), query_plan_hash=HASH, search_audit=raw, created_at="2026-08-10T00:00:00Z"
    )
    second = build_corpus_snapshot(
        tuple(reversed(_papers())), query_plan_hash=HASH, search_audit=raw, created_at="later"
    )

    assert first["snapshot_hash"] == second["snapshot_hash"]
    assert first["schema_version"] == "2"
    assert first["analysis_token_estimator"] == "frozen-stage4-input-estimate-v1"
    assert [paper["paper_id"] for paper in first["papers"]] == ["p1", "p2", "p3"]
    assert first["papers"][0]["lineage_hashes"] == ["c" * 64, "d" * 64]
    changed = deepcopy(raw)
    changed["plan_hash"] = "b" * 64
    with pytest.raises(ReportPlanError, match="QueryPlan hash"):
        build_corpus_snapshot(_papers(), query_plan_hash=HASH, search_audit=changed, created_at="now")


def test_search_audit_pack_reconciles_flow_and_discloses_incomplete_sources() -> None:
    corpus, audit = _inputs()

    assert audit["flow"] == {
        "raw_discovered": 8,
        "unique_after_dedup": 6,
        "stage2_screened": 5,
        "included": 3,
        "excluded": 2,
        "excluded_by_reason": {"duplicate_version": 1, "off_topic": 1},
        "full_pdf": 1,
        "abstract_only": 1,
        "missing": 1,
    }
    assert audit["source_categories"] == {
        "citation_snowball": 1,
        "newly_discovered": 1,
        "user_library": 1,
    }
    assert audit["required_provider_failures"] == ["semantic_scholar"]
    assert audit["budget_exhausted"] is True
    assert "search budget exhausted" in audit["limitations"]
    assert audit["query_manifest"] == _raw_search_audit()["queries"]
    assert audit["corpus_snapshot_hash"] == corpus["snapshot_hash"]


def test_search_audit_pack_rejects_unreconciled_screening_counts() -> None:
    raw = _raw_search_audit()
    corpus = build_corpus_snapshot(
        _papers(), query_plan_hash=HASH, search_audit=raw, created_at="now"
    )
    with pytest.raises(ReportPlanError, match="exclusion reasons"):
        build_search_audit_pack(
            raw,
            corpus,
            screening_flow={
                "raw_discovered": 8,
                "unique_after_dedup": 6,
                "stage2_screened": 5,
                "included": 3,
            },
            exclusion_reasons={"off_topic": 1},
            created_at="now",
        )


def test_store_keeps_approved_bundle_immutable_and_validates_on_load(tmp_path) -> None:
    plan, corpus, audit = _compiled()
    store = ReportPlanStore(tmp_path)
    store.save_draft(plan)
    approved = store.approve_and_save(
        plan,
        plan["plan_hash"],
        approved_by="owner",
        approved_at="2026-08-10T00:03:00Z",
        corpus_snapshot=corpus,
        search_audit_pack=audit,
    )

    bundle = store.load_bundle(approved["plan_id"])
    assert bundle.plan == approved
    assert bundle.corpus_snapshot == corpus
    assert json.loads(store.latest_path.read_text()) == {
        "plan_id": approved["plan_id"],
        "plan_hash": approved["plan_hash"],
    }

    changed_audit = deepcopy(audit)
    changed_audit["limitations"].append("new limitation")
    changed_audit["pack_hash"] = "0" * 64
    with pytest.raises(ReportPlanError, match="hash has drifted"):
        store.save_bundle(approved, corpus, changed_audit)

    path = store.approved_path(approved["plan_id"])
    tampered = json.loads(path.read_text())
    tampered["audience"] = "Other audience"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ReportPlanError, match="drifted"):
        store.load_approved(approved["plan_id"])


def test_approved_plan_has_an_idempotent_immutable_database_entrypoint(tmp_path) -> None:
    plan, _, _ = _compiled()
    approved = approve_report_plan(
        plan,
        plan["plan_hash"],
        approved_by="owner",
        approved_at="2026-08-10T00:03:00Z",
    )
    with Database(tmp_path / "paper-agent.sqlite") as database:
        database.migrate()
        persist_approved_report_plan(database, approved)
        persist_approved_report_plan(database, approved)
        row = database.connection.execute(
            """SELECT content_hash, status FROM report_plans
               WHERE report_plan_id = ?""",
            (approved["plan_id"],),
        ).fetchone()
        assert tuple(row) == (approved["plan_hash"], "approved")
        database.connection.execute(
            "UPDATE report_plans SET plan_json = '{}' WHERE report_plan_id = ?",
            (approved["plan_id"],),
        )
        database.connection.commit()
        with pytest.raises(ReportPlanError, match="immutable"):
            persist_approved_report_plan(database, approved)


def test_store_load_rechecks_query_plan_hash_across_the_bundle(tmp_path) -> None:
    plan, corpus, audit = _compiled()
    store = ReportPlanStore(tmp_path)
    approved = store.approve_and_save(
        plan,
        plan["plan_hash"],
        approved_by="owner",
        approved_at="2026-08-10T00:03:00Z",
        corpus_snapshot=corpus,
        search_audit_pack=audit,
    )
    raw = _raw_search_audit()
    raw["plan_hash"] = "b" * 64
    foreign_corpus = build_corpus_snapshot(
        _papers(),
        query_plan_hash="b" * 64,
        search_audit=raw,
        created_at="2026-08-10T00:04:00Z",
    )
    foreign_audit = build_search_audit_pack(
        raw,
        foreign_corpus,
        screening_flow={
            "raw_discovered": 8,
            "unique_after_dedup": 6,
            "stage2_screened": 5,
            "included": 3,
        },
        exclusion_reasons={"off_topic": 1, "duplicate_version": 1},
        required_providers=("openalex", "semantic_scholar"),
        search_limitations=("One venue API was unavailable",),
        created_at="2026-08-10T00:05:00Z",
    )
    directory = store.directory(approved["plan_id"])
    (directory / "CORPUS_SNAPSHOT.json").write_bytes(canonical_json(foreign_corpus))
    (directory / "SEARCH_AUDIT.json").write_bytes(canonical_json(foreign_audit))

    with pytest.raises(ReportPlanError, match="QueryPlan"):
        store.load_bundle(approved["plan_id"])
