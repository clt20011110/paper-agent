from __future__ import annotations

from copy import deepcopy
import sqlite3

import pytest

from paper_agent.canonical import content_hash
from paper_agent.report_artifacts import validate_claim_relations
from paper_agent.report_facts import (
    ReportFactError,
    materialize_verified_report_facts,
    require_verified_report_claims,
    require_verified_report_facts,
)
from paper_agent.reporting import (
    comparison_assessment,
    derive_comparison_groups,
    stable_claim_id,
)
from paper_agent.storage import Database


def _seed_runs(database: Database) -> None:
    connection = database.connection
    connection.execute("INSERT INTO papers(paper_id, title) VALUES ('paper-1', 'Paper')")
    connection.executemany(
        """INSERT INTO pipeline_runs(
               run_id, stage, status, input_hash, config_hash, implementation_version
           ) VALUES (?, ?, 'complete', ?, 'config', 'test')""",
        (
            ("analysis-pipeline", "stage4", "analysis-input"),
            ("search-pipeline", "search", "search-input"),
            ("previous-pipeline", "stage4b", "previous-input"),
            ("current-pipeline", "stage4b", "current-input"),
        ),
    )
    connection.execute(
        """INSERT INTO analysis_runs(
               analysis_run_id, run_id, paper_id, input_hash, input_scope,
               model_id, model_revision, prompt_hash, schema_hash,
               implementation_version, policy_version, policy_decision, status
           ) VALUES (
               'analysis-1', 'analysis-pipeline', 'paper-1', 'analysis-input',
               'full_pdf', 'model', 'revision', 'prompt', 'schema', 'test',
               'policy', 'allow', 'complete'
           )"""
    )
    connection.execute(
        """INSERT INTO search_plans(
               search_plan_id, content_hash, schema_version, plan_json, status
           ) VALUES ('search-plan', 'search-plan-hash', '1', '{}', 'approved')"""
    )
    connection.execute(
        """INSERT INTO crawl_runs(
               crawl_run_id, run_id, search_plan_id, status
           ) VALUES ('crawl-1', 'search-pipeline', 'search-plan', 'complete')"""
    )
    connection.execute(
        """INSERT INTO source_runs(
               source_run_id, crawl_run_id, provider, provider_version, role, status
           ) VALUES ('source-1', 'crawl-1', 'openalex', '1', 'search', 'complete')"""
    )
    connection.execute(
        """INSERT INTO search_queries(
               query_id, search_plan_id, source_run_id, provider,
               provider_version, query_compiler_version, role, query_text,
               query_hash, status
           ) VALUES (
               'query-1', 'search-plan', 'source-1', 'openalex', '1',
               'compiler-1', 'search', 'frozen query', 'query-hash', 'complete'
           )"""
    )
    connection.executemany(
        """INSERT INTO report_plans(
               report_plan_id, content_hash, schema_version, plan_json,
               approval_json, status
           ) VALUES (?, ?, '1', '{}', '{}', 'approved')""",
        (
            ("previous-plan", "previous-plan-hash"),
            ("current-plan", "current-plan-hash"),
        ),
    )
    connection.executemany(
        """INSERT INTO report_runs(
               report_run_id, run_id, report_plan_id, corpus_snapshot_hash,
               aggregation_tree_json, model_id, model_revision, prompt_hash,
               schema_hash, status
           ) VALUES (?, ?, ?, 'corpus', '{}', 'gpt-5.6-sol',
                     'codex-cli-managed', 'prompt', 'schema', 'running')""",
        (
            ("previous-report", "previous-pipeline", "previous-plan"),
            ("current-report", "current-pipeline", "current-plan"),
        ),
    )
    connection.executemany(
        """INSERT INTO report_audit_runs(
               report_run_id, input_snapshot_hash, base_artifact_hash,
               current_artifact_hash, current_bundle_json, rubric_hash,
               profile, model_id, reasoning_effort, config_hash, execution_mode,
               worst_case_calls, worst_case_input_tokens, status
           ) VALUES (?, 'snapshot', 'base', 'current', '{}', 'rubric',
                     'stage4b_summary_sol', 'gpt-5.6-sol', 'high', 'config',
                     'attended', 1, 1, 'running')""",
        (("previous-report",), ("current-report",)),
    )
    connection.commit()


def _evidence_reference() -> dict:
    unit = {
        "claim": "A measured result.",
        "direction": "support",
        "task_id": "task-1",
        "dataset_id": "dataset-1",
        "dataset_version": "v1",
        "split_id": "test",
        "metric_id": "accuracy",
        "metric_definition_hash": "1" * 64,
        "unit": "%",
        "optimization_direction": "maximize",
        "value": 91.0,
        "uncertainty": None,
        "statistical_method": None,
        "protocol_id": "protocol-1",
        "protocol_hash": "2" * 64,
        "sample_size": 100,
        "baseline_id": "baseline-1",
        "baseline_version": "v1",
        "conditions": ["frozen"],
        "locator": {"kind": "page", "value": "7"},
        "normalization_method": "identity",
        "normalizer_version": "v1",
        "source_value": 91.0,
        "comparison_eligibility": "comparable",
        "missing_fields": [],
    }
    return {
        "kind": "paper_evidence",
        "evidence_level": "full_text_direct",
        "paper_id": "paper-1",
        "analysis_run_id": "analysis-1",
        "evidence_unit": unit,
        "locator": "page:7",
        "search_plan_id": None,
        "source_run_id": None,
        "query_id": None,
        "statistic": None,
        "calculation": None,
    }


def _claim(subject: str) -> dict:
    reference = _evidence_reference()
    provisional = {
        "claim_id": "pending",
        "claim_key": {
            "subject_id": subject,
            "predicate_id": "improves",
            "object_or_scope_id": "scope-1",
            "qualifier_context_hash": "3" * 64,
            "comparison_group_id": None,
        },
        "research_question_id": "rq-1",
        "report_section": "evidence",
        "claim_text": f"{subject} improves the measured result.",
        "claim_type": "finding",
        "supporting_evidence": [reference],
        "contradicting_evidence": [],
        "evidence_level": "full_text_direct",
        "comparison_group_id": None,
        "confidence": "high",
        "known_limitations": [],
        "status": "supported",
        "mapping_status": "mapped",
    }
    group_id = comparison_assessment(reference["evidence_unit"]).comparison_group_id
    assert group_id is not None
    provisional["comparison_group_id"] = group_id
    provisional["claim_key"]["comparison_group_id"] = group_id
    provisional["claim_id"] = stable_claim_id(provisional["claim_key"], report_run_id="unused")
    return provisional


def _corpus_claim() -> dict:
    provisional = {
        "claim_id": "pending",
        "claim_key": {
            "subject_id": "frozen-corpus",
            "predicate_id": "contains",
            "object_or_scope_id": "eligible-papers",
            "qualifier_context_hash": "4" * 64,
            "comparison_group_id": None,
        },
        "research_question_id": "rq-1",
        "report_section": "evidence",
        "claim_text": "The frozen corpus contains two eligible papers.",
        "claim_type": "corpus_stat",
        "supporting_evidence": [{
            "kind": "corpus_evidence",
            "evidence_level": "corpus_stat",
            "paper_id": None,
            "analysis_run_id": None,
            "evidence_unit": None,
            "locator": None,
            "search_plan_id": "search-plan",
            "source_run_id": "source-1",
            "query_id": "query-1",
            "statistic": "eligible_paper_count=2",
            "calculation": "count(frozen eligible paper ids)",
        }],
        "contradicting_evidence": [],
        "evidence_level": "corpus_stat",
        "comparison_group_id": None,
        "confidence": "high",
        "known_limitations": [],
        "status": "supported",
        "mapping_status": "mapped",
    }
    provisional["claim_id"] = stable_claim_id(
        provisional["claim_key"], report_run_id="unused"
    )
    return provisional


def _verification(document: dict) -> dict:
    return {
        "report_document_hash": content_hash(document),
        "claim_count": 1,
        "coverage_complete": True,
        "checks": {
            "no_unsupported_claims": True,
            "citation_coverage": True,
            "table_provenance": True,
            "search_limitations": True,
            "extraction_scope": True,
            "no_fabricated_statistics": True,
        },
    }


def _bundle(report_run_id: str, claim: dict, relations=()) -> dict:
    return {
        "document": {"report_run_id": report_run_id},
        "claims": [claim],
        "comparison_groups": derive_comparison_groups([claim]),
        "claim_relations": list(relations),
    }


def test_verified_report_facts_are_normalized_sealed_and_idempotent(tmp_path) -> None:
    with Database(tmp_path / "papers.sqlite3") as database:
        database.migrate()
        _seed_runs(database)
        previous_claim = _claim("previous-subject")
        current_claim = _claim("current-subject")
        relation = validate_claim_relations(
            {"claims": [previous_claim]},
            [current_claim],
            [{
                "previous_claim_id": previous_claim["claim_id"],
                "current_claim_id": current_claim["claim_id"],
                "relation_type": "refined",
                "reason": "The canonical subject was refined.",
                "evidence_diff": {
                    "added_support": [],
                    "removed_support": [],
                    "added_contradiction": [],
                    "removed_contradiction": [],
                },
            }],
        )
        previous = _bundle("previous-report", previous_claim)
        current = _bundle("current-report", current_claim, relation)

        with database.transaction() as connection:
            materialize_verified_report_facts(
                connection,
                report_run_id="previous-report",
                bundle=previous,
                deterministic_verification=_verification(previous["document"]),
            )
            connection.execute(
                "UPDATE report_runs SET status = 'complete' "
                "WHERE report_run_id = 'previous-report'"
            )
            connection.execute(
                "UPDATE report_audit_runs SET status = 'complete' "
                "WHERE report_run_id = 'previous-report'"
            )
        require_verified_report_claims(
            database.connection,
            report_run_id="previous-report",
            claims=previous["claims"],
        )
        drifted_previous_claims = deepcopy(previous["claims"])
        drifted_previous_claims[0]["claim_text"] = "Forged prior evidence context."
        with pytest.raises(ReportFactError, match="differ"):
            require_verified_report_claims(
                database.connection,
                report_run_id="previous-report",
                claims=drifted_previous_claims,
            )
        with database.transaction() as connection:
            materialize_verified_report_facts(
                connection,
                report_run_id="current-report",
                bundle=current,
                deterministic_verification=_verification(current["document"]),
                previous_report_run_id="previous-report",
            )
        before = database.connection.total_changes
        with database.transaction() as connection:
            materialize_verified_report_facts(
                connection,
                report_run_id="current-report",
                bundle=current,
                deterministic_verification=_verification(current["document"]),
                previous_report_run_id="previous-report",
            )
        assert database.connection.total_changes == before

        fact_set = database.connection.execute(
            "SELECT sealed, claim_count, evidence_count, comparison_group_count, "
            "claim_relation_count FROM report_fact_sets WHERE report_run_id = 'current-report'"
        ).fetchone()
        assert tuple(fact_set) == (1, 1, 1, 1, 1)
        claim = database.connection.execute(
            "SELECT confidence, evidence_level, mapping_status FROM report_claims "
            "WHERE report_run_id = 'current-report'"
        ).fetchone()
        assert tuple(claim) == ("high", "full_text_direct", "mapped")
        evidence = database.connection.execute(
            "SELECT direction, ordinal, paper_id, analysis_run_id FROM claim_evidence "
            "WHERE report_run_id = 'current-report'"
        ).fetchone()
        assert tuple(evidence) == ("support", 0, "paper-1", "analysis-1")
        relation_row = database.connection.execute(
            "SELECT previous_report_run_id, current_report_run_id, relation_type "
            "FROM claim_relations WHERE current_report_run_id = 'current-report'"
        ).fetchone()
        assert tuple(relation_row) == (
            "previous-report",
            "current-report",
            "refined",
        )

        require_verified_report_facts(
            database.connection,
            report_run_id="current-report",
            bundle=current,
            deterministic_verification=_verification(current["document"]),
            previous_report_run_id="previous-report",
        )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            database.connection.execute(
                "UPDATE report_claims SET claim_text = 'drift' "
                "WHERE report_run_id = 'current-report'"
            )


def test_same_report_fact_key_with_different_values_is_rejected(tmp_path) -> None:
    with Database(tmp_path / "papers.sqlite3") as database:
        database.migrate()
        _seed_runs(database)
        claim = _claim("stable-subject")
        bundle = _bundle("current-report", claim)
        verification = _verification(bundle["document"])
        with database.transaction() as connection:
            materialize_verified_report_facts(
                connection,
                report_run_id="current-report",
                bundle=bundle,
                deterministic_verification=verification,
            )

        drifted = deepcopy(bundle)
        drifted["claims"][0]["claim_text"] = "Conflicting text for the same run key."
        with pytest.raises(ReportFactError, match="different values"):
            with database.transaction() as connection:
                materialize_verified_report_facts(
                    connection,
                    report_run_id="current-report",
                    bundle=drifted,
                    deterministic_verification=verification,
                )


def test_corpus_evidence_is_bound_to_the_frozen_search_lineage(tmp_path) -> None:
    with Database(tmp_path / "papers.sqlite3") as database:
        database.migrate()
        _seed_runs(database)
        claim = _corpus_claim()
        bundle = _bundle("current-report", claim)

        with database.transaction() as connection:
            materialize_verified_report_facts(
                connection,
                report_run_id="current-report",
                bundle=bundle,
                deterministic_verification=_verification(bundle["document"]),
            )

        evidence = database.connection.execute(
            """SELECT evidence_kind, evidence_level, search_plan_id,
                      source_run_id, query_id, statistic, calculation
               FROM claim_evidence WHERE report_run_id = 'current-report'"""
        ).fetchone()
        assert tuple(evidence) == (
            "corpus_evidence",
            "corpus_stat",
            "search-plan",
            "source-1",
            "query-1",
            "eligible_paper_count=2",
            "count(frozen eligible paper ids)",
        )
        assert database.connection.execute("PRAGMA foreign_key_check").fetchall() == []
