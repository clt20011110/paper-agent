from __future__ import annotations

import json
from pathlib import Path

import pytest

from paper_agent.approval import approve, approved_content_hash
from paper_agent.artifacts import ArtifactStore
from paper_agent.canonical import canonical_json, content_hash
from paper_agent.report_input_service import (
    ReportInputError,
    ReportInputRequest,
    ReportInputService,
)
from paper_agent.storage import Database


HASH = "a" * 64
CONFIG_HASH = "b" * 64
PROMPT_INPUT_HASH = "c" * 64
PROMPT_HASH = "d" * 64
SCHEMA_HASH = "e" * 64
RENDERED_HASH = "f" * 64


def _request(*, include_needs_review: bool = False) -> ReportInputRequest:
    return ReportInputRequest(
        crawl_run_id="crawl-1",
        filter_run_id="filter-1",
        stage4_run_id="stage4-1",
        recent_cutoff="2024-01-01",
        created_at="2026-08-11T00:00:00Z",
        include_needs_review=include_needs_review,
    )


def _fixture(tmp_path: Path) -> tuple[Database, ArtifactStore, ReportInputService]:
    database = Database(tmp_path / "papers.sqlite3")
    database.migrate()
    store = ArtifactStore(tmp_path / "store")
    plan = {
        "plan_id": "plan-1",
        "plan_hash": "",
        "status": "draft",
        "created_at": "2026-08-11T00:00:00Z",
        "scope": {"user_seeds": []},
        "execution": {"required_providers": ["openalex"]},
        "approval": None,
    }
    plan_hash = approved_content_hash(plan)
    plan["plan_hash"] = plan_hash
    plan = approve(
        plan,
        plan_hash,
        approved_by="fixture",
        approved_at="2026-08-11T00:00:00Z",
        hash_field="plan_hash",
    )
    database.connection.execute(
        """INSERT INTO search_plans(
               search_plan_id, content_hash, schema_version, plan_json, status
           ) VALUES ('plan-1', ?, '1', ?, 'approved')""",
        (plan_hash, json.dumps(plan)),
    )
    database.connection.executemany(
        """INSERT INTO pipeline_runs(
               run_id, stage, status, input_hash, config_hash, implementation_version
           ) VALUES (?, ?, ?, ?, ?, ?)""",
        (
            ("search-1", "search", "complete", "1" * 64, "2" * 64, "search-v1"),
            ("filter-1", "stage-2", "complete", "3" * 64, "4" * 64, "stage2-v1"),
            ("stage4-1", "stage4", "incomplete", HASH, CONFIG_HASH, "stage4-v1"),
        ),
    )
    database.connection.execute(
        """INSERT INTO crawl_runs(
               crawl_run_id, run_id, search_plan_id, status
           ) VALUES ('crawl-1', 'search-1', 'plan-1', 'complete')"""
    )
    database.connection.execute(
        """INSERT INTO source_runs(
               source_run_id, crawl_run_id, provider, provider_version, role, status
           ) VALUES ('source-1', 'crawl-1', 'openalex', '1', 'search', 'complete')"""
    )
    database.connection.execute(
        """INSERT INTO source_run_audits(
               source_run_id, raw_discovered, unique_after_dedup, screened,
               excluded, included, updated_at
           ) VALUES ('source-1', 4, 4, 4, 2, 2, '2026-08-11T00:00:00Z')"""
    )
    database.connection.execute(
        """INSERT INTO search_rounds(
               search_round_id, crawl_run_id, round_index, state,
               seed_manifest_hash, request_schedule_hash, stop_reason, completed_at
           ) VALUES (
               'round-1', 'crawl-1', 0, 'complete', ?, ?, 'sources_exhausted',
               '2026-08-11T00:00:00Z'
           )""",
        ("5" * 64, "6" * 64),
    )
    papers = (
        ("p1", "Analyzed", "[\"Ada Lovelace\"]", "2023-06-01", 2023, "Journal One", "journal"),
        ("p2", "Missing", "[\"Grace Hopper\"]", "2025-02-01", 2025, "Workshop Two", "conference"),
        ("p3", "Excluded", "[]", "2024-01-01", 2024, "Venue Three", "journal"),
        ("p4", "Review", "[]", None, 2025, "Venue Four", "preprint"),
    )
    database.connection.executemany(
        """INSERT INTO papers(
               paper_id, title, authors_json, publication_date, year,
               venue_name, venue_type, verification_status
           ) VALUES (?, ?, ?, ?, ?, ?, ?, 'verified')""",
        papers,
    )
    database.connection.executemany(
        """INSERT INTO crawl_paper_snapshots(
               crawl_run_id, paper_id, metadata_hash, status_version_json
           ) VALUES ('crawl-1', ?, ?, '{}')""",
        tuple((paper[0], content_hash({"paper_id": paper[0]})) for paper in papers),
    )
    database.connection.execute(
        """INSERT INTO search_round_seeds(
               search_round_id, paper_id, seed_reason, parent_round, depth,
               seed_rank, selector_version, selector_config_hash
           ) VALUES ('round-1', 'p1', 'user_seed', 0, 0, 0, 'selector-v1', ?)""",
        ("7" * 64,),
    )
    database.connection.execute(
        """INSERT INTO search_round_papers(
               search_round_id, paper_id, depth, first_seen, screening_status
           ) VALUES ('round-1', 'p2', 1, 1, 'relevant')"""
    )
    database.connection.executemany(
        """INSERT INTO filter_decisions(
               filter_decision_id, run_id, paper_id, status, threshold_version,
               reason, input_hash, implementation_version
           ) VALUES (?, 'filter-1', ?, ?, 'threshold-v1', ?, ?, 'stage2-v1')""",
        (
            ("decision-p1", "p1", "relevant", '{"reason_code":"topic_match"}', "8" * 64),
            ("decision-p2", "p2", "relevant", '{"reason_code":"topic_match"}', "9" * 64),
            ("decision-p3", "p3", "irrelevant", '{"reason_code":"off_topic"}', "0" * 64),
            ("decision-p4", "p4", "needs_review", '{"reason_code":"ambiguous"}', "1" * 64),
        ),
    )
    _analysis(database, store)
    database.connection.commit()
    return database, store, ReportInputService(database, store, tmp_path / "release")


def _analysis(database: Database, store: ArtifactStore) -> None:
    pdf = store.put_bytes(b"%PDF-1.7 source", mime_type="application/pdf")
    text = store.put_bytes(b"normalized source", mime_type="text/plain")
    database.connection.executemany(
        """INSERT INTO artifacts(
               artifact_id, paper_id, artifact_kind, relative_path, mime_type,
               byte_size, sha256, provenance_json
           ) VALUES (?, 'p1', ?, ?, ?, ?, ?, '{}')""",
        (
            ("pdf-p1", "pdf", pdf.relative_path, pdf.mime_type, pdf.size_bytes, pdf.artifact_hash),
            ("text-p1", "text", text.relative_path, text.mime_type, text.size_bytes, text.artifact_hash),
        ),
    )
    database.connection.execute(
        """INSERT INTO text_extractions(
               extraction_id, paper_id, source_artifact_id, source_sha256,
               output_artifact_id, extractor_name, extractor_version, page_count,
               character_count, text_coverage, printable_ratio, status
           ) VALUES (
               'extract-p1', 'p1', 'pdf-p1', ?, 'text-p1', 'fixture', '1',
               1, 17, 1.0, 1.0, 'full_text_ready'
           )""",
        (pdf.artifact_hash,),
    )
    document = {
        "paper_id": "p1",
        "artifact_hash": text.artifact_hash,
        "input_scope": "full_pdf",
        "model": "gpt-5.6-luna",
        "model_revision": "luna-fixture",
        "prompt_hash": PROMPT_HASH,
        "schema_hash": SCHEMA_HASH,
        "created_at": "2026-08-11T00:00:00Z",
        "research_question_and_motivation": "Question",
        "summary": "Summary",
        "methods": [],
        "key_techniques": [],
        "datasets": [],
        "experimental_setup": [],
        "metrics": [],
        "results": [],
        "limitations": [],
        "credibility": "Bounded",
        "resources": [],
        "topic_relevance": "Relevant",
        "labels": {
            "subquestion": [],
            "theme": ["theme-a"],
            "method_family": [],
            "task": [],
            "dataset": [],
            "benchmark": [],
            "evidence_type": [],
            "publication_status": "preprint",
            "study_setting": "theory",
        },
        "label_evidence": [],
        "evidence_units": [],
        "comparison_eligibility": "not_comparable",
        "missing_fields": ["comparison_evidence"],
    }
    payload = canonical_json(document)
    output = store.put_bytes(payload, mime_type="application/json")
    database.connection.execute(
        """INSERT INTO artifacts(
               artifact_id, paper_id, artifact_kind, relative_path, mime_type,
               byte_size, sha256, provenance_json
           ) VALUES (
               'analysis-output-p1', 'p1', 'analysis', ?, 'application/json', ?, ?, ?
           )""",
        (
            output.relative_path,
            output.size_bytes,
            output.artifact_hash,
            json.dumps({"analysis_run_id": "analysis-p1", "stage": "stage4", "format": "json"}),
        ),
    )
    policy_facts = {
        "paper_id": "p1",
        "artifact_hash": text.artifact_hash,
        "artifact": "normalized_text",
        "input_scope": "full_pdf",
        "license": "CC-BY-4.0",
        "access_basis": "open_license",
        "domain": "example.test",
        "mode": "attended",
        "collection_id": None,
        "collection_snapshot_hash": None,
        "selection_snapshot_hash": None,
        "data_category": "normalized_text",
    }
    decision = {
        "policy_version": "policy-v1",
        "policy_hash": "2" * 64,
        "outcome": "full_pdf",
        "reason_code": "allowed",
        "input_artifact_hash": text.artifact_hash,
        "provider": "codex_cli",
        "model": "gpt-5.6-luna",
        "purpose": "internal_analysis",
        "data_category": "normalized_text",
        "processing_grant_id": None,
        "authorized_by": "policy",
    }
    invocation = {
        "invocation_id": "invocation-p1",
        "profile": "stage4_analysis_luna",
        "model": "gpt-5.6-luna",
        "reasoning_effort": "medium",
        "schema_name": "paper-analysis.schema.json",
        "schema_hash": SCHEMA_HASH,
        "input_hash": PROMPT_INPUT_HASH,
        "prompt_name": "paper-analysis.md",
        "prompt_hash": PROMPT_HASH,
        "rendered_prompt_hash": RENDERED_HASH,
        "call_kind": None,
        "attempts": 1,
        "actual_model": "gpt-5.6-luna",
        "actual_profile": "stage4_analysis_luna",
    }
    metadata = json.dumps(
        {
            "report_input_tokens": len(payload),
            "input_policy_facts": policy_facts,
            "processing_decision": decision,
            "invocation": invocation,
        }
    )
    database.connection.execute(
        """INSERT INTO analysis_runs(
               analysis_run_id, run_id, paper_id, artifact_id, input_hash,
               input_scope, model_id, model_revision, prompt_hash, schema_hash,
               implementation_version, policy_version, policy_decision,
               invocation_metadata_json, status, output_artifact_id, completed_at
           ) VALUES (
               'analysis-p1', 'stage4-1', 'p1', 'text-p1', ?, 'full_pdf',
               'gpt-5.6-luna', 'luna-fixture', ?, ?, 'stage4-v1', 'policy-v1',
               'full_pdf', ?, 'complete', 'analysis-output-p1',
               '2026-08-11T00:00:00Z'
           )""",
        (PROMPT_INPUT_HASH, PROMPT_HASH, SCHEMA_HASH, metadata),
    )
    database.connection.execute(
        """INSERT INTO analysis_dispatches(
               dispatch_id, run_id, paper_id, artifact_hash, artifact_id,
               input_scope, config_hash, implementation_version, profile,
               model_id, prompt_hash, schema_hash, policy_version, policy_hash,
               stable_created_at, prompt_input_hash, rendered_prompt_hash,
               processing_decision_json, status, dispatch_count, invocation_id,
               invocation_metadata_json, analysis_run_id, completed_at
           ) VALUES (
               'dispatch-p1', 'stage4-1', 'p1', ?, 'text-p1', 'full_pdf', ?,
               'stage4-v1', 'stage4_analysis_luna', 'gpt-5.6-luna', ?, ?,
               'policy-v1', ?, '2026-08-11T00:00:00Z', ?, ?, ?, 'complete', 1,
               'invocation-p1', ?, 'analysis-p1', '2026-08-11T00:00:00Z'
           )""",
        (
            text.artifact_hash,
            CONFIG_HASH,
            PROMPT_HASH,
            SCHEMA_HASH,
            "2" * 64,
            PROMPT_INPUT_HASH,
            RENDERED_HASH,
            json.dumps(decision),
            json.dumps(invocation),
        ),
    )


def test_builds_canonical_inputs_from_persisted_data_and_keeps_missing_papers(
    tmp_path: Path,
) -> None:
    database, store, service = _fixture(tmp_path)
    try:
        result = service.build(_request())
        papers = {paper["paper_id"]: paper for paper in result.corpus_snapshot["papers"]}

        assert set(papers) == {"p1", "p2"}
        assert papers["p1"]["publication_status"] == "preprint"
        assert papers["p1"]["study_setting"] == "theory"
        assert papers["p1"]["source_category"] == "user_library"
        assert papers["p1"]["foundational"] is True
        assert papers["p1"]["recent"] is False
        assert papers["p1"]["analysis_artifact_hash"]
        assert len(papers["p1"]["lineage_hashes"]) == 2
        assert papers["p1"]["analysis_config_hash"] == CONFIG_HASH
        assert papers["p1"]["analysis_prompt_input_hash"] == PROMPT_INPUT_HASH
        assert papers["p1"]["analysis_rendered_prompt_hash"] == RENDERED_HASH
        assert papers["p1"]["analysis_policy_facts_hash"]

        assert papers["p2"]["source_category"] == "citation_snowball"
        assert papers["p2"]["recent"] is True
        assert papers["p2"]["input_scope"] == "missing"
        assert papers["p2"]["incomplete_reason"] == "stage4_analysis_missing"
        assert papers["p2"]["publication_status"] == "unknown"
        assert papers["p2"]["study_setting"] == "other"
        assert result.search_audit["flow"]["excluded_by_reason"] == {
            "ambiguous": 1,
            "off_topic": 1,
        }
        assert result.search_audit["required_provider_failures"] == []
        assert result.corpus_snapshot_path.read_bytes() == canonical_json(
            dict(result.corpus_snapshot)
        )
        assert result.search_audit_path.read_bytes() == canonical_json(
            dict(result.search_audit)
        )
        assert result.directory.name == result.bundle_id
        assert store.read_bytes(papers["p1"]["analysis_artifact_hash"])
    finally:
        database.close()


def test_dry_run_builds_and_validates_without_writing_bundle(tmp_path: Path) -> None:
    database, _, service = _fixture(tmp_path)
    try:
        result = service.build(_request(), save_bundle=False)

        assert result.saved is False
        assert not result.directory.exists()
        assert result.corpus_snapshot["snapshot_hash"]
        assert result.search_audit["pack_hash"]
    finally:
        database.close()


def test_needs_review_membership_is_explicit_and_changes_the_immutable_bundle(
    tmp_path: Path,
) -> None:
    database, _, service = _fixture(tmp_path)
    try:
        normal = service.build(_request())
        expanded = service.build(_request(include_needs_review=True))
        papers = {paper["paper_id"]: paper for paper in expanded.corpus_snapshot["papers"]}

        assert normal.bundle_id != expanded.bundle_id
        assert set(papers) == {"p1", "p2", "p4"}
        assert papers["p4"]["source_category"] == "newly_discovered"
        assert papers["p4"]["input_scope"] == "missing"
        assert expanded.search_audit["flow"]["excluded_by_reason"] == {"off_topic": 1}
    finally:
        database.close()


def test_existing_bundle_files_are_immutable(tmp_path: Path) -> None:
    database, _, service = _fixture(tmp_path)
    try:
        first = service.build(_request())
        assert service.build(_request()).bundle_id == first.bundle_id
        first.corpus_snapshot_path.write_text("{}", encoding="utf-8")

        with pytest.raises(ReportInputError, match="immutable"):
            service.build(_request())
    finally:
        database.close()


def test_global_unique_count_does_not_double_count_provider_overlap(
    tmp_path: Path,
) -> None:
    database, _, service = _fixture(tmp_path)
    try:
        database.connection.execute(
            """INSERT INTO source_runs(
                   source_run_id, crawl_run_id, provider, provider_version, role, status
               ) VALUES ('source-2', 'crawl-1', 'crossref', '1', 'verify', 'complete')"""
        )
        database.connection.execute(
            """INSERT INTO source_run_audits(
                   source_run_id, raw_discovered, unique_after_dedup, screened,
                   excluded, included, updated_at
               ) VALUES ('source-2', 4, 4, 4, 2, 2, '2026-08-11T00:00:00Z')"""
        )
        database.connection.commit()

        result = service.build(_request())

        assert result.search_audit["flow"]["raw_discovered"] == 8
        assert result.search_audit["flow"]["unique_after_dedup"] == 4
    finally:
        database.close()


def test_crawl_owned_user_seed_is_counted_without_a_provider_snapshot(
    tmp_path: Path,
) -> None:
    database, _, service = _fixture(tmp_path)
    try:
        database.connection.execute(
            """INSERT INTO papers(
                   paper_id, title, authors_json, year, verification_status
               ) VALUES ('p5', 'Explicit seed', '[]', 2025, 'verified')"""
        )
        database.connection.execute(
            """INSERT INTO search_round_seeds(
                   search_round_id, paper_id, seed_reason, parent_round, depth,
                   seed_rank, selector_version, selector_config_hash
               ) VALUES (
                   'round-1', 'p5', 'user_seed', 0, 0, 1, 'selector-v1', ?
               )""",
            ("7" * 64,),
        )
        database.connection.execute(
            """INSERT INTO filter_decisions(
                   filter_decision_id, run_id, paper_id, status, threshold_version,
                   reason, input_hash, implementation_version
               ) VALUES (
                   'decision-p5', 'filter-1', 'p5', 'relevant', 'threshold-v1',
                   '{"reason_code":"topic_match"}', ?, 'stage2-v1'
               )""",
            ("2" * 64,),
        )
        database.connection.commit()

        result = service.build(_request())
        papers = {paper["paper_id"]: paper for paper in result.corpus_snapshot["papers"]}

        assert papers["p5"]["source_category"] == "user_library"
        assert result.search_audit["flow"]["raw_discovered"] == 5
        assert result.search_audit["flow"]["unique_after_dedup"] == 5
        assert result.search_audit["flow"]["stage2_screened"] == 5
    finally:
        database.close()


def test_rejects_a_tampered_persisted_query_plan(tmp_path: Path) -> None:
    database, _, service = _fixture(tmp_path)
    try:
        row = database.connection.execute(
            "SELECT plan_json FROM search_plans WHERE search_plan_id = 'plan-1'"
        ).fetchone()
        plan = json.loads(row[0])
        plan["scope"]["user_seeds"] = ["paper-injected"]
        database.connection.execute(
            "UPDATE search_plans SET plan_json = ? WHERE search_plan_id = 'plan-1'",
            (json.dumps(plan),),
        )
        database.connection.commit()

        with pytest.raises(ReportInputError, match="approval is invalid"):
            service.build(_request())
    finally:
        database.close()


def test_rejects_stage2_decisions_not_bound_to_the_selected_crawl(tmp_path: Path) -> None:
    database, _, service = _fixture(tmp_path)
    try:
        database.connection.execute(
            "INSERT INTO papers(paper_id, title) VALUES ('foreign', 'Foreign')"
        )
        database.connection.execute(
            """INSERT INTO filter_decisions(
                   filter_decision_id, run_id, paper_id, status, threshold_version,
                   reason, input_hash, implementation_version
               ) VALUES (
                   'decision-foreign', 'filter-1', 'foreign', 'relevant',
                   'threshold-v1', '{"reason_code":"topic_match"}', ?, 'stage2-v1'
               )""",
            ("3" * 64,),
        )
        database.connection.commit()

        with pytest.raises(ReportInputError, match="outside the selected crawl"):
            service.build(_request())
    finally:
        database.close()
