from __future__ import annotations

from copy import deepcopy

import pytest

from paper_agent.canonical import content_hash
from paper_agent.report_artifacts import (
    ReportArtifactStore,
    ReportVerificationError,
    render_markdown,
    report_diff,
    verify_report,
)
from paper_agent.reporting import stable_claim_id


def _unit() -> dict:
    return {
        "claim": "Measured result", "direction": "support", "task_id": "classification",
        "dataset_id": "dataset", "dataset_version": "v1", "split_id": "test",
        "metric_id": "accuracy", "metric_definition_hash": "a" * 64,
        "unit": "percent", "optimization_direction": "maximize", "value": 91.0,
        "uncertainty": "not reported", "statistical_method": "point estimate",
        "protocol_id": "protocol", "protocol_hash": "b" * 64, "sample_size": 10,
        "baseline_id": "baseline", "baseline_version": "v1", "conditions": ["same split"],
        "locator": {"kind": "page", "value": "4"}, "normalization_method": "exact",
        "normalizer_version": "v1", "source_value": 91.0,
        "comparison_eligibility": "comparable", "missing_fields": [],
    }


def _claim() -> dict:
    key = {
        "subject_id": "method", "predicate_id": "improves", "object_or_scope_id": "task",
        "qualifier_context_hash": "c" * 64, "comparison_group_id": None,
    }
    return {
        "claim_id": stable_claim_id(key, report_run_id="report-1"), "claim_key": key,
        "research_question_id": "rq1", "report_section": "evidence", "claim_text": "方法在指定条件下达到 91%。",
        "claim_type": "finding", "supporting_evidence": [{
            "kind": "paper_evidence", "evidence_level": "full_text_direct", "paper_id": "p1",
            "analysis_run_id": "analysis-p1", "evidence_unit": _unit(), "locator": "page 4",
            "search_plan_id": None, "source_run_id": None, "query_id": None,
            "statistic": None, "calculation": None,
        }],
        "contradicting_evidence": [], "evidence_level": "full_text_direct",
        "comparison_group_id": None, "confidence": "medium", "known_limitations": ["单一数据集"],
        "status": "supported", "mapping_status": "mapped",
    }


def _bundle() -> dict:
    claim = _claim()
    disclosure = "抽取范围：full_pdf=1；全文、摘要和元数据证据已分层，缺失全文不作全文事实表述。"
    return {
        "plan": {
            "report_run_id": "report-1", "objective": "测试领域综述",
            "sections": [{"id": "evidence", "title": "证据综合"}],
        },
        "search_audit": {"limitations": []},
        "corpus_snapshot": {"papers": [{"paper_id": "p1", "input_scope": "full_pdf"}]},
        "claims": [claim],
        "coverage": {
            "complete": True, "missing_paper_ids": [], "uncovered_claim_ids": [],
            "papers": [{"paper_id": "p1", "disposition": "evidence"}],
        },
        "bibliography": {
            "p1": {"title": "A Paper", "authors": ["Ada Lovelace"], "year": 2026,
                   "venue_name": "TestConf", "doi": "10.1000/test"},
        },
        "document": {"report_run_id": "report-1", "blocks": [
            {"block_id": "b1", "block_kind": "prose", "section_id": "evidence",
             "text": "方法在指定条件下达到 91%。[@p1]", "claim_ids": [claim["claim_id"]],
             "citation_paper_ids": ["p1"]},
            {"block_id": "b2", "block_kind": "caption", "section_id": "evidence",
             "text": disclosure + " [@p1]", "claim_ids": [claim["claim_id"]],
             "citation_paper_ids": ["p1"]},
        ]},
    }


def test_verifier_renderer_and_immutable_publish(tmp_path) -> None:
    bundle = _bundle()
    checklist = verify_report(
        plan=bundle["plan"], document=bundle["document"], claims=bundle["claims"],
        coverage=bundle["coverage"], bibliography=bundle["bibliography"],
        search_audit=bundle["search_audit"], corpus_snapshot=bundle["corpus_snapshot"],
    )
    assert checklist["checks"]["no_fabricated_statistics"]
    markdown, sidecar = render_markdown(
        plan=bundle["plan"], document=bundle["document"], claims=bundle["claims"], bibliography=bundle["bibliography"],
        search_audit=bundle["search_audit"], corpus_snapshot=bundle["corpus_snapshot"],
    )
    assert "# 测试领域综述" in markdown
    assert sidecar["blocks"][0]["claim_ids"] == [bundle["claims"][0]["claim_id"]]

    audit = {"report_document_hash": content_hash(bundle["document"]), "coverage_complete": True, "findings": []}
    store = ReportArtifactStore(tmp_path)
    output = store.write(
        plan=bundle["plan"], search_audit=bundle["search_audit"], corpus_snapshot=bundle["corpus_snapshot"],
        claims=bundle["claims"], comparison_groups={}, claim_relations=[], document=bundle["document"],
        coverage=bundle["coverage"], bibliography=bundle["bibliography"], audit=audit,
    )
    assert (output / "REPORT.md").read_text(encoding="utf-8") == (tmp_path / "reports/latest.md").read_text(encoding="utf-8")
    assert (output / "REPORT_SIDECAR.json").is_file()
    assert not (tmp_path / "reports/.report-1.tmp").exists()
    with pytest.raises(Exception, match="immutable"):
        store.write(
            plan=bundle["plan"], search_audit=bundle["search_audit"], corpus_snapshot=bundle["corpus_snapshot"],
            claims=bundle["claims"], comparison_groups={}, claim_relations=[], document=bundle["document"],
            coverage=bundle["coverage"], bibliography=bundle["bibliography"], audit=audit,
        )


def test_verifier_rejects_missing_claim_citation_limitation_and_ungrounded_number() -> None:
    bundle = _bundle()
    missing_citation = deepcopy(bundle)
    missing_citation["document"]["blocks"][0]["citation_paper_ids"] = []
    with pytest.raises(ReportVerificationError):
        verify_report(
            plan=missing_citation["plan"], document=missing_citation["document"], claims=missing_citation["claims"],
            coverage=missing_citation["coverage"], bibliography=missing_citation["bibliography"],
            corpus_snapshot=missing_citation["corpus_snapshot"],
        )

    missing_disclosure = deepcopy(bundle)
    missing_disclosure["document"]["blocks"] = missing_disclosure["document"]["blocks"][:1]
    with pytest.raises(ReportVerificationError, match="limitations"):
        verify_report(
            plan=missing_disclosure["plan"], document=missing_disclosure["document"], claims=missing_disclosure["claims"],
            coverage=missing_disclosure["coverage"], bibliography=missing_disclosure["bibliography"],
            corpus_snapshot=missing_disclosure["corpus_snapshot"],
        )

    ungrounded = deepcopy(bundle)
    ungrounded["claims"][0]["supporting_evidence"][0]["evidence_unit"]["value"] = "not numeric"
    with pytest.raises(ReportVerificationError, match="numeric"):
        verify_report(
            plan=ungrounded["plan"], document=ungrounded["document"], claims=ungrounded["claims"],
            coverage=ungrounded["coverage"], bibliography=ungrounded["bibliography"],
            corpus_snapshot=ungrounded["corpus_snapshot"],
        )


def test_verifier_rejects_corpus_coverage_scope_and_erased_conflicts() -> None:
    bundle = _bundle()
    missing_coverage = deepcopy(bundle)
    missing_coverage["coverage"]["papers"] = []
    with pytest.raises(ReportVerificationError, match="frozen corpus"):
        verify_report(
            plan=missing_coverage["plan"], document=missing_coverage["document"],
            claims=missing_coverage["claims"], coverage=missing_coverage["coverage"],
            bibliography=missing_coverage["bibliography"], corpus_snapshot=missing_coverage["corpus_snapshot"],
        )

    overstated = deepcopy(bundle)
    overstated["corpus_snapshot"]["papers"][0]["input_scope"] = "abstract_only"
    with pytest.raises(ReportVerificationError, match="overstates"):
        verify_report(
            plan=overstated["plan"], document=overstated["document"], claims=overstated["claims"],
            coverage=overstated["coverage"], bibliography=overstated["bibliography"],
            corpus_snapshot=overstated["corpus_snapshot"],
        )

    erased = deepcopy(bundle)
    opposite = deepcopy(erased["claims"][0]["supporting_evidence"][0])
    opposite["evidence_unit"]["direction"] = "contradict"
    erased["claims"][0]["contradicting_evidence"] = [opposite]
    erased["claims"][0]["status"] = "mixed"
    with pytest.raises(ReportVerificationError, match="erases"):
        verify_report(
            plan=erased["plan"], document=erased["document"], claims=erased["claims"],
            coverage=erased["coverage"], bibliography=erased["bibliography"],
            corpus_snapshot=erased["corpus_snapshot"],
        )


def test_incremental_diff_uses_stable_claim_ids_and_explicit_relations() -> None:
    bundle = _bundle()
    previous = {"plan": bundle["plan"], "claims": [], "corpus_snapshot": {"papers": []}}
    current = {"plan": bundle["plan"], "claims": bundle["claims"], "corpus_snapshot": bundle["corpus_snapshot"]}
    diff = report_diff(previous, current)
    assert diff["added_paper_ids"] == ["p1"]
    assert diff["added_claim_ids"] == [bundle["claims"][0]["claim_id"]]
    assert diff["unmapped_claim_ids"] == [bundle["claims"][0]["claim_id"]]
    assert diff["evidence_diff"] == {}


def test_diff_tracks_publication_evidence_and_unchanged_sections() -> None:
    bundle = _bundle()
    previous_claim = deepcopy(bundle["claims"][0])
    current_claim = deepcopy(previous_claim)
    current_claim["known_limitations"].append("新限制")
    previous = {
        "plan": {**bundle["plan"], "sections": [
            {"id": "evidence", "title": "证据综合"},
            {"id": "limits", "title": "限制"},
        ]},
        "claims": [previous_claim],
        "corpus_snapshot": {"papers": [{"paper_id": "p1", "publication_status": "preprint"}]},
    }
    current = deepcopy(previous)
    current["claims"] = [current_claim]
    current["corpus_snapshot"]["papers"][0]["publication_status"] = "peer_reviewed"

    diff = report_diff(previous, current)

    assert diff["changed_claim_ids"] == [current_claim["claim_id"]]
    assert current_claim["claim_id"] in diff["evidence_diff"]
    assert diff["publication_or_retraction_changes"][0]["paper_id"] == "p1"
    assert diff["unchanged_sections"] == ["limits"]
