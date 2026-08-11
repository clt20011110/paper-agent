from __future__ import annotations

from copy import deepcopy

import pytest

from paper_agent.canonical import content_hash
from paper_agent.report_artifacts import (
    ReportArtifactError,
    ReportArtifactStore,
    ReportVerificationError,
    audit_coverage_ledger,
    audit_rubric_hash,
    render_markdown,
    report_artifact_hash,
    report_diff,
    validate_claim_relations,
    verify_report,
    _resource_tables_document,
)
from paper_agent.reporting import (
    comparison_assessment,
    derive_comparison_groups,
    stable_claim_id,
)


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
            "report_language": "zh-CN",
            "sections": [{"id": "evidence", "title": "证据综合"}],
        },
        "search_audit": {
            "search_status": "complete",
            "required_provider_failures": [],
            "budget_exhausted": False,
            "limitations": [],
        },
        "corpus_snapshot": {"papers": [{
            "paper_id": "p1", "input_scope": "full_pdf",
            "publication_status": "peer_reviewed",
        }]},
        "claims": [claim],
        "coverage": {
            "complete": True, "missing_paper_ids": [], "uncovered_claim_ids": [],
            "papers": [{
                "paper_id": "p1", "evidence_claim_ids": [claim["claim_id"]],
                "consumed_node_ids": ["section:evidence:1"],
                "disposition": "evidence", "reason": None,
            }],
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

    audit = {
        "audit_pass": "A",
        "report_document_hash": content_hash(bundle["document"]),
        "report_artifact_hash": report_artifact_hash(
            document=bundle["document"], claims=bundle["claims"],
            coverage=bundle["coverage"], comparison_groups={}, claim_relations=[],
            bibliography=bundle["bibliography"],
        ),
        "report_plan_hash": content_hash(bundle["plan"]),
        "rubric_hash": audit_rubric_hash(),
        "search_limitations_hash": content_hash([sidecar["search_limitations"][0]]),
        "coverage_complete": True,
        "coverage_ledger": audit_coverage_ledger(bundle["document"], bundle["claims"]),
        "findings": [],
    }
    store = ReportArtifactStore(tmp_path)
    output = store.write(
        plan=bundle["plan"], search_audit=bundle["search_audit"], corpus_snapshot=bundle["corpus_snapshot"],
        claims=bundle["claims"], comparison_groups={}, claim_relations=[], document=bundle["document"],
        coverage=bundle["coverage"], bibliography=bundle["bibliography"], audit=audit,
    )
    assert (output / "REPORT.md").read_text(encoding="utf-8") == (tmp_path / "reports/latest.md").read_text(encoding="utf-8")
    assert (output / "REPORT_SIDECAR.json").is_file()
    assert (output / "BIBLIOGRAPHY.json").is_file()
    assert not (tmp_path / "reports/.report-1.tmp").exists()
    (tmp_path / "reports/latest.md").unlink()
    reconciled = store.reconcile(
        plan=bundle["plan"], search_audit=bundle["search_audit"],
        corpus_snapshot=bundle["corpus_snapshot"], claims=bundle["claims"],
        comparison_groups={}, claim_relations=[], document=bundle["document"],
        coverage=bundle["coverage"], bibliography=bundle["bibliography"], audit=audit,
    )
    assert reconciled == output
    assert (tmp_path / "reports/latest.md").read_text(encoding="utf-8") == (
        output / "REPORT.md"
    ).read_text(encoding="utf-8")
    with pytest.raises(Exception, match="immutable"):
        store.write(
            plan=bundle["plan"], search_audit=bundle["search_audit"], corpus_snapshot=bundle["corpus_snapshot"],
            claims=bundle["claims"], comparison_groups={}, claim_relations=[], document=bundle["document"],
            coverage=bundle["coverage"], bibliography=bundle["bibliography"], audit=audit,
        )
    (output / "COVERAGE.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ReportArtifactError, match="conflicts"):
        store.reconcile(
            plan=bundle["plan"], search_audit=bundle["search_audit"],
            corpus_snapshot=bundle["corpus_snapshot"], claims=bundle["claims"],
            comparison_groups={}, claim_relations=[], document=bundle["document"],
            coverage=bundle["coverage"], bibliography=bundle["bibliography"], audit=audit,
        )


def test_zh_cn_verifier_rejects_an_all_english_report() -> None:
    bundle = _bundle()
    bundle["plan"]["objective"] = "Evidence review"
    bundle["plan"]["sections"][0]["title"] = "Evidence synthesis"
    bundle["claims"][0]["claim_text"] = "The method reaches the measured result."

    with pytest.raises(ReportVerificationError, match="objective must contain CJK"):
        verify_report(
            plan=bundle["plan"], document=bundle["document"], claims=bundle["claims"],
            coverage=bundle["coverage"], bibliography=bundle["bibliography"],
            search_audit=bundle["search_audit"], corpus_snapshot=bundle["corpus_snapshot"],
        )


@pytest.mark.parametrize(
    ("target", "message"),
    (("section", "section title must contain CJK"), ("claim", "claim_text must contain CJK")),
)
def test_zh_cn_verifier_checks_section_titles_and_claim_text(
    target: str, message: str
) -> None:
    bundle = _bundle()
    if target == "section":
        bundle["plan"]["sections"][0]["title"] = "Evidence synthesis"
    else:
        bundle["claims"][0]["claim_text"] = "The method reaches the measured result."

    with pytest.raises(ReportVerificationError, match=message):
        verify_report(
            plan=bundle["plan"], document=bundle["document"], claims=bundle["claims"],
            coverage=bundle["coverage"], bibliography=bundle["bibliography"],
            search_audit=bundle["search_audit"], corpus_snapshot=bundle["corpus_snapshot"],
        )


def test_zh_cn_verifier_rejects_english_prose_despite_chinese_claim_and_caption() -> None:
    bundle = _bundle()
    bundle["document"]["blocks"][0]["text"] = "The method reaches 91%. [@p1]"

    with pytest.raises(ReportVerificationError, match="prose/list_item block must contain CJK"):
        verify_report(
            plan=bundle["plan"], document=bundle["document"], claims=bundle["claims"],
            coverage=bundle["coverage"], bibliography=bundle["bibliography"],
            search_audit=bundle["search_audit"], corpus_snapshot=bundle["corpus_snapshot"],
        )


def test_table_caption_only_claim_requires_one_cjk_block_but_allows_numeric_cells() -> None:
    bundle = _bundle()
    bundle["document"]["blocks"][0].update({
        "block_kind": "table_cell",
        "text": "91 [@p1]",
    })
    bundle["document"]["blocks"][1]["text"] = "Supporting caption [@p1]"

    with pytest.raises(ReportVerificationError, match="table/caption block"):
        verify_report(
            plan=bundle["plan"], document=bundle["document"], claims=bundle["claims"],
            coverage=bundle["coverage"], bibliography=bundle["bibliography"],
            search_audit=bundle["search_audit"], corpus_snapshot=None,
        )

    bundle["document"]["blocks"][1]["text"] = "中文说明 [@p1]"
    verify_report(
        plan=bundle["plan"], document=bundle["document"], claims=bundle["claims"],
        coverage=bundle["coverage"], bibliography=bundle["bibliography"],
        search_audit=bundle["search_audit"], corpus_snapshot=None,
    )


@pytest.mark.parametrize(
    ("publication_status", "disclosure"),
    (
        (
            "preprint",
            "出版状态分层：预印本=1；预印本和研讨会论文与正式同行评审论文分层呈现，不视为同等证据。",
        ),
        (
            "workshop",
            "出版状态分层：研讨会论文=1；预印本和研讨会论文与正式同行评审论文分层呈现，不视为同等证据。",
        ),
    ),
)
def test_preprint_and_workshop_cohorts_require_exact_disclosure_in_body_and_sidecar(
    publication_status: str, disclosure: str
) -> None:
    bundle = _bundle()
    bundle["corpus_snapshot"]["papers"][0]["publication_status"] = publication_status

    with pytest.raises(ReportVerificationError, match="publication-status limitations"):
        verify_report(
            plan=bundle["plan"], document=bundle["document"], claims=bundle["claims"],
            coverage=bundle["coverage"], bibliography=bundle["bibliography"],
            search_audit=bundle["search_audit"], corpus_snapshot=bundle["corpus_snapshot"],
        )

    bundle["document"]["blocks"][1]["text"] += " " + disclosure
    verify_report(
        plan=bundle["plan"], document=bundle["document"], claims=bundle["claims"],
        coverage=bundle["coverage"], bibliography=bundle["bibliography"],
        search_audit=bundle["search_audit"], corpus_snapshot=bundle["corpus_snapshot"],
    )
    markdown, sidecar = render_markdown(
        plan=bundle["plan"], document=bundle["document"], claims=bundle["claims"],
        bibliography=bundle["bibliography"], search_audit=bundle["search_audit"],
        corpus_snapshot=bundle["corpus_snapshot"],
    )

    assert disclosure in markdown
    assert disclosure in sidecar["search_limitations"]


def test_report_directory_rejects_path_traversal(tmp_path) -> None:
    store = ReportArtifactStore(tmp_path)
    for report_run_id in ("../escape", "nested/run", "nested\\run", ".", ".."):
        with pytest.raises(ReportArtifactError, match="safe path"):
            store.directory(report_run_id)


@pytest.mark.parametrize(
    "search_fault",
    (
        {"search_status": "incomplete"},
        {"required_provider_failures": ["openalex"]},
        {"budget_exhausted": True},
    ),
)
@pytest.mark.parametrize("operation", ("write", "reconcile"))
def test_incomplete_search_never_writes_or_restores_latest(
    tmp_path, search_fault, operation
) -> None:
    bundle = _bundle()
    bundle["search_audit"].update(search_fault)
    store = ReportArtifactStore(tmp_path)
    store.latest_path.parent.mkdir(parents=True)
    store.latest_path.write_text("previous report\n", encoding="utf-8")
    arguments = {
        "plan": bundle["plan"],
        "search_audit": bundle["search_audit"],
        "corpus_snapshot": bundle["corpus_snapshot"],
        "claims": bundle["claims"],
        "comparison_groups": {},
        "claim_relations": [],
        "document": bundle["document"],
        "coverage": bundle["coverage"],
        "bibliography": bundle["bibliography"],
        "audit": {},
    }

    with pytest.raises(ReportVerificationError, match="not publication-ready"):
        getattr(store, operation)(**arguments)

    assert not store.directory("report-1").exists()
    assert store.latest_path.read_text(encoding="utf-8") == "previous report\n"


def test_verifier_rejects_arbitrary_or_oversized_comparison_group_values() -> None:
    bundle = _bundle()
    claim = bundle["claims"][0]
    first = deepcopy(claim["supporting_evidence"][0])
    second = deepcopy(first)
    second["evidence_unit"]["value"] = 89.0
    second["evidence_unit"]["source_value"] = 89.0
    group_id = comparison_assessment(first["evidence_unit"]).comparison_group_id
    assert group_id is not None
    claim["claim_type"] = "comparison"
    claim["supporting_evidence"] = [first, second]
    claim["comparison_group_id"] = group_id
    claim["claim_key"]["comparison_group_id"] = group_id
    claim["claim_id"] = stable_claim_id(claim["claim_key"], report_run_id="report-1")
    for block in bundle["document"]["blocks"]:
        block["claim_ids"] = [claim["claim_id"]]
    bundle["coverage"]["papers"][0]["evidence_claim_ids"] = [claim["claim_id"]]
    expected = derive_comparison_groups(bundle["claims"])

    verify_report(
        plan=bundle["plan"],
        document=bundle["document"],
        claims=bundle["claims"],
        coverage=bundle["coverage"],
        bibliography=bundle["bibliography"],
        comparison_groups=expected,
        search_audit=bundle["search_audit"],
        corpus_snapshot=bundle["corpus_snapshot"],
    )
    with pytest.raises(ReportVerificationError, match="deterministic evidence-derived"):
        verify_report(
            plan=bundle["plan"],
            document=bundle["document"],
            claims=bundle["claims"],
            coverage=bundle["coverage"],
            bibliography=bundle["bibliography"],
            comparison_groups={group_id: {"arbitrary": "x" * 2_000_000}},
            search_audit=bundle["search_audit"],
            corpus_snapshot=bundle["corpus_snapshot"],
        )


def test_non_incremental_report_rejects_untyped_claim_relations() -> None:
    bundle = _bundle()
    with pytest.raises(ReportVerificationError, match="non-incremental"):
        verify_report(
            plan=bundle["plan"],
            document=bundle["document"],
            claims=bundle["claims"],
            coverage=bundle["coverage"],
            bibliography=bundle["bibliography"],
            search_audit=bundle["search_audit"],
            corpus_snapshot=bundle["corpus_snapshot"],
            claim_relations=[{"garbage": "x" * 1_000}],
        )


def test_incremental_claim_relations_require_exact_evidence_diff_and_cardinality() -> None:
    before = _claim()
    after = deepcopy(before)
    after["claim_key"]["subject_id"] = "refined-method"
    after["claim_id"] = stable_claim_id(after["claim_key"], report_run_id="report-1")
    after["supporting_evidence"][0]["evidence_unit"]["value"] = 92.0
    after["supporting_evidence"][0]["evidence_unit"]["source_value"] = 92.0
    old_ref = content_hash(before["supporting_evidence"][0])
    new_ref = content_hash(after["supporting_evidence"][0])
    relation = {
        "previous_claim_id": before["claim_id"],
        "current_claim_id": after["claim_id"],
        "relation_type": "refined",
        "reason": "The evidence and claim scope were explicitly refined.",
        "evidence_diff": {
            "added_support": [new_ref],
            "removed_support": [old_ref],
            "added_contradiction": [],
            "removed_contradiction": [],
        },
    }

    assert validate_claim_relations(
        {"claims": [before]}, [after], [relation]
    ) == (relation,)
    forged = deepcopy(relation)
    forged["evidence_diff"]["removed_support"] = []
    with pytest.raises(ReportVerificationError, match="evidence diff"):
        validate_claim_relations({"claims": [before]}, [after], [forged])
    split = deepcopy(relation)
    split["relation_type"] = "split"
    with pytest.raises(ReportVerificationError, match="one-to-many"):
        validate_claim_relations({"claims": [before]}, [after], [split])


def test_resource_table_sidecar_is_deterministic_from_frozen_plan_and_corpus() -> None:
    plan = {
        "report_language": "zh-CN",
        "artifacts": {"resource_tables": ["code-and-data"]},
        "paper_memberships": [
            {
                "paper_id": "p2",
                "coverage_disposition": "resource_or_background_table",
                "resource_table_ids": ["code-and-data"],
            }
        ],
    }
    corpus = {
        "papers": [
            {
                "paper_id": "p2",
                "origin": "newly_discovered",
                "publication_year": 2025,
                "venue_name": "FixtureConf",
                "publication_status": "peer_reviewed",
                "study_setting": "real",
                "input_scope": "abstract_only",
                "evidence_level": "abstract_direct",
            }
        ]
    }

    first = _resource_tables_document(plan, corpus)
    second = _resource_tables_document(deepcopy(plan), deepcopy(corpus))

    assert first == second
    assert first["tables"][0]["table_id"] == "code-and-data"
    assert first["tables"][0]["rows"][0]["paper_id"] == "p2"


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
