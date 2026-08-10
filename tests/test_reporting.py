from __future__ import annotations

from copy import deepcopy

import pytest

from paper_agent.canonical import content_hash
from paper_agent.reporting import (
    AnalysisRecord,
    BudgetExceeded,
    EvidenceValidationError,
    ReportPlanner,
    ReportPlanningError,
    SectionRule,
    SynthesisValidator,
    ValidatedSection,
    build_coverage_ledger,
    comparison_assessment,
    corpus_evidence_allowlist,
    require_parallel_comparison,
    stable_claim_id,
)


def _unit(direction: str, value: float, *, dataset_version: str = "v1") -> dict:
    return {
        "claim": "Measured result",
        "direction": direction,
        "task_id": "classification",
        "dataset_id": "dataset-a",
        "dataset_version": dataset_version,
        "split_id": "test",
        "metric_id": "accuracy",
        "metric_definition_hash": "a" * 64,
        "unit": "percent",
        "optimization_direction": "maximize",
        "value": value,
        "uncertainty": "none reported",
        "statistical_method": "point estimate",
        "protocol_id": "protocol-a",
        "protocol_hash": "b" * 64,
        "sample_size": 100,
        "baseline_id": "baseline-a",
        "baseline_version": "v1",
        "conditions": ["same hardware"],
        "locator": {"kind": "page", "value": "4"},
        "normalization_method": "registry_exact",
        "normalizer_version": "registry-v1",
        "source_value": value,
        "comparison_eligibility": "comparable",
        "missing_fields": [],
    }


def _record(
    paper_id: str,
    tokens: int,
    theme: str,
    unit: dict,
    *,
    input_scope: str = "full_pdf",
) -> AnalysisRecord:
    return AnalysisRecord(
        paper_id=paper_id,
        analysis_run_id=f"analysis-{paper_id}",
        analysis_hash=content_hash({"paper_id": paper_id, "unit": unit}),
        input_scope=input_scope,
        input_tokens=tokens,
        classifications={"theme": (theme,), "publication_status": ("peer_reviewed",)},
        evidence_units=(unit,),
    )


def _plan(*, max_calls: int = 100, max_tokens: int = 100_000, retries: int = 0) -> dict:
    return {
        "classification_axes": ["theme", "publication_status"],
        "sections": [
            {
                "id": "s1",
                "subquestion_ids": ["rq1"],
                "allowed_evidence_levels": ["full_text_direct", "abstract_direct"],
            },
            {
                "id": "s2",
                "subquestion_ids": ["rq2"],
                "allowed_evidence_levels": ["full_text_direct"],
            },
        ],
        "paper_memberships": [
            {"paper_id": "p1", "section_ids": ["s1", "s2"], "primary_section_id": "s1"},
            {"paper_id": "p2", "section_ids": ["s1"], "primary_section_id": "s1"},
            {"paper_id": "p3", "section_ids": ["s1"], "primary_section_id": "s1"},
        ],
        "budget": {
            "max_sol_calls": max_calls,
            "max_input_tokens": max_tokens,
            "max_retries": retries,
            "audit_calls": 2,
            "repair_calls": 1,
        },
    }


def _records() -> tuple[AnalysisRecord, ...]:
    return (
        _record("p2", 6, "alpha", _unit("contradict", 81.0)),
        _record("p1", 6, "alpha", _unit("support", 83.0)),
        _record("p3", 5, "beta", _unit("support", 79.0)),
    )


def _planner(plan: dict | None = None, records: tuple[AnalysisRecord, ...] | None = None) -> ReportPlanner:
    return ReportPlanner(
        plan or _plan(),
        records or _records(),
        max_chunk_input_tokens=10,
        reduce_output_tokens=4,
        audit_input_tokens=20,
        repair_input_tokens=10,
    )


def test_semantic_chunking_precedes_token_chunking_and_tree_is_stable() -> None:
    first = _planner().build()
    second = _planner(records=tuple(reversed(_records()))).build()

    assert first == second
    s1 = [chunk for chunk in first.chunks if chunk.section_id == "s1"]
    assert [chunk.paper_ids for chunk in s1] == [("p1",), ("p2",), ("p3",)]
    assert all(len({value for value in chunk.classification_key}) == len(chunk.classification_key) for chunk in s1)
    assert [node.call_kind for node in first.nodes].count("section_reduce") == 6
    assert [node.call_kind for node in first.nodes].count("cross_section_reduce") == 1
    assert [node.call_kind for node in first.nodes].count("final_reduce") == 1
    assert first.budget.generation_calls == 8
    assert first.budget.worst_case_calls == 11


def test_planner_never_truncates_an_oversize_paper_or_silently_exceeds_budget() -> None:
    records = list(_records())
    records[0] = _record("p2", 11, "alpha", _unit("contradict", 81.0))

    with pytest.raises(BudgetExceeded, match="truncation is forbidden"):
        _planner(records=tuple(records)).build()
    with pytest.raises(BudgetExceeded, match="Sol calls"):
        _planner(_plan(max_calls=10)).build()
    with pytest.raises(BudgetExceeded, match="input tokens"):
        _planner(_plan(max_tokens=1)).build()


def test_planner_requires_exact_corpus_membership_and_worst_case_audit_repair_reserve() -> None:
    plan = _plan()
    plan["paper_memberships"] = plan["paper_memberships"][:-1]
    with pytest.raises(ReportPlanningError, match="membership mismatch"):
        _planner(plan).build()

    plan = _plan()
    plan["budget"]["repair_calls"] = 0
    with pytest.raises(ReportPlanningError, match="two audits, and one repair"):
        _planner(plan).build()


def test_comparison_group_is_stable_and_changes_with_protocol_identity() -> None:
    first = comparison_assessment(_unit("support", 83.0))
    second = comparison_assessment(_unit("contradict", 81.0))
    changed = comparison_assessment(_unit("support", 83.0, dataset_version="v2"))

    assert first.comparison_group_id == second.comparison_group_id
    assert changed.comparison_group_id != first.comparison_group_id
    assert require_parallel_comparison((_unit("support", 83.0), _unit("contradict", 81.0))) == first.comparison_group_id
    with pytest.raises(EvidenceValidationError, match="different comparison groups"):
        require_parallel_comparison((_unit("support", 83.0), _unit("support", 83.0, dataset_version="v2")))


def test_not_comparable_requires_declared_missing_fields_and_cannot_be_ranked() -> None:
    unit = _unit("support", 83.0)
    unit.update({
        "dataset_version": None,
        "comparison_eligibility": "not_comparable",
        "missing_fields": ["dataset_version"],
    })

    assert comparison_assessment(unit).missing_fields == ("dataset_version",)
    with pytest.raises(EvidenceValidationError, match="cannot be ranked"):
        require_parallel_comparison((unit, unit))
    unit["missing_fields"] = []
    with pytest.raises(EvidenceValidationError, match="does not cover"):
        comparison_assessment(unit)


def _sections() -> tuple[SectionRule, ...]:
    return (
        SectionRule("s1", frozenset({"rq1"}), frozenset({"full_text_direct", "abstract_direct"})),
        SectionRule("s2", frozenset({"rq2"}), frozenset({"full_text_direct"})),
    )


def _claim(
    *,
    section_id: str,
    question_id: str,
    support: list[dict],
    contradict: list[dict] | None = None,
    comparison_group_id: str | None = None,
    claim_type: str = "finding",
    status: str = "supported",
    subject: str = "method-a",
) -> dict:
    key = {
        "subject_id": subject,
        "predicate_id": "improves",
        "object_or_scope_id": "task-a",
        "qualifier_context_hash": "c" * 64,
        "comparison_group_id": comparison_group_id,
    }
    return {
        "claim_id": stable_claim_id(key, report_run_id="report-1"),
        "claim_key": key,
        "research_question_id": question_id,
        "report_section": section_id,
        "claim_text": "Method A has evidence under the stated protocol.",
        "claim_type": claim_type,
        "supporting_evidence": support,
        "contradicting_evidence": contradict or [],
        "evidence_level": "full_text_direct",
        "comparison_group_id": comparison_group_id,
        "confidence": "medium",
        "known_limitations": ["One protocol only"],
        "status": status,
        "mapping_status": "mapped",
    }


def _paper_ref(record: AnalysisRecord, unit: dict, *, level: str = "full_text_direct") -> dict:
    return {
        "kind": "paper_evidence",
        "evidence_level": level,
        "paper_id": record.paper_id,
        "analysis_run_id": record.analysis_run_id,
        "evidence_unit": unit,
        "locator": "page 4",
        "search_plan_id": None,
        "source_run_id": None,
        "query_id": None,
        "statistic": None,
        "calculation": None,
    }


def _validator(records: tuple[AnalysisRecord, ...] | None = None) -> SynthesisValidator:
    return SynthesisValidator(
        report_run_id="report-1",
        analyses=records or _records(),
        sections=_sections(),
        memberships={"p1": ("s1", "s2"), "p2": ("s1",), "p3": ("s1",)},
    )


def _valid_s1() -> dict:
    records = {item.paper_id: item for item in _records()}
    group_id = comparison_assessment(records["p1"].evidence_units[0]).comparison_group_id
    claim = _claim(
        section_id="s1",
        question_id="rq1",
        support=[_paper_ref(records["p1"], records["p1"].evidence_units[0])],
        contradict=[_paper_ref(records["p2"], records["p2"].evidence_units[0])],
        comparison_group_id=group_id,
        claim_type="comparison",
        status="mixed",
    )
    return {
        "section_id": "s1",
        "draft": "Evidence is mixed [@p1] [@p2].",
        "claims": [claim],
        "citation_paper_ids": ["p1", "p2"],
        "unresolved_conflicts": ["p1 and p2 disagree under the same protocol"],
    }


def test_section_validation_binds_claim_ids_analysis_units_and_citations() -> None:
    validated = _validator().validate_section(_valid_s1())

    assert validated.section_id == "s1"
    assert validated.citation_paper_ids == ("p1", "p2")

    invented = deepcopy(_valid_s1())
    invented["claims"][0]["supporting_evidence"][0]["evidence_unit"]["value"] = 99.0
    with pytest.raises(EvidenceValidationError, match="not in the bound analysis"):
        _validator().validate_section(invented)

    bad_id = deepcopy(_valid_s1())
    bad_id["claims"][0]["claim_id"] = "00000000-0000-0000-0000-000000000000"
    with pytest.raises(EvidenceValidationError, match="canonical claim_key"):
        _validator().validate_section(bad_id)

    hallucinated_citation = deepcopy(_valid_s1())
    hallucinated_citation["draft"] += " [@unknown]"
    with pytest.raises(EvidenceValidationError, match="non-allowlisted paper marker"):
        _validator().validate_section(hallucinated_citation)

    mixed_shape = deepcopy(_valid_s1())
    mixed_shape["claims"][0]["supporting_evidence"][0]["query_id"] = "query-1"
    with pytest.raises(EvidenceValidationError, match="corpus-only fields"):
        _validator().validate_section(mixed_shape)


def test_corpus_stat_uses_frozen_search_ids_and_recomputed_count() -> None:
    audit = {
        "source_round_audit": {
            "search_plan_id": "search-plan-1",
            "sources": [{"source_run_id": "source-run-1"}],
        },
        "query_manifest": [{"query_id": "query-1"}],
        "flow": {"included": 3, "excluded": 2},
        "source_categories": {"newly_discovered": 3},
        "cohorts": {"recent": 2},
        "publication_status": {"peer_reviewed": 3},
        "input_scope": {"full_pdf": 2, "abstract_only": 1},
    }
    allowlist = corpus_evidence_allowlist(audit)
    assert allowlist.document()["statistics"] == [
        {"statistic": "cohorts.recent", "calculation": "2"},
        {"statistic": "flow.excluded", "calculation": "2"},
        {"statistic": "flow.included", "calculation": "3"},
        {"statistic": "input_scope.abstract_only", "calculation": "1"},
        {"statistic": "input_scope.full_pdf", "calculation": "2"},
        {"statistic": "publication_status.peer_reviewed", "calculation": "3"},
        {"statistic": "source_categories.newly_discovered", "calculation": "3"},
    ]
    reference = {
        "kind": "corpus_evidence",
        "evidence_level": "corpus_stat",
        "paper_id": None,
        "analysis_run_id": None,
        "evidence_unit": None,
        "locator": None,
        "search_plan_id": "search-plan-1",
        "source_run_id": "source-run-1",
        "query_id": "query-1",
        "statistic": "flow.included",
        "calculation": "3",
    }
    claim = _claim(
        section_id="s1",
        question_id="rq1",
        support=[reference],
        claim_type="corpus_stat",
        subject="frozen-corpus",
    )
    claim["evidence_level"] = "corpus_stat"
    document = {
        "section_id": "s1",
        "draft": "The frozen corpus contains three included papers.",
        "claims": [claim],
        "citation_paper_ids": [],
        "unresolved_conflicts": [],
    }
    sections = (
        SectionRule(
            "s1", frozenset({"rq1"}),
            frozenset({"full_text_direct", "abstract_direct", "corpus_stat"}),
        ),
        _sections()[1],
    )
    validator = SynthesisValidator(
        report_run_id="report-1",
        analyses=_records(),
        sections=sections,
        memberships={"p1": ("s1", "s2"), "p2": ("s1",), "p3": ("s1",)},
        corpus_evidence=allowlist,
    )

    assert validator.validate_section(document).claims[0]["claim_type"] == "corpus_stat"
    mixed_shape = deepcopy(document)
    mixed_shape["claims"][0]["supporting_evidence"][0]["paper_id"] = "p1"
    with pytest.raises(EvidenceValidationError, match="paper-only fields"):
        validator.validate_section(mixed_shape)
    tampered = deepcopy(document)
    tampered["claims"][0]["supporting_evidence"][0]["calculation"] = "999"
    with pytest.raises(EvidenceValidationError, match="frozen search audit"):
        validator.validate_section(tampered)


def test_abstract_evidence_cannot_be_labeled_as_full_text() -> None:
    abstract_unit = _unit("support", 83.0)
    abstract_record = _record("p1", 6, "alpha", abstract_unit, input_scope="abstract_only")
    records = (abstract_record, _records()[0], _records()[2])

    with pytest.raises(EvidenceValidationError, match="overstates its analysis input scope"):
        _validator(records).validate_section(_valid_s1())


def _valid_s2() -> dict:
    record = next(item for item in _records() if item.paper_id == "p1")
    claim = _claim(
        section_id="s2",
        question_id="rq2",
        support=[_paper_ref(record, record.evidence_units[0])],
        subject="method-a-resource",
    )
    return {
        "section_id": "s2",
        "draft": "The method is relevant [@p1].",
        "claims": [claim],
        "citation_paper_ids": ["p1"],
        "unresolved_conflicts": [],
    }


def test_cross_section_validation_preserves_every_claim_citation_and_conflict() -> None:
    validator = _validator()
    first = validator.validate_section(_valid_s1())
    second = validator.validate_section(_valid_s2())
    document = {
        "section_ids": ["s1", "s2"],
        "draft": "Combined evidence [@p1] [@p2].",
        "claims": [*first.claims, *second.claims],
        "citation_paper_ids": ["p1", "p2"],
        "unresolved_conflicts": list(first.unresolved_conflicts),
    }

    validated = validator.validate_cross_section(document, (second, first))
    assert len(validated.claims) == 2

    dropped = deepcopy(document)
    dropped["claims"] = dropped["claims"][:-1]
    with pytest.raises(EvidenceValidationError, match="added or dropped claims"):
        validator.validate_cross_section(dropped, (first, second))
    erased = deepcopy(document)
    erased["unresolved_conflicts"] = []
    with pytest.raises(EvidenceValidationError, match="erased an unresolved conflict"):
        validator.validate_cross_section(erased, (first, second))


def test_coverage_counts_each_paper_once_and_chunk_consumption_alone_is_not_coverage() -> None:
    reduce_plan = _planner().build()
    first = _validator().validate_section(_valid_s1())
    second = _validator().validate_section(_valid_s2())

    incomplete = build_coverage_ledger(("p1", "p2", "p3"), (first, second), reduce_plan.chunks)
    assert incomplete.missing_paper_ids == ("p3",)
    assert len(incomplete.papers) == 3
    assert len(next(item for item in incomplete.papers if item.paper_id == "p1").consumed_node_ids) == 2
    with pytest.raises(EvidenceValidationError, match="coverage incomplete"):
        incomplete.require_complete()

    complete = build_coverage_ledger(
        ("p1", "p2", "p3"),
        (first, second),
        reduce_plan.chunks,
        background_only={"p3": "Included only to describe historical context"},
    )
    complete.require_complete()
    assert complete.complete


def test_claim_wording_does_not_change_stable_id_but_run_scoped_unmapped_ids_do() -> None:
    key = {
        "subject_id": "subject",
        "predicate_id": "predicate",
        "object_or_scope_id": "scope",
        "qualifier_context_hash": "d" * 64,
        "comparison_group_id": None,
    }

    assert stable_claim_id(key, report_run_id="one") == stable_claim_id(key, report_run_id="two")
    assert stable_claim_id(key, report_run_id="one", mapping_status="unmapped_new") != stable_claim_id(
        key, report_run_id="two", mapping_status="unmapped_new"
    )
