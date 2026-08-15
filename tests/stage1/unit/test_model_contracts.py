"""Cross-model contract tests for the frozen Stage 1 JSON values."""

import json

import pytest

from paper_agent import models
from paper_agent.errors import ContractError
from paper_agent.models import (
    AccessStatus,
    FieldSources,
    IssueKind,
    IssueRecord,
    MissingField,
    Pagination,
    PaperRecord,
    RunCounts,
    RunRecord,
    RunStatus,
    SourceTotal,
    SourceTotalScope,
    VenueType,
)


def _sources(doi=True, landing=True, pdf=True):
    return FieldSources("proceedings", "proceedings", "proceedings", "proceedings" if doi else None, "proceedings" if landing else None, "proceedings" if pdf else None)


def _paper(**overrides) -> PaperRecord:
    values = {"venue_id": "example-conf", "venue_name": "Example Conference", "venue_type": VenueType.CONFERENCE, "year": 2024, "source_name": "proceedings", "source_id": "paper-1", "title": "Café — Résumé", "authors": ("Zoë Example", "Café Example"), "abstract": "Résumé for the example paper.", "doi": "10.1234/cafe.1", "landing_url": "https://example.org/papers/1", "pdf_url": "https://example.org/papers/1.pdf", "access_status": AccessStatus.DIRECT_PDF}
    values.update(overrides)
    if "field_sources" not in overrides:
        values["field_sources"] = _sources(values["doi"] is not None, values["landing_url"] is not None, values["pdf_url"] is not None)
    return PaperRecord(**values)


def _issue(**overrides) -> IssueRecord:
    values = {"issue_kind": IssueKind.INCOMPLETE_PAPER, "venue_id": "example-conf", "year": 2024, "source_name": "proceedings", "source_id": "paper-2", "source_locator": "https://example.org/papers/2", "title": "Café — Résumé", "authors": ("Zoë Example",), "abstract": None, "doi": "10.1234/cafe.2", "landing_url": "https://example.org/papers/2", "missing_fields": (MissingField.ABSTRACT,), "reason_codes": ("missing_abstract",), "message": "Authoritative metadata did not provide an abstract."}
    values.update(overrides)
    return IssueRecord(**values)


def _counts(**overrides) -> RunCounts:
    values = {"raw_items": 0, "included_papers": 0, "complete_papers": 0, "incomplete_papers": 0, "excluded_non_papers": 0, "duplicate_occurrences": 0, "parse_rejects": 0, "issue_records": 0}
    values.update(overrides)
    return RunCounts(**values)


def _run(**overrides) -> RunRecord:
    values = {"status": RunStatus.PARTIAL, "venue_id": "example-conf", "venue_name": "Example Conference", "venue_type": VenueType.CONFERENCE, "year": 2024, "source_name": "proceedings", "membership_complete": True, "metadata_complete": False, "complete": False, "counts": _counts(raw_items=2, included_papers=2, complete_papers=1, incomplete_papers=1, issue_records=1), "pagination": Pagination(1, True, SourceTotal(2, SourceTotalScope.INCLUDED_PAPERS)), "warnings": ("partial warning",), "errors": ()}
    values.update(overrides)
    return RunRecord(**values)


def _assert_json_primitives(value) -> None:
    if value is None or type(value) in (str, int, bool):
        return
    if type(value) is list:
        for item in value:
            _assert_json_primitives(item)
        return
    if type(value) is dict:
        for key, item in value.items():
            assert type(key) is str
            _assert_json_primitives(item)
        return
    raise AssertionError(f"non-JSON primitive value: {type(value)!r}")


def test_public_surface_schema_and_complete_snapshots() -> None:
    assert models.__all__ == ["SCHEMA_VERSION", "VenueType", "AccessStatus", "IssueKind", "MissingField", "SourceIdentity", "FieldSources", "PaperRecord", "IssueRecord", "RunStatus", "SourceTotalScope", "RunCounts", "SourceTotal", "Pagination", "RunRecord"]
    assert models.SCHEMA_VERSION == 1 and type(models.SCHEMA_VERSION) is int
    paper, issue, run = _paper(), _issue(), _run()
    expected = [
        {"schema_version": 1, "venue_id": "example-conf", "venue_name": "Example Conference", "venue_type": "conference", "year": 2024, "source_name": "proceedings", "source_id": "paper-1", "title": "Café — Résumé", "authors": ["Zoë Example", "Café Example"], "abstract": "Résumé for the example paper.", "doi": "10.1234/cafe.1", "landing_url": "https://example.org/papers/1", "pdf_url": "https://example.org/papers/1.pdf", "access_status": "direct_pdf", "field_sources": {"title": "proceedings", "authors": "proceedings", "abstract": "proceedings", "doi": "proceedings", "landing_url": "proceedings", "pdf_url": "proceedings"}},
        {"schema_version": 1, "issue_kind": "incomplete_paper", "venue_id": "example-conf", "year": 2024, "source_name": "proceedings", "source_id": "paper-2", "source_locator": "https://example.org/papers/2", "title": "Café — Résumé", "authors": ["Zoë Example"], "abstract": None, "doi": "10.1234/cafe.2", "landing_url": "https://example.org/papers/2", "missing_fields": ["abstract"], "reason_codes": ["missing_abstract"], "message": "Authoritative metadata did not provide an abstract."},
        {"schema_version": 1, "status": "partial", "venue_id": "example-conf", "venue_name": "Example Conference", "venue_type": "conference", "year": 2024, "source_name": "proceedings", "membership_complete": True, "metadata_complete": False, "complete": False, "counts": {"raw_items": 2, "included_papers": 2, "complete_papers": 1, "incomplete_papers": 1, "excluded_non_papers": 0, "duplicate_occurrences": 0, "parse_rejects": 0, "issue_records": 1}, "pagination": {"pages_fetched": 1, "terminal_reached": True, "source_total": {"value": 2, "scope": "included_papers"}}, "warnings": ["partial warning"], "errors": []},
    ]
    for record, expected_dict in zip((paper, issue, run), expected):
        actual = record.to_dict()
        assert actual == expected_dict
        _assert_json_primitives(actual)
        text = json.dumps(actual, ensure_ascii=False, separators=(",", ":"))
        assert json.loads(text) == expected_dict
        if record is paper:
            assert "Café" in text
        if record is issue:
            assert "Résumé" in text
    assert all(record.to_dict()["schema_version"] == 1 for record in (paper, issue, run))
    assert "blocks_membership" not in issue.to_dict() and "blocks_metadata" not in issue.to_dict()


def test_to_dict_returns_independent_containers() -> None:
    paper = _paper()
    first, second = paper.to_dict(), paper.to_dict()
    assert first == second and first is not second and first["authors"] is not second["authors"] and first["field_sources"] is not second["field_sources"]
    first["authors"].append("Injected")
    first["field_sources"]["title"] = "changed"
    assert paper.to_dict() == second
    issue = _issue()
    first, second = issue.to_dict(), issue.to_dict()
    assert all(first[key] is not second[key] for key in ("authors", "missing_fields", "reason_codes"))
    first["authors"].append("Injected")
    first["missing_fields"].append("title")
    first["reason_codes"].append("injected")
    assert issue.to_dict() == second
    run = _run()
    first, second = run.to_dict(), run.to_dict()
    assert first["counts"] is not second["counts"] and first["pagination"] is not second["pagination"] and first["pagination"]["source_total"] is not second["pagination"]["source_total"] and first["warnings"] is not second["warnings"] and first["errors"] is not second["errors"]
    first["counts"]["raw_items"] = 99
    first["pagination"]["source_total"]["value"] = 99
    first["warnings"].append("Injected")
    first["errors"].append("Injected")
    assert run.to_dict() == second


@pytest.mark.parametrize("status,doi,pdf,valid", [(AccessStatus.DIRECT_PDF, None, "https://example.org/paper.pdf", True), (AccessStatus.DIRECT_PDF, "10.1234/example", "https://example.org/paper.pdf", True), (AccessStatus.DIRECT_PDF, None, None, False), (AccessStatus.DOI_ONLY, "10.1234/example", None, True), (AccessStatus.DOI_ONLY, None, None, False), (AccessStatus.DOI_ONLY, "10.1234/example", "https://example.org/paper.pdf", False), (AccessStatus.DOI_ONLY, None, "https://example.org/paper.pdf", False)])
def test_access_truth_table(status, doi, pdf, valid) -> None:
    sources = _sources(doi is not None, True, pdf is not None)
    kwargs = {"access_status": status, "doi": doi, "pdf_url": pdf, "field_sources": sources}
    if valid:
        assert _paper(**kwargs).access_status is status
    else:
        with pytest.raises(ContractError):
            _paper(**kwargs)


@pytest.mark.parametrize("field,kwargs,source_value", [("doi", {}, None), ("doi", {"doi": None}, "proceedings"), ("landing_url", {}, None), ("landing_url", {"landing_url": None}, "proceedings"), ("pdf_url", {}, None), ("pdf_url", {"pdf_url": None, "access_status": AccessStatus.DOI_ONLY}, "proceedings")])
def test_field_sources_nullness_mismatch_is_rejected(field, kwargs, source_value) -> None:
    source_values = {"title": "proceedings", "authors": "proceedings", "abstract": "proceedings", "doi": "proceedings", "landing_url": "proceedings", "pdf_url": "proceedings"}
    source_values[field] = source_value
    if field == "landing_url" and kwargs.get("landing_url") is None:
        assert _paper(landing_url=None).to_dict()["field_sources"]["landing_url"] is None
    with pytest.raises(ContractError):
        _paper(field_sources=FieldSources(**source_values), **kwargs)


@pytest.mark.parametrize("kind,membership,metadata", [(IssueKind.INCOMPLETE_PAPER, False, True), (IssueKind.PARSE_REJECT, True, False), (IssueKind.IDENTITY_CONFLICT, True, False), (IssueKind.FIELD_CONFLICT, False, True)])
def test_issue_completeness_effect_matrix(kind, membership, metadata) -> None:
    issue = _issue(issue_kind=kind, missing_fields=(MissingField.ABSTRACT,) if kind is IssueKind.INCOMPLETE_PAPER else ())
    assert issue.blocks_membership is membership and issue.blocks_metadata is metadata
    assert "blocks_membership" not in issue.to_dict() and "blocks_metadata" not in issue.to_dict()


def test_mixed_counts_keep_only_the_two_contract_equations() -> None:
    counts = _counts(raw_items=10, included_papers=5, complete_papers=3, incomplete_papers=2, excluded_non_papers=2, duplicate_occurrences=1, parse_rejects=2, issue_records=4)
    assert counts.raw_items == counts.included_papers + counts.excluded_non_papers + counts.duplicate_occurrences + counts.parse_rejects
    assert counts.included_papers == counts.complete_papers + counts.incomplete_papers
    assert counts.to_dict() == {"raw_items": 10, "included_papers": 5, "complete_papers": 3, "incomplete_papers": 2, "excluded_non_papers": 2, "duplicate_occurrences": 1, "parse_rejects": 2, "issue_records": 4}


def test_applicable_empty_complete_and_not_applicable_are_distinct() -> None:
    empty = _run(status=RunStatus.COMPLETE, source_name="authoritative-empty", membership_complete=True, metadata_complete=True, complete=True, counts=_counts(), pagination=Pagination(1, True, SourceTotal(0, SourceTotalScope.INCLUDED_PAPERS)), warnings=(), errors=())
    not_applicable = _run(status=RunStatus.NOT_APPLICABLE, source_name=None, membership_complete=False, metadata_complete=False, complete=False, counts=_counts(), pagination=None, warnings=(), errors=())
    assert empty.to_dict()["status"] == "complete" and not_applicable.to_dict()["status"] == "not_applicable"
    assert empty.to_dict()["source_name"] != not_applicable.to_dict()["source_name"] and empty.to_dict()["pagination"] is not None and not_applicable.to_dict()["pagination"] is None


def test_partial_supports_metadata_and_membership_directions() -> None:
    metadata = _run()
    membership = _run(membership_complete=False, metadata_complete=True, complete=False, counts=_counts(raw_items=2, included_papers=2, complete_papers=2), pagination=Pagination(1, False, None), warnings=(), errors=("membership not closed",))
    assert metadata.membership_complete and not metadata.metadata_complete and metadata.counts.incomplete_papers > 0
    assert not membership.membership_complete and membership.metadata_complete and membership.counts.complete_papers > 0 and membership.counts.incomplete_papers == 0


@pytest.mark.parametrize("scope", [SourceTotalScope.RAW_ITEMS, SourceTotalScope.INCLUDED_PAPERS])
def test_source_total_scope_match_mismatch_and_partial_diagnostic(scope) -> None:
    counts = _counts(raw_items=3, included_papers=2, complete_papers=2, excluded_non_papers=1)
    expected = counts.raw_items if scope is SourceTotalScope.RAW_ITEMS else counts.included_papers
    exact = _run(status=RunStatus.COMPLETE, membership_complete=True, metadata_complete=True, complete=True, counts=counts, pagination=Pagination(1, True, SourceTotal(expected, scope)), warnings=(), errors=())
    assert exact.pagination.source_total.scope is scope
    with pytest.raises(ContractError):
        _run(status=RunStatus.COMPLETE, membership_complete=True, metadata_complete=True, complete=True, counts=counts, pagination=Pagination(1, True, SourceTotal(expected + 1, scope)), warnings=(), errors=())
    diagnostic = _run(membership_complete=False, metadata_complete=True, complete=False, counts=counts, pagination=Pagination(1, True, SourceTotal(expected + 1, scope)), warnings=(), errors=("source total mismatch",))
    assert diagnostic.pagination.source_total.value == expected + 1


def test_warning_and_error_order_duplicates_are_preserved() -> None:
    complete = _run(status=RunStatus.COMPLETE, membership_complete=True, metadata_complete=True, complete=True, counts=_counts(raw_items=3, included_papers=3, complete_papers=3), pagination=Pagination(1, True, SourceTotal(3, SourceTotalScope.RAW_ITEMS)), warnings=("first", "same", "same", "last"), errors=())
    failed = _run(status=RunStatus.FAILED, membership_complete=False, metadata_complete=False, complete=False, counts=_counts(raw_items=1, parse_rejects=1, issue_records=1), pagination=Pagination(1, False, None), warnings=(), errors=("first_error", "same_error", "same_error", "last_error"))
    assert complete.warnings == ("first", "same", "same", "last") and complete.to_dict()["warnings"] == ["first", "same", "same", "last"]
    assert failed.errors == ("first_error", "same_error", "same_error", "last_error") and failed.to_dict()["errors"] == ["first_error", "same_error", "same_error", "last_error"]
