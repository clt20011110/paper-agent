"""Offline unit tests for the standalone Stage 1 contract models."""

from dataclasses import FrozenInstanceError
from unicodedata import normalize

import pytest

from paper_agent_next.errors import ContractError
from paper_agent_next.models import (
    AccessStatus,
    FieldSources,
    IssueKind,
    IssueRecord,
    MissingField,
    PaperRecord,
    SourceIdentity,
    VenueType,
)


def _paper(
    *,
    venue_type=VenueType.CONFERENCE,
    year=2024,
    title="A normalized title",
    authors=("Alice Example", "Bob Example", "Alice Example"),
    abstract="A normalized abstract.",
    doi=None,
    landing_url="https://example.org/paper/1",
    pdf_url="https://example.org/paper/1.pdf",
    access_status=AccessStatus.DIRECT_PDF,
    field_sources=None,
    **overrides,
):
    if field_sources is None:
        field_sources = FieldSources(
            "proceedings",
            "proceedings",
            "proceedings",
            "proceedings" if doi is not None else None,
            "proceedings" if landing_url is not None else None,
            "proceedings" if pdf_url is not None else None,
        )
    values = {
        "venue_id": "example-conf",
        "venue_name": "Example Conference",
        "venue_type": venue_type,
        "year": year,
        "source_name": "proceedings",
        "source_id": "paper-1",
        "title": title,
        "authors": authors,
        "abstract": abstract,
        "doi": doi,
        "landing_url": landing_url,
        "pdf_url": pdf_url,
        "access_status": access_status,
        "field_sources": field_sources,
    }
    values.update(overrides)
    return PaperRecord(**values)


def _issue(
    *,
    issue_kind=IssueKind.INCOMPLETE_PAPER,
    source_name="proceedings",
    source_id="paper-1",
    source_locator="https://example.org/paper/1",
    title="A normalized title",
    authors=("Alice Example",),
    abstract="A normalized abstract.",
    doi="10.1234/example.1",
    landing_url="https://example.org/paper/1",
    missing_fields=(MissingField.ABSTRACT,),
    reason_codes=("missing_abstract",),
    message="The abstract is missing.",
    **overrides,
):
    values = {
        "issue_kind": issue_kind,
        "venue_id": "example-conf",
        "year": 2024,
        "source_name": source_name,
        "source_id": source_id,
        "source_locator": source_locator,
        "title": title,
        "authors": authors,
        "abstract": abstract,
        "doi": doi,
        "landing_url": landing_url,
        "missing_fields": missing_fields,
        "reason_codes": reason_codes,
        "message": message,
    }
    values.update(overrides)
    return IssueRecord(**values)


def test_source_identity_validates_and_preserves_tuple_order() -> None:
    identity = SourceIdentity("example-conf", 2024, "proceedings", "paper-1")

    assert identity.as_tuple() == ("example-conf", 2024, "proceedings", "paper-1")
    assert not hasattr(identity, "__dict__")
    with pytest.raises(FrozenInstanceError):
        identity.year = 2025


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("venue_id", ""),
        ("venue_id", " example-conf"),
        ("source_name", "proceedings "),
        ("source_id", " "),
    ],
)
def test_source_identity_rejects_empty_or_surrounding_whitespace(field, value) -> None:
    values = {"venue_id": "example-conf", "year": 2024, "source_name": "proceedings", "source_id": "paper-1"}
    values[field] = value
    with pytest.raises(ContractError, match=field):
        SourceIdentity(**values)


@pytest.mark.parametrize("year", [True, False, 999, 10000, "2024"])
def test_source_identity_rejects_invalid_year(year) -> None:
    with pytest.raises(ContractError, match="year"):
        SourceIdentity("example-conf", year, "proceedings", "paper-1")


def test_field_sources_has_only_fixed_keys_and_nullable_markers() -> None:
    sources = FieldSources("proceedings", "crossref", "proceedings", None, "crossref", None)

    assert list(sources.to_dict()) == ["title", "authors", "abstract", "doi", "landing_url", "pdf_url"]
    assert sources.to_dict() == {
        "title": "proceedings",
        "authors": "crossref",
        "abstract": "proceedings",
        "doi": None,
        "landing_url": "crossref",
        "pdf_url": None,
    }
    assert not hasattr(sources, "__dict__")
    with pytest.raises(FrozenInstanceError):
        sources.title = "other"


@pytest.mark.parametrize("value", ["", " source", "source ", " "])
def test_field_sources_rejects_empty_source_markers(value) -> None:
    with pytest.raises(ContractError):
        FieldSources(value, "source", "source", None, None, None)


def test_paper_record_supports_both_access_modes() -> None:
    direct_pdf = _paper()
    direct_pdf_with_doi = _paper(doi="10.1234/example.1")
    doi_only = _paper(
        doi="10.1234/example.1",
        pdf_url=None,
        access_status=AccessStatus.DOI_ONLY,
    )

    assert direct_pdf.doi is None
    assert direct_pdf.access_status is AccessStatus.DIRECT_PDF
    assert direct_pdf_with_doi.access_status is AccessStatus.DIRECT_PDF
    assert doi_only.pdf_url is None
    assert doi_only.access_status is AccessStatus.DOI_ONLY
    assert direct_pdf.authors == ("Alice Example", "Bob Example", "Alice Example")
    assert direct_pdf.identity.as_tuple() == (
        "example-conf",
        2024,
        "proceedings",
        "paper-1",
    )


def test_paper_record_serializes_exact_schema_and_nested_sources() -> None:
    record = _paper(doi="10.1234/example.1")
    serialized = record.to_dict()

    assert list(serialized) == [
        "schema_version",
        "venue_id",
        "venue_name",
        "venue_type",
        "year",
        "source_name",
        "source_id",
        "title",
        "authors",
        "abstract",
        "doi",
        "landing_url",
        "pdf_url",
        "access_status",
        "field_sources",
    ]
    assert serialized["schema_version"] == 1
    assert serialized["venue_type"] == "conference"
    assert serialized["access_status"] == "direct_pdf"
    assert serialized["authors"] == ["Alice Example", "Bob Example", "Alice Example"]
    assert list(serialized["field_sources"]) == [
        "title",
        "authors",
        "abstract",
        "doi",
        "landing_url",
        "pdf_url",
    ]
    assert serialized["field_sources"]["doi"] == "proceedings"
    assert not hasattr(record, "__dict__")
    with pytest.raises(FrozenInstanceError):
        record.title = "other"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"title": ""},
        {"abstract": ""},
        {"authors": ()},
        {"authors": ("",)},
        {"authors": ["Alice Example"]},
        {"title": normalize("NFD", "Café")},
        {"authors": (normalize("NFD", "Café"),)},
        {"abstract": normalize("NFD", "Résumé")},
        {"doi": "10.1234/ABC"},
        {"doi": "https://doi.org/10.1234/example.1"},
        {"doi": "10.1234/example 1"},
        {"doi": "not-a-doi"},
        {"landing_url": "/paper/1"},
        {"landing_url": "https://user@example.org/paper/1"},
        {"landing_url": "https://user:password@example.org/paper/1"},
    ],
)
def test_paper_record_rejects_invalid_normalization_and_types(kwargs) -> None:
    with pytest.raises(ContractError):
        _paper(**kwargs)


def test_paper_record_rejects_silent_title_normalization() -> None:
    with pytest.raises(ContractError, match="title"):
        _paper(title=" Title ")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"pdf_url": None},
        {"pdf_url": "https://example.org/paper/1.pdf", "access_status": AccessStatus.DOI_ONLY},
        {"doi": None, "pdf_url": None, "access_status": AccessStatus.DOI_ONLY},
        {"doi": None, "pdf_url": None},
        {"pdf_url": "https://example.org/paper/1.pdf", "access_status": AccessStatus.DOI_ONLY},
        {"doi": "10.1234/example.1", "field_sources": FieldSources("source", "source", "source", None, "source", "source")},
        {"landing_url": None, "field_sources": FieldSources("source", "source", "source", None, "source", "source")},
        {"field_sources": {"title": "source"}},
        {"venue_type": "conference"},
        {"access_status": "direct_pdf"},
    ],
)
def test_paper_record_rejects_access_and_source_invariant_violations(kwargs) -> None:
    with pytest.raises(ContractError):
        _paper(**kwargs)


@pytest.mark.parametrize(
    ("issue_kind", "missing_fields", "blocks_membership", "blocks_metadata"),
    [
        (IssueKind.INCOMPLETE_PAPER, (MissingField.TITLE,), False, True),
        (IssueKind.PARSE_REJECT, (), True, False),
        (IssueKind.IDENTITY_CONFLICT, (), True, False),
        (IssueKind.FIELD_CONFLICT, (), False, True),
    ],
)
def test_issue_record_effects_and_valid_shapes(
    issue_kind, missing_fields, blocks_membership, blocks_metadata
) -> None:
    issue = _issue(
        issue_kind=issue_kind,
        missing_fields=missing_fields,
        authors=() if issue_kind is IssueKind.PARSE_REJECT else ("Alice Example",),
        title=None if issue_kind is IssueKind.PARSE_REJECT else "A normalized title",
        abstract=None if issue_kind is IssueKind.PARSE_REJECT else "A normalized abstract.",
        doi=None if issue_kind is IssueKind.PARSE_REJECT else "10.1234/example.1",
        landing_url=None if issue_kind is IssueKind.PARSE_REJECT else "https://example.org/paper/1",
        source_name=None if issue_kind is IssueKind.PARSE_REJECT else "proceedings",
        source_id=None if issue_kind is IssueKind.PARSE_REJECT else "paper-1",
        source_locator=None if issue_kind is IssueKind.PARSE_REJECT else "https://example.org/paper/1",
    )

    assert issue.blocks_membership is blocks_membership
    assert issue.blocks_metadata is blocks_metadata
    assert not hasattr(issue, "__dict__")
    with pytest.raises(FrozenInstanceError):
        issue.message = "other"


def test_issue_record_serializes_exact_schema() -> None:
    issue = _issue()

    assert list(issue.to_dict()) == [
        "schema_version",
        "issue_kind",
        "venue_id",
        "year",
        "source_name",
        "source_id",
        "source_locator",
        "title",
        "authors",
        "abstract",
        "doi",
        "landing_url",
        "missing_fields",
        "reason_codes",
        "message",
    ]
    assert issue.to_dict()["issue_kind"] == "incomplete_paper"
    assert issue.to_dict()["missing_fields"] == ["abstract"]
    assert issue.to_dict()["reason_codes"] == ["missing_abstract"]


def test_parse_reject_allows_empty_optional_fields_and_authors() -> None:
    issue = _issue(
        issue_kind=IssueKind.PARSE_REJECT,
        source_name=None,
        source_id=None,
        source_locator=None,
        title=None,
        authors=(),
        abstract=None,
        doi=None,
        landing_url=None,
        missing_fields=(),
        reason_codes=("parse_rejected",),
    )

    assert issue.authors == ()
    assert issue.to_dict()["source_name"] is None


def test_issue_record_rejects_string_missing_field() -> None:
    with pytest.raises(ContractError, match="missing_fields"):
        _issue(missing_fields=("title",))


def test_paper_helper_does_not_replace_falsey_invalid_field_sources() -> None:
    with pytest.raises(ContractError, match="field_sources"):
        _paper(field_sources={})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reason_codes": ()},
        {"reason_codes": ("Bad-Code",)},
        {"reason_codes": ("bad__code",)},
        {"reason_codes": ("same", "same")},
        {"missing_fields": (MissingField.TITLE, MissingField.TITLE)},
        {"missing_fields": [MissingField.TITLE]},
        {"reason_codes": ["missing_title"]},
        {"issue_kind": "incomplete_paper"},
        {"authors": ["Alice Example"]},
        {"title": " "},
        {"title": normalize("NFD", "Café")},
        {"authors": (normalize("NFD", "Café"),)},
        {"abstract": normalize("NFD", "Résumé")},
        {"doi": "10.1234/ABC"},
        {"landing_url": "relative/path"},
        {"source_id": "paper-1", "source_name": None},
        {"issue_kind": IssueKind.INCOMPLETE_PAPER, "missing_fields": ()},
    ],
)
def test_issue_record_rejects_invalid_values(kwargs) -> None:
    with pytest.raises(ContractError):
        _issue(**kwargs)
