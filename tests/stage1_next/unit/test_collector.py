"""Offline tests for the minimal in-memory collector composition."""

from email.message import Message
from pathlib import Path

import pytest

from paper_agent_next import http as http_module
from paper_agent_next.adapters.base import CollectedPaper, CollectionResult, ParseReject
from paper_agent_next.adapters.pmlr import PmlrAdapter
from paper_agent_next.catalog import load_venue_spec
from paper_agent_next.collector import CollectionOutcome, collect_venue_year
from paper_agent_next.errors import CollectionError, InputError
from paper_agent_next.http import HttpClient, PrefixResponse
from paper_agent_next.models import (
    AccessStatus,
    IssueKind,
    MissingField,
    Pagination,
    RunStatus,
    SourceTotal,
    SourceTotalScope,
)


FIXTURES = Path(__file__).parents[1] / "fixtures" / "pmlr"
VOLUME_URL = "https://proceedings.mlr.press/v235/"
ADA_URL = "https://proceedings.mlr.press/v235/lovelace24a.html"
TURING_URL = "https://proceedings.mlr.press/v235/turing24a.html"
RAW_ADA_PDF = "https://raw.githubusercontent.com/mlresearch/v235/main/assets/lovelace24a/lovelace24a.pdf"


class FakeAdapter:
    source_name = "authoritative"

    def __init__(self, result=None, error: BaseException | None = None) -> None:
        self.result = result
        self.error = error
        self.calls: list[tuple[object, int, object]] = []

    def collect(self, venue_spec, year: int, http_client):
        self.calls.append((venue_spec, year, http_client))
        if self.error is not None:
            raise self.error
        return self.result


class PrefixClient:
    def __init__(self, responses: dict[str, PrefixResponse | BaseException]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, int]] = []

    def get_prefix(self, url: str, max_bytes: int) -> PrefixResponse:
        self.calls.append((url, max_bytes))
        response = self.responses[url]
        if isinstance(response, BaseException):
            raise response
        return response


class FixtureResponse:
    def __init__(self, body: bytes, content_type: str) -> None:
        self.body = body
        self.headers = Message()
        self.headers["Content-Type"] = content_type
        self.closed = False
        self.read_limits: list[int | None] = []

    def read(self, max_bytes: int | None = None) -> bytes:
        self.read_limits.append(max_bytes)
        return self.body if max_bytes is None else self.body[:max_bytes]

    def close(self) -> None:
        self.closed = True


class PmlrFixtureOpener:
    def __init__(self, responses: dict[str, bytes | BaseException]) -> None:
        self.responses = responses
        self.calls: list[tuple[object, float]] = []
        self.responses_seen: list[FixtureResponse] = []

    def __call__(self, request, *, timeout: float) -> FixtureResponse:
        self.calls.append((request, timeout))
        url = request.full_url
        if url not in self.responses:
            raise AssertionError(f"unexpected offline URL: {url}")
        response = self.responses[url]
        if isinstance(response, BaseException):
            raise response
        content_type = "application/pdf" if url == RAW_ADA_PDF else "text/html; charset=utf-8"
        fixture_response = FixtureResponse(response, content_type)
        self.responses_seen.append(fixture_response)
        return fixture_response


def _paper(source_id: str = "paper-1", **overrides) -> CollectedPaper:
    values = {
        "source_id": source_id,
        "title": "A paper title",
        "authors": ("Ada Lovelace",),
        "abstract": "A paper abstract.",
        "doi": None,
        "landing_url": f"https://example.test/{source_id}",
        "pdf_candidates": (f"https://example.test/{source_id}.pdf",),
    }
    values.update(overrides)
    return CollectedPaper(**values)


def _result(
    papers: tuple[CollectedPaper, ...] = (),
    *,
    excluded_non_papers: int = 0,
    duplicate_occurrences: int = 0,
    parse_rejects: tuple[ParseReject, ...] = (),
    raw_items: int | None = None,
    pagination: Pagination | None = None,
) -> CollectionResult:
    if raw_items is None:
        raw_items = len(papers) + excluded_non_papers + duplicate_occurrences + len(parse_rejects)
    if pagination is None:
        pagination = Pagination(1, True, None)
    return CollectionResult(
        source_name="authoritative",
        papers=papers,
        raw_items=raw_items,
        excluded_non_papers=excluded_non_papers,
        duplicate_occurrences=duplicate_occurrences,
        parse_rejects=parse_rejects,
        pagination=pagination,
    )


def _pdf_client(*urls: str) -> PrefixClient:
    return PrefixClient(
        {
            url: PrefixResponse("application/pdf", b"%PDF-1.7\nbody")
            for url in urls
        }
    )


def _collect(result: CollectionResult, client: PrefixClient | None = None) -> CollectionOutcome:
    return collect_venue_year(
        load_venue_spec("icml"),
        2024,
        FakeAdapter(result=result),
        PrefixClient({}) if client is None else client,
    )


def test_not_applicable_is_exact_and_does_not_call_adapter_or_http() -> None:
    adapter = FakeAdapter(result=_result())
    client = PrefixClient({})

    outcome = collect_venue_year(load_venue_spec("icml"), 1981, adapter, client)

    assert outcome.papers == ()
    assert outcome.issues == ()
    assert outcome.run.status is RunStatus.NOT_APPLICABLE
    assert outcome.run.source_name is None
    assert outcome.run.membership_complete is False
    assert outcome.run.metadata_complete is False
    assert outcome.run.complete is False
    assert outcome.run.counts.to_dict() == {
        "raw_items": 0,
        "included_papers": 0,
        "complete_papers": 0,
        "incomplete_papers": 0,
        "excluded_non_papers": 0,
        "duplicate_occurrences": 0,
        "parse_rejects": 0,
        "issue_records": 0,
    }
    assert outcome.run.pagination is None
    assert outcome.run.warnings == ()
    assert outcome.run.errors == ()
    assert adapter.calls == []
    assert client.calls == []
    assert not hasattr(outcome, "__dict__")


def test_invalid_year_is_allowed_to_propagate_before_adapter_call() -> None:
    adapter = FakeAdapter(result=_result())

    with pytest.raises(InputError):
        collect_venue_year(load_venue_spec("icml"), 99, adapter, PrefixClient({}))

    assert adapter.calls == []


def test_collection_error_is_failed_without_exception_details_or_retry() -> None:
    error = CollectionError("secret URL https://private.test body credential=secret")
    adapter = FakeAdapter(error=error)
    client = PrefixClient({})

    outcome = collect_venue_year(load_venue_spec("icml"), 2024, adapter, client)

    assert outcome.papers == ()
    assert outcome.issues == ()
    assert outcome.run.status is RunStatus.FAILED
    assert outcome.run.source_name == "authoritative"
    assert outcome.run.membership_complete is False
    assert outcome.run.metadata_complete is False
    assert outcome.run.complete is False
    assert outcome.run.pagination is None
    assert outcome.run.counts.raw_items == 0
    assert outcome.run.errors == ("authoritative membership collection failed",)
    assert str(error) not in outcome.run.errors
    assert adapter.calls and len(adapter.calls) == 1
    assert client.calls == []


def test_non_collection_errors_are_not_caught() -> None:
    error = RuntimeError("implementation error")
    adapter = FakeAdapter(error=error)

    with pytest.raises(RuntimeError, match="implementation error"):
        collect_venue_year(load_venue_spec("icml"), 2024, adapter, PrefixClient({}))


def test_complete_paper_preserves_identity_and_uses_direct_pdf_field_sources() -> None:
    candidate = "https://example.test/opaque-candidate"
    paper = _paper(
        "opaque/source/id",
        title=" <p>A Paper &amp; Title</p> ",
        authors=(" Ada Lovelace ",),
        abstract="<div> An abstract. </div>",
        doi="https://doi.org/10.1234/EXAMPLE.1",
        pdf_candidates=(candidate,),
    )
    result = _result(
        (paper,),
        raw_items=1,
        pagination=Pagination(1, True, SourceTotal(1, SourceTotalScope.RAW_ITEMS)),
    )
    client = _pdf_client(candidate)

    outcome = _collect(result, client)

    assert len(outcome.papers) == 1
    assert outcome.issues == ()
    record = outcome.papers[0]
    assert record.identity.as_tuple() == ("icml", 2024, "authoritative", "opaque/source/id")
    assert record.title == "A Paper & Title"
    assert record.authors == ("Ada Lovelace",)
    assert record.abstract == "An abstract."
    assert record.doi == "10.1234/example.1"
    assert record.landing_url == paper.landing_url
    assert record.pdf_url == candidate
    assert record.access_status is AccessStatus.DIRECT_PDF
    assert record.field_sources.to_dict() == {
        "title": "authoritative",
        "authors": "authoritative",
        "abstract": "authoritative",
        "doi": "authoritative",
        "landing_url": "authoritative",
        "pdf_url": "authoritative",
    }
    assert client.calls == [(candidate, 4096)]
    assert outcome.run.status is RunStatus.COMPLETE
    assert outcome.run.membership_complete is True
    assert outcome.run.metadata_complete is True
    assert outcome.run.complete is True
    assert outcome.run.counts.to_dict() == {
        "raw_items": 1,
        "included_papers": 1,
        "complete_papers": 1,
        "incomplete_papers": 0,
        "excluded_non_papers": 0,
        "duplicate_occurrences": 0,
        "parse_rejects": 0,
        "issue_records": 0,
    }
    assert outcome.run.pagination == result.pagination
    assert outcome.run.errors == ()


def test_valid_doi_without_verified_pdf_is_complete_doi_only() -> None:
    paper = _paper(
        "doi-only",
        doi=" DOI:10.1234/DOI.1 ",
        pdf_candidates=(),
    )

    outcome = _collect(
        _result(
            (paper,),
            raw_items=1,
            pagination=Pagination(1, True, SourceTotal(1, SourceTotalScope.RAW_ITEMS)),
        )
    )

    assert outcome.issues == ()
    record = outcome.papers[0]
    assert record.doi == "10.1234/doi.1"
    assert record.pdf_url is None
    assert record.access_status is AccessStatus.DOI_ONLY
    assert record.field_sources.to_dict()["doi"] == "authoritative"
    assert record.field_sources.to_dict()["pdf_url"] is None
    assert outcome.run.status is RunStatus.COMPLETE
    assert outcome.run.complete is True


@pytest.mark.parametrize(
    "doi", [None, "", "not-a-doi"], ids=["missing", "empty", "invalid"]
)
def test_missing_or_invalid_doi_without_verified_pdf_is_incomplete(doi: str | None) -> None:
    paper = _paper("no-access", doi=doi, pdf_candidates=())

    outcome = _collect(_result((paper,), raw_items=1))

    assert outcome.papers == ()
    assert len(outcome.issues) == 1
    issue = outcome.issues[0]
    assert issue.doi is None
    assert issue.missing_fields == (MissingField.ACCESS_LOCATOR,)
    assert issue.reason_codes == ("no_verified_pdf_or_doi",)
    assert outcome.run.status is RunStatus.PARTIAL
    assert outcome.run.complete is False


def test_normalized_title_repeated_abstract_is_incomplete() -> None:
    candidate = "https://example.test/repeated.pdf"
    paper = _paper(
        "repeated",
        title="<p> Repeated Title </p>",
        abstract="<div>Repeated   Title</div>",
        pdf_candidates=(candidate,),
    )

    outcome = _collect(
        _result(
            (paper,),
            raw_items=1,
            pagination=Pagination(1, True, SourceTotal(1, SourceTotalScope.RAW_ITEMS)),
        ),
        _pdf_client(candidate),
    )

    assert outcome.papers == ()
    assert len(outcome.issues) == 1
    issue = outcome.issues[0]
    assert issue.missing_fields == (MissingField.ABSTRACT,)
    assert issue.reason_codes == ("missing_abstract",)
    assert issue.title == "Repeated Title"
    assert issue.abstract is None
    assert outcome.run.counts.to_dict() == {
        "raw_items": 1,
        "included_papers": 1,
        "complete_papers": 0,
        "incomplete_papers": 1,
        "excluded_non_papers": 0,
        "duplicate_occurrences": 0,
        "parse_rejects": 0,
        "issue_records": 1,
    }
    assert outcome.run.membership_complete is True
    assert outcome.run.metadata_complete is False
    assert outcome.run.complete is False
    assert outcome.run.status is RunStatus.PARTIAL


def test_nonidentical_normalized_abstract_remains_complete() -> None:
    candidate = "https://example.test/normal.pdf"
    paper = _paper(
        "normal",
        title="Repeated Title",
        abstract="repeated title",
        pdf_candidates=(candidate,),
    )

    outcome = _collect(
        _result(
            (paper,),
            raw_items=1,
            pagination=Pagination(1, True, SourceTotal(1, SourceTotalScope.RAW_ITEMS)),
        ),
        _pdf_client(candidate),
    )

    assert len(outcome.papers) == 1
    assert outcome.issues == ()
    assert outcome.papers[0].abstract == "repeated title"
    assert outcome.run.counts.complete_papers == 1
    assert outcome.run.status is RunStatus.COMPLETE
    assert outcome.run.complete is True


def test_normalization_keeps_author_order_duplicates_and_distinct_source_membership() -> None:
    first = _paper(
        "first",
        title="<p>Shared title</p>",
        authors=(" <b>Ada Lovelace</b> ", " ", "Grace Hopper", "Grace Hopper"),
        abstract="<p>First abstract</p>",
        pdf_candidates=("https://example.test/first-access",),
    )
    second = _paper(
        "second",
        title=" Shared title ",
        authors=("Alan Turing",),
        abstract="Second abstract",
        pdf_candidates=("https://example.test/second-access",),
    )
    client = _pdf_client("https://example.test/first-access", "https://example.test/second-access")

    outcome = _collect(_result((first, second), raw_items=2), client)

    assert [paper.source_id for paper in outcome.papers] == ["first", "second"]
    assert outcome.papers[0].title == "Shared title"
    assert outcome.papers[0].authors == ("Ada Lovelace", "Grace Hopper", "Grace Hopper")
    assert outcome.papers[0].abstract == "First abstract"
    assert outcome.run.counts.complete_papers == 2


def test_incomplete_issue_has_fixed_field_and_reason_order_and_retains_fields() -> None:
    paper = _paper(
        "incomplete",
        title=" ",
        authors=(" ",),
        abstract="<div> </div>",
        doi=" DOI:10.1234/INCOMPLETE.1 ",
        pdf_candidates=(),
    )

    outcome = _collect(_result((paper,), raw_items=1))

    assert outcome.papers == ()
    assert len(outcome.issues) == 1
    issue = outcome.issues[0]
    assert issue.issue_kind is IssueKind.INCOMPLETE_PAPER
    assert issue.source_name == "authoritative"
    assert issue.source_id == "incomplete"
    assert issue.source_locator == paper.landing_url
    assert issue.title is None
    assert issue.authors == ()
    assert issue.abstract is None
    assert issue.doi == "10.1234/incomplete.1"
    assert issue.landing_url == paper.landing_url
    assert issue.missing_fields == (
        MissingField.TITLE,
        MissingField.AUTHORS,
        MissingField.ABSTRACT,
    )
    assert issue.reason_codes == (
        "missing_title",
        "missing_authors",
        "missing_abstract",
    )
    assert issue.message == "required metadata or access locator is missing"
    assert outcome.run.metadata_complete is False


def test_unverified_pdf_without_doi_is_an_access_locator_blocker() -> None:
    candidate = "https://example.test/not-a-pdf"
    client = PrefixClient(
        {candidate: PrefixResponse("text/html", b"<html>login</html>")}
    )
    paper = _paper("no-access", pdf_candidates=(candidate,))

    outcome = _collect(_result((paper,), raw_items=1), client)

    issue = outcome.issues[0]
    assert issue.missing_fields == (MissingField.ACCESS_LOCATOR,)
    assert issue.reason_codes == ("no_verified_pdf_or_doi",)
    assert client.calls == [(candidate, 4096)]


def test_parse_rejects_map_to_issue_kinds_and_only_block_membership() -> None:
    paper = _paper("complete", pdf_candidates=("https://example.test/complete",))
    rejects = (
        ParseReject("https://example.test/#one", "parse_failed", "parse failed"),
        ParseReject("https://example.test/#two", "identity_conflict", "identity conflict"),
    )
    result = _result(
        (paper,),
        parse_rejects=rejects,
        raw_items=3,
        pagination=Pagination(1, True, SourceTotal(3, SourceTotalScope.RAW_ITEMS)),
    )

    outcome = _collect(result, _pdf_client("https://example.test/complete"))

    assert [issue.issue_kind for issue in outcome.issues] == [
        IssueKind.PARSE_REJECT,
        IssueKind.IDENTITY_CONFLICT,
    ]
    assert [issue.source_id for issue in outcome.issues] == [None, None]
    assert [issue.reason_codes for issue in outcome.issues] == [
        ("parse_failed",),
        ("identity_conflict",),
    ]
    assert [issue.source_locator for issue in outcome.issues] == [
        "https://example.test/#one",
        "https://example.test/#two",
    ]
    assert outcome.run.counts.parse_rejects == 2
    assert outcome.run.counts.issue_records == 2
    assert outcome.run.membership_complete is False
    assert outcome.run.metadata_complete is True
    assert outcome.run.status is RunStatus.PARTIAL
    assert outcome.run.errors == ()


def test_nonterminal_pagination_is_a_run_blocker_with_metadata_still_complete() -> None:
    candidate = "https://example.test/nonterminal"
    result = _result(
        (_paper("nonterminal", pdf_candidates=(candidate,)),),
        raw_items=1,
        pagination=Pagination(1, False, None),
    )

    outcome = _collect(result, _pdf_client(candidate))

    assert outcome.run.errors == ("authoritative pagination did not reach a terminal state",)
    assert outcome.run.membership_complete is False
    assert outcome.run.metadata_complete is True
    assert outcome.run.status is RunStatus.PARTIAL


@pytest.mark.parametrize(
    ("scope", "value"),
    [
        (SourceTotalScope.RAW_ITEMS, 2),
        (SourceTotalScope.INCLUDED_PAPERS, 1),
    ],
)
def test_matching_source_total_scope_can_complete_membership(scope, value: int) -> None:
    candidate = "https://example.test/total"
    result = _result(
        (_paper("total", pdf_candidates=(candidate,)),),
        excluded_non_papers=1,
        raw_items=2,
        pagination=Pagination(1, True, SourceTotal(value, scope)),
    )

    outcome = _collect(result, _pdf_client(candidate))

    assert outcome.run.pagination == result.pagination
    assert outcome.run.membership_complete is True
    assert outcome.run.status is RunStatus.COMPLETE
    assert outcome.run.errors == ()


@pytest.mark.parametrize(
    ("scope", "value"),
    [
        (SourceTotalScope.RAW_ITEMS, 1),
        (SourceTotalScope.INCLUDED_PAPERS, 2),
    ],
)
def test_mismatched_source_total_is_a_run_blocker(scope, value: int) -> None:
    candidate = "https://example.test/mismatch"
    result = _result(
        (_paper("mismatch", pdf_candidates=(candidate,)),),
        excluded_non_papers=1,
        raw_items=2,
        pagination=Pagination(1, True, SourceTotal(value, scope)),
    )

    outcome = _collect(result, _pdf_client(candidate))

    assert outcome.run.errors == ("source total does not match collected counts",)
    assert outcome.run.membership_complete is False
    assert outcome.run.metadata_complete is True
    assert outcome.run.status is RunStatus.PARTIAL


@pytest.mark.parametrize(
    ("pagination", "expected_status", "expected_errors"),
    [
        (
            Pagination(1, True, None),
            RunStatus.PARTIAL,
            ("applicable venue-year has no authoritative zero-paper proof",),
        ),
        (
            Pagination(1, True, SourceTotal(0, SourceTotalScope.RAW_ITEMS)),
            RunStatus.COMPLETE,
            (),
        ),
        (
            Pagination(1, True, SourceTotal(1, SourceTotalScope.INCLUDED_PAPERS)),
            RunStatus.PARTIAL,
            (
                "source total does not match collected counts",
                "applicable venue-year has no authoritative zero-paper proof",
            ),
        ),
    ],
)
def test_applicable_empty_requires_authoritative_zero_proof(
    pagination: Pagination, expected_status: RunStatus, expected_errors: tuple[str, ...]
) -> None:
    outcome = _collect(_result((), raw_items=0, pagination=pagination))

    assert outcome.papers == ()
    assert outcome.issues == ()
    assert outcome.run.status is expected_status
    assert outcome.run.membership_complete == (expected_status is RunStatus.COMPLETE)
    assert outcome.run.metadata_complete is True
    assert outcome.run.errors == expected_errors


def test_counts_equations_issue_order_and_source_identity_are_exact() -> None:
    complete_candidate = "https://example.test/complete-counts"
    complete = _paper("opaque/complete", pdf_candidates=(complete_candidate,))
    incomplete = _paper("opaque/incomplete", abstract=None, pdf_candidates=())
    reject = ParseReject("https://example.test/#reject", "parse_failed", "parse failed")
    result = _result(
        (complete, incomplete),
        excluded_non_papers=1,
        duplicate_occurrences=1,
        parse_rejects=(reject,),
        raw_items=5,
        pagination=Pagination(1, True, SourceTotal(5, SourceTotalScope.RAW_ITEMS)),
    )

    outcome = _collect(result, _pdf_client(complete_candidate))

    assert [paper.identity.as_tuple() for paper in outcome.papers] == [
        ("icml", 2024, "authoritative", "opaque/complete")
    ]
    assert [issue.issue_kind for issue in outcome.issues] == [
        IssueKind.PARSE_REJECT,
        IssueKind.INCOMPLETE_PAPER,
    ]
    assert outcome.issues[1].source_id == "opaque/incomplete"
    assert outcome.run.counts.to_dict() == {
        "raw_items": 5,
        "included_papers": 2,
        "complete_papers": 1,
        "incomplete_papers": 1,
        "excluded_non_papers": 1,
        "duplicate_occurrences": 1,
        "parse_rejects": 1,
        "issue_records": 2,
    }
    assert outcome.run.membership_complete is False
    assert outcome.run.metadata_complete is False
    assert outcome.run.complete is False
    assert outcome.run.status is RunStatus.PARTIAL
    assert outcome.run.errors == ()


def test_real_pmlr_caller_is_complete_vertical_slice_without_network_or_files(monkeypatch, tmp_path) -> None:
    responses = {
        VOLUME_URL: (FIXTURES / "volume-v235.html").read_bytes(),
        ADA_URL: (FIXTURES / "lovelace24a.html").read_bytes(),
        TURING_URL: (FIXTURES / "turing24a.html").read_bytes(),
        RAW_ADA_PDF: b"%PDF-1.7\n" + b"x" * 10000,
    }
    opener = PmlrFixtureOpener(responses)
    monkeypatch.setattr(http_module, "urlopen", opener)
    output_before = tuple(tmp_path.iterdir())

    outcome = collect_venue_year(
        load_venue_spec("icml"),
        2024,
        PmlrAdapter(),
        HttpClient("collector@example.org", 6.5),
    )

    assert [paper.source_id for paper in outcome.papers] == ["v235/lovelace24a"]
    ada = outcome.papers[0]
    assert ada.title == "Reliable Small Models & Graphs"
    assert ada.authors == ("Ada Lovelace", "Grace Hopper")
    assert ada.abstract == "Reliable small models & graphs for reproducible experiments."
    assert ada.pdf_url == RAW_ADA_PDF
    assert ada.access_status is AccessStatus.DIRECT_PDF
    assert len(outcome.issues) == 2
    assert outcome.issues[0].issue_kind is IssueKind.PARSE_REJECT
    assert outcome.issues[1].issue_kind is IssueKind.INCOMPLETE_PAPER
    assert outcome.issues[1].source_id == "v235/turing24a"
    assert outcome.issues[1].missing_fields == (MissingField.ACCESS_LOCATOR,)
    assert outcome.issues[1].reason_codes == ("no_verified_pdf_or_doi",)
    assert outcome.run.status is RunStatus.PARTIAL
    assert outcome.run.membership_complete is False
    assert outcome.run.metadata_complete is False
    assert outcome.run.complete is False
    assert outcome.run.counts.to_dict() == {
        "raw_items": 5,
        "included_papers": 2,
        "complete_papers": 1,
        "incomplete_papers": 1,
        "excluded_non_papers": 1,
        "duplicate_occurrences": 1,
        "parse_rejects": 1,
        "issue_records": 2,
    }
    requests = [request.full_url for request, _ in opener.calls]
    assert requests == [VOLUME_URL, ADA_URL, TURING_URL, RAW_ADA_PDF]
    assert opener.responses_seen[-1].read_limits == [4096]
    assert all(passed_timeout == 6.5 for _, passed_timeout in opener.calls)
    assert tuple(tmp_path.iterdir()) == output_before


def test_real_pmlr_detail_failure_is_partial_with_membership_preserved(monkeypatch) -> None:
    volume = f"""\
    <html><body>
      <div class="paper">
        <p class="title">Failed Detail</p>
        <span class="authors"><a>Ada Lovelace</a></span>
        <a href="lovelace24a.html">abs</a>
        <a href="{RAW_ADA_PDF}">Download PDF</a>
      </div>
      <div class="paper">
        <p class="title">Later Detail</p>
        <span class="authors"><a>Alan Turing</a></span>
        <a href="turing24a.html">abs</a>
      </div>
    </body></html>
    """
    responses = {
        VOLUME_URL: volume.encode(),
        ADA_URL: CollectionError("detail request failed"),
        TURING_URL: (FIXTURES / "turing24a.html").read_bytes(),
        RAW_ADA_PDF: b"%PDF-1.7\n" + b"x" * 10000,
    }
    opener = PmlrFixtureOpener(responses)
    monkeypatch.setattr(http_module, "urlopen", opener)

    outcome = collect_venue_year(
        load_venue_spec("icml"),
        2024,
        PmlrAdapter(),
        HttpClient("collector@example.org", 6.5),
    )

    assert outcome.papers == ()
    assert outcome.run.status is RunStatus.PARTIAL
    assert outcome.run.status is not RunStatus.FAILED
    assert outcome.run.membership_complete is True
    assert outcome.run.metadata_complete is False
    assert outcome.run.complete is False
    assert outcome.run.counts.to_dict() == {
        "raw_items": 2,
        "included_papers": 2,
        "complete_papers": 0,
        "incomplete_papers": 2,
        "excluded_non_papers": 0,
        "duplicate_occurrences": 0,
        "parse_rejects": 0,
        "issue_records": 2,
    }
    failed_detail = outcome.issues[0]
    assert failed_detail.source_id == "v235/lovelace24a"
    assert failed_detail.missing_fields == (MissingField.ABSTRACT,)
    assert failed_detail.reason_codes == ("missing_abstract",)
    assert [request.full_url for request, _ in opener.calls] == [
        VOLUME_URL,
        ADA_URL,
        TURING_URL,
        RAW_ADA_PDF,
    ]


def test_real_pmlr_volume_failure_remains_failed(monkeypatch) -> None:
    opener = PmlrFixtureOpener({VOLUME_URL: CollectionError("volume request failed")})
    monkeypatch.setattr(http_module, "urlopen", opener)

    outcome = collect_venue_year(
        load_venue_spec("icml"),
        2024,
        PmlrAdapter(),
        HttpClient("collector@example.org", 6.5),
    )

    assert outcome.papers == ()
    assert outcome.issues == ()
    assert outcome.run.status is RunStatus.FAILED
    assert outcome.run.membership_complete is False
    assert outcome.run.metadata_complete is False
    assert outcome.run.complete is False
    assert outcome.run.counts.raw_items == 0
    assert outcome.run.errors == ("authoritative membership collection failed",)
    assert [request.full_url for request, _ in opener.calls] == [VOLUME_URL]
