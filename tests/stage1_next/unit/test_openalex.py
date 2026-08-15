"""Offline contract coverage for strict OpenAlex residual enrichment."""

import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from paper_agent_next.adapters.base import CollectedPaper, CollectionResult
from paper_agent_next.catalog import load_venue_spec
from paper_agent_next.collector import collect_venue_year
from paper_agent_next.enrichers.base import FrozenPaper
from paper_agent_next.enrichers.openalex import OpenAlexEnricher
from paper_agent_next.errors import EnrichmentError
from paper_agent_next.http import PrefixResponse
from paper_agent_next.models import Pagination, SourceIdentity


FIXTURES = Path(__file__).parents[1] / "fixtures" / "openalex"
SELECT = (
    "id,doi,display_name,publication_year,authorships,"
    "abstract_inverted_index,best_oa_location,primary_location"
)


class _JsonClient:
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    def get_json(self, url: str) -> object:
        self.calls.append(url)
        return self.responses.pop(0)


class _CollectionClient(_JsonClient):
    def __init__(self, responses: list[object], pdf_url: str) -> None:
        super().__init__(responses)
        self.pdf_url = pdf_url
        self.pdf_calls: list[str] = []

    def get_prefix(self, url: str, max_bytes: int) -> PrefixResponse:
        self.pdf_calls.append(url)
        if url == self.pdf_url:
            return PrefixResponse("application/pdf", b"%PDF-1.7")
        return PrefixResponse("text/html", b"not a pdf")


class _Adapter:
    source_name = "primary"

    def collect(self, venue_spec, year, http_client) -> CollectionResult:
        return CollectionResult(
            source_name=self.source_name,
            papers=(
                CollectedPaper(
                    source_id="metadata",
                    title="Metadata exact paper",
                    authors=("Ada Lovelace",),
                    abstract=None,
                    doi=None,
                    landing_url="https://example.test/metadata",
                    pdf_candidates=(),
                ),
            ),
            raw_items=1,
            excluded_non_papers=0,
            duplicate_occurrences=0,
            parse_rejects=(),
            pagination=Pagination(1, True, None),
        )


def _fixture(name: str) -> object:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _view(
    source_id: str,
    *,
    title: str = "A paper",
    author: str = "Ada Lovelace",
    doi: str | None = None,
    abstract: str | None = None,
) -> FrozenPaper:
    return FrozenPaper(
        identity=SourceIdentity("dac", 2024, "dblp_toc", source_id),
        title=title,
        authors=(author,),
        abstract=abstract,
        doi=doi,
        landing_url=f"https://dblp.org/rec/{source_id}",
        pdf_candidates=(),
    )


def _response(results: list[object], *, count: int | None = None) -> dict[str, object]:
    return {"meta": {"count": len(results) if count is None else count}, "results": results}


def _work(
    doi: str | None,
    *,
    title: str = "A paper",
    year: int | None = 2024,
    author: str = "Ada Lovelace",
    abstract_index: object = None,
    best_pdf: object = None,
    primary_pdf: object = None,
) -> dict[str, object]:
    return {
        "id": "https://openalex.org/W1",
        "doi": doi,
        "display_name": title,
        "publication_year": year,
        "authorships": [
            {"author_position": "first", "author": {"display_name": author}}
        ],
        "abstract_inverted_index": abstract_index,
        "best_oa_location": None if best_pdf is None else {"pdf_url": best_pdf},
        "primary_location": None if primary_pdf is None else {"pdf_url": primary_pdf},
    }


def test_doi_batch_uses_encoded_filter_and_binds_reordered_results_by_normalized_doi() -> None:
    papers = (
        _view("one", doi="10.1234/one"),
        _view("two", title="Second paper", doi="10.1234/two"),
    )
    client = _JsonClient([_fixture("doi-reordered.json")])

    patches = OpenAlexEnricher().enrich(papers, client)

    assert {patch.identity.source_id: patch.abstract for patch in patches} == {
        "one": "first abstract",
        "two": "second abstract",
    }
    assert all(patch.doi is None for patch in patches)
    assert len(client.calls) == 1
    query = parse_qs(urlsplit(client.calls[0]).query)
    assert query == {
        "filter": ["doi:https://doi.org/10.1234/one|doi:https://doi.org/10.1234/two"],
        "per-page": ["100"],
        "select": [SELECT],
    }


def test_doi_batch_missing_result_is_a_normal_no_match() -> None:
    papers = (
        _view("one", doi="10.1234/one"),
        _view("two", doi="10.1234/two"),
    )
    result = _work("10.1234/one", abstract_index={"only": [0]})
    client = _JsonClient([_response([result])])

    patches = OpenAlexEnricher().enrich(papers, client)

    assert [patch.identity.source_id for patch in patches] == ["one"]
    assert client.calls


@pytest.mark.parametrize(
    "result",
    [
        _work("10.1234/unknown"),
        [_work("10.1234/one"), _work("10.1234/one")],
        _work("not-a-doi"),
    ],
    ids=["unknown-doi", "duplicate-doi", "malformed-doi"],
)
def test_doi_batch_unknown_duplicate_and_malformed_doi_are_typed_failures(
    result: object,
) -> None:
    if isinstance(result, list):
        response = _response(result)
    else:
        response = _response([result])
    client = _JsonClient([response])

    with pytest.raises(EnrichmentError):
        OpenAlexEnricher().enrich(
            (_view("one", doi="10.1234/one"),),
            client,
        )


def test_doi_batch_truncation_is_a_typed_failure() -> None:
    client = _JsonClient(
        [_response([_work("10.1234/one")], count=2)]
    )

    with pytest.raises(EnrichmentError):
        OpenAlexEnricher().enrich((_view("one", doi="10.1234/one"),), client)


def test_duplicate_frozen_doi_and_satisfied_doi_are_not_requested() -> None:
    duplicate = (
        _view("one", doi="10.1234/same"),
        _view("two", doi="DOI:10.1234/SAME"),
    )
    already_complete = _view(
        "complete", doi="10.1234/complete", abstract="Already present"
    )
    client = _JsonClient([])

    assert OpenAlexEnricher().enrich(duplicate + (already_complete,), client) == ()
    assert client.calls == []


def test_strict_metadata_match_adds_abstract_doi_and_stable_pdf_candidates() -> None:
    paper = _view("metadata", title="Metadata exact paper")
    client = _JsonClient([_fixture("metadata-exact.json")])

    patches = OpenAlexEnricher().enrich((paper,), client)

    assert len(patches) == 1
    patch = patches[0]
    assert patch.identity == paper.identity
    assert patch.abstract == "Exact metadata abstract"
    assert patch.doi == "10.5555/metadata"
    assert patch.pdf_candidates == ("https://example.test/openalex.pdf",)
    query = parse_qs(urlsplit(client.calls[0]).query)
    assert query["search.exact"] == ["Metadata exact paper"]
    assert query["filter"] == ["publication_year:2024"]
    assert query["per-page"] == ["100"]
    assert query["select"] == [SELECT]


def test_openalex_pdf_candidate_is_unverified_until_collector_access_verifies_it() -> None:
    pdf_url = "https://example.test/openalex.pdf"
    client = _CollectionClient([_fixture("metadata-exact.json")], pdf_url)

    outcome = collect_venue_year(
        load_venue_spec("dac"),
        2024,
        _Adapter(),
        client,  # type: ignore[arg-type]
        enrichers=(OpenAlexEnricher(),),
    )

    assert outcome.run.complete is True
    assert outcome.issues == ()
    record = outcome.papers[0]
    assert record.pdf_url == pdf_url
    assert record.access_status.value == "direct_pdf"
    assert record.field_sources.pdf_url == "openalex"
    assert record.field_sources.abstract == "openalex"
    assert record.field_sources.doi == "openalex"
    assert client.pdf_calls == [pdf_url]


@pytest.mark.parametrize(
    "work",
    [
        _work("10.5555/wrong-title", title="Other paper"),
        _work("10.5555/wrong-year", year=2023),
        _work("10.5555/wrong-author", author="Grace Hopper"),
    ],
    ids=["title", "year", "first-author"],
)
def test_strict_metadata_rejects_single_candidate_without_all_three_exact_fields(
    work: dict[str, object],
) -> None:
    client = _JsonClient([_response([work])])

    assert OpenAlexEnricher().enrich((_view("paper"),), client) == ()


@pytest.mark.parametrize(
    "response",
    [
        _response([_work("10.5555/one"), _work("10.5555/two")]),
        _response([_work("10.5555/one")], count=2),
    ],
    ids=["multiple-exact", "truncated-search"],
)
def test_strict_metadata_rejects_multiple_or_truncated_search_results(
    response: dict[str, object],
) -> None:
    client = _JsonClient([response])

    assert OpenAlexEnricher().enrich((_view("paper"),), client) == ()


def test_duplicate_frozen_strict_key_is_not_searched() -> None:
    papers = (_view("one"), _view("two"))
    client = _JsonClient([])

    assert OpenAlexEnricher().enrich(papers, client) == ()
    assert client.calls == []


def test_metadata_candidate_doi_owned_by_another_frozen_paper_is_not_assigned() -> None:
    owner = _view(
        "owner",
        title="Other paper",
        doi="10.5555/owned",
        abstract="Owned abstract",
    )
    target = _view("target")
    client = _JsonClient(
        [_response([_work("10.5555/owned", abstract_index={"unsafe": [0]})])]
    )

    assert OpenAlexEnricher().enrich((target, owner), client) == ()


@pytest.mark.parametrize(
    "abstract_index",
    [
        {},
        {"word": "not-a-list"},
        {"word": [-1]},
        {"first": [0], "duplicate": [0]},
    ],
    ids=["empty", "wrong-type", "negative-position", "duplicate-position"],
)
def test_malformed_abstract_inverted_index_is_a_typed_failure(
    abstract_index: object,
) -> None:
    client = _JsonClient(
        [_response([_work("10.1234/one", abstract_index=abstract_index)])]
    )

    with pytest.raises(EnrichmentError):
        OpenAlexEnricher().enrich((_view("one", doi="10.1234/one"),), client)


def test_work_with_no_new_values_does_not_return_an_empty_patch() -> None:
    client = _JsonClient(
        [_response([_work("10.1234/one", abstract_index=None)])]
    )

    assert OpenAlexEnricher().enrich((_view("one", doi="10.1234/one"),), client) == ()
