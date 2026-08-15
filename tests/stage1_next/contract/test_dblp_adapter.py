"""Contract tests for the isolated DBLP TOC adapter."""

from pathlib import Path

import pytest

from paper_agent_next.adapters.dblp import DblpTocAdapter
from paper_agent_next.catalog import load_venue_spec
from paper_agent_next.errors import CollectionError
from paper_agent_next.models import Pagination


FIXTURES = Path(__file__).parents[1] / "fixtures" / "dblp"


class FakeTextClient:
    def __init__(self, responses: dict[str, str | BaseException]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    def get_text(self, url: str) -> str:
        self.calls.append(url)
        response = self.responses[url]
        if isinstance(response, BaseException):
            raise response
        return response


def _fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


EDA_SPECS = (
    ("dac", "dac", "Frontmatter"),
    ("iccad", "iccad", "Foreword"),
    ("date", "date", "Frontmatter"),
    ("aspdac", "aspdac", "Frontmatter"),
    ("ispd", "ispd", "Frontmatter"),
)


@pytest.mark.parametrize(("venue_id", "series", "exclude_title"), EDA_SPECS)
def test_eda_specs_use_one_dblp_url_and_preserve_source_metadata(
    venue_id: str, series: str, exclude_title: str
) -> None:
    spec = load_venue_spec(venue_id)
    url = f"https://dblp.org/db/conf/{series}/{series}2024.xml"
    client = FakeTextClient({url: _fixture("valid.xml")})

    result = DblpTocAdapter().collect(spec, 2024, client)

    assert spec.adapter == "adapters.dblp:DblpTocAdapter"
    assert spec.enrichers == ()
    assert dict(spec.source_for_year(2024)) == {
        "series": series,
        "exclude_title": exclude_title,
    }
    assert spec.is_applicable(2016) is True
    assert spec.is_applicable(2025) is True
    assert spec.is_applicable(2015) is False
    assert spec.is_applicable(2026) is False
    assert client.calls == [url]
    assert result.source_name == "dblp_toc"
    assert result.pagination == Pagination(1, True, None)
    assert result.raw_items == 1
    assert result.excluded_non_papers == 0
    assert result.duplicate_occurrences == 0
    assert result.parse_rejects == ()

    paper = result.papers[0]
    assert paper.source_id == "conf/fixture/Fixture24"
    assert paper.title == "DBLP inline title & check"
    assert paper.authors == ("Ada Lovelace",)
    assert paper.abstract is None
    assert paper.doi == "10.1145/fixture.dac"
    assert paper.landing_url == "https://dblp.org/rec/conf/fixture/Fixture24"
    assert paper.pdf_candidates == ()


def test_every_dblp_raw_item_is_accounted_for() -> None:
    url = "https://dblp.org/db/conf/dac/dac2024.xml"
    result = DblpTocAdapter().collect(
        load_venue_spec("dac"),
        2024,
        FakeTextClient({url: _fixture("counts.xml")}),
    )

    assert [paper.source_id for paper in result.papers] == ["conf/dac/Paper24"]
    assert result.raw_items == 8
    assert result.excluded_non_papers == 1
    assert result.duplicate_occurrences == 1
    assert [reject.reason_code for reject in result.parse_rejects] == [
        "identity_conflict",
        "unsupported_record_kind",
        "missing_source_id",
        "record_year_conflict",
        "doi_conflict",
    ]
    assert result.raw_items == (
        len(result.papers)
        + result.excluded_non_papers
        + result.duplicate_occurrences
        + len(result.parse_rejects)
    )
    assert result.papers[0].doi == "10.1145/counts.dac"
    assert result.papers[0].pdf_candidates == ()


@pytest.mark.parametrize("fixture", ["malformed.xml", "unrecognized.xml"])
def test_malformed_or_unrecognized_toc_fails_closed(fixture: str) -> None:
    url = "https://dblp.org/db/conf/dac/dac2024.xml"

    with pytest.raises(CollectionError):
        DblpTocAdapter().collect(
            load_venue_spec("dac"),
            2024,
            FakeTextClient({url: _fixture(fixture)}),
        )


def test_empty_dblp_citations_are_an_explicit_zero_membership_result() -> None:
    url = "https://dblp.org/db/conf/dac/dac2024.xml"
    result = DblpTocAdapter().collect(
        load_venue_spec("dac"),
        2024,
        FakeTextClient({url: _fixture("empty.xml")}),
    )

    assert result.papers == ()
    assert result.raw_items == 0
    assert result.excluded_non_papers == 0
    assert result.duplicate_occurrences == 0
    assert result.parse_rejects == ()
    assert result.pagination == Pagination(1, True, None)


def test_empty_or_non_text_response_does_not_become_a_complete_census() -> None:
    url = "https://dblp.org/db/conf/dac/dac2024.xml"

    with pytest.raises(CollectionError):
        DblpTocAdapter().collect(
            load_venue_spec("dac"),
            2024,
            FakeTextClient({url: ""}),
        )

    with pytest.raises(CollectionError):
        DblpTocAdapter().collect(
            load_venue_spec("dac"),
            2024,
            FakeTextClient({url: b"not text"}),  # type: ignore[arg-type]
        )


def test_source_parameters_are_strict_and_do_not_accept_a_list_dsl() -> None:
    from dataclasses import replace

    spec = load_venue_spec("dac")
    url = "https://dblp.org/db/conf/dac/dac2024.xml"
    client = FakeTextClient({url: _fixture("valid.xml")})

    for source in (
        {"series": "DAC", "exclude_title": "Frontmatter"},
        {"series": "dac-name", "exclude_title": "Frontmatter"},
        {"series": "dac", "exclude_titles": ["Frontmatter"]},
        {"series": "dac", "exclude_title": ["Frontmatter"]},
    ):
        with pytest.raises(CollectionError):
            DblpTocAdapter().collect(
                replace(spec, source=source),
                2024,
                client,
            )
