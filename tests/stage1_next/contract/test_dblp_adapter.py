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
    assert spec.enrichers == (
        "enrichers.semantic_scholar:SemanticScholarEnricher",
    )
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
    assert result.raw_items == 9
    assert result.excluded_non_papers == 2
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


@pytest.mark.parametrize(
    ("venue_id", "expected_ids"),
    [
        (
            "dac",
            (
                "conf/dac/First24",
                "conf/iccad/Foreword24",
                "conf/dac/Second24",
                "conf/dac/Third24",
            ),
        ),
        (
            "iccad",
            (
                "conf/dac/First24",
                "conf/dac/Frontmatter24",
                "conf/dac/Second24",
                "conf/dac/Third24",
            ),
        ),
    ],
)
def test_real_dblp_bht_partitions_are_flattened_in_document_order(
    venue_id: str, expected_ids: tuple[str, ...]
) -> None:
    spec = load_venue_spec(venue_id)
    series = spec.source_for_year(2024)["series"]
    url = f"https://dblp.org/db/conf/{series}/{series}2024.xml"

    result = DblpTocAdapter().collect(
        spec,
        2024,
        FakeTextClient({url: _fixture("real_structure.xml")}),
    )

    assert [paper.source_id for paper in result.papers] == list(expected_ids)
    assert result.raw_items == 6
    assert result.excluded_non_papers == 2
    assert result.duplicate_occurrences == 0
    assert result.parse_rejects == ()
    assert result.raw_items == (
        len(result.papers)
        + result.excluded_non_papers
        + result.duplicate_occurrences
        + len(result.parse_rejects)
    )


def test_exclude_title_matching_stays_exact_after_small_normalization() -> None:
    url = "https://dblp.org/db/conf/dac/dac2024.xml"
    result = DblpTocAdapter().collect(
        load_venue_spec("dac"),
        2024,
        FakeTextClient({url: _fixture("title_boundaries.xml")}),
    )

    assert [paper.source_id for paper in result.papers] == [
        "conf/dac/Extended24",
        "conf/dac/DoublePeriod24",
        "conf/dac/Substring24",
    ]
    assert result.raw_items == 5
    assert result.excluded_non_papers == 2
    assert result.parse_rejects == ()


@pytest.mark.parametrize(
    "fixture",
    ["malformed.xml", "unrecognized.xml", "unknown_group.xml", "namespaced.xml"],
)
def test_malformed_or_unrecognized_toc_fails_closed(fixture: str) -> None:
    url = "https://dblp.org/db/conf/dac/dac2024.xml"

    with pytest.raises(CollectionError):
        DblpTocAdapter().collect(
            load_venue_spec("dac"),
            2024,
            FakeTextClient({url: _fixture(fixture)}),
        )


def test_empty_dblp_citations_have_no_authoritative_zero_proof() -> None:
    url = "https://dblp.org/db/conf/dac/dac2024.xml"
    with pytest.raises(CollectionError, match="no authoritative zero proof"):
        DblpTocAdapter().collect(
            load_venue_spec("dac"),
            2024,
            FakeTextClient({url: _fixture("empty.xml")}),
        )


def test_record_shape_and_unknown_kind_rejects_remain_accounted_for() -> None:
    url = "https://dblp.org/db/conf/dac/dac2024.xml"
    result = DblpTocAdapter().collect(
        load_venue_spec("dac"),
        2024,
        FakeTextClient({url: _fixture("record_shapes.xml")}),
    )

    assert result.papers == ()
    assert result.raw_items == 3
    assert result.excluded_non_papers == 0
    assert result.duplicate_occurrences == 0
    assert [reject.reason_code for reject in result.parse_rejects] == [
        "ambiguous_record",
        "ambiguous_record",
        "unsupported_record_kind",
    ]


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
