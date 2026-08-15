"""Catalog and trusted-loader coverage for the Crossref EDA journal family."""

import pytest

from paper_agent_next.adapters.crossref import CrossrefSerialAdapter
from paper_agent_next.catalog import load_venue_spec
from paper_agent_next.loading import load_adapter, load_enrichers
from paper_agent_next.models import VenueType


_EXPECTED = {
    "tcad": {
        "name": "IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems",
        "start_year": 1982,
        "issn": "0278-0070",
    },
    "todaes": {
        "name": "ACM Transactions on Design Automation of Electronic Systems",
        "start_year": 1996,
        "issn": "1084-4309",
    },
    "tvlsi": {
        "name": "IEEE Transactions on Very Large Scale Integration (VLSI) Systems",
        "start_year": 1993,
        "issn": "1063-8210",
    },
    "jssc": {
        "name": "IEEE Journal of Solid-State Circuits",
        "start_year": 1966,
        "issn": "0018-9200",
    },
}


@pytest.mark.parametrize("venue_id", tuple(_EXPECTED))
def test_eda_specs_have_exact_canonical_crossref_configuration(venue_id: str) -> None:
    expected = _EXPECTED[venue_id]
    spec = load_venue_spec(venue_id)

    assert spec.venue_id == venue_id
    assert spec.name == expected["name"]
    assert spec.venue_type is VenueType.JOURNAL
    assert spec.adapter == "adapters.crossref:CrossrefSerialAdapter"
    assert spec.enrichers == (
        "enrichers.semantic_scholar:SemanticScholarEnricher",
        "enrichers.openalex:OpenAlexEnricher",
    )
    assert spec.start_year == expected["start_year"]
    assert spec.end_year is None
    assert spec.held_years is None
    assert dict(spec.source) == {"issn": expected["issn"]}
    assert dict(spec.year_overrides) == {}


@pytest.mark.parametrize("venue_id", tuple(_EXPECTED))
def test_eda_specs_load_the_same_explicit_adapter_and_ordered_enrichers(
    venue_id: str,
) -> None:
    spec = load_venue_spec(venue_id)

    adapter = load_adapter(spec.adapter)
    enrichers = load_enrichers(spec.enrichers)

    assert type(adapter) is CrossrefSerialAdapter
    assert adapter.source_name == "crossref_serial"
    assert [enricher.source_name for enricher in enrichers] == [
        "semantic_scholar",
        "openalex",
    ]
