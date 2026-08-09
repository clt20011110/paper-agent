from importlib import import_module
import json
from pathlib import Path

from paper_agent.manifests import load_catalog
from paper_agent.providers.builtin import FixtureTransport, create_builtin
from paper_agent.providers.api import CrawlWindow
from paper_agent.schema import validate


ROOT = Path(__file__).parents[1]


PRIMARY = {
    "neurips": "neurips_proceedings",
    "icml": "pmlr",
    "iclr": "openreview",
    "aaai": "aaai_ojs",
    "acl": "acl_anthology",
    "cvpr": "cvf_open_access",
    "iccv": "cvf_open_access",
    "ijcai": "ijcai_proceedings",
    "dac": "eda_proceedings",
    "iccad": "eda_proceedings",
    "tcad": "ieee_xplore",
    "nature_machine_intelligence": "springer_nature",
    "nature_chemistry": "springer_nature",
    "nature_computational_science": "springer_nature",
    "nature_communications": "springer_nature",
    "nature_catalysis": "springer_nature",
    "nature_biotechnology": "springer_nature",
    "nature_biomedical_engineering": "springer_nature",
    "cell": "cell_press",
    "science": "aaas_science",
}

FALLBACKS = {
    "neurips": ["openreview", "crossref", "dblp", "semantic_scholar", "openalex"],
    "icml": ["openreview", "crossref", "dblp", "semantic_scholar", "openalex"],
    "iclr": ["arxiv", "dblp", "semantic_scholar", "openalex"],
    "aaai": ["crossref", "dblp", "semantic_scholar", "openalex"],
    "acl": ["crossref", "dblp", "semantic_scholar", "openalex"],
    "cvpr": ["ieee_xplore", "crossref", "dblp", "semantic_scholar", "openalex"],
    "iccv": ["ieee_xplore", "crossref", "dblp", "semantic_scholar", "openalex"],
    "ijcai": ["crossref", "dblp", "semantic_scholar", "openalex"],
    "dac": ["ieee_xplore", "crossref", "dblp", "semantic_scholar", "openalex"],
    "iccad": ["ieee_xplore", "crossref", "dblp", "semantic_scholar", "openalex"],
    "tcad": ["crossref", "dblp", "semantic_scholar", "openalex"],
    "nature_machine_intelligence": ["crossref"],
    "nature_chemistry": ["crossref", "pubmed", "europe_pmc"],
    "nature_computational_science": ["crossref"],
    "nature_communications": ["crossref", "pubmed", "europe_pmc"],
    "nature_catalysis": ["crossref", "pubmed", "europe_pmc"],
    "nature_biotechnology": ["crossref", "pubmed", "europe_pmc"],
    "nature_biomedical_engineering": ["crossref", "pubmed", "europe_pmc"],
    "cell": ["crossref", "pubmed", "europe_pmc", "semantic_scholar", "openalex"],
    "science": ["crossref", "pubmed", "europe_pmc", "semantic_scholar", "openalex"],
}


def test_built_in_manifests_are_schema_valid_and_unique() -> None:
    catalog = load_catalog(ROOT)
    assert set(catalog.venues) == set(PRIMARY)
    assert set(catalog.acceptances) == set(PRIMARY)
    assert len(catalog.providers) == 25
    for path in (ROOT / "providers").glob("*.yaml"):
        validate(catalog.providers[path.stem], "provider-manifest.schema.json")
    for path in (ROOT / "venues").glob("*.yaml"):
        validate(catalog.venues[path.stem], "venue-descriptor.schema.json")
    for path in (ROOT / "acceptance").glob("*.yaml"):
        validate(catalog.acceptances[path.stem], "acceptance-manifest.schema.json")


def test_venue_primary_and_fallbacks_are_manifest_driven() -> None:
    catalog = load_catalog(ROOT)
    assert {venue_id: venue["primary_provider"] for venue_id, venue in catalog.venues.items()} == PRIMARY
    for venue_id, acceptance in catalog.acceptances.items():
        provider = catalog.provider(acceptance["primary_provider"])
        assert provider["enabled"]
        assert "venue_primary" in provider["roles"]
        assert set(acceptance["required_capabilities"]).issubset(provider["capabilities"])
        for fallback in acceptance["fallbacks"]:
            assert fallback["role"] in catalog.provider(fallback["provider"])["roles"]
        assert [fallback["provider"] for fallback in acceptance["fallbacks"]] == FALLBACKS[venue_id]


def test_optional_discovery_providers_are_disabled() -> None:
    catalog = load_catalog(ROOT)
    for provider in ("exa", "gemini_search", "deepxiv", "alphaxiv"):
        manifest = catalog.provider(provider)
        assert not manifest["enabled"]
        assert not manifest["builtin"]
        assert manifest["artifact_sha256"] is None
        assert manifest["authentication"]["required"]


def test_enabled_builtin_entry_points_and_artifacts_are_loadable() -> None:
    catalog = load_catalog(ROOT)
    for manifest in catalog.providers.values():
        if not (manifest["enabled"] and manifest["builtin"]):
            continue
        module_name, attribute = manifest["entry_point"].split(":", 1)
        assert getattr(import_module(module_name), attribute)


def test_every_venue_acceptance_fixture_runs_through_its_declared_adapter() -> None:
    catalog = load_catalog(ROOT)
    fixture_payload = json.loads((ROOT / "tests/fixtures/providers/official-page-1.json").read_text())
    responses = {
        f"{provider}:discover:first": fixture_payload
        for provider in set(PRIMARY.values())
    }
    responses["openreview:resolve_invitation:first"] = {
        "invitation": "ICLR.cc/2025/Conference/-/Decision",
        "api_version": "v2",
    }
    responses["pmlr:resolve_volume:first"] = {"official_url": "https://proceedings.mlr.press/v235/"}
    transport = FixtureTransport(responses)

    for venue_id in sorted(PRIMARY):
        descriptor = catalog.runtime_venue(venue_id)
        batch = create_builtin(descriptor.provider, transport).discover(
            descriptor,
            CrawlWindow(year=2025),
        )
        acceptance = catalog.acceptance(venue_id)
        assert [entry.external_id for entry in batch.entries] == acceptance["expected_stable_ids"]
        assert batch.entries[0].metadata["official_membership"] is True
        assert batch.entries[0].metadata["venue_id"] == venue_id


def test_frozen_journal_identifiers_and_venue_constraints() -> None:
    catalog = load_catalog(ROOT)
    expected_journals = {
        "nature_machine_intelligence": ("natmachintell", ["2522-5839"]),
        "nature_chemistry": ("nchem", ["1755-4330", "1755-4349"]),
        "nature_computational_science": ("natcomputsci", ["2662-8457"]),
        "nature_communications": ("ncomms", ["2041-1723"]),
        "nature_catalysis": ("natcatal", ["2520-1158"]),
        "nature_biotechnology": ("nbt", ["1087-0156", "1546-1696"]),
        "nature_biomedical_engineering": ("natbiomedeng", ["2157-846X"]),
        "cell": ("cell", ["0092-8674"]),
        "science": ("science", ["0036-8075", "1095-9203"]),
        "tcad": ("ieee-tcad", ["0278-0070"]),
    }
    for venue_id, (slug, issns) in expected_journals.items():
        journal = catalog.acceptance(venue_id)["journal"]
        assert journal["slug"] == slug
        assert journal["issns"] == issns
    assert catalog.venue("iclr")["provider_params"]["accepted_decision_required"]
    assert catalog.venue("acl")["provider_params"]["collections"] == ["main", "findings", "workshop"]
    assert catalog.venue("cvpr")["provider_params"]["exclude_workshops"]
    assert catalog.venue("iccv")["provider_params"]["exclude_workshops"]
    for venue_id in ("dac", "iccad"):
        assert catalog.venue(venue_id)["provider_params"]["deduplicate_by"] == "doi"
    assert catalog.venue("tcad")["provider_params"]["publication_number"] == 43
