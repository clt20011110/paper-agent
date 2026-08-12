from dataclasses import replace
from email.message import Message
from hashlib import sha256
from importlib import import_module
import json
from pathlib import Path

from paper_agent.http_transport import ControlledHTTPTransport
from paper_agent.manifests import load_catalog
from paper_agent.provider_runtime import ProviderRuntime, policy_from_manifest
from paper_agent.providers.builtin import (
    BUILTIN_CLASSES,
    FixtureTransport,
    create_builtin,
    manifest_from_document,
)
from paper_agent.providers.factory import create_core_provider
from paper_agent.providers.api import CrawlWindow, VenueDescriptor
from paper_agent.schema import validate


ROOT = Path(__file__).parents[1]


PRIMARY = {
    "aistats": "pmlr",
    "uai": "pmlr",
    "jmlr": "jmlr_official",
    "colt": "pmlr",
    "corl": "pmlr",
    "neurips": "neurips_proceedings",
    "icml": "pmlr",
    "iclr": "openreview",
    "aaai": "aaai_ojs",
    "acl": "acl_anthology",
    "cvpr": "cvf_open_access",
    "emnlp": "acl_anthology",
    "naacl": "acl_anthology",
    "coling": "acl_anthology",
    "iccv": "cvf_open_access",
    "ijcai": "ijcai_proceedings",
    "dac": "eda_proceedings",
    "iccad": "eda_proceedings",
    "tcad": "crossref_serial",
    "nature_machine_intelligence": "crossref_serial",
    "nature_synthesis": "crossref_serial",
    "nature_chemistry": "crossref_serial",
    "jacs": "crossref_serial",
    "angewandte_chemie": "crossref_serial",
    "chemical_science": "crossref_serial",
    "acs_central_science": "crossref_serial",
    "jcim": "crossref_serial",
    "jctc": "crossref_serial",
    "nature_computational_science": "crossref_serial",
    "nature_communications": "crossref_serial",
    "nature_catalysis": "crossref_serial",
    "nature_biotechnology": "crossref_serial",
    "nature_biomedical_engineering": "crossref_serial",
    "cell": "crossref_serial",
    "science": "crossref_serial",
}

FALLBACKS = {
    "aistats": ["openreview", "crossref", "dblp", "semantic_scholar", "openalex"],
    "uai": ["openreview", "crossref", "dblp", "semantic_scholar", "openalex"],
    "jmlr": ["crossref", "dblp", "semantic_scholar", "openalex"],
    "colt": ["crossref", "dblp", "semantic_scholar", "openalex"],
    "corl": ["crossref", "dblp", "semantic_scholar", "openalex"],
    "neurips": ["openreview", "crossref", "dblp", "semantic_scholar", "openalex"],
    "icml": ["openreview", "crossref", "dblp", "semantic_scholar", "openalex"],
    "iclr": ["arxiv", "dblp", "semantic_scholar", "openalex"],
    "aaai": ["crossref", "dblp", "semantic_scholar", "openalex"],
    "acl": ["crossref", "dblp", "semantic_scholar", "openalex"],
    "cvpr": ["ieee_xplore", "crossref", "dblp", "semantic_scholar", "openalex"],
    "emnlp": ["crossref", "dblp", "semantic_scholar", "openalex"],
    "naacl": ["crossref", "dblp", "semantic_scholar", "openalex"],
    "coling": ["crossref", "dblp", "semantic_scholar", "openalex"],
    "iccv": ["ieee_xplore", "crossref", "dblp", "semantic_scholar", "openalex"],
    "ijcai": ["crossref", "dblp", "semantic_scholar", "openalex"],
    "dac": ["ieee_xplore", "crossref", "dblp", "semantic_scholar", "openalex"],
    "iccad": ["ieee_xplore", "crossref", "dblp", "semantic_scholar", "openalex"],
    "tcad": ["ieee_xplore", "crossref", "dblp", "semantic_scholar", "openalex"],
    "nature_machine_intelligence": ["springer_nature", "crossref", "semantic_scholar", "openalex"],
    "nature_synthesis": ["springer_nature", "crossref", "semantic_scholar", "openalex"],
    "nature_chemistry": ["springer_nature", "crossref", "pubmed", "europe_pmc"],
    "jacs": ["crossref", "semantic_scholar", "openalex"],
    "angewandte_chemie": ["crossref", "semantic_scholar", "openalex"],
    "chemical_science": ["crossref", "semantic_scholar", "openalex"],
    "acs_central_science": ["crossref", "semantic_scholar", "openalex"],
    "jcim": ["crossref", "semantic_scholar", "openalex"],
    "jctc": ["crossref", "semantic_scholar", "openalex"],
    "nature_computational_science": ["springer_nature", "crossref"],
    "nature_communications": ["springer_nature", "crossref", "pubmed", "europe_pmc"],
    "nature_catalysis": ["springer_nature", "crossref", "pubmed", "europe_pmc"],
    "nature_biotechnology": ["springer_nature", "crossref", "pubmed", "europe_pmc"],
    "nature_biomedical_engineering": ["springer_nature", "crossref", "pubmed", "europe_pmc"],
    "cell": ["cell_press", "crossref", "pubmed", "europe_pmc", "semantic_scholar", "openalex"],
    "science": ["aaas_science", "crossref", "pubmed", "europe_pmc", "semantic_scholar", "openalex"],
}


def _fixture_venue_ids(value: object) -> list[str]:
    if isinstance(value, dict):
        return [
            *([str(value["venue_id"])] if "venue_id" in value else []),
            *(venue_id for item in value.values() for venue_id in _fixture_venue_ids(item)),
        ]
    if isinstance(value, list):
        return [venue_id for item in value for venue_id in _fixture_venue_ids(item)]
    return []


class _NativeFixtureResponse:
    def __init__(self, body: bytes, content_type: str) -> None:
        self.body = body
        self.headers = Message()
        self.headers["Content-Type"] = content_type

    def read(self) -> bytes:
        return self.body

    def __enter__(self) -> "_NativeFixtureResponse":
        return self

    def __exit__(self, *_: object) -> None:
        return None


class _AcceptanceFixtureOpener:
    def __init__(self, routes: list[dict[str, str]]) -> None:
        self.routes = routes
        self.used: set[str] = set()
        for route in routes:
            body = (ROOT / route["path"]).read_bytes()
            assert sha256(body).hexdigest() == route["sha256"]

    def __call__(self, request, timeout: float) -> _NativeFixtureResponse:
        matches = [route for route in self.routes if route["url_contains"] in request.full_url]
        assert matches, f"no acceptance fixture route for {request.full_url}"
        route = max(matches, key=lambda value: len(value["url_contains"]))
        self.used.add(route["url_contains"])
        return _NativeFixtureResponse((ROOT / route["path"]).read_bytes(), route["content_type"])


def _acceptance_runtime(provider: str, provider_document: dict[str, object]):
    provider_manifest = manifest_from_document(provider_document)
    base_policy = replace(
        policy_from_manifest(provider_manifest, terms_accepted=True, robots_allowed=True),
        queries_per_second=None,
        retry_attempts=1,
        jitter_seconds=0,
    )
    policies = {provider: base_policy}
    environment = {
        name: f"{provider}-fixture-secret"
        for name in provider_manifest.credential_policy.environment_variables
    }
    upstream_policies = provider_document.get("upstream_policies", {})
    assert isinstance(upstream_policies, dict)
    for upstream, document in upstream_policies.items():
        assert isinstance(upstream, str)
        assert isinstance(document, dict)
        authentication = document.get("authentication", {})
        assert isinstance(authentication, dict)
        credential_names = []
        if isinstance(authentication.get("credential_env"), str):
            credential_names.append(authentication["credential_env"])
        credential_envs = authentication.get("credential_envs", {})
        assert isinstance(credential_envs, dict)
        credential_names.extend(str(value) for value in credential_envs.values())
        policy_provider = f"{provider}:{upstream}"
        policies[policy_provider] = replace(
            base_policy,
            provider=policy_provider,
            credentials_required=bool(authentication.get("required", False)),
            credential_environment_variables=tuple(credential_names),
        )
        environment.update(
            {name: f"{policy_provider}-fixture-secret" for name in credential_names}
        )
    return ProviderRuntime(policies), environment, provider_manifest


def _assert_native_contract(entry, contract: str, year: int) -> None:
    if contract == "stable_id":
        assert entry.external_id
    elif contract == "metadata":
        assert entry.title and entry.metadata
    elif contract == "abstract":
        assert entry.abstract
    elif contract == "date_filter":
        assert entry.publication_date and entry.year == year
    else:
        raise AssertionError(f"native acceptance has no evidence check for {contract!r}")


def test_built_in_manifests_are_schema_valid_and_unique() -> None:
    catalog = load_catalog(ROOT)
    assert set(catalog.venues) == set(PRIMARY)
    assert set(catalog.acceptances) == set(PRIMARY)
    assert len(catalog.providers) == len(list((ROOT / "providers").glob("*.yaml")))
    assert len({acceptance["fixture_path"] for acceptance in catalog.acceptances.values()}) == len(PRIMARY)
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
        assert any(fallback["role"] == "search" for fallback in acceptance["fallbacks"])
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
    for venue_id in sorted(PRIMARY):
        acceptance = catalog.acceptance(venue_id)
        fixture_payload = json.loads((ROOT / acceptance["fixture_path"]).read_text())
        descriptor = catalog.runtime_venue(venue_id)
        provider = descriptor.provider
        responses = {f"{provider}:discover:first": fixture_payload}
        if provider == "openreview":
            responses["openreview:resolve_invitation:first"] = {
                "invitation": "ICLR.cc/2024/Conference/-/Decision",
                "api_version": "v2",
                "accepted_venue_ids": ["ICLR.cc/2024/Conference"],
            }
        if provider == "pmlr":
            responses["pmlr:resolve_volume:first"] = {
                "official_url": "https://proceedings.mlr.press/v235/"
            }
        transport = FixtureTransport(responses)
        start = acceptance["test_window"]["start"]
        end = acceptance["test_window"]["end"]
        window = CrawlWindow(date_from=start, date_to=end, year=int(start[:4]))
        batch = create_core_provider(
            descriptor.provider, transport, catalog.provider(descriptor.provider)
        ).discover(
            descriptor,
            window,
        )

        assert fixture_payload["fixture_venue_id"] == venue_id
        assert set(_fixture_venue_ids(fixture_payload)) == {venue_id}
        assert [entry.external_id for entry in batch.entries] == acceptance["expected_stable_ids"]
        assert batch.next_cursor == f"{venue_id}:page-2"
        for entry in batch.entries:
            assert entry.metadata["official_membership"] is (
                catalog.provider(provider)["authority"] == "official"
            )
            assert entry.metadata["venue_id"] == venue_id
            assert entry.title
            if "abstract" in acceptance["required_fields"]:
                assert entry.abstract
            if "date_filter" in acceptance["required_fields"]:
                assert entry.publication_date is not None
                assert start <= entry.publication_date <= end

        discover_call = next(call for call in transport.calls if call[1] == "discover")
        assert discover_call[2]["venue_id"] == venue_id
        assert discover_call[2]["date_from"] == start
        assert discover_call[2]["date_to"] == end
        assert discover_call[2]["year"] == int(start[:4])

        final_page = create_core_provider(
            descriptor.provider, transport, catalog.provider(descriptor.provider)
        ).discover(
            descriptor,
            window,
            batch.next_cursor,
        )
        assert final_page.entries == ()
        assert final_page.next_cursor is None


def test_every_venue_acceptance_runs_through_its_native_http_transport() -> None:
    catalog = load_catalog(ROOT)
    for venue_id in sorted(PRIMARY):
        acceptance = catalog.acceptance(venue_id)
        descriptor = catalog.runtime_venue(venue_id)
        provider_document = catalog.provider(descriptor.provider)
        runtime, environment, provider_manifest = _acceptance_runtime(
            descriptor.provider, provider_document
        )
        routes = acceptance["transport_fixture_routes"]
        opener = _AcceptanceFixtureOpener(routes)
        transport = ControlledHTTPTransport(
            "operator@example.test",
            opener=opener,
            runtime=runtime,
            environment=environment,
        )
        start = acceptance["test_window"]["start"]
        end = acceptance["test_window"]["end"]
        year = int(start[:4])

        batch = create_core_provider(
            descriptor.provider, transport, provider_document
        ).discover(
            descriptor,
            CrawlWindow(date_from=start, date_to=end, year=year),
        )

        assert batch.status.value == "success"
        assert [entry.external_id for entry in batch.entries] == acceptance["transport_expected_stable_ids"]
        assert opener.used == {route["url_contains"] for route in routes}
        assert batch.raw_response_artifact_hash
        for entry in batch.entries:
            assert entry.metadata["official_membership"] is (
                provider_document["authority"] == "official"
            )
            assert entry.metadata["venue_id"] == venue_id
            for contract in set(acceptance["required_fields"]) | set(
                acceptance["required_capabilities"]
            ):
                _assert_native_contract(entry, contract, year)


def test_platform_native_acceptance_shapes_select_only_expected_records() -> None:
    catalog = load_catalog(ROOT)

    expected_metadata = {
        "aaai": ("ojs_issue_id", "aaai-2024-issue-1"),
        "cvpr": ("cvf_track", "main"),
        "iccv": ("cvf_track", "main"),
    }
    for venue_id, (key, value) in expected_metadata.items():
        acceptance = catalog.acceptance(venue_id)
        payload = json.loads((ROOT / acceptance["fixture_path"]).read_text())
        descriptor = catalog.runtime_venue(venue_id)
        batch = create_builtin(
            descriptor.provider,
            FixtureTransport({f"{descriptor.provider}:discover:first": payload}),
        ).discover(descriptor, CrawlWindow(year=int(acceptance["test_window"]["start"][:4])))
        assert batch.entries[0].metadata[key] == value

    for venue_id in ("dac", "iccad"):
        acceptance = catalog.acceptance(venue_id)
        payload = json.loads((ROOT / acceptance["fixture_path"]).read_text())
        descriptor = catalog.runtime_venue(venue_id)
        batch = create_builtin(
            descriptor.provider,
            FixtureTransport({f"{descriptor.provider}:discover:first": payload}),
        ).discover(descriptor, CrawlWindow(year=2024))
        assert [entry.metadata["upstream"] for entry in batch.entries] == ["ieee_xplore", "acm_dl"]

    iclr = catalog.acceptance("iclr")
    payload = json.loads((ROOT / iclr["fixture_path"]).read_text())
    descriptor = catalog.runtime_venue("iclr")
    transport = FixtureTransport(
        {
            "openreview:resolve_invitation:first": {
                "invitation": "ICLR.cc/2024/Conference/-/Decision",
                "api_version": "v2",
                "accepted_venue_ids": ["ICLR.cc/2024/Conference"],
            },
            "openreview:discover:first": payload,
        }
    )
    batch = create_builtin("openreview", transport).discover(descriptor, CrawlWindow(year=2024))
    assert [entry.external_id for entry in batch.entries] == ["iclr-2024-openreview-0001"]


def test_same_platform_venues_are_yaml_descriptors_not_python_registrations() -> None:
    catalog = load_catalog(ROOT)
    shared_platforms = (
        ("cvpr", "iccv"),
        ("dac", "iccad"),
    )
    for venues in shared_platforms:
        assert len({catalog.venue(venue_id)["primary_provider"] for venue_id in venues}) == 1
        assert not set(venues).intersection(BUILTIN_CLASSES)

    descriptor = VenueDescriptor(
        1,
        "nature_methods",
        "springer_nature",
        "springer_nature",
        {
            "journal_slug": "nmeth",
            "issns": ["1548-7091", "1548-7105"],
            "article_types": ["article"],
        },
    )
    payload = {
        "entries": [
            {
                "stable_id": "10.1038/s41592-024-00001-1",
                "title": "YAML-only venue fixture",
                "abstract": "No venue-specific Python adapter is needed.",
                "publication_date": "2024-01-15",
                "venue_id": "nature_methods",
            }
        ]
    }
    batch = create_builtin(
        "springer_nature",
        FixtureTransport({"springer_nature:discover:first": payload}),
    ).discover(descriptor, CrawlWindow(year=2024))
    assert batch.entries[0].metadata["venue_id"] == "nature_methods"


def test_frozen_journal_identifiers_and_venue_constraints() -> None:
    catalog = load_catalog(ROOT)
    expected_journals = {
        "nature_machine_intelligence": ("natmachintell", ["2522-5839"]),
        "nature_chemistry": ("nchem", ["1755-4330", "1755-4349"]),
        "jacs": ("jacs", ["0002-7863", "1520-5126"]),
        "angewandte_chemie": ("angewandte-chemie", ["1433-7851", "1521-3773"]),
        "chemical_science": ("chemical-science", ["2041-6520", "2041-6539"]),
        "acs_central_science": ("acs-central-science", ["2374-7943", "2374-7951"]),
        "jcim": ("jcim", ["0095-2338", "1549-9596", "1549-960X"]),
        "jctc": ("jctc", ["1549-9618", "1549-9626"]),
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
        assert catalog.runtime_venue(venue_id).parameters == catalog.venue(venue_id)["provider_params"]
    assert catalog.venue("iclr")["provider_params"]["accepted_decision_required"]
    acl_parameters = catalog.venue("acl")["provider_params"]
    assert "collections" not in acl_parameters
    assert acl_parameters["tracks"] == [
        "main", "long", "short", "demo", "demos", "industry", "tutorial", "tutorials"
    ]
    acl_snapshot = catalog.venue("acl")["provider_params"]["snapshot_version"]
    assert acl_snapshot == "59602102d4cfc36e56554579e7dac2a762b56e92"
    acl_fixture = json.loads((ROOT / catalog.acceptance("acl")["fixture_path"]).read_text())
    assert acl_fixture["entries"][0]["snapshot_version"] == acl_snapshot
    assert catalog.venue("cvpr")["provider_params"]["exclude_workshops"]
    assert catalog.venue("iccv")["provider_params"]["exclude_workshops"]
    for venue_id in ("dac", "iccad"):
        assert catalog.venue(venue_id)["provider_params"]["deduplicate_by"] == "doi"
    assert catalog.venue("tcad")["provider_params"]["registry_issn"] == "0278-0070"
