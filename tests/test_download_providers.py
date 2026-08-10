from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from paper_agent.domain import (
    AccessBasis,
    AccessLocationCandidate,
    DownloadResult,
    DownloadStatus,
    FetchDecision,
    FetchDecisionStatus,
    FetchRequest,
    Paper,
    PaperSource,
    PublicationVersion,
)
from paper_agent.download_providers import (
    DEFAULT_PROVIDER_ORDER,
    DEFAULT_RESOLVER_ORDER,
    AccessResolver,
    DownloadProviderDescriptor,
    DownloadProviderError,
    DownloadProviderRegistry,
    FetchContext,
    MatchedArxivResolver,
    PersistedRequestDownloadProvider,
    ProbeContext,
    ResolverContext,
    ResolverDescriptor,
    ResolverEvidence,
    ResolverRegistry,
    RoutedDownloadProvider,
    UnavailableDownloadProvider,
    default_download_provider_registry,
    default_resolver_registry,
    provider_contract,
)


FIXTURES = Path(__file__).parent / "fixtures" / "download_providers"
NOW = "2026-08-10T00:00:00Z"


def evidence_fixture() -> dict[str, ResolverEvidence]:
    raw = json.loads((FIXTURES / "resolver-evidence.json").read_text(encoding="utf-8"))
    return {
        name: ResolverEvidence(
            payload=value["payload"],
            raw_evidence_hash=value["raw_evidence_hash"],
            retrieved_at=value["retrieved_at"],
        )
        for name, value in raw.items()
    }


class FixtureLookup:
    def __init__(self, responses: dict[str, ResolverEvidence]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, str]] = []

    def __call__(self, resolver: str, paper: Paper) -> ResolverEvidence:
        self.calls.append((resolver, paper.paper_id))
        return self.responses[resolver]


def paper() -> Paper:
    return Paper("paper-1", "Paper", doi="10.1000/test", arxiv_id="2501.01234v2")


def official_source(*, basis: AccessBasis = AccessBasis.OPEN_LICENSE) -> PaperSource:
    return PaperSource(
        "official-1",
        "paper-1",
        "venue_primary",
        "official-1",
        landing_url="https://venue.example/paper",
        pdf_url="https://venue.example/paper.pdf",
        publication_version=PublicationVersion.PUBLISHED,
        license="CC-BY-4.0",
        access_basis=basis,
    )


def test_default_resolver_order_uses_only_metadata_evidence_and_preserves_license_boundaries() -> None:
    lookup = FixtureLookup(evidence_fixture())

    candidates = default_resolver_registry().resolve(
        ResolverContext(
            paper=paper(),
            official_sources=(official_source(),),
            lookup=lookup,
            matched_arxiv=True,
            retrieved_at=NOW,
        )
    )

    assert DEFAULT_RESOLVER_ORDER == default_resolver_registry().names
    assert [candidate.resolver for candidate in candidates] == [
        "publisher_public",
        "europe_pmc",
        "unpaywall",
        "unpaywall",
        "arxiv",
    ]
    assert lookup.calls == [
        ("europe_pmc", "paper-1"),
        ("unpaywall", "paper-1"),
        ("arxiv", "paper-1"),
    ]
    bronze, repository = [candidate for candidate in candidates if candidate.resolver == "unpaywall"]
    assert bronze.access_basis is AccessBasis.PUBLIC_READ_ONLY
    assert repository.access_basis is AccessBasis.PUBLIC_READ_ONLY
    assert all(candidate.host for candidate in candidates)
    assert all(candidate.raw_evidence_hash is not None for candidate in candidates)


def test_official_subscription_source_is_preserved_without_being_promoted_to_public() -> None:
    lookup = FixtureLookup(evidence_fixture())
    candidates = default_resolver_registry().resolve(
        ResolverContext(
            paper=paper(), official_sources=(official_source(basis=AccessBasis.USER_SUBSCRIPTION),), lookup=lookup
        )
    )

    publisher = next(candidate for candidate in candidates if candidate.resolver == "publisher_public")
    assert publisher.access_basis is AccessBasis.USER_SUBSCRIPTION
    assert publisher.raw_evidence_hash is not None
    assert all(candidate.resolver != "arxiv" for candidate in candidates)
    assert MatchedArxivResolver().resolve(
        ResolverContext(paper=paper(), lookup=lookup, matched_arxiv=True)
    )[-1].url == "https://arxiv.org/pdf/2501.01234v2"


def test_arxiv_base_identifier_accepts_the_versioned_feed_record() -> None:
    lookup = FixtureLookup(evidence_fixture())
    value = MatchedArxivResolver().resolve(
        ResolverContext(
            paper=Paper("paper-1", "Paper", arxiv_id="2501.01234"),
            lookup=lookup,
            matched_arxiv=True,
        )
    )

    assert value[-1].url == "https://arxiv.org/pdf/2501.01234v2"


def test_arxiv_old_style_identifier_keeps_its_archive_prefix() -> None:
    evidence = ResolverEvidence(
        payload={"feed": {"entry": {"id": "https://arxiv.org/abs/hep-th/9901001v2"}}},
        retrieved_at=NOW,
    )
    value = MatchedArxivResolver().resolve(
        ResolverContext(
            paper=Paper("paper-1", "Paper", arxiv_id="hep-th/9901001"),
            lookup=FixtureLookup({"arxiv": evidence}),
            matched_arxiv=True,
        )
    )

    assert value[-1].url == "https://arxiv.org/pdf/hep-th/9901001v2"


@dataclass
class RecordingService:
    policy: object
    probes: list[tuple[AccessLocationCandidate, dict]]
    fetches: list[tuple[FetchRequest, dict]]

    def probe(self, candidate: AccessLocationCandidate, **kwargs) -> FetchDecision:
        self.probes.append((candidate, kwargs))
        return FetchDecision(candidate.candidate_id, FetchDecisionStatus.MANUAL, "fixture", "v1")

    def fetch(self, request: FetchRequest, **kwargs) -> DownloadResult:
        self.fetches.append((request, kwargs))
        return DownloadResult(request.request_id, "paper-1", DownloadStatus.DOWNLOADED, request.provider)


def candidate(resolver: str = "fixture") -> AccessLocationCandidate:
    return AccessLocationCandidate(
        "candidate-1", "paper-1", resolver, "https://example.test/paper.pdf", host="example.test"
    )


def request(provider: str = "fixture") -> FetchRequest:
    return FetchRequest("request-1", "candidate-1", "v1", "personal_research", provider, NOW, "2026-08-11T00:00:00Z", "key")


def descriptor(name: str, provider: RoutedDownloadProvider, handles) -> DownloadProviderDescriptor:
    return DownloadProviderDescriptor(name, provider, handles, provider_contract())


def test_persisted_adapter_probes_without_fetching_and_fetch_requires_bound_provider() -> None:
    service = RecordingService(policy=type("Policy", (), {"version": "v1"})(), probes=[], fetches=[])
    adapter = PersistedRequestDownloadProvider("fixture", service)  # type: ignore[arg-type]
    probe_context = ProbeContext("personal_research", NOW, run_id="run-1")

    decision = adapter.probe(candidate(), probe_context)

    assert decision.status is FetchDecisionStatus.MANUAL
    assert len(service.probes) == 1
    assert service.fetches == []
    with pytest.raises(DownloadProviderError, match="different provider"):
        adapter.fetch(request("other"), FetchContext("run-1", NOW))
    result = adapter.fetch(request(), FetchContext("run-1", NOW))
    assert result.status is DownloadStatus.DOWNLOADED
    assert service.fetches[0][1]["run_id"] == "run-1"


def test_registry_order_and_placeholder_boundaries_are_explicit_and_extensible() -> None:
    manual = UnavailableDownloadProvider("manual", "v1", "manual_queue_required")
    registry = DownloadProviderRegistry(
        (descriptor("manual", manual, lambda _candidate: True),)
    )

    attempt = registry.probe(candidate(), ProbeContext("personal_research", NOW))

    assert attempt.provider == "manual"
    assert attempt.decision.status is FetchDecisionStatus.MANUAL
    with pytest.raises(DownloadProviderError, match="cannot fetch"):
        registry.fetch(request("manual"), FetchContext("run-1", NOW))
    with pytest.raises(DownloadProviderError, match="unknown download provider"):
        registry.probe_with("missing", candidate(), ProbeContext("personal_research", NOW))

    class ExtraResolver:
        name = "extra"

        def resolve(self, context: ResolverContext) -> tuple[AccessLocationCandidate, ...]:
            return (candidate("extra"),)

    resolver_registry = ResolverRegistry((ResolverDescriptor("extra", ExtraResolver()),))
    assert resolver_registry.resolve(ResolverContext(paper=paper()))[0].resolver == "extra"
    assert DEFAULT_PROVIDER_ORDER == (
        "public_direct", "europe_pmc", "unpaywall_location", "arxiv", "authorized_skill", "manual"
    )


def test_default_download_descriptors_declare_complete_contracts() -> None:
    registry = default_download_provider_registry(SimpleNamespace(policy=SimpleNamespace(version="v1")))

    assert registry.names == DEFAULT_PROVIDER_ORDER
    for name in registry.names:
        contract = registry.descriptor(name).contract
        assert set(contract) == {
            "authentication_required", "supports_main_document", "supports_supplements",
            "supports_version_selection", "allows_unattended", "handled_domains",
            "handled_resolvers", "retry_semantics", "probe_input_schema_id",
            "probe_output_schema_id", "fetch_input_schema_id", "fetch_output_schema_id",
            "idempotency_key_boundary", "side_effect_boundary",
        }
        assert contract["probe_input_schema_id"].endswith("probe-input.v1")
        assert contract["fetch_output_schema_id"].endswith("fetch-output.v1")
    assert registry.descriptor("authorized_skill").contract["authentication_required"] is True
    assert registry.descriptor("authorized_skill").contract["allows_unattended"] is False
    assert registry.descriptor("manual").contract["supports_main_document"] is False


@pytest.mark.parametrize("mutate", (
    lambda contract: contract.pop("retry_semantics"),
    lambda contract: contract.__setitem__("unexpected", "value"),
    lambda contract: contract.__setitem__("side_effect_boundary", "fetch_any_url"),
))
def test_registry_rejects_incomplete_or_unsafe_extension_contracts(mutate) -> None:
    provider = UnavailableDownloadProvider("extension", "v1", "manual")
    contract = provider_contract(handled_resolvers=("fixture",), handled_domains=("example.test",))
    mutate(contract)

    with pytest.raises(DownloadProviderError):
        DownloadProviderRegistry((DownloadProviderDescriptor("extension", provider, lambda _candidate: True, contract),))


def test_extension_descriptor_routes_only_its_declared_domain_and_resolver() -> None:
    provider = UnavailableDownloadProvider("extension", "v1", "manual")
    registry = DownloadProviderRegistry((
        DownloadProviderDescriptor(
            "extension", provider, lambda _candidate: True,
            provider_contract(handled_resolvers=("fixture",), handled_domains=("example.test",)),
        ),
    ))

    attempt = registry.probe(candidate(), ProbeContext("personal_research", NOW))

    assert attempt.provider == "extension"
    with pytest.raises(DownloadProviderError, match="no download provider accepts resolver"):
        registry.probe(candidate("other"), ProbeContext("personal_research", NOW))
    with pytest.raises(DownloadProviderError, match="does not handle other"):
        registry.probe_with("extension", candidate("other"), ProbeContext("personal_research", NOW))
