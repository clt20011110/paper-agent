from __future__ import annotations

import json
from pathlib import Path

import pytest

from paper_agent.domain import (
    AccessBasis,
    EnvelopeStatus,
    Paper,
    ProviderCapability,
    ProviderRole,
    PublicationVersion,
    QuerySpec,
    SourceEntry,
    VerificationStatus,
)
from paper_agent.providers.api import AccessPolicy, IdentityCandidate
from paper_agent.providers.builtin import FixtureTransport, create_builtin


FIXTURES = Path(__file__).parent / "fixtures" / "providers"


def fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def envelope(name: str, provider: str, operation: str) -> dict:
    return {
        **fixture(name),
        "source_run_id": f"{provider}:{operation}:fixture",
        "raw_response_artifact_hash": f"{provider}-{operation}-response",
    }


ROLE_MATRIX = {
    "arxiv": {
        ProviderRole.METADATA_ENRICHER,
        ProviderRole.METADATA_VERIFIER,
        ProviderRole.OA_RESOLVER,
    },
    "crossref": {ProviderRole.METADATA_ENRICHER, ProviderRole.METADATA_VERIFIER},
    "dblp": {ProviderRole.METADATA_ENRICHER, ProviderRole.METADATA_VERIFIER},
    "semantic_scholar": {ProviderRole.METADATA_ENRICHER, ProviderRole.METADATA_VERIFIER},
    "openalex": {
        ProviderRole.METADATA_ENRICHER,
        ProviderRole.METADATA_VERIFIER,
        ProviderRole.OA_RESOLVER,
    },
    "pubmed": {ProviderRole.METADATA_ENRICHER, ProviderRole.METADATA_VERIFIER},
    "europe_pmc": {
        ProviderRole.METADATA_ENRICHER,
        ProviderRole.METADATA_VERIFIER,
        ProviderRole.OA_RESOLVER,
    },
    "unpaywall": {ProviderRole.OA_RESOLVER},
}


@pytest.mark.parametrize("provider", ROLE_MATRIX)
def test_manifest_role_and_capability_matrix_is_the_runtime_gate(provider: str) -> None:
    instance = create_builtin(provider, FixtureTransport({}))
    declared = set(instance.manifest.roles).intersection(
        {
            ProviderRole.METADATA_ENRICHER,
            ProviderRole.METADATA_VERIFIER,
            ProviderRole.OA_RESOLVER,
        }
    )

    assert declared == ROLE_MATRIX[provider]
    if declared.intersection({ProviderRole.METADATA_ENRICHER, ProviderRole.METADATA_VERIFIER}):
        assert instance.manifest.supports(ProviderCapability.METADATA)
    if ProviderRole.OA_RESOLVER in declared:
        assert instance.manifest.supports(ProviderCapability.OA_LOCATIONS)


ENRICH_CASES = (
    ("crossref", "crossref-work.json", "10.1000/crossref.work", "2025-01-01", "A singleton Crossref work response."),
    ("dblp", "dblp-native.json", "conf/icml/native-2025", None, None),
    ("semantic_scholar", "semantic-scholar-paper.json", "s2-paper-role-1", "2025-12-31", "A singleton Graph API paper response."),
    ("openalex", "openalex-work.json", "https://openalex.org/W9876543210", "2025-06-30", "Singleton OpenAlex work"),
    ("pubmed", "pubmed-esummary.json", "39900001", "2025-01-17", None),
    ("europe_pmc", "europe-pmc-record.json", "40000001", "2025-07-01", "A Europe PMC metadata and open-access response."),
    ("arxiv", "arxiv-atom.json", "2501.01234v2", "2025-01-15", "A decoded Atom summary."),
)

ENRICH_FIELDS = {
    "crossref": {
        "authors": ("Ada Lovelace", "Grace Hopper"),
        "doi": "10.1000/crossref.work",
        "arxiv_id": None,
        "venue_name": "Boundary Journal",
    },
    "dblp": {
        "authors": ("Ada Lovelace", "Grace Hopper"),
        "doi": "10.1000/dblp.native",
        "arxiv_id": None,
        "venue_name": "ICML",
    },
    "semantic_scholar": {
        "authors": ("Ada Lovelace", "Grace Hopper"),
        "doi": "10.1000/s2.role",
        "arxiv_id": "2512.99999",
        "venue_name": "Boundary Conference",
    },
    "openalex": {
        "authors": ("Ada Lovelace", "Grace Hopper"),
        "doi": "10.1000/openalex.work",
        "arxiv_id": "2506.12345",
        "venue_name": "Boundary Journal",
    },
    "pubmed": {
        "authors": ("Lovelace A", "Hopper G"),
        "doi": "10.1000/pubmed.native",
        "arxiv_id": None,
        "venue_name": "Nature",
    },
    "europe_pmc": {
        "authors": ("Ada Lovelace", "Grace Hopper"),
        "doi": "10.1000/epmc.role",
        "arxiv_id": None,
        "venue_name": "Boundary Journal",
    },
    "arxiv": {
        "authors": ("Ada Lovelace", "Grace Hopper"),
        "doi": "10.1000/arxiv.native",
        "arxiv_id": "2501.01234v2",
        "venue_name": "Machine Learning Journal",
    },
}


@pytest.mark.parametrize(
    ("provider", "fixture_name", "external_id", "publication_date", "abstract"),
    ENRICH_CASES,
)
def test_native_metadata_enricher_contract(
    provider: str,
    fixture_name: str,
    external_id: str,
    publication_date: str | None,
    abstract: str | None,
) -> None:
    payload = envelope(fixture_name, provider, "enrich")
    transport = FixtureTransport({f"{provider}:enrich:first": payload})
    raw = SourceEntry(
        provider="seed",
        external_id="seed-id",
        title="Seed title",
        doi="10.1000/seed",
        arxiv_id="2501.01234v2",
    )

    result = create_builtin(provider, transport).enrich(raw)

    assert result.provider == provider
    assert result.source_run_id == f"{provider}:enrich:fixture"
    assert result.raw_response_artifact_hash == f"{provider}-enrich-response"
    assert result.entry.provider == provider
    assert result.entry.external_id == external_id
    assert result.entry.publication_date == publication_date
    assert result.entry.abstract == abstract
    assert result.entry.title
    assert {
        "authors": result.entry.authors,
        "doi": result.entry.doi,
        "arxiv_id": result.entry.arxiv_id,
        "venue_name": result.entry.venue_name,
    } == ENRICH_FIELDS[provider]
    assert transport.calls == [
        (
            provider,
            "enrich",
            {"external_id": "seed-id", "doi": "10.1000/seed", "arxiv_id": "2501.01234v2"},
        )
    ]


VERIFY_CASES = (
    ("crossref", "crossref-native.json", "10.1000/crossref.native"),
    ("dblp", "dblp-native.json", "conf/icml/native-2025"),
    ("semantic_scholar", "semantic-scholar-paper.json", "s2-paper-role-1"),
    ("openalex", "openalex-work.json", "https://openalex.org/W9876543210"),
    ("pubmed", "pubmed-esummary.json", "39900001"),
    ("europe_pmc", "europe-pmc-record.json", "40000001"),
    ("arxiv", "arxiv-atom.json", "2501.01234v2"),
)


@pytest.mark.parametrize(("provider", "fixture_name", "external_id"), VERIFY_CASES)
def test_native_metadata_verifier_contract(provider: str, fixture_name: str, external_id: str) -> None:
    payload = envelope(fixture_name, provider, "verify")
    transport = FixtureTransport({f"{provider}:verify:first": payload})
    candidate = IdentityCandidate(
        "Candidate title",
        ("Ada Lovelace",),
        2025,
        "10.1000/candidate",
        "2501.01234v2",
    )

    result = create_builtin(provider, transport).verify(candidate, ())

    assert result.candidate == candidate
    assert result.status is VerificationStatus.SINGLE_SOURCE
    assert result.provider == provider
    assert result.evidence == (external_id,)
    assert transport.calls == [
        (
            provider,
            "verify",
            {
                "doi": "10.1000/candidate",
                "arxiv_id": "2501.01234v2",
                "title": "Candidate title",
            },
        )
    ]


OA_CASES = (
    (
        "openalex",
        "openalex-work.json",
        Paper("paper-openalex", "OpenAlex paper", doi="10.1000/openalex.work"),
        (
            (
                "https://publisher.test/openalex-work.pdf",
                "https://publisher.test/openalex-work",
                "Boundary Journal",
                PublicationVersion.PUBLISHED,
                AccessBasis.OPEN_LICENSE,
            ),
            (
                "https://repository.test/openalex-work",
                "https://repository.test/openalex-work",
                "Institutional Repository",
                PublicationVersion.ACCEPTED_MANUSCRIPT,
                AccessBasis.PUBLIC_READ_ONLY,
            ),
        ),
    ),
    (
        "europe_pmc",
        "europe-pmc-record.json",
        Paper("paper-europe-pmc", "Europe PMC paper", doi="10.1000/epmc.role"),
        (
            (
                "https://europepmc.org/articles/PMC7654321/bin/main.pdf",
                "https://europepmc.org/article/MED/40000001",
                "Europe_PMC",
                PublicationVersion.PUBLISHED,
                AccessBasis.PUBLIC_READ_ONLY,
            ),
        ),
    ),
    (
        "arxiv",
        "arxiv-atom.json",
        Paper("paper-arxiv", "arXiv paper", arxiv_id="2501.01234v2"),
        (
            (
                "http://arxiv.org/pdf/2501.01234v2",
                "http://arxiv.org/abs/2501.01234v2",
                "arxiv.org",
                PublicationVersion.PREPRINT,
                AccessBasis.PUBLIC_READ_ONLY,
            ),
        ),
    ),
    (
        "unpaywall",
        "unpaywall-native.json",
        Paper("paper-unpaywall", "Unpaywall paper", doi="10.1000/unpaywall.native"),
        (
            (
                "https://example.test/article.pdf",
                "https://example.test/article",
                "publisher",
                PublicationVersion.PUBLISHED,
                AccessBasis.OPEN_LICENSE,
            ),
            (
                "https://repository.test/item/1",
                "https://repository.test/item/1",
                "repository",
                PublicationVersion.ACCEPTED_MANUSCRIPT,
                AccessBasis.PUBLIC_READ_ONLY,
            ),
        ),
    ),
)


@pytest.mark.parametrize(("provider", "fixture_name", "paper", "expected"), OA_CASES)
def test_native_open_access_resolver_contract(
    provider: str,
    fixture_name: str,
    paper: Paper,
    expected: tuple[tuple[str, str, str, PublicationVersion, AccessBasis], ...],
) -> None:
    payload = envelope(fixture_name, provider, "resolve")
    transport = FixtureTransport({f"{provider}:resolve:first": payload})

    candidates = create_builtin(provider, transport).resolve(paper, AccessPolicy("research"))

    assert tuple(
        (
            candidate.url,
            candidate.landing_url,
            candidate.host,
            candidate.publication_version,
            candidate.access_basis,
        )
        for candidate in candidates
    ) == expected
    assert all(candidate.paper_id == paper.paper_id for candidate in candidates)
    assert all(candidate.resolver == provider for candidate in candidates)
    assert all(candidate.raw_evidence_hash == f"{provider}-resolve-response" for candidate in candidates)
    assert transport.calls == [
        (
            provider,
            "resolve",
            {
                "paper_id": paper.paper_id,
                "doi": paper.doi,
                "arxiv_id": paper.arxiv_id,
                "purpose": "research",
            },
        )
    ]


@pytest.mark.parametrize("provider", ("crossref", "dblp", "semantic_scholar", "pubmed"))
def test_oa_capability_does_not_bypass_missing_resolver_role(provider: str) -> None:
    instance = create_builtin(provider, FixtureTransport({}))
    with pytest.raises(ValueError, match="role oa_resolver"):
        instance.resolve(Paper("paper", "Paper", doi="10.1000/paper"), AccessPolicy("research"))


def test_unpaywall_does_not_gain_metadata_roles_from_its_metadata_capability() -> None:
    instance = create_builtin("unpaywall", FixtureTransport({}))
    raw = SourceEntry("seed", "seed", "Seed")
    candidate = IdentityCandidate("Seed")

    with pytest.raises(ValueError, match="role metadata_enricher"):
        instance.enrich(raw)
    with pytest.raises(ValueError, match="role metadata_verifier"):
        instance.verify(candidate, (raw,))


SEARCH_ENVELOPE_CASES = (
    ("crossref", "crossref-native.json", "DnF1ZXJ5VGhlbkZldGNoBQAAAAA=", "2025-02-03"),
    ("dblp", "dblp-native.json", "1", None),
    ("semantic_scholar", "semantic-scholar-search.json", "1", "2025-01-15"),
    ("openalex", "openalex-native.json", "IlsxMDAuMCwgJ2h0dHBzOi8vb3BlbmFsZXgub3JnL1cyJ10i", "2025-02-03"),
    ("pubmed", "pubmed-esummary.json", None, "2025-01-17"),
    ("europe_pmc", "europe-pmc-native.json", "AoIIP_4r0ig1NTQ0NTA0OA==", "2025-01-17"),
    ("arxiv", "arxiv-atom.json", "1", "2025-01-15"),
)


@pytest.mark.parametrize(
    ("provider", "fixture_name", "next_cursor", "publication_date"),
    SEARCH_ENVELOPE_CASES,
)
def test_native_search_envelope_keeps_pagination_and_date_mapping(
    provider: str,
    fixture_name: str,
    next_cursor: str | None,
    publication_date: str | None,
) -> None:
    payload = envelope(fixture_name, provider, "search")
    batch = create_builtin(
        provider,
        FixtureTransport({f"{provider}:search:first": payload}),
    ).search(QuerySpec(1, "role-contract", "fixture query"))

    assert batch.status is EnvelopeStatus.SUCCESS
    assert batch.source_run_id == f"{provider}:search:fixture"
    assert batch.raw_response_artifact_hash == f"{provider}-search-response"
    assert batch.next_cursor == next_cursor
    assert batch.entries[0].publication_date == publication_date
