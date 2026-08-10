"""Strict, read-only contracts between Stage 1 providers and the coordinator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence, runtime_checkable

from paper_agent.domain import (
    AccessLocationCandidate,
    CitationBatch,
    DownloadResult,
    FetchDecision,
    FetchRequest,
    Paper,
    ProviderCapability,
    ProviderRole,
    QuerySpec,
    SourceBatch,
    SourceEntry,
    VerificationStatus,
)


@dataclass(frozen=True, slots=True)
class VenueDescriptor:
    schema_version: int
    venue_id: str
    provider: str
    adapter: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CrawlWindow:
    date_from: str | None = None
    date_to: str | None = None
    year: int | None = None
    volume: str | None = None
    issue: str | None = None


@dataclass(frozen=True, slots=True)
class SeedInput:
    kind: str
    value: str
    source_name: str | None = None


@dataclass(frozen=True, slots=True)
class IdentityCandidate:
    title: str
    authors: tuple[str, ...] = ()
    year: int | None = None
    doi: str | None = None
    arxiv_id: str | None = None


@dataclass(frozen=True, slots=True)
class EnrichmentResult:
    entry: SourceEntry
    provider: str
    source_run_id: str
    raw_response_artifact_hash: str | None = None


@dataclass(frozen=True, slots=True)
class VerificationResult:
    candidate: IdentityCandidate
    status: VerificationStatus
    provider: str
    evidence: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AccessPolicy:
    purpose: str
    allowed_access_bases: tuple[str, ...] = ()
    allow_browser: bool = False


@dataclass(frozen=True, slots=True)
class CredentialPolicy:
    required: bool = False
    environment_variables: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RateLimitPolicy:
    queries_per_second: float | None = None
    max_concurrency: int = 1
    cache_ttl_seconds: int | None = None


@dataclass(frozen=True, slots=True)
class ProviderManifest:
    provider: str
    version: str
    roles: tuple[ProviderRole, ...]
    capabilities: tuple[ProviderCapability, ...]
    stable_identifier: str
    distribution: str | None = None
    entry_point: str | None = None
    artifact_sha256: str | None = None
    enabled: bool = True
    builtin: bool = True
    authority: str = "scholarly_graph"
    credential_policy: CredentialPolicy = CredentialPolicy()
    rate_limit_policy: RateLimitPolicy = RateLimitPolicy()
    terms_url: str | None = None
    independence_group: str | None = None
    upstream_families: tuple[str, ...] = ()

    def supports(self, capability: ProviderCapability) -> bool:
        return capability in self.capabilities


def validate_source_batch(batch: SourceBatch) -> SourceBatch:
    if not batch.source_run_id or not batch.query_hash:
        raise ValueError("source batches require source_run_id and query_hash")
    if batch.status.value == "failed" and not batch.error:
        raise ValueError("failed source batches require an error")
    if batch.status.value == "success" and batch.error:
        raise ValueError("successful source batches may not carry an error")
    return batch


def validate_citation_batch(batch: CitationBatch) -> CitationBatch:
    if not batch.source_run_id or not batch.query_hash:
        raise ValueError("citation batches require source_run_id and query_hash")
    if batch.status.value == "failed" and not batch.error:
        raise ValueError("failed citation batches require an error")
    if batch.status.value == "success" and batch.error:
        raise ValueError("successful citation batches may not carry an error")
    return batch


@runtime_checkable
class VenueAdapter(Protocol):
    manifest: ProviderManifest

    def discover(self, descriptor: VenueDescriptor, window: CrawlWindow, cursor: str | None) -> SourceBatch: ...


@runtime_checkable
class SearchProvider(Protocol):
    manifest: ProviderManifest

    def search(self, query_spec: QuerySpec, cursor: str | None) -> SourceBatch: ...


@runtime_checkable
class CitationProvider(Protocol):
    manifest: ProviderManifest

    def references(self, seed: Paper, cursor: str | None) -> CitationBatch: ...

    def citations(self, seed: Paper, cursor: str | None) -> CitationBatch: ...


@runtime_checkable
class LibraryProvider(Protocol):
    manifest: ProviderManifest

    def import_seeds(self, input_spec: Sequence[SeedInput]) -> SourceBatch: ...


@runtime_checkable
class MetadataEnricher(Protocol):
    manifest: ProviderManifest

    def enrich(self, raw_paper: SourceEntry) -> EnrichmentResult: ...


@runtime_checkable
class MetadataVerifier(Protocol):
    manifest: ProviderManifest

    def verify(self, identity_candidate: IdentityCandidate, evidence: Sequence[SourceEntry]) -> VerificationResult: ...


@runtime_checkable
class OpenAccessResolver(Protocol):
    manifest: ProviderManifest

    def resolve(self, paper: Paper, policy: AccessPolicy) -> list[AccessLocationCandidate]: ...


@runtime_checkable
class DownloadProvider(Protocol):
    manifest: ProviderManifest

    def probe(self, candidate: AccessLocationCandidate, policy: AccessPolicy) -> FetchDecision: ...

    def fetch(self, request: FetchRequest, authorization_context: object) -> DownloadResult: ...
