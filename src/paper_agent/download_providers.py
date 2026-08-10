"""Extensible Stage 3 access-location resolvers and download-provider routing.

This module deliberately contains no PDF transport.  Resolvers consume
metadata evidence and produce untrusted :class:`AccessLocationCandidate`
objects; provider adapters delegate policy and persistence to
``DownloadService`` and fetch only an already-persisted ``FetchRequest``.
New resolvers and download providers are registered by descriptors, so their
addition does not require editing a central conditional chain.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
import re
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlsplit

from .canonical import content_hash
from .domain import (
    AccessBasis,
    AccessLocationCandidate,
    DownloadResult,
    FetchDecision,
    FetchDecisionStatus,
    FetchRequest,
    Paper,
    PaperSource,
    PublicationVersion,
)
from .downloads import AuthorizationContext, DownloadService


DEFAULT_RESOLVER_ORDER = (
    "publisher_public",
    "europe_pmc",
    "unpaywall",
    "arxiv",
)
DEFAULT_PROVIDER_ORDER = (
    "public_direct",
    "europe_pmc",
    "unpaywall_location",
    "arxiv",
    "authorized_skill",
    "manual",
)

PROBE_INPUT_SCHEMA_ID = "paper-agent.stage3.probe-input.v1"
PROBE_OUTPUT_SCHEMA_ID = "paper-agent.stage3.probe-output.v1"
FETCH_INPUT_SCHEMA_ID = "paper-agent.stage3.fetch-input.v1"
FETCH_OUTPUT_SCHEMA_ID = "paper-agent.stage3.fetch-output.v1"
_PROVIDER_CONTRACT_FIELDS = frozenset({
    "authentication_required",
    "supports_main_document",
    "supports_supplements",
    "supports_version_selection",
    "allows_unattended",
    "handled_domains",
    "handled_resolvers",
    "retry_semantics",
    "probe_input_schema_id",
    "probe_output_schema_id",
    "fetch_input_schema_id",
    "fetch_output_schema_id",
    "idempotency_key_boundary",
    "side_effect_boundary",
})
_RETRY_SEMANTICS = frozenset({"not_retryable", "transient_retryable", "external_ledger_resumable"})
_IDEMPOTENCY_KEY_BOUNDARY = "persisted_fetch_request_idempotency_key"
_SIDE_EFFECT_BOUNDARY = "probe_no_body_download__fetch_persisted_request_only"


class DownloadProviderError(ValueError):
    """A routing or adapter-boundary invariant was not met."""


@dataclass(frozen=True, slots=True)
class ResolverSnapshot:
    """Canonical identity for the exact ordered resolver registry in one run."""

    document: Mapping[str, Any]
    snapshot_hash: str

    def to_dict(self) -> dict[str, Any]:
        return deepcopy(dict(self.document))


@dataclass(frozen=True, slots=True)
class ResolverEvidence:
    """Metadata response retained by the coordinator, never a fetched PDF body."""

    payload: Mapping[str, Any]
    raw_evidence_hash: str | None = None
    retrieved_at: str | None = None


@runtime_checkable
class MetadataResolverTransport(Protocol):
    """A metadata-only lookup boundary for remote OA catalogues."""

    def __call__(self, resolver: str, paper: Paper) -> ResolverEvidence | None: ...

    def canonical_identity(self) -> Mapping[str, Any]:
        """Return the versioned registry contract used to produce lookups."""

        ...


@dataclass(frozen=True, slots=True)
class ResolverContext:
    """Coordinator-owned inputs available to an access-location resolver."""

    paper: Paper
    official_sources: tuple[PaperSource, ...] = ()
    lookup: MetadataResolverTransport | None = None
    matched_arxiv: bool = False
    include_arxiv_candidates: bool = False
    retrieved_at: str | None = None


@dataclass(frozen=True, slots=True)
class ProbeContext:
    purpose: str
    now: str
    authorization_grant_id: str | None = None
    mode: str = "attended"
    skill_digest: str | None = None
    dependency_digest: str | None = None
    collection_id: str | None = None
    collection_snapshot_hash: str | None = None
    selection_snapshot_hash: str | None = None
    run_id: str | None = None


@dataclass(frozen=True, slots=True)
class FetchContext:
    run_id: str
    now: str
    authorization_context: AuthorizationContext | None = None


@runtime_checkable
class AccessResolver(Protocol):
    name: str

    def resolve(self, context: ResolverContext) -> tuple[AccessLocationCandidate, ...]: ...


@runtime_checkable
class RoutedDownloadProvider(Protocol):
    name: str

    def probe(self, candidate: AccessLocationCandidate, context: ProbeContext) -> FetchDecision: ...

    def fetch(self, request: FetchRequest, context: FetchContext) -> DownloadResult: ...


def _candidate(
    paper: Paper,
    *,
    resolver: str,
    url: str,
    landing_url: str | None,
    publication_version: PublicationVersion,
    license: str | None,
    access_basis: AccessBasis,
    evidence: ResolverEvidence | None = None,
    retrieved_at: str | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> AccessLocationCandidate | None:
    """Construct a stable untrusted candidate, rejecting non-HTTP locations."""

    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return None
    candidate_id = "location-" + content_hash(
        {
            "paper_id": paper.paper_id,
            "resolver": resolver,
            "url": url,
            "publication_version": publication_version.value,
        }
    )[:32]
    return AccessLocationCandidate(
        candidate_id=candidate_id,
        paper_id=paper.paper_id,
        resolver=resolver,
        url=url,
        landing_url=landing_url,
        host=parsed.hostname.lower(),
        publication_version=publication_version,
        license=license,
        access_basis=access_basis,
        retrieved_at=(evidence.retrieved_at if evidence else None) or retrieved_at,
        raw_evidence_hash=(
            evidence.raw_evidence_hash or content_hash(evidence.payload)
            if evidence
            else content_hash(provenance or {})
        ),
        provenance=dict(provenance or {}),
    )


def _as_records(value: Any) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, Mapping):
        return (value,)
    if isinstance(value, list):
        return tuple(item for item in value if isinstance(item, Mapping))
    return ()


def _publication_version(value: Any) -> PublicationVersion:
    normalized = {
        "publishedVersion": "published",
        "acceptedVersion": "accepted_manuscript",
        "submittedVersion": "preprint",
    }.get(str(value), str(value or "unknown"))
    try:
        return PublicationVersion(normalized)
    except ValueError:
        return PublicationVersion.UNKNOWN


def _arxiv_identifier(value: str | None) -> str | None:
    if not value:
        return None
    normalized = value.strip().replace("arXiv:", "").replace("arxiv:", "")
    parsed = urlsplit(normalized)
    if parsed.scheme and parsed.hostname:
        path = parsed.path.rstrip("/")
        for prefix in ("/abs/", "/pdf/"):
            if path.startswith(prefix):
                normalized = path.removeprefix(prefix)
                break
    if normalized.endswith(".pdf"):
        normalized = normalized[:-4]
    return normalized or None


def _arxiv_identifier_matches(expected: str, observed: str) -> bool:
    expected_base = re.sub(r"v\d+$", "", expected)
    observed_base = re.sub(r"v\d+$", "", observed)
    return observed == expected if expected != expected_base else observed_base == expected_base


class OfficialPublicResolver:
    """Turn coordinator-confirmed official source URLs into candidates."""

    name = "publisher_public"

    def resolve(self, context: ResolverContext) -> tuple[AccessLocationCandidate, ...]:
        output: list[AccessLocationCandidate] = []
        for source in context.official_sources:
            if source.paper_id != context.paper.paper_id or not source.pdf_url:
                continue
            # An official source still has to positively describe public access;
            # a subscription URL must never be silently promoted to public.
            if source.access_basis not in {
                AccessBasis.OPEN_LICENSE,
                AccessBasis.PUBLIC_READ_ONLY,
                AccessBasis.USER_SUBSCRIPTION,
            }:
                continue
            evidence = ResolverEvidence(
                payload=source.to_dict(),
                retrieved_at=source.last_seen_at or context.retrieved_at,
            )
            candidate = _candidate(
                context.paper,
                resolver=self.name,
                url=source.pdf_url,
                landing_url=source.landing_url,
                publication_version=source.publication_version,
                license=source.license,
                access_basis=source.access_basis,
                evidence=evidence,
                retrieved_at=context.retrieved_at,
                provenance={"source_id": source.source_id, "source_provider": source.provider},
            )
            if candidate:
                output.append(candidate)
        return tuple(_dedupe(output))


class EuropePMCOpenAccessResolver:
    """Resolve only Europe PMC records explicitly marked open access."""

    name = "europe_pmc"

    def resolve(self, context: ResolverContext) -> tuple[AccessLocationCandidate, ...]:
        evidence = _lookup(context, self.name)
        if evidence is None:
            return ()
        result_list = evidence.payload.get("resultList")
        records = _as_records(result_list.get("result")) if isinstance(result_list, Mapping) else ()
        output: list[AccessLocationCandidate] = []
        for record in records:
            if str(record.get("isOpenAccess", "")).upper() != "Y":
                continue
            urls = record.get("fullTextUrlList")
            locations = _as_records(urls.get("fullTextUrl")) if isinstance(urls, Mapping) else ()
            landing_url = _europe_pmc_landing_url(record)
            for location in locations:
                if str(location.get("availability", "")).lower() != "open access":
                    continue
                url = _text(location.get("url"))
                if not url:
                    continue
                license = _text(location.get("license") or record.get("license"))
                candidate = _candidate(
                    context.paper,
                    resolver=self.name,
                    url=url,
                    landing_url=landing_url,
                    publication_version=PublicationVersion.PUBLISHED,
                    license=license,
                    access_basis=(AccessBasis.OPEN_LICENSE if license else AccessBasis.PUBLIC_READ_ONLY),
                    evidence=evidence,
                    retrieved_at=context.retrieved_at,
                    provenance={"pmcid": record.get("pmcid"), "site": location.get("site")},
                )
                if candidate:
                    output.append(candidate)
        return tuple(_dedupe(output))


class UnpaywallOpenAccessResolver:
    """Emit each Unpaywall OA location without treating bronze/null as licensed."""

    name = "unpaywall"

    def resolve(self, context: ResolverContext) -> tuple[AccessLocationCandidate, ...]:
        if not context.paper.doi:
            return ()
        evidence = _lookup(context, self.name)
        if evidence is None or evidence.payload.get("is_oa") is not True:
            return ()
        locations = list(_as_records(evidence.payload.get("oa_locations")))
        best = evidence.payload.get("best_oa_location")
        if isinstance(best, Mapping):
            locations.insert(0, best)
        output: list[AccessLocationCandidate] = []
        oa_status = str(evidence.payload.get("oa_status") or "").lower()
        for location in locations:
            url = _text(location.get("url_for_pdf") or location.get("url"))
            if not url:
                continue
            license = _text(location.get("license"))
            # Bronze is public-read-only even where a malformed response happens
            # to include a label.  Null licenses remain public-read-only as well.
            licensed = bool(license) and oa_status != "bronze"
            candidate = _candidate(
                context.paper,
                resolver=self.name,
                url=url,
                landing_url=_text(location.get("url_for_landing_page") or location.get("url")),
                publication_version=_publication_version(location.get("version")),
                license=license,
                access_basis=AccessBasis.OPEN_LICENSE if licensed else AccessBasis.PUBLIC_READ_ONLY,
                evidence=evidence,
                retrieved_at=context.retrieved_at,
                provenance={
                    "oa_status": oa_status or None,
                    "host_type": location.get("host_type"),
                    "endpoint_id": location.get("endpoint_id"),
                },
            )
            if candidate:
                output.append(candidate)
        return tuple(_dedupe(output))


class MatchedArxivResolver:
    """Use arXiv only for a matched paper or explicitly included arXiv candidate."""

    name = "arxiv"

    def resolve(self, context: ResolverContext) -> tuple[AccessLocationCandidate, ...]:
        expected = _arxiv_identifier(context.paper.arxiv_id)
        if not expected or not (context.matched_arxiv or context.include_arxiv_candidates):
            return ()
        evidence = _lookup(context, self.name)
        if evidence is None:
            return ()
        feed = evidence.payload.get("feed")
        entries = _as_records(feed.get("entry")) if isinstance(feed, Mapping) else ()
        output: list[AccessLocationCandidate] = []
        for entry in entries:
            entry_id = _arxiv_identifier(_text(entry.get("id")))
            if entry_id is None or not _arxiv_identifier_matches(expected, entry_id):
                continue
            landing_url = _text(entry.get("id"))
            url = landing_url.replace("/abs/", "/pdf/") if landing_url else None
            if not url:
                continue
            candidate = _candidate(
                context.paper,
                resolver=self.name,
                url=url,
                landing_url=landing_url,
                publication_version=PublicationVersion.PREPRINT,
                license=None,
                access_basis=AccessBasis.PUBLIC_READ_ONLY,
                evidence=evidence,
                retrieved_at=context.retrieved_at,
                provenance={"arxiv_id": expected, "matched": context.matched_arxiv},
            )
            if candidate:
                output.append(candidate)
        return tuple(_dedupe(output))


@dataclass(frozen=True, slots=True)
class ResolverDescriptor:
    name: str
    resolver: AccessResolver
    implementation_version: str = "resolver-v1"
    candidate_config: Mapping[str, Any] = field(default_factory=dict)


class ResolverRegistry:
    """Ordered resolver descriptors; registration is the extension mechanism."""

    def __init__(self, descriptors: Sequence[ResolverDescriptor] = ()) -> None:
        self._descriptors: dict[str, ResolverDescriptor] = {}
        for descriptor in descriptors:
            self.register(descriptor)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._descriptors)

    def descriptor(self, name: str) -> ResolverDescriptor:
        try:
            return self._descriptors[name]
        except KeyError as error:
            raise DownloadProviderError(f"unknown resolver: {name}") from error

    def register(self, descriptor: ResolverDescriptor) -> None:
        if not descriptor.name or descriptor.name != descriptor.resolver.name:
            raise DownloadProviderError("resolver descriptor name must match its resolver")
        if descriptor.name in self._descriptors:
            raise DownloadProviderError(f"duplicate resolver descriptor: {descriptor.name}")
        if (
            not isinstance(descriptor.implementation_version, str)
            or not descriptor.implementation_version.strip()
        ):
            raise DownloadProviderError(
                "resolver descriptor requires an implementation_version"
            )
        if not isinstance(descriptor.candidate_config, Mapping):
            raise DownloadProviderError("resolver candidate_config must be a mapping")
        content_hash(descriptor.candidate_config)
        self._descriptors[descriptor.name] = replace(
            descriptor,
            candidate_config=deepcopy(dict(descriptor.candidate_config)),
        )

    def freeze(
        self,
        *,
        configured_order: Sequence[str],
        runtime_config: Mapping[str, Mapping[str, Any]],
        download_config_hash: str,
    ) -> ResolverSnapshot:
        """Freeze the registry and settings that affect emitted candidates."""

        if (
            len(download_config_hash) != 64
            or any(character not in "0123456789abcdef" for character in download_config_hash)
        ):
            raise DownloadProviderError(
                "resolver snapshot requires a download config SHA-256"
            )
        order = tuple(configured_order)
        if self.names != order:
            raise DownloadProviderError(
                "resolver registry does not match the configured frozen order"
            )
        if set(runtime_config) != set(order):
            raise DownloadProviderError(
                "resolver runtime config must exactly cover the frozen registry"
            )
        resolvers: list[dict[str, Any]] = []
        for name in order:
            descriptor = self._descriptors[name]
            current = runtime_config[name]
            if not isinstance(current, Mapping):
                raise DownloadProviderError(
                    f"resolver runtime config must be a mapping: {name}"
                )
            resolvers.append({
                "name": name,
                "implementation_version": descriptor.implementation_version,
                "candidate_config": deepcopy(dict(descriptor.candidate_config)),
                "runtime_config": deepcopy(dict(current)),
            })
        document = {
            "schema_version": "1",
            "resolver_order": list(order),
            "download_config_hash": download_config_hash,
            "resolvers": resolvers,
        }
        return ResolverSnapshot(document, content_hash(document))

    def resolve(self, context: ResolverContext) -> tuple[AccessLocationCandidate, ...]:
        candidates: list[AccessLocationCandidate] = []
        for descriptor in self._descriptors.values():
            candidates.extend(descriptor.resolver.resolve(context))
        return tuple(_dedupe(candidates))


class PersistedRequestDownloadProvider:
    """Policy/fetch adapter that preserves ``DownloadService`` side-effect rules."""

    def __init__(self, name: str, service: DownloadService) -> None:
        self.name = name
        self.service = service

    def probe(self, candidate: AccessLocationCandidate, context: ProbeContext) -> FetchDecision:
        return self.service.probe(
            candidate,
            purpose=context.purpose,
            provider=self.name,
            now=context.now,
            authorization_grant_id=context.authorization_grant_id,
            mode=context.mode,
            skill_digest=context.skill_digest,
            dependency_digest=context.dependency_digest,
            collection_id=context.collection_id,
            collection_snapshot_hash=context.collection_snapshot_hash,
            selection_snapshot_hash=context.selection_snapshot_hash,
            run_id=context.run_id,
        )

    def fetch(self, request: FetchRequest, context: FetchContext) -> DownloadResult:
        if request.provider != self.name:
            raise DownloadProviderError("fetch request is bound to a different provider")
        return self.service.fetch(
            request,
            run_id=context.run_id,
            now=context.now,
            authorization_context=context.authorization_context,
        )


class UnavailableDownloadProvider:
    """A safe placeholder for manual queues and unaudited skill integrations."""

    def __init__(self, name: str, policy_version: str, reason_code: str) -> None:
        self.name = name
        self.policy_version = policy_version
        self.reason_code = reason_code

    def probe(self, candidate: AccessLocationCandidate, context: ProbeContext) -> FetchDecision:
        return FetchDecision(
            candidate_id=candidate.candidate_id,
            status=FetchDecisionStatus.MANUAL,
            reason_code=self.reason_code,
            policy_version=self.policy_version,
        )

    def fetch(self, request: FetchRequest, context: FetchContext) -> DownloadResult:
        raise DownloadProviderError(f"{self.name} cannot fetch without an installed adapter")


@runtime_checkable
class AuthorizedSkillAdapter(RoutedDownloadProvider, Protocol):
    """Boundary for an independently audited authorized-browser skill adapter."""


@dataclass(frozen=True, slots=True)
class DownloadProviderDescriptor:
    """Declarative contract for a routed Stage 3 download provider.

    ``contract`` is deliberately a closed mapping rather than an informal
    provider comment.  Registry validation makes an extension state every
    security and execution boundary before it can receive a candidate.
    """

    name: str
    provider: RoutedDownloadProvider
    handles: Callable[[AccessLocationCandidate], bool]
    contract: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ProbeAttempt:
    candidate: AccessLocationCandidate
    provider: str
    decision: FetchDecision


class DownloadProviderRegistry:
    """Ordered provider descriptors with persisted-request-only fetch dispatch."""

    def __init__(self, descriptors: Sequence[DownloadProviderDescriptor] = ()) -> None:
        self._descriptors: dict[str, DownloadProviderDescriptor] = {}
        for descriptor in descriptors:
            self.register(descriptor)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._descriptors)

    def descriptor(self, name: str) -> DownloadProviderDescriptor:
        """Return the validated descriptor, including its frozen contract."""

        try:
            return self._descriptors[name]
        except KeyError as error:
            raise DownloadProviderError(f"unknown download provider: {name}") from error

    def register(self, descriptor: DownloadProviderDescriptor) -> None:
        if not descriptor.name or descriptor.name != descriptor.provider.name:
            raise DownloadProviderError("provider descriptor name must match its provider")
        if descriptor.name in self._descriptors:
            raise DownloadProviderError(f"duplicate download provider descriptor: {descriptor.name}")
        self._descriptors[descriptor.name] = replace(
            descriptor, contract=_validate_provider_contract(descriptor.contract)
        )

    def probe(self, candidate: AccessLocationCandidate, context: ProbeContext) -> ProbeAttempt:
        for descriptor in self._descriptors.values():
            if _contract_handles(descriptor.contract, candidate) and descriptor.handles(candidate):
                return self.probe_with(descriptor.name, candidate, context)
        raise DownloadProviderError(f"no download provider accepts resolver {candidate.resolver}")

    def probe_with(
        self, provider: str, candidate: AccessLocationCandidate, context: ProbeContext
    ) -> ProbeAttempt:
        """Run an explicitly selected provider, e.g. an approved skill adapter.

        This keeps authorized-browser use opt-in: normal routing reaches its
        manual fallback rather than silently entering an authenticated session.
        """

        try:
            descriptor = self._descriptors[provider]
        except KeyError as error:
            raise DownloadProviderError(f"unknown download provider: {provider}") from error
        if not _contract_handles(descriptor.contract, candidate):
            raise DownloadProviderError(
                f"download provider {provider} does not handle {candidate.resolver} at {candidate.host}"
            )
        return ProbeAttempt(candidate, descriptor.name, descriptor.provider.probe(candidate, context))

    def probe_all(
        self, candidates: Sequence[AccessLocationCandidate], context: ProbeContext
    ) -> tuple[ProbeAttempt, ...]:
        return tuple(self.probe(candidate, context) for candidate in candidates)

    def fetch(self, request: FetchRequest, context: FetchContext) -> DownloadResult:
        try:
            descriptor = self._descriptors[request.provider]
        except KeyError as error:
            raise DownloadProviderError(f"unknown persisted request provider: {request.provider}") from error
        return descriptor.provider.fetch(request, context)


def default_resolver_registry() -> ResolverRegistry:
    return ResolverRegistry(
        (
            ResolverDescriptor(
                "publisher_public",
                OfficialPublicResolver(),
                implementation_version="publisher-public-resolver-v1",
            ),
            ResolverDescriptor(
                "europe_pmc",
                EuropePMCOpenAccessResolver(),
                implementation_version="europe-pmc-oa-resolver-v1",
            ),
            ResolverDescriptor(
                "unpaywall",
                UnpaywallOpenAccessResolver(),
                implementation_version="unpaywall-oa-resolver-v1",
            ),
            ResolverDescriptor(
                "arxiv",
                MatchedArxivResolver(),
                implementation_version="matched-arxiv-resolver-v1",
            ),
        )
    )


def default_download_provider_registry(
    service: DownloadService,
    *,
    authorized_skill: AuthorizedSkillAdapter | None = None,
) -> DownloadProviderRegistry:
    """Create the frozen default order while leaving extension to descriptors."""

    policy_version = service.policy.version
    skill = authorized_skill or UnavailableDownloadProvider(
        "authorized_skill", policy_version, "authorized_skill_adapter_unavailable"
    )
    return DownloadProviderRegistry(
        (
            DownloadProviderDescriptor(
                "public_direct", PersistedRequestDownloadProvider("public_direct", service),
                lambda candidate: candidate.resolver == "publisher_public",
                provider_contract(
                    handled_domains=("*",), handled_resolvers=("publisher_public",),
                ),
            ),
            DownloadProviderDescriptor(
                "europe_pmc", PersistedRequestDownloadProvider("europe_pmc", service),
                lambda candidate: candidate.resolver == "europe_pmc",
                provider_contract(
                    handled_domains=("*",),
                    handled_resolvers=("europe_pmc",),
                ),
            ),
            DownloadProviderDescriptor(
                "unpaywall_location", PersistedRequestDownloadProvider("unpaywall_location", service),
                lambda candidate: candidate.resolver == "unpaywall",
                provider_contract(
                    handled_domains=("*",), handled_resolvers=("unpaywall",),
                ),
            ),
            DownloadProviderDescriptor(
                "arxiv", PersistedRequestDownloadProvider("arxiv", service),
                lambda candidate: candidate.resolver == "arxiv",
                provider_contract(
                    handled_domains=("*",),
                    handled_resolvers=("arxiv",),
                ),
            ),
            DownloadProviderDescriptor(
                "authorized_skill", skill, lambda _candidate: False,
                provider_contract(
                    authentication_required=True,
                    supports_supplements=True,
                    supports_version_selection=True,
                    allows_unattended=False,
                    handled_domains=("*",),
                    handled_resolvers=("*",),
                    retry_semantics="external_ledger_resumable",
                ),
            ),
            DownloadProviderDescriptor(
                "manual",
                UnavailableDownloadProvider("manual", policy_version, "manual_queue_required"),
                lambda _candidate: True,
                provider_contract(
                    supports_main_document=False,
                    handled_domains=("*",),
                    handled_resolvers=("*",),
                    retry_semantics="not_retryable",
                ),
            ),
        )
    )


def provider_contract(
    *,
    authentication_required: bool = False,
    supports_main_document: bool = True,
    supports_supplements: bool = False,
    supports_version_selection: bool = False,
    allows_unattended: bool = True,
    handled_domains: Sequence[str] = ("*",),
    handled_resolvers: Sequence[str] = ("*",),
    retry_semantics: str = "transient_retryable",
    probe_input_schema_id: str = PROBE_INPUT_SCHEMA_ID,
    probe_output_schema_id: str = PROBE_OUTPUT_SCHEMA_ID,
    fetch_input_schema_id: str = FETCH_INPUT_SCHEMA_ID,
    fetch_output_schema_id: str = FETCH_OUTPUT_SCHEMA_ID,
    idempotency_key_boundary: str = _IDEMPOTENCY_KEY_BOUNDARY,
    side_effect_boundary: str = _SIDE_EFFECT_BOUNDARY,
) -> dict[str, Any]:
    """Build the explicit, closed contract required by a descriptor.

    Extensions may use provider-specific schema IDs and route constraints, but
    cannot omit the probe/fetch idempotency and side-effect declarations.
    """

    return {
        "authentication_required": authentication_required,
        "supports_main_document": supports_main_document,
        "supports_supplements": supports_supplements,
        "supports_version_selection": supports_version_selection,
        "allows_unattended": allows_unattended,
        "handled_domains": tuple(handled_domains),
        "handled_resolvers": tuple(handled_resolvers),
        "retry_semantics": retry_semantics,
        "probe_input_schema_id": probe_input_schema_id,
        "probe_output_schema_id": probe_output_schema_id,
        "fetch_input_schema_id": fetch_input_schema_id,
        "fetch_output_schema_id": fetch_output_schema_id,
        "idempotency_key_boundary": idempotency_key_boundary,
        "side_effect_boundary": side_effect_boundary,
    }


def _validate_provider_contract(contract: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(contract, Mapping) or set(contract) != _PROVIDER_CONTRACT_FIELDS:
        raise DownloadProviderError("provider descriptor contract must declare exactly the required fields")
    booleans = (
        "authentication_required", "supports_main_document", "supports_supplements",
        "supports_version_selection", "allows_unattended",
    )
    if any(type(contract[field]) is not bool for field in booleans):
        raise DownloadProviderError("provider descriptor boolean contract fields must be booleans")
    for field in ("handled_domains", "handled_resolvers"):
        values = contract[field]
        if not isinstance(values, tuple) or not values or not all(isinstance(item, str) and item for item in values):
            raise DownloadProviderError(f"provider descriptor {field} must be a non-empty string tuple")
    if contract["retry_semantics"] not in _RETRY_SEMANTICS:
        raise DownloadProviderError("provider descriptor retry_semantics is unsupported")
    for field in (
        "probe_input_schema_id", "probe_output_schema_id", "fetch_input_schema_id", "fetch_output_schema_id",
    ):
        if not isinstance(contract[field], str) or not contract[field]:
            raise DownloadProviderError(f"provider descriptor {field} must be a non-empty schema ID")
    if contract["idempotency_key_boundary"] != _IDEMPOTENCY_KEY_BOUNDARY:
        raise DownloadProviderError("provider descriptor must bind fetch to the persisted idempotency key")
    if contract["side_effect_boundary"] != _SIDE_EFFECT_BOUNDARY:
        raise DownloadProviderError("provider descriptor must keep probe free of body downloads")
    return MappingProxyType(dict(contract))


def _contract_handles(contract: Mapping[str, Any], candidate: AccessLocationCandidate) -> bool:
    resolvers = contract["handled_resolvers"]
    domains = contract["handled_domains"]
    return (
        "*" in resolvers or candidate.resolver in resolvers
    ) and (
        "*" in domains or candidate.host in domains
    )


def _lookup(context: ResolverContext, resolver: str) -> ResolverEvidence | None:
    return context.lookup(resolver, context.paper) if context.lookup else None


def _europe_pmc_landing_url(record: Mapping[str, Any]) -> str | None:
    source, identifier = _text(record.get("source")), _text(record.get("id"))
    if not source or not identifier:
        return None
    return f"https://europepmc.org/article/{source}/{identifier}"


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _dedupe(
    candidates: Sequence[AccessLocationCandidate],
) -> list[AccessLocationCandidate]:
    output: list[AccessLocationCandidate] = []
    seen: set[tuple[str, str, PublicationVersion]] = set()
    for candidate in candidates:
        key = (candidate.resolver, candidate.url, candidate.publication_version)
        if key not in seen:
            seen.add(key)
            output.append(candidate)
    return output
