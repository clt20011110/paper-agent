"""Production metadata lookup boundary for the public Stage 3 resolvers.

The descriptors in this module declare how a resolver identifies one paper.
They deliberately only call provider API metadata operations; an access URL
returned in that metadata remains an untrusted candidate for Stage 3's normal
policy-governed provider chain.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any, Protocol

from .canonical import content_hash
from .domain import Paper
from .download_providers import ResolverEvidence
from .provider_runtime import ProviderRequestError, RetryableProviderError


LOOKUP_IMPLEMENTATION_VERSION = "stage3-metadata-lookup-v1"
CONTROLLED_HTTP_TRANSPORT_IMPLEMENTATION_VERSION = (
    "stage3-controlled-http-metadata-transport-v1"
)


class PublicMetadataTransport(Protocol):
    """A metadata-only provider request boundary, injectable for offline tests."""

    def __call__(
        self, provider: str, operation: str, parameters: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...


LookupParameters = Callable[[Paper], Mapping[str, Any] | None]


@dataclass(frozen=True, slots=True)
class MetadataLookupDescriptor:
    """One resolver's public metadata query contract."""

    resolver: str
    provider: str
    operation: str
    implementation_version: str
    parameter_contract: Mapping[str, Any]
    parameters: LookupParameters


class MetadataLookupRegistry:
    """Named descriptors keep Stage 3 source additions out of a central branch."""

    def __init__(self, descriptors: Sequence[MetadataLookupDescriptor] = ()) -> None:
        self._descriptors: dict[str, MetadataLookupDescriptor] = {}
        for descriptor in descriptors:
            self.register(descriptor)

    def register(self, descriptor: MetadataLookupDescriptor) -> None:
        if not descriptor.resolver or descriptor.resolver in self._descriptors:
            raise ValueError(f"duplicate or empty metadata resolver: {descriptor.resolver}")
        if not descriptor.provider or not descriptor.operation:
            raise ValueError("metadata lookup provider and operation are required")
        if (
            not isinstance(descriptor.implementation_version, str)
            or not descriptor.implementation_version.strip()
        ):
            raise ValueError("metadata lookup implementation_version is required")
        if not isinstance(descriptor.parameter_contract, Mapping):
            raise ValueError("metadata lookup parameter_contract must be a mapping")
        content_hash(descriptor.parameter_contract)
        self._descriptors[descriptor.resolver] = replace(
            descriptor,
            parameter_contract=deepcopy(dict(descriptor.parameter_contract)),
        )

    def get(self, resolver: str) -> MetadataLookupDescriptor | None:
        return self._descriptors.get(resolver)

    def canonical_identity(self) -> Mapping[str, Any]:
        """Describe the actual ordered lookup registry used at runtime."""

        return {
            "schema_version": "1",
            "descriptors": [
                {
                    "resolver": descriptor.resolver,
                    "provider": descriptor.provider,
                    "operation": descriptor.operation,
                    "implementation_version": descriptor.implementation_version,
                    "parameter_contract": deepcopy(
                        dict(descriptor.parameter_contract)
                    ),
                }
                for descriptor in self._descriptors.values()
            ],
        }


def default_metadata_lookup_registry() -> MetadataLookupRegistry:
    """The built-in public resolver contracts in frozen Stage 3 order."""

    # Europe PMC and arXiv search retain the native evidence fields that the
    # policy-aware resolvers validate; the generic resolve operation is a
    # lossy candidate envelope intended for the Stage 1 provider adapter.
    return MetadataLookupRegistry((
        MetadataLookupDescriptor(
            "europe_pmc",
            "europe_pmc",
            "search",
            "europe-pmc-doi-parameters-v2",
            {
                "paper_field": "doi",
                "parameter": "doi",
                "fixed_parameters": {"resultType": "core"},
                "missing": "skip",
            },
            _europe_pmc_doi_parameters,
        ),
        MetadataLookupDescriptor(
            "unpaywall",
            "unpaywall",
            "resolve",
            "unpaywall-doi-parameters-v1",
            {"paper_field": "doi", "parameter": "doi", "missing": "skip"},
            _doi_parameters,
        ),
        MetadataLookupDescriptor(
            "arxiv",
            "arxiv",
            "search",
            "arxiv-id-parameters-v1",
            {
                "paper_field": "arxiv_id",
                "parameter": "query",
                "template": "id:{value}",
                "missing": "skip",
            },
            _arxiv_parameters,
        ),
    ))


@dataclass(frozen=True, slots=True)
class Stage3MetadataLookup:
    """Turn controlled public API responses into resolver evidence.

    Expected network/provider failures intentionally return no evidence.  The
    existing resolver pipeline then reaches its durable manual-queue outcome;
    no PDF, landing page, or authenticated-browser request is attempted here.
    """

    transport: PublicMetadataTransport
    retrieved_at: Callable[[], datetime]
    registry: MetadataLookupRegistry
    transport_identity: Mapping[str, Any] | None = None

    def canonical_identity(self) -> Mapping[str, Any]:
        transport_identity = self.transport_identity
        if transport_identity is None:
            identity = getattr(self.transport, "canonical_identity", None)
            if not callable(identity):
                raise ValueError(
                    "Stage 3 metadata transport must expose canonical_identity"
                )
            transport_identity = identity()
        if not isinstance(transport_identity, Mapping):
            raise ValueError(
                "Stage 3 metadata transport canonical_identity must be a mapping"
            )
        frozen_transport_identity = deepcopy(dict(transport_identity))
        content_hash(frozen_transport_identity)
        return {
            "schema_version": "1",
            "implementation_version": LOOKUP_IMPLEMENTATION_VERSION,
            "transport": frozen_transport_identity,
            "registry": self.registry.canonical_identity(),
        }

    def __call__(self, resolver: str, paper: Paper) -> ResolverEvidence | None:
        descriptor = self.registry.get(resolver)
        if descriptor is None:
            return None
        parameters = descriptor.parameters(paper)
        if parameters is None:
            return None
        try:
            payload = self.transport(descriptor.provider, descriptor.operation, parameters)
        except (OSError, ProviderRequestError, RetryableProviderError):
            return None
        raw_evidence_hash = payload.get("raw_response_artifact_hash")
        return ResolverEvidence(
            payload=payload,
            raw_evidence_hash=raw_evidence_hash if isinstance(raw_evidence_hash, str) else None,
            retrieved_at=_timestamp(self.retrieved_at()),
        )


def _doi_parameters(paper: Paper) -> Mapping[str, Any] | None:
    if not paper.doi:
        return None
    return {"doi": paper.doi}


def _europe_pmc_doi_parameters(paper: Paper) -> Mapping[str, Any] | None:
    if not paper.doi:
        return None
    return {"doi": paper.doi, "resultType": "core"}


def _arxiv_parameters(paper: Paper) -> Mapping[str, Any] | None:
    if not paper.arxiv_id:
        return None
    return {"query": f"id:{paper.arxiv_id}"}


def _timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("Stage 3 metadata clock must return a timezone-aware datetime")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
