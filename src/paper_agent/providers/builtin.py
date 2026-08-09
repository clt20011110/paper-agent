"""Deterministic built-in Stage 1 provider adapters.

The adapters in this module deliberately stop at a provider response envelope.
They translate provider-native pagination and records into ``SourceBatch`` or
``CitationBatch``; normalisation, membership decisions, and persistence belong
to the coordinator.  ``FixtureTransport`` is useful both for contract tests
and for callers that want to supply an HTTP transport without coupling the
adapter to a particular client library.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from hashlib import sha256
from pathlib import Path
import json
import re
from typing import Any, Callable, Mapping, Sequence

from paper_agent.domain import (
    AccessBasis,
    AccessLocationCandidate,
    CitationBatch,
    CitationEdge,
    CitationEdgeType,
    EnvelopeStatus,
    Paper,
    ProviderCapability,
    ProviderRole,
    PublicationVersion,
    QuerySpec,
    SourceBatch,
    SourceEntry,
    VerificationStatus,
)
from paper_agent.providers.api import (
    AccessPolicy,
    CrawlWindow,
    EnrichmentResult,
    IdentityCandidate,
    ProviderManifest,
    SeedInput,
    VenueDescriptor,
    VerificationResult,
    validate_citation_batch,
    validate_source_batch,
)


Transport = Callable[[str, str, Mapping[str, Any]], Mapping[str, Any]]


@dataclass(slots=True)
class FixtureTransport:
    """A transport keyed by ``provider:operation:cursor`` for fixed fixtures."""

    responses: Mapping[str, Mapping[str, Any]]
    calls: list[tuple[str, str, dict[str, Any]]] = field(default_factory=list)

    def __call__(self, provider: str, operation: str, parameters: Mapping[str, Any]) -> Mapping[str, Any]:
        params = dict(parameters)
        self.calls.append((provider, operation, params))
        cursor = params.get("cursor") or "first"
        key = f"{provider}:{operation}:{cursor}"
        return self.responses.get(key, self.responses.get(f"{provider}:{operation}", {"entries": []}))


def manifest_from_document(document: Mapping[str, Any]) -> ProviderManifest:
    """Turn the versioned YAML document into the runtime manifest contract."""

    authentication = document["authentication"]
    rate_limit = document["rate_limit"]
    from paper_agent.providers.api import CredentialPolicy, RateLimitPolicy

    return ProviderManifest(
        provider=str(document["provider"]),
        version=str(document["version"]),
        roles=tuple(ProviderRole(item) for item in document["roles"]),
        capabilities=tuple(ProviderCapability(item) for item in document["capabilities"]),
        stable_identifier=f"{document['provider']}:external_id",
        distribution=str(document["distribution"]),
        entry_point=str(document["entry_point"]),
        artifact_sha256=document["artifact_sha256"],
        enabled=bool(document["enabled"]),
        builtin=bool(document["builtin"]),
        authority=str(document["authority"]),
        credential_policy=CredentialPolicy(
            required=bool(authentication["required"]),
            environment_variables=(authentication["credential_env"],) if "credential_env" in authentication else (),
        ),
        rate_limit_policy=RateLimitPolicy(
            queries_per_second=float(rate_limit["global_qps"]),
            max_concurrency=int(rate_limit["max_concurrency"]),
            cache_ttl_seconds=int(rate_limit["cache_ttl_seconds"]),
        ),
        terms_url=document["terms"].get("url"),
        independence_group=str(document["independence_group"]),
        upstream_families=tuple(str(item) for item in document["upstream_families"]),
    )


def load_builtin_manifest(provider: str, root: Path | None = None) -> ProviderManifest:
    """Load a built-in manifest from the catalog instead of duplicating its facts."""

    import yaml

    catalog_root = root or Path(__file__).resolve().parents[3]
    document = yaml.safe_load((catalog_root / "providers" / f"{provider}.yaml").read_text(encoding="utf-8"))
    return manifest_from_document(document)


def _hash(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _value(value: Any) -> Any:
    if isinstance(value, Mapping):
        if "value" in value:
            return value["value"]
        if "values" in value:
            return value["values"]
    return value


def _text(value: Any) -> str | None:
    value = _value(value)
    if value is None:
        return None
    if isinstance(value, list):
        return _text(value[0]) if value else None
    return str(value)


def _authors(value: Any) -> tuple[str, ...]:
    if not value:
        return ()
    if isinstance(value, str):
        return tuple(part.strip() for part in re.split(r"\s+and\s+|;", value) if part.strip())
    output = []
    for author in value:
        if isinstance(author, Mapping):
            name = author.get("name") or author.get("display_name")
            if not name:
                name = " ".join(str(part) for part in (author.get("given"), author.get("family")) if part)
            output.append(str(name or ""))
        else:
            output.append(str(author))
    return tuple(author for author in output if author)


def _date(record: Mapping[str, Any]) -> str | None:
    for key in ("publication_date", "published", "date", "published_date", "publicationDate"):
        raw_value = _value(record.get(key))
        if isinstance(raw_value, Mapping):
            parts = raw_value.get("date-parts")
            if isinstance(parts, list) and parts and isinstance(parts[0], list):
                return "-".join(f"{int(part):02d}" if index else str(part) for index, part in enumerate(parts[0]))
        value = _text(raw_value)
        if value:
            return value[:10]
    return None


def _year(record: Mapping[str, Any], publication_date: str | None) -> int | None:
    value = record.get("year") or record.get("publication_year")
    if value is not None:
        return int(value)
    if publication_date and publication_date[:4].isdigit():
        return int(publication_date[:4])
    return None


def _records(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    for key in ("entries", "results", "items", "works", "papers", "articles", "data", "notes"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, Mapping)]
    return []


def _provider_records(provider: str, payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Extract the documented top-level collection without doing any selection."""

    if provider == "crossref" and isinstance(payload.get("message"), Mapping):
        return _records(payload["message"])
    if provider == "dblp":
        result = payload.get("result")
        if isinstance(result, Mapping) and isinstance(result.get("hits"), Mapping):
            return _records(result["hits"])
    if provider == "europe_pmc" and isinstance(payload.get("resultList"), Mapping):
        return _records(payload["resultList"])
    if provider == "arxiv" and isinstance(payload.get("feed"), Mapping):
        return _records(payload["feed"])
    return _records(payload)


def _next_cursor(payload: Mapping[str, Any]) -> str | None:
    value = payload.get("next_cursor") or payload.get("next") or payload.get("cursor")
    return str(value) if value else None


def _source_entry(provider: str, record: Mapping[str, Any]) -> SourceEntry:
    content = record.get("content") if isinstance(record.get("content"), Mapping) else record
    external_id = (
        record.get("external_id")
        or record.get("stable_id")
        or record.get("id")
        or record.get("paperId")
        or record.get("doi")
        or record.get("DOI")
        or record.get("arxiv_id")
        or record.get("arxivId")
    )
    if external_id is None:
        raise ValueError(f"{provider}: provider record has no stable identifier")
    title = _text(content.get("title") or record.get("title"))
    if not title:
        raise ValueError(f"{provider}: provider record {external_id} has no title")
    publication_date = _date(record) or _date(content)
    doi = _text(record.get("doi") or record.get("DOI") or content.get("doi"))
    arxiv_id = _text(record.get("arxiv_id") or record.get("arxivId") or content.get("arxiv_id"))
    return SourceEntry(
        provider=provider,
        external_id=str(external_id),
        title=title,
        authors=_authors(record.get("authors") or record.get("author") or content.get("authors") or content.get("author")),
        abstract=_text(content.get("abstract") or record.get("abstract")),
        doi=doi.lower() if doi else None,
        arxiv_id=arxiv_id,
        publication_date=publication_date,
        year=_year(record, publication_date),
        venue_name=_text(record.get("venue") or record.get("venue_name") or content.get("venue")),
        landing_url=_text(record.get("landing_url") or record.get("url") or record.get("URL") or record.get("html_url")),
        metadata={key: value for key, value in record.items() if key not in {"content", "authors"}},
    )


class BuiltinProvider:
    """Shared protocol mapping for transport-injected built-in providers."""

    provider: str

    def __init__(self, provider: str, transport: Transport, manifest: ProviderManifest | None = None) -> None:
        self.provider = provider
        self.transport = transport
        self.manifest = manifest or load_builtin_manifest(provider)
        if self.manifest.provider != provider:
            raise ValueError(f"manifest provider {self.manifest.provider} does not match {provider}")

    def _require(self, role: ProviderRole, capability: ProviderCapability | None = None) -> None:
        if role not in self.manifest.roles:
            raise ValueError(f"{self.provider} does not declare role {role.value}")
        if capability and capability not in self.manifest.capabilities:
            raise ValueError(f"{self.provider} does not declare capability {capability.value}")

    def _request(self, operation: str, parameters: Mapping[str, Any]) -> Mapping[str, Any]:
        return self.transport(self.provider, operation, parameters)

    def _batch(self, payload: Mapping[str, Any], source_run_id: str, query_hash: str) -> SourceBatch:
        status = EnvelopeStatus(str(payload.get("status", "success")))
        batch = SourceBatch(
            source_run_id=source_run_id,
            query_hash=query_hash,
            entries=tuple(_source_entry(self.provider, record) for record in _provider_records(self.provider, payload)),
            next_cursor=_next_cursor(payload),
            status=status,
            error=_text(payload.get("error")),
            raw_response_artifact_hash=_text(payload.get("raw_response_artifact_hash")),
        )
        return validate_source_batch(batch)

    def search(self, query_spec: QuerySpec, cursor: str | None = None) -> SourceBatch:
        self._require(ProviderRole.SEARCH, ProviderCapability.METADATA)
        parameters = {
            "query": query_spec.original_query,
            "synonym_groups": dict(query_spec.synonym_groups),
            "date_from": query_spec.date_from,
            "date_to": query_spec.date_to,
            "fields": query_spec.fields,
            "venues": query_spec.venue_ids,
            "sort": query_spec.sort,
            "page_size": query_spec.page_size,
            "cursor": cursor,
        }
        payload = self._request("search", parameters)
        return self._batch(payload, str(payload.get("source_run_id") or f"{self.provider}:search"), _hash(query_spec.original_query))

    def enrich(self, raw_paper: SourceEntry) -> EnrichmentResult:
        self._require(ProviderRole.METADATA_ENRICHER, ProviderCapability.METADATA)
        payload = self._request("enrich", {"external_id": raw_paper.external_id, "doi": raw_paper.doi, "arxiv_id": raw_paper.arxiv_id})
        records = _provider_records(self.provider, payload)
        entry = _source_entry(self.provider, records[0]) if records else raw_paper
        return EnrichmentResult(entry, self.provider, str(payload.get("source_run_id") or f"{self.provider}:enrich"), _text(payload.get("raw_response_artifact_hash")))

    def verify(self, identity_candidate: IdentityCandidate, evidence: Sequence[SourceEntry]) -> VerificationResult:
        self._require(ProviderRole.METADATA_VERIFIER, ProviderCapability.METADATA)
        payload = self._request("verify", {"doi": identity_candidate.doi, "arxiv_id": identity_candidate.arxiv_id, "title": identity_candidate.title})
        status = VerificationStatus(str(payload.get("status", "single_source")))
        return VerificationResult(identity_candidate, status, self.provider, tuple(str(item) for item in payload.get("evidence", ())))

    def resolve(self, paper: Paper, policy: AccessPolicy) -> list[AccessLocationCandidate]:
        self._require(ProviderRole.OA_RESOLVER, ProviderCapability.OA_LOCATIONS)
        payload = self._request("resolve", {"paper_id": paper.paper_id, "doi": paper.doi, "purpose": policy.purpose})
        return [
            AccessLocationCandidate(
                candidate_id=str(record.get("candidate_id") or f"{self.provider}:{paper.paper_id}:{index}"),
                paper_id=paper.paper_id,
                resolver=self.provider,
                url=str(record["url"]),
                landing_url=_text(record.get("landing_url")),
                host=_text(record.get("host")),
                publication_version=PublicationVersion(str(record.get("publication_version", "unknown"))),
                license=_text(record.get("license")),
                access_basis=AccessBasis(str(record.get("access_basis", "unknown"))),
                raw_evidence_hash=_text(payload.get("raw_response_artifact_hash")),
                provenance={"provider": self.provider},
            )
            for index, record in enumerate(_records(payload))
            if record.get("url")
        ]

    def _citations(self, direction: CitationEdgeType, seed: Paper, cursor: str | None) -> CitationBatch:
        capability = ProviderCapability.REFERENCES if direction is CitationEdgeType.REFERENCES else ProviderCapability.CITATIONS
        self._require(ProviderRole.CITATION, capability)
        operation = "references" if direction is CitationEdgeType.REFERENCES else "citations"
        payload = self._request(operation, {"paper_id": seed.paper_id, "doi": seed.doi, "cursor": cursor})
        edges = []
        for record in _records(payload):
            result_id = record.get("paper_id") or record.get("external_id") or record.get("id")
            if not result_id:
                continue
            source_id, target_id = (
                (seed.paper_id, str(result_id))
                if direction is CitationEdgeType.REFERENCES
                else (str(result_id), seed.paper_id)
            )
            edges.append(
                CitationEdge(
                    source_paper_id=source_id,
                    target_paper_id=target_id,
                    edge_type=direction,
                    provider=self.provider,
                    observed_at=str(record.get("observed_at") or payload.get("observed_at") or ""),
                    raw_evidence=dict(record),
                )
            )
        batch = CitationBatch(
            str(payload.get("source_run_id") or f"{self.provider}:{operation}"),
            _hash(f"{seed.paper_id}:{operation}"),
            tuple(edges),
            _next_cursor(payload),
            EnvelopeStatus(str(payload.get("status", "success"))),
            _text(payload.get("error")),
            _text(payload.get("raw_response_artifact_hash")),
        )
        return validate_citation_batch(batch)

    def references(self, seed: Paper, cursor: str | None = None) -> CitationBatch:
        return self._citations(CitationEdgeType.REFERENCES, seed, cursor)

    def citations(self, seed: Paper, cursor: str | None = None) -> CitationBatch:
        return self._citations(CitationEdgeType.CITATIONS, seed, cursor)


class VenueBuiltinAdapter(BuiltinProvider):
    """Official venue adapter: descriptor/window/cursor are mapped but never filtered."""

    def discover(self, descriptor: VenueDescriptor, window: CrawlWindow, cursor: str | None = None) -> SourceBatch:
        self._require(ProviderRole.VENUE_PRIMARY, ProviderCapability.METADATA)
        if descriptor.provider != self.provider:
            raise ValueError(f"{descriptor.venue_id} is assigned to {descriptor.provider}, not {self.provider}")
        parameters = dict(descriptor.parameters)
        parameters.update(
            {
                "venue_id": descriptor.venue_id,
                "adapter": descriptor.adapter,
                "date_from": window.date_from,
                "date_to": window.date_to,
                "year": window.year,
                "volume": window.volume,
                "issue": window.issue,
                "cursor": cursor,
            }
        )
        payload = self._shape_discovery_payload(self._request("discover", parameters), descriptor)
        source_run_id = str(payload.get("source_run_id") or parameters.get("source_run_id") or f"{self.provider}:{descriptor.venue_id}")
        query_hash = str(payload.get("query_hash") or _hash(json.dumps(parameters, sort_keys=True, default=str)))
        batch = self._batch(payload, source_run_id, query_hash)
        return replace(
            batch,
            entries=tuple(
                replace(
                    entry,
                    metadata={
                        **entry.metadata,
                        "official_membership": True,
                        "venue_id": descriptor.venue_id,
                    },
                )
                for entry in batch.entries
            ),
        )

    def _shape_discovery_payload(
        self, payload: Mapping[str, Any], descriptor: VenueDescriptor
    ) -> Mapping[str, Any]:
        return payload


class OpenReviewAdapter(VenueBuiltinAdapter):
    """OpenReview v1/v2 adapter with invitation resolved by the descriptor."""

    def discover(self, descriptor: VenueDescriptor, window: CrawlWindow, cursor: str | None = None) -> SourceBatch:
        parameters = dict(descriptor.parameters)
        if "invitation" not in parameters:
            if not parameters.get("venue_group") or window.year is None:
                raise ValueError("openreview descriptors require venue_group and a year")
            resolved = self._request(
                "resolve_invitation",
                {
                    "venue_group": parameters["venue_group"],
                    "year": window.year,
                    "decision": "accepted",
                },
            )
            parameters["invitation"] = str(resolved["invitation"])
            parameters["api_version"] = str(resolved["api_version"])
        runtime = VenueDescriptor(
            descriptor.schema_version,
            descriptor.venue_id,
            descriptor.provider,
            descriptor.adapter,
            parameters,
        )
        return super().discover(runtime, window, cursor)

    def _shape_discovery_payload(
        self, payload: Mapping[str, Any], descriptor: VenueDescriptor
    ) -> Mapping[str, Any]:
        if not descriptor.parameters.get("accepted_decision_required", True):
            return payload
        accepted = [record for record in _provider_records(self.provider, payload) if _openreview_accepted(record)]
        return {**payload, "entries": accepted, "notes": None}


class AAAIOJSAdapter(VenueBuiltinAdapter):
    def _shape_discovery_payload(
        self, payload: Mapping[str, Any], descriptor: VenueDescriptor
    ) -> Mapping[str, Any]:
        issues = payload.get("issues")
        if not isinstance(issues, list):
            return payload
        entries = []
        for issue in issues:
            if not isinstance(issue, Mapping):
                continue
            for article in issue.get("articles", ()):
                if isinstance(article, Mapping):
                    entries.append({**article, "ojs_issue_id": issue.get("id")})
        return {**payload, "entries": entries}


class CVFOpenAccessAdapter(VenueBuiltinAdapter):
    def discover(self, descriptor: VenueDescriptor, window: CrawlWindow, cursor: str | None = None) -> SourceBatch:
        parameters = dict(descriptor.parameters)
        if "track" not in parameters and parameters.get("exclude_workshops") is True:
            parameters["track"] = "main"
        if parameters.get("track") not in {"main", "workshop"}:
            raise ValueError("cvf descriptors require track=main or track=workshop")
        runtime = VenueDescriptor(
            descriptor.schema_version, descriptor.venue_id, descriptor.provider, descriptor.adapter, parameters
        )
        return super().discover(runtime, window, cursor)

    def _shape_discovery_payload(
        self, payload: Mapping[str, Any], descriptor: VenueDescriptor
    ) -> Mapping[str, Any]:
        track = descriptor.parameters["track"]
        records = payload.get(track)
        if isinstance(records, list):
            return {**payload, "entries": [{**record, "cvf_track": track} for record in records if isinstance(record, Mapping)]}
        return payload


class EDAProceedingsAdapter(VenueBuiltinAdapter):
    def discover(self, descriptor: VenueDescriptor, window: CrawlWindow, cursor: str | None = None) -> SourceBatch:
        parameters = dict(descriptor.parameters)
        if "upstreams" not in parameters and "sources" in parameters:
            parameters["upstreams"] = parameters["sources"]
        if "upstreams" not in parameters:
            raise ValueError("eda descriptors require year-specific upstreams")
        runtime = VenueDescriptor(
            descriptor.schema_version, descriptor.venue_id, descriptor.provider, descriptor.adapter, parameters
        )
        return super().discover(runtime, window, cursor)

    def _shape_discovery_payload(
        self, payload: Mapping[str, Any], descriptor: VenueDescriptor
    ) -> Mapping[str, Any]:
        upstream_payloads = payload.get("upstreams")
        if not isinstance(upstream_payloads, Mapping):
            return payload
        entries = []
        for upstream in descriptor.parameters["upstreams"]:
            for record in _records(upstream_payloads.get(str(upstream), {})):
                entries.append({**record, "upstream": upstream})
        return {**payload, "entries": entries}


class IEEEXploreAdapter(VenueBuiltinAdapter):
    def discover(self, descriptor: VenueDescriptor, window: CrawlWindow, cursor: str | None = None) -> SourceBatch:
        if descriptor.parameters.get("publication_number") != 43:
            raise ValueError("ieee_xplore TCAD descriptors require publication_number=43")
        return super().discover(descriptor, window, cursor)


class SpringerNatureAdapter(VenueBuiltinAdapter):
    def discover(self, descriptor: VenueDescriptor, window: CrawlWindow, cursor: str | None = None) -> SourceBatch:
        required = {"journal_slug", "issns", "article_types"}
        if not required.issubset(descriptor.parameters):
            raise ValueError("springer_nature descriptors require exact journal_slug, issns, and article_types")
        return super().discover(descriptor, window, cursor)


class CellPressAdapter(VenueBuiltinAdapter):
    def discover(self, descriptor: VenueDescriptor, window: CrawlWindow, cursor: str | None = None) -> SourceBatch:
        issns = descriptor.parameters.get("issns", (descriptor.parameters.get("issn"),))
        if set(issns) != {"0092-8674"}:
            raise ValueError("cell_press descriptors require Cell ISSN 0092-8674")
        return super().discover(descriptor, window, cursor)


class AAASScienceAdapter(VenueBuiltinAdapter):
    def discover(self, descriptor: VenueDescriptor, window: CrawlWindow, cursor: str | None = None) -> SourceBatch:
        if set(descriptor.parameters.get("issns", ())) != {"0036-8075", "1095-9203"}:
            raise ValueError("aaas_science descriptors require exact Science ISSNs")
        return super().discover(descriptor, window, cursor)


class LibrarySeedImporter(BuiltinProvider):
    """Read user-supplied identifiers and bibliography records without network access."""

    def import_seeds(self, input_spec: Sequence[SeedInput]) -> SourceBatch:
        self._require(ProviderRole.LIBRARY, ProviderCapability.METADATA)
        entries = tuple(_seed_entry(self.provider, item) for item in input_spec)
        return validate_source_batch(SourceBatch(f"{self.provider}:seeds", _hash("|".join(item.value for item in input_spec)), entries, None, EnvelopeStatus.SUCCESS))


def _seed_entry(provider: str, item: SeedInput) -> SourceEntry:
    kind = item.kind.lower()
    value = item.value.strip()
    if kind == "doi":
        return SourceEntry(provider, value.lower(), value, doi=value.lower(), metadata={"seed_kind": "doi"})
    if kind in {"arxiv", "arxiv_id"}:
        identifier = value.removeprefix("arXiv:")
        return SourceEntry(provider, identifier, identifier, arxiv_id=identifier, metadata={"seed_kind": "arxiv"})
    if kind == "url":
        return SourceEntry(provider, value, value, landing_url=value, metadata={"seed_kind": "url"})
    if kind == "bibtex":
        return _bibliography_entry(provider, value, "bibtex")
    if kind == "ris":
        return _bibliography_entry(provider, value, "ris")
    if kind in {"csl-json", "csl_json"}:
        return _bibliography_entry(provider, value, "csl-json")
    if kind in {"local_pdf", "pdf"}:
        path = Path(value).resolve()
        return SourceEntry(provider, value, path.stem, landing_url=path.as_uri(), metadata={"seed_kind": "local_pdf", "path": value})
    raise ValueError(f"unsupported seed kind {item.kind}")


def _bibliography_entry(provider: str, value: str, kind: str) -> SourceEntry:
    if kind == "csl-json":
        record = json.loads(value)
        if isinstance(record, list):
            record = record[0]
        return _source_entry(provider, {"external_id": record.get("id") or record.get("DOI") or record["title"], "title": record["title"], "authors": record.get("author", ()), "doi": record.get("DOI"), "year": (record.get("issued", {}).get("date-parts", [[None]])[0][0])})
    if kind == "ris":
        fields = {line[:2]: line[6:] for line in value.splitlines() if len(line) > 6 and line[2:6] == "  - "}
        return _source_entry(provider, {"external_id": fields.get("DO") or fields.get("UR") or fields.get("TI"), "title": fields["TI"], "authors": [line[6:] for line in value.splitlines() if line.startswith("AU  - ")], "doi": fields.get("DO"), "year": (fields.get("PY") or "")[:4]})
    fields = {
        key.lower(): text.strip(" {},\n")
        for key, text in re.findall(r'(title|author|year|doi)\s*=\s*[\{\"]([^}\"]+)', value, re.I)
    }
    title = fields.get("title")
    if not title:
        raise ValueError("BibTeX seed requires title")
    return _source_entry(provider, {"external_id": fields.get("doi") or title, "title": title, "authors": fields.get("author"), "doi": fields.get("doi"), "year": fields.get("year")})


class NeurIPSProceedingsAdapter(VenueBuiltinAdapter):
    pass


class PMLRAdapter(VenueBuiltinAdapter):
    def discover(self, descriptor: VenueDescriptor, window: CrawlWindow, cursor: str | None = None) -> SourceBatch:
        parameters = dict(descriptor.parameters)
        if "volume_id" not in parameters:
            volume_url = parameters.get("volume_url")
            if not volume_url and window.year is not None:
                resolved = self._request(
                    "resolve_volume",
                    {"series": parameters.get("series"), "year": window.year},
                )
                volume_url = resolved.get("official_url")
            match = re.search(r"/v(\d+)(?:/|$)", str(volume_url or ""))
            if not match:
                raise ValueError("pmlr descriptors require volume_id or an official volume_url")
            parameters["volume_id"] = f"v{match.group(1)}"
        return super().discover(VenueDescriptor(descriptor.schema_version, descriptor.venue_id, descriptor.provider, descriptor.adapter, parameters), window, cursor)


class ACLAnthologyAdapter(VenueBuiltinAdapter):
    def discover(self, descriptor: VenueDescriptor, window: CrawlWindow, cursor: str | None = None) -> SourceBatch:
        if "snapshot_version" not in descriptor.parameters:
            raise ValueError("acl descriptors require a frozen snapshot_version")
        return super().discover(descriptor, window, cursor)


class IJCAIAdapter(VenueBuiltinAdapter):
    pass


class ArXivProvider(BuiltinProvider):
    pass


class CrossrefProvider(BuiltinProvider):
    pass


class DBLPProvider(BuiltinProvider):
    pass


class SemanticScholarProvider(BuiltinProvider):
    pass


class OpenAlexProvider(BuiltinProvider):
    pass


class PubMedProvider(BuiltinProvider):
    pass


class EuropePMCProvider(BuiltinProvider):
    pass


class UnpaywallProvider(BuiltinProvider):
    pass


def _openreview_accepted(record: Mapping[str, Any]) -> bool:
    content = record.get("content") if isinstance(record.get("content"), Mapping) else {}
    decision = _text(record.get("decision") or content.get("decision"))
    venue = _text(record.get("venue") or content.get("venue"))
    evidence = " ".join(item.casefold() for item in (decision, venue) if item)
    if any(term in evidence for term in ("reject", "withdraw", "desk reject")):
        return False
    accepted_label = any(term in evidence for term in ("accept", "oral", "spotlight", "poster", "conference"))
    return accepted_label or bool(venue and "submitted" not in venue.casefold())


BUILTIN_CLASSES: Mapping[str, type[BuiltinProvider]] = {
    "neurips_proceedings": NeurIPSProceedingsAdapter,
    "pmlr": PMLRAdapter,
    "openreview": OpenReviewAdapter,
    "aaai_ojs": AAAIOJSAdapter,
    "acl_anthology": ACLAnthologyAdapter,
    "cvf_open_access": CVFOpenAccessAdapter,
    "ijcai_proceedings": IJCAIAdapter,
    "eda_proceedings": EDAProceedingsAdapter,
    "ieee_xplore": IEEEXploreAdapter,
    "springer_nature": SpringerNatureAdapter,
    "cell_press": CellPressAdapter,
    "aaas_science": AAASScienceAdapter,
    "arxiv": ArXivProvider,
    "crossref": CrossrefProvider,
    "dblp": DBLPProvider,
    "semantic_scholar": SemanticScholarProvider,
    "openalex": OpenAlexProvider,
    "pubmed": PubMedProvider,
    "europe_pmc": EuropePMCProvider,
    "unpaywall": UnpaywallProvider,
    "user_library": LibrarySeedImporter,
}


def create_builtin(provider: str, transport: Transport, manifest: ProviderManifest | None = None) -> BuiltinProvider:
    """Create only an enabled core built-in provider; optional products stay inert."""

    if provider not in BUILTIN_CLASSES:
        raise ValueError(f"no active built-in implementation for {provider}")
    runtime_manifest = manifest or load_builtin_manifest(provider)
    if not runtime_manifest.enabled or not runtime_manifest.builtin:
        raise ValueError(f"{provider} is not an enabled built-in provider")
    return BUILTIN_CLASSES[provider](provider, transport, runtime_manifest)
