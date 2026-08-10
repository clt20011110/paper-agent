"""Policy-enforced HTTP transport for the public metadata providers.

This module intentionally fetches API metadata only.  It never follows a
landing-page or PDF URL returned by a provider.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from email.utils import parsedate_to_datetime
from hashlib import sha256
import json
import os
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen
from xml.etree import ElementTree

from paper_agent.provider_runtime import (
    BulkSnapshot,
    ProviderRequestError,
    ProviderRuntime,
    RetryableProviderError,
    policy_from_manifest,
)


Opener = Callable[..., Any]
_METADATA_PROVIDERS = (
    "arxiv",
    "crossref",
    "dblp",
    "semantic_scholar",
    "openalex",
    "pubmed",
    "europe_pmc",
    "unpaywall",
)
_S2_FIELDS = "paperId,title,abstract,authors,year,venue,externalIds,publicationDate,url"


@dataclass(frozen=True, slots=True)
class CachedResponse:
    body: bytes
    etag: str | None
    last_modified: str | None
    content_type: str = ""


@dataclass(frozen=True, slots=True)
class ApprovedMetadataSnapshot:
    """Approved bytes for replaying one recorded JSON or XML API response.

    This is intentionally a response replay mechanism, not support for any
    provider's bulk data format or a substitute for a provider API client.
    """

    body: bytes
    sha256: str
    content_type: str


@dataclass(slots=True)
class ApprovedSnapshotTransport:
    """Network-free transport for approved metadata response snapshots."""

    responses: Mapping[tuple[str, str], ApprovedMetadataSnapshot]
    runtime: ProviderRuntime
    environment: Mapping[str, str] | None = None

    def __call__(self, provider: str, operation: str, parameters: Mapping[str, Any]) -> Mapping[str, Any]:
        try:
            snapshot = self.responses[(provider, operation)]
        except KeyError as error:
            raise ValueError(f"no approved response snapshot for {provider}:{operation}") from error
        content = self.runtime.request(
            provider,
            query_hash=sha256(
                json.dumps({"provider": provider, "operation": operation, "parameters": parameters}, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest(),
            cursor=str(parameters["cursor"]) if parameters.get("cursor") is not None else None,
            api_version=f"{provider}:approved-response-v1",
            mode="snapshot",
            snapshot=BulkSnapshot(snapshot.body, snapshot.sha256),
            expected_snapshot_hash=snapshot.sha256,
            environment=self.environment,
        )
        payload = ControlledHTTPTransport._decode(content, snapshot.content_type)
        if not isinstance(payload, dict):
            raise ProviderRequestError(f"{provider}: approved snapshot is not an object")
        response = dict(payload)
        if response.get("status") == "ok":
            response["provider_status"] = "ok"
            response["status"] = "success"
        response["raw_response_artifact_hash"] = snapshot.sha256
        return response


@dataclass(slots=True)
class ControlledHTTPTransport:
    """Route built-in metadata operations through one :class:`ProviderRuntime`.

    The default runtime is derived exclusively from the installed provider
    manifests.  A caller can inject a runtime to share the same limits and
    circuit state across multiple transport instances.
    """

    contact: str
    timeout_seconds: float = 15.0
    user_agent: str | None = None
    opener: Opener = urlopen
    runtime: ProviderRuntime | None = None
    environment: Mapping[str, str] | None = None
    _cache: dict[str, CachedResponse] = field(default_factory=dict, init=False)
    _credential_envs: dict[str, dict[str, str]] = field(default_factory=dict, init=False)
    last_request_url: str | None = field(default=None, init=False)
    last_response_sha256: str | None = field(default=None, init=False)
    last_response_body: bytes | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.contact.strip():
            raise ValueError("a provider contact URL or mailto address is required")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.user_agent is None:
            self.user_agent = f"paper-agent/2.0 ({self.contact})"
        if self.runtime is None:
            # Importing here keeps the runtime independent of built-in adapters
            # and makes manifest policy the single source of truth.
            from paper_agent import manifests
            from paper_agent.providers.builtin import load_builtin_manifest
            import yaml

            root = manifests.manifest_directory()
            documents = {
                provider: yaml.safe_load((root / "providers" / f"{provider}.yaml").read_text(encoding="utf-8"))
                for provider in _METADATA_PROVIDERS
            }
            self._credential_envs = {
                provider: _credential_environment_variables(document)
                for provider, document in documents.items()
            }

            self.runtime = ProviderRuntime(
                {
                    provider: replace(
                        policy_from_manifest(
                            load_builtin_manifest(provider),
                            terms_accepted=document["terms"].get("data_use") == "permitted",
                            robots_allowed=True,
                        ),
                        credentials_required=bool(document["authentication"].get("required", False)),
                        credential_environment_variables=tuple(self._credential_envs[provider].values()),
                    )
                    for provider, document in documents.items()
                }
            )
        else:
            # An injected runtime supplies policy, but credential names remain
            # manifest-owned rather than becoming an arbitrary caller contract.
            self._credential_envs = self._load_credential_environment_variables()

    def __call__(self, provider: str, operation: str, parameters: Mapping[str, Any]) -> Mapping[str, Any]:
        if provider not in _METADATA_PROVIDERS:
            raise ValueError(f"no public HTTP mapping for {provider}:{operation}")
        payload, bodies = self._operation(provider, operation, parameters)
        self.last_response_body = b"".join(bodies)
        self.last_response_sha256 = sha256(self.last_response_body).hexdigest()
        # Do not mutate a value that can have been returned from ProviderRuntime's
        # cache: callers may safely add their own envelope data.
        response = dict(payload)
        if response.get("status") == "ok":
            response["provider_status"] = "ok"
            response["status"] = "success"
        response["raw_response_artifact_hash"] = self.last_response_sha256
        return response

    def _operation(
        self, provider: str, operation: str, parameters: Mapping[str, Any]
    ) -> tuple[Mapping[str, Any], tuple[bytes, ...]]:
        if provider == "pubmed" and operation in {"search", "enrich", "verify"}:
            return self._pubmed(operation, parameters)
        if provider == "openalex" and operation == "references":
            return self._openalex_references(parameters)
        if provider == "openalex" and operation == "citations":
            return self._openalex_citations(parameters)
        url = self._url(provider, operation, parameters)
        response = self._fetch(provider, url)
        payload = self._decode(response.body, response.content_type)
        if not isinstance(payload, dict):
            raise ProviderRequestError(f"{provider}: metadata response must be an object")
        if operation == "verify":
            return self._verification_payload(payload), (response.body,)
        if operation == "enrich":
            return _enrichment_payload(provider, payload), (response.body,)
        if operation == "resolve":
            return _access_payload(provider, payload), (response.body,)
        return payload, (response.body,)

    def _url(self, provider: str, operation: str, parameters: Mapping[str, Any]) -> str:
        if provider == "crossref":
            if operation == "search":
                query = _copy_parameters(parameters, "cursor", "page_size", "date_from", "date_to")
                query.setdefault("rows", parameters.get("page_size") or 20)
                if parameters.get("cursor"):
                    query["cursor"] = parameters["cursor"]
                if "filter" not in query:
                    filters = _date_filters(parameters)
                    if filters:
                        query["filter"] = ",".join(filters)
                if "@" in self.contact:
                    query.setdefault("mailto", self.contact.removeprefix("mailto:"))
                return _url("https://api.crossref.org/works", query)
            if operation == "enrich":
                doi = _required(parameters, "doi", "crossref enrich")
                return f"https://api.crossref.org/works/{quote(str(doi), safe='') }"
            if operation == "verify":
                query = {"query.bibliographic": _verification_query(parameters), "rows": 1}
                if "@" in self.contact:
                    query["mailto"] = self.contact.removeprefix("mailto:")
                return _url("https://api.crossref.org/works", query)

        if provider == "dblp":
            if operation in {"search", "enrich", "verify"}:
                query = _copy_parameters(parameters, "cursor", "query", "external_id", "doi", "arxiv_id", "title", "page_size")
                query["q"] = str(query.get("q") or _identifier_or_query(parameters))
                query["format"] = "json"
                query.setdefault("h", parameters.get("page_size") or 20)
                if parameters.get("cursor"):
                    query["f"] = parameters["cursor"]
                return _url("https://dblp.org/search/publ/api", query)

        if provider == "semantic_scholar":
            if operation == "search":
                query = _copy_parameters(parameters, "cursor", "page_size")
                query.setdefault("query", parameters.get("query"))
                query.setdefault("limit", parameters.get("page_size") or 20)
                query.setdefault("fields", _S2_FIELDS)
                if parameters.get("cursor"):
                    query["offset"] = parameters["cursor"]
                return _url("https://api.semanticscholar.org/graph/v1/paper/search", query)
            if operation in {"enrich", "verify"}:
                identifier = _semantic_identifier(parameters)
                return _url(
                    f"https://api.semanticscholar.org/graph/v1/paper/{quote(identifier, safe=':')}",
                    {"fields": _S2_FIELDS},
                )
            if operation in {"citations", "references"}:
                identifier = _semantic_identifier(parameters)
                query = {"limit": parameters.get("page_size") or 100, "fields": _S2_FIELDS}
                if parameters.get("cursor"):
                    query["offset"] = parameters["cursor"]
                return _url(
                    f"https://api.semanticscholar.org/graph/v1/paper/{quote(identifier, safe=':')}/{operation}", query
                )

        if provider == "openalex":
            if operation == "search":
                query = _copy_parameters(parameters, "cursor", "page_size")
                query.setdefault("per-page", parameters.get("page_size") or 20)
                if parameters.get("cursor"):
                    query["cursor"] = parameters["cursor"]
                return _url("https://api.openalex.org/works", query)
            identifier = _openalex_identifier(parameters)
            if operation in {"enrich", "verify"}:
                return f"https://api.openalex.org/works/{quote(identifier, safe=':/')}"
            if operation == "resolve":
                return f"https://api.openalex.org/works/{quote(identifier, safe=':/')}"
            if operation == "citations":
                query = {"filter": f"cites:{identifier}", "per-page": parameters.get("page_size") or 100}
                if parameters.get("cursor"):
                    query["cursor"] = parameters["cursor"]
                return _url("https://api.openalex.org/works", query)

        if provider == "europe_pmc" and operation in {"search", "enrich", "verify", "resolve"}:
            query = _copy_parameters(parameters, "cursor", "page_size", "doi", "external_id", "arxiv_id", "paper_id", "purpose")
            query["query"] = str(query.get("query") or _europe_pmc_query(operation, parameters))
            query["format"] = "json"
            query.setdefault("pageSize", parameters.get("page_size") or 20)
            if parameters.get("cursor"):
                query["cursorMark"] = parameters["cursor"]
            return _url("https://www.ebi.ac.uk/europepmc/webservices/rest/search", query)

        if provider == "arxiv" and operation in {"search", "enrich", "verify", "resolve"}:
            query = _copy_parameters(parameters, "cursor", "query", "page_size", "external_id", "doi", "arxiv_id", "paper_id", "purpose")
            if operation != "search":
                query = {"search_query": f"id:{_required(parameters, 'arxiv_id', 'arXiv metadata lookup')}", "start": 0, "max_results": 1}
            else:
                query.setdefault("search_query", parameters.get("query"))
                query.setdefault("start", parameters.get("cursor") or 0)
                query.setdefault("max_results", parameters.get("page_size") or 20)
            return _url("https://export.arxiv.org/api/query", query)

        if provider == "unpaywall" and operation == "resolve":
            doi = _required(parameters, "doi", "Unpaywall resolve")
            email = self._credentials("unpaywall").get("email") or self.contact.removeprefix("mailto:")
            if "@" not in email:
                raise ValueError("Unpaywall requires an operator email contact")
            return _url(f"https://api.unpaywall.org/v2/{quote(str(doi), safe='')}", {"email": email})
        raise ValueError(f"no public HTTP mapping for {provider}:{operation}")

    def _pubmed(self, operation: str, parameters: Mapping[str, Any]) -> tuple[Mapping[str, Any], tuple[bytes, ...]]:
        if operation == "search":
            term = str(parameters.get("term") or parameters.get("query") or "")
        else:
            identifier = parameters.get("external_id") or parameters.get("doi")
            term = f"{identifier}[uid]" if parameters.get("external_id") else f"{identifier}[doi]"
            if not identifier:
                term = _verification_query(parameters)
        query = _copy_parameters(parameters, "cursor", "query", "page_size", "external_id", "doi", "arxiv_id", "title")
        query.update({"db": "pubmed", "term": term, "retmode": "json"})
        query.setdefault("retmax", parameters.get("page_size") or 20)
        query["retstart"] = parameters.get("cursor") or query.get("retstart") or 0
        search_response = self._fetch("pubmed", _url("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi", query))
        search = self._decode(search_response.body, search_response.content_type)
        result = search.get("esearchresult", {}) if isinstance(search, Mapping) else {}
        ids = [str(identifier) for identifier in result.get("idlist", ())]
        if not ids:
            return {"result": {"uids": []}}, (search_response.body,)
        summary_response = self._fetch(
            "pubmed",
            _url("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi", {"db": "pubmed", "id": ",".join(ids), "retmode": "json"}),
        )
        summary = self._decode(summary_response.body, summary_response.content_type)
        if not isinstance(summary, dict):
            raise ProviderRequestError("pubmed: ESummary response must be an object")
        if operation == "verify":
            return self._verification_payload(summary), (search_response.body, summary_response.body)
        count = int(result.get("count", len(ids)))
        start = int(query["retstart"])
        if start + len(ids) < count:
            summary = {**summary, "next_cursor": str(start + len(ids))}
        return summary, (search_response.body, summary_response.body)

    def _openalex_references(self, parameters: Mapping[str, Any]) -> tuple[Mapping[str, Any], tuple[bytes, ...]]:
        work_response = self._fetch("openalex", self._url("openalex", "enrich", parameters))
        work = self._decode(work_response.body, work_response.content_type)
        if not isinstance(work, dict):
            raise ProviderRequestError("openalex: work response must be an object")
        references = tuple(str(value) for value in work.get("referenced_works", ()))
        start = int(parameters.get("cursor") or 0)
        identifiers = tuple(value.rsplit("/", 1)[-1] for value in references[start : start + 100])
        if not identifiers:
            return {"results": []}, (work_response.body,)
        references_response = self._fetch(
            "openalex",
            _url(
                "https://api.openalex.org/works",
                {"filter": f"openalex:{'|'.join(identifiers)}", "per-page": len(identifiers)},
            ),
        )
        payload = self._decode(references_response.body, references_response.content_type)
        if not isinstance(payload, dict):
            raise ProviderRequestError("openalex: referenced works response must be an object")
        next_cursor = str(start + len(identifiers)) if start + len(identifiers) < len(references) else None
        return {**payload, "next_cursor": next_cursor}, (work_response.body, references_response.body)

    def _openalex_citations(self, parameters: Mapping[str, Any]) -> tuple[Mapping[str, Any], tuple[bytes, ...]]:
        work_response = self._fetch("openalex", self._url("openalex", "enrich", parameters))
        work = self._decode(work_response.body, work_response.content_type)
        if not isinstance(work, Mapping) or not work.get("id"):
            raise ProviderRequestError("openalex: citation expansion requires a resolved work ID")
        query = {
            "filter": f"cites:{str(work['id']).rsplit('/', 1)[-1]}",
            "per-page": parameters.get("page_size") or 100,
        }
        if parameters.get("cursor"):
            query["cursor"] = parameters["cursor"]
        citations_response = self._fetch("openalex", _url("https://api.openalex.org/works", query))
        citations = self._decode(citations_response.body, citations_response.content_type)
        if not isinstance(citations, dict):
            raise ProviderRequestError("openalex: citations response must be an object")
        return citations, (work_response.body, citations_response.body)

    def _fetch(self, provider: str, url: str) -> CachedResponse:
        credentials = self._credentials(provider)
        request_url = _with_request_credentials(provider, url, credentials)
        audit_url = _redacted_url(request_url, credentials)
        self.last_request_url = audit_url
        assert self.runtime is not None
        return self.runtime.request(
            provider,
            query_hash=sha256(audit_url.encode("utf-8")).hexdigest(),
            cursor=_request_cursor(request_url),
            api_version=_api_version(provider),
            send=lambda: self._read(request_url, provider, credentials),
            environment=self.environment,
        )  # type: ignore[return-value]

    def _read(self, url: str, provider: str, credentials: Mapping[str, str]) -> CachedResponse:
        headers = {"Accept": "application/json, application/xml;q=0.9", "User-Agent": str(self.user_agent)}
        if provider == "semantic_scholar" and credentials.get("api_key"):
            headers["x-api-key"] = credentials["api_key"]
        cached = self._cache.get(url)
        if cached and cached.etag:
            headers["If-None-Match"] = cached.etag
        if cached and cached.last_modified:
            headers["If-Modified-Since"] = cached.last_modified
        request = Request(url, headers=headers)
        try:
            with self.opener(request, timeout=self.timeout_seconds) as response:
                response_value = CachedResponse(
                    response.read(),
                    response.headers.get("ETag"),
                    response.headers.get("Last-Modified"),
                    response.headers.get("Content-Type", ""),
                )
        except HTTPError as error:
            if error.code == 304 and cached:
                return cached
            self._raise_http_error(error, _redacted_url(url, credentials))
        except URLError as error:
            raise ProviderRequestError(f"HTTP request failed for {_redacted_url(url, credentials)}: {error.reason}") from error
        self._cache[url] = response_value
        return response_value

    def _raise_http_error(self, error: HTTPError, url: str) -> None:
        retry_after = _retry_after(error.headers.get("Retry-After"))
        message = f"HTTP {error.code} for {url}"
        if error.code == 429 or 500 <= error.code < 600:
            raise RetryableProviderError(message, retry_after=retry_after) from error
        raise ProviderRequestError(message, retry_after=retry_after) from error

    @staticmethod
    def _decode(body: bytes, content_type: str) -> Any:
        if "json" in content_type.casefold() or body.lstrip().startswith((b"{", b"[")):
            return json.loads(body)
        return _xml_object(ElementTree.fromstring(body))

    @staticmethod
    def _verification_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
        return {"status": "single_source", "evidence": tuple(str(value) for value in _identifiers(payload))}

    def _load_credential_environment_variables(self) -> dict[str, dict[str, str]]:
        from paper_agent import manifests
        import yaml

        root = manifests.manifest_directory()
        return {
            provider: _credential_environment_variables(
                yaml.safe_load((root / "providers" / f"{provider}.yaml").read_text(encoding="utf-8"))
            )
            for provider in _METADATA_PROVIDERS
        }

    def _credentials(self, provider: str) -> dict[str, str]:
        environment = self.environment if self.environment is not None else os.environ
        return {
            alias: value
            for alias, name in self._credential_envs[provider].items()
            if (value := environment.get(name))
        }


def _url(base: str, parameters: Mapping[str, Any]) -> str:
    return f"{base}?{urlencode({key: value for key, value in parameters.items() if value is not None}, doseq=True)}"


def _copy_parameters(parameters: Mapping[str, Any], *drop: str) -> dict[str, Any]:
    return {key: value for key, value in parameters.items() if key not in drop and value is not None}


def _credential_environment_variables(document: object) -> dict[str, str]:
    authentication = document.get("authentication", {}) if isinstance(document, Mapping) else {}
    if not isinstance(authentication, Mapping):
        return {}
    values: dict[str, str] = {}
    legacy = authentication.get("credential_env")
    if isinstance(legacy, str):
        values["api_key"] = legacy
    declared = authentication.get("credential_envs", {})
    if isinstance(declared, Mapping):
        values.update({str(alias): str(name) for alias, name in declared.items() if isinstance(name, str)})
    return values


def _with_request_credentials(provider: str, url: str, credentials: Mapping[str, str]) -> str:
    if provider == "semantic_scholar":
        return url
    query = dict(parse_qsl(urlsplit(url).query, keep_blank_values=True))
    if provider == "openalex" and credentials.get("api_key"):
        query["api_key"] = credentials["api_key"]
    if provider == "pubmed":
        for name in ("api_key", "tool", "email"):
            if credentials.get(name):
                query[name] = credentials[name]
    if provider == "unpaywall" and credentials.get("email"):
        query["email"] = credentials["email"]
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def _redacted_url(url: str, credentials: Mapping[str, str]) -> str:
    parts = urlsplit(url)
    secret_values = set(credentials.values())
    query = [
        (name, "<redacted>" if value in secret_values else value)
        for name, value in parse_qsl(parts.query, keep_blank_values=True)
    ]
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def _request_cursor(url: str) -> str | None:
    values = dict(parse_qsl(urlsplit(url).query, keep_blank_values=True))
    for name in ("cursor", "offset", "f", "retstart", "cursorMark", "start"):
        if values.get(name):
            return values[name]
    return None


def _api_version(provider: str) -> str:
    return {
        "arxiv": "atom-api-v1",
        "crossref": "rest-v1",
        "dblp": "publ-api-v1",
        "semantic_scholar": "graph-v1",
        "openalex": "works-api-v1",
        "pubmed": "eutils-v1",
        "europe_pmc": "rest-v1",
        "unpaywall": "v2",
    }[provider]


def _required(parameters: Mapping[str, Any], name: str, context: str) -> Any:
    value = parameters.get(name)
    if value is None or not str(value).strip():
        raise ValueError(f"{context} requires {name}")
    return value


def _date_filters(parameters: Mapping[str, Any]) -> list[str]:
    filters = []
    if parameters.get("date_from"):
        filters.append(f"from-pub-date:{parameters['date_from']}")
    if parameters.get("date_to"):
        filters.append(f"until-pub-date:{parameters['date_to']}")
    return filters


def _identifier_or_query(parameters: Mapping[str, Any]) -> str:
    return str(parameters.get("external_id") or parameters.get("doi") or parameters.get("query") or parameters.get("title") or "")


def _semantic_identifier(parameters: Mapping[str, Any]) -> str:
    if parameters.get("doi"):
        return f"DOI:{parameters['doi']}"
    if parameters.get("arxiv_id"):
        return f"ARXIV:{parameters['arxiv_id']}"
    return str(_required(parameters, "external_id", "Semantic Scholar metadata lookup"))


def _openalex_identifier(parameters: Mapping[str, Any]) -> str:
    if parameters.get("doi"):
        return f"doi:{parameters['doi']}"
    if parameters.get("arxiv_id"):
        return f"https://arxiv.org/abs/{parameters['arxiv_id']}"
    return str(_required(parameters, "external_id", "OpenAlex metadata lookup"))


def _europe_pmc_query(operation: str, parameters: Mapping[str, Any]) -> str:
    if parameters.get("doi"):
        return f"DOI:{parameters['doi']}"
    if parameters.get("external_id"):
        return f"EXT_ID:{parameters['external_id']}"
    if operation == "resolve":
        raise ValueError("Europe PMC resolve requires DOI or external_id")
    return _verification_query(parameters)


def _verification_query(parameters: Mapping[str, Any]) -> str:
    return str(parameters.get("doi") or parameters.get("arxiv_id") or parameters.get("title") or "")


def _identifiers(payload: Mapping[str, Any]) -> tuple[str, ...]:
    for key in ("message", "result", "data", "results"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            return tuple(str(item) for item in value.get("uids", ()) if item)
        if isinstance(value, list):
            return tuple(str(item.get("id") or item.get("paperId") or item.get("DOI")) for item in value if isinstance(item, Mapping))
    return ()


def _enrichment_payload(provider: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Normalize documented singleton responses to each adapter's list envelope."""
    if provider == "crossref" and isinstance(payload.get("message"), Mapping):
        return {**payload, "message": {"items": [payload["message"]]}}
    if provider == "semantic_scholar" and "paperId" in payload:
        return {"data": [payload]}
    if provider == "openalex" and ("id" in payload or "ids" in payload):
        return {"results": [payload]}
    return payload


def _access_payload(provider: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Expose returned OA locations without dereferencing their article URLs."""
    if provider == "openalex":
        locations = [payload.get("best_oa_location"), *payload.get("locations", ())]
        return {
            "entries": [
                {
                    "url": location.get("pdf_url") or location.get("landing_page_url"),
                    "landing_url": location.get("landing_page_url"),
                    "version": location.get("version"),
                    "license": location.get("license"),
                }
                for location in locations
                if isinstance(location, Mapping) and (location.get("pdf_url") or location.get("landing_page_url"))
            ]
        }
    if provider == "europe_pmc":
        records = payload.get("resultList", {})
        results = records.get("result", ()) if isinstance(records, Mapping) else ()
        return {
            "entries": [
                {"url": location.get("url"), "landing_url": f"https://europepmc.org/article/{record.get('source')}/{record.get('id')}"}
                for record in results if isinstance(record, Mapping)
                for location in (record.get("fullTextUrlList", {}).get("fullTextUrl", ()) if isinstance(record.get("fullTextUrlList"), Mapping) else ())
                if isinstance(location, Mapping) and location.get("url")
            ]
        }
    if provider == "arxiv":
        feed = payload.get("feed", {})
        entries = feed.get("entry", ()) if isinstance(feed, Mapping) else ()
        if isinstance(entries, Mapping):
            entries = (entries,)
        return {
            "entries": [
                {
                    "url": str(entry["id"]).replace("/abs/", "/pdf/"),
                    "landing_url": entry["id"],
                    "version": "preprint",
                }
                for entry in entries
                if isinstance(entry, Mapping) and entry.get("id")
            ]
        }
    return payload


def _retry_after(value: str | None) -> float | None:
    if value is None:
        return None
    if value.isdigit():
        return float(value)
    return max(0.0, parsedate_to_datetime(value).timestamp() - __import__("time").time())


def _xml_object(element: ElementTree.Element) -> dict[str, Any]:
    children = list(element)
    name = element.tag.rsplit("}", 1)[-1]
    if not children:
        return {name: (element.text or "").strip()}
    result: dict[str, Any] = {}
    for child in children:
        child_name, value = next(iter(_xml_object(child).items()))
        current = result.get(child_name)
        if current is None:
            result[child_name] = value
        elif isinstance(current, list):
            current.append(value)
        else:
            result[child_name] = [current, value]
    return {name: result}
