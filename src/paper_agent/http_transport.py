"""Small, policy-visible HTTP transport for public metadata providers.

The transport intentionally supports metadata responses only.  It does not
follow or download article/PDF links.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from hashlib import sha256
import json
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from xml.etree import ElementTree

from paper_agent.provider_runtime import ProviderRequestError, RetryableProviderError


Opener = Callable[..., Any]


@dataclass(frozen=True, slots=True)
class CachedResponse:
    body: bytes
    etag: str | None
    last_modified: str | None


@dataclass(slots=True)
class ControlledHTTPTransport:
    """Map built-in provider calls to documented public JSON/XML endpoints."""

    contact: str
    timeout_seconds: float = 15.0
    user_agent: str | None = None
    opener: Opener = urlopen
    _cache: dict[str, CachedResponse] = field(default_factory=dict, init=False)
    last_request_url: str | None = field(default=None, init=False)
    last_response_sha256: str | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if not self.contact.strip():
            raise ValueError("a provider contact URL or mailto address is required")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.user_agent is None:
            self.user_agent = f"paper-agent/2.0 ({self.contact})"

    def __call__(self, provider: str, operation: str, parameters: Mapping[str, Any]) -> Mapping[str, Any]:
        url = self._url(provider, operation, parameters)
        self.last_request_url = url
        body, headers = self._read(url)
        self.last_response_sha256 = sha256(body).hexdigest()
        payload = self._decode(body, headers.get("Content-Type", ""))
        if not isinstance(payload, dict):
            raise ValueError(f"{provider}: metadata response must be an object")
        if payload.get("status") == "ok":
            payload["provider_status"] = "ok"
            payload["status"] = "success"
        payload["raw_response_artifact_hash"] = self.last_response_sha256
        return payload

    def _url(self, provider: str, operation: str, parameters: Mapping[str, Any]) -> str:
        if provider == "crossref" and operation == "search":
            query: dict[str, Any] = {"query": parameters["query"], "rows": parameters.get("page_size") or 20}
            if parameters.get("cursor"):
                query["cursor"] = parameters["cursor"]
            filters = []
            if parameters.get("date_from"):
                filters.append(f"from-pub-date:{parameters['date_from']}")
            if parameters.get("date_to"):
                filters.append(f"until-pub-date:{parameters['date_to']}")
            if filters:
                query["filter"] = ",".join(filters)
            return f"https://api.crossref.org/works?{urlencode(query)}"
        if provider == "crossref" and operation == "enrich":
            doi = parameters.get("doi")
            if not doi:
                raise ValueError("crossref enrich requires doi")
            return f"https://api.crossref.org/works/{urlencode({'doi': str(doi)})[4:]}"
        if provider == "europe_pmc" and operation == "search":
            query = {"query": parameters["query"], "format": "json", "pageSize": parameters.get("page_size") or 20}
            if parameters.get("cursor"):
                query["cursorMark"] = parameters["cursor"]
            return f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?{urlencode(query)}"
        raise ValueError(f"no public HTTP mapping for {provider}:{operation}")

    def _read(self, url: str) -> tuple[bytes, Mapping[str, str]]:
        headers = {"Accept": "application/json, application/xml;q=0.9", "User-Agent": str(self.user_agent)}
        cached = self._cache.get(url)
        if cached and cached.etag:
            headers["If-None-Match"] = cached.etag
        if cached and cached.last_modified:
            headers["If-Modified-Since"] = cached.last_modified
        request = Request(url, headers=headers)
        try:
            with self.opener(request, timeout=self.timeout_seconds) as response:
                body = response.read()
                response_headers = dict(response.headers.items())
        except HTTPError as error:
            if error.code == 304 and cached:
                return cached.body, dict(error.headers.items())
            self._raise_http_error(error)
        except URLError as error:
            raise ProviderRequestError(f"HTTP request failed for {url}: {error.reason}") from error
        self._cache[url] = CachedResponse(body, response_headers.get("ETag"), response_headers.get("Last-Modified"))
        return body, response_headers

    def _raise_http_error(self, error: HTTPError) -> None:
        retry_after = _retry_after(error.headers.get("Retry-After"))
        message = f"HTTP {error.code} for {error.url}"
        if error.code == 429 or 500 <= error.code < 600:
            raise RetryableProviderError(message, retry_after=retry_after) from error
        raise ProviderRequestError(message, retry_after=retry_after) from error

    @staticmethod
    def _decode(body: bytes, content_type: str) -> Any:
        if "json" in content_type.casefold() or body.lstrip().startswith((b"{", b"[")):
            return json.loads(body)
        return _xml_object(ElementTree.fromstring(body))


def _retry_after(value: str | None) -> float | None:
    if value is None:
        return None
    if value.isdigit():
        return float(value)
    return max(0.0, parsedate_to_datetime(value).timestamp() - __import__("time").time())


def _xml_object(element: ElementTree.Element) -> dict[str, Any]:
    children = list(element)
    if not children:
        return {element.tag.rsplit("}", 1)[-1]: element.text or ""}
    result: dict[str, Any] = {}
    for child in children:
        name = child.tag.rsplit("}", 1)[-1]
        value = _xml_object(child)[name]
        current = result.get(name)
        if current is None:
            result[name] = value
        elif isinstance(current, list):
            current.append(value)
        else:
            result[name] = [current, value]
    return {element.tag.rsplit("}", 1)[-1]: result}
