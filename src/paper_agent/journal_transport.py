"""Official, metadata-only HTTP routes for the journal primary adapters.

This module is intentionally separate from the general scholarly-graph
transport.  It makes one read-only request through ``ProviderRuntime`` and
converts each publisher's native response into the small envelope consumed by
``VenueBuiltinAdapter``.  It never follows an article, HTML full-text, or PDF
link returned in metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from hashlib import sha256
from html.parser import HTMLParser
import json
import os
from typing import Any, Callable, Mapping, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen
from xml.etree import ElementTree

from paper_agent.provider_runtime import ProviderRequestError, ProviderRuntime, RetryableProviderError


Opener = Callable[..., Any]
_JOURNAL_PROVIDERS = {"ieee_xplore", "springer_nature", "cell_press", "aaas_science"}
_CREDENTIALS = {
    "ieee_xplore": "IEEE_XPLORE_API_KEY",
    "springer_nature": "SPRINGER_NATURE_API_KEY",
    "cell_press": "ELSEVIER_API_KEY",
}


class JournalFetchResponse(Protocol):
    body: bytes
    content_type: str


JournalFetch = Callable[[str, str], JournalFetchResponse]


@dataclass(frozen=True, slots=True)
class JournalOperationResult:
    payload: Mapping[str, Any]
    bodies: tuple[bytes, ...]


def journal_provider_names() -> tuple[str, ...]:
    return tuple(sorted(_JOURNAL_PROVIDERS))


def execute_journal_operation(
    provider: str,
    operation: str,
    parameters: Mapping[str, Any],
    fetch: JournalFetch,
) -> JournalOperationResult:
    """Execute one journal route through a caller-owned policy/runtime fetch."""
    if provider not in _JOURNAL_PROVIDERS or operation != "discover":
        raise ValueError(f"no official journal mapping for {provider}:{operation}")
    response = fetch(_journal_url(provider, parameters), _api_version(provider))
    payload = _decode(response.body, response.content_type)
    return JournalOperationResult(
        {
            "entries": _entries(provider, payload),
            "next_cursor": _next_cursor(provider, payload, parameters),
        },
        (response.body,),
    )
@dataclass(slots=True)
class JournalHTTPTransport:
    """Map four official journal discovery APIs/pages through a shared runtime."""

    runtime: ProviderRuntime
    opener: Opener = urlopen
    environment: Mapping[str, str] | None = None
    timeout_seconds: float = 15.0
    user_agent: str = "paper-agent/2.0 journal-metadata"
    last_request_url: str | None = field(default=None, init=False)
    last_response_sha256: str | None = field(default=None, init=False)

    def __call__(self, provider: str, operation: str, parameters: Mapping[str, Any]) -> Mapping[str, Any]:
        if provider not in _JOURNAL_PROVIDERS or operation != "discover":
            raise ValueError(f"no official journal mapping for {provider}:{operation}")
        url = self.url_for(provider, parameters)
        self.last_request_url = url
        payload, body = self.runtime.request(
            provider,
            query_hash=sha256(json.dumps({"provider": provider, "parameters": parameters}, sort_keys=True, default=str).encode()).hexdigest(),
            cursor=str(parameters["cursor"]) if parameters.get("cursor") is not None else None,
            api_version=_api_version(provider),
            send=lambda: self._fetch(provider, url),
            environment=self.environment,
        )
        self.last_response_sha256 = sha256(body).hexdigest()
        return {
            "entries": _entries(provider, payload),
            "next_cursor": _next_cursor(provider, payload, parameters),
            "raw_response_artifact_hash": self.last_response_sha256,
        }

    def url_for(self, provider: str, parameters: Mapping[str, Any]) -> str:
        """Return the publisher-documented metadata URL for an exact descriptor."""
        return _journal_url(provider, parameters)

    def _credential(self, provider: str) -> str:
        name = _CREDENTIALS.get(provider)
        if name is None:
            return ""
        values = self.environment if self.environment is not None else os.environ
        return str(values.get(name) or "")

    def _fetch(self, provider: str, url: str) -> tuple[Any, bytes]:
        headers = {"Accept": "application/json, application/rss+xml, text/html", "User-Agent": self.user_agent}
        if provider == "cell_press":
            headers["X-ELS-APIKey"] = self._credential("cell_press")
        request = Request(_authorized_url(provider, url, self._credential(provider)), headers=headers)
        try:
            with self.opener(request, timeout=self.timeout_seconds) as response:
                body = response.read()
                content_type = response.headers.get("Content-Type", "")
        except HTTPError as error:
            if error.code == 429 or 500 <= error.code < 600:
                raise RetryableProviderError(f"HTTP {error.code} for {url}") from error
            raise ProviderRequestError(f"HTTP {error.code} for {url}") from error
        except URLError as error:
            raise RetryableProviderError(f"network error for {url}: {error.reason}") from error
        return _decode(body, content_type), body


def _journal_url(
    provider: str,
    parameters: Mapping[str, Any],
) -> str:
    if provider == "ieee_xplore":
        _exact_ieee(parameters)
        query: dict[str, Any] = {
            "publication_number": 43,
            "issn": _ieee_issn(parameters),
            "content_type": "Journals",
            "start_record": int(parameters.get("cursor") or 1),
            "max_records": min(int(parameters.get("page_size") or 100), 200),
            "format": "json",
        }
        if parameters.get("date_from"):
            query["start_year"] = str(parameters["date_from"])[:4]
        if parameters.get("date_to"):
            query["end_year"] = str(parameters["date_to"])[:4]
        if parameters.get("year"):
            query["start_year"] = query["end_year"] = parameters["year"]
        if parameters.get("issue"):
            query["is_number"] = parameters["issue"]
        if parameters.get("volume"):
            query["volume"] = parameters["volume"]
        return _url("https://ieeexploreapi.ieee.org/api/v1/search/articles", query)

    if provider == "springer_nature":
        issns = tuple(str(value) for value in parameters.get("issns", ()))
        if not issns or not parameters.get("journal_slug") or not parameters.get("article_types"):
            raise ValueError("springer_nature requires journal_slug, issns, and article_types")
        clauses = ["(" + " OR ".join(f"issn:{value}" for value in issns) + ")"]
        clauses.append("(" + " OR ".join(f"articletype:{value}" for value in parameters["article_types"]) + ")")
        if parameters.get("volume"):
            clauses.append(f"volume:{parameters['volume']}")
        if parameters.get("issue"):
            clauses.append(f"issue:{parameters['issue']}")
        if parameters.get("date_from"):
            clauses.append(f"onlinedatefrom:{parameters['date_from']}")
        if parameters.get("date_to"):
            clauses.append(f"onlinedateto:{parameters['date_to']}")
        if parameters.get("year") and not parameters.get("date_from") and not parameters.get("date_to"):
            clauses.append(f"year:{parameters['year']}")
        return _url(
            "https://api.springernature.com/metadata/v1/articles",
            {
                "q": " AND ".join(clauses),
                "s": int(parameters.get("cursor") or 1),
                "p": min(int(parameters.get("page_size") or 100), 100),
            },
        )

    if provider == "cell_press":
        if set(parameters.get("issns", (parameters.get("issn"),))) != {"0092-8674"}:
            raise ValueError("cell_press requires Cell ISSN 0092-8674")
        clauses = ["issn(0092-8674)"]
        if parameters.get("volume"):
            clauses.append(f"volume({parameters['volume']})")
        if parameters.get("issue"):
            clauses.append(f"issue({parameters['issue']})")
        return _url(
            "https://api.elsevier.com/content/metadata/article",
            {
                "query": " AND ".join(clauses),
                "date": _year_range(parameters),
                "start": int(parameters.get("cursor") or 0),
                "count": _elsevier_page_size(parameters),
                "sort": "coverDate",
                "httpAccept": "application/json",
            },
        )

    if provider != "aaas_science":
        raise ValueError(f"no official journal mapping for {provider}:discover")
    if set(parameters.get("issns", ())) != {"0036-8075", "1095-9203"}:
        raise ValueError("aaas_science requires Science ISSNs 0036-8075 and 1095-9203")
    if parameters.get("cursor"):
        year = parameters.get("year") or str(parameters.get("date_from") or "")[:4]
        if year:
            return f"https://www.science.org/toc/science/{year}?page={parameters['cursor']}"
        raise ValueError("aaas_science cursor requires a year or date_from")
    if parameters.get("volume") and parameters.get("issue"):
        return f"https://www.science.org/toc/science/{parameters['volume']}/{parameters['issue']}"
    year = parameters.get("year") or str(parameters.get("date_from") or "")[:4]
    if year:
        return f"https://www.science.org/toc/science/{year}"
    return "https://www.science.org/action/showFeed?type=etoc&feed=rss&jc=science"


def _exact_ieee(parameters: Mapping[str, Any]) -> None:
    if parameters.get("publication_number") != 43 or _ieee_issn(parameters) != "0278-0070":
        raise ValueError("ieee_xplore requires TCAD publication_number=43 and ISSN 0278-0070")


def _ieee_issn(parameters: Mapping[str, Any]) -> str:
    values = parameters.get("issns", (parameters.get("issn"),))
    if set(str(value) for value in values if value) != {"0278-0070"}:
        return ""
    return "0278-0070"


def _url(base: str, query: Mapping[str, Any]) -> str:
    values = {key: value for key, value in query.items() if value not in (None, "")}
    return f"{base}?{urlencode(values, doseq=True)}"


def _authorized_url(provider: str, url: str, credential: str) -> str:
    if not credential or provider == "cell_press":
        return url
    key = "apikey" if provider == "ieee_xplore" else "api_key"
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query[key] = credential
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def _year_range(parameters: Mapping[str, Any]) -> str | None:
    if parameters.get("year"):
        return str(parameters["year"])
    start = str(parameters.get("date_from") or "")[:4]
    end = str(parameters.get("date_to") or "")[:4]
    return f"{start}-{end}" if start and end else start or end or None


def _elsevier_page_size(parameters: Mapping[str, Any]) -> int:
    requested = int(parameters.get("page_size") or 100)
    return min((10, 25, 50, 100), key=lambda value: abs(value - requested))


def _decode(body: bytes, content_type: str) -> Any:
    if "json" in content_type.casefold() or body.lstrip().startswith((b"{", b"[")):
        return json.loads(body)
    if "xml" in content_type.casefold() or body.lstrip().startswith(b"<rss"):
        return _rss_entries(body)
    return _ScienceTOCParser.parse(body.decode("utf-8"))


def _entries(provider: str, payload: Any) -> list[dict[str, Any]]:
    if provider == "ieee_xplore":
        return [_ieee_entry(value) for value in payload.get("articles", ())]
    if provider == "springer_nature":
        return [_springer_entry(value) for value in payload.get("records", ())]
    if provider == "cell_press":
        results = payload.get("search-results", {}).get("entry", ())
        return [_elsevier_entry(value) for value in results]
    return list(payload.get("entries", ()))


def _ieee_entry(record: Mapping[str, Any]) -> dict[str, Any]:
    authors = record.get("authors", {}).get("authors", ()) if isinstance(record.get("authors"), Mapping) else ()
    return {
        "stable_id": record["article_number"], "title": record["title"], "doi": record.get("doi"),
        "authors": [value.get("full_name") for value in authors if value.get("full_name")],
        "abstract": record.get("abstract"), "publication_date": record.get("publication_date"),
        "year": record.get("publication_year"), "venue": record.get("publication_title"),
        "landing_url": record.get("abstract_url") or record.get("html_url"), "volume": record.get("volume"), "issue": record.get("issue"),
    }


def _springer_entry(record: Mapping[str, Any]) -> dict[str, Any]:
    urls = record.get("url", ())
    landing = next((value.get("value") for value in urls if isinstance(value, Mapping) and value.get("value")), None)
    creators = record.get("creators", ())
    return {
        "stable_id": record.get("doi") or record["identifier"], "title": record["title"], "doi": record.get("doi"),
        "authors": [value.get("creator") for value in creators if isinstance(value, Mapping) and value.get("creator")],
        "abstract": record.get("abstract"), "publication_date": record.get("publicationDate"),
        "venue": record.get("publicationName") or record.get("journalTitle"), "landing_url": landing,
        "volume": record.get("volume"), "issue": record.get("number") or record.get("issue"), "article_type": record.get("contentType"),
    }


def _elsevier_entry(record: Mapping[str, Any]) -> dict[str, Any]:
    links = record.get("link", ())
    landing = next((value.get("@href") for value in links if value.get("@ref") == "scidir"), None)
    return {
        "stable_id": record.get("dc:identifier") or record.get("prism:doi"), "title": record.get("dc:title"),
        "doi": record.get("prism:doi"), "authors": record.get("dc:creator", "").split("; "),
        "abstract": record.get("dc:description"), "publication_date": record.get("prism:coverDate"),
        "venue": record.get("prism:publicationName"), "landing_url": landing,
        "volume": record.get("prism:volume"), "issue": record.get("prism:issueIdentifier"),
    }


def _next_cursor(provider: str, payload: Any, parameters: Mapping[str, Any]) -> str | None:
    if provider == "ieee_xplore":
        start = int(parameters.get("cursor") or 1)
        sent = len(payload.get("articles", ()))
        total = int(payload.get("totalfound") or 0)
        return str(start + sent) if sent and start + sent <= total else None
    if provider == "springer_nature":
        result = next(iter(payload.get("result", ())), {})
        start = int(parameters.get("cursor") or 1)
        sent = len(payload.get("records", ()))
        total = int(result.get("total") or 0)
        return str(start + sent) if sent and start + sent <= total else None
    if provider == "cell_press":
        result = payload.get("search-results", {})
        start = int(parameters.get("cursor") or 0)
        sent = len(result.get("entry", ()))
        total = int(result.get("opensearch:totalResults") or 0)
        return str(start + sent) if sent and start + sent < total else None
    return None


def _rss_entries(body: bytes) -> dict[str, list[dict[str, Any]]]:
    root = ElementTree.fromstring(body)
    entries = []
    for item in (element for element in root.iter() if _local_name(element.tag) == "item"):
        values = {_local_name(element.tag): element.text for element in item.iter() if element is not item}
        link = values.get("url") or values.get("link")
        doi = values.get("doi") or _doi_from_link(link)
        entries.append({
            "stable_id": doi or values.get("guid") or link, "title": values.get("title"), "doi": doi,
            "publication_date": _rss_date(values.get("date") or values.get("pubDate")),
            "venue": values.get("publicationName") or "Science", "landing_url": link, "abstract": values.get("description"),
            "authors": [value.strip() for value in (values.get("creator") or "").split(" and ") if value.strip()],
            "volume": values.get("volume"), "issue": values.get("number"), "article_type": values.get("type"),
        })
    return {"entries": entries}


def _doi_from_link(value: str | None) -> str | None:
    marker = "/doi/"
    return value.split(marker, 1)[1] if value and marker in value else None


def _rss_date(value: str | None) -> str | None:
    if not value:
        return None
    if len(value) > 10 and value[10] == "T":
        return value[:10]
    return parsedate_to_datetime(value).date().isoformat()


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


class _ScienceTOCParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.entries: list[dict[str, Any]] = []
        self._entry: dict[str, Any] | None = None
        self._field: str | None = None

    @classmethod
    def parse(cls, source: str) -> dict[str, list[dict[str, Any]]]:
        parser = cls()
        parser.feed(source)
        return {"entries": parser.entries}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        classes = values.get("class", "")
        if tag == "article" and "issue-item" in classes:
            self._entry = {"venue": "Science"}
        if self._entry is None:
            return
        if tag == "a" and "/doi/" in (values.get("href") or ""):
            href = values["href"]
            self._entry["landing_url"] = href if href.startswith("http") else f"https://www.science.org{href}"
            self._entry["doi"] = _doi_from_link(href)
            self._entry["stable_id"] = self._entry["doi"]
            self._field = "title"
        elif tag == "time":
            if values.get("datetime"):
                self._entry["publication_date"] = values["datetime"]
                self._field = None
            else:
                self._field = "publication_date"
        elif tag == "p" and "abstract" in classes:
            self._field = "abstract"

    def handle_data(self, data: str) -> None:
        if self._entry is not None and self._field:
            self._entry[self._field] = (self._entry.get(self._field, "") + data).strip()

    def handle_endtag(self, tag: str) -> None:
        if tag == "article" and self._entry is not None:
            if self._entry.get("stable_id") and self._entry.get("title"):
                self.entries.append(self._entry)
            self._entry = None
        self._field = None


def _api_version(provider: str) -> str:
    return {"ieee_xplore": "metadata-api-v1", "springer_nature": "metadata-api-v1", "cell_press": "sciencedirect-metadata-v1", "aaas_science": "science-etoc-v1"}[provider]
