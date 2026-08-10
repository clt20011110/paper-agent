from __future__ import annotations

from email.message import Message
from io import BytesIO
from urllib.error import HTTPError

import pytest

from paper_agent.http_transport import ControlledHTTPTransport
from paper_agent.provider_runtime import ProviderRuntime, ProviderRuntimePolicy, RetryableProviderError


class Response:
    def __init__(self, body: bytes, headers: dict[str, str]) -> None:
        self._body = body
        self.headers = Message()
        for key, value in headers.items():
            self.headers[key] = value

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "Response":
        return self

    def __exit__(self, *_: object) -> None:
        return None


@pytest.fixture(autouse=True)
def required_metadata_credentials(monkeypatch) -> None:
    monkeypatch.setenv("OPENALEX_API_KEY", "openalex-test-key")
    monkeypatch.setenv("UNPAYWALL_EMAIL", "operator@example.test")


def test_crossref_request_sets_contact_timeout_and_response_artifact() -> None:
    calls = []

    def opener(request, timeout):
        calls.append((request, timeout))
        return Response(b'{"status":"ok","message":{"items":[{"DOI":"10.1/example","title":["Example"]}]}}', {"Content-Type": "application/json", "ETag": "one"})

    transport = ControlledHTTPTransport("https://example.test/contact", timeout_seconds=4, opener=opener)
    payload = transport("crossref", "search", {"query": "graph learning", "page_size": 1, "cursor": None, "date_from": None, "date_to": None})

    request, timeout = calls[0]
    assert timeout == 4
    assert request.get_header("User-agent") == "paper-agent/2.0 (https://example.test/contact)"
    assert "query=graph+learning" in request.full_url
    assert payload["message"]["items"][0]["DOI"] == "10.1/example"
    assert payload["status"] == "success"
    assert payload["provider_status"] == "ok"
    assert len(payload["raw_response_artifact_hash"]) == 64


def test_conditional_request_reuses_cached_body_on_not_modified() -> None:
    count = 0

    def opener(request, timeout):
        nonlocal count
        count += 1
        if count == 1:
            return Response(b'{"message":{"items":[]}}', {"Content-Type": "application/json", "ETag": "one"})
        assert request.get_header("If-none-match") == "one"
        raise HTTPError(request.full_url, 304, "not modified", Message(), BytesIO())

    transport = ControlledHTTPTransport("https://example.test/contact", opener=opener)
    first = transport("crossref", "search", {"query": "x", "page_size": 1})
    second = transport("crossref", "search", {"query": "x", "page_size": 1})
    assert first["raw_response_artifact_hash"] == second["raw_response_artifact_hash"]


def test_retry_after_is_exposed_for_rate_limits() -> None:
    def opener(request, timeout):
        headers = Message()
        headers["Retry-After"] = "3"
        raise HTTPError(request.full_url, 429, "limited", headers, BytesIO())

    transport = ControlledHTTPTransport("https://example.test/contact", opener=opener)
    with pytest.raises(RetryableProviderError) as error:
        transport("crossref", "search", {"query": "x", "page_size": 1})
    assert error.value.retry_after == 3


def test_xml_metadata_response_is_decoded() -> None:
    def opener(request, timeout):
        return Response(b"<root><record>one</record></root>", {"Content-Type": "application/xml"})

    payload = ControlledHTTPTransport("https://example.test/contact", opener=opener)("crossref", "search", {"query": "x", "page_size": 1})
    assert payload["root"]["record"] == "one"


@pytest.mark.parametrize(
    ("provider", "operation", "parameters", "expected_path"),
    [
        ("crossref", "enrich", {"doi": "10.1/example"}, "/works/10.1%2Fexample"),
        ("dblp", "search", {"query": "graph learning", "page_size": 2}, "/search/publ/api"),
        ("semantic_scholar", "search", {"query": "graph learning", "page_size": 2}, "/graph/v1/paper/search"),
        ("semantic_scholar", "references", {"doi": "10.1/example"}, "/graph/v1/paper/DOI:10.1%2Fexample/references"),
        ("semantic_scholar", "citations", {"doi": "10.1/example"}, "/graph/v1/paper/DOI:10.1%2Fexample/citations"),
        ("openalex", "search", {"search": "graph learning", "page_size": 2}, "/works"),
        ("openalex", "references", {"doi": "10.1/example"}, "/works/doi:10.1/example"),
        ("europe_pmc", "search", {"query": "graph learning", "page_size": 2}, "/europepmc/webservices/rest/search"),
        ("arxiv", "search", {"search_query": "all:graph", "page_size": 2}, "/api/query"),
    ],
)
def test_public_metadata_operations_use_native_official_routes(
    provider: str, operation: str, parameters: dict[str, object], expected_path: str
) -> None:
    urls: list[str] = []

    def opener(request, timeout):
        urls.append(request.full_url)
        return Response(b"{}", {"Content-Type": "application/json"})

    ControlledHTTPTransport("mailto:operator@example.test", opener=opener)(provider, operation, parameters)

    assert expected_path in urls[0]


def test_openalex_citations_resolve_external_id_before_filtering() -> None:
    urls: list[str] = []

    def opener(request, timeout):
        urls.append(request.full_url)
        if "/works/doi:" in request.full_url:
            return Response(
                b'{"id":"https://openalex.org/W42","referenced_works":[]}',
                {"Content-Type": "application/json"},
            )
        return Response(b'{"meta":{},"results":[]}', {"Content-Type": "application/json"})

    ControlledHTTPTransport("operator@example.test", opener=opener)(
        "openalex", "citations", {"doi": "10.1/example"}
    )

    assert "/works/doi:10.1/example" in urls[0]
    assert "filter=cites%3AW42" in urls[1]


def test_openalex_references_batch_resolve_candidate_metadata() -> None:
    urls: list[str] = []

    def opener(request, timeout):
        urls.append(request.full_url)
        if "/works/doi:" in request.full_url:
            return Response(
                b'{"id":"https://openalex.org/W1","referenced_works":["https://openalex.org/W2"]}',
                {"Content-Type": "application/json"},
            )
        return Response(
            b'{"meta":{},"results":[{"id":"https://openalex.org/W2","title":"Cited","ids":{"openalex":"https://openalex.org/W2"},"authorships":[]}]}',
            {"Content-Type": "application/json"},
        )

    payload = ControlledHTTPTransport("operator@example.test", opener=opener)(
        "openalex", "references", {"doi": "10.1/example"}
    )

    assert "filter=openalex%3AW2" in urls[1]
    assert payload["results"][0]["title"] == "Cited"


def test_pubmed_search_uses_esearch_then_esummary_without_full_text() -> None:
    urls: list[str] = []

    def opener(request, timeout):
        urls.append(request.full_url)
        if "esearch.fcgi" in request.full_url:
            return Response(b'{"esearchresult":{"count":"1","idlist":["42"]}}', {"Content-Type": "application/json"})
        return Response(b'{"result":{"uids":["42"],"42":{"uid":"42","title":"Metadata only"}}}', {"Content-Type": "application/json"})

    payload = ControlledHTTPTransport("mailto:operator@example.test", opener=opener)(
        "pubmed", "search", {"term": "metadata", "retmax": 1}
    )

    assert "esearch.fcgi" in urls[0]
    assert "esummary.fcgi" in urls[1]
    assert payload["result"]["uids"] == ["42"]
    assert all("efetch" not in url and ".pdf" not in url for url in urls)


def test_unpaywall_uses_operator_contact_and_never_fetches_the_returned_pdf() -> None:
    urls: list[str] = []

    def opener(request, timeout):
        urls.append(request.full_url)
        return Response(b'{"best_oa_location":{"url_for_pdf":"https://example.test/paper.pdf"}}', {"Content-Type": "application/json"})

    payload = ControlledHTTPTransport("operator@example.test", opener=opener)(
        "unpaywall", "resolve", {"doi": "10.1/example"}
    )

    assert "/v2/10.1%2Fexample" in urls[0]
    assert "email=operator%40example.test" in urls[0]
    assert payload["best_oa_location"]["url_for_pdf"].endswith(".pdf")
    assert len(urls) == 1


def test_transport_uses_injected_runtime_for_retries() -> None:
    attempts = 0
    runtime = ProviderRuntime(
        {"crossref": ProviderRuntimePolicy("crossref", retry_attempts=2, initial_backoff_seconds=0, max_backoff_seconds=0)}
    )

    def opener(request, timeout):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise HTTPError(request.full_url, 503, "down", Message(), BytesIO())
        return Response(b'{"message":{"items":[]}}', {"Content-Type": "application/json"})

    payload = ControlledHTTPTransport("operator@example.test", opener=opener, runtime=runtime)(
        "crossref", "search", {"query": "x", "page_size": 1}
    )
    assert payload["message"] == {"items": []}
    assert attempts == 2


def test_declared_environment_credentials_are_routed_without_scanning_environment() -> None:
    calls = []
    environment = {
        "SEMANTIC_SCHOLAR_API_KEY": "s2-secret",
        "OPENALEX_API_KEY": "oa-secret",
        "UNPAYWALL_EMAIL": "oa@example.test",
        "UNDECLARED": "never",
    }

    def opener(request, timeout):
        calls.append(request)
        return Response(b"{}", {"Content-Type": "application/json"})

    transport = ControlledHTTPTransport("operator@example.test", opener=opener, environment=environment)

    transport("semantic_scholar", "search", {"query": "x"})
    transport("openalex", "search", {"search": "x"})
    transport("unpaywall", "resolve", {"doi": "10.1/example"})

    assert calls[0].get_header("X-api-key") == "s2-secret"
    assert "api_key=oa-secret" in calls[1].full_url
    assert "email=oa%40example.test" in calls[2].full_url
    assert "secret" not in (transport.last_request_url or "")
