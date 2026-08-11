from __future__ import annotations

from email.message import Message
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlsplit

import pytest

from paper_agent.http_transport import ControlledHTTPTransport
from paper_agent.domain import QuerySpec, SourceEntry
from paper_agent.provider_runtime import ProviderRequestError, ProviderRuntime, ProviderRuntimePolicy, RetryableProviderError
from paper_agent.providers.api import IdentityCandidate
from paper_agent.providers.builtin import create_builtin


FIXTURES = Path(__file__).parent / "fixtures" / "providers"


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


def test_request_audit_collects_only_allowlisted_rate_limit_headers() -> None:
    def opener(request, timeout):
        return Response(
            b'{"message":{"items":[]}}',
            {
                "Content-Type": "application/json",
                "rAtElImIt-ReMaInInG": "48",
                "X-Rate-Limit-Interval": "1s",
                "X-CREDIT-USED": "2.5",
                "Authorization": "Bearer secret",
                "Set-Cookie": "session=secret",
                "X-Request-Id": "internal-id",
            },
        )

    payload = ControlledHTTPTransport(
        "https://example.test/contact", opener=opener
    )("crossref", "search", {"query": "x", "page_size": 1})

    assert payload["_request_audit"][0]["rate_limit"] == {
        "ratelimit-remaining": "48",
        "x-rate-limit-interval": "1s",
        "x-credit-used": "2.5",
    }
    assert "secret" not in str(payload["_request_audit"])


def test_conditional_request_reuses_cached_body_on_not_modified() -> None:
    count = 0

    def opener(request, timeout):
        nonlocal count
        count += 1
        if count == 1:
            return Response(
                b'{"message":{"items":[]}}',
                {
                    "Content-Type": "application/json",
                    "ETag": "one",
                    "RateLimit-Remaining": "7",
                },
            )
        assert request.get_header("If-none-match") == "one"
        raise HTTPError(request.full_url, 304, "not modified", Message(), BytesIO())

    transport = ControlledHTTPTransport("https://example.test/contact", opener=opener)
    first = transport("crossref", "search", {"query": "x", "page_size": 1})
    second = transport("crossref", "search", {"query": "x", "page_size": 1})
    assert first["raw_response_artifact_hash"] == second["raw_response_artifact_hash"]
    assert first["_request_audit"][0]["rate_limit"] == {"ratelimit-remaining": "7"}
    assert second["_request_audit"][0]["rate_limit"] == {"ratelimit-remaining": "7"}


def test_retry_after_is_exposed_for_rate_limits() -> None:
    def opener(request, timeout):
        headers = Message()
        headers["Retry-After"] = "3"
        raise HTTPError(request.full_url, 429, "limited", headers, BytesIO())

    transport = ControlledHTTPTransport("https://example.test/contact", opener=opener)
    with pytest.raises(RetryableProviderError) as error:
        transport("crossref", "search", {"query": "x", "page_size": 1})
    assert error.value.retry_after == 3
    assert transport.request_audit[0]["retry_after_seconds"] == 3


def test_cloudflare_403_is_classified_as_external_challenge() -> None:
    def opener(request, timeout):
        raise HTTPError(
            request.full_url,
            403,
            "forbidden",
            Message(),
            BytesIO(b"<title>Just a moment...</title> Cloudflare challenge"),
        )

    transport = ControlledHTTPTransport("https://example.test/contact", opener=opener)
    with pytest.raises(ProviderRequestError, match="challenge verification required"):
        transport("crossref", "search", {"query": "x", "page_size": 1})


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


def test_europe_pmc_result_type_is_forwarded_to_the_search_url() -> None:
    urls: list[str] = []

    def opener(request, timeout):
        urls.append(request.full_url)
        return Response(b"{}", {"Content-Type": "application/json"})

    ControlledHTTPTransport("mailto:operator@example.test", opener=opener)(
        "europe_pmc",
        "search",
        {"doi": "10.3758/s13421-020-01060-2", "resultType": "core"},
    )

    query = parse_qs(urlsplit(urls[0]).query)
    assert query["query"] == ["DOI:10.3758/s13421-020-01060-2"]
    assert query["resultType"] == ["core"]


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


def _pubmed_opener(urls: list[str]):
    summary = (FIXTURES / "pubmed-esummary.json").read_bytes()
    abstract = (FIXTURES / "pubmed-efetch.xml").read_bytes()

    def opener(request, timeout):
        urls.append(request.full_url)
        if "esearch.fcgi" in request.full_url:
            return Response(
                b'{"esearchresult":{"count":"2","idlist":["39900001"]}}',
                {"Content-Type": "application/json"},
            )
        if "esummary.fcgi" in request.full_url:
            return Response(summary, {"Content-Type": "application/json"})
        assert "efetch.fcgi" in request.full_url
        return Response(abstract, {"Content-Type": "application/xml"})

    return opener


def test_pubmed_search_merges_efetch_abstract_without_fetching_full_text() -> None:
    urls: list[str] = []
    transport = ControlledHTTPTransport(
        "mailto:operator@example.test",
        opener=_pubmed_opener(urls),
    )

    batch = create_builtin("pubmed", transport).search(
        QuerySpec(
            1,
            "pubmed-role-contract",
            "metadata",
            native_parameters={"db": "pubmed", "term": "metadata", "retmax": 1},
            native_query_hash="pubmed-native-query",
        )
    )

    assert "esearch.fcgi" in urls[0]
    assert "esummary.fcgi" in urls[1]
    assert "efetch.fcgi" in urls[2]
    assert "retmode=xml" in urls[2]
    assert "rettype=" not in urls[2]
    assert len(urls) == 3
    assert all("eutils.ncbi.nlm.nih.gov/entrez/eutils/" in url for url in urls)
    assert batch.next_cursor == "1"
    assert batch.entries[0].abstract == (
        "BACKGROUND: Graph retrieval needs reliable metadata. "
        "RESULTS: The method preserves native PubMed abstracts."
    )
    assert all("pmc" not in url.casefold() and ".pdf" not in url.casefold() for url in urls)


def test_pubmed_enrich_merges_the_same_efetch_metadata() -> None:
    urls: list[str] = []
    transport = ControlledHTTPTransport(
        "mailto:operator@example.test",
        opener=_pubmed_opener(urls),
    )

    result = create_builtin("pubmed", transport).enrich(
        SourceEntry("seed", "39900001", "Seed PubMed record")
    )

    assert result.entry.external_id == "39900001"
    assert result.entry.abstract == (
        "BACKGROUND: Graph retrieval needs reliable metadata. "
        "RESULTS: The method preserves native PubMed abstracts."
    )
    assert [url.split("?", 1)[0].rsplit("/", 1)[-1] for url in urls] == [
        "esearch.fcgi",
        "esummary.fcgi",
        "efetch.fcgi",
    ]
    assert all("eutils.ncbi.nlm.nih.gov/entrez/eutils/" in url for url in urls)
    assert all("pmc" not in url.casefold() and ".pdf" not in url.casefold() for url in urls)


def test_pubmed_verify_reuses_esearch_and_esummary_without_efetch() -> None:
    urls: list[str] = []
    transport = ControlledHTTPTransport(
        "mailto:operator@example.test",
        opener=_pubmed_opener(urls),
    )

    result = create_builtin("pubmed", transport).verify(
        IdentityCandidate("Native PubMed ESummary Fixture", doi="10.1000/pubmed.native"),
        (),
    )

    assert result.evidence == ("39900001",)
    assert [url.split("?", 1)[0].rsplit("/", 1)[-1] for url in urls] == [
        "esearch.fcgi",
        "esummary.fcgi",
    ]


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
