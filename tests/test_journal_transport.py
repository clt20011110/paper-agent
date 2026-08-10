from __future__ import annotations

from email.message import Message
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from paper_agent.journal_transport import (
    JournalHTTPTransport,
    execute_journal_operation,
    journal_provider_names,
)
from paper_agent.provider_runtime import ProviderPolicyDenied, ProviderRuntime, ProviderRuntimePolicy


FIXTURES = Path(__file__).parent / "fixtures" / "providers"


class Response:
    def __init__(self, body: bytes, content_type: str) -> None:
        self._body = body
        self.headers = Message()
        self.headers["Content-Type"] = content_type

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "Response":
        return self

    def __exit__(self, *_: object) -> None:
        return None


def _runtime() -> ProviderRuntime:
    return ProviderRuntime(
        {
            provider: ProviderRuntimePolicy(
                provider,
                credentials_required=provider != "aaas_science",
                credential_environment_variables=(() if provider == "aaas_science" else ({
                    "ieee_xplore": "IEEE_XPLORE_API_KEY",
                    "springer_nature": "SPRINGER_NATURE_API_KEY",
                    "cell_press": "ELSEVIER_API_KEY",
                }[provider],)),
                retry_attempts=1,
                jitter_seconds=0,
            )
            for provider in ("ieee_xplore", "springer_nature", "cell_press", "aaas_science")
        }
    )


def _transport(body: str, content_type: str = "application/json") -> tuple[JournalHTTPTransport, list]:
    calls = []

    def opener(request, timeout):
        calls.append((request, timeout))
        return Response((FIXTURES / body).read_bytes(), content_type)

    return JournalHTTPTransport(
        _runtime(), opener=opener,
        environment={"IEEE_XPLORE_API_KEY": "ieee-key", "SPRINGER_NATURE_API_KEY": "springer-key", "ELSEVIER_API_KEY": "elsevier-key"},
    ), calls


def test_ieee_tcad_request_uses_fixed_publication_issn_window_issue_and_cursor() -> None:
    transport, calls = _transport("ieee-xplore-native.json")
    payload = transport("ieee_xplore", "discover", {
        "publication_number": 43, "issns": ["0278-0070"], "date_from": "2024-01-01", "date_to": "2024-12-31",
        "volume": "43", "issue": "4", "cursor": "100", "page_size": 100,
    })
    query = parse_qs(urlsplit(calls[0][0].full_url).query)
    assert calls[0][1] == 15
    assert urlsplit(calls[0][0].full_url).path == "/api/v1/search/articles"
    assert query["publication_number"] == ["43"]
    assert query["issn"] == ["0278-0070"]
    assert query["start_year"] == ["2024"] and query["end_year"] == ["2024"]
    assert query["volume"] == ["43"] and query["is_number"] == ["4"] and query["start_record"] == ["100"]
    assert payload["next_cursor"] == "101"
    assert payload["entries"][0]["stable_id"] == "10000001"
    assert payload["entries"][0]["authors"] == ["Ada Lovelace"]
    assert query["apikey"] == ["ieee-key"]
    assert "apikey=" not in (transport.last_request_url or "")


def test_springer_request_keeps_descriptor_type_date_volume_issue_and_page() -> None:
    transport, calls = _transport("springer-native.json")
    payload = transport("springer_nature", "discover", {
        "journal_slug": "natmachintell", "issns": ["2522-5839"], "article_types": ["Article", "Review"],
        "date_from": "2024-01-01", "date_to": "2024-12-31", "volume": "6", "issue": "2", "cursor": "51", "page_size": 10,
    })
    query = parse_qs(urlsplit(calls[0][0].full_url).query)
    assert urlsplit(calls[0][0].full_url).path == "/metadata/v1/articles"
    assert query["s"] == ["51"] and query["p"] == ["10"]
    assert "issn:2522-5839" in query["q"][0]
    assert "articletype:Article OR articletype:Review" in query["q"][0]
    assert "onlinedatefrom:2024-01-01" in query["q"][0]
    assert "onlinedateto:2024-12-31" in query["q"][0]
    assert "volume:6" in query["q"][0] and "issue:2" in query["q"][0]
    assert payload["next_cursor"] == "52"
    assert payload["entries"][0]["doi"] == "10.1038/s42256-024-00001-1"
    assert query["api_key"] == ["springer-key"]
    assert "api_key=" not in (transport.last_request_url or "")


def test_cell_request_uses_official_metadata_api_header_and_offset() -> None:
    transport, calls = _transport("cell-elsevier-native.json")
    payload = transport("cell_press", "discover", {
        "issn": "0092-8674", "year": 2024, "volume": "187", "issue": "18", "cursor": "50", "page_size": 25,
    })
    request = calls[0][0]
    query = parse_qs(urlsplit(request.full_url).query)
    assert urlsplit(request.full_url).path == "/content/metadata/article"
    assert request.get_header("X-els-apikey") == "elsevier-key"
    assert query["query"] == ["issn(0092-8674) AND volume(187) AND issue(18)"]
    assert query["date"] == ["2024"] and query["start"] == ["50"] and query["count"] == ["25"]
    assert payload["next_cursor"] == "51"
    assert payload["entries"][0]["abstract"] == "Elsevier metadata abstract."


def test_science_toc_is_static_publisher_html_with_no_pdf_follow() -> None:
    transport, calls = _transport("science-toc.html", "text/html")
    payload = transport("aaas_science", "discover", {"issns": ["0036-8075", "1095-9203"], "volume": "384", "issue": "6699"})
    assert calls[0][0].full_url == "https://www.science.org/toc/science/384/6699"
    assert payload["next_cursor"] is None
    assert payload["entries"] == [{
        "stable_id": "10.1126/science.abc0009", "title": "Science native metadata", "doi": "10.1126/science.abc0009",
        "publication_date": "2024-10-11", "venue": "Science", "landing_url": "https://www.science.org/doi/10.1126/science.abc0009",
        "abstract": "AAAS Science table of contents abstract.",
    }]


def test_science_etoc_rss_is_official_current_issue_fallback() -> None:
    transport, calls = _transport("science-etoc.xml", "application/rss+xml")
    payload = transport("aaas_science", "discover", {"issns": ["0036-8075", "1095-9203"]})
    assert "action/showFeed" in calls[0][0].full_url
    assert payload["entries"][0]["publication_date"] == "2024-10-11"
    assert payload["entries"][0]["doi"] == "10.1126/science.abc0010"


def test_science_year_archive_keeps_date_window_and_cursor_in_publisher_route() -> None:
    transport, calls = _transport("science-toc.html", "text/html")
    transport("aaas_science", "discover", {"issns": ["0036-8075", "1095-9203"], "date_from": "2024-01-01", "date_to": "2024-12-31", "cursor": "2"})
    assert calls[0][0].full_url == "https://www.science.org/toc/science/2024?page=2"


def test_declared_credentials_are_enforced_by_provider_runtime() -> None:
    transport = JournalHTTPTransport(_runtime(), opener=lambda *_: pytest.fail("must not open"), environment={})
    with pytest.raises(ProviderPolicyDenied, match="ieee_xplore"):
        transport("ieee_xplore", "discover", {"publication_number": 43, "issn": "0278-0070"})


def test_journal_delegate_uses_caller_owned_fetch_without_embedding_credentials() -> None:
    calls = []

    def fetch(url: str, api_version: str):
        calls.append((url, api_version))
        return type(
            "Fetched",
            (),
            {
                "body": (FIXTURES / "ieee-xplore-native.json").read_bytes(),
                "content_type": "application/json",
            },
        )()

    result = execute_journal_operation(
        "ieee_xplore",
        "discover",
        {"publication_number": 43, "issns": ["0278-0070"], "year": 2024},
        fetch,
    )

    assert journal_provider_names() == ("aaas_science", "cell_press", "ieee_xplore", "springer_nature")
    assert calls[0][1] == "metadata-api-v1"
    assert "apikey=" not in calls[0][0]
    assert result.payload["entries"][0]["stable_id"] == "10000001"
    assert result.bodies == ((FIXTURES / "ieee-xplore-native.json").read_bytes(),)
