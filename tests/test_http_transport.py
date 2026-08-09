from __future__ import annotations

from email.message import Message
from io import BytesIO
from urllib.error import HTTPError

import pytest

from paper_agent.http_transport import ControlledHTTPTransport
from paper_agent.provider_runtime import RetryableProviderError


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
