"""Offline tests for the small POST JSON HTTP boundary."""

from email.message import Message
from http.client import IncompleteRead
from io import BytesIO
from urllib.error import HTTPError

import pytest

from paper_agent import http as http_module
from paper_agent.errors import EnrichmentError
from paper_agent.http import HttpClient


class _Response:
    def __init__(self, body: bytes, *, charset: str | None = "utf-8") -> None:
        self.body = body
        self.headers = Message()
        if charset is not None:
            self.headers["Content-Type"] = f"application/json; charset={charset}"
        self.closed = False
        self.read_count = 0

    def read(self) -> bytes:
        self.read_count += 1
        return self.body

    def close(self) -> None:
        self.closed = True


def test_post_json_is_compact_utf8_json_and_reuses_timeout_and_user_agent(monkeypatch) -> None:
    response = _Response('{"摘要":"已补全"}'.encode("utf-8"))
    calls: list[tuple[object, float]] = []

    def opener(request, *, timeout: float):
        calls.append((request, timeout))
        return response

    monkeypatch.setattr(http_module, "urlopen", opener)
    result = HttpClient("researcher@example.org", 7.5).post_json(
        "https://api.example.test/batch",
        {"ids": ["DOI:10.1234/example"], "摘要": "已补全"},
    )

    request, timeout = calls[0]
    assert result == {"摘要": "已补全"}
    assert request.get_method() == "POST"
    assert request.data == '{"ids":["DOI:10.1234/example"],"摘要":"已补全"}'.encode("utf-8")
    assert request.get_header("Content-type") == "application/json"
    assert request.get_header("Accept") == "application/json"
    assert "researcher@example.org" in request.get_header("User-agent")
    assert timeout == 7.5
    assert response.closed is True


@pytest.mark.parametrize(
    "failure",
    [
        b"not json",
        b"\xff",
    ],
    ids=["bad-json", "bad-encoding"],
)
def test_post_json_bad_response_is_typed_and_closes(monkeypatch, failure: bytes) -> None:
    response = _Response(failure)
    monkeypatch.setattr(http_module, "urlopen", lambda request, *, timeout: response)

    with pytest.raises(EnrichmentError):
        HttpClient("researcher@example.org", 1.0).post_json("https://example.test", {})
    assert response.closed is True


def test_post_json_http_read_failures_are_typed(monkeypatch) -> None:
    error = HTTPError(
        "https://example.test",
        429,
        "too many requests",
        Message(),
        BytesIO(b"secret"),
    )
    monkeypatch.setattr(http_module, "urlopen", lambda request, *, timeout: (_ for _ in ()).throw(error))
    with pytest.raises(EnrichmentError) as caught:
        HttpClient("researcher@example.org", 1.0).post_json("https://example.test", {})
    assert caught.value.__cause__ is error
    assert caught.value.status_code == 429
    assert error.fp is not None and error.fp.closed

    class _ReadFailure(_Response):
        def read(self) -> bytes:
            raise IncompleteRead(b"partial")

    response = _ReadFailure(b"")
    monkeypatch.setattr(http_module, "urlopen", lambda request, *, timeout: response)
    with pytest.raises(EnrichmentError):
        HttpClient("researcher@example.org", 1.0).post_json("https://example.test", {})
    assert response.closed is True


def test_get_json_is_json_get_with_shared_timeout_user_agent_and_utf8_charset(
    monkeypatch,
) -> None:
    response = _Response('{"摘要":"已补全"}'.encode("utf-8"), charset="utf-8")
    calls: list[tuple[object, float]] = []

    def opener(request, *, timeout: float):
        calls.append((request, timeout))
        return response

    monkeypatch.setattr(http_module, "urlopen", opener)
    result = HttpClient("researcher@example.org", 7.5).get_json(
        "https://api.example.test/works?select=x"
    )

    request, timeout = calls[0]
    assert result == {"摘要": "已补全"}
    assert request.get_method() == "GET"
    assert request.data is None
    assert request.get_header("Accept") == "application/json"
    assert "researcher@example.org" in request.get_header("User-agent")
    assert timeout == 7.5
    assert response.closed is True


@pytest.mark.parametrize(
    "body",
    [b"not json", b"\xff"],
    ids=["bad-json", "bad-encoding"],
)
def test_get_json_bad_response_is_typed_and_closes(monkeypatch, body: bytes) -> None:
    response = _Response(body)
    monkeypatch.setattr(http_module, "urlopen", lambda request, *, timeout: response)

    with pytest.raises(EnrichmentError):
        HttpClient("researcher@example.org", 1.0).get_json("https://example.test")
    assert response.closed is True


def test_get_json_http_and_read_failures_are_typed(monkeypatch) -> None:
    error = HTTPError(
        "https://example.test",
        503,
        "unavailable",
        Message(),
        BytesIO(b"secret"),
    )
    monkeypatch.setattr(
        http_module,
        "urlopen",
        lambda request, *, timeout: (_ for _ in ()).throw(error),
    )
    with pytest.raises(EnrichmentError) as caught:
        HttpClient("researcher@example.org", 1.0).get_json("https://example.test")
    assert caught.value.__cause__ is error
    assert error.fp is not None and error.fp.closed

    class _ReadFailure(_Response):
        def read(self) -> bytes:
            raise IncompleteRead(b"partial")

    response = _ReadFailure(b"")
    monkeypatch.setattr(http_module, "urlopen", lambda request, *, timeout: response)
    with pytest.raises(EnrichmentError):
        HttpClient("researcher@example.org", 1.0).get_json("https://example.test")
    assert response.closed is True
