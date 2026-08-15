from email.message import Message
from http.client import IncompleteRead
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest

from paper_agent import http as http_module
from paper_agent.adapters.pmlr import PmlrAdapter
from paper_agent.catalog import load_venue_spec
from paper_agent.errors import CollectionError, InputError
from paper_agent.http import HttpClient


FIXTURES = Path(__file__).parents[1] / "fixtures" / "pmlr"
VOLUME_URL = "https://proceedings.mlr.press/v235/"
ADA_URL = "https://proceedings.mlr.press/v235/lovelace24a.html"
TURING_URL = "https://proceedings.mlr.press/v235/turing24a.html"
RAW_ADA_PDF = "https://raw.githubusercontent.com/mlresearch/v235/main/assets/lovelace24a/lovelace24a.pdf"
CONTACT = "collector@example.org"


class FakeResponse:
    def __init__(
        self,
        body: bytes,
        *,
        content_type: str | None = None,
        read_error: BaseException | None = None,
    ) -> None:
        self.body = body
        self.read_error = read_error
        self.closed = False
        self.read_limits: list[int | None] = []
        self.headers = Message()
        if content_type is not None:
            self.headers["Content-Type"] = content_type

    def read(self, max_bytes: int | None = None) -> bytes:
        self.read_limits.append(max_bytes)
        if self.read_error is not None:
            raise self.read_error
        return self.body if max_bytes is None else self.body[:max_bytes]

    def close(self) -> None:
        self.closed = True


def test_get_text_passes_get_timeout_and_user_agent_and_closes_response(monkeypatch) -> None:
    response = FakeResponse(b"PMLR", content_type="text/html; charset=utf-8")
    calls: list[tuple[object, object]] = []

    def opener(request, *, timeout):
        calls.append((request, timeout))
        return response

    monkeypatch.setattr(http_module, "urlopen", opener)

    timeout = 4.25
    assert HttpClient(CONTACT, timeout).get_text("https://example.test/page") == "PMLR"

    assert len(calls) == 1
    request, passed_timeout = calls[0]
    assert request.full_url == "https://example.test/page"
    assert request.get_method() == "GET"
    assert passed_timeout == timeout
    assert CONTACT in request.get_header("User-agent")
    assert "paper-agent" in request.get_header("User-agent")
    assert response.closed is True


def test_get_prefix_passes_get_timeout_and_user_agent_reads_bound_and_closes_response(
    monkeypatch,
) -> None:
    body = b"%PDF-1.7" + b"x" * 4096
    response = FakeResponse(body, content_type="application/pdf; version=1.7")
    calls: list[tuple[object, object]] = []

    def opener(request, *, timeout):
        calls.append((request, timeout))
        return response

    monkeypatch.setattr(http_module, "urlopen", opener)

    timeout = 4.25
    result = HttpClient(CONTACT, timeout).get_prefix(
        "https://example.test/paper.pdf", 4096
    )

    assert result.content_type == "application/pdf; version=1.7"
    assert result.body == body[:4096]
    assert len(calls) == 1
    request, passed_timeout = calls[0]
    assert request.full_url == "https://example.test/paper.pdf"
    assert request.get_method() == "GET"
    assert passed_timeout == timeout
    assert CONTACT in request.get_header("User-agent")
    assert "paper-agent" in request.get_header("User-agent")
    assert response.read_limits == [4096]
    assert response.closed is True


def test_get_prefix_preserves_missing_content_type_as_none(monkeypatch) -> None:
    response = FakeResponse(b"%PDF-1.7")
    monkeypatch.setattr(http_module, "urlopen", lambda request, *, timeout: response)

    result = HttpClient(CONTACT, 1.0).get_prefix("https://example.test/paper.pdf", 4096)

    assert result.content_type is None
    assert result.body == b"%PDF-1.7"
    assert response.closed is True


@pytest.mark.parametrize(
    "max_bytes",
    [0, -1, True, 1.5, "4096"],
    ids=["zero", "negative", "bool", "float", "string"],
)
def test_get_prefix_rejects_invalid_max_bytes_before_network(monkeypatch, max_bytes) -> None:
    def opener(request, *, timeout):
        raise AssertionError("urlopen must not be called")

    monkeypatch.setattr(http_module, "urlopen", opener)

    with pytest.raises(InputError):
        HttpClient(CONTACT, 1.0).get_prefix("https://example.test/paper.pdf", max_bytes)


def test_get_prefix_http_error_becomes_collection_error_and_closes_error(monkeypatch) -> None:
    error = HTTPError(
        "https://example.test/paper.pdf",
        403,
        "forbidden",
        Message(),
        BytesIO(b"error"),
    )
    calls = 0

    def opener(request, *, timeout):
        nonlocal calls
        calls += 1
        raise error

    monkeypatch.setattr(http_module, "urlopen", opener)

    with pytest.raises(CollectionError, match="HTTP 403") as caught:
        HttpClient(CONTACT, 1.0).get_prefix("https://example.test/paper.pdf", 4096)

    assert caught.value.__cause__ is error
    assert calls == 1
    assert error.fp is not None and error.fp.closed


@pytest.mark.parametrize(
    "failure",
    [IncompleteRead(b"partial"), ConnectionResetError("connection reset")],
    ids=["incomplete-read", "connection-reset"],
)
def test_get_prefix_read_failure_becomes_collection_error_and_closes_response(
    monkeypatch, failure: BaseException
) -> None:
    response = FakeResponse(b"ignored", read_error=failure)
    monkeypatch.setattr(http_module, "urlopen", lambda request, *, timeout: response)

    with pytest.raises(CollectionError) as caught:
        HttpClient(CONTACT, 1.0).get_prefix("https://example.test/paper.pdf", 4096)

    assert caught.value.__cause__ is failure
    assert response.read_limits == [4096]
    assert response.closed is True


@pytest.mark.parametrize(
    ("content_type", "body", "expected"),
    [
        ("text/html; charset=utf-8", "摘要".encode("utf-8"), "摘要"),
        ("text/html", "无声明".encode("utf-8"), "无声明"),
    ],
)
def test_get_text_decodes_declared_or_default_utf8(
    monkeypatch, content_type: str, body: bytes, expected: str
) -> None:
    response = FakeResponse(body, content_type=content_type)
    monkeypatch.setattr(http_module, "urlopen", lambda request, *, timeout: response)

    assert HttpClient(CONTACT, 1.0).get_text("https://example.test/page") == expected
    assert response.closed is True


def test_http_error_becomes_collection_error_with_cause_and_one_call(monkeypatch) -> None:
    error = HTTPError(
        "https://example.test/page",
        503,
        "unavailable",
        Message(),
        BytesIO(b"error"),
    )
    calls = 0

    def opener(request, *, timeout):
        nonlocal calls
        calls += 1
        raise error

    monkeypatch.setattr(http_module, "urlopen", opener)

    with pytest.raises(CollectionError, match="HTTP 503") as caught:
        HttpClient(CONTACT, 1.0).get_text("https://example.test/page")

    assert caught.value.__cause__ is error
    assert calls == 1
    assert error.fp is not None and error.fp.closed


@pytest.mark.parametrize(
    "failure",
    [URLError("offline"), TimeoutError("timed out")],
    ids=["urlerror", "timeout"],
)
def test_open_failure_becomes_collection_error_with_cause_and_one_call(
    monkeypatch, failure: BaseException
) -> None:
    calls = 0

    def opener(request, *, timeout):
        nonlocal calls
        calls += 1
        raise failure

    monkeypatch.setattr(http_module, "urlopen", opener)

    with pytest.raises(CollectionError) as caught:
        HttpClient(CONTACT, 1.0).get_text("https://example.test/page")

    assert caught.value.__cause__ is failure
    assert calls == 1


@pytest.mark.parametrize(
    "failure",
    [IncompleteRead(b"partial"), ConnectionResetError("connection reset")],
    ids=["incomplete-read", "connection-reset"],
)
def test_read_failure_becomes_collection_error_with_cause_and_closes_response(
    monkeypatch, failure: BaseException
) -> None:
    response = FakeResponse(b"ignored", read_error=failure)
    monkeypatch.setattr(http_module, "urlopen", lambda request, *, timeout: response)

    with pytest.raises(CollectionError) as caught:
        HttpClient(CONTACT, 1.0).get_text("https://example.test/page")

    assert caught.value.__cause__ is failure
    assert response.closed is True


def test_invalid_utf8_becomes_collection_error_with_decode_cause_and_closes_response(
    monkeypatch,
) -> None:
    response = FakeResponse(b"\xff", content_type="text/html; charset=utf-8")
    monkeypatch.setattr(http_module, "urlopen", lambda request, *, timeout: response)

    with pytest.raises(CollectionError) as caught:
        HttpClient(CONTACT, 1.0).get_text("https://example.test/page")

    assert isinstance(caught.value.__cause__, UnicodeDecodeError)
    assert response.closed is True


@pytest.mark.parametrize("contact", ["", "bad\ncontact", "bad\r\ncontact"])
def test_invalid_contact_fails_before_network(monkeypatch, contact: str) -> None:
    def opener(request, *, timeout):
        raise AssertionError("urlopen must not be called")

    monkeypatch.setattr(http_module, "urlopen", opener)

    with pytest.raises(InputError):
        HttpClient(contact, 1.0)


@pytest.mark.parametrize(
    "timeout",
    [0, -1, float("nan"), float("inf"), float("-inf"), True],
    ids=["zero", "negative", "nan", "positive-infinity", "negative-infinity", "bool"],
)
def test_invalid_timeout_fails_before_network(monkeypatch, timeout: object) -> None:
    def opener(request, *, timeout):
        raise AssertionError("urlopen must not be called")

    monkeypatch.setattr(http_module, "urlopen", opener)

    with pytest.raises(InputError):
        HttpClient(CONTACT, timeout)  # type: ignore[arg-type]


class FixtureOpener:
    def __init__(self, responses: dict[str, bytes]) -> None:
        self.responses = responses
        self.calls: list[tuple[object, float]] = []

    def __call__(self, request, *, timeout: float) -> FakeResponse:
        self.calls.append((request, timeout))
        if request.full_url not in self.responses:
            raise AssertionError(f"unexpected offline URL: {request.full_url}")
        return FakeResponse(self.responses[request.full_url], content_type="text/html; charset=utf-8")


def test_pmlr_adapter_collects_through_the_real_http_client_offline(monkeypatch) -> None:
    responses = {
        VOLUME_URL: (FIXTURES / "volume-v235.html").read_bytes(),
        ADA_URL: (FIXTURES / "lovelace24a.html").read_bytes(),
        TURING_URL: (FIXTURES / "turing24a.html").read_bytes(),
    }
    opener = FixtureOpener(responses)
    monkeypatch.setattr(http_module, "urlopen", opener)

    timeout = 6.5
    result = PmlrAdapter().collect(
        load_venue_spec("icml"), 2024, HttpClient(CONTACT, timeout)
    )

    assert [paper.source_id for paper in result.papers] == [
        "v235/lovelace24a",
        "v235/turing24a",
    ]
    ada, turing = result.papers
    assert ada.pdf_candidates == (RAW_ADA_PDF,)
    assert ada.abstract == "Reliable small models & graphs for reproducible experiments."
    assert turing.abstract == "Parallel inference reduces latency. It also preserves & checks accuracy."

    requests = [request for request, _ in opener.calls]
    assert [request.full_url for request in requests] == [VOLUME_URL, ADA_URL, TURING_URL]
    assert all(not request.full_url.casefold().endswith(".pdf") for request in requests)
    assert [passed_timeout for _, passed_timeout in opener.calls] == [timeout] * 3
    assert all(CONTACT in request.get_header("User-agent") for request in requests)
