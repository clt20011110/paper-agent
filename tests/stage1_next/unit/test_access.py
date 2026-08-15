from email.message import Message
from io import BytesIO
from urllib.error import HTTPError, URLError

import pytest

from paper_agent_next import http as http_module
from paper_agent_next.access import resolve_access
from paper_agent_next.http import HttpClient
from paper_agent_next.models import AccessStatus


CONTACT = "collector@example.org"
DOI = "10.1000/example.1"
RAW_PMLR_PDF = (
    "https://raw.githubusercontent.com/mlresearch/v235/main/assets/"
    "lovelace24a/lovelace24a.pdf"
)


class FakeResponse:
    def __init__(
        self,
        body: bytes,
        *,
        content_type: str | None = None,
        final_url: str | None = None,
    ) -> None:
        self.body = body
        self.final_url = final_url
        self.closed = False
        self.read_limits: list[int] = []
        self.headers = Message()
        if content_type is not None:
            self.headers["Content-Type"] = content_type

    def read(self, max_bytes: int) -> bytes:
        self.read_limits.append(max_bytes)
        return self.body[:max_bytes]

    def close(self) -> None:
        self.closed = True


class SequenceOpener:
    def __init__(self, outcomes: list[FakeResponse | BaseException]) -> None:
        self.outcomes = outcomes
        self.calls: list[tuple[object, float]] = []

    def __call__(self, request, *, timeout: float):
        self.calls.append((request, timeout))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _client() -> HttpClient:
    return HttpClient(CONTACT, 3.5)


def _http_error(url: str, status: int) -> HTTPError:
    return HTTPError(url, status, "failure", Message(), BytesIO(b"error"))


def test_pmlr_raw_octet_stream_pdf_is_direct_pdf(monkeypatch) -> None:
    response = FakeResponse(b"%PDF-1.7\nbody", content_type="application/octet-stream")
    opener = SequenceOpener([response])
    monkeypatch.setattr(http_module, "urlopen", opener)

    decision = resolve_access((RAW_PMLR_PDF,), None, _client())

    assert decision.access_status is AccessStatus.DIRECT_PDF
    assert decision.pdf_url == RAW_PMLR_PDF
    assert decision.reason_code is None
    assert response.read_limits == [4096]
    assert response.closed is True


def test_application_pdf_parameters_and_magic_are_direct_pdf(monkeypatch) -> None:
    candidate = "https://example.test/paper.pdf"
    response = FakeResponse(b"%PDF-1.4\nbody", content_type=" application/pdf ; charset=binary ")
    opener = SequenceOpener([response])
    monkeypatch.setattr(http_module, "urlopen", opener)

    decision = resolve_access((candidate,), DOI, _client())

    assert decision.access_status is AccessStatus.DIRECT_PDF
    assert decision.pdf_url == candidate
    assert decision.reason_code is None
    assert response.closed is True


def test_verified_pdf_keeps_original_candidate_when_response_has_final_url(monkeypatch) -> None:
    candidate = "https://example.test/paper?download=1"
    response = FakeResponse(
        b"%PDF-1.7\nbody",
        content_type="application/pdf",
        final_url="https://signed.example.test/temporary.pdf?signature=secret",
    )
    opener = SequenceOpener([response])
    monkeypatch.setattr(http_module, "urlopen", opener)

    decision = resolve_access((candidate,), None, _client())

    assert decision.access_status is AccessStatus.DIRECT_PDF
    assert decision.pdf_url == candidate
    assert decision.pdf_url != response.final_url


@pytest.mark.parametrize(
    ("page", "body"),
    [
        ("login", b"<html><body>login required</body></html>"),
        ("captcha", b"<html><body>CAPTCHA challenge</body></html>"),
        ("paywall", b"<html><body>paywall</body></html>"),
    ],
)
def test_html_login_captcha_and_paywall_are_doi_only(monkeypatch, page: str, body: bytes) -> None:
    response = FakeResponse(body, content_type="text/html; charset=utf-8")
    opener = SequenceOpener([response])
    monkeypatch.setattr(http_module, "urlopen", opener)

    decision = resolve_access((f"https://example.test/{page}",), DOI, _client())

    assert decision.access_status is AccessStatus.DOI_ONLY
    assert decision.pdf_url is None
    assert decision.reason_code is None


@pytest.mark.parametrize(
    ("content_type", "body"),
    [
        ("application/pdf", b"not a PDF"),
        ("text/html", b"%PDF-1.7\nnot really a PDF response"),
    ],
)
def test_content_type_and_pdf_magic_are_both_required(
    monkeypatch, content_type: str, body: bytes
) -> None:
    response = FakeResponse(body, content_type=content_type)
    opener = SequenceOpener([response])
    monkeypatch.setattr(http_module, "urlopen", opener)

    decision = resolve_access(("https://example.test/paper",), DOI, _client())

    assert decision.access_status is AccessStatus.DOI_ONLY
    assert decision.pdf_url is None
    assert decision.reason_code is None


@pytest.mark.parametrize("status", [401, 403])
@pytest.mark.parametrize("doi", [DOI, None], ids=["doi", "no-doi"])
def test_http_auth_errors_are_candidate_failures_with_doi_fallback(
    monkeypatch, status: int, doi: str | None
) -> None:
    candidate = "https://example.test/paper.pdf"
    error = _http_error(candidate, status)
    opener = SequenceOpener([error])
    monkeypatch.setattr(http_module, "urlopen", opener)

    decision = resolve_access((candidate,), doi, _client())

    if doi is not None:
        assert decision.access_status is AccessStatus.DOI_ONLY
        assert decision.reason_code is None
    else:
        assert decision.access_status is None
        assert decision.reason_code == "no_verified_pdf_or_doi"
    assert decision.pdf_url is None
    assert error.fp is not None and error.fp.closed


def test_ordered_candidates_choose_first_verified_pdf(monkeypatch) -> None:
    first = "https://example.test/first.pdf"
    second = "https://example.test/second.pdf"
    first_response = FakeResponse(b"not a PDF", content_type="application/pdf")
    second_response = FakeResponse(b"%PDF-1.7\nbody", content_type="application/pdf")
    opener = SequenceOpener([first_response, second_response])
    monkeypatch.setattr(http_module, "urlopen", opener)

    decision = resolve_access((first, second), None, _client())

    assert decision.access_status is AccessStatus.DIRECT_PDF
    assert decision.pdf_url == second
    assert [request.full_url for request, _ in opener.calls] == [first, second]
    assert first_response.closed is True
    assert second_response.closed is True


@pytest.mark.parametrize(
    "candidate",
    [
        "https://example.test/paper.pdf?token=secret",
        "https://example.test/paper.pdf?X-AmZ-Signature=secret",
        "https://example.test/paper.pdf?x-goog-credential=secret",
        "https://user:password@example.test/paper.pdf",
    ],
)
def test_credential_or_signed_candidates_are_rejected_before_network(
    monkeypatch, candidate: str
) -> None:
    def opener(request, *, timeout):
        raise AssertionError("urlopen must not be called for an unsafe candidate")

    monkeypatch.setattr(http_module, "urlopen", opener)

    decision = resolve_access((candidate,), DOI, _client())

    assert decision.access_status is AccessStatus.DOI_ONLY
    assert decision.pdf_url is None
    assert decision.reason_code is None


def test_stable_download_query_is_not_rejected(monkeypatch) -> None:
    candidate = "https://example.test/paper.pdf?download=1"
    response = FakeResponse(b"%PDF-1.7\nbody", content_type="application/pdf")
    opener = SequenceOpener([response])
    monkeypatch.setattr(http_module, "urlopen", opener)

    decision = resolve_access((candidate,), None, _client())

    assert decision.access_status is AccessStatus.DIRECT_PDF
    assert decision.pdf_url == candidate
    assert len(opener.calls) == 1


@pytest.mark.parametrize(
    "candidate",
    [
        "relative.pdf",
        "ftp://example.test/paper.pdf",
        "https://example.test/paper with-space.pdf",
        "https://example.test:bad-port/paper.pdf",
        "https:///paper.pdf",
    ],
)
def test_non_http_or_invalid_candidates_are_rejected_before_network(
    monkeypatch, candidate: str
) -> None:
    def opener(request, *, timeout):
        raise AssertionError("urlopen must not be called for an invalid candidate")

    monkeypatch.setattr(http_module, "urlopen", opener)

    decision = resolve_access((candidate,), None, _client())

    assert decision.access_status is None
    assert decision.pdf_url is None
    assert decision.reason_code == "no_verified_pdf_or_doi"


def test_network_failure_is_skipped_and_next_candidate_is_tried(monkeypatch) -> None:
    first = "https://example.test/first.pdf"
    second = "https://example.test/second.pdf"
    second_response = FakeResponse(b"%PDF-1.7\nbody", content_type="application/pdf")
    opener = SequenceOpener([URLError("offline"), second_response])
    monkeypatch.setattr(http_module, "urlopen", opener)

    decision = resolve_access((first, second), None, _client())

    assert decision.access_status is AccessStatus.DIRECT_PDF
    assert decision.pdf_url == second
    assert [request.full_url for request, _ in opener.calls] == [first, second]


def test_no_candidates_with_doi_makes_no_network_call(monkeypatch) -> None:
    def opener(request, *, timeout):
        raise AssertionError("urlopen must not be called without candidates")

    monkeypatch.setattr(http_module, "urlopen", opener)

    decision = resolve_access((), DOI, _client())

    assert decision.access_status is AccessStatus.DOI_ONLY
    assert decision.pdf_url is None
    assert decision.reason_code is None


def test_no_verified_pdf_without_doi_has_exact_incomplete_decision(monkeypatch) -> None:
    response = FakeResponse(b"not a PDF", content_type="text/plain")
    opener = SequenceOpener([response])
    monkeypatch.setattr(http_module, "urlopen", opener)

    decision = resolve_access(("https://example.test/paper",), None, _client())

    assert decision.access_status is None
    assert decision.pdf_url is None
    assert decision.reason_code == "no_verified_pdf_or_doi"


def test_all_pdf_probes_read_only_4096_bytes(monkeypatch) -> None:
    first = FakeResponse(b"not a PDF" + b"x" * 10000, content_type="application/pdf")
    second = FakeResponse(b"%PDF-1.7" + b"x" * 10000, content_type="application/pdf")
    opener = SequenceOpener([first, second])
    monkeypatch.setattr(http_module, "urlopen", opener)

    decision = resolve_access(
        ("https://example.test/first", "https://example.test/second"), None, _client()
    )

    assert decision.access_status is AccessStatus.DIRECT_PDF
    assert first.read_limits == [4096]
    assert second.read_limits == [4096]
