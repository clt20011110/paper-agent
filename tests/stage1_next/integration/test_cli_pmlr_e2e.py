import json
from email.message import Message
from pathlib import Path
from urllib.error import URLError

import pytest

from paper_agent_next import cli
from paper_agent_next import http as http_module
from paper_agent_next import output as output_module


FIXTURES = Path(__file__).parents[1] / "fixtures" / "pmlr"
VOLUME_URL = "https://proceedings.mlr.press/v235/"
ADA_URL = "https://proceedings.mlr.press/v235/lovelace24a.html"
TURING_URL = "https://proceedings.mlr.press/v235/turing24a.html"
RAW_ADA_PDF = "https://raw.githubusercontent.com/mlresearch/v235/main/assets/lovelace24a/lovelace24a.pdf"
SITE_ADA_PDF = "https://proceedings.mlr.press/v235/lovelace24a.pdf"
CONTACT = "integration@example.org"


class _FakeResponse:
    def __init__(self, body: bytes, content_type: str) -> None:
        self._body = body
        self.closed = False
        self.read_limits: list[int | None] = []
        self.headers = Message()
        self.headers["Content-Type"] = content_type

    def read(self, max_bytes: int | None = None) -> bytes:
        self.read_limits.append(max_bytes)
        return self._body if max_bytes is None else self._body[:max_bytes]

    def close(self) -> None:
        self.closed = True


class _FixtureOpener:
    def __init__(self, responses: dict[str, tuple[bytes, str]]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, float, object]] = []
        self.response_by_url: dict[str, _FakeResponse] = {}

    def __call__(self, request, *, timeout: float) -> _FakeResponse:
        url = request.full_url
        if url not in self.responses:
            raise AssertionError(f"unexpected offline URL: {url}")
        body, content_type = self.responses[url]
        response = _FakeResponse(body, content_type)
        self.calls.append((url, timeout, request))
        self.response_by_url[url] = response
        return response


def _fixture_response(name: str) -> tuple[bytes, str]:
    return (FIXTURES / name).read_bytes(), "text/html; charset=utf-8"


def _complete_responses() -> dict[str, tuple[bytes, str]]:
    return {
        VOLUME_URL: _fixture_response("volume-v235-complete.html"),
        ADA_URL: _fixture_response("lovelace24a.html"),
        SITE_ADA_PDF: (b"%PDF-1.7 offline fixture", "application/pdf"),
    }


def _partial_responses() -> dict[str, tuple[bytes, str]]:
    return {
        VOLUME_URL: _fixture_response("volume-v235.html"),
        ADA_URL: _fixture_response("lovelace24a.html"),
        TURING_URL: _fixture_response("turing24a.html"),
        RAW_ADA_PDF: (b"%PDF-1.7 offline fixture", "application/pdf"),
    }


def _collect_args(output_dir: Path, year: int = 2024) -> list[str]:
    return [
        "collect",
        "--venue",
        "icml",
        "--year",
        str(year),
        "--output",
        str(output_dir),
        "--contact",
        CONTACT,
    ]


def _track_replacements(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    destinations: list[str] = []
    real_replace = output_module.os.replace

    def recording_replace(source, destination):
        destinations.append(Path(destination).name)
        return real_replace(source, destination)

    monkeypatch.setattr(output_module.os, "replace", recording_replace)
    return destinations


def _assert_http_trace(
    opener: _FixtureOpener,
    expected_urls: list[str],
) -> None:
    assert [url for url, _, _ in opener.calls] == expected_urls
    assert [timeout for _, timeout, _ in opener.calls] == [30.0] * len(expected_urls)
    assert all(
        CONTACT in request.get_header("User-agent")
        for _, _, request in opener.calls
    )
    for url in expected_urls:
        expected_limit = 4096 if url in {RAW_ADA_PDF, SITE_ADA_PDF} else None
        assert opener.response_by_url[url].read_limits == [expected_limit]


def _compact_json(record: dict[str, object]) -> bytes:
    return json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"


def _assert_jsonl(path: Path, expected: list[dict[str, object]]) -> list[dict[str, object]]:
    data = path.read_bytes()
    assert not data.startswith(b"\xef\xbb\xbf")
    if not expected:
        assert data == b""
        return []

    assert data.endswith(b"\n")
    lines = data.splitlines(keepends=True)
    assert len(lines) == len(expected)
    actual: list[dict[str, object]] = []
    for line, expected_record in zip(lines, expected):
        assert line == _compact_json(expected_record)
        actual_record = json.loads(line)
        assert actual_record == expected_record
        actual.append(actual_record)
    return actual


def _assert_json(path: Path, expected: dict[str, object]) -> dict[str, object]:
    data = path.read_bytes()
    assert not data.startswith(b"\xef\xbb\xbf")
    assert data == _compact_json(expected)
    actual = json.loads(data)
    assert actual == expected
    return actual


def _assert_artifacts(output_dir: Path) -> None:
    assert sorted(path.name for path in output_dir.iterdir()) == [
        "issues.jsonl",
        "papers.jsonl",
        "run.json",
    ]
    assert not list(output_dir.glob(".paper-agent-*.tmp"))


def _assert_no_contact(output_dir: Path) -> None:
    for artifact in output_dir.iterdir():
        assert CONTACT.encode("utf-8") not in artifact.read_bytes()


def test_cli_pmlr_complete_publishes_verified_paper_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    opener = _FixtureOpener(_complete_responses())
    monkeypatch.setattr(http_module, "urlopen", opener)
    replacements = _track_replacements(monkeypatch)
    output_dir = tmp_path / "complete"

    assert cli.main(_collect_args(output_dir)) == 0

    _assert_http_trace(opener, [VOLUME_URL, ADA_URL, SITE_ADA_PDF])
    assert replacements == ["papers.jsonl", "issues.jsonl", "run.json"]
    _assert_artifacts(output_dir)
    assert capsys.readouterr().err == ""

    expected_paper = {
        "schema_version": 1,
        "venue_id": "icml",
        "venue_name": "International Conference on Machine Learning",
        "venue_type": "conference",
        "year": 2024,
        "source_name": "pmlr",
        "source_id": "v235/lovelace24a",
        "title": "Reliable Small Models & Graphs",
        "authors": ["Ada Lovelace", "Grace Hopper"],
        "abstract": "Reliable small models & graphs for reproducible experiments.",
        "doi": None,
        "landing_url": ADA_URL,
        "pdf_url": SITE_ADA_PDF,
        "access_status": "direct_pdf",
        "field_sources": {
            "title": "pmlr",
            "authors": "pmlr",
            "abstract": "pmlr",
            "doi": None,
            "landing_url": "pmlr",
            "pdf_url": "pmlr",
        },
    }
    papers = _assert_jsonl(output_dir / "papers.jsonl", [expected_paper])
    issues = _assert_jsonl(output_dir / "issues.jsonl", [])
    run = _assert_json(
        output_dir / "run.json",
        {
            "schema_version": 1,
            "status": "complete",
            "venue_id": "icml",
            "venue_name": "International Conference on Machine Learning",
            "venue_type": "conference",
            "year": 2024,
            "source_name": "pmlr",
            "membership_complete": True,
            "metadata_complete": True,
            "complete": True,
            "counts": {
                "raw_items": 1,
                "included_papers": 1,
                "complete_papers": 1,
                "incomplete_papers": 0,
                "excluded_non_papers": 0,
                "duplicate_occurrences": 0,
                "parse_rejects": 0,
                "issue_records": 0,
            },
            "pagination": {
                "pages_fetched": 1,
                "terminal_reached": True,
                "source_total": None,
            },
            "warnings": [],
            "errors": [],
        },
    )
    assert len(papers) == run["counts"]["complete_papers"]
    assert len(issues) == run["counts"]["issue_records"]
    _assert_no_contact(output_dir)


def test_cli_pmlr_partial_publishes_complete_members_and_blockers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    opener = _FixtureOpener(_partial_responses())
    monkeypatch.setattr(http_module, "urlopen", opener)
    replacements = _track_replacements(monkeypatch)
    output_dir = tmp_path / "partial"

    assert cli.main(_collect_args(output_dir)) == 3

    _assert_http_trace(opener, [VOLUME_URL, ADA_URL, TURING_URL, RAW_ADA_PDF])
    assert replacements == ["papers.jsonl", "issues.jsonl", "run.json"]
    _assert_artifacts(output_dir)
    assert capsys.readouterr().err == ""

    expected_paper = {
        "schema_version": 1,
        "venue_id": "icml",
        "venue_name": "International Conference on Machine Learning",
        "venue_type": "conference",
        "year": 2024,
        "source_name": "pmlr",
        "source_id": "v235/lovelace24a",
        "title": "Reliable Small Models & Graphs",
        "authors": ["Ada Lovelace", "Grace Hopper"],
        "abstract": "Reliable small models & graphs for reproducible experiments.",
        "doi": None,
        "landing_url": ADA_URL,
        "pdf_url": RAW_ADA_PDF,
        "access_status": "direct_pdf",
        "field_sources": {
            "title": "pmlr",
            "authors": "pmlr",
            "abstract": "pmlr",
            "doi": None,
            "landing_url": "pmlr",
            "pdf_url": "pmlr",
        },
    }
    expected_issues = [
        {
            "schema_version": 1,
            "issue_kind": "parse_reject",
            "venue_id": "icml",
            "year": 2024,
            "source_name": "pmlr",
            "source_id": None,
            "source_locator": f"{VOLUME_URL}#paper-5",
            "title": None,
            "authors": [],
            "abstract": None,
            "doi": None,
            "landing_url": None,
            "missing_fields": [],
            "reason_codes": ["missing_landing_url"],
            "message": "source item did not provide a valid PMLR landing URL",
        },
        {
            "schema_version": 1,
            "issue_kind": "incomplete_paper",
            "venue_id": "icml",
            "year": 2024,
            "source_name": "pmlr",
            "source_id": "v235/turing24a",
            "source_locator": TURING_URL,
            "title": "Parallel Inference",
            "authors": ["Alan Turing", "Grace Hopper"],
            "abstract": "Parallel inference reduces latency. It also preserves & checks accuracy.",
            "doi": None,
            "landing_url": TURING_URL,
            "missing_fields": ["access_locator"],
            "reason_codes": ["no_verified_pdf_or_doi"],
            "message": "required metadata or direct PDF access is missing",
        },
    ]
    papers = _assert_jsonl(output_dir / "papers.jsonl", [expected_paper])
    issues = _assert_jsonl(output_dir / "issues.jsonl", expected_issues)
    run = _assert_json(
        output_dir / "run.json",
        {
            "schema_version": 1,
            "status": "partial",
            "venue_id": "icml",
            "venue_name": "International Conference on Machine Learning",
            "venue_type": "conference",
            "year": 2024,
            "source_name": "pmlr",
            "membership_complete": False,
            "metadata_complete": False,
            "complete": False,
            "counts": {
                "raw_items": 5,
                "included_papers": 2,
                "complete_papers": 1,
                "incomplete_papers": 1,
                "excluded_non_papers": 1,
                "duplicate_occurrences": 1,
                "parse_rejects": 1,
                "issue_records": 2,
            },
            "pagination": {
                "pages_fetched": 1,
                "terminal_reached": True,
                "source_total": None,
            },
            "warnings": [],
            "errors": [],
        },
    )
    assert len(papers) == run["counts"]["complete_papers"]
    assert len(issues) == run["counts"]["issue_records"]
    assert issues[1]["missing_fields"] == ["access_locator"]
    assert issues[1]["reason_codes"] == ["no_verified_pdf_or_doi"]
    _assert_no_contact(output_dir)


def test_cli_pmlr_failed_membership_publishes_safe_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "secret-network-token=do-not-leak"
    calls: list[tuple[str, float]] = []

    def fail_on_call(request, *, timeout: float):
        calls.append((request.full_url, timeout))
        raise URLError(secret)

    monkeypatch.setattr(http_module, "urlopen", fail_on_call)
    output_dir = tmp_path / "failed"

    assert cli.main(_collect_args(output_dir)) == 4

    assert calls == [(VOLUME_URL, 30.0)]
    _assert_artifacts(output_dir)
    stderr = capsys.readouterr().err
    assert stderr == ""
    assert _assert_jsonl(output_dir / "papers.jsonl", []) == []
    assert _assert_jsonl(output_dir / "issues.jsonl", []) == []
    run = _assert_json(
        output_dir / "run.json",
        {
            "schema_version": 1,
            "status": "failed",
            "venue_id": "icml",
            "venue_name": "International Conference on Machine Learning",
            "venue_type": "conference",
            "year": 2024,
            "source_name": "pmlr",
            "membership_complete": False,
            "metadata_complete": False,
            "complete": False,
            "counts": {
                "raw_items": 0,
                "included_papers": 0,
                "complete_papers": 0,
                "incomplete_papers": 0,
                "excluded_non_papers": 0,
                "duplicate_occurrences": 0,
                "parse_rejects": 0,
                "issue_records": 0,
            },
            "pagination": None,
            "warnings": [],
            "errors": ["authoritative membership collection failed"],
        },
    )
    assert all(value is False for value in (
        run["membership_complete"],
        run["metadata_complete"],
        run["complete"],
    ))
    for artifact in output_dir.iterdir():
        assert secret.encode("utf-8") not in artifact.read_bytes()
    assert secret not in stderr
    _assert_no_contact(output_dir)


def test_cli_preflight_rejects_existing_artifact_without_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[object] = []

    def fail_on_call(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("network access")

    monkeypatch.setattr(http_module, "urlopen", fail_on_call)
    output_dir = tmp_path / "preflight"
    output_dir.mkdir()
    original = b"ORIGINAL-ARTIFACT\x00\xff\n"
    (output_dir / "papers.jsonl").write_bytes(original)

    assert cli.main(_collect_args(output_dir)) == 2

    assert calls == []
    assert capsys.readouterr().err == (
        "error: output_dir: formal artifact already exists: papers.jsonl\n"
    )
    assert (output_dir / "papers.jsonl").read_bytes() == original
    assert sorted(path.name for path in output_dir.iterdir()) == ["papers.jsonl"]
    assert not list(output_dir.glob(".paper-agent-*.tmp"))


def test_cli_not_applicable_publishes_empty_run_without_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[object] = []

    def fail_on_call(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("network access")

    monkeypatch.setattr(http_module, "urlopen", fail_on_call)
    output_dir = tmp_path / "not-applicable"

    assert cli.main(_collect_args(output_dir, year=1981)) == 0

    assert calls == []
    _assert_artifacts(output_dir)
    assert capsys.readouterr().err == ""
    assert _assert_jsonl(output_dir / "papers.jsonl", []) == []
    assert _assert_jsonl(output_dir / "issues.jsonl", []) == []
    run = _assert_json(
        output_dir / "run.json",
        {
            "schema_version": 1,
            "status": "not_applicable",
            "venue_id": "icml",
            "venue_name": "International Conference on Machine Learning",
            "venue_type": "conference",
            "year": 1981,
            "source_name": None,
            "membership_complete": False,
            "metadata_complete": False,
            "complete": False,
            "counts": {
                "raw_items": 0,
                "included_papers": 0,
                "complete_papers": 0,
                "incomplete_papers": 0,
                "excluded_non_papers": 0,
                "duplicate_occurrences": 0,
                "parse_rejects": 0,
                "issue_records": 0,
            },
            "pagination": None,
            "warnings": [],
            "errors": [],
        },
    )
    assert all(value is False for value in (
        run["membership_complete"],
        run["metadata_complete"],
        run["complete"],
    ))
    _assert_no_contact(output_dir)
