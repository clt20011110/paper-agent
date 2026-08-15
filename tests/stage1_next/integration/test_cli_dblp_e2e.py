"""Offline CLI and collector E2E coverage for DBLP TOC."""

import json
from email.message import Message
from pathlib import Path

from paper_agent_next import cli
from paper_agent_next import http as http_module


FIXTURE = Path(__file__).parents[1] / "fixtures" / "dblp" / "partial.xml"
TOC_URL = "https://dblp.org/db/conf/dac/dac2024.xml"
CONTACT = "integration@example.org"


class _Response:
    def __init__(self, body: bytes) -> None:
        self.body = body
        self.headers = Message()
        self.headers["Content-Type"] = "text/xml; charset=utf-8"
        self.closed = False
        self.read_limits: list[int | None] = []

    def read(self, limit: int | None = None) -> bytes:
        self.read_limits.append(limit)
        return self.body

    def close(self) -> None:
        self.closed = True


class _FixtureOpener:
    def __init__(self, url: str, body: bytes) -> None:
        self.url = url
        self.body = body
        self.calls: list[str] = []
        self.responses: list[_Response] = []

    def __call__(self, request, *, timeout: float) -> _Response:
        assert timeout == 30.0
        assert request.full_url == self.url
        self.calls.append(request.full_url)
        response = _Response(self.body)
        self.responses.append(response)
        return response


def test_cli_keeps_doi_and_reports_only_missing_abstract_for_dblp_partial_run(
    tmp_path: Path, monkeypatch
) -> None:
    opener = _FixtureOpener(TOC_URL, FIXTURE.read_bytes())
    monkeypatch.setattr(http_module, "urlopen", opener)
    output_dir = tmp_path / "dac-2024"

    exit_code = cli.main(
        [
            "collect",
            "--venue",
            "dac",
            "--year",
            "2024",
            "--output",
            str(output_dir),
            "--contact",
            CONTACT,
        ]
    )

    assert exit_code == 3
    assert opener.calls == [TOC_URL]
    assert opener.responses[0].read_limits == [None]

    run = json.loads((output_dir / "run.json").read_text(encoding="utf-8"))
    assert run["status"] == "partial"
    assert run["membership_complete"] is True
    assert run["metadata_complete"] is False
    assert run["complete"] is False
    assert run["counts"] == {
        "raw_items": 1,
        "included_papers": 1,
        "complete_papers": 0,
        "incomplete_papers": 1,
        "excluded_non_papers": 0,
        "duplicate_occurrences": 0,
        "parse_rejects": 0,
        "issue_records": 1,
    }
    assert run["pagination"] == {
        "pages_fetched": 1,
        "terminal_reached": True,
        "source_total": None,
    }
    assert run["errors"] == []

    papers = (output_dir / "papers.jsonl").read_text(encoding="utf-8")
    assert papers == ""
    issues = [
        json.loads(line)
        for line in (output_dir / "issues.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(issues) == 1
    assert issues[0]["source_id"] == "conf/dac/Partial24"
    assert issues[0]["doi"] == "10.1145/partial.dac"
    assert issues[0]["landing_url"] == "https://dblp.org/rec/conf/dac/Partial24"
    assert issues[0]["missing_fields"] == ["abstract"]
    assert issues[0]["reason_codes"] == ["missing_abstract"]


def test_cli_does_not_mark_an_empty_applicable_dblp_toc_complete(
    tmp_path: Path, monkeypatch
) -> None:
    empty_url = TOC_URL
    opener = _FixtureOpener(
        empty_url,
        (Path(__file__).parents[1] / "fixtures" / "dblp" / "empty.xml").read_bytes(),
    )
    monkeypatch.setattr(http_module, "urlopen", opener)
    output_dir = tmp_path / "empty-dac-2024"

    assert cli.main(
        [
            "collect",
            "--venue",
            "dac",
            "--year",
            "2024",
            "--output",
            str(output_dir),
            "--contact",
            CONTACT,
        ]
    ) == 3

    run = json.loads((output_dir / "run.json").read_text(encoding="utf-8"))
    assert run["status"] == "partial"
    assert run["membership_complete"] is False
    assert run["metadata_complete"] is True
    assert run["complete"] is False
    assert run["counts"]["raw_items"] == 0
    assert run["errors"] == [
        "applicable venue-year has no authoritative zero-paper proof"
    ]
