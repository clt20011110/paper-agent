"""Offline CLI E2E coverage for the TCAD Crossref journal path."""

import json
from email.message import Message
from pathlib import Path

from paper_agent_next import cli
from paper_agent_next import http as http_module


FIXTURES = Path(__file__).parents[1] / "fixtures"
CONTACT = "integration@example.org"
CROSSREF_URL = (
    "https://api.crossref.org/journals/0278-0070/works?"
    "filter=from-pub-date%3A2024-01-01%2Cuntil-pub-date%3A2024-12-31%2Ctype%3Ajournal-article"
    "&rows=1000&cursor=%2A"
)
SEMANTIC_SCHOLAR_URL = (
    "https://api.semanticscholar.org/graph/v1/paper/batch"
    "?fields=abstract,externalIds,openAccessPdf"
)


class _Response:
    def __init__(self, body: bytes, content_type: str) -> None:
        self.body = body
        self.headers = Message()
        self.headers["Content-Type"] = content_type
        self.closed = False
        self.read_limits: list[int | None] = []

    def read(self, limit: int | None = None) -> bytes:
        self.read_limits.append(limit)
        return self.body

    def close(self) -> None:
        self.closed = True


class _FixtureOpener:
    def __init__(self, responses_by_url: dict[str, tuple[bytes, str]]) -> None:
        self.responses_by_url = responses_by_url
        self.calls: list[str] = []
        self.bodies: list[bytes | None] = []
        self.responses: list[_Response] = []

    def __call__(self, request, *, timeout: float) -> _Response:
        assert timeout == 30.0
        if request.full_url not in self.responses_by_url:
            raise AssertionError(f"unexpected offline URL: {request.full_url}")
        body, content_type = self.responses_by_url[request.full_url]
        response = _Response(body, content_type)
        self.calls.append(request.full_url)
        self.bodies.append(request.data)
        self.responses.append(response)
        return response


def test_cli_tcad_crossref_non_papers_and_s2_abstract_complete_offline(
    tmp_path: Path,
    monkeypatch,
) -> None:
    opener = _FixtureOpener(
        {
            CROSSREF_URL: (
                (FIXTURES / "crossref" / "tcad-e2e.json").read_bytes(),
                "application/json; charset=utf-8",
            ),
            SEMANTIC_SCHOLAR_URL: (
                (FIXTURES / "semantic_scholar" / "tcad-abstract.json").read_bytes(),
                "application/json; charset=utf-8",
            ),
        }
    )
    monkeypatch.setattr(http_module, "urlopen", opener)
    output_dir = tmp_path / "tcad-2024"

    assert cli.main(
        [
            "collect",
            "--venue",
            "tcad",
            "--year",
            "2024",
            "--output",
            str(output_dir),
            "--contact",
            CONTACT,
        ]
    ) == 0

    assert opener.calls == [CROSSREF_URL, SEMANTIC_SCHOLAR_URL]
    assert opener.bodies == [
        None,
        b'{"ids":["DOI:10.5555/tcad.missing"]}',
    ]
    assert all(response.closed for response in opener.responses)
    assert all("openalex" not in url for url in opener.calls)
    assert opener.responses[0].read_limits == [None]
    assert opener.responses[1].read_limits == [None]

    assert sorted(path.name for path in output_dir.iterdir()) == [
        "issues.jsonl",
        "papers.jsonl",
        "run.json",
    ]
    run = json.loads((output_dir / "run.json").read_text(encoding="utf-8"))
    assert run["status"] == "complete"
    assert run["membership_complete"] is True
    assert run["metadata_complete"] is True
    assert run["complete"] is True
    assert run["counts"] == {
        "raw_items": 3,
        "included_papers": 2,
        "complete_papers": 2,
        "incomplete_papers": 0,
        "excluded_non_papers": 1,
        "duplicate_occurrences": 0,
        "parse_rejects": 0,
        "issue_records": 0,
    }
    assert run["pagination"] == {
        "pages_fetched": 1,
        "terminal_reached": True,
        "source_total": {"value": 3, "scope": "raw_items"},
    }

    papers = [
        json.loads(line)
        for line in (output_dir / "papers.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [paper["source_id"] for paper in papers] == [
        "10.5555/tcad.missing",
        "10.5555/tcad.present",
    ]
    missing = papers[0]
    assert missing["abstract"] == "Semantic Scholar supplies the missing abstract."
    assert missing["field_sources"] == {
        "title": "crossref_serial",
        "authors": "crossref_serial",
        "abstract": "semantic_scholar",
        "doi": "crossref_serial",
        "landing_url": "crossref_serial",
        "pdf_url": None,
    }
    assert missing["access_status"] == "doi_only"
    assert missing["pdf_url"] is None
    assert papers[1]["abstract"] == "The Crossref abstract is already complete."
    assert papers[1]["field_sources"]["abstract"] == "crossref_serial"
    assert (output_dir / "issues.jsonl").read_text(encoding="utf-8") == ""
