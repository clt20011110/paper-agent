"""Offline CLI and collector E2E coverage for DBLP TOC."""

import json
from email.message import Message
from pathlib import Path

from paper_agent_next import cli
from paper_agent_next import http as http_module


FIXTURE = Path(__file__).parents[1] / "fixtures" / "dblp" / "partial.xml"
TOC_URL = "https://dblp.org/db/conf/dac/dac2024.xml"
S2_URL = (
    "https://api.semanticscholar.org/graph/v1/paper/batch"
    "?fields=abstract,externalIds,openAccessPdf"
)
OPENALEX_URL = (
    "https://api.openalex.org/works?"
    "filter=doi%3Ahttps%3A%2F%2Fdoi.org%2F10.1145%2Fpartial.dac"
    "&per-page=100"
    "&select=id%2Cdoi%2Cdisplay_name%2Cpublication_year%2Cauthorships%2C"
    "abstract_inverted_index%2Cbest_oa_location%2Cprimary_location"
)
CONTACT = "integration@example.org"


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
    def __init__(self, responses: dict[str, bytes]) -> None:
        self.responses_by_url = responses
        self.calls: list[str] = []
        self.bodies: list[bytes | None] = []
        self.responses: list[_Response] = []

    def __call__(self, request, *, timeout: float) -> _Response:
        assert timeout == 30.0
        assert request.full_url in self.responses_by_url
        self.calls.append(request.full_url)
        self.bodies.append(request.data)
        content_type = (
            "application/json; charset=utf-8"
            if request.full_url in {S2_URL, OPENALEX_URL}
            else "text/xml; charset=utf-8"
        )
        response = _Response(self.responses_by_url[request.full_url], content_type)
        self.responses.append(response)
        return response


def test_cli_dblp_s2_and_openalex_no_result_keeps_doi_and_reports_missing_abstract(
    tmp_path: Path, monkeypatch
) -> None:
    no_result = Path(__file__).parents[1] / "fixtures" / "semantic_scholar" / "dblp-no-result.json"
    openalex_no_result = Path(__file__).parents[1] / "fixtures" / "openalex" / "dblp-no-result.json"
    opener = _FixtureOpener(
        {
            TOC_URL: FIXTURE.read_bytes(),
            S2_URL: no_result.read_bytes(),
            OPENALEX_URL: openalex_no_result.read_bytes(),
        }
    )
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
    assert opener.calls == [TOC_URL, S2_URL, OPENALEX_URL]
    assert opener.bodies == [
        None,
        b'{"ids":["DOI:10.1145/partial.dac"]}',
        None,
    ]
    assert opener.responses[0].read_limits == [None]
    assert opener.responses[1].read_limits == [None]

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


def test_cli_dblp_s2_abstract_completes_as_doi_only_without_openalex_request(
    tmp_path: Path, monkeypatch
) -> None:
    abstract_fixture = Path(__file__).parents[1] / "fixtures" / "semantic_scholar" / "dblp-abstract.json"
    opener = _FixtureOpener({TOC_URL: FIXTURE.read_bytes(), S2_URL: abstract_fixture.read_bytes()})
    monkeypatch.setattr(http_module, "urlopen", opener)
    output_dir = tmp_path / "dac-2024-complete"

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
    ) == 0

    assert opener.calls == [TOC_URL, S2_URL]
    assert opener.bodies == [None, b'{"ids":["DOI:10.1145/partial.dac"]}']
    run = json.loads((output_dir / "run.json").read_text(encoding="utf-8"))
    assert run["status"] == "complete"
    assert run["metadata_complete"] is True
    assert run["complete"] is True
    assert json.loads((output_dir / "issues.jsonl").read_text(encoding="utf-8") or "null") is None
    papers = [
        json.loads(line)
        for line in (output_dir / "papers.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(papers) == 1
    assert papers[0]["abstract"] == "This abstract was supplied by Semantic Scholar."
    assert papers[0]["access_status"] == "doi_only"
    assert papers[0]["field_sources"]["abstract"] == "semantic_scholar"


def test_cli_dblp_s2_no_result_openalex_exact_completes_as_doi_only(
    tmp_path: Path, monkeypatch
) -> None:
    s2_no_result = Path(__file__).parents[1] / "fixtures" / "semantic_scholar" / "dblp-no-result.json"
    openalex_exact = Path(__file__).parents[1] / "fixtures" / "openalex" / "dblp-exact.json"
    opener = _FixtureOpener(
        {
            TOC_URL: FIXTURE.read_bytes(),
            S2_URL: s2_no_result.read_bytes(),
            OPENALEX_URL: openalex_exact.read_bytes(),
        }
    )
    monkeypatch.setattr(http_module, "urlopen", opener)
    output_dir = tmp_path / "dac-2024-openalex-complete"

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
    ) == 0

    assert opener.calls == [TOC_URL, S2_URL, OPENALEX_URL]
    assert opener.bodies == [
        None,
        b'{"ids":["DOI:10.1145/partial.dac"]}',
        None,
    ]
    run = json.loads((output_dir / "run.json").read_text(encoding="utf-8"))
    assert run["status"] == "complete"
    papers = [
        json.loads(line)
        for line in (output_dir / "papers.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(papers) == 1
    assert papers[0]["abstract"] == "OpenAlex abstract fills fixture"
    assert papers[0]["doi"] == "10.1145/partial.dac"
    assert papers[0]["access_status"] == "doi_only"
    assert papers[0]["field_sources"]["abstract"] == "openalex"


def test_cli_dblp_s2_failure_preserves_membership_and_is_partial(
    tmp_path: Path, monkeypatch
) -> None:
    failure_fixture = Path(__file__).parents[1] / "fixtures" / "semantic_scholar" / "dblp-bad.json"
    openalex_no_result = Path(__file__).parents[1] / "fixtures" / "openalex" / "dblp-no-result.json"
    opener = _FixtureOpener(
        {
            TOC_URL: FIXTURE.read_bytes(),
            S2_URL: failure_fixture.read_bytes(),
            OPENALEX_URL: openalex_no_result.read_bytes(),
        }
    )
    monkeypatch.setattr(http_module, "urlopen", opener)
    output_dir = tmp_path / "dac-2024-failure"

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
    assert opener.calls == [TOC_URL, S2_URL, OPENALEX_URL]
    assert opener.bodies == [
        None,
        b'{"ids":["DOI:10.1145/partial.dac"]}',
        None,
    ]
    assert run["membership_complete"] is True
    assert run["metadata_complete"] is False
    assert run["complete"] is False
    assert run["errors"] == ["enrichment semantic_scholar failed"]
    assert json.loads(
        (output_dir / "issues.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )["source_id"] == "conf/dac/Partial24"


def test_cli_marks_empty_applicable_dblp_toc_failed(
    tmp_path: Path, monkeypatch
) -> None:
    empty_url = TOC_URL
    opener = _FixtureOpener(
        {empty_url: (Path(__file__).parents[1] / "fixtures" / "dblp" / "empty.xml").read_bytes()}
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
    ) == 4

    run = json.loads((output_dir / "run.json").read_text(encoding="utf-8"))
    assert run["status"] == "failed"
    assert run["membership_complete"] is False
    assert run["metadata_complete"] is False
    assert run["complete"] is False
    assert run["counts"] == {
        "raw_items": 0,
        "included_papers": 0,
        "complete_papers": 0,
        "incomplete_papers": 0,
        "excluded_non_papers": 0,
        "duplicate_occurrences": 0,
        "parse_rejects": 0,
        "issue_records": 0,
    }
    assert run["pagination"] is None
    assert run["errors"] == [
        "authoritative membership collection failed"
    ]
