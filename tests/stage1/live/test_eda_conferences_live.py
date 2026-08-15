"""Explicitly enabled anonymous live acceptance for five 2024 EDA conferences."""

import json
import os
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from paper_agent import cli


_VENUES = ("dac", "iccad", "date", "aspdac", "ispd")
_YEAR = 2024


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    payload = path.read_bytes()
    assert not payload.startswith(b"\xef\xbb\xbf")
    text = payload.decode("utf-8")
    if text:
        assert text.endswith("\n")
    records = []
    for line in text.splitlines():
        record = json.loads(line)
        assert isinstance(record, dict)
        records.append(record)
    return records


def _assert_http_url(value: object) -> None:
    assert isinstance(value, str) and value
    assert not any(character.isspace() for character in value)
    parsed = urlsplit(value)
    assert parsed.scheme in {"http", "https"}
    assert parsed.netloc and parsed.hostname
    assert parsed.username is None
    assert parsed.password is None


def _assert_complete_run(output_dir: Path, venue: str, contact: str) -> None:
    assert output_dir.is_dir()
    assert {path.name for path in output_dir.iterdir()} == {
        "papers.jsonl",
        "issues.jsonl",
        "run.json",
    }
    artifacts = {
        name: (output_dir / name).read_bytes()
        for name in ("papers.jsonl", "issues.jsonl", "run.json")
    }
    assert all(contact.encode("utf-8") not in payload for payload in artifacts.values())

    papers = _read_jsonl(output_dir / "papers.jsonl")
    issues = _read_jsonl(output_dir / "issues.jsonl")
    run_text = artifacts["run.json"].decode("utf-8")
    assert not run_text.startswith("\ufeff")
    assert run_text.endswith("\n")
    run = json.loads(run_text)
    assert isinstance(run, dict)
    assert run["status"] == "complete"
    assert run["venue_id"] == venue
    assert run["year"] == _YEAR
    assert run["membership_complete"] is True
    assert run["metadata_complete"] is True
    assert run["complete"] is True
    assert run["errors"] == []
    assert isinstance(run["warnings"], list)
    assert issues == []

    counts = run["counts"]
    assert isinstance(counts, dict)
    assert counts["included_papers"] > 0
    assert counts["raw_items"] == (
        counts["included_papers"]
        + counts["excluded_non_papers"]
        + counts["duplicate_occurrences"]
        + counts["parse_rejects"]
    )
    assert counts["included_papers"] == counts["complete_papers"]
    assert counts["included_papers"] == (
        counts["complete_papers"] + counts["incomplete_papers"]
    )
    assert counts["incomplete_papers"] == 0
    assert counts["parse_rejects"] == 0
    assert counts["issue_records"] == 0
    assert counts["issue_records"] == len(issues)
    assert len(papers) == counts["complete_papers"]

    identities = []
    for paper in papers:
        assert paper["title"]
        authors = paper["authors"]
        assert isinstance(authors, list) and authors
        assert all(isinstance(author, str) and author for author in authors)
        assert paper["abstract"]
        identities.append(
            (paper["venue_id"], paper["year"], paper["source_name"], paper["source_id"])
        )
        if paper["access_status"] == "direct_pdf":
            _assert_http_url(paper["pdf_url"])
        elif paper["access_status"] == "doi_only":
            assert paper["pdf_url"] is None
            doi = paper["doi"]
            assert isinstance(doi, str) and doi == doi.lower()
            assert doi.startswith("10.") and "/" in doi
            assert not any(character.isspace() for character in doi)
        else:
            pytest.fail(f"{venue}: paper has an invalid access_status")

    assert len(identities) == len(set(identities))
    assert identities == sorted(identities)


@pytest.mark.live_smoke
@pytest.mark.skipif(
    os.environ.get("PAPER_AGENT_RUN_LIVE_SMOKE") != "1",
    reason="set PAPER_AGENT_RUN_LIVE_SMOKE=1 to authorize the live smoke test",
)
def test_eda_conferences_2024_live_acceptance() -> None:
    contact = os.environ.get("PAPER_AGENT_SMOKE_CONTACT")
    if not contact:
        pytest.fail("PAPER_AGENT_SMOKE_CONTACT is required when live smoke is enabled")
    raw_output_root = os.environ.get("PAPER_AGENT_STAGE1_LIVE_OUTPUT_DIR")
    if not raw_output_root:
        pytest.fail(
            "PAPER_AGENT_STAGE1_LIVE_OUTPUT_DIR is required when live smoke is enabled"
        )

    output_root = Path(raw_output_root).resolve()
    repository_root = Path(__file__).resolve().parents[3]
    if output_root == repository_root or repository_root in output_root.parents:
        pytest.fail("PAPER_AGENT_STAGE1_LIVE_OUTPUT_DIR must be outside the repository")
    if output_root.exists():
        if not output_root.is_dir():
            pytest.fail("PAPER_AGENT_STAGE1_LIVE_OUTPUT_DIR must be a directory")
        if any(output_root.iterdir()):
            pytest.fail("PAPER_AGENT_STAGE1_LIVE_OUTPUT_DIR must be fresh and empty")

    failures = []
    for venue in _VENUES:
        output_dir = output_root / f"{venue}-{_YEAR}"
        exit_code = cli.main(
            [
                "collect",
                "--venue",
                venue,
                "--year",
                str(_YEAR),
                "--output",
                str(output_dir),
                "--contact",
                contact,
            ]
        )
        try:
            assert exit_code == 0, f"{venue}: expected complete exit code 0"
            _assert_complete_run(output_dir, venue, contact)
        except AssertionError as error:
            failures.append(f"{venue}: {error}")

    if failures:
        pytest.fail("\n".join(failures))
