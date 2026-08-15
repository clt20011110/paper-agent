"""Explicitly enabled anonymous live acceptance for four 2024 EDA journals."""

import json
import os
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from paper_agent import cli


_VENUES = ("tcad", "todaes", "tvlsi", "jssc")
_YEAR = 2024
_EXPECTED_COUNTS = {
    "tcad": {
        "raw_items": 429,
        "included_papers": 392,
        "complete_papers": 392,
        "incomplete_papers": 0,
        "excluded_non_papers": 37,
        "duplicate_occurrences": 0,
        "parse_rejects": 0,
        "issue_records": 0,
    },
    "todaes": {
        "raw_items": 99,
        "included_papers": 98,
        "complete_papers": 98,
        "incomplete_papers": 0,
        "excluded_non_papers": 1,
        "duplicate_occurrences": 0,
        "parse_rejects": 0,
        "issue_records": 0,
    },
    "tvlsi": {
        "raw_items": 269,
        "included_papers": 230,
        "complete_papers": 230,
        "incomplete_papers": 0,
        "excluded_non_papers": 39,
        "duplicate_occurrences": 0,
        "parse_rejects": 0,
        "issue_records": 0,
    },
    "jssc": {
        "raw_items": 409,
        # Four already-complete records from the prior run also have the
        # observed non-paper title "New Associate Editor".
        "included_papers": 334,
        "complete_papers": 334,
        "incomplete_papers": 0,
        "excluded_non_papers": 75,
        "duplicate_occurrences": 0,
        "parse_rejects": 0,
        "issue_records": 0,
    },
}
_OBSERVED_NON_PAPER_TITLES = {
    "todaes": {
        "Introduction to the Special Issue on Embedded System Software/Tools".casefold(),
    },
    "tvlsi": {"IEEE Foundation - Reflecting on 50 Years of Impact".casefold()},
    "jssc": {
        "IEEE JOURNAL OF SOLID-STATE CIRCUITS".casefold(),
        "IEEE Journal of Solid-State Circuits Information for Authors".casefold(),
        "Information For Authors".casefold(),
        "Together, we are advancing technology".casefold(),
        "Introducing IEEE Collabratec".casefold(),
        "TechRxiv".casefold(),
        "TechRxiv: Share Your Preprint Research with the World!".casefold(),
        "New Associate Editor".casefold(),
    },
}
_PAPER_KEYS = {
    "schema_version",
    "venue_id",
    "venue_name",
    "venue_type",
    "year",
    "source_name",
    "source_id",
    "title",
    "authors",
    "abstract",
    "doi",
    "landing_url",
    "pdf_url",
    "access_status",
    "field_sources",
}
_FIELD_SOURCE_KEYS = {
    "title",
    "authors",
    "abstract",
    "doi",
    "landing_url",
    "pdf_url",
}
_FIELD_SOURCE_NAMES = {"crossref_serial", "semantic_scholar", "openalex"}


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


def _assert_doi(value: object) -> None:
    assert isinstance(value, str) and value
    assert value == value.strip() == value.lower()
    assert value.startswith("10.") and "/" in value
    assert not any(character.isspace() for character in value)
    assert not value.startswith(("http:", "https:", "doi:"))


def _assert_complete_run(output_dir: Path, venue: str, contact: str) -> dict[str, object]:
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
    assert run["venue_type"] == "journal"
    assert run["year"] == _YEAR
    assert run["source_name"] == "crossref_serial"
    assert run["membership_complete"] is True
    assert run["metadata_complete"] is True
    assert run["complete"] is True
    assert run["errors"] == []
    assert run["warnings"] == []
    assert issues == []

    counts = run["counts"]
    assert isinstance(counts, dict)
    assert counts == _EXPECTED_COUNTS[venue]
    assert counts["raw_items"] > 0
    assert counts["included_papers"] > 0
    assert counts["raw_items"] == (
        counts["included_papers"]
        + counts["excluded_non_papers"]
        + counts["duplicate_occurrences"]
        + counts["parse_rejects"]
    )
    assert counts["included_papers"] == (
        counts["complete_papers"] + counts["incomplete_papers"]
    )
    assert counts["included_papers"] == counts["complete_papers"]
    assert counts["incomplete_papers"] == 0
    assert counts["parse_rejects"] == 0
    assert counts["issue_records"] == 0
    assert counts["issue_records"] == len(issues)
    assert len(papers) == counts["complete_papers"]

    pagination = run["pagination"]
    assert isinstance(pagination, dict)
    assert pagination["terminal_reached"] is True
    source_total = pagination["source_total"]
    assert isinstance(source_total, dict)
    assert source_total["scope"] == "raw_items"
    assert source_total["value"] == counts["raw_items"]

    identities = []
    included_titles = {paper["title"].casefold() for paper in papers}
    assert not included_titles.intersection(_OBSERVED_NON_PAPER_TITLES.get(venue, set()))
    for paper in papers:
        assert paper.keys() == _PAPER_KEYS
        assert paper["schema_version"] == 1
        assert paper["venue_id"] == venue
        assert paper["venue_type"] == "journal"
        assert paper["year"] == _YEAR
        assert paper["source_name"] == "crossref_serial"
        source_id = paper["source_id"]
        assert isinstance(source_id, str) and source_id
        _assert_doi(paper["doi"])
        assert source_id == paper["doi"]
        assert isinstance(paper["title"], str) and paper["title"].strip() == paper["title"]
        assert paper["title"]
        authors = paper["authors"]
        assert isinstance(authors, list) and authors
        assert all(isinstance(author, str) and author.strip() == author and author for author in authors)
        assert isinstance(paper["abstract"], str)
        assert paper["abstract"].strip() == paper["abstract"] and paper["abstract"]
        _assert_http_url(paper["landing_url"])

        access_status = paper["access_status"]
        if access_status == "direct_pdf":
            _assert_http_url(paper["pdf_url"])
        elif access_status == "doi_only":
            assert paper["pdf_url"] is None
        else:
            assert access_status in {"direct_pdf", "doi_only"}

        field_sources = paper["field_sources"]
        assert isinstance(field_sources, dict)
        assert field_sources.keys() == _FIELD_SOURCE_KEYS
        assert field_sources["title"] == "crossref_serial"
        assert field_sources["authors"] == "crossref_serial"
        assert field_sources["abstract"] in _FIELD_SOURCE_NAMES
        assert field_sources["doi"] == "crossref_serial"
        assert field_sources["landing_url"] == "crossref_serial"
        if paper["pdf_url"] is None:
            assert field_sources["pdf_url"] is None
        else:
            assert field_sources["pdf_url"] in _FIELD_SOURCE_NAMES
        for field in ("title", "authors", "abstract", "doi", "landing_url", "pdf_url"):
            assert (paper[field] is None) == (field_sources[field] is None)

        identities.append((venue, _YEAR, "crossref_serial", source_id))

    assert len(identities) == len(set(identities))
    assert identities == sorted(identities)
    return run


def _failure_details(output_dir: Path, venue: str, error: Exception) -> str:
    details = [f"{venue}: {type(error).__name__}: {error}"]
    run_path = output_dir / "run.json"
    if run_path.is_file():
        try:
            run = json.loads(run_path.read_text(encoding="utf-8"))
            details.append(
                "run="
                + json.dumps(
                    {
                        "status": run.get("status"),
                        "errors": run.get("errors"),
                        "counts": run.get("counts"),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        except (OSError, UnicodeError, ValueError) as read_error:
            details.append(f"run_read_error={read_error}")
    issues_path = output_dir / "issues.jsonl"
    if issues_path.is_file():
        try:
            issues = _read_jsonl(issues_path)
            details.append(
                "issues="
                + json.dumps(
                    [
                        {
                            "issue_kind": issue.get("issue_kind"),
                            "reason_codes": issue.get("reason_codes"),
                            "source_id": issue.get("source_id"),
                            "doi": issue.get("doi"),
                        }
                        for issue in issues
                    ],
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        except (OSError, UnicodeError, ValueError, AssertionError) as read_error:
            details.append(f"issues_read_error={read_error}")
    return " ".join(details)


@pytest.mark.live_smoke
@pytest.mark.skipif(
    os.environ.get("PAPER_AGENT_RUN_LIVE_SMOKE") != "1",
    reason="set PAPER_AGENT_RUN_LIVE_SMOKE=1 to authorize the live smoke test",
)
def test_eda_journals_2024_live_acceptance() -> None:
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
        try:
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
            assert exit_code == 0, f"{venue}: expected complete exit code 0"
            _assert_complete_run(output_dir, venue, contact)
        except Exception as error:
            failures.append(_failure_details(output_dir, venue, error))

    if failures:
        pytest.fail("\n".join(failures))
