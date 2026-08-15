"""Explicitly enabled anonymous live acceptance for the historical ICML PMLR volume."""

import json
import os
from pathlib import Path

import pytest

from paper_agent_next import cli
from paper_agent_next.access import resolve_access
from paper_agent_next.http import HttpClient
from paper_agent_next.models import AccessStatus


@pytest.mark.live_smoke
@pytest.mark.skipif(
    os.environ.get("PAPER_AGENT_RUN_LIVE_SMOKE") != "1",
    reason="set PAPER_AGENT_RUN_LIVE_SMOKE=1 to authorize the live smoke test",
)
def test_pmlr_icml_2015_live_acceptance() -> None:
    contact = os.environ.get("PAPER_AGENT_SMOKE_CONTACT")
    if not contact:
        pytest.fail("PAPER_AGENT_SMOKE_CONTACT is required when live smoke is enabled")
    raw_output_dir = os.environ.get("PAPER_AGENT_STAGE1_LIVE_OUTPUT_DIR")
    if not raw_output_dir:
        pytest.fail("PAPER_AGENT_STAGE1_LIVE_OUTPUT_DIR is required when live smoke is enabled")

    output_dir = Path(raw_output_dir).resolve()
    repository_root = Path(__file__).resolve().parents[3]
    if output_dir == repository_root or repository_root in output_dir.parents:
        pytest.fail("PAPER_AGENT_STAGE1_LIVE_OUTPUT_DIR must be outside the repository")
    if output_dir.exists():
        if not output_dir.is_dir():
            pytest.fail("PAPER_AGENT_STAGE1_LIVE_OUTPUT_DIR must be a directory")
        if any(output_dir.iterdir()):
            pytest.fail("PAPER_AGENT_STAGE1_LIVE_OUTPUT_DIR must be fresh and empty")

    exit_code = cli.main(
        [
            "collect",
            "--venue",
            "icml",
            "--year",
            "2015",
            "--output",
            str(output_dir),
            "--contact",
            contact,
        ]
    )
    assert exit_code in {0, 3}

    assert output_dir.is_dir()
    assert {path.name for path in output_dir.iterdir()} == {
        "papers.jsonl",
        "issues.jsonl",
        "run.json",
    }
    artifacts = {name: (output_dir / name).read_bytes() for name in (
        "papers.jsonl",
        "issues.jsonl",
        "run.json",
    )}
    assert all(contact.encode("utf-8") not in data for data in artifacts.values())

    def read_jsonl(name: str) -> list[dict[str, object]]:
        text = artifacts[name].decode("utf-8")
        if text:
            assert text.endswith("\n")
        records = []
        for line in text.splitlines():
            record = json.loads(line)
            assert isinstance(record, dict)
            records.append(record)
        return records

    papers = read_jsonl("papers.jsonl")
    issues = read_jsonl("issues.jsonl")
    run_text = artifacts["run.json"].decode("utf-8")
    assert run_text.endswith("\n")
    run = json.loads(run_text)
    assert isinstance(run, dict)

    assert (exit_code, run["status"]) in {(0, "complete"), (3, "partial")}
    assert run["venue_id"] == "icml"
    assert run["year"] == 2015
    assert run["source_name"] == "pmlr"
    pagination = run["pagination"]
    assert isinstance(pagination, dict)
    assert pagination["pages_fetched"] == 1
    assert pagination["terminal_reached"] is True

    counts = run["counts"]
    assert isinstance(counts, dict)
    assert counts["raw_items"] > 0
    assert counts["included_papers"] > 0
    assert counts["raw_items"] == (
        counts["included_papers"]
        + counts["excluded_non_papers"]
        + counts["duplicate_occurrences"]
        + counts["parse_rejects"]
    )
    assert counts["included_papers"] == counts["complete_papers"] + counts["incomplete_papers"]
    assert len(papers) == counts["complete_papers"]
    assert len(issues) == counts["issue_records"]

    identities = []
    for paper in papers:
        assert paper["title"]
        assert paper["authors"]
        assert all(author for author in paper["authors"])
        assert paper["abstract"]
        identities.append((paper["venue_id"], paper["year"], paper["source_name"], paper["source_id"]))
        if paper["access_status"] == "direct_pdf":
            assert paper["pdf_url"]
        elif paper["access_status"] == "doi_only":
            assert paper["pdf_url"] is None
            assert paper["doi"]
        else:
            pytest.fail("paper has an invalid access_status")
    assert len(identities) == len(set(identities))
    assert [identity[3] for identity in identities] == sorted(identity[3] for identity in identities)

    assert all(issue["reason_codes"] for issue in issues)
    incomplete_issues = [issue for issue in issues if issue["issue_kind"] == "incomplete_paper"]
    membership_issues = [
        issue for issue in issues if issue["issue_kind"] in {"parse_reject", "identity_conflict"}
    ]
    assert len(incomplete_issues) == counts["incomplete_papers"]
    assert len(membership_issues) == counts["parse_rejects"]

    if run["status"] == "complete":
        assert exit_code == 0
        assert run["membership_complete"] is True
        assert run["metadata_complete"] is True
        assert run["complete"] is True
        assert issues == []
        assert run["errors"] == []
    else:
        assert exit_code == 3
        assert run["complete"] is False
        assert counts["issue_records"] > 0 or run["errors"]
        if not run["membership_complete"]:
            assert membership_issues or run["errors"]
        if not run["metadata_complete"]:
            assert incomplete_issues or run["errors"]

    pdf_urls = [paper["pdf_url"] for paper in papers if paper["access_status"] == "direct_pdf"]
    probe_urls = list(dict.fromkeys(pdf_urls[:1] + pdf_urls[-1:]))
    http_client = HttpClient(contact, 30.0)
    for pdf_url in probe_urls:
        decision = resolve_access((pdf_url,), None, http_client)
        assert decision.access_status is AccessStatus.DIRECT_PDF
        assert decision.pdf_url == pdf_url
