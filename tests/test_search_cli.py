import json
from contextlib import closing
from pathlib import Path
import sqlite3

import pytest

from paper_agent.approval import ApprovalError
from paper_agent.cli import main
from paper_agent.query_plan import QueryPlanDriftError


@pytest.fixture(autouse=True)
def openalex_credentials(monkeypatch) -> None:
    monkeypatch.setenv("OPENALEX_API_KEY", "test-key")


def _digest(character: str) -> str:
    return character * 64


def _draft() -> dict[str, object]:
    return {
        "created_at": "2026-08-09T00:00:00Z",
        "research": {
            "objective": "map graph learning",
            "audience": "researchers",
            "primary_question": "What methods work?",
            "subquestions": [{"id": "sq1", "question": "Which methods?"}],
        },
        "scope": {
            "date_from": "2020-01-01",
            "date_to": "2024-12-31",
            "venues": [],
            "fields": ["computer science"],
            "languages": ["en"],
            "document_types": ["article"],
            "user_seeds": [],
        },
        "inclusion": {"criteria": ["empirical"], "exclusion_criteria": ["unrelated"]},
        "query_variants": [
            {
                "id": "q1",
                "subquestion_id": "sq1",
                "alias_group": "graph",
                "raw_query": "graph learning",
                "synonyms": ["GNN"],
            }
        ],
        "providers": ["openalex"],
        "filter": {
            "profile": "fake",
            "config_hash": _digest("c"),
            "thresholds_hash": _digest("d"),
            "seed_selector_version": "1",
            "seed_selector_config_hash": _digest("e"),
            "round_state_machine_version": "1",
        },
        "citation_snowball": {
            "enabled": True,
            "directions": ["references", "citations"],
            "max_depth": 2,
            "max_rounds": 3,
            "max_per_seed_per_source": 20,
        },
        "budgets": {
            "max_requests": 100,
            "max_candidates": 1000,
            "max_seconds": 300,
            "saturation": {"min_unique_included_yield": 0.05, "consecutive_low_yield_rounds": 2},
        },
        "provider_policy": "all_resolved",
        "required_roles": ["search"],
        "required_providers": ["openalex"],
    }


def _plan(tmp_path, capsys) -> tuple[dict[str, object], str]:
    input_path = tmp_path / "search.yaml"
    input_path.write_text(json.dumps(_draft()), encoding="utf-8")
    assert main(["search", "plan", "--input", str(input_path), "--output-root", str(tmp_path / "output")]) == 0
    output = json.loads(capsys.readouterr().out)
    return output, output["draft_path"]


def test_search_plan_approval_run_and_history_are_frozen(tmp_path, capsys, monkeypatch) -> None:
    first, draft_path = _plan(tmp_path, capsys)
    draft = json.loads((tmp_path / "output" / "search" / first["plan_id"] / "QUERY_PLAN.draft.json").read_text())

    with pytest.raises(SystemExit):
        main(["search", "approve", "--plan", draft_path, "--approved-by", "owner"])

    with pytest.raises(ApprovalError, match="content hash mismatch"):
        main(
            [
                "search",
                "approve",
                "--plan",
                draft_path,
                "--hash",
                _digest("0"),
                "--approved-by",
                "owner",
                "--approved-at",
                "2026-08-09T01:00:00Z",
            ]
        )

    assert main(
        [
            "search",
            "approve",
            "--plan",
            draft_path,
            "--hash",
            first["plan_hash"],
            "--approved-by",
            "owner",
            "--approved-at",
            "2026-08-09T01:00:00Z",
        ]
    ) == 0
    approved = json.loads(capsys.readouterr().out)
    assert (tmp_path / "output" / "search" / "latest-approved.json").exists()

    assert main(["--dry-run", "search", "run", "--plan", approved["approved_path"]]) == 0
    assert json.loads(capsys.readouterr().out)["provider_invocation"] == "skipped_dry_run"

    monkeypatch.delenv("OPENALEX_API_KEY")
    with pytest.raises(QueryPlanDriftError, match="credential|unavailable"):
        main(["--dry-run", "search", "run", "--plan", approved["approved_path"]])
    monkeypatch.setenv("OPENALEX_API_KEY", "test-key")

    changed = _draft()
    changed["query_variants"][0]["raw_query"] = "graph representation learning"  # type: ignore[index]
    input_path = tmp_path / "search.yaml"
    input_path.write_text(json.dumps(changed), encoding="utf-8")
    assert main(["search", "plan", "--input", str(input_path), "--output-root", str(tmp_path / "output")]) == 0
    second = json.loads(capsys.readouterr().out)
    assert second["plan_id"] != first["plan_id"]
    assert (tmp_path / "output" / "search" / first["plan_id"] / "QUERY_PLAN.json").exists()
    assert (tmp_path / "output" / "search" / second["plan_id"] / "QUERY_PLAN.draft.json").exists()


def test_search_run_executes_library_provider_and_is_idempotent(tmp_path, capsys) -> None:
    document = _draft()
    document["providers"] = ["user_library"]
    document["scope"]["user_seeds"] = ["doi:10.1000/library-seed"]  # type: ignore[index]
    document["citation_snowball"]["enabled"] = False  # type: ignore[index]
    document["required_roles"] = ["library"]
    document["required_providers"] = ["user_library"]
    input_path = tmp_path / "library-search.yaml"
    input_path.write_text(json.dumps(document), encoding="utf-8")

    assert main(
        ["search", "plan", "--input", str(input_path), "--output-root", str(tmp_path / "output")]
    ) == 0
    draft = json.loads(capsys.readouterr().out)
    assert main(
        [
            "search",
            "approve",
            "--plan",
            draft["draft_path"],
            "--hash",
            draft["plan_hash"],
            "--approved-by",
            "owner",
            "--approved-at",
            "2026-08-09T01:00:00Z",
        ]
    ) == 0
    approved = json.loads(capsys.readouterr().out)
    database = tmp_path / "papers.sqlite3"
    command = [
        "--run-id",
        "library-run",
        "search",
        "run",
        "--plan",
        approved["approved_path"],
        "--database",
        str(database),
    ]

    assert main(command) == 0
    first = json.loads(capsys.readouterr().out)
    assert (first["provider_invocation"], first["status"], first["paper_count"]) == (
        "completed",
        "complete",
        1,
    )
    assert main(command) == 0
    assert json.loads(capsys.readouterr().out)["crawl_run_id"] == first["crawl_run_id"]

    with closing(sqlite3.connect(database)) as connection:
        assert connection.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM source_runs").fetchone()[0] == 1


def test_crawl_and_citation_planning_emit_stable_audit_intent(tmp_path, capsys) -> None:
    first, draft_path = _plan(tmp_path, capsys)
    assert main(
        [
            "search",
            "approve",
            "--plan",
            draft_path,
            "--hash",
            first["plan_hash"],
            "--approved-by",
            "owner",
            "--approved-at",
            "2026-08-09T01:00:00Z",
        ]
    ) == 0
    approved = json.loads(capsys.readouterr().out)
    seeds = [
        {
            "paper_id": "paper-1",
            "seed_reason": "user_seed",
            "parent_round": 0,
            "depth": 0,
            "subquestion_id": "sq1",
            "rank": 0,
            "selector_version": "1",
            "selector_config_hash": _digest("e"),
        }
    ]
    seeds_path = tmp_path / "seeds.json"
    seeds_path.write_text(json.dumps(seeds), encoding="utf-8")
    command = [
        "search",
        "expand-citations",
        "--plan",
        approved["approved_path"],
        "--seeds",
        str(seeds_path),
        "--round-index",
        "1",
    ]
    assert main(command) == 0
    first_manifest = capsys.readouterr().out
    assert main(command) == 0
    assert capsys.readouterr().out == first_manifest

    assert main(["crawl", "--venue", "neurips"]) == 0
    crawl = json.loads(capsys.readouterr().out)
    assert crawl["mode"] == "venue_descriptor_compatibility"
    assert crawl["search_audit_intent"]["venue_ids"] == ["neurips"]


def test_venue_scope_automatically_freezes_exact_primary_provider(tmp_path, capsys) -> None:
    document = _draft()
    document["scope"]["venues"] = ["icml"]  # type: ignore[index]
    input_path = tmp_path / "search.yaml"
    input_path.write_text(json.dumps(document), encoding="utf-8")

    assert main(["search", "plan", "--input", str(input_path), "--output-root", str(tmp_path / "output")]) == 0
    result = json.loads(capsys.readouterr().out)
    plan = json.loads(Path(result["draft_path"]).read_text())

    assert plan["execution"]["required_providers"] == ["openalex", "pmlr"]
    assert {provider["provider"] for provider in plan["providers"]} == {"openalex", "pmlr"}
