"""Productization contracts kept in sync with the thin Codex skill."""

from __future__ import annotations

from pathlib import Path

import pytest

from paper_agent.cli import build_parser


ROOT = Path(__file__).parents[1]
SKILL = ROOT / "skills" / "paper-agent" / "SKILL.md"

# This is the Phase 7 public command surface from task.md.  It is deliberately
# independent of implementation modules: deleting a parser entry must fail CI
# rather than leaving the skill to promise a private Python fallback.
REQUIRED_TOP_LEVEL_COMMANDS = frozenset({
    "doctor", "search", "import-seeds", "crawl", "filter", "grant", "download",
    "analyze", "report", "verify-report", "run", "resume", "export", "import",
    "migrate-config", "benchmark-stage2",
})

REQUIRED_SKILL_COMMANDS = (
    "paper-agent doctor",
    "paper-agent search plan | approve | run | expand-citations | audit",
    "paper-agent import-seeds",
    "paper-agent crawl",
    "paper-agent filter",
    "paper-agent grant create | approve | revoke",
    "paper-agent download",
    "paper-agent analyze",
    "paper-agent report prepare-inputs",
    "paper-agent report --plan-only",
    "paper-agent report approve --plan <REPORT_PLAN.json> --hash <sha256>",
    "paper-agent report --plan <REPORT_PLAN.json>",
    "paper-agent report --diff-from <previous_report_run_id> --report-run-id <current_report_run_id> --output-root <DIR>",
    "paper-agent verify-report",
    "paper-agent run",
    "paper-agent resume",
    "paper-agent export",
    "paper-agent import",
    "paper-agent migrate-config",
    "paper-agent benchmark-stage2",
)


def test_phase7_cli_surface_is_registered() -> None:
    parser = build_parser()
    action = next(
        action for action in parser._actions  # noqa: SLF001 - argparse has no public lookup.
        if getattr(action, "choices", None) is not None
    )
    assert REQUIRED_TOP_LEVEL_COMMANDS <= set(action.choices)


def test_skill_lists_only_the_phase7_cli_contract() -> None:
    rendered = SKILL.read_text(encoding="utf-8")
    assert "only business implementation" in rendered
    for command in REQUIRED_SKILL_COMMANDS:
        assert command in rendered


def test_workflow_report_and_authorized_download_contracts_are_explicit() -> None:
    parser = build_parser()
    run = parser.parse_args([
        "run", "--workflow", "workflow.json", "--workflow-run-id", "workflow-1",
    ])
    resume = parser.parse_args([
        "resume", "--workflow", "workflow.json", "--workflow-run-id", "workflow-1",
    ])
    assert run.workflow == Path("workflow.json")
    assert resume.workflow == Path("workflow.json")
    with pytest.raises(SystemExit):
        parser.parse_args(["resume"])

    parser.parse_args([
        "download", "--authorized-skill-queue", "queue.csv",
        "--authorized-skill-output", "output",
        "--authorized-skill-root", "installed-skill",
    ])
    parser.parse_args([
        "report", "--plan", "REPORT_PLAN.json", "--database", "papers.sqlite3",
        "--output-root", "reports", "--processing-grant", f"{'a' * 64}=grant-1",
    ])

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    operations = (ROOT / "docs" / "operations.md").read_text(encoding="utf-8")
    skill = SKILL.read_text(encoding="utf-8")
    for rendered in (readme, operations, skill):
        assert "--workflow" in rendered
        assert "--authorized-skill-queue" in rendered
        assert "--processing-grant" in rendered
    assert "paper-agent resume --run-id" not in operations
