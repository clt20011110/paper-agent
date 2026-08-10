"""Offline CLI acceptance for the typed five-stage workflow."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from paper_agent import cli
from paper_agent.storage import Database
from paper_agent.workflow import (
    AnalyzeStep,
    DownloadStep,
    FileRef,
    FilterStep,
    ReportStep,
    SearchStep,
    StageKind,
    WorkflowManifest,
)
from paper_agent.workflow_adapters import (
    AnalyzeStageAdapter,
    DownloadStageAdapter,
    FilterStageAdapter,
    ReportStageAdapter,
    SearchStageAdapter,
)


ROOT = Path(__file__).parents[1]


def _ref(path: Path) -> FileRef:
    return FileRef(path.name, sha256(path.read_bytes()).hexdigest(), path)


def _json_ref(root: Path, name: str, value: object) -> FileRef:
    path = root / name
    path.write_text(json.dumps(value), encoding="utf-8")
    return _ref(path)


def _workflow_inputs(root: Path, database_path: Path) -> Path:
    config = yaml.safe_load(
        (ROOT / "configs" / "abstract_focus.yaml").read_text(encoding="utf-8")
    )
    config["storage"]["sqlite_path"] = str(database_path)
    config_path = root / "research.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    policy_path = root / "artifact-policy.yaml"
    policy_path.write_text(
        (ROOT / "policies" / "artifact-processing-v1.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    plan = _json_ref(root, "QUERY_PLAN.json", {"plan_id": "offline-e2e"})
    release = _json_ref(root, "stage2-release.json", {"release": "offline-e2e"})
    selection = _json_ref(
        root, "selection.json", {"schema_version": "1", "paper_ids": ["paper-e2e"]}
    )
    analysis_input = _json_ref(root, "analysis-input.json", {
        "schema_version": "1",
        "paper_ids": ["paper-e2e"],
        "stage3_artifact_ids": [],
    })
    report_plan = _json_ref(root, "REPORT_PLAN.json", {"plan_id": "report-e2e"})
    corpus = _json_ref(root, "CORPUS_SNAPSHOT.json", {"corpus_id": "offline-e2e"})
    audit = _json_ref(root, "SEARCH_AUDIT.json", {"crawl_run_id": "offline-e2e"})
    grants = _json_ref(root, "processing-grants.json", {
        "schema_version": "1", "grants": {"a" * 64: "processing-grant"},
    })
    manifest = WorkflowManifest(
        "offline-e2e",
        _ref(config_path),
        (
            SearchStep("search", plan, release, (), False),
            FilterStep("filter", plan, release, selection),
            DownloadStep("download", selection, None, None),
            AnalyzeStep("analyze", analysis_input, None, _ref(policy_path)),
            ReportStep("report", report_plan, corpus, audit, grants, None, _ref(policy_path)),
        ),
        root / "workflow.json",
    )
    workflow_path = root / "workflow.json"
    workflow_path.write_text(json.dumps(manifest.document()), encoding="utf-8")
    return workflow_path


def test_cli_typed_workflow_recovers_midway_with_only_fake_stage_boundaries(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """Exercise CLI -> orchestrator -> real adapters without external boundaries."""
    database_path = tmp_path / "papers.sqlite3"
    workflow_path = _workflow_inputs(tmp_path, database_path)
    calls: list[tuple[str, str]] = []
    pause_recovery_download = True

    def search_runner(_plan: object, _database: Path, **options: Any):
        calls.append(("search", options["run_id"]))
        return (
            SimpleNamespace(status="complete", paper_ids=("paper-e2e",)),
            options["run_id"],
            f"{options['run_id']}:crawl",
        )

    def filter_runner(**options: Any) -> dict[str, Any]:
        calls.append(("filter", options["campaign_id"]))
        return {"status": "complete", "campaign_id": options["campaign_id"]}

    class FakeDownloadService:
        def run(self, **options: Any) -> SimpleNamespace:
            nonlocal pause_recovery_download
            run_id = options["run_id"]
            calls.append(("download", run_id))
            if run_id == "recovery:download" and pause_recovery_download:
                pause_recovery_download = False
                status = "incomplete"
            else:
                status = "complete"
            return SimpleNamespace(
                run_id=run_id,
                paper_ids=tuple(options["paper_ids"]),
                status=status,
                dry_run=options["dry_run"],
            )

    class FakeAnalysisService:
        def run(self, run_id: str, manifest: Any, **options: Any) -> SimpleNamespace:
            calls.append(("analyze", run_id))
            return SimpleNamespace(
                run_id=run_id,
                dry_run=options["dry_run"],
                selected_paper_ids=manifest.paper_ids,
                input_scopes=("metadata_only",),
                result=SimpleNamespace(papers=(SimpleNamespace(status="complete"),)),
            )

    class FakeReportService:
        def run(self, report_run_id: str, pipeline_run_id: str, _bundle: Any, **options: Any) -> SimpleNamespace:
            assert report_run_id == pipeline_run_id
            calls.append(("report", report_run_id))
            return SimpleNamespace(
                report_run_id=report_run_id,
                status="complete",
                dry_run=options["dry_run"],
            )

    adapters = {
        StageKind.SEARCH: SearchStageAdapter(search_runner),
        StageKind.FILTER: FilterStageAdapter(filter_runner),
        StageKind.DOWNLOAD: DownloadStageAdapter(lambda *_args: FakeDownloadService()),
        StageKind.ANALYZE: AnalyzeStageAdapter(lambda *_args: FakeAnalysisService()),
        StageKind.REPORT: ReportStageAdapter(lambda *_args: FakeReportService()),
    }
    assert isinstance(adapters[StageKind.SEARCH], SearchStageAdapter)
    assert isinstance(adapters[StageKind.FILTER], FilterStageAdapter)
    assert isinstance(adapters[StageKind.DOWNLOAD], DownloadStageAdapter)
    assert isinstance(adapters[StageKind.ANALYZE], AnalyzeStageAdapter)
    assert isinstance(adapters[StageKind.REPORT], ReportStageAdapter)
    monkeypatch.setattr(cli, "default_stage_adapters", lambda: adapters)

    def recursive_cli(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("a workflow adapter must not recursively invoke the CLI")

    monkeypatch.setattr(cli, "entrypoint", recursive_cli)

    def invoke(command: str, run_id: str, expected_exit: int) -> dict[str, Any]:
        assert cli.main([
            "--config", str(tmp_path / "research.yaml"),
            command,
            "--workflow", str(workflow_path),
            "--database", str(database_path),
            "--workflow-run-id", run_id,
        ]) == expected_exit
        return json.loads(capsys.readouterr().out)

    first = invoke("run", "recovery", 1)
    assert first["status"] == "incomplete"
    assert [step["status"] for step in first["steps"]] == [
        "complete", "complete", "incomplete", "pending", "pending",
    ]
    assert calls == [
        ("search", "recovery:search"),
        ("filter", "recovery:filter"),
        ("download", "recovery:download"),
    ]

    resumed = invoke("resume", "recovery", 0)
    assert resumed["status"] == "complete"
    assert [step["status"] for step in resumed["steps"]] == ["complete"] * 5
    assert calls == [
        ("search", "recovery:search"),
        ("filter", "recovery:filter"),
        ("download", "recovery:download"),
        ("download", "recovery:download"),
        ("analyze", "recovery:analyze"),
        ("report", "recovery:report"),
    ]

    clean = invoke("run", "clean", 0)
    assert clean["status"] == "complete"
    assert calls[-5:] == [
        ("search", "clean:search"),
        ("filter", "clean:filter"),
        ("download", "clean:download"),
        ("analyze", "clean:analyze"),
        ("report", "clean:report"),
    ]

    with Database(database_path, read_only=True) as database:
        for run_id in ("recovery", "clean"):
            row = database.connection.execute(
                "SELECT status FROM workflow_runs WHERE workflow_run_id = ?", (run_id,)
            ).fetchone()
            assert row["status"] == "complete"
            steps = database.connection.execute(
                """SELECT child_run_id, status FROM workflow_steps
                   WHERE workflow_run_id = ? ORDER BY ordinal""",
                (run_id,),
            ).fetchall()
            assert [(row["child_run_id"], row["status"]) for row in steps] == [
                (f"{run_id}:{stage}", "complete")
                for stage in ("search", "filter", "download", "analyze", "report")
            ]
