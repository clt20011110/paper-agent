from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace
from typing import Any, Mapping

import pytest
import yaml

from paper_agent.storage import Database
from paper_agent.workflow import (
    AnalyzeStep,
    DownloadStep,
    FileRef,
    FilterStep,
    ReportStep,
    SearchStep,
    SequentialWorkflowOrchestrator,
    StageKind,
    StepContext,
    StepObservation,
    WorkflowManifest,
)
from paper_agent.workflow_adapters import (
    AnalyzeStageAdapter,
    DownloadStageAdapter,
    FilterStageAdapter,
    ReportStageAdapter,
    SearchStageAdapter,
)


ROOT = Path(__file__).resolve().parents[1]


def _write(tmp_path: Path, name: str, value: object) -> FileRef:
    path = tmp_path / name
    path.write_text(json.dumps(value), encoding="utf-8")
    return _ref(path)


def _ref(path: Path) -> FileRef:
    return FileRef(path.name, sha256(path.read_bytes()).hexdigest(), path)


def _context(tmp_path: Path, *, dry_run: bool = False) -> StepContext:
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    return StepContext(
        database_path=tmp_path / "papers.sqlite3",
        config_path=config,
        workflow_run_id="workflow-7",
        child_run_id="workflow-7:step",
        dry_run=dry_run,
    )


def test_adapters_call_typed_services_with_fixed_child_run_id(
    tmp_path: Path, monkeypatch
) -> None:
    context = _context(tmp_path)
    policy = _ref(ROOT / "policies" / "artifact-processing-v1.yaml")
    selection = _write(tmp_path, "selection.json", {"schema_version": "1", "paper_ids": ["paper-1"]})
    analysis_selection = _write(tmp_path, "analysis.json", {
        "schema_version": "1", "paper_ids": ["paper-1"], "stage3_artifact_ids": [],
    })
    grants = _write(tmp_path, "grants.json", {"a" * 64: "grant-7"})
    plan = _write(tmp_path, "plan.json", {"plan": "frozen"})
    release = _write(tmp_path, "release.json", {"release": "frozen"})
    corpus = _write(tmp_path, "corpus.json", {"corpus": "frozen"})
    audit = _write(tmp_path, "audit.json", {"audit": "frozen"})
    calls: list[tuple[str, Any]] = []
    analysis_factory_options: list[Mapping[str, Any]] = []
    report_factory_args: list[tuple[Any, ...]] = []
    report_runtime_checks: list[tuple[Mapping[str, Any], str]] = []

    def search_runner(plan_value, database_path, **kwargs):
        calls.append(("search", (plan_value, database_path, kwargs)))
        return SimpleNamespace(status="complete", paper_ids=("paper-1",)), kwargs["run_id"], "crawl-7"

    def filter_runner(**kwargs):
        calls.append(("filter", kwargs))
        return {"status": "complete", "campaign_id": kwargs["campaign_id"]}

    @dataclass
    class DownloadService:
        def run(self, **kwargs):
            calls.append(("download", kwargs))
            return SimpleNamespace(
                run_id=kwargs["run_id"], paper_ids=tuple(kwargs["paper_ids"]),
                status="complete", dry_run=kwargs["dry_run"],
            )

    def download_factory(*_args):
        return DownloadService()

    @dataclass
    class AnalysisService:
        def run(self, run_id, manifest, **kwargs):
            calls.append(("analyze", (run_id, manifest, kwargs)))
            return SimpleNamespace(
                run_id=run_id, dry_run=kwargs["dry_run"], selected_paper_ids=manifest.paper_ids,
                input_scopes=("abstract_only",),
                result=SimpleNamespace(papers=(SimpleNamespace(status="complete"),)),
            )

    def analysis_factory(*_args, **options):
        analysis_factory_options.append(options)
        return AnalysisService()

    @dataclass
    class ReportService:
        def run(self, report_run_id, pipeline_run_id, bundle, **kwargs):
            calls.append(("report", (report_run_id, pipeline_run_id, bundle, kwargs)))
            return SimpleNamespace(report_run_id=report_run_id, status="complete", dry_run=kwargs["dry_run"])

    def report_factory(*args):
        report_factory_args.append(args)
        return ReportService()

    # Only execution needs the v2 configuration; adapters remain independently
    # testable with fake typed services.
    report_root = tmp_path / "configured-output"
    report_runtime = SimpleNamespace(
        enabled=True,
        resources=object(),
        validate_for_run=lambda plan, *, execution_mode: report_runtime_checks.append(
            (plan, execution_mode)
        ),
    )
    monkeypatch.setattr(
        "paper_agent.workflow_adapters.load_config",
        lambda _path: {
            "project": {"output_dir": str(report_root)},
            "analysis": {
                "workers": 7,
                "allow_abstract_only": False,
                "output_schema": "./schemas/paper-analysis.schema.json",
            },
        },
    )
    monkeypatch.setattr(
        "paper_agent.workflow_adapters._report_runtime_config",
        lambda _config, _path: report_runtime,
    )
    monkeypatch.setattr(
        "paper_agent.workflow_adapters.assert_report_plan_resource_binding",
        lambda _plan, _resources: None,
    )

    search = SearchStageAdapter(search_runner)
    search_spec = SearchStep("step", plan, release, (), True)
    search.execute(context, search_spec, search.validate(context, search_spec))

    filtering = FilterStageAdapter(filter_runner)
    filter_spec = FilterStep("step", plan, release, selection)
    filtering.execute(context, filter_spec, filtering.validate(context, filter_spec))

    download = DownloadStageAdapter(download_factory)
    download_spec = DownloadStep("step", selection, "download-grant", None)
    download.execute(context, download_spec, download.validate(context, download_spec))

    analysis = AnalyzeStageAdapter(analysis_factory)
    analyze_spec = AnalyzeStep("step", analysis_selection, "process-grant", policy)
    analysis.execute(context, analyze_spec, analysis.validate(context, analyze_spec))

    report = ReportStageAdapter(report_factory)
    report_spec = ReportStep("step", plan, corpus, audit, grants, None, policy)
    report.execute(context, report_spec, report.validate(context, report_spec))

    assert [name for name, _ in calls] == ["search", "filter", "download", "analyze", "report"]
    assert calls[0][1][2]["run_id"] == "workflow-7:step"
    assert calls[1][1]["campaign_id"] == "workflow-7:step"
    assert calls[2][1]["run_id"] == "workflow-7:step"
    assert calls[3][1][0] == "workflow-7:step"
    assert analysis_factory_options == [{
        "workers": 7,
        "allow_abstract_only": False,
        "output_schema_path": ROOT / "schemas" / "paper-analysis.schema.json",
    }]
    assert calls[4][1][:2] == ("workflow-7:step", "workflow-7:step")
    assert calls[4][1][3]["processing_grants"] == {"a" * 64: "grant-7"}
    assert report_factory_args[0][1].root == report_root
    assert report_factory_args[0][3].root == report_root
    assert report_factory_args[0][4] is report_runtime
    assert report_factory_args[0][5] == "unattended"
    assert report_runtime_checks == [({"plan": "frozen"}, "unattended")]
    with Database(context.database_path, read_only=True) as database:
        assert database.connection.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0] > 0


def test_disabled_workflow_report_skips_before_bundle_policy_and_database(
    tmp_path: Path, monkeypatch
) -> None:
    context = _context(tmp_path)
    plan = _write(tmp_path, "plan.json", {"not": "opened"})
    corpus = _write(tmp_path, "corpus.json", {"not": "opened"})
    audit = _write(tmp_path, "audit.json", {"not": "opened"})
    spec = ReportStep("step", plan, corpus, audit, None, None, None)
    runtime = SimpleNamespace(enabled=False)
    monkeypatch.setattr(
        "paper_agent.workflow_adapters.load_config", lambda _path: {}
    )
    monkeypatch.setattr(
        "paper_agent.workflow_adapters._report_runtime_config",
        lambda _config, _path: runtime,
    )
    monkeypatch.setattr(
        "paper_agent.workflow_adapters._report_bundle",
        lambda _spec: pytest.fail("disabled report must not load its bundle"),
    )

    adapter = ReportStageAdapter(
        lambda *_args: pytest.fail("disabled report must not construct a service")
    )
    outcome = adapter.execute(context, spec, adapter.validate(context, spec))

    assert outcome.status == "complete"
    assert outcome.payload["skipped"] is True
    assert not context.database_path.exists()


def test_workflow_report_checks_unattended_pin_before_policy(
    tmp_path: Path, monkeypatch
) -> None:
    context = _context(tmp_path)
    plan = _write(tmp_path, "plan.json", {"plan_hash": "a" * 64})
    corpus = _write(tmp_path, "corpus.json", {})
    audit = _write(tmp_path, "audit.json", {})
    spec = ReportStep("step", plan, corpus, audit, None, None, None)
    checked_modes: list[str] = []

    class PinnedRuntime:
        enabled = True
        resources = object()

        def validate_for_run(self, _plan, *, execution_mode):
            checked_modes.append(execution_mode)
            raise RuntimeError("configured plan pin checked")

    monkeypatch.setattr(
        "paper_agent.workflow_adapters.load_config", lambda _path: {}
    )
    monkeypatch.setattr(
        "paper_agent.workflow_adapters._report_runtime_config",
        lambda _config, _path: PinnedRuntime(),
    )

    adapter = ReportStageAdapter(
        lambda *_args: pytest.fail("pin failure must prevent service construction")
    )
    identity = adapter.validate(context, spec)
    with pytest.raises(RuntimeError, match="plan pin checked"):
        adapter.execute(context, spec, identity)

    assert checked_modes == ["unattended"]
    assert not context.database_path.exists()


def test_orchestrator_dry_run_only_validates_and_observes(tmp_path: Path) -> None:
    context = _context(tmp_path)
    config_source = ROOT / "configs" / "abstract_focus.yaml"
    config = yaml.safe_load(config_source.read_text(encoding="utf-8"))
    config["storage"]["sqlite_path"] = str(context.database_path)
    context.config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    policy = _ref(ROOT / "policies" / "artifact-processing-v1.yaml")
    config = _ref(context.config_path)
    analysis_selection = _write(tmp_path, "analysis.json", {
        "schema_version": "1", "paper_ids": ["paper-1"], "stage3_artifact_ids": [],
    })
    calls: list[str] = []

    def unexpected(*_args, **_kwargs):
        calls.append("executed")
        raise AssertionError("dry run must not execute a stage service")

    manifest = WorkflowManifest(
        "fixture", config,
        (
            AnalyzeStep("analyze", analysis_selection, None, policy),
        ),
        tmp_path / "workflow.json",
    )
    database = Database(context.database_path)
    database.migrate()
    try:
        adapters = {
            StageKind.ANALYZE: AnalyzeStageAdapter(unexpected),
        }
        result = SequentialWorkflowOrchestrator(database, manifest, adapters).run("dry-7", dry_run=True)
    finally:
        database.close()

    assert result["status"] == "validated"
    assert calls == []


def test_analyze_adapter_propagates_terminal_pipeline_failure(tmp_path: Path, monkeypatch) -> None:
    context = _context(tmp_path)
    policy = _ref(ROOT / "policies" / "artifact-processing-v1.yaml")
    selection = _write(tmp_path, "analysis.json", {
        "schema_version": "1", "paper_ids": ["paper-1"], "stage3_artifact_ids": [],
    })
    spec = AnalyzeStep("step", selection, None, policy)
    monkeypatch.setattr(
        "paper_agent.workflow_adapters.load_config",
        lambda _path: {
            "analysis": {
                "workers": 3,
                "allow_abstract_only": True,
                "output_schema": "./schemas/paper-analysis.schema.json",
            },
        },
    )

    class FailedAnalysisService:
        def run(self, run_id: str, manifest: Any, **kwargs: Any) -> SimpleNamespace:
            assert run_id == context.child_run_id
            assert manifest.paper_ids == ("paper-1",)
            assert not kwargs["dry_run"]
            return SimpleNamespace(
                run_id=run_id,
                dry_run=False,
                selected_paper_ids=("paper-1",),
                input_scopes=("metadata_only",),
                result=SimpleNamespace(papers=(SimpleNamespace(status="failed"),)),
            )

    database = Database(context.database_path)
    database.migrate()
    database.connection.execute(
        """INSERT INTO pipeline_runs(run_id, stage, status, input_hash, config_hash, implementation_version)
           VALUES (?, 'stage4', 'failed', 'input', 'config', 'test')""",
        (context.child_run_id,),
    )
    database.connection.commit()
    try:
        adapter = AnalyzeStageAdapter(
            lambda *_args, **_options: FailedAnalysisService()
        )
        outcome = adapter.execute(context, spec, adapter.validate(context, spec))
        assert outcome.status == "uncertain_terminal"
        assert outcome.payload["pipeline_status"] == "failed"
        assert adapter.observe(context, spec, adapter.validate(context, spec)) is StepObservation.UNCERTAIN_TERMINAL
    finally:
        database.close()


def test_dry_run_rejects_invalid_analysis_input_before_constructing_service(tmp_path: Path) -> None:
    context = _context(tmp_path, dry_run=True)
    config = yaml.safe_load((ROOT / "configs" / "abstract_focus.yaml").read_text(encoding="utf-8"))
    context.config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    policy = _ref(ROOT / "policies" / "artifact-processing-v1.yaml")
    selection = _write(tmp_path, "analysis.json", {"schema_version": "1", "paper_ids": []})
    adapter = AnalyzeStageAdapter(lambda *_args: pytest.fail("dry preflight must not construct a service"))
    spec = AnalyzeStep("step", selection, None, policy)

    with pytest.raises(ValueError, match="analysis input manifest"):
        adapter.validate(context, spec)


def test_dry_run_rejects_an_existing_unreadable_workflow_database(tmp_path: Path) -> None:
    context = _context(tmp_path, dry_run=True)
    config = yaml.safe_load((ROOT / "configs" / "abstract_focus.yaml").read_text(encoding="utf-8"))
    context.config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    policy = _ref(ROOT / "policies" / "artifact-processing-v1.yaml")
    selection = _write(tmp_path, "analysis.json", {
        "schema_version": "1", "paper_ids": ["paper-1"], "stage3_artifact_ids": [],
    })
    sqlite3.connect(context.database_path).close()
    before = context.database_path.read_bytes()
    adapter = AnalyzeStageAdapter()
    spec = AnalyzeStep("step", selection, None, policy)

    with pytest.raises(ValueError, match="cannot inspect the existing database state"):
        adapter.observe(context, spec, adapter.validate(context, spec))

    assert context.database_path.read_bytes() == before


def test_workflow_adapters_do_not_reenter_command_or_lease_layers() -> None:
    source = (ROOT / "src" / "paper_agent" / "workflow_adapters.py").read_text(encoding="utf-8")
    assert "from .cli import" not in source
    assert "LeaseQueue" not in source
    assert "argv" not in source


def test_incomplete_child_run_is_safe_to_resume_and_complete_run_is_not_replayed(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    selection = _write(tmp_path, "analysis.json", {
        "schema_version": "1", "paper_ids": ["paper-1"], "stage3_artifact_ids": [],
    })
    policy = _ref(ROOT / "policies" / "artifact-processing-v1.yaml")
    spec = AnalyzeStep("step", selection, None, policy)
    adapter = AnalyzeStageAdapter()
    identity = adapter.validate(context, spec)
    database = Database(context.database_path)
    database.migrate()
    try:
        database.connection.execute(
            """INSERT INTO pipeline_runs(
                   run_id, stage, status, input_hash, config_hash, implementation_version
               ) VALUES (?, 'stage4', 'incomplete', 'input', 'config', 'test')""",
            (context.child_run_id,),
        )
        database.connection.commit()
        assert adapter.observe(context, spec, identity) is StepObservation.SAFE_TO_RESUME
        database.connection.execute(
            "UPDATE pipeline_runs SET status = 'complete' WHERE run_id = ?",
            (context.child_run_id,),
        )
        database.connection.commit()
        assert adapter.observe(context, spec, identity) is StepObservation.COMPLETE
    finally:
        database.close()
