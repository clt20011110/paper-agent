from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
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
    StageIdentity,
    StageKind,
    StepContext,
    StepObservation,
    StepOutputRef,
    StepResultRef,
    WorkflowManifest,
)
from paper_agent.workflow_adapters import (
    _report_dispatch_is_live,
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


def _stage3_binding(run_id: str = "workflow-7:download") -> dict[str, str]:
    return {
        "run_id": run_id,
        "stage": "stage-3-download",
        "status": "complete",
        "input_hash": "stage3-input",
        "config_hash": "stage3-config",
        "implementation_version": "stage3-test",
    }


def _analysis_context(
    tmp_path: Path,
    *,
    payload: Mapping[str, Any] | None = None,
) -> StepContext:
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    return StepContext(
        database_path=tmp_path / "papers.sqlite3",
        config_path=config,
        workflow_run_id="workflow-7",
        child_run_id="workflow-7:analyze",
        dry_run=False,
        upstream_results=(
            StepResultRef(
                "download",
                StageKind.DOWNLOAD,
                "workflow-7:download",
                "a" * 64,
                payload
                if payload is not None
                else {
                    "run_id": "workflow-7:download",
                    "paper_ids": ["paper-current"],
                    "_pipeline_binding": _stage3_binding(),
                },
            ),
        ),
    )


def _insert_complete_stage3_binding(database: Database) -> None:
    binding = _stage3_binding()
    database.connection.execute(
        """INSERT INTO pipeline_runs(
               run_id, stage, status, input_hash, config_hash,
               implementation_version
           ) VALUES (?, ?, ?, ?, ?, ?)""",
        (
            binding["run_id"],
            binding["stage"],
            binding["status"],
            binding["input_hash"],
            binding["config_hash"],
            binding["implementation_version"],
        ),
    )
    database.connection.commit()


def _insert_running_stage4(database: Database, run_id: str) -> None:
    database.connection.execute(
        """INSERT INTO pipeline_runs(
               run_id, stage, status, input_hash, config_hash, implementation_version
           ) VALUES (?, 'stage4', 'running', 'input', 'config', 'test')""",
        (run_id,),
    )


def _insert_running_analysis_dispatch(
    database: Database, run_id: str, lease_expires_at: str
) -> None:
    database.connection.execute(
        "INSERT INTO papers(paper_id, title) VALUES ('paper-1', 'Paper')"
    )
    database.connection.execute(
        """INSERT INTO analysis_dispatches(
               dispatch_id, run_id, paper_id, artifact_hash, input_scope,
               config_hash, implementation_version, profile, model_id,
               prompt_hash, schema_hash, policy_version, policy_hash,
               stable_created_at, prompt_input_hash, status, dispatch_count,
               lease_owner, lease_token, lease_expires_at
           ) VALUES (?, ?, 'paper-1', ?, 'abstract_only', ?, 'test',
                     'stage4_analysis_luna', 'gpt-5.6-luna', ?, ?, 'policy-v1', ?,
                     ?, ?, 'running', 1, 'worker-1', 1, ?)""",
        (
            "analysis-dispatch-1",
            run_id,
            "a" * 64,
            "config",
            "b" * 64,
            "c" * 64,
            "d" * 64,
            "2026-08-11T00:00:00.000000Z",
            "e" * 64,
            lease_expires_at,
        ),
    )


def _observe_running_stage4(
    tmp_path: Path, lease_expires_at: str | None
) -> StepObservation:
    context = _context(tmp_path)
    selection = _write(tmp_path, "analysis.json", {
        "schema_version": "1", "paper_ids": ["paper-1"], "stage3_artifact_ids": [],
    })
    policy = _ref(ROOT / "policies" / "artifact-processing-v1.yaml")
    spec = AnalyzeStep("step", selection, None, policy)
    adapter = AnalyzeStageAdapter(
        clock=lambda: datetime(2026, 8, 11, tzinfo=UTC)
    )
    database = Database(context.database_path)
    database.migrate()
    _insert_running_stage4(database, context.child_run_id)
    if lease_expires_at is not None:
        _insert_running_analysis_dispatch(
            database, context.child_run_id, lease_expires_at
        )
    database.connection.commit()
    database.close()
    before = context.database_path.read_bytes()

    observation = adapter.observe(context, spec, adapter.validate(context, spec))

    assert context.database_path.read_bytes() == before
    return observation


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


@pytest.mark.parametrize(
    ("eligible_paper_ids", "include_needs_review"),
    [
        ((), False),
        (("paper-eligible-2", "paper-eligible-1"), True),
    ],
)
def test_v2_workflow_hands_exact_search_and_filter_outputs_to_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    eligible_paper_ids: tuple[str, ...],
    include_needs_review: bool,
) -> None:
    context = _context(tmp_path)
    plan = _write(
        tmp_path,
        "plan.json",
        {"scope": {"include_arxiv_candidates": False}},
    )
    release = _write(tmp_path, "release.json", {})
    config = _ref(context.config_path)
    search = SearchStep("search", plan, release, (), False)
    filtering = FilterStep(
        "filter", plan, release, StepOutputRef("search")
    )
    download = DownloadStep(
        "download",
        StepOutputRef("filter"),
        "download-grant",
        None,
        include_needs_review,
    )
    manifest = WorkflowManifest(
        "lineage",
        config,
        (search, filtering, download),
        tmp_path / "workflow.json",
        "2",
    )
    filter_calls: list[Mapping[str, Any]] = []
    download_calls: list[Mapping[str, Any]] = []

    def search_runner(*_args: Any, **kwargs: Any):
        database.connection.execute(
            """INSERT INTO pipeline_runs(
                   run_id, stage, status, input_hash, config_hash, implementation_version
               ) VALUES (?, 'stage-1', 'complete', 'search-input', 'search-config', 'test')""",
            (kwargs["run_id"],),
        )
        database.connection.commit()
        result = SimpleNamespace(
            status="complete",
            # This must never replace an explicitly empty eligible set.
            paper_ids=("paper-must-not-leak",),
            arxiv_candidate_ids=("arxiv-must-not-leak",),
            eligible_paper_ids=eligible_paper_ids,
        )
        return result, kwargs["run_id"], "crawl-current"

    def filter_runner(**kwargs: Any) -> Mapping[str, Any]:
        filter_calls.append(kwargs)
        database.connection.execute(
            """INSERT INTO pipeline_runs(
                   run_id, stage, status, input_hash, config_hash, implementation_version
               ) VALUES ('stage2-current', 'stage-2', 'complete',
                         'filter-input', 'filter-config', 'test')"""
        )
        database.connection.commit()
        return {
            "status": "complete",
            "campaign_id": kwargs["campaign_id"],
            "stage2_run_ids": ["stage2-current"],
        }

    @dataclass
    class DownloadService:
        def run(self, **kwargs: Any) -> SimpleNamespace:
            download_calls.append(kwargs)
            return SimpleNamespace(
                run_id=kwargs["run_id"],
                paper_ids=(),
                status="complete",
                dry_run=kwargs["dry_run"],
            )

    monkeypatch.setattr("paper_agent.workflow_adapters.load_config", lambda _path: {})
    database = Database(context.database_path)
    database.migrate()
    try:
        result = SequentialWorkflowOrchestrator(
            database,
            manifest,
            {
                StageKind.SEARCH: SearchStageAdapter(search_runner),
                StageKind.FILTER: FilterStageAdapter(filter_runner),
                StageKind.DOWNLOAD: DownloadStageAdapter(
                    lambda *_args: DownloadService()
                ),
            },
        ).run("workflow-lineage")
    finally:
        database.close()

    assert result["status"] == "complete"
    assert len(filter_calls) == 1
    assert filter_calls[0]["paper_ids"] == eligible_paper_ids
    assert filter_calls[0]["paper_ids"] is not None
    assert len(download_calls) == 1
    assert download_calls[0]["paper_ids"] == ()
    assert download_calls[0]["filter_run_id"] == "stage2-current"
    assert download_calls[0]["include_needs_review"] is include_needs_review


def test_v2_workflow_analyze_uses_the_exact_current_download_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    database_path = tmp_path / "papers.sqlite3"
    download_context = StepContext(
        database_path=database_path,
        config_path=config,
        workflow_run_id="workflow-7",
        child_run_id="workflow-7:download",
        dry_run=False,
    )
    selection = _write(
        tmp_path,
        "download-selection.json",
        {"schema_version": "1", "paper_ids": ["paper-current"]},
    )
    policy = _ref(ROOT / "policies" / "artifact-processing-v1.yaml")
    analysis_calls: list[tuple[str, str, Mapping[str, Any]]] = []

    analysis_config = {
        "analysis": {
            "workers": 8,
            "allow_abstract_only": True,
            "output_schema": "./schemas/paper-analysis.schema.json",
        },
    }
    monkeypatch.setattr(
        "paper_agent.workflow_adapters.load_config",
        lambda _path: analysis_config,
    )

    @dataclass
    class DownloadService:
        database: Database

        def run(self, **kwargs: Any) -> SimpleNamespace:
            assert kwargs["run_id"] == "workflow-7:download"
            binding = _stage3_binding(kwargs["run_id"])
            self.database.connection.execute(
                """INSERT INTO pipeline_runs(
                       run_id, stage, status, input_hash, config_hash,
                       implementation_version
                   ) VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    binding["run_id"],
                    binding["stage"],
                    binding["status"],
                    binding["input_hash"],
                    binding["config_hash"],
                    binding["implementation_version"],
                ),
            )
            self.database.connection.commit()
            return SimpleNamespace(
                run_id=kwargs["run_id"],
                paper_ids=("paper-current",),
                status="complete",
                dry_run=False,
            )

    download_adapter = DownloadStageAdapter(
        lambda database, *_args: DownloadService(database)
    )
    download_spec = DownloadStep("download", selection, None, None, False)
    download_outcome = download_adapter.execute(
        download_context,
        download_spec,
        download_adapter.validate(download_context, download_spec),
    )
    assert download_outcome.payload["_pipeline_binding"] == _stage3_binding()

    monkeypatch.setattr(
        "paper_agent.workflow_adapters.load_analysis_input_manifest",
        lambda *_args: pytest.fail(
            "dynamic analysis must not load a stale selection FileRef"
        ),
    )

    @dataclass
    class AnalysisService:
        def run(self, *_args: Any, **_kwargs: Any) -> None:
            pytest.fail("dynamic analysis must not call the static manifest API")

        def run_from_stage3(
            self,
            run_id: str,
            stage3_run_id: str,
            **kwargs: Any,
        ) -> SimpleNamespace:
            analysis_calls.append((run_id, stage3_run_id, kwargs))
            return SimpleNamespace(
                run_id=run_id,
                dry_run=False,
                selected_paper_ids=("paper-current",),
                input_scopes=("full_text",),
                result=SimpleNamespace(
                    papers=(SimpleNamespace(status="complete"),)
                ),
            )

    analysis_context = _analysis_context(
        tmp_path, payload=download_outcome.payload
    )
    analysis_adapter = AnalyzeStageAdapter(
        lambda *_args, **_kwargs: AnalysisService()
    )
    analysis_spec = AnalyzeStep(
        "analyze", StepOutputRef("download"), "processing-grant", policy
    )
    outcome = analysis_adapter.execute(
        analysis_context,
        analysis_spec,
        analysis_adapter.validate(analysis_context, analysis_spec),
    )

    assert outcome.status == "complete"
    assert analysis_calls == [
        (
            "workflow-7:analyze",
            "workflow-7:download",
            {
                "expected_paper_ids": ("paper-current",),
                "processing_grant_id": "processing-grant",
                "dry_run": False,
            },
        )
    ]


@pytest.mark.parametrize(
    ("payload", "insert_current"),
    [
        (
            {
                "run_id": "workflow-7:download",
                "paper_ids": ["paper-current"],
                "_pipeline_binding": {
                    **_stage3_binding(),
                    "input_hash": "drifted-input",
                },
            },
            True,
        ),
        (
            {
                "run_id": "workflow-7:download",
                "paper_ids": ["paper-current"],
            },
            True,
        ),
        (
            {
                "run_id": "workflow-7:download",
                "paper_ids": ["paper-current"],
                "_pipeline_binding": _stage3_binding(),
            },
            False,
        ),
    ],
    ids=("binding-drifted", "recorded-binding-missing", "current-run-missing"),
)
def test_v2_workflow_analyze_rejects_missing_or_drifted_download_binding_before_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: Mapping[str, Any],
    insert_current: bool,
) -> None:
    context = _analysis_context(tmp_path, payload=payload)
    database = Database(context.database_path)
    database.migrate()
    try:
        if insert_current:
            _insert_complete_stage3_binding(database)
    finally:
        database.close()
    monkeypatch.setattr(
        "paper_agent.workflow_adapters.load_config",
        lambda _path: {
            "analysis": {
                "workers": 8,
                "allow_abstract_only": True,
                "output_schema": "./schemas/paper-analysis.schema.json",
            },
        },
    )
    service_calls: list[str] = []

    @dataclass
    class AnalysisService:
        def run(self, *_args: Any, **_kwargs: Any) -> None:
            service_calls.append("run")

        def run_from_stage3(self, *_args: Any, **_kwargs: Any) -> None:
            service_calls.append("run_from_stage3")

    adapter = AnalyzeStageAdapter(
        lambda *_args, **_kwargs: AnalysisService()
    )
    policy = _ref(ROOT / "policies" / "artifact-processing-v1.yaml")
    spec = AnalyzeStep(
        "analyze", StepOutputRef("download"), None, policy
    )

    with pytest.raises(ValueError, match="binding"):
        adapter.execute(context, spec, adapter.validate(context, spec))

    assert service_calls == []


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


def test_running_stage4_with_live_dispatch_is_observed_as_running(tmp_path: Path) -> None:
    assert (
        _observe_running_stage4(tmp_path, "2026-08-11T00:05:00.000000Z")
        is StepObservation.RUNNING
    )


def test_running_stage4_with_expired_dispatch_is_safe_to_resume(tmp_path: Path) -> None:
    assert (
        _observe_running_stage4(tmp_path, "2026-08-11T00:00:00.000000Z")
        is StepObservation.SAFE_TO_RESUME
    )


def test_running_stage4_without_dispatch_is_safe_to_resume(tmp_path: Path) -> None:
    assert _observe_running_stage4(tmp_path, None) is StepObservation.SAFE_TO_RESUME


@pytest.mark.parametrize(
    "table",
    ("report_reduce_nodes", "report_audit_steps", "report_audit_shard_steps"),
)
def test_report_live_dispatch_detection_covers_every_stage4b_queue(
    table: str,
) -> None:
    connection = sqlite3.connect(":memory:")
    try:
        for name in (
            "report_reduce_nodes",
            "report_audit_steps",
            "report_audit_shard_steps",
        ):
            connection.execute(
                f"""CREATE TABLE {name}(
                    report_run_id TEXT, status TEXT, lease_expires_at TEXT
                )"""
            )
        connection.execute(
            f"""INSERT INTO {table}(report_run_id, status, lease_expires_at)
                VALUES ('workflow-7:step', 'running',
                        '2026-08-11T00:05:00.000000Z')"""
        )

        assert _report_dispatch_is_live(
            connection,
            "workflow-7:step",
            "2026-08-11T00:00:00.000000Z",
        )
        assert not _report_dispatch_is_live(
            connection,
            "workflow-7:step",
            "2026-08-11T00:10:00.000000Z",
        )
    finally:
        connection.close()


@pytest.mark.parametrize(
    ("live", "expected"),
    ((True, StepObservation.RUNNING), (False, StepObservation.SAFE_TO_RESUME)),
)
def test_running_stage4b_observation_uses_child_dispatch_leases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    live: bool,
    expected: StepObservation,
) -> None:
    context = _context(tmp_path)
    plan = _write(tmp_path, "report-plan.json", {})
    corpus = _write(tmp_path, "corpus.json", {})
    audit = _write(tmp_path, "audit.json", {})
    spec = ReportStep("step", plan, corpus, audit, None, None, None)
    database = Database(context.database_path)
    database.migrate()
    database.connection.execute(
        """INSERT INTO pipeline_runs(
               run_id, stage, status, input_hash, config_hash,
               implementation_version
           ) VALUES (?, 'stage4b', 'running', 'input', 'config', 'test')""",
        (context.child_run_id,),
    )
    database.connection.commit()
    database.close()
    monkeypatch.setattr(
        "paper_agent.workflow_adapters._report_dispatch_is_live",
        lambda *_args: live,
    )
    adapter = ReportStageAdapter(
        clock=lambda: datetime(2026, 8, 11, tzinfo=UTC)
    )

    assert adapter.observe(context, spec, adapter.validate(context, spec)) is expected


def test_incomplete_and_uncheckpointed_complete_child_runs_are_safe_to_reconcile(
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
        assert adapter.observe(context, spec, identity) is StepObservation.SAFE_TO_RESUME
    finally:
        database.close()
