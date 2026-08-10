"""Offline CLI acceptance for the typed recoverable workflow."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from paper_agent import cli
from paper_agent.canonical import canonical_json
from paper_agent.storage import Database
from paper_agent.workflow import (
    AnalyzeStep,
    DownloadStep,
    FileRef,
    FilterStep,
    SearchStep,
    StageKind,
    StepOutputRef,
    WorkflowManifest,
)
from paper_agent.workflow_adapters import (
    AnalyzeStageAdapter,
    DownloadStageAdapter,
    FilterStageAdapter,
    ReportStageAdapter,
    SearchStageAdapter,
)
from test_report_input_service import CONFIG_HASH, HASH, _fixture as report_fixture
from test_report_plan import _draft as report_draft


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
    config["summary"]["enabled"] = True
    config_path = root / "research.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    plan = _json_ref(root, "QUERY_PLAN.json", {"plan_id": "offline-e2e"})
    release = _json_ref(root, "stage2-release.json", {"release": "offline-e2e"})
    manifest = WorkflowManifest(
        "offline-e2e",
        _ref(config_path),
        (
            SearchStep("search", plan, release, (), False),
            FilterStep("filter", plan, release, StepOutputRef("search")),
            DownloadStep("download", StepOutputRef("filter"), None, None, False),
            AnalyzeStep("analyze", StepOutputRef("download"), None, None),
        ),
        root / "workflow.json",
        "2",
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
        with Database(database_path) as database:
            database.connection.execute(
                """INSERT INTO pipeline_runs(
                       run_id, stage, status, input_hash, config_hash, implementation_version
                   ) VALUES (?, 'stage-1', 'complete', 'search-input', 'search-config', 'test')""",
                (options["run_id"],),
            )
            database.connection.commit()
        return (
            SimpleNamespace(status="complete", paper_ids=("paper-e2e",)),
            options["run_id"],
            f"{options['run_id']}:crawl",
        )

    def filter_runner(**options: Any) -> dict[str, Any]:
        calls.append(("filter", options["campaign_id"]))
        stage2_run_id = f"{options['campaign_id']}:stage2"
        with Database(database_path) as database:
            database.connection.execute(
                """INSERT INTO pipeline_runs(
                       run_id, stage, status, input_hash, config_hash, implementation_version
                   ) VALUES (?, 'stage-2', 'complete', 'filter-input', 'filter-config', 'test')""",
                (stage2_run_id,),
            )
            database.connection.commit()
        return {
            "status": "complete",
            "campaign_id": options["campaign_id"],
            "stage2_run_ids": [stage2_run_id],
        }

    class FakeDownloadService:
        def run(self, **options: Any) -> SimpleNamespace:
            nonlocal pause_recovery_download
            run_id = options["run_id"]
            assert options["filter_run_id"] == (
                f"{run_id.replace(':download', ':filter')}:stage2"
            )
            calls.append(("download", run_id))
            if run_id == "recovery:download" and pause_recovery_download:
                pause_recovery_download = False
                status = "incomplete"
            else:
                status = "complete"
            with Database(database_path) as database:
                database.connection.execute(
                    """INSERT INTO pipeline_runs(
                           run_id, stage, status, input_hash, config_hash,
                           implementation_version
                       ) VALUES (?, 'stage-3-download', ?, 'download-input',
                                 'download-config', 'test')
                       ON CONFLICT(run_id) DO UPDATE SET status = excluded.status""",
                    (run_id, status),
                )
                database.connection.commit()
            return SimpleNamespace(
                run_id=run_id,
                paper_ids=("paper-e2e",),
                status=status,
                dry_run=options["dry_run"],
            )

    class FakeAnalysisService:
        def run_from_stage3(
            self, run_id: str, stage3_run_id: str, **options: Any
        ) -> SimpleNamespace:
            calls.append(("analyze", run_id))
            assert stage3_run_id == run_id.replace(":analyze", ":download")
            assert options["expected_paper_ids"] == ("paper-e2e",)
            with Database(database_path) as database:
                database.connection.execute(
                    """INSERT INTO pipeline_runs(
                           run_id, stage, status, input_hash, config_hash,
                           implementation_version
                       ) VALUES (?, 'stage4', 'complete', 'analysis-input',
                                 'analysis-config', 'test')""",
                    (run_id,),
                )
                database.connection.commit()
            return SimpleNamespace(
                run_id=run_id,
                selected_paper_ids=("paper-e2e",),
                input_scopes=("abstract",),
                result=SimpleNamespace(
                    papers=(SimpleNamespace(status="complete"),)
                ),
                dry_run=options["dry_run"],
            )

    adapters = {
        StageKind.SEARCH: SearchStageAdapter(search_runner),
        StageKind.FILTER: FilterStageAdapter(filter_runner),
        StageKind.DOWNLOAD: DownloadStageAdapter(lambda *_args: FakeDownloadService()),
        StageKind.ANALYZE: AnalyzeStageAdapter(
            lambda *_args, **_kwargs: FakeAnalysisService()
        ),
    }
    assert isinstance(adapters[StageKind.SEARCH], SearchStageAdapter)
    assert isinstance(adapters[StageKind.FILTER], FilterStageAdapter)
    assert isinstance(adapters[StageKind.DOWNLOAD], DownloadStageAdapter)
    assert isinstance(adapters[StageKind.ANALYZE], AnalyzeStageAdapter)
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
    assert first["event_code"] == "run.incomplete"
    assert [step["status"] for step in first["steps"]] == [
        "complete", "complete", "incomplete", "pending",
    ]
    assert calls == [
        ("search", "recovery:search"),
        ("filter", "recovery:filter"),
        ("download", "recovery:download"),
    ]

    resumed = invoke("resume", "recovery", 0)
    assert resumed["status"] == "complete"
    assert [step["status"] for step in resumed["steps"]] == ["complete"] * 4
    assert calls == [
        ("search", "recovery:search"),
        ("filter", "recovery:filter"),
        ("download", "recovery:download"),
        ("download", "recovery:download"),
        ("analyze", "recovery:analyze"),
    ]

    clean = invoke("run", "clean", 0)
    assert clean["status"] == "complete"
    assert calls[-4:] == [
        ("search", "clean:search"),
        ("filter", "clean:filter"),
        ("download", "clean:download"),
        ("analyze", "clean:analyze"),
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
                for stage in ("search", "filter", "download", "analyze")
            ]


def test_offline_user_seed_workflow_handoff_and_report_are_end_to_end_resumable(
    tmp_path: Path, monkeypatch, capsys, socket_disabled
) -> None:
    """Run the Phase 7 chain with local fixtures and fake oMLX/Codex only."""
    del socket_disabled
    fixture_database, artifact_store, _ = report_fixture(tmp_path)
    database_path = fixture_database.path
    artifact_root = artifact_store.root
    fixture_database.close()
    analysis_workflow = _workflow_inputs(tmp_path, database_path)
    analysis_run_id = "phase7-analysis"
    stage_calls: list[tuple[str, str]] = []
    omlx_calls: list[str] = []
    omlx_failures: list[str] = []
    analysis_codex_calls: list[str] = []
    download_attempts = 0

    def invoke(arguments: list[str], expected_exit: int = 0) -> dict[str, Any]:
        assert cli.main(arguments) == expected_exit
        return json.loads(capsys.readouterr().out)

    def save_pipeline(
        database: Database,
        run_id: str,
        stage: str,
        status: str,
        input_hash: str,
        config_hash: str,
        version: str,
    ) -> None:
        database.connection.execute(
            """INSERT INTO pipeline_runs(
                   run_id, stage, status, input_hash, config_hash,
                   implementation_version
               ) VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(run_id) DO UPDATE SET status = excluded.status""",
            (run_id, stage, status, input_hash, config_hash, version),
        )
        database.connection.commit()

    def fixture_search(_plan: object, _database: Path, **options: Any):
        run_id = options["run_id"]
        stage_calls.append(("search", run_id))
        with Database(database_path) as database:
            save_pipeline(
                database,
                run_id,
                "stage-1",
                "complete",
                "1" * 64,
                "2" * 64,
                "fixture-search-v1",
            )
            database.connection.execute(
                "UPDATE crawl_runs SET run_id = ? WHERE crawl_run_id = 'crawl-1'",
                (run_id,),
            )
            database.connection.commit()
        paper_ids = ("p1", "p2", "p3", "p4")
        return (
            SimpleNamespace(
                status="complete",
                paper_ids=paper_ids,
                eligible_paper_ids=paper_ids,
            ),
            run_id,
            "crawl-1",
        )

    def fake_omlx(paper_id: str) -> str:
        omlx_calls.append(paper_id)
        if paper_id == "p2":
            omlx_failures.append(paper_id)
            raise RuntimeError("isolated fixture request failure")
        return "relevant" if paper_id == "p1" else "irrelevant"

    def fixture_filter(**options: Any) -> dict[str, Any]:
        stage_calls.append(("filter", options["campaign_id"]))
        decisions: dict[str, str] = {}
        reasons: dict[str, str] = {}
        for paper_id in options["paper_ids"]:
            try:
                decisions[paper_id] = fake_omlx(paper_id)
                reasons[paper_id] = (
                    "topic_match"
                    if decisions[paper_id] == "relevant"
                    else "off_topic"
                )
            except RuntimeError:
                decisions[paper_id] = "needs_review"
                reasons[paper_id] = "mock_omlx_request_failed"
        with Database(database_path) as database:
            for paper_id, status in decisions.items():
                database.connection.execute(
                    """UPDATE filter_decisions
                       SET status = ?, reason = ?
                       WHERE run_id = 'filter-1' AND paper_id = ?""",
                    (status, json.dumps({"reason_code": reasons[paper_id]}), paper_id),
                )
            database.connection.commit()
        return {
            "status": "complete",
            "campaign_id": options["campaign_id"],
            "stage2_run_ids": ["filter-1"],
            "decisions": decisions,
        }

    class FixtureDownloadService:
        def __init__(self, database: Database) -> None:
            self.database = database

        def run(self, **options: Any) -> SimpleNamespace:
            nonlocal download_attempts
            download_attempts += 1
            run_id = options["run_id"]
            stage_calls.append(("download", run_id))
            assert options["filter_run_id"] == "filter-1"
            status = "incomplete" if download_attempts == 1 else "complete"
            save_pipeline(
                self.database,
                run_id,
                "stage-3-download",
                status,
                "3" * 64,
                "4" * 64,
                "fixture-download-v1",
            )
            if status == "complete":
                self.database.connection.execute(
                    """INSERT OR IGNORE INTO download_candidates(
                           candidate_id, paper_id, resolver, url, host, license,
                           access_basis, retrieved_at, provenance_json
                       ) VALUES (
                           'phase7-candidate', 'p1', 'fixture',
                           'https://example.test/p1.pdf', 'example.test',
                           'CC-BY-4.0', 'open_license',
                           '2026-08-11T00:00:00Z', '{"fixture":true}'
                       )"""
                )
                self.database.connection.execute(
                    """INSERT OR IGNORE INTO fetch_requests(
                           request_id, candidate_id, policy_version, policy_hash,
                           purpose, provider, created_at, expires_at,
                           idempotency_key, fencing_token, status
                       ) VALUES (
                           'phase7-request', 'phase7-candidate', 'fixture-v1', ?,
                           'personal_research', 'fixture',
                           '2026-08-11T00:00:00Z', '2026-08-12T00:00:00Z',
                           'phase7-request-key', 1, 'consumed'
                       )""",
                    ("5" * 64,),
                )
                self.database.connection.execute(
                    """INSERT OR IGNORE INTO download_attempts(
                           download_attempt_id, run_id, candidate_id, provider,
                           fetch_request_id, result_status, artifact_id
                       ) VALUES (
                           'phase7-download-attempt', ?, 'phase7-candidate',
                           'fixture', 'phase7-request', 'downloaded', 'pdf-p1'
                       )""",
                    (run_id,),
                )
                self.database.connection.execute(
                    """INSERT OR REPLACE INTO stage3_paper_results(
                           run_id, paper_id, status, reason_code, updated_at
                       ) VALUES (?, 'p1', 'downloaded', 'downloaded',
                                 '2026-08-11T00:00:00Z')""",
                    (run_id,),
                )
                self.database.connection.commit()
            return SimpleNamespace(
                run_id=run_id,
                paper_ids=("p1",),
                status=status,
                dry_run=options["dry_run"],
            )

    class FixtureAnalysisCodex:
        def __init__(self, database: Database) -> None:
            self.database = database

        def run_from_stage3(
            self, run_id: str, stage3_run_id: str, **options: Any
        ) -> SimpleNamespace:
            stage_calls.append(("analyze", run_id))
            analysis_codex_calls.append("p1")
            assert stage3_run_id == f"{analysis_run_id}:download"
            assert options["expected_paper_ids"] == ("p1",)
            save_pipeline(
                self.database,
                run_id,
                "stage4",
                "complete",
                HASH,
                CONFIG_HASH,
                "stage4-v1",
            )
            self.database.connection.execute(
                "UPDATE analysis_runs SET run_id = ? WHERE analysis_run_id = 'analysis-p1'",
                (run_id,),
            )
            self.database.connection.execute(
                "UPDATE analysis_dispatches SET run_id = ? WHERE dispatch_id = 'dispatch-p1'",
                (run_id,),
            )
            self.database.connection.commit()
            return SimpleNamespace(
                run_id=run_id,
                selected_paper_ids=("p1",),
                input_scopes=("full_pdf",),
                result=SimpleNamespace(
                    papers=(SimpleNamespace(status="complete"),)
                ),
                dry_run=options["dry_run"],
            )

    analysis_adapters = {
        StageKind.SEARCH: SearchStageAdapter(fixture_search),
        StageKind.FILTER: FilterStageAdapter(fixture_filter),
        StageKind.DOWNLOAD: DownloadStageAdapter(
            lambda database, *_args: FixtureDownloadService(database)
        ),
        StageKind.ANALYZE: AnalyzeStageAdapter(
            lambda database, *_args, **_kwargs: FixtureAnalysisCodex(database)
        ),
    }
    monkeypatch.setattr(cli, "default_stage_adapters", lambda: analysis_adapters)

    analysis_args = [
        "--config",
        str(tmp_path / "research.yaml"),
        "run",
        "--workflow",
        str(analysis_workflow),
        "--database",
        str(database_path),
        "--workflow-run-id",
        analysis_run_id,
    ]
    first = invoke(analysis_args, 1)
    assert first["status"] == "incomplete"
    assert [step["status"] for step in first["steps"]] == [
        "complete",
        "complete",
        "incomplete",
        "pending",
    ]

    resumed = invoke([
        *analysis_args[:2],
        "resume",
        *analysis_args[3:],
    ])
    assert resumed["status"] == "complete"
    calls_after_completion = tuple(stage_calls)
    replayed = invoke(analysis_args)
    assert replayed["status"] == "complete"
    assert tuple(stage_calls) == calls_after_completion
    assert omlx_calls == ["p1", "p2", "p3", "p4"]
    assert omlx_failures == ["p2"]
    assert analysis_codex_calls == ["p1"]

    release = tmp_path / "release"
    prepared = invoke([
        "report",
        "prepare-inputs",
        "--database",
        str(database_path),
        "--artifact-root",
        str(artifact_root),
        "--output-root",
        str(release),
        "--workflow-run-id",
        analysis_run_id,
        "--recent-cutoff",
        "2024-01-01",
        "--created-at",
        "2026-08-11T00:00:00Z",
    ])
    corpus = json.loads(Path(prepared["corpus_snapshot_path"]).read_text())
    assert [paper["paper_id"] for paper in corpus["papers"]] == ["p1"]
    assert corpus["papers"][0]["source_category"] == "user_library"

    draft = report_draft()
    draft["created_at"] = "2026-08-11T00:01:00Z"
    draft["paper_memberships"] = [draft["paper_memberships"][0]]
    draft_path = tmp_path / "REPORT_DRAFT.json"
    draft_path.write_text(json.dumps(draft), encoding="utf-8")
    planned = invoke([
        "report",
        "--plan-only",
        "--handoff-id",
        prepared["handoff_id"],
        "--draft",
        str(draft_path),
        "--database",
        str(database_path),
        "--artifact-root",
        str(artifact_root),
        "--output-root",
        str(release),
    ])
    approved_path = Path(planned["draft_path"]).with_name("REPORT_PLAN.json")

    report_config = yaml.safe_load(
        (ROOT / "configs" / "abstract_focus.yaml").read_text(encoding="utf-8")
    )
    report_config["project"]["output_dir"] = str(release)
    report_config["storage"]["sqlite_path"] = str(database_path)
    report_config["summary"]["enabled"] = True
    report_config["summary"]["report_plan"]["input_path"] = str(approved_path)
    report_config["summary"]["report_plan"]["content_hash"] = planned["plan_hash"]
    report_config_path = tmp_path / "report-config.yaml"
    report_config_path.write_text(
        yaml.safe_dump(report_config, sort_keys=False), encoding="utf-8"
    )
    report_policy = tmp_path / "report-policy.yaml"
    report_policy.write_bytes(
        (ROOT / "policies" / "artifact-processing-v1.yaml").read_bytes()
    )
    report_workflow = tmp_path / "report-workflow.json"
    approved = invoke([
        "report",
        "approve",
        "--plan",
        planned["draft_path"],
        "--hash",
        planned["plan_hash"],
        "--approved-by",
        "phase7-fixture",
        "--approved-at",
        "2026-08-11T00:02:00Z",
        "--corpus-snapshot",
        prepared["corpus_snapshot_path"],
        "--search-audit",
        prepared["search_audit_path"],
        "--output-root",
        str(release),
        "--handoff-id",
        prepared["handoff_id"],
        "--database",
        str(database_path),
        "--artifact-root",
        str(artifact_root),
        "--workflow-config",
        str(report_config_path),
        "--workflow-manifest",
        str(report_workflow),
        "--workflow-policy",
        str(report_policy),
    ])

    report_codex_calls: list[str] = []
    coverage_path = release / "reports" / "phase7-mock-coverage.json"

    class FixtureReportCodex:
        def __init__(self, database: Database) -> None:
            self.database = database

        def run(
            self,
            report_run_id: str,
            pipeline_run_id: str,
            bundle: Any,
            **options: Any,
        ) -> SimpleNamespace:
            assert report_run_id == pipeline_run_id
            report_codex_calls.append(report_run_id)
            plan_ids = sorted(
                membership["paper_id"]
                for membership in bundle.plan["paper_memberships"]
            )
            corpus_ids = sorted(
                paper["paper_id"] for paper in bundle.corpus_snapshot["papers"]
            )
            assert plan_ids == corpus_ids
            coverage_path.parent.mkdir(parents=True, exist_ok=True)
            coverage_path.write_bytes(canonical_json({
                "paper_ids": plan_ids,
                "covered_paper_ids": corpus_ids,
            }))
            save_pipeline(
                self.database,
                pipeline_run_id,
                "stage4b",
                "complete",
                "6" * 64,
                "7" * 64,
                "fixture-report-v1",
            )
            return SimpleNamespace(
                report_run_id=report_run_id,
                status="complete",
                dry_run=options["dry_run"],
            )

    def report_factory(
        database: Database, runtime_artifacts: Any, *_args: Any
    ) -> FixtureReportCodex:
        assert runtime_artifacts.root.resolve() == artifact_root.resolve()
        return FixtureReportCodex(database)

    monkeypatch.setattr(
        cli,
        "default_stage_adapters",
        lambda: {StageKind.REPORT: ReportStageAdapter(report_factory)},
    )
    report_run_id = approved["report_workflow"]["report_workflow_run_id"]
    report_args = [
        "run",
        "--workflow",
        str(report_workflow),
        "--database",
        str(database_path),
        "--workflow-run-id",
        report_run_id,
    ]
    first_report = invoke(report_args)
    assert first_report["status"] == "complete"
    invoke(["resume", *report_args[1:]])
    invoke(report_args)
    assert report_codex_calls == [f"{report_run_id}:report"]
    assert json.loads(coverage_path.read_text()) == {
        "paper_ids": ["p1"],
        "covered_paper_ids": ["p1"],
    }
    with Database(database_path, read_only=True) as database:
        failed_decision = database.connection.execute(
            """SELECT status, reason FROM filter_decisions
               WHERE run_id = 'filter-1' AND paper_id = 'p2'"""
        ).fetchone()
        assert failed_decision["status"] == "needs_review"
        assert json.loads(failed_decision["reason"])["reason_code"] == (
            "mock_omlx_request_failed"
        )
        assert database.connection.execute(
            "SELECT COUNT(*) FROM download_attempts WHERE run_id = ?",
            (f"{analysis_run_id}:download",),
        ).fetchone()[0] == 1
