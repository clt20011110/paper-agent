from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace
from typing import Any, Mapping

import pytest
import yaml

import paper_agent.workflow_report_handoff as handoff_module
from paper_agent import cli
from paper_agent.artifacts import ArtifactStore
from paper_agent.canonical import content_hash
from paper_agent.report_input_service import ReportInputService
from paper_agent.report_cli_service import approve_report_plan_from_files
from paper_agent.report_config import ReportResources
from paper_agent.report_plan import ReportPlanBundle
from paper_agent.storage import Database
from paper_agent.workflow import (
    AnalyzeStep,
    DownloadStep,
    FileRef,
    FilterStep,
    SearchStep,
    SequentialWorkflowOrchestrator,
    StageIdentity,
    StageKind,
    StageOutcome,
    StepObservation,
    StepOutputRef,
    WorkflowManifest,
    load_workflow_manifest,
)
from paper_agent.workflow_report_handoff import (
    WorkflowReportHandoffError,
    WorkflowReportExecutionRequest,
    WorkflowReportHandoffRequest,
    WorkflowReportHandoffService,
)
from paper_agent.workflow_adapters import ReportStageAdapter
from test_report_input_service import CONFIG_HASH, HASH, _fixture
from test_report_plan import _draft


WORKFLOW_RUN_ID = "workflow-report-source"
SEARCH_RUN_ID = f"{WORKFLOW_RUN_ID}:search"
DOWNLOAD_RUN_ID = f"{WORKFLOW_RUN_ID}:download"
STAGE4_RUN_ID = f"{WORKFLOW_RUN_ID}:analyze"
ROOT = Path(__file__).resolve().parents[1]


def _configure_reduce_tree_summary(config: dict[str, Any]) -> None:
    """Keep legacy reducer workflow fixtures explicit under one-shot defaults."""
    summary = config["summary"]
    summary.update({
        "enabled": True,
        "execution_strategy": "reduce_tree",
        "profile": "stage4b_summary_sol",
        "semantic_chunking": True,
    })
    summary["final_audit"].update({
        "independent_sol_session": True,
        "max_repair_calls": 1,
        "reverify_and_reaudit_after_repair": True,
    })


class _PersistedOutcomeAdapter:
    def __init__(self, outcomes: Mapping[StageKind, StageOutcome]) -> None:
        self.outcomes = outcomes
        self.executions: list[StageKind] = []

    def validate(self, context: Any, spec: Any) -> StageIdentity:
        return StageIdentity(
            content_hash(
                {
                    "child_run_id": context.child_run_id,
                    "spec": spec.document(),
                }
            )
        )

    def observe(
        self, context: Any, spec: Any, identity: StageIdentity
    ) -> StepObservation:
        del context, spec, identity
        return StepObservation.PENDING

    def execute(
        self, context: Any, spec: Any, identity: StageIdentity
    ) -> StageOutcome:
        del context, identity
        self.executions.append(spec.stage)
        return self.outcomes[spec.stage]


def _pipeline_binding(database: Any, run_id: str) -> dict[str, str]:
    row = database.connection.execute(
        """SELECT stage, status, input_hash, config_hash, implementation_version
           FROM pipeline_runs WHERE run_id = ?""",
        (run_id,),
    ).fetchone()
    assert row is not None
    return {
        "run_id": run_id,
        "stage": str(row["stage"]),
        "status": str(row["status"]),
        "input_hash": str(row["input_hash"]),
        "config_hash": str(row["config_hash"]),
        "implementation_version": str(row["implementation_version"]),
    }


def _complete_analysis_workflow(tmp_path: Path):
    database, artifact_store, _ = _fixture(tmp_path)
    database.connection.executemany(
        """INSERT INTO pipeline_runs(
               run_id, stage, status, input_hash, config_hash,
               implementation_version
           ) VALUES (?, ?, 'complete', ?, ?, ?)""",
        (
            (SEARCH_RUN_ID, "stage-1", "1" * 64, "2" * 64, "search-v1"),
            (
                DOWNLOAD_RUN_ID,
                "stage-3-download",
                "5" * 64,
                "6" * 64,
                "stage3-v1",
            ),
            (STAGE4_RUN_ID, "stage4", HASH, CONFIG_HASH, "stage4-v1"),
        ),
    )
    database.connection.execute(
        "UPDATE crawl_runs SET run_id = ? WHERE crawl_run_id = 'crawl-1'",
        (SEARCH_RUN_ID,),
    )
    database.connection.execute(
        """INSERT INTO stage3_paper_results(
               run_id, paper_id, status, reason_code, updated_at
           ) VALUES (?, 'p1', 'downloaded', 'downloaded',
                     '2026-08-11T00:00:00Z')""",
        (DOWNLOAD_RUN_ID,),
    )
    database.connection.execute(
        "UPDATE analysis_runs SET run_id = ? WHERE run_id = 'stage4-1'",
        (STAGE4_RUN_ID,),
    )
    database.connection.execute(
        "UPDATE analysis_dispatches SET run_id = ? WHERE run_id = 'stage4-1'",
        (STAGE4_RUN_ID,),
    )
    database.connection.execute(
        """UPDATE filter_decisions
           SET status = 'irrelevant', reason = '{"reason_code":"off_topic"}'
           WHERE run_id = 'filter-1' AND paper_id = 'p2'"""
    )
    database.connection.commit()

    frozen = tmp_path / "workflow-input.json"
    frozen.write_text("{}", encoding="utf-8")
    reference = FileRef(
        frozen.name,
        sha256(frozen.read_bytes()).hexdigest(),
        frozen,
    )
    search = SearchStep("search", reference, reference, (), False)
    filtering = FilterStep(
        "filter", reference, reference, StepOutputRef("search")
    )
    download = DownloadStep(
        "download", StepOutputRef("filter"), None, None, False
    )
    analyze = AnalyzeStep(
        "analyze", StepOutputRef("download"), None, None
    )
    manifest = WorkflowManifest(
        "analysis-workflow",
        reference,
        (search, filtering, download, analyze),
        tmp_path / "workflow.json",
        "2",
    )
    outcomes = {
        StageKind.SEARCH: StageOutcome(
            "complete",
            {
                "run_id": SEARCH_RUN_ID,
                "crawl_run_id": "crawl-1",
                "paper_ids": ["p1", "p2", "p3", "p4"],
                "eligible_paper_ids": ["p1", "p2", "p3", "p4"],
                "_pipeline_binding": _pipeline_binding(database, SEARCH_RUN_ID),
            },
        ),
        StageKind.FILTER: StageOutcome(
            "complete",
            {
                "status": "complete",
                "stage2_run_ids": ["filter-1"],
                "decisions": {
                    "p1": "relevant",
                    "p2": "irrelevant",
                    "p3": "irrelevant",
                    "p4": "needs_review",
                },
                "_pipeline_binding": _pipeline_binding(database, "filter-1"),
            },
        ),
        StageKind.DOWNLOAD: StageOutcome(
            "complete",
            {
                "run_id": DOWNLOAD_RUN_ID,
                "paper_ids": ["p1"],
                "stage_status": "complete",
                "_pipeline_binding": _pipeline_binding(database, DOWNLOAD_RUN_ID),
            },
        ),
        StageKind.ANALYZE: StageOutcome(
            "complete",
            {
                "run_id": STAGE4_RUN_ID,
                "paper_ids": ["p1"],
                "stage_status": "complete",
                "_pipeline_binding": _pipeline_binding(database, STAGE4_RUN_ID),
            },
        ),
    }
    adapter = _PersistedOutcomeAdapter(outcomes)
    orchestrator = SequentialWorkflowOrchestrator(
        database,
        manifest,
        {stage: adapter for stage in StageKind if stage is not StageKind.REPORT},
    )
    result = orchestrator.run(WORKFLOW_RUN_ID)
    assert result["status"] == "complete"
    assert adapter.executions == [
        StageKind.SEARCH,
        StageKind.FILTER,
        StageKind.DOWNLOAD,
        StageKind.ANALYZE,
    ]
    return database, artifact_store, adapter


def _request() -> WorkflowReportHandoffRequest:
    return WorkflowReportHandoffRequest(
        workflow_run_id=WORKFLOW_RUN_ID,
        recent_cutoff="2024-01-01",
        created_at="2026-08-11T00:00:00Z",
    )


def _report_draft(tmp_path: Path) -> Path:
    draft = _draft()
    draft["created_at"] = "2026-08-11T00:01:00Z"
    draft["paper_memberships"] = [draft["paper_memberships"][0]]
    path = tmp_path / "report-draft.json"
    path.write_text(json.dumps(draft), encoding="utf-8")
    return path


def test_completed_workflow_persists_exact_hash_bound_report_inputs(
    tmp_path: Path,
) -> None:
    database, artifact_store, adapter = _complete_analysis_workflow(tmp_path)
    try:
        result = WorkflowReportHandoffService(
            database, artifact_store, tmp_path / "release"
        ).prepare(_request())

        assert result.status == "complete"
        assert result.persisted is True
        assert result.resumed is False
        assert result.write_performed is True
        assert result.crawl_run_id == "crawl-1"
        assert result.filter_run_id == "filter-1"
        assert result.download_run_id == DOWNLOAD_RUN_ID
        assert result.stage4_run_id == STAGE4_RUN_ID
        assert result.include_needs_review is False
        assert result.report_inputs.corpus_snapshot_path.is_file()
        assert result.report_inputs.search_audit_path.is_file()
        assert result.document()["corpus_snapshot_hash"] == (
            result.report_inputs.corpus_snapshot["snapshot_hash"]
        )
        row = database.connection.execute(
            "SELECT * FROM workflow_report_handoffs WHERE handoff_id = ?",
            (result.handoff_id,),
        ).fetchone()
        assert row is not None
        assert row["status"] == "complete"
        assert row["workflow_binding_hash"] == result.workflow_binding_hash
        assert row["bundle_hash"] == result.bundle_hash
        assert (
            row["corpus_file_sha256"]
            == result.document()["corpus_snapshot_file_sha256"]
        )
        assert (
            row["search_audit_file_sha256"]
            == result.document()["search_audit_file_sha256"]
        )
        assert adapter.executions == [
            StageKind.SEARCH,
            StageKind.FILTER,
            StageKind.DOWNLOAD,
            StageKind.ANALYZE,
        ]
    finally:
        database.close()


def test_handoff_dry_run_validates_without_database_or_filesystem_writes(
    tmp_path: Path,
) -> None:
    database, artifact_store, adapter = _complete_analysis_workflow(tmp_path)
    try:
        result = WorkflowReportHandoffService(
            database, artifact_store, tmp_path / "dry-release"
        ).prepare(_request(), save_bundle=False)

        assert result.status == "validated"
        assert result.persisted is False
        assert result.write_performed is False
        assert not result.report_inputs.directory.exists()
        assert database.connection.execute(
            "SELECT COUNT(*) FROM workflow_report_handoffs"
        ).fetchone()[0] == 0
        assert len(adapter.executions) == 4
    finally:
        database.close()


def test_complete_handoff_resume_does_not_rebuild_inputs_or_rerun_workflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database, artifact_store, adapter = _complete_analysis_workflow(tmp_path)
    try:
        service = WorkflowReportHandoffService(
            database, artifact_store, tmp_path / "release"
        )
        first = service.prepare(_request())

        def unexpected_build(*_args: Any, **_kwargs: Any) -> None:
            pytest.fail("a complete handoff must load its frozen input bundle")

        monkeypatch.setattr(ReportInputService, "build", unexpected_build)
        resumed = service.prepare(_request())

        assert resumed.handoff_id == first.handoff_id
        assert resumed.bundle_hash == first.bundle_hash
        assert resumed.resumed is True
        assert resumed.write_performed is False
        assert len(adapter.executions) == 4
    finally:
        database.close()


def test_interrupted_handoff_resumes_preparing_checkpoint_without_workflow_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database, artifact_store, adapter = _complete_analysis_workflow(tmp_path)
    original_build = ReportInputService.build

    def interrupt_after_write(
        service: ReportInputService, *args: Any, **kwargs: Any
    ) -> Any:
        original_build(service, *args, **kwargs)
        raise RuntimeError("simulated process loss")

    try:
        service = WorkflowReportHandoffService(
            database, artifact_store, tmp_path / "release"
        )
        monkeypatch.setattr(ReportInputService, "build", interrupt_after_write)
        with pytest.raises(RuntimeError, match="simulated process loss"):
            service.prepare(_request())
        row = database.connection.execute(
            "SELECT status FROM workflow_report_handoffs"
        ).fetchone()
        assert row["status"] == "preparing"

        monkeypatch.setattr(ReportInputService, "build", original_build)
        resumed = service.prepare(_request())

        assert resumed.status == "complete"
        assert resumed.resumed is True
        assert resumed.write_performed is True
        assert resumed.report_inputs.saved is False
        assert len(adapter.executions) == 4
    finally:
        database.close()


def test_completed_handoff_rejects_report_input_file_drift(tmp_path: Path) -> None:
    database, artifact_store, _ = _complete_analysis_workflow(tmp_path)
    try:
        service = WorkflowReportHandoffService(
            database, artifact_store, tmp_path / "release"
        )
        result = service.prepare(_request())
        result.report_inputs.corpus_snapshot_path.write_text("{}", encoding="utf-8")

        with pytest.raises(WorkflowReportHandoffError, match="file hash has drifted"):
            service.prepare(_request())
    finally:
        database.close()


def test_handoff_rejects_tampered_child_run_payload(tmp_path: Path) -> None:
    database, artifact_store, _ = _complete_analysis_workflow(tmp_path)
    try:
        row = database.connection.execute(
            """SELECT result_json FROM workflow_steps
               WHERE workflow_run_id = ? AND stage = 'analyze'""",
            (WORKFLOW_RUN_ID,),
        ).fetchone()
        payload = json.loads(row["result_json"])
        payload["run_id"] = "filter-1"
        database.connection.execute(
            """UPDATE workflow_steps SET result_json = ?
               WHERE workflow_run_id = ? AND stage = 'analyze'""",
            (json.dumps(payload), WORKFLOW_RUN_ID),
        )
        database.connection.commit()

        with pytest.raises(WorkflowReportHandoffError, match="Stage 4 run ID has drifted"):
            WorkflowReportHandoffService(
                database, artifact_store, tmp_path / "release"
            ).prepare(_request())
        assert not (tmp_path / "release" / "reports" / "inputs").exists()
    finally:
        database.close()


def test_handoff_intentionally_rejects_an_incomplete_stage4_pipeline(
    tmp_path: Path,
) -> None:
    database, artifact_store, _ = _complete_analysis_workflow(tmp_path)
    try:
        database.connection.execute(
            "UPDATE pipeline_runs SET status = 'incomplete' WHERE run_id = ?",
            (STAGE4_RUN_ID,),
        )
        database.connection.commit()

        with pytest.raises(
            WorkflowReportHandoffError, match="not a complete stage4 run"
        ):
            WorkflowReportHandoffService(
                database, artifact_store, tmp_path / "release"
            ).prepare(_request())
    finally:
        database.close()


def test_handoff_rejects_filter_decision_drift_before_freezing_inputs(
    tmp_path: Path,
) -> None:
    database, artifact_store, _ = _complete_analysis_workflow(tmp_path)
    try:
        database.connection.execute(
            """UPDATE filter_decisions
               SET status = 'relevant', reason = '{"reason_code":"include"}'
               WHERE run_id = 'filter-1' AND paper_id = 'p2'"""
        )
        database.connection.commit()

        with pytest.raises(
            WorkflowReportHandoffError, match="Filter decisions have drifted"
        ):
            WorkflowReportHandoffService(
                database, artifact_store, tmp_path / "release"
            ).prepare(_request())
        assert not (tmp_path / "release" / "reports" / "inputs").exists()
    finally:
        database.close()


def test_handoff_rejects_download_and_analyze_payload_selection_drift(
    tmp_path: Path,
) -> None:
    database, artifact_store, _ = _complete_analysis_workflow(tmp_path)
    try:
        for stage in ("download", "analyze"):
            row = database.connection.execute(
                """SELECT result_json FROM workflow_steps
                   WHERE workflow_run_id = ? AND stage = ?""",
                (WORKFLOW_RUN_ID, stage),
            ).fetchone()
            payload = json.loads(row["result_json"])
            payload["paper_ids"] = ["foreign-paper"]
            database.connection.execute(
                """UPDATE workflow_steps SET result_json = ?
                   WHERE workflow_run_id = ? AND stage = ?""",
                (json.dumps(payload), WORKFLOW_RUN_ID, stage),
            )
        database.connection.commit()

        with pytest.raises(
            WorkflowReportHandoffError, match="Download papers have drifted"
        ):
            WorkflowReportHandoffService(
                database, artifact_store, tmp_path / "release"
            ).prepare(_request())
    finally:
        database.close()


def test_handoff_requires_exact_terminal_stage3_database_corpus(
    tmp_path: Path,
) -> None:
    database, artifact_store, _ = _complete_analysis_workflow(tmp_path)
    try:
        database.connection.execute(
            "DELETE FROM stage3_paper_results WHERE run_id = ?",
            (DOWNLOAD_RUN_ID,),
        )
        database.connection.commit()

        with pytest.raises(
            WorkflowReportHandoffError, match="Stage 3 paper checkpoints"
        ):
            WorkflowReportHandoffService(
                database, artifact_store, tmp_path / "release"
            ).prepare(_request())
    finally:
        database.close()


def test_handoff_requires_exact_complete_stage4_lineage(tmp_path: Path) -> None:
    database, artifact_store, _ = _complete_analysis_workflow(tmp_path)
    try:
        database.connection.execute(
            """UPDATE analysis_dispatches SET prompt_input_hash = ?
               WHERE run_id = ? AND paper_id = 'p1'""",
            ("f" * 64, STAGE4_RUN_ID),
        )
        database.connection.commit()

        with pytest.raises(
            WorkflowReportHandoffError, match="analysis lineage has drifted"
        ):
            WorkflowReportHandoffService(
                database, artifact_store, tmp_path / "release"
            ).prepare(_request())
    finally:
        database.close()


def test_handoff_result_filerefs_fail_closed_after_file_loss(tmp_path: Path) -> None:
    database, artifact_store, _ = _complete_analysis_workflow(tmp_path)
    try:
        result = WorkflowReportHandoffService(
            database, artifact_store, tmp_path / "release"
        ).prepare(_request())
        result.report_inputs.corpus_snapshot_path.unlink()

        with pytest.raises(WorkflowReportHandoffError, match="unavailable"):
            result.document()
    finally:
        database.close()


def test_handoff_compiles_approves_and_resumes_one_independent_report_workflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database, artifact_store, analysis_adapter = _complete_analysis_workflow(tmp_path)
    release = tmp_path / "release"
    service = WorkflowReportHandoffService(database, artifact_store, release)
    report_calls: list[str] = []
    try:
        handoff = service.prepare(_request())
        compiled = service.compile_plan(handoff.handoff_id, _report_draft(tmp_path))
        assert compiled.plan["workflow_handoff"] == {
            "handoff_id": handoff.handoff_id,
            "workflow_binding_hash": handoff.workflow_binding_hash,
            "bundle_hash": handoff.bundle_hash,
        }
        approved = approve_report_plan_from_files(
            compiled.path,
            handoff.report_inputs.corpus_snapshot_path,
            handoff.report_inputs.search_audit_path,
            release,
            expected_hash=str(compiled.plan["plan_hash"]),
            approved_by="owner",
            approved_at="2026-08-11T00:02:00Z",
        )
        config_path = tmp_path / "report-config.yaml"
        config = yaml.safe_load(
            (ROOT / "configs" / "abstract_focus.yaml").read_text(encoding="utf-8")
        )
        config["project"]["output_dir"] = str(release)
        config["storage"]["sqlite_path"] = str(database.path)
        _configure_reduce_tree_summary(config)
        config["summary"]["report_plan"]["input_path"] = str(approved.path)
        config["summary"]["report_plan"]["content_hash"] = approved.plan["plan_hash"]
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        policy_path = tmp_path / "report-policy.yaml"
        policy_path.write_bytes(
            (ROOT / "policies" / "artifact-processing-v1.yaml").read_bytes()
        )
        execution_request = WorkflowReportExecutionRequest(
            handoff.handoff_id,
            approved.path,
            config_path,
            tmp_path / "report-workflow.json",
            policy_path=policy_path,
        )
        prepared = service.prepare_report_workflow(execution_request)
        resumed_binding = service.prepare_report_workflow(execution_request)

        assert prepared.write_performed is True
        assert resumed_binding.resumed is True
        assert resumed_binding.write_performed is False
        assert database.connection.execute(
            "SELECT COUNT(*) FROM workflow_report_executions"
        ).fetchone()[0] == 1
        assert len(prepared.manifest.steps) == 1
        assert prepared.manifest.steps[0].stage is StageKind.REPORT

        runtime = SimpleNamespace(
            enabled=True,
            resources=ReportResources.defaults(),
            validate_for_run=lambda _plan, *, execution_mode: None,
        )
        monkeypatch.setattr(
            "paper_agent.workflow_adapters.load_config",
            lambda _path: {"project": {"output_dir": str(release)}},
        )
        monkeypatch.setattr(
            "paper_agent.workflow_adapters._report_runtime_config",
            lambda _config, _path: runtime,
        )

        class ReportService:
            def run(
                self,
                report_run_id: str,
                pipeline_run_id: str,
                bundle: Any,
                **kwargs: Any,
            ) -> SimpleNamespace:
                assert report_run_id == pipeline_run_id
                assert bundle.plan["workflow_handoff"]["handoff_id"] == handoff.handoff_id
                report_calls.append(report_run_id)
                return SimpleNamespace(
                    report_run_id=report_run_id,
                    status="complete",
                    dry_run=kwargs["dry_run"],
                )

        adapter = ReportStageAdapter(lambda *_args: ReportService())
        first = service.run_report_workflow(
            handoff.handoff_id, adapter=adapter
        )
        resumed = service.run_report_workflow(
            handoff.handoff_id, resume=True, adapter=adapter
        )

        assert first["status"] == "complete"
        assert resumed["status"] == "complete"
        assert report_calls == [f"{prepared.report_workflow_run_id}:report"]
        assert analysis_adapter.executions == [
            StageKind.SEARCH,
            StageKind.FILTER,
            StageKind.DOWNLOAD,
            StageKind.ANALYZE,
        ]

        row = database.connection.execute(
            """SELECT result_json FROM workflow_steps
               WHERE workflow_run_id = ? AND stage = 'analyze'""",
            (WORKFLOW_RUN_ID,),
        ).fetchone()
        changed = json.loads(row["result_json"])
        changed["unexpected_after_handoff"] = True
        database.connection.execute(
            """UPDATE workflow_steps SET result_json = ?
               WHERE workflow_run_id = ? AND stage = 'analyze'""",
            (json.dumps(changed), WORKFLOW_RUN_ID),
        )
        database.connection.commit()
        with pytest.raises(WorkflowReportHandoffError, match="binding has drifted"):
            service.run_report_workflow(handoff.handoff_id, resume=True, adapter=adapter)
        assert report_calls == [f"{prepared.report_workflow_run_id}:report"]
    finally:
        database.close()


def test_parallel_prepare_converges_on_one_complete_handoff(tmp_path: Path) -> None:
    database, artifact_store, _ = _complete_analysis_workflow(tmp_path)
    database_path = database.path
    artifact_root = artifact_store.root
    database.close()
    barrier = Barrier(2)

    def prepare() -> Any:
        with Database(database_path) as connection:
            barrier.wait()
            return WorkflowReportHandoffService(
                connection,
                ArtifactStore(artifact_root),
                tmp_path / "release",
            ).prepare(_request())

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(lambda _index: prepare(), range(2)))

    assert len({result.handoff_id for result in results}) == 1
    assert len({result.bundle_hash for result in results}) == 1
    assert sum(result.write_performed for result in results) == 1
    with Database(database_path, read_only=True) as connection:
        row = connection.connection.execute(
            "SELECT status, COUNT(*) AS count FROM workflow_report_handoffs"
        ).fetchone()
        assert tuple(row) == ("complete", 1)


def test_conflicting_parallel_report_workflow_reservations_leave_no_orphan_manifest(
    tmp_path: Path,
) -> None:
    database, artifact_store, _ = _complete_analysis_workflow(tmp_path)
    release = tmp_path / "release"
    service = WorkflowReportHandoffService(database, artifact_store, release)
    handoff = service.prepare(_request())
    compiled = service.compile_plan(handoff.handoff_id, _report_draft(tmp_path))
    approved = approve_report_plan_from_files(
        compiled.path,
        handoff.report_inputs.corpus_snapshot_path,
        handoff.report_inputs.search_audit_path,
        release,
        expected_hash=str(compiled.plan["plan_hash"]),
        approved_by="owner",
        approved_at="2026-08-11T00:02:00Z",
        save_bundle=False,
    )
    config_path = tmp_path / "parallel-report-config.yaml"
    config = yaml.safe_load(
        (ROOT / "configs" / "abstract_focus.yaml").read_text(encoding="utf-8")
    )
    config["project"]["output_dir"] = str(release)
    config["storage"]["sqlite_path"] = str(database.path)
    _configure_reduce_tree_summary(config)
    config["summary"]["report_plan"]["input_path"] = str(approved.path)
    config["summary"]["report_plan"]["content_hash"] = approved.plan["plan_hash"]
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    policy_path = tmp_path / "parallel-report-policy.yaml"
    policy_path.write_bytes(
        (ROOT / "policies" / "artifact-processing-v1.yaml").read_bytes()
    )
    database_path = database.path
    artifact_root = artifact_store.root
    bundle = ReportPlanBundle(
        approved.plan,
        handoff.report_inputs.corpus_snapshot,
        handoff.report_inputs.search_audit,
    )
    database.close()
    barrier = Barrier(2)
    manifest_paths = (
        tmp_path / "parallel-a-workflow.json",
        tmp_path / "parallel-b-workflow.json",
    )

    def reserve(index: int) -> object:
        request = WorkflowReportExecutionRequest(
            handoff.handoff_id,
            approved.path,
            config_path,
            manifest_paths[index],
            policy_path=policy_path,
            workflow_id=f"report-parallel-{index}",
            workflow_run_id=f"report-parallel-run-{index}",
        )
        with Database(database_path) as connection:
            barrier.wait()
            try:
                return WorkflowReportHandoffService(
                    connection, ArtifactStore(artifact_root), release
                ).prepare_report_workflow(request, approved_bundle=bundle)
            except WorkflowReportHandoffError as error:
                return error

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(reserve, range(2)))

    assert sum(not isinstance(result, Exception) for result in results) == 1
    assert sum(path.is_file() for path in manifest_paths) == 1
    with Database(database_path, read_only=True) as connection:
        assert connection.connection.execute(
            "SELECT COUNT(*) FROM workflow_report_executions"
        ).fetchone()[0] == 1
        assert connection.connection.execute(
            "SELECT COUNT(*) FROM report_plans"
        ).fetchone()[0] == 1


def test_report_workflow_reservation_recovers_after_bundle_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, artifact_store, _ = _complete_analysis_workflow(tmp_path)
    release = tmp_path / "release"
    service = WorkflowReportHandoffService(database, artifact_store, release)
    try:
        handoff = service.prepare(_request())
        compiled = service.compile_plan(handoff.handoff_id, _report_draft(tmp_path))
        approved = approve_report_plan_from_files(
            compiled.path,
            handoff.report_inputs.corpus_snapshot_path,
            handoff.report_inputs.search_audit_path,
            release,
            expected_hash=str(compiled.plan["plan_hash"]),
            approved_by="owner",
            approved_at="2026-08-11T00:02:00Z",
            save_bundle=False,
        )
        bundle = ReportPlanBundle(
            approved.plan,
            handoff.report_inputs.corpus_snapshot,
            handoff.report_inputs.search_audit,
        )
        config_path = tmp_path / "recover-report-config.yaml"
        config = yaml.safe_load(
            (ROOT / "configs" / "abstract_focus.yaml").read_text(encoding="utf-8")
        )
        config["project"]["output_dir"] = str(release)
        config["storage"]["sqlite_path"] = str(database.path)
        _configure_reduce_tree_summary(config)
        config["summary"]["report_plan"]["input_path"] = str(approved.path)
        config["summary"]["report_plan"]["content_hash"] = approved.plan["plan_hash"]
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        policy_path = tmp_path / "recover-report-policy.yaml"
        policy_path.write_bytes(
            (ROOT / "policies" / "artifact-processing-v1.yaml").read_bytes()
        )
        manifest_path = tmp_path / "recover-report-workflow.json"
        request = WorkflowReportExecutionRequest(
            handoff.handoff_id,
            approved.path,
            config_path,
            manifest_path,
            policy_path=policy_path,
        )
        original_write_bundle = handoff_module._write_report_plan_bundle
        write_attempts = 0

        def fail_first_bundle_write(*args: Any, **kwargs: Any) -> bool:
            nonlocal write_attempts
            write_attempts += 1
            if write_attempts == 1:
                raise OSError("simulated bundle write failure")
            return original_write_bundle(*args, **kwargs)

        monkeypatch.setattr(
            handoff_module, "_write_report_plan_bundle", fail_first_bundle_write
        )
        with pytest.raises(OSError, match="simulated bundle write failure"):
            service.prepare_report_workflow(request, approved_bundle=bundle)

        assert database.connection.execute(
            "SELECT COUNT(*) FROM workflow_report_executions"
        ).fetchone()[0] == 1
        assert not approved.path.exists()
        assert not manifest_path.exists()
        assert not (release / "reports" / "latest-approved-plan.json").exists()

        conflict_path = tmp_path / "conflicting-report-workflow.json"
        conflicting = WorkflowReportExecutionRequest(
            handoff.handoff_id,
            approved.path,
            config_path,
            conflict_path,
            policy_path=policy_path,
            workflow_id="different-report-workflow",
            workflow_run_id="different-report-workflow-run",
        )
        with pytest.raises(
            WorkflowReportHandoffError,
            match="persisted Report workflow binding has drifted",
        ):
            service.prepare_report_workflow(conflicting, approved_bundle=bundle)
        assert write_attempts == 1
        assert not conflict_path.exists()

        recovered = service.prepare_report_workflow(request, approved_bundle=bundle)

        assert recovered.persisted is True
        assert recovered.resumed is True
        assert recovered.write_performed is True
        assert write_attempts == 2
        assert approved.path.is_file()
        assert manifest_path.is_file()
        assert (release / "reports" / "latest-approved-plan.json").is_file()
    finally:
        database.close()


def test_cli_prepares_workflow_handoff_and_materializes_report_manifest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, artifact_store, analysis_adapter = _complete_analysis_workflow(tmp_path)
    database_path = database.path
    artifact_root = artifact_store.root
    database.close()
    release = tmp_path / "release"

    assert cli.main([
        "report",
        "prepare-inputs",
        "--database",
        str(database_path),
        "--artifact-root",
        str(artifact_root),
        "--output-root",
        str(release),
        "--workflow-run-id",
        WORKFLOW_RUN_ID,
        "--recent-cutoff",
        "2024-01-01",
        "--created-at",
        "2026-08-11T00:00:00Z",
    ]) == 0
    prepared = json.loads(capsys.readouterr().out)
    assert prepared["command"] == "report.prepare-workflow-inputs"

    assert cli.main([
        "report",
        "--plan-only",
        "--handoff-id",
        prepared["handoff_id"],
        "--draft",
        str(_report_draft(tmp_path)),
        "--database",
        str(database_path),
        "--artifact-root",
        str(artifact_root),
        "--output-root",
        str(release),
    ]) == 0
    planned = json.loads(capsys.readouterr().out)
    approved_path = Path(planned["draft_path"]).with_name("REPORT_PLAN.json")

    config_path = tmp_path / "report-cli-config.yaml"
    config = yaml.safe_load(
        (ROOT / "configs" / "abstract_focus.yaml").read_text(encoding="utf-8")
    )
    config["project"]["output_dir"] = str(release)
    config["storage"]["sqlite_path"] = str(database_path)
    _configure_reduce_tree_summary(config)
    config["summary"]["report_plan"]["input_path"] = str(approved_path)
    config["summary"]["report_plan"]["content_hash"] = planned["plan_hash"]
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    policy_path = tmp_path / "report-cli-policy.yaml"
    policy_path.write_bytes(
        (ROOT / "policies" / "artifact-processing-v1.yaml").read_bytes()
    )
    manifest_path = tmp_path / "report-cli-workflow.json"

    dry_approve_args = [
        "report",
        "approve",
        "--plan",
        planned["draft_path"],
        "--hash",
        planned["plan_hash"],
        "--approved-by",
        "owner",
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
        str(config_path),
        "--workflow-manifest",
        str(manifest_path),
        "--workflow-policy",
        str(policy_path),
    ]
    assert cli.main(["--dry-run", *dry_approve_args]) == 0
    dry_approved = json.loads(capsys.readouterr().out)
    assert dry_approved["status"] == "validated"
    assert dry_approved["write_performed"] is False
    assert dry_approved["report_workflow"]["persisted"] is False
    assert not approved_path.exists()
    assert not manifest_path.exists()
    assert not (release / "reports" / "latest-approved-plan.json").exists()
    with Database(database_path, read_only=True) as dry_database:
        assert dry_database.connection.execute(
            "SELECT COUNT(*) FROM workflow_report_executions"
        ).fetchone()[0] == 0

    assert cli.main([
        "report",
        "approve",
        "--plan",
        planned["draft_path"],
        "--hash",
        planned["plan_hash"],
        "--approved-by",
        "owner",
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
        str(config_path),
        "--workflow-manifest",
        str(manifest_path),
        "--workflow-policy",
        str(policy_path),
    ]) == 0
    approved = json.loads(capsys.readouterr().out)

    assert approved["report_workflow"]["manifest_path"] == str(manifest_path)
    manifest = load_workflow_manifest(manifest_path)
    assert tuple(step.stage for step in manifest.steps) == (StageKind.REPORT,)
    assert manifest.steps[0].artifact_root is not None
    assert manifest.steps[0].artifact_root.resolved_path == artifact_root.resolve()

    report_calls: list[str] = []

    class ReportService:
        def run(
            self,
            report_run_id: str,
            pipeline_run_id: str,
            bundle: Any,
            **kwargs: Any,
        ) -> SimpleNamespace:
            assert report_run_id == pipeline_run_id
            assert bundle.plan["workflow_handoff"]["handoff_id"] == prepared["handoff_id"]
            report_calls.append(report_run_id)
            return SimpleNamespace(
                report_run_id=report_run_id,
                status="complete",
                dry_run=kwargs["dry_run"],
            )

    def report_factory(
        _database: Any,
        store: ArtifactStore,
        *_args: Any,
    ) -> ReportService:
        assert store.root.resolve() == artifact_root.resolve()
        return ReportService()

    monkeypatch.setattr(
        cli,
        "default_stage_adapters",
        lambda: {StageKind.REPORT: ReportStageAdapter(report_factory)},
    )
    workflow_run_id = approved["report_workflow"]["report_workflow_run_id"]
    assert cli.main([
        "run",
        "--workflow",
        str(manifest_path),
        "--database",
        str(database_path),
        "--workflow-run-id",
        workflow_run_id,
    ]) == 0
    first_run = json.loads(capsys.readouterr().out)
    assert first_run["status"] == "complete"

    assert cli.main([
        "resume",
        "--workflow",
        str(manifest_path),
        "--database",
        str(database_path),
        "--workflow-run-id",
        workflow_run_id,
    ]) == 0
    resumed = json.loads(capsys.readouterr().out)
    assert resumed["status"] == "complete"
    assert report_calls == [f"{workflow_run_id}:report"]
    assert analysis_adapter.executions == [
        StageKind.SEARCH,
        StageKind.FILTER,
        StageKind.DOWNLOAD,
        StageKind.ANALYZE,
    ]
