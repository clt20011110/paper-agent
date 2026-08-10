"""Durable handoff from a completed analysis workflow to report planning."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from .approval import ApprovalError, require_valid_approval
from .artifacts import ArtifactStore
from .canonical import canonical_json, content_hash
from .config import load_config
from .report_cli_service import ReportPlanFileResult, compile_report_plan_from_files
from .report_config import ReportResources, ReportRuntimeConfig
from .report_input_service import ReportInputRequest, ReportInputResult, ReportInputService
from .report_plan import (
    ReportPlanBundle,
    ReportPlanError,
    ReportPlanStore,
    assert_report_runtime_matches,
)
from .schema import SchemaValidationError, validate
from .storage import Database
from .workflow import (
    DirectoryRef,
    FileRef,
    ReportStep,
    SequentialWorkflowOrchestrator,
    StageAdapter,
    StageKind,
    WorkflowManifest,
    load_workflow_manifest,
)


class WorkflowReportHandoffError(ValueError):
    """A persisted workflow cannot be trusted as a report input source."""


@dataclass(frozen=True, slots=True)
class WorkflowReportHandoffRequest:
    workflow_run_id: str
    recent_cutoff: str
    created_at: str

    def __post_init__(self) -> None:
        if not self.workflow_run_id:
            raise WorkflowReportHandoffError("workflow_run_id is required")
        try:
            date.fromisoformat(self.recent_cutoff)
            timestamp = datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))
        except ValueError as error:
            raise WorkflowReportHandoffError(
                "recent_cutoff and created_at must be ISO values"
            ) from error
        if timestamp.tzinfo is None:
            raise WorkflowReportHandoffError("created_at must include a timezone")


@dataclass(frozen=True, slots=True)
class _WorkflowBinding:
    workflow_run_id: str
    manifest_hash: str
    binding_hash: str
    crawl_run_id: str
    filter_run_id: str
    download_run_id: str
    stage4_run_id: str
    include_needs_review: bool
    selected_paper_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class WorkflowReportHandoffResult:
    handoff_id: str
    workflow_run_id: str
    workflow_manifest_hash: str
    workflow_binding_hash: str
    request_hash: str
    crawl_run_id: str
    filter_run_id: str
    download_run_id: str
    stage4_run_id: str
    include_needs_review: bool
    artifact_root: Path
    bundle_hash: str
    report_inputs: ReportInputResult
    corpus_snapshot_file_sha256: str
    search_audit_file_sha256: str
    status: str
    persisted: bool
    resumed: bool
    write_performed: bool

    def verify_files(self) -> None:
        """Fail closed before exposing persisted inputs as workflow FileRefs."""
        if not self.persisted:
            return
        if (
            _file_hash(self.report_inputs.corpus_snapshot_path)
            != self.corpus_snapshot_file_sha256
            or _file_hash(self.report_inputs.search_audit_path)
            != self.search_audit_file_sha256
        ):
            raise WorkflowReportHandoffError(
                "persisted report input file hash has drifted"
            )

    def document(self) -> dict[str, Any]:
        """Return the paths and hashes consumed by plan/approve and Report workflow."""
        self.verify_files()
        return {
            "handoff_id": self.handoff_id,
            "workflow_run_id": self.workflow_run_id,
            "workflow_manifest_hash": self.workflow_manifest_hash,
            "workflow_binding_hash": self.workflow_binding_hash,
            "request_hash": self.request_hash,
            "crawl_run_id": self.crawl_run_id,
            "filter_run_id": self.filter_run_id,
            "download_run_id": self.download_run_id,
            "stage4_run_id": self.stage4_run_id,
            "include_needs_review": self.include_needs_review,
            "artifact_root": str(self.artifact_root),
            "bundle_id": self.report_inputs.bundle_id,
            "bundle_hash": self.bundle_hash,
            "corpus_snapshot_hash": self.report_inputs.corpus_snapshot["snapshot_hash"],
            "corpus_snapshot_path": str(self.report_inputs.corpus_snapshot_path),
            "corpus_snapshot_file_sha256": self.corpus_snapshot_file_sha256,
            "search_audit_hash": self.report_inputs.search_audit["pack_hash"],
            "search_audit_path": str(self.report_inputs.search_audit_path),
            "search_audit_file_sha256": self.search_audit_file_sha256,
            "status": self.status,
            "persisted": self.persisted,
            "resumed": self.resumed,
            "write_performed": self.write_performed,
        }


@dataclass(frozen=True, slots=True)
class WorkflowReportExecutionRequest:
    handoff_id: str
    approved_plan_path: Path
    config_path: Path
    manifest_path: Path
    processing_grants_path: Path | None = None
    previous_report_run_id: str | None = None
    policy_path: Path | None = None
    workflow_id: str | None = None
    workflow_run_id: str | None = None
    step_id: str = "report"

    def __post_init__(self) -> None:
        if not self.handoff_id or not self.step_id:
            raise WorkflowReportHandoffError(
                "handoff_id and report step_id are required"
            )
        for value, label in ((self.config_path, "workflow config"),):
            if not Path(value).is_file():
                raise WorkflowReportHandoffError(f"{label} is unavailable: {value}")
        for value, label in (
            (self.processing_grants_path, "processing grants"),
            (self.policy_path, "report policy"),
        ):
            if value is not None and not Path(value).is_file():
                raise WorkflowReportHandoffError(f"{label} is unavailable: {value}")


@dataclass(frozen=True, slots=True)
class WorkflowReportExecutionResult:
    handoff_id: str
    report_plan_id: str
    report_plan_hash: str
    report_workflow_id: str
    report_workflow_run_id: str
    manifest_path: Path
    manifest_hash: str
    manifest: WorkflowManifest
    persisted: bool
    resumed: bool
    write_performed: bool

    def document(self) -> dict[str, Any]:
        if self.persisted and (
            not self.manifest_path.is_file()
            or _file_hash(self.manifest_path) != self.manifest_hash
        ):
            raise WorkflowReportHandoffError(
                "persisted report workflow manifest hash has drifted"
            )
        return {
            "handoff_id": self.handoff_id,
            "report_plan_id": self.report_plan_id,
            "report_plan_hash": self.report_plan_hash,
            "report_workflow_id": self.report_workflow_id,
            "report_workflow_run_id": self.report_workflow_run_id,
            "manifest_path": str(self.manifest_path),
            "manifest_hash": self.manifest_hash,
            "persisted": self.persisted,
            "resumed": self.resumed,
            "write_performed": self.write_performed,
        }


class WorkflowReportHandoffService:
    """Freeze a fully complete Search-through-Analyze workflow for reporting.

    This boundary is intentionally narrower than the generic ReportInputService:
    an incomplete or failed Stage 4 remains resumable upstream and cannot be
    presented as a completed workflow handoff.
    """

    def __init__(
        self,
        database: Database,
        artifact_store: ArtifactStore,
        output_root: str | Path,
    ) -> None:
        self.database = database
        self.artifact_store = artifact_store
        self.artifact_root = artifact_store.root.resolve()
        self.output_root = Path(output_root).resolve()

    def prepare(
        self,
        request: WorkflowReportHandoffRequest,
        *,
        save_bundle: bool = True,
    ) -> WorkflowReportHandoffResult:
        binding = self._binding(request.workflow_run_id)
        request_hash = _request_hash(
            binding, request, self.artifact_root, self.output_root
        )
        handoff_id = f"workflow-report-{request_hash[:16]}"
        input_request = ReportInputRequest(
            binding.crawl_run_id,
            binding.filter_run_id,
            binding.stage4_run_id,
            request.recent_cutoff,
            request.created_at,
            binding.include_needs_review,
        )
        input_service = ReportInputService(
            self.database, self.artifact_store, self.output_root
        )
        if not save_bundle:
            inputs = input_service.build(input_request, save_bundle=False)
            self._assert_report_inputs(binding, inputs)
            if self._binding(request.workflow_run_id) != binding:
                raise WorkflowReportHandoffError(
                    "workflow report inputs changed during validation"
                )
            return self._result(
                handoff_id, binding, request_hash, inputs,
                status="validated", persisted=False, resumed=False,
                write_performed=False,
            )
        if self.database.read_only:
            raise WorkflowReportHandoffError(
                "persisting a workflow report handoff requires a writable database"
            )

        row = self._row(handoff_id)
        resumed = row is not None
        if row is None:
            self._checkpoint(handoff_id, binding, request, request_hash)
            row = self._row(handoff_id)
            assert row is not None
        self._assert_row(row, binding, request, request_hash)
        if row["status"] == "complete":
            inputs = self._load(row)
            return self._result(
                handoff_id, binding, request_hash, inputs,
                status="complete", persisted=True, resumed=True,
                write_performed=False, bundle_hash=str(row["bundle_hash"]),
                corpus_file_hash=str(row["corpus_file_sha256"]),
                audit_file_hash=str(row["search_audit_file_sha256"]),
            )

        inputs = input_service.build(input_request, save_bundle=True)
        self._assert_report_inputs(binding, inputs)
        bundle_hash, completed = self._finish(
            handoff_id, binding, request_hash, inputs
        )
        row = self._row(handoff_id)
        if row is None or row["status"] != "complete":
            raise WorkflowReportHandoffError("workflow report handoff was not completed")
        if row["bundle_hash"] != bundle_hash:
            inputs = self._load(row)
            bundle_hash = str(row["bundle_hash"])
            resumed = True
        return self._result(
            handoff_id, binding, request_hash, inputs,
            status="complete", persisted=True, resumed=resumed,
            write_performed=completed, bundle_hash=bundle_hash,
        )

    def load(self, handoff_id: str) -> WorkflowReportHandoffResult:
        """Load one complete handoff and revalidate its workflow and files."""
        row = self._row(handoff_id)
        if row is None or row["status"] != "complete":
            raise WorkflowReportHandoffError("workflow report handoff is not complete")
        request = WorkflowReportHandoffRequest(
            str(row["workflow_run_id"]),
            str(row["recent_cutoff"]),
            str(row["input_created_at"]),
        )
        binding = self._binding(request.workflow_run_id)
        request_hash = _request_hash(
            binding, request, self.artifact_root, self.output_root
        )
        self._assert_row(row, binding, request, request_hash)
        inputs = self._load(row)
        self._assert_report_inputs(binding, inputs)
        if self._binding(request.workflow_run_id) != binding:
            raise WorkflowReportHandoffError(
                "workflow report source changed while loading the handoff"
            )
        return self._result(
            handoff_id,
            binding,
            request_hash,
            inputs,
            status="complete",
            persisted=True,
            resumed=True,
            write_performed=False,
            bundle_hash=str(row["bundle_hash"]),
            corpus_file_hash=str(row["corpus_file_sha256"]),
            audit_file_hash=str(row["search_audit_file_sha256"]),
        )

    def compile_plan(
        self,
        handoff_id: str,
        draft_path: str | Path,
        *,
        save_draft: bool = True,
        resources: ReportResources | None = None,
    ) -> ReportPlanFileResult:
        """Compile a ReportPlan whose approval hash includes this handoff."""
        handoff = self.load(handoff_id)
        return compile_report_plan_from_files(
            draft_path,
            handoff.report_inputs.corpus_snapshot_path,
            handoff.report_inputs.search_audit_path,
            self.output_root,
            save_draft=save_draft,
            resources=resources,
            workflow_handoff=_plan_binding(handoff),
        )

    def prepare_report_workflow(
        self,
        request: WorkflowReportExecutionRequest,
        *,
        save_manifest: bool = True,
        approved_bundle: ReportPlanBundle | None = None,
    ) -> WorkflowReportExecutionResult:
        """Bind an approved plan to one standalone, immutable Report workflow."""
        handoff = self.load(request.handoff_id)
        plan_path = Path(request.approved_plan_path).resolve()
        if approved_bundle is None:
            plan, corpus, audit = self._approved_bundle(handoff, plan_path)
        else:
            plan = approved_bundle.plan
            corpus = approved_bundle.corpus_snapshot
            audit = approved_bundle.search_audit
            self._assert_approved_documents(handoff, plan, corpus, audit)
        expected_plan_path = ReportPlanStore(self.output_root).approved_path(
            str(plan["plan_id"])
        ).resolve()
        if plan_path != expected_plan_path:
            raise WorkflowReportHandoffError(
                "approved ReportPlan path is not the canonical handoff bundle path"
            )
        self._assert_report_config(Path(request.config_path).resolve(), plan)
        workflow_id = request.workflow_id or f"report-{handoff.bundle_hash[:16]}"
        workflow_run_id = request.workflow_run_id or workflow_id
        manifest_path = Path(request.manifest_path).resolve()
        plan_payload = canonical_json(dict(plan))
        corpus_payload = canonical_json(dict(corpus))
        audit_payload = canonical_json(dict(audit))
        planned_files = {
            plan_path: plan_payload,
            (plan_path.parent / "CORPUS_SNAPSHOT.json").resolve(): corpus_payload,
            (plan_path.parent / "SEARCH_AUDIT.json").resolve(): audit_payload,
        }
        manifest = _report_workflow_manifest(
            workflow_id=workflow_id,
            step_id=request.step_id,
            source_path=manifest_path,
            config_path=Path(request.config_path).resolve(),
            plan_path=plan_path,
            corpus_path=(plan_path.parent / "CORPUS_SNAPSHOT.json").resolve(),
            audit_path=(plan_path.parent / "SEARCH_AUDIT.json").resolve(),
            processing_grants_path=(
                Path(request.processing_grants_path).resolve()
                if request.processing_grants_path is not None
                else None
            ),
            previous_report_run_id=request.previous_report_run_id,
            policy_path=(
                Path(request.policy_path).resolve()
                if request.policy_path is not None
                else None
            ),
            artifact_root=handoff.artifact_root,
            planned_files=planned_files,
        )
        manifest_payload = canonical_json(manifest.document())
        manifest_hash = sha256(manifest_payload).hexdigest()
        if manifest_hash != manifest.manifest_hash:
            raise WorkflowReportHandoffError(
                "report workflow manifest hash is inconsistent"
            )
        if manifest_path.exists() and (
            not manifest_path.is_file()
            or manifest_path.read_bytes() != manifest_payload
        ):
            raise WorkflowReportHandoffError(
                f"Report workflow manifest is immutable: {manifest_path}"
            )
        result = WorkflowReportExecutionResult(
            handoff.handoff_id,
            str(plan["plan_id"]),
            str(plan["plan_hash"]),
            workflow_id,
            workflow_run_id,
            manifest_path,
            manifest_hash,
            manifest,
            False,
            False,
            False,
        )
        if not save_manifest:
            return result
        if self.database.read_only:
            raise WorkflowReportHandoffError(
                "persisting a Report workflow requires a writable database"
            )

        expected = (
            handoff.handoff_id,
            str(plan["plan_id"]),
            str(plan["plan_hash"]),
            str(plan_path),
            sha256(plan_payload).hexdigest(),
            workflow_id,
            workflow_run_id,
            manifest_hash,
            manifest_payload.decode("utf-8"),
            str(manifest_path),
        )
        inserted = self._reserve_report_workflow(plan, expected)
        wrote_bundle = _write_report_plan_bundle(
            self.output_root, plan, corpus, audit
        )
        wrote_manifest = _write_immutable(manifest_path, manifest_payload)
        _write_latest_report_plan(self.output_root, plan)
        loaded = self.load_report_workflow(handoff.handoff_id)
        return WorkflowReportExecutionResult(
            loaded.handoff_id,
            loaded.report_plan_id,
            loaded.report_plan_hash,
            loaded.report_workflow_id,
            loaded.report_workflow_run_id,
            loaded.manifest_path,
            loaded.manifest_hash,
            loaded.manifest,
            True,
            not inserted,
            inserted or wrote_bundle or wrote_manifest,
        )

    def load_report_workflow(
        self, handoff_id: str
    ) -> WorkflowReportExecutionResult:
        """Verify the approved plan, all FileRefs, and the execution registry."""
        handoff = self.load(handoff_id)
        row = self.database.connection.execute(
            "SELECT * FROM workflow_report_executions WHERE handoff_id = ?",
            (handoff_id,),
        ).fetchone()
        if row is None:
            raise WorkflowReportHandoffError(
                "handoff has no approved standalone Report workflow"
            )
        manifest_path = Path(str(row["report_manifest_path"]))
        if (
            _file_hash(manifest_path) != row["report_manifest_hash"]
            or _file_hash(Path(str(row["report_plan_path"])))
            != row["report_plan_file_sha256"]
        ):
            raise WorkflowReportHandoffError(
                "Report workflow FileRef content has drifted"
            )
        manifest = load_workflow_manifest(manifest_path)
        if (
            manifest.manifest_hash != row["report_manifest_hash"]
            or canonical_json(manifest.document()).decode("utf-8")
            != row["report_manifest_json"]
            or manifest.workflow_id != row["report_workflow_id"]
            or len(manifest.steps) != 1
            or not isinstance(manifest.steps[0], ReportStep)
        ):
            raise WorkflowReportHandoffError(
                "standalone Report workflow manifest has drifted"
            )
        step = manifest.steps[0]
        plan_path = Path(str(row["report_plan_path"]))
        if step.plan.resolved_path != plan_path:
            raise WorkflowReportHandoffError(
                "Report workflow plan FileRef has drifted"
            )
        plan, corpus, audit = self._approved_bundle(handoff, plan_path)
        if (
            plan["plan_id"] != row["report_plan_id"]
            or plan["plan_hash"] != row["report_plan_hash"]
            or step.corpus_snapshot.resolved_path
            != (plan_path.parent / "CORPUS_SNAPSHOT.json").resolve()
            or step.search_audit.resolved_path
            != (plan_path.parent / "SEARCH_AUDIT.json").resolve()
            or step.artifact_root is None
            or step.artifact_root.resolved_path != handoff.artifact_root
        ):
            raise WorkflowReportHandoffError(
                "Report workflow approved input bundle has drifted"
            )
        assert_report_handoff_runtime(
            self.database,
            plan,
            corpus,
            audit,
            workflow_run_id=str(row["report_workflow_run_id"]),
            require_persisted_workflow=False,
        )
        return WorkflowReportExecutionResult(
            handoff_id,
            str(row["report_plan_id"]),
            str(row["report_plan_hash"]),
            str(row["report_workflow_id"]),
            str(row["report_workflow_run_id"]),
            manifest_path,
            str(row["report_manifest_hash"]),
            manifest,
            True,
            True,
            False,
        )

    def run_report_workflow(
        self,
        handoff_id: str,
        *,
        resume: bool = False,
        dry_run: bool = False,
        adapter: StageAdapter | None = None,
    ) -> dict[str, Any]:
        """Run or resume only the handoff-bound standalone Report stage."""
        execution = self.load_report_workflow(handoff_id)
        if adapter is None:
            from .workflow_adapters import ReportStageAdapter

            adapter = ReportStageAdapter()
        orchestrator = SequentialWorkflowOrchestrator(
            self.database,
            execution.manifest,
            {StageKind.REPORT: adapter},
        )
        operation = orchestrator.resume if resume else orchestrator.run
        return operation(execution.report_workflow_run_id, dry_run=dry_run)

    def _approved_bundle(
        self,
        handoff: WorkflowReportHandoffResult,
        plan_path: Path,
    ) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
        try:
            plan = _object(plan_path.read_bytes(), "approved ReportPlan")
            corpus_path = (plan_path.parent / "CORPUS_SNAPSHOT.json").resolve()
            audit_path = (plan_path.parent / "SEARCH_AUDIT.json").resolve()
            corpus = _object(corpus_path.read_bytes(), "approved corpus snapshot")
            audit = _object(audit_path.read_bytes(), "approved search audit")
            self._assert_approved_documents(handoff, plan, corpus, audit)
        except OSError as error:
            raise WorkflowReportHandoffError(
                f"approved ReportPlan bundle is invalid: {error}"
            ) from error
        if (
            _file_hash(corpus_path) != handoff.corpus_snapshot_file_sha256
            or _file_hash(audit_path) != handoff.search_audit_file_sha256
        ):
            raise WorkflowReportHandoffError(
                "approved ReportPlan inputs do not match the workflow handoff"
            )
        return plan, corpus, audit

    @staticmethod
    def _assert_approved_documents(
        handoff: WorkflowReportHandoffResult,
        plan: Mapping[str, Any],
        corpus: Mapping[str, Any],
        audit: Mapping[str, Any],
    ) -> None:
        try:
            require_valid_approval(plan, "plan_hash")
            validate(plan, "report-plan.schema.json")
            assert_report_runtime_matches(
                plan,
                plan,
                corpus_snapshot=corpus,
                search_audit_pack=audit,
            )
        except (OSError, ApprovalError, SchemaValidationError, ReportPlanError) as error:
            raise WorkflowReportHandoffError(
                f"approved ReportPlan bundle is invalid: {error}"
            ) from error
        if plan.get("workflow_handoff") != _plan_binding(handoff):
            raise WorkflowReportHandoffError(
                "approved ReportPlan is not bound to this workflow handoff"
            )
        if (
            dict(corpus) != dict(handoff.report_inputs.corpus_snapshot)
            or dict(audit) != dict(handoff.report_inputs.search_audit)
        ):
            raise WorkflowReportHandoffError(
                "approved ReportPlan inputs do not match the workflow handoff"
            )

    def _reserve_report_workflow(
        self, plan: Mapping[str, Any], expected: tuple[object, ...]
    ) -> bool:
        """Reserve the only execution binding before writing any workflow files."""
        plan_text = canonical_json(dict(plan)).decode("utf-8")
        approval_text = canonical_json(dict(plan["approval"])).decode("utf-8")
        plan_expected = (
            str(plan["plan_hash"]),
            str(plan["schema_version"]),
            plan_text,
            approval_text,
            "approved",
        )
        with self.database.transaction() as connection:
            plan_row = connection.execute(
                """SELECT content_hash, schema_version, plan_json,
                          approval_json, status
                   FROM report_plans WHERE report_plan_id = ?""",
                (plan["plan_id"],),
            ).fetchone()
            if plan_row is None:
                connection.execute(
                    """INSERT INTO report_plans(
                           report_plan_id, content_hash, schema_version,
                           plan_json, approval_json, status
                       ) VALUES (?, ?, ?, ?, ?, 'approved')""",
                    (plan["plan_id"], *plan_expected[:-1]),
                )
            elif tuple(plan_row) != plan_expected:
                raise WorkflowReportHandoffError(
                    "persisted approved ReportPlan is immutable"
                )
            inserted = connection.execute(
                """INSERT OR IGNORE INTO workflow_report_executions(
                       handoff_id, report_plan_id, report_plan_hash,
                       report_plan_path, report_plan_file_sha256,
                       report_workflow_id, report_workflow_run_id,
                       report_manifest_hash, report_manifest_json,
                       report_manifest_path
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                expected,
            ).rowcount == 1
            row = connection.execute(
                "SELECT * FROM workflow_report_executions WHERE handoff_id = ?",
                (expected[0],),
            ).fetchone()
            fields = (
                "handoff_id",
                "report_plan_id",
                "report_plan_hash",
                "report_plan_path",
                "report_plan_file_sha256",
                "report_workflow_id",
                "report_workflow_run_id",
                "report_manifest_hash",
                "report_manifest_json",
                "report_manifest_path",
            )
            if row is None or tuple(row[key] for key in fields) != expected:
                raise WorkflowReportHandoffError(
                    "persisted Report workflow binding has drifted"
                )
        return inserted

    def _assert_report_config(
        self, config_path: Path, plan: Mapping[str, Any]
    ) -> None:
        config = load_config(config_path)
        runtime = (
            ReportRuntimeConfig.from_config(config, config_path)
            if "summary" in config
            else ReportRuntimeConfig.defaults()
        )
        if not runtime.enabled:
            raise WorkflowReportHandoffError(
                "standalone Report workflow requires summary.enabled=true"
            )
        expected_plan_path = ReportPlanStore(self.output_root).approved_path(
            str(plan["plan_id"])
        ).resolve()
        if (
            runtime.report_plan_path is None
            or runtime.report_plan_path.resolve() != expected_plan_path
            or runtime.report_plan_hash != plan.get("plan_hash")
        ):
            raise WorkflowReportHandoffError(
                "Report workflow config does not pin the approved handoff plan"
            )
        if runtime.report_plan_path.is_file():
            runtime.validate_for_run(plan, execution_mode="unattended")
        else:
            runtime.resources.validate_files()
            if runtime.rubric_path is not None and not runtime.rubric_path.is_file():
                raise WorkflowReportHandoffError(
                    f"summary audit rubric is unavailable: {runtime.rubric_path}"
                )
        try:
            output = Path(str(config["project"]["output_dir"]))
            database = Path(str(config["storage"]["sqlite_path"]))
        except (KeyError, TypeError) as error:
            raise WorkflowReportHandoffError(
                "Report workflow config lacks project/storage bindings"
            ) from error
        resolved_output = (
            output.resolve()
            if output.is_absolute()
            else (config_path.parent / output).resolve()
        )
        resolved_database = (
            database.resolve()
            if database.is_absolute()
            else (config_path.parent / database).resolve()
        )
        if resolved_output != self.output_root:
            raise WorkflowReportHandoffError(
                "Report workflow config output_root differs from the handoff"
            )
        if resolved_database != self.database.path.resolve():
            raise WorkflowReportHandoffError(
                "Report workflow config database differs from the handoff"
            )

    def _binding(self, workflow_run_id: str) -> _WorkflowBinding:
        run = self.database.connection.execute(
            """SELECT manifest_hash, manifest_json, status FROM workflow_runs
               WHERE workflow_run_id = ?""",
            (workflow_run_id,),
        ).fetchone()
        if run is None or run["status"] != "complete":
            raise WorkflowReportHandoffError("report handoff requires a complete workflow run")
        manifest = _object(run["manifest_json"], "workflow manifest")
        if content_hash(dict(manifest)) != run["manifest_hash"]:
            raise WorkflowReportHandoffError("workflow manifest hash has drifted")
        if manifest.get("schema_version") != "2":
            raise WorkflowReportHandoffError("report handoff requires workflow schema version 2")
        specs = manifest.get("steps")
        if not isinstance(specs, list) or not all(isinstance(item, Mapping) for item in specs):
            raise WorkflowReportHandoffError("workflow manifest steps are malformed")
        by_stage = {str(item.get("stage")): item for item in specs}
        order = ("search", "filter", "download", "analyze")
        if tuple(by_stage) != order:
            raise WorkflowReportHandoffError(
                "report handoff requires exactly search, filter, download, and analyze"
            )
        for stage, upstream in zip(order[1:], order):
            if by_stage[stage].get("selection") != {"from_step": by_stage[upstream].get("id")}:
                raise WorkflowReportHandoffError(f"workflow {stage} lineage has drifted")

        rows = self.database.connection.execute(
            """SELECT step_id, ordinal, stage, child_run_id, spec_hash,
                      identity_hash, status, result_json
               FROM workflow_steps WHERE workflow_run_id = ? ORDER BY ordinal""",
            (workflow_run_id,),
        ).fetchall()
        if tuple(str(row["stage"]) for row in rows) != order:
            raise WorkflowReportHandoffError("persisted workflow steps are incomplete")
        payloads: dict[str, Mapping[str, Any]] = {}
        step_hashes = []
        for row, stage in zip(rows, order):
            spec = by_stage[stage]
            if (
                row["status"] != "complete"
                or row["step_id"] != spec.get("id")
                or row["child_run_id"] != f"{workflow_run_id}:{row['step_id']}"
                or row["spec_hash"] != content_hash(dict(spec))
                or not _hash(row["identity_hash"])
            ):
                raise WorkflowReportHandoffError(f"persisted {stage} step has drifted")
            payload = _object(row["result_json"], f"{stage} workflow result")
            payloads[stage] = payload
            step_hashes.append({
                "step_id": row["step_id"],
                "identity_hash": row["identity_hash"],
                "result_hash": content_hash(dict(payload)),
            })

        search_run_id = _string(payloads["search"], "run_id", "search")
        crawl_run_id = _string(payloads["search"], "crawl_run_id", "search")
        filter_run_ids = payloads["filter"].get("stage2_run_ids")
        if not (
            isinstance(filter_run_ids, list)
            and len(filter_run_ids) == 1
            and isinstance(filter_run_ids[0], str)
            and filter_run_ids[0]
        ):
            raise WorkflowReportHandoffError(
                "workflow filter result must contain exactly one Stage 2 run ID"
            )
        filter_run_id = filter_run_ids[0]
        download_run_id = _string(payloads["download"], "run_id", "download")
        stage4_run_id = _string(payloads["analyze"], "run_id", "analyze")
        if search_run_id != rows[0]["child_run_id"]:
            raise WorkflowReportHandoffError("workflow search run ID has drifted")
        if download_run_id != rows[2]["child_run_id"]:
            raise WorkflowReportHandoffError("workflow download run ID has drifted")
        if stage4_run_id != rows[3]["child_run_id"]:
            raise WorkflowReportHandoffError("workflow Stage 4 run ID has drifted")

        eligible_paper_ids = _paper_ids(
            payloads["search"], "eligible_paper_ids", "search"
        )
        reported_decisions = _decision_map(payloads["filter"])
        if set(reported_decisions) != set(eligible_paper_ids):
            raise WorkflowReportHandoffError(
                "workflow Filter decisions do not exhaust the Search selection"
            )
        decision_rows = self.database.connection.execute(
            """SELECT paper_id, status FROM filter_decisions
               WHERE run_id = ? ORDER BY paper_id""",
            (filter_run_id,),
        ).fetchall()
        persisted_decisions = {
            str(row["paper_id"]): str(row["status"]) for row in decision_rows
        }
        if persisted_decisions != reported_decisions:
            raise WorkflowReportHandoffError(
                "workflow Filter decisions have drifted from Stage 2"
            )
        include_needs_review = by_stage["download"].get("include_needs_review")
        if not isinstance(include_needs_review, bool):
            raise WorkflowReportHandoffError("workflow Download selection is not frozen")
        included_statuses = {"relevant"}
        if include_needs_review:
            included_statuses.add("needs_review")
        selected_paper_ids = tuple(sorted(
            paper_id
            for paper_id, status in reported_decisions.items()
            if status in included_statuses
        ))
        if not selected_paper_ids:
            raise WorkflowReportHandoffError(
                "completed workflow selected no papers for Report"
            )
        download_paper_ids = tuple(sorted(_paper_ids(
            payloads["download"], "paper_ids", "download"
        )))
        analyze_paper_ids = tuple(sorted(_paper_ids(
            payloads["analyze"], "paper_ids", "analyze"
        )))
        if download_paper_ids != selected_paper_ids:
            raise WorkflowReportHandoffError(
                "workflow Download papers have drifted from the frozen Filter selection"
            )
        if analyze_paper_ids != download_paper_ids:
            raise WorkflowReportHandoffError(
                "workflow Analyze papers have drifted from Download"
            )

        pipelines = (
            self._pipeline(payloads["search"], search_run_id, "stage-1"),
            self._pipeline(payloads["filter"], filter_run_id, "stage-2"),
            self._pipeline(payloads["download"], download_run_id, "stage-3-download"),
            self._pipeline(payloads["analyze"], stage4_run_id, "stage4"),
        )
        stage3_corpus = self._stage3_corpus(
            download_run_id, selected_paper_ids
        )
        analysis_corpus = self._analysis_corpus(
            stage4_run_id, selected_paper_ids, pipelines[3]
        )
        crawl = self.database.connection.execute(
            "SELECT run_id, status FROM crawl_runs WHERE crawl_run_id = ?",
            (crawl_run_id,),
        ).fetchone()
        if crawl is None or tuple(crawl) != (search_run_id, "complete"):
            raise WorkflowReportHandoffError("crawl run is not bound to the search child")
        binding_hash = content_hash({
            "schema_version": "1",
            "workflow_run_id": workflow_run_id,
            "manifest_hash": run["manifest_hash"],
            "steps": step_hashes,
            "pipelines": pipelines,
            "stage3_corpus": stage3_corpus,
            "analysis_corpus": analysis_corpus,
            "crawl_run_id": crawl_run_id,
            "filter_run_id": filter_run_id,
            "download_run_id": download_run_id,
            "stage4_run_id": stage4_run_id,
            "include_needs_review": include_needs_review,
            "selected_paper_ids": list(selected_paper_ids),
        })
        return _WorkflowBinding(
            workflow_run_id, str(run["manifest_hash"]), binding_hash,
            crawl_run_id, filter_run_id, download_run_id, stage4_run_id,
            include_needs_review, selected_paper_ids,
        )

    def _pipeline(
        self, payload: Mapping[str, Any], run_id: str, expected_stage: str
    ) -> dict[str, str]:
        row = self.database.connection.execute(
            """SELECT stage, status, input_hash, config_hash, implementation_version
               FROM pipeline_runs WHERE run_id = ?""",
            (run_id,),
        ).fetchone()
        if row is None or row["stage"] != expected_stage or row["status"] != "complete":
            raise WorkflowReportHandoffError(
                f"workflow child {run_id} is not a complete {expected_stage} run"
            )
        current = {
            "run_id": run_id,
            "stage": str(row["stage"]),
            "status": str(row["status"]),
            "input_hash": str(row["input_hash"]),
            "config_hash": str(row["config_hash"]),
            "implementation_version": str(row["implementation_version"]),
        }
        if payload.get("_pipeline_binding") != current:
            raise WorkflowReportHandoffError(f"workflow child {run_id} binding has drifted")
        return current

    def _stage3_corpus(
        self, run_id: str, expected_paper_ids: tuple[str, ...]
    ) -> tuple[dict[str, str], ...]:
        rows = self.database.connection.execute(
            """SELECT paper_id, status, reason_code
               FROM stage3_paper_results
               WHERE run_id = ? ORDER BY paper_id""",
            (run_id,),
        ).fetchall()
        paper_ids = tuple(str(row["paper_id"]) for row in rows)
        if paper_ids != expected_paper_ids:
            raise WorkflowReportHandoffError(
                "Stage 3 paper checkpoints do not match the Download selection"
            )
        terminal = {"downloaded", "not_available", "failed_terminal"}
        if any(row["status"] not in terminal for row in rows):
            raise WorkflowReportHandoffError(
                "complete Stage 3 workflow contains a non-terminal paper checkpoint"
            )
        return tuple({
            "paper_id": str(row["paper_id"]),
            "status": str(row["status"]),
            "reason_code": str(row["reason_code"]),
        } for row in rows)

    def _analysis_corpus(
        self,
        run_id: str,
        expected_paper_ids: tuple[str, ...],
        pipeline: Mapping[str, str],
    ) -> tuple[dict[str, str], ...]:
        dispatches = self.database.connection.execute(
            """SELECT dispatch_id, paper_id, artifact_hash, artifact_id,
                      input_scope, config_hash, implementation_version,
                      model_id, prompt_hash, schema_hash, prompt_input_hash,
                      status, dispatch_count, analysis_run_id
               FROM analysis_dispatches
               WHERE run_id = ? ORDER BY paper_id""",
            (run_id,),
        ).fetchall()
        analyses = self.database.connection.execute(
            """SELECT analysis_run_id, paper_id, artifact_id, input_hash,
                      input_scope, model_id, prompt_hash, schema_hash,
                      implementation_version, status, output_artifact_id
               FROM analysis_runs
               WHERE run_id = ? ORDER BY paper_id, analysis_run_id""",
            (run_id,),
        ).fetchall()
        dispatch_papers = tuple(str(row["paper_id"]) for row in dispatches)
        analysis_papers = tuple(str(row["paper_id"]) for row in analyses)
        if (
            dispatch_papers != expected_paper_ids
            or analysis_papers != expected_paper_ids
        ):
            raise WorkflowReportHandoffError(
                "Stage 4 persisted corpus does not match the Analyze selection"
            )
        analyses_by_id = {
            str(row["analysis_run_id"]): row for row in analyses
        }
        documents: list[dict[str, str]] = []
        for dispatch in dispatches:
            analysis_run_id = dispatch["analysis_run_id"]
            analysis = analyses_by_id.get(str(analysis_run_id))
            if (
                dispatch["status"] != "complete"
                or dispatch["dispatch_count"] != 1
                or analysis is None
                or analysis["status"] != "complete"
                or analysis["paper_id"] != dispatch["paper_id"]
                or analysis["artifact_id"] != dispatch["artifact_id"]
                or analysis["input_scope"] != dispatch["input_scope"]
                or analysis["model_id"] != dispatch["model_id"]
                or analysis["prompt_hash"] != dispatch["prompt_hash"]
                or analysis["schema_hash"] != dispatch["schema_hash"]
                or analysis["implementation_version"]
                != dispatch["implementation_version"]
                or analysis["input_hash"] != dispatch["prompt_input_hash"]
                or dispatch["config_hash"] != pipeline["config_hash"]
                or dispatch["implementation_version"]
                != pipeline["implementation_version"]
                or analysis["output_artifact_id"] is None
            ):
                raise WorkflowReportHandoffError(
                    "Stage 4 dispatch and analysis lineage has drifted"
                )
            documents.append({
                "paper_id": str(dispatch["paper_id"]),
                "dispatch_id": str(dispatch["dispatch_id"]),
                "analysis_run_id": str(analysis_run_id),
                "artifact_hash": str(dispatch["artifact_hash"]),
                "input_scope": str(dispatch["input_scope"]),
                "prompt_input_hash": str(dispatch["prompt_input_hash"]),
                "output_artifact_id": str(analysis["output_artifact_id"]),
            })
        return tuple(documents)

    @staticmethod
    def _assert_report_inputs(
        binding: _WorkflowBinding, inputs: ReportInputResult
    ) -> None:
        papers = inputs.corpus_snapshot.get("papers")
        if not isinstance(papers, list) or not all(
            isinstance(paper, Mapping) for paper in papers
        ):
            raise WorkflowReportHandoffError("corpus snapshot papers are malformed")
        paper_ids = tuple(sorted(str(paper.get("paper_id")) for paper in papers))
        if (
            len(set(paper_ids)) != len(paper_ids)
            or paper_ids != binding.selected_paper_ids
        ):
            raise WorkflowReportHandoffError(
                "corpus snapshot does not match the completed Analyze selection"
            )
        if any(
            not isinstance(paper.get("analysis_run_id"), str)
            or not paper.get("analysis_run_id")
            for paper in papers
        ):
            raise WorkflowReportHandoffError(
                "completed Stage 4 workflow has a missing paper analysis"
            )
        audit_hash = inputs.search_audit.get("corpus_snapshot_hash")
        if audit_hash != inputs.corpus_snapshot.get("snapshot_hash"):
            raise WorkflowReportHandoffError(
                "search audit is not bound to the generated corpus snapshot"
            )

    def _checkpoint(
        self,
        handoff_id: str,
        binding: _WorkflowBinding,
        request: WorkflowReportHandoffRequest,
        request_hash: str,
    ) -> None:
        with self.database.transaction() as connection:
            connection.execute(
                """INSERT OR IGNORE INTO workflow_report_handoffs(
                       handoff_id, workflow_run_id, workflow_manifest_hash,
                       workflow_binding_hash, request_hash, crawl_run_id,
                       filter_run_id, download_run_id, stage4_run_id,
                       recent_cutoff, input_created_at, include_needs_review,
                       artifact_root, output_root, status
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'preparing')""",
                (
                    handoff_id, binding.workflow_run_id, binding.manifest_hash,
                    binding.binding_hash, request_hash, binding.crawl_run_id,
                    binding.filter_run_id, binding.download_run_id,
                    binding.stage4_run_id, request.recent_cutoff,
                    request.created_at, int(binding.include_needs_review),
                    str(self.artifact_root),
                    str(self.output_root),
                ),
            )

    def _finish(
        self,
        handoff_id: str,
        binding: _WorkflowBinding,
        request_hash: str,
        inputs: ReportInputResult,
    ) -> tuple[str, bool]:
        corpus_file_hash = _file_hash(inputs.corpus_snapshot_path)
        audit_file_hash = _file_hash(inputs.search_audit_path)
        bundle_hash = _bundle_hash(
            binding.binding_hash, request_hash, inputs,
            corpus_file_hash, audit_file_hash, self.artifact_root,
        )
        with self.database.transaction() as connection:
            current = self._binding(binding.workflow_run_id)
            if current != binding:
                raise WorkflowReportHandoffError(
                    "workflow report source changed while freezing inputs"
                )
            self._assert_report_inputs(current, inputs)
            updated = connection.execute(
                """UPDATE workflow_report_handoffs
                   SET status = 'complete', bundle_id = ?, bundle_hash = ?,
                       corpus_snapshot_hash = ?, corpus_file_sha256 = ?,
                       corpus_snapshot_path = ?, search_audit_pack_hash = ?,
                       search_audit_file_sha256 = ?, search_audit_path = ?,
                       prepared_at = ?, updated_at = CURRENT_TIMESTAMP
                   WHERE handoff_id = ? AND status = 'preparing'
                     AND workflow_binding_hash = ? AND request_hash = ?""",
                (
                    inputs.bundle_id, bundle_hash,
                    inputs.corpus_snapshot["snapshot_hash"], corpus_file_hash,
                    str(inputs.corpus_snapshot_path.resolve()),
                    inputs.search_audit["pack_hash"], audit_file_hash,
                    str(inputs.search_audit_path.resolve()), _utc_now(),
                    handoff_id, binding.binding_hash, request_hash,
                ),
            )
        return bundle_hash, updated.rowcount == 1

    def _assert_row(
        self,
        row: Any,
        binding: _WorkflowBinding,
        request: WorkflowReportHandoffRequest,
        request_hash: str,
    ) -> None:
        expected = {
            "workflow_run_id": binding.workflow_run_id,
            "workflow_manifest_hash": binding.manifest_hash,
            "workflow_binding_hash": binding.binding_hash,
            "request_hash": request_hash,
            "crawl_run_id": binding.crawl_run_id,
            "filter_run_id": binding.filter_run_id,
            "download_run_id": binding.download_run_id,
            "stage4_run_id": binding.stage4_run_id,
            "recent_cutoff": request.recent_cutoff,
            "input_created_at": request.created_at,
            "include_needs_review": int(binding.include_needs_review),
            "artifact_root": str(self.artifact_root),
            "output_root": str(self.output_root),
        }
        if any(row[key] != value for key, value in expected.items()):
            raise WorkflowReportHandoffError("persisted handoff binding has drifted")

    def _load(self, row: Any) -> ReportInputResult:
        bundle_id = str(row["bundle_id"])
        directory = self.output_root / "reports" / "inputs" / bundle_id
        corpus_path = Path(str(row["corpus_snapshot_path"]))
        audit_path = Path(str(row["search_audit_path"]))
        if (
            corpus_path != (directory / "CORPUS_SNAPSHOT.json").resolve()
            or audit_path != (directory / "SEARCH_AUDIT.json").resolve()
            or _file_hash(corpus_path) != row["corpus_file_sha256"]
            or _file_hash(audit_path) != row["search_audit_file_sha256"]
        ):
            raise WorkflowReportHandoffError("persisted report input file hash has drifted")
        corpus = _object(corpus_path.read_bytes(), "corpus snapshot")
        audit = _object(audit_path.read_bytes(), "search audit")
        if (
            corpus.get("snapshot_hash") != row["corpus_snapshot_hash"]
            or audit.get("pack_hash") != row["search_audit_pack_hash"]
            or audit.get("corpus_snapshot_hash") != corpus.get("snapshot_hash")
        ):
            raise WorkflowReportHandoffError("persisted report input hash has drifted")
        result = ReportInputResult(
            bundle_id, directory, corpus_path, audit_path, corpus, audit, False
        )
        expected = _bundle_hash(
            str(row["workflow_binding_hash"]), str(row["request_hash"]), result,
            str(row["corpus_file_sha256"]), str(row["search_audit_file_sha256"]),
            Path(str(row["artifact_root"])),
        )
        if expected != row["bundle_hash"]:
            raise WorkflowReportHandoffError("persisted report bundle hash has drifted")
        return result

    def _row(self, handoff_id: str):
        return self.database.connection.execute(
            "SELECT * FROM workflow_report_handoffs WHERE handoff_id = ?",
            (handoff_id,),
        ).fetchone()

    def _result(
        self,
        handoff_id: str,
        binding: _WorkflowBinding,
        request_hash: str,
        inputs: ReportInputResult,
        *,
        status: str,
        persisted: bool,
        resumed: bool,
        write_performed: bool,
        bundle_hash: str | None = None,
        corpus_file_hash: str | None = None,
        audit_file_hash: str | None = None,
    ) -> WorkflowReportHandoffResult:
        resolved_corpus_file_hash = corpus_file_hash or _result_file_hash(
            inputs.corpus_snapshot_path, inputs.corpus_snapshot
        )
        resolved_audit_file_hash = audit_file_hash or _result_file_hash(
            inputs.search_audit_path, inputs.search_audit
        )
        resolved_hash = bundle_hash or _bundle_hash(
            binding.binding_hash, request_hash, inputs,
            resolved_corpus_file_hash,
            resolved_audit_file_hash,
            self.artifact_root,
        )
        return WorkflowReportHandoffResult(
            handoff_id, binding.workflow_run_id, binding.manifest_hash,
            binding.binding_hash, request_hash, binding.crawl_run_id,
            binding.filter_run_id, binding.download_run_id, binding.stage4_run_id,
            binding.include_needs_review, self.artifact_root, resolved_hash, inputs,
            resolved_corpus_file_hash, resolved_audit_file_hash, status,
            persisted, resumed, write_performed,
        )


def assert_report_handoff_runtime(
    database: Database,
    plan: Mapping[str, Any],
    corpus_snapshot: Mapping[str, Any],
    search_audit: Mapping[str, Any],
    *,
    workflow_run_id: str | None = None,
    report_run_id: str | None = None,
    runtime_artifact_root: Path | None = None,
    require_persisted_workflow: bool = True,
) -> bool:
    """Verify a handoff-bound plan at the actual standalone workflow boundary."""
    binding = plan.get("workflow_handoff")
    if binding is None:
        return False
    if not isinstance(binding, Mapping) or set(binding) != {
        "handoff_id",
        "workflow_binding_hash",
        "bundle_hash",
    }:
        raise WorkflowReportHandoffError("ReportPlan workflow handoff is malformed")
    handoff_id = binding.get("handoff_id")
    if not isinstance(handoff_id, str) or not handoff_id:
        raise WorkflowReportHandoffError("ReportPlan workflow handoff ID is invalid")
    row = database.connection.execute(
        """SELECT h.*, e.report_plan_id, e.report_plan_hash,
                  e.report_plan_path, e.report_plan_file_sha256,
                  e.report_workflow_id, e.report_workflow_run_id,
                  e.report_manifest_hash, e.report_manifest_json,
                  e.report_manifest_path
           FROM workflow_report_handoffs h
           LEFT JOIN workflow_report_executions e ON e.handoff_id = h.handoff_id
           WHERE h.handoff_id = ?""",
        (handoff_id,),
    ).fetchone()
    if row is None or row["status"] != "complete" or row["report_plan_id"] is None:
        raise WorkflowReportHandoffError(
            "ReportPlan handoff has no complete execution binding"
        )
    artifact_root = Path(str(row["artifact_root"])).resolve()
    if (
        runtime_artifact_root is not None
        and Path(runtime_artifact_root).resolve() != artifact_root
    ):
        raise WorkflowReportHandoffError(
            "Report runtime artifact root is not bound to the handoff"
        )
    output_root = Path(str(row["output_root"])).resolve()
    current_source = WorkflowReportHandoffService(
        database, ArtifactStore(artifact_root), output_root
    )._binding(str(row["workflow_run_id"]))
    if (
        current_source.binding_hash != row["workflow_binding_hash"]
        or current_source.manifest_hash != row["workflow_manifest_hash"]
        or current_source.crawl_run_id != row["crawl_run_id"]
        or current_source.filter_run_id != row["filter_run_id"]
        or current_source.download_run_id != row["download_run_id"]
        or current_source.stage4_run_id != row["stage4_run_id"]
    ):
        raise WorkflowReportHandoffError(
            "source workflow has drifted after the Report handoff"
        )
    expected_binding = {
        "handoff_id": handoff_id,
        "workflow_binding_hash": str(row["workflow_binding_hash"]),
        "bundle_hash": str(row["bundle_hash"]),
    }
    if dict(binding) != expected_binding:
        raise WorkflowReportHandoffError("ReportPlan workflow handoff has drifted")
    expected_bundle_hash = content_hash({
        "schema_version": "1",
        "workflow_binding_hash": row["workflow_binding_hash"],
        "request_hash": row["request_hash"],
        "bundle_id": row["bundle_id"],
        "corpus_snapshot_hash": row["corpus_snapshot_hash"],
        "corpus_file_sha256": row["corpus_file_sha256"],
        "search_audit_pack_hash": row["search_audit_pack_hash"],
        "search_audit_file_sha256": row["search_audit_file_sha256"],
        "artifact_root": str(artifact_root),
    })
    if (
        expected_bundle_hash != row["bundle_hash"]
        or plan.get("plan_id") != row["report_plan_id"]
        or plan.get("plan_hash") != row["report_plan_hash"]
        or corpus_snapshot.get("snapshot_hash") != row["corpus_snapshot_hash"]
        or search_audit.get("pack_hash") != row["search_audit_pack_hash"]
        or search_audit.get("corpus_snapshot_hash")
        != corpus_snapshot.get("snapshot_hash")
        or sha256(canonical_json(dict(corpus_snapshot))).hexdigest()
        != row["corpus_file_sha256"]
        or sha256(canonical_json(dict(search_audit))).hexdigest()
        != row["search_audit_file_sha256"]
    ):
        raise WorkflowReportHandoffError(
            "ReportPlan runtime inputs have drifted from the workflow handoff"
        )
    plan_path = Path(str(row["report_plan_path"]))
    if (
        _file_hash(plan_path) != row["report_plan_file_sha256"]
        or dict(_object(plan_path.read_bytes(), "approved ReportPlan")) != dict(plan)
    ):
        raise WorkflowReportHandoffError("approved ReportPlan file has drifted")
    if workflow_run_id is not None and workflow_run_id != row["report_workflow_run_id"]:
        raise WorkflowReportHandoffError(
            "Report workflow run ID is not bound to the approved handoff"
        )
    if report_run_id is not None:
        manifest = _object(row["report_manifest_json"], "Report workflow manifest")
        steps = manifest.get("steps")
        if not (
            isinstance(steps, list)
            and len(steps) == 1
            and isinstance(steps[0], Mapping)
            and report_run_id
            == f"{row['report_workflow_run_id']}:{steps[0].get('id')}"
        ):
            raise WorkflowReportHandoffError(
                "Report child run ID is not bound to the approved handoff"
            )
        artifact_ref = steps[0].get("artifact_root")
        manifest_root = Path(str(row["report_manifest_path"])).parent.resolve()
        if not (
            isinstance(artifact_ref, Mapping)
            and set(artifact_ref) == {"path"}
            and isinstance(artifact_ref.get("path"), str)
            and (manifest_root / str(artifact_ref["path"])).resolve()
            == artifact_root
        ):
            raise WorkflowReportHandoffError(
                "Report workflow artifact root is not bound to the handoff"
            )
    if require_persisted_workflow:
        workflow = database.connection.execute(
            """SELECT manifest_hash, manifest_json FROM workflow_runs
               WHERE workflow_run_id = ?""",
            (row["report_workflow_run_id"],),
        ).fetchone()
        if workflow is None or (
            workflow["manifest_hash"], workflow["manifest_json"]
        ) != (row["report_manifest_hash"], row["report_manifest_json"]):
            raise WorkflowReportHandoffError(
                "persisted Report workflow manifest is not handoff-bound"
            )
    return True


def _request_hash(
    binding: _WorkflowBinding,
    request: WorkflowReportHandoffRequest,
    artifact_root: Path,
    output_root: Path,
) -> str:
    return content_hash({
        "schema_version": "1",
        "workflow_run_id": binding.workflow_run_id,
        "workflow_binding_hash": binding.binding_hash,
        "recent_cutoff": request.recent_cutoff,
        "created_at": request.created_at,
        "include_needs_review": binding.include_needs_review,
        "artifact_root": str(artifact_root),
        "output_root": str(output_root),
    })


def _plan_binding(handoff: WorkflowReportHandoffResult) -> dict[str, str]:
    return {
        "handoff_id": handoff.handoff_id,
        "workflow_binding_hash": handoff.workflow_binding_hash,
        "bundle_hash": handoff.bundle_hash,
    }


def _paper_ids(
    payload: Mapping[str, Any], key: str, stage: str
) -> tuple[str, ...]:
    values = payload.get(key)
    if not isinstance(values, list) or not all(
        isinstance(value, str) and value for value in values
    ):
        raise WorkflowReportHandoffError(
            f"workflow {stage} has invalid {key}"
        )
    if len(set(values)) != len(values):
        raise WorkflowReportHandoffError(
            f"workflow {stage} has duplicate {key}"
        )
    return tuple(values)


def _decision_map(payload: Mapping[str, Any]) -> dict[str, str]:
    values = payload.get("decisions")
    allowed = {"relevant", "irrelevant", "needs_review"}
    if not isinstance(values, Mapping) or not all(
        isinstance(paper_id, str)
        and paper_id
        and isinstance(status, str)
        and status in allowed
        for paper_id, status in values.items()
    ):
        raise WorkflowReportHandoffError(
            "workflow Filter decisions are malformed"
        )
    return {str(paper_id): str(status) for paper_id, status in values.items()}


def _report_workflow_manifest(
    *,
    workflow_id: str,
    step_id: str,
    source_path: Path,
    config_path: Path,
    plan_path: Path,
    corpus_path: Path,
    audit_path: Path,
    processing_grants_path: Path | None,
    previous_report_run_id: str | None,
    policy_path: Path | None,
    artifact_root: Path,
    planned_files: Mapping[Path, bytes] | None = None,
) -> WorkflowManifest:
    if not workflow_id:
        raise WorkflowReportHandoffError("report workflow_id is required")
    root = source_path.parent.resolve()
    planned = planned_files or {}
    config = _workflow_ref(config_path, root, "workflow config", planned.get(config_path))
    report = ReportStep(
        step_id,
        _workflow_ref(
            plan_path, root, "approved ReportPlan", planned.get(plan_path)
        ),
        _workflow_ref(
            corpus_path, root, "corpus snapshot", planned.get(corpus_path)
        ),
        _workflow_ref(
            audit_path, root, "search audit", planned.get(audit_path)
        ),
        _workflow_ref(
            processing_grants_path,
            root,
            "processing grants",
            planned.get(processing_grants_path),
        )
        if processing_grants_path is not None
        else None,
        previous_report_run_id,
        _workflow_ref(
            policy_path, root, "report policy", planned.get(policy_path)
        )
        if policy_path is not None
        else None,
        _workflow_directory_ref(artifact_root, root, "report artifact root"),
    )
    manifest = WorkflowManifest(
        workflow_id,
        config,
        (report,),
        source_path,
        "2",
    )
    if not planned_files:
        manifest.verify_files()
    return manifest


def _workflow_ref(
    path: Path,
    root: Path,
    label: str,
    planned_payload: bytes | None = None,
) -> FileRef:
    resolved = path.resolve()
    if not resolved.is_file() and planned_payload is None:
        raise WorkflowReportHandoffError(f"{label} is unavailable: {resolved}")
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise WorkflowReportHandoffError(
            f"{label} must be inside the Report workflow manifest directory"
        ) from error
    if not relative.parts:
        raise WorkflowReportHandoffError(f"{label} has an invalid workflow path")
    digest = (
        _file_hash(resolved)
        if resolved.is_file()
        else sha256(planned_payload or b"").hexdigest()
    )
    if planned_payload is not None and digest != sha256(planned_payload).hexdigest():
        raise WorkflowReportHandoffError(f"{label} content has drifted: {resolved}")
    return FileRef(relative.as_posix(), digest, resolved)


def _workflow_directory_ref(
    path: Path, root: Path, label: str
) -> DirectoryRef:
    resolved = path.resolve()
    if not resolved.is_dir():
        raise WorkflowReportHandoffError(f"{label} is unavailable: {resolved}")
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise WorkflowReportHandoffError(
            f"{label} must be inside the Report workflow manifest directory"
        ) from error
    return DirectoryRef(relative.as_posix(), resolved)


def _write_immutable(path: Path, payload: bytes) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise WorkflowReportHandoffError(
                f"workflow-bound file is immutable: {path}"
            )
        return False
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as temporary:
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        try:
            os.link(temporary_path, path)
            wrote = True
        except FileExistsError:
            if not path.is_file() or path.read_bytes() != payload:
                raise WorkflowReportHandoffError(
                    f"workflow-bound file is immutable: {path}"
                )
            wrote = False
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return wrote
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _write_report_plan_bundle(
    output_root: Path,
    plan: Mapping[str, Any],
    corpus: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> bool:
    directory = ReportPlanStore(output_root).directory(str(plan["plan_id"]))
    documents = (
        (directory / "REPORT_PLAN.json", plan),
        (directory / "CORPUS_SNAPSHOT.json", corpus),
        (directory / "SEARCH_AUDIT.json", audit),
    )
    wrote = False
    for path, document in documents:
        wrote = (
            _write_immutable(path.resolve(), canonical_json(dict(document)))
            or wrote
        )
    return wrote


def _write_latest_report_plan(
    output_root: Path, plan: Mapping[str, Any]
) -> None:
    path = ReportPlanStore(output_root).latest_path.resolve()
    payload = canonical_json({
        "plan_id": plan["plan_id"],
        "plan_hash": plan["plan_hash"],
    })
    if path.is_file() and path.read_bytes() == payload:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as temporary:
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _bundle_hash(
    binding_hash: str,
    request_hash: str,
    inputs: ReportInputResult,
    corpus_file_hash: str,
    audit_file_hash: str,
    artifact_root: Path,
) -> str:
    return content_hash({
        "schema_version": "1",
        "workflow_binding_hash": binding_hash,
        "request_hash": request_hash,
        "bundle_id": inputs.bundle_id,
        "corpus_snapshot_hash": inputs.corpus_snapshot["snapshot_hash"],
        "corpus_file_sha256": corpus_file_hash,
        "search_audit_pack_hash": inputs.search_audit["pack_hash"],
        "search_audit_file_sha256": audit_file_hash,
        "artifact_root": str(artifact_root.resolve()),
    })


def _string(value: Mapping[str, Any], key: str, stage: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise WorkflowReportHandoffError(f"workflow {stage} has an invalid {key}")
    return item


def _object(value: object, name: str) -> Mapping[str, Any]:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise WorkflowReportHandoffError(f"{name} is not valid JSON") from error
    if not isinstance(value, Mapping):
        raise WorkflowReportHandoffError(f"{name} must be a JSON object")
    return value


def _file_hash(path: Path) -> str:
    try:
        return sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise WorkflowReportHandoffError(f"report input is unavailable: {path}") from error


def _result_file_hash(path: Path, document: Mapping[str, Any]) -> str:
    return _file_hash(path) if path.is_file() else sha256(canonical_json(dict(document))).hexdigest()


def _hash(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
