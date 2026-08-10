"""Typed workflow adapters for the five persisted pipeline stages.

The workflow orchestrator owns workflow leases and checkpoints.  This module
only translates already-validated :mod:`paper_agent.workflow` specs into the
existing service APIs; it does not call the command-line layer or create a
second lease system.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import sysconfig
from typing import Any, TypeVar, cast
from uuid import NAMESPACE_URL, uuid5

from .analysis_cli_service import AnalysisCliService, load_analysis_input_manifest
from .artifacts import ArtifactStore
from .canonical import content_hash
from .config import load_config
from .download_cli_service import Stage3DownloadService, load_provider_terms
from .grants import GrantStore
from .processing import ArtifactProcessingPolicy, ProcessingGate
from .report_artifacts import ReportArtifactStore
from .report_cli_service import load_report_run_bundle
from .report_execution_service import ReportExecutionService
from .report_plan import ReportPlanBundle, assert_report_runtime_matches
from .search_execution import execute_search_plan
from .stage2_commands import filter_database
from .stage2_search import load_stage2_release
from .storage import Database
from .workflow import (
    AnalyzeStep,
    DownloadStep,
    FilterStep,
    ReportStep,
    SearchStep,
    StageIdentity,
    StageKind,
    StageOutcome,
    StepContext,
    StepObservation,
)


TStep = TypeVar("TStep", SearchStep, FilterStep, DownloadStep, AnalyzeStep, ReportStep)


@dataclass(frozen=True, slots=True)
class PaperSelection:
    """An explicit selection of paper IDs or a persisted Stage 2 run."""

    paper_ids: tuple[str, ...] = ()
    filter_run_id: str | None = None

    def __post_init__(self) -> None:
        if bool(self.paper_ids) == bool(self.filter_run_id):
            raise ValueError("selection must contain exactly one of paper_ids or filter_run_id")
        if len(set(self.paper_ids)) != len(self.paper_ids) or any(not item for item in self.paper_ids):
            raise ValueError("selection paper_ids must be unique non-empty strings")


class _WorkflowAdapter:
    """Shared identity and read-only run observation helpers."""

    expected_type: type[Any]
    stage: StageKind

    def validate(self, context: StepContext, spec: TStep) -> StageIdentity:
        self._validate_context(context, spec)
        for reference in spec.file_refs():
            reference.verify()
        if context.dry_run:
            # A workflow dry-run is a real preflight, not merely a manifest
            # checksum.  Keep this side-effect-free: stage services can have
            # useful dry-run modes, but some also probe providers or prepare
            # queues, so adapters validate their frozen local inputs here.
            self._validate_dry_inputs(context, spec)
        return StageIdentity(content_hash({
            "stage": self.stage.value,
            "child_run_id": context.child_run_id,
            "config_sha256": _file_sha256(context.config_path),
            "spec": spec.document(),
        }))

    def _validate_dry_inputs(self, context: StepContext, spec: TStep) -> None:
        """Parse the stage's local, frozen input contract without dispatching."""
        del context, spec

    def _validate_context(self, context: StepContext, spec: TStep) -> None:
        if not isinstance(spec, self.expected_type):
            raise TypeError(f"{self.stage.value} adapter received {type(spec).__name__}")
        expected = f"{context.workflow_run_id}:{spec.step_id}"
        if context.child_run_id != expected:
            raise ValueError("workflow child_run_id must be fixed to workflow_run_id:step_id")
        if not context.config_path.is_file():
            raise FileNotFoundError(f"workflow config is unavailable: {context.config_path}")

    @staticmethod
    def _observe_pipeline(
        context: StepContext, expected_stage: str, run_id: str | None = None,
    ) -> StepObservation:
        if not context.database_path.is_file():
            return StepObservation.PENDING
        try:
            with Database(context.database_path, read_only=True) as database:
                row = database.connection.execute(
                    "SELECT stage, status FROM pipeline_runs WHERE run_id = ?",
                    (run_id or context.child_run_id,),
                ).fetchone()
        except (OSError, sqlite3.Error) as error:
            if context.dry_run:
                raise ValueError(
                    "workflow dry-run cannot inspect the existing database state"
                ) from error
            return StepObservation.PENDING
        if row is None:
            return StepObservation.PENDING
        if row["stage"] != expected_stage:
            return StepObservation.UNCERTAIN_TERMINAL
        return _run_observation(str(row["status"]))


class SearchStageAdapter(_WorkflowAdapter):
    expected_type = SearchStep
    stage = StageKind.SEARCH

    def __init__(
        self,
        runner: Callable[..., tuple[Any, str, str]] = execute_search_plan,
    ) -> None:
        self.runner = runner

    def _validate_dry_inputs(self, context: StepContext, spec: SearchStep) -> None:
        # ``load_stage2_release`` also validates the approved QueryPlan and
        # every local release-bundle reference.  It performs no model request.
        load_config(context.config_path)
        plan = _json_object(spec.plan.resolved_path, "search plan")
        load_stage2_release(spec.stage2_release.resolved_path, plan)
        for snapshot in spec.snapshots:
            _json_object(snapshot.file.resolved_path, f"search snapshot {snapshot.provider}")

    def observe(
        self, context: StepContext, spec: SearchStep, identity: StageIdentity
    ) -> StepObservation:
        del spec, identity
        return self._observe_pipeline(context, "stage-1")

    def execute(
        self, context: StepContext, spec: SearchStep, identity: StageIdentity
    ) -> StageOutcome:
        del identity
        plan = _json_object(spec.plan.resolved_path, "search plan")
        result, run_id, crawl_run_id = self.runner(
            plan,
            context.database_path,
            run_id=context.child_run_id,
            snapshot_paths={item.provider: item.file.resolved_path for item in spec.snapshots},
            stage2_release_path=spec.stage2_release.resolved_path,
            historical_replay=spec.historical_replay,
        )
        status = _outcome_status(str(result.status))
        return StageOutcome(status, {
            "run_id": run_id,
            "crawl_run_id": crawl_run_id,
            "pipeline_status": result.status,
            "paper_ids": list(result.paper_ids),
        })


class FilterStageAdapter(_WorkflowAdapter):
    expected_type = FilterStep
    stage = StageKind.FILTER

    def __init__(self, runner: Callable[..., Mapping[str, Any]] = filter_database) -> None:
        self.runner = runner

    def _validate_dry_inputs(self, context: StepContext, spec: FilterStep) -> None:
        load_config(context.config_path)
        plan = _json_object(spec.plan.resolved_path, "filter plan")
        load_stage2_release(spec.stage2_release.resolved_path, plan)
        if spec.selection is not None:
            _selection(spec.selection.resolved_path)

    def observe(
        self, context: StepContext, spec: FilterStep, identity: StageIdentity
    ) -> StepObservation:
        del identity
        # ``filter_database`` exposes campaign_id rather than a run ID.  The
        # released screener deterministically derives its first Stage 2 run
        # from that campaign, so it remains safe to observe without replaying
        # a completed filter stage.
        return self._observe_pipeline(context, "stage-2", _filter_run_id(context, spec))

    def execute(
        self, context: StepContext, spec: FilterStep, identity: StageIdentity
    ) -> StageOutcome:
        del identity
        selection = _selection(spec.selection.resolved_path) if spec.selection else None
        if selection is not None and selection.filter_run_id is not None:
            raise ValueError("filter selection must contain paper_ids, not filter_run_id")
        if not context.dry_run:
            # The service opens its own database handle, so ensure a direct
            # adapter invocation has the same migrated schema as the CLI path.
            with Database(context.database_path) as database:
                database.migrate()
        result = self.runner(
            plan_path=spec.plan.resolved_path,
            release_path=spec.stage2_release.resolved_path,
            database_path=context.database_path,
            campaign_id=context.child_run_id,
            paper_ids=selection.paper_ids if selection else (),
            dry_run=context.dry_run,
        )
        return StageOutcome(_outcome_status(str(result["status"])), dict(result))


DownloadServiceFactory = Callable[[Database, Mapping[str, Any], Path, Path, Mapping[str, Any] | None], Any]


class DownloadStageAdapter(_WorkflowAdapter):
    expected_type = DownloadStep
    stage = StageKind.DOWNLOAD

    def __init__(self, service_factory: DownloadServiceFactory | None = None) -> None:
        self.service_factory = service_factory or _download_service

    def _validate_dry_inputs(self, context: StepContext, spec: DownloadStep) -> None:
        # Constructing the Stage 3 service would prepare provider state.  The
        # strict config parser plus typed selection/terms parsers cover its
        # frozen local inputs without touching a provider or SQLite.
        load_config(context.config_path)
        _selection(spec.selection.resolved_path)
        if spec.provider_terms is not None:
            load_provider_terms(spec.provider_terms.resolved_path)

    def observe(
        self, context: StepContext, spec: DownloadStep, identity: StageIdentity
    ) -> StepObservation:
        del spec, identity
        return self._observe_pipeline(context, "stage-3-download")

    def execute(
        self, context: StepContext, spec: DownloadStep, identity: StageIdentity
    ) -> StageOutcome:
        del identity
        selection = _selection(spec.selection.resolved_path)
        config = load_config(context.config_path)
        terms = load_provider_terms(spec.provider_terms.resolved_path) if spec.provider_terms else None
        with _database(context) as database:
            service = self.service_factory(
                database, config, context.config_path.parent, _artifact_root(context), terms,
            )
            result = service.run(
                paper_ids=selection.paper_ids,
                filter_run_id=selection.filter_run_id,
                authorization_grant_id=spec.authorization_grant_id,
                run_id=context.child_run_id,
                dry_run=context.dry_run,
            )
        return StageOutcome(_outcome_status(str(result.status)), {
            "run_id": result.run_id,
            "paper_ids": list(result.paper_ids),
            "stage_status": result.status,
            "dry_run": result.dry_run,
        })


AnalysisServiceFactory = Callable[[Database, ArtifactStore, ArtifactProcessingPolicy], Any]


class AnalyzeStageAdapter(_WorkflowAdapter):
    expected_type = AnalyzeStep
    stage = StageKind.ANALYZE

    def __init__(self, service_factory: AnalysisServiceFactory | None = None) -> None:
        self.service_factory = service_factory or _analysis_service

    def _validate_dry_inputs(self, context: StepContext, spec: AnalyzeStep) -> None:
        config = load_config(context.config_path)
        load_analysis_input_manifest(spec.selection.resolved_path)
        ArtifactProcessingPolicy.load(_policy_path(
            spec.policy.resolved_path if spec.policy else None,
            config,
            context.config_path,
            "analysis",
        ))

    def observe(
        self, context: StepContext, spec: AnalyzeStep, identity: StageIdentity
    ) -> StepObservation:
        del spec, identity
        return self._observe_pipeline(context, "stage4")

    def execute(
        self, context: StepContext, spec: AnalyzeStep, identity: StageIdentity
    ) -> StageOutcome:
        del identity
        manifest = load_analysis_input_manifest(spec.selection.resolved_path)
        config = load_config(context.config_path)
        policy = ArtifactProcessingPolicy.load(_policy_path(
            spec.policy.resolved_path if spec.policy else None, config, context.config_path, "analysis",
        ))
        with _database(context) as database:
            service = self.service_factory(database, ArtifactStore(_artifact_root(context)), policy)
            result = service.run(
                context.child_run_id,
                manifest,
                processing_grant_id=spec.processing_grant_id,
                dry_run=context.dry_run,
            )
        pipeline_status = _pipeline_status(context, "stage4", result.run_id)
        status = "validated" if result.dry_run else (
            "failed" if pipeline_status == "failed" else
            "complete" if result.result is not None and all(item.status == "complete" for item in result.result.papers) else "incomplete"
        )
        outcome_status = (
            "uncertain_terminal" if pipeline_status == "failed" else _outcome_status(status)
        )
        return StageOutcome(outcome_status, {
            "run_id": result.run_id,
            "paper_ids": list(result.selected_paper_ids),
            "input_scopes": list(result.input_scopes),
            "stage_status": status,
            "pipeline_status": pipeline_status,
            "dry_run": result.dry_run,
        })


ReportServiceFactory = Callable[[Database, ArtifactStore, ProcessingGate, ReportArtifactStore], Any]


class ReportStageAdapter(_WorkflowAdapter):
    expected_type = ReportStep
    stage = StageKind.REPORT

    def __init__(self, service_factory: ReportServiceFactory | None = None) -> None:
        self.service_factory = service_factory or _report_service

    def _validate_dry_inputs(self, context: StepContext, spec: ReportStep) -> None:
        config = load_config(context.config_path)
        bundle = _report_bundle(spec)
        assert_report_runtime_matches(
            bundle.plan,
            bundle.plan,
            corpus_snapshot=bundle.corpus_snapshot,
            search_audit_pack=bundle.search_audit,
        )
        ArtifactProcessingPolicy.load(_policy_path(
            spec.policy.resolved_path if spec.policy else None,
            config,
            context.config_path,
            "summary",
        ))
        if spec.processing_grants is not None:
            _grant_mapping(spec.processing_grants.resolved_path)
        if spec.previous_report_run_id is not None:
            load_report_run_bundle(_report_output_root(config, context.config_path), spec.previous_report_run_id)

    def observe(
        self, context: StepContext, spec: ReportStep, identity: StageIdentity
    ) -> StepObservation:
        del spec, identity
        return self._observe_pipeline(context, "stage4b")

    def execute(
        self, context: StepContext, spec: ReportStep, identity: StageIdentity
    ) -> StageOutcome:
        del identity
        config = load_config(context.config_path)
        bundle = _report_bundle(spec)
        policy = ArtifactProcessingPolicy.load(_policy_path(
            spec.policy.resolved_path if spec.policy else None, config, context.config_path, "summary",
        ))
        root = _report_output_root(config, context.config_path)
        previous = None
        if spec.previous_report_run_id is not None:
            previous = load_report_run_bundle(root, spec.previous_report_run_id).diff_input()
        with Database(context.database_path) as database:
            service = self.service_factory(
                database, ArtifactStore(root), ProcessingGate(policy, GrantStore(database)), ReportArtifactStore(root),
            )
            result = service.run(
                context.child_run_id,
                context.child_run_id,
                bundle,
                processing_grants=_grant_mapping(spec.processing_grants.resolved_path)
                if spec.processing_grants else None,
                previous=previous,
                dry_run=context.dry_run,
            )
        return StageOutcome(_outcome_status(str(result.status)), {
            "report_run_id": result.report_run_id,
            "stage_status": result.status,
            "dry_run": result.dry_run,
        })


def default_stage_adapters() -> dict[StageKind, _WorkflowAdapter]:
    """Return the five direct-service adapters used by a workflow runner."""
    return {
        StageKind.SEARCH: SearchStageAdapter(),
        StageKind.FILTER: FilterStageAdapter(),
        StageKind.DOWNLOAD: DownloadStageAdapter(),
        StageKind.ANALYZE: AnalyzeStageAdapter(),
        StageKind.REPORT: ReportStageAdapter(),
    }


def _download_service(
    database: Database, config: Mapping[str, Any], config_root: Path,
    artifact_root: Path, provider_terms: Mapping[str, Any] | None,
) -> Stage3DownloadService:
    return Stage3DownloadService(
        database, config, config_root=config_root, artifact_root=artifact_root,
        provider_terms=provider_terms,
    )


def _analysis_service(
    database: Database, artifact_store: ArtifactStore, policy: ArtifactProcessingPolicy,
) -> AnalysisCliService:
    return AnalysisCliService(database, artifact_store, policy, grants=GrantStore(database))


def _report_service(
    database: Database, artifact_store: ArtifactStore, gate: ProcessingGate,
    report_store: ReportArtifactStore,
) -> ReportExecutionService:
    return ReportExecutionService(database, artifact_store, gate, report_store)


def _artifact_root(context: StepContext) -> Path:
    # This is the established direct-service default: artifacts share the
    # database parent unless a future typed workflow input makes it explicit.
    return context.database_path.parent


def _report_output_root(config: Mapping[str, Any], config_path: Path) -> Path:
    """Resolve the report store exactly as the standalone report command does."""
    try:
        value = Path(str(config["project"]["output_dir"]))
    except (KeyError, TypeError) as error:
        raise ValueError("workflow report stage requires project.output_dir") from error
    return value if value.is_absolute() else config_path.parent / value


def _database(context: StepContext) -> Database:
    """Open a current schema for direct adapter use outside the CLI runner."""
    database = Database(context.database_path, read_only=context.dry_run)
    if not context.dry_run:
        database.migrate()
    return database


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _json_object(path: Path, label: str) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return cast(Mapping[str, Any], value)


def _selection(path: Path) -> PaperSelection:
    value = _json_object(path, "workflow selection")
    if value.get("schema_version") != "1":
        raise ValueError("workflow selection must use schema_version 1")
    keys = set(value)
    if keys == {"schema_version", "paper_ids"}:
        paper_ids = value["paper_ids"]
        if not isinstance(paper_ids, list) or not all(isinstance(item, str) for item in paper_ids):
            raise ValueError("workflow selection paper_ids must be a string list")
        return PaperSelection(tuple(paper_ids))
    if keys == {"schema_version", "filter_run_id"} and isinstance(value["filter_run_id"], str):
        return PaperSelection(filter_run_id=value["filter_run_id"])
    raise ValueError("workflow selection must contain paper_ids or filter_run_id")


def _grant_mapping(path: Path) -> Mapping[str, str]:
    value = _json_object(path, "processing grants")
    if set(value) == {"schema_version", "grants"}:
        if value["schema_version"] != "1":
            raise ValueError("processing grants must use schema_version 1")
        value = value["grants"]
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str)
        and len(key) == 64
        and all(character in "0123456789abcdef" for character in key)
        and isinstance(item, str)
        and item
        for key, item in value.items()
    ):
        raise ValueError("processing grants must map artifact SHA-256 hashes to grant IDs")
    return cast(Mapping[str, str], value)


def _report_bundle(spec: ReportStep) -> ReportPlanBundle:
    return ReportPlanBundle(
        _json_object(spec.plan.resolved_path, "report plan"),
        _json_object(spec.corpus_snapshot.resolved_path, "corpus snapshot"),
        _json_object(spec.search_audit.resolved_path, "search audit"),
    )


def _pipeline_status(context: StepContext, expected_stage: str, run_id: str) -> str | None:
    """Read a child pipeline's terminal state without treating absence as one."""
    if not context.database_path.is_file():
        return None
    try:
        with Database(context.database_path, read_only=True) as database:
            row = database.connection.execute(
                "SELECT stage, status FROM pipeline_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
    except (OSError, sqlite3.Error):
        return None
    if row is None or row["stage"] != expected_stage:
        return None
    return str(row["status"])


def _policy_path(
    explicit: Path | None, config: Mapping[str, Any], config_path: Path, section: str,
) -> Path:
    if explicit is not None:
        return explicit
    try:
        configured = config[section]["remote_model_processing"]["policy_matrix"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"workflow {section} stage requires an explicit processing policy") from error
    path = Path(str(configured))
    if path.is_absolute():
        return path
    configured_path = config_path.parent / path
    repository_path = Path(__file__).resolve().parents[2] / path
    installed_path = Path(sysconfig.get_path("data")) / "share" / "paper-agent" / path
    return next(
        (candidate for candidate in (configured_path, repository_path, installed_path) if candidate.is_file()),
        configured_path,
    )


def _filter_run_id(context: StepContext, spec: FilterStep) -> str:
    del spec
    return f"stage2-{uuid5(NAMESPACE_URL, f'{context.child_run_id}:0').hex}"


def _run_observation(status: str) -> StepObservation:
    return {
        "complete": StepObservation.COMPLETE,
        "running": StepObservation.RUNNING,
        "incomplete": StepObservation.SAFE_TO_RESUME,
        "failed": StepObservation.UNCERTAIN_TERMINAL,
        "cancelled": StepObservation.UNCERTAIN_TERMINAL,
        "draft": StepObservation.PENDING,
        "approved": StepObservation.PENDING,
        "pending": StepObservation.PENDING,
    }.get(status, StepObservation.UNCERTAIN_TERMINAL)


def _outcome_status(status: str) -> str:
    if status in {"complete", "validated"}:
        return "complete"
    if status == "manual_required":
        return "blocked"
    if status == "failed":
        return "failed"
    return "incomplete"
