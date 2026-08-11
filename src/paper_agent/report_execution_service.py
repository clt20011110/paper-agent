"""File/SQLite-facing execution adapter for an approved Stage 4b ReportPlan."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import json
from typing import Any

from .artifacts import ArtifactStore
from .canonical import content_hash
from .processing import ProcessingGate
from .report_artifacts import ReportArtifactStore
from .report_config import ReportRuntimeConfig
from .report_direct import (
    DirectReportBudgetError,
    DirectReportCoordinator,
    DirectReportError,
    DirectReportInvoker,
    DirectReportResult,
)
from .report_plan import (
    ReportPlanBundle,
    ReportPlanDriftError,
    assert_report_runtime_matches,
)
from .report_reduce import FrozenDerivedArtifact
from .reporting import AnalysisRecord, SectionRule
from .storage import Database
from .workflow_report_handoff import assert_report_handoff_runtime


CODEX_BUDGET_ALARM = "report.codex_budget_exhausted"


@dataclass(frozen=True, slots=True)
class ReportCodexBudget:
    calls_reserved: int
    input_tokens_reserved: int
    approved_call_limit: int
    approved_input_token_limit: int


@dataclass(frozen=True, slots=True)
class ReportExecutionResult:
    report_run_id: str
    status: str
    dry_run: bool
    direct: DirectReportResult | None = None
    skipped: bool = False
    codex_budget: ReportCodexBudget | None = None
    alarm_codes: tuple[str, ...] = ()
    error: Mapping[str, str] | None = None


class ReportExecutionService:
    """Construct trusted Stage 4b inputs; coordinators retain all gate logic."""

    def __init__(
        self,
        database: Database,
        artifact_store: ArtifactStore,
        gate: ProcessingGate,
        report_store: ReportArtifactStore,
        *,
        direct_invoker_factory: Callable[[], DirectReportInvoker] | None = None,
        execution_mode: str = "attended",
        runtime_config: ReportRuntimeConfig | None = None,
    ) -> None:
        self.database = database
        self.artifact_store = artifact_store
        self.gate = gate
        self.report_store = report_store
        self.direct_invoker_factory = direct_invoker_factory
        self.execution_mode = execution_mode
        self.runtime_config = runtime_config or ReportRuntimeConfig.defaults()

    def run(
        self,
        report_run_id: str,
        pipeline_run_id: str,
        bundle: ReportPlanBundle,
        *,
        processing_grants: Mapping[str, str] | None = None,
        previous: Mapping[str, Any] | None = None,
        dry_run: bool = False,
        workflow_run_id: str | None = None,
    ) -> ReportExecutionResult:
        if (
            bundle.plan.get("workflow_handoff") is not None
            and workflow_run_id is None
        ):
            raise ReportPlanDriftError(
                "handoff-bound ReportPlan requires its registered standalone workflow"
            )
        if not self.runtime_config.enabled:
            return self._result(report_run_id, "complete", dry_run, skipped=True)
        self.runtime_config.validate_for_run(
            bundle.plan, execution_mode=self.execution_mode
        )
        _validate_processing_grants(processing_grants)
        assert_report_runtime_matches(
            bundle.plan, bundle.plan,
            corpus_snapshot=bundle.corpus_snapshot,
            search_audit_pack=bundle.search_audit,
        )
        if workflow_run_id is not None:
            assert_report_handoff_runtime(
                self.database,
                bundle.plan,
                bundle.corpus_snapshot,
                bundle.search_audit,
                workflow_run_id=workflow_run_id,
                report_run_id=report_run_id,
                runtime_artifact_root=self.artifact_store.root,
            )
        analyses, artifacts = self._analyses(bundle)
        direct_options: dict[str, Any] = {
            "resources": self.runtime_config.resources,
            "execution_mode": self.execution_mode,
        }
        if self.direct_invoker_factory is not None:
            direct_options["invoker_factory"] = self.direct_invoker_factory
        coordinator = DirectReportCoordinator(
            self.database,
            self.artifact_store,
            self.gate,
            self.report_store,
            analyses,
            artifacts,
            self._sections(bundle.plan),
            self._memberships(bundle.plan),
            **direct_options,
        )
        if dry_run:
            try:
                coordinator.preflight(report_run_id, bundle, previous=previous)
            except DirectReportError as error:
                budget_error = isinstance(error, DirectReportBudgetError)
                return self._result(
                    report_run_id,
                    "incomplete",
                    True,
                    codex_budget=self._codex_budget(report_run_id, bundle.plan),
                    alarm_codes=(CODEX_BUDGET_ALARM,) if budget_error else (),
                    error={
                        "type": type(error).__name__,
                        "message": str(error),
                        **(
                            {"event_code": CODEX_BUDGET_ALARM}
                            if budget_error
                            else {}
                        ),
                    },
                )
            return self._result(
                report_run_id,
                "validated",
                True,
                codex_budget=self._codex_budget(report_run_id, bundle.plan),
            )

        direct = coordinator.run(
            report_run_id,
            pipeline_run_id,
            bundle,
            processing_grants=processing_grants or {},
            previous=previous,
        )
        return self._result(
            report_run_id,
            direct.status,
            False,
            direct=direct,
            codex_budget=self._codex_budget(report_run_id, bundle.plan),
            alarm_codes=(CODEX_BUDGET_ALARM,) if direct.budget_exhausted else (),
            error=(
                {
                    "type": (
                        "DirectReportBudgetError"
                        if direct.budget_exhausted
                        else "DirectReportError"
                    ),
                    "message": direct.error,
                    **(
                        {"event_code": CODEX_BUDGET_ALARM}
                        if direct.budget_exhausted
                        else {}
                    ),
                }
                if direct.error is not None
                else None
            ),
        )

    @staticmethod
    def _result(
        report_run_id: str,
        status: str,
        dry_run: bool,
        *,
        direct: DirectReportResult | None = None,
        skipped: bool = False,
        codex_budget: ReportCodexBudget | None = None,
        alarm_codes: tuple[str, ...] = (),
        error: Mapping[str, str] | None = None,
    ) -> ReportExecutionResult:
        return ReportExecutionResult(
            report_run_id=report_run_id,
            status=status,
            dry_run=dry_run,
            direct=direct,
            skipped=skipped,
            codex_budget=codex_budget,
            alarm_codes=alarm_codes,
            error=error,
        )

    def _codex_budget(
        self, report_run_id: str, plan: Mapping[str, Any]
    ) -> ReportCodexBudget:
        row = self.database.connection.execute(
            """SELECT COALESCE(budget_calls_reserved, 0),
                      COALESCE(budget_tokens_reserved, 0)
                 FROM report_one_shot_runs WHERE report_run_id = ?""",
            (report_run_id,),
        ).fetchone()
        budget = plan["budget"]
        return ReportCodexBudget(
            int(row[0]) if row is not None else 0,
            int(row[1]) if row is not None else 0,
            int(budget["max_sol_calls"]),
            int(budget["max_input_tokens"]),
        )

    def _analyses(self, bundle: ReportPlanBundle) -> tuple[tuple[AnalysisRecord, ...], tuple[FrozenDerivedArtifact, ...]]:
        records: list[AnalysisRecord] = []
        artifacts: list[FrozenDerivedArtifact] = []
        for paper in bundle.corpus_snapshot["papers"]:
            if paper["input_scope"] == "missing":
                raise ValueError("Report execution requires a persisted Stage 4 analysis for every frozen paper")
            payload = self.artifact_store.read_bytes(str(paper["analysis_artifact_hash"]))
            document = _object(payload, "persisted Stage 4 analysis")
            labels = document["labels"]
            classifications = _classifications(labels, paper)
            records.append(AnalysisRecord(
                paper_id=str(paper["paper_id"]), analysis_run_id=str(paper["analysis_run_id"]),
                analysis_hash=str(paper["analysis_artifact_hash"]), input_scope=str(paper["input_scope"]),
                input_tokens=int(paper["analysis_input_tokens"]), classifications=classifications,
                evidence_units=tuple(document["evidence_units"]),
            ))
            lineage = tuple(sorted(str(item) for item in paper["lineage_hashes"]))
            artifacts.append(FrozenDerivedArtifact(
                artifact_hash=str(paper["analysis_artifact_hash"]), payload=payload, artifact_kind="analysis",
                input_scope=str(paper["input_scope"]), license=None, access_basis="unknown",
                lineage_hash=content_hash(lineage), source_lineage_hashes=lineage,
                source_paper_ids=(str(paper["paper_id"]),), paper_id=str(paper["paper_id"]),
                mode=self.execution_mode,
            ))
        return tuple(records), tuple(artifacts)

    @staticmethod
    def _sections(plan: Mapping[str, Any]) -> tuple[SectionRule, ...]:
        return tuple(SectionRule(
            str(item["id"]), frozenset(str(value) for value in item["subquestion_ids"]),
            frozenset(str(value) for value in item["allowed_evidence_levels"]),
        ) for item in plan["sections"])

    @staticmethod
    def _memberships(plan: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
        return {str(item["paper_id"]): tuple(str(value) for value in item["section_ids"])
                for item in plan["paper_memberships"]}


def _object(payload: bytes, name: str) -> Mapping[str, Any]:
    value = json.loads(payload)
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _validate_processing_grants(grants: Mapping[str, str] | None) -> None:
    """Reject malformed authorization bindings before a dry-run returns success."""
    if grants is None:
        return
    if not all(
        isinstance(artifact_hash, str)
        and len(artifact_hash) == 64
        and all(character in "0123456789abcdef" for character in artifact_hash)
        and isinstance(grant_id, str)
        and grant_id
        for artifact_hash, grant_id in grants.items()
    ):
        raise ValueError("processing grants must map artifact SHA-256 hashes to grant IDs")


def _values(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        value = (value,)
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(sorted(set(str(item) for item in value if str(item))))


def _classifications(labels: Mapping[str, Any], paper: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    result = {axis: _values(labels.get(axis)) for axis in (
        "subquestion", "theme", "method_family", "task", "dataset", "benchmark", "evidence_type",
    )}
    result = {axis: values for axis, values in result.items() if values}
    result["publication_status"] = (str(paper["publication_status"]),)
    result["study_setting"] = (str(paper["study_setting"]),)
    year = paper.get("publication_year") or (str(paper.get("publication_date") or "")[:4])
    if year:
        result["time"] = (str(year),)
    venue = paper.get("venue_id") or paper.get("venue_name")
    if venue:
        result["venue"] = (str(venue),)
    return result
