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
from .report_audit import (
    ReportAuditBudgetError,
    ReportAuditCoordinator,
    ReportAuditResult,
    ReportBundle,
    SolInvoker as AuditInvoker,
)
from .report_config import ReportRuntimeConfig
from .report_plan import (
    ReportPlanBundle,
    ReportPlanDriftError,
    assert_report_runtime_matches,
    persist_approved_report_plan,
)
from .report_reduce import (
    FrozenDerivedArtifact,
    ReportReduceResult,
    SolBudgetError,
    SolInvoker as ReduceInvoker,
    SolReduceCoordinator,
)
from .reporting import (
    AnalysisRecord,
    BudgetExceeded,
    ReportPlanner,
    SectionRule,
    derive_comparison_groups,
)
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
    reduce: ReportReduceResult | None = None
    audit: ReportAuditResult | None = None
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
        reduce_invoker_factory: Callable[[], ReduceInvoker] | None = None,
        audit_invoker_factory: Callable[[], AuditInvoker] | None = None,
        execution_mode: str = "attended",
        runtime_config: ReportRuntimeConfig | None = None,
    ) -> None:
        self.database = database
        self.artifact_store = artifact_store
        self.gate = gate
        self.report_store = report_store
        self.reduce_invoker_factory = reduce_invoker_factory
        self.audit_invoker_factory = audit_invoker_factory
        self.execution_mode = execution_mode
        self.runtime_config = runtime_config or ReportRuntimeConfig.defaults()
        self.last_reduce: SolReduceCoordinator | None = None
        self.last_audit: ReportAuditCoordinator | None = None

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
        planner = ReportPlanner(
            bundle.plan, analyses,
            max_chunk_input_tokens=int(bundle.plan["aggregation"]["max_chunk_input_tokens"]),
            reduce_output_tokens=int(bundle.plan["aggregation"]["reduce_output_tokens"]),
            audit_input_tokens=1,
            repair_input_tokens=1,
        )
        try:
            reduce_plan = planner.build()
        except BudgetExceeded as error:
            document = _budget_error(error)
            if dry_run:
                return self._result(
                    report_run_id,
                    "incomplete",
                    True,
                    codex_budget=self._codex_budget(report_run_id, bundle.plan),
                    alarm_codes=(CODEX_BUDGET_ALARM,),
                    error=document,
                )
            return self._budget_exhausted_result(
                report_run_id, pipeline_run_id, bundle.plan, error
            )
        if dry_run:
            # Selection and tree compilation are pure reads; do not persist a
            # ReportPlan or create a coordinator that could dispatch Sol.
            return self._result(
                report_run_id,
                "validated",
                True,
                codex_budget=self._codex_budget(report_run_id, bundle.plan),
            )

        persist_approved_report_plan(self.database, bundle.plan)
        reduce_options: dict[str, Any] = {
            "execution_mode": self.execution_mode,
            "resources": self.runtime_config.resources,
            "rubric_path": self.runtime_config.rubric_path,
        }
        if self.reduce_invoker_factory is not None:
            reduce_options["invoker_factory"] = self.reduce_invoker_factory
        reducer = SolReduceCoordinator(
            self.database, self.artifact_store, self.gate, analyses,
            self._sections(bundle.plan), self._memberships(bundle.plan), **reduce_options,
        )
        self.last_reduce = reducer
        try:
            reduced = reducer.run(
                report_run_id, pipeline_run_id, bundle.plan, reduce_plan, artifacts,
                corpus_snapshot=bundle.corpus_snapshot, search_audit_pack=bundle.search_audit,
                processing_grants=processing_grants,
            )
        except SolBudgetError as error:
            return self._budget_exhausted_result(
                report_run_id, pipeline_run_id, bundle.plan, error
            )
        if reduced.status != "generation_complete":
            budget_error = self._persisted_budget_error(report_run_id)
            if budget_error is not None:
                self._mark_budget_incomplete(
                    report_run_id, pipeline_run_id, budget_error
                )
                return self._result(
                    report_run_id,
                    "incomplete",
                    False,
                    reduce=reduced,
                    codex_budget=self._codex_budget(report_run_id, bundle.plan),
                    alarm_codes=(CODEX_BUDGET_ALARM,),
                    error=budget_error,
                )
            return self._result(
                report_run_id,
                reduced.status,
                False,
                reduce=reduced,
                codex_budget=self._codex_budget(report_run_id, bundle.plan),
            )

        audit_options: dict[str, Any] = {
            "execution_mode": self.execution_mode,
            "resources": self.runtime_config.resources,
            "rubric_path": self.runtime_config.rubric_path,
        }
        if self.audit_invoker_factory is not None:
            audit_options["invoker_factory"] = self.audit_invoker_factory
        auditor = ReportAuditCoordinator(
            self.database, self.artifact_store, self.gate, self.report_store, **audit_options,
        )
        self.last_audit = auditor
        report_bundle = self._report_bundle(report_run_id, bundle, auditor)
        try:
            audited = auditor.run(
                report_run_id,
                report_bundle,
                processing_grants=processing_grants,
                previous=previous,
            )
        except ReportAuditBudgetError as error:
            return self._budget_exhausted_result(
                report_run_id,
                pipeline_run_id,
                bundle.plan,
                error,
                reduce=reduced,
            )
        budget_error = self._persisted_budget_error(report_run_id)
        if budget_error is not None:
            self._mark_budget_incomplete(
                report_run_id, pipeline_run_id, budget_error
            )
        return self._result(
            report_run_id,
            "incomplete" if budget_error is not None else audited.status,
            False,
            reduce=reduced,
            audit=audited,
            codex_budget=self._codex_budget(report_run_id, bundle.plan),
            alarm_codes=(CODEX_BUDGET_ALARM,) if budget_error is not None else (),
            error=budget_error,
        )

    @staticmethod
    def _result(
        report_run_id: str,
        status: str,
        dry_run: bool,
        *,
        reduce: ReportReduceResult | None = None,
        audit: ReportAuditResult | None = None,
        skipped: bool = False,
        codex_budget: ReportCodexBudget | None = None,
        alarm_codes: tuple[str, ...] = (),
        error: Mapping[str, str] | None = None,
    ) -> ReportExecutionResult:
        return ReportExecutionResult(
            report_run_id=report_run_id,
            status=status,
            dry_run=dry_run,
            reduce=reduce,
            audit=audit,
            skipped=skipped,
            codex_budget=codex_budget,
            alarm_codes=alarm_codes,
            error=error,
        )

    def _budget_exhausted_result(
        self,
        report_run_id: str,
        pipeline_run_id: str,
        plan: Mapping[str, Any],
        error: BudgetExceeded | SolBudgetError | ReportAuditBudgetError,
        *,
        reduce: ReportReduceResult | None = None,
    ) -> ReportExecutionResult:
        document = _budget_error(error)
        self._mark_budget_incomplete(report_run_id, pipeline_run_id, document)
        return self._result(
            report_run_id,
            "incomplete",
            False,
            reduce=reduce,
            codex_budget=self._codex_budget(report_run_id, plan),
            alarm_codes=(CODEX_BUDGET_ALARM,),
            error=document,
        )

    def _codex_budget(
        self, report_run_id: str, plan: Mapping[str, Any]
    ) -> ReportCodexBudget:
        row = self.database.connection.execute(
            """SELECT COALESCE(SUM(calls), 0), COALESCE(SUM(tokens), 0)
               FROM (
                   SELECT budget_calls_reserved AS calls,
                          budget_tokens_reserved AS tokens
                     FROM report_reduce_nodes WHERE report_run_id = ?
                   UNION ALL
                   SELECT budget_calls_reserved, budget_tokens_reserved
                     FROM report_audit_steps WHERE report_run_id = ?
                   UNION ALL
                   SELECT budget_calls_reserved, budget_tokens_reserved
                     FROM report_audit_shard_steps WHERE report_run_id = ?
               )""",
            (report_run_id, report_run_id, report_run_id),
        ).fetchone()
        budget = plan["budget"]
        return ReportCodexBudget(
            int(row[0]),
            int(row[1]),
            int(budget["max_sol_calls"]),
            int(budget["max_input_tokens"]),
        )

    def _persisted_budget_error(
        self, report_run_id: str
    ) -> Mapping[str, str] | None:
        rows = self.database.connection.execute(
            """SELECT error_json FROM report_reduce_nodes WHERE report_run_id = ?
               UNION ALL
               SELECT error_json FROM report_audit_runs WHERE report_run_id = ?
               UNION ALL
               SELECT error_json FROM report_audit_steps WHERE report_run_id = ?
               UNION ALL
               SELECT error_json FROM report_audit_shard_steps WHERE report_run_id = ?""",
            (report_run_id, report_run_id, report_run_id, report_run_id),
        ).fetchall()
        budget_types = {
            BudgetExceeded.__name__,
            SolBudgetError.__name__,
            ReportAuditBudgetError.__name__,
        }
        for row in rows:
            if row["error_json"] is None:
                continue
            document = json.loads(row["error_json"])
            if document.get("error") in budget_types:
                return {
                    "type": str(document["error"]),
                    "message": str(document.get("message") or document["error"]),
                    "event_code": CODEX_BUDGET_ALARM,
                }
        return None

    def _mark_budget_incomplete(
        self,
        report_run_id: str,
        pipeline_run_id: str,
        error: Mapping[str, str],
    ) -> None:
        persisted_error = json.dumps(
            {
                "error": error["type"],
                "message": error["message"],
                "event_code": CODEX_BUDGET_ALARM,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        with self.database.transaction() as connection:
            connection.execute(
                """UPDATE report_audit_runs SET status = 'incomplete', error_json = ?,
                          completed_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
                     WHERE report_run_id = ? AND status <> 'complete'""",
                (persisted_error, report_run_id),
            )
            connection.execute(
                """UPDATE report_runs SET status = 'incomplete',
                          completed_at = CURRENT_TIMESTAMP
                     WHERE report_run_id = ? AND status <> 'complete'""",
                (report_run_id,),
            )
            connection.execute(
                """UPDATE pipeline_runs SET status = 'incomplete',
                          completed_at = CURRENT_TIMESTAMP
                     WHERE run_id = ? AND status <> 'complete'""",
                (pipeline_run_id,),
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

    def _report_bundle(
        self, report_run_id: str, frozen: ReportPlanBundle, auditor: ReportAuditCoordinator,
    ) -> ReportBundle:
        rows = self.database.connection.execute(
            """SELECT node_id, call_kind, dependency_ids_json, output_hash FROM report_reduce_nodes
               WHERE report_run_id = ? AND status = 'complete'""", (report_run_id,)
        ).fetchall()
        final = [row for row in rows if row["call_kind"] == "final_reduce"]
        if len(final) != 1 or not final[0]["output_hash"]:
            raise ValueError("generation_complete report lacks one persisted final reduce output")
        document = _object(self.artifact_store.read_bytes(str(final[0]["output_hash"])), "final reduce output")
        dependencies = json.loads(final[0]["dependency_ids_json"])
        if len(dependencies) != 1:
            raise ValueError("final reduce must have one synthesis dependency")
        synthesis = next((row for row in rows if row["node_id"] == dependencies[0]), None)
        if synthesis is None or not synthesis["output_hash"]:
            raise ValueError("final reduce synthesis output is missing")
        claims = tuple(_object(self.artifact_store.read_bytes(str(synthesis["output_hash"])), "synthesis output")["claims"])
        groups = derive_comparison_groups(claims)
        bibliography = _bibliography(frozen.corpus_snapshot, claims)
        provisional = ReportBundle(
            frozen.plan, frozen.search_audit, frozen.corpus_snapshot, claims, groups, (), document, {}, bibliography,
        )
        coverage = auditor._rebuild_persisted_coverage(report_run_id, {  # noqa: SLF001 - shared persisted-source verifier.
            "plan": provisional.plan, "corpus_snapshot": provisional.corpus_snapshot, "claims": provisional.claims,
        })
        return ReportBundle(
            frozen.plan, frozen.search_audit, frozen.corpus_snapshot, claims, groups, (), document, coverage, bibliography,
        )


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


def _budget_error(
    error: BudgetExceeded | SolBudgetError | ReportAuditBudgetError,
) -> dict[str, str]:
    return {
        "type": type(error).__name__,
        "message": str(error),
        "event_code": CODEX_BUDGET_ALARM,
    }


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


def _bibliography(corpus: Mapping[str, Any], claims: tuple[Mapping[str, Any], ...]) -> dict[str, dict[str, Any]]:
    cited = {str(ref["paper_id"]) for claim in claims for field in ("supporting_evidence", "contradicting_evidence")
             for ref in claim[field] if ref.get("kind") == "paper_evidence"}
    papers = {str(item["paper_id"]): item for item in corpus["papers"]}
    result = {}
    for paper_id in sorted(cited):
        paper = papers[paper_id]
        result[paper_id] = {key: value for key, value in {
            "title": paper.get("title"), "authors": paper.get("authors"), "year": paper.get("publication_year"),
            "venue_name": paper.get("venue_name"), "doi": paper.get("doi"), "canonical_url": paper.get("canonical_url"),
        }.items() if value is not None}
    return result
