"""One Sol call over every frozen Stage 4 analysis, followed by local publication."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
import json
from pathlib import Path
from time import monotonic, sleep
from typing import Any, Protocol

from .approval import require_valid_approval
from .artifacts import ArtifactMetadataConflict, ArtifactStore, StoredArtifact
from .canonical import canonical_json, content_hash
from .codex_exec import (
    CALL_KIND_PROMPTS,
    CALL_KIND_SCHEMAS,
    FROZEN_PROFILES,
    CodexExec,
    CodexExecRequest,
    CodexExecResult,
    InvocationMetadata,
)
from .processing import ProcessingDecision, ProcessingGate, SUMMARY_MODEL
from .report_artifacts import (
    LOCAL_REFERENCES_NOTE,
    RENDERER_VERSION,
    ReportArtifactError,
    ReportArtifactStore,
    audit_coverage_ledger,
    audit_rubric_hash,
    audit_search_limitations,
    render_markdown,
    report_artifact_hash,
    validate_claim_relations,
    verify_report,
    _claim_evidence_diff,
)
from .report_config import ReportResources
from .report_facts import (
    ReportFactError,
    materialize_verified_report_facts,
    require_verified_report_claims,
)
from .report_invocations import register_report_invocation
from .report_plan import ReportPlanBundle, persist_approved_report_plan
from .report_reduce import (
    FrozenDerivedArtifact,
    ReportReduceError,
    _validate_report_document,
    _validate_section_coverage_dispositions,
)
from .reporting import (
    AnalysisRecord,
    CoverageLedger,
    EvidenceValidationError,
    PaperCoverage,
    SectionRule,
    SynthesisValidator,
    comparison_assessment,
    corpus_evidence_allowlist,
    derive_comparison_groups,
    stable_claim_id,
)
from .storage import Database
from .schema import SchemaValidationError


PROFILE = "stage4b_oneshot_sol"
CALL_KIND = "one_shot_report"
MODEL = SUMMARY_MODEL
REASONING_EFFORT = "high"
IMPLEMENTATION_VERSION = "stage4b-one-shot-v3"
DETERMINISTIC_VALIDATION_VERSION = "deterministic-report-v3"
NODE_ID = "one_shot:0001"
MAX_OUTPUT_BYTES = 1_048_576
DISPATCH_GRACE_SECONDS = 600
PUBLICATION_RACE_WAIT_SECONDS = 5.0


class DirectReportError(RuntimeError):
    pass


class DirectReportBudgetError(DirectReportError):
    pass


class DirectReportInvoker(Protocol):
    def invoke(self, request: CodexExecRequest) -> CodexExecResult: ...


@dataclass(frozen=True, slots=True)
class DirectReportResult:
    report_run_id: str
    status: str
    published_path: Path | None = None
    error: str | None = None
    budget_exhausted: bool = False


@dataclass(frozen=True, slots=True)
class DirectReportPreflight:
    prompt: str
    rendered_prompt: str
    input_tokens: int


def one_shot_config_hash(
    processing_policy_hash: str,
    *,
    execution_mode: str = "attended",
    resources: ReportResources | None = None,
) -> str:
    selected = resources or ReportResources.defaults()
    selected.validate_files()
    profile = FROZEN_PROFILES[PROFILE]
    return content_hash({
        "strategy": "one_shot",
        "profile": PROFILE,
        "model": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "sandbox": profile.sandbox,
        "network": profile.network,
        "timeout_seconds": profile.timeout_seconds,
        "max_retries": profile.max_retries,
        "input_budget_measure": "utf8_bytes_upper_bound",
        "schema_hash": content_hash(selected.schema(CALL_KIND)),
        "service_schema_hash": selected.service_schema_hash(CALL_KIND),
        "prompt_hash": sha256(selected.prompt_paths[CALL_KIND].read_bytes()).hexdigest(),
        "processing_policy_hash": processing_policy_hash,
        "execution_mode": execution_mode,
        "implementation_version": IMPLEMENTATION_VERSION,
    })


def one_shot_validation_config_hash() -> str:
    return content_hash({
        "strategy": "one_shot",
        "validation": DETERMINISTIC_VALIDATION_VERSION,
        # The local verifier/bibliography semantics changed, but the emitted
        # Markdown layout did not.  Keep the renderer's independent v1 label.
        "renderer_version": RENDERER_VERSION,
        "rubric_hash": audit_rubric_hash(),
        "implementation_version": IMPLEMENTATION_VERSION,
    })


class DirectReportCoordinator:
    """Execute and persist at most one Sol invocation for one report run."""

    def __init__(
        self,
        database: Database,
        artifact_store: ArtifactStore,
        gate: ProcessingGate,
        report_store: ReportArtifactStore,
        analyses: Sequence[AnalysisRecord],
        source_artifacts: Sequence[FrozenDerivedArtifact],
        sections: Sequence[SectionRule],
        memberships: Mapping[str, Sequence[str]],
        *,
        invoker_factory: Callable[[], DirectReportInvoker] = CodexExec,
        resources: ReportResources | None = None,
        execution_mode: str = "attended",
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.database = database
        self.artifact_store = artifact_store
        self.gate = gate
        self.report_store = report_store
        self.analyses = tuple(analyses)
        self.artifacts = tuple(source_artifacts)
        self.sections = tuple(sections)
        self.memberships = {
            paper_id: tuple(section_ids)
            for paper_id, section_ids in memberships.items()
        }
        self.invoker_factory = invoker_factory
        self.resources = resources or ReportResources.defaults()
        self.resources.validate_files()
        self.execution_mode = execution_mode
        self.clock = clock or (lambda: datetime.now(UTC))

    def run(
        self,
        report_run_id: str,
        pipeline_run_id: str,
        bundle: ReportPlanBundle,
        *,
        processing_grants: Mapping[str, str],
        previous: Mapping[str, Any] | None = None,
    ) -> DirectReportResult:
        try:
            preflight = self.preflight(report_run_id, bundle, previous=previous)
        except DirectReportError as error:
            return DirectReportResult(
                report_run_id,
                "incomplete",
                error=str(error),
                budget_exhausted=isinstance(error, DirectReportBudgetError),
            )
        persist_approved_report_plan(self.database, bundle.plan)
        self._ensure_run(
            report_run_id,
            pipeline_run_id,
            bundle,
            preflight.prompt,
            preflight.rendered_prompt,
            preflight.input_tokens,
        )
        row = self._row(report_run_id)
        if row["status"] == "failed":
            return DirectReportResult(report_run_id, "failed", error=_error(row["error_json"]))
        if row["status"] == "running":
            if self._dispatch_expired(row):
                error = DirectReportError(
                    "the sole Sol dispatch passed its deadline with an uncertain outcome; it will not be dispatched again"
                )
                self._mark_failed(report_run_id, error)
                return DirectReportResult(report_run_id, "failed", error=str(error))
            return DirectReportResult(
                report_run_id,
                "running",
                error="the sole Sol dispatch is still running or its outcome is uncertain",
            )
        if row["status"] != "complete":
            decisions, decision_time = self._authorize(processing_grants)
            if not all(decision.is_authorized for decision in decisions):
                self._mark_manual(report_run_id, decisions)
                return DirectReportResult(
                    report_run_id,
                    "manual_required",
                    error=next(
                        decision.reason_code
                        for decision in decisions
                        if not decision.is_authorized
                    ),
                )
            self._claim_dispatch(report_run_id, decisions, decision_time)
            request = self._request(preflight.prompt, preflight.rendered_prompt)
            try:
                invoked = self.invoker_factory().invoke(request)
                self._persist_output(
                    report_run_id, request, preflight.rendered_prompt, invoked
                )
            except Exception as error:
                self._mark_failed(report_run_id, error)
                return DirectReportResult(report_run_id, "failed", error=str(error))
        try:
            published = self._publish(report_run_id, bundle, previous)
        except (
            DirectReportError,
            EvidenceValidationError,
            ReportArtifactError,
            ReportReduceError,
            ReportFactError,
            SchemaValidationError,
            ArtifactMetadataConflict,
            OSError,
            ValueError,
        ) as error:
            self._mark_failed(report_run_id, error)
            return DirectReportResult(report_run_id, "failed", error=str(error))
        self._mark_complete(report_run_id, pipeline_run_id, published)
        return DirectReportResult(report_run_id, "complete", published)

    def preflight(
        self,
        report_run_id: str,
        bundle: ReportPlanBundle,
        *,
        previous: Mapping[str, Any] | None = None,
    ) -> DirectReportPreflight:
        """Validate and size the exact request without persistence or model work."""
        self._validate_plan(bundle.plan)
        self._validate_previous(report_run_id, previous)
        self._validate_bibliography_metadata(bundle)
        prompt = self._prompt(report_run_id, bundle, previous)
        rendered = self._rendered_prompt(prompt)
        input_tokens = len(rendered.encode("utf-8"))
        if input_tokens > int(bundle.plan["budget"]["max_input_tokens"]):
            raise DirectReportBudgetError(
                "one-shot Sol prompt exceeds the approved input budget"
            )
        return DirectReportPreflight(prompt, rendered, input_tokens)

    def _validate_bibliography_metadata(self, bundle: ReportPlanBundle) -> None:
        papers = {
            str(item["paper_id"]): item
            for item in bundle.corpus_snapshot["papers"]
        }
        evidence_paper_ids = {
            str(item["paper_id"])
            for item in bundle.plan["paper_memberships"]
            if item["coverage_disposition"] == "evidence"
        }
        for paper_id in sorted(evidence_paper_ids):
            paper = papers.get(paper_id)
            if paper is None:
                raise DirectReportError(
                    "ReportPlan bibliography paper is absent from the frozen corpus"
                )
            _canonical_bibliography_metadata(
                self.database,
                paper_id,
                paper,
                frozen_snapshot_hash=str(bundle.corpus_snapshot["snapshot_hash"]),
            )

    def _validate_plan(self, plan: Mapping[str, Any]) -> None:
        require_valid_approval(plan, "plan_hash")
        budget = plan["budget"]
        if (
            plan.get("execution_strategy") != "one_shot"
            or int(budget["max_sol_calls"]) != 1
            or int(budget["max_retries"]) != 0
            or int(budget["audit_calls"]) != 0
            or int(budget["repair_calls"]) != 0
        ):
            raise DirectReportError("one-shot ReportPlan must allow exactly one Sol call and no retries")
        if plan["stage4b_config_hash"] != one_shot_config_hash(
            self.gate.policy.hash,
            execution_mode=self.execution_mode,
            resources=self.resources,
        ):
            raise DirectReportError("one-shot Stage 4b configuration has drifted")
        if plan["stage4b_audit_config_hash"] != one_shot_validation_config_hash():
            raise DirectReportError("one-shot deterministic validation configuration has drifted")
        if any(not section["subquestion_ids"] for section in plan["sections"]):
            raise DirectReportError(
                "one-shot ReportPlan requires at least one subquestion for every section"
            )

    def _validate_previous(
        self,
        report_run_id: str,
        previous: Mapping[str, Any] | None,
    ) -> None:
        if previous is None:
            return
        previous_run_id = previous.get("report_run_id")
        claims = previous.get("claims")
        if (
            not isinstance(previous_run_id, str)
            or not previous_run_id
            or previous_run_id == report_run_id
            or not isinstance(claims, Sequence)
            or isinstance(claims, (str, bytes))
        ):
            raise DirectReportError(
                "incremental one-shot input requires a distinct previous report and its claims"
            )
        try:
            require_verified_report_claims(
                self.database.connection,
                report_run_id=previous_run_id,
                claims=claims,
            )
        except (KeyError, ReportFactError) as error:
            raise DirectReportError(str(error)) from error

    def _prompt(
        self,
        report_run_id: str,
        bundle: ReportPlanBundle,
        previous: Mapping[str, Any] | None,
    ) -> str:
        documents = []
        for artifact in self.artifacts:
            analysis = _object(artifact.payload, "Stage 4 analysis")
            documents.append({
                "paper_id": artifact.paper_id,
                "analysis_run_id": next(
                    item.analysis_run_id
                    for item in self.analyses
                    if item.paper_id == artifact.paper_id
                ),
                "analysis_artifact_hash": artifact.artifact_hash,
                "lineage_hash": artifact.lineage_hash,
                "document": analysis,
            })
        payload = {
            "report_run_id": report_run_id,
            "report_plan": dict(bundle.plan),
            "corpus_summary": _corpus_prompt_summary(bundle.corpus_snapshot),
            "search_audit_summary": _search_audit_prompt_summary(
                bundle.search_audit
            ),
            "corpus_evidence": corpus_evidence_allowlist(bundle.search_audit).document(),
            "allowed_evidence_references": _allowed_evidence_references(
                bundle, self.analyses
            ),
            "required_disclosures": list(
                audit_search_limitations(bundle.search_audit, bundle.corpus_snapshot)
            ),
            "analyses": documents,
            "previous_report": previous,
        }
        return canonical_json(payload).decode("utf-8")

    def _rendered_prompt(self, prompt: str) -> str:
        template = self.resources.prompt(CALL_KIND)
        encoded = json.dumps(
            {"authorized_input": prompt}, ensure_ascii=False, separators=(",", ":")
        )
        return f"{template.rstrip()}\n\nThe authorized input follows as JSON data:\n{encoded}\n"

    def _request(self, prompt: str, rendered: str) -> CodexExecRequest:
        return CodexExecRequest(
            profile=PROFILE,
            prompt=prompt,
            output_schema=self.resources.schema(CALL_KIND),
            schema_name=CALL_KIND_SCHEMAS[CALL_KIND],
            prompt_name=CALL_KIND_PROMPTS[CALL_KIND],
            input_hash=sha256(prompt.encode("utf-8")).hexdigest(),
            call_kind=CALL_KIND,
            schema_path=self.resources.schema_path(CALL_KIND),
            prompt_path=self.resources.prompt_path(CALL_KIND),
            expected_prompt_hash=sha256(
                self.resources.prompt_paths[CALL_KIND].read_bytes()
            ).hexdigest(),
            schema_resource_paths=self.resources.configured_schema_resources(),
            expected_service_schema_hash=self.resources.service_schema_hash(CALL_KIND),
        )

    def _authorize(
        self, processing_grants: Mapping[str, str]
    ) -> tuple[tuple[ProcessingDecision, ...], datetime]:
        decision_time = self._now()
        grant_papers: dict[str, set[str]] = defaultdict(set)
        for artifact in self.artifacts:
            grant_id = processing_grants.get(artifact.artifact_hash)
            if grant_id:
                grant_papers[grant_id].update(artifact.source_paper_ids)
        decisions = []
        for artifact in self.artifacts:
            grant_id = processing_grants.get(artifact.artifact_hash)
            dispatched = self.gate.dispatch(
                artifact.processing_request(),
                lambda invocation: invocation,
                processing_grant_id=grant_id,
                now=decision_time,
                paper_count=max(1, len(grant_papers.get(grant_id or "", ()))),
            )
            decisions.append(dispatched.decision)
        return tuple(decisions), decision_time

    def _ensure_run(
        self,
        report_run_id: str,
        pipeline_run_id: str,
        bundle: ReportPlanBundle,
        prompt: str,
        rendered: str,
        input_tokens: int,
    ) -> None:
        input_hash = sha256(prompt.encode("utf-8")).hexdigest()
        rendered_hash = sha256(rendered.encode("utf-8")).hexdigest()
        prompt_hash = sha256(self.resources.prompt_paths[CALL_KIND].read_bytes()).hexdigest()
        schema_hash = content_hash(self.resources.schema(CALL_KIND))
        tree = {
            "strategy": "one_shot",
            "node_id": NODE_ID,
            "paper_ids": sorted(item.paper_id for item in self.analyses),
            "section_ids": [str(item["id"]) for item in bundle.plan["sections"]],
        }
        pipeline_input_hash = content_hash({
            "plan_hash": bundle.plan["plan_hash"],
            "corpus_snapshot_hash": bundle.corpus_snapshot["snapshot_hash"],
            "search_audit_pack_hash": bundle.search_audit["pack_hash"],
            "input_hash": input_hash,
        })
        with self.database.transaction() as connection:
            pipeline = connection.execute(
                "SELECT stage, input_hash, config_hash, implementation_version FROM pipeline_runs WHERE run_id = ?",
                (pipeline_run_id,),
            ).fetchone()
            expected_pipeline = (
                "stage4b",
                pipeline_input_hash,
                bundle.plan["stage4b_config_hash"],
                IMPLEMENTATION_VERSION,
            )
            if pipeline is None:
                connection.execute(
                    """INSERT INTO pipeline_runs(
                           run_id, stage, status, input_hash, config_hash,
                           implementation_version, started_at
                       ) VALUES (?, 'stage4b', 'running', ?, ?, ?, CURRENT_TIMESTAMP)""",
                    (pipeline_run_id, *expected_pipeline[1:]),
                )
            elif tuple(pipeline) != expected_pipeline:
                raise DirectReportError("one-shot pipeline run binding has drifted")
            report = connection.execute(
                """SELECT run_id, report_plan_id, corpus_snapshot_hash,
                          aggregation_tree_json, model_id, prompt_hash, schema_hash
                   FROM report_runs WHERE report_run_id = ?""",
                (report_run_id,),
            ).fetchone()
            expected_report = (
                pipeline_run_id,
                bundle.plan["plan_id"],
                bundle.corpus_snapshot["snapshot_hash"],
                _json(tree),
                MODEL,
                prompt_hash,
                schema_hash,
            )
            if report is None:
                connection.execute(
                    """INSERT INTO report_runs(
                           report_run_id, run_id, report_plan_id, corpus_snapshot_hash,
                           aggregation_tree_json, model_id, model_revision,
                           prompt_hash, schema_hash, status
                       ) VALUES (?, ?, ?, ?, ?, ?, 'codex-cli-managed', ?, ?, 'running')""",
                    (report_run_id, *expected_report),
                )
            elif tuple(report) != expected_report:
                raise DirectReportError("one-shot report run binding has drifted")
            row = connection.execute(
                """SELECT input_artifact_hashes_json, input_hash,
                          rendered_prompt_hash, actual_input_tokens, profile,
                          model_id, reasoning_effort, prompt_name, prompt_hash,
                          schema_name, schema_hash
                     FROM report_one_shot_runs WHERE report_run_id = ?""",
                (report_run_id,),
            ).fetchone()
            if row is None:
                connection.execute(
                    """INSERT INTO report_one_shot_runs(
                           report_run_id, input_artifact_hashes_json, input_hash,
                           rendered_prompt_hash, actual_input_tokens, profile, model_id,
                           reasoning_effort, prompt_name, prompt_hash, schema_name,
                           schema_hash, status
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')""",
                    (
                        report_run_id,
                        _json([item.artifact_hash for item in self.artifacts]),
                        input_hash,
                        rendered_hash,
                        input_tokens,
                        PROFILE,
                        MODEL,
                        REASONING_EFFORT,
                        CALL_KIND_PROMPTS[CALL_KIND],
                        prompt_hash,
                        CALL_KIND_SCHEMAS[CALL_KIND],
                        schema_hash,
                    ),
                )
            else:
                expected_one_shot = (
                    _json([item.artifact_hash for item in self.artifacts]),
                    input_hash,
                    rendered_hash,
                    input_tokens,
                    PROFILE,
                    MODEL,
                    REASONING_EFFORT,
                    CALL_KIND_PROMPTS[CALL_KIND],
                    prompt_hash,
                    CALL_KIND_SCHEMAS[CALL_KIND],
                    schema_hash,
                )
                if tuple(row) != expected_one_shot:
                    raise DirectReportError("one-shot run input or resource binding has drifted")

    def _claim_dispatch(
        self,
        report_run_id: str,
        decisions: Sequence[ProcessingDecision],
        decision_time: datetime,
    ) -> None:
        dispatch_expires_at = _timestamp(
            decision_time
            + timedelta(
                seconds=(
                    FROZEN_PROFILES[PROFILE].timeout_seconds
                    + DISPATCH_GRACE_SECONDS
                )
            )
        )
        with self.database.transaction() as connection:
            updated = connection.execute(
                """UPDATE report_one_shot_runs
                   SET status = 'running', dispatch_count = 1,
                       budget_calls_reserved = 1,
                       budget_tokens_reserved = actual_input_tokens,
                       dispatch_expires_at = ?,
                       processing_decisions_json = ?, processing_grant_ids_json = ?,
                       error_json = NULL, updated_at = CURRENT_TIMESTAMP
                   WHERE report_run_id = ?
                     AND status IN ('pending', 'manual_required')
                     AND dispatch_count = 0""",
                (
                    dispatch_expires_at,
                    _json([_decision(item) for item in decisions]),
                    _json([item.processing_grant_id for item in decisions]),
                    report_run_id,
                ),
            )
            if updated.rowcount != 1:
                raise DirectReportError("one-shot Sol dispatch was already claimed")

    def _persist_output(
        self,
        report_run_id: str,
        request: CodexExecRequest,
        rendered_prompt: str,
        invoked: CodexExecResult,
    ) -> None:
        metadata = invoked.metadata
        expected = (
            PROFILE,
            MODEL,
            REASONING_EFFORT,
            CALL_KIND,
            CALL_KIND_SCHEMAS[CALL_KIND],
            content_hash(self.resources.schema(CALL_KIND)),
            CALL_KIND_PROMPTS[CALL_KIND],
            sha256(self.resources.prompt_paths[CALL_KIND].read_bytes()).hexdigest(),
            request.input_hash,
            MODEL,
            PROFILE,
            sha256(rendered_prompt.encode("utf-8")).hexdigest(),
            1,
        )
        actual = (
            metadata.profile,
            metadata.model,
            metadata.reasoning_effort,
            metadata.call_kind,
            metadata.schema_name,
            metadata.schema_hash,
            metadata.prompt_name,
            metadata.prompt_hash,
            metadata.input_hash,
            metadata.actual_model,
            metadata.actual_profile,
            metadata.rendered_prompt_hash,
            metadata.attempts,
        )
        if (
            actual != expected
            or not self.resources.accepts_metadata_paths(
                CALL_KIND, metadata.schema_path, metadata.prompt_path
            )
            or not isinstance(metadata.invocation_id, str)
            or not metadata.invocation_id.strip()
            or metadata.output_hash != content_hash(dict(invoked.output))
        ):
            raise DirectReportError("one-shot Sol invocation metadata does not match its request")
        # ``CodexExec`` freezes its top-level output with ``MappingProxyType``.
        # jsonschema's default object checker recognizes concrete dicts, not
        # arbitrary Mapping implementations, so thaw the already-validated
        # structured result before applying the configured resource schema.
        self.resources.validate(dict(invoked.output), CALL_KIND)
        payload = canonical_json(dict(invoked.output))
        if len(payload) > MAX_OUTPUT_BYTES:
            raise DirectReportError("one-shot Sol output exceeds the frozen byte limit")
        stored = self.artifact_store.put_bytes(
            payload,
            mime_type="application/json",
            metadata={"kind": "stage4b_one_shot_output"},
        )
        metadata_document = asdict(metadata)
        with self.database.transaction() as connection:
            register_report_invocation(
                connection,
                report_run_id=report_run_id,
                invocation_id=metadata.invocation_id,
                phase="reduce",
                node_key=NODE_ID,
                metadata=metadata_document,
            )
            artifact_id = _save_artifact(connection, stored)
            updated = connection.execute(
                """UPDATE report_one_shot_runs
                   SET status = 'complete', invocation_metadata_json = ?,
                       invocation_id = ?, output_artifact_id = ?, output_hash = ?,
                       error_json = NULL, completed_at = CURRENT_TIMESTAMP,
                       updated_at = CURRENT_TIMESTAMP
                   WHERE report_run_id = ? AND status = 'running'""",
                (
                    _json(metadata_document),
                    metadata.invocation_id,
                    artifact_id,
                    stored.artifact_hash,
                    report_run_id,
                ),
            )
            if updated.rowcount != 1:
                raise DirectReportError(
                    "one-shot completion lost ownership of the sole dispatch"
                )

    def _publish(
        self,
        report_run_id: str,
        frozen: ReportPlanBundle,
        previous: Mapping[str, Any] | None,
    ) -> Path:
        row = self._row(report_run_id)
        output = _object(
            self.artifact_store.read_bytes(str(row["output_hash"])),
            "one-shot Sol output",
        )
        _require_allowed_evidence(
            output, _allowed_evidence_references(frozen, self.analyses)
        )
        claims, document, claim_ids, procedural_refs = self._normalize_output(
            report_run_id, output
        )
        groups = derive_comparison_groups(claims)
        prior_claims = (
            {
                str(claim["claim_id"]): claim
                for claim in previous.get("claims", ())
            }
            if previous is not None
            else {}
        )
        current_claims = {str(claim["claim_id"]): claim for claim in claims}
        relation_drafts = []
        for relation in output["claim_relations"]:
            if str(relation["current_claim_ref"]) in procedural_refs:
                raise DirectReportError(
                    "one-shot claim relation cannot target a procedural reference note"
                )
            current_id = claim_ids.get(str(relation["current_claim_ref"]))
            previous_id = str(relation["previous_claim_id"])
            if current_id is None or previous_id not in prior_claims:
                raise DirectReportError("one-shot claim relation has an unknown endpoint")
            relation_drafts.append({
                "previous_claim_id": previous_id,
                "current_claim_id": current_id,
                "relation_type": relation["relation_type"],
                "reason": relation["reason"],
                "evidence_diff": _claim_evidence_diff(
                    prior_claims[previous_id], current_claims[current_id]
                ),
            })
        relations = validate_claim_relations(previous, claims, relation_drafts)
        coverage = _coverage(frozen.plan, claims)
        bibliography = _bibliography(
            frozen.corpus_snapshot, claims, database=self.database
        )
        validator = SynthesisValidator(
            report_run_id=report_run_id,
            analyses=self.analyses,
            sections=self.sections,
            memberships=self.memberships,
            corpus_evidence=corpus_evidence_allowlist(frozen.search_audit),
        )
        for section in self.sections:
            section_claims = tuple(
                claim for claim in claims if claim["report_section"] == section.section_id
            )
            section_blocks = tuple(
                block for block in document["blocks"] if block["section_id"] == section.section_id
            )
            evidence_papers = sorted({
                str(reference["paper_id"])
                for claim in section_claims
                for field in ("supporting_evidence", "contradicting_evidence")
                for reference in claim[field]
                if reference["kind"] == "paper_evidence"
            })
            section_document = {
                "section_id": section.section_id,
                "draft": "\n".join(str(block["text"]) for block in section_blocks),
                "claims": list(section_claims),
                "citation_paper_ids": evidence_papers,
                "unresolved_conflicts": list(output["unresolved_conflicts"]),
            }
            validator.validate_section(section_document)
            _validate_section_coverage_dispositions(frozen.plan, section_document)
        _validate_report_document(
            report_run_id,
            frozen.plan,
            document,
            {
                "claims": list(claims),
                "unresolved_conflicts": list(output["unresolved_conflicts"]),
            },
        )
        coverage.require_complete()
        report_bundle = {
            "plan": frozen.plan,
            "search_audit": frozen.search_audit,
            "corpus_snapshot": frozen.corpus_snapshot,
            "claims": claims,
            "comparison_groups": groups,
            "claim_relations": relations,
            "document": document,
            "coverage": coverage,
            "bibliography": bibliography,
        }
        verification = _deterministic_verify(report_bundle, previous=previous)
        audit = _deterministic_audit(report_bundle)
        target = self.report_store.directory(report_run_id)
        if not target.exists():
            try:
                self.report_store.write(
                    plan=frozen.plan,
                    search_audit=frozen.search_audit,
                    corpus_snapshot=frozen.corpus_snapshot,
                    claims=claims,
                    comparison_groups=groups,
                    claim_relations=relations,
                    document=document,
                    coverage=coverage,
                    bibliography=bibliography,
                    audit=audit,
                    previous=previous,
                    advance_latest=False,
                )
            except ReportArtifactError:
                if not _wait_for_concurrent_publication(target):
                    raise
        previous_report_run_id = _previous_report_run_id(
            report_run_id, previous, relations
        )
        # Verify an existing immutable bundle before changing its database
        # attestation.  This matters when resuming a report produced under an
        # older serialization contract: a reconcile failure must leave the
        # original audit hashes available for strict legacy verification.
        self.report_store.reconcile(
            plan=frozen.plan,
            search_audit=frozen.search_audit,
            corpus_snapshot=frozen.corpus_snapshot,
            claims=claims,
            comparison_groups=groups,
            claim_relations=relations,
            document=document,
            coverage=coverage,
            bibliography=bibliography,
            audit=audit,
            previous=previous,
            advance_latest=False,
        )
        with self.database.transaction() as connection:
            _upsert_local_audit_run(
                connection,
                report_run_id,
                report_bundle,
                audit,
                target,
                self.execution_mode,
                previous,
            )
            materialize_verified_report_facts(
                connection,
                report_run_id=report_run_id,
                bundle=report_bundle,
                deterministic_verification=verification,
                previous_report_run_id=previous_report_run_id,
            )
        self.report_store.reconcile(
            plan=frozen.plan,
            search_audit=frozen.search_audit,
            corpus_snapshot=frozen.corpus_snapshot,
            claims=claims,
            comparison_groups=groups,
            claim_relations=relations,
            document=document,
            coverage=coverage,
            bibliography=bibliography,
            audit=audit,
            previous=previous,
        )
        return target

    def _normalize_output(
        self, report_run_id: str, output: Mapping[str, Any]
    ) -> tuple[
        tuple[dict[str, Any], ...],
        dict[str, Any],
        dict[str, str],
        frozenset[str],
    ]:
        drafts: dict[str, Mapping[str, Any]] = {}
        procedural_refs: set[str] = set()
        for draft in output["claims"]:
            claim_ref = str(draft["claim_ref"])
            if claim_ref in drafts:
                raise DirectReportError(f"duplicate one-shot claim_ref: {claim_ref}")
            drafts[claim_ref] = draft
            if _is_procedural_reference_note(draft):
                procedural_refs.add(claim_ref)

        resolved_blocks: list[
            tuple[Mapping[str, Any], tuple[str, ...], bool]
        ] = []
        usage_sections: dict[str, set[str]] = defaultdict(set)
        procedural_uses: dict[str, int] = defaultdict(int)
        for block in output["blocks"]:
            section_id = str(block["section_id"])
            raw_refs = tuple(str(value) for value in block["claim_refs"])
            if any(claim_ref not in drafts for claim_ref in raw_refs):
                raise DirectReportError(
                    "one-shot block references an unknown claim_ref"
                )
            local_procedural = [
                claim_ref for claim_ref in raw_refs if claim_ref in procedural_refs
            ]
            if local_procedural:
                if (
                    len(local_procedural) != len(raw_refs)
                    or section_id != "references_and_appendices"
                    or block["citation_paper_ids"]
                ):
                    raise DirectReportError(
                        "procedural reference notes must exclusively bind an uncited references block"
                    )
                for claim_ref in local_procedural:
                    procedural_uses[claim_ref] += 1
                resolved_blocks.append((block, (), True))
                continue
            resolved_refs: list[str] = []
            for claim_ref in raw_refs:
                if claim_ref not in resolved_refs:
                    resolved_refs.append(claim_ref)
                    usage_sections[claim_ref].add(section_id)
            resolved_blocks.append((block, tuple(resolved_refs), False))
        if set(procedural_uses) != procedural_refs or any(
            count != 1 for count in procedural_uses.values()
        ):
            raise DirectReportError(
                "each procedural reference note must bind exactly one references block"
            )

        claims = []
        section_refs: dict[tuple[str, str], str] = {}
        relation_refs: dict[str, str] = {}
        for claim_ref, draft in drafts.items():
            if claim_ref in procedural_refs:
                continue
            sections = sorted(usage_sections.get(claim_ref, ()))
            if not sections:
                raise DirectReportError(
                    f"one-shot claim_ref is absent from every block: {claim_ref}"
                )
            paper_units = [
                reference["evidence_unit"]
                for field in ("supporting_evidence", "contradicting_evidence")
                for reference in draft[field]
                if reference["kind"] == "paper_evidence"
            ]
            group_id = None
            if draft["claim_type"] == "comparison":
                assessments = tuple(comparison_assessment(unit) for unit in paper_units)
                comparable = tuple(
                    item.comparison_group_id
                    for item in assessments
                    if item.eligibility == "comparable"
                )
                if len(comparable) >= 2 and len(set(comparable)) == 1:
                    group_id = comparable[0]
            for section_id in sections:
                qualifier = {
                    "qualifier_context": draft["qualifier_context"],
                    "normalized_report_section": section_id,
                }
                claim_key = {
                    "subject_id": draft["subject_id"],
                    "predicate_id": draft["predicate_id"],
                    "object_or_scope_id": draft["object_or_scope_id"],
                    "qualifier_context_hash": content_hash(qualifier),
                    "comparison_group_id": group_id,
                }
                claim_id = stable_claim_id(claim_key, report_run_id=report_run_id)
                section_refs[(claim_ref, section_id)] = claim_id
                relation_refs.setdefault(claim_ref, claim_id)
                claims.append({
                    "claim_id": claim_id,
                    "claim_key": claim_key,
                    "research_question_id": draft["research_question_id"],
                    "report_section": section_id,
                    "claim_text": draft["claim_text"],
                    "claim_type": draft["claim_type"],
                    "supporting_evidence": draft["supporting_evidence"],
                    "contradicting_evidence": draft["contradicting_evidence"],
                    "evidence_level": draft["evidence_level"],
                    "comparison_group_id": group_id,
                    "confidence": draft["confidence"],
                    "known_limitations": draft["known_limitations"],
                    "status": draft["status"],
                    "mapping_status": "mapped",
                })
        if len({claim["claim_id"] for claim in claims}) != len(claims):
            raise DirectReportError("one-shot claims collapse to a duplicate stable claim ID")
        blocks = []
        for block, resolved_refs, procedural in resolved_blocks:
            section_id = str(block["section_id"])
            claim_ids = (
                []
                if procedural
                else [
                    section_refs[(claim_ref, section_id)]
                    for claim_ref in resolved_refs
                ]
            )
            blocks.append({
                "block_id": block["block_id"],
                "block_kind": "caption" if procedural else block["block_kind"],
                "section_id": section_id,
                "text": LOCAL_REFERENCES_NOTE if procedural else block["text"],
                "claim_ids": claim_ids,
                "citation_paper_ids": [] if procedural else block["citation_paper_ids"],
            })
        return (
            tuple(claims),
            {"report_run_id": report_run_id, "blocks": blocks},
            relation_refs,
            frozenset(procedural_refs),
        )

    def _mark_manual(
        self, report_run_id: str, decisions: Sequence[ProcessingDecision]
    ) -> None:
        with self.database.transaction() as connection:
            connection.execute(
                """UPDATE report_one_shot_runs
                   SET status = 'manual_required', processing_decisions_json = ?,
                       processing_grant_ids_json = ?, error_json = ?,
                       updated_at = CURRENT_TIMESTAMP
                   WHERE report_run_id = ? AND dispatch_count = 0""",
                (
                    _json([_decision(item) for item in decisions]),
                    _json([item.processing_grant_id for item in decisions]),
                    _json({"error": "processing_not_authorized"}),
                    report_run_id,
                ),
            )
            connection.execute(
                """UPDATE report_runs SET status = 'incomplete'
                   WHERE report_run_id = ?""",
                (report_run_id,),
            )
            connection.execute(
                """UPDATE pipeline_runs SET status = 'incomplete'
                   WHERE run_id = (SELECT run_id FROM report_runs WHERE report_run_id = ?)""",
                (report_run_id,),
            )

    def _mark_failed(self, report_run_id: str, error: Exception) -> None:
        with self.database.transaction() as connection:
            connection.execute(
                """UPDATE report_one_shot_runs
                   SET status = CASE WHEN status = 'complete' THEN 'complete' ELSE 'failed' END,
                       error_json = ?, completed_at = CURRENT_TIMESTAMP,
                       updated_at = CURRENT_TIMESTAMP WHERE report_run_id = ?""",
                (_json({"error": type(error).__name__, "message": str(error)}), report_run_id),
            )
            connection.execute(
                """UPDATE report_runs SET status = 'failed', completed_at = CURRENT_TIMESTAMP
                   WHERE report_run_id = ? AND status <> 'complete'""",
                (report_run_id,),
            )
            connection.execute(
                """UPDATE pipeline_runs SET status = 'failed', completed_at = CURRENT_TIMESTAMP
                   WHERE run_id = (SELECT run_id FROM report_runs WHERE report_run_id = ?)
                     AND status <> 'complete'""",
                (report_run_id,),
            )

    def _mark_complete(
        self, report_run_id: str, pipeline_run_id: str, target: Path
    ) -> None:
        relative = str(target.relative_to(self.report_store.root))
        with self.database.transaction() as connection:
            connection.execute(
                """UPDATE report_one_shot_runs
                   SET error_json = NULL, updated_at = CURRENT_TIMESTAMP
                   WHERE report_run_id = ? AND status = 'complete'""",
                (report_run_id,),
            )
            connection.execute(
                """UPDATE report_runs SET status = 'complete', output_relative_path = ?,
                          completed_at = CURRENT_TIMESTAMP WHERE report_run_id = ?""",
                (relative, report_run_id),
            )
            connection.execute(
                """UPDATE pipeline_runs SET status = 'complete', completed_at = CURRENT_TIMESTAMP
                   WHERE run_id = ?""",
                (pipeline_run_id,),
            )

    def _row(self, report_run_id: str) -> Any:
        row = self.database.connection.execute(
            "SELECT * FROM report_one_shot_runs WHERE report_run_id = ?",
            (report_run_id,),
        ).fetchone()
        if row is None:
            raise DirectReportError("one-shot report run is missing")
        return row

    def _now(self) -> datetime:
        value = self.clock()
        if not isinstance(value, datetime) or value.tzinfo is None:
            raise DirectReportError("one-shot authorization clock must be timezone-aware")
        return value.astimezone(UTC)

    def _dispatch_expired(self, row: Mapping[str, Any]) -> bool:
        value = row["dispatch_expires_at"]
        if not isinstance(value, str) or not value:
            return True
        try:
            deadline = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise DirectReportError("one-shot dispatch deadline is malformed") from error
        if deadline.tzinfo is None:
            raise DirectReportError("one-shot dispatch deadline must be timezone-aware")
        return self._now() >= deadline.astimezone(UTC)


def _wait_for_concurrent_publication(target: Path) -> bool:
    deadline = monotonic() + PUBLICATION_RACE_WAIT_SECONDS
    while monotonic() < deadline:
        if target.is_dir():
            return True
        sleep(0.05)
    return target.is_dir()


def _corpus_prompt_summary(corpus: Mapping[str, Any]) -> dict[str, Any]:
    paper_fields = (
        "paper_id",
        "title",
        "publication_year",
        "venue_id",
        "venue_name",
        "publication_status",
        "study_setting",
        "input_scope",
        "evidence_level",
        "source_category",
        "foundational",
        "recent",
    )
    return {
        "snapshot_hash": corpus["snapshot_hash"],
        "query_plan_hash": corpus["query_plan_hash"],
        "papers": [
            {key: paper.get(key) for key in paper_fields}
            for paper in sorted(
                corpus["papers"], key=lambda item: str(item["paper_id"])
            )
        ],
    }


def _search_audit_prompt_summary(audit: Mapping[str, Any]) -> dict[str, Any]:
    summary_fields = (
        "pack_id",
        "pack_hash",
        "query_plan_hash",
        "corpus_snapshot_hash",
        "search_status",
        "flow_label",
        "flow",
        "source_categories",
        "cohorts",
        "publication_status",
        "input_scope",
        "study_setting",
        "required_provider_failures",
        "incomplete_source_runs",
        "budget_exhausted",
        "limitations",
    )
    query_fields = (
        "query_id",
        "provider",
        "role",
        "round",
        "subquestion_id",
        "status",
    )
    source_fields = (
        "source_run_id",
        "provider",
        "role",
        "round",
        "status",
        "stop_reason",
    )
    source_audit = audit.get("source_round_audit", {})
    sources = (
        source_audit.get("sources", ())
        if isinstance(source_audit, Mapping)
        else ()
    )
    return {
        **{key: audit.get(key) for key in summary_fields},
        "queries": [
            {key: query.get(key) for key in query_fields if key in query}
            for query in audit.get("query_manifest", ())
            if isinstance(query, Mapping)
        ],
        "sources": [
            {key: source.get(key) for key in source_fields if key in source}
            for source in sources
            if isinstance(source, Mapping)
        ],
    }


def _allowed_evidence_references(
    bundle: ReportPlanBundle,
    analyses: Sequence[AnalysisRecord],
) -> tuple[dict[str, Any], ...]:
    analysis_by_paper = {item.paper_id: item for item in analyses}
    evidence_papers = {
        str(item["paper_id"])
        for item in bundle.plan["paper_memberships"]
        if item["coverage_disposition"] == "evidence"
    }
    references: list[dict[str, Any]] = []
    for paper in sorted(
        bundle.corpus_snapshot["papers"], key=lambda item: str(item["paper_id"])
    ):
        paper_id = str(paper["paper_id"])
        if paper_id not in evidence_papers:
            continue
        analysis = analysis_by_paper[paper_id]
        for unit in analysis.evidence_units:
            if unit["direction"] not in {"support", "contradict"}:
                continue
            locator = unit["locator"]
            references.append({
                "kind": "paper_evidence",
                "evidence_level": str(paper["evidence_level"]),
                "paper_id": paper_id,
                "analysis_run_id": analysis.analysis_run_id,
                "evidence_unit": unit,
                "locator": f"{locator['kind']} {locator['value']}",
                "search_plan_id": None,
                "source_run_id": None,
                "query_id": None,
                "statistic": None,
                "calculation": None,
            })
    corpus = corpus_evidence_allowlist(bundle.search_audit)
    if corpus.search_plan_ids and corpus.source_run_ids and corpus.query_ids:
        search_plan_id = min(corpus.search_plan_ids)
        source_run_id = min(corpus.source_run_ids)
        query_id = min(corpus.query_ids)
        for statistic, calculation in sorted(corpus.statistics):
            references.append({
                "kind": "corpus_evidence",
                "evidence_level": "corpus_stat",
                "paper_id": None,
                "analysis_run_id": None,
                "evidence_unit": None,
                "locator": None,
                "search_plan_id": search_plan_id,
                "source_run_id": source_run_id,
                "query_id": query_id,
                "statistic": statistic,
                "calculation": calculation,
            })
    return tuple(sorted(references, key=canonical_json))


def _require_allowed_evidence(
    output: Mapping[str, Any],
    allowed: Sequence[Mapping[str, Any]],
) -> None:
    allowed_documents = {canonical_json(dict(item)) for item in allowed}
    for claim in output["claims"]:
        for field in ("supporting_evidence", "contradicting_evidence"):
            for reference in claim[field]:
                if canonical_json(dict(reference)) not in allowed_documents:
                    raise DirectReportError(
                        "one-shot output used an evidence reference outside the exact prompt allowlist"
                    )


def _previous_report_run_id(
    current_report_run_id: str,
    previous: Mapping[str, Any] | None,
    relations: Sequence[Mapping[str, Any]],
) -> str | None:
    if not relations:
        return None
    value = previous.get("report_run_id") if previous is not None else None
    if not isinstance(value, str) or not value or value == current_report_run_id:
        raise DirectReportError(
            "claim relations require a distinct previous report_run_id binding"
        )
    return value


def _coverage(
    plan: Mapping[str, Any], claims: Sequence[Mapping[str, Any]]
) -> CoverageLedger:
    paper_claims: dict[str, set[str]] = defaultdict(set)
    for claim in claims:
        for field in ("supporting_evidence", "contradicting_evidence"):
            for reference in claim[field]:
                if reference["kind"] == "paper_evidence":
                    paper_claims[str(reference["paper_id"])].add(str(claim["claim_id"]))
    papers = []
    missing = []
    for membership in sorted(plan["paper_memberships"], key=lambda item: str(item["paper_id"])):
        paper_id = str(membership["paper_id"])
        evidence_claim_ids = tuple(sorted(paper_claims[paper_id]))
        if membership["coverage_disposition"] == "evidence" and not evidence_claim_ids:
            missing.append(paper_id)
        papers.append(PaperCoverage(
            paper_id=paper_id,
            evidence_claim_ids=evidence_claim_ids,
            consumed_node_ids=(NODE_ID,),
            disposition=str(membership["coverage_disposition"]),
            reason=membership["coverage_reason"],
        ))
    uncovered = tuple(sorted(
        str(claim["claim_id"])
        for claim in claims
        if not claim["supporting_evidence"] and not claim["contradicting_evidence"]
    ))
    return CoverageLedger(tuple(papers), tuple(missing), uncovered)


def _bibliography(
    corpus: Mapping[str, Any],
    claims: Sequence[Mapping[str, Any]],
    *,
    database: Database | None = None,
) -> dict[str, dict[str, Any]]:
    cited = {
        str(reference["paper_id"])
        for claim in claims
        for field in ("supporting_evidence", "contradicting_evidence")
        for reference in claim[field]
        if reference["kind"] == "paper_evidence"
    }
    papers = {str(item["paper_id"]): item for item in corpus["papers"]}
    result = {}
    for paper_id in sorted(cited):
        paper = papers[paper_id]
        result[paper_id] = (
            _canonical_bibliography_metadata(
                database,
                paper_id,
                paper,
                frozen_snapshot_hash=str(corpus.get("snapshot_hash") or ""),
            )
            if database is not None
            else _frozen_bibliography_metadata(paper)
        )
    return result


def _frozen_bibliography_metadata(paper: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in {
            "title": paper.get("title"),
            "authors": paper.get("authors"),
            "year": paper.get("publication_year"),
            "venue_name": paper.get("venue_name"),
            "doi": paper.get("doi"),
            "canonical_url": paper.get("canonical_url"),
        }.items()
        if value is not None
    }


def _canonical_bibliography_metadata(
    database: Database,
    paper_id: str,
    frozen: Mapping[str, Any],
    *,
    frozen_snapshot_hash: str | None = None,
) -> dict[str, Any]:
    """Resolve a fill-only, provenance-bound bibliography metadata overlay.

    The frozen corpus remains the sole Sol input.  A later canonical correction
    may fill a field that was empty at freeze time, but cannot replace any
    non-empty frozen value.  The exact official provenance is included in the
    bibliography artifact so the report hash binds the local correction.
    """
    result = _frozen_bibliography_metadata(frozen)
    frozen_verification_status = str(frozen.get("verification_status") or "")
    frozen_verified = frozen_verification_status in {"verified", "single_source"}
    if _bibliography_metadata_complete(result) and frozen_verified:
        return result
    row = database.connection.execute(
        """SELECT title, authors_json, year, venue_name, doi, canonical_url,
                  verification_status
             FROM papers WHERE paper_id = ?""",
        (paper_id,),
    ).fetchone()
    if row is None or row["verification_status"] not in {"verified", "single_source"}:
        raise DirectReportError(
            f"canonical bibliography metadata is not verified for {paper_id}"
        )
    try:
        authors = json.loads(str(row["authors_json"]))
    except json.JSONDecodeError as error:
        raise DirectReportError(
            f"canonical bibliography authors are malformed for {paper_id}"
        ) from error
    live = {
        "title": row["title"],
        "authors": authors,
        "year": row["year"],
        "venue_name": row["venue_name"],
        "doi": row["doi"],
        "canonical_url": row["canonical_url"],
    }
    for field in ("title", "authors", "year", "venue_name", "doi", "canonical_url"):
        frozen_value = result.get(field)
        live_value = live.get(field)
        if (
            _bibliography_value_present(frozen_value)
            and _bibliography_value_present(live_value)
            and canonical_json(frozen_value) != canonical_json(live_value)
        ):
            raise DirectReportError(
                f"canonical bibliography metadata would overwrite frozen {field} for {paper_id}"
            )

    overlay_fields: dict[str, Any] = {}
    for field in ("title", "authors", "year", "venue_name"):
        if _bibliography_value_present(result.get(field)):
            continue
        value = live.get(field)
        if not _bibliography_value_present(value):
            continue
        provenance = _official_field_provenance(
            database, paper_id, field, value, frozen
        )
        if provenance is None:
            raise DirectReportError(
                f"canonical bibliography {field} lacks official provenance for {paper_id}"
            )
        result[field] = value
        overlay_fields[field] = provenance
    if not (
        _bibliography_value_present(result.get("doi"))
        or _bibliography_value_present(result.get("canonical_url"))
    ):
        for field in ("doi", "canonical_url"):
            value = live.get(field)
            if not _bibliography_value_present(value):
                continue
            provenance = _official_field_provenance(
                database, paper_id, field, value, frozen
            )
            if provenance is None:
                continue
            result[field] = value
            overlay_fields[field] = provenance
            break
    if not _bibliography_metadata_complete(result):
        raise DirectReportError(
            f"canonical metadata is incomplete for citation {paper_id}"
        )
    if overlay_fields or not frozen_verified:
        overlay = {
            "paper_id": paper_id,
            "mode": "fill_only",
            "frozen_snapshot_hash": frozen_snapshot_hash,
            "frozen_verification_status": frozen_verification_status,
            "verification_status": str(row["verification_status"]),
            "fields": overlay_fields,
        }
        result["canonical_metadata_overlay"] = {
            **overlay,
            "overlay_hash": content_hash(overlay),
        }
    return result


def _bibliography_metadata_complete(metadata: Mapping[str, Any]) -> bool:
    authors = metadata.get("authors")
    return bool(
        str(metadata.get("title") or "").strip()
        and isinstance(authors, Sequence)
        and not isinstance(authors, (str, bytes))
        and any(str(author).strip() for author in authors)
        and isinstance(metadata.get("year"), int)
        and not isinstance(metadata.get("year"), bool)
        and str(metadata.get("venue_name") or "").strip()
        and (
            str(metadata.get("doi") or "").strip()
            or str(metadata.get("canonical_url") or "").strip()
        )
    )


def _bibliography_value_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(str(item).strip() for item in value)
    return True


def _official_field_provenance(
    database: Database,
    paper_id: str,
    field: str,
    value: Any,
    frozen: Mapping[str, Any],
) -> dict[str, Any] | None:
    rows = database.connection.execute(
        """SELECT fp.provenance_id, fp.source_id, fp.field_value_json,
                  fp.observed_at, ps.provider, ps.landing_url,
                  ps.raw_metadata_json
             FROM paper_field_provenance fp
             JOIN paper_sources ps ON ps.source_id = fp.source_id
            WHERE fp.paper_id = ? AND fp.field_name = ?
              AND ps.host_type = 'official'
            ORDER BY fp.observed_at DESC, fp.source_id""",
        (paper_id, field),
    ).fetchall()
    frozen_url = str(frozen.get("canonical_url") or "").rstrip("/")
    for row in rows:
        try:
            observed = json.loads(str(row["field_value_json"]))
            source_metadata = json.loads(str(row["raw_metadata_json"]))
        except json.JSONDecodeError:
            continue
        landing_url = str(row["landing_url"] or "").rstrip("/")
        if canonical_json(observed) != canonical_json(value):
            continue
        if frozen_url and landing_url != frozen_url:
            continue
        return {
            "provenance_id": str(row["provenance_id"]),
            "source_id": str(row["source_id"]),
            "provider": str(row["provider"]),
            "landing_url": str(row["landing_url"] or ""),
            "observed_at": str(row["observed_at"]),
            "field_value_hash": content_hash(value),
            "source_metadata_hash": content_hash(source_metadata),
        }
    return None


def _deterministic_verify(
    bundle: Mapping[str, Any],
    *,
    previous: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    checklist = verify_report(
        plan=bundle["plan"],
        document=bundle["document"],
        claims=bundle["claims"],
        coverage=bundle["coverage"],
        bibliography=bundle["bibliography"],
        comparison_groups=bundle["comparison_groups"],
        search_audit=bundle["search_audit"],
        corpus_snapshot=bundle["corpus_snapshot"],
        previous=previous,
        claim_relations=bundle["claim_relations"],
    )
    markdown, sidecar = render_markdown(
        plan=bundle["plan"],
        document=bundle["document"],
        claims=bundle["claims"],
        bibliography=bundle["bibliography"],
        search_audit=bundle["search_audit"],
        corpus_snapshot=bundle["corpus_snapshot"],
    )
    return {
        **checklist,
        "report_document_hash": content_hash(bundle["document"]),
        "rendered_markdown_hash": content_hash(markdown),
        "sidecar_hash": content_hash(sidecar),
    }


def _deterministic_audit(bundle: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "audit_pass": "deterministic",
        "report_document_hash": content_hash(bundle["document"]),
        "report_artifact_hash": report_artifact_hash(
            document=bundle["document"],
            claims=bundle["claims"],
            coverage=bundle["coverage"],
            comparison_groups=bundle["comparison_groups"],
            claim_relations=bundle["claim_relations"],
            bibliography=bundle["bibliography"],
        ),
        "report_plan_hash": bundle["plan"]["plan_hash"],
        "rubric_hash": audit_rubric_hash(),
        "search_limitations_hash": content_hash(list(
            audit_search_limitations(bundle["search_audit"], bundle["corpus_snapshot"])
        )),
        "coverage_complete": True,
        "coverage_ledger": audit_coverage_ledger(bundle["document"], bundle["claims"]),
        "findings": [],
    }


def _upsert_local_audit_run(
    connection: Any,
    report_run_id: str,
    bundle: Mapping[str, Any],
    audit: Mapping[str, Any],
    target: Path,
    execution_mode: str,
    previous: Mapping[str, Any] | None,
) -> None:
    mutable = {
        "document": bundle["document"],
        "claims": list(bundle["claims"]),
        "comparison_groups": bundle["comparison_groups"],
        "claim_relations": list(bundle["claim_relations"]),
        "coverage": asdict(bundle["coverage"]),
        "bibliography": bundle["bibliography"],
    }
    artifact_hash = report_artifact_hash(
        document=bundle["document"],
        claims=bundle["claims"],
        coverage=bundle["coverage"],
        comparison_groups=bundle["comparison_groups"],
        claim_relations=bundle["claim_relations"],
        bibliography=bundle["bibliography"],
    )
    relative = str(target.relative_to(target.parents[1]))
    connection.execute(
        """INSERT INTO report_audit_runs(
               report_run_id, input_snapshot_hash, base_artifact_hash,
               current_artifact_hash, current_bundle_json, rubric_hash,
               profile, model_id, reasoning_effort, config_hash, execution_mode,
               worst_case_calls, worst_case_input_tokens, repair_count, status,
               final_audit_step, published_relative_path, error_json,
               completed_at, updated_at
           ) VALUES (?, ?, ?, ?, ?, ?, 'stage4b_summary_sol', 'gpt-5.6-sol',
                     'high', ?, ?, 1, 1, 0, 'complete', NULL, ?, NULL,
                     CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
           ON CONFLICT(report_run_id) DO UPDATE SET
               input_snapshot_hash = excluded.input_snapshot_hash,
               base_artifact_hash = excluded.base_artifact_hash,
               current_artifact_hash = excluded.current_artifact_hash,
               current_bundle_json = excluded.current_bundle_json,
               rubric_hash = excluded.rubric_hash,
               config_hash = excluded.config_hash,
               execution_mode = excluded.execution_mode,
               status = 'complete',
               published_relative_path = excluded.published_relative_path,
               error_json = NULL,
               completed_at = CURRENT_TIMESTAMP,
               updated_at = CURRENT_TIMESTAMP""",
        (
            report_run_id,
            content_hash({
                "plan": bundle["plan"],
                "search_audit": bundle["search_audit"],
                "corpus_snapshot": bundle["corpus_snapshot"],
                "previous": previous,
            }),
            artifact_hash,
            artifact_hash,
            _json(mutable),
            audit["rubric_hash"],
            one_shot_validation_config_hash(),
            execution_mode,
            relative,
        ),
    )


def _save_artifact(connection: Any, stored: StoredArtifact) -> str:
    artifact_id = "artifact-" + stored.artifact_hash
    row = connection.execute(
        """SELECT artifact_id, paper_id, artifact_kind, relative_path,
                  mime_type, byte_size, provenance_json, processing_status
             FROM artifacts WHERE sha256 = ?""",
        (stored.artifact_hash,),
    ).fetchone()
    if row is not None:
        expected = (
            None,
            "report",
            stored.relative_path,
            stored.mime_type,
            stored.size_bytes,
            {"stage": "stage4b", "content_hash": stored.artifact_hash},
            "available",
        )
        actual = (
            row["paper_id"],
            row["artifact_kind"],
            row["relative_path"],
            row["mime_type"],
            row["byte_size"],
            _object(str(row["provenance_json"]).encode("utf-8"), "report artifact provenance"),
            row["processing_status"],
        )
        if actual != expected:
            raise DirectReportError(
                "report artifact metadata conflicts with existing content"
            )
        return str(row["artifact_id"])
    connection.execute(
        """INSERT INTO artifacts(
               artifact_id, paper_id, artifact_kind, relative_path, mime_type,
               byte_size, sha256, provenance_json
           ) VALUES (?, NULL, 'report', ?, ?, ?, ?, ?)""",
        (
            artifact_id,
            stored.relative_path,
            stored.mime_type,
            stored.size_bytes,
            stored.artifact_hash,
            _json({"stage": "stage4b", "content_hash": stored.artifact_hash}),
        ),
    )
    return artifact_id


def _decision(decision: ProcessingDecision) -> dict[str, Any]:
    value = asdict(decision)
    value["outcome"] = decision.outcome.value
    value["audit_hash"] = decision.audit_hash
    return value


def _object(payload: bytes, label: str) -> Mapping[str, Any]:
    value = json.loads(payload)
    if not isinstance(value, Mapping):
        raise DirectReportError(f"{label} must be a JSON object")
    return value


def _json(value: Any) -> str:
    return canonical_json(value).decode("utf-8")


def _error(value: str | None) -> str | None:
    if value is None:
        return None
    document = json.loads(value)
    return str(document.get("message") or document.get("error"))


def _is_procedural_reference_note(claim: Mapping[str, Any]) -> bool:
    """Recognize the sole evidence-free note allowed outside the claim ledger.

    The references appendix is rendered from frozen local metadata, so a model
    may describe that procedure without turning it into a research claim.  No
    other evidence-free model output is normalized away.  The normalizer then
    requires this candidate to be the exclusive claim on one uncited references
    block and replaces all of its model text with ``LOCAL_REFERENCES_NOTE``.
    """
    semantic_text = " ".join(
        str(claim.get(field) or "")
        for field in (
            "subject_id",
            "predicate_id",
            "object_or_scope_id",
            "qualifier_context",
            "claim_text",
        )
    ).casefold()
    names_references = any(
        marker in semantic_text
        for marker in ("bibliograph", "reference", "citation", "参考文献", "书目")
    )
    names_local_generation = any(
        marker in semantic_text
        for marker in ("local", "coordinator", "renderer", "canonical", "本地", "协调器", "渲染")
    )
    return (
        claim.get("report_section") == "references_and_appendices"
        and claim.get("claim_type") == "recommendation"
        and not claim.get("supporting_evidence")
        and not claim.get("contradicting_evidence")
        and names_references
        and names_local_generation
    )


def _timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )
