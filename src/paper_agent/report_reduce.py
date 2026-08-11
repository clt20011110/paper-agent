"""Policy-gated, resumable Sol reduce coordinator for Stage 4b."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from .approval import ApprovalError, require_valid_approval
from .artifacts import ArtifactStore, StoredArtifact
from .canonical import canonical_json, content_hash
from .codex_exec import (
    CALL_KIND_PROMPTS,
    CALL_KIND_SCHEMAS,
    FROZEN_PROFILES,
    CodexExec,
    CodexExecRequest,
    CodexExecResult,
    InvocationMetadata,
    prompt_directory,
)
from .processing import (
    PROCESSING_MODEL,
    PROCESSING_PROVIDER,
    SUMMARY_MODEL,
    ModelInvocation,
    ProcessingDecision,
    ProcessingGate,
    ProcessingRequest,
)
from .report_artifacts import is_local_references_block
from .report_budget import canonical_report_budget
from .report_config import ReportResources
from .report_invocations import (
    ReportInvocationError,
    register_report_invocation,
    require_report_invocation,
)
from .report_plan import (
    ReportPlanDriftError,
    ReportPlanError,
    assert_report_runtime_matches,
    compile_report_plan,
)
from .reporting import (
    PAPER_MARKER,
    AnalysisRecord,
    CorpusEvidenceAllowlist,
    ReduceNode,
    ReducePlan,
    ReportPlanner,
    SectionRule,
    SynthesisValidator,
    corpus_evidence_allowlist,
)
from .schema import SchemaValidationError, schema_directory, validate
from .storage import Database


PROFILE = "stage4b_summary_sol"
REASONING_EFFORT = "high"
MAX_RETRIES = FROZEN_PROFILES[PROFILE].max_retries
STAGE4_PROFILE = "stage4_analysis_luna"
STAGE4_REASONING_EFFORT = "medium"
STAGE4_SCHEMA = "paper-analysis.schema.json"
STAGE4_PROMPT = "paper-analysis.md"
STAGE4_MAX_RETRIES = FROZEN_PROFILES[STAGE4_PROFILE].max_retries
PURPOSE = "research_synthesis"
IMPLEMENTATION_VERSION = "stage4b-reduce-v2"
DERIVED_KINDS = frozenset({"analysis", "evidence", "claim_ledger", "report_draft"})
PROMPT_TOKEN_ESTIMATOR = "utf8-byte-upper-bound-v1"
OUTPUT_BYTES_PER_ESTIMATED_TOKEN = 15
MIN_NODE_OUTPUT_BYTES = 4_096
MAX_NODE_OUTPUT_BYTES = 262_144
LEASE_SECONDS = FROZEN_PROFILES[PROFILE].timeout_seconds * (MAX_RETRIES + 1) + 60
OUTPUT_KIND = {
    "section_reduce": "evidence",
    "cross_section_reduce": "claim_ledger",
    "final_reduce": "report_draft",
}


def stage4b_reduce_config_hash(
    processing_policy_hash: str,
    *,
    execution_mode: str = "attended",
    implementation_version: str = IMPLEMENTATION_VERSION,
    schema_root: Path | None = None,
    prompt_root: Path | None = None,
    resources: ReportResources | None = None,
) -> str:
    if execution_mode not in {"attended", "unattended"}:
        raise ValueError("execution_mode must be attended or unattended")
    report_resources = resources or ReportResources.defaults(
        schema_root=schema_root, prompt_root=prompt_root
    )
    report_resources.validate_files()
    schema_hashes = {
        call_kind: _json_hash(report_resources.schema(call_kind))
        for call_kind in CALL_KIND_SCHEMAS
    }
    service_schema_hashes = {
        call_kind: report_resources.service_schema_hash(call_kind)
        for call_kind in CALL_KIND_SCHEMAS
    }
    prompt_hashes = {
        call_kind: sha256(report_resources.prompt_paths[call_kind].read_bytes()).hexdigest()
        for call_kind in CALL_KIND_PROMPTS
    }
    return content_hash(_stage4b_config_document(
        processing_policy_hash,
        schema_hashes,
        service_schema_hashes,
        prompt_hashes,
        execution_mode,
        implementation_version,
    ))


def _stage4b_config_document(
    processing_policy_hash: str,
    schema_hashes: Mapping[str, str],
    service_schema_hashes: Mapping[str, str],
    prompt_hashes: Mapping[str, str],
    execution_mode: str,
    implementation_version: str,
) -> dict[str, Any]:
    return {
        "profile": PROFILE,
        "model": SUMMARY_MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "schema_hashes": dict(schema_hashes),
        "service_schema_hashes": dict(service_schema_hashes),
        "prompt_hashes": dict(prompt_hashes),
        "processing_policy_hash": processing_policy_hash,
        "prompt_token_estimator": PROMPT_TOKEN_ESTIMATOR,
        "output_bytes_per_estimated_token": OUTPUT_BYTES_PER_ESTIMATED_TOKEN,
        "min_node_output_bytes": MIN_NODE_OUTPUT_BYTES,
        "max_node_output_bytes": MAX_NODE_OUTPUT_BYTES,
        "lease_seconds": LEASE_SECONDS,
        "execution_mode": execution_mode,
        "implementation_version": implementation_version,
    }


class ReportReduceError(RuntimeError):
    pass


class SolBudgetError(ReportReduceError):
    pass


class SolOutputError(ReportReduceError):
    pass


class SolInvoker(Protocol):
    def invoke(self, request: CodexExecRequest) -> CodexExecResult: ...


@dataclass(frozen=True, slots=True)
class FrozenDerivedArtifact:
    artifact_hash: str
    payload: bytes
    artifact_kind: str
    input_scope: str
    license: str | None
    access_basis: str
    lineage_hash: str
    source_lineage_hashes: tuple[str, ...]
    source_paper_ids: tuple[str, ...] = ()
    paper_id: str | None = None
    domain: str | None = None
    mode: str = "attended"
    collection_id: str | None = None
    collection_snapshot_hash: str | None = None
    selection_snapshot_hash: str | None = None

    def __post_init__(self) -> None:
        if self.artifact_kind not in DERIVED_KINDS:
            raise ValueError(f"unsupported Stage 4b input kind: {self.artifact_kind}")
        if not _is_sha256(self.artifact_hash) or sha256(self.payload).hexdigest() != self.artifact_hash:
            raise ValueError("derived artifact hash does not match its payload")
        if self.input_scope not in {"full_pdf", "abstract_only", "metadata_only"}:
            raise ValueError("derived artifact input_scope is invalid")
        if not _is_sha256(self.lineage_hash):
            raise ValueError("derived artifact requires a lineage hash")
        if any(not _is_sha256(value) for value in self.source_lineage_hashes):
            raise ValueError("derived artifact source lineage hashes must be SHA-256 digests")
        if self.source_paper_ids != tuple(sorted(set(self.source_paper_ids))):
            raise ValueError("derived artifact source paper IDs must be unique and sorted")
        if self.access_basis not in {
            "open_license", "public_read_only", "user_subscription", "user_supplied", "unknown"
        }:
            raise ValueError("derived artifact access_basis is invalid")
        if self.mode not in {"attended", "unattended"}:
            raise ValueError("derived artifact mode is invalid")

    def processing_request(self) -> ProcessingRequest:
        return ProcessingRequest(
            artifact_hash=self.artifact_hash,
            artifact=self.artifact_kind,
            input_scope=self.input_scope,
            license=self.license,
            access_basis=self.access_basis,
            purpose=PURPOSE,
            data_category=self.artifact_kind,
            provider=PROCESSING_PROVIDER,
            model=SUMMARY_MODEL,
            paper_id=self.paper_id,
            source_paper_ids=self.source_paper_ids,
            domain=self.domain,
            mode=self.mode,
            collection_id=self.collection_id,
            collection_snapshot_hash=self.collection_snapshot_hash,
            selection_snapshot_hash=self.selection_snapshot_hash,
            lineage_hash=self.lineage_hash,
            derived_bytes=self.payload,
        )


@dataclass(frozen=True, slots=True)
class ReduceNodeResult:
    node_id: str
    status: str
    output_hash: str | None = None
    resumed: bool = False
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ReportReduceResult:
    report_run_id: str
    status: str
    nodes: tuple[ReduceNodeResult, ...]

    @property
    def final_output_hash(self) -> str | None:
        for node in reversed(self.nodes):
            if node.status == "complete" and node.node_id.startswith("final_reduce:"):
                return node.output_hash
        return None


class SolReduceCoordinator:
    """Execute one frozen reduce tree without bypassing artifact policy."""

    def __init__(
        self,
        database: Database,
        artifact_store: ArtifactStore,
        gate: ProcessingGate,
        analyses: Sequence[AnalysisRecord],
        sections: Sequence[SectionRule],
        memberships: Mapping[str, Sequence[str]],
        *,
        invoker_factory: Callable[[], SolInvoker] = CodexExec,
        schema_root: Path | None = None,
        prompt_root: Path | None = None,
        resources: ReportResources | None = None,
        rubric_path: Path | None = None,
        implementation_version: str = IMPLEMENTATION_VERSION,
        execution_mode: str = "attended",
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if execution_mode not in {"attended", "unattended"}:
            raise ValueError("execution_mode must be attended or unattended")
        self.database = database
        self.artifact_store = artifact_store
        self.gate = gate
        self.analyses = tuple(analyses)
        self.sections = tuple(sections)
        self.memberships = {paper_id: tuple(values) for paper_id, values in memberships.items()}
        self.invoker_factory = invoker_factory
        self.schema_root = schema_directory(schema_root)
        self.prompt_root = prompt_directory() if prompt_root is None else prompt_root
        self.resources = resources or ReportResources.defaults(
            schema_root=schema_root, prompt_root=prompt_root
        )
        self.resources.validate_files()
        self.rubric_path = rubric_path
        self.implementation_version = implementation_version
        self.execution_mode = execution_mode
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.schemas = {
            call_kind: self.resources.schema(call_kind)
            for call_kind in CALL_KIND_SCHEMAS
        }
        self.analysis_schema = json.loads(
            (self.schema_root / "paper-analysis.schema.json").read_text(encoding="utf-8")
        )
        self.analysis_schema_hash = _json_hash(self.analysis_schema)
        self.analysis_prompt_hash = sha256(
            (self.prompt_root / STAGE4_PROMPT).read_bytes()
        ).hexdigest()
        self.schema_hashes = {
            call_kind: _json_hash(schema) for call_kind, schema in self.schemas.items()
        }
        self.service_schema_hashes = {
            call_kind: self.resources.service_schema_hash(call_kind)
            for call_kind in CALL_KIND_SCHEMAS
        }
        self.prompt_hashes = {
            call_kind: sha256(self.resources.prompt_paths[call_kind].read_bytes()).hexdigest()
            for call_kind in CALL_KIND_PROMPTS
        }
        self.config_hash = content_hash(_stage4b_config_document(
            self.gate.policy.hash,
            self.schema_hashes,
            self.service_schema_hashes,
            self.prompt_hashes,
            execution_mode,
            implementation_version,
        ))
        from .report_audit import stage4b_audit_config_hash

        self.audit_config_hash = stage4b_audit_config_hash(
            self.gate.policy.hash,
            execution_mode=execution_mode,
            schema_root=self.schema_root,
            prompt_root=self.prompt_root,
            resources=self.resources,
            rubric_path=self.rubric_path,
        )

    def run(
        self,
        report_run_id: str,
        pipeline_run_id: str,
        approved_plan: Mapping[str, Any],
        reduce_plan: ReducePlan,
        source_artifacts: Sequence[FrozenDerivedArtifact],
        *,
        corpus_snapshot: Mapping[str, Any],
        search_audit_pack: Mapping[str, Any],
        processing_grants: Mapping[str, str] | None = None,
        worker_id: str | None = None,
    ) -> ReportReduceResult:
        self._verify_plan(
            approved_plan,
            reduce_plan,
            corpus_snapshot=corpus_snapshot,
            search_audit_pack=search_audit_pack,
        )
        corpus_evidence = corpus_evidence_allowlist(search_audit_pack)
        chunks = {chunk.node_id: chunk for chunk in reduce_plan.chunks}
        sources = self._source_map(chunks, source_artifacts, corpus_snapshot)
        self._verify_canonical_reduce_plan(approved_plan, reduce_plan)
        prompt_bounds, output_limits, audit_bounds = self._preflight_prompt_budget(
            report_run_id,
            approved_plan,
            reduce_plan,
            chunks,
            sources,
            corpus_snapshot,
            search_audit_pack,
            corpus_evidence,
        )
        self._ensure_run(
            report_run_id,
            pipeline_run_id,
            approved_plan,
            reduce_plan,
            corpus_snapshot,
            search_audit_pack,
            prompt_bounds,
            output_limits,
            audit_bounds,
        )
        attempts = int(approved_plan["budget"]["max_retries"]) + 1
        generation_calls = len(reduce_plan.nodes) * attempts
        generation_bound = sum(prompt_bounds.values()) * attempts
        budget_error: SolBudgetError | None = None
        if (
            generation_calls + audit_bounds.worst_case_calls
            > int(approved_plan["budget"]["max_sol_calls"])
        ):
            budget_error = SolBudgetError(
                "rendered Sol call bound leaves no room for the frozen audit/repair gate"
            )
        elif (
            generation_bound + audit_bounds.worst_case_input_tokens
            > int(approved_plan["budget"]["max_input_tokens"])
        ):
            budget_error = SolBudgetError(
                "rendered Sol prompt upper bound plus audit/repair reserve exceeds the approved input-token budget"
            )
        if budget_error is not None:
            self._set_run_status(report_run_id, pipeline_run_id, "incomplete")
            raise budget_error
        owner = worker_id or f"report-worker-{uuid4()}"
        self._recover_stale_nodes(report_run_id, self._now())
        self._set_run_status(report_run_id, pipeline_run_id, "running")

        outputs: dict[str, Mapping[str, Any]] = {}
        output_artifacts: dict[str, FrozenDerivedArtifact] = {}
        results: list[ReduceNodeResult] = []
        terminal_failure = False
        for node in reduce_plan.nodes:
            moment = self._now()
            decision_now = _timestamp(moment)
            row = self._node_row(report_run_id, node.node_id)
            self._assert_node_binding(row, node, prompt_bounds[node.node_id], output_limits[node.node_id])
            if terminal_failure and row["status"] not in {"complete", "failed"}:
                results.append(ReduceNodeResult(node.node_id, "pending"))
                continue
            inputs = self._node_inputs(node, chunks, sources, output_artifacts)
            if row["status"] == "complete":
                if inputs is None:
                    raise ReportReduceError(f"completed node has unavailable dependencies: {node.node_id}")
                output, artifact = self._load_completed(
                    row,
                    node,
                    inputs,
                    outputs,
                    approved_plan,
                    output_limits[node.node_id],
                    decision_now,
                    corpus_evidence,
                )
                outputs[node.node_id] = output
                output_artifacts[node.node_id] = artifact
                results.append(ReduceNodeResult(node.node_id, "complete", artifact.artifact_hash, resumed=True))
                continue
            if row["status"] == "failed":
                error = _error_message(row["error_json"])
                results.append(ReduceNodeResult(node.node_id, "failed", error=error))
                terminal_failure = True
                continue
            if row["status"] == "running":
                results.append(ReduceNodeResult(node.node_id, "running"))
                continue
            if inputs is None:
                results.append(ReduceNodeResult(node.node_id, "pending"))
                continue
            result, output, artifact = self._execute_node(
                report_run_id,
                approved_plan,
                node,
                inputs,
                outputs,
                processing_grants or {},
                decision_now,
                owner,
                moment,
                prompt_bounds[node.node_id],
                output_limits[node.node_id],
                audit_bounds,
                corpus_evidence,
            )
            results.append(result)
            terminal_failure = terminal_failure or result.status == "failed"
            if output is not None and artifact is not None:
                outputs[node.node_id] = output
                output_artifacts[node.node_id] = artifact

        statuses = {item.status for item in results}
        if statuses == {"complete"}:
            status = "generation_complete"
            database_status = "running"
        elif "failed" in statuses:
            status = "failed"
            database_status = "failed"
        else:
            status = "incomplete"
            database_status = "incomplete"
        self._set_run_status(report_run_id, pipeline_run_id, database_status)
        return ReportReduceResult(report_run_id, status, tuple(results))

    def _now(self) -> datetime:
        moment = self.clock()
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        return moment.astimezone(timezone.utc)

    def _verify_plan(
        self,
        plan: Mapping[str, Any],
        reduce_plan: ReducePlan,
        *,
        corpus_snapshot: Mapping[str, Any],
        search_audit_pack: Mapping[str, Any],
    ) -> None:
        try:
            require_valid_approval(plan, "plan_hash")
            self.resources.validate(plan, "planning_assist")
            runtime_plan = compile_report_plan(
                plan,
                corpus_snapshot=corpus_snapshot,
                search_audit_pack=search_audit_pack,
                plan_id=str(plan["plan_id"]),
                created_at=str(plan["created_at"]),
                schema_root=self.schema_root,
                prompt_root=self.prompt_root,
                resources=self.resources,
            )
            assert_report_runtime_matches(
                plan,
                runtime_plan,
                corpus_snapshot=corpus_snapshot,
                search_audit_pack=search_audit_pack,
            )
        except (
            ApprovalError,
            SchemaValidationError,
            ReportPlanError,
            ReportPlanDriftError,
        ) as error:
            raise ReportReduceError(str(error)) from error
        if plan["schema_hash"] != self.schema_hashes["planning_assist"]:
            raise ReportReduceError("approved ReportPlan schema hash does not match the frozen runtime")
        if plan["stage4b_config_hash"] != self.config_hash:
            raise ReportReduceError("approved ReportPlan Stage 4b configuration has drifted")
        if plan["stage4b_audit_config_hash"] != self.audit_config_hash:
            raise ReportReduceError(
                "approved ReportPlan Stage 4b audit configuration has drifted"
            )
        if dict(plan["prompt_hashes"]) != self.prompt_hashes:
            raise ReportReduceError("approved ReportPlan prompt hashes do not match the frozen runtime")
        budget = plan["budget"]
        if int(budget["audit_calls"]) != 2 or int(budget["repair_calls"]) != 1:
            raise SolBudgetError(
                "approved ReportPlan must reserve exactly two audits and one repair"
            )
        if int(budget["max_retries"]) != MAX_RETRIES:
            raise SolBudgetError("approved retry budget does not match the frozen Sol profile")
        attempts = int(budget["max_retries"]) + 1
        expected_generation_calls = len(reduce_plan.nodes)
        expected_generation_tokens = sum(node.input_tokens for node in reduce_plan.nodes)
        expected_worst_calls = (
            expected_generation_calls + int(budget["audit_calls"]) + int(budget["repair_calls"])
        ) * attempts
        if (
            reduce_plan.budget.generation_calls != expected_generation_calls
            or reduce_plan.budget.audit_calls != int(budget["audit_calls"])
            or reduce_plan.budget.repair_calls != int(budget["repair_calls"])
            or reduce_plan.budget.generation_input_tokens != expected_generation_tokens
            or reduce_plan.budget.worst_case_calls != expected_worst_calls
        ):
            raise SolBudgetError("frozen reduce-tree budget does not match its nodes")
        if reduce_plan.budget.worst_case_calls > int(budget["max_sol_calls"]):
            raise SolBudgetError("frozen reduce tree exceeds the approved Sol call budget")
        expected_sections = tuple(
            SectionRule(
                str(section["id"]),
                frozenset(str(item) for item in section["subquestion_ids"]),
                frozenset(str(item) for item in section["allowed_evidence_levels"]),
            )
            for section in plan["sections"]
        )
        if self.sections != expected_sections:
            raise ReportReduceError("runtime section validation rules have drifted from ReportPlan")
        self._verify_tree_bindings(plan, reduce_plan)

    def _verify_tree_bindings(self, plan: Mapping[str, Any], reduce_plan: ReducePlan) -> None:
        analyses = {item.paper_id: item for item in self.analyses}
        if len(analyses) != len(self.analyses):
            raise ReportReduceError("Stage 4 analysis records contain duplicate paper IDs")
        memberships = {
            str(item["paper_id"]): tuple(str(section) for section in item["section_ids"])
            for item in plan["paper_memberships"]
        }
        if set(memberships) != set(analyses) or memberships != self.memberships:
            raise ReportReduceError("reduce inputs do not match approved paper memberships")
        expected_assignments = Counter(
            (paper_id, section_id)
            for paper_id, section_ids in memberships.items()
            for section_id in section_ids
        )
        actual_assignments: Counter[tuple[str, str]] = Counter()
        chunks = {chunk.node_id: chunk for chunk in reduce_plan.chunks}
        if len(chunks) != len(reduce_plan.chunks):
            raise ReportReduceError("semantic chunks contain duplicate node IDs")
        classification_axes = tuple(str(axis) for axis in plan["classification_axes"])
        for chunk in reduce_plan.chunks:
            if len(chunk.paper_ids) != len(chunk.analysis_hashes):
                raise ReportReduceError("semantic chunk paper and analysis bindings disagree")
            for paper_id, analysis_hash in zip(chunk.paper_ids, chunk.analysis_hashes, strict=True):
                actual_assignments[(paper_id, chunk.section_id)] += 1
                if paper_id not in analyses or analyses[paper_id].analysis_hash != analysis_hash:
                    raise ReportReduceError("semantic chunk uses a foreign analysis artifact")
                expected_key = tuple(
                    (
                        axis,
                        tuple(
                            sorted(
                                str(value)
                                for value in analyses[paper_id].classifications.get(axis, ())
                            )
                        ),
                    )
                    for axis in classification_axes
                )
                if chunk.classification_key != expected_key:
                    raise ReportReduceError("semantic chunk classification key has drifted")
            if chunk.input_tokens != sum(analyses[paper_id].input_tokens for paper_id in chunk.paper_ids):
                raise ReportReduceError("semantic chunk token count has drifted")
        if actual_assignments != expected_assignments:
            raise ReportReduceError("semantic chunks changed approved section membership")

        seen: set[str] = set()
        leaf_node_ids: set[str] = set()
        prior: dict[str, ReduceNode] = {}
        final_nodes = 0
        for node in reduce_plan.nodes:
            if node.call_kind not in {"section_reduce", "cross_section_reduce", "final_reduce"}:
                raise ReportReduceError(f"unsupported reduce call kind: {node.call_kind}")
            if node.call_kind == "section_reduce" and len(node.section_ids) != 1:
                raise ReportReduceError("section_reduce cannot cross section boundaries")
            if node.call_kind == "cross_section_reduce" and len(node.section_ids) < 2:
                raise ReportReduceError("cross_section_reduce requires multiple sections")
            if not set(node.dependency_ids).issubset(seen):
                raise ReportReduceError("reduce tree is not topologically ordered")
            if node.node_id in seen:
                raise ReportReduceError("reduce tree contains duplicate node IDs")
            if not node.dependency_ids:
                chunk = chunks.get(node.node_id)
                if (
                    chunk is None
                    or node.call_kind != "section_reduce"
                    or node.section_ids != (chunk.section_id,)
                    or node.paper_ids != chunk.paper_ids
                    or node.input_tokens != chunk.input_tokens
                    or node.planned_input_hash != content_hash({
                        "section_id": chunk.section_id,
                        "classification_key": chunk.classification_key,
                        "analysis_hashes": chunk.analysis_hashes,
                    })
                ):
                    raise ReportReduceError("section leaf binding has drifted")
                leaf_node_ids.add(node.node_id)
            else:
                dependencies = tuple(prior[value] for value in node.dependency_ids)
                expected_sections = tuple(dict.fromkeys(
                    section for dependency in dependencies for section in dependency.section_ids
                ))
                expected_papers = tuple(sorted({
                    paper for dependency in dependencies for paper in dependency.paper_ids
                }))
                expected_hash = content_hash({
                    "call_kind": node.call_kind,
                    "dependencies": [
                        {"node_id": dependency.node_id, "planned_input_hash": dependency.planned_input_hash}
                        for dependency in dependencies
                    ],
                })
                if (
                    node.section_ids != expected_sections
                    or node.paper_ids != expected_papers
                    or node.planned_input_hash != expected_hash
                ):
                    raise ReportReduceError("reduce parent binding has drifted")
            seen.add(node.node_id)
            prior[node.node_id] = node
            final_nodes += node.call_kind == "final_reduce"
        if final_nodes != 1 or reduce_plan.nodes[-1].call_kind != "final_reduce":
            raise ReportReduceError("reduce tree requires one final_reduce root")
        if leaf_node_ids != set(chunks):
            raise ReportReduceError("every semantic chunk requires exactly one reduce leaf")
        root = reduce_plan.nodes[-1]
        expected_root_sections = tuple(str(section["id"]) for section in plan["sections"])
        expected_root_papers = tuple(sorted(memberships))
        if root.section_ids != expected_root_sections or root.paper_ids != expected_root_papers:
            raise ReportReduceError("final reduce root does not cover the approved corpus and sections")
        if len(root.dependency_ids) != 1 or prior[root.dependency_ids[0]].call_kind != "cross_section_reduce":
            raise ReportReduceError("final_reduce must consume exactly one cross-section root")
        if any(
            node.call_kind != "final_reduce"
            and node.dependency_ids
            and len(node.dependency_ids) != 2
            for node in reduce_plan.nodes
        ):
            raise ReportReduceError("reduce parents must use the frozen binary tree shape")
        consumers = Counter(
            dependency
            for node in reduce_plan.nodes
            for dependency in node.dependency_ids
        )
        if any(consumers[node.node_id] != 1 for node in reduce_plan.nodes[:-1]):
            raise ReportReduceError("every reduce node must feed the single final root exactly once")
        reduce_output_tokens = root.input_tokens
        if reduce_output_tokens < 1 or any(
            node.dependency_ids
            and node.input_tokens != len(node.dependency_ids) * reduce_output_tokens
            for node in reduce_plan.nodes
        ):
            raise ReportReduceError("reduce tree token estimates have drifted")

    def _verify_canonical_reduce_plan(
        self, plan: Mapping[str, Any], reduce_plan: ReducePlan
    ) -> None:
        aggregation = plan["aggregation"]
        expected = ReportPlanner(
            plan,
            self.analyses,
            max_chunk_input_tokens=int(aggregation["max_chunk_input_tokens"]),
            reduce_output_tokens=int(aggregation["reduce_output_tokens"]),
            audit_input_tokens=1,
            repair_input_tokens=1,
        ).build()
        if (
            canonical_json([asdict(item) for item in reduce_plan.chunks])
            != canonical_json([asdict(item) for item in expected.chunks])
            or canonical_json([asdict(item) for item in reduce_plan.nodes])
            != canonical_json([asdict(item) for item in expected.nodes])
        ):
            raise ReportReduceError(
                "reduce plan differs from the canonical tree compiled from approved aggregation settings"
            )

    def _source_map(
        self,
        chunks: Mapping[str, Any],
        artifacts: Sequence[FrozenDerivedArtifact],
        corpus_snapshot: Mapping[str, Any],
    ) -> Mapping[str, FrozenDerivedArtifact]:
        supplied = {artifact.artifact_hash: artifact for artifact in artifacts}
        if len(supplied) != len(artifacts):
            raise ReportReduceError("source artifacts contain duplicate hashes")
        expected = {artifact_hash for chunk in chunks.values() for artifact_hash in chunk.analysis_hashes}
        if set(supplied) != expected:
            raise ReportReduceError("source artifacts do not exactly match the frozen analysis set")
        corpus = {str(item["paper_id"]): item for item in corpus_snapshot["papers"]}
        analyses = {item.paper_id: item for item in self.analyses}
        if set(corpus) != set(analyses):
            raise ReportReduceError("frozen corpus does not exactly match the Stage 4 analysis set")
        sources: dict[str, FrozenDerivedArtifact] = {}
        for chunk in chunks.values():
            for paper_id, artifact_hash in zip(chunk.paper_ids, chunk.analysis_hashes, strict=True):
                artifact = supplied[artifact_hash]
                analysis = analyses[paper_id]
                frozen_paper = corpus[paper_id]
                if artifact.artifact_kind != "analysis" or artifact.paper_id != paper_id:
                    raise ReportReduceError("section leaf is not bound to its frozen paper analysis")
                if (
                    frozen_paper["analysis_run_id"] != analysis.analysis_run_id
                    or frozen_paper["analysis_artifact_hash"] != analysis.analysis_hash
                    or frozen_paper["analysis_artifact_hash"] != artifact_hash
                    or frozen_paper["input_scope"] != analysis.input_scope
                ):
                    raise ReportReduceError("Stage 4 analysis has drifted from the frozen corpus")
                trusted = self._trusted_stage4_artifact(artifact, analysis, frozen_paper)
                previous = sources.get(artifact_hash)
                if previous is not None and previous != trusted:
                    raise ReportReduceError("one analysis hash resolved to conflicting trusted provenance")
                sources[artifact_hash] = trusted
        return sources

    def _trusted_stage4_artifact(
        self,
        supplied: FrozenDerivedArtifact,
        analysis: AnalysisRecord,
        frozen_paper: Mapping[str, Any],
    ) -> FrozenDerivedArtifact:
        row = self.database.connection.execute(
            """SELECT ar.*, output.artifact_id AS bound_output_id,
                      output.paper_id AS output_paper_id,
                      output.artifact_kind AS output_kind,
                      output.relative_path AS output_relative_path,
                      output.mime_type AS output_mime_type,
                      output.byte_size AS output_byte_size,
                      output.sha256 AS output_sha256,
                      output.provenance_json AS output_provenance_json,
                      output.processing_status AS output_processing_status,
                      stage4.stage AS pipeline_stage,
                      stage4.status AS pipeline_status,
                      stage4.input_hash AS pipeline_input_hash,
                      stage4.config_hash AS pipeline_config_hash,
                      stage4.implementation_version AS pipeline_implementation_version
               FROM analysis_runs ar
               JOIN artifacts output ON output.artifact_id = ar.output_artifact_id
               JOIN pipeline_runs stage4 ON stage4.run_id = ar.run_id
               WHERE ar.analysis_run_id = ?""",
            (analysis.analysis_run_id,),
        ).fetchone()
        if row is None:
            raise ReportReduceError("frozen corpus analysis_run_id has no persisted Stage 4 output")
        expected_row = (
            analysis.paper_id,
            analysis.input_scope,
            PROCESSING_MODEL,
            "complete",
            analysis.analysis_hash,
            analysis.paper_id,
            "analysis",
            "available",
            "stage4",
            "complete",
            frozen_paper["analysis_pipeline_input_hash"],
            frozen_paper["analysis_config_hash"],
            frozen_paper["analysis_implementation_version"],
            frozen_paper["analysis_implementation_version"],
        )
        actual_row = (
            row["paper_id"],
            row["input_scope"],
            row["model_id"],
            row["status"],
            row["output_sha256"],
            row["output_paper_id"],
            row["output_kind"],
            row["output_processing_status"],
            row["pipeline_stage"],
            row["pipeline_status"],
            row["pipeline_input_hash"],
            row["pipeline_config_hash"],
            row["pipeline_implementation_version"],
            row["implementation_version"],
        )
        if actual_row != expected_row:
            raise ReportReduceError("persisted Stage 4 analysis output binding has drifted")
        if analysis.input_tokens != int(frozen_paper["analysis_input_tokens"]):
            raise ReportReduceError("Stage 4 analysis token estimate has drifted from the frozen corpus")
        if (
            row["output_relative_path"] != self.artifact_store.relative_path(analysis.analysis_hash)
            or row["output_mime_type"] != "application/json"
            or int(row["output_byte_size"]) != len(supplied.payload)
        ):
            raise ReportReduceError("persisted Stage 4 analysis artifact metadata has drifted")
        provenance = _mapping_document(row["output_provenance_json"], "analysis artifact provenance")
        if provenance.get("analysis_run_id") != analysis.analysis_run_id or provenance.get("stage") != "stage4":
            raise ReportReduceError("analysis artifact lacks its Stage 4 provenance binding")
        payload = self.artifact_store.read_bytes(analysis.analysis_hash)
        if payload != supplied.payload:
            raise ReportReduceError("supplied analysis bytes differ from the persisted Stage 4 artifact")
        document = _json_document(payload)
        try:
            validate(document, "paper-analysis.schema.json", self.schema_root)
        except SchemaValidationError as error:
            raise ReportReduceError(str(error)) from error
        if (
            document["paper_id"] != analysis.paper_id
            or document["input_scope"] != analysis.input_scope
            or canonical_json(document["evidence_units"]) != canonical_json(analysis.evidence_units)
        ):
            raise ReportReduceError("AnalysisRecord disagrees with its persisted Stage 4 document")
        labels = document["labels"]
        expected_classifications = {
            axis: values
            for axis in (
                "subquestion",
                "theme",
                "method_family",
                "task",
                "dataset",
                "benchmark",
                "evidence_type",
            )
            if (values := _classification_values(labels.get(axis)))
        }
        for axis in ("publication_status", "study_setting"):
            frozen_value = str(frozen_paper[axis])
            if _classification_values(labels.get(axis)) != (frozen_value,):
                raise ReportReduceError("analysis classification labels conflict with the frozen corpus")
            expected_classifications[axis] = (frozen_value,)
        year = (
            str(frozen_paper["publication_year"])
            if frozen_paper["publication_year"] is not None
            else ""
        )
        if not year and frozen_paper["publication_date"]:
            year = str(frozen_paper["publication_date"])[:4]
        if year:
            expected_classifications["time"] = (year,)
        venue = str(
            frozen_paper["venue_id"] or frozen_paper["venue_name"] or ""
        ).strip()
        if venue:
            expected_classifications["venue"] = (venue,)
        actual_classifications: dict[str, tuple[str, ...]] = {}
        for axis, values in analysis.classifications.items():
            raw_values = (values,) if isinstance(values, str) else tuple(
                str(item) for item in values
            )
            normalized = _classification_values(raw_values)
            if tuple(raw_values) != normalized:
                raise ReportReduceError(
                    "AnalysisRecord classifications must be sorted, unique, and normalized"
                )
            if normalized:
                actual_classifications[str(axis)] = normalized
        if actual_classifications != expected_classifications:
            raise ReportReduceError("AnalysisRecord classifications are not the trusted frozen labels")

        detail = _mapping_document(row["invocation_metadata_json"], "Stage 4 invocation metadata")
        input_policy_facts = detail.get("input_policy_facts")
        if not isinstance(input_policy_facts, Mapping):
            raise ReportReduceError("Stage 4 analysis lacks persisted input policy facts")
        if (
            detail.get("report_input_tokens") != frozen_paper["analysis_input_tokens"]
            or detail.get("report_input_tokens") != analysis.input_tokens
            or content_hash(dict(input_policy_facts)) != frozen_paper["analysis_policy_facts_hash"]
            or input_policy_facts.get("paper_id") != analysis.paper_id
            or input_policy_facts.get("artifact_hash") != document["artifact_hash"]
            or input_policy_facts.get("input_scope") != analysis.input_scope
        ):
            raise ReportReduceError("Stage 4 input policy facts have drifted from the frozen corpus")
        invocation_document = detail.get("invocation")
        if not isinstance(invocation_document, Mapping):
            raise ReportReduceError("Stage 4 analysis lacks persisted invocation metadata")
        try:
            invocation = InvocationMetadata(**dict(invocation_document))
        except TypeError as error:
            raise ReportReduceError("persisted Stage 4 invocation metadata is malformed") from error
        expected_invocation = (
            STAGE4_PROFILE,
            PROCESSING_MODEL,
            STAGE4_REASONING_EFFORT,
            STAGE4_SCHEMA,
            self.analysis_schema_hash,
            frozen_paper["analysis_prompt_input_hash"],
            STAGE4_PROMPT,
            self.analysis_prompt_hash,
            None,
            PROCESSING_MODEL,
            STAGE4_PROFILE,
        )
        actual_invocation = (
            invocation.profile,
            invocation.model,
            invocation.reasoning_effort,
            invocation.schema_name,
            invocation.schema_hash,
            invocation.input_hash,
            invocation.prompt_name,
            invocation.prompt_hash,
            invocation.call_kind,
            invocation.actual_model,
            invocation.actual_profile,
        )
        if (
            actual_invocation != expected_invocation
            or invocation.invocation_id != frozen_paper["analysis_invocation_id"]
            or not _is_sha256(invocation.input_hash)
            or invocation.rendered_prompt_hash
            != frozen_paper["analysis_rendered_prompt_hash"]
            or row["input_hash"] != frozen_paper["analysis_prompt_input_hash"]
            or invocation.attempts < 1
            or invocation.attempts > STAGE4_MAX_RETRIES + 1
        ):
            raise ReportReduceError("Stage 4 invocation metadata does not match the frozen Luna profile")
        decision = detail.get("processing_decision")
        if not isinstance(decision, Mapping):
            raise ReportReduceError("Stage 4 analysis lacks a persisted processing decision")
        expected_decision = (
            self.gate.policy.version,
            self.gate.policy.hash,
            PROCESSING_PROVIDER,
            PROCESSING_MODEL,
            "internal_analysis",
            analysis.input_scope,
        )
        actual_decision = (
            decision.get("policy_version"),
            decision.get("policy_hash"),
            decision.get("provider"),
            decision.get("model"),
            decision.get("purpose"),
            decision.get("outcome"),
        )
        if actual_decision != expected_decision:
            raise ReportReduceError("Stage 4 processing decision is missing or foreign")
        if (
            row["policy_version"] != decision.get("policy_version")
            or row["policy_decision"] != decision.get("outcome")
            or row["authorization_grant_id"] != decision.get("processing_grant_id")
        ):
            raise ReportReduceError("Stage 4 policy row disagrees with its persisted decision")
        allowed_categories = {
            "full_pdf": {"full_text", "normalized_text"},
            "abstract_only": {"abstract"},
            "metadata_only": {"metadata"},
        }
        if (
            decision.get("authorized_by") not in {"policy", "grant"}
            or decision.get("data_category") not in allowed_categories[analysis.input_scope]
        ):
            raise ReportReduceError("Stage 4 processing decision was not authorized for its input")
        source_hash = str(decision.get("input_artifact_hash") or "")
        if (
            document["artifact_hash"] != source_hash
            or document["prompt_hash"] != row["prompt_hash"]
            or document["schema_hash"] != row["schema_hash"]
            or document["model"] != PROCESSING_MODEL
            or document["model_revision"] != row["model_revision"]
            or row["prompt_hash"] != self.analysis_prompt_hash
            or row["schema_hash"] != self.analysis_schema_hash
        ):
            raise ReportReduceError("persisted Stage 4 document binding has drifted")
        lineage_hashes = tuple(sorted(set(str(item) for item in frozen_paper["lineage_hashes"])))
        if not lineage_hashes or source_hash not in lineage_hashes or any(not _is_sha256(item) for item in lineage_hashes):
            raise ReportReduceError("frozen corpus lineage does not bind the Stage 4 source artifact")
        license_value, access_basis, domain = self._trusted_source_policy_facts(
            row["artifact_id"], analysis.paper_id, source_hash, input_policy_facts
        )
        return FrozenDerivedArtifact(
            artifact_hash=analysis.analysis_hash,
            payload=payload,
            artifact_kind="analysis",
            input_scope=analysis.input_scope,
            license=license_value,
            access_basis=access_basis,
            lineage_hash=content_hash(lineage_hashes),
            source_lineage_hashes=lineage_hashes,
            source_paper_ids=(analysis.paper_id,),
            paper_id=analysis.paper_id,
            domain=domain,
            mode=self.execution_mode,
        )

    def _trusted_source_policy_facts(
        self,
        artifact_id: str | None,
        paper_id: str,
        source_hash: str,
        input_policy_facts: Mapping[str, Any],
    ) -> tuple[str | None, str, str | None]:
        if not artifact_id:
            access_basis = str(input_policy_facts.get("access_basis") or "unknown")
            if access_basis not in {
                "open_license", "public_read_only", "user_subscription", "user_supplied", "unknown"
            }:
                raise ReportReduceError("Stage 4 input policy access basis is invalid")
            return (
                input_policy_facts.get("license"),
                access_basis,
                input_policy_facts.get("domain"),
            )
        source = self.database.connection.execute(
            """SELECT artifact_id, paper_id, artifact_kind, relative_path, mime_type,
                      byte_size, sha256, provenance_json, processing_status
               FROM artifacts WHERE artifact_id = ?""",
            (artifact_id,),
        ).fetchone()
        if (
            source is None
            or source["paper_id"] != paper_id
            or source["sha256"] != source_hash
            or source["processing_status"] != "available"
        ):
            raise ReportReduceError("Stage 4 source artifact binding has drifted")
        selected_payload = self.artifact_store.read_bytes(source_hash)
        if (
            source["relative_path"] != self.artifact_store.relative_path(source_hash)
            or int(source["byte_size"]) != len(selected_payload)
        ):
            raise ReportReduceError("Stage 4 source artifact metadata has drifted")

        pdf = source
        if source["artifact_kind"] == "text":
            if not str(source["mime_type"]).startswith("text/plain"):
                raise ReportReduceError("normalized Stage 4 input has an invalid MIME type")
            extractions = self.database.connection.execute(
                """SELECT te.source_artifact_id, te.source_sha256,
                          pdf.artifact_id, pdf.paper_id, pdf.artifact_kind,
                          pdf.relative_path, pdf.mime_type, pdf.byte_size,
                          pdf.sha256, pdf.provenance_json, pdf.processing_status
                   FROM text_extractions te
                   JOIN artifacts pdf ON pdf.artifact_id = te.source_artifact_id
                   WHERE te.output_artifact_id = ? AND te.paper_id = ?
                     AND te.status = 'full_text_ready'""",
                (artifact_id, paper_id),
            ).fetchall()
            bindings = {
                tuple(row[key] for key in (
                    "source_artifact_id", "source_sha256", "artifact_id", "paper_id",
                    "artifact_kind", "relative_path", "mime_type", "byte_size", "sha256",
                    "provenance_json", "processing_status",
                ))
                for row in extractions
            }
            if len(bindings) != 1:
                raise ReportReduceError("normalized text lacks one exact full-text extraction binding")
            extraction = extractions[0]
            if extraction["source_artifact_id"] != extraction["artifact_id"]:
                raise ReportReduceError("normalized text source PDF binding has drifted")
            pdf = extraction
        elif source["artifact_kind"] != "pdf":
            access_basis = str(input_policy_facts.get("access_basis") or "unknown")
            return input_policy_facts.get("license"), access_basis, input_policy_facts.get("domain")

        if (
            pdf["paper_id"] != paper_id
            or pdf["artifact_kind"] != "pdf"
            or pdf["mime_type"] != "application/pdf"
            or pdf["processing_status"] != "available"
            or (source["artifact_kind"] == "text" and pdf["source_sha256"] != pdf["sha256"])
        ):
            raise ReportReduceError("Stage 4 source PDF binding has drifted")
        pdf_payload = self.artifact_store.read_bytes(str(pdf["sha256"]))
        if (
            pdf["relative_path"] != self.artifact_store.relative_path(str(pdf["sha256"]))
            or int(pdf["byte_size"]) != len(pdf_payload)
        ):
            raise ReportReduceError("Stage 4 source PDF metadata has drifted")
        provenance = _mapping_document(
            pdf["provenance_json"], "Stage 4 source PDF provenance"
        )
        provenance_candidate_id = provenance.get("candidate_id")
        if not isinstance(provenance_candidate_id, str) or not provenance_candidate_id:
            return None, "unknown", None
        rows = self.database.connection.execute(
            """SELECT dc.candidate_id, dc.license, dc.access_basis, dc.host
               FROM download_attempts da
               JOIN fetch_requests fr
                 ON fr.request_id = da.fetch_request_id
                AND fr.candidate_id = da.candidate_id
                AND fr.status = 'consumed'
               JOIN download_candidates dc ON dc.candidate_id = da.candidate_id
               WHERE da.artifact_id = ? AND da.result_status = 'downloaded'
                 AND dc.paper_id = ?""",
            (pdf["artifact_id"], paper_id),
        ).fetchall()
        facts = {
            (
                str(row["candidate_id"]),
                row["license"],
                str(row["access_basis"]),
                str(row["host"]),
            )
            for row in rows
        }
        if len(facts) != 1:
            return None, "unknown", None
        candidate_id, license_value, access_basis, domain = next(iter(facts))
        if candidate_id != provenance_candidate_id:
            return None, "unknown", None
        return license_value, access_basis, domain

    def _preflight_prompt_budget(
        self,
        report_run_id: str,
        plan: Mapping[str, Any],
        reduce_plan: ReducePlan,
        chunks: Mapping[str, Any],
        sources: Mapping[str, FrozenDerivedArtifact],
        corpus_snapshot: Mapping[str, Any],
        search_audit_pack: Mapping[str, Any],
        corpus_evidence: CorpusEvidenceAllowlist,
    ) -> tuple[dict[str, int], dict[str, int], Any]:
        from .report_audit import (
            ReportAuditBudgetError,
            stage4b_audit_repair_budget_bounds,
        )

        output_limits = self._output_limits(plan, reduce_plan)
        prompt_bounds: dict[str, int] = {}
        node_by_id = {node.node_id: node for node in reduce_plan.nodes}
        for node in reduce_plan.nodes:
            if not node.dependency_ids:
                inputs = tuple(sources[value] for value in chunks[node.node_id].analysis_hashes)
                prompt = canonical_json(
                    self._prompt_payload(
                        report_run_id, plan, node, inputs, corpus_evidence
                    )
                ).decode("utf-8")
                prompt_bounds[node.node_id] = _prompt_token_upper_bound(
                    self._rendered_prompt(node.call_kind, prompt)
                )
                continue
            placeholders = [
                {
                    "artifact_hash": "0" * 64,
                    "artifact_kind": OUTPUT_KIND[node_by_id[dependency].call_kind],
                    "lineage_hash": "0" * 64,
                    "document": {},
                }
                for dependency in node.dependency_ids
            ]
            payload = self._prompt_payload_document(
                report_run_id, plan, node, placeholders, corpus_evidence
            )
            empty_prompt = canonical_json(payload).decode("utf-8")
            fixed = _prompt_token_upper_bound(self._rendered_prompt(node.call_kind, empty_prompt))
            # Canonical output JSON is embedded once in the node payload and the
            # full payload is JSON-escaped once more by CodexExec.  Two bytes per
            # persisted output byte is therefore a conservative upper bound.
            prompt_bounds[node.node_id] = fixed + 2 * sum(
                output_limits[dependency] for dependency in node.dependency_ids
            )

        final = reduce_plan.nodes[-1]
        synthesis_id = final.dependency_ids[0]
        try:
            audit_bounds = stage4b_audit_repair_budget_bounds(
                plan,
                corpus_snapshot,
                search_audit_pack,
                final_output_byte_limit=output_limits[final.node_id],
                synthesis_output_byte_limit=output_limits[synthesis_id],
                rubric_path=self.rubric_path,
            )
        except ReportAuditBudgetError as error:
            raise SolBudgetError(str(error)) from error
        return prompt_bounds, output_limits, audit_bounds

    @staticmethod
    def _output_limits(
        plan: Mapping[str, Any], reduce_plan: ReducePlan
    ) -> dict[str, int]:
        consumers: dict[str, list[ReduceNode]] = {}
        for parent in reduce_plan.nodes:
            for dependency in parent.dependency_ids:
                consumers.setdefault(dependency, []).append(parent)
        final_words = sum(int(section["target_words"]) for section in plan["sections"])
        limits: dict[str, int] = {}
        for node in reduce_plan.nodes:
            if node.call_kind == "final_reduce":
                estimated_tokens = max(1, final_words * 2)
            else:
                contributions = [
                    math.ceil(parent.input_tokens / len(parent.dependency_ids))
                    for parent in consumers.get(node.node_id, ())
                ]
                estimated_tokens = max(contributions or [node.input_tokens])
            limits[node.node_id] = min(
                MAX_NODE_OUTPUT_BYTES,
                max(MIN_NODE_OUTPUT_BYTES, estimated_tokens * OUTPUT_BYTES_PER_ESTIMATED_TOKEN),
            )
        return limits

    def _rendered_prompt(self, call_kind: str, prompt: str) -> str:
        template = self.resources.prompt(call_kind)
        encoded = json.dumps(
            {"authorized_input": prompt}, ensure_ascii=False, separators=(",", ":")
        )
        return f"{template.rstrip()}\n\nThe authorized input follows as JSON data:\n{encoded}\n"

    def _ensure_run(
        self,
        report_run_id: str,
        pipeline_run_id: str,
        plan: Mapping[str, Any],
        reduce_plan: ReducePlan,
        corpus_snapshot: Mapping[str, Any],
        search_audit_pack: Mapping[str, Any],
        prompt_bounds: Mapping[str, int],
        output_limits: Mapping[str, int],
        audit_bounds: Any,
    ) -> None:
        tree = {
            **_tree_document(
                reduce_plan,
                canonical_report_budget(
                    reduce_plan.nodes,
                    prompt_bounds,
                    asdict(audit_bounds),
                    max_retries=int(plan["budget"]["max_retries"]),
                ),
            ),
            "execution_mode": self.execution_mode,
            "audit_repair_budget_bounds": asdict(audit_bounds),
        }
        input_hash = content_hash({
            "plan_hash": plan["plan_hash"],
            "corpus_snapshot_hash": corpus_snapshot["snapshot_hash"],
            "search_audit_pack_hash": search_audit_pack["pack_hash"],
            "tree": tree,
            "prompt_token_bounds": dict(prompt_bounds),
            "output_byte_limits": dict(output_limits),
            "audit_repair_budget_bounds": asdict(audit_bounds),
        })
        prompt_hash = content_hash(self.prompt_hashes)
        schema_hash = content_hash(self.schema_hashes)
        plan_text = _json_text(plan)
        tree_text = _json_text(tree)
        with self.database.transaction() as connection:
            stored_plan = connection.execute(
                """SELECT content_hash, schema_version, plan_json, approval_json, status
                   FROM report_plans WHERE report_plan_id = ?""",
                (plan["plan_id"],),
            ).fetchone()
            if stored_plan is None:
                raise ReportReduceError(
                    "approved ReportPlan must be persisted before Stage 4b reduce"
                )
            expected_plan = (
                plan["plan_hash"],
                plan.get("schema_version", "1"),
                plan_text,
                _json_text(plan["approval"]),
                "approved",
            )
            if tuple(stored_plan) != expected_plan:
                raise ReportReduceError("stored ReportPlan is immutable")

            pipeline = connection.execute(
                "SELECT stage, input_hash, config_hash, implementation_version FROM pipeline_runs WHERE run_id = ?",
                (pipeline_run_id,),
            ).fetchone()
            expected_pipeline = ("stage4b", input_hash, self.config_hash, self.implementation_version)
            if pipeline is None:
                connection.execute(
                    """INSERT INTO pipeline_runs(
                        run_id, stage, status, input_hash, config_hash, implementation_version, started_at
                    ) VALUES (?, 'stage4b', 'running', ?, ?, ?, CURRENT_TIMESTAMP)""",
                    (pipeline_run_id, input_hash, self.config_hash, self.implementation_version),
                )
            elif tuple(pipeline) != expected_pipeline:
                raise ReportReduceError("Stage 4b pipeline run input or configuration has drifted")

            report = connection.execute(
                """SELECT run_id, report_plan_id, corpus_snapshot_hash, aggregation_tree_json,
                          model_id, prompt_hash, schema_hash FROM report_runs WHERE report_run_id = ?""",
                (report_run_id,),
            ).fetchone()
            expected_report = (
                pipeline_run_id, plan["plan_id"], plan["corpus_snapshot_hash"], tree_text,
                SUMMARY_MODEL, prompt_hash, schema_hash,
            )
            if report is None:
                connection.execute(
                    """INSERT INTO report_runs(
                        report_run_id, run_id, report_plan_id, corpus_snapshot_hash,
                        aggregation_tree_json, model_id, model_revision, prompt_hash, schema_hash, status
                    ) VALUES (?, ?, ?, ?, ?, ?, 'codex-cli-managed', ?, ?, 'running')""",
                    (report_run_id, *expected_report),
                )
            elif tuple(report) != expected_report:
                raise ReportReduceError("report run input or aggregation tree has drifted")

            for node in reduce_plan.nodes:
                self._insert_or_check_node(
                    connection,
                    report_run_id,
                    node,
                    prompt_bounds[node.node_id],
                    output_limits[node.node_id],
                )

    def _insert_or_check_node(
        self,
        connection: Any,
        report_run_id: str,
        node: ReduceNode,
        prompt_token_bound: int,
        output_byte_limit: int,
    ) -> None:
        node_key = "report-node-" + content_hash([report_run_id, node.node_id])
        values = (
            report_run_id,
            node.node_id,
            node.call_kind,
            _json_text(node.section_ids),
            _json_text(node.paper_ids),
            _json_text(node.dependency_ids),
            node.planned_input_hash,
            node.input_tokens,
            prompt_token_bound,
            output_byte_limit,
            PROFILE,
            SUMMARY_MODEL,
            REASONING_EFFORT,
            CALL_KIND_PROMPTS[node.call_kind],
            self.prompt_hashes[node.call_kind],
            CALL_KIND_SCHEMAS[node.call_kind],
            self.schema_hashes[node.call_kind],
        )
        row = connection.execute(
            """SELECT report_run_id, node_id, call_kind, section_ids_json, paper_ids_json,
                      dependency_ids_json, planned_input_hash, input_tokens, prompt_token_bound,
                      output_byte_limit, profile, model_id, reasoning_effort, prompt_name,
                      prompt_hash, schema_name, schema_hash
               FROM report_reduce_nodes WHERE report_reduce_node_id = ?""",
            (node_key,),
        ).fetchone()
        if row is None:
            connection.execute(
                """INSERT INTO report_reduce_nodes(
                    report_reduce_node_id, report_run_id, node_id, call_kind, section_ids_json,
                    paper_ids_json, dependency_ids_json, planned_input_hash, input_tokens,
                    prompt_token_bound, output_byte_limit, profile, model_id, reasoning_effort,
                    prompt_name, prompt_hash, schema_name, schema_hash, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')""",
                (node_key, *values),
            )
        elif tuple(row) != values:
            raise ReportReduceError(f"stored reduce node has drifted: {node.node_id}")

    def _execute_node(
        self,
        report_run_id: str,
        plan: Mapping[str, Any],
        node: ReduceNode,
        inputs: Sequence[FrozenDerivedArtifact],
        dependency_outputs: Mapping[str, Mapping[str, Any]],
        processing_grants: Mapping[str, str],
        now: str | None,
        worker_id: str,
        moment: datetime,
        prompt_token_bound: int,
        output_byte_limit: int,
        audit_bounds: Any,
        corpus_evidence: CorpusEvidenceAllowlist,
    ) -> tuple[ReduceNodeResult, Mapping[str, Any] | None, FrozenDerivedArtifact | None]:
        decisions: list[ProcessingDecision] = []
        grant_papers: dict[str, set[str]] = {}
        for artifact in inputs:
            grant_id = processing_grants.get(artifact.artifact_hash)
            if grant_id:
                grant_papers.setdefault(grant_id, set()).update(
                    artifact.source_paper_ids
                    or ((artifact.paper_id,) if artifact.paper_id else ())
                )
        for artifact in inputs:
            grant_id = processing_grants.get(artifact.artifact_hash)
            dispatched = self.gate.dispatch(
                artifact.processing_request(),
                lambda invocation: invocation,
                processing_grant_id=grant_id,
                now=now,
                paper_count=max(1, len(grant_papers.get(grant_id, ()))),
            )
            decisions.append(dispatched.decision)
            if not dispatched.decision.is_authorized:
                self._persist_manual(report_run_id, node.node_id, inputs, decisions)
                return ReduceNodeResult(
                    node.node_id, "manual_required", error=dispatched.decision.reason_code
                ), None, None
            invocation = dispatched.result
            assert isinstance(invocation, ModelInvocation) and invocation.derived_bytes is not None

        payload = self._prompt_payload(
            report_run_id, plan, node, inputs, corpus_evidence
        )
        prompt = canonical_json(payload).decode("utf-8")
        input_hash = sha256(prompt.encode("utf-8")).hexdigest()
        rendered_prompt = self._rendered_prompt(node.call_kind, prompt)
        rendered_prompt_hash = sha256(rendered_prompt.encode("utf-8")).hexdigest()
        actual_input_tokens = _prompt_token_upper_bound(rendered_prompt)
        if actual_input_tokens > prompt_token_bound:
            error = SolBudgetError("rendered Sol prompt exceeded its frozen preflight bound")
            self._persist_unclaimed_failure(report_run_id, node.node_id, error)
            return ReduceNodeResult(node.node_id, "failed", error=str(error)), None, None
        lease_token: int | None = None
        try:
            lease_token = self._reserve_budget_and_claim(
                report_run_id,
                plan,
                node,
                input_hash,
                rendered_prompt_hash,
                actual_input_tokens,
                inputs,
                decisions,
                worker_id,
                moment,
                audit_bounds,
            )
            if lease_token is None:
                return ReduceNodeResult(node.node_id, "running"), None, None
            request = CodexExecRequest(
                profile=PROFILE,
                prompt=prompt,
                output_schema=self.schemas[node.call_kind],
                schema_name=CALL_KIND_SCHEMAS[node.call_kind],
                prompt_name=CALL_KIND_PROMPTS[node.call_kind],
                input_hash=input_hash,
                call_kind=node.call_kind,
                schema_path=self.resources.schema_path(node.call_kind),
                prompt_path=self.resources.prompt_path(node.call_kind),
                expected_prompt_hash=self.prompt_hashes[node.call_kind],
                schema_resource_paths=self.resources.configured_schema_resources(),
                expected_service_schema_hash=self.service_schema_hashes[
                    node.call_kind
                ],
            )
            result = self.invoker_factory().invoke(request)
            self._validate_metadata(
                result.metadata, node, input_hash, rendered_prompt_hash
            )
            if result.metadata.output_hash != content_hash(dict(result.output)):
                raise SolOutputError("Sol invocation output hash does not match its result")
            self._validate_output(
                report_run_id,
                plan,
                node,
                result.output,
                dependency_outputs,
                output_byte_limit,
                corpus_evidence,
            )
            output_payload = canonical_json(result.output)
            stored = self.artifact_store.put_bytes(
                output_payload,
                mime_type="application/json",
                metadata={"kind": "stage4b_reduce_output"},
            )
            output_policy = _output_policy(node, inputs, stored)
            self._persist_complete(
                report_run_id,
                node,
                result.metadata,
                stored,
                output_policy,
                worker_id,
                lease_token,
            )
            return (
                ReduceNodeResult(node.node_id, "complete", stored.artifact_hash),
                dict(result.output),
                output_policy,
            )
        except Exception as error:
            if lease_token is None:
                self._persist_unclaimed_failure(report_run_id, node.node_id, error)
            else:
                self._persist_dispatch_error(
                    report_run_id,
                    node.node_id,
                    error,
                    worker_id,
                    lease_token,
                )
            return ReduceNodeResult(node.node_id, "failed", error=str(error)), None, None

    @staticmethod
    def _prompt_payload(
        report_run_id: str,
        plan: Mapping[str, Any],
        node: ReduceNode,
        inputs: Sequence[FrozenDerivedArtifact],
        corpus_evidence: CorpusEvidenceAllowlist,
    ) -> dict[str, Any]:
        documents = [
            {
                "artifact_hash": artifact.artifact_hash,
                "artifact_kind": artifact.artifact_kind,
                "lineage_hash": artifact.lineage_hash,
                "document": _json_document(artifact.payload),
            }
            for artifact in inputs
        ]
        return SolReduceCoordinator._prompt_payload_document(
            report_run_id, plan, node, documents, corpus_evidence
        )

    @staticmethod
    def _prompt_payload_document(
        report_run_id: str,
        plan: Mapping[str, Any],
        node: ReduceNode,
        documents: Sequence[Mapping[str, Any]],
        corpus_evidence: CorpusEvidenceAllowlist,
    ) -> dict[str, Any]:
        payload = {
            "report_run_id": report_run_id,
            "report_plan": dict(plan),
            "node": {
                "node_id": node.node_id,
                "call_kind": node.call_kind,
                "section_ids": list(node.section_ids),
                "paper_ids": list(node.paper_ids),
                "dependency_ids": list(node.dependency_ids),
                "planned_input_hash": node.planned_input_hash,
            },
            "inputs": [dict(item) for item in documents],
        }
        if node.call_kind == "section_reduce":
            payload["corpus_evidence"] = corpus_evidence.document()
        return payload

    def _validate_metadata(
        self,
        metadata: InvocationMetadata,
        node: ReduceNode,
        input_hash: str,
        rendered_prompt_hash: str,
    ) -> None:
        expected = (
            PROFILE,
            SUMMARY_MODEL,
            REASONING_EFFORT,
            node.call_kind,
            CALL_KIND_SCHEMAS[node.call_kind],
            self.schema_hashes[node.call_kind],
            CALL_KIND_PROMPTS[node.call_kind],
            self.prompt_hashes[node.call_kind],
            input_hash,
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
        )
        if (
            actual != expected
            or not self.resources.accepts_metadata_paths(
                node.call_kind, metadata.schema_path, metadata.prompt_path
            )
            or metadata.actual_model != SUMMARY_MODEL
            or metadata.actual_profile != PROFILE
            or metadata.rendered_prompt_hash != rendered_prompt_hash
            or not isinstance(metadata.invocation_id, str)
            or not metadata.invocation_id.strip()
        ):
            raise SolOutputError("Sol invocation metadata does not match the frozen node binding")
        if metadata.attempts < 1 or metadata.attempts > MAX_RETRIES + 1:
            raise SolOutputError("Sol invocation exceeded the frozen retry budget")

    def _validate_output(
        self,
        report_run_id: str,
        plan: Mapping[str, Any],
        node: ReduceNode,
        output: Mapping[str, Any],
        dependency_outputs: Mapping[str, Mapping[str, Any]],
        output_byte_limit: int,
        corpus_evidence: CorpusEvidenceAllowlist,
    ) -> None:
        if len(canonical_json(output)) > output_byte_limit:
            raise SolOutputError("Sol output exceeded the frozen node byte limit")
        try:
            self.resources.validate(output, node.call_kind)
        except SchemaValidationError as error:
            raise SolOutputError(str(error)) from error
        dependencies = tuple(dependency_outputs[dependency] for dependency in node.dependency_ids)
        if node.call_kind == "section_reduce":
            validator = SynthesisValidator(
                report_run_id=report_run_id,
                analyses=self.analyses,
                sections=self.sections,
                memberships={paper: self.memberships[paper] for paper in node.paper_ids},
                corpus_evidence=corpus_evidence,
            )
            validator.validate_section(output)
            _validate_section_coverage_dispositions(plan, output)
            _require_exact_markers(
                str(output["draft"]),
                set(str(item) for item in output["citation_paper_ids"]),
                "section draft",
            )
            if dependencies:
                _assert_synthesis_preserved(output, dependencies, cross_section=False)
        elif node.call_kind == "cross_section_reduce":
            _assert_synthesis_preserved(output, dependencies, cross_section=True)
        else:
            _validate_report_document(report_run_id, plan, output, dependencies[0])

    def _node_inputs(
        self,
        node: ReduceNode,
        chunks: Mapping[str, Any],
        sources: Mapping[str, FrozenDerivedArtifact],
        outputs: Mapping[str, FrozenDerivedArtifact],
    ) -> tuple[FrozenDerivedArtifact, ...] | None:
        if not node.dependency_ids:
            return tuple(sources[value] for value in chunks[node.node_id].analysis_hashes)
        if any(dependency not in outputs for dependency in node.dependency_ids):
            return None
        return tuple(outputs[dependency] for dependency in node.dependency_ids)

    def _reserve_budget_and_claim(
        self,
        report_run_id: str,
        plan: Mapping[str, Any],
        node: ReduceNode,
        input_hash: str,
        rendered_prompt_hash: str,
        actual_input_tokens: int,
        inputs: Sequence[FrozenDerivedArtifact],
        decisions: Sequence[ProcessingDecision],
        worker_id: str,
        moment: datetime,
        audit_bounds: Any,
    ) -> int | None:
        budget = plan["budget"]
        attempts = int(budget["max_retries"]) + 1
        generation_call_limit = int(budget["max_sol_calls"]) - int(audit_bounds.worst_case_calls)
        generation_token_limit = (
            int(budget["max_input_tokens"]) - int(audit_bounds.worst_case_input_tokens)
        )
        token_reservation = actual_input_tokens * attempts
        lease_expires_at = _timestamp(moment + timedelta(seconds=LEASE_SECONDS))
        with self.database.transaction() as connection:
            current = connection.execute(
                """SELECT status, lease_token FROM report_reduce_nodes
                   WHERE report_run_id = ? AND node_id = ?""",
                (report_run_id, node.node_id),
            ).fetchone()
            if current is None:
                raise ReportReduceError(f"missing persisted reduce node: {node.node_id}")
            if current["status"] not in {"pending", "manual_required"}:
                return None
            used = connection.execute(
                """SELECT COALESCE(SUM(budget_calls_reserved), 0),
                          COALESCE(SUM(budget_tokens_reserved), 0)
                   FROM report_reduce_nodes WHERE report_run_id = ?""",
                (report_run_id,),
            ).fetchone()
            if int(used[0]) + attempts > generation_call_limit:
                raise SolBudgetError("Sol generation call budget is exhausted before dispatch")
            if int(used[1]) + token_reservation > generation_token_limit:
                raise SolBudgetError("Sol generation input-token budget is exhausted before dispatch")
            next_token = int(current["lease_token"]) + 1
            updated = connection.execute(
                """UPDATE report_reduce_nodes SET status = 'running', actual_input_hash = ?,
                       rendered_prompt_hash = ?, actual_input_tokens = ?,
                       input_artifact_hashes_json = ?, processing_decisions_json = ?,
                       processing_grant_ids_json = ?,
                       budget_calls_reserved = budget_calls_reserved + ?,
                       budget_tokens_reserved = budget_tokens_reserved + ?,
                       dispatch_count = dispatch_count + 1, lease_owner = ?,
                       lease_token = ?, lease_expires_at = ?, error_json = NULL,
                       completed_at = NULL, updated_at = CURRENT_TIMESTAMP
                   WHERE report_run_id = ? AND node_id = ? AND status = ? AND lease_token = ?""",
                (
                    input_hash,
                    rendered_prompt_hash,
                    actual_input_tokens,
                    _json_text([item.artifact_hash for item in inputs]),
                    _json_text([_decision_document(item) for item in decisions]),
                    _json_text([item.processing_grant_id for item in decisions]),
                    attempts,
                    token_reservation,
                    worker_id,
                    next_token,
                    lease_expires_at,
                    report_run_id,
                    node.node_id,
                    current["status"],
                    current["lease_token"],
                ),
            )
            if updated.rowcount != 1:
                return None
            return next_token

    def _persist_manual(
        self,
        report_run_id: str,
        node_id: str,
        inputs: Sequence[FrozenDerivedArtifact],
        decisions: Sequence[ProcessingDecision],
    ) -> None:
        with self.database.transaction() as connection:
            connection.execute(
                """UPDATE report_reduce_nodes SET status = 'manual_required',
                       input_artifact_hashes_json = ?, processing_decisions_json = ?,
                       processing_grant_ids_json = ?, error_json = ?, updated_at = CURRENT_TIMESTAMP
                   WHERE report_run_id = ? AND node_id = ?
                     AND status IN ('pending', 'manual_required')""",
                (
                    _json_text([item.artifact_hash for item in inputs]),
                    _json_text([_decision_document(item) for item in decisions]),
                    _json_text([item.processing_grant_id for item in decisions]),
                    _json_text({"error": "processing_not_authorized", "reason": decisions[-1].reason_code}),
                    report_run_id,
                    node_id,
                ),
            )

    def _persist_complete(
        self,
        report_run_id: str,
        node: ReduceNode,
        metadata: InvocationMetadata,
        stored: StoredArtifact,
        output_policy: FrozenDerivedArtifact,
        worker_id: str,
        lease_token: int,
    ) -> None:
        metadata_document = asdict(metadata)
        with self.database.transaction() as connection:
            register_report_invocation(
                connection,
                report_run_id=report_run_id,
                invocation_id=metadata.invocation_id,
                phase="reduce",
                node_key=node.node_id,
                metadata=metadata_document,
            )
            artifact_id = self._save_artifact(connection, report_run_id, node.node_id, stored)
            updated = connection.execute(
                """UPDATE report_reduce_nodes SET status = 'complete', invocation_metadata_json = ?,
                       invocation_id = ?,
                       output_artifact_id = ?, output_hash = ?, output_policy_json = ?, error_json = NULL,
                       lease_owner = NULL, lease_expires_at = NULL,
                       completed_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
                   WHERE report_run_id = ? AND node_id = ? AND status = 'running'
                     AND lease_owner = ? AND lease_token = ?""",
                (
                    _json_text(metadata_document),
                    metadata.invocation_id,
                    artifact_id,
                    stored.artifact_hash,
                    _json_text(_artifact_policy_document(output_policy)),
                    report_run_id,
                    node.node_id,
                    worker_id,
                    lease_token,
                ),
            )
            if updated.rowcount != 1:
                raise ReportReduceError("Sol completion lost its node lease or fencing token")

    def _persist_dispatch_error(
        self,
        report_run_id: str,
        node_id: str,
        error: Exception,
        worker_id: str,
        lease_token: int,
    ) -> None:
        with self.database.transaction() as connection:
            updated = connection.execute(
                """UPDATE report_reduce_nodes SET status = 'failed', error_json = ?,
                       lease_owner = NULL, lease_expires_at = NULL,
                       completed_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
                   WHERE report_run_id = ? AND node_id = ? AND status = 'running'
                     AND lease_owner = ? AND lease_token = ?""",
                (
                    _json_text({"error": type(error).__name__, "message": str(error)}),
                    report_run_id,
                    node_id,
                    worker_id,
                    lease_token,
                ),
            )
            if updated.rowcount != 1:
                raise ReportReduceError("Sol failure lost its node lease or fencing token")

    def _persist_unclaimed_failure(
        self, report_run_id: str, node_id: str, error: Exception
    ) -> None:
        with self.database.transaction() as connection:
            connection.execute(
                """UPDATE report_reduce_nodes SET status = 'failed', error_json = ?,
                       completed_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
                   WHERE report_run_id = ? AND node_id = ?
                     AND status IN ('pending', 'manual_required', 'retryable')""",
                (
                    _json_text({"error": type(error).__name__, "message": str(error)}),
                    report_run_id,
                    node_id,
                ),
            )

    def _load_completed(
        self,
        row: Any,
        node: ReduceNode,
        inputs: Sequence[FrozenDerivedArtifact],
        dependency_outputs: Mapping[str, Mapping[str, Any]],
        plan: Mapping[str, Any],
        output_byte_limit: int,
        now: str | None,
        corpus_evidence: CorpusEvidenceAllowlist,
    ) -> tuple[Mapping[str, Any], FrozenDerivedArtifact]:
        required = (
            "output_artifact_id",
            "output_hash",
            "output_policy_json",
            "actual_input_hash",
            "rendered_prompt_hash",
            "actual_input_tokens",
            "invocation_metadata_json",
            "invocation_id",
        )
        if any(row[key] is None for key in required):
            raise ReportReduceError(f"completed node is missing its output binding: {node.node_id}")
        artifact_row = self.database.connection.execute(
            """SELECT paper_id, artifact_kind, relative_path, mime_type, byte_size, sha256,
                      provenance_json, processing_status
               FROM artifacts WHERE artifact_id = ?""",
            (row["output_artifact_id"],),
        ).fetchone()
        if (
            artifact_row is None
            or artifact_row["paper_id"] is not None
            or artifact_row["sha256"] != row["output_hash"]
            or artifact_row["artifact_kind"] != "report"
            or artifact_row["processing_status"] != "available"
        ):
            raise ReportReduceError(f"completed node output artifact has drifted: {node.node_id}")
        artifact_provenance = _mapping_document(
            artifact_row["provenance_json"], "report reduce artifact provenance"
        )
        if artifact_provenance != {
            "stage": "stage4b",
            "content_hash": row["output_hash"],
        }:
            raise ReportReduceError(f"completed node artifact provenance has drifted: {node.node_id}")
        payload = self.artifact_store.read_bytes(row["output_hash"])
        if (
            artifact_row["relative_path"] != self.artifact_store.relative_path(row["output_hash"])
            or artifact_row["mime_type"] != "application/json"
            or int(artifact_row["byte_size"]) != len(payload)
        ):
            raise ReportReduceError(f"completed node artifact metadata has drifted: {node.node_id}")
        output = _json_document(payload)
        prompt = canonical_json(
            self._prompt_payload(
                str(row["report_run_id"]), plan, node, inputs, corpus_evidence
            )
        ).decode("utf-8")
        input_hash = sha256(prompt.encode("utf-8")).hexdigest()
        rendered = self._rendered_prompt(node.call_kind, prompt)
        rendered_hash = sha256(rendered.encode("utf-8")).hexdigest()
        actual_tokens = _prompt_token_upper_bound(rendered)
        if (
            row["actual_input_hash"] != input_hash
            or row["rendered_prompt_hash"] != rendered_hash
            or int(row["actual_input_tokens"]) != actual_tokens
            or actual_tokens > int(row["prompt_token_bound"])
            or json.loads(row["input_artifact_hashes_json"])
            != [item.artifact_hash for item in inputs]
        ):
            raise ReportReduceError(f"completed node input binding has drifted: {node.node_id}")
        metadata_document = _mapping_document(
            row["invocation_metadata_json"], "Sol invocation metadata"
        )
        try:
            metadata = InvocationMetadata(**metadata_document)
        except TypeError as error:
            raise ReportReduceError("persisted Sol invocation metadata is malformed") from error
        self._validate_metadata(metadata, node, input_hash, rendered_hash)
        if metadata.output_hash is not None and metadata.output_hash != row["output_hash"]:
            raise ReportReduceError(
                f"completed node invocation output has drifted: {node.node_id}"
            )
        if row["invocation_id"] != metadata.invocation_id:
            raise ReportReduceError(
                f"completed node invocation identity has drifted: {node.node_id}"
            )
        try:
            require_report_invocation(
                self.database.connection,
                report_run_id=str(row["report_run_id"]),
                invocation_id=metadata.invocation_id,
                phase="reduce",
                node_key=node.node_id,
                metadata=metadata_document,
            )
        except ReportInvocationError as error:
            raise ReportReduceError(str(error)) from error
        if (
            int(row["dispatch_count"]) < 1
            or int(row["budget_calls_reserved"]) < metadata.attempts
            or int(row["budget_tokens_reserved"]) < actual_tokens * metadata.attempts
        ):
            raise ReportReduceError(f"completed node budget ledger has drifted: {node.node_id}")
        self._validate_persisted_decisions(row, inputs, now)
        self._validate_output(
            str(row["report_run_id"]),
            plan,
            node,
            output,
            dependency_outputs,
            output_byte_limit,
            corpus_evidence,
        )
        stored = StoredArtifact(
            artifact_hash=str(row["output_hash"]),
            mime_type=str(artifact_row["mime_type"]),
            size_bytes=int(artifact_row["byte_size"]),
            relative_path=str(artifact_row["relative_path"]),
            path=self.artifact_store.path_for(str(row["output_hash"])),
        )
        recomputed_policy = _output_policy(node, inputs, stored)
        persisted_policy = _mapping_document(row["output_policy_json"], "reduce output policy")
        if persisted_policy != _artifact_policy_document(recomputed_policy):
            raise ReportReduceError(f"completed node output policy has drifted: {node.node_id}")
        return output, recomputed_policy

    def _validate_persisted_decisions(
        self,
        row: Any,
        inputs: Sequence[FrozenDerivedArtifact],
        now: str | None,
    ) -> None:
        decisions = json.loads(row["processing_decisions_json"])
        grant_ids = json.loads(row["processing_grant_ids_json"])
        if not isinstance(decisions, list) or not isinstance(grant_ids, list):
            raise ReportReduceError("persisted Sol processing decisions are malformed")
        if len(decisions) != len(inputs) or len(grant_ids) != len(inputs):
            raise ReportReduceError("persisted Sol processing decisions are incomplete")
        grant_papers: dict[str, set[str]] = {}
        for artifact, grant_id in zip(inputs, grant_ids, strict=True):
            if grant_id:
                grant_papers.setdefault(str(grant_id), set()).update(
                    artifact.source_paper_ids
                    or ((artifact.paper_id,) if artifact.paper_id else ())
                )
        for artifact, decision, grant_id in zip(inputs, decisions, grant_ids, strict=True):
            if not isinstance(decision, Mapping):
                raise ReportReduceError("persisted Sol processing decision is malformed")
            fresh = self.gate.decide(
                artifact.processing_request(),
                processing_grant_id=grant_id,
                now=now,
                paper_count=max(1, len(grant_papers.get(str(grant_id), ()))),
            )
            if (
                not fresh.is_authorized
                or decision != _decision_document(fresh)
                or decision.get("audit_hash") != fresh.audit_hash
            ):
                raise ReportReduceError("persisted Sol processing decision has drifted")

    @staticmethod
    def _save_artifact(connection: Any, report_run_id: str, node_id: str, stored: StoredArtifact) -> str:
        row = connection.execute(
            """SELECT artifact_id, paper_id, artifact_kind, relative_path, mime_type, byte_size,
                      provenance_json, processing_status
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
                _mapping_document(row["provenance_json"], "existing report artifact provenance"),
                row["processing_status"],
            )
            if actual != expected:
                raise ReportReduceError("report artifact metadata conflicts with existing content")
            return str(row["artifact_id"])
        artifact_id = "artifact-" + stored.artifact_hash
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
                _json_text({"stage": "stage4b", "content_hash": stored.artifact_hash}),
            ),
        )
        return artifact_id

    def _node_row(self, report_run_id: str, node_id: str) -> Any:
        row = self.database.connection.execute(
            "SELECT * FROM report_reduce_nodes WHERE report_run_id = ? AND node_id = ?",
            (report_run_id, node_id),
        ).fetchone()
        if row is None:
            raise ReportReduceError(f"missing persisted reduce node: {node_id}")
        return row

    def _assert_node_binding(
        self,
        row: Any,
        node: ReduceNode,
        prompt_token_bound: int,
        output_byte_limit: int,
    ) -> None:
        expected = (
            node.call_kind,
            _json_text(node.section_ids),
            _json_text(node.paper_ids),
            _json_text(node.dependency_ids),
            node.planned_input_hash,
            node.input_tokens,
            prompt_token_bound,
            output_byte_limit,
            PROFILE,
            SUMMARY_MODEL,
            REASONING_EFFORT,
            CALL_KIND_PROMPTS[node.call_kind],
            self.prompt_hashes[node.call_kind],
            CALL_KIND_SCHEMAS[node.call_kind],
            self.schema_hashes[node.call_kind],
        )
        actual = tuple(row[key] for key in (
            "call_kind", "section_ids_json", "paper_ids_json", "dependency_ids_json",
            "planned_input_hash", "input_tokens", "prompt_token_bound", "output_byte_limit",
            "profile", "model_id", "reasoning_effort",
            "prompt_name", "prompt_hash", "schema_name", "schema_hash",
        ))
        if actual != expected:
            raise ReportReduceError(f"persisted reduce node binding has drifted: {node.node_id}")

    def _recover_stale_nodes(self, report_run_id: str, moment: datetime) -> None:
        with self.database.transaction() as connection:
            connection.execute(
                """UPDATE report_reduce_nodes SET status = 'failed', error_json = ?,
                       lease_owner = NULL, lease_expires_at = NULL,
                       completed_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
                   WHERE report_run_id = ? AND status = 'running'
                     AND lease_expires_at IS NOT NULL AND lease_expires_at <= ?""",
                (
                    _json_text({"error": "InterruptedDispatch", "message": "prior Sol dispatch ended without a result"}),
                    report_run_id,
                    _timestamp(moment),
                ),
            )

    def _set_run_status(self, report_run_id: str, pipeline_run_id: str, status: str) -> None:
        with self.database.transaction() as connection:
            connection.execute(
                "UPDATE report_runs SET status = ?, completed_at = CASE WHEN ? IN ('failed', 'incomplete') THEN CURRENT_TIMESTAMP ELSE NULL END WHERE report_run_id = ? AND status <> 'complete'",
                (status, status, report_run_id),
            )
            connection.execute(
                "UPDATE pipeline_runs SET status = ?, completed_at = CASE WHEN ? IN ('failed', 'incomplete') THEN CURRENT_TIMESTAMP ELSE NULL END WHERE run_id = ? AND status <> 'complete'",
                (status, status, pipeline_run_id),
            )


def _assert_synthesis_preserved(
    output: Mapping[str, Any], inputs: Sequence[Mapping[str, Any]], *, cross_section: bool
) -> None:
    expected_claims: dict[str, Mapping[str, Any]] = {}
    expected_citations: set[str] = set()
    expected_conflicts: set[str] = set()
    expected_sections: set[str] = set()
    for document in inputs:
        sections = document["section_ids"] if "section_ids" in document else (document["section_id"],)
        expected_sections.update(str(item) for item in sections)
        expected_citations.update(str(item) for item in document["citation_paper_ids"])
        expected_conflicts.update(str(item) for item in document["unresolved_conflicts"])
        for claim in document["claims"]:
            claim_id = str(claim["claim_id"])
            previous = expected_claims.get(claim_id)
            if previous is not None and canonical_json(previous) != canonical_json(claim):
                raise SolOutputError(f"dependency claims disagree for {claim_id}")
            expected_claims[claim_id] = claim
    actual_claims = {str(claim["claim_id"]): claim for claim in output["claims"]}
    if len(actual_claims) != len(output["claims"]) or set(actual_claims) != set(expected_claims) or any(
        canonical_json(actual_claims[key]) != canonical_json(value)
        for key, value in expected_claims.items()
    ):
        raise SolOutputError("reduce output added, dropped, or changed a validated claim")
    if set(output["citation_paper_ids"]) != expected_citations:
        raise SolOutputError("reduce output added or dropped a validated citation")
    actual_conflicts = tuple(str(item) for item in output["unresolved_conflicts"])
    if len(set(actual_conflicts)) != len(actual_conflicts) or set(actual_conflicts) != expected_conflicts:
        raise SolOutputError("reduce output added, duplicated, or erased an unresolved conflict")
    if cross_section:
        if tuple(output["section_ids"]) != tuple(sorted(expected_sections)):
            raise SolOutputError("cross-section output changed the stable section set")
    elif output["section_id"] != next(iter(expected_sections)):
        raise SolOutputError("section reduction changed section identity")
    _require_exact_markers(str(output["draft"]), expected_citations, "reduce draft")


def _require_exact_markers(text: str, expected: set[str], label: str) -> None:
    markers = tuple(PAPER_MARKER.findall(text))
    if len(markers) != len(set(markers)) or set(markers) != expected:
        raise SolOutputError(f"{label} markers do not exactly match its citation binding")


def _validate_section_coverage_dispositions(
    plan: Mapping[str, Any], output: Mapping[str, Any]
) -> None:
    allowed = {
        str(item["paper_id"])
        for item in plan["paper_memberships"]
        if item["coverage_disposition"] == "evidence"
    }
    referenced = {
        str(reference["paper_id"])
        for claim in output["claims"]
        for field in ("supporting_evidence", "contradicting_evidence")
        for reference in claim[field]
        if reference["kind"] == "paper_evidence"
    }
    if not referenced.issubset(allowed):
        raise SolOutputError(
            "section output used a resource/background-only paper as claim evidence"
        )


def _validate_report_document(
    report_run_id: str,
    plan: Mapping[str, Any],
    output: Mapping[str, Any],
    synthesis: Mapping[str, Any],
) -> None:
    if output["report_run_id"] != report_run_id:
        raise SolOutputError("ReportDocument belongs to another report run")
    claims = {str(claim["claim_id"]): claim for claim in synthesis["claims"]}
    if not claims or not output["blocks"]:
        raise SolOutputError("ReportDocument requires validated claims and substantive blocks")
    evidence_papers = {
        str(ref["paper_id"])
        for claim in claims.values()
        for ref in tuple(claim["supporting_evidence"]) + tuple(claim["contradicting_evidence"])
        if ref["kind"] == "paper_evidence"
    }
    approved_evidence_papers = {
        str(item["paper_id"])
        for item in plan["paper_memberships"]
        if item["coverage_disposition"] == "evidence"
    }
    if evidence_papers != approved_evidence_papers:
        raise SolOutputError(
            "ReportDocument evidence does not match the frozen evidence dispositions"
        )
    section_ids = {str(section["id"]) for section in plan["sections"]}
    block_ids: set[str] = set()
    used_sections: set[str] = set()
    used_claims: set[str] = set()
    rendered_text: list[str] = []
    for block in output["blocks"]:
        if block["block_id"] in block_ids or block["section_id"] not in section_ids:
            raise SolOutputError("ReportDocument has a duplicate block or unknown section")
        block_ids.add(str(block["block_id"]))
        used_sections.add(str(block["section_id"]))
        rendered_text.append(str(block["text"]))
        claim_ids = set(str(item) for item in block["claim_ids"])
        if not claim_ids.issubset(claims):
            raise SolOutputError("ReportDocument block introduced an unknown claim")
        if not claim_ids and not is_local_references_block(block):
            raise SolOutputError(
                "claim-free ReportDocument block is not the deterministic references note"
            )
        if any(
            str(claims[claim_id]["report_section"]) != str(block["section_id"])
            for claim_id in claim_ids
        ):
            raise SolOutputError(
                "ReportDocument block attached a claim to a different frozen section"
            )
        expected_papers = {
            str(ref["paper_id"])
            for claim_id in claim_ids
            for ref in tuple(claims[claim_id]["supporting_evidence"])
            + tuple(claims[claim_id]["contradicting_evidence"])
            if ref["kind"] == "paper_evidence"
        }
        if set(block["citation_paper_ids"]) != expected_papers:
            raise SolOutputError("ReportDocument block citations do not match its claims")
        _require_exact_markers(
            str(block["text"]), expected_papers, "ReportDocument block text"
        )
        used_claims.update(claim_ids)
    if used_claims != set(claims):
        raise SolOutputError("ReportDocument omitted validated claims")
    if used_sections != section_ids:
        raise SolOutputError("ReportDocument does not cover the exact frozen section set")
    combined_text = "\n".join(rendered_text)
    missing_conflicts = [
        str(item) for item in synthesis["unresolved_conflicts"]
        if str(item).strip() and str(item) not in combined_text
    ]
    if missing_conflicts:
        raise SolOutputError("ReportDocument erased an unresolved conflict disclosure")


def _output_policy(
    node: ReduceNode,
    inputs: Sequence[FrozenDerivedArtifact],
    stored: StoredArtifact,
) -> FrozenDerivedArtifact:
    scope_rank = {"metadata_only": 0, "abstract_only": 1, "full_pdf": 2}
    input_scope = max((item.input_scope for item in inputs), key=scope_rank.__getitem__)
    licenses = {item.license for item in inputs}
    license_value = next(iter(licenses)) if len(licenses) == 1 else None
    access_values = {item.access_basis for item in inputs}
    if access_values == {"open_license"}:
        access_basis = "open_license"
    elif access_values.issubset({"open_license", "public_read_only"}):
        access_basis = "public_read_only"
    elif "user_subscription" in access_values:
        access_basis = "user_subscription"
    elif "user_supplied" in access_values:
        access_basis = "user_supplied"
    else:
        access_basis = "unknown"
    source_lineages = tuple(sorted({
        lineage
        for item in inputs
        for lineage in (item.source_lineage_hashes or (item.lineage_hash,))
    }))
    source_paper_ids = tuple(sorted({
        paper_id
        for item in inputs
        for paper_id in (item.source_paper_ids or ((item.paper_id,) if item.paper_id else ()))
    }))
    paper_ids = {item.paper_id for item in inputs}
    paper_id = next(iter(paper_ids)) if len(paper_ids) == 1 else None
    modes = {item.mode for item in inputs}
    if len(modes) != 1:
        raise ReportReduceError("reduce inputs have conflicting execution modes")
    domains = {item.domain for item in inputs}
    collections = {item.collection_id for item in inputs}
    collection_snapshots = {item.collection_snapshot_hash for item in inputs}
    selection_snapshots = {item.selection_snapshot_hash for item in inputs}
    return FrozenDerivedArtifact(
        artifact_hash=stored.artifact_hash,
        payload=stored.path.read_bytes(),
        artifact_kind=OUTPUT_KIND[node.call_kind],
        input_scope=input_scope,
        license=license_value,
        access_basis=access_basis,
        lineage_hash=content_hash(source_lineages),
        source_lineage_hashes=source_lineages,
        source_paper_ids=source_paper_ids,
        paper_id=paper_id,
        domain=next(iter(domains)) if len(domains) == 1 else None,
        mode=next(iter(modes)),
        collection_id=next(iter(collections)) if len(collections) == 1 else None,
        collection_snapshot_hash=(
            next(iter(collection_snapshots)) if len(collection_snapshots) == 1 else None
        ),
        selection_snapshot_hash=(
            next(iter(selection_snapshots)) if len(selection_snapshots) == 1 else None
        ),
    )


def _tree_document(
    plan: ReducePlan, budget: Mapping[str, int] | None = None
) -> dict[str, Any]:
    return {
        "chunks": [asdict(chunk) for chunk in plan.chunks],
        "nodes": [asdict(node) for node in plan.nodes],
        "budget": dict(budget) if budget is not None else asdict(plan.budget),
    }


def _artifact_policy_document(artifact: FrozenDerivedArtifact) -> dict[str, Any]:
    return {
        "artifact_kind": artifact.artifact_kind,
        "input_scope": artifact.input_scope,
        "license": artifact.license,
        "access_basis": artifact.access_basis,
        "lineage_hash": artifact.lineage_hash,
        "source_lineage_hashes": list(artifact.source_lineage_hashes),
        "source_paper_ids": list(artifact.source_paper_ids),
        "paper_id": artifact.paper_id,
        "domain": artifact.domain,
        "mode": artifact.mode,
        "collection_id": artifact.collection_id,
        "collection_snapshot_hash": artifact.collection_snapshot_hash,
        "selection_snapshot_hash": artifact.selection_snapshot_hash,
    }


def _decision_document(decision: ProcessingDecision) -> dict[str, Any]:
    value = asdict(decision)
    value["outcome"] = decision.outcome.value
    value["audit_hash"] = decision.audit_hash
    return value


def _json_document(payload: bytes) -> Mapping[str, Any]:
    value = json.loads(payload)
    if not isinstance(value, Mapping):
        raise SolOutputError("Stage 4b derived inputs and outputs must be JSON objects")
    return value


def _mapping_document(value: str | bytes, label: str) -> Mapping[str, Any]:
    try:
        document = json.loads(value)
    except (TypeError, json.JSONDecodeError) as error:
        raise ReportReduceError(f"{label} is not valid JSON") from error
    if not isinstance(document, Mapping):
        raise ReportReduceError(f"{label} must be a JSON object")
    return document


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _classification_values(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    raw = (value,) if isinstance(value, str) else tuple(value)  # type: ignore[arg-type]
    return tuple(sorted({str(item).strip() for item in raw if str(item).strip()}))


def _prompt_token_upper_bound(rendered_prompt: str) -> int:
    # A BPE token cannot outnumber the bytes supplied to the tokenizer.  Using
    # bytes as "tokens" deliberately over-reserves multilingual prompts while
    # remaining dependency-free and deterministic across machines.
    return max(1, len(rendered_prompt.encode("utf-8")))


def _timestamp(value: datetime) -> str:
    moment = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_hash(value: Mapping[str, Any]) -> str:
    return sha256(_json_text(value).encode("utf-8")).hexdigest()


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _error_message(value: str | None) -> str | None:
    if not value:
        return None
    document = json.loads(value)
    return str(document.get("message") or document.get("reason") or document.get("error"))
