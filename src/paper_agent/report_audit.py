"""Persistent Stage 4b deterministic verification, Sol audit, repair, and publish gate."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import sysconfig
from typing import Any, Protocol
from uuid import uuid4

import yaml

from .approval import ApprovalError, require_valid_approval
from .artifacts import ArtifactStore
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
from .report_budget import CanonicalReportBudgetError, canonical_report_budget
from .report_config import ReportResources
from .report_invocations import (
    ReportInvocationError,
    register_report_invocation,
    report_invocation_metadata_hash,
    require_report_invocation,
)
from .report_artifacts import (
    ReportArtifactError,
    ReportArtifactStore,
    ReportVerificationError,
    audit_coverage_ledger,
    audit_rubric_hash,
    audit_search_limitations,
    render_markdown,
    report_artifact_hash,
    report_diff,
    search_publication_blockers,
    validate_claim_relations,
    verify_report,
)
from .report_plan import (
    ReportPlanDriftError,
    ReportPlanError,
    assert_report_runtime_matches,
    compile_report_plan,
)
from .reporting import (
    AnalysisRecord,
    EvidenceValidationError,
    ReportPlanner,
    ReportPlanningError,
    require_exact_comparison_groups,
)
from .schema import SchemaValidationError, schema_directory, validate
from .storage import Database


PROFILE = "stage4b_summary_sol"
MODEL = SUMMARY_MODEL
REASONING_EFFORT = "high"
PURPOSE = "research_synthesis"
MAX_RETRIES = FROZEN_PROFILES[PROFILE].max_retries
STAGE4_PROFILE = "stage4_analysis_luna"
STAGE4_REASONING_EFFORT = "medium"
STAGE4_SCHEMA = "paper-analysis.schema.json"
STAGE4_PROMPT = "paper-analysis.md"
STAGE4_MAX_RETRIES = FROZEN_PROFILES[STAGE4_PROFILE].max_retries
LEASE_SECONDS = FROZEN_PROFILES[PROFILE].timeout_seconds * (MAX_RETRIES + 1) + 60
MAX_OUTPUT_BYTES = 262_144
AUDIT_HARD_CONTEXT_TOKENS = 1_179_648
AUDIT_OUTPUT_TOKEN_RESERVE = 131_072
AUDIT_MAX_INPUT_TOKENS = AUDIT_HARD_CONTEXT_TOKENS - AUDIT_OUTPUT_TOKEN_RESERVE
AUDIT_OUTPUT_BYTE_LIMIT = 65_536
REPAIR_OUTPUT_TOKEN_RESERVE = MAX_OUTPUT_BYTES
REPAIR_MAX_INPUT_TOKENS = AUDIT_HARD_CONTEXT_TOKENS - REPAIR_OUTPUT_TOKEN_RESERVE
IMPLEMENTATION_VERSION = "stage4b-audit-gate-v3"
SEVERE = frozenset({"blocker", "major"})
_BUDGET_FIXED_BYTES = 16_384
_BUDGET_PROMPT_WRAPPER_BYTES = 4_096


def stage4b_audit_config_hash(
    processing_policy_hash: str,
    *,
    execution_mode: str = "attended",
    schema_root: Path | None = None,
    prompt_root: Path | None = None,
    resources: ReportResources | None = None,
    rubric_path: Path | None = None,
    implementation_version: str = IMPLEMENTATION_VERSION,
) -> str:
    """Return the frozen audit-gate configuration bound into ReportPlan."""
    if execution_mode not in {"attended", "unattended"}:
        raise ValueError("execution_mode must be attended or unattended")
    schemas = schema_directory(schema_root)
    prompts = prompt_directory() if prompt_root is None else prompt_root
    report_resources = resources or ReportResources.defaults(
        schema_root=schema_root, prompt_root=prompt_root
    )
    report_resources.validate_files()
    rubric_file = rubric_path or _default_rubric_path()
    schema_hashes = {
        kind: _json_hash(report_resources.schema(kind))
        for kind in ("quality_audit", "repair")
    }
    service_schema_hashes = {
        kind: report_resources.service_schema_hash(kind)
        for kind in ("quality_audit", "repair")
    }
    prompt_hashes = {
        kind: sha256(report_resources.prompt_paths[kind].read_bytes()).hexdigest()
        for kind in ("quality_audit", "repair")
    }
    stage4_schema_hash = _json_hash(json.loads(
        (schemas / STAGE4_SCHEMA).read_text(encoding="utf-8")
    ))
    stage4_prompt_hash = sha256((prompts / STAGE4_PROMPT).read_bytes()).hexdigest()
    rubric_hash = content_hash(_load_mapping(rubric_file, "report audit rubric"))
    return content_hash({
        "implementation_version": implementation_version,
        "profile": PROFILE,
        "model": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "max_retries": MAX_RETRIES,
        "processing_policy_hash": processing_policy_hash,
        "rubric_hash": rubric_hash,
        "schema_hashes": schema_hashes,
        "service_schema_hashes": service_schema_hashes,
        "prompt_hashes": prompt_hashes,
        "stage4_schema_hash": stage4_schema_hash,
        "stage4_prompt_hash": stage4_prompt_hash,
        "lease_seconds": LEASE_SECONDS,
        "audit_hard_context_tokens": AUDIT_HARD_CONTEXT_TOKENS,
        "audit_output_token_reserve": AUDIT_OUTPUT_TOKEN_RESERVE,
        "audit_max_input_tokens": AUDIT_MAX_INPUT_TOKENS,
        "audit_output_byte_limit": AUDIT_OUTPUT_BYTE_LIMIT,
        "repair_output_token_reserve": REPAIR_OUTPUT_TOKEN_RESERVE,
        "repair_max_input_tokens": REPAIR_MAX_INPUT_TOKENS,
        "repair_output_byte_limit": MAX_OUTPUT_BYTES,
        "prompt_token_estimator": "utf8-byte-upper-bound-v1",
        "budget_fixed_bytes": _BUDGET_FIXED_BYTES,
        "budget_prompt_wrapper_bytes": _BUDGET_PROMPT_WRAPPER_BYTES,
        "execution_mode": execution_mode,
    })


class ReportAuditError(RuntimeError):
    """The audit gate could not safely continue."""


class ReportAuditOutputError(ReportAuditError):
    """A Sol result failed a frozen output or semantic contract."""


class ReportAuditBudgetError(ReportAuditError):
    """A paid call would exceed the approved worst-case budget."""


class SolInvoker(Protocol):
    def invoke(self, request: CodexExecRequest) -> CodexExecResult: ...


@dataclass(frozen=True, slots=True)
class ReportBundle:
    plan: Mapping[str, Any]
    search_audit: Mapping[str, Any]
    corpus_snapshot: Mapping[str, Any]
    claims: Sequence[Mapping[str, Any]]
    comparison_groups: Mapping[str, Mapping[str, Any]]
    claim_relations: Sequence[Mapping[str, Any]]
    document: Mapping[str, Any]
    coverage: Mapping[str, Any]
    bibliography: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class ReportAuditResult:
    report_run_id: str
    status: str
    report_document_hash: str
    audit_passes: tuple[str, ...]
    repair_count: int
    published_path: Path | None = None
    resumed_steps: tuple[str, ...] = ()
    error: str | None = None


@dataclass(frozen=True, slots=True)
class AuditRepairBudgetBounds:
    audit_a_input_tokens: int
    repair_input_tokens: int
    audit_c_input_tokens: int
    audit_shards_per_pass: int
    audit_reduce_calls_per_pass: int
    audit_calls_per_pass: int
    audit_a_calls: int
    audit_c_calls: int
    hard_context_tokens: int
    max_audit_input_tokens: int
    max_repair_input_tokens: int
    worst_case_calls: int
    worst_case_input_tokens: int


def stage4b_audit_repair_budget_bounds(
    approved_plan: Mapping[str, Any],
    frozen_corpus_snapshot: Mapping[str, Any],
    frozen_search_audit_pack: Mapping[str, Any],
    *,
    final_output_byte_limit: int,
    synthesis_output_byte_limit: int,
    rubric_path: Path | None = None,
) -> AuditRepairBudgetBounds:
    """Pure conservative bound shared by reduce preflight and the audit gate.

    The two output limits are the persisted limits for ``final_reduce`` and
    its single synthesis dependency. JSON is embedded once in the audit step
    and escaped once by ``CodexExec``; the factor of two follows the same
    byte-upper-bound convention as the reduce coordinator. If that bound does
    not fit the frozen hard context, this function reserves a complete stable
    shard pass and a binary, non-sampling audit-reduce tree for both A and C.
    Reducer preflight can call this function without importing coordinator
    state or trusting caller token estimates.
    """
    if final_output_byte_limit < 1 or synthesis_output_byte_limit < 1:
        raise ValueError("Stage 4b audit budget requires positive output byte limits")
    frozen_context = len(canonical_json({
        "report_plan": approved_plan,
        "corpus_snapshot": frozen_corpus_snapshot,
        "search_audit_pack": frozen_search_audit_pack,
    }))
    shared_context = len(canonical_json({
        "report_plan": approved_plan,
        "search_audit_pack": frozen_search_audit_pack,
    }))
    frozen_bibliography = {
        str(paper["paper_id"]): {
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
        for paper in frozen_corpus_snapshot["papers"]
    }
    paper_count = len(frozen_corpus_snapshot["papers"])
    membership_links = sum(
        len(item.get("section_ids", ()))
        for item in approved_plan.get("paper_memberships", ())
    )
    longest_section = max(
        (
            len(str(item["id"]).encode("utf-8"))
            for item in approved_plan.get("sections", ())
        ),
        default=1,
    )
    # Coverage repeats evidence claim IDs and records every stable consuming
    # node.  The synthesis limit covers the former; this combinatorial term
    # safely covers even a maximally split section tree for the latter.
    coverage_upper_bound = (
        synthesis_output_byte_limit
        + len(canonical_json(frozen_corpus_snapshot))
        + membership_links * max(1, paper_count) * (longest_section + 160)
        + _BUDGET_FIXED_BYTES
    )
    rubric_bytes = len(canonical_json(
        _load_mapping(rubric_path or _default_rubric_path(), "report audit rubric")
    ))
    variable_source = (
        final_output_byte_limit
        # Persisted claims, their deterministic comparison-group projection,
        # and bounded claim relations are each no larger than the synthesis
        # output limit before prompt JSON escaping.
        + 3 * synthesis_output_byte_limit
        + len(canonical_json(frozen_corpus_snapshot))
        + len(canonical_json(frozen_bibliography))
        + coverage_upper_bound
    )
    repeated = _BUDGET_PROMPT_WRAPPER_BYTES + 2 * (
        shared_context + rubric_bytes + _BUDGET_FIXED_BYTES
    )
    if repeated >= AUDIT_MAX_INPUT_TOKENS:
        raise ReportAuditBudgetError(
            "frozen audit rubric leaves no room for one stable audit component"
        )
    source_capacity = max(1, (AUDIT_MAX_INPUT_TOKENS - repeated) // 2)

    def audit_shape(variable_bytes: int) -> tuple[int, int, int, int]:
        direct = _BUDGET_PROMPT_WRAPPER_BYTES + 2 * (
            frozen_context + rubric_bytes + variable_bytes + _BUDGET_FIXED_BYTES
        )
        if direct <= AUDIT_MAX_INPUT_TOKENS:
            return 1, 0, 1, direct
        # Stable sequential packing is a next-fit partition.  For components
        # no larger than one shard, next-fit uses fewer than twice the optimal
        # bin count; the extra shard covers integer and wrapper boundaries.
        shards = max(1, 2 * math.ceil(variable_bytes / source_capacity) + 1)
        reduce_calls = max(1, shards - 1)
        calls = shards + reduce_calls
        return shards, reduce_calls, calls, calls * AUDIT_MAX_INPUT_TOKENS

    a_shards, a_reduce_calls, a_calls, audit_a = audit_shape(variable_source)
    c_shards, c_reduce_calls, c_calls, audit_c = audit_shape(
        variable_source + MAX_OUTPUT_BYTES
    )
    repair_payload = (
        frozen_context
        + final_output_byte_limit
        # Repair sees claims plus their deterministic comparison projection;
        # claim relations are audit-only and are not part of the repair payload.
        + 2 * synthesis_output_byte_limit
        + coverage_upper_bound
        + AUDIT_OUTPUT_BYTE_LIMIT
        + _BUDGET_FIXED_BYTES
    )
    repair = _BUDGET_PROMPT_WRAPPER_BYTES + 2 * repair_payload
    if repair > REPAIR_MAX_INPUT_TOKENS:
        raise ReportAuditBudgetError(
            "worst-case repair prompt exceeds the frozen hard context"
        )
    audit_shards = max(a_shards, c_shards)
    audit_reduce_calls = max(a_reduce_calls, c_reduce_calls)
    audit_calls = max(a_calls, c_calls)
    attempts = MAX_RETRIES + 1
    return AuditRepairBudgetBounds(
        audit_a_input_tokens=audit_a,
        repair_input_tokens=repair,
        audit_c_input_tokens=audit_c,
        audit_shards_per_pass=audit_shards,
        audit_reduce_calls_per_pass=audit_reduce_calls,
        audit_calls_per_pass=audit_calls,
        audit_a_calls=a_calls,
        audit_c_calls=c_calls,
        hard_context_tokens=AUDIT_HARD_CONTEXT_TOKENS,
        max_audit_input_tokens=AUDIT_MAX_INPUT_TOKENS,
        max_repair_input_tokens=REPAIR_MAX_INPUT_TOKENS,
        worst_case_calls=(a_calls + c_calls + 1) * attempts,
        worst_case_input_tokens=(audit_a + repair + audit_c) * attempts,
    )


@dataclass(frozen=True, slots=True)
class _PolicyFacts:
    input_scope: str
    license: str | None
    access_basis: str
    lineage_hash: str
    execution_mode: str


@dataclass(frozen=True, slots=True)
class _StepResult:
    status: str
    output: Mapping[str, Any] | None = None
    metadata: InvocationMetadata | None = None
    resumed: bool = False
    error: str | None = None


@dataclass(frozen=True, slots=True)
class _AuditShardSpec:
    node_id: str
    node_kind: str
    payload: Mapping[str, Any]
    coverage: Mapping[str, Any]
    source_node_ids: tuple[str, ...] = ()
    paper_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _AuditPassPlan:
    audit_pass: str
    full_coverage: Mapping[str, Any]
    direct_payload: Mapping[str, Any] | None
    shards: tuple[_AuditShardSpec, ...]

    @property
    def worst_case_calls(self) -> int:
        if self.direct_payload is not None:
            return 1
        count = len(self.shards)
        return count + max(1, count - 1)


class ReportAuditCoordinator:
    """Run the one-audit/one-repair/fresh-reaudit release state machine."""

    def __init__(
        self,
        database: Database,
        artifact_store: ArtifactStore,
        processing_gate: ProcessingGate,
        report_store: ReportArtifactStore,
        *,
        invoker_factory: Callable[[], SolInvoker] = CodexExec,
        schema_root: Path | None = None,
        prompt_root: Path | None = None,
        resources: ReportResources | None = None,
        rubric_path: Path | None = None,
        execution_mode: str = "attended",
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if execution_mode not in {"attended", "unattended"}:
            raise ValueError("execution_mode must be attended or unattended")
        self.database = database
        self.artifact_store = artifact_store
        self.gate = processing_gate
        self.report_store = report_store
        self.invoker_factory = invoker_factory
        self.schema_root = schema_directory(schema_root)
        self.prompt_root = prompt_directory() if prompt_root is None else prompt_root
        self.resources = resources or ReportResources.defaults(
            schema_root=schema_root, prompt_root=prompt_root
        )
        self.resources.validate_files()
        self.rubric_path = rubric_path or _default_rubric_path()
        self.execution_mode = execution_mode
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.rubric = _load_mapping(self.rubric_path, "report audit rubric")
        self.rubric_hash = content_hash(self.rubric)
        if self.rubric_hash != audit_rubric_hash(self.rubric_path):
            raise ReportAuditError("report audit rubric hash is not deterministic")
        self.schemas = {
            kind: self.resources.schema(kind)
            for kind in ("quality_audit", "repair")
        }
        self.schema_hashes = {kind: _json_hash(value) for kind, value in self.schemas.items()}
        self.service_schema_hashes = {
            kind: self.resources.service_schema_hash(kind)
            for kind in ("quality_audit", "repair")
        }
        self.prompt_hashes = {
            kind: sha256(self.resources.prompt_paths[kind].read_bytes()).hexdigest()
            for kind in ("quality_audit", "repair")
        }
        self.stage4_schema_hash = _json_hash(json.loads(
            (self.schema_root / STAGE4_SCHEMA).read_text(encoding="utf-8")
        ))
        self.stage4_prompt_hash = sha256(
            (self.prompt_root / STAGE4_PROMPT).read_bytes()
        ).hexdigest()
        self.config_hash = stage4b_audit_config_hash(
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
        bundle: ReportBundle,
        *,
        processing_grants: Mapping[str, str] | None = None,
        now: str | None = None,
        worker_id: str | None = None,
        previous: Mapping[str, Any] | None = None,
    ) -> ReportAuditResult:
        """Resume or execute the frozen audit gate and publish only a passing bundle."""
        # Kept as a source-compatible no-op; authorization and leases always
        # use the injected trusted clock immediately before each operation.
        _ = now
        owner = worker_id or f"report-audit-{uuid4()}"
        grants = processing_grants or {}
        initial = _mutable_bundle(bundle)
        try:
            self.report_store.directory(report_run_id)
        except ReportArtifactError as error:
            raise ReportAuditError(str(error)) from error
        self._validate_frozen_inputs(report_run_id, initial)
        try:
            initial["comparison_groups"] = require_exact_comparison_groups(
                initial["claims"], bundle.comparison_groups
            )
        except EvidenceValidationError as error:
            raise ReportAuditError(str(error)) from error
        try:
            initial["claim_relations"] = list(validate_claim_relations(
                previous, initial["claims"], bundle.claim_relations
            ))
        except ReportVerificationError as error:
            raise ReportAuditError(str(error)) from error
        material_limit = self._synthesis_output_byte_limit(report_run_id)
        if (
            len(canonical_json(initial["comparison_groups"])) > material_limit
            or len(canonical_json(initial["claim_relations"])) > material_limit
        ):
            raise ReportAuditError(
                "derived comparison or claim-lineage material exceeds the frozen synthesis bound"
            )
        publication_blockers = search_publication_blockers(initial["search_audit"])
        if publication_blockers:
            snapshot_hash = self._input_snapshot_hash(initial, previous)
            base_hash = _bundle_hash(initial)
            self._ensure_run(
                report_run_id,
                snapshot_hash,
                base_hash,
                initial,
                {"worst_calls": 1, "worst_tokens": 1},
            )
            persisted = self._run_row(report_run_id)
            if persisted["status"] == "complete":
                raise ReportAuditError(
                    "completed report violates the frozen search publication gate"
                )
            if persisted["status"] in {"failed", "incomplete"}:
                return self._terminal_result(report_run_id, persisted)
            return self._finish_incomplete(
                report_run_id,
                initial,
                (),
                (),
                "search audit is not publication-ready: "
                + "; ".join(publication_blockers),
            )
        verification = self._deterministic_verify(initial)
        if previous is not None:
            try:
                report_diff(
                    previous,
                    {
                        "plan": initial["plan"],
                        "claims": initial["claims"],
                        "corpus_snapshot": initial["corpus_snapshot"],
                    },
                    claim_relations=initial["claim_relations"],
                )
            except ReportVerificationError as error:
                raise ReportAuditError(str(error)) from error
        policy_facts = self._trusted_policy_facts(initial)
        initial_coverage = audit_coverage_ledger(initial["document"], initial["claims"])
        preflight_audit = self._audit_payload(
            report_run_id, "A", initial, verification, initial_coverage
        )
        audit_a_plan = self._audit_pass_plan(
            report_run_id,
            "A",
            initial,
            verification,
            initial_coverage,
            preflight_audit,
        )
        budget = self._audit_budget(
            report_run_id, initial["plan"], initial, audit_a_plan
        )
        snapshot_hash = self._input_snapshot_hash(initial, previous)
        base_hash = _bundle_hash(initial)
        self._ensure_run(
            report_run_id,
            snapshot_hash,
            base_hash,
            initial,
            budget,
        )
        persisted = self._run_row(report_run_id)
        if persisted["status"] == "complete":
            return self._completed_result(
                report_run_id,
                persisted,
                initial,
                policy_facts,
                budget,
                previous,
                audit_a_plan,
            )
        if persisted["status"] == "failed":
            return self._terminal_result(report_run_id, persisted)
        persisted_current = self._load_current_bundle(persisted, initial)
        self._set_running(report_run_id)

        # Audit A and its repair are always replayed against the immutable
        # final_reduce output, even after a crash persisted the repaired state.
        current = initial
        expected_coverage = initial_coverage
        reduce_invocations = self._reduce_invocation_ids(report_run_id)
        audit_a = self._run_audit_pass(
            report_run_id,
            "audit_a",
            audit_a_plan,
            current,
            policy_facts,
            budget,
            grants,
            owner,
            forbidden_invocation_ids=reduce_invocations,
        )
        if audit_a.status != "complete":
            return self._stop_for_step(report_run_id, current, audit_a)
        assert audit_a.output is not None and audit_a.metadata is not None
        resumed = ["audit_a"] if audit_a.resumed else []
        audits = ["A"]
        if audit_a.metadata.invocation_id in reduce_invocations:
            return self._finish_failed(
                report_run_id,
                current,
                audits,
                resumed,
                "audit A must use a fresh invocation outside the reduce tree",
            )
        if not bool(audit_a.output["coverage_complete"]):
            return self._finish_incomplete(
                report_run_id, current, audits, resumed,
                "audit A declared incomplete exhaustive coverage",
            )
        severe = _severe_findings(audit_a.output)
        if not severe:
            if int(persisted["repair_count"]) != 0:
                raise ReportAuditError("persisted repair state exists after a passing audit A")
            return self._publish(
                report_run_id, current, audit_a.output, audits, resumed, previous
            )

        repair_payload = self._repair_payload(report_run_id, current, audit_a.output)
        repair = self._run_step(
            report_run_id,
            "repair",
            "repair",
            current,
            repair_payload,
            policy_facts,
            budget["repair"],
            grants,
            self._now(),
            owner,
            None,
            paper_count=len({
                str(item["paper_id"])
                for item in current["corpus_snapshot"]["papers"]
            }),
        )
        if repair.status != "complete":
            return self._stop_for_step(report_run_id, current, repair, audit_passes=audits, resumed=resumed)
        assert repair.output is not None and repair.metadata is not None
        if repair.resumed:
            resumed.append("repair")
        if repair.metadata.invocation_id in {
            *reduce_invocations,
            audit_a.metadata.invocation_id,
        }:
            return self._finish_failed(
                report_run_id,
                current,
                audits,
                resumed,
                "repair B must use a fresh invocation outside audit A and the reduce tree",
            )
        try:
            repaired = self._apply_repair(current, audit_a.output, repair.output)
            try:
                repaired["claim_relations"] = list(validate_claim_relations(
                    previous, repaired["claims"], repaired["claim_relations"]
                ))
            except ReportVerificationError as error:
                raise ReportAuditOutputError(str(error)) from error
            repaired_verification = self._deterministic_verify(repaired)
            if content_hash(repaired["document"]) == content_hash(current["document"]):
                raise ReportAuditOutputError("repair must produce a new ReportDocument hash")
            self._persist_repaired_bundle(report_run_id, current, repaired)
            if int(persisted["repair_count"]) == 1 and _bundle_hash(persisted_current) != _bundle_hash(repaired):
                raise ReportAuditError("resumed repaired bundle differs from its typed patch replay")
        except ReportAuditError as error:
            return self._finish_failed(report_run_id, current, audits, resumed, str(error))
        current = repaired

        expected_coverage = audit_coverage_ledger(
            current["document"], current["claims"]
        )
        audit_c_payload = self._audit_payload(
            report_run_id, "C", current, repaired_verification, expected_coverage
        )
        audit_c_plan = self._audit_pass_plan(
            report_run_id,
            "C",
            current,
            repaired_verification,
            expected_coverage,
            audit_c_payload,
        )
        audit_c = self._run_audit_pass(
            report_run_id,
            "audit_c",
            audit_c_plan,
            current,
            policy_facts,
            budget,
            grants,
            owner,
            forbidden_invocation_ids=frozenset({
                *reduce_invocations,
                audit_a.metadata.invocation_id,
                repair.metadata.invocation_id,
            }),
        )
        if audit_c.status != "complete":
            return self._stop_for_step(report_run_id, current, audit_c, audit_passes=audits, resumed=resumed)
        assert audit_c.output is not None and audit_c.metadata is not None
        if audit_c.resumed:
            resumed.append("audit_c")
        audits.append("C")
        if audit_c.metadata.invocation_id in {
            *reduce_invocations,
            audit_a.metadata.invocation_id,
            repair.metadata.invocation_id,
        }:
            return self._finish_failed(
                report_run_id,
                current,
                audits,
                resumed,
                "audit C must use a fresh invocation outside reduce, audit A, and repair B",
            )
        if not bool(audit_c.output["coverage_complete"]) or _severe_findings(audit_c.output):
            return self._finish_incomplete(
                report_run_id,
                current,
                audits,
                resumed,
                "fresh audit C did not pass full coverage with zero blocker/major findings",
            )
        return self._publish(
            report_run_id, current, audit_c.output, audits, resumed, previous
        )

    def _now(self) -> datetime:
        moment = self.clock()
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        return moment.astimezone(timezone.utc)

    def _validate_frozen_inputs(self, report_run_id: str, bundle: Mapping[str, Any]) -> None:
        plan = bundle["plan"]
        corpus = bundle["corpus_snapshot"]
        search = bundle["search_audit"]
        try:
            require_valid_approval(plan, "plan_hash")
            self.resources.validate(plan, "planning_assist")
            runtime = compile_report_plan(
                plan,
                corpus_snapshot=corpus,
                search_audit_pack=search,
                plan_id=str(plan["plan_id"]),
                created_at=str(plan["created_at"]),
                schema_root=self.schema_root,
                prompt_root=self.prompt_root,
                resources=self.resources,
            )
            assert_report_runtime_matches(
                plan,
                runtime,
                corpus_snapshot=corpus,
                search_audit_pack=search,
            )
        except (
            ApprovalError,
            SchemaValidationError,
            ReportPlanError,
            ReportPlanDriftError,
        ) as error:
            raise ReportAuditError(str(error)) from error
        if str(bundle["document"].get("report_run_id")) != report_run_id:
            raise ReportAuditError("ReportDocument belongs to another report run")
        if plan["schema_hash"] != content_hash(self.resources.schema("planning_assist")):
            raise ReportAuditError("approved ReportPlan schema has drifted")
        expected_prompts = {
            kind: sha256(self.resources.prompt_paths[kind].read_bytes()).hexdigest()
            for kind in CALL_KIND_PROMPTS
        }
        if dict(plan["prompt_hashes"]) != expected_prompts:
            raise ReportAuditError("approved ReportPlan prompt hashes have drifted")
        if plan["stage4b_audit_config_hash"] != self.config_hash:
            raise ReportAuditError(
                "approved ReportPlan Stage 4b audit configuration has drifted"
            )
        self._validate_report_run_binding(report_run_id, bundle)
        self._validate_invocation_registry(report_run_id)
        self._validate_claim_lineage(bundle)
        self._validate_bibliography(bundle)
        rebuilt_coverage = self._rebuild_persisted_coverage(report_run_id, bundle)
        if canonical_json(rebuilt_coverage) != canonical_json(bundle["coverage"]):
            raise ReportAuditError(
                "caller coverage ledger differs from persisted reduce outputs and frozen corpus"
            )

    def _validate_report_run_binding(self, report_run_id: str, bundle: Mapping[str, Any]) -> None:
        stored_plan = self.database.connection.execute(
            """SELECT content_hash, schema_version, plan_json, approval_json, status
               FROM report_plans WHERE report_plan_id = ?""",
            (bundle["plan"]["plan_id"],),
        ).fetchone()
        if stored_plan is None or tuple(stored_plan) != (
            bundle["plan"]["plan_hash"],
            bundle["plan"]["schema_version"],
            _json_text(bundle["plan"]),
            _json_text(bundle["plan"]["approval"]),
            "approved",
        ):
            raise ReportAuditError("persisted approved ReportPlan has drifted")
        self._validate_canonical_reduce_tree(report_run_id, bundle)
        report = self.database.connection.execute(
            """SELECT rr.run_id, rr.report_plan_id, rr.corpus_snapshot_hash,
                      rr.aggregation_tree_json, rr.model_id, rr.model_revision,
                      rr.prompt_hash, rr.schema_hash, rr.status,
                      pr.stage AS pipeline_stage, pr.status AS pipeline_status
               FROM report_runs rr
               JOIN pipeline_runs pr ON pr.run_id = rr.run_id
               WHERE rr.report_run_id = ?""",
            (report_run_id,),
        ).fetchone()
        if report is None:
            raise ReportAuditError("report run must be created by the Stage 4b reduce coordinator")
        completed_audit = self.database.connection.execute(
            "SELECT status FROM report_audit_runs WHERE report_run_id = ?",
            (report_run_id,),
        ).fetchone()
        if (
            report["report_plan_id"] != bundle["plan"]["plan_id"]
            or report["corpus_snapshot_hash"] != bundle["corpus_snapshot"]["snapshot_hash"]
            or report["model_id"] != MODEL
            or report["model_revision"] != "codex-cli-managed"
            or report["pipeline_stage"] != "stage4b"
            or report["pipeline_status"] != report["status"]
            or report["prompt_hash"] != content_hash({
                kind: sha256(self.resources.prompt_paths[kind].read_bytes()).hexdigest()
                for kind in CALL_KIND_PROMPTS
            })
            or report["schema_hash"] != content_hash({
                kind: _json_hash(self.resources.schema(kind))
                for kind in CALL_KIND_SCHEMAS
            })
            or (
                report["status"] == "complete"
                and (completed_audit is None or completed_audit["status"] != "complete")
            )
        ):
            raise ReportAuditError("persisted report run binding has drifted")
        rows = self.database.connection.execute(
            """SELECT rrn.*, a.paper_id AS artifact_paper_id,
                      a.artifact_kind, a.relative_path AS artifact_relative_path,
                      a.mime_type AS artifact_mime_type, a.byte_size AS artifact_byte_size,
                      a.sha256 AS artifact_sha256,
                      a.provenance_json AS artifact_provenance, a.processing_status
               FROM report_reduce_nodes rrn
               LEFT JOIN artifacts a ON a.artifact_id = rrn.output_artifact_id
               WHERE rrn.report_run_id = ? AND rrn.call_kind = 'final_reduce'""",
            (report_run_id,),
        ).fetchall()
        if len(rows) != 1:
            raise ReportAuditError("report run requires exactly one persisted final_reduce output")
        row = rows[0]
        document_hash = content_hash(bundle["document"])
        if (
            row["status"] != "complete"
            or row["output_hash"] != document_hash
            or row["artifact_sha256"] != document_hash
            or row["artifact_kind"] != "report"
            or row["artifact_paper_id"] is not None
            or row["artifact_relative_path"] != self.artifact_store.relative_path(document_hash)
            or row["artifact_mime_type"] != "application/json"
            or row["processing_status"] != "available"
        ):
            raise ReportAuditError("final_reduce output is not the supplied ReportDocument")
        provenance = _json_mapping(row["artifact_provenance"], "final report provenance")
        if provenance != {"stage": "stage4b", "content_hash": document_hash}:
            raise ReportAuditError("final_reduce artifact provenance has drifted")
        try:
            payload = self.artifact_store.read_bytes(
                document_hash, max_bytes=int(row["output_byte_limit"])
            )
        except (OSError, ValueError) as error:
            raise ReportAuditError(
                "final_reduce output exceeds its frozen artifact limit"
            ) from error
        if (
            int(row["artifact_byte_size"]) != len(payload)
            or json.loads(payload) != bundle["document"]
        ):
            raise ReportAuditError("final_reduce artifact bytes differ from ReportDocument")
        self._validate_reduce_node_contract(row, bundle["document"])

    def _validate_canonical_reduce_tree(
        self, report_run_id: str, bundle: Mapping[str, Any]
    ) -> None:
        """Rebuild and replay-bind the complete approved reduce tree before audit."""
        from .report_reduce import (
            IMPLEMENTATION_VERSION as REDUCE_IMPLEMENTATION_VERSION,
            MAX_NODE_OUTPUT_BYTES,
            MIN_NODE_OUTPUT_BYTES,
            OUTPUT_BYTES_PER_ESTIMATED_TOKEN,
            stage4b_reduce_config_hash,
        )

        plan = bundle["plan"]
        corpus_papers = {
            str(item["paper_id"]): item
            for item in bundle["corpus_snapshot"]["papers"]
        }
        records: list[AnalysisRecord] = []
        source_documents: dict[str, Mapping[str, Any]] = {}
        source_lineages: dict[str, tuple[str, ...]] = {}
        for paper_id in sorted(corpus_papers):
            paper = corpus_papers[paper_id]
            artifact_hash = str(paper["analysis_artifact_hash"])
            document = _json_mapping(
                self.artifact_store.read_bytes(artifact_hash),
                "frozen Stage 4 analysis",
            )
            try:
                validate(document, "paper-analysis.schema.json", self.schema_root)
            except SchemaValidationError as error:
                raise ReportAuditError(str(error)) from error
            if (
                document["paper_id"] != paper_id
                or document["input_scope"] != paper["input_scope"]
            ):
                raise ReportAuditError("frozen Stage 4 analysis document has drifted")
            labels = document["labels"]
            classifications = {
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
                frozen_value = str(paper[axis])
                if _classification_values(labels.get(axis)) != (frozen_value,):
                    raise ReportAuditError(
                        "analysis classification labels conflict with the frozen corpus"
                    )
                classifications[axis] = (frozen_value,)
            year = (
                str(paper["publication_year"])
                if paper.get("publication_year") is not None
                else str(paper.get("publication_date") or "")[:4]
            )
            if year:
                classifications["time"] = (year,)
            venue = str(paper.get("venue_id") or paper.get("venue_name") or "").strip()
            if venue:
                classifications["venue"] = (venue,)
            records.append(AnalysisRecord(
                paper_id=paper_id,
                analysis_run_id=str(paper["analysis_run_id"]),
                analysis_hash=artifact_hash,
                input_scope=str(paper["input_scope"]),
                input_tokens=int(paper["analysis_input_tokens"]),
                classifications=classifications,
                evidence_units=tuple(document["evidence_units"]),
            ))
            lineages = tuple(sorted(set(str(item) for item in paper["lineage_hashes"])))
            if not lineages or any(not _is_hash(item) for item in lineages):
                raise ReportAuditError("frozen corpus analysis has invalid source lineage")
            source_documents[artifact_hash] = {
                "artifact_hash": artifact_hash,
                "artifact_kind": "analysis",
                "lineage_hash": content_hash(lineages),
                "document": document,
            }
            source_lineages[artifact_hash] = lineages

        aggregation = plan["aggregation"]
        try:
            reduce_plan = ReportPlanner(
                plan,
                tuple(records),
                max_chunk_input_tokens=int(aggregation["max_chunk_input_tokens"]),
                reduce_output_tokens=int(aggregation["reduce_output_tokens"]),
                audit_input_tokens=1,
                repair_input_tokens=1,
            ).build()
        except (ReportPlanningError, ValueError, TypeError) as error:
            raise ReportAuditError("approved inputs cannot rebuild the canonical reduce tree") from error

        rows = self.database.connection.execute(
            "SELECT * FROM report_reduce_nodes WHERE report_run_id = ?",
            (report_run_id,),
        ).fetchall()
        by_node = {str(row["node_id"]): row for row in rows}
        expected_ids = {node.node_id for node in reduce_plan.nodes}
        if len(by_node) != len(rows) or set(by_node) != expected_ids:
            raise ReportAuditError(
                "persisted reduce nodes do not exactly match the canonical tree"
            )

        consumers: dict[str, list[Any]] = {}
        for parent in reduce_plan.nodes:
            for dependency in parent.dependency_ids:
                consumers.setdefault(dependency, []).append(parent)
        final_words = sum(int(section["target_words"]) for section in plan["sections"])
        output_limits: dict[str, int] = {}
        for node in reduce_plan.nodes:
            if node.call_kind == "final_reduce":
                estimated_tokens = max(1, final_words * 2)
            else:
                contributions = [
                    math.ceil(parent.input_tokens / len(parent.dependency_ids))
                    for parent in consumers.get(node.node_id, ())
                ]
                estimated_tokens = max(contributions or [node.input_tokens])
            output_limits[node.node_id] = min(
                MAX_NODE_OUTPUT_BYTES,
                max(
                    MIN_NODE_OUTPUT_BYTES,
                    estimated_tokens * OUTPUT_BYTES_PER_ESTIMATED_TOKEN,
                ),
            )

        chunks = {chunk.node_id: chunk for chunk in reduce_plan.chunks}
        prompt_bounds: dict[str, int] = {}
        for node in reduce_plan.nodes:
            if not node.dependency_ids:
                inputs = [
                    source_documents[value]
                    for value in chunks[node.node_id].analysis_hashes
                ]
                payload = _reduce_prompt_payload(report_run_id, plan, node, inputs)
                prompt = canonical_json(payload).decode("utf-8")
                prompt_bounds[node.node_id] = _token_upper_bound(
                    self._rendered_prompt(node.call_kind, prompt)
                )
            else:
                placeholders = [
                    {
                        "artifact_hash": "0" * 64,
                        "artifact_kind": _reduce_output_kind(
                            next(
                                item.call_kind
                                for item in reduce_plan.nodes
                                if item.node_id == dependency
                            )
                        ),
                        "lineage_hash": "0" * 64,
                        "document": {},
                    }
                    for dependency in node.dependency_ids
                ]
                empty_prompt = canonical_json(
                    _reduce_prompt_payload(report_run_id, plan, node, placeholders)
                ).decode("utf-8")
                fixed = _token_upper_bound(
                    self._rendered_prompt(node.call_kind, empty_prompt)
                )
                prompt_bounds[node.node_id] = fixed + 2 * sum(
                    output_limits[dependency] for dependency in node.dependency_ids
                )

        final = reduce_plan.nodes[-1]
        synthesis_id = final.dependency_ids[0]
        audit_bounds = stage4b_audit_repair_budget_bounds(
            plan,
            bundle["corpus_snapshot"],
            bundle["search_audit"],
            final_output_byte_limit=output_limits[final.node_id],
            synthesis_output_byte_limit=output_limits[synthesis_id],
            rubric_path=self.rubric_path,
        )
        report = self.database.connection.execute(
            """SELECT rr.aggregation_tree_json, rr.run_id,
                      pr.input_hash, pr.config_hash, pr.implementation_version
               FROM report_runs rr
               JOIN pipeline_runs pr ON pr.run_id = rr.run_id
               WHERE rr.report_run_id = ?""",
            (report_run_id,),
        ).fetchone()
        if report is None:
            raise ReportAuditError("canonical reduce run is missing")
        tree = _json_mapping(report["aggregation_tree_json"], "aggregation tree")
        try:
            expected_budget = canonical_report_budget(
                reduce_plan.nodes,
                prompt_bounds,
                asdict(audit_bounds),
                max_retries=int(plan["budget"]["max_retries"]),
            )
        except CanonicalReportBudgetError as error:
            raise ReportAuditError(str(error)) from error
        expected_tree = {
            "chunks": [asdict(item) for item in reduce_plan.chunks],
            "nodes": [asdict(item) for item in reduce_plan.nodes],
            "budget": expected_budget,
            "execution_mode": self.execution_mode,
            "audit_repair_budget_bounds": asdict(audit_bounds),
        }
        if (
            expected_budget["worst_case_calls"]
            > int(plan["budget"]["max_sol_calls"])
            or expected_budget["worst_case_input_tokens"]
            > int(plan["budget"]["max_input_tokens"])
            or canonical_json(tree) != canonical_json(expected_tree)
        ):
            raise ReportAuditError(
                "persisted aggregation tree differs from the canonical approved tree"
            )

        expected_reduce_config = stage4b_reduce_config_hash(
            self.gate.policy.hash,
            execution_mode=self.execution_mode,
            schema_root=self.schema_root,
            prompt_root=self.prompt_root,
            resources=self.resources,
        )
        pipeline_input_hash = content_hash({
            "plan_hash": plan["plan_hash"],
            "corpus_snapshot_hash": bundle["corpus_snapshot"]["snapshot_hash"],
            "search_audit_pack_hash": bundle["search_audit"]["pack_hash"],
            "tree": expected_tree,
            "prompt_token_bounds": prompt_bounds,
            "output_byte_limits": output_limits,
            "audit_repair_budget_bounds": asdict(audit_bounds),
        })
        if (
            plan["stage4b_config_hash"] != expected_reduce_config
            or report["input_hash"] != pipeline_input_hash
            or report["config_hash"] != expected_reduce_config
            or report["implementation_version"] != REDUCE_IMPLEMENTATION_VERSION
        ):
            raise ReportAuditError(
                "Stage 4b pipeline input, configuration, or implementation has drifted"
            )

        outputs: dict[str, Mapping[str, Any]] = {}
        node_lineages: dict[str, tuple[str, ...]] = {}
        node_by_id = {node.node_id: node for node in reduce_plan.nodes}
        for node in reduce_plan.nodes:
            row = by_node[node.node_id]
            expected_static = (
                "complete",
                node.call_kind,
                _json_text(node.section_ids),
                _json_text(node.paper_ids),
                _json_text(node.dependency_ids),
                node.planned_input_hash,
                node.input_tokens,
                prompt_bounds[node.node_id],
                output_limits[node.node_id],
                PROFILE,
                MODEL,
                REASONING_EFFORT,
                CALL_KIND_PROMPTS[node.call_kind],
                sha256(self.resources.prompt_paths[node.call_kind].read_bytes()).hexdigest(),
                CALL_KIND_SCHEMAS[node.call_kind],
                _json_hash(self.resources.schema(node.call_kind)),
            )
            actual_static = tuple(row[key] for key in (
                "status",
                "call_kind",
                "section_ids_json",
                "paper_ids_json",
                "dependency_ids_json",
                "planned_input_hash",
                "input_tokens",
                "prompt_token_bound",
                "output_byte_limit",
                "profile",
                "model_id",
                "reasoning_effort",
                "prompt_name",
                "prompt_hash",
                "schema_name",
                "schema_hash",
            ))
            if actual_static != expected_static:
                raise ReportAuditError(
                    f"persisted reduce node binding has drifted: {node.node_id}"
                )
            output = self._persisted_reduce_output(row)
            if not node.dependency_ids:
                input_documents = [
                    source_documents[value]
                    for value in chunks[node.node_id].analysis_hashes
                ]
                lineages = tuple(sorted({
                    lineage
                    for value in chunks[node.node_id].analysis_hashes
                    for lineage in source_lineages[value]
                }))
            else:
                lineages = tuple(sorted({
                    lineage
                    for dependency in node.dependency_ids
                    for lineage in node_lineages[dependency]
                }))
                input_documents = [
                    {
                        "artifact_hash": str(by_node[dependency]["output_hash"]),
                        "artifact_kind": _reduce_output_kind(
                            node_by_id[dependency].call_kind
                        ),
                        "lineage_hash": content_hash(node_lineages[dependency]),
                        "document": outputs[dependency],
                    }
                    for dependency in node.dependency_ids
                ]
            prompt = canonical_json(
                _reduce_prompt_payload(report_run_id, plan, node, input_documents)
            ).decode("utf-8")
            rendered = self._rendered_prompt(node.call_kind, prompt)
            if (
                row["actual_input_hash"] != sha256(prompt.encode("utf-8")).hexdigest()
                or row["rendered_prompt_hash"]
                != sha256(rendered.encode("utf-8")).hexdigest()
                or int(row["actual_input_tokens"]) != _token_upper_bound(rendered)
                or json.loads(row["input_artifact_hashes_json"])
                != [str(item["artifact_hash"]) for item in input_documents]
            ):
                raise ReportAuditError(
                    f"persisted reduce node input replay has drifted: {node.node_id}"
                )
            policy = _json_mapping(row["output_policy_json"], "reduce output policy")
            if (
                policy.get("artifact_kind") != _reduce_output_kind(node.call_kind)
                or policy.get("lineage_hash") != content_hash(lineages)
                or policy.get("source_lineage_hashes") != list(lineages)
                or policy.get("source_paper_ids") != list(node.paper_ids)
                or policy.get("mode") != self.execution_mode
            ):
                raise ReportAuditError(
                    f"persisted reduce output policy has drifted: {node.node_id}"
                )
            outputs[node.node_id] = output
            node_lineages[node.node_id] = lineages

    def _validate_reduce_node_contract(
        self, row: Any, output: Mapping[str, Any]
    ) -> None:
        call_kind = str(row["call_kind"])
        if call_kind not in {"section_reduce", "cross_section_reduce", "final_reduce"}:
            raise ReportAuditError("persisted reduce node has an unsupported call kind")
        schema_name = CALL_KIND_SCHEMAS[call_kind]
        prompt_name = CALL_KIND_PROMPTS[call_kind]
        try:
            schema = self.resources.schema(call_kind)
            self.resources.validate(output, call_kind)
        except (OSError, SchemaValidationError) as error:
            raise ReportAuditError("persisted reduce output violates its frozen schema") from error
        expected_row = (
            "complete",
            PROFILE,
            MODEL,
            REASONING_EFFORT,
            prompt_name,
            sha256(self.resources.prompt_paths[call_kind].read_bytes()).hexdigest(),
            schema_name,
            _json_hash(schema),
        )
        actual_row = (
            row["status"],
            row["profile"],
            row["model_id"],
            row["reasoning_effort"],
            row["prompt_name"],
            row["prompt_hash"],
            row["schema_name"],
            row["schema_hash"],
        )
        metadata = _metadata(row["invocation_metadata_json"])
        expected_metadata = (
            PROFILE,
            MODEL,
            REASONING_EFFORT,
            call_kind,
            schema_name,
            _json_hash(schema),
            prompt_name,
            expected_row[5],
            row["actual_input_hash"],
            row["rendered_prompt_hash"],
            MODEL,
            PROFILE,
        )
        actual_metadata = (
            metadata.profile,
            metadata.model,
            metadata.reasoning_effort,
            metadata.call_kind,
            metadata.schema_name,
            metadata.schema_hash,
            metadata.prompt_name,
            metadata.prompt_hash,
            metadata.input_hash,
            metadata.rendered_prompt_hash,
            metadata.actual_model,
            metadata.actual_profile,
        )
        actual_tokens = row["actual_input_tokens"]
        if (
            actual_row != expected_row
            or actual_metadata != expected_metadata
            or not self.resources.accepts_metadata_paths(
                call_kind, metadata.schema_path, metadata.prompt_path
            )
            or not str(metadata.invocation_id).strip()
            or metadata.attempts < 1
            or metadata.attempts > MAX_RETRIES + 1
            or not _is_hash(row["actual_input_hash"])
            or not _is_hash(row["rendered_prompt_hash"])
            or actual_tokens is None
            or int(actual_tokens) > int(row["prompt_token_bound"])
            or int(row["dispatch_count"]) != 1
            or int(row["budget_calls_reserved"]) != MAX_RETRIES + 1
            or int(row["budget_tokens_reserved"])
            != int(actual_tokens) * (MAX_RETRIES + 1)
            or row["output_hash"] != content_hash(output)
            or len(canonical_json(output)) > int(row["output_byte_limit"])
        ):
            raise ReportAuditError("persisted reduce invocation binding has drifted")
        try:
            require_report_invocation(
                self.database.connection,
                report_run_id=str(row["report_run_id"]),
                invocation_id=metadata.invocation_id,
                phase="reduce",
                node_key=str(row["node_id"]),
                metadata=asdict(metadata),
            )
        except ReportInvocationError as error:
            raise ReportAuditError(str(error)) from error

    def _reduce_invocation_ids(self, report_run_id: str) -> frozenset[str]:
        rows = self.database.connection.execute(
            """SELECT status, node_id, invocation_id, invocation_metadata_json
               FROM report_reduce_nodes
               WHERE report_run_id = ?""",
            (report_run_id,),
        ).fetchall()
        if not rows or any(row["status"] != "complete" for row in rows):
            raise ReportAuditError("reduce invocation ledger is incomplete")
        invocation_ids: list[str] = []
        for row in rows:
            metadata = _metadata(row["invocation_metadata_json"])
            if row["invocation_id"] != metadata.invocation_id:
                raise ReportAuditError("reduce invocation identity has drifted")
            try:
                require_report_invocation(
                    self.database.connection,
                    report_run_id=report_run_id,
                    invocation_id=metadata.invocation_id,
                    phase="reduce",
                    node_key=str(row["node_id"]),
                    metadata=asdict(metadata),
                )
            except ReportInvocationError as error:
                raise ReportAuditError(str(error)) from error
            invocation_ids.append(str(metadata.invocation_id))
        if any(not value.strip() for value in invocation_ids) or len(set(invocation_ids)) != len(
            invocation_ids
        ):
            raise ReportAuditError("reduce nodes did not use distinct fresh invocations")
        return frozenset(invocation_ids)

    def _validate_invocation_registry(self, report_run_id: str) -> None:
        expected: dict[tuple[str, str], tuple[str, str]] = {}
        sources = (
            (
                "reduce",
                """SELECT node_id AS node_key, invocation_metadata_json
                   FROM report_reduce_nodes
                   WHERE report_run_id = ? AND status = 'complete'""",
            ),
            (
                "audit_step",
                """SELECT step_name AS node_key, invocation_metadata_json
                   FROM report_audit_steps
                   WHERE report_run_id = ? AND status = 'complete'""",
            ),
        )
        for phase, statement in sources:
            for row in self.database.connection.execute(
                statement, (report_run_id,)
            ).fetchall():
                metadata = _metadata(row["invocation_metadata_json"])
                expected[(phase, str(row["node_key"]))] = (
                    metadata.invocation_id,
                    report_invocation_metadata_hash(asdict(metadata)),
                )
        for row in self.database.connection.execute(
            """SELECT audit_pass, node_id, invocation_metadata_json
               FROM report_audit_shard_steps
               WHERE report_run_id = ? AND status = 'complete'""",
            (report_run_id,),
        ).fetchall():
            metadata = _metadata(row["invocation_metadata_json"])
            expected[("audit_shard", f"{row['audit_pass']}:{row['node_id']}")] = (
                metadata.invocation_id,
                report_invocation_metadata_hash(asdict(metadata)),
            )
        persisted = {
            (str(row["phase"]), str(row["node_key"])): (
                str(row["invocation_id"]),
                str(row["metadata_hash"]),
            )
            for row in self.database.connection.execute(
                """SELECT phase, node_key, invocation_id, metadata_hash
                   FROM report_sol_invocations WHERE report_run_id = ?""",
                (report_run_id,),
            ).fetchall()
        }
        if persisted != expected or len({value[0] for value in persisted.values()}) != len(
            persisted
        ):
            raise ReportAuditError(
                "run-wide Sol invocation registry is incomplete, duplicated, or orphaned"
            )

    def _validate_claim_lineage(self, bundle: Mapping[str, Any]) -> None:
        corpus = {
            str(item["paper_id"]): item for item in bundle["corpus_snapshot"]["papers"]
        }
        evidence_hashes: dict[str, set[str]] = {}
        for paper_id, paper in corpus.items():
            analysis = _json_mapping(
                self.artifact_store.read_bytes(str(paper["analysis_artifact_hash"])),
                "frozen Stage 4 analysis",
            )
            if (
                analysis.get("paper_id") != paper_id
                or analysis.get("input_scope") != paper["input_scope"]
            ):
                raise ReportAuditError("frozen Stage 4 analysis document has drifted")
            evidence_hashes[paper_id] = {
                content_hash(item) for item in analysis.get("evidence_units", ())
            }
        for claim in bundle["claims"]:
            for field, direction in (
                ("supporting_evidence", "support"),
                ("contradicting_evidence", "contradict"),
            ):
                for ref in claim[field]:
                    if ref["kind"] != "paper_evidence":
                        continue
                    paper = corpus.get(str(ref["paper_id"]))
                    if paper is None or ref["analysis_run_id"] != paper["analysis_run_id"]:
                        raise ReportAuditError("claim evidence is not bound to the frozen Stage 4 analysis")
                    unit = ref.get("evidence_unit")
                    if (
                        not isinstance(unit, Mapping)
                        or content_hash(unit) not in evidence_hashes[str(ref["paper_id"])]
                        or unit.get("direction") != direction
                        or not str(ref.get("locator") or "").strip()
                    ):
                        raise ReportAuditError(
                            "claim evidence unit or locator differs from its Stage 4 analysis"
                        )

    def _validate_bibliography(self, bundle: Mapping[str, Any]) -> None:
        cited = {
            str(paper_id)
            for block in bundle["document"]["blocks"]
            for paper_id in block["citation_paper_ids"]
        }
        if set(str(value) for value in bundle["bibliography"]) != cited:
            raise ReportAuditError("bibliography does not exactly match cited papers")
        corpus = {
            str(item["paper_id"]): item
            for item in bundle["corpus_snapshot"]["papers"]
        }
        corpus_ids = set(corpus)
        if not cited.issubset(corpus_ids):
            raise ReportAuditError("bibliography cites a paper outside the frozen corpus")
        for paper_id in sorted(cited):
            frozen = corpus[paper_id]
            if frozen.get("verification_status") not in {"verified", "single_source"}:
                raise ReportAuditError(
                    f"bibliography paper lacks frozen verified metadata: {paper_id}"
                )
            expected = {
                key: value
                for key, value in {
                    "title": frozen.get("title"),
                    "authors": frozen.get("authors"),
                    "year": frozen.get("publication_year"),
                    "venue_name": frozen.get("venue_name"),
                    "doi": frozen.get("doi"),
                    "canonical_url": frozen.get("canonical_url"),
                }.items()
                if value is not None
            }
            row = self.database.connection.execute(
                """SELECT title, authors_json, year, venue_name, doi, canonical_url,
                          verification_status
                   FROM papers WHERE paper_id = ?""",
                (paper_id,),
            ).fetchone()
            if row is None or row["verification_status"] not in {"verified", "single_source"}:
                raise ReportAuditError(
                    f"bibliography paper lacks verified canonical metadata: {paper_id}"
                )
            try:
                authors = json.loads(row["authors_json"])
            except (TypeError, json.JSONDecodeError) as error:
                raise ReportAuditError("canonical paper authors are malformed") from error
            live = {
                key: value
                for key, value in {
                    "title": row["title"],
                    "authors": authors,
                    "year": row["year"],
                    "venue_name": row["venue_name"],
                    "doi": row["doi"],
                    "canonical_url": row["canonical_url"],
                }.items()
                if value is not None
            }
            if canonical_json(live) != canonical_json(expected):
                raise ReportAuditError(
                    f"canonical bibliography metadata drifted after corpus freeze: {paper_id}"
                )
            if canonical_json(bundle["bibliography"][paper_id]) != canonical_json(expected):
                raise ReportAuditError(
                    f"bibliography differs from canonical metadata: {paper_id}"
                )

    def _rebuild_persisted_coverage(
        self, report_run_id: str, bundle: Mapping[str, Any]
    ) -> dict[str, Any]:
        rows = self.database.connection.execute(
            """SELECT * FROM report_reduce_nodes
               WHERE report_run_id = ? ORDER BY node_id""",
            (report_run_id,),
        ).fetchall()
        if not rows or any(row["status"] != "complete" for row in rows):
            raise ReportAuditError("all persisted reduce nodes must complete before report audit")
        by_node = {str(row["node_id"]): row for row in rows}
        if len(by_node) != len(rows):
            raise ReportAuditError("persisted reduce tree contains duplicate node IDs")
        selected = {
            str(item["paper_id"]): item for item in bundle["corpus_snapshot"]["papers"]
        }
        paper_claims: dict[str, set[str]] = {paper_id: set() for paper_id in selected}
        consumed: dict[str, set[str]] = {paper_id: set() for paper_id in selected}
        persisted_outputs = {
            str(row["node_id"]): self._persisted_reduce_output(row) for row in rows
        }
        for row in rows:
            if row["call_kind"] != "section_reduce":
                continue
            planned_papers = tuple(str(item) for item in json.loads(row["paper_ids_json"]))
            if not set(planned_papers).issubset(selected):
                raise ReportAuditError("section reduce node contains a paper outside the frozen corpus")
            output = persisted_outputs[str(row["node_id"])]
            for paper_id in planned_papers:
                consumed[paper_id].add(str(row["node_id"]))
            for claim in output.get("claims", ()):
                claim_id = str(claim.get("claim_id") or "")
                for field in ("supporting_evidence", "contradicting_evidence"):
                    for reference in claim.get(field, ()):
                        if reference.get("kind") != "paper_evidence":
                            continue
                        paper_id = str(reference.get("paper_id") or "")
                        if paper_id not in planned_papers:
                            raise ReportAuditError(
                                "section reduce output introduced evidence outside its planned papers"
                            )
                        paper_claims[paper_id].add(claim_id)

        final_rows = [row for row in rows if row["call_kind"] == "final_reduce"]
        if len(final_rows) != 1:
            raise ReportAuditError("persisted reduce tree requires one final_reduce node")
        dependencies = tuple(
            str(item) for item in json.loads(final_rows[0]["dependency_ids_json"])
        )
        if len(dependencies) != 1 or dependencies[0] not in by_node:
            raise ReportAuditError("final_reduce must bind one persisted synthesis dependency")
        synthesis = persisted_outputs[dependencies[0]]
        if canonical_json(synthesis.get("claims", ())) != canonical_json(bundle["claims"]):
            raise ReportAuditError("caller Claims-Evidence matrix differs from persisted reduce output")

        memberships = {
            str(item["paper_id"]): item for item in bundle["plan"]["paper_memberships"]
        }
        if set(memberships) != set(selected):
            raise ReportAuditError(
                "approved coverage dispositions do not match the frozen corpus"
            )
        for paper_id, membership in memberships.items():
            has_evidence = bool(paper_claims[paper_id])
            expects_evidence = membership["coverage_disposition"] == "evidence"
            if has_evidence != expects_evidence:
                raise ReportAuditError(
                    "persisted evidence differs from the approved coverage disposition: "
                    f"{paper_id}"
                )
        uncovered_claims = sorted(
            str(claim["claim_id"])
            for claim in bundle["claims"]
            if not tuple(claim["supporting_evidence"]) + tuple(claim["contradicting_evidence"])
        )
        return {
            "papers": [
                {
                    "paper_id": paper_id,
                    "evidence_claim_ids": sorted(paper_claims[paper_id]),
                    "consumed_node_ids": sorted(consumed[paper_id]),
                    "disposition": memberships[paper_id]["coverage_disposition"],
                    "reason": memberships[paper_id]["coverage_reason"],
                }
                for paper_id in sorted(selected)
            ],
            "missing_paper_ids": [],
            "uncovered_claim_ids": uncovered_claims,
            "complete": not uncovered_claims,
        }

    def _persisted_reduce_output(self, row: Any) -> Mapping[str, Any]:
        if row["output_artifact_id"] is None or not _is_hash(row["output_hash"]):
            raise ReportAuditError("completed reduce node lacks an output artifact binding")
        artifact = self.database.connection.execute(
            """SELECT paper_id, artifact_kind, relative_path, mime_type, byte_size,
                      sha256, provenance_json, processing_status
               FROM artifacts WHERE artifact_id = ?""",
            (row["output_artifact_id"],),
        ).fetchone()
        if artifact is None or (
            artifact["paper_id"],
            artifact["artifact_kind"],
            artifact["relative_path"],
            artifact["mime_type"],
            artifact["sha256"],
            artifact["processing_status"],
        ) != (
            None,
            "report",
            self.artifact_store.relative_path(str(row["output_hash"])),
            "application/json",
            row["output_hash"],
            "available",
        ):
            raise ReportAuditError("persisted reduce output artifact has drifted")
        provenance = _json_mapping(artifact["provenance_json"], "reduce output provenance")
        if provenance != {"stage": "stage4b", "content_hash": row["output_hash"]}:
            raise ReportAuditError("persisted reduce output provenance has drifted")
        try:
            payload = self.artifact_store.read_bytes(
                str(row["output_hash"]), max_bytes=int(row["output_byte_limit"])
            )
        except (OSError, ValueError) as error:
            raise ReportAuditError("persisted reduce output exceeds its frozen limit") from error
        if int(artifact["byte_size"]) != len(payload):
            raise ReportAuditError("persisted reduce output byte size has drifted")
        output = _json_mapping(payload, "persisted reduce output")
        self._validate_reduce_node_contract(row, output)
        return output

    def _deterministic_verify(self, bundle: Mapping[str, Any]) -> Mapping[str, Any]:
        try:
            first = verify_report(
                plan=bundle["plan"],
                document=bundle["document"],
                claims=bundle["claims"],
                coverage=bundle["coverage"],
                bibliography=bundle["bibliography"],
                comparison_groups=bundle["comparison_groups"],
                search_audit=bundle["search_audit"],
                corpus_snapshot=bundle["corpus_snapshot"],
            )
            first_render = render_markdown(
                plan=bundle["plan"],
                document=bundle["document"],
                claims=bundle["claims"],
                bibliography=bundle["bibliography"],
                search_audit=bundle["search_audit"],
                corpus_snapshot=bundle["corpus_snapshot"],
            )
            second_render = render_markdown(
                plan=bundle["plan"],
                document=bundle["document"],
                claims=bundle["claims"],
                bibliography=bundle["bibliography"],
                search_audit=bundle["search_audit"],
                corpus_snapshot=bundle["corpus_snapshot"],
            )
        except ReportVerificationError as error:
            raise ReportAuditError(str(error)) from error
        if first_render != second_render:
            raise ReportAuditError("report renderer is not deterministic")
        markdown, sidecar = first_render
        return {
            **first,
            "report_document_hash": content_hash(bundle["document"]),
            "renderer_version": first["renderer_version"],
            "rendered_markdown_hash": content_hash(markdown),
            "sidecar_hash": content_hash(sidecar),
        }

    def _trusted_policy_facts(self, bundle: Mapping[str, Any]) -> _PolicyFacts:
        scopes: list[str] = []
        licenses: set[str | None] = set()
        access: set[str] = set()
        lineages: set[str] = set()
        for paper in bundle["corpus_snapshot"]["papers"]:
            row = self.database.connection.execute(
                """SELECT ar.*,
                          output.paper_id AS output_paper_id,
                          output.artifact_kind AS output_kind,
                          output.relative_path AS output_relative_path,
                          output.mime_type AS output_mime_type,
                          output.byte_size AS output_byte_size,
                          output.sha256 AS output_sha256,
                          output.provenance_json AS output_provenance_json,
                          output.processing_status AS output_status,
                          stage4.stage AS pipeline_stage,
                          stage4.status AS pipeline_status,
                          stage4.input_hash AS pipeline_input_hash,
                          stage4.config_hash AS pipeline_config_hash,
                          stage4.implementation_version AS pipeline_implementation_version
                   FROM analysis_runs ar
                   JOIN artifacts output ON output.artifact_id = ar.output_artifact_id
                   JOIN pipeline_runs stage4 ON stage4.run_id = ar.run_id
                   WHERE ar.analysis_run_id = ?""",
                (paper["analysis_run_id"],),
            ).fetchone()
            if row is None or (
                row["paper_id"],
                row["input_scope"],
                row["model_id"],
                row["status"],
                row["output_sha256"],
                row["output_paper_id"],
                row["output_kind"],
                row["output_status"],
                row["pipeline_stage"],
                row["pipeline_status"],
                row["pipeline_input_hash"],
                row["pipeline_config_hash"],
                row["pipeline_implementation_version"],
                row["implementation_version"],
            ) != (
                paper["paper_id"],
                paper["input_scope"],
                PROCESSING_MODEL,
                "complete",
                paper["analysis_artifact_hash"],
                paper["paper_id"],
                "analysis",
                "available",
                "stage4",
                "complete",
                paper["analysis_pipeline_input_hash"],
                paper["analysis_config_hash"],
                paper["analysis_implementation_version"],
                paper["analysis_implementation_version"],
            ):
                raise ReportAuditError("frozen Stage 4 analysis provenance has drifted")
            analysis_payload = self.artifact_store.read_bytes(str(row["output_sha256"]))
            if (
                row["output_relative_path"]
                != self.artifact_store.relative_path(str(row["output_sha256"]))
                or row["output_mime_type"] != "application/json"
                or int(row["output_byte_size"]) != len(analysis_payload)
            ):
                raise ReportAuditError("persisted Stage 4 analysis artifact metadata has drifted")
            analysis_document = _json_mapping(analysis_payload, "persisted Stage 4 analysis")
            try:
                validate(analysis_document, "paper-analysis.schema.json", self.schema_root)
            except SchemaValidationError as error:
                raise ReportAuditError(str(error)) from error
            output_provenance = _json_mapping(
                row["output_provenance_json"], "Stage 4 analysis provenance"
            )
            if (
                output_provenance.get("analysis_run_id") != paper["analysis_run_id"]
                or output_provenance.get("stage") != "stage4"
                or analysis_document["paper_id"] != paper["paper_id"]
                or analysis_document["input_scope"] != paper["input_scope"]
                or analysis_document["model"] != PROCESSING_MODEL
            ):
                raise ReportAuditError("persisted Stage 4 analysis document binding has drifted")
            detail = _json_mapping(
                row["invocation_metadata_json"], "Stage 4 invocation metadata"
            )
            input_policy_facts = detail.get("input_policy_facts")
            if not isinstance(input_policy_facts, Mapping):
                raise ReportAuditError(
                    "Stage 4 analysis lacks persisted input policy facts"
                )
            if (
                detail.get("report_input_tokens") != paper["analysis_input_tokens"]
                or content_hash(dict(input_policy_facts))
                != paper["analysis_policy_facts_hash"]
                or input_policy_facts.get("paper_id") != paper["paper_id"]
                or input_policy_facts.get("artifact_hash")
                != analysis_document["artifact_hash"]
                or input_policy_facts.get("input_scope") != paper["input_scope"]
            ):
                raise ReportAuditError(
                    "Stage 4 input policy facts have drifted from the frozen corpus"
                )
            invocation_document = detail.get("invocation")
            if not isinstance(invocation_document, Mapping):
                raise ReportAuditError("Stage 4 analysis lacks persisted Luna invocation metadata")
            try:
                invocation = InvocationMetadata(**dict(invocation_document))
            except TypeError as error:
                raise ReportAuditError("Stage 4 Luna invocation metadata is malformed") from error
            expected_invocation = (
                STAGE4_PROFILE,
                PROCESSING_MODEL,
                STAGE4_REASONING_EFFORT,
                STAGE4_SCHEMA,
                self.stage4_schema_hash,
                paper["analysis_prompt_input_hash"],
                STAGE4_PROMPT,
                self.stage4_prompt_hash,
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
                or invocation.invocation_id != paper["analysis_invocation_id"]
                or not _is_hash(invocation.input_hash)
                or invocation.rendered_prompt_hash
                != paper["analysis_rendered_prompt_hash"]
                or row["input_hash"] != paper["analysis_prompt_input_hash"]
                or invocation.attempts < 1
                or invocation.attempts > STAGE4_MAX_RETRIES + 1
            ):
                raise ReportAuditError(
                    "Stage 4 invocation metadata does not match the frozen Luna profile"
                )
            decision = detail.get("processing_decision")
            if not isinstance(decision, Mapping):
                raise ReportAuditError("Stage 4 analysis lacks its processing decision")
            allowed_categories = {
                "full_pdf": {"full_text", "normalized_text"},
                "abstract_only": {"abstract"},
                "metadata_only": {"metadata"},
            }
            if (
                decision.get("policy_version") != self.gate.policy.version
                or decision.get("policy_hash") != self.gate.policy.hash
                or decision.get("provider") != PROCESSING_PROVIDER
                or decision.get("model") != PROCESSING_MODEL
                or decision.get("purpose") != "internal_analysis"
                or decision.get("outcome") != paper["input_scope"]
                or decision.get("data_category") not in allowed_categories[str(paper["input_scope"])]
                or decision.get("authorized_by") not in {"policy", "grant"}
                or analysis_document["artifact_hash"] != decision.get("input_artifact_hash")
                or row["policy_version"] != decision.get("policy_version")
                or row["policy_decision"] != decision.get("outcome")
                or row["authorization_grant_id"] != decision.get("processing_grant_id")
                or analysis_document["prompt_hash"] != row["prompt_hash"]
                or analysis_document["schema_hash"] != row["schema_hash"]
                or analysis_document["model_revision"] != row["model_revision"]
                or row["prompt_hash"] != self.stage4_prompt_hash
                or row["schema_hash"] != self.stage4_schema_hash
            ):
                raise ReportAuditError("Stage 4 processing decision is missing or foreign")
            paper_lineages = {str(value) for value in paper["lineage_hashes"]}
            if not paper_lineages or any(not _is_hash(value) for value in paper_lineages):
                raise ReportAuditError("corpus paper lacks frozen source lineage")
            lineages.update(paper_lineages)
            license_value, access_basis, source_hash = self._source_policy_facts(
                row["artifact_id"], str(paper["paper_id"])
            )
            if (
                source_hash is not None
                and (
                    source_hash != decision.get("input_artifact_hash")
                    or source_hash not in paper_lineages
                )
            ):
                raise ReportAuditError("corpus lineage does not bind its Stage 4 source artifact")
            scopes.append(str(row["input_scope"]))
            licenses.add(license_value)
            access.add(access_basis)
        if not scopes:
            raise ReportAuditError("audit requires a non-empty frozen corpus")
        rank = {"metadata_only": 0, "abstract_only": 1, "full_pdf": 2}
        input_scope = max(scopes, key=rank.__getitem__)
        license_value = next(iter(licenses)) if len(licenses) == 1 else None
        if access == {"open_license"}:
            access_basis = "open_license"
        elif access.issubset({"open_license", "public_read_only"}):
            access_basis = "public_read_only"
        elif "user_subscription" in access:
            access_basis = "user_subscription"
        elif "user_supplied" in access:
            access_basis = "user_supplied"
        else:
            access_basis = "unknown"
        return _PolicyFacts(
            input_scope=input_scope,
            license=license_value,
            access_basis=access_basis,
            lineage_hash=content_hash(sorted(lineages)),
            execution_mode=self.execution_mode,
        )

    def _source_policy_facts(
        self, artifact_id: str | None, paper_id: str
    ) -> tuple[str | None, str, str | None]:
        if artifact_id is None:
            return None, "unknown", None
        analyzed = self.database.connection.execute(
            """SELECT artifact_id, paper_id, artifact_kind, relative_path, mime_type,
                      byte_size, sha256, provenance_json, processing_status
               FROM artifacts WHERE artifact_id = ?""",
            (artifact_id,),
        ).fetchone()
        if analyzed is None or analyzed["paper_id"] != paper_id or analyzed["processing_status"] != "available":
            raise ReportAuditError("Stage 4 source artifact binding has drifted")
        analyzed_hash = str(analyzed["sha256"])
        analyzed_payload = self.artifact_store.read_bytes(analyzed_hash)
        if (
            analyzed["relative_path"] != self.artifact_store.relative_path(analyzed_hash)
            or int(analyzed["byte_size"]) != len(analyzed_payload)
        ):
            raise ReportAuditError("Stage 4 source artifact metadata has drifted")
        pdf = analyzed
        if analyzed["artifact_kind"] == "text":
            if not str(analyzed["mime_type"]).startswith("text/plain"):
                raise ReportAuditError("normalized Stage 4 input has an invalid MIME type")
            extractions = self.database.connection.execute(
                """SELECT te.source_artifact_id, te.source_sha256,
                          source.artifact_id, source.paper_id, source.artifact_kind,
                          source.relative_path, source.mime_type, source.byte_size,
                          source.sha256, source.provenance_json, source.processing_status
                   FROM text_extractions te
                   JOIN artifacts source ON source.artifact_id = te.source_artifact_id
                   WHERE te.output_artifact_id = ? AND te.paper_id = ?
                     AND te.status = 'full_text_ready'""",
                (artifact_id, paper_id),
            ).fetchall()
            bindings = {
                tuple(row[key] for key in (
                    "source_artifact_id",
                    "source_sha256",
                    "artifact_id",
                    "paper_id",
                    "artifact_kind",
                    "relative_path",
                    "mime_type",
                    "byte_size",
                    "sha256",
                    "provenance_json",
                    "processing_status",
                ))
                for row in extractions
            }
            if len(bindings) != 1:
                return None, "unknown", analyzed_hash
            pdf = extractions[0]
            if pdf["source_artifact_id"] != pdf["artifact_id"]:
                raise ReportAuditError("normalized text source PDF binding has drifted")
        elif analyzed["artifact_kind"] != "pdf":
            return None, "unknown", analyzed_hash

        if (
            pdf["paper_id"] != paper_id
            or pdf["artifact_kind"] != "pdf"
            or pdf["mime_type"] != "application/pdf"
            or pdf["processing_status"] != "available"
            or (
                analyzed["artifact_kind"] == "text"
                and pdf["source_sha256"] != pdf["sha256"]
            )
        ):
            raise ReportAuditError("Stage 4 source PDF binding has drifted")
        pdf_hash = str(pdf["sha256"])
        pdf_payload = self.artifact_store.read_bytes(pdf_hash)
        if (
            pdf["relative_path"] != self.artifact_store.relative_path(pdf_hash)
            or int(pdf["byte_size"]) != len(pdf_payload)
        ):
            raise ReportAuditError("Stage 4 source PDF metadata has drifted")
        provenance = _json_mapping(pdf["provenance_json"], "Stage 4 source PDF provenance")
        provenance_candidate_id = provenance.get("candidate_id")
        if not isinstance(provenance_candidate_id, str) or not provenance_candidate_id:
            return None, "unknown", analyzed_hash
        attempts = self.database.connection.execute(
            """SELECT dc.candidate_id, dc.paper_id, dc.license, dc.access_basis
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
                str(row["paper_id"]),
                row["license"],
                str(row["access_basis"]),
            )
            for row in attempts
        }
        if len(facts) != 1:
            return None, "unknown", analyzed_hash
        candidate_id, authorized_paper, license_value, access_basis = next(iter(facts))
        if authorized_paper != paper_id or candidate_id != provenance_candidate_id:
            return None, "unknown", analyzed_hash
        return license_value, access_basis, analyzed_hash

    def _audit_budget(
        self,
        report_run_id: str,
        plan: Mapping[str, Any],
        bundle: Mapping[str, Any],
        audit_plan: _AuditPassPlan,
    ) -> Mapping[str, int]:
        row = self.database.connection.execute(
            "SELECT aggregation_tree_json FROM report_runs WHERE report_run_id = ?",
            (report_run_id,),
        ).fetchone()
        tree = _json_mapping(row["aggregation_tree_json"], "aggregation tree")
        if tree.get("execution_mode") != self.execution_mode:
            raise ReportAuditBudgetError(
                "audit execution mode differs from the frozen reduce run"
            )
        declared = tree.get("budget")
        if not isinstance(declared, Mapping):
            raise ReportAuditBudgetError("aggregation tree lacks its audit/repair call reserve")
        if int(declared.get("audit_calls", 0)) != 2 or int(declared.get("repair_calls", 0)) != 1:
            raise ReportAuditBudgetError("report plan must reserve two audits and exactly one repair")
        if int(plan["budget"]["max_retries"]) != MAX_RETRIES:
            raise ReportAuditBudgetError("approved retry budget differs from the frozen Sol profile")
        final = self.database.connection.execute(
            """SELECT dependency_ids_json, output_byte_limit
               FROM report_reduce_nodes
               WHERE report_run_id = ? AND call_kind = 'final_reduce'""",
            (report_run_id,),
        ).fetchall()
        if len(final) != 1:
            raise ReportAuditBudgetError("audit budget requires one final_reduce node")
        dependencies = json.loads(final[0]["dependency_ids_json"])
        if not isinstance(dependencies, list) or len(dependencies) != 1:
            raise ReportAuditBudgetError("final_reduce audit budget requires one synthesis dependency")
        synthesis = self.database.connection.execute(
            """SELECT output_byte_limit FROM report_reduce_nodes
               WHERE report_run_id = ? AND node_id = ?""",
            (report_run_id, dependencies[0]),
        ).fetchone()
        if synthesis is None:
            raise ReportAuditBudgetError("audit budget synthesis dependency is missing")
        bounds = stage4b_audit_repair_budget_bounds(
            plan,
            bundle["corpus_snapshot"],
            bundle["search_audit"],
            final_output_byte_limit=int(final[0]["output_byte_limit"]),
            synthesis_output_byte_limit=int(synthesis["output_byte_limit"]),
            rubric_path=self.rubric_path,
        )
        if canonical_json(tree.get("audit_repair_budget_bounds")) != canonical_json(
            asdict(bounds)
        ):
            raise ReportAuditBudgetError(
                "frozen reduce run has a different audit/repair budget bound"
            )
        if audit_plan.worst_case_calls > bounds.audit_calls_per_pass:
            raise ReportAuditBudgetError(
                "actual stable audit shard tree exceeds the shared preflight call bound"
            )
        if audit_plan.direct_payload is not None:
            actual_a_tokens = _token_upper_bound(self._rendered_prompt(
                "quality_audit",
                canonical_json(audit_plan.direct_payload).decode("utf-8"),
            ))
        else:
            actual_a_tokens = sum(
                _token_upper_bound(self._rendered_prompt(
                    "quality_audit", canonical_json(item.payload).decode("utf-8")
                ))
                for item in audit_plan.shards
            ) + max(1, len(audit_plan.shards) - 1) * AUDIT_MAX_INPUT_TOKENS
        if actual_a_tokens > bounds.audit_a_input_tokens:
            raise ReportAuditBudgetError(
                "actual exhaustive audit plan exceeds the shared preflight token bound"
            )
        used = self.database.connection.execute(
            """SELECT COALESCE(SUM(budget_calls_reserved), 0),
                      COALESCE(SUM(budget_tokens_reserved), 0)
               FROM report_reduce_nodes WHERE report_run_id = ?""",
            (report_run_id,),
        ).fetchone()
        if int(used[0]) + bounds.worst_case_calls > int(plan["budget"]["max_sol_calls"]):
            raise ReportAuditBudgetError("audit/repair worst-case calls exceed the approved reserve")
        if int(used[1]) + bounds.worst_case_input_tokens > int(plan["budget"]["max_input_tokens"]):
            raise ReportAuditBudgetError("audit/repair worst-case tokens exceed the approved reserve")
        return {
            "audit_a": min(bounds.audit_a_input_tokens, AUDIT_MAX_INPUT_TOKENS),
            "audit_c": min(bounds.audit_c_input_tokens, AUDIT_MAX_INPUT_TOKENS),
            "repair": bounds.repair_input_tokens,
            "audit_calls_per_pass": bounds.audit_calls_per_pass,
            "worst_calls": bounds.worst_case_calls,
            "worst_tokens": bounds.worst_case_input_tokens,
        }

    def _synthesis_output_byte_limit(self, report_run_id: str) -> int:
        row = self.database.connection.execute(
            """SELECT dependency_ids_json
               FROM report_reduce_nodes
               WHERE report_run_id = ? AND call_kind = 'final_reduce'""",
            (report_run_id,),
        ).fetchall()
        if len(row) != 1:
            raise ReportAuditError("report run requires one final synthesis binding")
        dependencies = json.loads(row[0]["dependency_ids_json"])
        if not isinstance(dependencies, list) or len(dependencies) != 1:
            raise ReportAuditError("final report requires one synthesis dependency")
        synthesis = self.database.connection.execute(
            """SELECT output_byte_limit FROM report_reduce_nodes
               WHERE report_run_id = ? AND node_id = ?""",
            (report_run_id, dependencies[0]),
        ).fetchone()
        if synthesis is None or int(synthesis["output_byte_limit"]) < 1:
            raise ReportAuditError("synthesis output bound is missing")
        return int(synthesis["output_byte_limit"])

    def _ensure_run(
        self,
        report_run_id: str,
        snapshot_hash: str,
        base_hash: str,
        bundle: Mapping[str, Any],
        budget: Mapping[str, int],
    ) -> None:
        current_json = _json_text(_mutable_part(bundle))
        expected = (
            snapshot_hash,
            base_hash,
            base_hash,
            current_json,
            self.rubric_hash,
            PROFILE,
            MODEL,
            REASONING_EFFORT,
            self.config_hash,
            self.execution_mode,
            budget["worst_calls"],
            budget["worst_tokens"],
        )
        with self.database.transaction() as connection:
            row = connection.execute(
                """SELECT input_snapshot_hash, base_artifact_hash, current_artifact_hash,
                          current_bundle_json, rubric_hash, profile, model_id, reasoning_effort,
                          config_hash, execution_mode, worst_case_calls,
                          worst_case_input_tokens
                   FROM report_audit_runs WHERE report_run_id = ?""",
                (report_run_id,),
            ).fetchone()
            if row is None:
                connection.execute(
                    """INSERT INTO report_audit_runs(
                           report_run_id, input_snapshot_hash, base_artifact_hash,
                           current_artifact_hash, current_bundle_json, rubric_hash,
                           profile, model_id, reasoning_effort, config_hash,
                           execution_mode, worst_case_calls, worst_case_input_tokens, status
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')""",
                    (report_run_id, *expected),
                )
            elif tuple(row) != expected:
                # A repaired run intentionally has a different current bundle.
                if tuple(row)[0:2] != expected[0:2] or tuple(row)[4:] != expected[4:]:
                    raise ReportAuditError("persisted report audit input or configuration has drifted")

    def _run_row(self, report_run_id: str) -> Any:
        row = self.database.connection.execute(
            "SELECT * FROM report_audit_runs WHERE report_run_id = ?", (report_run_id,)
        ).fetchone()
        if row is None:
            raise ReportAuditError("report audit run was not persisted")
        return row

    def _load_current_bundle(
        self, row: Any, initial: Mapping[str, Any]
    ) -> dict[str, Any]:
        mutable = _json_mapping(row["current_bundle_json"], "current report bundle")
        current = dict(initial)
        current.update(deepcopy(dict(mutable)))
        if _bundle_hash(current) != row["current_artifact_hash"]:
            raise ReportAuditError("persisted current report bundle hash has drifted")
        self._deterministic_verify(current)
        return current

    def _input_snapshot_hash(
        self, bundle: Mapping[str, Any], previous: Mapping[str, Any] | None
    ) -> str:
        return content_hash({
            **bundle,
            "previous": previous,
            "rubric_hash": self.rubric_hash,
            "config_hash": self.config_hash,
        })

    def _audit_payload(
        self,
        report_run_id: str,
        audit_pass: str,
        bundle: Mapping[str, Any],
        verification: Mapping[str, Any],
        coverage: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "report_run_id": report_run_id,
            "audit_pass": audit_pass,
            "report_document_hash": content_hash(bundle["document"]),
            "report_artifact_hash": _bundle_hash(bundle),
            "report_plan_hash": bundle["plan"]["plan_hash"],
            "rubric": self.rubric,
            "rubric_hash": self.rubric_hash,
            "search_limitations": list(_search_limitations(bundle)),
            "search_limitations_hash": content_hash(list(_search_limitations(bundle))),
            "expected_coverage_ledger": coverage,
            "deterministic_verification": dict(verification),
            "report_plan": bundle["plan"],
            "corpus_snapshot": bundle["corpus_snapshot"],
            "report_document": bundle["document"],
            "claims_evidence": bundle["claims"],
            "coverage": bundle["coverage"],
            "comparison_groups": bundle["comparison_groups"],
            "claim_relations": bundle["claim_relations"],
            "bibliography": bundle["bibliography"],
        }

    def _audit_pass_plan(
        self,
        report_run_id: str,
        audit_pass: str,
        bundle: Mapping[str, Any],
        verification: Mapping[str, Any],
        coverage: Mapping[str, Any],
        direct_payload: Mapping[str, Any],
    ) -> _AuditPassPlan:
        direct_prompt = canonical_json(direct_payload).decode("utf-8")
        if _token_upper_bound(
            self._rendered_prompt("quality_audit", direct_prompt)
        ) <= AUDIT_MAX_INPUT_TOKENS:
            return _AuditPassPlan(audit_pass, coverage, direct_payload, ())

        components = self._audit_components(bundle)
        shards: list[_AuditShardSpec] = []
        current: list[tuple[tuple[str, ...], tuple[str, ...]]] = []

        def build(
            values: Sequence[tuple[tuple[str, ...], tuple[str, ...]]]
        ) -> _AuditShardSpec:
            block_ids = tuple(sorted({value for item, _ in values for value in item}))
            claim_ids = tuple(sorted({value for _, item in values for value in item}))
            shard_coverage = _coverage_subset(coverage, block_ids, claim_ids)
            node_id = "audit-shard-" + content_hash({
                "report_run_id": report_run_id,
                "audit_pass": audit_pass,
                "coverage": shard_coverage,
            })[:32]
            payload = self._audit_shard_payload(
                report_run_id,
                audit_pass,
                node_id,
                bundle,
                verification,
                shard_coverage,
                block_ids,
                claim_ids,
            )
            paper_ids = tuple(sorted(
                str(item["paper_id"])
                for item in payload["corpus_snapshot"]["papers"]
            ))
            return _AuditShardSpec(
                node_id, "shard", payload, shard_coverage, (), paper_ids
            )

        for component in components:
            candidate = [*current, component]
            spec = build(candidate)
            prompt = canonical_json(spec.payload).decode("utf-8")
            if _token_upper_bound(
                self._rendered_prompt("quality_audit", prompt)
            ) <= AUDIT_MAX_INPUT_TOKENS:
                current = candidate
                continue
            if not current:
                raise ReportAuditBudgetError(
                    "one stable section/claim audit component exceeds the frozen Sol context"
                )
            shards.append(build(current))
            current = [component]
            spec = build(current)
            prompt = canonical_json(spec.payload).decode("utf-8")
            if _token_upper_bound(
                self._rendered_prompt("quality_audit", prompt)
            ) > AUDIT_MAX_INPUT_TOKENS:
                raise ReportAuditBudgetError(
                    "one stable section/claim audit component exceeds the frozen Sol context"
                )
        if current:
            shards.append(build(current))
        if not shards:
            raise ReportAuditError("exhaustive audit sharding produced no shards")
        merged = _coverage_union(coverage, [item.coverage for item in shards])
        if canonical_json(merged) != canonical_json(coverage):
            raise ReportAuditError("stable audit shards do not exactly cover the full ledger")
        return _AuditPassPlan(audit_pass, coverage, None, tuple(shards))

    def _audit_components(
        self, bundle: Mapping[str, Any]
    ) -> tuple[tuple[tuple[str, ...], tuple[str, ...]], ...]:
        blocks = {
            str(item["block_id"]): item for item in bundle["document"]["blocks"]
        }
        claims = {str(item["claim_id"]): item for item in bundle["claims"]}
        section_order = {
            str(item["id"]): index
            for index, item in enumerate(bundle["plan"]["sections"])
        }
        ordered_blocks = sorted(
            blocks,
            key=lambda block_id: (
                section_order[str(blocks[block_id]["section_id"])],
                block_id,
            ),
        )
        claim_blocks: dict[str, set[str]] = {claim_id: set() for claim_id in claims}
        for block_id, block in blocks.items():
            for claim_id in block["claim_ids"]:
                claim_blocks[str(claim_id)].add(block_id)
        if any(not values for values in claim_blocks.values()):
            raise ReportAuditError("audit sharding found a claim outside every substantive block")

        remaining = set(blocks)
        components: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
        for first in ordered_blocks:
            if first not in remaining:
                continue
            section_id = str(blocks[first]["section_id"])
            pending = [first]
            component_blocks: set[str] = set()
            component_claims: set[str] = set()
            while pending:
                block_id = pending.pop()
                if block_id in component_blocks:
                    continue
                if str(blocks[block_id]["section_id"]) != section_id:
                    raise ReportAuditError(
                        "one claim spans ReportPlan sections and cannot be stably audited"
                    )
                component_blocks.add(block_id)
                remaining.discard(block_id)
                for claim_id_value in blocks[block_id]["claim_ids"]:
                    claim_id = str(claim_id_value)
                    component_claims.add(claim_id)
                    pending.extend(sorted(claim_blocks[claim_id] - component_blocks))
            components.append((
                tuple(sorted(component_blocks)),
                tuple(sorted(component_claims)),
            ))
        if remaining or {value for _, values in components for value in values} != set(claims):
            raise ReportAuditError("stable audit components omit report blocks or claims")
        return tuple(components)

    def _audit_shard_payload(
        self,
        report_run_id: str,
        audit_pass: str,
        node_id: str,
        bundle: Mapping[str, Any],
        verification: Mapping[str, Any],
        coverage: Mapping[str, Any],
        block_ids: Sequence[str],
        claim_ids: Sequence[str],
    ) -> dict[str, Any]:
        block_set = set(block_ids)
        claim_set = set(claim_ids)
        blocks = [
            item for item in bundle["document"]["blocks"]
            if str(item["block_id"]) in block_set
        ]
        claims = [
            item for item in bundle["claims"]
            if str(item["claim_id"]) in claim_set
        ]
        paper_ids = {
            str(value) for block in blocks for value in block["citation_paper_ids"]
        }
        for claim in claims:
            for field in ("supporting_evidence", "contradicting_evidence"):
                for reference in claim[field]:
                    if reference["kind"] == "paper_evidence":
                        paper_ids.add(str(reference["paper_id"]))
        comparison_ids = {
            str(value)
            for claim in claims
            for value in (claim["claim_key"].get("comparison_group_id"),)
            if value is not None
        }
        return {
            "report_run_id": report_run_id,
            "audit_pass": audit_pass,
            "audit_scope": {
                "kind": "stable_section_claim_shard",
                "node_id": node_id,
                "section_ids": sorted({str(item["section_id"]) for item in blocks}),
            },
            "instructions": (
                "Audit every supplied block, claim, and evidence reference. The coverage "
                "ledger is exact; do not sample. Prefix every finding_id with the supplied "
                "node_id followed by a colon. Bind the output to the full report hashes."
            ),
            "report_document_hash": content_hash(bundle["document"]),
            "report_artifact_hash": _bundle_hash(bundle),
            "report_plan_hash": bundle["plan"]["plan_hash"],
            "rubric": self.rubric,
            "rubric_hash": self.rubric_hash,
            "search_limitations": list(_search_limitations(bundle)),
            "search_limitations_hash": content_hash(list(_search_limitations(bundle))),
            "expected_coverage_ledger": coverage,
            "deterministic_verification": dict(verification),
            "report_plan": bundle["plan"],
            "corpus_snapshot": {
                "snapshot_hash": bundle["corpus_snapshot"]["snapshot_hash"],
                "input_scope": bundle["corpus_snapshot"].get("input_scope", {}),
                "papers": [
                    item for item in bundle["corpus_snapshot"]["papers"]
                    if str(item["paper_id"]) in paper_ids
                ],
            },
            "report_document": {
                "report_run_id": bundle["document"]["report_run_id"],
                "blocks": blocks,
            },
            "claims_evidence": claims,
            "coverage": {
                **bundle["coverage"],
                "papers": [
                    item for item in bundle["coverage"]["papers"]
                    if str(item["paper_id"]) in paper_ids
                ],
            },
            "comparison_groups": {
                key: value for key, value in bundle["comparison_groups"].items()
                if str(key) in comparison_ids
            },
            "claim_relations": [
                item for item in bundle["claim_relations"]
                if str(item["current_claim_id"]) in claim_set
            ],
            "bibliography": {
                key: value for key, value in bundle["bibliography"].items()
                if str(key) in paper_ids
            },
        }

    def _audit_reduce_payload(
        self,
        report_run_id: str,
        audit_pass: str,
        node_id: str,
        bundle: Mapping[str, Any],
        coverage: Mapping[str, Any],
        sources: Sequence[tuple[str, Mapping[str, Any]]],
    ) -> dict[str, Any]:
        return {
            "report_run_id": report_run_id,
            "audit_pass": audit_pass,
            "audit_scope": {
                "kind": "exhaustive_audit_reduce",
                "node_id": node_id,
                "source_node_ids": [source_id for source_id, _ in sources],
            },
            "instructions": (
                "Merge every source audit without sampling, summarizing away, changing, or "
                "dropping any finding. coverage_complete is true only if every source is true. "
                "Return the exact expected coverage ledger and the exact union of source findings."
            ),
            "report_document_hash": content_hash(bundle["document"]),
            "report_artifact_hash": _bundle_hash(bundle),
            "report_plan_hash": bundle["plan"]["plan_hash"],
            "rubric_hash": self.rubric_hash,
            "search_limitations_hash": content_hash(list(_search_limitations(bundle))),
            "expected_coverage_ledger": coverage,
            "source_audits": [
                {
                    "node_id": source_id,
                    "output_hash": content_hash(output),
                    "audit": output,
                }
                for source_id, output in sources
            ],
        }

    def _run_audit_pass(
        self,
        report_run_id: str,
        step_name: str,
        plan: _AuditPassPlan,
        bundle: Mapping[str, Any],
        policy_facts: _PolicyFacts,
        budget: Mapping[str, int],
        grants: Mapping[str, str],
        owner: str,
        *,
        forbidden_invocation_ids: frozenset[str],
    ) -> _StepResult:
        if plan.worst_case_calls > budget["audit_calls_per_pass"]:
            raise ReportAuditBudgetError(
                f"audit {plan.audit_pass} shard tree exceeds its frozen preflight reserve"
            )
        if plan.direct_payload is not None:
            unexpected = self.database.connection.execute(
                """SELECT 1 FROM report_audit_shard_steps
                   WHERE report_run_id = ? AND audit_pass = ? LIMIT 1""",
                (report_run_id, plan.audit_pass),
            ).fetchone()
            if unexpected is not None:
                raise ReportAuditError(
                    "direct audit pass has an unexpected persisted shard ledger"
                )
            return self._run_step(
                report_run_id,
                step_name,
                "quality_audit",
                bundle,
                plan.direct_payload,
                policy_facts,
                min(budget[step_name], AUDIT_MAX_INPUT_TOKENS),
                grants,
                self._now(),
                owner,
                plan.full_coverage,
                paper_count=len({
                    str(item["paper_id"])
                    for item in bundle["corpus_snapshot"]["papers"]
                }),
            )

        nodes: list[
            tuple[
                str,
                Mapping[str, Any],
                Mapping[str, Any],
                InvocationMetadata,
                tuple[str, ...],
            ]
        ] = []
        resumed = False
        invocation_ids = set(forbidden_invocation_ids)
        expected_aux_ids = {item.node_id for item in plan.shards}
        for ordinal, spec in enumerate(plan.shards):
            result = self._run_aux_audit_step(
                report_run_id,
                plan.audit_pass,
                spec,
                bundle,
                policy_facts,
                grants,
                self._now(),
                owner,
                ordinal,
            )
            if result.status != "complete":
                return result
            assert result.output is not None and result.metadata is not None
            if result.metadata.invocation_id in invocation_ids:
                return _StepResult("failed", error="audit shard reused a prior Sol invocation")
            invocation_ids.add(result.metadata.invocation_id)
            resumed = resumed or result.resumed
            nodes.append((
                spec.node_id,
                result.output,
                spec.coverage,
                result.metadata,
                spec.paper_ids,
            ))

        level = 0
        ordinal = len(nodes)
        while len(nodes) > 2:
            next_level: list[
                tuple[
                    str,
                    Mapping[str, Any],
                    Mapping[str, Any],
                    InvocationMetadata,
                    tuple[str, ...],
                ]
            ] = []
            for index in range(0, len(nodes), 2):
                group = nodes[index:index + 2]
                if len(group) == 1:
                    next_level.append(group[0])
                    continue
                source_ids = tuple(item[0] for item in group)
                coverage = _coverage_union(
                    plan.full_coverage, [item[2] for item in group]
                )
                node_id = "audit-reduce-" + content_hash({
                    "report_run_id": report_run_id,
                    "audit_pass": plan.audit_pass,
                    "level": level,
                    "sources": source_ids,
                })[:32]
                payload = self._audit_reduce_payload(
                    report_run_id,
                    plan.audit_pass,
                    node_id,
                    bundle,
                    coverage,
                    [(item[0], item[1]) for item in group],
                )
                if _token_upper_bound(self._rendered_prompt(
                    "quality_audit", canonical_json(payload).decode("utf-8")
                )) > AUDIT_MAX_INPUT_TOKENS:
                    return _StepResult(
                        "failed",
                        error=(
                            "exhaustive audit reduce exceeds the frozen hard context; "
                            "no source finding can be sampled"
                        ),
                    )
                spec = _AuditShardSpec(
                    node_id,
                    "audit_reduce",
                    payload,
                    coverage,
                    source_ids,
                    tuple(sorted({
                        paper_id for item in group for paper_id in item[4]
                    })),
                )
                expected_aux_ids.add(node_id)
                result = self._run_aux_audit_step(
                    report_run_id,
                    plan.audit_pass,
                    spec,
                    bundle,
                    policy_facts,
                    grants,
                    self._now(),
                    owner,
                    ordinal,
                )
                ordinal += 1
                if result.status != "complete":
                    return result
                assert result.output is not None and result.metadata is not None
                if result.metadata.invocation_id in invocation_ids:
                    return _StepResult("failed", error="audit reduce reused a prior Sol invocation")
                invocation_ids.add(result.metadata.invocation_id)
                resumed = resumed or result.resumed
                next_level.append((
                    node_id,
                    result.output,
                    coverage,
                    result.metadata,
                    spec.paper_ids,
                ))
            nodes = next_level
            level += 1

        root_id = "audit-root-" + content_hash({
            "report_run_id": report_run_id,
            "audit_pass": plan.audit_pass,
            "sources": [item[0] for item in nodes],
        })[:32]
        root_payload = self._audit_reduce_payload(
            report_run_id,
            plan.audit_pass,
            root_id,
            bundle,
            plan.full_coverage,
            [(item[0], item[1]) for item in nodes],
        )
        if _token_upper_bound(self._rendered_prompt(
            "quality_audit", canonical_json(root_payload).decode("utf-8")
        )) > AUDIT_MAX_INPUT_TOKENS:
            return _StepResult(
                "failed",
                error=(
                    "final exhaustive audit reduce exceeds the frozen hard context; "
                    "the report remains unpublished"
                ),
            )
        result = self._run_step(
            report_run_id,
            step_name,
            "quality_audit",
            bundle,
            root_payload,
            policy_facts,
            AUDIT_MAX_INPUT_TOKENS,
            grants,
            self._now(),
            owner,
            plan.full_coverage,
            paper_count=len({
                str(item["paper_id"])
                for item in bundle["corpus_snapshot"]["papers"]
            }),
        )
        if result.status != "complete":
            return result
        assert result.metadata is not None
        if result.metadata.invocation_id in invocation_ids:
            return _StepResult("failed", error="final audit reduce reused a prior Sol invocation")
        persisted_aux = self.database.connection.execute(
            """SELECT node_id, status FROM report_audit_shard_steps
               WHERE report_run_id = ? AND audit_pass = ?""",
            (report_run_id, plan.audit_pass),
        ).fetchall()
        if {str(row["node_id"]): str(row["status"]) for row in persisted_aux} != {
            node_id: "complete" for node_id in expected_aux_ids
        }:
            raise ReportAuditError("audit pass has an unexpected shard/reduce ledger")
        return _StepResult(
            result.status,
            result.output,
            result.metadata,
            resumed or result.resumed,
            result.error,
        )

    def _repair_payload(
        self,
        report_run_id: str,
        bundle: Mapping[str, Any],
        audit: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "report_run_id": report_run_id,
            "base_artifact_hash": _bundle_hash(bundle),
            "report_document_hash": content_hash(bundle["document"]),
            "blocker_major_findings": _severe_findings(audit),
            "allowed_targets": {
                "REPORT_DOCUMENT": "replace_block by stable block_id",
                "CLAIMS_EVIDENCE": "replace_claim by stable claim_id without new evidence",
                "COVERAGE": (
                    "replace_paper by stable paper_id; the value must preserve the exact "
                    "deterministic coverage fact"
                ),
            },
            "report_plan": bundle["plan"],
            "corpus_snapshot": bundle["corpus_snapshot"],
            "report_document": bundle["document"],
            "claims_evidence": bundle["claims"],
            "coverage": bundle["coverage"],
            "comparison_groups": bundle["comparison_groups"],
        }

    def _run_step(
        self,
        report_run_id: str,
        step_name: str,
        call_kind: str,
        bundle: Mapping[str, Any],
        payload: Mapping[str, Any],
        policy_facts: _PolicyFacts,
        token_limit: int,
        grants: Mapping[str, str],
        moment: datetime,
        owner: str,
        expected_coverage: Mapping[str, Any] | None,
        *,
        paper_count: int = 1,
    ) -> _StepResult:
        prompt = canonical_json(payload).decode("utf-8")
        input_hash = sha256(prompt.encode("utf-8")).hexdigest()
        rendered = self._rendered_prompt(call_kind, prompt)
        rendered_hash = sha256(rendered.encode("utf-8")).hexdigest()
        input_tokens = _token_upper_bound(rendered)
        hard_input_limit = (
            AUDIT_MAX_INPUT_TOKENS
            if call_kind == "quality_audit"
            else REPAIR_MAX_INPUT_TOKENS
        )
        if token_limit > hard_input_limit or input_tokens > token_limit:
            raise ReportAuditBudgetError(
                f"{step_name} rendered prompt exceeds its frozen input-token reserve"
            )
        coverage_hash = content_hash(expected_coverage) if expected_coverage is not None else None
        self._ensure_step(
            report_run_id,
            step_name,
            call_kind,
            _bundle_hash(bundle),
            input_hash,
            coverage_hash,
            token_limit,
            policy_facts,
        )
        self._expire_stale_step(report_run_id, step_name, moment)
        row = self._step_row(report_run_id, step_name)
        if row["status"] == "complete":
            return self._load_completed_step(
                row,
                call_kind,
                prompt,
                rendered_hash,
                expected_coverage,
                bundle,
                moment,
                paper_count,
            )
        if row["status"] in {"running", "failed"}:
            return _StepResult(row["status"], error=_error_message(row["error_json"]))

        request = ProcessingRequest(
            artifact_hash=input_hash,
            artifact="report_draft",
            input_scope=policy_facts.input_scope,
            license=policy_facts.license,
            access_basis=policy_facts.access_basis,
            purpose=PURPOSE,
            data_category="report_draft",
            provider=PROCESSING_PROVIDER,
            model=MODEL,
            source_paper_ids=tuple(sorted(
                str(item["paper_id"])
                for item in bundle["corpus_snapshot"]["papers"]
            )),
            mode=policy_facts.execution_mode,
            lineage_hash=policy_facts.lineage_hash,
            derived_bytes=prompt.encode("utf-8"),
        )
        dispatched = self.gate.dispatch(
            request,
            lambda invocation: invocation,
            processing_grant_id=grants.get(input_hash),
            now=_timestamp(moment),
            paper_count=paper_count,
        )
        if not dispatched.decision.is_authorized:
            self._persist_manual(report_run_id, step_name, dispatched.decision)
            return _StepResult("manual_required", error=dispatched.decision.reason_code)
        invocation = dispatched.result
        if not isinstance(invocation, ModelInvocation) or invocation.derived_bytes != prompt.encode("utf-8"):
            raise ReportAuditError("processing gate exposed a payload outside its exact decision")

        lease_token = self._claim_step(
            report_run_id,
            step_name,
            input_hash,
            rendered_hash,
            input_tokens,
            dispatched.decision,
            owner,
            moment,
        )
        if lease_token is None:
            return _StepResult("running")
        try:
            request_value = CodexExecRequest(
                profile=PROFILE,
                prompt=prompt,
                output_schema=self.schemas[call_kind],
                schema_name=CALL_KIND_SCHEMAS[call_kind],
                prompt_name=CALL_KIND_PROMPTS[call_kind],
                input_hash=input_hash,
                call_kind=call_kind,
                schema_path=self.resources.schema_path(call_kind),
                prompt_path=self.resources.prompt_path(call_kind),
                expected_prompt_hash=self.prompt_hashes[call_kind],
                schema_resource_paths=self.resources.configured_schema_resources(),
                expected_service_schema_hash=self.service_schema_hashes[call_kind],
            )
            result = self.invoker_factory().invoke(request_value)
            self._validate_metadata(result.metadata, call_kind, input_hash, rendered_hash)
            self._validate_step_output(
                call_kind, result.output, expected_coverage, bundle, payload
            )
            output_limit = (
                AUDIT_OUTPUT_BYTE_LIMIT
                if call_kind == "quality_audit"
                else MAX_OUTPUT_BYTES
            )
            if len(canonical_json(result.output)) > output_limit:
                raise ReportAuditOutputError("Sol audit/repair output exceeds the frozen byte limit")
            self._persist_complete_step(
                report_run_id,
                step_name,
                result,
                owner,
                lease_token,
            )
            return _StepResult("complete", dict(result.output), result.metadata)
        except Exception as error:
            self._persist_failed_step(
                report_run_id, step_name, error, owner, lease_token
            )
            return _StepResult("failed", error=str(error))

    def _run_aux_audit_step(
        self,
        report_run_id: str,
        audit_pass: str,
        spec: _AuditShardSpec,
        bundle: Mapping[str, Any],
        policy_facts: _PolicyFacts,
        grants: Mapping[str, str],
        moment: datetime,
        owner: str,
        ordinal: int,
    ) -> _StepResult:
        prompt = canonical_json(spec.payload).decode("utf-8")
        input_hash = sha256(prompt.encode("utf-8")).hexdigest()
        rendered = self._rendered_prompt("quality_audit", prompt)
        rendered_hash = sha256(rendered.encode("utf-8")).hexdigest()
        input_tokens = _token_upper_bound(rendered)
        if input_tokens > AUDIT_MAX_INPUT_TOKENS:
            raise ReportAuditBudgetError(
                f"{spec.node_id} exceeds the frozen Sol audit hard context"
            )
        self._ensure_aux_audit_step(
            report_run_id,
            audit_pass,
            spec,
            ordinal,
            _bundle_hash(bundle),
            input_hash,
            policy_facts,
        )
        self._expire_stale_aux_audit_step(report_run_id, audit_pass, spec.node_id, moment)
        row = self._aux_audit_step_row(report_run_id, audit_pass, spec.node_id)
        if row["status"] == "complete":
            return self._load_completed_aux_audit_step(
                row,
                spec,
                prompt,
                rendered_hash,
                bundle,
                moment,
            )
        if row["status"] in {"running", "failed"}:
            return _StepResult(row["status"], error=_error_message(row["error_json"]))

        request = ProcessingRequest(
            artifact_hash=input_hash,
            artifact="report_draft",
            input_scope=policy_facts.input_scope,
            license=policy_facts.license,
            access_basis=policy_facts.access_basis,
            purpose=PURPOSE,
            data_category="report_draft",
            provider=PROCESSING_PROVIDER,
            model=MODEL,
            source_paper_ids=spec.paper_ids,
            mode=policy_facts.execution_mode,
            lineage_hash=policy_facts.lineage_hash,
            derived_bytes=prompt.encode("utf-8"),
        )
        dispatched = self.gate.dispatch(
            request,
            lambda invocation: invocation,
            processing_grant_id=grants.get(input_hash),
            now=_timestamp(moment),
            paper_count=max(1, len(spec.paper_ids)),
        )
        if not dispatched.decision.is_authorized:
            self._persist_manual_aux_audit_step(
                report_run_id, audit_pass, spec.node_id, dispatched.decision
            )
            return _StepResult("manual_required", error=dispatched.decision.reason_code)
        invocation = dispatched.result
        if (
            not isinstance(invocation, ModelInvocation)
            or invocation.derived_bytes != prompt.encode("utf-8")
        ):
            raise ReportAuditError("processing gate exposed an audit shard outside its decision")

        lease_token = self._claim_aux_audit_step(
            report_run_id,
            audit_pass,
            spec.node_id,
            input_hash,
            rendered_hash,
            input_tokens,
            dispatched.decision,
            owner,
            moment,
        )
        if lease_token is None:
            return _StepResult("running")
        try:
            request_value = CodexExecRequest(
                profile=PROFILE,
                prompt=prompt,
                output_schema=self.schemas["quality_audit"],
                schema_name=CALL_KIND_SCHEMAS["quality_audit"],
                prompt_name=CALL_KIND_PROMPTS["quality_audit"],
                input_hash=input_hash,
                call_kind="quality_audit",
                schema_path=self.resources.schema_path("quality_audit"),
                prompt_path=self.resources.prompt_path("quality_audit"),
                expected_prompt_hash=self.prompt_hashes["quality_audit"],
                schema_resource_paths=self.resources.configured_schema_resources(),
                expected_service_schema_hash=self.service_schema_hashes[
                    "quality_audit"
                ],
            )
            result = self.invoker_factory().invoke(request_value)
            self._validate_metadata(
                result.metadata, "quality_audit", input_hash, rendered_hash
            )
            self._validate_step_output(
                "quality_audit", result.output, spec.coverage, bundle, spec.payload
            )
            if len(canonical_json(result.output)) > AUDIT_OUTPUT_BYTE_LIMIT:
                raise ReportAuditOutputError(
                    "Sol audit shard/reduce output exceeds its frozen byte limit"
                )
            self._persist_complete_aux_audit_step(
                report_run_id,
                audit_pass,
                spec.node_id,
                result,
                owner,
                lease_token,
            )
            return _StepResult("complete", dict(result.output), result.metadata)
        except Exception as error:
            self._persist_failed_aux_audit_step(
                report_run_id,
                audit_pass,
                spec.node_id,
                error,
                owner,
                lease_token,
            )
            return _StepResult("failed", error=str(error))

    def _ensure_aux_audit_step(
        self,
        report_run_id: str,
        audit_pass: str,
        spec: _AuditShardSpec,
        ordinal: int,
        bundle_hash: str,
        input_hash: str,
        policy_facts: _PolicyFacts,
    ) -> None:
        step_id = "report-audit-shard-step-" + content_hash([
            report_run_id, audit_pass, spec.node_id
        ])
        values = (
            report_run_id,
            audit_pass,
            spec.node_id,
            spec.node_kind,
            ordinal,
            _json_text(list(spec.source_node_ids)),
            input_hash,
            bundle_hash,
            content_hash(spec.coverage),
            AUDIT_MAX_INPUT_TOKENS,
            AUDIT_OUTPUT_BYTE_LIMIT,
            PROFILE,
            MODEL,
            REASONING_EFFORT,
            CALL_KIND_PROMPTS["quality_audit"],
            self.prompt_hashes["quality_audit"],
            CALL_KIND_SCHEMAS["quality_audit"],
            self.schema_hashes["quality_audit"],
            _json_text(asdict(policy_facts)),
        )
        with self.database.transaction() as connection:
            row = connection.execute(
                """SELECT report_run_id, audit_pass, node_id, node_kind, ordinal,
                          source_node_ids_json, input_artifact_hash, input_bundle_hash,
                          expected_coverage_hash, input_token_limit, output_byte_limit,
                          profile, model_id, reasoning_effort, prompt_name, prompt_hash,
                          schema_name, schema_hash, processing_facts_json
                   FROM report_audit_shard_steps
                   WHERE report_audit_shard_step_id = ?""",
                (step_id,),
            ).fetchone()
            if row is None:
                connection.execute(
                    """INSERT INTO report_audit_shard_steps(
                           report_audit_shard_step_id, report_run_id, audit_pass,
                           node_id, node_kind, ordinal, source_node_ids_json,
                           input_artifact_hash, input_bundle_hash, expected_coverage_hash,
                           input_token_limit, output_byte_limit, profile, model_id,
                           reasoning_effort, prompt_name, prompt_hash, schema_name,
                           schema_hash, processing_facts_json, status
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')""",
                    (step_id, *values),
                )
            elif tuple(row) != values:
                raise ReportAuditError(
                    f"persisted audit shard binding has drifted: {spec.node_id}"
                )

    def _claim_aux_audit_step(
        self,
        report_run_id: str,
        audit_pass: str,
        node_id: str,
        input_hash: str,
        rendered_hash: str,
        input_tokens: int,
        decision: ProcessingDecision,
        owner: str,
        moment: datetime,
    ) -> int | None:
        attempts = MAX_RETRIES + 1
        reserved_tokens = input_tokens * attempts
        with self.database.transaction() as connection:
            row = connection.execute(
                """SELECT status, lease_token FROM report_audit_shard_steps
                   WHERE report_run_id = ? AND audit_pass = ? AND node_id = ?""",
                (report_run_id, audit_pass, node_id),
            ).fetchone()
            if row is None or row["status"] not in {"pending", "manual_required"}:
                return None
            plan = connection.execute(
                """SELECT rp.plan_json FROM report_runs rr
                   JOIN report_plans rp ON rp.report_plan_id = rr.report_plan_id
                   WHERE rr.report_run_id = ?""",
                (report_run_id,),
            ).fetchone()
            plan_document = _json_mapping(plan["plan_json"], "persisted ReportPlan")
            used = self._reserved_sol_budget(connection, report_run_id)
            if used[0] + attempts > int(plan_document["budget"]["max_sol_calls"]):
                raise ReportAuditBudgetError(
                    "Sol call budget is exhausted before audit shard dispatch"
                )
            if used[1] + reserved_tokens > int(plan_document["budget"]["max_input_tokens"]):
                raise ReportAuditBudgetError(
                    "Sol input-token budget is exhausted before audit shard dispatch"
                )
            next_token = int(row["lease_token"]) + 1
            expires = _timestamp(moment + timedelta(seconds=LEASE_SECONDS))
            updated = connection.execute(
                """UPDATE report_audit_shard_steps SET status = 'running',
                       actual_input_hash = ?, rendered_prompt_hash = ?, actual_input_tokens = ?,
                       budget_calls_reserved = budget_calls_reserved + ?,
                       budget_tokens_reserved = budget_tokens_reserved + ?,
                       processing_decision_json = ?, processing_grant_id = ?,
                       dispatch_count = dispatch_count + 1, lease_owner = ?, lease_token = ?,
                       lease_expires_at = ?, error_json = NULL, completed_at = NULL,
                       updated_at = CURRENT_TIMESTAMP
                   WHERE report_run_id = ? AND audit_pass = ? AND node_id = ?
                     AND status = ? AND lease_token = ?""",
                (
                    input_hash,
                    rendered_hash,
                    input_tokens,
                    attempts,
                    reserved_tokens,
                    _json_text(_decision_document(decision)),
                    decision.processing_grant_id,
                    owner,
                    next_token,
                    expires,
                    report_run_id,
                    audit_pass,
                    node_id,
                    row["status"],
                    row["lease_token"],
                ),
            )
            return next_token if updated.rowcount == 1 else None

    @staticmethod
    def _reserved_sol_budget(connection: Any, report_run_id: str) -> tuple[int, int]:
        totals = [0, 0]
        for table in (
            "report_reduce_nodes", "report_audit_steps", "report_audit_shard_steps"
        ):
            row = connection.execute(
                f"""SELECT COALESCE(SUM(budget_calls_reserved), 0),
                            COALESCE(SUM(budget_tokens_reserved), 0)
                     FROM {table} WHERE report_run_id = ?""",
                (report_run_id,),
            ).fetchone()
            totals[0] += int(row[0])
            totals[1] += int(row[1])
        return totals[0], totals[1]

    def _persist_manual_aux_audit_step(
        self,
        report_run_id: str,
        audit_pass: str,
        node_id: str,
        decision: ProcessingDecision,
    ) -> None:
        with self.database.transaction() as connection:
            connection.execute(
                """UPDATE report_audit_shard_steps SET status = 'manual_required',
                       processing_decision_json = ?, processing_grant_id = ?, error_json = ?,
                       updated_at = CURRENT_TIMESTAMP
                   WHERE report_run_id = ? AND audit_pass = ? AND node_id = ?
                     AND status IN ('pending', 'manual_required')""",
                (
                    _json_text(_decision_document(decision)),
                    decision.processing_grant_id,
                    _json_text({
                        "error": "processing_not_authorized",
                        "reason": decision.reason_code,
                    }),
                    report_run_id,
                    audit_pass,
                    node_id,
                ),
            )

    def _persist_complete_aux_audit_step(
        self,
        report_run_id: str,
        audit_pass: str,
        node_id: str,
        result: CodexExecResult,
        owner: str,
        lease_token: int,
    ) -> None:
        output = dict(result.output)
        metadata_document = asdict(result.metadata)
        with self.database.transaction() as connection:
            register_report_invocation(
                connection,
                report_run_id=report_run_id,
                invocation_id=result.metadata.invocation_id,
                phase="audit_shard",
                node_key=f"{audit_pass}:{node_id}",
                metadata=metadata_document,
            )
            updated = connection.execute(
                """UPDATE report_audit_shard_steps SET status = 'complete',
                       invocation_metadata_json = ?, output_json = ?, output_hash = ?,
                       lease_owner = NULL, lease_expires_at = NULL, error_json = NULL,
                       completed_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
                   WHERE report_run_id = ? AND audit_pass = ? AND node_id = ?
                     AND status = 'running' AND lease_owner = ? AND lease_token = ?""",
                (
                    _json_text(metadata_document),
                    _json_text(output),
                    content_hash(output),
                    report_run_id,
                    audit_pass,
                    node_id,
                    owner,
                    lease_token,
                ),
            )
            if updated.rowcount != 1:
                raise ReportAuditError("Sol result lost its audit-shard fencing token")

    def _persist_failed_aux_audit_step(
        self,
        report_run_id: str,
        audit_pass: str,
        node_id: str,
        error: Exception,
        owner: str,
        lease_token: int,
    ) -> None:
        with self.database.transaction() as connection:
            updated = connection.execute(
                """UPDATE report_audit_shard_steps SET status = 'failed', error_json = ?,
                       lease_owner = NULL, lease_expires_at = NULL,
                       completed_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
                   WHERE report_run_id = ? AND audit_pass = ? AND node_id = ?
                     AND status = 'running' AND lease_owner = ? AND lease_token = ?""",
                (
                    _json_text({"error": type(error).__name__, "message": str(error)}),
                    report_run_id,
                    audit_pass,
                    node_id,
                    owner,
                    lease_token,
                ),
            )
            if updated.rowcount != 1:
                raise ReportAuditError("Sol failure lost its audit-shard fencing token")

    def _expire_stale_aux_audit_step(
        self,
        report_run_id: str,
        audit_pass: str,
        node_id: str,
        moment: datetime,
    ) -> None:
        with self.database.transaction() as connection:
            connection.execute(
                """UPDATE report_audit_shard_steps SET status = 'failed', error_json = ?,
                       lease_owner = NULL, lease_expires_at = NULL,
                       completed_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
                   WHERE report_run_id = ? AND audit_pass = ? AND node_id = ?
                     AND status = 'running' AND lease_expires_at <= ?""",
                (
                    _json_text({
                        "error": "UncertainDispatch",
                        "message": (
                            "expired audit shard dispatch is terminal because it may already "
                            "have incurred cost"
                        ),
                    }),
                    report_run_id,
                    audit_pass,
                    node_id,
                    _timestamp(moment),
                ),
            )

    def _aux_audit_step_row(
        self, report_run_id: str, audit_pass: str, node_id: str
    ) -> Any:
        row = self.database.connection.execute(
            """SELECT * FROM report_audit_shard_steps
               WHERE report_run_id = ? AND audit_pass = ? AND node_id = ?""",
            (report_run_id, audit_pass, node_id),
        ).fetchone()
        if row is None:
            raise ReportAuditError(f"missing persisted audit shard: {node_id}")
        return row

    def _load_completed_aux_audit_step(
        self,
        row: Any,
        spec: _AuditShardSpec,
        prompt: str,
        rendered_hash: str,
        bundle: Mapping[str, Any],
        moment: datetime,
    ) -> _StepResult:
        required = (
            "actual_input_hash",
            "rendered_prompt_hash",
            "actual_input_tokens",
            "processing_decision_json",
            "invocation_metadata_json",
            "output_json",
            "output_hash",
        )
        if any(row[key] is None for key in required):
            raise ReportAuditError("completed audit shard is missing a frozen binding")
        input_hash = sha256(prompt.encode("utf-8")).hexdigest()
        rendered = self._rendered_prompt("quality_audit", prompt)
        if (
            row["actual_input_hash"] != input_hash
            or row["rendered_prompt_hash"] != rendered_hash
            or row["input_artifact_hash"] != input_hash
            or row["input_bundle_hash"] != _bundle_hash(bundle)
            or row["expected_coverage_hash"] != content_hash(spec.coverage)
            or row["source_node_ids_json"] != _json_text(list(spec.source_node_ids))
            or int(row["actual_input_tokens"]) != _token_upper_bound(rendered)
            or int(row["actual_input_tokens"]) > AUDIT_MAX_INPUT_TOKENS
            or int(row["output_byte_limit"]) != AUDIT_OUTPUT_BYTE_LIMIT
        ):
            raise ReportAuditError("completed audit shard input has drifted")
        metadata = _metadata(row["invocation_metadata_json"])
        self._validate_metadata(metadata, "quality_audit", input_hash, rendered_hash)
        try:
            require_report_invocation(
                self.database.connection,
                report_run_id=str(row["report_run_id"]),
                invocation_id=metadata.invocation_id,
                phase="audit_shard",
                node_key=f"{row['audit_pass']}:{row['node_id']}",
                metadata=asdict(metadata),
            )
        except ReportInvocationError as error:
            raise ReportAuditError(str(error)) from error
        output = _json_mapping(row["output_json"], "audit shard output")
        if (
            content_hash(output) != row["output_hash"]
            or len(canonical_json(output)) > AUDIT_OUTPUT_BYTE_LIMIT
        ):
            raise ReportAuditError("completed audit shard output has drifted")
        self._validate_step_output(
            "quality_audit", output, spec.coverage, bundle, spec.payload
        )
        decision = _json_mapping(row["processing_decision_json"], "processing decision")
        facts = _json_mapping(row["processing_facts_json"], "trusted processing facts")
        fresh = self._fresh_processing_decision(
            prompt,
            input_hash,
            facts,
            row["processing_grant_id"],
            moment,
            max(1, len(spec.paper_ids)),
            spec.paper_ids,
        )
        if (
            not fresh.is_authorized
            or canonical_json(_decision_document(fresh)) != canonical_json(decision)
            or
            decision.get("policy_version") != self.gate.policy.version
            or decision.get("policy_hash") != self.gate.policy.hash
            or decision.get("outcome") != facts.get("input_scope")
            or decision.get("input_artifact_hash") != input_hash
            or decision.get("provider") != PROCESSING_PROVIDER
            or decision.get("model") != MODEL
            or decision.get("purpose") != PURPOSE
            or decision.get("data_category") != "report_draft"
            or decision.get("authorized_by") not in {"policy", "grant"}
            or decision.get("processing_grant_id") != row["processing_grant_id"]
            or not _is_hash(facts.get("lineage_hash"))
            or facts.get("execution_mode") != self.execution_mode
            or int(row["dispatch_count"]) != 1
            or int(row["budget_calls_reserved"]) != MAX_RETRIES + 1
            or int(row["budget_tokens_reserved"])
            != int(row["actual_input_tokens"]) * (MAX_RETRIES + 1)
        ):
            raise ReportAuditError("completed audit-shard processing ledger has drifted")
        return _StepResult("complete", output, metadata, resumed=True)

    def _ensure_step(
        self,
        report_run_id: str,
        step_name: str,
        call_kind: str,
        bundle_hash: str,
        input_hash: str,
        coverage_hash: str | None,
        token_limit: int,
        policy_facts: _PolicyFacts,
    ) -> None:
        step_id = "report-audit-step-" + content_hash([report_run_id, step_name])
        values = (
            report_run_id,
            step_name,
            call_kind,
            input_hash,
            bundle_hash,
            coverage_hash,
            token_limit,
            PROFILE,
            MODEL,
            REASONING_EFFORT,
            CALL_KIND_PROMPTS[call_kind],
            self.prompt_hashes[call_kind],
            CALL_KIND_SCHEMAS[call_kind],
            self.schema_hashes[call_kind],
            _json_text(asdict(policy_facts)),
        )
        with self.database.transaction() as connection:
            row = connection.execute(
                """SELECT report_run_id, step_name, call_kind, input_artifact_hash,
                          input_bundle_hash, expected_coverage_hash, input_token_limit,
                          profile, model_id, reasoning_effort, prompt_name, prompt_hash,
                          schema_name, schema_hash, processing_facts_json
                   FROM report_audit_steps WHERE report_audit_step_id = ?""",
                (step_id,),
            ).fetchone()
            if row is None:
                connection.execute(
                    """INSERT INTO report_audit_steps(
                           report_audit_step_id, report_run_id, step_name, call_kind,
                           input_artifact_hash, input_bundle_hash, expected_coverage_hash,
                           input_token_limit, profile, model_id, reasoning_effort,
                           prompt_name, prompt_hash, schema_name, schema_hash,
                           processing_facts_json, status
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')""",
                    (step_id, *values),
                )
            elif tuple(row) != values:
                raise ReportAuditError(f"persisted {step_name} binding has drifted")

    def _claim_step(
        self,
        report_run_id: str,
        step_name: str,
        input_hash: str,
        rendered_hash: str,
        input_tokens: int,
        decision: ProcessingDecision,
        owner: str,
        moment: datetime,
    ) -> int | None:
        attempts = MAX_RETRIES + 1
        reserved_tokens = input_tokens * attempts
        with self.database.transaction() as connection:
            row = connection.execute(
                """SELECT status, lease_token FROM report_audit_steps
                   WHERE report_run_id = ? AND step_name = ?""",
                (report_run_id, step_name),
            ).fetchone()
            if row is None or row["status"] not in {"pending", "manual_required"}:
                return None
            plan = connection.execute(
                """SELECT rp.plan_json FROM report_runs rr
                   JOIN report_plans rp ON rp.report_plan_id = rr.report_plan_id
                   WHERE rr.report_run_id = ?""",
                (report_run_id,),
            ).fetchone()
            plan_document = _json_mapping(plan["plan_json"], "persisted ReportPlan")
            used_calls, used_tokens = self._reserved_sol_budget(
                connection, report_run_id
            )
            if used_calls + attempts > int(plan_document["budget"]["max_sol_calls"]):
                raise ReportAuditBudgetError("Sol call budget is exhausted before audit dispatch")
            if used_tokens + reserved_tokens > int(plan_document["budget"]["max_input_tokens"]):
                raise ReportAuditBudgetError("Sol input-token budget is exhausted before audit dispatch")
            next_token = int(row["lease_token"]) + 1
            expires = _timestamp(moment + timedelta(seconds=LEASE_SECONDS))
            updated = connection.execute(
                """UPDATE report_audit_steps SET status = 'running', actual_input_hash = ?,
                       rendered_prompt_hash = ?, actual_input_tokens = ?,
                       budget_calls_reserved = budget_calls_reserved + ?,
                       budget_tokens_reserved = budget_tokens_reserved + ?,
                       processing_decision_json = ?, processing_grant_id = ?,
                       dispatch_count = dispatch_count + 1, lease_owner = ?,
                       lease_token = ?, lease_expires_at = ?, error_json = NULL,
                       completed_at = NULL, updated_at = CURRENT_TIMESTAMP
                   WHERE report_run_id = ? AND step_name = ? AND status = ? AND lease_token = ?""",
                (
                    input_hash,
                    rendered_hash,
                    input_tokens,
                    attempts,
                    reserved_tokens,
                    _json_text(_decision_document(decision)),
                    decision.processing_grant_id,
                    owner,
                    next_token,
                    expires,
                    report_run_id,
                    step_name,
                    row["status"],
                    row["lease_token"],
                ),
            )
            return next_token if updated.rowcount == 1 else None

    def _persist_manual(
        self, report_run_id: str, step_name: str, decision: ProcessingDecision
    ) -> None:
        with self.database.transaction() as connection:
            connection.execute(
                """UPDATE report_audit_steps SET status = 'manual_required',
                       processing_decision_json = ?, processing_grant_id = ?, error_json = ?,
                       updated_at = CURRENT_TIMESTAMP
                   WHERE report_run_id = ? AND step_name = ?
                     AND status IN ('pending', 'manual_required')""",
                (
                    _json_text(_decision_document(decision)),
                    decision.processing_grant_id,
                    _json_text({"error": "processing_not_authorized", "reason": decision.reason_code}),
                    report_run_id,
                    step_name,
                ),
            )

    def _persist_complete_step(
        self,
        report_run_id: str,
        step_name: str,
        result: CodexExecResult,
        owner: str,
        lease_token: int,
    ) -> None:
        output = dict(result.output)
        metadata_document = asdict(result.metadata)
        with self.database.transaction() as connection:
            register_report_invocation(
                connection,
                report_run_id=report_run_id,
                invocation_id=result.metadata.invocation_id,
                phase="audit_step",
                node_key=step_name,
                metadata=metadata_document,
            )
            updated = connection.execute(
                """UPDATE report_audit_steps SET status = 'complete',
                       invocation_metadata_json = ?, output_json = ?, output_hash = ?,
                       lease_owner = NULL, lease_expires_at = NULL, error_json = NULL,
                       completed_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
                   WHERE report_run_id = ? AND step_name = ? AND status = 'running'
                     AND lease_owner = ? AND lease_token = ?""",
                (
                    _json_text(metadata_document),
                    _json_text(output),
                    content_hash(output),
                    report_run_id,
                    step_name,
                    owner,
                    lease_token,
                ),
            )
            if updated.rowcount != 1:
                raise ReportAuditError("Sol result lost its audit-step fencing token")

    def _persist_failed_step(
        self,
        report_run_id: str,
        step_name: str,
        error: Exception,
        owner: str,
        lease_token: int,
    ) -> None:
        with self.database.transaction() as connection:
            updated = connection.execute(
                """UPDATE report_audit_steps SET status = 'failed', error_json = ?,
                       lease_owner = NULL, lease_expires_at = NULL,
                       completed_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
                   WHERE report_run_id = ? AND step_name = ? AND status = 'running'
                     AND lease_owner = ? AND lease_token = ?""",
                (
                    _json_text({"error": type(error).__name__, "message": str(error)}),
                    report_run_id,
                    step_name,
                    owner,
                    lease_token,
                ),
            )
            if updated.rowcount != 1:
                raise ReportAuditError("Sol failure lost its audit-step fencing token")

    def _expire_stale_step(
        self, report_run_id: str, step_name: str, moment: datetime
    ) -> None:
        with self.database.transaction() as connection:
            connection.execute(
                """UPDATE report_audit_steps SET status = 'failed', error_json = ?,
                       lease_owner = NULL, lease_expires_at = NULL,
                       completed_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
                   WHERE report_run_id = ? AND step_name = ? AND status = 'running'
                     AND lease_expires_at <= ?""",
                (
                    _json_text({
                        "error": "UncertainDispatch",
                        "message": "expired dispatch is not repeated because it may already have incurred cost",
                    }),
                    report_run_id,
                    step_name,
                    _timestamp(moment),
                ),
            )

    def _step_row(self, report_run_id: str, step_name: str) -> Any:
        row = self.database.connection.execute(
            "SELECT * FROM report_audit_steps WHERE report_run_id = ? AND step_name = ?",
            (report_run_id, step_name),
        ).fetchone()
        if row is None:
            raise ReportAuditError(f"missing persisted audit step: {step_name}")
        return row

    def _load_completed_step(
        self,
        row: Any,
        call_kind: str,
        prompt: str,
        rendered_hash: str,
        expected_coverage: Mapping[str, Any] | None,
        bundle: Mapping[str, Any],
        moment: datetime,
        paper_count: int,
    ) -> _StepResult:
        required = (
            "actual_input_hash",
            "rendered_prompt_hash",
            "actual_input_tokens",
            "processing_decision_json",
            "invocation_metadata_json",
            "output_json",
            "output_hash",
        )
        if any(row[key] is None for key in required):
            raise ReportAuditError("completed audit step is missing a frozen binding")
        input_hash = sha256(prompt.encode("utf-8")).hexdigest()
        rendered = self._rendered_prompt(call_kind, prompt)
        if (
            row["actual_input_hash"] != input_hash
            or row["rendered_prompt_hash"] != rendered_hash
            or row["input_artifact_hash"] != input_hash
            or int(row["actual_input_tokens"]) != _token_upper_bound(rendered)
            or int(row["actual_input_tokens"]) > int(row["input_token_limit"])
            or int(row["actual_input_tokens"])
            > (
                AUDIT_MAX_INPUT_TOKENS
                if call_kind == "quality_audit"
                else REPAIR_MAX_INPUT_TOKENS
            )
        ):
            raise ReportAuditError("completed audit step input has drifted")
        metadata = _metadata(row["invocation_metadata_json"])
        self._validate_metadata(metadata, call_kind, input_hash, rendered_hash)
        try:
            require_report_invocation(
                self.database.connection,
                report_run_id=str(row["report_run_id"]),
                invocation_id=metadata.invocation_id,
                phase="audit_step",
                node_key=str(row["step_name"]),
                metadata=asdict(metadata),
            )
        except ReportInvocationError as error:
            raise ReportAuditError(str(error)) from error
        output = _json_mapping(row["output_json"], "audit step output")
        output_limit = (
            AUDIT_OUTPUT_BYTE_LIMIT
            if call_kind == "quality_audit"
            else MAX_OUTPUT_BYTES
        )
        if (
            content_hash(output) != row["output_hash"]
            or len(canonical_json(output)) > output_limit
        ):
            raise ReportAuditError("completed audit step output hash has drifted")
        replay_binding = _json_mapping(prompt, "persisted audit step input")
        self._validate_step_output(
            call_kind, output, expected_coverage, bundle, replay_binding
        )
        decision = _json_mapping(row["processing_decision_json"], "processing decision")
        facts = _json_mapping(row["processing_facts_json"], "trusted processing facts")
        fresh = self._fresh_processing_decision(
            prompt,
            input_hash,
            facts,
            row["processing_grant_id"],
            moment,
            paper_count,
            tuple(sorted(
                str(item["paper_id"])
                for item in bundle["corpus_snapshot"]["papers"]
            )),
        )
        if (
            not fresh.is_authorized
            or canonical_json(_decision_document(fresh)) != canonical_json(decision)
            or decision.get("policy_version") != self.gate.policy.version
            or decision.get("policy_hash") != self.gate.policy.hash
            or decision.get("outcome") != facts.get("input_scope")
            or decision.get("input_artifact_hash") != input_hash
            or decision.get("provider") != PROCESSING_PROVIDER
            or decision.get("model") != MODEL
            or decision.get("purpose") != PURPOSE
            or decision.get("data_category") != "report_draft"
            or decision.get("authorized_by") not in {"policy", "grant"}
            or decision.get("processing_grant_id") != row["processing_grant_id"]
            or not _is_hash(facts.get("lineage_hash"))
            or facts.get("execution_mode") != self.execution_mode
        ):
            raise ReportAuditError("completed audit processing decision has drifted")
        if (
            int(row["dispatch_count"]) != 1
            or int(row["budget_calls_reserved"]) != MAX_RETRIES + 1
            or int(row["budget_tokens_reserved"])
            != int(row["actual_input_tokens"]) * (MAX_RETRIES + 1)
        ):
            raise ReportAuditError("completed audit dispatch ledger has drifted")
        return _StepResult("complete", output, metadata, resumed=True)

    def _fresh_processing_decision(
        self,
        prompt: str,
        input_hash: str,
        facts: Mapping[str, Any],
        processing_grant_id: str | None,
        moment: datetime,
        paper_count: int,
        source_paper_ids: tuple[str, ...],
    ) -> ProcessingDecision:
        request = ProcessingRequest(
            artifact_hash=input_hash,
            artifact="report_draft",
            input_scope=str(facts["input_scope"]),
            license=facts.get("license"),
            access_basis=str(facts["access_basis"]),
            purpose=PURPOSE,
            data_category="report_draft",
            provider=PROCESSING_PROVIDER,
            model=MODEL,
            source_paper_ids=source_paper_ids,
            mode=str(facts["execution_mode"]),
            lineage_hash=str(facts["lineage_hash"]),
            derived_bytes=prompt.encode("utf-8"),
        )
        return self.gate.decide(
            request,
            processing_grant_id=processing_grant_id,
            now=_timestamp(moment),
            paper_count=paper_count,
        )

    def _validate_metadata(
        self,
        metadata: InvocationMetadata,
        call_kind: str,
        input_hash: str,
        rendered_hash: str,
    ) -> None:
        expected = (
            PROFILE,
            MODEL,
            REASONING_EFFORT,
            call_kind,
            CALL_KIND_SCHEMAS[call_kind],
            self.schema_hashes[call_kind],
            CALL_KIND_PROMPTS[call_kind],
            self.prompt_hashes[call_kind],
            input_hash,
            rendered_hash,
            MODEL,
            PROFILE,
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
            metadata.rendered_prompt_hash,
            metadata.actual_model,
            metadata.actual_profile,
        )
        if actual != expected or not self.resources.accepts_metadata_paths(
            call_kind, metadata.schema_path, metadata.prompt_path
        ):
            raise ReportAuditOutputError("Sol invocation metadata does not match the frozen audit step")
        if not str(metadata.invocation_id).strip():
            raise ReportAuditOutputError("Sol invocation metadata lacks a fresh invocation ID")
        if metadata.attempts < 1 or metadata.attempts > MAX_RETRIES + 1:
            raise ReportAuditOutputError("Sol invocation exceeded the frozen retry budget")

    def _validate_step_output(
        self,
        call_kind: str,
        output: Mapping[str, Any],
        expected_coverage: Mapping[str, Any] | None,
        bundle: Mapping[str, Any],
        payload: Mapping[str, Any],
    ) -> None:
        try:
            self.resources.validate(output, call_kind)
        except SchemaValidationError as error:
            raise ReportAuditOutputError(str(error)) from error
        if call_kind == "quality_audit":
            expected_pass = str(payload.get("audit_pass") or "A")
            self._validate_audit(output, bundle, expected_coverage, expected_pass)
            scope = payload.get("audit_scope")
            if isinstance(scope, Mapping) and scope.get("kind") == "stable_section_claim_shard":
                prefix = f"{scope['node_id']}:"
                if any(
                    not str(finding["finding_id"]).startswith(prefix)
                    for finding in output["findings"]
                ):
                    raise ReportAuditOutputError(
                        "audit shard finding IDs must be namespaced by the stable shard ID"
                    )
            if isinstance(scope, Mapping) and scope.get("kind") == "exhaustive_audit_reduce":
                self._validate_audit_reduce(output, expected_coverage, payload)
        else:
            base_hash = payload.get("base_artifact_hash") or _bundle_hash(bundle)
            if output["base_artifact_hash"] != base_hash:
                raise ReportAuditOutputError("repair patch set is bound to another report artifact")

    def _validate_audit(
        self,
        audit: Mapping[str, Any],
        bundle: Mapping[str, Any],
        expected_coverage: Mapping[str, Any] | None,
        expected_pass: str,
    ) -> None:
        assert expected_coverage is not None
        expected = (
            expected_pass,
            content_hash(bundle["document"]),
            _bundle_hash(bundle),
            bundle["plan"]["plan_hash"],
            self.rubric_hash,
            content_hash(list(_search_limitations(bundle))),
            canonical_json(expected_coverage),
        )
        actual = (
            audit["audit_pass"],
            audit["report_document_hash"],
            audit["report_artifact_hash"],
            audit["report_plan_hash"],
            audit["rubric_hash"],
            audit["search_limitations_hash"],
            canonical_json(audit["coverage_ledger"]),
        )
        if actual != expected:
            raise ReportAuditOutputError("quality audit does not cover the exact frozen report inputs")
        block_ids = set(expected_coverage["block_ids"])
        claim_ids = set(expected_coverage["claim_ids"])
        corpus_papers = {
            str(item["paper_id"]) for item in bundle["corpus_snapshot"]["papers"]
        }
        finding_ids: set[str] = set()
        for finding in audit["findings"]:
            if finding["finding_id"] in finding_ids:
                raise ReportAuditOutputError("quality audit contains duplicate finding IDs")
            finding_ids.add(str(finding["finding_id"]))
            if not set(finding["block_ids"]).issubset(block_ids):
                raise ReportAuditOutputError("quality audit finding names an unknown block")
            if not set(finding["claim_ids"]).issubset(claim_ids):
                raise ReportAuditOutputError("quality audit finding names an unknown claim")
            if not set(finding["paper_ids"]).issubset(corpus_papers):
                raise ReportAuditOutputError("quality audit finding names an unknown paper")

    def _validate_audit_reduce(
        self,
        output: Mapping[str, Any],
        expected_coverage: Mapping[str, Any] | None,
        payload: Mapping[str, Any],
    ) -> None:
        assert expected_coverage is not None
        sources = payload.get("source_audits")
        scope = payload.get("audit_scope")
        if not isinstance(sources, Sequence) or isinstance(sources, (str, bytes)) or not sources:
            raise ReportAuditOutputError("audit reduce requires every source audit")
        if not isinstance(scope, Mapping):
            raise ReportAuditOutputError("audit reduce lacks its stable source manifest")
        source_ids: list[str] = []
        source_coverages: list[Mapping[str, Any]] = []
        source_findings: dict[str, Mapping[str, Any]] = {}
        coverage_complete = True
        for source in sources:
            if not isinstance(source, Mapping) or not isinstance(source.get("audit"), Mapping):
                raise ReportAuditOutputError("audit reduce source is malformed")
            source_id = str(source.get("node_id") or "")
            audit = source["audit"]
            if (
                not source_id
                or source.get("output_hash") != content_hash(audit)
                or audit.get("audit_pass") != output["audit_pass"]
            ):
                raise ReportAuditOutputError("audit reduce source binding has drifted")
            source_ids.append(source_id)
            source_coverages.append(audit["coverage_ledger"])
            coverage_complete = coverage_complete and bool(audit["coverage_complete"])
            for finding in audit["findings"]:
                finding_id = str(finding["finding_id"])
                if finding_id in source_findings:
                    raise ReportAuditOutputError(
                        "audit reduce sources contain duplicate finding IDs"
                    )
                source_findings[finding_id] = finding
        if (
            source_ids != list(scope.get("source_node_ids", ()))
            or len(source_ids) != len(set(source_ids))
            or canonical_json(_coverage_union(expected_coverage, source_coverages))
            != canonical_json(expected_coverage)
            or bool(output["coverage_complete"]) != coverage_complete
        ):
            raise ReportAuditOutputError(
                "audit reduce sampled or misreported its source coverage"
            )
        output_findings = {
            str(item["finding_id"]): item for item in output["findings"]
        }
        if (
            len(output_findings) != len(output["findings"])
            or canonical_json(output_findings) != canonical_json(source_findings)
        ):
            raise ReportAuditOutputError(
                "audit reduce must preserve the exact union of every shard finding"
            )

    def _apply_repair(
        self,
        bundle: Mapping[str, Any],
        audit: Mapping[str, Any],
        repair: Mapping[str, Any],
    ) -> dict[str, Any]:
        if repair["base_artifact_hash"] != _bundle_hash(bundle):
            raise ReportAuditOutputError("repair patch set base hash has drifted")
        severe = _severe_findings(audit)
        allowed_blocks = {str(value) for item in severe for value in item["block_ids"]}
        allowed_claims = {str(value) for item in severe for value in item["claim_ids"]}
        allowed_papers = {str(value) for item in severe for value in item["paper_ids"]}
        repaired = deepcopy(dict(bundle))
        blocks = {str(item["block_id"]): item for item in repaired["document"]["blocks"]}
        claims = {str(item["claim_id"]): item for item in repaired["claims"]}
        papers = {str(item["paper_id"]): item for item in repaired["coverage"]["papers"]}
        seen: set[tuple[str, str]] = set()
        for patch in repair["patches"]:
            target = str(patch["target"])
            if target == "REPORT_DOCUMENT":
                identifier = str(patch["block_id"])
                key = (target, identifier)
                old = blocks.get(identifier)
                if old is None or identifier not in allowed_blocks:
                    raise ReportAuditOutputError("repair targets a block outside blocker/major findings")
                if patch["operation"] != "replace_block" or content_hash(old) != patch["precondition_hash"]:
                    raise ReportAuditOutputError("report block repair precondition failed")
                value = deepcopy(dict(patch["value"]))
                if (
                    value["block_id"] != identifier
                    or value["block_kind"] != old["block_kind"]
                    or value["section_id"] != old["section_id"]
                ):
                    raise ReportAuditOutputError("repair cannot change stable block identity or placement")
                blocks[identifier] = value
            elif target == "CLAIMS_EVIDENCE":
                identifier = str(patch["claim_id"])
                key = (target, identifier)
                old = claims.get(identifier)
                if old is None or identifier not in allowed_claims:
                    raise ReportAuditOutputError("repair targets a claim outside blocker/major findings")
                if patch["operation"] != "replace_claim" or content_hash(old) != patch["precondition_hash"]:
                    raise ReportAuditOutputError("claim repair precondition failed")
                value = deepcopy(dict(patch["value"]))
                immutable = (
                    "claim_id", "claim_key", "research_question_id", "report_section",
                    "claim_type", "mapping_status",
                )
                if any(canonical_json(value[field]) != canonical_json(old[field]) for field in immutable):
                    raise ReportAuditOutputError("repair cannot change stable claim identity or classification")
                old_refs = {
                    content_hash(ref)
                    for field in ("supporting_evidence", "contradicting_evidence")
                    for ref in old[field]
                }
                new_refs = {
                    content_hash(ref)
                    for field in ("supporting_evidence", "contradicting_evidence")
                    for ref in value[field]
                }
                if new_refs != old_refs:
                    raise ReportAuditOutputError(
                        "repair cannot add, remove, or rewrite frozen evidence references"
                    )
                claims[identifier] = value
            else:
                identifier = str(patch["paper_id"])
                key = (target, identifier)
                old = papers.get(identifier)
                if old is None or identifier not in allowed_papers:
                    raise ReportAuditOutputError("repair targets coverage outside blocker/major findings")
                if patch["operation"] != "replace_paper" or content_hash(old) != patch["precondition_hash"]:
                    raise ReportAuditOutputError("coverage repair precondition failed")
                value = deepcopy(dict(patch["value"]))
                if value["paper_id"] != identifier:
                    raise ReportAuditOutputError("repair cannot change stable coverage paper identity")
                if (
                    set(value["evidence_claim_ids"]) != set(old.get("evidence_claim_ids", ()))
                    or set(value["consumed_node_ids"]) != set(old.get("consumed_node_ids", ()))
                ):
                    raise ReportAuditOutputError("repair cannot fabricate coverage consumption")
                papers[identifier] = value
            if key in seen:
                raise ReportAuditOutputError("repair contains duplicate targets")
            seen.add(key)
        repaired["document"]["blocks"] = [
            blocks[str(item["block_id"])] for item in bundle["document"]["blocks"]
        ]
        repaired["claims"] = [claims[str(item["claim_id"])] for item in bundle["claims"]]
        repaired["coverage"]["papers"] = [
            papers[str(item["paper_id"])] for item in bundle["coverage"]["papers"]
        ]
        # Coverage disposition, table assignment, and any background-only
        # reason were approved before Sol ran. A model repair cannot rewrite
        # that deterministic coverage fact.
        if canonical_json(repaired["coverage"]) != canonical_json(bundle["coverage"]):
            raise ReportAuditOutputError(
                "repair cannot change deterministic selected-paper coverage"
            )
        if _bundle_hash(repaired) == _bundle_hash(bundle):
            raise ReportAuditOutputError("repair patch set made no structured change")
        return repaired

    def _persist_repaired_bundle(
        self,
        report_run_id: str,
        previous: Mapping[str, Any],
        repaired: Mapping[str, Any],
    ) -> None:
        with self.database.transaction() as connection:
            updated = connection.execute(
                """UPDATE report_audit_runs SET current_artifact_hash = ?,
                       current_bundle_json = ?, repair_count = 1,
                       updated_at = CURRENT_TIMESTAMP
                   WHERE report_run_id = ? AND current_artifact_hash = ? AND repair_count = 0""",
                (
                    _bundle_hash(repaired),
                    _json_text(_mutable_part(repaired)),
                    report_run_id,
                    _bundle_hash(previous),
                ),
            )
            if updated.rowcount != 1:
                row = connection.execute(
                    """SELECT current_artifact_hash, current_bundle_json, repair_count
                       FROM report_audit_runs WHERE report_run_id = ?""",
                    (report_run_id,),
                ).fetchone()
                if row is None or (
                    row["current_artifact_hash"] != _bundle_hash(repaired)
                    or row["current_bundle_json"] != _json_text(_mutable_part(repaired))
                    or row["repair_count"] != 1
                ):
                    raise ReportAuditError("persisted repair state has drifted")

    def _publish(
        self,
        report_run_id: str,
        bundle: Mapping[str, Any],
        audit: Mapping[str, Any],
        audit_passes: Sequence[str],
        resumed: Sequence[str],
        previous: Mapping[str, Any] | None,
    ) -> ReportAuditResult:
        if not audit["coverage_complete"] or _severe_findings(audit):
            raise ReportAuditError("publish requires complete audit coverage and zero severe findings")
        self._validate_invocation_registry(report_run_id)
        try:
            target = self.report_store.directory(report_run_id)
            if target.exists():
                self._reconcile_existing_publish(target, bundle, audit, previous)
            else:
                try:
                    self.report_store.write(
                        plan=bundle["plan"],
                        search_audit=bundle["search_audit"],
                        corpus_snapshot=bundle["corpus_snapshot"],
                        claims=bundle["claims"],
                        comparison_groups=bundle["comparison_groups"],
                        claim_relations=bundle["claim_relations"],
                        document=bundle["document"],
                        coverage=bundle["coverage"],
                        bibliography=bundle["bibliography"],
                        audit=audit,
                        previous=previous,
                        rubric_path=self.rubric_path,
                    )
                except ReportArtifactError:
                    if not target.exists():
                        raise
                    self._reconcile_existing_publish(target, bundle, audit, previous)
        except (ReportArtifactError, ReportAuditError) as error:
            return self._finish_failed(
                report_run_id,
                bundle,
                audit_passes,
                resumed,
                f"immutable report publish failed: {error}",
            )
        assert target is not None
        relative = str(target.relative_to(self.report_store.root))
        with self.database.transaction() as connection:
            connection.execute(
                """UPDATE report_audit_runs SET status = 'complete', final_audit_step = ?,
                       published_relative_path = ?, error_json = NULL,
                       completed_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
                   WHERE report_run_id = ?""",
                ("audit_c" if audit["audit_pass"] == "C" else "audit_a", relative, report_run_id),
            )
            report = connection.execute(
                "SELECT run_id FROM report_runs WHERE report_run_id = ?", (report_run_id,)
            ).fetchone()
            connection.execute(
                """UPDATE report_runs SET status = 'complete', output_relative_path = ?,
                       completed_at = CURRENT_TIMESTAMP WHERE report_run_id = ?""",
                (relative, report_run_id),
            )
            connection.execute(
                """UPDATE pipeline_runs SET status = 'complete', completed_at = CURRENT_TIMESTAMP
                   WHERE run_id = ?""",
                (report["run_id"],),
            )
        return ReportAuditResult(
            report_run_id,
            "complete",
            content_hash(bundle["document"]),
            tuple(audit_passes),
            1 if "C" in audit_passes else 0,
            target,
            tuple(resumed),
        )

    def _reconcile_existing_publish(
        self,
        target: Path,
        bundle: Mapping[str, Any],
        audit: Mapping[str, Any],
        previous: Mapping[str, Any] | None,
    ) -> None:
        try:
            if target != self.report_store.directory(str(bundle["document"]["report_run_id"])):
                raise ReportArtifactError("existing immutable report directory is foreign")
            self.report_store.reconcile(
                plan=bundle["plan"],
                search_audit=bundle["search_audit"],
                corpus_snapshot=bundle["corpus_snapshot"],
                claims=bundle["claims"],
                comparison_groups=bundle["comparison_groups"],
                claim_relations=bundle["claim_relations"],
                document=bundle["document"],
                coverage=bundle["coverage"],
                bibliography=bundle["bibliography"],
                audit=audit,
                previous=previous,
                rubric_path=self.rubric_path,
            )
        except ReportArtifactError as error:
            raise ReportAuditError(
                "existing immutable report bundle conflicts with final audit state"
            ) from error

    def _finish_incomplete(
        self,
        report_run_id: str,
        bundle: Mapping[str, Any],
        audits: Sequence[str],
        resumed: Sequence[str],
        error: str,
    ) -> ReportAuditResult:
        self._mark_terminal(report_run_id, "incomplete", error)
        return ReportAuditResult(
            report_run_id,
            "incomplete",
            content_hash(bundle["document"]),
            tuple(audits),
            int(self._run_row(report_run_id)["repair_count"]),
            resumed_steps=tuple(resumed),
            error=error,
        )

    def _finish_failed(
        self,
        report_run_id: str,
        bundle: Mapping[str, Any],
        audits: Sequence[str],
        resumed: Sequence[str],
        error: str,
    ) -> ReportAuditResult:
        self._mark_terminal(report_run_id, "failed", error)
        return ReportAuditResult(
            report_run_id,
            "failed",
            content_hash(bundle["document"]),
            tuple(audits),
            int(self._run_row(report_run_id)["repair_count"]),
            resumed_steps=tuple(resumed),
            error=error,
        )

    def _stop_for_step(
        self,
        report_run_id: str,
        bundle: Mapping[str, Any],
        step: _StepResult,
        *,
        audit_passes: Sequence[str] = (),
        resumed: Sequence[str] = (),
    ) -> ReportAuditResult:
        status = step.status if step.status in {"manual_required", "running"} else "failed"
        if status != "running":
            self._mark_terminal(report_run_id, status, step.error or status)
        return ReportAuditResult(
            report_run_id,
            status,
            content_hash(bundle["document"]),
            tuple(audit_passes),
            int(self._run_row(report_run_id)["repair_count"]),
            resumed_steps=tuple(resumed),
            error=step.error,
        )

    def _mark_terminal(self, report_run_id: str, status: str, error: str) -> None:
        database_status = "incomplete" if status == "manual_required" else status
        with self.database.transaction() as connection:
            connection.execute(
                """UPDATE report_audit_runs SET status = ?, error_json = ?,
                       completed_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
                   WHERE report_run_id = ?""",
                (status, _json_text({"error": error}), report_run_id),
            )
            report = connection.execute(
                "SELECT run_id FROM report_runs WHERE report_run_id = ?", (report_run_id,)
            ).fetchone()
            connection.execute(
                """UPDATE report_runs SET status = ?, completed_at = CURRENT_TIMESTAMP
                   WHERE report_run_id = ?""",
                (database_status, report_run_id),
            )
            connection.execute(
                """UPDATE pipeline_runs SET status = ?, completed_at = CURRENT_TIMESTAMP
                   WHERE run_id = ?""",
                (database_status, report["run_id"]),
            )

    def _set_running(self, report_run_id: str) -> None:
        with self.database.transaction() as connection:
            connection.execute(
                """UPDATE report_audit_runs SET status = 'running', error_json = NULL,
                       completed_at = NULL, updated_at = CURRENT_TIMESTAMP
                   WHERE report_run_id = ? AND status NOT IN ('complete', 'failed')""",
                (report_run_id,),
            )
            report = connection.execute(
                "SELECT run_id FROM report_runs WHERE report_run_id = ?", (report_run_id,)
            ).fetchone()
            connection.execute(
                "UPDATE report_runs SET status = 'running', completed_at = NULL WHERE report_run_id = ?",
                (report_run_id,),
            )
            connection.execute(
                "UPDATE pipeline_runs SET status = 'running', completed_at = NULL WHERE run_id = ?",
                (report["run_id"],),
            )

    def _completed_result(
        self,
        report_run_id: str,
        row: Any,
        initial: Mapping[str, Any],
        policy_facts: _PolicyFacts,
        budget: Mapping[str, int],
        previous: Mapping[str, Any] | None,
        audit_a_plan: _AuditPassPlan,
    ) -> ReportAuditResult:
        def replay(
            step_name: str,
            call_kind: str,
            bundle: Mapping[str, Any],
            payload: Mapping[str, Any],
            expected_coverage: Mapping[str, Any] | None,
        ) -> _StepResult:
            persisted_step = self._step_row(report_run_id, step_name)
            if persisted_step["status"] != "complete":
                raise ReportAuditError(
                    f"completed report audit has a non-complete step: {step_name}"
                )
            result = self._run_step(
                report_run_id,
                step_name,
                call_kind,
                bundle,
                payload,
                policy_facts,
                budget["repair" if call_kind == "repair" else step_name],
                {},
                self._now(),
                "completed-audit-replay",
                expected_coverage,
                paper_count=len({
                    str(item["paper_id"])
                    for item in bundle["corpus_snapshot"]["papers"]
                }),
            )
            if result.status != "complete" or result.output is None or result.metadata is None:
                raise ReportAuditError(f"completed report audit cannot replay {step_name}")
            return result

        reduce_invocations = self._reduce_invocation_ids(report_run_id)
        audit_a = self._run_audit_pass(
            report_run_id,
            "audit_a",
            audit_a_plan,
            initial,
            policy_facts,
            budget,
            {},
            "completed-audit-replay",
            forbidden_invocation_ids=reduce_invocations,
        )
        if audit_a.status != "complete":
            raise ReportAuditError("completed report audit cannot replay audit_a")
        assert audit_a.output is not None and audit_a.metadata is not None
        if audit_a.metadata.invocation_id in reduce_invocations:
            raise ReportAuditError("completed audit A reused a reduce invocation")
        expected_steps = {"audit_a"}
        audits = ("A",)
        current = initial
        final_audit = audit_a.output
        severe = _severe_findings(audit_a.output)
        if not audit_a.output["coverage_complete"]:
            raise ReportAuditError("completed report audit has incomplete audit A coverage")
        if severe:
            repair_payload = self._repair_payload(report_run_id, initial, audit_a.output)
            repair = replay("repair", "repair", initial, repair_payload, None)
            assert repair.output is not None and repair.metadata is not None
            if repair.metadata.invocation_id in {
                *reduce_invocations,
                audit_a.metadata.invocation_id,
            }:
                raise ReportAuditError("completed repair B reused a prior invocation")
            current = self._apply_repair(initial, audit_a.output, repair.output)
            if content_hash(current["document"]) == content_hash(initial["document"]):
                raise ReportAuditError("completed repair did not change ReportDocument")
            repaired_verification = self._deterministic_verify(current)
            coverage_c = audit_coverage_ledger(current["document"], current["claims"])
            audit_c_payload = self._audit_payload(
                report_run_id,
                "C",
                current,
                repaired_verification,
                coverage_c,
            )
            audit_c_plan = self._audit_pass_plan(
                report_run_id,
                "C",
                current,
                repaired_verification,
                coverage_c,
                audit_c_payload,
            )
            audit_c = self._run_audit_pass(
                report_run_id,
                "audit_c",
                audit_c_plan,
                current,
                policy_facts,
                budget,
                {},
                "completed-audit-replay",
                forbidden_invocation_ids=frozenset({
                    *reduce_invocations,
                    audit_a.metadata.invocation_id,
                    repair.metadata.invocation_id,
                }),
            )
            if audit_c.status != "complete":
                raise ReportAuditError("completed report audit cannot replay audit_c")
            assert audit_c.output is not None and audit_c.metadata is not None
            if audit_c.metadata.invocation_id in {
                *reduce_invocations,
                audit_a.metadata.invocation_id,
                repair.metadata.invocation_id,
            }:
                raise ReportAuditError("completed audit C reused a prior invocation")
            if not audit_c.output["coverage_complete"] or _severe_findings(audit_c.output):
                raise ReportAuditError("completed audit C does not satisfy the publish gate")
            expected_steps.update({"repair", "audit_c"})
            audits = ("A", "C")
            final_audit = audit_c.output
        persisted_current = self._load_current_bundle(row, initial)
        if _bundle_hash(persisted_current) != _bundle_hash(current):
            raise ReportAuditError("completed report bundle differs from replayed audit state")
        expected_repair_count = 1 if severe else 0
        if (
            int(row["repair_count"]) != expected_repair_count
            or row["final_audit_step"] != ("audit_c" if severe else "audit_a")
        ):
            raise ReportAuditError("completed report audit terminal state has drifted")
        steps = self.database.connection.execute(
            """SELECT step_name, status FROM report_audit_steps
               WHERE report_run_id = ?""",
            (report_run_id,),
        ).fetchall()
        if {str(item["step_name"]): str(item["status"]) for item in steps} != {
            name: "complete" for name in expected_steps
        }:
            raise ReportAuditError("completed report audit has an unexpected step ledger")
        expected_path = self.report_store.directory(report_run_id)
        expected_relative = str(expected_path.relative_to(self.report_store.root))
        if row["published_relative_path"] != expected_relative:
            raise ReportAuditError("completed report audit publish path has drifted")
        self._reconcile_existing_publish(
            expected_path, current, final_audit, previous
        )
        return ReportAuditResult(
            report_run_id,
            "complete",
            content_hash(current["document"]),
            audits,
            expected_repair_count,
            expected_path,
            tuple(sorted(expected_steps)),
        )

    def _terminal_result(self, report_run_id: str, row: Any) -> ReportAuditResult:
        current = _json_mapping(row["current_bundle_json"], "terminal report bundle")
        return ReportAuditResult(
            report_run_id,
            str(row["status"]),
            content_hash(current["document"]),
            (),
            int(row["repair_count"]),
            error=_error_message(row["error_json"]),
        )

    def _rendered_prompt(self, call_kind: str, prompt: str) -> str:
        template = self.resources.prompt(call_kind)
        encoded = json.dumps(
            {"authorized_input": prompt}, ensure_ascii=False, separators=(",", ":")
        )
        return f"{template.rstrip()}\n\nThe authorized input follows as JSON data:\n{encoded}\n"


def _mutable_bundle(bundle: ReportBundle) -> dict[str, Any]:
    return {
        "plan": deepcopy(dict(bundle.plan)),
        "search_audit": deepcopy(dict(bundle.search_audit)),
        "corpus_snapshot": deepcopy(dict(bundle.corpus_snapshot)),
        "claims": deepcopy([dict(item) for item in bundle.claims]),
        # Caller values are untrusted and may be arbitrarily large.  They are
        # compared with a bounded walk and replaced by a persisted-claims
        # derivation before any hash, copy, budget, or model call.
        "comparison_groups": {},
        "claim_relations": [],
        "document": deepcopy(dict(bundle.document)),
        "coverage": deepcopy(dict(bundle.coverage)),
        "bibliography": deepcopy(dict(bundle.bibliography)),
    }


def _mutable_part(bundle: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "document": bundle["document"],
        "claims": bundle["claims"],
        "coverage": bundle["coverage"],
    }


def _bundle_hash(bundle: Mapping[str, Any]) -> str:
    return report_artifact_hash(
        document=bundle["document"],
        claims=bundle["claims"],
        coverage=bundle["coverage"],
        comparison_groups=bundle["comparison_groups"],
        claim_relations=bundle["claim_relations"],
        bibliography=bundle["bibliography"],
    )


def _classification_values(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    raw = (value,) if isinstance(value, str) else tuple(value)  # type: ignore[arg-type]
    return tuple(sorted({str(item).strip() for item in raw if str(item).strip()}))


def _reduce_output_kind(call_kind: str) -> str:
    try:
        return {
            "section_reduce": "evidence",
            "cross_section_reduce": "claim_ledger",
            "final_reduce": "report_draft",
        }[call_kind]
    except KeyError as error:
        raise ReportAuditError("canonical reduce tree has an unsupported call kind") from error


def _reduce_prompt_payload(
    report_run_id: str,
    plan: Mapping[str, Any],
    node: Any,
    documents: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
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


def _search_limitations(bundle: Mapping[str, Any]) -> tuple[str, ...]:
    return audit_search_limitations(
        bundle["search_audit"], bundle["corpus_snapshot"]
    )


def _severe_findings(audit: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [item for item in audit["findings"] if item["severity"] in SEVERE]


def _coverage_subset(
    full: Mapping[str, Any],
    block_ids: Sequence[str],
    claim_ids: Sequence[str],
) -> dict[str, Any]:
    block_set = {str(value) for value in block_ids}
    claim_set = {str(value) for value in claim_ids}
    return {
        "block_ids": [
            str(value) for value in full["block_ids"] if str(value) in block_set
        ],
        "claim_ids": [
            str(value) for value in full["claim_ids"] if str(value) in claim_set
        ],
        "evidence_refs": [
            item for item in full["evidence_refs"]
            if str(item["claim_id"]) in claim_set
        ],
    }


def _coverage_union(
    full: Mapping[str, Any], parts: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    block_ids = {
        str(value) for part in parts for value in part.get("block_ids", ())
    }
    claim_ids = {
        str(value) for part in parts for value in part.get("claim_ids", ())
    }
    if (
        not block_ids.issubset({str(value) for value in full["block_ids"]})
        or not claim_ids.issubset({str(value) for value in full["claim_ids"]})
    ):
        raise ReportAuditOutputError(
            "audit shard coverage names unknown blocks or claims"
        )
    expected_evidence = {
        content_hash(item): item for item in full["evidence_refs"]
    }
    evidence_hashes = {
        content_hash(item)
        for part in parts
        for item in part.get("evidence_refs", ())
    }
    unknown = evidence_hashes - set(expected_evidence)
    if unknown:
        raise ReportAuditOutputError("audit shard coverage names unknown evidence refs")
    return {
        "block_ids": [
            str(value) for value in full["block_ids"] if str(value) in block_ids
        ],
        "claim_ids": [
            str(value) for value in full["claim_ids"] if str(value) in claim_ids
        ],
        "evidence_refs": [
            item for item in full["evidence_refs"]
            if content_hash(item) in evidence_hashes
        ],
    }


def _load_mapping(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ReportAuditError(f"{label} is unavailable") from error
    if not isinstance(value, Mapping):
        raise ReportAuditError(f"{label} must be an object")
    return dict(value)


def _default_rubric_path() -> Path:
    repository = Path(__file__).resolve().parents[2] / "policies" / "report-audit-rubric-v1.yaml"
    if repository.is_file():
        return repository
    installed = (
        Path(sysconfig.get_path("data"))
        / "share" / "paper-agent" / "policies" / "report-audit-rubric-v1.yaml"
    )
    if installed.is_file():
        return installed
    raise ReportAuditError("frozen report audit rubric is unavailable")


def _metadata(value: str | bytes | None) -> InvocationMetadata:
    document = _json_mapping(value, "Sol invocation metadata")
    try:
        return InvocationMetadata(**document)
    except TypeError as error:
        raise ReportAuditError("Sol invocation metadata is malformed") from error


def _json_mapping(value: str | bytes | None, label: str) -> Mapping[str, Any]:
    try:
        document = json.loads(value) if isinstance(value, (str, bytes)) else value
    except json.JSONDecodeError as error:
        raise ReportAuditError(f"{label} is not valid JSON") from error
    if not isinstance(document, Mapping):
        raise ReportAuditError(f"{label} must be a JSON object")
    return document


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _json_hash(value: Mapping[str, Any]) -> str:
    return sha256(_json_text(value).encode("utf-8")).hexdigest()


def _decision_document(decision: ProcessingDecision) -> dict[str, Any]:
    value = asdict(decision)
    value["outcome"] = decision.outcome.value
    return value


def _token_upper_bound(value: str) -> int:
    return max(1, len(value.encode("utf-8")))


def _is_hash(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _error_message(value: str | None) -> str | None:
    if not value:
        return None
    document = _json_mapping(value, "persisted report audit error")
    return str(document.get("message") or document.get("reason") or document.get("error"))
