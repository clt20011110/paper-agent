"""Frozen ReportPlan, corpus snapshot, and search-audit inputs for Stage 4b."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any

from .approval import ApprovalError, approve, approved_content_hash, require_valid_approval
from .canonical import canonical_json, content_hash
from .codex_exec import CALL_KIND_PROMPTS, prompt_directory
from .schema import SchemaValidationError, schema_directory, validate


REPORT_SECTION_IDS = (
    "executive_summary",
    "scope_and_methods",
    "search_flow_and_corpus",
    "field_taxonomy",
    "evidence_synthesis",
    "resource_comparison",
    "conflicts_and_limitations",
    "research_gaps",
    "practical_recommendations",
    "report_limitations",
    "references_and_appendices",
)
CLASSIFICATION_AXES = (
    "subquestion",
    "theme",
    "method_family",
    "task",
    "dataset",
    "benchmark",
    "time",
    "venue",
    "publication_status",
    "evidence_type",
    "study_setting",
)
SOURCE_CATEGORIES = frozenset({"user_library", "newly_discovered", "citation_snowball"})
PUBLICATION_STATUSES = frozenset({"peer_reviewed", "workshop", "preprint", "unknown"})
STUDY_SETTINGS = frozenset({"real", "simulation", "theory", "other"})
INPUT_SCOPES = frozenset({"full_pdf", "abstract_only", "metadata_only", "missing"})
EVIDENCE_LEVELS = frozenset({
    "full_text_direct",
    "full_text_inferred",
    "abstract_direct",
    "metadata_only",
})


class ReportPlanError(ValueError):
    pass


class ReportPlanDriftError(ReportPlanError):
    pass


@dataclass(frozen=True, slots=True)
class CorpusPaper:
    paper_id: str
    analysis_run_id: str | None
    analysis_artifact_hash: str | None
    lineage_hashes: tuple[str, ...]
    source_category: str
    publication_status: str
    study_setting: str
    input_scope: str
    evidence_level: str
    foundational: bool
    recent: bool
    incomplete_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.paper_id:
            raise ReportPlanError("corpus paper_id is required")
        if self.source_category not in SOURCE_CATEGORIES:
            raise ReportPlanError(f"unknown source category: {self.source_category}")
        if self.publication_status not in PUBLICATION_STATUSES:
            raise ReportPlanError(f"unknown publication status: {self.publication_status}")
        if self.study_setting not in STUDY_SETTINGS:
            raise ReportPlanError(f"unknown study setting: {self.study_setting}")
        if self.input_scope not in INPUT_SCOPES or self.evidence_level not in EVIDENCE_LEVELS:
            raise ReportPlanError("unknown corpus input scope or evidence level")
        if self.input_scope != "missing" and (not self.analysis_run_id or not self.analysis_artifact_hash):
            raise ReportPlanError("an analyzed corpus paper requires a bound analysis run and artifact")
        if self.input_scope == "missing" and not self.incomplete_reason:
            raise ReportPlanError("missing corpus papers require an incomplete reason")
        allowed_levels = {
            "full_pdf": {"full_text_direct", "full_text_inferred", "abstract_direct", "metadata_only"},
            "abstract_only": {"abstract_direct", "metadata_only"},
            "metadata_only": {"metadata_only"},
            "missing": {"metadata_only"},
        }
        if self.evidence_level not in allowed_levels[self.input_scope]:
            raise ReportPlanError("corpus evidence level overstates its input scope")

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "analysis_run_id": self.analysis_run_id,
            "analysis_artifact_hash": self.analysis_artifact_hash,
            "lineage_hashes": sorted(set(self.lineage_hashes)),
            "source_category": self.source_category,
            "publication_status": self.publication_status,
            "study_setting": self.study_setting,
            "input_scope": self.input_scope,
            "evidence_level": self.evidence_level,
            "foundational": self.foundational,
            "recent": self.recent,
            "incomplete_reason": self.incomplete_reason,
        }


@dataclass(frozen=True, slots=True)
class ReportPlanBundle:
    plan: Mapping[str, Any]
    corpus_snapshot: Mapping[str, Any]
    search_audit: Mapping[str, Any]


def build_corpus_snapshot(
    papers: Sequence[CorpusPaper],
    *,
    query_plan_hash: str,
    search_audit: Mapping[str, Any],
    created_at: str,
) -> dict[str, Any]:
    if not created_at:
        raise ReportPlanError("corpus snapshot created_at is required")
    if len({paper.paper_id for paper in papers}) != len(papers):
        raise ReportPlanError("corpus snapshot contains duplicate paper_ids")
    if search_audit.get("plan_hash") != query_plan_hash:
        raise ReportPlanError("search audit does not match the QueryPlan hash")
    source_audit_hash = content_hash(search_audit)
    core = {
        "schema_version": "1",
        "query_plan_hash": query_plan_hash,
        "search_audit_source_hash": source_audit_hash,
        "papers": [paper.to_dict() for paper in sorted(papers, key=lambda item: item.paper_id)],
    }
    snapshot_hash = content_hash(core)
    snapshot = {
        **core,
        "snapshot_id": f"corpus-{snapshot_hash[:12]}",
        "snapshot_hash": snapshot_hash,
        "created_at": created_at,
    }
    _validate_corpus_snapshot(snapshot)
    return snapshot


def build_search_audit_pack(
    search_audit: Mapping[str, Any],
    corpus_snapshot: Mapping[str, Any],
    *,
    screening_flow: Mapping[str, int],
    exclusion_reasons: Mapping[str, int],
    required_providers: Sequence[str] = (),
    search_limitations: Sequence[str] = (),
    budget_exhausted: bool = False,
    created_at: str,
) -> dict[str, Any]:
    if not created_at:
        raise ReportPlanError("search audit pack created_at is required")
    _validate_corpus_snapshot(corpus_snapshot)
    if content_hash(search_audit) != corpus_snapshot["search_audit_source_hash"]:
        raise ReportPlanError("search audit has drifted from the corpus snapshot")
    required_flow = {"raw_discovered", "unique_after_dedup", "stage2_screened", "included"}
    if set(screening_flow) != required_flow:
        raise ReportPlanError("screening_flow fields are incomplete")
    counts = {name: int(value) for name, value in screening_flow.items()}
    reasons = {str(reason): int(count) for reason, count in sorted(exclusion_reasons.items())}
    if any(value < 0 for value in (*counts.values(), *reasons.values())):
        raise ReportPlanError("search flow counts cannot be negative")
    paper_count = len(corpus_snapshot["papers"])
    if counts["included"] != paper_count:
        raise ReportPlanError("search flow included count does not match the frozen corpus")
    if not (
        counts["raw_discovered"] >= counts["unique_after_dedup"] >= counts["stage2_screened"]
    ):
        raise ReportPlanError("search flow counts are not monotonic")
    expected_excluded = counts["stage2_screened"] - counts["included"]
    if expected_excluded < 0 or sum(reasons.values()) != expected_excluded:
        raise ReportPlanError("exclusion reasons do not reconcile with Stage 2 screening")

    provider_statuses: dict[str, list[str]] = {}
    for source in search_audit.get("sources", ()):
        provider_statuses.setdefault(str(source["provider"]), []).append(str(source["status"]))
    required_failures = tuple(sorted(
        provider
        for provider in set(str(item) for item in required_providers)
        if provider not in provider_statuses
        or any(status != "complete" for status in provider_statuses[provider])
    ))
    round_budget_exhausted = any(
        round_.get("stop_reason") == "budget_exhausted"
        for round_ in search_audit.get("rounds", ())
    )
    exhausted = budget_exhausted or round_budget_exhausted
    incomplete_sources = tuple(sorted(str(item) for item in search_audit.get("incomplete_sources", ())))
    limitations = tuple(dict.fromkeys(
        [str(item) for item in search_limitations]
        + [f"incomplete source run: {item}" for item in incomplete_sources]
        + [f"required provider failed: {item}" for item in required_failures]
        + (["search budget exhausted"] if exhausted else [])
        + ([f"search run status: {search_audit.get('status')}"] if search_audit.get("status") != "complete" else [])
    ))
    papers = tuple(corpus_snapshot["papers"])
    input_scope_counts = Counter(str(paper["input_scope"]) for paper in papers)
    core = {
        "schema_version": "1",
        "source_audit_hash": corpus_snapshot["search_audit_source_hash"],
        "query_plan_hash": corpus_snapshot["query_plan_hash"],
        "corpus_snapshot_hash": corpus_snapshot["snapshot_hash"],
        "search_status": search_audit.get("status"),
        "flow_label": "PRISMA-style retrieval flow; not a claim of clinical PRISMA compliance",
        "flow": {
            **counts,
            "excluded": expected_excluded,
            "excluded_by_reason": reasons,
            "full_pdf": input_scope_counts["full_pdf"],
            "abstract_only": input_scope_counts["abstract_only"],
            "missing": input_scope_counts["metadata_only"] + input_scope_counts["missing"],
        },
        "source_categories": _counts(papers, "source_category", SOURCE_CATEGORIES),
        "cohorts": {
            "foundational": sum(bool(paper["foundational"]) for paper in papers),
            "recent": sum(bool(paper["recent"]) for paper in papers),
        },
        "publication_status": _counts(papers, "publication_status", PUBLICATION_STATUSES),
        "input_scope": _counts(papers, "input_scope", INPUT_SCOPES),
        "study_setting": _counts(papers, "study_setting", STUDY_SETTINGS),
        "required_provider_failures": list(required_failures),
        "incomplete_source_runs": list(incomplete_sources),
        "budget_exhausted": exhausted,
        "limitations": list(limitations),
        "query_manifest": deepcopy(list(search_audit.get("queries", ()))),
        "source_round_audit": deepcopy(dict(search_audit)),
    }
    pack_hash = content_hash(core)
    return {
        **core,
        "pack_id": f"search-audit-{pack_hash[:12]}",
        "pack_hash": pack_hash,
        "created_at": created_at,
    }


def compile_report_plan(
    draft: Mapping[str, Any],
    *,
    corpus_snapshot: Mapping[str, Any],
    search_audit_pack: Mapping[str, Any],
    plan_id: str | None = None,
    created_at: str | None = None,
    schema_root: Path | None = None,
    prompt_root: Path | None = None,
) -> dict[str, Any]:
    _validate_report_inputs(corpus_snapshot, search_audit_pack)
    source = deepcopy(dict(draft))
    timestamp = created_at or str(source.pop("created_at", ""))
    if not timestamp:
        raise ReportPlanError("created_at is required to compile a ReportPlan")
    root = schema_directory(schema_root)
    report_schema = json.loads((root / "report-plan.schema.json").read_text(encoding="utf-8"))
    prompts = prompt_directory() if prompt_root is None else prompt_root
    prompt_hashes = {
        call_kind: sha256((prompts / filename).read_bytes()).hexdigest()
        for call_kind, filename in CALL_KIND_PROMPTS.items()
    }
    fields = (
        "objective",
        "audience",
        "primary_question",
        "subquestions",
        "synthesis_question",
        "scope",
        "sections",
        "classification_axes",
        "cohort_rules",
        "paper_memberships",
        "artifacts",
        "budget",
    )
    missing = [field for field in fields if field not in source]
    if missing:
        raise ReportPlanError(f"ReportPlan draft is missing fields: {missing}")
    plan = {
        "schema_version": "1",
        "plan_id": "",
        "plan_hash": "",
        "status": "draft",
        "created_at": timestamp,
        **{field: source[field] for field in fields[:6]},
        "query_plan_hash": corpus_snapshot["query_plan_hash"],
        "corpus_snapshot_hash": corpus_snapshot["snapshot_hash"],
        **{field: source[field] for field in fields[6:]},
        "schema_hash": content_hash(report_schema),
        "prompt_hashes": prompt_hashes,
        "approval": None,
    }
    _validate_plan_semantics(plan, corpus_snapshot)
    plan_hash = approved_content_hash(plan)
    plan["plan_id"] = plan_id or f"report-plan-{plan_hash[:12]}"
    plan["plan_hash"] = plan_hash
    try:
        validate(plan, "report-plan.schema.json", root)
    except SchemaValidationError as error:
        raise ReportPlanError(str(error)) from error
    return plan


def approve_report_plan(
    plan: Mapping[str, Any],
    expected_hash: str,
    *,
    approved_by: str,
    approved_at: str,
) -> dict[str, Any]:
    try:
        approved = approve(
            plan,
            expected_hash,
            approved_by=approved_by,
            approved_at=approved_at,
            hash_field="plan_hash",
        )
        validate(approved, "report-plan.schema.json")
    except (ApprovalError, SchemaValidationError) as error:
        raise ReportPlanError(str(error)) from error
    return approved


def assert_report_runtime_matches(
    approved_plan: Mapping[str, Any],
    runtime_plan: Mapping[str, Any],
    *,
    corpus_snapshot: Mapping[str, Any],
    search_audit_pack: Mapping[str, Any],
) -> None:
    try:
        require_valid_approval(approved_plan, "plan_hash")
    except ApprovalError as error:
        raise ReportPlanDriftError(str(error)) from error
    try:
        _validate_report_inputs(corpus_snapshot, search_audit_pack)
    except ReportPlanError as error:
        raise ReportPlanDriftError(str(error)) from error
    if corpus_snapshot["snapshot_hash"] != approved_plan["corpus_snapshot_hash"]:
        raise ReportPlanDriftError("corpus snapshot has drifted")
    if corpus_snapshot["query_plan_hash"] != approved_plan["query_plan_hash"]:
        raise ReportPlanDriftError("QueryPlan hash has drifted")
    if approved_content_hash(runtime_plan) != approved_plan["plan_hash"]:
        raise ReportPlanDriftError("ReportPlan scope, membership, schema, prompt, or budget has drifted")


class ReportPlanStore:
    """Immutable filesystem bundle for an approved plan and its frozen inputs."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def directory(self, plan_id: str) -> Path:
        return self.root / "reports" / "plans" / plan_id

    def draft_path(self, plan_id: str) -> Path:
        return self.directory(plan_id) / "REPORT_PLAN.draft.json"

    def approved_path(self, plan_id: str) -> Path:
        return self.directory(plan_id) / "REPORT_PLAN.json"

    @property
    def latest_path(self) -> Path:
        return self.root / "reports" / "latest-approved-plan.json"

    def save_draft(self, plan: Mapping[str, Any]) -> Path:
        if plan.get("status") != "draft":
            raise ReportPlanError("save_draft only accepts draft plans")
        path = self.draft_path(str(plan["plan_id"]))
        self._atomic_write(path, canonical_json(dict(plan)))
        return path

    def approve_and_save(
        self,
        plan: Mapping[str, Any],
        expected_hash: str,
        *,
        approved_by: str,
        approved_at: str,
        corpus_snapshot: Mapping[str, Any],
        search_audit_pack: Mapping[str, Any],
    ) -> dict[str, Any]:
        approved = approve_report_plan(
            plan,
            expected_hash,
            approved_by=approved_by,
            approved_at=approved_at,
        )
        self.save_bundle(approved, corpus_snapshot, search_audit_pack)
        return approved

    def save_bundle(
        self,
        plan: Mapping[str, Any],
        corpus_snapshot: Mapping[str, Any],
        search_audit_pack: Mapping[str, Any],
    ) -> None:
        try:
            require_valid_approval(plan, "plan_hash")
            validate(plan, "report-plan.schema.json")
        except (ApprovalError, SchemaValidationError) as error:
            raise ReportPlanError(str(error)) from error
        _validate_report_inputs(corpus_snapshot, search_audit_pack)
        if plan["query_plan_hash"] != corpus_snapshot["query_plan_hash"]:
            raise ReportPlanError("ReportPlan QueryPlan hash does not match its corpus")
        if plan["corpus_snapshot_hash"] != corpus_snapshot["snapshot_hash"]:
            raise ReportPlanError("ReportPlan corpus snapshot hash does not match")
        directory = self.directory(str(plan["plan_id"]))
        documents = {
            directory / "REPORT_PLAN.json": plan,
            directory / "CORPUS_SNAPSHOT.json": corpus_snapshot,
            directory / "SEARCH_AUDIT.json": search_audit_pack,
        }
        for path, document in documents.items():
            self._write_immutable(path, canonical_json(dict(document)))
        self._atomic_write(
            self.latest_path,
            canonical_json({"plan_id": plan["plan_id"], "plan_hash": plan["plan_hash"]}),
        )

    def load_approved(self, plan_id: str) -> dict[str, Any]:
        document = json.loads(self.approved_path(plan_id).read_text(encoding="utf-8"))
        try:
            require_valid_approval(document, "plan_hash")
            validate(document, "report-plan.schema.json")
        except (ApprovalError, SchemaValidationError) as error:
            raise ReportPlanError(str(error)) from error
        return document

    def load_bundle(self, plan_id: str) -> ReportPlanBundle:
        directory = self.directory(plan_id)
        plan = self.load_approved(plan_id)
        corpus = json.loads((directory / "CORPUS_SNAPSHOT.json").read_text(encoding="utf-8"))
        audit = json.loads((directory / "SEARCH_AUDIT.json").read_text(encoding="utf-8"))
        _validate_report_inputs(corpus, audit)
        if plan["corpus_snapshot_hash"] != corpus["snapshot_hash"]:
            raise ReportPlanError("stored corpus does not match the approved ReportPlan")
        return ReportPlanBundle(plan, corpus, audit)

    @staticmethod
    def _write_immutable(path: Path, payload: bytes) -> None:
        if path.exists():
            if path.read_bytes() != payload:
                raise ReportPlanError(f"approved report input is immutable: {path.name}")
            return
        ReportPlanStore._atomic_write(path, payload)

    @staticmethod
    def _atomic_write(path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_bytes(payload)
        os.replace(temporary, path)


def _validate_plan_semantics(plan: Mapping[str, Any], corpus_snapshot: Mapping[str, Any]) -> None:
    subquestions = tuple(str(item["id"]) for item in plan["subquestions"])
    if len(set(subquestions)) != len(subquestions):
        raise ReportPlanError("ReportPlan subquestion IDs must be unique")
    section_ids = tuple(str(item["id"]) for item in plan["sections"])
    if len(set(section_ids)) != len(section_ids):
        raise ReportPlanError("ReportPlan section IDs must be unique")
    missing_sections = set(REPORT_SECTION_IDS) - set(section_ids)
    if missing_sections:
        raise ReportPlanError(f"ReportPlan is missing required report sections: {sorted(missing_sections)}")
    for section in plan["sections"]:
        unknown = set(str(item) for item in section["subquestion_ids"]) - set(subquestions)
        if unknown:
            raise ReportPlanError(f"section {section['id']} references unknown subquestions")
    if tuple(plan["classification_axes"]) != CLASSIFICATION_AXES:
        raise ReportPlanError("ReportPlan must freeze the complete classification axis order")

    memberships: dict[str, Mapping[str, Any]] = {}
    for membership in plan["paper_memberships"]:
        paper_id = str(membership["paper_id"])
        assigned = tuple(str(item) for item in membership["section_ids"])
        if paper_id in memberships or membership["primary_section_id"] not in assigned:
            raise ReportPlanError(f"invalid or duplicate paper membership: {paper_id}")
        if not assigned or not set(assigned).issubset(section_ids):
            raise ReportPlanError(f"paper {paper_id} references unknown sections")
        memberships[paper_id] = membership
    corpus_ids = {str(paper["paper_id"]) for paper in corpus_snapshot["papers"]}
    if set(memberships) != corpus_ids:
        raise ReportPlanError("ReportPlan paper membership does not exactly match the corpus snapshot")


def _validate_report_inputs(
    corpus_snapshot: Mapping[str, Any], search_audit_pack: Mapping[str, Any]
) -> None:
    _validate_corpus_snapshot(corpus_snapshot)
    _validate_search_audit_pack(search_audit_pack)
    if search_audit_pack["corpus_snapshot_hash"] != corpus_snapshot["snapshot_hash"]:
        raise ReportPlanError("search audit pack does not match the corpus snapshot")
    if search_audit_pack["query_plan_hash"] != corpus_snapshot["query_plan_hash"]:
        raise ReportPlanError("search audit pack does not match the QueryPlan")
    if search_audit_pack["source_audit_hash"] != corpus_snapshot["search_audit_source_hash"]:
        raise ReportPlanError("search audit source hash does not match the corpus snapshot")


def _validate_corpus_snapshot(snapshot: Mapping[str, Any]) -> None:
    core = {
        "schema_version": snapshot["schema_version"],
        "query_plan_hash": snapshot["query_plan_hash"],
        "search_audit_source_hash": snapshot["search_audit_source_hash"],
        "papers": snapshot["papers"],
    }
    if snapshot["schema_version"] != "1" or snapshot["snapshot_hash"] != content_hash(core):
        raise ReportPlanError("corpus snapshot hash has drifted")
    paper_ids = [str(paper["paper_id"]) for paper in snapshot["papers"]]
    if paper_ids != sorted(paper_ids) or len(set(paper_ids)) != len(paper_ids):
        raise ReportPlanError("corpus papers must be unique and sorted by stable paper_id")
    for paper in snapshot["papers"]:
        CorpusPaper(
            paper_id=str(paper["paper_id"]),
            analysis_run_id=paper["analysis_run_id"],
            analysis_artifact_hash=paper["analysis_artifact_hash"],
            lineage_hashes=tuple(paper["lineage_hashes"]),
            source_category=str(paper["source_category"]),
            publication_status=str(paper["publication_status"]),
            study_setting=str(paper["study_setting"]),
            input_scope=str(paper["input_scope"]),
            evidence_level=str(paper["evidence_level"]),
            foundational=bool(paper["foundational"]),
            recent=bool(paper["recent"]),
            incomplete_reason=paper["incomplete_reason"],
        )


def _validate_search_audit_pack(pack: Mapping[str, Any]) -> None:
    core = {
        key: value
        for key, value in pack.items()
        if key not in {"pack_id", "pack_hash", "created_at"}
    }
    if pack["schema_version"] != "1" or pack["pack_hash"] != content_hash(core):
        raise ReportPlanError("search audit pack hash has drifted")
    if content_hash(pack["source_round_audit"]) != pack["source_audit_hash"]:
        raise ReportPlanError("search audit source content has drifted")


def _counts(
    papers: Sequence[Mapping[str, Any]], field: str, vocabulary: Sequence[str] | frozenset[str]
) -> dict[str, int]:
    counts = Counter(str(paper[field]) for paper in papers)
    return {value: counts[value] for value in sorted(vocabulary)}
