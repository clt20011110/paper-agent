"""Thin file-facing adapters for the Stage 4b command-line surface.

The CLI owns argument parsing and structured output.  This module only turns
explicit JSON files and immutable report directories into the inputs expected
by the report-plan and report-artifact services.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

from .canonical import content_hash
from .report_artifacts import ReportArtifactStore, report_diff, verify_report
from .report_config import ReportResources
from .report_plan import (
    ReportPlanError,
    ReportPlanStore,
    approve_report_plan,
    assert_report_runtime_matches,
    compile_report_plan,
)
from .schema import SchemaValidationError


@dataclass(frozen=True, slots=True)
class ReportPlanFileResult:
    plan: Mapping[str, Any]
    path: Path
    saved: bool


@dataclass(frozen=True, slots=True)
class ReportRunBundle:
    plan: Mapping[str, Any]
    search_audit: Mapping[str, Any]
    corpus_snapshot: Mapping[str, Any]
    claims: tuple[Mapping[str, Any], ...]
    comparison_groups: Mapping[str, Mapping[str, Any]]
    claim_relations: tuple[Mapping[str, Any], ...]
    document: Mapping[str, Any]
    coverage: Mapping[str, Any]
    bibliography: Mapping[str, Mapping[str, Any]]

    def diff_input(self) -> dict[str, Any]:
        value = {
            "plan": self.plan,
            "claims": self.claims,
            "corpus_snapshot": self.corpus_snapshot,
        }
        report_run_id = self.document.get("report_run_id")
        if isinstance(report_run_id, str) and report_run_id:
            value["report_run_id"] = report_run_id
        return value


def compile_report_plan_from_files(
    draft_path: str | Path,
    corpus_snapshot_path: str | Path,
    search_audit_path: str | Path,
    output_root: str | Path,
    *,
    save_draft: bool = True,
    resources: ReportResources | None = None,
) -> ReportPlanFileResult:
    """Compile a ReportPlan draft from three explicit JSON inputs."""
    plan = compile_report_plan(
        _load_mapping(draft_path),
        corpus_snapshot=_load_mapping(corpus_snapshot_path),
        search_audit_pack=_load_mapping(search_audit_path),
        resources=resources,
    )
    store = ReportPlanStore(output_root)
    path = store.draft_path(str(plan["plan_id"]))
    if save_draft:
        store.save_draft(plan)
    return ReportPlanFileResult(plan, path, save_draft)


def approve_report_plan_from_files(
    draft_path: str | Path,
    corpus_snapshot_path: str | Path,
    search_audit_path: str | Path,
    output_root: str | Path,
    *,
    expected_hash: str,
    approved_by: str,
    approved_at: str,
    save_bundle: bool = True,
    resources: ReportResources | None = None,
) -> ReportPlanFileResult:
    """Approve a persisted draft and write its immutable input bundle."""
    draft = _load_mapping(draft_path)
    assert_report_plan_resource_binding(draft, resources)
    plan = approve_report_plan(
        draft,
        expected_hash,
        approved_by=approved_by,
        approved_at=approved_at,
    )
    store = ReportPlanStore(output_root)
    corpus_snapshot = _load_mapping(corpus_snapshot_path)
    search_audit = _load_mapping(search_audit_path)
    assert_report_runtime_matches(
        plan,
        plan,
        corpus_snapshot=corpus_snapshot,
        search_audit_pack=search_audit,
    )
    if save_bundle:
        store.save_bundle(plan, corpus_snapshot, search_audit)
    return ReportPlanFileResult(
        plan, store.approved_path(str(plan["plan_id"])), save_bundle
    )


def assert_report_plan_resource_binding(
    plan: Mapping[str, Any], resources: ReportResources | None = None
) -> None:
    """Require a frozen plan to match the schemas and prompts used at runtime."""
    selected = resources or ReportResources.defaults()
    selected.validate_files()
    expected_schema_hash = content_hash(selected.schema("planning_assist"))
    expected_prompt_hashes = {
        call_kind: sha256(path.read_bytes()).hexdigest()
        for call_kind, path in selected.prompt_paths.items()
    }
    if plan.get("schema_hash") != expected_schema_hash:
        raise ReportPlanError(
            "ReportPlan schema hash does not match the configured planning schema"
        )
    if plan.get("prompt_hashes") != expected_prompt_hashes:
        raise ReportPlanError(
            "ReportPlan prompt hashes do not match the configured prompts"
        )
    try:
        selected.validate(plan, "planning_assist")
    except SchemaValidationError as error:
        raise ReportPlanError(str(error)) from error


def verify_report_run(
    output_root: str | Path,
    report_run_id: str,
    *,
    previous_report_run_id: str | None = None,
) -> dict[str, Any]:
    """Run the deterministic verifier over one immutable report bundle."""
    bundle = load_report_run_bundle(output_root, report_run_id)
    previous = (
        load_report_run_bundle(output_root, previous_report_run_id).diff_input()
        if previous_report_run_id is not None
        else None
    )
    return verify_report(
        plan=bundle.plan,
        document=bundle.document,
        claims=bundle.claims,
        coverage=bundle.coverage,
        bibliography=bundle.bibliography,
        comparison_groups=bundle.comparison_groups,
        search_audit=bundle.search_audit,
        corpus_snapshot=bundle.corpus_snapshot,
        previous=previous,
        claim_relations=bundle.claim_relations,
    )


def diff_report_runs(
    output_root: str | Path,
    previous_report_run_id: str,
    report_run_id: str,
) -> dict[str, Any]:
    """Return the explicit lineage-based diff between two immutable runs."""
    previous = load_report_run_bundle(output_root, previous_report_run_id)
    current = load_report_run_bundle(output_root, report_run_id)
    return report_diff(
        previous.diff_input(),
        current.diff_input(),
        claim_relations=current.claim_relations,
    )


def load_report_run_bundle(
    output_root: str | Path, report_run_id: str
) -> ReportRunBundle:
    """Load only the deterministic-verifier inputs from a published run."""
    directory = ReportArtifactStore(output_root).directory(report_run_id)
    bundle = ReportRunBundle(
        plan=_load_mapping(directory / "REPORT_PLAN.json"),
        search_audit=_load_mapping(directory / "SEARCH_AUDIT.json"),
        corpus_snapshot=_load_mapping(directory / "CORPUS_SNAPSHOT.json"),
        claims=_load_jsonl(directory / "CLAIMS_EVIDENCE.jsonl"),
        comparison_groups=_load_mapping(directory / "COMPARISON_GROUPS.json"),
        claim_relations=_load_json_list(directory / "CLAIM_RELATIONS.json"),
        document=_load_mapping(directory / "REPORT_DOCUMENT.json"),
        coverage=_load_mapping(directory / "COVERAGE.json"),
        bibliography=_load_mapping(directory / "BIBLIOGRAPHY.json"),
    )
    if bundle.document.get("report_run_id") != report_run_id:
        raise ValueError(
            "REPORT_DOCUMENT report_run_id does not match its immutable directory"
        )
    return bundle


def _load_mapping(path: str | Path) -> Mapping[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{Path(path).name} must contain a JSON object")
    return value


def _load_json_list(path: str | Path) -> tuple[Mapping[str, Any], ...]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(item, Mapping) for item in value):
        raise ValueError(f"{Path(path).name} must contain a JSON object list")
    return tuple(value)


def _load_jsonl(path: str | Path) -> tuple[Mapping[str, Any], ...]:
    claims = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise ValueError(f"{Path(path).name} line {line_number} must contain a JSON object")
        claims.append(value)
    return tuple(claims)
