"""Run the policy-gated Stage 4 and one-shot Stage 4b venue smoke tail.

This module deliberately starts from *persisted* native Stage 1--3 run IDs.  It
never guesses a latest run and it never invokes a model unless
``--execute-models`` is present.  The default invocation prepares exact grants
and stops at the next paid boundary.

The public :func:`execute_stage4_and_4b` entry point is intentionally separate
from ``run_venue_e2e_matrix.py`` so a venue runner can hand off a completed
native pipeline without duplicating Stage 4/4b policy or resume logic.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any

import yaml

# Keep the checked-out script directly runnable before an editable install.
ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from paper_agent.analysis import AnalysisInvoker
from paper_agent.analysis_cli_service import AnalysisCliService
from paper_agent.approval import require_valid_approval
from paper_agent.artifacts import ArtifactStore
from paper_agent.canonical import canonical_json, content_hash
from paper_agent.config import load_config
from paper_agent.grants import GrantError, GrantStore, create_grant_draft
from paper_agent.processing import ArtifactProcessingPolicy
from paper_agent.report_artifacts import ReportArtifactStore
from paper_agent.report_cli_service import verify_report_run
from paper_agent.report_config import ReportResources, ReportRuntimeConfig
from paper_agent.report_direct import (
    DirectReportInvoker,
    one_shot_config_hash,
    one_shot_validation_config_hash,
)
from paper_agent.report_execution_service import ReportExecutionService
from paper_agent.report_input_service import ReportInputRequest, ReportInputService
from paper_agent.report_plan import (
    CLASSIFICATION_AXES,
    REPORT_SECTION_IDS,
    ReportPlanBundle,
    ReportPlanStore,
    compile_report_plan,
)
from paper_agent.processing import ProcessingGate
from paper_agent.schema import schema_directory
from paper_agent.storage import Database


STATE_SCHEMA_VERSION = "paper-agent.venue-e2e-stage4b.v1"
DEFAULT_POLICY = ROOT / "policies" / "artifact-processing-v1.yaml"
STATE_PATH = Path("stage4b") / "runtime.json"
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
ONE_SHOT_NODE_ID = "one_shot:0001"

SECTION_TITLES = {
    "executive_summary": "执行摘要",
    "scope_and_methods": "研究范围与方法",
    "search_flow_and_corpus": "检索流与语料画像",
    "field_taxonomy": "领域图景与分类体系",
    "evidence_synthesis": "证据综合",
    "resource_comparison": "方法、数据集、基准与资源对比",
    "conflicts_and_limitations": "矛盾结论、不可比项与证据局限",
    "research_gaps": "研究空白与可检验机会",
    "practical_recommendations": "实践结论与建议",
    "report_limitations": "本报告限制与更新状态",
    "references_and_appendices": "参考文献与附录",
}


class VenueE2ERuntimeError(RuntimeError):
    """The frozen venue handoff cannot safely continue."""


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise VenueE2ERuntimeError(f"cannot read JSON object: {path}") from error
    if not isinstance(value, dict):
        raise VenueE2ERuntimeError(f"JSON root must be an object: {path}")
    return value


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temporary.write(payload)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


def _write_state(path: Path, document: Mapping[str, Any]) -> None:
    _atomic_write(path, canonical_json(dict(document)))


def _write_immutable(path: Path, document: Mapping[str, Any]) -> None:
    payload = canonical_json(dict(document))
    if path.exists():
        if path.read_bytes() != payload:
            raise VenueE2ERuntimeError(f"immutable runtime artifact has drifted: {path}")
        return
    _atomic_write(path, payload)


def _frozen_report_inputs(
    run_dir: Path, state: Mapping[str, Any], *, resume: bool
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str] | None:
    """Load the already-approved Stage 4b input on resume without rebuilding it.

    Canonical metadata may be enriched after the sole Sol request was frozen.
    Rebuilding a corpus snapshot at that point would silently change the prompt
    binding and either strand or redispatch the saved output.  The approved plan
    directory is therefore the only resume source once it exists.
    """
    stage4b = state.get("stage4b")
    if not resume or not isinstance(stage4b, Mapping):
        return None
    plan_id = stage4b.get("report_plan_id")
    bundle_id = stage4b.get("report_input_bundle_id")
    if not isinstance(plan_id, str) or not plan_id:
        return None
    if not isinstance(bundle_id, str) or not bundle_id:
        raise VenueE2ERuntimeError(
            "frozen Stage 4b plan lacks its report input bundle binding"
        )
    plan_directory = run_dir / "reports" / "plans" / plan_id
    plan = _load_object(plan_directory / "REPORT_PLAN.json")
    corpus = _load_object(plan_directory / "CORPUS_SNAPSHOT.json")
    audit = _load_object(plan_directory / "SEARCH_AUDIT.json")
    stage_plan = _load_object(run_dir / "stage4b" / "REPORT_PLAN.json")
    if canonical_json(plan) != canonical_json(stage_plan):
        raise VenueE2ERuntimeError(
            "Stage 4b approved plan differs from its immutable plan-store copy"
        )
    if (
        plan.get("plan_id") != plan_id
        or plan.get("plan_hash") != stage4b.get("report_plan_hash")
        or plan.get("corpus_snapshot_hash") != corpus.get("snapshot_hash")
        or plan.get("search_audit_pack_hash") != audit.get("pack_hash")
    ):
        raise VenueE2ERuntimeError("frozen Stage 4b input bindings have drifted")
    return plan, corpus, audit, bundle_id


def _timestamp(value: datetime) -> str:
    return value.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _explicit_path(value: str | Path | None) -> Path | None:
    return None if value is None else Path(value).expanduser().resolve()


def _manifest_path(value: object, run_dir: Path) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value).expanduser()
    return (run_dir / path).resolve() if not path.is_absolute() else path.resolve()


def _resource_path(config_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    configured = (config_path.parent / path).resolve()
    if configured.is_file():
        return configured
    return (ROOT / path).resolve()


def _file_hash(path: Path | None) -> str | None:
    return sha256(path.read_bytes()).hexdigest() if path is not None else None


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return normalized[:48] or "venue"


def _native_handoff(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    run_path = run_dir / "run.json"
    run = _load_object(run_path) if run_path.is_file() else {}
    native = run.get("native_pipeline", {})
    if not isinstance(native, Mapping):
        raise VenueE2ERuntimeError("run.json native_pipeline must be an object")
    return run, dict(native)


def _venue_details(
    run_dir: Path,
    run: Mapping[str, Any],
    venue: str | None,
    topic: str | None,
    year: int | None,
) -> tuple[str, str, int]:
    snapshot_path = run_dir / "stage1" / "approved-snapshot.json"
    snapshot = _load_object(snapshot_path) if snapshot_path.is_file() else {}
    frozen_venue = snapshot.get("venue", {})
    if not isinstance(frozen_venue, Mapping):
        frozen_venue = {}
    selected_venue = venue or _string(run.get("venue")) or _string(frozen_venue.get("id"))
    matrix_venue = _matrix_venue(selected_venue)
    selected_topic = (
        topic
        or _string(frozen_venue.get("topic"))
        or _string(matrix_venue.get("topic"))
    )
    raw_year = (
        year
        if year is not None
        else frozen_venue.get("year", matrix_venue.get("year"))
    )
    if not selected_venue:
        raise VenueE2ERuntimeError("venue is required (argument or frozen Stage 1 snapshot)")
    if not selected_topic:
        raise VenueE2ERuntimeError("topic is required (argument or frozen Stage 1 snapshot)")
    if isinstance(raw_year, bool) or not isinstance(raw_year, int):
        raise VenueE2ERuntimeError("year is required (argument or frozen Stage 1 snapshot)")
    return selected_venue, selected_topic, raw_year


def _matrix_venue(venue: str | None) -> Mapping[str, Any]:
    """Use the curated matrix only to fill descriptive topic/year metadata."""
    path = ROOT / "configs" / "e2e" / "venue-smoke-matrix.yaml"
    if venue is None or not path.is_file():
        return {}
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    if not isinstance(document, Mapping) or not isinstance(document.get("venues"), list):
        return {}
    return next(
        (
            item
            for item in document["venues"]
            if isinstance(item, Mapping) and item.get("id") == venue
        ),
        {},
    )


def _string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _required_binding(
    name: str,
    explicit: str | None,
    native: Mapping[str, Any],
) -> str:
    value = explicit or _string(native.get(name))
    if value is None:
        raise VenueE2ERuntimeError(
            f"missing {name}; pass it explicitly or freeze it in run.json native_pipeline"
        )
    return value


def _pipeline_ids(
    run_id: str,
    native: Mapping[str, Any],
    *,
    stage4_run_id: str | None,
    report_run_id: str | None,
    report_pipeline_run_id: str | None,
) -> tuple[str, str, str]:
    selected_stage4 = _immutable_tail_binding(
        "stage4_run_id", stage4_run_id, native, f"{run_id}:stage4"
    )
    selected_report = _immutable_tail_binding(
        "report_run_id", report_run_id, native, f"{run_id}:report"
    )
    selected_report_pipeline = _immutable_tail_binding(
        "report_pipeline_run_id",
        report_pipeline_run_id,
        native,
        f"{run_id}:stage4b",
    )
    return selected_stage4, selected_report, selected_report_pipeline


def _immutable_tail_binding(
    name: str,
    explicit: str | None,
    native: Mapping[str, Any],
    default: str,
) -> str:
    """Resolve a paid-stage run ID without permitting manifest rebinding."""
    if explicit is not None and _string(explicit) is None:
        raise VenueE2ERuntimeError(f"explicit {name} must be a non-empty string")
    frozen_value = native.get(name)
    if frozen_value is not None:
        frozen = _string(frozen_value)
        if frozen is None:
            raise VenueE2ERuntimeError(
                f"run.json native_pipeline.{name} must be null or a non-empty string"
            )
        if explicit is not None and explicit != frozen:
            raise VenueE2ERuntimeError(
                f"explicit {name} conflicts with its frozen run.json binding"
            )
        return frozen
    return explicit or default


def _stage3_paper_ids(database: Database, stage3_run_id: str) -> tuple[str, ...]:
    run = database.connection.execute(
        "SELECT stage, status FROM pipeline_runs WHERE run_id = ?", (stage3_run_id,)
    ).fetchone()
    if run is None or tuple(run) != ("stage-3-download", "complete"):
        raise VenueE2ERuntimeError(
            "stage3_run_id must name a complete native Stage 3 download run"
        )
    rows = database.connection.execute(
        "SELECT paper_id, status FROM stage3_paper_results WHERE run_id = ? ORDER BY paper_id",
        (stage3_run_id,),
    ).fetchall()
    paper_ids = tuple(str(row["paper_id"]) for row in rows)
    if not paper_ids:
        raise VenueE2ERuntimeError("the selected Stage 3 run has no paper checkpoints")
    if any(row["status"] != "downloaded" for row in rows):
        raise VenueE2ERuntimeError(
            "venue E2E Stage 4 requires every Stage 3 checkpoint to be downloaded"
        )
    return paper_ids


def _model_runtime(
    config_path: Path | None,
    policy_override: Path | None,
    workers_override: int | None,
) -> tuple[
    ArtifactProcessingPolicy,
    Path,
    int,
    bool,
    Path | None,
    ReportResources,
    ReportRuntimeConfig,
]:
    config: Mapping[str, Any] | None = None
    if config_path is not None:
        config = load_config(config_path)
        analysis = config["analysis"]
        summary = config["summary"]
        expected = (
            analysis.get("profile"),
            analysis.get("model"),
            analysis.get("reasoning_effort"),
            summary.get("profile"),
            summary.get("model"),
            summary.get("reasoning_effort"),
            summary.get("execution_strategy"),
        )
        if expected != (
            "stage4_analysis_luna",
            "gpt-5.6-luna",
            "medium",
            "stage4b_oneshot_sol",
            "gpt-5.6-sol",
            "high",
            "one_shot",
        ):
            raise VenueE2ERuntimeError(
                "config must freeze Luna Stage 4 and high-reasoning one-shot Sol Stage 4b"
            )
        configured_runtime = ReportRuntimeConfig.from_config(config, config_path)
        resources = configured_runtime.resources
        analysis_schema = _resource_path(config_path, str(analysis["output_schema"]))
        policy_value = analysis["remote_model_processing"]["policy_matrix"]
        configured_policy = _resource_path(config_path, str(policy_value))
        workers = int(analysis["workers"])
        allow_abstract_only = bool(analysis["allow_abstract_only"])
    else:
        resources = ReportResources.defaults()
        analysis_schema = None
        configured_policy = DEFAULT_POLICY
        workers = 4
        allow_abstract_only = True
    policy_path = policy_override or configured_policy
    if not policy_path.is_file():
        raise VenueE2ERuntimeError(f"processing policy is missing: {policy_path}")
    if analysis_schema is not None and not analysis_schema.is_file():
        raise VenueE2ERuntimeError(f"analysis schema is missing: {analysis_schema}")
    selected_workers = workers if workers_override is None else workers_override
    if isinstance(selected_workers, bool) or selected_workers < 1:
        raise VenueE2ERuntimeError("workers must be a positive integer")
    policy = ArtifactProcessingPolicy.load(policy_path)
    # This explicit E2E entry point may run even when a broad smoke config keeps
    # summary.enabled=false.  The immutable one-shot plan remains the authority.
    report_runtime = ReportRuntimeConfig(
        enabled=True,
        resources=resources,
        profile="stage4b_oneshot_sol",
        execution_strategy="one_shot",
    )
    resources.validate_files()
    return (
        policy,
        policy_path,
        selected_workers,
        allow_abstract_only,
        analysis_schema,
        resources,
        report_runtime,
    )


def _initial_state(
    *,
    identity: Mapping[str, Any],
    approved_by: str,
    grant_ttl_hours: int,
    stage4_run_id: str,
    report_run_id: str,
    report_pipeline_run_id: str,
) -> dict[str, Any]:
    now = datetime.now(UTC).replace(microsecond=0)
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "identity": dict(identity),
        "created_at": _timestamp(now),
        "approved_by": approved_by,
        "approved_at": _timestamp(now),
        "grant_expires_at": _timestamp(now + timedelta(hours=grant_ttl_hours)),
        "models": {
            "stage4": {"profile": "stage4_analysis_luna", "model": "gpt-5.6-luna"},
            "stage4b": {
                "profile": "stage4b_oneshot_sol",
                "model": "gpt-5.6-sol",
                "execution_strategy": "one_shot",
            },
        },
        "stage4": {"run_id": stage4_run_id, "status": "pending", "invocations": 0},
        "stage4b": {
            "report_run_id": report_run_id,
            "pipeline_run_id": report_pipeline_run_id,
            "status": "pending",
            "invocations": 0,
        },
    }


def _load_or_create_state(
    state_path: Path,
    *,
    identity: Mapping[str, Any],
    resume: bool,
    approved_by: str,
    grant_ttl_hours: int,
    stage4_run_id: str,
    report_run_id: str,
    report_pipeline_run_id: str,
) -> dict[str, Any]:
    if state_path.exists():
        if not resume:
            raise VenueE2ERuntimeError(
                f"runtime state already exists; pass --resume: {state_path}"
            )
        state = _load_object(state_path)
        if state.get("schema_version") != STATE_SCHEMA_VERSION:
            raise VenueE2ERuntimeError("runtime state schema version has drifted")
        if state.get("identity") != dict(identity):
            raise VenueE2ERuntimeError("resume inputs differ from the frozen runtime identity")
        if state.get("approved_by") != approved_by:
            raise VenueE2ERuntimeError("resume approved_by differs from the frozen runtime")
        return state
    if resume:
        raise VenueE2ERuntimeError(f"cannot resume missing runtime state: {state_path}")
    state = _initial_state(
        identity=identity,
        approved_by=approved_by,
        grant_ttl_hours=grant_ttl_hours,
        stage4_run_id=stage4_run_id,
        report_run_id=report_run_id,
        report_pipeline_run_id=report_pipeline_run_id,
    )
    _write_state(state_path, state)
    return state


def _grant_scope(
    *,
    paper_ids: Sequence[str],
    artifact_hashes: Sequence[str],
    model: str,
    data_categories: Sequence[str],
) -> dict[str, Any]:
    return {
        "paper_ids": sorted(set(paper_ids)),
        "artifact_hashes": sorted(set(artifact_hashes)),
        "collection_ids": [],
        "collection_snapshot_hash": None,
        "selection_snapshot_hash": None,
        "domains": [],
        "provider": "codex_cli",
        "model": model,
        "data_categories": sorted(set(data_categories)),
    }


def _approve_exact_grant(
    store: GrantStore,
    directory: Path,
    *,
    grant_id: str,
    purpose: str,
    scope: Mapping[str, Any],
    max_papers: int,
    expires_at: str,
    approved_by: str,
    approved_at: str,
    lineage_hash: str | None = None,
) -> dict[str, Any]:
    draft = create_grant_draft(
        kind="remote_model_processing",
        actions=["remote_model_processing"],
        purpose=purpose,
        mode="attended",
        allow_unattended=False,
        scope=scope,
        max_papers=max_papers,
        expires_at=expires_at,
        lineage_hash=lineage_hash,
        grant_id=grant_id,
    )
    _write_immutable(directory / f"{grant_id}.draft.json", draft)
    existing = store.database.connection.execute(
        "SELECT grant_id FROM authorization_grants WHERE grant_id = ?", (grant_id,)
    ).fetchone()
    if existing is None:
        approved = store.approve(
            draft,
            str(draft["content_hash"]),
            approved_by=approved_by,
            approved_at=approved_at,
        )
    else:
        active = store.load(grant_id)
        approved = active.document
        if approved["content_hash"] != draft["content_hash"]:
            raise VenueE2ERuntimeError(f"persisted grant binding has drifted: {grant_id}")
    _write_immutable(directory / f"{grant_id}.json", approved)
    return approved


def _stage4_input_bindings(requests: Sequence[Any]) -> tuple[dict[str, str], ...]:
    """Freeze the exact full-text payload identity used by the venue E2E."""
    bindings: list[dict[str, str]] = []
    for request in requests:
        if request.input_scope != "full_pdf" or request.data_category not in {
            "full_text",
            "normalized_text",
        }:
            raise VenueE2ERuntimeError(
                "venue E2E Stage 4 requires full_pdf input for every selected paper"
            )
        paper_id = _string(request.paper_id)
        if paper_id is None or _SHA256.fullmatch(request.artifact_hash) is None:
            raise VenueE2ERuntimeError(
                "venue E2E Stage 4 input lacks an exact paper or artifact binding"
            )
        bindings.append(
            {
                "paper_id": paper_id,
                "artifact_hash": request.artifact_hash,
                "input_scope": request.input_scope,
                "data_category": request.data_category,
            }
        )
    bindings.sort(key=lambda item: item["paper_id"])
    if len({item["paper_id"] for item in bindings}) != len(bindings):
        raise VenueE2ERuntimeError("Stage 4 input bindings contain duplicate paper IDs")
    return tuple(bindings)


def _frozen_stage4_input_bindings(value: object) -> tuple[dict[str, str], ...]:
    if not isinstance(value, list) or not value:
        raise VenueE2ERuntimeError("frozen Stage 4 inputs must be a non-empty array")
    expected_keys = {"paper_id", "artifact_hash", "input_scope", "data_category"}
    bindings: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, Mapping) or set(item) != expected_keys:
            raise VenueE2ERuntimeError("frozen Stage 4 input binding is malformed")
        if not all(isinstance(item[key], str) for key in expected_keys):
            raise VenueE2ERuntimeError("frozen Stage 4 input binding is malformed")
        if (
            not item["paper_id"]
            or _SHA256.fullmatch(item["artifact_hash"]) is None
            or item["input_scope"] != "full_pdf"
            or item["data_category"] not in {"full_text", "normalized_text"}
        ):
            raise VenueE2ERuntimeError(
                "frozen Stage 4 inputs do not prove exact full_pdf processing"
            )
        bindings.append({key: str(item[key]) for key in expected_keys})
    bindings.sort(key=lambda item: item["paper_id"])
    if len({item["paper_id"] for item in bindings}) != len(bindings):
        raise VenueE2ERuntimeError("frozen Stage 4 inputs contain duplicate paper IDs")
    return tuple(bindings)


def _stage4_dispatch_summary(
    database: Database,
    run_id: str,
    expected_inputs: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    rows = database.connection.execute(
        """SELECT paper_id, artifact_hash, input_scope, status,
                  dispatch_count, profile, model_id
             FROM analysis_dispatches WHERE run_id = ? ORDER BY paper_id""",
        (run_id,),
    ).fetchall()
    expected = {
        str(item["paper_id"]): (
            str(item["artifact_hash"]),
            str(item["input_scope"]),
        )
        for item in expected_inputs
    }
    actual = {
        str(row["paper_id"]): (str(row["artifact_hash"]), str(row["input_scope"]))
        for row in rows
    }
    if actual != expected:
        raise VenueE2ERuntimeError(
            "Stage 4 dispatch artifact/input bindings do not exactly match frozen inputs"
        )
    if any(
        tuple(row[key] for key in ("status", "dispatch_count", "profile", "model_id"))
        != ("complete", 1, "stage4_analysis_luna", "gpt-5.6-luna")
        for row in rows
    ):
        raise VenueE2ERuntimeError(
            "Stage 4 requires one completed gpt-5.6-luna dispatch per selected paper"
        )
    return {
        "papers": len(rows),
        "invocations": sum(int(row["dispatch_count"]) for row in rows),
        "model": "gpt-5.6-luna",
        "input_scope": "full_pdf",
    }


def _one_shot_summary(
    database: Database,
    artifact_store: ArtifactStore,
    report_run_id: str,
) -> dict[str, Any]:
    row = database.connection.execute(
        """SELECT os.status, os.dispatch_count, os.budget_calls_reserved,
                  os.profile, os.model_id, os.reasoning_effort,
                  os.invocation_id, os.output_artifact_id, os.output_hash,
                  a.artifact_id, a.paper_id, a.artifact_kind, a.relative_path,
                  a.mime_type, a.byte_size, a.sha256, a.processing_status
             FROM report_one_shot_runs AS os
             LEFT JOIN artifacts AS a ON a.artifact_id = os.output_artifact_id
            WHERE os.report_run_id = ?""",
        (report_run_id,),
    ).fetchone()
    expected = (
        "complete",
        1,
        1,
        "stage4b_oneshot_sol",
        "gpt-5.6-sol",
        "high",
    )
    if row is None or tuple(
        row[key]
        for key in (
            "status",
            "dispatch_count",
            "budget_calls_reserved",
            "profile",
            "model_id",
            "reasoning_effort",
        )
    ) != expected or not row["invocation_id"]:
        raise VenueE2ERuntimeError(
            "Stage 4b must contain exactly one completed high-reasoning gpt-5.6-sol dispatch"
        )
    output_hash = row["output_hash"]
    if (
        not isinstance(output_hash, str)
        or _SHA256.fullmatch(output_hash) is None
        or row["output_artifact_id"] != row["artifact_id"]
        or row["paper_id"] is not None
        or row["artifact_kind"] != "report"
        or row["relative_path"] != artifact_store.relative_path(output_hash)
        or row["mime_type"] != "application/json"
        or row["sha256"] != output_hash
        or row["processing_status"] != "available"
    ):
        raise VenueE2ERuntimeError(
            "Stage 4b output artifact is not exactly bound to the sole Sol dispatch"
        )
    try:
        payload = artifact_store.read_bytes(output_hash)
        document = json.loads(payload)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise VenueE2ERuntimeError(
            "Stage 4b output artifact bytes failed CAS validation"
        ) from error
    if (
        not isinstance(document, Mapping)
        or isinstance(row["byte_size"], bool)
        or row["byte_size"] != len(payload)
    ):
        raise VenueE2ERuntimeError(
            "Stage 4b output artifact metadata differs from its JSON payload"
        )
    invocations = database.connection.execute(
        """SELECT invocation_id, phase, node_key
             FROM report_sol_invocations
            WHERE report_run_id = ?
            ORDER BY invocation_id""",
        (report_run_id,),
    ).fetchall()
    if len(invocations) != 1 or tuple(
        invocations[0][key] for key in ("invocation_id", "phase", "node_key")
    ) != (row["invocation_id"], "reduce", ONE_SHOT_NODE_ID):
        raise VenueE2ERuntimeError(
            "Stage 4b must contain one exact Sol invocation-ledger binding"
        )
    return {"invocations": 1, "model": "gpt-5.6-sol", "strategy": "one_shot"}


def build_one_shot_report_draft(
    *,
    corpus_snapshot: Mapping[str, Any],
    venue: str,
    topic: str,
    year: int,
    recent_cutoff: str,
    policy_hash: str,
    resources: ReportResources,
    max_input_tokens: int,
) -> dict[str, Any]:
    """Build the complete zh-CN one-shot plan consumed by the existing compiler."""
    paper_ids = [str(paper["paper_id"]) for paper in corpus_snapshot["papers"]]
    all_sections = list(REPORT_SECTION_IDS)
    sections = [
        {
            "id": section_id,
            "title": SECTION_TITLES[section_id],
            "subquestion_ids": ["sq-main"],
            "target_words": 450 if section_id == "evidence_synthesis" else 220,
            "evidence_requirements": [
                "所有实质性结论必须绑定冻结的 claim 与 evidence",
                "不可比数值必须显式标注，不得推断补齐",
            ],
            "allowed_evidence_levels": [
                "full_text_direct",
                "full_text_inferred",
                "abstract_direct",
                "metadata_only",
                "corpus_stat",
            ],
        }
        for section_id in REPORT_SECTION_IDS
    ]
    return {
        "created_at": str(corpus_snapshot["created_at"]),
        "objective": f"基于冻结的 {venue} {year} 语料，对“{topic}”形成可追溯的中文综述。",
        "report_language": "zh-CN",
        "audience": "相关领域研究者与工程实践者",
        "primary_question": f"{venue} {year} 关于“{topic}”的工作提出了什么方法、证据与局限？",
        "subquestions": [
            {"id": "sq-main", "question": "各论文的方法、实验依据、可比边界和共同局限是什么？"}
        ],
        "synthesis_question": "在冻结的任务、数据和评测条件下，各项发现为何一致、冲突或不可直接比较？",
        "scope": {
            "date_from": f"{year:04d}-01-01",
            "date_to": f"{year:04d}-12-31",
            "venues": [venue],
            "document_types": ["article", "proceedings-article", "preprint"],
            "languages": ["en"],
            "inclusion_criteria": [f"与冻结主题“{topic}”直接相关"],
            "exclusion_criteria": ["与冻结主题无直接关系或缺少可审计元数据"],
        },
        "execution_strategy": "one_shot",
        "stage4b_config_hash": one_shot_config_hash(
            policy_hash, execution_mode="attended", resources=resources
        ),
        "stage4b_audit_config_hash": one_shot_validation_config_hash(),
        "aggregation": {"max_chunk_input_tokens": max_input_tokens, "reduce_output_tokens": 4096},
        "sections": sections,
        "classification_axes": list(CLASSIFICATION_AXES),
        "cohort_rules": {
            "recent_cutoff": recent_cutoff,
            "foundational_rule": "仅使用冻结语料中的显式用户种子",
            "peer_review_rule": "仅使用 canonical publication status",
            "study_setting_rule": "仅使用 Stage 4 证据支持的研究设置标签",
        },
        "paper_memberships": [
            {
                "paper_id": paper_id,
                "section_ids": all_sections,
                "primary_section_id": "evidence_synthesis",
                "coverage_disposition": "evidence",
                "coverage_reason": None,
                "resource_table_ids": [],
            }
            for paper_id in paper_ids
        ],
        "artifacts": {
            "comparison_tables": ["方法与实验协议可比性矩阵"],
            "trend_statistics": ["方法、任务与年份分布"],
            "resource_tables": ["代码、数据、模型与可复现资源"],
            "appendices": ["query manifest", "exclusion reasons", "coverage ledger", "claim ledger"],
        },
        "budget": {
            "max_sol_calls": 1,
            "max_input_tokens": max_input_tokens,
            "max_retries": 0,
            "audit_calls": 0,
            "repair_calls": 0,
        },
    }


def _approved_report_plan(
    database: Database,
    output_root: Path,
    draft: Mapping[str, Any],
    corpus: Mapping[str, Any],
    audit: Mapping[str, Any],
    *,
    approved_by: str,
    approved_at: str,
    resources: ReportResources,
) -> dict[str, Any]:
    compiled = compile_report_plan(
        draft,
        corpus_snapshot=corpus,
        search_audit_pack=audit,
        created_at=str(draft["created_at"]),
        resources=resources,
    )
    store = ReportPlanStore(output_root)
    _write_immutable(store.draft_path(str(compiled["plan_id"])), compiled)
    existing = database.connection.execute(
        "SELECT plan_json FROM report_plans WHERE content_hash = ?",
        (compiled["plan_hash"],),
    ).fetchone()
    if existing is None:
        approved = store.approve_and_save(
            compiled,
            str(compiled["plan_hash"]),
            approved_by=approved_by,
            approved_at=approved_at,
            corpus_snapshot=corpus,
            search_audit_pack=audit,
        )
    else:
        approved = json.loads(str(existing["plan_json"]))
        if not isinstance(approved, dict):
            raise VenueE2ERuntimeError("persisted ReportPlan is not a JSON object")
        require_valid_approval(approved, "plan_hash")
        if approved.get("plan_hash") != compiled["plan_hash"]:
            raise VenueE2ERuntimeError("persisted ReportPlan content hash has drifted")
        store.save_bundle(approved, corpus, audit)
    return approved


def execute_stage4_and_4b(
    run_dir: str | Path,
    venue: str | None = None,
    *,
    database_path: str | Path | None = None,
    artifact_root: str | Path | None = None,
    crawl_run_id: str | None = None,
    filter_run_id: str | None = None,
    stage3_run_id: str | None = None,
    stage4_run_id: str | None = None,
    report_run_id: str | None = None,
    report_pipeline_run_id: str | None = None,
    config_path: str | Path | None = None,
    policy_path: str | Path | None = None,
    topic: str | None = None,
    year: int | None = None,
    paper_ids: Sequence[str] = (),
    through_stage: str = "stage4b",
    resume: bool = False,
    execute_models: bool = False,
    approved_by: str = "venue-e2e-operator",
    grant_ttl_hours: int = 168,
    workers: int | None = None,
    include_needs_review: bool = False,
    max_report_input_tokens: int = 8_000_000,
    analysis_invoker_factory: Callable[[], AnalysisInvoker] | None = None,
    direct_invoker_factory: Callable[[], DirectReportInvoker] | None = None,
) -> dict[str, Any]:
    """Prepare or execute Stage 4/4b from exact persisted upstream run IDs.

    Real ``CodexExec`` invokers are used when factories are omitted, but model
    dispatch remains impossible unless ``execute_models`` is true.
    """
    selected_run_dir = Path(run_dir).expanduser().resolve()
    selected_run_dir.mkdir(parents=True, exist_ok=True)
    run, native = _native_handoff(selected_run_dir)
    selected_venue, selected_topic, selected_year = _venue_details(
        selected_run_dir, run, venue, topic, year
    )
    run_id = _string(run.get("run_id")) or selected_run_dir.name

    explicit_database = _explicit_path(database_path)
    selected_database = explicit_database or _manifest_path(native.get("database"), selected_run_dir)
    explicit_artifacts = _explicit_path(artifact_root)
    selected_artifacts = explicit_artifacts or _manifest_path(
        native.get("artifact_root"), selected_run_dir
    )
    if selected_database is None:
        raise VenueE2ERuntimeError(
            "missing database; pass --database or freeze native_pipeline.database"
        )
    if selected_artifacts is None:
        raise VenueE2ERuntimeError(
            "missing artifact root; pass --artifact-root or freeze native_pipeline.artifact_root"
        )
    if not selected_database.is_file():
        raise VenueE2ERuntimeError(f"database does not exist: {selected_database}")
    selected_crawl = _required_binding("crawl_run_id", crawl_run_id, native)
    selected_filter = _required_binding("filter_run_id", filter_run_id, native)
    selected_stage3 = _required_binding("stage3_run_id", stage3_run_id, native)
    stage4_run_id, report_run_id, report_pipeline_run_id = _pipeline_ids(
        run_id,
        native,
        stage4_run_id=stage4_run_id,
        report_run_id=report_run_id,
        report_pipeline_run_id=report_pipeline_run_id,
    )

    selected_config = _explicit_path(config_path) or _manifest_path(
        native.get("config"), selected_run_dir
    )
    selected_policy_override = _explicit_path(policy_path)
    (
        policy,
        selected_policy,
        selected_workers,
        allow_abstract_only,
        analysis_schema,
        resources,
        report_runtime,
    ) = _model_runtime(selected_config, selected_policy_override, workers)
    if through_stage not in {"stage4", "stage4b"}:
        raise VenueE2ERuntimeError("through_stage must be stage4 or stage4b")
    if not approved_by:
        raise VenueE2ERuntimeError("approved_by is required")
    if isinstance(grant_ttl_hours, bool) or grant_ttl_hours < 1:
        raise VenueE2ERuntimeError("grant_ttl_hours must be positive")
    if isinstance(max_report_input_tokens, bool) or max_report_input_tokens < 1:
        raise VenueE2ERuntimeError("max_report_input_tokens must be positive")

    state_path = selected_run_dir / STATE_PATH
    artifact_store = ArtifactStore(selected_artifacts)
    with Database(selected_database) as database:
        database.migrate()
        frozen_stage3_ids = _stage3_paper_ids(database, selected_stage3)
        expected_paper_ids = tuple(sorted(set(paper_ids))) if paper_ids else frozen_stage3_ids
        if expected_paper_ids != frozen_stage3_ids:
            raise VenueE2ERuntimeError(
                "explicit paper IDs must exactly match the complete Stage 3 checkpoints"
            )
        identity = {
            "run_id": run_id,
            "venue": selected_venue,
            "topic": selected_topic,
            "year": selected_year,
            "database": str(selected_database),
            "artifact_root": str(selected_artifacts),
            "crawl_run_id": selected_crawl,
            "filter_run_id": selected_filter,
            "stage3_run_id": selected_stage3,
            "stage4_run_id": stage4_run_id,
            "report_run_id": report_run_id,
            "report_pipeline_run_id": report_pipeline_run_id,
            "paper_ids": list(expected_paper_ids),
            "config": str(selected_config) if selected_config is not None else None,
            "config_sha256": _file_hash(selected_config),
            "policy": str(selected_policy),
            "policy_hash": policy.hash,
            "analysis_schema": str(analysis_schema) if analysis_schema is not None else str(
                schema_directory() / "paper-analysis.schema.json"
            ),
            "analysis_schema_sha256": _file_hash(
                analysis_schema or (schema_directory() / "paper-analysis.schema.json")
            ),
            "workers": selected_workers,
            "allow_abstract_only": allow_abstract_only,
            "include_needs_review": include_needs_review,
            "max_report_input_tokens": max_report_input_tokens,
        }
        state = _load_or_create_state(
            state_path,
            identity=identity,
            resume=resume,
            approved_by=approved_by,
            grant_ttl_hours=grant_ttl_hours,
            stage4_run_id=stage4_run_id,
            report_run_id=report_run_id,
            report_pipeline_run_id=report_pipeline_run_id,
        )
        grant_store = GrantStore(database)
        analysis_options: dict[str, Any] = {}
        if analysis_invoker_factory is not None:
            analysis_options["invoker_factory"] = analysis_invoker_factory
        analysis_service = AnalysisCliService(
            database,
            artifact_store,
            policy,
            grants=grant_store,
            workers=selected_workers,
            # This acceptance tail proves the PDF path.  A public abstract or
            # metadata fallback may be useful in production, but cannot count
            # as a successful per-venue end-to-end test.
            allow_abstract_only=False,
            output_schema_path=analysis_schema,
            **analysis_options,
        )

        # Local extraction is required before approval because the grant must
        # bind the exact normalized-text hash.  This private adapter is the same
        # deterministic path used by run_from_stage3 and performs no model work.
        stage4_inputs = analysis_service._stage3_inputs(  # noqa: SLF001
            selected_stage3,
            expected_paper_ids=expected_paper_ids,
            preview=False,
        )
        requests = tuple(item.processing_request() for item in stage4_inputs)
        current_input_bindings = _stage4_input_bindings(requests)
        stage4_state = state.get("stage4")
        if not isinstance(stage4_state, dict):
            raise VenueE2ERuntimeError("runtime state Stage 4 entry is malformed")
        frozen_input_value = stage4_state.get("inputs")
        if frozen_input_value is not None:
            frozen_input_bindings = _frozen_stage4_input_bindings(
                frozen_input_value
            )
            if frozen_input_bindings != current_input_bindings:
                raise VenueE2ERuntimeError(
                    "current Stage 4 artifact/input bindings differ from frozen inputs"
                )
        elif resume and stage4_state.get("status") == "complete":
            raise VenueE2ERuntimeError(
                "completed Stage 4 runtime state lacks frozen input bindings"
            )
        stage4_row = database.connection.execute(
            "SELECT status FROM pipeline_runs WHERE run_id = ?", (stage4_run_id,)
        ).fetchone()
        stage4_complete = stage4_row is not None and stage4_row["status"] == "complete"
        if stage4_state.get("status") == "complete" and not stage4_complete:
            raise VenueE2ERuntimeError(
                "completed Stage 4 runtime state conflicts with its pipeline ledger"
            )

        if stage4_complete:
            # Verify the paid dispatch against both the freshly resolved input
            # and the immutable runtime binding before writing any state.
            stage4_summary = _stage4_dispatch_summary(
                database, stage4_run_id, current_input_bindings
            )
            if frozen_input_value is None:
                stage4_state["inputs"] = [
                    dict(item) for item in current_input_bindings
                ]
            stage4_state.update({"status": "complete", **stage4_summary})
            _write_state(state_path, state)
        else:
            stage4_scope = _grant_scope(
                paper_ids=[str(request.paper_id) for request in requests],
                artifact_hashes=[request.artifact_hash for request in requests],
                model="gpt-5.6-luna",
                data_categories=[request.data_category for request in requests],
            )
            luna_grant_digest = content_hash(
                {
                    "scope": stage4_scope,
                    "expires_at": state["grant_expires_at"],
                    "purpose": "internal_analysis",
                }
            )
            luna_grant_id = f"grant-stage4-luna-{luna_grant_digest[:24]}"
            luna_grant = _approve_exact_grant(
                grant_store,
                selected_run_dir / "stage4" / "grants",
                grant_id=luna_grant_id,
                purpose="internal_analysis",
                scope=stage4_scope,
                max_papers=len(expected_paper_ids),
                expires_at=str(state["grant_expires_at"]),
                approved_by=approved_by,
                approved_at=str(state["approved_at"]),
            )
            decisions = tuple(
                analysis_service.gate.decide(
                    request,
                    processing_grant_id=luna_grant_id,
                    now=str(state["approved_at"]),
                    paper_count=len(expected_paper_ids),
                )
                for request in requests
            )
            if not all(decision.is_authorized for decision in decisions):
                reason = next(
                    decision.reason_code
                    for decision in decisions
                    if not decision.is_authorized
                )
                raise VenueE2ERuntimeError(
                    f"Stage 4 exact-grant preflight failed: {reason}"
                )
            if frozen_input_value is None:
                stage4_state["inputs"] = [
                    dict(item) for item in current_input_bindings
                ]
            stage4_state.update(
                {
                    "status": "ready",
                    "grant_id": luna_grant["grant_id"],
                    "grant_content_hash": luna_grant["content_hash"],
                }
            )
            _write_state(state_path, state)

        if execute_models and not stage4_complete:
            result = analysis_service.run_from_stage3(
                stage4_run_id,
                selected_stage3,
                expected_paper_ids=expected_paper_ids,
                processing_grant_id=luna_grant_id,
                dry_run=False,
            )
            if result.result is None or any(
                paper.status != "complete" for paper in result.result.papers
            ):
                raise VenueE2ERuntimeError("Stage 4 did not complete every selected paper")
            stage4_complete = True
        if not stage4_complete:
            return _runtime_result(
                state,
                selected_run_dir,
                through_stage=through_stage,
                status="ready_for_stage4",
                execute_models=execute_models,
            )
        if stage4_state.get("status") != "complete":
            stage4_summary = _stage4_dispatch_summary(
                database, stage4_run_id, current_input_bindings
            )
            stage4_state.update({"status": "complete", **stage4_summary})
            _write_state(state_path, state)
        if through_stage == "stage4":
            return _runtime_result(
                state,
                selected_run_dir,
                through_stage=through_stage,
                status="complete",
                execute_models=execute_models,
            )

        recent_cutoff = f"{selected_year:04d}-01-01"
        frozen_report_inputs = _frozen_report_inputs(
            selected_run_dir, state, resume=resume
        )
        if frozen_report_inputs is None:
            report_inputs = ReportInputService(
                database, artifact_store, selected_run_dir
            ).build(
                ReportInputRequest(
                    crawl_run_id=selected_crawl,
                    filter_run_id=selected_filter,
                    stage4_run_id=stage4_run_id,
                    recent_cutoff=recent_cutoff,
                    created_at=str(state["created_at"]),
                    include_needs_review=include_needs_review,
                ),
                save_bundle=True,
            )
            corpus_snapshot = report_inputs.corpus_snapshot
            search_audit_pack = report_inputs.search_audit
            report_input_bundle_id = report_inputs.bundle_id
            draft = build_one_shot_report_draft(
                corpus_snapshot=corpus_snapshot,
                venue=selected_venue,
                topic=selected_topic,
                year=selected_year,
                recent_cutoff=recent_cutoff,
                policy_hash=policy.hash,
                resources=resources,
                max_input_tokens=max_report_input_tokens,
            )
            _write_immutable(
                selected_run_dir / "stage4b" / "REPORT_DRAFT_ONE_SHOT.json",
                draft,
            )
            approved_plan = _approved_report_plan(
                database,
                selected_run_dir,
                draft,
                corpus_snapshot,
                search_audit_pack,
                approved_by=approved_by,
                approved_at=str(state["approved_at"]),
                resources=resources,
            )
            _write_immutable(
                selected_run_dir / "stage4b" / "REPORT_PLAN.json", approved_plan
            )
        else:
            (
                approved_plan,
                corpus_snapshot,
                search_audit_pack,
                report_input_bundle_id,
            ) = frozen_report_inputs
        corpus_ids = tuple(
            str(paper["paper_id"]) for paper in corpus_snapshot["papers"]
        )
        if corpus_ids != expected_paper_ids or any(
            paper.get("input_scope") != "full_pdf"
            for paper in corpus_snapshot["papers"]
        ):
            raise VenueE2ERuntimeError(
                "report corpus must exactly match complete full_pdf Stage 4 membership"
            )

        sol_grants: dict[str, str] = {}
        sol_grant_hashes: dict[str, str] = {}
        for paper in corpus_snapshot["papers"]:
            artifact_hash = str(paper["analysis_artifact_hash"])
            if _SHA256.fullmatch(artifact_hash) is None:
                raise VenueE2ERuntimeError("report corpus analysis artifact lacks a SHA-256")
            lineage_hash = content_hash(tuple(sorted(str(value) for value in paper["lineage_hashes"])))
            scope = _grant_scope(
                paper_ids=[str(paper["paper_id"])],
                artifact_hashes=[artifact_hash],
                model="gpt-5.6-sol",
                data_categories=["analysis"],
            )
            grant_digest = content_hash(
                {
                    "scope": scope,
                    "lineage_hash": lineage_hash,
                    "expires_at": state["grant_expires_at"],
                    "purpose": "research_synthesis",
                }
            )
            grant_id = f"grant-stage4b-sol-{grant_digest[:24]}"
            approved_grant = _approve_exact_grant(
                grant_store,
                selected_run_dir / "stage4b" / "grants",
                grant_id=grant_id,
                purpose="research_synthesis",
                scope=scope,
                max_papers=1,
                expires_at=str(state["grant_expires_at"]),
                approved_by=approved_by,
                approved_at=str(state["approved_at"]),
                lineage_hash=lineage_hash,
            )
            sol_grants[artifact_hash] = grant_id
            sol_grant_hashes[artifact_hash] = str(approved_grant["content_hash"])
        grant_map = {"schema_version": "1", "grants": sol_grants}
        _write_immutable(selected_run_dir / "stage4b" / "processing-grants.json", grant_map)

        gate = ProcessingGate(policy, grant_store)
        report_options: dict[str, Any] = {}
        if direct_invoker_factory is not None:
            report_options["direct_invoker_factory"] = direct_invoker_factory
        report_service = ReportExecutionService(
            database,
            artifact_store,
            gate,
            ReportArtifactStore(selected_run_dir),
            execution_mode="attended",
            runtime_config=report_runtime,
            **report_options,
        )
        bundle = ReportPlanBundle(
            approved_plan, corpus_snapshot, search_audit_pack
        )
        preflight = report_service.run(
            report_run_id,
            report_pipeline_run_id,
            bundle,
            processing_grants=sol_grants,
            dry_run=True,
        )
        if preflight.status != "validated":
            message = preflight.error["message"] if preflight.error else preflight.status
            raise VenueE2ERuntimeError(f"one-shot Sol preflight failed: {message}")
        state["stage4b"].update(
            {
                "status": "ready",
                "report_input_bundle_id": report_input_bundle_id,
                "report_plan_id": approved_plan["plan_id"],
                "report_plan_hash": approved_plan["plan_hash"],
                "processing_grants": sol_grants,
                "processing_grant_hashes": sol_grant_hashes,
                "approved_call_limit": 1,
                "max_retries": 0,
            }
        )
        _write_state(state_path, state)
        if not execute_models:
            return _runtime_result(
                state,
                selected_run_dir,
                through_stage=through_stage,
                status="ready_for_stage4b",
                execute_models=False,
            )

        report_result = report_service.run(
            report_run_id,
            report_pipeline_run_id,
            bundle,
            processing_grants=sol_grants,
            dry_run=False,
        )
        if report_result.status != "complete":
            message = report_result.error["message"] if report_result.error else report_result.status
            raise VenueE2ERuntimeError(f"one-shot Sol report did not complete: {message}")
        sol_summary = _one_shot_summary(database, artifact_store, report_run_id)
        verification = verify_report_run(selected_run_dir, report_run_id)
        _write_immutable(selected_run_dir / "stage4b" / "VERIFICATION.json", verification)
        state["stage4b"].update(
            {
                "status": "complete",
                **sol_summary,
                "verification": "passed",
                "verification_path": str(selected_run_dir / "stage4b" / "VERIFICATION.json"),
                "published_path": (
                    str(report_result.direct.published_path)
                    if report_result.direct and report_result.direct.published_path
                    else None
                ),
            }
        )
        _write_state(state_path, state)
        return _runtime_result(
            state,
            selected_run_dir,
            through_stage=through_stage,
            status="complete",
            execute_models=True,
        )


def _runtime_result(
    state: Mapping[str, Any],
    run_dir: Path,
    *,
    through_stage: str,
    status: str,
    execute_models: bool,
) -> dict[str, Any]:
    _update_run_manifest(run_dir, state, status=status)
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "run_id": state["identity"]["run_id"],
        "run_dir": str(run_dir),
        "venue": state["identity"]["venue"],
        "through_stage": through_stage,
        "execute_models": execute_models,
        "status": status,
        "stage4": dict(state["stage4"]),
        "stage4b": dict(state["stage4b"]),
        "state_path": str(run_dir / STATE_PATH),
    }


def _update_run_manifest(
    run_dir: Path, state: Mapping[str, Any], *, status: str
) -> None:
    """Publish exact tail bindings without guessing or discarding Stage 1--3 data."""
    path = run_dir / "run.json"
    document = _load_object(path) if path.is_file() else {
        "schema_version": STATE_SCHEMA_VERSION,
        "run_id": state["identity"]["run_id"],
        "venue": state["identity"]["venue"],
        "stages": [],
    }
    native = document.get("native_pipeline")
    if not isinstance(native, dict):
        native = {}
    native.update(
        {
            "database": native.get("database") or state["identity"]["database"],
            "artifact_root": native.get("artifact_root") or state["identity"]["artifact_root"],
            "crawl_run_id": state["identity"]["crawl_run_id"],
            "filter_run_id": state["identity"]["filter_run_id"],
            "stage3_run_id": state["identity"]["stage3_run_id"],
        }
    )
    for name in (
        "stage4_run_id",
        "report_run_id",
        "report_pipeline_run_id",
    ):
        frozen = native.get(name)
        selected = state["identity"][name]
        if frozen is not None and frozen != selected:
            raise VenueE2ERuntimeError(
                f"run.json native_pipeline.{name} conflicts with runtime state"
            )
        native[name] = selected
    document["native_pipeline"] = native
    document["stage4b_runtime_status"] = status
    stages = document.get("stages")
    if not isinstance(stages, list):
        stages = []
    by_stage = {
        item.get("stage"): item for item in stages if isinstance(item, dict)
    }
    stage4 = by_stage.setdefault("stage4", {"stage": "stage4"})
    stage4.update(
        {
            "status": _published_stage_status(str(state["stage4"]["status"])),
            "model": "gpt-5.6-luna",
            "invocations": int(state["stage4"].get("invocations", 0)),
        }
    )
    stage4b = by_stage.setdefault("stage4b", {"stage": "stage4b"})
    stage4b.update(
        {
            "status": _published_stage_status(str(state["stage4b"]["status"])),
            "model": "gpt-5.6-sol",
            "invocations": int(state["stage4b"].get("invocations", 0)),
            "execution_strategy": "one_shot",
        }
    )
    existing_names = {
        item.get("stage") for item in stages if isinstance(item, dict)
    }
    for name in ("stage4", "stage4b"):
        if name not in existing_names:
            stages.append(by_stage[name])
    document["stages"] = stages
    _atomic_write(path, canonical_json(document))


def _published_stage_status(status: str) -> str:
    if status == "complete":
        return "complete"
    if status == "ready":
        return "ready_for_explicit_execution"
    return "blocked_pending_upstream_stage"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--venue")
    parser.add_argument("--topic")
    parser.add_argument("--year", type=int)
    parser.add_argument("--database", type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--crawl-run-id")
    parser.add_argument("--filter-run-id")
    parser.add_argument("--stage3-run-id")
    parser.add_argument("--stage4-run-id")
    parser.add_argument("--report-run-id")
    parser.add_argument("--report-pipeline-run-id")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--paper-id", action="append", default=[])
    parser.add_argument("--through-stage", choices=("stage4", "stage4b"), default="stage4b")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--execute-models",
        action="store_true",
        help="cross the paid boundary; without this flag no Luna or Sol process is created",
    )
    parser.add_argument("--approved-by", default="venue-e2e-operator")
    parser.add_argument("--grant-ttl-hours", type=int, default=168)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--include-needs-review", action="store_true")
    parser.add_argument("--max-report-input-tokens", type=int, default=8_000_000)
    args = parser.parse_args(argv)
    try:
        result = execute_stage4_and_4b(
            args.run_dir,
            args.venue,
            database_path=args.database,
            artifact_root=args.artifact_root,
            crawl_run_id=args.crawl_run_id,
            filter_run_id=args.filter_run_id,
            stage3_run_id=args.stage3_run_id,
            stage4_run_id=args.stage4_run_id,
            report_run_id=args.report_run_id,
            report_pipeline_run_id=args.report_pipeline_run_id,
            config_path=args.config,
            policy_path=args.policy,
            topic=args.topic,
            year=args.year,
            paper_ids=args.paper_id,
            through_stage=args.through_stage,
            resume=args.resume,
            execute_models=args.execute_models,
            approved_by=args.approved_by,
            grant_ttl_hours=args.grant_ttl_hours,
            workers=args.workers,
            include_needs_review=args.include_needs_review,
            max_report_input_tokens=args.max_report_input_tokens,
        )
    except (VenueE2ERuntimeError, GrantError, ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
