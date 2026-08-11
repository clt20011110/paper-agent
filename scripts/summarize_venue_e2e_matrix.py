#!/usr/bin/env python3
"""Summarize isolated per-venue E2E evidence without running any pipeline stage.

Each immediate child of ``--run-root`` is treated as an independent venue run
directory.  A run directory must contain one SQLite database and the immutable
report bundle for its selected Stage 4b report.  When ``--run-root`` itself
contains a database it is treated as a single run directory.

The script opens SQLite in read-only mode and never invokes Paper Agent, Luna,
or Sol.  It deliberately distinguishes the one-shot invocation recorded under
the legacy ``report_sol_invocations.phase=reduce`` enum from actual reduce-tree
nodes: one-shot acceptance requires one total Sol ledger row and zero reduce,
audit, or repair step rows.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from hashlib import sha256
import json
from pathlib import Path
import re
import sqlite3
from typing import Any, Callable, Iterable, Mapping, Sequence

import yaml

from paper_agent.approval import ApprovalError, require_valid_approval
from paper_agent.canonical import canonical_json, content_hash
from paper_agent.report_artifacts import (
    ReportVerificationError,
    _validate_audit_binding,
    report_artifact_hash,
)
from paper_agent.report_cli_service import verify_report_run
from paper_agent.schema import validate


DATABASE_SUFFIXES = frozenset({".sqlite3", ".sqlite", ".db"})
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VENUE_CATALOG_ROOT = ROOT / "venues"
DEFAULT_ACCEPTANCE_IMPORT_MANIFEST = (
    ROOT / "configs" / "e2e" / "venue-e2e-acceptance-imports.json"
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
ACCEPTED_STAGE4B_IMPLEMENTATIONS = frozenset(
    {"stage4b-one-shot-v1", "stage4b-one-shot-v2", "stage4b-one-shot-v3"}
)
LEGACY_CLAIM_ORDER_IMPLEMENTATIONS = frozenset(
    {"stage4b-one-shot-v1", "stage4b-one-shot-v2"}
)
VERIFIER_VERSION = "paper-agent.venue-e2e-verifier.v2"
REQUIRED_REPORT_FILES = frozenset(
    {
        "REPORT_PLAN.json",
        "SEARCH_AUDIT.json",
        "CORPUS_SNAPSHOT.json",
        "CLAIMS_EVIDENCE.jsonl",
        "COMPARISON_GROUPS.json",
        "CLAIM_RELATIONS.json",
        "REPORT_DOCUMENT.json",
        "COVERAGE.json",
        "RESOURCE_TABLES.json",
        "BIBLIOGRAPHY.json",
        "REPORT_SIDECAR.json",
        "AUDIT.json",
        "VERIFICATION.json",
        "REPORT.md",
    }
)
STAGE_ALIASES = {
    "stage1": ("stage-1", "stage1", "search"),
    "stage2": ("stage-2", "stage2", "filter"),
    "stage3": ("stage-3-download", "stage3", "download"),
    "stage4": ("stage4", "stage-4", "analyze"),
    "stage4b": ("stage4b", "stage-4b", "report"),
}
CHECK_ORDER = (
    "stage1_complete",
    "stage2_test_only_relevant_one",
    "stage3_pdf_checkpoint_complete",
    "stage4_luna_invocation_one",
    "stage4b_sol_one_shot_only",
    "verify_passed",
)
CHECK_LABELS = {
    "stage1_complete": "Stage 1",
    "stage2_test_only_relevant_one": "Stage 2",
    "stage3_pdf_checkpoint_complete": "Stage 3 PDF",
    "stage4_luna_invocation_one": "Luna",
    "stage4b_sol_one_shot_only": "Sol one-shot",
    "verify_passed": "Verify",
}


class MatrixError(RuntimeError):
    """The requested evidence matrix cannot be assembled safely."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _check(passed: bool, expected: str, **actual: Any) -> dict[str, Any]:
    return {"passed": bool(passed), "expected": expected, **actual}


def _failed_check(expected: str, error: str) -> dict[str, Any]:
    return _check(False, expected, error=error)


def _database_candidates(run_dir: Path) -> tuple[Path, ...]:
    candidates = []
    for path in run_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in DATABASE_SUFFIXES:
            continue
        relative_parts = path.relative_to(run_dir).parts
        if any(part.startswith(".") for part in relative_parts):
            continue
        candidates.append(path)
    return tuple(sorted(candidates))


def _resolve_within_run(run_dir: Path, value: str | Path, *, label: str) -> Path:
    root = run_dir.resolve()
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise MatrixError(f"{label} escapes its venue run directory: {resolved}") from error
    return resolved


def _resolve_database(
    run_dir: Path,
    relative_path: Path | None,
    run_binding: Mapping[str, Any],
) -> Path:
    configured = run_binding.get("database")
    if not isinstance(configured, str) or not configured:
        raise MatrixError("run.json native_pipeline.database is required")
    bound = _resolve_within_run(run_dir, configured, label="run.json database")
    if relative_path is not None:
        override = _resolve_within_run(
            run_dir, relative_path, label="database-relative-path"
        )
        if override != bound:
            raise MatrixError(
                "database-relative-path conflicts with run.json native_pipeline.database"
            )
    if not bound.is_file():
        raise MatrixError(f"run.json database does not exist: {bound}")
    return bound


def _bound_string(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    if isinstance(value, Mapping):
        for key in ("run_id", "id", "path"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate:
                return candidate
    return None


def _load_run_binding(run_dir: Path) -> dict[str, Any]:
    """Read the runner's explicit paths and run IDs when ``run.json`` exists."""

    path = run_dir / "run.json"
    if not path.is_file():
        raise MatrixError("acceptance run lacks required run.json")
    document = _load_json_object(path)
    native = document.get("native_pipeline")
    if not isinstance(native, Mapping) or not native:
        raise MatrixError("run.json lacks required native_pipeline bindings")
    def select_native(*keys: str) -> str | None:
        for key in keys:
            selected = _bound_string(native.get(key))
            if selected is not None:
                return selected
        return None

    return {
        "venue": _bound_string(document.get("venue"))
        or select_native("venue", "venue_id"),
        "database": select_native("database", "database_path"),
        "artifact_root": select_native("artifact_root", "artifacts"),
        "output_root": select_native("output_root", "report_output_root"),
        "search_run_id": select_native("search_run_id", "search"),
        "crawl_run_id": select_native("crawl_run_id", "crawl"),
        "filter_run_id": select_native("filter_run_id", "filter", "stage2_run_id"),
        "stage3_run_id": select_native("stage3_run_id", "stage3", "download_run_id"),
        "stage4_run_id": select_native("stage4_run_id", "stage4", "analysis_run_id"),
        "report_run_id": select_native("report_run_id", "report", "stage4b_run_id"),
        "report_pipeline_run_id": select_native("report_pipeline_run_id"),
        "strict": True,
    }


def _resolve_bound_directory(
    run_dir: Path,
    configured: Any,
    *,
    label: str,
    default: Path,
) -> Path:
    candidate = configured if isinstance(configured, str) and configured else default
    resolved = _resolve_within_run(run_dir, candidate, label=label)
    if not resolved.is_dir():
        raise MatrixError(f"{label} does not exist: {resolved}")
    return resolved


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_integrity(
    artifact_root: Path,
    row: Mapping[str, Any],
    *,
    expected_kind: str,
    expected_mime_type: str,
    require_pdf_signature: bool = False,
) -> dict[str, Any]:
    artifact_id = row.get("artifact_id")
    artifact_hash = row.get("sha256")
    relative_path = row.get("relative_path")
    byte_size = row.get("byte_size")
    if not isinstance(artifact_id, str) or not artifact_id:
        raise MatrixError("artifact_id is missing")
    if not isinstance(artifact_hash, str) or SHA256_PATTERN.fullmatch(artifact_hash) is None:
        raise MatrixError(f"artifact {artifact_id} lacks a lowercase SHA-256")
    if row.get("paper_id") is None and expected_kind != "report":
        raise MatrixError(f"artifact {artifact_id} is not bound to a paper")
    if row.get("artifact_kind") != expected_kind:
        raise MatrixError(f"artifact {artifact_id} kind is not {expected_kind}")
    if row.get("mime_type") != expected_mime_type:
        raise MatrixError(
            f"artifact {artifact_id} MIME is not {expected_mime_type}"
        )
    if row.get("processing_status") != "available":
        raise MatrixError(f"artifact {artifact_id} is not available")
    if not isinstance(relative_path, str) or relative_path != (
        f"artifacts/{artifact_hash[:2]}/{artifact_hash}"
    ):
        raise MatrixError(f"artifact {artifact_id} has a non-canonical CAS path")
    if not isinstance(byte_size, int) or byte_size < 0:
        raise MatrixError(f"artifact {artifact_id} has an invalid byte_size")
    path = (artifact_root / relative_path).resolve()
    try:
        path.relative_to(artifact_root)
    except ValueError as error:
        raise MatrixError(f"artifact {artifact_id} escapes artifact_root") from error
    if path.is_symlink() or not path.is_file():
        raise MatrixError(f"artifact {artifact_id} CAS payload is unavailable")
    if path.stat().st_size != byte_size:
        raise MatrixError(f"artifact {artifact_id} byte_size does not match CAS")
    if _file_sha256(path) != artifact_hash:
        raise MatrixError(f"artifact {artifact_id} SHA-256 does not match CAS")
    if require_pdf_signature:
        with path.open("rb") as handle:
            if handle.read(5) != b"%PDF-":
                raise MatrixError(f"artifact {artifact_id} lacks a PDF signature")
    return {
        "artifact_id": artifact_id,
        "sha256": artifact_hash,
        "relative_path": relative_path,
        "byte_size": byte_size,
        "path": str(path),
    }


def discover_run_dirs(run_root: Path) -> tuple[Path, ...]:
    """Return deterministic venue run directories below ``run_root``."""

    root = run_root.resolve()
    if not root.is_dir():
        raise MatrixError(f"run root is not a directory: {root}")
    if _database_candidates(root):
        direct = tuple(
            path for path in root.iterdir()
            if path.is_file() and path.suffix.lower() in DATABASE_SUFFIXES
        )
        if direct:
            return (root,)
    children = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        if (
            (child / "run.json").is_file()
            or _database_candidates(child)
            or any(child.rglob("VERIFICATION.json"))
        ):
            children.append(child)
    if not children:
        raise MatrixError(f"no venue run directories found below {root}")
    return tuple(children)


def _connect_read_only(database: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(
        f"{database.resolve().as_uri()}?mode=ro",
        uri=True,
        isolation_level="DEFERRED",
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA busy_timeout=5000")
    return connection


def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
    return connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone() is not None


def _latest_pipeline_run(
    connection: sqlite3.Connection, aliases: Sequence[str]
) -> dict[str, Any] | None:
    placeholders = ",".join("?" for _ in aliases)
    row = connection.execute(
        f"""SELECT run_id, stage, status, input_hash, config_hash,
                      implementation_version, started_at, completed_at, created_at
              FROM pipeline_runs
             WHERE stage IN ({placeholders})
             ORDER BY COALESCE(completed_at, started_at, created_at) DESC,
                      created_at DESC, run_id DESC
             LIMIT 1""",
        tuple(aliases),
    ).fetchone()
    return dict(row) if row is not None else None


def _pipeline_run(
    connection: sqlite3.Connection,
    aliases: Sequence[str],
    run_id: str | None,
) -> dict[str, Any] | None:
    if run_id is None:
        return _latest_pipeline_run(connection, aliases)
    placeholders = ",".join("?" for _ in aliases)
    row = connection.execute(
        f"""SELECT run_id, stage, status, input_hash, config_hash,
                      implementation_version, started_at, completed_at, created_at
              FROM pipeline_runs
             WHERE run_id = ? AND stage IN ({placeholders})""",
        (run_id, *aliases),
    ).fetchone()
    return dict(row) if row is not None else None


def _stage1_check(
    connection: sqlite3.Connection,
    expected_venue: str,
    crawl_run_id: str | None = None,
    pipeline_run_id: str | None = None,
) -> dict[str, Any]:
    expected = (
        "the explicitly bound one-record Stage 1 crawl has an approved QueryPlan "
        "for this venue and a frozen paper membership"
    )
    for table in (
        "pipeline_runs",
        "crawl_runs",
        "search_plans",
        "crawl_paper_snapshots",
    ):
        if not _table_exists(connection, table):
            return _failed_check(expected, f"missing {table} table")
    bound_crawl = None
    if crawl_run_id is not None:
        row = connection.execute(
            """SELECT cr.crawl_run_id, cr.run_id, cr.status, cr.search_plan_id,
                      sp.content_hash AS query_plan_hash,
                      sp.status AS query_plan_status, sp.plan_json
                 FROM crawl_runs AS cr
                 LEFT JOIN search_plans AS sp
                   ON sp.search_plan_id = cr.search_plan_id
                WHERE cr.crawl_run_id = ?""",
            (crawl_run_id,),
        ).fetchone()
        bound_crawl = dict(row) if row is not None else None
        if bound_crawl is None:
            return _failed_check(
                expected,
                f"configured crawl_run_id does not exist: {crawl_run_id}",
            )
        if pipeline_run_id is not None and bound_crawl["run_id"] != pipeline_run_id:
            return _failed_check(
                expected,
                "configured crawl_run_id is not bound to native_pipeline.search_run_id",
            )
    run = _pipeline_run(
        connection,
        STAGE_ALIASES["stage1"],
        str(bound_crawl["run_id"]) if bound_crawl is not None else None,
    )
    if run is None:
        detail = f" for crawl_run_id={crawl_run_id}" if crawl_run_id else ""
        return _failed_check(expected, f"Stage 1 pipeline run not found{detail}")
    crawl = bound_crawl
    if crawl is None:
        row = connection.execute(
            """SELECT cr.crawl_run_id, cr.run_id, cr.status, cr.search_plan_id,
                      sp.content_hash AS query_plan_hash,
                      sp.status AS query_plan_status, sp.plan_json
                 FROM crawl_runs AS cr
                 LEFT JOIN search_plans AS sp
                   ON sp.search_plan_id = cr.search_plan_id
                WHERE cr.run_id = ?
                ORDER BY cr.started_at DESC, cr.crawl_run_id DESC LIMIT 1""",
            (run["run_id"],),
        ).fetchone()
        crawl = dict(row) if row is not None else None
    plan: Mapping[str, Any] = {}
    plan_error: str | None = None
    if crawl is not None:
        try:
            raw_plan = json.loads(str(crawl.get("plan_json") or ""))
            if not isinstance(raw_plan, Mapping):
                raise MatrixError("persisted QueryPlan is not an object")
            require_valid_approval(raw_plan, "plan_hash")
            if (
                crawl.get("query_plan_status") != "approved"
                or raw_plan.get("plan_id") != crawl.get("search_plan_id")
                or raw_plan.get("plan_hash") != crawl.get("query_plan_hash")
            ):
                raise MatrixError("persisted QueryPlan identity has drifted")
            scope = raw_plan.get("scope")
            venues = scope.get("venues") if isinstance(scope, Mapping) else None
            if venues != [expected_venue]:
                raise MatrixError(
                    "persisted QueryPlan venue scope does not match run.json"
                )
            plan = raw_plan
        except (json.JSONDecodeError, ApprovalError, MatrixError) as error:
            plan_error = str(error)
    membership_rows = connection.execute(
        """SELECT paper_id FROM crawl_paper_snapshots
             WHERE crawl_run_id = ? ORDER BY paper_id""",
        (crawl["crawl_run_id"] if crawl else "",),
    ).fetchall()
    paper_ids = [str(row["paper_id"]) for row in membership_rows]
    one_record = len(paper_ids) == 1 and len(set(paper_ids)) == 1
    passed = bool(
        run["status"] == "complete"
        and crawl is not None
        and crawl["status"] == "complete"
        and plan_error is None
        and plan
        and one_record
        and (pipeline_run_id is None or run["run_id"] == pipeline_run_id)
    )
    return _check(
        passed,
        expected,
        run_id=run["run_id"],
        configured_run_id=pipeline_run_id,
        pipeline_status=run["status"],
        crawl_run_id=crawl["crawl_run_id"] if crawl else None,
        crawl_status=crawl["status"] if crawl else None,
        query_plan_id=crawl.get("search_plan_id") if crawl else None,
        query_plan_hash=crawl.get("query_plan_hash") if crawl else None,
        query_plan_venues=(
            list(plan.get("scope", {}).get("venues", ())) if plan else None
        ),
        paper_ids=paper_ids,
        one_record_snapshot=one_record,
        error=plan_error,
    )


def _reason_marks_test_only(value: Any) -> bool:
    if isinstance(value, Mapping):
        if value.get("test_only") is True:
            return True
        return any(_reason_marks_test_only(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_reason_marks_test_only(item) for item in value)
    return False


def _decision_is_test_only(row: Mapping[str, Any]) -> bool:
    model_id = row.get("model_id")
    if isinstance(model_id, str) and (
        model_id.upper().startswith("TEST_ONLY/")
        or "test-only" in model_id.lower()
        or "test_only" in model_id.lower()
    ):
        return True
    reason = row.get("reason")
    if not isinstance(reason, str):
        return False
    try:
        return _reason_marks_test_only(json.loads(reason))
    except json.JSONDecodeError:
        return False


def _stage2_check(
    connection: sqlite3.Connection,
    stage1_paper_ids: Sequence[str],
    run_id: str | None = None,
) -> dict[str, Any]:
    expected = (
        "the bound Stage 2 run decides the complete Stage 1 membership and selects "
        "exactly one TEST_ONLY relevant paper"
    )
    for table in ("pipeline_runs", "filter_decisions"):
        if not _table_exists(connection, table):
            return _failed_check(expected, f"missing {table} table")
    run = _pipeline_run(connection, STAGE_ALIASES["stage2"], run_id)
    if run is None:
        return _failed_check(expected, "Stage 2 pipeline run not found")
    rows = [
        dict(row) for row in connection.execute(
            """SELECT paper_id, status, model_id, reason
                 FROM filter_decisions WHERE run_id = ? ORDER BY paper_id""",
            (run["run_id"],),
        ).fetchall()
    ]
    relevant = [row for row in rows if row["status"] == "relevant"]
    relevant_test_only = bool(relevant) and all(_decision_is_test_only(row) for row in relevant)
    decision_ids = [str(row["paper_id"]) for row in rows]
    frozen_ids = sorted(str(paper_id) for paper_id in stage1_paper_ids)
    membership_closed = (
        bool(frozen_ids)
        and decision_ids == frozen_ids
        and len(set(decision_ids)) == len(decision_ids)
    )
    passed = bool(
        run["status"] == "complete"
        and len(relevant) == 1
        and relevant_test_only
        and membership_closed
    )
    return _check(
        passed,
        expected,
        run_id=run["run_id"],
        pipeline_status=run["status"],
        decision_count=len(rows),
        relevant_count=len(relevant),
        relevant_paper_id=relevant[0]["paper_id"] if len(relevant) == 1 else None,
        relevant_model_ids=sorted({str(row["model_id"]) for row in relevant if row["model_id"]}),
        test_only=relevant_test_only,
        stage1_paper_ids=frozen_ids,
        decision_paper_ids=decision_ids,
        stage1_membership_closed=membership_closed,
    )


def _stage3_check(
    connection: sqlite3.Connection,
    relevant_paper_id: str | None,
    artifact_root: Path,
    run_id: str | None = None,
) -> dict[str, Any]:
    expected = (
        "the bound Stage 3 run has one checkpoint and one exact candidate/attempt/"
        "available application/pdf artifact whose CAS bytes match its SHA-256"
    )
    for table in (
        "pipeline_runs",
        "stage3_paper_results",
        "download_candidates",
        "download_attempts",
        "fetch_requests",
        "artifacts",
    ):
        if not _table_exists(connection, table):
            return _failed_check(expected, f"missing {table} table")
    run = _pipeline_run(connection, STAGE_ALIASES["stage3"], run_id)
    if run is None:
        return _failed_check(expected, "Stage 3 pipeline run not found")
    checkpoints = [
        dict(row) for row in connection.execute(
            """SELECT paper_id, status, reason_code FROM stage3_paper_results
                 WHERE run_id = ? ORDER BY paper_id""",
            (run["run_id"],),
        ).fetchall()
    ]
    downloaded = [row for row in checkpoints if row["status"] == "downloaded"]
    downloaded_ids = [str(row["paper_id"]) for row in downloaded]
    attempts = [
        dict(row)
        for row in connection.execute(
            """SELECT da.download_attempt_id, da.run_id, da.candidate_id,
                      da.fetch_request_id, da.result_status,
                      da.artifact_id AS attempted_artifact_id,
                      dc.paper_id AS candidate_paper_id,
                      fr.candidate_id AS fetch_candidate_id,
                      a.artifact_id, a.paper_id, a.artifact_kind,
                      a.relative_path, a.mime_type, a.byte_size, a.sha256,
                      a.processing_status
                 FROM download_attempts AS da
                 LEFT JOIN download_candidates AS dc
                   ON dc.candidate_id = da.candidate_id
                 LEFT JOIN fetch_requests AS fr
                   ON fr.request_id = da.fetch_request_id
                 LEFT JOIN artifacts AS a ON a.artifact_id = da.artifact_id
                WHERE da.run_id = ?
                ORDER BY da.attempted_at, da.download_attempt_id""",
            (run["run_id"],),
        ).fetchall()
    ]
    same_paper = relevant_paper_id is not None and downloaded_ids == [relevant_paper_id]
    attempt = attempts[0] if len(attempts) == 1 else None
    exact_attempt = bool(
        attempt is not None
        and attempt["run_id"] == run["run_id"]
        and attempt["result_status"] == "downloaded"
        and attempt["candidate_id"] == attempt["fetch_candidate_id"]
        and attempt["attempted_artifact_id"] == attempt["artifact_id"]
        and attempt["candidate_paper_id"] == relevant_paper_id
        and attempt["paper_id"] == relevant_paper_id
    )
    artifact: dict[str, Any] | None = None
    artifact_error: str | None = None
    if exact_attempt and attempt is not None:
        try:
            artifact = _artifact_integrity(
                artifact_root,
                attempt,
                expected_kind="pdf",
                expected_mime_type="application/pdf",
                require_pdf_signature=True,
            )
        except (MatrixError, OSError) as error:
            artifact_error = str(error)
    passed = (
        run["status"] == "complete"
        and len(checkpoints) == 1
        and len(downloaded) == 1
        and same_paper
        and exact_attempt
        and artifact is not None
    )
    return _check(
        passed,
        expected,
        run_id=run["run_id"],
        pipeline_status=run["status"],
        checkpoint_count=len(checkpoints),
        downloaded_count=len(downloaded),
        downloaded_paper_id=downloaded_ids[0] if len(downloaded_ids) == 1 else None,
        attempt_count=len(attempts),
        exact_attempt_binding=exact_attempt,
        download_attempt_id=(attempt["download_attempt_id"] if attempt else None),
        candidate_id=(attempt["candidate_id"] if attempt else None),
        artifact_id=(artifact["artifact_id"] if artifact else None),
        artifact_sha256=(artifact["sha256"] if artifact else None),
        artifact_mime_type=(attempt["mime_type"] if attempt else None),
        artifact_error=artifact_error,
        matches_stage2_paper=same_paper,
    )


def _stage4_check(
    connection: sqlite3.Connection,
    relevant_paper_id: str | None,
    stage3: Mapping[str, Any],
    artifact_root: Path,
    run_id: str | None = None,
) -> dict[str, Any]:
    expected = (
        "the bound Stage 4 run has one full_pdf Luna dispatch bound through the "
        "Stage 3 PDF lineage to intact JSON and Markdown output artifacts"
    )
    for table in (
        "pipeline_runs",
        "analysis_dispatches",
        "analysis_runs",
        "artifacts",
        "text_extractions",
    ):
        if not _table_exists(connection, table):
            return _failed_check(expected, f"missing {table} table")
    run = _pipeline_run(connection, STAGE_ALIASES["stage4"], run_id)
    if run is None:
        return _failed_check(expected, "Stage 4 pipeline run not found")
    dispatches = [
        dict(row) for row in connection.execute(
            """SELECT dispatch_id, paper_id, artifact_hash, artifact_id,
                      input_scope, config_hash, implementation_version,
                      profile, model_id, prompt_hash, schema_hash,
                      prompt_input_hash, rendered_prompt_hash, status,
                      dispatch_count, invocation_id, analysis_run_id
                 FROM analysis_dispatches WHERE run_id = ? ORDER BY paper_id""",
            (run["run_id"],),
        ).fetchall()
    ]
    analyses = [
        dict(row) for row in connection.execute(
            """SELECT analysis_run_id, run_id, paper_id, artifact_id, input_hash,
                      input_scope, model_id, prompt_hash, schema_hash,
                      implementation_version, status, output_artifact_id,
                      markdown_artifact_id
                 FROM analysis_runs WHERE run_id = ? ORDER BY paper_id""",
            (run["run_id"],),
        ).fetchall()
    ]
    complete_dispatches = [row for row in dispatches if row["status"] == "complete"]
    complete_analyses = [row for row in analyses if row["status"] == "complete"]
    paper_ids = [str(row["paper_id"]) for row in complete_dispatches]
    same_paper = relevant_paper_id is not None and paper_ids == [relevant_paper_id]
    dispatch = complete_dispatches[0] if len(complete_dispatches) == 1 else None
    analysis = complete_analyses[0] if len(complete_analyses) == 1 else None
    exact_rows = bool(
        dispatch is not None
        and analysis is not None
        and dispatch["analysis_run_id"] == analysis["analysis_run_id"]
        and dispatch["paper_id"] == analysis["paper_id"] == relevant_paper_id
        and dispatch["artifact_id"] == analysis["artifact_id"]
        and dispatch["input_scope"] == analysis["input_scope"] == "full_pdf"
        and dispatch["profile"] == "stage4_analysis_luna"
        and dispatch["model_id"] == analysis["model_id"] == "gpt-5.6-luna"
        and dispatch["config_hash"] == run["config_hash"]
        and dispatch["implementation_version"]
        == analysis["implementation_version"]
        == run["implementation_version"]
        and dispatch["prompt_hash"] == analysis["prompt_hash"]
        and dispatch["schema_hash"] == analysis["schema_hash"]
        and dispatch["prompt_input_hash"] == analysis["input_hash"]
        and bool(dispatch["rendered_prompt_hash"])
        and bool(dispatch["invocation_id"])
        and int(dispatch["dispatch_count"]) == 1
    )
    source_artifact: dict[str, Any] | None = None
    output_artifact: dict[str, Any] | None = None
    markdown_artifact: dict[str, Any] | None = None
    artifact_error: str | None = None
    lineage_matches_stage3 = False
    if exact_rows and dispatch is not None and analysis is not None:
        source = connection.execute(
            "SELECT * FROM artifacts WHERE artifact_id = ?",
            (dispatch["artifact_id"],),
        ).fetchone()
        output = connection.execute(
            "SELECT * FROM artifacts WHERE artifact_id = ?",
            (analysis["output_artifact_id"],),
        ).fetchone()
        markdown = connection.execute(
            "SELECT * FROM artifacts WHERE artifact_id = ?",
            (analysis["markdown_artifact_id"],),
        ).fetchone()
        try:
            if source is None or output is None or markdown is None:
                raise MatrixError("Stage 4 references a missing artifact row")
            source_row = dict(source)
            source_kind = str(source_row.get("artifact_kind") or "")
            source_mime = {
                "pdf": "application/pdf",
                "text": "text/plain; charset=utf-8",
            }.get(source_kind)
            if source_mime is None:
                raise MatrixError("Stage 4 input artifact is neither PDF nor normalized text")
            source_artifact = _artifact_integrity(
                artifact_root,
                source_row,
                expected_kind=source_kind,
                expected_mime_type=source_mime,
                require_pdf_signature=source_kind == "pdf",
            )
            if dispatch["artifact_hash"] != source_artifact["sha256"]:
                raise MatrixError("Stage 4 dispatch artifact_hash does not match its CAS input")
            stage3_artifact_id = stage3.get("artifact_id")
            stage3_artifact_hash = stage3.get("artifact_sha256")
            if source_kind == "pdf":
                lineage_matches_stage3 = (
                    source_artifact["artifact_id"] == stage3_artifact_id
                    and source_artifact["sha256"] == stage3_artifact_hash
                )
            else:
                extractions = connection.execute(
                    """SELECT source_artifact_id, source_sha256, output_artifact_id,
                              paper_id, status
                         FROM text_extractions
                        WHERE output_artifact_id = ?""",
                    (source_artifact["artifact_id"],),
                ).fetchall()
                lineage_matches_stage3 = bool(
                    len(extractions) == 1
                    and extractions[0]["status"] == "full_text_ready"
                    and extractions[0]["paper_id"] == relevant_paper_id
                    and extractions[0]["source_artifact_id"] == stage3_artifact_id
                    and extractions[0]["source_sha256"] == stage3_artifact_hash
                )
            if not lineage_matches_stage3:
                raise MatrixError("Stage 4 input does not trace to the exact Stage 3 PDF")
            output_artifact = _artifact_integrity(
                artifact_root,
                dict(output),
                expected_kind="analysis",
                expected_mime_type="application/json",
            )
            markdown_artifact = _artifact_integrity(
                artifact_root,
                dict(markdown),
                expected_kind="analysis",
                expected_mime_type="text/markdown; charset=utf-8",
            )
            if (
                dict(output)["paper_id"] != relevant_paper_id
                or dict(markdown)["paper_id"] != relevant_paper_id
            ):
                raise MatrixError("Stage 4 output artifact paper binding has drifted")
        except (MatrixError, OSError) as error:
            artifact_error = str(error)
    passed = (
        run["status"] == "complete"
        and len(dispatches) == 1
        and len(complete_dispatches) == 1
        and len(analyses) == 1
        and len(complete_analyses) == 1
        and same_paper
        and exact_rows
        and lineage_matches_stage3
        and source_artifact is not None
        and output_artifact is not None
        and markdown_artifact is not None
    )
    return _check(
        passed,
        expected,
        run_id=run["run_id"],
        pipeline_status=run["status"],
        dispatch_rows=len(dispatches),
        complete_dispatches=len(complete_dispatches),
        dispatch_count=sum(int(row["dispatch_count"]) for row in dispatches),
        complete_analysis_rows=len(complete_analyses),
        paper_id=paper_ids[0] if len(paper_ids) == 1 else None,
        input_scopes=sorted({str(row["input_scope"]) for row in dispatches}),
        model_ids=sorted({str(row["model_id"]) for row in dispatches}),
        exact_dispatch_analysis_binding=exact_rows,
        input_artifact_id=(source_artifact["artifact_id"] if source_artifact else None),
        input_artifact_sha256=(source_artifact["sha256"] if source_artifact else None),
        stage3_lineage_match=lineage_matches_stage3,
        output_artifact_id=(output_artifact["artifact_id"] if output_artifact else None),
        output_artifact_sha256=(output_artifact["sha256"] if output_artifact else None),
        markdown_artifact_id=(
            markdown_artifact["artifact_id"] if markdown_artifact else None
        ),
        markdown_artifact_sha256=(
            markdown_artifact["sha256"] if markdown_artifact else None
        ),
        artifact_error=artifact_error,
        matches_stage2_paper=same_paper,
    )


def _table_counts(
    connection: sqlite3.Connection,
    table: str,
    report_run_id: str,
) -> tuple[int, int]:
    if not _table_exists(connection, table):
        return (-1, -1)
    row = connection.execute(
        f"""SELECT COUNT(*) AS rows,
                   COALESCE(SUM(dispatch_count), 0) AS dispatches
              FROM {table} WHERE report_run_id = ?""",
        (report_run_id,),
    ).fetchone()
    return (int(row["rows"]), int(row["dispatches"]))


def _stage4b_check(
    connection: sqlite3.Connection,
    artifact_root: Path,
    expected_analysis_hashes: Sequence[str],
    report_run_id: str | None = None,
    pipeline_run_id: str | None = None,
) -> dict[str, Any]:
    expected = (
        "latest Stage 4b run has one one-shot Sol invocation and zero "
        "reduce/audit/repair calls"
    )
    required = (
        "pipeline_runs",
        "artifacts",
        "report_runs",
        "report_one_shot_runs",
        "report_sol_invocations",
        "report_reduce_nodes",
        "report_audit_steps",
        "report_audit_shard_steps",
    )
    for table in required:
        if not _table_exists(connection, table):
            return _failed_check(expected, f"missing {table} table")
    if report_run_id is not None:
        row = connection.execute(
            """SELECT rr.report_run_id, rr.run_id AS pipeline_run_id,
                      rr.status AS report_status,
                      rr.output_relative_path, rr.aggregation_tree_json,
                      os.input_artifact_hashes_json, os.input_hash,
                      os.rendered_prompt_hash,
                      os.status AS one_shot_status, os.dispatch_count,
                      os.budget_calls_reserved, os.profile, os.model_id,
                      os.reasoning_effort, os.invocation_id, os.output_hash,
                      os.output_artifact_id,
                      pr.status AS pipeline_status, pr.stage AS pipeline_stage,
                      pr.config_hash AS pipeline_config_hash,
                      pr.implementation_version AS pipeline_implementation_version
                 FROM report_runs AS rr
                 JOIN report_one_shot_runs AS os USING(report_run_id)
                 JOIN pipeline_runs AS pr ON pr.run_id = rr.run_id
                WHERE rr.report_run_id = ?""",
            (report_run_id,),
        ).fetchone()
        if row is not None and pipeline_run_id is not None and row["pipeline_run_id"] != pipeline_run_id:
            return _failed_check(
                expected,
                "report_run_id is not bound to the configured report_pipeline_run_id",
            )
        pipeline = (
            {
                "run_id": row["pipeline_run_id"],
                "status": row["pipeline_status"],
                "stage": row["pipeline_stage"],
                "config_hash": row["pipeline_config_hash"],
                "implementation_version": row["pipeline_implementation_version"],
            }
            if row is not None
            else None
        )
    else:
        pipeline = _pipeline_run(
            connection, STAGE_ALIASES["stage4b"], pipeline_run_id
        )
        if pipeline is None:
            return _failed_check(expected, "Stage 4b pipeline run not found")
        row = connection.execute(
            """SELECT rr.report_run_id, rr.run_id AS pipeline_run_id,
                  rr.status AS report_status,
                  rr.output_relative_path, rr.aggregation_tree_json,
                  os.input_artifact_hashes_json, os.input_hash,
                  os.rendered_prompt_hash,
                  os.status AS one_shot_status, os.dispatch_count,
                  os.budget_calls_reserved, os.profile, os.model_id,
                  os.reasoning_effort, os.invocation_id, os.output_hash,
                  os.output_artifact_id,
                  ? AS pipeline_status, ? AS pipeline_stage,
                  ? AS pipeline_config_hash,
                  ? AS pipeline_implementation_version
             FROM report_runs AS rr
             JOIN report_one_shot_runs AS os USING(report_run_id)
            WHERE rr.run_id = ?""",
            (
                pipeline["status"],
                pipeline["stage"],
                pipeline["config_hash"],
                pipeline["implementation_version"],
                pipeline["run_id"],
            ),
        ).fetchone()
    if row is None:
        return _failed_check(expected, "one-shot report row not found for Stage 4b run")
    report = dict(row)
    report_run_id = str(report["report_run_id"])
    invocations = [
        dict(item) for item in connection.execute(
            """SELECT invocation_id, phase, node_key FROM report_sol_invocations
                 WHERE report_run_id = ? ORDER BY created_at, invocation_id""",
            (report_run_id,),
        ).fetchall()
    ]
    reduce_rows, reduce_dispatches = _table_counts(
        connection, "report_reduce_nodes", report_run_id
    )
    audit_rows, audit_dispatches = _table_counts(
        connection, "report_audit_steps", report_run_id
    )
    shard_rows, shard_dispatches = _table_counts(
        connection, "report_audit_shard_steps", report_run_id
    )
    repair_row = connection.execute(
        """SELECT COUNT(*) AS rows, COALESCE(SUM(dispatch_count), 0) AS dispatches
             FROM report_audit_steps
            WHERE report_run_id = ? AND step_name = 'repair'""",
        (report_run_id,),
    ).fetchone()
    repair_rows = int(repair_row["rows"])
    repair_dispatches = int(repair_row["dispatches"])
    try:
        strategy = json.loads(str(report["aggregation_tree_json"])).get("strategy")
    except (json.JSONDecodeError, AttributeError):
        strategy = None
    try:
        input_artifact_hashes = json.loads(str(report["input_artifact_hashes_json"]))
    except json.JSONDecodeError:
        input_artifact_hashes = None
    exact_inputs = bool(
        isinstance(input_artifact_hashes, list)
        and all(isinstance(item, str) for item in input_artifact_hashes)
        and input_artifact_hashes == list(expected_analysis_hashes)
        and len(set(input_artifact_hashes)) == len(input_artifact_hashes)
    )
    one_ledger_row = (
        len(invocations) == 1
        and invocations[0]["invocation_id"] == report["invocation_id"]
        and invocations[0]["phase"] == "reduce"
        and invocations[0]["node_key"] == "one_shot:0001"
    )
    output_artifact: dict[str, Any] | None = None
    output_artifact_error: str | None = None
    output_row = connection.execute(
        "SELECT * FROM artifacts WHERE artifact_id = ?",
        (report["output_artifact_id"],),
    ).fetchone()
    if output_row is not None:
        try:
            output_artifact = _artifact_integrity(
                artifact_root,
                dict(output_row),
                expected_kind="report",
                expected_mime_type="application/json",
            )
            if output_artifact["sha256"] != report["output_hash"]:
                raise MatrixError(
                    "Stage 4b output_hash does not match its exact CAS artifact"
                )
        except (MatrixError, OSError) as error:
            output_artifact_error = str(error)
            output_artifact = None
    else:
        output_artifact_error = "Stage 4b output_artifact_id has no artifact row"
    implementation_version = str(pipeline.get("implementation_version") or "")
    implementation_qualification = {
        "stage4b-one-shot-v1": "legacy_v1_requires_current_verifier",
        "stage4b-one-shot-v2": "legacy_v2_requires_current_verifier",
        "stage4b-one-shot-v3": "current_v3",
    }.get(implementation_version, "unsupported")
    config_hash = str(pipeline.get("config_hash") or "")
    passed = (
        pipeline is not None
        and pipeline["stage"] in STAGE_ALIASES["stage4b"]
        and pipeline["status"] == "complete"
        and report["report_status"] == "complete"
        and report["one_shot_status"] == "complete"
        and strategy == "one_shot"
        and report["profile"] == "stage4b_oneshot_sol"
        and report["model_id"] == "gpt-5.6-sol"
        and report["reasoning_effort"] == "high"
        and implementation_version in ACCEPTED_STAGE4B_IMPLEMENTATIONS
        and SHA256_PATTERN.fullmatch(config_hash) is not None
        and int(report["dispatch_count"]) == 1
        and int(report["budget_calls_reserved"]) == 1
        and SHA256_PATTERN.fullmatch(str(report["input_hash"] or "")) is not None
        and SHA256_PATTERN.fullmatch(str(report["rendered_prompt_hash"] or "")) is not None
        and exact_inputs
        and output_artifact is not None
        and one_ledger_row
        and reduce_rows == reduce_dispatches == 0
        and audit_rows == audit_dispatches == 0
        and shard_rows == shard_dispatches == 0
        and repair_rows == repair_dispatches == 0
    )
    return _check(
        passed,
        expected,
        run_id=pipeline["run_id"],
        pipeline_status=pipeline["status"],
        report_run_id=report_run_id,
        report_status=report["report_status"],
        one_shot_status=report["one_shot_status"],
        strategy=strategy,
        profile=report["profile"],
        model_id=report["model_id"],
        implementation_version=implementation_version,
        implementation_qualification=implementation_qualification,
        config_hash=config_hash,
        one_shot_dispatch_count=int(report["dispatch_count"]),
        input_artifact_hashes=input_artifact_hashes,
        exact_stage4_inputs=exact_inputs,
        sol_invocation_ledger_count=len(invocations),
        sol_invocation_ledger_phase=(invocations[0]["phase"] if len(invocations) == 1 else None),
        reduce_nodes=reduce_rows,
        reduce_dispatches=reduce_dispatches,
        audit_steps=audit_rows,
        audit_dispatches=audit_dispatches,
        audit_shard_steps=shard_rows,
        audit_shard_dispatches=shard_dispatches,
        repair_steps=repair_rows,
        repair_dispatches=repair_dispatches,
        output_artifact_id=(output_artifact["artifact_id"] if output_artifact else None),
        output_hash=(output_artifact["sha256"] if output_artifact else report["output_hash"]),
        output_artifact_error=output_artifact_error,
        output_relative_path=report["output_relative_path"],
    )


def _implementation_qualification(
    implementation_version: str,
    *,
    verifier_passed: bool,
    audit_binding_mode: str | None,
) -> str:
    if implementation_version not in ACCEPTED_STAGE4B_IMPLEMENTATIONS:
        return "unsupported"
    if not verifier_passed:
        return (
            "legacy_v1_current_verifier_not_passed"
            if implementation_version == "stage4b-one-shot-v1"
            else f"{implementation_version.removeprefix('stage4b-one-shot-')}_current_verifier_not_passed"
        )
    if audit_binding_mode == "legacy_runtime_claim_order_verified":
        return (
            "legacy_v1_audit_order_reverified"
            if implementation_version == "stage4b-one-shot-v1"
            else "legacy_v2_audit_order_reverified"
        )
    if implementation_version == "stage4b-one-shot-v1":
        return "legacy_v1_reverified_by_current_verifier"
    if implementation_version == "stage4b-one-shot-v2":
        return "legacy_v2_reverified_by_current_verifier"
    return "current_v3"


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MatrixError(f"cannot read valid JSON from {path}: {error}") from error
    if not isinstance(value, dict):
        raise MatrixError(f"expected a JSON object in {path}")
    return value


def _load_json_array(path: Path) -> list[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MatrixError(f"cannot read valid JSON from {path}: {error}") from error
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise MatrixError(f"expected an array of JSON objects in {path}")
    return value


def _load_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise MatrixError(f"cannot read JSONL from {path}: {error}") from error
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise MatrixError(
                f"invalid JSONL in {path.name} line {line_number}: {error}"
            ) from error
        if not isinstance(value, dict):
            raise MatrixError(
                f"expected a JSON object in {path.name} line {line_number}"
            )
        values.append(value)
    return values


def _report_directory(
    output_root: Path,
    report_run_id: str,
    output_relative_path: Any,
) -> Path:
    if not isinstance(output_relative_path, str) or not output_relative_path:
        raise MatrixError("Stage 4b output_relative_path is missing")
    relative = Path(output_relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise MatrixError("Stage 4b output_relative_path is unsafe")
    if relative != Path("reports") / report_run_id:
        raise MatrixError(
            "Stage 4b output_relative_path does not match reports/<report_run_id>"
        )
    report_dir = (output_root / relative).resolve()
    try:
        report_dir.relative_to(output_root)
    except ValueError as error:
        raise MatrixError("Stage 4b report directory escapes output_root") from error
    if report_dir.is_symlink() or not report_dir.is_dir():
        raise MatrixError(f"report output directory not found: {report_dir}")
    return report_dir


def _logical_file_hash(path: Path) -> str:
    if path.name == "CLAIMS_EVIDENCE.jsonl":
        values = []
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                values.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise MatrixError(
                    f"invalid JSONL in {path.name} line {line_number}: {error}"
                ) from error
        return content_hash(values)
    if path.suffix == ".json":
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise MatrixError(f"invalid JSON in {path.name}: {error}") from error
        return content_hash(value)
    return content_hash(path.read_text(encoding="utf-8"))


def _verify_report_manifest(
    report_dir: Path, report_run_id: str
) -> dict[str, Any]:
    manifest_path = report_dir / "MANIFEST.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise MatrixError("immutable report bundle lacks MANIFEST.json")
    manifest = _load_json_object(manifest_path)
    if set(manifest) != {"report_run_id", "artifacts"}:
        raise MatrixError("MANIFEST.json has unexpected top-level fields")
    if manifest.get("report_run_id") != report_run_id:
        raise MatrixError("MANIFEST report_run_id does not match the selected run")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise MatrixError("MANIFEST artifacts must be an object")
    artifact_names = set(artifacts)
    if not REQUIRED_REPORT_FILES.issubset(artifact_names):
        missing = sorted(REQUIRED_REPORT_FILES - artifact_names)
        raise MatrixError("MANIFEST lacks required files: " + ", ".join(missing))
    actual_names = {path.name for path in report_dir.iterdir()}
    expected_names = artifact_names | {"MANIFEST.json"}
    if actual_names != expected_names:
        raise MatrixError("immutable report bundle file set does not match MANIFEST")
    for name, expected_hash in sorted(artifacts.items()):
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or not isinstance(expected_hash, str)
            or SHA256_PATTERN.fullmatch(expected_hash) is None
        ):
            raise MatrixError("MANIFEST contains an unsafe name or invalid logical hash")
        path = report_dir / name
        if path.is_symlink() or not path.is_file():
            raise MatrixError(f"manifest artifact is not a regular file: {name}")
        if _logical_file_hash(path) != expected_hash:
            raise MatrixError(f"manifest logical hash mismatch: {name}")
    return {
        "manifest_hash": content_hash(manifest),
        "report_markdown_hash": artifacts["REPORT.md"],
        "artifact_count": len(artifacts),
    }


def _claim_id_order(
    claims: Any, *, label: str
) -> list[Mapping[str, Any]]:
    if not isinstance(claims, list) or any(
        not isinstance(item, Mapping) for item in claims
    ):
        raise MatrixError(f"{label} claims must be a JSON object list")
    claim_ids = [item.get("claim_id") for item in claims]
    if any(not isinstance(item, str) or not item for item in claim_ids):
        raise MatrixError(f"{label} claims lack non-empty string claim_id values")
    if len(set(claim_ids)) != len(claim_ids):
        raise MatrixError(f"{label} claims contain duplicate claim_id values")
    return sorted(claims, key=lambda item: str(item["claim_id"]))


def _legacy_coverage(
    value: Any, *, published: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise MatrixError("legacy report audit coverage is not an object")
    coverage = dict(value)
    missing = coverage.get("missing_paper_ids")
    uncovered = coverage.get("uncovered_claim_ids")
    if (
        not isinstance(missing, list)
        or not isinstance(uncovered, list)
        or any(not isinstance(item, str) for item in (*missing, *uncovered))
    ):
        raise MatrixError("legacy report audit coverage lists are malformed")
    derived_complete = not missing and not uncovered
    if "complete" in coverage and coverage["complete"] is not derived_complete:
        raise MatrixError("legacy report audit coverage.complete is inconsistent")
    coverage["complete"] = derived_complete
    if canonical_json(coverage) != canonical_json(dict(published)):
        raise MatrixError(
            "legacy report audit coverage differs from the published coverage"
        )
    return coverage


def _validate_legacy_claim_order_binding(
    connection: sqlite3.Connection,
    *,
    report_run_id: str,
    implementation_version: str,
    audit: Mapping[str, Any],
    plan: Mapping[str, Any],
    document: Mapping[str, Any],
    claims: list[dict[str, Any]],
    coverage: Mapping[str, Any],
    comparison_groups: Mapping[str, Any],
    claim_relations: list[dict[str, Any]],
    bibliography: Mapping[str, Any],
    search_audit: Mapping[str, Any],
    corpus_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove the v1/v2 runtime-order hash behind a canonicalized report file.

    Historical one-shot reports hashed claims in their runtime order and then
    persisted the same claims in claim-id order.  Compatibility is accepted
    only when the exact database bundle proves that this lossless ordering was
    the sole difference from the immutable report inputs.
    """
    if implementation_version not in LEGACY_CLAIM_ORDER_IMPLEMENTATIONS:
        raise MatrixError(
            "legacy claim-order binding requires stage4b-one-shot-v1 or v2"
        )
    if not _table_exists(connection, "report_audit_runs"):
        raise MatrixError("legacy claim-order binding lacks report_audit_runs")

    rows = connection.execute(
        """SELECT report_run_id, base_artifact_hash, current_artifact_hash,
                  current_bundle_json
             FROM report_audit_runs WHERE report_run_id = ?""",
        (report_run_id,),
    ).fetchall()
    if len(rows) != 1 or rows[0]["report_run_id"] != report_run_id:
        raise MatrixError(
            "legacy claim-order binding lacks one exact report_audit_runs row"
        )
    row = rows[0]
    audit_hash = audit.get("report_artifact_hash")
    if (
        not isinstance(audit_hash, str)
        or SHA256_PATTERN.fullmatch(audit_hash) is None
        or row["base_artifact_hash"] != audit_hash
        or row["current_artifact_hash"] != audit_hash
    ):
        raise MatrixError(
            "legacy report audit base/current/artifact hashes are not identical"
        )
    try:
        bundle = json.loads(str(row["current_bundle_json"]))
    except json.JSONDecodeError as error:
        raise MatrixError("legacy report audit current_bundle_json is invalid") from error
    expected_keys = {
        "document",
        "claims",
        "coverage",
        "comparison_groups",
        "claim_relations",
        "bibliography",
    }
    if not isinstance(bundle, Mapping) or set(bundle) != expected_keys:
        raise MatrixError("legacy report audit current bundle has unexpected fields")
    if (
        not isinstance(bundle["document"], Mapping)
        or bundle["document"].get("report_run_id") != report_run_id
        or canonical_json(bundle["document"]) != canonical_json(document)
    ):
        raise MatrixError("legacy report audit document binding has drifted")

    runtime_claims = bundle["claims"]
    ordered_runtime_claims = _claim_id_order(
        runtime_claims, label="legacy runtime"
    )
    ordered_published_claims = _claim_id_order(
        claims, label="published"
    )
    if canonical_json(ordered_runtime_claims) != canonical_json(
        ordered_published_claims
    ):
        raise MatrixError(
            "legacy runtime claims differ from published claims beyond ordering"
        )
    runtime_coverage = _legacy_coverage(bundle["coverage"], published=coverage)
    component_pairs = (
        ("comparison groups", bundle["comparison_groups"], comparison_groups),
        ("claim relations", bundle["claim_relations"], claim_relations),
        ("bibliography", bundle["bibliography"], bibliography),
    )
    for label, runtime_value, published_value in component_pairs:
        if canonical_json(runtime_value) != canonical_json(published_value):
            raise MatrixError(f"legacy report audit {label} binding has drifted")

    legacy_hash = content_hash({
        "document": bundle["document"],
        "claims": list(runtime_claims),
        "coverage": runtime_coverage,
        "comparison_groups": bundle["comparison_groups"],
        "claim_relations": list(bundle["claim_relations"]),
        "bibliography": bundle["bibliography"],
    })
    if legacy_hash != audit_hash:
        raise MatrixError(
            "legacy runtime claim order does not reproduce the report audit hash"
        )

    # Prove that the historical artifact hash is the only non-canonical field;
    # all other audit hashes, exhaustive coverage, and finding references remain
    # subject to the current strict validator.
    canonical_audit = dict(audit)
    canonical_audit["report_artifact_hash"] = report_artifact_hash(
        document=document,
        claims=claims,
        coverage=coverage,
        comparison_groups=comparison_groups,
        claim_relations=claim_relations,
        bibliography=bibliography,
    )
    _validate_audit_binding(
        canonical_audit,
        plan=plan,
        document=document,
        claims=claims,
        coverage=coverage,
        comparison_groups=comparison_groups,
        claim_relations=claim_relations,
        bibliography=bibliography,
        search_audit=search_audit,
        corpus_snapshot=corpus_snapshot,
    )
    return {
        "mode": "legacy_runtime_claim_order_verified",
        "database_row_bound": True,
        "legacy_artifact_hash": legacy_hash,
    }


def _validate_report_audit_binding(
    connection: sqlite3.Connection,
    *,
    report_run_id: str,
    implementation_version: str,
    audit: Mapping[str, Any],
    plan: Mapping[str, Any],
    document: Mapping[str, Any],
    claims: list[dict[str, Any]],
    coverage: Mapping[str, Any],
    comparison_groups: Mapping[str, Any],
    claim_relations: list[dict[str, Any]],
    bibliography: Mapping[str, Any],
    search_audit: Mapping[str, Any],
    corpus_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    arguments = {
        "plan": plan,
        "document": document,
        "claims": claims,
        "coverage": coverage,
        "comparison_groups": comparison_groups,
        "claim_relations": claim_relations,
        "bibliography": bibliography,
        "search_audit": search_audit,
        "corpus_snapshot": corpus_snapshot,
    }
    try:
        _validate_audit_binding(audit, **arguments)
    except ReportVerificationError as canonical_error:
        canonical_hash = report_artifact_hash(
            document=document,
            claims=claims,
            coverage=coverage,
            comparison_groups=comparison_groups,
            claim_relations=claim_relations,
            bibliography=bibliography,
        )
        if audit.get("report_artifact_hash") == canonical_hash:
            raise canonical_error
        try:
            return _validate_legacy_claim_order_binding(
                connection,
                report_run_id=report_run_id,
                implementation_version=implementation_version,
                audit=audit,
                **arguments,
            )
        except (MatrixError, ReportVerificationError, sqlite3.Error) as legacy_error:
            raise MatrixError(
                f"{canonical_error}; legacy claim-order binding rejected: {legacy_error}"
            ) from legacy_error
    return {
        "mode": "canonical_claim_id_order",
        "database_row_bound": False,
        "legacy_artifact_hash": None,
    }


def _verify_check(
    connection: sqlite3.Connection,
    output_root: Path,
    stage4b: Mapping[str, Any],
    stage4: Mapping[str, Any],
    stage1: Mapping[str, Any],
    *,
    expected_venue: str,
) -> dict[str, Any]:
    expected = (
        "the exact immutable report bundle passes its MANIFEST, saved verification, "
        "audit, report hash, and the complete deterministic verifier"
    )
    report_run_id = stage4b.get("report_run_id")
    if not isinstance(report_run_id, str) or not report_run_id:
        return _failed_check(expected, "Stage 4b did not select a report run")
    try:
        report_dir = _report_directory(
            output_root,
            report_run_id,
            stage4b.get("output_relative_path"),
        )
        manifest = _verify_report_manifest(report_dir, report_run_id)
        plan = _load_json_object(report_dir / "REPORT_PLAN.json")
        search_audit = _load_json_object(report_dir / "SEARCH_AUDIT.json")
        verification = _load_json_object(report_dir / "VERIFICATION.json")
        audit = _load_json_object(report_dir / "AUDIT.json")
        corpus = _load_json_object(report_dir / "CORPUS_SNAPSHOT.json")
        claims = _load_jsonl_objects(report_dir / "CLAIMS_EVIDENCE.jsonl")
        comparison_groups = _load_json_object(
            report_dir / "COMPARISON_GROUPS.json"
        )
        claim_relations = _load_json_array(report_dir / "CLAIM_RELATIONS.json")
        document = _load_json_object(report_dir / "REPORT_DOCUMENT.json")
        coverage = _load_json_object(report_dir / "COVERAGE.json")
        bibliography = _load_json_object(report_dir / "BIBLIOGRAPHY.json")
        regenerated = verify_report_run(output_root, report_run_id)
        validate(audit, "report-audit.schema.json")
        audit_binding = _validate_report_audit_binding(
            connection,
            report_run_id=report_run_id,
            implementation_version=str(
                stage4b.get("implementation_version") or ""
            ),
            audit=audit,
            plan=plan,
            document=document,
            claims=claims,
            coverage=coverage,
            comparison_groups=comparison_groups,
            claim_relations=claim_relations,
            bibliography=bibliography,
            search_audit=search_audit,
            corpus_snapshot=corpus,
        )
    except (MatrixError, OSError, UnicodeError, ValueError, TypeError) as error:
        return _failed_check(expected, str(error))
    verifier_matches_saved = regenerated == verification
    corpus_papers = corpus.get("papers")
    exact_corpus_binding = bool(
        isinstance(corpus_papers, list)
        and len(corpus_papers) == 1
        and isinstance(corpus_papers[0], Mapping)
        and corpus_papers[0].get("paper_id") == stage4.get("paper_id")
        and corpus_papers[0].get("input_scope") == "full_pdf"
        and corpus_papers[0].get("analysis_artifact_hash")
        == stage4.get("output_artifact_sha256")
    )
    scope = plan.get("scope")
    report_venues = scope.get("venues") if isinstance(scope, Mapping) else None
    query_plan_hash = stage1.get("query_plan_hash")
    exact_stage1_scope_binding = bool(
        isinstance(query_plan_hash, str)
        and SHA256_PATTERN.fullmatch(query_plan_hash) is not None
        and plan.get("query_plan_hash") == query_plan_hash
        and search_audit.get("query_plan_hash") == query_plan_hash
        and report_venues == [expected_venue]
    )
    stage4b_config_binding = bool(
        isinstance(stage4b.get("config_hash"), str)
        and SHA256_PATTERN.fullmatch(str(stage4b["config_hash"])) is not None
        and plan.get("stage4b_config_hash") == stage4b.get("config_hash")
    )
    checks = verification.get("checks")
    checks_passed = (
        isinstance(checks, Mapping)
        and bool(checks)
        and all(value is True for value in checks.values())
    )
    coverage_passed = verification.get("coverage_complete") is True
    explicit_status = verification.get("status")
    # ``verify_report`` currently has no top-level status field; the CLI adds
    # no synthetic value.  Treat absence as the native shape, while an explicit
    # value must still be ``passed``.
    status_passed = explicit_status in (None, "passed")
    findings = audit.get("findings")
    severe_findings = []
    if isinstance(findings, list):
        severe_findings = [
            item for item in findings
            if isinstance(item, Mapping) and item.get("severity") in {"blocker", "major"}
        ]
    audit_passed = (
        audit.get("audit_pass") == "deterministic"
        and audit.get("coverage_complete") is True
        and isinstance(findings, list)
        and not severe_findings
    )
    report_exists = (report_dir / "REPORT.md").is_file()
    passed = (
        checks_passed
        and coverage_passed
        and status_passed
        and audit_passed
        and report_exists
        and verifier_matches_saved
        and exact_corpus_binding
        and exact_stage1_scope_binding
        and stage4b_config_binding
    )
    return _check(
        passed,
        expected,
        report_run_id=report_run_id,
        report_directory=str(report_dir),
        verification_status="passed" if checks_passed and coverage_passed and status_passed else "failed",
        verification_checks=dict(checks) if isinstance(checks, Mapping) else None,
        coverage_complete=coverage_passed,
        audit_pass=audit.get("audit_pass"),
        audit_binding_mode=audit_binding["mode"],
        audit_binding_database_row_bound=audit_binding["database_row_bound"],
        legacy_audit_artifact_hash=audit_binding["legacy_artifact_hash"],
        blocker_or_major_findings=len(severe_findings),
        report_markdown_present=report_exists,
        verifier_matches_saved=verifier_matches_saved,
        exact_stage4_corpus_binding=exact_corpus_binding,
        exact_stage1_query_plan_scope_binding=exact_stage1_scope_binding,
        report_scope_venues=report_venues,
        stage4b_config_binding=stage4b_config_binding,
        complete_verifier_status=regenerated.get("status"),
        manifest_hash=manifest["manifest_hash"],
        report_markdown_hash=manifest["report_markdown_hash"],
        manifest_artifact_count=manifest["artifact_count"],
    )


def summarize_venue(
    run_dir: Path, *, database_relative_path: Path | None = None
) -> dict[str, Any]:
    """Summarize one venue directory without modifying its database or artifacts."""

    try:
        run_binding = _load_run_binding(run_dir)
    except MatrixError as error:
        run_binding = {"binding_error": str(error), "strict": True}
    venue = (
        str(run_binding["venue"])
        if isinstance(run_binding.get("venue"), str)
        else run_dir.name
    )
    binding_error = run_binding.get("binding_error")
    if isinstance(binding_error, str):
        checks = {
            name: _failed_check("run.json is valid", binding_error)
            for name in CHECK_ORDER
        }
        return {
            "venue": venue,
            "run_dir": str(run_dir.resolve()),
            "database": None,
            "passed": False,
            "checks": checks,
        }
    strict = run_binding.get("strict") is True
    if strict:
        required_bindings = (
            "venue",
            "database",
            "artifact_root",
            "search_run_id",
            "crawl_run_id",
            "filter_run_id",
            "stage3_run_id",
            "stage4_run_id",
            "report_run_id",
            "report_pipeline_run_id",
        )
        missing = [name for name in required_bindings if not run_binding.get(name)]
        if missing:
            error = "run.json native_pipeline is missing: " + ", ".join(missing)
            checks = {
                name: _failed_check("explicit native pipeline bindings are complete", error)
                for name in CHECK_ORDER
            }
            return {
                "venue": venue,
                "run_dir": str(run_dir.resolve()),
                "database": None,
                "passed": False,
                "checks": checks,
            }
    try:
        database = _resolve_database(run_dir, database_relative_path, run_binding)
        artifact_root = _resolve_bound_directory(
            run_dir,
            run_binding.get("artifact_root"),
            label="artifact_root",
            default=Path("artifacts"),
        )
        output_root = _resolve_bound_directory(
            run_dir,
            run_binding.get("output_root"),
            label="report output_root",
            default=Path("."),
        )
    except MatrixError as error:
        checks = {
            name: _failed_check("evidence is available", str(error))
            for name in CHECK_ORDER
        }
        return {
            "venue": venue,
            "run_dir": str(run_dir.resolve()),
            "database": None,
            "passed": False,
            "checks": checks,
        }
    try:
        connection = _connect_read_only(database)
    except sqlite3.Error as error:
        checks = {
            name: _failed_check("evidence is readable", f"cannot open SQLite: {error}")
            for name in CHECK_ORDER
        }
        return {
            "venue": venue,
            "run_dir": str(run_dir.resolve()),
            "database": str(database.resolve()),
            "passed": False,
            "checks": checks,
        }
    try:
        stage1 = (
            _stage1_check(
                connection,
                venue,
                run_binding.get("crawl_run_id"),
                run_binding.get("search_run_id"),
            )
            if not strict or run_binding.get("crawl_run_id")
            else _failed_check(
                "latest Stage 1 pipeline and crawl run are complete",
                "run.json native_pipeline.crawl_run_id is missing",
            )
        )
        stage2 = (
            _stage2_check(
                connection,
                (
                    stage1.get("paper_ids", ())
                    if isinstance(stage1.get("paper_ids"), Sequence)
                    and not isinstance(stage1.get("paper_ids"), (str, bytes))
                    else ()
                ),
                run_binding.get("filter_run_id"),
            )
            if not strict or run_binding.get("filter_run_id")
            else _failed_check(
                "latest Stage 2 run is complete with exactly one TEST_ONLY relevant paper",
                "run.json native_pipeline.filter_run_id is missing",
            )
        )
        relevant_paper_id = (
            stage2.get("relevant_paper_id")
            if isinstance(stage2.get("relevant_paper_id"), str)
            else None
        )
        stage3 = (
            _stage3_check(
                connection,
                relevant_paper_id,
                artifact_root,
                run_binding.get("stage3_run_id"),
            )
            if not strict or run_binding.get("stage3_run_id")
            else _failed_check(
                "latest Stage 3 run is complete with one downloaded PDF checkpoint",
                "run.json native_pipeline.stage3_run_id is missing",
            )
        )
        stage4 = (
            _stage4_check(
                connection,
                relevant_paper_id,
                stage3,
                artifact_root,
                run_binding.get("stage4_run_id"),
            )
            if not strict or run_binding.get("stage4_run_id")
            else _failed_check(
                "latest Stage 4 run has exactly one completed gpt-5.6-luna invocation",
                "run.json native_pipeline.stage4_run_id is missing",
            )
        )
        stage4b = (
            _stage4b_check(
                connection,
                artifact_root,
                (
                    [str(stage4["output_artifact_sha256"])]
                    if isinstance(stage4.get("output_artifact_sha256"), str)
                    else []
                ),
                run_binding.get("report_run_id"),
                run_binding.get("report_pipeline_run_id"),
            )
            if (
                not strict
                or (
                    run_binding.get("report_run_id")
                    and run_binding.get("report_pipeline_run_id")
                )
            )
            else _failed_check(
                "latest Stage 4b run has one one-shot Sol invocation and zero reduce/audit/repair calls",
                "run.json native_pipeline report bindings are missing",
            )
        )
    except (sqlite3.Error, MatrixError, OSError, ValueError, TypeError) as error:
        checks = {
            name: _failed_check("evidence matches the acceptance contract", str(error))
            for name in CHECK_ORDER
        }
    else:
        verify = _verify_check(
            connection,
            output_root,
            stage4b,
            stage4,
            stage1,
            expected_venue=venue,
        )
        stage4b["implementation_qualification"] = _implementation_qualification(
            str(stage4b.get("implementation_version") or ""),
            verifier_passed=verify.get("passed") is True,
            audit_binding_mode=(
                str(verify["audit_binding_mode"])
                if isinstance(verify.get("audit_binding_mode"), str)
                else None
            ),
        )
        checks = {
            "stage1_complete": stage1,
            "stage2_test_only_relevant_one": stage2,
            "stage3_pdf_checkpoint_complete": stage3,
            "stage4_luna_invocation_one": stage4,
            "stage4b_sol_one_shot_only": stage4b,
            "verify_passed": verify,
        }
    finally:
        connection.close()
    return {
        "venue": venue,
        "run_dir": str(run_dir.resolve()),
        "database": str(database.resolve()),
        "database_sha256": _file_sha256(database),
        "binding_source": "run.json/native_pipeline" if run_binding.get("strict") else "auto-discovery",
        "passed": all(bool(checks[name]["passed"]) for name in CHECK_ORDER),
        "checks": checks,
    }


def _venue_catalog_ids(catalog_root: Path) -> tuple[str, ...]:
    root = catalog_root.resolve()
    if not root.is_dir():
        raise MatrixError(f"venue catalog root is not a directory: {root}")
    venue_ids: list[str] = []
    for path in sorted(root.glob("*.yaml")):
        try:
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, yaml.YAMLError) as error:
            raise MatrixError(f"cannot read venue descriptor {path}: {error}") from error
        if not isinstance(document, Mapping):
            raise MatrixError(f"venue descriptor is not an object: {path}")
        venue_id = document.get("venue_id")
        if not isinstance(venue_id, str) or not venue_id:
            raise MatrixError(f"venue descriptor lacks venue_id: {path}")
        venue_ids.append(venue_id)
    if not venue_ids:
        raise MatrixError(f"venue catalog contains no descriptors: {root}")
    duplicates = sorted({item for item in venue_ids if venue_ids.count(item) > 1})
    if duplicates:
        raise MatrixError("duplicate venue_id values in catalog: " + ", ".join(duplicates))
    return tuple(sorted(venue_ids))


def _normalized_venue(value: Any) -> str:
    return "".join(character for character in str(value).casefold() if character.isalnum())


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _historical_evidence_row(
    *,
    venue_id: str,
    evidence_path: Path,
    evidence_sha256: str,
    document: Mapping[str, Any],
    repository_root: Path,
) -> dict[str, Any]:
    if document.get("schema_version") != "1":
        raise MatrixError(
            f"{venue_id}: unsupported historical acceptance evidence schema_version"
        )
    scope = _mapping(document.get("scope"))
    pipeline = _mapping(document.get("pipeline"))
    paper_count = scope.get("paper_count")
    scope_matches = (
        isinstance(paper_count, int)
        and paper_count > 0
        and _normalized_venue(scope.get("venue")) == _normalized_venue(venue_id)
    )

    stage1_value = _mapping(pipeline.get("stage1"))
    stage1_passed = bool(
        scope_matches
        and stage1_value.get("status") == "complete"
        and isinstance(stage1_value.get("run_id"), str)
        and bool(stage1_value.get("run_id"))
        and isinstance(stage1_value.get("crawl_run_id"), str)
        and bool(stage1_value.get("crawl_run_id"))
    )
    stage1 = _check(
        stage1_passed,
        "hash-pinned historical evidence names complete Stage 1 and crawl runs",
        run_id=stage1_value.get("run_id"),
        crawl_run_id=stage1_value.get("crawl_run_id"),
        pipeline_status=stage1_value.get("status"),
        evidence_mode="reused_historical_evidence",
    )

    stage2_value = _mapping(pipeline.get("stage2"))
    selected = stage2_value.get("relevant_selected_for_downstream")
    evaluated = stage2_value.get("evaluated_decisions")
    irrelevant = stage2_value.get("irrelevant")
    needs_review = stage2_value.get("needs_review")
    model_id = stage2_value.get("model_id")
    stage2_passed = bool(
        scope_matches
        and stage2_value.get("status") == "complete"
        and stage2_value.get("test_only") is True
        and stage2_value.get("production_release") is False
        and isinstance(model_id, str)
        and model_id.upper().startswith("TEST_ONLY/")
        and selected == paper_count
        and isinstance(evaluated, int)
        and isinstance(irrelevant, int)
        and isinstance(needs_review, int)
        and evaluated == selected + irrelevant + needs_review
    )
    stage2 = _check(
        stage2_passed,
        "hash-pinned historical evidence records a complete TEST_ONLY selection",
        run_id=stage2_value.get("run_id"),
        pipeline_status=stage2_value.get("status"),
        relevant_count=selected,
        decision_count=evaluated,
        model_id=model_id,
        test_only=stage2_value.get("test_only"),
        production_release=stage2_value.get("production_release"),
    )

    stage3_value = _mapping(pipeline.get("stage3"))
    stage3_passed = bool(
        scope_matches
        and stage3_value.get("status") == "complete"
        and isinstance(stage3_value.get("run_id"), str)
        and stage3_value.get("attempts") == paper_count
        and stage3_value.get("downloaded") == paper_count
        and stage3_value.get("http_status") == 200
        and stage3_value.get("policy_decision") == "allow"
        and isinstance(stage3_value.get("policy_grant_id"), str)
        and bool(stage3_value.get("policy_grant_id"))
    )
    stage3 = _check(
        stage3_passed,
        "hash-pinned historical evidence records one successful PDF per selected paper",
        run_id=stage3_value.get("run_id"),
        pipeline_status=stage3_value.get("status"),
        attempt_count=stage3_value.get("attempts"),
        downloaded_count=stage3_value.get("downloaded"),
        access_basis=stage3_value.get("access_basis"),
        provider=stage3_value.get("provider"),
        evidence_mode="attestation_only_runtime_artifacts_not_copied",
    )

    stage4_value = _mapping(pipeline.get("stage4"))
    stage4_passed = bool(
        scope_matches
        and stage4_value.get("status") == "complete"
        and stage4_value.get("papers") == paper_count
        and stage4_value.get("complete") == paper_count
        and stage4_value.get("output_artifacts") == paper_count
        and stage4_value.get("markdown_artifacts") == paper_count
        and stage4_value.get("input_scope") == "full_pdf"
        and stage4_value.get("model_id") == "gpt-5.6-luna"
    )
    stage4 = _check(
        stage4_passed,
        "hash-pinned historical evidence records complete full_pdf Luna outputs",
        run_id=stage4_value.get("run_id"),
        pipeline_status=stage4_value.get("status"),
        paper_count=stage4_value.get("papers"),
        complete_analysis_rows=stage4_value.get("complete"),
        input_scopes=[stage4_value.get("input_scope")],
        model_ids=[stage4_value.get("model_id")],
        evidence_mode="attestation_only_runtime_artifacts_not_copied",
    )

    stage4b_value = _mapping(pipeline.get("stage4b"))
    stage4b_passed = bool(
        stage4b_value.get("report_status") == "complete"
        and stage4b_value.get("one_shot_status") == "complete"
        and stage4b_value.get("strategy") == "one_shot"
        and stage4b_value.get("profile") == "stage4b_oneshot_sol"
        and stage4b_value.get("model_id") == "gpt-5.6-sol"
        and stage4b_value.get("reasoning_effort") == "high"
        and stage4b_value.get("dispatch_count") == 1
        and stage4b_value.get("budget_calls_reserved") == 1
        and stage4b_value.get("sol_invocation_ledger_count") == 1
        and stage4b_value.get("sol_invocation_ledger_phase") == "reduce"
        and stage4b_value.get("reduce_nodes") == 0
        and stage4b_value.get("audit_steps") == 0
        and stage4b_value.get("audit_shard_steps") == 0
        and SHA256_PATTERN.fullmatch(str(stage4b_value.get("output_hash") or ""))
        is not None
    )
    stage4b = _check(
        stage4b_passed,
        "hash-pinned historical evidence records exactly one Sol call and no reduce/audit/repair calls",
        run_id=stage4b_value.get("pipeline_run_id"),
        pipeline_status=stage4b_value.get("report_status"),
        report_run_id=stage4b_value.get("report_run_id"),
        report_status=stage4b_value.get("report_status"),
        one_shot_status=stage4b_value.get("one_shot_status"),
        strategy=stage4b_value.get("strategy"),
        profile=stage4b_value.get("profile"),
        model_id=stage4b_value.get("model_id"),
        implementation_version=stage4b_value.get("implementation_version"),
        implementation_qualification=(
            "historical_attestation_version_not_recorded"
            if stage4b_value.get("implementation_version") is None
            else "historical_attestation"
        ),
        config_hash=stage4b_value.get("config_hash"),
        one_shot_dispatch_count=stage4b_value.get("dispatch_count"),
        sol_invocation_ledger_count=stage4b_value.get("sol_invocation_ledger_count"),
        reduce_nodes=stage4b_value.get("reduce_nodes"),
        audit_steps=stage4b_value.get("audit_steps"),
        audit_shard_steps=stage4b_value.get("audit_shard_steps"),
        output_hash=stage4b_value.get("output_hash"),
        evidence_mode="attestation_only_raw_sol_output_not_copied",
    )

    verification = _mapping(document.get("verification"))
    verification_checks = verification.get("checks")
    audit = _mapping(verification.get("audit"))
    findings = audit.get("findings")
    severe_findings = (
        [
            item
            for item in findings
            if isinstance(item, Mapping)
            and item.get("severity") in {"blocker", "major"}
        ]
        if isinstance(findings, list)
        else ["missing-findings"]
    )
    durability = _mapping(document.get("durability"))
    committed_report = _mapping(durability.get("committed_report"))
    selected_hashes = _mapping(durability.get("selected_published_manifest_hashes"))
    report_error: str | None = None
    report_path: Path | None = None
    report_hash: str | None = None
    try:
        configured_report_path = committed_report.get("path")
        if not isinstance(configured_report_path, str) or not configured_report_path:
            raise MatrixError("historical evidence lacks committed report path")
        relative_report_path = Path(configured_report_path)
        if relative_report_path.is_absolute() or ".." in relative_report_path.parts:
            raise MatrixError("historical evidence report path is unsafe")
        report_path = (repository_root / relative_report_path).resolve()
        report_path.relative_to(repository_root)
        if report_path.is_symlink() or not report_path.is_file():
            raise MatrixError("historical committed report is unavailable")
        raw_hash = _file_sha256(report_path)
        raw_size = report_path.stat().st_size
        report_hash = content_hash(report_path.read_text(encoding="utf-8"))
        if (
            raw_hash != committed_report.get("file_sha256")
            or raw_size != committed_report.get("size_bytes")
            or report_hash != committed_report.get("manifest_logical_hash")
            or report_hash != selected_hashes.get("REPORT.md")
        ):
            raise MatrixError("historical committed report hash or size has drifted")
    except (MatrixError, OSError, UnicodeError, ValueError) as error:
        report_error = str(error)
    required_verification_checks = {
        "citation_coverage",
        "extraction_scope",
        "no_fabricated_statistics",
        "no_unsupported_claims",
        "search_limitations",
        "table_provenance",
    }
    verify_passed = bool(
        verification.get("status") == "passed"
        and verification.get("coverage_complete") is True
        and isinstance(verification_checks, Mapping)
        and required_verification_checks.issubset(verification_checks)
        and all(value is True for value in verification_checks.values())
        and audit.get("audit_pass") == "deterministic"
        and audit.get("coverage_complete") is True
        and not severe_findings
        and report_path is not None
        and report_error is None
    )
    verify = _check(
        verify_passed,
        "the pinned historical attestation and committed report hashes verify",
        verification_status=verification.get("status"),
        verification_checks=(
            dict(verification_checks) if isinstance(verification_checks, Mapping) else None
        ),
        coverage_complete=verification.get("coverage_complete"),
        audit_pass=audit.get("audit_pass"),
        blocker_or_major_findings=len(severe_findings),
        report_directory=None,
        committed_report_path=(str(report_path) if report_path else None),
        report_markdown_hash=report_hash,
        acceptance_evidence_sha256=evidence_sha256,
        complete_verifier_status="historical_attestation_only",
        report_error=report_error,
    )
    checks = {
        "stage1_complete": stage1,
        "stage2_test_only_relevant_one": stage2,
        "stage3_pdf_checkpoint_complete": stage3,
        "stage4_luna_invocation_one": stage4,
        "stage4b_sol_one_shot_only": stage4b,
        "verify_passed": verify,
    }
    return {
        "venue": venue_id,
        "run_dir": None,
        "database": None,
        "binding_source": "acceptance-evidence-import",
        "evidence_reuse": "reused_historical_evidence",
        "evidence_path": str(evidence_path),
        "evidence_sha256": evidence_sha256,
        "source_revision": document.get("source_revision"),
        "recovery_disclosure": document.get("recovery_disclosure"),
        "passed": all(bool(checks[name]["passed"]) for name in CHECK_ORDER),
        "checks": checks,
    }


def _load_acceptance_imports(manifest_path: Path | None) -> tuple[dict[str, Any], ...]:
    if manifest_path is None:
        return ()
    resolved_manifest = manifest_path.resolve()
    manifest = _load_json_object(resolved_manifest)
    if manifest.get("schema_version") != "paper-agent.venue-e2e-acceptance-imports.v1":
        raise MatrixError("unsupported acceptance import manifest schema_version")
    repository_value = manifest.get("repository_root", ".")
    if not isinstance(repository_value, str) or not repository_value:
        raise MatrixError("acceptance import repository_root must be a path string")
    repository_root = Path(repository_value)
    if not repository_root.is_absolute():
        repository_root = resolved_manifest.parent / repository_root
    repository_root = repository_root.resolve()
    if not repository_root.is_dir():
        raise MatrixError("acceptance import repository_root does not exist")
    entries = manifest.get("imports")
    if not isinstance(entries, list):
        raise MatrixError("acceptance import manifest imports must be an array")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise MatrixError("acceptance import entry must be an object")
        venue_id = entry.get("venue_id")
        evidence_schema_version = entry.get("evidence_schema_version")
        relative_value = entry.get("evidence_path")
        expected_sha256 = entry.get("sha256")
        if not isinstance(venue_id, str) or not venue_id or venue_id in seen:
            raise MatrixError("acceptance import venue_id is missing or duplicated")
        if evidence_schema_version != "1":
            raise MatrixError(
                f"{venue_id}: unsupported acceptance evidence schema_version"
            )
        if (
            not isinstance(relative_value, str)
            or not relative_value
            or not isinstance(expected_sha256, str)
            or SHA256_PATTERN.fullmatch(expected_sha256) is None
        ):
            raise MatrixError(f"{venue_id}: acceptance evidence path/hash is invalid")
        relative = Path(relative_value)
        if relative.is_absolute() or ".." in relative.parts:
            raise MatrixError(f"{venue_id}: acceptance evidence path is unsafe")
        evidence_path = (repository_root / relative).resolve()
        try:
            evidence_path.relative_to(repository_root)
        except ValueError as error:
            raise MatrixError(f"{venue_id}: acceptance evidence escapes repository") from error
        if evidence_path.is_symlink() or not evidence_path.is_file():
            raise MatrixError(f"{venue_id}: acceptance evidence file is unavailable")
        actual_sha256 = _file_sha256(evidence_path)
        if actual_sha256 != expected_sha256:
            raise MatrixError(f"{venue_id}: acceptance evidence SHA-256 mismatch")
        document = _load_json_object(evidence_path)
        if document.get("schema_version") != evidence_schema_version:
            raise MatrixError(
                f"{venue_id}: acceptance evidence schema_version mismatch"
            )
        rows.append(
            _historical_evidence_row(
                venue_id=venue_id,
                evidence_path=evidence_path,
                evidence_sha256=actual_sha256,
                document=document,
                repository_root=repository_root,
            )
        )
        seen.add(venue_id)
    return tuple(rows)


def _reject_cross_venue_evidence_reuse(rows: Sequence[Mapping[str, Any]]) -> None:
    identities: tuple[tuple[str, Callable[[Mapping[str, Any]], Any]], ...] = (
        ("database path", lambda row: row.get("database")),
        ("database SHA-256", lambda row: row.get("database_sha256")),
        (
            "report directory",
            lambda row: _mapping(_mapping(row.get("checks")).get("verify_passed")).get(
                "report_directory"
            ),
        ),
        (
            "report manifest hash",
            lambda row: _mapping(_mapping(row.get("checks")).get("verify_passed")).get(
                "manifest_hash"
            ),
        ),
    )
    for label, extract in identities:
        observed: dict[str, str] = {}
        for row in rows:
            venue = str(row.get("venue") or "")
            value = extract(row)
            if not venue or not isinstance(value, str) or not value:
                continue
            previous = observed.get(value)
            if previous is not None and previous != venue:
                raise MatrixError(
                    f"cross-venue {label} reuse is forbidden: {previous}, {venue}"
                )
            observed[value] = venue


def build_matrix(
    run_root: Path,
    *,
    database_relative_path: Path | None = None,
    venue_catalog_root: Path = DEFAULT_VENUE_CATALOG_ROOT,
    acceptance_import_manifest: Path | None = None,
) -> dict[str, Any]:
    root = run_root.resolve()
    live_venues = [
        summarize_venue(path, database_relative_path=database_relative_path)
        for path in discover_run_dirs(root)
    ]
    _reject_cross_venue_evidence_reuse(live_venues)
    imported_venues = list(_load_acceptance_imports(acceptance_import_manifest))
    venues = sorted(live_venues + imported_venues, key=lambda item: str(item["venue"]))
    expected_venues = set(_venue_catalog_ids(venue_catalog_root))
    observed_ids = [str(venue["venue"]) for venue in venues]
    observed_venues = set(observed_ids)
    duplicate_venues = sorted(
        {venue_id for venue_id in observed_ids if observed_ids.count(venue_id) > 1}
    )
    missing_venues = sorted(expected_venues - observed_venues)
    unexpected_venues = sorted(observed_venues - expected_venues)
    coverage_complete = not duplicate_venues and not missing_venues and not unexpected_venues
    passed = sum(1 for venue in venues if venue["passed"])
    return {
        "schema_version": "1",
        "generated_at": _utc_now(),
        "run_root": str(root),
        "summary": {
            "venue_count": len(venues),
            "catalog_venue_count": len(expected_venues),
            "passed": passed,
            "failed": len(venues) - passed,
            "coverage_complete": coverage_complete,
            "missing_venues": missing_venues,
            "unexpected_venues": unexpected_venues,
            "duplicate_venues": duplicate_venues,
            "all_passed": passed == len(venues) and coverage_complete,
        },
        "venue_catalog_root": str(venue_catalog_root.resolve()),
        "acceptance_import_manifest": (
            str(acceptance_import_manifest.resolve())
            if acceptance_import_manifest is not None
            else None
        ),
        "verifier": {
            "version": VERIFIER_VERSION,
            "path": str(Path(__file__).resolve()),
            "sha256": _file_sha256(Path(__file__).resolve()),
        },
        "venues": venues,
    }


def _portable_matrix(
    matrix: Mapping[str, Any],
    *,
    run_root: Path,
    repository_root: Path = ROOT,
) -> dict[str, Any]:
    """Replace machine-local absolute roots in durable acceptance evidence."""
    replacements = (
        (str(run_root.resolve()), "$RUN_ROOT"),
        (str(repository_root.resolve()), "$REPOSITORY_ROOT"),
    )

    def replace(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): replace(item) for key, item in value.items()}
        if isinstance(value, list):
            return [replace(item) for item in value]
        if isinstance(value, tuple):
            return [replace(item) for item in value]
        if not isinstance(value, str):
            return value
        portable_value = value
        for prefix, placeholder in replacements:
            portable_value = portable_value.replace(prefix, placeholder)
        return portable_value

    portable = replace(matrix)
    if not isinstance(portable, dict):  # pragma: no cover - mapping contract
        raise MatrixError("portable matrix transformation did not produce an object")
    portable["portable_paths"] = {
        "$RUN_ROOT": "directory supplied to --run-root",
        "$REPOSITORY_ROOT": "repository checkout root",
    }
    return portable


def _markdown_cell(check: Mapping[str, Any]) -> str:
    return "PASS" if check.get("passed") is True else "FAIL"


def _escape_markdown(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_markdown(matrix: Mapping[str, Any]) -> str:
    summary = matrix["summary"]
    lines = [
        "# Venue E2E Acceptance Matrix",
        "",
        (
            f"Result: {summary['passed']}/{summary['catalog_venue_count']} catalog venues passed; "
            f"run root: `{matrix['run_root']}`."
        ),
        "",
        (
            "Qualification: current matrix runs use approved one-record Stage 1 "
            "snapshots and a deterministic TEST_ONLY Stage 2 selector. They exercise "
            "the native adapter/persistence pipeline but do not prove live provider "
            "transport or the production Stage 2 model gate. Stage 3 uses real public "
            "PDF bytes; Stage 4 and Stage 4b use real Luna and one-shot Sol calls."
        ),
        "",
        (
            "Imported rows are hash-pinned historical attestations and are not presented "
            "as current-revision re-executions."
        ),
        "",
        "| Venue | Stage 1 | Stage 2 test-only | Stage 3 PDF | Luna full_pdf | Sol one-shot=1; reduce/audit/repair=0 | Stage4b implementation | Verify | Overall |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for venue in matrix["venues"]:
        checks = venue["checks"]
        stage4b = checks["stage4b_sol_one_shot_only"]
        implementation = stage4b.get("implementation_version") or "unknown"
        qualification = stage4b.get("implementation_qualification") or "unqualified"
        cells = [
            _escape_markdown(venue["venue"]),
            *(
                _markdown_cell(checks[name])
                for name in CHECK_ORDER
                if name != "verify_passed"
            ),
            _escape_markdown(f"{implementation} ({qualification})"),
            _markdown_cell(checks["verify_passed"]),
            "PASS" if venue["passed"] else "FAIL",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    if not summary["coverage_complete"]:
        lines.extend(("", "## Catalog coverage", ""))
        for label, key in (
            ("Missing", "missing_venues"),
            ("Unexpected", "unexpected_venues"),
            ("Duplicated", "duplicate_venues"),
        ):
            values = summary[key]
            if values:
                lines.append(f"- {label}: {_escape_markdown(', '.join(values))}")
    failures = [venue for venue in matrix["venues"] if not venue["passed"]]
    if failures:
        lines.extend(("", "## Failures", ""))
        for venue in failures:
            lines.append(f"### {_escape_markdown(venue['venue'])}")
            lines.append("")
            for name in CHECK_ORDER:
                check = venue["checks"][name]
                if check["passed"]:
                    continue
                detail = check.get("error") or check.get("expected")
                lines.append(f"- {CHECK_LABELS[name]}: {_escape_markdown(detail)}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize read-only per-venue Stage 1 through Stage 4b acceptance evidence."
    )
    parser.add_argument(
        "--run-root",
        required=True,
        type=Path,
        help="directory containing one independent child directory per venue",
    )
    parser.add_argument(
        "--database-relative-path",
        type=Path,
        help="optional database path relative to every venue run directory",
    )
    parser.add_argument(
        "--venue-catalog-root",
        type=Path,
        default=DEFAULT_VENUE_CATALOG_ROOT,
        help="descriptor directory whose venue_id values define complete coverage",
    )
    parser.add_argument(
        "--acceptance-import-manifest",
        type=Path,
        default=DEFAULT_ACCEPTANCE_IMPORT_MANIFEST,
        help=(
            "hash-pinned manifest for explicitly reused historical evidence "
            "(default: checked-in NeurIPS acceptance import)"
        ),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="JSON output path (default: RUN_ROOT/venue-e2e-matrix.json)",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        help="Markdown output path (default: RUN_ROOT/venue-e2e-matrix.md)",
    )
    parser.add_argument(
        "--portable-paths",
        action="store_true",
        help=(
            "replace absolute run/repository roots with $RUN_ROOT and "
            "$REPOSITORY_ROOT in generated evidence"
        ),
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    try:
        matrix = build_matrix(
            args.run_root,
            database_relative_path=args.database_relative_path,
            venue_catalog_root=args.venue_catalog_root,
            acceptance_import_manifest=args.acceptance_import_manifest,
        )
    except MatrixError as error:
        raise SystemExit(str(error)) from error
    if args.portable_paths:
        matrix = _portable_matrix(matrix, run_root=args.run_root)
    json_output = args.json_output or args.run_root / "venue-e2e-matrix.json"
    markdown_output = args.markdown_output or args.run_root / "venue-e2e-matrix.md"
    _write(json_output, json.dumps(matrix, ensure_ascii=False, indent=2) + "\n")
    _write(markdown_output, render_markdown(matrix))
    print(json.dumps({
        "json_output": str(json_output.resolve()),
        "markdown_output": str(markdown_output.resolve()),
        **matrix["summary"],
    }, ensure_ascii=False, sort_keys=True))
    return 0 if matrix["summary"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
