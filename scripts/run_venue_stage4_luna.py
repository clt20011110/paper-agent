#!/usr/bin/env python3
"""Run the native Stage 4 Luna boundary for one venue smoke directory.

The runner consumes only the paths and Stage 3 lineage frozen in ``run.json``.
It requires every Stage 3 checkpoint to be ``downloaded``, extracts each
unique PDF locally, and approves a two-day attended processing grant scoped to
the exact paper IDs and normalized-text SHA-256 digests.  By default it stops
after a Codex/runtime and authorization preflight.  A real Luna dispatch is
possible only with the explicit ``--execute-model`` flag.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from paper_agent.analysis import AnalysisInvoker
from paper_agent.analysis_cli_service import AnalysisCliService
from paper_agent.artifacts import ArtifactStore
from paper_agent.canonical import content_hash
from paper_agent.codex_exec import CodexExec, DoctorReport
from paper_agent.extraction import ExtractionStatus, PdfTextExtractor
from paper_agent.grants import GrantError, GrantStore
from paper_agent.processing import ArtifactProcessingPolicy
from paper_agent.storage import Database


ROOT = Path(__file__).resolve().parents[1]
MODEL = "gpt-5.6-luna"
PROFILE = "stage4_analysis_luna"
POLICY_PATH = ROOT / "policies" / "artifact-processing-v1.yaml"
OUTPUT_SCHEMA_PATH = ROOT / "schemas" / "paper-analysis.schema.json"
APPROVED_BY = "user-standing-authorization"


class Stage4RunnerError(RuntimeError):
    """The frozen venue run cannot safely advance through Stage 4."""


def _utc_timestamp(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _trusted_now(value: datetime | None) -> datetime:
    moment = datetime.now(UTC) if value is None else value
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise Stage4RunnerError("Stage 4 clock must be timezone-aware")
    return moment.astimezone(UTC)


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Stage4RunnerError(f"cannot read valid JSON: {path}") from error
    if not isinstance(value, dict):
        raise Stage4RunnerError(f"JSON document must be an object: {path}")
    return value


def _required_string(source: Mapping[str, Any], key: str) -> str:
    value = source.get(key)
    if not isinstance(value, str) or not value.strip():
        raise Stage4RunnerError(f"run.json native_pipeline.{key} is required")
    return value


def _resolve_run_path(run_dir: Path, value: str, *, label: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = run_dir / candidate
    resolved = candidate.resolve()
    if label == "database" and not resolved.is_file():
        raise Stage4RunnerError(f"run.json database does not exist: {resolved}")
    if label == "artifact_root" and not resolved.is_dir():
        raise Stage4RunnerError(f"run.json artifact_root does not exist: {resolved}")
    return resolved


def _stage3_pdf_inputs(database: Database, stage3_run_id: str) -> tuple[dict[str, str], ...]:
    run = database.connection.execute(
        "SELECT stage, status FROM pipeline_runs WHERE run_id = ?",
        (stage3_run_id,),
    ).fetchone()
    if run is None or tuple(run) != ("stage-3-download", "complete"):
        raise Stage4RunnerError(
            "stage3_run_id must name a complete stage-3-download pipeline run"
        )
    checkpoints = database.connection.execute(
        """SELECT paper_id, status FROM stage3_paper_results
           WHERE run_id = ? ORDER BY paper_id""",
        (stage3_run_id,),
    ).fetchall()
    if not checkpoints:
        raise Stage4RunnerError("Stage 3 has no paper checkpoints")
    incomplete = [
        f"{row['paper_id']}={row['status']}"
        for row in checkpoints
        if row["status"] != "downloaded"
    ]
    if incomplete:
        raise Stage4RunnerError(
            "Stage 4 requires every Stage 3 paper to be downloaded: "
            + ", ".join(incomplete)
        )

    selected: list[dict[str, str]] = []
    for checkpoint in checkpoints:
        paper_id = str(checkpoint["paper_id"])
        artifacts = database.connection.execute(
            """SELECT DISTINCT a.artifact_id, a.sha256
                 FROM download_attempts da
                 JOIN download_candidates dc ON dc.candidate_id = da.candidate_id
                 JOIN artifacts a ON a.artifact_id = da.artifact_id
                WHERE da.run_id = ? AND dc.paper_id = ?
                  AND da.result_status = 'downloaded'
                  AND a.paper_id = dc.paper_id
                  AND a.artifact_kind = 'pdf'
                  AND a.mime_type = 'application/pdf'
                  AND a.processing_status = 'available'
                ORDER BY a.artifact_id""",
            (stage3_run_id, paper_id),
        ).fetchall()
        if len(artifacts) != 1:
            raise Stage4RunnerError(
                f"downloaded Stage 3 paper {paper_id} must bind exactly one available PDF"
            )
        selected.append(
            {
                "paper_id": paper_id,
                "pdf_artifact_id": str(artifacts[0]["artifact_id"]),
                "pdf_sha256": str(artifacts[0]["sha256"]),
            }
        )
    return tuple(selected)


def _extract_normalized_text(
    database: Database,
    artifact_store: ArtifactStore,
    stage3_inputs: Sequence[Mapping[str, str]],
) -> tuple[dict[str, str], ...]:
    extractor = PdfTextExtractor(database, artifact_store)
    extracted_inputs: list[dict[str, str]] = []
    for item in stage3_inputs:
        result = extractor.extract(item["paper_id"], item["pdf_artifact_id"])
        if (
            result.status is not ExtractionStatus.FULL_TEXT_READY
            or not result.output_artifact_id
            or not result.normalized_text_sha256
        ):
            raise Stage4RunnerError(
                f"PDF text extraction is not full_text_ready for {item['paper_id']}: "
                f"{result.status.value}"
            )
        extracted_inputs.append(
            {
                **dict(item),
                "text_artifact_id": result.output_artifact_id,
                "normalized_text_sha256": result.normalized_text_sha256,
            }
        )
    return tuple(extracted_inputs)


def _grant_scope(extracted_inputs: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    return {
        "paper_ids": sorted(item["paper_id"] for item in extracted_inputs),
        "artifact_hashes": sorted(
            item["normalized_text_sha256"] for item in extracted_inputs
        ),
        "collection_ids": [],
        "collection_snapshot_hash": None,
        "selection_snapshot_hash": None,
        "domains": [],
        "provider": "codex_cli",
        "model": MODEL,
        "data_categories": ["normalized_text"],
    }


def _grant_matches(document: Mapping[str, Any], scope: Mapping[str, Any]) -> bool:
    return (
        document.get("kind") == "remote_model_processing"
        and document.get("actions") == ["remote_model_processing"]
        and document.get("purpose") == "internal_analysis"
        and document.get("mode") == "attended"
        and document.get("allow_unattended") is False
        and document.get("scope") == scope
        and document.get("max_papers") == len(scope["paper_ids"])
        and document.get("lineage_hash") is None
    )


def _approved_processing_grant(
    database: Database,
    extracted_inputs: Sequence[Mapping[str, str]],
    stage4_run_id: str,
    *,
    now: datetime,
) -> dict[str, Any]:
    grants = GrantStore(database)
    scope = _grant_scope(extracted_inputs)
    identity = content_hash(
        {
            "stage4_run_id": stage4_run_id,
            "scope": scope,
            "purpose": "internal_analysis",
            "mode": "attended",
        }
    )
    base_grant_id = f"grant-stage4-luna-{identity[:24]}"
    existing = database.connection.execute(
        "SELECT 1 FROM authorization_grants WHERE grant_id = ?",
        (base_grant_id,),
    ).fetchone()
    if existing is not None:
        try:
            loaded = grants.load(
                base_grant_id, kind="remote_model_processing", now=now
            ).document
        except GrantError as error:
            if "expired" not in str(error):
                raise Stage4RunnerError(str(error)) from error
        else:
            if not _grant_matches(loaded, scope):
                raise Stage4RunnerError(
                    f"existing Stage 4 grant has incompatible scope: {base_grant_id}"
                )
            return loaded
        grant_id = f"{base_grant_id}-{now.strftime('%Y%m%dT%H%M%S')}"
    else:
        grant_id = base_grant_id

    approved_at = _utc_timestamp(now)
    expires_at = _utc_timestamp(now + timedelta(days=2))
    draft = grants.create_draft(
        grant_id=grant_id,
        kind="remote_model_processing",
        actions=["remote_model_processing"],
        purpose="internal_analysis",
        mode="attended",
        allow_unattended=False,
        scope=scope,
        max_papers=len(extracted_inputs),
        expires_at=expires_at,
    )
    return grants.approve(
        draft,
        str(draft["content_hash"]),
        approved_by=APPROVED_BY,
        approved_at=approved_at,
    )


def _doctor(codex_factory: Callable[[], CodexExec]) -> DoctorReport:
    report = codex_factory().doctor(prove_model_availability=False)
    if not report.authenticated:
        raise Stage4RunnerError("Codex CLI is not authenticated")
    if report.model_availability.get(PROFILE) == "unavailable":
        raise Stage4RunnerError(f"{MODEL} is unavailable in the Codex model catalog")
    return report


def _stage4_evidence(
    database: Database, stage4_run_id: str, expected_paper_ids: Sequence[str]
) -> dict[str, Any]:
    run = database.connection.execute(
        "SELECT stage, status FROM pipeline_runs WHERE run_id = ?",
        (stage4_run_id,),
    ).fetchone()
    if run is None or tuple(run) != ("stage4", "complete"):
        raise Stage4RunnerError("Stage 4 pipeline did not finish complete")
    dispatches = database.connection.execute(
        """SELECT paper_id, status, dispatch_count, profile, model_id,
                  invocation_id, analysis_run_id, input_scope
             FROM analysis_dispatches WHERE run_id = ? ORDER BY paper_id""",
        (stage4_run_id,),
    ).fetchall()
    analyses = database.connection.execute(
        """SELECT paper_id, status, model_id, output_artifact_id
             FROM analysis_runs WHERE run_id = ? ORDER BY paper_id""",
        (stage4_run_id,),
    ).fetchall()
    expected = tuple(sorted(expected_paper_ids))
    dispatch_papers = tuple(str(row["paper_id"]) for row in dispatches)
    analysis_papers = tuple(str(row["paper_id"]) for row in analyses)
    valid_dispatches = (
        dispatch_papers == expected
        and all(
            row["status"] == "complete"
            and int(row["dispatch_count"]) == 1
            and row["profile"] == PROFILE
            and row["model_id"] == MODEL
            and bool(row["invocation_id"])
            and bool(row["analysis_run_id"])
            and row["input_scope"] == "full_pdf"
            for row in dispatches
        )
    )
    valid_analyses = (
        analysis_papers == expected
        and all(
            row["status"] == "complete"
            and row["model_id"] == MODEL
            and bool(row["output_artifact_id"])
            for row in analyses
        )
    )
    if not valid_dispatches or not valid_analyses:
        raise Stage4RunnerError(
            "Stage 4 persisted evidence does not prove one completed Luna invocation per paper"
        )
    return {
        "invocations": sum(int(row["dispatch_count"]) for row in dispatches),
        "paper_ids": list(expected),
        "input_scopes": sorted({str(row["input_scope"]) for row in dispatches}),
        "analysis_output_artifact_ids": [
            str(row["output_artifact_id"]) for row in analyses
        ],
    }


def _atomic_write_json(path: Path, document: Mapping[str, Any]) -> None:
    payload = json.dumps(
        document, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _update_run_json(
    run_path: Path,
    *,
    stage4_run_id: str,
    invocations: int,
) -> None:
    document = _load_object(run_path)
    native = document.get("native_pipeline")
    if not isinstance(native, dict):
        raise Stage4RunnerError("run.json native_pipeline must be an object")
    existing_run_id = native.get("stage4_run_id")
    if existing_run_id not in {None, stage4_run_id}:
        raise Stage4RunnerError(
            f"run.json is already bound to another Stage 4 run: {existing_run_id}"
        )
    native["stage4_run_id"] = stage4_run_id

    stages = document.get("stages")
    if not isinstance(stages, list):
        raise Stage4RunnerError("run.json stages must be an array")
    stage4_rows = [
        item for item in stages
        if isinstance(item, dict) and item.get("stage") == "stage4"
    ]
    if len(stage4_rows) != 1:
        raise Stage4RunnerError("run.json must contain exactly one Stage 4 status row")
    stage4_rows[0].update(
        {
            "status": "complete",
            "model": MODEL,
            "invocations": invocations,
        }
    )
    _atomic_write_json(run_path, document)


def run(
    run_dir: Path,
    *,
    execute_model: bool = False,
    stage4_run_id: str | None = None,
    now: datetime | None = None,
    codex_factory: Callable[[], CodexExec] | None = None,
    invoker_factory: Callable[[], AnalysisInvoker] | None = None,
) -> dict[str, Any]:
    """Preflight or execute one Stage 4 run without broadening its lineage."""

    resolved_run_dir = run_dir.resolve()
    run_path = resolved_run_dir / "run.json"
    document = _load_object(run_path)
    native = document.get("native_pipeline")
    if not isinstance(native, Mapping):
        raise Stage4RunnerError("run.json native_pipeline must be an object")
    database_path = _resolve_run_path(
        resolved_run_dir,
        _required_string(native, "database"),
        label="database",
    )
    artifact_root = _resolve_run_path(
        resolved_run_dir,
        _required_string(native, "artifact_root"),
        label="artifact_root",
    )
    stage3_run_id = _required_string(native, "stage3_run_id")
    bound_stage4_run_id = native.get("stage4_run_id")
    if bound_stage4_run_id is not None and (
        not isinstance(bound_stage4_run_id, str) or not bound_stage4_run_id
    ):
        raise Stage4RunnerError("run.json native_pipeline.stage4_run_id is invalid")
    if stage4_run_id is not None and bound_stage4_run_id not in {None, stage4_run_id}:
        raise Stage4RunnerError(
            f"requested Stage 4 run conflicts with run.json: {bound_stage4_run_id}"
        )
    base_run_id = document.get("run_id")
    if not isinstance(base_run_id, str) or not base_run_id:
        raise Stage4RunnerError("run.json run_id is required")
    resolved_stage4_run_id = (
        stage4_run_id or bound_stage4_run_id or f"{base_run_id}-stage4-luna"
    )
    current = _trusted_now(now)

    with Database(database_path) as database:
        database.migrate()
        artifact_store = ArtifactStore(artifact_root)
        stage3_inputs = _stage3_pdf_inputs(database, stage3_run_id)
        extracted_inputs = _extract_normalized_text(
            database, artifact_store, stage3_inputs
        )
        grant = _approved_processing_grant(
            database,
            extracted_inputs,
            resolved_stage4_run_id,
            now=current,
        )
        doctor = _doctor(codex_factory or CodexExec)
        service_options: dict[str, Any] = {}
        if invoker_factory is not None:
            service_options["invoker_factory"] = invoker_factory
        service = AnalysisCliService(
            database,
            artifact_store,
            ArtifactProcessingPolicy.load(POLICY_PATH),
            grants=GrantStore(database),
            clock=lambda: current,
            workers=1,
            allow_abstract_only=False,
            output_schema_path=OUTPUT_SCHEMA_PATH,
            **service_options,
        )
        service_result = service.run_from_stage3(
            resolved_stage4_run_id,
            stage3_run_id,
            expected_paper_ids=tuple(
                item["paper_id"] for item in extracted_inputs
            ),
            processing_grant_id=str(grant["grant_id"]),
            dry_run=not execute_model,
        )
        common = {
            "run_dir": str(resolved_run_dir),
            "stage3_run_id": stage3_run_id,
            "stage4_run_id": resolved_stage4_run_id,
            "model": MODEL,
            "profile": PROFILE,
            "grant_id": str(grant["grant_id"]),
            "grant_expires_at": str(grant["expires_at"]),
            "paper_ids": [item["paper_id"] for item in extracted_inputs],
            "normalized_text_artifacts": [
                {
                    "paper_id": item["paper_id"],
                    "artifact_id": item["text_artifact_id"],
                    "sha256": item["normalized_text_sha256"],
                }
                for item in extracted_inputs
            ],
            "runtime": {
                "authenticated": doctor.authenticated,
                "model_availability": doctor.model_availability[PROFILE],
                "version": doctor.version,
            },
        }
        if not execute_model:
            return {
                **common,
                "status": "preflight_complete",
                "execute_model": False,
                "invocations": 0,
                "input_scopes": list(service_result.input_scopes),
            }

        if service_result.result is None:
            raise Stage4RunnerError("Stage 4 execution returned no paper results")
        paper_results = service_result.result.papers
        failures = [
            {
                "paper_id": item.paper_id,
                "status": item.status,
                "error": item.error,
            }
            for item in paper_results
            if item.status != "complete"
        ]
        if failures:
            raise Stage4RunnerError(
                "Stage 4 Luna execution was incomplete: "
                + json.dumps(failures, ensure_ascii=False, sort_keys=True)
            )
        evidence = _stage4_evidence(
            database,
            resolved_stage4_run_id,
            tuple(item["paper_id"] for item in extracted_inputs),
        )

    result = {
        **common,
        **evidence,
        "status": "complete",
        "execute_model": True,
        "papers": [
            {
                "paper_id": item.paper_id,
                "status": item.status,
                "resumed": item.resumed,
                "input_scope": item.input_scope,
            }
            for item in paper_results
        ],
    }
    _atomic_write_json(resolved_run_dir / "stage4" / "result.json", result)
    _update_run_json(
        run_path,
        stage4_run_id=resolved_stage4_run_id,
        invocations=int(evidence["invocations"]),
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="venue smoke directory containing run.json",
    )
    parser.add_argument(
        "--stage4-run-id",
        help="override the deterministic Stage 4 run ID before it is bound",
    )
    parser.add_argument(
        "--execute-model",
        action="store_true",
        help="explicitly authorize the real gpt-5.6-luna dispatch",
    )
    args = parser.parse_args(argv)
    try:
        result = run(
            args.run_dir,
            execute_model=args.execute_model,
            stage4_run_id=args.stage4_run_id,
        )
    except (Stage4RunnerError, GrantError, ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
