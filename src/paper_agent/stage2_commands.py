"""Thin command adapters for released Stage 2 screening and gate artifacts."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .stage2_evaluation import (
    BenchmarkEnvironment,
    PerformanceCase,
    PerformanceRoutingManifest,
    PerformanceRunRecord,
    SoakManifest,
    SoakRunRecord,
    performance_gate,
    performance_summary,
    soak_gate,
)
from .stage2_search import ReleasedStage2, load_stage2_release
from .storage import Database


def filter_database(
    *,
    plan_path: Path,
    release_path: Path,
    database_path: Path,
    campaign_id: str,
    paper_ids: Sequence[str] = (),
    dry_run: bool = False,
    release_loader: Callable[[Path, Mapping[str, Any]], ReleasedStage2] = load_stage2_release,
) -> dict[str, Any]:
    """Screen an explicit or complete canonical-paper set with one frozen release."""
    plan = _object(plan_path)
    release = release_loader(release_path, plan)
    if not database_path.is_file():
        raise FileNotFoundError(f"database does not exist: {database_path}")

    with Database(database_path, read_only=dry_run) as database:
        if not dry_run:
            database.migrate()
        selected = _paper_ids(database, paper_ids)
        if dry_run:
            return {
                "campaign_id": campaign_id,
                "command": "filter",
                "database_path": str(database_path),
                "paper_count": len(selected),
                "paper_ids": list(selected),
                "profile": release.profile_name,
                "release_hash": release.release_hash,
                "status": "validated",
            }

        screener = release.screener(database, campaign_id)
        statuses = screener.screen(selected)
        counts = Counter(status.value for status in statuses.values())
        return {
            "campaign_id": campaign_id,
            "command": "filter",
            "counts": {name: counts[name] for name in sorted(counts)},
            "decisions": {
                paper_id: statuses[paper_id].value for paper_id in sorted(statuses)
            },
            "paper_count": len(selected),
            "profile": release.profile_name,
            "release_hash": release.release_hash,
            "stage2_run_ids": list(screener.run_ids),
            "status": "complete",
        }


def evaluate_benchmark_artifacts(
    *,
    manifest_path: Path,
    record_paths: Sequence[Path],
    soak_manifest_path: Path | None = None,
    soak_record_path: Path | None = None,
) -> dict[str, Any]:
    """Apply the frozen performance and optional soak gates to measured records."""
    manifest = _performance_manifest(_object(manifest_path))
    records = tuple(
        _performance_record(document)
        for path in record_paths
        for document in _objects(path)
    )
    performance = performance_gate(manifest, records)
    result: dict[str, Any] = {
        "command": "benchmark-stage2",
        "manifest_hash": manifest.hash(),
        "performance": {
            "failures": list(performance.failures),
            "passed": performance.passed,
            "record_count": len(records),
        },
    }
    for scenario in ("normal", "stress"):
        if sum(record.scenario == scenario for record in records) == 3:
            result["performance"][scenario] = dict(
                performance_summary(records, scenario)
            )

    if (soak_manifest_path is None) != (soak_record_path is None):
        raise ValueError("--soak-manifest and --soak-record must be supplied together")
    if soak_manifest_path is not None and soak_record_path is not None:
        soak_manifest = _soak_manifest(_object(soak_manifest_path))
        soak = soak_gate(soak_manifest, _soak_record(_object(soak_record_path)))
        result["soak"] = {
            "failures": list(soak.failures),
            "manifest_hash": soak_manifest.hash(),
            "passed": soak.passed,
        }
    result["status"] = "passed" if performance.passed and result.get("soak", {}).get("passed", True) else "failed"
    return result


def _paper_ids(database: Database, requested: Sequence[str]) -> tuple[str, ...]:
    available = {
        str(row["paper_id"])
        for row in database.connection.execute(
            "SELECT paper_id FROM papers ORDER BY paper_id"
        ).fetchall()
    }
    selected = tuple(sorted(set(requested))) if requested else tuple(sorted(available))
    missing = sorted(set(selected) - available)
    if missing:
        raise ValueError(f"filter papers do not exist: {missing}")
    return selected


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _objects(path: Path) -> tuple[dict[str, Any], ...]:
    value = json.loads(path.read_text(encoding="utf-8"))
    documents = value if isinstance(value, list) else [value]
    if not all(isinstance(item, dict) for item in documents):
        raise ValueError(f"expected a JSON object or array of objects: {path}")
    return tuple(documents)


def _cases(value: Sequence[Mapping[str, Any]]) -> tuple[PerformanceCase, ...]:
    return tuple(
        PerformanceCase(
            pair_id=str(item["pair_id"]),
            input_tokens=int(item["input_tokens"]),
            abstract_missing=bool(item.get("abstract_missing", False)),
        )
        for item in value
    )


def _environment(value: Mapping[str, Any]) -> BenchmarkEnvironment:
    return BenchmarkEnvironment(
        machine_model=str(value["machine_model"]),
        memory_gb=int(value["memory_gb"]),
        macos_version=str(value["macos_version"]),
        omlx_version=str(value["omlx_version"]),
        mlx_version=str(value["mlx_version"]),
        power_mode=str(value["power_mode"]),
        background_load=str(value["background_load"]),
        batch_config={str(key): int(item) for key, item in value["batch_config"].items()},
        resident_model_instances={
            str(key): int(item)
            for key, item in value["resident_model_instances"].items()
        },
    )


def _performance_manifest(value: Mapping[str, Any]) -> PerformanceRoutingManifest:
    return PerformanceRoutingManifest(
        version=int(value["version"]),
        corpus_hash=str(value["corpus_hash"]),
        stage2_config_hash=str(value["stage2_config_hash"]),
        model_lock_hashes=tuple(value["model_lock_hashes"]),
        threshold_artifact_hashes=tuple(value["threshold_artifact_hashes"]),
        output_token_limit=int(value["output_token_limit"]),
        cases=_cases(value["cases"]),
        normal_qwen_ids=frozenset(value["normal_qwen_ids"]),
        stress_qwen_ids=frozenset(value["stress_qwen_ids"]),
        pipeline_components=tuple(value.get("pipeline_components", ("rules", "reranker", "qwen", "schema_validation", "sqlite_commit"))),
    )


def _performance_record(value: Mapping[str, Any]) -> PerformanceRunRecord:
    return PerformanceRunRecord(
        scenario=str(value["scenario"]),
        run_id=str(value["run_id"]),
        manifest_hash=str(value["manifest_hash"]),
        stage2_config_hash=str(value["stage2_config_hash"]),
        model_lock_hashes=tuple(value["model_lock_hashes"]),
        duration_seconds=float(value["duration_seconds"]),
        p50_seconds=float(value["p50_seconds"]),
        p95_seconds=float(value["p95_seconds"]),
        peak_memory_gb=float(value["peak_memory_gb"]),
        request_count=int(value["request_count"]),
        failed_request_count=int(value["failed_request_count"]),
        completed_pair_ids=tuple(value["completed_pair_ids"]),
        needs_review_pair_ids=tuple(value["needs_review_pair_ids"]),
        failed_request_pair_ids=tuple(value["failed_request_pair_ids"]),
        qwen_pair_ids=tuple(value["qwen_pair_ids"]),
        environment=_environment(value["environment"]),
        executed_components=tuple(value["executed_components"]),
        sqlite_commit_count=int(value["sqlite_commit_count"]),
        warmed=bool(value["warmed"]),
        oom=bool(value.get("oom", False)),
        process_crash=bool(value.get("process_crash", False)),
        memory_pressure_critical=bool(value.get("memory_pressure_critical", False)),
        unbounded_memory_growth=bool(value.get("unbounded_memory_growth", False)),
    )


def _soak_manifest(value: Mapping[str, Any]) -> SoakManifest:
    return SoakManifest(
        version=int(value["version"]),
        corpus_hash=str(value["corpus_hash"]),
        stage2_config_hash=str(value["stage2_config_hash"]),
        model_lock_hashes=tuple(value["model_lock_hashes"]),
        threshold_artifact_hashes=tuple(value["threshold_artifact_hashes"]),
        output_token_limit=int(value["output_token_limit"]),
        cases=_cases(value["cases"]),
    )


def _soak_record(value: Mapping[str, Any]) -> SoakRunRecord:
    return SoakRunRecord(
        run_id=str(value["run_id"]),
        manifest_hash=str(value["manifest_hash"]),
        stage2_config_hash=str(value["stage2_config_hash"]),
        model_lock_hashes=tuple(value["model_lock_hashes"]),
        duration_seconds=float(value["duration_seconds"]),
        peak_memory_gb=float(value["peak_memory_gb"]),
        request_count=int(value["request_count"]),
        failed_request_count=int(value["failed_request_count"]),
        completed_pair_ids=tuple(value["completed_pair_ids"]),
        needs_review_pair_ids=tuple(value["needs_review_pair_ids"]),
        failed_request_pair_ids=tuple(value["failed_request_pair_ids"]),
        environment=_environment(value["environment"]),
        executed_components=tuple(value["executed_components"]),
        sqlite_commit_count=int(value["sqlite_commit_count"]),
        warmed=bool(value["warmed"]),
        oom=bool(value.get("oom", False)),
        process_crash=bool(value.get("process_crash", False)),
        memory_pressure_critical=bool(value.get("memory_pressure_critical", False)),
        unbounded_memory_growth=bool(value.get("unbounded_memory_growth", False)),
    )
