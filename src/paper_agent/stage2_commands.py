"""Thin command adapters for released Stage 2 screening and gate artifacts."""

from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .canonical import content_hash
from .schema import schema_directory
from .stage2_backends import OmlxTransport, UrlLibOmlxTransport
from .stage2_benchmark import (
    BenchmarkRunSpec,
    MacOSMemoryObserver,
    Stage2BenchmarkRunner,
)
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
from .stage2_pipeline import Stage2Paper
from .stage2_search import (
    ReleasedStage2,
    load_stage2_benchmark_candidate,
    load_stage2_release,
)
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
        soak_record = _soak_record(_object(soak_record_path))
        soak = soak_gate(soak_manifest, soak_record)
        result["soak"] = {
            "failures": list(soak.failures),
            "manifest_hash": soak_manifest.hash(),
            "passed": soak.passed,
            "service_request_failure_rate": soak_record.service_request_failure_rate,
        }
    result["status"] = "passed" if performance.passed and result.get("soak", {}).get("passed", True) else "failed"
    return result


def measure_stage2_benchmark(
    *,
    manifest_path: Path,
    papers_path: Path,
    candidate_path: Path,
    environment_path: Path,
    database_path: Path,
    output_path: Path,
    scenario: str,
    run_id: str,
    omlx_pids: Sequence[int],
    sample_interval_seconds: float = 0.25,
    dry_run: bool = False,
    candidate_loader: Callable[[Path], ReleasedStage2] = load_stage2_benchmark_candidate,
    transport: OmlxTransport | None = None,
    memory_observer: MacOSMemoryObserver | None = None,
) -> dict[str, Any]:
    """Run one production-scale benchmark against the released local oMLX service."""

    if not run_id:
        raise ValueError("measured Stage 2 benchmark requires --run-id")
    if scenario not in {"normal", "stress", "soak"}:
        raise ValueError("measured Stage 2 scenario must be normal, stress, or soak")
    if not dry_run and output_path.exists():
        raise FileExistsError(f"benchmark execution record already exists: {output_path}")
    if dry_run and not database_path.is_file():
        raise FileNotFoundError(f"benchmark dry-run database does not exist: {database_path}")

    release = candidate_loader(candidate_path)
    spec = _benchmark_spec(_object(manifest_path), scenario)
    papers = _benchmark_papers(papers_path)
    observed_corpus_hash = benchmark_corpus_hash(papers)
    if observed_corpus_hash != spec.corpus_hash:
        raise ValueError("benchmark papers do not match the frozen manifest corpus hash")
    environment = _environment(_object(environment_path))
    observer = memory_observer or MacOSMemoryObserver.current(omlx_pids)
    observer.preflight(environment)
    api_key = os.environ.get(release.api_key_env) if release.api_key_env else None
    if release.api_key_env and not api_key:
        raise ValueError(
            f"measured Stage 2 benchmark requires environment variable {release.api_key_env}"
        )
    local_transport = transport or UrlLibOmlxTransport(
        release.omlx_base_url,
        api_key=api_key,
    )
    schema = json.loads(
        (schema_directory() / release.profile.schema_version).read_text(encoding="utf-8")
    )

    with Database(database_path, read_only=dry_run) as database:
        if not dry_run:
            database.migrate()
        runner = Stage2BenchmarkRunner.from_omlx(
            database=database,
            profile=release.profile,
            transport=local_transport,
            schema=schema,
            environment=environment,
            release_hash=release.release_hash,
            rss_sampler=observer.current_rss_bytes,
            rss_scope=observer.rss_scope,
            rss_sample_interval_seconds=sample_interval_seconds,
            memory_pressure_sampler=observer.memory_pressure_critical,
        )
        runner.validate(spec, papers)
        if dry_run:
            return {
                "artifact_path": str(output_path),
                "case_count": len(spec.cases),
                "command": "benchmark-stage2.measure",
                "kind": spec.kind,
                "manifest_hash": spec.manifest_hash,
                "rss_scope": observer.rss_scope,
                "run_id": run_id,
                "scenario": spec.scenario,
                "status": "validated",
            }
        record = runner.run(spec, papers, run_id=run_id)

    record.write(output_path)
    document = record.document()
    return {
        "artifact_hash": record.hash(),
        "artifact_path": str(output_path),
        "case_count": document["case_count"],
        "command": "benchmark-stage2.measure",
        "kind": document["kind"],
        "manifest_hash": document["manifest_hash"],
        "record_version": document["record_version"],
        "request_failure_rate": document["request_failure_rate"],
        "rss_scope": document["rss_scope"],
        "run_id": document["run_id"],
        "scenario": document["scenario"],
        "service_request_count": document["service_request_count"],
        "service_request_failure_rate": document["service_request_failure_rate"],
        "status": "complete",
    }


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


def _benchmark_spec(value: Mapping[str, Any], scenario: str) -> BenchmarkRunSpec:
    if scenario == "soak":
        return BenchmarkRunSpec.soak(_soak_manifest(value))
    return BenchmarkRunSpec.performance(_performance_manifest(value), scenario)


def _benchmark_papers(path: Path) -> tuple[Stage2Paper, ...]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        if set(value) != {"papers"}:
            raise ValueError("benchmark papers object must contain only the papers array")
        value = value["papers"]
    if not isinstance(value, list) or not value:
        raise ValueError("benchmark papers must be a non-empty JSON array")
    allowed = {
        "paper_id",
        "title",
        "abstract",
        "keywords",
        "document_type",
        "possibly_truncated",
        "multi_condition_conflict",
        "language_anomaly",
    }
    papers: list[Stage2Paper] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict) or not set(item) <= allowed:
            raise ValueError(f"benchmark paper {index} has invalid fields")
        keywords = item.get("keywords", [])
        if not isinstance(keywords, list) or not all(isinstance(keyword, str) for keyword in keywords):
            raise ValueError(f"benchmark paper {index} keywords must be strings")
        paper_id = item.get("paper_id")
        title = item.get("title")
        if not isinstance(paper_id, str) or not isinstance(title, str):
            raise ValueError(f"benchmark paper {index} paper_id and title must be strings")
        abstract = item.get("abstract")
        document_type = item.get("document_type")
        if abstract is not None and not isinstance(abstract, str):
            raise ValueError(f"benchmark paper {index} abstract must be a string or null")
        if document_type is not None and not isinstance(document_type, str):
            raise ValueError(f"benchmark paper {index} document_type must be a string or null")
        flags = {
            name: item.get(name, False)
            for name in ("possibly_truncated", "multi_condition_conflict", "language_anomaly")
        }
        if not all(isinstance(flag, bool) for flag in flags.values()):
            raise ValueError(f"benchmark paper {index} flags must be booleans")
        papers.append(Stage2Paper(
            paper_id=paper_id,
            title=title,
            abstract=abstract,
            keywords=tuple(keywords),
            document_type=document_type,
            **flags,
        ))
    return tuple(papers)


def benchmark_corpus_hash(papers: Sequence[Stage2Paper]) -> str:
    """Hash the normalized paper content consumed by the measured runner."""

    return content_hash({
        "schema_version": 1,
        "papers": [
            {
                "paper_id": paper.paper_id,
                "title": paper.title,
                "abstract": paper.abstract,
                "keywords": list(paper.keywords),
                "document_type": paper.document_type,
                "possibly_truncated": paper.possibly_truncated,
                "multi_condition_conflict": paper.multi_condition_conflict,
                "language_anomaly": paper.language_anomaly,
            }
            for paper in sorted(papers, key=lambda item: item.paper_id)
        ],
    })


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
    _require_record_v2(value, "performance")
    return PerformanceRunRecord(
        record_version=int(value["record_version"]),
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
        service_request_count=int(value["service_request_count"]),
        service_failed_request_count=int(value["service_failed_request_count"]),
        resume_verified=bool(value["resume_verified"]),
        resume_model_call_count=int(value["resume_model_call_count"]),
        resumed_pair_count=int(value["resumed_pair_count"]),
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
    _require_record_v2(value, "soak")
    return SoakRunRecord(
        record_version=int(value["record_version"]),
        run_id=str(value["run_id"]),
        manifest_hash=str(value["manifest_hash"]),
        stage2_config_hash=str(value["stage2_config_hash"]),
        model_lock_hashes=tuple(value["model_lock_hashes"]),
        duration_seconds=float(value["duration_seconds"]),
        peak_memory_gb=float(value["peak_memory_gb"]),
        request_count=int(value["request_count"]),
        failed_request_count=int(value["failed_request_count"]),
        service_request_count=int(value["service_request_count"]),
        service_failed_request_count=int(value["service_failed_request_count"]),
        resume_verified=bool(value["resume_verified"]),
        resume_model_call_count=int(value["resume_model_call_count"]),
        resumed_pair_count=int(value["resumed_pair_count"]),
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


def _require_record_v2(value: Mapping[str, Any], kind: str) -> None:
    version = value.get("record_version")
    if type(version) is not int or version != 2:
        observed = "missing" if version is None else repr(version)
        raise ValueError(
            f"{kind} benchmark record_version 2 is required; observed {observed}. "
            "Legacy records do not contain audited oMLX service-request metrics and must be rerun."
        )
    required_fields = (
        "service_request_count",
        "service_failed_request_count",
        "resume_verified",
        "resume_model_call_count",
        "resumed_pair_count",
    )
    missing = sorted(set(required_fields) - set(value))
    if missing:
        raise ValueError(f"{kind} benchmark v2 record is missing fields: {missing}")
    count_fields = (
        "service_request_count",
        "service_failed_request_count",
        "resume_model_call_count",
        "resumed_pair_count",
    )
    if any(type(value[field]) is not int for field in count_fields):
        raise ValueError(f"{kind} benchmark v2 service request counts must be integers")
    if type(value["resume_verified"]) is not bool:
        raise ValueError(f"{kind} benchmark v2 resume_verified must be a boolean")
