"""Thin command adapters for released Stage 2 screening and gate artifacts."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Mapping, Sequence

from .canonical import content_hash
from .schema import schema_directory
from .stage2_backends import ModelLock, OmlxTransport, UrlLibOmlxTransport
from .stage2_benchmark import (
    BenchmarkRunSpec,
    MacOSMemoryObserver,
    Stage2BenchmarkRunner,
)
from .stage2_benchmark_inputs import (
    benchmark_corpus_hash,
    benchmark_papers_from_document,
)
from .stage2_benchmark_freeze import (
    freeze_candidate_benchmark_manifests,
    publish_candidate_benchmark_manifests,
)
from .stage2_evaluation import (
    BenchmarkEnvironment,
    CalibrationPath,
    PerformanceCase,
    PerformanceRoutingManifest,
    PerformanceRunRecord,
    SoakManifest,
    SoakRunRecord,
    performance_gate,
    performance_summary,
    soak_gate,
    load_gold_manifest,
)
from .stage2_candidate import build_stage2_candidate_bundle
from .stage2_dev_calibration import (
    Stage2DevRawScoreRunner,
    dev_scoring_cases,
    load_frozen_dev_raw_scores,
)
from .stage2_pipeline import (
    ERROR_RATE_ALARM,
    MEMORY_WATERMARK_ALARM,
    Stage2Paper,
    Stage2Profile,
)
from .stage2_hidden_submission import (
    HiddenPromotionSubmissionRunner,
    hidden_submission_cases,
)
from .stage2_search import (
    ReleasedStage2,
    load_stage2_benchmark_candidate,
    load_stage2_release,
    stage2_base_profile,
)
from .stage2_promotion_artifacts import (
    load_private_gold_labels,
    promotion_submission_document,
)
from .stage2_sampling import load_private_corpus_snapshot
from .stage2_structured_replay import (
    StructuredReplayRunner,
    freeze_structured_replay_manifest,
)
from .storage import Database


def filter_database(
    *,
    plan_path: Path,
    release_path: Path,
    database_path: Path,
    campaign_id: str,
    paper_ids: Sequence[str] | None = None,
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
                "alarm_codes": [],
                "campaign_id": campaign_id,
                "command": "filter",
                "database_path": str(database_path),
                "paper_count": len(selected),
                "paper_ids": list(selected),
                "profile": release.profile_name,
                "release_hash": release.release_hash,
                "stage2": None,
                "status": "validated",
            }

        screener = release.screener(database, campaign_id)
        statuses = screener.screen(selected)
        counts = Counter(status.value for status in statuses.values())
        telemetry_method = getattr(screener, "telemetry", None)
        stage2 = dict(telemetry_method()) if callable(telemetry_method) else {}
        alarm_codes = list(stage2.get("alarm_codes", ()))
        return {
            "alarm_codes": alarm_codes,
            "campaign_id": campaign_id,
            "command": "filter",
            "counts": {name: counts[name] for name in sorted(counts)},
            "decisions": {
                paper_id: statuses[paper_id].value for paper_id in sorted(statuses)
            },
            "paper_count": len(selected),
            "profile": release.profile_name,
            "release_hash": release.release_hash,
            "stage2": stage2,
            "stage2_run_ids": list(screener.run_ids),
            "status": "incomplete" if ERROR_RATE_ALARM in alarm_codes else "complete",
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


def freeze_stage2_dev_scores(
    *,
    manifest_path: Path,
    snapshot_path: Path,
    topic_queries_path: Path,
    runtime_path: Path,
    reranker_lock_path: Path,
    adjudicator_lock_path: Path,
    output_path: Path,
    dry_run: bool = False,
    transport: OmlxTransport | None = None,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Freeze both unlabelled DEV score paths under evaluator custody."""

    if os.path.lexists(output_path):
        raise FileExistsError(f"DEV raw-score output already exists: {output_path}")
    manifest = load_gold_manifest(manifest_path)
    snapshot = load_private_corpus_snapshot(snapshot_path)
    runtime = _object(runtime_path)
    profile, model_lock_hashes = _stage2_input_profile(
        runtime,
        reranker_lock_path,
        adjudicator_lock_path,
    )
    topic_queries = _topic_queries(_object(topic_queries_path))
    cases = dev_scoring_cases(manifest, snapshot)
    expected_query_keys = {(case.topic, case.language) for case in cases}
    if set(topic_queries) != expected_query_keys:
        raise ValueError(
            "topic query input must exactly cover DEV topic-language combinations"
        )
    if dict(topic_queries) != profile.evaluation_topic_query_map:
        raise ValueError("topic query input does not match the frozen Stage 2 runtime")
    if dry_run:
        return {
            "case_count": len(cases),
            "command": "stage2-calibration.freeze-dev-scores",
            "dev_manifest_hash": manifest.dev_hash(),
            "gold_manifest_hash": manifest.hash(),
            "output": str(output_path),
            "stage2_config_hash": profile.base_runtime_config_hash,
            "status": "validated",
            "topic_query_count": len(topic_queries),
            "written": False,
        }

    values = environment if environment is not None else os.environ
    api_key = values.get(profile.api_key_env) if profile.api_key_env else None
    if profile.api_key_env and not api_key:
        raise ValueError(
            f"DEV scoring requires environment variable {profile.api_key_env}"
        )
    local_transport = transport or UrlLibOmlxTransport(
        profile.omlx_base_url,
        api_key=api_key,
    )
    artifact = Stage2DevRawScoreRunner(
        profile,
        local_transport,
        model_lock_hashes,
        topic_queries,
    ).run(manifest, snapshot, output_path=output_path)
    return {
        "artifact_hash": artifact.hash(),
        "case_count": len(cases),
        "command": "stage2-calibration.freeze-dev-scores",
        "dev_manifest_hash": artifact.dev_manifest_hash,
        "gold_manifest_hash": artifact.gold_manifest_hash,
        "output": str(output_path),
        "output_sha256": sha256(output_path.read_bytes()).hexdigest(),
        "qwen_retry_count": artifact.qwen_retry_count,
        "stage2_config_hash": artifact.stage2_config_hash,
        "status": "complete",
        "topic_query_count": len(artifact.topic_queries),
        "written": True,
    }


def build_stage2_candidate(
    *,
    manifest_path: Path,
    private_labels_path: Path,
    raw_scores_path: Path,
    runtime_path: Path,
    reranker_lock_path: Path,
    adjudicator_lock_path: Path,
    candidate_id: str,
    output_dir: Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Build the DEV-calibrated schema-v2 candidate, without opening hidden labels."""

    if os.path.lexists(output_dir):
        raise FileExistsError(f"Stage 2 candidate output already exists: {output_dir}")
    manifest = load_gold_manifest(manifest_path)
    private_labels = load_private_gold_labels(
        private_labels_path,
        manifest=manifest,
    )
    raw_scores = load_frozen_dev_raw_scores(raw_scores_path)
    runtime = _object(runtime_path)

    if dry_run:
        with TemporaryDirectory(prefix="paper-agent-stage2-candidate-") as directory:
            result = build_stage2_candidate_bundle(
                manifest=manifest,
                private_labels=private_labels,
                raw_scores=raw_scores,
                runtime=runtime,
                reranker_lock_path=reranker_lock_path,
                adjudicator_lock_path=adjudicator_lock_path,
                candidate_id=candidate_id,
                output_dir=Path(directory) / "candidate",
            )
    else:
        result = build_stage2_candidate_bundle(
            manifest=manifest,
            private_labels=private_labels,
            raw_scores=raw_scores,
            runtime=runtime,
            reranker_lock_path=reranker_lock_path,
            adjudicator_lock_path=adjudicator_lock_path,
            candidate_id=candidate_id,
            output_dir=output_dir,
        )
    return {
        "candidate_id": result.release.profile_name,
        "candidate_path": str(output_dir / "stage2-candidate-v2.json"),
        "command": "stage2-calibration.build-candidate",
        "dev_label_hash": result.dev_label_hash,
        "gold_manifest_hash": manifest.hash(),
        "raw_score_hash": result.raw_score_hash,
        "release_hash": result.release.release_hash,
        "selections": {
            path.value: dict(result.selections[path])
            for path in sorted(result.selections, key=str)
        },
        "stage2_config_hash": result.release.profile.base_runtime_config_hash,
        "status": "validated" if dry_run else "complete",
        "written": not dry_run,
    }


def freeze_stage2_benchmark_manifests(
    *,
    candidate_path: Path,
    performance_papers_path: Path,
    soak_papers_path: Path,
    selection_receipt_path: Path,
    performance_output: Path,
    soak_output: Path,
    dry_run: bool = False,
    candidate_loader: Callable[[Path], ReleasedStage2] = load_stage2_benchmark_candidate,
) -> dict[str, Any]:
    """Bind candidate-independent 1k/10k workloads to one schema-v2 candidate."""

    if performance_output == soak_output:
        raise ValueError("performance and soak manifest outputs must differ")
    existing = next(
        (
            path
            for path in (performance_output, soak_output)
            if os.path.lexists(path)
        ),
        None,
    )
    if existing is not None:
        raise FileExistsError(f"benchmark manifest output already exists: {existing}")
    release = candidate_loader(candidate_path)
    performance_papers = _benchmark_papers(performance_papers_path)
    soak_papers = _benchmark_papers(soak_papers_path)
    manifests = freeze_candidate_benchmark_manifests(
        release,
        performance_papers=performance_papers,
        soak_papers=soak_papers,
        selection_receipt=_object(selection_receipt_path),
    )
    if not dry_run:
        publish_candidate_benchmark_manifests(
            manifests,
            performance_output=performance_output,
            soak_output=soak_output,
        )
    return {
        "command": "benchmark-stage2.freeze-manifests",
        "performance_case_count": len(manifests.performance.cases),
        "performance_manifest_hash": manifests.performance.hash(),
        "performance_output": str(performance_output),
        "soak_case_count": len(manifests.soak.cases),
        "soak_manifest_hash": manifests.soak.hash(),
        "soak_output": str(soak_output),
        "stage2_config_hash": release.profile.base_runtime_config_hash,
        "status": "validated" if dry_run else "complete",
        "written": not dry_run,
    }


def build_hidden_promotion_submission(
    *,
    manifest_path: Path,
    snapshot_path: Path,
    candidate_path: Path,
    output_path: Path,
    dry_run: bool = False,
    candidate_loader: Callable[[Path], ReleasedStage2] = load_stage2_benchmark_candidate,
    transport: OmlxTransport | None = None,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Create three candidate predictions over the sealed hidden universe."""

    if os.path.lexists(output_path):
        raise FileExistsError(
            f"hidden promotion submission output already exists: {output_path}"
        )
    manifest = load_gold_manifest(manifest_path)
    snapshot = load_private_corpus_snapshot(snapshot_path)
    release = candidate_loader(candidate_path)
    cases = hidden_submission_cases(manifest, snapshot, release.profile)
    if dry_run:
        return {
            "candidate_id": release.profile_name,
            "case_count": len(cases),
            "command": "stage2-evaluator.predict-hidden",
            "gold_manifest_hash": manifest.hash(),
            "output": str(output_path),
            "run_count": 3,
            "stage2_config_hash": release.profile.base_runtime_config_hash,
            "status": "validated",
            "written": False,
        }
    values = environment if environment is not None else os.environ
    api_key = values.get(release.api_key_env) if release.api_key_env else None
    if release.api_key_env and not api_key:
        raise ValueError(
            "hidden prediction requires environment variable "
            f"{release.api_key_env}"
        )
    local_transport = transport or UrlLibOmlxTransport(
        release.omlx_base_url,
        api_key=api_key,
    )
    submission = HiddenPromotionSubmissionRunner(release, local_transport).run(
        manifest,
        snapshot,
        output_path=output_path,
    )
    document = promotion_submission_document(submission)
    return {
        "candidate_id": submission.candidate_id,
        "case_count": len(submission.runs[0]),
        "command": "stage2-evaluator.predict-hidden",
        "gold_manifest_hash": manifest.hash(),
        "output": str(output_path),
        "output_sha256": sha256(output_path.read_bytes()).hexdigest(),
        "run_count": len(submission.runs),
        "stage2_config_hash": release.profile.base_runtime_config_hash,
        "status": "complete",
        "submission_hash": content_hash(document),
        "written": True,
    }


def run_structured_replay(
    *,
    papers_path: Path,
    candidate_path: Path,
    manifest_output: Path,
    records_output: Path,
    dry_run: bool = False,
    candidate_loader: Callable[[Path], ReleasedStage2] = load_stage2_benchmark_candidate,
    transport: OmlxTransport | None = None,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Run the frozen 1,000-request adjudicator replay against local oMLX."""

    if manifest_output == records_output:
        raise ValueError("structured replay outputs must use different paths")
    existing = next(
        (path for path in (manifest_output, records_output) if path.exists()),
        None,
    )
    if existing is not None:
        raise FileExistsError(f"refusing to replace structured replay artifact: {existing}")

    release = candidate_loader(candidate_path)
    papers = _benchmark_papers(papers_path)
    manifest = freeze_structured_replay_manifest(papers, release.profile)
    if dry_run:
        return {
            "case_count": len(papers),
            "command": "stage2-replay",
            "corpus_hash": manifest.corpus_hash,
            "manifest_hash": manifest.hash(),
            "model_lock_hash": manifest.model_lock_hash,
            "status": "validated",
            "written": False,
        }

    values = environment if environment is not None else os.environ
    api_key = values.get(release.api_key_env) if release.api_key_env else None
    if release.api_key_env and not api_key:
        raise ValueError(
            f"structured replay requires environment variable {release.api_key_env}"
        )
    local_transport = transport or UrlLibOmlxTransport(
        release.omlx_base_url,
        api_key=api_key,
    )
    run = StructuredReplayRunner(release.profile, local_transport).run(
        papers,
        manifest=manifest,
        manifest_path=manifest_output,
        records_path=records_output,
    )
    result = run.result
    return {
        "case_count": len(run.records),
        "command": "stage2-replay",
        "corpus_hash": run.manifest.corpus_hash,
        "deterministic_repairs": result.deterministic_repairs,
        "failures": list(result.gate.failures),
        "first_valid_rate": result.first_valid_rate,
        "manifest_hash": result.manifest_hash,
        "manifest_output": str(manifest_output),
        "manifest_sha256": sha256(manifest_output.read_bytes()).hexdigest(),
        "model_retries": result.model_retries,
        "records_output": str(records_output),
        "records_sha256": sha256(records_output.read_bytes()).hexdigest(),
        "retry_error_counts": {
            error.value: count for error, count in result.retry_error_counts.items()
        },
        "schema_errors": result.schema_errors,
        "service_errors": result.service_errors,
        "status": "passed" if result.gate.passed else "failed",
        "timeouts": result.timeouts,
        "written": True,
    }


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
    return _measurement_result(record, output_path)


def _measurement_result(record: Any, output_path: Path) -> dict[str, Any]:
    document = record.document()
    alarm_codes = list(document["alarm_codes"])
    return {
        "alarm_codes": alarm_codes,
        "artifact_hash": record.hash(),
        "artifact_path": str(output_path),
        "case_count": document["case_count"],
        "command": "benchmark-stage2.measure",
        "kind": document["kind"],
        "manifest_hash": document["manifest_hash"],
        "record_version": document["record_version"],
        "adjudicator_capacity": document["adjudicator_capacity"],
        "adjudicator_count": document["adjudicator_count"],
        "adjudicator_share": document["adjudicator_share"],
        "qwen_capacity_level": document["qwen_capacity_level"],
        "qwen_count": document["qwen_count"],
        "qwen_share": document["qwen_share"],
        "request_failure_rate": document["request_failure_rate"],
        "peak_memory_gb": document["peak_memory_gb"],
        "memory_pressure_critical": document["memory_pressure_critical"],
        "unbounded_memory_growth": document["unbounded_memory_growth"],
        "rss_scope": document["rss_scope"],
        "run_id": document["run_id"],
        "scenario": document["scenario"],
        "service_request_count": document["service_request_count"],
        "service_failed_request_count": document["service_failed_request_count"],
        "service_request_failure_rate": document["service_request_failure_rate"],
        "status": (
            "incomplete"
            if ERROR_RATE_ALARM in alarm_codes or MEMORY_WATERMARK_ALARM in alarm_codes
            else "complete"
        ),
    }


def _paper_ids(
    database: Database, requested: Sequence[str] | None
) -> tuple[str, ...]:
    available = {
        str(row["paper_id"])
        for row in database.connection.execute(
            "SELECT paper_id FROM papers ORDER BY paper_id"
        ).fetchall()
    }
    selected = (
        tuple(sorted(available))
        if requested is None
        else tuple(sorted(set(requested)))
    )
    missing = sorted(set(selected) - available)
    if missing:
        raise ValueError(f"filter papers do not exist: {missing}")
    return selected


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _stage2_input_profile(
    runtime: Mapping[str, Any],
    reranker_lock_path: Path,
    adjudicator_lock_path: Path,
) -> tuple[Stage2Profile, Mapping[CalibrationPath, str]]:
    reranker_bytes = reranker_lock_path.read_bytes()
    adjudicator_bytes = adjudicator_lock_path.read_bytes()
    reranker_document = json.loads(reranker_bytes)
    adjudicator_document = json.loads(adjudicator_bytes)
    if not isinstance(reranker_document, dict) or not isinstance(
        adjudicator_document, dict
    ):
        raise ValueError("Stage 2 model locks must be JSON objects")
    reranker_hash = sha256(reranker_bytes).hexdigest()
    adjudicator_hash = sha256(adjudicator_bytes).hexdigest()
    profile = stage2_base_profile(
        runtime,
        ModelLock(**reranker_document),
        ModelLock(**adjudicator_document),
        reranker_lock_hash=reranker_hash,
        adjudicator_lock_hash=adjudicator_hash,
    )
    return profile, {
        CalibrationPath.RERANKER: reranker_hash,
        CalibrationPath.QWEN: adjudicator_hash,
    }


def _topic_queries(document: Mapping[str, Any]) -> dict[tuple[str, str], str]:
    topics = document.get("topics")
    if not isinstance(topics, list) or not topics:
        raise ValueError("topic query input requires a non-empty topics array")
    result: dict[tuple[str, str], str] = {}
    for topic in topics:
        if not isinstance(topic, dict):
            raise ValueError("topic query entries must be objects")
        topic_id = topic.get("id")
        queries = topic.get("queries")
        if not isinstance(topic_id, str) or not topic_id or not isinstance(
            queries, list
        ):
            raise ValueError("topic query entry requires id and queries")
        for query in queries:
            if not isinstance(query, dict):
                raise ValueError("topic query variants must be objects")
            language = query.get("language")
            text = query.get("query")
            if (
                not isinstance(language, str)
                or not language
                or not isinstance(text, str)
                or not text.strip()
            ):
                raise ValueError("topic query variants require language and query")
            key = topic_id, language
            if key in result:
                raise ValueError("topic query input contains a duplicate topic-language")
            result[key] = text
    return result


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
    return benchmark_papers_from_document(json.loads(path.read_text(encoding="utf-8")))


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
