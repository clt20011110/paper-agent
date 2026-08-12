from __future__ import annotations

from base64 import b64encode
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from paper_agent.canonical import content_hash
from paper_agent.schema import SchemaValidationError, validate
from paper_agent.stage2_benchmark import BenchmarkExecutionRecord, benchmark_workload_hash
from paper_agent.stage2_benchmark_inputs import (
    benchmark_corpus_hash,
    benchmark_papers_from_document,
)
from paper_agent.stage2_evaluation import (
    GoldManifest,
    GoldPair,
    GoldSplit,
    ParityManifest,
    ParityScore,
    PerformanceCase,
    PerformanceRoutingManifest,
    RationaleAuditCase,
    RationaleAuditManifest,
    RationaleAuditRecord,
    RationaleStratum,
    ReplayError,
    SoakManifest,
    Stage2Decision,
    StructuredReplayManifest,
    StructuredReplayRecord,
    pair_universe_hash,
    write_gold_manifest,
)
from paper_agent.stage2_public_gates import verify_public_stage2_gates
from paper_agent.stage2_pipeline import Stage2Profile
from paper_agent.stage2_prompt_contract import (
    adjudication_messages,
    estimate_omlx_chat_input_token_proxy,
)
from paper_agent.stage2_release_evidence import (
    Stage2EvidenceError,
    load_stage2_release_evidence_index,
)


HASH = "a" * 64


def _write(path: Path, value: object) -> dict[str, str]:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return {"path": path.name, "sha256": sha256(path.read_bytes()).hexdigest()}


def _gold_manifest() -> GoldManifest:
    pairs = []
    for split, size in (
        (GoldSplit.DEV, 300),
        (GoldSplit.HIDDEN_HARD, 150),
        (GoldSplit.HIDDEN_REAL, 150),
    ):
        for index in range(size):
            pairs.append(GoldPair(
                paper_id=f"paper-{split.value}-{index}",
                topic=f"topic-{index % 6}",
                language="zh" if index % 2 else "en",
                source="frozen-crawler-snapshot",
                sampling_probability=(
                    0.2 if split is GoldSplit.HIDDEN_REAL else None
                ),
                paper_family=f"family-{split.value}-{index}",
                corpus_hash="corpus-v1",
                split=split,
                abstract_incomplete=(
                    split is not GoldSplit.HIDDEN_REAL and index < size // 10
                ),
                sampled_from_natural_distribution=split is GoldSplit.HIDDEN_REAL,
                cross_language_match=index % 20 == 0,
            ))
    return GoldManifest(1, "corpus-v1", tuple(pairs), ("en", "zh"))


def _attestation(
    *,
    candidate_id: str,
    evaluation_manifest_hash: str,
    stage2_config_hash: str,
    hashes: dict[str, str],
    gold: GoldManifest | None = None,
) -> dict:
    universe_hashes = {
        split.value: pair_universe_hash(
            [pair.pair_id for pair in gold.pairs if pair.split is split]
        )
        for split in (GoldSplit.HIDDEN_HARD, GoldSplit.HIDDEN_REAL)
    } if gold is not None else {"hidden_hard": HASH, "hidden_real": HASH}
    payload = {
        "schema_version": "1",
        "attestation_type": "stage2-hidden-promotion",
        "evaluator_key_id": "hidden-evaluator-2026-01",
        "evaluator_id": "evaluation-team-1",
        "trust_manifest_hash": HASH,
        "issued_at": "2026-08-11T00:00:00Z",
        "candidate_id": candidate_id,
        "evaluation_manifest_hash": evaluation_manifest_hash,
        "evaluation_run_id": "promotion-1",
        "stage2_config_hash": stage2_config_hash,
        "model_lock_hashes": hashes,
        "calibrator_hashes": hashes,
        "threshold_hashes": hashes,
        "hidden_pair_universe_hashes": universe_hashes,
        "hidden_split_pair_counts": {
            "hidden_hard": 150,
            "hidden_real": 150,
        },
        "prediction_submission_hash": HASH,
        "promotion_marker_hash": HASH,
        "winner_candidate_id": candidate_id,
        "public_gate_artifact_hashes": {name: HASH for name in ("structured_replay", "rationale", "parity", "benchmark", "soak")},
        "throughput_runs": [1, 1, 1],
        "consumed_hidden_splits": ["hidden_hard", "hidden_real"],
        "gate_policy_hash": HASH,
        "result_summary": {
            "passed": True,
            "failures": [],
            "gate_versions": {"promotion": "1", "determinism": "1"},
        },
    }
    return {
        "schema_version": "1",
        "payload": payload,
        "payload_sha256": content_hash(payload),
        "signature_b64": b64encode(bytes(64)).decode(),
    }


def _index(tmp_path: Path) -> tuple[Path, dict]:
    gold = _gold_manifest()
    gold_path = tmp_path / "gold.json"
    write_gold_manifest(gold_path, gold)
    gold_ref = {"path": gold_path.name, "sha256": sha256(gold_path.read_bytes()).hexdigest()}
    hashes = {"reranker": HASH, "qwen": "b" * 64}
    stage2_config_hash = "d" * 64
    attestation_ref = _write(
        tmp_path / "attestation.json",
        _attestation(
            candidate_id="candidate-1",
            evaluation_manifest_hash=gold.hash(),
            stage2_config_hash=stage2_config_hash,
            hashes=hashes,
            gold=gold,
        ),
    )
    refs = {
        name: _write(tmp_path / f"{name}.json", {"kind": name})
        for name in (
            "structured-manifest",
            "structured-records",
            "rationale-manifest",
            "rationale-records",
            "parity-manifest",
            "parity-scores",
            "benchmark-manifest",
            "benchmark-papers",
            "soak-manifest",
            "soak-papers",
            "soak-record",
        )
    }
    benchmark_records = [
        _write(tmp_path / f"benchmark-record-{index}.json", {"run": index})
        for index in range(6)
    ]
    document = {
        "schema_version": "1",
        "evidence_type": "stage2_release_evidence",
        "candidate_id": "candidate-1",
        "evaluation_manifest_hash": gold.hash(),
        "stage2_config_hash": stage2_config_hash,
        "model_lock_hashes": hashes,
        "calibrator_hashes": hashes,
        "threshold_hashes": hashes,
        "gold_manifest": gold_ref,
        "hidden_attestation": attestation_ref,
        "public_gates": {
            "structured_replay": {
                "manifest": refs["structured-manifest"],
                "records": refs["structured-records"],
            },
            "rationale": {
                "manifest": refs["rationale-manifest"],
                "records": refs["rationale-records"],
            },
            "parity": {
                "manifest": refs["parity-manifest"],
                "scores": refs["parity-scores"],
            },
            "benchmark": {
                "manifest": refs["benchmark-manifest"],
                "papers": refs["benchmark-papers"],
                "records": benchmark_records,
            },
            "soak": {
                "manifest": refs["soak-manifest"],
                "papers": refs["soak-papers"],
                "record": refs["soak-record"],
            },
        },
    }
    path = tmp_path / "stage2-release-evidence.json"
    path.write_text(json.dumps(document, sort_keys=True) + "\n", encoding="utf-8")
    return path, document


def _benchmark_paper_document(pair_ids: list[str], missing: set[str]) -> dict:
    return {
        "schema_version": "1",
        "kind": "stage2_benchmark_papers",
        "papers": [
            {
                "paper_id": pair_id,
                "title": f"Title {pair_id}",
                "abstract": None if pair_id in missing else f"Abstract {pair_id}",
                "keywords": ["topic"],
                "document_type": "article",
                "possibly_truncated": False,
                "multi_condition_conflict": False,
                "language_anomaly": False,
            }
            for pair_id in pair_ids
        ],
    }


def _execution_payload(
    manifest: PerformanceRoutingManifest | SoakManifest,
    papers_document: dict,
    profile: Stage2Profile,
    *,
    kind: str,
    scenario: str | None,
    run_id: str,
    qwen_pair_ids: list[str],
    duration_seconds: float,
) -> dict:
    papers = benchmark_papers_from_document(papers_document)
    pair_ids = [item.pair_id for item in manifest.cases]
    rerank_trace = [
        {
            "path": "/v1/rerank",
            "duration_seconds": 0.01,
            "document_count": min(32, len(pair_ids) - offset),
            "failed": False,
        }
        for offset in range(0, len(pair_ids), 32)
    ]
    qwen_trace = [
        {
            "path": "/v1/chat/completions",
            "duration_seconds": 0.01,
            "document_count": 1,
            "failed": False,
        }
        for _ in qwen_pair_ids
    ]
    service_trace = rerank_trace + qwen_trace
    model_call_trace = [
        {
            "backend": "reranker",
            "pair_ids": pair_ids,
            "duration_seconds": 0.5,
            "failed": False,
        }
    ] + [
        {
            "backend": "qwen",
            "pair_ids": [pair_id],
            "duration_seconds": 0.01,
            "failed": False,
        }
        for pair_id in qwen_pair_ids
    ]
    rss_samples = [8 * 1024**3] * 4
    qwen_count = len(qwen_pair_ids)
    qwen_share = qwen_count / len(pair_ids)
    capacity = "severe" if qwen_share > 0.30 else "warning" if qwen_share > 0.15 else "normal"
    alarms = ["stage2.adjudicator_share_exceeded"] if qwen_share > 0.15 else []
    input_tokens = sum(item.input_tokens for item in manifest.cases)
    components = ["rules", "reranker", "qwen", "schema_validation", "sqlite_commit"]
    latency = {
        "reranker": {"sample_count": len(rerank_trace), "p50_seconds": 0.01, "p95_seconds": 0.01},
        "qwen": {"sample_count": len(qwen_trace), "p50_seconds": 0.01, "p95_seconds": 0.01},
    }
    model_hashes = list(manifest.model_lock_hashes)
    threshold_hashes = list(manifest.threshold_artifact_hashes)
    environment = {
        "machine_model": "Apple Silicon M4 Max",
        "memory_gb": 36,
        "macos_version": "15.6",
        "omlx_version": "0.5.7",
        "mlx_version": "0.29.0",
        "power_mode": "automatic",
        "background_load": "idle",
        "batch_config": {
            "document_batch_size": 32,
            "reranker_max_in_flight": 2,
            "adjudicator_concurrency": 4,
        },
        "resident_model_instances": {item: 1 for item in model_hashes},
    }
    return {
        "record_version": 2,
        "measurement_evidence_version": "1",
        "kind": kind,
        "scenario": scenario,
        "run_id": run_id,
        "manifest_hash": manifest.hash(),
        "corpus_hash": manifest.corpus_hash,
        "input_hash": benchmark_workload_hash(manifest.cases, papers),
        "release_hash": "1" * 64,
        "stage2_config_hash": manifest.stage2_config_hash,
        "observed_stage2_config_hash": manifest.stage2_config_hash,
        "full_profile_hash": profile.full_profile_hash,
        "model_lock_hashes": model_hashes,
        "threshold_artifact_hashes": threshold_hashes,
        "observed_threshold_artifact_hashes": threshold_hashes,
        "model_releases": {
            "reranker": {
                "model_id": profile.reranker_model_id,
                "revision": profile.reranker_revision,
            },
            "qwen": {
                "model_id": profile.adjudicator_model_id,
                "revision": profile.adjudicator_revision,
            },
        },
        "prompt_hash": profile.prompt_hash,
        "schema_hash": profile.schema_hash,
        "output_token_limit": manifest.output_token_limit,
        "observed_output_token_limit": manifest.output_token_limit,
        "fixture_scale": False,
        "case_count": len(pair_ids),
        "input_token_count": input_tokens,
        "duration_seconds": duration_seconds,
        "p50_seconds": 0.01,
        "p95_seconds": 0.01,
        "latency_sample_count": len(service_trace),
        "latency_sample_unit": "omlx_service_request",
        "latency_by_path": latency,
        "papers_per_second": len(pair_ids) / duration_seconds,
        "input_tokens_per_second": input_tokens / duration_seconds,
        "pair_tokens_per_second": input_tokens / duration_seconds,
        "peak_memory_gb": 8.0,
        "rss_start_bytes": rss_samples[0],
        "rss_end_bytes": rss_samples[-1],
        "peak_rss_bytes": max(rss_samples),
        "rss_sample_count": len(rss_samples),
        "rss_samples_bytes": rss_samples,
        "memory_pressure_samples": [False] * len(rss_samples),
        "rss_sample_interval_seconds": 0.25,
        "rss_scope": "macos_ps_current_rss:runner_pid=100;omlx_pids=200",
        "request_count": len(pair_ids),
        "request_count_unit": "manifest_case",
        "request_failure_rate": 0.0,
        "pair_attempt_count": len(pair_ids) + qwen_count,
        "model_call_count": len(model_call_trace),
        "model_call_trace": model_call_trace,
        "service_request_count": len(service_trace),
        "service_request_trace": service_trace,
        "service_pair_attempt_count": sum(item["document_count"] for item in service_trace),
        "service_failed_request_count": 0,
        "service_request_failure_rate": 0.0,
        "reranker_batch_call_count": 1,
        "reranker_fallback_count": 0,
        "reranker_fallback_measurement_available": True,
        "adjudicator_call_count": qwen_count,
        "backend_failed_call_count": 0,
        "backend_call_failure_rate": 0.0,
        "failed_request_count": 0,
        "completed_pair_ids": sorted(pair_ids),
        "needs_review_pair_ids": [],
        "failed_request_pair_ids": [],
        "qwen_pair_ids": sorted(qwen_pair_ids),
        "qwen_count": qwen_count,
        "qwen_share": qwen_share,
        "qwen_share_alarms": alarms,
        "qwen_capacity_level": capacity,
        "adjudicator_count": qwen_count,
        "adjudicator_share": qwen_share,
        "adjudicator_capacity": capacity,
        "alarm_codes": alarms,
        "frozen_qwen_routing_matches": True if kind == "performance" else None,
        "routing_mode": "performance_only_manifest" if kind == "performance" else "quality_thresholds",
        "batch_concurrency": {
            "document_batch_size": 32,
            "reranker_max_in_flight": 2,
            "adjudicator_concurrency": 4,
        },
        "environment": environment,
        "expected_components": components,
        "executed_components": components,
        "sqlite_commit_count": len(pair_ids),
        "sqlite_commit_unit": "persisted_filter_decision",
        "result_count": len(pair_ids),
        "missing_result_count": 0,
        "duplicate_result_count": 0,
        "warmed": True,
        "resume_verified": True,
        "resume_model_call_count": 0,
        "resume_service_request_trace": [],
        "resumed_pair_count": len(pair_ids),
        "resumed_pair_ids": sorted(pair_ids),
        "oom": False,
        "process_crash": False,
        "memory_pressure_critical": False,
        "memory_pressure_sampled": True,
        "memory_growth_detector": {
            "version": "post_warmup_monotonic_25_percent_v1",
            "monotonic_non_decreasing": True,
            "growth_bytes": 0,
            "growth_ratio": 1.0,
            "minimum_sample_count": 4,
        },
        "unbounded_memory_growth": False,
    }


def _write_execution(path: Path, payload: dict) -> dict[str, str]:
    record = BenchmarkExecutionRecord(payload)
    path.write_bytes(record.canonical_bytes())
    return {"path": path.name, "sha256": sha256(path.read_bytes()).hexdigest()}


def _install_public_gate_evidence(
    tmp_path: Path,
    path: Path,
    index_document: dict,
    *,
    profile: Stage2Profile,
) -> None:
    model_hashes = index_document["model_lock_hashes"]
    threshold_hashes = index_document["threshold_hashes"]
    config_hash = index_document["stage2_config_hash"]
    gold = _gold_manifest()

    replay_ids = [f"replay-{index}" for index in range(1_000)]
    replay_manifest = StructuredReplayManifest(
        1,
        tuple(replay_ids),
        "5" * 64,
        config_hash,
        model_hashes["qwen"],
        profile.prompt_hash,
        profile.schema_hash,
    )
    replay_records = [
        StructuredReplayRecord(
            pair_id,
            replay_manifest.hash(),
            ReplayError.NONE,
            pair_id,
            False,
            False,
            0,
            0,
            None,
            True,
            pair_id,
            False,
            False,
            Stage2Decision.RELEVANT,
        ).document()
        for pair_id in replay_ids
    ]
    index_document["public_gates"]["structured_replay"] = {
        "manifest": _write(tmp_path / "structured-manifest.json", replay_manifest.document()),
        "records": _write(tmp_path / "structured-records.json", {
            "schema_version": "1",
            "kind": "stage2_structured_replay_records",
            "records": replay_records,
        }),
    }

    rationale_cases = tuple(
        RationaleAuditCase(
            f"rationale-{stratum.value}-{language}-{index}",
            stratum,
            language,
            "8" * 64,
        )
        for stratum in RationaleStratum
        for language in ("en", "zh")
        for index in range(25)
    )
    rationale_manifest = RationaleAuditManifest(
        1,
        rationale_cases,
        "9" * 64,
        model_hashes["qwen"],
        "a" * 64,
        "b" * 64,
    )
    rationale_records = [
        RationaleAuditRecord(item.pair_id, rationale_manifest.hash(), True, False).document()
        for item in rationale_cases
    ]
    index_document["public_gates"]["rationale"] = {
        "manifest": _write(tmp_path / "rationale-manifest.json", rationale_manifest.document()),
        "records": _write(tmp_path / "rationale-records.json", {
            "schema_version": "1",
            "kind": "stage2_rationale_audit_records",
            "records": rationale_records,
        }),
    }

    parity_ids = [f"parity-{index}" for index in range(10_000)]
    parity_manifest = ParityManifest(
        1,
        tuple(parity_ids),
        "c" * 64,
        "d" * 64,
        "e" * 64,
        "f" * 64,
        model_hashes["reranker"],
        "1" * 64,
        threshold_hashes["reranker"],
        gold.dev_hash(),
        "2" * 64,
        0.2,
        0.8,
        0.2,
        0.8,
        frozenset(parity_ids[:100]),
        frozenset(parity_ids[-100:]),
        "closest to oracle low threshold",
        "closest to oracle high threshold",
    )
    parity_scores = [
        ParityScore(pair_id, parity_manifest.hash(), index / 10_000, index / 10_000).document()
        for index, pair_id in enumerate(parity_ids)
    ]
    index_document["public_gates"]["parity"] = {
        "manifest": _write(tmp_path / "parity-manifest.json", parity_manifest.document()),
        "scores": _write(tmp_path / "parity-scores.json", {
            "schema_version": "1",
            "kind": "stage2_parity_scores",
            "scores": parity_scores,
        }),
    }

    performance_ids = [f"performance-{index}" for index in range(1_000)]
    performance_missing = set(performance_ids[:100])
    performance_papers_document = _benchmark_paper_document(performance_ids, performance_missing)
    performance_papers = benchmark_papers_from_document(performance_papers_document)
    performance_manifest = PerformanceRoutingManifest(
        1,
        benchmark_corpus_hash(performance_papers),
        config_hash,
        (model_hashes["reranker"], model_hashes["qwen"]),
        (threshold_hashes["reranker"], threshold_hashes["qwen"]),
        256,
        tuple(
            PerformanceCase(
                paper.paper_id,
                _benchmark_input_tokens(profile, paper),
                paper.paper_id in performance_missing,
            )
            for paper in performance_papers
        ),
        frozenset(performance_ids[:150]),
        frozenset(performance_ids[:300]),
    )
    benchmark_records = []
    for scenario, qwen_ids, duration in (
        ("normal", performance_ids[:150], 100.0),
        ("stress", performance_ids[:300], 200.0),
    ):
        for run in range(3):
            payload = _execution_payload(
                performance_manifest,
                performance_papers_document,
                profile,
                kind="performance",
                scenario=scenario,
                run_id=f"{scenario}-{run}",
                qwen_pair_ids=qwen_ids,
                duration_seconds=duration + run,
            )
            benchmark_records.append(_write_execution(
                tmp_path / f"benchmark-{scenario}-{run}.json",
                payload,
            ))
    index_document["public_gates"]["benchmark"] = {
        "manifest": _write(tmp_path / "benchmark-manifest.json", performance_manifest.document()),
        "papers": _write(tmp_path / "benchmark-papers.json", performance_papers_document),
        "records": benchmark_records,
    }

    soak_ids = [f"soak-{index}" for index in range(10_000)]
    soak_papers_document = _benchmark_paper_document(soak_ids, set())
    soak_papers = benchmark_papers_from_document(soak_papers_document)
    soak_manifest = SoakManifest(
        1,
        benchmark_corpus_hash(soak_papers),
        config_hash,
        (model_hashes["reranker"], model_hashes["qwen"]),
        (threshold_hashes["reranker"], threshold_hashes["qwen"]),
        256,
        tuple(
            PerformanceCase(
                paper.paper_id,
                _benchmark_input_tokens(profile, paper),
                False,
            )
            for paper in soak_papers
        ),
    )
    soak_payload = _execution_payload(
        soak_manifest,
        soak_papers_document,
        profile,
        kind="soak",
        scenario=None,
        run_id="soak-0",
        qwen_pair_ids=soak_ids[:1_000],
        duration_seconds=500.0,
    )
    index_document["public_gates"]["soak"] = {
        "manifest": _write(tmp_path / "soak-manifest.json", soak_manifest.document()),
        "papers": _write(tmp_path / "soak-papers.json", soak_papers_document),
        "record": _write_execution(tmp_path / "soak-record.json", soak_payload),
    }
    path.write_text(json.dumps(index_document, sort_keys=True) + "\n", encoding="utf-8")


def _benchmark_input_tokens(profile: Stage2Profile, paper: Stage2Paper) -> int:
    return estimate_omlx_chat_input_token_proxy(adjudication_messages(
        query_version=profile.query_version,
        query=profile.query,
        paper=paper,
    ))


def _public_profile() -> SimpleNamespace:
    return SimpleNamespace(
        query="frozen query",
        query_version="query-v1",
        base_runtime_config_hash="d" * 64,
        full_profile_hash="2" * 64,
        prompt_hash="3" * 64,
        schema_hash="4" * 64,
        reranker_lock_hash=HASH,
        adjudicator_lock_hash="b" * 64,
        reranker_calibration=SimpleNamespace(
            calibrator=SimpleNamespace(hash=lambda: HASH),
            threshold=SimpleNamespace(hash=lambda: HASH),
        ),
        adjudicator_calibration=SimpleNamespace(
            calibrator=SimpleNamespace(hash=lambda: "b" * 64),
            threshold=SimpleNamespace(hash=lambda: "b" * 64),
        ),
        reranker_model_id="reranker",
        reranker_revision="revision-1",
        adjudicator_model_id="qwen",
        adjudicator_revision="revision-1",
    )


def test_release_evidence_index_verifies_every_bound_file(tmp_path: Path) -> None:
    path, _ = _index(tmp_path)

    index = load_stage2_release_evidence_index(path)

    assert index.candidate_id == "candidate-1"


def test_public_promotion_evidence_does_not_require_a_circular_hidden_attestation(
    tmp_path: Path,
) -> None:
    path, document = _index(tmp_path)
    document["evidence_type"] = "stage2_public_promotion_evidence"
    document.pop("hidden_attestation")
    path.write_text(json.dumps(document, sort_keys=True) + "\n", encoding="utf-8")

    index = load_stage2_release_evidence_index(path)

    assert index.hidden_attestation is None
    assert tuple(index.public_gates) == (
        "structured_replay",
        "rationale",
        "parity",
        "benchmark",
        "soak",
    )
    assert len(index.public_gates["benchmark"].records) == 6


def test_release_evidence_index_rejects_claimed_gate_outcomes(tmp_path: Path) -> None:
    path, document = _index(tmp_path)
    document["passed"] = True
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(SchemaValidationError, match="Additional properties"):
        load_stage2_release_evidence_index(path)


def test_release_evidence_index_rejects_file_hash_drift(tmp_path: Path) -> None:
    path, _ = _index(tmp_path)
    (tmp_path / "parity-scores.json").write_text("[]\n", encoding="utf-8")

    with pytest.raises(Stage2EvidenceError, match="drifted: parity-scores.json"):
        load_stage2_release_evidence_index(path)


def test_release_evidence_index_binds_public_gold_manifest_identity(tmp_path: Path) -> None:
    path, document = _index(tmp_path)
    document["evaluation_manifest_hash"] = "f" * 64
    attestation = json.loads((tmp_path / "attestation.json").read_text(encoding="utf-8"))
    attestation["payload"]["evaluation_manifest_hash"] = "f" * 64
    document["hidden_attestation"] = _write(tmp_path / "attestation.json", attestation)
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(Stage2EvidenceError, match="gold manifest does not match"):
        load_stage2_release_evidence_index(path)


def test_release_evidence_index_binds_hidden_attestation_identity(tmp_path: Path) -> None:
    path, document = _index(tmp_path)
    attestation = json.loads((tmp_path / "attestation.json").read_text(encoding="utf-8"))
    attestation["payload"]["candidate_id"] = "different-candidate"
    attestation["payload_sha256"] = content_hash(attestation["payload"])
    document["hidden_attestation"] = _write(tmp_path / "attestation.json", attestation)
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(Stage2EvidenceError, match="candidate_id does not match"):
        load_stage2_release_evidence_index(path)


def test_release_evidence_index_recomputes_attestation_payload_hash(tmp_path: Path) -> None:
    path, document = _index(tmp_path)
    attestation = json.loads((tmp_path / "attestation.json").read_text(encoding="utf-8"))
    attestation["payload_sha256"] = "f" * 64
    document["hidden_attestation"] = _write(tmp_path / "attestation.json", attestation)
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(Stage2EvidenceError, match="payload hash is invalid"):
        load_stage2_release_evidence_index(path)


def test_release_evidence_index_binds_hidden_pair_universes(tmp_path: Path) -> None:
    path, document = _index(tmp_path)
    attestation = json.loads((tmp_path / "attestation.json").read_text(encoding="utf-8"))
    attestation["payload"]["hidden_pair_universe_hashes"]["hidden_real"] = "f" * 64
    attestation["payload_sha256"] = content_hash(attestation["payload"])
    document["hidden_attestation"] = _write(tmp_path / "attestation.json", attestation)
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(Stage2EvidenceError, match="pair universes do not match"):
        load_stage2_release_evidence_index(path)


def test_release_evidence_index_rejects_symlink_escape(tmp_path: Path) -> None:
    path, document = _index(tmp_path)
    outside = tmp_path.parent / "outside-stage2-evidence.json"
    outside.write_text("{}\n", encoding="utf-8")
    link = tmp_path / "linked-evidence.json"
    link.symlink_to(outside)
    document["gold_manifest"] = {
        "path": link.name,
        "sha256": sha256(outside.read_bytes()).hexdigest(),
    }
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(Stage2EvidenceError, match="stay inside the bundle"):
        load_stage2_release_evidence_index(path)


def test_hidden_attestation_schema_forbids_private_evaluator_content() -> None:
    document = _attestation(
        candidate_id="candidate-1",
        evaluation_manifest_hash=HASH,
        stage2_config_hash=HASH,
        hashes={"reranker": HASH, "qwen": HASH},
    )
    validate(document, "stage2-hidden-evaluator-attestation.schema.json")

    document["payload"]["labels"] = {"hidden-pair": 3}
    with pytest.raises(SchemaValidationError, match="Additional properties"):
        validate(document, "stage2-hidden-evaluator-attestation.schema.json")


def test_public_stage2_gates_are_recomputed_from_raw_evidence(tmp_path: Path) -> None:
    path, document = _index(tmp_path)
    profile = _public_profile()
    _install_public_gate_evidence(tmp_path, path, document, profile=profile)

    result = verify_public_stage2_gates(
        load_stage2_release_evidence_index(path), profile=profile,
    )

    assert result.passed
    assert result.throughput_runs == (10.0, 1000 / 101, 1000 / 102)
    assert all(
        gate.document()["verification"] == "recomputed_from_raw_evidence"
        for gate in result.gates.values()
    )
    assert result.gates["structured_replay"].metrics["record_count"] == 1_000
    assert result.gates["rationale"].metrics["record_count"] == 100
    assert result.gates["parity"].metrics["score_count"] == 10_000
    assert result.gates["benchmark"].metrics["record_count"] == 6
    assert result.gates["soak"].metrics["request_count"] == 10_000


def test_public_verifier_recomputes_benchmark_prompt_token_bounds(tmp_path: Path) -> None:
    path, document = _index(tmp_path)
    profile = _public_profile()
    _install_public_gate_evidence(tmp_path, path, document, profile=profile)
    manifest_path = tmp_path / "benchmark-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cases"][0]["input_tokens"] += 1
    document["public_gates"]["benchmark"]["manifest"] = _write(
        manifest_path, manifest,
    )
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(Stage2EvidenceError, match="input_tokens do not match"):
        verify_public_stage2_gates(
            load_stage2_release_evidence_index(path), profile=profile,
        )


@pytest.mark.parametrize("field", ("prompt_hash", "schema_hash", "full_profile_hash"))
def test_public_verifier_binds_execution_profile_hashes(
    tmp_path: Path, field: str,
) -> None:
    path, document = _index(tmp_path)
    profile = _public_profile()
    _install_public_gate_evidence(tmp_path, path, document, profile=profile)
    record_path = tmp_path / document["public_gates"]["benchmark"]["records"][0]["path"]
    record = json.loads(record_path.read_bytes())
    record[field] = "f" * 64
    document["public_gates"]["benchmark"]["records"][0] = _write_execution(
        record_path, record,
    )
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(Stage2EvidenceError, match=f"execution {field}"):
        verify_public_stage2_gates(
            load_stage2_release_evidence_index(path), profile=profile,
        )


def test_public_verifier_binds_execution_model_identity(tmp_path: Path) -> None:
    path, document = _index(tmp_path)
    profile = _public_profile()
    _install_public_gate_evidence(tmp_path, path, document, profile=profile)
    record_path = tmp_path / document["public_gates"]["benchmark"]["records"][0]["path"]
    record = json.loads(record_path.read_bytes())
    record["model_releases"]["qwen"]["revision"] = "other-revision"
    document["public_gates"]["benchmark"]["records"][0] = _write_execution(
        record_path, record,
    )
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(Stage2EvidenceError, match="execution model_releases"):
        verify_public_stage2_gates(
            load_stage2_release_evidence_index(path), profile=profile,
        )


def test_public_stage2_gates_ignore_hash_valid_but_failing_rationale_claims(
    tmp_path: Path,
) -> None:
    path, document = _index(tmp_path)
    profile = _public_profile()
    _install_public_gate_evidence(tmp_path, path, document, profile=profile)
    records_path = tmp_path / "rationale-records.json"
    records = json.loads(records_path.read_text(encoding="utf-8"))
    for item in records["records"][:6]:
        item["evidence_supported"] = False
    document["public_gates"]["rationale"]["records"] = _write(records_path, records)
    path.write_text(json.dumps(document), encoding="utf-8")

    result = verify_public_stage2_gates(
        load_stage2_release_evidence_index(path), profile=profile,
    )

    assert not result.passed
    assert result.gates["rationale"].gate.failures == (
        "rationale evidence support < 95%",
    )


def test_public_stage2_gates_reject_rewritten_benchmark_memory_summary(
    tmp_path: Path,
) -> None:
    path, document = _index(tmp_path)
    profile = _public_profile()
    _install_public_gate_evidence(tmp_path, path, document, profile=profile)
    record_ref = document["public_gates"]["benchmark"]["records"][0]
    record_path = tmp_path / record_ref["path"]
    record = json.loads(record_path.read_bytes())
    record["peak_memory_gb"] = 7.0
    document["public_gates"]["benchmark"]["records"][0] = _write_execution(
        record_path,
        record,
    )
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(Stage2EvidenceError, match="memory summaries do not match"):
        verify_public_stage2_gates(
            load_stage2_release_evidence_index(path), profile=profile,
        )


def test_public_stage2_gates_require_chat_trace_for_every_qwen_route(
    tmp_path: Path,
) -> None:
    path, document = _index(tmp_path)
    profile = _public_profile()
    _install_public_gate_evidence(tmp_path, path, document, profile=profile)
    record_ref = document["public_gates"]["benchmark"]["records"][0]
    record_path = tmp_path / record_ref["path"]
    record = json.loads(record_path.read_bytes())
    for item in record["service_request_trace"]:
        item["path"] = "/v1/rerank"
    record["latency_by_path"] = {
        "reranker": {
            "sample_count": len(record["service_request_trace"]),
            "p50_seconds": 0.01,
            "p95_seconds": 0.01,
        },
        "qwen": {"sample_count": 0, "p50_seconds": 0.0, "p95_seconds": 0.0},
    }
    record["reranker_fallback_count"] = 150
    document["public_gates"]["benchmark"]["records"][0] = _write_execution(
        record_path,
        record,
    )
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(Stage2EvidenceError, match="model-call routing"):
        verify_public_stage2_gates(
            load_stage2_release_evidence_index(path), profile=profile,
        )


def test_public_stage2_gates_require_real_boolean_primitives(tmp_path: Path) -> None:
    path, document = _index(tmp_path)
    profile = _public_profile()
    _install_public_gate_evidence(tmp_path, path, document, profile=profile)
    record_ref = document["public_gates"]["benchmark"]["records"][0]
    record_path = tmp_path / record_ref["path"]
    record = json.loads(record_path.read_bytes())
    record["fixture_scale"] = 0
    document["public_gates"]["benchmark"]["records"][0] = _write_execution(
        record_path,
        record,
    )
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(Stage2EvidenceError, match="JSON booleans"):
        verify_public_stage2_gates(
            load_stage2_release_evidence_index(path), profile=profile,
        )
