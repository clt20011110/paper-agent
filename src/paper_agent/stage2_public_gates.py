"""Deterministically recompute public Stage 2 release gates from raw evidence."""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import ceil, isclose
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .canonical import canonical_json, content_hash
from .schema import validate
from .stage2_benchmark import (
    BENCHMARK_EXECUTION_FIELDS,
    BenchmarkExecutionRecord,
    benchmark_alarm_codes,
    benchmark_workload_hash,
)
from .stage2_benchmark_inputs import (
    benchmark_corpus_hash,
    benchmark_papers_from_document,
)
from .stage2_backends import ModelLock
from .stage2_evaluation import (
    GateResult,
    GoldManifest,
    ParityManifest,
    ParityScore,
    PathCalibrator,
    PerformanceCase,
    PerformanceRoutingManifest,
    PerformanceRunRecord,
    RationaleAuditCase,
    RationaleAuditManifest,
    RationaleAuditRecord,
    RationaleStratum,
    ReplayError,
    SoakManifest,
    SoakRunRecord,
    Stage2Decision,
    StructuredReplayManifest,
    StructuredReplayRecord,
    gold_manifest_from_document,
    parity_gate,
    performance_gate,
    performance_summary,
    rationale_audit_gate,
    soak_gate,
    structured_replay_gate,
    ThresholdArtifact,
)
from .stage2_parity_oracle_trust import ParityOracleTrust
from .stage2_release_evidence import (
    ArtifactRef,
    GateEvidenceRefs,
    ParityEvidenceRefs,
    Stage2EvidenceError,
    Stage2ReleaseEvidenceIndex,
)
from .stage2_pipeline import Stage2Paper, Stage2Profile
from .stage2_prompt_contract import (
    adjudication_messages,
    estimate_omlx_chat_input_token_proxy,
)


@dataclass(frozen=True, slots=True)
class VerifiedPublicGate:
    evidence_hash: str
    manifest_hash: str
    gate: GateResult
    metrics: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))

    def document(self) -> dict[str, Any]:
        return {
            "evidence_hash": self.evidence_hash,
            "failures": list(self.gate.failures),
            "manifest_hash": self.manifest_hash,
            "metrics": dict(self.metrics),
            "passed": self.gate.passed,
            "verification": "recomputed_from_raw_evidence",
        }


@dataclass(frozen=True, slots=True)
class VerifiedPublicStage2Evidence:
    gates: Mapping[str, VerifiedPublicGate]
    throughput_runs: tuple[float, float, float]

    def __post_init__(self) -> None:
        expected = {"structured_replay", "rationale", "parity", "benchmark", "soak"}
        if set(self.gates) != expected or len(self.throughput_runs) != 3:
            raise ValueError("verified public evidence requires all five gates and three throughput runs")
        object.__setattr__(self, "gates", MappingProxyType(dict(self.gates)))

    @property
    def passed(self) -> bool:
        return all(item.gate.passed for item in self.gates.values())

    def document(self) -> dict[str, Any]:
        return {
            "gates": {
                name: self.gates[name].document() for name in sorted(self.gates)
            },
            "passed": self.passed,
            "throughput_runs": list(self.throughput_runs),
        }


def verify_public_stage2_gates(
    index: Stage2ReleaseEvidenceIndex,
    *,
    profile: Stage2Profile,
    oracle_trust: ParityOracleTrust,
) -> VerifiedPublicStage2Evidence:
    """Recompute every non-hidden Stage 2 gate against one released profile."""

    _verify_profile_binding(index, profile)
    gold = gold_manifest_from_document(index.gold_manifest.read_json(index.bundle_root))
    structured = _verify_structured_replay(index, profile)
    rationale = _verify_rationale(index)
    parity = _verify_parity(index, gold, profile, oracle_trust)
    benchmark, throughput = _verify_benchmark(index, profile)
    soak = _verify_soak(index, profile)
    return VerifiedPublicStage2Evidence(
        MappingProxyType({
            "structured_replay": structured,
            "rationale": rationale,
            "parity": parity,
            "benchmark": benchmark,
            "soak": soak,
        }),
        throughput,
    )


def _verify_profile_binding(
    index: Stage2ReleaseEvidenceIndex,
    profile: Stage2Profile,
) -> None:
    reranker = profile.reranker_calibration
    qwen = profile.adjudicator_calibration
    if reranker is None or qwen is None:
        raise Stage2EvidenceError("public Stage 2 verification requires a calibrated profile")
    expected_models = {
        "reranker": profile.reranker_lock_hash,
        "qwen": profile.adjudicator_lock_hash,
    }
    expected_thresholds = {
        "reranker": reranker.threshold.hash(),
        "qwen": qwen.threshold.hash(),
    }
    expected_calibrators = {
        "reranker": reranker.calibrator.hash(),
        "qwen": qwen.calibrator.hash(),
    }
    if index.stage2_config_hash != profile.base_runtime_config_hash:
        raise Stage2EvidenceError("public evidence config does not match the released profile")
    if dict(index.model_lock_hashes) != expected_models:
        raise Stage2EvidenceError("public evidence models do not match the released profile")
    if dict(index.calibrator_hashes) != expected_calibrators:
        raise Stage2EvidenceError("public evidence calibrators do not match the released profile")
    if dict(index.threshold_hashes) != expected_thresholds:
        raise Stage2EvidenceError("public evidence thresholds do not match the released profile")


def _verify_structured_replay(
    index: Stage2ReleaseEvidenceIndex,
    profile: Stage2Profile,
) -> VerifiedPublicGate:
    refs = index.public_gates["structured_replay"]
    manifest_document = refs.manifest.read_json(index.bundle_root)
    validate(manifest_document, "stage2-structured-replay-manifest.schema.json")
    manifest = StructuredReplayManifest(
        version=manifest_document["version"],
        pair_ids=tuple(manifest_document["pair_ids"]),
        corpus_hash=manifest_document["corpus_hash"],
        stage2_config_hash=manifest_document["stage2_config_hash"],
        model_lock_hash=manifest_document["model_lock_hash"],
        prompt_hash=manifest_document["prompt_hash"],
        schema_hash=manifest_document["schema_hash"],
    )
    if manifest.stage2_config_hash != index.stage2_config_hash:
        raise Stage2EvidenceError("structured replay config does not match release evidence")
    if manifest.model_lock_hash != index.model_lock_hashes["qwen"]:
        raise Stage2EvidenceError("structured replay model does not match released Qwen")
    if manifest.prompt_hash != profile.prompt_hash or manifest.schema_hash != profile.schema_hash:
        raise Stage2EvidenceError("structured replay prompt or schema does not match the released profile")
    records_document = _one_document(index, refs)
    validate(records_document, "stage2-structured-replay-records.schema.json")
    records = tuple(
        StructuredReplayRecord(
            pair_id=item["pair_id"],
            manifest_hash=item["manifest_hash"],
            first_error=ReplayError(item["first_error"]),
            first_returned_pair_id=item["first_returned_pair_id"],
            first_schema_outside_text=item["first_schema_outside_text"],
            first_think_tag_leak=item["first_think_tag_leak"],
            deterministic_repairs=item["deterministic_repairs"],
            model_retries=item["model_retries"],
            retry_error=(
                ReplayError(item["retry_error"])
                if item["retry_error"] is not None
                else None
            ),
            final_valid=item["final_valid"],
            final_returned_pair_id=item["final_returned_pair_id"],
            final_schema_outside_text=item["final_schema_outside_text"],
            final_think_tag_leak=item["final_think_tag_leak"],
            final_decision=Stage2Decision(item["final_decision"]),
        )
        for item in records_document["records"]
    )
    result = structured_replay_gate(manifest, records)
    return VerifiedPublicGate(
        _evidence_hash(refs),
        result.manifest_hash,
        result.gate,
        {
            "deterministic_repairs": result.deterministic_repairs,
            "first_valid_rate": result.first_valid_rate,
            "model_retries": result.model_retries,
            "record_count": len(records),
            "schema_errors": result.schema_errors,
            "service_errors": result.service_errors,
            "timeouts": result.timeouts,
        },
    )


def _verify_rationale(index: Stage2ReleaseEvidenceIndex) -> VerifiedPublicGate:
    refs = index.public_gates["rationale"]
    manifest_document = refs.manifest.read_json(index.bundle_root)
    validate(manifest_document, "stage2-rationale-audit-manifest.schema.json")
    manifest = RationaleAuditManifest(
        version=manifest_document["version"],
        cases=tuple(
            RationaleAuditCase(item[0], RationaleStratum(item[1]), item[2], item[3])
            for item in manifest_document["cases"]
        ),
        corpus_hash=manifest_document["corpus_hash"],
        model_lock_hash=manifest_document["model_lock_hash"],
        evidence_rubric_hash=manifest_document["evidence_rubric_hash"],
        fabrication_rubric_hash=manifest_document["fabrication_rubric_hash"],
    )
    if manifest.model_lock_hash != index.model_lock_hashes["qwen"]:
        raise Stage2EvidenceError("rationale audit model does not match released Qwen")
    records_document = _one_document(index, refs)
    validate(records_document, "stage2-rationale-audit-records.schema.json")
    records = tuple(
        RationaleAuditRecord(
            item["pair_id"],
            item["manifest_hash"],
            item["evidence_supported"],
            item["severe_fabrication"],
        )
        for item in records_document["records"]
    )
    gate = rationale_audit_gate(manifest, records)
    return VerifiedPublicGate(
        _evidence_hash(refs),
        manifest.hash(),
        gate,
        {
            "evidence_support_rate": sum(item.evidence_supported for item in records) / len(records),
            "record_count": len(records),
            "severe_fabrication_rate": sum(item.severe_fabrication for item in records) / len(records),
        },
    )


def _verify_parity(
    index: Stage2ReleaseEvidenceIndex,
    gold: GoldManifest,
    profile: Stage2Profile,
    oracle_trust: ParityOracleTrust,
) -> VerifiedPublicGate:
    from .stage2_parity import (
        PREPROCESS_CONTRACT,
        WINDOW_SELECTOR,
        parity_workload_from_document,
    )

    refs = index.public_gates["parity"]
    if not isinstance(refs, ParityEvidenceRefs):
        raise Stage2EvidenceError("parity evidence references are invalid")
    manifest_document = refs.manifest.read_json(index.bundle_root)
    validate(manifest_document, "stage2-parity-manifest.schema.json")
    manifest = ParityManifest(
        version=manifest_document["version"],
        pair_ids=tuple(manifest_document["pair_ids"]),
        workload_hash=manifest_document["workload_hash"],
        selection_receipt_hash=manifest_document["selection_receipt_hash"],
        pair_universe_hash=manifest_document["pair_universe_hash"],
        query_assignment_hash=manifest_document["query_assignment_hash"],
        corpus_hash=manifest_document["corpus_hash"],
        tokenizer_hash=manifest_document["tokenizer_hash"],
        preprocess_hash=manifest_document["preprocess_hash"],
        oracle_model_lock_hash=manifest_document["oracle_model_lock_hash"],
        candidate_model_lock_hash=manifest_document["candidate_model_lock_hash"],
        oracle_calibrator_hash=manifest_document["oracle_calibrator_hash"],
        candidate_calibrator_hash=manifest_document["candidate_calibrator_hash"],
        oracle_threshold_artifact_hash=manifest_document["oracle_threshold_artifact_hash"],
        candidate_threshold_artifact_hash=manifest_document["candidate_threshold_artifact_hash"],
        gold_manifest_hash=manifest_document["gold_manifest_hash"],
        dev_manifest_hash=manifest_document["dev_manifest_hash"],
        dev_label_hash=manifest_document["dev_label_hash"],
        calibration_pair_ids_hash=manifest_document["calibration_pair_ids_hash"],
        window_selector_hash=manifest_document["window_selector_hash"],
        low_window_pair_ids=frozenset(manifest_document["low_window_pair_ids"]),
        high_window_pair_ids=frozenset(manifest_document["high_window_pair_ids"]),
    )
    workload = parity_workload_from_document(
        _mapping_document(refs.workload.read_json(index.bundle_root), "parity workload")
    )
    selection_receipt = _mapping_document(
        refs.selection_receipt.read_json(index.bundle_root),
        "parity selection receipt",
    )
    oracle_lock = ModelLock(**_mapping_document(
        refs.oracle_model_lock.read_json(index.bundle_root), "parity oracle model lock"
    ))
    candidate_lock = ModelLock(**_mapping_document(
        refs.candidate_model_lock.read_json(index.bundle_root), "parity candidate model lock"
    ))
    oracle_calibrator = PathCalibrator(**_mapping_document(
        refs.oracle_calibrator.read_json(index.bundle_root), "parity oracle calibrator"
    ))
    candidate_calibrator = PathCalibrator(**_mapping_document(
        refs.candidate_calibrator.read_json(index.bundle_root), "parity candidate calibrator"
    ))
    oracle_threshold = ThresholdArtifact(**_mapping_document(
        refs.oracle_threshold.read_json(index.bundle_root), "parity oracle threshold"
    ))
    candidate_threshold = ThresholdArtifact(**_mapping_document(
        refs.candidate_threshold.read_json(index.bundle_root), "parity candidate threshold"
    ))
    _verify_parity_models(
        index,
        profile,
        oracle_trust,
        refs,
        manifest,
        oracle_lock,
        candidate_lock,
        content_hash(PREPROCESS_CONTRACT),
    )
    _verify_parity_calibration(
        index,
        profile,
        oracle_trust,
        refs,
        manifest,
        oracle_calibrator,
        candidate_calibrator,
        oracle_threshold,
        candidate_threshold,
    )
    _verify_parity_workload(
        manifest,
        workload,
        selection_receipt,
        gold,
        content_hash(WINDOW_SELECTOR),
    )
    scores_document = refs.scores.read_json(index.bundle_root)
    validate(scores_document, "stage2-parity-scores.schema.json")
    expected_score_bindings = {
        "manifest_hash": manifest.hash(),
        "workload_hash": manifest.workload_hash,
        "oracle_model_lock_hash": manifest.oracle_model_lock_hash,
        "candidate_model_lock_hash": manifest.candidate_model_lock_hash,
        "score_count": 10_000,
        "failure_count": 0,
    }
    if any(scores_document[field] != value for field, value in expected_score_bindings.items()):
        raise Stage2EvidenceError("parity scores do not match their manifest and workload")
    scores = tuple(
        ParityScore(
            item["pair_id"],
            item["oracle_score"],
            item["candidate_score"],
        )
        for item in scores_document["scores"]
    )
    _verify_parity_windows(manifest, scores, oracle_calibrator, oracle_threshold)
    result = parity_gate(
        manifest,
        scores,
        oracle_calibrator,
        oracle_threshold,
        candidate_calibrator,
        candidate_threshold,
    )
    return VerifiedPublicGate(
        _parity_evidence_hash(refs),
        result.manifest_hash,
        result.gate,
        {
            "high_threshold_agreement": result.high_threshold_agreement,
            "high_threshold_denominator": result.high_threshold_denominator,
            "kendall_tau_b": result.kendall_tau_b,
            "low_threshold_agreement": result.low_threshold_agreement,
            "low_threshold_denominator": result.low_threshold_denominator,
            "score_count": len(scores),
        },
    )


def _verify_parity_models(
    index: Stage2ReleaseEvidenceIndex,
    profile: Stage2Profile,
    trust: ParityOracleTrust,
    refs: ParityEvidenceRefs,
    manifest: ParityManifest,
    oracle: ModelLock,
    candidate: ModelLock,
    expected_preprocess: str,
) -> None:
    expected_oracle_hash = trust.official_oracle_model_lock_hash
    expected_candidate_hash = index.model_lock_hashes["reranker"]
    if (
        refs.oracle_model_lock.sha256 != expected_oracle_hash
        or manifest.oracle_model_lock_hash != expected_oracle_hash
    ):
        raise Stage2EvidenceError("parity oracle model is not deployment-trusted")
    if (
        refs.candidate_model_lock.sha256 != expected_candidate_hash
        or manifest.candidate_model_lock_hash != expected_candidate_hash
        or profile.reranker_lock_hash != expected_candidate_hash
        or profile.reranker_model_id != candidate.model_id
        or profile.reranker_revision
        != (candidate.conversion_revision or candidate.source_revision)
    ):
        raise Stage2EvidenceError("parity candidate model does not match the released reranker")
    if expected_oracle_hash == expected_candidate_hash:
        raise Stage2EvidenceError("parity rejects oracle self-comparison")
    if (
        oracle.backend != "omlx_rerank"
        or oracle.conversion_repo is not None
        or oracle.format != "safetensors-fp32"
        or oracle.quantization != "none"
    ):
        raise Stage2EvidenceError("parity oracle must be the official FP32 reranker")
    if (
        candidate.backend != "omlx_rerank"
        or candidate.conversion_repo is None
        or candidate.format != "safetensors-bf16"
        or candidate.quantization != "none"
    ):
        raise Stage2EvidenceError("parity candidate must be the audited BF16 conversion")
    if (oracle.source_repo, oracle.source_revision) != (
        candidate.source_repo,
        candidate.source_revision,
    ):
        raise Stage2EvidenceError("parity models do not share the same upstream revision")
    oracle_tokenizer = oracle.file_hashes.get("tokenizer.json")
    candidate_tokenizer = candidate.file_hashes.get("tokenizer.json")
    if (
        oracle_tokenizer != trust.tokenizer_hash
        or candidate_tokenizer != trust.tokenizer_hash
        or manifest.tokenizer_hash != trust.tokenizer_hash
    ):
        raise Stage2EvidenceError("parity models do not share the trusted tokenizer")
    oracle_weights = sorted(
        digest for name, digest in oracle.file_hashes.items() if name.endswith(".safetensors")
    )
    candidate_weights = sorted(
        digest for name, digest in candidate.file_hashes.items() if name.endswith(".safetensors")
    )
    if not oracle_weights or not candidate_weights or oracle_weights == candidate_weights:
        raise Stage2EvidenceError("parity rejects missing or identical model weights")
    if trust.preprocess_hash != expected_preprocess or manifest.preprocess_hash != expected_preprocess:
        raise Stage2EvidenceError("parity preprocessing does not match the trusted contract")


def _verify_parity_calibration(
    index: Stage2ReleaseEvidenceIndex,
    profile: Stage2Profile,
    trust: ParityOracleTrust,
    refs: ParityEvidenceRefs,
    manifest: ParityManifest,
    oracle_calibrator: PathCalibrator,
    candidate_calibrator: PathCalibrator,
    oracle_threshold: ThresholdArtifact,
    candidate_threshold: ThresholdArtifact,
) -> None:
    candidate_binding = profile.reranker_calibration
    if candidate_binding is None:
        raise Stage2EvidenceError("parity candidate requires released reranker calibration")
    oracle_calibrator_hash = oracle_calibrator.hash()
    candidate_calibrator_hash = candidate_calibrator.hash()
    oracle_threshold_hash = oracle_threshold.hash()
    candidate_threshold_hash = candidate_threshold.hash()
    if (
        oracle_calibrator_hash != trust.oracle_calibrator_hash
        or oracle_threshold_hash != trust.oracle_threshold_artifact_hash
        or manifest.oracle_calibrator_hash != trust.oracle_calibrator_hash
        or manifest.oracle_threshold_artifact_hash
        != trust.oracle_threshold_artifact_hash
    ):
        raise Stage2EvidenceError("parity oracle calibration is not deployment-trusted")
    if (
        candidate_calibrator_hash != index.calibrator_hashes["reranker"]
        or candidate_threshold_hash != index.threshold_hashes["reranker"]
        or manifest.candidate_calibrator_hash != candidate_calibrator_hash
        or manifest.candidate_threshold_artifact_hash != candidate_threshold_hash
        or candidate_binding.calibrator.hash() != candidate_calibrator_hash
        or candidate_binding.threshold.hash() != candidate_threshold_hash
    ):
        raise Stage2EvidenceError("parity candidate calibration does not match the released profile")
    if (
        oracle_calibrator.model_lock_hash != refs.oracle_model_lock.sha256
        or oracle_threshold.model_lock_hash != refs.oracle_model_lock.sha256
        or candidate_calibrator.model_lock_hash != refs.candidate_model_lock.sha256
        or candidate_threshold.model_lock_hash != refs.candidate_model_lock.sha256
        or candidate_threshold.stage2_config_hash != index.stage2_config_hash
    ):
        raise Stage2EvidenceError("parity calibration model or config binding is invalid")


def _verify_parity_workload(
    manifest: ParityManifest,
    workload: Any,
    receipt: Mapping[str, Any],
    gold: GoldManifest,
    expected_selector_hash: str,
) -> None:
    pair_ids = tuple(pair.pair_id for pair in workload.pairs)
    if (
        tuple(manifest.pair_ids) != pair_ids
        or manifest.workload_hash != workload.hash()
        or manifest.query_assignment_hash != workload.query_assignment_hash()
        or manifest.corpus_hash != workload.corpus_hash()
        or manifest.gold_manifest_hash != gold.hash()
        or manifest.dev_manifest_hash != gold.dev_hash()
        or manifest.window_selector_hash != expected_selector_hash
    ):
        raise Stage2EvidenceError("parity workload or release provenance does not match its manifest")
    if manifest.selection_receipt_hash != content_hash(receipt):
        raise Stage2EvidenceError("parity selection receipt hash does not match its manifest")
    parity_receipt = receipt.get("parity")
    paper_ids = sorted(pair.paper_id for pair in workload.pairs)
    if (
        not isinstance(parity_receipt, Mapping)
        or parity_receipt.get("paper_count") != 10_000
        or parity_receipt.get("paper_ids") != paper_ids
        or parity_receipt.get("papers_corpus_hash") != workload.corpus_hash()
    ):
        raise Stage2EvidenceError("parity receipt does not exactly bind the 10,000-paper workload")


def _verify_parity_windows(
    manifest: ParityManifest,
    scores: Sequence[ParityScore],
    oracle_calibrator: PathCalibrator,
    oracle_threshold: ThresholdArtifact,
) -> None:
    by_id = {item.pair_id: item for item in scores}
    if len(by_id) != 10_000 or set(by_id) != set(manifest.pair_ids):
        raise Stage2EvidenceError("parity scores do not exactly cover the 10,000-pair workload")

    def window(threshold: float) -> frozenset[str]:
        return frozenset(pair_id for _, pair_id in sorted(
            (
                abs(oracle_calibrator.predict(score.oracle_score) - threshold),
                pair_id,
            )
            for pair_id, score in by_id.items()
        )[:200])

    if (
        manifest.low_window_pair_ids != window(oracle_threshold.low)
        or manifest.high_window_pair_ids != window(oracle_threshold.high)
    ):
        raise Stage2EvidenceError("parity threshold windows do not match the trusted selector")


def _verify_benchmark(
    index: Stage2ReleaseEvidenceIndex,
    profile: Stage2Profile,
) -> tuple[VerifiedPublicGate, tuple[float, float, float]]:
    refs = index.public_gates["benchmark"]
    manifest_document = refs.manifest.read_json(index.bundle_root)
    validate(manifest_document, "stage2-performance-manifest.schema.json")
    manifest = PerformanceRoutingManifest(
        version=manifest_document["version"],
        corpus_hash=manifest_document["corpus_hash"],
        stage2_config_hash=manifest_document["stage2_config_hash"],
        model_lock_hashes=tuple(manifest_document["model_lock_hashes"]),
        threshold_artifact_hashes=tuple(manifest_document["threshold_artifact_hashes"]),
        output_token_limit=manifest_document["output_token_limit"],
        cases=tuple(
            PerformanceCase(item["pair_id"], item["input_tokens"], item["abstract_missing"])
            for item in manifest_document["cases"]
        ),
        normal_qwen_ids=frozenset(manifest_document["normal_qwen_ids"]),
        stress_qwen_ids=frozenset(manifest_document["stress_qwen_ids"]),
        pipeline_components=tuple(manifest_document["pipeline_components"]),
        input_token_estimator=manifest_document["input_token_estimator"],
    )
    _verify_benchmark_manifest_binding(index, manifest)
    papers, input_hash = _verify_benchmark_papers(index, refs, manifest, profile)
    records = tuple(
        _performance_execution_record(
            _execution_document(index, ref),
            manifest,
            input_hash,
            profile,
        )
        for ref in refs.records
    )
    gate = performance_gate(manifest, records)
    normal_records = tuple(item for item in records if item.scenario == "normal")
    throughput = tuple(len(manifest.cases) / item.duration_seconds for item in normal_records)
    metrics: dict[str, Any] = {"record_count": len(records)}
    for scenario in ("normal", "stress"):
        if sum(item.scenario == scenario for item in records) == 3:
            metrics[scenario] = dict(performance_summary(records, scenario))
    return (
        VerifiedPublicGate(
            _evidence_hash(refs),
            manifest.hash(),
            gate,
            metrics,
        ),
        throughput,
    )


def _verify_soak(
    index: Stage2ReleaseEvidenceIndex,
    profile: Stage2Profile,
) -> VerifiedPublicGate:
    refs = index.public_gates["soak"]
    manifest_document = refs.manifest.read_json(index.bundle_root)
    validate(manifest_document, "stage2-soak-manifest.schema.json")
    manifest = SoakManifest(
        version=manifest_document["version"],
        corpus_hash=manifest_document["corpus_hash"],
        stage2_config_hash=manifest_document["stage2_config_hash"],
        model_lock_hashes=tuple(manifest_document["model_lock_hashes"]),
        threshold_artifact_hashes=tuple(manifest_document["threshold_artifact_hashes"]),
        output_token_limit=manifest_document["output_token_limit"],
        cases=tuple(PerformanceCase(item[0], item[1], item[2]) for item in manifest_document["cases"]),
        input_token_estimator=manifest_document["input_token_estimator"],
    )
    _verify_benchmark_manifest_binding(index, manifest)
    _, input_hash = _verify_benchmark_papers(index, refs, manifest, profile)
    record = _soak_execution_record(
        _execution_document(index, refs.records[0]),
        manifest,
        input_hash,
        profile,
    )
    gate = soak_gate(manifest, record)
    return VerifiedPublicGate(
        _evidence_hash(refs),
        manifest.hash(),
        gate,
        {
            "duration_seconds": record.duration_seconds,
            "request_count": record.request_count,
            "service_request_failure_rate": record.service_request_failure_rate,
        },
    )


def _verify_benchmark_manifest_binding(
    index: Stage2ReleaseEvidenceIndex,
    manifest: PerformanceRoutingManifest | SoakManifest,
) -> None:
    expected_models = (
        index.model_lock_hashes["reranker"],
        index.model_lock_hashes["qwen"],
    )
    expected_thresholds = (
        index.threshold_hashes["reranker"],
        index.threshold_hashes["qwen"],
    )
    if manifest.stage2_config_hash != index.stage2_config_hash:
        raise Stage2EvidenceError("benchmark config does not match release evidence")
    if manifest.model_lock_hashes != expected_models:
        raise Stage2EvidenceError("benchmark models do not match release evidence")
    if manifest.threshold_artifact_hashes != expected_thresholds:
        raise Stage2EvidenceError("benchmark thresholds do not match release evidence")


def _performance_execution_record(
    document: Any,
    manifest: PerformanceRoutingManifest,
    input_hash: str,
    profile: Stage2Profile,
) -> PerformanceRunRecord:
    _verify_execution_record(document, manifest, "performance", input_hash, profile)
    return BenchmarkExecutionRecord(document).as_performance_record()


def _soak_execution_record(
    document: Any,
    manifest: SoakManifest,
    input_hash: str,
    profile: Stage2Profile,
) -> SoakRunRecord:
    _verify_execution_record(document, manifest, "soak", input_hash, profile)
    return BenchmarkExecutionRecord(document).as_soak_record()


def _verify_execution_record(
    document: Any,
    manifest: PerformanceRoutingManifest | SoakManifest,
    kind: str,
    input_hash: str,
    profile: Stage2Profile,
) -> None:
    if not isinstance(document, dict):
        raise Stage2EvidenceError("benchmark execution record must be a JSON object")
    required = {
        "record_version", "kind", "scenario", "run_id", "manifest_hash", "corpus_hash",
        "input_hash", "release_hash", "stage2_config_hash", "observed_stage2_config_hash",
        "full_profile_hash", "model_lock_hashes", "threshold_artifact_hashes",
        "observed_threshold_artifact_hashes", "model_releases", "prompt_hash", "schema_hash",
        "output_token_limit", "observed_output_token_limit", "fixture_scale", "case_count",
        "input_token_count", "latency_sample_unit", "peak_memory_gb", "rss_scope",
        "request_count", "request_count_unit", "service_request_count", "service_request_trace",
        "service_failed_request_count", "reranker_fallback_measurement_available",
        "completed_pair_ids", "needs_review_pair_ids", "failed_request_pair_ids",
        "qwen_pair_ids", "frozen_qwen_routing_matches", "routing_mode", "environment",
        "expected_components", "executed_components", "sqlite_commit_count", "sqlite_commit_unit",
        "result_count", "missing_result_count", "duplicate_result_count", "warmed",
        "resume_verified", "resume_model_call_count", "resumed_pair_count", "oom",
        "process_crash", "memory_pressure_critical", "memory_pressure_sampled",
        "unbounded_memory_growth", "duration_seconds", "failed_request_count",
        "measurement_evidence_version", "rss_samples_bytes", "memory_pressure_samples",
        "resumed_pair_ids", "model_call_trace", "resume_service_request_trace",
    }
    if kind == "performance":
        required |= {"p50_seconds", "p95_seconds"}
    missing = sorted(required - set(document))
    if missing:
        raise Stage2EvidenceError(f"benchmark execution record is missing audited fields: {missing}")
    if set(document) != BENCHMARK_EXECUTION_FIELDS:
        raise Stage2EvidenceError("benchmark execution record has unsupported or missing fields")
    boolean_fields = (
        "fixture_scale",
        "warmed",
        "resume_verified",
        "oom",
        "process_crash",
        "memory_pressure_critical",
        "memory_pressure_sampled",
        "unbounded_memory_growth",
        "reranker_fallback_measurement_available",
    )
    if any(type(document[field]) is not bool for field in boolean_fields):
        raise Stage2EvidenceError("benchmark execution boolean fields must be JSON booleans")
    expected_case_count = 1_000 if kind == "performance" else 10_000
    expected_scenario = {"normal", "stress"} if kind == "performance" else {None}
    exact = {
        "record_version": 2,
        "measurement_evidence_version": "1",
        "kind": kind,
        "manifest_hash": manifest.hash(),
        "corpus_hash": manifest.corpus_hash,
        "input_hash": input_hash,
        "stage2_config_hash": manifest.stage2_config_hash,
        "observed_stage2_config_hash": manifest.stage2_config_hash,
        "full_profile_hash": profile.full_profile_hash,
        "model_lock_hashes": list(manifest.model_lock_hashes),
        "threshold_artifact_hashes": list(manifest.threshold_artifact_hashes),
        "observed_threshold_artifact_hashes": list(manifest.threshold_artifact_hashes),
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
        "case_count": expected_case_count,
        "input_token_count": sum(item.input_tokens for item in manifest.cases),
        "latency_sample_unit": "omlx_service_request",
        "request_count_unit": "manifest_case",
        "reranker_fallback_measurement_available": True,
        "expected_components": list(("rules", "reranker", "qwen", "schema_validation", "sqlite_commit")),
        "sqlite_commit_unit": "persisted_filter_decision",
        "result_count": expected_case_count,
        "missing_result_count": 0,
        "duplicate_result_count": 0,
        "memory_pressure_sampled": True,
        "resume_verified": True,
        "resume_model_call_count": 0,
        "resumed_pair_count": expected_case_count,
        "resumed_pair_ids": sorted(item.pair_id for item in manifest.cases),
    }
    for field, value in exact.items():
        if document[field] != value:
            raise Stage2EvidenceError(f"benchmark execution {field} does not match its manifest")
    if document["scenario"] not in expected_scenario:
        raise Stage2EvidenceError("benchmark execution scenario is invalid")
    if document["rss_scope"] == "runner_process_high_water_rss":
        raise Stage2EvidenceError("benchmark RSS evidence does not cover the model service")
    if not re.fullmatch(
        r"macos_ps_current_rss:runner_pid=[1-9][0-9]*;omlx_pids=[1-9][0-9]*(?:,[1-9][0-9]*)*",
        document["rss_scope"],
    ):
        raise Stage2EvidenceError("benchmark RSS scope is not a runner plus oMLX process set")
    if not all(
        isinstance(document[field], str)
        and re.fullmatch(r"[a-f0-9]{64}", document[field]) is not None
        for field in ("input_hash", "release_hash", "full_profile_hash", "prompt_hash", "schema_hash")
    ):
        raise Stage2EvidenceError("benchmark execution provenance hashes are incomplete")
    if (
        set(document["model_releases"]) != {"reranker", "qwen"}
        or any(
            not isinstance(item, dict)
            or set(item) != {"model_id", "revision"}
            or not all(isinstance(value, str) and value for value in item.values())
            for item in document["model_releases"].values()
        )
    ):
        raise Stage2EvidenceError("benchmark execution model releases are incomplete")
    _verify_service_measurements(document)
    _verify_model_call_measurements(document)
    _verify_memory_measurements(document)
    _verify_result_aggregates(document, expected_case_count)
    _verify_execution_aggregates(document, expected_case_count)
    if kind == "performance":
        if (
            type(document["frozen_qwen_routing_matches"]) is not bool
            or document["frozen_qwen_routing_matches"] is not True
            or document["routing_mode"] != "performance_only_manifest"
        ):
            raise Stage2EvidenceError("performance execution did not use frozen Qwen routing")
    elif document["frozen_qwen_routing_matches"] is not None or document["routing_mode"] != "quality_thresholds":
        raise Stage2EvidenceError("soak execution did not use released quality thresholds")


def _verify_benchmark_papers(
    index: Stage2ReleaseEvidenceIndex,
    refs: GateEvidenceRefs,
    manifest: PerformanceRoutingManifest | SoakManifest,
    profile: Stage2Profile,
) -> tuple[Sequence[Any], str]:
    if refs.papers is None:
        raise Stage2EvidenceError("benchmark evidence is missing its public paper inputs")
    document = refs.papers.read_json(index.bundle_root)
    validate(document, "stage2-benchmark-papers.schema.json")
    papers = benchmark_papers_from_document(document)
    expected = {item.pair_id for item in manifest.cases}
    actual = {item.paper_id for item in papers}
    if len(actual) != len(papers) or actual != expected:
        raise Stage2EvidenceError("benchmark papers do not exactly match the manifest cases")
    expected_missing = {item.pair_id for item in manifest.cases if item.abstract_missing}
    observed_missing = {
        item.paper_id for item in papers if not item.abstract or not item.abstract.strip()
    }
    if observed_missing != expected_missing:
        raise Stage2EvidenceError("benchmark papers do not match missing-abstract flags")
    if benchmark_corpus_hash(papers) != manifest.corpus_hash:
        raise Stage2EvidenceError("benchmark papers do not match the manifest corpus hash")
    _verify_benchmark_case_tokens(manifest, papers, profile)
    return papers, benchmark_workload_hash(manifest.cases, papers)


def _verify_benchmark_case_tokens(
    manifest: PerformanceRoutingManifest | SoakManifest,
    papers: Sequence[Stage2Paper],
    profile: Stage2Profile,
) -> None:
    paper_by_id = {paper.paper_id: paper for paper in papers}
    for case in manifest.cases:
        paper = paper_by_id[case.pair_id]
        messages = adjudication_messages(
            query_version=profile.query_version,
            query=profile.query,
            paper=paper,
        )
        if case.input_tokens != estimate_omlx_chat_input_token_proxy(messages):
            raise Stage2EvidenceError(
                "benchmark case input_tokens do not match the released prompt proxy estimator"
            )


def _verify_service_measurements(document: Mapping[str, Any]) -> None:
    trace = document["service_request_trace"]
    if not isinstance(trace, list) or not trace:
        raise Stage2EvidenceError("benchmark requires raw oMLX service request evidence")
    allowed_paths = {"/v1/rerank", "/v1/chat/completions"}
    for item in trace:
        if (
            not isinstance(item, dict)
            or set(item) != {"path", "duration_seconds", "document_count", "failed"}
            or item["path"] not in allowed_paths
            or isinstance(item["duration_seconds"], bool)
            or not isinstance(item["duration_seconds"], (int, float))
            or item["duration_seconds"] < 0
            or type(item["document_count"]) is not int
            or item["document_count"] < 1
            or type(item["failed"]) is not bool
        ):
            raise Stage2EvidenceError("benchmark service request trace is invalid")
    service_count = len(trace)
    failed_count = sum(item["failed"] for item in trace)
    if (
        document["service_request_count"] != service_count
        or document["service_failed_request_count"] != failed_count
        or document["service_pair_attempt_count"] != sum(item["document_count"] for item in trace)
        or document["latency_sample_count"] != service_count
        or not isclose(
            document["service_request_failure_rate"],
            failed_count / service_count,
            rel_tol=0,
            abs_tol=1e-15,
        )
    ):
        raise Stage2EvidenceError("benchmark service request aggregates do not match raw evidence")
    latency = document["latency_by_path"]
    if not isinstance(latency, dict) or set(latency) != {"reranker", "qwen"}:
        raise Stage2EvidenceError("benchmark latency path summary is incomplete")
    for name, path in (("reranker", "/v1/rerank"), ("qwen", "/v1/chat/completions")):
        values = sorted(item["duration_seconds"] for item in trace if item["path"] == path)
        summary = latency[name]
        if not isinstance(summary, dict) or set(summary) != {"sample_count", "p50_seconds", "p95_seconds"}:
            raise Stage2EvidenceError("benchmark latency summary shape is invalid")
        p50 = values[max(0, -(-len(values) // 2) - 1)] if values else 0.0
        p95 = values[max(0, -(-(95 * len(values)) // 100) - 1)] if values else 0.0
        if summary != {"sample_count": len(values), "p50_seconds": p50, "p95_seconds": p95}:
            raise Stage2EvidenceError("benchmark latency summary does not match raw evidence")
    all_values = sorted(item["duration_seconds"] for item in trace)
    p50 = all_values[max(0, -(-len(all_values) // 2) - 1)]
    p95 = all_values[max(0, -(-(95 * len(all_values)) // 100) - 1)]
    if document["p50_seconds"] != p50 or document["p95_seconds"] != p95:
        raise Stage2EvidenceError("benchmark overall latency does not match raw evidence")


def _verify_model_call_measurements(document: Mapping[str, Any]) -> None:
    trace = document["model_call_trace"]
    if not isinstance(trace, list) or not trace:
        raise Stage2EvidenceError("benchmark requires raw model-call routing evidence")
    for item in trace:
        if (
            not isinstance(item, dict)
            or set(item) != {"backend", "pair_ids", "duration_seconds", "failed"}
            or item["backend"] not in {"reranker", "qwen"}
            or not isinstance(item["pair_ids"], list)
            or not item["pair_ids"]
            or len(item["pair_ids"]) != len(set(item["pair_ids"]))
            or not all(isinstance(pair_id, str) and pair_id for pair_id in item["pair_ids"])
            or (item["backend"] == "qwen" and len(item["pair_ids"]) != 1)
            or isinstance(item["duration_seconds"], bool)
            or not isinstance(item["duration_seconds"], (int, float))
            or item["duration_seconds"] < 0
            or type(item["failed"]) is not bool
        ):
            raise Stage2EvidenceError("benchmark model-call trace is invalid")
    reranker = [item for item in trace if item["backend"] == "reranker"]
    qwen = [item for item in trace if item["backend"] == "qwen"]
    service = document["service_request_trace"]
    service_reranker_count = sum(item["path"] == "/v1/rerank" for item in service)
    service_qwen_count = sum(item["path"] == "/v1/chat/completions" for item in service)
    expected_fallbacks = max(
        0,
        service_reranker_count - sum(
            ceil(len(item["pair_ids"]) / document["batch_concurrency"]["document_batch_size"])
            for item in reranker
        ),
    )
    if (
        document["model_call_count"] != len(trace)
        or document["reranker_batch_call_count"] != len(reranker)
        or document["adjudicator_call_count"] != len(qwen)
        or document["pair_attempt_count"] != sum(len(item["pair_ids"]) for item in trace)
        or document["backend_failed_call_count"] != sum(item["failed"] for item in trace)
        or set(pair_id for item in qwen for pair_id in item["pair_ids"])
        != set(document["qwen_pair_ids"])
        or service_qwen_count != len(qwen)
        or service_reranker_count < len(reranker)
        or document["reranker_fallback_count"] != expected_fallbacks
        or document["service_pair_attempt_count"] < document["pair_attempt_count"]
        or document["resume_service_request_trace"] != []
    ):
        raise Stage2EvidenceError("benchmark model-call routing does not match raw evidence")


def _verify_memory_measurements(document: Mapping[str, Any]) -> None:
    samples = document["rss_samples_bytes"]
    pressure = document["memory_pressure_samples"]
    if (
        not isinstance(samples, list)
        or len(samples) < 4
        or any(type(value) is not int or value < 0 for value in samples)
        or not isinstance(pressure, list)
        or len(pressure) != len(samples)
        or any(type(value) is not bool for value in pressure)
    ):
        raise Stage2EvidenceError("benchmark raw RSS or memory-pressure evidence is invalid")
    peak = max(samples)
    monotonic = all(right >= left for left, right in zip(samples, samples[1:]))
    unbounded = samples[0] > 0 and monotonic and samples[-1] > samples[0] * 1.25
    growth = {
        "version": "post_warmup_monotonic_25_percent_v1",
        "monotonic_non_decreasing": monotonic,
        "growth_bytes": samples[-1] - samples[0],
        "growth_ratio": samples[-1] / samples[0] if samples[0] else None,
        "minimum_sample_count": 4,
    }
    if (
        document["rss_start_bytes"] != samples[0]
        or document["rss_end_bytes"] != samples[-1]
        or document["peak_rss_bytes"] != peak
        or document["rss_sample_count"] != len(samples)
        or document["peak_memory_gb"] != peak / 1024**3
        or document["memory_pressure_critical"] != any(pressure)
        or document["memory_growth_detector"] != growth
        or document["unbounded_memory_growth"] != unbounded
    ):
        raise Stage2EvidenceError("benchmark memory summaries do not match raw evidence")


def _verify_result_aggregates(document: Mapping[str, Any], case_count: int) -> None:
    completed = document["completed_pair_ids"]
    needs_review = document["needs_review_pair_ids"]
    qwen = document["qwen_pair_ids"]
    failed = document["failed_request_pair_ids"]
    qwen_count = len(qwen)
    qwen_share = qwen_count / case_count
    capacity = "severe" if qwen_share > 0.30 else "warning" if qwen_share > 0.15 else "normal"
    share_alarms = ["stage2.adjudicator_share_exceeded"] if qwen_share > 0.15 else []
    if (
        document["request_count"] != case_count
        or document["failed_request_count"] != len(failed)
        or document["request_failure_rate"] != len(failed) / case_count
        or document["result_count"] != len(completed) + len(needs_review)
        or document["qwen_count"] != qwen_count
        or document["adjudicator_count"] != qwen_count
        or document["qwen_share"] != qwen_share
        or document["adjudicator_share"] != qwen_share
        or document["qwen_capacity_level"] != capacity
        or document["adjudicator_capacity"] != capacity
        or document["qwen_share_alarms"] != share_alarms
    ):
        raise Stage2EvidenceError("benchmark result aggregates do not match raw evidence")


def _verify_execution_aggregates(document: Mapping[str, Any], case_count: int) -> None:
    duration = document["duration_seconds"]
    input_tokens = document["input_token_count"]
    expected_alarms = list(benchmark_alarm_codes(
        document["qwen_share_alarms"],
        request_failure_rate=document["request_failure_rate"],
        service_request_failure_rate=document["service_request_failure_rate"],
        peak_memory_gb=document["peak_memory_gb"],
        memory_pressure_critical=document["memory_pressure_critical"],
        unbounded_memory_growth=document["unbounded_memory_growth"],
    ))
    batch = document["batch_concurrency"]
    environment_batch = document["environment"]["batch_config"]
    model_call_count = document["model_call_count"]
    if (
        isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or duration <= 0
        or document["papers_per_second"] != case_count / duration
        or document["input_tokens_per_second"] != input_tokens / duration
        or document["pair_tokens_per_second"] != input_tokens / duration
        or type(model_call_count) is not int
        or model_call_count < 1
        or document["backend_call_failure_rate"]
        != document["backend_failed_call_count"] / model_call_count
        or document["alarm_codes"] != expected_alarms
        or not isinstance(batch, dict)
        or any(batch.get(key) != value for key, value in environment_batch.items())
        or any(
            type(batch.get(key)) is not int or batch[key] < 1
            for key in (
                "document_batch_size",
                "reranker_max_in_flight",
                "adjudicator_concurrency",
            )
        )
    ):
        raise Stage2EvidenceError("benchmark execution aggregates do not match raw evidence")


def _one_document(
    index: Stage2ReleaseEvidenceIndex,
    refs: GateEvidenceRefs,
) -> Any:
    if len(refs.records) != 1:
        raise Stage2EvidenceError("Stage 2 gate requires exactly one records artifact")
    return refs.records[0].read_json(index.bundle_root)


def _evidence_hash(refs: GateEvidenceRefs) -> str:
    return content_hash({
        "manifest": refs.manifest.sha256,
        "papers": refs.papers.sha256 if refs.papers is not None else None,
        "records": [item.sha256 for item in refs.records],
    })


def _parity_evidence_hash(refs: ParityEvidenceRefs) -> str:
    return content_hash({
        "candidate_calibrator": refs.candidate_calibrator.sha256,
        "candidate_model_lock": refs.candidate_model_lock.sha256,
        "candidate_threshold": refs.candidate_threshold.sha256,
        "manifest": refs.manifest.sha256,
        "oracle_calibrator": refs.oracle_calibrator.sha256,
        "oracle_model_lock": refs.oracle_model_lock.sha256,
        "oracle_threshold": refs.oracle_threshold.sha256,
        "scores": refs.scores.sha256,
        "selection_receipt": refs.selection_receipt.sha256,
        "workload": refs.workload.sha256,
    })


def _mapping_document(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise Stage2EvidenceError(f"{label} must be a JSON object")
    return value


def _execution_document(
    index: Stage2ReleaseEvidenceIndex,
    ref: ArtifactRef,
) -> Any:
    raw = ref.read_bytes(index.bundle_root)
    document = json.loads(raw)
    if raw != canonical_json(document):
        raise Stage2EvidenceError("benchmark execution record is not canonical JSON")
    return document
