from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from math import nan

import pytest

from paper_agent.stage2_evaluation import (
    Adjudication,
    Annotation,
    BenchmarkEnvironment,
    BootstrapInterval,
    CalibrationExample,
    CalibrationPath,
    CandidateEvaluator,
    CandidateModelArtifacts,
    GateResult,
    GoldLabelStore,
    GoldManifest,
    GoldPair,
    GoldSplit,
    ParityManifest,
    ParityScore,
    PathCalibrator,
    PerformanceCase,
    PerformanceRoutingManifest,
    PerformanceRunRecord,
    Prediction,
    PromotionEvaluator,
    PromotionSubmission,
    RationaleAuditCase,
    RationaleAuditManifest,
    RationaleAuditRecord,
    RationaleStratum,
    ReplayError,
    ReviewReason,
    SoakManifest,
    SoakRunRecord,
    Stage2Decision,
    StructuredReplayManifest,
    StructuredReplayRecord,
    ThresholdArtifact,
    complete_double_annotation,
    determinism_gate,
    fit_path_calibrator,
    kendall_tau_b,
    load_gold_manifest,
    load_promotion_marker,
    measure_predictions,
    paired_bootstrap_comparison,
    parity_gate,
    performance_gate,
    phase3_release_gate,
    promotion_gate,
    rationale_audit_gate,
    soak_gate,
    structured_replay_gate,
    winner_gate,
    write_gold_manifest,
)


def _gold() -> tuple[GoldManifest, GoldLabelStore]:
    pairs: list[GoldPair] = []
    for split, size in ((GoldSplit.DEV, 300), (GoldSplit.HIDDEN_HARD, 150), (GoldSplit.HIDDEN_REAL, 150)):
        for index in range(size):
            pairs.append(
                GoldPair(
                    f"paper-{split.value}-{index}", f"topic-{index % 6}", "zh" if index % 2 else "en",
                    "frozen-crawler-snapshot", 0.2, f"family-{split.value}-{index}", "corpus-v1", split,
                    abstract_incomplete=split is not GoldSplit.HIDDEN_REAL and index < size // 10,
                    sampled_from_natural_distribution=split is GoldSplit.HIDDEN_REAL,
                    cross_language_match=index % 20 == 0,
                )
            )
    manifest = GoldManifest(1, "corpus-v1", tuple(pairs), ("en", "zh"))
    # Labels depend on a private salt, not a public paper-id pattern.
    labels = {
        pair.pair_id: 2 if sha256(f"private-label-salt:{pair.pair_id}".encode()).digest()[0] % 3 else 0
        for pair in pairs
    }
    dev = [pair for pair in pairs if pair.split is GoldSplit.DEV]
    hard = [pair for pair in pairs if pair.split is GoldSplit.HIDDEN_HARD]
    hard_negatives = {pair.pair_id for pair in dev[:60]} | {pair.pair_id for pair in hard[:30]}
    for pair_id in hard_negatives:
        labels[pair_id] = 0
    hard_positives = {dev[-1].pair_id, hard[-1].pair_id}
    for pair_id in hard_positives:
        labels[pair_id] = 3
    return manifest, GoldLabelStore(
        labels, "annotation-artifact-v1", frozenset(hard_negatives), frozenset(hard_positives)
    )


def _dev_labels(manifest: GoldManifest, labels: GoldLabelStore) -> GoldLabelStore:
    ids = {pair.pair_id for pair in manifest.pairs if pair.split is GoldSplit.DEV}
    return GoldLabelStore(
        {pair_id: labels.labels[pair_id] for pair_id in ids}, labels.annotation_artifact_hash
    )


def _candidate_artifacts(
    manifest: GoldManifest, labels: GoldLabelStore, candidate_id: str
) -> CandidateModelArtifacts:
    dev_labels = _dev_labels(manifest, labels)
    calibration_pair_ids = tuple(sorted(dev_labels.labels))
    calibration_pair_ids_hash = sha256(
        json.dumps(calibration_pair_ids, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    calibrators = {
        path: PathCalibrator(
            1, path, 1.0, 0.0, manifest.dev_hash(), manifest.hash(),
            f"{candidate_id}-{path.value}-model-lock", dev_labels.hash(),
            calibration_pair_ids_hash, 300, calibration_pair_ids,
        )
        for path in CalibrationPath
    }
    thresholds = {
        path: ThresholdArtifact(
            1, path, 0.25, 0.75, calibrator.hash(), calibrator.model_lock_hash,
            manifest.dev_hash(), dev_labels.hash(), f"{candidate_id}-stage2-config",
        )
        for path, calibrator in calibrators.items()
    }
    return CandidateModelArtifacts(candidate_id, calibrators, thresholds)


def _prediction(
    pair: GoldPair,
    manifest: GoldManifest,
    artifacts: CandidateModelArtifacts,
    decision: Stage2Decision,
    *,
    run: int = 0,
    path: CalibrationPath = CalibrationPath.RERANKER,
    review_reason: ReviewReason | None = None,
) -> Prediction:
    calibrator = artifacts.calibrators[path]
    threshold = artifacts.thresholds[path]
    raw_score: float | None
    if decision is Stage2Decision.RELEVANT:
        raw_score = 4.0
    elif decision is Stage2Decision.IRRELEVANT:
        raw_score = -4.0
    elif review_reason in {ReviewReason.SCHEMA_ERROR, ReviewReason.TIMEOUT, ReviewReason.SERVICE_ERROR}:
        raw_score = None
    else:
        raw_score = 0.0
    probability = 0.5 if raw_score is None else calibrator.predict(raw_score)
    inference_hash = sha256(f"{artifacts.candidate_id}:{run}:{pair.pair_id}:{decision.value}".encode()).hexdigest()
    return Prediction(
        pair.pair_id, artifacts.candidate_id, decision, raw_score, probability, path, calibrator.hash(),
        threshold.hash(), calibrator.model_lock_hash, manifest.hash(), threshold.stage2_config_hash,
        inference_hash, review_reason,
    )


def _ideal_predictions(
    pairs: list[GoldPair] | tuple[GoldPair, ...],
    manifest: GoldManifest,
    labels: GoldLabelStore,
    artifacts: CandidateModelArtifacts,
    *,
    run: int = 0,
) -> tuple[Prediction, ...]:
    return tuple(
        _prediction(
            pair, manifest, artifacts,
            Stage2Decision.RELEVANT if labels.labels[pair.pair_id] >= 2 else Stage2Decision.IRRELEVANT,
            run=run,
        )
        for pair in pairs
    )


def _submission(
    candidate_id: str, pairs: tuple[GoldPair, ...], manifest: GoldManifest, labels: GoldLabelStore,
    artifacts: CandidateModelArtifacts,
) -> PromotionSubmission:
    return PromotionSubmission(
        candidate_id,
        tuple(_ideal_predictions(pairs, manifest, labels, artifacts, run=run) for run in range(3)),
    )


def test_public_manifest_is_exact_has_stable_pair_ids_and_never_serializes_private_strata(tmp_path) -> None:
    manifest, labels = _gold()
    manifest.validate(labels)
    path = tmp_path / "gold.json"
    write_gold_manifest(path, manifest)
    document = json.loads(path.read_text())
    assert len({item["pair_id"] for item in document["pairs"]}) == 600
    assert all("hard_negative" not in item and "hard_positive" not in item for item in document["pairs"])
    assert load_gold_manifest(path).hash() == manifest.hash()

    document["labels"] = {document["pairs"][0]["pair_id"]: 3}
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="object with pairs"):
        load_gold_manifest(path)


def test_manifest_enforces_pair_family_language_cross_match_and_equal_probability_real_sampling() -> None:
    manifest, _ = _gold()
    duplicate = replace(manifest.pairs[1], paper_id=manifest.pairs[0].paper_id, topic=manifest.pairs[0].topic)
    with pytest.raises(ValueError, match="pair_ids"):
        GoldManifest(1, "corpus-v1", (manifest.pairs[0], duplicate))

    pairs = list(manifest.pairs)
    hidden_index = next(index for index, pair in enumerate(pairs) if pair.split is GoldSplit.HIDDEN_HARD)
    pairs[hidden_index] = replace(pairs[hidden_index], paper_family=pairs[0].paper_family)
    with pytest.raises(ValueError, match="family"):
        replace(manifest, pairs=tuple(pairs)).validate_sampling_structure()

    pairs = list(manifest.pairs)
    real_index = next(index for index, pair in enumerate(pairs) if pair.split is GoldSplit.HIDDEN_REAL)
    pairs[real_index] = replace(pairs[real_index], sampling_probability=0.1)
    with pytest.raises(ValueError, match="equal-probability"):
        replace(manifest, pairs=tuple(pairs)).validate_sampling_structure()


def test_double_annotation_has_fixed_alignment_and_only_disagreements_are_adjudicated() -> None:
    manifest, _ = _gold()
    pairs = manifest.pairs[:4]
    first, second = (0, 1, 2, 3), (0, 1, 2, 2)
    annotations = tuple(
        Annotation(pair.pair_id, annotator, label)
        for pair, a, b in zip(pairs, first, second, strict=True)
        for annotator, label in (("b", b), ("a", a))
    )
    disagreement = pairs[-1].pair_id
    summary = complete_double_annotation(
        pairs, tuple(reversed(annotations)), (Adjudication(disagreement, "c", 3),),
        annotator_order=("a", "b"), adjudicator_id="c", rubric_version=1, rubric_hash="rubric-v1",
    )
    assert summary.quadratic_weighted_kappa >= 0.75
    assert summary.labels[disagreement] == 3
    assert summary.rubric_hash == "rubric-v1" and summary.annotation_artifact_hash
    with pytest.raises(ValueError, match="only disagreements"):
        complete_double_annotation(
            pairs, annotations,
            (Adjudication(disagreement, "c", 3), Adjudication(pairs[0].pair_id, "c", 0)),
            annotator_order=("a", "b"), adjudicator_id="c", rubric_version=1, rubric_hash="rubric-v1",
        )


def test_calibration_requires_authoritative_dev_labels_and_predictions_recompute_probability() -> None:
    manifest, labels = _gold()
    dev = [pair for pair in manifest.pairs if pair.split is GoldSplit.DEV]
    dev_labels = _dev_labels(manifest, labels)
    negative = next(pair for pair in dev if dev_labels.labels[pair.pair_id] < 2)
    positive = next(pair for pair in dev if dev_labels.labels[pair.pair_id] >= 2)
    examples = (
        CalibrationExample(negative.pair_id, GoldSplit.DEV, -3, dev_labels.labels[negative.pair_id]),
        CalibrationExample(positive.pair_id, GoldSplit.DEV, 3, dev_labels.labels[positive.pair_id]),
    )
    calibrator = fit_path_calibrator(CalibrationPath.RERANKER, examples, manifest, dev_labels, "model-lock")
    assert calibrator.dev_label_hash == dev_labels.hash()
    wrong = replace(examples[0], gold_label=3)
    with pytest.raises(ValueError, match="do not match"):
        fit_path_calibrator(CalibrationPath.RERANKER, (wrong, examples[1]), manifest, dev_labels, "model-lock")

    artifacts = _candidate_artifacts(manifest, labels, "candidate")
    evaluator = CandidateEvaluator(manifest, dev_labels, artifacts.calibrators, artifacts.thresholds)
    predictions = _ideal_predictions(dev, manifest, labels, artifacts)
    assert evaluator.evaluate(predictions).trace.candidate_id == "candidate"
    forged = list(predictions)
    forged[0] = replace(forged[0], probability=0.99)
    with pytest.raises(ValueError, match="does not match"):
        evaluator.evaluate(forged)
    wrong_decision = list(predictions)
    negative_index = next(index for index, pair in enumerate(dev) if labels.labels[pair.pair_id] < 2)
    wrong_decision[negative_index] = replace(wrong_decision[negative_index], decision=Stage2Decision.RELEVANT)
    with pytest.raises(ValueError, match="high threshold"):
        evaluator.evaluate(wrong_decision)


def test_metrics_preserve_needs_review_semantics_and_zero_positive_slices() -> None:
    manifest, labels = _gold()
    artifacts = _candidate_artifacts(manifest, labels, "metrics")
    pairs = list(manifest.pairs[:4])
    private = GoldLabelStore(
        {pairs[0].pair_id: 3, pairs[1].pair_id: 2, pairs[2].pair_id: 0, pairs[3].pair_id: 0}, "metric-labels"
    )
    predictions = (
        _prediction(pairs[0], manifest, artifacts, Stage2Decision.RELEVANT),
        _prediction(pairs[1], manifest, artifacts, Stage2Decision.NEEDS_REVIEW, review_reason=ReviewReason.UNCERTAIN),
        _prediction(pairs[2], manifest, artifacts, Stage2Decision.RELEVANT),
        _prediction(pairs[3], manifest, artifacts, Stage2Decision.IRRELEVANT),
    )
    metrics = measure_predictions(pairs, private, predictions)
    assert metrics.automatic_precision == 0.5
    assert metrics.automatic_recall == 0.5
    assert metrics.retention_recall == 1
    assert metrics.automatic_coverage == 0.75
    assert metrics.automatic_recall_interval is not None

    negatives = GoldLabelStore({pair.pair_id: 0 for pair in pairs}, "negative-labels")
    rejected = tuple(_prediction(pair, manifest, artifacts, Stage2Decision.IRRELEVANT) for pair in pairs)
    zero = measure_predictions(pairs, negatives, rejected)
    assert zero.positive_count == 0 and zero.retention_interval is None and zero.automatic_recall_interval is None


def test_promotion_evaluates_all_candidates_once_and_atomically_persists_regression_marker(tmp_path) -> None:
    manifest, labels = _gold()
    hidden = tuple(pair for pair in manifest.pairs if pair.split is not GoldSplit.DEV)
    incumbent = _candidate_artifacts(manifest, labels, "incumbent")
    challenger = _candidate_artifacts(manifest, labels, "challenger")
    evaluator = PromotionEvaluator(
        manifest, labels, {"incumbent": incumbent, "challenger": challenger}, "evaluator-a", tmp_path
    )
    batch = evaluator.evaluate_candidates(
        (_submission("incumbent", hidden, manifest, labels, incumbent),
         _submission("challenger", hidden, manifest, labels, challenger)),
        incumbent_candidate_id="incumbent", evaluation_run_id="blind-run", bootstrap_iterations=100,
    )
    assert set(batch.candidates) == {"incumbent", "challenger"}
    assert set(batch.comparisons) == {
        ("challenger", GoldSplit.HIDDEN_HARD), ("challenger", GoldSplit.HIDDEN_REAL)
    }
    assert batch.marker.regression_splits == {GoldSplit.HIDDEN_HARD, GoldSplit.HIDDEN_REAL}
    assert load_promotion_marker(evaluator.marker_path).consumed
    assert promotion_gate(batch.candidates["incumbent"]).passed
    incumbent_result = batch.candidates["incumbent"]
    assert incumbent_result.evaluations[GoldSplit.HIDDEN_REAL].ece_interval is not None
    forged_evaluations = dict(incumbent_result.evaluations)
    forged_evaluations[GoldSplit.HIDDEN_REAL] = replace(
        forged_evaluations[GoldSplit.HIDDEN_REAL], size=149
    )
    with pytest.raises(ValueError, match="exactly 150"):
        promotion_gate(replace(incumbent_result, evaluations=forged_evaluations))

    rebuilt = PromotionEvaluator(
        manifest, labels, {"incumbent": incumbent}, "different-evaluator-id", tmp_path
    )
    with pytest.raises(ValueError, match="consumed"):
        rebuilt.evaluate_candidates(
            (_submission("incumbent", hidden, manifest, labels, incumbent),),
            incumbent_candidate_id="incumbent", evaluation_run_id="replay", bootstrap_iterations=100,
        )


def test_reveal_before_evaluation_persists_regression_and_blocks_fresh_process(tmp_path) -> None:
    manifest, labels = _gold()
    artifacts = _candidate_artifacts(manifest, labels, "candidate")
    evaluator = PromotionEvaluator(manifest, labels, {"candidate": artifacts}, "eval", tmp_path)
    revealed = evaluator.reveal_for_regression(GoldSplit.HIDDEN_HARD)
    assert len(revealed.labels) == 150
    marker = load_promotion_marker(evaluator.marker_path)
    assert marker.regression_splits == {GoldSplit.HIDDEN_HARD}
    rebuilt = PromotionEvaluator(manifest, labels, {"candidate": artifacts}, "new", tmp_path)
    hidden = tuple(pair for pair in manifest.pairs if pair.split is not GoldSplit.DEV)
    with pytest.raises(ValueError, match="regression-only"):
        rebuilt.evaluate_candidates(
            (_submission("candidate", hidden, manifest, labels, artifacts),),
            incumbent_candidate_id="candidate", evaluation_run_id="forbidden", bootstrap_iterations=100,
        )


def test_determinism_binds_hidden_universe_and_all_model_config_provenance() -> None:
    manifest, labels = _gold()
    hidden = tuple(pair for pair in manifest.pairs if pair.split is not GoldSplit.DEV)
    artifacts = _candidate_artifacts(manifest, labels, "stable")
    runs = tuple(_ideal_predictions(hidden, manifest, labels, artifacts, run=index) for index in range(3))
    expected = [pair.pair_id for pair in hidden]
    assert determinism_gate(runs, expected).passed
    with pytest.raises(ValueError, match="hidden pair universe"):
        determinism_gate(tuple(run[:-1] for run in runs), expected)
    other = _candidate_artifacts(manifest, labels, "other")
    mixed = (runs[0], runs[1], _ideal_predictions(hidden, manifest, labels, other, run=2))
    with pytest.raises(ValueError, match="identical candidate"):
        determinism_gate(mixed, expected)


def test_language_and_core_recall_gates_start_at_30_positive_examples(tmp_path) -> None:
    manifest, base_labels = _gold()
    hidden = tuple(pair for pair in manifest.pairs if pair.split is not GoldSplit.DEV)
    hard = [pair for pair in hidden if pair.split is GoldSplit.HIDDEN_HARD]

    def run_case(count: int, root_name: str):
        labels = dict(base_labels.labels)
        eligible_en = [
            pair for pair in hard
            if pair.language == "en" and pair.pair_id not in base_labels.hard_negative_pair_ids
        ]
        for pair in eligible_en:
            labels[pair.pair_id] = 0
        for pair in eligible_en[:count]:
            labels[pair.pair_id] = 2
        eligible_core = [
            pair for pair in hard
            if pair.language == "zh" and pair.pair_id not in base_labels.hard_negative_pair_ids
        ]
        eligible_core.sort(key=lambda pair: pair.pair_id not in base_labels.hard_positive_pair_ids)
        for pair in eligible_core:
            labels[pair.pair_id] = 2
        for pair in eligible_core[:count]:
            labels[pair.pair_id] = 3
        private = GoldLabelStore(
            labels, f"threshold-labels-{count}", base_labels.hard_negative_pair_ids, base_labels.hard_positive_pair_ids
        )
        artifacts = _candidate_artifacts(manifest, private, f"candidate-{count}")
        rejected_ids = {pair.pair_id for pair in eligible_en[:count]} | {eligible_core[0].pair_id}
        runs = []
        for run in range(3):
            predictions = []
            for pair in hidden:
                decision = (
                    Stage2Decision.IRRELEVANT if pair.pair_id in rejected_ids
                    else Stage2Decision.RELEVANT if private.labels[pair.pair_id] >= 2
                    else Stage2Decision.IRRELEVANT
                )
                predictions.append(_prediction(pair, manifest, artifacts, decision, run=run))
            runs.append(tuple(predictions))
        evaluator = PromotionEvaluator(
            manifest, private, {artifacts.candidate_id: artifacts}, f"eval-{count}", tmp_path / root_name
        )
        batch = evaluator.evaluate_candidates(
            (PromotionSubmission(artifacts.candidate_id, tuple(runs)),),
            incumbent_candidate_id=artifacts.candidate_id, evaluation_run_id=f"run-{count}", bootstrap_iterations=100,
        )
        return promotion_gate(batch.candidates[artifacts.candidate_id])

    below = run_case(29, "below")
    assert not any("hard/en" in failure or "hard core" in failure for failure in below.failures)
    at_threshold = run_case(30, "at-threshold")
    assert any("hard/en" in failure for failure in at_threshold.failures)
    assert any("hard core" in failure for failure in at_threshold.failures)


def _performance_manifest() -> PerformanceRoutingManifest:
    cases = tuple(PerformanceCase(f"perf-{index}", 100 + index % 20, index < 100) for index in range(1_000))
    return PerformanceRoutingManifest(
        1, "perf-corpus", "stage2-config", ("reranker-lock", "qwen-lock"),
        ("reranker-threshold", "qwen-threshold"), 256, cases,
        frozenset(case.pair_id for case in cases[:150]), frozenset(case.pair_id for case in cases[:300]),
    )


def _environment() -> BenchmarkEnvironment:
    return BenchmarkEnvironment(
        "Apple Silicon M4 Max", 36, "15.6", "0.5.7", "0.32.0", "AC", "idle",
        {"rerank_batch": 32, "qwen_concurrency": 4}, {"reranker-lock": 1, "qwen-lock": 1},
    )


def _performance_record(manifest: PerformanceRoutingManifest, scenario: str, run_id: str) -> PerformanceRunRecord:
    ids = tuple(case.pair_id for case in manifest.cases)
    qwen = manifest.normal_qwen_ids if scenario == "normal" else manifest.stress_qwen_ids
    return PerformanceRunRecord(
        record_version=2,
        scenario=scenario, run_id=run_id, manifest_hash=manifest.hash(), stage2_config_hash=manifest.stage2_config_hash,
        model_lock_hashes=manifest.model_lock_hashes, duration_seconds=800 if scenario == "normal" else 1_200,
        p50_seconds=0.5, p95_seconds=1.5, peak_memory_gb=24, request_count=1_000, failed_request_count=0,
        service_request_count=1_000, service_failed_request_count=0,
        resume_verified=True, resume_model_call_count=0, resumed_pair_count=1_000,
        completed_pair_ids=ids, needs_review_pair_ids=(), failed_request_pair_ids=(), qwen_pair_ids=tuple(sorted(qwen)),
        environment=_environment(), executed_components=manifest.pipeline_components, sqlite_commit_count=1_000, warmed=True,
    )


def test_benchmark_is_exact_1000_ten_percent_missing_and_failure_rate_is_derived() -> None:
    manifest = _performance_manifest()
    records = tuple(
        _performance_record(manifest, scenario, f"{scenario}-{index}")
        for scenario in ("normal", "stress") for index in range(3)
    )
    assert performance_gate(manifest, records).passed
    assert records[0].request_failure_rate == 0
    assert records[0].service_request_failure_rate == 0
    with pytest.raises(ValueError, match="failed request count"):
        replace(records[0], failed_request_count=1)
    with pytest.raises(ValueError, match="mutually exclusive"):
        replace(records[0], needs_review_pair_ids=(records[0].completed_pair_ids[0],))
    with pytest.raises(ValueError, match="finite"):
        replace(records[0], peak_memory_gb=nan)
    assert not performance_gate(manifest, (replace(records[0], warmed=False), *records[1:])).passed
    service_failure = replace(
        records[0], service_failed_request_count=5
    )
    service_gate = performance_gate(manifest, (service_failure, *records[1:]))
    assert any("service request failure rate" in failure for failure in service_gate.failures)
    below_service_gate = replace(records[0], service_failed_request_count=4)
    assert performance_gate(manifest, (below_service_gate, *records[1:])).passed
    resume_failure = replace(
        records[0], resume_verified=False, resume_model_call_count=1
    )
    resume_gate = performance_gate(manifest, (resume_failure, *records[1:]))
    assert any("zero-call SQLite resume" in failure for failure in resume_gate.failures)
    with pytest.raises(ValueError, match="record_version 2"):
        replace(records[0], record_version=1)

    bad_cases = tuple(replace(case, abstract_missing=False) if index == 99 else case for index, case in enumerate(manifest.cases))
    with pytest.raises(ValueError, match="10%"):
        replace(manifest, cases=bad_cases)


def test_soak_is_a_separate_10000_case_contract_and_all_failures_fail_open() -> None:
    cases = tuple(PerformanceCase(f"soak-{index}", 80) for index in range(10_000))
    manifest = SoakManifest(
        1, "soak-corpus", "stage2-config", ("reranker-lock", "qwen-lock"),
        ("reranker-threshold", "qwen-threshold"), 256, cases,
    )
    review = (cases[0].pair_id,)
    record = SoakRunRecord(
        record_version=2,
        run_id="soak-run",
        manifest_hash=manifest.hash(),
        stage2_config_hash=manifest.stage2_config_hash,
        model_lock_hashes=manifest.model_lock_hashes,
        duration_seconds=3_000,
        peak_memory_gb=25,
        request_count=10_000,
        failed_request_count=1,
        service_request_count=10_000,
        service_failed_request_count=1,
        resume_verified=True,
        resume_model_call_count=0,
        resumed_pair_count=10_000,
        completed_pair_ids=tuple(case.pair_id for case in cases[1:]),
        needs_review_pair_ids=review,
        failed_request_pair_ids=review,
        environment=_environment(),
        executed_components=("rules", "reranker", "qwen", "schema_validation", "sqlite_commit"),
        sqlite_commit_count=10_000,
        warmed=True,
    )
    assert soak_gate(manifest, record).passed
    assert not soak_gate(
        manifest, replace(record, service_failed_request_count=50)
    ).passed
    bad = replace(record, failed_request_pair_ids=(cases[1].pair_id,))
    assert any("needs_review" in failure for failure in soak_gate(manifest, bad).failures)


def _structured_manifest() -> StructuredReplayManifest:
    return StructuredReplayManifest(
        1, tuple(f"replay-{index}" for index in range(1_000)), "replay-corpus", "config", "qwen-lock", "prompt", "schema"
    )


def _structured_records(manifest: StructuredReplayManifest, error_count: int) -> tuple[StructuredReplayRecord, ...]:
    records = []
    for index, pair_id in enumerate(manifest.pair_ids):
        errored = index < error_count
        records.append(
            StructuredReplayRecord(
                pair_id, manifest.hash(), ReplayError.SCHEMA if errored else ReplayError.NONE, pair_id,
                False, False, 1 if errored else 0, 1 if errored else 0, ReplayError.NONE if errored else None,
                True, pair_id, False, False, Stage2Decision.RELEVANT,
            )
        )
    return tuple(records)


def test_structured_replay_binds_schema_model_and_audits_retry_output() -> None:
    manifest = _structured_manifest()
    records = _structured_records(manifest, 5)
    result = structured_replay_gate(manifest, records)
    assert result.gate.passed and result.first_valid_rate == 0.995 and result.schema_errors == 5
    assert result.retry_error_counts == {ReplayError.NONE: 5}
    leaked = replace(records[0], final_think_tag_leak=True)
    assert any("think-tag" in failure for failure in structured_replay_gate(manifest, (leaked, *records[1:])).gate.failures)
    assert not structured_replay_gate(manifest, _structured_records(manifest, 6)).gate.passed
    invalid = replace(records[0], final_valid=False, final_returned_pair_id=None, final_decision=Stage2Decision.NEEDS_REVIEW)
    assert structured_replay_gate(manifest, (invalid, *records[1:])).gate.passed


def test_rationale_manifest_freezes_balanced_strata_language_and_artifact_provenance() -> None:
    cases = tuple(
        RationaleAuditCase(
            f"audit-{index}", RationaleStratum.RELEVANT if index < 50 else RationaleStratum.BOUNDARY,
            "en" if index % 2 else "zh", f"rationale-{index}",
        )
        for index in range(100)
    )
    manifest = RationaleAuditManifest(1, cases, "corpus", "model-lock", "evidence-rubric", "fabrication-rubric")
    records = tuple(
        RationaleAuditRecord(case.pair_id, manifest.hash(), index >= 5, index == 0)
        for index, case in enumerate(cases)
    )
    assert rationale_audit_gate(manifest, records).passed
    changed = list(records)
    changed[5] = replace(changed[5], evidence_supported=False)
    changed[1] = replace(changed[1], severe_fabrication=True)
    failures = rationale_audit_gate(manifest, changed).failures
    assert any("support" in failure for failure in failures) and any("fabrication" in failure for failure in failures)


def _parity_manifest() -> ParityManifest:
    pair_ids = tuple(f"parity-{index}" for index in range(10_000))
    return ParityManifest(
        1, pair_ids, "parity-corpus", "tokenizer", "preprocess", "fp32-lock", "quantized-lock",
        "oracle-thresholds", "candidate-thresholds", "dev-manifest", "selector-v1", 100, 9_900, 100, 9_900,
        frozenset(pair_ids[:200]), frozenset(pair_ids[-200:]), "abs(low-score)<=100", "abs(high-score)<=100",
    )


def test_kendall_tau_b_ties_and_dual_threshold_parity_are_bound_to_manifest() -> None:
    assert kendall_tau_b((1, 1, 2, 3), (1, 2, 2, 3)) == pytest.approx(0.8)
    manifest = _parity_manifest()
    scores = tuple(
        ParityScore(pair_id, manifest.hash(), float(index), float(index))
        for index, pair_id in enumerate(manifest.pair_ids)
    )
    result = parity_gate(manifest, scores)
    assert result.gate.passed and result.kendall_tau_b == 1
    assert result.low_threshold_denominator == result.high_threshold_denominator == 200

    changed = list(scores)
    for index in (0, 1):
        changed[index] = replace(changed[index], candidate_score=1_000)
    failed = parity_gate(manifest, changed)
    assert failed.low_threshold_agreement == 0.99
    assert any("low-threshold" in failure for failure in failed.gate.failures)


def _release(candidate_id: str, manifest_hash: str, throughput: tuple[float, float, float], passed: bool = True):
    gate = GateResult(passed, () if passed else ("failed",))
    artifacts = {
        name: (f"{candidate_id}-{name}-artifact", gate)
        for name in ("promotion", "structured_replay", "rationale", "parity", "benchmark", "soak")
    }
    return phase3_release_gate(
        candidate_id=candidate_id, evaluation_manifest_hash=manifest_hash, artifacts=artifacts,
        throughput_runs=throughput,
    )


def test_paired_bootstrap_is_split_bound_reports_wilson_and_winner_requires_all_gates() -> None:
    manifest, labels = _gold()
    incumbent_artifacts = _candidate_artifacts(manifest, labels, "incumbent")
    challenger_artifacts = _candidate_artifacts(manifest, labels, "challenger")
    comparisons = {}
    for split in (GoldSplit.HIDDEN_HARD, GoldSplit.HIDDEN_REAL):
        pairs = [pair for pair in manifest.pairs if pair.split is split]
        incumbent = _ideal_predictions(pairs, manifest, labels, incumbent_artifacts)
        challenger = _ideal_predictions(pairs, manifest, labels, challenger_artifacts)
        comparisons[split] = paired_bootstrap_comparison(
            pairs, labels, incumbent, challenger, iterations=200, seed=7
        )
        assert comparisons[split].in_tie_band
        assert comparisons[split].incumbent_retention_wilson is not None
    incumbent_release = _release("incumbent", manifest.hash(), (100, 100, 100))
    challenger_release = _release("challenger", manifest.hash(), (120, 120, 120))
    assert winner_gate(incumbent_release, challenger_release, comparisons).replace_incumbent
    assert not winner_gate(
        incumbent_release, _release("challenger", manifest.hash(), (200, 200, 200), passed=False), comparisons
    ).replace_incumbent

    mixed_pairs = [manifest.pairs[0], next(pair for pair in manifest.pairs if pair.split is GoldSplit.HIDDEN_HARD)]
    with pytest.raises(ValueError, match="cannot mix"):
        paired_bootstrap_comparison(
            mixed_pairs, labels,
            _ideal_predictions(mixed_pairs, manifest, labels, incumbent_artifacts),
            _ideal_predictions(mixed_pairs, manifest, labels, challenger_artifacts), iterations=100,
        )


def test_winner_rejects_point_improvement_when_paired_interval_crosses_zero() -> None:
    manifest, labels = _gold()
    incumbent_artifacts = _candidate_artifacts(manifest, labels, "incumbent")
    challenger_artifacts = _candidate_artifacts(manifest, labels, "challenger")
    comparisons = {}
    for split in (GoldSplit.HIDDEN_HARD, GoldSplit.HIDDEN_REAL):
        pairs = [pair for pair in manifest.pairs if pair.split is split]
        base = paired_bootstrap_comparison(
            pairs, labels, _ideal_predictions(pairs, manifest, labels, incumbent_artifacts),
            _ideal_predictions(pairs, manifest, labels, challenger_artifacts), iterations=100,
        )
        comparisons[split] = replace(
            base, retention_delta=BootstrapInterval(0.02, -0.01, 0.04),
            positive_f1_delta=BootstrapInterval(0, -0.01, 0.01),
        )
    assert not winner_gate(
        _release("incumbent", manifest.hash(), (100, 100, 100)),
        _release("challenger", manifest.hash(), (100, 100, 100)), comparisons,
    ).replace_incumbent
