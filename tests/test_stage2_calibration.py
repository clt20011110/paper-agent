from __future__ import annotations

from dataclasses import replace

import pytest

from paper_agent.stage2_calibration import (
    RecallFirstThresholdPolicy,
    build_stage2_calibration_bundle,
    freeze_dev_scores,
    select_recall_first_threshold,
)
from paper_agent.stage2_backends import ThresholdArtifact as LegacyThresholdArtifact
from paper_agent.stage2_evaluation import (
    CalibrationExample,
    CalibrationPath,
    GoldLabelStore,
    GoldManifest,
    GoldPair,
    GoldSplit,
    fit_path_calibrator,
)
from paper_agent.stage2_pipeline import Stage2Profile


def _manifest_and_dev_labels() -> tuple[GoldManifest, GoldLabelStore]:
    pairs: list[GoldPair] = []
    for split, size in ((GoldSplit.DEV, 300), (GoldSplit.HIDDEN_HARD, 150), (GoldSplit.HIDDEN_REAL, 150)):
        for index in range(size):
            pairs.append(GoldPair(
                f"paper-{split.value}-{index}", f"topic-{index % 6}", "en" if index % 2 else "zh",
                "crawler-v1",
                0.5 if split is GoldSplit.HIDDEN_REAL else None,
                f"family-{split.value}-{index}", "corpus-v1", split,
                abstract_incomplete=split is not GoldSplit.HIDDEN_REAL and index < size // 10,
                sampled_from_natural_distribution=split is GoldSplit.HIDDEN_REAL,
                cross_language_match=index == 0,
            ))
    manifest = GoldManifest(1, "corpus-v1", tuple(pairs), ("en", "zh"))
    dev_pairs = [pair for pair in pairs if pair.split is GoldSplit.DEV]
    labels = {pair.pair_id: 3 if index % 3 else 0 for index, pair in enumerate(dev_pairs)}
    return manifest, GoldLabelStore(labels, "dev-label-artifact-v1")


def _scores(manifest: GoldManifest, labels: GoldLabelStore) -> dict[CalibrationPath, dict[str, float]]:
    dev = [pair for pair in manifest.pairs if pair.split is GoldSplit.DEV]
    return {
        CalibrationPath.RERANKER: {
            pair.pair_id: 4.0 if labels.labels[pair.pair_id] >= 2 else -4.0
            for pair in dev
        },
        CalibrationPath.QWEN: {
            pair.pair_id: 3.0 if labels.labels[pair.pair_id] >= 2 else -3.0
            for pair in dev
        },
    }


def _profile_kwargs() -> dict[str, object]:
    return {
        "query": "topic",
        "query_version": "topic-v1",
        "reranker_model_id": "reranker-model",
        "reranker_revision": "reranker-revision",
        "adjudicator_model_id": "qwen-model",
        "adjudicator_revision": "qwen-revision",
        "screening_scope_hash": "0" * 64,
        "reranker_lock_hash": "reranker-lock",
        "adjudicator_lock_hash": "qwen-lock",
    }


def test_freeze_and_build_produce_both_stage2_profile_calibrations_from_dev_only() -> None:
    manifest, dev_labels = _manifest_and_dev_labels()
    provisional = Stage2Profile(
        thresholds=LegacyThresholdArtifact("fixture", "reranker-lock", "raw_reranker_score", -1.0, 1.0),
        **_profile_kwargs(),
    )
    artifact = freeze_dev_scores(
        manifest, dev_labels, _scores(manifest, dev_labels),
        {CalibrationPath.RERANKER: "reranker-lock", CalibrationPath.QWEN: "qwen-lock"},
        provisional.base_runtime_config_hash,
    )

    bundle = build_stage2_calibration_bundle(artifact, manifest, dev_labels)

    assert bundle.reranker_calibration.calibrator.path is CalibrationPath.RERANKER
    assert bundle.adjudicator_calibration.calibrator.path is CalibrationPath.QWEN
    for path, selection in bundle.selections.items():
        assert selection.target == "P(gold_label >= 2)"
        assert selection.positive_retention == 1.0
        assert selection.relevant_recall >= 0.98
        assert selection.needs_review_rate <= 0.50
        assert selection.threshold.path is path
        assert selection.threshold.stage2_config_hash == provisional.base_runtime_config_hash
        assert selection.threshold.calibrator_hash == bundle.calibrations[path].calibrator.hash()

    profile = Stage2Profile(
        thresholds=None,
        reranker_calibration=bundle.reranker_calibration,
        adjudicator_calibration=bundle.adjudicator_calibration,
        **_profile_kwargs(),
    )
    assert profile.production_calibrated


def test_freeze_rejects_missing_path_and_non_dev_label_input() -> None:
    manifest, dev_labels = _manifest_and_dev_labels()
    scores = _scores(manifest, dev_labels)
    locks = {CalibrationPath.RERANKER: "reranker-lock", CalibrationPath.QWEN: "qwen-lock"}
    with pytest.raises(ValueError, match="reranker and qwen"):
        freeze_dev_scores(manifest, dev_labels, {CalibrationPath.RERANKER: scores[CalibrationPath.RERANKER]}, locks, "config")

    hidden = next(pair for pair in manifest.pairs if pair.split is GoldSplit.HIDDEN_HARD)
    contaminated = GoldLabelStore({**dev_labels.labels, hidden.pair_id: 2}, "contaminated-labels")
    with pytest.raises(ValueError, match="only the authoritative DEV labels"):
        freeze_dev_scores(manifest, contaminated, scores, locks, "config")


def test_threshold_selection_keeps_positives_before_enforcing_review_budget() -> None:
    manifest, dev_labels = _manifest_and_dev_labels()
    scores = _scores(manifest, dev_labels)[CalibrationPath.RERANKER]
    examples = tuple(
        CalibrationExample(pair_id, GoldSplit.DEV, score, dev_labels.labels[pair_id])
        for pair_id, score in sorted(scores.items())
    )
    calibrator = fit_path_calibrator(CalibrationPath.RERANKER, examples, manifest, dev_labels, "reranker-lock")

    selection = select_recall_first_threshold(
        calibrator, examples, "config", RecallFirstThresholdPolicy(1.0, 0.98, 0.01)
    )

    assert selection.positive_retention == 1.0
    assert selection.relevant_recall >= 0.98
    assert selection.needs_review_rate == 0.0
    assert selection.threshold.low < selection.threshold.high

    ambiguous = tuple(
        replace(item, raw_score=1.0 + index / len(examples) if item.gold_label >= 2 else -1.0)
        for index, item in enumerate(examples)
    )
    ambiguous_calibrator = fit_path_calibrator(CalibrationPath.RERANKER, ambiguous, manifest, dev_labels, "reranker-lock")
    with pytest.raises(ValueError, match="needs_review budget"):
        select_recall_first_threshold(
            ambiguous_calibrator, ambiguous, "config", RecallFirstThresholdPolicy(1.0, 0.98, 0.01)
        )
