"""Frozen DEV-only calibration artifacts for the released Stage 2 paths."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isfinite, nextafter
from types import MappingProxyType
from typing import Mapping

from .canonical import content_hash
from .stage2_evaluation import (
    CalibrationExample,
    CalibrationPath,
    GoldLabelStore,
    GoldManifest,
    GoldSplit,
    PathCalibrator,
    ThresholdArtifact,
    fit_path_calibrator,
)
from .stage2_pipeline import PathCalibration


_PATHS = frozenset({CalibrationPath.RERANKER, CalibrationPath.QWEN})
_TARGET = "P(gold_label >= 2)"


@dataclass(frozen=True, slots=True)
class FrozenDevScoreArtifact:
    """The complete labelled DEV score set for both released model paths."""

    version: int
    scores: Mapping[CalibrationPath, tuple[CalibrationExample, ...]]
    model_lock_hashes: Mapping[CalibrationPath, str]
    stage2_config_hash: str
    gold_manifest_hash: str
    dev_manifest_hash: str
    dev_label_hash: str

    def __post_init__(self) -> None:
        scores = {CalibrationPath(path): tuple(values) for path, values in self.scores.items()}
        model_locks = {CalibrationPath(path): value for path, value in self.model_lock_hashes.items()}
        object.__setattr__(self, "scores", MappingProxyType(scores))
        object.__setattr__(self, "model_lock_hashes", MappingProxyType(model_locks))
        if self.version != 1 or set(scores) != _PATHS or set(model_locks) != _PATHS:
            raise ValueError("frozen DEV scores require exactly reranker and qwen paths")
        if not all((self.stage2_config_hash, self.gold_manifest_hash, self.dev_manifest_hash, self.dev_label_hash)):
            raise ValueError("frozen DEV scores require config, manifest, and label provenance")
        if not all(model_locks.values()):
            raise ValueError("frozen DEV scores require both model locks")
        for values in scores.values():
            pair_ids = [item.pair_id for item in values]
            if not values or len(pair_ids) != len(set(pair_ids)) or any(item.split is not GoldSplit.DEV for item in values):
                raise ValueError("frozen scores must contain unique DEV examples only")

    def document(self) -> dict[str, object]:
        return {
            "version": self.version,
            "scores": {
                path.value: [
                    {"pair_id": item.pair_id, "split": item.split.value, "raw_score": item.raw_score, "gold_label": item.gold_label}
                    for item in self.scores[path]
                ]
                for path in sorted(self.scores, key=str)
            },
            "model_lock_hashes": {path.value: self.model_lock_hashes[path] for path in sorted(self.model_lock_hashes, key=str)},
            "stage2_config_hash": self.stage2_config_hash,
            "gold_manifest_hash": self.gold_manifest_hash,
            "dev_manifest_hash": self.dev_manifest_hash,
            "dev_label_hash": self.dev_label_hash,
            "calibration_target": _TARGET,
        }

    def hash(self) -> str:
        return content_hash(self.document())


def freeze_dev_scores(
    manifest: GoldManifest,
    dev_labels: GoldLabelStore,
    raw_scores: Mapping[CalibrationPath, Mapping[str, float]],
    model_lock_hashes: Mapping[CalibrationPath, str],
    stage2_config_hash: str,
) -> FrozenDevScoreArtifact:
    """Bind both score paths to the full, authoritative DEV label set."""

    manifest.validate_sampling_structure()
    dev_ids = {pair.pair_id for pair in manifest.pairs if pair.split is GoldSplit.DEV}
    if set(dev_labels.labels) != dev_ids:
        raise ValueError("DEV calibration must receive only the authoritative DEV labels")
    paths = {CalibrationPath(path): values for path, values in raw_scores.items()}
    if set(paths) != _PATHS:
        raise ValueError("DEV score input requires reranker and qwen scores")
    examples: dict[CalibrationPath, tuple[CalibrationExample, ...]] = {}
    for path, values in paths.items():
        if set(values) != dev_ids or any(not isfinite(score) for score in values.values()):
            raise ValueError("each path must score every DEV pair exactly once")
        examples[path] = tuple(
            CalibrationExample(pair_id, GoldSplit.DEV, values[pair_id], dev_labels.labels[pair_id])
            for pair_id in sorted(dev_ids)
        )
    artifact = FrozenDevScoreArtifact(
        1, examples, model_lock_hashes, stage2_config_hash,
        manifest.hash(), manifest.dev_hash(), dev_labels.hash(),
    )
    return artifact


@dataclass(frozen=True, slots=True)
class RecallFirstThresholdPolicy:
    """DEV policy for P(label >= 2), retaining positives before controlling review load."""

    positive_retention_target: float = 1.0
    relevant_recall_target: float = 0.98
    max_needs_review_rate: float = 0.50

    def __post_init__(self) -> None:
        targets = (self.positive_retention_target, self.relevant_recall_target)
        if not all(isfinite(value) and 0 < value <= 1 for value in targets):
            raise ValueError("recall-first retention targets must be finite probabilities in (0, 1]")
        if not isfinite(self.max_needs_review_rate) or not 0 <= self.max_needs_review_rate <= 1:
            raise ValueError("needs_review budget must be a finite probability in [0, 1]")
        if self.relevant_recall_target > self.positive_retention_target:
            raise ValueError("relevant recall cannot exceed positive retention")


@dataclass(frozen=True, slots=True)
class ThresholdSelection:
    threshold: ThresholdArtifact
    target: str
    positive_retention: float
    relevant_recall: float
    needs_review_rate: float


def select_recall_first_threshold(
    calibrator: PathCalibrator,
    examples: tuple[CalibrationExample, ...],
    stage2_config_hash: str,
    policy: RecallFirstThresholdPolicy = RecallFirstThresholdPolicy(),
) -> ThresholdSelection:
    """Select the strictest DEV thresholds that satisfy recall-first targets."""

    if not examples or any(item.split is not GoldSplit.DEV for item in examples):
        raise ValueError("threshold selection only accepts DEV examples")
    positives = sorted(calibrator.predict(item.raw_score) for item in examples if item.gold_label >= 2)
    if not positives:
        raise ValueError("threshold selection requires positive DEV labels")
    retain_count = ceil(len(positives) * policy.positive_retention_target)
    recall_count = ceil(len(positives) * policy.relevant_recall_target)
    low = max(0.0, nextafter(positives[len(positives) - retain_count], float("-inf")))
    high = positives[len(positives) - recall_count]
    if low >= high:
        low = max(0.0, nextafter(high, float("-inf")))
    threshold = ThresholdArtifact(
        1, calibrator.path, low, high, calibrator.hash(), calibrator.model_lock_hash,
        calibrator.dev_manifest_hash, calibrator.dev_label_hash, stage2_config_hash,
    )
    probabilities = [(calibrator.predict(item.raw_score), item.gold_label >= 2) for item in examples]
    positive_retention = sum(probability > low for probability, label in probabilities if label) / len(positives)
    relevant_recall = sum(probability >= high for probability, label in probabilities if label) / len(positives)
    needs_review_rate = sum(low < probability < high for probability, _ in probabilities) / len(probabilities)
    if needs_review_rate > policy.max_needs_review_rate:
        raise ValueError("recall-first threshold selection exceeds the needs_review budget")
    return ThresholdSelection(threshold, _TARGET, positive_retention, relevant_recall, needs_review_rate)


@dataclass(frozen=True, slots=True)
class Stage2CalibrationBundle:
    """Both probability calibrations in the shape consumed by ``Stage2Profile``."""

    calibrations: Mapping[CalibrationPath, PathCalibration]
    selections: Mapping[CalibrationPath, ThresholdSelection]

    def __post_init__(self) -> None:
        calibrations = {CalibrationPath(path): value for path, value in self.calibrations.items()}
        selections = {CalibrationPath(path): value for path, value in self.selections.items()}
        object.__setattr__(self, "calibrations", MappingProxyType(calibrations))
        object.__setattr__(self, "selections", MappingProxyType(selections))
        if set(calibrations) != _PATHS or set(selections) != _PATHS:
            raise ValueError("Stage 2 calibration bundles require both model paths")

    @property
    def reranker_calibration(self) -> PathCalibration:
        return self.calibrations[CalibrationPath.RERANKER]

    @property
    def adjudicator_calibration(self) -> PathCalibration:
        return self.calibrations[CalibrationPath.QWEN]


def build_stage2_calibration_bundle(
    artifact: FrozenDevScoreArtifact,
    manifest: GoldManifest,
    dev_labels: GoldLabelStore,
    policy: RecallFirstThresholdPolicy = RecallFirstThresholdPolicy(),
) -> Stage2CalibrationBundle:
    """Fit frozen reranker/Qwen calibrators and probability thresholds from DEV only."""

    if (
        artifact.gold_manifest_hash != manifest.hash()
        or artifact.dev_manifest_hash != manifest.dev_hash()
        or artifact.dev_label_hash != dev_labels.hash()
    ):
        raise ValueError("frozen DEV score provenance does not match this calibration input")
    calibrations: dict[CalibrationPath, PathCalibration] = {}
    selections: dict[CalibrationPath, ThresholdSelection] = {}
    for path in (CalibrationPath.RERANKER, CalibrationPath.QWEN):
        calibrator = fit_path_calibrator(
            path, artifact.scores[path], manifest, dev_labels, artifact.model_lock_hashes[path]
        )
        selection = select_recall_first_threshold(
            calibrator, artifact.scores[path], artifact.stage2_config_hash, policy
        )
        calibrations[path] = PathCalibration(calibrator, selection.threshold)
        selections[path] = selection
    return Stage2CalibrationBundle(calibrations, selections)
