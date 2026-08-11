"""Auditable Stage 2 evaluation contracts and statistics.

This module does not contain a gold corpus, human annotations, model outputs, or
benchmark measurements.  It only validates artifacts supplied by an evaluation
run and computes the frozen gates in ``task.md`` sections 5.4 and 5.5.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
import json
from math import exp, isfinite, sqrt
from pathlib import Path
from random import Random
from statistics import median
from types import MappingProxyType
from typing import Mapping, Sequence

from .stage2_hidden_attestation import hidden_promotion_gate_policy_document


def _hash(document: object) -> str:
    encoded = json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()


def make_pair_id(topic: str, paper_id: str) -> str:
    """Return the stable identifier of the exact ``(topic, paper)`` pair."""

    if not topic or not paper_id:
        raise ValueError("pair_id requires topic and paper_id")
    return f"pair-{_hash([topic, paper_id])}"


def pair_universe_hash(pair_ids: Sequence[str]) -> str:
    """Hash a pair universe with the ordering rule used by Stage 2 evaluation."""

    return _hash(sorted(pair_ids))


class GoldSplit(StrEnum):
    DEV = "dev"
    HIDDEN_HARD = "hidden_hard"
    HIDDEN_REAL = "hidden_real"


class Stage2Decision(StrEnum):
    RELEVANT = "relevant"
    IRRELEVANT = "irrelevant"
    NEEDS_REVIEW = "needs_review"


class ReviewReason(StrEnum):
    UNCERTAIN = "uncertain"
    SCHEMA_ERROR = "schema_error"
    TIMEOUT = "timeout"
    SERVICE_ERROR = "service_error"
    MODEL_CONFLICT = "model_conflict"


class CalibrationPath(StrEnum):
    RERANKER = "reranker"
    QWEN = "qwen"


@dataclass(frozen=True, slots=True)
class GoldPair:
    """Public sampling metadata.  No label-derived difficulty tag lives here."""

    paper_id: str
    topic: str
    language: str
    source: str
    sampling_probability: float
    paper_family: str
    corpus_hash: str
    split: GoldSplit
    abstract_incomplete: bool = False
    sampled_from_natural_distribution: bool = False
    cross_language_match: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.split, GoldSplit):
            object.__setattr__(self, "split", GoldSplit(self.split))
        required = (self.paper_id, self.topic, self.language, self.source, self.paper_family, self.corpus_hash)
        if not all(required):
            raise ValueError("gold pair sampling fields are required")
        if not isfinite(self.sampling_probability) or not 0 < self.sampling_probability <= 1:
            raise ValueError("sampling_probability must be finite and in (0, 1]")
        if (self.split is GoldSplit.HIDDEN_REAL) != self.sampled_from_natural_distribution:
            raise ValueError("hidden_real is exactly the natural-distribution sample")

    @property
    def pair_id(self) -> str:
        return make_pair_id(self.topic, self.paper_id)

    def document(self) -> dict[str, object]:
        return {
            "pair_id": self.pair_id,
            "paper_id": self.paper_id,
            "topic": self.topic,
            "language": self.language,
            "source": self.source,
            "sampling_probability": self.sampling_probability,
            "paper_family": self.paper_family,
            "corpus_hash": self.corpus_hash,
            "split": self.split.value,
            "abstract_incomplete": self.abstract_incomplete,
            "sampled_from_natural_distribution": self.sampled_from_natural_distribution,
            "cross_language_match": self.cross_language_match,
        }


@dataclass(frozen=True, slots=True)
class GoldLabelStore:
    """Private labels and label-derived sampling strata."""

    labels: Mapping[str, int]
    annotation_artifact_hash: str
    hard_negative_pair_ids: frozenset[str] = frozenset()
    hard_positive_pair_ids: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(self, "labels", MappingProxyType(dict(self.labels)))
        object.__setattr__(self, "hard_negative_pair_ids", frozenset(self.hard_negative_pair_ids))
        object.__setattr__(self, "hard_positive_pair_ids", frozenset(self.hard_positive_pair_ids))
        ids = set(self.labels)
        if not self.annotation_artifact_hash or not ids or any(not pair_id or label not in range(4) for pair_id, label in self.labels.items()):
            raise ValueError("gold labels must be a non-empty pair_id -> 0..3 mapping")
        if not self.hard_negative_pair_ids <= ids or not self.hard_positive_pair_ids <= ids:
            raise ValueError("private difficulty strata must reference labelled pairs")
        if self.hard_negative_pair_ids & self.hard_positive_pair_ids:
            raise ValueError("a pair cannot be both a hard negative and hard positive")
        if any(self.labels[pair_id] >= 2 for pair_id in self.hard_negative_pair_ids):
            raise ValueError("hard negatives must have label 0 or 1")
        if any(self.labels[pair_id] != 3 for pair_id in self.hard_positive_pair_ids):
            raise ValueError("hard positives must have label 3")

    def hash(self) -> str:
        return _hash({
            "labels": sorted(self.labels.items()),
            "annotation_artifact_hash": self.annotation_artifact_hash,
            "hard_negative_pair_ids": sorted(self.hard_negative_pair_ids),
            "hard_positive_pair_ids": sorted(self.hard_positive_pair_ids),
        })


@dataclass(frozen=True, slots=True)
class GoldManifest:
    version: int
    corpus_hash: str
    pairs: tuple[GoldPair, ...]
    main_languages: tuple[str, ...] = ("en", "zh")

    def __post_init__(self) -> None:
        object.__setattr__(self, "pairs", tuple(self.pairs))
        object.__setattr__(self, "main_languages", tuple(self.main_languages))
        if self.version != 1 or not self.corpus_hash:
            raise ValueError("gold manifest requires version 1 and corpus_hash")
        if any(pair.corpus_hash != self.corpus_hash for pair in self.pairs):
            raise ValueError("every pair must name the frozen corpus_hash")
        if len(self.main_languages) < 2 or len(set(self.main_languages)) != len(self.main_languages) or not all(self.main_languages):
            raise ValueError("gold manifest must freeze at least two unique main languages")
        pair_ids = [pair.pair_id for pair in self.pairs]
        if len(pair_ids) != len(set(pair_ids)):
            raise ValueError("(topic, paper) pair_ids must be unique")

    def document(self) -> dict[str, object]:
        return {
            "version": self.version, "corpus_hash": self.corpus_hash, "main_languages": list(self.main_languages),
            "pairs": [pair.document() for pair in self.pairs],
        }

    def hash(self) -> str:
        return _hash(self.document())

    def dev_hash(self) -> str:
        return _hash({
            "version": self.version,
            "corpus_hash": self.corpus_hash,
            "split": GoldSplit.DEV.value,
            "main_languages": list(self.main_languages),
            "pairs": [pair.document() for pair in self.pairs if pair.split is GoldSplit.DEV],
        })

    def validate_sampling_structure(self) -> None:
        if len(self.pairs) != 600:
            raise ValueError("gold manifest must contain exactly 600 pairs")
        expected = {GoldSplit.DEV: 300, GoldSplit.HIDDEN_HARD: 150, GoldSplit.HIDDEN_REAL: 150}
        if Counter(pair.split for pair in self.pairs) != expected:
            raise ValueError("gold manifest split sizes must be dev=300, hidden_hard=150, hidden_real=150")
        family_splits: dict[str, GoldSplit] = {}
        for pair in self.pairs:
            previous = family_splits.setdefault(pair.paper_family, pair.split)
            if previous is not pair.split:
                raise ValueError("a paper family cannot cross gold-set splits")
        if not 6 <= len({pair.topic for pair in self.pairs}) <= 8:
            raise ValueError("gold manifest must cover 6..8 topics")
        if not set(self.main_languages) <= {pair.language for pair in self.pairs}:
            raise ValueError("every frozen main language must occur in the gold manifest")
        if not any(pair.cross_language_match for pair in self.pairs):
            raise ValueError("gold manifest must contain cross-language matching pairs")
        real_probabilities = {
            pair.sampling_probability for pair in self.pairs if pair.split is GoldSplit.HIDDEN_REAL
        }
        if len(real_probabilities) != 1:
            raise ValueError("hidden_real must use equal-probability sampling for unweighted operational metrics")
        for split in (GoldSplit.DEV, GoldSplit.HIDDEN_HARD):
            rows = [pair for pair in self.pairs if pair.split is split]
            if sum(pair.abstract_incomplete for pair in rows) < 0.1 * len(rows):
                raise ValueError(f"{split.value} requires at least 10% incomplete abstracts")

    def validate(self, labels: GoldLabelStore) -> None:
        self.validate_sampling_structure()
        ids = {pair.pair_id for pair in self.pairs}
        if set(labels.labels) != ids:
            raise ValueError("private labels must exactly cover the sampling manifest")
        for language in self.main_languages:
            positives = sum(labels.labels[pair.pair_id] >= 2 for pair in self.pairs if pair.language == language)
            if positives < 30:
                raise ValueError(f"main language {language} requires at least 30 positives")
        for split in (GoldSplit.DEV, GoldSplit.HIDDEN_HARD):
            split_ids = {pair.pair_id for pair in self.pairs if pair.split is split}
            if len(labels.hard_negative_pair_ids & split_ids) < 0.2 * len(split_ids):
                raise ValueError(f"{split.value} requires at least 20% private hard negatives")
            if not labels.hard_positive_pair_ids & split_ids:
                raise ValueError(f"{split.value} requires at least one private hard positive")


def write_gold_manifest(path: Path, manifest: GoldManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.document(), ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def load_gold_manifest(path: Path) -> GoldManifest:
    return gold_manifest_from_document(json.loads(path.read_text(encoding="utf-8")))


def gold_manifest_from_document(document: object) -> GoldManifest:
    """Parse the exact public gold-manifest format without exposing labels."""

    if (
        not isinstance(document, dict)
        or set(document) != {"version", "corpus_hash", "main_languages", "pairs"}
        or not isinstance(document.get("pairs"), list)
    ):
        raise ValueError("gold manifest must be an object with pairs")
    pairs: list[GoldPair] = []
    allowed = {
        "pair_id", "paper_id", "topic", "language", "source", "sampling_probability", "paper_family",
        "corpus_hash", "split", "abstract_incomplete", "sampled_from_natural_distribution", "cross_language_match",
    }
    for item in document["pairs"]:
        if not isinstance(item, dict) or set(item) - allowed:
            raise ValueError("public gold manifest contains unsupported or private fields")
        pair_id = item.get("pair_id")
        values = {key: value for key, value in item.items() if key != "pair_id"}
        pair = GoldPair(**values)
        if pair_id != pair.pair_id:
            raise ValueError("stored pair_id does not match (topic, paper_id)")
        pairs.append(pair)
    return GoldManifest(document["version"], document["corpus_hash"], tuple(pairs), tuple(document["main_languages"]))


@dataclass(frozen=True, slots=True)
class Annotation:
    pair_id: str
    annotator_id: str
    label: int

    def __post_init__(self) -> None:
        if not self.pair_id or not self.annotator_id or self.label not in range(4):
            raise ValueError("annotation requires pair_id, annotator, and label 0..3")


@dataclass(frozen=True, slots=True)
class Adjudication:
    pair_id: str
    adjudicator_id: str
    label: int

    def __post_init__(self) -> None:
        if not self.pair_id or not self.adjudicator_id or self.label not in range(4):
            raise ValueError("adjudication requires pair_id, adjudicator, and label 0..3")


@dataclass(frozen=True, slots=True)
class AnnotationSummary:
    quadratic_weighted_kappa: float
    labels: Mapping[str, int]
    disagreement_pair_ids: frozenset[str]
    rubric_version: int
    rubric_hash: str
    annotation_artifact_hash: str


def quadratic_weighted_kappa(left: Sequence[int], right: Sequence[int]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("kappa requires equally sized non-empty label sequences")
    if any(label not in range(4) for label in (*left, *right)):
        raise ValueError("kappa labels must be in 0..3")
    observed = [[0] * 4 for _ in range(4)]
    for first, second in zip(left, right, strict=True):
        observed[first][second] += 1
    first_counts = [sum(row) for row in observed]
    second_counts = [sum(observed[row][column] for row in range(4)) for column in range(4)]
    count = len(left)
    observed_distance = expected_distance = 0.0
    for first in range(4):
        for second in range(4):
            weight = ((first - second) / 3) ** 2
            observed_distance += weight * observed[first][second] / count
            expected_distance += weight * first_counts[first] * second_counts[second] / (count * count)
    return 1.0 if expected_distance == 0 else 1 - observed_distance / expected_distance


def complete_double_annotation(
    pairs: Sequence[GoldPair],
    annotations: Sequence[Annotation],
    adjudications: Sequence[Adjudication],
    *,
    annotator_order: tuple[str, str],
    adjudicator_id: str,
    rubric_version: int,
    rubric_hash: str,
) -> AnnotationSummary:
    """Align the same two annotators for every pair and adjudicate disagreements only."""

    pair_ids = {pair.pair_id for pair in pairs}
    if len(pair_ids) != len(pairs):
        raise ValueError("annotation pairs must be unique")
    if len(set(annotator_order)) != 2 or not all(annotator_order):
        raise ValueError("annotator_order must name two distinct fixed annotators")
    if not adjudicator_id or adjudicator_id in annotator_order or rubric_version != 1 or not rubric_hash:
        raise ValueError("annotation requires a distinct fixed adjudicator and versioned rubric")
    by_key: dict[tuple[str, str], Annotation] = {}
    for item in annotations:
        key = (item.pair_id, item.annotator_id)
        if item.pair_id not in pair_ids or item.annotator_id not in annotator_order or key in by_key:
            raise ValueError("annotations must exactly align the fixed annotators to manifest pairs")
        by_key[key] = item
    expected = {(pair_id, annotator) for pair_id in pair_ids for annotator in annotator_order}
    if set(by_key) != expected:
        raise ValueError("every pair requires exactly the same two annotators")
    ordered_ids = sorted(pair_ids)
    left = [by_key[(pair_id, annotator_order[0])].label for pair_id in ordered_ids]
    right = [by_key[(pair_id, annotator_order[1])].label for pair_id in ordered_ids]
    kappa = quadratic_weighted_kappa(left, right)
    if kappa < 0.75:
        raise ValueError(f"pre-adjudication quadratic weighted kappa {kappa:.3f} is below 0.75")
    disagreements = {pair_id for pair_id in pair_ids if by_key[(pair_id, annotator_order[0])].label != by_key[(pair_id, annotator_order[1])].label}
    rulings: dict[str, Adjudication] = {}
    for ruling in adjudications:
        if ruling.pair_id not in disagreements or ruling.pair_id in rulings:
            raise ValueError("only disagreements receive exactly one adjudication")
        if ruling.adjudicator_id != adjudicator_id:
            raise ValueError("every disagreement must use the fixed third adjudicator")
        rulings[ruling.pair_id] = ruling
    if set(rulings) != disagreements:
        raise ValueError("every disagreement requires exactly one adjudication")
    final = {
        pair_id: rulings[pair_id].label if pair_id in disagreements else by_key[(pair_id, annotator_order[0])].label
        for pair_id in pair_ids
    }
    artifact_hash = _hash({
        "pair_ids": ordered_ids, "annotator_order": list(annotator_order), "adjudicator_id": adjudicator_id,
        "rubric_version": rubric_version, "rubric_hash": rubric_hash,
        "annotations": sorted((item.pair_id, item.annotator_id, item.label) for item in annotations),
        "adjudications": sorted((item.pair_id, item.adjudicator_id, item.label) for item in adjudications),
    })
    return AnnotationSummary(kappa, MappingProxyType(final), frozenset(disagreements), rubric_version, rubric_hash, artifact_hash)


@dataclass(frozen=True, slots=True)
class CalibrationExample:
    pair_id: str
    split: GoldSplit
    raw_score: float
    gold_label: int

    def __post_init__(self) -> None:
        if not isinstance(self.split, GoldSplit):
            object.__setattr__(self, "split", GoldSplit(self.split))
        if not self.pair_id or self.split is not GoldSplit.DEV or not isfinite(self.raw_score) or self.gold_label not in range(4):
            raise ValueError("calibration examples must be finite, explicitly DEV, and labelled 0..3")


@dataclass(frozen=True, slots=True)
class PathCalibrator:
    """Frozen Platt calibrator whose output is P(gold_label >= 2)."""

    version: int
    path: CalibrationPath
    slope: float
    intercept: float
    dev_manifest_hash: str
    gold_manifest_hash: str
    model_lock_hash: str
    dev_label_hash: str
    calibration_pair_ids_hash: str
    calibration_pair_count: int
    calibration_pair_ids: tuple[str, ...]
    calibration_target: str = "P(gold_label >= 2)"

    def __post_init__(self) -> None:
        if not isinstance(self.path, CalibrationPath):
            object.__setattr__(self, "path", CalibrationPath(self.path))
        required = (
            self.dev_manifest_hash, self.gold_manifest_hash, self.model_lock_hash, self.dev_label_hash,
            self.calibration_pair_ids_hash,
        )
        if self.version != 1 or not all(required) or not isfinite(self.slope) or not isfinite(self.intercept):
            raise ValueError("calibrator requires finite parameters and manifest/model-lock provenance")
        object.__setattr__(self, "calibration_pair_ids", tuple(self.calibration_pair_ids))
        if (
            self.calibration_pair_count < 2
            or self.calibration_pair_count != len(self.calibration_pair_ids)
            or len(set(self.calibration_pair_ids)) != len(self.calibration_pair_ids)
            or self.calibration_pair_ids_hash != _hash(sorted(self.calibration_pair_ids))
        ):
            raise ValueError("calibrator must record its frozen calibration pair set")
        if self.calibration_target != "P(gold_label >= 2)":
            raise ValueError("calibrator target must be P(gold_label >= 2)")

    def predict(self, raw_score: float) -> float:
        if not isfinite(raw_score):
            raise ValueError("raw score must be finite")
        value = max(-60.0, min(60.0, self.slope * raw_score + self.intercept))
        return 1 / (1 + exp(-value))

    def document(self) -> dict[str, object]:
        return {
            "version": self.version,
            "path": self.path.value,
            "slope": self.slope,
            "intercept": self.intercept,
            "dev_manifest_hash": self.dev_manifest_hash,
            "gold_manifest_hash": self.gold_manifest_hash,
            "model_lock_hash": self.model_lock_hash,
            "dev_label_hash": self.dev_label_hash,
            "calibration_pair_ids_hash": self.calibration_pair_ids_hash,
            "calibration_pair_count": self.calibration_pair_count,
            "calibration_pair_ids": list(self.calibration_pair_ids),
            "calibration_target": self.calibration_target,
        }

    def hash(self) -> str:
        return _hash(self.document())


def write_path_calibrator(path: Path, calibrator: PathCalibrator) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(calibrator.document(), sort_keys=True, indent=2) + "\n", encoding="utf-8")


def load_path_calibrator(path: Path) -> PathCalibrator:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("calibrator artifact must be an object")
    return PathCalibrator(**document)


def fit_path_calibrator(
    path: CalibrationPath,
    examples: Sequence[CalibrationExample],
    manifest: GoldManifest,
    dev_labels: GoldLabelStore,
    model_lock_hash: str,
) -> PathCalibrator:
    """Fit only explicitly identified development pairs from a frozen manifest."""

    manifest.validate_sampling_structure()
    if not isinstance(path, CalibrationPath):
        path = CalibrationPath(path)
    if len(examples) < 2 or not model_lock_hash:
        raise ValueError("calibration needs development examples and a model lock hash")
    dev_ids = {pair.pair_id for pair in manifest.pairs if pair.split is GoldSplit.DEV}
    if set(dev_labels.labels) != dev_ids:
        raise ValueError("calibration requires the authoritative DEV label artifact")
    example_ids = [item.pair_id for item in examples]
    if len(example_ids) != len(set(example_ids)) or not set(example_ids) <= dev_ids:
        raise ValueError("calibration examples must be unique pairs from this manifest's DEV split")
    if any(item.gold_label != dev_labels.labels[item.pair_id] for item in examples):
        raise ValueError("calibration example labels do not match the DEV label artifact")
    targets = {item.gold_label >= 2 for item in examples}
    if targets != {False, True}:
        raise ValueError("calibration requires both positive and negative DEV examples")
    slope = intercept = 0.0
    for _ in range(2_000):
        grad_slope = grad_intercept = 0.0
        for item in examples:
            value = max(-60.0, min(60.0, slope * item.raw_score + intercept))
            residual = 1 / (1 + exp(-value)) - int(item.gold_label >= 2)
            grad_slope += residual * item.raw_score
            grad_intercept += residual
        slope -= 0.05 * grad_slope / len(examples)
        intercept -= 0.05 * grad_intercept / len(examples)
    return PathCalibrator(
        1, path, slope, intercept, manifest.dev_hash(), manifest.hash(), model_lock_hash, dev_labels.hash(),
        _hash(sorted(example_ids)), len(example_ids), tuple(sorted(example_ids)),
    )


@dataclass(frozen=True, slots=True)
class ThresholdArtifact:
    version: int
    path: CalibrationPath
    low: float
    high: float
    calibrator_hash: str
    model_lock_hash: str
    dev_manifest_hash: str
    dev_label_hash: str
    stage2_config_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.path, CalibrationPath):
            object.__setattr__(self, "path", CalibrationPath(self.path))
        required = (
            self.calibrator_hash, self.model_lock_hash, self.dev_manifest_hash, self.dev_label_hash,
            self.stage2_config_hash,
        )
        if self.version != 1 or not all(required) or not all(isfinite(value) for value in (self.low, self.high)):
            raise ValueError("threshold artifact requires finite thresholds and complete provenance")
        if not 0 <= self.low < self.high <= 1:
            raise ValueError("thresholds must satisfy 0 <= low < high <= 1")

    def document(self) -> dict[str, object]:
        return {
            "version": self.version, "path": self.path.value, "low": self.low, "high": self.high,
            "calibrator_hash": self.calibrator_hash, "model_lock_hash": self.model_lock_hash,
            "dev_manifest_hash": self.dev_manifest_hash, "dev_label_hash": self.dev_label_hash,
            "stage2_config_hash": self.stage2_config_hash,
        }

    def hash(self) -> str:
        return _hash(self.document())


@dataclass(frozen=True, slots=True)
class Prediction:
    pair_id: str
    candidate_id: str
    decision: Stage2Decision
    raw_score: float | None
    probability: float
    path: CalibrationPath
    calibrator_hash: str
    threshold_hash: str
    model_lock_hash: str
    manifest_hash: str
    stage2_config_hash: str
    inference_artifact_hash: str
    review_reason: ReviewReason | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.decision, Stage2Decision):
            object.__setattr__(self, "decision", Stage2Decision(self.decision))
        if not isinstance(self.path, CalibrationPath):
            object.__setattr__(self, "path", CalibrationPath(self.path))
        if self.review_reason is not None and not isinstance(self.review_reason, ReviewReason):
            object.__setattr__(self, "review_reason", ReviewReason(self.review_reason))
        required = (
            self.pair_id, self.candidate_id, self.calibrator_hash, self.threshold_hash, self.model_lock_hash,
            self.manifest_hash, self.stage2_config_hash, self.inference_artifact_hash,
        )
        if not all(required) or not isfinite(self.probability) or not 0 <= self.probability <= 1:
            raise ValueError("prediction requires finite probability and complete evaluation provenance")
        if self.raw_score is not None and not isfinite(self.raw_score):
            raise ValueError("prediction raw_score must be finite when present")
        if (self.decision is Stage2Decision.NEEDS_REVIEW) != (self.review_reason is not None):
            raise ValueError("only needs_review predictions carry a review reason")
        if self.raw_score is None and (
            self.decision is not Stage2Decision.NEEDS_REVIEW
            or self.review_reason not in {ReviewReason.SCHEMA_ERROR, ReviewReason.TIMEOUT, ReviewReason.SERVICE_ERROR}
            or self.probability != 0.5
        ):
            raise ValueError("missing model scores must fail open to needs_review with probability 0.5")


def _validate_prediction_provenance(
    predictions: Sequence[Prediction],
    manifest: GoldManifest,
    calibrators: Mapping[CalibrationPath, PathCalibrator],
    thresholds: Mapping[CalibrationPath, ThresholdArtifact],
) -> None:
    manifest_hash = manifest.hash()
    dev_manifest_hash = manifest.dev_hash()
    dev_pair_ids = {pair.pair_id for pair in manifest.pairs if pair.split is GoldSplit.DEV}
    for prediction in predictions:
        calibrator = calibrators.get(prediction.path)
        threshold = thresholds.get(prediction.path)
        if calibrator is None:
            raise ValueError(f"prediction path {prediction.path.value} has no frozen calibrator")
        if threshold is None:
            raise ValueError(f"prediction path {prediction.path.value} has no frozen threshold artifact")
        if (
            prediction.manifest_hash != manifest_hash
            or calibrator.gold_manifest_hash != manifest_hash
            or calibrator.dev_manifest_hash != dev_manifest_hash
            or calibrator.path is not prediction.path
        ):
            raise ValueError("prediction/calibrator manifest provenance mismatch")
        if prediction.calibrator_hash != calibrator.hash() or prediction.model_lock_hash != calibrator.model_lock_hash:
            raise ValueError("prediction calibrator/model-lock provenance mismatch")
        if not set(calibrator.calibration_pair_ids) <= dev_pair_ids:
            raise ValueError("calibrator contains a non-DEV calibration pair")
        if (
            prediction.threshold_hash != threshold.hash()
            or threshold.path is not prediction.path
            or threshold.calibrator_hash != calibrator.hash()
            or threshold.model_lock_hash != calibrator.model_lock_hash
            or threshold.dev_manifest_hash != dev_manifest_hash
            or threshold.dev_label_hash != calibrator.dev_label_hash
            or prediction.stage2_config_hash != threshold.stage2_config_hash
        ):
            raise ValueError("prediction threshold/config provenance mismatch")
        if prediction.raw_score is None:
            continue
        if abs(prediction.probability - calibrator.predict(prediction.raw_score)) > 1e-12:
            raise ValueError("prediction probability does not match its path calibrator")
        if prediction.decision is Stage2Decision.RELEVANT and prediction.probability < threshold.high:
            raise ValueError("relevant decision is below the frozen high threshold")
        if prediction.decision is Stage2Decision.IRRELEVANT and prediction.probability > threshold.low:
            raise ValueError("irrelevant decision is above the frozen low threshold")
        if (
            prediction.decision is Stage2Decision.NEEDS_REVIEW
            and prediction.review_reason is ReviewReason.UNCERTAIN
            and not threshold.low < prediction.probability < threshold.high
        ):
            raise ValueError("uncertain decision is outside the frozen threshold band")


@dataclass(frozen=True, slots=True)
class WilsonInterval:
    point: float
    lower: float
    upper: float
    numerator: float
    denominator: float


def wilson_interval(numerator: float, denominator: float, z: float = 1.959963984540054) -> WilsonInterval:
    if not isfinite(numerator) or not isfinite(denominator) or denominator <= 0 or not 0 <= numerator <= denominator:
        raise ValueError("Wilson interval requires finite 0 <= numerator <= denominator and denominator > 0")
    point = numerator / denominator
    factor = 1 + z * z / denominator
    centre = (point + z * z / (2 * denominator)) / factor
    spread = z * sqrt(point * (1 - point) / denominator + z * z / (4 * denominator * denominator)) / factor
    return WilsonInterval(point, max(0.0, centre - spread), min(1.0, centre + spread), numerator, denominator)


@dataclass(frozen=True, slots=True)
class SliceMetrics:
    automatic_precision: float
    automatic_recall: float
    positive_f1: float
    retention_recall: float
    automatic_coverage: float
    brier_score: float
    ece_10: float
    false_retained_per_thousand: float
    false_rejected_per_thousand: float
    error_needs_review_rate: float
    positive_count: int
    retention_interval: WilsonInterval | None
    core_retention_interval: WilsonInterval | None
    automatic_recall_interval: WilsonInterval | None
    review_reason_counts: Mapping[ReviewReason, int]


@dataclass(frozen=True, slots=True)
class EvaluationTrace:
    manifest_hash: str
    pair_universe_hash: str
    candidate_id: str
    calibrator_hashes: tuple[str, ...]
    threshold_hashes: tuple[str, ...]
    model_lock_hashes: tuple[str, ...]
    stage2_config_hash: str
    paths: tuple[CalibrationPath, ...]


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    point: float
    lower: float
    upper: float


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    metrics: SliceMetrics
    trace: EvaluationTrace


@dataclass(frozen=True, slots=True)
class HiddenEvaluation:
    split: GoldSplit
    metrics: SliceMetrics
    topic_macro_positive_f1: float
    language_metrics: Mapping[str, SliceMetrics]
    language_positive_counts: Mapping[str, int]
    size: int
    core_count: int
    ece_interval: ConfidenceInterval | None
    trace: EvaluationTrace


@dataclass(frozen=True, slots=True)
class InverseProbabilityMetrics:
    automatic_precision: float
    automatic_recall: float
    positive_f1: float
    retention_recall: float
    brier_score: float


def _trace(predictions: Sequence[Prediction], manifest: GoldManifest) -> EvaluationTrace:
    candidates = {item.candidate_id for item in predictions}
    configs = {item.stage2_config_hash for item in predictions}
    if len(candidates) != 1 or len(configs) != 1:
        raise ValueError("an evaluation run must use one candidate and one frozen Stage 2 config")
    return EvaluationTrace(
        manifest.hash(),
        _hash(sorted(item.pair_id for item in predictions)),
        next(iter(candidates)),
        tuple(sorted({item.calibrator_hash for item in predictions})),
        tuple(sorted({item.threshold_hash for item in predictions})),
        tuple(sorted({item.model_lock_hash for item in predictions})),
        next(iter(configs)),
        tuple(sorted({item.path for item in predictions}, key=str)),
    )


def _classification_metrics(predictions: Sequence[Prediction], labels: Mapping[str, int]) -> tuple[float, float, float, float, int]:
    positives = sum(labels[item.pair_id] >= 2 for item in predictions)
    tp = sum(item.decision is Stage2Decision.RELEVANT and labels[item.pair_id] >= 2 for item in predictions)
    fp = sum(item.decision is Stage2Decision.RELEVANT and labels[item.pair_id] < 2 for item in predictions)
    review_positive = sum(item.decision is Stage2Decision.NEEDS_REVIEW and labels[item.pair_id] >= 2 for item in predictions)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / positives if positives else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    retention = (tp + review_positive) / positives if positives else 0.0
    return precision, recall, f1, retention, positives


def _ece(predictions: Sequence[Prediction], labels: Mapping[str, int]) -> float:
    ordered = sorted(predictions, key=lambda item: (item.probability, item.pair_id))
    bins = min(10, len(ordered))
    quotient, remainder = divmod(len(ordered), bins)
    start = 0
    total = 0.0
    for index in range(bins):
        size = quotient + (index < remainder)
        bucket = ordered[start:start + size]
        start += size
        confidence = sum(item.probability for item in bucket) / size
        frequency = sum(labels[item.pair_id] >= 2 for item in bucket) / size
        total += size / len(ordered) * abs(confidence - frequency)
    return total


def _ece_bootstrap(predictions: Sequence[Prediction], labels: Mapping[str, int], iterations: int = 1_000) -> ConfidenceInterval:
    seed = int(_hash(sorted(item.pair_id for item in predictions))[:16], 16)
    random = Random(seed)
    values = []
    for _ in range(iterations):
        sample = [predictions[random.randrange(len(predictions))] for _ in predictions]
        values.append(_ece(sample, labels))
    values.sort()
    return ConfidenceInterval(
        _ece(predictions, labels), values[int(0.025 * iterations)], values[min(iterations - 1, int(0.975 * iterations))]
    )


def measure_predictions(pairs: Sequence[GoldPair], labels: GoldLabelStore, predictions: Sequence[Prediction]) -> SliceMetrics:
    if not pairs:
        raise ValueError("metrics require at least one pair")
    pair_ids = {pair.pair_id for pair in pairs}
    if not pair_ids <= set(labels.labels):
        raise ValueError("metrics require labels for every pair")
    by_id = {item.pair_id: item for item in predictions}
    if len(by_id) != len(predictions) or set(by_id) != pair_ids:
        raise ValueError("predictions must exactly cover evaluated pair_ids once")
    ordered = [by_id[pair.pair_id] for pair in pairs]
    precision, recall, f1, retention, positives = _classification_metrics(ordered, labels.labels)
    automatic = sum(item.decision is not Stage2Decision.NEEDS_REVIEW for item in ordered)
    false_retained = sum(item.decision is Stage2Decision.RELEVANT and labels.labels[item.pair_id] < 2 for item in ordered)
    false_rejected = sum(item.decision is Stage2Decision.IRRELEVANT and labels.labels[item.pair_id] >= 2 for item in ordered)
    brier = sum((item.probability - int(labels.labels[item.pair_id] >= 2)) ** 2 for item in ordered) / len(ordered)
    retained_positive = sum(item.decision is not Stage2Decision.IRRELEVANT and labels.labels[item.pair_id] >= 2 for item in ordered)
    core = [item for item in ordered if labels.labels[item.pair_id] == 3]
    core_retained = sum(item.decision is not Stage2Decision.IRRELEVANT for item in core)
    error_review = sum(
        item.decision is Stage2Decision.NEEDS_REVIEW
        and item.review_reason in {ReviewReason.SCHEMA_ERROR, ReviewReason.TIMEOUT, ReviewReason.SERVICE_ERROR}
        for item in ordered
    )
    return SliceMetrics(
        precision, recall, f1, retention, automatic / len(ordered), brier, _ece(ordered, labels.labels),
        false_retained * 1000 / len(ordered), false_rejected * 1000 / len(ordered), error_review / len(ordered),
        positives, wilson_interval(retained_positive, positives) if positives else None,
        wilson_interval(core_retained, len(core)) if core else None,
        wilson_interval(
            sum(item.decision is Stage2Decision.RELEVANT and labels.labels[item.pair_id] >= 2 for item in ordered),
            positives,
        ) if positives else None,
        Counter(item.review_reason for item in ordered if item.review_reason is not None),
    )


def inverse_probability_metrics(
    pairs: Sequence[GoldPair], labels: GoldLabelStore, predictions: Sequence[Prediction], *, use_inverse_probability_weights: bool = False
) -> InverseProbabilityMetrics:
    if not use_inverse_probability_weights:
        raise ValueError("cross-set metrics require explicit inverse-probability weighting")
    by_id = {item.pair_id: item for item in predictions}
    if len(by_id) != len(predictions) or set(by_id) != {pair.pair_id for pair in pairs}:
        raise ValueError("predictions must exactly cover weighted pair_ids once")
    tp = fp = retained = positives = brier = total_weight = 0.0
    for pair in pairs:
        prediction = by_id[pair.pair_id]
        positive = labels.labels[pair.pair_id] >= 2
        weight = 1 / pair.sampling_probability
        total_weight += weight
        positives += weight * positive
        brier += weight * (prediction.probability - int(positive)) ** 2
        if prediction.decision is Stage2Decision.RELEVANT:
            tp += weight * positive
            fp += weight * (not positive)
        if prediction.decision is not Stage2Decision.IRRELEVANT:
            retained += weight * positive
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / positives if positives else 0.0
    return InverseProbabilityMetrics(
        precision, recall, 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        retained / positives if positives else 0.0, brier / total_weight,
    )


class CandidateEvaluator:
    """Development-only evaluator; hidden labels are rejected at construction."""

    def __init__(
        self,
        manifest: GoldManifest,
        dev_labels: GoldLabelStore,
        calibrators: Mapping[CalibrationPath, PathCalibrator],
        thresholds: Mapping[CalibrationPath, ThresholdArtifact],
    ) -> None:
        manifest.validate_sampling_structure()
        self._manifest = manifest
        self._pairs = tuple(pair for pair in manifest.pairs if pair.split is GoldSplit.DEV)
        if set(dev_labels.labels) != {pair.pair_id for pair in self._pairs}:
            raise ValueError("candidate evaluation accepts exactly DEV labels")
        self._labels = dev_labels
        self._calibrators = calibrators
        self._thresholds = thresholds

    def evaluate(self, predictions: Sequence[Prediction]) -> CandidateEvaluation:
        _validate_prediction_provenance(predictions, self._manifest, self._calibrators, self._thresholds)
        if any(self._calibrators[item.path].dev_label_hash != self._labels.hash() for item in predictions):
            raise ValueError("candidate calibrator does not bind the authoritative DEV labels")
        return CandidateEvaluation(measure_predictions(self._pairs, self._labels, predictions), _trace(predictions, self._manifest))


@dataclass(frozen=True, slots=True)
class GateResult:
    passed: bool
    failures: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.passed == bool(self.failures):
            raise ValueError("gate result must pass exactly when it has no failures")


PromotionResult = GateResult


@dataclass(frozen=True, slots=True)
class PromotionMarker:
    version: int
    evaluator_id: str
    manifest_hash: str
    consumed_run_id: str | None = None
    regression_splits: frozenset[GoldSplit] = frozenset()

    def __post_init__(self) -> None:
        if self.version != 1 or not self.evaluator_id or not self.manifest_hash:
            raise ValueError("promotion marker requires version, evaluator_id, and manifest hash")
        converted = frozenset(GoldSplit(item) for item in self.regression_splits)
        if GoldSplit.DEV in converted:
            raise ValueError("DEV is not a hidden regression split")
        object.__setattr__(self, "regression_splits", converted)

    @property
    def consumed(self) -> bool:
        return self.consumed_run_id is not None

    def document(self) -> dict[str, object]:
        return {
            "version": self.version, "evaluator_id": self.evaluator_id, "manifest_hash": self.manifest_hash,
            "consumed_run_id": self.consumed_run_id,
            "regression_splits": sorted(split.value for split in self.regression_splits),
        }


def write_promotion_marker(path: Path, marker: PromotionMarker) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(marker.document(), sort_keys=True, indent=2) + "\n")


def load_promotion_marker(path: Path) -> PromotionMarker:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or set(document) != {
        "version", "evaluator_id", "manifest_hash", "consumed_run_id", "regression_splits"
    }:
        raise ValueError("promotion marker must be an exact versioned object")
    return PromotionMarker(**document)


@dataclass(frozen=True, slots=True)
class CandidateModelArtifacts:
    candidate_id: str
    calibrators: Mapping[CalibrationPath, PathCalibrator]
    thresholds: Mapping[CalibrationPath, ThresholdArtifact]

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.calibrators or set(self.calibrators) != set(self.thresholds):
            raise ValueError("candidate artifacts require matching calibrator and threshold paths")
        object.__setattr__(self, "calibrators", MappingProxyType(dict(self.calibrators)))
        object.__setattr__(self, "thresholds", MappingProxyType(dict(self.thresholds)))


@dataclass(frozen=True, slots=True)
class PromotionSubmission:
    candidate_id: str
    runs: tuple[tuple[Prediction, ...], tuple[Prediction, ...], tuple[Prediction, ...]]

    def __post_init__(self) -> None:
        if not self.candidate_id or len(self.runs) != 3 or any(not run for run in self.runs):
            raise ValueError("promotion submission requires one candidate and three complete runs")
        if any(any(item.candidate_id != self.candidate_id for item in run) for run in self.runs):
            raise ValueError("every prediction must bind the submission candidate_id")


@dataclass(frozen=True, slots=True)
class RunProvenance:
    candidate_id: str
    manifest_hash: str
    stage2_config_hash: str
    calibrator_hashes: tuple[str, ...]
    threshold_hashes: tuple[str, ...]
    model_lock_hashes: tuple[str, ...]
    pair_universe_hash: str


@dataclass(frozen=True, slots=True)
class DeterminismResult:
    gate: GateResult
    identical_ratio: float
    provenance: RunProvenance

    def __post_init__(self) -> None:
        if not 0 <= self.identical_ratio <= 1:
            raise ValueError("determinism ratio must be in [0, 1]")
        if self.gate.passed != (self.identical_ratio >= 0.99):
            raise ValueError("determinism gate and ratio disagree")

    @property
    def passed(self) -> bool:
        return self.gate.passed

    @property
    def failures(self) -> tuple[str, ...]:
        return self.gate.failures


@dataclass(frozen=True, slots=True)
class PromotionCandidateResult:
    candidate_id: str
    manifest_hash: str
    evaluation_run_id: str
    evaluations: Mapping[GoldSplit, HiddenEvaluation]
    determinism: DeterminismResult
    hidden_pair_universe_hashes: Mapping[GoldSplit, str]
    main_languages: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.manifest_hash or not self.evaluation_run_id:
            raise ValueError("promotion candidate result requires bound identity and run")
        object.__setattr__(self, "evaluations", MappingProxyType(dict(self.evaluations)))
        object.__setattr__(self, "hidden_pair_universe_hashes", MappingProxyType(dict(self.hidden_pair_universe_hashes)))


@dataclass(frozen=True, slots=True)
class PromotionBatchResult:
    manifest_hash: str
    evaluation_run_id: str
    incumbent_candidate_id: str
    candidates: Mapping[str, PromotionCandidateResult]
    comparisons: Mapping[tuple[str, GoldSplit], PairedBootstrapResult]
    marker: PromotionMarker

    def __post_init__(self) -> None:
        if (
            not self.manifest_hash or not self.evaluation_run_id or self.incumbent_candidate_id not in self.candidates
            or self.marker.manifest_hash != self.manifest_hash or self.marker.consumed_run_id != self.evaluation_run_id
        ):
            raise ValueError("promotion batch result provenance mismatch")
        object.__setattr__(self, "candidates", MappingProxyType(dict(self.candidates)))
        object.__setattr__(self, "comparisons", MappingProxyType(dict(self.comparisons)))


class PromotionEvaluator:
    """Sealed one-shot evaluator for all candidates in one blind promotion batch.

    The state path is derived from the manifest hash.  The consumed marker is
    created atomically before any hidden result is returned, so a new evaluator
    ID or process cannot reuse the same holdout under the same state root.
    """

    def __init__(
        self,
        manifest: GoldManifest,
        private_labels: GoldLabelStore,
        candidate_artifacts: Mapping[str, CandidateModelArtifacts],
        evaluator_id: str,
        state_root: Path,
    ) -> None:
        manifest.validate(private_labels)
        if not evaluator_id or not candidate_artifacts:
            raise ValueError("promotion evaluator requires evaluator_id and candidate artifacts")
        if set(candidate_artifacts) != {item.candidate_id for item in candidate_artifacts.values()}:
            raise ValueError("candidate artifact keys must match candidate_id")
        self._manifest = manifest
        self._labels = GoldLabelStore(
            private_labels.labels, private_labels.annotation_artifact_hash,
            private_labels.hard_negative_pair_ids, private_labels.hard_positive_pair_ids
        )
        self._candidate_artifacts = MappingProxyType(dict(candidate_artifacts))
        self._evaluator_id = evaluator_id
        self._marker_path = state_root / f"{manifest.hash()}.promotion.json"
        self._marker = load_promotion_marker(self._marker_path) if self._marker_path.exists() else PromotionMarker(
            1, evaluator_id, manifest.hash()
        )
        if self._marker.manifest_hash != manifest.hash():
            raise ValueError("promotion marker manifest mismatch")

    @property
    def marker_path(self) -> Path:
        return self._marker_path

    @property
    def marker(self) -> PromotionMarker:
        return self._marker

    def evaluate_candidates(
        self,
        submissions: Sequence[PromotionSubmission],
        *,
        incumbent_candidate_id: str,
        evaluation_run_id: str,
        bootstrap_iterations: int = 2_000,
        bootstrap_seed: int = 0,
    ) -> PromotionBatchResult:
        if not evaluation_run_id or self._marker.consumed or self._marker.regression_splits:
            raise ValueError("hidden holdout is consumed or regression-only")
        by_candidate = {item.candidate_id: item for item in submissions}
        if len(by_candidate) != len(submissions) or incumbent_candidate_id not in by_candidate:
            raise ValueError("promotion batch requires unique candidates and a submitted incumbent")
        hidden_pairs = tuple(pair for pair in self._manifest.pairs if pair.split is not GoldSplit.DEV)
        expected = tuple(pair.pair_id for pair in hidden_pairs)
        split_by_id = {pair.pair_id: pair.split for pair in hidden_pairs}
        candidate_results: dict[str, PromotionCandidateResult] = {}
        first_runs: dict[str, tuple[Prediction, ...]] = {}
        for candidate_id, submission in by_candidate.items():
            artifacts = self._candidate_artifacts.get(candidate_id)
            if artifacts is None:
                raise ValueError(f"candidate {candidate_id} has no frozen model artifacts")
            dev_pair_ids = {pair.pair_id for pair in self._manifest.pairs if pair.split is GoldSplit.DEV}
            dev_labels = GoldLabelStore(
                {pair_id: self._labels.labels[pair_id] for pair_id in dev_pair_ids}, self._labels.annotation_artifact_hash
            )
            if any(calibrator.dev_label_hash != dev_labels.hash() for calibrator in artifacts.calibrators.values()):
                raise ValueError("candidate calibrator does not bind the authoritative DEV labels")
            for run in submission.runs:
                if len({item.pair_id for item in run}) != len(run) or {item.pair_id for item in run} != set(expected):
                    raise ValueError("every promotion run must exactly cover the hidden pair universe")
                _validate_prediction_provenance(run, self._manifest, artifacts.calibrators, artifacts.thresholds)
            determinism = determinism_gate(submission.runs, expected)
            first_run = submission.runs[0]
            first_runs[candidate_id] = first_run
            evaluations = {
                split: self._evaluate_split(split, [item for item in first_run if split_by_id[item.pair_id] is split])
                for split in (GoldSplit.HIDDEN_HARD, GoldSplit.HIDDEN_REAL)
            }
            universe_hashes = {
                split: pair_universe_hash(
                    [pair.pair_id for pair in hidden_pairs if pair.split is split]
                )
                for split in (GoldSplit.HIDDEN_HARD, GoldSplit.HIDDEN_REAL)
            }
            candidate_results[candidate_id] = PromotionCandidateResult(
                candidate_id, self._manifest.hash(), evaluation_run_id, MappingProxyType(evaluations), determinism,
                MappingProxyType(universe_hashes), self._manifest.main_languages,
            )
        comparisons: dict[tuple[str, GoldSplit], PairedBootstrapResult] = {}
        for candidate_id, predictions in first_runs.items():
            if candidate_id == incumbent_candidate_id:
                continue
            for split in (GoldSplit.HIDDEN_HARD, GoldSplit.HIDDEN_REAL):
                pairs = [pair for pair in hidden_pairs if pair.split is split]
                pair_ids = {pair.pair_id for pair in pairs}
                comparisons[(candidate_id, split)] = paired_bootstrap_comparison(
                    pairs, self._labels,
                    [item for item in first_runs[incumbent_candidate_id] if item.pair_id in pair_ids],
                    [item for item in predictions if item.pair_id in pair_ids],
                    iterations=bootstrap_iterations, seed=bootstrap_seed,
                )
        marker = PromotionMarker(
            1, self._evaluator_id, self._manifest.hash(), evaluation_run_id,
            frozenset({GoldSplit.HIDDEN_HARD, GoldSplit.HIDDEN_REAL}),
        )
        self._claim_marker(marker)
        return PromotionBatchResult(
            self._manifest.hash(), evaluation_run_id, incumbent_candidate_id,
            MappingProxyType(candidate_results), MappingProxyType(comparisons), marker,
        )

    def reveal_for_regression(self, split: GoldSplit) -> GoldLabelStore:
        split = GoldSplit(split)
        if split is GoldSplit.DEV:
            raise ValueError("development labels are not hidden")
        if not self._marker_path.exists():
            self._claim_marker(PromotionMarker(1, self._evaluator_id, self._manifest.hash(), None, frozenset({split})))
        pair_ids = {pair.pair_id for pair in self._manifest.pairs if pair.split is split}
        return GoldLabelStore(
            {pair_id: self._labels.labels[pair_id] for pair_id in pair_ids},
            self._labels.annotation_artifact_hash,
            self._labels.hard_negative_pair_ids & pair_ids,
            self._labels.hard_positive_pair_ids & pair_ids,
        )

    def _claim_marker(self, marker: PromotionMarker) -> None:
        self._marker_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self._marker_path.open("x", encoding="utf-8") as handle:
                handle.write(json.dumps(marker.document(), sort_keys=True, indent=2) + "\n")
        except FileExistsError as error:
            self._marker = load_promotion_marker(self._marker_path)
            raise ValueError("hidden holdout already has a persistent consumed/regression marker") from error
        self._marker = marker

    def _evaluate_split(self, split: GoldSplit, predictions: Sequence[Prediction]) -> HiddenEvaluation:
        pairs = tuple(pair for pair in self._manifest.pairs if pair.split is split)
        by_id = {item.pair_id: item for item in predictions}
        topic_metrics = [
            measure_predictions(
                [pair for pair in pairs if pair.topic == topic], self._labels,
                [by_id[pair.pair_id] for pair in pairs if pair.topic == topic],
            ).positive_f1
            for topic in sorted({pair.topic for pair in pairs})
        ]
        language_metrics: dict[str, SliceMetrics] = {}
        language_counts: dict[str, int] = {}
        for language in self._manifest.main_languages:
            language_pairs = [pair for pair in pairs if pair.language == language]
            metrics = measure_predictions(language_pairs, self._labels, [by_id[pair.pair_id] for pair in language_pairs])
            language_metrics[language] = metrics
            language_counts[language] = metrics.positive_count
        return HiddenEvaluation(
            split, measure_predictions(pairs, self._labels, predictions), sum(topic_metrics) / len(topic_metrics),
            MappingProxyType(language_metrics), MappingProxyType(language_counts), len(pairs),
            sum(self._labels.labels[pair.pair_id] == 3 for pair in pairs),
            _ece_bootstrap(predictions, self._labels.labels) if split is GoldSplit.HIDDEN_REAL else None,
            _trace(predictions, self._manifest),
        )


def _run_provenance(run: Sequence[Prediction]) -> RunProvenance:
    trace_candidates = {item.candidate_id for item in run}
    manifests = {item.manifest_hash for item in run}
    configs = {item.stage2_config_hash for item in run}
    if len(trace_candidates) != 1 or len(manifests) != 1 or len(configs) != 1:
        raise ValueError("a determinism run must bind one candidate, manifest, and config")
    return RunProvenance(
        next(iter(trace_candidates)), next(iter(manifests)), next(iter(configs)),
        tuple(sorted({item.calibrator_hash for item in run})),
        tuple(sorted({item.threshold_hash for item in run})),
        tuple(sorted({item.model_lock_hash for item in run})),
        _hash(sorted(item.pair_id for item in run)),
    )


def determinism_gate(runs: Sequence[Sequence[Prediction]], expected_hidden_pair_ids: Sequence[str]) -> DeterminismResult:
    policy = hidden_promotion_gate_policy_document()["determinism"]
    if len(runs) != policy["runs"]:
        raise ValueError("determinism gate requires exactly three complete runs")
    expected = set(expected_hidden_pair_ids)
    if not expected or len(expected) != len(expected_hidden_pair_ids):
        raise ValueError("expected hidden pair universe must be non-empty and unique")
    maps = [{item.pair_id: item.decision for item in run} for run in runs]
    if any(len(mapping) != len(run) or set(mapping) != expected for mapping, run in zip(maps, runs, strict=True)):
        raise ValueError("every determinism run must exactly cover the expected hidden pair universe")
    provenance = tuple(_run_provenance(run) for run in runs)
    if provenance[1:] != provenance[:1] * 2:
        raise ValueError("determinism runs must use identical candidate/model/config/manifest provenance")
    identical = sum(maps[0][pair_id] == maps[1][pair_id] == maps[2][pair_id] for pair_id in expected)
    ratio = identical / len(expected)
    gate = GateResult(
        ratio >= policy["agreement_min"],
        () if ratio >= policy["agreement_min"] else (f"determinism {ratio:.3%} < 99%",),
    )
    return DeterminismResult(gate, ratio, provenance[0])


def promotion_gate(candidate: PromotionCandidateResult) -> GateResult:
    policy = hidden_promotion_gate_policy_document()
    per_split = policy["per_split"]
    language_policy = policy["main_language"]
    hard_policy = policy["hidden_hard"]
    real_policy = policy["hidden_real"]
    evaluations = candidate.evaluations
    if set(evaluations) != {GoldSplit.HIDDEN_HARD, GoldSplit.HIDDEN_REAL}:
        raise ValueError("promotion requires separate hard and real hidden evaluations")
    hard, real = evaluations[GoldSplit.HIDDEN_HARD], evaluations[GoldSplit.HIDDEN_REAL]
    if hard.split is not GoldSplit.HIDDEN_HARD or real.split is not GoldSplit.HIDDEN_REAL:
        raise ValueError("hidden evaluation split metadata mismatch")
    if hard.size != 150 or real.size != 150:
        raise ValueError("each hidden evaluation must contain exactly 150 pairs")
    if set(candidate.hidden_pair_universe_hashes) != {GoldSplit.HIDDEN_HARD, GoldSplit.HIDDEN_REAL}:
        raise ValueError("promotion candidate lacks frozen hidden pair universes")
    for evaluation in (hard, real):
        if (
            evaluation.trace.candidate_id != candidate.candidate_id
            or evaluation.trace.manifest_hash != candidate.manifest_hash
            or evaluation.trace.pair_universe_hash != candidate.hidden_pair_universe_hashes[evaluation.split]
        ):
            raise ValueError("hidden evaluation trace does not match promotion candidate artifact")
    if (
        hard.trace.stage2_config_hash != real.trace.stage2_config_hash
        or hard.trace.calibrator_hashes != real.trace.calibrator_hashes
        or hard.trace.threshold_hashes != real.trace.threshold_hashes
        or hard.trace.model_lock_hashes != real.trace.model_lock_hashes
    ):
        raise ValueError("hard and real evaluations must use one frozen candidate configuration")
    if candidate.determinism.provenance.candidate_id != candidate.candidate_id:
        raise ValueError("determinism provenance does not match promotion candidate")
    if (
        candidate.determinism.provenance.manifest_hash != candidate.manifest_hash
        or candidate.determinism.provenance.stage2_config_hash != hard.trace.stage2_config_hash
        or candidate.determinism.provenance.calibrator_hashes != hard.trace.calibrator_hashes
        or candidate.determinism.provenance.threshold_hashes != hard.trace.threshold_hashes
        or candidate.determinism.provenance.model_lock_hashes != hard.trace.model_lock_hashes
    ):
        raise ValueError("determinism model/config provenance does not match hidden evaluations")
    if len(candidate.main_languages) < 2:
        raise ValueError("promotion candidate must preserve frozen main-language slices")
    if set(hard.language_metrics) != set(candidate.main_languages) or set(real.language_metrics) != set(candidate.main_languages):
        raise ValueError("promotion candidate omitted a frozen main-language slice")
    failures = list(candidate.determinism.failures)
    for name, evaluation in (("hard", hard), ("real", real)):
        if evaluation.metrics.retention_recall < per_split["retention_recall_min"]:
            failures.append(f"{name} retention recall < 0.95")
        if evaluation.metrics.automatic_coverage < per_split["automatic_coverage_min"]:
            failures.append(f"{name} automatic coverage < 0.95")
        if evaluation.metrics.error_needs_review_rate > per_split["error_needs_review_rate_max"]:
            failures.append(f"{name} error needs_review rate > 0.5%")
        interval = evaluation.metrics.core_retention_interval
        if (
            evaluation.core_count >= per_split["core_retention_min_count"]
            and interval
            and interval.point < per_split["core_retention_recall_min"]
        ):
            failures.append(f"{name} core retention recall < 0.97")
        for language, metrics in evaluation.language_metrics.items():
            if (
                evaluation.language_positive_counts[language] >= language_policy["positive_min_count"]
                and metrics.retention_recall < language_policy["retention_recall_min"]
            ):
                failures.append(f"{name}/{language} retention recall < 0.90")
    core_intervals = [item.metrics.core_retention_interval for item in (hard, real) if item.metrics.core_retention_interval]
    core_numerator = sum(item.numerator for item in core_intervals)
    core_denominator = sum(item.denominator for item in core_intervals)
    if (
        not core_denominator
        or core_numerator / core_denominator < policy["combined_hidden"]["core_retention_recall_min"]
    ):
        failures.append("combined hidden core retention recall < 0.97")
    if real.metrics.automatic_precision < real_policy["operational_precision_min"]:
        failures.append("real operational precision < 0.80")
    if hard.metrics.positive_f1 < hard_policy["positive_f1_min"]:
        failures.append("hard positive F1 < 0.88")
    if hard.topic_macro_positive_f1 < hard_policy["topic_macro_positive_f1_min"]:
        failures.append("hard topic macro positive F1 < 0.82")
    if real.metrics.brier_score > real_policy["brier_score_max"]:
        failures.append("real Brier score > 0.15")
    if (
        real.size >= real_policy["ece_10_min_count"]
        and real.metrics.ece_10 > real_policy["ece_10_max"]
    ):
        failures.append("real ECE > 0.08")
    return GateResult(not failures, tuple(failures))


@dataclass(frozen=True, slots=True)
class PerformanceCase:
    pair_id: str
    input_tokens: int
    abstract_missing: bool = False

    def __post_init__(self) -> None:
        if not self.pair_id or not isfinite(self.input_tokens) or self.input_tokens < 1:
            raise ValueError("performance case requires pair_id and positive input_tokens")


@dataclass(frozen=True, slots=True)
class PerformanceRoutingManifest:
    version: int
    corpus_hash: str
    stage2_config_hash: str
    model_lock_hashes: tuple[str, ...]
    threshold_artifact_hashes: tuple[str, ...]
    output_token_limit: int
    cases: tuple[PerformanceCase, ...]
    normal_qwen_ids: frozenset[str]
    stress_qwen_ids: frozenset[str]
    pipeline_components: tuple[str, ...] = ("rules", "reranker", "qwen", "schema_validation", "sqlite_commit")

    def __post_init__(self) -> None:
        if self.version != 1 or len(self.cases) != 1_000 or not self.corpus_hash or not self.stage2_config_hash:
            raise ValueError("benchmark manifest requires version 1, hashes, and exactly 1,000 cases")
        if not self.model_lock_hashes or len(set(self.model_lock_hashes)) != len(self.model_lock_hashes) or not all(self.model_lock_hashes):
            raise ValueError("benchmark manifest requires unique model lock hashes")
        if not self.threshold_artifact_hashes or not all(self.threshold_artifact_hashes) or self.output_token_limit < 1:
            raise ValueError("benchmark manifest requires thresholds and a positive output cap")
        ids = {case.pair_id for case in self.cases}
        if len(ids) != len(self.cases) or not self.normal_qwen_ids <= ids or not self.stress_qwen_ids <= ids:
            raise ValueError("benchmark routing IDs must uniquely belong to the corpus")
        if sum(case.abstract_missing for case in self.cases) != 100:
            raise ValueError("1,000-case benchmark must contain exactly 10% missing abstracts")
        if len(self.normal_qwen_ids) != 150 or len(self.stress_qwen_ids) != 300:
            raise ValueError("normal/stress routing must send exactly 15%/30% to Qwen")
        if self.pipeline_components != ("rules", "reranker", "qwen", "schema_validation", "sqlite_commit"):
            raise ValueError("benchmark must execute the complete Stage 2 pipeline")

    def document(self) -> dict[str, object]:
        return {
            "version": self.version, "kind": "benchmark", "corpus_hash": self.corpus_hash,
            "stage2_config_hash": self.stage2_config_hash, "model_lock_hashes": list(self.model_lock_hashes),
            "threshold_artifact_hashes": list(self.threshold_artifact_hashes), "output_token_limit": self.output_token_limit,
            "cases": [{"pair_id": item.pair_id, "input_tokens": item.input_tokens, "abstract_missing": item.abstract_missing} for item in self.cases],
            "normal_qwen_ids": sorted(self.normal_qwen_ids), "stress_qwen_ids": sorted(self.stress_qwen_ids),
            "pipeline_components": list(self.pipeline_components),
        }

    def hash(self) -> str:
        return _hash(self.document())

    def route(self, scenario: str, pair_id: str) -> bool:
        if scenario == "normal":
            return pair_id in self.normal_qwen_ids
        if scenario == "stress":
            return pair_id in self.stress_qwen_ids
        raise ValueError("performance scenario must be normal or stress")


@dataclass(frozen=True, slots=True)
class SoakManifest:
    version: int
    corpus_hash: str
    stage2_config_hash: str
    model_lock_hashes: tuple[str, ...]
    threshold_artifact_hashes: tuple[str, ...]
    output_token_limit: int
    cases: tuple[PerformanceCase, ...]

    def __post_init__(self) -> None:
        if self.version != 1 or len(self.cases) != 10_000 or not self.corpus_hash or not self.stage2_config_hash:
            raise ValueError("soak manifest requires version 1, hashes, and exactly 10,000 cases")
        if len({case.pair_id for case in self.cases}) != len(self.cases):
            raise ValueError("soak pair_ids must be unique")
        if not self.model_lock_hashes or len(set(self.model_lock_hashes)) != len(self.model_lock_hashes) or not all(self.model_lock_hashes):
            raise ValueError("soak manifest requires unique model lock hashes")
        if not self.threshold_artifact_hashes or not all(self.threshold_artifact_hashes) or self.output_token_limit < 1:
            raise ValueError("soak manifest requires thresholds and a positive output cap")

    def document(self) -> dict[str, object]:
        return {
            "version": self.version, "kind": "soak", "corpus_hash": self.corpus_hash,
            "stage2_config_hash": self.stage2_config_hash, "model_lock_hashes": list(self.model_lock_hashes),
            "threshold_artifact_hashes": list(self.threshold_artifact_hashes), "output_token_limit": self.output_token_limit,
            "cases": [[item.pair_id, item.input_tokens, item.abstract_missing] for item in self.cases],
        }

    def hash(self) -> str:
        return _hash(self.document())


@dataclass(frozen=True, slots=True)
class BenchmarkEnvironment:
    machine_model: str
    memory_gb: int
    macos_version: str
    omlx_version: str
    mlx_version: str
    power_mode: str
    background_load: str
    batch_config: Mapping[str, int]
    resident_model_instances: Mapping[str, int]

    def __post_init__(self) -> None:
        required = (self.machine_model, self.macos_version, self.omlx_version, self.mlx_version, self.power_mode, self.background_load)
        if (
            not all(required) or self.machine_model != "Apple Silicon M4 Max" or self.memory_gb != 36
            or not self.batch_config or not self.resident_model_instances
        ):
            raise ValueError("benchmark environment provenance is incomplete")
        if tuple(int(part) for part in self.omlx_version.split(".")[:3]) < (0, 5, 7):
            raise ValueError("benchmark requires oMLX >= 0.5.7")
        if any(not isfinite(value) or value <= 0 for value in self.batch_config.values()):
            raise ValueError("batch/concurrency settings must be positive")
        if any(value != 1 for value in self.resident_model_instances.values()):
            raise ValueError("each model must have exactly one resident instance")
        object.__setattr__(self, "batch_config", MappingProxyType(dict(self.batch_config)))
        object.__setattr__(self, "resident_model_instances", MappingProxyType(dict(self.resident_model_instances)))


def _validate_result_partition(completed: Sequence[str], needs_review: Sequence[str]) -> None:
    if len(completed) != len(set(completed)) or len(needs_review) != len(set(needs_review)):
        raise ValueError("result IDs must be unique within each status")
    if set(completed) & set(needs_review):
        raise ValueError("completed and needs_review results must be mutually exclusive")


@dataclass(frozen=True, slots=True)
class PerformanceRunRecord:
    record_version: int
    scenario: str
    run_id: str
    manifest_hash: str
    stage2_config_hash: str
    model_lock_hashes: tuple[str, ...]
    duration_seconds: float
    p50_seconds: float
    p95_seconds: float
    peak_memory_gb: float
    request_count: int
    failed_request_count: int
    service_request_count: int
    service_failed_request_count: int
    resume_verified: bool
    resume_model_call_count: int
    resumed_pair_count: int
    completed_pair_ids: tuple[str, ...]
    needs_review_pair_ids: tuple[str, ...]
    failed_request_pair_ids: tuple[str, ...]
    qwen_pair_ids: tuple[str, ...]
    environment: BenchmarkEnvironment
    executed_components: tuple[str, ...]
    sqlite_commit_count: int
    warmed: bool
    oom: bool = False
    process_crash: bool = False
    memory_pressure_critical: bool = False
    unbounded_memory_growth: bool = False

    def __post_init__(self) -> None:
        if self.record_version != 2:
            raise ValueError(
                "benchmark record_version 2 is required for audited service-request metrics"
            )
        if self.scenario not in {"normal", "stress"} or not self.run_id or not self.manifest_hash or not self.stage2_config_hash:
            raise ValueError("benchmark record has invalid identity/provenance")
        values = (self.duration_seconds, self.p50_seconds, self.p95_seconds, self.peak_memory_gb)
        if any(not isfinite(value) or value < 0 for value in values):
            raise ValueError("benchmark measurements must be finite and non-negative")
        if self.request_count < 1 or not 0 <= self.failed_request_count <= self.request_count:
            raise ValueError("benchmark request counts are invalid")
        if (
            self.service_request_count < 1
            or not 0 <= self.service_failed_request_count <= self.service_request_count
        ):
            raise ValueError("benchmark service request counts are invalid")
        if (
            type(self.resume_verified) is not bool
            or self.resume_model_call_count < 0
            or not 0 <= self.resumed_pair_count <= self.request_count
        ):
            raise ValueError("benchmark resume measurements are invalid")
        if self.p50_seconds > self.p95_seconds:
            raise ValueError("p50 cannot exceed p95")
        if not self.model_lock_hashes or not all(self.model_lock_hashes) or len(set(self.model_lock_hashes)) != len(self.model_lock_hashes):
            raise ValueError("benchmark model locks must be non-empty and unique")
        _validate_result_partition(self.completed_pair_ids, self.needs_review_pair_ids)
        if len(self.failed_request_pair_ids) != len(set(self.failed_request_pair_ids)):
            raise ValueError("failed request IDs must be unique")
        if self.failed_request_count != len(self.failed_request_pair_ids):
            raise ValueError("failed request count must equal its auditable pair list")
        if len(self.qwen_pair_ids) != len(set(self.qwen_pair_ids)):
            raise ValueError("Qwen result IDs must be unique")

    @property
    def request_failure_rate(self) -> float:
        return self.failed_request_count / self.request_count

    @property
    def service_request_failure_rate(self) -> float:
        return self.service_failed_request_count / self.service_request_count


def _record_matches_manifest(record: PerformanceRunRecord, manifest: PerformanceRoutingManifest) -> bool:
    return (
        record.manifest_hash == manifest.hash()
        and record.stage2_config_hash == manifest.stage2_config_hash
        and record.model_lock_hashes == manifest.model_lock_hashes
    )


def performance_gate(manifest: PerformanceRoutingManifest, records: Sequence[PerformanceRunRecord]) -> GateResult:
    failures: list[str] = []
    expected = {case.pair_id for case in manifest.cases}
    if len({record.run_id for record in records}) != len(records):
        failures.append("benchmark run_ids must be globally unique")
    if records and any(record.environment != records[0].environment for record in records[1:]):
        failures.append("all benchmark runs must use the same frozen environment and batch configuration")
    for scenario, duration_limit in (("normal", 15 * 60), ("stress", 25 * 60)):
        runs = [record for record in records if record.scenario == scenario]
        if len(runs) != 3:
            failures.append(f"{scenario} requires exactly three warmed runs")
            continue
        for record in runs:
            prefix = f"{scenario}/{record.run_id}"
            if not _record_matches_manifest(record, manifest):
                failures.append(f"{prefix} has manifest/config/model provenance mismatch")
            if set(record.environment.resident_model_instances) != set(manifest.model_lock_hashes):
                failures.append(f"{prefix} resident models do not match model locks")
            if record.executed_components != manifest.pipeline_components or record.sqlite_commit_count != len(expected):
                failures.append(f"{prefix} did not execute/commit the complete Stage 2 pipeline")
            if not record.warmed:
                failures.append(f"{prefix} is not a warmed benchmark run")
            if (
                not record.resume_verified
                or record.resume_model_call_count != 0
                or record.resumed_pair_count != len(expected)
            ):
                failures.append(f"{prefix} did not verify zero-call SQLite resume")
            if set(record.completed_pair_ids) | set(record.needs_review_pair_ids) != expected:
                failures.append(f"{prefix} has missing or unknown results")
            if record.request_count != len(expected):
                failures.append(f"{prefix} request count does not match the frozen case set")
            if not set(record.failed_request_pair_ids) <= set(record.needs_review_pair_ids):
                failures.append(f"{prefix} failed request did not route to needs_review")
            qwen = manifest.normal_qwen_ids if scenario == "normal" else manifest.stress_qwen_ids
            if set(record.qwen_pair_ids) != qwen:
                failures.append(f"{prefix} did not use frozen Qwen routing")
            if record.duration_seconds > duration_limit:
                failures.append(f"{prefix} exceeded duration limit")
            if record.peak_memory_gb > 28 or record.memory_pressure_critical:
                failures.append(f"{prefix} exceeded memory limit")
            if record.request_failure_rate >= 0.005:
                failures.append(f"{prefix} request failure rate >= 0.5%")
            if record.service_request_failure_rate >= 0.005:
                failures.append(f"{prefix} service request failure rate >= 0.5%")
            if record.oom or record.process_crash or record.unbounded_memory_growth:
                failures.append(f"{prefix} failed stability gate")
    return GateResult(not failures, tuple(failures))


def performance_summary(records: Sequence[PerformanceRunRecord], scenario: str) -> Mapping[str, float]:
    runs = [record for record in records if record.scenario == scenario]
    if len(runs) != 3:
        raise ValueError("performance summary requires exactly three runs for one scenario")
    return {
        "median_seconds": median(record.duration_seconds for record in runs),
        "p50_seconds": median(record.p50_seconds for record in runs),
        "p95_seconds": median(record.p95_seconds for record in runs),
        "service_request_failure_rate": median(
            record.service_request_failure_rate for record in runs
        ),
    }


@dataclass(frozen=True, slots=True)
class SoakRunRecord:
    record_version: int
    run_id: str
    manifest_hash: str
    stage2_config_hash: str
    model_lock_hashes: tuple[str, ...]
    duration_seconds: float
    peak_memory_gb: float
    request_count: int
    failed_request_count: int
    service_request_count: int
    service_failed_request_count: int
    resume_verified: bool
    resume_model_call_count: int
    resumed_pair_count: int
    completed_pair_ids: tuple[str, ...]
    needs_review_pair_ids: tuple[str, ...]
    failed_request_pair_ids: tuple[str, ...]
    environment: BenchmarkEnvironment
    executed_components: tuple[str, ...]
    sqlite_commit_count: int
    warmed: bool
    oom: bool = False
    process_crash: bool = False
    memory_pressure_critical: bool = False
    unbounded_memory_growth: bool = False

    def __post_init__(self) -> None:
        if self.record_version != 2:
            raise ValueError(
                "soak record_version 2 is required for audited service-request metrics"
            )
        if not self.run_id or not self.manifest_hash or not self.stage2_config_hash:
            raise ValueError("soak record has invalid identity/provenance")
        values = (self.duration_seconds, self.peak_memory_gb)
        if any(not isfinite(value) or value < 0 for value in values):
            raise ValueError("soak measurements must be finite and non-negative")
        if self.request_count < 1 or not 0 <= self.failed_request_count <= self.request_count:
            raise ValueError("soak request counts are invalid")
        if (
            self.service_request_count < 1
            or not 0 <= self.service_failed_request_count <= self.service_request_count
        ):
            raise ValueError("soak service request counts are invalid")
        if (
            type(self.resume_verified) is not bool
            or self.resume_model_call_count < 0
            or not 0 <= self.resumed_pair_count <= self.request_count
        ):
            raise ValueError("soak resume measurements are invalid")
        if not self.model_lock_hashes or not all(self.model_lock_hashes) or len(set(self.model_lock_hashes)) != len(self.model_lock_hashes):
            raise ValueError("soak model locks must be non-empty and unique")
        _validate_result_partition(self.completed_pair_ids, self.needs_review_pair_ids)
        if len(self.failed_request_pair_ids) != len(set(self.failed_request_pair_ids)):
            raise ValueError("failed request IDs must be unique")
        if self.failed_request_count != len(self.failed_request_pair_ids):
            raise ValueError("failed request count must equal its auditable pair list")

    @property
    def request_failure_rate(self) -> float:
        return self.failed_request_count / self.request_count

    @property
    def service_request_failure_rate(self) -> float:
        return self.service_failed_request_count / self.service_request_count


def soak_gate(manifest: SoakManifest, record: SoakRunRecord) -> GateResult:
    failures: list[str] = []
    expected = {case.pair_id for case in manifest.cases}
    if (
        record.manifest_hash != manifest.hash()
        or record.stage2_config_hash != manifest.stage2_config_hash
        or record.model_lock_hashes != manifest.model_lock_hashes
    ):
        failures.append("soak manifest/config/model provenance mismatch")
    if set(record.completed_pair_ids) | set(record.needs_review_pair_ids) != expected:
        failures.append("soak has missing or unknown results")
    if record.request_count != len(expected):
        failures.append("soak request count does not match the frozen case set")
    if set(record.environment.resident_model_instances) != set(manifest.model_lock_hashes):
        failures.append("soak resident models do not match model locks")
    if record.executed_components != ("rules", "reranker", "qwen", "schema_validation", "sqlite_commit") or record.sqlite_commit_count != len(expected):
        failures.append("soak did not execute/commit the complete Stage 2 pipeline")
    if not record.warmed:
        failures.append("soak is not warmed")
    if (
        not record.resume_verified
        or record.resume_model_call_count != 0
        or record.resumed_pair_count != len(expected)
    ):
        failures.append("soak did not verify zero-call SQLite resume")
    if not set(record.failed_request_pair_ids) <= set(record.needs_review_pair_ids):
        failures.append("every failed request must route to needs_review")
    if record.peak_memory_gb > 28 or record.memory_pressure_critical:
        failures.append("soak exceeded memory limit")
    if record.request_failure_rate >= 0.005:
        failures.append("soak request failure rate >= 0.5%")
    if record.service_request_failure_rate >= 0.005:
        failures.append("soak service request failure rate >= 0.5%")
    if record.oom or record.process_crash or record.unbounded_memory_growth:
        failures.append("soak failed stability gate")
    return GateResult(not failures, tuple(failures))


class ReplayError(StrEnum):
    NONE = "none"
    SCHEMA = "schema"
    TIMEOUT = "timeout"
    SERVICE = "service"


@dataclass(frozen=True, slots=True)
class StructuredReplayManifest:
    version: int
    pair_ids: tuple[str, ...]
    corpus_hash: str
    stage2_config_hash: str
    model_lock_hash: str
    prompt_hash: str
    schema_hash: str

    def __post_init__(self) -> None:
        hashes = (self.corpus_hash, self.stage2_config_hash, self.model_lock_hash, self.prompt_hash, self.schema_hash)
        if self.version != 1 or len(self.pair_ids) < 1_000 or len(set(self.pair_ids)) != len(self.pair_ids) or not all(hashes):
            raise ValueError("structured replay manifest requires >=1,000 unique requests and complete provenance")

    def document(self) -> dict[str, object]:
        return {
            "version": self.version, "pair_ids": list(self.pair_ids), "corpus_hash": self.corpus_hash,
            "stage2_config_hash": self.stage2_config_hash, "model_lock_hash": self.model_lock_hash,
            "prompt_hash": self.prompt_hash, "schema_hash": self.schema_hash,
        }

    def hash(self) -> str:
        return _hash(self.document())


@dataclass(frozen=True, slots=True)
class StructuredReplayRecord:
    pair_id: str
    manifest_hash: str
    first_error: ReplayError
    first_returned_pair_id: str | None
    first_schema_outside_text: bool
    first_think_tag_leak: bool
    deterministic_repairs: int
    model_retries: int
    retry_error: ReplayError | None
    final_valid: bool
    final_returned_pair_id: str | None
    final_schema_outside_text: bool
    final_think_tag_leak: bool
    final_decision: Stage2Decision

    def __post_init__(self) -> None:
        if not isinstance(self.first_error, ReplayError):
            object.__setattr__(self, "first_error", ReplayError(self.first_error))
        if self.retry_error is not None and not isinstance(self.retry_error, ReplayError):
            object.__setattr__(self, "retry_error", ReplayError(self.retry_error))
        if not isinstance(self.final_decision, Stage2Decision):
            object.__setattr__(self, "final_decision", Stage2Decision(self.final_decision))
        if not self.pair_id or not self.manifest_hash or self.deterministic_repairs not in {0, 1} or self.model_retries not in {0, 1}:
            raise ValueError("structured replay allows at most one repair and one model retry")
        if (self.model_retries == 0) != (self.retry_error is None):
            raise ValueError("retry_error must be recorded exactly when a model retry occurs")
        if self.first_error in {ReplayError.TIMEOUT, ReplayError.SERVICE} and self.first_returned_pair_id is not None:
            raise ValueError("timeout/service errors cannot claim a returned pair_id")
        if self.final_valid and self.final_returned_pair_id is None:
            raise ValueError("valid final response requires a returned pair_id")
        if not self.final_valid and self.final_decision is not Stage2Decision.NEEDS_REVIEW:
            raise ValueError("every final invalid response must route to needs_review")
        first_valid = (
            self.first_error is ReplayError.NONE
            and self.first_returned_pair_id == self.pair_id
            and not self.first_schema_outside_text
            and not self.first_think_tag_leak
        )
        recovery_actions = self.deterministic_repairs + self.model_retries
        if first_valid and recovery_actions:
            raise ValueError("a valid first response cannot claim a repair or retry")
        if first_valid and not self.final_valid:
            raise ValueError("a valid first response cannot become a final invalid response")
        if not first_valid and self.final_valid and not recovery_actions:
            raise ValueError("a recovered valid response must record its repair or retry")

    def document(self) -> dict[str, object]:
        return {
            "pair_id": self.pair_id,
            "manifest_hash": self.manifest_hash,
            "first_error": self.first_error.value,
            "first_returned_pair_id": self.first_returned_pair_id,
            "first_schema_outside_text": self.first_schema_outside_text,
            "first_think_tag_leak": self.first_think_tag_leak,
            "deterministic_repairs": self.deterministic_repairs,
            "model_retries": self.model_retries,
            "retry_error": self.retry_error.value if self.retry_error is not None else None,
            "final_valid": self.final_valid,
            "final_returned_pair_id": self.final_returned_pair_id,
            "final_schema_outside_text": self.final_schema_outside_text,
            "final_think_tag_leak": self.final_think_tag_leak,
            "final_decision": self.final_decision.value,
        }


@dataclass(frozen=True, slots=True)
class StructuredReplayResult:
    manifest_hash: str
    gate: GateResult
    first_valid_rate: float
    schema_errors: int
    timeouts: int
    service_errors: int
    deterministic_repairs: int
    model_retries: int
    retry_error_counts: Mapping[ReplayError, int]


def structured_replay_gate(manifest: StructuredReplayManifest, records: Sequence[StructuredReplayRecord]) -> StructuredReplayResult:
    expected = set(manifest.pair_ids)
    manifest_hash = manifest.hash()
    by_id = {item.pair_id: item for item in records}
    if len(by_id) != len(records) or set(by_id) != expected:
        raise ValueError("structured replay must exactly cover its frozen request universe")
    if any(item.manifest_hash != manifest_hash for item in records):
        raise ValueError("structured replay record provenance mismatch")
    failures: list[str] = []
    first_valid = sum(
        item.first_error is ReplayError.NONE
        and item.first_returned_pair_id == item.pair_id
        and not item.first_schema_outside_text
        and not item.first_think_tag_leak
        for item in records
    )
    valid_rate = first_valid / len(records)
    if valid_rate < 0.995:
        failures.append("first-response JSON/schema validity < 99.5%")
    if any(item.first_returned_pair_id not in {None, item.pair_id} or item.final_returned_pair_id not in {None, item.pair_id} for item in records):
        failures.append("paper_id fidelity < 100%")
    if any(item.first_schema_outside_text or item.final_schema_outside_text for item in records):
        failures.append("schema-external text leakage is non-zero")
    if any(item.first_think_tag_leak or item.final_think_tag_leak for item in records):
        failures.append("think-tag leakage is non-zero")
    if any(not item.final_valid and item.final_decision is not Stage2Decision.NEEDS_REVIEW for item in records):
        failures.append("final invalid response did not route to needs_review")
    if any(item.final_valid and item.final_returned_pair_id != item.pair_id for item in records):
        failures.append("a final valid response has the wrong paper_id")
    return StructuredReplayResult(
        manifest_hash, GateResult(not failures, tuple(failures)), valid_rate,
        sum(item.first_error is ReplayError.SCHEMA for item in records),
        sum(item.first_error is ReplayError.TIMEOUT for item in records),
        sum(item.first_error is ReplayError.SERVICE for item in records),
        sum(item.deterministic_repairs for item in records), sum(item.model_retries for item in records),
        Counter(item.retry_error for item in records if item.retry_error is not None),
    )


class RationaleStratum(StrEnum):
    RELEVANT = "relevant"
    BOUNDARY = "boundary"


@dataclass(frozen=True, slots=True)
class RationaleAuditCase:
    pair_id: str
    stratum: RationaleStratum
    language: str
    rationale_artifact_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.stratum, RationaleStratum):
            object.__setattr__(self, "stratum", RationaleStratum(self.stratum))
        if not self.pair_id or not self.language or not self.rationale_artifact_hash:
            raise ValueError("rationale audit case requires frozen pair/stratum/language/artifact")


@dataclass(frozen=True, slots=True)
class RationaleAuditManifest:
    version: int
    cases: tuple[RationaleAuditCase, ...]
    corpus_hash: str
    model_lock_hash: str
    evidence_rubric_hash: str
    fabrication_rubric_hash: str

    def __post_init__(self) -> None:
        pair_ids = [item.pair_id for item in self.cases]
        if self.version != 1 or len(pair_ids) < 100 or len(set(pair_ids)) != len(pair_ids):
            raise ValueError("rationale audit requires at least 100 unique frozen pairs")
        if not all((self.corpus_hash, self.model_lock_hash, self.evidence_rubric_hash, self.fabrication_rubric_hash)):
            raise ValueError("rationale audit requires corpus/model/rubric provenance")
        languages = {item.language for item in self.cases}
        if len(languages) < 2:
            raise ValueError("rationale audit requires at least two frozen language strata")
        cells = Counter((item.stratum, item.language) for item in self.cases)
        if any(cells[(stratum, language)] < 10 for stratum in RationaleStratum for language in languages):
            raise ValueError("each rationale stratum/language cell requires at least 10 samples")

    def document(self) -> dict[str, object]:
        return {
            "version": self.version,
            "cases": [[item.pair_id, item.stratum.value, item.language, item.rationale_artifact_hash] for item in self.cases],
            "corpus_hash": self.corpus_hash, "model_lock_hash": self.model_lock_hash,
            "evidence_rubric_hash": self.evidence_rubric_hash,
            "fabrication_rubric_hash": self.fabrication_rubric_hash,
        }

    def hash(self) -> str:
        return _hash(self.document())


@dataclass(frozen=True, slots=True)
class RationaleAuditRecord:
    pair_id: str
    manifest_hash: str
    evidence_supported: bool
    severe_fabrication: bool

    def __post_init__(self) -> None:
        if not self.pair_id or not self.manifest_hash:
            raise ValueError("rationale audit record requires pair_id and manifest hash")

    def document(self) -> dict[str, object]:
        return {
            "pair_id": self.pair_id,
            "manifest_hash": self.manifest_hash,
            "evidence_supported": self.evidence_supported,
            "severe_fabrication": self.severe_fabrication,
        }


def rationale_audit_gate(manifest: RationaleAuditManifest, records: Sequence[RationaleAuditRecord]) -> GateResult:
    by_id = {item.pair_id: item for item in records}
    if len(by_id) != len(records) or set(by_id) != {item.pair_id for item in manifest.cases}:
        raise ValueError("rationale audit must exactly cover its frozen manifest")
    manifest_hash = manifest.hash()
    if any(item.manifest_hash != manifest_hash for item in records):
        raise ValueError("rationale audit record provenance mismatch")
    support_rate = sum(item.evidence_supported for item in records) / len(records)
    fabrication_rate = sum(item.severe_fabrication for item in records) / len(records)
    failures = []
    if support_rate < 0.95:
        failures.append("rationale evidence support < 95%")
    if fabrication_rate > 0.01:
        failures.append("rationale severe fabrication > 1%")
    return GateResult(not failures, tuple(failures))


@dataclass(frozen=True, slots=True)
class ParityManifest:
    version: int
    pair_ids: tuple[str, ...]
    corpus_hash: str
    tokenizer_hash: str
    preprocess_hash: str
    oracle_model_lock_hash: str
    candidate_model_lock_hash: str
    oracle_threshold_artifact_hash: str
    candidate_threshold_artifact_hash: str
    dev_manifest_hash: str
    window_selector_hash: str
    oracle_low: float
    oracle_high: float
    candidate_low: float
    candidate_high: float
    low_window_pair_ids: frozenset[str]
    high_window_pair_ids: frozenset[str]
    low_window_definition: str
    high_window_definition: str

    def __post_init__(self) -> None:
        ids = set(self.pair_ids)
        hashes = (
            self.corpus_hash, self.tokenizer_hash, self.preprocess_hash, self.oracle_model_lock_hash,
            self.candidate_model_lock_hash, self.oracle_threshold_artifact_hash,
            self.candidate_threshold_artifact_hash, self.dev_manifest_hash, self.window_selector_hash,
        )
        if self.version != 1 or len(self.pair_ids) < 10_000 or len(ids) != len(self.pair_ids) or not all(hashes):
            raise ValueError("parity manifest requires at least 10,000 unique pairs and complete provenance")
        if (
            not self.low_window_pair_ids or not self.high_window_pair_ids
            or not self.low_window_pair_ids <= ids or not self.high_window_pair_ids <= ids
            or not self.low_window_definition or not self.high_window_definition
        ):
            raise ValueError("parity manifest must freeze non-empty low/high windows and denominators")
        thresholds = (self.oracle_low, self.oracle_high, self.candidate_low, self.candidate_high)
        if not all(isfinite(item) for item in thresholds) or not self.oracle_low < self.oracle_high or not self.candidate_low < self.candidate_high:
            raise ValueError("parity low/high thresholds must be finite and ordered")

    def document(self) -> dict[str, object]:
        return {
            "version": self.version, "pair_ids": list(self.pair_ids), "corpus_hash": self.corpus_hash,
            "tokenizer_hash": self.tokenizer_hash, "preprocess_hash": self.preprocess_hash,
            "oracle_model_lock_hash": self.oracle_model_lock_hash,
            "candidate_model_lock_hash": self.candidate_model_lock_hash,
            "oracle_threshold_artifact_hash": self.oracle_threshold_artifact_hash,
            "candidate_threshold_artifact_hash": self.candidate_threshold_artifact_hash,
            "dev_manifest_hash": self.dev_manifest_hash, "window_selector_hash": self.window_selector_hash,
            "oracle_low": self.oracle_low, "oracle_high": self.oracle_high,
            "candidate_low": self.candidate_low, "candidate_high": self.candidate_high,
            "low_window_pair_ids": sorted(self.low_window_pair_ids),
            "high_window_pair_ids": sorted(self.high_window_pair_ids),
            "low_window_definition": self.low_window_definition,
            "high_window_definition": self.high_window_definition,
        }

    def hash(self) -> str:
        return _hash(self.document())


@dataclass(frozen=True, slots=True)
class ParityScore:
    pair_id: str
    manifest_hash: str
    oracle_score: float
    candidate_score: float

    def __post_init__(self) -> None:
        if not self.pair_id or not self.manifest_hash or not isfinite(self.oracle_score) or not isfinite(self.candidate_score):
            raise ValueError("parity scores require pair_id and finite values")

    def document(self) -> dict[str, object]:
        return {
            "pair_id": self.pair_id,
            "manifest_hash": self.manifest_hash,
            "oracle_score": self.oracle_score,
            "candidate_score": self.candidate_score,
        }


@dataclass(frozen=True, slots=True)
class ParityResult:
    manifest_hash: str
    gate: GateResult
    kendall_tau_b: float
    low_threshold_agreement: float
    high_threshold_agreement: float
    low_threshold_denominator: int
    high_threshold_denominator: int


def _tie_pairs(values: Sequence[float]) -> int:
    return sum(count * (count - 1) // 2 for count in Counter(values).values())


def kendall_tau_b(left: Sequence[float], right: Sequence[float]) -> float:
    """Compute Kendall tau-b in O(n log n), including ties in both rankings."""

    if len(left) != len(right) or len(left) < 2 or any(not isfinite(value) for value in (*left, *right)):
        raise ValueError("Kendall tau-b requires equal finite sequences of length >= 2")
    y_values = {value: index + 1 for index, value in enumerate(sorted(set(right)))}
    tree = [0] * (len(y_values) + 1)

    def add(index: int) -> None:
        while index < len(tree):
            tree[index] += 1
            index += index & -index

    def prefix(index: int) -> int:
        total = 0
        while index:
            total += tree[index]
            index -= index & -index
        return total

    rows = sorted(zip(left, right, strict=True), key=lambda item: (item[0], item[1]))
    discordant = seen = 0
    start = 0
    while start < len(rows):
        end = start
        while end < len(rows) and rows[end][0] == rows[start][0]:
            end += 1
        for _, right_value in rows[start:end]:
            rank = y_values[right_value]
            discordant += seen - prefix(rank)
        for _, right_value in rows[start:end]:
            add(y_values[right_value])
            seen += 1
        start = end
    total_pairs = len(left) * (len(left) - 1) // 2
    ties_left = _tie_pairs(left)
    ties_right = _tie_pairs(right)
    ties_both = sum(count * (count - 1) // 2 for count in Counter(zip(left, right, strict=True)).values())
    comparable = total_pairs - ties_left - ties_right + ties_both
    denominator = sqrt((total_pairs - ties_left) * (total_pairs - ties_right))
    return 0.0 if denominator == 0 else (comparable - 2 * discordant) / denominator


def parity_gate(manifest: ParityManifest, scores: Sequence[ParityScore]) -> ParityResult:
    by_id = {item.pair_id: item for item in scores}
    if len(by_id) != len(scores) or set(by_id) != set(manifest.pair_ids):
        raise ValueError("parity scores must exactly cover the frozen pair universe")
    manifest_hash = manifest.hash()
    if any(item.manifest_hash != manifest_hash for item in scores):
        raise ValueError("parity score provenance mismatch")
    ordered = [by_id[pair_id] for pair_id in manifest.pair_ids]
    tau = kendall_tau_b([item.oracle_score for item in ordered], [item.candidate_score for item in ordered])
    low_window = [by_id[pair_id] for pair_id in manifest.low_window_pair_ids]
    high_window = [by_id[pair_id] for pair_id in manifest.high_window_pair_ids]
    low_agreement = sum(
        (item.oracle_score <= manifest.oracle_low) == (item.candidate_score <= manifest.candidate_low)
        for item in low_window
    ) / len(low_window)
    high_agreement = sum(
        (item.oracle_score >= manifest.oracle_high) == (item.candidate_score >= manifest.candidate_high)
        for item in high_window
    ) / len(high_window)
    failures = []
    if tau < 0.995:
        failures.append("Kendall tau-b < 0.995")
    if low_agreement < 0.995:
        failures.append("low-threshold classification agreement < 99.5%")
    if high_agreement < 0.995:
        failures.append("high-threshold classification agreement < 99.5%")
    return ParityResult(
        manifest_hash, GateResult(not failures, tuple(failures)), tau, low_agreement, high_agreement,
        len(low_window), len(high_window),
    )


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    point: float
    lower: float
    upper: float


@dataclass(frozen=True, slots=True)
class PairedBootstrapResult:
    manifest_hash: str
    split: GoldSplit
    incumbent_candidate_id: str
    challenger_candidate_id: str
    retention_delta: BootstrapInterval
    positive_f1_delta: BootstrapInterval
    incumbent_retention_wilson: WilsonInterval | None
    challenger_retention_wilson: WilsonInterval | None
    iterations: int
    seed: int

    @property
    def in_tie_band(self) -> bool:
        return abs(self.retention_delta.point) <= 0.01 and abs(self.positive_f1_delta.point) <= 0.02


def _resampled_metrics(predictions: Sequence[Prediction], gold: Sequence[int], indices: Sequence[int]) -> tuple[float, float]:
    positives = sum(gold[index] >= 2 for index in indices)
    tp = sum(predictions[index].decision is Stage2Decision.RELEVANT and gold[index] >= 2 for index in indices)
    fp = sum(predictions[index].decision is Stage2Decision.RELEVANT and gold[index] < 2 for index in indices)
    review_positive = sum(predictions[index].decision is Stage2Decision.NEEDS_REVIEW and gold[index] >= 2 for index in indices)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / positives if positives else 0.0
    retention = (tp + review_positive) / positives if positives else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return retention, f1


def paired_bootstrap_comparison(
    pairs: Sequence[GoldPair],
    labels: GoldLabelStore,
    incumbent: Sequence[Prediction],
    challenger: Sequence[Prediction],
    *,
    iterations: int = 2_000,
    seed: int = 0,
) -> PairedBootstrapResult:
    if not pairs or iterations < 100:
        raise ValueError("paired bootstrap requires pairs and at least 100 iterations")
    splits = {pair.split for pair in pairs}
    if len(splits) != 1:
        raise ValueError("paired bootstrap cannot mix gold-set splits")
    pair_ids = [pair.pair_id for pair in pairs]
    incumbent_by_id = {item.pair_id: item for item in incumbent}
    challenger_by_id = {item.pair_id: item for item in challenger}
    if len(incumbent_by_id) != len(incumbent) or len(challenger_by_id) != len(challenger):
        raise ValueError("paired bootstrap predictions must be unique")
    if set(incumbent_by_id) != set(pair_ids) or set(challenger_by_id) != set(pair_ids):
        raise ValueError("paired bootstrap predictions must cover the same frozen pairs")
    manifests = {item.manifest_hash for item in (*incumbent, *challenger)}
    incumbent_candidates = {item.candidate_id for item in incumbent}
    challenger_candidates = {item.candidate_id for item in challenger}
    if len(manifests) != 1 or len(incumbent_candidates) != 1 or len(challenger_candidates) != 1:
        raise ValueError("paired bootstrap requires bound manifest and candidate provenance")
    incumbent_ordered = [incumbent_by_id[pair_id] for pair_id in pair_ids]
    challenger_ordered = [challenger_by_id[pair_id] for pair_id in pair_ids]
    gold = [labels.labels[pair_id] for pair_id in pair_ids]
    all_indices = tuple(range(len(pairs)))
    incumbent_point = _resampled_metrics(incumbent_ordered, gold, all_indices)
    challenger_point = _resampled_metrics(challenger_ordered, gold, all_indices)
    random = Random(seed)
    retention_deltas: list[float] = []
    f1_deltas: list[float] = []
    for _ in range(iterations):
        indices = [random.randrange(len(pairs)) for _ in pairs]
        incumbent_metrics = _resampled_metrics(incumbent_ordered, gold, indices)
        challenger_metrics = _resampled_metrics(challenger_ordered, gold, indices)
        retention_deltas.append(challenger_metrics[0] - incumbent_metrics[0])
        f1_deltas.append(challenger_metrics[1] - incumbent_metrics[1])
    retention_deltas.sort()
    f1_deltas.sort()
    lower = int(0.025 * iterations)
    upper = min(iterations - 1, int(0.975 * iterations))
    incumbent_metrics = measure_predictions(pairs, labels, incumbent)
    challenger_metrics = measure_predictions(pairs, labels, challenger)
    return PairedBootstrapResult(
        next(iter(manifests)), next(iter(splits)), next(iter(incumbent_candidates)), next(iter(challenger_candidates)),
        BootstrapInterval(challenger_point[0] - incumbent_point[0], retention_deltas[lower], retention_deltas[upper]),
        BootstrapInterval(challenger_point[1] - incumbent_point[1], f1_deltas[lower], f1_deltas[upper]),
        incumbent_metrics.retention_interval, challenger_metrics.retention_interval,
        iterations, seed,
    )


@dataclass(frozen=True, slots=True)
class WinnerResult:
    replace_incumbent: bool
    reason: str


@dataclass(frozen=True, slots=True)
class ReleaseGateResult:
    candidate_id: str
    evaluation_manifest_hash: str
    artifact_hashes: Mapping[str, str]
    gate: GateResult
    throughput_runs: tuple[float, float, float]

    def __post_init__(self) -> None:
        required = {"promotion", "structured_replay", "rationale", "parity", "benchmark", "soak"}
        if (
            not self.candidate_id or not self.evaluation_manifest_hash or set(self.artifact_hashes) != required
            or not all(self.artifact_hashes.values())
        ):
            raise ValueError("release result must bind all Phase 3 gate artifacts")
        if any(not isfinite(value) or value <= 0 for value in self.throughput_runs):
            raise ValueError("release throughput runs must be finite and positive")


def phase3_release_gate(
    *,
    candidate_id: str,
    evaluation_manifest_hash: str,
    artifacts: Mapping[str, tuple[str, GateResult]],
    throughput_runs: tuple[float, float, float],
) -> ReleaseGateResult:
    required = {"promotion", "structured_replay", "rationale", "parity", "benchmark", "soak"}
    if set(artifacts) != required:
        raise ValueError("Phase 3 release gate requires every quality/performance artifact")
    failures = tuple(
        f"{name}: {failure}"
        for name, (_, gate) in artifacts.items()
        for failure in gate.failures
    )
    result = GateResult(not failures, failures)
    return ReleaseGateResult(
        candidate_id, evaluation_manifest_hash, MappingProxyType({name: value[0] for name, value in artifacts.items()}),
        result, throughput_runs,
    )


def winner_gate(
    incumbent: ReleaseGateResult,
    challenger: ReleaseGateResult,
    comparisons: Mapping[GoldSplit, PairedBootstrapResult],
) -> WinnerResult:
    if set(comparisons) != {GoldSplit.HIDDEN_HARD, GoldSplit.HIDDEN_REAL}:
        raise ValueError("winner gate requires separate hard and real paired comparisons")
    for split, comparison in comparisons.items():
        if (
            comparison.split is not split
            or comparison.manifest_hash != incumbent.evaluation_manifest_hash
            or comparison.manifest_hash != challenger.evaluation_manifest_hash
            or comparison.incumbent_candidate_id != incumbent.candidate_id
            or comparison.challenger_candidate_id != challenger.candidate_id
        ):
            raise ValueError("winner comparison provenance does not match release artifacts")
    if not challenger.gate.passed:
        return WinnerResult(False, "challenger failed a Phase 3 release gate")
    if not incumbent.gate.passed:
        return WinnerResult(True, "challenger is the only release-qualified cascade")
    if all(item.in_tie_band for item in comparisons.values()):
        speedup = median(challenger.throughput_runs) / median(incumbent.throughput_runs)
        return WinnerResult(
            speedup >= 1.2,
            "tie-band challenger meets 20% speed gate" if speedup >= 1.2 else "tie-band challenger is not 20% faster",
        )
    deltas = [
        (comparison.retention_delta, 0.01) for comparison in comparisons.values()
    ] + [
        (comparison.positive_f1_delta, 0.02) for comparison in comparisons.values()
    ]
    not_materially_worse = all(interval.point >= -band for interval, band in deltas)
    statistically_better = any(interval.point > band and interval.lower > 0 for interval, band in deltas)
    return WinnerResult(
        not_materially_worse and statistically_better,
        "challenger wins quality outside tie band with paired support"
        if not_materially_worse and statistically_better
        else "challenger lacks supported quality improvement without material regression",
    )
