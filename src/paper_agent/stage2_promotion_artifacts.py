"""Strict private artifact loading for one-shot Stage 2 promotion evaluation.

The input files handled here contain hidden labels or raw model predictions.
They are evaluator-custody inputs only: this module intentionally provides no
template writer and returns only commitment hashes and gate summaries after an
evaluation has consumed the hidden holdout marker.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .canonical import content_hash
from .stage2_evaluation import (
    CalibrationPath,
    CandidateModelArtifacts,
    GoldLabelStore,
    GoldManifest,
    GoldSplit,
    Prediction,
    PromotionEvaluator,
    PromotionSubmission,
    ReviewReason,
    Stage2Decision,
    load_gold_manifest,
    promotion_gate,
)
from .stage2_hidden_attestation import HIDDEN_PROMOTION_GATE_POLICY_HASH
from .schema import SchemaValidationError, validate


class PrivatePromotionArtifactError(ValueError):
    """A private evaluator input is malformed, unbound, or internally unsafe."""


@dataclass(frozen=True, slots=True)
class PromotionSigningInput:
    """Public-safe, unsigned fields for one candidate's attestation payload."""

    candidate_id: str
    evaluator_id: str
    evaluation_manifest_hash: str
    evaluation_run_id: str
    stage2_config_hash: str
    model_lock_hashes: Mapping[str, str]
    calibrator_hashes: Mapping[str, str]
    threshold_hashes: Mapping[str, str]
    hidden_pair_universe_hashes: Mapping[str, str]
    hidden_split_pair_counts: Mapping[str, int]
    prediction_submission_hash: str
    promotion_marker_hash: str
    consumed_hidden_splits: tuple[str, str]
    gate_policy_hash: str
    passed: bool
    failures: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_lock_hashes", MappingProxyType(dict(self.model_lock_hashes)))
        object.__setattr__(self, "calibrator_hashes", MappingProxyType(dict(self.calibrator_hashes)))
        object.__setattr__(self, "threshold_hashes", MappingProxyType(dict(self.threshold_hashes)))
        object.__setattr__(self, "hidden_pair_universe_hashes", MappingProxyType(dict(self.hidden_pair_universe_hashes)))
        object.__setattr__(self, "hidden_split_pair_counts", MappingProxyType(dict(self.hidden_split_pair_counts)))

    def document(self) -> dict[str, Any]:
        """Return unsigned public-safe fields, with no labels or raw predictions."""

        return {
            "candidate_id": self.candidate_id,
            "evaluation_manifest_hash": self.evaluation_manifest_hash,
            "evaluation_run_id": self.evaluation_run_id,
            "stage2_config_hash": self.stage2_config_hash,
            "model_lock_hashes": dict(self.model_lock_hashes),
            "calibrator_hashes": dict(self.calibrator_hashes),
            "threshold_hashes": dict(self.threshold_hashes),
            "hidden_pair_universe_hashes": dict(self.hidden_pair_universe_hashes),
            "hidden_split_pair_counts": dict(self.hidden_split_pair_counts),
            "prediction_submission_hash": self.prediction_submission_hash,
            "promotion_marker_hash": self.promotion_marker_hash,
            "consumed_hidden_splits": list(self.consumed_hidden_splits),
            "gate_policy_hash": self.gate_policy_hash,
            "result_summary": {
                "passed": self.passed,
                "failures": list(self.failures),
                "gate_versions": {"promotion": "1", "determinism": "1"},
            },
        }

    def attestation_payload(
        self,
        *,
        evaluator_key_id: str,
        trust_manifest_hash: str,
        issued_at: str,
    ) -> dict[str, Any]:
        """Build a complete public-safe payload for the separate signing step."""

        payload = {
            "schema_version": "1",
            "attestation_type": "stage2-hidden-promotion",
            "evaluator_key_id": evaluator_key_id,
            "evaluator_id": self.evaluator_id,
            "trust_manifest_hash": trust_manifest_hash,
            "issued_at": issued_at,
            **self.document(),
        }
        _validate(payload, "stage2-hidden-evaluator-signing-input.schema.json")
        return payload


@dataclass(frozen=True, slots=True)
class PromotionEvaluationInputs:
    """Public-safe output of one sealed promotion batch evaluation."""

    evaluation_manifest_hash: str
    evaluation_run_id: str
    incumbent_candidate_id: str
    promotion_marker_hash: str
    candidates: Mapping[str, PromotionSigningInput]

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", MappingProxyType(dict(self.candidates)))


def private_gold_labels_from_document(document: Mapping[str, Any], *, manifest: GoldManifest) -> GoldLabelStore:
    """Validate and reconstruct a complete private ``GoldLabelStore``."""

    _validate(document, "stage2-private-gold-labels.schema.json")
    if document["gold_manifest_hash"] != manifest.hash():
        raise PrivatePromotionArtifactError("private labels do not bind the supplied gold manifest")
    labels: dict[str, int] = {}
    for row in document["labels"]:
        pair_id = row["pair_id"]
        if pair_id in labels:
            raise PrivatePromotionArtifactError("private labels contain a duplicate pair_id")
        labels[pair_id] = row["label"]
    try:
        result = GoldLabelStore(
            labels,
            document["annotation_artifact_hash"],
            frozenset(document["hard_negative_pair_ids"]),
            frozenset(document["hard_positive_pair_ids"]),
        )
        manifest.validate(result)
    except ValueError as error:
        raise PrivatePromotionArtifactError(f"private gold labels are invalid: {error}") from error
    return result


def load_private_gold_labels(path: Path, *, manifest: GoldManifest) -> GoldLabelStore:
    """Load the strictly versioned private label artifact for ``manifest``."""

    return private_gold_labels_from_document(_read_object(path, "private gold labels"), manifest=manifest)


def prediction_document(prediction: Prediction) -> dict[str, Any]:
    """Return the exact private interchange representation of a prediction."""

    return {
        "pair_id": prediction.pair_id,
        "candidate_id": prediction.candidate_id,
        "decision": prediction.decision.value,
        "raw_score": prediction.raw_score,
        "probability": prediction.probability,
        "path": prediction.path.value,
        "calibrator_hash": prediction.calibrator_hash,
        "threshold_hash": prediction.threshold_hash,
        "model_lock_hash": prediction.model_lock_hash,
        "manifest_hash": prediction.manifest_hash,
        "stage2_config_hash": prediction.stage2_config_hash,
        "inference_artifact_hash": prediction.inference_artifact_hash,
        "review_reason": prediction.review_reason.value if prediction.review_reason is not None else None,
    }


def promotion_submission_document(submission: PromotionSubmission) -> dict[str, Any]:
    """Return an exact private interchange document without writing a fixture."""

    return {
        "schema_version": "1",
        "candidate_id": submission.candidate_id,
        "runs": [[prediction_document(prediction) for prediction in run] for run in submission.runs],
    }


def promotion_submission_from_document(document: Mapping[str, Any], *, manifest: GoldManifest) -> PromotionSubmission:
    """Validate and reconstruct an exact three-run hidden prediction submission."""

    _validate(document, "stage2-promotion-submission.schema.json")
    runs = tuple(tuple(_prediction_from_document(item) for item in run) for run in document["runs"])
    try:
        submission = PromotionSubmission(document["candidate_id"], runs)
    except ValueError as error:
        raise PrivatePromotionArtifactError(f"promotion submission is invalid: {error}") from error
    expected_pair_ids = {pair.pair_id for pair in manifest.pairs if pair.split is not GoldSplit.DEV}
    for run in submission.runs:
        if len({prediction.pair_id for prediction in run}) != len(run) or {prediction.pair_id for prediction in run} != expected_pair_ids:
            raise PrivatePromotionArtifactError("every promotion submission run must exactly cover the hidden pair universe")
        if any(prediction.manifest_hash != manifest.hash() for prediction in run):
            raise PrivatePromotionArtifactError("promotion submission predictions do not bind the supplied gold manifest")
    return submission


def load_promotion_submission(path: Path, *, manifest: GoldManifest) -> PromotionSubmission:
    """Load a strictly versioned three-run private prediction submission."""

    return promotion_submission_from_document(_read_object(path, "promotion submission"), manifest=manifest)


def candidate_artifacts_from_v2_bundle(path: Path) -> CandidateModelArtifacts:
    """Derive candidate provenance from a frozen v2 bundle, never a submission."""

    from .stage2_search import Stage2ReleaseError, load_stage2_benchmark_candidate

    try:
        released = load_stage2_benchmark_candidate(path)
        profile = released.profile
        reranker = profile.reranker_calibration
        qwen = profile.adjudicator_calibration
        if reranker is None or qwen is None:
            raise PrivatePromotionArtifactError("v2 candidate bundle has no probability calibrations")
        return CandidateModelArtifacts(
            released.profile_name,
            {CalibrationPath.RERANKER: reranker.calibrator, CalibrationPath.QWEN: qwen.calibrator},
            {CalibrationPath.RERANKER: reranker.threshold, CalibrationPath.QWEN: qwen.threshold},
        )
    except (OSError, Stage2ReleaseError, ValueError) as error:
        if isinstance(error, PrivatePromotionArtifactError):
            raise
        raise PrivatePromotionArtifactError(f"v2 candidate bundle is invalid: {error}") from error


def run_promotion_evaluation(
    *,
    manifest_path: Path,
    private_labels_path: Path,
    submission_paths: Mapping[str, Path],
    candidate_bundle_paths: Mapping[str, Path],
    evaluator_id: str,
    state_root: Path,
    incumbent_candidate_id: str,
    evaluation_run_id: str,
    bootstrap_iterations: int = 2_000,
    bootstrap_seed: int = 0,
) -> PromotionEvaluationInputs:
    """Run one sealed batch and return only public-safe unsigned inputs.

    Gate failures are returned only after ``PromotionEvaluator`` writes its
    consumed marker. Invalid input/provenance remains a pre-evaluation error,
    preserving the evaluator's existing marker semantics.
    """

    manifest = load_gold_manifest(manifest_path)
    labels = load_private_gold_labels(private_labels_path, manifest=manifest)
    candidate_artifacts = {
        candidate_id: candidate_artifacts_from_v2_bundle(candidate_path)
        for candidate_id, candidate_path in candidate_bundle_paths.items()
    }
    if set(candidate_artifacts) != set(submission_paths):
        raise PrivatePromotionArtifactError("candidate bundles and promotion submissions must name the same candidates")
    if any(candidate_id != artifact.candidate_id for candidate_id, artifact in candidate_artifacts.items()):
        raise PrivatePromotionArtifactError("candidate bundle mapping keys must match frozen v2 profile names")
    submissions = {
        candidate_id: load_promotion_submission(submission_path, manifest=manifest)
        for candidate_id, submission_path in submission_paths.items()
    }
    if any(candidate_id != submission.candidate_id for candidate_id, submission in submissions.items()):
        raise PrivatePromotionArtifactError("promotion submission mapping keys must match their candidate_id")
    if incumbent_candidate_id not in submissions:
        raise PrivatePromotionArtifactError("the incumbent must have a promotion submission")
    try:
        batch = PromotionEvaluator(
            manifest, labels, candidate_artifacts, evaluator_id, state_root
        ).evaluate_candidates(
            tuple(submissions.values()),
            incumbent_candidate_id=incumbent_candidate_id,
            evaluation_run_id=evaluation_run_id,
            bootstrap_iterations=bootstrap_iterations,
            bootstrap_seed=bootstrap_seed,
        )
    except ValueError as error:
        raise PrivatePromotionArtifactError(f"promotion evaluation failed: {error}") from error
    marker_hash = content_hash(batch.marker.document())
    signing_inputs: dict[str, PromotionSigningInput] = {}
    for candidate_id, result in batch.candidates.items():
        gate = promotion_gate(result)
        provenance = result.determinism.provenance
        artifacts = candidate_artifacts[candidate_id]
        signing_inputs[candidate_id] = PromotionSigningInput(
            candidate_id=candidate_id,
            evaluator_id=batch.marker.evaluator_id,
            evaluation_manifest_hash=batch.manifest_hash,
            evaluation_run_id=batch.evaluation_run_id,
            stage2_config_hash=provenance.stage2_config_hash,
            model_lock_hashes={path.value: artifact.model_lock_hash for path, artifact in artifacts.calibrators.items()},
            calibrator_hashes={path.value: artifact.hash() for path, artifact in artifacts.calibrators.items()},
            threshold_hashes={path.value: artifact.hash() for path, artifact in artifacts.thresholds.items()},
            hidden_pair_universe_hashes={split.value: hash_value for split, hash_value in result.hidden_pair_universe_hashes.items()},
            hidden_split_pair_counts={split.value: evaluation.size for split, evaluation in result.evaluations.items()},
            prediction_submission_hash=content_hash(promotion_submission_document(submissions[candidate_id])),
            promotion_marker_hash=marker_hash,
            consumed_hidden_splits=(GoldSplit.HIDDEN_HARD.value, GoldSplit.HIDDEN_REAL.value),
            gate_policy_hash=HIDDEN_PROMOTION_GATE_POLICY_HASH,
            passed=gate.passed,
            failures=gate.failures,
        )
    return PromotionEvaluationInputs(
        batch.manifest_hash, batch.evaluation_run_id, batch.incumbent_candidate_id, marker_hash, signing_inputs
    )


def _prediction_from_document(document: Mapping[str, Any]) -> Prediction:
    try:
        return Prediction(
            pair_id=document["pair_id"],
            candidate_id=document["candidate_id"],
            decision=Stage2Decision(document["decision"]),
            raw_score=document["raw_score"],
            probability=document["probability"],
            path=CalibrationPath(document["path"]),
            calibrator_hash=document["calibrator_hash"],
            threshold_hash=document["threshold_hash"],
            model_lock_hash=document["model_lock_hash"],
            manifest_hash=document["manifest_hash"],
            stage2_config_hash=document["stage2_config_hash"],
            inference_artifact_hash=document["inference_artifact_hash"],
            review_reason=ReviewReason(document["review_reason"]) if document["review_reason"] is not None else None,
        )
    except (TypeError, ValueError) as error:
        raise PrivatePromotionArtifactError(f"promotion prediction is invalid: {error}") from error


def _read_object(path: Path, label: str) -> Mapping[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, PrivatePromotionArtifactError) as error:
        raise PrivatePromotionArtifactError(f"cannot read {label}: {path}") from error
    if not isinstance(document, dict):
        raise PrivatePromotionArtifactError(f"{label} must be a JSON object")
    return document


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    document: dict[str, Any] = {}
    for key, value in pairs:
        if key in document:
            raise PrivatePromotionArtifactError(f"duplicate JSON object key: {key}")
        document[key] = value
    return document


def _validate(document: Mapping[str, Any], schema_name: str) -> None:
    try:
        validate(document, schema_name)
    except SchemaValidationError as error:
        raise PrivatePromotionArtifactError(str(error)) from error
