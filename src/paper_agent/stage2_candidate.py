"""Build a calibrated schema-v2 Stage 2 benchmark candidate from DEV only."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from tempfile import mkstemp
from typing import Any, Mapping

from .stage2_backends import ModelLock
from .stage2_calibration import (
    Stage2CalibrationBundle,
    build_stage2_calibration_bundle,
    freeze_dev_scores,
)
from .stage2_dev_calibration import FrozenDevRawScoreArtifact
from .stage2_evaluation import CalibrationPath, GoldLabelStore, GoldManifest, GoldSplit
from .stage2_pipeline import Stage2Profile
from .stage2_search import ReleasedStage2, load_stage2_benchmark_candidate, stage2_base_profile


_CANDIDATE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_CANDIDATE_NAME = "stage2-candidate-v2.json"


@dataclass(frozen=True, slots=True)
class Stage2CandidateBuild:
    """A published candidate and its calibrated DEV provenance."""

    candidate_path: Path
    release: ReleasedStage2
    dev_label_hash: str
    raw_score_hash: str
    selections: Mapping[CalibrationPath, Mapping[str, float]]


def build_stage2_candidate_bundle(
    *,
    manifest: GoldManifest,
    private_labels: GoldLabelStore,
    raw_scores: FrozenDevRawScoreArtifact,
    runtime: Mapping[str, Any],
    reranker_lock_path: Path,
    adjudicator_lock_path: Path,
    candidate_id: str,
    output_dir: Path,
) -> Stage2CandidateBuild:
    """Calibrate on the authoritative DEV split and atomically publish one bundle."""

    if _CANDIDATE_ID.fullmatch(candidate_id) is None:
        raise ValueError("Stage 2 candidate_id is invalid")
    if os.path.lexists(output_dir):
        raise FileExistsError(
            f"Stage 2 candidate output already exists: {output_dir}"
        )
    manifest.validate(private_labels)
    reranker_bytes = reranker_lock_path.read_bytes()
    adjudicator_bytes = adjudicator_lock_path.read_bytes()
    reranker_hash = sha256(reranker_bytes).hexdigest()
    adjudicator_hash = sha256(adjudicator_bytes).hexdigest()
    reranker_lock = ModelLock(**_json_object(reranker_bytes, "reranker model lock"))
    adjudicator_lock = ModelLock(
        **_json_object(adjudicator_bytes, "adjudicator model lock")
    )
    profile = stage2_base_profile(
        runtime,
        reranker_lock,
        adjudicator_lock,
        reranker_lock_hash=reranker_hash,
        adjudicator_lock_hash=adjudicator_hash,
    )
    _validate_raw_scores(
        raw_scores,
        manifest,
        profile,
        reranker_hash,
        adjudicator_hash,
    )
    dev_ids = {
        pair.pair_id for pair in manifest.pairs if pair.split is GoldSplit.DEV
    }
    dev_labels = GoldLabelStore(
        {pair_id: private_labels.labels[pair_id] for pair_id in dev_ids},
        private_labels.annotation_artifact_hash,
    )
    frozen = freeze_dev_scores(
        manifest,
        dev_labels,
        raw_scores.scores,
        raw_scores.model_lock_hashes,
        profile.base_runtime_config_hash,
    )
    calibration = build_stage2_calibration_bundle(frozen, manifest, dev_labels)
    selections = {
        path: {
            "positive_retention": selection.positive_retention,
            "relevant_recall": selection.relevant_recall,
            "needs_review_rate": selection.needs_review_rate,
        }
        for path, selection in calibration.selections.items()
    }
    dev_label_hash = dev_labels.hash()
    raw_score_hash = raw_scores.hash()

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_dir.mkdir()
    except FileExistsError:
        raise FileExistsError(
            f"Stage 2 candidate output already exists: {output_dir}"
        ) from None
    candidate_path, loaded = _write_candidate_files(
        output_dir,
        runtime,
        candidate_id,
        reranker_bytes,
        adjudicator_bytes,
        calibration,
    )
    return Stage2CandidateBuild(
        candidate_path,
        loaded,
        dev_label_hash,
        raw_score_hash,
        selections,
    )


def _validate_raw_scores(
    artifact: FrozenDevRawScoreArtifact,
    manifest: GoldManifest,
    profile: Stage2Profile,
    reranker_lock_hash: str,
    adjudicator_lock_hash: str,
) -> None:
    expected_ids = {
        pair.pair_id for pair in manifest.pairs if pair.split is GoldSplit.DEV
    }
    if (
        artifact.gold_manifest_hash != manifest.hash()
        or artifact.dev_manifest_hash != manifest.dev_hash()
        or artifact.private_snapshot_corpus_hash != manifest.corpus_hash
        or artifact.stage2_config_hash != profile.base_runtime_config_hash
    ):
        raise ValueError("DEV raw scores do not match the manifest or Stage 2 runtime")
    expected_locks = {
        CalibrationPath.RERANKER: reranker_lock_hash,
        CalibrationPath.QWEN: adjudicator_lock_hash,
    }
    if dict(artifact.model_lock_hashes) != expected_locks:
        raise ValueError("DEV raw scores do not match the supplied model locks")
    if any(set(artifact.scores[path]) != expected_ids for path in CalibrationPath):
        raise ValueError("DEV raw scores do not exactly cover the manifest DEV split")
    expected_query_keys = {
        (pair.topic, pair.language)
        for pair in manifest.pairs
        if pair.split is GoldSplit.DEV
    }
    if set(artifact.topic_queries) != expected_query_keys:
        raise ValueError("DEV topic queries do not exactly cover the manifest DEV split")
    if dict(artifact.topic_queries) != profile.evaluation_topic_query_map:
        raise ValueError("DEV topic queries do not match the frozen Stage 2 runtime")


def _write_candidate_files(
    root: Path,
    runtime: Mapping[str, Any],
    candidate_id: str,
    reranker_bytes: bytes,
    adjudicator_bytes: bytes,
    calibration: Stage2CalibrationBundle,
) -> tuple[Path, ReleasedStage2]:
    reranker_lock_path = root / "reranker.lock.json"
    adjudicator_lock_path = root / "adjudicator.lock.json"
    _write_bytes_new(reranker_lock_path, reranker_bytes)
    _write_bytes_new(adjudicator_lock_path, adjudicator_bytes)

    calibrations: dict[str, dict[str, dict[str, str]]] = {}
    for path in CalibrationPath:
        binding = calibration.calibrations[path]
        calibrator_path = root / f"{path.value}-calibrator.json"
        threshold_path = root / f"{path.value}-threshold.json"
        _write_json(calibrator_path, binding.calibrator.document())
        _write_json(threshold_path, binding.threshold.document())
        calibrations[path.value] = {
            "calibrator": _ref(calibrator_path),
            "threshold": _ref(threshold_path),
        }

    candidate = {
        "schema_version": "2",
        "profile": candidate_id,
        "reranker_lock": _ref(reranker_lock_path),
        "adjudicator_lock": _ref(adjudicator_lock_path),
        "calibration": calibrations,
        "runtime": dict(runtime),
    }
    candidate_path = root / _CANDIDATE_NAME
    payload = _json_bytes(candidate)
    descriptor, temporary_name = mkstemp(
        dir=root, prefix=f".{_CANDIDATE_NAME}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        loaded = load_stage2_benchmark_candidate(temporary_path)
        os.link(temporary_path, candidate_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return candidate_path, loaded


def _ref(path: Path) -> dict[str, str]:
    return {"path": path.name, "sha256": sha256(path.read_bytes()).hexdigest()}


def _write_json(path: Path, document: Mapping[str, Any]) -> None:
    _write_bytes_new(path, _json_bytes(document))


def _json_bytes(document: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def _write_bytes_new(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _json_object(payload: bytes, label: str) -> dict[str, Any]:
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError(f"Stage 2 {label} must be a JSON object")
    return value
