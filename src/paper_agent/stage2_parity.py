"""Freeze and execute trusted 10k FP32/BF16 reranker parity evidence."""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import isfinite
import os
from pathlib import Path
from tempfile import mkstemp
from typing import Any, Mapping, Sequence

from .canonical import content_hash
from .schema import validate
from .stage2_backends import (
    ModelLock,
    RerankBatchError,
    RerankInput,
    RerankScore,
    RerankerBackend,
    Stage2BackendError,
)
from .stage2_benchmark_inputs import benchmark_corpus_hash
from .stage2_evaluation import (
    ParityManifest,
    ParityResult,
    ParityScore,
    PathCalibrator,
    ThresholdArtifact,
    pair_universe_hash,
    parity_gate,
)
from .stage2_pipeline import PathCalibration, Stage2Paper
from .stage2_search import ReleasedStage2


PAIR_COUNT = 10_000
WINDOW_SIZE = 200
PREPROCESS_CONTRACT = {
    "kind": "stage2-parity-preprocess-v2",
    "document_template": "Title: {title}\nAbstract: {abstract_or_empty}\nKeywords: {keywords}",
    "missing_abstract": "empty_string",
    "keyword_separator": ", ",
    "endpoint": "/v1/rerank",
    "endpoint_owns_truncation": True,
    "client_max_length": None,
}
WINDOW_SELECTOR = {
    "kind": "oracle-calibrated-probability-nearest-v2",
    "window_size": WINDOW_SIZE,
    "distance": "abs(oracle_calibrator.predict(raw_score)-oracle_probability_threshold)",
    "tie_break": "pair_id_ascending",
}


class ParityEvidenceError(ValueError):
    """Parity inputs or model output cannot produce trustworthy evidence."""


@dataclass(frozen=True, slots=True)
class ParityPair:
    pair_id: str
    paper_id: str
    topic: str
    language: str
    query_version: str
    query: str
    title: str
    abstract: str | None
    keywords: tuple[str, ...]
    document_type: str | None = None
    possibly_truncated: bool = False
    multi_condition_conflict: bool = False
    language_anomaly: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "keywords", tuple(self.keywords))
        if not all((self.pair_id, self.paper_id, self.topic, self.language, self.query_version, self.query, self.title)):
            raise ValueError("parity workload rows require complete query and paper fields")
        if self.pair_id != make_parity_pair_id(
            self.paper_id, self.topic, self.language, self.query_version, self.query
        ):
            raise ValueError("parity pair_id does not match its query identity and paper_id")

    def document(self) -> dict[str, object]:
        return {
            "pair_id": self.pair_id,
            "paper_id": self.paper_id,
            "topic": self.topic,
            "language": self.language,
            "query_version": self.query_version,
            "query": self.query,
            "title": self.title,
            "abstract": self.abstract,
            "keywords": list(self.keywords),
            "document_type": self.document_type,
            "possibly_truncated": self.possibly_truncated,
            "multi_condition_conflict": self.multi_condition_conflict,
            "language_anomaly": self.language_anomaly,
        }

    @property
    def rendered_document(self) -> str:
        return (
            f"Title: {self.title}\n"
            f"Abstract: {self.abstract or ''}\n"
            f"Keywords: {', '.join(self.keywords)}"
        )


@dataclass(frozen=True, slots=True)
class ParityWorkload:
    pairs: tuple[ParityPair, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "pairs", tuple(self.pairs))
        ids = [pair.pair_id for pair in self.pairs]
        paper_ids = [pair.paper_id for pair in self.pairs]
        if len(ids) != PAIR_COUNT or len(set(ids)) != PAIR_COUNT:
            raise ValueError("parity workload requires exactly 10,000 unique pairs")
        if len(set(paper_ids)) != PAIR_COUNT:
            raise ValueError("parity workload requires exactly 10,000 unique papers")

    def document(self) -> dict[str, object]:
        return {
            "schema_version": "2",
            "kind": "stage2_parity_workload",
            "pair_count": PAIR_COUNT,
            "pairs": [pair.document() for pair in self.pairs],
        }

    def hash(self) -> str:
        return content_hash(self.document())

    def corpus_hash(self) -> str:
        return benchmark_corpus_hash(tuple(
            Stage2Paper(
                pair.paper_id,
                pair.title,
                pair.abstract,
                pair.keywords,
                pair.document_type,
                pair.possibly_truncated,
                pair.multi_condition_conflict,
                pair.language_anomaly,
            )
            for pair in self.pairs
        ))

    def query_assignment_hash(self) -> str:
        return content_hash([
            {
                "pair_id": pair.pair_id,
                "topic": pair.topic,
                "language": pair.language,
                "query_version": pair.query_version,
                "query": pair.query,
            }
            for pair in self.pairs
        ])


@dataclass(frozen=True, slots=True)
class ParityEvidence:
    manifest: ParityManifest
    scores: tuple[ParityScore, ...]
    oracle_calibrator: PathCalibrator
    oracle_threshold: ThresholdArtifact
    candidate_calibrator: PathCalibrator
    candidate_threshold: ThresholdArtifact

    @property
    def result(self) -> ParityResult:
        return parity_gate(
            self.manifest,
            self.scores,
            self.oracle_calibrator,
            self.oracle_threshold,
            self.candidate_calibrator,
            self.candidate_threshold,
        )

    def scores_document(self) -> dict[str, object]:
        return {
            "schema_version": "2",
            "kind": "stage2_parity_scores",
            "manifest_hash": self.manifest.hash(),
            "workload_hash": self.manifest.workload_hash,
            "oracle_model_lock_hash": self.manifest.oracle_model_lock_hash,
            "candidate_model_lock_hash": self.manifest.candidate_model_lock_hash,
            "score_count": PAIR_COUNT,
            "failure_count": 0,
            "scores": [score.document() for score in self.scores],
        }


def make_parity_pair_id(
    paper_id: str,
    topic: str,
    language: str,
    query_version: str,
    query: str,
) -> str:
    return f"parity-{content_hash([paper_id, topic, language, query_version, query])[:32]}"


def parity_workload_from_document(value: Mapping[str, Any]) -> ParityWorkload:
    validate(value, "stage2-parity-workload.schema.json")
    return ParityWorkload(tuple(ParityPair(
        pair_id=row["pair_id"],
        paper_id=row["paper_id"],
        topic=row["topic"],
        language=row["language"],
        query_version=row["query_version"],
        query=row["query"],
        title=row["title"],
        abstract=row["abstract"],
        keywords=tuple(row["keywords"]),
        document_type=row["document_type"],
        possibly_truncated=row["possibly_truncated"],
        multi_condition_conflict=row["multi_condition_conflict"],
        language_anomaly=row["language_anomaly"],
    ) for row in value["pairs"]))


def freeze_parity_workload(
    papers: Sequence[Stage2Paper],
    *,
    topic: str,
    language: str,
    query_version: str,
    query: str,
) -> ParityWorkload:
    ordered = tuple(sorted(papers, key=lambda paper: paper.paper_id))
    return ParityWorkload(tuple(ParityPair(
        pair_id=make_parity_pair_id(paper.paper_id, topic, language, query_version, query),
        paper_id=paper.paper_id,
        topic=topic,
        language=language,
        query_version=query_version,
        query=query,
        title=paper.title,
        abstract=paper.abstract,
        keywords=paper.keywords,
        document_type=paper.document_type,
        possibly_truncated=paper.possibly_truncated,
        multi_condition_conflict=paper.multi_condition_conflict,
        language_anomaly=paper.language_anomaly,
    ) for paper in ordered))


def build_parity_evidence(
    workload: ParityWorkload,
    *,
    selection_receipt: Mapping[str, Any],
    selection_receipt_hash: str,
    oracle: ReleasedStage2,
    candidate: ReleasedStage2,
    oracle_lock: ModelLock,
    candidate_lock: ModelLock,
    oracle_lock_hash: str,
    candidate_lock_hash: str,
    oracle_reranker: RerankerBackend,
    candidate_reranker: RerankerBackend,
) -> ParityEvidence:
    preflight_parity_evidence(
        workload,
        selection_receipt=selection_receipt,
        selection_receipt_hash=selection_receipt_hash,
        oracle=oracle,
        candidate=candidate,
        oracle_lock=oracle_lock,
        candidate_lock=candidate_lock,
        oracle_lock_hash=oracle_lock_hash,
        candidate_lock_hash=candidate_lock_hash,
    )
    oracle_binding = _binding(oracle, oracle_lock, oracle_lock_hash, "oracle")
    candidate_binding = _binding(candidate, candidate_lock, candidate_lock_hash, "candidate")
    _validate_backends(oracle_reranker, candidate_reranker, oracle, candidate)

    oracle_scores = _score(workload, oracle_reranker, "oracle")
    candidate_scores = _score(workload, candidate_reranker, "candidate")
    low_window = _window(oracle_scores, oracle_binding.calibrator, oracle_binding.threshold.low)
    high_window = _window(oracle_scores, oracle_binding.calibrator, oracle_binding.threshold.high)
    pair_ids = tuple(pair.pair_id for pair in workload.pairs)
    manifest = ParityManifest(
        version=2,
        pair_ids=pair_ids,
        workload_hash=workload.hash(),
        selection_receipt_hash=selection_receipt_hash,
        pair_universe_hash=pair_universe_hash(pair_ids),
        query_assignment_hash=workload.query_assignment_hash(),
        corpus_hash=workload.corpus_hash(),
        tokenizer_hash=oracle_lock.file_hashes["tokenizer.json"],
        preprocess_hash=content_hash(PREPROCESS_CONTRACT),
        oracle_model_lock_hash=oracle_lock_hash,
        candidate_model_lock_hash=candidate_lock_hash,
        oracle_calibrator_hash=oracle_binding.calibrator.hash(),
        candidate_calibrator_hash=candidate_binding.calibrator.hash(),
        oracle_threshold_artifact_hash=oracle_binding.threshold.hash(),
        candidate_threshold_artifact_hash=candidate_binding.threshold.hash(),
        gold_manifest_hash=oracle_binding.calibrator.gold_manifest_hash,
        dev_manifest_hash=oracle_binding.calibrator.dev_manifest_hash,
        dev_label_hash=oracle_binding.calibrator.dev_label_hash,
        calibration_pair_ids_hash=oracle_binding.calibrator.calibration_pair_ids_hash,
        window_selector_hash=content_hash(WINDOW_SELECTOR),
        low_window_pair_ids=frozenset(low_window),
        high_window_pair_ids=frozenset(high_window),
    )
    evidence = ParityEvidence(
        manifest,
        tuple(ParityScore(pair_id, oracle_scores[pair_id], candidate_scores[pair_id]) for pair_id in pair_ids),
        oracle_binding.calibrator,
        oracle_binding.threshold,
        candidate_binding.calibrator,
        candidate_binding.threshold,
    )
    evidence.result
    return evidence


def preflight_parity_evidence(
    workload: ParityWorkload,
    *,
    selection_receipt: Mapping[str, Any],
    selection_receipt_hash: str,
    oracle: ReleasedStage2,
    candidate: ReleasedStage2,
    oracle_lock: ModelLock,
    candidate_lock: ModelLock,
    oracle_lock_hash: str,
    candidate_lock_hash: str,
) -> None:
    """Validate all offline parity bindings without invoking either model."""
    oracle_binding = _binding(oracle, oracle_lock, oracle_lock_hash, "oracle")
    candidate_binding = _binding(candidate, candidate_lock, candidate_lock_hash, "candidate")
    _validate_model_pair(oracle_lock, candidate_lock, oracle_lock_hash, candidate_lock_hash)
    _validate_calibration_pair(oracle_binding, candidate_binding)
    _validate_receipt(workload, selection_receipt, selection_receipt_hash)


def validate_parity_workload_receipt(
    workload: ParityWorkload,
    selection_receipt: Mapping[str, Any],
) -> str:
    """Validate a workload's receipt and return its canonical content hash."""
    receipt_hash = content_hash(selection_receipt)
    _validate_receipt(workload, selection_receipt, receipt_hash)
    return receipt_hash


def write_parity_workload(workload: ParityWorkload, *, output_path: Path) -> None:
    """Write one immutable, schema-v2 parity workload."""
    document = workload.document()
    validate(document, "stage2-parity-workload.schema.json")
    if os.path.lexists(output_path):
        raise FileExistsError(f"parity workload output already exists: {output_path}")
    _write_json_no_replace(output_path, document)


def write_parity_evidence_artifacts(
    evidence: ParityEvidence,
    *,
    manifest_path: Path,
    scores_path: Path,
) -> None:
    if manifest_path.absolute() == scores_path.absolute():
        raise ValueError("parity manifest and score paths must differ")
    documents = (
        (manifest_path, evidence.manifest.document(), "stage2-parity-manifest.schema.json"),
        (scores_path, evidence.scores_document(), "stage2-parity-scores.schema.json"),
    )
    for path, document, schema in documents:
        validate(document, schema)
        if os.path.lexists(path):
            raise FileExistsError(f"parity output already exists: {path}")
    for path, document, _ in reversed(documents):
        _write_json_no_replace(path, document)


def _binding(
    release: ReleasedStage2,
    lock: ModelLock,
    lock_hash: str,
    role: str,
) -> PathCalibration:
    binding = release.profile.reranker_calibration
    if binding is None or not release.profile.production_calibrated:
        raise ParityEvidenceError(f"{role} requires a calibrated Stage 2 candidate")
    if release.profile.reranker_lock_hash != lock_hash or release.profile.reranker_model_id != lock.model_id:
        raise ParityEvidenceError(f"{role} release does not match its model lock")
    return binding


def _validate_model_pair(
    oracle: ModelLock,
    candidate: ModelLock,
    oracle_hash: str,
    candidate_hash: str,
) -> None:
    if oracle_hash == candidate_hash:
        raise ParityEvidenceError("parity rejects oracle self-comparison")
    if oracle.backend != candidate.backend or oracle.backend != "omlx_rerank":
        raise ParityEvidenceError("parity requires two omlx_rerank model locks")
    if (oracle.source_repo, oracle.source_revision) != (candidate.source_repo, candidate.source_revision):
        raise ParityEvidenceError("parity models must share the exact upstream source revision")
    if oracle.conversion_repo is not None or candidate.conversion_repo is None:
        raise ParityEvidenceError("parity requires an official oracle and a conversion candidate")
    if oracle.format != "safetensors-fp32" or oracle.quantization != "none":
        raise ParityEvidenceError("parity oracle must use official FP32 weights")
    if candidate.format != "safetensors-bf16" or candidate.quantization != "none":
        raise ParityEvidenceError("parity candidate must use calibrated BF16 weights")
    if oracle.file_hashes.get("tokenizer.json") != candidate.file_hashes.get("tokenizer.json"):
        raise ParityEvidenceError("parity models must use the same tokenizer artifact")
    if oracle.file_hashes.get("model.safetensors") == candidate.file_hashes.get("model.safetensors"):
        raise ParityEvidenceError("parity rejects identical oracle and candidate weights")


def _validate_calibration_pair(oracle: PathCalibration, candidate: PathCalibration) -> None:
    left = oracle.calibrator
    right = candidate.calibrator
    if (
        left.gold_manifest_hash,
        left.dev_manifest_hash,
        left.dev_label_hash,
        left.calibration_pair_ids,
    ) != (
        right.gold_manifest_hash,
        right.dev_manifest_hash,
        right.dev_label_hash,
        right.calibration_pair_ids,
    ):
        raise ParityEvidenceError("parity calibrators must share frozen DEV provenance")


def _validate_receipt(
    workload: ParityWorkload,
    receipt: Mapping[str, Any],
    receipt_hash: str,
) -> None:
    if content_hash(receipt) != receipt_hash:
        raise ParityEvidenceError("parity selection receipt hash is invalid")
    parity = receipt.get("parity")
    if not isinstance(parity, Mapping):
        raise ParityEvidenceError("parity selection receipt has no parity workload")
    paper_ids = [pair.paper_id for pair in workload.pairs]
    if parity.get("paper_count") != PAIR_COUNT or parity.get("paper_ids") != sorted(paper_ids):
        raise ParityEvidenceError("parity workload does not match its selection receipt")
    if parity.get("papers_corpus_hash") != workload.corpus_hash():
        raise ParityEvidenceError("parity workload corpus does not match its selection receipt")


def _validate_backends(
    oracle_backend: RerankerBackend,
    candidate_backend: RerankerBackend,
    oracle: ReleasedStage2,
    candidate: ReleasedStage2,
) -> None:
    for role, backend, release in (
        ("oracle", oracle_backend, oracle),
        ("candidate", candidate_backend, candidate),
    ):
        if getattr(backend, "backend_name", None) != "omlx_rerank":
            raise ParityEvidenceError(f"{role} parity backend must be omlx_rerank")
        if getattr(backend, "model", None) != release.profile.reranker_model_id:
            raise ParityEvidenceError(f"{role} backend model does not match its frozen release")


def _score(
    workload: ParityWorkload,
    backend: RerankerBackend,
    role: str,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    by_query: dict[str, list[ParityPair]] = {}
    for pair in workload.pairs:
        by_query.setdefault(pair.query, []).append(pair)
    try:
        for query in sorted(by_query):
            pairs = by_query[query]
            returned = backend.rerank(query, tuple(
                RerankInput(pair.pair_id, pair.rendered_document) for pair in pairs
            ))
            for row in returned:
                if not isinstance(row, RerankScore) or row.paper_id in scores or not isfinite(row.raw_score):
                    raise ParityEvidenceError(f"{role} returned invalid or duplicate scores")
                scores[row.paper_id] = float(row.raw_score)
    except RerankBatchError as error:
        raise ParityEvidenceError(f"{role} failed {len(error.failed_paper_ids)} parity pairs") from error
    except Stage2BackendError as error:
        raise ParityEvidenceError(f"{role} reranker failed: {error}") from error
    expected = {pair.pair_id for pair in workload.pairs}
    if set(scores) != expected:
        raise ParityEvidenceError(f"{role} scores do not exactly cover all 10,000 pairs")
    return scores


def _window(
    scores: Mapping[str, float],
    calibrator: PathCalibrator,
    threshold: float,
) -> tuple[str, ...]:
    return tuple(pair_id for _, pair_id in sorted(
        (abs(calibrator.predict(score) - threshold), pair_id)
        for pair_id, score in scores.items()
    )[:WINDOW_SIZE])


def _write_json_no_replace(path: Path, document: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write((json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode())
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
