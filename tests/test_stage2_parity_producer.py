from __future__ import annotations

from dataclasses import dataclass, field, replace
from hashlib import sha256
import json
from pathlib import Path
from typing import Callable, Sequence

import pytest

from paper_agent.canonical import content_hash
from paper_agent.stage2_backends import ModelLock, RerankBatchError, RerankInput, RerankScore
from paper_agent.stage2_benchmark_inputs import benchmark_corpus_hash
from paper_agent.stage2_evaluation import CalibrationPath, PathCalibrator, ThresholdArtifact
from paper_agent.stage2_parity import (
    ParityEvidenceError,
    build_parity_evidence,
    freeze_parity_workload,
    make_parity_pair_id,
    parity_workload_from_document,
    write_parity_evidence_artifacts,
)
from paper_agent.stage2_pipeline import PathCalibration, Stage2Paper, Stage2Profile
from paper_agent.stage2_search import ReleasedStage2


TOKENIZER_HASH = "a" * 64
DEV_MANIFEST_HASH = "b" * 64
GOLD_MANIFEST_HASH = "c" * 64
DEV_LABEL_HASH = "d" * 64


def _hash(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _lock(name: str) -> tuple[ModelLock, str]:
    oracle = name == "oracle"
    lock = ModelLock(
        lock_version=1,
        backend="omlx_rerank",
        model_id=f"{name}-bge-reranker",
        source_repo="BAAI/bge-reranker-v2-m3",
        source_revision="953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
        conversion_repo=None if oracle else "soichisumi/bge-reranker-v2-m3-mlx",
        conversion_revision=None if oracle else "b4577f49e18adb53ed9e557192094f69f3dc2c1c",
        format="safetensors-fp32" if oracle else "safetensors-bf16",
        quantization="none",
        license="apache-2.0",
        parameter_count=567_755_777,
        omlx_version="0.5.7",
        mlx_version="0.32.0",
        file_hashes={"model.safetensors": ("1" if oracle else "2") * 64, "tokenizer.json": TOKENIZER_HASH},
    )
    return lock, _hash(lock.document())


def _binding(
    model_lock_hash: str,
    stage2_config_hash: str,
    slope: float,
    intercept: float,
    path: CalibrationPath = CalibrationPath.RERANKER,
) -> PathCalibration:
    pair_ids = ("dev-negative", "dev-positive")
    calibrator = PathCalibrator(
        1, path, slope, intercept, DEV_MANIFEST_HASH,
        GOLD_MANIFEST_HASH, model_lock_hash, DEV_LABEL_HASH,
        _hash(sorted(pair_ids)), 2, pair_ids,
    )
    threshold = ThresholdArtifact(
        1, path, 0.01, 0.99, calibrator.hash(), model_lock_hash,
        DEV_MANIFEST_HASH, DEV_LABEL_HASH, stage2_config_hash,
    )
    return PathCalibration(calibrator, threshold)


def _release(name: str, lock_hash: str, slope: float, intercept: float) -> ReleasedStage2:
    adjudicator_hash = ("3" if name == "oracle" else "4") * 64
    base = Stage2Profile(
        query="unused aggregate query",
        query_version="parity-v2",
        thresholds=None,
        reranker_model_id=f"{name}-bge-reranker",
        reranker_revision="revision",
        adjudicator_model_id=f"{name}-qwen",
        adjudicator_revision="revision",
        screening_scope_hash="5" * 64,
        reranker_lock_hash=lock_hash,
        adjudicator_lock_hash=adjudicator_hash,
    )
    reranker = _binding(lock_hash, base.base_runtime_config_hash, slope, intercept)
    qwen = _binding(
        adjudicator_hash,
        base.base_runtime_config_hash,
        1,
        0,
        CalibrationPath.QWEN,
    )
    return ReleasedStage2(name, replace(base, reranker_calibration=reranker, adjudicator_calibration=qwen), "6" * 64, "http://127.0.0.1:8000")


def _papers(count: int = 10_000) -> tuple[Stage2Paper, ...]:
    return tuple(Stage2Paper(f"paper-{index:05d}", f"Paper {index}", None if index == 0 else f"Abstract {index}", ("molecule",)) for index in range(count))


def _workload(count: int = 10_000):
    return freeze_parity_workload(
        _papers(count), topic="molecular_generation", language="en",
        query_version="molecular-generation-v1", query="molecular generation",
    )


def _receipt(workload) -> dict:
    return {
        "schema_version": 1,
        "parity": {
            "paper_count": 10_000,
            "paper_ids": sorted(pair.paper_id for pair in workload.pairs),
            "papers_corpus_hash": workload.corpus_hash(),
        },
    }


def _score(pair_id: str) -> float:
    paper_id = pair_id.rsplit("-", 1)[-1]
    return (int(paper_id[:8], 16) % 10_000 - 5_000) / 1_000


@dataclass
class _FakeReranker:
    model: str
    transform: Callable[[float], float]
    mode: str = "ok"
    backend_name: str = field(default="omlx_rerank", init=False)
    calls: int = field(default=0, init=False)
    inputs: list[tuple[RerankInput, ...]] = field(default_factory=list, init=False)

    def rerank(self, query: str, documents: Sequence[RerankInput]) -> tuple[RerankScore, ...]:
        self.calls += 1
        frozen = tuple(documents)
        self.inputs.append(frozen)
        rows = tuple(RerankScore(item.paper_id, self.transform(_score(item.paper_id))) for item in frozen)
        if self.mode == "missing":
            return rows[:-1]
        if self.mode == "duplicate":
            return (*rows[:-1], rows[0])
        if self.mode == "nonfinite":
            return (RerankScore(rows[0].paper_id, float("nan")), *rows[1:])
        if self.mode == "batch_error":
            raise RerankBatchError(rows[:-1], (rows[-1].paper_id,))
        return rows


@pytest.fixture(scope="module")
def parity_run():
    workload = _workload()
    receipt = _receipt(workload)
    oracle_lock, oracle_hash = _lock("oracle")
    candidate_lock, candidate_hash = _lock("candidate")
    oracle = _release("oracle", oracle_hash, 1, 0)
    candidate = _release("candidate", candidate_hash, 0.5, -0.5)
    oracle_backend = _FakeReranker(oracle_lock.model_id, lambda score: score)
    candidate_backend = _FakeReranker(candidate_lock.model_id, lambda score: 2 * score + 1)
    evidence = build_parity_evidence(
        workload,
        selection_receipt=receipt,
        selection_receipt_hash=content_hash(receipt),
        oracle=oracle,
        candidate=candidate,
        oracle_lock=oracle_lock,
        candidate_lock=candidate_lock,
        oracle_lock_hash=oracle_hash,
        candidate_lock_hash=candidate_hash,
        oracle_reranker=oracle_backend,
        candidate_reranker=candidate_backend,
    )
    return evidence, workload, oracle_backend, candidate_backend


def test_workload_binds_each_query_identity_and_exact_10k() -> None:
    workload = _workload()
    assert len(workload.pairs) == 10_000
    first = workload.pairs[0]
    assert first.pair_id == make_parity_pair_id(first.paper_id, first.topic, first.language, first.query_version, first.query)
    assert parity_workload_from_document(workload.document()).hash() == workload.hash()
    changed = workload.document()
    changed["pairs"][0]["query"] = "changed"
    with pytest.raises(ValueError, match="query identity"):
        parity_workload_from_document(changed)
    with pytest.raises(ValueError, match="10,000"):
        _workload(9_999)


def test_workload_preserves_the_complete_selected_corpus_identity() -> None:
    papers = list(_papers())
    papers[0] = Stage2Paper(
        papers[0].paper_id,
        papers[0].title,
        papers[0].abstract,
        papers[0].keywords,
        document_type="proceedings-article",
        possibly_truncated=True,
        multi_condition_conflict=True,
        language_anomaly=True,
    )
    workload = freeze_parity_workload(
        papers,
        topic="molecular_generation",
        language="en",
        query_version="molecular-generation-v1",
        query="molecular generation",
    )

    assert workload.corpus_hash() == benchmark_corpus_hash(papers)
    assert parity_workload_from_document(workload.document()).corpus_hash() == workload.corpus_hash()


def test_producer_runs_identical_query_documents_and_calibrated_windows(parity_run) -> None:
    evidence, workload, oracle_backend, candidate_backend = parity_run
    assert oracle_backend.calls == candidate_backend.calls == 1
    assert oracle_backend.inputs == candidate_backend.inputs
    assert oracle_backend.inputs[0][0].document == "Title: Paper 0\nAbstract: \nKeywords: molecule"
    assert evidence.manifest.workload_hash == workload.hash()
    assert evidence.manifest.query_assignment_hash == workload.query_assignment_hash()
    assert len(evidence.manifest.low_window_pair_ids) == len(evidence.manifest.high_window_pair_ids) == 200
    assert evidence.result.gate.passed
    assert evidence.result.kendall_tau_b == pytest.approx(1)


@pytest.mark.parametrize(("mode", "message"), [
    ("missing", "exactly cover"),
    ("duplicate", "invalid or duplicate"),
    ("nonfinite", "invalid or duplicate"),
    ("batch_error", "failed 1"),
])
def test_producer_rejects_incomplete_scores(mode: str, message: str) -> None:
    workload = _workload()
    receipt = _receipt(workload)
    oracle_lock, oracle_hash = _lock("oracle")
    candidate_lock, candidate_hash = _lock("candidate")
    oracle = _release("oracle", oracle_hash, 1, 0)
    candidate = _release("candidate", candidate_hash, 1, 0)
    with pytest.raises(ParityEvidenceError, match=message):
        build_parity_evidence(
            workload,
            selection_receipt=receipt,
            selection_receipt_hash=content_hash(receipt),
            oracle=oracle, candidate=candidate,
            oracle_lock=oracle_lock, candidate_lock=candidate_lock,
            oracle_lock_hash=oracle_hash, candidate_lock_hash=candidate_hash,
            oracle_reranker=_FakeReranker(oracle_lock.model_id, lambda score: score),
            candidate_reranker=_FakeReranker(candidate_lock.model_id, lambda score: score, mode),
        )


def test_producer_rejects_self_comparison_and_tokenizer_drift_before_calls() -> None:
    workload = _workload()
    receipt = _receipt(workload)
    oracle_lock, oracle_hash = _lock("oracle")
    candidate_lock, candidate_hash = _lock("candidate")
    oracle = _release("oracle", oracle_hash, 1, 0)
    candidate = _release("candidate", candidate_hash, 1, 0)
    oracle_backend = _FakeReranker(oracle_lock.model_id, lambda score: score)
    candidate_backend = _FakeReranker(candidate_lock.model_id, lambda score: score)
    with pytest.raises(ParityEvidenceError, match="self-comparison"):
        build_parity_evidence(
            workload, selection_receipt=receipt, selection_receipt_hash=content_hash(receipt),
            oracle=oracle, candidate=oracle, oracle_lock=oracle_lock, candidate_lock=oracle_lock,
            oracle_lock_hash=oracle_hash, candidate_lock_hash=oracle_hash,
            oracle_reranker=oracle_backend, candidate_reranker=oracle_backend,
        )
    drifted_lock = replace(candidate_lock, file_hashes={**candidate_lock.file_hashes, "tokenizer.json": "e" * 64})
    with pytest.raises(ParityEvidenceError, match="same tokenizer"):
        build_parity_evidence(
            workload, selection_receipt=receipt, selection_receipt_hash=content_hash(receipt),
            oracle=oracle, candidate=candidate, oracle_lock=oracle_lock, candidate_lock=drifted_lock,
            oracle_lock_hash=oracle_hash, candidate_lock_hash=candidate_hash,
            oracle_reranker=oracle_backend, candidate_reranker=candidate_backend,
        )
    assert oracle_backend.calls == candidate_backend.calls == 0


def test_writer_validates_and_never_replaces(tmp_path: Path, parity_run) -> None:
    evidence, _, _, _ = parity_run
    manifest_path = tmp_path / "parity-manifest.json"
    scores_path = tmp_path / "parity-scores.json"
    write_parity_evidence_artifacts(evidence, manifest_path=manifest_path, scores_path=scores_path)
    assert json.loads(manifest_path.read_text()) == evidence.manifest.document()
    assert json.loads(scores_path.read_text()) == evidence.scores_document()
    with pytest.raises(FileExistsError, match="already exists"):
        write_parity_evidence_artifacts(evidence, manifest_path=manifest_path, scores_path=scores_path)
