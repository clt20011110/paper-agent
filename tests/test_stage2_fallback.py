from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import pytest

from paper_agent.stage2_backends import (
    AdjudicationDecision,
    RerankInput,
    RerankScore,
    Stage2BackendError,
)
from paper_agent.stage2_evaluation import CalibrationPath, PathCalibrator, ThresholdArtifact, _hash
from paper_agent.stage2_fallback import (
    FallbackReleaseBinding,
    LocalCalibratedRerankerFallback,
)
from paper_agent.stage2_pipeline import PathCalibration, Stage2Paper, Stage2Pipeline, Stage2Profile
from paper_agent.storage import Database


@dataclass
class _Reranker:
    score: float
    fail: bool = False
    is_local: bool = False
    backend_name: str = "fixture"
    calls: int = 0

    def rerank(self, query: str, documents: Sequence[RerankInput]) -> tuple[RerankScore, ...]:
        self.calls += 1
        if self.fail:
            raise Stage2BackendError("primary unavailable")
        return tuple(RerankScore(item.paper_id, self.score) for item in documents)


@dataclass
class _Adjudicator:
    backend_name: str = "fixture"

    def adjudicate(self, request):
        return AdjudicationDecision(request.paper_id, "needs_review", 0.5, (), "", ())


def _profile() -> Stage2Profile:
    kwargs = dict(
        query="topic",
        query_version="v1",
        thresholds=None,
        reranker_model_id="primary",
        reranker_revision="r1",
        adjudicator_model_id="qwen",
        adjudicator_revision="q1",
        screening_scope_hash="0" * 64,
        reranker_lock_hash="a" * 64,
        adjudicator_lock_hash="b" * 64,
        release_gate_hash="gate-hash",
    )
    base = Stage2Profile(**kwargs)
    pair_ids = ("pair-negative", "pair-positive")
    bindings = {}
    for path, lock in ((CalibrationPath.RERANKER, "a" * 64), (CalibrationPath.QWEN, "b" * 64)):
        calibrator = PathCalibrator(
            1, path, 1.0, 0.0, "dev", "gold", lock, "labels", _hash(sorted(pair_ids)), 2, pair_ids,
        )
        bindings[path] = PathCalibration(calibrator, ThresholdArtifact(
            1, path, 0.25, 0.75, calibrator.hash(), lock, "dev", "labels", base.base_runtime_config_hash,
        ))
    return Stage2Profile(
        **kwargs,
        reranker_calibration=bindings[CalibrationPath.RERANKER],
        adjudicator_calibration=bindings[CalibrationPath.QWEN],
    )


def _backup_calibration(profile: Stage2Profile) -> PathCalibration:
    pair_ids = ("pair-negative", "pair-positive")
    calibrator = PathCalibrator(
        1, CalibrationPath.RERANKER, 1.0, 0.0, "dev", "gold", "c" * 64,
        "labels", _hash(sorted(pair_ids)), 2, pair_ids,
    )
    return PathCalibration(calibrator, ThresholdArtifact(
        1, CalibrationPath.RERANKER, 0.25, 0.75, calibrator.hash(), "c" * 64,
        "dev", "labels", profile.base_runtime_config_hash,
    ))


def _binding(
    primary: str = "a" * 64,
    backup: str = "b" * 64,
) -> FallbackReleaseBinding:
    return FallbackReleaseBinding(
        primary,
        backup,
        "c" * 64,
        "d" * 64,
    )


def test_fallback_requires_local_backend_and_independent_evidence() -> None:
    profile = _profile()
    with pytest.raises(ValueError, match="explicitly local"):
        LocalCalibratedRerankerFallback(
            _Reranker(2.0), "c" * 64, _backup_calibration(profile),
            _binding(),
        )
    with pytest.raises(ValueError, match="independent release evidence"):
        _binding("a" * 64, "a" * 64)


def test_primary_failure_uses_only_a_qualified_local_backup(tmp_path) -> None:
    profile = _profile()
    primary = _Reranker(0.0, fail=True)
    backup = _Reranker(2.0, is_local=True)
    fallback = LocalCalibratedRerankerFallback(
        backup, "c" * 64, _backup_calibration(profile),
        _binding(),
    )
    database = Database(tmp_path / "papers.sqlite3")
    database.migrate()
    paper = Stage2Paper("paper-1", "A title", "An abstract")
    with database.transaction() as connection:
        connection.execute("INSERT INTO papers(paper_id, title, abstract) VALUES (?, ?, ?)", (paper.paper_id, paper.title, paper.abstract))

    try:
        summary = Stage2Pipeline(database, primary, _Adjudicator(), profile, reranker_fallback=fallback).run("run-1", (paper,))

        assert primary.calls == 1
        assert backup.calls == 1
        assert summary.decisions[0].status.value == "relevant"
    finally:
        database.close()
