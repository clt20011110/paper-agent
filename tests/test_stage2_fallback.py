from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Sequence

import pytest

from paper_agent.stage2_backends import (
    AdjudicationDecision,
    ModelLock,
    RerankBatchError,
    RerankInput,
    RerankScore,
    Stage2BackendError,
)
from paper_agent.stage2_evaluation import CalibrationPath, PathCalibrator, ThresholdArtifact, _hash
from paper_agent.stage2_fallback import (
    FallbackReleaseBinding,
    LocalCalibratedRerankerFallback,
    stage2_shared_runtime_hash,
)
from paper_agent.stage2_pipeline import PathCalibration, Stage2Paper, Stage2Pipeline, Stage2Profile
from paper_agent.stage2_search import ReleasedRerankerFallback, ReleasedStage2
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
class _PartialReranker:
    returned: tuple[str, ...]
    score: float
    batch_error: bool = False
    unknown_id: str | None = None
    is_local: bool = False
    backend_name: str = "fixture_partial"

    def rerank(self, query: str, documents: Sequence[RerankInput]) -> tuple[RerankScore, ...]:
        scores = tuple(
            RerankScore(item.paper_id, self.score)
            for item in documents
            if item.paper_id in self.returned
        )
        if self.unknown_id is not None:
            scores = (*scores, RerankScore(self.unknown_id, self.score))
        if self.batch_error:
            failed = tuple(
                item.paper_id for item in documents if item.paper_id not in self.returned
            )
            raise RerankBatchError(scores, failed)
        return scores


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


def _binding(profile: Stage2Profile) -> FallbackReleaseBinding:
    return FallbackReleaseBinding(
        "a" * 64,
        "b" * 64,
        "c" * 64,
        "d" * 64,
        stage2_shared_runtime_hash(profile),
    )


def _fallback(profile: Stage2Profile, backend) -> LocalCalibratedRerankerFallback:
    return LocalCalibratedRerankerFallback(
        backend=backend,
        model_id="backup",
        model_revision="backup-r1",
        model_lock_hash="c" * 64,
        calibration=_backup_calibration(profile),
        release_binding=_binding(profile),
        runtime_config_hash="e" * 64,
    )


def test_fallback_requires_local_backend_and_hash_bound_release_evidence() -> None:
    profile = _profile()
    with pytest.raises(ValueError, match="explicitly local"):
        _fallback(profile, _Reranker(2.0))
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        FallbackReleaseBinding(
            "not-a-hash", "b" * 64, "c" * 64, "d" * 64,
            stage2_shared_runtime_hash(profile),
        )


def test_primary_failure_uses_only_a_qualified_local_backup(tmp_path) -> None:
    profile = _profile()
    primary = _Reranker(0.0, fail=True)
    backup = _Reranker(2.0, is_local=True)
    fallback = _fallback(profile, backup)
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
        provenance = summary.decisions[0].reranker_provenance
        assert provenance is not None
        assert provenance.fallback_used is True
        assert provenance.model_id == "backup"
        persisted = database.connection.execute(
            "SELECT model_id, model_revision, reason FROM filter_decisions WHERE run_id = ?",
            ("run-1",),
        ).fetchone()
        assert tuple(persisted[:2]) == ("backup", "backup-r1")
        reason = json.loads(persisted["reason"])
        assert reason["reranker_lock_hash"] == "c" * 64
        assert reason["reranker_provenance"]["fallback_used"] is True

        resumed = Stage2Pipeline(
            database, primary, _Adjudicator(), profile, reranker_fallback=fallback
        ).run("run-1", (paper,))
        assert resumed.decisions[0].resumed is True
        assert resumed.decisions[0].reranker_provenance == provenance

        changed = LocalCalibratedRerankerFallback(
            backend=backup,
            model_id=fallback.model_id,
            model_revision=fallback.model_revision,
            model_lock_hash=fallback.model_lock_hash,
            calibration=fallback.calibration,
            release_binding=fallback.release_binding,
            runtime_config_hash="f" * 64,
        )
        with pytest.raises(ValueError, match="immutable"):
            Stage2Pipeline(
                database, primary, _Adjudicator(), profile,
                reranker_fallback=changed,
            ).run("run-1", (paper,))
    finally:
        database.close()


def test_same_endpoint_with_different_api_key_uses_fallback_authentication() -> None:
    profile = _profile()
    model_lock = ModelLock(
        1,
        "omlx_rerank",
        "backup",
        "example/backup",
        "backup-source-r1",
        None,
        None,
        "safetensors-fp32",
        "none",
        "apache-2.0",
        1,
        "1",
        "1",
        {"model.safetensors": "weights"},
    )
    released_fallback = ReleasedRerankerFallback(
        model_lock=model_lock,
        model_lock_hash="c" * 64,
        calibration=_backup_calibration(profile),
        omlx_base_url="http://127.0.0.1:8000",
        api_key_env="BACKUP_OMLX_KEY",
        release_binding=_binding(profile),
        runtime_config_hash="e" * 64,
    )
    released = ReleasedStage2(
        "primary",
        profile,
        "release-hash",
        "http://127.0.0.1:8000",
        "PRIMARY_OMLX_KEY",
        released_fallback,
    )
    primary_transport = object()

    runtime = released._reranker_fallback(
        primary_transport,  # type: ignore[arg-type]
        {"PRIMARY_OMLX_KEY": "primary-secret", "BACKUP_OMLX_KEY": "backup-secret"},
    )

    assert runtime is not None
    assert runtime.backend.transport is not primary_transport
    assert runtime.backend.transport.api_key == "backup-secret"


@pytest.mark.parametrize("batch_error", (False, True))
def test_partial_primary_response_uses_fallback_only_for_missing_papers(
    tmp_path, batch_error: bool,
) -> None:
    profile = _profile()
    primary = _PartialReranker(("paper-primary",), 2.0, batch_error=batch_error)
    backup = _PartialReranker(("paper-fallback",), 2.0, is_local=True)
    papers = (
        Stage2Paper("paper-primary", "Primary", "Abstract"),
        Stage2Paper("paper-fallback", "Fallback", "Abstract"),
    )
    with Database(tmp_path / "partial.sqlite3") as database:
        database.migrate()
        with database.transaction() as connection:
            connection.executemany(
                "INSERT INTO papers(paper_id, title, abstract) VALUES (?, ?, ?)",
                ((paper.paper_id, paper.title, paper.abstract) for paper in papers),
            )
        summary = Stage2Pipeline(
            database, primary, _Adjudicator(), profile,
            reranker_fallback=_fallback(profile, backup),
        ).run(f"partial-{batch_error}", papers)

    decisions = {item.paper_id: item for item in summary.decisions}
    assert decisions["paper-primary"].reranker_provenance.source == "primary"
    assert decisions["paper-fallback"].reranker_provenance.source == "fallback"


def test_invalid_primary_and_partial_fallback_fail_closed_per_paper(tmp_path) -> None:
    profile = _profile()
    primary = _PartialReranker((), 2.0, unknown_id="unknown")
    backup = _PartialReranker(("paper-good",), 2.0, is_local=True)
    papers = (
        Stage2Paper("paper-good", "Good", "Abstract"),
        Stage2Paper("paper-missing", "Missing", "Abstract"),
    )
    with Database(tmp_path / "invalid.sqlite3") as database:
        database.migrate()
        with database.transaction() as connection:
            connection.executemany(
                "INSERT INTO papers(paper_id, title, abstract) VALUES (?, ?, ?)",
                ((paper.paper_id, paper.title, paper.abstract) for paper in papers),
            )
        summary = Stage2Pipeline(
            database, primary, _Adjudicator(), profile,
            reranker_fallback=_fallback(profile, backup),
        ).run("invalid-partial", papers)

    decisions = {item.paper_id: item for item in summary.decisions}
    assert decisions["paper-good"].status.value == "relevant"
    assert decisions["paper-good"].reranker_provenance.source == "fallback"
    assert decisions["paper-missing"].status.value == "needs_review"
    assert decisions["paper-missing"].reason_code == "reranker_backend_failure"
    assert decisions["paper-missing"].reranker_provenance.source == "fallback"
