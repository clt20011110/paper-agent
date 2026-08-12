from __future__ import annotations

from paper_agent.stage2_evaluation import BenchmarkEnvironment, PerformanceRunRecord
from paper_agent.stage2_tuning import (
    PerformanceMeasurement,
    SoakSelectionInput,
    Stage2TuningError,
    TuningCandidate,
    TuningConfiguration,
    select_stage2_tuning_winner,
)
import pytest


HASH = "a" * 64
LOCKS = ("b" * 64, "c" * 64)


def _environment(batch: int, concurrency: int) -> BenchmarkEnvironment:
    return BenchmarkEnvironment(
        machine_model="Apple Silicon M4 Max",
        memory_gb=36,
        macos_version="15.6",
        omlx_version="0.5.7",
        mlx_version="0.27.0",
        power_mode="automatic",
        background_load="isolated",
        batch_config={"document_batch_size": batch, "adjudicator_concurrency": concurrency},
        resident_model_instances={LOCKS[0]: 1, LOCKS[1]: 1},
    )


def _record(
    scenario: str,
    run_id: str,
    environment: BenchmarkEnvironment,
    *,
    duration: float,
    p95: float = 1.0,
    peak_memory_gb: float = 20.0,
) -> PerformanceRunRecord:
    return PerformanceRunRecord(
        record_version=2, scenario=scenario, run_id=run_id,
        manifest_hash=HASH, stage2_config_hash=HASH, model_lock_hashes=LOCKS,
        duration_seconds=duration, p50_seconds=0.5, p95_seconds=p95,
        peak_memory_gb=peak_memory_gb, request_count=100, failed_request_count=0,
        service_request_count=100, service_failed_request_count=0,
        resume_verified=True, resume_model_call_count=0, resumed_pair_count=100,
        completed_pair_ids=tuple(f"p-{index}" for index in range(100)),
        needs_review_pair_ids=(), failed_request_pair_ids=(), qwen_pair_ids=(),
        environment=environment,
        executed_components=("rules", "reranker", "qwen", "schema_validation", "sqlite_commit"),
        sqlite_commit_count=100, warmed=True,
    )


def _candidate(
    batch: int,
    concurrency: int,
    *,
    duration: float = 10.0,
    p95: float = 1.0,
    peak_memory_gb: float = 20.0,
    quality: bool = True,
) -> TuningCandidate:
    environment = _environment(batch, concurrency)
    measurements = tuple(
        PerformanceMeasurement(
            f"{index + 1:064x}",
            _record(scenario, f"{batch}-{concurrency}-{scenario}-{run}", environment,
                    duration=duration, p95=p95, peak_memory_gb=peak_memory_gb),
            quality,
        )
        for index, (scenario, run) in enumerate(
            [("normal", 1), ("normal", 2), ("normal", 3), ("stress", 1), ("stress", 2), ("stress", 3)]
        )
    )
    return TuningCandidate(
        TuningConfiguration(batch, concurrency), measurements[:3], measurements[3:],
        selection_input=SoakSelectionInput(f"{batch * 100 + concurrency:064x}", "soak scheduled after selection"),
    )


def _grid(overrides: dict[tuple[int, int], dict[str, object]] | None = None) -> list[TuningCandidate]:
    overrides = overrides or {}
    return [
        _candidate(batch, concurrency, **overrides.get((batch, concurrency), {}))
        for batch in (16, 32, 64) for concurrency in (4, 8, 16)
    ]


def test_requires_the_exact_complete_grid() -> None:
    with pytest.raises(Stage2TuningError, match="every 16/32/64 by 4/8/16"):
        select_stage2_tuning_winner(_grid()[:-1])


def test_selects_valid_throughput_winner_and_freezes_all_bindings() -> None:
    candidates = _grid({
        (32, 8): {"duration": 2.0},
        (64, 8): {"duration": 1.0, "peak_memory_gb": 28.1},
        (64, 16): {"duration": 1.1, "quality": False},
    })

    winner = select_stage2_tuning_winner(candidates)

    assert (winner.document_batch_size, winner.adjudicator_concurrency) == (32, 8)
    assert winner.throughput == 50.0
    assert len(winner.input_record_hashes) == 63
    assert winner.document()["qwen_runtime_auto_increase"] is False
    assert winner.environment.batch_config == {"document_batch_size": 32, "adjudicator_concurrency": 8}
    assert len(winner.selection_hash) == 64


def test_rejects_p95_regression_and_falls_back_to_next_smaller_batch() -> None:
    candidates = _grid({
        (16, 4): {"duration": 10.0, "p95": 1.0},
        (32, 4): {"duration": 9.5, "p95": 1.3},  # <10% gain and >25% p95 growth
        (64, 4): {"duration": 2.0, "peak_memory_gb": 29.0},
    })

    winner = select_stage2_tuning_winner(candidates)

    assert (winner.document_batch_size, winner.adjudicator_concurrency) == (16, 4)


def test_candidate_requires_one_soak_or_explicit_selection_input() -> None:
    candidate = _candidate(16, 4)
    with pytest.raises(ValueError, match="exactly one soak or selection input"):
        TuningCandidate(candidate.configuration, candidate.normal, candidate.stress)
