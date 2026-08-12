from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from paper_agent.stage2_backends import ModelLock, OmlxResponse, ThresholdArtifact
from paper_agent.stage2_benchmark import (
    BenchmarkExecutionRecord,
    BenchmarkRunSpec,
    MacOSMemoryObserver,
    Stage2BenchmarkRunner,
    benchmark_alarm_codes,
)
from paper_agent.stage2_evaluation import (
    BenchmarkEnvironment,
    CalibrationPath,
    PathCalibrator,
    PerformanceCase,
    ThresholdArtifact as ProbabilityThresholdArtifact,
    _hash,
)
from paper_agent.stage2_fallback import FallbackReleaseBinding
from paper_agent.stage2_pipeline import (
    ADJUDICATOR_SHARE_ALARM,
    ERROR_RATE_ALARM,
    MEMORY_WATERMARK_ALARM,
    PathCalibration,
    Stage2Paper,
    Stage2Profile,
)
from paper_agent.stage2_prompt_contract import (
    adjudication_messages,
    estimate_omlx_chat_input_token_proxy,
)
from paper_agent.storage import Database


@dataclass
class StepClock:
    value: float = 0.0
    step: float = 0.01
    lock: Lock = field(default_factory=Lock)

    def __call__(self) -> float:
        with self.lock:
            self.value += self.step
            return self.value


@dataclass
class FakeOmlxTransport:
    malformed_chat_ids: frozenset[str] = frozenset()
    fail_first_multi_rerank: bool = False
    requests: list[tuple[str, Mapping[str, Any]]] = field(default_factory=list)
    lock: Lock = field(default_factory=Lock)
    _multi_rerank_failed: bool = False

    def request(self, path: str, payload: Mapping[str, Any]) -> OmlxResponse:
        with self.lock:
            self.requests.append((path, payload))
        if path == "/v1/rerank":
            if self.fail_first_multi_rerank and len(payload["documents"]) > 1 and not self._multi_rerank_failed:
                self._multi_rerank_failed = True
                return OmlxResponse(503, b'{"error":"MLX out of memory"}')
            scores = []
            for index, document in enumerate(payload["documents"]):
                title = document.splitlines()[0].removeprefix("Title: ")
                score = -2.0 if title == "p-low" else 0.5 if title == "p-gray" else 3.0
                scores.append({"index": index, "relevance_score": score})
            return OmlxResponse(200, json.dumps({"model": payload["model"], "results": scores}).encode())
        assert path == "/v1/chat/completions"
        prompt = payload["messages"][-1]["content"]
        paper_id = prompt.split("Paper ID: ", 1)[1].splitlines()[0]
        if paper_id in self.malformed_chat_ids:
            content = "not-json"
        else:
            content = json.dumps({
                "paper_id": paper_id,
                "decision": "relevant",
                "score": 0.9,
                "reason_codes": ["topic_match"],
                "rationale": "The paper matches the frozen query.",
                "evidence_fields": ["title"],
            })
        return OmlxResponse(200, json.dumps({
            "model": payload["model"],
            "choices": [{"message": {"content": content}}],
        }).encode())


class PrimaryUnavailableTransport(FakeOmlxTransport):
    def request(self, path: str, payload: Mapping[str, Any]) -> OmlxResponse:
        if path == "/v1/rerank":
            with self.lock:
                self.requests.append((path, payload))
            return OmlxResponse(503, b'{"error":"service unavailable"}')
        return super().request(path, payload)


def _profile(document_batch_size: int = 16) -> Stage2Profile:
    return Stage2Profile(
        query="frozen benchmark topic",
        query_version="benchmark-topic-v1",
        thresholds=ThresholdArtifact("threshold-v1", "fixture-reranker-lock", "raw_reranker_score", -1.0, 2.0),
        reranker_model_id="fixture-reranker",
        reranker_revision="reranker-revision",
        adjudicator_model_id="fixture-qwen",
        adjudicator_revision="qwen-revision",
        screening_scope_hash="0" * 64,
        token_bucket_width=10_000,
        document_batch_size=document_batch_size,
        reranker_max_in_flight=1,
        adjudicator_concurrency=2,
    )


def _papers() -> tuple[Stage2Paper, ...]:
    return (
        Stage2Paper("p-low", "p-low", "irrelevant material"),
        Stage2Paper("p-high", "p-high", "strong topic match"),
        Stage2Paper("p-gray", "p-gray", "forced performance route"),
        Stage2Paper("p-missing", "p-missing", None),
    )


def _cases() -> tuple[PerformanceCase, ...]:
    profile = _profile()
    return tuple(
        PerformanceCase(
            paper.paper_id,
            estimate_omlx_chat_input_token_proxy(adjudication_messages(
                query_version=profile.query_version,
                query=profile.query,
                paper=paper,
            )),
            paper.abstract is None,
        )
        for paper in _papers()
    )


def _environment(document_batch_size: int = 16) -> BenchmarkEnvironment:
    return BenchmarkEnvironment(
        machine_model="Apple Silicon M4 Max",
        memory_gb=36,
        macos_version="15.6",
        omlx_version="0.5.7",
        mlx_version="0.27.0",
        power_mode="automatic",
        background_load="isolated fixture",
        batch_config={"document_batch_size": document_batch_size, "reranker_max_in_flight": 1, "adjudicator_concurrency": 2},
        resident_model_instances={"fixture-reranker-lock": 1, "fixture-qwen-lock": 1},
    )


def _runner(
    tmp_path: Path,
    transport: FakeOmlxTransport,
    *,
    rss_bytes: int = 2 * 1024 ** 3,
    document_batch_size: int = 16,
) -> tuple[Database, Stage2BenchmarkRunner]:
    database = Database(tmp_path / "benchmark.sqlite3")
    database.migrate()
    profile = _profile(document_batch_size)
    schema = json.loads(Path("schemas/filter-decision.schema.json").read_text())
    runner = Stage2BenchmarkRunner.from_omlx(
        database=database,
        profile=profile,
        transport=transport,
        schema=schema,
        environment=_environment(document_batch_size),
        release_hash="release-fixture-hash",
        clock=StepClock(),
        rss_sampler=lambda: rss_bytes,
        rss_scope="fixture_constant_rss",
    )
    return database, runner


def _fallback_release(profile: Stage2Profile):
    pair_ids = ("p-gray", "p-high", "p-low", "p-missing")
    calibrator = PathCalibrator(
        1,
        CalibrationPath.RERANKER,
        1.0,
        0.0,
        "dev",
        "gold",
        "c" * 64,
        "labels",
        _hash(list(pair_ids)),
        4,
        pair_ids,
    )
    return SimpleNamespace(
        model_lock=ModelLock(
            1,
            "omlx_rerank",
            "backup-reranker",
            "backup/repo",
            "revision",
            None,
            None,
            "bf16",
            "none",
            "Apache-2.0",
            4_000_000_000,
            "0.5.7",
            "0.27.0",
            {"model.safetensors": "1" * 64},
        ),
        model_lock_hash="c" * 64,
        calibration=PathCalibration(
            calibrator,
            ProbabilityThresholdArtifact(
                1,
                CalibrationPath.RERANKER,
                0.25,
                0.75,
                calibrator.hash(),
                "c" * 64,
                "dev",
                "labels",
                profile.base_runtime_config_hash,
            ),
        ),
        release_binding=FallbackReleaseBinding(
            "1" * 64,
            "2" * 64,
            "3" * 64,
            "4" * 64,
        ),
    )


def test_performance_runner_executes_omlx_pipeline_and_writes_canonical_measurements(tmp_path) -> None:
    transport = FakeOmlxTransport()
    database, runner = _runner(tmp_path, transport)
    spec = BenchmarkRunSpec.fixture(
        kind="performance",
        scenario="normal",
        cases=_cases(),
        stage2_config_hash=runner.profile.base_runtime_config_hash,
        forced_qwen_pair_ids=("p-gray", "p-missing"),
    )

    record = runner.run(spec, _papers(), run_id="performance-fixture")
    document = record.document()

    assert {path for path, _ in transport.requests} == {"/v1/rerank", "/v1/chat/completions"}
    assert document["case_count"] == 4
    assert document["record_version"] == 2
    assert document["fixture_scale"] is True
    assert document["duration_seconds"] > 0
    assert document["p50_seconds"] <= document["p95_seconds"]
    assert document["papers_per_second"] == pytest.approx(4 / document["duration_seconds"])
    assert document["input_tokens_per_second"] == pytest.approx(sum(case.input_tokens for case in _cases()) / document["duration_seconds"])
    assert document["peak_rss_bytes"] == 2 * 1024 ** 3
    assert document["batch_concurrency"]["document_batch_size"] == 16
    assert document["qwen_pair_ids"] == ["p-gray", "p-missing"]
    assert document["qwen_share"] == 0.5
    assert document["adjudicator_capacity"] == "severe"
    assert document["qwen_capacity_level"] == "severe"
    assert document["alarm_codes"] == [ADJUDICATOR_SHARE_ALARM]
    assert document["frozen_qwen_routing_matches"] is True
    assert document["request_count"] == 4
    assert document["request_count_unit"] == "manifest_case"
    assert document["pair_attempt_count"] == 6
    assert document["sqlite_commit_count"] == 4
    assert document["sqlite_commit_unit"] == "persisted_filter_decision"
    assert document["resume_verified"] is True
    assert document["resume_model_call_count"] == 0
    assert document["resumed_pair_count"] == 4
    assert document["release_hash"] == "release-fixture-hash"
    assert document["observed_stage2_config_hash"] == runner.profile.base_runtime_config_hash
    assert document["full_profile_hash"] == runner.profile.full_profile_hash
    with pytest.raises(ValueError, match="fixture-scale"):
        record.as_performance_record()

    artifact = tmp_path / "artifacts" / "performance.json"
    record.write(artifact)
    assert artifact.read_bytes() == record.canonical_bytes()
    assert json.loads(artifact.read_bytes())["input_hash"] == document["input_hash"]
    with pytest.raises(FileExistsError):
        record.write(artifact)
    assert artifact.read_bytes() == record.canonical_bytes()
    assert database.connection.execute(
        "SELECT COUNT(*) FROM filter_decisions WHERE run_id = 'performance-fixture'"
    ).fetchone()[0] == 4
    gray_reason = json.loads(database.connection.execute(
        "SELECT reason FROM filter_decisions WHERE run_id = 'performance-fixture' AND paper_id = 'p-gray'"
    ).fetchone()[0])
    assert gray_reason["reason_code"].startswith("performance_manifest_forced_qwen:")
    database.close()


def test_benchmark_records_only_real_backup_model_calls_as_fallback(tmp_path) -> None:
    primary = PrimaryUnavailableTransport()
    backup = FakeOmlxTransport()
    database = Database(tmp_path / "backup-benchmark.sqlite3")
    database.migrate()
    profile = replace(
        _profile(),
        reranker_lock_hash="a" * 64,
        release_gate_hash="release-gate",
    )
    schema = json.loads(Path("schemas/filter-decision.schema.json").read_text())
    runner = Stage2BenchmarkRunner.from_omlx(
        database=database,
        profile=profile,
        transport=primary,
        fallback_transport=backup,
        reranker_fallback=_fallback_release(profile),
        schema=schema,
        environment=_environment(),
        release_hash="release-fixture-hash",
        clock=StepClock(),
        rss_sampler=lambda: 2 * 1024 ** 3,
        rss_scope="fixture_constant_rss",
    )
    spec = BenchmarkRunSpec.fixture(
        kind="performance",
        scenario="normal",
        cases=_cases(),
        stage2_config_hash=profile.base_runtime_config_hash,
        forced_qwen_pair_ids=("p-gray", "p-missing"),
    )

    record = runner.run(spec, _papers(), run_id="backup-model-fixture").document()

    assert record["reranker_fallback_count"] == 1
    fallback_trace = record["reranker_fallback_trace"]
    assert fallback_trace[0]["model_lock_hash"] == "c" * 64
    assert fallback_trace[0]["pair_ids"] == ["p-gray", "p-high", "p-low", "p-missing"]
    assert fallback_trace[0]["duration_seconds"] > 0
    assert fallback_trace[0]["failed"] is False
    assert record["needs_review_pair_ids"] == []
    assert {payload["model"] for path, payload in backup.requests if path == "/v1/rerank"} == {"backup-reranker"}
    database.close()


def test_performance_manifest_is_the_exact_qwen_route_and_other_gray_cases_fail_open(
    tmp_path: Path,
) -> None:
    transport = FakeOmlxTransport()
    database, runner = _runner(tmp_path, transport)
    spec = BenchmarkRunSpec.fixture(
        kind="performance",
        scenario="normal",
        cases=_cases(),
        stage2_config_hash=runner.profile.base_runtime_config_hash,
        forced_qwen_pair_ids=("p-missing",),
    )

    record = runner.run(spec, _papers(), run_id="exact-performance-routing")

    assert record.document()["qwen_pair_ids"] == ["p-missing"]
    row = database.connection.execute(
        "SELECT status FROM filter_decisions WHERE run_id = ? AND paper_id = ?",
        ("exact-performance-routing", "p-gray"),
    ).fetchone()
    assert row["status"] == "needs_review"
    database.close()


def test_macos_memory_observer_sums_current_runner_and_omlx_rss() -> None:
    commands: list[tuple[str, ...]] = []

    def command(arguments) -> str:
        commands.append(tuple(arguments))
        if arguments[0] == "/bin/ps":
            return "10 1024\n20 2048\n21 4096\n"
        return "1\n"

    observer = MacOSMemoryObserver(
        runner_pid=10,
        omlx_pids=(20, 21),
        command_runner=command,
        platform_name="darwin",
    )

    observer.preflight()

    assert observer.current_rss_bytes() == 7 * 1024**2
    assert observer.memory_pressure_critical() is False
    assert observer.rss_scope == "macos_ps_current_rss:runner_pid=10;omlx_pids=20,21"
    assert any(command[0] == "/bin/ps" for command in commands)
    assert any(command[0] == "/usr/sbin/sysctl" for command in commands)


def test_macos_memory_observer_fails_when_a_required_process_disappears() -> None:
    observer = MacOSMemoryObserver(
        runner_pid=10,
        omlx_pids=(20,),
        command_runner=lambda arguments: "10 1024\n",
        platform_name="darwin",
    )

    with pytest.raises(RuntimeError, match="lost required process IDs"):
        observer.current_rss_bytes()


def test_macos_memory_observer_rejects_critical_pressure_before_run() -> None:
    observer = MacOSMemoryObserver(
        runner_pid=10,
        omlx_pids=(20,),
        command_runner=lambda arguments: (
            "10 1024\n20 2048\n" if arguments[0] == "/bin/ps" else "4\n"
        ),
        platform_name="darwin",
    )

    with pytest.raises(RuntimeError, match="critical before"):
        observer.preflight()


def test_runner_measures_terminal_schema_failure_and_fail_closed_resume(tmp_path) -> None:
    transport = FakeOmlxTransport(frozenset({"p-gray"}))
    database, runner = _runner(tmp_path, transport)
    spec = BenchmarkRunSpec.fixture(
        kind="performance",
        scenario="stress",
        cases=_cases(),
        stage2_config_hash=runner.profile.base_runtime_config_hash,
        forced_qwen_pair_ids=("p-gray", "p-missing"),
    )

    record = runner.run(spec, _papers(), run_id="failure-fixture").document()

    assert record["backend_failed_call_count"] == 2
    assert record["failed_request_count"] == 1
    assert record["failed_request_pair_ids"] == ["p-gray"]
    assert "p-gray" in record["needs_review_pair_ids"]
    assert "p-gray" not in record["completed_pair_ids"]
    assert set(record["completed_pair_ids"]) | set(record["needs_review_pair_ids"]) == {
        paper.paper_id for paper in _papers()
    }
    assert record["request_count"] == 4
    assert record["pair_attempt_count"] == 7
    assert record["request_failure_rate"] == 0.25
    assert record["resume_verified"] is True
    database.close()


def test_soak_fixture_uses_the_same_measured_runner_without_weakening_production_scale(tmp_path) -> None:
    transport = FakeOmlxTransport()
    database, runner = _runner(tmp_path, transport)
    spec = BenchmarkRunSpec.fixture(
        kind="soak",
        cases=_cases(),
        stage2_config_hash=runner.profile.base_runtime_config_hash,
    )

    record = runner.run(spec, _papers(), run_id="soak-fixture")

    assert record.document()["kind"] == "soak"
    assert record.document()["scenario"] is None
    assert record.document()["sqlite_commit_count"] == 4
    with pytest.raises(ValueError, match="fixture-scale"):
        record.as_soak_record()
    with pytest.raises(ValueError, match="exactly 1,000"):
        BenchmarkRunSpec(
            kind="performance",
            scenario="normal",
            manifest_hash="manifest",
            corpus_hash="corpus",
            stage2_config_hash=runner.profile.base_runtime_config_hash,
            model_lock_hashes=("reranker", "qwen"),
            threshold_artifact_hashes=("threshold",),
            output_token_limit=256,
            cases=_cases(),
        )
    database.close()


def test_omlx_transport_probe_measures_memory_batch_downgrades_and_oom(tmp_path) -> None:
    transport = FakeOmlxTransport(fail_first_multi_rerank=True)
    database, runner = _runner(tmp_path, transport, document_batch_size=32)
    papers = tuple(
        Stage2Paper(f"paper-{index:02d}", f"paper-{index:02d}", "topic match")
        for index in range(20)
    )
    cases = tuple(
        PerformanceCase(
            paper.paper_id,
            estimate_omlx_chat_input_token_proxy(adjudication_messages(
                query_version=runner.profile.query_version,
                query=runner.profile.query,
                paper=paper,
            )),
            False,
        )
        for paper in papers
    )
    spec = BenchmarkRunSpec.fixture(
        kind="performance",
        scenario="normal",
        cases=cases,
        stage2_config_hash=runner.profile.base_runtime_config_hash,
        forced_qwen_pair_ids=(papers[0].paper_id,),
    )

    record = runner.run(spec, papers, run_id="fallback-fixture").document()

    assert record["reranker_fallback_measurement_available"] is True
    assert record["reranker_fallback_count"] == 0
    assert record["reranker_fallback_trace"] == []
    assert record["service_request_count"] == 4
    assert record["service_failed_request_count"] == 1
    assert record["service_request_failure_rate"] == pytest.approx(
        1 / record["service_request_count"]
    )
    assert record["oom"] is True
    assert [
        item["resource_exhausted"] for item in record["service_request_trace"]
    ] == [True, False, False, False]
    assert ERROR_RATE_ALARM in record["alarm_codes"]
    assert record["request_count"] == 20
    assert record["failed_request_count"] == 0
    assert record["request_failure_rate"] == 0
    assert record["service_pair_attempt_count"] > record["pair_attempt_count"]
    assert record["latency_by_path"]["reranker"]["sample_count"] == 3
    tampered = {**record, "oom": False}
    with pytest.raises(ValueError, match="oom flag"):
        BenchmarkExecutionRecord(tampered)
    database.close()


def test_runner_alarms_on_terminal_schema_failure_even_when_http_requests_succeed(
    tmp_path,
) -> None:
    transport = FakeOmlxTransport(malformed_chat_ids=frozenset({"p-gray"}))
    database, runner = _runner(tmp_path, transport)
    spec = BenchmarkRunSpec.fixture(
        kind="performance",
        scenario="normal",
        cases=_cases(),
        stage2_config_hash=runner.profile.base_runtime_config_hash,
        forced_qwen_pair_ids=("p-gray", "p-missing"),
    )

    record = runner.run(spec, _papers(), run_id="terminal-error-fixture").document()

    assert record["request_failure_rate"] == 0.25
    assert record["service_request_failure_rate"] == 0
    assert ERROR_RATE_ALARM in record["alarm_codes"]
    database.close()


@pytest.mark.parametrize(("rss_gb", "expected"), ((28, False), (29, True)))
def test_runner_uses_a_strict_28_gb_memory_watermark(
    tmp_path, rss_gb, expected
) -> None:
    transport = FakeOmlxTransport()
    database, runner = _runner(tmp_path, transport, rss_bytes=rss_gb * 1024 ** 3)
    spec = BenchmarkRunSpec.fixture(
        kind="performance",
        scenario="normal",
        cases=_cases(),
        stage2_config_hash=runner.profile.base_runtime_config_hash,
        forced_qwen_pair_ids=("p-gray", "p-missing"),
    )

    record = runner.run(spec, _papers(), run_id="memory-alarm-fixture").document()

    assert record["peak_memory_gb"] == rss_gb
    assert (MEMORY_WATERMARK_ALARM in record["alarm_codes"]) is expected
    database.close()


def test_benchmark_error_alarm_includes_exact_half_percent_from_either_rate() -> None:
    common = {
        "adjudicator_alarms": (),
        "peak_memory_gb": 28,
        "memory_pressure_critical": False,
        "unbounded_memory_growth": False,
    }

    below = benchmark_alarm_codes(
        **common,
        request_failure_rate=0.0049,
        service_request_failure_rate=0.0049,
    )
    terminal_boundary = benchmark_alarm_codes(
        **common,
        request_failure_rate=0.005,
        service_request_failure_rate=0.0,
    )
    service_boundary = benchmark_alarm_codes(
        **common,
        request_failure_rate=0.0,
        service_request_failure_rate=0.005,
    )

    assert ERROR_RATE_ALARM not in below
    assert ERROR_RATE_ALARM in terminal_boundary
    assert ERROR_RATE_ALARM in service_boundary


def test_runner_rejects_input_drift_before_warmup(tmp_path) -> None:
    transport = FakeOmlxTransport()
    database, runner = _runner(tmp_path, transport)
    cases = list(_cases())
    cases[-1] = PerformanceCase("p-missing", cases[-1].input_tokens, False)
    spec = BenchmarkRunSpec.fixture(
        kind="soak",
        cases=cases,
        stage2_config_hash=runner.profile.base_runtime_config_hash,
    )

    with pytest.raises(ValueError, match="missing-abstract"):
        runner.run(spec, _papers(), run_id="drift-fixture")

    assert transport.requests == []
    database.close()


def test_runner_rejects_tampered_prompt_token_proxy_before_transport(tmp_path) -> None:
    transport = FakeOmlxTransport()
    database, runner = _runner(tmp_path, transport)
    cases = list(_cases())
    first = cases[0]
    cases[0] = PerformanceCase(
        first.pair_id, first.input_tokens + 1, first.abstract_missing,
    )
    spec = BenchmarkRunSpec.fixture(
        kind="soak",
        cases=cases,
        stage2_config_hash=runner.profile.base_runtime_config_hash,
    )

    with pytest.raises(ValueError, match="prompt proxy estimator"):
        runner.run(spec, _papers(), run_id="token-proxy-drift")

    assert transport.requests == []
    database.close()
