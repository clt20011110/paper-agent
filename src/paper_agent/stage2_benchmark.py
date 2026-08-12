"""Measured Stage 2 performance and soak benchmark execution.

Unlike :mod:`paper_agent.stage2_commands`, this module runs the real Stage 2
pipeline.  The manifest supplies frozen inputs and provenance; elapsed time,
RSS, model calls, decisions, SQLite commits, and resume behaviour are observed
during this process and are never accepted from an external JSON document.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from math import ceil, isfinite
import os
from pathlib import Path
import json
import resource
import subprocess
import sys
from tempfile import mkstemp
from threading import Event, Lock, Thread
from time import monotonic
from types import MappingProxyType
from typing import Any, Literal

from .canonical import canonical_json, content_hash
from .domain import FilterStatus
from .schema import schema_directory
from .stage2_backends import (
    AdjudicationDecision,
    AdjudicationInput,
    AdjudicatorBackend,
    CascadeRoute,
    OmlxChatBackend,
    OmlxRerankBackend,
    OmlxResponse,
    OmlxTransport,
    RerankInput,
    RerankScore,
    RerankerBackend,
    omlx_response_is_memory_exhaustion,
)
from .stage2_evaluation import (
    BenchmarkEnvironment,
    PerformanceCase,
    PerformanceRoutingManifest,
    PerformanceRunRecord,
    SoakManifest,
    SoakRunRecord,
)
from .stage2_fallback import LocalCalibratedRerankerFallback
from .stage2_pipeline import (
    ERROR_RATE_ALARM,
    MEMORY_WATERMARK_ALARM,
    TERMINAL_TECHNICAL_REASONS,
    Stage2Paper,
    Stage2Pipeline,
    Stage2Profile,
    adjudicator_capacity,
)
from .stage2_prompt_contract import (
    OMLX_CHAT_INPUT_TOKEN_PROXY_ESTIMATOR,
    adjudication_messages,
    estimate_omlx_chat_input_token_proxy,
)
from .storage import Database


BenchmarkKind = Literal["performance", "soak"]
Clock = Callable[[], float]
RssSampler = Callable[[], int]
PressureSampler = Callable[[], bool]
CommandRunner = Callable[[Sequence[str]], str]
_COMPONENTS = ("rules", "reranker", "qwen", "schema_validation", "sqlite_commit")
BENCHMARK_EXECUTION_FIELDS = frozenset({
    "record_version",
    "measurement_evidence_version",
    "kind",
    "scenario",
    "run_id",
    "manifest_hash",
    "corpus_hash",
    "input_hash",
    "release_hash",
    "stage2_config_hash",
    "observed_stage2_config_hash",
    "full_profile_hash",
    "model_lock_hashes",
    "threshold_artifact_hashes",
    "observed_threshold_artifact_hashes",
    "model_releases",
    "prompt_hash",
    "schema_hash",
    "output_token_limit",
    "observed_output_token_limit",
    "fixture_scale",
    "case_count",
    "input_token_count",
    "duration_seconds",
    "p50_seconds",
    "p95_seconds",
    "latency_sample_count",
    "latency_sample_unit",
    "latency_by_path",
    "papers_per_second",
    "input_tokens_per_second",
    "pair_tokens_per_second",
    "peak_memory_gb",
    "rss_start_bytes",
    "rss_end_bytes",
    "peak_rss_bytes",
    "rss_sample_count",
    "rss_samples_bytes",
    "memory_pressure_samples",
    "rss_sample_interval_seconds",
    "rss_scope",
    "request_count",
    "request_count_unit",
    "request_failure_rate",
    "pair_attempt_count",
    "model_call_count",
    "model_call_trace",
    "service_request_count",
    "service_request_trace",
    "service_pair_attempt_count",
    "service_failed_request_count",
    "service_request_failure_rate",
    "reranker_batch_call_count",
    "reranker_fallback_count",
    "reranker_fallback_trace",
    "reranker_fallback_measurement_available",
    "adjudicator_call_count",
    "backend_failed_call_count",
    "backend_call_failure_rate",
    "failed_request_count",
    "completed_pair_ids",
    "needs_review_pair_ids",
    "failed_request_pair_ids",
    "qwen_pair_ids",
    "qwen_count",
    "qwen_share",
    "qwen_share_alarms",
    "qwen_capacity_level",
    "adjudicator_count",
    "adjudicator_share",
    "adjudicator_capacity",
    "alarm_codes",
    "frozen_qwen_routing_matches",
    "routing_mode",
    "batch_concurrency",
    "environment",
    "expected_components",
    "executed_components",
    "sqlite_commit_count",
    "sqlite_commit_unit",
    "result_count",
    "missing_result_count",
    "duplicate_result_count",
    "warmed",
    "resume_verified",
    "resume_model_call_count",
    "resume_service_request_trace",
    "resumed_pair_count",
    "resumed_pair_ids",
    "oom",
    "process_crash",
    "memory_pressure_critical",
    "memory_pressure_sampled",
    "memory_growth_detector",
    "unbounded_memory_growth",
})


def benchmark_alarm_codes(
    adjudicator_alarms: Sequence[str],
    *,
    request_failure_rate: float,
    service_request_failure_rate: float | None,
    peak_memory_gb: float,
    memory_pressure_critical: bool,
    unbounded_memory_growth: bool,
) -> tuple[str, ...]:
    alarms = list(adjudicator_alarms)
    if request_failure_rate >= 0.005 or (
        service_request_failure_rate is not None
        and service_request_failure_rate >= 0.005
    ):
        alarms.append(ERROR_RATE_ALARM)
    if peak_memory_gb > 28 or memory_pressure_critical or unbounded_memory_growth:
        alarms.append(MEMORY_WATERMARK_ALARM)
    return tuple(alarms)


def _process_rss_bytes() -> int:
    """Return the current process high-water RSS using platform units."""

    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _command_output(arguments: Sequence[str]) -> str:
    try:
        completed = subprocess.run(
            tuple(arguments),
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
    except OSError as error:
        raise RuntimeError(f"memory observation command is unavailable: {arguments[0]}") from error
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no diagnostic output"
        raise RuntimeError(f"memory observation command failed: {detail}")
    return completed.stdout


@dataclass(frozen=True, slots=True)
class MacOSMemoryObserver:
    """Sample current RSS for this runner plus explicit oMLX service processes."""

    runner_pid: int
    omlx_pids: tuple[int, ...]
    command_runner: CommandRunner = field(default=_command_output, repr=False, compare=False)
    platform_name: str = field(default_factory=lambda: sys.platform, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "omlx_pids", tuple(self.omlx_pids))
        if self.platform_name != "darwin":
            raise RuntimeError("production Stage 2 memory observation requires macOS")
        if self.runner_pid < 1 or not self.omlx_pids or any(pid < 1 for pid in self.omlx_pids):
            raise ValueError("runner and oMLX process IDs must be positive")
        process_ids = (self.runner_pid, *self.omlx_pids)
        if len(process_ids) != len(set(process_ids)):
            raise ValueError("runner and oMLX process IDs must be unique")

    @classmethod
    def current(cls, omlx_pids: Sequence[int]) -> MacOSMemoryObserver:
        return cls(os.getpid(), tuple(omlx_pids))

    @property
    def rss_scope(self) -> str:
        omlx = ",".join(str(pid) for pid in self.omlx_pids)
        return f"macos_ps_current_rss:runner_pid={self.runner_pid};omlx_pids={omlx}"

    def current_rss_bytes(self) -> int:
        process_ids = (self.runner_pid, *self.omlx_pids)
        output = self.command_runner((
            "/bin/ps",
            "-o",
            "pid=,rss=",
            "-p",
            ",".join(str(pid) for pid in process_ids),
        ))
        rss_by_pid: dict[int, int] = {}
        for line in output.splitlines():
            fields = line.split()
            if len(fields) != 2:
                raise RuntimeError("macOS ps returned an invalid current-RSS sample")
            try:
                pid, rss_kib = (int(field) for field in fields)
            except ValueError as error:
                raise RuntimeError("macOS ps returned a non-numeric current-RSS sample") from error
            rss_by_pid[pid] = rss_kib
        missing = sorted(set(process_ids) - set(rss_by_pid))
        if missing:
            raise RuntimeError(f"memory observation lost required process IDs: {missing}")
        return sum(rss_by_pid[pid] for pid in process_ids) * 1024

    def memory_pressure_critical(self) -> bool:
        output = self.command_runner((
            "/usr/sbin/sysctl",
            "-n",
            "kern.memorystatus_vm_pressure_level",
        )).strip()
        try:
            level = int(output)
        except ValueError as error:
            raise RuntimeError("macOS returned an invalid memory-pressure level") from error
        if level not in {1, 2, 4}:
            raise RuntimeError(f"macOS returned an unsupported memory-pressure level: {level}")
        return level == 4

    def validate_environment(self, environment: BenchmarkEnvironment) -> None:
        hardware = json.loads(self.command_runner((
            "/usr/sbin/system_profiler",
            "SPHardwareDataType",
            "-json",
        )))
        rows = hardware.get("SPHardwareDataType") if isinstance(hardware, dict) else None
        if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
            raise RuntimeError("system_profiler returned invalid hardware provenance")
        row = rows[0]
        if row.get("chip_type") != "Apple M4 Max" or row.get("physical_memory") != "36 GB":
            raise RuntimeError("benchmark host is not the frozen Apple M4 Max / 36 GB target")
        macos_version = self.command_runner(("/usr/bin/sw_vers", "-productVersion")).strip()
        if macos_version != environment.macos_version:
            raise RuntimeError(
                "benchmark environment macOS version does not match the measured host"
            )

    def preflight(self, environment: BenchmarkEnvironment | None = None) -> None:
        if self.current_rss_bytes() > 28 * 1024**3:
            raise RuntimeError("benchmark processes already exceed the 28 GB memory gate")
        if self.memory_pressure_critical():
            raise RuntimeError("macOS memory pressure is critical before the benchmark")
        if environment is not None:
            self.validate_environment(environment)


@dataclass(frozen=True, slots=True)
class BenchmarkRunSpec:
    """One frozen production run, or an explicitly marked small test fixture."""

    kind: BenchmarkKind
    scenario: str | None
    manifest_hash: str
    corpus_hash: str
    stage2_config_hash: str
    model_lock_hashes: tuple[str, ...]
    threshold_artifact_hashes: tuple[str, ...]
    output_token_limit: int
    cases: tuple[PerformanceCase, ...]
    forced_qwen_pair_ids: frozenset[str] = frozenset()
    executed_components: tuple[str, ...] = _COMPONENTS
    fixture_scale: bool = False
    input_token_estimator: str = OMLX_CHAT_INPUT_TOKEN_PROXY_ESTIMATOR

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_lock_hashes", tuple(self.model_lock_hashes))
        object.__setattr__(self, "threshold_artifact_hashes", tuple(self.threshold_artifact_hashes))
        object.__setattr__(self, "cases", tuple(self.cases))
        object.__setattr__(self, "forced_qwen_pair_ids", frozenset(self.forced_qwen_pair_ids))
        object.__setattr__(self, "executed_components", tuple(self.executed_components))
        if self.kind not in {"performance", "soak"}:
            raise ValueError("benchmark kind must be performance or soak")
        if self.kind == "performance" and self.scenario not in {"normal", "stress"}:
            raise ValueError("performance benchmark scenario must be normal or stress")
        if self.kind == "soak" and self.scenario is not None:
            raise ValueError("soak benchmark does not have a performance scenario")
        if self.kind == "soak" and self.forced_qwen_pair_ids:
            raise ValueError("soak benchmark must use released quality-threshold routing")
        required = (self.manifest_hash, self.corpus_hash, self.stage2_config_hash)
        if not all(required) or not self.model_lock_hashes or not self.threshold_artifact_hashes:
            raise ValueError("benchmark spec requires complete hash provenance")
        if len(set(self.model_lock_hashes)) != len(self.model_lock_hashes):
            raise ValueError("benchmark model lock hashes must be unique")
        if self.output_token_limit < 1 or not self.cases:
            raise ValueError("benchmark spec requires cases and a positive output token limit")
        if self.input_token_estimator != OMLX_CHAT_INPUT_TOKEN_PROXY_ESTIMATOR:
            raise ValueError("benchmark spec requires the released input-token proxy estimator")
        pair_ids = [case.pair_id for case in self.cases]
        if len(pair_ids) != len(set(pair_ids)) or not self.forced_qwen_pair_ids <= set(pair_ids):
            raise ValueError("benchmark pair IDs must be unique and contain all frozen Qwen routes")
        if self.executed_components != _COMPONENTS:
            raise ValueError("benchmark must execute the complete Stage 2 pipeline")
        expected = 1_000 if self.kind == "performance" else 10_000
        if self.fixture_scale:
            if len(self.cases) >= expected:
                raise ValueError("fixture-scale benchmark must remain below the production case count")
        else:
            if len(self.cases) != expected:
                raise ValueError(f"{self.kind} benchmark requires exactly {expected:,} cases")
            if self.kind == "performance":
                if sum(case.abstract_missing for case in self.cases) != 100:
                    raise ValueError("performance benchmark requires exactly 10% missing abstracts")
                qwen_count = 150 if self.scenario == "normal" else 300
                if len(self.forced_qwen_pair_ids) != qwen_count:
                    raise ValueError("performance benchmark requires exactly 15%/30% frozen Qwen routing")

    @classmethod
    def performance(cls, manifest: PerformanceRoutingManifest, scenario: str) -> BenchmarkRunSpec:
        if scenario == "normal":
            qwen_ids = manifest.normal_qwen_ids
        elif scenario == "stress":
            qwen_ids = manifest.stress_qwen_ids
        else:
            raise ValueError("performance scenario must be normal or stress")
        return cls(
            kind="performance",
            scenario=scenario,
            manifest_hash=manifest.hash(),
            corpus_hash=manifest.corpus_hash,
            stage2_config_hash=manifest.stage2_config_hash,
            model_lock_hashes=manifest.model_lock_hashes,
            threshold_artifact_hashes=manifest.threshold_artifact_hashes,
            output_token_limit=manifest.output_token_limit,
            cases=manifest.cases,
            input_token_estimator=manifest.input_token_estimator,
            forced_qwen_pair_ids=qwen_ids,
            executed_components=manifest.pipeline_components,
        )

    @classmethod
    def soak(cls, manifest: SoakManifest) -> BenchmarkRunSpec:
        return cls(
            kind="soak",
            scenario=None,
            manifest_hash=manifest.hash(),
            corpus_hash=manifest.corpus_hash,
            stage2_config_hash=manifest.stage2_config_hash,
            model_lock_hashes=manifest.model_lock_hashes,
            threshold_artifact_hashes=manifest.threshold_artifact_hashes,
            output_token_limit=manifest.output_token_limit,
            cases=manifest.cases,
            input_token_estimator=manifest.input_token_estimator,
        )

    @classmethod
    def fixture(
        cls,
        *,
        kind: BenchmarkKind,
        cases: Sequence[PerformanceCase],
        stage2_config_hash: str,
        scenario: str | None = None,
        forced_qwen_pair_ids: Sequence[str] = (),
        corpus_hash: str = "fixture-corpus",
        model_lock_hashes: tuple[str, ...] = ("fixture-reranker-lock", "fixture-qwen-lock"),
        threshold_artifact_hashes: tuple[str, ...] = ("fixture-threshold",),
        output_token_limit: int = 256,
        input_token_estimator: str = OMLX_CHAT_INPUT_TOKEN_PROXY_ESTIMATOR,
    ) -> BenchmarkRunSpec:
        """Build a deliberately non-production-scale spec for offline tests."""

        case_tuple = tuple(cases)
        identity = {
            "kind": kind,
            "scenario": scenario,
            "corpus_hash": corpus_hash,
            "stage2_config_hash": stage2_config_hash,
            "model_lock_hashes": list(model_lock_hashes),
            "threshold_artifact_hashes": list(threshold_artifact_hashes),
            "output_token_limit": output_token_limit,
            "input_token_estimator": input_token_estimator,
            "cases": [[item.pair_id, item.input_tokens, item.abstract_missing] for item in case_tuple],
            "forced_qwen_pair_ids": sorted(forced_qwen_pair_ids),
            "fixture_scale": True,
        }
        return cls(
            kind=kind,
            scenario=scenario,
            manifest_hash=content_hash(identity),
            corpus_hash=corpus_hash,
            stage2_config_hash=stage2_config_hash,
            model_lock_hashes=model_lock_hashes,
            threshold_artifact_hashes=threshold_artifact_hashes,
            output_token_limit=output_token_limit,
            cases=case_tuple,
            input_token_estimator=input_token_estimator,
            forced_qwen_pair_ids=frozenset(forced_qwen_pair_ids),
            fixture_scale=True,
        )


@dataclass(frozen=True, slots=True)
class BenchmarkExecutionRecord:
    """A canonical, immutable record produced only by a measured execution."""

    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        if set(self.payload) != BENCHMARK_EXECUTION_FIELDS:
            raise ValueError("benchmark execution record must use the exact evidence-v1 fields")
        service_trace = self.payload["service_request_trace"]
        if not isinstance(service_trace, Sequence) or isinstance(service_trace, (str, bytes)):
            raise ValueError("benchmark service request trace must be a sequence")
        for request in service_trace:
            if (
                not isinstance(request, Mapping)
                or type(request.get("resource_exhausted")) is not bool
                or (
                    request["resource_exhausted"]
                    and (request.get("path") != "/v1/rerank" or request.get("failed") is not True)
                )
            ):
                raise ValueError("benchmark service request resource observation is invalid")
        if self.payload["oom"] != any(
            request["resource_exhausted"] for request in service_trace
        ):
            raise ValueError("benchmark oom flag does not match service request observations")
        frozen = _freeze_json(dict(self.payload))
        assert isinstance(frozen, Mapping)
        object.__setattr__(self, "payload", frozen)

    def document(self) -> dict[str, Any]:
        document = _thaw_json(self.payload)
        assert isinstance(document, dict)
        return document

    def canonical_bytes(self) -> bytes:
        return canonical_json(self.document())

    def hash(self) -> str:
        return content_hash(self.document())

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(self.canonical_bytes())
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)

    def as_performance_record(self) -> PerformanceRunRecord:
        if self.payload["kind"] != "performance":
            raise ValueError("only performance executions convert to PerformanceRunRecord")
        if self.payload["fixture_scale"]:
            raise ValueError("fixture-scale benchmark records cannot enter production gates")
        return PerformanceRunRecord(**_evaluation_fields(self.payload, performance=True))

    def as_soak_record(self) -> SoakRunRecord:
        if self.payload["kind"] != "soak":
            raise ValueError("only soak executions convert to SoakRunRecord")
        if self.payload["fixture_scale"]:
            raise ValueError("fixture-scale benchmark records cannot enter production gates")
        return SoakRunRecord(**_evaluation_fields(self.payload, performance=False))


@dataclass(slots=True)
class _PeakRss:
    sampler: RssSampler
    pressure_sampler: PressureSampler | None = None
    samples: list[int] = field(default_factory=list)
    pressure_samples: list[bool] = field(default_factory=list)
    _lock: Lock = field(default_factory=Lock)

    def observe(self) -> int:
        with self._lock:
            value = int(self.sampler())
            if value < 0:
                raise ValueError("RSS sampler cannot return a negative value")
            pressure = bool(self.pressure_sampler()) if self.pressure_sampler is not None else None
            self.samples.append(value)
            if pressure is not None:
                self.pressure_samples.append(pressure)
        return value

    def reset(self) -> None:
        with self._lock:
            self.samples.clear()
            self.pressure_samples.clear()


@dataclass(slots=True)
class _PeriodicRssMonitor:
    rss: _PeakRss
    interval_seconds: float
    _stop: Event = field(default_factory=Event)
    _thread: Thread | None = None
    error: Exception | None = None

    def start(self) -> None:
        self._stop.clear()
        self.error = None
        self._thread = Thread(target=self._sample, name="stage2-benchmark-rss", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    def _sample(self) -> None:
        while not self._stop.is_set():
            try:
                self.rss.observe()
            except Exception as error:
                self.error = error
                self._stop.set()
                return
            self._stop.wait(self.interval_seconds)


@dataclass(slots=True)
class _MeasuredReranker:
    backend: RerankerBackend
    clock: Clock
    call_durations: list[float] = field(default_factory=list)
    batch_sizes: list[int] = field(default_factory=list)
    pair_ids: list[tuple[str, ...]] = field(default_factory=list)
    call_failures: list[bool] = field(default_factory=list)
    pair_attempt_count: int = 0
    call_count: int = 0
    failed_call_count: int = 0
    _lock: Lock = field(default_factory=Lock)

    @property
    def backend_name(self) -> str:
        return self.backend.backend_name

    @property
    def is_local(self) -> bool:
        return getattr(self.backend, "is_local", False)

    def rerank(self, query: str, documents: Sequence[RerankInput]) -> tuple[RerankScore, ...]:
        started = self.clock()
        failed = False
        try:
            return self.backend.rerank(query, documents)
        except Exception:
            failed = True
            raise
        finally:
            duration = max(0.0, self.clock() - started)
            with self._lock:
                self.call_count += 1
                self.pair_attempt_count += len(documents)
                self.failed_call_count += int(failed)
                self.call_durations.append(duration)
                self.batch_sizes.append(len(documents))
                self.pair_ids.append(tuple(item.paper_id for item in documents))
                self.call_failures.append(failed)

    def reset(self) -> None:
        with self._lock:
            self.call_durations.clear()
            self.batch_sizes.clear()
            self.pair_ids.clear()
            self.call_failures.clear()
            self.pair_attempt_count = self.call_count = self.failed_call_count = 0


@dataclass(slots=True)
class _MeasuredAdjudicator:
    backend: AdjudicatorBackend
    clock: Clock
    call_durations: list[float] = field(default_factory=list)
    pair_ids: list[str] = field(default_factory=list)
    call_failures: list[bool] = field(default_factory=list)
    call_count: int = 0
    failed_call_count: int = 0
    _lock: Lock = field(default_factory=Lock)

    @property
    def backend_name(self) -> str:
        return self.backend.backend_name

    def adjudicate(self, request: AdjudicationInput) -> AdjudicationDecision:
        started = self.clock()
        failed = False
        try:
            return self.backend.adjudicate(request)
        except Exception:
            failed = True
            raise
        finally:
            duration = max(0.0, self.clock() - started)
            with self._lock:
                self.call_count += 1
                self.failed_call_count += int(failed)
                self.pair_ids.append(request.paper_id)
                self.call_durations.append(duration)
                self.call_failures.append(failed)

    def reset(self) -> None:
        with self._lock:
            self.call_durations.clear()
            self.pair_ids.clear()
            self.call_failures.clear()
            self.call_count = self.failed_call_count = 0


@dataclass(frozen=True, slots=True)
class _ServiceRequest:
    path: str
    duration_seconds: float
    document_count: int
    failed: bool
    resource_exhausted: bool


@dataclass(slots=True)
class _MeasuredOmlxTransport:
    """Observe the concrete HTTP-compatible requests made inside oMLX backends."""

    transport: OmlxTransport
    clock: Clock
    requests: list[_ServiceRequest] = field(default_factory=list)
    _lock: Lock = field(default_factory=Lock)

    def request(self, path: str, payload: Mapping[str, Any]) -> OmlxResponse:
        started = self.clock()
        failed = False
        resource_exhausted = False
        try:
            response = self.transport.request(path, payload)
            failed = response.status_code != 200
            resource_exhausted = (
                path == "/v1/rerank" and omlx_response_is_memory_exhaustion(response)
            )
            return response
        except Exception as error:
            failed = True
            resource_exhausted = path == "/v1/rerank" and isinstance(error, MemoryError)
            raise
        finally:
            duration = max(0.0, self.clock() - started)
            documents = payload.get("documents")
            document_count = len(documents) if isinstance(documents, Sequence) else 1
            with self._lock:
                self.requests.append(_ServiceRequest(
                    path,
                    duration,
                    document_count,
                    failed,
                    resource_exhausted,
                ))

    def reset(self) -> None:
        with self._lock:
            self.requests.clear()


class _BenchmarkPipeline(Stage2Pipeline):
    """Performance-only frozen routing layered on the production pipeline."""

    def __init__(self, *args: Any, forced_qwen_pair_ids: frozenset[str], **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._forced_qwen_pair_ids = forced_qwen_pair_ids

    def _reranker_route(self, paper: Stage2Paper, score: float) -> tuple[CascadeRoute, float | None]:
        route, probability = super()._reranker_route(paper, score)
        if paper.paper_id in self._forced_qwen_pair_ids:
            return CascadeRoute.ADJUDICATE, probability
        if route is CascadeRoute.ADJUDICATE:
            return CascadeRoute.NEEDS_REVIEW, probability
        return route, probability

    def _adjudication_reason(self, paper: Stage2Paper) -> str:
        if paper.paper_id in self._forced_qwen_pair_ids:
            return "performance_manifest_forced_qwen"
        return super()._adjudication_reason(paper)


@dataclass(slots=True)
class Stage2BenchmarkRunner:
    """Execute a frozen Stage 2 workload and produce its measured record."""

    database: Database
    profile: Stage2Profile
    reranker: RerankerBackend
    adjudicator: AdjudicatorBackend
    environment: BenchmarkEnvironment
    release_hash: str
    adjudicator_output_token_limit: int = 256
    clock: Clock = monotonic
    rss_sampler: RssSampler = _process_rss_bytes
    rss_scope: str = "runner_process_high_water_rss"
    rss_sample_interval_seconds: float = 0.25
    memory_pressure_sampler: PressureSampler | None = None
    reranker_fallback: LocalCalibratedRerankerFallback | None = None
    _transport_probe: _MeasuredOmlxTransport | None = field(default=None, repr=False)
    _fallback_transport_probe: _MeasuredOmlxTransport | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if (
            not self.release_hash
            or not self.rss_scope
            or self.adjudicator_output_token_limit < 1
            or not isfinite(self.rss_sample_interval_seconds)
            or self.rss_sample_interval_seconds <= 0
        ):
            raise ValueError("benchmark runner requires release and RSS provenance")

    @classmethod
    def from_omlx(
        cls,
        *,
        database: Database,
        profile: Stage2Profile,
        transport: OmlxTransport,
        schema: Mapping[str, Any],
        environment: BenchmarkEnvironment,
        release_hash: str,
        clock: Clock = monotonic,
        rss_sampler: RssSampler = _process_rss_bytes,
        rss_scope: str = "runner_process_high_water_rss",
        rss_sample_interval_seconds: float = 0.25,
        memory_pressure_sampler: PressureSampler | None = None,
        reranker_fallback: Any | None = None,
        fallback_transport: OmlxTransport | None = None,
    ) -> Stage2BenchmarkRunner:
        """Bind the measured runner to the concrete oMLX production backends."""

        expected_schema = json.loads((schema_directory() / profile.schema_version).read_text(encoding="utf-8"))
        if dict(schema) != expected_schema:
            raise ValueError("oMLX benchmark schema does not match the released Stage 2 schema")
        probe = _MeasuredOmlxTransport(transport, clock)
        reranker = OmlxRerankBackend(
            profile.reranker_model_id,
            probe,
            document_batch_size=profile.document_batch_size,
            max_in_flight=profile.reranker_max_in_flight,
        )
        adjudicator = OmlxChatBackend(
            profile.adjudicator_model_id,
            probe,
            schema,
            seed=profile.adjudicator_seed,
            max_context_window=profile.adjudicator_max_context_window,
            max_output_tokens=profile.adjudicator_max_output_tokens,
        )
        fallback_probe: _MeasuredOmlxTransport | None = None
        fallback: LocalCalibratedRerankerFallback | None = None
        if reranker_fallback is not None:
            fallback_probe = _MeasuredOmlxTransport(
                fallback_transport or transport,
                clock,
            )
            fallback = LocalCalibratedRerankerFallback(
                backend=OmlxRerankBackend(
                    reranker_fallback.model_lock.model_id,
                    fallback_probe,
                    document_batch_size=profile.document_batch_size,
                    max_in_flight=profile.reranker_max_in_flight,
                ),
                model_id=reranker_fallback.model_lock.model_id,
                model_revision=(
                    reranker_fallback.model_lock.conversion_revision
                    or reranker_fallback.model_lock.source_revision
                ),
                model_lock_hash=reranker_fallback.model_lock_hash,
                calibration=reranker_fallback.calibration,
                release_binding=reranker_fallback.release_binding,
                runtime_config_hash=reranker_fallback.runtime_config_hash,
            )
        return cls(
            database=database,
            profile=profile,
            reranker=reranker,
            adjudicator=adjudicator,
            environment=environment,
            release_hash=release_hash,
            adjudicator_output_token_limit=profile.adjudicator_max_output_tokens,
            clock=clock,
            rss_sampler=rss_sampler,
            rss_scope=rss_scope,
            rss_sample_interval_seconds=rss_sample_interval_seconds,
            memory_pressure_sampler=memory_pressure_sampler,
            reranker_fallback=fallback,
            _transport_probe=probe,
            _fallback_transport_probe=fallback_probe,
        )

    def run(
        self,
        spec: BenchmarkRunSpec,
        papers: Sequence[Stage2Paper],
        *,
        run_id: str,
        warmup_paper: Stage2Paper | None = None,
        verify_resume: bool = True,
    ) -> BenchmarkExecutionRecord:
        if not run_id:
            raise ValueError("benchmark run_id is required")
        ordered_papers = self.validate(spec, papers, verify_resume=verify_resume)
        self._ensure_papers((*ordered_papers, *((warmup_paper,) if warmup_paper else ())))

        rss = _PeakRss(self.rss_sampler, self.memory_pressure_sampler)
        reranker = _MeasuredReranker(self.reranker, self.clock)
        fallback_reranker = (
            _MeasuredReranker(self.reranker_fallback.backend, self.clock)
            if self.reranker_fallback is not None
            else None
        )
        adjudicator = _MeasuredAdjudicator(self.adjudicator, self.clock)
        measured_fallback = (
            LocalCalibratedRerankerFallback(
                backend=fallback_reranker,
                model_id=self.reranker_fallback.model_id,
                model_revision=self.reranker_fallback.model_revision,
                model_lock_hash=self.reranker_fallback.model_lock_hash,
                calibration=self.reranker_fallback.calibration,
                release_binding=self.reranker_fallback.release_binding,
                runtime_config_hash=self.reranker_fallback.runtime_config_hash,
            )
            if fallback_reranker is not None and self.reranker_fallback is not None
            else None
        )
        pipeline = _BenchmarkPipeline(
            self.database,
            reranker,
            adjudicator,
            self.profile,
            forced_qwen_pair_ids=spec.forced_qwen_pair_ids,
            reranker_fallback=measured_fallback,
        )

        warm = warmup_paper or self._default_warmup_paper(ordered_papers)
        warmup = _BenchmarkPipeline(
            self.database,
            reranker,
            adjudicator,
            self.profile,
            forced_qwen_pair_ids=frozenset({warm.paper_id}),
            reranker_fallback=measured_fallback,
        ).run(f"{run_id}--warmup", (warm,))
        if warmup.reranked_count != 1 or warmup.qwen_count != 1:
            raise RuntimeError("benchmark warm-up did not execute both Stage 2 models")
        reranker.reset()
        if fallback_reranker is not None:
            fallback_reranker.reset()
        adjudicator.reset()
        rss.reset()
        if self._transport_probe is not None:
            self._transport_probe.reset()
        if self._fallback_transport_probe is not None:
            self._fallback_transport_probe.reset()

        rss_start = rss.observe()
        started = self.clock()
        monitor = _PeriodicRssMonitor(rss, self.rss_sample_interval_seconds)
        monitor.start()
        try:
            summary = pipeline.run(run_id, ordered_papers)
        finally:
            monitor.stop()
        if monitor.error is not None:
            raise RuntimeError("periodic RSS sampler failed") from monitor.error
        duration = max(0.0, self.clock() - started)
        rss_end = rss.observe()
        if duration == 0:
            raise RuntimeError("benchmark clock produced a zero-duration measurement")

        fallback_call_count = fallback_reranker.call_count if fallback_reranker is not None else 0
        fallback_pair_attempt_count = (
            fallback_reranker.pair_attempt_count if fallback_reranker is not None else 0
        )
        fallback_failed_call_count = (
            fallback_reranker.failed_call_count if fallback_reranker is not None else 0
        )
        first_run_call_count = reranker.call_count + fallback_call_count + adjudicator.call_count
        first_run_pair_attempt_count = reranker.pair_attempt_count + fallback_pair_attempt_count + adjudicator.call_count
        first_reranker_call_count = reranker.call_count
        first_adjudicator_call_count = adjudicator.call_count
        first_backend_failed_call_count = reranker.failed_call_count + fallback_failed_call_count + adjudicator.failed_call_count
        first_reranker_batch_sizes = tuple(reranker.batch_sizes)
        primary_service_requests = (
            tuple(self._transport_probe.requests) if self._transport_probe is not None else ()
        )
        fallback_service_requests = (
            tuple(self._fallback_transport_probe.requests)
            if self._fallback_transport_probe is not None
            else ()
        )
        service_requests = primary_service_requests + fallback_service_requests
        if first_run_call_count < 1 or first_run_pair_attempt_count < 1:
            raise RuntimeError("benchmark did not execute a model request")
        resume_call_count = 0
        resumed_count = 0
        resumed_pair_ids: tuple[str, ...] = ()
        resume_service_requests: tuple[_ServiceRequest, ...] = ()
        if verify_resume:
            calls_before_resume = reranker.call_count + (fallback_reranker.call_count if fallback_reranker is not None else 0) + adjudicator.call_count
            resumed = pipeline.run(run_id, ordered_papers)
            resume_call_count = reranker.call_count + (fallback_reranker.call_count if fallback_reranker is not None else 0) + adjudicator.call_count - calls_before_resume
            resumed_count = sum(item.resumed for item in resumed.decisions)
            resumed_pair_ids = tuple(sorted(
                item.paper_id for item in resumed.decisions if item.resumed
            ))
            if self._transport_probe is not None:
                resume_service_requests = tuple(
                    self._transport_probe.requests[len(primary_service_requests):]
                )
                if self._fallback_transport_probe is not None:
                    resume_service_requests += tuple(
                        self._fallback_transport_probe.requests[
                            len(fallback_service_requests):
                        ]
                    )

        durations = (
            tuple(item.duration_seconds for item in service_requests)
            if service_requests else tuple(reranker.call_durations + adjudicator.call_durations)[:first_run_call_count]
        )
        reranker_durations = (
            tuple(item.duration_seconds for item in service_requests if item.path == "/v1/rerank")
            if service_requests else tuple(reranker.call_durations[:first_reranker_call_count])
        )
        qwen_durations = (
            tuple(item.duration_seconds for item in service_requests if item.path == "/v1/chat/completions")
            if service_requests else tuple(adjudicator.call_durations[:first_adjudicator_call_count])
        )
        p50 = _percentile(durations, 0.50)
        p95 = _percentile(durations, 0.95)
        terminal_failures = tuple(sorted(
            item.paper_id for item in summary.decisions
            if item.status is FilterStatus.NEEDS_REVIEW and _is_terminal_request_failure(item.reason_code)
        ))
        completed = tuple(sorted(
            item.paper_id for item in summary.decisions if item.status is not FilterStatus.NEEDS_REVIEW
        ))
        needs_review = tuple(sorted(
            item.paper_id for item in summary.decisions if item.status is FilterStatus.NEEDS_REVIEW
        ))
        qwen_ids = tuple(sorted(item.paper_id for item in summary.decisions if item.adjudicated))
        committed_decision_count = int(self.database.connection.execute(
            "SELECT COUNT(*) FROM filter_decisions WHERE run_id = ?", (run_id,)
        ).fetchone()[0])
        samples = tuple(rss.samples)
        pressure_samples = tuple(rss.pressure_samples)
        peak_rss = max(samples)
        peak_memory_gb = peak_rss / (1024 ** 3)
        memory_pressure_critical = any(pressure_samples)
        unbounded_memory_growth = _unbounded_growth(samples)
        service_request_failure_rate = (
            sum(item.failed for item in service_requests) / len(service_requests)
            if service_requests
            else None
        )
        request_failure_rate = len(terminal_failures) / len(spec.cases)
        alarm_codes = benchmark_alarm_codes(
            summary.qwen_alarms,
            request_failure_rate=request_failure_rate,
            service_request_failure_rate=service_request_failure_rate,
            peak_memory_gb=peak_memory_gb,
            memory_pressure_critical=memory_pressure_critical,
            unbounded_memory_growth=unbounded_memory_growth,
        )
        input_tokens = sum(item.input_tokens for item in spec.cases)
        input_hash = benchmark_workload_hash(spec.cases, ordered_papers)
        observed_components = ["rules"]
        if first_reranker_call_count:
            observed_components.append("reranker")
        if first_adjudicator_call_count:
            observed_components.extend(("qwen", "schema_validation"))
        if committed_decision_count:
            observed_components.append("sqlite_commit")
        payload: dict[str, Any] = {
            "record_version": 2,
            "measurement_evidence_version": "1",
            "kind": spec.kind,
            "scenario": spec.scenario,
            "run_id": run_id,
            "manifest_hash": spec.manifest_hash,
            "corpus_hash": spec.corpus_hash,
            "input_hash": input_hash,
            "release_hash": self.release_hash,
            "stage2_config_hash": spec.stage2_config_hash,
            "observed_stage2_config_hash": self.profile.base_runtime_config_hash,
            "full_profile_hash": self.profile.full_profile_hash,
            "model_lock_hashes": list(spec.model_lock_hashes),
            "threshold_artifact_hashes": list(spec.threshold_artifact_hashes),
            "observed_threshold_artifact_hashes": list(self._observed_threshold_hashes()),
            "model_releases": {
                "reranker": {"model_id": self.profile.reranker_model_id, "revision": self.profile.reranker_revision},
                "qwen": {"model_id": self.profile.adjudicator_model_id, "revision": self.profile.adjudicator_revision},
            },
            "prompt_hash": self.profile.prompt_hash,
            "schema_hash": self.profile.schema_hash,
            "output_token_limit": spec.output_token_limit,
            "observed_output_token_limit": self.adjudicator_output_token_limit,
            "fixture_scale": spec.fixture_scale,
            "case_count": len(spec.cases),
            "input_token_count": input_tokens,
            "duration_seconds": duration,
            "p50_seconds": p50,
            "p95_seconds": p95,
            "latency_sample_count": len(durations),
            "latency_sample_unit": "omlx_service_request" if service_requests else "backend_method_call",
            "latency_by_path": {
                "reranker": _latency_document(reranker_durations),
                "qwen": _latency_document(qwen_durations),
            },
            "papers_per_second": len(spec.cases) / duration,
            "input_tokens_per_second": input_tokens / duration,
            "pair_tokens_per_second": input_tokens / duration,
            "peak_memory_gb": peak_memory_gb,
            "rss_start_bytes": rss_start,
            "rss_end_bytes": rss_end,
            "peak_rss_bytes": peak_rss,
            "rss_sample_count": len(samples),
            "rss_samples_bytes": list(samples),
            "memory_pressure_samples": list(pressure_samples),
            "rss_sample_interval_seconds": self.rss_sample_interval_seconds,
            "rss_scope": self.rss_scope,
            # PerformanceRunRecord audits failures by unique paper ID, so its
            # denominator is the frozen case set.  Model and HTTP attempts are
            # separate because Qwen routing and retries legitimately add work.
            "request_count": len(spec.cases),
            "request_count_unit": "manifest_case",
            "request_failure_rate": request_failure_rate,
            "pair_attempt_count": first_run_pair_attempt_count,
            "model_call_count": first_run_call_count,
            "model_call_trace": [
                {
                    "backend": "reranker",
                    "pair_ids": list(pair_ids),
                    "duration_seconds": duration_seconds,
                    "failed": failed,
                }
                for pair_ids, duration_seconds, failed in zip(
                    reranker.pair_ids[:first_reranker_call_count],
                    reranker.call_durations[:first_reranker_call_count],
                    reranker.call_failures[:first_reranker_call_count],
                    strict=True,
                )
            ] + [
                {
                    "backend": "qwen",
                    "pair_ids": [pair_id],
                    "duration_seconds": duration_seconds,
                    "failed": failed,
                }
                for pair_id, duration_seconds, failed in zip(
                    adjudicator.pair_ids[:first_adjudicator_call_count],
                    adjudicator.call_durations[:first_adjudicator_call_count],
                    adjudicator.call_failures[:first_adjudicator_call_count],
                    strict=True,
                )
            ],
            "service_request_count": len(service_requests) if service_requests else None,
            "service_request_trace": [
                {
                    "path": item.path,
                    "duration_seconds": item.duration_seconds,
                    "document_count": item.document_count,
                    "failed": item.failed,
                    "resource_exhausted": item.resource_exhausted,
                }
                for item in service_requests
            ],
            "service_pair_attempt_count": (
                sum(item.document_count for item in service_requests) if service_requests else None
            ),
            "service_failed_request_count": sum(item.failed for item in service_requests) if service_requests else None,
            "service_request_failure_rate": service_request_failure_rate,
            "reranker_batch_call_count": first_reranker_call_count,
            "reranker_fallback_count": fallback_call_count,
            "reranker_fallback_trace": (
                [
                    {
                        "model_lock_hash": self.reranker_fallback.model_lock_hash,
                        "pair_ids": list(pair_ids),
                        "duration_seconds": duration_seconds,
                        "failed": failed,
                    }
                    for pair_ids, duration_seconds, failed in zip(
                        fallback_reranker.pair_ids[:fallback_call_count],
                        fallback_reranker.call_durations[:fallback_call_count],
                        fallback_reranker.call_failures[:fallback_call_count],
                        strict=True,
                    )
                ]
                if fallback_reranker is not None and self.reranker_fallback is not None
                else []
            ),
            "reranker_fallback_measurement_available": bool(service_requests),
            "adjudicator_call_count": first_adjudicator_call_count,
            "backend_failed_call_count": first_backend_failed_call_count,
            "backend_call_failure_rate": first_backend_failed_call_count / first_run_call_count,
            "failed_request_count": len(terminal_failures),
            "completed_pair_ids": list(completed),
            "needs_review_pair_ids": list(needs_review),
            "failed_request_pair_ids": list(terminal_failures),
            "qwen_pair_ids": list(qwen_ids),
            "qwen_count": summary.qwen_count,
            "qwen_share": summary.qwen_share,
            "qwen_share_alarms": list(summary.qwen_alarms),
            "qwen_capacity_level": summary.capacity_level,
            "adjudicator_count": summary.qwen_count,
            "adjudicator_share": summary.qwen_share,
            "adjudicator_capacity": adjudicator_capacity(summary.qwen_share),
            "alarm_codes": list(alarm_codes),
            "frozen_qwen_routing_matches": set(qwen_ids) == set(spec.forced_qwen_pair_ids) if spec.kind == "performance" else None,
            "routing_mode": "performance_only_manifest" if spec.kind == "performance" else "quality_thresholds",
            "batch_concurrency": {
                "document_batch_size": self.profile.document_batch_size,
                "reranker_max_in_flight": self.profile.reranker_max_in_flight,
                "adjudicator_concurrency": self.profile.adjudicator_concurrency,
                **dict(self.environment.batch_config),
            },
            "environment": _environment_document(self.environment),
            "expected_components": list(spec.executed_components),
            "executed_components": observed_components,
            # The evaluation contract calls this sqlite_commit_count and
            # requires one durable result per case.  It is a row count, not a
            # claim that Stage2Pipeline opened one SQLite transaction per row.
            "sqlite_commit_count": committed_decision_count,
            "sqlite_commit_unit": "persisted_filter_decision",
            "result_count": len(summary.decisions),
            "missing_result_count": len(set(item.pair_id for item in spec.cases) - {item.paper_id for item in summary.decisions}),
            "duplicate_result_count": len(summary.decisions) - len({item.paper_id for item in summary.decisions}),
            "warmed": True,
            "resume_verified": verify_resume and resume_call_count == 0 and resumed_count == len(spec.cases),
            "resume_model_call_count": resume_call_count,
            "resume_service_request_trace": [
                {
                    "path": item.path,
                    "duration_seconds": item.duration_seconds,
                    "document_count": item.document_count,
                    "failed": item.failed,
                    "resource_exhausted": item.resource_exhausted,
                }
                for item in resume_service_requests
            ],
            "resumed_pair_count": resumed_count,
            "resumed_pair_ids": list(resumed_pair_ids),
            "oom": any(item.resource_exhausted for item in service_requests),
            "process_crash": False,
            "memory_pressure_critical": memory_pressure_critical,
            "memory_pressure_sampled": self.memory_pressure_sampler is not None,
            "memory_growth_detector": _memory_growth_document(samples),
            "unbounded_memory_growth": unbounded_memory_growth,
        }
        return BenchmarkExecutionRecord(payload)

    def validate(
        self,
        spec: BenchmarkRunSpec,
        papers: Sequence[Stage2Paper],
        *,
        verify_resume: bool = True,
    ) -> tuple[Stage2Paper, ...]:
        """Validate a measured run contract without issuing model requests."""

        if not spec.fixture_scale and not verify_resume:
            raise ValueError("production benchmark must verify its SQLite resume path")
        ordered_papers = tuple(sorted(papers, key=lambda item: item.paper_id))
        self._validate_inputs(spec, ordered_papers)
        if not spec.fixture_scale and (
            not isinstance(self.reranker, OmlxRerankBackend)
            or not isinstance(self.adjudicator, OmlxChatBackend)
            or self._transport_probe is None
        ):
            raise ValueError("production benchmark must execute the instrumented oMLX backends")
        if not spec.fixture_scale and self.memory_pressure_sampler is None:
            raise ValueError("production benchmark requires an observed macOS memory-pressure sampler")
        if not spec.fixture_scale and self.rss_scope == "runner_process_high_water_rss":
            raise ValueError("production benchmark RSS sampler must cover the runner and resident model service")
        if not spec.fixture_scale and self.rss_sampler is _process_rss_bytes:
            raise ValueError("production benchmark requires an explicit current-RSS sampler")
        return ordered_papers

    def _validate_inputs(self, spec: BenchmarkRunSpec, papers: Sequence[Stage2Paper]) -> None:
        expected = {item.pair_id for item in spec.cases}
        actual = {item.paper_id for item in papers}
        if len(actual) != len(papers) or actual != expected:
            raise ValueError("benchmark papers must exactly and uniquely match manifest pair IDs")
        missing = {item.pair_id for item in spec.cases if item.abstract_missing}
        observed_missing = {item.paper_id for item in papers if not item.abstract or not item.abstract.strip()}
        if missing != observed_missing:
            raise ValueError("benchmark paper abstracts do not match frozen missing-abstract flags")
        paper_by_id = {paper.paper_id: paper for paper in papers}
        for case in spec.cases:
            messages = adjudication_messages(
                query_version=self.profile.query_version,
                query=self.profile.query,
                paper=paper_by_id[case.pair_id],
            )
            observed = estimate_omlx_chat_input_token_proxy(messages)
            if case.input_tokens != observed:
                raise ValueError(
                    "benchmark case input_tokens do not match the released prompt proxy estimator"
                )
        if spec.stage2_config_hash != self.profile.base_runtime_config_hash:
            raise ValueError("benchmark manifest does not match the executed Stage 2 config")
        if spec.output_token_limit != self.adjudicator_output_token_limit:
            raise ValueError("benchmark manifest output token limit does not match the executed adjudicator")
        frozen_batch = self.environment.batch_config
        expected_batch = {
            "document_batch_size": self.profile.document_batch_size,
            "rerank_batch": self.profile.document_batch_size,
            "reranker_max_in_flight": self.profile.reranker_max_in_flight,
            "adjudicator_concurrency": self.profile.adjudicator_concurrency,
            "qwen_concurrency": self.profile.adjudicator_concurrency,
        }
        if any(key in frozen_batch and frozen_batch[key] != value for key, value in expected_batch.items()):
            raise ValueError("benchmark environment batch/concurrency does not match the executed Stage 2 profile")
        if not spec.fixture_scale:
            profile_locks = (
                self.profile.reranker_lock_hash,
                self.profile.adjudicator_lock_hash,
            )
            if any(value is None for value in profile_locks) or tuple(profile_locks) != spec.model_lock_hashes:
                raise ValueError("benchmark manifest does not match the executed model locks")
            if set(self.environment.resident_model_instances) != set(spec.model_lock_hashes):
                raise ValueError("benchmark resident model instances do not match the frozen model locks")
            if self._observed_threshold_hashes() != spec.threshold_artifact_hashes:
                raise ValueError("benchmark manifest does not match the executed threshold artifacts")

    def _ensure_papers(self, papers: Sequence[Stage2Paper]) -> None:
        with self.database.transaction() as connection:
            for paper in papers:
                existing = connection.execute(
                    "SELECT title, abstract, keywords_json FROM papers WHERE paper_id = ?", (paper.paper_id,)
                ).fetchone()
                keywords = json.dumps(list(paper.keywords), ensure_ascii=False, separators=(",", ":"))
                if existing is None:
                    connection.execute(
                        "INSERT INTO papers(paper_id, title, abstract, keywords_json) VALUES (?, ?, ?, ?)",
                        (paper.paper_id, paper.title, paper.abstract, keywords),
                    )
                elif (
                    existing["title"] != paper.title
                    or existing["abstract"] != paper.abstract
                    or json.loads(existing["keywords_json"]) != list(paper.keywords)
                ):
                    raise ValueError(f"database paper {paper.paper_id} does not match frozen benchmark input")

    def _default_warmup_paper(self, papers: Sequence[Stage2Paper]) -> Stage2Paper:
        included = self.profile.include_document_types
        excluded = self.profile.exclude_document_types
        for paper in papers:
            document_type = (paper.document_type or "").strip().casefold()
            if document_type not in included and document_type not in excluded:
                return paper
        raise ValueError("benchmark warm-up requires at least one paper not handled by deterministic rules")

    def _observed_threshold_hashes(self) -> tuple[str, ...]:
        if self.profile.reranker_calibration is not None and self.profile.adjudicator_calibration is not None:
            return (
                self.profile.reranker_calibration.threshold.hash(),
                self.profile.adjudicator_calibration.threshold.hash(),
            )
        return (self.profile.threshold_hash,)


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    if any(not isfinite(value) or value < 0 for value in values):
        raise ValueError("latency samples must be finite and non-negative")
    ordered = sorted(values)
    return ordered[max(0, ceil(quantile * len(ordered)) - 1)]


def _latency_document(values: Sequence[float]) -> dict[str, float | int]:
    return {
        "sample_count": len(values),
        "p50_seconds": _percentile(values, 0.50),
        "p95_seconds": _percentile(values, 0.95),
    }


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def benchmark_workload_hash(
    cases: Sequence[PerformanceCase],
    papers: Sequence[Stage2Paper],
) -> str:
    """Hash the exact ordered cases and normalized papers consumed by a run."""

    by_id = {paper.paper_id: paper for paper in papers}
    return content_hash([
        {
            "pair_id": case.pair_id,
            "input_tokens": case.input_tokens,
            "abstract_missing": case.abstract_missing,
            "paper": {
                "paper_id": by_id[case.pair_id].paper_id,
                "title": by_id[case.pair_id].title,
                "abstract": by_id[case.pair_id].abstract,
                "keywords": list(by_id[case.pair_id].keywords),
                "document_type": by_id[case.pair_id].document_type,
                "possibly_truncated": by_id[case.pair_id].possibly_truncated,
                "multi_condition_conflict": by_id[case.pair_id].multi_condition_conflict,
                "language_anomaly": by_id[case.pair_id].language_anomaly,
            },
        }
        for case in cases
    ])


def _is_terminal_request_failure(reason_code: str) -> bool:
    return reason_code in TERMINAL_TECHNICAL_REASONS


def _unbounded_growth(samples: Sequence[int]) -> bool:
    """Flag a sustained monotonic rise of more than 25% after warm-up."""

    if len(samples) < 4 or samples[0] == 0:
        return False
    return all(right >= left for left, right in zip(samples, samples[1:])) and samples[-1] > samples[0] * 1.25


def _memory_growth_document(samples: Sequence[int]) -> dict[str, Any]:
    start = samples[0]
    end = samples[-1]
    return {
        "version": "post_warmup_monotonic_25_percent_v1",
        "monotonic_non_decreasing": all(right >= left for left, right in zip(samples, samples[1:])),
        "growth_bytes": end - start,
        "growth_ratio": end / start if start else None,
        "minimum_sample_count": 4,
    }


def _environment_document(environment: BenchmarkEnvironment) -> dict[str, Any]:
    return {
        "machine_model": environment.machine_model,
        "memory_gb": environment.memory_gb,
        "macos_version": environment.macos_version,
        "omlx_version": environment.omlx_version,
        "mlx_version": environment.mlx_version,
        "power_mode": environment.power_mode,
        "background_load": environment.background_load,
        "batch_config": dict(environment.batch_config),
        "resident_model_instances": dict(environment.resident_model_instances),
    }


def _evaluation_fields(payload: Mapping[str, Any], *, performance: bool) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "record_version": payload["record_version"],
        "run_id": payload["run_id"],
        "manifest_hash": payload["manifest_hash"],
        "stage2_config_hash": payload["stage2_config_hash"],
        "model_lock_hashes": tuple(payload["model_lock_hashes"]),
        "duration_seconds": payload["duration_seconds"],
        "peak_memory_gb": payload["peak_memory_gb"],
        "request_count": payload["request_count"],
        "failed_request_count": payload["failed_request_count"],
        "service_request_count": payload["service_request_count"],
        "service_failed_request_count": payload["service_failed_request_count"],
        "resume_verified": payload["resume_verified"],
        "resume_model_call_count": payload["resume_model_call_count"],
        "resumed_pair_count": payload["resumed_pair_count"],
        "completed_pair_ids": tuple(payload["completed_pair_ids"]),
        "needs_review_pair_ids": tuple(payload["needs_review_pair_ids"]),
        "failed_request_pair_ids": tuple(payload["failed_request_pair_ids"]),
        "environment": BenchmarkEnvironment(**payload["environment"]),
        "executed_components": tuple(payload["executed_components"]),
        "sqlite_commit_count": payload["sqlite_commit_count"],
        "warmed": payload["warmed"],
        "oom": payload["oom"],
        "process_crash": payload["process_crash"],
        "memory_pressure_critical": payload["memory_pressure_critical"],
        "unbounded_memory_growth": payload["unbounded_memory_growth"],
    }
    if performance:
        fields.update({
            "scenario": payload["scenario"],
            "p50_seconds": payload["p50_seconds"],
            "p95_seconds": payload["p95_seconds"],
            "qwen_pair_ids": tuple(payload["qwen_pair_ids"]),
        })
    return fields
