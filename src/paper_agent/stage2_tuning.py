"""Deterministic selection of a frozen Stage 2 benchmark configuration.

This is deliberately a small, pure core.  Runners produce the audited
``PerformanceRunRecord`` and ``SoakRunRecord`` instances; callers attach the
hash of each published record and hand this module the complete 3 x 3 grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
from statistics import median
from typing import Any, Literal, Mapping, Sequence

from .canonical import content_hash
from .schema import validate
from .stage2_benchmark import BenchmarkExecutionRecord
from .stage2_commands import _performance_manifest, _soak_manifest
from .stage2_evaluation import BenchmarkEnvironment, PerformanceRunRecord, SoakRunRecord, performance_gate, soak_gate
from .stage2_search import load_stage2_benchmark_candidate


DOCUMENT_BATCH_SIZES = (16, 32, 64)
ADJUDICATOR_CONCURRENCIES = (4, 8, 16)
Scenario = Literal["normal", "stress"]


class Stage2TuningError(ValueError):
    """The supplied benchmark evidence cannot yield a frozen configuration."""


@dataclass(frozen=True, slots=True, order=True)
class TuningConfiguration:
    """The only two production knobs selected by this benchmark."""

    document_batch_size: int
    adjudicator_concurrency: int

    def __post_init__(self) -> None:
        if self.document_batch_size not in DOCUMENT_BATCH_SIZES:
            raise ValueError("document_batch_size must be one of 16, 32, 64")
        if self.adjudicator_concurrency not in ADJUDICATOR_CONCURRENCIES:
            raise ValueError("adjudicator_concurrency must be one of 4, 8, 16")


@dataclass(frozen=True, slots=True)
class SoakSelectionInput:
    """A recorded decision to defer a configuration's 10,000-paper soak."""

    record_hash: str
    reason: str

    def __post_init__(self) -> None:
        _require_hash(self.record_hash, "selection input")
        if not self.reason.strip():
            raise ValueError("selection input requires a reason")


@dataclass(frozen=True, slots=True)
class PerformanceMeasurement:
    record_hash: str
    record: PerformanceRunRecord
    quality_gate_passed: bool

    def __post_init__(self) -> None:
        _require_hash(self.record_hash, "performance record")
        if type(self.quality_gate_passed) is not bool:
            raise ValueError("quality_gate_passed must be boolean")


@dataclass(frozen=True, slots=True)
class SoakMeasurement:
    record_hash: str
    record: SoakRunRecord
    quality_gate_passed: bool

    def __post_init__(self) -> None:
        _require_hash(self.record_hash, "soak record")
        if type(self.quality_gate_passed) is not bool:
            raise ValueError("quality_gate_passed must be boolean")


@dataclass(frozen=True, slots=True)
class TuningCandidate:
    configuration: TuningConfiguration
    normal: tuple[PerformanceMeasurement, ...]
    stress: tuple[PerformanceMeasurement, ...]
    soak: SoakMeasurement | None = None
    selection_input: SoakSelectionInput | None = None

    def __post_init__(self) -> None:
        if len(self.normal) != 3 or len(self.stress) != 3:
            raise ValueError("each configuration requires exactly three normal and three stress measurements")
        if (self.soak is None) == (self.selection_input is None):
            raise ValueError("each configuration requires exactly one soak or selection input")
        records = (*self.normal, *self.stress)
        if len({item.record.run_id for item in records}) != 6:
            raise ValueError("performance measurements require six distinct run IDs")
        if any(item.record.scenario != "normal" for item in self.normal):
            raise ValueError("normal measurements must have scenario normal")
        if any(item.record.scenario != "stress" for item in self.stress):
            raise ValueError("stress measurements must have scenario stress")
        environments = [item.record.environment for item in records]
        if self.soak is not None:
            environments.append(self.soak.record.environment)
        if any(environment != environments[0] for environment in environments[1:]):
            raise ValueError("all measurements for a configuration must use one frozen environment")
        if any(not _matches_configuration(item.record.environment, self.configuration) for item in records):
            raise ValueError("performance environment does not bind the tuning configuration")
        if self.soak is not None and not _matches_configuration(self.soak.record.environment, self.configuration):
            raise ValueError("soak environment does not bind the tuning configuration")
        if any(item.record.duration_seconds <= 0 for item in records):
            raise ValueError("performance measurements require positive duration")

    @property
    def environment(self) -> BenchmarkEnvironment:
        return self.normal[0].record.environment

    @property
    def throughput(self) -> float:
        """Median manifest-case throughput across the six required runs."""

        return median(item.record.request_count / item.record.duration_seconds for item in (*self.normal, *self.stress))

    @property
    def p95_seconds(self) -> float:
        return median(item.record.p95_seconds for item in (*self.normal, *self.stress))

    @property
    def input_record_hashes(self) -> tuple[str, ...]:
        hashes = [item.record_hash for item in (*self.normal, *self.stress)]
        hashes.append(self.soak.record_hash if self.soak is not None else self.selection_input.record_hash)
        return tuple(sorted(hashes))

    def failures(self) -> tuple[str, ...]:
        failures: list[str] = []
        for item in (*self.normal, *self.stress):
            failures.extend(_performance_failures(item))
        if self.soak is not None:
            failures.extend(_soak_failures(self.soak))
        return tuple(sorted(set(failures)))


@dataclass(frozen=True, slots=True)
class FrozenTuningWinner:
    """Selection result safe to copy into a production runtime configuration."""

    configuration: TuningConfiguration
    throughput: float
    environment: BenchmarkEnvironment
    input_record_hashes: tuple[str, ...]
    selection_hash: str

    @property
    def document_batch_size(self) -> int:
        return self.configuration.document_batch_size

    @property
    def adjudicator_concurrency(self) -> int:
        return self.configuration.adjudicator_concurrency

    def document(self) -> dict[str, object]:
        return {
            "document_batch_size": self.document_batch_size,
            "adjudicator_concurrency": self.adjudicator_concurrency,
            "throughput": self.throughput,
            "environment": _environment_document(self.environment),
            "input_record_hashes": list(self.input_record_hashes),
            "selection_hash": self.selection_hash,
            "qwen_runtime_auto_increase": False,
        }


def select_stage2_tuning_winner(candidates: Sequence[TuningCandidate]) -> FrozenTuningWinner:
    """Validate the complete grid and freeze the deterministic throughput winner.

    Resource failures eliminate that measurement and therefore naturally move
    selection from 64 to 32 to 16.  Qwen concurrency is always copied verbatim
    from the measured winner: this function never raises it at runtime.
    """

    _require_complete_grid(candidates)
    by_configuration = {item.configuration: item for item in candidates}
    failures = {configuration: list(candidate.failures()) for configuration, candidate in by_configuration.items()}
    for configuration, candidate in by_configuration.items():
        lower = _next_smaller_batch(configuration.document_batch_size)
        if lower is None or failures[configuration]:
            continue
        baseline = by_configuration[TuningConfiguration(lower, configuration.adjudicator_concurrency)]
        if failures[baseline.configuration]:
            continue
        if candidate.p95_seconds > baseline.p95_seconds * 1.25 and candidate.throughput < baseline.throughput * 1.10:
            failures[configuration].append("p95 resource regression without 10% throughput gain")
    valid = [candidate for candidate in candidates if not failures[candidate.configuration]]
    if not valid:
        raise Stage2TuningError("no valid configuration after resource and quality gates")
    winner = min(valid, key=lambda item: (-item.throughput, item.configuration))
    hashes = tuple(sorted(
        record_hash
        for candidate in candidates
        for record_hash in candidate.input_record_hashes
    ))
    selection_hash = content_hash({
        "configuration": [
            winner.configuration.document_batch_size,
            winner.configuration.adjudicator_concurrency,
        ],
        "throughput": winner.throughput,
        "environment": _environment_document(winner.environment),
        "input_record_hashes": list(hashes),
    })
    return FrozenTuningWinner(winner.configuration, winner.throughput, winner.environment, hashes, selection_hash)


def build_stage2_tuning_winner_document(input_path: Path) -> dict[str, object]:
    """Validate one frozen production grid and return its immutable winner."""

    root = input_path.parent.resolve()
    document = _json_object(input_path)
    validate(document, "stage2-tuning-selection-input.schema.json")
    candidates: list[TuningCandidate] = []
    bindings: dict[TuningConfiguration, tuple[dict[str, str], dict[str, str], str]] = {}
    for item in document["candidates"]:
        assert isinstance(item, Mapping)
        configuration_data = _mapping(item, "configuration")
        configuration = TuningConfiguration(
            int(configuration_data["document_batch_size"]),
            int(configuration_data["adjudicator_concurrency"]),
        )
        candidate_ref, candidate_path = _artifact_ref(_mapping(item, "candidate"), root)
        release = load_stage2_benchmark_candidate(candidate_path)
        if (
            release.profile.document_batch_size != configuration.document_batch_size
            or release.profile.adjudicator_concurrency != configuration.adjudicator_concurrency
        ):
            raise Stage2TuningError("candidate runtime knobs do not match its tuning configuration")
        manifest_ref, manifest_path = _artifact_ref(_mapping(item, "benchmark_manifest"), root)
        manifest = _performance_manifest(_json_object(manifest_path))
        if (
            manifest.stage2_config_hash != release.profile.base_runtime_config_hash
            or manifest.model_lock_hashes
            != (release.profile.reranker_lock_hash, release.profile.adjudicator_lock_hash)
        ):
            raise Stage2TuningError("benchmark manifest does not bind its candidate runtime")
        normal = _measurements(_list(item, "normal"), root, manifest)
        stress = _measurements(_list(item, "stress"), root, manifest)
        records = [measurement.record for measurement in (*normal, *stress)]
        gate = performance_gate(manifest, records)
        if not gate.passed:
            raise Stage2TuningError("performance records do not pass the frozen benchmark gate")
        candidate_values: dict[str, object] = {
            "configuration": configuration,
            "normal": tuple(PerformanceMeasurement(measurement.record_hash, measurement.record, True) for measurement in normal),
            "stress": tuple(PerformanceMeasurement(measurement.record_hash, measurement.record, True) for measurement in stress),
        }
        if "selection_input" in item:
            selection = _mapping(item, "selection_input")
            selection_ref, selection_path = _artifact_ref(_mapping(selection, "record"), root)
            selection_document = _json_object(selection_path)
            if selection_document.get("candidate_independent") is not True:
                raise Stage2TuningError("selection input must be candidate-independent")
            candidate_values["selection_input"] = SoakSelectionInput(selection_ref["sha256"], str(selection["reason"]))
        else:
            _, soak_manifest_path = _artifact_ref(_mapping(item, "soak_manifest"), root)
            soak_manifest = _soak_manifest(_json_object(soak_manifest_path))
            soak_ref, soak_path = _artifact_ref(_mapping(_mapping(item, "soak"), "record"), root)
            soak = BenchmarkExecutionRecord(_json_object(soak_path)).as_soak_record()
            if soak.manifest_hash != soak_manifest.hash() or not soak_gate(soak_manifest, soak).passed:
                raise Stage2TuningError("soak record does not pass its frozen manifest gate")
            candidate_values["soak"] = SoakMeasurement(soak_ref["sha256"], soak, True)
        candidates.append(TuningCandidate(**candidate_values))  # type: ignore[arg-type]
        bindings[configuration] = (candidate_ref, manifest_ref, release.profile_name)
    winner = select_stage2_tuning_winner(candidates)
    candidate_ref, manifest_ref, candidate_id = bindings[winner.configuration]
    output = {
        "schema_version": 1,
        "kind": "stage2_tuning_winner",
        "candidate_id": candidate_id,
        "candidate": candidate_ref,
        "benchmark_manifest": manifest_ref,
        **winner.document(),
    }
    validate(output, "stage2-tuning-winner.schema.json")
    return output


def write_stage2_tuning_winner(input_path: Path, output_path: Path) -> dict[str, object]:
    """Build one schema-validated winner without replacing an existing artifact."""

    if os.path.lexists(output_path):
        raise FileExistsError(f"Stage 2 tuning winner output already exists: {output_path}")
    document = build_stage2_tuning_winner_document(input_path)
    payload = json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return document


def _measurements(
    values: Sequence[object], root: Path, manifest: object,
) -> tuple[PerformanceMeasurement, ...]:
    measurements: list[PerformanceMeasurement] = []
    for value in values:
        if not isinstance(value, Mapping):
            raise Stage2TuningError("tuning measurement must be an object")
        reference, path = _artifact_ref(_mapping(value, "record"), root)
        record = BenchmarkExecutionRecord(_json_object(path)).as_performance_record()
        if record.manifest_hash != manifest.hash():
            raise Stage2TuningError("performance record does not bind its benchmark manifest")
        measurements.append(PerformanceMeasurement(reference["sha256"], record, True))
    return tuple(measurements)


def _artifact_ref(value: Mapping[str, object], root: Path) -> tuple[dict[str, str], Path]:
    path_value = value.get("path")
    expected = value.get("sha256")
    if not isinstance(path_value, str) or not isinstance(expected, str):
        raise Stage2TuningError("tuning artifact reference is invalid")
    path = (root / path_value).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise Stage2TuningError("tuning artifact must remain inside the frozen input bundle") from error
    if not path.is_file():
        raise Stage2TuningError(f"tuning artifact is missing: {path_value}")
    observed = sha256(path.read_bytes()).hexdigest()
    if observed != expected:
        raise Stage2TuningError(f"tuning artifact hash mismatch: {path_value}")
    return {"path": path.relative_to(root).as_posix(), "sha256": observed}, path


def _json_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Stage2TuningError(f"tuning JSON artifact must be an object: {path.name}")
    return value


def _mapping(value: Mapping[str, object], field: str) -> Mapping[str, object]:
    item = value.get(field)
    if not isinstance(item, Mapping):
        raise Stage2TuningError(f"tuning input requires object {field}")
    return item


def _list(value: Mapping[str, object], field: str) -> Sequence[object]:
    item = value.get(field)
    if not isinstance(item, list):
        raise Stage2TuningError(f"tuning input requires list {field}")
    return item


def _require_complete_grid(candidates: Sequence[TuningCandidate]) -> None:
    expected = {TuningConfiguration(batch, concurrency) for batch in DOCUMENT_BATCH_SIZES for concurrency in ADJUDICATOR_CONCURRENCIES}
    observed = [candidate.configuration for candidate in candidates]
    if len(observed) != len(set(observed)):
        raise Stage2TuningError("tuning grid contains duplicate configurations")
    if set(observed) != expected:
        raise Stage2TuningError("tuning grid must contain every 16/32/64 by 4/8/16 configuration")


def _next_smaller_batch(batch: int) -> int | None:
    index = DOCUMENT_BATCH_SIZES.index(batch)
    return None if index == 0 else DOCUMENT_BATCH_SIZES[index - 1]


def _performance_failures(item: PerformanceMeasurement) -> tuple[str, ...]:
    record = item.record
    failures = [] if item.quality_gate_passed else ["quality gate failed"]
    if record.oom or record.process_crash or record.unbounded_memory_growth:
        failures.append("stability failure")
    if record.peak_memory_gb > 28 or record.memory_pressure_critical:
        failures.append("memory limit exceeded")
    if record.request_failure_rate >= 0.005 or record.service_request_failure_rate >= 0.005:
        failures.append("request failure rate >= 0.5%")
    return tuple(failures)


def _soak_failures(item: SoakMeasurement) -> tuple[str, ...]:
    record = item.record
    failures = [] if item.quality_gate_passed else ["quality gate failed"]
    if record.oom or record.process_crash or record.unbounded_memory_growth:
        failures.append("stability failure")
    if record.peak_memory_gb > 28 or record.memory_pressure_critical:
        failures.append("memory limit exceeded")
    if record.request_failure_rate >= 0.005 or record.service_request_failure_rate >= 0.005:
        failures.append("request failure rate >= 0.5%")
    return tuple(failures)


def _matches_configuration(environment: BenchmarkEnvironment, configuration: TuningConfiguration) -> bool:
    return environment.batch_config.get("document_batch_size") == configuration.document_batch_size and environment.batch_config.get("adjudicator_concurrency") == configuration.adjudicator_concurrency


def _environment_document(environment: BenchmarkEnvironment) -> dict[str, object]:
    return {
        "machine_model": environment.machine_model,
        "memory_gb": environment.memory_gb,
        "macos_version": environment.macos_version,
        "omlx_version": environment.omlx_version,
        "mlx_version": environment.mlx_version,
        "power_mode": environment.power_mode,
        "background_load": environment.background_load,
        "batch_config": dict(sorted(environment.batch_config.items())),
        "resident_model_instances": dict(sorted(environment.resident_model_instances.items())),
    }


def _require_hash(value: str, label: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} hash must be a lowercase SHA-256")
