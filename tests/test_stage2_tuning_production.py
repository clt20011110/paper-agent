from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from paper_agent.stage2_evaluation import BenchmarkEnvironment, PerformanceRunRecord
import paper_agent.stage2_tuning as tuning


LOCKS = ("b" * 64, "c" * 64)


def _write(path: Path, value: object) -> dict[str, str]:
    path.write_text(json.dumps(value), encoding="utf-8")
    return {"path": path.name, "sha256": sha256(path.read_bytes()).hexdigest()}


def _record(batch: int, concurrency: int, scenario: str, run: int) -> PerformanceRunRecord:
    environment = BenchmarkEnvironment(
        "Apple Silicon M4 Max", 36, "15.6", "0.5.7", "0.27.0", "automatic", "isolated",
        {"document_batch_size": batch, "adjudicator_concurrency": concurrency}, {LOCKS[0]: 1, LOCKS[1]: 1},
    )
    return PerformanceRunRecord(
        2, scenario, f"{batch}-{concurrency}-{scenario}-{run}", "a" * 64, "a" * 64, LOCKS,
        10.0, 0.5, 1.0, 20.0, 100, 0, 100, 0, True, 0, 100,
        tuple(f"paper-{index}" for index in range(100)), (), (), (), environment,
        ("rules", "reranker", "qwen", "schema_validation", "sqlite_commit"), 100, True,
    )


def test_production_input_freezes_all_63_hashes_and_writes_no_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = []
    for batch in (16, 32, 64):
        for concurrency in (4, 8, 16):
            candidate = _write(tmp_path / f"candidate-{batch}-{concurrency}.json", {})
            manifest = _write(tmp_path / f"manifest-{batch}-{concurrency}.json", {})
            selection = _write(tmp_path / f"selection-{batch}-{concurrency}.json", {"candidate_independent": True})
            normal = [{"record": _write(tmp_path / f"n-{batch}-{concurrency}-{run}.json", {})} for run in range(3)]
            stress = [{"record": _write(tmp_path / f"s-{batch}-{concurrency}-{run}.json", {})} for run in range(3)]
            entries.append({"configuration": {"document_batch_size": batch, "adjudicator_concurrency": concurrency}, "candidate": candidate, "benchmark_manifest": manifest, "normal": normal, "stress": stress, "selection_input": {"record": selection, "reason": "candidate-independent workload freeze"}})
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps({"schema_version": 1, "candidates": entries}), encoding="utf-8")
    current: list[tuple[int, int]] = []

    def candidate_loader(path: Path) -> SimpleNamespace:
        _, batch, concurrency = path.stem.split("-")
        return SimpleNamespace(profile_name=path.stem, profile=SimpleNamespace(
            document_batch_size=int(batch), adjudicator_concurrency=int(concurrency),
            base_runtime_config_hash="a" * 64, reranker_lock_hash=LOCKS[0], adjudicator_lock_hash=LOCKS[1],
        ))

    def measurements(values: list[object], _root: Path, _manifest: object):
        name = values[0]["record"]["path"]  # type: ignore[index]
        _, batch, concurrency, _ = Path(name).stem.split("-")
        scenario = "normal" if name.startswith("n-") else "stress"
        return tuple(tuning.PerformanceMeasurement(value["record"]["sha256"], _record(int(batch), int(concurrency), scenario, index), True) for index, value in enumerate(values))  # type: ignore[index]

    monkeypatch.setattr(tuning, "load_stage2_benchmark_candidate", candidate_loader)
    monkeypatch.setattr(tuning, "_performance_manifest", lambda _value: SimpleNamespace(
        hash=lambda: "a" * 64, stage2_config_hash="a" * 64, model_lock_hashes=LOCKS,
    ))
    monkeypatch.setattr(tuning, "_measurements", measurements)
    monkeypatch.setattr(tuning, "performance_gate", lambda _manifest, _records: SimpleNamespace(passed=True))

    output = tmp_path / "winner.json"
    winner = tuning.write_stage2_tuning_winner(input_path, output)

    assert len(winner["input_record_hashes"]) == 63
    assert winner["qwen_runtime_auto_increase"] is False
    assert output.is_file()
    with pytest.raises(FileExistsError, match="already exists"):
        tuning.write_stage2_tuning_winner(input_path, output)


def test_production_input_rejects_artifact_hash_drift(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text("first", encoding="utf-8")
    reference = {"path": artifact.name, "sha256": sha256(artifact.read_bytes()).hexdigest()}
    artifact.write_text("drifted", encoding="utf-8")

    with pytest.raises(tuning.Stage2TuningError, match="hash mismatch"):
        tuning._artifact_ref(reference, tmp_path)
