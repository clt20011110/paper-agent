from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import paper_agent.stage2_release_evidence_producer as producer
from paper_agent.stage2_evaluation import GoldManifest, GoldPair, GoldSplit, write_gold_manifest
from paper_agent.stage2_release_evidence import load_stage2_release_evidence_index


def _gold() -> GoldManifest:
    pairs = []
    for split, count in ((GoldSplit.DEV, 300), (GoldSplit.HIDDEN_HARD, 150), (GoldSplit.HIDDEN_REAL, 150)):
        for number in range(count):
            pairs.append(GoldPair(
                paper_id=f"{split.value}-{number}", topic=f"topic-{number % 6}",
                language="en" if number % 2 else "zh", source="frozen",
                sampling_probability=0.2 if split is GoldSplit.HIDDEN_REAL else None,
                paper_family=f"{split.value}-family-{number}", corpus_hash="a" * 64,
                split=split, abstract_incomplete=(split is not GoldSplit.HIDDEN_REAL and number < count // 10),
                sampled_from_natural_distribution=split is GoldSplit.HIDDEN_REAL,
                cross_language_match=number == 0,
            ))
    return GoldManifest(1, "a" * 64, tuple(pairs), ("en", "zh"))


def _write(path: Path, payload: object = None) -> Path:
    path.write_text(json.dumps({"artifact": path.name} if payload is None else payload) + "\n")
    return path


def _candidate(gold_hash: str) -> SimpleNamespace:
    def binding(path_hash: str) -> SimpleNamespace:
        return SimpleNamespace(
            calibrator=SimpleNamespace(gold_manifest_hash=gold_hash, hash=lambda: path_hash),
            threshold=SimpleNamespace(hash=lambda: path_hash),
        )
    profile = SimpleNamespace(
        reranker_calibration=binding("b" * 64), adjudicator_calibration=binding("c" * 64),
        base_runtime_config_hash="d" * 64, reranker_lock_hash="e" * 64,
        adjudicator_lock_hash="f" * 64,
    )
    return SimpleNamespace(profile_name="candidate-v2", profile=profile)


def _write_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **overrides: object) -> Path:
    gold_path = tmp_path / "gold.json"
    gold = _gold()
    write_gold_manifest(gold_path, gold)
    monkeypatch.setattr(producer, "load_stage2_benchmark_candidate", lambda _path: _candidate(gold.hash()))
    names = (
        "structured_manifest", "structured_records", "structured_papers", "rationale_manifest", "rationale_worklist", "rationale_records",
        "parity_manifest", "parity_workload", "parity_receipt", "parity_scores",
        "parity_oracle_lock", "parity_candidate_lock", "parity_oracle_calibrator",
        "parity_candidate_calibrator", "parity_oracle_threshold", "parity_candidate_threshold",
        "benchmark_manifest", "benchmark_papers", "soak_manifest", "soak_papers", "soak_record",
    )
    paths = {name: _write(tmp_path / f"{name}.json") for name in names}
    records = tuple(_write(tmp_path / f"benchmark-record-{number}.json") for number in range(6))
    values: dict[str, object] = {
        "output_path": tmp_path / "stage2-release-evidence.json", "candidate_bundle_path": tmp_path / "candidate-v2.json",
        "gold_manifest_path": gold_path,
        "structured_manifest_path": paths["structured_manifest"], "structured_records_path": paths["structured_records"],
        "structured_papers_path": paths["structured_papers"],
        "rationale_manifest_path": paths["rationale_manifest"], "rationale_worklist_path": paths["rationale_worklist"], "rationale_records_path": paths["rationale_records"],
        "parity_manifest_path": paths["parity_manifest"], "parity_workload_path": paths["parity_workload"],
        "parity_selection_receipt_path": paths["parity_receipt"], "parity_scores_path": paths["parity_scores"],
        "parity_oracle_model_lock_path": paths["parity_oracle_lock"], "parity_candidate_model_lock_path": paths["parity_candidate_lock"],
        "parity_oracle_calibrator_path": paths["parity_oracle_calibrator"], "parity_candidate_calibrator_path": paths["parity_candidate_calibrator"],
        "parity_oracle_threshold_path": paths["parity_oracle_threshold"], "parity_candidate_threshold_path": paths["parity_candidate_threshold"],
        "benchmark_manifest_path": paths["benchmark_manifest"], "benchmark_papers_path": paths["benchmark_papers"],
        "benchmark_record_paths": records, "soak_manifest_path": paths["soak_manifest"],
        "soak_papers_path": paths["soak_papers"], "soak_record_path": paths["soak_record"],
    }
    values.update(overrides)
    return producer.write_stage2_release_evidence_index(**values)  # type: ignore[arg-type]


def test_producer_writes_public_index_with_byte_refs_and_candidate_bindings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _write_index(tmp_path, monkeypatch)
    document = json.loads(path.read_text())
    assert document["evidence_type"] == "stage2_public_promotion_evidence"
    assert "hidden_attestation" not in document
    assert document["candidate_id"] == "candidate-v2"
    assert document["model_lock_hashes"] == {"reranker": "e" * 64, "qwen": "f" * 64}
    assert document["public_gates"]["structured_replay"]["papers"]["path"] == "structured_papers.json"
    assert document["public_gates"]["benchmark"]["records"][0]["sha256"] == sha256((tmp_path / "benchmark-record-0.json").read_bytes()).hexdigest()
    assert load_stage2_release_evidence_index(path).candidate_id == "candidate-v2"


def test_producer_refuses_to_replace_existing_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_index(tmp_path, monkeypatch)
    with pytest.raises(FileExistsError, match="already exists"):
        _write_index(tmp_path, monkeypatch)


def test_producer_rejects_artifacts_outside_output_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    outside = _write(tmp_path.parent / "outside-evidence.json")
    with pytest.raises(producer.Stage2EvidenceProducerError, match="inside the output bundle"):
        _write_index(tmp_path, monkeypatch, structured_manifest_path=outside)
