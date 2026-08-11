from __future__ import annotations

from base64 import b64encode
from hashlib import sha256
import json
from pathlib import Path

import pytest

from paper_agent.canonical import content_hash
from paper_agent.schema import SchemaValidationError, validate
from paper_agent.stage2_evaluation import (
    GoldManifest,
    GoldPair,
    GoldSplit,
    pair_universe_hash,
    write_gold_manifest,
)
from paper_agent.stage2_release_evidence import (
    Stage2EvidenceError,
    load_stage2_release_evidence_index,
)


HASH = "a" * 64


def _write(path: Path, value: object) -> dict[str, str]:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return {"path": path.name, "sha256": sha256(path.read_bytes()).hexdigest()}


def _gold_manifest() -> GoldManifest:
    pairs = []
    for split, size in (
        (GoldSplit.DEV, 300),
        (GoldSplit.HIDDEN_HARD, 150),
        (GoldSplit.HIDDEN_REAL, 150),
    ):
        for index in range(size):
            pairs.append(GoldPair(
                paper_id=f"paper-{split.value}-{index}",
                topic=f"topic-{index % 6}",
                language="zh" if index % 2 else "en",
                source="frozen-crawler-snapshot",
                sampling_probability=0.2,
                paper_family=f"family-{split.value}-{index}",
                corpus_hash="corpus-v1",
                split=split,
                abstract_incomplete=(
                    split is not GoldSplit.HIDDEN_REAL and index < size // 10
                ),
                sampled_from_natural_distribution=split is GoldSplit.HIDDEN_REAL,
                cross_language_match=index % 20 == 0,
            ))
    return GoldManifest(1, "corpus-v1", tuple(pairs), ("en", "zh"))


def _attestation(
    *,
    candidate_id: str,
    evaluation_manifest_hash: str,
    stage2_config_hash: str,
    hashes: dict[str, str],
    gold: GoldManifest | None = None,
) -> dict:
    universe_hashes = {
        split.value: pair_universe_hash(
            [pair.pair_id for pair in gold.pairs if pair.split is split]
        )
        for split in (GoldSplit.HIDDEN_HARD, GoldSplit.HIDDEN_REAL)
    } if gold is not None else {"hidden_hard": HASH, "hidden_real": HASH}
    payload = {
        "schema_version": "1",
        "attestation_type": "stage2-hidden-promotion",
        "evaluator_key_id": "hidden-evaluator-2026-01",
        "evaluator_id": "evaluation-team-1",
        "trust_manifest_hash": HASH,
        "issued_at": "2026-08-11T00:00:00Z",
        "candidate_id": candidate_id,
        "evaluation_manifest_hash": evaluation_manifest_hash,
        "evaluation_run_id": "promotion-1",
        "stage2_config_hash": stage2_config_hash,
        "model_lock_hashes": hashes,
        "calibrator_hashes": hashes,
        "threshold_hashes": hashes,
        "hidden_pair_universe_hashes": universe_hashes,
        "hidden_split_pair_counts": {
            "hidden_hard": 150,
            "hidden_real": 150,
        },
        "prediction_submission_hash": HASH,
        "promotion_marker_hash": HASH,
        "consumed_hidden_splits": ["hidden_hard", "hidden_real"],
        "gate_policy_hash": HASH,
        "result_summary": {
            "passed": True,
            "failures": [],
            "gate_versions": {"promotion": "1", "determinism": "1"},
        },
    }
    return {
        "schema_version": "1",
        "payload": payload,
        "payload_sha256": content_hash(payload),
        "signature_b64": b64encode(bytes(64)).decode(),
    }


def _index(tmp_path: Path) -> tuple[Path, dict]:
    gold = _gold_manifest()
    gold_path = tmp_path / "gold.json"
    write_gold_manifest(gold_path, gold)
    gold_ref = {"path": gold_path.name, "sha256": sha256(gold_path.read_bytes()).hexdigest()}
    hashes = {"reranker": HASH, "qwen": "b" * 64}
    stage2_config_hash = "d" * 64
    attestation_ref = _write(
        tmp_path / "attestation.json",
        _attestation(
            candidate_id="candidate-1",
            evaluation_manifest_hash=gold.hash(),
            stage2_config_hash=stage2_config_hash,
            hashes=hashes,
            gold=gold,
        ),
    )
    refs = {
        name: _write(tmp_path / f"{name}.json", {"kind": name})
        for name in (
            "structured-manifest",
            "structured-records",
            "rationale-manifest",
            "rationale-records",
            "parity-manifest",
            "parity-scores",
            "benchmark-manifest",
            "soak-manifest",
            "soak-record",
        )
    }
    benchmark_records = [
        _write(tmp_path / f"benchmark-record-{index}.json", {"run": index})
        for index in range(6)
    ]
    document = {
        "schema_version": "1",
        "evidence_type": "stage2_release_evidence",
        "candidate_id": "candidate-1",
        "evaluation_manifest_hash": gold.hash(),
        "stage2_config_hash": stage2_config_hash,
        "model_lock_hashes": hashes,
        "calibrator_hashes": hashes,
        "threshold_hashes": hashes,
        "gold_manifest": gold_ref,
        "hidden_attestation": attestation_ref,
        "public_gates": {
            "structured_replay": {
                "manifest": refs["structured-manifest"],
                "records": refs["structured-records"],
            },
            "rationale": {
                "manifest": refs["rationale-manifest"],
                "records": refs["rationale-records"],
            },
            "parity": {
                "manifest": refs["parity-manifest"],
                "scores": refs["parity-scores"],
            },
            "benchmark": {
                "manifest": refs["benchmark-manifest"],
                "records": benchmark_records,
            },
            "soak": {
                "manifest": refs["soak-manifest"],
                "record": refs["soak-record"],
            },
        },
    }
    path = tmp_path / "stage2-release-evidence.json"
    path.write_text(json.dumps(document, sort_keys=True) + "\n", encoding="utf-8")
    return path, document


def test_release_evidence_index_verifies_every_bound_file(tmp_path: Path) -> None:
    path, _ = _index(tmp_path)

    index = load_stage2_release_evidence_index(path)

    assert index.candidate_id == "candidate-1"
    assert tuple(index.public_gates) == (
        "structured_replay",
        "rationale",
        "parity",
        "benchmark",
        "soak",
    )
    assert len(index.public_gates["benchmark"].records) == 6


def test_release_evidence_index_rejects_claimed_gate_outcomes(tmp_path: Path) -> None:
    path, document = _index(tmp_path)
    document["passed"] = True
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(SchemaValidationError, match="Additional properties"):
        load_stage2_release_evidence_index(path)


def test_release_evidence_index_rejects_file_hash_drift(tmp_path: Path) -> None:
    path, _ = _index(tmp_path)
    (tmp_path / "parity-scores.json").write_text("[]\n", encoding="utf-8")

    with pytest.raises(Stage2EvidenceError, match="drifted: parity-scores.json"):
        load_stage2_release_evidence_index(path)


def test_release_evidence_index_binds_public_gold_manifest_identity(tmp_path: Path) -> None:
    path, document = _index(tmp_path)
    document["evaluation_manifest_hash"] = "f" * 64
    attestation = json.loads((tmp_path / "attestation.json").read_text(encoding="utf-8"))
    attestation["payload"]["evaluation_manifest_hash"] = "f" * 64
    document["hidden_attestation"] = _write(tmp_path / "attestation.json", attestation)
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(Stage2EvidenceError, match="gold manifest does not match"):
        load_stage2_release_evidence_index(path)


def test_release_evidence_index_binds_hidden_attestation_identity(tmp_path: Path) -> None:
    path, document = _index(tmp_path)
    attestation = json.loads((tmp_path / "attestation.json").read_text(encoding="utf-8"))
    attestation["payload"]["candidate_id"] = "different-candidate"
    attestation["payload_sha256"] = content_hash(attestation["payload"])
    document["hidden_attestation"] = _write(tmp_path / "attestation.json", attestation)
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(Stage2EvidenceError, match="candidate_id does not match"):
        load_stage2_release_evidence_index(path)


def test_release_evidence_index_recomputes_attestation_payload_hash(tmp_path: Path) -> None:
    path, document = _index(tmp_path)
    attestation = json.loads((tmp_path / "attestation.json").read_text(encoding="utf-8"))
    attestation["payload_sha256"] = "f" * 64
    document["hidden_attestation"] = _write(tmp_path / "attestation.json", attestation)
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(Stage2EvidenceError, match="payload hash is invalid"):
        load_stage2_release_evidence_index(path)


def test_release_evidence_index_binds_hidden_pair_universes(tmp_path: Path) -> None:
    path, document = _index(tmp_path)
    attestation = json.loads((tmp_path / "attestation.json").read_text(encoding="utf-8"))
    attestation["payload"]["hidden_pair_universe_hashes"]["hidden_real"] = "f" * 64
    attestation["payload_sha256"] = content_hash(attestation["payload"])
    document["hidden_attestation"] = _write(tmp_path / "attestation.json", attestation)
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(Stage2EvidenceError, match="pair universes do not match"):
        load_stage2_release_evidence_index(path)


def test_release_evidence_index_rejects_symlink_escape(tmp_path: Path) -> None:
    path, document = _index(tmp_path)
    outside = tmp_path.parent / "outside-stage2-evidence.json"
    outside.write_text("{}\n", encoding="utf-8")
    link = tmp_path / "linked-evidence.json"
    link.symlink_to(outside)
    document["gold_manifest"] = {
        "path": link.name,
        "sha256": sha256(outside.read_bytes()).hexdigest(),
    }
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(Stage2EvidenceError, match="stay inside the bundle"):
        load_stage2_release_evidence_index(path)


def test_hidden_attestation_schema_forbids_private_evaluator_content() -> None:
    document = _attestation(
        candidate_id="candidate-1",
        evaluation_manifest_hash=HASH,
        stage2_config_hash=HASH,
        hashes={"reranker": HASH, "qwen": HASH},
    )
    validate(document, "stage2-hidden-evaluator-attestation.schema.json")

    document["payload"]["labels"] = {"hidden-pair": 3}
    with pytest.raises(SchemaValidationError, match="Additional properties"):
        validate(document, "stage2-hidden-evaluator-attestation.schema.json")
