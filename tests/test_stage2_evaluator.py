from __future__ import annotations

from base64 import b64encode
import os
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.asymmetric.rsa import generate_private_key
import pytest

from paper_agent.canonical import content_hash
from paper_agent.stage2_evaluator import (
    Stage2EvaluatorError,
    issue_hidden_promotion_from_payload,
    load_hidden_evaluator_private_key,
    verify_public_stage2_release,
)
from paper_agent.stage2_hidden_attestation import HIDDEN_PROMOTION_GATE_POLICY_HASH


def _write_private_key(path: Path, key: Ed25519PrivateKey) -> None:
    path.write_bytes(key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ))
    path.chmod(0o600)


def _payload() -> dict[str, object]:
    key = Ed25519PrivateKey.generate()
    public_key = key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    trust = {
        "schema_version": "1",
        "trust_manifest_type": "stage2-hidden-evaluator",
        "trust_manifest_id": "test-trust",
        "keys": [{
            "key_id": "evaluator-1",
            "algorithm": "Ed25519",
            "purpose": "stage2-hidden-promotion",
            "public_key_b64": b64encode(public_key).decode("ascii"),
            "status": "active",
        }],
    }
    return {
        "schema_version": "1",
        "attestation_type": "stage2-hidden-promotion",
        "evaluator_key_id": "evaluator-1",
        "evaluator_id": "test-team",
        "trust_manifest_hash": content_hash(trust),
        "issued_at": "2026-08-11T00:00:00Z",
        "candidate_id": "candidate-1",
        "evaluation_manifest_hash": "a" * 64,
        "evaluation_run_id": "run-1",
        "stage2_config_hash": "b" * 64,
        "model_lock_hashes": {"reranker": "c" * 64, "qwen": "d" * 64},
        "calibrator_hashes": {"reranker": "e" * 64, "qwen": "f" * 64},
        "threshold_hashes": {"reranker": "0" * 64, "qwen": "1" * 64},
        "hidden_pair_universe_hashes": {"hidden_hard": "2" * 64, "hidden_real": "3" * 64},
        "hidden_split_pair_counts": {"hidden_hard": 150, "hidden_real": 150},
        "prediction_submission_hash": "4" * 64,
        "promotion_marker_hash": "5" * 64,
        "consumed_hidden_splits": ["hidden_hard", "hidden_real"],
        "gate_policy_hash": HIDDEN_PROMOTION_GATE_POLICY_HASH,
        "result_summary": {
            "passed": True,
            "failures": [],
            "gate_versions": {"promotion": "1", "determinism": "1"},
        },
    }


def test_loads_owner_only_pkcs8_ed25519_private_key(tmp_path: Path) -> None:
    source = Ed25519PrivateKey.generate()
    key_path = tmp_path / "evaluator-private.pem"
    _write_private_key(key_path, source)

    loaded = load_hidden_evaluator_private_key(key_path)
    attestation = issue_hidden_promotion_from_payload(_payload(), loaded)

    assert loaded.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    ) == source.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    assert attestation["payload_sha256"] == content_hash(attestation["payload"])


@pytest.mark.parametrize("mode", (0o640, 0o644, 0o400))
def test_rejects_private_key_file_without_exact_owner_only_mode(
    tmp_path: Path, mode: int
) -> None:
    key_path = tmp_path / "evaluator-private.pem"
    _write_private_key(key_path, Ed25519PrivateKey.generate())
    key_path.chmod(mode)

    with pytest.raises(Stage2EvaluatorError, match="mode 0600"):
        load_hidden_evaluator_private_key(key_path)


def test_rejects_non_pem_and_wrong_key_type(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw-private-key"
    raw_path.write_bytes(bytes(32))
    raw_path.chmod(0o600)
    with pytest.raises(Stage2EvaluatorError, match="PKCS#8 PEM"):
        load_hidden_evaluator_private_key(raw_path)

    wrong_path = tmp_path / "rsa-private.pem"
    wrong_path.write_bytes(generate_private_key(public_exponent=65537, key_size=2048).private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ))
    wrong_path.chmod(0o600)
    with pytest.raises(Stage2EvaluatorError, match="must use Ed25519"):
        load_hidden_evaluator_private_key(wrong_path)


def test_rejects_concatenated_private_key_blocks(tmp_path: Path) -> None:
    first = tmp_path / "first.pem"
    second = tmp_path / "second.pem"
    _write_private_key(first, Ed25519PrivateKey.generate())
    _write_private_key(second, Ed25519PrivateKey.generate())
    first.write_bytes(first.read_bytes() + second.read_bytes())

    with pytest.raises(Stage2EvaluatorError, match="one canonical PKCS#8 PEM block"):
        load_hidden_evaluator_private_key(first)


def test_rejects_same_inode_key_rewrite_while_opening(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key_path = tmp_path / "evaluator-private.pem"
    _write_private_key(key_path, Ed25519PrivateKey.generate())
    replacement = Ed25519PrivateKey.generate().private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    real_open = os.open

    def replace_then_open(path: object, flags: int) -> int:
        key_path.write_bytes(replacement)
        return real_open(path, flags)

    monkeypatch.setattr("paper_agent.stage2_evaluator.os.open", replace_then_open)

    with pytest.raises(Stage2EvaluatorError, match="changed while opening"):
        load_hidden_evaluator_private_key(key_path)


def test_rejects_symlink_private_key_file(tmp_path: Path) -> None:
    actual_path = tmp_path / "actual-private.pem"
    _write_private_key(actual_path, Ed25519PrivateKey.generate())
    link_path = tmp_path / "evaluator-private.pem"
    link_path.symlink_to(actual_path)

    with pytest.raises(Stage2EvaluatorError, match="not a regular file"):
        load_hidden_evaluator_private_key(link_path)


@pytest.mark.parametrize("field", ("raw_hidden_labels", "raw_predictions"))
def test_rejects_hidden_content_in_signing_input(field: str) -> None:
    payload = _payload()
    payload[field] = ["must never be signed"]

    with pytest.raises(Stage2EvaluatorError, match="Additional properties"):
        issue_hidden_promotion_from_payload(payload, Ed25519PrivateKey.generate())


def test_public_coordinator_requires_an_explicit_trust_path(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_loader(release_path: Path, plan: object, *, hidden_trust_path: Path) -> object:
        captured.update({
            "release_path": release_path,
            "plan": plan,
            "hidden_trust_path": hidden_trust_path,
        })
        return "verified"

    monkeypatch.setattr("paper_agent.stage2_evaluator.load_stage2_release", fake_loader)
    release_path = Path("release.json")
    trust_path = Path("trust.json")
    plan = {"schema_version": "1"}

    assert verify_public_stage2_release(
        release_path, plan, hidden_trust_path=trust_path
    ) == "verified"
    assert captured == {
        "release_path": release_path,
        "plan": plan,
        "hidden_trust_path": trust_path,
    }
