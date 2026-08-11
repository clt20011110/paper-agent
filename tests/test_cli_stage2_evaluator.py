from __future__ import annotations

from base64 import b64encode
import json
from pathlib import Path
import traceback

import pytest

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from paper_agent import cli
from paper_agent.canonical import canonical_json, content_hash
from paper_agent.stage2_hidden_attestation import HIDDEN_PROMOTION_GATE_POLICY_HASH


def _payload(*, trust_manifest_hash: str = "a" * 64) -> dict[str, object]:
    return {
        "schema_version": "1",
        "attestation_type": "stage2-hidden-promotion",
        "evaluator_key_id": "evaluator-1",
        "evaluator_id": "evaluation-team-1",
        "trust_manifest_hash": trust_manifest_hash,
        "issued_at": "2026-08-11T00:00:00Z",
        "candidate_id": "candidate-1",
        "evaluation_manifest_hash": "b" * 64,
        "evaluation_run_id": "evaluation-1",
        "stage2_config_hash": "c" * 64,
        "model_lock_hashes": {"reranker": "d" * 64, "qwen": "e" * 64},
        "calibrator_hashes": {"reranker": "f" * 64, "qwen": "0" * 64},
        "threshold_hashes": {"reranker": "1" * 64, "qwen": "2" * 64},
        "hidden_pair_universe_hashes": {
            "hidden_hard": "3" * 64,
            "hidden_real": "4" * 64,
        },
        "hidden_split_pair_counts": {"hidden_hard": 150, "hidden_real": 150},
        "prediction_submission_hash": "5" * 64,
        "promotion_marker_hash": "6" * 64,
        "consumed_hidden_splits": ["hidden_hard", "hidden_real"],
        "gate_policy_hash": HIDDEN_PROMOTION_GATE_POLICY_HASH,
        "result_summary": {
            "passed": True,
            "failures": [],
            "gate_versions": {"promotion": "1", "determinism": "1"},
        },
    }


def _write_private_key(path: Path) -> Ed25519PrivateKey:
    key = Ed25519PrivateKey.generate()
    path.write_bytes(key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ))
    path.chmod(0o600)
    return key


def _write_trust_manifest(
    path: Path,
    key: Ed25519PrivateKey,
    *,
    evaluator_key_id: str = "evaluator-1",
    status: str = "active",
    additional_keys: list[dict[str, object]] | None = None,
) -> str:
    public_key = key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    document = {
        "schema_version": "1",
        "trust_manifest_type": "stage2-hidden-evaluator",
        "trust_manifest_id": "test-trust",
        "keys": [{
            "key_id": evaluator_key_id,
            "algorithm": "Ed25519",
            "purpose": "stage2-hidden-promotion",
            "public_key_b64": b64encode(public_key).decode("ascii"),
            "status": status,
        }, *(additional_keys or [])],
    }
    path.write_bytes(canonical_json(document))
    return content_hash(document)


def _stdout(capsys: pytest.CaptureFixture[str]) -> dict[str, object]:
    return json.loads(capsys.readouterr().out)


def test_stage2_evaluator_attest_signs_public_safe_payload(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    key_path = tmp_path / "evaluator-private.pem"
    key = _write_private_key(key_path)
    trust_path = tmp_path / "deployment-trust.json"
    payload = _payload(trust_manifest_hash=_write_trust_manifest(trust_path, key))
    payload_path = tmp_path / "payload.json"
    payload_path.write_bytes(canonical_json(payload))
    output = tmp_path / "attestations" / "promotion.json"

    assert cli.main([
        "stage2-evaluator", "attest", "--payload", str(payload_path),
        "--signing-key-file", str(key_path), "--trust-manifest", str(trust_path),
        "--output", str(output),
    ]) == 0

    result = _stdout(capsys)
    attestation = json.loads(output.read_text(encoding="utf-8"))
    assert output.read_bytes() == canonical_json(attestation)
    assert result == {
        "attestation_sha256": content_hash(attestation),
        "candidate_id": "candidate-1",
        "command": "stage2-evaluator.attest",
        "evaluation_run_id": "evaluation-1",
        "evaluator_key_id": "evaluator-1",
        "event_code": "stage2-evaluator.attest.completed",
        "output": str(output),
        "passed": True,
        "payload_sha256": content_hash(payload),
        "signed": True,
        "stage": "stage2",
        "status": "complete",
    }
    serialized = json.dumps(result)
    for forbidden in ("evaluator_id", "trust_manifest_hash", "signing-key-file", str(key_path)):
        assert forbidden not in serialized


def test_stage2_evaluator_attest_dry_run_does_not_touch_private_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    trust_key = Ed25519PrivateKey.generate()
    trust_path = tmp_path / "deployment-trust.json"
    payload = _payload(trust_manifest_hash=_write_trust_manifest(trust_path, trust_key))
    payload_path = tmp_path / "payload.json"
    payload_path.write_bytes(canonical_json(payload))
    output = tmp_path / "nested" / "promotion.json"
    key_path = tmp_path / "missing-private.pem"

    def must_not_load(_: Path) -> object:
        raise AssertionError("dry-run must not inspect the private key")

    monkeypatch.setattr(cli, "load_hidden_evaluator_private_key", must_not_load)
    assert cli.main([
        "stage2-evaluator", "attest", "--payload", str(payload_path),
        "--signing-key-file", str(key_path), "--trust-manifest", str(trust_path),
        "--output", str(output), "--dry-run",
    ]) == 0

    result = _stdout(capsys)
    assert result["status"] == "validated"
    assert result["signed"] is False
    assert not output.exists()
    assert not output.parent.exists()


def test_stage2_evaluator_attest_refuses_existing_output_before_loading_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trust_path = tmp_path / "deployment-trust.json"
    trust_key = Ed25519PrivateKey.generate()
    payload_path = tmp_path / "payload.json"
    payload_path.write_bytes(canonical_json(_payload(
        trust_manifest_hash=_write_trust_manifest(trust_path, trust_key)
    )))
    output = tmp_path / "promotion.json"
    output.write_text("existing", encoding="utf-8")

    def must_not_load(_: Path) -> object:
        raise AssertionError("existing output must be rejected before key loading")

    monkeypatch.setattr(cli, "load_hidden_evaluator_private_key", must_not_load)
    with pytest.raises(FileExistsError, match="already exists"):
        cli.main([
            "stage2-evaluator", "attest", "--payload", str(payload_path),
            "--signing-key-file", str(tmp_path / "private.pem"),
            "--trust-manifest", str(trust_path), "--output", str(output),
        ])


def test_stage2_evaluator_attest_rejects_non_public_payload_before_loading_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _payload()
    payload["raw_hidden_labels"] = ["must not be accepted"]
    payload_path = tmp_path / "payload.json"
    payload_path.write_bytes(canonical_json(payload))

    def must_not_load(_: Path) -> object:
        raise AssertionError("invalid payload must not reach private key loading")

    monkeypatch.setattr(cli, "load_hidden_evaluator_private_key", must_not_load)
    with pytest.raises(ValueError, match="failed schema validation"):
        cli.main([
            "stage2-evaluator", "attest", "--payload", str(payload_path),
            "--signing-key-file", str(tmp_path / "private.pem"),
            "--trust-manifest", str(tmp_path / "unused-trust.json"),
            "--output", str(tmp_path / "promotion.json"),
        ])


def test_stage2_evaluator_attest_schema_failure_does_not_echo_private_instance(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    secret = "PRIVATE-HIDDEN-LABEL-DO-NOT-ECHO"
    payload = _payload()
    payload["raw_hidden_labels"] = [secret]
    payload_path = tmp_path / "payload.json"
    payload_path.write_bytes(canonical_json(payload))

    assert cli.entrypoint([
        "stage2-evaluator", "attest", "--payload", str(payload_path),
        "--signing-key-file", str(tmp_path / "private.pem"),
        "--trust-manifest", str(tmp_path / "trust.json"),
        "--output", str(tmp_path / "promotion.json"),
    ]) == 1

    result = _stdout(capsys)
    assert result["error"] == "hidden evaluator signing payload failed schema validation"
    assert secret not in json.dumps(result)
    assert "raw_hidden_labels" not in json.dumps(result)

    payload["prediction_submission_hash"] = [
        {"pair_id": secret, "score": 0.99}
    ]
    try:
        cli.validate_hidden_promotion_payload(payload)
    except ValueError as error:
        rendered = "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        )
        assert error.__cause__ is None
        assert error.__context__ is None
        assert secret not in rendered
        assert "pair_id" not in rendered
        assert "score" not in rendered
    else:  # pragma: no cover - the schema must reject the private instance
        raise AssertionError("private payload unexpectedly passed schema validation")


def test_stage2_evaluator_attest_rejects_trust_hash_mismatch_before_key_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trust_path = tmp_path / "deployment-trust.json"
    _write_trust_manifest(trust_path, Ed25519PrivateKey.generate())
    payload_path = tmp_path / "payload.json"
    payload_path.write_bytes(canonical_json(_payload()))

    def must_not_load(_: Path) -> object:
        raise AssertionError("trust mismatch must fail before private key loading")

    monkeypatch.setattr(cli, "load_hidden_evaluator_private_key", must_not_load)
    with pytest.raises(ValueError, match="trust manifest hash"):
        cli.main([
            "stage2-evaluator", "attest", "--payload", str(payload_path),
            "--signing-key-file", str(tmp_path / "private.pem"),
            "--trust-manifest", str(trust_path), "--output", str(tmp_path / "output" / "promotion.json"),
        ])


def test_stage2_evaluator_attest_rejects_non_active_evaluator_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    retired = Ed25519PrivateKey.generate()
    active = Ed25519PrivateKey.generate()
    active_public = active.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    trust_path = tmp_path / "deployment-trust.json"
    trust_hash = _write_trust_manifest(
        trust_path,
        retired,
        status="retired",
        additional_keys=[{
            "key_id": "other-active-key",
            "algorithm": "Ed25519",
            "purpose": "stage2-hidden-promotion",
            "public_key_b64": b64encode(active_public).decode("ascii"),
            "status": "active",
        }],
    )
    payload_path = tmp_path / "payload.json"
    payload_path.write_bytes(canonical_json(_payload(trust_manifest_hash=trust_hash)))

    def must_not_load(_: Path) -> object:
        raise AssertionError("inactive evaluator key must fail before private key loading")

    monkeypatch.setattr(cli, "load_hidden_evaluator_private_key", must_not_load)
    with pytest.raises(ValueError, match="not an active trusted key"):
        cli.main([
            "stage2-evaluator", "attest", "--payload", str(payload_path),
            "--signing-key-file", str(tmp_path / "private.pem"),
            "--trust-manifest", str(trust_path), "--output", str(tmp_path / "output" / "promotion.json"),
        ])


def test_stage2_evaluator_attest_rejects_private_key_not_in_deployment_trust(
    tmp_path: Path
) -> None:
    signing_key_path = tmp_path / "evaluator-private.pem"
    _write_private_key(signing_key_path)
    trust_path = tmp_path / "deployment-trust.json"
    trust_hash = _write_trust_manifest(trust_path, Ed25519PrivateKey.generate())
    payload_path = tmp_path / "payload.json"
    payload_path.write_bytes(canonical_json(_payload(trust_manifest_hash=trust_hash)))
    output = tmp_path / "output" / "promotion.json"

    with pytest.raises(ValueError, match="does not match the active deployment trust key"):
        cli.main([
            "stage2-evaluator", "attest", "--payload", str(payload_path),
            "--signing-key-file", str(signing_key_path),
            "--trust-manifest", str(trust_path), "--output", str(output),
        ])
    assert not output.exists()
    assert not output.parent.exists()


@pytest.mark.parametrize("dry_run", (False, True))
def test_stage2_evaluator_attest_rejects_dangling_output_symlink_before_key_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dry_run: bool
) -> None:
    trust_path = tmp_path / "deployment-trust.json"
    trust_key = Ed25519PrivateKey.generate()
    payload_path = tmp_path / "payload.json"
    payload_path.write_bytes(canonical_json(_payload(
        trust_manifest_hash=_write_trust_manifest(trust_path, trust_key)
    )))
    output = tmp_path / "promotion.json"
    output.symlink_to(tmp_path / "missing-target")

    def must_not_load(_: Path) -> object:
        raise AssertionError("existing symlink must fail before private key loading")

    monkeypatch.setattr(cli, "load_hidden_evaluator_private_key", must_not_load)
    arguments = [
        "stage2-evaluator", "attest", "--payload", str(payload_path),
        "--signing-key-file", str(tmp_path / "private.pem"),
        "--trust-manifest", str(trust_path), "--output", str(output),
    ]
    if dry_run:
        arguments.append("--dry-run")
    with pytest.raises(FileExistsError, match="already exists"):
        cli.main(arguments)
