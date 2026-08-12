from __future__ import annotations

from base64 import b64encode
from dataclasses import replace
import json
import os
from pathlib import Path
from types import SimpleNamespace
import traceback

import pytest

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from paper_agent import cli
from paper_agent.canonical import canonical_json, content_hash
from paper_agent.stage2_evaluator import load_hidden_evaluator_private_key
from paper_agent.stage2_hidden_attestation import HIDDEN_PROMOTION_GATE_POLICY_HASH
from paper_agent.stage2_evaluation import GateResult
from paper_agent.stage2_promotion_artifacts import (
    PrivatePromotionArtifactError,
    PromotionSigningInput,
)


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
        "promotion_batch_hash": "9" * 64,
        "winner_candidate_id": "candidate-1",
        "release_role": "winner",
        "public_gate_artifact_hashes": {name: "7" * 64 for name in ("structured_replay", "rationale", "parity", "benchmark", "soak")},
        "throughput_runs": [1, 1, 1],
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


def _promote_argv(
    tmp_path: Path,
    *,
    trust_path: Path,
    key_path: Path,
    output: Path,
    candidate_id: str = "candidate-1",
    parity_trust_path: Path | None = None,
) -> list[str]:
    parity_trust_path = parity_trust_path or tmp_path / "parity-oracle-trust.json"
    return [
        "stage2-evaluator", "promote",
        "--manifest", str(tmp_path / "gold-manifest.json"),
        "--private-labels", str(tmp_path / "private-labels.json"),
        "--candidate", candidate_id + "=" + str(tmp_path / "candidate.json"),
        "--submission", candidate_id + "=" + str(tmp_path / "private-submission.json"),
        "--public-evidence", candidate_id + "=" + str(tmp_path / "public-evidence.json"),
        "--incumbent-candidate-id", candidate_id,
        "--evaluator-id", "isolated-evaluator-1",
        "--evaluation-run-id", "evaluation-1",
        "--state-root", str(tmp_path / "sealed-state"),
        "--evaluator-key-id", "evaluator-1",
        "--issued-at", "2026-08-11T00:00:00Z",
        "--trust-manifest", str(trust_path),
        "--parity-oracle-trust", str(parity_trust_path),
        "--signing-key-file", str(key_path),
        "--output", str(output),
    ]


def _replace_option(argv: list[str], option: str, value: str) -> list[str]:
    result = list(argv)
    result[result.index(option) + 1] = value
    return result


def _patch_promote_public_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import paper_agent.stage2_evaluation as evaluation

    monkeypatch.setattr(
        evaluation,
        "load_gold_manifest",
        lambda _path: SimpleNamespace(
            hash=lambda: "b" * 64,
            validate_sampling_structure=lambda: None,
        ),
    )
    monkeypatch.setattr(
        cli,
        "validate_promotion_candidate_bundles",
        lambda _paths, *, expected_manifest_hash: None,
    )
    monkeypatch.setattr(
        cli,
        "validate_promotion_public_evidence",
        lambda _candidates, _evidence, _manifest_hash, *, parity_oracle_trust_path: None,
    )


def _promotion_evaluation(payload: dict[str, object]) -> object:
    signing = SimpleNamespace(attestation_payload=lambda **_kwargs: payload)
    return SimpleNamespace(
        candidates={"candidate-1": signing},
        winner_candidate_id="candidate-1",
        evaluation_manifest_hash=payload["evaluation_manifest_hash"],
        evaluation_run_id=payload["evaluation_run_id"],
        promotion_marker_hash="7" * 64,
        promotion_batch_hash="9" * 64,
    )


def _promotion_signing_input(*, passed: bool = True) -> PromotionSigningInput:
    return PromotionSigningInput(
        candidate_id="candidate-1",
        evaluator_id="isolated-evaluator-1",
        evaluation_manifest_hash="b" * 64,
        evaluation_run_id="evaluation-1",
        stage2_config_hash="c" * 64,
        model_lock_hashes={"reranker": "d" * 64, "qwen": "e" * 64},
        calibrator_hashes={"reranker": "f" * 64, "qwen": "0" * 64},
        threshold_hashes={"reranker": "1" * 64, "qwen": "2" * 64},
        hidden_pair_universe_hashes={
            "hidden_hard": "3" * 64,
            "hidden_real": "4" * 64,
        },
        hidden_split_pair_counts={"hidden_hard": 150, "hidden_real": 150},
        prediction_submission_hash="5" * 64,
        promotion_marker_hash="7" * 64,
        promotion_batch_hash="9" * 64,
        winner_candidate_id="candidate-1",
        release_role="winner",
        public_gate_artifact_hashes={name: "8" * 64 for name in ("structured_replay", "rationale", "parity", "benchmark", "soak")},
        throughput_runs=(1, 1, 1),
        consumed_hidden_splits=("hidden_hard", "hidden_real"),
        gate_policy_hash=HIDDEN_PROMOTION_GATE_POLICY_HASH,
        passed=passed,
        failures=() if passed else ("hidden gate failed",),
    )


def _promotion_evaluation_from_signing(signing: PromotionSigningInput) -> object:
    return SimpleNamespace(
        candidates={signing.candidate_id: signing},
        winner_candidate_id=signing.winner_candidate_id,
        evaluation_manifest_hash=signing.evaluation_manifest_hash,
        evaluation_run_id=signing.evaluation_run_id,
        promotion_marker_hash=signing.promotion_marker_hash,
        promotion_batch_hash=signing.promotion_batch_hash,
    )


def _verified_public_gates() -> SimpleNamespace:
    return SimpleNamespace(
        gates={
            name: SimpleNamespace(evidence_hash="a" * 64, gate=GateResult(True, ()))
            for name in ("structured_replay", "rationale", "parity", "benchmark", "soak")
        },
        throughput_runs=(100, 100, 100),
    )


def test_stage2_evaluator_promote_runs_once_and_signs_selected_public_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    key_path = tmp_path / "evaluator-private.pem"
    key = _write_private_key(key_path)
    trust_path = tmp_path / "deployment-trust.json"
    trust_hash = _write_trust_manifest(trust_path, key)
    signing = _promotion_signing_input()
    output = tmp_path / "release-bundle" / "hidden-attestation.json"
    _patch_promote_public_preflight(monkeypatch)
    observed: list[object] = []

    def run_once(**kwargs: object) -> object:
        observed.append(kwargs)
        return _promotion_evaluation_from_signing(signing)

    monkeypatch.setattr(cli, "run_promotion_evaluation", run_once)
    assert cli.main(_promote_argv(
        tmp_path, trust_path=trust_path, key_path=key_path, output=output
    )) == 0

    result = _stdout(capsys)
    assert len(observed) == 1
    assert observed[0]["parity_oracle_trust_path"] == tmp_path / "parity-oracle-trust.json"
    payload = json.loads(output.read_text(encoding="utf-8"))["payload"]
    assert output.stat().st_mode & 0o777 == 0o600
    assert payload["evaluator_key_id"] == "evaluator-1"
    assert payload["trust_manifest_hash"] == trust_hash
    assert payload["issued_at"] == "2026-08-11T00:00:00Z"
    assert payload["evaluator_id"] == "isolated-evaluator-1"
    assert result["signed"] is True and result["evaluated"] is True and result["passed"] is True
    assert result["status"] == "complete"
    rendered = json.dumps(result)
    for forbidden in ("isolated-evaluator-1", str(key_path), str(trust_path), str(output)):
        assert forbidden not in rendered


@pytest.mark.parametrize("requested_becomes_winner", (True, False))
def test_stage2_evaluator_promote_keeps_winner_when_requested_fallback_is_unqualified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    requested_becomes_winner: bool,
) -> None:
    key_path = tmp_path / "evaluator-private.pem"
    key = _write_private_key(key_path)
    trust_path = tmp_path / "deployment-trust.json"
    _write_trust_manifest(trust_path, key)
    winner_output = tmp_path / "winner-attestation.json"
    fallback_output = tmp_path / "fallback-attestation.json"
    _patch_promote_public_preflight(monkeypatch)
    base = _promotion_signing_input()
    if requested_becomes_winner:
        winner = replace(
            base,
            candidate_id="backup",
            winner_candidate_id="backup",
        )
        candidates = {"backup": winner}
    else:
        winner = base
        candidates = {
            "candidate-1": winner,
            "backup": replace(
                base,
                candidate_id="backup",
                winner_candidate_id="candidate-1",
                release_role="qualified_fallback",
                passed=False,
                failures=("hidden gate failed",),
            ),
        }
    calls: list[object] = []

    def run_once(**kwargs: object) -> object:
        calls.append(kwargs)
        return SimpleNamespace(
            candidates=candidates,
            winner_candidate_id=winner.candidate_id,
            evaluation_manifest_hash=winner.evaluation_manifest_hash,
            evaluation_run_id=winner.evaluation_run_id,
            promotion_marker_hash=winner.promotion_marker_hash,
            promotion_batch_hash=winner.promotion_batch_hash,
        )

    monkeypatch.setattr(cli, "run_promotion_evaluation", run_once)
    argv = _promote_argv(
        tmp_path,
        trust_path=trust_path,
        key_path=key_path,
        output=winner_output,
    )
    argv.extend([
        "--candidate", f"backup={tmp_path / 'backup-candidate.json'}",
        "--submission", f"backup={tmp_path / 'backup-submission.json'}",
        "--public-evidence", f"backup={tmp_path / 'backup-evidence.json'}",
        "--qualified-fallback-output", f"backup={fallback_output}",
    ])

    assert cli.main(argv) == 0
    result = _stdout(capsys)
    assert len(calls) == 1
    assert winner_output.is_file()
    assert not fallback_output.exists()
    assert result["candidate_id"] == winner.candidate_id
    assert result["unqualified_fallback_candidate_ids"] == ["backup"]
    assert result["qualified_fallback_attestation_sha256"] == {}


def test_stage2_evaluator_promote_signs_requested_qualified_fallback_in_same_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    key_path = tmp_path / "evaluator-private.pem"
    key = _write_private_key(key_path)
    trust_path = tmp_path / "deployment-trust.json"
    _write_trust_manifest(trust_path, key)
    winner_output = tmp_path / "winner-attestation.json"
    fallback_output = tmp_path / "fallback-attestation.json"
    _patch_promote_public_preflight(monkeypatch)
    winner = _promotion_signing_input()
    fallback = replace(
        winner,
        candidate_id="backup",
        release_role="qualified_fallback",
    )
    calls: list[object] = []

    def run_once(**kwargs: object) -> object:
        calls.append(kwargs)
        return SimpleNamespace(
            candidates={"candidate-1": winner, "backup": fallback},
            winner_candidate_id="candidate-1",
            evaluation_manifest_hash=winner.evaluation_manifest_hash,
            evaluation_run_id=winner.evaluation_run_id,
            promotion_marker_hash=winner.promotion_marker_hash,
            promotion_batch_hash=winner.promotion_batch_hash,
        )

    monkeypatch.setattr(cli, "run_promotion_evaluation", run_once)
    argv = _promote_argv(
        tmp_path,
        trust_path=trust_path,
        key_path=key_path,
        output=winner_output,
    )
    argv.extend([
        "--candidate", f"backup={tmp_path / 'backup-candidate.json'}",
        "--submission", f"backup={tmp_path / 'backup-submission.json'}",
        "--public-evidence", f"backup={tmp_path / 'backup-evidence.json'}",
        "--qualified-fallback-output", f"backup={fallback_output}",
    ])

    assert cli.main(argv) == 0
    result = _stdout(capsys)
    assert len(calls) == 1
    assert winner_output.is_file() and fallback_output.is_file()
    assert result["unqualified_fallback_candidate_ids"] == []
    assert result["qualified_fallback_attestation_sha256"] == {
        "backup": content_hash(json.loads(fallback_output.read_text(encoding="utf-8")))
    }
    assert json.loads(fallback_output.read_text(encoding="utf-8"))["payload"][
        "release_role"
    ] == "qualified_fallback"


def test_stage2_evaluator_promote_signs_failed_gate_and_returns_failed_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    key_path = tmp_path / "evaluator-private.pem"
    key = _write_private_key(key_path)
    trust_path = tmp_path / "deployment-trust.json"
    payload = _payload(trust_manifest_hash=_write_trust_manifest(trust_path, key))
    payload["result_summary"] = {
        "passed": False,
        "failures": ["hidden gate failed"],
        "gate_versions": {"promotion": "1", "determinism": "1"},
    }
    output = tmp_path / "failed-attestation.json"
    _patch_promote_public_preflight(monkeypatch)
    monkeypatch.setattr(cli, "run_promotion_evaluation", lambda **_kwargs: _promotion_evaluation(payload))

    assert cli.main(_promote_argv(
        tmp_path, trust_path=trust_path, key_path=key_path, output=output
    )) == 1

    result = _stdout(capsys)
    assert json.loads(output.read_text(encoding="utf-8"))["payload"]["result_summary"]["passed"] is False
    assert result["signed"] is True and result["evaluated"] is True and result["passed"] is False
    assert result["status"] == "failed"


def test_stage2_evaluator_promote_real_failed_gate_consumes_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import paper_agent.stage2_promotion_artifacts as artifacts_io
    from paper_agent.stage2_evaluation import write_gold_manifest
    from test_stage2_promotion_artifacts import _candidate, _gold, _submission

    manifest, labels_document = _gold()
    labels = artifacts_io.private_gold_labels_from_document(labels_document, manifest=manifest)
    candidate = _candidate(manifest, labels)
    manifest_path = tmp_path / "gold-manifest.json"
    labels_path = tmp_path / "private-labels.json"
    submission_path = tmp_path / "private-submission.json"
    write_gold_manifest(manifest_path, manifest)
    labels_path.write_text(json.dumps(labels_document), encoding="utf-8")
    submission_path.write_text(
        json.dumps(artifacts_io.promotion_submission_document(
            _submission(manifest, labels, candidate, reject_all=True)
        )),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        cli,
        "validate_promotion_candidate_bundles",
        lambda _paths, *, expected_manifest_hash: None,
    )
    monkeypatch.setattr(
        cli,
        "validate_promotion_public_evidence",
        lambda _candidates, _evidence, _manifest_hash, *, parity_oracle_trust_path: None,
    )
    monkeypatch.setattr(
        artifacts_io,
        "_candidate_artifacts_and_release_hash",
        lambda _path: (candidate, "d" * 64),
    )
    monkeypatch.setattr(
        artifacts_io,
        "_public_release_evidence",
        lambda *_args, **_kwargs: {"candidate": _verified_public_gates()},
    )
    key_path = tmp_path / "evaluator-private.pem"
    key = _write_private_key(key_path)
    trust_path = tmp_path / "deployment-trust.json"
    _write_trust_manifest(trust_path, key)
    output = tmp_path / "failed-attestation.json"
    argv = _promote_argv(
        tmp_path,
        trust_path=trust_path,
        key_path=key_path,
        output=output,
        candidate_id="candidate",
    ) + ["--bootstrap-iterations", "10"]

    assert cli.main(argv) == 1
    result = _stdout(capsys)
    assert result["signed"] is True and result["passed"] is False
    assert output.is_file()
    assert (tmp_path / "sealed-state" / f"{manifest.hash()}.promotion.json").is_file()


def test_stage2_evaluator_promote_dry_run_never_reads_private_inputs_key_or_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    trust_path = tmp_path / "deployment-trust.json"
    trust_key = Ed25519PrivateKey.generate()
    _write_trust_manifest(trust_path, trust_key)
    output = tmp_path / "new" / "attestation.json"
    _patch_promote_public_preflight(monkeypatch)

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("dry-run must not read sealed inputs or the private key")

    monkeypatch.setattr(cli, "run_promotion_evaluation", forbidden)
    monkeypatch.setattr(cli, "load_hidden_evaluator_private_key", forbidden)
    argv = _promote_argv(
        tmp_path, trust_path=trust_path, key_path=tmp_path / "missing.pem", output=output
    ) + ["--dry-run"]
    assert cli.main(argv) == 0

    result = _stdout(capsys)
    assert result["status"] == "validated"
    assert result["evaluated"] is False and result["signed"] is False
    assert not output.exists() and not output.parent.exists()


def test_stage2_evaluator_promote_does_not_accept_operator_selected_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_promote_public_preflight(monkeypatch)
    argv = _promote_argv(
        tmp_path,
        trust_path=tmp_path / "unused-trust.json",
        key_path=tmp_path / "unused-key.pem",
        output=tmp_path / "attestation.json",
    ) + ["--dry-run", "--selected-candidate-id", "candidate-1"]

    with pytest.raises(SystemExit):
        cli.main(argv)


@pytest.mark.parametrize(
    ("option", "value"),
    (
        ("--candidate", "bad candidate=/tmp/candidate.json"),
        ("--submission", "bad submission=/tmp/submission.json"),
        ("--incumbent-candidate-id", "bad incumbent"),
        ("--evaluator-id", "bad evaluator"),
        ("--evaluation-run-id", "bad evaluation"),
        ("--evaluator-key-id", "bad key"),
    ),
)
def test_stage2_evaluator_promote_rejects_every_non_schema_identifier_before_dry_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    option: str,
    value: str,
) -> None:
    trust_path = tmp_path / "deployment-trust.json"
    output = tmp_path / "attestation.json"
    argv = _replace_option(
        _promote_argv(
            tmp_path,
            trust_path=trust_path,
            key_path=tmp_path / "missing.pem",
            output=output,
        ) + ["--dry-run"],
        option,
        value,
    )
    monkeypatch.setattr(
        cli,
        "validate_promotion_candidate_bundles",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("identifier validation must happen before public preflight")
        ),
    )

    with pytest.raises(ValueError, match="identifier|ID=PATH"):
        cli.main(argv)


def test_stage2_evaluator_promote_uses_schema_date_time_shape_in_dry_run(
    tmp_path: Path,
) -> None:
    argv = _replace_option(
        _promote_argv(
            tmp_path,
            trust_path=tmp_path / "unused-trust.json",
            key_path=tmp_path / "unused-key.pem",
            output=tmp_path / "attestation.json",
        ) + ["--dry-run"],
        "--issued-at",
        "2026-08-11 00:00:00Z",
    )

    with pytest.raises(ValueError, match="schema-valid date-time"):
        cli.main(argv)


def test_stage2_evaluator_promote_public_preflight_validates_sampling_and_manifest_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import paper_agent.stage2_evaluation as evaluation

    observed: list[object] = []
    manifest = SimpleNamespace(
        hash=lambda: "b" * 64,
        validate_sampling_structure=lambda: observed.append("sampling"),
    )
    monkeypatch.setattr(evaluation, "load_gold_manifest", lambda _path: manifest)
    monkeypatch.setattr(
        cli,
        "validate_promotion_candidate_bundles",
        lambda paths, *, expected_manifest_hash: observed.append(
            (set(paths), expected_manifest_hash)
        ),
    )
    monkeypatch.setattr(
        cli,
        "validate_promotion_public_evidence",
        lambda candidates, _evidence, manifest_hash, *, parity_oracle_trust_path: observed.append(
            (set(candidates), manifest_hash, "public", parity_oracle_trust_path)
        ),
    )
    trust_path = tmp_path / "deployment-trust.json"
    _write_trust_manifest(trust_path, Ed25519PrivateKey.generate())

    assert cli.main(_promote_argv(
        tmp_path,
        trust_path=trust_path,
        key_path=tmp_path / "unused.pem",
        output=tmp_path / "attestation.json",
    ) + ["--dry-run"]) == 0
    assert observed == [
        "sampling",
        ({"candidate-1"}, "b" * 64),
        ({"candidate-1"}, "b" * 64, "public", tmp_path / "parity-oracle-trust.json"),
    ]


def test_stage2_evaluator_promote_rejects_invalid_sampling_before_candidate_or_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import paper_agent.stage2_evaluation as evaluation

    manifest = SimpleNamespace(
        hash=lambda: "b" * 64,
        validate_sampling_structure=lambda: (_ for _ in ()).throw(
            ValueError("private-looking pair detail")
        ),
    )
    monkeypatch.setattr(evaluation, "load_gold_manifest", lambda _path: manifest)
    monkeypatch.setattr(
        cli,
        "validate_promotion_candidate_bundles",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("invalid sampling must stop candidate validation")
        ),
    )

    with pytest.raises(ValueError, match="public inputs are invalid"):
        cli.main(_promote_argv(
            tmp_path,
            trust_path=tmp_path / "unused-trust.json",
            key_path=tmp_path / "unused-key.pem",
            output=tmp_path / "attestation.json",
        ))


def test_stage2_evaluator_promote_existing_output_and_bad_key_stop_before_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trust_path = tmp_path / "deployment-trust.json"
    trust_key = Ed25519PrivateKey.generate()
    _write_trust_manifest(trust_path, trust_key)
    _patch_promote_public_preflight(monkeypatch)
    output = tmp_path / "attestation.json"
    output.write_text("existing", encoding="utf-8")

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("marker evaluation must not run")

    monkeypatch.setattr(cli, "run_promotion_evaluation", forbidden)
    monkeypatch.setattr(cli, "load_hidden_evaluator_private_key", forbidden)
    with pytest.raises(FileExistsError, match="already exists"):
        cli.main(_promote_argv(
            tmp_path, trust_path=trust_path, key_path=tmp_path / "missing.pem", output=output
        ))

    output.unlink()
    mismatched_key_path = tmp_path / "mismatched-private.pem"
    _write_private_key(mismatched_key_path)
    monkeypatch.setattr(
        cli, "load_hidden_evaluator_private_key", load_hidden_evaluator_private_key
    )
    with pytest.raises(ValueError, match="signing prerequisites"):
        cli.main(_promote_argv(
            tmp_path, trust_path=trust_path, key_path=mismatched_key_path, output=output
        ))
    assert not output.exists()


def test_stage2_evaluator_promote_redacts_private_evaluation_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    key_path = tmp_path / "evaluator-private.pem"
    key = _write_private_key(key_path)
    trust_path = tmp_path / "deployment-trust.json"
    _write_trust_manifest(trust_path, key)
    _patch_promote_public_preflight(monkeypatch)
    secret = "PRIVATE-pair-012-label-3-prediction"
    monkeypatch.setattr(
        cli, "run_promotion_evaluation",
        lambda **_kwargs: (_ for _ in ()).throw(PrivatePromotionArtifactError(secret)),
    )
    output = tmp_path / "attestation.json"

    assert cli.entrypoint(_promote_argv(
        tmp_path, trust_path=trust_path, key_path=key_path, output=output
    )) == 1
    result = _stdout(capsys)
    assert result["error"] == "sealed Stage 2 promotion evaluation failed"
    assert secret not in json.dumps(result)
    assert not output.exists()


def test_stage2_evaluator_promote_removes_partial_reserved_output_on_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    key_path = tmp_path / "evaluator-private.pem"
    key = _write_private_key(key_path)
    trust_path = tmp_path / "deployment-trust.json"
    _write_trust_manifest(trust_path, key)
    output = tmp_path / "attestation.json"
    _patch_promote_public_preflight(monkeypatch)
    monkeypatch.setattr(
        cli,
        "run_promotion_evaluation",
        lambda **_kwargs: _promotion_evaluation_from_signing(_promotion_signing_input()),
    )

    def partial_write(descriptor: int, _attestation: object) -> None:
        os.write(descriptor, b"partial")
        raise OSError("simulated write failure")

    monkeypatch.setattr(cli, "_write_reserved_hidden_promotion_output", partial_write)
    assert cli.entrypoint(_promote_argv(
        tmp_path, trust_path=trust_path, key_path=key_path, output=output
    )) == 1
    assert _stdout(capsys)["error"] == "Stage 2 promotion attestation output failed"
    assert not output.exists()


def test_stage2_evaluator_promote_removes_winner_and_fallback_on_second_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    key_path = tmp_path / "evaluator-private.pem"
    key = _write_private_key(key_path)
    trust_path = tmp_path / "deployment-trust.json"
    _write_trust_manifest(trust_path, key)
    winner_output = tmp_path / "winner.json"
    fallback_output = tmp_path / "fallback.json"
    _patch_promote_public_preflight(monkeypatch)
    winner = _promotion_signing_input()
    fallback = replace(
        winner, candidate_id="backup", release_role="qualified_fallback"
    )
    monkeypatch.setattr(
        cli,
        "run_promotion_evaluation",
        lambda **_kwargs: SimpleNamespace(
            candidates={"candidate-1": winner, "backup": fallback},
            winner_candidate_id="candidate-1",
            evaluation_manifest_hash=winner.evaluation_manifest_hash,
            evaluation_run_id=winner.evaluation_run_id,
            promotion_marker_hash=winner.promotion_marker_hash,
            promotion_batch_hash=winner.promotion_batch_hash,
        ),
    )
    write = cli._write_reserved_hidden_promotion_output
    calls = 0

    def fail_second(descriptor: int, attestation: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            os.write(descriptor, b"partial fallback")
            raise OSError("second output failed")
        write(descriptor, attestation)

    monkeypatch.setattr(cli, "_write_reserved_hidden_promotion_output", fail_second)
    argv = _promote_argv(
        tmp_path, trust_path=trust_path, key_path=key_path, output=winner_output
    )
    argv.extend([
        "--candidate", f"backup={tmp_path / 'backup-candidate.json'}",
        "--submission", f"backup={tmp_path / 'backup-submission.json'}",
        "--public-evidence", f"backup={tmp_path / 'backup-evidence.json'}",
        "--qualified-fallback-output", f"backup={fallback_output}",
    ])

    assert cli.entrypoint(argv) == 1
    assert _stdout(capsys)["error"] == "Stage 2 promotion attestation output failed"
    assert not winner_output.exists()
    assert not fallback_output.exists()


def test_stage2_evaluator_promote_cleans_reservation_when_mode_preflight_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    key_path = tmp_path / "evaluator-private.pem"
    key = _write_private_key(key_path)
    trust_path = tmp_path / "deployment-trust.json"
    _write_trust_manifest(trust_path, key)
    output = tmp_path / "attestation.json"
    _patch_promote_public_preflight(monkeypatch)
    monkeypatch.setattr(
        cli.os,
        "fchmod",
        lambda *_args: (_ for _ in ()).throw(OSError("simulated chmod failure")),
    )

    assert cli.entrypoint(_promote_argv(
        tmp_path, trust_path=trust_path, key_path=key_path, output=output
    )) == 1
    assert _stdout(capsys)["error"] == "Stage 2 promotion attestation output failed"
    assert not output.exists()


def test_stage2_evaluator_promote_never_removes_replacement_inode_on_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    key_path = tmp_path / "evaluator-private.pem"
    key = _write_private_key(key_path)
    trust_path = tmp_path / "deployment-trust.json"
    _write_trust_manifest(trust_path, key)
    output = tmp_path / "attestation.json"
    replacement = tmp_path / "replacement.json"
    _patch_promote_public_preflight(monkeypatch)
    monkeypatch.setattr(
        cli,
        "run_promotion_evaluation",
        lambda **_kwargs: _promotion_evaluation_from_signing(_promotion_signing_input()),
    )

    def replace_then_fail(descriptor: int, _attestation: object) -> None:
        os.write(descriptor, b"partial old inode")
        replacement.write_bytes(b"replacement must survive")
        os.replace(replacement, output)
        raise OSError("simulated replacement race")

    monkeypatch.setattr(cli, "_write_reserved_hidden_promotion_output", replace_then_fail)
    assert cli.entrypoint(_promote_argv(
        tmp_path, trust_path=trust_path, key_path=key_path, output=output
    )) == 1
    assert _stdout(capsys)["error"] == "Stage 2 promotion attestation output failed"
    assert output.read_bytes() == b"replacement must survive"


def test_stage2_evaluator_promote_fails_closed_when_output_is_replaced_after_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    key_path = tmp_path / "evaluator-private.pem"
    key = _write_private_key(key_path)
    trust_path = tmp_path / "deployment-trust.json"
    _write_trust_manifest(trust_path, key)
    output = tmp_path / "attestation.json"
    replacement = tmp_path / "replacement.json"
    _patch_promote_public_preflight(monkeypatch)
    monkeypatch.setattr(
        cli,
        "run_promotion_evaluation",
        lambda **_kwargs: _promotion_evaluation_from_signing(_promotion_signing_input()),
    )
    write_reserved = cli._write_reserved_hidden_promotion_output

    def write_then_replace(descriptor: int, attestation: object) -> None:
        write_reserved(descriptor, attestation)
        replacement.write_bytes(b"replacement must survive")
        os.replace(replacement, output)

    monkeypatch.setattr(cli, "_write_reserved_hidden_promotion_output", write_then_replace)
    assert cli.entrypoint(_promote_argv(
        tmp_path, trust_path=trust_path, key_path=key_path, output=output
    )) == 1
    assert _stdout(capsys)["error"] == "Stage 2 promotion attestation output failed"
    assert output.read_bytes() == b"replacement must survive"


def test_stage2_evaluator_promote_fails_closed_when_output_parent_is_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    key_path = tmp_path / "evaluator-private.pem"
    key = _write_private_key(key_path)
    trust_path = tmp_path / "deployment-trust.json"
    _write_trust_manifest(trust_path, key)
    output = tmp_path / "release" / "attestation.json"
    moved_parent = tmp_path / "moved-release"
    _patch_promote_public_preflight(monkeypatch)
    monkeypatch.setattr(
        cli,
        "run_promotion_evaluation",
        lambda **_kwargs: _promotion_evaluation_from_signing(_promotion_signing_input()),
    )
    write_reserved = cli._write_reserved_hidden_promotion_output

    def write_then_replace_parent(descriptor: int, attestation: object) -> None:
        write_reserved(descriptor, attestation)
        os.replace(output.parent, moved_parent)
        output.parent.mkdir()
        output.write_bytes(b"replacement must survive")

    monkeypatch.setattr(
        cli, "_write_reserved_hidden_promotion_output", write_then_replace_parent
    )
    assert cli.entrypoint(_promote_argv(
        tmp_path, trust_path=trust_path, key_path=key_path, output=output
    )) == 1
    assert _stdout(capsys)["error"] == "Stage 2 promotion attestation output failed"
    assert output.read_bytes() == b"replacement must survive"
    assert not (moved_parent / output.name).exists()


def test_stage2_evaluator_promote_rejects_wrong_bytes_written_to_reserved_inode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    key_path = tmp_path / "evaluator-private.pem"
    key = _write_private_key(key_path)
    trust_path = tmp_path / "deployment-trust.json"
    _write_trust_manifest(trust_path, key)
    output = tmp_path / "attestation.json"
    _patch_promote_public_preflight(monkeypatch)
    monkeypatch.setattr(
        cli,
        "run_promotion_evaluation",
        lambda **_kwargs: _promotion_evaluation_from_signing(_promotion_signing_input()),
    )

    def write_wrong_bytes(descriptor: int, _attestation: object) -> None:
        os.write(descriptor, b"wrong attestation bytes")
        os.fsync(descriptor)

    monkeypatch.setattr(cli, "_write_reserved_hidden_promotion_output", write_wrong_bytes)
    assert cli.entrypoint(_promote_argv(
        tmp_path, trust_path=trust_path, key_path=key_path, output=output
    )) == 1
    assert _stdout(capsys)["error"] == "Stage 2 promotion attestation output failed"
    assert not output.exists()


def test_stage2_evaluator_promote_rejects_candidate_mapping_before_private_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trust_path = tmp_path / "deployment-trust.json"
    trust_key = Ed25519PrivateKey.generate()
    _write_trust_manifest(trust_path, trust_key)
    import paper_agent.stage2_evaluation as evaluation

    monkeypatch.setattr(
        evaluation,
        "load_gold_manifest",
        lambda _path: SimpleNamespace(
            hash=lambda: "b" * 64,
            validate_sampling_structure=lambda: None,
        ),
    )
    monkeypatch.setattr(
        cli, "validate_promotion_candidate_bundles",
        lambda _paths, *, expected_manifest_hash: (_ for _ in ()).throw(
            ValueError("candidate path details")
        ),
    )

    def forbidden(_path: Path) -> object:
        raise AssertionError("bad public candidate mapping must not load the private key")

    monkeypatch.setattr(cli, "load_hidden_evaluator_private_key", forbidden)
    with pytest.raises(ValueError, match="public inputs are invalid"):
        cli.main(_promote_argv(
            tmp_path,
            trust_path=trust_path,
            key_path=tmp_path / "missing.pem",
            output=tmp_path / "attestation.json",
        ))


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
