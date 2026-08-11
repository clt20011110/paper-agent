from __future__ import annotations

from base64 import b64encode
from copy import deepcopy

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from paper_agent.canonical import content_hash
from paper_agent.stage2_hidden_attestation import (
    HIDDEN_PROMOTION_GATE_POLICY_HASH,
    HiddenPromotionAttestationError,
    HiddenPromotionBindings,
    hidden_evaluator_trust_from_document,
    hidden_promotion_gate_policy_document,
    issue_hidden_promotion_attestation,
    verify_hidden_promotion_attestation,
)


HASH = "a" * 64
PRIVATE_KEY = Ed25519PrivateKey.from_private_bytes(bytes(range(32)))


def _trust_document(*, status: str = "active") -> dict:
    public_bytes = PRIVATE_KEY.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    return {
        "schema_version": "1",
        "trust_manifest_type": "stage2-hidden-evaluator",
        "trust_manifest_id": "stage2-evaluator-v1",
        "keys": [{
            "key_id": "evaluator-2026-01",
            "algorithm": "Ed25519",
            "purpose": "stage2-hidden-promotion",
            "public_key_b64": b64encode(public_bytes).decode("ascii"),
            "status": status,
        }],
    }


def _bindings() -> HiddenPromotionBindings:
    path_hashes = {"reranker": "b" * 64, "qwen": "c" * 64}
    return HiddenPromotionBindings(
        candidate_id="candidate-1",
        evaluation_manifest_hash="d" * 64,
        stage2_config_hash="e" * 64,
        model_lock_hashes=path_hashes,
        calibrator_hashes=path_hashes,
        threshold_hashes=path_hashes,
        hidden_pair_universe_hashes={"hidden_hard": "f" * 64, "hidden_real": HASH},
        hidden_split_pair_counts={"hidden_hard": 150, "hidden_real": 150},
    )


def _payload(*, trust_hash: str, bindings: HiddenPromotionBindings | None = None) -> dict:
    active_bindings = bindings or _bindings()
    return {
        "schema_version": "1",
        "attestation_type": "stage2-hidden-promotion",
        "evaluator_key_id": "evaluator-2026-01",
        "evaluator_id": "evaluation-team-1",
        "trust_manifest_hash": trust_hash,
        "issued_at": "2026-08-11T00:00:00Z",
        "evaluation_run_id": "promotion-1",
        "prediction_submission_hash": HASH,
        "promotion_marker_hash": "1" * 64,
        "consumed_hidden_splits": ["hidden_hard", "hidden_real"],
        "gate_policy_hash": HIDDEN_PROMOTION_GATE_POLICY_HASH,
        "result_summary": {
            "passed": True,
            "failures": [],
            "gate_versions": {"promotion": "1", "determinism": "1"},
        },
        **active_bindings.document(),
    }


def _signed_document() -> tuple[dict, object, HiddenPromotionBindings]:
    trust = hidden_evaluator_trust_from_document(_trust_document())
    bindings = _bindings()
    return (
        issue_hidden_promotion_attestation(
            _payload(trust_hash=trust.manifest_hash, bindings=bindings), PRIVATE_KEY
        ),
        trust,
        bindings,
    )


def _verify(document: dict, trust: object, bindings: HiddenPromotionBindings) -> object:
    return verify_hidden_promotion_attestation(
        document,
        trust,
        expected_bindings=bindings,
    )


def _noncanonical_base64(value: str) -> str:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    position = value.index("=") - 1
    return value[:position] + alphabet[alphabet.index(value[position]) + 1] + value[position + 1:]


def test_hidden_promotion_gate_policy_is_stable_and_returns_a_fresh_document() -> None:
    expected = {
        "schema_version": "1",
        "gate_versions": {"promotion": "1", "determinism": "1"},
        "consumed_hidden_splits": ["hidden_hard", "hidden_real"],
        "per_split": {
            "retention_recall_min": 0.95,
            "automatic_coverage_min": 0.95,
            "error_needs_review_rate_max": 0.005,
            "core_retention_recall_min": 0.97,
            "core_retention_min_count": 30,
        },
        "main_language": {"retention_recall_min": 0.90, "positive_min_count": 30},
        "combined_hidden": {
            "core_retention_recall_min": 0.97,
            "requires_nonzero_core_examples": True,
        },
        "hidden_hard": {"positive_f1_min": 0.88, "topic_macro_positive_f1_min": 0.82},
        "hidden_real": {
            "operational_precision_min": 0.80,
            "brier_score_max": 0.15,
            "ece_10_max": 0.08,
            "ece_10_min_count": 500,
        },
        "determinism": {
            "agreement_min": 0.99,
            "runs": 3,
            "requires_exact_hidden_universe": True,
            "requires_identical_provenance": True,
        },
    }

    assert hidden_promotion_gate_policy_document() == expected
    assert HIDDEN_PROMOTION_GATE_POLICY_HASH == "be69a447ad1b01ade1aacb597dc0d1a2ecd1d2c264f99a4bdda86954238001e6"
    changed = hidden_promotion_gate_policy_document()
    changed["schema_version"] = "2"
    assert hidden_promotion_gate_policy_document() == expected


def test_issues_and_verifies_a_domain_separated_attestation() -> None:
    document, trust, bindings = _signed_document()

    result = _verify(document, trust, bindings)

    assert result.evaluator_key_id == "evaluator-2026-01"
    assert result.payload_sha256 == content_hash(document["payload"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("candidate_id", "candidate-2"),
        ("evaluation_manifest_hash", "3" * 64),
        ("stage2_config_hash", "4" * 64),
        ("model_lock_hashes", {"reranker": "5" * 64, "qwen": "c" * 64}),
        ("hidden_pair_universe_hashes", {"hidden_hard": "6" * 64, "hidden_real": HASH}),
    ],
)
def test_rejects_each_mismatched_release_binding(field: str, value: object) -> None:
    document, trust, bindings = _signed_document()
    changed_payload = deepcopy(document["payload"])
    changed_payload[field] = value
    changed = issue_hidden_promotion_attestation(changed_payload, PRIVATE_KEY)

    with pytest.raises(HiddenPromotionAttestationError, match="expected release binding"):
        _verify(changed, trust, bindings)


def test_rejects_mismatched_hidden_split_counts() -> None:
    document, trust, bindings = _signed_document()
    wrong_bindings = HiddenPromotionBindings(
        **{**bindings.document(), "hidden_split_pair_counts": {"hidden_hard": 149, "hidden_real": 150}}
    )

    with pytest.raises(HiddenPromotionAttestationError, match="expected release binding"):
        _verify(document, trust, wrong_bindings)


def test_rejects_a_tampered_payload_hash_before_signature_check() -> None:
    document, trust, bindings = _signed_document()
    document["payload_sha256"] = "0" * 64

    with pytest.raises(HiddenPromotionAttestationError, match="payload hash"):
        _verify(document, trust, bindings)


def test_rejects_a_tampered_signature() -> None:
    document, trust, bindings = _signed_document()
    document["signature_b64"] = b64encode(bytes(64)).decode("ascii")

    with pytest.raises(HiddenPromotionAttestationError, match="signature is invalid"):
        _verify(document, trust, bindings)


def test_rejects_noncanonical_signature_base64() -> None:
    document, trust, bindings = _signed_document()
    document["signature_b64"] = _noncanonical_base64(document["signature_b64"])

    with pytest.raises(HiddenPromotionAttestationError, match="canonical base64"):
        _verify(document, trust, bindings)


def test_rejects_trust_manifest_hash_drift() -> None:
    document, trust, bindings = _signed_document()
    changed_payload = deepcopy(document["payload"])
    changed_payload["trust_manifest_hash"] = "7" * 64
    changed = issue_hidden_promotion_attestation(changed_payload, PRIVATE_KEY)

    with pytest.raises(HiddenPromotionAttestationError, match="trust manifest hash"):
        _verify(changed, trust, bindings)


@pytest.mark.parametrize(
    "summary",
    [
        {"passed": False, "failures": [], "gate_versions": {"promotion": "1", "determinism": "1"}},
        {"passed": True, "failures": ["promotion"], "gate_versions": {"promotion": "1", "determinism": "1"}},
    ],
)
def test_rejects_non_passing_hidden_gate_summary(summary: dict) -> None:
    document, trust, bindings = _signed_document()
    changed_payload = deepcopy(document["payload"])
    changed_payload["result_summary"] = summary
    changed = issue_hidden_promotion_attestation(changed_payload, PRIVATE_KEY)

    with pytest.raises(HiddenPromotionAttestationError, match="gates did not pass"):
        _verify(changed, trust, bindings)


def test_rejects_wrong_gate_policy_and_untrusted_key() -> None:
    document, trust, bindings = _signed_document()
    changed_payload = deepcopy(document["payload"])
    changed_payload["gate_policy_hash"] = "8" * 64
    changed = issue_hidden_promotion_attestation(changed_payload, PRIVATE_KEY)
    with pytest.raises(HiddenPromotionAttestationError, match="gate policy hash"):
        _verify(changed, trust, bindings)

    changed_payload = deepcopy(document["payload"])
    changed_payload["evaluator_key_id"] = "not-trusted"
    changed = issue_hidden_promotion_attestation(changed_payload, PRIVATE_KEY)
    with pytest.raises(HiddenPromotionAttestationError, match="not an active trusted key"):
        _verify(changed, trust, bindings)


def test_trust_manifest_requires_canonical_keys_unique_ids_and_an_active_key() -> None:
    retired = _trust_document(status="retired")
    with pytest.raises(HiddenPromotionAttestationError, match="no active key"):
        hidden_evaluator_trust_from_document(retired)

    duplicate = _trust_document()
    duplicate["keys"].append(deepcopy(duplicate["keys"][0]))
    with pytest.raises(HiddenPromotionAttestationError, match="unique"):
        hidden_evaluator_trust_from_document(duplicate)

    noncanonical = _trust_document()
    noncanonical["keys"][0]["public_key_b64"] = _noncanonical_base64(
        noncanonical["keys"][0]["public_key_b64"]
    )
    with pytest.raises(HiddenPromotionAttestationError, match="canonical base64"):
        hidden_evaluator_trust_from_document(noncanonical)


def test_attestation_schema_excludes_extra_content() -> None:
    document, trust, bindings = _signed_document()
    document["payload"]["raw_hidden_labels"] = ["never public"]
    document["payload_sha256"] = content_hash(document["payload"])

    with pytest.raises(HiddenPromotionAttestationError, match="Additional properties"):
        _verify(document, trust, bindings)
