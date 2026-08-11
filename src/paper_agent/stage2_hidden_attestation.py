"""Ed25519 verification for Stage 2 hidden-promotion attestations."""

from __future__ import annotations

from base64 import b64decode, b64encode
from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from .canonical import canonical_json, content_hash
from .schema import SchemaValidationError, validate


ATTESTATION_DOMAIN = b"paper-agent/stage2-hidden-promotion-attestation/v1\x00"
_RELEASE_BINDING_FIELDS = (
    "candidate_id",
    "evaluation_manifest_hash",
    "stage2_config_hash",
    "model_lock_hashes",
    "calibrator_hashes",
    "threshold_hashes",
    "hidden_pair_universe_hashes",
    "hidden_split_pair_counts",
)


class HiddenPromotionAttestationError(ValueError):
    """A hidden-promotion trust root or signed statement is invalid."""


@dataclass(frozen=True, slots=True)
class HiddenPromotionBindings:
    candidate_id: str
    evaluation_manifest_hash: str
    stage2_config_hash: str
    model_lock_hashes: Mapping[str, str]
    calibrator_hashes: Mapping[str, str]
    threshold_hashes: Mapping[str, str]
    hidden_pair_universe_hashes: Mapping[str, str]
    hidden_split_pair_counts: Mapping[str, int]

    def document(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "evaluation_manifest_hash": self.evaluation_manifest_hash,
            "stage2_config_hash": self.stage2_config_hash,
            "model_lock_hashes": dict(self.model_lock_hashes),
            "calibrator_hashes": dict(self.calibrator_hashes),
            "threshold_hashes": dict(self.threshold_hashes),
            "hidden_pair_universe_hashes": dict(self.hidden_pair_universe_hashes),
            "hidden_split_pair_counts": dict(self.hidden_split_pair_counts),
        }


@dataclass(frozen=True, slots=True)
class HiddenEvaluatorTrustKey:
    key_id: str
    public_key: Ed25519PublicKey


@dataclass(frozen=True, slots=True)
class HiddenEvaluatorTrust:
    manifest_hash: str
    keys: Mapping[str, HiddenEvaluatorTrustKey]


@dataclass(frozen=True, slots=True)
class VerifiedHiddenPromotionAttestation:
    evaluator_key_id: str
    payload_sha256: str


def hidden_promotion_gate_policy_document() -> dict[str, Any]:
    """Return the fixed evaluator policy as a JSON-ready document."""

    return {
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


HIDDEN_PROMOTION_GATE_POLICY_HASH = content_hash(
    hidden_promotion_gate_policy_document()
)


def load_hidden_evaluator_trust(path: Path) -> HiddenEvaluatorTrust:
    """Load a deployment-controlled public-key trust manifest."""

    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HiddenPromotionAttestationError(
            f"cannot read hidden evaluator trust manifest: {path}"
        ) from error
    return hidden_evaluator_trust_from_document(document)


def hidden_evaluator_trust_from_document(document: Mapping[str, Any]) -> HiddenEvaluatorTrust:
    """Validate a trust manifest and retain only active promotion keys."""

    _validate(document, "stage2-hidden-evaluator-trust.schema.json")
    keys: dict[str, HiddenEvaluatorTrustKey] = {}
    seen_key_ids: set[str] = set()
    for item in document["keys"]:
        key_id = str(item["key_id"])
        if key_id in seen_key_ids:
            raise HiddenPromotionAttestationError("hidden evaluator key IDs must be unique")
        seen_key_ids.add(key_id)
        public_bytes = _canonical_base64_bytes(
            item["public_key_b64"], expected_length=32, label="public key"
        )
        if item["status"] == "active":
            keys[key_id] = HiddenEvaluatorTrustKey(
                key_id, Ed25519PublicKey.from_public_bytes(public_bytes)
            )
    if not keys:
        raise HiddenPromotionAttestationError("hidden evaluator trust manifest has no active key")
    return HiddenEvaluatorTrust(content_hash(document), MappingProxyType(keys))


def issue_hidden_promotion_attestation(
    payload: Mapping[str, Any], private_key: Ed25519PrivateKey
) -> dict[str, Any]:
    """Sign a validated payload; callers retain custody of the private key."""

    payload_document = dict(payload)
    provisional = {
        "schema_version": "1",
        "payload": payload_document,
        "payload_sha256": content_hash(payload_document),
        "signature_b64": b64encode(bytes(64)).decode("ascii"),
    }
    _validate(provisional, "stage2-hidden-evaluator-attestation.schema.json")
    signature = private_key.sign(_signed_bytes(payload_document))
    return {
        **provisional,
        "signature_b64": b64encode(signature).decode("ascii"),
    }


def verify_hidden_promotion_attestation(
    document: Mapping[str, Any],
    trust: HiddenEvaluatorTrust,
    *,
    expected_bindings: HiddenPromotionBindings,
) -> VerifiedHiddenPromotionAttestation:
    """Verify signature, release bindings, policy version, and passing outcome."""

    _validate(document, "stage2-hidden-evaluator-attestation.schema.json")
    payload = document["payload"]
    if document["payload_sha256"] != content_hash(payload):
        raise HiddenPromotionAttestationError("hidden promotion payload hash is invalid")
    if payload["trust_manifest_hash"] != trust.manifest_hash:
        raise HiddenPromotionAttestationError("hidden promotion trust manifest hash is invalid")
    key_id = str(payload["evaluator_key_id"])
    key = trust.keys.get(key_id)
    if key is None:
        raise HiddenPromotionAttestationError("hidden promotion signer is not an active trusted key")
    signature = _canonical_base64_bytes(
        document["signature_b64"], expected_length=64, label="signature"
    )
    try:
        key.public_key.verify(signature, _signed_bytes(payload))
    except InvalidSignature as error:
        raise HiddenPromotionAttestationError("hidden promotion signature is invalid") from error
    expected = expected_bindings.document()
    for field in _RELEASE_BINDING_FIELDS:
        if canonical_json(payload[field]) != canonical_json(expected[field]):
            raise HiddenPromotionAttestationError(
                f"hidden promotion {field} does not match the expected release binding"
            )
    if payload["gate_policy_hash"] != HIDDEN_PROMOTION_GATE_POLICY_HASH:
        raise HiddenPromotionAttestationError("hidden promotion gate policy hash is invalid")
    summary = payload["result_summary"]
    if summary["passed"] is not True or summary["failures"]:
        raise HiddenPromotionAttestationError("hidden promotion gates did not pass")
    return VerifiedHiddenPromotionAttestation(
        evaluator_key_id=key_id,
        payload_sha256=str(document["payload_sha256"]),
    )


def _signed_bytes(payload: Mapping[str, Any]) -> bytes:
    return ATTESTATION_DOMAIN + canonical_json(payload)


def _canonical_base64_bytes(value: object, *, expected_length: int, label: str) -> bytes:
    if not isinstance(value, str):
        raise HiddenPromotionAttestationError(f"hidden promotion {label} is not base64")
    try:
        decoded = b64decode(value, validate=True)
    except ValueError as error:
        raise HiddenPromotionAttestationError(
            f"hidden promotion {label} is not base64"
        ) from error
    if len(decoded) != expected_length or b64encode(decoded).decode("ascii") != value:
        raise HiddenPromotionAttestationError(
            f"hidden promotion {label} is not canonical base64"
        )
    return decoded


def _validate(document: Mapping[str, Any], schema_name: str) -> None:
    try:
        validate(document, schema_name)
    except SchemaValidationError as error:
        raise HiddenPromotionAttestationError(str(error)) from error
