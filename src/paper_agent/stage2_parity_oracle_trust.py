"""Deployment-controlled trust root for the Stage 2 FP32 parity oracle."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from .canonical import content_hash
from .schema import validate


@dataclass(frozen=True, slots=True)
class ParityOracleTrust:
    manifest_hash: str
    trust_manifest_id: str
    official_oracle_model_lock_hash: str
    oracle_calibrator_hash: str
    oracle_threshold_artifact_hash: str
    tokenizer_hash: str
    preprocess_hash: str

    def __post_init__(self) -> None:
        hashes = (
            self.manifest_hash,
            self.official_oracle_model_lock_hash,
            self.oracle_calibrator_hash,
            self.oracle_threshold_artifact_hash,
            self.tokenizer_hash,
            self.preprocess_hash,
        )
        if not self.trust_manifest_id or any(
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in hashes
        ):
            raise ValueError("parity oracle trust requires an ID and lowercase SHA-256 hashes")


def load_parity_oracle_trust(path: Path) -> ParityOracleTrust:
    document = json.loads(path.resolve(strict=True).read_bytes())
    return parity_oracle_trust_from_document(document)


def parity_oracle_trust_from_document(
    document: Mapping[str, Any],
) -> ParityOracleTrust:
    validate(document, "stage2-parity-oracle-trust.schema.json")
    return ParityOracleTrust(
        manifest_hash=content_hash(document),
        trust_manifest_id=str(document["trust_manifest_id"]),
        official_oracle_model_lock_hash=str(
            document["official_oracle_model_lock_hash"]
        ),
        oracle_calibrator_hash=str(document["oracle_calibrator_hash"]),
        oracle_threshold_artifact_hash=str(
            document["oracle_threshold_artifact_hash"]
        ),
        tokenizer_hash=str(document["tokenizer_hash"]),
        preprocess_hash=str(document["preprocess_hash"]),
    )
