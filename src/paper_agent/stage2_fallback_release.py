"""Build a qualified reranker-fallback descriptor for final release assembly."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .stage2_fallback import FallbackReleaseBinding, stage2_shared_runtime_hash
from .stage2_hidden_attestation import HIDDEN_PROMOTION_GATE_POLICY_HASH
from .stage2_release_evidence import load_stage2_release_evidence_index_bytes
from .stage2_search import (
    Stage2ReleaseError,
    _load_stage2_benchmark_candidate_bytes,
    _validate_evidence_bindings,
)


class Stage2FallbackReleaseError(ValueError):
    """Primary and backup artifacts cannot form a qualified fallback candidate."""


@dataclass(frozen=True, slots=True)
class QualifiedStage2Fallback:
    document: Mapping[str, object]
    primary_candidate_id: str
    backup_candidate_id: str
    backup_model_lock_hash: str
    evaluation_manifest_hash: str
    shared_runtime_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "document", MappingProxyType(dict(self.document)))

    def summary(self) -> dict[str, object]:
        return {
            "backup_candidate_id": self.backup_candidate_id,
            "backup_model_lock_hash": self.backup_model_lock_hash,
            "fallback_evaluation_manifest_hash": self.evaluation_manifest_hash,
            "fallback_shared_runtime_hash": self.shared_runtime_hash,
        }


def qualify_stage2_reranker_fallback(
    *,
    primary_candidate_path: Path,
    backup_candidate_path: Path,
    backup_evidence_path: Path,
    omlx_base_url: str | None = None,
    api_key_env: str | None = None,
) -> QualifiedStage2Fallback:
    """Build one descriptor for assembly without mutating the signed candidate.

    Primary and backup v2 candidates are evaluated together.  Their evidence
    binds those unchanged bytes; only final schema-v3 assembly injects this
    descriptor, avoiding a candidate/evidence hash cycle.
    """

    try:
        primary_path = primary_candidate_path.resolve(strict=True)
        backup_path = backup_candidate_path.resolve(strict=True)
        evidence_path = backup_evidence_path.resolve(strict=True)
    except OSError as error:
        raise Stage2FallbackReleaseError(
            f"Stage 2 fallback input cannot be resolved: {error}"
        ) from error
    root = primary_path.parent
    if any(not path.is_relative_to(root) for path in (backup_path, evidence_path)):
        raise Stage2FallbackReleaseError(
            "Stage 2 fallback candidate and evidence must stay inside the primary bundle"
        )
    try:
        primary_bytes = primary_path.read_bytes()
        backup_bytes = backup_path.read_bytes()
        evidence_bytes = evidence_path.read_bytes()
        primary_document = _json_object(primary_bytes, "primary candidate")
        if "reranker_fallback" in primary_document:
            raise Stage2FallbackReleaseError(
                "Stage 2 primary candidate already has a reranker fallback"
            )
        primary = _load_stage2_benchmark_candidate_bytes(primary_path, primary_bytes)
        backup = _load_stage2_benchmark_candidate_bytes(backup_path, backup_bytes)
        if backup.reranker_fallback is not None:
            raise Stage2FallbackReleaseError(
                "Stage 2 backup candidates cannot contain nested fallbacks"
            )
        if primary.profile.reranker_lock_hash == backup.profile.reranker_lock_hash:
            raise Stage2FallbackReleaseError(
                "Stage 2 fallback must use a distinct reranker model lock"
            )
        shared_runtime_hash = stage2_shared_runtime_hash(primary.profile)
        if stage2_shared_runtime_hash(backup.profile) != shared_runtime_hash:
            raise Stage2FallbackReleaseError(
                "Stage 2 fallback query, Qwen path, or runtime semantics differ from the primary"
            )
        index = load_stage2_release_evidence_index_bytes(evidence_path, evidence_bytes)
        backup_hash = sha256(backup_bytes).hexdigest()
        if index.hidden_attestation is None:
            raise Stage2FallbackReleaseError(
                "Stage 2 fallback requires final backup release evidence"
            )
        if index.candidate_bundle_sha256 != backup_hash:
            raise Stage2FallbackReleaseError(
                "Stage 2 fallback evidence does not bind the backup candidate"
            )
        _validate_evidence_bindings(
            index,
            candidate_id=backup.profile_name,
            candidate_bundle_sha256=backup_hash,
            evaluation_manifest_hash=index.evaluation_manifest_hash,
            profile=backup.profile,
        )
    except Stage2FallbackReleaseError:
        raise
    except (OSError, Stage2ReleaseError, ValueError, json.JSONDecodeError) as error:
        raise Stage2FallbackReleaseError(
            f"Stage 2 fallback qualification failed: {error}"
        ) from error

    endpoint = omlx_base_url or backup.omlx_base_url
    runtime = {
        "omlx_base_url": endpoint,
        "api_key_env": backup.api_key_env if api_key_env is None else api_key_env,
    }
    binding = FallbackReleaseBinding(
        backup_candidate_hash=backup_hash,
        backup_release_evidence_hash=sha256(evidence_bytes).hexdigest(),
        evaluation_manifest_hash=index.evaluation_manifest_hash,
        gate_policy_hash=HIDDEN_PROMOTION_GATE_POLICY_HASH,
        shared_runtime_hash=shared_runtime_hash,
    )
    fallback_document = {
        "candidate": _artifact_ref(backup_path, root, backup_bytes),
        "release_evidence": _artifact_ref(evidence_path, root, evidence_bytes),
        "runtime": runtime,
        "release_binding": binding.document(),
    }
    return QualifiedStage2Fallback(
        document=fallback_document,
        primary_candidate_id=primary.profile_name,
        backup_candidate_id=backup.profile_name,
        backup_model_lock_hash=backup.profile.reranker_lock_hash or "",
        evaluation_manifest_hash=index.evaluation_manifest_hash,
        shared_runtime_hash=shared_runtime_hash,
    )


def _artifact_ref(path: Path, root: Path, payload: bytes) -> dict[str, str]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256(payload).hexdigest(),
    }


def _json_object(payload: bytes, label: str) -> dict[str, Any]:
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise Stage2FallbackReleaseError(f"Stage 2 {label} must be an object")
    return value
