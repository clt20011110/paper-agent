"""Reserved queue and fail-closed local reranker fallback contracts for Stage 2.

There is deliberately no external queue implementation in this module.  Stage 2
currently uses its SQLite lease queue; ``QueueBackend`` only fixes the minimal
shape a future implementation must preserve.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Protocol, Sequence

from .canonical import content_hash
from .leases import TaskLease, TaskLeaseSpec
from .stage2_backends import RerankerBackend
from .stage2_evaluation import CalibrationPath

if TYPE_CHECKING:
    from .stage2_pipeline import PathCalibration, Stage2Profile


class QueueBackend(Protocol):
    """Minimal reservation for durable, paper-level Stage 2 work queues."""

    def enqueue_many(
        self,
        *,
        run_id: str,
        stage: str,
        specs: Sequence[TaskLeaseSpec],
        now: str,
    ) -> tuple[TaskLease, ...]: ...

    def claim(
        self,
        *,
        worker_id: str,
        now: str,
        expires_at: str,
        limit: int,
        run_id: str | None = None,
        stage: str | None = None,
        output_kind: str | None = None,
        output_kind_prefix: str | None = None,
    ) -> tuple[TaskLease, ...]: ...


class LocalRerankerBackend(RerankerBackend, Protocol):
    """A reranker that is explicitly safe to run on the local machine only."""

    is_local: bool


@dataclass(frozen=True, slots=True)
class FallbackReleaseBinding:
    """A released backup evaluated under the primary's frozen semantics."""

    backup_candidate_hash: str
    backup_release_evidence_hash: str
    evaluation_manifest_hash: str
    gate_policy_hash: str
    shared_runtime_hash: str

    def __post_init__(self) -> None:
        values = (
            self.backup_candidate_hash,
            self.backup_release_evidence_hash,
            self.evaluation_manifest_hash,
            self.gate_policy_hash,
            self.shared_runtime_hash,
        )
        if any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in values
        ):
            raise ValueError("fallback release bindings require lowercase SHA-256 values")

    def document(self) -> dict[str, str]:
        return {
            "backup_candidate_hash": self.backup_candidate_hash,
            "backup_release_evidence_hash": self.backup_release_evidence_hash,
            "evaluation_manifest_hash": self.evaluation_manifest_hash,
            "gate_policy_hash": self.gate_policy_hash,
            "shared_runtime_hash": self.shared_runtime_hash,
        }


def stage2_effective_config_hash(
    primary_profile_hash: str,
    fallback_identity: Mapping[str, object] | None,
) -> str:
    """Return the deployment/run identity while preserving no-fallback hashes."""

    if fallback_identity is None:
        return primary_profile_hash
    return content_hash({
        "kind": "stage2-effective-config-v1",
        "primary_profile_hash": primary_profile_hash,
        "reranker_fallback": dict(fallback_identity),
    })


def stage2_shared_runtime_hash(profile: Stage2Profile) -> str:
    """Hash every primary/backup semantic except the reranker release itself.

    A backup is evaluated as a complete cascade.  This identity proves that its
    query, Qwen path, screening rules, batching, prompt/schema, and local runtime
    equal the primary's; only its reranker lock and reranker calibration may
    differ.  Qwen's threshold ``stage2_config_hash`` is intentionally omitted
    because that enclosing hash includes the (allowed-to-differ) reranker lock.
    Loopback endpoint and API-key variable are deployment coordinates rather
    than inference semantics; the exact fallback coordinates are separately
    bound by ``runtime_config_hash``.
    """

    qwen = profile.adjudicator_calibration
    qwen_threshold = None
    qwen_calibrator_hash = None
    if qwen is not None:
        qwen_calibrator_hash = qwen.calibrator.hash()
        qwen_threshold = qwen.threshold.document()
        qwen_threshold.pop("stage2_config_hash")
    return content_hash({
        "kind": "stage2-primary-fallback-shared-runtime-v1",
        "query": profile.query,
        "query_version": profile.query_version,
        "screening_scope_hash": profile.screening_scope_hash,
        "evaluation_topic_queries": [
            {"topic": topic, "language": language, "query": query}
            for topic, language, query in profile.evaluation_topic_queries
        ],
        "include_document_types": sorted(profile.include_document_types),
        "exclude_document_types": sorted(profile.exclude_document_types),
        "token_bucket_width": profile.token_bucket_width,
        "document_batch_size": profile.document_batch_size,
        "reranker_max_in_flight": profile.reranker_max_in_flight,
        "adjudicator_concurrency": profile.adjudicator_concurrency,
        "adjudicator_seed": profile.adjudicator_seed,
        "adjudicator_max_context_window": profile.adjudicator_max_context_window,
        "adjudicator_max_output_tokens": profile.adjudicator_max_output_tokens,
        "prompt_version": profile.prompt_version,
        "prompt_hash": profile.prompt_hash,
        "schema_version": profile.schema_version,
        "schema_hash": profile.schema_hash,
        "adjudicator": {
            "model_id": profile.adjudicator_model_id,
            "revision": profile.adjudicator_revision,
            "model_lock_hash": profile.adjudicator_lock_hash,
            "calibrator_hash": qwen_calibrator_hash,
            "threshold": qwen_threshold,
        },
        "legacy_thresholds": (
            profile.thresholds.document() if profile.thresholds is not None else None
        ),
    })


@dataclass(frozen=True, slots=True)
class LocalCalibratedRerankerFallback:
    """An opt-in backup model, never a cloud or automatic model selection path."""

    backend: LocalRerankerBackend
    model_id: str
    model_revision: str
    model_lock_hash: str
    calibration: PathCalibration
    release_binding: FallbackReleaseBinding
    runtime_config_hash: str

    def __post_init__(self) -> None:
        if getattr(self.backend, "is_local", False) is not True:
            raise ValueError("Stage 2 reranker fallback must be explicitly local")
        if not self.model_id or not self.model_revision:
            raise ValueError("fallback model identity is required")
        for label, value in (
            ("model lock", self.model_lock_hash),
            ("runtime config", self.runtime_config_hash),
        ):
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"fallback {label} hash must be a lowercase SHA-256")
        if self.calibration.calibrator.path is not CalibrationPath.RERANKER:
            raise ValueError("fallback calibration must use the reranker path")
        if self.calibration.calibrator.model_lock_hash != self.model_lock_hash:
            raise ValueError("fallback calibration does not match its model lock")

    def identity_document(self) -> dict[str, object]:
        return {
            "backend": self.backend.backend_name,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "model_lock_hash": self.model_lock_hash,
            "calibrator_hash": self.calibration.calibrator.hash(),
            "threshold_hash": self.calibration.threshold.hash(),
            "release_binding": self.release_binding.document(),
            "runtime_config_hash": self.runtime_config_hash,
        }

    def validate_primary_binding(
        self,
        *,
        primary_model_lock_hash: str | None,
        primary_shared_runtime_hash: str,
    ) -> None:
        """Fail closed unless the backup is distinct but semantically compatible."""
        if not primary_model_lock_hash:
            raise ValueError("fallback requires a model-locked primary")
        if self.model_lock_hash == primary_model_lock_hash:
            raise ValueError("fallback must be a distinct local reranker model")
        if self.release_binding.shared_runtime_hash != primary_shared_runtime_hash:
            raise ValueError("fallback release evidence does not match the primary runtime")
