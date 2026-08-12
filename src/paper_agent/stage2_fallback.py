"""Reserved queue and fail-closed local reranker fallback contracts for Stage 2.

There is deliberately no external queue implementation in this module.  Stage 2
currently uses its SQLite lease queue; ``QueueBackend`` only fixes the minimal
shape a future implementation must preserve.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, Sequence

from .leases import TaskLease, TaskLeaseSpec
from .stage2_backends import RerankerBackend
from .stage2_evaluation import CalibrationPath

if TYPE_CHECKING:
    from .stage2_pipeline import PathCalibration


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
    """Independent release evidence evaluated under one frozen gate policy."""

    primary_release_evidence_hash: str
    backup_release_evidence_hash: str
    evaluation_manifest_hash: str
    gate_policy_hash: str

    def __post_init__(self) -> None:
        if not all((
            self.primary_release_evidence_hash,
            self.backup_release_evidence_hash,
            self.evaluation_manifest_hash,
            self.gate_policy_hash,
        )):
            raise ValueError("fallback release evidence and gate policy hashes are required")
        if self.primary_release_evidence_hash == self.backup_release_evidence_hash:
            raise ValueError("primary and backup require independent release evidence")

    def document(self) -> dict[str, str]:
        return {
            "primary_release_evidence_hash": self.primary_release_evidence_hash,
            "backup_release_evidence_hash": self.backup_release_evidence_hash,
            "evaluation_manifest_hash": self.evaluation_manifest_hash,
            "gate_policy_hash": self.gate_policy_hash,
        }


@dataclass(frozen=True, slots=True)
class LocalCalibratedRerankerFallback:
    """An opt-in backup model, never a cloud or automatic model selection path."""

    backend: LocalRerankerBackend
    model_lock_hash: str
    calibration: PathCalibration
    release_binding: FallbackReleaseBinding

    def __post_init__(self) -> None:
        if getattr(self.backend, "is_local", False) is not True:
            raise ValueError("Stage 2 reranker fallback must be explicitly local")
        if not self.model_lock_hash:
            raise ValueError("fallback model lock hash is required")
        if self.calibration.calibrator.path is not CalibrationPath.RERANKER:
            raise ValueError("fallback calibration must use the reranker path")
        if self.calibration.calibrator.model_lock_hash != self.model_lock_hash:
            raise ValueError("fallback calibration does not match its model lock")

    def validate_primary_binding(
        self,
        *,
        primary_model_lock_hash: str | None,
        primary_release_gate_hash: str | None,
    ) -> None:
        """Fail closed unless the backup remains model-distinct from its primary."""
        if not primary_model_lock_hash:
            raise ValueError("fallback requires a model-locked primary")
        if self.model_lock_hash == primary_model_lock_hash:
            raise ValueError("fallback must be a distinct local reranker model")
