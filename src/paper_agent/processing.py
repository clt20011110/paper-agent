"""Artifact-policy gate for content sent to a remote model.

This is deliberately a pre-dispatch boundary: the callback receives a fresh,
minimal ``ModelInvocation`` only after policy or an immutable processing grant
authorizes the exact input artifact.  In particular, a denied request never
hands the callback PDF or normalized-text bytes.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypeVar

import yaml

from .canonical import content_hash
from .grants import GrantError, GrantStore


PROCESSING_PROVIDER = "codex_cli"
PROCESSING_MODEL = "gpt-5.6-luna"
PROCESSING_ACTION = "remote_model_processing"
_DIMENSIONS = (
    "artifact", "input_scope", "license", "access_basis", "provider", "model", "purpose", "data_category",
)


class ProcessingPolicyError(ValueError):
    """The versioned artifact-processing matrix is malformed."""


class ProcessingOutcome(StrEnum):
    FULL_PDF = "full_pdf"
    ABSTRACT_ONLY = "abstract_only"
    METADATA_ONLY = "metadata_only"
    MANUAL = "manual"
    ANALYSIS_NOT_AUTHORIZED = "analysis_not_authorized"


_AUTHORIZED_OUTCOMES = frozenset({
    ProcessingOutcome.FULL_PDF, ProcessingOutcome.ABSTRACT_ONLY, ProcessingOutcome.METADATA_ONLY,
})
_SCOPE_OUTCOMES = {
    "full_pdf": ProcessingOutcome.FULL_PDF,
    "abstract_only": ProcessingOutcome.ABSTRACT_ONLY,
    "metadata_only": ProcessingOutcome.METADATA_ONLY,
}


@dataclass(frozen=True, slots=True)
class ProcessingRequest:
    """Facts and candidate bytes for one remote model invocation.

    ``artifact_hash`` is always the hash of the artifact selected by
    ``artifact``.  Extra bytes may be present here (for example because a
    caller already has a PDF); dispatch intentionally does not expose them
    for an abstract- or metadata-only decision.
    """

    artifact_hash: str
    artifact: str
    input_scope: str
    license: str | None
    access_basis: str
    purpose: str
    data_category: str
    provider: str = PROCESSING_PROVIDER
    model: str = PROCESSING_MODEL
    paper_id: str | None = None
    domain: str | None = None
    mode: str = "attended"
    collection_id: str | None = None
    collection_snapshot_hash: str | None = None
    selection_snapshot_hash: str | None = None
    skill_digest: str | None = None
    dependency_digest: str | None = None
    lineage_hash: str | None = None
    pdf_bytes: bytes | None = None
    normalized_text_bytes: bytes | None = None
    abstract_bytes: bytes | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if len(self.artifact_hash) != 64 or any(char not in "0123456789abcdef" for char in self.artifact_hash):
            raise ValueError("artifact_hash must be a lowercase SHA-256 digest")
        if self.input_scope not in _SCOPE_OUTCOMES:
            raise ValueError("input_scope must be full_pdf, abstract_only, or metadata_only")
        if self.mode not in {"attended", "unattended"}:
            raise ValueError("mode must be attended or unattended")
        payload_hash = _selected_payload_hash(self)
        if payload_hash != self.artifact_hash:
            raise ValueError("artifact_hash does not match the selected processing payload")


@dataclass(frozen=True, slots=True)
class ProcessingDecision:
    """An immutable, content-addressed authorization decision for one input."""

    policy_version: str
    policy_hash: str
    outcome: ProcessingOutcome
    reason_code: str
    input_artifact_hash: str
    provider: str
    model: str
    purpose: str
    data_category: str
    processing_grant_id: str | None
    authorized_by: str | None

    @property
    def audit_hash(self) -> str:
        return content_hash({**asdict(self), "outcome": self.outcome.value})

    @property
    def is_authorized(self) -> bool:
        return self.outcome in _AUTHORIZED_OUTCOMES


@dataclass(frozen=True, slots=True)
class ModelInvocation:
    """The only material passed through the pre-dispatch boundary."""

    decision: ProcessingDecision
    pdf_bytes: bytes | None = None
    normalized_text_bytes: bytes | None = None
    abstract_bytes: bytes | None = None
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class DispatchResult:
    decision: ProcessingDecision
    result: Any | None = None


class ArtifactProcessingPolicy:
    """Strict, ordered evaluation of ``artifact-processing-v1.yaml``."""

    def __init__(self, document: Mapping[str, Any]) -> None:
        self.document = MappingProxyType(dict(document))
        if document.get("schema_version") != "1":
            raise ProcessingPolicyError("artifact-processing policy schema_version must be 1")
        version = document.get("policy_version")
        if not isinstance(version, str) or not version:
            raise ProcessingPolicyError("policy_version is required")
        matrix = document.get("matrix")
        aliases = document.get("license_aliases")
        if not isinstance(matrix, list) or not matrix:
            raise ProcessingPolicyError("matrix must be a non-empty list")
        if not isinstance(aliases, dict):
            raise ProcessingPolicyError("license_aliases must be a mapping")
        self.version = version
        self.hash = content_hash(document)
        self.license_aliases = MappingProxyType({
            _license_key(str(key)): _license_key(str(value)) for key, value in aliases.items()
        })
        self.matrix = tuple(_freeze_rule(rule, index) for index, rule in enumerate(matrix))

    @classmethod
    def load(cls, path: str | Path) -> "ArtifactProcessingPolicy":
        document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ProcessingPolicyError("artifact-processing policy must be a mapping")
        return cls(document)

    def normalize_license(self, value: str | None) -> str:
        return self.license_aliases.get(_license_key(value or ""), _license_key(value or ""))

    def evaluate(self, request: ProcessingRequest) -> tuple[ProcessingOutcome, str]:
        facts = {
            "artifact": request.artifact,
            "input_scope": request.input_scope,
            "license": self.normalize_license(request.license),
            "access_basis": request.access_basis,
            "provider": request.provider,
            "model": request.model,
            "purpose": request.purpose,
            "data_category": request.data_category,
        }
        for index, rule in enumerate(self.matrix):
            if all("*" in rule[key] or facts[key] in rule[key] for key in _DIMENSIONS):
                return ProcessingOutcome(rule["outcome"]), f"policy_rule_{index}"
        return ProcessingOutcome.ANALYSIS_NOT_AUTHORIZED, "no_compatible_policy_rule"


T = TypeVar("T")


class ProcessingGate:
    """Evaluate and dispatch only the bytes covered by a frozen decision."""

    def __init__(self, policy: ArtifactProcessingPolicy, grants: GrantStore | None = None) -> None:
        self.policy = policy
        self.grants = grants

    def decide(
        self,
        request: ProcessingRequest,
        *,
        processing_grant_id: str | None = None,
        now: datetime | str | None = None,
    ) -> ProcessingDecision:
        if request.provider != PROCESSING_PROVIDER or request.model != PROCESSING_MODEL:
            return self._decision(request, ProcessingOutcome.ANALYSIS_NOT_AUTHORIZED, "remote_target_not_permitted", None, None)

        outcome, reason = self.policy.evaluate(request)
        expected = _SCOPE_OUTCOMES[request.input_scope]
        if outcome is expected:
            return self._decision(request, outcome, reason, None, "policy")

        # ``manual`` and unmatched rules can be overridden only by a precise,
        # active grant.  The policy itself never turns a grant id into scope.
        if processing_grant_id and self.grants is not None and now is not None:
            try:
                loaded = self.grants.load(
                    processing_grant_id, kind=PROCESSING_ACTION, now=now
                )
                scope = loaded.document["scope"]
                active = self.grants.require_active(
                    processing_grant_id, kind=PROCESSING_ACTION, action=PROCESSING_ACTION,
                    purpose=request.purpose, mode=request.mode, now=now,
                    paper_id=request.paper_id if scope["paper_ids"] else None,
                    artifact_hash=request.artifact_hash,
                    collection_id=request.collection_id if scope["collection_ids"] else None,
                    collection_snapshot_hash=(
                        request.collection_snapshot_hash if scope["collection_snapshot_hash"] else None
                    ),
                    selection_snapshot_hash=(
                        request.selection_snapshot_hash if scope["selection_snapshot_hash"] else None
                    ),
                    domain=request.domain if scope["domains"] else None,
                    provider=PROCESSING_PROVIDER, model=PROCESSING_MODEL,
                    data_category=request.data_category if scope["data_categories"] else None,
                    skill_digest=request.skill_digest if loaded.document["skill_digest"] else None,
                    dependency_digest=(
                        request.dependency_digest if loaded.document["dependency_digest"] else None
                    ),
                    lineage_hash=request.lineage_hash if loaded.document["lineage_hash"] else None,
                )
                scope = active.document["scope"]
                if not scope["artifact_hashes"] or request.artifact_hash not in scope["artifact_hashes"]:
                    raise GrantError("processing grant must bind the exact artifact hash")
                if scope["provider"] != PROCESSING_PROVIDER or scope["model"] != PROCESSING_MODEL:
                    raise GrantError("processing grant must bind the frozen provider and model")
            except GrantError as error:
                return self._decision(request, _denied_outcome(outcome), f"grant_rejected:{error}", processing_grant_id, None)
            return self._decision(request, expected, "authorized_by_exact_processing_grant", active.grant_id, "grant")

        return self._decision(request, _denied_outcome(outcome), reason, processing_grant_id, None)

    def dispatch(
        self,
        request: ProcessingRequest,
        invoke: Callable[[ModelInvocation], T],
        *,
        processing_grant_id: str | None = None,
        now: datetime | str | None = None,
    ) -> DispatchResult:
        decision = self.decide(request, processing_grant_id=processing_grant_id, now=now)
        if not decision.is_authorized:
            return DispatchResult(decision)
        return DispatchResult(decision, invoke(_invocation_for(request, decision)))

    def _decision(
        self,
        request: ProcessingRequest,
        outcome: ProcessingOutcome,
        reason: str,
        grant_id: str | None,
        authorized_by: str | None,
    ) -> ProcessingDecision:
        return ProcessingDecision(
            policy_version=self.policy.version, policy_hash=self.policy.hash, outcome=outcome,
            reason_code=reason, input_artifact_hash=request.artifact_hash,
            provider=request.provider, model=request.model, purpose=request.purpose,
            data_category=request.data_category, processing_grant_id=grant_id,
            authorized_by=authorized_by,
        )


def _freeze_rule(rule: object, index: int) -> Mapping[str, tuple[str, ...] | str]:
    if not isinstance(rule, dict) or set(rule) != {*_DIMENSIONS, "outcome"}:
        raise ProcessingPolicyError(f"matrix rule {index} must name every processing dimension and outcome")
    frozen: dict[str, tuple[str, ...] | str] = {}
    for key in _DIMENSIONS:
        values = rule[key]
        if not isinstance(values, list) or not values or not all(isinstance(value, str) and value for value in values):
            raise ProcessingPolicyError(f"matrix rule {index} has invalid {key}")
        frozen[key] = tuple(values)
    outcome = rule["outcome"]
    if outcome not in {item.value for item in ProcessingOutcome}:
        raise ProcessingPolicyError(f"matrix rule {index} has invalid outcome")
    frozen["outcome"] = outcome
    return MappingProxyType(frozen)


def _license_key(value: str) -> str:
    return value.strip().lower()


def _denied_outcome(policy_outcome: ProcessingOutcome) -> ProcessingOutcome:
    return ProcessingOutcome.MANUAL if policy_outcome is ProcessingOutcome.MANUAL else ProcessingOutcome.ANALYSIS_NOT_AUTHORIZED


def _invocation_for(request: ProcessingRequest, decision: ProcessingDecision) -> ModelInvocation:
    if decision.outcome is ProcessingOutcome.FULL_PDF:
        if request.artifact == "pdf":
            return ModelInvocation(decision, pdf_bytes=request.pdf_bytes)
        return ModelInvocation(decision, normalized_text_bytes=request.normalized_text_bytes)
    if decision.outcome is ProcessingOutcome.ABSTRACT_ONLY:
        return ModelInvocation(decision, abstract_bytes=request.abstract_bytes)
    if decision.outcome is ProcessingOutcome.METADATA_ONLY:
        metadata = None if request.metadata is None else MappingProxyType(dict(request.metadata))
        return ModelInvocation(decision, metadata=metadata)
    raise AssertionError("unauthorized decisions cannot create model invocations")


def _selected_payload_hash(request: ProcessingRequest) -> str:
    if request.artifact == "pdf" and request.input_scope == "full_pdf" and request.data_category == "full_text":
        if request.pdf_bytes is None:
            raise ValueError("pdf processing requires pdf_bytes")
        return sha256(request.pdf_bytes).hexdigest()
    if (
        request.artifact == "normalized_text"
        and request.input_scope == "full_pdf"
        and request.data_category == "normalized_text"
    ):
        if request.normalized_text_bytes is None:
            raise ValueError("normalized-text processing requires normalized_text_bytes")
        return sha256(request.normalized_text_bytes).hexdigest()
    if request.artifact == "abstract" and request.input_scope == "abstract_only" and request.data_category == "abstract":
        if request.abstract_bytes is None:
            raise ValueError("abstract processing requires abstract_bytes")
        return sha256(request.abstract_bytes).hexdigest()
    if request.artifact == "metadata" and request.input_scope == "metadata_only" and request.data_category == "metadata":
        if request.metadata is None:
            raise ValueError("metadata processing requires metadata")
        return content_hash(dict(request.metadata))
    raise ValueError("artifact, input_scope, and data_category do not identify one payload")
