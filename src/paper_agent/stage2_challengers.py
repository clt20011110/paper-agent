"""Checked-in Stage 2 challenger registry and conservative evaluation preflight.

The registry is deliberately not a model lock: it may record a model-card
candidate while source provenance, license review, and runtime audit are still
pending.  A candidate may be evaluated only after those concrete prerequisites
are satisfied; evaluation eligibility never grants production approval.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping


_IMMUTABLE_REVISION = re.compile(r"^[0-9a-f]{40}$")
_ACCEPTABLE_LOCAL_LICENSES = frozenset({"apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause"})
_REQUIRED_FIELDS = frozenset({
    "id", "source_repo", "source_revision", "revision_status", "parameter_count",
    "parameter_evidence_url", "parameter_evidence", "license",
    "local_deployment_license_status", "backend_capability_status", "production_approved",
})


class ChallengerRegistryError(ValueError):
    """The checked-in challenger registry is malformed or unsafe to use."""


@dataclass(frozen=True, slots=True)
class Stage2Challenger:
    id: str
    source_repo: str
    source_revision: str | None
    revision_status: str
    parameter_count: int
    parameter_evidence_url: str
    parameter_evidence: str
    license: str
    local_deployment_license_status: str
    backend_capability_status: str
    production_approved: bool

    @property
    def has_immutable_revision(self) -> bool:
        return self.source_revision is not None and _IMMUTABLE_REVISION.fullmatch(self.source_revision) is not None

    @property
    def has_acceptable_local_license(self) -> bool:
        return (
            self.local_deployment_license_status == "acceptable"
            and self.license in _ACCEPTABLE_LOCAL_LICENSES
        )

    @property
    def is_within_parameter_bound(self) -> bool:
        return 0 < self.parameter_count <= 10_000_000_000

    @property
    def has_verified_backend_capability(self) -> bool:
        return self.backend_capability_status == "verified"

    @property
    def evaluation_eligible(self) -> bool:
        """Whether provenance, licensing, size, and local execution permit evaluation."""
        return (
            self.has_immutable_revision
            and self.has_acceptable_local_license
            and self.is_within_parameter_bound
            and self.has_verified_backend_capability
        )


def load_stage2_challenger_registry(path: Path) -> tuple[Stage2Challenger, ...]:
    """Load and validate a version-1 challenger registry without touching weights."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ChallengerRegistryError(f"cannot read challenger registry: {error}") from error
    if not isinstance(document, dict) or set(document) != {"kind", "version", "candidates"}:
        raise ChallengerRegistryError("registry must contain only kind, version, and candidates")
    if document["kind"] != "stage2_challenger_registry" or document["version"] != 1:
        raise ChallengerRegistryError("unsupported challenger registry kind or version")
    candidates = document["candidates"]
    if not isinstance(candidates, list) or not candidates:
        raise ChallengerRegistryError("registry candidates must be a non-empty list")
    loaded = tuple(_parse_candidate(value) for value in candidates)
    if len({candidate.id for candidate in loaded}) != len(loaded):
        raise ChallengerRegistryError("candidate ids must be unique")
    return loaded


def evaluation_challengers(path: Path) -> tuple[Stage2Challenger, ...]:
    """Return only challengers safe to admit to a local evaluation queue."""
    return tuple(candidate for candidate in load_stage2_challenger_registry(path) if candidate.evaluation_eligible)


def _parse_candidate(value: Any) -> Stage2Challenger:
    if not isinstance(value, dict) or set(value) != _REQUIRED_FIELDS:
        raise ChallengerRegistryError("each challenger must have exactly the required fields")
    if value["source_revision"] is not None and not isinstance(value["source_revision"], str):
        raise ChallengerRegistryError("source_revision must be a string or null")
    candidate = Stage2Challenger(**value)
    if not all((candidate.id, candidate.source_repo, candidate.parameter_evidence_url, candidate.parameter_evidence, candidate.license, candidate.backend_capability_status)):
        raise ChallengerRegistryError("challenger identity, evidence, license, and backend status are required")
    if not candidate.parameter_evidence_url.startswith("https://huggingface.co/"):
        raise ChallengerRegistryError("parameter evidence must be an authoritative model-card URL")
    if type(candidate.parameter_count) is not int or not candidate.is_within_parameter_bound:
        raise ChallengerRegistryError("parameter_count must be in 1..10B")
    if type(candidate.production_approved) is not bool or candidate.production_approved:
        raise ChallengerRegistryError("challenger registry cannot production-approve a candidate")
    if candidate.source_revision is None:
        if candidate.revision_status != "unapproved_candidate":
            raise ChallengerRegistryError("an unpinned candidate must be explicitly unapproved")
    elif not candidate.has_immutable_revision:
        raise ChallengerRegistryError("source_revision must be a full immutable 40-character SHA")
    elif candidate.revision_status != "repository_recorded":
        raise ChallengerRegistryError("pinned candidate revision must state its provenance status")
    if candidate.local_deployment_license_status not in {"acceptable", "unverified", "unacceptable"}:
        raise ChallengerRegistryError("unknown local deployment license status")
    return candidate
