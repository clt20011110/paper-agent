"""Authoring helpers for the human Stage 2 rationale audit.

This module freezes already-selected rationale examples and turns completed
human worklists into the raw artifacts consumed by ``rationale_audit_gate``.
It deliberately does not generate, infer, or default audit labels.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .canonical import content_hash
from .schema import validate
from .stage2_evaluation import (
    RationaleAuditCase,
    RationaleAuditManifest,
    RationaleAuditRecord,
    RationaleStratum,
)


EVIDENCE_SUPPORT_RUBRIC = {
    "version": 1,
    "question": "Is every material claim in the rationale supported by the frozen evidence?",
    "true_when": (
        "Every material topic, method, result, population, and relevance claim in the "
        "rationale is directly supported by the supplied frozen evidence."
    ),
    "false_when": (
        "Any material claim is unsupported, contradicted, or cannot be checked from the "
        "supplied frozen evidence."
    ),
    "instruction": "Judge only the frozen evidence; do not fill gaps from outside knowledge.",
}

SEVERE_FABRICATION_RUBRIC = {
    "version": 1,
    "question": "Does the rationale contain a severe fabrication?",
    "true_when": (
        "It invents or contradicts a material study fact, method, result, population, "
        "comparison, or conclusion that could change the relevance judgment."
    ),
    "false_when": (
        "No material invented or contradictory claim is present. Minor wording imprecision "
        "that cannot change the relevance judgment is not severe fabrication."
    ),
    "instruction": "Judge only the frozen evidence; do not fill gaps from outside knowledge.",
}

EVIDENCE_SUPPORT_RUBRIC_HASH = content_hash(EVIDENCE_SUPPORT_RUBRIC)
SEVERE_FABRICATION_RUBRIC_HASH = content_hash(SEVERE_FABRICATION_RUBRIC)


@dataclass(frozen=True, slots=True)
class RationaleAuditExample:
    """One pre-stratified model rationale and the evidence a reviewer may use."""

    pair_id: str
    stratum: RationaleStratum
    language: str
    rationale_artifact_hash: str
    evidence: str
    rationale: str

    def __post_init__(self) -> None:
        if not isinstance(self.stratum, RationaleStratum):
            object.__setattr__(self, "stratum", RationaleStratum(self.stratum))
        if not all((self.pair_id, self.language, self.rationale_artifact_hash, self.evidence, self.rationale)):
            raise ValueError("rationale audit examples require frozen identity, evidence, and rationale")

    def case(self) -> RationaleAuditCase:
        return RationaleAuditCase(
            self.pair_id, self.stratum, self.language, self.rationale_artifact_hash
        )

    def worklist_row(self) -> dict[str, Any]:
        row = {
            "pair_id": self.pair_id,
            "stratum": self.stratum.value,
            "language": self.language,
            "rationale_artifact_hash": self.rationale_artifact_hash,
            "evidence": self.evidence,
            "rationale": self.rationale,
            "evidence_supported": None,
            "severe_fabrication": None,
        }
        row["content_hash"] = _worklist_row_content_hash(row)
        return row


@dataclass(frozen=True, slots=True)
class FrozenRationaleAudit:
    """A manifest plus the editable, deliberately unlabelled human worklist."""

    manifest: RationaleAuditManifest
    worklist: Mapping[str, Any]


def freeze_rationale_audit(
    examples: Sequence[RationaleAuditExample],
    *,
    corpus_hash: str,
    model_lock_hash: str,
    reviewer_id: str,
) -> FrozenRationaleAudit:
    """Freeze the supplied stratified examples before any human labels exist."""

    if not reviewer_id.strip():
        raise ValueError("rationale audit reviewer_id is required")
    manifest = RationaleAuditManifest(
        version=1,
        cases=tuple(example.case() for example in examples),
        corpus_hash=corpus_hash,
        model_lock_hash=model_lock_hash,
        evidence_rubric_hash=EVIDENCE_SUPPORT_RUBRIC_HASH,
        fabrication_rubric_hash=SEVERE_FABRICATION_RUBRIC_HASH,
    )
    validate(manifest.document(), "stage2-rationale-audit-manifest.schema.json")
    worklist = {
        "schema_version": "1",
        "kind": "stage2_human_rationale_audit_worklist",
        "manifest_hash": manifest.hash(),
        "reviewer_id": reviewer_id,
        "evidence_support_rubric_hash": EVIDENCE_SUPPORT_RUBRIC_HASH,
        "severe_fabrication_rubric_hash": SEVERE_FABRICATION_RUBRIC_HASH,
        "evidence_support_rubric": EVIDENCE_SUPPORT_RUBRIC,
        "severe_fabrication_rubric": SEVERE_FABRICATION_RUBRIC,
        "rows": [example.worklist_row() for example in examples],
    }
    return FrozenRationaleAudit(manifest, worklist)


def import_completed_rationale_audit(
    worklist: Mapping[str, Any], *, manifest: RationaleAuditManifest
) -> tuple[RationaleAuditRecord, ...]:
    """Import explicit human labels; blank or non-boolean labels are rejected."""

    if worklist.get("kind") != "stage2_human_rationale_audit_worklist":
        raise ValueError("not a Stage 2 human rationale audit worklist")
    if worklist.get("manifest_hash") != manifest.hash():
        raise ValueError("rationale audit worklist does not bind the frozen manifest")
    if (
        worklist.get("evidence_support_rubric_hash") != manifest.evidence_rubric_hash
        or worklist.get("severe_fabrication_rubric_hash") != manifest.fabrication_rubric_hash
        or worklist.get("evidence_support_rubric") != EVIDENCE_SUPPORT_RUBRIC
        or worklist.get("severe_fabrication_rubric") != SEVERE_FABRICATION_RUBRIC
    ):
        raise ValueError("rationale audit worklist rubrics do not match the frozen manifest")
    rows = worklist.get("rows")
    if not isinstance(rows, list):
        raise ValueError("rationale audit worklist rows must be a list")
    expected_cases = {case.pair_id: case for case in manifest.cases}
    if {row.get("pair_id") for row in rows if isinstance(row, Mapping)} != set(expected_cases) or len(rows) != len(expected_cases):
        raise ValueError("rationale audit worklist must exactly cover the frozen manifest")
    records = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("rationale audit worklist rows must be objects")
        case = expected_cases[row["pair_id"]]
        frozen_fields = ("stratum", "language", "rationale_artifact_hash")
        expected_values = (case.stratum.value, case.language, case.rationale_artifact_hash)
        if tuple(row.get(field) for field in frozen_fields) != expected_values:
            raise ValueError("rationale audit worklist row changed frozen provenance")
        if row.get("content_hash") != _worklist_row_content_hash(row):
            raise ValueError("rationale audit worklist row content drifted")
        evidence_supported = row.get("evidence_supported")
        severe_fabrication = row.get("severe_fabrication")
        if type(evidence_supported) is not bool or type(severe_fabrication) is not bool:
            raise ValueError("rationale audit worklist has unfilled human labels")
        records.append(RationaleAuditRecord(
            row["pair_id"], manifest.hash(), evidence_supported, severe_fabrication
        ))
    return tuple(records)


def rationale_audit_records_document(
    records: Sequence[RationaleAuditRecord],
) -> dict[str, Any]:
    """Return the existing ``stage2-rationale-audit-records`` schema shape."""

    document = {
        "schema_version": "1",
        "kind": "stage2_rationale_audit_records",
        "records": [record.document() for record in records],
    }
    validate(document, "stage2-rationale-audit-records.schema.json")
    return document


def write_rationale_audit_artifacts(
    frozen: FrozenRationaleAudit,
    records: Sequence[RationaleAuditRecord],
    *,
    manifest_path: Path,
    records_path: Path,
) -> None:
    """Publish no-replace artifacts, with records visible before the manifest."""

    if manifest_path == records_path:
        raise ValueError("rationale audit manifest and records paths must differ")
    if manifest_path.exists() or records_path.exists():
        raise FileExistsError("rationale audit output already exists")
    manifest_document = frozen.manifest.document()
    validate(manifest_document, "stage2-rationale-audit-manifest.schema.json")
    records_document = rationale_audit_records_document(records)
    _write_json_no_replace(records_path, records_document)
    _write_json_no_replace(manifest_path, manifest_document)


def write_rationale_worklist_no_replace(path: Path, worklist: Mapping[str, Any]) -> None:
    """Publish an editable human worklist without replacing an earlier copy."""

    _write_json_no_replace(path, worklist)


def _worklist_row_content_hash(row: Mapping[str, Any]) -> str:
    return content_hash({
        field: row[field]
        for field in (
            "pair_id", "stratum", "language", "rationale_artifact_hash", "evidence", "rationale"
        )
    })


def _write_json_no_replace(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(document, handle, sort_keys=True, indent=2)
        handle.write("\n")
