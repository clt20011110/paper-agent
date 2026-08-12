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
from .stage2_benchmark_inputs import benchmark_corpus_hash, benchmark_papers_from_document
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


_SOURCE_LEDGER_FIELDS = frozenset({
    "schema_version", "kind", "candidate", "benchmark_papers_sha256",
    "corpus_hash", "query_metadata_sha256", "records",
})
_SOURCE_CANDIDATE_FIELDS = frozenset({
    "candidate_id", "bundle_sha256", "release_hash", "adjudicator_model_id",
    "adjudicator_model_lock_hash", "prompt_version", "response_schema",
})
_QUERY_METADATA_FIELDS = frozenset({
    "schema_version", "kind", "benchmark_papers_sha256", "primary_languages",
    "assignments",
})
_QUERY_ASSIGNMENT_FIELDS = frozenset({
    "paper_id", "language", "topic", "query_version", "query",
})
_LEDGER_RECORD_FIELDS = frozenset({
    "paper_id", "decision", "score", "rationale", "evidence_fields",
})
_DERIVED_DOCUMENT_FIELDS = frozenset({
    "schema_version", "kind", "corpus_hash", "model_lock_hash",
    "candidate_bundle_sha256", "source_ledger_sha256", "query_metadata_sha256",
    "examples",
})


def derive_rationale_audit_examples(
    source_ledger: object,
    *,
    source_ledger_sha256: str,
    candidate: Any,
    candidate_bundle_sha256: str,
    benchmark_papers_document: object,
    benchmark_papers_sha256: str,
    query_metadata: object,
    query_metadata_sha256: str,
) -> dict[str, Any]:
    """Deterministically select review cases from a bound Qwen response ledger.

    The ledger is deliberately a strict, text-bearing artifact: the model's
    rationale is copied verbatim, while the reviewer evidence is rendered only
    from the separately frozen benchmark paper fields named by the ledger.
    """

    validate(source_ledger, "stage2-rationale-source-ledger.schema.json")
    validate(query_metadata, "stage2-rationale-query-metadata.schema.json")
    validate(benchmark_papers_document, "stage2-benchmark-papers.schema.json")
    if not isinstance(source_ledger, Mapping) or not isinstance(query_metadata, Mapping):
        raise ValueError("rationale source inputs must be JSON objects")
    if set(source_ledger) != _SOURCE_LEDGER_FIELDS:
        raise ValueError("rationale source ledger has an unsupported shape")
    if set(query_metadata) != _QUERY_METADATA_FIELDS:
        raise ValueError("rationale query metadata has an unsupported shape")
    if source_ledger["benchmark_papers_sha256"] != benchmark_papers_sha256:
        raise ValueError("rationale source ledger does not bind the benchmark papers bytes")
    if source_ledger["query_metadata_sha256"] != query_metadata_sha256:
        raise ValueError("rationale source ledger does not bind the query metadata bytes")
    if query_metadata["benchmark_papers_sha256"] != benchmark_papers_sha256:
        raise ValueError("rationale query metadata does not bind the benchmark papers bytes")

    papers = benchmark_papers_from_document(benchmark_papers_document)
    corpus_hash = benchmark_corpus_hash(papers)
    if source_ledger["corpus_hash"] != corpus_hash:
        raise ValueError("rationale source ledger corpus hash does not match benchmark papers")
    _validate_candidate_binding(source_ledger["candidate"], candidate, candidate_bundle_sha256)

    papers_by_id = {paper.paper_id: paper for paper in papers}
    if len(papers_by_id) != len(papers):
        raise ValueError("benchmark papers must have unique paper_id values")
    assignments = _query_assignments(query_metadata, papers_by_id)
    source_records = _ledger_records(source_ledger, assignments, papers_by_id)
    examples: list[dict[str, Any]] = []
    for language in query_metadata["primary_languages"]:
        for stratum, decision in (
            (RationaleStratum.RELEVANT, "relevant"),
            (RationaleStratum.BOUNDARY, "needs_review"),
        ):
            candidates = sorted(
                (
                    record for record in source_records
                    if record["decision"] == decision
                    and assignments[record["paper_id"]]["language"] == language
                ),
                key=lambda record: (record["paper_id"], record["rationale"]),
            )
            if len(candidates) < 25:
                raise ValueError(
                    f"rationale source ledger needs at least 25 {stratum.value} "
                    f"records for primary language {language}"
                )
            for record in candidates[:25]:
                paper = papers_by_id[record["paper_id"]]
                examples.append({
                    "pair_id": record["paper_id"],
                    "stratum": stratum.value,
                    "language": language,
                    "rationale_artifact_hash": source_ledger_sha256,
                    "evidence": _render_evidence(paper, record["evidence_fields"]),
                    "rationale": record["rationale"],
                })

    document = {
        "schema_version": "2",
        "kind": "stage2_rationale_audit_derived_examples",
        "corpus_hash": corpus_hash,
        "model_lock_hash": source_ledger["candidate"]["adjudicator_model_lock_hash"],
        "candidate_bundle_sha256": candidate_bundle_sha256,
        "source_ledger_sha256": source_ledger_sha256,
        "query_metadata_sha256": query_metadata_sha256,
        "examples": examples,
    }
    validate(document, "stage2-rationale-derived-examples.schema.json")
    return document


def qwen_adjudication_ledger_document(
    decisions: Sequence[Any],
    *,
    candidate: Any,
    candidate_bundle_sha256: str,
    benchmark_papers_document: object,
    benchmark_papers_sha256: str,
    query_metadata_sha256: str,
) -> dict[str, Any]:
    """Freeze actual Qwen ``Stage2Decision`` outputs as a source ledger.

    This is the production boundary for rationale text.  It accepts the typed
    decisions emitted by ``Stage2Pipeline`` rather than author-supplied
    rationale dictionaries, and refuses non-Qwen or incomplete outcomes.
    """

    validate(benchmark_papers_document, "stage2-benchmark-papers.schema.json")
    papers = benchmark_papers_from_document(benchmark_papers_document)
    paper_ids = {paper.paper_id for paper in papers}
    profile = candidate.profile
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for decision in decisions:
        paper_id = getattr(decision, "paper_id", None)
        route = getattr(getattr(decision, "route", None), "value", None)
        score = getattr(decision, "adjudicator_score", None)
        rationale = getattr(decision, "rationale", None)
        evidence_fields = getattr(decision, "evidence_fields", None)
        if (
            getattr(decision, "adjudicated", None) is not True
            or paper_id not in paper_ids
            or paper_id in seen
            or route not in {"relevant", "irrelevant", "needs_review"}
            or type(score) not in {int, float}
            or not isinstance(rationale, str)
            or not rationale.strip()
            or not isinstance(evidence_fields, tuple)
        ):
            raise ValueError("rationale ledger requires complete Qwen Stage2Decision outputs")
        seen.add(paper_id)
        records.append({
            "paper_id": paper_id,
            "decision": route,
            "score": float(score),
            "rationale": rationale,
            "evidence_fields": list(evidence_fields),
        })
    if len(records) < 100:
        raise ValueError("rationale ledger requires at least 100 Qwen decisions")
    document = {
        "schema_version": "1",
        "kind": "stage2_qwen_adjudication_ledger",
        "candidate": {
            "candidate_id": candidate.profile_name,
            "bundle_sha256": candidate_bundle_sha256,
            "release_hash": candidate.release_hash,
            "adjudicator_model_id": profile.adjudicator_model_id,
            "adjudicator_model_lock_hash": profile.adjudicator_lock_hash,
            "prompt_version": profile.prompt_version,
            "response_schema": profile.schema_version,
        },
        "benchmark_papers_sha256": benchmark_papers_sha256,
        "corpus_hash": benchmark_corpus_hash(papers),
        "query_metadata_sha256": query_metadata_sha256,
        "records": records,
    }
    validate(document, "stage2-rationale-source-ledger.schema.json")
    return document


def write_qwen_adjudication_ledger_no_replace(path: Path, document: Mapping[str, Any]) -> None:
    """Publish typed Qwen outcomes as the immutable rationale source artifact."""

    validate(document, "stage2-rationale-source-ledger.schema.json")
    _write_json_no_replace(path, document)


def write_derived_rationale_examples_no_replace(path: Path, document: Mapping[str, Any]) -> None:
    """Write a schema-validated deterministic rationale source without replacement."""

    validate(document, "stage2-rationale-derived-examples.schema.json")
    _write_json_no_replace(path, document)


def _validate_candidate_binding(
    binding: object, candidate: Any, candidate_bundle_sha256: str
) -> None:
    if not isinstance(binding, Mapping) or set(binding) != _SOURCE_CANDIDATE_FIELDS:
        raise ValueError("rationale source ledger candidate binding has an unsupported shape")
    profile = candidate.profile
    expected = {
        "candidate_id": candidate.profile_name,
        "bundle_sha256": candidate_bundle_sha256,
        "release_hash": candidate.release_hash,
        "adjudicator_model_id": profile.adjudicator_model_id,
        "adjudicator_model_lock_hash": profile.adjudicator_lock_hash,
        "prompt_version": profile.prompt_version,
        "response_schema": profile.schema_version,
    }
    if binding != expected:
        raise ValueError("rationale source ledger is not bound to the frozen Stage 2 candidate")


def _query_assignments(
    metadata: Mapping[str, Any], papers_by_id: Mapping[str, Any]
) -> Mapping[str, Mapping[str, Any]]:
    assignments = metadata["assignments"]
    if not isinstance(assignments, list):
        raise ValueError("rationale query metadata assignments must be a list")
    by_paper: dict[str, Mapping[str, Any]] = {}
    for assignment in assignments:
        if not isinstance(assignment, Mapping) or set(assignment) != _QUERY_ASSIGNMENT_FIELDS:
            raise ValueError("rationale query metadata assignment has an unsupported shape")
        paper_id = assignment["paper_id"]
        if paper_id in by_paper or paper_id not in papers_by_id:
            raise ValueError("rationale query metadata must assign each benchmark paper at most once")
        by_paper[paper_id] = assignment
    primary_languages = metadata["primary_languages"]
    if not isinstance(primary_languages, list) or len(primary_languages) < 2:
        raise ValueError("rationale query metadata needs at least two primary languages")
    return by_paper


def _ledger_records(
    ledger: Mapping[str, Any],
    assignments: Mapping[str, Mapping[str, Any]],
    papers_by_id: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    records = ledger["records"]
    if not isinstance(records, list):
        raise ValueError("rationale source ledger records must be a list")
    by_paper: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping) or set(record) != _LEDGER_RECORD_FIELDS:
            raise ValueError("rationale source ledger record has an unsupported shape")
        paper_id = record["paper_id"]
        if paper_id in by_paper or paper_id not in papers_by_id or paper_id not in assignments:
            raise ValueError("rationale source ledger records need one bound paper and query assignment")
        by_paper[paper_id] = record
    return tuple(by_paper.values())


def _render_evidence(paper: Any, evidence_fields: object) -> str:
    if not isinstance(evidence_fields, list) or not evidence_fields:
        raise ValueError("rationale source ledger evidence_fields must be a non-empty list")
    fields = tuple(evidence_fields)
    if len(set(fields)) != len(fields) or any(field not in {"title", "abstract", "keywords"} for field in fields):
        raise ValueError("rationale source ledger evidence_fields are invalid")
    parts: list[str] = []
    for field in fields:
        if field == "title":
            if not paper.title:
                raise ValueError("rationale source ledger requested an empty title")
            parts.append(f"title: {paper.title}")
        elif field == "abstract":
            if not paper.abstract or not paper.abstract.strip():
                raise ValueError("rationale source ledger requested a missing abstract")
            parts.append(f"abstract: {paper.abstract}")
        else:
            if not paper.keywords:
                raise ValueError("rationale source ledger requested missing keywords")
            parts.append("keywords: " + ", ".join(paper.keywords))
    return "\n".join(parts)


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


def rationale_audit_examples_from_document(
    document: object,
) -> tuple[tuple[RationaleAuditExample, ...], str, str]:
    """Load the explicit, already-selected examples used to freeze an audit."""

    if isinstance(document, Mapping) and document.get("kind") == "stage2_rationale_audit_derived_examples":
        validate(document, "stage2-rationale-derived-examples.schema.json")
        if set(document) != _DERIVED_DOCUMENT_FIELDS or document.get("schema_version") != "2":
            raise ValueError("rationale audit derived examples have an unsupported shape")
        examples_document = document
    else:
        examples_document = document
    if not isinstance(examples_document, Mapping) or set(examples_document) not in ({
        "schema_version", "kind", "corpus_hash", "model_lock_hash", "examples",
    }, _DERIVED_DOCUMENT_FIELDS):
        raise ValueError("rationale audit examples have an unsupported shape")
    if (
        (examples_document["schema_version"] == "1" and examples_document["kind"] != "stage2_rationale_audit_examples")
        or (examples_document["schema_version"] == "2" and examples_document["kind"] != "stage2_rationale_audit_derived_examples")
        or examples_document["schema_version"] not in {"1", "2"}
        or not isinstance(examples_document["examples"], list)
    ):
        raise ValueError("not a Stage 2 rationale audit examples document")
    examples: list[RationaleAuditExample] = []
    required = {
        "pair_id", "stratum", "language", "rationale_artifact_hash",
        "evidence", "rationale",
    }
    for row in examples_document["examples"]:
        if not isinstance(row, Mapping) or set(row) != required:
            raise ValueError("rationale audit example has an unsupported shape")
        if (
            examples_document["schema_version"] == "2"
            and row["rationale_artifact_hash"] != examples_document["source_ledger_sha256"]
        ):
            raise ValueError("derived rationale example is not bound to its source ledger")
        examples.append(RationaleAuditExample(
            pair_id=row["pair_id"],
            stratum=RationaleStratum(row["stratum"]),
            language=row["language"],
            rationale_artifact_hash=row["rationale_artifact_hash"],
            evidence=row["evidence"],
            rationale=row["rationale"],
        ))
    return tuple(examples), examples_document["corpus_hash"], examples_document["model_lock_hash"]


def rationale_audit_manifest_from_document(document: object) -> RationaleAuditManifest:
    """Load one schema-validated frozen rationale audit manifest."""

    validate(document, "stage2-rationale-audit-manifest.schema.json")
    if not isinstance(document, Mapping):
        raise ValueError("rationale audit manifest must be an object")
    return RationaleAuditManifest(
        version=document["version"],
        cases=tuple(
            RationaleAuditCase(row[0], RationaleStratum(row[1]), row[2], row[3])
            for row in document["cases"]
        ),
        corpus_hash=document["corpus_hash"],
        model_lock_hash=document["model_lock_hash"],
        evidence_rubric_hash=document["evidence_rubric_hash"],
        fabrication_rubric_hash=document["fabrication_rubric_hash"],
    )


def load_rationale_audit_manifest(path: Path) -> RationaleAuditManifest:
    return rationale_audit_manifest_from_document(json.loads(path.read_text(encoding="utf-8")))


def load_rationale_worklist(path: Path) -> Mapping[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise ValueError("rationale audit worklist must be an object")
    return document


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

    if set(worklist) != {
        "schema_version", "kind", "manifest_hash", "reviewer_id",
        "evidence_support_rubric_hash", "severe_fabrication_rubric_hash",
        "evidence_support_rubric", "severe_fabrication_rubric", "rows",
    } or worklist.get("schema_version") != "1" or worklist.get("kind") != "stage2_human_rationale_audit_worklist":
        raise ValueError("not a Stage 2 human rationale audit worklist")
    if not isinstance(worklist.get("reviewer_id"), str) or not worklist["reviewer_id"].strip():
        raise ValueError("rationale audit worklist reviewer_id is required")
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
    *,
    worklist_sha256: str,
) -> dict[str, Any]:
    """Return the existing ``stage2-rationale-audit-records`` schema shape."""

    document = {
        "schema_version": "1",
        "kind": "stage2_rationale_audit_records",
        "worklist_sha256": worklist_sha256,
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
    worklist_sha256: str,
) -> None:
    """Publish no-replace artifacts, with records visible before the manifest."""

    if manifest_path == records_path:
        raise ValueError("rationale audit manifest and records paths must differ")
    if manifest_path.exists() or records_path.exists():
        raise FileExistsError("rationale audit output already exists")
    manifest_document = frozen.manifest.document()
    validate(manifest_document, "stage2-rationale-audit-manifest.schema.json")
    records_document = rationale_audit_records_document(
        records, worklist_sha256=worklist_sha256
    )
    _write_json_no_replace(records_path, records_document)
    _write_json_no_replace(manifest_path, manifest_document)


def write_frozen_rationale_audit(
    frozen: FrozenRationaleAudit,
    *,
    manifest_path: Path,
    worklist_path: Path,
) -> None:
    """Publish a human worklist first and its completion-marker manifest last."""

    if manifest_path.absolute() == worklist_path.absolute():
        raise ValueError("rationale audit manifest and worklist paths must differ")
    if manifest_path.exists() or worklist_path.exists():
        raise FileExistsError("rationale audit output already exists")
    manifest_document = frozen.manifest.document()
    validate(manifest_document, "stage2-rationale-audit-manifest.schema.json")
    _write_json_no_replace(worklist_path, frozen.worklist)
    _write_json_no_replace(manifest_path, manifest_document)


def write_rationale_records_no_replace(
    path: Path,
    records: Sequence[RationaleAuditRecord],
    *,
    worklist_sha256: str,
) -> None:
    """Publish completed human audit records without replacing prior evidence."""

    _write_json_no_replace(
        path, rationale_audit_records_document(records, worklist_sha256=worklist_sha256)
    )


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
