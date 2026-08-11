"""Private Stage 2 annotation-ledger loading and reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from .schema import SchemaValidationError, validate
from .stage2_evaluation import (
    Adjudication,
    Annotation,
    AnnotationSummary,
    GoldLabelStore,
    GoldManifest,
    complete_double_annotation,
)


class AnnotationLedgerArtifactError(ValueError):
    """A private annotation ledger is malformed or does not bind its manifest."""


@dataclass(frozen=True, slots=True)
class AnnotationLedger:
    """The recomputed annotation result held by the evaluator."""

    summary: AnnotationSummary
    gold_labels: GoldLabelStore


def annotation_ledger_from_document(
    document: Mapping[str, Any], *, manifest: GoldManifest
) -> AnnotationLedger:
    """Validate a private ledger and reconstruct its authoritative labels."""

    _validate(document)
    if document["gold_manifest_hash"] != manifest.hash():
        raise AnnotationLedgerArtifactError("annotation ledger does not bind the supplied gold manifest")
    try:
        summary = complete_double_annotation(
            manifest.pairs,
            tuple(Annotation(**row) for row in document["annotations"]),
            tuple(Adjudication(**row) for row in document["adjudications"]),
            annotator_order=tuple(document["annotator_ids"]),
            adjudicator_id=document["adjudicator_id"],
            rubric_version=document["rubric_version"],
            rubric_hash=document["rubric_hash"],
        )
        gold_labels = GoldLabelStore(
            summary.labels,
            summary.annotation_artifact_hash,
            frozenset(document["hard_negative_pair_ids"]),
            frozenset(document["hard_positive_pair_ids"]),
        )
        manifest.validate(gold_labels)
    except ValueError as error:
        raise AnnotationLedgerArtifactError(f"annotation ledger is invalid: {error}") from error
    return AnnotationLedger(summary, gold_labels)


def load_annotation_ledger(path: Path, *, manifest: GoldManifest) -> AnnotationLedger:
    """Load the versioned evaluator-private annotation ledger for ``manifest``."""

    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AnnotationLedgerArtifactError(f"cannot read annotation ledger: {path}") from error
    if not isinstance(document, dict):
        raise AnnotationLedgerArtifactError("annotation ledger must be a JSON object")
    return annotation_ledger_from_document(document, manifest=manifest)


def _validate(document: Mapping[str, Any]) -> None:
    try:
        validate(document, "stage2-annotation-ledger.schema.json")
    except SchemaValidationError as error:
        raise AnnotationLedgerArtifactError(str(error)) from error
