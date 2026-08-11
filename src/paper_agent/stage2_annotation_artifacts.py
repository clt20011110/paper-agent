"""Private Stage 2 annotation-ledger loading and reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from tempfile import mkstemp
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


def private_gold_labels_document(
    ledger: AnnotationLedger, *, manifest: GoldManifest
) -> dict[str, Any]:
    """Derive the sealed promotion-label artifact from a verified ledger."""

    if (
        ledger.summary.annotation_artifact_hash
        != ledger.gold_labels.annotation_artifact_hash
        or ledger.summary.labels != ledger.gold_labels.labels
    ):
        raise AnnotationLedgerArtifactError(
            "annotation ledger summary and gold labels do not match"
        )
    manifest.validate(ledger.gold_labels)
    document = {
        "schema_version": "1",
        "gold_manifest_hash": manifest.hash(),
        "annotation_artifact_hash": ledger.gold_labels.annotation_artifact_hash,
        "labels": [
            {"pair_id": pair_id, "label": ledger.gold_labels.labels[pair_id]}
            for pair_id in sorted(ledger.gold_labels.labels)
        ],
        "hard_negative_pair_ids": sorted(ledger.gold_labels.hard_negative_pair_ids),
        "hard_positive_pair_ids": sorted(ledger.gold_labels.hard_positive_pair_ids),
    }
    _validate_private_gold_labels(document)
    return document


def write_private_gold_labels(
    path: Path, ledger: AnnotationLedger, *, manifest: GoldManifest
) -> None:
    """Atomically create one private gold-label artifact without replacement."""

    document = private_gold_labels_document(ledger, manifest=manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(
                json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
                + b"\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _validate(document: Mapping[str, Any]) -> None:
    try:
        validate(document, "stage2-annotation-ledger.schema.json")
    except SchemaValidationError as error:
        raise AnnotationLedgerArtifactError(str(error)) from error


def _validate_private_gold_labels(document: Mapping[str, Any]) -> None:
    try:
        validate(document, "stage2-private-gold-labels.schema.json")
    except SchemaValidationError as error:
        raise AnnotationLedgerArtifactError(str(error)) from error
