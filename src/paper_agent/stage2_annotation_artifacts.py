"""Private Stage 2 annotation-ledger loading and reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from tempfile import mkstemp
from typing import Any, Collection, Mapping

from .canonical import content_hash
from .schema import SchemaValidationError, validate
from .stage2_evaluation import (
    Adjudication,
    Annotation,
    AnnotationSummary,
    GoldLabelStore,
    GoldManifest,
    complete_double_annotation,
    quadratic_weighted_kappa,
)
from .stage2_sampling import PrivateCorpusSnapshot, PrivateSamplingAnnotations


class AnnotationLedgerArtifactError(ValueError):
    """A private annotation ledger is malformed or does not bind its manifest."""


STAGE2_ANNOTATION_RUBRIC = {
    "version": 1,
    "positive_labels": [2, 3],
    "labels": {
        "0": "Clearly irrelevant / 明确无关",
        "1": "Weakly related, background-only, or insufficient evidence / 弱相关、仅背景提及或证据不足",
        "2": "Directly related and should be retained / 与主题直接相关，应保留",
        "3": "Core paper for the topic and must be retained / 主题核心论文，必须保留",
    },
}
STAGE2_ANNOTATION_RUBRIC_HASH = content_hash(STAGE2_ANNOTATION_RUBRIC)


@dataclass(frozen=True, slots=True)
class AnnotationLedger:
    """The recomputed annotation result held by the evaluator."""

    summary: AnnotationSummary
    gold_labels: GoldLabelStore


def make_human_annotation_worklist(
    manifest: GoldManifest,
    snapshot: PrivateCorpusSnapshot,
    *,
    participant_id: str,
    role: str = "annotator",
    pair_ids: Collection[str] | None = None,
    annotation_input_hash: str | None = None,
) -> dict[str, Any]:
    """Create a blind worklist from the selected manifest and private text."""

    manifest.validate_sampling_structure()
    selected_ids = {pair.pair_id for pair in manifest.pairs} if pair_ids is None else set(pair_ids)
    worklist = {
        "schema_version": "1",
        "kind": "stage2_human_annotation_worklist",
        "gold_manifest_hash": manifest.hash(),
        "rubric_version": 1,
        "rubric_hash": STAGE2_ANNOTATION_RUBRIC_HASH,
        "role": role,
        "participant_id": participant_id,
        "annotation_input_hash": annotation_input_hash,
        "rubric": {
            "version": 1,
            "positive_labels": [2, 3],
            "labels": dict(STAGE2_ANNOTATION_RUBRIC["labels"]),
        },
        "rows": _blind_rows(manifest, snapshot, selected_ids),
    }
    _validate_human_worklist(worklist)
    return worklist


def human_annotation_worklist_from_document(
    document: Mapping[str, Any],
    *,
    manifest: GoldManifest,
    snapshot: PrivateCorpusSnapshot,
    role: str,
    require_complete: bool = False,
    expected_pair_ids: Collection[str] | None = None,
) -> dict[str, Any]:
    """Load an editable worklist and rebind its visible text to the snapshot."""

    _validate_human_worklist(document)
    manifest.validate_sampling_structure()
    if (
        document["gold_manifest_hash"] != manifest.hash()
        or document["rubric_version"] != 1
        or document["rubric_hash"] != STAGE2_ANNOTATION_RUBRIC_HASH
        or document["rubric"] != STAGE2_ANNOTATION_RUBRIC
    ):
        raise AnnotationLedgerArtifactError("human annotation worklist binding does not match")
    if document["role"] != role:
        raise AnnotationLedgerArtifactError(f"expected a {role} worklist")

    if role == "annotator":
        expected_ids = {pair.pair_id for pair in manifest.pairs}
    elif expected_pair_ids is not None:
        expected_ids = set(expected_pair_ids)
    else:
        expected_ids = {row["pair_id"] for row in document["rows"]}
    actual_ids = [row["pair_id"] for row in document["rows"]]
    if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != expected_ids:
        raise AnnotationLedgerArtifactError(
            f"{role} worklist must exactly cover its expected pairs"
        )
    visible = {
        row["pair_id"]: (row["topic"], row["title"], row["abstract"], row["language"])
        for row in document["rows"]
    }
    expected_visible = {
        row["pair_id"]: (row["topic"], row["title"], row["abstract"], row["language"])
        for row in _blind_rows(manifest, snapshot, expected_ids)
    }
    if visible != expected_visible:
        raise AnnotationLedgerArtifactError(
            "human annotation worklist text does not match the private snapshot"
        )
    if require_complete and any(row["label"] is None for row in document["rows"]):
        raise AnnotationLedgerArtifactError(f"{role} worklist has unfilled labels")
    return dict(document)


def load_human_annotation_worklist(
    path: Path,
    *,
    manifest: GoldManifest,
    snapshot: PrivateCorpusSnapshot,
    role: str,
    require_complete: bool = False,
    expected_pair_ids: Collection[str] | None = None,
) -> dict[str, Any]:
    document = _read_json_object(path, "human annotation worklist")
    return human_annotation_worklist_from_document(
        document,
        manifest=manifest,
        snapshot=snapshot,
        role=role,
        require_complete=require_complete,
        expected_pair_ids=expected_pair_ids,
    )


def write_human_annotation_worklist(path: Path, worklist: Mapping[str, Any]) -> None:
    _write_json_no_replace(path, worklist)


def annotation_agreement(
    manifest: GoldManifest,
    first: Mapping[str, Any],
    second: Mapping[str, Any],
) -> tuple[float, frozenset[str]]:
    """Measure the completed independent labels before any adjudication."""

    expected_ids = {pair.pair_id for pair in manifest.pairs}
    if (
        first["role"] != "annotator"
        or second["role"] != "annotator"
        or first["participant_id"] == second["participant_id"]
        or {row["pair_id"] for row in first["rows"]} != expected_ids
        or {row["pair_id"] for row in second["rows"]} != expected_ids
    ):
        raise AnnotationLedgerArtifactError(
            "annotation requires two distinct annotators covering every manifest pair"
        )
    first_labels = _completed_labels(first, "annotator")
    second_labels = _completed_labels(second, "annotator")
    ordered_ids = sorted(expected_ids)
    left = [first_labels[pair_id] for pair_id in ordered_ids]
    right = [second_labels[pair_id] for pair_id in ordered_ids]
    kappa = quadratic_weighted_kappa(left, right)
    if kappa < 0.75:
        raise AnnotationLedgerArtifactError(
            f"pre-adjudication quadratic weighted kappa {kappa:.3f} is below 0.75"
        )
    disagreements = frozenset(
        pair_id
        for pair_id in ordered_ids
        if first_labels[pair_id] != second_labels[pair_id]
    )
    return kappa, disagreements


def make_adjudication_worklist(
    manifest: GoldManifest,
    snapshot: PrivateCorpusSnapshot,
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    participant_id: str,
) -> tuple[dict[str, Any], float]:
    kappa, disagreements = annotation_agreement(manifest, first, second)
    if participant_id in {first["participant_id"], second["participant_id"]}:
        raise AnnotationLedgerArtifactError("adjudicator must be distinct from both annotators")
    return (
        make_human_annotation_worklist(
            manifest,
            snapshot,
            participant_id=participant_id,
            role="adjudicator",
            pair_ids=disagreements,
            annotation_input_hash=annotation_inputs_hash(first, second),
        ),
        kappa,
    )


def assemble_annotation_ledger(
    manifest: GoldManifest,
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    adjudication: Mapping[str, Any],
    curated_annotations: PrivateSamplingAnnotations,
    *,
    sampling_provenance_hash: str,
) -> tuple[dict[str, Any], AnnotationLedger]:
    """Assemble and fully validate the private ledger from human handoffs."""

    _, disagreements = annotation_agreement(manifest, first, second)
    input_hash = annotation_inputs_hash(first, second)
    if (
        adjudication["role"] != "adjudicator"
        or adjudication["participant_id"]
        in {first["participant_id"], second["participant_id"]}
        or adjudication["annotation_input_hash"] != input_hash
        or {row["pair_id"] for row in adjudication["rows"]} != set(disagreements)
    ):
        raise AnnotationLedgerArtifactError(
            "adjudication must be completed by a distinct third person for every disagreement"
        )
    annotators = sorted((first, second), key=lambda item: item["participant_id"])
    labels_by_annotator = {
        item["participant_id"]: _completed_labels(item, "annotator") for item in annotators
    }
    adjudicated_labels = _completed_labels(adjudication, "adjudicator")
    final_labels = {
        pair_id: adjudicated_labels.get(
            pair_id,
            labels_by_annotator[annotators[0]["participant_id"]][pair_id],
        )
        for pair_id in labels_by_annotator[annotators[0]["participant_id"]]
    }
    hard_negative_pair_ids: list[str] = []
    hard_positive_pair_ids: list[str] = []
    for pair in manifest.pairs:
        candidate = curated_annotations.by_key.get((pair.topic, pair.paper_id))
        if candidate is not None and candidate.hard_negative and final_labels[pair.pair_id] < 2:
            hard_negative_pair_ids.append(pair.pair_id)
        if candidate is not None and candidate.hard_positive and final_labels[pair.pair_id] == 3:
            hard_positive_pair_ids.append(pair.pair_id)
    document = {
        "schema_version": "1",
        "gold_manifest_hash": manifest.hash(),
        "rubric_version": 1,
        "rubric_hash": STAGE2_ANNOTATION_RUBRIC_HASH,
        "sampling_provenance_hash": sampling_provenance_hash,
        "annotation_input_hash": input_hash,
        "annotator_ids": [item["participant_id"] for item in annotators],
        "adjudicator_id": adjudication["participant_id"],
        "annotations": [
            {
                "pair_id": pair_id,
                "annotator_id": annotator["participant_id"],
                "label": labels_by_annotator[annotator["participant_id"]][pair_id],
            }
            for pair_id in sorted(final_labels)
            for annotator in annotators
        ],
        "adjudications": [
            {
                "pair_id": pair_id,
                "adjudicator_id": adjudication["participant_id"],
                "label": adjudicated_labels[pair_id],
            }
            for pair_id in sorted(adjudicated_labels)
        ],
        "hard_negative_pair_ids": sorted(hard_negative_pair_ids),
        "hard_positive_pair_ids": sorted(hard_positive_pair_ids),
    }
    return document, annotation_ledger_from_document(document, manifest=manifest)


def write_annotation_ledger(path: Path, document: Mapping[str, Any]) -> None:
    _write_json_no_replace(path, document)


def annotation_ledger_from_document(
    document: Mapping[str, Any], *, manifest: GoldManifest
) -> AnnotationLedger:
    """Validate a private ledger and reconstruct its authoritative labels."""

    _validate(document)
    if document["gold_manifest_hash"] != manifest.hash():
        raise AnnotationLedgerArtifactError("annotation ledger does not bind the supplied gold manifest")
    if (
        document["rubric_version"] != 1
        or document["rubric_hash"] != STAGE2_ANNOTATION_RUBRIC_HASH
    ):
        raise AnnotationLedgerArtifactError("annotation ledger does not use the frozen rubric")
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

    return annotation_ledger_from_document(
        _read_json_object(path, "annotation ledger"), manifest=manifest
    )


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
    _write_json_no_replace(path, document)


def _read_json_object(path: Path, name: str) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AnnotationLedgerArtifactError(f"cannot read {name}: {path}") from error
    if not isinstance(document, dict):
        raise AnnotationLedgerArtifactError(f"{name} must be a JSON object")
    return document


def _blind_rows(
    manifest: GoldManifest,
    snapshot: PrivateCorpusSnapshot,
    pair_ids: Collection[str],
) -> list[dict[str, Any]]:
    if snapshot.corpus_hash != manifest.corpus_hash:
        raise AnnotationLedgerArtifactError(
            "private snapshot does not bind the supplied gold manifest corpus"
        )
    manifest_by_id = {pair.pair_id: pair for pair in manifest.pairs}
    if not set(pair_ids) <= set(manifest_by_id):
        raise AnnotationLedgerArtifactError("annotation worklist contains pairs outside the manifest")
    snapshot_by_key = {paper.key: paper for paper in snapshot.papers}
    rows = []
    for pair_id in sorted(pair_ids):
        pair = manifest_by_id[pair_id]
        paper = snapshot_by_key.get((pair.topic, pair.paper_id))
        if paper is None:
            raise AnnotationLedgerArtifactError(
                "private snapshot does not contain every manifest pair"
            )
        rows.append({
            "pair_id": pair_id,
            "topic": pair.topic,
            "title": paper.title,
            "abstract": paper.abstract,
            "language": paper.language,
            "label": None,
        })
    return rows


def _completed_labels(worklist: Mapping[str, Any], role: str) -> dict[str, int]:
    labels = {row["pair_id"]: row["label"] for row in worklist["rows"]}
    if any(label is None for label in labels.values()):
        raise AnnotationLedgerArtifactError(f"{role} worklist has unfilled labels")
    return {pair_id: label for pair_id, label in labels.items() if label is not None}


def annotation_inputs_hash(
    first: Mapping[str, Any], second: Mapping[str, Any]
) -> str:
    ordered = sorted((first, second), key=lambda item: item["participant_id"])
    return content_hash(ordered)


def _write_json_no_replace(path: Path, document: Mapping[str, Any]) -> None:
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


def _validate_human_worklist(document: Mapping[str, Any]) -> None:
    try:
        validate(document, "stage2-human-annotation-worklist.schema.json")
    except SchemaValidationError as error:
        raise AnnotationLedgerArtifactError(str(error)) from error
