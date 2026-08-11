from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import pytest

from paper_agent.stage2_annotation_artifacts import (
    AnnotationLedgerArtifactError,
    STAGE2_ANNOTATION_RUBRIC_HASH,
    annotation_agreement,
    annotation_inputs_hash,
    annotation_ledger_from_document,
    assemble_annotation_ledger,
    load_annotation_ledger,
    private_gold_labels_document,
    write_private_gold_labels,
)
from paper_agent.stage2_evaluation import GoldLabelStore, GoldManifest, GoldPair, GoldSplit
from paper_agent.stage2_promotion_artifacts import load_private_gold_labels
from paper_agent.stage2_sampling import PrivateSamplingAnnotation, PrivateSamplingAnnotations


def _manifest() -> GoldManifest:
    pairs: list[GoldPair] = []
    for split, size in ((GoldSplit.DEV, 300), (GoldSplit.HIDDEN_HARD, 150), (GoldSplit.HIDDEN_REAL, 150)):
        for index in range(size):
            pairs.append(GoldPair(
                f"paper-{split.value}-{index}", f"topic-{index % 6}", "zh" if index % 2 else "en",
                "synthetic-corpus",
                0.2 if split is GoldSplit.HIDDEN_REAL else None,
                f"family-{split.value}-{index}", "a" * 64, split,
                abstract_incomplete=split is not GoldSplit.HIDDEN_REAL and index < size // 10,
                sampled_from_natural_distribution=split is GoldSplit.HIDDEN_REAL,
                cross_language_match=index % 20 == 0,
            ))
    return GoldManifest(1, "a" * 64, tuple(pairs), ("en", "zh"))


def _ledger(manifest: GoldManifest) -> dict[str, object]:
    labels = {pair.pair_id: 2 for pair in manifest.pairs}
    dev = [pair for pair in manifest.pairs if pair.split is GoldSplit.DEV]
    hard = [pair for pair in manifest.pairs if pair.split is GoldSplit.HIDDEN_HARD]
    hard_negatives = {pair.pair_id for pair in dev[:60]} | {pair.pair_id for pair in hard[:30]}
    hard_positives = {dev[-1].pair_id, hard[-1].pair_id}
    for pair_id in hard_negatives:
        labels[pair_id] = 0
    for pair_id in hard_positives:
        labels[pair_id] = 3
    disagreement = manifest.pairs[100].pair_id
    annotations = []
    for pair in manifest.pairs:
        annotations.extend((
            {"pair_id": pair.pair_id, "annotator_id": "annotator-a", "label": labels[pair.pair_id]},
            {"pair_id": pair.pair_id, "annotator_id": "annotator-b", "label": 1 if pair.pair_id == disagreement else labels[pair.pair_id]},
        ))
    return {
        "schema_version": "1",
        "gold_manifest_hash": manifest.hash(),
        "rubric_version": 1,
        "rubric_hash": STAGE2_ANNOTATION_RUBRIC_HASH,
        "sampling_provenance_hash": "c" * 64,
        "annotation_input_hash": "d" * 64,
        "annotator_ids": ["annotator-a", "annotator-b"],
        "adjudicator_id": "adjudicator-c",
        "annotations": annotations,
        "adjudications": [{"pair_id": disagreement, "adjudicator_id": "adjudicator-c", "label": labels[disagreement]}],
        "hard_negative_pair_ids": sorted(hard_negatives),
        "hard_positive_pair_ids": sorted(hard_positives),
    }


def test_annotation_ledger_recomputes_complete_double_annotation_deterministically(tmp_path: Path) -> None:
    manifest = _manifest()
    document = _ledger(manifest)
    first = annotation_ledger_from_document(document, manifest=manifest)
    reordered = deepcopy(document)
    reordered["annotations"].reverse()
    second = annotation_ledger_from_document(reordered, manifest=manifest)

    assert first.summary.annotation_artifact_hash == second.summary.annotation_artifact_hash
    assert first.gold_labels.annotation_artifact_hash == first.summary.annotation_artifact_hash
    assert first.summary.quadratic_weighted_kappa >= 0.75
    assert first.summary.labels == second.summary.labels

    path = tmp_path / "ledger.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    assert load_annotation_ledger(path, manifest=manifest) == first


def test_annotation_ledger_rejects_unbound_or_incomplete_annotation_records() -> None:
    manifest = _manifest()
    document = _ledger(manifest)
    document["gold_manifest_hash"] = "b" * 64
    with pytest.raises(AnnotationLedgerArtifactError, match="does not bind"):
        annotation_ledger_from_document(document, manifest=manifest)

    document = _ledger(manifest)
    document["annotations"].pop()
    with pytest.raises(AnnotationLedgerArtifactError, match="exactly the same two annotators"):
        annotation_ledger_from_document(document, manifest=manifest)

    document = _ledger(manifest)
    document["rubric_hash"] = sha256(b"other-rubric").hexdigest()
    with pytest.raises(AnnotationLedgerArtifactError, match="frozen rubric"):
        annotation_ledger_from_document(document, manifest=manifest)


def test_annotation_ledger_requires_a_distinct_third_adjudicator_for_disagreements() -> None:
    manifest = _manifest()
    document = _ledger(manifest)
    document["adjudicator_id"] = "annotator-a"
    document["adjudications"][0]["adjudicator_id"] = "annotator-a"
    with pytest.raises(AnnotationLedgerArtifactError, match="distinct fixed adjudicator"):
        annotation_ledger_from_document(document, manifest=manifest)

    document = _ledger(manifest)
    document["adjudications"].append({
        "pair_id": manifest.pairs[0].pair_id, "adjudicator_id": "adjudicator-c", "label": 2,
    })
    with pytest.raises(AnnotationLedgerArtifactError, match="only disagreements"):
        annotation_ledger_from_document(document, manifest=manifest)


def test_verified_annotation_ledger_writes_private_gold_labels_without_raw_annotations(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    ledger = annotation_ledger_from_document(_ledger(manifest), manifest=manifest)

    document = private_gold_labels_document(ledger, manifest=manifest)
    assert document["labels"] == sorted(document["labels"], key=lambda row: row["pair_id"])
    assert document["annotation_artifact_hash"] == ledger.gold_labels.annotation_artifact_hash
    assert document["hard_negative_pair_ids"] == sorted(ledger.gold_labels.hard_negative_pair_ids)
    assert document["hard_positive_pair_ids"] == sorted(ledger.gold_labels.hard_positive_pair_ids)
    assert not {
        "annotator_ids", "adjudicator_id", "annotations", "adjudications", "rubric_version", "rubric_hash",
    } & set(document)

    path = tmp_path / "private-gold-labels.json"
    write_private_gold_labels(path, ledger, manifest=manifest)
    assert load_private_gold_labels(path, manifest=manifest) == ledger.gold_labels

    with pytest.raises(FileExistsError):
        write_private_gold_labels(path, ledger, manifest=manifest)

    mismatched = replace(
        ledger,
        gold_labels=GoldLabelStore(
            ledger.gold_labels.labels,
            "0" * 64,
            ledger.gold_labels.hard_negative_pair_ids,
            ledger.gold_labels.hard_positive_pair_ids,
        ),
    )
    with pytest.raises(AnnotationLedgerArtifactError, match="do not match"):
        private_gold_labels_document(mismatched, manifest=manifest)


def _human_worklist(
    manifest: GoldManifest,
    participant_id: str,
    labels: dict[str, int],
    *,
    role: str = "annotator",
) -> dict[str, object]:
    return {
        "role": role,
        "participant_id": participant_id,
        "annotation_input_hash": None,
        "rows": [
            {
                "pair_id": pair.pair_id,
                "topic": pair.topic,
                "title": f"Title {pair.paper_id}",
                "abstract": f"Abstract {pair.paper_id}",
                "language": pair.language,
                "label": labels[pair.pair_id],
            }
            for pair in manifest.pairs
            if pair.pair_id in labels
        ],
    }


def test_human_annotation_agreement_requires_qwk_gate_and_distinct_people() -> None:
    manifest = _manifest()
    labels = {pair.pair_id: index % 4 for index, pair in enumerate(manifest.pairs)}
    first = _human_worklist(manifest, "annotator-a", labels)
    second = _human_worklist(
        manifest,
        "annotator-b",
        {pair_id: 3 - label for pair_id, label in labels.items()},
    )

    with pytest.raises(AnnotationLedgerArtifactError, match="below 0.75"):
        annotation_agreement(manifest, first, second)
    same_person = dict(second)
    same_person["participant_id"] = "annotator-a"
    with pytest.raises(AnnotationLedgerArtifactError, match="two distinct annotators"):
        annotation_agreement(manifest, first, same_person)


def test_ledger_assembly_filters_provisional_hard_candidates_by_final_human_label() -> None:
    manifest = _manifest()
    source = _ledger(manifest)
    final_labels = {
        row["pair_id"]: row["label"]
        for row in source["annotations"]
        if row["annotator_id"] == "annotator-a"
    }
    disagreement = source["adjudications"][0]["pair_id"]
    second_labels = dict(final_labels)
    second_labels[disagreement] = 1
    first = _human_worklist(manifest, "annotator-a", final_labels)
    second = _human_worklist(manifest, "annotator-b", second_labels)
    adjudication = _human_worklist(
        manifest,
        "adjudicator-c",
        {disagreement: final_labels[disagreement]},
        role="adjudicator",
    )
    adjudication["annotation_input_hash"] = annotation_inputs_hash(first, second)
    hard_negatives = set(source["hard_negative_pair_ids"])
    hard_positives = set(source["hard_positive_pair_ids"])
    compatible = [
        PrivateSamplingAnnotation(
            pair.topic,
            pair.paper_id,
            final_labels[pair.pair_id],
            pair.pair_id in hard_negatives,
            pair.pair_id in hard_positives,
        )
        for pair in manifest.pairs
        if pair.pair_id in hard_negatives | hard_positives
    ]
    incompatible_negative = next(
        pair for pair in manifest.pairs
        if pair.split is GoldSplit.DEV and final_labels[pair.pair_id] == 2
    )
    incompatible_positive = next(
        pair for pair in manifest.pairs
        if pair.split is GoldSplit.HIDDEN_HARD and final_labels[pair.pair_id] == 2
    )
    curated = PrivateSamplingAnnotations(tuple((
        *compatible,
        PrivateSamplingAnnotation(
            incompatible_negative.topic,
            incompatible_negative.paper_id,
            0,
            hard_negative=True,
        ),
        PrivateSamplingAnnotation(
            incompatible_positive.topic,
            incompatible_positive.paper_id,
            3,
            hard_positive=True,
        ),
    )))

    document, ledger = assemble_annotation_ledger(
        manifest,
        first,
        second,
        adjudication,
        curated,
        sampling_provenance_hash="e" * 64,
    )

    assert document["rubric_hash"] == STAGE2_ANNOTATION_RUBRIC_HASH
    assert ledger.gold_labels.hard_negative_pair_ids == frozenset(hard_negatives)
    assert ledger.gold_labels.hard_positive_pair_ids == frozenset(hard_positives)
    assert incompatible_negative.pair_id not in document["hard_negative_pair_ids"]
    assert incompatible_positive.pair_id not in document["hard_positive_pair_ids"]

    changed_first = deepcopy(first)
    agreement = next(pair_id for pair_id in final_labels if pair_id != disagreement)
    replacement = 3 if final_labels[agreement] != 3 else 2
    changed_first["rows"] = [
        {**row, "label": replacement} if row["pair_id"] == agreement else row
        for row in changed_first["rows"]
    ]
    changed_second = deepcopy(second)
    changed_second["rows"] = [
        {**row, "label": replacement} if row["pair_id"] == agreement else row
        for row in changed_second["rows"]
    ]
    with pytest.raises(AnnotationLedgerArtifactError, match="adjudication must"):
        assemble_annotation_ledger(
            manifest,
            changed_first,
            changed_second,
            adjudication,
            curated,
            sampling_provenance_hash="e" * 64,
        )
