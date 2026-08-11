from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest

from paper_agent.stage2_annotation_artifacts import (
    AnnotationLedgerArtifactError,
    annotation_ledger_from_document,
    load_annotation_ledger,
)
from paper_agent.stage2_evaluation import GoldManifest, GoldPair, GoldSplit


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
        "rubric_hash": sha256(b"stage2-rubric-v1").hexdigest(),
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
