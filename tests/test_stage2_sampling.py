from __future__ import annotations

from dataclasses import replace

import pytest

from paper_agent.stage2_evaluation import GoldLabelStore, GoldSplit, make_pair_id
from paper_agent.stage2_sampling import (
    CorpusPaper,
    PrivateCorpusSnapshot,
    PrivateSamplingAnnotation,
    PrivateSamplingAnnotations,
    SamplingPolicy,
    build_gold_sampling,
    private_corpus_snapshot_from_document,
)


def _inputs(*, probability: float = 150 / 1080):
    papers: list[CorpusPaper] = []
    annotations: list[PrivateSamplingAnnotation] = []
    for topic_index in range(6):
        for index in range(180):
            paper_id = f"paper-{topic_index}-{index}"
            hard_negative = index % 4 == 0
            hard_positive = not hard_negative and index % 9 == 1
            label = 0 if hard_negative else 3 if hard_positive else 2
            papers.append(CorpusPaper(
                topic=f"topic-{topic_index}",
                paper_id=paper_id,
                title=f"Title {paper_id}",
                abstract=None if index % 8 == 0 else f"Abstract {paper_id}",
                metadata={"crawler_id": paper_id, "year": 2025},
                source="frozen-crawler-snapshot",
                language="zh" if index % 2 else "en",
                paper_family=f"family-{paper_id}",
                sampling_weight=1 + (index % 3),
                sampling_probability=probability,
                abstract_incomplete=index % 8 == 0,
                cross_language_match=index % 23 == 0,
            ))
            annotations.append(PrivateSamplingAnnotation(
                f"topic-{topic_index}", paper_id, label, hard_negative, hard_positive
            ))
    policy = SamplingPolicy("stage2-producer-v1", 741)
    snapshot = PrivateCorpusSnapshot(1, policy.version, policy.seed, tuple(papers))
    return snapshot, PrivateSamplingAnnotations(tuple(annotations)), policy


def test_producer_builds_reproducible_valid_gold_manifest_and_private_binding() -> None:
    snapshot, annotations, policy = _inputs()

    first = build_gold_sampling(snapshot, annotations, policy)
    second = build_gold_sampling(snapshot, annotations, policy)

    assert first.manifest.hash() == second.manifest.hash()
    annotation_by_pair_id = {
        make_pair_id(row.topic, row.paper_id): row
        for row in annotations.rows
    }
    selected_annotations = {
        pair.pair_id: annotation_by_pair_id[pair.pair_id]
        for pair in first.manifest.pairs
    }
    labels = GoldLabelStore(
        {pair_id: row.label for pair_id, row in selected_annotations.items()},
        annotations.hash(),
        frozenset(pair_id for pair_id, row in selected_annotations.items() if row.hard_negative),
        frozenset(pair_id for pair_id, row in selected_annotations.items() if row.hard_positive),
    )
    assert first.manifest.validate(labels) is None
    assert {pair.split for pair in first.manifest.pairs} == set(GoldSplit)
    assert len(first.manifest.pairs) == 600
    assert len({pair.topic for pair in first.manifest.pairs}) == 6
    assert {pair.language for pair in first.manifest.pairs} >= {"en", "zh"}
    assert all(
        pair.sampled_from_natural_distribution is (pair.split is GoldSplit.HIDDEN_REAL)
        for pair in first.manifest.pairs
    )
    families: dict[str, GoldSplit] = {}
    for pair in first.manifest.pairs:
        assert families.setdefault(pair.paper_family, pair.split) is pair.split
    assert first.provenance.snapshot_hash == snapshot.hash()
    assert first.provenance.corpus_hash == snapshot.corpus_hash
    assert first.provenance.sampling_policy_version == policy.version
    assert first.provenance.sampling_seed == policy.seed
    assert first.provenance.sampling_annotations_hash == annotations.hash()
    assert first.provenance.gold_manifest_hash == first.manifest.hash()


def test_private_snapshot_round_trip_keeps_text_metadata_and_corpus_hash() -> None:
    snapshot, _, _ = _inputs()

    restored = private_corpus_snapshot_from_document(snapshot.document())

    assert restored.hash() == snapshot.hash()
    assert restored.papers[0].title == snapshot.papers[0].title
    assert restored.papers[0].metadata == snapshot.papers[0].metadata
    document = snapshot.document()
    document["papers"][0]["title"] = "changed"
    with pytest.raises(ValueError, match="corpus_hash"):
        private_corpus_snapshot_from_document(document)


def test_hidden_real_selection_ignores_labels_and_keeps_recorded_natural_probability() -> None:
    snapshot, annotations, policy = _inputs()
    first = build_gold_sampling(snapshot, annotations, policy)
    changed_rows: list[PrivateSamplingAnnotation] = []
    for row in annotations.rows:
        index = int(row.paper_id.rsplit("-", 1)[1])
        hard_negative = index % 4 == 2
        hard_positive = not hard_negative and index % 9 == 3
        label = 1 if hard_negative else 3 if hard_positive else (row.label + 1) % 4
        changed_rows.append(PrivateSamplingAnnotation(
            row.topic,
            row.paper_id,
            label,
            hard_negative,
            hard_positive,
        ))
    changed_annotations = PrivateSamplingAnnotations(tuple(changed_rows))
    assert all(
        changed.document() != original.document()
        for original, changed in zip(annotations.rows, changed_annotations.rows, strict=True)
    )

    second = build_gold_sampling(snapshot, changed_annotations, policy)

    assert {
        pair.pair_id for pair in first.manifest.pairs if pair.split is GoldSplit.HIDDEN_REAL
    } == {
        pair.pair_id for pair in second.manifest.pairs if pair.split is GoldSplit.HIDDEN_REAL
    }
    natural_frame_size = sum(paper.natural_crawler_population for paper in snapshot.papers)
    assert {
        pair.sampling_probability for pair in first.manifest.pairs if pair.split is GoldSplit.HIDDEN_REAL
    } == {policy.hidden_real_size / natural_frame_size}


def test_producer_rejects_nonuniform_natural_probability() -> None:
    snapshot, annotations, policy = _inputs()
    papers = tuple(
        replace(paper, sampling_probability=0.1 if index % 2 else 0.2)
        for index, paper in enumerate(snapshot.papers)
    )
    nonuniform = PrivateCorpusSnapshot(1, policy.version, policy.seed, papers)

    with pytest.raises(ValueError, match="natural frame size"):
        build_gold_sampling(nonuniform, annotations, policy)


def test_producer_rejects_consistent_but_incorrect_natural_probability() -> None:
    snapshot, annotations, policy = _inputs(probability=0.2)

    with pytest.raises(ValueError, match="natural frame size"):
        build_gold_sampling(snapshot, annotations, policy)


def test_unannotated_natural_rows_can_enter_hidden_real_but_never_labelled_strata() -> None:
    snapshot, annotations, policy = _inputs()
    curated = PrivateSamplingAnnotations(tuple(
        row for row in annotations.rows
        if int(row.paper_id.rsplit("-", 1)[1]) < 150
    ))

    result = build_gold_sampling(snapshot, curated, policy)
    selected = {
        (pair.topic, pair.paper_id): pair.split
        for pair in result.manifest.pairs
    }

    assert any(
        key not in curated.by_key and split is GoldSplit.HIDDEN_REAL
        for key, split in selected.items()
    )
    assert all(
        key in curated.by_key
        for key, split in selected.items()
        if split in (GoldSplit.DEV, GoldSplit.HIDDEN_HARD)
    )


def test_producer_rejects_annotation_row_outside_snapshot() -> None:
    snapshot, annotations, policy = _inputs()
    unknown = PrivateSamplingAnnotation("topic-unknown", "paper-unknown", 0)

    with pytest.raises(ValueError, match="outside the private corpus snapshot"):
        build_gold_sampling(snapshot, PrivateSamplingAnnotations((*annotations.rows, unknown)), policy)


def test_producer_fails_when_curated_pool_cannot_fill_labelled_strata() -> None:
    snapshot, annotations, policy = _inputs()
    insufficient = PrivateSamplingAnnotations(annotations.rows[:449])

    with pytest.raises(ValueError, match="curated candidates"):
        build_gold_sampling(snapshot, insufficient, policy)
