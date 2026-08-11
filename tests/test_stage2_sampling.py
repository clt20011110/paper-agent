from __future__ import annotations

from dataclasses import replace

import pytest

from paper_agent.stage2_evaluation import GoldSplit
from paper_agent.stage2_sampling import (
    CorpusPaper,
    PrivateCorpusSnapshot,
    PrivateSamplingAnnotation,
    PrivateSamplingAnnotations,
    SamplingPolicy,
    build_gold_sampling,
    private_corpus_snapshot_from_document,
)


def _inputs(*, probability: float = 0.2):
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
    assert first.manifest.validate(first.labels) is None
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
    real_keys = {
        (pair.topic, pair.paper_id)
        for pair in first.manifest.pairs
        if pair.split is GoldSplit.HIDDEN_REAL
    }
    changed_annotations = PrivateSamplingAnnotations(tuple(
        replace(row, label=0)
        if row.key in real_keys and not row.hard_negative and not row.hard_positive
        else row
        for row in annotations.rows
    ))

    second = build_gold_sampling(snapshot, changed_annotations, policy)

    assert {
        pair.pair_id for pair in first.manifest.pairs if pair.split is GoldSplit.HIDDEN_REAL
    } == {
        pair.pair_id for pair in second.manifest.pairs if pair.split is GoldSplit.HIDDEN_REAL
    }
    assert {
        pair.sampling_probability for pair in first.manifest.pairs if pair.split is GoldSplit.HIDDEN_REAL
    } == {0.2}


def test_producer_rejects_nonuniform_natural_probability() -> None:
    snapshot, annotations, policy = _inputs()
    papers = tuple(
        replace(paper, sampling_probability=0.1 if index % 2 else 0.2)
        for index, paper in enumerate(snapshot.papers)
    )
    nonuniform = PrivateCorpusSnapshot(1, policy.version, policy.seed, papers)

    with pytest.raises(ValueError, match="one recorded sampling_probability"):
        build_gold_sampling(nonuniform, annotations, policy)
