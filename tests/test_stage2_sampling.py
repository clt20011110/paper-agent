from __future__ import annotations

from dataclasses import replace
import json

import pytest

from paper_agent.schema import SchemaValidationError, validate
from paper_agent.stage2_evaluation import GoldLabelStore, GoldSplit
from paper_agent.stage2_sampling import (
    CorpusPaper,
    PrivateCorpusSnapshot,
    PrivateSamplingAnnotation,
    PrivateSamplingAnnotations,
    SamplingPolicy,
    build_gold_sampling,
    gold_sampling_provenance_from_document,
    hidden_real_selection_from_document,
    load_hidden_real_selection,
    load_gold_sampling_provenance,
    load_private_corpus_snapshot,
    load_private_sampling_annotations,
    private_corpus_snapshot_from_document,
    private_sampling_annotations_from_document,
    select_hidden_real,
    write_gold_sampling_manifest,
    write_gold_sampling_provenance,
    write_hidden_real_selection,
    write_private_corpus_snapshot,
    write_private_sampling_annotations,
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
    selection_probability = policy.hidden_real_size / len(papers)
    selection_snapshot = PrivateCorpusSnapshot(
        1,
        policy.version,
        policy.seed,
        tuple(
            replace(paper, sampling_probability=selection_probability)
            for paper in papers
        ),
    )
    hidden = select_hidden_real(selection_snapshot, policy)
    by_key = {paper.key: paper for paper in snapshot.papers}
    hidden_families = {by_key[key].paper_family for key in hidden.pair_keys}
    curated = PrivateSamplingAnnotations(tuple(
        row for row in annotations if by_key[row.key].paper_family not in hidden_families
    ))
    return snapshot, curated, policy


def test_producer_builds_reproducible_valid_gold_manifest_and_private_binding() -> None:
    snapshot, annotations, policy = _inputs()

    first = build_gold_sampling(snapshot, annotations, policy)
    second = build_gold_sampling(snapshot, annotations, policy)

    assert first.manifest.hash() == second.manifest.hash()
    final_labels: dict[str, int] = {}
    hard_negatives: set[str] = set()
    hard_positives: set[str] = set()
    for pair in first.manifest.pairs:
        index = int(pair.paper_id.rsplit("-", 1)[1])
        if index % 4 == 0:
            final_labels[pair.pair_id] = 0
            hard_negatives.add(pair.pair_id)
        elif index % 9 == 1:
            final_labels[pair.pair_id] = 3
            hard_positives.add(pair.pair_id)
        else:
            final_labels[pair.pair_id] = 2
    labels = GoldLabelStore(
        final_labels,
        "post-selection-double-annotation-fixture",
        frozenset(hard_negatives),
        frozenset(hard_positives),
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
    assert all(
        (pair.sampling_probability is not None)
        is (pair.split is GoldSplit.HIDDEN_REAL)
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


def test_private_sampling_artifacts_round_trip_bind_inputs_and_keep_public_manifest_safe(tmp_path) -> None:
    snapshot, annotations, policy = _inputs()
    snapshot_path = tmp_path / "private-snapshot.json"
    annotations_path = tmp_path / "private-annotations.json"
    provenance_path = tmp_path / "sampling-provenance.json"
    manifest_path = tmp_path / "gold-manifest.json"

    write_private_corpus_snapshot(snapshot_path, snapshot)
    restored_snapshot = load_private_corpus_snapshot(snapshot_path)
    write_private_sampling_annotations(annotations_path, annotations, snapshot=restored_snapshot)
    restored_annotations = load_private_sampling_annotations(annotations_path, snapshot=restored_snapshot)
    result = build_gold_sampling(restored_snapshot, restored_annotations, policy)
    write_gold_sampling_provenance(provenance_path, result.provenance)
    restored_provenance = load_gold_sampling_provenance(
        provenance_path,
        snapshot=restored_snapshot,
        annotations=restored_annotations,
        manifest=result.manifest,
    )

    assert restored_snapshot.hash() == snapshot.hash()
    assert restored_annotations.hash() == annotations.hash()
    assert restored_provenance.hash() == result.provenance.hash()

    unknown = annotations.document(snapshot=snapshot)
    unknown["label"] = 3
    with pytest.raises(ValueError, match="Additional properties"):
        private_sampling_annotations_from_document(unknown, snapshot=snapshot)
    wrong_binding = annotations.document(snapshot=snapshot)
    wrong_binding["snapshot_hash"] = "0" * 64
    with pytest.raises(ValueError, match="do not bind"):
        private_sampling_annotations_from_document(wrong_binding, snapshot=snapshot)
    bad_provenance = result.provenance.document()
    bad_provenance["gold_manifest_hash"] = "0" * 64
    with pytest.raises(ValueError, match="gold manifest"):
        gold_sampling_provenance_from_document(
            bad_provenance,
            snapshot=snapshot,
            annotations=annotations,
            manifest=result.manifest,
        )

    write_gold_sampling_manifest(manifest_path, result.manifest)
    with pytest.raises(FileExistsError):
        write_gold_sampling_manifest(manifest_path, result.manifest)
    with pytest.raises(FileExistsError):
        write_gold_sampling_provenance(provenance_path, result.provenance)
    public_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate(public_manifest, "stage2-gold-manifest.schema.json")
    assert "labels" not in public_manifest
    assert all(not {"title", "abstract", "labels"} & set(pair) for pair in public_manifest["pairs"])
    dev_pair = next(pair for pair in public_manifest["pairs"] if pair["split"] == "dev")
    dev_pair["sampling_probability"] = 0.2
    with pytest.raises(SchemaValidationError):
        validate(public_manifest, "stage2-gold-manifest.schema.json")


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


def test_hidden_real_freeze_frame_round_trips_and_binds_snapshot_and_policy(tmp_path) -> None:
    snapshot, _, policy = _inputs()
    selection = select_hidden_real(snapshot, policy)
    path = tmp_path / "hidden-real-freeze-frame.json"

    write_hidden_real_selection(path, selection)
    restored = load_hidden_real_selection(path, snapshot=snapshot, policy=policy)

    assert restored == selection
    assert restored.hash() == selection.hash()
    document = selection.document()
    document["snapshot_hash"] = "0" * 64
    with pytest.raises(ValueError, match="does not match"):
        hidden_real_selection_from_document(
            document,
            snapshot=snapshot,
            policy=policy,
        )


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


def test_producer_rejects_curated_rows_from_a_hidden_real_family() -> None:
    snapshot, annotations, policy = _inputs()
    hidden = select_hidden_real(snapshot, policy)
    topic, paper_id = hidden.pair_keys[0]
    contaminated = PrivateSamplingAnnotation(topic, paper_id, 0, True)

    with pytest.raises(ValueError, match="HIDDEN_REAL paper family"):
        build_gold_sampling(
            snapshot,
            PrivateSamplingAnnotations((*annotations.rows, contaminated)),
            policy,
            hidden_real_selection=hidden,
        )


def test_producer_fails_when_curated_pool_cannot_fill_labelled_strata() -> None:
    snapshot, annotations, policy = _inputs()
    insufficient = PrivateSamplingAnnotations(annotations.rows[:449])

    with pytest.raises(ValueError, match="curated candidates"):
        build_gold_sampling(snapshot, insufficient, policy)
