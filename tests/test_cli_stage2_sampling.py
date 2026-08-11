from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from paper_agent import cli
from paper_agent.stage2_evaluation import load_gold_manifest
from paper_agent.stage2_sampling import (
    CorpusPaper,
    PrivateCorpusSnapshot,
    PrivateSamplingAnnotation,
    PrivateSamplingAnnotations,
    SamplingPolicy,
    load_gold_sampling_provenance,
    load_private_corpus_snapshot,
    load_private_sampling_annotations,
    write_private_corpus_snapshot,
    write_private_sampling_annotations,
)


@pytest.fixture(scope="module")
def sampling_inputs(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("stage2-sampling-cli")
    probability = 150 / 1080
    papers: list[CorpusPaper] = []
    rows: list[PrivateSamplingAnnotation] = []
    for topic_index in range(6):
        for index in range(180):
            paper_id = f"paper-{topic_index}-{index}"
            hard_negative = index % 4 == 0
            hard_positive = not hard_negative and index % 9 == 1
            papers.append(CorpusPaper(
                topic=f"topic-{topic_index}",
                paper_id=paper_id,
                title=f"Private title {paper_id}",
                abstract=f"Private abstract {paper_id}",
                metadata={"crawler_id": paper_id},
                source="frozen-crawler-snapshot",
                language="zh" if index % 2 else "en",
                paper_family=f"family-{paper_id}",
                sampling_weight=1,
                sampling_probability=probability,
                abstract_incomplete=index % 8 == 0,
                cross_language_match=index % 23 == 0,
            ))
            if index < 150:
                rows.append(PrivateSamplingAnnotation(
                    f"topic-{topic_index}",
                    paper_id,
                    0 if hard_negative else 3 if hard_positive else 2,
                    hard_negative,
                    hard_positive,
                ))

    policy = SamplingPolicy("stage2-sampling-cli-v1", 741)
    snapshot = PrivateCorpusSnapshot(1, policy.version, policy.seed, tuple(papers))
    annotations = PrivateSamplingAnnotations(tuple(rows))
    snapshot_path = root / "private-snapshot.json"
    annotations_path = root / "curated-annotations.json"
    write_private_corpus_snapshot(snapshot_path, snapshot)
    write_private_sampling_annotations(
        annotations_path, annotations, snapshot=snapshot
    )
    return snapshot_path, annotations_path


def _arguments(
    snapshot: Path,
    annotations: Path,
    manifest: Path,
    provenance: Path,
) -> list[str]:
    return [
        "stage2-sampling",
        "build",
        "--private-snapshot",
        str(snapshot),
        "--curated-annotations",
        str(annotations),
        "--gold-manifest-output",
        str(manifest),
        "--provenance-output",
        str(provenance),
    ]


def test_stage2_sampling_cli_dry_run_then_builds_private_bound_public_manifest(
    tmp_path: Path,
    sampling_inputs: tuple[Path, Path],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot_path, annotations_path = sampling_inputs
    manifest_path = tmp_path / "gold-manifest.json"
    provenance_path = tmp_path / "sampling-provenance.json"
    arguments = _arguments(
        snapshot_path, annotations_path, manifest_path, provenance_path
    )
    access_order: list[str] = []
    real_select_hidden = cli.select_hidden_real
    real_load_annotations = cli.load_private_sampling_annotations

    def select_hidden(*args, **kwargs):
        access_order.append("hidden_real_frozen")
        return real_select_hidden(*args, **kwargs)

    def load_annotations(*args, **kwargs):
        access_order.append("curated_annotations_opened")
        return real_load_annotations(*args, **kwargs)

    monkeypatch.setattr(cli, "select_hidden_real", select_hidden)
    monkeypatch.setattr(cli, "load_private_sampling_annotations", load_annotations)

    assert cli.main(["--dry-run", *arguments]) == 0
    dry_run = json.loads(capsys.readouterr().out)
    assert dry_run["status"] == "validated"
    assert dry_run["written"] is False
    assert dry_run["split_counts"] == {
        "dev": 300,
        "hidden_hard": 150,
        "hidden_real": 150,
    }
    assert not manifest_path.exists()
    assert not provenance_path.exists()
    assert access_order == ["hidden_real_frozen", "curated_annotations_opened"]

    assert cli.main(arguments) == 0
    built = json.loads(capsys.readouterr().out)
    assert built["status"] == "complete"
    assert built["written"] is True
    assert "Private title" not in json.dumps(built)
    assert "Private abstract" not in json.dumps(built)
    assert access_order == [
        "hidden_real_frozen",
        "curated_annotations_opened",
        "hidden_real_frozen",
        "curated_annotations_opened",
    ]

    snapshot = load_private_corpus_snapshot(snapshot_path)
    annotations = load_private_sampling_annotations(
        annotations_path, snapshot=snapshot
    )
    manifest = load_gold_manifest(manifest_path)
    manifest.validate_sampling_structure()
    provenance = load_gold_sampling_provenance(
        provenance_path,
        snapshot=snapshot,
        annotations=annotations,
        manifest=manifest,
    )
    assert built["gold_manifest_hash"] == manifest.hash()
    assert built["provenance_hash"] == provenance.hash()

    public_document = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "labels" not in public_document
    assert all(
        not {"title", "abstract", "label", "hard_negative", "hard_positive"} & set(pair)
        for pair in public_document["pairs"]
    )

    with pytest.raises(FileExistsError, match="already exists"):
        cli.main(arguments)


def test_stage2_sampling_cli_rejects_one_path_for_both_outputs(
    tmp_path: Path,
    sampling_inputs: tuple[Path, Path],
) -> None:
    snapshot_path, annotations_path = sampling_inputs
    output = tmp_path / "same-output.json"

    with pytest.raises(cli.CliUsageError, match="different paths"):
        cli.main(_arguments(snapshot_path, annotations_path, output, output))


def test_stage2_finalize_annotations_cli_validates_before_private_write(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_hash = "a" * 64
    annotation_hash = "b" * 64
    label_store_hash = "c" * 64
    manifest = SimpleNamespace(
        hash=lambda: manifest_hash,
        validate_sampling_structure=lambda: None,
    )
    gold_labels = SimpleNamespace(
        labels={f"pair-{index}": index % 4 for index in range(600)},
        hash=lambda: label_store_hash,
    )
    ledger = SimpleNamespace(
        gold_labels=gold_labels,
        summary=SimpleNamespace(
            annotation_artifact_hash=annotation_hash,
            quadratic_weighted_kappa=0.81,
        ),
    )
    writes: list[tuple[Path, object, object]] = []
    monkeypatch.setattr(cli, "load_gold_manifest", lambda _path: manifest)
    monkeypatch.setattr(
        cli,
        "load_annotation_ledger",
        lambda _path, *, manifest: ledger,
    )
    monkeypatch.setattr(
        cli,
        "write_private_gold_labels",
        lambda path, value, *, manifest: writes.append((path, value, manifest)),
    )
    output = tmp_path / "private-labels.json"
    arguments = [
        "stage2-sampling",
        "finalize-annotations",
        "--gold-manifest",
        str(tmp_path / "gold-manifest.json"),
        "--annotation-ledger",
        str(tmp_path / "annotation-ledger.json"),
        "--private-labels-output",
        str(output),
    ]

    assert cli.main(["--dry-run", *arguments]) == 0
    dry_run = json.loads(capsys.readouterr().out)
    assert dry_run["status"] == "validated"
    assert dry_run["label_count"] == 600
    assert dry_run["pre_adjudication_quadratic_weighted_kappa"] == 0.81
    assert writes == []

    assert cli.main(arguments) == 0
    completed = json.loads(capsys.readouterr().out)
    assert completed["status"] == "complete"
    assert completed["annotation_artifact_hash"] == annotation_hash
    assert completed["gold_label_store_hash"] == label_store_hash
    assert writes == [(output, ledger, manifest)]
    assert "annotator_ids" not in completed
    assert "annotations" not in completed
    assert "adjudications" not in completed

    output.write_text("reserved", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        cli.main(arguments)
