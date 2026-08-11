from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from paper_agent import cli
from paper_agent.stage2_evaluation import load_gold_manifest
from paper_agent.stage2_sampling import (
    CorpusPaper,
    CurationDecision,
    CurationDecisions,
    PrivateCorpusSnapshot,
    SamplingPolicy,
    curated_annotations_from_decisions,
    load_hidden_real_selection,
    load_gold_sampling_provenance,
    load_private_corpus_snapshot,
    load_private_sampling_annotations,
    make_curation_receipt,
    make_curation_worklist,
    select_hidden_real,
    write_curation_receipt,
    write_hidden_real_selection,
    write_private_corpus_snapshot,
    write_private_sampling_annotations,
)


@pytest.fixture(scope="module")
def sampling_inputs(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path, Path]:
    root = tmp_path_factory.mktemp("stage2-sampling-cli")
    probability = 150 / 1080
    papers: list[CorpusPaper] = []
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
    policy = SamplingPolicy("stage2-sampling-cli-v1", 741)
    snapshot = PrivateCorpusSnapshot(1, policy.version, policy.seed, tuple(papers))
    snapshot_path = root / "private-snapshot.json"
    annotations_path = root / "curated-annotations.json"
    freeze_frame_path = root / "hidden-real-freeze-frame.json"
    receipt_path = root / "curation-receipt.json"
    write_private_corpus_snapshot(snapshot_path, snapshot)
    hidden = select_hidden_real(snapshot, policy)
    write_hidden_real_selection(freeze_frame_path, hidden)
    worklist = make_curation_worklist(snapshot, hidden)
    decisions = CurationDecisions(worklist.hash(), tuple(
        CurationDecision(
            row.topic,
            row.paper_id,
            0 if int(row.paper_id.rsplit("-", 1)[1]) % 4 == 0 else 3 if int(row.paper_id.rsplit("-", 1)[1]) % 9 == 1 else 2,
            int(row.paper_id.rsplit("-", 1)[1]) % 4 == 0,
            int(row.paper_id.rsplit("-", 1)[1]) % 4 != 0 and int(row.paper_id.rsplit("-", 1)[1]) % 9 == 1,
            "human_provisional",
        )
        for row in worklist.rows
    ))
    annotations = curated_annotations_from_decisions(
        decisions, worklist=worklist, snapshot=snapshot
    )
    receipt = make_curation_receipt(snapshot, hidden, worklist, decisions, annotations)
    write_private_sampling_annotations(annotations_path, annotations, snapshot=snapshot)
    write_curation_receipt(receipt_path, receipt)
    return snapshot_path, annotations_path, freeze_frame_path, receipt_path


def _arguments(
    snapshot: Path,
    annotations: Path,
    freeze_frame: Path,
    receipt: Path,
    manifest: Path,
    provenance: Path,
) -> list[str]:
    return [
        "stage2-sampling",
        "build",
        "--private-snapshot",
        str(snapshot),
        "--hidden-real-freeze-frame",
        str(freeze_frame),
        "--curated-annotations",
        str(annotations),
        "--curation-receipt",
        str(receipt),
        "--gold-manifest-output",
        str(manifest),
        "--provenance-output",
        str(provenance),
    ]


def test_stage2_sampling_cli_dry_run_then_builds_private_bound_public_manifest(
    tmp_path: Path,
    sampling_inputs: tuple[Path, Path, Path, Path],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot_path, annotations_path, freeze_frame_path, receipt_path = sampling_inputs
    manifest_path = tmp_path / "gold-manifest.json"
    provenance_path = tmp_path / "sampling-provenance.json"
    arguments = _arguments(
        snapshot_path, annotations_path, freeze_frame_path, receipt_path, manifest_path, provenance_path
    )
    access_order: list[str] = []
    real_load_hidden = cli.load_hidden_real_selection
    real_load_receipt = cli.load_curation_receipt
    real_load_annotations = cli.load_private_sampling_annotations

    def load_hidden(*args, **kwargs):
        access_order.append("hidden_real_frame_validated")
        return real_load_hidden(*args, **kwargs)

    def load_annotations(*args, **kwargs):
        access_order.append("curated_annotations_opened")
        return real_load_annotations(*args, **kwargs)

    def load_receipt(*args, **kwargs):
        access_order.append("curation_receipt_validated")
        return real_load_receipt(*args, **kwargs)

    monkeypatch.setattr(cli, "load_hidden_real_selection", load_hidden)
    monkeypatch.setattr(cli, "load_curation_receipt", load_receipt)
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
    assert access_order == [
        "hidden_real_frame_validated",
        "curation_receipt_validated",
        "curated_annotations_opened",
    ]

    assert cli.main(arguments) == 0
    built = json.loads(capsys.readouterr().out)
    assert built["status"] == "complete"
    assert built["written"] is True
    assert "Private title" not in json.dumps(built)
    assert "Private abstract" not in json.dumps(built)
    assert access_order == [
        "hidden_real_frame_validated",
        "curation_receipt_validated",
        "curated_annotations_opened",
        "hidden_real_frame_validated",
        "curation_receipt_validated",
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
    sampling_inputs: tuple[Path, Path, Path, Path],
) -> None:
    snapshot_path, annotations_path, freeze_frame_path, receipt_path = sampling_inputs
    output = tmp_path / "same-output.json"

    with pytest.raises(cli.CliUsageError, match="different paths"):
        cli.main(_arguments(snapshot_path, annotations_path, freeze_frame_path, receipt_path, output, output))


def test_stage2_sampling_freeze_frame_cli_writes_before_any_annotations_are_opened(
    tmp_path: Path,
    sampling_inputs: tuple[Path, Path, Path, Path],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot_path, _, _, _ = sampling_inputs
    output = tmp_path / "hidden-real-freeze-frame.json"
    monkeypatch.setattr(
        cli,
        "load_private_sampling_annotations",
        lambda *args, **kwargs: pytest.fail("freeze-frame must not open annotations"),
    )
    arguments = [
        "stage2-sampling",
        "freeze-frame",
        "--private-snapshot",
        str(snapshot_path),
        "--output",
        str(output),
    ]

    assert cli.main(["--dry-run", *arguments]) == 0
    dry_run = json.loads(capsys.readouterr().out)
    assert dry_run["status"] == "validated"
    assert dry_run["hidden_real_count"] == 150
    assert not output.exists()

    assert cli.main(arguments) == 0
    completed = json.loads(capsys.readouterr().out)
    assert completed["status"] == "complete"
    snapshot = load_private_corpus_snapshot(snapshot_path)
    policy = SamplingPolicy(snapshot.sampling_policy_version, snapshot.sampling_seed)
    assert load_hidden_real_selection(output, snapshot=snapshot, policy=policy).hash() == (
        completed["hidden_real_freeze_frame_hash"]
    )
    with pytest.raises(FileExistsError, match="already exists"):
        cli.main(arguments)


def test_stage2_curation_cli_exports_and_imports_private_bound_annotations(
    tmp_path: Path,
    sampling_inputs: tuple[Path, Path, Path, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    snapshot_path, _, freeze_frame_path, _ = sampling_inputs
    worklist_path = tmp_path / "curation-worklist.json"
    worklist_arguments = [
        "stage2-sampling", "curation-worklist",
        "--private-snapshot", str(snapshot_path),
        "--hidden-real-freeze-frame", str(freeze_frame_path),
        "--output", str(worklist_path),
    ]
    assert cli.main(["--dry-run", *worklist_arguments]) == 0
    dry_run = json.loads(capsys.readouterr().out)
    assert dry_run["status"] == "validated"
    assert not worklist_path.exists()

    assert cli.main(worklist_arguments) == 0
    exported = json.loads(capsys.readouterr().out)
    snapshot = load_private_corpus_snapshot(snapshot_path)
    policy = SamplingPolicy(snapshot.sampling_policy_version, snapshot.sampling_seed)
    hidden = load_hidden_real_selection(freeze_frame_path, snapshot=snapshot, policy=policy)
    worklist = make_curation_worklist(snapshot, hidden)
    assert exported["worklist_hash"] == worklist.hash()
    hidden_families = {
        {paper.key: paper for paper in snapshot.papers}[key].paper_family
        for key in hidden.pair_keys
    }
    assert all(row.key not in hidden.pair_keys for row in worklist.rows)
    assert all(row.paper_family not in hidden_families for row in worklist.rows)
    with pytest.raises(FileExistsError, match="already exists"):
        cli.main(worklist_arguments)

    decisions = CurationDecisions(worklist.hash(), tuple(
        CurationDecision(
            row.topic, row.paper_id, 0 if index % 4 == 0 else 3 if index % 9 == 1 else 2,
            index % 4 == 0,
            index % 4 != 0 and index % 9 == 1,
            ("human_provisional", "model_provisional", "human_reviewed_model_suggestion")[index % 3],
        )
        for index, row in enumerate(worklist.rows)
    ))
    decisions_path = tmp_path / "curation-decisions.json"
    decisions_path.write_text(json.dumps(decisions.document()), encoding="utf-8")
    annotations_path = tmp_path / "curated-annotations.json"
    receipt_path = tmp_path / "curation-receipt.json"
    import_arguments = [
        "stage2-sampling", "curation-import",
        "--private-snapshot", str(snapshot_path),
        "--hidden-real-freeze-frame", str(freeze_frame_path),
        "--worklist", str(worklist_path),
        "--decisions", str(decisions_path),
        "--curated-annotations-output", str(annotations_path),
        "--receipt-output", str(receipt_path),
    ]
    assert cli.main(["--dry-run", *import_arguments]) == 0
    assert json.loads(capsys.readouterr().out)["written"] is False
    assert not annotations_path.exists()
    assert not receipt_path.exists()

    incomplete = decisions.document()
    incomplete["rows"].pop()
    incomplete_path = tmp_path / "incomplete-curation-decisions.json"
    incomplete_path.write_text(json.dumps(incomplete), encoding="utf-8")
    invalid_arguments = [
        *import_arguments[:9], str(incomplete_path),
        *import_arguments[10:],
    ]
    with pytest.raises(ValueError, match="exactly cover"):
        cli.main(["--dry-run", *invalid_arguments])

    assert cli.main(import_arguments) == 0
    imported = json.loads(capsys.readouterr().out)
    assert imported["source_counts"] == decisions.source_counts()
    assert load_private_sampling_annotations(annotations_path, snapshot=snapshot).hash() == imported["curated_annotations_hash"]
    assert json.loads(receipt_path.read_text(encoding="utf-8"))["worklist_hash"] == worklist.hash()
    with pytest.raises(FileExistsError, match="already exists"):
        cli.main(import_arguments)


def test_stage2_sampling_build_validates_freeze_frame_before_opening_annotations(
    tmp_path: Path,
    sampling_inputs: tuple[Path, Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot_path, annotations_path, _, receipt_path = sampling_inputs
    snapshot = load_private_corpus_snapshot(snapshot_path)
    policy = SamplingPolicy(snapshot.sampling_policy_version, snapshot.sampling_seed)
    document = select_hidden_real(snapshot, policy).document()
    document["snapshot_hash"] = "0" * 64
    freeze_frame = tmp_path / "bad-freeze-frame.json"
    freeze_frame.write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setattr(
        cli,
        "load_private_sampling_annotations",
        lambda *args, **kwargs: pytest.fail("invalid freeze frame must stop before annotations"),
    )

    with pytest.raises(ValueError, match="does not match"):
        cli.main(_arguments(
            snapshot_path,
            annotations_path,
            freeze_frame,
            receipt_path,
            tmp_path / "gold-manifest.json",
            tmp_path / "provenance.json",
        ))


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
