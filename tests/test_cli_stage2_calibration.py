from __future__ import annotations

import json
from pathlib import Path

import pytest

from paper_agent import cli
from paper_agent.stage2_commands import build_stage2_candidate, freeze_stage2_dev_scores
from paper_agent.stage2_dev_calibration import write_frozen_dev_raw_scores
from paper_agent.stage2_evaluation import write_gold_manifest
from paper_agent.stage2_sampling import write_private_corpus_snapshot

from test_stage2_candidate import (
    ADJUDICATOR_LOCK,
    RERANKER_LOCK,
    _labels,
    _raw_scores,
    _runtime,
)
from test_stage2_dev_calibration import _inputs


def _files(tmp_path: Path) -> dict[str, Path]:
    manifest, snapshot = _inputs()
    manifest_path = tmp_path / "gold-manifest.json"
    snapshot_path = tmp_path / "private-snapshot.json"
    runtime_path = tmp_path / "runtime.json"
    topic_queries_path = tmp_path / "topic-queries.json"
    write_gold_manifest(manifest_path, manifest)
    write_private_corpus_snapshot(snapshot_path, snapshot)
    runtime_path.write_text(json.dumps(_runtime()), encoding="utf-8")
    topic_rows: dict[str, dict[str, str]] = {}
    for pair in manifest.pairs:
        if pair.split.value != "dev":
            continue
        topic_rows.setdefault(pair.topic, {})[pair.language] = (
            f"query for {pair.topic} in {pair.language}"
        )
    topic_queries_path.write_text(json.dumps({
        "topics": [
            {
                "id": topic,
                "queries": [
                    {"language": language, "query": query}
                    for language, query in sorted(queries.items())
                ],
            }
            for topic, queries in sorted(topic_rows.items())
        ],
    }), encoding="utf-8")
    return {
        "manifest": manifest_path,
        "snapshot": snapshot_path,
        "runtime": runtime_path,
        "topic_queries": topic_queries_path,
    }


def test_freeze_dev_scores_dry_run_validates_without_model_calls_or_output(
    tmp_path: Path,
) -> None:
    paths = _files(tmp_path)
    output = tmp_path / "raw-scores.json"

    result = freeze_stage2_dev_scores(
        manifest_path=paths["manifest"],
        snapshot_path=paths["snapshot"],
        topic_queries_path=paths["topic_queries"],
        runtime_path=paths["runtime"],
        reranker_lock_path=RERANKER_LOCK,
        adjudicator_lock_path=ADJUDICATOR_LOCK,
        output_path=output,
        dry_run=True,
    )

    assert result["case_count"] == 300
    assert result["topic_query_count"] == 6
    assert result["status"] == "validated"
    assert result["written"] is False
    assert not output.exists()


def test_build_candidate_dry_run_consumes_private_labels_but_writes_no_bundle(
    tmp_path: Path,
) -> None:
    paths = _files(tmp_path)
    manifest, snapshot = _inputs()
    labels = _labels(manifest)
    private_labels_path = tmp_path / "private-labels.json"
    private_labels_path.write_text(json.dumps({
        "schema_version": "1",
        "gold_manifest_hash": manifest.hash(),
        "annotation_artifact_hash": labels.annotation_artifact_hash,
        "labels": [
            {"pair_id": pair_id, "label": labels.labels[pair_id]}
            for pair_id in sorted(labels.labels)
        ],
        "hard_negative_pair_ids": sorted(labels.hard_negative_pair_ids),
        "hard_positive_pair_ids": sorted(labels.hard_positive_pair_ids),
    }), encoding="utf-8")
    raw_scores_path = tmp_path / "raw-scores.json"
    write_frozen_dev_raw_scores(
        raw_scores_path,
        _raw_scores(manifest, snapshot, labels, _runtime()),
    )
    output_dir = tmp_path / "candidate"

    result = build_stage2_candidate(
        manifest_path=paths["manifest"],
        private_labels_path=private_labels_path,
        raw_scores_path=raw_scores_path,
        runtime_path=paths["runtime"],
        reranker_lock_path=RERANKER_LOCK,
        adjudicator_lock_path=ADJUDICATOR_LOCK,
        candidate_id="crossref-dev-v1",
        output_dir=output_dir,
        dry_run=True,
    )

    assert result["candidate_id"] == "crossref-dev-v1"
    assert result["status"] == "validated"
    assert result["written"] is False
    assert not output_dir.exists()


@pytest.mark.parametrize(
    ("subcommand", "function_name", "result"),
    (
        (
            "freeze-dev-scores",
            "freeze_stage2_dev_scores",
            {"command": "stage2-calibration.freeze-dev-scores", "status": "validated"},
        ),
        (
            "build-candidate",
            "build_stage2_candidate",
            {"command": "stage2-calibration.build-candidate", "status": "validated"},
        ),
    ),
)
def test_stage2_calibration_cli_dispatches_both_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    subcommand: str,
    function_name: str,
    result: dict[str, str],
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        cli,
        function_name,
        lambda **kwargs: calls.append(kwargs) or result,
    )
    common = [
        "--gold-manifest", str(tmp_path / "manifest.json"),
        "--runtime", str(tmp_path / "runtime.json"),
        "--reranker-lock", str(RERANKER_LOCK),
        "--adjudicator-lock", str(ADJUDICATOR_LOCK),
    ]
    specific = (
        [
            "--private-snapshot", str(tmp_path / "snapshot.json"),
            "--topic-queries", str(tmp_path / "queries.json"),
            "--output", str(tmp_path / "scores.json"),
        ]
        if subcommand == "freeze-dev-scores"
        else [
            "--private-labels", str(tmp_path / "labels.json"),
            "--raw-scores", str(tmp_path / "scores.json"),
            "--candidate-id", "candidate-v1",
            "--output-dir", str(tmp_path / "candidate"),
        ]
    )

    assert cli.main(["--dry-run", "stage2-calibration", subcommand, *common, *specific]) == 0

    assert calls[0]["dry_run"] is True
    assert json.loads(capsys.readouterr().out)["status"] == "validated"
