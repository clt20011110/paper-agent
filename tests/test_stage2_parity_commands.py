from __future__ import annotations

import json
from pathlib import Path

from paper_agent.cli import build_parser
from paper_agent.stage2_commands import freeze_stage2_parity_workload
from paper_agent.stage2_parity import freeze_parity_workload
from paper_agent.stage2_pipeline import Stage2Paper


def _papers() -> tuple[Stage2Paper, ...]:
    return tuple(
        Stage2Paper(f"paper-{index:05d}", f"Paper {index}", None, ("keyword",))
        for index in range(10_000)
    )


def test_freeze_parity_workload_dry_run_validates_receipt_without_writing(
    tmp_path: Path, monkeypatch
) -> None:
    papers = _papers()
    workload = freeze_parity_workload(
        papers, topic="topic", language="en", query_version="v1", query="query"
    )
    receipt = {
        "parity": {
            "paper_count": 10_000,
            "paper_ids": sorted(pair.paper_id for pair in workload.pairs),
            "papers_corpus_hash": workload.corpus_hash(),
        }
    }
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    output = tmp_path / "workload.json"
    monkeypatch.setattr("paper_agent.stage2_commands._benchmark_papers", lambda _: papers)

    result = freeze_stage2_parity_workload(
        papers_path=tmp_path / "papers.json",
        selection_receipt_path=receipt_path,
        topic="topic",
        language="en",
        query_version="v1",
        query="query",
        output_path=output,
        dry_run=True,
    )

    assert result["status"] == "validated"
    assert result["workload_hash"] == workload.hash()
    assert not output.exists()


def test_stage2_parity_cli_has_only_frozen_inputs() -> None:
    args = build_parser().parse_args([
        "stage2-parity", "run",
        "--workload", "workload.json",
        "--selection-receipt", "receipt.json",
        "--oracle-stage2-candidate", "oracle.json",
        "--oracle-model-lock", "oracle-lock.json",
        "--stage2-candidate", "candidate.json",
        "--candidate-model-lock", "candidate-lock.json",
        "--manifest-output", "manifest.json",
        "--scores-output", "scores.json",
    ])

    assert args.command == "stage2-parity"
    assert args.stage2_parity_command == "run"
    assert not hasattr(args, "endpoint")
    assert not hasattr(args, "concurrency")
    assert not hasattr(args, "model")
