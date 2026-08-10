from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from paper_agent import cli
from paper_agent.domain import FilterStatus, SourceEntry
from paper_agent.repository import PaperRepository
from paper_agent.stage2_commands import evaluate_benchmark_artifacts, filter_database
from paper_agent.stage2_evaluation import PerformanceCase, PerformanceRoutingManifest
from paper_agent.storage import Database


class _FakeScreener:
    run_ids = ["stage2-test"]

    def screen(self, paper_ids):
        return {
            paper_id: (
                FilterStatus.RELEVANT
                if index == 0
                else FilterStatus.NEEDS_REVIEW
            )
            for index, paper_id in enumerate(paper_ids)
        }


class _FakeRelease:
    profile_name = "released-small-model"
    release_hash = "a" * 64

    def screener(self, database, campaign_id):
        assert campaign_id == "campaign-1"
        return _FakeScreener()


def test_filter_database_selects_canonical_papers_and_dry_run_is_read_only(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "papers.sqlite3"
    with Database(database_path) as database:
        database.migrate()
        repository = PaperRepository(database)
        first = repository.ingest(SourceEntry("fixture", "1", "First paper"))
        second = repository.ingest(SourceEntry("fixture", "2", "Second paper"))
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({"plan_id": "plan-1"}), encoding="utf-8")
    release_path = tmp_path / "release.json"
    release_path.write_text("{}", encoding="utf-8")
    loader = lambda path, plan: _FakeRelease()

    preview = filter_database(
        plan_path=plan_path,
        release_path=release_path,
        database_path=database_path,
        campaign_id="campaign-1",
        paper_ids=(second.paper_id, first.paper_id, second.paper_id),
        dry_run=True,
        release_loader=loader,
    )
    assert preview["paper_ids"] == sorted((first.paper_id, second.paper_id))
    assert preview["status"] == "validated"

    result = filter_database(
        plan_path=plan_path,
        release_path=release_path,
        database_path=database_path,
        campaign_id="campaign-1",
        release_loader=loader,
    )
    assert result["counts"] == {"needs_review": 1, "relevant": 1}
    assert result["stage2_run_ids"] == ["stage2-test"]


def _benchmark_files(tmp_path: Path) -> tuple[Path, Path]:
    cases = tuple(
        PerformanceCase(f"perf-{index}", 100, index < 100)
        for index in range(1_000)
    )
    manifest = PerformanceRoutingManifest(
        1,
        "corpus",
        "config",
        ("reranker", "qwen"),
        ("reranker-threshold", "qwen-threshold"),
        256,
        cases,
        frozenset(case.pair_id for case in cases[:150]),
        frozenset(case.pair_id for case in cases[:300]),
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.document()), encoding="utf-8")
    ids = [case.pair_id for case in cases]
    environment = {
        "machine_model": "Apple Silicon M4 Max",
        "memory_gb": 36,
        "macos_version": "15.6",
        "omlx_version": "0.5.7",
        "mlx_version": "0.32.0",
        "power_mode": "AC",
        "background_load": "idle",
        "batch_config": {"rerank_batch": 32, "qwen_concurrency": 4},
        "resident_model_instances": {"reranker": 1, "qwen": 1},
    }
    records = []
    for scenario in ("normal", "stress"):
        for index in range(3):
            records.append({
                "scenario": scenario,
                "run_id": f"{scenario}-{index}",
                "manifest_hash": manifest.hash(),
                "stage2_config_hash": "config",
                "model_lock_hashes": ["reranker", "qwen"],
                "duration_seconds": 800 if scenario == "normal" else 1200,
                "p50_seconds": 0.5,
                "p95_seconds": 1.5,
                "peak_memory_gb": 24,
                "request_count": 1000,
                "failed_request_count": 0,
                "completed_pair_ids": ids,
                "needs_review_pair_ids": [],
                "failed_request_pair_ids": [],
                "qwen_pair_ids": sorted(
                    manifest.normal_qwen_ids
                    if scenario == "normal"
                    else manifest.stress_qwen_ids
                ),
                "environment": environment,
                "executed_components": list(manifest.pipeline_components),
                "sqlite_commit_count": 1000,
                "warmed": True,
            })
    records_path = tmp_path / "records.json"
    records_path.write_text(json.dumps(records), encoding="utf-8")
    return manifest_path, records_path


def test_benchmark_stage2_cli_applies_frozen_performance_gate(
    tmp_path: Path, capsys
) -> None:
    manifest, records = _benchmark_files(tmp_path)
    result = evaluate_benchmark_artifacts(
        manifest_path=manifest, record_paths=(records,)
    )
    assert result["status"] == "passed"
    assert result["performance"]["record_count"] == 6

    assert cli.main([
        "benchmark-stage2",
        "--manifest",
        str(manifest),
        "--record",
        str(records),
        "--run-id",
        "benchmark-1",
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["run_id"] == "benchmark-1"
    assert payload["stage"] == "stage2"
    assert payload["status"] == "passed"


def test_stage2_parser_surface() -> None:
    parser = cli.build_parser()
    filtered = parser.parse_args([
        "filter",
        "--plan",
        "plan.json",
        "--stage2-release",
        "release.json",
        "--database",
        "papers.sqlite3",
    ])
    benchmark = parser.parse_args([
        "benchmark-stage2",
        "--manifest",
        "manifest.json",
        "--record",
        "record.json",
    ])
    assert (filtered.command, benchmark.command) == ("filter", "benchmark-stage2")
