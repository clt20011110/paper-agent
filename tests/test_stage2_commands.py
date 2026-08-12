from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from paper_agent import cli
from paper_agent.domain import FilterStatus, SourceEntry
from paper_agent.repository import PaperRepository
from paper_agent.stage2_backends import OmlxResponse, ThresholdArtifact
from paper_agent.stage2_benchmark import MacOSMemoryObserver
from paper_agent.stage2_commands import (
    _measurement_result,
    benchmark_corpus_hash,
    evaluate_benchmark_artifacts,
    filter_database,
    measure_stage2_benchmark,
    run_structured_replay,
)
from paper_agent.stage2_evaluation import PerformanceCase, PerformanceRoutingManifest
from paper_agent.stage2_pipeline import (
    ADJUDICATOR_SHARE_ALARM,
    ERROR_RATE_ALARM,
    MEMORY_WATERMARK_ALARM,
    Stage2Paper,
    Stage2Profile,
)
from paper_agent.stage2_prompt_contract import (
    adjudication_messages,
    estimate_omlx_chat_input_token_proxy,
)
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

    def telemetry(self):
        return {
            "stage2_run_ids": list(self.run_ids),
            "screened_count": 2,
            "reranked_count": 2,
            "adjudicator_count": 1,
            "adjudicator_share": 0.5,
            "adjudicator_capacity": "severe",
            "error_count": 0,
            "error_rate": 0.0,
            "alarm_codes": [ADJUDICATOR_SHARE_ALARM],
            "run_details": [],
        }


class _FakeRelease:
    profile_name = "released-small-model"
    release_hash = "a" * 64

    def screener(self, database, campaign_id):
        assert campaign_id == "campaign-1"
        return _FakeScreener()


def test_benchmark_corpus_hash_binds_normalized_paper_content() -> None:
    first = Stage2Paper("paper-1", "First", "Abstract", ("keyword",))
    second = Stage2Paper("paper-2", "Second", None)

    assert benchmark_corpus_hash((first, second)) == benchmark_corpus_hash(
        (second, first)
    )
    assert benchmark_corpus_hash((first, second)) != benchmark_corpus_hash(
        (Stage2Paper("paper-1", "First", "Changed", ("keyword",)), second)
    )


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
    assert preview["stage2"] is None
    assert preview["alarm_codes"] == []

    result = filter_database(
        plan_path=plan_path,
        release_path=release_path,
        database_path=database_path,
        campaign_id="campaign-1",
        paper_ids=None,
        release_loader=loader,
    )
    assert result["counts"] == {"needs_review": 1, "relevant": 1}
    assert result["paper_count"] == 2
    assert result["stage2_run_ids"] == ["stage2-test"]
    assert result["stage2"]["adjudicator_share"] == 0.5
    assert result["alarm_codes"] == [ADJUDICATOR_SHARE_ALARM]
    assert result["status"] == "complete"

    class ErrorRelease(_FakeRelease):
        def screener(self, database, campaign_id):
            screener = _FakeScreener()
            screener.telemetry = lambda: {
                **_FakeScreener.telemetry(screener),
                "error_count": 1,
                "error_rate": 0.5,
                "alarm_codes": [ERROR_RATE_ALARM],
            }
            return screener

    failed = filter_database(
        plan_path=plan_path,
        release_path=release_path,
        database_path=database_path,
        campaign_id="campaign-1",
        release_loader=lambda path, plan: ErrorRelease(),
    )
    assert failed["status"] == "incomplete"
    assert failed["alarm_codes"] == [ERROR_RATE_ALARM]

    empty_preview = filter_database(
        plan_path=plan_path,
        release_path=release_path,
        database_path=database_path,
        campaign_id="campaign-1",
        paper_ids=(),
        dry_run=True,
        release_loader=loader,
    )
    assert empty_preview["paper_count"] == 0
    assert empty_preview["paper_ids"] == []


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
                "record_version": 2,
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
                "service_request_count": 1000,
                "service_failed_request_count": 0,
                "resume_verified": True,
                "resume_model_call_count": 0,
                "resumed_pair_count": 1000,
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


def test_benchmark_stage2_rejects_legacy_records_without_service_metrics(
    tmp_path: Path,
) -> None:
    manifest, records = _benchmark_files(tmp_path)
    documents = json.loads(records.read_text(encoding="utf-8"))
    for document in documents:
        document.pop("record_version")
        document.pop("service_request_count")
        document.pop("service_failed_request_count")
        document.pop("resume_verified")
        document.pop("resume_model_call_count")
        document.pop("resumed_pair_count")
    records.write_text(json.dumps(documents), encoding="utf-8")

    with pytest.raises(ValueError, match="Legacy records.*must be rerun"):
        evaluate_benchmark_artifacts(
            manifest_path=manifest, record_paths=(records,)
        )


def test_benchmark_stage2_measure_cli_dispatches_explicit_production_inputs(
    monkeypatch, capsys
) -> None:
    captured = {}

    def measure(**kwargs):
        captured.update(kwargs)
        return {"command": "benchmark-stage2.measure", "status": "validated"}

    monkeypatch.setattr(cli, "measure_stage2_benchmark", measure)

    exit_code = cli.main([
        "benchmark-stage2",
        "measure",
        "--manifest", "manifest.json",
        "--papers", "papers.json",
        "--stage2-candidate", "candidate.json",
        "--environment", "environment.json",
        "--database", "benchmark.sqlite3",
        "--output", "record.json",
        "--scenario", "stress",
        "--omlx-pid", "101",
        "--omlx-pid", "102",
        "--run-id", "stress-1",
        "--dry-run",
    ])

    assert exit_code == 0
    assert captured["run_id"] == "stress-1"
    assert captured["omlx_pids"] == [101, 102]
    assert captured["scenario"] == "stress"
    assert captured["dry_run"] is True
    assert json.loads(capsys.readouterr().out)["status"] == "validated"


def test_benchmark_freeze_manifests_cli_uses_only_frozen_inputs(
    monkeypatch, capsys,
) -> None:
    captured = {}

    def freeze(**kwargs):
        captured.update(kwargs)
        return {
            "command": "benchmark-stage2.freeze-manifests",
            "status": "validated",
        }

    monkeypatch.setattr(cli, "freeze_stage2_benchmark_manifests", freeze)
    exit_code = cli.main([
        "--dry-run",
        "benchmark-stage2",
        "freeze-manifests",
        "--stage2-candidate", "candidate.json",
        "--performance-papers", "performance-papers.json",
        "--soak-papers", "soak-papers.json",
        "--selection-receipt", "selection-receipt.json",
        "--performance-output", "performance-manifest.json",
        "--soak-output", "soak-manifest.json",
    ])

    assert exit_code == 0
    assert captured == {
        "candidate_path": Path("candidate.json"),
        "performance_papers_path": Path("performance-papers.json"),
        "soak_papers_path": Path("soak-papers.json"),
        "selection_receipt_path": Path("selection-receipt.json"),
        "performance_output": Path("performance-manifest.json"),
        "soak_output": Path("soak-manifest.json"),
        "dry_run": True,
    }
    assert json.loads(capsys.readouterr().out)["status"] == "validated"


def test_hidden_prediction_cli_uses_candidate_and_sealed_snapshot(
    monkeypatch, capsys,
) -> None:
    captured = {}

    def predict(**kwargs):
        captured.update(kwargs)
        return {
            "command": "stage2-evaluator.predict-hidden",
            "status": "validated",
        }

    monkeypatch.setattr(cli, "build_hidden_promotion_submission", predict)
    exit_code = cli.main([
        "--dry-run",
        "stage2-evaluator",
        "predict-hidden",
        "--manifest", "gold-manifest.json",
        "--private-snapshot", "private-snapshot.json",
        "--stage2-candidate", "candidate.json",
        "--output", "submission.json",
    ])

    assert exit_code == 0
    assert captured == {
        "manifest_path": Path("gold-manifest.json"),
        "snapshot_path": Path("private-snapshot.json"),
        "candidate_path": Path("candidate.json"),
        "output_path": Path("submission.json"),
        "dry_run": True,
    }
    assert json.loads(capsys.readouterr().out)["status"] == "validated"


def test_stage2_replay_dry_run_and_measured_execution(
    tmp_path: Path,
) -> None:
    profile = Stage2Profile(
        query="frozen replay topic",
        query_version="replay-v1",
        thresholds=ThresholdArtifact(
            "threshold-v1", "reranker-lock", "raw_reranker_score", -1, 1
        ),
        reranker_model_id="reranker",
        reranker_revision="reranker-revision",
        adjudicator_model_id="qwen",
        adjudicator_revision="qwen-revision",
        screening_scope_hash="0" * 64,
        reranker_lock_hash="a" * 64,
        adjudicator_lock_hash="b" * 64,
        adjudicator_concurrency=4,
    )
    papers_path = tmp_path / "papers.json"
    papers_path.write_text(json.dumps({
        "schema_version": "1",
        "kind": "stage2_benchmark_papers",
        "papers": [
            {
                "paper_id": f"paper-{index:04d}",
                "title": f"Paper {index}",
                "abstract": "Frozen abstract",
                "keywords": [],
            }
            for index in range(1_000)
        ],
    }), encoding="utf-8")
    release = SimpleNamespace(
        profile=profile,
        omlx_base_url="http://127.0.0.1:8000",
        api_key_env=None,
    )
    candidate_loader = lambda _path: release
    manifest_path = tmp_path / "replay-manifest.json"
    records_path = tmp_path / "replay-records.json"

    preview = run_structured_replay(
        papers_path=papers_path,
        candidate_path=tmp_path / "candidate.json",
        manifest_output=manifest_path,
        records_output=records_path,
        dry_run=True,
        candidate_loader=candidate_loader,
    )
    assert preview["status"] == "validated"
    assert preview["case_count"] == 1_000
    assert not manifest_path.exists() and not records_path.exists()

    class Transport:
        calls = 0

        def request(self, path, payload):
            assert path == "/v1/chat/completions"
            self.calls += 1
            paper_id = payload["messages"][1]["content"].split("Paper ID: ", 1)[1].split("\n", 1)[0]
            decision = {
                "paper_id": paper_id,
                "decision": "relevant",
                "score": 0.9,
                "reason_codes": ["topic_match"],
                "rationale": "Directly relevant.",
                "evidence_fields": ["title", "abstract"],
            }
            return OmlxResponse(200, json.dumps({
                "model": "qwen",
                "choices": [{"message": {"content": json.dumps(decision)}}],
            }).encode())

    transport = Transport()
    result = run_structured_replay(
        papers_path=papers_path,
        candidate_path=tmp_path / "candidate.json",
        manifest_output=manifest_path,
        records_output=records_path,
        candidate_loader=candidate_loader,
        transport=transport,
    )

    assert result["status"] == "passed"
    assert result["first_valid_rate"] == 1
    assert result["deterministic_repairs"] == 0
    assert result["model_retries"] == 0
    assert len(result["manifest_sha256"]) == len(result["records_sha256"]) == 64
    assert transport.calls == 1_000
    assert manifest_path.is_file() and records_path.is_file()


def test_stage2_replay_cli_uses_only_frozen_inputs(monkeypatch, capsys) -> None:
    captured = {}

    def replay(**kwargs):
        captured.update(kwargs)
        return {"command": "stage2-replay", "status": "validated"}

    monkeypatch.setattr(cli, "run_structured_replay", replay)
    exit_code = cli.main([
        "--dry-run",
        "stage2-replay",
        "--papers", "papers.json",
        "--stage2-candidate", "candidate.json",
        "--manifest-output", "manifest.json",
        "--records-output", "records.json",
    ])

    assert exit_code == 0
    assert captured == {
        "papers_path": Path("papers.json"),
        "candidate_path": Path("candidate.json"),
        "manifest_output": Path("manifest.json"),
        "records_output": Path("records.json"),
        "dry_run": True,
    }
    output = json.loads(capsys.readouterr().out)
    assert output["stage"] == "stage2"
    assert output["status"] == "validated"


def test_measure_stage2_dry_run_validates_release_workload_and_macos_observation(
    tmp_path: Path,
) -> None:
    profile = Stage2Profile(
        query="frozen benchmark topic",
        query_version="benchmark-v1",
        thresholds=ThresholdArtifact(
            "threshold-v1", "reranker-lock", "raw_reranker_score", -1, 1
        ),
        reranker_model_id="reranker",
        reranker_revision="reranker-revision",
        adjudicator_model_id="qwen",
        adjudicator_revision="qwen-revision",
        screening_scope_hash="0" * 64,
        reranker_lock_hash="reranker-lock",
        adjudicator_lock_hash="qwen-lock",
        document_batch_size=32,
        reranker_max_in_flight=2,
        adjudicator_concurrency=4,
    )
    papers = tuple(
        Stage2Paper(
            f"paper-{index}",
            f"Paper {index}",
            None if index < 100 else "Frozen abstract",
        )
        for index in range(1_000)
    )
    cases = tuple(
        PerformanceCase(
            paper.paper_id,
            estimate_omlx_chat_input_token_proxy(adjudication_messages(
                query_version=profile.query_version,
                query=profile.query,
                paper=paper,
            )),
            paper.abstract is None,
        )
        for paper in papers
    )
    manifest = PerformanceRoutingManifest(
        1,
        benchmark_corpus_hash(papers),
        profile.base_runtime_config_hash,
        ("reranker-lock", "qwen-lock"),
        (profile.threshold_hash,),
        256,
        cases,
        frozenset(case.pair_id for case in cases[:150]),
        frozenset(case.pair_id for case in cases[:300]),
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.document()), encoding="utf-8")
    papers_path = tmp_path / "papers.json"
    papers_path.write_text(json.dumps([
        {
            "paper_id": paper.paper_id,
            "title": paper.title,
            "abstract": paper.abstract,
            "keywords": [],
        }
        for paper in papers
    ]), encoding="utf-8")
    environment_path = tmp_path / "environment.json"
    environment_path.write_text(json.dumps({
        "machine_model": "Apple Silicon M4 Max",
        "memory_gb": 36,
        "macos_version": "15.6",
        "omlx_version": "0.5.7",
        "mlx_version": "0.32.0",
        "power_mode": "AC",
        "background_load": "idle",
        "batch_config": {
            "document_batch_size": 32,
            "reranker_max_in_flight": 2,
            "adjudicator_concurrency": 4,
        },
        "resident_model_instances": {"reranker-lock": 1, "qwen-lock": 1},
    }), encoding="utf-8")
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text("{}", encoding="utf-8")
    database_path = tmp_path / "benchmark.sqlite3"
    with Database(database_path) as database:
        database.migrate()

    def memory_command(arguments) -> str:
        if arguments[0] == "/bin/ps":
            return "10 1024\n20 2048\n"
        if arguments[0] == "/usr/sbin/sysctl":
            return "1\n"
        if arguments[0] == "/usr/sbin/system_profiler":
            return json.dumps({
                "SPHardwareDataType": [{
                    "chip_type": "Apple M4 Max",
                    "physical_memory": "36 GB",
                }]
            })
        return "15.6\n"

    observer = MacOSMemoryObserver(
        runner_pid=10,
        omlx_pids=(20,),
        command_runner=memory_command,
        platform_name="darwin",
    )
    release = SimpleNamespace(
        profile=profile,
        release_hash="release-hash",
        omlx_base_url="http://127.0.0.1:8000",
        api_key_env=None,
        reranker_fallback=None,
    )

    result = measure_stage2_benchmark(
        manifest_path=manifest_path,
        papers_path=papers_path,
        candidate_path=candidate_path,
        environment_path=environment_path,
        database_path=database_path,
        output_path=tmp_path / "normal-1.json",
        scenario="normal",
        run_id="normal-1",
        omlx_pids=(20,),
        dry_run=True,
        candidate_loader=lambda path: release,
        memory_observer=observer,
    )

    assert result["status"] == "validated"
    assert result["case_count"] == 1_000
    assert result["rss_scope"].endswith("omlx_pids=20")
    assert not (tmp_path / "normal-1.json").exists()


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
    measured = parser.parse_args([
        "--run-id", "normal-1",
        "benchmark-stage2", "measure",
        "--manifest", "manifest.json",
        "--papers", "papers.json",
        "--stage2-config", "candidate.json",
        "--environment", "environment.json",
        "--database", "benchmark.sqlite3",
        "--output", "normal-1.json",
        "--scenario", "normal",
        "--omlx-pid", "101",
    ])
    assert (filtered.command, benchmark.command) == ("filter", "benchmark-stage2")
    assert measured.benchmark_command == "measure"


@pytest.mark.parametrize(
    ("alarm_codes", "expected_status"),
    (
        ([ADJUDICATOR_SHARE_ALARM], "complete"),
        ([ERROR_RATE_ALARM], "incomplete"),
        ([MEMORY_WATERMARK_ALARM], "incomplete"),
    ),
)
def test_measure_result_exposes_stage2_alarms_and_operational_status(
    tmp_path, alarm_codes, expected_status
) -> None:
    document = {
        "alarm_codes": alarm_codes,
        "case_count": 200,
        "kind": "performance",
        "manifest_hash": "manifest",
        "record_version": 2,
        "adjudicator_capacity": "warning",
        "adjudicator_count": 31,
        "adjudicator_share": 0.155,
        "qwen_capacity_level": "warning",
        "qwen_count": 31,
        "qwen_share": 0.155,
        "request_failure_rate": 0.005,
        "peak_memory_gb": 29,
        "memory_pressure_critical": False,
        "unbounded_memory_growth": False,
        "rss_scope": "fixture",
        "run_id": "measured-run",
        "scenario": "normal",
        "service_request_count": 200,
        "service_failed_request_count": 1,
        "service_request_failure_rate": 0.005,
    }
    record = SimpleNamespace(document=lambda: document, hash=lambda: "artifact-hash")

    result = _measurement_result(record, tmp_path / "record.json")

    assert result["alarm_codes"] == alarm_codes
    assert result["adjudicator_count"] == 31
    assert result["service_request_failure_rate"] == 0.005
    assert result["peak_memory_gb"] == 29
    assert result["status"] == expected_status
