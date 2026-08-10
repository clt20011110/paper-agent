from __future__ import annotations

from contextlib import closing
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest
import yaml

from paper_agent import cli
from paper_agent.report_input_service import ReportInputRequest
from paper_agent.storage import Database


ROOT = Path(__file__).parents[1]


def _payload(capsys) -> dict[str, object]:
    return json.loads(capsys.readouterr().out)


def test_report_execution_cli_wires_frozen_inputs_and_processing_grants(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    database_path = tmp_path / "papers.sqlite3"
    with Database(database_path) as database:
        database.migrate()
    approved = tmp_path / "report-plan" / "REPORT_PLAN.json"
    approved.parent.mkdir()
    approved.write_text('{"plan_id":"plan-1"}', encoding="utf-8")
    (approved.parent / "CORPUS_SNAPSHOT.json").write_text("{}", encoding="utf-8")
    (approved.parent / "SEARCH_AUDIT.json").write_text("{}", encoding="utf-8")
    grants = tmp_path / "grants.json"
    grants.write_text(
        json.dumps({"schema_version": "1", "grants": {"a" * 64: "grant-1"}}),
        encoding="utf-8",
    )
    output_root = tmp_path / "output"
    config_document = yaml.safe_load(
        (ROOT / "configs" / "abstract_focus.yaml").read_text(encoding="utf-8")
    )
    config_document["storage"]["sqlite_path"] = str(database_path)
    config_document["project"]["output_dir"] = str(output_root)
    config_document["summary"]["enabled"] = True
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(config_document, sort_keys=False), encoding="utf-8"
    )
    captured = {}

    class FakeService:
        def __init__(self, database, artifact_store, gate, report_store, **options):
            captured.update({
                "database": database,
                "artifact_store": artifact_store,
                "gate": gate,
                "report_store": report_store,
                "options": options,
            })

        def run(self, report_run_id, pipeline_run_id, bundle, **options):
            captured.update({
                "report_run_id": report_run_id,
                "pipeline_run_id": pipeline_run_id,
                "bundle": bundle,
                "run_options": options,
            })
            return SimpleNamespace(
                report_run_id=report_run_id,
                status="validated",
                dry_run=True,
                audit=None,
            )

    monkeypatch.setattr(cli, "ReportExecutionService", FakeService)
    monkeypatch.setattr(
        cli,
        "assert_report_plan_resource_binding",
        lambda plan, resources: captured.update({
            "resource_plan": plan,
            "resources": resources,
        }),
    )
    assert cli.main([
        "report",
        "--config", str(config_path),
        "--plan", str(approved),
        "--processing-grants", str(grants),
        "--report-run-id", "report-1",
        "--dry-run",
    ]) == 0

    result = _payload(capsys)
    assert result["status"] == "validated"
    assert captured["report_run_id"] == "report-1"
    assert captured["pipeline_run_id"] == "report-1:stage4b"
    assert captured["run_options"]["processing_grants"] == {"a" * 64: "grant-1"}
    assert captured["run_options"]["dry_run"] is True
    assert captured["options"]["execution_mode"] == "attended"
    assert captured["options"]["runtime_config"].enabled
    assert captured["options"]["runtime_config"].resources is captured["resources"]


def test_disabled_report_cli_skips_before_opening_any_inputs(
    tmp_path: Path, capsys
) -> None:
    config_document = yaml.safe_load(
        (ROOT / "configs" / "abstract_focus.yaml").read_text(encoding="utf-8")
    )
    config_document["storage"]["sqlite_path"] = str(tmp_path / "missing.sqlite3")
    config_document["project"]["output_dir"] = str(tmp_path / "missing-output")
    config_path = tmp_path / "disabled.yaml"
    config_path.write_text(
        yaml.safe_dump(config_document, sort_keys=False), encoding="utf-8"
    )

    assert cli.main([
        "report",
        "--config", str(config_path),
        "--plan", str(tmp_path / "missing-plan.json"),
        "--policy", str(tmp_path / "missing-policy.yaml"),
    ]) == 0

    result = _payload(capsys)
    assert result["status"] == "complete"
    assert result["skipped"] is True
    assert not (tmp_path / "missing.sqlite3").exists()


def test_report_prepare_inputs_cli_is_read_only_in_dry_run(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    database_path = tmp_path / "papers.sqlite3"
    with Database(database_path) as database:
        database.migrate()
    captured = {}

    class FakeService:
        def __init__(self, database, artifact_store, output_root):
            captured.update({
                "database": database,
                "artifact_root": artifact_store.root,
                "output_root": output_root,
            })

        def build(self, request, *, save_bundle):
            captured.update({"request": request, "save_bundle": save_bundle})
            return SimpleNamespace(
                bundle_id="report-input-fixture",
                directory=tmp_path / "output" / "reports" / "inputs" / "report-input-fixture",
                corpus_snapshot_path=tmp_path / "output" / "CORPUS_SNAPSHOT.json",
                search_audit_path=tmp_path / "output" / "SEARCH_AUDIT.json",
                corpus_snapshot={"snapshot_hash": "a" * 64},
                search_audit={"pack_hash": "b" * 64},
                saved=save_bundle,
            )

    monkeypatch.setattr(cli, "ReportInputService", FakeService)
    output_root = tmp_path / "output"
    assert cli.main([
        "--dry-run",
        "report",
        "prepare-inputs",
        "--database", str(database_path),
        "--artifact-root", str(tmp_path / "store"),
        "--output-root", str(output_root),
        "--crawl-run-id", "crawl-1",
        "--filter-run-id", "filter-1",
        "--stage4-run-id", "stage4-1",
        "--recent-cutoff", "2024-01-01",
        "--created-at", "2026-08-11T00:00:00Z",
        "--include-needs-review",
    ]) == 0

    result = _payload(capsys)
    assert result["command"] == "report.prepare-inputs"
    assert result["status"] == "validated"
    assert result["write_performed"] is False
    assert captured["database"].read_only is True
    assert captured["artifact_root"] == tmp_path / "store"
    assert captured["output_root"] == output_root
    assert captured["request"] == ReportInputRequest(
        crawl_run_id="crawl-1",
        filter_run_id="filter-1",
        stage4_run_id="stage4-1",
        recent_cutoff="2024-01-01",
        created_at="2026-08-11T00:00:00Z",
        include_needs_review=True,
    )
    assert captured["save_bundle"] is False


def test_report_plan_and_approval_use_resources_from_config(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    config_document = yaml.safe_load(
        (ROOT / "configs" / "abstract_focus.yaml").read_text(encoding="utf-8")
    )
    config_document["summary"]["enabled"] = True
    config_path = tmp_path / "enabled.yaml"
    config_path.write_text(
        yaml.safe_dump(config_document, sort_keys=False), encoding="utf-8"
    )
    captured = {}

    def fake_compile(*_args, **options):
        captured["compile_resources"] = options["resources"]
        return SimpleNamespace(
            path=tmp_path / "REPORT_PLAN.draft.json",
            plan={"plan_id": "plan-1", "plan_hash": "a" * 64},
            saved=False,
        )

    def fake_approve(*_args, **options):
        captured["approve_resources"] = options["resources"]
        return SimpleNamespace(
            path=tmp_path / "REPORT_PLAN.json",
            plan={"plan_id": "plan-1", "plan_hash": "a" * 64},
            saved=False,
        )

    monkeypatch.setattr(cli, "compile_report_plan_from_files", fake_compile)
    monkeypatch.setattr(cli, "approve_report_plan_from_files", fake_approve)
    missing = tmp_path / "intentionally-unread.json"

    assert cli.main([
        "--config", str(config_path),
        "--dry-run",
        "report",
        "--plan-only",
        "--draft", str(missing),
        "--corpus-snapshot", str(missing),
        "--search-audit", str(missing),
        "--output-root", str(tmp_path),
    ]) == 0
    _payload(capsys)
    assert cli.main([
        "--config", str(config_path),
        "--dry-run",
        "report",
        "approve",
        "--plan", str(missing),
        "--hash", "a" * 64,
        "--approved-by", "owner",
        "--corpus-snapshot", str(missing),
        "--search-audit", str(missing),
        "--output-root", str(tmp_path),
    ]) == 0
    _payload(capsys)

    compiled = captured["compile_resources"]
    approved = captured["approve_resources"]
    assert compiled.configured and approved.configured
    assert compiled.schema_paths == approved.schema_paths
    assert compiled.prompt_paths == approved.prompt_paths


def test_download_cli_wires_explicit_authorized_skill_handoff(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    config_document = yaml.safe_load(
        (ROOT / "configs" / "abstract_focus.yaml").read_text(encoding="utf-8")
    )
    database_path = tmp_path / "papers.sqlite3"
    config_document["storage"]["sqlite_path"] = str(database_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(config_document, sort_keys=False), encoding="utf-8"
    )
    with Database(database_path) as database:
        database.migrate()
    captured = {}

    class FakeService:
        def __init__(self, *_args, **_kwargs):
            pass

        def run(self, **options):
            captured.update(options)
            return SimpleNamespace(
                run_id="stage3-1",
                paper_ids=("paper-1",),
                status="incomplete",
                dry_run=False,
                run=None,
                planned_decisions=(),
                authorized_queue_path=options["authorized_skill"].queue_path,
            )

    monkeypatch.setattr(cli, "Stage3DownloadService", FakeService)
    queue = tmp_path / "handoff" / "papers.csv"
    assert cli.main([
        "download",
        "--config", str(config_path),
        "--paper-id", "paper-1",
        "--authorized-skill-queue", str(queue),
        "--authorized-skill-output", str(tmp_path / "handoff-output"),
        "--authorized-skill-root", str(tmp_path / "skills"),
    ]) == 1

    result = _payload(capsys)
    assert result["authorized_queue_path"] == str(queue)
    assert result["event_code"] == "download.incomplete"
    options = captured["authorized_skill"]
    assert options.queue_path == queue
    assert options.skill_roots == (tmp_path / "skills",)
    assert "now" not in captured


def test_analyze_default_run_id_hashes_manifest_fields(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    database_path = tmp_path / "papers.sqlite3"
    with Database(database_path) as database:
        database.migrate()
    selection = tmp_path / "analysis-input.json"
    selection.write_text(
        json.dumps({
            "schema_version": "1",
            "paper_ids": ["paper-1"],
            "stage3_artifact_ids": [],
        }),
        encoding="utf-8",
    )

    class FakeService:
        def __init__(self, *_args, **_kwargs):
            pass

        def run(self, run_id, _manifest, **_options):
            return SimpleNamespace(
                run_id=run_id,
                dry_run=True,
                selected_paper_ids=("paper-1",),
                input_scopes=("metadata_only",),
                result=None,
            )

    monkeypatch.setattr(cli, "AnalysisCliService", FakeService)
    arguments = [
        "analyze",
        "--database", str(database_path),
        "--input", str(selection),
        "--policy", str(ROOT / "policies" / "artifact-processing-v1.yaml"),
        "--dry-run",
    ]
    assert cli.main(arguments) == 0
    first = _payload(capsys)
    assert cli.main(arguments) == 0
    second = _payload(capsys)
    assert first["run_id"] == second["run_id"]
    assert str(first["run_id"]).startswith("analysis-")


def test_analyze_wires_stage4_config_and_preflights_before_dispatch(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    database_path = tmp_path / "papers.sqlite3"
    with Database(database_path) as database:
        database.migrate()
    config_document = yaml.safe_load(
        (ROOT / "configs" / "abstract_focus.yaml").read_text(encoding="utf-8")
    )
    config_document["storage"]["sqlite_path"] = str(database_path)
    config_document["analysis"]["workers"] = 3
    config_document["analysis"]["allow_abstract_only"] = False
    config_document["analysis"]["output_schema"] = str(
        ROOT / "schemas" / "paper-analysis.schema.json"
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(config_document, sort_keys=False), encoding="utf-8"
    )
    selection = tmp_path / "analysis-input.json"
    selection.write_text(
        json.dumps({
            "schema_version": "1",
            "paper_ids": ["paper-1"],
            "stage3_artifact_ids": [],
        }),
        encoding="utf-8",
    )
    events: list[str] = []
    captured: dict[str, object] = {}

    class FakeService:
        def __init__(self, *_args, **options):
            events.append("service")
            captured.update(options)

        def run(self, run_id, _manifest, **_options):
            events.append("dispatch")
            paper = SimpleNamespace(
                error=None,
                input_scope="metadata_only",
                paper_id="paper-1",
                resumed=False,
                status="complete",
            )
            return SimpleNamespace(
                run_id=run_id,
                dry_run=False,
                selected_paper_ids=("paper-1",),
                input_scopes=("metadata_only",),
                result=SimpleNamespace(papers=(paper,)),
            )

    def preflight() -> None:
        events.append("preflight")

    monkeypatch.setattr(cli, "AnalysisCliService", FakeService)
    monkeypatch.setattr(cli, "_analysis_codex_preflight", preflight)

    assert cli.main([
        "--config", str(config_path), "analyze", "--input", str(selection)
    ]) == 0
    assert _payload(capsys)["status"] == "complete"
    assert events == ["service", "preflight", "dispatch"]
    assert captured["workers"] == 3
    assert captured["allow_abstract_only"] is False
    assert captured["output_schema_path"] == (
        ROOT / "schemas" / "paper-analysis.schema.json"
    )


def test_analyze_schema_drift_fails_before_codex_preflight(
    tmp_path: Path, monkeypatch
) -> None:
    database_path = tmp_path / "papers.sqlite3"
    with Database(database_path) as database:
        database.migrate()
    drifted_schema = json.loads(
        (ROOT / "schemas" / "paper-analysis.schema.json").read_text(
            encoding="utf-8"
        )
    )
    drifted_schema["title"] = "drifted"
    schema_path = tmp_path / "paper-analysis.schema.json"
    schema_path.write_text(json.dumps(drifted_schema), encoding="utf-8")
    config_document = yaml.safe_load(
        (ROOT / "configs" / "abstract_focus.yaml").read_text(encoding="utf-8")
    )
    config_document["storage"]["sqlite_path"] = str(database_path)
    config_document["analysis"]["output_schema"] = str(schema_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(config_document, sort_keys=False), encoding="utf-8"
    )
    selection = tmp_path / "analysis-input.json"
    selection.write_text(
        json.dumps({
            "schema_version": "1",
            "paper_ids": ["paper-1"],
            "stage3_artifact_ids": [],
        }),
        encoding="utf-8",
    )
    preflight_calls = 0

    def forbidden_preflight() -> None:
        nonlocal preflight_calls
        preflight_calls += 1
        raise AssertionError("schema drift must precede Codex preflight")

    monkeypatch.setattr(cli, "_analysis_codex_preflight", forbidden_preflight)
    with pytest.raises(ValueError, match="does not match the frozen schema"):
        cli.main([
            "--config", str(config_path), "analyze", "--input", str(selection)
        ])
    assert preflight_calls == 0


def test_cli_rejects_user_controlled_authorization_time() -> None:
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["download", "--now", "2020-01-01T00:00:00Z"])
    with pytest.raises(SystemExit):
        parser.parse_args(["analyze", "--input", "selection.json", "--now", "2020-01-01T00:00:00Z"])


def test_download_dry_run_does_not_migrate_an_uninitialized_database(
    tmp_path: Path,
) -> None:
    config_document = yaml.safe_load(
        (ROOT / "configs" / "abstract_focus.yaml").read_text(encoding="utf-8")
    )
    database_path = tmp_path / "uninitialized.sqlite3"
    sqlite3.connect(database_path).close()
    config_document["storage"]["sqlite_path"] = str(database_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config_document, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="fully migrated"):
        cli.main([
            "--dry-run", "--config", str(config_path), "download", "--paper-id", "paper-1",
        ])

    with closing(sqlite3.connect(database_path)) as connection:
        assert connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'schema_migrations'"
        ).fetchone() is None
    assert not database_path.with_name(f"{database_path.name}-wal").exists()
    assert not database_path.with_name(f"{database_path.name}-shm").exists()

    missing_path = tmp_path / "missing.sqlite3"
    config_document["storage"]["sqlite_path"] = str(missing_path)
    config_path.write_text(yaml.safe_dump(config_document, sort_keys=False), encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        cli.main([
            "--dry-run", "--config", str(config_path), "download", "--paper-id", "paper-1",
        ])
    assert not missing_path.exists()


def test_download_dry_run_uses_a_disposable_database_clone(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    config_document = yaml.safe_load(
        (ROOT / "configs" / "abstract_focus.yaml").read_text(encoding="utf-8")
    )
    database_path = tmp_path / "papers.sqlite3"
    with Database(database_path) as database:
        database.migrate()
        database.connection.execute(
            "INSERT INTO papers(paper_id, title) VALUES ('paper-1', 'Paper')"
        )
        database.connection.commit()
    config_document["storage"]["sqlite_path"] = str(database_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config_document, sort_keys=False), encoding="utf-8")
    before = database_path.read_bytes()
    sidecars = tuple(
        database_path.with_name(f"{database_path.name}-{suffix}")
        for suffix in ("wal", "shm")
    )
    before_sidecars = {
        path: path.read_bytes() if path.exists() else None for path in sidecars
    }
    captured: dict[str, Path] = {}

    class FakeService:
        def __init__(self, database, _config, **options):
            captured["database"] = database.path
            captured["artifact_root"] = options["artifact_root"]

        def run(self, **_options):
            return SimpleNamespace(
                run_id="stage3-dry",
                paper_ids=("paper-1",),
                status="validated",
                dry_run=True,
                run=None,
                planned_decisions=(),
                authorized_queue_path=None,
            )

    monkeypatch.setattr(cli, "Stage3DownloadService", FakeService)
    assert cli.main([
        "--dry-run", "--config", str(config_path), "download", "--paper-id", "paper-1",
    ]) == 0
    assert _payload(capsys)["status"] == "validated"
    assert captured["database"] != database_path
    assert not captured["database"].exists()
    assert not captured["artifact_root"].exists()
    assert database_path.read_bytes() == before
    assert {
        path: path.read_bytes() if path.exists() else None for path in sidecars
    } == before_sidecars
