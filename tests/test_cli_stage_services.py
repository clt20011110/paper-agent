from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest
import yaml

from paper_agent import cli
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
    output_root = tmp_path / "output"
    assert cli.main([
        "report",
        "--plan", str(approved),
        "--database", str(database_path),
        "--output-root", str(output_root),
        "--policy", str(ROOT / "policies" / "artifact-processing-v1.yaml"),
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
    ]) == 0

    result = _payload(capsys)
    assert result["authorized_queue_path"] == str(queue)
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

    with sqlite3.connect(database_path) as connection:
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
