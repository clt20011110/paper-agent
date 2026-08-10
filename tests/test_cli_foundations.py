from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from paper_agent import cli
from paper_agent.approval import approved_content_hash
from paper_agent.authorized_skill_runtime import load_audit_record
from paper_agent.config import ConfigError
from paper_agent.doctor import DoctorCheck, SystemDoctorReport
from paper_agent.domain import SourceEntry
from paper_agent.downloads import (
    DownloadScopeSnapshotStore,
    build_download_scope_snapshot,
)
from paper_agent.exchange import export_csv, export_jsonl
from paper_agent.grants import GrantError
from paper_agent.repository import PaperRepository
from paper_agent.storage import Database


FUTURE = "2026-08-11T00:00:00Z"
NOW = "2026-08-10T00:00:00Z"


def _payload(capsys) -> dict[str, object]:
    return json.loads(capsys.readouterr().out)


def _download_config(tmp_path: Path) -> tuple[Path, Path]:
    document = yaml.safe_load(
        (Path(__file__).parents[1] / "configs" / "abstract_focus.yaml").read_text(
            encoding="utf-8"
        )
    )
    database = tmp_path / "state" / "papers.sqlite3"
    document["storage"]["sqlite_path"] = str(database)
    defaults = document["download"]["authorized_skill"]["grant_defaults"]
    audit = load_audit_record()
    defaults.update({
        "source_zip_sha256": audit.original_zip_sha256,
        "installed_content_sha256": audit.installed_content_sha256,
        "dependency_lock_sha256": audit.dependency_lock_sha256,
        "allowed_domains": ["publisher.example"],
        "paper_ids": ["paper-1"],
        "max_papers": 1,
        "authorization_expires_at": FUTURE,
    })
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path, database


def test_grant_cli_creates_unapproved_draft_then_approves_and_revokes(tmp_path: Path, capsys) -> None:
    config, database = _download_config(tmp_path)
    draft_path = tmp_path / "grant.json"
    assert cli.main([
        "grant", "create", "--kind", "download", "--output", str(draft_path),
        "--config", str(config), "--run-id", "grant-run",
    ]) == 0
    created = _payload(capsys)
    draft = json.loads(draft_path.read_text(encoding="utf-8"))
    assert created["status"] == "draft"
    assert created["run_id"] == "grant-run"
    assert draft["status"] == "draft"
    assert draft["approval"] is None
    assert draft["scope"]["domains"] == ["publisher.example"]
    assert not database.exists()

    assert cli.main([
        "grant", "approve", "--grant", str(draft_path), "--config", str(config),
        "--hash", draft["content_hash"], "--approved-by", "owner", "--approved-at", NOW,
    ]) == 0
    assert _payload(capsys)["status"] == "approved"
    assert cli.main([
        "grant", "approve", "--grant", str(draft_path), "--config", str(config),
        "--hash", draft["content_hash"], "--approved-by", "owner", "--approved-at", NOW,
    ]) == 0
    assert _payload(capsys)["status"] == "approved"
    assert cli.main([
        "grant", "revoke", draft["grant_id"], "--config", str(config),
        "--actor", "owner", "--at", "2026-08-10T00:01:00Z", "--dry-run",
    ]) == 0
    assert _payload(capsys)["status"] == "validated"
    with Database(database, read_only=True) as stored:
        assert stored.connection.execute(
            "SELECT COUNT(*) FROM authorization_grant_events WHERE event_type = 'revoked'"
        ).fetchone()[0] == 0
    assert cli.main([
        "grant", "revoke", draft["grant_id"], "--config", str(config),
        "--actor", "owner", "--at", "2026-08-10T00:01:00Z",
    ]) == 0
    assert _payload(capsys)["status"] == "revoked"
    assert cli.main([
        "grant", "revoke", draft["grant_id"], "--config", str(config),
        "--actor", "owner", "--at", "2026-08-10T00:01:00Z",
    ]) == 0
    assert _payload(capsys)["status"] == "revoked"


def test_download_grant_uses_only_defaults_and_create_dry_run_has_no_writes(
    tmp_path: Path, capsys
) -> None:
    config, database = _download_config(tmp_path)
    output = tmp_path / "drafts" / "grant.json"
    assert cli.main([
        "grant", "create", "--kind", "download", "--output", str(output),
        "--config", str(config), "--dry-run",
    ]) == 0
    assert _payload(capsys)["status"] == "validated"
    assert not output.exists()
    assert not database.exists()

    with pytest.raises(ConfigError, match="only from grant_defaults"):
        cli.main([
            "grant", "create", "--kind", "download", "--output", str(output),
            "--config", str(config), "--paper-id", "paper-2",
        ])

    drifted = yaml.safe_load(config.read_text(encoding="utf-8"))
    drifted["download"]["authorized_skill"]["grant_defaults"][
        "dependency_lock_sha256"
    ] = "d" * 64
    config.write_text(yaml.safe_dump(drifted, sort_keys=False), encoding="utf-8")
    with pytest.raises(ConfigError, match="checked-in skill audit"):
        cli.main([
            "grant", "create", "--kind", "download", "--output", str(output),
            "--config", str(config), "--dry-run",
        ])


def test_download_cli_resolves_hash_bound_snapshot_path_and_persisted_id(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "papers.sqlite3"
    snapshot_path = tmp_path / "selection.json"
    document = build_download_scope_snapshot(
        "selection", ["paper-1"], created_at=NOW, snapshot_id="selection-1"
    )
    snapshot_path.write_text(json.dumps(document), encoding="utf-8")
    with Database(database_path) as database:
        database.migrate()
        database.connection.execute(
            "INSERT INTO papers(paper_id, title) VALUES ('paper-1', 'Paper')"
        )
        database.connection.commit()
        from_path = cli.build_parser().parse_args([
            "download", "--selection-snapshot", str(snapshot_path)
        ])
        store = DownloadScopeSnapshotStore(database)
        path_binding = cli._download_scope_binding(from_path, store)
        from_id = cli.build_parser().parse_args([
            "download", "--selection-snapshot-id", "selection-1"
        ])
        id_binding = cli._download_scope_binding(
            from_id, DownloadScopeSnapshotStore(database)
        )

    assert path_binding == id_binding
    assert path_binding.selection_snapshot_hash == document["snapshot_hash"]

def test_grant_approve_dry_run_uses_full_semantic_validation(
    tmp_path: Path, capsys
) -> None:
    config, database = _download_config(tmp_path)
    output = tmp_path / "grant.json"
    assert cli.main([
        "grant", "create", "--kind", "download", "--output", str(output),
        "--config", str(config),
    ]) == 0
    _payload(capsys)
    invalid = json.loads(output.read_text(encoding="utf-8"))
    invalid["scope"]["paper_ids"] = ["paper-1", "paper-1"]
    invalid["content_hash"] = approved_content_hash(invalid)
    output.write_text(json.dumps(invalid), encoding="utf-8")

    with pytest.raises(GrantError, match="unique"):
        cli.main([
            "grant", "approve", "--grant", str(output), "--hash", invalid["content_hash"],
            "--approved-by", "owner", "--approved-at", NOW, "--dry-run",
        ])
    assert not database.exists()


def test_grant_revoke_dry_run_validates_database_state_and_event(
    tmp_path: Path
) -> None:
    config, database = _download_config(tmp_path)
    with Database(database) as stored:
        stored.migrate()
    with pytest.raises(GrantError, match="not found"):
        cli.main([
            "grant", "revoke", "missing", "--database", str(database),
            "--actor", "owner", "--at", NOW, "--dry-run",
        ])
    draft_path = tmp_path / "grant.json"
    assert cli.main([
        "grant", "create", "--kind", "download", "--output", str(draft_path),
        "--config", str(config),
    ]) == 0
    draft = json.loads(draft_path.read_text(encoding="utf-8"))
    assert cli.main([
        "grant", "approve", "--grant", str(draft_path), "--config", str(config),
        "--hash", draft["content_hash"], "--approved-by", "owner", "--approved-at", NOW,
    ]) == 0
    with pytest.raises(GrantError, match="actor"):
        cli.main([
            "grant", "revoke", draft["grant_id"], "--database", str(database),
            "--actor", "", "--at", "not-a-time", "--dry-run",
        ])


def test_export_cli_uses_configured_database_and_exchange_format(tmp_path: Path, capsys) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        (Path(__file__).parents[1] / "configs" / "abstract_focus.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    database_path = tmp_path / "paper_research_abstract_focus" / "papers.sqlite3"
    with Database(database_path) as database:
        database.migrate()
        PaperRepository(database).ingest(SourceEntry("openalex", "1", "A title"))
    dry_output = tmp_path / "dry-run.jsonl"
    assert cli.main([
        "export", "--format", "jsonl", "--output", str(dry_output),
        "--config", str(config), "--dry-run",
    ]) == 0
    dry_result = _payload(capsys)
    assert dry_result["planned_paper_count"] == 1
    assert not dry_output.exists()
    output = tmp_path / "papers.jsonl"
    assert cli.main(["--config", str(config), "export", "--format", "jsonl", "--output", str(output)]) == 0
    assert _payload(capsys)["exported_count"] == 2
    assert json.loads(output.read_text(encoding="utf-8").splitlines()[0])["record_type"] == "paper"


def test_export_rejects_database_and_sidecar_aliases_before_opening_output(
    tmp_path: Path
) -> None:
    database = tmp_path / "papers.sqlite3"
    with Database(database) as stored:
        stored.migrate()
        PaperRepository(stored).ingest(SourceEntry("openalex", "1", "A title"))
    original_size = database.stat().st_size
    hardlink = tmp_path / "database-hardlink.jsonl"
    hardlink.hardlink_to(database)

    for output in (database, hardlink, Path(f"{database}-wal")):
        with pytest.raises(ConfigError, match="SQLite fact store"):
            cli.main([
                "export", "--database", str(database), "--format", "jsonl",
                "--output", str(output),
            ])

    assert database.stat().st_size == original_size
    with Database(database, read_only=True) as stored:
        assert stored.connection.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 1


def test_export_dry_run_materializes_canonical_values_without_writing(
    tmp_path: Path
) -> None:
    database = tmp_path / "papers.sqlite3"
    with Database(database) as stored:
        stored.migrate()
        PaperRepository(stored).ingest(SourceEntry("openalex", "1", "A title"))
        stored.connection.execute("UPDATE papers SET authors_json = '{'")
        stored.connection.commit()
    output = tmp_path / "papers.jsonl"

    with pytest.raises(json.JSONDecodeError):
        cli.main([
            "export", "--database", str(database), "--format", "jsonl",
            "--output", str(output), "--dry-run",
        ])
    assert not output.exists()


@pytest.mark.parametrize(
    ("exchange_format", "exporter", "suffix"),
    (("jsonl", export_jsonl, ".jsonl"), ("csv", export_csv, ".csv")),
)
def test_import_cli_validates_then_creates_a_new_database(
    tmp_path: Path, capsys, exchange_format, exporter, suffix: str
) -> None:
    source_path = tmp_path / "source.sqlite3"
    with Database(source_path) as source_database:
        source_database.migrate()
        source = PaperRepository(source_database)
        source.ingest(SourceEntry("openalex", "1", "Imported title"))
        exchange_path = tmp_path / f"papers{suffix}"
        exporter(source, exchange_path)

    destination = tmp_path / "destination" / "papers.sqlite3"
    command = [
        "import", "--database", str(destination), "--format", exchange_format,
        "--input", str(exchange_path),
    ]
    assert cli.main([*command, "--dry-run"]) == 0
    dry_run = _payload(capsys)
    assert dry_run["status"] == "validated"
    assert dry_run["counts"]["papers"] == 1
    assert not destination.exists()

    assert cli.main(command) == 0
    imported = _payload(capsys)
    assert imported["status"] == "complete"
    assert imported["counts"]["sources"] == 1
    with Database(destination, read_only=True) as database:
        assert database.connection.execute("SELECT title FROM papers").fetchone()[0] == "Imported title"


def test_import_rejects_the_database_as_its_input(tmp_path: Path) -> None:
    database_path = tmp_path / "papers.sqlite3"
    with Database(database_path) as database:
        database.migrate()
    with pytest.raises(ConfigError, match="must not alias"):
        cli.main([
            "import", "--database", str(database_path), "--format", "jsonl",
            "--input", str(database_path),
        ])


def test_migrate_config_is_dry_by_default_and_writes_only_when_requested(tmp_path: Path, capsys) -> None:
    legacy = tmp_path / "legacy.yaml"
    destination = tmp_path / "nested" / "v2.yaml"
    legacy.write_text("topic: Migration\noutput_dir: ./out\n", encoding="utf-8")
    assert cli.main(["migrate-config", "--input", str(legacy)]) == 0
    assert _payload(capsys)["status"] == "validated"
    assert not destination.exists()
    assert cli.main([
        "migrate-config", "--input", str(legacy), "--write", str(destination),
        "--dry-run",
    ]) == 0
    assert _payload(capsys)["status"] == "validated"
    assert not destination.exists()
    assert cli.main(["migrate-config", "--input", str(legacy), "--write", str(destination)]) == 0
    assert _payload(capsys)["status"] == "written"
    assert destination.exists()


def test_doctor_returns_nonzero_for_required_blockers(monkeypatch, capsys) -> None:
    class FakeDoctor:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def run(self) -> SystemDoctorReport:
            return SystemDoctorReport((DoctorCheck("required", "blocker", True, "nope"),))

    monkeypatch.setattr(cli, "SystemDoctor", FakeDoctor)
    assert cli.main(["doctor", "--run-id", "doctor-run"]) == 1
    payload = _payload(capsys)
    assert payload["ready"] is False
    assert payload["event_code"] == "doctor.blocked"
    assert payload["run_id"] == "doctor-run"


@pytest.mark.parametrize(
    ("target", "argv", "command", "status"),
    (
        ("_search_run", ["search", "run", "--plan", "plan.json"], "search.run", "incomplete"),
        ("_crawl", ["crawl", "--venue", "venue-1"], "crawl", "incomplete"),
        ("_filter", ["filter", "--plan", "plan.json"], "filter", "failed"),
        ("_download", ["download"], "download", "incomplete"),
        ("_analyze", ["analyze", "--input", "input.json"], "analyze", "incomplete"),
        ("_report", ["report", "--plan", "plan.json"], "report", "manual_required"),
    ),
)
def test_stage_non_success_status_returns_nonzero_and_matching_event(
    monkeypatch, capsys, target, argv, command, status
) -> None:
    monkeypatch.setattr(
        cli,
        target,
        lambda *_args, **_kwargs: {"command": command, "status": status},
    )

    assert cli.main(argv) == 1
    payload = _payload(capsys)
    assert payload["event_code"] == f"{command}.{status}"
    assert payload["status"] == status


@pytest.mark.parametrize(
    "status",
    (
        "approved",
        "complete",
        "draft",
        "passed",
        "ready",
        "revoked",
        "runtime_validated",
        "validated",
        "written",
    ),
)
def test_finish_success_status_contract(status, capsys) -> None:
    args = cli.build_parser().parse_args(["crawl", "--venue", "venue-1"])

    assert cli._finish(args, {"command": "fixture", "status": status}) == 0
    assert _payload(capsys)["event_code"] == "fixture.completed"


@pytest.mark.parametrize(
    "status",
    (
        "blocked",
        "cancelled",
        "failed",
        "failed_terminal",
        "incomplete",
        "manual_required",
        "pending",
        "retryable",
        "running",
        "uncertain_terminal",
    ),
)
def test_finish_known_non_success_status_contract(status, capsys) -> None:
    args = cli.build_parser().parse_args(["crawl", "--venue", "venue-1"])

    assert cli._finish(args, {"command": "fixture", "status": status}) == 1
    assert _payload(capsys)["event_code"] == f"fixture.{status}"


def test_finish_missing_and_unknown_statuses_fail_closed(capsys) -> None:
    args = cli.build_parser().parse_args(["crawl", "--venue", "venue-1"])

    assert cli._finish(args, {"command": "default"}) == 0
    assert _payload(capsys)["status"] == "complete"
    assert cli._finish(
        args,
        {"command": "future", "status": "future_status", "event_code": "spoofed"},
    ) == 1
    payload = _payload(capsys)
    assert payload["status"] == "future_status"
    assert payload["event_code"] == "future.failed"


def test_doctor_cli_wires_local_model_probe_and_audited_skill_runtime(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    captured: dict[str, object] = {}

    class FakeDoctor:
        def __init__(self, paths, **kwargs) -> None:
            captured["paths"] = paths
            captured.update(kwargs)

        def run(self) -> SystemDoctorReport:
            return SystemDoctorReport((DoctorCheck("required", "pass", True, "ok"),))

    monkeypatch.setattr(cli, "SystemDoctor", FakeDoctor)
    skill_root = tmp_path / "skills"
    assert cli.main([
        "doctor", "--authorized-skill-root", str(skill_root),
        "--authorized-skill-zip", str(tmp_path / "skill.zip"),
    ]) == 0
    _payload(capsys)
    assert callable(captured["http_probe"])
    assert captured["paths"].authorized_skill_runtime is not None


def test_console_entrypoint_emits_structured_failure_without_creating_paths(
    tmp_path: Path, capsys
) -> None:
    database = tmp_path / "missing" / "papers.sqlite3"
    output = tmp_path / "export.jsonl"
    assert cli.entrypoint([
        "export", "--database", str(database), "--format", "jsonl",
        "--output", str(output), "--run-id", "export-run", "--dry-run",
    ]) == 1
    payload = _payload(capsys)
    assert payload["event_code"] == "export.failed"
    assert payload["run_id"] == "export-run"
    assert payload["status"] == "failed"
    assert not database.parent.exists()
    assert not output.exists()


def test_console_entrypoint_structures_argument_errors(capsys) -> None:
    assert cli.entrypoint(["grant"]) == 1
    payload = _payload(capsys)
    assert payload["command"] == "grant"
    assert payload["error_type"] == "CliUsageError"
    assert payload["event_code"] == "grant.failed"
    assert capsys.readouterr().err == ""
