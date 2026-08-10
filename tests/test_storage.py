from __future__ import annotations

import sqlite3

import pytest

from paper_agent.storage import Database


def test_migrate_new_database_and_is_idempotent(tmp_path) -> None:
    with Database(tmp_path / "papers.sqlite3") as database:
        applied = database.migrate(applied_by="test")

        assert [migration.version for migration in applied] == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        assert database.current_version() == 19
        assert database.migrate() == ()
        assert database.connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'report_sol_invocations'"
        ).fetchone()[0] == "report_sol_invocations"
        migration = database.connection.execute(
            "SELECT name, applied_by FROM schema_migrations"
        ).fetchone()
        assert dict(migration) == {"name": "initial", "applied_by": "test"}


def test_dry_run_does_not_create_schema(tmp_path) -> None:
    with Database(tmp_path / "papers.sqlite3") as database:
        pending = database.migrate(dry_run=True)

        assert [migration.name for migration in pending] == [
            "initial",
            "task_lease_uniqueness",
            "search_campaigns",
            "search_audit",
            "citation_round_completion",
            "metadata_verification_audit",
            "incremental_crawl_snapshots",
            "authorization_unattended",
            "download_audit",
            "text_extractions",
            "analysis_markdown",
            "report_reduce_nodes",
            "report_audit_runs",
            "report_audit_shards",
            "workflow_runs",
            "analysis_dispatches",
            "stage3_luna_decisions",
            "stage2_adjudicator_retries",
            "stage3_paper_results",
        ]
        assert database.current_version() == 0


def test_read_only_database_never_creates_or_mutates_storage(tmp_path) -> None:
    missing = tmp_path / "missing" / "papers.sqlite3"
    with pytest.raises(sqlite3.OperationalError):
        Database(missing, read_only=True)
    assert not missing.parent.exists()

    path = tmp_path / "papers.sqlite3"
    with Database(path) as database:
        database.migrate()
        database.connection.execute(
            "INSERT INTO papers(paper_id, title) VALUES ('paper-1', 'Paper')"
        )
        database.connection.commit()

    with Database(path, read_only=True) as database:
        assert database.connection.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 1
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            database.connection.execute(
                "INSERT INTO papers(paper_id, title) VALUES ('paper-2', 'No write')"
            )


def test_sqlite_pragmas_enable_wal_foreign_keys_and_busy_timeout(tmp_path) -> None:
    with Database(tmp_path / "papers.sqlite3") as database:
        assert database.connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert database.connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert database.connection.execute("PRAGMA busy_timeout").fetchone()[0] == 5000


def test_constraints_and_foreign_keys_are_enforced(tmp_path) -> None:
    with Database(tmp_path / "papers.sqlite3") as database:
        database.migrate()

        with pytest.raises(sqlite3.IntegrityError):
            database.connection.execute(
                "INSERT INTO paper_sources(source_id, paper_id, provider, external_id, raw_metadata_json) "
                "VALUES ('source-1', 'missing', 'openalex', '1', '{}')"
            )

        database.connection.execute("INSERT INTO papers(paper_id, title) VALUES ('paper-1', 'Paper')")
        with pytest.raises(sqlite3.IntegrityError):
            database.connection.execute(
                "UPDATE papers SET verification_status = 'invented' WHERE paper_id = 'paper-1'"
            )
        with pytest.raises(sqlite3.IntegrityError):
            database.connection.execute(
                "INSERT INTO paper_collections(paper_id, collection_id, membership_status) "
                "VALUES ('paper-1', 'collection-1', 'included')"
            )
        database.connection.rollback()


def test_stage3_paper_result_constraints_are_enforced(tmp_path) -> None:
    with Database(tmp_path / "papers.sqlite3") as database:
        database.migrate()
        database.connection.executemany(
            "INSERT INTO papers(paper_id, title) VALUES (?, ?)",
            (("paper-1", "One"), ("paper-2", "Two")),
        )
        database.connection.execute(
            """INSERT INTO pipeline_runs(
                   run_id, stage, status, input_hash, config_hash,
                   implementation_version
               ) VALUES ('stage3-run', 'stage-3-download', 'running',
                         'input', 'config', 'stage3-cli-v2')"""
        )
        statement = """INSERT INTO stage3_paper_results(
                           run_id, paper_id, status, reason_code, updated_at
                       ) VALUES (?, ?, ?, ?, ?)"""
        valid = ("stage3-run", "paper-1", "not_available", "http_404", "now")
        database.connection.execute(statement, valid)

        with pytest.raises(sqlite3.IntegrityError):
            database.connection.execute(statement, valid)
        with pytest.raises(sqlite3.IntegrityError):
            database.connection.execute(
                statement,
                ("stage3-run", "paper-2", "invented", "bad_status", "now"),
            )
        with pytest.raises(sqlite3.IntegrityError):
            database.connection.execute(
                statement,
                ("stage3-run", "paper-2", "not_available", "  ", "now"),
            )
        with pytest.raises(sqlite3.IntegrityError):
            database.connection.execute(
                statement,
                ("missing-run", "paper-2", "not_available", "http_404", "now"),
            )
        with pytest.raises(sqlite3.IntegrityError):
            database.connection.execute(
                statement,
                ("stage3-run", "missing-paper", "not_available", "http_404", "now"),
            )


def test_transaction_rolls_back_on_error(tmp_path) -> None:
    with Database(tmp_path / "papers.sqlite3") as database:
        database.migrate()

        with pytest.raises(RuntimeError):
            with database.transaction() as connection:
                connection.execute("INSERT INTO papers(paper_id, title) VALUES ('paper-1', 'Paper')")
                raise RuntimeError("stop")

        assert database.connection.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 0
