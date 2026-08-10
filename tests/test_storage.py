from __future__ import annotations

import sqlite3

import pytest

from paper_agent.storage import Database


def test_migrate_new_database_and_is_idempotent(tmp_path) -> None:
    with Database(tmp_path / "papers.sqlite3") as database:
        applied = database.migrate(applied_by="test")

        assert [migration.version for migration in applied] == [1, 2, 3, 4, 5]
        assert database.current_version() == 5
        assert database.migrate() == ()
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
        ]
        assert database.current_version() == 0


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


def test_transaction_rolls_back_on_error(tmp_path) -> None:
    with Database(tmp_path / "papers.sqlite3") as database:
        database.migrate()

        with pytest.raises(RuntimeError):
            with database.transaction() as connection:
                connection.execute("INSERT INTO papers(paper_id, title) VALUES ('paper-1', 'Paper')")
                raise RuntimeError("stop")

        assert database.connection.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 0
