"""Small SQLite wrapper used as the pipeline's local source of truth."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
import re
import sqlite3
from collections.abc import Iterator


_MIGRATION_NAME = re.compile(r"(?P<version>\d+)_(?P<name>[a-z0-9_]+)\.sql$")


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    sql: str


class Database:
    """A single-node SQLite database with versioned SQL migrations."""

    def __init__(self, path: str | Path, *, read_only: bool = False) -> None:
        self.path = Path(path)
        self.read_only = read_only
        if read_only:
            uri = f"{self.path.resolve().as_uri()}?mode=ro"
            self.connection = sqlite3.connect(
                uri, uri=True, isolation_level="DEFERRED"
            )
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(self.path, isolation_level="DEFERRED")
        self.connection.row_factory = sqlite3.Row
        if not read_only:
            self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.execute("PRAGMA busy_timeout=5000")

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Run a short write transaction; SQLite rolls it back on an exception."""
        if self.connection.in_transaction:
            yield self.connection
            return
        with self.connection:
            self.connection.execute("BEGIN IMMEDIATE")
            yield self.connection

    @staticmethod
    def migrations() -> tuple[Migration, ...]:
        directory = files("paper_agent.storage.migrations")
        migrations: list[Migration] = []
        for entry in directory.iterdir():
            matched = _MIGRATION_NAME.fullmatch(entry.name)
            if matched is not None:
                migrations.append(
                    Migration(
                        version=int(matched["version"]),
                        name=matched["name"],
                        sql=entry.read_text(encoding="utf-8"),
                    )
                )
        migrations.sort(key=lambda migration: migration.version)
        versions = [migration.version for migration in migrations]
        if len(versions) != len(set(versions)):
            raise ValueError("migration versions must be unique")
        return tuple(migrations)

    def current_version(self) -> int:
        table = self.connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'schema_migrations'"
        ).fetchone()
        if table is None:
            return 0
        row = self.connection.execute("SELECT COALESCE(MAX(version), 0) FROM schema_migrations").fetchone()
        return int(row[0])

    def migrate(self, *, dry_run: bool = False, applied_by: str = "paper-agent") -> tuple[Migration, ...]:
        pending = tuple(
            migration
            for migration in self.migrations()
            if migration.version > self.current_version()
        )
        if dry_run:
            return pending
        if self.read_only and pending:
            raise sqlite3.OperationalError("cannot apply migrations through a read-only database")

        for migration in pending:
            with self.transaction() as connection:
                for statement in _statements(migration.sql):
                    connection.execute(statement)
                connection.execute(
                    "INSERT INTO schema_migrations(version, name, applied_by) VALUES (?, ?, ?)",
                    (migration.version, migration.name, applied_by),
                )
        return pending


def _statements(script: str) -> Iterator[str]:
    """Split the deliberately simple migration SQL without losing quoted literals."""
    statement = ""
    for line in script.splitlines(keepends=True):
        statement += line
        if sqlite3.complete_statement(statement):
            if statement.strip():
                yield statement
            statement = ""
    if statement.strip():
        raise ValueError("migration has an incomplete SQL statement")
