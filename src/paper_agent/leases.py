"""Persistent, fenced task leases backed by SQLite."""

from __future__ import annotations

from dataclasses import dataclass
import sqlite3
from uuid import uuid4

from paper_agent.storage import Database


@dataclass(frozen=True)
class TaskLease:
    task_id: str
    run_id: str
    stage: str
    paper_id: str | None
    output_kind: str
    input_hash: str
    status: str
    worker_id: str | None
    lease_expires_at: str | None
    attempt: int
    fencing_token: int
    error_json: str | None
    created_at: str
    updated_at: str


class LeaseQueue:
    """Coordinate resumable work across independent SQLite connections."""

    def __init__(self, database: Database) -> None:
        self.database = database

    def enqueue(
        self,
        *,
        run_id: str,
        stage: str,
        paper_id: str | None,
        output_kind: str,
        input_hash: str,
        now: str,
    ) -> TaskLease:
        """Create one task for its logical output, or return the existing task."""
        with self.database.transaction() as connection:
            row = connection.execute(
                """
                SELECT * FROM task_leases
                WHERE run_id = ? AND stage = ? AND paper_id IS ? AND output_kind = ?
                """,
                (run_id, stage, paper_id, output_kind),
            ).fetchone()
            if row is not None and row["input_hash"] != input_hash:
                raise ValueError("a task already exists for this output with a different input_hash")
            if row is None:
                task_id = uuid4().hex
                connection.execute(
                    """
                    INSERT INTO task_leases(
                        task_id, run_id, stage, paper_id, output_kind, input_hash,
                        status, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?)
                    """,
                    (task_id, run_id, stage, paper_id, output_kind, input_hash, now, now),
                )
                row = connection.execute(
                    "SELECT * FROM task_leases WHERE task_id = ?", (task_id,)
                ).fetchone()
        return _task_lease(row)

    def claim(
        self,
        *,
        worker_id: str,
        now: str,
        expires_at: str,
        limit: int,
    ) -> tuple[TaskLease, ...]:
        """Atomically claim at most ``limit`` eligible tasks in a stable order."""
        if limit <= 0:
            return ()

        with self.database.transaction() as connection:
            rows = connection.execute(
                """
                SELECT task_id FROM task_leases
                WHERE status IN ('pending', 'failed_retryable')
                   OR (status = 'running' AND lease_expires_at <= ?)
                ORDER BY created_at, task_id
                LIMIT ?
                """,
                (now, limit),
            ).fetchall()
            task_ids = [row["task_id"] for row in rows]
            for task_id in task_ids:
                connection.execute(
                    """
                    UPDATE task_leases
                    SET status = 'running', worker_id = ?, lease_expires_at = ?,
                        attempt = attempt + 1, fencing_token = fencing_token + 1,
                        updated_at = ?, error_json = NULL
                    WHERE task_id = ?
                    """,
                    (worker_id, expires_at, now, task_id),
                )
            claimed = [
                connection.execute("SELECT * FROM task_leases WHERE task_id = ?", (task_id,)).fetchone()
                for task_id in task_ids
            ]
        return tuple(_task_lease(row) for row in claimed)

    def complete(
        self,
        *,
        task_id: str,
        worker_id: str,
        fencing_token: int,
        now: str,
    ) -> bool:
        """Mark a live lease complete only when its owner and fence still match."""
        return self._finish(
            task_id=task_id,
            worker_id=worker_id,
            fencing_token=fencing_token,
            now=now,
            status="complete",
            error_json=None,
        )

    def fail(
        self,
        *,
        task_id: str,
        worker_id: str,
        fencing_token: int,
        now: str,
        retryable: bool,
        error_json: str | None = None,
    ) -> bool:
        """Finish a live lease as retryable or terminal failure."""
        return self._finish(
            task_id=task_id,
            worker_id=worker_id,
            fencing_token=fencing_token,
            now=now,
            status="failed_retryable" if retryable else "failed_terminal",
            error_json=error_json,
        )

    def resume(self, *, now: str, run_id: str | None = None) -> int:
        """Recover only expired running leases, leaving all other tasks untouched."""
        with self.database.transaction() as connection:
            if run_id is None:
                result = connection.execute(
                    """
                    UPDATE task_leases
                    SET status = 'failed_retryable', worker_id = NULL,
                        lease_expires_at = NULL, updated_at = ?
                    WHERE status = 'running' AND lease_expires_at <= ?
                    """,
                    (now, now),
                )
            else:
                result = connection.execute(
                    """
                    UPDATE task_leases
                    SET status = 'failed_retryable', worker_id = NULL,
                        lease_expires_at = NULL, updated_at = ?
                    WHERE run_id = ? AND status = 'running' AND lease_expires_at <= ?
                    """,
                    (now, run_id, now),
                )
        return result.rowcount

    def _finish(
        self,
        *,
        task_id: str,
        worker_id: str,
        fencing_token: int,
        now: str,
        status: str,
        error_json: str | None,
    ) -> bool:
        with self.database.transaction() as connection:
            result = connection.execute(
                """
                UPDATE task_leases
                SET status = ?, worker_id = NULL, lease_expires_at = NULL,
                    error_json = ?, updated_at = ?
                WHERE task_id = ? AND status = 'running' AND worker_id = ?
                  AND fencing_token = ? AND lease_expires_at > ?
                """,
                (status, error_json, now, task_id, worker_id, fencing_token, now),
            )
        return result.rowcount == 1


def _task_lease(row: sqlite3.Row) -> TaskLease:
    return TaskLease(
        task_id=row["task_id"],
        run_id=row["run_id"],
        stage=row["stage"],
        paper_id=row["paper_id"],
        output_kind=row["output_kind"],
        input_hash=row["input_hash"],
        status=row["status"],
        worker_id=row["worker_id"],
        lease_expires_at=row["lease_expires_at"],
        attempt=row["attempt"],
        fencing_token=row["fencing_token"],
        error_json=row["error_json"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
