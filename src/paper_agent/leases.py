"""Persistent, fenced task leases backed by SQLite."""

from __future__ import annotations

from dataclasses import dataclass
import sqlite3
from typing import Sequence
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


@dataclass(frozen=True)
class TaskLeaseSpec:
    """One logical output that can be claimed independently."""

    paper_id: str | None
    output_kind: str
    input_hash: str


class LeaseNotCurrent(RuntimeError):
    """A worker tried to publish after losing or outliving its lease."""


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
        return self.enqueue_many(
            run_id=run_id,
            stage=stage,
            specs=(TaskLeaseSpec(paper_id, output_kind, input_hash),),
            now=now,
        )[0]

    def enqueue_many(
        self,
        *,
        run_id: str,
        stage: str,
        specs: Sequence[TaskLeaseSpec],
        now: str,
    ) -> tuple[TaskLease, ...]:
        """Create a batch of tasks in one short transaction.

        Existing tasks are returned only when their immutable input hashes
        still match.  This is the enqueue primitive used by model workers so
        preparing a large run does not open one transaction per paper.
        """
        requested = tuple(specs)
        logical_outputs = tuple(
            (spec.paper_id, spec.output_kind) for spec in requested
        )
        if len(logical_outputs) != len(set(logical_outputs)):
            raise ValueError("a lease batch contains duplicate logical outputs")

        rows: list[sqlite3.Row] = []
        with self.database.transaction() as connection:
            for spec in requested:
                row = connection.execute(
                    """
                    SELECT * FROM task_leases
                    WHERE run_id = ? AND stage = ? AND paper_id IS ? AND output_kind = ?
                    """,
                    (run_id, stage, spec.paper_id, spec.output_kind),
                ).fetchone()
                if row is not None and row["input_hash"] != spec.input_hash:
                    raise ValueError(
                        "a task already exists for this output with a different input_hash"
                    )
                if row is None:
                    task_id = uuid4().hex
                    connection.execute(
                        """
                        INSERT INTO task_leases(
                            task_id, run_id, stage, paper_id, output_kind, input_hash,
                            status, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?)
                        """,
                        (
                            task_id,
                            run_id,
                            stage,
                            spec.paper_id,
                            spec.output_kind,
                            spec.input_hash,
                            now,
                            now,
                        ),
                    )
                    row = connection.execute(
                        "SELECT * FROM task_leases WHERE task_id = ?", (task_id,)
                    ).fetchone()
                rows.append(row)
        return tuple(_task_lease(row) for row in rows)

    def claim(
        self,
        *,
        worker_id: str,
        now: str,
        expires_at: str,
        limit: int,
        run_id: str | None = None,
        stage: str | None = None,
        output_kind: str | None = None,
        output_kind_prefix: str | None = None,
    ) -> tuple[TaskLease, ...]:
        """Atomically claim at most ``limit`` eligible tasks in a stable order."""
        if limit <= 0:
            return ()
        if output_kind is not None and output_kind_prefix is not None:
            raise ValueError("claim accepts output_kind or output_kind_prefix, not both")

        conditions = [
            "(status IN ('pending', 'failed_retryable') "
            "OR (status = 'running' AND lease_expires_at <= ?))"
        ]
        parameters: list[object] = [now]
        if run_id is not None:
            conditions.append("run_id = ?")
            parameters.append(run_id)
        if stage is not None:
            conditions.append("stage = ?")
            parameters.append(stage)
        if output_kind is not None:
            conditions.append("output_kind = ?")
            parameters.append(output_kind)
        if output_kind_prefix is not None:
            conditions.append("substr(output_kind, 1, ?) = ?")
            parameters.extend((len(output_kind_prefix), output_kind_prefix))
        parameters.append(limit)
        with self.database.transaction() as connection:
            rows = connection.execute(
                f"""
                SELECT task_id FROM task_leases
                WHERE {' AND '.join(conditions)}
                ORDER BY created_at, output_kind, COALESCE(paper_id, ''), task_id
                LIMIT ?
                """,
                tuple(parameters),
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

    @staticmethod
    def require_current(
        connection: sqlite3.Connection,
        *,
        task_id: str,
        worker_id: str,
        fencing_token: int,
        now: str,
    ) -> None:
        """Assert an unexpired fence inside the caller's write transaction."""
        row = connection.execute(
            """
            SELECT 1 FROM task_leases
            WHERE task_id = ? AND status = 'running' AND worker_id = ?
              AND fencing_token = ? AND lease_expires_at > ?
            """,
            (task_id, worker_id, fencing_token, now),
        ).fetchone()
        if row is None:
            raise LeaseNotCurrent(
                f"task lease {task_id} is expired, superseded, or owned by another worker"
            )

    @classmethod
    def complete_in_transaction(
        cls,
        connection: sqlite3.Connection,
        *,
        task_id: str,
        worker_id: str,
        fencing_token: int,
        now: str,
        retain_claim: bool = False,
    ) -> None:
        """Complete a fence without opening a second transaction.

        Callers can write their result rows and complete the task in the same
        transaction.  Raising rolls back both the result and the transition.
        ``retain_claim`` preserves worker/expiry metadata for an audit trail.
        """
        cls.require_current(
            connection,
            task_id=task_id,
            worker_id=worker_id,
            fencing_token=fencing_token,
            now=now,
        )
        claim_fields = "" if retain_claim else ", worker_id = NULL, lease_expires_at = NULL"
        result = connection.execute(
            f"""
            UPDATE task_leases
            SET status = 'complete', error_json = NULL, updated_at = ?{claim_fields}
            WHERE task_id = ? AND status = 'running' AND worker_id = ?
              AND fencing_token = ? AND lease_expires_at > ?
            """,
            (now, task_id, worker_id, fencing_token, now),
        )
        if result.rowcount != 1:
            raise LeaseNotCurrent(
                f"task lease {task_id} changed before completion"
            )

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
