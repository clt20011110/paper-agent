"""Persisted pipeline run lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .storage import Database


class RunStatus(StrEnum):
    DRAFT = "draft"
    APPROVED = "approved"
    RUNNING = "running"
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    FAILED = "failed"
    CANCELLED = "cancelled"


TRANSITIONS = {
    RunStatus.DRAFT: {RunStatus.APPROVED, RunStatus.CANCELLED},
    RunStatus.APPROVED: {RunStatus.RUNNING, RunStatus.CANCELLED},
    RunStatus.RUNNING: {
        RunStatus.COMPLETE,
        RunStatus.INCOMPLETE,
        RunStatus.FAILED,
        RunStatus.CANCELLED,
    },
    RunStatus.INCOMPLETE: {RunStatus.RUNNING},
    RunStatus.FAILED: {RunStatus.RUNNING},
    RunStatus.COMPLETE: set(),
    RunStatus.CANCELLED: set(),
}


@dataclass(frozen=True, slots=True)
class PipelineRun:
    run_id: str
    stage: str
    status: RunStatus
    input_hash: str
    config_hash: str
    implementation_version: str
    started_at: str | None
    completed_at: str | None
    created_at: str


class RunStore:
    def __init__(self, database: Database) -> None:
        self.database = database

    def create(
        self,
        *,
        run_id: str,
        stage: str,
        input_hash: str,
        config_hash: str,
        implementation_version: str,
        status: RunStatus = RunStatus.DRAFT,
    ) -> PipelineRun:
        existing = self.get(run_id)
        if existing is not None:
            identity = (stage, input_hash, config_hash, implementation_version)
            recorded = (
                existing.stage,
                existing.input_hash,
                existing.config_hash,
                existing.implementation_version,
            )
            if identity != recorded:
                raise ValueError("run_id already exists with different frozen inputs")
            return existing
        with self.database.transaction() as connection:
            connection.execute(
                """INSERT INTO pipeline_runs(
                    run_id, stage, status, input_hash, config_hash, implementation_version
                ) VALUES (?, ?, ?, ?, ?, ?)""",
                (run_id, stage, status, input_hash, config_hash, implementation_version),
            )
        return self.get(run_id)  # type: ignore[return-value]

    def get(self, run_id: str) -> PipelineRun | None:
        row = self.database.connection.execute(
            "SELECT * FROM pipeline_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            return None
        return PipelineRun(
            run_id=row["run_id"],
            stage=row["stage"],
            status=RunStatus(row["status"]),
            input_hash=row["input_hash"],
            config_hash=row["config_hash"],
            implementation_version=row["implementation_version"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            created_at=row["created_at"],
        )

    def transition(self, run_id: str, status: RunStatus, *, at: str) -> PipelineRun:
        run = self.get(run_id)
        if run is None:
            raise KeyError(run_id)
        if status == run.status:
            return run
        if status not in TRANSITIONS[run.status]:
            raise ValueError(f"invalid run transition: {run.status} -> {status}")
        started_at = at if status is RunStatus.RUNNING else run.started_at
        completed_at = at if status in {
            RunStatus.COMPLETE,
            RunStatus.INCOMPLETE,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
        } else None
        with self.database.transaction() as connection:
            connection.execute(
                "UPDATE pipeline_runs SET status = ?, started_at = ?, completed_at = ? WHERE run_id = ?",
                (status, started_at, completed_at, run_id),
            )
        return self.get(run_id)  # type: ignore[return-value]

    def resumable(self) -> tuple[PipelineRun, ...]:
        rows = self.database.connection.execute(
            "SELECT run_id FROM pipeline_runs WHERE status IN ('incomplete', 'failed') ORDER BY created_at, run_id"
        ).fetchall()
        return tuple(self.get(row["run_id"]) for row in rows)  # type: ignore[misc]
