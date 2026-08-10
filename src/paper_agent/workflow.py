"""Typed, fenced orchestration over the existing stage services."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from hashlib import sha256
import json
from pathlib import Path
import signal
import sqlite3
from threading import Event, Lock, Thread
from typing import Any, Protocol, TypeAlias
from uuid import uuid4

from .canonical import canonical_json, content_hash
from .storage import Database


class StageKind(StrEnum):
    SEARCH = "search"
    FILTER = "filter"
    DOWNLOAD = "download"
    ANALYZE = "analyze"
    REPORT = "report"


class StepObservation(StrEnum):
    PENDING = "pending"
    COMPLETE = "complete"
    RUNNING = "running"
    SAFE_TO_RESUME = "safe_to_resume"
    BLOCKED = "blocked"
    UNCERTAIN_TERMINAL = "uncertain_terminal"


@dataclass(frozen=True, slots=True)
class FileRef:
    path: str
    sha256: str
    resolved_path: Path

    def __post_init__(self) -> None:
        if not self.path or not _is_hash(self.sha256):
            raise ValueError("workflow FileRef is invalid")

    def verify(self) -> None:
        if not self.resolved_path.is_file() or _file_hash(self.resolved_path) != self.sha256:
            raise ValueError(f"workflow file reference drifted: {self.path}")

    def document(self) -> dict[str, str]:
        return {"path": self.path, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class SnapshotRef:
    provider: str
    file: FileRef

    def document(self) -> dict[str, Any]:
        return {"provider": self.provider, "file": self.file.document()}


@dataclass(frozen=True, slots=True)
class SearchStep:
    step_id: str
    plan: FileRef
    stage2_release: FileRef
    snapshots: tuple[SnapshotRef, ...]
    historical_replay: bool
    stage: StageKind = StageKind.SEARCH

    def document(self) -> dict[str, Any]:
        return {
            "id": self.step_id,
            "stage": self.stage.value,
            "plan": self.plan.document(),
            "stage2_release": self.stage2_release.document(),
            "snapshots": [item.document() for item in self.snapshots],
            "historical_replay": self.historical_replay,
        }

    def file_refs(self) -> tuple[FileRef, ...]:
        return (self.plan, self.stage2_release, *(item.file for item in self.snapshots))


@dataclass(frozen=True, slots=True)
class FilterStep:
    step_id: str
    plan: FileRef
    stage2_release: FileRef
    selection: FileRef | None
    stage: StageKind = StageKind.FILTER

    def document(self) -> dict[str, Any]:
        return {
            "id": self.step_id,
            "stage": self.stage.value,
            "plan": self.plan.document(),
            "stage2_release": self.stage2_release.document(),
            "selection": self.selection.document() if self.selection else None,
        }

    def file_refs(self) -> tuple[FileRef, ...]:
        return (self.plan, self.stage2_release, *((self.selection,) if self.selection else ()))


@dataclass(frozen=True, slots=True)
class DownloadStep:
    step_id: str
    selection: FileRef
    authorization_grant_id: str | None
    provider_terms: FileRef | None
    stage: StageKind = StageKind.DOWNLOAD

    def document(self) -> dict[str, Any]:
        return {
            "id": self.step_id,
            "stage": self.stage.value,
            "selection": self.selection.document(),
            "authorization_grant_id": self.authorization_grant_id,
            "provider_terms": self.provider_terms.document() if self.provider_terms else None,
        }

    def file_refs(self) -> tuple[FileRef, ...]:
        return (self.selection, *((self.provider_terms,) if self.provider_terms else ()))


@dataclass(frozen=True, slots=True)
class AnalyzeStep:
    step_id: str
    selection: FileRef
    processing_grant_id: str | None
    policy: FileRef | None
    stage: StageKind = StageKind.ANALYZE

    def document(self) -> dict[str, Any]:
        return {
            "id": self.step_id,
            "stage": self.stage.value,
            "selection": self.selection.document(),
            "processing_grant_id": self.processing_grant_id,
            "policy": self.policy.document() if self.policy else None,
        }

    def file_refs(self) -> tuple[FileRef, ...]:
        return (self.selection, *((self.policy,) if self.policy else ()))


@dataclass(frozen=True, slots=True)
class ReportStep:
    step_id: str
    plan: FileRef
    corpus_snapshot: FileRef
    search_audit: FileRef
    processing_grants: FileRef | None
    previous_report_run_id: str | None
    policy: FileRef | None
    stage: StageKind = StageKind.REPORT

    def document(self) -> dict[str, Any]:
        return {
            "id": self.step_id,
            "stage": self.stage.value,
            "plan": self.plan.document(),
            "corpus_snapshot": self.corpus_snapshot.document(),
            "search_audit": self.search_audit.document(),
            "processing_grants": self.processing_grants.document() if self.processing_grants else None,
            "previous_report_run_id": self.previous_report_run_id,
            "policy": self.policy.document() if self.policy else None,
        }

    def file_refs(self) -> tuple[FileRef, ...]:
        optional = tuple(
            item for item in (self.processing_grants, self.policy) if item is not None
        )
        return (self.plan, self.corpus_snapshot, self.search_audit, *optional)


StageSpec: TypeAlias = SearchStep | FilterStep | DownloadStep | AnalyzeStep | ReportStep


@dataclass(frozen=True, slots=True)
class WorkflowManifest:
    workflow_id: str
    config: FileRef
    steps: tuple[StageSpec, ...]
    source_path: Path

    def __post_init__(self) -> None:
        if not self.workflow_id or not self.steps:
            raise ValueError("workflow manifest identity is invalid")
        if len({step.step_id for step in self.steps}) != len(self.steps):
            raise ValueError("workflow step IDs must be unique")
        if len({step.stage for step in self.steps}) != len(self.steps):
            raise ValueError("workflow stages must be unique")
        order = {stage: index for index, stage in enumerate(StageKind)}
        positions = [order[step.stage] for step in self.steps]
        if positions != sorted(positions):
            raise ValueError(
                "workflow stages must follow search -> filter -> download -> analyze -> report"
            )

    def document(self) -> dict[str, Any]:
        return {
            "schema_version": "1",
            "workflow_id": self.workflow_id,
            "config": self.config.document(),
            "steps": [step.document() for step in self.steps],
        }

    @property
    def manifest_hash(self) -> str:
        return content_hash(self.document())

    def verify_files(self) -> None:
        self.config.verify()
        for step in self.steps:
            for reference in step.file_refs():
                reference.verify()


@dataclass(frozen=True, slots=True)
class StepContext:
    database_path: Path
    config_path: Path
    workflow_run_id: str
    child_run_id: str
    dry_run: bool


@dataclass(frozen=True, slots=True)
class StageIdentity:
    identity_hash: str


@dataclass(frozen=True, slots=True)
class StageOutcome:
    status: str
    payload: Mapping[str, Any]


class StageAdapter(Protocol):
    def validate(self, context: StepContext, spec: StageSpec) -> StageIdentity: ...
    def observe(
        self, context: StepContext, spec: StageSpec, identity: StageIdentity
    ) -> StepObservation: ...
    def execute(
        self, context: StepContext, spec: StageSpec, identity: StageIdentity
    ) -> StageOutcome: ...


class StopToken:
    def __init__(self) -> None:
        self._event = Event()
        self._lock = Lock()
        self._signal: int | None = None

    def request_stop(self, signum: int | None = None) -> None:
        with self._lock:
            self._signal = signum
            self._event.set()

    def requested(self) -> bool:
        return self._event.is_set()

    @property
    def signal(self) -> int | None:
        with self._lock:
            return self._signal

    @contextmanager
    def install_signal_handlers(self) -> Iterator[None]:
        """Turn termination signals into a request observed between stages.

        This deliberately does not raise from a handler: an in-flight adapter gets
        to return or reach its own safe point before the workflow is checkpointed.
        Signal handlers can only be installed by the main thread.
        """
        previous = {
            signum: signal.getsignal(signum)
            for signum in (signal.SIGINT, signal.SIGTERM)
        }

        def request(signum: int, _frame: object) -> None:
            self.request_stop(signum)

        try:
            for signum in previous:
                signal.signal(signum, request)
            yield
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)


class SequentialWorkflowOrchestrator:
    def __init__(
        self,
        database: Database | None,
        manifest: WorkflowManifest,
        adapters: Mapping[StageKind, StageAdapter],
        *,
        owner_id: str | None = None,
        clock: Callable[[], datetime] | None = None,
        stop_token: StopToken | None = None,
        lease_ttl: timedelta = timedelta(minutes=5),
    ) -> None:
        if lease_ttl.total_seconds() <= 0:
            raise ValueError("workflow lease_ttl must be positive")
        self.database = database
        self.manifest = manifest
        self.adapters = dict(adapters)
        self.owner_id = owner_id or f"workflow-{uuid4()}"
        self.clock = clock or (lambda: datetime.now(UTC))
        self.stop_token = stop_token or StopToken()
        self.lease_ttl = lease_ttl

    def run(self, workflow_run_id: str, *, dry_run: bool = False) -> dict[str, Any]:
        identities, observations = self._preflight(workflow_run_id, dry_run=dry_run)
        if dry_run:
            return self._dry_result(workflow_run_id, identities, observations)
        existing = self._run_row(workflow_run_id)
        if existing is not None:
            self._assert_manifest(existing)
            if existing["status"] == "complete":
                return self._result(workflow_run_id)
            raise ValueError("existing non-complete workflow requires resume")
        try:
            self._create(workflow_run_id, identities)
        except sqlite3.IntegrityError:
            # A peer may have persisted the immutable run between our lookup and
            # insert.  Continue through the fenced acquisition path instead of
            # treating that harmless race as an execution error.
            existing = self._run_row(workflow_run_id)
            if existing is None:
                raise
            self._assert_manifest(existing)
        return self._process(workflow_run_id, identities)

    def resume(self, workflow_run_id: str, *, dry_run: bool = False) -> dict[str, Any]:
        identities, observations = self._preflight(workflow_run_id, dry_run=dry_run)
        if dry_run:
            return self._dry_result(workflow_run_id, identities, observations)
        self._database()
        existing = self._run_row(workflow_run_id)
        if existing is None:
            raise ValueError("workflow resume requires an existing run")
        self._assert_manifest(existing)
        if existing["status"] == "complete":
            return self._result(workflow_run_id)
        return self._process(workflow_run_id, identities)

    def _preflight(
        self, workflow_run_id: str, *, dry_run: bool
    ) -> tuple[dict[str, StageIdentity], dict[str, StepObservation]]:
        self.manifest.verify_files()
        identities: dict[str, StageIdentity] = {}
        observations: dict[str, StepObservation] = {}
        for step in self.manifest.steps:
            adapter = self.adapters.get(step.stage)
            if adapter is None:
                raise ValueError(f"workflow stage adapter is unavailable: {step.stage.value}")
            context = self._context(workflow_run_id, step, dry_run=dry_run)
            identity = adapter.validate(context, step)
            if not _is_hash(identity.identity_hash):
                raise ValueError("workflow stage identity must be a SHA-256 digest")
            identities[step.step_id] = identity
            observations[step.step_id] = adapter.observe(context, step, identity)
        return identities, observations

    def _process(
        self, workflow_run_id: str, identities: Mapping[str, StageIdentity]
    ) -> dict[str, Any]:
        token = self._acquire_workflow(workflow_run_id)
        if token is None:
            result = self._result(workflow_run_id)
            result["outcome"] = "already_running"
            return result
        try:
            for step in self.manifest.steps:
                if self.stop_token.requested():
                    self._set_run_status(workflow_run_id, token, "incomplete", "stop_requested")
                    return self._result(workflow_run_id)
                self.manifest.verify_files()
                identity = identities[step.step_id]
                self._assert_step_identity(workflow_run_id, step, identity)
                if self._step_status(workflow_run_id, step.step_id) == "complete":
                    continue
                context = self._context(workflow_run_id, step, dry_run=False)
                adapter = self.adapters[step.stage]
                observation = adapter.observe(context, step, identity)
                if observation is StepObservation.COMPLETE:
                    self._checkpoint_observed_complete(workflow_run_id, token, step)
                    continue
                if observation in {
                    StepObservation.RUNNING,
                    StepObservation.BLOCKED,
                    StepObservation.UNCERTAIN_TERMINAL,
                }:
                    status = (
                        "uncertain_terminal"
                        if observation is StepObservation.UNCERTAIN_TERMINAL
                        else "blocked"
                    )
                    self._checkpoint_stop(workflow_run_id, token, step, status, observation.value)
                    return self._result(workflow_run_id)
                step_token = self._claim_step(workflow_run_id, token, step)
                if step_token is None:
                    return self._result(workflow_run_id)
                heartbeat_stop = Event()
                heartbeat = self._start_heartbeat(
                    workflow_run_id, token, step, step_token, heartbeat_stop
                )
                try:
                    outcome = adapter.execute(context, step, identity)
                except Exception as error:
                    outcome = StageOutcome(
                        "failed",
                        {"error_type": type(error).__name__, "message": str(error)},
                    )
                finally:
                    heartbeat_stop.set()
                    heartbeat.join()
                self._finish_step(
                    workflow_run_id, token, step, step_token, outcome
                )
                if outcome.status != "complete":
                    return self._result(workflow_run_id)
            self._set_run_status(workflow_run_id, token, "complete", None)
            return self._result(workflow_run_id)
        finally:
            self._release_workflow(workflow_run_id, token)

    def _dry_result(
        self,
        workflow_run_id: str,
        identities: Mapping[str, StageIdentity],
        observations: Mapping[str, StepObservation],
    ) -> dict[str, Any]:
        return {
            "command": "run",
            "manifest_hash": self.manifest.manifest_hash,
            "status": "validated",
            "steps": [
                {
                    "id": step.step_id,
                    "stage": step.stage.value,
                    "identity_hash": identities[step.step_id].identity_hash,
                    "observation": observations[step.step_id].value,
                }
                for step in self.manifest.steps
            ],
            "workflow_run_id": workflow_run_id,
        }

    def _create(
        self, workflow_run_id: str, identities: Mapping[str, StageIdentity]
    ) -> None:
        database = self._database()
        manifest_json = canonical_json(self.manifest.document()).decode("utf-8")
        with database.transaction() as connection:
            connection.execute(
                """INSERT INTO workflow_runs(
                       workflow_run_id, manifest_hash, manifest_json, status
                   ) VALUES (?, ?, ?, 'pending')""",
                (workflow_run_id, self.manifest.manifest_hash, manifest_json),
            )
            for ordinal, step in enumerate(self.manifest.steps):
                connection.execute(
                    """INSERT INTO workflow_steps(
                           workflow_run_id, step_id, ordinal, stage, child_run_id,
                           spec_hash, identity_hash, status
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending')""",
                    (
                        workflow_run_id,
                        step.step_id,
                        ordinal,
                        step.stage.value,
                        f"{workflow_run_id}:{step.step_id}",
                        content_hash(step.document()),
                        identities[step.step_id].identity_hash,
                    ),
                )

    def _acquire_workflow(self, workflow_run_id: str) -> int | None:
        database = self._database()
        now, expires = self._lease_times()
        with database.transaction() as connection:
            result = connection.execute(
                """UPDATE workflow_runs
                   SET status = 'running', lease_owner = ?,
                       lease_token = lease_token + 1, lease_expires_at = ?,
                       updated_at = CURRENT_TIMESTAMP
                   WHERE workflow_run_id = ?
                     AND (lease_owner IS NULL OR lease_expires_at <= ?)""",
                (self.owner_id, expires, workflow_run_id, now),
            )
            if result.rowcount != 1:
                return None
            row = connection.execute(
                "SELECT lease_token FROM workflow_runs WHERE workflow_run_id = ?",
                (workflow_run_id,),
            ).fetchone()
        return int(row["lease_token"])

    def _claim_step(
        self, workflow_run_id: str, workflow_token: int, step: StageSpec
    ) -> int | None:
        database = self._database()
        now, expires = self._lease_times()
        with database.transaction() as connection:
            if not self._owns_workflow(connection, workflow_run_id, workflow_token, now):
                return None
            result = connection.execute(
                """UPDATE workflow_steps
                   SET status = 'running', lease_owner = ?,
                       lease_token = lease_token + 1, lease_expires_at = ?,
                       started_at = COALESCE(started_at, CURRENT_TIMESTAMP),
                       completed_at = NULL, error_json = NULL
                   WHERE workflow_run_id = ? AND step_id = ?
                     AND status IN ('pending', 'running', 'incomplete', 'blocked', 'failed')
                     AND (lease_owner IS NULL OR lease_expires_at <= ?)""",
                (
                    self.owner_id,
                    expires,
                    workflow_run_id,
                    step.step_id,
                    now,
                ),
            )
            if result.rowcount != 1:
                return None
            row = connection.execute(
                """SELECT lease_token FROM workflow_steps
                   WHERE workflow_run_id = ? AND step_id = ?""",
                (workflow_run_id, step.step_id),
            ).fetchone()
        return int(row["lease_token"])

    def _finish_step(
        self,
        workflow_run_id: str,
        workflow_token: int,
        step: StageSpec,
        step_token: int,
        outcome: StageOutcome,
    ) -> None:
        allowed = {
            "complete": "complete",
            "incomplete": "incomplete",
            "blocked": "blocked",
            "uncertain_terminal": "uncertain_terminal",
            "failed": "failed",
        }
        if outcome.status not in allowed:
            raise ValueError(f"unsupported workflow stage outcome: {outcome.status}")
        step_status = allowed[outcome.status]
        run_status = "running" if step_status == "complete" else (
            "incomplete" if step_status == "incomplete" else "blocked" if step_status in {"blocked", "uncertain_terminal"} else "failed"
        )
        database = self._database()
        now = self._now_text()
        with database.transaction() as connection:
            if not self._owns_workflow(connection, workflow_run_id, workflow_token, now):
                raise RuntimeError("workflow result lost its fencing token")
            result = connection.execute(
                """UPDATE workflow_steps
                   SET status = ?, result_json = ?, lease_owner = NULL,
                       lease_expires_at = NULL, completed_at = CURRENT_TIMESTAMP
                   WHERE workflow_run_id = ? AND step_id = ? AND status = 'running'
                     AND lease_owner = ? AND lease_token = ? AND lease_expires_at > ?""",
                (
                    step_status,
                    canonical_json(dict(outcome.payload)).decode("utf-8"),
                    workflow_run_id,
                    step.step_id,
                    self.owner_id,
                    step_token,
                    now,
                ),
            )
            if result.rowcount != 1:
                raise RuntimeError("workflow step result lost its fencing token")
            connection.execute(
                """UPDATE workflow_runs SET status = ?, updated_at = CURRENT_TIMESTAMP
                   WHERE workflow_run_id = ? AND lease_owner = ? AND lease_token = ?""",
                (run_status, workflow_run_id, self.owner_id, workflow_token),
            )

    def _checkpoint_observed_complete(
        self, workflow_run_id: str, workflow_token: int, step: StageSpec
    ) -> None:
        database = self._database()
        with database.transaction() as connection:
            if not self._owns_workflow(
                connection, workflow_run_id, workflow_token, self._now_text()
            ):
                raise RuntimeError("workflow checkpoint lost its fencing token")
            connection.execute(
                """UPDATE workflow_steps
                   SET status = 'complete', lease_owner = NULL, lease_expires_at = NULL,
                       completed_at = COALESCE(completed_at, CURRENT_TIMESTAMP)
                   WHERE workflow_run_id = ? AND step_id = ?""",
                (workflow_run_id, step.step_id),
            )

    def _checkpoint_stop(
        self,
        workflow_run_id: str,
        workflow_token: int,
        step: StageSpec,
        step_status: str,
        reason: str,
    ) -> None:
        database = self._database()
        with database.transaction() as connection:
            if not self._owns_workflow(
                connection, workflow_run_id, workflow_token, self._now_text()
            ):
                raise RuntimeError("workflow stop checkpoint lost its fencing token")
            connection.execute(
                """UPDATE workflow_steps SET status = ?, error_json = ?,
                       lease_owner = NULL, lease_expires_at = NULL,
                       completed_at = CURRENT_TIMESTAMP
                   WHERE workflow_run_id = ? AND step_id = ?""",
                (
                    step_status,
                    canonical_json({"reason": reason}).decode("utf-8"),
                    workflow_run_id,
                    step.step_id,
                ),
            )
            connection.execute(
                """UPDATE workflow_runs SET status = 'blocked', error_json = ?,
                       updated_at = CURRENT_TIMESTAMP
                   WHERE workflow_run_id = ? AND lease_owner = ? AND lease_token = ?""",
                (
                    canonical_json({"reason": reason}).decode("utf-8"),
                    workflow_run_id,
                    self.owner_id,
                    workflow_token,
                ),
            )

    def _set_run_status(
        self,
        workflow_run_id: str,
        workflow_token: int,
        status: str,
        reason: str | None,
    ) -> None:
        database = self._database()
        error = (
            canonical_json({"reason": reason}).decode("utf-8") if reason else None
        )
        with database.transaction() as connection:
            result = connection.execute(
                """UPDATE workflow_runs SET status = ?, error_json = ?,
                       updated_at = CURRENT_TIMESTAMP
                   WHERE workflow_run_id = ? AND lease_owner = ? AND lease_token = ?""",
                (status, error, workflow_run_id, self.owner_id, workflow_token),
            )
            if result.rowcount != 1:
                raise RuntimeError("workflow status update lost its fencing token")

    def _start_heartbeat(
        self,
        workflow_run_id: str,
        workflow_token: int,
        step: StageSpec,
        step_token: int,
        stop: Event,
    ) -> Thread:
        # Renew three times per lease, capped so long-running stages stay observable.
        interval = min(30.0, self.lease_ttl.total_seconds() / 3)

        def heartbeat() -> None:
            while not stop.wait(interval):
                now, expires = self._lease_times()
                with Database(self._database().path) as database:
                    with database.transaction() as connection:
                        workflow = connection.execute(
                            """UPDATE workflow_runs SET lease_expires_at = ?, updated_at = CURRENT_TIMESTAMP
                               WHERE workflow_run_id = ? AND lease_owner = ? AND lease_token = ?
                                 AND lease_expires_at > ?""",
                            (
                                expires,
                                workflow_run_id,
                                self.owner_id,
                                workflow_token,
                                now,
                            ),
                        )
                        step_result = connection.execute(
                            """UPDATE workflow_steps SET lease_expires_at = ?
                               WHERE workflow_run_id = ? AND step_id = ? AND status = 'running'
                                 AND lease_owner = ? AND lease_token = ? AND lease_expires_at > ?""",
                            (
                                expires,
                                workflow_run_id,
                                step.step_id,
                                self.owner_id,
                                step_token,
                                now,
                            ),
                        )
                        if workflow.rowcount != 1 or step_result.rowcount != 1:
                            return

        thread = Thread(target=heartbeat, daemon=True)
        thread.start()
        return thread

    def _release_workflow(self, workflow_run_id: str, workflow_token: int) -> None:
        database = self._database()
        with database.transaction() as connection:
            connection.execute(
                """UPDATE workflow_runs SET lease_owner = NULL, lease_expires_at = NULL,
                       updated_at = CURRENT_TIMESTAMP
                   WHERE workflow_run_id = ? AND lease_owner = ? AND lease_token = ?""",
                (workflow_run_id, self.owner_id, workflow_token),
            )

    def _assert_manifest(self, row: Any) -> None:
        expected = (
            self.manifest.manifest_hash,
            canonical_json(self.manifest.document()).decode("utf-8"),
        )
        if (row["manifest_hash"], row["manifest_json"]) != expected:
            raise ValueError("workflow run input is immutable")

    def _assert_step_identity(
        self, workflow_run_id: str, step: StageSpec, identity: StageIdentity
    ) -> None:
        row = self._database().connection.execute(
            """SELECT stage, child_run_id, spec_hash, identity_hash
               FROM workflow_steps WHERE workflow_run_id = ? AND step_id = ?""",
            (workflow_run_id, step.step_id),
        ).fetchone()
        expected = (
            step.stage.value,
            f"{workflow_run_id}:{step.step_id}",
            content_hash(step.document()),
            identity.identity_hash,
        )
        if row is None or tuple(row) != expected:
            raise ValueError("workflow step identity has drifted")

    def _owns_workflow(
        self, connection: Any, workflow_run_id: str, workflow_token: int, now: str
    ) -> bool:
        row = connection.execute(
            """SELECT 1 FROM workflow_runs WHERE workflow_run_id = ?
               AND lease_owner = ? AND lease_token = ? AND lease_expires_at > ?""",
            (workflow_run_id, self.owner_id, workflow_token, now),
        ).fetchone()
        return row is not None

    def _result(self, workflow_run_id: str) -> dict[str, Any]:
        database = self._database()
        run = self._run_row(workflow_run_id)
        rows = database.connection.execute(
            """SELECT step_id, ordinal, stage, child_run_id, status, result_json, error_json
               FROM workflow_steps WHERE workflow_run_id = ? ORDER BY ordinal""",
            (workflow_run_id,),
        ).fetchall()
        return {
            "command": "run",
            "manifest_hash": run["manifest_hash"],
            "status": run["status"],
            "steps": [
                {
                    "id": row["step_id"],
                    "stage": row["stage"],
                    "child_run_id": row["child_run_id"],
                    "status": row["status"],
                    "result": json.loads(row["result_json"])
                    if row["result_json"]
                    else None,
                    "error": json.loads(row["error_json"])
                    if row["error_json"]
                    else None,
                }
                for row in rows
            ],
            "workflow_run_id": workflow_run_id,
        }

    def _run_row(self, workflow_run_id: str):
        return self._database().connection.execute(
            "SELECT * FROM workflow_runs WHERE workflow_run_id = ?",
            (workflow_run_id,),
        ).fetchone()

    def _step_status(self, workflow_run_id: str, step_id: str) -> str:
        row = self._database().connection.execute(
            """SELECT status FROM workflow_steps
               WHERE workflow_run_id = ? AND step_id = ?""",
            (workflow_run_id, step_id),
        ).fetchone()
        if row is None:
            raise ValueError("workflow step is missing from persisted run")
        return str(row["status"])

    def _context(
        self, workflow_run_id: str, step: StageSpec, *, dry_run: bool
    ) -> StepContext:
        database_path = self.database.path if self.database is not None else Path(":memory:")
        return StepContext(
            database_path=database_path,
            config_path=self.manifest.config.resolved_path,
            workflow_run_id=workflow_run_id,
            child_run_id=f"{workflow_run_id}:{step.step_id}",
            dry_run=dry_run,
        )

    def _database(self) -> Database:
        if self.database is None:
            raise ValueError("workflow persistence requires a database")
        return self.database

    def _lease_times(self) -> tuple[str, str]:
        moment = self._now()
        return _timestamp(moment), _timestamp(moment + self.lease_ttl)

    def _now_text(self) -> str:
        return _timestamp(self._now())

    def _now(self) -> datetime:
        moment = self.clock()
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=UTC)
        return moment.astimezone(UTC)


def load_workflow_manifest(path: Path) -> WorkflowManifest:
    source_path = path.resolve()
    value = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "workflow_id",
        "config",
        "steps",
    }:
        raise ValueError("workflow manifest has unexpected or missing fields")
    if value["schema_version"] != "1" or not isinstance(value["workflow_id"], str) or not value["workflow_id"]:
        raise ValueError("workflow manifest identity is invalid")
    if not isinstance(value["steps"], list) or not value["steps"]:
        raise ValueError("workflow manifest requires at least one step")
    root = source_path.parent
    config = _file_ref(value["config"], root, "config")
    steps = tuple(_step(item, root) for item in value["steps"])
    if len({step.step_id for step in steps}) != len(steps):
        raise ValueError("workflow step IDs must be unique")
    if len({step.stage for step in steps}) != len(steps):
        raise ValueError("workflow stages must be unique")
    manifest = WorkflowManifest(value["workflow_id"], config, steps, source_path)
    manifest.verify_files()
    return manifest


def _step(value: object, root: Path) -> StageSpec:
    if not isinstance(value, dict) or not isinstance(value.get("stage"), str):
        raise ValueError("workflow step must be a typed object")
    try:
        stage = StageKind(value["stage"])
    except ValueError as error:
        raise ValueError(f"unsupported workflow stage: {value['stage']}") from error
    step_id = value.get("id")
    if not isinstance(step_id, str) or not step_id:
        raise ValueError("workflow step id is required")
    if stage is StageKind.SEARCH:
        _exact(value, {"id", "stage", "plan", "stage2_release", "snapshots", "historical_replay"})
        snapshots = value["snapshots"]
        if not isinstance(snapshots, list):
            raise ValueError("search snapshots must be a list")
        parsed_snapshots = tuple(_snapshot(item, root) for item in snapshots)
        if len({item.provider for item in parsed_snapshots}) != len(parsed_snapshots):
            raise ValueError("search snapshot providers must be unique")
        if not isinstance(value["historical_replay"], bool):
            raise ValueError("historical_replay must be boolean")
        return SearchStep(
            step_id,
            _file_ref(value["plan"], root, "search plan"),
            _file_ref(value["stage2_release"], root, "Stage 2 release"),
            parsed_snapshots,
            value["historical_replay"],
        )
    if stage is StageKind.FILTER:
        _exact(value, {"id", "stage", "plan", "stage2_release", "selection"})
        return FilterStep(
            step_id,
            _file_ref(value["plan"], root, "filter plan"),
            _file_ref(value["stage2_release"], root, "Stage 2 release"),
            _optional_file_ref(value["selection"], root, "filter selection"),
        )
    if stage is StageKind.DOWNLOAD:
        _exact(value, {"id", "stage", "selection", "authorization_grant_id", "provider_terms"})
        return DownloadStep(
            step_id,
            _file_ref(value["selection"], root, "download selection"),
            _optional_text(value["authorization_grant_id"], "authorization_grant_id"),
            _optional_file_ref(value["provider_terms"], root, "provider terms"),
        )
    if stage is StageKind.ANALYZE:
        _exact(value, {"id", "stage", "selection", "processing_grant_id", "policy"})
        return AnalyzeStep(
            step_id,
            _file_ref(value["selection"], root, "analysis selection"),
            _optional_text(value["processing_grant_id"], "processing_grant_id"),
            _optional_file_ref(value["policy"], root, "analysis policy"),
        )
    _exact(value, {
        "id", "stage", "plan", "corpus_snapshot", "search_audit",
        "processing_grants", "previous_report_run_id", "policy",
    })
    return ReportStep(
        step_id,
        _file_ref(value["plan"], root, "report plan"),
        _file_ref(value["corpus_snapshot"], root, "corpus snapshot"),
        _file_ref(value["search_audit"], root, "search audit"),
        _optional_file_ref(value["processing_grants"], root, "processing grants"),
        _optional_text(value["previous_report_run_id"], "previous_report_run_id"),
        _optional_file_ref(value["policy"], root, "report policy"),
    )


def _snapshot(value: object, root: Path) -> SnapshotRef:
    if not isinstance(value, dict):
        raise ValueError("search snapshot must be an object")
    _exact(value, {"provider", "file"})
    if not isinstance(value["provider"], str) or not value["provider"]:
        raise ValueError("search snapshot provider is required")
    return SnapshotRef(value["provider"], _file_ref(value["file"], root, "search snapshot"))


def _file_ref(value: object, root: Path, label: str) -> FileRef:
    if not isinstance(value, dict):
        raise ValueError(f"workflow {label} must be a FileRef")
    _exact(value, {"path", "sha256"})
    path, digest = value["path"], value["sha256"]
    if not isinstance(path, str) or not path or not _is_hash(digest):
        raise ValueError(f"workflow {label} FileRef is invalid")
    candidate = Path(path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"workflow {label} FileRef path must be relative to the manifest")
    resolved = (root / candidate).resolve()
    if root not in resolved.parents and resolved != root:
        raise ValueError(f"workflow {label} FileRef escapes the manifest directory")
    reference = FileRef(path, digest, resolved)
    reference.verify()
    return reference


def _optional_file_ref(value: object, root: Path, label: str) -> FileRef | None:
    return None if value is None else _file_ref(value, root, label)


def _optional_text(value: object, label: str) -> str | None:
    if value is not None and (not isinstance(value, str) or not value):
        raise ValueError(f"workflow {label} must be a non-empty string or null")
    return value


def _exact(value: Mapping[str, Any], fields: set[str]) -> None:
    if set(value) != fields:
        raise ValueError("workflow object has unexpected or missing fields")


def _file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_hash(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def run_workflow(
    *,
    database_path: Path,
    manifest: WorkflowManifest,
    workflow_run_id: str,
    adapters: Mapping[StageKind, StageAdapter] | None = None,
    dry_run: bool = False,
    **legacy: Any,
) -> dict[str, Any]:
    """Compatibility entry point for callers that provide typed stage adapters.

    Workflow orchestration intentionally no longer accepts argv callbacks.  The
    command boundary must bind real stage services to ``StageAdapter`` instances;
    keeping that binding out of this module prevents recursive CLI execution.
    """
    if legacy:
        names = ", ".join(sorted(legacy))
        raise ValueError(f"workflow does not accept legacy execution arguments: {names}")
    if adapters is None:
        raise ValueError("workflow requires typed stage adapters")
    if dry_run and not database_path.exists():
        return SequentialWorkflowOrchestrator(None, manifest, adapters).run(
            workflow_run_id, dry_run=True
        )
    with Database(database_path, read_only=dry_run) as database:
        if not dry_run:
            database.migrate()
        orchestrator = SequentialWorkflowOrchestrator(database, manifest, adapters)
        return orchestrator.run(workflow_run_id, dry_run=dry_run)
