"""Durable at-most-once intent ledger for paid Stage 4 analysis calls.

The ledger deliberately favors a missed result over a duplicate paid call.  A
worker must durably claim a prepared, authorized intent before constructing an
executor.  Once claimed, the intent's dispatch budget is consumed forever: an
expired lease or any uncertain post-claim outcome is terminal.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from enum import StrEnum
import json
import sqlite3
from typing import Any, Mapping

from .canonical import content_hash
from .processing import ProcessingDecision
from .storage import Database


class AnalysisDispatchStateError(RuntimeError):
    """A dispatch transition did not match its immutable state or lease."""


class AnalysisDispatchStatus(StrEnum):
    PREPARED = "prepared"
    RUNNING = "running"
    COMPLETE = "complete"
    MANUAL_REQUIRED = "manual_required"
    FAILED_TERMINAL = "failed_terminal"


@dataclass(frozen=True, slots=True)
class AnalysisDispatchBinding:
    run_id: str
    paper_id: str
    artifact_hash: str
    artifact_id: str | None
    input_scope: str
    config_hash: str
    implementation_version: str
    profile: str
    model_id: str
    prompt_hash: str
    schema_hash: str
    policy_version: str
    policy_hash: str

    @property
    def dispatch_id(self) -> str:
        return "analysis-dispatch-" + content_hash(asdict(self))


@dataclass(frozen=True, slots=True)
class AnalysisDispatchRecord:
    dispatch_id: str
    run_id: str
    paper_id: str
    artifact_hash: str
    artifact_id: str | None
    input_scope: str
    config_hash: str
    implementation_version: str
    profile: str
    model_id: str
    prompt_hash: str
    schema_hash: str
    policy_version: str
    policy_hash: str
    stable_created_at: str
    prompt_input_hash: str | None
    rendered_prompt_hash: str | None
    processing_decision: Mapping[str, Any] | None
    processing_grant_id: str | None
    status: AnalysisDispatchStatus
    dispatch_count: int
    lease_owner: str | None
    lease_token: int
    lease_expires_at: str | None
    invocation_id: str | None
    invocation_metadata: Mapping[str, Any] | None
    analysis_run_id: str | None
    error: Mapping[str, Any] | None
    completed_at: str | None


@dataclass(frozen=True, slots=True)
class AnalysisDispatchClaim:
    dispatch_id: str
    owner: str
    token: int
    lease_expires_at: str


class AnalysisDispatchStore:
    """Compare-and-swap transitions for one paid-call intent per run/paper."""

    def __init__(self, database: Database) -> None:
        self.database = database

    def prepare(
        self,
        binding: AnalysisDispatchBinding,
        *,
        stable_created_at: datetime | str,
        connection: sqlite3.Connection | None = None,
    ) -> AnalysisDispatchRecord:
        self._validate_binding(binding)
        created_at = _timestamp(stable_created_at)

        def operation(active: sqlite3.Connection) -> AnalysisDispatchRecord:
            active.execute(
                """INSERT OR IGNORE INTO analysis_dispatches(
                    dispatch_id, run_id, paper_id, artifact_hash, artifact_id, input_scope,
                    config_hash, implementation_version, profile, model_id, prompt_hash,
                    schema_hash, policy_version, policy_hash, stable_created_at, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'prepared')""",
                (
                    binding.dispatch_id,
                    binding.run_id,
                    binding.paper_id,
                    binding.artifact_hash,
                    binding.artifact_id,
                    binding.input_scope,
                    binding.config_hash,
                    binding.implementation_version,
                    binding.profile,
                    binding.model_id,
                    binding.prompt_hash,
                    binding.schema_hash,
                    binding.policy_version,
                    binding.policy_hash,
                    created_at,
                ),
            )
            row = active.execute(
                "SELECT * FROM analysis_dispatches WHERE run_id = ? AND paper_id = ?",
                (binding.run_id, binding.paper_id),
            ).fetchone()
            if row is None:
                raise AnalysisDispatchStateError("analysis dispatch preparation was not persisted")
            record = _record(row)
            self.assert_binding(record, binding)
            return record

        return self._write(operation, connection)

    def find(
        self,
        run_id: str,
        paper_id: str,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> AnalysisDispatchRecord | None:
        active = connection or self.database.connection
        row = active.execute(
            "SELECT * FROM analysis_dispatches WHERE run_id = ? AND paper_id = ?",
            (run_id, paper_id),
        ).fetchone()
        return None if row is None else _record(row)

    @staticmethod
    def assert_binding(
        record: AnalysisDispatchRecord,
        binding: AnalysisDispatchBinding,
        *,
        allow_legacy_terminal: bool = False,
        legacy_config_hash: str | None = None,
    ) -> None:
        """Reject policy or payload drift for a durable paid-call intent.

        Migration 016 cannot reproduce the RFC-8785 dispatch id for an older
        row in SQL.  Its conservative terminal tombstones therefore use a
        marked legacy id, while every recoverable/new intent remains an exact
        binding match.  Unknown legacy policy fields are tolerated only for
        such a terminal tombstone: it can never consume another dispatch.
        """
        legacy_terminal = allow_legacy_terminal and _is_legacy_terminal(record)
        fields = (
            "artifact_hash",
            "artifact_id",
            "input_scope",
            "implementation_version",
            "profile",
            "model_id",
            "prompt_hash",
            "schema_hash",
        )
        if any(getattr(record, name) != getattr(binding, name) for name in fields):
            raise AnalysisDispatchStateError("analysis dispatch binding is immutable")
        if record.config_hash != binding.config_hash and not (
            legacy_terminal and record.config_hash == legacy_config_hash
        ):
            raise AnalysisDispatchStateError("analysis dispatch binding is immutable")
        if not legacy_terminal and record.dispatch_id != binding.dispatch_id:
            raise AnalysisDispatchStateError("analysis dispatch binding is immutable")
        if record.policy_version != binding.policy_version and not (
            legacy_terminal and record.policy_version == "unavailable"
        ):
            raise AnalysisDispatchStateError("analysis dispatch policy binding is immutable")
        if record.policy_hash != binding.policy_hash and not (
            legacy_terminal and record.policy_hash == "legacy-unavailable"
        ):
            raise AnalysisDispatchStateError("analysis dispatch policy binding is immutable")

    def get(
        self,
        dispatch_id: str,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> AnalysisDispatchRecord:
        active = connection or self.database.connection
        row = active.execute(
            "SELECT * FROM analysis_dispatches WHERE dispatch_id = ?", (dispatch_id,)
        ).fetchone()
        if row is None:
            raise KeyError(dispatch_id)
        return _record(row)

    def record_manual(
        self,
        dispatch_id: str,
        decision: ProcessingDecision,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> AnalysisDispatchRecord:
        if decision.is_authorized:
            raise ValueError("an authorized processing decision cannot be recorded as manual")
        decision_json = _json_text(_decision_document(decision))

        def operation(active: sqlite3.Connection) -> AnalysisDispatchRecord:
            active.execute(
                """UPDATE analysis_dispatches
                   SET status = 'manual_required', processing_decision_json = ?,
                       processing_grant_id = NULL, error_json = NULL,
                       updated_at = CURRENT_TIMESTAMP
                   WHERE dispatch_id = ? AND dispatch_count = 0
                     AND status IN ('prepared', 'manual_required')""",
                (decision_json, dispatch_id),
            )
            return self.get(dispatch_id, connection=active)

        return self._write(operation, connection)

    def claim(
        self,
        dispatch_id: str,
        decision: ProcessingDecision,
        *,
        owner: str,
        prompt_input_hash: str,
        now: datetime | str,
        lease_seconds: int,
        connection: sqlite3.Connection | None = None,
    ) -> AnalysisDispatchClaim | None:
        if not decision.is_authorized:
            raise ValueError("only an authorized processing decision may claim a dispatch")
        if not owner:
            raise ValueError("dispatch owner is required")
        _validate_hash(prompt_input_hash, "prompt_input_hash")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        moment = _moment(now)
        expires_at = _ordered_timestamp(moment + timedelta(seconds=lease_seconds))
        decision_json = _json_text(_decision_document(decision))

        def operation(active: sqlite3.Connection) -> AnalysisDispatchClaim | None:
            cursor = active.execute(
                """UPDATE analysis_dispatches
                   SET status = 'running', dispatch_count = 1,
                       lease_owner = ?, lease_token = lease_token + 1,
                       lease_expires_at = ?, prompt_input_hash = ?,
                       processing_decision_json = ?, processing_grant_id = ?,
                       analysis_run_id = NULL, error_json = NULL,
                       completed_at = NULL, updated_at = CURRENT_TIMESTAMP
                   WHERE dispatch_id = ? AND dispatch_count = 0
                     AND status IN ('prepared', 'manual_required')""",
                (
                    owner,
                    expires_at,
                    prompt_input_hash,
                    decision_json,
                    decision.processing_grant_id,
                    dispatch_id,
                ),
            )
            if cursor.rowcount != 1:
                return None
            current = self.get(dispatch_id, connection=active)
            return AnalysisDispatchClaim(
                dispatch_id=current.dispatch_id,
                owner=owner,
                token=current.lease_token,
                lease_expires_at=expires_at,
            )

        return self._write(operation, connection)

    def expire_stale(
        self,
        dispatch_id: str,
        *,
        now: datetime | str,
        connection: sqlite3.Connection | None = None,
    ) -> AnalysisDispatchRecord:
        expired_at = _ordered_timestamp(now)
        failure = _json_text(
            {
                "error": "UncertainDispatch",
                "message": "analysis dispatch lease expired; remote invocation outcome is uncertain",
                "reason": "lease_expired",
            }
        )

        def operation(active: sqlite3.Connection) -> AnalysisDispatchRecord:
            active.execute(
                """UPDATE analysis_dispatches
                   SET status = 'failed_terminal', lease_owner = NULL,
                       lease_expires_at = NULL, error_json = ?,
                       completed_at = ?, updated_at = CURRENT_TIMESTAMP
                   WHERE dispatch_id = ? AND status = 'running'
                     AND lease_expires_at <= ?""",
                (failure, expired_at, dispatch_id, expired_at),
            )
            return self.get(dispatch_id, connection=active)

        return self._write(operation, connection)

    def complete(
        self,
        claim: AnalysisDispatchClaim,
        *,
        analysis_run_id: str,
        invocation_id: str,
        rendered_prompt_hash: str,
        invocation_metadata: Mapping[str, Any],
        now: datetime | str,
        connection: sqlite3.Connection | None = None,
    ) -> AnalysisDispatchRecord:
        if not analysis_run_id or not invocation_id:
            raise ValueError("analysis_run_id and invocation_id are required")
        _validate_hash(rendered_prompt_hash, "rendered_prompt_hash", allow_non_digest=True)
        completed_at = _ordered_timestamp(now)
        metadata_json = _json_text(dict(invocation_metadata))

        def operation(active: sqlite3.Connection) -> AnalysisDispatchRecord:
            cursor = active.execute(
                """UPDATE analysis_dispatches
                   SET status = 'complete', lease_owner = NULL, lease_expires_at = NULL,
                       invocation_id = ?, invocation_metadata_json = ?,
                       rendered_prompt_hash = ?, analysis_run_id = ?, error_json = NULL,
                       completed_at = ?, updated_at = CURRENT_TIMESTAMP
                   WHERE dispatch_id = ? AND status = 'running' AND dispatch_count = 1
                     AND lease_owner = ? AND lease_token = ?
                     AND lease_expires_at > ?""",
                (
                    invocation_id,
                    metadata_json,
                    rendered_prompt_hash,
                    analysis_run_id,
                    completed_at,
                    claim.dispatch_id,
                    claim.owner,
                    claim.token,
                    completed_at,
                ),
            )
            if cursor.rowcount != 1:
                raise AnalysisDispatchStateError(
                    "analysis dispatch completion lost its active lease; outcome is uncertain"
                )
            return self.get(claim.dispatch_id, connection=active)

        return self._write(operation, connection)

    def fail_terminal(
        self,
        claim: AnalysisDispatchClaim,
        *,
        error: Mapping[str, Any],
        now: datetime | str,
        invocation_id: str | None = None,
        rendered_prompt_hash: str | None = None,
        invocation_metadata: Mapping[str, Any] | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> AnalysisDispatchRecord:
        failed_at = _ordered_timestamp(now)
        error_json = _json_text(dict(error))
        metadata_json = None if invocation_metadata is None else _json_text(dict(invocation_metadata))

        def operation(active: sqlite3.Connection) -> AnalysisDispatchRecord:
            cursor = active.execute(
                """UPDATE analysis_dispatches
                   SET status = 'failed_terminal', lease_owner = NULL, lease_expires_at = NULL,
                       invocation_id = COALESCE(?, invocation_id),
                       invocation_metadata_json = COALESCE(?, invocation_metadata_json),
                       rendered_prompt_hash = COALESCE(?, rendered_prompt_hash),
                       error_json = ?, completed_at = ?, updated_at = CURRENT_TIMESTAMP
                   WHERE dispatch_id = ? AND status = 'running' AND dispatch_count = 1
                     AND lease_owner = ? AND lease_token = ?""",
                (
                    invocation_id,
                    metadata_json,
                    rendered_prompt_hash,
                    error_json,
                    failed_at,
                    claim.dispatch_id,
                    claim.owner,
                    claim.token,
                ),
            )
            current = self.get(claim.dispatch_id, connection=active)
            if cursor.rowcount == 1 or current.status in {
                AnalysisDispatchStatus.COMPLETE,
                AnalysisDispatchStatus.FAILED_TERMINAL,
            }:
                return current
            raise AnalysisDispatchStateError("analysis dispatch failure lost its claimed lease")

        return self._write(operation, connection)

    def link_analysis_run(
        self,
        dispatch_id: str,
        analysis_run_id: str,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> AnalysisDispatchRecord:
        if not analysis_run_id:
            raise ValueError("analysis_run_id is required")

        def operation(active: sqlite3.Connection) -> AnalysisDispatchRecord:
            cursor = active.execute(
                """UPDATE analysis_dispatches
                   SET analysis_run_id = ?, updated_at = CURRENT_TIMESTAMP
                   WHERE dispatch_id = ?
                     AND status IN ('manual_required', 'complete', 'failed_terminal')""",
                (analysis_run_id, dispatch_id),
            )
            if cursor.rowcount != 1:
                raise AnalysisDispatchStateError("analysis run cannot be linked in the current dispatch state")
            return self.get(dispatch_id, connection=active)

        return self._write(operation, connection)

    def _write(self, operation: Any, connection: sqlite3.Connection | None) -> Any:
        if connection is not None:
            return operation(connection)
        with self.database.transaction() as active:
            return operation(active)

    @staticmethod
    def _validate_binding(binding: AnalysisDispatchBinding) -> None:
        for name in (
            "run_id",
            "paper_id",
            "config_hash",
            "implementation_version",
            "profile",
            "model_id",
            "prompt_hash",
            "schema_hash",
            "policy_version",
            "policy_hash",
        ):
            if not getattr(binding, name):
                raise ValueError(f"{name} is required")
        _validate_hash(binding.artifact_hash, "artifact_hash")
        if binding.input_scope not in {"full_pdf", "abstract_only", "metadata_only"}:
            raise ValueError("invalid analysis dispatch input_scope")
        if binding.profile != "stage4_analysis_luna" or binding.model_id != "gpt-5.6-luna":
            raise ValueError("analysis dispatch requires the frozen Stage 4 profile and model")
        _validate_hash(binding.policy_hash, "policy_hash")


def _record(row: sqlite3.Row) -> AnalysisDispatchRecord:
    return AnalysisDispatchRecord(
        dispatch_id=row["dispatch_id"],
        run_id=row["run_id"],
        paper_id=row["paper_id"],
        artifact_hash=row["artifact_hash"],
        artifact_id=row["artifact_id"],
        input_scope=row["input_scope"],
        config_hash=row["config_hash"],
        implementation_version=row["implementation_version"],
        profile=row["profile"],
        model_id=row["model_id"],
        prompt_hash=row["prompt_hash"],
        schema_hash=row["schema_hash"],
        policy_version=row["policy_version"],
        policy_hash=row["policy_hash"],
        stable_created_at=row["stable_created_at"],
        prompt_input_hash=row["prompt_input_hash"],
        rendered_prompt_hash=row["rendered_prompt_hash"],
        processing_decision=_json_object(row["processing_decision_json"]),
        processing_grant_id=row["processing_grant_id"],
        status=AnalysisDispatchStatus(row["status"]),
        dispatch_count=int(row["dispatch_count"]),
        lease_owner=row["lease_owner"],
        lease_token=int(row["lease_token"]),
        lease_expires_at=row["lease_expires_at"],
        invocation_id=row["invocation_id"],
        invocation_metadata=_json_object(row["invocation_metadata_json"]),
        analysis_run_id=row["analysis_run_id"],
        error=_json_object(row["error_json"]),
        completed_at=row["completed_at"],
    )


def _is_legacy_terminal(record: AnalysisDispatchRecord) -> bool:
    return (
        record.status is AnalysisDispatchStatus.FAILED_TERMINAL
        and record.dispatch_id.startswith("analysis-dispatch-legacy-")
        and record.error is not None
        and record.error.get("reason") in {
            "pre_migration_failed",
            "pre_migration_running",
        }
    )


def _decision_document(decision: ProcessingDecision) -> dict[str, Any]:
    document = asdict(decision)
    document["outcome"] = decision.outcome.value
    return document


def _json_object(value: str | None) -> Mapping[str, Any] | None:
    if value is None:
        return None
    loaded = json.loads(value)
    if not isinstance(loaded, dict):
        raise AnalysisDispatchStateError("analysis dispatch JSON fields must contain objects")
    return loaded


def _json_text(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _validate_hash(value: str, name: str, *, allow_non_digest: bool = False) -> None:
    if allow_non_digest and value:
        return
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _moment(value: datetime | str) -> datetime:
    if isinstance(value, str):
        try:
            moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("dispatch timestamps must be ISO-8601") from error
    else:
        moment = value
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def _timestamp(value: datetime | str) -> str:
    return _moment(value).isoformat().replace("+00:00", "Z")


def _ordered_timestamp(value: datetime | str) -> str:
    """Return a fixed-width UTC timestamp suitable for SQLite text ordering."""
    return _moment(value).isoformat(timespec="microseconds").replace("+00:00", "Z")
