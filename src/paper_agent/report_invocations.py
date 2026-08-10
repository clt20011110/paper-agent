"""Run-wide uniqueness and replay bindings for paid Stage 4b invocations."""

from __future__ import annotations

from collections.abc import Mapping
import sqlite3
from typing import Any

from .canonical import content_hash


_PHASES = frozenset({"reduce", "audit_step", "audit_shard"})


class ReportInvocationError(RuntimeError):
    """A report invocation was reused or its persisted binding drifted."""


def register_report_invocation(
    connection: sqlite3.Connection,
    *,
    report_run_id: str,
    invocation_id: str,
    phase: str,
    node_key: str,
    metadata: Mapping[str, Any],
) -> None:
    """Atomically reserve one invocation identity for one report-run node."""
    _validate_identity(invocation_id, phase, node_key)
    try:
        connection.execute(
            """INSERT INTO report_sol_invocations(
                   report_run_id, invocation_id, phase, node_key, metadata_hash
               ) VALUES (?, ?, ?, ?, ?)""",
            (
                report_run_id,
                invocation_id,
                phase,
                node_key,
                report_invocation_metadata_hash(metadata),
            ),
        )
    except sqlite3.IntegrityError as error:
        raise ReportInvocationError(
            "Sol invocation identity is not fresh for this report run"
        ) from error


def require_report_invocation(
    connection: sqlite3.Connection,
    *,
    report_run_id: str,
    invocation_id: str,
    phase: str,
    node_key: str,
    metadata: Mapping[str, Any],
) -> None:
    """Verify a completed node against its immutable run-wide registry row."""
    _validate_identity(invocation_id, phase, node_key)
    row = connection.execute(
        """SELECT invocation_id, metadata_hash
           FROM report_sol_invocations
           WHERE report_run_id = ? AND phase = ? AND node_key = ?""",
        (report_run_id, phase, node_key),
    ).fetchone()
    if (
        row is None
        or row["invocation_id"] != invocation_id
        or row["metadata_hash"] != report_invocation_metadata_hash(metadata)
    ):
        raise ReportInvocationError(
            "persisted Sol invocation registry binding has drifted"
        )


def _validate_identity(invocation_id: str, phase: str, node_key: str) -> None:
    if phase not in _PHASES:
        raise ReportInvocationError("unsupported report invocation phase")
    if not invocation_id.strip() or not node_key.strip():
        raise ReportInvocationError("report invocation identity must not be empty")


def report_invocation_metadata_hash(metadata: Mapping[str, Any]) -> str:
    """Hash metadata while preserving compatibility with pre-path records."""
    document = dict(metadata)
    for key in ("schema_path", "prompt_path"):
        if document.get(key) is None:
            document.pop(key, None)
    return content_hash(document)
