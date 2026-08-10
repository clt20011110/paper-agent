"""Durable, single-dispatch decisions for the authorized Stage 3 Luna planner."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import json
from typing import Any
from uuid import uuid4

from .storage import Database


@dataclass(frozen=True, slots=True)
class StoredLunaDecision:
    planner_decision_id: str
    selected: bool
    reason_code: str
    status: str
    page_state: str
    next_action: str
    invocation_metadata: Mapping[str, Any]


class Stage3LunaDecisionStore:
    """Persist one planner outcome per Stage 3 run and candidate.

    A pending row is deliberately never dispatched again: if a process dies
    after reserving the row, the paid-call result is uncertain and the paper
    returns to the attended manual queue instead of paying Luna twice.
    """

    def __init__(self, database: Database, run_id: str, authorization_grant_id: str) -> None:
        self.database = database
        self.run_id = run_id
        self.authorization_grant_id = authorization_grant_id

    def decide(self, control: Any, planner: Callable[[Any], Any], *, decided_at: str) -> StoredLunaDecision:
        existing = self._load(control.candidate_id)
        if existing is not None:
            return existing

        decision_id = f"stage3-luna-{uuid4()}"
        with self.database.transaction() as connection:
            connection.execute(
                """INSERT INTO stage3_luna_decisions(
                       planner_decision_id, run_id, candidate_id, authorization_grant_id,
                       status, reason_code, decided_at
                   ) VALUES (?, ?, ?, ?, 'pending', 'authorized_planner_pending', ?)""",
                (decision_id, self.run_id, control.candidate_id, self.authorization_grant_id, decided_at),
            )

        outcome = planner(control)
        metadata = dict(outcome.invocation_metadata)
        _validate_metadata(metadata)
        selected = bool(outcome.selected)
        result = StoredLunaDecision(
            decision_id,
            selected,
            str(outcome.reason_code),
            str(outcome.status),
            str(outcome.page_state),
            str(outcome.next_action),
            metadata,
        )
        with self.database.transaction() as connection:
            connection.execute(
                """UPDATE stage3_luna_decisions
                   SET status = 'complete', selected = ?, planner_status = ?,
                       page_state = ?, next_action = ?, reason_code = ?,
                       invocation_metadata_json = ?
                   WHERE planner_decision_id = ? AND status = 'pending'""",
                (
                    int(result.selected), result.status, result.page_state,
                    result.next_action, result.reason_code,
                    json.dumps(result.invocation_metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                    result.planner_decision_id,
                ),
            )
        return result

    def _load(self, candidate_id: str) -> StoredLunaDecision | None:
        row = self.database.connection.execute(
            """SELECT * FROM stage3_luna_decisions
               WHERE run_id = ? AND candidate_id = ?""",
            (self.run_id, candidate_id),
        ).fetchone()
        if row is None:
            return None
        if row["authorization_grant_id"] != self.authorization_grant_id:
            raise ValueError("persisted Luna planner decision has a different authorization grant")
        if row["status"] == "pending":
            return StoredLunaDecision(
                str(row["planner_decision_id"]), False,
                "authorized_planner_result_uncertain", "pending", "unknown", "manual_queue", {},
            )
        metadata = json.loads(row["invocation_metadata_json"])
        if not isinstance(metadata, Mapping):
            raise ValueError("persisted Luna planner metadata is invalid")
        _validate_metadata(metadata)
        return StoredLunaDecision(
            str(row["planner_decision_id"]), bool(row["selected"]), str(row["reason_code"]),
            str(row["planner_status"]), str(row["page_state"]), str(row["next_action"]), dict(metadata),
        )


def _validate_metadata(metadata: Mapping[str, Any]) -> None:
    expected = {
        "profile": "stage3_authorized_luna",
        "model": "gpt-5.6-luna",
        "reasoning_effort": "low",
        "actual_model": "gpt-5.6-luna",
        "actual_profile": "stage3_authorized_luna",
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("Luna invocation metadata does not match the frozen authorized-download profile")
