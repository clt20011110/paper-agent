from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from hashlib import sha256
import json
from pathlib import Path
from threading import Event, Thread
from time import monotonic, sleep
from typing import Any

import pytest

from paper_agent.storage import Database
from paper_agent.workflow import (
    FileRef,
    FilterStep,
    SearchStep,
    SequentialWorkflowOrchestrator,
    StageIdentity,
    StageKind,
    StageOutcome,
    StepObservation,
    StopToken,
    WorkflowManifest,
    load_workflow_manifest,
)


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _ref(path: Path) -> FileRef:
    return FileRef(path.name, _digest(path), path)


def _manifest(tmp_path: Path, *, two_steps: bool = False) -> WorkflowManifest:
    config = tmp_path / "config.json"
    plan = tmp_path / "plan.json"
    release = tmp_path / "release.json"
    for path, content in (
        (config, "{}"),
        (plan, '{"query": "test"}'),
        (release, '{"release": "v1"}'),
    ):
        path.write_text(content, encoding="utf-8")
    steps: tuple[Any, ...] = (
        SearchStep("search", _ref(plan), _ref(release), (), False),
    )
    if two_steps:
        steps += (FilterStep("filter", _ref(plan), _ref(release), None),)
    return WorkflowManifest("fixture", _ref(config), steps, tmp_path / "workflow.json")


def _database(tmp_path: Path) -> Database:
    database = Database(tmp_path / "papers.sqlite3")
    database.migrate()
    return database


@dataclass
class FakeAdapter:
    outcomes: dict[str, StageOutcome] = field(default_factory=dict)
    observations: dict[str, StepObservation] = field(default_factory=dict)
    before_return: Any = None
    executions: list[str] = field(default_factory=list)

    def validate(self, _context: Any, spec: Any) -> StageIdentity:
        return StageIdentity(sha256(spec.step_id.encode()).hexdigest())

    def observe(self, _context: Any, spec: Any, _identity: StageIdentity) -> StepObservation:
        return self.observations.get(spec.step_id, StepObservation.PENDING)

    def execute(self, _context: Any, spec: Any, _identity: StageIdentity) -> StageOutcome:
        self.executions.append(spec.step_id)
        if self.before_return is not None:
            self.before_return(spec)
        return self.outcomes.get(spec.step_id, StageOutcome("complete", {"step": spec.step_id}))


def _adapters(adapter: FakeAdapter) -> dict[StageKind, FakeAdapter]:
    return {stage: adapter for stage in StageKind}


def test_manifest_is_strictly_typed_and_file_refs_are_frozen(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, two_steps=True)
    document = manifest.document()
    path = tmp_path / "workflow.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    loaded = load_workflow_manifest(path)
    assert loaded.document() == document

    invalid = json.loads(path.read_text(encoding="utf-8"))
    invalid["steps"][0]["argv"] = ["search"]
    path.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected or missing"):
        load_workflow_manifest(path)

    invalid = manifest.document()
    invalid["config"]["path"] = "../outside.json"
    path.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(ValueError, match="relative"):
        load_workflow_manifest(path)

    (tmp_path / "config.json").write_text('{"changed": true}', encoding="utf-8")
    with pytest.raises(ValueError, match="drifted"):
        manifest.verify_files()


def test_dry_run_validates_without_database_writes_or_execution(tmp_path: Path) -> None:
    database = _database(tmp_path)
    try:
        adapter = FakeAdapter()
        result = SequentialWorkflowOrchestrator(
            database, _manifest(tmp_path), _adapters(adapter)
        ).run("dry-1", dry_run=True)

        assert result["status"] == "validated"
        assert adapter.executions == []
        assert database.connection.execute("SELECT COUNT(*) FROM workflow_runs").fetchone()[0] == 0
        assert database.connection.execute("SELECT COUNT(*) FROM workflow_steps").fetchone()[0] == 0
    finally:
        database.close()


def test_resume_skips_persisted_completed_step(tmp_path: Path) -> None:
    database = _database(tmp_path)
    try:
        manifest = _manifest(tmp_path, two_steps=True)
        adapter = FakeAdapter(outcomes={"filter": StageOutcome("incomplete", {"reason": "pause"})})
        first = SequentialWorkflowOrchestrator(database, manifest, _adapters(adapter)).run("resume-1")
        assert first["status"] == "incomplete"
        assert adapter.executions == ["search", "filter"]

        adapter.outcomes["filter"] = StageOutcome("complete", {"step": "filter"})
        resumed = SequentialWorkflowOrchestrator(database, manifest, _adapters(adapter)).resume("resume-1")
        assert resumed["status"] == "complete"
        assert adapter.executions == ["search", "filter", "filter"]
    finally:
        database.close()


def test_blocked_step_can_be_claimed_again_after_its_blocker_is_cleared(tmp_path: Path) -> None:
    database = _database(tmp_path)
    try:
        manifest = _manifest(tmp_path)
        adapter = FakeAdapter(outcomes={"search": StageOutcome("blocked", {"reason": "grant_required"})})
        first = SequentialWorkflowOrchestrator(database, manifest, _adapters(adapter)).run("blocked-1")
        assert first["status"] == "blocked"
        assert first["steps"][0]["status"] == "blocked"

        adapter.outcomes["search"] = StageOutcome("complete", {"step": "search"})
        resumed = SequentialWorkflowOrchestrator(database, manifest, _adapters(adapter)).resume("blocked-1")
        assert resumed["status"] == "complete"
        assert resumed["steps"][0]["status"] == "complete"
        assert adapter.executions == ["search", "search"]
    finally:
        database.close()


def test_stop_token_stops_only_at_a_stage_boundary(tmp_path: Path) -> None:
    database = _database(tmp_path)
    try:
        token = StopToken()
        adapter = FakeAdapter(before_return=lambda spec: token.request_stop(15) if spec.step_id == "search" else None)
        result = SequentialWorkflowOrchestrator(
            database, _manifest(tmp_path, two_steps=True), _adapters(adapter), stop_token=token
        ).run("stop-1")

        assert token.signal == 15
        assert result["status"] == "incomplete"
        assert adapter.executions == ["search"]
        assert [item["status"] for item in result["steps"]] == ["complete", "pending"]
    finally:
        database.close()


def test_concurrent_resume_is_fenced_and_heartbeat_keeps_ownership(tmp_path: Path) -> None:
    database = _database(tmp_path)
    entered = Event()
    release = Event()
    adapter = FakeAdapter()
    lease_ttl = timedelta(seconds=1)

    def block(_spec: Any) -> None:
        entered.set()
        assert release.wait(10)

    adapter.before_return = block
    manifest = _manifest(tmp_path)
    result: dict[str, Any] = {}
    errors: list[BaseException] = []

    def run_first() -> None:
        try:
            with Database(database.path) as first_database:
                first = SequentialWorkflowOrchestrator(
                    first_database,
                    manifest,
                    _adapters(adapter),
                    owner_id="owner-a",
                    lease_ttl=lease_ttl,
                )
                result.update(first.run("fenced-1"))
        except BaseException as error:
            errors.append(error)

    thread = Thread(target=run_first)
    thread.start()
    try:
        assert entered.wait(2)

        def lease_expiry() -> datetime:
            row = database.connection.execute(
                "SELECT lease_expires_at FROM workflow_runs WHERE workflow_run_id = ?",
                ("fenced-1",),
            ).fetchone()
            assert row is not None and row["lease_expires_at"] is not None
            return datetime.fromisoformat(
                str(row["lease_expires_at"]).replace("Z", "+00:00")
            )

        original_expiry = lease_expiry()
        deadline = monotonic() + 5
        while monotonic() < deadline:
            renewed_expiry = lease_expiry()
            if renewed_expiry > original_expiry:
                break
            sleep(0.02)
        else:
            pytest.fail("workflow heartbeat did not renew the original lease")

        probe_time = original_expiry + (renewed_expiry - original_expiry) / 2
        with Database(database.path) as second_database:
            second = SequentialWorkflowOrchestrator(
                second_database,
                manifest,
                _adapters(adapter),
                owner_id="owner-b",
                clock=lambda: probe_time,
                lease_ttl=lease_ttl,
            ).resume("fenced-1")
            assert second["outcome"] == "already_running"
            assert adapter.executions == ["search"]
    finally:
        release.set()
        thread.join(5)
        database.close()

    assert not thread.is_alive()
    assert not errors
    assert result["status"] == "complete"


def test_resume_rejects_file_drift_before_acquiring_or_writing(tmp_path: Path) -> None:
    database = _database(tmp_path)
    try:
        manifest = _manifest(tmp_path)
        adapter = FakeAdapter(outcomes={"search": StageOutcome("incomplete", {"reason": "pause"})})
        SequentialWorkflowOrchestrator(database, manifest, _adapters(adapter)).run("drift-1")
        before = database.connection.execute(
            "SELECT status, updated_at FROM workflow_runs WHERE workflow_run_id = 'drift-1'"
        ).fetchone()

        (tmp_path / "plan.json").write_text('{"query": "changed"}', encoding="utf-8")
        with pytest.raises(ValueError, match="drifted"):
            SequentialWorkflowOrchestrator(database, manifest, _adapters(adapter)).resume("drift-1")

        after = database.connection.execute(
            "SELECT status, updated_at FROM workflow_runs WHERE workflow_run_id = 'drift-1'"
        ).fetchone()
        assert tuple(after) == tuple(before)
    finally:
        database.close()
