import pytest

from paper_agent.runs import RunStatus, RunStore
from paper_agent.storage import Database


def test_run_lifecycle_and_resume_preserve_frozen_inputs(tmp_path) -> None:
    with Database(tmp_path / "papers.sqlite3") as database:
        database.migrate()
        runs = RunStore(database)
        run = runs.create(
            run_id="run-1",
            stage="stage2",
            input_hash="input",
            config_hash="config",
            implementation_version="2.0",
        )
        assert runs.create(
            run_id="run-1",
            stage="stage2",
            input_hash="input",
            config_hash="config",
            implementation_version="2.0",
        ) == run

        runs.transition("run-1", RunStatus.APPROVED, at="2026-08-09T00:00:00Z")
        runs.transition("run-1", RunStatus.RUNNING, at="2026-08-09T00:01:00Z")
        incomplete = runs.transition("run-1", RunStatus.INCOMPLETE, at="2026-08-09T00:02:00Z")
        assert runs.resumable() == (incomplete,)

        resumed = runs.transition("run-1", RunStatus.RUNNING, at="2026-08-09T00:03:00Z")
        assert resumed.started_at == "2026-08-09T00:03:00Z"
        assert resumed.completed_at is None


def test_run_id_drift_and_invalid_transition_are_rejected(tmp_path) -> None:
    with Database(tmp_path / "papers.sqlite3") as database:
        database.migrate()
        runs = RunStore(database)
        runs.create(
            run_id="run-1",
            stage="stage2",
            input_hash="input",
            config_hash="config",
            implementation_version="2.0",
        )
        with pytest.raises(ValueError, match="different frozen inputs"):
            runs.create(
                run_id="run-1",
                stage="stage2",
                input_hash="changed",
                config_hash="config",
                implementation_version="2.0",
            )
        with pytest.raises(ValueError, match="invalid run transition"):
            runs.transition("run-1", RunStatus.COMPLETE, at="2026-08-09T00:00:00Z")
