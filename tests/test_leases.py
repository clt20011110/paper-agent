from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from paper_agent.leases import LeaseQueue
from paper_agent.storage import Database


NOW = "2026-08-09T00:00:00Z"


def _queue(database: Database, run_id: str = "run-1") -> LeaseQueue:
    database.connection.execute(
        """
        INSERT INTO pipeline_runs(run_id, stage, status, input_hash, config_hash, implementation_version)
        VALUES (?, 'stage-2', 'running', 'input', 'config', 'test')
        """,
        (run_id,),
    )
    database.connection.commit()
    return LeaseQueue(database)


def test_enqueue_is_idempotent_for_the_same_logical_output(tmp_path) -> None:
    with Database(tmp_path / "papers.sqlite3") as database:
        database.migrate()
        queue = _queue(database)

        first = queue.enqueue(
            run_id="run-1", stage="stage-2", paper_id=None, output_kind="screening", input_hash="a", now=NOW
        )
        second = queue.enqueue(
            run_id="run-1", stage="stage-2", paper_id=None, output_kind="screening", input_hash="a", now="2026-08-10T00:00:00Z"
        )

        assert second == first
        with pytest.raises(ValueError, match="different input_hash"):
            queue.enqueue(
                run_id="run-1", stage="stage-2", paper_id=None, output_kind="screening", input_hash="changed", now="2026-08-10T00:00:00Z"
            )
        assert database.connection.execute("SELECT COUNT(*) FROM task_leases").fetchone()[0] == 1


def test_two_connections_cannot_claim_the_same_task(tmp_path) -> None:
    path = tmp_path / "papers.sqlite3"
    with Database(path) as first_database, Database(path) as second_database:
        first_database.migrate()
        first_queue = _queue(first_database)
        second_queue = LeaseQueue(second_database)
        task = first_queue.enqueue(
            run_id="run-1", stage="stage-2", paper_id=None, output_kind="screening", input_hash="a", now=NOW
        )

        first_claim = first_queue.claim(
            worker_id="worker-a", now=NOW, expires_at="2026-08-09T00:05:00Z", limit=1
        )
        second_claim = second_queue.claim(
            worker_id="worker-b", now=NOW, expires_at="2026-08-09T00:05:00Z", limit=1
        )

        assert [lease.task_id for lease in first_claim] == [task.task_id]
        assert second_claim == ()


def test_expired_task_is_reclaimed_with_a_new_fencing_token(tmp_path) -> None:
    with Database(tmp_path / "papers.sqlite3") as database:
        database.migrate()
        queue = _queue(database)
        queue.enqueue(run_id="run-1", stage="stage-2", paper_id=None, output_kind="screening", input_hash="a", now=NOW)

        first = queue.claim(worker_id="worker-a", now=NOW, expires_at="2026-08-09T00:01:00Z", limit=1)[0]
        second = queue.claim(
            worker_id="worker-b", now="2026-08-09T00:01:00Z", expires_at="2026-08-09T00:02:00Z", limit=1
        )[0]

        assert (second.attempt, second.fencing_token, second.worker_id) == (2, 2, "worker-b")
        assert not queue.complete(
            task_id=first.task_id, worker_id="worker-a", fencing_token=first.fencing_token, now="2026-08-09T00:01:00Z"
        )


def test_complete_is_idempotent_and_requires_the_current_fence(tmp_path) -> None:
    with Database(tmp_path / "papers.sqlite3") as database:
        database.migrate()
        queue = _queue(database)
        queue.enqueue(run_id="run-1", stage="stage-2", paper_id=None, output_kind="screening", input_hash="a", now=NOW)
        lease = queue.claim(worker_id="worker-a", now=NOW, expires_at="2026-08-09T00:05:00Z", limit=1)[0]

        assert not queue.complete(task_id=lease.task_id, worker_id="worker-a", fencing_token=lease.fencing_token + 1, now=NOW)
        assert queue.complete(task_id=lease.task_id, worker_id="worker-a", fencing_token=lease.fencing_token, now=NOW)
        assert not queue.complete(task_id=lease.task_id, worker_id="worker-a", fencing_token=lease.fencing_token, now=NOW)


def test_failed_retryable_tasks_can_be_claimed_again_and_resume_is_precise(tmp_path) -> None:
    with Database(tmp_path / "papers.sqlite3") as database:
        database.migrate()
        queue = _queue(database)
        queue.enqueue(run_id="run-1", stage="stage-2", paper_id=None, output_kind="screening", input_hash="a", now=NOW)
        lease = queue.claim(worker_id="worker-a", now=NOW, expires_at="2026-08-09T00:05:00Z", limit=1)[0]

        assert queue.fail(
            task_id=lease.task_id,
            worker_id="worker-a",
            fencing_token=lease.fencing_token,
            now="2026-08-09T00:01:00Z",
            retryable=True,
            error_json='{"code":"temporary"}',
        )
        retried = queue.claim(
            worker_id="worker-b", now="2026-08-09T00:02:00Z", expires_at="2026-08-09T00:05:00Z", limit=1
        )[0]
        assert (retried.attempt, retried.fencing_token) == (2, 2)
        assert queue.resume(now="2026-08-09T00:03:00Z") == 0
        assert queue.resume(now="2026-08-09T00:05:00Z", run_id="run-1") == 1
        assert queue.claim(
            worker_id="worker-c", now="2026-08-09T00:05:00Z", expires_at="2026-08-09T00:06:00Z", limit=1
        )[0].attempt == 3


def test_parallel_workers_claim_each_task_once(tmp_path) -> None:
    path = tmp_path / "papers.sqlite3"
    with Database(path) as database:
        database.migrate()
        queue = _queue(database)
        for index in range(20):
            queue.enqueue(
                run_id="run-1",
                stage="stage-2",
                paper_id=None,
                output_kind=f"screening-{index}",
                input_hash="a",
                now=NOW,
            )

    def claim(worker_id: str) -> tuple[str, ...]:
        with Database(path) as database:
            leases = LeaseQueue(database).claim(
                worker_id=worker_id,
                now=NOW,
                expires_at="2026-08-09T00:05:00Z",
                limit=5,
            )
            return tuple(lease.task_id for lease in leases)

    with ThreadPoolExecutor(max_workers=4) as workers:
        claimed = tuple(task_id for batch in workers.map(claim, (f"worker-{index}" for index in range(4))) for task_id in batch)

    assert len(claimed) == 20
    assert len(set(claimed)) == 20
