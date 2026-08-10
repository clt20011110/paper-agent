from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from threading import Barrier, Event
from typing import Sequence

import pytest

from paper_agent.domain import FilterStatus
from paper_agent.leases import LeaseNotCurrent, LeaseQueue, TaskLeaseSpec
from paper_agent.stage2_backends import (
    AdjudicationDecision,
    AdjudicationInput,
    RerankBatchError,
    RerankInput,
    RerankScore,
    Stage2BackendError,
    StructuredOutputError,
    ThresholdArtifact,
)
from paper_agent.stage2_pipeline import Stage2Paper, Stage2Pipeline, Stage2Profile
from paper_agent.storage import Database


@dataclass
class FakeReranker:
    scores: dict[str, float]
    fail: bool = False
    backend_name: str = "fake_reranker"
    requests: list[tuple[str, tuple[RerankInput, ...]]] = field(default_factory=list)

    def rerank(self, query: str, documents: Sequence[RerankInput]) -> tuple[RerankScore, ...]:
        self.requests.append((query, tuple(documents)))
        if self.fail:
            raise Stage2BackendError("service unavailable")
        return tuple(RerankScore(item.paper_id, self.scores[item.paper_id]) for item in documents)


@dataclass
class PartiallyFailingReranker(FakeReranker):
    failed_paper_ids: tuple[str, ...] = ()

    def rerank(self, query: str, documents: Sequence[RerankInput]) -> tuple[RerankScore, ...]:
        self.requests.append((query, tuple(documents)))
        scores = tuple(
            RerankScore(item.paper_id, self.scores[item.paper_id])
            for item in documents
            if item.paper_id not in self.failed_paper_ids
        )
        raise RerankBatchError(scores, self.failed_paper_ids)


@dataclass
class FakeAdjudicator:
    decisions: dict[str, AdjudicationDecision]
    backend_name: str = "fake_adjudicator"
    requests: list[AdjudicationInput] = field(default_factory=list)

    def adjudicate(self, request: AdjudicationInput) -> AdjudicationDecision:
        self.requests.append(request)
        return self.decisions[request.paper_id]


@dataclass
class SequencedAdjudicator:
    outcomes: dict[str, list[AdjudicationDecision | Exception]]
    backend_name: str = "sequenced_adjudicator"
    requests: list[AdjudicationInput] = field(default_factory=list)

    def adjudicate(self, request: AdjudicationInput) -> AdjudicationDecision:
        self.requests.append(request)
        outcome = self.outcomes[request.paper_id].pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@dataclass
class SequencedReranker:
    scores: list[float]
    backend_name: str = "sequenced_reranker"
    requests: list[tuple[str, tuple[RerankInput, ...]]] = field(default_factory=list)

    def rerank(self, query: str, documents: Sequence[RerankInput]) -> tuple[RerankScore, ...]:
        self.requests.append((query, tuple(documents)))
        score = self.scores.pop(0)
        return tuple(RerankScore(item.paper_id, score) for item in documents)


class InterruptingReranker:
    backend_name = "interrupting_reranker"

    def rerank(self, query: str, documents: Sequence[RerankInput]) -> tuple[RerankScore, ...]:
        raise KeyboardInterrupt("simulated worker crash")


class BarrierReranker:
    backend_name = "barrier_reranker"

    def __init__(self, barrier: Barrier) -> None:
        self.barrier = barrier
        self.batch_sizes: list[int] = []

    def rerank(self, query: str, documents: Sequence[RerankInput]) -> tuple[RerankScore, ...]:
        self.batch_sizes.append(len(documents))
        self.barrier.wait(timeout=5)
        return tuple(RerankScore(item.paper_id, 3.0) for item in documents)


@dataclass
class SequenceClock:
    moments: list[datetime]

    def __call__(self) -> datetime:
        if len(self.moments) > 1:
            return self.moments.pop(0)
        return self.moments[0]


def _profile() -> Stage2Profile:
    return Stage2Profile(
        query="versioned research topic",
        query_version="topic-v1",
        thresholds=ThresholdArtifact("thresholds-v1", "reranker.lock", "raw_reranker_score", -1.0, 2.0),
        reranker_model_id="reranker-model",
        reranker_revision="reranker-revision",
        adjudicator_model_id="qwen-model",
        adjudicator_revision="qwen-revision",
        token_bucket_width=1_000,
        adjudicator_concurrency=2,
    )


def _papers() -> tuple[Stage2Paper, ...]:
    return (
        Stage2Paper("editorial", "Editorial", "irrelevant note", document_type="editorial"),
        Stage2Paper("low", "Low score", "ordinary abstract"),
        Stage2Paper("high", "High score", "ordinary abstract"),
        Stage2Paper("gray", "Gray score", "ordinary abstract"),
        Stage2Paper("missing", "Missing abstract", None),
    )


def _database(tmp_path) -> Database:
    database = Database(tmp_path / "papers.sqlite3")
    database.migrate()
    with database.transaction() as connection:
        for paper in _papers():
            connection.execute("INSERT INTO papers(paper_id, title, abstract) VALUES (?, ?, ?)", (paper.paper_id, paper.title, paper.abstract))
    return database


def test_stage2_batches_reranking_adjudicates_anomalies_and_persists_immutable_provenance(tmp_path) -> None:
    database = _database(tmp_path)
    reranker = FakeReranker({"low": -2.0, "high": 3.0, "gray": 0.0, "missing": 3.0})
    adjudicator = FakeAdjudicator({
        "gray": AdjudicationDecision("gray", "relevant", 0.8, ("topic_match",), "This addresses the topic.", ("abstract",)),
        "missing": AdjudicationDecision("missing", "irrelevant", 0.1, ("insufficient_topic",), "No topic signal.", ("title",)),
    })
    pipeline = Stage2Pipeline(database, reranker, adjudicator, _profile())

    summary = pipeline.run("stage2-run", _papers())

    assert [item.paper_id for item in summary.decisions] == ["editorial", "gray", "high", "low", "missing"]
    assert {item.paper_id: item.status for item in summary.decisions} == {
        "editorial": FilterStatus.IRRELEVANT,
        "low": FilterStatus.IRRELEVANT,
        "high": FilterStatus.RELEVANT,
        "gray": FilterStatus.RELEVANT,
        "missing": FilterStatus.IRRELEVANT,
    }
    assert len(reranker.requests) == 1
    query, documents = reranker.requests[0]
    assert query == "versioned research topic"
    assert [item.paper_id for item in documents] == ["gray", "high", "low", "missing"]
    assert all(item.document.startswith("Title: ") and "\nAbstract: " in item.document and "\nKeywords: " in item.document for item in documents)
    assert [item.paper_id for item in adjudicator.requests] == ["gray", "missing"]
    assert summary.qwen_count == 2
    assert summary.qwen_share == 0.4
    assert summary.qwen_alarms == ("qwen_share_over_15_percent", "qwen_share_over_30_percent")

    rows = database.connection.execute(
        "SELECT paper_id, input_hash, reason, model_id, model_revision FROM filter_decisions WHERE run_id = ? ORDER BY paper_id",
        ("stage2-run",),
    ).fetchall()
    assert len(rows) == 5
    gray = next(row for row in rows if row["paper_id"] == "gray")
    assert "This addresses the topic." in gray["reason"]
    low = next(row for row in rows if row["paper_id"] == "low")
    assert "rationale" not in low["reason"]
    assert gray["model_id"] == "qwen-model"
    editorial = next(row for row in rows if row["paper_id"] == "editorial")
    assert editorial["model_id"] is None
    events = database.connection.execute("SELECT COUNT(*) FROM screening_events WHERE run_id = ?", ("stage2-run",)).fetchone()[0]
    assert events == 5
    assert database.connection.execute(
        "SELECT status FROM pipeline_runs WHERE run_id = 'stage2-run'"
    ).fetchone()[0] == "complete"
    database.close()


def test_stage2_resume_skips_exact_inputs_and_changed_input_requires_a_new_run(tmp_path) -> None:
    database = _database(tmp_path)
    reranker = FakeReranker({"low": -2.0, "high": 3.0, "gray": 0.0, "missing": 3.0})
    adjudicator = FakeAdjudicator({
        "gray": AdjudicationDecision("gray", "relevant", 0.8, ("topic_match",), "kept", ("abstract",)),
        "missing": AdjudicationDecision("missing", "relevant", 0.8, ("title_only",), "kept", ("title",)),
    })
    pipeline = Stage2Pipeline(database, reranker, adjudicator, _profile())
    pipeline.run("resume-run", _papers())
    rerank_calls = len(reranker.requests)
    adjudication_calls = len(adjudicator.requests)

    resumed = pipeline.run("resume-run", _papers())

    assert len(reranker.requests) == rerank_calls
    assert len(adjudicator.requests) == adjudication_calls
    assert all(item.resumed for item in resumed.decisions)
    assert resumed.reranked_count == 0
    assert resumed.qwen_count == 2

    changed = tuple(
        Stage2Paper("high", "Changed high score", "ordinary abstract") if item.paper_id == "high" else item
        for item in _papers()
    )
    with pytest.raises(ValueError, match="immutable"):
        pipeline.run("resume-run", changed)

    changed_summary = pipeline.run("resume-run-v2", (changed[2],))

    assert len(reranker.requests) == rerank_calls + 1
    assert [item.paper_id for item in reranker.requests[-1][1]] == ["high"]
    assert next(item for item in changed_summary.decisions if item.paper_id == "high").resumed is False
    assert database.connection.execute("SELECT COUNT(*) FROM filter_decisions WHERE run_id = ?", ("resume-run",)).fetchone()[0] == 5
    assert database.connection.execute("SELECT COUNT(*) FROM filter_decisions WHERE run_id = ?", ("resume-run-v2",)).fetchone()[0] == 1
    database.close()


def test_stage2_backend_and_schema_failures_never_auto_exclude(tmp_path) -> None:
    database = _database(tmp_path)
    reranker = FakeReranker({}, fail=True)
    adjudicator = FakeAdjudicator({})
    pipeline = Stage2Pipeline(database, reranker, adjudicator, _profile())

    summary = pipeline.run("failure-run", _papers())

    decisions = {item.paper_id: item for item in summary.decisions}
    assert decisions["editorial"].status is FilterStatus.IRRELEVANT
    for paper_id in ("gray", "high", "low", "missing"):
        assert decisions[paper_id].status is FilterStatus.NEEDS_REVIEW
        assert decisions[paper_id].reason_code == "reranker_backend_failure"
    assert not adjudicator.requests
    database.close()


def test_stage2_preserves_successful_peers_when_reranker_isolates_failures(tmp_path) -> None:
    database = _database(tmp_path)
    reranker = PartiallyFailingReranker(
        {"low": -2.0, "high": 3.0, "gray": 0.0, "missing": 3.0},
        failed_paper_ids=("high",),
    )
    adjudicator = FakeAdjudicator({
        "gray": AdjudicationDecision("gray", "relevant", 0.8, ("topic_match",), "kept", ("abstract",)),
        "missing": AdjudicationDecision("missing", "relevant", 0.8, ("title_only",), "kept", ("title",)),
    })
    summary = Stage2Pipeline(database, reranker, adjudicator, _profile()).run(
        "partial-rerank-failure", _papers()
    )

    decisions = {item.paper_id: item for item in summary.decisions}
    assert decisions["high"].status is FilterStatus.NEEDS_REVIEW
    assert decisions["high"].reason_code == "reranker_backend_failure"
    assert decisions["low"].status is FilterStatus.IRRELEVANT
    assert decisions["gray"].status is FilterStatus.RELEVANT
    assert decisions["missing"].status is FilterStatus.RELEVANT
    assert [request.paper_id for request in adjudicator.requests] == ["gray", "missing"]
    database.close()


def test_stage2_invalid_adjudication_is_needs_review_with_a_standard_reason(tmp_path) -> None:
    database = _database(tmp_path)
    reranker = FakeReranker({"low": -2.0, "high": 3.0, "gray": 0.0, "missing": 3.0})
    adjudicator = FakeAdjudicator({
        "gray": AdjudicationDecision("wrong-paper", "irrelevant", 0.1, ("bad",), "bad", ("abstract",)),
        "missing": AdjudicationDecision("missing", "relevant", 0.8, ("title",), "kept", ("title",)),
    })
    pipeline = Stage2Pipeline(database, reranker, adjudicator, _profile())

    summary = pipeline.run("schema-failure-run", _papers())

    gray = next(item for item in summary.decisions if item.paper_id == "gray")
    assert gray.status is FilterStatus.NEEDS_REVIEW
    assert gray.reason_code == "adjudicator_schema_failure"
    database.close()


def test_stage2_retries_one_structured_output_failure_and_persists_telemetry(tmp_path) -> None:
    database = _database(tmp_path)
    reranker = FakeReranker({"low": -2.0, "high": 3.0, "gray": 0.0, "missing": 3.0})
    adjudicator = SequencedAdjudicator({
        "gray": [
            StructuredOutputError("invalid schema"),
            AdjudicationDecision("gray", "relevant", 0.8, ("topic_match",), "kept", ("abstract",)),
        ],
        "missing": [
            AdjudicationDecision("missing", "relevant", 0.8, ("title_only",), "kept", ("title",)),
        ],
    })

    summary = Stage2Pipeline(database, reranker, adjudicator, _profile()).run(
        "retry-success", _papers()
    )

    gray = next(decision for decision in summary.decisions if decision.paper_id == "gray")
    assert gray.status is FilterStatus.RELEVANT
    assert gray.adjudicator_attempt_count == 2
    assert gray.adjudicator_retry_reason == "adjudicator_schema_failure"
    assert gray.adjudicator_retry_outcome == "succeeded"
    assert sum(request.paper_id == "gray" for request in adjudicator.requests) == 2
    row = database.connection.execute(
        """SELECT status, adjudicator_attempt_count, adjudicator_retry_reason,
                  adjudicator_retry_outcome, reason
           FROM filter_decisions WHERE run_id = ? AND paper_id = 'gray'""",
        ("retry-success",),
    ).fetchone()
    assert tuple(row[:4]) == ("relevant", 2, "adjudicator_schema_failure", "succeeded")
    assert json.loads(row["reason"])["adjudicator_retry_outcome"] == "succeeded"
    database.close()


def test_stage2_persists_terminal_retry_failure_and_resume_does_not_recall_qwen(tmp_path) -> None:
    database = _database(tmp_path)
    reranker = FakeReranker({"low": -2.0, "high": 3.0, "gray": 0.0, "missing": 3.0})
    adjudicator = SequencedAdjudicator({
        "gray": [Stage2BackendError("temporary"), StructuredOutputError("still invalid")],
        "missing": [
            AdjudicationDecision("missing", "relevant", 0.8, ("title_only",), "kept", ("title",)),
        ],
    })
    pipeline = Stage2Pipeline(database, reranker, adjudicator, _profile())

    first = pipeline.run("retry-terminal", _papers())

    gray = next(decision for decision in first.decisions if decision.paper_id == "gray")
    assert gray.status is FilterStatus.NEEDS_REVIEW
    assert gray.reason_code == "adjudicator_schema_failure"
    assert gray.adjudicator_attempt_count == 2
    assert gray.adjudicator_retry_reason == "adjudicator_backend_failure"
    assert gray.adjudicator_retry_outcome == "failed"
    calls = len(adjudicator.requests)

    resumed = pipeline.run("retry-terminal", _papers())

    assert len(adjudicator.requests) == calls
    assert all(decision.resumed for decision in resumed.decisions)
    row = database.connection.execute(
        """SELECT status, adjudicator_attempt_count, adjudicator_retry_reason,
                  adjudicator_retry_outcome
           FROM filter_decisions WHERE run_id = ? AND paper_id = 'gray'""",
        ("retry-terminal",),
    ).fetchone()
    assert tuple(row) == ("needs_review", 2, "adjudicator_backend_failure", "failed")
    database.close()


def test_stage2_two_sqlite_workers_compete_without_duplicate_results(tmp_path) -> None:
    path = tmp_path / "parallel.sqlite3"
    papers = tuple(
        Stage2Paper(f"paper-{index:03d}", f"Paper {index}", "ordinary abstract")
        for index in range(128)
    )
    with Database(path) as database:
        database.migrate()
        with database.transaction() as connection:
            connection.executemany(
                "INSERT INTO papers(paper_id, title, abstract) VALUES (?, ?, ?)",
                ((paper.paper_id, paper.title, paper.abstract) for paper in papers),
            )

    barrier = Barrier(2)

    def screen(worker_id: str) -> tuple[int, list[int]]:
        with Database(path) as database:
            reranker = BarrierReranker(barrier)
            pipeline = Stage2Pipeline(
                database,
                reranker,
                FakeAdjudicator({}),
                _profile(),
                worker_id=worker_id,
                peer_wait_seconds=5,
                lease_poll_seconds=0.005,
            )
            summary = pipeline.run("parallel-run", papers)
            return len(summary.decisions), reranker.batch_sizes

    with ThreadPoolExecutor(max_workers=2) as workers:
        results = tuple(workers.map(screen, ("worker-a", "worker-b")))

    assert sorted(results) == [(128, [64]), (128, [64])]
    with Database(path) as database:
        leases = database.connection.execute(
            """SELECT worker_id, COUNT(*) AS task_count, MIN(attempt) AS min_attempt,
                      MAX(attempt) AS max_attempt, MIN(fencing_token) AS min_token,
                      MAX(fencing_token) AS max_token,
                      SUM(lease_expires_at IS NOT NULL) AS expiry_count
               FROM task_leases WHERE run_id = 'parallel-run'
               GROUP BY worker_id ORDER BY worker_id"""
        ).fetchall()
        assert [tuple(row) for row in leases] == [
            ("worker-a", 64, 1, 1, 1, 1, 64),
            ("worker-b", 64, 1, 1, 1, 1, 64),
        ]
        assert database.connection.execute(
            "SELECT COUNT(*) FROM filter_decisions WHERE run_id = 'parallel-run'"
        ).fetchone()[0] == 128
        assert database.connection.execute(
            "SELECT COUNT(*) FROM screening_events WHERE run_id = 'parallel-run'"
        ).fetchone()[0] == 128


def test_stage2_expired_model_result_is_dropped_before_atomic_publish(tmp_path) -> None:
    path = tmp_path / "expired.sqlite3"
    paper = Stage2Paper("paper-stale", "Stale result", "ordinary abstract")
    with Database(path) as database:
        database.migrate()
        database.connection.execute(
            "INSERT INTO papers(paper_id, title, abstract) VALUES (?, ?, ?)",
            (paper.paper_id, paper.title, paper.abstract),
        )
        database.connection.commit()
        clock = SequenceClock([
            datetime(2026, 8, 9, 0, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 8, 9, 0, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 8, 9, 0, 0, 2, tzinfo=timezone.utc),
            datetime(2026, 8, 9, 0, 0, 2, tzinfo=timezone.utc),
            datetime(2026, 8, 9, 0, 0, 2, 500_000, tzinfo=timezone.utc),
        ])
        reranker = SequencedReranker([-2.0, 3.0])
        summary = Stage2Pipeline(
            database,
            reranker,
            FakeAdjudicator({}),
            _profile(),
            worker_id="worker-retry",
            lease_seconds=1,
            lease_clock=clock,
        ).run("expired-run", (paper,))

        assert summary.decisions[0].status is FilterStatus.RELEVANT
        assert len(reranker.requests) == 2
        lease = database.connection.execute(
            """SELECT status, worker_id, attempt, fencing_token
               FROM task_leases WHERE run_id = 'expired-run'"""
        ).fetchone()
        assert tuple(lease) == ("complete", "worker-retry", 2, 2)
        decision = database.connection.execute(
            "SELECT reason FROM filter_decisions WHERE run_id = 'expired-run'"
        ).fetchone()
        provenance = json.loads(decision["reason"])
        assert provenance["reranker_score"] == 3.0
        assert provenance["task_lease"]["fencing_token"] == 2
        assert database.connection.execute(
            "SELECT COUNT(*) FROM screening_events WHERE run_id = 'expired-run'"
        ).fetchone()[0] == 1


def test_stage2_rechecks_expiry_after_waiting_for_the_sqlite_write_lock(tmp_path) -> None:
    path = tmp_path / "locked-expiry.sqlite3"
    paper = Stage2Paper("paper-locked", "Locked result", "ordinary abstract")
    claimed_at = "2026-08-09T00:00:00.000000Z"
    expires_at = "2026-08-09T00:00:01.000000Z"
    expired = datetime(2026, 8, 9, 0, 0, 2, tzinfo=timezone.utc)
    with Database(path) as setup:
        setup.migrate()
        setup.connection.execute(
            "INSERT INTO papers(paper_id, title, abstract) VALUES (?, ?, ?)",
            (paper.paper_id, paper.title, paper.abstract),
        )
        setup.connection.commit()
        pipeline = Stage2Pipeline(
            setup,
            FakeReranker({paper.paper_id: 3.0}),
            FakeAdjudicator({}),
            _profile(),
            worker_id="locked-worker",
        )
        pipeline._ensure_run("locked-run", (paper,))
        queue = LeaseQueue(setup)
        queue.enqueue_many(
            run_id="locked-run",
            stage="stage-2",
            specs=(
                TaskLeaseSpec(
                    paper.paper_id,
                    pipeline._lease_output_kind(paper),
                    pipeline.input_hash(paper),
                ),
            ),
            now=claimed_at,
        )
        lease = queue.claim(
            worker_id="locked-worker",
            now=claimed_at,
            expires_at=expires_at,
            limit=1,
            run_id="locked-run",
        )[0]
        decision = pipeline._screen_batch((paper,))[0]

    begin_attempted = Event()
    clock_calls: list[datetime] = []

    def publish() -> None:
        with Database(path) as worker_database:
            worker_database.connection.set_trace_callback(
                lambda statement: begin_attempted.set()
                if statement.strip().upper() == "BEGIN IMMEDIATE"
                else None
            )
            worker = Stage2Pipeline(
                worker_database,
                FakeReranker({}),
                FakeAdjudicator({}),
                _profile(),
                worker_id="locked-worker",
                lease_clock=lambda: clock_calls.append(expired) or expired,
            )
            worker._persist_claimed("locked-run", ((lease, decision),))

    with Database(path) as blocker:
        blocker.connection.execute("BEGIN IMMEDIATE")
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(publish)
            assert begin_attempted.wait(timeout=2)
            assert not future.done()
            assert clock_calls == []
            blocker.connection.rollback()
            with pytest.raises(LeaseNotCurrent, match="expired"):
                future.result(timeout=2)

    with Database(path) as database:
        assert database.connection.execute(
            "SELECT COUNT(*) FROM filter_decisions WHERE run_id = 'locked-run'"
        ).fetchone()[0] == 0
        assert database.connection.execute(
            "SELECT status FROM task_leases WHERE run_id = 'locked-run'"
        ).fetchone()[0] == "running"


def test_stage2_resume_reclaims_work_left_by_an_interrupted_worker(tmp_path) -> None:
    path = tmp_path / "resume-lease.sqlite3"
    paper = Stage2Paper("paper-resume", "Resume me", "ordinary abstract")
    first_moment = datetime(2026, 8, 9, 0, 0, 0, tzinfo=timezone.utc)
    resumed_moment = datetime(2026, 8, 9, 0, 0, 2, tzinfo=timezone.utc)
    with Database(path) as first_database:
        first_database.migrate()
        first_database.connection.execute(
            "INSERT INTO papers(paper_id, title, abstract) VALUES (?, ?, ?)",
            (paper.paper_id, paper.title, paper.abstract),
        )
        first_database.connection.commit()
        crashed = Stage2Pipeline(
            first_database,
            InterruptingReranker(),
            FakeAdjudicator({}),
            _profile(),
            worker_id="worker-crashed",
            lease_seconds=1,
            lease_clock=lambda: first_moment,
        )
        with pytest.raises(KeyboardInterrupt, match="simulated worker crash"):
            crashed.run("crash-run", (paper,))

        running = first_database.connection.execute(
            """SELECT status, worker_id, attempt, fencing_token
               FROM task_leases WHERE run_id = 'crash-run'"""
        ).fetchone()
        assert tuple(running) == ("running", "worker-crashed", 1, 1)

        with Database(path) as resumed_database:
            summary = Stage2Pipeline(
                resumed_database,
                FakeReranker({paper.paper_id: 3.0}),
                FakeAdjudicator({}),
                _profile(),
                worker_id="worker-resumed",
                lease_seconds=1,
                lease_clock=lambda: resumed_moment,
            ).run("crash-run", (paper,))

            assert summary.decisions[0].status is FilterStatus.RELEVANT
            completed = resumed_database.connection.execute(
                """SELECT status, worker_id, attempt, fencing_token
                   FROM task_leases WHERE run_id = 'crash-run'"""
            ).fetchone()
            assert tuple(completed) == ("complete", "worker-resumed", 2, 2)


def test_stage2_claims_large_model_batches_instead_of_per_paper_calls(tmp_path) -> None:
    path = tmp_path / "batch.sqlite3"
    papers = tuple(
        Stage2Paper(f"batch-{index:03d}", f"Batch paper {index}", "ordinary abstract")
        for index in range(70)
    )
    with Database(path) as database:
        database.migrate()
        with database.transaction() as connection:
            connection.executemany(
                "INSERT INTO papers(paper_id, title, abstract) VALUES (?, ?, ?)",
                ((paper.paper_id, paper.title, paper.abstract) for paper in papers),
            )
        reranker = FakeReranker({paper.paper_id: 3.0 for paper in papers})

        summary = Stage2Pipeline(
            database,
            reranker,
            FakeAdjudicator({}),
            _profile(),
            worker_id="batch-worker",
        ).run("batch-run", papers)

        assert len(summary.decisions) == 70
        assert [len(documents) for _query, documents in reranker.requests] == [64, 6]
        assert database.connection.execute(
            """SELECT COUNT(*) FROM task_leases
               WHERE run_id = 'batch-run' AND status = 'complete'
                 AND worker_id = 'batch-worker' AND attempt = 1 AND fencing_token = 1"""
        ).fetchone()[0] == 70
