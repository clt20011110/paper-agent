from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import pytest

from paper_agent.domain import FilterStatus
from paper_agent.stage2_backends import (
    AdjudicationDecision,
    AdjudicationInput,
    RerankBatchError,
    RerankInput,
    RerankScore,
    Stage2BackendError,
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
