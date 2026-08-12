from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from paper_agent.stage2_backends import OmlxResponse, ThresholdArtifact
from paper_agent.stage2_evaluation import ReplayError, Stage2Decision
from paper_agent.stage2_pipeline import Stage2Paper, Stage2Profile
from paper_agent.stage2_structured_replay import (
    StructuredReplayRunner,
    freeze_structured_replay_manifest,
)


class FakeTransport:
    def __init__(self, respond):
        self.respond = respond
        self.calls: dict[str, int] = {}
        self.requests: list[tuple[str, dict[str, object]]] = []
        self.attempts: list[tuple[str, int]] = []

    def request(self, path: str, payload: dict[str, object]) -> OmlxResponse:
        self.requests.append((path, payload))
        messages = payload["messages"]
        assert isinstance(messages, list)
        user = messages[1]
        assert isinstance(user, dict)
        paper_id = str(user["content"]).split("Paper ID: ", 1)[1].split("\n", 1)[0]
        attempt = self.calls.get(paper_id, 0) + 1
        self.calls[paper_id] = attempt
        self.attempts.append((paper_id, attempt))
        return self.respond(paper_id, attempt)


def _profile() -> Stage2Profile:
    return Stage2Profile(
        query="frozen topic",
        query_version="topic-v1",
        thresholds=ThresholdArtifact("thresholds-v1", "reranker-lock", "raw_reranker_score", -1, 1),
        reranker_model_id="reranker",
        reranker_revision="reranker-revision",
        adjudicator_model_id="qwen",
        adjudicator_revision="qwen-revision",
        screening_scope_hash="0" * 64,
        adjudicator_lock_hash="qwen-lock",
        adjudicator_concurrency=1,
    )


def _papers() -> tuple[Stage2Paper, ...]:
    return tuple(Stage2Paper(f"paper-{index:04d}", f"Paper {index}", "A frozen abstract.") for index in range(1_000))


def _decision(paper_id: str) -> OmlxResponse:
    return OmlxResponse(200, json.dumps({
        "model": "qwen",
        "choices": [{"message": {"content": json.dumps({
            "paper_id": paper_id,
            "decision": "relevant",
            "score": 0.9,
            "reason_codes": ["topic_match"],
            "rationale": "The paper directly addresses the frozen topic.",
            "evidence_fields": ["title", "abstract"],
        })}}],
    }).encode())


def test_runner_freezes_and_writes_a_successful_gate(tmp_path: Path) -> None:
    transport = FakeTransport(lambda paper_id, _attempt: _decision(paper_id))
    run = StructuredReplayRunner(_profile(), transport).run(
        _papers(), manifest_path=tmp_path / "manifest.json", records_path=tmp_path / "records.json",
    )

    assert run.result.gate.passed
    assert len(run.records) == 1_000
    assert len(run.manifest.model_lock_hash) == 64
    assert all(record.final_valid and record.model_retries == 0 for record in run.records)
    assert transport.requests[0][0] == "/v1/chat/completions"
    assert transport.requests[0][1]["structured_outputs"]
    assert json.loads((tmp_path / "records.json").read_text())["records"][0]["pair_id"] == "paper-0000"
    with pytest.raises(FileExistsError, match="refusing to replace"):
        StructuredReplayRunner(_profile(), transport).run(
            _papers(), manifest_path=tmp_path / "manifest.json", records_path=tmp_path / "again.json",
        )


def test_runner_retries_one_schema_error_and_preserves_raw_leaks() -> None:
    def respond(paper_id: str, attempt: int) -> OmlxResponse:
        if paper_id == "paper-0000" and attempt == 1:
            return OmlxResponse(200, b'{"model":"qwen","choices":[{"message":{"content":"<think>hidden</think>{\\"paper_id\\":\\"paper-0000\\"}"}}]}')
        return _decision(paper_id)

    run = StructuredReplayRunner(_profile(), FakeTransport(respond)).run(_papers())
    first = run.records[0]

    assert first.first_error is ReplayError.SCHEMA
    assert first.first_think_tag_leak
    assert first.first_schema_outside_text
    assert first.first_returned_pair_id == "paper-0000"
    assert first.model_retries == 1 and first.retry_error is ReplayError.NONE
    assert first.final_valid and first.final_returned_pair_id == "paper-0000"
    assert first.deterministic_repairs == 0


def test_runner_retries_only_after_the_complete_parallel_first_round() -> None:
    def respond(paper_id: str, attempt: int) -> OmlxResponse:
        if paper_id == "paper-0000" and attempt == 1:
            return OmlxResponse(200, b'{"model":"qwen","choices":[{"message":{"content":"not json"}}]}')
        return _decision(paper_id)

    transport = FakeTransport(respond)
    run = StructuredReplayRunner(
        replace(_profile(), adjudicator_concurrency=8), transport,
    ).run(_papers())

    retry_index = transport.attempts.index(("paper-0000", 2))
    assert retry_index == 1_000
    assert all(attempt == 1 for _, attempt in transport.attempts[:retry_index])
    assert run.records[0].model_retries == 1 and run.records[0].final_valid


def test_runner_routes_a_final_invalid_response_to_needs_review() -> None:
    def respond(paper_id: str, _attempt: int) -> OmlxResponse:
        if paper_id == "paper-0000":
            return OmlxResponse(200, b'{"model":"qwen","choices":[{"message":{"content":"not json"}}]}')
        return _decision(paper_id)

    run = StructuredReplayRunner(_profile(), FakeTransport(respond)).run(_papers())
    record = run.records[0]

    assert record.first_error is ReplayError.SCHEMA
    assert record.model_retries == 1 and record.retry_error is ReplayError.SCHEMA
    assert not record.final_valid
    assert record.final_decision is Stage2Decision.NEEDS_REVIEW


def test_runner_classifies_timeout_and_service_before_a_successful_retry() -> None:
    def respond(paper_id: str, attempt: int) -> OmlxResponse:
        if attempt == 1 and paper_id == "paper-0000":
            raise TimeoutError("request timed out")
        if attempt == 1 and paper_id == "paper-0001":
            return OmlxResponse(503, b'{"error":"busy"}')
        return _decision(paper_id)

    run = StructuredReplayRunner(_profile(), FakeTransport(respond)).run(_papers())

    assert run.records[0].first_error is ReplayError.TIMEOUT
    assert run.records[1].first_error is ReplayError.SERVICE
    assert all(record.retry_error is ReplayError.NONE for record in run.records[:2])
    assert run.result.timeouts == 1 and run.result.service_errors == 1


def test_runner_rejects_manifest_or_output_drift_before_requests(tmp_path: Path) -> None:
    papers = _papers()
    profile = _profile()
    transport = FakeTransport(lambda paper_id, _attempt: _decision(paper_id))
    manifest = freeze_structured_replay_manifest(papers, profile)
    changed = (Stage2Paper(papers[0].paper_id, "Changed title", papers[0].abstract), *papers[1:])

    with pytest.raises(ValueError, match="papers and profile"):
        StructuredReplayRunner(profile, transport).run(changed, manifest=manifest)
    assert transport.requests == []

    output = tmp_path / "manifest.json"
    output.write_text("reserved", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to replace"):
        StructuredReplayRunner(profile, transport).run(
            papers,
            manifest_path=output,
            records_path=tmp_path / "records.json",
        )
    assert transport.requests == []


def test_runner_treats_grammar_warning_http_400_and_model_mismatch_as_schema_errors() -> None:
    def respond(paper_id: str, attempt: int) -> OmlxResponse:
        if paper_id == "paper-0000" and attempt == 1:
            return OmlxResponse(400, b'{"error":"grammar failed"}')
        if paper_id == "paper-0001" and attempt == 1:
            return OmlxResponse(200, b'{}', {"Warning": "grammar skipped"})
        if paper_id == "paper-0002" and attempt == 1:
            response = json.loads(_decision(paper_id).body)
            response["model"] = "other-model"
            return OmlxResponse(200, json.dumps(response).encode())
        return _decision(paper_id)

    run = StructuredReplayRunner(_profile(), FakeTransport(respond)).run(_papers())

    assert [record.first_error for record in run.records[:3]] == [
        ReplayError.SCHEMA,
        ReplayError.SCHEMA,
        ReplayError.SCHEMA,
    ]
    assert all(record.model_retries == 1 and record.final_valid for record in run.records[:3])


def test_runner_requires_at_least_one_thousand_unique_papers() -> None:
    transport = FakeTransport(lambda paper_id, _attempt: _decision(paper_id))
    with pytest.raises(ValueError, match="at least 1,000"):
        StructuredReplayRunner(_profile(), transport).run(_papers()[:-1])
    duplicate = (*_papers()[:-1], _papers()[0])
    with pytest.raises(ValueError, match="must be unique"):
        StructuredReplayRunner(_profile(), transport).run(duplicate)
    assert transport.requests == []
