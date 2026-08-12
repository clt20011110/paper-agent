from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from threading import Lock

import pytest

from paper_agent.stage2_backends import OmlxResponse, ThresholdArtifact as LegacyThresholdArtifact
from paper_agent.stage2_evaluation import CalibrationPath, PathCalibrator, ThresholdArtifact
from paper_agent.stage2_hidden_submission import (
    HiddenPromotionSubmissionRunner,
    hidden_submission_cases,
)
from paper_agent.stage2_pipeline import PathCalibration, Stage2Profile
from paper_agent.stage2_promotion_artifacts import load_promotion_submission
from paper_agent.stage2_search import ReleasedStage2

from test_stage2_dev_calibration import _inputs


def _hash(value: object) -> str:
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class FakeTransport:
    def __init__(
        self,
        *,
        fail_rerank: bool = False,
        rerank_score: float = 8.0,
        wrong_model_once: bool = False,
    ) -> None:
        self.fail_rerank = fail_rerank
        self.rerank_score = rerank_score
        self.wrong_model_once = wrong_model_once
        self.chat_count = 0
        self.chat_attempts: list[tuple[str, int]] = []
        self.wrong_model_pair_id: str | None = None
        self._chat_counts: dict[str, int] = {}
        self._chat_lock = Lock()
        self.requests: list[tuple[str, dict[str, object]]] = []

    def request(self, path: str, payload: dict[str, object]) -> OmlxResponse:
        self.requests.append((path, payload))
        if path == "/v1/rerank":
            if self.fail_rerank:
                return OmlxResponse(503, b'{"error":"busy"}')
            documents = payload["documents"]
            assert isinstance(documents, list)
            return OmlxResponse(200, json.dumps({"results": [
                {"index": index, "relevance_score": self.rerank_score}
                for index in range(len(documents))
            ]}).encode())
        if path == "/v1/chat/completions":
            messages = payload["messages"]
            assert isinstance(messages, list)
            user = messages[1]
            assert isinstance(user, dict)
            pair_id = str(user["content"]).split("Paper ID: ", 1)[1].split("\n", 1)[0]
            with self._chat_lock:
                attempt = self._chat_counts.get(pair_id, 0) + 1
                self._chat_counts[pair_id] = attempt
                wrong_model = self.wrong_model_once and self.wrong_model_pair_id is None
                if wrong_model:
                    self.wrong_model_pair_id = pair_id
                self.chat_count += 1
                self.chat_attempts.append((pair_id, attempt))
            model = "wrong-model" if wrong_model else payload["model"]
            return OmlxResponse(200, json.dumps({
                "model": model,
                "choices": [{"message": {"content": json.dumps({
                    "paper_id": pair_id,
                    "decision": "needs_review",
                    "score": 0.5,
                    "reason_codes": ["uncertain"],
                    "rationale": "Insufficient evidence.",
                    "evidence_fields": ["title"],
                })}}],
            }).encode())
        raise AssertionError(f"unexpected model path: {path}")


def _candidate(manifest) -> ReleasedStage2:
    query_rows = tuple(
        (f"topic-{index}", "zh" if index % 2 == 0 else "en", f"query-{index}")
        for index in range(6)
    )
    kwargs = dict(
        query="combined query",
        query_version="fixture-v1",
        thresholds=None,
        reranker_model_id="bge",
        reranker_revision="bge-r1",
        adjudicator_model_id="qwen",
        adjudicator_revision="qwen-r1",
        screening_scope_hash="0" * 64,
        reranker_lock_hash="a" * 64,
        adjudicator_lock_hash="b" * 64,
        evaluation_topic_queries=query_rows,
        document_batch_size=64,
        reranker_max_in_flight=2,
        adjudicator_concurrency=4,
    )
    base = Stage2Profile(**kwargs)
    dev_ids = tuple(sorted(
        pair.pair_id for pair in manifest.pairs if pair.split.value == "dev"
    ))
    bindings = {}
    for path, lock in ((CalibrationPath.RERANKER, "a" * 64), (CalibrationPath.QWEN, "b" * 64)):
        calibrator = PathCalibrator(
            1, path, 1.0, 0.0, manifest.dev_hash(), manifest.hash(), lock,
            "c" * 64, _hash(list(dev_ids)), len(dev_ids), dev_ids,
        )
        threshold = ThresholdArtifact(
            1, path, 0.25, 0.75, calibrator.hash(), lock,
            manifest.dev_hash(), "c" * 64, base.base_runtime_config_hash,
        )
        bindings[path] = PathCalibration(calibrator, threshold)
    profile = Stage2Profile(
        **kwargs,
        reranker_calibration=bindings[CalibrationPath.RERANKER],
        adjudicator_calibration=bindings[CalibrationPath.QWEN],
    )
    return ReleasedStage2("candidate-v2", profile, "d" * 64, "http://127.0.0.1:8000")


def test_hidden_runner_writes_three_complete_query_bound_runs(tmp_path: Path) -> None:
    manifest, snapshot = _inputs()
    candidate = _candidate(manifest)
    transport = FakeTransport()
    output = tmp_path / "hidden-submission.json"

    submission = HiddenPromotionSubmissionRunner(candidate, transport).run(
        manifest, snapshot, output_path=output
    )

    assert submission.candidate_id == "candidate-v2"
    assert len(submission.runs) == 3
    assert all(len(run) == 300 for run in submission.runs)
    assert all(prediction.pair_id not in {
        pair.pair_id for pair in manifest.pairs if pair.split.value == "dev"
    } for run in submission.runs for prediction in run)
    incomplete_ids = {
        pair.pair_id for pair in manifest.pairs
        if pair.split.value != "dev" and pair.abstract_incomplete
    }
    assert all(
        prediction.path is (
            CalibrationPath.QWEN
            if prediction.pair_id in incomplete_ids
            else CalibrationPath.RERANKER
        )
        for run in submission.runs
        for prediction in run
    )
    assert load_promotion_submission(output, manifest=manifest) == submission
    reranks = [payload for path, payload in transport.requests if path == "/v1/rerank"]
    assert len(reranks) == 18
    assert {payload["query"] for payload in reranks} == {f"query-{index}" for index in range(6)}
    with pytest.raises(FileExistsError, match="refusing to replace"):
        HiddenPromotionSubmissionRunner(candidate, transport).run(manifest, snapshot, output_path=output)


def test_hidden_runner_fails_open_on_reranker_service_failure() -> None:
    manifest, snapshot = _inputs()
    candidate = _candidate(manifest)

    submission = HiddenPromotionSubmissionRunner(candidate, FakeTransport(fail_rerank=True)).run(manifest, snapshot)

    assert all(prediction.decision.value == "needs_review" for run in submission.runs for prediction in run)
    assert all(prediction.raw_score is None and prediction.probability == 0.5 for run in submission.runs for prediction in run)
    assert all(prediction.review_reason.value == "service_error" for run in submission.runs for prediction in run)


def test_hidden_runner_does_not_hide_programming_errors_as_service_failures() -> None:
    manifest, snapshot = _inputs()
    candidate = _candidate(manifest)

    class BrokenTransport(FakeTransport):
        def request(self, path, payload):
            if path == "/v1/rerank":
                raise AssertionError("implementation bug")
            return super().request(path, payload)

    with pytest.raises(AssertionError, match="implementation bug"):
        HiddenPromotionSubmissionRunner(candidate, BrokenTransport()).run(
            manifest, snapshot
        )


def test_hidden_runner_uses_qwen_for_the_calibrated_uncertainty_band() -> None:
    manifest, snapshot = _inputs()
    candidate = _candidate(manifest)
    transport = FakeTransport(rerank_score=0.0)

    submission = HiddenPromotionSubmissionRunner(candidate, transport).run(manifest, snapshot)

    assert all(prediction.path is CalibrationPath.QWEN for run in submission.runs for prediction in run)
    assert all(prediction.decision.value == "needs_review" for run in submission.runs for prediction in run)
    assert all(prediction.review_reason.value == "uncertain" for run in submission.runs for prediction in run)
    qwen_prompts = [str(payload["messages"][1]["content"]) for path, payload in transport.requests if path == "/v1/chat/completions"]
    assert len(qwen_prompts) == 900
    assert any("Query: query-0" in prompt for prompt in qwen_prompts)
    assert any("Query: query-5" in prompt for prompt in qwen_prompts)


def test_hidden_runner_retries_after_the_concurrent_batch_drains() -> None:
    manifest, snapshot = _inputs()
    candidate = _candidate(manifest)
    transport = FakeTransport(rerank_score=0.0, wrong_model_once=True)

    submission = HiddenPromotionSubmissionRunner(candidate, transport).run(
        manifest, snapshot
    )

    assert all(len(run) == 300 for run in submission.runs)
    qwen_requests = [
        payload for path, payload in transport.requests
        if path == "/v1/chat/completions"
    ]
    assert len(qwen_requests) == 901
    assert len({pair_id for pair_id, _ in transport.chat_attempts[:300]}) == 300
    assert all(attempt == 1 for _, attempt in transport.chat_attempts[:300])
    assert transport.chat_attempts[300] == (transport.wrong_model_pair_id, 2)


def test_hidden_cases_require_every_frozen_topic_language_query_before_model_calls() -> None:
    manifest, snapshot = _inputs()
    candidate = _candidate(manifest)
    profile = candidate.profile
    profile_without_one_query = Stage2Profile(
        **{
            name: getattr(profile, name)
            for name in (
                "query", "query_version", "reranker_model_id", "reranker_revision",
                "adjudicator_model_id", "adjudicator_revision", "screening_scope_hash",
                "reranker_lock_hash",
                "adjudicator_lock_hash", "release_gate_hash", "include_document_types",
                "exclude_document_types", "token_bucket_width", "document_batch_size",
                "reranker_max_in_flight", "adjudicator_concurrency", "adjudicator_seed",
                "adjudicator_max_context_window", "omlx_base_url", "api_key_env", "prompt_version",
                "schema_version",
            )
        },
        thresholds=LegacyThresholdArtifact("fixture", "a" * 64, "raw_reranker_score", -1.0, 1.0),
        evaluation_topic_queries=profile.evaluation_topic_queries[1:],
    )
    with pytest.raises(ValueError, match="no frozen evaluation query"):
        hidden_submission_cases(manifest, snapshot, profile_without_one_query)
