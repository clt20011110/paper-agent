from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from paper_agent.stage2_backends import OmlxResponse, ThresholdArtifact
from paper_agent.stage2_dev_calibration import (
    CalibrationPath,
    FrozenDevRawScoreArtifact,
    Stage2DevRawScoreRunner,
    dev_stage2_papers,
    load_frozen_dev_raw_scores,
)
from paper_agent.stage2_evaluation import GoldManifest, GoldPair, GoldSplit
from paper_agent.stage2_pipeline import Stage2Profile
from paper_agent.stage2_sampling import CorpusPaper, PrivateCorpusSnapshot


class FakeTransport:
    def __init__(
        self,
        fail_qwen: bool = False,
        wrong_model: bool = False,
        wrong_model_once: bool = False,
    ) -> None:
        self.fail_qwen = fail_qwen
        self.wrong_model = wrong_model
        self.wrong_model_once = wrong_model_once
        self.chat_count = 0
        self.requests: list[tuple[str, dict[str, object]]] = []

    def request(self, path: str, payload: dict[str, object]) -> OmlxResponse:
        self.requests.append((path, payload))
        if path == "/v1/rerank":
            documents = payload["documents"]
            assert isinstance(documents, list)
            return OmlxResponse(200, json.dumps({"results": [
                {"index": index, "relevance_score": index / 10} for index in range(len(documents))
            ]}).encode())
        if self.fail_qwen:
            return OmlxResponse(503, b'{"error":"busy"}')
        messages = payload["messages"]
        assert isinstance(messages, list)
        user = messages[1]
        assert isinstance(user, dict)
        pair_id = str(user["content"]).split("Paper ID: ", 1)[1].split("\n", 1)[0]
        wrong_model = self.wrong_model or (
            self.wrong_model_once and self.chat_count == 0
        )
        self.chat_count += 1
        return OmlxResponse(200, json.dumps({"model": "wrong-model" if wrong_model else payload["model"], "choices": [{"message": {"content": json.dumps({
            "paper_id": pair_id, "decision": "relevant", "score": 0.8,
            "reason_codes": ["topic_match"], "rationale": "Directly in scope.",
            "evidence_fields": ["title", "abstract"],
        })}}]}).encode())


def _inputs() -> tuple[GoldManifest, PrivateCorpusSnapshot]:
    corpus: list[CorpusPaper] = []
    for split, count in ((GoldSplit.DEV, 300), (GoldSplit.HIDDEN_HARD, 150), (GoldSplit.HIDDEN_REAL, 150)):
        for index in range(count):
            paper_id, topic = f"{split.value}-{index}", f"topic-{index % 6}"
            corpus.append(CorpusPaper(
                topic, paper_id, f"Title {paper_id}", "Abstract.", {"keywords": ["molecule"]},
                "crossref", "en" if index % 2 else "zh", f"family-{split.value}-{index}", 1.0, 0.5,
                abstract_incomplete=split is not GoldSplit.HIDDEN_REAL and index < count // 10,
            ))
    snapshot = PrivateCorpusSnapshot(1, "policy-v1", 7, tuple(corpus))
    pairs = tuple(GoldPair(
        paper.paper_id, paper.topic, paper.language, paper.source,
        0.5 if paper.paper_id.startswith("hidden_real-") else None,
        paper.paper_family, snapshot.corpus_hash,
        GoldSplit.HIDDEN_REAL if paper.paper_id.startswith("hidden_real-") else GoldSplit.HIDDEN_HARD
        if paper.paper_id.startswith("hidden_hard-") else GoldSplit.DEV,
        abstract_incomplete=paper.abstract_incomplete,
        sampled_from_natural_distribution=paper.paper_id.startswith("hidden_real-"),
        cross_language_match=paper.paper_id.endswith("-0"),
    ) for paper in corpus)
    return GoldManifest(1, snapshot.corpus_hash, pairs, ("en", "zh")), snapshot


def _profile() -> Stage2Profile:
    return Stage2Profile(
        query="molecular generation", query_version="topic-v1",
        thresholds=ThresholdArtifact("fixture", "a" * 64, "raw_reranker_score", -1, 1),
        reranker_model_id="bge", reranker_revision="bge-rev", adjudicator_model_id="qwen",
        adjudicator_revision="qwen-rev", screening_scope_hash="0" * 64,
        reranker_lock_hash="a" * 64, adjudicator_lock_hash="b" * 64,
        document_batch_size=64, reranker_max_in_flight=2, adjudicator_concurrency=7,
    )


def _runner(transport: FakeTransport) -> Stage2DevRawScoreRunner:
    return Stage2DevRawScoreRunner(_profile(), transport, {
        CalibrationPath.RERANKER: "a" * 64, CalibrationPath.QWEN: "b" * 64,
    }, {
        (f"topic-{topic}", "zh" if topic % 2 == 0 else "en"):
            f"query for topic {topic} in {'zh' if topic % 2 == 0 else 'en'}"
        for topic in range(6)
    })


def test_runner_collects_exact_unlabelled_dev_scores_and_writes_no_replace(tmp_path: Path) -> None:
    manifest, snapshot = _inputs()
    transport = FakeTransport()
    output = tmp_path / "raw-scores.json"

    artifact = _runner(transport).run(manifest, snapshot, output_path=output)

    assert set(artifact.scores) == {CalibrationPath.RERANKER, CalibrationPath.QWEN}
    assert all(len(scores) == 300 for scores in artifact.scores.values())
    assert artifact.gold_manifest_hash == manifest.hash()
    assert artifact.private_snapshot_hash == snapshot.hash()
    assert load_frozen_dev_raw_scores(output) == artifact
    rerank_requests = [payload for path, payload in transport.requests if path == "/v1/rerank"]
    assert len(rerank_requests) == 6
    assert {payload["query"] for payload in rerank_requests} == set(artifact.topic_queries.values())
    qwen_requests = [payload for path, payload in transport.requests if path == "/v1/chat/completions"]
    assert len(qwen_requests) == 300
    assert all(payload["structured_outputs"] for payload in qwen_requests)
    qwen_prompts = {str(payload["messages"][1]["content"]) for payload in qwen_requests}
    assert any("Query: query for topic 0 in zh" in prompt for prompt in qwen_prompts)
    assert any("Query: query for topic 1 in en" in prompt for prompt in qwen_prompts)
    with pytest.raises(FileExistsError, match="refusing to replace"):
        _runner(transport).run(manifest, snapshot, output_path=output)


def test_runner_does_not_publish_partial_artifact_when_qwen_fails(tmp_path: Path) -> None:
    manifest, snapshot = _inputs()
    output = tmp_path / "raw-scores.json"

    with pytest.raises(Exception, match="HTTP 503"):
        _runner(FakeTransport(fail_qwen=True)).run(manifest, snapshot, output_path=output)

    assert not output.exists()


def test_wrong_qwen_response_model_fails_without_publishing(tmp_path: Path) -> None:
    manifest, snapshot = _inputs()
    output = tmp_path / "raw-scores.json"

    with pytest.raises(Exception, match="does not match the frozen adjudicator"):
        _runner(FakeTransport(wrong_model=True)).run(manifest, snapshot, output_path=output)

    assert not output.exists()


def test_qwen_retries_each_pair_at_most_once() -> None:
    manifest, snapshot = _inputs()
    transport = FakeTransport(wrong_model_once=True)

    artifact = _runner(transport).run(manifest, snapshot)

    assert artifact.qwen_retry_count == 1
    qwen_requests = [
        payload for path, payload in transport.requests
        if path == "/v1/chat/completions"
    ]
    assert len(qwen_requests) == 301
    assert qwen_requests[-1]["messages"] == qwen_requests[0]["messages"]


def test_missing_topic_query_fails_before_any_model_call() -> None:
    manifest, snapshot = _inputs()
    transport = FakeTransport()
    runner = _runner(transport)
    runner.topic_queries = {
        key: query for key, query in runner.topic_queries.items() if key != ("topic-0", "zh")
    }

    with pytest.raises(ValueError, match="exactly cover"):
        runner.run(manifest, snapshot)

    assert transport.requests == []


def test_invalid_profile_lock_fails_before_any_model_call() -> None:
    manifest, snapshot = _inputs()
    transport = FakeTransport()
    runner = _runner(transport)
    runner.profile = replace(runner.profile, reranker_lock_hash="not-a-sha")

    with pytest.raises(ValueError, match="profile requires two SHA-256"):
        runner.run(manifest, snapshot)

    assert transport.requests == []


def test_dev_papers_require_exact_private_snapshot_binding() -> None:
    manifest, snapshot = _inputs()
    papers = dev_stage2_papers(manifest, snapshot)

    assert len(papers) == 300
    assert papers[0].paper_id.startswith("pair-")
    changed = GoldManifest(1, "f" * 64, tuple(
        replace(pair, corpus_hash="f" * 64) for pair in manifest.pairs
    ), manifest.main_languages)
    with pytest.raises(ValueError, match="corpus hashes"):
        dev_stage2_papers(changed, snapshot)


def test_present_malformed_keywords_are_not_silently_discarded() -> None:
    manifest, snapshot = _inputs()
    changed_papers = (replace(snapshot.papers[0], metadata={"keywords": "molecule"}), *snapshot.papers[1:])
    changed_snapshot = PrivateCorpusSnapshot(
        snapshot.schema_version, snapshot.sampling_policy_version, snapshot.sampling_seed, changed_papers,
    )
    changed_manifest = GoldManifest(
        manifest.version,
        changed_snapshot.corpus_hash,
        tuple(replace(pair, corpus_hash=changed_snapshot.corpus_hash) for pair in manifest.pairs),
        manifest.main_languages,
    )

    with pytest.raises(ValueError, match="array of strings"):
        dev_stage2_papers(changed_manifest, changed_snapshot)


def test_artifact_parser_rejects_duplicate_or_nonfinite_rows() -> None:
    manifest, snapshot = _inputs()
    artifact = _runner(FakeTransport()).run(manifest, snapshot)
    document = artifact.document()
    document["scores"]["reranker"][1]["pair_id"] = document["scores"]["reranker"][0]["pair_id"]
    with pytest.raises(ValueError):
        FrozenDevRawScoreArtifact.from_document(document)


def test_artifact_parser_rejects_duplicate_topic_query_rows() -> None:
    manifest, snapshot = _inputs()
    document = _runner(FakeTransport()).run(manifest, snapshot).document()
    document["topic_queries"].append(dict(document["topic_queries"][0]))

    with pytest.raises(ValueError, match="must be unique"):
        FrozenDevRawScoreArtifact.from_document(document)
