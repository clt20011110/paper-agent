from __future__ import annotations

import json
from pathlib import Path

import pytest

from paper_agent.stage2_backends import (
    AdjudicationInput,
    CascadeInput,
    CascadeRoute,
    DeterministicRuleDecision,
    ModelLock,
    OmlxChatBackend,
    OmlxRerankBackend,
    OmlxResponse,
    RerankBatchError,
    RerankInput,
    Stage2BackendError,
    StructuredOutputError,
    ThresholdArtifact,
    UrlLibOmlxTransport,
    load_model_lock,
    load_threshold_artifact,
    route_cascade,
    write_model_lock,
    write_threshold_artifact,
)


class FakeTransport:
    def __init__(self, responses: list[OmlxResponse]) -> None:
        self.responses = responses
        self.requests: list[tuple[str, dict[str, object]]] = []

    def request(self, path: str, payload: dict[str, object]) -> OmlxResponse:
        self.requests.append((path, payload))
        return self.responses.pop(0)


def _response(value: dict[str, object], headers: dict[str, str] | None = None, status: int = 200) -> OmlxResponse:
    return OmlxResponse(status, json.dumps(value).encode(), headers or {})


def _schema() -> dict[str, object]:
    return {"type": "object", "required": ["paper_id"]}


def _decision(paper_id: str = "paper-1") -> dict[str, object]:
    return {
        "paper_id": paper_id,
        "decision": "relevant",
        "score": 0.9,
        "reason_codes": ["topic_match"],
        "rationale": "The abstract directly studies the requested topic.",
        "evidence_fields": ["title", "abstract"],
    }


def test_production_schema_keeps_rationale_last_for_omlx_grammar() -> None:
    schema = json.loads(
        (Path(__file__).parents[1] / "schemas" / "filter-decision.schema.json")
        .read_text(encoding="utf-8")
    )

    assert list(schema["properties"])[-2:] == ["evidence_fields", "rationale"]


def test_omlx_rerank_batches_documents_and_never_sends_unsupported_limits() -> None:
    transport = FakeTransport([
        _response({"model": "BAAI/bge-reranker-v2-m3", "results": [{"index": 1, "relevance_score": -2.0}, {"index": 0, "relevance_score": 4.0}]})
    ])
    backend = OmlxRerankBackend("BAAI/bge-reranker-v2-m3", transport, document_batch_size=16)

    scores = backend.rerank("versioned topic query", [RerankInput("p1", "one"), RerankInput("p2", "two")])

    assert scores[0].paper_id == "p1"
    assert scores[0].raw_score == 4.0
    assert scores[1].raw_score == -2.0
    path, payload = transport.requests[0]
    assert path == "/v1/rerank"
    assert payload == {
        "model": "BAAI/bge-reranker-v2-m3",
        "query": "versioned topic query",
        "documents": ["one", "two"],
        "return_documents": False,
    }
    assert "max_length" not in payload
    assert "max_chunks_per_doc" not in payload


def test_omlx_rerank_rejects_incomplete_result_indexes() -> None:
    transport = FakeTransport([
        _response({"model": "model", "results": [{"index": 0, "relevance_score": 0.1}]}),
    ])
    backend = OmlxRerankBackend("model", transport)

    with pytest.raises(Stage2BackendError, match="invalid result count"):
        backend.rerank("q", [RerankInput("p1", "one"), RerankInput("p2", "two")])
    assert len(transport.requests) == 1


def test_omlx_rerank_downgrades_64_to_32_to_16_only_for_memory_exhaustion() -> None:
    class ResourceLimitedTransport:
        def __init__(self) -> None:
            self.batch_sizes: list[int] = []

        def request(self, path: str, payload: dict[str, object]) -> OmlxResponse:
            documents = payload["documents"]
            assert isinstance(documents, list)
            self.batch_sizes.append(len(documents))
            if len(documents) > 16:
                return _response({"error": "MLX out of memory"}, status=503)
            return _response({
                "model": "model",
                "results": [
                    {"index": index, "relevance_score": float(index)}
                    for index in range(len(documents))
                ]
            })

    transport = ResourceLimitedTransport()
    backend = OmlxRerankBackend("model", transport, document_batch_size=64, max_in_flight=1)
    documents = [
        RerankInput(f"paper-{index:02d}", f"document-{index}")
        for index in range(64)
    ]

    scores = backend.rerank("q", documents)

    assert transport.batch_sizes[:3] == [64, 32, 16]
    assert set(transport.batch_sizes) == {64, 32, 16}
    assert {score.paper_id for score in scores} == {
        f"paper-{index:02d}" for index in range(64)
    }


def test_omlx_rerank_does_not_split_generic_http_failure() -> None:
    transport = FakeTransport([_response({"error": "service unavailable"}, status=503)])
    backend = OmlxRerankBackend("model", transport, document_batch_size=64, max_in_flight=1)
    documents = [RerankInput(f"paper-{index:02d}", "document") for index in range(64)]

    with pytest.raises(Stage2BackendError, match="HTTP 503"):
        backend.rerank("q", documents)

    assert len(transport.requests) == 1


def test_omlx_rerank_marks_minimum_memory_limited_batch_as_paper_failures() -> None:
    transport = FakeTransport([_response({"detail": "OOM"}, status=500)])
    backend = OmlxRerankBackend("model", transport, document_batch_size=16)
    documents = [RerankInput(f"paper-{index:02d}", "document") for index in range(16)]

    with pytest.raises(RerankBatchError) as raised:
        backend.rerank("q", documents)

    assert raised.value.failed_paper_ids == tuple(item.paper_id for item in documents)
    assert len(transport.requests) == 1


@pytest.mark.parametrize("model", [None, "other-model"])
def test_omlx_rerank_rejects_missing_or_wrong_response_model(model: object) -> None:
    response: dict[str, object] = {
        "results": [{"index": 0, "relevance_score": 0.1}],
    }
    if model is not None:
        response["model"] = model
    backend = OmlxRerankBackend("model", FakeTransport([_response(response)]))

    with pytest.raises(Stage2BackendError, match="response model"):
        backend.rerank("q", [RerankInput("p1", "one")])
    assert len(backend.transport.requests) == 1


@pytest.mark.parametrize("score", [True, float("nan"), float("inf"), float("-inf")])
def test_omlx_rerank_rejects_non_finite_or_boolean_relevance_scores(score: object) -> None:
    backend = OmlxRerankBackend("model", FakeTransport([_response({
        "model": "model",
        "results": [{"index": 0, "relevance_score": score}],
    })]))

    with pytest.raises(Stage2BackendError, match="invalid result"):
        backend.rerank("q", [RerankInput("p1", "one")])
    assert len(backend.transport.requests) == 1


def test_omlx_chat_uses_fixed_generation_and_structured_output_contract() -> None:
    transport = FakeTransport([_response({
        "model": "mlx-community/Qwen3.5-9B-8bit",
        "choices": [{"message": {"content": json.dumps(_decision())}}],
    })])
    backend = OmlxChatBackend(
        "mlx-community/Qwen3.5-9B-8bit",
        transport,
        _schema(),
        seed=7,
        max_output_tokens=256,
    )

    decision = backend.adjudicate(AdjudicationInput("paper-1", ({"role": "user", "content": "classify"},)))

    assert decision.decision == "relevant"
    assert decision.score == 0.9
    path, payload = transport.requests[0]
    assert path == "/v1/chat/completions"
    assert payload["temperature"] == 0
    assert payload["seed"] == 7
    assert payload["stream"] is False
    assert payload["max_tokens"] == 256
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["structured_outputs"] == {"json": _schema()}
    assert "thinking" not in payload


def test_urllib_transport_adds_optional_bearer_auth_without_changing_payload(monkeypatch) -> None:
    captured = {}

    class Response:
        status = 200
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return b'{}'

    def fake_urlopen(request, timeout):
        captured["authorization"] = request.get_header("Authorization")
        captured["body"] = json.loads(request.data)
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr("paper_agent.stage2_backends.urlopen", fake_urlopen)
    response = UrlLibOmlxTransport(api_key="local-secret", timeout_seconds=7).request("/v1/rerank", {"query": "topic"})

    assert response.status_code == 200
    assert captured == {"authorization": "Bearer local-secret", "body": {"query": "topic"}, "timeout": 7}


def test_omlx_chat_revalidates_the_exact_wire_schema() -> None:
    schema = {**_schema(), "properties": {"paper_id": {"const": "different-paper"}}}
    backend = OmlxChatBackend("model", FakeTransport([_response({
        "model": "model",
        "choices": [{"message": {"content": json.dumps(_decision())}}],
    })]), schema)

    with pytest.raises(StructuredOutputError, match="violates schema"):
        backend.adjudicate(AdjudicationInput("paper-1", ({"role": "user", "content": "classify"},)))


@pytest.mark.parametrize(
    ("response", "error"),
    [
        (_response({"model": "model", "choices": [{"message": {"content": json.dumps(_decision())}}]}, {"Warning": "grammar skipped"}), "Warning"),
        (_response({"model": "model", "choices": [{"message": {"content": json.dumps(_decision("wrong"))}}]}), "paper_id"),
        (_response({"model": "model", "choices": [{"message": {"content": "not json"}}]}), "JSON decision"),
        (_response({"model": "other", "choices": [{"message": {"content": json.dumps(_decision())}}]}), "model does not match"),
        (OmlxResponse(200, b"not-json"), "JSON decision"),
        (_response({"error": "grammar failed"}, status=400), "HTTP 400"),
    ],
)
def test_omlx_chat_fails_closed_for_untrusted_structured_output(response: OmlxResponse, error: str) -> None:
    backend = OmlxChatBackend("model", FakeTransport([response]), _schema())

    with pytest.raises(StructuredOutputError, match=error):
        backend.adjudicate(AdjudicationInput("paper-1", ({"role": "user", "content": "classify"},)))


def test_model_lock_preserves_source_and_conversion_revisions_separately(tmp_path) -> None:
    lock = ModelLock(
        lock_version=1,
        backend="omlx_rerank",
        model_id="mlx-community/Qwen3-Reranker-4B-mxfp8",
        source_repo="Qwen/Qwen3-Reranker-4B",
        source_revision="22e683669bc0f0bd69640a1354a6d0aebcfeede5",
        conversion_repo="mlx-community/Qwen3-Reranker-4B-mxfp8",
        conversion_revision="25f203a0",
        format="mlx",
        quantization="mxfp8",
        license="apache-2.0",
        parameter_count=4_000_000_000,
        omlx_version="0.5.7",
        mlx_version="0.30.0",
        file_hashes={"model.safetensors": "a" * 64},
    )

    path = tmp_path / "model.lock.json"
    write_model_lock(path, lock)

    assert load_model_lock(path) == lock
    with pytest.raises(ValueError, match="1..10B"):
        ModelLock(**{**lock.document(), "parameter_count": 10_000_000_001})
    with pytest.raises(ValueError, match="specified together"):
        ModelLock(**{**lock.document(), "conversion_revision": None})


def test_shipped_model_locks_pin_verified_local_artifacts() -> None:
    root = Path(__file__).parents[1] / "configs" / "stage2" / "models"
    bge = load_model_lock(root / "bge-reranker-v2-m3-fp32.lock.json")
    qwen = load_model_lock(root / "qwen3.5-9b-8bit.lock.json")

    assert bge.source_revision == "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
    assert bge.conversion_revision is None
    assert bge.parameter_count == 567_755_777
    assert qwen.source_revision != qwen.conversion_revision
    assert qwen.conversion_revision == "16daa4818c54ce5f5436f929d52542eb65bbed9d"
    assert qwen.parameter_count == 9_409_813_744


def test_thresholds_keep_raw_scores_and_route_uncertain_inputs_to_qwen(tmp_path) -> None:
    thresholds = ThresholdArtifact("v1", "bge.lock.json", "raw_reranker_score", -1.2, 2.4)
    path = tmp_path / "thresholds.json"
    write_threshold_artifact(path, thresholds)

    assert load_threshold_artifact(path) == thresholds
    assert route_cascade(CascadeInput(raw_score=-2), thresholds) is CascadeRoute.IRRELEVANT
    assert route_cascade(CascadeInput(raw_score=3), thresholds) is CascadeRoute.RELEVANT
    assert route_cascade(CascadeInput(raw_score=0), thresholds) is CascadeRoute.ADJUDICATE
    assert route_cascade(CascadeInput(raw_score=-3, abstract_missing=True), thresholds) is CascadeRoute.ADJUDICATE
    assert route_cascade(CascadeInput(raw_score=3, deterministic=DeterministicRuleDecision(CascadeRoute.NEEDS_REVIEW, "metadata_conflict")), thresholds) is CascadeRoute.NEEDS_REVIEW
