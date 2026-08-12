"""Local Stage 2 model backends and deterministic cascade primitives.

The module keeps transport mechanics separate from routing: raw reranker scores
are never treated as probabilities, and malformed adjudicator output is an
explicit error for the caller to route to ``needs_review``.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from enum import StrEnum
import json
import math
import re
from jsonschema import ValidationError as JsonSchemaValidationError
from jsonschema import validate as validate_json_schema
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .schema import SchemaValidationError, validate
from .stage2_prompt_contract import estimate_omlx_chat_input_token_proxy


class Stage2BackendError(RuntimeError):
    """A local model service did not satisfy the Stage 2 contract."""


class StructuredOutputError(Stage2BackendError):
    """The adjudicator response cannot safely be auto-classified."""


class RerankerResourceError(Stage2BackendError):
    """The reranker reported a memory-exhaustion condition."""


_MEMORY_EXHAUSTION = re.compile(
    r"\b(?:oom|out[ -]of[ -]memory|bad_alloc)\b|"
    r"insufficient memory|memory pressure|memory allocation failed|failed to allocate memory",
    re.IGNORECASE,
)


class CascadeRoute(StrEnum):
    RELEVANT = "relevant"
    IRRELEVANT = "irrelevant"
    ADJUDICATE = "adjudicate"
    NEEDS_REVIEW = "needs_review"


@dataclass(frozen=True, slots=True)
class DeterministicRuleDecision:
    """An auditable explicit include/exclude result before semantic scoring."""

    route: CascadeRoute
    reason_code: str

    def __post_init__(self) -> None:
        if self.route not in {CascadeRoute.RELEVANT, CascadeRoute.IRRELEVANT, CascadeRoute.NEEDS_REVIEW}:
            raise ValueError("deterministic rules cannot route directly to adjudicate")
        if not self.reason_code:
            raise ValueError("deterministic rule decision requires a reason_code")


@dataclass(frozen=True, slots=True)
class OmlxResponse:
    status_code: int
    body: bytes
    headers: Mapping[str, str] = field(default_factory=dict)

    def json(self) -> Mapping[str, Any]:
        value = json.loads(self.body)
        if not isinstance(value, dict):
            raise Stage2BackendError("oMLX response must be a JSON object")
        return value


def omlx_response_is_memory_exhaustion(response: OmlxResponse) -> bool:
    """Return whether a failed oMLX response explicitly reports memory exhaustion."""

    if response.status_code == 200:
        return False
    return _MEMORY_EXHAUSTION.search(response.body.decode("utf-8", errors="replace")) is not None


class OmlxTransport(Protocol):
    def request(self, path: str, payload: Mapping[str, Any]) -> OmlxResponse: ...


@dataclass(slots=True)
class UrlLibOmlxTransport:
    """Small OpenAI-compatible HTTP transport; tests normally inject a fake."""

    base_url: str = "http://127.0.0.1:8000"
    timeout_seconds: float = 120.0
    api_key: str | None = None

    def request(self, path: str, payload: Mapping[str, Any]) -> OmlxResponse:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(
            f"{self.base_url.rstrip('/')}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return OmlxResponse(response.status, response.read(), dict(response.headers.items()))
        except HTTPError as error:
            return OmlxResponse(error.code, error.read(), dict(error.headers.items()))


@dataclass(frozen=True, slots=True)
class ModelLock:
    """Unambiguous provenance for one evaluated local model artifact."""

    lock_version: int
    backend: str
    model_id: str
    source_repo: str
    source_revision: str
    conversion_repo: str | None
    conversion_revision: str | None
    format: str
    quantization: str
    license: str
    parameter_count: int
    omlx_version: str
    mlx_version: str
    file_hashes: Mapping[str, str]

    def __post_init__(self) -> None:
        if self.lock_version != 1:
            raise ValueError("unsupported model lock version")
        if self.backend not in {"omlx_rerank", "omlx_chat", "mlx_native"}:
            raise ValueError("unsupported Stage 2 backend")
        if not all((self.model_id, self.source_repo, self.source_revision, self.format, self.quantization, self.license, self.omlx_version, self.mlx_version)):
            raise ValueError("model lock provenance fields are required")
        if bool(self.conversion_repo) != bool(self.conversion_revision):
            raise ValueError("conversion_repo and conversion_revision must be specified together")
        if not 0 < self.parameter_count <= 10_000_000_000:
            raise ValueError("Stage 2 model parameter_count must be in 1..10B")
        if not self.file_hashes or any(not name or not digest for name, digest in self.file_hashes.items()):
            raise ValueError("model lock requires named model file hashes")

    def document(self) -> dict[str, Any]:
        return asdict(self)


def write_model_lock(path: Path, lock: ModelLock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(lock.document(), sort_keys=True, indent=2) + "\n", encoding="utf-8")


def load_model_lock(path: Path) -> ModelLock:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("model lock must be an object")
    return ModelLock(**value)


@dataclass(frozen=True, slots=True)
class RerankInput:
    paper_id: str
    document: str


@dataclass(frozen=True, slots=True)
class RerankScore:
    paper_id: str
    raw_score: float


class RerankBatchError(Stage2BackendError):
    """A rerank request that preserved successful scores and isolated failures."""

    def __init__(
        self,
        scores: Sequence[RerankScore],
        failed_paper_ids: Sequence[str],
    ) -> None:
        self.scores = tuple(scores)
        self.failed_paper_ids = tuple(failed_paper_ids)
        super().__init__("oMLX rerank could not score every paper")


class RerankerBackend(Protocol):
    backend_name: str

    def rerank(self, query: str, documents: Sequence[RerankInput]) -> tuple[RerankScore, ...]: ...


@dataclass(slots=True)
class OmlxRerankBackend:
    """oMLX native ``/v1/rerank`` client.

    Each HTTP request carries one query and many documents.  The endpoint's
    XLM-R implementation owns its 512-token behaviour, so this client never
    invents ``max_length`` or chunking parameters.
    """

    model: str
    transport: OmlxTransport
    document_batch_size: int = 32
    max_in_flight: int = 2
    backend_name: str = field(default="omlx_rerank", init=False)
    is_local: bool = field(default=True, init=False)

    def __post_init__(self) -> None:
        if self.document_batch_size not in {16, 32, 64}:
            raise ValueError("document_batch_size must be one of 16, 32, 64")
        if not 1 <= self.max_in_flight <= 2:
            raise ValueError("max_in_flight must be in 1..2")

    def rerank(self, query: str, documents: Sequence[RerankInput]) -> tuple[RerankScore, ...]:
        if not query.strip():
            raise ValueError("reranker query is required")
        batches = [documents[index : index + self.document_batch_size] for index in range(0, len(documents), self.document_batch_size)]
        with ThreadPoolExecutor(max_workers=self.max_in_flight) as executor:
            outcomes = list(executor.map(lambda batch: self._rerank_with_downgrade(query, batch), batches))
        scores = tuple(score for batch_scores, _ in outcomes for score in batch_scores)
        failures = tuple(paper_id for _, paper_ids in outcomes for paper_id in paper_ids)
        if failures:
            raise RerankBatchError(scores, failures)
        return scores

    def _rerank_with_downgrade(
        self,
        query: str,
        documents: Sequence[RerankInput],
    ) -> tuple[tuple[RerankScore, ...], tuple[str, ...]]:
        try:
            return self._rerank_batch(query, documents), ()
        except RerankerResourceError:
            next_size = 32 if len(documents) > 32 else 16 if len(documents) > 16 else None
            if next_size is not None:
                outcomes = [
                    self._rerank_with_downgrade(query, documents[index : index + next_size])
                    for index in range(0, len(documents), next_size)
                ]
                return (
                    tuple(score for scores, _ in outcomes for score in scores),
                    tuple(paper_id for _, paper_ids in outcomes for paper_id in paper_ids),
                )
            return (), tuple(document.paper_id for document in documents)

    def _rerank_batch(self, query: str, documents: Sequence[RerankInput]) -> tuple[RerankScore, ...]:
        payload = {
            "model": self.model,
            "query": query,
            "documents": [item.document for item in documents],
            "return_documents": False,
        }
        try:
            response = self.transport.request("/v1/rerank", payload)
        except MemoryError as error:
            raise RerankerResourceError("oMLX rerank exhausted local memory") from error
        if response.status_code != 200:
            if omlx_response_is_memory_exhaustion(response):
                raise RerankerResourceError("oMLX rerank exhausted local memory")
            raise Stage2BackendError(f"oMLX rerank returned HTTP {response.status_code}")
        response_document = response.json()
        if response_document.get("model") != self.model:
            raise Stage2BackendError(
                "oMLX rerank response model does not match the frozen reranker"
            )
        results = response_document.get("results")
        if not isinstance(results, list) or len(results) != len(documents):
            raise Stage2BackendError("oMLX rerank response has an invalid result count")
        by_index: dict[int, float] = {}
        for result in results:
            relevance_score = result.get("relevance_score") if isinstance(result, dict) else None
            if (
                not isinstance(result, dict)
                or not isinstance(result.get("index"), int)
                or isinstance(relevance_score, bool)
                or not isinstance(relevance_score, (int, float))
                or not math.isfinite(relevance_score)
            ):
                raise Stage2BackendError("oMLX rerank response has an invalid result")
            by_index[result["index"]] = float(relevance_score)
        if set(by_index) != set(range(len(documents))):
            raise Stage2BackendError("oMLX rerank response indexes do not match documents")
        return tuple(RerankScore(item.paper_id, by_index[index]) for index, item in enumerate(documents))


@dataclass(frozen=True, slots=True)
class AdjudicationInput:
    paper_id: str
    messages: tuple[Mapping[str, str], ...]


@dataclass(frozen=True, slots=True)
class AdjudicationDecision:
    paper_id: str
    decision: str
    score: float
    reason_codes: tuple[str, ...]
    rationale: str
    evidence_fields: tuple[str, ...]


class AdjudicatorBackend(Protocol):
    backend_name: str

    def adjudicate(self, request: AdjudicationInput) -> AdjudicationDecision: ...


@dataclass(slots=True)
class OmlxChatBackend:
    """Fail-closed Qwen adjudicator client for oMLX chat completions."""

    model: str
    transport: OmlxTransport
    schema: Mapping[str, Any]
    seed: int = 42
    max_context_window: int = 16_384
    max_output_tokens: int = 256
    backend_name: str = field(default="omlx_chat", init=False)

    def __post_init__(self) -> None:
        if not 1 <= self.max_context_window <= 32_768:
            raise ValueError("max_context_window must be in 1..32768")
        if not 1 <= self.max_output_tokens <= 1_024:
            raise ValueError("max_output_tokens must be in 1..1024")

    def adjudicate(self, request: AdjudicationInput) -> AdjudicationDecision:
        if self._estimated_token_proxy(request.messages) > self.max_context_window:
            raise StructuredOutputError("adjudicator input exceeds configured max_context_window")
        payload = {
            "model": self.model,
            "messages": list(request.messages),
            "temperature": 0,
            "seed": self.seed,
            "stream": False,
            "max_tokens": self.max_output_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
            # This is the wire representation of OpenAI SDK's
            # extra_body={"structured_outputs": {"json": schema}}.
            "structured_outputs": {"json": dict(self.schema)},
        }
        response = self.transport.request("/v1/chat/completions", payload)
        if any(name.lower() == "warning" for name in response.headers):
            raise StructuredOutputError("oMLX returned Warning: structured output is not guaranteed")
        if response.status_code != 200:
            raise StructuredOutputError(f"oMLX chat returned HTTP {response.status_code}")
        try:
            response_document = response.json()
            content = response_document["choices"][0]["message"]["content"]
            value = json.loads(content)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
            raise StructuredOutputError("oMLX chat response is not a JSON decision") from error
        if response_document.get("model") != self.model:
            raise StructuredOutputError(
                "oMLX chat response model does not match the frozen adjudicator"
            )
        if not isinstance(value, dict):
            raise StructuredOutputError("oMLX chat decision must be an object")
        try:
            validate_json_schema(value, self.schema)
            validate(value, "filter-decision.schema.json")
        except (JsonSchemaValidationError, SchemaValidationError) as error:
            raise StructuredOutputError(f"oMLX chat decision violates schema: {error}") from error
        if value["paper_id"] != request.paper_id:
            raise StructuredOutputError("oMLX chat decision paper_id does not match request")
        return AdjudicationDecision(
            paper_id=value["paper_id"],
            decision=value["decision"],
            score=float(value["score"]),
            reason_codes=tuple(value["reason_codes"]),
            rationale=value["rationale"],
            evidence_fields=tuple(value["evidence_fields"]),
        )

    @staticmethod
    def _estimated_token_proxy(messages: Sequence[Mapping[str, str]]) -> int:
        return estimate_omlx_chat_input_token_proxy(messages)


@dataclass(slots=True)
class MlxNativeExperimentalBackend:
    """Explicitly opt-in local experimental backend; never a silent fallback."""

    execute: Callable[[str, Sequence[RerankInput]], Sequence[float]]
    enabled: bool = False
    backend_name: str = field(default="mlx_native", init=False)
    is_local: bool = field(default=True, init=False)

    def rerank(self, query: str, documents: Sequence[RerankInput]) -> tuple[RerankScore, ...]:
        if not self.enabled:
            raise Stage2BackendError("mlx_native is experimental and disabled")
        scores = tuple(self.execute(query, documents))
        if len(scores) != len(documents):
            raise Stage2BackendError("mlx_native returned an invalid score count")
        return tuple(RerankScore(item.paper_id, float(score)) for item, score in zip(documents, scores, strict=True))


@dataclass(frozen=True, slots=True)
class ThresholdArtifact:
    """Versioned dev-set calibration for raw reranker scores, not probabilities."""

    version: str
    model_lock: str
    score_kind: str
    t_low: float
    t_high: float
    calibration_target: str = "P(gold_label >= 2)"

    def __post_init__(self) -> None:
        if not self.version or not self.model_lock:
            raise ValueError("threshold artifact version and model_lock are required")
        if self.score_kind != "raw_reranker_score":
            raise ValueError("threshold artifact must declare raw_reranker_score")
        if self.t_low >= self.t_high:
            raise ValueError("threshold t_low must be less than t_high")

    def document(self) -> dict[str, Any]:
        return asdict(self)


def write_threshold_artifact(path: Path, artifact: ThresholdArtifact) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact.document(), sort_keys=True, indent=2) + "\n", encoding="utf-8")


def load_threshold_artifact(path: Path) -> ThresholdArtifact:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("threshold artifact must be an object")
    return ThresholdArtifact(**value)


@dataclass(frozen=True, slots=True)
class CascadeInput:
    raw_score: float | None
    deterministic: DeterministicRuleDecision | None = None
    abstract_missing: bool = False
    possibly_truncated: bool = False
    multi_condition_conflict: bool = False
    language_anomaly: bool = False


def route_cascade(input: CascadeInput, thresholds: ThresholdArtifact) -> CascadeRoute:
    """Apply deterministic rules, then protect uncertain samples with Qwen."""

    if input.deterministic is not None:
        return input.deterministic.route
    if input.raw_score is None or input.abstract_missing or input.possibly_truncated or input.multi_condition_conflict or input.language_anomaly:
        return CascadeRoute.ADJUDICATE
    if input.raw_score < thresholds.t_low:
        return CascadeRoute.IRRELEVANT
    if input.raw_score >= thresholds.t_high:
        return CascadeRoute.RELEVANT
    return CascadeRoute.ADJUDICATE
