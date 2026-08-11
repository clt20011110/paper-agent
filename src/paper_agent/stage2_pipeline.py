"""Production orchestration for the local Stage 2 screening cascade.

The backend module owns model transports.  This module owns the stable input
contract, cascade routing, and SQLite evidence trail used to resume a run.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import hashlib
import json
from time import monotonic, sleep
from typing import Callable, Iterable, Sequence
from uuid import uuid4

from .domain import FilterStatus
from .leases import LeaseNotCurrent, LeaseQueue, TaskLease, TaskLeaseSpec
from .stage2_backends import (
    AdjudicationDecision,
    AdjudicationInput,
    AdjudicatorBackend,
    CascadeInput,
    CascadeRoute,
    DeterministicRuleDecision,
    RerankBatchError,
    RerankInput,
    RerankerBackend,
    Stage2BackendError,
    StructuredOutputError,
    ThresholdArtifact as LegacyThresholdArtifact,
    route_cascade,
)
from .stage2_evaluation import (
    CalibrationPath,
    PathCalibrator,
    ThresholdArtifact as ProbabilityThresholdArtifact,
)
from .schema import schema_directory
from .storage import Database


IMPLEMENTATION_VERSION = "stage2-cascade-v2"
STAGE2_LEASE_OUTPUT_PREFIX = "filter-decision:"
DEFAULT_LEASE_SECONDS = 3_600
DEFAULT_PEER_WAIT_SECONDS = 3_900
ADJUDICATION_SYSTEM_PROMPT = "Return only the required structured screening decision."
ADJUDICATION_USER_TEMPLATE = (
    "Query version: {query_version}\nQuery: {query}\nPaper ID: {paper_id}\n{document}"
)
_FAILURES = (Stage2BackendError, StructuredOutputError, TimeoutError, OSError, ValueError)
_RETRYABLE_ADJUDICATOR_FAILURES = (Stage2BackendError, TimeoutError, OSError)
ADJUDICATOR_SHARE_ALARM = "stage2.adjudicator_share_exceeded"
ERROR_RATE_ALARM = "stage2.error_rate_exceeded"
MEMORY_WATERMARK_ALARM = "stage2.memory_watermark_exceeded"
TERMINAL_TECHNICAL_REASONS = frozenset({
    "reranker_backend_failure",
    "reranker_response_failure",
    "reranker_calibration_failure",
    "adjudicator_schema_failure",
    "adjudicator_backend_failure",
    "adjudicator_calibration_failure",
})


def adjudicator_capacity(adjudicator_share: float) -> str:
    if adjudicator_share > 0.30:
        return "severe"
    if adjudicator_share > 0.15:
        return "warning"
    return "normal"


def qwen_capacity_level(qwen_share: float) -> str:
    """Backward-compatible alias for the canonical adjudicator capacity."""

    return adjudicator_capacity(qwen_share)


def adjudicator_share_alarms(adjudicator_share: float) -> tuple[str, ...]:
    return (
        (ADJUDICATOR_SHARE_ALARM,)
        if adjudicator_share > 0.15
        else ()
    )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp(moment: datetime) -> str:
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError("Stage 2 lease clock must return a timezone-aware datetime")
    return moment.astimezone(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


@dataclass(frozen=True, slots=True)
class Stage2Paper:
    """Canonical metadata passed to Stage 2 for exactly one paper."""

    paper_id: str
    title: str
    abstract: str | None
    keywords: tuple[str, ...] = ()
    document_type: str | None = None
    possibly_truncated: bool = False
    multi_condition_conflict: bool = False
    language_anomaly: bool = False

    def __post_init__(self) -> None:
        if not self.paper_id or not self.title.strip():
            raise ValueError("Stage 2 paper_id and title are required")


@dataclass(frozen=True, slots=True)
class PathCalibration:
    """One released model path and its probability-space decision thresholds."""

    calibrator: PathCalibrator
    threshold: ProbabilityThresholdArtifact

    def __post_init__(self) -> None:
        if self.calibrator.path is not self.threshold.path:
            raise ValueError("Stage 2 calibrator and threshold paths do not match")
        if self.threshold.calibrator_hash != self.calibrator.hash():
            raise ValueError("Stage 2 threshold is not bound to its calibrator")
        if self.threshold.model_lock_hash != self.calibrator.model_lock_hash:
            raise ValueError("Stage 2 calibrator and threshold model locks do not match")
        if self.threshold.dev_manifest_hash != self.calibrator.dev_manifest_hash:
            raise ValueError("Stage 2 calibrator and threshold DEV manifests do not match")
        if self.threshold.dev_label_hash != self.calibrator.dev_label_hash:
            raise ValueError("Stage 2 calibrator and threshold DEV labels do not match")


@dataclass(frozen=True, slots=True)
class Stage2Profile:
    """Frozen, explicit Stage 2 runtime configuration."""

    query: str
    query_version: str
    thresholds: LegacyThresholdArtifact | None
    reranker_model_id: str
    reranker_revision: str
    adjudicator_model_id: str
    adjudicator_revision: str
    screening_scope_hash: str
    reranker_calibration: PathCalibration | None = None
    adjudicator_calibration: PathCalibration | None = None
    reranker_lock_hash: str | None = None
    adjudicator_lock_hash: str | None = None
    release_gate_hash: str | None = None
    include_document_types: frozenset[str] = frozenset()
    exclude_document_types: frozenset[str] = frozenset({"editorial", "retraction"})
    token_bucket_width: int = 128
    document_batch_size: int = 32
    reranker_max_in_flight: int = 2
    adjudicator_concurrency: int = 4
    adjudicator_seed: int = 42
    adjudicator_max_context_window: int = 16_384
    omlx_base_url: str = "http://127.0.0.1:8000"
    api_key_env: str | None = None
    prompt_version: str = "stage2-adjudication-v1"
    schema_version: str = "filter-decision.schema.json"

    def __post_init__(self) -> None:
        if not self.query.strip() or not self.query_version:
            raise ValueError("Stage 2 query and query_version are required")
        if not all((self.reranker_model_id, self.reranker_revision, self.adjudicator_model_id, self.adjudicator_revision)):
            raise ValueError("Stage 2 model provenance is required")
        if (
            not isinstance(self.screening_scope_hash, str)
            or len(self.screening_scope_hash) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.screening_scope_hash
            )
        ):
            raise ValueError("Stage 2 screening scope hash must be a lowercase SHA-256")
        if self.include_document_types & self.exclude_document_types:
            raise ValueError("document types cannot be both included and excluded")
        if self.token_bucket_width < 1 or self.adjudicator_concurrency < 1:
            raise ValueError("Stage 2 bucket width and adjudicator concurrency must be positive")
        if self.document_batch_size not in {16, 32, 64} or not 1 <= self.reranker_max_in_flight <= 2:
            raise ValueError("Stage 2 reranker runtime settings are invalid")
        if not 1 <= self.adjudicator_max_context_window <= 32_768:
            raise ValueError("Stage 2 adjudicator context window is invalid")
        calibrations = (self.reranker_calibration, self.adjudicator_calibration)
        if self.thresholds is not None and any(calibrations):
            raise ValueError("legacy raw thresholds cannot be mixed with probability calibrations")
        if (self.reranker_calibration is None) != (self.adjudicator_calibration is None):
            raise ValueError("Stage 2 production profiles require both calibration paths")
        if self.reranker_calibration is not None:
            self._validate_production_calibrations()

    @property
    def base_runtime_config_hash(self) -> str:
        return _hash({
            "kind": "stage2-base-runtime-v2",
            "query": self.query,
            "query_version": self.query_version,
            "screening_scope_hash": self.screening_scope_hash,
            "reranker": [self.reranker_model_id, self.reranker_revision],
            "adjudicator": [self.adjudicator_model_id, self.adjudicator_revision],
            "reranker_lock_hash": self.reranker_lock_hash,
            "adjudicator_lock_hash": self.adjudicator_lock_hash,
            "include_document_types": sorted(self.include_document_types),
            "exclude_document_types": sorted(self.exclude_document_types),
            "token_bucket_width": self.token_bucket_width,
            "document_batch_size": self.document_batch_size,
            "reranker_max_in_flight": self.reranker_max_in_flight,
            "adjudicator_concurrency": self.adjudicator_concurrency,
            "adjudicator_seed": self.adjudicator_seed,
            "adjudicator_max_context_window": self.adjudicator_max_context_window,
            "omlx_base_url": self.omlx_base_url,
            "api_key_env": self.api_key_env,
            "prompt_hash": self.prompt_hash,
            "schema_hash": self.schema_hash,
        })

    @property
    def threshold_bundle_hash(self) -> str | None:
        if self.reranker_calibration is not None and self.adjudicator_calibration is not None:
            return _hash({
                "kind": "stage2-probability-threshold-bundle-v1",
                "paths": {
                    CalibrationPath.RERANKER.value: {
                        "calibrator_hash": self.reranker_calibration.calibrator.hash(),
                        "threshold_hash": self.reranker_calibration.threshold.hash(),
                    },
                    CalibrationPath.QWEN.value: {
                        "calibrator_hash": self.adjudicator_calibration.calibrator.hash(),
                        "threshold_hash": self.adjudicator_calibration.threshold.hash(),
                    },
                },
            })
        if self.thresholds is not None:
            return _hash({"kind": "stage2-legacy-raw-threshold-v1", "artifact": self.thresholds.document()})
        return None

    @property
    def full_profile_hash(self) -> str:
        threshold_bundle_hash = self.threshold_bundle_hash
        if threshold_bundle_hash is None:
            raise ValueError("Stage 2 profile has no threshold bundle")
        return _hash({
            "kind": "stage2-full-profile-v1",
            "base_runtime_config_hash": self.base_runtime_config_hash,
            "threshold_bundle_hash": threshold_bundle_hash,
            "release_gate_hash": self.release_gate_hash,
        })

    @property
    def config_hash(self) -> str:
        return self.full_profile_hash

    @property
    def threshold_hash(self) -> str:
        threshold_bundle_hash = self.threshold_bundle_hash
        if threshold_bundle_hash is None:
            raise ValueError("Stage 2 profile has no threshold bundle")
        return threshold_bundle_hash

    @property
    def threshold_version(self) -> str:
        if self.reranker_calibration is not None:
            return f"probability-v1:{self.threshold_hash[:16]}"
        if self.thresholds is not None:
            return self.thresholds.version
        raise ValueError("Stage 2 profile has no thresholds")

    @property
    def production_calibrated(self) -> bool:
        return self.reranker_calibration is not None

    def assert_runtime_ready(self) -> None:
        if self.thresholds is None and self.reranker_calibration is None:
            raise ValueError("Stage 2 runtime requires released probability calibrations or explicit legacy test thresholds")

    def _validate_production_calibrations(self) -> None:
        reranker = self.reranker_calibration
        adjudicator = self.adjudicator_calibration
        assert reranker is not None and adjudicator is not None
        if reranker.calibrator.path is not CalibrationPath.RERANKER:
            raise ValueError("reranker calibration must use the reranker path")
        if adjudicator.calibrator.path is not CalibrationPath.QWEN:
            raise ValueError("adjudicator calibration must use the qwen path")
        if reranker.calibrator.model_lock_hash != self.reranker_lock_hash:
            raise ValueError("reranker calibration does not match the released model lock")
        if adjudicator.calibrator.model_lock_hash != self.adjudicator_lock_hash:
            raise ValueError("qwen calibration does not match the released model lock")
        if any(
            binding.threshold.stage2_config_hash != self.base_runtime_config_hash
            for binding in (reranker, adjudicator)
        ):
            raise ValueError("probability thresholds do not match the base Stage 2 runtime config")
        provenance = {
            (
                binding.calibrator.gold_manifest_hash,
                binding.calibrator.dev_manifest_hash,
                binding.calibrator.dev_label_hash,
            )
            for binding in (reranker, adjudicator)
        }
        if len(provenance) != 1:
            raise ValueError("Stage 2 calibration paths do not share frozen DEV provenance")

    @property
    def prompt_hash(self) -> str:
        return _hash({
            "version": self.prompt_version,
            "system": ADJUDICATION_SYSTEM_PROMPT,
            "user_template": ADJUDICATION_USER_TEMPLATE,
        })

    @property
    def schema_hash(self) -> str:
        return hashlib.sha256(
            (schema_directory() / self.schema_version).read_bytes()
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class Stage2Decision:
    paper_id: str
    status: FilterStatus
    reason_code: str
    input_hash: str
    route: CascadeRoute
    reranker_score: float | None = None
    reranker_probability: float | None = None
    adjudicator_score: float | None = None
    adjudicator_probability: float | None = None
    rationale: str | None = None
    adjudicated: bool = False
    adjudicator_attempt_count: int = 0
    adjudicator_retry_reason: str | None = None
    adjudicator_retry_outcome: str | None = None
    resumed: bool = False


@dataclass(frozen=True, slots=True)
class Stage2Summary:
    decisions: tuple[Stage2Decision, ...]
    reranked_count: int
    qwen_count: int
    qwen_share: float
    qwen_alarms: tuple[str, ...]
    error_count: int = 0
    error_rate: float = 0.0

    @property
    def capacity_level(self) -> str:
        return qwen_capacity_level(self.qwen_share)

    @property
    def alarm_codes(self) -> tuple[str, ...]:
        alarms = list(self.qwen_alarms)
        if self.error_rate >= 0.005:
            alarms.append(ERROR_RATE_ALARM)
        return tuple(alarms)

    def telemetry(self, run_id: str) -> dict[str, object]:
        reranked_count = sum(
            not decision.reason_code.startswith("document_type_")
            for decision in self.decisions
        )
        return {
            "stage2_run_ids": [run_id],
            "run_id": run_id,
            "screened_count": len(self.decisions),
            "reranked_count": reranked_count,
            "adjudicator_count": self.qwen_count,
            "adjudicator_share": self.qwen_share,
            "adjudicator_capacity": adjudicator_capacity(self.qwen_share),
            "paper_count": len(self.decisions),
            "qwen_count": self.qwen_count,
            "qwen_share": self.qwen_share,
            "error_count": self.error_count,
            "error_rate": self.error_rate,
            "capacity_level": self.capacity_level,
            "alarm_codes": list(self.alarm_codes),
        }


@dataclass(slots=True)
class Stage2Pipeline:
    database: Database
    reranker: RerankerBackend
    adjudicator: AdjudicatorBackend
    profile: Stage2Profile
    implementation_version: str = IMPLEMENTATION_VERSION
    worker_id: str = field(default_factory=lambda: f"stage2-worker-{uuid4().hex}")
    lease_seconds: int = DEFAULT_LEASE_SECONDS
    peer_wait_seconds: float = DEFAULT_PEER_WAIT_SECONDS
    lease_poll_seconds: float = 0.25
    lease_clock: Callable[[], datetime] = field(default=_utc_now, repr=False)
    wait_clock: Callable[[], float] = field(default=monotonic, repr=False)
    sleeper: Callable[[float], None] = field(default=sleep, repr=False)
    _completed: dict[tuple[str, str], Stage2Decision] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if not self.worker_id:
            raise ValueError("Stage 2 worker_id is required")
        if self.lease_seconds <= 0:
            raise ValueError("Stage 2 lease_seconds must be positive")
        if self.peer_wait_seconds <= 0 or self.lease_poll_seconds <= 0:
            raise ValueError("Stage 2 peer wait settings must be positive")

    def run(self, run_id: str, papers: Iterable[Stage2Paper]) -> Stage2Summary:
        self.profile.assert_runtime_ready()
        candidates = tuple(sorted(papers, key=lambda item: item.paper_id))
        if len({item.paper_id for item in candidates}) != len(candidates):
            raise ValueError("a Stage 2 call cannot contain duplicate paper_ids")
        self._ensure_run(run_id, candidates)
        self._completed = self._load_completed(run_id)
        by_id = {paper.paper_id: paper for paper in candidates}
        pending = tuple(
            paper
            for paper in candidates
            if (paper.paper_id, self.input_hash(paper)) not in self._completed
        )
        queue = LeaseQueue(self.database)
        if pending:
            queue.enqueue_many(
                run_id=run_id,
                stage="stage-2",
                specs=tuple(
                    TaskLeaseSpec(
                        paper.paper_id,
                        self._lease_output_kind(paper),
                        self.input_hash(paper),
                    )
                    for paper in pending
                ),
                now=_timestamp(self.lease_clock()),
            )

        fresh: dict[str, Stage2Decision] = {}
        reranked_count = 0
        deadline = self.wait_clock() + self.peer_wait_seconds
        claim_limit = self.profile.document_batch_size * self.profile.reranker_max_in_flight
        poll_seconds = self.lease_poll_seconds
        while pending:
            now = self.lease_clock()
            now_text = _timestamp(now)
            claims = queue.claim(
                worker_id=self.worker_id,
                now=now_text,
                expires_at=_timestamp(now + timedelta(seconds=self.lease_seconds)),
                limit=claim_limit,
                run_id=run_id,
                stage="stage-2",
                output_kind_prefix=STAGE2_LEASE_OUTPUT_PREFIX,
            )
            if claims:
                claimed_papers = tuple(by_id[claim.paper_id] for claim in claims if claim.paper_id)
                if len(claimed_papers) != len(claims):
                    raise RuntimeError("Stage 2 claimed a task without a paper_id")
                batch_decisions = self._screen_batch(claimed_papers)
                reranked_count += sum(
                    not decision.reason_code.startswith("document_type_")
                    for decision in batch_decisions
                )
                try:
                    self._persist_claimed(
                        run_id,
                        tuple(zip(claims, batch_decisions, strict=True)),
                    )
                except LeaseNotCurrent:
                    # A slow model response may arrive after another worker has
                    # reclaimed the task.  Its result is intentionally dropped.
                    continue
                fresh.update((decision.paper_id, decision) for decision in batch_decisions)
                self._completed.update(
                    ((decision.paper_id, decision.input_hash), decision)
                    for decision in batch_decisions
                )
                poll_seconds = self.lease_poll_seconds
            else:
                self._completed = self._load_completed(run_id)

            pending = tuple(
                paper
                for paper in candidates
                if (paper.paper_id, self.input_hash(paper)) not in self._completed
            )
            if not pending:
                break
            if claims:
                continue
            if self.wait_clock() >= deadline:
                raise TimeoutError(
                    f"Stage 2 timed out waiting for peer workers on run {run_id}"
                )
            self.sleeper(poll_seconds)
            poll_seconds = min(poll_seconds * 2, 1.0)

        persisted = self._load_completed(run_id)
        ordered = tuple(
            fresh.get(paper.paper_id)
            or persisted[(paper.paper_id, self.input_hash(paper))]
            for paper in candidates
        )
        self._completed = persisted
        with self.database.transaction() as connection:
            connection.execute(
                """UPDATE pipeline_runs
                   SET status = 'complete', completed_at = CURRENT_TIMESTAMP
                   WHERE run_id = ? AND NOT EXISTS (
                       SELECT 1 FROM task_leases
                       WHERE run_id = ? AND stage = 'stage-2'
                         AND substr(output_kind, 1, ?) = ? AND status <> 'complete'
                   )""",
                (
                    run_id,
                    run_id,
                    len(STAGE2_LEASE_OUTPUT_PREFIX),
                    STAGE2_LEASE_OUTPUT_PREFIX,
                ),
            )
        qwen_count = sum(item.adjudicated for item in ordered)
        qwen_share = qwen_count / len(candidates) if candidates else 0.0
        qwen_alarms = adjudicator_share_alarms(qwen_share)
        error_count = sum(
            item.reason_code in TERMINAL_TECHNICAL_REASONS for item in ordered
        )
        error_rate = error_count / len(candidates) if candidates else 0.0
        return Stage2Summary(
            ordered,
            reranked_count,
            qwen_count,
            qwen_share,
            qwen_alarms,
            error_count,
            error_rate,
        )

    def _screen_batch(
        self, candidates: Sequence[Stage2Paper]
    ) -> tuple[Stage2Decision, ...]:
        """Run one claimed paper batch without touching SQLite."""
        decisions: dict[str, Stage2Decision] = {}
        rerank_inputs: list[Stage2Paper] = []
        for paper in candidates:
            input_hash = self.input_hash(paper)
            deterministic = self._deterministic(paper)
            if deterministic is not None:
                decisions[paper.paper_id] = self._from_route(
                    paper,
                    input_hash,
                    deterministic.route,
                    deterministic.reason_code,
                )
            else:
                rerank_inputs.append(paper)

        scored, rerank_failures = self._rerank(rerank_inputs)
        for paper in rerank_failures:
            decisions[paper.paper_id] = self._from_route(
                paper,
                self.input_hash(paper),
                CascadeRoute.NEEDS_REVIEW,
                "reranker_backend_failure",
                reranker_probability=self._failure_probability(),
            )

        adjudication: list[tuple[Stage2Paper, float | None, float | None, str]] = []
        for paper in rerank_inputs:
            if paper.paper_id in decisions:
                continue
            score = scored.get(paper.paper_id)
            if score is None:
                decisions[paper.paper_id] = self._from_route(
                    paper,
                    self.input_hash(paper),
                    CascadeRoute.NEEDS_REVIEW,
                    "reranker_response_failure",
                    reranker_probability=self._failure_probability(),
                )
                continue
            try:
                route, probability = self._reranker_route(paper, score)
            except ValueError:
                decisions[paper.paper_id] = self._from_route(
                    paper,
                    self.input_hash(paper),
                    CascadeRoute.NEEDS_REVIEW,
                    "reranker_calibration_failure",
                    score,
                )
                continue
            if route is CascadeRoute.ADJUDICATE:
                reason = self._adjudication_reason(paper)
                adjudication.append((paper, score, probability, reason))
            else:
                decisions[paper.paper_id] = self._from_route(
                    paper,
                    self.input_hash(paper),
                    route,
                    "reranker_probability_threshold"
                    if probability is not None
                    else "reranker_threshold",
                    score,
                    probability,
                )

        for decision in self._adjudicate(adjudication):
            decisions[decision.paper_id] = decision
        return tuple(decisions[paper.paper_id] for paper in candidates)

    def input_hash(self, paper: Stage2Paper) -> str:
        return _hash({
            "paper_id": paper.paper_id,
            "title": paper.title,
            "abstract": paper.abstract,
            "keywords": paper.keywords,
            "document_type": paper.document_type,
            "possibly_truncated": paper.possibly_truncated,
            "multi_condition_conflict": paper.multi_condition_conflict,
            "language_anomaly": paper.language_anomaly,
            "query": self.profile.query,
            "query_version": self.profile.query_version,
        })

    def document(self, paper: Stage2Paper) -> str:
        keywords = ", ".join(paper.keywords)
        return f"Title: {paper.title}\nAbstract: {paper.abstract or ''}\nKeywords: {keywords}"

    def _deterministic(self, paper: Stage2Paper) -> DeterministicRuleDecision | None:
        document_type = (paper.document_type or "").strip().casefold()
        if document_type in self.profile.exclude_document_types:
            return DeterministicRuleDecision(CascadeRoute.IRRELEVANT, f"document_type_excluded:{document_type}")
        if document_type in self.profile.include_document_types:
            return DeterministicRuleDecision(CascadeRoute.RELEVANT, f"document_type_included:{document_type}")
        return None

    def _rerank(self, papers: Sequence[Stage2Paper]) -> tuple[dict[str, float], tuple[Stage2Paper, ...]]:
        scores: dict[str, float] = {}
        failures: list[Stage2Paper] = []
        buckets: dict[int, list[Stage2Paper]] = {}
        for paper in papers:
            buckets.setdefault(self._bucket(paper), []).append(paper)
        for bucket in sorted(buckets):
            batch = tuple(sorted(buckets[bucket], key=lambda item: item.paper_id))
            try:
                response = self.reranker.rerank(
                    self.profile.query,
                    tuple(RerankInput(paper.paper_id, self.document(paper)) for paper in batch),
                )
            except RerankBatchError as error:
                scores.update({item.paper_id: item.raw_score for item in error.scores})
                failed = set(error.failed_paper_ids)
                failures.extend(paper for paper in batch if paper.paper_id in failed)
                continue
            except _FAILURES:
                failures.extend(batch)
                continue
            expected = {paper.paper_id for paper in batch}
            returned = {item.paper_id: item.raw_score for item in response}
            if not set(returned) <= expected:
                failures.extend(batch)
                continue
            scores.update(returned)
        return scores, tuple(failures)

    def _reranker_route(self, paper: Stage2Paper, score: float) -> tuple[CascadeRoute, float | None]:
        cascade_input = CascadeInput(
            raw_score=score,
            abstract_missing=not bool(paper.abstract and paper.abstract.strip()),
            possibly_truncated=paper.possibly_truncated,
            multi_condition_conflict=paper.multi_condition_conflict,
            language_anomaly=paper.language_anomaly,
        )
        binding = self.profile.reranker_calibration
        if binding is None:
            assert self.profile.thresholds is not None
            return route_cascade(cascade_input, self.profile.thresholds), None
        probability = binding.calibrator.predict(score)
        if any((
            cascade_input.abstract_missing,
            cascade_input.possibly_truncated,
            cascade_input.multi_condition_conflict,
            cascade_input.language_anomaly,
        )):
            return CascadeRoute.ADJUDICATE, probability
        threshold = binding.threshold
        if probability <= threshold.low:
            return CascadeRoute.IRRELEVANT, probability
        if probability >= threshold.high:
            return CascadeRoute.RELEVANT, probability
        return CascadeRoute.ADJUDICATE, probability

    def _adjudicate(
        self,
        requests: Sequence[tuple[Stage2Paper, float | None, float | None, str]],
    ) -> tuple[Stage2Decision, ...]:
        if not requests:
            return ()
        with ThreadPoolExecutor(max_workers=self.profile.adjudicator_concurrency) as executor:
            return tuple(executor.map(lambda item: self._adjudicate_one(*item), requests))

    def _adjudicate_one(
        self,
        paper: Stage2Paper,
        reranker_score: float | None,
        reranker_probability: float | None,
        route_reason: str,
    ) -> Stage2Decision:
        input_hash = self.input_hash(paper)
        request = AdjudicationInput(paper.paper_id, (
            {"role": "system", "content": ADJUDICATION_SYSTEM_PROMPT},
            {"role": "user", "content": self._adjudication_prompt(paper)},
        ))
        retry_reason: str | None = None
        for attempt_count in (1, 2):
            try:
                response = self.adjudicator.adjudicate(request)
            except StructuredOutputError:
                failure_reason = "adjudicator_schema_failure"
            except _RETRYABLE_ADJUDICATOR_FAILURES:
                failure_reason = "adjudicator_backend_failure"
            else:
                if self._valid_adjudication(response, paper.paper_id):
                    break
                failure_reason = "adjudicator_schema_failure"
            if attempt_count == 1:
                retry_reason = failure_reason
                continue
            return Stage2Decision(
                paper_id=paper.paper_id,
                status=FilterStatus.NEEDS_REVIEW,
                reason_code=failure_reason,
                input_hash=input_hash,
                route=CascadeRoute.NEEDS_REVIEW,
                reranker_score=reranker_score,
                reranker_probability=reranker_probability,
                adjudicator_probability=self._failure_probability(),
                adjudicated=True,
                adjudicator_attempt_count=attempt_count,
                adjudicator_retry_reason=retry_reason,
                adjudicator_retry_outcome="failed",
            )
        structured_route = CascadeRoute(response.decision)
        retry_outcome = "succeeded" if retry_reason is not None else None
        adjudicator_probability = None
        route = structured_route
        conflict = False
        binding = self.profile.adjudicator_calibration
        if binding is not None:
            try:
                adjudicator_probability = binding.calibrator.predict(response.score)
            except ValueError:
                return Stage2Decision(
                    paper_id=paper.paper_id,
                    status=FilterStatus.NEEDS_REVIEW,
                    reason_code="adjudicator_calibration_failure",
                    input_hash=input_hash,
                    route=CascadeRoute.NEEDS_REVIEW,
                    reranker_score=reranker_score,
                    reranker_probability=reranker_probability,
                    adjudicator_score=response.score,
                    adjudicator_probability=self._failure_probability(),
                    adjudicated=True,
                    adjudicator_attempt_count=attempt_count,
                    adjudicator_retry_reason=retry_reason,
                    adjudicator_retry_outcome=retry_outcome,
                )
            threshold = binding.threshold
            if adjudicator_probability <= threshold.low:
                calibrated_route = CascadeRoute.IRRELEVANT
            elif adjudicator_probability >= threshold.high:
                calibrated_route = CascadeRoute.RELEVANT
            else:
                calibrated_route = CascadeRoute.NEEDS_REVIEW
            conflict = structured_route is not calibrated_route
            route = CascadeRoute.NEEDS_REVIEW if conflict else calibrated_route
        reason = ",".join(response.reason_codes)
        if conflict:
            reason = f"qwen_calibration_conflict:{structured_route.value}:{reason}"
        rationale = response.rationale if route in {CascadeRoute.RELEVANT, CascadeRoute.NEEDS_REVIEW} else None
        return Stage2Decision(
            paper_id=paper.paper_id,
            status=_status(route),
            reason_code=f"{route_reason}:{reason}",
            input_hash=input_hash,
            route=route,
            reranker_score=reranker_score,
            reranker_probability=reranker_probability,
            adjudicator_score=response.score,
            adjudicator_probability=adjudicator_probability,
            rationale=rationale,
            adjudicated=True,
            adjudicator_attempt_count=attempt_count,
            adjudicator_retry_reason=retry_reason,
            adjudicator_retry_outcome=retry_outcome,
        )

    def _adjudication_prompt(self, paper: Stage2Paper) -> str:
        return ADJUDICATION_USER_TEMPLATE.format(
            query_version=self.profile.query_version,
            query=self.profile.query,
            paper_id=paper.paper_id,
            document=self.document(paper),
        )

    def _adjudication_reason(self, paper: Stage2Paper) -> str:
        if not paper.abstract or not paper.abstract.strip():
            return "missing_abstract"
        if paper.possibly_truncated:
            return "possible_truncation"
        if paper.multi_condition_conflict:
            return "multi_condition_conflict"
        if paper.language_anomaly:
            return "language_anomaly"
        return "uncertain_reranker_band"

    def _from_route(
        self,
        paper: Stage2Paper,
        input_hash: str,
        route: CascadeRoute,
        reason_code: str,
        reranker_score: float | None = None,
        reranker_probability: float | None = None,
    ) -> Stage2Decision:
        return Stage2Decision(
            paper_id=paper.paper_id,
            status=_status(route),
            reason_code=reason_code,
            input_hash=input_hash,
            route=route,
            reranker_score=reranker_score,
            reranker_probability=reranker_probability,
        )

    def _bucket(self, paper: Stage2Paper) -> int:
        tokens = max(1, len(self.document(paper).split()))
        return (tokens // self.profile.token_bucket_width) * self.profile.token_bucket_width

    def _lease_output_kind(self, paper: Stage2Paper) -> str:
        return f"{STAGE2_LEASE_OUTPUT_PREFIX}{self._bucket(paper):012d}"

    @staticmethod
    def _valid_adjudication(response: AdjudicationDecision, paper_id: str) -> bool:
        return (
            response.paper_id == paper_id
            and response.decision in {"relevant", "irrelevant", "needs_review"}
            and 0 <= response.score <= 1
            and bool(response.reason_codes)
        )

    def _ensure_run(self, run_id: str, papers: Sequence[Stage2Paper]) -> None:
        input_hash = _hash([self.input_hash(paper) for paper in papers])
        with self.database.transaction() as connection:
            existing = connection.execute(
                "SELECT stage, input_hash, config_hash, implementation_version FROM pipeline_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            expected = (
                "stage-2",
                input_hash,
                self.profile.config_hash,
                self.implementation_version,
            )
            if existing is not None:
                actual = tuple(existing[column] for column in (
                    "stage", "input_hash", "config_hash", "implementation_version"
                ))
                if actual != expected:
                    raise ValueError("Stage 2 run input or configuration is immutable")
                return
            connection.execute(
                """INSERT INTO pipeline_runs(run_id, stage, status, input_hash, config_hash, implementation_version, started_at)
                   VALUES (?, 'stage-2', 'running', ?, ?, ?, CURRENT_TIMESTAMP)""",
                (run_id, input_hash, self.profile.config_hash, self.implementation_version),
            )

    def _load_completed(self, run_id: str) -> dict[tuple[str, str], Stage2Decision]:
        rows = self.database.connection.execute(
            """SELECT paper_id, status, score, reason, input_hash, adjudicator_attempt_count,
                      adjudicator_retry_reason, adjudicator_retry_outcome FROM filter_decisions
               WHERE run_id = ? ORDER BY created_at, filter_decision_id""",
            (run_id,),
        ).fetchall()
        completed: dict[tuple[str, str], Stage2Decision] = {}
        for row in rows:
            detail = json.loads(row["reason"])
            completed[(row["paper_id"], row["input_hash"])] = Stage2Decision(
                paper_id=row["paper_id"],
                status=FilterStatus(row["status"]),
                reason_code=detail["reason_code"],
                input_hash=row["input_hash"],
                route=CascadeRoute(detail["route"]),
                reranker_score=detail.get("reranker_score"),
                reranker_probability=detail.get("reranker_probability"),
                adjudicator_score=detail.get("adjudicator_score"),
                adjudicator_probability=detail.get("adjudicator_probability"),
                rationale=detail.get("rationale"),
                adjudicated=bool(detail.get("adjudicated")),
                adjudicator_attempt_count=int(row["adjudicator_attempt_count"]),
                adjudicator_retry_reason=row["adjudicator_retry_reason"],
                adjudicator_retry_outcome=row["adjudicator_retry_outcome"],
                resumed=True,
            )
        return completed

    def _persist_claimed(
        self,
        run_id: str,
        claimed: Sequence[tuple[TaskLease, Stage2Decision]],
    ) -> None:
        """Publish model results and consume their fences atomically."""
        with self.database.transaction() as connection:
            validation_time = _timestamp(self.lease_clock())
            for lease, decision in claimed:
                if (
                    lease.run_id != run_id
                    or lease.stage != "stage-2"
                    or not lease.output_kind.startswith(STAGE2_LEASE_OUTPUT_PREFIX)
                    or lease.paper_id != decision.paper_id
                    or lease.input_hash != decision.input_hash
                    or lease.worker_id != self.worker_id
                ):
                    raise LeaseNotCurrent("Stage 2 result does not match its claimed task")
                LeaseQueue.require_current(
                    connection,
                    task_id=lease.task_id,
                    worker_id=self.worker_id,
                    fencing_token=lease.fencing_token,
                    now=validation_time,
                )

            for lease, decision in claimed:
                model_id, model_revision = self._decision_model(decision)
                provenance = {
                    "reason_code": decision.reason_code,
                    "route": decision.route.value,
                    "reranker_score": decision.reranker_score,
                    "reranker_probability": decision.reranker_probability,
                    "adjudicator_score": decision.adjudicator_score,
                    "adjudicator_probability": decision.adjudicator_probability,
                    "model": model_id,
                    "revision": model_revision,
                    "adjudicated": decision.adjudicated,
                    "adjudicator_attempt_count": decision.adjudicator_attempt_count,
                    "adjudicator_retry_reason": decision.adjudicator_retry_reason,
                    "adjudicator_retry_outcome": decision.adjudicator_retry_outcome,
                    "prompt_version": self.profile.prompt_version,
                    "schema_version": self.profile.schema_version,
                    "screening_scope_hash": self.profile.screening_scope_hash,
                    "base_runtime_config_hash": self.profile.base_runtime_config_hash,
                    "threshold_bundle_hash": self.profile.threshold_bundle_hash,
                    "full_profile_hash": self.profile.full_profile_hash,
                    "reranker_calibrator_hash": (
                        self.profile.reranker_calibration.calibrator.hash()
                        if self.profile.reranker_calibration is not None else None
                    ),
                    "reranker_threshold_hash": (
                        self.profile.reranker_calibration.threshold.hash()
                        if self.profile.reranker_calibration is not None else self.profile.threshold_hash
                    ),
                    "qwen_calibrator_hash": (
                        self.profile.adjudicator_calibration.calibrator.hash()
                        if self.profile.adjudicator_calibration is not None else None
                    ),
                    "qwen_threshold_hash": (
                        self.profile.adjudicator_calibration.threshold.hash()
                        if self.profile.adjudicator_calibration is not None else None
                    ),
                    "reranker_lock_hash": self.profile.reranker_lock_hash,
                    "adjudicator_lock_hash": self.profile.adjudicator_lock_hash,
                    "release_gate_hash": self.profile.release_gate_hash,
                    "task_lease": {
                        "task_id": lease.task_id,
                        "worker_id": self.worker_id,
                        "lease_expires_at": lease.lease_expires_at,
                        "attempt": lease.attempt,
                        "fencing_token": lease.fencing_token,
                    },
                }
                if decision.rationale is not None:
                    provenance["rationale"] = decision.rationale
                detail = json.dumps(provenance, sort_keys=True, separators=(",", ":"))
                decision_id = _hash([run_id, decision.paper_id, "filter"])
                event_id = _hash([run_id, decision.paper_id, "screening"])
                connection.execute(
                    """INSERT INTO filter_decisions(
                        filter_decision_id, run_id, paper_id, status, score, threshold_version, reason,
                        input_hash, implementation_version, model_id, model_revision, prompt_hash, schema_hash,
                        adjudicator_attempt_count, adjudicator_retry_reason, adjudicator_retry_outcome
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        decision_id, run_id, decision.paper_id, decision.status.value,
                        self._decision_probability(decision),
                        self.profile.threshold_version, detail, decision.input_hash, self.implementation_version,
                        provenance["model"], provenance["revision"], self.profile.prompt_hash, self.profile.schema_hash,
                        decision.adjudicator_attempt_count, decision.adjudicator_retry_reason,
                        decision.adjudicator_retry_outcome,
                    ),
                )
                connection.execute(
                    """INSERT INTO screening_events(
                        screening_event_id, run_id, paper_id, criterion_id, decision, reason_code,
                        input_hash, implementation_version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        event_id, run_id, decision.paper_id,
                        self.implementation_version,
                        _screening_status(decision.status), decision.reason_code, decision.input_hash,
                        self.implementation_version,
                    ),
                )

            for lease, _decision in claimed:
                LeaseQueue.complete_in_transaction(
                    connection,
                    task_id=lease.task_id,
                    worker_id=self.worker_id,
                    fencing_token=lease.fencing_token,
                    now=_timestamp(self.lease_clock()),
                    retain_claim=True,
                )

    def _decision_model(self, decision: Stage2Decision) -> tuple[str | None, str | None]:
        if decision.adjudicated:
            return self.profile.adjudicator_model_id, self.profile.adjudicator_revision
        if decision.reason_code.startswith("document_type_"):
            return None, None
        return self.profile.reranker_model_id, self.profile.reranker_revision

    def _failure_probability(self) -> float | None:
        return 0.5 if self.profile.production_calibrated else None

    @staticmethod
    def _decision_probability(decision: Stage2Decision) -> float | None:
        for value in (
            decision.adjudicator_probability,
            decision.reranker_probability,
            decision.adjudicator_score,
            decision.reranker_score,
        ):
            if value is not None:
                return value
        return None


def _status(route: CascadeRoute) -> FilterStatus:
    return FilterStatus.NEEDS_REVIEW if route is CascadeRoute.NEEDS_REVIEW else FilterStatus(route.value)


def _screening_status(status: FilterStatus) -> str:
    return {FilterStatus.RELEVANT: "included", FilterStatus.IRRELEVANT: "excluded", FilterStatus.NEEDS_REVIEW: "needs_review"}[status]


def _hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
