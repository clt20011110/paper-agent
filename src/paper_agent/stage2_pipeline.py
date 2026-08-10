"""Production orchestration for the local Stage 2 screening cascade.

The backend module owns model transports.  This module owns the stable input
contract, cascade routing, and SQLite evidence trail used to resume a run.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import hashlib
import json
from typing import Iterable, Sequence

from .domain import FilterStatus
from .stage2_backends import (
    AdjudicationDecision,
    AdjudicationInput,
    AdjudicatorBackend,
    CascadeInput,
    CascadeRoute,
    DeterministicRuleDecision,
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


IMPLEMENTATION_VERSION = "stage2-cascade-v1"
ADJUDICATION_SYSTEM_PROMPT = "Return only the required structured screening decision."
ADJUDICATION_USER_TEMPLATE = (
    "Query version: {query_version}\nQuery: {query}\nPaper ID: {paper_id}\n{document}"
)
_FAILURES = (Stage2BackendError, StructuredOutputError, TimeoutError, OSError, ValueError)


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
            "kind": "stage2-base-runtime-v1",
            "query": self.query,
            "query_version": self.query_version,
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
    resumed: bool = False


@dataclass(frozen=True, slots=True)
class Stage2Summary:
    decisions: tuple[Stage2Decision, ...]
    reranked_count: int
    qwen_count: int
    qwen_share: float
    qwen_alarms: tuple[str, ...]


@dataclass(slots=True)
class Stage2Pipeline:
    database: Database
    reranker: RerankerBackend
    adjudicator: AdjudicatorBackend
    profile: Stage2Profile
    implementation_version: str = IMPLEMENTATION_VERSION
    _completed: dict[tuple[str, str], Stage2Decision] = field(default_factory=dict, init=False)

    def run(self, run_id: str, papers: Iterable[Stage2Paper]) -> Stage2Summary:
        self.profile.assert_runtime_ready()
        candidates = tuple(sorted(papers, key=lambda item: item.paper_id))
        if len({item.paper_id for item in candidates}) != len(candidates):
            raise ValueError("a Stage 2 call cannot contain duplicate paper_ids")
        self._ensure_run(run_id, candidates)
        self._completed = self._load_completed(run_id)

        decisions: dict[str, Stage2Decision] = {}
        rerank_inputs: list[Stage2Paper] = []
        for paper in candidates:
            input_hash = self.input_hash(paper)
            completed = self._completed.get((paper.paper_id, input_hash))
            if completed is not None:
                decisions[paper.paper_id] = completed
                continue
            deterministic = self._deterministic(paper)
            if deterministic is not None:
                decisions[paper.paper_id] = self._from_route(paper, input_hash, deterministic.route, deterministic.reason_code)
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
                    "reranker_probability_threshold" if probability is not None else "reranker_threshold",
                    score,
                    probability,
                )

        for decision in self._adjudicate(adjudication):
            decisions[decision.paper_id] = decision

        ordered = tuple(decisions[paper.paper_id] for paper in candidates)
        self._persist(run_id, ordered)
        with self.database.transaction() as connection:
            connection.execute(
                "UPDATE pipeline_runs SET status = 'complete', completed_at = CURRENT_TIMESTAMP WHERE run_id = ?",
                (run_id,),
            )
        reranked_count = len(rerank_inputs)
        qwen_count = sum(item.adjudicated for item in ordered)
        qwen_share = qwen_count / len(candidates) if candidates else 0.0
        alarms = tuple(
            label for limit, label in ((0.15, "qwen_share_over_15_percent"), (0.30, "qwen_share_over_30_percent"))
            if qwen_share > limit
        )
        return Stage2Summary(ordered, reranked_count, qwen_count, qwen_share, alarms)

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
        try:
            response = self.adjudicator.adjudicate(request)
        except _FAILURES:
            return Stage2Decision(
                paper_id=paper.paper_id,
                status=FilterStatus.NEEDS_REVIEW,
                reason_code="adjudicator_backend_failure",
                input_hash=input_hash,
                route=CascadeRoute.NEEDS_REVIEW,
                reranker_score=reranker_score,
                reranker_probability=reranker_probability,
                adjudicator_probability=self._failure_probability(),
                adjudicated=True,
            )
        if not self._valid_adjudication(response, paper.paper_id):
            return Stage2Decision(
                paper_id=paper.paper_id,
                status=FilterStatus.NEEDS_REVIEW,
                reason_code="adjudicator_schema_failure",
                input_hash=input_hash,
                route=CascadeRoute.NEEDS_REVIEW,
                reranker_score=reranker_score,
                reranker_probability=reranker_probability,
                adjudicator_probability=self._failure_probability(),
                adjudicated=True,
            )
        structured_route = CascadeRoute(response.decision)
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
            """SELECT paper_id, status, score, reason, input_hash FROM filter_decisions
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
                resumed=True,
            )
        return completed

    def _persist(self, run_id: str, decisions: Sequence[Stage2Decision]) -> None:
        with self.database.transaction() as connection:
            for decision in decisions:
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
                    "prompt_version": self.profile.prompt_version,
                    "schema_version": self.profile.schema_version,
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
                }
                if decision.rationale is not None:
                    provenance["rationale"] = decision.rationale
                detail = json.dumps(provenance, sort_keys=True, separators=(",", ":"))
                decision_id = _hash([run_id, decision.paper_id, "filter"])
                event_id = _hash([run_id, decision.paper_id, "screening"])
                connection.execute(
                    """INSERT INTO filter_decisions(
                        filter_decision_id, run_id, paper_id, status, score, threshold_version, reason,
                        input_hash, implementation_version, model_id, model_revision, prompt_hash, schema_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_id, paper_id) DO NOTHING""",
                    (
                        decision_id, run_id, decision.paper_id, decision.status.value,
                        self._decision_probability(decision),
                        self.profile.threshold_version, detail, decision.input_hash, self.implementation_version,
                        provenance["model"], provenance["revision"], self.profile.prompt_hash, self.profile.schema_hash,
                    ),
                )
                connection.execute(
                    """INSERT INTO screening_events(
                        screening_event_id, run_id, paper_id, criterion_id, decision, reason_code,
                        input_hash, implementation_version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_id, paper_id, criterion_id) DO NOTHING""",
                    (
                        event_id, run_id, decision.paper_id,
                        self.implementation_version,
                        _screening_status(decision.status), decision.reason_code, decision.input_hash,
                        self.implementation_version,
                    ),
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
