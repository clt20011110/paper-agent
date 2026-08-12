"""Released local Stage 2 runtime used by search and citation rounds."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse
from uuid import NAMESPACE_URL, uuid5

from .approval import ApprovalError, require_valid_approval
from .canonical import content_hash
from .domain import FilterStatus
from .query_plan import QueryPlanError, assert_screening_scope_hash
from .repository import PaperRepository
from .schema import SchemaValidationError, schema_directory, validate
from .stage2_backends import (
    ModelLock,
    OmlxChatBackend,
    OmlxRerankBackend,
    OmlxTransport,
    UrlLibOmlxTransport,
)
from .stage2_evaluation import (
    CalibrationPath,
    GateResult,
    PathCalibrator,
    ReleaseGateResult,
    ThresholdArtifact as ProbabilityThresholdArtifact,
    GoldSplit,
    gold_manifest_from_document,
    pair_universe_hash,
    phase3_release_gate,
)
from .stage2_fallback import (
    FallbackReleaseBinding,
    LocalCalibratedRerankerFallback,
    stage2_effective_config_hash,
    stage2_shared_runtime_hash,
)
from .stage2_hidden_attestation import HIDDEN_PROMOTION_GATE_POLICY_HASH
from .stage2_hidden_attestation import (
    HiddenEvaluatorTrust,
    HiddenPromotionBindings,
    ReleaseRole,
    load_hidden_evaluator_trust,
    verify_hidden_promotion_attestation,
)
from .stage2_parity_oracle_trust import (
    ParityOracleTrust,
    load_parity_oracle_trust,
)
from .stage2_public_gates import verify_public_stage2_gates
from .stage2_release_evidence import (
    Stage2ReleaseEvidenceIndex,
    load_stage2_release_evidence_index_bytes,
)
from .stage2_pipeline import (
    ADJUDICATOR_SHARE_ALARM,
    ERROR_RATE_ALARM,
    PathCalibration,
    Stage2Decision,
    Stage2Paper,
    Stage2Pipeline,
    Stage2Profile,
    Stage2Summary,
    adjudicator_capacity,
    qwen_capacity_level,
)
from .storage import Database


_RELEASE_FIELDS = frozenset({
    "schema_version", "profile", "reranker_lock", "adjudicator_lock",
    "calibration", "release_gate", "runtime",
})
_BENCHMARK_CANDIDATE_FIELDS = _RELEASE_FIELDS - {"release_gate"}
_RUNTIME_FIELDS = frozenset({
    "query", "query_version", "screening_scope_hash",
    "evaluation_topic_queries",
    "include_document_types", "exclude_document_types",
    "token_bucket_width", "document_batch_size", "max_in_flight",
    "adjudicator_concurrency", "adjudicator_seed", "max_context_window", "max_tokens",
    "omlx_base_url", "api_key_env", "prompt_version", "schema_version",
})
_RELEASE_GATE_FIELDS = frozenset({
    "candidate_id", "candidate_bundle_sha256", "evaluation_manifest_hash", "evidence",
})
_PATH_NAMES = frozenset({CalibrationPath.RERANKER.value, CalibrationPath.QWEN.value})


class Stage2ReleaseError(ValueError):
    """A production Stage 2 release is missing, failed, or drifted."""


@dataclass(frozen=True, slots=True)
class ReleasedRerankerFallback:
    """A separately calibrated, independently released local backup reranker."""

    model_lock: ModelLock
    model_lock_hash: str
    calibration: PathCalibration
    omlx_base_url: str
    api_key_env: str | None
    release_binding: FallbackReleaseBinding
    runtime_config_hash: str

    def identity_document(self) -> dict[str, object]:
        """Match the runtime pipeline identity before any request is sent."""

        return {
            "backend": "omlx_rerank",
            "model_id": self.model_lock.model_id,
            "model_revision": _runtime_revision(self.model_lock),
            "model_lock_hash": self.model_lock_hash,
            "calibrator_hash": self.calibration.calibrator.hash(),
            "threshold_hash": self.calibration.threshold.hash(),
            "release_binding": self.release_binding.document(),
            "runtime_config_hash": self.runtime_config_hash,
        }


@dataclass(frozen=True, slots=True)
class ReleasedStage2:
    profile_name: str
    profile: Stage2Profile
    release_hash: str
    omlx_base_url: str
    api_key_env: str | None = None
    reranker_fallback: ReleasedRerankerFallback | None = None

    @property
    def effective_config_hash(self) -> str:
        """Configuration hash that QueryPlan and SQLite resume must bind."""

        return stage2_effective_config_hash(
            self.profile.config_hash,
            (
                self.reranker_fallback.identity_document()
                if self.reranker_fallback is not None
                else None
            ),
        )
    def screener(
        self,
        database: Database,
        campaign_id: str,
        *,
        transport: OmlxTransport | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> "Stage2SearchScreener":
        values = environment if environment is not None else os.environ
        local_transport = transport or UrlLibOmlxTransport(
            self.omlx_base_url,
            api_key=values.get(self.api_key_env) if self.api_key_env else None,
        )
        schema = json.loads(
            (schema_directory() / self.profile.schema_version).read_text(encoding="utf-8")
        )
        pipeline = Stage2Pipeline(
            database,
            OmlxRerankBackend(
                self.profile.reranker_model_id,
                local_transport,
                document_batch_size=self.profile.document_batch_size,
                max_in_flight=self.profile.reranker_max_in_flight,
            ),
            OmlxChatBackend(
                self.profile.adjudicator_model_id,
                local_transport,
                schema,
                seed=self.profile.adjudicator_seed,
                max_context_window=self.profile.adjudicator_max_context_window,
                max_output_tokens=self.profile.adjudicator_max_output_tokens,
            ),
            self.profile,
            reranker_fallback=self._reranker_fallback(local_transport, values),
        )
        return Stage2SearchScreener(database, pipeline, campaign_id)

    def _reranker_fallback(
        self,
        primary_transport: OmlxTransport,
        environment: Mapping[str, str],
    ) -> LocalCalibratedRerankerFallback | None:
        fallback = self.reranker_fallback
        if fallback is None:
            return None
        transport = primary_transport
        if (
            fallback.omlx_base_url != self.omlx_base_url
            or fallback.api_key_env != self.api_key_env
        ):
            transport = UrlLibOmlxTransport(
                fallback.omlx_base_url,
                api_key=environment.get(fallback.api_key_env) if fallback.api_key_env else None,
            )
        return LocalCalibratedRerankerFallback(
            backend=OmlxRerankBackend(
                fallback.model_lock.model_id,
                transport,
                document_batch_size=self.profile.document_batch_size,
                max_in_flight=self.profile.reranker_max_in_flight,
            ),
            model_id=fallback.model_lock.model_id,
            model_revision=_runtime_revision(fallback.model_lock),
            model_lock_hash=fallback.model_lock_hash,
            calibration=fallback.calibration,
            release_binding=fallback.release_binding,
            runtime_config_hash=fallback.runtime_config_hash,
        )


@dataclass(frozen=True, slots=True)
class HiddenPromotionBatchBinding:
    """Public fields proving two attestations came from one sealed batch."""

    promotion_batch_hash: str
    evaluation_run_id: str
    promotion_marker_hash: str
    winner_candidate_id: str
    evaluator_id: str
    trust_manifest_hash: str
    issued_at: str

    @classmethod
    def from_attestation(
        cls, document: Mapping[str, Any]
    ) -> "HiddenPromotionBatchBinding":
        payload = _object(document, "payload")
        return cls(
            _sha256_text(payload, "promotion_batch_hash"),
            _text(payload, "evaluation_run_id"),
            _sha256_text(payload, "promotion_marker_hash"),
            _text(payload, "winner_candidate_id"),
            _text(payload, "evaluator_id"),
            _sha256_text(payload, "trust_manifest_hash"),
            _text(payload, "issued_at"),
        )


def stage2_base_profile(
    runtime: Mapping[str, Any],
    reranker_lock: ModelLock,
    adjudicator_lock: ModelLock,
    *,
    reranker_lock_hash: str,
    adjudicator_lock_hash: str,
    release_gate_hash: str | None = None,
) -> Stage2Profile:
    """Build the uncalibrated profile shared by candidate creation and loading."""

    _validate_locks(reranker_lock, adjudicator_lock)
    if any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in (reranker_lock_hash, adjudicator_lock_hash)
    ):
        raise Stage2ReleaseError(
            "Stage 2 model lock hashes must be lowercase SHA-256 values"
        )
    _exact_fields(runtime, _RUNTIME_FIELDS, "Stage 2 runtime")
    screening_scope_hash = _sha256_text(runtime, "screening_scope_hash")
    base_url = _text(runtime, "omlx_base_url")
    _require_loopback(base_url)
    api_key_env = runtime.get("api_key_env")
    if api_key_env is not None and (
        not isinstance(api_key_env, str) or not api_key_env
    ):
        raise Stage2ReleaseError(
            "Stage 2 api_key_env must be a non-empty string or null"
        )
    prompt_version = _text(runtime, "prompt_version")
    schema_version = _text(runtime, "schema_version")
    if (
        prompt_version != "stage2-adjudication-v1"
        or schema_version != "filter-decision.schema.json"
    ):
        raise Stage2ReleaseError(
            "Stage 2 release uses an unsupported prompt or schema version"
        )
    try:
        return Stage2Profile(
            query=_text(runtime, "query"),
            query_version=_text(runtime, "query_version"),
            thresholds=None,
            reranker_model_id=reranker_lock.model_id,
            reranker_revision=_runtime_revision(reranker_lock),
            adjudicator_model_id=adjudicator_lock.model_id,
            adjudicator_revision=_runtime_revision(adjudicator_lock),
            screening_scope_hash=screening_scope_hash,
            evaluation_topic_queries=_evaluation_topic_queries(runtime),
            reranker_lock_hash=reranker_lock_hash,
            adjudicator_lock_hash=adjudicator_lock_hash,
            release_gate_hash=release_gate_hash,
            include_document_types=frozenset(
                _string_list(runtime, "include_document_types")
            ),
            exclude_document_types=frozenset(
                _string_list(runtime, "exclude_document_types")
            ),
            token_bucket_width=_integer(runtime, "token_bucket_width"),
            document_batch_size=_integer(runtime, "document_batch_size"),
            reranker_max_in_flight=_integer(runtime, "max_in_flight"),
            adjudicator_concurrency=_integer(runtime, "adjudicator_concurrency"),
            adjudicator_seed=_integer(runtime, "adjudicator_seed"),
            adjudicator_max_context_window=_integer(runtime, "max_context_window"),
            adjudicator_max_output_tokens=_integer(runtime, "max_tokens"),
            omlx_base_url=base_url,
            api_key_env=api_key_env,
            prompt_version=prompt_version,
            schema_version=schema_version,
        )
    except (OSError, ValueError) as error:
        raise Stage2ReleaseError(f"Stage 2 runtime is invalid: {error}") from error


@dataclass(slots=True)
class Stage2SearchScreener:
    """Adapt canonical database papers to the existing Stage 2 cascade."""

    database: Database
    pipeline: Stage2Pipeline
    campaign_id: str
    repository: PaperRepository = field(init=False)
    decisions: dict[str, Stage2Decision] = field(default_factory=dict, init=False)
    run_ids: list[str] = field(default_factory=list, init=False)
    summaries: dict[str, Stage2Summary] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.repository = PaperRepository(self.database)

    def screen(self, paper_ids: Sequence[str]) -> Mapping[str, FilterStatus]:
        ordered_ids = tuple(sorted(set(paper_ids)))
        papers = tuple(self._paper(paper_id) for paper_id in ordered_ids)
        run_id = f"stage2-{uuid5(NAMESPACE_URL, f'{self.campaign_id}:{len(self.run_ids)}').hex}"
        summary = self.pipeline.run(run_id, papers)
        self.run_ids.append(run_id)
        self.summaries[run_id] = summary
        self.decisions.update((decision.paper_id, decision) for decision in summary.decisions)
        return {decision.paper_id: decision.status for decision in summary.decisions}

    def telemetry(self) -> dict[str, object]:
        run_details = [
            self.summaries[run_id].telemetry(run_id) for run_id in self.run_ids
        ]
        paper_count = sum(int(run["paper_count"]) for run in run_details)
        reranked_count = sum(int(run["reranked_count"]) for run in run_details)
        qwen_count = sum(int(run["qwen_count"]) for run in run_details)
        error_count = sum(int(run["error_count"]) for run in run_details)
        error_rate = error_count / paper_count if paper_count else 0.0
        max_run_qwen_share = max(
            (float(run["qwen_share"]) for run in run_details),
            default=0.0,
        )
        max_run_error_rate = max(
            (float(run["error_rate"]) for run in run_details),
            default=0.0,
        )
        alarm_codes = tuple(
            alarm
            for alarm in (
                ADJUDICATOR_SHARE_ALARM,
                ERROR_RATE_ALARM,
            )
            if (
                alarm == ADJUDICATOR_SHARE_ALARM
                and any(alarm in run["alarm_codes"] for run in run_details)
            )
            or (alarm == ERROR_RATE_ALARM and error_rate >= 0.005)
        )
        return {
            "stage2_run_ids": list(self.run_ids),
            "screened_count": paper_count,
            "reranked_count": reranked_count,
            "adjudicator_count": qwen_count,
            "adjudicator_share": qwen_count / paper_count if paper_count else 0.0,
            "adjudicator_capacity": adjudicator_capacity(max_run_qwen_share),
            "paper_count": paper_count,
            "qwen_count": qwen_count,
            "qwen_share": qwen_count / paper_count if paper_count else 0.0,
            "error_count": error_count,
            "error_rate": error_rate,
            "max_run_error_rate": max_run_error_rate,
            "max_run_qwen_share": max_run_qwen_share,
            "max_run_adjudicator_share": max_run_qwen_share,
            "capacity_level": qwen_capacity_level(max_run_qwen_share),
            "run_details": run_details,
            "alarm_codes": list(alarm_codes),
        }

    def reranker_score(self, paper_id: str) -> float:
        decision = self.decisions[paper_id]
        if decision.reranker_score is not None:
            return decision.reranker_score
        return 1.0 if decision.status is FilterStatus.RELEVANT else 0.0

    def _paper(self, paper_id: str) -> Stage2Paper:
        paper = self.repository.get_paper(paper_id)
        if paper is None:
            raise ValueError(f"Stage 2 paper does not exist: {paper_id}")
        return Stage2Paper(
            paper.paper_id,
            paper.title,
            paper.abstract,
            paper.keywords,
            document_type=self._document_type(paper_id),
            possibly_truncated=_possibly_truncated(paper.title, paper.abstract, paper.keywords),
        )

    def _document_type(self, paper_id: str) -> str | None:
        rows = self.database.connection.execute(
            "SELECT raw_metadata_json FROM paper_sources WHERE paper_id = ? ORDER BY provider, external_id",
            (paper_id,),
        ).fetchall()
        for row in rows:
            metadata = json.loads(row["raw_metadata_json"])
            for key in ("document_type", "publication_type", "type"):
                value = metadata.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip().casefold()
        return None


def load_stage2_release(
    path: Path,
    plan: Mapping[str, Any],
    *,
    hidden_trust_path: Path | None = None,
    parity_oracle_trust_path: Path | None = None,
    environment: Mapping[str, str] | None = None,
) -> ReleasedStage2:
    """Load a v3 release and bind independent public/hidden trust to its profile.

    Trust roots are deployment-controlled: they are supplied as explicit paths
    or through ``PAPER_AGENT_STAGE2_HIDDEN_TRUST`` and
    ``PAPER_AGENT_STAGE2_PARITY_ORACLE_TRUST``.  A release bundle cannot select
    its own trust roots.
    """

    values = environment if environment is not None else os.environ
    trust_path = hidden_trust_path
    if trust_path is None:
        configured_path = values.get("PAPER_AGENT_STAGE2_HIDDEN_TRUST")
        if configured_path:
            trust_path = Path(configured_path)
    oracle_trust_path = parity_oracle_trust_path
    if oracle_trust_path is None:
        configured_path = values.get("PAPER_AGENT_STAGE2_PARITY_ORACLE_TRUST")
        if configured_path:
            oracle_trust_path = Path(configured_path)
    return _load_stage2_bundle(
        path,
        plan,
        hidden_trust_path=trust_path,
        parity_oracle_trust_path=oracle_trust_path,
    )


def load_stage2_benchmark_candidate(path: Path) -> ReleasedStage2:
    """Load frozen models, calibrations, thresholds, and runtime before throughput gates."""

    return _load_stage2_bundle(
        path,
        None,
        hidden_trust_path=None,
        parity_oracle_trust_path=None,
    )


def _load_stage2_benchmark_candidate_bytes(
    path: Path,
    payload: bytes,
) -> ReleasedStage2:
    """Load one caller-captured benchmark candidate byte snapshot."""

    return _load_stage2_bundle(
        path,
        None,
        hidden_trust_path=None,
        parity_oracle_trust_path=None,
        bundle_bytes=payload,
    )


def _load_stage2_bundle(
    path: Path,
    plan: Mapping[str, Any] | None,
    *,
    hidden_trust_path: Path | None,
    parity_oracle_trust_path: Path | None,
    bundle_bytes: bytes | None = None,
) -> ReleasedStage2:
    released = plan is not None
    expected_screening_scope_hash: str | None = None
    if plan is not None:
        try:
            validate(plan, "query-plan.schema.json")
            require_valid_approval(plan, "plan_hash")
            expected_screening_scope_hash = assert_screening_scope_hash(plan)
        except (ApprovalError, QueryPlanError, SchemaValidationError) as error:
            raise Stage2ReleaseError(f"Stage 2 requires an exact approved QueryPlan: {error}") from error
    if not path.is_file():
        label = "release" if released else "benchmark candidate"
        raise Stage2ReleaseError(f"Stage 2 {label} artifact is required: {path}")
    bundle_root = path.parent.resolve(strict=True)
    release_bytes = (
        bundle_bytes
        if bundle_bytes is not None
        else _read_bytes(path, "Stage 2 bundle")
    )
    document = _json_object_bytes(release_bytes, "Stage 2 bundle")
    if "thresholds" in document:
        raise Stage2ReleaseError("legacy raw-score thresholds are forbidden in production releases")
    expected_fields = _RELEASE_FIELDS if released else _BENCHMARK_CANDIDATE_FIELDS
    _exact_optional_fields(
        document,
        expected_fields,
        frozenset({"reranker_fallback"}),
        "Stage 2 bundle",
    )
    expected_schema_version = "3" if released else "2"
    if document.get("schema_version") != expected_schema_version:
        raise Stage2ReleaseError(
            f"Stage 2 {'release' if released else 'benchmark candidate'} must use "
            f"schema_version {expected_schema_version}"
        )
    if not released and "reranker_fallback" in document:
        raise Stage2ReleaseError(
            "Stage 2 fallback is injected only during final schema-v3 assembly"
        )
    hidden_trust: HiddenEvaluatorTrust | None = None
    parity_oracle_trust: ParityOracleTrust | None = None
    if released:
        if hidden_trust_path is None:
            raise Stage2ReleaseError(
                "Stage 2 release requires a deployment-controlled hidden evaluator "
                "trust manifest (hidden_trust_path or PAPER_AGENT_STAGE2_HIDDEN_TRUST)"
            )
        hidden_trust = _load_deployment_hidden_trust(
            hidden_trust_path,
            bundle_root=bundle_root,
        )
        if parity_oracle_trust_path is None:
            raise Stage2ReleaseError(
                "Stage 2 release requires a deployment-controlled parity oracle "
                "trust manifest (parity_oracle_trust_path or "
                "PAPER_AGENT_STAGE2_PARITY_ORACLE_TRUST)"
            )
        parity_oracle_trust = _load_deployment_parity_oracle_trust(
            parity_oracle_trust_path,
            bundle_root=bundle_root,
        )
    profile_name = _text(document, "profile")
    if plan is not None and profile_name != plan["filter"]["profile"]:
        raise Stage2ReleaseError("Stage 2 release profile does not match QueryPlan")
    gate_document = _object(document, "release_gate") if released else None

    _, reranker_hash, reranker_bytes = _artifact(path, _object(document, "reranker_lock"))
    _, adjudicator_hash, adjudicator_bytes = _artifact(path, _object(document, "adjudicator_lock"))
    try:
        reranker_lock = ModelLock(**_json_object_bytes(reranker_bytes, "Stage 2 reranker model lock"))
        adjudicator_lock = ModelLock(**_json_object_bytes(adjudicator_bytes, "Stage 2 qwen model lock"))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise Stage2ReleaseError(f"Stage 2 model lock is invalid: {error}") from error
    runtime = _object(document, "runtime")
    release_gate_hash = (
        content_hash(gate_document) if gate_document is not None else None
    )
    base_profile = stage2_base_profile(
        runtime,
        reranker_lock,
        adjudicator_lock,
        reranker_lock_hash=reranker_hash,
        adjudicator_lock_hash=adjudicator_hash,
        release_gate_hash=release_gate_hash,
    )
    if (
        expected_screening_scope_hash is not None
        and base_profile.screening_scope_hash != expected_screening_scope_hash
    ):
        raise Stage2ReleaseError(
            "Stage 2 release screening scope does not match QueryPlan"
        )
    calibration = _object(document, "calibration")
    if set(calibration) != _PATH_NAMES:
        raise Stage2ReleaseError("Stage 2 release must bind reranker and qwen probability calibrations")
    try:
        reranker_calibration = _load_path_calibration(
            path,
            _object(calibration, CalibrationPath.RERANKER.value),
            CalibrationPath.RERANKER,
        )
        adjudicator_calibration = _load_path_calibration(
            path,
            _object(calibration, CalibrationPath.QWEN.value),
            CalibrationPath.QWEN,
        )
        profile = replace(
            base_profile,
            reranker_calibration=reranker_calibration,
            adjudicator_calibration=adjudicator_calibration,
        )
    except (TypeError, ValueError) as error:
        raise Stage2ReleaseError(f"Stage 2 probability calibration is invalid: {error}") from error
    if reranker_calibration.calibrator.model_lock_hash != reranker_hash:
        raise Stage2ReleaseError("reranker calibrator does not match the released model lock")
    if adjudicator_calibration.calibrator.model_lock_hash != adjudicator_hash:
        raise Stage2ReleaseError("qwen calibrator does not match the released model lock")
    if any(
        binding.threshold.stage2_config_hash != base_profile.base_runtime_config_hash
        for binding in (reranker_calibration, adjudicator_calibration)
    ):
        raise Stage2ReleaseError("Stage 2 thresholds do not match the base runtime config")
    if plan is not None:
        if profile.threshold_hash != plan["filter"]["thresholds_hash"]:
            raise Stage2ReleaseError("Stage 2 probability threshold bundle does not match QueryPlan")
    primary_batch_binding: HiddenPromotionBatchBinding | None = None
    if gate_document is not None:
        assert hidden_trust is not None and parity_oracle_trust is not None
        _, primary_batch_binding = _release_gate(
            path,
            gate_document,
            profile_name,
            profile,
            hidden_trust,
            parity_oracle_trust,
        )

    fallback = _load_released_reranker_fallback(
        path,
        document.get("reranker_fallback"),
        primary_profile=profile,
        primary_evaluation_manifest_hash=(
            _sha256_text(gate_document, "evaluation_manifest_hash")
            if gate_document is not None
            else None
        ),
        hidden_trust=hidden_trust,
        parity_oracle_trust=parity_oracle_trust,
        primary_batch_binding=primary_batch_binding,
    )

    released_stage2 = ReleasedStage2(
        profile_name,
        profile,
        sha256(release_bytes).hexdigest(),
        base_profile.omlx_base_url,
        base_profile.api_key_env,
        fallback,
    )
    if (
        plan is not None
        and released_stage2.effective_config_hash != plan["filter"]["config_hash"]
    ):
        raise Stage2ReleaseError(
            "Stage 2 effective release configuration does not match QueryPlan"
        )
    return released_stage2


def _release_gate(
    release_path: Path,
    document: Mapping[str, Any],
    profile_name: str,
    profile: Stage2Profile,
    hidden_trust: HiddenEvaluatorTrust,
    oracle_trust: ParityOracleTrust,
) -> tuple[ReleaseGateResult, HiddenPromotionBatchBinding]:
    _exact_fields(
        document,
        _RELEASE_GATE_FIELDS,
        "Stage 2 release gate",
    )
    candidate_id = _text(document, "candidate_id")
    evaluation_manifest_hash = _sha256_text(document, "evaluation_manifest_hash")
    if candidate_id != profile_name:
        raise Stage2ReleaseError("Stage 2 release gate candidate does not match the profile")

    evidence_path, _, evidence_bytes = _artifact(
        release_path,
        _object(document, "evidence"),
    )
    try:
        index = load_stage2_release_evidence_index_bytes(evidence_path, evidence_bytes)
        gate = verify_stage2_release_evidence_index(
            index,
            candidate_id=candidate_id,
            candidate_bundle_sha256=_sha256_text(
                document, "candidate_bundle_sha256"
            ),
            evaluation_manifest_hash=evaluation_manifest_hash,
            profile=profile,
            hidden_trust=hidden_trust,
            oracle_trust=oracle_trust,
        )
        return gate, _hidden_promotion_batch_binding(index)
    except Stage2ReleaseError:
        raise
    except (OSError, ValueError) as error:
        raise Stage2ReleaseError(
            f"Stage 2 release evidence verification failed: {error}"
        ) from error


def _load_released_reranker_fallback(
    release_path: Path,
    document: Any,
    *,
    primary_profile: Stage2Profile,
    primary_evaluation_manifest_hash: str | None,
    hidden_trust: HiddenEvaluatorTrust | None,
    parity_oracle_trust: ParityOracleTrust | None,
    primary_batch_binding: HiddenPromotionBatchBinding | None = None,
) -> ReleasedRerankerFallback | None:
    if document is None:
        return None
    if not isinstance(document, Mapping):
        raise Stage2ReleaseError("Stage 2 reranker_fallback must be an object")
    _exact_fields(
        document,
        frozenset({"candidate", "release_evidence", "runtime", "release_binding"}),
        "Stage 2 reranker fallback",
    )
    candidate_path, candidate_hash, candidate_bytes = _artifact(
        release_path, _object(document, "candidate")
    )
    backup = _load_stage2_benchmark_candidate_bytes(candidate_path, candidate_bytes)
    if backup.reranker_fallback is not None:
        raise Stage2ReleaseError("Stage 2 reranker fallback candidates cannot nest fallbacks")
    if backup.profile.reranker_lock_hash == primary_profile.reranker_lock_hash:
        raise Stage2ReleaseError("Stage 2 reranker fallback must use a distinct model lock")
    shared_runtime_hash = stage2_shared_runtime_hash(primary_profile)
    if stage2_shared_runtime_hash(backup.profile) != shared_runtime_hash:
        raise Stage2ReleaseError(
            "Stage 2 fallback query, Qwen path, or runtime semantics differ from the primary"
        )
    runtime = _object(document, "runtime")
    _exact_fields(
        runtime,
        frozenset({"omlx_base_url", "api_key_env"}),
        "Stage 2 reranker fallback runtime",
    )
    endpoint = _text(runtime, "omlx_base_url")
    _require_loopback(endpoint)
    api_key_env = runtime["api_key_env"]
    if api_key_env is not None and (
        not isinstance(api_key_env, str) or not api_key_env
    ):
        raise Stage2ReleaseError("Stage 2 fallback api_key_env must be a non-empty string or null")
    binding_document = _object(document, "release_binding")
    _exact_fields(
        binding_document,
        frozenset({
            "backup_candidate_hash",
            "backup_release_evidence_hash",
            "evaluation_manifest_hash",
            "gate_policy_hash",
            "shared_runtime_hash",
        }),
        "Stage 2 reranker fallback release binding",
    )
    binding = FallbackReleaseBinding(
        _sha256_text(binding_document, "backup_candidate_hash"),
        _sha256_text(binding_document, "backup_release_evidence_hash"),
        _sha256_text(binding_document, "evaluation_manifest_hash"),
        _sha256_text(binding_document, "gate_policy_hash"),
        _sha256_text(binding_document, "shared_runtime_hash"),
    )
    if binding.backup_candidate_hash != candidate_hash:
        raise Stage2ReleaseError("Stage 2 fallback candidate does not match its binding")
    if binding.shared_runtime_hash != shared_runtime_hash:
        raise Stage2ReleaseError("Stage 2 fallback shared runtime binding drifted")
    if binding.gate_policy_hash != HIDDEN_PROMOTION_GATE_POLICY_HASH:
        raise Stage2ReleaseError("Stage 2 fallback does not use the frozen promotion gate policy")
    backup_evidence_path, backup_evidence_hash, backup_evidence_bytes = _artifact(
        release_path, _object(document, "release_evidence")
    )
    if backup_evidence_hash != binding.backup_release_evidence_hash:
        raise Stage2ReleaseError("Stage 2 fallback backup release evidence does not match its binding")
    reranker = backup.profile.reranker_calibration
    if reranker is None or backup.profile.reranker_lock_hash is None:
        raise Stage2ReleaseError("Stage 2 fallback candidate is missing reranker calibration")
    try:
        backup_index = load_stage2_release_evidence_index_bytes(
            backup_evidence_path,
            backup_evidence_bytes,
        )
        if backup_index.hidden_attestation is None:
            raise Stage2ReleaseError(
                "Stage 2 fallback requires final release evidence with hidden attestation"
            )
        if backup_index.candidate_bundle_sha256 != candidate_hash:
            raise Stage2ReleaseError(
                "Stage 2 fallback evidence does not bind the backup candidate"
            )
        if backup_index.evaluation_manifest_hash != binding.evaluation_manifest_hash:
            raise Stage2ReleaseError(
                "Stage 2 fallback evidence does not match its evaluation manifest binding"
            )
        _validate_evidence_bindings(
            backup_index,
            candidate_id=backup.profile_name,
            candidate_bundle_sha256=candidate_hash,
            evaluation_manifest_hash=binding.evaluation_manifest_hash,
            profile=backup.profile,
        )
    except Stage2ReleaseError:
        raise
    except (OSError, ValueError) as error:
        raise Stage2ReleaseError(
            f"Stage 2 fallback release evidence verification failed: {error}"
        ) from error
    runtime_config_hash = content_hash(document)
    fallback_model_lock = _model_lock_from_profile_candidate(
        candidate_path, candidate_bytes
    )
    if primary_evaluation_manifest_hash is None:
        return ReleasedRerankerFallback(
            fallback_model_lock,
            backup.profile.reranker_lock_hash,
            reranker,
            endpoint,
            api_key_env,
            binding,
            runtime_config_hash,
        )
    if (
        hidden_trust is None
        or parity_oracle_trust is None
        or primary_batch_binding is None
    ):
        raise Stage2ReleaseError("Stage 2 fallback requires verified primary release evidence")
    if binding.evaluation_manifest_hash != primary_evaluation_manifest_hash:
        raise Stage2ReleaseError("Stage 2 fallback evaluation manifest does not match the primary release")
    backup_batch_binding = _hidden_promotion_batch_binding(backup_index)
    if backup_batch_binding != primary_batch_binding:
        raise Stage2ReleaseError(
            "Stage 2 fallback attestation does not belong to the primary sealed batch"
        )
    try:
        backup_gate = verify_stage2_release_evidence_index(
            backup_index,
            candidate_id=backup.profile_name,
            candidate_bundle_sha256=backup.release_hash,
            evaluation_manifest_hash=binding.evaluation_manifest_hash,
            profile=backup.profile,
            hidden_trust=hidden_trust,
            oracle_trust=parity_oracle_trust,
            expected_release_role="qualified_fallback",
        )
    except (OSError, ValueError) as error:
        raise Stage2ReleaseError(
            f"Stage 2 fallback release evidence verification failed: {error}"
        ) from error
    if not backup_gate.gate.passed:
        raise Stage2ReleaseError("Stage 2 fallback release gates did not pass")
    return ReleasedRerankerFallback(
        fallback_model_lock,
        backup.profile.reranker_lock_hash,
        reranker,
        endpoint,
        api_key_env,
        binding,
        runtime_config_hash,
    )


def _model_lock_from_profile_candidate(candidate_path: Path, candidate_bytes: bytes) -> ModelLock:
    candidate = _json_object_bytes(candidate_bytes, "Stage 2 fallback candidate")
    _, _, lock_bytes = _artifact(candidate_path, _object(candidate, "reranker_lock"))
    try:
        return ModelLock(**_json_object_bytes(lock_bytes, "Stage 2 fallback model lock"))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise Stage2ReleaseError(f"Stage 2 fallback model lock is invalid: {error}") from error


def verify_stage2_release_evidence(
    evidence_path: Path,
    *,
    candidate_id: str,
    candidate_bundle_sha256: str,
    evaluation_manifest_hash: str,
    profile: Stage2Profile,
    hidden_trust_path: Path,
    parity_oracle_trust_path: Path,
) -> ReleaseGateResult:
    """Verify a single evidence-index snapshot for one frozen Stage 2 profile.

    This path-based compatibility entrypoint reads each top-level input once,
    then delegates to the same object-based core used by runtime release loading.
    """

    try:
        resolved_evidence_path = evidence_path.resolve(strict=True)
        evidence_bytes = _read_bytes(
            resolved_evidence_path,
            "Stage 2 release evidence index",
        )
        index = load_stage2_release_evidence_index_bytes(
            resolved_evidence_path,
            evidence_bytes,
        )
        hidden_trust = _load_deployment_hidden_trust(
            hidden_trust_path,
            bundle_root=resolved_evidence_path.parent,
        )
        oracle_trust = _load_deployment_parity_oracle_trust(
            parity_oracle_trust_path,
            bundle_root=resolved_evidence_path.parent,
        )
    except Stage2ReleaseError:
        raise
    except (OSError, ValueError) as error:
        raise Stage2ReleaseError(
            f"Stage 2 release evidence verification failed: {error}"
        ) from error
    return verify_stage2_release_evidence_index(
        index,
        candidate_id=candidate_id,
        candidate_bundle_sha256=candidate_bundle_sha256,
        evaluation_manifest_hash=evaluation_manifest_hash,
        profile=profile,
        hidden_trust=hidden_trust,
        oracle_trust=oracle_trust,
    )


def verify_stage2_release_evidence_index(
    index: Stage2ReleaseEvidenceIndex,
    *,
    candidate_id: str,
    candidate_bundle_sha256: str,
    evaluation_manifest_hash: str,
    profile: Stage2Profile,
    hidden_trust: HiddenEvaluatorTrust,
    oracle_trust: ParityOracleTrust,
    expected_release_role: ReleaseRole = "winner",
) -> ReleaseGateResult:
    """Verify an already captured evidence index and trust manifest."""

    calibrations = (profile.reranker_calibration, profile.adjudicator_calibration)
    assert all(binding is not None for binding in calibrations)
    if any(
        binding.calibrator.gold_manifest_hash != evaluation_manifest_hash
        for binding in calibrations
        if binding is not None
    ):
        raise Stage2ReleaseError(
            "Stage 2 release gate and calibration gold manifest do not match"
        )
    try:
        _validate_evidence_bindings(
            index,
            candidate_id=candidate_id,
            candidate_bundle_sha256=candidate_bundle_sha256,
            evaluation_manifest_hash=evaluation_manifest_hash,
            profile=profile,
        )
        manifest = gold_manifest_from_document(
            index.gold_manifest.read_json(index.bundle_root)
        )
        # Public benchmark evidence is produced by the v2 candidate before a
        # release-gate hash exists, so verify its full profile in that state.
        public_evidence = verify_public_stage2_gates(
            index,
            profile=replace(profile, release_gate_hash=None),
            candidate_bundle_sha256=candidate_bundle_sha256,
            oracle_trust=oracle_trust,
        )
        if not public_evidence.passed:
            failures = [
                f"{name}: {failure}"
                for name, result in public_evidence.gates.items()
                for failure in result.gate.failures
            ]
            raise Stage2ReleaseError(
                "Stage 2 public release gates did not pass: "
                + ("; ".join(failures) or "a recomputed gate failed")
            )
        hidden_bindings = HiddenPromotionBindings(
            candidate_id=candidate_id,
            evaluation_manifest_hash=evaluation_manifest_hash,
            stage2_config_hash=profile.base_runtime_config_hash,
            model_lock_hashes=_profile_model_lock_hashes(profile),
            calibrator_hashes=_profile_calibrator_hashes(profile),
            threshold_hashes=_profile_threshold_hashes(profile),
            hidden_pair_universe_hashes={
                split.value: pair_universe_hash(tuple(
                    pair.pair_id for pair in manifest.pairs if pair.split is split
                ))
                for split in (GoldSplit.HIDDEN_HARD, GoldSplit.HIDDEN_REAL)
            },
            hidden_split_pair_counts={
                split.value: sum(pair.split is split for pair in manifest.pairs)
                for split in (GoldSplit.HIDDEN_HARD, GoldSplit.HIDDEN_REAL)
            },
            public_gate_artifact_hashes={
                name: gate.evidence_hash
                for name, gate in public_evidence.gates.items()
            },
            throughput_runs=public_evidence.throughput_runs,
        )
        if index.hidden_attestation is None:
            raise Stage2ReleaseError("Stage 2 release evidence requires a hidden attestation")
        attestation_document = index.hidden_attestation.read_json(index.bundle_root)
        verify_hidden_promotion_attestation(
            attestation_document,
            hidden_trust,
            expected_bindings=hidden_bindings,
            expected_release_role=expected_release_role,
        )
        artifacts = {
            "promotion": index.hidden_attestation.sha256,
            **{
                name: gate.evidence_hash
                for name, gate in public_evidence.gates.items()
            },
        }
        return phase3_release_gate(
            candidate_id=candidate_id,
            evaluation_manifest_hash=evaluation_manifest_hash,
            artifacts={
                "promotion": (artifacts["promotion"], GateResult(True, ())),
                **{
                    name: (artifacts[name], gate.gate)
                    for name, gate in public_evidence.gates.items()
                },
            },
            throughput_runs=public_evidence.throughput_runs,
        )
    except Stage2ReleaseError:
        raise
    except (OSError, ValueError) as error:
        raise Stage2ReleaseError(
            f"Stage 2 release evidence verification failed: {error}"
        ) from error


def _hidden_promotion_batch_binding(
    index: Stage2ReleaseEvidenceIndex,
) -> HiddenPromotionBatchBinding:
    if index.hidden_attestation is None:
        raise Stage2ReleaseError(
            "Stage 2 release evidence requires a hidden attestation"
        )
    document = index.hidden_attestation.read_json(index.bundle_root)
    if not isinstance(document, Mapping):
        raise Stage2ReleaseError("Stage 2 hidden attestation must be an object")
    return HiddenPromotionBatchBinding.from_attestation(document)


def _load_deployment_hidden_trust(
    path: Path,
    *,
    bundle_root: Path,
) -> HiddenEvaluatorTrust:
    lexical_path = path.absolute()
    try:
        parent_resolved_lexical_path = path.parent.resolve(strict=True) / path.name
        resolved_path = path.resolve(strict=True)
    except OSError as error:
        raise Stage2ReleaseError(
            f"Stage 2 hidden evaluator trust manifest cannot be resolved: {error}"
        ) from error
    if (
        lexical_path.is_relative_to(bundle_root)
        or parent_resolved_lexical_path.is_relative_to(bundle_root)
        or resolved_path.is_relative_to(bundle_root)
    ):
        raise Stage2ReleaseError(
            "Stage 2 hidden evaluator trust manifest must stay outside the release bundle"
        )
    try:
        return load_hidden_evaluator_trust(resolved_path)
    except (OSError, ValueError) as error:
        raise Stage2ReleaseError(
            f"Stage 2 hidden evaluator trust manifest is invalid: {error}"
        ) from error


def _load_deployment_parity_oracle_trust(
    path: Path,
    *,
    bundle_root: Path,
) -> ParityOracleTrust:
    lexical_path = path.absolute()
    try:
        parent_resolved_lexical_path = path.parent.resolve(strict=True) / path.name
        resolved_path = path.resolve(strict=True)
    except OSError as error:
        raise Stage2ReleaseError(
            f"Stage 2 parity oracle trust manifest cannot be resolved: {error}"
        ) from error
    if (
        lexical_path.is_relative_to(bundle_root)
        or parent_resolved_lexical_path.is_relative_to(bundle_root)
        or resolved_path.is_relative_to(bundle_root)
    ):
        raise Stage2ReleaseError(
            "Stage 2 parity oracle trust manifest must stay outside the release bundle"
        )
    try:
        return load_parity_oracle_trust(resolved_path)
    except (OSError, ValueError) as error:
        raise Stage2ReleaseError(
            f"Stage 2 parity oracle trust manifest is invalid: {error}"
        ) from error


def _validate_evidence_bindings(
    index: Stage2ReleaseEvidenceIndex,
    *,
    candidate_id: str,
    candidate_bundle_sha256: str,
    evaluation_manifest_hash: str,
    profile: Stage2Profile,
) -> None:
    expected = {
        "candidate_id": candidate_id,
        "candidate_bundle_sha256": candidate_bundle_sha256,
        "evaluation_manifest_hash": evaluation_manifest_hash,
        "stage2_config_hash": profile.base_runtime_config_hash,
        "model_lock_hashes": _profile_model_lock_hashes(profile),
        "calibrator_hashes": _profile_calibrator_hashes(profile),
        "threshold_hashes": _profile_threshold_hashes(profile),
    }
    for field, value in expected.items():
        if getattr(index, field) != value:
            raise Stage2ReleaseError(
                f"Stage 2 release evidence {field} does not match the release profile"
            )


def _profile_model_lock_hashes(profile: Stage2Profile) -> dict[str, str]:
    return {
        CalibrationPath.RERANKER.value: profile.reranker_lock_hash,
        CalibrationPath.QWEN.value: profile.adjudicator_lock_hash,
    }


def _profile_calibrator_hashes(profile: Stage2Profile) -> dict[str, str]:
    reranker = profile.reranker_calibration
    qwen = profile.adjudicator_calibration
    assert reranker is not None and qwen is not None
    return {
        CalibrationPath.RERANKER.value: reranker.calibrator.hash(),
        CalibrationPath.QWEN.value: qwen.calibrator.hash(),
    }


def _profile_threshold_hashes(profile: Stage2Profile) -> dict[str, str]:
    reranker = profile.reranker_calibration
    qwen = profile.adjudicator_calibration
    assert reranker is not None and qwen is not None
    return {
        CalibrationPath.RERANKER.value: reranker.threshold.hash(),
        CalibrationPath.QWEN.value: qwen.threshold.hash(),
    }


def _artifact(release_path: Path, document: Mapping[str, Any]) -> tuple[Path, str, bytes]:
    _exact_fields(document, frozenset({"path", "sha256"}), "Stage 2 artifact reference")
    relative = Path(_text(document, "path"))
    if relative.is_absolute() or ".." in relative.parts:
        raise Stage2ReleaseError("Stage 2 release artifact paths must stay inside the bundle")
    bundle_root = release_path.parent.resolve()
    path = (bundle_root / relative).resolve()
    if not path.is_relative_to(bundle_root):
        raise Stage2ReleaseError("Stage 2 release artifact paths must stay inside the bundle")
    expected = _sha256_text(document, "sha256")
    if not path.is_file():
        raise Stage2ReleaseError(f"Stage 2 release artifact drifted: {relative}")
    artifact_bytes = _read_bytes(path, f"Stage 2 release artifact {relative}")
    if sha256(artifact_bytes).hexdigest() != expected:
        raise Stage2ReleaseError(f"Stage 2 release artifact drifted: {relative}")
    return path, expected, artifact_bytes


def _load_path_calibration(
    release_path: Path,
    document: Mapping[str, Any],
    expected_path: CalibrationPath,
) -> PathCalibration:
    _exact_fields(
        document,
        frozenset({"calibrator", "threshold"}),
        f"Stage 2 {expected_path.value} calibration",
    )
    _, _, calibrator_bytes = _artifact(release_path, _object(document, "calibrator"))
    _, _, threshold_bytes = _artifact(release_path, _object(document, "threshold"))
    calibrator = PathCalibrator(**_json_object_bytes(calibrator_bytes, "Stage 2 path calibrator"))
    threshold_document = _json_object_bytes(threshold_bytes, "Stage 2 probability threshold artifact")
    threshold = ProbabilityThresholdArtifact(**threshold_document)
    if calibrator.path is not expected_path or threshold.path is not expected_path:
        raise Stage2ReleaseError(f"Stage 2 calibration artifact is not for {expected_path.value}")
    return PathCalibration(calibrator, threshold)


def _validate_locks(
    reranker: ModelLock,
    adjudicator: ModelLock,
) -> None:
    if reranker.backend != "omlx_rerank" or adjudicator.backend != "omlx_chat":
        raise Stage2ReleaseError("Stage 2 production release supports only local oMLX backends")
    if _strict_version(reranker.omlx_version) < (0, 5, 7) or _strict_version(adjudicator.omlx_version) < (0, 5, 7):
        raise Stage2ReleaseError("Stage 2 release requires oMLX 0.5.7 or newer")


def _runtime_revision(lock: ModelLock) -> str:
    return lock.conversion_revision or lock.source_revision


def _require_loopback(base_url: str) -> None:
    parsed = urlparse(base_url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise Stage2ReleaseError("Stage 2 oMLX endpoint must be local; cloud fallback is forbidden")


def _object(document: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = document.get(key)
    if not isinstance(value, dict):
        raise Stage2ReleaseError(f"Stage 2 release {key} must be an object")
    return value


def _read_bytes(path: Path, label: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise Stage2ReleaseError(f"{label} cannot be read: {error}") from error


def _json_object_bytes(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Stage2ReleaseError(f"{label} is not valid JSON: {error}") from error
    if not isinstance(value, dict):
        raise Stage2ReleaseError(f"{label} must be an object")
    return value


def _exact_fields(document: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    if set(document) != expected:
        missing = sorted(expected - set(document))
        extra = sorted(set(document) - expected)
        raise Stage2ReleaseError(f"{label} fields are not exact; missing={missing}, extra={extra}")


def _exact_optional_fields(
    document: Mapping[str, Any],
    required: frozenset[str],
    optional: frozenset[str],
    label: str,
) -> None:
    actual = set(document)
    if not required <= actual or not actual <= required | optional:
        missing = sorted(required - actual)
        extra = sorted(actual - required - optional)
        raise Stage2ReleaseError(f"{label} fields are not exact; missing={missing}, extra={extra}")


def _text(document: Mapping[str, Any], key: str) -> str:
    value = document.get(key)
    if not isinstance(value, str) or not value:
        raise Stage2ReleaseError(f"Stage 2 release {key} must be a non-empty string")
    return value


def _sha256_text(document: Mapping[str, Any], key: str) -> str:
    value = _text(document, key)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise Stage2ReleaseError(f"Stage 2 release {key} must be a lowercase SHA-256")
    return value


def _integer(document: Mapping[str, Any], key: str) -> int:
    value = document.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise Stage2ReleaseError(f"Stage 2 runtime {key} must be an integer")
    return value


def _string_list(document: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = document.get(key)
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise Stage2ReleaseError(f"Stage 2 runtime {key} must be a list of non-empty strings")
    return tuple(value)


def _evaluation_topic_queries(
    document: Mapping[str, Any],
) -> tuple[tuple[str, str, str], ...]:
    value = document.get("evaluation_topic_queries")
    if not isinstance(value, list):
        raise Stage2ReleaseError(
            "Stage 2 runtime evaluation_topic_queries must be an array"
        )
    rows: list[tuple[str, str, str]] = []
    keys: set[tuple[str, str]] = set()
    expected = frozenset({"topic", "language", "query"})
    for row in value:
        if not isinstance(row, dict):
            raise Stage2ReleaseError(
                "Stage 2 runtime evaluation topic query entries must be objects"
            )
        _exact_fields(row, expected, "Stage 2 evaluation topic query")
        topic = _text(row, "topic")
        language = _text(row, "language")
        query = _text(row, "query")
        key = (topic, language)
        if key in keys:
            raise Stage2ReleaseError(
                "Stage 2 runtime evaluation topic queries contain duplicates"
            )
        keys.add(key)
        rows.append((topic, language, query))
    return tuple(sorted(rows))


def _strict_version(value: str) -> tuple[int, int, int]:
    if not isinstance(value, str):
        raise Stage2ReleaseError("Stage 2 model locks require a strict MAJOR.MINOR.PATCH semantic version")
    match = re.fullmatch(r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)", value)
    if match is None:
        raise Stage2ReleaseError("Stage 2 model locks require a strict MAJOR.MINOR.PATCH semantic version")
    major, minor, patch = (int(part) for part in match.groups())
    return major, minor, patch


def _possibly_truncated(title: str, abstract: str | None, keywords: Sequence[str]) -> bool:
    document = f"{title}\n{abstract or ''}\n{' '.join(keywords)}"
    return len(document) // 4 >= 480
