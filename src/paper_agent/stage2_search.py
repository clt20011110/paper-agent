"""Released local Stage 2 runtime used by search and citation rounds."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from hashlib import sha256
import json
from math import isfinite
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse
from uuid import NAMESPACE_URL, uuid5

from .approval import ApprovalError, require_valid_approval
from .canonical import content_hash
from .domain import FilterStatus
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
)
from .stage2_pipeline import (
    PathCalibration,
    Stage2Decision,
    Stage2Paper,
    Stage2Pipeline,
    Stage2Profile,
)
from .storage import Database


RELEASE_ARTIFACT_NAMES = frozenset(
    {"promotion", "structured_replay", "rationale", "parity", "benchmark", "soak"}
)
_RELEASE_FIELDS = frozenset({
    "schema_version", "profile", "reranker_lock", "adjudicator_lock",
    "calibration", "release_gate", "runtime",
})
_RUNTIME_FIELDS = frozenset({
    "query", "query_version", "include_document_types", "exclude_document_types",
    "token_bucket_width", "document_batch_size", "max_in_flight",
    "adjudicator_concurrency", "adjudicator_seed", "max_context_window",
    "omlx_base_url", "api_key_env", "prompt_version", "schema_version",
})
_RELEASE_GATE_FIELDS = frozenset({
    "candidate_id", "evaluation_manifest_hash", "artifacts", "passed", "failures",
    "throughput_runs",
})
_GATE_ARTIFACT_FIELDS = frozenset({
    "schema_version", "gate", "candidate_id", "evaluation_manifest_hash",
    "stage2_config_hash", "model_lock_hashes", "calibrator_hashes",
    "threshold_hashes", "passed", "failures",
})
_PATH_NAMES = frozenset({CalibrationPath.RERANKER.value, CalibrationPath.QWEN.value})


class Stage2ReleaseError(ValueError):
    """A production Stage 2 release is missing, failed, or drifted."""


@dataclass(frozen=True, slots=True)
class ReleasedStage2:
    profile_name: str
    profile: Stage2Profile
    release_hash: str
    omlx_base_url: str
    api_key_env: str | None = None

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
            ),
            self.profile,
        )
        return Stage2SearchScreener(database, pipeline, campaign_id)


@dataclass(slots=True)
class Stage2SearchScreener:
    """Adapt canonical database papers to the existing Stage 2 cascade."""

    database: Database
    pipeline: Stage2Pipeline
    campaign_id: str
    repository: PaperRepository = field(init=False)
    decisions: dict[str, Stage2Decision] = field(default_factory=dict, init=False)
    run_ids: list[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.repository = PaperRepository(self.database)

    def screen(self, paper_ids: Sequence[str]) -> Mapping[str, FilterStatus]:
        ordered_ids = tuple(sorted(set(paper_ids)))
        if not ordered_ids:
            return {}
        papers = tuple(self._paper(paper_id) for paper_id in ordered_ids)
        run_id = f"stage2-{uuid5(NAMESPACE_URL, f'{self.campaign_id}:{len(self.run_ids)}').hex}"
        summary = self.pipeline.run(run_id, papers)
        self.run_ids.append(run_id)
        self.decisions.update((decision.paper_id, decision) for decision in summary.decisions)
        return {decision.paper_id: decision.status for decision in summary.decisions}

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


def load_stage2_release(path: Path, plan: Mapping[str, Any]) -> ReleasedStage2:
    """Load and bind a passed local release bundle to an approved QueryPlan."""
    try:
        validate(plan, "query-plan.schema.json")
        require_valid_approval(plan, "plan_hash")
    except (ApprovalError, SchemaValidationError) as error:
        raise Stage2ReleaseError(f"Stage 2 requires an exact approved QueryPlan: {error}") from error
    if not path.is_file():
        raise Stage2ReleaseError(f"Stage 2 release artifact is required: {path}")
    release_bytes = _read_bytes(path, "Stage 2 release")
    document = _json_object_bytes(release_bytes, "Stage 2 release")
    if "thresholds" in document:
        raise Stage2ReleaseError("legacy raw-score thresholds are forbidden in production releases")
    _exact_fields(document, _RELEASE_FIELDS, "Stage 2 release")
    if document.get("schema_version") != "1":
        raise Stage2ReleaseError("Stage 2 release must use schema_version 1")

    profile_name = _text(document, "profile")
    if profile_name != plan["filter"]["profile"]:
        raise Stage2ReleaseError("Stage 2 release profile does not match QueryPlan")
    gate_document = _object(document, "release_gate")

    _, reranker_hash, reranker_bytes = _artifact(path, _object(document, "reranker_lock"))
    _, adjudicator_hash, adjudicator_bytes = _artifact(path, _object(document, "adjudicator_lock"))
    try:
        reranker_lock = ModelLock(**_json_object_bytes(reranker_bytes, "Stage 2 reranker model lock"))
        adjudicator_lock = ModelLock(**_json_object_bytes(adjudicator_bytes, "Stage 2 qwen model lock"))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise Stage2ReleaseError(f"Stage 2 model lock is invalid: {error}") from error
    _validate_locks(reranker_lock, adjudicator_lock)

    runtime = _object(document, "runtime")
    _exact_fields(runtime, _RUNTIME_FIELDS, "Stage 2 runtime")
    base_url = _text(runtime, "omlx_base_url")
    _require_loopback(base_url)
    api_key_env = runtime.get("api_key_env")
    if api_key_env is not None and (not isinstance(api_key_env, str) or not api_key_env):
        raise Stage2ReleaseError("Stage 2 api_key_env must be a non-empty string or null")
    prompt_version = _text(runtime, "prompt_version")
    schema_version = _text(runtime, "schema_version")
    if prompt_version != "stage2-adjudication-v1" or schema_version != "filter-decision.schema.json":
        raise Stage2ReleaseError("Stage 2 release uses an unsupported prompt or schema version")
    try:
        base_profile = Stage2Profile(
            query=_text(runtime, "query"),
            query_version=_text(runtime, "query_version"),
            thresholds=None,
            reranker_model_id=reranker_lock.model_id,
            reranker_revision=_runtime_revision(reranker_lock),
            adjudicator_model_id=adjudicator_lock.model_id,
            adjudicator_revision=_runtime_revision(adjudicator_lock),
            reranker_lock_hash=reranker_hash,
            adjudicator_lock_hash=adjudicator_hash,
            release_gate_hash=content_hash(gate_document),
            include_document_types=frozenset(_string_list(runtime, "include_document_types")),
            exclude_document_types=frozenset(_string_list(runtime, "exclude_document_types")),
            token_bucket_width=_integer(runtime, "token_bucket_width"),
            document_batch_size=_integer(runtime, "document_batch_size"),
            reranker_max_in_flight=_integer(runtime, "max_in_flight"),
            adjudicator_concurrency=_integer(runtime, "adjudicator_concurrency"),
            adjudicator_seed=_integer(runtime, "adjudicator_seed"),
            adjudicator_max_context_window=_integer(runtime, "max_context_window"),
            omlx_base_url=base_url,
            api_key_env=api_key_env,
            prompt_version=prompt_version,
            schema_version=schema_version,
        )
    except Stage2ReleaseError:
        raise
    except (OSError, ValueError) as error:
        raise Stage2ReleaseError(f"Stage 2 runtime is invalid: {error}") from error
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
    _release_gate(path, gate_document, profile_name, profile)
    if profile.threshold_hash != plan["filter"]["thresholds_hash"]:
        raise Stage2ReleaseError("Stage 2 probability threshold bundle does not match QueryPlan")
    if profile.config_hash != plan["filter"]["config_hash"]:
        raise Stage2ReleaseError("Stage 2 release configuration does not match QueryPlan")

    return ReleasedStage2(
        profile_name,
        profile,
        sha256(release_bytes).hexdigest(),
        base_url,
        api_key_env,
    )


def _release_gate(
    release_path: Path,
    document: Mapping[str, Any],
    profile_name: str,
    profile: Stage2Profile,
) -> ReleaseGateResult:
    _exact_fields(document, _RELEASE_GATE_FIELDS, "Stage 2 release gate")
    artifacts = document.get("artifacts")
    runs = document.get("throughput_runs")
    failures = document.get("failures")
    if not isinstance(artifacts, dict) or set(artifacts) != RELEASE_ARTIFACT_NAMES:
        raise Stage2ReleaseError("Stage 2 release must bind all six Phase 3 gate artifacts")
    if (
        not isinstance(runs, list)
        or len(runs) != 3
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(value)
            or value <= 0
            for value in runs
        )
    ):
        raise Stage2ReleaseError("Stage 2 release needs three throughput runs")
    if not isinstance(failures, list) or not all(isinstance(item, str) for item in failures):
        raise Stage2ReleaseError("Stage 2 release failures must be a list of strings")
    if document.get("passed") is not True or failures:
        raise Stage2ReleaseError("Stage 2 release has not passed every Phase 3 gate")
    candidate_id = _text(document, "candidate_id")
    evaluation_manifest_hash = _sha256_text(document, "evaluation_manifest_hash")
    if candidate_id != profile_name:
        raise Stage2ReleaseError("Stage 2 release gate candidate does not match the profile")
    calibrations = (profile.reranker_calibration, profile.adjudicator_calibration)
    assert all(binding is not None for binding in calibrations)
    if any(
        binding.calibrator.gold_manifest_hash != evaluation_manifest_hash
        for binding in calibrations
        if binding is not None
    ):
        raise Stage2ReleaseError("Stage 2 release gate and calibration gold manifest do not match")

    artifact_hashes: dict[str, str] = {}
    for gate_name in sorted(RELEASE_ARTIFACT_NAMES):
        _, artifact_hash, artifact_bytes = _artifact(
            release_path,
            _object(artifacts, gate_name),
        )
        artifact_document = _json_object_bytes(
            artifact_bytes,
            f"Stage 2 {gate_name} gate artifact",
        )
        _validate_gate_artifact(
            artifact_document,
            gate_name,
            candidate_id,
            evaluation_manifest_hash,
            profile,
        )
        artifact_hashes[gate_name] = artifact_hash
    return ReleaseGateResult(
        candidate_id,
        evaluation_manifest_hash,
        artifact_hashes,
        GateResult(True, ()),
        tuple(float(value) for value in runs),
    )


def _validate_gate_artifact(
    document: Mapping[str, Any],
    gate_name: str,
    candidate_id: str,
    evaluation_manifest_hash: str,
    profile: Stage2Profile,
) -> None:
    _exact_fields(document, _GATE_ARTIFACT_FIELDS, f"Stage 2 {gate_name} gate artifact")
    if document.get("schema_version") != "1" or document.get("gate") != gate_name:
        raise Stage2ReleaseError(f"Stage 2 {gate_name} gate artifact identity is invalid")
    if document.get("candidate_id") != candidate_id:
        raise Stage2ReleaseError(f"Stage 2 {gate_name} gate candidate does not match the release")
    if document.get("evaluation_manifest_hash") != evaluation_manifest_hash:
        raise Stage2ReleaseError(f"Stage 2 {gate_name} gate evaluation manifest does not match the release")
    if document.get("stage2_config_hash") != profile.base_runtime_config_hash:
        raise Stage2ReleaseError(f"Stage 2 {gate_name} gate runtime config does not match the release")
    reranker = profile.reranker_calibration
    qwen = profile.adjudicator_calibration
    assert reranker is not None and qwen is not None
    expected = {
        "model_lock_hashes": {
            CalibrationPath.RERANKER.value: profile.reranker_lock_hash,
            CalibrationPath.QWEN.value: profile.adjudicator_lock_hash,
        },
        "calibrator_hashes": {
            CalibrationPath.RERANKER.value: reranker.calibrator.hash(),
            CalibrationPath.QWEN.value: qwen.calibrator.hash(),
        },
        "threshold_hashes": {
            CalibrationPath.RERANKER.value: reranker.threshold.hash(),
            CalibrationPath.QWEN.value: qwen.threshold.hash(),
        },
    }
    for field, expected_hashes in expected.items():
        hashes = document.get(field)
        if not isinstance(hashes, dict) or set(hashes) != _PATH_NAMES:
            raise Stage2ReleaseError(f"Stage 2 {gate_name} gate {field} is incomplete")
        if hashes != expected_hashes:
            raise Stage2ReleaseError(f"Stage 2 {gate_name} gate {field} does not match the release")
    failures = document.get("failures")
    if document.get("passed") is not True or failures != []:
        raise Stage2ReleaseError(f"Stage 2 {gate_name} gate did not pass")


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
