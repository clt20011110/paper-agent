"""Create private three-run Stage 2 hidden-promotion submissions.

This module deliberately accepts no labels.  It resolves the frozen hidden
papers from the private corpus snapshot, uses the candidate's frozen
``(topic, language)`` query mapping, and records only model predictions for
the sealed evaluator.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import mkstemp
from typing import Any, Mapping

from .canonical import content_hash
from .stage2_backends import (
    AdjudicationDecision,
    AdjudicationInput,
    OmlxChatBackend,
    OmlxRerankBackend,
    OmlxTransport,
    RerankBatchError,
    RerankInput,
    Stage2BackendError,
    StructuredOutputError,
)
from .stage2_evaluation import (
    CalibrationPath,
    GoldManifest,
    GoldSplit,
    Prediction,
    PromotionSubmission,
    ReviewReason,
    Stage2Decision,
)
from .stage2_pipeline import Stage2Paper, Stage2Profile
from .stage2_prompt_contract import adjudication_messages, render_stage2_document
from .stage2_promotion_artifacts import promotion_submission_document
from .stage2_sampling import PrivateCorpusSnapshot
from .schema import validate
from .stage2_search import ReleasedStage2


@dataclass(frozen=True, slots=True)
class HiddenSubmissionCase:
    """One text-bearing hidden pair and its candidate-bound query."""

    topic: str
    language: str
    query: str
    paper: Stage2Paper


def hidden_submission_cases(
    manifest: GoldManifest,
    snapshot: PrivateCorpusSnapshot,
    profile: Stage2Profile,
) -> tuple[HiddenSubmissionCase, ...]:
    """Resolve the exact 150 hard + 150 real pair universe without labels."""

    manifest.validate_sampling_structure()
    if manifest.corpus_hash != snapshot.corpus_hash:
        raise ValueError("gold manifest and private snapshot corpus hashes do not match")
    by_key = {paper.key: paper for paper in snapshot.papers}
    cases: list[HiddenSubmissionCase] = []
    for pair in manifest.pairs:
        if pair.split is GoldSplit.DEV:
            continue
        source = by_key.get((pair.topic, pair.paper_id))
        if source is None:
            raise ValueError("private snapshot does not contain every hidden pair")
        if (
            source.language != pair.language
            or source.source != pair.source
            or source.paper_family != pair.paper_family
        ):
            raise ValueError("private snapshot metadata does not match the hidden manifest")
        keywords = source.metadata.get("keywords", ())
        if "keywords" in source.metadata and (
            not isinstance(keywords, list)
            or not all(isinstance(item, str) for item in keywords)
        ):
            raise ValueError("private snapshot keywords must be an array of strings")
        document_type = _document_type(source.metadata)
        normalized_type = (document_type or "").strip().casefold()
        if normalized_type in profile.include_document_types or normalized_type in profile.exclude_document_types:
            raise ValueError(
                "hidden promotion submission cannot encode a deterministic "
                "document-type outcome as a calibrated Prediction"
            )
        cases.append(HiddenSubmissionCase(
            pair.topic,
            pair.language,
            profile.evaluation_query(pair.topic, pair.language),
            Stage2Paper(
                pair.pair_id,
                source.title,
                source.abstract,
                tuple(keywords),
                document_type=document_type,
                possibly_truncated=(
                    pair.abstract_incomplete
                    or _possibly_truncated(source.title, source.abstract, tuple(keywords))
                ),
            ),
        ))
    if len(cases) != 300 or len({case.paper.paper_id for case in cases}) != 300:
        raise ValueError("gold manifest must resolve exactly 300 unique hidden papers")
    return tuple(sorted(cases, key=lambda case: case.paper.paper_id))


@dataclass(slots=True)
class HiddenPromotionSubmissionRunner:
    """Run the frozen candidate cascade three times over the hidden universe."""

    candidate: ReleasedStage2
    transport: OmlxTransport

    def run(
        self,
        manifest: GoldManifest,
        snapshot: PrivateCorpusSnapshot,
        *,
        output_path: Path | None = None,
    ) -> PromotionSubmission:
        if output_path is not None and os.path.lexists(output_path):
            raise FileExistsError("refusing to replace hidden promotion submission")
        profile = self.candidate.profile
        profile.assert_runtime_ready()
        if not profile.production_calibrated:
            raise ValueError("hidden promotion requires a calibrated schema-v2 candidate")
        cases = hidden_submission_cases(manifest, snapshot, profile)
        frozen_config_hash = profile.base_runtime_config_hash
        # The installed package path is not necessarily the source tree; use the
        # same resolver as Stage2Profile for the actual schema bytes.
        from .schema import schema_directory

        schema_payload = (schema_directory() / profile.schema_version).read_bytes()
        schema = json.loads(schema_payload)
        schema_hash = sha256(schema_payload).hexdigest()
        if profile.schema_hash != schema_hash or not isinstance(schema, dict):
            raise ValueError("Stage 2 adjudication schema changed before hidden submission")
        runs = tuple(self._run_once(cases, manifest, profile, schema, run_number) for run_number in range(3))
        if (
            profile.base_runtime_config_hash != frozen_config_hash
            or profile.schema_hash != schema_hash
        ):
            raise ValueError("Stage 2 profile or schema changed during hidden submission")
        submission = PromotionSubmission(self.candidate.profile_name, runs)
        if output_path is not None:
            write_hidden_promotion_submission(output_path, submission)
        return submission

    def _run_once(
        self,
        cases: tuple[HiddenSubmissionCase, ...],
        manifest: GoldManifest,
        profile: Stage2Profile,
        schema: Mapping[str, Any],
        run_number: int,
    ) -> tuple[Prediction, ...]:
        reranker = OmlxRerankBackend(
            profile.reranker_model_id,
            self.transport,
            profile.document_batch_size,
            profile.reranker_max_in_flight,
        )
        by_id = {case.paper.paper_id: case for case in cases}
        reranker_scores: dict[str, float] = {}
        failures: dict[str, ReviewReason] = {}
        for query in sorted({case.query for case in cases}):
            group = tuple(case for case in cases if case.query == query)
            try:
                scores = reranker.rerank(query, tuple(
                    RerankInput(case.paper.paper_id, render_stage2_document(case.paper))
                    for case in group
                ))
            except RerankBatchError as error:
                reranker_scores.update({item.paper_id: item.raw_score for item in error.scores})
                failures.update({paper_id: ReviewReason.SERVICE_ERROR for paper_id in error.failed_paper_ids})
            except _TECHNICAL_FAILURES as error:
                reason = _technical_reason(error)
                failures.update({case.paper.paper_id: reason for case in group})
            else:
                returned = {item.paper_id: item.raw_score for item in scores}
                if set(returned) != {case.paper.paper_id for case in group}:
                    failures.update({case.paper.paper_id: ReviewReason.SERVICE_ERROR for case in group})
                else:
                    reranker_scores.update(returned)

        predictions: dict[str, Prediction] = {}
        qwen_cases: list[tuple[HiddenSubmissionCase, float, float]] = []
        reranker_binding = profile.reranker_calibration
        qwen_binding = profile.adjudicator_calibration
        assert reranker_binding is not None and qwen_binding is not None
        for case in cases:
            paper_id = case.paper.paper_id
            if paper_id in failures or paper_id not in reranker_scores:
                predictions[paper_id] = _technical_prediction(
                    paper_id, self.candidate.profile_name, CalibrationPath.RERANKER,
                    reranker_binding, manifest, profile, run_number,
                    failures.get(paper_id, ReviewReason.SERVICE_ERROR), case.query,
                )
                continue
            score = reranker_scores[paper_id]
            probability = reranker_binding.calibrator.predict(score)
            if _forces_adjudication(case.paper):
                qwen_cases.append((case, score, probability))
                continue
            threshold = reranker_binding.threshold
            if probability <= threshold.low:
                decision = Stage2Decision.IRRELEVANT
            elif probability >= threshold.high:
                decision = Stage2Decision.RELEVANT
            else:
                qwen_cases.append((case, score, probability))
                continue
            predictions[paper_id] = _prediction(
                paper_id, self.candidate.profile_name, decision, score, probability,
                CalibrationPath.RERANKER, reranker_binding, manifest, profile, run_number,
                case.query,
            )

        adjudicator = OmlxChatBackend(
            profile.adjudicator_model_id,
            self.transport,
            schema,
            profile.adjudicator_seed,
            profile.adjudicator_max_context_window,
            profile.adjudicator_max_output_tokens,
        )
        with ThreadPoolExecutor(max_workers=profile.adjudicator_concurrency) as executor:
            first_attempts = tuple(executor.map(
                lambda item: _adjudication_attempt(
                    adjudicator,
                    _adjudication_request(profile, item[0]),
                ),
                qwen_cases,
            ))
        results: list[tuple[str, Prediction]] = []
        for item, (response, error) in zip(qwen_cases, first_attempts, strict=True):
            case, _reranker_score, _reranker_probability = item
            if error is not None:
                try:
                    response = adjudicator.adjudicate(
                        _adjudication_request(profile, case)
                    )
                except _TECHNICAL_FAILURES as retry_error:
                    results.append((
                        case.paper.paper_id,
                        _technical_prediction(
                            case.paper.paper_id,
                            self.candidate.profile_name,
                            CalibrationPath.QWEN,
                            qwen_binding,
                            manifest,
                            profile,
                            run_number,
                            _technical_reason(retry_error),
                            case.query,
                        ),
                    ))
                    continue
            assert response is not None
            results.append((
                case.paper.paper_id,
                _qwen_prediction(
                    response,
                    case,
                    self.candidate.profile_name,
                    qwen_binding,
                    manifest,
                    profile,
                    run_number,
                ),
            ))
        predictions.update(results)
        if set(predictions) != set(by_id):
            raise ValueError("hidden submission did not produce every prediction")
        return tuple(predictions[case.paper.paper_id] for case in cases)

def write_hidden_promotion_submission(path: Path, submission: PromotionSubmission) -> None:
    """Publish a complete private submission without replacing evidence."""

    if os.path.lexists(path):
        raise FileExistsError("refusing to replace hidden promotion submission")
    document = promotion_submission_document(submission)
    validate(document, "stage2-promotion-submission.schema.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2).encode() + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)

def _prediction(
    pair_id: str, candidate_id: str, decision: Stage2Decision, raw_score: float | None,
    probability: float, path: CalibrationPath, binding: Any, manifest: GoldManifest,
    profile: Stage2Profile, run_number: int, query: str,
    review_reason: ReviewReason | None = None,
) -> Prediction:
    return Prediction(
        pair_id, candidate_id, decision, raw_score, probability, path,
        binding.calibrator.hash(), binding.threshold.hash(), binding.calibrator.model_lock_hash,
        manifest.hash(), profile.base_runtime_config_hash,
        _inference_hash(candidate_id, run_number, pair_id, query, decision, raw_score, probability, path, review_reason),
        review_reason,
    )


def _technical_prediction(
    pair_id: str, candidate_id: str, path: CalibrationPath, binding: Any,
    manifest: GoldManifest, profile: Stage2Profile, run_number: int,
    reason: ReviewReason, query: str,
) -> Prediction:
    return _prediction(
        pair_id, candidate_id, Stage2Decision.NEEDS_REVIEW, None, 0.5, path,
        binding, manifest, profile, run_number, query, reason,
    )


def _inference_hash(
    candidate_id: str, run_number: int, pair_id: str, query: str,
    decision: Stage2Decision, raw_score: float | None, probability: float,
    path: CalibrationPath, review_reason: ReviewReason | None,
) -> str:
    return content_hash({
        "candidate_id": candidate_id, "run": run_number, "pair_id": pair_id,
        "query": query, "decision": decision.value, "raw_score": raw_score,
        "probability": probability, "path": path.value,
        "review_reason": review_reason.value if review_reason else None,
    })


def _technical_reason(error: Exception) -> ReviewReason:
    if isinstance(error, TimeoutError):
        return ReviewReason.TIMEOUT
    if isinstance(error, StructuredOutputError):
        return ReviewReason.SCHEMA_ERROR
    return ReviewReason.SERVICE_ERROR


_TECHNICAL_FAILURES = (
    Stage2BackendError,
    TimeoutError,
    OSError,
)


def _adjudication_request(
    profile: Stage2Profile,
    case: HiddenSubmissionCase,
) -> AdjudicationInput:
    return AdjudicationInput(
        case.paper.paper_id,
        adjudication_messages(
            query_version=profile.query_version,
            query=case.query,
            paper=case.paper,
        ),
    )


def _adjudication_attempt(
    adjudicator: OmlxChatBackend,
    request: AdjudicationInput,
) -> tuple[AdjudicationDecision | None, Exception | None]:
    try:
        return adjudicator.adjudicate(request), None
    except _TECHNICAL_FAILURES as error:
        return None, error


def _qwen_prediction(
    response: AdjudicationDecision,
    case: HiddenSubmissionCase,
    candidate_id: str,
    binding: Any,
    manifest: GoldManifest,
    profile: Stage2Profile,
    run_number: int,
) -> Prediction:
    probability = binding.calibrator.predict(response.score)
    if probability <= binding.threshold.low:
        calibrated = Stage2Decision.IRRELEVANT
    elif probability >= binding.threshold.high:
        calibrated = Stage2Decision.RELEVANT
    else:
        calibrated = Stage2Decision.NEEDS_REVIEW
    structured = Stage2Decision(response.decision)
    if structured is not calibrated:
        decision, review_reason = Stage2Decision.NEEDS_REVIEW, ReviewReason.MODEL_CONFLICT
    elif calibrated is Stage2Decision.NEEDS_REVIEW:
        decision, review_reason = calibrated, ReviewReason.UNCERTAIN
    else:
        decision, review_reason = calibrated, None
    return _prediction(
        case.paper.paper_id,
        candidate_id,
        decision,
        response.score,
        probability,
        CalibrationPath.QWEN,
        binding,
        manifest,
        profile,
        run_number,
        case.query,
        review_reason,
    )


def _forces_adjudication(paper: Stage2Paper) -> bool:
    return any((
        not bool(paper.abstract and paper.abstract.strip()), paper.possibly_truncated,
        paper.multi_condition_conflict, paper.language_anomaly,
    ))


def _document_type(metadata: Mapping[str, Any]) -> str | None:
    for key in ("document_type", "publication_type", "type"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().casefold()
    return None


def _possibly_truncated(title: str, abstract: str | None, keywords: tuple[str, ...]) -> bool:
    return len(f"{title}\n{abstract or ''}\n{' '.join(keywords)}") // 4 >= 480
