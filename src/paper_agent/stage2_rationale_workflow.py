"""Produce and verify the human Stage 2 rationale-audit evidence chain."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import ctypes
import ctypes.util
from dataclasses import dataclass, replace
import errno
from hashlib import sha256
import json
from math import isclose, isfinite
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from typing import Any, Mapping, Sequence

from .canonical import content_hash
from .schema import schema_directory, validate
from .stage2_backends import (
    AdjudicationDecision,
    AdjudicationInput,
    OmlxChatBackend,
    OmlxRerankBackend,
    OmlxTransport,
    RerankInput,
    StructuredOutputError,
)
from .stage2_benchmark_inputs import benchmark_corpus_hash, benchmark_papers_from_document
from .stage2_evaluation import (
    RationaleAuditCase,
    RationaleAuditManifest,
    RationaleAuditRecord,
    RationaleStratum,
)
from .stage2_pipeline import Stage2Paper
from .stage2_prompt_contract import adjudication_messages, render_stage2_document


EVIDENCE_SUPPORT_RUBRIC = {
    "version": 1,
    "question": "Is every material claim in the rationale supported by the frozen evidence?",
    "true_when": (
        "Every material topic, method, result, population, and relevance claim in the "
        "rationale is directly supported by the supplied frozen evidence."
    ),
    "false_when": (
        "Any material claim is unsupported, contradicted, or cannot be checked from the "
        "supplied frozen evidence."
    ),
    "instruction": "Judge only the frozen evidence; do not fill gaps from outside knowledge.",
}

SEVERE_FABRICATION_RUBRIC = {
    "version": 1,
    "question": "Does the rationale contain a severe fabrication?",
    "true_when": (
        "It invents or contradicts a material study fact, method, result, population, "
        "comparison, or conclusion that could change the relevance judgment."
    ),
    "false_when": (
        "No material invented or contradictory claim is present. Minor wording imprecision "
        "that cannot change the relevance judgment is not severe fabrication."
    ),
    "instruction": "Judge only the frozen evidence; do not fill gaps from outside knowledge.",
}

EVIDENCE_SUPPORT_RUBRIC_HASH = content_hash(EVIDENCE_SUPPORT_RUBRIC)
SEVERE_FABRICATION_RUBRIC_HASH = content_hash(SEVERE_FABRICATION_RUBRIC)


_SOURCE_LEDGER_FIELDS = frozenset({
    "schema_version", "kind", "candidate", "benchmark_papers_sha256",
    "corpus_hash", "query_metadata_sha256", "records",
})
_SOURCE_CANDIDATE_FIELDS = frozenset({
    "candidate_id", "bundle_sha256", "release_hash", "adjudicator_model_id",
    "adjudicator_model_lock_hash", "prompt_version", "response_schema",
})
_QUERY_METADATA_FIELDS = frozenset({
    "schema_version", "kind", "candidate_id", "candidate_bundle_sha256",
    "benchmark_papers_sha256", "primary_languages", "scores", "assignments",
})
_QUERY_SCORE_FIELDS = frozenset({
    "pair_id", "source_paper_id", "language", "topic", "query_version",
    "query", "stratum", "reranker_raw_score", "reranker_probability",
})
_LEDGER_RECORD_FIELDS = frozenset({
    "pair_id", "decision", "score", "rationale", "evidence_fields",
})
_DERIVED_DOCUMENT_FIELDS = frozenset({
    "schema_version", "kind", "corpus_hash", "model_lock_hash",
    "candidate_bundle_sha256", "source_ledger_sha256", "query_metadata_sha256",
    "examples",
})


@dataclass(frozen=True, slots=True)
class RationaleSourcePlan:
    """Model-free summary of the immutable rationale source run."""

    candidate_id: str
    paper_count: int
    topic_query_count: int
    reranker_pair_count: int
    qwen_pair_count: int
    primary_languages: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RationaleSourceArtifacts:
    """Machine-produced query assignments and their typed Qwen outcomes."""

    query_metadata: Mapping[str, Any]
    source_ledger: Mapping[str, Any]


def rationale_source_plan(
    candidate: Any,
    *,
    benchmark_papers_document: object,
) -> RationaleSourcePlan:
    """Validate source inputs without calling either local model."""

    validate(benchmark_papers_document, "stage2-benchmark-papers.schema.json")
    papers = benchmark_papers_from_document(benchmark_papers_document)
    if len({paper.paper_id for paper in papers}) != len(papers):
        raise ValueError("benchmark papers must have unique paper_id values")
    queries = candidate.profile.evaluation_topic_queries
    languages = tuple(sorted({language for _, language, _ in queries}))
    if not queries or len(languages) < 2:
        raise ValueError(
            "Stage 2 candidate needs frozen topic queries in at least two languages"
        )
    if candidate.profile.reranker_calibration is None:
        raise ValueError("Stage 2 candidate needs a released reranker calibration")
    return RationaleSourcePlan(
        candidate.profile_name,
        len(papers),
        len(queries),
        len(papers) * len(queries),
        50 * len(languages),
        languages,
    )


def collect_rationale_source_artifacts(
    candidate: Any,
    *,
    candidate_bundle_sha256: str,
    benchmark_papers_document: object,
    benchmark_papers_sha256: str,
    transport: OmlxTransport,
) -> RationaleSourceArtifacts:
    """Run the candidate's local BGE and Qwen paths and freeze their outputs."""

    plan = rationale_source_plan(
        candidate, benchmark_papers_document=benchmark_papers_document
    )
    papers = benchmark_papers_from_document(benchmark_papers_document)
    profile = candidate.profile
    calibration = profile.reranker_calibration
    assert calibration is not None
    reranker = OmlxRerankBackend(
        profile.reranker_model_id,
        transport,
        document_batch_size=profile.document_batch_size,
        max_in_flight=profile.reranker_max_in_flight,
    )
    documents = tuple(
        RerankInput(paper.paper_id, render_stage2_document(paper))
        for paper in papers
    )
    scored: list[dict[str, Any]] = []
    for topic, language, query in profile.evaluation_topic_queries:
        for score in reranker.rerank(query, documents):
            probability = calibration.calibrator.predict(score.raw_score)
            if not isfinite(score.raw_score) or not isfinite(probability):
                raise ValueError("rationale source reranker scores must be finite")
            if probability >= calibration.threshold.high:
                stratum = RationaleStratum.RELEVANT.value
            elif calibration.threshold.low < probability < calibration.threshold.high:
                stratum = RationaleStratum.BOUNDARY.value
            else:
                stratum = "irrelevant"
            pair_id = _rationale_pair_id(
                score.paper_id,
                topic=topic,
                language=language,
                query_version=profile.query_version,
                query=query,
            )
            scored.append({
                "pair_id": pair_id,
                "source_paper_id": score.paper_id,
                "language": language,
                "topic": topic,
                "query_version": profile.query_version,
                "query": query,
                "stratum": stratum,
                "reranker_raw_score": score.raw_score,
                "reranker_probability": probability,
            })
    if len(scored) != plan.reranker_pair_count or len({row["pair_id"] for row in scored}) != len(scored):
        raise ValueError("rationale source reranker must score every topic-query/paper pair once")
    scored = sorted(scored, key=lambda row: row["pair_id"])
    assignments = _select_rationale_assignments(
        scored,
        plan.primary_languages,
        boundary_midpoint=(calibration.threshold.low + calibration.threshold.high) / 2,
    )
    query_metadata = {
        "schema_version": "3",
        "kind": "stage2_rationale_query_metadata",
        "candidate_id": candidate.profile_name,
        "candidate_bundle_sha256": candidate_bundle_sha256,
        "benchmark_papers_sha256": benchmark_papers_sha256,
        "primary_languages": list(plan.primary_languages),
        "scores": scored,
        "assignments": assignments,
    }
    validate(query_metadata, "stage2-rationale-query-metadata.schema.json")
    query_metadata_sha256 = sha256(_json_bytes(query_metadata)).hexdigest()

    response_schema = json.loads(
        (schema_directory() / profile.schema_version).read_text(encoding="utf-8")
    )
    adjudicator = OmlxChatBackend(
        profile.adjudicator_model_id,
        transport,
        response_schema,
        seed=profile.adjudicator_seed,
        max_context_window=profile.adjudicator_max_context_window,
        max_output_tokens=profile.adjudicator_max_output_tokens,
    )
    papers_by_id = {paper.paper_id: paper for paper in papers}
    requests = tuple(
        _rationale_adjudication_input(assignment, papers_by_id)
        for assignment in assignments
    )
    with ThreadPoolExecutor(max_workers=profile.adjudicator_concurrency) as executor:
        first_attempts = tuple(
            executor.map(
                lambda request: _rationale_adjudication_attempt(adjudicator, request),
                requests,
            )
        )
    decisions: list[AdjudicationDecision] = []
    for request, (decision, error) in zip(requests, first_attempts, strict=True):
        if error is None:
            assert decision is not None
            decisions.append(decision)
        else:
            decisions.append(adjudicator.adjudicate(request))
    source_ledger = qwen_adjudication_ledger_document(
        decisions,
        candidate=candidate,
        candidate_bundle_sha256=candidate_bundle_sha256,
        benchmark_papers_document=benchmark_papers_document,
        benchmark_papers_sha256=benchmark_papers_sha256,
        query_metadata_document=query_metadata,
        query_metadata_sha256=query_metadata_sha256,
    )
    return RationaleSourceArtifacts(query_metadata, source_ledger)


def write_rationale_source_artifacts_no_replace(
    artifacts: RationaleSourceArtifacts,
    *,
    output_directory: Path,
) -> tuple[Path, Path]:
    """Atomically publish the two-file source bundle without replacement."""

    validate(artifacts.query_metadata, "stage2-rationale-query-metadata.schema.json")
    validate(artifacts.source_ledger, "stage2-rationale-source-ledger.schema.json")
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix=f".{output_directory.name}.", dir=output_directory.parent
    ) as temporary:
        temporary_directory = Path(temporary)
        _write_json_no_replace(
            temporary_directory / "query-metadata.json", artifacts.query_metadata
        )
        _write_json_no_replace(
            temporary_directory / "source-ledger.json", artifacts.source_ledger
        )
        _rename_directory_no_replace(temporary_directory, output_directory)
    return (
        output_directory / "query-metadata.json",
        output_directory / "source-ledger.json",
    )


def _rename_directory_no_replace(source: Path, target: Path) -> None:
    if sys.platform == "darwin":
        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        rename_no_replace = libc.renameatx_np
        rename_no_replace.argtypes = (
            ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint,
        )
        rename_no_replace.restype = ctypes.c_int
        result = rename_no_replace(
            -2,
            ctypes.c_char_p(bytes(source)),
            -2,
            ctypes.c_char_p(bytes(target)),
            0x00000004,
        )
    elif sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        rename_no_replace = libc.renameat2
        rename_no_replace.argtypes = (
            ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint,
        )
        rename_no_replace.restype = ctypes.c_int
        result = rename_no_replace(
            -100,
            ctypes.c_char_p(bytes(source)),
            -100,
            ctypes.c_char_p(bytes(target)),
            1,
        )
    else:
        raise OSError(errno.ENOTSUP, "atomic no-replace directory publish is unsupported")
    if result:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(
                f"rationale source output already exists: {target}"
            )
        raise OSError(error, f"cannot publish rationale source output: {target}")


def _select_rationale_assignments(
    scored: Sequence[Mapping[str, Any]],
    primary_languages: Sequence[str],
    *,
    boundary_midpoint: float,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for language in primary_languages:
        for stratum in RationaleStratum:
            candidates = [
                row for row in scored
                if row["language"] == language and row["stratum"] == stratum.value
            ]
            candidates.sort(
                key=(
                    (lambda row: (-row["reranker_probability"], row["pair_id"]))
                    if stratum is RationaleStratum.RELEVANT
                    else (
                        lambda row: (
                            abs(row["reranker_probability"] - boundary_midpoint),
                            row["pair_id"],
                        )
                    )
                )
            )
            if len(candidates) < 25:
                raise ValueError(
                    f"rationale source needs at least 25 {stratum.value} "
                    f"pairs for primary language {language}"
                )
            selected.extend(dict(row) for row in candidates[:25])
    return sorted(selected, key=lambda row: row["pair_id"])


def _rationale_pair_id(
    source_paper_id: str,
    *,
    topic: str,
    language: str,
    query_version: str,
    query: str,
) -> str:
    return content_hash({
        "kind": "stage2-rationale-pair-v1",
        "source_paper_id": source_paper_id,
        "topic": topic,
        "language": language,
        "query_version": query_version,
        "query": query,
    })


def _rationale_adjudication_input(
    assignment: Mapping[str, Any],
    papers_by_id: Mapping[str, Stage2Paper],
) -> AdjudicationInput:
    paper = replace(
        papers_by_id[assignment["source_paper_id"]],
        paper_id=assignment["pair_id"],
    )
    return AdjudicationInput(
        paper.paper_id,
        adjudication_messages(
            query_version=assignment["query_version"],
            query=assignment["query"],
            paper=paper,
        ),
    )


def _rationale_adjudication_attempt(
    adjudicator: OmlxChatBackend,
    request: AdjudicationInput,
) -> tuple[AdjudicationDecision | None, Exception | None]:
    try:
        return adjudicator.adjudicate(request), None
    except (StructuredOutputError, TimeoutError, OSError) as error:
        return None, error


def _json_bytes(document: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def derive_rationale_audit_examples(
    source_ledger: object,
    *,
    source_ledger_sha256: str,
    candidate: Any,
    candidate_bundle_sha256: str,
    benchmark_papers_document: object,
    benchmark_papers_sha256: str,
    query_metadata: object,
    query_metadata_sha256: str,
) -> dict[str, Any]:
    """Deterministically select review cases from a bound Qwen response ledger.

    The ledger is deliberately a strict, text-bearing artifact: the model's
    rationale is copied verbatim, while the reviewer evidence is rendered only
    from the separately frozen benchmark paper fields named by the ledger.
    """

    validate(source_ledger, "stage2-rationale-source-ledger.schema.json")
    validate(query_metadata, "stage2-rationale-query-metadata.schema.json")
    validate(benchmark_papers_document, "stage2-benchmark-papers.schema.json")
    if not isinstance(source_ledger, Mapping) or not isinstance(query_metadata, Mapping):
        raise ValueError("rationale source inputs must be JSON objects")
    if set(source_ledger) != _SOURCE_LEDGER_FIELDS:
        raise ValueError("rationale source ledger has an unsupported shape")
    if set(query_metadata) != _QUERY_METADATA_FIELDS:
        raise ValueError("rationale query metadata has an unsupported shape")
    if source_ledger["benchmark_papers_sha256"] != benchmark_papers_sha256:
        raise ValueError("rationale source ledger does not bind the benchmark papers bytes")
    if source_ledger["query_metadata_sha256"] != query_metadata_sha256:
        raise ValueError("rationale source ledger does not bind the query metadata bytes")
    if query_metadata["benchmark_papers_sha256"] != benchmark_papers_sha256:
        raise ValueError("rationale query metadata does not bind the benchmark papers bytes")
    if (
        query_metadata["candidate_id"] != candidate.profile_name
        or query_metadata["candidate_bundle_sha256"] != candidate_bundle_sha256
    ):
        raise ValueError("rationale query metadata does not bind the Stage 2 candidate")

    papers = benchmark_papers_from_document(benchmark_papers_document)
    corpus_hash = benchmark_corpus_hash(papers)
    if source_ledger["corpus_hash"] != corpus_hash:
        raise ValueError("rationale source ledger corpus hash does not match benchmark papers")
    _validate_candidate_binding(source_ledger["candidate"], candidate, candidate_bundle_sha256)

    papers_by_id = {paper.paper_id: paper for paper in papers}
    if len(papers_by_id) != len(papers):
        raise ValueError("benchmark papers must have unique paper_id values")
    assignments = _query_assignments(
        query_metadata,
        papers_by_id,
        candidate=candidate,
    )
    source_records = _ledger_records(source_ledger, assignments)
    examples: list[dict[str, Any]] = []
    for language in query_metadata["primary_languages"]:
        for stratum in (RationaleStratum.RELEVANT, RationaleStratum.BOUNDARY):
            candidates = sorted(
                (
                    record for record in source_records
                    if assignments[record["pair_id"]]["stratum"] == stratum.value
                    and assignments[record["pair_id"]]["language"] == language
                ),
                key=lambda record: (record["pair_id"], record["rationale"]),
            )
            if len(candidates) < 25:
                raise ValueError(
                    f"rationale source ledger needs at least 25 {stratum.value} "
                    f"records for primary language {language}"
                )
            for record in candidates[:25]:
                assignment = assignments[record["pair_id"]]
                paper = papers_by_id[assignment["source_paper_id"]]
                examples.append({
                    "pair_id": record["pair_id"],
                    "stratum": stratum.value,
                    "language": language,
                    "rationale_artifact_hash": source_ledger_sha256,
                    "evidence": _render_evidence(paper, record["evidence_fields"]),
                    "rationale": record["rationale"],
                })

    document = {
        "schema_version": "2",
        "kind": "stage2_rationale_audit_derived_examples",
        "corpus_hash": corpus_hash,
        "model_lock_hash": source_ledger["candidate"]["adjudicator_model_lock_hash"],
        "candidate_bundle_sha256": candidate_bundle_sha256,
        "source_ledger_sha256": source_ledger_sha256,
        "query_metadata_sha256": query_metadata_sha256,
        "examples": examples,
    }
    validate(document, "stage2-rationale-derived-examples.schema.json")
    return document


def qwen_adjudication_ledger_document(
    decisions: Sequence[Any],
    *,
    candidate: Any,
    candidate_bundle_sha256: str,
    benchmark_papers_document: object,
    benchmark_papers_sha256: str,
    query_metadata_document: object,
    query_metadata_sha256: str,
) -> dict[str, Any]:
    """Freeze actual typed ``AdjudicationDecision`` outputs as a source ledger.

    This is the production boundary for rationale text.  It accepts the typed
    decisions emitted by ``OmlxChatBackend`` rather than author-supplied
    rationale dictionaries.
    """

    validate(benchmark_papers_document, "stage2-benchmark-papers.schema.json")
    validate(query_metadata_document, "stage2-rationale-query-metadata.schema.json")
    papers = benchmark_papers_from_document(benchmark_papers_document)
    if not isinstance(query_metadata_document, Mapping):
        raise ValueError("rationale query metadata must be an object")
    assignment_ids = {
        assignment["pair_id"] for assignment in query_metadata_document["assignments"]
    }
    profile = candidate.profile
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for decision in decisions:
        if not isinstance(decision, AdjudicationDecision):
            raise ValueError("rationale ledger requires complete typed Qwen decisions")
        pair_id = decision.paper_id
        if (
            pair_id not in assignment_ids
            or pair_id in seen
        ):
            raise ValueError("rationale ledger requires complete typed Qwen decisions")
        seen.add(pair_id)
        records.append({
            "pair_id": pair_id,
            "decision": decision.decision,
            "score": decision.score,
            "rationale": decision.rationale,
            "evidence_fields": list(decision.evidence_fields),
        })
    if seen != assignment_ids:
        raise ValueError("rationale ledger must cover every generated query assignment")
    document = {
        "schema_version": "2",
        "kind": "stage2_qwen_adjudication_ledger",
        "candidate": {
            "candidate_id": candidate.profile_name,
            "bundle_sha256": candidate_bundle_sha256,
            "release_hash": candidate.release_hash,
            "adjudicator_model_id": profile.adjudicator_model_id,
            "adjudicator_model_lock_hash": profile.adjudicator_lock_hash,
            "prompt_version": profile.prompt_version,
            "response_schema": profile.schema_version,
        },
        "benchmark_papers_sha256": benchmark_papers_sha256,
        "corpus_hash": benchmark_corpus_hash(papers),
        "query_metadata_sha256": query_metadata_sha256,
        "records": records,
    }
    validate(document, "stage2-rationale-source-ledger.schema.json")
    return document


def write_qwen_adjudication_ledger_no_replace(path: Path, document: Mapping[str, Any]) -> None:
    """Publish typed Qwen outcomes as the immutable rationale source artifact."""

    validate(document, "stage2-rationale-source-ledger.schema.json")
    _write_json_no_replace(path, document)


def write_derived_rationale_examples_no_replace(path: Path, document: Mapping[str, Any]) -> None:
    """Write a schema-validated deterministic rationale source without replacement."""

    validate(document, "stage2-rationale-derived-examples.schema.json")
    _write_json_no_replace(path, document)


def _validate_candidate_binding(
    binding: object, candidate: Any, candidate_bundle_sha256: str
) -> None:
    if not isinstance(binding, Mapping) or set(binding) != _SOURCE_CANDIDATE_FIELDS:
        raise ValueError("rationale source ledger candidate binding has an unsupported shape")
    profile = candidate.profile
    expected = {
        "candidate_id": candidate.profile_name,
        "bundle_sha256": candidate_bundle_sha256,
        "release_hash": candidate.release_hash,
        "adjudicator_model_id": profile.adjudicator_model_id,
        "adjudicator_model_lock_hash": profile.adjudicator_lock_hash,
        "prompt_version": profile.prompt_version,
        "response_schema": profile.schema_version,
    }
    if binding != expected:
        raise ValueError("rationale source ledger is not bound to the frozen Stage 2 candidate")


def _query_assignments(
    metadata: Mapping[str, Any],
    papers_by_id: Mapping[str, Any],
    *,
    candidate: Any,
) -> Mapping[str, Mapping[str, Any]]:
    scores = metadata["scores"]
    assignments = metadata["assignments"]
    if not isinstance(scores, list) or not isinstance(assignments, list):
        raise ValueError("rationale query metadata scores and assignments must be lists")
    scored_by_pair: dict[str, Mapping[str, Any]] = {}
    profile = candidate.profile
    expected_queries = profile.evaluation_topic_query_map
    calibration = profile.reranker_calibration
    if calibration is None:
        raise ValueError("Stage 2 candidate needs a released reranker calibration")
    expected_pairs = {
        _rationale_pair_id(
            paper_id,
            topic=topic,
            language=language,
            query_version=profile.query_version,
            query=query,
        )
        for topic, language, query in profile.evaluation_topic_queries
        for paper_id in papers_by_id
    }
    for score in scores:
        if not isinstance(score, Mapping) or set(score) != _QUERY_SCORE_FIELDS:
            raise ValueError("rationale query score has an unsupported shape")
        pair_id = score["pair_id"]
        source_paper_id = score["source_paper_id"]
        key = (score["topic"], score["language"])
        expected_pair_id = _rationale_pair_id(
            source_paper_id,
            topic=score["topic"],
            language=score["language"],
            query_version=score["query_version"],
            query=score["query"],
        )
        if (
            pair_id in scored_by_pair
            or source_paper_id not in papers_by_id
            or expected_queries.get(key) != score["query"]
            or score["query_version"] != profile.query_version
            or pair_id != expected_pair_id
        ):
            raise ValueError("rationale query score is not derived from the candidate")
        probability = calibration.calibrator.predict(score["reranker_raw_score"])
        if not isfinite(score["reranker_raw_score"]) or not isfinite(probability):
            raise ValueError("rationale query score must be finite")
        if not isclose(probability, score["reranker_probability"], abs_tol=1e-12):
            raise ValueError("rationale query score probability is not reproducible")
        expected_stratum = (
            RationaleStratum.RELEVANT.value
            if probability >= calibration.threshold.high
            else RationaleStratum.BOUNDARY.value
            if calibration.threshold.low < probability < calibration.threshold.high
            else "irrelevant"
        )
        if score["stratum"] != expected_stratum:
            raise ValueError("rationale query score stratum is not reproducible")
        scored_by_pair[pair_id] = score
    primary_languages = metadata["primary_languages"]
    expected_languages = sorted({language for _, language in expected_queries})
    if primary_languages != expected_languages or len(primary_languages) < 2:
        raise ValueError("rationale query metadata needs at least two primary languages")
    if set(scored_by_pair) != expected_pairs:
        raise ValueError("rationale query metadata must cover every topic-query/paper pair")
    expected_assignments = _select_rationale_assignments(
        tuple(scored_by_pair.values()),
        primary_languages,
        boundary_midpoint=(calibration.threshold.low + calibration.threshold.high) / 2,
    )
    if assignments != expected_assignments:
        raise ValueError("rationale query assignments are not the deterministic score selection")
    by_pair = {assignment["pair_id"]: assignment for assignment in assignments}
    return by_pair


def _ledger_records(
    ledger: Mapping[str, Any],
    assignments: Mapping[str, Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    records = ledger["records"]
    if not isinstance(records, list):
        raise ValueError("rationale source ledger records must be a list")
    by_pair: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping) or set(record) != _LEDGER_RECORD_FIELDS:
            raise ValueError("rationale source ledger record has an unsupported shape")
        pair_id = record["pair_id"]
        if pair_id in by_pair or pair_id not in assignments:
            raise ValueError("rationale source ledger records need one bound query assignment")
        by_pair[pair_id] = record
    if set(by_pair) != set(assignments):
        raise ValueError("rationale source ledger must cover every query assignment")
    return tuple(by_pair.values())


def _render_evidence(paper: Any, evidence_fields: object) -> str:
    if not isinstance(evidence_fields, list) or not evidence_fields:
        raise ValueError("rationale source ledger evidence_fields must be a non-empty list")
    fields = tuple(evidence_fields)
    if len(set(fields)) != len(fields) or any(field not in {"title", "abstract", "keywords"} for field in fields):
        raise ValueError("rationale source ledger evidence_fields are invalid")
    parts: list[str] = []
    for field in fields:
        if field == "title":
            if not paper.title:
                raise ValueError("rationale source ledger requested an empty title")
            parts.append(f"title: {paper.title}")
        elif field == "abstract":
            if not paper.abstract or not paper.abstract.strip():
                raise ValueError("rationale source ledger requested a missing abstract")
            parts.append(f"abstract: {paper.abstract}")
        else:
            if not paper.keywords:
                raise ValueError("rationale source ledger requested missing keywords")
            parts.append("keywords: " + ", ".join(paper.keywords))
    return "\n".join(parts)


@dataclass(frozen=True, slots=True)
class RationaleAuditExample:
    """One pre-stratified model rationale and the evidence a reviewer may use."""

    pair_id: str
    stratum: RationaleStratum
    language: str
    rationale_artifact_hash: str
    evidence: str
    rationale: str

    def __post_init__(self) -> None:
        if not isinstance(self.stratum, RationaleStratum):
            object.__setattr__(self, "stratum", RationaleStratum(self.stratum))
        if not all((self.pair_id, self.language, self.rationale_artifact_hash, self.evidence, self.rationale)):
            raise ValueError("rationale audit examples require frozen identity, evidence, and rationale")

    def case(self) -> RationaleAuditCase:
        return RationaleAuditCase(
            self.pair_id, self.stratum, self.language, self.rationale_artifact_hash
        )

    def worklist_row(self) -> dict[str, Any]:
        row = {
            "pair_id": self.pair_id,
            "stratum": self.stratum.value,
            "language": self.language,
            "rationale_artifact_hash": self.rationale_artifact_hash,
            "evidence": self.evidence,
            "rationale": self.rationale,
            "evidence_supported": None,
            "severe_fabrication": None,
        }
        row["content_hash"] = _worklist_row_content_hash(row)
        return row


@dataclass(frozen=True, slots=True)
class FrozenRationaleAudit:
    """A manifest plus the editable, deliberately unlabelled human worklist."""

    manifest: RationaleAuditManifest
    worklist: Mapping[str, Any]


def rationale_audit_examples_from_document(
    document: object,
    *,
    require_derived: bool = False,
) -> tuple[tuple[RationaleAuditExample, ...], str, str]:
    """Load the explicit, already-selected examples used to freeze an audit."""

    if isinstance(document, Mapping) and document.get("kind") == "stage2_rationale_audit_derived_examples":
        validate(document, "stage2-rationale-derived-examples.schema.json")
        if set(document) != _DERIVED_DOCUMENT_FIELDS or document.get("schema_version") != "2":
            raise ValueError("rationale audit derived examples have an unsupported shape")
        examples_document = document
    elif require_derived:
        raise ValueError("production rationale audits require schema-v2 derived examples")
    else:
        examples_document = document
    if not isinstance(examples_document, Mapping) or set(examples_document) not in ({
        "schema_version", "kind", "corpus_hash", "model_lock_hash", "examples",
    }, _DERIVED_DOCUMENT_FIELDS):
        raise ValueError("rationale audit examples have an unsupported shape")
    if (
        (examples_document["schema_version"] == "1" and examples_document["kind"] != "stage2_rationale_audit_examples")
        or (examples_document["schema_version"] == "2" and examples_document["kind"] != "stage2_rationale_audit_derived_examples")
        or examples_document["schema_version"] not in {"1", "2"}
        or not isinstance(examples_document["examples"], list)
    ):
        raise ValueError("not a Stage 2 rationale audit examples document")
    examples: list[RationaleAuditExample] = []
    required = {
        "pair_id", "stratum", "language", "rationale_artifact_hash",
        "evidence", "rationale",
    }
    for row in examples_document["examples"]:
        if not isinstance(row, Mapping) or set(row) != required:
            raise ValueError("rationale audit example has an unsupported shape")
        if (
            examples_document["schema_version"] == "2"
            and row["rationale_artifact_hash"] != examples_document["source_ledger_sha256"]
        ):
            raise ValueError("derived rationale example is not bound to its source ledger")
        examples.append(RationaleAuditExample(
            pair_id=row["pair_id"],
            stratum=RationaleStratum(row["stratum"]),
            language=row["language"],
            rationale_artifact_hash=row["rationale_artifact_hash"],
            evidence=row["evidence"],
            rationale=row["rationale"],
        ))
    return tuple(examples), examples_document["corpus_hash"], examples_document["model_lock_hash"]


def rationale_audit_manifest_from_document(document: object) -> RationaleAuditManifest:
    """Load one schema-validated frozen rationale audit manifest."""

    validate(document, "stage2-rationale-audit-manifest.schema.json")
    if not isinstance(document, Mapping):
        raise ValueError("rationale audit manifest must be an object")
    return RationaleAuditManifest(
        version=document["version"],
        cases=tuple(
            RationaleAuditCase(row[0], RationaleStratum(row[1]), row[2], row[3])
            for row in document["cases"]
        ),
        corpus_hash=document["corpus_hash"],
        model_lock_hash=document["model_lock_hash"],
        evidence_rubric_hash=document["evidence_rubric_hash"],
        fabrication_rubric_hash=document["fabrication_rubric_hash"],
    )


def load_rationale_audit_manifest(path: Path) -> RationaleAuditManifest:
    return rationale_audit_manifest_from_document(json.loads(path.read_text(encoding="utf-8")))


def load_rationale_worklist(path: Path) -> Mapping[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise ValueError("rationale audit worklist must be an object")
    return document


def freeze_rationale_audit(
    examples: Sequence[RationaleAuditExample],
    *,
    corpus_hash: str,
    model_lock_hash: str,
    reviewer_id: str,
) -> FrozenRationaleAudit:
    """Freeze the supplied stratified examples before any human labels exist."""

    if not reviewer_id.strip():
        raise ValueError("rationale audit reviewer_id is required")
    manifest = RationaleAuditManifest(
        version=1,
        cases=tuple(example.case() for example in examples),
        corpus_hash=corpus_hash,
        model_lock_hash=model_lock_hash,
        evidence_rubric_hash=EVIDENCE_SUPPORT_RUBRIC_HASH,
        fabrication_rubric_hash=SEVERE_FABRICATION_RUBRIC_HASH,
    )
    validate(manifest.document(), "stage2-rationale-audit-manifest.schema.json")
    worklist = {
        "schema_version": "1",
        "kind": "stage2_human_rationale_audit_worklist",
        "manifest_hash": manifest.hash(),
        "reviewer_id": reviewer_id,
        "evidence_support_rubric_hash": EVIDENCE_SUPPORT_RUBRIC_HASH,
        "severe_fabrication_rubric_hash": SEVERE_FABRICATION_RUBRIC_HASH,
        "evidence_support_rubric": EVIDENCE_SUPPORT_RUBRIC,
        "severe_fabrication_rubric": SEVERE_FABRICATION_RUBRIC,
        "rows": [example.worklist_row() for example in examples],
    }
    return FrozenRationaleAudit(manifest, worklist)


def import_completed_rationale_audit(
    worklist: Mapping[str, Any], *, manifest: RationaleAuditManifest
) -> tuple[RationaleAuditRecord, ...]:
    """Import explicit human labels; blank or non-boolean labels are rejected."""

    if set(worklist) != {
        "schema_version", "kind", "manifest_hash", "reviewer_id",
        "evidence_support_rubric_hash", "severe_fabrication_rubric_hash",
        "evidence_support_rubric", "severe_fabrication_rubric", "rows",
    } or worklist.get("schema_version") != "1" or worklist.get("kind") != "stage2_human_rationale_audit_worklist":
        raise ValueError("not a Stage 2 human rationale audit worklist")
    if not isinstance(worklist.get("reviewer_id"), str) or not worklist["reviewer_id"].strip():
        raise ValueError("rationale audit worklist reviewer_id is required")
    if worklist.get("manifest_hash") != manifest.hash():
        raise ValueError("rationale audit worklist does not bind the frozen manifest")
    if (
        worklist.get("evidence_support_rubric_hash") != manifest.evidence_rubric_hash
        or worklist.get("severe_fabrication_rubric_hash") != manifest.fabrication_rubric_hash
        or worklist.get("evidence_support_rubric") != EVIDENCE_SUPPORT_RUBRIC
        or worklist.get("severe_fabrication_rubric") != SEVERE_FABRICATION_RUBRIC
    ):
        raise ValueError("rationale audit worklist rubrics do not match the frozen manifest")
    rows = worklist.get("rows")
    if not isinstance(rows, list):
        raise ValueError("rationale audit worklist rows must be a list")
    expected_cases = {case.pair_id: case for case in manifest.cases}
    if {row.get("pair_id") for row in rows if isinstance(row, Mapping)} != set(expected_cases) or len(rows) != len(expected_cases):
        raise ValueError("rationale audit worklist must exactly cover the frozen manifest")
    records = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("rationale audit worklist rows must be objects")
        case = expected_cases[row["pair_id"]]
        frozen_fields = ("stratum", "language", "rationale_artifact_hash")
        expected_values = (case.stratum.value, case.language, case.rationale_artifact_hash)
        if tuple(row.get(field) for field in frozen_fields) != expected_values:
            raise ValueError("rationale audit worklist row changed frozen provenance")
        if row.get("content_hash") != _worklist_row_content_hash(row):
            raise ValueError("rationale audit worklist row content drifted")
        evidence_supported = row.get("evidence_supported")
        severe_fabrication = row.get("severe_fabrication")
        if type(evidence_supported) is not bool or type(severe_fabrication) is not bool:
            raise ValueError("rationale audit worklist has unfilled human labels")
        records.append(RationaleAuditRecord(
            row["pair_id"], manifest.hash(), evidence_supported, severe_fabrication
        ))
    return tuple(records)


def rationale_audit_records_document(
    records: Sequence[RationaleAuditRecord],
    *,
    worklist_sha256: str,
) -> dict[str, Any]:
    """Return the existing ``stage2-rationale-audit-records`` schema shape."""

    document = {
        "schema_version": "1",
        "kind": "stage2_rationale_audit_records",
        "worklist_sha256": worklist_sha256,
        "records": [record.document() for record in records],
    }
    validate(document, "stage2-rationale-audit-records.schema.json")
    return document


def write_rationale_audit_artifacts(
    frozen: FrozenRationaleAudit,
    records: Sequence[RationaleAuditRecord],
    *,
    manifest_path: Path,
    records_path: Path,
    worklist_sha256: str,
) -> None:
    """Publish no-replace artifacts, with records visible before the manifest."""

    if manifest_path == records_path:
        raise ValueError("rationale audit manifest and records paths must differ")
    if manifest_path.exists() or records_path.exists():
        raise FileExistsError("rationale audit output already exists")
    manifest_document = frozen.manifest.document()
    validate(manifest_document, "stage2-rationale-audit-manifest.schema.json")
    records_document = rationale_audit_records_document(
        records, worklist_sha256=worklist_sha256
    )
    _write_json_no_replace(records_path, records_document)
    _write_json_no_replace(manifest_path, manifest_document)


def write_frozen_rationale_audit(
    frozen: FrozenRationaleAudit,
    *,
    manifest_path: Path,
    worklist_path: Path,
) -> None:
    """Publish a human worklist first and its completion-marker manifest last."""

    if manifest_path.absolute() == worklist_path.absolute():
        raise ValueError("rationale audit manifest and worklist paths must differ")
    if manifest_path.exists() or worklist_path.exists():
        raise FileExistsError("rationale audit output already exists")
    manifest_document = frozen.manifest.document()
    validate(manifest_document, "stage2-rationale-audit-manifest.schema.json")
    _write_json_no_replace(worklist_path, frozen.worklist)
    _write_json_no_replace(manifest_path, manifest_document)


def write_rationale_records_no_replace(
    path: Path,
    records: Sequence[RationaleAuditRecord],
    *,
    worklist_sha256: str,
) -> None:
    """Publish completed human audit records without replacing prior evidence."""

    _write_json_no_replace(
        path, rationale_audit_records_document(records, worklist_sha256=worklist_sha256)
    )


def write_rationale_worklist_no_replace(path: Path, worklist: Mapping[str, Any]) -> None:
    """Publish an editable human worklist without replacing an earlier copy."""

    _write_json_no_replace(path, worklist)


def _worklist_row_content_hash(row: Mapping[str, Any]) -> str:
    return content_hash({
        field: row[field]
        for field in (
            "pair_id", "stratum", "language", "rationale_artifact_hash", "evidence", "rationale"
        )
    })


def _write_json_no_replace(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(_json_bytes(document))
