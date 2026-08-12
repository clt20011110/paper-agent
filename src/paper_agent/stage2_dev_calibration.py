"""Collect and freeze unlabelled Stage 2 DEV raw scores.

This module deliberately does not read human labels or run the cascade.  It
only records the two model outputs that a later, separate calibration step may
join to the authoritative DEV labels.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite
import os
from pathlib import Path
from tempfile import mkstemp
from types import MappingProxyType
from typing import Any, Mapping

from .canonical import content_hash
from .schema import validate
from .stage2_backends import (
    AdjudicationDecision,
    AdjudicationInput,
    OmlxChatBackend,
    OmlxRerankBackend,
    OmlxTransport,
    RerankInput,
    StructuredOutputError,
)
from .stage2_evaluation import CalibrationPath, GoldManifest, GoldSplit
from .stage2_pipeline import ADJUDICATION_SYSTEM_PROMPT, ADJUDICATION_USER_TEMPLATE, Stage2Paper, Stage2Profile
from .stage2_sampling import PrivateCorpusSnapshot


_PATHS = frozenset({CalibrationPath.RERANKER, CalibrationPath.QWEN})


@dataclass(frozen=True, slots=True)
class FrozenDevRawScoreArtifact:
    """Both unlabelled score paths over the exact 300-pair DEV split."""

    version: int
    scores: Mapping[CalibrationPath, Mapping[str, float]]
    model_lock_hashes: Mapping[CalibrationPath, str]
    gold_manifest_hash: str
    dev_manifest_hash: str
    private_snapshot_hash: str
    private_snapshot_corpus_hash: str
    stage2_config_hash: str
    topic_queries: Mapping[tuple[str, str], str]
    qwen_retry_count: int = 0

    def __post_init__(self) -> None:
        if any(
            type(raw_score) not in {int, float}
            for values in self.scores.values()
            for raw_score in values.values()
        ):
            raise ValueError("DEV raw scores must be numeric")
        scores = {
            CalibrationPath(path): MappingProxyType({pair_id: float(raw_score) for pair_id, raw_score in values.items()})
            for path, values in self.scores.items()
        }
        locks = {CalibrationPath(path): value for path, value in self.model_lock_hashes.items()}
        topic_queries = _topic_query_mapping(self.topic_queries)
        object.__setattr__(self, "scores", MappingProxyType(scores))
        object.__setattr__(self, "model_lock_hashes", MappingProxyType(locks))
        object.__setattr__(self, "topic_queries", MappingProxyType(topic_queries))
        if self.version != 1 or set(scores) != _PATHS or set(locks) != _PATHS:
            raise ValueError("DEV raw scores require exactly reranker and qwen paths")
        if not isinstance(self.qwen_retry_count, int) or not 0 <= self.qwen_retry_count <= 300:
            raise ValueError("DEV raw scores require qwen_retry_count in 0..300")
        if not all(_is_sha256(value) for value in (
            self.gold_manifest_hash, self.dev_manifest_hash, self.private_snapshot_hash,
            self.private_snapshot_corpus_hash, self.stage2_config_hash, *locks.values(),
        )):
            raise ValueError("DEV raw scores require SHA-256 provenance")
        expected_ids: set[str] | None = None
        for values in scores.values():
            if len(values) != 300 or any(not _is_pair_id(pair_id) or not isfinite(raw_score) for pair_id, raw_score in values.items()):
                raise ValueError("each DEV raw-score path requires exactly 300 finite pair scores")
            if expected_ids is None:
                expected_ids = set(values)
            elif set(values) != expected_ids:
                raise ValueError("both DEV raw-score paths must cover the same pairs")

    def document(self) -> dict[str, object]:
        return {
            "schema_version": "1",
            "kind": "stage2_dev_raw_scores",
            "gold_manifest_hash": self.gold_manifest_hash,
            "dev_manifest_hash": self.dev_manifest_hash,
            "private_snapshot_hash": self.private_snapshot_hash,
            "private_snapshot_corpus_hash": self.private_snapshot_corpus_hash,
            "stage2_config_hash": self.stage2_config_hash,
            "qwen_retry_count": self.qwen_retry_count,
            "model_lock_hashes": {path.value: self.model_lock_hashes[path] for path in sorted(_PATHS, key=str)},
            "topic_queries": [
                {"topic": topic, "language": language, "query": self.topic_queries[(topic, language)]}
                for topic, language in sorted(self.topic_queries)
            ],
            "scores": {
                path.value: [
                    {"pair_id": pair_id, "raw_score": self.scores[path][pair_id]}
                    for pair_id in sorted(self.scores[path])
                ]
                for path in sorted(_PATHS, key=str)
            },
        }

    def hash(self) -> str:
        return content_hash(self.document())

    @classmethod
    def from_document(cls, document: object) -> "FrozenDevRawScoreArtifact":
        validate(document, "stage2-dev-raw-scores.schema.json")
        if not isinstance(document, dict):
            raise ValueError("DEV raw-score artifact must be an object")
        scores_value = document["scores"]
        locks_value = document["model_lock_hashes"]
        topic_queries_value = document["topic_queries"]
        assert isinstance(scores_value, dict) and isinstance(locks_value, dict) and isinstance(topic_queries_value, list)
        topic_queries: dict[tuple[str, str], str] = {}
        for row in topic_queries_value:
            if not isinstance(row, dict) or set(row) != {"topic", "language", "query"}:
                raise ValueError("DEV topic-query row has unsupported fields")
            key = (row["topic"], row["language"])
            if key in topic_queries:
                raise ValueError("DEV topic-query rows must be unique")
            topic_queries[key] = row["query"]
        scores: dict[CalibrationPath, dict[str, float]] = {}
        for path in _PATHS:
            rows = scores_value[path.value]
            if not isinstance(rows, list):
                raise ValueError("DEV raw-score path must be an array")
            path_scores: dict[str, float] = {}
            for row in rows:
                if not isinstance(row, dict) or set(row) != {"pair_id", "raw_score"}:
                    raise ValueError("DEV raw-score row has unsupported fields")
                pair_id, raw_score = row["pair_id"], row["raw_score"]
                if type(raw_score) not in {int, float} or pair_id in path_scores:
                    raise ValueError("DEV raw-score rows must have unique numeric scores")
                path_scores[pair_id] = float(raw_score)
            scores[path] = path_scores
        return cls(
            1, scores, {CalibrationPath(path): value for path, value in locks_value.items()},
            document["gold_manifest_hash"], document["dev_manifest_hash"], document["private_snapshot_hash"],
            document["private_snapshot_corpus_hash"], document["stage2_config_hash"], topic_queries,
            document["qwen_retry_count"],
        )


def write_frozen_dev_raw_scores(path: Path, artifact: FrozenDevRawScoreArtifact) -> None:
    """Publish one complete artifact without replacing an existing evidence file."""

    document = artifact.document()
    validate(document, "stage2-dev-raw-scores.schema.json")
    if os.path.lexists(path):
        raise FileExistsError(f"refusing to replace DEV raw-score artifact: {path}")
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


def load_frozen_dev_raw_scores(path: Path) -> FrozenDevRawScoreArtifact:
    return FrozenDevRawScoreArtifact.from_document(json.loads(path.read_text(encoding="utf-8")))


@dataclass(frozen=True, slots=True)
class DevScoringCase:
    topic: str
    language: str
    paper: Stage2Paper


def dev_scoring_cases(manifest: GoldManifest, snapshot: PrivateCorpusSnapshot) -> tuple[DevScoringCase, ...]:
    """Return the exact, text-bearing 300-pair DEV universe from frozen inputs."""

    manifest.validate_sampling_structure()
    if manifest.corpus_hash != snapshot.corpus_hash:
        raise ValueError("gold manifest and private snapshot corpus hashes do not match")
    by_key = {paper.key: paper for paper in snapshot.papers}
    cases: list[DevScoringCase] = []
    for pair in manifest.pairs:
        if pair.split is not GoldSplit.DEV:
            continue
        corpus_paper = by_key.get((pair.topic, pair.paper_id))
        if corpus_paper is None:
            raise ValueError("private snapshot does not contain every DEV pair")
        if (
            corpus_paper.language != pair.language
            or corpus_paper.source != pair.source
            or corpus_paper.paper_family != pair.paper_family
        ):
            raise ValueError("private snapshot metadata does not match the DEV manifest")
        keywords = corpus_paper.metadata.get("keywords", ())
        if "keywords" in corpus_paper.metadata and (
            not isinstance(keywords, list)
            or not all(isinstance(item, str) for item in keywords)
        ):
            raise ValueError("private snapshot keywords must be an array of strings")
        cases.append(DevScoringCase(
            pair.topic,
            pair.language,
            Stage2Paper(pair.pair_id, corpus_paper.title, corpus_paper.abstract, tuple(keywords)),
        ))
    if len(cases) != 300 or len({case.paper.paper_id for case in cases}) != 300:
        raise ValueError("gold manifest must resolve exactly 300 unique DEV papers")
    return tuple(cases)


def dev_stage2_papers(manifest: GoldManifest, snapshot: PrivateCorpusSnapshot) -> tuple[Stage2Paper, ...]:
    """Compatibility view of the text-bearing DEV cases without topic metadata."""

    return tuple(case.paper for case in dev_scoring_cases(manifest, snapshot))


@dataclass(slots=True)
class Stage2DevRawScoreRunner:
    """Run both frozen Stage 2 model paths without labels or cascade routing."""

    profile: Stage2Profile
    transport: OmlxTransport
    model_lock_hashes: Mapping[CalibrationPath, str]
    topic_queries: Mapping[tuple[str, str], str]

    def run(
        self,
        manifest: GoldManifest,
        snapshot: PrivateCorpusSnapshot,
        *,
        output_path: Path | None = None,
    ) -> FrozenDevRawScoreArtifact:
        if output_path is not None and os.path.lexists(output_path):
            raise FileExistsError(f"refusing to replace DEV raw-score artifact: {output_path}")
        cases = dev_scoring_cases(manifest, snapshot)
        topic_queries = _validated_topic_queries(cases, self.topic_queries)
        if (
            self.profile.evaluation_topic_queries
            and dict(topic_queries) != self.profile.evaluation_topic_query_map
        ):
            raise ValueError(
                "topic_queries do not match the frozen Stage 2 runtime"
            )
        locks = _validated_locks(self.profile, self.model_lock_hashes)
        schema, schema_hash = _load_schema(self.profile.schema_version)
        frozen_config_hash = self.profile.base_runtime_config_hash
        if self.profile.schema_hash != schema_hash:
            raise ValueError("Stage 2 adjudication schema changed while freezing DEV scoring")
        reranker = OmlxRerankBackend(
            self.profile.reranker_model_id, self.transport,
            self.profile.document_batch_size, self.profile.reranker_max_in_flight,
        )
        reranker_scores = tuple(
            score
            for key in sorted(topic_queries)
            for score in reranker.rerank(
                topic_queries[key],
                tuple(
                    RerankInput(case.paper.paper_id, _paper_text(case.paper))
                    for case in cases if (case.topic, case.language) == key
                ),
            )
        )
        reranker_by_id = {score.paper_id: score.raw_score for score in reranker_scores}
        if set(reranker_by_id) != {case.paper.paper_id for case in cases}:
            raise ValueError("reranker did not return every DEV pair")
        adjudicator = OmlxChatBackend(
            self.profile.adjudicator_model_id,
            self.transport,
            schema,
            self.profile.adjudicator_seed,
            self.profile.adjudicator_max_context_window,
            self.profile.adjudicator_max_output_tokens,
        )
        qwen_requests = tuple(
            _adjudication_input(
                self.profile,
                case.paper,
                topic_queries[(case.topic, case.language)],
            )
            for case in cases
        )
        with ThreadPoolExecutor(max_workers=self.profile.adjudicator_concurrency) as executor:
            first_attempts = tuple(
                executor.map(
                    lambda request: _adjudication_attempt(adjudicator, request),
                    qwen_requests,
                )
            )
        # oMLX 0.5.7 can occasionally stall a grammar-constrained request when
        # several generations share a continuous batch.  Retry only failed
        # requests after the batch drains, one at a time; an immediate retry in
        # the same batch deterministically repeats the truncation.
        decisions: list[AdjudicationDecision] = []
        qwen_retry_count = 0
        for request, (decision, error) in zip(qwen_requests, first_attempts, strict=True):
            if error is None:
                assert decision is not None
                decisions.append(decision)
                continue
            qwen_retry_count += 1
            decisions.append(adjudicator.adjudicate(request))
        qwen_by_id = {decision.paper_id: decision.score for decision in decisions}
        if set(qwen_by_id) != {case.paper.paper_id for case in cases}:
            raise ValueError("Qwen did not return every DEV pair")
        if (
            self.profile.base_runtime_config_hash != frozen_config_hash
            or self.profile.schema_hash != schema_hash
        ):
            raise ValueError("Stage 2 profile or schema changed during DEV scoring")
        artifact = FrozenDevRawScoreArtifact(
            1,
            {CalibrationPath.RERANKER: reranker_by_id, CalibrationPath.QWEN: qwen_by_id},
            locks, manifest.hash(), manifest.dev_hash(), snapshot.hash(), snapshot.corpus_hash,
            frozen_config_hash, topic_queries, qwen_retry_count,
        )
        if output_path is not None:
            write_frozen_dev_raw_scores(output_path, artifact)
        return artifact

def _adjudication_attempt(
    adjudicator: OmlxChatBackend,
    request: AdjudicationInput,
) -> tuple[AdjudicationDecision | None, Exception | None]:
    try:
        return adjudicator.adjudicate(request), None
    except (StructuredOutputError, TimeoutError, OSError) as error:
        return None, error


def _adjudication_input(profile: Stage2Profile, paper: Stage2Paper, query: str) -> AdjudicationInput:
    return AdjudicationInput(paper.paper_id, (
        {"role": "system", "content": ADJUDICATION_SYSTEM_PROMPT},
        {"role": "user", "content": ADJUDICATION_USER_TEMPLATE.format(
            query_version=profile.query_version, query=query, paper_id=paper.paper_id,
            document=_paper_text(paper),
        )},
    ))


def _paper_text(paper: Stage2Paper) -> str:
    return f"Title: {paper.title}\nAbstract: {paper.abstract or ''}\nKeywords: {', '.join(paper.keywords)}"


def _load_schema(name: str) -> tuple[Mapping[str, Any], str]:
    from .schema import schema_directory
    payload = (schema_directory() / name).read_bytes()
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("Stage 2 adjudication schema must be an object")
    return value, sha256(payload).hexdigest()


def _validated_locks(profile: Stage2Profile, values: Mapping[CalibrationPath, str]) -> Mapping[CalibrationPath, str]:
    locks = {CalibrationPath(path): value for path, value in values.items()}
    if set(locks) != _PATHS or not all(_is_sha256(value) for value in locks.values()):
        raise ValueError("DEV raw-score runner requires two SHA-256 model lock hashes")
    expected = {
        CalibrationPath.RERANKER: profile.reranker_lock_hash,
        CalibrationPath.QWEN: profile.adjudicator_lock_hash,
    }
    if not all(_is_sha256(expected[path]) for path in _PATHS):
        raise ValueError("frozen Stage 2 profile requires two SHA-256 model lock hashes")
    if any(locks[path] != expected[path] for path in _PATHS):
        raise ValueError("DEV raw-score model locks do not match the frozen Stage 2 profile")
    return locks


def _validated_topic_queries(
    cases: tuple[DevScoringCase, ...],
    values: Mapping[tuple[str, str], str],
) -> Mapping[tuple[str, str], str]:
    topic_queries = _topic_query_mapping(values)
    expected = {(case.topic, case.language) for case in cases}
    if set(topic_queries) != expected:
        raise ValueError("topic_queries must exactly cover DEV topic-language combinations")
    return MappingProxyType(topic_queries)


def _topic_query_mapping(values: Mapping[tuple[str, str], str]) -> dict[tuple[str, str], str]:
    topic_queries: dict[tuple[str, str], str] = {}
    for key, query in values.items():
        if (
            not isinstance(key, tuple)
            or len(key) != 2
            or not all(isinstance(value, str) and value for value in key)
            or not isinstance(query, str)
            or not query.strip()
        ):
            raise ValueError("topic_queries require non-empty (topic, language) keys and queries")
        topic_queries[key] = query
    if not topic_queries:
        raise ValueError("topic_queries cannot be empty")
    return topic_queries


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _is_pair_id(value: object) -> bool:
    return isinstance(value, str) and value.startswith("pair-") and _is_sha256(value[5:])
