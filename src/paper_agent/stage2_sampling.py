"""Deterministic construction of the private Stage 2 sampling corpus.

The public :class:`~paper_agent.stage2_evaluation.GoldManifest` deliberately
does not contain paper text, annotations, or private sampling strata.  This
module keeps those inputs in a separate private snapshot and emits the public
manifest together with a small binding artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log
from random import Random
from types import MappingProxyType
from typing import Any, Mapping

from .canonical import content_hash
from .stage2_evaluation import GoldLabelStore, GoldManifest, GoldPair, GoldSplit, make_pair_id


def _pair_key(topic: str, paper_id: str) -> tuple[str, str]:
    if not topic or not paper_id:
        raise ValueError("topic and paper_id are required")
    return topic, paper_id


@dataclass(frozen=True, slots=True)
class CorpusPaper:
    """One private topic-paper candidate from a frozen crawler population."""

    topic: str
    paper_id: str
    title: str
    abstract: str | None
    metadata: Mapping[str, Any]
    source: str
    language: str
    paper_family: str
    sampling_weight: float
    sampling_probability: float
    abstract_incomplete: bool = False
    natural_crawler_population: bool = True
    cross_language_match: bool = False

    def __post_init__(self) -> None:
        _pair_key(self.topic, self.paper_id)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        if not all((self.title, self.source, self.language, self.paper_family)):
            raise ValueError("corpus paper text and sampling fields are required")
        if self.abstract is not None and not isinstance(self.abstract, str):
            raise ValueError("abstract must be a string or null")
        if not isfinite(self.sampling_weight) or self.sampling_weight <= 0:
            raise ValueError("sampling_weight must be finite and positive")
        if not isfinite(self.sampling_probability) or not 0 < self.sampling_probability <= 1:
            raise ValueError("sampling_probability must be finite and in (0, 1]")

    @property
    def key(self) -> tuple[str, str]:
        return _pair_key(self.topic, self.paper_id)

    @property
    def pair_id(self) -> str:
        return make_pair_id(self.topic, self.paper_id)

    def document(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "paper_id": self.paper_id,
            "title": self.title,
            "abstract": self.abstract,
            "metadata": dict(self.metadata),
            "source": self.source,
            "language": self.language,
            "paper_family": self.paper_family,
            "sampling_weight": self.sampling_weight,
            "sampling_probability": self.sampling_probability,
            "abstract_incomplete": self.abstract_incomplete,
            "natural_crawler_population": self.natural_crawler_population,
            "cross_language_match": self.cross_language_match,
        }


@dataclass(frozen=True, slots=True)
class PrivateCorpusSnapshot:
    """Private input contract binding crawl text, policy version, and seed."""

    schema_version: int
    sampling_policy_version: str
    sampling_seed: int
    papers: tuple[CorpusPaper, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "papers", tuple(self.papers))
        if self.schema_version != 1 or not self.sampling_policy_version or not self.papers:
            raise ValueError("private corpus snapshot requires version, policy, and papers")
        if len({paper.key for paper in self.papers}) != len(self.papers):
            raise ValueError("private corpus snapshot has duplicate topic-paper rows")

    @property
    def corpus_hash(self) -> str:
        return content_hash({
            "schema_version": self.schema_version,
            "papers": [paper.document() for paper in sorted(self.papers, key=lambda item: item.key)],
        })

    def document(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sampling_policy_version": self.sampling_policy_version,
            "sampling_seed": self.sampling_seed,
            "corpus_hash": self.corpus_hash,
            "papers": [paper.document() for paper in sorted(self.papers, key=lambda item: item.key)],
        }

    def hash(self) -> str:
        return content_hash(self.document())


def private_corpus_snapshot_from_document(document: object) -> PrivateCorpusSnapshot:
    """Parse the exact private snapshot document and verify its corpus binding."""

    if not isinstance(document, dict) or set(document) != {
        "schema_version", "sampling_policy_version", "sampling_seed", "corpus_hash", "papers",
    }:
        raise ValueError("private corpus snapshot has unsupported fields")
    papers_value = document["papers"]
    if not isinstance(papers_value, list):
        raise ValueError("private corpus snapshot papers must be an array")
    papers: list[CorpusPaper] = []
    fields = {
        "topic", "paper_id", "title", "abstract", "metadata", "source", "language", "paper_family",
        "sampling_weight", "sampling_probability", "abstract_incomplete", "natural_crawler_population",
        "cross_language_match",
    }
    for item in papers_value:
        if not isinstance(item, dict) or set(item) != fields:
            raise ValueError("private corpus paper has unsupported fields")
        papers.append(CorpusPaper(**item))
    snapshot = PrivateCorpusSnapshot(
        document["schema_version"], document["sampling_policy_version"], document["sampling_seed"], tuple(papers)
    )
    if document["corpus_hash"] != snapshot.corpus_hash:
        raise ValueError("private corpus snapshot corpus_hash does not match its papers")
    return snapshot


@dataclass(frozen=True, slots=True)
class PrivateSamplingAnnotation:
    """Private label and difficulty stratum for one snapshot pair."""

    topic: str
    paper_id: str
    label: int
    hard_negative: bool = False
    hard_positive: bool = False

    def __post_init__(self) -> None:
        _pair_key(self.topic, self.paper_id)
        if self.label not in range(4):
            raise ValueError("private sampling annotation label must be 0..3")
        if self.hard_negative and self.label >= 2:
            raise ValueError("private hard negatives must have label 0 or 1")
        if self.hard_positive and self.label != 3:
            raise ValueError("private hard positives must have label 3")
        if self.hard_negative and self.hard_positive:
            raise ValueError("a pair cannot be both a hard negative and positive")

    @property
    def key(self) -> tuple[str, str]:
        return _pair_key(self.topic, self.paper_id)

    def document(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "paper_id": self.paper_id,
            "label": self.label,
            "hard_negative": self.hard_negative,
            "hard_positive": self.hard_positive,
        }


@dataclass(frozen=True, slots=True)
class PrivateSamplingAnnotations:
    """Complete private annotation view of one :class:`PrivateCorpusSnapshot`."""

    rows: tuple[PrivateSamplingAnnotation, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "rows", tuple(self.rows))
        if not self.rows or len({row.key for row in self.rows}) != len(self.rows):
            raise ValueError("private sampling annotations must have unique rows")

    @property
    def by_key(self) -> Mapping[tuple[str, str], PrivateSamplingAnnotation]:
        return MappingProxyType({row.key: row for row in self.rows})

    def hash(self) -> str:
        return content_hash([row.document() for row in sorted(self.rows, key=lambda item: item.key)])


@dataclass(frozen=True, slots=True)
class SamplingPolicy:
    """Frozen producer policy.  The fixed quotas mirror ``GoldManifest``."""

    version: str
    seed: int
    dev_size: int = 300
    hidden_hard_size: int = 150
    hidden_real_size: int = 150

    def __post_init__(self) -> None:
        if not self.version or (self.dev_size, self.hidden_hard_size, self.hidden_real_size) != (300, 150, 150):
            raise ValueError("sampling policy must retain the Stage 2 300/150/150 quotas")


@dataclass(frozen=True, slots=True)
class GoldSamplingProvenance:
    """The minimal private artifact binding a public manifest to its producer."""

    schema_version: int
    snapshot_hash: str
    corpus_hash: str
    sampling_policy_version: str
    sampling_seed: int
    gold_manifest_hash: str

    def __post_init__(self) -> None:
        if self.schema_version != 1 or not all((
            self.snapshot_hash, self.corpus_hash, self.sampling_policy_version, self.gold_manifest_hash,
        )):
            raise ValueError("gold sampling provenance requires complete bindings")

    def document(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "snapshot_hash": self.snapshot_hash,
            "corpus_hash": self.corpus_hash,
            "sampling_policy_version": self.sampling_policy_version,
            "sampling_seed": self.sampling_seed,
            "gold_manifest_hash": self.gold_manifest_hash,
        }

    def hash(self) -> str:
        return content_hash(self.document())


@dataclass(frozen=True, slots=True)
class GoldSamplingResult:
    manifest: GoldManifest
    labels: GoldLabelStore
    provenance: GoldSamplingProvenance


class _StratifiedSampler:
    def __init__(self, snapshot: PrivateCorpusSnapshot, annotations: Mapping[tuple[str, str], PrivateSamplingAnnotation], seed: int) -> None:
        self.snapshot = snapshot
        self.annotations = annotations
        self.random = Random(seed)
        self.selected: dict[GoldSplit, list[CorpusPaper]] = {split: [] for split in GoldSplit}
        self.selected_ids: set[tuple[str, str]] = set()
        self.family_splits: dict[str, GoldSplit] = {}

    def _eligible(self, paper: CorpusPaper, split: GoldSplit) -> bool:
        assigned_split = self.family_splits.get(paper.paper_family)
        return paper.key not in self.selected_ids and (assigned_split is None or assigned_split is split)

    def _ranked(self, split: GoldSplit, predicate, *, weighted: bool = True) -> list[CorpusPaper]:
        ranked: list[tuple[float, tuple[str, str], CorpusPaper]] = []
        for paper in sorted(self.snapshot.papers, key=lambda item: item.key):
            if self._eligible(paper, split) and predicate(paper):
                draw = -log(max(self.random.random(), 1e-12))
                ranked.append((draw / paper.sampling_weight if weighted else draw, paper.key, paper))
        return [paper for _, _, paper in sorted(ranked)]

    def take(self, split: GoldSplit, count: int, predicate, description: str, *, weighted: bool = True) -> None:
        if count <= 0:
            return
        candidates = self._ranked(split, predicate, weighted=weighted)
        if len(candidates) < count:
            raise ValueError(f"insufficient eligible corpus rows for {split.value} {description}")
        for paper in candidates[:count]:
            self.selected[split].append(paper)
            self.selected_ids.add(paper.key)
            self.family_splits.setdefault(paper.paper_family, split)

    def ensure(self, split: GoldSplit, required: int, predicate, description: str) -> None:
        current = sum(predicate(paper) for paper in self.selected[split])
        self.take(split, required - current, predicate, description)


def _annotation_for(sampler: _StratifiedSampler, paper: CorpusPaper) -> PrivateSamplingAnnotation:
    return sampler.annotations[paper.key]


def _select_stratified_split(sampler: _StratifiedSampler, split: GoldSplit, size: int, topics: tuple[str, ...]) -> None:
    for topic in topics:
        sampler.ensure(split, 1, lambda paper, topic=topic: paper.topic == topic, f"topic {topic}")
    for language in ("en", "zh"):
        sampler.ensure(split, 1, lambda paper, language=language: paper.language == language, f"language {language}")
    sampler.ensure(split, 1, lambda paper: paper.cross_language_match, "cross-language match")
    sampler.ensure(
        split, (size + 4) // 5,
        lambda paper: _annotation_for(sampler, paper).hard_negative,
        "hard negatives",
    )
    sampler.ensure(split, (size + 9) // 10, lambda paper: paper.abstract_incomplete, "incomplete abstracts")
    sampler.ensure(split, 1, lambda paper: _annotation_for(sampler, paper).hard_positive, "hard positives")
    sampler.take(split, size - len(sampler.selected[split]), lambda paper: True, "quota")


def build_gold_sampling(
    snapshot: PrivateCorpusSnapshot,
    annotations: PrivateSamplingAnnotations,
    policy: SamplingPolicy,
) -> GoldSamplingResult:
    """Construct an exact, reproducible 600-pair Stage 2 gold-set artifact.

    The real-distribution split is selected after the two labelled strata and
    never consults ``annotations``.  It is a random sample of the remaining
    crawler population with one recorded inclusion probability, as required by
    the public ``GoldManifest`` contract.
    """

    if (snapshot.sampling_policy_version, snapshot.sampling_seed) != (policy.version, policy.seed):
        raise ValueError("sampling policy must match the frozen private corpus snapshot")
    topics = tuple(sorted({paper.topic for paper in snapshot.papers}))
    if not 6 <= len(topics) <= 8:
        raise ValueError("private corpus snapshot must cover 6..8 topics")
    by_key = annotations.by_key
    snapshot_keys = {paper.key for paper in snapshot.papers}
    if set(by_key) != snapshot_keys:
        raise ValueError("private sampling annotations must exactly cover the private corpus snapshot")

    sampler = _StratifiedSampler(snapshot, by_key, policy.seed)
    _select_stratified_split(sampler, GoldSplit.DEV, policy.dev_size, topics)
    _select_stratified_split(sampler, GoldSplit.HIDDEN_HARD, policy.hidden_hard_size, topics)

    natural = [
        paper for paper in snapshot.papers
        if sampler._eligible(paper, GoldSplit.HIDDEN_REAL) and paper.natural_crawler_population
    ]
    probabilities = {paper.sampling_probability for paper in natural}
    if len(probabilities) != 1:
        raise ValueError("hidden_real crawler population must have one recorded sampling_probability")
    sampler.take(
        GoldSplit.HIDDEN_REAL,
        policy.hidden_real_size,
        lambda paper: paper.natural_crawler_population,
        "natural crawler population",
        weighted=False,
    )

    pairs = tuple(
        GoldPair(
            paper.paper_id,
            paper.topic,
            paper.language,
            paper.source,
            paper.sampling_probability,
            paper.paper_family,
            snapshot.corpus_hash,
            split,
            abstract_incomplete=paper.abstract_incomplete,
            sampled_from_natural_distribution=split is GoldSplit.HIDDEN_REAL,
            cross_language_match=paper.cross_language_match,
        )
        for split in GoldSplit
        for paper in sampler.selected[split]
    )
    manifest = GoldManifest(1, snapshot.corpus_hash, pairs, ("en", "zh"))
    selected_annotations = {paper.pair_id: by_key[paper.key] for papers in sampler.selected.values() for paper in papers}
    labels = GoldLabelStore(
        {pair_id: annotation.label for pair_id, annotation in selected_annotations.items()},
        annotations.hash(),
        frozenset(pair_id for pair_id, annotation in selected_annotations.items() if annotation.hard_negative),
        frozenset(pair_id for pair_id, annotation in selected_annotations.items() if annotation.hard_positive),
    )
    manifest.validate(labels)
    provenance = GoldSamplingProvenance(
        1, snapshot.hash(), snapshot.corpus_hash, policy.version, policy.seed, manifest.hash()
    )
    return GoldSamplingResult(manifest, labels, provenance)
