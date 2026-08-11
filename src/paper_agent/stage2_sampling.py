"""Deterministic construction of the private Stage 2 sampling corpus.

The public :class:`~paper_agent.stage2_evaluation.GoldManifest` deliberately
does not contain paper text, annotations, or private sampling strata.  This
module keeps those inputs in a separate private snapshot and emits the public
manifest together with a small binding artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import isclose, isfinite, log
import os
from random import Random
from pathlib import Path
from tempfile import mkstemp
from types import MappingProxyType
from typing import Any, Mapping

from .canonical import content_hash
from .schema import SchemaValidationError, validate
from .stage2_evaluation import GoldManifest, GoldPair, GoldSplit, make_pair_id


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

    _validate_document(document, "stage2-private-corpus-snapshot.schema.json")
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


def write_private_corpus_snapshot(path: Path, snapshot: PrivateCorpusSnapshot) -> None:
    """Atomically write a private frozen crawler snapshot."""

    _write_json_atomically(path, snapshot.document())


def load_private_corpus_snapshot(path: Path) -> PrivateCorpusSnapshot:
    """Load and verify a private frozen crawler snapshot."""

    return private_corpus_snapshot_from_document(_read_json_object(path, "private corpus snapshot"))


@dataclass(frozen=True, slots=True)
class PrivateSamplingAnnotation:
    """Provisional curation label and difficulty stratum used only for sampling."""

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
    """Provisional sampling strata for curated candidates, never final gold labels."""

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

    def document(self, *, snapshot: PrivateCorpusSnapshot) -> dict[str, Any]:
        """Return the versioned private annotation artifact bound to ``snapshot``."""

        return {
            "schema_version": 1,
            "snapshot_hash": snapshot.hash(),
            "rows": [row.document() for row in sorted(self.rows, key=lambda item: item.key)],
        }


def private_sampling_annotations_from_document(
    document: object, *, snapshot: PrivateCorpusSnapshot
) -> PrivateSamplingAnnotations:
    """Parse private sampling labels and require their exact snapshot binding."""

    _validate_document(document, "stage2-private-sampling-annotations.schema.json")
    assert isinstance(document, dict)
    if document["snapshot_hash"] != snapshot.hash():
        raise ValueError("private sampling annotations do not bind the supplied corpus snapshot")
    rows = tuple(PrivateSamplingAnnotation(**row) for row in document["rows"])
    if not {row.key for row in rows} <= {paper.key for paper in snapshot.papers}:
        raise ValueError("private sampling annotations contain rows outside the supplied corpus snapshot")
    return PrivateSamplingAnnotations(rows)


def write_private_sampling_annotations(
    path: Path, annotations: PrivateSamplingAnnotations, *, snapshot: PrivateCorpusSnapshot
) -> None:
    """Atomically write private sampling labels bound to a frozen snapshot."""

    _write_json_atomically(path, annotations.document(snapshot=snapshot))


def load_private_sampling_annotations(
    path: Path, *, snapshot: PrivateCorpusSnapshot
) -> PrivateSamplingAnnotations:
    """Load private sampling labels bound to a frozen snapshot."""

    return private_sampling_annotations_from_document(
        _read_json_object(path, "private sampling annotations"), snapshot=snapshot
    )


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
    sampling_annotations_hash: str
    gold_manifest_hash: str

    def __post_init__(self) -> None:
        if self.schema_version != 1 or not all((
            self.snapshot_hash,
            self.corpus_hash,
            self.sampling_policy_version,
            self.sampling_annotations_hash,
            self.gold_manifest_hash,
        )):
            raise ValueError("gold sampling provenance requires complete bindings")

    def document(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "snapshot_hash": self.snapshot_hash,
            "corpus_hash": self.corpus_hash,
            "sampling_policy_version": self.sampling_policy_version,
            "sampling_seed": self.sampling_seed,
            "sampling_annotations_hash": self.sampling_annotations_hash,
            "gold_manifest_hash": self.gold_manifest_hash,
        }

    def hash(self) -> str:
        return content_hash(self.document())


def gold_sampling_provenance_from_document(
    document: object,
    *,
    snapshot: PrivateCorpusSnapshot,
    annotations: PrivateSamplingAnnotations,
    manifest: GoldManifest,
) -> GoldSamplingProvenance:
    """Parse provenance and verify all producer and public-manifest bindings."""

    _validate_document(document, "stage2-gold-sampling-provenance.schema.json")
    assert isinstance(document, dict)
    provenance = GoldSamplingProvenance(**document)
    if provenance.snapshot_hash != snapshot.hash() or provenance.corpus_hash != snapshot.corpus_hash:
        raise ValueError("gold sampling provenance does not bind the supplied corpus snapshot")
    if (provenance.sampling_policy_version, provenance.sampling_seed) != (
        snapshot.sampling_policy_version,
        snapshot.sampling_seed,
    ):
        raise ValueError("gold sampling provenance does not bind the supplied sampling policy")
    if provenance.sampling_annotations_hash != annotations.hash():
        raise ValueError("gold sampling provenance does not bind the supplied sampling annotations")
    if provenance.gold_manifest_hash != manifest.hash() or manifest.corpus_hash != snapshot.corpus_hash:
        raise ValueError("gold sampling provenance does not bind the supplied gold manifest")
    return provenance


def write_gold_sampling_provenance(path: Path, provenance: GoldSamplingProvenance) -> None:
    """Atomically write the private producer-to-manifest binding artifact."""

    _write_json_atomically(path, provenance.document())


def load_gold_sampling_provenance(
    path: Path,
    *,
    snapshot: PrivateCorpusSnapshot,
    annotations: PrivateSamplingAnnotations,
    manifest: GoldManifest,
) -> GoldSamplingProvenance:
    """Load provenance and verify its snapshot, annotation, and manifest bindings."""

    return gold_sampling_provenance_from_document(
        _read_json_object(path, "gold sampling provenance"),
        snapshot=snapshot,
        annotations=annotations,
        manifest=manifest,
    )


@dataclass(frozen=True, slots=True)
class GoldSamplingResult:
    manifest: GoldManifest
    provenance: GoldSamplingProvenance


@dataclass(frozen=True, slots=True)
class HiddenRealSelection:
    """Natural-distribution rows frozen before curated labels are opened."""

    snapshot_hash: str
    sampling_policy_version: str
    sampling_seed: int
    sampling_probability: float
    pair_keys: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "pair_keys", tuple(self.pair_keys))
        if (
            not self.snapshot_hash
            or not self.sampling_policy_version
            or not self.pair_keys
            or len(set(self.pair_keys)) != len(self.pair_keys)
        ):
            raise ValueError("hidden_real selection requires one unique frozen pair set")

    def document(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "snapshot_hash": self.snapshot_hash,
            "sampling_policy_version": self.sampling_policy_version,
            "sampling_seed": self.sampling_seed,
            "sampling_probability": self.sampling_probability,
            "pair_keys": [
                {"topic": topic, "paper_id": paper_id}
                for topic, paper_id in self.pair_keys
            ],
        }

    def hash(self) -> str:
        return content_hash(self.document())


def hidden_real_selection_from_document(
    document: object,
    *,
    snapshot: PrivateCorpusSnapshot,
    policy: SamplingPolicy,
) -> HiddenRealSelection:
    """Load a frozen HIDDEN_REAL draw and bind it to its snapshot and policy."""

    _validate_document(document, "stage2-hidden-real-freeze-frame.schema.json")
    assert isinstance(document, dict)
    selection = HiddenRealSelection(
        document["snapshot_hash"],
        document["sampling_policy_version"],
        document["sampling_seed"],
        document["sampling_probability"],
        tuple((row["topic"], row["paper_id"]) for row in document["pair_keys"]),
    )
    if selection != select_hidden_real(snapshot, policy):
        raise ValueError("hidden_real freeze frame does not match the supplied snapshot and policy")
    return selection


def write_hidden_real_selection(path: Path, selection: HiddenRealSelection) -> None:
    """Atomically write the evaluator-private HIDDEN_REAL freeze frame."""

    _write_json_atomically(path, selection.document())


def load_hidden_real_selection(
    path: Path,
    *,
    snapshot: PrivateCorpusSnapshot,
    policy: SamplingPolicy,
) -> HiddenRealSelection:
    """Load and verify one evaluator-private HIDDEN_REAL freeze frame."""

    return hidden_real_selection_from_document(
        _read_json_object(path, "hidden_real freeze frame"),
        snapshot=snapshot,
        policy=policy,
    )


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
    def curated(predicate):
        return lambda paper: paper.key in sampler.annotations and predicate(paper)

    for topic in topics:
        sampler.ensure(
            split,
            1,
            curated(lambda paper, topic=topic: paper.topic == topic),
            f"curated candidates for topic {topic}",
        )
    for language in ("en", "zh"):
        sampler.ensure(
            split,
            1,
            curated(lambda paper, language=language: paper.language == language),
            f"curated candidates for language {language}",
        )
    sampler.ensure(split, 1, curated(lambda paper: paper.cross_language_match), "curated cross-language candidates")
    sampler.ensure(
        split, (size + 4) // 5,
        curated(lambda paper: _annotation_for(sampler, paper).hard_negative),
        "curated hard negatives",
    )
    sampler.ensure(
        split,
        (size + 9) // 10,
        curated(lambda paper: paper.abstract_incomplete),
        "curated incomplete abstracts",
    )
    sampler.ensure(
        split,
        1,
        curated(lambda paper: _annotation_for(sampler, paper).hard_positive),
        "curated hard positives",
    )
    sampler.take(split, size - len(sampler.selected[split]), curated(lambda paper: True), "curated quota")


def _draw_hidden_real(
    snapshot: PrivateCorpusSnapshot,
    policy: SamplingPolicy,
    annotations: Mapping[tuple[str, str], PrivateSamplingAnnotation],
) -> tuple[_StratifiedSampler, tuple[str, ...], float]:
    if (snapshot.sampling_policy_version, snapshot.sampling_seed) != (
        policy.version,
        policy.seed,
    ):
        raise ValueError("sampling policy must match the frozen private corpus snapshot")
    topics = tuple(sorted({paper.topic for paper in snapshot.papers}))
    if not 6 <= len(topics) <= 8:
        raise ValueError("private corpus snapshot must cover 6..8 topics")
    natural = [paper for paper in snapshot.papers if paper.natural_crawler_population]
    if len(natural) < policy.hidden_real_size:
        raise ValueError("insufficient natural crawler rows for hidden_real")
    sampling_probability = policy.hidden_real_size / len(natural)
    if any(
        not isclose(
            paper.sampling_probability,
            sampling_probability,
            rel_tol=1e-9,
            abs_tol=1e-12,
        )
        for paper in natural
    ):
        raise ValueError(
            "natural crawler rows must record hidden_real_size / natural frame size as sampling_probability"
        )

    sampler = _StratifiedSampler(snapshot, annotations, policy.seed)
    sampler.take(
        GoldSplit.HIDDEN_REAL,
        policy.hidden_real_size,
        lambda paper: paper.natural_crawler_population,
        "natural crawler population",
        weighted=False,
    )
    return sampler, topics, sampling_probability


def select_hidden_real(
    snapshot: PrivateCorpusSnapshot,
    policy: SamplingPolicy,
) -> HiddenRealSelection:
    """Freeze HIDDEN_REAL without receiving or opening curated annotations."""

    sampler, _, sampling_probability = _draw_hidden_real(snapshot, policy, {})
    return HiddenRealSelection(
        snapshot.hash(),
        policy.version,
        policy.seed,
        sampling_probability,
        tuple(paper.key for paper in sampler.selected[GoldSplit.HIDDEN_REAL]),
    )


def build_gold_sampling(
    snapshot: PrivateCorpusSnapshot,
    annotations: PrivateSamplingAnnotations,
    policy: SamplingPolicy,
    *,
    hidden_real_selection: HiddenRealSelection | None = None,
) -> GoldSamplingResult:
    """Construct an exact, reproducible 600-pair Stage 2 gold-set artifact.

    HIDDEN_REAL is selected first from the complete natural crawler population
    without consulting annotations.  DEV and HIDDEN_HARD are then drawn from
    the remaining provisional curation strata.  All selected rows receive
    authoritative double annotation in a separate post-selection ledger before
    ``manifest.validate(labels)``.
    """

    expected_hidden_real = select_hidden_real(snapshot, policy)
    if hidden_real_selection is None:
        hidden_real_selection = expected_hidden_real
    elif hidden_real_selection != expected_hidden_real:
        raise ValueError("hidden_real selection does not match the frozen snapshot and policy")

    by_key = annotations.by_key
    snapshot_keys = {paper.key for paper in snapshot.papers}
    if not set(by_key) <= snapshot_keys:
        raise ValueError("private sampling annotations contain rows outside the private corpus snapshot")

    sampler, topics, hidden_real_probability = _draw_hidden_real(
        snapshot, policy, by_key
    )
    if tuple(
        paper.key for paper in sampler.selected[GoldSplit.HIDDEN_REAL]
    ) != hidden_real_selection.pair_keys:
        raise ValueError("hidden_real selection changed after curated annotations were opened")
    _select_stratified_split(sampler, GoldSplit.DEV, policy.dev_size, topics)
    _select_stratified_split(sampler, GoldSplit.HIDDEN_HARD, policy.hidden_hard_size, topics)

    pairs = tuple(
        GoldPair(
            paper.paper_id,
            paper.topic,
            paper.language,
            paper.source,
            hidden_real_probability if split is GoldSplit.HIDDEN_REAL else None,
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
    manifest.validate_sampling_structure()
    provenance = GoldSamplingProvenance(
        1,
        snapshot.hash(),
        snapshot.corpus_hash,
        policy.version,
        policy.seed,
        annotations.hash(),
        manifest.hash(),
    )
    return GoldSamplingResult(manifest, provenance)


def write_gold_sampling_manifest(path: Path, manifest: GoldManifest) -> None:
    """Validate and atomically publish one public, label-free gold manifest."""

    manifest.validate_sampling_structure()
    document = manifest.document()
    _validate_document(document, "stage2-gold-manifest.schema.json")
    _write_json_atomically(path, document)


def _validate_document(document: object, schema_name: str) -> None:
    try:
        validate(document, schema_name)
    except SchemaValidationError as error:
        raise ValueError(str(error)) from error


def _read_json_object(path: Path, artifact_name: str) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"{artifact_name} must be a JSON object")
    return document


def _write_json_atomically(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
