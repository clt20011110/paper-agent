"""Bind a schema-v2 Stage 2 candidate to pre-frozen benchmark workloads."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import os
from pathlib import Path
from tempfile import mkstemp

from .canonical import canonical_json
from .schema import validate
from .stage2_benchmark_inputs import benchmark_corpus_hash
from .stage2_evaluation import PerformanceCase, PerformanceRoutingManifest, SoakManifest
from .stage2_pipeline import Stage2Paper
from .stage2_search import ReleasedStage2


_PIPELINE_COMPONENTS = (
    "rules", "reranker", "qwen", "schema_validation", "sqlite_commit",
)
class BenchmarkFreezeError(ValueError):
    """A candidate cannot be safely bound to the supplied frozen workload."""


@dataclass(frozen=True, slots=True)
class CandidateBenchmarkManifests:
    """The two public benchmark manifests derived from one frozen candidate."""

    performance: PerformanceRoutingManifest
    soak: SoakManifest


def freeze_candidate_benchmark_manifests(
    release: ReleasedStage2,
    *,
    performance_papers: Sequence[Stage2Paper],
    soak_papers: Sequence[Stage2Paper],
    selection_receipt: Mapping[str, object],
) -> CandidateBenchmarkManifests:
    """Create candidate-bound performance and soak manifests without labels.

    ``selection_receipt`` is the immutable, candidate-independent routing draw
    published by :mod:`scripts.build_stage2_workload_frame`.  It supplies the
    only allowed performance Qwen routes; calibration thresholds are bound from
    ``release`` and are never used to redraw the workload.
    """

    profile = release.profile
    if not profile.production_calibrated:
        raise BenchmarkFreezeError("benchmark freezer requires a calibrated schema-v2 candidate")
    reranker = profile.reranker_calibration
    qwen = profile.adjudicator_calibration
    if (
        reranker is None
        or qwen is None
        or profile.reranker_lock_hash is None
        or profile.adjudicator_lock_hash is None
    ):
        raise BenchmarkFreezeError("candidate has incomplete production calibration provenance")

    performance = tuple(sorted(performance_papers, key=lambda paper: paper.paper_id))
    soak = tuple(sorted(soak_papers, key=lambda paper: paper.paper_id))
    performance_receipt, soak_receipt = _validate_receipt(
        selection_receipt, performance, soak
    )
    normal_qwen_ids = frozenset(_string_ids(performance_receipt, "normal_qwen_ids"))
    stress_qwen_ids = frozenset(_string_ids(performance_receipt, "stress_qwen_ids"))
    if not normal_qwen_ids <= stress_qwen_ids:
        raise BenchmarkFreezeError("normal Qwen routes must be a subset of stress routes")

    provenance = {
        "stage2_config_hash": profile.base_runtime_config_hash,
        "model_lock_hashes": (profile.reranker_lock_hash, profile.adjudicator_lock_hash),
        "threshold_artifact_hashes": (
            reranker.threshold.hash(), qwen.threshold.hash(),
        ),
    }
    performance_manifest = PerformanceRoutingManifest(
        version=1,
        corpus_hash=benchmark_corpus_hash(performance),
        **provenance,
        output_token_limit=profile.adjudicator_max_output_tokens,
        cases=tuple(_case(profile.query, paper) for paper in performance),
        normal_qwen_ids=normal_qwen_ids,
        stress_qwen_ids=stress_qwen_ids,
        pipeline_components=_PIPELINE_COMPONENTS,
    )
    soak_manifest = SoakManifest(
        version=1,
        corpus_hash=benchmark_corpus_hash(soak),
        **provenance,
        output_token_limit=profile.adjudicator_max_output_tokens,
        cases=tuple(_case(profile.query, paper) for paper in soak),
    )
    validate(performance_manifest.document(), "stage2-performance-manifest.schema.json")
    validate(soak_manifest.document(), "stage2-soak-manifest.schema.json")
    return CandidateBenchmarkManifests(performance_manifest, soak_manifest)


def publish_candidate_benchmark_manifests(
    manifests: CandidateBenchmarkManifests,
    *,
    performance_output: Path,
    soak_output: Path,
) -> None:
    """Persist two canonical manifests with no replacement semantics."""

    if performance_output == soak_output:
        raise ValueError("performance and soak manifest outputs must differ")
    existing = next(
        (path for path in (performance_output, soak_output) if os.path.lexists(path)),
        None,
    )
    if existing is not None:
        raise FileExistsError(f"benchmark manifest output already exists: {existing}")
    _write_new(performance_output, canonical_json(manifests.performance.document()))
    _write_new(soak_output, canonical_json(manifests.soak.document()))


def _validate_receipt(
    receipt: Mapping[str, object],
    performance: Sequence[Stage2Paper],
    soak: Sequence[Stage2Paper],
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    if receipt.get("schema_version") != 1 or receipt.get("candidate_independent") is not True:
        raise BenchmarkFreezeError("workload receipt is not a candidate-independent v1 receipt")
    omitted = receipt.get("omitted_bindings")
    if omitted != ["stage2_config_hash", "threshold_artifact_hashes", "model_lock_hashes"]:
        raise BenchmarkFreezeError("workload receipt has unexpected candidate bindings")
    performance_receipt = _mapping(receipt, "performance")
    soak_receipt = _mapping(receipt, "soak")
    _validate_workload(
        performance_receipt,
        performance,
        expected_count=1_000,
        label="performance",
    )
    _validate_workload(soak_receipt, soak, expected_count=10_000, label="soak")
    normal = _string_ids(performance_receipt, "normal_qwen_ids")
    stress = _string_ids(performance_receipt, "stress_qwen_ids")
    if (
        performance_receipt.get("abstract_present_count") != 900
        or performance_receipt.get("abstract_missing_count") != 100
    ):
        raise BenchmarkFreezeError("receipt must freeze 900 present and 100 missing performance abstracts")
    if len(normal) != 150 or len(stress) != 300:
        raise BenchmarkFreezeError("receipt must freeze 15% normal and 30% stress Qwen routes")
    if len(set(normal)) != len(normal) or len(set(stress)) != len(stress):
        raise BenchmarkFreezeError("receipt Qwen routes must be unique")
    missing = {paper.paper_id for paper in performance if not paper.abstract or not paper.abstract.strip()}
    if len(missing) != 100 or not missing <= set(normal):
        raise BenchmarkFreezeError("receipt normal routing must contain every missing abstract")
    return performance_receipt, soak_receipt


def _validate_workload(
    receipt: Mapping[str, object],
    papers: Sequence[Stage2Paper],
    *,
    expected_count: int,
    label: str,
) -> None:
    paper_ids = _string_ids(receipt, "paper_ids")
    observed_ids = [paper.paper_id for paper in papers]
    if len(papers) != expected_count or len(set(observed_ids)) != expected_count:
        raise BenchmarkFreezeError(f"{label} workload must contain exactly {expected_count:,} unique papers")
    if paper_ids != tuple(sorted(observed_ids)):
        raise BenchmarkFreezeError(f"{label} workload paper IDs do not match its receipt")
    observed_hash = benchmark_corpus_hash(papers)
    if receipt.get("papers_corpus_hash") != observed_hash:
        raise BenchmarkFreezeError(f"{label} workload corpus hash does not match its receipt")


def _case(query: str, paper: Stage2Paper) -> PerformanceCase:
    document = f"Title: {paper.title}\nAbstract: {paper.abstract or ''}\nKeywords: {', '.join(paper.keywords)}"
    # The serving client uses a stable characters/4 bound for chat context;
    # include the reranker query because every non-rule paper receives it.
    input_tokens = max(1, (len(query) + len(document)) // 4 + 1)
    return PerformanceCase(
        pair_id=paper.paper_id,
        input_tokens=input_tokens,
        abstract_missing=not bool(paper.abstract and paper.abstract.strip()),
    )


def _mapping(value: Mapping[str, object], field: str) -> Mapping[str, object]:
    item = value.get(field)
    if not isinstance(item, Mapping):
        raise BenchmarkFreezeError(f"workload receipt requires an object {field}")
    return item


def _string_ids(value: Mapping[str, object], field: str) -> tuple[str, ...]:
    item = value.get(field)
    if not isinstance(item, list) or not all(isinstance(entry, str) and entry for entry in item):
        raise BenchmarkFreezeError(f"workload receipt requires non-empty string IDs for {field}")
    return tuple(item)


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
