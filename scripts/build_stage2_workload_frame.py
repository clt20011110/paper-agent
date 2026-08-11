#!/usr/bin/env python3
"""Freeze label-free Stage 2 performance, soak, and parity paper workloads."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
from random import Random
import sqlite3
from tempfile import mkstemp
from typing import Any, Iterable, Mapping

from paper_agent.canonical import content_hash
from paper_agent.stage2_benchmark_inputs import benchmark_corpus_hash, benchmark_papers_from_document
from paper_agent.stage2_pipeline import Stage2Paper
from paper_agent.stage2_sampling import load_private_corpus_snapshot


def _paper_document(paper: Stage2Paper) -> dict[str, Any]:
    return {
        "paper_id": paper.paper_id,
        "title": paper.title,
        "abstract": paper.abstract,
        "keywords": list(paper.keywords),
        "document_type": paper.document_type,
        "possibly_truncated": paper.possibly_truncated,
        "multi_condition_conflict": paper.multi_condition_conflict,
        "language_anomaly": paper.language_anomaly,
    }


def _papers_document(papers: Iterable[Stage2Paper]) -> dict[str, Any]:
    return {
        "schema_version": "1",
        "kind": "stage2_benchmark_papers",
        "papers": [_paper_document(paper) for paper in sorted(papers, key=lambda item: item.paper_id)],
    }


def _load_json_papers(path: Path) -> tuple[Stage2Paper, ...]:
    return benchmark_papers_from_document(json.loads(path.read_text(encoding="utf-8")))


def _load_sqlite_papers(path: Path) -> tuple[Stage2Paper, ...]:
    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT paper_id, title, abstract, keywords_json FROM papers ORDER BY paper_id"
        ).fetchall()
    finally:
        connection.close()
    papers = []
    for paper_id, title, abstract, keywords_json in rows:
        keywords = json.loads(keywords_json)
        if not isinstance(keywords, list) or not all(isinstance(value, str) for value in keywords):
            raise ValueError(f"SQLite paper {paper_id} has invalid keywords_json")
        papers.append(Stage2Paper(str(paper_id), str(title), abstract, tuple(keywords)))
    return tuple(papers)


def _load_crossref_snapshot(path: Path) -> tuple[Stage2Paper, ...]:
    snapshot = load_private_corpus_snapshot(path)
    return tuple(
        Stage2Paper(paper.paper_id, paper.title, paper.abstract)
        for paper in snapshot.papers
    )


def _deduplicate(sources: Iterable[Iterable[Stage2Paper]]) -> tuple[Stage2Paper, ...]:
    selected: dict[str, Stage2Paper] = {}
    for source in sources:
        for paper in source:
            current = selected.get(paper.paper_id)
            if current is None or (current.abstract is None and paper.abstract is not None):
                selected[paper.paper_id] = paper
    return tuple(selected[key] for key in sorted(selected))


def build_workload_frame(papers: Iterable[Stage2Paper], *, seed: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Select exact public workloads without inventing model or threshold bindings."""

    universe = tuple(sorted(papers, key=lambda item: item.paper_id))
    present = tuple(paper for paper in universe if paper.abstract is not None)
    missing = tuple(paper for paper in universe if paper.abstract is None)
    if len(universe) < 10_000:
        raise ValueError("workload frame requires at least 10,000 unique papers")
    if len(present) < 900 or len(missing) < 100:
        raise ValueError("workload frame requires at least 900 abstracts and 100 missing abstracts")
    random = Random(seed)
    performance = tuple(random.sample(present, 900) + random.sample(missing, 100))
    soak = tuple(random.sample(universe, 10_000))
    performance_ids = tuple(paper.paper_id for paper in performance)
    missing_ids = {paper.paper_id for paper in performance if paper.abstract is None}
    present_ids = [paper.paper_id for paper in performance if paper.abstract is not None]
    normal_extra = set(random.sample(present_ids, 50))
    stress_extra = normal_extra | set(
        random.sample([paper_id for paper_id in present_ids if paper_id not in normal_extra], 150)
    )
    normal_qwen_ids = tuple(sorted(missing_ids | normal_extra))
    stress_qwen_ids = tuple(sorted(missing_ids | stress_extra))
    performance_document = _papers_document(performance)
    soak_document = _papers_document(soak)
    receipt = {
        "schema_version": 1,
        "kind": "stage2_candidate_independent_workload_frame",
        "seed": seed,
        "candidate_independent": True,
        "omitted_bindings": ["stage2_config_hash", "threshold_artifact_hashes", "model_lock_hashes"],
        "universe": {
            "paper_count": len(universe),
            "paper_ids_hash": content_hash([paper.paper_id for paper in universe]),
            "abstract_present_count": len(present),
            "abstract_missing_count": len(missing),
        },
        "performance": {
            "papers_corpus_hash": benchmark_corpus_hash(benchmark_papers_from_document(performance_document)),
            "paper_ids": sorted(performance_ids),
            "abstract_present_count": 900,
            "abstract_missing_count": 100,
            "normal_qwen_ids": list(normal_qwen_ids),
            "stress_qwen_ids": list(stress_qwen_ids),
        },
        "soak": {
            "papers_corpus_hash": benchmark_corpus_hash(benchmark_papers_from_document(soak_document)),
            "paper_ids": sorted(paper.paper_id for paper in soak),
            "paper_count": 10_000,
        },
        "parity": {
            "papers_corpus_hash": benchmark_corpus_hash(benchmark_papers_from_document(soak_document)),
            "paper_ids": sorted(paper.paper_id for paper in soak),
            "paper_count": 10_000,
        },
    }
    return performance_document, soak_document, receipt


def _write_no_replace(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def publish(*, performance: Mapping[str, Any], soak: Mapping[str, Any], receipt: Mapping[str, Any], performance_output: Path, soak_output: Path, receipt_output: Path) -> None:
    targets = (performance_output, soak_output, receipt_output)
    if len({path.resolve() for path in targets}) != len(targets):
        raise ValueError("all output paths must differ")
    existing = next((path for path in targets if os.path.lexists(path)), None)
    if existing is not None:
        raise FileExistsError(f"output already exists: {existing}")
    for path, document in zip(targets, (performance, soak, receipt), strict=True):
        _write_no_replace(path, (json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode())


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", type=Path, action="append", default=[])
    parser.add_argument("--corpus", type=Path, action="append", default=[])
    parser.add_argument("--crossref-snapshot", type=Path, action="append", default=[])
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--performance-papers-output", required=True, type=Path)
    parser.add_argument("--soak-papers-output", required=True, type=Path)
    parser.add_argument("--selection-receipt-output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    sources = [
        *(_load_sqlite_papers(path) for path in args.sqlite),
        *(_load_json_papers(path) for path in args.corpus),
        *(_load_crossref_snapshot(path) for path in args.crossref_snapshot),
    ]
    if not sources:
        raise ValueError("supply at least one --sqlite, --corpus, or --crossref-snapshot")
    performance, soak, receipt = build_workload_frame(_deduplicate(sources), seed=args.seed)
    publish(
        performance=performance,
        soak=soak,
        receipt=receipt,
        performance_output=args.performance_papers_output,
        soak_output=args.soak_papers_output,
        receipt_output=args.selection_receipt_output,
    )
    print(json.dumps({"status": "complete", "performance_count": 1_000, "soak_count": 10_000, "receipt_hash": sha256(json.dumps(receipt, sort_keys=True).encode()).hexdigest()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
