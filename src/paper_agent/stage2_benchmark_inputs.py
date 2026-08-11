"""Normalized public inputs shared by Stage 2 benchmark runners and verifiers."""

from __future__ import annotations

from typing import Any, Sequence

from .canonical import content_hash
from .stage2_pipeline import Stage2Paper


def benchmark_papers_from_document(value: Any) -> tuple[Stage2Paper, ...]:
    """Parse normalized benchmark papers from a CLI or release-evidence document."""

    if isinstance(value, dict):
        if set(value) == {"schema_version", "kind", "papers"}:
            if value["schema_version"] != "1" or value["kind"] != "stage2_benchmark_papers":
                raise ValueError("benchmark papers object has an unsupported identity")
        elif set(value) != {"papers"}:
            raise ValueError("benchmark papers object must contain only the papers array")
        value = value["papers"]
    if not isinstance(value, list) or not value:
        raise ValueError("benchmark papers must be a non-empty JSON array")
    allowed = {
        "paper_id",
        "title",
        "abstract",
        "keywords",
        "document_type",
        "possibly_truncated",
        "multi_condition_conflict",
        "language_anomaly",
    }
    papers: list[Stage2Paper] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict) or not set(item) <= allowed:
            raise ValueError(f"benchmark paper {index} has invalid fields")
        keywords = item.get("keywords", [])
        if not isinstance(keywords, list) or not all(isinstance(keyword, str) for keyword in keywords):
            raise ValueError(f"benchmark paper {index} keywords must be strings")
        paper_id = item.get("paper_id")
        title = item.get("title")
        if not isinstance(paper_id, str) or not isinstance(title, str):
            raise ValueError(f"benchmark paper {index} paper_id and title must be strings")
        abstract = item.get("abstract")
        document_type = item.get("document_type")
        if abstract is not None and not isinstance(abstract, str):
            raise ValueError(f"benchmark paper {index} abstract must be a string or null")
        if document_type is not None and not isinstance(document_type, str):
            raise ValueError(f"benchmark paper {index} document_type must be a string or null")
        flags = {
            name: item.get(name, False)
            for name in ("possibly_truncated", "multi_condition_conflict", "language_anomaly")
        }
        if not all(isinstance(flag, bool) for flag in flags.values()):
            raise ValueError(f"benchmark paper {index} flags must be booleans")
        papers.append(Stage2Paper(
            paper_id=paper_id,
            title=title,
            abstract=abstract,
            keywords=tuple(keywords),
            document_type=document_type,
            **flags,
        ))
    return tuple(papers)


def benchmark_corpus_hash(papers: Sequence[Stage2Paper]) -> str:
    """Hash the normalized paper content consumed by the measured runner."""

    return content_hash({
        "schema_version": 1,
        "papers": [
            {
                "paper_id": paper.paper_id,
                "title": paper.title,
                "abstract": paper.abstract,
                "keywords": list(paper.keywords),
                "document_type": paper.document_type,
                "possibly_truncated": paper.possibly_truncated,
                "multi_condition_conflict": paper.multi_condition_conflict,
                "language_anomaly": paper.language_anomaly,
            }
            for paper in sorted(papers, key=lambda item: item.paper_id)
        ],
    })
