"""Portable imports and exports for the canonical paper store."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from .domain import CollectionMembership, MembershipStatus, Paper, PaperSource
from .identity import paper_id_for, source_id_for, title_author_year_key
from .repository import PaperRepository


@dataclass(frozen=True)
class ImportReport:
    """What an import would write, including explicit legacy field mappings."""

    counts: Mapping[str, int]
    mappings: Mapping[str, str]
    warnings: tuple[str, ...] = ()
    unmigrated: tuple[str, ...] = ()


def export_jsonl(repository: PaperRepository, path: str | Path) -> int:
    """Export canonical papers, provider sources, and collection memberships."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = _export_rows(repository)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    return len(rows)


def export_csv(repository: PaperRepository, path: str | Path) -> int:
    """Export one flat row per paper; every nested value is JSON text."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    papers, sources, memberships = _export_values(repository)
    sources_by_paper: dict[str, list[dict[str, Any]]] = {}
    memberships_by_paper: dict[str, list[dict[str, Any]]] = {}
    for source in sources:
        sources_by_paper.setdefault(source["paper_id"], []).append(source)
    for membership in memberships:
        memberships_by_paper.setdefault(membership["membership"]["paper_id"], []).append(membership)

    paper_fields = tuple(Paper.__dataclass_fields__)
    fieldnames = (*paper_fields, "sources_json", "memberships_json")
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for paper in papers:
            writer.writerow(
                {
                    **paper,
                    "authors": _json(paper["authors"]),
                    "keywords": _json(paper["keywords"]),
                    "sources_json": _json(sources_by_paper.get(paper["paper_id"], [])),
                    "memberships_json": _json(memberships_by_paper.get(paper["paper_id"], [])),
                }
            )
    return len(papers)


def import_jsonl(
    repository: PaperRepository, path: str | Path, *, dry_run: bool = False
) -> ImportReport:
    """Import records written by :func:`export_jsonl` without duplicate rows."""
    records = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"JSONL line {line_number} must be an object")
            records.append(record)

    papers = [record["paper"] for record in records if _record_kind(record) == "paper"]
    sources = [record["source"] for record in records if _record_kind(record) == "source"]
    memberships = [record for record in records if _record_kind(record) == "membership"]
    if len(papers) + len(sources) + len(memberships) != len(records):
        raise ValueError("JSONL records must be paper, source, or membership records")

    if not dry_run:
        for paper in papers:
            repository.save_paper(Paper.from_dict(paper))
        for source in sources:
            repository.upsert_source(PaperSource.from_dict(source))
        for record in memberships:
            _save_membership(repository, record)

    return ImportReport(
        counts={"papers": len(papers), "sources": len(sources), "memberships": len(memberships)},
        mappings={},
    )


def import_legacy_json(
    repository: PaperRepository, path: str | Path, *, dry_run: bool = False
) -> ImportReport:
    """Import v1 JSON documents containing a list or a ``papers`` list."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = document if isinstance(document, list) else document.get("papers") if isinstance(document, dict) else None
    if not isinstance(rows, list):
        raise ValueError("legacy JSON must be a list or an object with a papers list")

    converted: list[tuple[Paper, PaperSource]] = []
    warnings: list[str] = []
    unmigrated: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            unmigrated.append(f"papers[{index}]")
            continue
        converted_row, row_warnings, row_unmigrated = _legacy_paper(row, index)
        converted.append(converted_row)
        warnings.extend(row_warnings)
        unmigrated.extend(row_unmigrated)

    if not dry_run:
        for paper, source in converted:
            repository.save_paper(paper)
            repository.upsert_source(source)

    return ImportReport(
        counts={"papers": len(converted), "sources": len(converted), "memberships": 0},
        mappings=_LEGACY_MAPPINGS,
        warnings=tuple(warnings),
        unmigrated=tuple(unmigrated),
    )


def _export_rows(repository: PaperRepository) -> list[dict[str, Any]]:
    papers, sources, memberships = _export_values(repository)
    return (
        [{"record_type": "paper", "paper": paper} for paper in papers]
        + [{"record_type": "source", "source": source} for source in sources]
        + [{"record_type": "membership", **membership} for membership in memberships]
    )


def _export_values(
    repository: PaperRepository,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    connection = repository.connection
    papers = [
        Paper.from_dict(
            {
                **dict(row),
                "authors": json.loads(row["authors_json"]),
                "keywords": json.loads(row["keywords_json"]),
            }
        ).to_dict()
        for row in connection.execute("SELECT * FROM papers ORDER BY paper_id")
    ]
    sources = [
        PaperSource.from_dict(
            {
                **dict(row),
                "raw_metadata": json.loads(row["raw_metadata_json"]),
                "metadata_capabilities": json.loads(row["metadata_capabilities_json"]),
                "download_capabilities": json.loads(row["download_capabilities_json"]),
            }
        ).to_dict()
        for row in connection.execute("SELECT * FROM paper_sources ORDER BY source_id")
    ]
    memberships = [
        {
            "membership": CollectionMembership(
                collection_id=row["collection_id"],
                paper_id=row["paper_id"],
                membership_status=MembershipStatus(row["membership_status"]),
                official_evidence=tuple(json.loads(row["official_evidence_json"] or "[]")),
                observed_at=row["observed_at"],
            ).to_dict(),
            "collection": {
                "collection_id": row["collection_id"],
                "name": row["name"],
                "collection_type": row["collection_type"],
                "venue_id": row["venue_id"],
                "descriptor": json.loads(row["descriptor_json"]),
            },
        }
        for row in connection.execute(
            """SELECT pc.*, c.name, c.collection_type, c.venue_id, c.descriptor_json
               FROM paper_collections pc JOIN collections c ON c.collection_id = pc.collection_id
               ORDER BY pc.paper_id, pc.collection_id"""
        )
    ]
    return papers, sources, memberships


def _record_kind(record: Mapping[str, Any]) -> str:
    return str(record.get("record_type", record.get("type", "")))


def _save_membership(repository: PaperRepository, record: Mapping[str, Any]) -> None:
    collection = record["collection"]
    membership = CollectionMembership.from_dict(record["membership"])
    repository.save_collection(
        collection["collection_id"],
        collection["name"],
        collection["collection_type"],
        venue_id=collection.get("venue_id"),
        descriptor=collection.get("descriptor"),
    )
    repository.upsert_membership(membership)


_LEGACY_MAPPINGS = {
    "id": "paper.paper_id/source.external_id",
    "title": "paper.title",
    "abstract": "paper.abstract",
    "authors": "paper.authors",
    "keywords": "paper.keywords",
    "year": "paper.year",
    "venue": "paper.venue_name",
    "venue_type": "paper.venue_type",
    "source_platform": "source.provider",
    "pdf_url": "source.pdf_url",
    "doi": "paper.doi",
    "bibtex": "source.bibtex",
    "citation_count": "source.citation_count",
    "arxiv_id": "paper.arxiv_id",
}


def _legacy_paper(
    row: Mapping[str, Any], index: int
) -> tuple[tuple[Paper, PaperSource], list[str], list[str]]:
    title = str(row.get("title", "")).strip()
    if not title:
        raise ValueError(f"legacy papers[{index}] has no title")
    authors = _strings(row.get("authors"))
    keywords = _strings(row.get("keywords"))
    year = _integer(row.get("year"))
    provider = str(row.get("source_platform", "legacy")).strip().lower() or "legacy"
    external_id = str(row.get("id") or row.get("external_id") or title_author_year_key(title, authors, year))
    paper_id = paper_id_for(
        doi=_string_or_none(row.get("doi")),
        arxiv_id=_string_or_none(row.get("arxiv_id")),
        provider=provider,
        external_id=external_id,
    )
    venue_type = _venue_type(row.get("venue_type"))
    unmigrated = [
        f"papers[{index}].{key}"
        for key in row
        if key not in _LEGACY_MAPPINGS and key not in {"external_id", "publication_date", "url", "canonical_url"}
    ]
    warnings = []
    if not row.get("id") and not row.get("external_id"):
        warnings.append(f"papers[{index}] has no legacy ID; title/author/year became its source ID")
    paper = Paper(
        paper_id=paper_id,
        title=title,
        abstract=_string_or_none(row.get("abstract")),
        authors=authors,
        keywords=keywords,
        publication_date=_string_or_none(row.get("publication_date")),
        year=year,
        venue_name=_string_or_none(row.get("venue")),
        venue_type=venue_type,
        doi=_string_or_none(row.get("doi")),
        arxiv_id=_string_or_none(row.get("arxiv_id")),
        canonical_url=_string_or_none(row.get("canonical_url") or row.get("url")),
    )
    source = PaperSource(
        source_id=source_id_for(provider, external_id),
        paper_id=paper_id,
        provider=provider,
        external_id=external_id,
        landing_url=paper.canonical_url,
        pdf_url=_string_or_none(row.get("pdf_url")),
        bibtex=_string_or_none(row.get("bibtex")),
        citation_count=_integer(row.get("citation_count")),
        raw_metadata={"legacy": dict(row)},
    )
    return (paper, source), warnings, unmigrated


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _strings(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    if isinstance(value, list):
        return tuple(str(part) for part in value)
    raise ValueError("legacy authors and keywords must be strings or lists")


def _string_or_none(value: Any) -> str | None:
    return str(value).strip() if value is not None and str(value).strip() else None


def _integer(value: Any) -> int | None:
    return int(value) if value is not None and str(value).strip() else None


def _venue_type(value: Any) -> str | None:
    venue_type = _string_or_none(value)
    if venue_type in {"conference", "journal", "preprint", "other"}:
        return venue_type
    return "other" if venue_type else None
