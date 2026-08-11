"""Canonical SQLite persistence for papers and their source records."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

from .domain import CollectionMembership, MembershipStatus, Paper, PaperSource, SourceEntry
from .identity import (
    manual_queue_id,
    normalize_arxiv_id,
    normalize_doi,
    paper_id_for,
    provider_external_key,
    source_id_for,
    title_author_year_key,
)
from .storage import Database


_PAPER_COLUMNS = (
    "paper_id", "title", "abstract", "authors_json", "keywords_json", "publication_date", "year",
    "venue_id", "venue_name", "venue_type", "doi", "arxiv_id", "canonical_url", "volume", "issue",
    "pages", "verification_status", "created_at", "updated_at",
)
_PROVENANCE_FIELDS = (
    "title", "abstract", "authors", "publication_date", "year", "venue_name", "doi", "arxiv_id",
    "canonical_url",
)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _paper_from_row(row: Any) -> Paper:
    value = dict(row)
    value["authors"] = tuple(json.loads(value.pop("authors_json")))
    value["keywords"] = tuple(json.loads(value.pop("keywords_json")))
    return Paper.from_dict(value)


class PaperRepository:
    """The coordinator-owned canonical paper store.

    Stable identifiers decide automatic identity.  A title/author/year match is
    deliberately only a manual-review candidate.
    """

    def __init__(self, database: Database) -> None:
        self.database = database
        self.connection = database.connection

    def get_paper(self, paper_id: str) -> Paper | None:
        row = self.connection.execute(
            f"SELECT {', '.join(_PAPER_COLUMNS)} FROM papers WHERE paper_id = ?", (paper_id,)
        ).fetchone()
        return _paper_from_row(row) if row else None

    def find_paper(
        self,
        *,
        doi: str | None = None,
        arxiv_id: str | None = None,
        provider: str | None = None,
        external_id: str | None = None,
    ) -> Paper | None:
        if normalized_doi := normalize_doi(doi):
            row = self.connection.execute(
                f"SELECT {', '.join(_PAPER_COLUMNS)} FROM papers WHERE doi = ?", (normalized_doi,)
            ).fetchone()
            if row:
                return _paper_from_row(row)
        if normalized_arxiv := normalize_arxiv_id(arxiv_id):
            row = self.connection.execute(
                f"SELECT {', '.join(_PAPER_COLUMNS)} FROM papers WHERE arxiv_id = ?", (normalized_arxiv,)
            ).fetchone()
            if row:
                return _paper_from_row(row)
        if provider and external_id:
            row = self.connection.execute(
                f"""SELECT {', '.join(f'p.{column}' for column in _PAPER_COLUMNS)}
                    FROM papers p JOIN paper_sources s ON s.paper_id = p.paper_id
                    WHERE s.provider = ? AND s.external_id = ?""",
                (provider.strip().lower(), external_id.strip()),
            ).fetchone()
            if row:
                return _paper_from_row(row)
        return None

    def save_paper(self, paper: Paper) -> Paper:
        """Insert a canonical paper once; distinct incoming values stay in provenance."""
        normalized = Paper(
            **{
                **paper.to_dict(),
                "doi": normalize_doi(paper.doi),
                "arxiv_id": normalize_arxiv_id(paper.arxiv_id),
            }
        )
        existing = self.get_paper(normalized.paper_id)
        if existing:
            return existing
        with self.database.transaction() as connection:
            connection.execute(
                """INSERT INTO papers(
                    paper_id, title, abstract, authors_json, keywords_json, publication_date, year,
                    venue_id, venue_name, venue_type, doi, arxiv_id, canonical_url, volume, issue,
                    pages, verification_status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP), COALESCE(?, CURRENT_TIMESTAMP))""",
                (
                    normalized.paper_id, normalized.title, normalized.abstract, _json(normalized.authors),
                    _json(normalized.keywords), normalized.publication_date, normalized.year, normalized.venue_id,
                    normalized.venue_name, normalized.venue_type, normalized.doi, normalized.arxiv_id,
                    normalized.canonical_url, normalized.volume, normalized.issue, normalized.pages,
                    normalized.verification_status, normalized.created_at, normalized.updated_at,
                ),
            )
        return self.get_paper(normalized.paper_id) or normalized

    def upsert_source(self, source: PaperSource) -> PaperSource:
        """Upsert one provider record without ever reassigning it to another paper."""
        provider = source.provider.strip().lower()
        external_id = source.external_id.strip()
        existing = self.connection.execute(
            "SELECT paper_id FROM paper_sources WHERE provider = ? AND external_id = ?", (provider, external_id)
        ).fetchone()
        if existing and existing["paper_id"] != source.paper_id:
            self.enqueue_manual(
                "merge_conflict",
                provider_external_key(provider, external_id),
                source.paper_id,
                {"existing_paper_id": existing["paper_id"], "provider": provider, "external_id": external_id},
            )
            raise ValueError("provider external ID is already bound to another paper")

        with self.database.transaction() as connection:
            connection.execute(
                """INSERT INTO paper_sources(
                    source_id, paper_id, provider, external_id, landing_url, pdf_url, metadata_url, bibtex,
                    citation_count, citation_count_as_of, publication_version, license, host_type, access_basis,
                    raw_metadata_json, metadata_capabilities_json, download_capabilities_json, first_seen_at,
                    last_seen_at, source_updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    COALESCE(?, CURRENT_TIMESTAMP), COALESCE(?, CURRENT_TIMESTAMP), ?)
                ON CONFLICT(provider, external_id) DO UPDATE SET
                    landing_url = excluded.landing_url, pdf_url = excluded.pdf_url,
                    metadata_url = excluded.metadata_url, bibtex = excluded.bibtex,
                    citation_count = excluded.citation_count, citation_count_as_of = excluded.citation_count_as_of,
                    publication_version = excluded.publication_version, license = excluded.license,
                    host_type = excluded.host_type, access_basis = excluded.access_basis,
                    raw_metadata_json = excluded.raw_metadata_json,
                    metadata_capabilities_json = excluded.metadata_capabilities_json,
                    download_capabilities_json = excluded.download_capabilities_json,
                    last_seen_at = excluded.last_seen_at, source_updated_at = excluded.source_updated_at""",
                (
                    source.source_id, source.paper_id, provider, external_id, source.landing_url, source.pdf_url,
                    source.metadata_url, source.bibtex, source.citation_count, source.citation_count_as_of,
                    source.publication_version, source.license, source.host_type, source.access_basis,
                    _json(source.raw_metadata), _json(source.metadata_capabilities),
                    _json(source.download_capabilities), source.first_seen_at, source.last_seen_at,
                    source.source_updated_at,
                ),
            )
        row = self.connection.execute(
            "SELECT * FROM paper_sources WHERE provider = ? AND external_id = ?", (provider, external_id)
        ).fetchone()
        return PaperSource.from_dict(
            {
                **dict(row),
                "raw_metadata": json.loads(row["raw_metadata_json"]),
                "metadata_capabilities": json.loads(row["metadata_capabilities_json"]),
                "download_capabilities": json.loads(row["download_capabilities_json"]),
            }
        )

    def record_field_provenance(self, paper_id: str, source_id: str, fields: Mapping[str, Any]) -> None:
        with self.database.transaction() as connection:
            for field_name, field_value in fields.items():
                if field_value is not None:
                    connection.execute(
                        """INSERT INTO paper_field_provenance(provenance_id, paper_id, source_id, field_name, field_value_json)
                        VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT(paper_id, source_id, field_name) DO UPDATE SET
                            field_value_json = excluded.field_value_json, observed_at = CURRENT_TIMESTAMP""",
                        (
                            f"provenance-{source_id}-{field_name}", paper_id, source_id, field_name,
                            _json(field_value),
                        ),
                    )

    def record_citation_count(self, paper_id: str, provider: str, count: int, observed_at: str) -> None:
        with self.database.transaction() as connection:
            connection.execute(
                """INSERT INTO citation_counts(citation_count_id, paper_id, provider, count, observed_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(paper_id, provider, observed_at) DO UPDATE SET count = excluded.count""",
                (f"citation-{paper_id}-{provider}-{observed_at}", paper_id, provider.strip().lower(), count, observed_at),
            )

    def save_collection(
        self, collection_id: str, name: str, collection_type: str, *, venue_id: str | None = None,
        descriptor: Mapping[str, Any] | None = None,
    ) -> None:
        with self.database.transaction() as connection:
            connection.execute(
                """INSERT INTO collections(collection_id, name, collection_type, venue_id, descriptor_json)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(collection_id) DO UPDATE SET name = excluded.name, collection_type = excluded.collection_type,
                    venue_id = excluded.venue_id, descriptor_json = excluded.descriptor_json""",
                (collection_id, name, collection_type, venue_id, _json(descriptor or {})),
            )

    def upsert_membership(self, membership: CollectionMembership) -> None:
        with self.database.transaction() as connection:
            connection.execute(
                """INSERT INTO paper_collections(
                    paper_id, collection_id, membership_status, official_evidence_json, observed_at
                ) VALUES (?, ?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP))
                ON CONFLICT(paper_id, collection_id) DO UPDATE SET
                    membership_status = excluded.membership_status,
                    official_evidence_json = excluded.official_evidence_json,
                    observed_at = excluded.observed_at""",
                (
                    membership.paper_id, membership.collection_id, membership.membership_status,
                    _json(membership.official_evidence), membership.observed_at,
                ),
            )

    def arxiv_candidates(self) -> tuple[Paper, ...]:
        rows = self.connection.execute(
            f"""SELECT DISTINCT {', '.join(f'p.{column}' for column in _PAPER_COLUMNS)}
                FROM papers p JOIN paper_collections pc ON pc.paper_id = p.paper_id
                JOIN collections c ON c.collection_id = pc.collection_id
                WHERE c.collection_type = 'arxiv' AND pc.membership_status != ? ORDER BY p.paper_id""",
            (MembershipStatus.NOT_MEMBER,),
        ).fetchall()
        return tuple(_paper_from_row(row) for row in rows)

    list_arxiv_candidates = arxiv_candidates

    def enqueue_manual(
        self, queue_type: str, dedup_key: str, paper_id: str | None, reason: Mapping[str, Any]
    ) -> None:
        with self.database.transaction() as connection:
            connection.execute(
                """INSERT INTO manual_queue(manual_queue_id, queue_type, dedup_key, paper_id, reason_json, status)
                VALUES (?, ?, ?, ?, ?, 'pending') ON CONFLICT(queue_type, dedup_key) DO NOTHING""",
                (manual_queue_id(queue_type, dedup_key), queue_type, dedup_key, paper_id, _json(reason)),
            )

    def resolve_manual(
        self,
        queue_type: str,
        dedup_key: str,
        resolution: Mapping[str, Any],
        *,
        resolved_at: str,
    ) -> None:
        with self.database.transaction() as connection:
            connection.execute(
                """UPDATE manual_queue
                   SET status = 'resolved', resolution_json = ?, resolved_at = ?
                   WHERE queue_type = ? AND dedup_key = ? AND status = 'pending'""",
                (_json(resolution), resolved_at, queue_type, dedup_key),
            )

    def ingest(self, entry: SourceEntry) -> Paper:
        """Save a source entry under a stable identity, preserving fuzzy duplicates for review."""
        existing = self.find_paper(
            doi=entry.doi, arxiv_id=entry.arxiv_id, provider=entry.provider, external_id=entry.external_id
        )
        incoming = Paper(
            paper_id=paper_id_for(
                doi=entry.doi, arxiv_id=entry.arxiv_id, provider=entry.provider, external_id=entry.external_id
            ),
            title=entry.title,
            abstract=entry.abstract,
            authors=entry.authors,
            publication_date=entry.publication_date,
            year=entry.year,
            venue_name=entry.venue_name,
            doi=normalize_doi(entry.doi),
            arxiv_id=normalize_arxiv_id(entry.arxiv_id),
            canonical_url=entry.landing_url,
        )
        paper = self._fill_missing(existing, incoming) if existing else self.save_paper(incoming)
        source = self.upsert_source(
            PaperSource(
                source_id=source_id_for(entry.provider, entry.external_id), paper_id=paper.paper_id,
                provider=entry.provider, external_id=entry.external_id, landing_url=entry.landing_url,
                pdf_url=entry.pdf_url, publication_version=entry.publication_version,
                license=entry.license, host_type=entry.host_type, access_basis=entry.access_basis,
                raw_metadata=entry.metadata,
            )
        )
        source_fields = {
            "title": entry.title, "abstract": entry.abstract, "authors": entry.authors,
            "publication_date": entry.publication_date, "year": entry.year, "venue_name": entry.venue_name,
            "doi": normalize_doi(entry.doi), "arxiv_id": normalize_arxiv_id(entry.arxiv_id),
            "canonical_url": entry.landing_url,
        }
        self.record_field_provenance(paper.paper_id, source.source_id, source_fields)
        self._queue_conflicts(paper, source.source_id, source_fields)
        if not existing:
            self._queue_title_candidate(paper, entry)
        return paper

    save_source_entry = ingest

    def _queue_conflicts(self, paper: Paper, source_id: str, fields: Mapping[str, Any]) -> None:
        conflicts = {
            name: {"canonical": getattr(paper, name), "incoming": value}
            for name, value in fields.items()
            if value is not None and getattr(paper, name) not in (None, (), "") and getattr(paper, name) != value
        }
        if conflicts:
            self.enqueue_manual(
                "merge_conflict",
                f"paper:{paper.paper_id}:source:{source_id}",
                paper.paper_id,
                {"source_id": source_id, "fields": conflicts},
            )

    def _fill_missing(self, existing: Paper, incoming: Paper) -> Paper:
        values = {
            "abstract": incoming.abstract,
            "authors_json": _json(incoming.authors) if incoming.authors else None,
            "publication_date": incoming.publication_date,
            "year": incoming.year,
            "venue_name": incoming.venue_name,
            "doi": incoming.doi,
            "arxiv_id": incoming.arxiv_id,
            "canonical_url": incoming.canonical_url,
        }
        updates = {
            name: value
            for name, value in values.items()
            if value is not None
            and (
                (name == "authors_json" and not existing.authors)
                or (name != "authors_json" and getattr(existing, name) is None)
            )
        }
        if updates:
            assignments = ", ".join(f"{name} = ?" for name in updates)
            with self.database.transaction() as connection:
                connection.execute(
                    f"UPDATE papers SET {assignments}, updated_at = CURRENT_TIMESTAMP WHERE paper_id = ?",
                    (*updates.values(), existing.paper_id),
                )
        return self.get_paper(existing.paper_id) or existing

    def _queue_title_candidate(self, paper: Paper, entry: SourceEntry) -> None:
        dedup_key = title_author_year_key(entry.title, entry.authors, entry.year)
        rows = self.connection.execute(
            "SELECT paper_id, title, authors_json, year FROM papers WHERE paper_id != ?", (paper.paper_id,)
        ).fetchall()
        candidates = [
            row["paper_id"]
            for row in rows
            if title_author_year_key(row["title"], tuple(json.loads(row["authors_json"])), row["year"]) == dedup_key
        ]
        if candidates:
            self.enqueue_manual(
                "dedup", f"title-author-year:{dedup_key}", paper.paper_id,
                {"candidate_paper_ids": sorted(candidates), "incoming_paper_id": paper.paper_id},
            )


Repository = PaperRepository
