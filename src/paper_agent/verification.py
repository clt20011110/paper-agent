"""Coordinator-side metadata verification and venue membership decisions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Mapping

from .domain import (
    CollectionMembership,
    MembershipStatus,
    Paper,
    ProviderRole,
    PublicationVersion,
    SourceBatch,
    SourceEntry,
    VerificationStatus,
)
from .identity import normalize_arxiv_id, normalize_author, normalize_doi, normalize_title
from .repository import PaperRepository


@dataclass(frozen=True, slots=True)
class ProviderTrust:
    provider: str
    roles: frozenset[ProviderRole]
    independence_group: str
    upstream_families: frozenset[str]
    authority_rank: int = 3
    authority: str = "scholarly_graph"

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, Any]) -> ProviderTrust:
        authority = str(manifest["authority"])
        return cls(
            provider=str(manifest["provider"]),
            roles=frozenset(ProviderRole(role) for role in manifest["roles"]),
            independence_group=str(manifest["independence_group"]),
            upstream_families=frozenset(str(item) for item in manifest["upstream_families"]),
            authority_rank={
                "official": 0,
                "registry": 1,
                "domain_authority": 1,
                "scholarly_graph": 2,
                "oa_index": 3,
                "user": 4,
                "discovery_enhancement": 5,
            }[authority],
            authority=authority,
        )


@dataclass(frozen=True, slots=True)
class VerificationResult:
    status: VerificationStatus
    supporting_providers: tuple[str, ...]
    conflict_fields: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class VenueContext:
    collection_id: str
    venue_id: str
    name: str
    venue_type: str
    primary_provider: str
    descriptor: Mapping[str, Any]


def providers_are_independent(first: ProviderTrust, second: ProviderTrust) -> bool:
    return (
        first.independence_group != second.independence_group
        and first.upstream_families.isdisjoint(second.upstream_families)
    )


def _stable_id(entry: SourceEntry) -> tuple[str, str] | None:
    if doi := normalize_doi(entry.doi):
        return "doi", doi
    if arxiv_id := normalize_arxiv_id(entry.arxiv_id):
        return "arxiv", arxiv_id
    return None


def _core(entry: SourceEntry) -> tuple[tuple[str, str], str, str, int] | None:
    stable_id = _stable_id(entry)
    if stable_id is None or not entry.authors or entry.year is None:
        return None
    return stable_id, normalize_title(entry.title), normalize_author(entry.authors[0]), entry.year


def _field_values(entries: tuple[SourceEntry, ...]) -> dict[str, set[Any]]:
    return {
        "doi": {normalize_doi(entry.doi) for entry in entries if normalize_doi(entry.doi)},
        "arxiv_id": {normalize_arxiv_id(entry.arxiv_id) for entry in entries if normalize_arxiv_id(entry.arxiv_id)},
        "title": {normalize_title(entry.title) for entry in entries if entry.title},
        "first_author": {normalize_author(entry.authors[0]) for entry in entries if entry.authors},
        "year": {entry.year for entry in entries if entry.year is not None},
    }


class MetadataVerification:
    def __init__(self, trusts: Mapping[str, ProviderTrust]) -> None:
        self.trusts = trusts

    def verify(self, entries: tuple[SourceEntry, ...]) -> VerificationResult:
        if not entries:
            return VerificationResult(VerificationStatus.UNVERIFIED, ())

        conflicts = self._unresolved_conflicts(entries)
        if conflicts:
            return VerificationResult(
                VerificationStatus.CONFLICTED,
                tuple(sorted({entry.provider for entry in entries})),
                conflicts,
            )

        official = tuple(
            entry.provider
            for entry in entries
            if entry.provider in self.trusts
            and ProviderRole.VENUE_PRIMARY in self.trusts[entry.provider].roles
            and entry.metadata.get("official_membership") is True
            and _stable_id(entry) is not None
        )
        if official:
            return VerificationResult(VerificationStatus.VERIFIED, tuple(sorted(set(official))))

        for first, second in combinations(entries, 2):
            first_trust = self.trusts.get(first.provider)
            second_trust = self.trusts.get(second.provider)
            if (
                first_trust
                and second_trust
                and ProviderRole.METADATA_VERIFIER in first_trust.roles
                and ProviderRole.METADATA_VERIFIER in second_trust.roles
                and first_trust.authority != "discovery_enhancement"
                and second_trust.authority != "discovery_enhancement"
                and providers_are_independent(first_trust, second_trust)
                and _core(first) is not None
                and _core(first) == _core(second)
            ):
                return VerificationResult(
                    VerificationStatus.VERIFIED,
                    tuple(sorted((first.provider, second.provider))),
                )

        complete = tuple(entry.provider for entry in entries if _core(entry) is not None)
        if complete:
            return VerificationResult(VerificationStatus.SINGLE_SOURCE, tuple(sorted(set(complete))))
        return VerificationResult(VerificationStatus.UNVERIFIED, ())

    def _unresolved_conflicts(self, entries: tuple[SourceEntry, ...]) -> tuple[str, ...]:
        conflicting = {name for name, values in _field_values(entries).items() if len(values) > 1}
        unresolved: list[str] = []
        for field_name in conflicting:
            if field_name in {"doi", "arxiv_id"}:
                unresolved.append(field_name)
                continue
            ranked_values: list[tuple[int, Any]] = []
            for entry in entries:
                value = _entry_value(entry, field_name)
                if value is not None:
                    ranked_values.append((self.trusts.get(entry.provider, ProviderTrust(entry.provider, frozenset(), entry.provider, frozenset())).authority_rank, value))
            best_rank = min(rank for rank, _ in ranked_values)
            if len({value for rank, value in ranked_values if rank == best_rank}) > 1:
                unresolved.append(field_name)
        return tuple(name for name in ("doi", "arxiv_id", "title", "first_author", "year") if name in unresolved)


class MetadataCoordinator:
    """The only Stage 1 component that turns provider output into canonical rows."""

    def __init__(self, repository: PaperRepository, trusts: Mapping[str, ProviderTrust]) -> None:
        self.repository = repository
        self.trusts = trusts
        self.verification = MetadataVerification(trusts)

    def merge_batch(
        self,
        batch: SourceBatch,
        venue: VenueContext | None = None,
        *,
        candidate_only: bool = False,
    ) -> tuple[Paper, ...]:
        papers: dict[str, Paper] = {}
        if venue:
            self.repository.save_collection(
                venue.collection_id,
                venue.name,
                venue.venue_type,
                venue_id=venue.venue_id,
                descriptor=venue.descriptor,
            )

        for entry in batch.entries:
            paper = self.repository.ingest(entry)
            papers[paper.paper_id] = paper
            self._record_source_details(paper.paper_id, entry)
            if venue:
                self._record_membership(
                    paper.paper_id, entry, venue, candidate_only=candidate_only
                )

        for paper_id in papers:
            evidence = self._entries_for_paper(paper_id)
            result = self.verification.verify(evidence)
            self._apply_preferred_fields(paper_id, evidence)
            self.repository.connection.execute(
                "UPDATE papers SET verification_status = ?, updated_at = CURRENT_TIMESTAMP WHERE paper_id = ?",
                (result.status, paper_id),
            )
            if result.status is VerificationStatus.CONFLICTED:
                self.repository.enqueue_manual(
                    "merge_conflict",
                    f"verification:{paper_id}",
                    paper_id,
                    {"fields": result.conflict_fields, "providers": result.supporting_providers},
                )
        self.repository.connection.commit()
        return tuple(self.repository.get_paper(paper_id) for paper_id in sorted(papers))

    def _record_source_details(self, paper_id: str, entry: SourceEntry) -> None:
        metadata = entry.metadata
        publication_version = metadata.get("publication_version", PublicationVersion.UNKNOWN)
        self.repository.connection.execute(
            "UPDATE paper_sources SET publication_version = ? WHERE provider = ? AND external_id = ?",
            (PublicationVersion(publication_version), entry.provider, entry.external_id),
        )
        citation_count = metadata.get("citation_count")
        citation_count_as_of = metadata.get("citation_count_as_of")
        if isinstance(citation_count, int) and citation_count_as_of:
            self.repository.record_citation_count(
                paper_id,
                entry.provider,
                citation_count,
                str(citation_count_as_of),
            )

    def _record_membership(
        self,
        paper_id: str,
        entry: SourceEntry,
        venue: VenueContext,
        *,
        candidate_only: bool,
    ) -> None:
        current = self.repository.connection.execute(
            "SELECT membership_status, official_evidence_json FROM paper_collections WHERE paper_id = ? AND collection_id = ?",
            (paper_id, venue.collection_id),
        ).fetchone()
        official = (
            not candidate_only
            and entry.provider == venue.primary_provider
            and entry.metadata.get("official_membership") is True
            and entry.metadata.get("venue_id") == venue.venue_id
        )
        status = MembershipStatus.OFFICIAL_CONFIRMED if official else MembershipStatus.VENUE_CANDIDATE
        if current and current["membership_status"] == MembershipStatus.OFFICIAL_CONFIRMED:
            status = MembershipStatus.OFFICIAL_CONFIRMED
        evidence = (
            (f"{entry.provider}:{entry.external_id}",)
            if official
            else tuple(json.loads(current["official_evidence_json"] or "[]"))
            if current
            else ()
        )
        self.repository.upsert_membership(
            CollectionMembership(
                collection_id=venue.collection_id,
                paper_id=paper_id,
                membership_status=status,
                official_evidence=evidence,
            )
        )

    def _entries_for_paper(self, paper_id: str) -> tuple[SourceEntry, ...]:
        source_rows = self.repository.connection.execute(
            "SELECT source_id, provider, external_id, landing_url, raw_metadata_json FROM paper_sources WHERE paper_id = ?",
            (paper_id,),
        ).fetchall()
        entries: list[SourceEntry] = []
        for source in source_rows:
            provenance_rows = self.repository.connection.execute(
                "SELECT field_name, field_value_json FROM paper_field_provenance WHERE source_id = ?",
                (source["source_id"],),
            ).fetchall()
            fields = {row["field_name"]: json.loads(row["field_value_json"]) for row in provenance_rows}
            entries.append(
                SourceEntry(
                    provider=source["provider"],
                    external_id=source["external_id"],
                    title=fields["title"],
                    authors=tuple(fields.get("authors", ())),
                    abstract=fields.get("abstract"),
                    doi=fields.get("doi"),
                    arxiv_id=fields.get("arxiv_id"),
                    publication_date=fields.get("publication_date"),
                    year=fields.get("year"),
                    venue_name=fields.get("venue_name"),
                    landing_url=source["landing_url"],
                    metadata=json.loads(source["raw_metadata_json"]),
                )
            )
        return tuple(entries)

    def _apply_preferred_fields(self, paper_id: str, entries: tuple[SourceEntry, ...]) -> None:
        ranked = sorted(
            entries,
            key=lambda entry: (
                self.trusts.get(
                    entry.provider,
                    ProviderTrust(entry.provider, frozenset(), entry.provider, frozenset()),
                ).authority_rank,
                entry.provider,
            ),
        )
        selected: dict[str, Any] = {}
        for field_name in ("title", "abstract", "authors", "publication_date", "year", "venue_name", "landing_url"):
            for entry in ranked:
                value = getattr(entry, field_name)
                if value not in (None, (), ""):
                    selected[field_name] = value
                    break
        columns = {
            "title": selected.get("title"),
            "abstract": selected.get("abstract"),
            "authors_json": (
                json.dumps(selected["authors"], ensure_ascii=False, separators=(",", ":"))
                if "authors" in selected
                else None
            ),
            "publication_date": selected.get("publication_date"),
            "year": selected.get("year"),
            "venue_name": selected.get("venue_name"),
            "canonical_url": selected.get("landing_url"),
        }
        assignments = ", ".join(f"{name} = ?" for name, value in columns.items() if value is not None)
        values = [value for value in columns.values() if value is not None]
        if assignments:
            self.repository.connection.execute(
                f"UPDATE papers SET {assignments}, updated_at = CURRENT_TIMESTAMP WHERE paper_id = ?",
                (*values, paper_id),
            )


def _entry_value(entry: SourceEntry, field_name: str) -> Any:
    if field_name == "doi":
        return normalize_doi(entry.doi)
    if field_name == "arxiv_id":
        return normalize_arxiv_id(entry.arxiv_id)
    if field_name == "title":
        return normalize_title(entry.title)
    if field_name == "first_author":
        return normalize_author(entry.authors[0]) if entry.authors else None
    return entry.year
