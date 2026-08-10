"""Canonical citation edges and deterministic snowball search rounds."""

from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from typing import Mapping, Protocol, Sequence
from uuid import NAMESPACE_URL, uuid5

from .canonical import content_hash
from .domain import (
    CitationBatch,
    CitationEdge,
    CitationEdgeType,
    FilterStatus,
    VerificationStatus,
)
from .storage import Database


def reference_edge(
    seed_paper_id: str,
    referenced_paper_id: str,
    *,
    provider: str,
    observed_at: str,
    raw_evidence: Mapping[str, object],
) -> CitationEdge:
    """A backward result means the seed cites the returned paper."""
    return CitationEdge(
        source_paper_id=seed_paper_id,
        target_paper_id=referenced_paper_id,
        edge_type=CitationEdgeType.REFERENCES,
        provider=provider,
        observed_at=observed_at,
        raw_evidence=raw_evidence,
    )


def citation_edge(
    seed_paper_id: str,
    citing_paper_id: str,
    *,
    provider: str,
    observed_at: str,
    raw_evidence: Mapping[str, object],
) -> CitationEdge:
    """A forward result means the returned paper cites the seed."""
    return CitationEdge(
        source_paper_id=citing_paper_id,
        target_paper_id=seed_paper_id,
        edge_type=CitationEdgeType.CITATIONS,
        provider=provider,
        observed_at=observed_at,
        raw_evidence=raw_evidence,
    )


def version_edge(
    preprint_paper_id: str,
    published_paper_id: str,
    *,
    provider: str,
    observed_at: str,
    raw_evidence: Mapping[str, object],
) -> CitationEdge:
    return CitationEdge(
        source_paper_id=preprint_paper_id,
        target_paper_id=published_paper_id,
        edge_type=CitationEdgeType.VERSION_OF,
        provider=provider,
        observed_at=observed_at,
        raw_evidence=raw_evidence,
    )


class CitationRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def save(self, edge: CitationEdge) -> None:
        edge_id = f"citation-edge-{uuid5(NAMESPACE_URL, content_hash(edge.to_dict())).hex}"
        with self.database.transaction() as connection:
            connection.execute(
                """INSERT INTO citation_edges(
                    citation_edge_id, source_paper_id, target_paper_id, edge_type, provider,
                    observed_at, raw_evidence_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_paper_id, target_paper_id, edge_type, provider, observed_at)
                DO UPDATE SET raw_evidence_json = excluded.raw_evidence_json""",
                (
                    edge_id,
                    edge.source_paper_id,
                    edge.target_paper_id,
                    edge.edge_type,
                    edge.provider,
                    edge.observed_at,
                    json.dumps(edge.raw_evidence, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                ),
            )

    def depths(self, roots: Sequence[str]) -> dict[str, int]:
        adjacency: dict[str, set[str]] = defaultdict(set)
        rows = self.database.connection.execute(
            "SELECT source_paper_id, target_paper_id FROM citation_edges WHERE edge_type != 'version_of'"
        ).fetchall()
        for row in rows:
            adjacency[row["source_paper_id"]].add(row["target_paper_id"])
            adjacency[row["target_paper_id"]].add(row["source_paper_id"])

        depths = {paper_id: 0 for paper_id in roots}
        queue = deque(sorted(roots))
        while queue:
            paper_id = queue.popleft()
            for neighbor in sorted(adjacency[paper_id]):
                if neighbor not in depths:
                    depths[neighbor] = depths[paper_id] + 1
                    queue.append(neighbor)
        return depths


@dataclass(frozen=True, slots=True)
class SeedCandidate:
    paper_id: str
    subquestion_id: str | None
    status: FilterStatus
    reranker_score: float
    verification_status: VerificationStatus
    depth: int
    parent_round: int
    in_scope: bool = True


@dataclass(frozen=True, slots=True)
class SelectedSeed:
    paper_id: str
    seed_reason: str
    parent_round: int
    depth: int
    subquestion_id: str | None
    rank: int
    selector_version: str
    selector_config_hash: str


def select_seeds(
    candidates: Sequence[SeedCandidate],
    *,
    user_seed_ids: frozenset[str],
    expanded_paper_ids: frozenset[str],
    max_depth: int,
    per_subquestion: int,
    selector_version: str,
    selector_config_hash: str,
) -> tuple[SelectedSeed, ...]:
    eligible = {
        candidate.paper_id: candidate
        for candidate in candidates
        if candidate.in_scope
        and candidate.paper_id not in expanded_paper_ids
        and candidate.depth < max_depth
        and candidate.verification_status is not VerificationStatus.CONFLICTED
    }
    selected: list[tuple[SeedCandidate, str]] = []
    for paper_id in sorted(user_seed_ids):
        if paper_id in eligible:
            selected.append((eligible[paper_id], "user_seed"))

    by_subquestion: dict[str, list[SeedCandidate]] = defaultdict(list)
    for candidate in eligible.values():
        if (
            candidate.paper_id not in user_seed_ids
            and candidate.status is FilterStatus.RELEVANT
            and candidate.subquestion_id
        ):
            by_subquestion[candidate.subquestion_id].append(candidate)
    for subquestion_id in sorted(by_subquestion):
        ranked = sorted(
            by_subquestion[subquestion_id],
            key=lambda candidate: (-candidate.reranker_score, candidate.paper_id),
        )[:per_subquestion]
        selected.extend((candidate, "relevant_topk") for candidate in ranked)

    deduplicated: list[tuple[SeedCandidate, str]] = []
    seen: set[str] = set()
    for candidate, reason in selected:
        if candidate.paper_id not in seen:
            seen.add(candidate.paper_id)
            deduplicated.append((candidate, reason))
    return tuple(
        SelectedSeed(
            paper_id=candidate.paper_id,
            seed_reason=reason,
            parent_round=candidate.parent_round,
            depth=candidate.depth,
            subquestion_id=candidate.subquestion_id,
            rank=rank,
            selector_version=selector_version,
            selector_config_hash=selector_config_hash,
        )
        for rank, (candidate, reason) in enumerate(deduplicated)
    )


@dataclass(frozen=True, slots=True)
class CitationRequest:
    provider: str
    direction: CitationEdgeType
    seed_paper_id: str
    depth: int
    seed_rank: int
    schedule_order: int
    max_candidates: int


def schedule_requests(
    seeds: Sequence[SelectedSeed],
    *,
    providers: Sequence[str],
    directions: Sequence[CitationEdgeType],
    max_requests: int,
    max_candidates_per_request: int,
) -> tuple[CitationRequest, ...]:
    requests = [
        (provider, direction, seed)
        for provider in sorted(set(providers))
        for direction in sorted(set(directions), key=lambda item: item.value)
        for seed in seeds
    ]
    requests.sort(key=lambda item: (item[0], item[1].value, item[2].depth, item[2].rank, item[2].paper_id))
    return tuple(
        CitationRequest(
            provider=provider,
            direction=direction,
            seed_paper_id=seed.paper_id,
            depth=seed.depth + 1,
            seed_rank=seed.rank,
            schedule_order=order,
            max_candidates=max_candidates_per_request,
        )
        for order, (provider, direction, seed) in enumerate(requests[:max_requests])
    )


@dataclass(frozen=True, slots=True)
class RoundAudit:
    raw_discovered: int
    unique_after_dedup: int
    overlap: int
    screened_unique: int
    new_included_unique: int
    needs_review: int
    error_count: int
    edge_counts: Mapping[str, int] = field(default_factory=dict)
    source_stats: Mapping[str, Mapping[str, int]] = field(default_factory=dict)
    screening_complete: bool = True
    source_failed: bool = False

    @property
    def included_yield(self) -> float:
        return self.new_included_unique / max(1, self.screened_unique)


@dataclass(frozen=True, slots=True)
class StopDecision:
    stop: bool
    reason: str | None
    limited_scope: bool
    consecutive_low_yield_rounds: int


def decide_stop(
    audit: RoundAudit,
    *,
    previous_low_yield_rounds: int,
    min_unique_included_yield: float,
    required_low_yield_rounds: int,
    screening_complete: bool,
    sources_exhausted: bool,
    budget_exhausted: bool,
    source_failed: bool = False,
) -> StopDecision:
    low_yield_rounds = (
        previous_low_yield_rounds + 1
        if screening_complete and audit.included_yield < min_unique_included_yield
        else 0
    )
    if budget_exhausted:
        return StopDecision(True, "budget_exhausted", True, low_yield_rounds)
    if source_failed:
        # A failed graph source is neither an empty source nor evidence of
        # saturation.  Stop this replayable round as limited scope.
        return StopDecision(True, "saturated_with_unresolved", True, low_yield_rounds)
    saturated = sources_exhausted or low_yield_rounds >= required_low_yield_rounds
    if saturated and (not screening_complete or audit.needs_review):
        return StopDecision(True, "saturated_with_unresolved", True, low_yield_rounds)
    if sources_exhausted:
        return StopDecision(True, "sources_exhausted", False, low_yield_rounds)
    if saturated:
        return StopDecision(True, "saturated", False, low_yield_rounds)
    return StopDecision(False, None, False, low_yield_rounds)


class Screener(Protocol):
    def screen(self, paper_ids: Sequence[str]) -> Mapping[str, FilterStatus]: ...

    def reranker_score(self, paper_id: str) -> float: ...


class DeterministicFakeScreener:
    def __init__(self, relevant: frozenset[str], needs_review: frozenset[str] = frozenset()) -> None:
        self.relevant = relevant
        self.needs_review = needs_review
        self.screened: list[str] = []

    def screen(self, paper_ids: Sequence[str]) -> Mapping[str, FilterStatus]:
        self.screened.extend(paper_ids)
        return {
            paper_id: (
                FilterStatus.NEEDS_REVIEW
                if paper_id in self.needs_review
                else FilterStatus.RELEVANT
                if paper_id in self.relevant
                else FilterStatus.IRRELEVANT
            )
            for paper_id in paper_ids
        }

    def reranker_score(self, paper_id: str) -> float:
        return 1.0 if paper_id in self.relevant else 0.0


def process_citation_batches(
    batches: Sequence[CitationBatch],
    *,
    already_seen: frozenset[str],
    already_relevant: frozenset[str],
    screener: Screener,
) -> tuple[Mapping[str, FilterStatus], RoundAudit]:
    candidates: list[str] = []
    edge_counts: dict[str, int] = defaultdict(int)
    source_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"returned": 0, "errors": 0})
    for batch in batches:
        if batch.status.value == "failed" or batch.error:
            source_stats[batch.source_run_id]["errors"] += 1
        for edge in batch.entries:
            edge_counts[edge.edge_type.value] += 1
            source_stats[batch.source_run_id]["returned"] += 1
            candidate = (
                edge.target_paper_id
                if edge.edge_type is CitationEdgeType.REFERENCES
                else edge.source_paper_id
            )
            candidates.append(candidate)

    unique = tuple(sorted(set(candidates)))
    new_candidates = tuple(paper_id for paper_id in unique if paper_id not in already_seen)
    decisions = screener.screen(new_candidates)
    screening_complete = set(decisions) == set(new_candidates)
    newly_relevant = sum(
        status is FilterStatus.RELEVANT and paper_id not in already_relevant
        for paper_id, status in decisions.items()
    )
    audit = RoundAudit(
        raw_discovered=len(candidates),
        unique_after_dedup=len(unique),
        overlap=len(unique) - len(new_candidates),
        screened_unique=len(decisions),
        new_included_unique=newly_relevant,
        needs_review=sum(status is FilterStatus.NEEDS_REVIEW for status in decisions.values()),
        error_count=sum(stats["errors"] for stats in source_stats.values()),
        edge_counts=dict(edge_counts),
        source_stats={provider: dict(stats) for provider, stats in source_stats.items()},
        screening_complete=screening_complete and not any(
            status is FilterStatus.NEEDS_REVIEW for status in decisions.values()
        ),
        source_failed=any(batch.status.value == "failed" or batch.error for batch in batches),
    )
    return decisions, audit


class SearchRoundStore:
    def __init__(self, database: Database) -> None:
        self.database = database

    def freeze(
        self,
        *,
        crawl_run_id: str,
        round_index: int,
        seeds: Sequence[SelectedSeed],
        requests: Sequence[CitationRequest],
    ) -> str:
        round_id = f"search-round-{uuid5(NAMESPACE_URL, f'{crawl_run_id}:{round_index}').hex}"
        seed_hash = content_hash([asdict(seed) for seed in seeds])
        request_hash = content_hash(
            [
                {
                    "provider": request.provider,
                    "direction": request.direction.value,
                    "seed_paper_id": request.seed_paper_id,
                    "depth": request.depth,
                    "seed_rank": request.seed_rank,
                    "schedule_order": request.schedule_order,
                    "max_candidates": request.max_candidates,
                }
                for request in requests
            ]
        )
        existing = self.database.connection.execute(
            "SELECT seed_manifest_hash, request_schedule_hash FROM search_rounds WHERE search_round_id = ?",
            (round_id,),
        ).fetchone()
        if existing:
            if (existing["seed_manifest_hash"], existing["request_schedule_hash"]) != (seed_hash, request_hash):
                raise ValueError("frozen search round has drifted")
            return round_id

        with self.database.transaction() as connection:
            connection.execute(
                """INSERT INTO search_rounds(
                    search_round_id, crawl_run_id, round_index, state, seed_manifest_hash, request_schedule_hash
                ) VALUES (?, ?, ?, 'planned', ?, ?)""",
                (round_id, crawl_run_id, round_index, seed_hash, request_hash),
            )
            for seed in seeds:
                connection.execute(
                    """INSERT INTO search_round_seeds(
                        search_round_id, paper_id, seed_reason, parent_round, depth, subquestion_id,
                        seed_rank, selector_version, selector_config_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        round_id,
                        seed.paper_id,
                        seed.seed_reason,
                        seed.parent_round,
                        seed.depth,
                        seed.subquestion_id,
                        seed.rank,
                        seed.selector_version,
                        seed.selector_config_hash,
                    ),
                )
            for request in requests:
                request_id = f"citation-request-{uuid5(NAMESPACE_URL, f'{round_id}:{request.schedule_order}').hex}"
                connection.execute(
                    """INSERT INTO citation_requests(
                        citation_request_id, search_round_id, provider, direction, seed_paper_id,
                        depth, seed_rank, schedule_order, max_candidates, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'planned')""",
                    (
                        request_id,
                        round_id,
                        request.provider,
                        request.direction,
                        request.seed_paper_id,
                        request.depth,
                        request.seed_rank,
                        request.schedule_order,
                        request.max_candidates,
                    ),
                )
        return round_id

    def audit(self, round_id: str, audit: RoundAudit, decision: StopDecision, *, audited_at: str) -> None:
        with self.database.transaction() as connection:
            connection.execute(
                """INSERT INTO search_round_audits(
                    search_round_id, raw_discovered, unique_after_dedup, overlap, screened_unique,
                    new_included_unique, needs_review, error_count, edge_counts_json,
                    source_stats_json, screening_complete, source_failed, audited_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    round_id,
                    audit.raw_discovered,
                    audit.unique_after_dedup,
                    audit.overlap,
                    audit.screened_unique,
                    audit.new_included_unique,
                    audit.needs_review,
                    audit.error_count,
                    json.dumps(audit.edge_counts, sort_keys=True, separators=(",", ":")),
                    json.dumps(audit.source_stats, sort_keys=True, separators=(",", ":")),
                    int(audit.screening_complete),
                    int(audit.source_failed),
                    audited_at,
                ),
            )
            connection.execute(
                """UPDATE search_rounds SET state = ?, stop_reason = ?, limited_scope = ?, stats_json = ?,
                    completed_at = ? WHERE search_round_id = ?""",
                (
                    "stopped" if decision.stop else "complete",
                    decision.reason,
                    int(decision.limited_scope),
                    json.dumps(
                        {"consecutive_low_yield_rounds": decision.consecutive_low_yield_rounds},
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    audited_at,
                    round_id,
                ),
            )
