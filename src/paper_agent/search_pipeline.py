"""Deterministic coordinator for the read-only Phase 2 search pipeline.

Providers only return envelopes.  This module is the single writer that
records those envelopes, normalizes their entries, and expands citations.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence
from uuid import NAMESPACE_URL, uuid5

from .citations import (
    CitationRepository,
    CitationRequest,
    DeterministicFakeScreener,
    RoundAudit,
    SearchRoundStore,
    SeedCandidate,
    StopDecision,
    process_citation_batches,
    schedule_requests,
    select_seeds,
    version_edge,
)
from .domain import (
    CitationBatch,
    CitationEdgeType,
    EnvelopeStatus,
    FilterStatus,
    MembershipStatus,
    PublicationVersion,
    SourceBatch,
    SourceEntry,
    VerificationStatus,
)
from .fanout import (
    FanoutResult,
    ProviderOutcome,
    ProviderPage,
    fan_out,
    search_pages,
    venue_pages,
)
from .identity import normalize_author, normalize_doi, normalize_title
from .query_compilers import NativeQuery, compile_queries
from .query_plan import assert_runtime_matches
from .repository import PaperRepository
from .providers.api import CrawlWindow, SeedInput, VenueDescriptor
from .search_runs import SearchRunCoordinator, SourceMetrics
from .storage import Database
from .verification import MetadataCoordinator, ProviderTrust, VenueContext


@dataclass(frozen=True, slots=True)
class PipelineResult:
    crawl_run_id: str
    status: str
    paper_ids: tuple[str, ...]
    arxiv_candidate_ids: tuple[str, ...]
    fanout: FanoutResult
    citation_round_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class VenueRun:
    descriptor: VenueDescriptor
    window: CrawlWindow
    context: VenueContext


class SearchPipeline:
    """Run one approved QueryPlan without letting providers write canonical data."""

    def __init__(
        self,
        database: Database,
        plan: Mapping[str, Any],
        *,
        runtime_providers: Sequence[Mapping[str, Any]] | None = None,
        clients: Mapping[str, Any],
        trusts: Mapping[str, ProviderTrust],
        venue: VenueContext | None = None,
        venue_runs: Sequence[VenueRun] = (),
        seed_inputs: Sequence[SeedInput] = (),
        citation_clients: Mapping[str, Any] | None = None,
        screener: Any | None = None,
    ) -> None:
        self.database = database
        self.plan = dict(plan)
        self.runtime_providers = tuple(dict(item) for item in (runtime_providers or plan["providers"]))
        self.clients = clients
        self.trusts = trusts
        self.venue = venue
        self.venue_runs = tuple(venue_runs)
        self.seed_inputs = tuple(seed_inputs)
        self.citation_clients = citation_clients or {}
        self.screener = screener or DeterministicFakeScreener(frozenset())
        self.repository = PaperRepository(database)
        self.metadata = MetadataCoordinator(self.repository, trusts)
        self.runs = SearchRunCoordinator(database)
        self.citations = CitationRepository(database)
        self.rounds = SearchRoundStore(database)

    def run(
        self,
        *,
        run_id: str,
        crawl_run_id: str,
        observed_at: str,
        seed_paper_ids: Sequence[str] = (),
    ) -> PipelineResult:
        """Execute the frozen plan and persist a replayable audit trail."""
        assert_runtime_matches(self.plan, self.runtime_providers, budgets=self.plan["budgets"])
        self._ensure_run(run_id)
        self.runs.start_crawl(
            crawl_run_id=crawl_run_id,
            run_id=run_id,
            search_plan_id=str(self.plan["plan_id"]),
            window=dict(self.plan["scope"]),
        )

        fanout = fan_out(self.plan, self._execution_clients())
        all_paper_ids: set[str] = set()
        non_arxiv_ids: set[str] = set()
        library_seed_ids: set[str] = set()
        metrics: dict[str, SourceMetrics] = {}
        source_entries: dict[str, list[SourceEntry]] = {}
        for outcome in fanout.outcomes:
            provider = self._provider(outcome.provider)
            queries = self._queries(provider)
            pages = self._source_pages(crawl_run_id, provider, outcome)
            for page in pages:
                query = page.query or next(
                    (item for item in queries if item.query_hash == page.batch.query_hash), None
                )
                scope = page.scope_id or (query.variant_id if query else "default")
                batch = replace(
                    page.batch,
                    source_run_id=f"{crawl_run_id}:{outcome.provider}:{page.role}:{scope}",
                )
                if page.role == "search" and queries and query is None:
                    raise ValueError(f"provider {outcome.provider} returned an unfrozen query hash")
                self.runs.record_batch(
                    crawl_run_id=crawl_run_id,
                    provider=outcome.provider,
                    provider_version=str(provider["version"]),
                    role=page.role,
                    query_text=self._query_text(query),
                    provider_params=dict(query.parameters) if query else {},
                    query_compiler_version=str(provider["query_compiler_version"]),
                    batch=batch,
                    requested_at=observed_at,
                    completed_at=observed_at,
                    page=str(page.page),
                    cursor=page.cursor,
                    alias_group=query.variant_id if query else None,
                    filters=dict(self.plan["scope"]),
                )
                prior = metrics.get(batch.source_run_id, SourceMetrics())
                source_entries.setdefault(batch.source_run_id, []).extend(batch.entries)
                metrics[batch.source_run_id] = SourceMetrics(
                    raw_discovered=prior.raw_discovered + len(batch.entries),
                    unique_after_dedup=prior.unique_after_dedup,
                    overlap=prior.overlap,
                    screened=prior.screened,
                    excluded=prior.excluded,
                    included=prior.included,
                    full_text_available=prior.full_text_available,
                    error_count=prior.error_count + int(batch.status is EnvelopeStatus.FAILED),
                )
                if batch.status is EnvelopeStatus.FAILED:
                    continue
                venue = (
                    self._arxiv_context()
                    if outcome.provider == "arxiv"
                    else self._venue_context(page.scope_id)
                )
                papers = self.metadata.merge_batch(batch, venue)
                ids = {paper.paper_id for paper in papers}
                all_paper_ids.update(ids)
                if outcome.provider != "arxiv":
                    non_arxiv_ids.update(ids)
                if page.role == "library":
                    library_seed_ids.update(ids)

        for source_run_id, source_metrics in metrics.items():
            entries = source_entries[source_run_id]
            identities = {self._identity(entry) for entry in entries}
            raw = source_metrics.raw_discovered
            self.runs.record_metrics(
                source_run_id,
                replace(
                    source_metrics,
                    unique_after_dedup=len(identities),
                    overlap=raw - len(identities),
                ),
                updated_at=observed_at,
            )

        self._link_versions(observed_at)
        status = self.runs.finish_crawl(crawl_run_id, plan=self.plan, fanout=fanout, finished_at=observed_at)
        round_ids = self._run_citations(
            crawl_run_id,
            observed_at,
            (*seed_paper_ids, *sorted(library_seed_ids)),
        )
        citation_limited = self.database.connection.execute(
            "SELECT 1 FROM search_rounds WHERE crawl_run_id = ? AND limited_scope = 1 LIMIT 1",
            (crawl_run_id,),
        ).fetchone()
        if citation_limited:
            status = "incomplete"
            self.database.connection.execute(
                "UPDATE crawl_runs SET status = 'incomplete' WHERE crawl_run_id = ?",
                (crawl_run_id,),
            )
        self.database.connection.execute(
            "UPDATE pipeline_runs SET status = ?, completed_at = ? WHERE run_id = ?",
            (status, observed_at, run_id),
        )
        self.database.connection.commit()
        return PipelineResult(
            crawl_run_id,
            status,
            tuple(sorted(non_arxiv_ids)),
            tuple(sorted(all_paper_ids - non_arxiv_ids)),
            fanout,
            tuple(round_ids),
        )

    execute = run

    def _ensure_run(self, run_id: str) -> None:
        plan_id = str(self.plan["plan_id"])
        self.database.connection.execute(
            """INSERT INTO search_plans(search_plan_id, content_hash, schema_version, plan_json, approval_json, status)
               VALUES (?, ?, ?, ?, ?, 'approved') ON CONFLICT(search_plan_id) DO NOTHING""",
            (
                plan_id,
                str(self.plan["plan_hash"]),
                str(self.plan["schema_version"]),
                json.dumps(self.plan, sort_keys=True, separators=(",", ":")),
                json.dumps(self.plan["approval"], sort_keys=True, separators=(",", ":")),
            ),
        )
        self.database.connection.execute(
            """INSERT INTO pipeline_runs(run_id, stage, status, input_hash, config_hash, implementation_version, started_at)
               VALUES (?, 'stage-1', 'running', ?, ?, 'phase2-search-v1', ?)
               ON CONFLICT(run_id) DO NOTHING""",
            (run_id, str(self.plan["plan_hash"]), str(self.plan["filter"]["config_hash"]), self.plan["created_at"]),
        )
        self.database.connection.commit()

    def _provider(self, name: str) -> Mapping[str, Any]:
        return next(item for item in self.plan["providers"] if item["provider"] == name)

    def _execution_clients(self) -> dict[str, Any]:
        clients: dict[str, Any] = {}
        for provider in self.plan["providers"]:
            name = str(provider["provider"])
            client = self.clients.get(name)
            runs = tuple(run for run in self.venue_runs if run.descriptor.provider == name)
            if client is None or callable(client) or (not runs and "library" not in provider["roles"]):
                clients[name] = client
                continue

            def invoke(
                specification: Mapping[str, Any],
                queries: tuple[NativeQuery, ...],
                *,
                protocol_client: Any = client,
                venue_work: tuple[VenueRun, ...] = runs,
            ) -> tuple[ProviderPage, ...]:
                pages = list(search_pages(protocol_client, specification, queries)) if queries else []
                for venue_run in venue_work:
                    pages.extend(
                        venue_pages(protocol_client, venue_run.descriptor, venue_run.window)
                    )
                if "library" in specification["roles"] and self.seed_inputs:
                    batch = protocol_client.import_seeds(self.seed_inputs)
                    pages.append(ProviderPage("library", batch, None, 1, None))
                return tuple(pages)

            clients[name] = invoke
        return clients

    def _venue_context(self, venue_id: str | None) -> VenueContext | None:
        if venue_id is None:
            return self.venue
        return next(
            (run.context for run in self.venue_runs if run.descriptor.venue_id == venue_id),
            self.venue,
        )

    def _queries(self, provider: Mapping[str, Any]) -> tuple[NativeQuery, ...]:
        if "search" not in provider["roles"]:
            return ()
        return compile_queries(
            str(provider["provider"]),
            self.plan["query_variants"],
            self.plan["scope"],
            page_size=int(self.plan.get("page_size", 100)),
        )

    def _source_pages(
        self, crawl_run_id: str, provider: Mapping[str, Any], outcome: ProviderOutcome
    ) -> tuple[ProviderPage, ...]:
        queries = self._queries(provider)
        role = (
            "search"
            if queries
            else "venue_primary"
            if "venue_primary" in provider["roles"]
            else str(provider["roles"][0])
        )
        if outcome.status == "failed":
            return tuple(
                ProviderPage(
                    role,
                    SourceBatch(
                        f"{crawl_run_id}:{outcome.provider}:{role}",
                        query.query_hash,
                        (),
                        None,
                        EnvelopeStatus.FAILED,
                        outcome.error or "provider failed",
                    ),
                    query,
                    1,
                    None,
                )
                for query in queries
            ) or (
                ProviderPage(
                    role,
                    SourceBatch(
                        f"{crawl_run_id}:{outcome.provider}:{role}",
                        "no-query",
                        (),
                        None,
                        EnvelopeStatus.FAILED,
                        outcome.error or "provider failed",
                    ),
                    None,
                    1,
                    None,
                ),
            )
        values = self._flatten(outcome.result)
        pages = tuple(item for item in values if isinstance(item, ProviderPage))
        if pages:
            return pages
        batches = tuple(item for item in values if isinstance(item, SourceBatch))
        if batches:
            return tuple(
                ProviderPage(
                    role,
                    batch,
                    next((query for query in queries if query.query_hash == batch.query_hash), None),
                    index,
                    None,
                )
                for index, batch in enumerate(batches, start=1)
            )
        return tuple(
            ProviderPage(
                "search",
                SourceBatch(
                    f"{crawl_run_id}:{outcome.provider}:search",
                    query.query_hash,
                    (),
                    None,
                    EnvelopeStatus.SUCCESS,
                ),
                query,
                1,
                None,
            )
            for query in queries
        ) or (
            ProviderPage(
                role,
                SourceBatch(
                    f"{crawl_run_id}:{outcome.provider}:{role}",
                    "no-query",
                    (),
                    None,
                    EnvelopeStatus.SUCCESS,
                ),
                None,
                1,
                None,
            ),
        )

    @staticmethod
    def _flatten(value: Any) -> tuple[Any, ...]:
        if isinstance(value, (tuple, list)):
            return tuple(item for group in value for item in SearchPipeline._flatten(group))
        return (value,)

    @staticmethod
    def _query_text(query: NativeQuery | None) -> str:
        if query is None:
            return ""
        for key in ("query.bibliographic", "q", "query", "search", "term", "search_query"):
            if key in query.parameters:
                return str(query.parameters[key])
        return query.variant_id

    @staticmethod
    def _identity(entry: SourceEntry) -> tuple[str, ...]:
        if entry.doi:
            return ("doi", normalize_doi(entry.doi) or entry.doi)
        if entry.arxiv_id:
            return ("arxiv", entry.arxiv_id)
        return (entry.provider, entry.external_id)

    def _arxiv_context(self) -> VenueContext:
        return VenueContext(
            "arxiv_candidates", "arxiv", "arXiv candidates", "arxiv", "arxiv", {"kind": "candidate"}
        )

    def _link_versions(self, observed_at: str) -> None:
        rows = self.database.connection.execute(
            """SELECT p.paper_id, p.title, p.authors_json, p.year, s.publication_version
               FROM papers p JOIN paper_sources s ON s.paper_id = p.paper_id
               WHERE s.publication_version IN ('preprint', 'published') ORDER BY p.paper_id"""
        ).fetchall()
        published: dict[tuple[str, str, int | None], str] = {}
        preprints: list[tuple[str, tuple[str, str, int | None]]] = []
        for row in rows:
            authors = tuple(json.loads(row["authors_json"]))
            key = (normalize_title(row["title"]), normalize_author(authors[0]) if authors else "", row["year"])
            if row["publication_version"] == PublicationVersion.PUBLISHED:
                published.setdefault(key, row["paper_id"])
            else:
                preprints.append((row["paper_id"], key))
        for preprint_id, key in preprints:
            published_id = published.get(key)
            if published_id and published_id != preprint_id:
                self.citations.save(version_edge(preprint_id, published_id, provider="metadata", observed_at=observed_at, raw_evidence={"match": "title-author-year"}))

    def _run_citations(
        self, crawl_run_id: str, observed_at: str, seed_paper_ids: Sequence[str]
    ) -> list[str]:
        config = self.plan["citation_snowball"]
        if not config["enabled"] or not self.citation_clients:
            return []
        providers = tuple(sorted(self.citation_clients))
        roots = tuple(sorted({paper_id for paper_id in seed_paper_ids if self.repository.get_paper(paper_id)}))
        if not roots:
            return []
        max_requests = int(self.plan["budgets"]["max_requests"])
        max_depth = int(config["max_depth"])
        max_rounds = int(config["max_rounds"])
        max_per_request = int(config["max_per_seed_per_source"])
        max_candidates = int(self.plan["budgets"]["max_candidates"])
        deadline = time.monotonic() + int(self.plan["budgets"]["max_seconds"])
        directions = tuple(CitationEdgeType(value) for value in config["directions"])
        default_subquestion = next(
            (item.get("subquestion_id") for item in self.plan["query_variants"] if item.get("subquestion_id")),
            None,
        )
        candidates = {
            paper_id: SeedCandidate(
                paper_id, default_subquestion, FilterStatus.RELEVANT,
                1.0, self.repository.get_paper(paper_id).verification_status, 0, 0,
            )
            for paper_id in roots
        }
        seen = set(roots)
        expanded: set[str] = set()
        used_requests = 0
        used_candidates = 0
        low_yield_rounds = 0
        round_ids: list[str] = []
        for round_index in range(max_rounds):
            seeds = select_seeds(
                tuple(candidates.values()),
                user_seed_ids=frozenset(roots),
                expanded_paper_ids=frozenset(expanded),
                max_depth=max_depth,
                per_subquestion=20,
                selector_version=str(self.plan["filter"]["seed_selector_version"]),
                selector_config_hash=str(self.plan["filter"]["seed_selector_config_hash"]),
            )
            if not seeds:
                break
            requests = schedule_requests(
                seeds,
                providers=providers,
                directions=directions,
                max_requests=max(0, max_requests - used_requests),
                max_candidates_per_request=max_per_request,
            )
            round_id = self.rounds.freeze(crawl_run_id=crawl_run_id, round_index=round_index, seeds=seeds, requests=requests)
            round_ids.append(round_id)
            batches, time_exhausted, requests_made = self._execute_citation_requests(
                round_id,
                requests,
                observed_at,
                request_budget=max(0, max_requests - used_requests),
                candidate_budget=max(0, max_candidates - used_candidates),
                deadline=deadline,
            )
            used_requests += requests_made
            used_candidates += sum(len(batch.entries) for batch in batches)
            batches = self._canonicalize_citation_batches(batches, requests, observed_at)
            decisions, audit = process_citation_batches(
                batches,
                already_seen=frozenset(seen),
                already_relevant=frozenset(
                    paper_id for paper_id, candidate in candidates.items() if candidate.status is FilterStatus.RELEVANT
                ),
                screener=self.screener,
            )
            candidate_context = self._candidate_contexts(batches, requests, candidates)
            self._save_round_papers(round_id, decisions, candidate_context)
            for paper_id, decision in decisions.items():
                seen.add(paper_id)
                paper = self.repository.get_paper(paper_id)
                if paper:
                    depth, subquestion_id = candidate_context[paper_id]
                    candidates[paper_id] = SeedCandidate(
                        paper_id, subquestion_id, decision, 1.0, paper.verification_status,
                        depth, round_index,
                    )
            expanded.update(seed.paper_id for seed in seeds)
            exhausted = bool(batches) and not any(batch.entries for batch in batches) and not audit.source_failed
            budget_exhausted = (
                used_requests >= max_requests
                or used_candidates >= max_candidates
                or time_exhausted
            )
            saturation = self.plan["budgets"]["saturation"]
            from .citations import decide_stop

            decision = decide_stop(
                audit,
                previous_low_yield_rounds=low_yield_rounds,
                min_unique_included_yield=float(saturation["min_unique_included_yield"]),
                required_low_yield_rounds=int(saturation["consecutive_low_yield_rounds"]),
                screening_complete=audit.screening_complete,
                sources_exhausted=exhausted,
                budget_exhausted=budget_exhausted,
                source_failed=audit.source_failed,
            )
            low_yield_rounds = decision.consecutive_low_yield_rounds
            if round_index + 1 == max_rounds and not decision.stop:
                decision = StopDecision(
                    True,
                    "max_rounds",
                    not audit.screening_complete or bool(audit.needs_review),
                    low_yield_rounds,
                )
            self._audit_round(round_id, audit, decision, observed_at)
            if decision.stop:
                break
        return round_ids

    def _execute_citation_requests(
        self,
        round_id: str,
        requests: Sequence[CitationRequest],
        observed_at: str,
        *,
        request_budget: int,
        candidate_budget: int,
        deadline: float,
    ) -> tuple[tuple[CitationBatch, ...], bool, int]:
        batches: list[CitationBatch] = []
        time_exhausted = False
        remaining = candidate_budget
        requests_made = 0
        for request in requests:
            if remaining <= 0 or requests_made >= request_budget or time.monotonic() >= deadline:
                time_exhausted = True
                self.database.connection.execute(
                    "UPDATE citation_requests SET status = 'skipped_budget', error_json = ? WHERE search_round_id = ? AND schedule_order = ?",
                    (json.dumps({"message": "candidate or time budget exhausted"}), round_id, request.schedule_order),
                )
                continue
            client = self.citation_clients[request.provider]
            paper = self.repository.get_paper(request.seed_paper_id)
            operation = client.references if request.direction is CitationEdgeType.REFERENCES else client.citations
            entries = []
            cursor = None
            seen_cursors: set[str] = set()
            status = EnvelopeStatus.SUCCESS
            error_message = None
            budget_cutoff = False
            query_hash = str(request.schedule_order)
            raw_response_artifact_hash = None
            while len(entries) < min(request.max_candidates, remaining):
                if requests_made >= request_budget or time.monotonic() >= deadline:
                    time_exhausted = True
                    budget_cutoff = True
                    break
                try:
                    page = operation(paper, cursor)
                except Exception as error:
                    page = CitationBatch(
                        f"{round_id}:{request.schedule_order}",
                        query_hash,
                        (),
                        None,
                        EnvelopeStatus.FAILED,
                        str(error),
                    )
                requests_made += 1
                query_hash = page.query_hash
                raw_response_artifact_hash = page.raw_response_artifact_hash
                capacity = min(request.max_candidates, remaining) - len(entries)
                entries.extend(
                    replace(
                        edge,
                        raw_evidence={
                            **edge.raw_evidence,
                            "raw_response_artifact_hash": page.raw_response_artifact_hash,
                        },
                    )
                    for edge in page.entries[:capacity]
                )
                if page.status is EnvelopeStatus.FAILED or page.error:
                    status = EnvelopeStatus.PARTIAL if entries else EnvelopeStatus.FAILED
                    error_message = page.error or "citation page failed"
                    break
                cursor = page.next_cursor
                if not cursor or len(entries) >= min(request.max_candidates, remaining):
                    break
                if cursor in seen_cursors:
                    status = EnvelopeStatus.PARTIAL if entries else EnvelopeStatus.FAILED
                    error_message = f"provider {request.provider} repeated cursor {cursor}"
                    break
                seen_cursors.add(cursor)
            batch = CitationBatch(
                source_run_id=f"{round_id}:{request.schedule_order}",
                query_hash=query_hash,
                entries=tuple(entries),
                next_cursor=cursor,
                status=status,
                error=error_message,
                raw_response_artifact_hash=raw_response_artifact_hash,
            )
            remaining -= len(batch.entries)
            batches.append(batch)
            self.database.connection.execute(
                "UPDATE citation_requests SET status = ?, error_json = ? WHERE search_round_id = ? AND schedule_order = ?",
                (
                    "skipped_budget"
                    if budget_cutoff
                    else "failed"
                    if batch.status is EnvelopeStatus.FAILED
                    else "complete",
                    json.dumps({"message": batch.error or "request budget exhausted"})
                    if batch.error or budget_cutoff
                    else None,
                    round_id,
                    request.schedule_order,
                ),
            )
        self.database.connection.commit()
        return tuple(batches), time_exhausted, requests_made

    def _canonicalize_citation_batches(
        self,
        batches: Sequence[CitationBatch],
        requests: Sequence[CitationRequest],
        observed_at: str,
    ) -> tuple[CitationBatch, ...]:
        request_by_order = {request.schedule_order: request for request in requests}
        canonical_batches: list[CitationBatch] = []
        for batch in batches:
            request = request_by_order[int(batch.source_run_id.rsplit(":", 1)[1])]
            canonical_edges = []
            invalid = 0
            for edge in batch.entries:
                candidate_paper = None
                if edge.candidate is not None:
                    candidate_paper = self.metadata.merge_batch(
                        SourceBatch(
                            batch.source_run_id,
                            batch.query_hash,
                            (edge.candidate,),
                            None,
                            EnvelopeStatus.SUCCESS,
                            raw_response_artifact_hash=batch.raw_response_artifact_hash,
                        )
                    )[0]
                else:
                    endpoint = edge.target_paper_id if request.direction is CitationEdgeType.REFERENCES else edge.source_paper_id
                    candidate_paper = self.repository.get_paper(endpoint)
                if candidate_paper is None:
                    invalid += 1
                    continue
                if request.direction is CitationEdgeType.REFERENCES:
                    source_id, target_id = request.seed_paper_id, candidate_paper.paper_id
                else:
                    source_id, target_id = candidate_paper.paper_id, request.seed_paper_id
                canonical = replace(
                    edge,
                    source_paper_id=source_id,
                    target_paper_id=target_id,
                    edge_type=request.direction,
                    observed_at=edge.observed_at or observed_at,
                    raw_evidence={
                        **edge.raw_evidence,
                        "provider_native_source_id": edge.source_paper_id,
                        "provider_native_target_id": edge.target_paper_id,
                        "citation_source_run_id": batch.source_run_id,
                        "raw_response_artifact_hash": batch.raw_response_artifact_hash,
                    },
                    candidate=None,
                )
                self.citations.save(canonical)
                canonical_edges.append(canonical)
            error = batch.error
            if invalid:
                error = "; ".join(item for item in (error, f"{invalid} citation candidates lacked canonical metadata") if item)
            canonical_batches.append(replace(batch, entries=tuple(canonical_edges), error=error))
        return tuple(canonical_batches)

    def _candidate_contexts(
        self,
        batches: Sequence[CitationBatch],
        requests: Sequence[CitationRequest],
        candidates: Mapping[str, SeedCandidate],
    ) -> dict[str, tuple[int, str | None]]:
        request_by_order = {request.schedule_order: request for request in requests}
        contexts: dict[str, tuple[int, str | None]] = {}
        for batch in batches:
            request = request_by_order[int(batch.source_run_id.rsplit(":", 1)[1])]
            for edge in batch.entries:
                paper_id = edge.target_paper_id if edge.edge_type is CitationEdgeType.REFERENCES else edge.source_paper_id
                inherited_subquestion = candidates[request.seed_paper_id].subquestion_id
                current = contexts.get(paper_id)
                value = (request.depth, inherited_subquestion)
                if current is None or value < current:
                    contexts[paper_id] = value
        return contexts

    def _save_round_papers(
        self,
        round_id: str,
        decisions: Mapping[str, FilterStatus],
        candidate_context: Mapping[str, tuple[int, str | None]],
    ) -> None:
        for paper_id, status in decisions.items():
            if self.repository.get_paper(paper_id):
                depth, subquestion_id = candidate_context[paper_id]
                self.database.connection.execute(
                    """INSERT INTO search_round_papers(search_round_id, paper_id, depth, first_seen, screening_status, subquestion_id)
                       VALUES (?, ?, ?, 1, ?, ?) ON CONFLICT(search_round_id, paper_id) DO UPDATE SET screening_status = excluded.screening_status""",
                    (round_id, paper_id, depth, status, subquestion_id),
                )
        self.database.connection.commit()

    def _audit_round(self, round_id: str, audit: RoundAudit, decision: StopDecision, observed_at: str) -> None:
        existing = self.database.connection.execute(
            "SELECT search_round_id FROM search_round_audits WHERE search_round_id = ?", (round_id,)
        ).fetchone()
        if not existing:
            self.rounds.audit(round_id, audit, decision, audited_at=observed_at)


Phase2SearchPipeline = SearchPipeline
