"""Deterministic coordinator for the read-only Phase 2 search pipeline.

Providers only return envelopes.  This module is the single writer that
records those envelopes, normalizes their entries, and expands citations.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence
from uuid import NAMESPACE_URL, uuid5

from .canonical import content_hash
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
    PageStream,
    ProviderOutcome,
    ProviderPage,
    fan_out,
    search_streams,
)
from .identity import normalize_author, normalize_doi, normalize_title
from .query_compilers import NativeQuery, compile_queries
from .query_plan import assert_runtime_matches
from .repository import PaperRepository
from .providers.api import CrawlWindow, IdentityCandidate, SeedInput, VenueDescriptor
from .search_runs import IncrementalScope, SearchRunCoordinator, SourceMetrics
from .scope_filter import SCOPE_FILTER_VERSION, evaluate_scope, screening_scope_hash
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
    eligible_paper_ids: tuple[str, ...] = ()


SEARCH_IMPLEMENTATION_VERSION = "phase2-search-v3"
_OUTCOME_KEY = "pipeline_outcome_v1"


@dataclass(frozen=True, slots=True)
class VenueRun:
    descriptor: VenueDescriptor
    window: CrawlWindow
    context: VenueContext
    cursor: str | None = None
    historical_replay: bool = False


@dataclass(slots=True)
class _ScopeBoundScreener:
    pipeline: Any
    run_id: str
    scope_statuses: dict[str, FilterStatus] = field(default_factory=dict, init=False)

    def screen(self, paper_ids: Sequence[str]) -> Mapping[str, FilterStatus]:
        ordered_ids = tuple(sorted(set(paper_ids)))
        scope_decisions = self.pipeline._record_scope_screening(self.run_id, ordered_ids)
        self.scope_statuses.update(scope_decisions)
        eligible = tuple(
            paper_id
            for paper_id in ordered_ids
            if scope_decisions[paper_id] is FilterStatus.RELEVANT
        )
        stage2_decisions = dict(self.pipeline.screener.screen(eligible))
        if set(stage2_decisions) != set(eligible):
            raise ValueError("Stage 2 must return one decision for every in-scope paper")
        return {
            paper_id: (
                stage2_decisions[paper_id]
                if scope_decisions[paper_id] is FilterStatus.RELEVANT
                else scope_decisions[paper_id]
            )
            for paper_id in ordered_ids
        }

    def eligible_ids(self, paper_ids: Sequence[str]) -> tuple[str, ...]:
        return tuple(
            sorted(
                paper_id
                for paper_id in paper_ids
                if self.scope_statuses.get(paper_id) is FilterStatus.RELEVANT
            )
        )

    def reranker_score(self, paper_id: str) -> float:
        if self.scope_statuses.get(paper_id) is not FilterStatus.RELEVANT:
            return 0.0
        return self.pipeline.screener.reranker_score(paper_id)


class InitialFanoutAdapter:
    """Expose protocol search, venue, and library calls as budgeted page streams."""

    def __init__(self, client: Any, venue_runs: Sequence[VenueRun], seed_inputs: Sequence[SeedInput]) -> None:
        self.client = client
        self.venue_runs = tuple(venue_runs)
        self.seed_inputs = tuple(seed_inputs)

    def initial_streams(
        self, provider: Mapping[str, Any], queries: tuple[NativeQuery, ...]
    ) -> tuple[PageStream, ...]:
        streams = list(search_streams(self.client, provider, queries)) if queries else []
        streams.extend(
            PageStream(
                run.descriptor.provider,
                "venue_primary",
                None,
                run.descriptor.venue_id,
                lambda cursor, run=run: self.client.discover(
                    run.descriptor, run.window, cursor if cursor is not None else run.cursor
                ),
            )
            for run in self.venue_runs
        )
        if "library" in provider["roles"] and self.seed_inputs:
            streams.append(
                PageStream(
                    str(provider["provider"]),
                    "library",
                    None,
                    None,
                    lambda cursor: self.client.import_seeds(self.seed_inputs),
                )
            )
        return tuple(streams)


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
        venue_only: bool = False,
    ) -> None:
        self.database = database
        self.plan = dict(plan)
        self.runtime_providers = tuple(dict(item) for item in (runtime_providers or plan["providers"]))
        self.clients = clients
        self.trusts = trusts
        self.venue = venue
        self.venue_runs = tuple(venue_runs)
        self._active_venue_runs = self.venue_runs
        self.seed_inputs = tuple(seed_inputs)
        self.citation_clients = citation_clients or {}
        if screener is None:
            if self.plan["filter"]["profile"] != "fake":
                raise ValueError("SearchPipeline requires an explicit released Stage 2 screener")
            screener = DeterministicFakeScreener(frozenset())
        self.screener = screener
        self.venue_only = venue_only
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
        assert_runtime_matches(
            self.plan,
            self.runtime_providers,
            budgets=self.plan["budgets"],
            include_arxiv_candidates=self.plan["scope"]["include_arxiv_candidates"],
        )
        if self._ensure_run(run_id, observed_at) == "complete":
            return self._completed_result(run_id, crawl_run_id)
        self.runs.start_crawl(
            crawl_run_id=crawl_run_id,
            run_id=run_id,
            search_plan_id=str(self.plan["plan_id"]),
            window=dict(self.plan["scope"]),
        )
        self._active_venue_runs = self._watermarked_venue_runs()

        campaign_deadline = time.monotonic() + float(self.plan["budgets"]["max_seconds"])
        fanout = fan_out(self._discovery_plan(), self._execution_clients(), deadline=campaign_deadline)
        all_paper_ids: set[str] = set()
        non_arxiv_ids: set[str] = set()
        library_seed_ids: set[str] = set()
        metrics: dict[str, SourceMetrics] = {}
        source_entries: dict[str, list[SourceEntry]] = {}
        source_paper_ids: dict[str, set[str]] = {}
        paper_sources: dict[str, set[tuple[str, str]]] = {}
        paper_subquestions: dict[str, set[str]] = {}
        scope_states: dict[tuple[str, str], IncrementalScope] = {}
        for outcome in fanout.outcomes:
            provider = self._provider(outcome.provider)
            queries = self._queries(provider)
            pages = self._source_pages(crawl_run_id, provider, outcome)
            for page in pages:
                query = page.query or next(
                    (item for item in queries if item.query_hash == page.batch.query_hash), None
                )
                scope = page.scope_id or (query.variant_id if query else "default")
                descriptor_key = self._descriptor_key(page, query)
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
                    filters=self._filter_audit(query),
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
                    self._record_scope(
                        scope_states,
                        outcome.provider,
                        descriptor_key,
                        batch,
                        advance_watermark=self._advance_watermark(page, outcome.provider, descriptor_key),
                    )
                    continue
                venue = (
                    self._arxiv_context()
                    if outcome.provider == "arxiv"
                    else self._venue_context(page.scope_id)
                )
                papers = self.metadata.merge_batch(batch, venue)
                ids = {paper.paper_id for paper in papers}
                source_paper_ids.setdefault(batch.source_run_id, set()).update(ids)
                self._record_scope(
                    scope_states,
                    outcome.provider,
                    descriptor_key,
                    batch,
                    advance_watermark=self._advance_watermark(page, outcome.provider, descriptor_key),
                )
                for paper_id in ids:
                    paper_sources.setdefault(paper_id, set()).add((outcome.provider, descriptor_key))
                    subquestion_id = self._subquestion_id(query)
                    if subquestion_id:
                        paper_subquestions.setdefault(paper_id, set()).add(subquestion_id)
                all_paper_ids.update(ids)
                if outcome.provider != "arxiv":
                    non_arxiv_ids.update(ids)
                if page.role == "library":
                    library_seed_ids.update(ids)

        enrichment_requests, enrichment_candidates, metadata_failed, metadata_budget_exhausted = self._run_metadata(
            run_id,
            crawl_run_id,
            observed_at,
            tuple(sorted(all_paper_ids)),
            request_budget=max(
                0, int(self.plan["budgets"]["max_requests"]) - fanout.requests_made
            ),
            candidate_budget=max(
                0, int(self.plan["budgets"]["max_candidates"]) - fanout.candidates_returned
            ),
            deadline=campaign_deadline,
        )
        self._link_versions(observed_at)
        eligible_discoveries = (
            all_paper_ids
            if self.plan["scope"]["include_arxiv_candidates"]
            else non_arxiv_ids
        )
        root_paper_ids = tuple(sorted(
            eligible_discoveries
            | library_seed_ids
            | {paper_id for paper_id in seed_paper_ids if self.repository.get_paper(paper_id)}
        ))
        scope_screener = _ScopeBoundScreener(self, run_id)
        root_decisions = dict(scope_screener.screen(root_paper_ids))
        if set(root_decisions) != set(root_paper_ids):
            raise ValueError("Stage 2 must return one decision for every root discovery")
        eligible_root_paper_ids = scope_screener.eligible_ids(root_paper_ids)

        for source_run_id, source_metrics in metrics.items():
            entries = source_entries[source_run_id]
            identities = {self._identity(entry) for entry in entries}
            raw = source_metrics.raw_discovered
            screened_ids = source_paper_ids.get(source_run_id, set()) & set(root_decisions)
            self.runs.record_metrics(
                source_run_id,
                replace(
                    source_metrics,
                    unique_after_dedup=len(identities),
                    overlap=raw - len(identities),
                    screened=len(screened_ids),
                    excluded=sum(root_decisions[paper_id] is FilterStatus.IRRELEVANT for paper_id in screened_ids),
                    included=sum(root_decisions[paper_id] is FilterStatus.RELEVANT for paper_id in screened_ids),
                ),
                updated_at=observed_at,
            )
        self.runs.finalize_incremental_crawl(
            crawl_run_id,
            paper_sources=paper_sources,
            scopes=tuple(scope_states.values()),
            recorded_at=observed_at,
        )
        round_ids, citation_requests_made, citation_candidates_returned = self._run_citations(
            crawl_run_id,
            observed_at,
            tuple(sorted(set(seed_paper_ids) | library_seed_ids)),
            {
                paper_id: root_decisions[paper_id]
                for paper_id in eligible_root_paper_ids
            },
            paper_subquestions,
            screener=scope_screener,
            request_budget=max(
                0,
                int(self.plan["budgets"]["max_requests"])
                - fanout.requests_made
                - enrichment_requests,
            ),
            candidate_budget=max(
                0,
                int(self.plan["budgets"]["max_candidates"])
                - fanout.candidates_returned
                - enrichment_candidates,
            ),
            deadline=campaign_deadline,
        )
        eligible_paper_ids = scope_screener.eligible_ids(
            tuple(scope_screener.scope_statuses)
        )
        status = self.runs.finish_crawl(crawl_run_id, plan=self.plan, fanout=fanout, finished_at=observed_at)
        if (
            fanout.budget_exhausted
            or metadata_failed
            or any(decision is FilterStatus.NEEDS_REVIEW for decision in root_decisions.values())
        ):
            status = "incomplete"
            self.database.connection.execute(
                "UPDATE crawl_runs SET status = ? WHERE crawl_run_id = ?",
                (status, crawl_run_id),
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
        citation_budget_exhausted = self.database.connection.execute(
            "SELECT 1 FROM search_rounds WHERE crawl_run_id = ? AND stop_reason = 'budget_exhausted' LIMIT 1",
            (crawl_run_id,),
        ).fetchone()
        if fanout.budget_exhausted or metadata_budget_exhausted or citation_budget_exhausted:
            row = self.database.connection.execute(
                "SELECT stats_json FROM crawl_runs WHERE crawl_run_id = ?", (crawl_run_id,)
            ).fetchone()
            stats = json.loads(row["stats_json"])
            stats["budget"] = {
                "reason": "budget_exhausted",
                "requests_made": fanout.requests_made + enrichment_requests + citation_requests_made,
                "candidates_returned": fanout.candidates_returned + enrichment_candidates + citation_candidates_returned,
            }
            status = "incomplete"
            self.database.connection.execute(
                "UPDATE crawl_runs SET status = ?, stats_json = ? WHERE crawl_run_id = ?",
                (status, json.dumps(stats, sort_keys=True, separators=(",", ":")), crawl_run_id),
            )
        result = PipelineResult(
            crawl_run_id,
            status,
            tuple(sorted(non_arxiv_ids)),
            tuple(sorted(all_paper_ids - non_arxiv_ids)),
            fanout,
            tuple(round_ids),
            eligible_paper_ids,
        )
        row = self.database.connection.execute(
            "SELECT stats_json FROM crawl_runs WHERE crawl_run_id = ?", (crawl_run_id,)
        ).fetchone()
        stats = json.loads(row["stats_json"])
        stats[_OUTCOME_KEY] = self._outcome_document(result)
        self.database.connection.execute(
            "UPDATE crawl_runs SET stats_json = ? WHERE crawl_run_id = ?",
            (json.dumps(stats, sort_keys=True, separators=(",", ":")), crawl_run_id),
        )
        self.database.connection.execute(
            "UPDATE pipeline_runs SET status = ?, completed_at = ? WHERE run_id = ?",
            (status, observed_at, run_id),
        )
        self.database.connection.commit()
        return result

    execute = run

    def _ensure_run(self, run_id: str, observed_at: str) -> str:
        plan_id = str(self.plan["plan_id"])
        plan_binding = (
            str(self.plan["plan_hash"]),
            str(self.plan["schema_version"]),
            json.dumps(self.plan, sort_keys=True, separators=(",", ":")),
            json.dumps(self.plan["approval"], sort_keys=True, separators=(",", ":")),
        )
        run_binding = (
            "stage-1",
            str(self.plan["plan_hash"]),
            str(self.plan["filter"]["config_hash"]),
            SEARCH_IMPLEMENTATION_VERSION,
        )
        run = self.database.connection.execute(
            """SELECT stage, status, input_hash, config_hash, implementation_version
               FROM pipeline_runs WHERE run_id = ?""",
            (run_id,),
        ).fetchone()
        if run is not None and (
            run["stage"], run["input_hash"], run["config_hash"], run["implementation_version"]
        ) != run_binding:
            raise ValueError("run_id already exists with different frozen inputs")
        stored_plan = self.database.connection.execute(
            """SELECT content_hash, schema_version, plan_json, approval_json
               FROM search_plans WHERE search_plan_id = ?""",
            (plan_id,),
        ).fetchone()
        if stored_plan is not None and tuple(stored_plan) != plan_binding:
            raise ValueError("search plan ID already exists with different frozen inputs")
        if stored_plan is None:
            self.database.connection.execute(
                """INSERT INTO search_plans(
                       search_plan_id, content_hash, schema_version, plan_json, approval_json, status
                   ) VALUES (?, ?, ?, ?, ?, 'approved')""",
                (plan_id, *plan_binding),
            )
        if run is None:
            self.database.connection.execute(
                """INSERT INTO pipeline_runs(
                       run_id, stage, status, input_hash, config_hash,
                       implementation_version, started_at
                   ) VALUES (?, 'stage-1', 'running', ?, ?, ?, ?)""",
                (run_id, run_binding[1], run_binding[2], run_binding[3], observed_at),
            )
        self.database.connection.commit()
        return str(run["status"]) if run is not None else "running"

    def _completed_result(self, run_id: str, crawl_run_id: str) -> PipelineResult:
        row = self.database.connection.execute(
            """SELECT status, stats_json FROM crawl_runs
               WHERE run_id = ? AND crawl_run_id = ?""",
            (run_id, crawl_run_id),
        ).fetchone()
        if row is None:
            raise ValueError("complete search run has no matching crawl")
        outcome = json.loads(row["stats_json"]).get(_OUTCOME_KEY)
        if not isinstance(outcome, Mapping):
            raise ValueError("complete search run has no persisted outcome")
        fanout_document = outcome.get("fanout")
        if not isinstance(fanout_document, Mapping):
            raise ValueError("persisted search outcome is invalid")
        outcomes = fanout_document.get("outcomes")
        if not isinstance(outcomes, list):
            raise ValueError("persisted search outcome is invalid")
        return PipelineResult(
            crawl_run_id,
            str(outcome["status"]),
            tuple(str(item) for item in outcome["paper_ids"]),
            tuple(str(item) for item in outcome["arxiv_candidate_ids"]),
            FanoutResult(
                tuple(
                    ProviderOutcome(
                        str(item["provider"]), str(item["status"]), None,
                        str(item["error"]) if item["error"] is not None else None,
                    )
                    for item in outcomes
                ),
                bool(fanout_document["incomplete"]),
                bool(fanout_document["budget_exhausted"]),
                int(fanout_document["requests_made"]),
                int(fanout_document["candidates_returned"]),
            ),
            tuple(str(item) for item in outcome["citation_round_ids"]),
            tuple(str(item) for item in outcome["eligible_paper_ids"]),
        )

    @staticmethod
    def _outcome_document(result: PipelineResult) -> dict[str, object]:
        return {
            "status": result.status,
            "paper_ids": list(result.paper_ids),
            "arxiv_candidate_ids": list(result.arxiv_candidate_ids),
            "eligible_paper_ids": list(result.eligible_paper_ids),
            "citation_round_ids": list(result.citation_round_ids),
            "fanout": {
                "outcomes": [
                    {
                        "provider": item.provider,
                        "status": item.status,
                        "error": item.error,
                    }
                    for item in result.fanout.outcomes
                ],
                "incomplete": result.fanout.incomplete,
                "budget_exhausted": result.fanout.budget_exhausted,
                "requests_made": result.fanout.requests_made,
                "candidates_returned": result.fanout.candidates_returned,
            },
        }

    def _provider(self, name: str) -> Mapping[str, Any]:
        return next(item for item in self.plan["providers"] if item["provider"] == name)

    def _record_scope_screening(
        self, run_id: str, paper_ids: Sequence[str]
    ) -> dict[str, FilterStatus]:
        criterion_id = f"{SCOPE_FILTER_VERSION}:{screening_scope_hash(self.plan)}"
        decisions = {}
        with self.database.transaction() as connection:
            for paper_id in paper_ids:
                paper = self.repository.get_paper(paper_id)
                if paper is None:
                    raise ValueError(f"scope filter paper does not exist: {paper_id}")
                rows = connection.execute(
                    """SELECT raw_metadata_json FROM paper_sources
                       WHERE paper_id = ? ORDER BY provider, external_id""",
                    (paper_id,),
                ).fetchall()
                decision = evaluate_scope(
                    paper,
                    tuple(json.loads(row["raw_metadata_json"]) for row in rows),
                    self.plan["scope"],
                )
                decisions[paper_id] = decision.status
                event_id = content_hash(
                    {"run_id": run_id, "paper_id": paper_id, "criterion_id": criterion_id}
                )
                connection.execute(
                    """INSERT INTO screening_events(
                           screening_event_id, run_id, paper_id, criterion_id, decision,
                           reason_code, input_hash, implementation_version
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(run_id, paper_id, criterion_id) DO UPDATE SET
                           decision = excluded.decision,
                           reason_code = excluded.reason_code,
                           input_hash = excluded.input_hash,
                           implementation_version = excluded.implementation_version""",
                    (
                        event_id,
                        run_id,
                        paper_id,
                        criterion_id,
                        {
                            FilterStatus.RELEVANT: "included",
                            FilterStatus.IRRELEVANT: "excluded",
                            FilterStatus.NEEDS_REVIEW: "needs_review",
                        }[decision.status],
                        decision.reason_code,
                        decision.input_hash,
                        SCOPE_FILTER_VERSION,
                    ),
                )
        return decisions

    def _filter_audit(self, query: NativeQuery | None) -> dict[str, object]:
        if query is not None:
            return {
                "requested_filters": dict(query.requested_filters),
                "native_applied_filters": dict(query.native_applied_filters),
                "post_filters": dict(query.post_filters),
            }
        requested = {
            name: self.plan["scope"][name]
            for name in (
                "date_from",
                "date_to",
                "venues",
                "fields",
                "languages",
                "document_types",
            )
        }
        return {
            "requested_filters": requested,
            "native_applied_filters": {},
            "post_filters": requested,
        }

    def _execution_clients(self) -> dict[str, Any]:
        clients: dict[str, Any] = {}
        for provider in self.plan["providers"]:
            name = str(provider["provider"])
            client = self.clients.get(name)
            runs = tuple(run for run in self._active_venue_runs if run.descriptor.provider == name)
            if client is None or callable(client) or (not runs and "library" not in provider["roles"]):
                clients[name] = client
                continue
            clients[name] = InitialFanoutAdapter(client, runs, self.seed_inputs)
        return clients

    def _discovery_plan(self) -> dict[str, Any]:
        discovery_roles = {"venue_primary", "search", "library"}
        providers = [
            {
                **provider,
                "resolved": provider["resolved"] and bool(discovery_roles.intersection(provider["roles"])),
            }
            for provider in self.plan["providers"]
        ]
        execution = dict(self.plan["execution"])
        execution["required_roles"] = [
            role for role in execution["required_roles"] if role in discovery_roles
        ]
        execution["required_providers"] = [
            provider["provider"]
            for provider in providers
            if provider["provider"] in execution["required_providers"] and provider["resolved"]
        ]
        return {**self.plan, "providers": providers, "execution": execution}

    def _run_metadata(
        self,
        run_id: str,
        crawl_run_id: str,
        observed_at: str,
        paper_ids: Sequence[str],
        *,
        request_budget: int,
        candidate_budget: int,
        deadline: float,
    ) -> tuple[int, int, bool, bool]:
        providers = tuple(
            provider
            for provider in sorted(self.plan["providers"], key=lambda item: str(item["provider"]))
            if provider["resolved"]
            and {"metadata_enricher", "metadata_verifier"}.intersection(provider["roles"])
        )
        requests = candidates = 0
        failed = False
        for paper_id in sorted(paper_ids):
            paper = self.repository.get_paper(paper_id)
            if paper is None:
                continue
            for provider in providers:
                name = str(provider["provider"])
                client = self.clients.get(name)
                required = name in self.plan["execution"]["required_providers"]
                if "metadata_enricher" in provider["roles"]:
                    if requests >= request_budget or candidates >= candidate_budget or time.monotonic() >= deadline:
                        return requests, candidates, True, True
                    evidence = self.metadata._entries_for_paper(paper_id)
                    if client is None or not hasattr(client, "enrich") or not evidence:
                        self._record_metadata_failure(
                            crawl_run_id,
                            observed_at,
                            provider,
                            paper_id,
                            "metadata_enricher",
                            RuntimeError("metadata enrichment client or evidence is unavailable"),
                        )
                        failed = failed or required
                    else:
                        requests += 1
                        try:
                            result = client.enrich(evidence[0])
                        except Exception as error:
                            self._record_metadata_failure(
                                crawl_run_id, observed_at, provider, paper_id, "metadata_enricher", error
                            )
                            failed = failed or required
                        else:
                            batch = SourceBatch(
                                f"{crawl_run_id}:{name}:metadata_enricher",
                                content_hash({"paper_id": paper_id, "provider": name}),
                                (result.entry,),
                                None,
                                EnvelopeStatus.SUCCESS,
                                raw_response_artifact_hash=result.raw_response_artifact_hash,
                            )
                            self.runs.record_batch(
                                crawl_run_id=crawl_run_id,
                                provider=name,
                                provider_version=str(provider["version"]),
                                role="metadata_enricher",
                                query_text=paper_id,
                                provider_params={"paper_id": paper_id},
                                query_compiler_version="metadata-enrich-v1",
                                batch=batch,
                                requested_at=observed_at,
                                completed_at=observed_at,
                                page=paper_id,
                            )
                            self.metadata.merge_batch(batch)
                            candidates += 1
                if "metadata_verifier" in provider["roles"]:
                    if requests >= request_budget or time.monotonic() >= deadline:
                        return requests, candidates, True, True
                    evidence = self.metadata._entries_for_paper(paper_id)
                    if client is None or not hasattr(client, "verify"):
                        self._record_metadata_failure(
                            crawl_run_id,
                            observed_at,
                            provider,
                            paper_id,
                            "metadata_verifier",
                            RuntimeError("metadata verifier client is unavailable"),
                        )
                        failed = failed or required
                    else:
                        requests += 1
                        try:
                            result = client.verify(
                                IdentityCandidate(
                                    title=paper.title,
                                    authors=paper.authors,
                                    year=paper.year,
                                    doi=paper.doi,
                                    arxiv_id=paper.arxiv_id,
                                ),
                                evidence,
                            )
                        except Exception as error:
                            self._record_metadata_failure(
                                crawl_run_id, observed_at, provider, paper_id, "metadata_verifier", error
                            )
                            failed = failed or required
                        else:
                            audit_batch = SourceBatch(
                                f"{crawl_run_id}:{name}:metadata_verifier",
                                content_hash({"paper_id": paper_id, "provider": name, "kind": "verify"}),
                                (),
                                None,
                                EnvelopeStatus.SUCCESS,
                            )
                            self.runs.record_batch(
                                crawl_run_id=crawl_run_id,
                                provider=name,
                                provider_version=str(provider["version"]),
                                role="metadata_verifier",
                                query_text=paper_id,
                                provider_params={"paper_id": paper_id},
                                query_compiler_version="metadata-verify-v1",
                                batch=audit_batch,
                                requested_at=observed_at,
                                completed_at=observed_at,
                                page=paper_id,
                            )
                            self.database.connection.execute(
                                """INSERT INTO metadata_verification_events(
                                    verification_event_id, run_id, paper_id, provider, status,
                                    evidence_json, conflicts_json
                                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                                ON CONFLICT(run_id, paper_id, provider) DO UPDATE SET
                                    status = excluded.status, evidence_json = excluded.evidence_json,
                                    conflicts_json = excluded.conflicts_json""",
                                (
                                    f"metadata-verification-{uuid5(NAMESPACE_URL, f'{run_id}:{paper_id}:{name}').hex}",
                                    run_id,
                                    paper_id,
                                    name,
                                    result.status,
                                    json.dumps(result.evidence, sort_keys=True),
                                    json.dumps(result.conflicts, sort_keys=True),
                                ),
                            )
        self.database.connection.commit()
        return requests, candidates, failed, False

    def _record_metadata_failure(
        self,
        crawl_run_id: str,
        observed_at: str,
        provider: Mapping[str, Any],
        paper_id: str,
        role: str,
        error: Exception,
    ) -> None:
        name = str(provider["provider"])
        batch = SourceBatch(
            f"{crawl_run_id}:{name}:{role}",
            content_hash({"paper_id": paper_id, "provider": name, "kind": role}),
            (),
            None,
            EnvelopeStatus.FAILED,
            str(error),
        )
        self.runs.record_batch(
            crawl_run_id=crawl_run_id,
            provider=name,
            provider_version=str(provider["version"]),
            role=role,
            query_text=paper_id,
            provider_params={"paper_id": paper_id},
            query_compiler_version=f"{role}-v1",
            batch=batch,
            requested_at=observed_at,
            completed_at=observed_at,
            page=paper_id,
        )

    def _venue_context(self, venue_id: str | None) -> VenueContext | None:
        if venue_id is None:
            return self.venue
        return next(
            (run.context for run in self.venue_runs if run.descriptor.venue_id == venue_id),
            self.venue,
        )

    def _watermarked_venue_runs(self) -> tuple[VenueRun, ...]:
        active: list[VenueRun] = []
        for run in self.venue_runs:
            requested = self._window_mapping(run.window)
            resolved = self.runs.window_for(
                run.descriptor.provider,
                run.descriptor.venue_id,
                requested,
                replay_window=requested if run.historical_replay else None,
            )
            watermark = resolved.pop("watermark", {})
            active.append(
                replace(
                    run,
                    window=CrawlWindow(**{key: resolved.get(key) for key in requested}),
                    cursor=watermark.get("cursor") if isinstance(watermark, Mapping) else None,
                )
            )
        return tuple(active)

    @staticmethod
    def _window_mapping(window: CrawlWindow) -> dict[str, object]:
        return {key: value for key, value in {
            "date_from": window.date_from,
            "date_to": window.date_to,
            "year": window.year,
            "volume": window.volume,
            "issue": window.issue,
        }.items() if value is not None}

    @staticmethod
    def _descriptor_key(page: ProviderPage, query: NativeQuery | None) -> str:
        if page.role == "venue_primary":
            return page.scope_id or "default"
        if page.role == "search":
            return f"query:{query.variant_id if query else 'default'}"
        return page.role

    def _advance_watermark(self, page: ProviderPage, provider: str, descriptor_key: str) -> bool:
        return not (
            page.role == "venue_primary"
            and any(
                run.descriptor.provider == provider
                and run.descriptor.venue_id == descriptor_key
                and run.historical_replay
                for run in self._active_venue_runs
            )
        )

    @staticmethod
    def _record_scope(
        states: dict[tuple[str, str], IncrementalScope],
        provider: str,
        descriptor_key: str,
        batch: SourceBatch,
        *,
        advance_watermark: bool,
    ) -> None:
        key = (provider, descriptor_key)
        previous = states.get(key)
        states[key] = IncrementalScope(
            provider,
            descriptor_key,
            batch.next_cursor if batch.status is not EnvelopeStatus.FAILED else previous.cursor if previous else None,
            batch.status is EnvelopeStatus.SUCCESS and (previous.complete if previous else True),
            advance_watermark,
        )

    def _queries(self, provider: Mapping[str, Any]) -> tuple[NativeQuery, ...]:
        if self.venue_only or "search" not in provider["roles"]:
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
        if outcome.status in {"failed", "skipped_budget"}:
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

    def _subquestion_id(self, query: NativeQuery | None) -> str | None:
        if query is None:
            return None
        variant = next(
            item for item in self.plan["query_variants"] if item["id"] == query.variant_id
        )
        return variant.get("subquestion_id")

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
        self,
        crawl_run_id: str,
        observed_at: str,
        user_seed_ids: Sequence[str],
        root_decisions: Mapping[str, FilterStatus],
        paper_subquestions: Mapping[str, set[str]],
        *,
        screener: Any,
        request_budget: int,
        candidate_budget: int,
        deadline: float,
    ) -> tuple[list[str], int, int]:
        config = self.plan["citation_snowball"]
        if not config["enabled"] or not self.citation_clients:
            return [], 0, 0
        providers = tuple(sorted(self.citation_clients))
        users = frozenset(
            paper_id for paper_id in user_seed_ids if paper_id in root_decisions
        )
        if not root_decisions:
            return [], 0, 0
        max_requests = request_budget
        max_depth = int(config["max_depth"])
        max_rounds = int(config["max_rounds"])
        max_per_request = int(config["max_per_seed_per_source"])
        max_candidates = candidate_budget
        directions = tuple(CitationEdgeType(value) for value in config["directions"])
        default_subquestion = next(
            (item.get("subquestion_id") for item in self.plan["query_variants"] if item.get("subquestion_id")),
            None,
        )
        candidates = {
            paper_id: SeedCandidate(
                paper_id,
                next(iter(sorted(paper_subquestions.get(paper_id, ()))), default_subquestion),
                status,
                screener.reranker_score(paper_id),
                self.repository.get_paper(paper_id).verification_status,
                0,
                0,
            )
            for paper_id, status in root_decisions.items()
        }
        seen = set(root_decisions)
        expanded: set[str] = set()
        used_requests = 0
        used_candidates = 0
        low_yield_rounds = 0
        round_ids: list[str] = []
        for round_index in range(max_rounds):
            seeds = select_seeds(
                tuple(candidates.values()),
                user_seed_ids=users,
                expanded_paper_ids=frozenset(expanded),
                max_depth=max_depth,
                per_subquestion=20,
                selector_version=str(self.plan["filter"]["seed_selector_version"]),
                selector_config_hash=str(self.plan["filter"]["seed_selector_config_hash"]),
            )
            if not seeds:
                break
            scheduled_count = len(providers) * len(set(directions)) * len(seeds)
            requests = schedule_requests(
                seeds,
                providers=providers,
                directions=directions,
                max_requests=max(0, max_requests - used_requests),
                max_candidates_per_request=max_per_request,
            )
            schedule_cutoff = len(requests) < scheduled_count
            round_id = self.rounds.freeze(crawl_run_id=crawl_run_id, round_index=round_index, seeds=seeds, requests=requests)
            round_ids.append(round_id)
            batches, request_cutoff, requests_made = self._execute_citation_requests(
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
                screener=screener,
            )
            candidate_context = self._candidate_contexts(batches, requests, candidates)
            self._save_round_papers(round_id, decisions, candidate_context)
            for paper_id, decision in decisions.items():
                seen.add(paper_id)
                paper = self.repository.get_paper(paper_id)
                if paper:
                    depth, subquestion_id = candidate_context[paper_id]
                    candidates[paper_id] = SeedCandidate(
                        paper_id, subquestion_id, decision, screener.reranker_score(paper_id), paper.verification_status,
                        depth, round_index,
                    )
            expanded.update(seed.paper_id for seed in seeds)
            next_seeds = select_seeds(
                tuple(candidates.values()),
                user_seed_ids=users,
                expanded_paper_ids=frozenset(expanded),
                max_depth=max_depth,
                per_subquestion=20,
                selector_version=str(self.plan["filter"]["seed_selector_version"]),
                selector_config_hash=str(self.plan["filter"]["seed_selector_config_hash"]),
            )
            budget_exhausted = schedule_cutoff or request_cutoff
            exhausted = (
                bool(requests)
                and len(batches) == len(requests)
                and all(
                    batch.status is EnvelopeStatus.SUCCESS and not batch.error and batch.next_cursor is None
                    for batch in batches
                )
                and not budget_exhausted
                and not next_seeds
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
                    True,
                    low_yield_rounds,
                )
            self._audit_round(round_id, audit, decision, observed_at)
            if decision.stop:
                break
        return round_ids, used_requests, used_candidates

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
        budget_cutoff = False
        remaining = candidate_budget
        requests_made = 0
        for request in requests:
            if remaining <= 0 or requests_made >= request_budget or time.monotonic() >= deadline:
                budget_cutoff = True
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
            request_cutoff = False
            query_hash = str(request.schedule_order)
            raw_response_artifact_hash = None
            while len(entries) < min(request.max_candidates, remaining):
                if requests_made >= request_budget or time.monotonic() >= deadline:
                    budget_cutoff = True
                    request_cutoff = True
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
                page_truncated = len(page.entries) > capacity
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
                if page.status is not EnvelopeStatus.SUCCESS or page.error:
                    status = (
                        EnvelopeStatus.FAILED
                        if page.status is EnvelopeStatus.FAILED and not entries
                        else EnvelopeStatus.PARTIAL
                    )
                    error_message = page.error or "citation page failed"
                    break
                cursor = page.next_cursor
                if page_truncated:
                    budget_cutoff = True
                    request_cutoff = True
                    break
                if not cursor:
                    break
                if len(entries) >= min(request.max_candidates, remaining):
                    budget_cutoff = True
                    request_cutoff = True
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
                    if request_cutoff
                    else "failed"
                    if batch.status is EnvelopeStatus.FAILED
                    else "complete",
                    json.dumps({"message": batch.error or "request budget exhausted"})
                    if batch.error or request_cutoff
                    else None,
                    round_id,
                    request.schedule_order,
                ),
            )
        self.database.connection.commit()
        return tuple(batches), budget_cutoff, requests_made

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
