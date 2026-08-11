"""Deterministic coordinator for the read-only Phase 2 search pipeline.

Providers only return envelopes.  This module is the single writer that
records those envelopes, normalizes their entries, and expands citations.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field, replace
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
    ProviderRequest,
    ProviderResponse,
    RequestBudgetExhausted,
    fan_out,
    query_spec_for_native,
    search_streams,
)
from .identity import normalize_author, normalize_doi, normalize_title
from .query_compilers import NativeQuery, compile_queries
from .query_plan import assert_runtime_matches
from .repository import PaperRepository
from .providers.api import CrawlWindow, IdentityCandidate, SeedInput, VenueDescriptor
from .search_runs import IncrementalScope, SearchRunCoordinator, SourceMetrics
from .scope_filter import SCOPE_FILTER_VERSION, evaluate_scope, screening_scope_hash
from .stage2_pipeline import ERROR_RATE_ALARM
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
    stage2_metrics: Mapping[str, Any] = field(default_factory=dict)
    alarm_codes: tuple[str, ...] = ()

    @property
    def stage2(self) -> Mapping[str, Any]:
        return self.stage2_metrics


SEARCH_IMPLEMENTATION_VERSION = "phase2-search-v6"
_OUTCOME_KEY = "pipeline_outcome_v1"
_CAMPAIGN_USAGE_KEY = "campaign_usage_v1"


@dataclass(frozen=True, slots=True)
class VenueFallback:
    provider: str
    role: str
    native_query_hashes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class VenueRun:
    descriptor: VenueDescriptor
    window: CrawlWindow
    context: VenueContext
    cursor: str | None = None
    historical_replay: bool = False
    fallbacks: tuple[VenueFallback, ...] = ()


@dataclass(frozen=True, slots=True)
class _FallbackOperation:
    venue: VenueRun
    fallback: VenueFallback
    order: int
    provider: Mapping[str, Any] | None
    params: Mapping[str, object]


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


@dataclass(slots=True)
class _FallbackState:
    venue_paper_ids: dict[str, set[str]]
    all_paper_ids: set[str]
    non_arxiv_ids: set[str]
    metrics: dict[str, SourceMetrics]
    source_entries: dict[str, list[SourceEntry]]
    source_paper_ids: dict[str, set[str]]
    paper_sources: dict[str, set[tuple[str, str]]]
    paper_subquestions: dict[str, set[str]]
    scope_states: dict[tuple[str, str], IncrementalScope]


@dataclass(slots=True)
class _CampaignBudget:
    max_requests: int
    max_candidates: int
    max_seconds: float
    requests_made: int
    candidates_returned: int
    elapsed_seconds: float
    attempt_started: float
    crawl_attempt_id: str

    @property
    def request_budget(self) -> int:
        return max(0, self.max_requests - self.requests_made)

    @property
    def candidate_budget(self) -> int:
        return max(0, self.max_candidates - self.candidates_returned)

    @property
    def deadline(self) -> float:
        return self.attempt_started + max(0.0, self.max_seconds - self.elapsed_seconds)

    def spend(self, requests: int, candidates: int) -> None:
        self.requests_made += requests
        self.candidates_returned += candidates

    def snapshot(self) -> dict[str, int | float]:
        elapsed = self.elapsed_seconds + max(0.0, time.monotonic() - self.attempt_started)
        return {
            "requests_made": self.requests_made,
            "candidates_returned": self.candidates_returned,
            "elapsed_seconds": elapsed,
        }


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
        run_status = self._ensure_run(run_id, observed_at)
        completed = (
            self._completed_result(run_id, crawl_run_id)
            if run_status == "complete"
            else None
        )
        self.runs.start_crawl(
            crawl_run_id=crawl_run_id,
            run_id=run_id,
            search_plan_id=str(self.plan["plan_id"]),
            window=self._crawl_window(),
        )
        if completed is not None:
            return completed
        self._active_venue_runs = self._watermarked_venue_runs()

        campaign = self._campaign_budget(crawl_run_id)
        discovery_plan = self._discovery_plan()
        discovery_plan["budgets"] = {
            **discovery_plan["budgets"],
            "max_requests": campaign.request_budget,
            "max_candidates": campaign.candidate_budget,
        }
        fanout = fan_out(
            discovery_plan,
            self._execution_clients(),
            deadline=campaign.deadline,
            request_started=lambda request: self._reserve_fanout_request(
                crawl_run_id, observed_at, request
            ),
            request_finished=lambda request_attempt_id, response: (
                self._complete_fanout_request(
                    request_attempt_id, observed_at, response
                )
            ),
        )
        campaign.spend(fanout.requests_made, fanout.candidates_returned)
        self._save_campaign_budget(crawl_run_id, campaign)
        all_paper_ids: set[str] = set()
        non_arxiv_ids: set[str] = set()
        library_seed_ids: set[str] = set()
        metrics: dict[str, SourceMetrics] = {}
        source_entries: dict[str, list[SourceEntry]] = {}
        source_paper_ids: dict[str, set[str]] = {}
        paper_sources: dict[str, set[tuple[str, str]]] = {}
        paper_subquestions: dict[str, set[str]] = {}
        scope_states: dict[tuple[str, str], IncrementalScope] = {}
        venue_paper_ids: dict[str, set[str]] = {
            run.descriptor.venue_id: set() for run in self._active_venue_runs
        }
        self._restore_crawl_snapshot(
            crawl_run_id,
            all_paper_ids=all_paper_ids,
            non_arxiv_ids=non_arxiv_ids,
            paper_sources=paper_sources,
            venue_paper_ids=venue_paper_ids,
            paper_subquestions=paper_subquestions,
        )
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
                if not page.request_made and self._completed_projection_exists(
                    batch.source_run_id,
                    batch.query_hash,
                    page=str(page.page),
                    cursor=page.cursor,
                ):
                    continue
                self.runs.record_batch(
                    crawl_run_id=crawl_run_id,
                    provider=outcome.provider,
                    provider_version=str(provider["version"]),
                    role=page.role,
                    query_text=self._query_text(query),
                    provider_params={
                        **(dict(query.parameters) if query else {}),
                        "campaign_requests_made": int(page.request_made),
                        "campaign_candidates_returned": len(batch.entries),
                    },
                    query_compiler_version=str(provider["query_compiler_version"]),
                    batch=batch,
                    requested_at=observed_at,
                    completed_at=observed_at,
                    page=str(page.page),
                    cursor=page.cursor,
                    alias_group=query.variant_id if query else None,
                    filters=self._filter_audit(query),
                    source_operation_key=(
                        f"{page.role}:{descriptor_key}"
                        if page.role in {"venue_primary", "search"}
                        else None
                    ),
                    request_charged=(
                        0
                        if page.request_attempt_id is not None
                        else int(page.request_made)
                    ),
                    raw_returned_count=(
                        page.raw_returned_count
                        if page.raw_returned_count is not None
                        else len(batch.entries)
                    ),
                    request_attempt_id=page.request_attempt_id,
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
                if page.role == "venue_primary" and page.scope_id:
                    venue_paper_ids.setdefault(page.scope_id, set()).update(ids)
                if outcome.provider != "arxiv":
                    non_arxiv_ids.update(ids)
                if page.role == "library":
                    library_seed_ids.update(ids)

        fallback_state = _FallbackState(
            venue_paper_ids,
            all_paper_ids,
            non_arxiv_ids,
            metrics,
            source_entries,
            source_paper_ids,
            paper_sources,
            paper_subquestions,
            scope_states,
        )
        (
            fallback_requests,
            fallback_candidates,
            fallback_incomplete,
            fallback_budget_exhausted,
        ) = self._run_venue_fallbacks(
            run_id,
            crawl_run_id,
            observed_at,
            campaign.deadline,
            request_budget=campaign.request_budget,
            candidate_budget=campaign.candidate_budget,
            state=fallback_state,
        )
        campaign.spend(fallback_requests, fallback_candidates)
        self._save_campaign_budget(crawl_run_id, campaign)
        fanout = replace(
            fanout,
            incomplete=fanout.incomplete or fallback_incomplete or fallback_budget_exhausted,
            budget_exhausted=fanout.budget_exhausted or fallback_budget_exhausted,
            requests_made=fanout.requests_made + fallback_requests,
            candidates_returned=fanout.candidates_returned + fallback_candidates,
        )

        enrichment_requests, enrichment_candidates, metadata_failed, metadata_budget_exhausted = self._run_metadata(
            run_id,
            crawl_run_id,
            observed_at,
            tuple(sorted(all_paper_ids)),
            request_budget=campaign.request_budget,
            candidate_budget=campaign.candidate_budget,
            deadline=campaign.deadline,
        )
        campaign.spend(enrichment_requests, enrichment_candidates)
        self._save_campaign_budget(crawl_run_id, campaign)
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
            request_budget=campaign.request_budget,
            candidate_budget=campaign.candidate_budget,
            deadline=campaign.deadline,
        )
        campaign.spend(citation_requests_made, citation_candidates_returned)
        self._save_campaign_budget(crawl_run_id, campaign)
        eligible_paper_ids = scope_screener.eligible_ids(
            tuple(scope_screener.scope_statuses)
        )
        stage2 = self._stage2_telemetry()
        alarm_codes = tuple(str(code) for code in stage2.get("alarm_codes", ()))
        status = self.runs.finish_crawl(crawl_run_id, plan=self.plan, fanout=fanout, finished_at=observed_at)
        self._finish_campaign_budget(crawl_run_id, campaign)
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
                "requests_made": campaign.requests_made,
                "candidates_returned": campaign.candidates_returned,
            }
            status = "incomplete"
            self.database.connection.execute(
                "UPDATE crawl_runs SET status = ?, stats_json = ? WHERE crawl_run_id = ?",
                (status, json.dumps(stats, sort_keys=True, separators=(",", ":")), crawl_run_id),
            )
        if ERROR_RATE_ALARM in alarm_codes:
            status = "incomplete"
            self.database.connection.execute(
                "UPDATE crawl_runs SET status = 'incomplete' WHERE crawl_run_id = ?",
                (crawl_run_id,),
            )
        result = PipelineResult(
            crawl_run_id,
            status,
            tuple(sorted(non_arxiv_ids)),
            tuple(sorted(all_paper_ids - non_arxiv_ids)),
            fanout,
            tuple(round_ids),
            eligible_paper_ids,
            stage2,
            alarm_codes,
        )
        row = self.database.connection.execute(
            "SELECT stats_json FROM crawl_runs WHERE crawl_run_id = ?", (crawl_run_id,)
        ).fetchone()
        stats = json.loads(row["stats_json"])
        stats["stage2"] = stage2
        stats["alarm_codes"] = list(alarm_codes)
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
            dict(outcome.get("stage2", {})),
            tuple(str(item) for item in outcome.get("alarm_codes", ())),
        )

    @staticmethod
    def _outcome_document(result: PipelineResult) -> dict[str, object]:
        return {
            "status": result.status,
            "paper_ids": list(result.paper_ids),
            "arxiv_candidate_ids": list(result.arxiv_candidate_ids),
            "eligible_paper_ids": list(result.eligible_paper_ids),
            "citation_round_ids": list(result.citation_round_ids),
            "stage2": dict(result.stage2_metrics),
            "alarm_codes": list(result.alarm_codes),
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

    def _stage2_telemetry(self) -> dict[str, object]:
        telemetry = getattr(self.screener, "telemetry", None)
        return dict(telemetry()) if callable(telemetry) else {}

    def _provider(self, name: str) -> Mapping[str, Any]:
        return next(item for item in self.plan["providers"] if item["provider"] == name)

    def _campaign_budget(self, crawl_run_id: str) -> _CampaignBudget:
        now_epoch = time.time()
        with self.database.transaction() as connection:
            connection.execute(
                """UPDATE provider_request_attempts
                   SET status = 'failed',
                       error_json = COALESCE(error_json, ?),
                       completed_at = COALESCE(
                           completed_at,
                           strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                       )
                   WHERE crawl_run_id = ? AND status = 'running'""",
                (
                    json.dumps(
                        {"message": "request interrupted before response audit"},
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    crawl_run_id,
                ),
            )
            connection.execute(
                """UPDATE crawl_execution_attempts
                   SET completed_at_epoch = ?,
                       elapsed_seconds = MAX(0, ? - started_at_epoch),
                       status = 'failed'
                   WHERE crawl_run_id = ? AND status = 'running'""",
                (now_epoch, now_epoch, crawl_run_id),
            )
            budget_row = connection.execute(
                """SELECT COALESCE(SUM(request_charged), 0),
                          COALESCE(SUM(accepted_count), 0)
                   FROM provider_request_attempts WHERE crawl_run_id = ?""",
                (crawl_run_id,),
            ).fetchone()
            elapsed = float(
                connection.execute(
                    """SELECT COALESCE(SUM(elapsed_seconds), 0)
                       FROM crawl_execution_attempts WHERE crawl_run_id = ?""",
                    (crawl_run_id,),
                ).fetchone()[0]
            )
            attempt_no = int(
                connection.execute(
                    """SELECT COALESCE(MAX(attempt_no), 0) + 1
                       FROM crawl_execution_attempts WHERE crawl_run_id = ?""",
                    (crawl_run_id,),
                ).fetchone()[0]
            )
            crawl_attempt_id = (
                "crawl-attempt-"
                + uuid5(NAMESPACE_URL, f"{crawl_run_id}:{attempt_no}").hex
            )
            connection.execute(
                """INSERT INTO crawl_execution_attempts(
                       crawl_attempt_id, crawl_run_id, attempt_no,
                       started_at_epoch, elapsed_seconds, status
                   ) VALUES (?, ?, ?, ?, 0, 'running')""",
                (crawl_attempt_id, crawl_run_id, attempt_no, now_epoch),
            )
        limits = self.plan["budgets"]
        values = (
            int(budget_row[0]),
            int(budget_row[1]),
            elapsed,
        )
        if any(value < 0 for value in values):
            raise ValueError("persisted campaign usage cannot be negative")
        return _CampaignBudget(
            int(limits["max_requests"]),
            int(limits["max_candidates"]),
            float(limits["max_seconds"]),
            values[0],
            values[1],
            values[2],
            time.monotonic(),
            crawl_attempt_id,
        )

    def _save_campaign_budget(
        self, crawl_run_id: str, campaign: _CampaignBudget
    ) -> None:
        row = self.database.connection.execute(
            "SELECT stats_json FROM crawl_runs WHERE crawl_run_id = ?",
            (crawl_run_id,),
        ).fetchone()
        stats = json.loads(row["stats_json"])
        stats[_CAMPAIGN_USAGE_KEY] = campaign.snapshot()
        now_epoch = time.time()
        self.database.connection.execute(
            """UPDATE crawl_execution_attempts
               SET elapsed_seconds = MAX(0, ? - started_at_epoch)
               WHERE crawl_attempt_id = ? AND status = 'running'""",
            (now_epoch, campaign.crawl_attempt_id),
        )
        self.database.connection.execute(
            "UPDATE crawl_runs SET stats_json = ? WHERE crawl_run_id = ?",
            (json.dumps(stats, sort_keys=True, separators=(",", ":")), crawl_run_id),
        )
        self.database.connection.commit()

    def _finish_campaign_budget(
        self, crawl_run_id: str, campaign: _CampaignBudget
    ) -> None:
        self._save_campaign_budget(crawl_run_id, campaign)
        now_epoch = time.time()
        self.database.connection.execute(
            """UPDATE crawl_execution_attempts
               SET completed_at_epoch = ?,
                   elapsed_seconds = MAX(0, ? - started_at_epoch),
                   status = 'complete'
               WHERE crawl_attempt_id = ?""",
            (now_epoch, now_epoch, campaign.crawl_attempt_id),
        )
        self.database.connection.commit()

    def _crawl_window(self) -> dict[str, object]:
        """Freeze the venue operation graph with the ordinary search scope."""
        window: dict[str, object] = dict(self.plan["scope"])
        expected = {
            str(operation["venue_id"]): operation
            for operation in self.plan.get("venue_operations", ())
        }
        actual = {run.descriptor.venue_id: run for run in self.venue_runs}
        if set(expected) != set(actual):
            raise ValueError("runtime venue operations differ from the approved QueryPlan")
        frozen = []
        for venue_id in sorted(expected):
            operation = expected[venue_id]
            run = actual[venue_id]
            descriptor = operation["descriptor"]
            runtime_binding = {
                "schema_version": str(run.descriptor.schema_version),
                "provider": run.descriptor.provider,
                "adapter": run.descriptor.adapter,
                "parameters": dict(run.descriptor.parameters),
            }
            fallback_binding = [
                {
                    "order": order,
                    "provider": fallback.provider,
                    "role": fallback.role,
                    "native_query_hashes": list(fallback.native_query_hashes),
                }
                for order, fallback in enumerate(run.fallbacks, start=1)
            ]
            if runtime_binding != descriptor or fallback_binding != operation["fallbacks"]:
                raise ValueError("runtime venue operation has drifted from the approved QueryPlan")
            frozen.append({**operation, "window": self._window_mapping(run.window)})
        window["venue_fallback_graph"] = frozen
        return window

    def _restore_crawl_snapshot(
        self,
        crawl_run_id: str,
        *,
        all_paper_ids: set[str],
        non_arxiv_ids: set[str],
        paper_sources: dict[str, set[tuple[str, str]]],
        venue_paper_ids: dict[str, set[str]],
        paper_subquestions: dict[str, set[str]],
    ) -> None:
        """Restore the last finalized attempt before continuing persisted cursors."""
        rows = self.database.connection.execute(
            """SELECT paper_id, provider, descriptor_key
               FROM crawl_paper_snapshot_sources WHERE crawl_run_id = ?""",
            (crawl_run_id,),
        ).fetchall()
        primary_by_venue = {
            run.descriptor.venue_id: run.descriptor.provider
            for run in self._active_venue_runs
        }
        variant_subquestions = {
            str(variant["id"]): str(variant["subquestion_id"])
            for variant in self.plan["query_variants"]
            if variant.get("subquestion_id")
        }
        descriptor_subquestions = {
            f"query:{variant_id}": subquestion_id
            for variant_id, subquestion_id in variant_subquestions.items()
        }
        for operation in self.plan.get("venue_operations", ()):
            for fallback in operation["fallbacks"]:
                if fallback["role"] != "search":
                    continue
                for variant_id, subquestion_id in variant_subquestions.items():
                    descriptor_subquestions[
                        "search:fallback:"
                        f"{operation['venue_id']}:{fallback['order']}:{variant_id}"
                    ] = subquestion_id
        for row in rows:
            paper_id = str(row["paper_id"])
            provider = str(row["provider"])
            descriptor_key = str(row["descriptor_key"])
            all_paper_ids.add(paper_id)
            if provider != "arxiv":
                non_arxiv_ids.add(paper_id)
            paper_sources.setdefault(paper_id, set()).add((provider, descriptor_key))
            subquestion_id = descriptor_subquestions.get(descriptor_key)
            if subquestion_id:
                paper_subquestions.setdefault(paper_id, set()).add(subquestion_id)
            for venue_id, primary in primary_by_venue.items():
                if (
                    (provider == primary and descriptor_key == venue_id)
                    or f":fallback:{venue_id}:" in descriptor_key
                ):
                    venue_paper_ids.setdefault(venue_id, set()).add(paper_id)

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

    def _reserve_fanout_request(
        self,
        crawl_run_id: str,
        observed_at: str,
        request: ProviderRequest,
    ) -> str:
        descriptor = request.scope_id or request.query_hash
        return self.runs.reserve_request_attempt(
            crawl_run_id=crawl_run_id,
            operation_key=(
                f"fanout:{request.provider}:{request.role}:"
                f"{descriptor}:{request.cursor or ''}"
            ),
            provider=request.provider,
            role=request.role,
            query_hash=request.query_hash,
            requested_cursor=request.cursor,
            max_requests=int(self.plan["budgets"]["max_requests"]),
            started_at=observed_at,
        )

    def _complete_fanout_request(
        self,
        request_attempt_id: str,
        observed_at: str,
        response: ProviderResponse,
    ) -> None:
        self.runs.complete_request_attempt(
            request_attempt_id,
            accepted_count=response.accepted_count,
            raw_returned_count=response.raw_returned_count,
            status=response.status,
            error=response.error,
            response_hash=response.response_hash,
            completed_at=observed_at,
        )

    def _discovery_plan(self) -> dict[str, Any]:
        discovery_roles = {"venue_primary"} if self.venue_only else {
            "venue_primary",
            "search",
            "library",
        }
        venue_primaries = {
            run.descriptor.provider for run in self._active_venue_runs
        }
        providers = [
            {
                **provider,
                "resolved": provider["resolved"]
                and bool(discovery_roles.intersection(provider["roles"]))
                and (
                    not self.venue_only
                    or str(provider["provider"]) in venue_primaries
                ),
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

    def _run_venue_fallbacks(
        self,
        run_id: str,
        crawl_run_id: str,
        observed_at: str,
        deadline: float,
        *,
        request_budget: int,
        candidate_budget: int,
        state: _FallbackState,
    ) -> tuple[int, int, bool, bool]:
        """Run a venue's frozen fallback nodes after an incomplete primary."""
        requests = candidates = 0
        incomplete = budget_exhausted = False
        providers = {
            str(item["provider"]): item for item in self.plan["providers"]
        }
        for venue in sorted(
            self._active_venue_runs, key=lambda item: item.descriptor.venue_id
        ):
            venue_id = venue.descriptor.venue_id
            primary = venue.descriptor.provider
            primary_scope = state.scope_states.get((primary, venue_id))
            if (
                primary_scope is not None
                and primary_scope.complete
                and state.venue_paper_ids.get(venue_id)
            ):
                continue
            incomplete = True
            for order, fallback in enumerate(venue.fallbacks, start=1):
                operation = _FallbackOperation(
                    venue,
                    fallback,
                    order,
                    providers.get(fallback.provider),
                    {
                        "venue_id": venue_id,
                        "primary_provider": primary,
                        "fallback_order": order,
                        "fallback_role": fallback.role,
                    },
                )
                remaining_requests = max(0, request_budget - requests)
                remaining_candidates = max(0, candidate_budget - candidates)
                if fallback.role == "search":
                    made, returned, exhausted, complete = self._run_fallback_search(
                        crawl_run_id,
                        observed_at,
                        deadline,
                        operation,
                        remaining_requests,
                        remaining_candidates,
                        state,
                    )
                elif fallback.role in {"metadata_enricher", "metadata_verifier"}:
                    made, returned, exhausted, complete = self._run_fallback_metadata(
                        run_id,
                        crawl_run_id,
                        observed_at,
                        deadline,
                        operation,
                        remaining_requests,
                        remaining_candidates,
                        state,
                    )
                else:
                    made = returned = 0
                    exhausted = False
                    complete = False
                    self._record_fallback_batch(
                        crawl_run_id,
                        observed_at,
                        operation,
                        self._fallback_failure(
                            crawl_run_id,
                            operation,
                            f"unsupported venue fallback role {fallback.role}",
                        ),
                        operation.params,
                        query=None,
                        audited_paper_ids=(),
                        state=state,
                    )
                requests += made
                candidates += returned
                incomplete = incomplete or not complete
                budget_exhausted = budget_exhausted or exhausted
        return requests, candidates, incomplete, budget_exhausted

    def _run_fallback_search(
        self,
        crawl_run_id: str,
        observed_at: str,
        deadline: float,
        operation: _FallbackOperation,
        request_budget: int,
        candidate_budget: int,
        state: _FallbackState,
    ) -> tuple[int, int, bool, bool]:
        error = self._fallback_unavailable(operation)
        venue_scope = {
            **self.plan["scope"],
            "venues": [operation.venue.descriptor.venue_id],
        }
        queries = (
            ()
            if error
            else compile_queries(
                operation.fallback.provider,
                self.plan["query_variants"],
                venue_scope,
                page_size=int(self.plan.get("page_size", 100)),
            )
        )
        if not error and [query.query_hash for query in queries] != list(
            operation.fallback.native_query_hashes
        ):
            error = f"provider {operation.fallback.provider} native query has drifted"
        error = error or ("fallback search has no frozen query" if not queries else None)
        if error:
            self._record_fallback_batch(
                crawl_run_id,
                observed_at,
                operation,
                self._fallback_failure(crawl_run_id, operation, error),
                operation.params,
                query=None,
                audited_paper_ids=(),
                state=state,
            )
            return 0, 0, False, False

        client = self.clients.get(operation.fallback.provider)
        if client is None or not hasattr(client, "search"):
            self._record_fallback_batch(
                crawl_run_id,
                observed_at,
                operation,
                self._fallback_failure(
                    crawl_run_id, operation, "fallback search client is unavailable"
                ),
                operation.params,
                query=None,
                audited_paper_ids=(),
                state=state,
            )
            return 0, 0, False, False

        requests = candidates = 0
        exhausted = False
        operation_complete = True
        for query in queries:
            source_run_id = self._fallback_source_run_id(
                crawl_run_id, operation, query.variant_id
            )
            completed, cursor = self._fallback_resume_state(
                crawl_run_id, operation, query.variant_id
            )
            descriptor_key = self._fallback_descriptor_key(
                operation, query.variant_id
            )
            if completed:
                state.scope_states[(operation.fallback.provider, descriptor_key)] = (
                    IncrementalScope(
                        operation.fallback.provider,
                        descriptor_key,
                        cursor,
                        True,
                        False,
                    )
                )
                continue
            if cursor is not None:
                state.scope_states[(operation.fallback.provider, descriptor_key)] = (
                    IncrementalScope(
                        operation.fallback.provider,
                        descriptor_key,
                        cursor,
                        False,
                        False,
                    )
                )
            seen_cursors = {cursor} if cursor else set()
            while True:
                request_charged = 0
                raw_returned_count = 0
                response_hash = None
                request_attempt_id = None
                reservation_exhausted = False
                if (
                    requests >= request_budget
                    or candidates >= candidate_budget
                    or time.monotonic() >= deadline
                ):
                    exhausted = True
                    operation_complete = False
                    batch = SourceBatch(
                        source_run_id,
                        query.query_hash,
                        (),
                        cursor,
                        EnvelopeStatus.FAILED,
                        "budget_exhausted",
                    )
                else:
                    try:
                        request_attempt_id = self.runs.reserve_request_attempt(
                            crawl_run_id=crawl_run_id,
                            operation_key=(
                                f"fallback:{descriptor_key}:{query.query_hash}:"
                                f"{cursor or ''}"
                            ),
                            provider=operation.fallback.provider,
                            role=operation.fallback.role,
                            query_hash=query.query_hash,
                            requested_cursor=cursor,
                            max_requests=int(self.plan["budgets"]["max_requests"]),
                            started_at=observed_at,
                        )
                    except RequestBudgetExhausted:
                        exhausted = True
                        operation_complete = False
                        reservation_exhausted = True
                        batch = SourceBatch(
                            source_run_id,
                            query.query_hash,
                            (),
                            cursor,
                            EnvelopeStatus.FAILED,
                            "budget_exhausted",
                        )
                    else:
                        requests += 1
                        request_charged = 1
                        try:
                            batch = client.search(
                                query_spec_for_native(operation.provider, query), cursor
                            )
                            if not isinstance(batch, SourceBatch):
                                raise TypeError("fallback search did not return SourceBatch")
                            response_hash = content_hash(batch.to_dict())
                        except Exception as failure:
                            batch = SourceBatch(
                                source_run_id,
                                query.query_hash,
                                (),
                                cursor,
                                EnvelopeStatus.FAILED,
                                str(failure),
                            )
                        else:
                            batch = replace(batch, source_run_id=source_run_id)
                            raw_returned_count = len(batch.entries)
                            if batch.query_hash != query.query_hash:
                                batch = SourceBatch(
                                    source_run_id,
                                    query.query_hash,
                                    (),
                                    cursor,
                                    EnvelopeStatus.FAILED,
                                    f"provider {operation.fallback.provider} returned an unfrozen query hash",
                                )
                            available = max(0, candidate_budget - candidates)
                            cutoff = len(batch.entries) > available or (
                                bool(batch.next_cursor) and len(batch.entries) >= available
                            )
                            if cutoff:
                                batch = replace(
                                    batch,
                                    entries=batch.entries[:available],
                                    status=EnvelopeStatus.PARTIAL,
                                    error="budget_exhausted",
                                )
                                exhausted = True
                            if (
                                batch.status is EnvelopeStatus.SUCCESS
                                and batch.next_cursor
                                and batch.next_cursor in seen_cursors
                            ):
                                batch = replace(
                                    batch,
                                    status=(
                                        EnvelopeStatus.PARTIAL
                                        if batch.entries
                                        else EnvelopeStatus.FAILED
                                    ),
                                    error=f"provider {operation.fallback.provider} repeated cursor {batch.next_cursor}",
                                )
                        wrong_providers = sorted(
                            {
                                entry.provider
                                for entry in batch.entries
                                if entry.provider != operation.fallback.provider
                            }
                        )
                        response_status = (
                            EnvelopeStatus.FAILED if wrong_providers else batch.status
                        )
                        response_error = (
                            "fallback returned entries for a different provider: "
                            + ", ".join(wrong_providers)
                            if wrong_providers
                            else batch.error
                        )
                        self.runs.complete_request_attempt(
                            request_attempt_id,
                            accepted_count=0 if wrong_providers else len(batch.entries),
                            raw_returned_count=raw_returned_count,
                            status=response_status,
                            error=response_error,
                            response_hash=response_hash,
                            completed_at=observed_at,
                        )
                        candidates += 0 if wrong_providers else len(batch.entries)

                self._record_fallback_batch(
                    crawl_run_id,
                    observed_at,
                    operation,
                    batch,
                    {**operation.params, **dict(query.parameters)},
                    query=query,
                    cursor=cursor,
                    audited_paper_ids=(),
                    state=state,
                    request_charged=request_charged,
                    raw_returned_count=raw_returned_count,
                    request_attempt_id=request_attempt_id,
                    request_attempts_recorded=reservation_exhausted,
                )
                if batch.status is not EnvelopeStatus.SUCCESS or not batch.next_cursor:
                    operation_complete = (
                        operation_complete
                        and batch.status is EnvelopeStatus.SUCCESS
                        and not batch.next_cursor
                    )
                    break
                cursor = batch.next_cursor
                seen_cursors.add(cursor)
        return requests, candidates, exhausted, operation_complete

    def _run_fallback_metadata(
        self,
        run_id: str,
        crawl_run_id: str,
        observed_at: str,
        deadline: float,
        operation: _FallbackOperation,
        request_budget: int,
        candidate_budget: int,
        state: _FallbackState,
    ) -> tuple[int, int, bool, bool]:
        fallback = operation.fallback
        name = fallback.provider
        venue_id = operation.venue.descriptor.venue_id
        paper_ids = tuple(sorted(state.venue_paper_ids.get(venue_id, ())))
        params = {
            **operation.params,
            "paper_ids": list(paper_ids),
            "input_hash": self._fallback_metadata_input_hash(operation, paper_ids),
        }
        if self._fallback_metadata_completed(
            crawl_run_id, operation, params
        ):
            descriptor_key = self._fallback_descriptor_key(operation, "papers")
            state.scope_states[(name, descriptor_key)] = IncrementalScope(
                name, descriptor_key, None, True, False
            )
            return 0, 0, False, True
        errors: list[str] = []
        entries: list[SourceEntry] = []
        requests = successful = 0
        exhausted = False
        reservation_exhausted = False
        client = self.clients.get(name)
        method = "enrich" if fallback.role == "metadata_enricher" else "verify"
        invalid = self._fallback_unavailable(operation)
        if invalid is None and (client is None or not hasattr(client, method)):
            invalid = f"fallback {method} client is unavailable"
        if invalid is not None:
            errors.append(invalid)
        elif not paper_ids:
            errors.append("no venue candidates available for fallback")
        else:
            for paper_id in paper_ids:
                if requests >= request_budget or time.monotonic() >= deadline:
                    errors.append("budget_exhausted")
                    exhausted = True
                    break
                if fallback.role == "metadata_enricher" and len(entries) >= candidate_budget:
                    errors.append("budget_exhausted")
                    exhausted = True
                    break
                paper = self.repository.get_paper(paper_id)
                evidence = self.metadata._entries_for_paper(paper_id)
                if paper is None or not evidence:
                    errors.append(f"{paper_id}: metadata evidence is unavailable")
                    continue
                request_query_hash = content_hash(
                    {
                        "provider": name,
                        "role": fallback.role,
                        "paper_id": paper_id,
                        "input_hash": params["input_hash"],
                    }
                )
                try:
                    request_attempt_id = self.runs.reserve_request_attempt(
                        crawl_run_id=crawl_run_id,
                        operation_key=(
                            f"fallback:{self._fallback_descriptor_key(operation, 'papers')}:"
                            f"{paper_id}"
                        ),
                        provider=name,
                        role=fallback.role,
                        query_hash=request_query_hash,
                        requested_cursor=None,
                        max_requests=int(self.plan["budgets"]["max_requests"]),
                        started_at=observed_at,
                    )
                except RequestBudgetExhausted:
                    errors.append("budget_exhausted")
                    exhausted = True
                    reservation_exhausted = True
                    break
                requests += 1
                try:
                    if fallback.role == "metadata_enricher":
                        result = client.enrich(evidence[0])
                        if result.entry.provider != name:
                            raise ValueError(
                                "fallback enrichment returned the wrong provider"
                            )
                        entries.append(result.entry)
                    else:
                        result = client.verify(
                            IdentityCandidate(
                                paper.title,
                                paper.authors,
                                paper.year,
                                paper.doi,
                                paper.arxiv_id,
                            ),
                            evidence,
                        )
                        self._record_verification(run_id, paper_id, name, result)
                    successful += 1
                except Exception as error:
                    self.runs.complete_request_attempt(
                        request_attempt_id,
                        accepted_count=0,
                        raw_returned_count=0,
                        status=EnvelopeStatus.FAILED,
                        error=str(error),
                        response_hash=None,
                        completed_at=observed_at,
                    )
                    errors.append(f"{paper_id}: {error}")
                else:
                    accepted = int(fallback.role == "metadata_enricher")
                    self.runs.complete_request_attempt(
                        request_attempt_id,
                        accepted_count=accepted,
                        raw_returned_count=accepted,
                        status=EnvelopeStatus.SUCCESS,
                        error=None,
                        response_hash=content_hash(asdict(result)),
                        completed_at=observed_at,
                    )
                self.database.connection.commit()

        status = (
            EnvelopeStatus.SUCCESS
            if not errors
            else EnvelopeStatus.PARTIAL
            if successful
            else EnvelopeStatus.FAILED
        )
        self._record_fallback_batch(
            crawl_run_id,
            observed_at,
            operation,
            SourceBatch(
                self._fallback_source_run_id(crawl_run_id, operation, "papers"),
                content_hash(params),
                tuple(entries),
                None,
                status,
                "; ".join(errors) or None,
            ),
            params,
            query=None,
            audited_paper_ids=paper_ids,
            state=state,
            request_charged=requests,
            raw_returned_count=len(entries),
            request_attempt_id=None,
            request_attempts_recorded=bool(requests) or reservation_exhausted,
        )
        return requests, len(entries), exhausted, status is EnvelopeStatus.SUCCESS

    def _record_verification(
        self, run_id: str, paper_id: str, provider: str, result: Any
    ) -> None:
        self.database.connection.execute(
            """INSERT INTO metadata_verification_events(
                verification_event_id, run_id, paper_id, provider, status,
                evidence_json, conflicts_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, paper_id, provider) DO UPDATE SET
                status = excluded.status, evidence_json = excluded.evidence_json,
                conflicts_json = excluded.conflicts_json""",
            (
                f"metadata-verification-{uuid5(NAMESPACE_URL, f'{run_id}:{paper_id}:{provider}').hex}",
                run_id,
                paper_id,
                provider,
                result.status,
                json.dumps(result.evidence, sort_keys=True),
                json.dumps(result.conflicts, sort_keys=True),
            ),
        )

    def _fallback_failure(
        self, crawl_run_id: str, operation: _FallbackOperation, error: str
    ) -> SourceBatch:
        return SourceBatch(
            self._fallback_source_run_id(crawl_run_id, operation, "operation"),
            content_hash(operation.params),
            (),
            None,
            EnvelopeStatus.FAILED,
            error,
        )

    @staticmethod
    def _fallback_unavailable(operation: _FallbackOperation) -> str | None:
        provider, fallback = operation.provider, operation.fallback
        if provider is None:
            return "fallback provider is absent from the frozen QueryPlan"
        if not provider["resolved"]:
            return "fallback provider is unresolved in the frozen QueryPlan"
        if fallback.role not in provider["roles"]:
            return f"fallback provider does not declare role {fallback.role}"
        return None

    def _fallback_source_run_id(
        self, crawl_run_id: str, operation: _FallbackOperation, scope: str
    ) -> str:
        return (
            f"{crawl_run_id}:{operation.venue.descriptor.venue_id}:fallback:"
            f"{operation.order}:{operation.fallback.provider}:"
            f"{operation.fallback.role}:{scope}"
        )

    def _fallback_resume_state(
        self,
        crawl_run_id: str,
        operation: _FallbackOperation,
        scope: str,
    ) -> tuple[bool, str | None]:
        source_run_id = self._fallback_source_run_id(crawl_run_id, operation, scope)
        source = self.database.connection.execute(
            "SELECT 1 FROM source_runs WHERE source_run_id = ?",
            (source_run_id,),
        ).fetchone()
        if source is None:
            return False, None
        descriptor_key = self._fallback_descriptor_key(operation, scope)
        state = self.database.connection.execute(
            """SELECT cursor, complete FROM crawl_scope_statuses
               WHERE crawl_run_id = ? AND provider = ? AND descriptor_key = ?""",
            (crawl_run_id, operation.fallback.provider, descriptor_key),
        ).fetchone()
        if state is None:
            return False, None
        return bool(state["complete"]), str(state["cursor"]) if state["cursor"] else None

    def _fallback_metadata_completed(
        self,
        crawl_run_id: str,
        operation: _FallbackOperation,
        expected_params: Mapping[str, object],
    ) -> bool:
        complete, _ = self._fallback_resume_state(
            crawl_run_id, operation, "papers"
        )
        if not complete:
            return False
        source_run_id = self._fallback_source_run_id(
            crawl_run_id, operation, "papers"
        )
        row = self.database.connection.execute(
            """SELECT provider_params_json FROM search_queries
               WHERE source_run_id = ? AND status = 'complete'
               ORDER BY rowid DESC LIMIT 1""",
            (source_run_id,),
        ).fetchone()
        if row is None:
            return False
        params = json.loads(row["provider_params_json"])
        return all(
            params.get(key) == expected_params[key]
            for key in ("paper_ids", "input_hash")
        )

    def _fallback_metadata_input_hash(
        self, operation: _FallbackOperation, paper_ids: Sequence[str]
    ) -> str:
        evidence = []
        for paper_id in paper_ids:
            rows = self.database.connection.execute(
                """SELECT provider, external_id, raw_metadata_json
                   FROM paper_sources
                   WHERE paper_id = ? AND provider != ?
                   ORDER BY provider, external_id""",
                (paper_id, operation.fallback.provider),
            ).fetchall()
            evidence.append(
                {
                    "paper_id": paper_id,
                    "sources": [dict(row) for row in rows],
                }
            )
        return content_hash(evidence)

    @staticmethod
    def _fallback_descriptor_key(operation: _FallbackOperation, scope: str) -> str:
        return (
            f"{operation.fallback.role}:fallback:"
            f"{operation.venue.descriptor.venue_id}:{operation.order}:{scope}"
        )

    def _record_fallback_batch(
        self,
        crawl_run_id: str,
        observed_at: str,
        operation: _FallbackOperation,
        batch: SourceBatch,
        provider_params: Mapping[str, object],
        *,
        query: NativeQuery | None,
        audited_paper_ids: Sequence[str],
        state: _FallbackState,
        cursor: str | None = None,
        request_charged: int = 0,
        raw_returned_count: int | None = None,
        request_attempt_id: str | None = None,
        request_attempts_recorded: bool = False,
    ) -> None:
        fallback, provider = operation.fallback, operation.provider
        venue_id = operation.venue.descriptor.venue_id
        scope = (
            query.variant_id
            if query is not None
            else "papers"
            if fallback.role in {"metadata_enricher", "metadata_verifier"}
            else "operation"
        )
        descriptor_key = self._fallback_descriptor_key(operation, scope)
        attempt = self._next_fallback_attempt(batch.source_run_id, batch.query_hash)
        page = (
            f"fallback:{venue_id}:{operation.order}:{scope}:attempt:{attempt}"
        )
        wrong_providers = sorted(
            {entry.provider for entry in batch.entries if entry.provider != fallback.provider}
        )
        if wrong_providers:
            batch = SourceBatch(
                batch.source_run_id,
                batch.query_hash,
                (),
                batch.next_cursor if batch.next_cursor is not None else cursor,
                EnvelopeStatus.FAILED,
                "fallback returned entries for a different provider: "
                + ", ".join(wrong_providers),
                raw_response_artifact_hash=batch.raw_response_artifact_hash,
                request_audit=batch.request_audit,
            )
        self.runs.record_batch(
            crawl_run_id=crawl_run_id,
            provider=fallback.provider,
            provider_version=str(provider["version"]) if provider else "unresolved",
            role=fallback.role,
            query_text=self._query_text(query) if query else venue_id,
            provider_params=provider_params,
            query_compiler_version=(
                str(provider["query_compiler_version"])
                if query is not None and provider is not None
                else "venue-fallback-v1"
            ),
            batch=batch,
            requested_at=observed_at,
            completed_at=observed_at,
            page=page,
            cursor=cursor,
            alias_group=query.variant_id if query else None,
            filters=self._filter_audit(query),
            source_operation_key=descriptor_key,
            request_charged=(
                0 if request_attempt_id is not None else request_charged
            ),
            raw_returned_count=raw_returned_count,
            request_attempt_id=request_attempt_id,
            record_request_attempt=not request_attempts_recorded,
        )
        prior = state.metrics.get(batch.source_run_id, SourceMetrics())
        state.source_entries.setdefault(batch.source_run_id, []).extend(batch.entries)
        state.metrics[batch.source_run_id] = replace(
            prior,
            raw_discovered=prior.raw_discovered + len(batch.entries),
            error_count=prior.error_count + int(batch.status is EnvelopeStatus.FAILED),
        )
        state.source_paper_ids.setdefault(batch.source_run_id, set()).update(
            audited_paper_ids
        )
        self._record_scope(
            state.scope_states,
            fallback.provider,
            descriptor_key,
            batch,
            advance_watermark=False,
        )
        if batch.status is EnvelopeStatus.FAILED:
            return
        papers = self.metadata.merge_batch(
            batch, operation.venue.context, candidate_only=True
        )
        ids = {paper.paper_id for paper in papers}
        state.source_paper_ids[batch.source_run_id].update(ids)
        state.venue_paper_ids.setdefault(venue_id, set()).update(ids)
        state.all_paper_ids.update(ids)
        if fallback.provider != "arxiv":
            state.non_arxiv_ids.update(ids)
        for paper_id in ids:
            state.paper_sources.setdefault(paper_id, set()).add(
                (fallback.provider, descriptor_key)
            )
            subquestion_id = self._subquestion_id(query)
            if subquestion_id:
                state.paper_subquestions.setdefault(paper_id, set()).add(subquestion_id)

    def _next_fallback_attempt(self, source_run_id: str, query_hash: str) -> int:
        row = self.database.connection.execute(
            """SELECT COUNT(*) FROM search_queries
               WHERE source_run_id = ? AND query_hash = ?""",
            (source_run_id, query_hash),
        ).fetchone()
        return int(row[0]) + 1

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
        if self.venue_only:
            return 0, 0, False, False
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
                            request_charged=False,
                        )
                        failed = failed or required
                    else:
                        query_hash = content_hash(
                            {"paper_id": paper_id, "provider": name}
                        )
                        try:
                            request_attempt_id = self.runs.reserve_request_attempt(
                                crawl_run_id=crawl_run_id,
                                operation_key=f"metadata:enrich:{name}:{paper_id}",
                                provider=name,
                                role="metadata_enricher",
                                query_hash=query_hash,
                                requested_cursor=None,
                                max_requests=int(self.plan["budgets"]["max_requests"]),
                                started_at=observed_at,
                            )
                        except RequestBudgetExhausted:
                            self._record_metadata_failure(
                                crawl_run_id,
                                observed_at,
                                provider,
                                paper_id,
                                "metadata_enricher",
                                RuntimeError("budget_exhausted"),
                                request_charged=False,
                                record_request_attempt=False,
                            )
                            return requests, candidates, True, True
                        requests += 1
                        try:
                            result = client.enrich(evidence[0])
                        except Exception as error:
                            self.runs.complete_request_attempt(
                                request_attempt_id,
                                accepted_count=0,
                                raw_returned_count=0,
                                status=EnvelopeStatus.FAILED,
                                error=str(error),
                                response_hash=None,
                                completed_at=observed_at,
                            )
                            self._record_metadata_failure(
                                crawl_run_id,
                                observed_at,
                                provider,
                                paper_id,
                                "metadata_enricher",
                                error,
                                request_charged=True,
                                request_attempt_id=request_attempt_id,
                            )
                            failed = failed or required
                        else:
                            batch = SourceBatch(
                                f"{crawl_run_id}:{name}:metadata_enricher",
                                query_hash,
                                (result.entry,),
                                None,
                                EnvelopeStatus.SUCCESS,
                                raw_response_artifact_hash=result.raw_response_artifact_hash,
                            )
                            self.runs.complete_request_attempt(
                                request_attempt_id,
                                accepted_count=1,
                                raw_returned_count=1,
                                status=EnvelopeStatus.SUCCESS,
                                error=None,
                                response_hash=content_hash(asdict(result)),
                                completed_at=observed_at,
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
                                request_charged=0,
                                request_attempt_id=request_attempt_id,
                            )
                            self.metadata.merge_batch(batch)
                            candidates += 1
                            self.database.connection.commit()
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
                            request_charged=False,
                        )
                        failed = failed or required
                    else:
                        query_hash = content_hash(
                            {
                                "paper_id": paper_id,
                                "provider": name,
                                "kind": "verify",
                            }
                        )
                        try:
                            request_attempt_id = self.runs.reserve_request_attempt(
                                crawl_run_id=crawl_run_id,
                                operation_key=f"metadata:verify:{name}:{paper_id}",
                                provider=name,
                                role="metadata_verifier",
                                query_hash=query_hash,
                                requested_cursor=None,
                                max_requests=int(self.plan["budgets"]["max_requests"]),
                                started_at=observed_at,
                            )
                        except RequestBudgetExhausted:
                            self._record_metadata_failure(
                                crawl_run_id,
                                observed_at,
                                provider,
                                paper_id,
                                "metadata_verifier",
                                RuntimeError("budget_exhausted"),
                                request_charged=False,
                                record_request_attempt=False,
                            )
                            return requests, candidates, True, True
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
                            self.runs.complete_request_attempt(
                                request_attempt_id,
                                accepted_count=0,
                                raw_returned_count=0,
                                status=EnvelopeStatus.FAILED,
                                error=str(error),
                                response_hash=None,
                                completed_at=observed_at,
                            )
                            self._record_metadata_failure(
                                crawl_run_id,
                                observed_at,
                                provider,
                                paper_id,
                                "metadata_verifier",
                                error,
                                request_charged=True,
                                request_attempt_id=request_attempt_id,
                            )
                            failed = failed or required
                        else:
                            audit_batch = SourceBatch(
                                f"{crawl_run_id}:{name}:metadata_verifier",
                                query_hash,
                                (),
                                None,
                                EnvelopeStatus.SUCCESS,
                            )
                            self.runs.complete_request_attempt(
                                request_attempt_id,
                                accepted_count=0,
                                raw_returned_count=0,
                                status=EnvelopeStatus.SUCCESS,
                                error=None,
                                response_hash=content_hash(asdict(result)),
                                completed_at=observed_at,
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
                                request_charged=0,
                                request_attempt_id=request_attempt_id,
                            )
                            self._record_verification(run_id, paper_id, name, result)
                            self.database.connection.commit()
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
        *,
        request_charged: bool,
        request_attempt_id: str | None = None,
        record_request_attempt: bool = True,
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
            request_charged=(
                0 if request_attempt_id is not None else int(request_charged)
            ),
            request_attempt_id=request_attempt_id,
            record_request_attempt=record_request_attempt,
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
        next_cursor = (
            batch.next_cursor
            if batch.next_cursor is not None
            else previous.cursor
            if previous
            else None
        )
        states[key] = IncrementalScope(
            provider,
            descriptor_key,
            next_cursor,
            batch.status is EnvelopeStatus.SUCCESS and batch.next_cursor is None,
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
        venue_ids = tuple(
            sorted(
                run.descriptor.venue_id
                for run in self._active_venue_runs
                if run.descriptor.provider == outcome.provider
            )
        )
        role = (
            "search"
            if queries
            else "venue_primary"
            if "venue_primary" in provider["roles"]
            else str(provider["roles"][0])
        )
        values = self._flatten(outcome.result)
        provided_pages = tuple(
            item for item in values if isinstance(item, ProviderPage)
        )
        if provided_pages:
            pages = provided_pages
        elif outcome.status in {"failed", "skipped_budget"}:
            pages = tuple(
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
                    request_made=outcome.request_attempt_id is not None,
                    request_attempt_id=outcome.request_attempt_id,
                )
                for query in queries
            )
            if not pages and not venue_ids:
                pages = (
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
                    request_made=outcome.request_attempt_id is not None,
                    request_attempt_id=outcome.request_attempt_id,
                ),
                )
        else:
            batches = tuple(
                item for item in values if isinstance(item, SourceBatch)
            )
            pages = tuple(
                ProviderPage(
                    role,
                    batch,
                    next(
                        (
                            query
                            for query in queries
                            if query.query_hash == batch.query_hash
                        ),
                        None,
                    ),
                    index,
                    None,
                    request_attempt_id=outcome.request_attempt_id,
                )
                for index, batch in enumerate(batches, start=1)
            )
            if not pages and queries:
                pages = tuple(
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
                        request_attempt_id=outcome.request_attempt_id,
                    )
                    for query in queries
                )
            if not pages and not venue_ids:
                pages = (
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
                        request_attempt_id=outcome.request_attempt_id,
                    ),
                )
        represented = {
            str(page.scope_id)
            for page in pages
            if page.role == "venue_primary" and page.scope_id
        }
        missing_error = outcome.error or (
            "budget_exhausted"
            if outcome.status == "skipped_budget" or pages
            else "provider failed"
        )
        pages = (*pages, *(
            ProviderPage(
                "venue_primary",
                SourceBatch(
                    f"{crawl_run_id}:{outcome.provider}:venue_primary:{venue_id}",
                    "no-query",
                    (),
                    None,
                    EnvelopeStatus.FAILED,
                    missing_error,
                ),
                None,
                1,
                None,
                venue_id,
                False,
            )
            for venue_id in venue_ids
            if venue_id not in represented
        ))
        return tuple(
            sorted(
                pages,
                key=lambda page: (
                    page.role,
                    page.scope_id or "",
                    page.query.query_hash if page.query else "",
                    page.page,
                ),
            )
        )

    def _completed_projection_exists(
        self,
        source_run_id: str,
        query_hash: str,
        *,
        page: str,
        cursor: str | None,
    ) -> bool:
        return self.database.connection.execute(
            """SELECT 1 FROM search_queries
               WHERE source_run_id = ? AND query_hash = ? AND page = ?
                 AND cursor IS ? AND status = 'complete'""",
            (source_run_id, query_hash, page, cursor),
        ).fetchone() is not None

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
            # Freeze the round and its citation_request rows before provider I/O.
            self.database.connection.commit()
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
        crawl_run_id = str(
            self.database.connection.execute(
                "SELECT crawl_run_id FROM search_rounds WHERE search_round_id = ?",
                (round_id,),
            ).fetchone()[0]
        )
        budget_cutoff = False
        remaining = candidate_budget
        requests_made = 0
        for request in requests:
            citation_request_id = str(
                self.database.connection.execute(
                    """SELECT citation_request_id FROM citation_requests
                       WHERE search_round_id = ? AND schedule_order = ?""",
                    (round_id, request.schedule_order),
                ).fetchone()[0]
            )
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
            query_hash = ""
            raw_response_artifact_hash = None
            while len(entries) < min(request.max_candidates, remaining):
                if requests_made >= request_budget or time.monotonic() >= deadline:
                    budget_cutoff = True
                    request_cutoff = True
                    break
                request_query_hash = content_hash(
                    {
                        "citation_request_id": citation_request_id,
                        "provider": request.provider,
                        "direction": request.direction.value,
                        "seed_paper_id": request.seed_paper_id,
                        "depth": request.depth,
                        "seed_rank": request.seed_rank,
                        "schedule_order": request.schedule_order,
                        "max_candidates": request.max_candidates,
                        "cursor": cursor,
                    }
                )
                try:
                    request_attempt_id = self.runs.reserve_request_attempt(
                        crawl_run_id=crawl_run_id,
                        operation_key=(
                            f"citation:{round_id}:{request.schedule_order}:"
                            f"{cursor or ''}"
                        ),
                        provider=request.provider,
                        role="citation",
                        query_hash=request_query_hash,
                        requested_cursor=cursor,
                        max_requests=int(self.plan["budgets"]["max_requests"]),
                        started_at=observed_at,
                        citation_request_id=citation_request_id,
                    )
                except RequestBudgetExhausted:
                    budget_cutoff = True
                    request_cutoff = True
                    query_hash = request_query_hash
                    status = (
                        EnvelopeStatus.PARTIAL if entries else EnvelopeStatus.FAILED
                    )
                    error_message = "budget_exhausted"
                    break
                try:
                    page = operation(paper, cursor)
                except Exception as error:
                    page = CitationBatch(
                        f"{round_id}:{request.schedule_order}",
                        request_query_hash,
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
                accepted_page_count = min(len(page.entries), capacity)
                self.runs.complete_request_attempt(
                    request_attempt_id,
                    accepted_count=accepted_page_count,
                    raw_returned_count=len(page.entries),
                    status=(
                        EnvelopeStatus.PARTIAL if page_truncated else page.status
                    ),
                    error=page.error or ("budget_exhausted" if page_truncated else None),
                    response_hash=content_hash(page.to_dict()),
                    completed_at=observed_at,
                )
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
