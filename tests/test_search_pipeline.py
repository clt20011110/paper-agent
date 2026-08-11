from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from threading import Barrier, Lock

import pytest

from paper_agent.citations import DeterministicFakeScreener, citation_edge, reference_edge
from paper_agent.domain import CitationBatch, CitationEdge, CitationEdgeType, EnvelopeStatus, FilterStatus, MembershipStatus, ProviderRole, SourceBatch, SourceEntry, VerificationStatus
from paper_agent.fanout import RequestBudgetExhausted
from paper_agent.query_plan import approve_query_plan, compile_query_plan
from paper_agent.providers.api import CrawlWindow, EnrichmentResult, IdentityCandidate, SeedInput, VenueDescriptor, VerificationResult
from paper_agent.providers.builtin import FixtureTransport, create_builtin
from paper_agent.search_pipeline import SEARCH_IMPLEMENTATION_VERSION, SearchPipeline, VenueFallback, VenueRun
from paper_agent.search_audit import search_audit
from paper_agent.search_runs import RequestReservationTransactionError, SearchRunCoordinator
from paper_agent.stage2_pipeline import (
    ADJUDICATOR_SHARE_ALARM,
    ERROR_RATE_ALARM,
    Stage2Summary,
)
from paper_agent.storage import Database
from paper_agent.verification import ProviderTrust, VenueContext


NOW = "2026-08-09T00:00:00Z"


def _provider(name: str, roles: list[str] | None = None) -> dict[str, object]:
    return {
        "provider": name,
        "distribution": f"paper-agent-{name}",
        "version": "1.0.0",
        "artifact_sha256": "a" * 64,
        "manifest_hash": "b" * 64,
        "roles": roles or ["search"],
        "capabilities": ["stable_id", "metadata", "date_filter"],
        "enabled": True,
        "mode": "api",
        "credentials_present": True,
    }


def _plan(
    providers: list[dict[str, object]], *, required: list[str], citation_cap: int = 10,
    max_depth: int = 1, max_rounds: int = 2, max_requests: int = 4, max_candidates: int = 10,
    max_seconds: int = 10, required_roles: tuple[str, ...] = ("search",),
    citation_directions: tuple[str, ...] = ("references", "citations"),
    include_arxiv_candidates: bool | None = None,
    venue_fallbacks: tuple[tuple[str, str], ...] = (),
    venue_descriptor: dict[str, object] | None = None,
    venue_specifications: tuple[dict[str, object], ...] | None = None,
) -> dict[str, object]:
    descriptor = venue_descriptor or (
        {
            "schema_version": "1",
            "venue_id": "testconf",
            "name": "TestConf",
            "venue_type": "conference",
            "primary_provider": str(required[0]),
            "provider_params": {"volume_id": "v1"},
        }
        if venue_fallbacks
        else None
    )
    scope = {
        "date_from": "2024-01-01",
        "date_to": "2024-12-31",
        "venues": (
            [str(item["descriptor"]["venue_id"]) for item in venue_specifications]
            if venue_specifications is not None
            else [str(descriptor["venue_id"])]
            if descriptor
            else []
        ),
        "fields": ["computer science"],
        "languages": ["en"],
        "document_types": ["article"],
        "user_seeds": [],
    }
    if include_arxiv_candidates is not None:
        scope["include_arxiv_candidates"] = include_arxiv_candidates
    venue_specs = venue_specifications if venue_specifications is not None else (
        (
            {
                "descriptor": descriptor,
                "acceptance": {
                    "schema_version": "2",
                    "venue_id": str(descriptor["venue_id"]),
                    "primary_provider": str(descriptor["primary_provider"]),
                    "fallbacks": [
                        {"provider": provider, "role": role}
                        for provider, role in venue_fallbacks
                    ],
                },
            },
        )
        if descriptor
        else ()
    )
    plan = compile_query_plan(
        {
            "created_at": NOW,
            "research": {"objective": "test", "audience": "test", "primary_question": "test", "subquestions": []},
            "scope": scope,
            "inclusion": {"criteria": [], "exclusion_criteria": []},
            "query_variants": [{"id": "q", "subquestion_id": "q", "alias_group": "q", "raw_query": "paper agents", "synonyms": []}],
            "filter": {"profile": "fake", "config_hash": "c" * 64, "thresholds_hash": "d" * 64, "seed_selector_version": "v1", "seed_selector_config_hash": "e" * 64, "round_state_machine_version": "v1"},
            "citation_snowball": {"enabled": True, "directions": list(citation_directions), "max_depth": max_depth, "max_rounds": max_rounds, "max_per_seed_per_source": citation_cap},
            "budgets": {"max_requests": max_requests, "max_candidates": max_candidates, "max_seconds": max_seconds, "saturation": {"min_unique_included_yield": 0.01, "consecutive_low_yield_rounds": 2}},
            "provider_policy": "all_resolved",
            "required_roles": list(required_roles),
            "required_providers": required,
        },
        providers=providers,
        venue_specs=venue_specs,
    )
    return approve_query_plan(plan, plan["plan_hash"], approved_by="owner", approved_at=NOW)


def _venue_fallbacks(
    plan: dict[str, object], index: int = 0
) -> tuple[VenueFallback, ...]:
    operation = plan["venue_operations"][index]
    return tuple(
        VenueFallback(
            item["provider"], item["role"], tuple(item["native_query_hashes"])
        )
        for item in operation["fallbacks"]
    )


def _trust(name: str, *, primary: bool = False, upstream: str | None = None) -> ProviderTrust:
    return ProviderTrust(
        name,
        frozenset({ProviderRole.VENUE_PRIMARY} if primary else {ProviderRole.METADATA_VERIFIER}),
        name,
        frozenset({upstream or name}),
        0 if primary else 2,
    )


def _entry(provider: str, external_id: str, *, official: bool = False) -> SourceEntry:
    return SourceEntry(
        provider, external_id, "A Paper", ("Ada Lovelace",), doi="10.1000/a-paper", year=2024,
        metadata={
            "official_membership": official,
            "venue_id": "testconf",
            "publication_version": "published",
            "fields": ["computer science"],
            "language": "en",
            "document_type": "article",
        },
    )


@dataclass
class SearchFixture:
    batches: tuple[SourceBatch, ...] = ()
    failure: str | None = None

    def __call__(self, provider, queries):
        if self.failure:
            raise RuntimeError(self.failure)
        return self.batches


def _batch(provider: str, query_hash: str, entries: tuple[SourceEntry, ...]) -> SourceBatch:
    scoped_entries = tuple(
        _scoped(entry)
        for entry in entries
    )
    return SourceBatch(f"fixture:{provider}", query_hash, scoped_entries, None, EnvelopeStatus.SUCCESS, raw_response_artifact_hash=f"{provider}-raw")


def _scoped(entry: SourceEntry) -> SourceEntry:
    return replace(
        entry,
        year=entry.year or 2024,
        metadata={
            "fields": ["computer science"],
            "language": "en",
            "document_type": "article",
            **entry.metadata,
        },
    )


def test_primary_failure_with_shared_fallbacks_is_incomplete_candidate_only(tmp_path) -> None:
    providers = [_provider("openreview", ["venue_primary", "search"]), _provider("crossref"), _provider("semantic_scholar")]
    providers[0]["exact_required"] = True
    plan = _plan(providers, required=[])
    query_hashes = {item["provider"]: item["native_query_hashes"][0] for item in plan["providers"]}
    clients = {
        "openreview": SearchFixture(failure="maintenance"),
        "crossref": SearchFixture((_batch("crossref", query_hashes["crossref"], (_entry("crossref", "c1"),)),)),
        "semantic_scholar": SearchFixture((_batch("semantic_scholar", query_hashes["semantic_scholar"], (_entry("semantic_scholar", "s1"),)),)),
    }
    venue = VenueContext("venue-test", "testconf", "TestConf", "conference", "openreview", {})
    trusts = {"openreview": _trust("openreview", primary=True), "crossref": _trust("crossref", upstream="crossref"), "semantic_scholar": _trust("semantic_scholar", upstream="crossref")}
    with Database(tmp_path / "papers.sqlite3") as database:
        database.migrate()
        result = SearchPipeline(database, plan, runtime_providers=plan["providers"], clients=clients, trusts=trusts, venue=venue).run(run_id="run", crawl_run_id="crawl", observed_at=NOW)
        row = database.connection.execute("SELECT membership_status FROM paper_collections").fetchone()
        verification = database.connection.execute("SELECT verification_status FROM papers").fetchone()[0]
        assert result.status == "incomplete"
        assert row[0] == MembershipStatus.VENUE_CANDIDATE
        assert verification == "single_source"
        assert [outcome.status for outcome in result.fanout.outcomes] == ["success", "failed", "success"]


def test_official_primary_promotes_membership_and_arxiv_stays_candidate(tmp_path) -> None:
    providers = [_provider("openreview", ["venue_primary", "search"]), _provider("arxiv")]
    providers[0]["exact_required"] = True
    plan = _plan(providers, required=[])
    hashes = {item["provider"]: item["native_query_hashes"][0] for item in plan["providers"]}
    clients = {
        "openreview": SearchFixture((_batch("openreview", hashes["openreview"], (_entry("openreview", "o1", official=True),)),)),
        "arxiv": SearchFixture((_batch("arxiv", hashes["arxiv"], (SourceEntry("arxiv", "2401.00001", "Preprint Only", ("Grace Hopper",), arxiv_id="2401.00001", year=2024, metadata={"publication_version": "preprint"}),)),)),
    }
    with Database(tmp_path / "papers.sqlite3") as database:
        database.migrate()
        result = SearchPipeline(
            database, plan, runtime_providers=plan["providers"], clients=clients,
            trusts={"openreview": _trust("openreview", primary=True), "arxiv": _trust("arxiv")},
            venue=VenueContext("venue-test", "testconf", "TestConf", "conference", "openreview", {}),
        ).run(run_id="run", crawl_run_id="crawl", observed_at=NOW)
        statuses = {row["collection_id"]: row["membership_status"] for row in database.connection.execute("SELECT collection_id, membership_status FROM paper_collections")}
        assert statuses == {"arxiv_candidates": MembershipStatus.VENUE_CANDIDATE, "venue-test": MembershipStatus.OFFICIAL_CONFIRMED}
        assert len(result.paper_ids) == 1
        assert len(result.arxiv_candidate_ids) == 1


@pytest.mark.parametrize(
    ("include_arxiv_candidates", "explicit_seed", "expected_screened"),
    ((False, False, 0), (True, False, 1), (False, True, 1)),
)
def test_round_zero_arxiv_screening_uses_only_the_frozen_plan_policy(
    tmp_path,
    include_arxiv_candidates: bool,
    explicit_seed: bool,
    expected_screened: int,
) -> None:
    plan = _plan(
        [_provider("arxiv")],
        required=["arxiv"],
        include_arxiv_candidates=include_arxiv_candidates,
    )
    query_hash = plan["providers"][0]["native_query_hashes"][0]
    entry = SourceEntry(
        "arxiv",
        "2401.00001",
        "Preprint Only",
        ("Grace Hopper",),
        arxiv_id="2401.00001",
        year=2024,
        metadata={"publication_version": "preprint"},
    )

    class RecordingScreener:
        def __init__(self) -> None:
            self.calls: list[tuple[str, ...]] = []

        def screen(self, paper_ids):
            self.calls.append(tuple(paper_ids))
            return {paper_id: FilterStatus.IRRELEVANT for paper_id in paper_ids}

        def reranker_score(self, paper_id):
            return 0.1

    screener = RecordingScreener()
    with Database(tmp_path / "papers.sqlite3") as database:
        database.migrate()
        pipeline = SearchPipeline(
            database,
            plan,
            runtime_providers=plan["providers"],
            clients={"arxiv": SearchFixture((_batch("arxiv", query_hash, (entry,)),))},
            trusts={"arxiv": _trust("arxiv")},
            screener=screener,
        )
        seed_ids: list[str] = []
        if explicit_seed:
            seed_ids.append(pipeline.repository.ingest(entry).paper_id)
        pipeline.run(
            run_id="run",
            crawl_run_id="crawl",
            observed_at=NOW,
            seed_paper_ids=seed_ids,
        )

    assert len(screener.calls) == 1
    assert len(screener.calls[0]) == expected_screened


def test_canonical_scope_gate_filters_before_stage2_and_persists_reasons(tmp_path) -> None:
    plan = _plan([_provider("openalex")], required=["openalex"])
    query_hash = plan["providers"][0]["native_query_hashes"][0]
    entries = (
        SourceEntry(
            "openalex",
            "in-scope",
            "In scope",
            ("Ada",),
            doi="10.1000/in-scope",
            year=2024,
        ),
        SourceEntry(
            "openalex",
            "wrong-language",
            "Wrong language",
            ("Ada",),
            doi="10.1000/wrong-language",
            year=2024,
            metadata={"language": "zh"},
        ),
        SourceEntry(
            "openalex",
            "unknown-language",
            "Unknown language",
            ("Ada",),
            doi="10.1000/unknown-language",
            year=2024,
            metadata={"language": None},
        ),
    )

    class RecordingScreener:
        def __init__(self) -> None:
            self.calls = []

        def screen(self, paper_ids):
            self.calls.append(tuple(paper_ids))
            return {paper_id: FilterStatus.RELEVANT for paper_id in paper_ids}

        def reranker_score(self, paper_id):
            return 1.0

    screener = RecordingScreener()
    with Database(tmp_path / "papers.sqlite3") as database:
        database.migrate()
        result = SearchPipeline(
            database,
            plan,
            runtime_providers=plan["providers"],
            clients={"openalex": SearchFixture((_batch("openalex", query_hash, entries),))},
            trusts={"openalex": _trust("openalex")},
            screener=screener,
        ).run(run_id="run", crawl_run_id="crawl", observed_at=NOW)
        events = database.connection.execute(
            """SELECT decision, reason_code FROM screening_events
               WHERE implementation_version = 'query-scope-v1'
               ORDER BY reason_code"""
        ).fetchall()

    assert len(screener.calls) == 1
    assert len(screener.calls[0]) == 1
    assert result.eligible_paper_ids == screener.calls[0]
    assert result.status == "incomplete"
    assert [tuple(row) for row in events] == [
        ("excluded", "scope_language_mismatch"),
        ("needs_review", "scope_language_unverified"),
        ("included", "scope_match"),
    ]


def test_replay_has_the_same_canonical_ids_and_source_audit(tmp_path) -> None:
    plan = _plan([_provider("openalex")], required=["openalex"])
    query_hash = plan["providers"][0]["native_query_hashes"][0]
    clients = {"openalex": SearchFixture((_batch("openalex", query_hash, (_entry("openalex", "w1"),)),))}

    observed = []
    for name in ("first", "second"):
        with Database(tmp_path / f"{name}.sqlite3") as database:
            database.migrate()
            result = SearchPipeline(
                database, plan, runtime_providers=plan["providers"], clients=clients,
                trusts={"openalex": _trust("openalex")},
            ).run(run_id=f"run-{name}", crawl_run_id=f"crawl-{name}", observed_at=NOW)
            audit = database.connection.execute(
                "SELECT raw_discovered, unique_after_dedup, overlap, error_count FROM source_run_audits"
            ).fetchone()
            filter_audit = json.loads(
                database.connection.execute("SELECT filters_json FROM search_queries").fetchone()[0]
            )
            assert filter_audit == {
                "requested_filters": {
                    "date_from": "2024-01-01",
                    "date_to": "2024-12-31",
                    "venues": [],
                    "fields": ["computer science"],
                    "languages": ["en"],
                    "document_types": ["article"],
                },
                "native_applied_filters": {
                    "date_from": "2024-01-01",
                    "date_to": "2024-12-31",
                },
                "post_filters": {
                    "fields": ["computer science"],
                    "languages": ["en"],
                    "document_types": ["article"],
                },
            }
            observed.append((result.paper_ids, tuple(audit)))
    assert observed[0] == observed[1]


def test_complete_run_recovers_persisted_outcome_without_repeating_fanout(tmp_path) -> None:
    plan = _plan([_provider("openalex")], required=["openalex"])
    query_hash = plan["providers"][0]["native_query_hashes"][0]

    class RecordingClient:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, provider, queries):
            self.calls += 1
            return (
                _batch(
                    "openalex",
                    query_hash,
                    (_entry("openalex", "w1"),),
                ),
            )

    client = RecordingClient()
    with Database(tmp_path / "papers.sqlite3") as database:
        database.migrate()
        pipeline = SearchPipeline(
            database,
            plan,
            runtime_providers=plan["providers"],
            clients={"openalex": client},
            trusts={"openalex": _trust("openalex")},
        )
        first = pipeline.run(run_id="run", crawl_run_id="crawl", observed_at=NOW)
        recovered = pipeline.run(run_id="run", crawl_run_id="crawl", observed_at=NOW)

        assert client.calls == 1
        assert database.connection.execute("SELECT COUNT(*) FROM source_runs").fetchone()[0] == 1

    assert recovered.crawl_run_id == first.crawl_run_id
    assert recovered.status == first.status == "complete"
    assert recovered.paper_ids == first.paper_ids
    assert recovered.arxiv_candidate_ids == first.arxiv_candidate_ids
    assert recovered.eligible_paper_ids == first.eligible_paper_ids
    assert recovered.citation_round_ids == first.citation_round_ids
    assert (
        recovered.fanout.incomplete,
        recovered.fanout.budget_exhausted,
        recovered.fanout.requests_made,
        recovered.fanout.candidates_returned,
    ) == (
        first.fanout.incomplete,
        first.fanout.budget_exhausted,
        first.fanout.requests_made,
        first.fanout.candidates_returned,
    )
    assert [
        (outcome.provider, outcome.status, outcome.error)
        for outcome in recovered.fanout.outcomes
    ] == [
        (outcome.provider, outcome.status, outcome.error)
        for outcome in first.fanout.outcomes
    ]


def test_search_persists_and_replays_stage2_telemetry_without_share_quota(
    tmp_path,
) -> None:
    plan = _plan([_provider("openalex")], required=["openalex"])
    query_hash = plan["providers"][0]["native_query_hashes"][0]

    class TelemetryScreener:
        def screen(self, paper_ids):
            return {paper_id: FilterStatus.RELEVANT for paper_id in paper_ids}

        def reranker_score(self, paper_id):
            return 1.0

        def telemetry(self):
            return {
                "stage2_run_ids": ["stage2-search"],
                "screened_count": 1,
                "reranked_count": 1,
                "adjudicator_count": 1,
                "adjudicator_share": 1.0,
                "adjudicator_capacity": "severe",
                "error_count": 0,
                "error_rate": 0.0,
                "alarm_codes": [ADJUDICATOR_SHARE_ALARM],
                "run_details": [],
            }

    client = SearchFixture((
        _batch("openalex", query_hash, (_entry("openalex", "telemetry"),)),
    ))
    database_path = tmp_path / "telemetry.sqlite3"
    with Database(database_path) as database:
        database.migrate()
        pipeline = SearchPipeline(
            database,
            plan,
            runtime_providers=plan["providers"],
            clients={"openalex": client},
            trusts={"openalex": _trust("openalex")},
            screener=TelemetryScreener(),
        )
        first = pipeline.run(
            run_id="run-telemetry", crawl_run_id="crawl-telemetry", observed_at=NOW
        )
        replay = pipeline.run(
            run_id="run-telemetry", crawl_run_id="crawl-telemetry", observed_at=NOW
        )
        stats = json.loads(database.connection.execute(
            "SELECT stats_json FROM crawl_runs WHERE crawl_run_id = 'crawl-telemetry'"
        ).fetchone()[0])

    audit = search_audit(database_path, "crawl-telemetry")
    assert first.status == replay.status == "complete"
    assert first.alarm_codes == replay.alarm_codes == (ADJUDICATOR_SHARE_ALARM,)
    assert replay.stage2 == first.stage2
    assert stats["stage2"] == first.stage2
    assert stats["alarm_codes"] == [ADJUDICATOR_SHARE_ALARM]
    assert audit["stats"]["stage2"] == first.stage2
    assert audit["stats"]["alarm_codes"] == [ADJUDICATOR_SHARE_ALARM]


def test_search_stage2_terminal_error_alarm_marks_pipeline_incomplete(tmp_path) -> None:
    plan = _plan([_provider("openalex")], required=["openalex"])
    query_hash = plan["providers"][0]["native_query_hashes"][0]

    class TechnicalDecision:
        reason_code = "reranker_backend_failure"

    class ErrorScreener:
        def __init__(self):
            self.calls = 0
            self.summary = Stage2Summary((), 0, 0, 0.0, ())

        def screen(self, paper_ids):
            self.calls += 1
            decisions = tuple(TechnicalDecision() for _ in paper_ids)
            self.summary = Stage2Summary(
                decisions,
                len(decisions) if self.calls == 1 else 0,
                0,
                0.0,
                (),
                len(decisions),
                1.0 if decisions else 0.0,
            )
            return {paper_id: FilterStatus.NEEDS_REVIEW for paper_id in paper_ids}

        def reranker_score(self, paper_id):
            return 1.0

        def telemetry(self):
            return self.summary.telemetry("stage2-error")

    with Database(tmp_path / "error.sqlite3") as database:
        database.migrate()
        pipeline = SearchPipeline(
            database,
            plan,
            runtime_providers=plan["providers"],
            clients={
                "openalex": SearchFixture((
                    _batch("openalex", query_hash, (_entry("openalex", "error"),)),
                ))
            },
            trusts={"openalex": _trust("openalex")},
            screener=ErrorScreener(),
        )
        first = pipeline.run(
            run_id="run-error", crawl_run_id="crawl-error", observed_at=NOW
        )
        first_outcome = json.loads(database.connection.execute(
            "SELECT stats_json FROM crawl_runs WHERE crawl_run_id = 'crawl-error'"
        ).fetchone()[0])["pipeline_outcome_v1"]
        resumed = pipeline.run(
            run_id="run-error", crawl_run_id="crawl-error", observed_at=NOW
        )
        resumed_outcome = json.loads(database.connection.execute(
            "SELECT stats_json FROM crawl_runs WHERE crawl_run_id = 'crawl-error'"
        ).fetchone()[0])["pipeline_outcome_v1"]

    assert first.status == resumed.status == "incomplete"
    assert first.alarm_codes == resumed.alarm_codes == (ERROR_RATE_ALARM,)
    assert first.stage2_metrics == resumed.stage2_metrics
    assert first_outcome == resumed_outcome


def test_complete_primary_skips_fallback_and_recovery_rejects_graph_drift(tmp_path) -> None:
    plan = _plan(
        [
            _provider("pmlr", ["venue_primary"]),
            _provider("crossref", ["metadata_enricher"]),
        ],
        required=["pmlr"],
        required_roles=("venue_primary",),
        max_requests=10,
        venue_fallbacks=(("crossref", "metadata_enricher"),),
    )
    descriptor = VenueDescriptor(1, "testconf", "pmlr", "pmlr", {"volume_id": "v1"})
    context = VenueContext(
        "venue-test", "testconf", "TestConf", "conference", "pmlr", {}
    )

    class PrimaryClient:
        def __init__(self) -> None:
            self.calls = 0

        def discover(self, _descriptor, _window, _cursor):
            self.calls += 1
            return SourceBatch(
                "primary",
                "primary",
                (_entry("pmlr", "official", official=True),),
                None,
                EnvelopeStatus.SUCCESS,
            )

    class MustNotRun:
        def enrich(self, _entry):
            raise AssertionError("fallback must not run after a complete primary")

    client = PrimaryClient()
    run = VenueRun(
        descriptor,
        CrawlWindow(year=2024),
        context,
        fallbacks=_venue_fallbacks(plan),
    )
    with Database(tmp_path / "papers.sqlite3") as database:
        database.migrate()
        pipeline = SearchPipeline(
            database,
            plan,
            clients={"pmlr": client, "crossref": MustNotRun()},
            trusts={"pmlr": _trust("pmlr", primary=True)},
            venue_runs=(run,),
            venue_only=True,
        )
        first = pipeline.run(run_id="run", crawl_run_id="crawl", observed_at=NOW)
        recovered = pipeline.run(run_id="run", crawl_run_id="crawl", observed_at=NOW)
        frozen = json.loads(
            database.connection.execute(
                "SELECT window_json FROM crawl_runs WHERE crawl_run_id = 'crawl'"
            ).fetchone()[0]
        )

        drifted = replace(
            run,
            fallbacks=(VenueFallback("dblp", "metadata_enricher"),),
        )
        with pytest.raises(ValueError, match="runtime venue operation has drifted"):
            SearchPipeline(
                database,
                plan,
                clients={"pmlr": client},
                trusts={"pmlr": _trust("pmlr", primary=True)},
                venue_runs=(drifted,),
                venue_only=True,
            ).run(run_id="run", crawl_run_id="crawl", observed_at=NOW)

        fallback_sources = database.connection.execute(
            "SELECT COUNT(*) FROM source_runs WHERE role LIKE 'metadata_enricher:fallback:%'"
        ).fetchone()[0]

    assert first.status == recovered.status == "complete"
    assert client.calls == 1
    assert fallback_sources == 0
    assert [
        (item["provider"], item["role"])
        for item in frozen["venue_fallback_graph"][0]["fallbacks"]
    ] == [("crossref", "metadata_enricher")]


def test_partial_primary_runs_fallback_graph_in_order_and_keeps_candidate_membership(
    tmp_path,
) -> None:
    plan = _plan(
        [
            _provider("pmlr", ["venue_primary"]),
            _provider("openreview", ["search"]),
            _provider("crossref", ["metadata_enricher"]),
            _provider("pubmed", ["metadata_verifier"]),
        ],
        required=["pmlr"],
        required_roles=("venue_primary", "metadata_verifier"),
        max_requests=10,
        max_candidates=20,
        venue_fallbacks=(
            ("openreview", "search"),
            ("crossref", "metadata_enricher"),
            ("pubmed", "metadata_verifier"),
        ),
    )
    calls: list[str] = []

    class PrimaryClient:
        def discover(self, _descriptor, _window, _cursor):
            return SourceBatch(
                "primary", "primary", (), None, EnvelopeStatus.PARTIAL, "truncated"
            )

    class FallbackSearch:
        def __init__(self) -> None:
            self.calls = 0

        def search(self, query_spec, _cursor):
            self.calls += 1
            if self.calls == 1:
                calls.append("ordinary-search")
                entries = ()
            else:
                calls.append("fallback-search")
                entries = (_entry("openreview", "fallback", official=True),)
            return SourceBatch(
                "openreview",
                query_spec.native_query_hash,
                entries,
                None,
                EnvelopeStatus.SUCCESS,
            )

    class FallbackEnricher:
        def enrich(self, raw):
            calls.append("fallback-enrich")
            return EnrichmentResult(
                _scoped(
                    SourceEntry(
                        "crossref",
                        "fallback-doi",
                        raw.title,
                        raw.authors,
                        doi=raw.doi,
                        year=raw.year,
                        metadata={"official_membership": True, "venue_id": "testconf"},
                    )
                ),
                "crossref",
                "crossref:fallback",
            )

    class FallbackVerifier:
        def verify(self, candidate, _evidence):
            calls.append("fallback-verify")
            return VerificationResult(
                candidate, VerificationStatus.VERIFIED, "pubmed", ("pubmed",)
            )

    venue_run = VenueRun(
        VenueDescriptor(1, "testconf", "pmlr", "pmlr", {"volume_id": "v1"}),
        CrawlWindow(year=2024),
        VenueContext("venue-test", "testconf", "TestConf", "conference", "pmlr", {}),
        fallbacks=_venue_fallbacks(plan),
    )
    with Database(tmp_path / "papers.sqlite3") as database:
        database.migrate()
        result = SearchPipeline(
            database,
            plan,
            clients={
                "pmlr": PrimaryClient(),
                "openreview": FallbackSearch(),
                "crossref": FallbackEnricher(),
                "pubmed": FallbackVerifier(),
            },
            trusts={
                "pmlr": _trust("pmlr", primary=True),
                "openreview": _trust("openreview"),
                "crossref": _trust("crossref"),
                "pubmed": _trust("pubmed"),
            },
            venue_runs=(venue_run,),
        ).run(run_id="run", crawl_run_id="crawl", observed_at=NOW)
        membership = database.connection.execute(
            "SELECT membership_status FROM paper_collections"
        ).fetchone()[0]
        sources = database.connection.execute(
            "SELECT provider, role, status FROM source_runs ORDER BY provider, role"
        ).fetchall()
        fallback_query = database.connection.execute(
            """SELECT provider_params_json FROM search_queries
               WHERE provider = 'openreview' AND page LIKE 'fallback:%'"""
        ).fetchone()

    assert result.status == "incomplete"
    assert calls[:4] == [
        "ordinary-search",
        "fallback-search",
        "fallback-enrich",
        "fallback-verify",
    ]
    assert membership == MembershipStatus.VENUE_CANDIDATE
    assert any(
        provider == "crossref"
        and role.startswith("metadata_enricher:fallback:")
        and status == "complete"
        for provider, role, status in map(tuple, sources)
    )
    assert any(
        provider == "pubmed"
        and role.startswith("metadata_verifier:fallback:")
        and status == "complete"
        for provider, role, status in map(tuple, sources)
    )
    assert ("pmlr", "venue_primary:testconf", "incomplete") in map(tuple, sources)
    assert json.loads(fallback_query[0])["fallback_order"] == 1


def test_failed_fallback_operation_recovers_in_place_with_clean_audit(tmp_path) -> None:
    plan = _plan(
        [
            _provider("pmlr", ["venue_primary"]),
            _provider("crossref", ["metadata_enricher"]),
        ],
        required=["pmlr"],
        required_roles=("venue_primary", "metadata_enricher"),
        max_requests=10,
        max_candidates=20,
        venue_fallbacks=(("crossref", "metadata_enricher"),),
    )

    class PartialPrimary:
        def discover(self, _descriptor, _window, _cursor):
            return SourceBatch(
                "primary",
                "primary",
                (_entry("pmlr", "candidate"),),
                None,
                EnvelopeStatus.PARTIAL,
                "truncated",
            )

    class RecoveringEnricher:
        def __init__(self) -> None:
            self.calls = 0

        def enrich(self, raw):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("temporary fallback outage")
            return EnrichmentResult(
                _scoped(
                    SourceEntry(
                        "crossref", "recovered", raw.title, raw.authors, doi=raw.doi, year=raw.year
                    )
                ),
                "crossref",
                "crossref:recovered",
            )

    enricher = RecoveringEnricher()
    venue_run = VenueRun(
        VenueDescriptor(1, "testconf", "pmlr", "pmlr", {"volume_id": "v1"}),
        CrawlWindow(year=2024),
        VenueContext("venue-test", "testconf", "TestConf", "conference", "pmlr", {}),
        fallbacks=_venue_fallbacks(plan),
    )
    with Database(tmp_path / "papers.sqlite3") as database:
        database.migrate()
        pipeline = SearchPipeline(
            database,
            plan,
            clients={"pmlr": PartialPrimary(), "crossref": enricher},
            trusts={"pmlr": _trust("pmlr", primary=True), "crossref": _trust("crossref")},
            venue_runs=(venue_run,),
            venue_only=True,
        )
        first = pipeline.run(run_id="run", crawl_run_id="crawl", observed_at=NOW)
        failed = database.connection.execute(
            """SELECT source_run_id, status, error_json FROM source_runs
               WHERE provider = 'crossref' AND role LIKE 'metadata_enricher:fallback:%'"""
        ).fetchone()
        second = pipeline.run(run_id="run", crawl_run_id="crawl", observed_at=NOW)
        third = pipeline.run(run_id="run", crawl_run_id="crawl", observed_at=NOW)
        recovered = database.connection.execute(
            """SELECT s.source_run_id, s.status, s.error_json, a.error_count
               FROM source_runs s JOIN source_run_audits a USING(source_run_id)
               WHERE s.provider = 'crossref'
                 AND s.role LIKE 'metadata_enricher:fallback:%'"""
        ).fetchone()
        source_count = database.connection.execute(
            """SELECT COUNT(*) FROM source_runs
               WHERE provider = 'crossref' AND role LIKE 'metadata_enricher:fallback:%'"""
        ).fetchone()[0]
        query_count = database.connection.execute(
            """SELECT COUNT(*) FROM search_queries
               WHERE provider = 'crossref' AND role = 'metadata_enricher'"""
        ).fetchone()[0]
        attempts = database.connection.execute(
            """SELECT status, error_json FROM provider_request_attempts
               WHERE provider = 'crossref' ORDER BY rowid"""
        ).fetchall()

    assert first.status == second.status == third.status == "incomplete"
    assert enricher.calls == 2
    assert failed["status"] == "failed"
    assert "temporary fallback outage" in failed["error_json"]
    assert recovered["source_run_id"] == failed["source_run_id"]
    assert tuple(recovered[1:]) == ("complete", None, 1)
    assert (source_count, query_count) == (1, 2)
    assert [row["status"] for row in attempts] == ["failed", "success"]
    assert "temporary fallback outage" in attempts[0]["error_json"]


def test_fallback_metadata_attempts_do_not_claim_the_aggregate_batch(
    tmp_path,
) -> None:
    plan = _plan(
        [
            _provider("pmlr", ["venue_primary"]),
            _provider("crossref", ["metadata_enricher"]),
        ],
        required=["pmlr"],
        required_roles=("venue_primary",),
        max_requests=10,
        venue_fallbacks=(("crossref", "metadata_enricher"),),
    )

    class Primary:
        def discover(self, _descriptor, _window, _cursor):
            return SourceBatch(
                "primary",
                "primary",
                (
                    replace(_entry("pmlr", "one"), doi="10.1000/one"),
                    replace(_entry("pmlr", "two"), doi="10.1000/two"),
                ),
                None,
                EnvelopeStatus.PARTIAL,
                "primary incomplete",
            )

    class Enricher:
        def enrich(self, raw):
            return EnrichmentResult(
                _scoped(
                    SourceEntry(
                        "crossref",
                        f"enriched-{raw.external_id}",
                        raw.title,
                        raw.authors,
                        doi=raw.doi,
                        year=raw.year,
                    )
                ),
                "crossref",
                f"crossref:{raw.external_id}",
            )

    venue_run = VenueRun(
        VenueDescriptor(1, "testconf", "pmlr", "pmlr", {"volume_id": "v1"}),
        CrawlWindow(year=2024),
        VenueContext(
            "venue-test", "testconf", "TestConf", "conference", "pmlr", {}
        ),
        fallbacks=_venue_fallbacks(plan),
    )
    with Database(tmp_path / "papers.sqlite3") as database:
        database.migrate()
        SearchPipeline(
            database,
            plan,
            clients={"pmlr": Primary(), "crossref": Enricher()},
            trusts={
                "pmlr": _trust("pmlr", primary=True),
                "crossref": _trust("crossref"),
            },
            venue_runs=(venue_run,),
            venue_only=True,
        ).run(run_id="run", crawl_run_id="crawl", observed_at=NOW)
        attempts = database.connection.execute(
            """SELECT source_run_id, citation_request_id, response_artifact_id,
                      response_hash
               FROM provider_request_attempts
               WHERE provider = 'crossref' AND request_charged = 1
               ORDER BY operation_key"""
        ).fetchall()
        aggregate_projection = database.connection.execute(
            """SELECT request_attempt_id FROM search_queries
               WHERE provider = 'crossref' AND role = 'metadata_enricher'"""
        ).fetchone()[0]

    assert len(attempts) == 2
    assert all(tuple(row[:3]) == (None, None, None) for row in attempts)
    assert all(len(row[3]) == 64 for row in attempts)
    assert aggregate_projection is None


def test_fallback_from_primary_provider_is_structurally_candidate_only(tmp_path) -> None:
    plan = _plan(
        [_provider("openreview", ["venue_primary", "search"])],
        required=["openreview"],
        required_roles=("venue_primary",),
        max_requests=4,
        venue_fallbacks=(("openreview", "search"),),
    )

    class Client:
        def discover(self, _descriptor, _window, _cursor):
            return SourceBatch(
                "primary", "primary", (), None, EnvelopeStatus.PARTIAL, "truncated"
            )

        def search(self, query_spec, _cursor):
            return SourceBatch(
                "fallback",
                query_spec.native_query_hash,
                (_entry("openreview", "fallback-official", official=True),),
                None,
                EnvelopeStatus.SUCCESS,
            )

    venue_run = VenueRun(
        VenueDescriptor(
            1, "testconf", "openreview", "openreview", {"volume_id": "v1"}
        ),
        CrawlWindow(year=2024),
        VenueContext(
            "venue-test",
            "testconf",
            "TestConf",
            "conference",
            "openreview",
            {},
        ),
        fallbacks=_venue_fallbacks(plan),
    )
    with Database(tmp_path / "papers.sqlite3") as database:
        database.migrate()
        result = SearchPipeline(
            database,
            plan,
            clients={"openreview": Client()},
            trusts={"openreview": _trust("openreview", primary=True)},
            venue_runs=(venue_run,),
            venue_only=True,
        ).run(run_id="run", crawl_run_id="crawl", observed_at=NOW)
        status = database.connection.execute(
            "SELECT membership_status FROM paper_collections"
        ).fetchone()[0]

    assert result.status == "incomplete"
    assert status == MembershipStatus.VENUE_CANDIDATE


def test_successful_empty_primary_triggers_candidate_only_fallback(tmp_path) -> None:
    plan = _plan(
        [
            _provider("pmlr", ["venue_primary"]),
            _provider("crossref", ["search"]),
        ],
        required=["pmlr"],
        required_roles=("venue_primary",),
        max_requests=4,
        venue_fallbacks=(("crossref", "search"),),
    )

    class EmptyPrimary:
        def discover(self, _descriptor, _window, _cursor):
            return SourceBatch(
                "primary", "primary", (), None, EnvelopeStatus.SUCCESS
            )

    class Fallback:
        calls = 0

        def search(self, query_spec, _cursor):
            self.calls += 1
            return SourceBatch(
                "crossref",
                query_spec.native_query_hash,
                (_entry("crossref", "candidate", official=True),),
                None,
                EnvelopeStatus.SUCCESS,
            )

    fallback = Fallback()
    venue_run = VenueRun(
        VenueDescriptor(1, "testconf", "pmlr", "pmlr", {"volume_id": "v1"}),
        CrawlWindow(year=2024),
        VenueContext(
            "venue-test", "testconf", "TestConf", "conference", "pmlr", {}
        ),
        fallbacks=_venue_fallbacks(plan),
    )
    with Database(tmp_path / "papers.sqlite3") as database:
        database.migrate()
        pipeline = SearchPipeline(
            database,
            plan,
            clients={"pmlr": EmptyPrimary(), "crossref": fallback},
            trusts={
                "pmlr": _trust("pmlr", primary=True),
                "crossref": _trust("crossref"),
            },
            venue_runs=(venue_run,),
            venue_only=True,
        )
        result = pipeline.run(run_id="run", crawl_run_id="crawl", observed_at=NOW)
        membership = database.connection.execute(
            "SELECT membership_status FROM paper_collections"
        ).fetchone()[0]
        restored_subquestions: dict[str, set[str]] = {}
        pipeline._restore_crawl_snapshot(
            "crawl",
            all_paper_ids=set(),
            non_arxiv_ids=set(),
            paper_sources={},
            venue_paper_ids={"testconf": set()},
            paper_subquestions=restored_subquestions,
        )

    assert fallback.calls == 1
    assert result.status == "incomplete"
    assert len(result.paper_ids) == 1
    assert membership == MembershipStatus.VENUE_CANDIDATE
    assert restored_subquestions == {result.paper_ids[0]: {"q"}}


def test_fallback_rejects_entries_that_impersonate_the_primary_provider(tmp_path) -> None:
    plan = _plan(
        [
            _provider("pmlr", ["venue_primary"]),
            _provider("openreview", ["search"]),
        ],
        required=["pmlr"],
        required_roles=("venue_primary",),
        max_requests=4,
        venue_fallbacks=(("openreview", "search"),),
    )

    class Primary:
        def discover(self, _descriptor, _window, _cursor):
            return SourceBatch(
                "primary", "primary", (), None, EnvelopeStatus.PARTIAL, "truncated"
            )

    class ImpersonatingFallback:
        def search(self, query_spec, _cursor):
            return SourceBatch(
                "fallback",
                query_spec.native_query_hash,
                (_entry("pmlr", "forged-official", official=True),),
                None,
                EnvelopeStatus.SUCCESS,
            )

    venue_run = VenueRun(
        VenueDescriptor(1, "testconf", "pmlr", "pmlr", {"volume_id": "v1"}),
        CrawlWindow(year=2024),
        VenueContext("venue-test", "testconf", "TestConf", "conference", "pmlr", {}),
        fallbacks=_venue_fallbacks(plan),
    )
    with Database(tmp_path / "papers.sqlite3") as database:
        database.migrate()
        SearchPipeline(
            database,
            plan,
            clients={"pmlr": Primary(), "openreview": ImpersonatingFallback()},
            trusts={
                "pmlr": _trust("pmlr", primary=True),
                "openreview": _trust("openreview"),
            },
            venue_runs=(venue_run,),
            venue_only=True,
        ).run(run_id="run", crawl_run_id="crawl", observed_at=NOW)
        query = database.connection.execute(
            """SELECT status, error_json FROM search_queries
               WHERE provider = 'openreview' AND page LIKE 'fallback:%'"""
        ).fetchone()
        papers = database.connection.execute("SELECT COUNT(*) FROM papers").fetchone()[0]

    assert query["status"] == "failed"
    assert "different provider" in query["error_json"]
    assert papers == 0


def test_shared_fallback_is_venue_scoped_and_operation_statuses_do_not_overwrite(
    tmp_path,
) -> None:
    def specification(venue_id: str) -> dict[str, object]:
        return {
            "descriptor": {
                "schema_version": "1",
                "venue_id": venue_id,
                "name": venue_id.upper(),
                "venue_type": "conference",
                "primary_provider": "pmlr",
                "provider_params": {"volume_id": venue_id},
            },
            "acceptance": {
                "schema_version": "2",
                "venue_id": venue_id,
                "primary_provider": "pmlr",
                "fallbacks": [{"provider": "crossref", "role": "search"}],
            },
        }

    plan = _plan(
        [
            _provider("pmlr", ["venue_primary"]),
            _provider("crossref", ["search"]),
        ],
        required=["pmlr"],
        required_roles=("venue_primary",),
        max_requests=10,
        venue_specifications=(specification("venue-a"), specification("venue-b")),
    )

    class Primary:
        def discover(self, descriptor, _window, _cursor):
            return SourceBatch(
                descriptor.venue_id,
                descriptor.venue_id,
                (),
                None,
                EnvelopeStatus.PARTIAL,
                "truncated",
            )

    class SharedFallback:
        def __init__(self) -> None:
            self.calls: list[tuple[str, ...]] = []

        def search(self, query_spec, _cursor):
            self.calls.append(query_spec.venue_ids)
            venue_id = query_spec.venue_ids[0]
            if venue_id == "venue-a":
                raise RuntimeError("venue-a unavailable")
            entry = replace(
                _entry("crossref", "venue-b-paper"),
                metadata={
                    **_entry("crossref", "venue-b-paper").metadata,
                    "venue_id": "venue-b",
                },
            )
            return SourceBatch(
                "crossref",
                query_spec.native_query_hash,
                (entry,),
                None,
                EnvelopeStatus.SUCCESS,
            )

    fallback = SharedFallback()
    runs = tuple(
        VenueRun(
            VenueDescriptor(1, venue_id, "pmlr", "pmlr", {"volume_id": venue_id}),
            CrawlWindow(year=2024),
            VenueContext(
                f"collection-{venue_id}",
                venue_id,
                venue_id.upper(),
                "conference",
                "pmlr",
                {},
            ),
            fallbacks=_venue_fallbacks(plan, index),
        )
        for index, venue_id in enumerate(("venue-a", "venue-b"))
    )
    with Database(tmp_path / "papers.sqlite3") as database:
        database.migrate()
        result = SearchPipeline(
            database,
            plan,
            clients={"pmlr": Primary(), "crossref": fallback},
            trusts={
                "pmlr": _trust("pmlr", primary=True),
                "crossref": _trust("crossref"),
            },
            venue_runs=runs,
            venue_only=True,
        ).run(run_id="run", crawl_run_id="crawl", observed_at=NOW)
        sources = database.connection.execute(
            """SELECT role, status FROM source_runs
               WHERE provider = 'crossref' ORDER BY role"""
        ).fetchall()
        memberships = database.connection.execute(
            "SELECT collection_id, membership_status FROM paper_collections"
        ).fetchall()

    assert result.status == "incomplete"
    assert fallback.calls == [("venue-a",), ("venue-b",)]
    assert len(sources) == 2
    assert {row["status"] for row in sources} == {"failed", "complete"}
    assert [tuple(row) for row in memberships] == [
        ("collection-venue-b", MembershipStatus.VENUE_CANDIDATE)
    ]


@pytest.mark.parametrize(
    ("mode", "expected_statuses"),
    (("failed", ["failed", "failed"]), ("skipped", ["complete", "failed"])),
)
def test_shared_primary_failure_or_budget_skip_is_audited_per_venue(
    tmp_path, mode, expected_statuses
) -> None:
    def specification(venue_id: str) -> dict[str, object]:
        return {
            "descriptor": {
                "schema_version": "1",
                "venue_id": venue_id,
                "name": venue_id.upper(),
                "venue_type": "conference",
                "primary_provider": "pmlr",
                "provider_params": {"volume_id": venue_id},
            },
            "acceptance": {
                "schema_version": "2",
                "venue_id": venue_id,
                "primary_provider": "pmlr",
                "fallbacks": [],
            },
        }

    plan = _plan(
        [_provider("pmlr", ["venue_primary"])],
        required=["pmlr"],
        required_roles=("venue_primary",),
        max_requests=10 if mode == "failed" else 1,
        venue_specifications=(specification("venue-a"), specification("venue-b")),
    )

    class Primary:
        def discover(self, descriptor, _window, _cursor):
            return SourceBatch(
                descriptor.venue_id,
                descriptor.venue_id,
                (),
                None,
                EnvelopeStatus.SUCCESS,
            )

    runs = tuple(
        VenueRun(
            VenueDescriptor(
                1, venue_id, "pmlr", "pmlr", {"volume_id": venue_id}
            ),
            CrawlWindow(year=2024),
            VenueContext(
                f"collection-{venue_id}",
                venue_id,
                venue_id.upper(),
                "conference",
                "pmlr",
                {},
            ),
        )
        for venue_id in ("venue-a", "venue-b")
    )
    with Database(tmp_path / f"{mode}.sqlite3") as database:
        database.migrate()
        SearchPipeline(
            database,
            plan,
            clients={"pmlr": None if mode == "failed" else Primary()},
            trusts={"pmlr": _trust("pmlr", primary=True)},
            venue_runs=runs,
            venue_only=True,
        ).run(run_id="run", crawl_run_id="crawl", observed_at=NOW)
        sources = database.connection.execute(
            """SELECT role, status FROM source_runs
               WHERE provider = 'pmlr' ORDER BY role"""
        ).fetchall()

    assert [row["role"] for row in sources] == [
        "venue_primary:venue-a",
        "venue_primary:venue-b",
    ]
    assert [row["status"] for row in sources] == expected_statuses


def test_fallback_search_resumes_from_persisted_query_cursor(tmp_path) -> None:
    plan = _plan(
        [
            _provider("pmlr", ["venue_primary"]),
            _provider("crossref", ["search"]),
        ],
        required=["pmlr"],
        required_roles=("venue_primary",),
        max_requests=10,
        venue_fallbacks=(("crossref", "search"),),
    )

    class Primary:
        def discover(self, _descriptor, _window, _cursor):
            return SourceBatch(
                "primary", "primary", (), None, EnvelopeStatus.PARTIAL, "truncated"
            )

    class RecoveringSearch:
        def __init__(self) -> None:
            self.calls: list[str | None] = []

        def search(self, query_spec, cursor):
            self.calls.append(cursor)
            if cursor is None:
                return SourceBatch(
                    "crossref",
                    query_spec.native_query_hash,
                    (_entry("crossref", "page-1"),),
                    "page-2",
                    EnvelopeStatus.SUCCESS,
                )
            if self.calls.count("page-2") == 1:
                raise RuntimeError("temporary outage")
            return SourceBatch(
                "crossref",
                query_spec.native_query_hash,
                (
                    replace(
                        _entry("crossref", "page-2"),
                        doi="10.1000/page-2",
                        title="Page Two",
                    ),
                ),
                None,
                EnvelopeStatus.SUCCESS,
            )

    fallback = RecoveringSearch()
    venue_run = VenueRun(
        VenueDescriptor(1, "testconf", "pmlr", "pmlr", {"volume_id": "v1"}),
        CrawlWindow(year=2024),
        VenueContext("venue-test", "testconf", "TestConf", "conference", "pmlr", {}),
        fallbacks=_venue_fallbacks(plan),
    )
    with Database(tmp_path / "papers.sqlite3") as database:
        database.migrate()
        pipeline = SearchPipeline(
            database,
            plan,
            clients={"pmlr": Primary(), "crossref": fallback},
            trusts={
                "pmlr": _trust("pmlr", primary=True),
                "crossref": _trust("crossref"),
            },
            venue_runs=(venue_run,),
            venue_only=True,
        )
        pipeline.run(run_id="run", crawl_run_id="crawl", observed_at=NOW)
        second = pipeline.run(run_id="run", crawl_run_id="crawl", observed_at=NOW)
        third = pipeline.run(run_id="run", crawl_run_id="crawl", observed_at=NOW)
        queries = database.connection.execute(
            """SELECT COUNT(*) FROM search_queries
               WHERE provider = 'crossref' AND page LIKE 'fallback:%'"""
        ).fetchone()[0]
        scope = database.connection.execute(
            """SELECT cursor, complete FROM crawl_scope_statuses
               WHERE provider = 'crossref' AND descriptor_key LIKE 'search:fallback:%'"""
        ).fetchone()
        attempts = database.connection.execute(
            """SELECT requested_cursor, status, error_json
               FROM provider_request_attempts
               WHERE provider = 'crossref' ORDER BY rowid"""
        ).fetchall()
        error_count = database.connection.execute(
            """SELECT error_count FROM source_run_audits a
               JOIN source_runs s USING(source_run_id)
               WHERE s.provider = 'crossref'"""
        ).fetchone()[0]

    assert fallback.calls == [None, "page-2", "page-2"]
    assert len(second.paper_ids) == 2
    assert third.paper_ids == second.paper_ids
    assert queries == 3
    assert tuple(scope) == ("page-2", 1)
    assert [(row["requested_cursor"], row["status"]) for row in attempts] == [
        (None, "success"),
        ("page-2", "failed"),
        ("page-2", "success"),
    ]
    assert "temporary outage" in attempts[1]["error_json"]
    assert error_count == 1


def test_fallback_uses_only_the_remaining_campaign_request_budget(tmp_path) -> None:
    plan = _plan(
        [
            _provider("pmlr", ["venue_primary"]),
            _provider("crossref", ["search"]),
        ],
        required=["pmlr"],
        required_roles=("venue_primary",),
        max_requests=1,
        venue_fallbacks=(("crossref", "search"),),
    )

    class Primary:
        def discover(self, _descriptor, _window, _cursor):
            return SourceBatch(
                "primary", "primary", (), None, EnvelopeStatus.PARTIAL, "truncated"
            )

    class MustNotSpendAnotherRequest:
        calls = 0

        def search(self, _query_spec, _cursor):
            self.calls += 1
            raise AssertionError("fallback exceeded the frozen campaign budget")

    fallback = MustNotSpendAnotherRequest()
    venue_run = VenueRun(
        VenueDescriptor(1, "testconf", "pmlr", "pmlr", {"volume_id": "v1"}),
        CrawlWindow(year=2024),
        VenueContext("venue-test", "testconf", "TestConf", "conference", "pmlr", {}),
        fallbacks=_venue_fallbacks(plan),
    )
    with Database(tmp_path / "papers.sqlite3") as database:
        database.migrate()
        result = SearchPipeline(
            database,
            plan,
            clients={"pmlr": Primary(), "crossref": fallback},
            trusts={
                "pmlr": _trust("pmlr", primary=True),
                "crossref": _trust("crossref"),
            },
            venue_runs=(venue_run,),
            venue_only=True,
        ).run(run_id="run", crawl_run_id="crawl", observed_at=NOW)
        query = database.connection.execute(
            """SELECT status, error_json FROM search_queries
               WHERE provider = 'crossref' AND page LIKE 'fallback:%'"""
        ).fetchone()

    assert fallback.calls == 0
    assert result.fanout.requests_made == 1
    assert result.fanout.budget_exhausted
    assert query["status"] == "failed"
    assert "budget_exhausted" in query["error_json"]


def test_candidate_budget_is_cumulative_across_crawl_resume(tmp_path) -> None:
    plan = _plan(
        [
            _provider("pmlr", ["venue_primary"]),
            _provider("crossref", ["search"]),
        ],
        required=["pmlr"],
        required_roles=("venue_primary",),
        max_requests=10,
        max_candidates=1,
        venue_fallbacks=(("crossref", "search"),),
    )

    class Primary:
        calls = 0

        def discover(self, _descriptor, _window, _cursor):
            self.calls += 1
            return SourceBatch(
                "primary", "primary", (), None, EnvelopeStatus.PARTIAL, "truncated"
            )

    class Fallback:
        def __init__(self) -> None:
            self.calls: list[str | None] = []

        def search(self, query_spec, cursor):
            self.calls.append(cursor)
            return SourceBatch(
                "crossref",
                query_spec.native_query_hash,
                (
                    replace(
                        _entry("crossref", f"paper-{cursor or 'one'}"),
                        doi=f"10.1000/{cursor or 'one'}",
                    ),
                ),
                "page-2",
                EnvelopeStatus.SUCCESS,
            )

    primary = Primary()
    fallback = Fallback()
    venue_run = VenueRun(
        VenueDescriptor(1, "testconf", "pmlr", "pmlr", {"volume_id": "v1"}),
        CrawlWindow(year=2024),
        VenueContext(
            "venue-test", "testconf", "TestConf", "conference", "pmlr", {}
        ),
        fallbacks=_venue_fallbacks(plan),
    )
    with Database(tmp_path / "papers.sqlite3") as database:
        database.migrate()
        pipeline = SearchPipeline(
            database,
            plan,
            clients={"pmlr": primary, "crossref": fallback},
            trusts={
                "pmlr": _trust("pmlr", primary=True),
                "crossref": _trust("crossref"),
            },
            venue_runs=(venue_run,),
            venue_only=True,
        )
        first = pipeline.run(run_id="run", crawl_run_id="crawl", observed_at=NOW)
        first_usage = json.loads(
            database.connection.execute(
                "SELECT stats_json FROM crawl_runs WHERE crawl_run_id = 'crawl'"
            ).fetchone()[0]
        )["campaign_usage_v1"]
        second = pipeline.run(run_id="run", crawl_run_id="crawl", observed_at=NOW)
        usage = json.loads(
            database.connection.execute(
                "SELECT stats_json FROM crawl_runs WHERE crawl_run_id = 'crawl'"
            ).fetchone()[0]
        )["campaign_usage_v1"]
        returned = database.connection.execute(
            """SELECT SUM(raw_returned_count) FROM provider_request_attempts
               WHERE crawl_run_id = 'crawl'"""
        ).fetchone()[0]

    assert len(first.paper_ids) == len(second.paper_ids) == 1
    assert primary.calls == 1
    assert fallback.calls == [None]
    assert first_usage["candidates_returned"] == 1
    assert returned == usage["candidates_returned"] == 1


def test_persisted_request_attempt_survives_crash_before_usage_checkpoint(
    tmp_path, monkeypatch
) -> None:
    plan = _plan(
        [_provider("openalex", ["search"])],
        required=["openalex"],
        max_requests=1,
        max_candidates=10,
    )

    class Client:
        calls = 0

        def search(self, query_spec, _cursor):
            self.calls += 1
            return SourceBatch(
                "openalex",
                query_spec.native_query_hash,
                (_entry("openalex", "one"),),
                None,
                EnvelopeStatus.SUCCESS,
            )

    client = Client()
    with Database(tmp_path / "papers.sqlite3") as database:
        database.migrate()
        pipeline = SearchPipeline(
            database,
            plan,
            clients={"openalex": client},
            trusts={"openalex": _trust("openalex")},
        )
        original = pipeline._save_campaign_budget
        checkpoints = 0

        def interrupt_second_checkpoint(crawl_run_id, campaign):
            nonlocal checkpoints
            checkpoints += 1
            if checkpoints == 2:
                raise RuntimeError("simulated process interruption")
            original(crawl_run_id, campaign)

        monkeypatch.setattr(
            pipeline, "_save_campaign_budget", interrupt_second_checkpoint
        )
        with pytest.raises(RuntimeError, match="simulated process interruption"):
            pipeline.run(run_id="run", crawl_run_id="crawl", observed_at=NOW)
        assert database.connection.execute(
            """SELECT SUM(request_charged) FROM provider_request_attempts
               WHERE crawl_run_id = 'crawl'"""
        ).fetchone()[0] == 1

        monkeypatch.setattr(pipeline, "_save_campaign_budget", original)
        pipeline.run(run_id="run", crawl_run_id="crawl", observed_at=NOW)
        usage = json.loads(
            database.connection.execute(
                "SELECT stats_json FROM crawl_runs WHERE crawl_run_id = 'crawl'"
            ).fetchone()[0]
        )["campaign_usage_v1"]

    assert client.calls == 1
    assert usage["requests_made"] == 1


def test_callable_multiple_batches_charge_one_reserved_request(tmp_path) -> None:
    plan = _plan(
        [_provider("openalex", ["search"])],
        required=["openalex"],
        max_requests=1,
        max_candidates=10,
    )
    query_hash = plan["providers"][0]["native_query_hashes"][0]

    class CallableClient:
        calls = 0

        def __call__(self, _provider, _queries):
            self.calls += 1
            return (
                _batch(
                    "openalex",
                    query_hash,
                    (replace(_entry("openalex", "one"), doi="10.1000/one"),),
                ),
                _batch(
                    "openalex",
                    query_hash,
                    (replace(_entry("openalex", "two"), doi="10.1000/two"),),
                ),
            )

    client = CallableClient()
    with Database(tmp_path / "papers.sqlite3") as database:
        database.migrate()
        result = SearchPipeline(
            database,
            plan,
            clients={"openalex": client},
            trusts={"openalex": _trust("openalex")},
        ).run(run_id="run", crawl_run_id="crawl", observed_at=NOW)
        attempt = database.connection.execute(
            """SELECT COUNT(*), SUM(request_charged), SUM(accepted_count),
                      SUM(raw_returned_count)
               FROM provider_request_attempts WHERE crawl_run_id = 'crawl'"""
        ).fetchone()
        attempt_link = database.connection.execute(
            """SELECT request_attempt_id, source_run_id, citation_request_id,
                      response_artifact_id, response_hash
               FROM provider_request_attempts WHERE crawl_run_id = 'crawl'"""
        ).fetchone()
        projections = database.connection.execute(
            """SELECT request_attempt_id FROM search_queries
               WHERE provider = 'openalex' ORDER BY page"""
        ).fetchall()
        usage = json.loads(
            database.connection.execute(
                "SELECT stats_json FROM crawl_runs WHERE crawl_run_id = 'crawl'"
            ).fetchone()[0]
        )["campaign_usage_v1"]

    assert client.calls == 1
    assert result.fanout.requests_made == 1
    assert tuple(attempt) == (1, 1, 2, 2)
    assert tuple(attempt_link[1:4]) == (None, None, None)
    assert len(attempt_link[4]) == 64
    assert [row[0] for row in projections] == [attempt_link[0], attempt_link[0]]
    assert usage["requests_made"] == 1
    assert usage["candidates_returned"] == 2


def test_request_reservation_never_commits_an_outer_transaction(tmp_path) -> None:
    plan = _plan(
        [_provider("openalex", ["search"])],
        required=["openalex"],
        max_requests=1,
    )
    with Database(tmp_path / "papers.sqlite3") as database:
        database.migrate()
        pipeline = SearchPipeline(
            database,
            plan,
            clients={},
            trusts={"openalex": _trust("openalex")},
        )
        pipeline._ensure_run("run", NOW)
        pipeline.runs.start_crawl(
            crawl_run_id="crawl",
            run_id="run",
            search_plan_id=str(plan["plan_id"]),
            window=pipeline._crawl_window(),
        )

        with pytest.raises(
            RequestReservationTransactionError, match="clean transaction boundary"
        ):
            with database.transaction() as connection:
                connection.execute(
                    "INSERT INTO papers(paper_id, title) VALUES ('rollback', 'Rollback')"
                )
                pipeline.runs.reserve_request_attempt(
                    crawl_run_id="crawl",
                    operation_key="fanout:openalex:search:q:",
                    provider="openalex",
                    role="search",
                    query_hash="q",
                    requested_cursor=None,
                    max_requests=1,
                    started_at=NOW,
                )

        paper_count = database.connection.execute(
            "SELECT COUNT(*) FROM papers WHERE paper_id = 'rollback'"
        ).fetchone()[0]
        attempt_count = database.connection.execute(
            "SELECT COUNT(*) FROM provider_request_attempts"
        ).fetchone()[0]

    assert (paper_count, attempt_count) == (0, 0)


def test_concurrent_coordinators_atomically_enforce_frozen_request_cap(
    tmp_path,
) -> None:
    plan = _plan(
        [_provider("openalex", ["search"])],
        required=["openalex"],
        max_requests=1,
    )
    database_path = tmp_path / "papers.sqlite3"
    with Database(database_path) as database:
        database.migrate()
        pipeline = SearchPipeline(
            database,
            plan,
            clients={},
            trusts={"openalex": _trust("openalex")},
        )
        pipeline._ensure_run("run", NOW)
        pipeline.runs.start_crawl(
            crawl_run_id="crawl",
            run_id="run",
            search_plan_id=str(plan["plan_id"]),
            window=pipeline._crawl_window(),
        )

    barrier = Barrier(2)
    network_calls: list[str] = []
    lock = Lock()

    def reserve_then_call_provider(worker: str) -> str:
        with Database(database_path) as worker_database:
            coordinator = SearchRunCoordinator(worker_database)
            barrier.wait(timeout=2)
            try:
                coordinator.reserve_request_attempt(
                    crawl_run_id="crawl",
                    operation_key="fanout:openalex:search:q:",
                    provider="openalex",
                    role="search",
                    query_hash="q",
                    requested_cursor=None,
                    max_requests=1,
                    started_at=NOW,
                )
            except RequestBudgetExhausted:
                return "budget_exhausted"
            with lock:
                network_calls.append(worker)
            return "called"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = sorted(
            executor.map(reserve_then_call_provider, ("one", "two"))
        )

    with Database(database_path, read_only=True) as audit:
        ledger = audit.connection.execute(
            """SELECT COUNT(*), SUM(request_charged)
               FROM provider_request_attempts WHERE crawl_run_id = 'crawl'"""
        ).fetchone()

    assert outcomes == ["budget_exhausted", "called"]
    assert len(network_calls) == 1
    assert tuple(ledger) == (1, 1)


def test_fallback_search_handles_atomic_reservation_exhaustion_before_network(
    tmp_path, monkeypatch
) -> None:
    plan = _plan(
        [
            _provider("pmlr", ["venue_primary"]),
            _provider("crossref", ["search"]),
        ],
        required=["pmlr"],
        required_roles=("venue_primary",),
        max_requests=10,
        venue_fallbacks=(("crossref", "search"),),
    )

    class Primary:
        def discover(self, _descriptor, _window, _cursor):
            return SourceBatch(
                "primary", "primary", (), None, EnvelopeStatus.PARTIAL, "truncated"
            )

    class Fallback:
        calls = 0

        def search(self, _query_spec, _cursor):
            self.calls += 1
            raise AssertionError("fallback network call must not start")

    fallback = Fallback()
    venue_run = VenueRun(
        VenueDescriptor(1, "testconf", "pmlr", "pmlr", {"volume_id": "v1"}),
        CrawlWindow(year=2024),
        VenueContext("venue-test", "testconf", "TestConf", "conference", "pmlr", {}),
        fallbacks=_venue_fallbacks(plan),
    )
    with Database(tmp_path / "papers.sqlite3") as database:
        database.migrate()
        pipeline = SearchPipeline(
            database,
            plan,
            clients={"pmlr": Primary(), "crossref": fallback},
            trusts={
                "pmlr": _trust("pmlr", primary=True),
                "crossref": _trust("crossref"),
            },
            venue_runs=(venue_run,),
            venue_only=True,
        )
        reserve = pipeline.runs.reserve_request_attempt

        def exhaust_fallback(**kwargs):
            if str(kwargs["operation_key"]).startswith("fallback:"):
                raise RequestBudgetExhausted("frozen request budget is exhausted")
            return reserve(**kwargs)

        monkeypatch.setattr(pipeline.runs, "reserve_request_attempt", exhaust_fallback)
        result = pipeline.run(run_id="run", crawl_run_id="crawl", observed_at=NOW)
        audit = database.connection.execute(
            """SELECT status, error_json FROM search_queries
               WHERE provider = 'crossref' AND page LIKE 'fallback:%'"""
        ).fetchone()
        attempts = database.connection.execute(
            "SELECT COUNT(*) FROM provider_request_attempts WHERE provider = 'crossref'"
        ).fetchone()[0]

    assert fallback.calls == 0
    assert attempts == 0
    assert result.status == "incomplete"
    assert result.fanout.budget_exhausted
    assert tuple(audit) == ("failed", '{"message":"budget_exhausted"}')


def test_fallback_metadata_handles_atomic_reservation_exhaustion_before_network(
    tmp_path, monkeypatch
) -> None:
    plan = _plan(
        [
            _provider("pmlr", ["venue_primary"]),
            _provider("crossref", ["metadata_enricher"]),
        ],
        required=["pmlr"],
        required_roles=("venue_primary",),
        max_requests=10,
        venue_fallbacks=(("crossref", "metadata_enricher"),),
    )

    class Primary:
        def discover(self, _descriptor, _window, _cursor):
            return SourceBatch(
                "primary",
                "primary",
                (_entry("pmlr", "candidate"),),
                None,
                EnvelopeStatus.PARTIAL,
                "truncated",
            )

    class Enricher:
        calls = 0

        def enrich(self, _raw):
            self.calls += 1
            raise AssertionError("fallback metadata network call must not start")

    enricher = Enricher()
    venue_run = VenueRun(
        VenueDescriptor(1, "testconf", "pmlr", "pmlr", {"volume_id": "v1"}),
        CrawlWindow(year=2024),
        VenueContext("venue-test", "testconf", "TestConf", "conference", "pmlr", {}),
        fallbacks=_venue_fallbacks(plan),
    )
    with Database(tmp_path / "papers.sqlite3") as database:
        database.migrate()
        pipeline = SearchPipeline(
            database,
            plan,
            clients={"pmlr": Primary(), "crossref": enricher},
            trusts={
                "pmlr": _trust("pmlr", primary=True),
                "crossref": _trust("crossref"),
            },
            venue_runs=(venue_run,),
            venue_only=True,
        )
        reserve = pipeline.runs.reserve_request_attempt

        def exhaust_fallback(**kwargs):
            if str(kwargs["operation_key"]).startswith("fallback:"):
                raise RequestBudgetExhausted("frozen request budget is exhausted")
            return reserve(**kwargs)

        monkeypatch.setattr(pipeline.runs, "reserve_request_attempt", exhaust_fallback)
        result = pipeline.run(run_id="run", crawl_run_id="crawl", observed_at=NOW)
        audit = database.connection.execute(
            """SELECT status, error_json FROM search_queries
               WHERE provider = 'crossref' AND role = 'metadata_enricher'"""
        ).fetchone()
        attempts = database.connection.execute(
            "SELECT COUNT(*) FROM provider_request_attempts WHERE provider = 'crossref'"
        ).fetchone()[0]

    assert enricher.calls == 0
    assert attempts == 0
    assert result.status == "incomplete"
    assert result.fanout.budget_exhausted
    assert audit["status"] == "failed"
    assert "budget_exhausted" in audit["error_json"]


@pytest.mark.parametrize(
    ("role", "operation"),
    (
        ("metadata_enricher", "metadata:enrich:"),
        ("metadata_verifier", "metadata:verify:"),
    ),
)
def test_metadata_handles_atomic_reservation_exhaustion_before_network(
    tmp_path, monkeypatch, role, operation
) -> None:
    plan = _plan(
        [_provider("openalex"), _provider("crossref", [role])],
        required=[],
        max_requests=10,
    )
    query_hash = next(
        item["native_query_hashes"][0]
        for item in plan["providers"]
        if item["provider"] == "openalex"
    )

    class MetadataClient:
        calls = 0

        def enrich(self, raw):
            self.calls += 1
            return EnrichmentResult(
                _entry("crossref", "enriched"), "crossref", raw.external_id
            )

        def verify(self, candidate, _evidence):
            self.calls += 1
            return VerificationResult(
                candidate, VerificationStatus.VERIFIED, "crossref", ("crossref",)
            )

    metadata = MetadataClient()
    with Database(tmp_path / "papers.sqlite3") as database:
        database.migrate()
        pipeline = SearchPipeline(
            database,
            plan,
            clients={
                "openalex": SearchFixture(
                    (_batch("openalex", query_hash, (_entry("openalex", "work"),)),)
                ),
                "crossref": metadata,
            },
            trusts={
                "openalex": _trust("openalex"),
                "crossref": _trust("crossref"),
            },
        )
        reserve = pipeline.runs.reserve_request_attempt

        def exhaust_metadata(**kwargs):
            if str(kwargs["operation_key"]).startswith(operation):
                raise RequestBudgetExhausted("frozen request budget is exhausted")
            return reserve(**kwargs)

        monkeypatch.setattr(pipeline.runs, "reserve_request_attempt", exhaust_metadata)
        result = pipeline.run(run_id="run", crawl_run_id="crawl", observed_at=NOW)
        audit = database.connection.execute(
            """SELECT status, error_json FROM search_queries
               WHERE provider = 'crossref' AND role = ?""",
            (role,),
        ).fetchone()
        attempts = database.connection.execute(
            "SELECT COUNT(*) FROM provider_request_attempts WHERE provider = 'crossref'"
        ).fetchone()[0]
        stats = json.loads(
            database.connection.execute(
                "SELECT stats_json FROM crawl_runs WHERE crawl_run_id = 'crawl'"
            ).fetchone()[0]
        )

    assert metadata.calls == 0
    assert attempts == 0
    assert result.status == "incomplete"
    assert audit["status"] == "failed"
    assert "budget_exhausted" in audit["error_json"]
    assert stats["budget"]["reason"] == "budget_exhausted"


def test_citation_handles_atomic_reservation_exhaustion_before_network(
    tmp_path, monkeypatch
) -> None:
    plan = _plan(
        [_provider("openalex", ["search", "citation"])],
        required=["openalex"],
        citation_directions=("references",),
        max_rounds=1,
        max_requests=10,
    )
    query_hash = plan["providers"][0]["native_query_hashes"][0]

    class CitationClient:
        calls = 0

        def references(self, _paper, _cursor):
            self.calls += 1
            raise AssertionError("citation network call must not start")

        citations = references

    class RelevantScreener:
        def screen(self, paper_ids):
            return {paper_id: FilterStatus.RELEVANT for paper_id in paper_ids}

        def reranker_score(self, _paper_id):
            return 1.0

    citation = CitationClient()
    with Database(tmp_path / "papers.sqlite3") as database:
        database.migrate()
        pipeline = SearchPipeline(
            database,
            plan,
            clients={"openalex": SearchFixture((_batch("openalex", query_hash, ()),))},
            trusts={"openalex": _trust("openalex")},
            citation_clients={"openalex": citation},
            screener=RelevantScreener(),
        )
        root = pipeline.repository.ingest(
            _scoped(
                SourceEntry(
                    "openalex", "root", "Root", ("Ada",), doi="10.1000/root", year=2024
                )
            )
        )
        reserve = pipeline.runs.reserve_request_attempt

        def exhaust_citation(**kwargs):
            if kwargs["role"] == "citation":
                raise RequestBudgetExhausted("frozen request budget is exhausted")
            return reserve(**kwargs)

        monkeypatch.setattr(pipeline.runs, "reserve_request_attempt", exhaust_citation)
        result = pipeline.run(
            run_id="run",
            crawl_run_id="crawl",
            observed_at=NOW,
            seed_paper_ids=(root.paper_id,),
        )
        request_audit = database.connection.execute(
            "SELECT status, error_json FROM citation_requests"
        ).fetchone()
        stop_reason = database.connection.execute(
            "SELECT stop_reason FROM search_rounds"
        ).fetchone()[0]
        attempts = database.connection.execute(
            "SELECT COUNT(*) FROM provider_request_attempts WHERE role = 'citation'"
        ).fetchone()[0]

    assert citation.calls == 0
    assert attempts == 0
    assert result.status == "incomplete"
    assert tuple(request_audit)[0] == "skipped_budget"
    assert "budget_exhausted" in request_audit["error_json"]
    assert stop_reason == "budget_exhausted"


def test_wave_requests_are_reserved_before_second_batch_persistence(
    tmp_path, monkeypatch
) -> None:
    plan = _plan(
        [_provider("crossref", ["search"]), _provider("openalex", ["search"])],
        required=["crossref", "openalex"],
        max_requests=2,
        max_candidates=10,
    )
    calls: list[str] = []
    reservations_seen: list[int] = []
    database_path = tmp_path / "papers.sqlite3"

    class Client:
        def __init__(self, name: str) -> None:
            self.name = name

        def search(self, query_spec, _cursor):
            calls.append(self.name)
            with Database(database_path, read_only=True) as audit:
                reservations_seen.append(
                    int(
                        audit.connection.execute(
                            """SELECT COUNT(*) FROM provider_request_attempts
                               WHERE crawl_run_id = 'crawl' AND provider = ?
                                 AND request_charged = 1 AND status = 'running'""",
                            (self.name,),
                        ).fetchone()[0]
                    )
                )
            return SourceBatch(
                self.name,
                query_spec.native_query_hash,
                (
                    replace(
                        _entry(self.name, self.name),
                        doi=f"10.1000/{self.name}",
                    ),
                ),
                None,
                EnvelopeStatus.SUCCESS,
            )

    with Database(database_path) as database:
        database.migrate()
        pipeline = SearchPipeline(
            database,
            plan,
            clients={
                "crossref": Client("crossref"),
                "openalex": Client("openalex"),
            },
            trusts={
                "crossref": _trust("crossref"),
                "openalex": _trust("openalex"),
            },
        )
        original = pipeline.runs.record_batch
        persisted_batches = 0

        def interrupt_second_batch(**kwargs):
            nonlocal persisted_batches
            persisted_batches += 1
            if persisted_batches == 2:
                raise RuntimeError("simulated crash before second batch persistence")
            return original(**kwargs)

        monkeypatch.setattr(pipeline.runs, "record_batch", interrupt_second_batch)
        with pytest.raises(
            RuntimeError, match="simulated crash before second batch persistence"
        ):
            pipeline.run(run_id="run", crawl_run_id="crawl", observed_at=NOW)
        charged_after_crash = database.connection.execute(
            """SELECT COUNT(*), SUM(request_charged), SUM(accepted_count)
               FROM provider_request_attempts
               WHERE crawl_run_id = 'crawl' AND request_charged = 1"""
        ).fetchone()

        monkeypatch.setattr(pipeline.runs, "record_batch", original)
        resumed = pipeline.run(
            run_id="run", crawl_run_id="crawl", observed_at=NOW
        )
        usage = json.loads(
            database.connection.execute(
                "SELECT stats_json FROM crawl_runs WHERE crawl_run_id = 'crawl'"
            ).fetchone()[0]
        )["campaign_usage_v1"]
        persisted_first_batch = database.connection.execute(
            """SELECT status, returned_count FROM search_queries
               WHERE provider = 'crossref' AND requested_at = ?""",
            (NOW,),
        ).fetchone()

    assert sorted(calls) == ["crossref", "openalex"]
    assert sorted(reservations_seen) == [1, 1]
    assert tuple(charged_after_crash) == (2, 2, 2)
    assert resumed.status == "incomplete"
    assert tuple(persisted_first_batch) == ("complete", 1)
    assert usage["requests_made"] == 2
    assert usage["candidates_returned"] == 2


def test_time_budget_is_cumulative_across_crawl_resume(tmp_path, monkeypatch) -> None:
    plan = _plan(
        [_provider("openalex", ["search"])],
        required=["openalex"],
        max_requests=10,
        max_candidates=10,
        max_seconds=2,
    )
    clock = {"monotonic": 0.0, "epoch": 1_000.0}
    monkeypatch.setattr(
        "paper_agent.search_pipeline.time.monotonic",
        lambda: clock["monotonic"],
    )
    monkeypatch.setattr(
        "paper_agent.search_pipeline.time.time", lambda: clock["epoch"]
    )

    class SlowClient:
        calls = 0

        def search(self, query_spec, _cursor):
            self.calls += 1
            clock["monotonic"] += 3.0
            clock["epoch"] += 3.0
            return SourceBatch(
                "openalex",
                query_spec.native_query_hash,
                (),
                None,
                EnvelopeStatus.SUCCESS,
            )

    client = SlowClient()
    with Database(tmp_path / "papers.sqlite3") as database:
        database.migrate()
        pipeline = SearchPipeline(
            database,
            plan,
            clients={"openalex": client},
            trusts={"openalex": _trust("openalex")},
        )
        pipeline.run(run_id="run", crawl_run_id="crawl", observed_at=NOW)
        pipeline.run(run_id="run", crawl_run_id="crawl", observed_at=NOW)
        elapsed = database.connection.execute(
            """SELECT SUM(elapsed_seconds) FROM crawl_execution_attempts
               WHERE crawl_run_id = 'crawl'"""
        ).fetchone()[0]

    assert client.calls == 1
    assert elapsed >= 3.0


@pytest.mark.parametrize(
    ("column", "mismatched_value"),
    (
        ("input_hash", "wrong-input"),
        ("config_hash", "wrong-config"),
        ("implementation_version", "wrong-implementation"),
    ),
)
def test_existing_run_binding_mismatch_fails_before_fanout(
    tmp_path,
    column: str,
    mismatched_value: str,
) -> None:
    plan = _plan([_provider("openalex")], required=["openalex"])

    class RecordingClient:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, provider, queries):
            self.calls += 1
            return ()

    client = RecordingClient()
    binding = {
        "input_hash": plan["plan_hash"],
        "config_hash": plan["filter"]["config_hash"],
        "implementation_version": SEARCH_IMPLEMENTATION_VERSION,
    }
    binding[column] = mismatched_value
    with Database(tmp_path / "papers.sqlite3") as database:
        database.migrate()
        database.connection.execute(
            """INSERT INTO pipeline_runs(
                   run_id, stage, status, input_hash, config_hash,
                   implementation_version, started_at
               ) VALUES (?, 'stage-1', 'running', ?, ?, ?, ?)""",
            (
                "run",
                binding["input_hash"],
                binding["config_hash"],
                binding["implementation_version"],
                NOW,
            ),
        )
        database.connection.commit()

        with pytest.raises(ValueError, match="different frozen inputs"):
            SearchPipeline(
                database,
                plan,
                runtime_providers=plan["providers"],
                clients={"openalex": client},
                trusts={"openalex": _trust("openalex")},
            ).run(run_id="run", crawl_run_id="crawl", observed_at=NOW)

        assert client.calls == 0
        assert database.connection.execute("SELECT COUNT(*) FROM crawl_runs").fetchone()[0] == 0


def test_pipeline_persists_incremental_snapshot_and_does_not_remove_after_source_failure(tmp_path) -> None:
    plan = _plan([_provider("openalex")], required=["openalex"])
    query_hash = plan["providers"][0]["native_query_hashes"][0]
    first = SourceEntry("openalex", "first", "First", doi="10.1000/first", metadata={"publication_version": "published"})
    second = SourceEntry("openalex", "second", "Second", doi="10.1000/second", metadata={"publication_version": "published"})
    client = SearchFixture((_batch("openalex", query_hash, (first,)),))
    with Database(tmp_path / "papers.sqlite3") as database:
        database.migrate()
        pipeline = SearchPipeline(
            database, plan, runtime_providers=plan["providers"], clients={"openalex": client},
            trusts={"openalex": _trust("openalex")},
        )
        first_result = pipeline.run(run_id="run-1", crawl_run_id="crawl-1", observed_at=NOW)
        client.batches = (_batch("openalex", query_hash, (second,)),)
        second_result = pipeline.run(run_id="run-2", crawl_run_id="crawl-2", observed_at=NOW)
        changes = database.connection.execute(
            "SELECT paper_id, change_kind FROM incremental_diff_papers WHERE crawl_run_id = 'crawl-2' ORDER BY change_kind, paper_id"
        ).fetchall()
        assert [(row["paper_id"], row["change_kind"]) for row in changes] == [
            (second_result.paper_ids[0], "new"),
            (first_result.paper_ids[0], "removed"),
        ]

        client.failure = "offline"
        pipeline.run(run_id="run-3", crawl_run_id="crawl-3", observed_at=NOW)
        assert database.connection.execute(
            "SELECT COUNT(*) FROM incremental_diff_papers WHERE crawl_run_id = 'crawl-3'"
        ).fetchone()[0] == 0
        watermark = database.connection.execute(
            "SELECT watermark_json FROM provider_watermarks WHERE provider = 'openalex' AND descriptor_key = 'query:q'"
        ).fetchone()
        assert watermark is not None


def test_venue_crawl_uses_scoped_watermark_but_explicit_history_replay_bypasses_it(tmp_path) -> None:
    plan = _plan([_provider("openreview", ["venue_primary", "search"])], required=[])
    descriptor = VenueDescriptor(1, "testconf", "openreview", "openreview")
    with Database(tmp_path / "papers.sqlite3") as database:
        database.migrate()
        from paper_agent.search_runs import SearchRunCoordinator

        SearchRunCoordinator(database).set_watermark(
            "openreview", "testconf", {"cursor": "page-2"}, updated_at=NOW
        )
        pipeline = SearchPipeline(
            database, plan, runtime_providers=plan["providers"], clients={}, trusts={},
            venue_runs=(VenueRun(descriptor, CrawlWindow(date_from="2024-01-01", date_to="2024-12-31", year=2024), _trust("openreview")),),
        )
        assert pipeline._watermarked_venue_runs()[0].cursor == "page-2"
        pipeline.venue_runs = (
            VenueRun(
                descriptor,
                CrawlWindow(date_from="2024-01-01", date_to="2024-12-31", year=2024),
                _trust("openreview"),
                historical_replay=True,
            ),
        )
        assert pipeline._watermarked_venue_runs()[0].cursor is None


def test_historical_venue_replay_keeps_the_normal_watermark(tmp_path) -> None:
    plan = _plan(
        [_provider("openreview", ["venue_primary", "search"])],
        required=["openreview"],
        required_roles=("venue_primary",),
        venue_descriptor={
            "schema_version": "1",
            "venue_id": "testconf",
            "name": "TestConf",
            "venue_type": "conference",
            "primary_provider": "openreview",
            "provider_params": {},
        },
    )
    descriptor = VenueDescriptor(1, "testconf", "openreview", "openreview")

    class VenueClient:
        cursor: str | None = "not-called"

        def discover(self, _descriptor, _window, cursor):
            self.cursor = cursor
            return SourceBatch("history", "history", (), None, EnvelopeStatus.SUCCESS)

    with Database(tmp_path / "papers.sqlite3") as database:
        database.migrate()
        from paper_agent.search_runs import SearchRunCoordinator

        SearchRunCoordinator(database).set_watermark(
            "openreview", "testconf", {"cursor": "page-2"}, updated_at=NOW
        )
        client = VenueClient()
        SearchPipeline(
            database,
            plan,
            runtime_providers=plan["providers"],
            clients={"openreview": client},
            trusts={"openreview": _trust("openreview", primary=True)},
            venue_runs=(
                VenueRun(
                    descriptor,
                    CrawlWindow(date_from="2024-01-01", date_to="2024-12-31", year=2024),
                    VenueContext("venue", "testconf", "TestConf", "conference", "openreview", {}),
                    historical_replay=True,
                ),
            ),
            venue_only=True,
        ).run(run_id="history", crawl_run_id="history-crawl", observed_at=NOW)
        assert client.cursor is None
        watermark = database.connection.execute(
            "SELECT watermark_json FROM provider_watermarks WHERE provider = 'openreview' AND descriptor_key = 'testconf'"
        ).fetchone()
        assert json.loads(watermark["watermark_json"]) == {"cursor": "page-2"}


def test_protocol_client_uses_frozen_native_query_and_audits_each_page(tmp_path) -> None:
    plan = _plan([_provider("openalex")], required=["openalex"])

    class PaginatedClient:
        def search(self, query_spec, cursor):
            suffix = "first" if cursor is None else "second"
            return SourceBatch(
                "provider-source",
                query_spec.native_query_hash,
                (
                    SourceEntry(
                        "openalex",
                        suffix,
                        f"Paper {suffix}",
                        ("Ada Lovelace",),
                        doi=f"10.1000/{suffix}",
                        year=2024,
                    ),
                ),
                "page-2" if cursor is None else None,
                EnvelopeStatus.SUCCESS,
            )

    with Database(tmp_path / "papers.sqlite3") as database:
        database.migrate()
        result = SearchPipeline(
            database,
            plan,
            clients={"openalex": PaginatedClient()},
            trusts={"openalex": _trust("openalex")},
        ).run(run_id="run", crawl_run_id="crawl", observed_at=NOW)
        queries = database.connection.execute(
            "SELECT page, cursor, query_hash FROM search_queries ORDER BY page"
        ).fetchall()

    assert len(result.paper_ids) == 2
    assert [(row["page"], row["cursor"]) for row in queries] == [("1", None), ("2", "page-2")]
    assert {row["query_hash"] for row in queries} == {plan["providers"][0]["native_query_hashes"][0]}


def test_exact_venue_provider_routes_discover_without_a_search_compiler(tmp_path) -> None:
    providers = [_provider("pmlr", ["venue_primary"]), _provider("openalex")]
    plan = _plan(
        providers,
        required=["pmlr"],
        venue_descriptor={
            "schema_version": "1",
            "venue_id": "icml",
            "name": "ICML 2024",
            "venue_type": "conference",
            "primary_provider": "pmlr",
            "provider_params": {"volume_id": "v235"},
        },
    )
    openalex_hash = next(
        item["native_query_hashes"][0]
        for item in plan["providers"]
        if item["provider"] == "openalex"
    )
    pmlr = create_builtin(
        "pmlr",
        FixtureTransport(
            {
                "pmlr:discover:first": {
                    "entries": [
                        {
                            "id": "pmlr-v235-1",
                            "title": "Official ICML Paper",
                            "doi": "10.1000/icml.official",
                            "year": 2024,
                            "fields": ["computer science"],
                            "language": "en",
                            "document_type": "article",
                        }
                    ]
                }
            }
        ),
    )
    venue_run = VenueRun(
        VenueDescriptor(1, "icml", "pmlr", "pmlr", {"volume_id": "v235"}),
        CrawlWindow(year=2024),
        VenueContext("icml-2024", "icml", "ICML 2024", "conference", "pmlr", {}),
    )

    with Database(tmp_path / "papers.sqlite3") as database:
        database.migrate()
        result = SearchPipeline(
            database,
            plan,
            clients={
                "pmlr": pmlr,
                "openalex": SearchFixture((_batch("openalex", openalex_hash, ()),)),
            },
            trusts={"pmlr": _trust("pmlr", primary=True), "openalex": _trust("openalex")},
            venue_runs=(venue_run,),
        ).run(run_id="run", crawl_run_id="crawl", observed_at=NOW)
        sources = database.connection.execute(
            "SELECT provider, role, status FROM source_runs ORDER BY provider"
        ).fetchall()

    assert result.status == "complete"
    assert len(result.paper_ids) == 1
    assert [tuple(row) for row in sources] == [
        ("openalex", "search:query:q", "complete"),
        ("pmlr", "venue_primary:icml", "complete"),
    ]


def test_user_library_seeds_enter_the_same_single_writer_pipeline(tmp_path) -> None:
    providers = [_provider("user_library", ["library"]), _provider("openalex")]
    plan = _plan(providers, required=[])
    openalex_hash = next(
        item["native_query_hashes"][0]
        for item in plan["providers"]
        if item["provider"] == "openalex"
    )

    with Database(tmp_path / "papers.sqlite3") as database:
        database.migrate()
        result = SearchPipeline(
            database,
            plan,
            clients={
                "user_library": create_builtin("user_library", FixtureTransport({})),
                "openalex": SearchFixture((_batch("openalex", openalex_hash, ()),)),
            },
            trusts={"user_library": _trust("user_library"), "openalex": _trust("openalex")},
            seed_inputs=(SeedInput("doi", "10.1000/user-seed"),),
        ).run(run_id="run", crawl_run_id="crawl", observed_at=NOW)
        source = database.connection.execute(
            "SELECT provider, role, status FROM source_runs WHERE provider = 'user_library'"
        ).fetchone()

    assert len(result.paper_ids) == 1
    assert tuple(source) == ("user_library", "library", "complete")


class CitationFixture:
    def __init__(self, seed_id: str, cited_id: str, skipped_id: str, citing_id: str) -> None:
        self.seed_id, self.cited_id, self.skipped_id, self.citing_id = seed_id, cited_id, skipped_id, citing_id

    def references(self, seed, cursor):
        return SourceCitationBatch.references(self.seed_id, self.cited_id, self.skipped_id)

    def citations(self, seed, cursor):
        return SourceCitationBatch.citations(self.seed_id, self.citing_id)


class SourceCitationBatch:
    @staticmethod
    def references(seed: str, cited: str, skipped: str):
        from paper_agent.domain import CitationBatch
        return CitationBatch("fixture", "refs", (
            reference_edge(seed, cited, provider="openalex", observed_at=NOW, raw_evidence={}),
            reference_edge(seed, skipped, provider="openalex", observed_at=NOW, raw_evidence={}),
        ), None, EnvelopeStatus.SUCCESS)

    @staticmethod
    def citations(seed: str, citing: str):
        from paper_agent.domain import CitationBatch
        return CitationBatch("fixture", "cites", (citation_edge(seed, citing, provider="openalex", observed_at=NOW, raw_evidence={}),), None, EnvelopeStatus.SUCCESS)


def test_relevant_round_zero_discovery_becomes_a_citation_root_with_screening_audit(tmp_path) -> None:
    plan = _plan([_provider("openalex", ["search", "citation"])], required=["openalex"])
    query_hash = plan["providers"][0]["native_query_hashes"][0]

    class RelevantScreener:
        def screen(self, paper_ids):
            return {paper_id: FilterStatus.RELEVANT for paper_id in paper_ids}

        def reranker_score(self, paper_id):
            return 0.91

    class RecordingCitations:
        def __init__(self):
            self.seeds = []

        def references(self, seed, cursor):
            self.seeds.append(seed.paper_id)
            return CitationBatch("fixture", "refs", (), None, EnvelopeStatus.SUCCESS)

        def citations(self, seed, cursor):
            self.seeds.append(seed.paper_id)
            return CitationBatch("fixture", "cites", (), None, EnvelopeStatus.SUCCESS)

    citation_client = RecordingCitations()
    with Database(tmp_path / "papers.sqlite3") as database:
        database.migrate()
        result = SearchPipeline(
            database,
            plan,
            runtime_providers=plan["providers"],
            clients={
                "openalex": SearchFixture((
                    _batch(
                        "openalex",
                        query_hash,
                        (SourceEntry("openalex", "root", "Relevant root", doi="10.1000/root"),),
                    ),
                ))
            },
            trusts={"openalex": _trust("openalex")},
            citation_clients={"openalex": citation_client},
            screener=RelevantScreener(),
        ).run(run_id="run", crawl_run_id="crawl", observed_at=NOW)
        root_id = database.connection.execute(
            "SELECT paper_id FROM papers WHERE doi = '10.1000/root'"
        ).fetchone()[0]
        audit = database.connection.execute(
            "SELECT screened, included, excluded FROM source_run_audits"
        ).fetchone()

    assert len(result.citation_round_ids) == 1
    assert citation_client.seeds == [root_id, root_id]
    assert tuple(audit) == (1, 1, 0)


def test_citation_round_screens_every_candidate_and_respects_depth_and_cap(tmp_path) -> None:
    plan = _plan([_provider("openalex", ["search", "citation"])], required=["openalex"], citation_cap=1)
    query_hash = plan["providers"][0]["native_query_hashes"][0]
    entries = (
        SourceEntry("openalex", "seed", "Seed", ("Ada",), doi="10.1000/seed", year=2024),
        SourceEntry("openalex", "cited", "Cited", ("Ada",), doi="10.1000/cited", year=2024),
        SourceEntry("openalex", "skipped", "Skipped", ("Ada",), doi="10.1000/skipped", year=2024),
        SourceEntry("openalex", "citing", "Citing", ("Ada",), doi="10.1000/citing", year=2024),
    )
    with Database(tmp_path / "papers.sqlite3") as database:
        database.migrate()
        screener = DeterministicFakeScreener(frozenset())
        pipeline = SearchPipeline(
            database, plan, runtime_providers=plan["providers"],
            clients={"openalex": SearchFixture((_batch("openalex", query_hash, entries),))},
            trusts={"openalex": _trust("openalex")},
            citation_clients={}, screener=screener,
        )
        first = pipeline.run(run_id="run-search", crawl_run_id="crawl-search", observed_at=NOW)
        papers = {row["doi"]: row["paper_id"] for row in database.connection.execute("SELECT paper_id, doi FROM papers")}
        screener.screened.clear()
        pipeline.citation_clients = {"openalex": CitationFixture(papers["10.1000/seed"], papers["10.1000/cited"], papers["10.1000/skipped"], papers["10.1000/citing"])}
        second = pipeline.run(run_id="run-cite", crawl_run_id="crawl-cite", observed_at=NOW, seed_paper_ids=[papers["10.1000/seed"]])
        assert first.citation_round_ids == ()
        assert len(second.citation_round_ids) == 1
        assert screener.screened == sorted(papers.values())
        assert database.connection.execute("SELECT COUNT(*) FROM citation_requests").fetchone()[0] == 2
        assert database.connection.execute("SELECT COUNT(*) FROM citation_edges").fetchone()[0] == 2


def test_citation_candidate_is_canonicalized_ingested_screened_and_persisted(tmp_path) -> None:
    plan = _plan([_provider("openalex", ["search", "citation"])], required=["openalex"])
    query_hash = plan["providers"][0]["native_query_hashes"][0]

    class CandidateCitationFixture:
        def references(self, seed, cursor):
            return CitationBatch(
                "fixture", "refs", (
                    CitationEdge(
                        source_paper_id=seed.paper_id,
                        target_paper_id="https://openalex.org/W-native-only",
                        edge_type=CitationEdgeType.REFERENCES,
                        provider="openalex",
                        observed_at=NOW,
                        raw_evidence={"native_id": "https://openalex.org/W-native-only"},
                        candidate=_scoped(SourceEntry(
                            "openalex", "https://openalex.org/W-native-only", "Discovered only by citation",
                            ("Ada",), doi="10.1000/discovered", year=2024,
                        )),
                    ),
                ), None, EnvelopeStatus.SUCCESS,
            )

        def citations(self, seed, cursor):
            return CitationBatch("fixture", "cites", (), None, EnvelopeStatus.SUCCESS)

    with Database(tmp_path / "papers.sqlite3") as database:
        database.migrate()
        pipeline = SearchPipeline(
            database, plan, runtime_providers=plan["providers"],
            clients={"openalex": SearchFixture((_batch("openalex", query_hash, ()),))},
            trusts={"openalex": _trust("openalex")},
            citation_clients={"openalex": CandidateCitationFixture()},
            screener=DeterministicFakeScreener(frozenset()),
        )
        root = pipeline.repository.ingest(_scoped(SourceEntry("openalex", "root", "Root", ("Ada",), doi="10.1000/root", year=2024)))
        result = pipeline.run(
            run_id="run",
            crawl_run_id="crawl",
            observed_at=NOW,
            seed_paper_ids=[root.paper_id],
        )

        discovered = database.connection.execute(
            "SELECT paper_id FROM papers WHERE doi = '10.1000/discovered'"
        ).fetchone()["paper_id"]
        edge = database.connection.execute(
            "SELECT source_paper_id, target_paper_id, raw_evidence_json FROM citation_edges"
        ).fetchone()
        round_paper = database.connection.execute(
            "SELECT depth, subquestion_id, screening_status FROM search_round_papers WHERE paper_id = ?",
            (discovered,),
        ).fetchone()
        citation_attempts = database.connection.execute(
            """SELECT a.citation_request_id, a.source_run_id,
                      a.response_artifact_id, a.query_hash, a.response_hash
               FROM provider_request_attempts a
               JOIN citation_requests c
                 ON c.citation_request_id = a.citation_request_id
               WHERE a.role = 'citation'
               ORDER BY c.schedule_order"""
        ).fetchall()

    assert pipeline.screener.screened == [root.paper_id, discovered]
    assert result.eligible_paper_ids == tuple(sorted((root.paper_id, discovered)))
    assert tuple(edge[:2]) == (root.paper_id, discovered)
    assert "W-native-only" in edge[2]
    assert tuple(round_paper) == (1, "q", "irrelevant")
    assert len(citation_attempts) == 2
    assert all(row[0] is not None for row in citation_attempts)
    assert all(tuple(row[1:3]) == (None, None) for row in citation_attempts)
    assert all(len(row[3]) == len(row[4]) == 64 for row in citation_attempts)


def test_out_of_scope_citation_is_not_stage2_screened_or_eligible(tmp_path) -> None:
    plan = _plan([_provider("openalex", ["search", "citation"])], required=["openalex"])
    query_hash = plan["providers"][0]["native_query_hashes"][0]

    class CitationFixture:
        def references(self, seed, cursor):
            return CitationBatch(
                "fixture",
                "refs",
                (
                    CitationEdge(
                        seed.paper_id,
                        "native-wrong-language",
                        CitationEdgeType.REFERENCES,
                        "openalex",
                        NOW,
                        candidate=_scoped(
                            SourceEntry(
                                "openalex",
                                "native-wrong-language",
                                "Wrong language",
                                ("Ada",),
                                doi="10.1000/wrong-language-citation",
                                year=2024,
                                metadata={"language": "zh"},
                            )
                        ),
                    ),
                ),
                None,
                EnvelopeStatus.SUCCESS,
            )

        def citations(self, seed, cursor):
            return CitationBatch("fixture", "cites", (), None, EnvelopeStatus.SUCCESS)

    class RecordingScreener:
        def __init__(self) -> None:
            self.screened: list[str] = []

        def screen(self, paper_ids):
            self.screened.extend(paper_ids)
            return {paper_id: FilterStatus.RELEVANT for paper_id in paper_ids}

        def reranker_score(self, paper_id):
            return 1.0

    screener = RecordingScreener()
    with Database(tmp_path / "papers.sqlite3") as database:
        database.migrate()
        pipeline = SearchPipeline(
            database,
            plan,
            runtime_providers=plan["providers"],
            clients={"openalex": SearchFixture((_batch("openalex", query_hash, ()),))},
            trusts={"openalex": _trust("openalex")},
            citation_clients={"openalex": CitationFixture()},
            screener=screener,
        )
        root = pipeline.repository.ingest(
            _scoped(
                SourceEntry(
                    "openalex", "root", "Root", ("Ada",), doi="10.1000/root", year=2024
                )
            )
        )
        result = pipeline.run(
            run_id="run",
            crawl_run_id="crawl",
            observed_at=NOW,
            seed_paper_ids=[root.paper_id],
        )
        excluded = database.connection.execute(
            "SELECT paper_id FROM papers WHERE doi = '10.1000/wrong-language-citation'"
        ).fetchone()["paper_id"]

    assert screener.screened == [root.paper_id]
    assert result.eligible_paper_ids == (root.paper_id,)
    assert excluded not in result.eligible_paper_ids


def test_citation_pages_consume_the_global_request_budget(tmp_path) -> None:
    plan = _plan(
        [_provider("openalex", ["search", "citation"])],
        required=["openalex"],
        max_requests=3,
        max_candidates=10,
        citation_cap=5,
    )
    query_hash = plan["providers"][0]["native_query_hashes"][0]

    class PaginatedCitationFixture:
        reference_cursors: list[str | None] = []

        def citations(self, seed, cursor):
            return CitationBatch("fixture", "cites", (), None, EnvelopeStatus.SUCCESS)

        def references(self, seed, cursor):
            self.reference_cursors.append(cursor)
            number = 1 if cursor is None else 2
            return CitationBatch(
                "fixture",
                f"refs-{number}",
                (
                    CitationEdge(
                        seed.paper_id,
                        f"native-{number}",
                        CitationEdgeType.REFERENCES,
                        "openalex",
                        NOW,
                        candidate=_scoped(SourceEntry(
                            "openalex",
                            f"native-{number}",
                            f"Candidate {number}",
                            ("Ada",),
                            doi=f"10.1000/page-{number}",
                            year=2024,
                        )),
                    ),
                ),
                "next" if cursor is None else None,
                EnvelopeStatus.SUCCESS,
            )

    client = PaginatedCitationFixture()
    with Database(tmp_path / "papers.sqlite3") as database:
        database.migrate()
        pipeline = SearchPipeline(
            database,
            plan,
            clients={"openalex": SearchFixture((_batch("openalex", query_hash, ()),))},
            trusts={"openalex": _trust("openalex")},
            citation_clients={"openalex": client},
            screener=DeterministicFakeScreener(frozenset()),
        )
        root = pipeline.repository.ingest(
            _scoped(SourceEntry("openalex", "root", "Root", ("Ada",), doi="10.1000/root", year=2024))
        )
        pipeline.run(
            run_id="run", crawl_run_id="crawl", observed_at=NOW, seed_paper_ids=[root.paper_id]
        )

        assert database.connection.execute("SELECT COUNT(*) FROM citation_edges").fetchone()[0] == 1
        assert database.connection.execute(
            "SELECT stop_reason FROM search_rounds"
        ).fetchone()[0] == "budget_exhausted"

    assert client.reference_cursors == [None]


def test_exact_citation_request_cap_with_complete_pages_exhausts_sources(tmp_path) -> None:
    plan = _plan(
        [_provider("openalex", ["search", "citation"])],
        required=["openalex"],
        max_requests=2,
        citation_directions=("references",),
    )
    query_hash = plan["providers"][0]["native_query_hashes"][0]

    class CompleteCitationFixture:
        def references(self, _seed, _cursor):
            return CitationBatch("fixture", "refs", (), None, EnvelopeStatus.SUCCESS)

    with Database(tmp_path / "papers.sqlite3") as database:
        database.migrate()
        pipeline = SearchPipeline(
            database,
            plan,
            runtime_providers=plan["providers"],
            clients={"openalex": SearchFixture((_batch("openalex", query_hash, ()),))},
            trusts={"openalex": _trust("openalex")},
            citation_clients={"openalex": CompleteCitationFixture()},
        )
        root = pipeline.repository.ingest(
            _scoped(SourceEntry("openalex", "root", "Root", ("Ada",), doi="10.1000/root", year=2024))
        )
        result = pipeline.run(run_id="run", crawl_run_id="crawl", observed_at=NOW, seed_paper_ids=[root.paper_id])
        round_state = database.connection.execute(
            "SELECT stop_reason, limited_scope FROM search_rounds"
        ).fetchone()

    assert result.status == "complete"
    assert tuple(round_state) == ("sources_exhausted", 0)


def test_final_citation_page_larger_than_candidate_capacity_is_budget_exhausted(tmp_path) -> None:
    plan = _plan(
        [_provider("openalex", ["search", "citation"])],
        required=["openalex"],
        max_requests=4,
        max_candidates=1,
        citation_directions=("references",),
    )
    query_hash = plan["providers"][0]["native_query_hashes"][0]

    class OversizedFinalPage:
        def references(self, seed, _cursor):
            return CitationBatch(
                "fixture",
                "refs",
                tuple(
                    CitationEdge(
                        seed.paper_id,
                        f"native-{number}",
                        CitationEdgeType.REFERENCES,
                        "openalex",
                        NOW,
                        candidate=_scoped(SourceEntry(
                            "openalex", f"native-{number}", f"Paper {number}",
                            ("Ada",), doi=f"10.1000/final-{number}", year=2024,
                        )),
                    )
                    for number in (1, 2)
                ),
                None,
                EnvelopeStatus.SUCCESS,
            )

    with Database(tmp_path / "papers.sqlite3") as database:
        database.migrate()
        pipeline = SearchPipeline(
            database,
            plan,
            runtime_providers=plan["providers"],
            clients={"openalex": SearchFixture((_batch("openalex", query_hash, ()),))},
            trusts={"openalex": _trust("openalex")},
            citation_clients={"openalex": OversizedFinalPage()},
        )
        root = pipeline.repository.ingest(
            _scoped(SourceEntry("openalex", "root", "Root", ("Ada",), doi="10.1000/root", year=2024))
        )
        pipeline.run(run_id="run", crawl_run_id="crawl", observed_at=NOW, seed_paper_ids=[root.paper_id])
        round_state = database.connection.execute(
            "SELECT stop_reason, limited_scope FROM search_rounds"
        ).fetchone()

    assert tuple(round_state) == ("budget_exhausted", 1)


def test_partial_citation_page_without_error_is_unresolved_not_exhausted(tmp_path) -> None:
    plan = _plan(
        [_provider("openalex", ["search", "citation"])],
        required=["openalex"],
        citation_directions=("references",),
    )
    query_hash = plan["providers"][0]["native_query_hashes"][0]

    class PartialPage:
        def references(self, _seed, _cursor):
            return CitationBatch("fixture", "refs", (), None, EnvelopeStatus.PARTIAL)

    with Database(tmp_path / "papers.sqlite3") as database:
        database.migrate()
        pipeline = SearchPipeline(
            database,
            plan,
            runtime_providers=plan["providers"],
            clients={"openalex": SearchFixture((_batch("openalex", query_hash, ()),))},
            trusts={"openalex": _trust("openalex")},
            citation_clients={"openalex": PartialPage()},
        )
        root = pipeline.repository.ingest(
            _scoped(SourceEntry("openalex", "root", "Root", ("Ada",), doi="10.1000/root", year=2024))
        )
        result = pipeline.run(
            run_id="run", crawl_run_id="crawl", observed_at=NOW, seed_paper_ids=[root.paper_id]
        )
        round_state = database.connection.execute(
            "SELECT stop_reason, limited_scope FROM search_rounds"
        ).fetchone()

    assert result.status == "incomplete"
    assert tuple(round_state) == ("saturated_with_unresolved", 1)


def test_citation_depth_propagates_to_a_second_round_and_exact_candidate_cap_exhausts_sources(tmp_path) -> None:
    plan = _plan(
        [_provider("openalex", ["search", "citation"])], required=["openalex"],
        max_depth=2, max_rounds=2, max_requests=5, max_candidates=2,
    )
    query_hash = plan["providers"][0]["native_query_hashes"][0]

    class TwoDepthFixture:
        def references(self, seed, cursor):
            number = "one" if seed.doi == "10.1000/root" else "two"
            return CitationBatch(
                "fixture", number, (
                    CitationEdge(
                        seed.paper_id, f"native-{number}", CitationEdgeType.REFERENCES, "openalex", NOW,
                        candidate=_scoped(SourceEntry("openalex", f"native-{number}", number, ("Ada",), doi=f"10.1000/{number}", year=2024)),
                    ),
                ), None, EnvelopeStatus.SUCCESS,
            )

        def citations(self, seed, cursor):
            return CitationBatch("fixture", "cites", (), None, EnvelopeStatus.SUCCESS)

    with Database(tmp_path / "papers.sqlite3") as database:
        database.migrate()
        pipeline = SearchPipeline(
            database, plan, runtime_providers=plan["providers"],
            clients={"openalex": SearchFixture((_batch("openalex", query_hash, ()),))},
            trusts={"openalex": _trust("openalex")}, citation_clients={"openalex": TwoDepthFixture()},
            screener=DeterministicFakeScreener(frozenset()),
        )
        root = pipeline.repository.ingest(_scoped(SourceEntry("openalex", "root", "Root", ("Ada",), doi="10.1000/root", year=2024)))

        class RelevantScreener:
            screened: list[str] = []

            def screen(self, paper_ids):
                self.screened.extend(paper_ids)
                return {paper_id: FilterStatus.RELEVANT for paper_id in paper_ids}

            def reranker_score(self, paper_id):
                return 1.0

        pipeline.screener = RelevantScreener()
        pipeline.run(run_id="run", crawl_run_id="crawl", observed_at=NOW, seed_paper_ids=[root.paper_id])

        rows = database.connection.execute(
            "SELECT p.doi, srp.depth, srp.subquestion_id FROM search_round_papers srp JOIN papers p ON p.paper_id = srp.paper_id ORDER BY srp.depth"
        ).fetchall()
        round_state = database.connection.execute(
            "SELECT stop_reason, limited_scope FROM search_rounds ORDER BY round_index DESC LIMIT 1"
        ).fetchone()
        second_round_seed = database.connection.execute(
            "SELECT parent_round, depth, subquestion_id FROM search_round_seeds WHERE paper_id != ?",
            (root.paper_id,),
        ).fetchone()

    assert [tuple(row) for row in rows] == [("10.1000/one", 1, "q"), ("10.1000/two", 2, "q")]
    assert tuple(round_state) == ("sources_exhausted", 0)
    assert tuple(second_round_seed) == (0, 1, "q")


def test_failed_citation_round_is_incomplete_and_never_sources_exhausted(tmp_path) -> None:
    plan = _plan(
        [_provider("openalex", ["search", "citation"])], required=["openalex"], max_requests=4,
    )
    query_hash = plan["providers"][0]["native_query_hashes"][0]

    class FailedCitationFixture:
        def references(self, seed, cursor):
            raise RuntimeError("citation service unavailable")

        def citations(self, seed, cursor):
            return CitationBatch("fixture", "cites", (), None, EnvelopeStatus.SUCCESS)

    with Database(tmp_path / "papers.sqlite3") as database:
        database.migrate()
        pipeline = SearchPipeline(
            database, plan, runtime_providers=plan["providers"],
            clients={"openalex": SearchFixture((_batch("openalex", query_hash, ()),))},
            trusts={"openalex": _trust("openalex")}, citation_clients={"openalex": FailedCitationFixture()},
        )
        root = pipeline.repository.ingest(_scoped(SourceEntry("openalex", "root", "Root", ("Ada",), doi="10.1000/root", year=2024)))
        result = pipeline.run(run_id="run", crawl_run_id="crawl", observed_at=NOW, seed_paper_ids=[root.paper_id])
        round_state = database.connection.execute(
            "SELECT stop_reason, limited_scope FROM search_rounds"
        ).fetchone()
        audit = database.connection.execute(
            "SELECT screening_complete, source_failed FROM search_round_audits"
        ).fetchone()

    assert result.status == "incomplete"
    assert tuple(round_state) == ("saturated_with_unresolved", 1)
    assert tuple(audit) == (1, 1)


def test_time_budget_marks_the_citation_round_limited(tmp_path, monkeypatch) -> None:
    plan = _plan(
        [_provider("openalex", ["search", "citation"])], required=["openalex"], max_seconds=1,
    )
    query_hash = plan["providers"][0]["native_query_hashes"][0]
    clock = iter((0.0, 2.0, 4.0, *([6.0] * 30)))
    monkeypatch.setattr("paper_agent.search_pipeline.time.monotonic", lambda: next(clock))

    with Database(tmp_path / "papers.sqlite3") as database:
        database.migrate()
        pipeline = SearchPipeline(
            database, plan, runtime_providers=plan["providers"],
            clients={"openalex": SearchFixture((_batch("openalex", query_hash, ()),))},
            trusts={"openalex": _trust("openalex")}, citation_clients={"openalex": object()},
        )
        root = pipeline.repository.ingest(_scoped(SourceEntry("openalex", "root", "Root", ("Ada",), doi="10.1000/root", year=2024)))
        result = pipeline.run(run_id="run", crawl_run_id="crawl", observed_at=NOW, seed_paper_ids=[root.paper_id])
        row = database.connection.execute(
            "SELECT stop_reason, limited_scope FROM search_rounds"
        ).fetchone()

    assert result.status == "incomplete"
    assert tuple(row) == ("budget_exhausted", 1)


def test_library_seed_is_used_as_a_citation_root_without_a_canonical_id(tmp_path) -> None:
    plan = _plan(
        [_provider("user_library", ["library"]), _provider("openalex", ["search", "citation"])],
        required=["openalex"],
    )
    openalex_hash = next(
        provider["native_query_hashes"][0]
        for provider in plan["providers"]
        if provider["provider"] == "openalex"
    )

    class RootRecordingCitationFixture:
        def __init__(self):
            self.seed_dois: list[str | None] = []

        def references(self, seed, cursor):
            self.seed_dois.append(seed.doi)
            return CitationBatch("fixture", "refs", (), None, EnvelopeStatus.SUCCESS)

        def citations(self, seed, cursor):
            self.seed_dois.append(seed.doi)
            return CitationBatch("fixture", "cites", (), None, EnvelopeStatus.SUCCESS)

    citation_client = RootRecordingCitationFixture()

    class RecordingScreener:
        def __init__(self):
            self.paper_ids: list[str] = []

        def screen(self, paper_ids):
            self.paper_ids.extend(paper_ids)
            return {paper_id: FilterStatus.IRRELEVANT for paper_id in paper_ids}

        def reranker_score(self, paper_id):
            return 0.1

    screener = RecordingScreener()
    with Database(tmp_path / "papers.sqlite3") as database:
        database.migrate()
        SearchPipeline(
            database, plan, runtime_providers=plan["providers"],
            clients={
                "user_library": SearchFixture((
                    _batch("user_library", "library", (
                        SourceEntry("user_library", "doi:10.1000/library", "Library root", ("Ada",), doi="10.1000/library", year=2024),
                    )),
                )),
                "openalex": SearchFixture((_batch("openalex", openalex_hash, ()),)),
            },
            trusts={"user_library": _trust("user_library"), "openalex": _trust("openalex")},
            citation_clients={"openalex": citation_client},
            screener=screener,
        ).run(run_id="run", crawl_run_id="crawl", observed_at=NOW)

    assert citation_client.seed_dois == ["10.1000/library", "10.1000/library"]
    assert len(screener.paper_ids) == 1


def test_initial_fanout_budget_marks_crawl_incomplete_with_audit(tmp_path) -> None:
    plan = _plan(
        [_provider("openalex"), _provider("crossref")], required=["openalex"], max_requests=1,
    )

    class Client:
        def search(self, query_spec, cursor):
            return SourceBatch("source", query_spec.native_query_hash, (), None, EnvelopeStatus.SUCCESS)

    with Database(tmp_path / "papers.sqlite3") as database:
        database.migrate()
        result = SearchPipeline(
            database, plan, runtime_providers=plan["providers"],
            clients={"openalex": Client(), "crossref": Client()},
            trusts={"openalex": _trust("openalex"), "crossref": _trust("crossref")},
        ).run(run_id="run", crawl_run_id="crawl", observed_at=NOW)
        crawl = database.connection.execute(
            "SELECT status, stats_json FROM crawl_runs WHERE crawl_run_id = 'crawl'"
        ).fetchone()

    assert result.status == "incomplete"
    assert crawl["status"] == "incomplete"
    assert json.loads(crawl["stats_json"])["budget"] == {
        "candidates_returned": 0,
        "reason": "budget_exhausted",
        "requests_made": 1,
    }


def test_initial_and_citation_work_share_the_campaign_hard_budgets(tmp_path) -> None:
    plan = _plan(
        [_provider("openalex", ["search", "citation"])],
        required=["openalex"], max_requests=2, max_candidates=2,
    )
    query_hash = plan["providers"][0]["native_query_hashes"][0]

    class Client:
        def __init__(self):
            self.calls: list[str] = []

        def search(self, query_spec, cursor):
            self.calls.append("search")
            return SourceBatch(
                "openalex", query_spec.native_query_hash,
                (_scoped(SourceEntry("openalex", "initial", "Initial", ("Ada",), doi="10.1000/initial", year=2024)),),
                None, EnvelopeStatus.SUCCESS,
            )

        def citations(self, seed, cursor):
            self.calls.append("citations")
            return CitationBatch(
                "fixture", "cites", (
                    CitationEdge(
                        "native-citing", seed.paper_id, CitationEdgeType.CITATIONS, "openalex", NOW,
                        candidate=_scoped(SourceEntry("openalex", "citation", "Citation", ("Ada",), doi="10.1000/citation", year=2024)),
                    ),
                ), None, EnvelopeStatus.SUCCESS,
            )

        def references(self, seed, cursor):
            self.calls.append("references")
            return CitationBatch("fixture", "refs", (), None, EnvelopeStatus.SUCCESS)

    client = Client()
    with Database(tmp_path / "papers.sqlite3") as database:
        database.migrate()
        pipeline = SearchPipeline(
            database, plan, runtime_providers=plan["providers"], clients={"openalex": client},
            trusts={"openalex": _trust("openalex")}, citation_clients={"openalex": client},
        )
        root = pipeline.repository.ingest(_scoped(SourceEntry("openalex", "root", "Root", ("Ada",), doi="10.1000/root", year=2024)))
        pipeline.run(run_id="run", crawl_run_id="crawl", observed_at=NOW, seed_paper_ids=[root.paper_id])
        citation_requests = database.connection.execute(
            "SELECT COUNT(*) FROM citation_requests WHERE status = 'complete'"
        ).fetchone()[0]
        discovered = database.connection.execute(
            "SELECT COUNT(*) FROM papers WHERE doi IN ('10.1000/initial', '10.1000/citation')"
        ).fetchone()[0]
        crawl_stats = json.loads(
            database.connection.execute(
                "SELECT stats_json FROM crawl_runs WHERE crawl_run_id = 'crawl'"
            ).fetchone()["stats_json"]
        )

    assert client.calls == ["search", "citations"]
    assert citation_requests == 1
    assert discovered == 2
    assert crawl_stats["budget"] == {
        "candidates_returned": 2,
        "reason": "budget_exhausted",
        "requests_made": 2,
    }


def test_metadata_enrichment_and_verifier_follow_discovery_deterministically(tmp_path) -> None:
    plan = _plan(
        [_provider("openalex"), _provider("crossref", ["metadata_enricher", "metadata_verifier"])],
        required=["crossref"],
    )
    query_hash = next(item["native_query_hashes"][0] for item in plan["providers"] if item["provider"] == "openalex")

    class MetadataClient:
        def __init__(self):
            self.calls: list[str] = []

        def enrich(self, raw):
            self.calls.append(f"enrich:{raw.doi}")
            return EnrichmentResult(
                SourceEntry("crossref", "doi:10.1000/enriched", raw.title, raw.authors, doi=raw.doi, year=raw.year),
                "crossref",
                "crossref:enrich",
            )

        def verify(self, candidate: IdentityCandidate, evidence):
            self.calls.append(f"verify:{candidate.doi}")
            return VerificationResult(candidate, VerificationStatus.VERIFIED, "crossref", ("crossref",))

    metadata_client = MetadataClient()
    with Database(tmp_path / "papers.sqlite3") as database:
        database.migrate()
        result = SearchPipeline(
            database,
            plan,
            runtime_providers=plan["providers"],
            clients={
                "openalex": SearchFixture((_batch("openalex", query_hash, (
                    SourceEntry("openalex", "work", "Paper", ("Ada",), doi="10.1000/enriched", year=2024),
                )),)),
                "crossref": metadata_client,
            },
            trusts={"openalex": _trust("openalex"), "crossref": _trust("crossref")},
        ).run(run_id="run", crawl_run_id="crawl", observed_at=NOW)
        paper = database.connection.execute("SELECT verification_status FROM papers").fetchone()[0]
        events = database.connection.execute(
            "SELECT provider, status FROM metadata_verification_events"
        ).fetchall()

    assert result.status == "complete"
    assert metadata_client.calls == ["enrich:10.1000/enriched", "verify:10.1000/enriched"]
    assert paper == VerificationStatus.VERIFIED
    assert [tuple(event) for event in events] == [("crossref", VerificationStatus.VERIFIED)]


def test_required_metadata_provider_failure_marks_the_campaign_incomplete(tmp_path) -> None:
    plan = _plan(
        [_provider("openalex"), _provider("crossref", ["metadata_enricher"])],
        required=["crossref"],
    )
    query_hash = next(item["native_query_hashes"][0] for item in plan["providers"] if item["provider"] == "openalex")

    class FailingMetadataClient:
        def enrich(self, raw):
            raise RuntimeError("metadata service unavailable")

    with Database(tmp_path / "papers.sqlite3") as database:
        database.migrate()
        result = SearchPipeline(
            database,
            plan,
            runtime_providers=plan["providers"],
            clients={
                "openalex": SearchFixture((_batch("openalex", query_hash, (
                    SourceEntry("openalex", "work", "Paper", ("Ada",), doi="10.1000/failure", year=2024),
                )),)),
                "crossref": FailingMetadataClient(),
            },
            trusts={"openalex": _trust("openalex"), "crossref": _trust("crossref")},
        ).run(run_id="run", crawl_run_id="crawl", observed_at=NOW)
        source = database.connection.execute(
            "SELECT status, error_json FROM source_runs WHERE provider = 'crossref'"
        ).fetchone()
        crawl_status = database.connection.execute(
            "SELECT status FROM crawl_runs WHERE crawl_run_id = 'crawl'"
        ).fetchone()[0]

    assert result.status == "incomplete"
    assert source["status"] == "failed"
    assert "metadata service unavailable" in source["error_json"]
    assert crawl_status == "incomplete"
