from __future__ import annotations

from dataclasses import dataclass

from paper_agent.citations import DeterministicFakeScreener, citation_edge, reference_edge
from paper_agent.domain import EnvelopeStatus, MembershipStatus, ProviderRole, SourceBatch, SourceEntry
from paper_agent.query_plan import approve_query_plan, compile_query_plan
from paper_agent.providers.api import CrawlWindow, SeedInput, VenueDescriptor
from paper_agent.providers.builtin import FixtureTransport, create_builtin
from paper_agent.search_pipeline import SearchPipeline, VenueRun
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
    providers: list[dict[str, object]], *, required: list[str], citation_cap: int = 10
) -> dict[str, object]:
    plan = compile_query_plan(
        {
            "created_at": NOW,
            "research": {"objective": "test", "audience": "test", "primary_question": "test", "subquestions": []},
            "scope": {"date_from": "2024-01-01", "date_to": "2024-12-31", "venues": [], "fields": ["computer science"], "languages": ["en"], "document_types": ["article"], "user_seeds": []},
            "inclusion": {"criteria": [], "exclusion_criteria": []},
            "query_variants": [{"id": "q", "subquestion_id": "q", "alias_group": "q", "raw_query": "paper agents", "synonyms": []}],
            "filter": {"profile": "fake", "config_hash": "c" * 64, "thresholds_hash": "d" * 64, "seed_selector_version": "v1", "seed_selector_config_hash": "e" * 64, "round_state_machine_version": "v1"},
            "citation_snowball": {"enabled": True, "directions": ["references", "citations"], "max_depth": 1, "max_rounds": 2, "max_per_seed_per_source": citation_cap},
            "budgets": {"max_requests": 2, "max_candidates": 10, "max_seconds": 10, "saturation": {"min_unique_included_yield": 0.01, "consecutive_low_yield_rounds": 2}},
            "provider_policy": "all_resolved",
            "required_roles": ["search"],
            "required_providers": required,
        },
        providers=providers,
    )
    return approve_query_plan(plan, plan["plan_hash"], approved_by="owner", approved_at=NOW)


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
        metadata={"official_membership": official, "venue_id": "testconf", "publication_version": "published"},
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
    return SourceBatch(f"fixture:{provider}", query_hash, entries, None, EnvelopeStatus.SUCCESS, raw_response_artifact_hash=f"{provider}-raw")


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
            observed.append((result.paper_ids, tuple(audit)))
    assert observed[0] == observed[1]


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
    plan = _plan(providers, required=["pmlr"])
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
        ("openalex", "search", "complete"),
        ("pmlr", "venue_primary", "complete"),
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
        pipeline.citation_clients = {"openalex": CitationFixture(papers["10.1000/seed"], papers["10.1000/cited"], papers["10.1000/skipped"], papers["10.1000/citing"])}
        second = pipeline.run(run_id="run-cite", crawl_run_id="crawl-cite", observed_at=NOW, seed_paper_ids=[papers["10.1000/seed"]])
        assert first.citation_round_ids == ()
        assert len(second.citation_round_ids) == 1
        assert screener.screened == sorted([papers["10.1000/cited"], papers["10.1000/citing"]])
        assert database.connection.execute("SELECT COUNT(*) FROM citation_requests").fetchone()[0] == 2
        assert database.connection.execute("SELECT COUNT(*) FROM citation_edges").fetchone()[0] == 2
