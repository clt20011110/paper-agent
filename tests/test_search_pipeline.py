from __future__ import annotations

from dataclasses import dataclass

from paper_agent.citations import DeterministicFakeScreener, citation_edge, reference_edge
from paper_agent.domain import CitationBatch, CitationEdge, CitationEdgeType, EnvelopeStatus, FilterStatus, MembershipStatus, ProviderRole, SourceBatch, SourceEntry
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
    providers: list[dict[str, object]], *, required: list[str], citation_cap: int = 10,
    max_depth: int = 1, max_rounds: int = 2, max_requests: int = 2, max_candidates: int = 10,
    max_seconds: int = 10,
) -> dict[str, object]:
    plan = compile_query_plan(
        {
            "created_at": NOW,
            "research": {"objective": "test", "audience": "test", "primary_question": "test", "subquestions": []},
            "scope": {"date_from": "2024-01-01", "date_to": "2024-12-31", "venues": [], "fields": ["computer science"], "languages": ["en"], "document_types": ["article"], "user_seeds": []},
            "inclusion": {"criteria": [], "exclusion_criteria": []},
            "query_variants": [{"id": "q", "subquestion_id": "q", "alias_group": "q", "raw_query": "paper agents", "synonyms": []}],
            "filter": {"profile": "fake", "config_hash": "c" * 64, "thresholds_hash": "d" * 64, "seed_selector_version": "v1", "seed_selector_config_hash": "e" * 64, "round_state_machine_version": "v1"},
            "citation_snowball": {"enabled": True, "directions": ["references", "citations"], "max_depth": max_depth, "max_rounds": max_rounds, "max_per_seed_per_source": citation_cap},
            "budgets": {"max_requests": max_requests, "max_candidates": max_candidates, "max_seconds": max_seconds, "saturation": {"min_unique_included_yield": 0.01, "consecutive_low_yield_rounds": 2}},
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
                        candidate=SourceEntry(
                            "openalex", "https://openalex.org/W-native-only", "Discovered only by citation",
                            ("Ada",), doi="10.1000/discovered", year=2024,
                        ),
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
        root = pipeline.repository.ingest(SourceEntry("openalex", "root", "Root", ("Ada",), doi="10.1000/root", year=2024))
        pipeline.run(run_id="run", crawl_run_id="crawl", observed_at=NOW, seed_paper_ids=[root.paper_id])

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

    assert pipeline.screener.screened == [discovered]
    assert tuple(edge[:2]) == (root.paper_id, discovered)
    assert "W-native-only" not in edge[2]
    assert tuple(round_paper) == (1, "q", "irrelevant")


def test_citation_depth_propagates_to_a_second_round_and_global_candidate_cap_stops(tmp_path) -> None:
    plan = _plan(
        [_provider("openalex", ["search", "citation"])], required=["openalex"],
        max_depth=2, max_rounds=2, max_requests=4, max_candidates=2,
    )
    query_hash = plan["providers"][0]["native_query_hashes"][0]

    class TwoDepthFixture:
        def references(self, seed, cursor):
            number = "one" if seed.doi == "10.1000/root" else "two"
            return CitationBatch(
                "fixture", number, (
                    CitationEdge(
                        seed.paper_id, f"native-{number}", CitationEdgeType.REFERENCES, "openalex", NOW,
                        candidate=SourceEntry("openalex", f"native-{number}", number, ("Ada",), doi=f"10.1000/{number}", year=2024),
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
        root = pipeline.repository.ingest(SourceEntry("openalex", "root", "Root", ("Ada",), doi="10.1000/root", year=2024))

        class RelevantScreener:
            screened: list[str] = []

            def screen(self, paper_ids):
                self.screened.extend(paper_ids)
                return {paper_id: FilterStatus.RELEVANT for paper_id in paper_ids}

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
    assert tuple(round_state) == ("budget_exhausted", 1)
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
        root = pipeline.repository.ingest(SourceEntry("openalex", "root", "Root", ("Ada",), doi="10.1000/root", year=2024))
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
    clock = iter((0.0, 2.0, 2.0))
    monkeypatch.setattr("paper_agent.search_pipeline.time.monotonic", lambda: next(clock))

    with Database(tmp_path / "papers.sqlite3") as database:
        database.migrate()
        pipeline = SearchPipeline(
            database, plan, runtime_providers=plan["providers"],
            clients={"openalex": SearchFixture((_batch("openalex", query_hash, ()),))},
            trusts={"openalex": _trust("openalex")}, citation_clients={"openalex": object()},
        )
        root = pipeline.repository.ingest(SourceEntry("openalex", "root", "Root", ("Ada",), doi="10.1000/root", year=2024))
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
        ).run(run_id="run", crawl_run_id="crawl", observed_at=NOW)

    assert citation_client.seed_dois == ["10.1000/library", "10.1000/library"]
