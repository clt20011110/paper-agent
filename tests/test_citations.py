from __future__ import annotations

import pytest

from paper_agent.citations import (
    CitationRepository,
    DeterministicFakeScreener,
    RoundAudit,
    SearchRoundStore,
    SeedCandidate,
    citation_edge,
    decide_stop,
    process_citation_batches,
    reference_edge,
    schedule_requests,
    select_seeds,
    version_edge,
)
from paper_agent.domain import (
    CitationBatch,
    CitationEdgeType,
    EnvelopeStatus,
    FilterStatus,
    VerificationStatus,
)
from paper_agent.storage import Database


NOW = "2026-08-09T00:00:00Z"


def test_backward_forward_and_version_edges_have_canonical_direction() -> None:
    backward = reference_edge("seed", "cited", provider="graph", observed_at=NOW, raw_evidence={})
    forward = citation_edge("seed", "citing", provider="graph", observed_at=NOW, raw_evidence={})
    version = version_edge("preprint", "published", provider="crossref", observed_at=NOW, raw_evidence={})

    assert (backward.source_paper_id, backward.target_paper_id, backward.edge_type) == (
        "seed",
        "cited",
        CitationEdgeType.REFERENCES,
    )
    assert (forward.source_paper_id, forward.target_paper_id, forward.edge_type) == (
        "citing",
        "seed",
        CitationEdgeType.CITATIONS,
    )
    assert (version.source_paper_id, version.target_paper_id, version.edge_type) == (
        "preprint",
        "published",
        CitationEdgeType.VERSION_OF,
    )


def test_citation_repository_is_idempotent_and_depths_ignore_loops(tmp_path) -> None:
    with Database(tmp_path / "papers.sqlite3") as database:
        database.migrate()
        for paper_id in ("root", "one", "two"):
            database.connection.execute("INSERT INTO papers(paper_id, title) VALUES (?, ?)", (paper_id, paper_id))
        database.connection.commit()
        repository = CitationRepository(database)
        edges = (
            reference_edge("root", "one", provider="graph", observed_at=NOW, raw_evidence={"page": 1}),
            citation_edge("one", "two", provider="graph", observed_at=NOW, raw_evidence={"page": 2}),
            reference_edge("two", "root", provider="graph", observed_at=NOW, raw_evidence={"page": 3}),
        )
        for edge in (*edges, edges[0]):
            repository.save(edge)

        assert database.connection.execute("SELECT COUNT(*) FROM citation_edges").fetchone()[0] == 3
        assert repository.depths(("root",)) == {"root": 0, "one": 1, "two": 1}


def candidate(
    paper_id: str,
    subquestion: str | None,
    score: float,
    *,
    status: FilterStatus = FilterStatus.RELEVANT,
    verification: VerificationStatus = VerificationStatus.VERIFIED,
    depth: int = 0,
) -> SeedCandidate:
    return SeedCandidate(paper_id, subquestion, status, score, verification, depth, 0)


def test_seed_selection_and_request_budget_are_stable() -> None:
    seeds = select_seeds(
        (
            candidate("user", None, 0),
            candidate("paper-b", "sq", 0.9),
            candidate("paper-a", "sq", 0.9),
            candidate("review", "sq", 1.0, status=FilterStatus.NEEDS_REVIEW),
            candidate("conflict", "sq", 1.0, verification=VerificationStatus.CONFLICTED),
            candidate("deep", "sq", 1.0, depth=2),
        ),
        user_seed_ids=frozenset({"user"}),
        expanded_paper_ids=frozenset(),
        max_depth=2,
        per_subquestion=2,
        selector_version="relevant_topk_by_subquestion_v1",
        selector_config_hash="a" * 64,
    )
    assert [seed.paper_id for seed in seeds] == ["user", "paper-a", "paper-b"]

    requests = schedule_requests(
        seeds,
        providers=("semantic_scholar", "openalex"),
        directions=(CitationEdgeType.REFERENCES, CitationEdgeType.CITATIONS),
        max_requests=5,
        max_candidates_per_request=500,
    )
    assert len(requests) == 5
    assert [request.schedule_order for request in requests] == list(range(5))
    assert [(request.provider, request.direction.value) for request in requests[:3]] == [
        ("openalex", "citations"),
        ("openalex", "citations"),
        ("openalex", "citations"),
    ]


def test_every_new_citation_candidate_passes_through_fake_stage2() -> None:
    batches = (
        CitationBatch(
            "semantic_scholar",
            "q1",
            (
                reference_edge("seed", "new-a", provider="semantic_scholar", observed_at=NOW, raw_evidence={}),
                reference_edge("seed", "old", provider="semantic_scholar", observed_at=NOW, raw_evidence={}),
            ),
            None,
            EnvelopeStatus.SUCCESS,
        ),
        CitationBatch(
            "openalex",
            "q2",
            (citation_edge("seed", "new-b", provider="openalex", observed_at=NOW, raw_evidence={}),),
            None,
            EnvelopeStatus.SUCCESS,
        ),
    )
    screener = DeterministicFakeScreener(frozenset({"new-a"}), frozenset({"new-b"}))

    decisions, audit = process_citation_batches(
        batches,
        already_seen=frozenset({"old"}),
        already_relevant=frozenset(),
        screener=screener,
    )

    assert screener.screened == ["new-a", "new-b"]
    assert decisions == {"new-a": FilterStatus.RELEVANT, "new-b": FilterStatus.NEEDS_REVIEW}
    assert (audit.raw_discovered, audit.unique_after_dedup, audit.overlap, audit.screened_unique) == (3, 3, 1, 2)


def test_stop_gate_distinguishes_saturation_unresolved_and_budget() -> None:
    low = RoundAudit(2, 2, 0, 2, 0, 0, 0)
    first = decide_stop(
        low,
        previous_low_yield_rounds=0,
        min_unique_included_yield=0.05,
        required_low_yield_rounds=2,
        screening_complete=True,
        sources_exhausted=False,
        budget_exhausted=False,
    )
    assert not first.stop
    saturated = decide_stop(
        RoundAudit(2, 2, 0, 2, 0, 1, 0),
        previous_low_yield_rounds=first.consecutive_low_yield_rounds,
        min_unique_included_yield=0.05,
        required_low_yield_rounds=2,
        screening_complete=True,
        sources_exhausted=False,
        budget_exhausted=False,
    )
    assert (saturated.reason, saturated.limited_scope) == ("saturated_with_unresolved", True)
    budget = decide_stop(
        low,
        previous_low_yield_rounds=0,
        min_unique_included_yield=0.05,
        required_low_yield_rounds=2,
        screening_complete=True,
        sources_exhausted=False,
        budget_exhausted=True,
    )
    assert (budget.reason, budget.limited_scope) == ("budget_exhausted", True)


def test_failed_or_incomplete_screening_never_claims_sources_exhausted() -> None:
    unresolved = RoundAudit(2, 2, 0, 2, 0, 0, 0, screening_complete=False)
    incomplete = decide_stop(
        unresolved,
        previous_low_yield_rounds=0,
        min_unique_included_yield=0.05,
        required_low_yield_rounds=2,
        screening_complete=unresolved.screening_complete,
        sources_exhausted=True,
        budget_exhausted=False,
    )
    assert (incomplete.reason, incomplete.limited_scope) == ("saturated_with_unresolved", True)

    failed = decide_stop(
        RoundAudit(0, 0, 0, 0, 0, 0, 1, source_failed=True),
        previous_low_yield_rounds=0,
        min_unique_included_yield=0.05,
        required_low_yield_rounds=2,
        screening_complete=True,
        sources_exhausted=False,
        budget_exhausted=False,
        source_failed=True,
    )
    assert (failed.reason, failed.limited_scope) == ("saturated_with_unresolved", True)


def test_completion_columns_default_when_an_existing_database_is_upgraded(tmp_path) -> None:
    with Database(tmp_path / "papers.sqlite3") as database:
        for migration in (item for item in database.migrations() if item.version < 5):
            database.connection.executescript(migration.sql)
            database.connection.execute(
                "INSERT INTO schema_migrations(version, name, applied_by) VALUES (?, ?, 'test')",
                (migration.version, migration.name),
            )
        database.connection.commit()
        database.migrate()
        columns = {
            row["name"]: row["dflt_value"]
            for row in database.connection.execute("PRAGMA table_info(search_round_audits)")
        }
    assert columns["screening_complete"] == "1"
    assert columns["source_failed"] == "0"


def test_round_manifest_is_frozen_and_auditable(tmp_path) -> None:
    with Database(tmp_path / "papers.sqlite3") as database:
        database.migrate()
        database.connection.execute(
            "INSERT INTO pipeline_runs(run_id, stage, status, input_hash, config_hash, implementation_version) "
            "VALUES ('run-1', 'stage-1', 'running', 'input', 'config', 'test')"
        )
        database.connection.execute(
            "INSERT INTO crawl_runs(crawl_run_id, run_id, status) VALUES ('crawl-1', 'run-1', 'running')"
        )
        database.connection.execute("INSERT INTO papers(paper_id, title) VALUES ('seed', 'Seed')")
        database.connection.commit()
        seeds = select_seeds(
            (candidate("seed", "sq", 1.0),),
            user_seed_ids=frozenset({"seed"}),
            expanded_paper_ids=frozenset(),
            max_depth=2,
            per_subquestion=20,
            selector_version="v1",
            selector_config_hash="a" * 64,
        )
        requests = schedule_requests(
            seeds,
            providers=("openalex",),
            directions=(CitationEdgeType.REFERENCES, CitationEdgeType.CITATIONS),
            max_requests=10,
            max_candidates_per_request=100,
        )
        store = SearchRoundStore(database)
        round_id = store.freeze(crawl_run_id="crawl-1", round_index=1, seeds=seeds, requests=requests)
        assert store.freeze(crawl_run_id="crawl-1", round_index=1, seeds=seeds, requests=requests) == round_id
        with pytest.raises(ValueError, match="drifted"):
            store.freeze(crawl_run_id="crawl-1", round_index=1, seeds=seeds, requests=requests[:1])

        audit = RoundAudit(3, 2, 1, 2, 1, 0, 0, {"references": 3}, {"openalex": {"returned": 3}})
        decision = decide_stop(
            audit,
            previous_low_yield_rounds=0,
            min_unique_included_yield=0.05,
            required_low_yield_rounds=2,
            screening_complete=True,
            sources_exhausted=True,
            budget_exhausted=False,
        )
        store.audit(round_id, audit, decision, audited_at=NOW)
        row = database.connection.execute(
            "SELECT state, stop_reason, limited_scope FROM search_rounds WHERE search_round_id = ?",
            (round_id,),
        ).fetchone()
        assert tuple(row) == ("stopped", "sources_exhausted", 0)
