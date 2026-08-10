from __future__ import annotations

import json

from paper_agent.domain import EnvelopeStatus, SourceBatch, SourceEntry
from paper_agent.fanout import FanoutResult, ProviderOutcome
from paper_agent.search_runs import (
    IncrementalChange,
    IncrementalChangeKind,
    IncrementalScope,
    SearchRunCoordinator,
    SourceMetrics,
)
from paper_agent.storage import Database


NOW = "2026-08-09T00:00:00Z"


def _plan(*, required_providers: tuple[str, ...] = ("openalex",)) -> dict[str, object]:
    return {
        "execution": {
            "required_providers": required_providers,
            "required_roles": ("search",),
        }
    }


def _coordinator(tmp_path) -> tuple[Database, SearchRunCoordinator]:
    database = Database(tmp_path / "papers.sqlite3")
    database.migrate()
    database.connection.execute(
        """INSERT INTO pipeline_runs(run_id, stage, status, input_hash, config_hash, implementation_version)
           VALUES ('run-1', 'stage-1', 'running', 'input', 'config', 'test')"""
    )
    database.connection.execute(
        """INSERT INTO search_plans(search_plan_id, content_hash, schema_version, plan_json, status)
           VALUES ('plan-1', 'plan-hash', '1', '{}', 'approved')"""
    )
    database.connection.commit()
    coordinator = SearchRunCoordinator(database)
    coordinator.start_crawl(
        crawl_run_id="crawl-1",
        run_id="run-1",
        search_plan_id="plan-1",
        window={"date_from": "2024-01-01", "date_to": "2024-12-31"},
    )
    return database, coordinator


def _batch(
    provider: str,
    *,
    status: EnvelopeStatus = EnvelopeStatus.SUCCESS,
    entries: tuple[SourceEntry, ...] = (),
    error: str | None = None,
) -> SourceBatch:
    return SourceBatch(
        source_run_id=f"crawl-1:{provider}",
        query_hash=f"{provider}-query-hash",
        entries=entries,
        next_cursor="next-page",
        status=status,
        error=error,
        raw_response_artifact_hash=f"{provider}-response-hash",
    )


def _record(coordinator: SearchRunCoordinator, provider: str, batch: SourceBatch, *, cursor: str | None = None) -> str:
    return coordinator.record_batch(
        crawl_run_id="crawl-1",
        provider=provider,
        provider_version="2026.08",
        role="search",
        query_text="paper agents",
        provider_params={"search": "paper agents", "cursor": cursor},
        query_compiler_version="1",
        batch=batch,
        requested_at=NOW,
        completed_at=NOW,
        page="1",
        cursor=cursor,
        alias_group="topic-a",
        filters={"year": 2024},
    )


def test_optional_failure_continues_and_source_metrics_are_persisted(tmp_path) -> None:
    database, coordinator = _coordinator(tmp_path)
    try:
        source_run_id = _record(coordinator, "openalex", _batch("openalex"))
        _record(coordinator, "crossref", _batch("crossref", status=EnvelopeStatus.FAILED, error="timeout"))
        coordinator.record_metrics(
            source_run_id,
            SourceMetrics(5, 4, 1, 4, 1, 3, 2, 0),
            updated_at=NOW,
        )

        fanout = FanoutResult(
            (ProviderOutcome("openalex", "success", None, None), ProviderOutcome("crossref", "failed", None, "timeout")),
            incomplete=False,
        )
        assert coordinator.finish_crawl("crawl-1", plan=_plan(), fanout=fanout, finished_at=NOW) == "complete"
        assert coordinator.source_summary("crawl-1") == {
            "crossref:search": {
                "raw_discovered": 0,
                "unique_after_dedup": 0,
                "overlap": 0,
                "screened": 0,
                "excluded": 0,
                "included": 0,
                "full_text_available": 0,
                "error_count": 1,
            },
            "openalex:search": {
                "raw_discovered": 5,
                "unique_after_dedup": 4,
                "overlap": 1,
                "screened": 4,
                "excluded": 1,
                "included": 3,
                "full_text_available": 2,
                "error_count": 0,
            },
        }
    finally:
        database.close()


def test_required_provider_failure_marks_crawl_incomplete(tmp_path) -> None:
    database, coordinator = _coordinator(tmp_path)
    try:
        _record(coordinator, "openalex", _batch("openalex", status=EnvelopeStatus.FAILED, error="unavailable"))
        _record(coordinator, "crossref", _batch("crossref"))

        assert coordinator.finish_crawl("crawl-1", plan=_plan(), finished_at=NOW) == "incomplete"
        row = database.connection.execute("SELECT status FROM crawl_runs WHERE crawl_run_id = 'crawl-1'").fetchone()
        assert row["status"] == "incomplete"
    finally:
        database.close()


def test_query_audit_replays_exact_actual_request(tmp_path) -> None:
    database, coordinator = _coordinator(tmp_path)
    try:
        batch = _batch("openalex")
        _record(coordinator, "openalex", batch, cursor="cursor-1")
        _record(coordinator, "openalex", batch, cursor="cursor-1")

        row = database.connection.execute("SELECT * FROM search_queries").fetchone()
        assert database.connection.execute("SELECT COUNT(*) FROM search_queries").fetchone()[0] == 1
        assert json.loads(row["provider_params_json"]) == {"cursor": "cursor-1", "search": "paper agents"}
        assert (row["query_hash"], row["response_hash"], row["cursor"], row["returned_count"]) == (
            "openalex-query-hash",
            "openalex-response-hash",
            "cursor-1",
            0,
        )
        source = database.connection.execute("SELECT raw_response_hash FROM source_runs").fetchone()
        assert source["raw_response_hash"] == "openalex-response-hash"
    finally:
        database.close()


def test_paginated_batches_accumulate_raw_and_error_counts(tmp_path) -> None:
    database, coordinator = _coordinator(tmp_path)
    try:
        entries = (SourceEntry("openalex", "W1", "One"),)
        _record(coordinator, "openalex", _batch("openalex", entries=entries), cursor=None)
        _record(
            coordinator,
            "openalex",
            _batch("openalex", status=EnvelopeStatus.FAILED, error="page timeout"),
            cursor="next-page",
        )

        summary = coordinator.source_summary("crawl-1")["openalex:search"]
        assert summary["raw_discovered"] == 1
        assert summary["error_count"] == 1
        assert database.connection.execute("SELECT COUNT(*) FROM search_queries").fetchone()[0] == 2
    finally:
        database.close()


def test_watermark_is_scoped_and_explicit_replay_bypasses_it(tmp_path) -> None:
    database, coordinator = _coordinator(tmp_path)
    try:
        coordinator.set_watermark("openalex", "neurips-2024", {"cursor": "current"}, updated_at=NOW)
        incremental = coordinator.window_for("openalex", "neurips-2024", {"date_from": "2024-01-01"})
        replay = coordinator.window_for(
            "openalex",
            "neurips-2024",
            {"date_from": "2024-01-01"},
            replay_window={"date_from": "2020-01-01", "date_to": "2020-12-31"},
        )
        assert incremental == {"date_from": "2024-01-01", "watermark": {"cursor": "current"}}
        assert replay == {"date_from": "2020-01-01", "date_to": "2020-12-31"}
    finally:
        database.close()


def test_incremental_diff_records_every_required_change_and_paper(tmp_path) -> None:
    database, coordinator = _coordinator(tmp_path)
    try:
        for paper_id in ("new", "removed", "retracted", "changed", "published"):
            database.connection.execute("INSERT INTO papers(paper_id, title) VALUES (?, ?)", (paper_id, paper_id))
        database.connection.commit()
        coordinator.record_incremental_diff(
            "crawl-1",
            (
                IncrementalChange("new", IncrementalChangeKind.NEW),
                IncrementalChange("removed", IncrementalChangeKind.REMOVED),
                IncrementalChange("retracted", IncrementalChangeKind.RETRACTED),
                IncrementalChange("changed", IncrementalChangeKind.METADATA_CHANGED),
                IncrementalChange("published", IncrementalChangeKind.PREPRINT_REPLACED),
            ),
            recorded_at=NOW,
        )
        assert set(coordinator.incremental_diff("crawl-1")) == {
            IncrementalChange("new", IncrementalChangeKind.NEW),
            IncrementalChange("removed", IncrementalChangeKind.REMOVED),
            IncrementalChange("retracted", IncrementalChangeKind.RETRACTED),
            IncrementalChange("changed", IncrementalChangeKind.METADATA_CHANGED),
            IncrementalChange("published", IncrementalChangeKind.PREPRINT_REPLACED),
        }
        counts = database.connection.execute("SELECT * FROM incremental_diffs WHERE crawl_run_id = 'crawl-1'").fetchone()
        assert tuple(counts[key] for key in ("new_count", "removed_count", "retracted_count", "metadata_changed_count", "preprint_replaced_count")) == (1, 1, 1, 1, 1)
    finally:
        database.close()


def _start_next_crawl(database: Database, coordinator: SearchRunCoordinator, number: int) -> str:
    run_id = f"run-{number}"
    crawl_run_id = f"crawl-{number}"
    database.connection.execute(
        """INSERT INTO pipeline_runs(run_id, stage, status, input_hash, config_hash, implementation_version)
           VALUES (?, 'stage-1', 'running', 'input', 'config', 'test')""",
        (run_id,),
    )
    database.connection.commit()
    coordinator.start_crawl(
        crawl_run_id=crawl_run_id,
        run_id=run_id,
        search_plan_id="plan-1",
        window={"date_from": "2024-01-01", "date_to": "2024-12-31"},
    )
    return crawl_run_id


def _paper_source(database: Database, paper_id: str, *, version: str = "published", retracted: bool = False) -> None:
    database.connection.execute("INSERT OR IGNORE INTO papers(paper_id, title) VALUES (?, ?)", (paper_id, paper_id))
    database.connection.execute(
        """INSERT INTO paper_sources(source_id, paper_id, provider, external_id, publication_version, raw_metadata_json)
           VALUES (?, ?, 'fixture', ?, ?, ?)
           ON CONFLICT(provider, external_id) DO UPDATE SET
               publication_version = excluded.publication_version, raw_metadata_json = excluded.raw_metadata_json""",
        (f"source-{paper_id}", paper_id, paper_id, version, json.dumps({"retracted": retracted})),
    )
    database.connection.commit()


def test_incremental_snapshot_detects_metadata_and_retraction_changes(tmp_path) -> None:
    database, coordinator = _coordinator(tmp_path)
    try:
        _paper_source(database, "paper")
        scope = (IncrementalScope("fixture", "venue", None, True),)
        assert coordinator.finalize_incremental_crawl(
            "crawl-1", paper_sources={"paper": (("fixture", "venue"),)}, scopes=scope, recorded_at=NOW
        ) == (IncrementalChange("paper", IncrementalChangeKind.NEW),)

        crawl_2 = _start_next_crawl(database, coordinator, 2)
        database.connection.execute("UPDATE papers SET title = 'Updated title' WHERE paper_id = 'paper'")
        database.connection.commit()
        assert coordinator.finalize_incremental_crawl(
            crawl_2, paper_sources={"paper": (("fixture", "venue"),)}, scopes=scope, recorded_at=NOW
        ) == (IncrementalChange("paper", IncrementalChangeKind.METADATA_CHANGED),)

        crawl_3 = _start_next_crawl(database, coordinator, 3)
        _paper_source(database, "paper", retracted=True)
        assert coordinator.finalize_incremental_crawl(
            crawl_3, paper_sources={"paper": (("fixture", "venue"),)}, scopes=scope, recorded_at=NOW
        ) == (IncrementalChange("paper", IncrementalChangeKind.RETRACTED),)
    finally:
        database.close()


def test_incremental_snapshot_uses_explicit_version_edges_and_never_removes_after_failure(tmp_path) -> None:
    database, coordinator = _coordinator(tmp_path)
    try:
        _paper_source(database, "preprint", version="preprint")
        _paper_source(database, "published", version="published")
        scope = (IncrementalScope("fixture", "venue", None, True),)
        coordinator.finalize_incremental_crawl(
            "crawl-1", paper_sources={"preprint": (("fixture", "venue"),)}, scopes=scope, recorded_at=NOW
        )
        database.connection.execute(
            """INSERT INTO citation_edges(citation_edge_id, source_paper_id, target_paper_id, edge_type, provider, observed_at, raw_evidence_json)
               VALUES ('version', 'preprint', 'published', 'version_of', 'fixture', ?, '{"relation":"published_version"}')""",
            (NOW,),
        )
        database.connection.commit()
        crawl_2 = _start_next_crawl(database, coordinator, 2)
        assert coordinator.finalize_incremental_crawl(
            crawl_2, paper_sources={"published": (("fixture", "venue"),)}, scopes=scope, recorded_at=NOW
        ) == (IncrementalChange("published", IncrementalChangeKind.PREPRINT_REPLACED),)

        crawl_3 = _start_next_crawl(database, coordinator, 3)
        assert coordinator.finalize_incremental_crawl(
            crawl_3,
            paper_sources={},
            scopes=(IncrementalScope("fixture", "venue", None, False),),
            recorded_at=NOW,
        ) == ()
        assert coordinator.incremental_diff(crawl_3) == ()
    finally:
        database.close()


def test_incremental_removal_requires_every_previous_source_scope_to_complete(tmp_path) -> None:
    database, coordinator = _coordinator(tmp_path)
    try:
        _paper_source(database, "paper")
        coordinator.finalize_incremental_crawl(
            "crawl-1",
            paper_sources={"paper": (("fixture", "venue-a"), ("fixture", "venue-b"))},
            scopes=(
                IncrementalScope("fixture", "venue-a", None, True),
                IncrementalScope("fixture", "venue-b", None, True),
            ),
            recorded_at=NOW,
        )
        crawl_2 = _start_next_crawl(database, coordinator, 2)
        assert coordinator.finalize_incremental_crawl(
            crawl_2,
            paper_sources={},
            scopes=(
                IncrementalScope("fixture", "venue-a", None, True),
                IncrementalScope("other", "unrelated", None, True),
            ),
            recorded_at=NOW,
        ) == ()
    finally:
        database.close()


def test_zero_result_crawl_is_the_next_incremental_baseline(tmp_path) -> None:
    database, coordinator = _coordinator(tmp_path)
    try:
        _paper_source(database, "first")
        _paper_source(database, "next")
        scope = (IncrementalScope("fixture", "venue", None, True),)
        coordinator.finalize_incremental_crawl(
            "crawl-1", paper_sources={"first": (("fixture", "venue"),)}, scopes=scope, recorded_at=NOW
        )
        crawl_2 = _start_next_crawl(database, coordinator, 2)
        assert coordinator.finalize_incremental_crawl(
            crawl_2, paper_sources={}, scopes=scope, recorded_at=NOW
        ) == (IncrementalChange("first", IncrementalChangeKind.REMOVED),)
        crawl_3 = _start_next_crawl(database, coordinator, 3)
        assert coordinator.finalize_incremental_crawl(
            crawl_3, paper_sources={"next": (("fixture", "venue"),)}, scopes=scope, recorded_at=NOW
        ) == (IncrementalChange("next", IncrementalChangeKind.NEW),)
    finally:
        database.close()


def test_metadata_audit_completes_when_no_pdf_is_available(tmp_path) -> None:
    database, coordinator = _coordinator(tmp_path)
    try:
        entry = SourceEntry(provider="openalex", external_id="work-1", title="Metadata only")
        source_run_id = _record(coordinator, "openalex", _batch("openalex", entries=(entry,)))
        coordinator.record_metrics(source_run_id, SourceMetrics(1, 1, 0, 1, 0, 1, 0, 0), updated_at=NOW)

        assert coordinator.finish_crawl("crawl-1", plan=_plan(), finished_at=NOW) == "complete"
        assert coordinator.source_summary("crawl-1")["openalex:search"]["full_text_available"] == 0
    finally:
        database.close()
