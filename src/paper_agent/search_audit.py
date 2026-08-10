"""Read-only, drillable search campaign audit."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from typing import Any, Iterable, Mapping


_SOURCE_METRICS = (
    "raw_discovered",
    "unique_after_dedup",
    "overlap",
    "screened",
    "excluded",
    "included",
    "full_text_available",
    "error_count",
)
_ROUND_METRICS = (
    "raw_discovered",
    "unique_after_dedup",
    "overlap",
    "screened_unique",
    "new_included_unique",
    "needs_review",
    "error_count",
)


def search_audit(database_path: Path, crawl_run_id: str) -> dict[str, Any]:
    uri = f"file:{database_path.resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.row_factory = sqlite3.Row
        crawl = connection.execute(
            """SELECT c.*, p.content_hash AS plan_hash
               FROM crawl_runs c
               LEFT JOIN search_plans p ON p.search_plan_id = c.search_plan_id
               WHERE c.crawl_run_id = ?""",
            (crawl_run_id,),
        ).fetchone()
        if crawl is None:
            raise ValueError(f"unknown crawl_run_id: {crawl_run_id}")
        sources = _sources(connection, crawl_run_id)
        queries = _queries(connection, crawl_run_id)
        rounds = _rounds(connection, crawl_run_id)
        incremental = connection.execute(
            "SELECT * FROM incremental_diffs WHERE crawl_run_id = ?", (crawl_run_id,)
        ).fetchone()

    return {
        "schema_version": "1",
        "crawl_run_id": crawl_run_id,
        "run_id": crawl["run_id"],
        "search_plan_id": crawl["search_plan_id"],
        "plan_hash": crawl["plan_hash"],
        "status": crawl["status"],
        "started_at": crawl["started_at"],
        "completed_at": crawl["completed_at"],
        "window": _json(crawl["window_json"]),
        "cursor": _json(crawl["cursor_json"]),
        "stats": _json(crawl["stats_json"]),
        "error": _json(crawl["error_json"]),
        "totals": {
            "sources": _totals(sources, _SOURCE_METRICS),
            "citation_rounds": _totals(
                (round_["audit"] for round_ in rounds if round_["audit"] is not None),
                _ROUND_METRICS,
            ),
            "queries": len(queries),
            "rounds": len(rounds),
        },
        "incomplete_sources": [
            source["source_run_id"] for source in sources if source["status"] != "complete"
        ],
        "sources": sources,
        "queries": queries,
        "rounds": rounds,
        "incremental_diff": _row(incremental) if incremental else None,
    }


def _sources(connection: sqlite3.Connection, crawl_run_id: str) -> list[dict[str, Any]]:
    rows = connection.execute(
        """SELECT s.*, a.raw_discovered, a.unique_after_dedup, a.overlap, a.screened,
                  a.excluded, a.included, a.full_text_available, a.error_count, a.updated_at
           FROM source_runs s
           LEFT JOIN source_run_audits a ON a.source_run_id = s.source_run_id
           WHERE s.crawl_run_id = ?
           ORDER BY s.provider, s.role, s.source_run_id""",
        (crawl_run_id,),
    ).fetchall()
    return [
        {
            **_row(row),
            "cursor": _json(row["cursor_json"]),
            "error": _json(row["error_json"]),
            "metrics": {name: int(row[name] or 0) for name in _SOURCE_METRICS},
        }
        for row in rows
    ]


def _queries(connection: sqlite3.Connection, crawl_run_id: str) -> list[dict[str, Any]]:
    rows = connection.execute(
        """SELECT q.* FROM search_queries q
           JOIN source_runs s ON s.source_run_id = q.source_run_id
           WHERE s.crawl_run_id = ?
           ORDER BY q.provider, q.role, q.query_id""",
        (crawl_run_id,),
    ).fetchall()
    return [
        {
            **_row(row),
            "provider_params": _json(row["provider_params_json"]),
            "filters": _json(row["filters_json"]),
            "error": _json(row["error_json"]),
        }
        for row in rows
    ]


def _rounds(connection: sqlite3.Connection, crawl_run_id: str) -> list[dict[str, Any]]:
    rows = connection.execute(
        """SELECT r.*, a.raw_discovered, a.unique_after_dedup, a.overlap,
                  a.screened_unique, a.new_included_unique, a.needs_review,
                  a.error_count, a.edge_counts_json, a.source_stats_json,
                  a.screening_complete, a.source_failed, a.audited_at
           FROM search_rounds r
           LEFT JOIN search_round_audits a ON a.search_round_id = r.search_round_id
           WHERE r.crawl_run_id = ? ORDER BY r.round_index""",
        (crawl_run_id,),
    ).fetchall()
    output = []
    for row in rows:
        round_id = row["search_round_id"]
        seeds = [
            _row(seed)
            for seed in connection.execute(
                "SELECT * FROM search_round_seeds WHERE search_round_id = ? ORDER BY seed_rank, paper_id",
                (round_id,),
            ).fetchall()
        ]
        requests = [
            {**_row(request), "error": _json(request["error_json"])}
            for request in connection.execute(
                "SELECT * FROM citation_requests WHERE search_round_id = ? ORDER BY schedule_order",
                (round_id,),
            ).fetchall()
        ]
        audit = None
        if row["audited_at"] is not None:
            audit = {
                **{name: int(row[name]) for name in _ROUND_METRICS},
                "edge_counts": _json(row["edge_counts_json"]),
                "source_stats": _json(row["source_stats_json"]),
                "screening_complete": bool(row["screening_complete"]),
                "source_failed": bool(row["source_failed"]),
                "audited_at": row["audited_at"],
            }
        output.append(
            {
                "search_round_id": round_id,
                "round_index": row["round_index"],
                "state": row["state"],
                "seed_manifest_hash": row["seed_manifest_hash"],
                "request_schedule_hash": row["request_schedule_hash"],
                "stop_reason": row["stop_reason"],
                "limited_scope": bool(row["limited_scope"]),
                "stats": _json(row["stats_json"]),
                "created_at": row["created_at"],
                "completed_at": row["completed_at"],
                "seeds": seeds,
                "requests": requests,
                "audit": audit,
            }
        )
    return output


def _totals(values: Iterable[Mapping[str, Any]], names: tuple[str, ...]) -> dict[str, int]:
    records = tuple(values)
    return {name: sum(int(value.get("metrics", value).get(name, 0)) for value in records) for name in names}


def _row(row: sqlite3.Row) -> dict[str, Any]:
    return dict(row)


def _json(value: str | None) -> Any:
    return json.loads(value) if value else None
