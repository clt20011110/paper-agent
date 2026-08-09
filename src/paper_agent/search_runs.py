"""Coordinator-side audit records for read-only Phase 2 provider fan-out."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from uuid import NAMESPACE_URL, uuid5

from .domain import EnvelopeStatus, SourceBatch
from .fanout import FanoutResult
from .query_plan import runtime_requirements
from .storage import Database


class IncrementalChangeKind(StrEnum):
    NEW = "new"
    REMOVED = "removed"
    RETRACTED = "retracted"
    METADATA_CHANGED = "metadata_changed"
    PREPRINT_REPLACED = "preprint_replaced"


@dataclass(frozen=True, slots=True)
class SourceMetrics:
    raw_discovered: int = 0
    unique_after_dedup: int = 0
    overlap: int = 0
    screened: int = 0
    excluded: int = 0
    included: int = 0
    full_text_available: int = 0
    error_count: int = 0


@dataclass(frozen=True, slots=True)
class IncrementalChange:
    paper_id: str
    kind: IncrementalChangeKind


class SearchRunCoordinator:
    """The sole writer for crawl, source, query, and incremental search state."""

    def __init__(self, database: Database) -> None:
        self.database = database

    def start_crawl(
        self,
        *,
        crawl_run_id: str,
        run_id: str,
        search_plan_id: str | None,
        window: Mapping[str, object],
        cursor: Mapping[str, object] | None = None,
    ) -> None:
        payload = (_json(window), _json(cursor or {}))
        existing = self.database.connection.execute(
            "SELECT run_id, search_plan_id, window_json, cursor_json FROM crawl_runs WHERE crawl_run_id = ?",
            (crawl_run_id,),
        ).fetchone()
        if existing:
            recorded = (existing["run_id"], existing["search_plan_id"], existing["window_json"], existing["cursor_json"])
            expected = (run_id, search_plan_id, *payload)
            if recorded != expected:
                raise ValueError("crawl run has different frozen inputs")
            return
        with self.database.transaction() as connection:
            connection.execute(
                """INSERT INTO crawl_runs(
                    crawl_run_id, run_id, search_plan_id, window_json, cursor_json, status
                ) VALUES (?, ?, ?, ?, ?, 'running')""",
                (crawl_run_id, run_id, search_plan_id, *payload),
            )

    def record_batch(
        self,
        *,
        crawl_run_id: str,
        provider: str,
        provider_version: str,
        role: str,
        query_text: str,
        provider_params: Mapping[str, object],
        query_compiler_version: str,
        batch: SourceBatch,
        requested_at: str,
        completed_at: str,
        page: str | None = None,
        cursor: str | None = None,
        alias_group: str | None = None,
        filters: Mapping[str, object] | None = None,
    ) -> str:
        """Persist one provider response without inspecting or mutating its entries."""
        source_run_id = batch.source_run_id
        source_status, query_status = _statuses(batch.status)
        error_json = _error_json(batch.error)
        source = self.database.connection.execute(
            "SELECT crawl_run_id, provider, provider_version, role FROM source_runs WHERE source_run_id = ?",
            (source_run_id,),
        ).fetchone()
        expected_source = (crawl_run_id, provider, provider_version, role)
        if source and tuple(source) != expected_source:
            raise ValueError("source run is already bound to a different provider request")

        query_id = _query_id(source_run_id, batch.query_hash, page, cursor)
        with self.database.transaction() as connection:
            if source is None:
                connection.execute(
                    """INSERT INTO source_runs(
                        source_run_id, crawl_run_id, provider, provider_version, role, cursor_json,
                        status, error_json, raw_response_hash, started_at, completed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        source_run_id,
                        crawl_run_id,
                        provider,
                        provider_version,
                        role,
                        _json({"cursor": batch.next_cursor}),
                        source_status,
                        error_json,
                        batch.raw_response_artifact_hash,
                        requested_at,
                        completed_at,
                    ),
                )
            else:
                connection.execute(
                    """UPDATE source_runs SET cursor_json = ?, status = ?, error_json = ?,
                        raw_response_hash = ?, completed_at = ? WHERE source_run_id = ?""",
                    (
                        _json({"cursor": batch.next_cursor}),
                        source_status,
                        error_json,
                        batch.raw_response_artifact_hash,
                        completed_at,
                        source_run_id,
                    ),
                )

            connection.execute(
                """INSERT INTO search_queries(
                    query_id, search_plan_id, source_run_id, provider, provider_version,
                    query_compiler_version, role, query_text, provider_params_json, alias_group,
                    filters_json, page, cursor, requested_at, completed_at, query_hash,
                    response_hash, returned_count, status, error_json
                ) VALUES (?, (SELECT search_plan_id FROM crawl_runs WHERE crawl_run_id = ?), ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_run_id, query_hash, page, cursor) DO UPDATE SET
                    completed_at = excluded.completed_at, response_hash = excluded.response_hash,
                    returned_count = excluded.returned_count, status = excluded.status,
                    error_json = excluded.error_json, provider_params_json = excluded.provider_params_json""",
                (
                    query_id,
                    crawl_run_id,
                    source_run_id,
                    provider,
                    provider_version,
                    query_compiler_version,
                    role,
                    query_text,
                    _json(provider_params),
                    alias_group,
                    _json(filters or {}),
                    page,
                    cursor,
                    requested_at,
                    completed_at,
                    batch.query_hash,
                    batch.raw_response_artifact_hash,
                    len(batch.entries),
                    query_status,
                    error_json,
                ),
            )
            connection.execute(
                """INSERT INTO source_run_audits(
                    source_run_id, raw_discovered, unique_after_dedup, overlap, screened, excluded,
                    included, full_text_available, error_count, updated_at
                ) VALUES (?, ?, 0, 0, 0, 0, 0, 0, ?, ?)
                ON CONFLICT(source_run_id) DO UPDATE SET
                    raw_discovered = source_run_audits.raw_discovered + excluded.raw_discovered,
                    error_count = source_run_audits.error_count + excluded.error_count,
                    updated_at = excluded.updated_at""",
                (
                    source_run_id,
                    len(batch.entries),
                    int(batch.status is EnvelopeStatus.FAILED),
                    completed_at,
                ),
            )
        return source_run_id

    def record_metrics(self, source_run_id: str, metrics: SourceMetrics, *, updated_at: str) -> None:
        _non_negative(metrics)
        with self.database.transaction() as connection:
            connection.execute(
                """INSERT INTO source_run_audits(
                    source_run_id, raw_discovered, unique_after_dedup, overlap, screened, excluded,
                    included, full_text_available, error_count, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_run_id) DO UPDATE SET
                    raw_discovered = excluded.raw_discovered,
                    unique_after_dedup = excluded.unique_after_dedup,
                    overlap = excluded.overlap, screened = excluded.screened,
                    excluded = excluded.excluded, included = excluded.included,
                    full_text_available = excluded.full_text_available,
                    error_count = excluded.error_count, updated_at = excluded.updated_at""",
                (source_run_id, *asdict(metrics).values(), updated_at),
            )

    def finish_crawl(
        self,
        crawl_run_id: str,
        *,
        plan: Mapping[str, object],
        finished_at: str,
        fanout: FanoutResult | None = None,
    ) -> str:
        """Set the crawl status from persisted source runs and frozen requirements."""
        requirements = runtime_requirements(plan)
        sources = self.database.connection.execute(
            "SELECT provider, role, status FROM source_runs WHERE crawl_run_id = ?", (crawl_run_id,)
        ).fetchall()
        successful = {row["provider"] for row in sources if row["status"] == "complete"}
        successful_roles = {
            row["role"] for row in sources if row["status"] == "complete"
        }
        if fanout is not None:
            successful_names = set(fanout.successful_providers)
            successful_roles.update(
                role
                for provider in plan.get("providers", ())
                if provider["provider"] in successful_names
                for role in provider["roles"]
            )
        required_providers = set(requirements["required_providers"])
        required_roles = set(requirements["required_roles"])
        required_failure = not required_providers.issubset(successful) or not required_roles.issubset(successful_roles)
        if fanout is not None:
            required_failure = required_failure or fanout.incomplete
        status = "incomplete" if required_failure or not successful else "complete"
        summary = self.source_summary(crawl_run_id)
        with self.database.transaction() as connection:
            connection.execute(
                "UPDATE crawl_runs SET status = ?, stats_json = ?, completed_at = ? WHERE crawl_run_id = ?",
                (status, _json(summary), finished_at, crawl_run_id),
            )
        return status

    def source_summary(self, crawl_run_id: str) -> dict[str, dict[str, int]]:
        rows = self.database.connection.execute(
            """SELECT s.provider, s.role, a.raw_discovered, a.unique_after_dedup, a.overlap,
                      a.screened, a.excluded, a.included, a.full_text_available, a.error_count
               FROM source_runs s LEFT JOIN source_run_audits a ON a.source_run_id = s.source_run_id
               WHERE s.crawl_run_id = ? ORDER BY s.provider, s.role""",
            (crawl_run_id,),
        ).fetchall()
        return {
            f"{row['provider']}:{row['role']}": {
                "raw_discovered": row["raw_discovered"] or 0,
                "unique_after_dedup": row["unique_after_dedup"] or 0,
                "overlap": row["overlap"] or 0,
                "screened": row["screened"] or 0,
                "excluded": row["excluded"] or 0,
                "included": row["included"] or 0,
                "full_text_available": row["full_text_available"] or 0,
                "error_count": row["error_count"] or 0,
            }
            for row in rows
        }

    def set_watermark(
        self, provider: str, descriptor_key: str, watermark: Mapping[str, object], *, updated_at: str
    ) -> None:
        with self.database.transaction() as connection:
            connection.execute(
                """INSERT INTO provider_watermarks(provider, descriptor_key, watermark_json, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(provider, descriptor_key) DO UPDATE SET
                        watermark_json = excluded.watermark_json, updated_at = excluded.updated_at""",
                (provider, descriptor_key, _json(watermark), updated_at),
            )

    def window_for(
        self,
        provider: str,
        descriptor_key: str,
        window: Mapping[str, object],
        *,
        replay_window: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        """Use a persisted watermark unless an explicit historical replay was requested."""
        if replay_window is not None:
            return dict(replay_window)
        row = self.database.connection.execute(
            "SELECT watermark_json FROM provider_watermarks WHERE provider = ? AND descriptor_key = ?",
            (provider, descriptor_key),
        ).fetchone()
        result = dict(window)
        if row:
            result["watermark"] = json.loads(row["watermark_json"])
        return result

    def record_incremental_diff(
        self, crawl_run_id: str, changes: Sequence[IncrementalChange], *, recorded_at: str
    ) -> None:
        counts = Counter(change.kind.value for change in changes)
        with self.database.transaction() as connection:
            connection.execute("DELETE FROM incremental_diff_papers WHERE crawl_run_id = ?", (crawl_run_id,))
            for change in changes:
                connection.execute(
                    "INSERT INTO incremental_diff_papers(crawl_run_id, paper_id, change_kind) VALUES (?, ?, ?)",
                    (crawl_run_id, change.paper_id, change.kind),
                )
            connection.execute(
                """INSERT INTO incremental_diffs(
                    crawl_run_id, new_count, removed_count, retracted_count, metadata_changed_count,
                    preprint_replaced_count, recorded_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(crawl_run_id) DO UPDATE SET
                    new_count = excluded.new_count, removed_count = excluded.removed_count,
                    retracted_count = excluded.retracted_count,
                    metadata_changed_count = excluded.metadata_changed_count,
                    preprint_replaced_count = excluded.preprint_replaced_count,
                    recorded_at = excluded.recorded_at""",
                (
                    crawl_run_id,
                    counts["new"],
                    counts["removed"],
                    counts["retracted"],
                    counts["metadata_changed"],
                    counts["preprint_replaced"],
                    recorded_at,
                ),
            )

    def incremental_diff(self, crawl_run_id: str) -> tuple[IncrementalChange, ...]:
        rows = self.database.connection.execute(
            """SELECT paper_id, change_kind FROM incremental_diff_papers
               WHERE crawl_run_id = ? ORDER BY change_kind, paper_id""",
            (crawl_run_id,),
        ).fetchall()
        return tuple(IncrementalChange(row["paper_id"], IncrementalChangeKind(row["change_kind"])) for row in rows)


def _statuses(status: EnvelopeStatus) -> tuple[str, str]:
    if status is EnvelopeStatus.FAILED:
        return "failed", "failed"
    if status is EnvelopeStatus.PARTIAL:
        return "incomplete", "complete"
    return "complete", "complete"


def _query_id(source_run_id: str, query_hash: str, page: str | None, cursor: str | None) -> str:
    return f"query-{uuid5(NAMESPACE_URL, f'{source_run_id}:{query_hash}:{page}:{cursor}').hex}"


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _error_json(error: str | None) -> str | None:
    return _json({"message": error}) if error else None


def _non_negative(metrics: SourceMetrics) -> None:
    if any(value < 0 for value in asdict(metrics).values()):
        raise ValueError("source metrics must be non-negative")
