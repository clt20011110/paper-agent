"""Coordinator-side audit records for read-only Phase 2 provider fan-out."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from uuid import NAMESPACE_URL, uuid5

from .canonical import content_hash
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


@dataclass(frozen=True, slots=True)
class IncrementalScope:
    provider: str
    descriptor_key: str
    cursor: str | None
    complete: bool
    advance_watermark: bool = True


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
        recorded_params = dict(provider_params)
        if batch.request_audit:
            recorded_params["request_audit"] = [dict(record) for record in batch.request_audit]
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
                ON CONFLICT(query_id) DO UPDATE SET
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
                    _json(recorded_params),
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

    def finalize_incremental_crawl(
        self,
        crawl_run_id: str,
        *,
        paper_sources: Mapping[str, Sequence[tuple[str, str]]],
        scopes: Sequence[IncrementalScope],
        recorded_at: str,
    ) -> tuple[IncrementalChange, ...]:
        """Snapshot this crawl, compare its frozen scope with the preceding crawl, and advance watermarks."""
        snapshots = self._paper_snapshots(tuple(sorted(paper_sources)))
        current = self.database.connection.execute(
            "SELECT search_plan_id, window_json FROM crawl_runs WHERE crawl_run_id = ?", (crawl_run_id,)
        ).fetchone()
        if current is None:
            raise ValueError(f"unknown crawl run: {crawl_run_id}")
        previous = self.database.connection.execute(
            """SELECT c.crawl_run_id FROM crawl_runs c
               WHERE c.search_plan_id IS ? AND c.window_json = ? AND c.crawl_run_id != ?
                 AND EXISTS (SELECT 1 FROM incremental_diffs d WHERE d.crawl_run_id = c.crawl_run_id)
               ORDER BY c.rowid DESC LIMIT 1""",
            (current["search_plan_id"], current["window_json"], crawl_run_id),
        ).fetchone()
        prior = self._stored_snapshots(previous["crawl_run_id"]) if previous else {}
        prior_sources = self._stored_snapshot_sources(previous["crawl_run_id"]) if previous else {}
        changes = self._incremental_changes(
            prior,
            prior_sources,
            snapshots,
            scopes,
            previous["crawl_run_id"] if previous else None,
        )
        with self.database.transaction() as connection:
            connection.execute("DELETE FROM crawl_paper_snapshot_sources WHERE crawl_run_id = ?", (crawl_run_id,))
            connection.execute("DELETE FROM crawl_paper_snapshots WHERE crawl_run_id = ?", (crawl_run_id,))
            connection.execute("DELETE FROM crawl_scope_statuses WHERE crawl_run_id = ?", (crawl_run_id,))
            for paper_id, snapshot in snapshots.items():
                connection.execute(
                    """INSERT INTO crawl_paper_snapshots(crawl_run_id, paper_id, metadata_hash, status_version_json)
                       VALUES (?, ?, ?, ?)""",
                    (crawl_run_id, paper_id, snapshot["metadata_hash"], _json(snapshot["status_version"])),
                )
                for provider, descriptor_key in sorted(set(paper_sources[paper_id])):
                    connection.execute(
                        """INSERT INTO crawl_paper_snapshot_sources(crawl_run_id, paper_id, provider, descriptor_key)
                           VALUES (?, ?, ?, ?)""",
                        (crawl_run_id, paper_id, provider, descriptor_key),
                    )
            for scope in sorted(scopes, key=lambda item: (item.provider, item.descriptor_key)):
                connection.execute(
                    """INSERT INTO crawl_scope_statuses(crawl_run_id, provider, descriptor_key, cursor, complete)
                       VALUES (?, ?, ?, ?, ?)""",
                    (crawl_run_id, scope.provider, scope.descriptor_key, scope.cursor, int(scope.complete)),
                )
                if scope.complete and scope.advance_watermark:
                    connection.execute(
                        """INSERT INTO provider_watermarks(provider, descriptor_key, watermark_json, updated_at)
                           VALUES (?, ?, ?, ?)
                           ON CONFLICT(provider, descriptor_key) DO UPDATE SET
                               watermark_json = excluded.watermark_json, updated_at = excluded.updated_at""",
                        (
                            scope.provider,
                            scope.descriptor_key,
                            _json({"cursor": scope.cursor, "crawl_run_id": crawl_run_id}),
                            recorded_at,
                        ),
                    )
        self.record_incremental_diff(crawl_run_id, changes, recorded_at=recorded_at)
        return changes

    def _paper_snapshots(self, paper_ids: Sequence[str]) -> dict[str, dict[str, object]]:
        snapshots: dict[str, dict[str, object]] = {}
        for paper_id in paper_ids:
            paper = self.database.connection.execute(
                """SELECT title, abstract, authors_json, keywords_json, publication_date, year, venue_id,
                          venue_name, venue_type, doi, arxiv_id, canonical_url, volume, issue, pages,
                          verification_status
                   FROM papers WHERE paper_id = ?""",
                (paper_id,),
            ).fetchone()
            if paper is None:
                continue
            sources = self.database.connection.execute(
                """SELECT provider, external_id, publication_version, raw_metadata_json
                   FROM paper_sources WHERE paper_id = ? ORDER BY provider, external_id""",
                (paper_id,),
            ).fetchall()
            evidence = [_source_status_evidence(source) for source in sources]
            snapshots[paper_id] = {
                "metadata_hash": content_hash(dict(paper)),
                "status_version": {
                    "verification_status": paper["verification_status"],
                    "sources": evidence,
                    "retracted": any(item["retracted"] for item in evidence),
                    "preprint": any(item["publication_version"] == "preprint" for item in evidence),
                    "published": any(item["publication_version"] == "published" for item in evidence),
                },
            }
        return snapshots

    def _stored_snapshots(self, crawl_run_id: str) -> dict[str, dict[str, object]]:
        rows = self.database.connection.execute(
            """SELECT paper_id, metadata_hash, status_version_json FROM crawl_paper_snapshots
               WHERE crawl_run_id = ?""",
            (crawl_run_id,),
        ).fetchall()
        return {
            row["paper_id"]: {
                "metadata_hash": row["metadata_hash"],
                "status_version": json.loads(row["status_version_json"]),
            }
            for row in rows
        }

    def _stored_snapshot_sources(self, crawl_run_id: str) -> dict[str, set[tuple[str, str]]]:
        rows = self.database.connection.execute(
            """SELECT paper_id, provider, descriptor_key FROM crawl_paper_snapshot_sources
               WHERE crawl_run_id = ?""",
            (crawl_run_id,),
        ).fetchall()
        sources: dict[str, set[tuple[str, str]]] = {}
        for row in rows:
            sources.setdefault(row["paper_id"], set()).add((row["provider"], row["descriptor_key"]))
        return sources

    def _incremental_changes(
        self,
        prior: Mapping[str, Mapping[str, object]],
        prior_sources: Mapping[str, set[tuple[str, str]]],
        current: Mapping[str, Mapping[str, object]],
        scopes: Sequence[IncrementalScope],
        previous_crawl_run_id: str | None,
    ) -> tuple[IncrementalChange, ...]:
        if not prior:
            return tuple(IncrementalChange(paper_id, IncrementalChangeKind.NEW) for paper_id in sorted(current))
        prior_ids = set(prior)
        current_ids = set(current)
        replacements = self._preprint_replacements(previous_crawl_run_id, prior, current)
        changes: list[IncrementalChange] = []
        for paper_id in sorted(current_ids - prior_ids):
            kind = IncrementalChangeKind.PREPRINT_REPLACED if paper_id in replacements.values() else IncrementalChangeKind.NEW
            changes.append(IncrementalChange(paper_id, kind))
        for paper_id in sorted(prior_ids & current_ids):
            before = prior[paper_id]["status_version"]
            after = current[paper_id]["status_version"]
            if not before["retracted"] and after["retracted"]:
                changes.append(IncrementalChange(paper_id, IncrementalChangeKind.RETRACTED))
            elif before["preprint"] and after["published"]:
                changes.append(IncrementalChange(paper_id, IncrementalChangeKind.PREPRINT_REPLACED))
            elif prior[paper_id]["metadata_hash"] != current[paper_id]["metadata_hash"]:
                changes.append(IncrementalChange(paper_id, IncrementalChangeKind.METADATA_CHANGED))
        complete_scopes = {(scope.provider, scope.descriptor_key) for scope in scopes if scope.complete}
        replaced_preprints = set(replacements)
        for paper_id in sorted(prior_ids - current_ids - replaced_preprints):
            source_scopes = prior_sources.get(paper_id, set())
            if source_scopes and source_scopes.issubset(complete_scopes):
                changes.append(IncrementalChange(paper_id, IncrementalChangeKind.REMOVED))
        return tuple(changes)

    def _preprint_replacements(
        self,
        previous_crawl_run_id: str | None,
        prior: Mapping[str, Mapping[str, object]],
        current: Mapping[str, Mapping[str, object]],
    ) -> dict[str, str]:
        if previous_crawl_run_id is None:
            return {}
        rows = self.database.connection.execute(
            """SELECT source_paper_id, target_paper_id, provider, raw_evidence_json
               FROM citation_edges WHERE edge_type = 'version_of'
                 AND source_paper_id IN (SELECT paper_id FROM crawl_paper_snapshots WHERE crawl_run_id = ?)""",
            (previous_crawl_run_id,),
        ).fetchall()
        replacements: dict[str, str] = {}
        for row in rows:
            source_id, target_id = row["source_paper_id"], row["target_paper_id"]
            if source_id not in prior or target_id not in current:
                continue
            evidence = json.loads(row["raw_evidence_json"])
            if row["provider"] == "metadata" and evidence.get("match") == "title-author-year":
                continue
            if prior[source_id]["status_version"]["preprint"] and current[target_id]["status_version"]["published"]:
                replacements[source_id] = target_id
        return replacements


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


def _source_status_evidence(source: Mapping[str, object]) -> dict[str, object]:
    metadata = json.loads(str(source["raw_metadata_json"]))
    retracted = False
    retraction_evidence: dict[str, object] | None = None
    for key in ("retracted", "is_retracted", "publication_status", "status"):
        value = metadata.get(key)
        if value is True or isinstance(value, str) and value.casefold() == "retracted":
            retracted = True
            retraction_evidence = {"key": key, "value": value}
            break
    return {
        "provider": source["provider"],
        "external_id": source["external_id"],
        "publication_version": source["publication_version"] or "unknown",
        "retracted": retracted,
        "retraction_evidence": retraction_evidence,
    }


def _non_negative(metrics: SourceMetrics) -> None:
    if any(value < 0 for value in asdict(metrics).values()):
        raise ValueError("source metrics must be non-negative")
