"""Read-only, budgeted execution of every resolved QueryPlan provider."""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Any

from .domain import EnvelopeStatus, QuerySpec, SourceBatch
from .providers.api import CrawlWindow, VenueDescriptor
from .query_compilers import NativeQuery, compile_queries
from .query_plan import runtime_requirements


@dataclass(frozen=True, slots=True)
class ProviderOutcome:
    provider: str
    status: str
    result: Any | None
    error: str | None


@dataclass(frozen=True, slots=True)
class ProviderPage:
    role: str
    batch: SourceBatch
    query: NativeQuery | None
    page: int
    cursor: str | None
    scope_id: str | None = None


@dataclass(frozen=True, slots=True)
class PageStream:
    """One independently paginated provider request stream."""

    provider: str
    role: str
    query: NativeQuery | None
    scope_id: str | None
    fetch: Callable[[str | None], SourceBatch]

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (self.provider, self.role, self.scope_id or "", self.query.query_hash if self.query else "")


@dataclass(frozen=True, slots=True)
class FanoutResult:
    outcomes: tuple[ProviderOutcome, ...]
    incomplete: bool
    budget_exhausted: bool = False
    requests_made: int = 0
    candidates_returned: int = 0

    @property
    def successful_providers(self) -> tuple[str, ...]:
        return tuple(outcome.provider for outcome in self.outcomes if outcome.status == "success")


def fan_out(
    plan: Mapping[str, Any],
    clients: Mapping[str, Any],
    *,
    max_workers: int | None = None,
    deadline: float | None = None,
) -> FanoutResult:
    """Run frozen request streams in deterministic page waves.

    Every wave is concurrent, but its next wave is chosen only after results
    are ordered by frozen provider/query keys.  Thus request and candidate
    caps do not depend on which worker returns first.
    """
    resolved = tuple(
        sorted(
            (item for item in plan["providers"] if item["resolved"]),
            key=lambda item: str(item["provider"]),
        )
    )
    max_requests = int(plan["budgets"]["max_requests"])
    max_candidates = int(plan["budgets"]["max_candidates"])
    deadline = deadline if deadline is not None else time.monotonic() + float(plan["budgets"]["max_seconds"])
    streams: list[PageStream] = []
    immediate: dict[str, Any] = {}
    failures: dict[str, str] = {}
    immediate_requests = 0
    immediate_candidates = 0
    exhausted = False
    for provider in resolved:
        name = str(provider["provider"])
        client = clients.get(name)
        try:
            provider_streams = _streams(client, provider, _queries(plan, provider))
        except Exception as error:
            failures[name] = str(error)
        else:
            if provider_streams is None:
                if (
                    immediate_requests >= max_requests
                    or immediate_candidates >= max_candidates
                    or time.monotonic() >= deadline
                ):
                    exhausted = True
                    continue
                try:
                    result = _invoke(client, provider, _queries(plan, provider))
                except Exception as error:
                    failures[name] = str(error)
                else:
                    immediate_requests += 1
                    result, count, truncated = _truncate_result(result, max_candidates - immediate_candidates)
                    immediate[name] = result
                    immediate_candidates += count
                    exhausted = exhausted or truncated
            else:
                streams.extend(provider_streams)

    pages, stream_exhausted, requests_made, candidates = _run_streams(
        streams,
        max_requests=max(0, max_requests - immediate_requests),
        max_candidates=max(0, max_candidates - immediate_candidates),
        deadline=deadline,
        max_workers=max_workers,
    )
    exhausted = exhausted or stream_exhausted
    outcomes = _outcomes(resolved, pages, immediate, failures, exhausted)
    requirements = runtime_requirements(plan)
    successful = {outcome.provider for outcome in outcomes if outcome.status == "success"}
    successful_roles = {
        role for provider in resolved if str(provider["provider"]) in successful for role in provider["roles"]
    }
    required_failure = not set(requirements["required_providers"]).issubset(successful)
    missing_roles = not set(requirements["required_roles"]).issubset(successful_roles)
    return FanoutResult(
        tuple(outcomes),
        incomplete=exhausted or not successful or required_failure or missing_roles,
        budget_exhausted=exhausted,
        requests_made=immediate_requests + requests_made,
        candidates_returned=immediate_candidates + candidates,
    )


def _streams(
    client: Any, provider: Mapping[str, Any], queries: tuple[NativeQuery, ...]
) -> tuple[PageStream, ...] | None:
    if client is None:
        raise ValueError(f"no client registered for {provider['provider']}")
    if hasattr(client, "initial_streams"):
        return tuple(client.initial_streams(provider, queries))
    if callable(client):
        return None
    if not queries:
        raise ValueError(f"non-search provider {provider['provider']} requires an invocation adapter")
    return search_streams(client, provider, queries)


def _run_streams(
    streams: list[PageStream],
    *,
    max_requests: int,
    max_candidates: int,
    deadline: float,
    max_workers: int | None,
) -> tuple[dict[str, list[ProviderPage]], bool, int, int]:
    pages: dict[str, list[ProviderPage]] = defaultdict(list)
    active = [(stream, None, 1, frozenset()) for stream in sorted(streams, key=lambda item: item.key)]
    requests_made = 0
    candidates = 0
    exhausted = False
    while active:
        if requests_made >= max_requests or candidates >= max_candidates or time.monotonic() >= deadline:
            exhausted = True
            break
        slots = max_requests - requests_made
        wave, deferred = active[:slots], active[slots:]
        if deferred:
            exhausted = True
        with ThreadPoolExecutor(max_workers=max_workers or len(wave) or 1) as executor:
            futures = [
                (stream, cursor, page, seen, executor.submit(_fetch, stream, cursor))
                for stream, cursor, page, seen in wave
            ]
            results = [
                (stream, cursor, page, seen, future.result())
                for stream, cursor, page, seen, future in futures
            ]
        next_active: list[tuple[PageStream, str | None, int, frozenset[str]]] = []
        for stream, cursor, page, seen_cursors, batch in sorted(results, key=lambda item: item[0].key):
            requests_made += 1
            available = max(0, max_candidates - candidates)
            truncated = len(batch.entries) > available
            candidate_cutoff = truncated or (bool(batch.next_cursor) and len(batch.entries) >= available)
            if candidate_cutoff:
                batch = replace(
                    batch,
                    entries=batch.entries[:available],
                    status=EnvelopeStatus.PARTIAL,
                    error="budget_exhausted",
                )
                exhausted = True
            if batch.status is EnvelopeStatus.SUCCESS and batch.next_cursor in seen_cursors:
                batch = replace(
                    batch,
                    status=EnvelopeStatus.PARTIAL if batch.entries else EnvelopeStatus.FAILED,
                    error=f"provider {stream.provider} repeated cursor {batch.next_cursor}",
                )
            candidates += len(batch.entries)
            pages[stream.provider].append(
                ProviderPage(stream.role, batch, stream.query, page, cursor, stream.scope_id)
            )
            if batch.status is EnvelopeStatus.SUCCESS and batch.next_cursor:
                next_active.append((stream, batch.next_cursor, page + 1, seen_cursors | {batch.next_cursor}))
        active = sorted(next_active + deferred, key=lambda item: item[0].key)
        if time.monotonic() >= deadline:
            exhausted = True
            break
    return pages, exhausted, requests_made, candidates


def _fetch(stream: PageStream, cursor: str | None) -> SourceBatch:
    try:
        result = stream.fetch(cursor)
        if isinstance(result, SourceBatch):
            return result
        return SourceBatch(
            stream.provider,
            stream.query.query_hash if stream.query else stream.role,
            (),
            None,
            EnvelopeStatus.SUCCESS,
        )
    except Exception as error:
        return SourceBatch(
            stream.provider,
            stream.query.query_hash if stream.query else stream.role,
            (),
            None,
            EnvelopeStatus.FAILED,
            str(error),
        )


def _truncate_result(value: Any, available: int) -> tuple[Any, int, bool]:
    """Cap callback envelopes too; adapters that paginate use PageStream instead."""
    if isinstance(value, SourceBatch):
        entries = value.entries[:available]
        truncated = len(entries) != len(value.entries)
        return (
            (
                replace(
                    value,
                    entries=entries,
                    status=EnvelopeStatus.PARTIAL,
                    error="budget_exhausted",
                )
                if truncated
                else value
            ),
            len(entries),
            truncated,
        )
    if isinstance(value, tuple):
        items = []
        count = 0
        truncated = False
        for item in value:
            capped, added, cut = _truncate_result(item, max(0, available - count))
            items.append(capped)
            count += added
            truncated = truncated or cut
        return tuple(items), count, truncated
    if isinstance(value, list):
        capped, count, truncated = _truncate_result(tuple(value), available)
        return list(capped), count, truncated
    return value, 0, False


def _outcomes(
    resolved: tuple[Mapping[str, Any], ...],
    pages: Mapping[str, list[ProviderPage]],
    immediate: Mapping[str, Any],
    failures: Mapping[str, str],
    budget_exhausted: bool,
) -> list[ProviderOutcome]:
    outcomes: list[ProviderOutcome] = []
    for provider in resolved:
        name = str(provider["provider"])
        if name in failures:
            outcomes.append(ProviderOutcome(name, "failed", None, failures[name]))
            continue
        provider_pages = tuple(pages.get(name, ()))
        if provider_pages:
            failed = [page for page in provider_pages if page.batch.status is EnvelopeStatus.FAILED]
            partial = [page for page in provider_pages if page.batch.status is EnvelopeStatus.PARTIAL]
            status = "failed" if len(failed) == len(provider_pages) else "partial" if failed or partial else "success"
            error = "; ".join(page.batch.error or "provider page failed" for page in (*failed, *partial)) or None
            outcomes.append(ProviderOutcome(name, status, provider_pages, error))
        elif name in immediate:
            outcomes.append(ProviderOutcome(name, "success", immediate[name], None))
        elif budget_exhausted:
            outcomes.append(ProviderOutcome(name, "skipped_budget", (), "budget_exhausted"))
        else:
            outcomes.append(ProviderOutcome(name, "success", (), None))
    return outcomes


def _flatten(value: Any) -> tuple[Any, ...]:
    if isinstance(value, (tuple, list)):
        return tuple(item for group in value for item in _flatten(group))
    return (value,)


def _queries(plan: Mapping[str, Any], provider: Mapping[str, Any]) -> tuple[NativeQuery, ...]:
    if "search" not in provider["roles"]:
        return ()
    queries = compile_queries(
        str(provider["provider"]),
        plan["query_variants"],
        plan["scope"],
        page_size=int(plan.get("page_size", 100)),
    )
    if [query.query_hash for query in queries] != list(provider["native_query_hashes"]):
        raise ValueError(f"provider {provider['provider']} native query has drifted")
    return queries


def _invoke(client: Any, provider: Mapping[str, Any], queries: tuple[NativeQuery, ...]) -> Any:
    if client is None:
        raise ValueError(f"no client registered for {provider['provider']}")
    return client(provider, queries)


def search_streams(
    client: Any, provider: Mapping[str, Any], queries: tuple[NativeQuery, ...]
) -> tuple[PageStream, ...]:
    return tuple(
        PageStream(
            str(provider["provider"]),
            "search",
            query,
            None,
            lambda cursor, query=query: client.search(query_spec_for_native(provider, query), cursor),
        )
        for query in queries
    )


def search_pages(
    client: Any, provider: Mapping[str, Any], queries: tuple[NativeQuery, ...]
) -> tuple[ProviderPage, ...]:
    pages, _, _, _ = _run_streams(
        list(search_streams(client, provider, queries)),
        max_requests=10**9,
        max_candidates=10**9,
        deadline=float("inf"),
        max_workers=1,
    )
    return tuple(page for provider_pages in pages.values() for page in provider_pages)


def venue_stream(client: Any, descriptor: VenueDescriptor, window: CrawlWindow) -> PageStream:
    return PageStream(
        descriptor.provider,
        "venue_primary",
        None,
        descriptor.venue_id,
        lambda cursor: client.discover(descriptor, window, cursor),
    )


def venue_pages(client: Any, descriptor: VenueDescriptor, window: CrawlWindow) -> tuple[ProviderPage, ...]:
    pages, _, _, _ = _run_streams(
        [venue_stream(client, descriptor, window)],
        max_requests=10**9,
        max_candidates=10**9,
        deadline=float("inf"),
        max_workers=1,
    )
    return tuple(pages[descriptor.provider])


def query_spec_for_native(provider: Mapping[str, Any], query: NativeQuery) -> QuerySpec:
    parameters = query.parameters
    original_query = next(
        str(parameters[key])
        for key in ("query.bibliographic", "q", "query", "search", "term", "search_query")
        if key in parameters
    )
    page_size = int(
        parameters.get(
            "rows",
            parameters.get(
                "h",
                parameters.get(
                    "limit",
                    parameters.get(
                        "per-page",
                        parameters.get(
                            "retmax",
                            parameters.get("pageSize", parameters.get("max_results", 100)),
                        ),
                    ),
                ),
            ),
        )
    )
    return QuerySpec(
        schema_version=1,
        research_question_id=query.variant_id,
        original_query=original_query,
        alias_group=query.variant_id,
        page_size=page_size,
        native_parameters=dict(parameters),
        native_query_hash=query.query_hash,
    )
