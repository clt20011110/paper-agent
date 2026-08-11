"""Read-only, budgeted execution of every resolved QueryPlan provider."""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Any

from .canonical import content_hash
from .domain import EnvelopeStatus, QuerySpec, SourceBatch
from .providers.api import CrawlWindow, VenueDescriptor
from .query_compilers import NativeQuery, compile_queries
from .query_plan import runtime_requirements


class RequestBudgetExhausted(RuntimeError):
    """Raised before provider I/O when the frozen request cap is spent."""


@dataclass(frozen=True, slots=True)
class ProviderOutcome:
    provider: str
    status: str
    result: Any | None
    error: str | None
    request_attempt_id: str | None = None


@dataclass(frozen=True, slots=True)
class ProviderPage:
    role: str
    batch: SourceBatch
    query: NativeQuery | None
    page: int
    cursor: str | None
    scope_id: str | None = None
    request_made: bool = True
    raw_returned_count: int | None = None
    request_attempt_id: str | None = None


@dataclass(frozen=True, slots=True)
class ProviderRequest:
    provider: str
    role: str
    query_hash: str
    cursor: str | None
    scope_id: str | None


@dataclass(frozen=True, slots=True)
class ProviderResponse:
    status: EnvelopeStatus
    error: str | None
    accepted_count: int
    raw_returned_count: int
    response_hash: str | None


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
    request_started: Callable[[ProviderRequest], str] | None = None,
    request_finished: Callable[[str, ProviderResponse], None] | None = None,
) -> FanoutResult:
    """Run frozen request streams in deterministic page waves.

    Every wave is concurrent, but its next wave is chosen only after results
    are ordered by frozen provider/query keys.  Thus request and candidate
    caps do not depend on which worker returns first.
    """
    if (request_started is None) != (request_finished is None):
        raise ValueError("request_started and request_finished must be supplied together")
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
    immediate_attempts: dict[str, str] = {}
    immediate_requests = 0
    immediate_candidates = 0
    exhausted = False
    for provider in resolved:
        name = str(provider["provider"])
        client = clients.get(name)
        queries = _queries(plan, provider)
        try:
            provider_streams = _streams(client, provider, queries)
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
                    attempt_id = (
                        request_started(_immediate_request(provider, queries))
                        if request_started is not None
                        else None
                    )
                except RequestBudgetExhausted:
                    exhausted = True
                    continue
                if attempt_id is not None:
                    immediate_attempts[name] = attempt_id
                immediate_requests += 1
                try:
                    result = _invoke(client, provider, queries)
                except Exception as error:
                    if attempt_id is not None and request_finished is not None:
                        request_finished(
                            attempt_id,
                            ProviderResponse(
                                EnvelopeStatus.FAILED, str(error), 0, 0, None
                            ),
                        )
                    failures[name] = str(error)
                else:
                    raw_count = _entry_count(result)
                    response_hash = _response_hash(result)
                    result, count, truncated = _truncate_result(result, max_candidates - immediate_candidates)
                    status, error = _result_status(result, truncated)
                    if attempt_id is not None and request_finished is not None:
                        request_finished(
                            attempt_id,
                            ProviderResponse(
                                status,
                                error,
                                count,
                                raw_count,
                                response_hash,
                            ),
                        )
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
        request_started=request_started,
        request_finished=request_finished,
    )
    exhausted = exhausted or stream_exhausted
    outcomes = _outcomes(
        resolved,
        pages,
        immediate,
        failures,
        immediate_attempts,
        exhausted,
    )
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


def _immediate_request(
    provider: Mapping[str, Any], queries: tuple[NativeQuery, ...]
) -> ProviderRequest:
    hashes = tuple(query.query_hash for query in queries)
    query_hash = (
        hashes[0]
        if len(hashes) == 1
        else "|".join(hashes)
        if hashes
        else "no-query"
    )
    roles = tuple(str(role) for role in provider["roles"])
    return ProviderRequest(
        str(provider["provider"]),
        "search" if queries else roles[0],
        query_hash,
        None,
        None,
    )


def _stream_request(stream: PageStream, cursor: str | None) -> ProviderRequest:
    return ProviderRequest(
        stream.provider,
        stream.role,
        stream.query.query_hash if stream.query is not None else stream.role,
        cursor,
        stream.scope_id,
    )


def _entry_count(value: Any) -> int:
    return sum(
        len(item.entries)
        for item in _flatten(value)
        if isinstance(item, SourceBatch)
    )


def _response_hash(value: Any) -> str | None:
    batches = tuple(
        item for item in _flatten(value) if isinstance(item, SourceBatch)
    )
    return content_hash([batch.to_dict() for batch in batches]) if batches else None


def _result_status(
    value: Any, truncated: bool
) -> tuple[EnvelopeStatus, str | None]:
    batches = tuple(
        item for item in _flatten(value) if isinstance(item, SourceBatch)
    )
    if truncated:
        return EnvelopeStatus.PARTIAL, "budget_exhausted"
    failures = tuple(
        batch for batch in batches if batch.status is EnvelopeStatus.FAILED
    )
    partials = tuple(
        batch for batch in batches if batch.status is EnvelopeStatus.PARTIAL
    )
    errors = "; ".join(
        batch.error or "provider page failed" for batch in (*failures, *partials)
    ) or None
    if failures and len(failures) == len(batches):
        return EnvelopeStatus.FAILED, errors
    if failures or partials:
        return EnvelopeStatus.PARTIAL, errors
    return EnvelopeStatus.SUCCESS, None


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
    request_started: Callable[[ProviderRequest], str] | None,
    request_finished: Callable[[str, ProviderResponse], None] | None,
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
            futures = []
            for stream, cursor, page, seen in wave:
                try:
                    attempt_id = (
                        request_started(_stream_request(stream, cursor))
                        if request_started is not None
                        else None
                    )
                except RequestBudgetExhausted:
                    exhausted = True
                    continue
                futures.append(
                    (
                        stream,
                        cursor,
                        page,
                        seen,
                        attempt_id,
                        executor.submit(_fetch, stream, cursor),
                    )
                )
            results = [
                (stream, cursor, page, seen, attempt_id, future.result())
                for stream, cursor, page, seen, attempt_id, future in futures
            ]
        next_active: list[tuple[PageStream, str | None, int, frozenset[str]]] = []
        for stream, cursor, page, seen_cursors, attempt_id, batch in sorted(
            results, key=lambda item: item[0].key
        ):
            requests_made += 1
            raw_returned_count = len(batch.entries)
            response_hash = content_hash(batch.to_dict())
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
            if attempt_id is not None and request_finished is not None:
                request_finished(
                    attempt_id,
                    ProviderResponse(
                        batch.status,
                        batch.error,
                        len(batch.entries),
                        raw_returned_count,
                        response_hash,
                    ),
                )
            candidates += len(batch.entries)
            pages[stream.provider].append(
                ProviderPage(
                    stream.role,
                    batch,
                    stream.query,
                    page,
                    cursor,
                    stream.scope_id,
                    raw_returned_count=raw_returned_count,
                    request_attempt_id=attempt_id,
                )
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
    immediate_attempts: Mapping[str, str],
    budget_exhausted: bool,
) -> list[ProviderOutcome]:
    outcomes: list[ProviderOutcome] = []
    for provider in resolved:
        name = str(provider["provider"])
        if name in failures:
            outcomes.append(
                ProviderOutcome(
                    name,
                    "failed",
                    None,
                    failures[name],
                    immediate_attempts.get(name),
                )
            )
            continue
        provider_pages = tuple(pages.get(name, ()))
        if provider_pages:
            failed = [page for page in provider_pages if page.batch.status is EnvelopeStatus.FAILED]
            partial = [page for page in provider_pages if page.batch.status is EnvelopeStatus.PARTIAL]
            status = "failed" if len(failed) == len(provider_pages) else "partial" if failed or partial else "success"
            error = "; ".join(page.batch.error or "provider page failed" for page in (*failed, *partial)) or None
            outcomes.append(ProviderOutcome(name, status, provider_pages, error))
        elif name in immediate:
            status, error = _result_status(immediate[name], False)
            outcomes.append(
                ProviderOutcome(
                    name,
                    "success" if status is EnvelopeStatus.SUCCESS else "partial"
                    if status is EnvelopeStatus.PARTIAL
                    else "failed",
                    immediate[name],
                    error,
                    immediate_attempts.get(name),
                )
            )
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
    requested_filters = query.requested_filters
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
        date_from=str(requested_filters["date_from"]),
        date_to=str(requested_filters["date_to"]),
        venue_ids=tuple(str(value) for value in requested_filters.get("venues", ())),
        fields=tuple(str(value) for value in requested_filters.get("fields", ())),
        languages=tuple(str(value) for value in requested_filters.get("languages", ())),
        document_types=tuple(
            str(value) for value in requested_filters.get("document_types", ())
        ),
        page_size=page_size,
        native_parameters=dict(parameters),
        native_query_hash=query.query_hash,
    )
