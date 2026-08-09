"""Read-only parallel execution of every resolved QueryPlan provider."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from typing import Any

from .canonical import content_hash
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
class FanoutResult:
    outcomes: tuple[ProviderOutcome, ...]
    incomplete: bool

    @property
    def successful_providers(self) -> tuple[str, ...]:
        return tuple(outcome.provider for outcome in self.outcomes if outcome.status == "success")


def fan_out(
    plan: Mapping[str, Any],
    clients: Mapping[str, Any],
    *,
    max_workers: int | None = None,
) -> FanoutResult:
    """Invoke all frozen resolved providers, isolating each provider failure."""
    resolved = [provider for provider in plan["providers"] if provider["resolved"]]
    requirements = runtime_requirements(plan)
    outcomes: list[ProviderOutcome] = []
    with ThreadPoolExecutor(max_workers=max_workers or len(resolved) or 1) as executor:
        submitted = {
            executor.submit(_invoke, clients.get(str(provider["provider"])), provider, _queries(plan, provider)): provider
            for provider in resolved
        }
        for future in as_completed(submitted):
            provider = submitted[future]
            name = str(provider["provider"])
            try:
                result = future.result()
            except Exception as error:
                outcomes.append(ProviderOutcome(name, "failed", None, str(error)))
            else:
                pages = tuple(item for item in _flatten(result) if isinstance(item, ProviderPage))
                failed_pages = tuple(page for page in pages if page.batch.status is EnvelopeStatus.FAILED)
                partial_pages = tuple(page for page in pages if page.batch.status is EnvelopeStatus.PARTIAL)
                status = "failed" if pages and len(failed_pages) == len(pages) else "partial" if failed_pages else "success"
                if partial_pages and status == "success":
                    status = "partial"
                error = "; ".join(page.batch.error or "provider page failed" for page in failed_pages) or None
                outcomes.append(ProviderOutcome(name, status, result, error))
    outcomes.sort(key=lambda outcome: outcome.provider)
    successful = {outcome.provider for outcome in outcomes if outcome.status == "success"}
    successful_roles = {
        role
        for provider in resolved
        if str(provider["provider"]) in successful
        for role in provider["roles"]
    }
    required_failure = not set(requirements["required_providers"]).issubset(successful)
    missing_roles = not set(requirements["required_roles"]).issubset(successful_roles)
    return FanoutResult(tuple(outcomes), incomplete=not successful or required_failure or missing_roles)


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
    hashes = [query.query_hash for query in queries]
    if hashes != list(provider["native_query_hashes"]):
        raise ValueError(f"provider {provider['provider']} native query has drifted")
    return queries


def _invoke(client: Any, provider: Mapping[str, Any], queries: tuple[NativeQuery, ...]) -> Any:
    if client is None:
        raise ValueError(f"no client registered for {provider['provider']}")
    if callable(client):
        return client(provider, queries)
    if not queries:
        raise ValueError(f"non-search provider {provider['provider']} requires an invocation adapter")
    return search_pages(client, provider, queries)


def search_pages(
    client: Any,
    provider: Mapping[str, Any],
    queries: tuple[NativeQuery, ...],
) -> tuple[Any, ...]:
    pages: list[Any] = []
    for query in queries:
        cursor = None
        seen_cursors: set[str] = set()
        page = 1
        while True:
            try:
                batch = client.search(query_spec_for_native(provider, query), cursor)
            except Exception as error:
                batch = SourceBatch(
                    f"{provider['provider']}:search",
                    query.query_hash,
                    (),
                    None,
                    EnvelopeStatus.FAILED,
                    str(error),
                )
            if not isinstance(batch, SourceBatch):
                pages.append(batch)
                break
            pages.append(ProviderPage("search", batch, query, page, cursor))
            if batch.status is EnvelopeStatus.FAILED:
                break
            cursor = batch.next_cursor
            if not cursor:
                break
            if cursor in seen_cursors:
                raise ValueError(f"provider {provider['provider']} repeated cursor {cursor}")
            seen_cursors.add(cursor)
            page += 1
    return tuple(pages)


def venue_pages(
    client: Any,
    descriptor: VenueDescriptor,
    window: CrawlWindow,
) -> tuple[ProviderPage, ...]:
    pages: list[ProviderPage] = []
    cursor = None
    seen_cursors: set[str] = set()
    page = 1
    while True:
        try:
            batch = client.discover(descriptor, window, cursor)
        except Exception as error:
            batch = SourceBatch(
                f"{descriptor.provider}:{descriptor.venue_id}",
                content_hash(
                    {
                        "provider": descriptor.provider,
                        "venue_id": descriptor.venue_id,
                        "parameters": descriptor.parameters,
                        "window": asdict(window),
                    }
                ),
                (),
                None,
                EnvelopeStatus.FAILED,
                str(error),
            )
        pages.append(ProviderPage("venue_primary", batch, None, page, cursor, descriptor.venue_id))
        if batch.status is EnvelopeStatus.FAILED:
            break
        cursor = batch.next_cursor
        if not cursor:
            break
        if cursor in seen_cursors:
            raise ValueError(f"provider {descriptor.provider} repeated cursor {cursor}")
        seen_cursors.add(cursor)
        page += 1
    return tuple(pages)


def query_spec_for_native(provider: Mapping[str, Any], query: NativeQuery) -> QuerySpec:
    """Adapt a frozen native request to the shared provider API."""
    parameters = query.parameters
    original_query = next(
        str(parameters[key])
        for key in ("query.bibliographic", "q", "query", "search", "term", "search_query")
        if key in parameters
    )
    return QuerySpec(
        schema_version=1,
        research_question_id=query.variant_id,
        original_query=original_query,
        alias_group=query.variant_id,
        page_size=int(
            parameters.get("rows", parameters.get("h", parameters.get("limit", parameters.get("per-page", parameters.get("retmax", parameters.get("pageSize", parameters.get("max_results", 100)))))))
        ),
        native_parameters=dict(parameters),
        native_query_hash=query.query_hash,
    )
