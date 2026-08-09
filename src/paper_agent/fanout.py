"""Read-only parallel execution of every resolved QueryPlan provider."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from .domain import EnvelopeStatus, QuerySpec, SourceBatch
from .query_compilers import NativeQuery, compile_queries
from .query_plan import runtime_requirements


@dataclass(frozen=True, slots=True)
class ProviderOutcome:
    provider: str
    status: str
    result: Any | None
    error: str | None


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
                outcomes.append(ProviderOutcome(name, "success", result, None))
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
    batches: list[Any] = []
    for query in queries:
        cursor = None
        seen_cursors: set[str] = set()
        while True:
            batch = client.search(_query_spec(provider, query), cursor)
            batches.append(batch)
            if not isinstance(batch, SourceBatch) or batch.status is EnvelopeStatus.FAILED:
                break
            cursor = batch.next_cursor
            if not cursor:
                break
            if cursor in seen_cursors:
                raise ValueError(f"provider {provider['provider']} repeated cursor {cursor}")
            seen_cursors.add(cursor)
    return tuple(batches)


def _query_spec(provider: Mapping[str, Any], query: NativeQuery) -> QuerySpec:
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
