"""Deterministic, provider-specific native query compilers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .canonical import content_hash


COMPILER_VERSION = "2"


class QueryCompilerError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class NativeQuery:
    provider: str
    variant_id: str
    parameters: Mapping[str, Any]
    requested_filters: Mapping[str, Any] = field(default_factory=dict)
    native_applied_filters: Mapping[str, Any] = field(default_factory=dict)
    post_filters: Mapping[str, Any] = field(default_factory=dict)
    compiler_version: str = COMPILER_VERSION

    @property
    def query_hash(self) -> str:
        return content_hash(
            {
                "provider": self.provider,
                "variant_id": self.variant_id,
                "compiler_version": self.compiler_version,
                "parameters": dict(self.parameters),
                "requested_filters": dict(self.requested_filters),
                "native_applied_filters": dict(self.native_applied_filters),
                "post_filters": dict(self.post_filters),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "variant_id": self.variant_id,
            "parameters": dict(self.parameters),
            "requested_filters": dict(self.requested_filters),
            "native_applied_filters": dict(self.native_applied_filters),
            "post_filters": dict(self.post_filters),
            "compiler_version": self.compiler_version,
            "query_hash": self.query_hash,
        }


def compile_queries(
    provider: str,
    variants: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
    scope: Mapping[str, Any],
    *,
    page_size: int = 100,
) -> tuple[NativeQuery, ...]:
    """Compile every frozen query variant for one provider.

    Each platform receives its own parameter shape.  This makes the actual
    request, rather than a common intermediary string, part of the replayable
    QueryPlan identity.
    """
    compiler = _COMPILERS.get(provider)
    if compiler is None:
        raise QueryCompilerError(f"no query compiler registered for {provider}")
    return tuple(compiler(variant, scope, page_size) for variant in variants)


def compiled_query_hashes(
    provider: str,
    variants: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
    scope: Mapping[str, Any],
    *,
    page_size: int = 100,
) -> tuple[str, ...]:
    return tuple(query.query_hash for query in compile_queries(provider, variants, scope, page_size=page_size))


def _terms(variant: Mapping[str, Any]) -> tuple[str, ...]:
    return (str(variant["raw_query"]), *(str(term) for term in variant.get("synonyms", ())))


def _joined_terms(variant: Mapping[str, Any], separator: str = " OR ") -> str:
    return separator.join(f'"{term}"' for term in _terms(variant))


def _date_range(scope: Mapping[str, Any]) -> tuple[str, str]:
    return str(scope["date_from"]), str(scope["date_to"])


def _variant_id(variant: Mapping[str, Any]) -> str:
    return str(variant["id"])


def _requested_filters(scope: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "date_from": str(scope["date_from"]),
        "date_to": str(scope["date_to"]),
        "venues": [str(value) for value in scope.get("venues", ())],
        "fields": [str(value) for value in scope.get("fields", ())],
        "languages": [str(value) for value in scope.get("languages", ())],
        "document_types": [str(value) for value in scope.get("document_types", ())],
    }


def _native_query(
    provider: str,
    variant: Mapping[str, Any],
    scope: Mapping[str, Any],
    parameters: Mapping[str, Any],
    *,
    native_applied_filters: Mapping[str, Any],
    exact_filter_names: tuple[str, ...],
) -> NativeQuery:
    requested = _requested_filters(scope)
    exact = frozenset(exact_filter_names)
    post_filters = {
        name: value
        for name, value in requested.items()
        if name not in exact and value not in (None, "", [], ())
    }
    return NativeQuery(
        provider,
        _variant_id(variant),
        parameters,
        requested,
        dict(native_applied_filters),
        post_filters,
    )


def _crossref(variant: Mapping[str, Any], scope: Mapping[str, Any], page_size: int) -> NativeQuery:
    start, end = _date_range(scope)
    return _native_query(
        "crossref",
        variant,
        scope,
        {
            "query.bibliographic": " ".join(_terms(variant)),
            "filter": f"from-pub-date:{start},until-pub-date:{end}",
            "rows": page_size,
            "sort": "published",
            "order": "asc",
        },
        native_applied_filters={"date_from": start, "date_to": end},
        exact_filter_names=("date_from", "date_to"),
    )


def _dblp(variant: Mapping[str, Any], scope: Mapping[str, Any], page_size: int) -> NativeQuery:
    return _native_query(
        "dblp",
        variant,
        scope,
        {"q": " ".join(_terms(variant)), "format": "json", "h": page_size},
        native_applied_filters={},
        exact_filter_names=(),
    )


def _semantic_scholar(variant: Mapping[str, Any], scope: Mapping[str, Any], page_size: int) -> NativeQuery:
    start, end = _date_range(scope)
    return _native_query(
        "semantic_scholar",
        variant,
        scope,
        {
            "query": " ".join(_terms(variant)),
            "year": f"{start[:4]}-{end[:4]}",
            "limit": page_size,
            "fields": "paperId,title,abstract,authors,year,venue,externalIds,publicationDate",
        },
        native_applied_filters={"year_from": start[:4], "year_to": end[:4]},
        exact_filter_names=(),
    )


def _openalex(variant: Mapping[str, Any], scope: Mapping[str, Any], page_size: int) -> NativeQuery:
    start, end = _date_range(scope)
    return _native_query(
        "openalex",
        variant,
        scope,
        {
            "search": " ".join(_terms(variant)),
            "filter": f"from_publication_date:{start},to_publication_date:{end}",
            "per-page": page_size,
            "sort": "publication_date:asc",
        },
        native_applied_filters={"date_from": start, "date_to": end},
        exact_filter_names=("date_from", "date_to"),
    )


def _pubmed(variant: Mapping[str, Any], scope: Mapping[str, Any], page_size: int) -> NativeQuery:
    start, end = _date_range(scope)
    return _native_query(
        "pubmed",
        variant,
        scope,
        {
            "db": "pubmed",
            "term": f"({_joined_terms(variant)}) AND {start}:{end}[pdat]",
            "retmax": page_size,
            "sort": "pub date",
        },
        native_applied_filters={"date_from": start, "date_to": end},
        exact_filter_names=("date_from", "date_to"),
    )


def _europe_pmc(variant: Mapping[str, Any], scope: Mapping[str, Any], page_size: int) -> NativeQuery:
    start, end = _date_range(scope)
    return _native_query(
        "europe_pmc",
        variant,
        scope,
        {
            "query": f"({_joined_terms(variant)}) AND FIRST_PDATE:[{start} TO {end}]",
            "format": "json",
            "pageSize": page_size,
            "sort": "FIRST_PDATE_ASC",
        },
        native_applied_filters={"date_from": start, "date_to": end},
        exact_filter_names=("date_from", "date_to"),
    )


def _arxiv(variant: Mapping[str, Any], scope: Mapping[str, Any], page_size: int) -> NativeQuery:
    start, end = _date_range(scope)
    start_compact = start.replace("-", "") + "0000"
    end_compact = end.replace("-", "") + "2359"
    return _native_query(
        "arxiv",
        variant,
        scope,
        {
            "search_query": f"all:({_joined_terms(variant)}) AND submittedDate:[{start_compact} TO {end_compact}]",
            "start": 0,
            "max_results": page_size,
            "sortBy": "submittedDate",
            "sortOrder": "ascending",
        },
        native_applied_filters={"date_from": start, "date_to": end},
        exact_filter_names=("date_from", "date_to"),
    )


def _openreview(variant: Mapping[str, Any], scope: Mapping[str, Any], page_size: int) -> NativeQuery:
    start, end = _date_range(scope)
    venues = [str(venue) for venue in scope.get("venues", ())]
    return _native_query(
        "openreview",
        variant,
        scope,
        {
            "term": " ".join(_terms(variant)),
            "venue_ids": venues,
            "date_from": start,
            "date_to": end,
            "limit": page_size,
        },
        native_applied_filters={"date_from": start, "date_to": end, "venues": venues},
        exact_filter_names=("date_from", "date_to", "venues"),
    )


_COMPILERS = {
    "crossref": _crossref,
    "dblp": _dblp,
    "semantic_scholar": _semantic_scholar,
    "openalex": _openalex,
    "pubmed": _pubmed,
    "europe_pmc": _europe_pmc,
    "arxiv": _arxiv,
    "openreview": _openreview,
}
