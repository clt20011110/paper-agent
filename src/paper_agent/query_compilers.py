"""Deterministic, provider-specific native query compilers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .canonical import content_hash


COMPILER_VERSION = "1"


class QueryCompilerError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class NativeQuery:
    provider: str
    variant_id: str
    parameters: Mapping[str, Any]
    compiler_version: str = COMPILER_VERSION

    @property
    def query_hash(self) -> str:
        return content_hash(
            {
                "provider": self.provider,
                "variant_id": self.variant_id,
                "compiler_version": self.compiler_version,
                "parameters": dict(self.parameters),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "variant_id": self.variant_id,
            "parameters": dict(self.parameters),
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


def _crossref(variant: Mapping[str, Any], scope: Mapping[str, Any], page_size: int) -> NativeQuery:
    start, end = _date_range(scope)
    return NativeQuery(
        "crossref",
        _variant_id(variant),
        {
            "query.bibliographic": " ".join(_terms(variant)),
            "filter": f"from-pub-date:{start},until-pub-date:{end}",
            "rows": page_size,
            "sort": "published",
            "order": "asc",
        },
    )


def _dblp(variant: Mapping[str, Any], scope: Mapping[str, Any], page_size: int) -> NativeQuery:
    return NativeQuery(
        "dblp",
        _variant_id(variant),
        {"q": " ".join(_terms(variant)), "format": "json", "h": page_size},
    )


def _semantic_scholar(variant: Mapping[str, Any], scope: Mapping[str, Any], page_size: int) -> NativeQuery:
    start, end = _date_range(scope)
    return NativeQuery(
        "semantic_scholar",
        _variant_id(variant),
        {
            "query": " ".join(_terms(variant)),
            "year": f"{start[:4]}-{end[:4]}",
            "limit": page_size,
            "fields": "paperId,title,abstract,authors,year,venue,externalIds,publicationDate",
        },
    )


def _openalex(variant: Mapping[str, Any], scope: Mapping[str, Any], page_size: int) -> NativeQuery:
    start, end = _date_range(scope)
    return NativeQuery(
        "openalex",
        _variant_id(variant),
        {
            "search": " ".join(_terms(variant)),
            "filter": f"from_publication_date:{start},to_publication_date:{end}",
            "per-page": page_size,
            "sort": "publication_date:asc",
        },
    )


def _pubmed(variant: Mapping[str, Any], scope: Mapping[str, Any], page_size: int) -> NativeQuery:
    start, end = _date_range(scope)
    return NativeQuery(
        "pubmed",
        _variant_id(variant),
        {
            "db": "pubmed",
            "term": f"({_joined_terms(variant)}) AND {start}:{end}[pdat]",
            "retmax": page_size,
            "sort": "pub date",
        },
    )


def _europe_pmc(variant: Mapping[str, Any], scope: Mapping[str, Any], page_size: int) -> NativeQuery:
    start, end = _date_range(scope)
    return NativeQuery(
        "europe_pmc",
        _variant_id(variant),
        {
            "query": f"({_joined_terms(variant)}) AND FIRST_PDATE:[{start} TO {end}]",
            "format": "json",
            "pageSize": page_size,
            "sort": "FIRST_PDATE_ASC",
        },
    )


def _arxiv(variant: Mapping[str, Any], scope: Mapping[str, Any], page_size: int) -> NativeQuery:
    start, end = _date_range(scope)
    start_compact = start.replace("-", "") + "0000"
    end_compact = end.replace("-", "") + "2359"
    return NativeQuery(
        "arxiv",
        _variant_id(variant),
        {
            "search_query": f"all:({_joined_terms(variant)}) AND submittedDate:[{start_compact} TO {end_compact}]",
            "start": 0,
            "max_results": page_size,
            "sortBy": "submittedDate",
            "sortOrder": "ascending",
        },
    )


def _openreview(variant: Mapping[str, Any], scope: Mapping[str, Any], page_size: int) -> NativeQuery:
    start, end = _date_range(scope)
    return NativeQuery(
        "openreview",
        _variant_id(variant),
        {
            "term": " ".join(_terms(variant)),
            "venue_ids": [str(venue) for venue in scope.get("venues", ())],
            "date_from": start,
            "date_to": end,
            "limit": page_size,
        },
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
