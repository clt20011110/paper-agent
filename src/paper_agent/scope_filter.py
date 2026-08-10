"""Canonical post-filtering for approved QueryPlan scope fields."""

from __future__ import annotations

import calendar
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

from .canonical import content_hash
from .domain import FilterStatus, Paper


SCOPE_FILTER_VERSION = "query-scope-v1"
SCREENING_SCOPE_HASH_VERSION = "query-plan-screening-scope-v1"


@dataclass(frozen=True, slots=True)
class ScopeDecision:
    status: FilterStatus
    reason_code: str
    input_hash: str


def screening_scope_hash(plan: Mapping[str, Any]) -> str:
    """Bind the research intent and every screening boundary as one value."""
    return content_hash(
        {
            "kind": SCREENING_SCOPE_HASH_VERSION,
            "research": plan["research"],
            "inclusion": plan["inclusion"],
            "scope": plan["scope"],
        }
    )


def evaluate_scope(
    paper: Paper,
    source_metadata: Sequence[Mapping[str, Any]],
    scope: Mapping[str, Any],
) -> ScopeDecision:
    input_hash = content_hash(
        {
            "paper": paper.to_dict(),
            "source_metadata": [dict(value) for value in source_metadata],
            "scope": dict(scope),
            "implementation_version": SCOPE_FILTER_VERSION,
        }
    )
    checks = (
        ("date", _date_match(paper, str(scope["date_from"]), str(scope["date_to"]))),
        (
            "venue",
            _set_match(
                scope.get("venues", ()),
                (paper.venue_id, paper.venue_name)
                + _metadata_values(source_metadata, ("venue_id", "venue", "venue_name", "container-title")),
                normalizer=_text,
            ),
        ),
        (
            "field",
            _set_match(
                scope.get("fields", ()),
                _metadata_values(
                    source_metadata,
                    ("field", "fields", "fields_of_study", "concepts", "topics"),
                ),
                normalizer=_text,
            ),
        ),
        (
            "language",
            _set_match(
                scope.get("languages", ()),
                _metadata_values(source_metadata, ("language", "languages", "lang")),
                normalizer=_language,
            ),
        ),
        (
            "document_type",
            _set_match(
                scope.get("document_types", ()),
                _metadata_values(
                    source_metadata,
                    ("document_type", "document_types", "publication_type", "type", "genre"),
                ),
                normalizer=_document_type,
            ),
        ),
    )
    for name, result in checks:
        if result is False:
            return ScopeDecision(FilterStatus.IRRELEVANT, f"scope_{name}_mismatch", input_hash)
    for name, result in checks:
        if result is None:
            return ScopeDecision(FilterStatus.NEEDS_REVIEW, f"scope_{name}_unverified", input_hash)
    return ScopeDecision(FilterStatus.RELEVANT, "scope_match", input_hash)


def _date_match(paper: Paper, start: str, end: str) -> bool | None:
    interval = _publication_interval(paper.publication_date, paper.year)
    if interval is None:
        return None
    lower, upper = interval
    requested_start = date.fromisoformat(start)
    requested_end = date.fromisoformat(end)
    if upper < requested_start or lower > requested_end:
        return False
    if requested_start <= lower and upper <= requested_end:
        return True
    return None


def _publication_interval(value: str | None, year: int | None) -> tuple[date, date] | None:
    if value:
        parts = value.strip().split("-")
        try:
            parsed_year = int(parts[0])
            if len(parts) == 1:
                return date(parsed_year, 1, 1), date(parsed_year, 12, 31)
            parsed_month = int(parts[1])
            if len(parts) == 2:
                return (
                    date(parsed_year, parsed_month, 1),
                    date(parsed_year, parsed_month, calendar.monthrange(parsed_year, parsed_month)[1]),
                )
            parsed = date.fromisoformat("-".join(parts[:3]))
            return parsed, parsed
        except (ValueError, IndexError):
            return None
    if year is not None:
        return date(year, 1, 1), date(year, 12, 31)
    return None


def _set_match(
    requested: Sequence[Any],
    observed: Sequence[Any],
    *,
    normalizer: Callable[[Any], str],
) -> bool | None:
    expected = {value for item in requested if (value := normalizer(item))}
    if not expected:
        return True
    actual = {value for item in observed if (value := normalizer(item))}
    if not actual:
        return None
    return bool(expected & actual)


def _metadata_values(
    documents: Sequence[Mapping[str, Any]], keys: Sequence[str]
) -> tuple[Any, ...]:
    values: list[Any] = []
    wanted = {key.casefold() for key in keys}
    for document in documents:
        for key, value in document.items():
            if str(key).casefold() in wanted:
                values.extend(_flatten(value))
    return tuple(values)


def _flatten(value: Any) -> tuple[Any, ...]:
    if isinstance(value, Mapping):
        return tuple(item for nested in value.values() for item in _flatten(nested))
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(item for nested in value for item in _flatten(nested))
    return (value,)


def _text(value: Any) -> str:
    return " ".join(str(value).strip().casefold().split()) if value is not None else ""


def _language(value: Any) -> str:
    normalized = _text(value)
    return {"eng": "en", "english": "en", "zho": "zh", "chi": "zh", "chinese": "zh"}.get(
        normalized, normalized
    )


def _document_type(value: Any) -> str:
    normalized = re.sub(r"[\s_]", "-", _text(value))
    return {
        "journal-article": "article",
        "research-article": "article",
        "proceedings-article": "article",
        "conference-paper": "article",
    }.get(normalized, normalized)
