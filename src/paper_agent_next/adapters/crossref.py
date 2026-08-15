"""Crossref journal work-list adapter for authoritative serial membership."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import re
from urllib.parse import quote, urlencode, urlunsplit

from ..catalog import VenueSpec
from ..errors import CollectionError
from ..models import Pagination, SourceTotal, SourceTotalScope
from ..normalize import normalize_doi, normalize_text
from .base import CollectedPaper, CollectionResult, ParseReject, TextHttpClient

__all__ = ["CrossrefSerialAdapter"]


_ROWS = 1000
_ISSN_PATTERN = re.compile(r"[0-9]{4}-[0-9]{3}[0-9X]\Z")
_SOURCE_KEYS = frozenset({"issn"})
_MISSING = object()


class _PageFailure(Exception):
    """A source response or pagination invariant failed at a page boundary."""


class _ItemReject(Exception):
    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.message = message


@dataclass(frozen=True, slots=True)
class _Page:
    total: int
    items: tuple[object, ...]
    next_cursor: object


def _source_issn(venue_spec: VenueSpec, year: int) -> str:
    try:
        source = venue_spec.source_for_year(year)
    except Exception as error:
        raise CollectionError(
            "crossref_serial: could not resolve venue source parameters"
        ) from error

    if not isinstance(source, Mapping):
        raise CollectionError("crossref_serial: source parameters must be a mapping")
    unknown = sorted(set(source) - _SOURCE_KEYS)
    if unknown:
        raise CollectionError(
            "crossref_serial: source contains unsupported parameter(s): "
            + ", ".join(unknown)
        )

    issn = source.get("issn")
    if not isinstance(issn, str) or _ISSN_PATTERN.fullmatch(issn) is None:
        raise CollectionError(
            "crossref_serial: source.issn must match NNNN-NNNX with uppercase X"
        )
    return issn


def _request_url(issn: str, year: int, cursor: str) -> str:
    query = urlencode(
        (
            (
                "filter",
                f"from-pub-date:{year:04d}-01-01,"
                f"until-pub-date:{year:04d}-12-31,type:journal-article",
            ),
            ("rows", str(_ROWS)),
            ("cursor", cursor),
        )
    )
    return urlunsplit(
        (
            "https",
            "api.crossref.org",
            f"/journals/{quote(issn, safe='')}/works",
            query,
            "",
        )
    )


def _decode_page(response: object, url: str) -> _Page:
    if not isinstance(response, str):
        raise _PageFailure(f"crossref_serial: response for {url} was not text")
    try:
        payload = json.loads(response)
    except (TypeError, ValueError) as error:
        raise _PageFailure(f"crossref_serial: invalid JSON response from {url}") from error

    if not isinstance(payload, dict):
        raise _PageFailure(f"crossref_serial: top-level response for {url} was not an object")
    if payload.get("status") != "ok":
        raise _PageFailure(f"crossref_serial: response for {url} did not have status ok")
    if payload.get("message-type") != "work-list":
        raise _PageFailure(
            f"crossref_serial: response for {url} did not have message-type work-list"
        )

    message = payload.get("message")
    if not isinstance(message, dict):
        raise _PageFailure(f"crossref_serial: message for {url} was not an object")

    total = message.get("total-results", _MISSING)
    if type(total) is not int or total < 0:
        raise _PageFailure(
            f"crossref_serial: total-results for {url} was not a non-negative integer"
        )
    items = message.get("items", _MISSING)
    if not isinstance(items, list):
        raise _PageFailure(f"crossref_serial: items for {url} was not an array")
    return _Page(total=total, items=tuple(items), next_cursor=message.get("next-cursor", _MISSING))


def _fetch_page(http_client: TextHttpClient, url: str) -> _Page:
    try:
        response = http_client.get_text(url)
    except Exception as error:
        raise _PageFailure(f"crossref_serial: GET {url} failed") from error
    return _decode_page(response, url)


def _reject(reason_code: str, message: str) -> None:
    raise _ItemReject(reason_code, message)


def _title(item: dict[str, object]) -> str:
    raw_title = item.get("title", _MISSING)
    if not isinstance(raw_title, list):
        _reject("invalid_title", "item title was not a Crossref title array")

    candidates: list[str] = []
    for value in raw_title:
        if not isinstance(value, str):
            _reject("invalid_title", "item title array contained a non-string value")
        normalized = normalize_text(value)
        if normalized is not None and normalized not in candidates:
            candidates.append(normalized)
    if not candidates:
        _reject("missing_title", "item did not provide a non-empty normalized title")
    if len(candidates) != 1:
        _reject("ambiguous_title", "item provided more than one normalized title")
    return candidates[0]


def _publication_year(item: dict[str, object], year: int) -> None:
    published = item.get("published", _MISSING)
    if not isinstance(published, dict):
        _reject("missing_published_date", "item did not provide published.date-parts")
    date_parts = published.get("date-parts", _MISSING)
    if not isinstance(date_parts, list) or len(date_parts) != 1:
        _reject("invalid_published_date", "published.date-parts was not one date part")
    date_part = date_parts[0]
    if (
        not isinstance(date_part, list)
        or not 1 <= len(date_part) <= 3
        or any(type(value) is not int for value in date_part)
    ):
        _reject("invalid_published_date", "published.date-parts contained invalid values")
    if date_part[0] != year:
        _reject(
            "publication_year_mismatch",
            f"published year {date_part[0]!r} did not match requested year {year}",
        )


def _authors(item: dict[str, object]) -> tuple[str, ...]:
    raw_authors = item.get("author", _MISSING)
    if raw_authors is _MISSING:
        return ()
    if not isinstance(raw_authors, list):
        _reject("invalid_authors", "item author field was not an array")

    authors: list[str] = []
    for author in raw_authors:
        if not isinstance(author, dict):
            _reject("invalid_authors", "item author array contained a non-object")

        raw_name = author.get("name", _MISSING)
        if raw_name is not _MISSING and raw_name is not None:
            if not isinstance(raw_name, str):
                _reject("invalid_author_name", "author name was not a string")
            name = normalize_text(raw_name)
            if name is not None:
                authors.append(name)
                continue

        parts: list[str] = []
        for field in ("given", "family"):
            value = author.get(field, _MISSING)
            if value is _MISSING or value is None:
                continue
            if not isinstance(value, str):
                _reject("invalid_author_name", f"author {field} was not a string")
            normalized = normalize_text(value)
            if normalized is not None:
                parts.append(normalized)
        if not parts:
            _reject("invalid_author_name", "author did not provide a usable name")
        authors.append(" ".join(parts))
    return tuple(authors)


def _abstract(item: dict[str, object]) -> str | None:
    raw_abstract = item.get("abstract", _MISSING)
    if raw_abstract is _MISSING or raw_abstract is None:
        return None
    if not isinstance(raw_abstract, str):
        _reject("invalid_abstract", "item abstract was neither a string nor null")
    return normalize_text(raw_abstract)


def _pdf_candidates(item: dict[str, object]) -> tuple[str, ...]:
    raw_links = item.get("link", _MISSING)
    if raw_links is _MISSING:
        return ()
    if not isinstance(raw_links, list):
        _reject("invalid_links", "item link field was not an array")

    candidates: list[str] = []
    for link in raw_links:
        if not isinstance(link, dict):
            _reject("invalid_links", "item link array contained a non-object")
        if link.get("content-type") != "application/pdf":
            continue
        raw_url = link.get("URL")
        if not isinstance(raw_url, str):
            continue
        candidate = raw_url.strip()
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    return tuple(candidates)


def _parse_item(item: object, *, issn: str, year: int) -> CollectedPaper:
    if not isinstance(item, dict):
        _reject("item_not_object", "Crossref item was not an object")

    raw_doi = item.get("DOI", _MISSING)
    if raw_doi is _MISSING or raw_doi is None:
        _reject("missing_doi", "item did not provide a DOI")
    if not isinstance(raw_doi, str):
        _reject("invalid_doi", "item DOI was not a string")
    if not raw_doi.strip():
        _reject("missing_doi", "item DOI was empty")
    doi = normalize_doi(raw_doi)
    if doi is None:
        _reject("invalid_doi", "item DOI could not be normalized")

    if item.get("type", _MISSING) != "journal-article":
        _reject("non_journal_article", "item type was not journal-article")

    raw_issns = item.get("ISSN", _MISSING)
    if not isinstance(raw_issns, list):
        _reject("invalid_issn", "item ISSN field was not a string array")
    if any(not isinstance(value, str) for value in raw_issns):
        _reject("invalid_issn", "item ISSN array contained a non-string value")
    if issn not in raw_issns:
        _reject("issn_mismatch", "item ISSN array did not contain configured ISSN")

    _publication_year(item, year)
    title = _title(item)
    authors = _authors(item)
    abstract = _abstract(item)
    pdf_candidates = _pdf_candidates(item)
    return CollectedPaper(
        source_id=doi,
        title=title,
        authors=authors,
        abstract=abstract,
        doi=doi,
        landing_url=f"https://doi.org/{doi}",
        pdf_candidates=pdf_candidates,
    )


def _consume_items(
    items: tuple[object, ...],
    *,
    url: str,
    raw_items: int,
    issn: str,
    year: int,
    papers_by_doi: dict[str, CollectedPaper],
    ordered_dois: list[str],
    parse_rejects: list[ParseReject],
) -> tuple[int, int]:
    duplicate_occurrences = 0
    for offset, item in enumerate(items, start=1):
        locator = f"{url}#item-{raw_items + offset}"
        try:
            paper = _parse_item(item, issn=issn, year=year)
        except _ItemReject as error:
            parse_rejects.append(ParseReject(locator, error.reason_code, error.message))
            continue

        prior = papers_by_doi.get(paper.source_id)
        if prior is None:
            papers_by_doi[paper.source_id] = paper
            ordered_dois.append(paper.source_id)
        elif prior == paper:
            duplicate_occurrences += 1
        else:
            parse_rejects.append(
                ParseReject(
                    locator,
                    "identity_conflict",
                    f"DOI {paper.source_id!r} had conflicting normalized metadata",
                )
            )
    return len(items), duplicate_occurrences


class CrossrefSerialAdapter:
    """Enumerate one journal ISSN's Crossref journal-article work list."""

    source_name = "crossref_serial"

    def collect(
        self,
        venue_spec: VenueSpec,
        year: int,
        http_client: TextHttpClient,
    ) -> CollectionResult:
        if type(year) is not int or not 1000 <= year <= 9999:
            raise CollectionError("crossref_serial: year must be a four-digit integer")
        issn = _source_issn(venue_spec, year)

        papers_by_doi: dict[str, CollectedPaper] = {}
        ordered_dois: list[str] = []
        parse_rejects: list[ParseReject] = []
        raw_items = 0
        duplicate_occurrences = 0
        pages_fetched = 0
        total_results: int | None = None
        terminal_reached = False
        cursor = "*"
        seen_cursors: set[str] = set()

        while True:
            if cursor in seen_cursors:
                failure = _PageFailure("crossref_serial: cursor cycle detected")
                if pages_fetched == 0:
                    raise CollectionError(str(failure)) from failure
                break
            seen_cursors.add(cursor)
            url = _request_url(issn, year, cursor)

            try:
                page = _fetch_page(http_client, url)
            except _PageFailure as error:
                if pages_fetched == 0:
                    raise CollectionError(str(error)) from error
                break

            if total_results is None:
                total_results = page.total
            elif page.total != total_results:
                failure = _PageFailure("crossref_serial: total-results changed between pages")
                if pages_fetched == 0:
                    raise CollectionError(str(failure)) from failure
                break

            remaining = page.total - raw_items
            if len(page.items) > _ROWS or len(page.items) > remaining:
                failure = _PageFailure("crossref_serial: page items exceeded the remaining total")
                if pages_fetched == 0:
                    raise CollectionError(str(failure)) from failure
                break

            pages_fetched += 1
            consumed, duplicates = _consume_items(
                page.items,
                url=url,
                raw_items=raw_items,
                issn=issn,
                year=year,
                papers_by_doi=papers_by_doi,
                ordered_dois=ordered_dois,
                parse_rejects=parse_rejects,
            )
            raw_items += consumed
            duplicate_occurrences += duplicates

            if raw_items == total_results:
                terminal_reached = True
                break
            if len(page.items) < _ROWS:
                break

            next_cursor = page.next_cursor
            if not isinstance(next_cursor, str) or not next_cursor.strip():
                break
            if next_cursor in seen_cursors:
                break
            cursor = next_cursor

        if total_results is None:
            raise CollectionError("crossref_serial: no authoritative page was collected")
        return CollectionResult(
            source_name=self.source_name,
            papers=tuple(papers_by_doi[doi] for doi in ordered_dois),
            raw_items=raw_items,
            excluded_non_papers=0,
            duplicate_occurrences=duplicate_occurrences,
            parse_rejects=tuple(parse_rejects),
            pagination=Pagination(
                pages_fetched=pages_fetched,
                terminal_reached=terminal_reached,
                source_total=SourceTotal(total_results, SourceTotalScope.RAW_ITEMS),
            ),
        )
