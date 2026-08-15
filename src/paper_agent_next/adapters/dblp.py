"""DBLP conference table-of-contents adapter for Stage 1 membership."""

from __future__ import annotations

import re
from urllib.parse import quote
from xml.etree import ElementTree

from ..catalog import VenueSpec
from ..errors import CollectionError
from ..models import Pagination
from ..normalize import normalize_doi, normalize_text
from .base import CollectedPaper, CollectionResult, ParseReject, TextHttpClient

__all__ = ["DblpTocAdapter"]


_DBLP_TOC_URL = "https://dblp.org/db/conf/{series}/{series}{year}.xml"
_SERIES_PATTERN = re.compile(r"[a-z0-9]+\Z")
_YEAR_PATTERN = re.compile(r"[0-9]{4}\Z")
_SOURCE_KEYS = frozenset({"series", "exclude_title"})
_DOI_PREFIXES = (
    "doi:",
    "https://doi.org/",
    "http://doi.org/",
    "https://dx.doi.org/",
    "http://dx.doi.org/",
)


def _children(element: ElementTree.Element, name: str) -> list[ElementTree.Element]:
    return [child for child in element if child.tag == name]


def _element_text(element: ElementTree.Element | None) -> str | None:
    if element is None:
        return None
    value = "".join(element.itertext()).strip()
    return value or None


def _source_parameters(
    venue_spec: VenueSpec, year: int
) -> tuple[str, str | None]:
    source = venue_spec.source_for_year(year)
    unknown = sorted(set(source) - _SOURCE_KEYS)
    if unknown:
        raise CollectionError(
            "dblp: source contains unsupported parameter(s): " + ", ".join(unknown)
        )

    series = source.get("series")
    if not isinstance(series, str) or _SERIES_PATTERN.fullmatch(series) is None:
        raise CollectionError(
            "dblp: source.series must be a lowercase alphanumeric slug"
        )

    exclude_title = source.get("exclude_title")
    if exclude_title is not None and (
        not isinstance(exclude_title, str)
        or not exclude_title
        or exclude_title != exclude_title.strip()
    ):
        raise CollectionError(
            "dblp: source.exclude_title must be a non-empty scalar title"
        )
    return series, exclude_title


def _read_toc(http_client: TextHttpClient, url: str) -> ElementTree.Element:
    response = http_client.get_text(url)
    if not isinstance(response, str):
        raise CollectionError(f"dblp: response for {url} was not text")

    lowered = response.casefold()
    if "<!doctype" in lowered or "<!entity" in lowered:
        raise CollectionError(f"dblp: unsafe XML declaration in {url}")

    try:
        return ElementTree.fromstring(response, parser=ElementTree.XMLParser())
    except (ElementTree.ParseError, ValueError) as error:
        raise CollectionError(f"dblp: malformed TOC XML at {url}") from error


def _raw_records(
    root: ElementTree.Element, url: str
) -> tuple[ElementTree.Element, ...]:
    if root.tag != "bht":
        raise CollectionError(f"dblp: {url} is not a recognizable DBLP TOC")

    root_children = [child for child in root if isinstance(child.tag, str)]
    if any(child.tag not in {"h1", "h2", "dblpcites"} for child in root_children):
        raise CollectionError(f"dblp: {url} has an unrecognized TOC structure")
    citation_nodes = [child for child in root_children if child.tag == "dblpcites"]
    if not citation_nodes:
        raise CollectionError(f"dblp: {url} has an unrecognized TOC structure")

    records: list[ElementTree.Element] = []
    for citations in citation_nodes:
        citation_children = [
            child for child in citations if isinstance(child.tag, str)
        ]
        if any(child.tag != "r" for child in citation_children):
            raise CollectionError(f"dblp: {url} has unrecognized citation items")
        records.extend(citation_children)
    if not records:
        raise CollectionError(f"dblp: {url} has no authoritative zero proof")
    return tuple(records)


def _reject(url: str, position: int, reason_code: str, message: str) -> ParseReject:
    return ParseReject(
        source_locator=f"{url}#r-{position}",
        reason_code=reason_code,
        message=message,
    )


def _explicit_doi(value: str | None) -> str | None:
    if value is None:
        return None
    candidate = value.strip()
    lowered = candidate.casefold()
    if not candidate or not (
        lowered.startswith(_DOI_PREFIXES) or lowered.startswith("10.")
    ):
        return None
    return normalize_doi(candidate)


def _matches_excluded_title(title: str | None, exclude_title: str | None) -> bool:
    normalized_title = normalize_text(title)
    normalized_exclude = normalize_text(exclude_title)
    if normalized_title is None or normalized_exclude is None:
        return False

    title_key = normalized_title.casefold()
    exclude_key = normalized_exclude.casefold()
    return (
        title_key == exclude_key
        or title_key.endswith(".") and title_key[:-1] == exclude_key
        or exclude_key.endswith(".") and exclude_key[:-1] == title_key
    )


def _collect_record(
    raw_item: ElementTree.Element,
    *,
    url: str,
    position: int,
    year: int,
    exclude_title: str | None,
) -> tuple[str, CollectedPaper | ParseReject | None]:
    record_children = [child for child in raw_item if isinstance(child.tag, str)]
    if len(record_children) != 1:
        return "reject", _reject(
            url,
            position,
            "ambiguous_record",
            "raw citation item did not contain exactly one record",
        )

    record = record_children[0]
    if record.tag == "proceedings":
        return "exclude", None
    if record.tag != "inproceedings":
        return "reject", _reject(
            url,
            position,
            "unsupported_record_kind",
            f"raw citation item contained unsupported record kind {record.tag!r}",
        )

    title_nodes = _children(record, "title")
    if len(title_nodes) > 1:
        return "reject", _reject(
            url,
            position,
            "ambiguous_title",
            "inproceedings record contained multiple title elements",
        )
    title = _element_text(title_nodes[0] if title_nodes else None)

    raw_key = record.attrib.get("key")
    if not isinstance(raw_key, str) or not raw_key.strip():
        return "reject", _reject(
            url,
            position,
            "missing_source_id",
            "inproceedings record did not provide a stable DBLP key",
        )
    source_id = raw_key.strip()

    year_nodes = _children(record, "year")
    if len(year_nodes) != 1:
        return "reject", _reject(
            url,
            position,
            "missing_record_year" if not year_nodes else "ambiguous_record_year",
            "inproceedings record did not provide exactly one publication year",
        )
    record_year = _element_text(year_nodes[0])
    if record_year is None or _YEAR_PATTERN.fullmatch(record_year) is None:
        return "reject", _reject(
            url,
            position,
            "invalid_record_year",
            "inproceedings record has an invalid publication year",
        )
    if int(record_year) != year:
        return "reject", _reject(
            url,
            position,
            "record_year_conflict",
            f"inproceedings record year {record_year!r} conflicts with requested year {year}",
        )

    if _matches_excluded_title(title, exclude_title):
        return "exclude", None

    doi_values: list[str] = []
    for ee in _children(record, "ee"):
        doi = _explicit_doi(_element_text(ee))
        if doi is not None and doi not in doi_values:
            doi_values.append(doi)
    if len(doi_values) > 1:
        return "reject", _reject(
            url,
            position,
            "doi_conflict",
            f"stable DBLP key {source_id!r} has multiple normalized DOI values",
        )

    authors = tuple(
        author
        for author_node in _children(record, "author")
        if (author := _element_text(author_node)) is not None
    )
    landing_url = f"https://dblp.org/rec/{quote(source_id, safe='/')}"
    return "include", CollectedPaper(
        source_id=source_id,
        title=title,
        authors=authors,
        abstract=None,
        doi=doi_values[0] if doi_values else None,
        landing_url=landing_url,
        pdf_candidates=(),
    )


class DblpTocAdapter:
    """Collect one terminal DBLP conference TOC XML document."""

    source_name = "dblp_toc"

    def collect(
        self,
        venue_spec: VenueSpec,
        year: int,
        http_client: TextHttpClient,
    ) -> CollectionResult:
        series, exclude_title = _source_parameters(venue_spec, year)
        toc_url = _DBLP_TOC_URL.format(series=series, year=year)
        root = _read_toc(http_client, toc_url)
        raw_items = _raw_records(root, toc_url)

        papers_by_id: dict[str, CollectedPaper] = {}
        ordered_ids: list[str] = []
        parse_rejects: list[ParseReject] = []
        excluded_non_papers = 0
        duplicate_occurrences = 0

        for position, raw_item in enumerate(raw_items, start=1):
            classification, value = _collect_record(
                raw_item,
                url=toc_url,
                position=position,
                year=year,
                exclude_title=exclude_title,
            )
            if classification == "exclude":
                excluded_non_papers += 1
                continue
            if classification == "reject":
                if not isinstance(value, ParseReject):
                    raise CollectionError("dblp: internal parse classification error")
                parse_rejects.append(value)
                continue

            if not isinstance(value, CollectedPaper):
                raise CollectionError("dblp: internal paper classification error")
            paper = value
            prior = papers_by_id.get(paper.source_id)
            if prior is None:
                papers_by_id[paper.source_id] = paper
                ordered_ids.append(paper.source_id)
            elif prior == paper:
                duplicate_occurrences += 1
            else:
                parse_rejects.append(
                    _reject(
                        toc_url,
                        position,
                        "identity_conflict",
                        f"stable DBLP key {paper.source_id!r} has conflicting duplicate metadata",
                    )
                )

        return CollectionResult(
            source_name=self.source_name,
            papers=tuple(papers_by_id[source_id] for source_id in ordered_ids),
            raw_items=len(raw_items),
            excluded_non_papers=excluded_non_papers,
            duplicate_occurrences=duplicate_occurrences,
            parse_rejects=tuple(parse_rejects),
            pagination=Pagination(
                pages_fetched=1,
                terminal_reached=True,
                source_total=None,
            ),
        )
