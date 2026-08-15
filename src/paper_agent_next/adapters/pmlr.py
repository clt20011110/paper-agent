from __future__ import annotations

from dataclasses import dataclass, replace
from html.parser import HTMLParser
import re
from urllib.parse import urljoin, urlsplit

from ..catalog import VenueSpec
from ..errors import CollectionError
from ..models import Pagination
from .base import CollectedPaper, CollectionResult, ParseReject, TextHttpClient

__all__ = ["PmlrAdapter"]


_PMLR_BASE_URL = "https://proceedings.mlr.press/"
_PMLR_HOST = "proceedings.mlr.press"
_VOLUME_PATTERN = re.compile(r"v[0-9]+\Z")
_SLUG_PATTERN = r"[A-Za-z0-9][A-Za-z0-9._-]*"


def _clean(parts: list[str]) -> str | None:
    value = " ".join("".join(parts).split())
    return value or None


@dataclass(slots=True)
class _PaperCapture:
    title_parts: list[str]
    authors_parts: list[str]
    author_names: list[str]
    author_parts: list[str]
    hrefs: list[str]
    title_depth: int | None = None
    title_tag: str | None = None
    authors_depth: int | None = None
    authors_tag: str | None = None
    author_depth: int | None = None


@dataclass(frozen=True, slots=True)
class _RawPaper:
    title: str | None
    authors: tuple[str, ...]
    hrefs: tuple[str, ...]


class _PmlrHtmlParser(HTMLParser):
    def __init__(self, mode: str) -> None:
        super().__init__(convert_charrefs=True)
        self.mode = mode
        self.stack: list[str] = []
        self.paper_depth: int | None = None
        self.paper: _PaperCapture | None = None
        self.papers: list[_RawPaper] = []
        self.abstract_depth: int | None = None
        self.abstract_parts: list[str] = []

    def _append_active(self, text: str) -> None:
        if self.mode == "detail":
            if self.abstract_depth is not None:
                self.abstract_parts.append(text)
            return
        if self.paper is None:
            return
        if self.paper.title_depth is not None:
            self.paper.title_parts.append(text)
        if self.paper.authors_depth is not None:
            self.paper.authors_parts.append(text)
        if self.paper.author_depth is not None:
            self.paper.author_parts.append(text)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.casefold()
        attributes = {name.casefold(): value or "" for name, value in attrs}
        if tag == "br":
            self._append_active(" ")
            return
        self.stack.append(tag)
        depth = len(self.stack)

        if self.mode == "detail":
            if (
                self.abstract_depth is None
                and tag == "div"
                and attributes.get("id", "").casefold() == "abstract"
            ):
                self.abstract_depth = depth
            elif self.abstract_depth is not None and tag == "p":
                self.abstract_parts.append(" ")
            return

        if self.paper is None and tag == "div" and "paper" in attributes.get("class", "").split():
            self.paper = _PaperCapture([], [], [], [], [])
            self.paper_depth = depth
        if self.paper is None:
            return
        if tag == "a":
            self.paper.hrefs.append(attributes.get("href", ""))
            if self.paper.authors_depth is not None and self.paper.author_depth is None:
                self.paper.author_depth = depth
                self.paper.author_parts = []
        classes = attributes.get("class", "").split()
        if "title" in classes and self.paper.title_depth is None:
            self.paper.title_depth = depth
            self.paper.title_tag = tag
        if "authors" in classes and self.paper.authors_depth is None:
            self.paper.authors_depth = depth
            self.paper.authors_tag = tag

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag.casefold() != "br":
            self.handle_endtag(tag)

    def handle_data(self, data: str) -> None:
        self._append_active(data)

    def _finish_author(self) -> None:
        if self.paper is None or self.paper.author_depth is None:
            return
        author = _clean(self.paper.author_parts)
        if author is not None:
            self.paper.author_names.append(author)
        self.paper.author_depth = None
        self.paper.author_parts = []

    def _finish_paper(self) -> None:
        if self.paper is None:
            return
        self._finish_author()
        authors = tuple(self.paper.author_names)
        if not authors:
            authors = tuple(
                part.strip()
                for part in (_clean(self.paper.authors_parts) or "").split(",")
                if part.strip()
            )
        self.papers.append(
            _RawPaper(
                title=_clean(self.paper.title_parts),
                authors=authors,
                hrefs=tuple(self.paper.hrefs),
            )
        )
        self.paper = None
        self.paper_depth = None

    def _pop(self, tag: str) -> None:
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index] == tag:
                del self.stack[index:]
                return

    def handle_endtag(self, tag: str) -> None:
        tag = tag.casefold()
        depth = len(self.stack)
        if self.mode == "detail":
            if self.abstract_depth == depth and tag == "div":
                self.abstract_depth = None
            elif self.abstract_depth is not None and tag == "p":
                self.abstract_parts.append(" ")
        elif self.paper is not None:
            if self.paper.author_depth == depth and tag == "a":
                self._finish_author()
            if self.paper.title_depth == depth and tag == self.paper.title_tag:
                self.paper.title_depth = None
            if self.paper.authors_depth == depth and tag == self.paper.authors_tag:
                self.paper.authors_depth = None
            if self.paper_depth == depth and tag == "div":
                self._finish_paper()
        self._pop(tag)

    def finish(self) -> None:
        if self.mode == "volume" and self.paper is not None:
            self._finish_paper()

    @property
    def abstract(self) -> str | None:
        return _clean(self.abstract_parts)


def _read_html(http_client: TextHttpClient, url: str) -> str:
    text = http_client.get_text(url)
    if not isinstance(text, str):
        raise CollectionError(f"pmlr: response for {url} was not text")
    return text


def _official_landing_url(base_url: str, href: str) -> str | None:
    if not href or any(character.isspace() for character in href):
        return None
    try:
        absolute = urljoin(base_url, href)
        parsed = urlsplit(absolute)
        hostname = parsed.hostname
        parsed.port
    except (TypeError, ValueError):
        return None
    if (
        parsed.scheme not in {"http", "https"}
        or hostname is None
        or hostname.casefold() != _PMLR_HOST
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port is not None
    ):
        return None
    return absolute


def _source_native_pdf_url(href: str, volume: str, slug: str) -> str | None:
    if not href or any(character.isspace() for character in href):
        return None
    try:
        parsed = urlsplit(href)
        hostname = parsed.hostname
        parsed.port
    except (TypeError, ValueError):
        return None
    if (
        hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port is not None
        or parsed.query
        or parsed.fragment
    ):
        return None
    if (
        parsed.scheme == "https"
        and hostname.casefold() == "raw.githubusercontent.com"
        and parsed.path == f"/mlresearch/{volume}/main/assets/{slug}/{slug}.pdf"
    ):
        return href
    if (
        parsed.scheme in {"http", "https"}
        and hostname.casefold() == _PMLR_HOST
        and parsed.path == f"/{volume}/{slug}.pdf"
    ):
        return href
    return None


class PmlrAdapter:
    source_name = "pmlr"

    def collect(
        self,
        venue_spec: VenueSpec,
        year: int,
        http_client: TextHttpClient,
    ) -> CollectionResult:
        source = venue_spec.source_for_year(year)
        volume = source.get("volume")
        if not isinstance(volume, str) or _VOLUME_PATTERN.fullmatch(volume) is None:
            raise CollectionError(
                "pmlr: source_for_year(year) must provide an explicit volume such as 'v235'"
            )

        volume_url = f"{_PMLR_BASE_URL}{volume}/"
        volume_parser = _PmlrHtmlParser("volume")
        volume_parser.feed(_read_html(http_client, volume_url))
        volume_parser.close()
        volume_parser.finish()
        if not volume_parser.papers:
            raise CollectionError(f"pmlr: volume page {volume_url} contains no div.paper items")

        path_pattern = re.compile(rf"/{re.escape(volume)}/(?P<slug>{_SLUG_PATTERN})\.html\Z")
        volume_prefix = f"/{volume}/"
        included_by_id: dict[str, CollectedPaper] = {}
        ordered_ids: list[str] = []
        parse_rejects: list[ParseReject] = []
        excluded_non_papers = 0
        duplicate_occurrences = 0

        for position, raw in enumerate(volume_parser.papers, start=1):
            locator = f"{volume_url}#paper-{position}"
            if raw.title is not None and raw.title.casefold() == "preface":
                excluded_non_papers += 1
                continue

            landing_candidates: list[tuple[str, str, str]] = []
            for href in raw.hrefs:
                absolute = _official_landing_url(volume_url, href)
                if absolute is None:
                    continue
                path = urlsplit(absolute).path
                landing = path_pattern.fullmatch(path)
                if landing is not None:
                    slug = landing.group("slug")
                    landing_candidates.append(
                        (f"{volume}/{slug}", absolute, slug)
                    )
            landing_candidates = list(dict.fromkeys(landing_candidates))
            if not landing_candidates:
                reason_code = "missing_landing_url"
                for href in raw.hrefs:
                    try:
                        if urlsplit(href).path.casefold().endswith(".html"):
                            reason_code = "invalid_landing_url"
                            break
                    except ValueError:
                        continue
                parse_rejects.append(
                    ParseReject(
                        source_locator=locator,
                        reason_code=reason_code,
                        message="source item did not provide a valid PMLR landing URL",
                    )
                )
                continue
            if len({source_id for source_id, _, _ in landing_candidates}) != 1 or len(
                {landing_url for _, landing_url, _ in landing_candidates}
            ) != 1:
                parse_rejects.append(
                    ParseReject(
                        source_locator=locator,
                        reason_code="ambiguous_landing_url",
                        message="source item provided multiple landing URLs",
                    )
                )
                continue

            source_id, landing_url, slug = landing_candidates[0]
            pdf_candidates: list[str] = []
            for href in raw.hrefs:
                candidate = _source_native_pdf_url(href, volume, slug)
                if candidate is not None:
                    pdf_candidates.append(candidate)
            pdf_candidates = list(dict.fromkeys(pdf_candidates))
            candidate = CollectedPaper(
                source_id=source_id,
                title=raw.title,
                authors=raw.authors,
                abstract=None,
                landing_url=landing_url,
                pdf_candidates=tuple(pdf_candidates),
            )
            prior = included_by_id.get(source_id)
            if prior is None:
                included_by_id[source_id] = candidate
                ordered_ids.append(source_id)
            elif (
                prior.title,
                prior.authors,
                prior.landing_url,
                prior.pdf_candidates,
            ) == (
                candidate.title,
                candidate.authors,
                candidate.landing_url,
                candidate.pdf_candidates,
            ):
                duplicate_occurrences += 1
            else:
                parse_rejects.append(
                    ParseReject(
                        source_locator=locator,
                        reason_code="identity_conflict",
                        message=f"stable ID {source_id!r} has conflicting duplicate metadata",
                    )
                )

        papers = []
        for source_id in ordered_ids:
            candidate = included_by_id[source_id]
            try:
                detail_html = _read_html(http_client, candidate.landing_url)
            except CollectionError:
                abstract = None
            else:
                detail_parser = _PmlrHtmlParser("detail")
                detail_parser.feed(detail_html)
                detail_parser.close()
                detail_parser.finish()
                abstract = detail_parser.abstract
            papers.append(replace(candidate, abstract=abstract))

        return CollectionResult(
            source_name=self.source_name,
            papers=tuple(papers),
            raw_items=len(volume_parser.papers),
            excluded_non_papers=excluded_non_papers,
            duplicate_occurrences=duplicate_occurrences,
            parse_rejects=tuple(parse_rejects),
            pagination=Pagination(pages_fetched=1, terminal_reached=True, source_total=None),
        )
