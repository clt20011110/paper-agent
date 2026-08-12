"""Official, metadata-only routes for conference primary providers.

Each handler is registered independently so a new conference source can be
added without changing the dispatch code.  Handlers receive a fetch callback;
``ControlledHTTPTransport`` supplies one backed by ``ProviderRuntime`` so rate
limits, terms, retries, caching, and audit bytes apply to every request.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser
import json
import re
from typing import Any, Callable, Mapping, Protocol
from urllib.parse import quote, urlencode, urljoin, urlsplit
from xml.etree import ElementTree

from paper_agent.provider_runtime import ProviderPolicyDenied, ProviderRequestError


class VenueFetchResponse(Protocol):
    body: bytes
    content_type: str


class VenueFetch(Protocol):
    def __call__(
        self, url: str, api_version: str, policy_provider: str | None = None
    ) -> VenueFetchResponse: ...


VenueHandler = Callable[[str, Mapping[str, Any], VenueFetch], "VenueOperationResult"]


@dataclass(frozen=True, slots=True)
class VenueOperationResult:
    payload: Mapping[str, Any]
    bodies: tuple[bytes, ...] = ()


_HANDLERS: dict[str, VenueHandler] = {}


def register_venue_handler(provider: str) -> Callable[[VenueHandler], VenueHandler]:
    """Register one provider without extending a central conditional chain."""

    def register(handler: VenueHandler) -> VenueHandler:
        if provider in _HANDLERS:
            raise ValueError(f"duplicate venue HTTP handler: {provider}")
        _HANDLERS[provider] = handler
        return handler

    return register


def venue_provider_names() -> tuple[str, ...]:
    return tuple(sorted(_HANDLERS))


def execute_venue_operation(
    provider: str,
    operation: str,
    parameters: Mapping[str, Any],
    fetch: VenueFetch,
) -> VenueOperationResult:
    try:
        handler = _HANDLERS[provider]
    except KeyError as error:
        raise ValueError(f"no official venue HTTP mapping for {provider}:{operation}") from error
    return handler(operation, parameters, fetch)


@dataclass(slots=True)
class _HTMLNode:
    tag: str
    attributes: dict[str, str]
    children: list["_HTMLNode"] = field(default_factory=list)
    content: list[str | "_HTMLNode"] = field(default_factory=list)

    @property
    def text(self) -> str:
        values = [value.text if isinstance(value, _HTMLNode) else value for value in self.content]
        return " ".join(" ".join(values).split())

    def has_class(self, name: str) -> bool:
        return name in self.attributes.get("class", "").split()


class _HTMLTreeParser(HTMLParser):
    _VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}
    _AUTO_CLOSE = {
        "li": {"li"},
        "dt": {"dt", "dd"},
        "dd": {"dt", "dd"},
        "tr": {"tr"},
        "td": {"td", "th"},
        "th": {"td", "th"},
        "p": {"p"},
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = _HTMLNode("document", {})
        self._stack = [self.root]

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.casefold()
        if tag in self._AUTO_CLOSE and self._stack[-1].tag in self._AUTO_CLOSE[tag]:
            self._stack.pop()
        node = _HTMLNode(tag, {name.casefold(): value or "" for name, value in attrs})
        self._stack[-1].children.append(node)
        self._stack[-1].content.append(node)
        if tag not in self._VOID:
            self._stack.append(node)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag.casefold() not in self._VOID:
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.casefold()
        for index in range(len(self._stack) - 1, 0, -1):
            if self._stack[index].tag == tag:
                del self._stack[index:]
                return

    def handle_data(self, data: str) -> None:
        if data.strip():
            self._stack[-1].content.append(data)


def _html(body: bytes, provider: str) -> _HTMLNode:
    parser = _HTMLTreeParser()
    try:
        parser.feed(body.decode("utf-8"))
    except UnicodeDecodeError as error:
        raise ProviderRequestError(f"{provider}: response is not UTF-8 HTML") from error
    return parser.root


def _walk(node: _HTMLNode):
    for child in node.children:
        yield child
        yield from _walk(child)


def _nodes(node: _HTMLNode, tag: str | None = None, class_name: str | None = None) -> list[_HTMLNode]:
    return [
        child
        for child in _walk(node)
        if (tag is None or child.tag == tag) and (class_name is None or child.has_class(class_name))
    ]


def _first(node: _HTMLNode, tag: str | None = None, class_name: str | None = None) -> _HTMLNode | None:
    return next(iter(_nodes(node, tag, class_name)), None)


def _require_operation(provider: str, operation: str, expected: set[str]) -> None:
    if operation not in expected:
        raise ValueError(f"no official venue HTTP mapping for {provider}:{operation}")


def _year(parameters: Mapping[str, Any]) -> int:
    value = parameters.get("year")
    if value is not None:
        year = int(value)
        if year < 1900 or year > 2200:
            raise ValueError("conference year is outside the supported range")
        return year
    start = str(parameters.get("date_from") or "")[:4]
    end = str(parameters.get("date_to") or "")[:4]
    if start.isdigit() and (not end or end == start):
        return int(start)
    if end.isdigit() and not start:
        return int(end)
    raise ValueError("conference discovery requires year or a single-year date window")


def _page_size(parameters: Mapping[str, Any]) -> int:
    value = int(parameters.get("page_size") or 100)
    if value < 1 or value > 1000:
        raise ValueError("page_size must be between 1 and 1000")
    return value


def _offset(parameters: Mapping[str, Any]) -> int:
    raw = parameters.get("cursor")
    if raw in (None, ""):
        return 0
    if not str(raw).isdigit():
        raise ValueError("venue cursor must be a non-negative integer offset")
    return int(raw)


def _page(entries: list[dict[str, Any]], parameters: Mapping[str, Any]) -> tuple[list[dict[str, Any]], str | None]:
    start = _offset(parameters)
    size = _page_size(parameters)
    selected = entries[start : start + size]
    next_cursor = str(start + len(selected)) if start + len(selected) < len(entries) else None
    return selected, next_cursor


def _date_bounds(value: str | None, year: int | None) -> tuple[str, str]:
    if value and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return value, value
    if value and re.fullmatch(r"\d{4}-\d{2}", value):
        month = int(value[-2:])
        next_month = f"{int(value[:4]) + 1}-01-01" if month == 12 else f"{value[:4]}-{month + 1:02d}-01"
        end = datetime.fromisoformat(next_month).date().toordinal() - 1
        return f"{value}-01", datetime.fromordinal(end).date().isoformat()
    resolved_year = int(value[:4]) if value and value[:4].isdigit() else year
    if resolved_year is None:
        return "0000-01-01", "9999-12-31"
    return f"{resolved_year:04d}-01-01", f"{resolved_year:04d}-12-31"


def _in_window(entry: Mapping[str, Any], parameters: Mapping[str, Any]) -> bool:
    start, end = _date_bounds(
        str(entry["publication_date"]) if entry.get("publication_date") else None,
        int(entry["year"]) if entry.get("year") is not None else None,
    )
    date_from = str(parameters.get("date_from") or "0000-01-01")
    date_to = str(parameters.get("date_to") or "9999-12-31")
    if end < date_from or start > date_to:
        return False
    if parameters.get("volume") is not None:
        actual_volume = str(entry.get("volume"))
        requested_volume = str(parameters["volume"])
        if actual_volume != requested_volume and not (
            re.fullmatch(r"v?\d+", actual_volume, re.I)
            and re.fullmatch(r"v?\d+", requested_volume, re.I)
            and actual_volume.casefold().removeprefix("v") == requested_volume.casefold().removeprefix("v")
        ):
            return False
    if parameters.get("issue") is not None and str(entry.get("issue")) != str(parameters["issue"]):
        return False
    return True


def _filtered_page(entries: list[dict[str, Any]], parameters: Mapping[str, Any]) -> tuple[list[dict[str, Any]], str | None]:
    return _page([entry for entry in entries if _in_window(entry, parameters)], parameters)


def _census(entries: list[dict[str, Any]], *, raw_records: int | None = None) -> dict[str, int]:
    """Reconcile an authoritative container before local pagination."""

    raw = len(entries) if raw_records is None else raw_records
    return {
        "expected_total": len(entries),
        "parser_raw_records": raw,
        "parser_rejected_records": raw - len(entries),
    }


def _absolute(base: str, href: str) -> str:
    return urljoin(base, href)


def _authors(node: _HTMLNode | None, *, separator: str = ",") -> list[str]:
    if node is None:
        return []
    linked = [anchor.text for anchor in _nodes(node, "a") if anchor.text]
    if linked:
        return linked
    return [value.strip() for value in node.text.split(separator) if value.strip()]


def _month_number(value: str) -> int | None:
    names = {
        name.casefold(): index
        for index, name in enumerate(
            ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"),
            1,
        )
    }
    abbreviated = {name[:3].casefold(): index for name, index in names.items()}
    return names.get(value.strip().casefold()) or abbreviated.get(value.strip()[:3].casefold())


@register_venue_handler("neurips_proceedings")
def _neurips(operation: str, parameters: Mapping[str, Any], fetch: VenueFetch) -> VenueOperationResult:
    provider = "neurips_proceedings"
    _require_operation(provider, operation, {"discover"})
    if str(parameters.get("series") or "NeurIPS").casefold() != "neurips":
        raise ValueError("neurips_proceedings requires series=NeurIPS")
    year = _year(parameters)
    url = f"https://proceedings.neurips.cc/paper_files/paper/{year}"
    response = fetch(url, "neurips-proceedings-html-v1")
    root = _html(response.body, provider)
    entries: list[dict[str, Any]] = []
    paper_lists = _nodes(root, "ul", "paper-list")
    if not paper_lists:
        raise ProviderRequestError("neurips_proceedings: official page has no paper-list")
    for paper_list in paper_lists:
        for item in _nodes(paper_list, "li"):
            anchor = next(
                (
                    value
                    for value in _nodes(item, "a")
                    if re.search(r"-Abstract(?:-[A-Za-z0-9_]+)?\.html$", value.attributes.get("href", ""))
                ),
                None,
            )
            if anchor is None:
                continue
            href = anchor.attributes["href"]
            stable = re.sub(r"-Abstract(?:-[A-Za-z0-9_]+)?\.html$", "", href.rsplit("/", 1)[-1])
            landing_url = _absolute(url, href)
            attribute_title = anchor.attributes.get("title", "").strip()
            title = (
                anchor.text
                if not attribute_title or attribute_title.casefold() == "paper title"
                else attribute_title
            )
            entries.append(
                {
                    "external_id": f"NeurIPS-{year}-{stable}",
                    "title": title,
                    "authors": _authors(_first(item, "span", "paper-authors")),
                    "year": year,
                    "venue": f"NeurIPS {year}",
                    "landing_url": landing_url,
                    "pdf_url": re.sub(
                        r"-Abstract(?P<track>-[A-Za-z0-9_]+)?\.html$",
                        r"-Paper\g<track>.pdf",
                        landing_url.replace("/hash/", "/file/", 1),
                    ),
                    "publication_version": "published",
                    "host_type": "official",
                    "access_basis": "public_read_only",
                    "language": "en",
                    "document_type": "proceedings-article",
                    "track": (_first(item, "span", "paper-track-badge") or _HTMLNode("span", {})).text or "conference",
                }
            )
    if not entries:
        raise ProviderRequestError("neurips_proceedings: official page contained no conference papers")
    selected, cursor = _filtered_page(entries, parameters)
    return VenueOperationResult({"entries": selected, "next_cursor": cursor, "census": _census(entries)}, (response.body,))


@register_venue_handler("pmlr")
def _pmlr(operation: str, parameters: Mapping[str, Any], fetch: VenueFetch) -> VenueOperationResult:
    provider = "pmlr"
    _require_operation(provider, operation, {"resolve_volume", "discover"})
    if str(parameters.get("series") or "").casefold() != "icml":
        raise ValueError("pmlr conference discovery requires series=ICML")
    year = _year(parameters)
    if operation == "resolve_volume":
        url = "https://proceedings.mlr.press/"
        response = fetch(url, "pmlr-index-html-v1")
        root = _html(response.body, provider)
        matches: list[str] = []
        for item in _nodes(root, "li"):
            text = " ".join(item.text.split())
            anchor = next((value for value in _nodes(item, "a") if re.fullmatch(r"/?v\d+/?", value.attributes.get("href", ""))), None)
            main_volume = re.search(
                rf"\bProceedings of (?:the \d+(?:st|nd|rd|th) )?(?:ICML|International Conference on Machine Learning),? {year}\b",
                text,
                re.I,
            )
            if anchor and main_volume:
                matches.append(_absolute(url, anchor.attributes["href"]).rstrip("/") + "/")
        matches = list(dict.fromkeys(matches))
        if len(matches) != 1:
            raise ProviderRequestError(f"pmlr: expected one official ICML {year} volume, found {len(matches)}")
        return VenueOperationResult({"official_url": matches[0], "api_version": "pmlr-html-v1"}, (response.body,))

    volume = str(parameters.get("volume_id") or "")
    if not re.fullmatch(r"v\d+", volume):
        raise ValueError("pmlr discovery requires a resolved volume_id such as v235")
    url = f"https://proceedings.mlr.press/{volume}/"
    response = fetch(url, "pmlr-volume-html-v1")
    root = _html(response.body, provider)
    publication_date = _pmlr_publication_date(root, year)
    entries: list[dict[str, Any]] = []
    for paper in _nodes(root, "div", "paper"):
        title = _first(paper, class_name="title")
        author_node = _first(paper, class_name="authors")
        anchor = next(
            (
                value
                for value in _nodes(paper, "a")
                if value.attributes.get("href", "").rstrip("/").endswith(".html")
            ),
            None,
        )
        if title is None or anchor is None:
            continue
        landing = _absolute(url, anchor.attributes["href"])
        slug = urlsplit(landing).path.rsplit("/", 1)[-1].removesuffix(".html")
        entries.append(
            {
                "external_id": f"{volume}/{slug}",
                "title": title.text,
                "authors": _authors(author_node),
                "publication_date": publication_date,
                "year": year,
                "venue": f"ICML {year}",
                "landing_url": landing,
                "volume": volume.removeprefix("v"),
            }
        )
    if not entries:
        raise ProviderRequestError("pmlr: official volume contained no papers")
    selected, cursor = _filtered_page(entries, parameters)
    return VenueOperationResult({"entries": selected, "next_cursor": cursor, "census": _census(entries)}, (response.body,))


def _pmlr_publication_date(root: _HTMLNode, year: int) -> str:
    description = " ".join(
        node.attributes.get("content", "")
        for node in _nodes(root, "meta")
        if node.attributes.get("name", "").casefold() == "description"
        or node.attributes.get("property", "").casefold() in {"og:description", "twitter:description"}
    )
    match = re.search(r"\bon\s+(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})\b", description)
    if not match:
        return str(year)
    month = _month_number(match.group(2))
    if month is None:
        return str(year)
    return f"{int(match.group(3)):04d}-{month:02d}-{int(match.group(1)):02d}"


@register_venue_handler("acl_anthology")
def _acl(operation: str, parameters: Mapping[str, Any], fetch: VenueFetch) -> VenueOperationResult:
    provider = "acl_anthology"
    _require_operation(provider, operation, {"discover"})
    snapshot = str(parameters.get("snapshot_version") or "")
    if not re.fullmatch(r"[0-9a-f]{40}", snapshot):
        raise ValueError("acl_anthology requires a 40-character frozen snapshot_version")
    requested = {str(value) for value in parameters.get("collections", ("main", "findings", "workshop"))}
    if not requested or not requested.issubset({"main", "findings", "workshop"}):
        raise ValueError("acl_anthology collections must contain main, findings, or workshop")
    year = _year(parameters)
    url = f"https://raw.githubusercontent.com/acl-org/acl-anthology/{snapshot}/data/xml/{year}.acl.xml"
    response = fetch(url, f"acl-anthology-xml-{snapshot}")
    root = _acl_xml(response.body)
    bodies = [response.body]
    entries: list[dict[str, Any]] = []
    if "main" in requested:
        entries.extend(_acl_volume_entries(root, tuple(root.findall("./volume")), "main", year, snapshot))

    referenced: dict[str, list[str]] = {"findings": [], "workshop": []}
    event = root.find(f"./event[@id='acl-{year}']")
    if event is not None:
        for element in event.findall("./colocated/volume-id"):
            volume_id = _xml_text(element)
            if volume_id.startswith(f"{year}.findings-"):
                referenced["findings"].append(volume_id)
            elif volume_id:
                referenced["workshop"].append(volume_id)
    for collection in ("findings", "workshop"):
        if collection not in requested:
            continue
        volume_ids = referenced[collection]
        if not volume_ids:
            raise ProviderRequestError(
                f"acl_anthology: pinned ACL {year} event has no {collection} volume mapping"
            )
        by_file: dict[str, list[str]] = {}
        for volume_id in volume_ids:
            collection_id, native_volume_id = volume_id.split("-", 1)
            by_file.setdefault(collection_id, []).append(native_volume_id)
        for collection_id, native_volume_ids in by_file.items():
            collection_url = (
                f"https://raw.githubusercontent.com/acl-org/acl-anthology/{snapshot}/data/xml/{collection_id}.xml"
            )
            collection_response = fetch(collection_url, f"acl-anthology-xml-{snapshot}")
            bodies.append(collection_response.body)
            collection_root = _acl_xml(collection_response.body)
            volumes = [
                volume
                for volume in collection_root.findall("./volume")
                if volume.attrib.get("id") in native_volume_ids
            ]
            found = {str(volume.attrib.get("id")) for volume in volumes}
            if found != set(native_volume_ids):
                missing = sorted(set(native_volume_ids) - found)
                raise ProviderRequestError(
                    f"acl_anthology: pinned collection {collection_id} is missing event volumes {missing}"
                )
            entries.extend(_acl_volume_entries(collection_root, tuple(volumes), collection, year, snapshot))
    if not entries:
        raise ProviderRequestError("acl_anthology: pinned XML contained no matching papers")
    selected, cursor = _filtered_page(entries, parameters)
    return VenueOperationResult({"entries": selected, "next_cursor": cursor, "census": _census(entries)}, tuple(bodies))


def _acl_xml(body: bytes) -> ElementTree.Element:
    try:
        return ElementTree.fromstring(body)
    except ElementTree.ParseError as error:
        raise ProviderRequestError("acl_anthology: malformed official XML") from error


def _acl_volume_entries(
    root: ElementTree.Element,
    volumes: tuple[ElementTree.Element, ...],
    collection: str,
    year: int,
    snapshot: str,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for volume in volumes:
        meta = volume.find("./meta")
        if meta is None:
            continue
        volume_id = volume.attrib.get("id", "")
        booktitle = _xml_text(meta.find("./booktitle"))
        publication_date = _month_date(year, _xml_text(meta.find("./month")))
        for paper in volume.findall("./paper"):
            paper_id = paper.attrib.get("id")
            title = _xml_text(paper.find("./title"))
            if not paper_id or not title:
                continue
            stable = _xml_text(paper.find("./url")) or f"{root.attrib.get('id')}-{volume_id}.{paper_id}"
            authors = []
            for author in paper.findall("./author"):
                name = " ".join(
                    value for value in (_xml_text(author.find("./first")), _xml_text(author.find("./last"))) if value
                )
                if name:
                    authors.append(name)
            entries.append(
                {
                    "external_id": stable,
                    "title": title,
                    "authors": authors,
                    "abstract": _xml_text(paper.find("./abstract")) or None,
                    "doi": _xml_text(paper.find("./doi")) or None,
                    "publication_date": publication_date,
                    "year": year,
                    "venue": booktitle,
                    "landing_url": f"https://aclanthology.org/{stable}/",
                    "collection": collection,
                    "volume": volume_id,
                    "snapshot_version": snapshot,
                }
            )
    return entries


def _xml_text(element: ElementTree.Element | None) -> str:
    return " ".join("".join(element.itertext()).split()) if element is not None else ""


def _month_date(year: int, month: str) -> str:
    value = _month_number(month)
    return f"{year:04d}-{value:02d}" if value is not None else str(year)


@register_venue_handler("cvf_open_access")
def _cvf(operation: str, parameters: Mapping[str, Any], fetch: VenueFetch) -> VenueOperationResult:
    provider = "cvf_open_access"
    _require_operation(provider, operation, {"discover"})
    series = str(parameters.get("series") or "").upper()
    if series not in {"CVPR", "ICCV"}:
        raise ValueError("cvf_open_access requires series=CVPR or series=ICCV")
    track = str(parameters.get("track") or "")
    if track not in {"main", "workshop"}:
        raise ValueError("cvf_open_access requires track=main or track=workshop")
    if parameters.get("exclude_workshops") is True and track != "main":
        raise ValueError("cvf_open_access descriptor excludes workshops")
    year = _year(parameters)
    if track == "main":
        url = f"https://openaccess.thecvf.com/{series}{year}?day=all"
    else:
        slug = str(parameters.get("workshop_slug") or "")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", slug):
            raise ValueError("CVF workshop discovery requires workshop_slug")
        url = f"https://openaccess.thecvf.com/{series}{year}_workshops/{slug}"
    response = fetch(url, "cvf-open-access-html-v1")
    root = _html(response.body, provider)
    entries: list[dict[str, Any]] = []
    for parent in _walk(root):
        for index, node in enumerate(parent.children):
            if node.tag != "dt" or not node.has_class("ptitle"):
                continue
            anchor = _first(node, "a")
            if anchor is None or not anchor.attributes.get("href"):
                continue
            siblings = parent.children[index + 1 : index + 4]
            detail = next((value for value in siblings if value.tag == "dd"), None)
            bibref = next((value for value in siblings if value.has_class("bibref")), None)
            author_values = [
                value.attributes["value"]
                for value in (_nodes(detail, "input") if detail is not None else [])
                if value.attributes.get("name") == "query_author" and value.attributes.get("value")
            ]
            publication_date = _cvf_publication_date(bibref.text if bibref else "", year)
            landing = _absolute(url, anchor.attributes["href"])
            entries.append(
                {
                    "external_id": urlsplit(landing).path,
                    "title": anchor.text,
                    "authors": author_values or _authors(detail),
                    "publication_date": publication_date,
                    "year": year,
                    "venue": f"{series} {year}",
                    "landing_url": landing,
                }
            )
    if not entries:
        raise ProviderRequestError("cvf_open_access: official page contained no papers")
    selected, cursor = _filtered_page(entries, parameters)
    return VenueOperationResult({track: selected, "next_cursor": cursor, "census": _census(entries)}, (response.body,))


def _cvf_publication_date(bibtex: str, year: int) -> str:
    year_match = re.search(r"\byear\s*=\s*[\{\"]?(\d{4})", bibtex, re.I)
    month_match = re.search(r"\bmonth\s*=\s*[\{\"]?([A-Za-z]+)", bibtex, re.I)
    resolved_year = int(year_match.group(1)) if year_match else year
    month = _month_number(month_match.group(1)) if month_match else None
    return f"{resolved_year:04d}-{month:02d}" if month else str(resolved_year)


@register_venue_handler("ijcai_proceedings")
def _ijcai(operation: str, parameters: Mapping[str, Any], fetch: VenueFetch) -> VenueOperationResult:
    provider = "ijcai_proceedings"
    _require_operation(provider, operation, {"discover"})
    if str(parameters.get("series") or "IJCAI").upper() != "IJCAI":
        raise ValueError("ijcai_proceedings requires series=IJCAI")
    year = _year(parameters)
    url = f"https://www.ijcai.org/proceedings/{year}/"
    response = fetch(url, "ijcai-proceedings-html-v1")
    root = _html(response.body, provider)
    entries: list[dict[str, Any]] = []
    for paper in _nodes(root, "div", "paper_wrapper"):
        title = _first(paper, class_name="title")
        author_node = _first(paper, class_name="authors")
        details = _first(paper, class_name="details")
        anchor = next(
            (
                value
                for value in (_nodes(details, "a") if details is not None else [])
                if re.search(rf"/proceedings/{year}/\d+/?$", value.attributes.get("href", ""))
            ),
            None,
        )
        if title is None or anchor is None:
            continue
        landing = _absolute(url, anchor.attributes["href"])
        paper_id = urlsplit(landing).path.rstrip("/").rsplit("/", 1)[-1]
        entries.append(
            {
                "external_id": f"IJCAI-{year}-{paper_id}",
                "title": title.text,
                "authors": _authors(author_node),
                "year": year,
                "venue": f"IJCAI {year}",
                "landing_url": landing,
            }
        )
    if not entries:
        raise ProviderRequestError("ijcai_proceedings: official page contained no papers")
    selected, cursor = _filtered_page(entries, parameters)
    return VenueOperationResult({"entries": selected, "next_cursor": cursor, "census": _census(entries)}, (response.body,))


@register_venue_handler("openreview")
def _openreview(operation: str, parameters: Mapping[str, Any], fetch: VenueFetch) -> VenueOperationResult:
    provider = "openreview"
    _require_operation(provider, operation, {"resolve_invitation", "discover"})
    group = str(parameters.get("venue_group") or "")
    if operation == "resolve_invitation":
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", group):
            raise ValueError("openreview requires a valid venue_group")
        year = _year(parameters)
        venue_id = f"{group}/{year}/Conference"
        url = "https://api2.openreview.net/groups?" + urlencode({"id": venue_id})
        response = fetch(url, "openreview-api-v2-group")
        try:
            payload = json.loads(response.body)
        except json.JSONDecodeError as error:
            raise ProviderRequestError("openreview: malformed venue group response") from error
        groups = payload.get("groups") if isinstance(payload, Mapping) else None
        exact = [record for record in groups or () if isinstance(record, Mapping) and record.get("id") == venue_id]
        if len(exact) != 1:
            raise ProviderRequestError(
                f"openreview: expected one exact venue group {venue_id}, found {len(exact)}"
            )
        record = exact[0]
        content = record.get("content") if isinstance(record.get("content"), Mapping) else {}
        invitation = _openreview_value(content.get("submission_id"))
        if not isinstance(invitation, str) or "/-/" not in invitation:
            raise ProviderRequestError(
                "openreview: venue group does not publish an exact submission invitation; "
                "freeze invitation and api_version=v1 in the descriptor for a legacy venue"
            )
        api_version = "v2" if record.get("domain") else "v1"
        return VenueOperationResult(
            {
                "invitation": invitation,
                "api_version": api_version,
                "accepted_venue_ids": [venue_id],
            },
            (response.body,),
        )

    invitation = str(parameters.get("invitation") or "")
    if "/-/" not in invitation:
        raise ValueError("openreview discovery requires a resolved invitation")
    api_version = str(parameters.get("api_version") or "v2")
    offset = _offset(parameters)
    limit = _page_size(parameters)
    if api_version == "v2":
        accepted_venue_ids = [str(value) for value in parameters.get("accepted_venue_ids", ())]
        venue_id = accepted_venue_ids[0] if len(accepted_venue_ids) == 1 else invitation.split("/-/", 1)[0]
        url = "https://api2.openreview.net/notes?" + urlencode(
            {"venueid": venue_id, "limit": limit, "offset": offset}
        )
    elif api_version == "v1":
        url = "https://api.openreview.net/notes?" + urlencode(
            {"invitation": invitation, "limit": limit, "offset": offset}
        )
        venue_id = invitation.split("/-/", 1)[0]
    else:
        raise ValueError("openreview api_version must be v1 or v2")
    response = fetch(url, f"openreview-api-{api_version}")
    try:
        payload = json.loads(response.body)
    except json.JSONDecodeError as error:
        evidence = response.body[:10000].lower()
        if b"challenge" in evidence or b"cloudflare" in evidence or b"just a moment" in evidence:
            raise ProviderRequestError("openreview: public API requires challenge verification") from error
        raise ProviderRequestError("openreview: malformed API response") from error
    if not isinstance(payload, Mapping):
        raise ProviderRequestError("openreview: API response must be an object")
    error_name = str(payload.get("name") or payload.get("error") or "")
    if error_name:
        raise ProviderRequestError(f"openreview: API returned {error_name}")
    notes = payload.get("notes")
    if not isinstance(notes, list):
        raise ProviderRequestError("openreview: API response has no notes collection")
    year = _year(parameters)
    entries = [_openreview_entry(note, venue_id, year) for note in notes if isinstance(note, Mapping)]
    entries = [entry for entry in entries if _in_window(entry, parameters)]
    count = int(payload.get("count") or len(notes))
    next_cursor = str(offset + len(notes)) if notes and offset + len(notes) < count else None
    return VenueOperationResult(
        {
            "notes": entries,
            "next_cursor": next_cursor,
            "census": {
                "expected_total": count,
                "parser_raw_records": count,
                "parser_rejected_records": 0,
            },
        },
        (response.body,),
    )


def _openreview_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return value.get("value", value.get("values"))
    return value


def _openreview_entry(note: Mapping[str, Any], venue_id: str, year: int) -> dict[str, Any]:
    content = note.get("content") if isinstance(note.get("content"), Mapping) else {}
    title = _openreview_value(content.get("title") or note.get("title"))
    if not note.get("id") or not title:
        raise ProviderRequestError("openreview: note has no stable id or title")
    # ``cdate`` is commonly the submission date in the preceding calendar
    # year, not the conference publication date.  Only OpenReview's explicit
    # publication timestamp is safe for a publication-date filter.
    timestamp = note.get("pdate")
    publication_date = None
    if isinstance(timestamp, (int, float)) and timestamp > 0:
        publication_date = datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc).date().isoformat()
    authors = _openreview_value(content.get("authors") or note.get("authors")) or []
    venue = _openreview_value(content.get("venue") or note.get("venue")) or venue_id
    decision = _openreview_value(content.get("decision") or note.get("decision"))
    return {
        "external_id": str(note["id"]),
        "title": str(title),
        "authors": authors,
        "abstract": _openreview_value(content.get("abstract") or note.get("abstract")),
        "publication_date": publication_date,
        "year": year,
        "venue": venue,
        "decision": decision,
        "landing_url": f"https://openreview.net/forum?id={quote(str(note['id']), safe='')}",
        "content": dict(content),
    }


@register_venue_handler("aaai_ojs")
def _aaai(operation: str, parameters: Mapping[str, Any], fetch: VenueFetch) -> VenueOperationResult:
    provider = "aaai_ojs"
    _require_operation(provider, operation, {"discover"})
    if str(parameters.get("journal") or "AAAI").upper() != "AAAI":
        raise ValueError("aaai_ojs requires journal=AAAI")
    year = _year(parameters)
    bodies: list[bytes] = []
    issue_records = [{"id": str(value)} for value in parameters.get("issue_ids", ())]
    if not issue_records:
        archive_url = "https://ojs.aaai.org/index.php/AAAI/issue/archive"
        visited: set[str] = set()
        while archive_url and archive_url not in visited:
            visited.add(archive_url)
            response = fetch(archive_url, "aaai-ojs-archive-html-v1")
            bodies.append(response.body)
            issues, page_years, next_url = _aaai_archive(response.body, archive_url, year)
            issue_records.extend(issues)
            if page_years and max(page_years) < year:
                break
            archive_url = next_url
        if not issue_records:
            raise ProviderRequestError(f"aaai_ojs: no official AAAI issue found for {year}")
    if parameters.get("volume") is not None:
        issue_records = [
            issue for issue in issue_records if str(issue.get("volume")) == str(parameters["volume"])
        ]
    if parameters.get("issue") is not None:
        requested_issue = str(parameters["issue"])
        issue_records = [
            issue
            for issue in issue_records
            if str(issue["id"]) == requested_issue or str(issue.get("issue")) == requested_issue
        ]
    issues_payload: list[dict[str, Any]] = []
    all_articles: list[tuple[str, dict[str, Any]]] = []
    unique_issues = {str(issue["id"]): issue for issue in issue_records}
    for issue_id, issue_record in unique_issues.items():
        url = f"https://ojs.aaai.org/index.php/AAAI/issue/view/{quote(issue_id, safe='')}"
        response = fetch(url, "aaai-ojs-issue-html-v1")
        bodies.append(response.body)
        articles = _aaai_issue(response.body, url, issue_record, year)
        all_articles.extend((issue_id, article) for article in articles if _in_window(article, parameters))
    start = _offset(parameters)
    size = _page_size(parameters)
    selected = all_articles[start : start + size]
    for issue_id in dict.fromkeys(issue_id for issue_id, _ in selected):
        issues_payload.append(
            {"id": issue_id, "articles": [article for current, article in selected if current == issue_id]}
        )
    next_cursor = str(start + len(selected)) if start + len(selected) < len(all_articles) else None
    return VenueOperationResult(
        {"issues": issues_payload, "next_cursor": next_cursor, "census": _census([article for _, article in all_articles])},
        tuple(bodies),
    )


def _aaai_archive(body: bytes, url: str, year: int) -> tuple[list[dict[str, str]], set[int], str | None]:
    root = _html(body, "aaai_ojs")
    issues = []
    page_years: set[int] = set()
    for summary in _nodes(root, "div", "obj_issue_summary"):
        page_years.update(int(value) for value in re.findall(r"\b(?:19|20)\d{2}\b", summary.text))
        anchor = next(
            (value for value in _nodes(summary, "a") if "/issue/view/" in value.attributes.get("href", "")),
            None,
        )
        if anchor is None or str(year) not in summary.text:
            continue
        issue_id = urlsplit(_absolute(url, anchor.attributes["href"])).path.rstrip("/").rsplit("/", 1)[-1]
        if issue_id.isdigit():
            issue = {"id": issue_id, "title": anchor.text}
            numbering = re.search(r"\bVol\.?\s*(\d+)(?:\s+No\.?\s*(\d+))?", summary.text, re.I)
            if numbering:
                issue["volume"] = numbering.group(1)
                if numbering.group(2):
                    issue["issue"] = numbering.group(2)
            issues.append(issue)
    next_anchor = next(
        (
            value for value in _nodes(root, "a")
            if value.has_class("next") or value.attributes.get("rel", "").casefold() == "next"
        ),
        None,
    )
    next_url = _absolute(url, next_anchor.attributes["href"]) if next_anchor and next_anchor.attributes.get("href") else None
    return issues, page_years, next_url


def _aaai_issue(body: bytes, url: str, issue_record: Mapping[str, str], year: int) -> list[dict[str, Any]]:
    root = _html(body, "aaai_ojs")
    issue_id = issue_record["id"]
    published = _first(root, "div", "published")
    date_node = _first(published, class_name="value") if published is not None else None
    publication_date = date_node.text[:10] if date_node and re.match(r"\d{4}-\d{2}-\d{2}", date_node.text) else str(year)
    articles = []
    for summary in _nodes(root, "div", "obj_article_summary"):
        title_node = _first(summary, class_name="title")
        anchor = _first(title_node, "a") if title_node is not None else None
        if anchor is None or not anchor.attributes.get("href"):
            continue
        landing = _absolute(url, anchor.attributes["href"])
        match = re.search(r"/article/view/(\d+)", urlsplit(landing).path)
        if not match:
            continue
        articles.append(
            {
                "external_id": match.group(1),
                "title": anchor.text,
                "authors": _authors(_first(summary, "div", "authors")),
                "publication_date": publication_date,
                "year": year,
                "venue": f"AAAI {year}",
                "landing_url": landing,
                "volume": issue_record.get("volume"),
                "issue": issue_record.get("issue") or issue_id,
            }
        )
    if not articles:
        raise ProviderRequestError(f"aaai_ojs: issue {issue_id} contained no articles")
    return articles


@register_venue_handler("eda_proceedings")
def _eda(operation: str, parameters: Mapping[str, Any], fetch: VenueFetch) -> VenueOperationResult:
    provider = "eda_proceedings"
    _require_operation(provider, operation, {"discover"})
    series = str(parameters.get("series") or "").upper()
    if series not in {"DAC", "ICCAD"}:
        raise ValueError("eda_proceedings requires series=DAC or series=ICCAD")
    upstreams = [str(value) for value in parameters.get("upstreams", parameters.get("sources", ()))]
    if set(upstreams) != {"ieee_xplore", "acm_dl"}:
        raise ValueError("eda_proceedings requires both ieee_xplore and acm_dl upstreams")
    if parameters.get("resolve_platforms_by_year") is not True or parameters.get("deduplicate_by") != "doi":
        raise ValueError("eda_proceedings requires year-specific routes and DOI deduplication")
    year = _year(parameters)
    official_catalog = parameters.get("official_routes_by_year")
    official_route = None
    if isinstance(official_catalog, Mapping):
        official_route = official_catalog.get(str(year), official_catalog.get(year))
    if not isinstance(official_route, Mapping):
        raise ProviderRequestError(f"eda_proceedings: no frozen official {series} route for {year}")
    route_kind = str(official_route.get("route_kind") or "")
    html_route_kind = "dac_linklings_html" if series == "DAC" else "iccad_accepted_html"
    if route_kind not in {html_route_kind, "ieee_xplore_publication"}:
        raise ValueError(
            f"eda_proceedings {series} route_kind must be {html_route_kind} or ieee_xplore_publication"
        )
    evidence_url = str(
        official_route.get("evidence_url") or official_route.get("url") or ""
    )
    if urlsplit(evidence_url).scheme != "https" or not urlsplit(evidence_url).netloc:
        raise ValueError("eda_proceedings official route requires an exact HTTPS evidence_url")
    official_source = str(official_route.get("official_source") or route_kind)
    route_catalog = parameters.get("upstream_routes_by_year")
    year_routes: Mapping[str, Any] = {}
    if isinstance(route_catalog, Mapping):
        candidate = route_catalog.get(str(year), route_catalog.get(year))
        if isinstance(candidate, Mapping):
            year_routes = candidate
    unknown_upstreams = set(year_routes) - set(upstreams)
    if unknown_upstreams:
        raise ValueError(
            f"eda_proceedings has unsupported upstream routes: {', '.join(sorted(unknown_upstreams))}"
        )
    for upstream, route in year_routes.items():
        if not isinstance(route, Mapping) or type(route.get("required")) is not bool:
            raise ValueError(
                f"eda_proceedings {upstream} route for {year} requires an explicit required boolean"
            )
    if not any(bool(route["required"]) for route in year_routes.values()):
        raise ValueError(f"eda_proceedings requires at least one required upstream route for {year}")

    upstream_records: dict[str, list[dict[str, Any]]] = {}
    errors: list[str] = []
    warnings: list[str] = []
    unavailable_upstreams: list[str] = []
    bodies: list[bytes] = []
    official_catalog = route_kind == "ieee_xplore_publication"
    if official_catalog:
        official_publication_number = str(official_route.get("publication_number") or "")
        if not official_publication_number.isdigit():
            raise ValueError("eda official IEEE route requires an exact publication_number")
        ieee_route = year_routes.get("ieee_xplore")
        if isinstance(ieee_route, Mapping) and str(ieee_route.get("publication_number") or "") != official_publication_number:
            raise ValueError(
                "eda official and upstream IEEE routes must use the same publication_number"
            )
        ieee_records, ieee_bodies = _fetch_eda_upstream(
            "ieee_xplore", official_route, year, fetch
        )
        upstream_records["ieee_xplore"] = ieee_records
        bodies.extend(ieee_bodies)
        program_entries = [
            {
                **entry,
                "external_id": f"{series}-{year}-IEEE-{entry['external_id']}",
                "official_catalog_source": "ieee_xplore",
                "evidence_url": evidence_url,
            }
            for entry in ieee_records
        ]
    else:
        url = str(official_route.get("url") or "")
        if urlsplit(url).scheme != "https" or not urlsplit(url).netloc:
            raise ValueError("eda_proceedings HTML route requires an exact HTTPS URL")
        response = fetch(
            url,
            f"{route_kind.replace('_', '-')}-v1",
            f"eda_proceedings:{series.casefold()}_program",
        )
        bodies.append(response.body)
        program_entries = (
            _dac_entries(response.body, url, year)
            if series == "DAC"
            else _iccad_entries(response.body, url, year)
        )
    if not program_entries:
        raise ProviderRequestError(
            f"eda_proceedings: official {series} {year} route contained no papers"
        )
    program_entries = [entry for entry in program_entries if _in_window(entry, parameters)]

    for upstream in upstreams:
        route = year_routes.get(upstream)
        if not isinstance(route, Mapping):
            warnings.append(f"{upstream}: no frozen route for {year}")
            unavailable_upstreams.append(upstream)
            continue
        if upstream == "ieee_xplore" and official_catalog:
            continue
        try:
            records, upstream_bodies = _fetch_eda_upstream(upstream, route, year, fetch)
            bodies.extend(upstream_bodies)
            upstream_records[upstream] = records
        except (ProviderPolicyDenied, ProviderRequestError) as error:
            message = f"{upstream}: {error}"
            unavailable_upstreams.append(upstream)
            if route["required"]:
                errors.append(message)
            else:
                warnings.append(message)
    resolved: list[dict[str, Any]] = []
    doi_indexes: dict[str, int] = {}
    for program_entry in program_entries:
        matches = [
            (upstream, entry)
            for upstream, entries in upstream_records.items()
            for entry in entries
            if _title_key(str(entry.get("title") or "")) == _title_key(str(program_entry["title"]))
        ]
        official_id = str(program_entry["external_id"])
        record = {
            **program_entry,
            "official_program_id": official_id,
            "official_program_ids": [official_id],
            "provenance": {"source": official_source, "url": evidence_url},
            "field_provenance": {
                field: [{"source": official_source, "url": evidence_url}]
                for field, value in program_entry.items()
                if value not in (None, "", [], ())
            },
        }
        doi_candidates = [
            {
                "source": upstream,
                "doi": str(entry["doi"]).casefold(),
                "external_id": entry.get("external_id"),
                "landing_url": entry.get("landing_url"),
            }
            for upstream, entry in matches
            if entry.get("doi")
        ]
        distinct_dois = list(dict.fromkeys(candidate["doi"] for candidate in doi_candidates))
        sources = list(dict.fromkeys(upstream for upstream, _ in matches))
        if not distinct_dois and official_catalog and "ieee_xplore" in sources:
            record["upstream_resolution"] = {
                "status": "resolved_without_doi",
                "sources": sources,
                "doi_candidates": doi_candidates,
            }
            resolved.append(record)
            continue
        if not matches or not distinct_dois:
            record["upstream_resolution"] = {
                "status": "unresolved",
                "candidates": upstreams if not matches else sources,
                "doi_candidates": doi_candidates,
            }
            resolved.append(record)
            errors.append(f"{program_entry['external_id']}: no upstream metadata match")
            continue
        if len(distinct_dois) > 1:
            record["upstream_resolution"] = {
                "status": "conflicted",
                "sources": sources,
                "doi_candidates": doi_candidates,
            }
            resolved.append(record)
            errors.append(
                f"{official_id}: conflicting upstream DOIs {', '.join(distinct_dois)}"
            )
            continue
        doi = distinct_dois[0]
        matching_records = [
            (upstream, entry)
            for upstream, entry in matches
            if str(entry.get("doi") or "").casefold() == doi
        ]
        primary_source, primary = matching_records[0]
        record["doi"] = doi
        record["field_provenance"]["doi"] = [
            {"source": upstream, "external_id": entry.get("external_id")}
            for upstream, entry in matching_records
        ]
        for field in ("authors", "abstract", "publication_date", "year", "venue", "landing_url"):
            if record.get(field) in (None, "", [], ()) and primary.get(field) not in (None, "", [], ()):
                record[field] = primary[field]
                record["field_provenance"][field] = [
                    {"source": primary_source, "external_id": primary.get("external_id")}
                ]
        record["upstream_resolution"] = {
            "status": "resolved",
            "sources": list(dict.fromkeys(upstream for upstream, _ in matching_records)),
            "doi_candidates": doi_candidates,
        }
        if doi in doi_indexes:
            resolved[doi_indexes[doi]]["official_program_ids"].append(official_id)
            continue
        doi_indexes[doi] = len(resolved)
        resolved.append(record)
    selected, cursor = _page(resolved, parameters)
    status = "partial" if errors else "success"
    return VenueOperationResult(
        {
            "entries": selected,
            "next_cursor": cursor,
            "status": status,
            "incomplete_reasons": list(dict.fromkeys(errors)),
            "warnings": list(dict.fromkeys(warnings)),
            "unavailable_upstreams": list(dict.fromkeys(unavailable_upstreams)),
            "census": _census(resolved),
        },
        tuple(bodies),
    )


def _eda_upstream_url(
    upstream: str, route: Mapping[str, Any], year: int, *, start_record: int = 1
) -> str:
    if upstream == "ieee_xplore":
        publication_number = route.get("publication_number")
        if not str(publication_number or "").isdigit():
            raise ValueError("eda IEEE route requires an exact publication_number")
        return "https://ieeexploreapi.ieee.org/api/v1/search/articles?" + urlencode(
            {
                "publication_number": int(publication_number),
                "content_type": "Conferences",
                "start_year": year,
                "end_year": year,
                "start_record": start_record,
                "max_records": 200,
                "format": "json",
            }
        )
    if upstream == "acm_dl":
        doi = str(route.get("proceedings_doi") or "")
        if not re.fullmatch(r"10\.1145/\d+", doi):
            raise ValueError("eda ACM route requires an exact proceedings_doi")
        return f"https://dl.acm.org/doi/proceedings/{doi}"
    raise ValueError(f"unsupported EDA upstream {upstream}")


def _fetch_eda_upstream(
    upstream: str,
    route: Mapping[str, Any],
    year: int,
    fetch: VenueFetch,
) -> tuple[list[dict[str, Any]], list[bytes]]:
    bodies: list[bytes] = []
    if upstream != "ieee_xplore":
        url = _eda_upstream_url(upstream, route, year)
        response = fetch(
            url,
            f"eda-{upstream}-metadata-v1",
            f"eda_proceedings:{upstream}",
        )
        return _eda_upstream_entries(upstream, response.body, url, year), [response.body]
    records: list[dict[str, Any]] = []
    start = 1
    while True:
        url = _eda_upstream_url(upstream, route, year, start_record=start)
        response = fetch(
            url,
            "eda-ieee_xplore-metadata-v1",
            "eda_proceedings:ieee_xplore",
        )
        bodies.append(response.body)
        page, total = _eda_ieee_page(response.body, year)
        records.extend(page)
        if not page or start + len(page) > total:
            break
        start += len(page)
    if not records:
        raise ProviderRequestError("ieee_xplore returned no proceedings metadata records")
    return records, bodies


def _eda_upstream_entries(
    upstream: str, body: bytes, url: str, year: int
) -> list[dict[str, Any]]:
    if upstream == "ieee_xplore":
        entries, _ = _eda_ieee_page(body, year)
    else:
        root = _html(body, "eda_proceedings")
        entries = []
        for item in _nodes(root, class_name="issue-item"):
            anchor = next(
                (
                    value for value in _nodes(item, "a")
                    if re.search(r"/doi/(?:abs/)?10\.1145/", value.attributes.get("href", ""))
                ),
                None,
            )
            if anchor is None or not anchor.text:
                continue
            landing = _absolute(url, anchor.attributes["href"])
            doi_match = re.search(r"/doi/(?:abs/)?(10\.1145/[^?#]+)", urlsplit(landing).path)
            if not doi_match:
                continue
            abstract = _first(item, class_name="issue-item__abstract")
            entries.append(
                {
                    "external_id": doi_match.group(1),
                    "title": anchor.text,
                    "authors": [value.text for value in _nodes(item, class_name="loa__author-name") if value.text],
                    "abstract": abstract.text if abstract else None,
                    "doi": doi_match.group(1).casefold(),
                    "year": year,
                    "venue": f"ACM proceedings {year}",
                    "landing_url": landing,
                }
            )
    if not entries:
        raise ProviderRequestError(f"{upstream} returned no proceedings metadata records")
    return entries


def _eda_ieee_page(body: bytes, year: int) -> tuple[list[dict[str, Any]], int]:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as error:
        raise ProviderRequestError("IEEE Xplore returned malformed JSON") from error
    articles = payload.get("articles") if isinstance(payload, Mapping) else None
    if not isinstance(articles, list):
        raise ProviderRequestError("IEEE Xplore response has no articles collection")
    entries = [_eda_ieee_entry(article, year) for article in articles if isinstance(article, Mapping)]
    return entries, int(payload.get("totalfound") or len(entries))


def _eda_ieee_entry(article: Mapping[str, Any], year: int) -> dict[str, Any]:
    article_number = str(article.get("article_number") or "")
    if not article_number:
        raise ProviderRequestError("IEEE Xplore article is missing article_number")
    authors = article.get("authors")
    author_values = authors.get("authors", ()) if isinstance(authors, Mapping) else ()
    return {
        "external_id": article_number,
        "title": str(article.get("title") or ""),
        "authors": [
            str(value.get("full_name"))
            for value in author_values
            if isinstance(value, Mapping) and value.get("full_name")
        ],
        "abstract": article.get("abstract"),
        "doi": str(article["doi"]).casefold() if article.get("doi") else None,
        "publication_date": article.get("publication_date"),
        "year": int(article.get("publication_year") or year),
        "venue": article.get("publication_title"),
        "landing_url": article.get("abstract_url") or article.get("html_url"),
    }


def _title_key(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def _dac_entries(body: bytes, url: str, year: int) -> list[dict[str, Any]]:
    root = _html(body, "eda_proceedings")
    authors: dict[str, list[str]] = {}
    for anchor in _nodes(root, "a"):
        link_type = anchor.attributes.get("data-link-type", "")
        match = re.search(r"\.((?:RESEARCH)\d+)\.author\.person$", link_type)
        if match and anchor.text:
            authors.setdefault(match.group(1), []).append(anchor.text)
    entries = []
    for item in _nodes(root, "div", "presentation-title"):
        paper_id = item.attributes.get("ssid", "")
        if not paper_id.startswith("RESEARCH"):
            continue
        entry_type = _first(item, "div", "etype") or _first(item, "div", "etypes-list")
        if entry_type is not None and entry_type.text.casefold() != "research manuscript":
            continue
        anchor = next(
            (
                value for value in _nodes(item, "a")
                if value.attributes.get("data-link-type") == "search-page.presentation"
                or f"id={paper_id}" in value.attributes.get("href", "")
            ),
            None,
        )
        if anchor is None or not anchor.text:
            continue
        date_node = _first(item, "span", "dateTimeInfo") or _first(item, "span", "start-time")
        utc_time = date_node.attributes.get("utc_time", "") if date_node else ""
        publication_date = utc_time[:10] if re.match(r"\d{4}-\d{2}-\d{2}", utc_time) else str(year)
        abstract_node = _first(item, "span", "abstract")
        entries.append(
            {
                "external_id": f"DAC-{year}-{paper_id}",
                "title": anchor.text,
                "authors": list(dict.fromkeys(authors.get(paper_id, ()))),
                "abstract": abstract_node.text if abstract_node else None,
                "publication_date": publication_date,
                "year": year,
                "venue": f"DAC {year}",
                "landing_url": _absolute(url, anchor.attributes.get("href", "")),
            }
        )
    return entries


def _iccad_entries(body: bytes, url: str, year: int) -> list[dict[str, Any]]:
    root = _html(body, "eda_proceedings")
    entries = []
    for row in _nodes(root, "tr"):
        cells = [child for child in row.children if child.tag in {"td", "th"}]
        if len(cells) < 2:
            continue
        paper_id = cells[0].text
        title = cells[1].text
        if not paper_id.isdigit() or not title:
            continue
        entries.append(
            {
                "external_id": f"ICCAD-{year}-{paper_id}",
                "title": title,
                "authors": [],
                "year": year,
                "venue": f"ICCAD {year}",
                "landing_url": f"{url}#paper-{paper_id}",
            }
        )
    return entries
