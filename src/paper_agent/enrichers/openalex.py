"""Strict residual metadata enrichment through the OpenAlex works API."""

from dataclasses import dataclass
from urllib.parse import quote, urlencode

from ..errors import ContractError, EnrichmentError
from ..normalize import normalize_doi, normalize_text
from .base import EnrichmentPatch, FrozenPaper, JsonHttpClient

__all__ = ["OpenAlexEnricher"]


_WORKS_URL = "https://api.openalex.org/works"
_SELECT = (
    "id,doi,display_name,publication_year,authorships,"
    "abstract_inverted_index,best_oa_location,primary_location"
)


@dataclass(frozen=True, slots=True)
class _Work:
    doi: str | None
    display_name: str | None
    publication_year: int | None
    first_author_names: tuple[str, ...]
    abstract: str | None
    pdf_candidates: tuple[str, ...]


def _schema_error(message: str) -> EnrichmentError:
    return EnrichmentError(f"openalex: invalid works response ({message})")


def _works_url(parameters: tuple[tuple[str, str], ...]) -> str:
    return f"{_WORKS_URL}?{urlencode(parameters)}"


def _singleton_url(doi: str) -> str:
    return f"{_WORKS_URL}/doi:{quote(doi, safe='/')}?{urlencode((('select', _SELECT),))}"


def _is_not_found(error: EnrichmentError) -> bool:
    return error.status_code == 404


def _response_results(
    response: object,
    *,
    truncated_is_error: bool,
) -> tuple[object, ...] | None:
    if not isinstance(response, dict):
        raise _schema_error("response is not an object")
    meta = response.get("meta")
    if not isinstance(meta, dict):
        raise _schema_error("meta is not an object")
    count = meta.get("count")
    if type(count) is not int or count < 0:
        raise _schema_error("meta.count is not a non-negative integer")
    results = response.get("results")
    if not isinstance(results, list):
        raise _schema_error("results is not a list")
    if count < len(results):
        raise _schema_error("meta.count is smaller than results")
    if count > len(results):
        if truncated_is_error:
            raise _schema_error("DOI batch results are truncated")
        return None
    return tuple(results)


def _abstract_from_index(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, dict) or not value:
        raise _schema_error("abstract_inverted_index is malformed")

    words_by_position: dict[int, str] = {}
    for word, positions in value.items():
        if not isinstance(word, str) or not word:
            raise _schema_error("abstract_inverted_index contains an invalid word")
        if not isinstance(positions, list) or not positions:
            raise _schema_error("abstract_inverted_index contains invalid positions")
        for position in positions:
            if type(position) is not int or position < 0:
                raise _schema_error("abstract_inverted_index contains an invalid position")
            if position in words_by_position:
                raise _schema_error("abstract_inverted_index contains duplicate positions")
            words_by_position[position] = word

    return normalize_text(
        " ".join(words_by_position[position] for position in sorted(words_by_position))
    )


def _location_pdf(value: object, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise _schema_error(f"{field} is not an object or null")
    raw_url = value.get("pdf_url")
    if raw_url is None:
        return None
    if not isinstance(raw_url, str):
        raise _schema_error(f"{field}.pdf_url is not a string or null")
    stripped = raw_url.strip()
    return stripped or None


def _work_from_result(value: object) -> _Work:
    if not isinstance(value, dict):
        raise _schema_error("result item is not an object")

    raw_doi = value.get("doi")
    if raw_doi is None:
        doi = None
    elif isinstance(raw_doi, str):
        doi = normalize_doi(raw_doi)
        if doi is None:
            raise _schema_error("result DOI is invalid")
    else:
        raise _schema_error("result DOI is not a string or null")

    raw_display_name = value.get("display_name")
    if raw_display_name is not None and not isinstance(raw_display_name, str):
        raise _schema_error("result display_name is not a string or null")
    display_name = normalize_text(raw_display_name)

    raw_year = value.get("publication_year")
    if raw_year is not None and (type(raw_year) is not int or raw_year < 0):
        raise _schema_error("result publication_year is invalid")

    raw_authorships = value.get("authorships")
    if raw_authorships is None:
        authorships: list[object] = []
    elif isinstance(raw_authorships, list):
        authorships = raw_authorships
    else:
        raise _schema_error("result authorships is not a list or null")

    first_author_names: list[str] = []
    for authorship in authorships:
        if not isinstance(authorship, dict):
            raise _schema_error("result authorship is not an object")
        author_position = authorship.get("author_position")
        if not isinstance(author_position, str):
            raise _schema_error("result author_position is not a string")
        if "raw_author_name" not in authorship:
            raise _schema_error("result authorship raw_author_name is missing")
        raw_author_name = authorship["raw_author_name"]
        if not isinstance(raw_author_name, str):
            raise _schema_error("result authorship raw_author_name is not a string")
        author_name = normalize_text(raw_author_name)
        if author_name is None:
            raise _schema_error("result authorship raw_author_name is empty")
        if author_position == "first":
            first_author_names.append(author_name)

    abstract = _abstract_from_index(value.get("abstract_inverted_index"))
    candidates: list[str] = []
    for field, location in (
        ("best_oa_location", value.get("best_oa_location")),
        ("primary_location", value.get("primary_location")),
    ):
        pdf_url = _location_pdf(location, field)
        if pdf_url is not None and pdf_url not in candidates:
            candidates.append(pdf_url)

    return _Work(
        doi=doi,
        display_name=display_name,
        publication_year=raw_year,
        first_author_names=tuple(first_author_names),
        abstract=abstract,
        pdf_candidates=tuple(candidates),
    )


def _patch_for_work(
    paper: FrozenPaper,
    work: _Work,
    *,
    include_doi: bool,
) -> EnrichmentPatch | None:
    doi = work.doi if include_doi else None
    if work.abstract is None and doi is None and not work.pdf_candidates:
        return None
    return EnrichmentPatch(
        identity=paper.identity,
        abstract=work.abstract,
        doi=doi,
        pdf_candidates=work.pdf_candidates,
    )


def _strict_key(paper: FrozenPaper) -> tuple[str, int, str] | None:
    if paper.title is None or not paper.authors:
        return None
    title = normalize_text(paper.title)
    first_author = normalize_text(paper.authors[0])
    if title is None or first_author is None:
        return None
    return title, paper.identity.year, first_author


def _strict_match(paper: FrozenPaper, work: _Work) -> bool:
    title = normalize_text(paper.title)
    first_author = normalize_text(paper.authors[0]) if paper.authors else None
    return (
        work.display_name == title
        and work.publication_year == paper.identity.year
        and len(work.first_author_names) == 1
        and work.first_author_names[0] == first_author
    )


class OpenAlexEnricher:
    """Fill residual metadata only after exact local OpenAlex validation."""

    source_name = "openalex"

    def enrich(
        self,
        papers: tuple[FrozenPaper, ...],
        http_client: JsonHttpClient,
    ) -> tuple[EnrichmentPatch, ...]:
        if not isinstance(papers, tuple):
            raise ContractError("papers: must be a tuple")
        for index, paper in enumerate(papers):
            if not isinstance(paper, FrozenPaper):
                raise ContractError(f"papers[{index}]: must be a FrozenPaper")

        frozen_doi_counts: dict[str, int] = {}
        normalized_dois: list[str | None] = []
        for paper in papers:
            if paper.doi is None:
                normalized_dois.append(None)
                continue
            normalized_doi = normalize_doi(paper.doi)
            if normalized_doi is None:
                raise EnrichmentError("openalex: frozen paper DOI is invalid")
            normalized_dois.append(normalized_doi)
            frozen_doi_counts[normalized_doi] = (
                frozen_doi_counts.get(normalized_doi, 0) + 1
            )

        patches: list[EnrichmentPatch] = []
        requested_dois: list[str] = []
        paper_by_doi: dict[str, FrozenPaper] = {}
        for paper, normalized_doi in zip(papers, normalized_dois):
            if (
                normalized_doi is None
                or paper.abstract is not None
                or frozen_doi_counts[normalized_doi] != 1
            ):
                continue
            requested_dois.append(normalized_doi)
            paper_by_doi[normalized_doi] = paper

        for normalized_doi in requested_dois:
            try:
                response = http_client.get_json(_singleton_url(normalized_doi))
            except EnrichmentError as error:
                if _is_not_found(error):
                    continue
                raise

            work = _work_from_result(response)
            if work.doi != normalized_doi:
                raise _schema_error("singleton result DOI does not match the request")
            patch = _patch_for_work(
                paper_by_doi[normalized_doi], work, include_doi=False
            )
            if patch is not None:
                patches.append(patch)

        strict_keys: list[tuple[str, int, str] | None] = [
            _strict_key(paper) for paper in papers
        ]
        strict_key_counts: dict[tuple[str, int, str], int] = {}
        for key in strict_keys:
            if key is not None:
                strict_key_counts[key] = strict_key_counts.get(key, 0) + 1

        frozen_dois = frozenset(frozen_doi_counts)
        for paper, normalized_doi, key in zip(papers, normalized_dois, strict_keys):
            if (
                normalized_doi is not None
                or key is None
                or strict_key_counts[key] != 1
            ):
                continue

            url = _works_url(
                (
                    ("search.exact", key[0]),
                    ("filter", f"publication_year:{key[1]}"),
                    ("per_page", "100"),
                    ("select", _SELECT),
                )
            )
            response = http_client.get_json(url)
            results = _response_results(response, truncated_is_error=False)
            if results is None:
                continue

            matches: list[_Work] = []
            for result in results:
                work = _work_from_result(result)
                if _strict_match(paper, work):
                    matches.append(work)
            if len(matches) != 1:
                continue

            work = matches[0]
            if work.doi is not None and work.doi in frozen_dois:
                continue
            patch = _patch_for_work(paper, work, include_doi=True)
            if patch is not None:
                patches.append(patch)

        return tuple(patches)
