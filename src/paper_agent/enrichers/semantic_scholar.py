"""Semantic Scholar DOI-batch metadata enrichment for Stage 1."""

from collections.abc import Iterable
from urllib.parse import urlencode

from ..errors import ContractError, EnrichmentError
from ..normalize import normalize_doi, normalize_text
from .base import EnrichmentPatch, FrozenPaper, JsonHttpClient

__all__ = ["SemanticScholarEnricher"]


_BATCH_URL = (
    "https://api.semanticscholar.org/graph/v1/paper/batch"
    "?fields=abstract,externalIds,openAccessPdf"
)
_MATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search/match"
_MATCH_FIELDS = "externalIds,abstract,openAccessPdf"
_BATCH_SIZE = 500
_DBLP_SOURCE_NAMES = frozenset({"dblp", "dblp_toc"})


def _schema_error(message: str) -> EnrichmentError:
    return EnrichmentError(f"semantic_scholar: invalid batch response ({message})")


def _match_schema_error(message: str) -> EnrichmentError:
    return EnrichmentError(
        f"semantic_scholar: invalid search/match response ({message})"
    )


def _batches(values: tuple[str, ...], size: int) -> Iterable[tuple[str, ...]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _match_url(title: str) -> str:
    return f"{_MATCH_URL}?{urlencode((('query', title), ('fields', _MATCH_FIELDS)))}"


def _match_title(paper: FrozenPaper) -> str | None:
    if paper.identity.source_name not in _DBLP_SOURCE_NAMES:
        return None
    return normalize_text(paper.title)


def _match_patch(
    paper: FrozenPaper,
    response: object,
) -> EnrichmentPatch | None:
    if not isinstance(response, dict):
        raise _match_schema_error("response is not an object")
    candidates = response.get("data")
    if not isinstance(candidates, list):
        raise _match_schema_error("data is not a list")

    matches: list[tuple[str | None, str | None]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise _match_schema_error("data item is not an object")

        external_ids = candidate.get("externalIds")
        if not isinstance(external_ids, dict):
            raise _match_schema_error("data item has no externalIds object")
        raw_dblp = external_ids.get("DBLP")
        if raw_dblp is not None and not isinstance(raw_dblp, str):
            raise _match_schema_error("data item DBLP ID has the wrong type")

        abstract_value = candidate.get("abstract")
        if abstract_value is not None and not isinstance(abstract_value, str):
            raise _match_schema_error("data item abstract has the wrong type")
        abstract = normalize_text(abstract_value)

        pdf_value = candidate.get("openAccessPdf")
        if pdf_value is not None and not isinstance(pdf_value, dict):
            raise _match_schema_error("data item openAccessPdf has the wrong type")
        pdf_url: str | None = None
        if isinstance(pdf_value, dict):
            raw_url = pdf_value.get("url")
            if raw_url is not None and not isinstance(raw_url, str):
                raise _match_schema_error(
                    "data item openAccessPdf.url has the wrong type"
                )
            if isinstance(raw_url, str) and raw_url.strip():
                pdf_url = raw_url.strip()

        if raw_dblp == paper.identity.source_id:
            matches.append((abstract, pdf_url))

    if len(matches) > 1:
        raise _match_schema_error("multiple exact DBLP ID matches")
    if not matches:
        return None

    abstract, pdf_url = matches[0]
    if abstract is None and pdf_url is None:
        return None
    return EnrichmentPatch(
        identity=paper.identity,
        abstract=abstract,
        pdf_candidates=() if pdf_url is None else (pdf_url,),
    )


class SemanticScholarEnricher:
    """Fill missing abstracts and retain unverified Semantic Scholar PDF URLs."""

    source_name = "semantic_scholar"

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

        papers_by_doi: dict[str, list[FrozenPaper]] = {}
        requested_dois: list[str] = []
        for paper in papers:
            if paper.abstract is not None or paper.doi is None:
                continue
            if paper.doi not in papers_by_doi:
                requested_dois.append(paper.doi)
                papers_by_doi[paper.doi] = []
            else:
                raise EnrichmentError("semantic_scholar: duplicate requested DOI")
            papers_by_doi[paper.doi].append(paper)

        if not requested_dois:
            return ()

        patches: list[EnrichmentPatch] = []
        residual_papers: list[FrozenPaper] = []
        seen_response_dois: set[str] = set()
        for batch in _batches(tuple(requested_dois), _BATCH_SIZE):
            try:
                response = http_client.post_json(
                    _BATCH_URL,
                    {"ids": [f"DOI:{doi}" for doi in batch]},
                )
            except EnrichmentError as error:
                if error.status_code != 400 or not all(
                    _match_title(paper) is not None
                    for doi in batch
                    for paper in papers_by_doi[doi]
                ):
                    raise
                response = [None] * len(batch)
            if not isinstance(response, list):
                raise _schema_error("result is not a list")
            if len(response) != len(batch):
                raise _schema_error("result count does not match request count")

            for batch_index, item in enumerate(response):
                if item is None:
                    residual_papers.extend(papers_by_doi[batch[batch_index]])
                    continue
                if not isinstance(item, dict):
                    raise _schema_error("result item is not an object or null")

                external_ids = item.get("externalIds")
                if not isinstance(external_ids, dict):
                    raise _schema_error("result item has no externalIds object")
                raw_response_doi = external_ids.get("DOI")
                if not isinstance(raw_response_doi, str):
                    raise _schema_error("result DOI does not match a requested DOI")
                response_doi = normalize_doi(raw_response_doi)
                if response_doi is None or response_doi not in papers_by_doi:
                    raise _schema_error("result DOI does not match a requested DOI")
                if response_doi in seen_response_dois:
                    raise _schema_error("result contains a duplicate DOI")
                seen_response_dois.add(response_doi)

                abstract_value = item.get("abstract")
                if abstract_value is not None and not isinstance(abstract_value, str):
                    raise _schema_error("abstract has the wrong type")
                abstract = normalize_text(abstract_value)

                pdf_value = item.get("openAccessPdf")
                if pdf_value is not None and not isinstance(pdf_value, dict):
                    raise _schema_error("openAccessPdf has the wrong type")
                pdf_url: str | None = None
                if isinstance(pdf_value, dict):
                    raw_url = pdf_value.get("url")
                    if raw_url is not None and not isinstance(raw_url, str):
                        raise _schema_error("openAccessPdf.url has the wrong type")
                    if isinstance(raw_url, str) and raw_url.strip():
                        pdf_url = raw_url.strip()

                if abstract is None and pdf_url is None:
                    residual_papers.extend(papers_by_doi[response_doi])
                    continue
                for paper in papers_by_doi[response_doi]:
                    patches.append(
                        EnrichmentPatch(
                            identity=paper.identity,
                            abstract=abstract,
                            pdf_candidates=() if pdf_url is None else (pdf_url,),
                        )
                    )

        for paper in residual_papers:
            title = _match_title(paper)
            if title is None:
                continue
            response = http_client.get_json(_match_url(title))
            patch = _match_patch(paper, response)
            if patch is not None:
                patches.append(patch)

        return tuple(patches)
