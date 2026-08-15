"""Semantic Scholar DOI-batch metadata enrichment for Stage 1."""

from collections.abc import Iterable

from ..errors import ContractError, EnrichmentError
from ..normalize import normalize_doi, normalize_text
from .base import EnrichmentPatch, FrozenPaper, JsonHttpClient

__all__ = ["SemanticScholarEnricher"]


_BATCH_URL = (
    "https://api.semanticscholar.org/graph/v1/paper/batch"
    "?fields=abstract,externalIds,openAccessPdf"
)
_BATCH_SIZE = 500


def _schema_error(message: str) -> EnrichmentError:
    return EnrichmentError(f"semantic_scholar: invalid batch response ({message})")


def _batches(values: tuple[str, ...], size: int) -> Iterable[tuple[str, ...]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


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
        seen_response_dois: set[str] = set()
        for batch in _batches(tuple(requested_dois), _BATCH_SIZE):
            response = http_client.post_json(
                _BATCH_URL,
                {"ids": [f"DOI:{doi}" for doi in batch]},
            )
            if not isinstance(response, list):
                raise _schema_error("result is not a list")
            if len(response) != len(batch):
                raise _schema_error("result count does not match request count")

            for item in response:
                if item is None:
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
                    continue
                for paper in papers_by_doi[response_doi]:
                    patches.append(
                        EnrichmentPatch(
                            identity=paper.identity,
                            abstract=abstract,
                            pdf_candidates=() if pdf_url is None else (pdf_url,),
                        )
                    )

        return tuple(patches)
