from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..catalog import VenueSpec
from ..errors import ContractError
from ..models import Pagination

__all__ = [
    "CollectedPaper",
    "ParseReject",
    "CollectionResult",
    "TextHttpClient",
    "CorpusAdapter",
]


def _require_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{field}: must be a non-empty string")
    return value


def _require_nonnegative_int(value: object, field: str) -> None:
    if type(value) is not int or value < 0:
        raise ContractError(f"{field}: must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class CollectedPaper:
    source_id: str
    title: str | None
    authors: tuple[str, ...]
    abstract: str | None
    landing_url: str
    pdf_candidates: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_text(self.source_id, "source_id")
        if self.title is not None:
            _require_text(self.title, "title")
        if not isinstance(self.authors, tuple):
            raise ContractError("authors: must be a tuple")
        for index, author in enumerate(self.authors):
            _require_text(author, f"authors[{index}]")
        if self.abstract is not None:
            _require_text(self.abstract, "abstract")
        _require_text(self.landing_url, "landing_url")
        if not isinstance(self.pdf_candidates, tuple):
            raise ContractError("pdf_candidates: must be a tuple")
        for index, candidate in enumerate(self.pdf_candidates):
            _require_text(candidate, f"pdf_candidates[{index}]")


@dataclass(frozen=True, slots=True)
class ParseReject:
    source_locator: str
    reason_code: str
    message: str

    def __post_init__(self) -> None:
        _require_text(self.source_locator, "source_locator")
        _require_text(self.reason_code, "reason_code")
        _require_text(self.message, "message")


@dataclass(frozen=True, slots=True)
class CollectionResult:
    source_name: str
    papers: tuple[CollectedPaper, ...]
    raw_items: int
    excluded_non_papers: int
    duplicate_occurrences: int
    parse_rejects: tuple[ParseReject, ...]
    pagination: Pagination

    def __post_init__(self) -> None:
        _require_text(self.source_name, "source_name")
        if not isinstance(self.papers, tuple):
            raise ContractError("papers: must be a tuple")
        for index, paper in enumerate(self.papers):
            if not isinstance(paper, CollectedPaper):
                raise ContractError(f"papers[{index}]: must be a CollectedPaper")
        source_ids = tuple(paper.source_id for paper in self.papers)
        if len(set(source_ids)) != len(source_ids):
            raise ContractError("papers: source_id values must be unique")
        _require_nonnegative_int(self.raw_items, "raw_items")
        _require_nonnegative_int(self.excluded_non_papers, "excluded_non_papers")
        _require_nonnegative_int(self.duplicate_occurrences, "duplicate_occurrences")
        if not isinstance(self.parse_rejects, tuple):
            raise ContractError("parse_rejects: must be a tuple")
        for index, reject in enumerate(self.parse_rejects):
            if not isinstance(reject, ParseReject):
                raise ContractError(f"parse_rejects[{index}]: must be a ParseReject")
        if not isinstance(self.pagination, Pagination):
            raise ContractError("pagination: must be a Pagination")
        if self.raw_items != (
            len(self.papers)
            + self.excluded_non_papers
            + self.duplicate_occurrences
            + len(self.parse_rejects)
        ):
            raise ContractError(
                "raw_items: must equal included papers plus exclusions, duplicates, and parse rejects"
            )


class TextHttpClient(Protocol):
    def get_text(self, url: str) -> str:
        ...


class CorpusAdapter(Protocol):
    source_name: str

    def collect(
        self,
        venue_spec: VenueSpec,
        year: int,
        http_client: TextHttpClient,
    ) -> CollectionResult:
        ...
