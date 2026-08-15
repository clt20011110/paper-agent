"""Small immutable value objects shared by Stage 1 metadata enrichers."""

from dataclasses import dataclass
from typing import Protocol

from ..errors import ContractError
from ..models import SourceIdentity

__all__ = [
    "FrozenPaper",
    "EnrichmentPatch",
    "JsonHttpClient",
    "MetadataEnricher",
]


def _require_text(value: object, field: str, *, allow_none: bool = False) -> str | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, str) or not value or value != value.strip():
        raise ContractError(f"{field}: must be a non-empty string without surrounding whitespace")
    return value


def _require_text_tuple(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise ContractError(f"{field}: must be a tuple")
    for index, item in enumerate(value):
        _require_text(item, f"{field}[{index}]")
    return value


@dataclass(frozen=True, slots=True)
class FrozenPaper:
    """The normalized, membership-frozen view exposed to an enricher."""

    identity: SourceIdentity
    title: str | None
    authors: tuple[str, ...]
    abstract: str | None
    doi: str | None
    landing_url: str | None
    pdf_candidates: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.identity, SourceIdentity):
            raise ContractError("identity: must be a SourceIdentity")
        _require_text(self.title, "title", allow_none=True)
        _require_text_tuple(self.authors, "authors")
        _require_text(self.abstract, "abstract", allow_none=True)
        _require_text(self.doi, "doi", allow_none=True)
        _require_text(self.landing_url, "landing_url", allow_none=True)
        _require_text_tuple(self.pdf_candidates, "pdf_candidates")


@dataclass(frozen=True, slots=True)
class EnrichmentPatch:
    """A non-empty patch for exactly one frozen source identity."""

    identity: SourceIdentity
    abstract: str | None = None
    pdf_candidates: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.identity, SourceIdentity):
            raise ContractError("identity: must be a SourceIdentity")
        _require_text(self.abstract, "abstract", allow_none=True)
        _require_text_tuple(self.pdf_candidates, "pdf_candidates")
        if self.abstract is None and not self.pdf_candidates:
            raise ContractError("patch: must contain an abstract or pdf candidate")


class JsonHttpClient(Protocol):
    def post_json(self, url: str, payload: object) -> object:
        ...


class MetadataEnricher(Protocol):
    source_name: str

    def enrich(
        self,
        papers: tuple[FrozenPaper, ...],
        http_client: JsonHttpClient,
    ) -> tuple[EnrichmentPatch, ...]:
        ...
