"""Immutable value objects for the Stage 1 paper and issue contracts."""

from dataclasses import dataclass
from enum import StrEnum
from re import fullmatch
from typing import ClassVar
from unicodedata import is_normalized
from urllib.parse import urlsplit

from .errors import ContractError

__all__ = [
    "SCHEMA_VERSION",
    "VenueType",
    "AccessStatus",
    "IssueKind",
    "MissingField",
    "SourceIdentity",
    "FieldSources",
    "PaperRecord",
    "IssueRecord",
]

SCHEMA_VERSION = 1


class VenueType(StrEnum):
    CONFERENCE = "conference"
    JOURNAL = "journal"


class AccessStatus(StrEnum):
    DIRECT_PDF = "direct_pdf"
    DOI_ONLY = "doi_only"


class IssueKind(StrEnum):
    INCOMPLETE_PAPER = "incomplete_paper"
    PARSE_REJECT = "parse_reject"
    IDENTITY_CONFLICT = "identity_conflict"
    FIELD_CONFLICT = "field_conflict"


class MissingField(StrEnum):
    TITLE = "title"
    AUTHORS = "authors"
    ABSTRACT = "abstract"
    ACCESS_LOCATOR = "access_locator"


def _require_text(value: object, field: str, *, allow_none: bool = False) -> str | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, str):
        raise ContractError(f"{field}: must be a string")
    if not value or value != value.strip():
        raise ContractError(f"{field}: must be non-empty and have no surrounding whitespace")
    return value


def _require_year(value: object, field: str = "year") -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 1000 <= value <= 9999:
        raise ContractError(f"{field}: must be a four-digit integer")


def _require_nfc(value: str, field: str) -> None:
    if not is_normalized("NFC", value):
        raise ContractError(f"{field}: must already be Unicode NFC")


def _validate_authors(value: object, field: str, *, allow_empty: bool) -> None:
    if not isinstance(value, tuple):
        raise ContractError(f"{field}: must be a tuple")
    if not value and not allow_empty:
        raise ContractError(f"{field}: must contain at least one author")
    for index, author in enumerate(value):
        author_field = f"{field}[{index}]"
        _require_text(author, author_field)
        _require_nfc(author, author_field)


def _validate_optional_doi(value: object, field: str = "doi") -> None:
    doi = _require_text(value, field, allow_none=True)
    if doi is None:
        return
    if any(character.isspace() for character in doi):
        raise ContractError(f"{field}: must not contain whitespace")
    if doi != doi.lower():
        raise ContractError(f"{field}: must be lowercase")
    if doi.startswith(("https://doi.org/", "http://doi.org/", "http://dx.doi.org/", "doi:")):
        raise ContractError(f"{field}: must be a bare DOI")
    if not doi.startswith("10."):
        raise ContractError(f"{field}: must start with 10.")
    separator = doi.find("/", 3)
    if separator <= 3 or separator == len(doi) - 1:
        raise ContractError(f"{field}: must contain a slash followed by content")


def _validate_optional_url(value: object, field: str) -> None:
    url = _require_text(value, field, allow_none=True)
    if url is None:
        return
    if any(character.isspace() for character in url):
        raise ContractError(f"{field}: must not contain whitespace")
    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname
        parsed.port
    except ValueError as error:
        raise ContractError(f"{field}: must be a valid absolute HTTP or HTTPS URL") from error
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or hostname is None:
        raise ContractError(f"{field}: must be a valid absolute HTTP or HTTPS URL")
    if parsed.username is not None or parsed.password is not None:
        raise ContractError(f"{field}: must not contain username or password")


@dataclass(frozen=True, slots=True)
class SourceIdentity:
    venue_id: str
    year: int
    source_name: str
    source_id: str

    def __post_init__(self) -> None:
        _require_text(self.venue_id, "venue_id")
        _require_year(self.year)
        _require_text(self.source_name, "source_name")
        _require_text(self.source_id, "source_id")

    def as_tuple(self) -> tuple[str, int, str, str]:
        return self.venue_id, self.year, self.source_name, self.source_id


@dataclass(frozen=True, slots=True)
class FieldSources:
    title: str
    authors: str
    abstract: str
    doi: str | None
    landing_url: str | None
    pdf_url: str | None

    def __post_init__(self) -> None:
        _require_text(self.title, "title")
        _require_text(self.authors, "authors")
        _require_text(self.abstract, "abstract")
        _require_text(self.doi, "doi", allow_none=True)
        _require_text(self.landing_url, "landing_url", allow_none=True)
        _require_text(self.pdf_url, "pdf_url", allow_none=True)

    def to_dict(self) -> dict[str, str | None]:
        return {
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "doi": self.doi,
            "landing_url": self.landing_url,
            "pdf_url": self.pdf_url,
        }


@dataclass(frozen=True, slots=True)
class PaperRecord:
    venue_id: str
    venue_name: str
    venue_type: VenueType
    year: int
    source_name: str
    source_id: str
    title: str
    authors: tuple[str, ...]
    abstract: str
    doi: str | None
    landing_url: str | None
    pdf_url: str | None
    access_status: AccessStatus
    field_sources: FieldSources

    schema_version: ClassVar[int] = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_text(self.venue_id, "venue_id")
        _require_text(self.venue_name, "venue_name")
        if not isinstance(self.venue_type, VenueType):
            raise ContractError("venue_type: must be a VenueType")
        _require_year(self.year)
        _require_text(self.source_name, "source_name")
        _require_text(self.source_id, "source_id")
        _require_text(self.title, "title")
        _require_nfc(self.title, "title")
        _validate_authors(self.authors, "authors", allow_empty=False)
        _require_text(self.abstract, "abstract")
        _require_nfc(self.abstract, "abstract")
        _validate_optional_doi(self.doi)
        _validate_optional_url(self.landing_url, "landing_url")
        _validate_optional_url(self.pdf_url, "pdf_url")
        if not isinstance(self.access_status, AccessStatus):
            raise ContractError("access_status: must be an AccessStatus")
        if not isinstance(self.field_sources, FieldSources):
            raise ContractError("field_sources: must be a FieldSources")
        if self.access_status is AccessStatus.DIRECT_PDF and self.pdf_url is None:
            raise ContractError("pdf_url: required for direct_pdf")
        if self.access_status is AccessStatus.DOI_ONLY and self.pdf_url is not None:
            raise ContractError("pdf_url: must be null for doi_only")
        if self.access_status is AccessStatus.DOI_ONLY and self.doi is None:
            raise ContractError("doi: required for doi_only")
        if self.pdf_url is not None and self.access_status is not AccessStatus.DIRECT_PDF:
            raise ContractError("access_status: pdf_url requires direct_pdf")
        if self.doi is None and self.pdf_url is None:
            raise ContractError("doi: DOI or pdf_url is required")
        if (self.doi is None) != (self.field_sources.doi is None):
            raise ContractError("field_sources.doi: nullness must match doi")
        if (self.landing_url is None) != (self.field_sources.landing_url is None):
            raise ContractError("field_sources.landing_url: nullness must match landing_url")
        if (self.pdf_url is None) != (self.field_sources.pdf_url is None):
            raise ContractError("field_sources.pdf_url: nullness must match pdf_url")

    @property
    def identity(self) -> SourceIdentity:
        return SourceIdentity(self.venue_id, self.year, self.source_name, self.source_id)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "venue_id": self.venue_id,
            "venue_name": self.venue_name,
            "venue_type": self.venue_type.value,
            "year": self.year,
            "source_name": self.source_name,
            "source_id": self.source_id,
            "title": self.title,
            "authors": list(self.authors),
            "abstract": self.abstract,
            "doi": self.doi,
            "landing_url": self.landing_url,
            "pdf_url": self.pdf_url,
            "access_status": self.access_status.value,
            "field_sources": self.field_sources.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class IssueRecord:
    issue_kind: IssueKind
    venue_id: str
    year: int
    source_name: str | None
    source_id: str | None
    source_locator: str | None
    title: str | None
    authors: tuple[str, ...]
    abstract: str | None
    doi: str | None
    landing_url: str | None
    missing_fields: tuple[MissingField, ...]
    reason_codes: tuple[str, ...]
    message: str

    schema_version: ClassVar[int] = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.issue_kind, IssueKind):
            raise ContractError("issue_kind: must be an IssueKind")
        _require_text(self.venue_id, "venue_id")
        _require_year(self.year)
        _require_text(self.source_name, "source_name", allow_none=True)
        _require_text(self.source_id, "source_id", allow_none=True)
        _require_text(self.source_locator, "source_locator", allow_none=True)
        title = _require_text(self.title, "title", allow_none=True)
        if title is not None:
            _require_nfc(title, "title")
        _validate_authors(self.authors, "authors", allow_empty=True)
        abstract = _require_text(self.abstract, "abstract", allow_none=True)
        if abstract is not None:
            _require_nfc(abstract, "abstract")
        _validate_optional_doi(self.doi)
        _validate_optional_url(self.landing_url, "landing_url")
        if self.source_id is not None and self.source_name is None:
            raise ContractError("source_name: required when source_id is present")
        if not isinstance(self.missing_fields, tuple):
            raise ContractError("missing_fields: must be a tuple")
        if any(not isinstance(field, MissingField) for field in self.missing_fields):
            raise ContractError("missing_fields: every item must be a MissingField")
        if len(set(self.missing_fields)) != len(self.missing_fields):
            raise ContractError("missing_fields: duplicate fields are not allowed")
        if not isinstance(self.reason_codes, tuple):
            raise ContractError("reason_codes: must be a tuple")
        if not self.reason_codes:
            raise ContractError("reason_codes: must not be empty")
        for index, reason_code in enumerate(self.reason_codes):
            field = f"reason_codes[{index}]"
            _require_text(reason_code, field)
            if fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)*", reason_code) is None:
                raise ContractError(f"{field}: must be snake_case")
        if len(set(self.reason_codes)) != len(self.reason_codes):
            raise ContractError("reason_codes: duplicate codes are not allowed")
        _require_text(self.message, "message")
        if self.issue_kind is IssueKind.INCOMPLETE_PAPER and not self.missing_fields:
            raise ContractError("missing_fields: required for incomplete_paper")

    @property
    def blocks_membership(self) -> bool:
        return self.issue_kind in {IssueKind.PARSE_REJECT, IssueKind.IDENTITY_CONFLICT}

    @property
    def blocks_metadata(self) -> bool:
        return self.issue_kind in {IssueKind.INCOMPLETE_PAPER, IssueKind.FIELD_CONFLICT}

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "issue_kind": self.issue_kind.value,
            "venue_id": self.venue_id,
            "year": self.year,
            "source_name": self.source_name,
            "source_id": self.source_id,
            "source_locator": self.source_locator,
            "title": self.title,
            "authors": list(self.authors),
            "abstract": self.abstract,
            "doi": self.doi,
            "landing_url": self.landing_url,
            "missing_fields": [field.value for field in self.missing_fields],
            "reason_codes": list(self.reason_codes),
            "message": self.message,
        }
