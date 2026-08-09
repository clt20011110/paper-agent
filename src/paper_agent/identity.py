"""Stable identifiers and conservative paper identity keys."""

from __future__ import annotations

import re
from uuid import NAMESPACE_URL, uuid5


_DOI_PREFIX = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", re.IGNORECASE)
_ARXIV_PREFIX = re.compile(r"^(?:https?://arxiv\.org/(?:abs|pdf)/|arxiv:\s*)", re.IGNORECASE)
_ARXIV_VERSION = re.compile(r"v\d+$", re.IGNORECASE)
_TITLE_PUNCTUATION = re.compile(r"[^\w\s]")


def normalize_doi(value: str | None) -> str | None:
    if not value:
        return None
    normalized = _DOI_PREFIX.sub("", value.strip()).rstrip("/.,;)").lower()
    return normalized or None


def normalize_arxiv_id(value: str | None) -> str | None:
    if not value:
        return None
    normalized = _ARXIV_PREFIX.sub("", value.strip()).removesuffix(".pdf")
    normalized = _ARXIV_VERSION.sub("", normalized).lower()
    return normalized or None


def normalize_provider(value: str) -> str:
    return value.strip().lower()


def normalize_external_id(value: str) -> str:
    return value.strip()


def provider_external_key(provider: str, external_id: str) -> str:
    return f"{normalize_provider(provider)}:{normalize_external_id(external_id)}"


def normalize_title(value: str) -> str:
    return " ".join(_TITLE_PUNCTUATION.sub(" ", value).casefold().split())


def normalize_author(value: str | None) -> str:
    return " ".join((value or "").casefold().split())


def title_author_year_key(title: str, authors: tuple[str, ...], year: int | None) -> str:
    return f"{normalize_title(title)}|{normalize_author(authors[0] if authors else None)}|{year or ''}"


def stable_identity_key(
    *,
    doi: str | None = None,
    arxiv_id: str | None = None,
    provider: str | None = None,
    external_id: str | None = None,
) -> str:
    if normalized_doi := normalize_doi(doi):
        return f"doi:{normalized_doi}"
    if normalized_arxiv := normalize_arxiv_id(arxiv_id):
        return f"arxiv:{normalized_arxiv}"
    if provider and external_id:
        return provider_external_key(provider, external_id)
    raise ValueError("a DOI, arXiv ID, or provider external ID is required")


def paper_id_for(**identity: str | None) -> str:
    return f"paper-{uuid5(NAMESPACE_URL, stable_identity_key(**identity)).hex}"


def source_id_for(provider: str, external_id: str) -> str:
    return f"source-{uuid5(NAMESPACE_URL, provider_external_key(provider, external_id)).hex}"


def manual_queue_id(queue_type: str, dedup_key: str) -> str:
    return f"manual-{uuid5(NAMESPACE_URL, f'{queue_type}:{dedup_key}').hex}"
