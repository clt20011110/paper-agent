"""Verify stable anonymous PDF candidates through a bounded response prefix."""

from dataclasses import dataclass
from urllib.parse import parse_qsl, urlsplit

from .errors import CollectionError
from .http import HttpClient
from .models import AccessStatus

__all__ = ["AccessDecision", "resolve_access"]

_PREFIX_BYTES = 4096
_BLOCKED_QUERY_KEYS = frozenset(
    {
        "token",
        "access_token",
        "auth",
        "authorization",
        "signature",
        "sig",
        "expires",
        "expiry",
        "policy",
        "key-pair-id",
        "awsaccesskeyid",
    }
)


@dataclass(frozen=True, slots=True)
class AccessDecision:
    access_status: AccessStatus | None
    pdf_url: str | None
    reason_code: str | None


def _is_requestable_candidate(candidate: object) -> bool:
    if not isinstance(candidate, str) or any(character.isspace() for character in candidate):
        return False

    try:
        parsed = urlsplit(candidate)
        hostname = parsed.hostname
        parsed.port
    except ValueError:
        return False

    if (
        parsed.scheme.casefold() not in {"http", "https"}
        or not parsed.netloc
        or hostname is None
        or parsed.username is not None
        or parsed.password is not None
    ):
        return False

    for key, _ in parse_qsl(parsed.query, keep_blank_values=True):
        normalized_key = key.casefold()
        if normalized_key in _BLOCKED_QUERY_KEYS or normalized_key.startswith(
            ("x-amz-", "x-goog-")
        ):
            return False
    return True


def resolve_access(
    pdf_candidates: tuple[str, ...],
    doi: str | None,
    http_client: HttpClient,
) -> AccessDecision:
    for candidate in pdf_candidates:
        if not _is_requestable_candidate(candidate):
            continue

        try:
            response = http_client.get_prefix(candidate, _PREFIX_BYTES)
        except CollectionError:
            continue

        content_type = response.content_type
        normalized_content_type = (
            content_type.split(";", 1)[0].strip().casefold()
            if content_type is not None
            else None
        )
        if normalized_content_type not in {"application/pdf", "application/octet-stream"}:
            continue
        if not response.body.startswith(b"%PDF-"):
            continue
        return AccessDecision(AccessStatus.DIRECT_PDF, candidate, None)

    if doi is not None:
        return AccessDecision(AccessStatus.DOI_ONLY, None, None)
    return AccessDecision(None, None, "no_verified_pdf_or_doi")
