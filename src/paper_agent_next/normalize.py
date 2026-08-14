"""Pure normalization helpers for the standalone Stage 1 package."""

from urllib.parse import urlsplit

from paper_agent_next.errors import ContractError

__all__ = ["normalize_doi"]


def normalize_doi(value: str | None) -> str | None:
    """Return a lowercase bare DOI, or ``None`` for an invalid value."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ContractError("DOI must be a string or None")

    candidate = value.strip()
    if not candidate:
        return None

    lowered = candidate.lower()
    if lowered.startswith("doi:"):
        candidate = candidate[4:].strip()
    elif lowered.startswith(("http://", "https://")):
        try:
            parsed = urlsplit(candidate)
            hostname = parsed.hostname
        except ValueError:
            return None

        if hostname not in {"doi.org", "dx.doi.org"}:
            return None

        path_start = candidate.find("/", candidate.find("://") + 3)
        path_end = len(candidate)
        for delimiter in ("?", "#"):
            delimiter_index = candidate.find(delimiter, path_start)
            if delimiter_index != -1:
                path_end = min(path_end, delimiter_index)
        raw_path = candidate[path_start:path_end] if path_start != -1 else ""
        raw_candidate = raw_path[1:].strip() if raw_path.startswith("/") else ""
        if any(character.isspace() for character in raw_candidate):
            return None

        if not parsed.path.startswith("/"):
            return None
        candidate = parsed.path[1:].strip()

    candidate = candidate.lower()
    if any(character.isspace() for character in candidate):
        return None
    if not candidate.startswith("10."):
        return None

    separator = candidate.find("/")
    if separator <= len("10.") or separator == len(candidate) - 1:
        return None

    return candidate
