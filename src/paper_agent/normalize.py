"""Pure normalization helpers for the standalone Stage 1 package."""

from html.parser import HTMLParser
from unicodedata import category, normalize as normalize_unicode
from urllib.parse import urlsplit

from paper_agent.errors import ContractError

__all__ = ["normalize_doi", "normalize_text"]

_BLOCK_TAGS = ("p", "div", "li")
_IGNORED_TAGS = ("script", "style")
_VOID_TAGS = (
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "source",
    "track",
    "wbr",
)
_ZERO_WIDTH_NOISE = "\u200b\ufeff"


class _TextParser(HTMLParser):
    """Collect visible text while dropping markup and raw script/style data."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._ignored_tag: str | None = None
        self._open_tags: list[tuple[str, int]] = []

    def handle_data(self, data: str) -> None:
        if self._ignored_tag is None:
            self._parts.append(data)

    def handle_starttag(self, tag: str, attrs) -> None:
        if self._ignored_tag is not None:
            return

        local_name = tag.rsplit(":", 1)[-1].lower()
        if local_name in _IGNORED_TAGS:
            self._ignored_tag = local_name
        elif local_name in _VOID_TAGS:
            if local_name in ("br", "hr"):
                self._parts.append(" ")
        elif local_name in _BLOCK_TAGS:
            self._parts.append(" ")
        else:
            start_index = len(self._parts)
            self._parts.append(self.get_starttag_text() or f"<{tag}>")
            self._open_tags.append((tag, start_index))

    def handle_endtag(self, tag: str) -> None:
        local_name = tag.rsplit(":", 1)[-1].lower()
        if self._ignored_tag is not None:
            if local_name == self._ignored_tag:
                self._ignored_tag = None
            return

        if local_name in _BLOCK_TAGS:
            self._parts.append(" ")
            return

        for frame_index in range(len(self._open_tags) - 1, -1, -1):
            open_tag, start_index = self._open_tags[frame_index]
            if open_tag != tag:
                continue

            del self._open_tags[frame_index]
            self._parts[start_index] = ""
            return


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


def normalize_text(value: str | None) -> str | None:
    """Return visible, NFC-normalized text, or ``None`` if no text remains."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ContractError("text must be a string or None")

    parser = _TextParser()
    parser.feed(value)
    parser.close()

    cleaned: list[str] = []
    pending_space = False
    for character in "".join(parser._parts):
        if character.isspace() or category(character) == "Cc" or character in _ZERO_WIDTH_NOISE:
            pending_space = bool(cleaned)
            continue
        if pending_space:
            cleaned.append(" ")
        cleaned.append(character)
        pending_space = False

    normalized = normalize_unicode("NFC", "".join(cleaned))
    return normalized or None
