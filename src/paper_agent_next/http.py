"""Minimal synchronous HTTP boundary for Stage 1 text collection."""

from dataclasses import dataclass
from http.client import IncompleteRead
import math
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .errors import CollectionError, InputError

__all__ = ["PrefixResponse", "HttpClient"]


@dataclass(frozen=True, slots=True)
class PrefixResponse:
    content_type: str | None
    body: bytes


class HttpClient:
    def __init__(self, contact: str, timeout: float) -> None:
        if (
            not isinstance(contact, str)
            or not contact
            or "\r" in contact
            or "\n" in contact
        ):
            raise InputError("http: contact must be a non-empty string without CR or LF")
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise InputError("http: timeout must be a finite positive number")
        self._timeout = timeout
        self._user_agent = f"paper-agent/1.0 (+{contact})"

    def get_text(self, url: str) -> str:
        try:
            request = Request(url, headers={"User-Agent": self._user_agent})
            response = urlopen(request, timeout=self._timeout)
        except HTTPError as error:
            error.close()
            raise CollectionError(f"http: GET {url} returned HTTP {error.code}") from error
        except (URLError, TimeoutError, OSError, ValueError) as error:
            raise CollectionError(f"http: GET {url} failed") from error

        try:
            try:
                body = response.read()
            except (URLError, TimeoutError, IncompleteRead, OSError) as error:
                raise CollectionError(f"http: read {url} failed") from error

            charset = response.headers.get_content_charset() or "utf-8"
            try:
                return body.decode(charset)
            except (LookupError, UnicodeError) as error:
                raise CollectionError(f"http: invalid text encoding for {url}") from error
        finally:
            response.close()

    def get_prefix(self, url: str, max_bytes: int) -> PrefixResponse:
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
            raise InputError("http: max_bytes must be a positive integer")

        try:
            request = Request(url, headers={"User-Agent": self._user_agent})
            response = urlopen(request, timeout=self._timeout)
        except HTTPError as error:
            error.close()
            raise CollectionError(f"http: GET {url} returned HTTP {error.code}") from error
        except (URLError, TimeoutError, OSError, ValueError) as error:
            raise CollectionError(f"http: GET {url} failed") from error

        try:
            try:
                body = response.read(max_bytes)
            except (URLError, TimeoutError, IncompleteRead, OSError) as error:
                raise CollectionError(f"http: read {url} failed") from error
            content_type = response.headers.get("Content-Type")
            return PrefixResponse(content_type=content_type, body=body)
        finally:
            response.close()
