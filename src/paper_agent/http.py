"""Minimal synchronous HTTP boundary for Stage 1 text collection."""

from dataclasses import dataclass
from http.client import IncompleteRead
import json
import math
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .errors import CollectionError, EnrichmentError, InputError

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

    def post_json(self, url: str, payload: object) -> object:
        try:
            body = json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, UnicodeError, ValueError) as error:
            raise EnrichmentError("http: could not encode JSON request") from error

        try:
            request = Request(
                url,
                data=body,
                headers={
                    "User-Agent": self._user_agent,
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                method="POST",
            )
            response = urlopen(request, timeout=self._timeout)
        except HTTPError as error:
            error.close()
            raise EnrichmentError(
                f"http: POST {url} returned HTTP {error.code}",
                status_code=error.code,
            ) from error
        except (URLError, TimeoutError, OSError, ValueError) as error:
            raise EnrichmentError(f"http: POST {url} failed") from error

        try:
            status = getattr(response, "status", None)
            if status is None:
                getcode = getattr(response, "getcode", None)
                status = getcode() if callable(getcode) else None
            if isinstance(status, int) and status >= 400:
                raise EnrichmentError(
                    f"http: POST {url} returned HTTP {status}",
                    status_code=status,
                )

            try:
                response_body = response.read()
            except (URLError, TimeoutError, IncompleteRead, OSError, ValueError) as error:
                raise EnrichmentError(f"http: read {url} failed") from error
            if not isinstance(response_body, bytes):
                raise EnrichmentError(f"http: read {url} returned a non-byte body")

            headers = getattr(response, "headers", None)
            charset = (
                headers.get_content_charset() or "utf-8"
                if headers is not None and hasattr(headers, "get_content_charset")
                else "utf-8"
            )
            try:
                decoded = response_body.decode(charset)
            except (LookupError, UnicodeError) as error:
                raise EnrichmentError(f"http: invalid JSON encoding for {url}") from error
            try:
                return json.loads(decoded)
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                raise EnrichmentError(f"http: invalid JSON response from {url}") from error
        finally:
            response.close()

    def get_json(self, url: str) -> object:
        try:
            request = Request(
                url,
                headers={
                    "User-Agent": self._user_agent,
                    "Accept": "application/json",
                },
                method="GET",
            )
            response = urlopen(request, timeout=self._timeout)
        except HTTPError as error:
            error.close()
            raise EnrichmentError(
                f"http: GET {url} returned HTTP {error.code}",
                status_code=error.code,
            ) from error
        except (URLError, TimeoutError, OSError, ValueError) as error:
            raise EnrichmentError(f"http: GET {url} failed") from error

        try:
            status = getattr(response, "status", None)
            if status is None:
                getcode = getattr(response, "getcode", None)
                status = getcode() if callable(getcode) else None
            if isinstance(status, int) and status >= 400:
                raise EnrichmentError(
                    f"http: GET {url} returned HTTP {status}",
                    status_code=status,
                )

            try:
                response_body = response.read()
            except (URLError, TimeoutError, IncompleteRead, OSError, ValueError) as error:
                raise EnrichmentError(f"http: read {url} failed") from error
            if not isinstance(response_body, bytes):
                raise EnrichmentError(f"http: read {url} returned a non-byte body")

            headers = getattr(response, "headers", None)
            charset = (
                headers.get_content_charset() or "utf-8"
                if headers is not None and hasattr(headers, "get_content_charset")
                else "utf-8"
            )
            try:
                decoded = response_body.decode(charset)
            except (LookupError, UnicodeError) as error:
                raise EnrichmentError(f"http: invalid JSON encoding for {url}") from error
            try:
                return json.loads(decoded)
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                raise EnrichmentError(f"http: invalid JSON response from {url}") from error
        finally:
            response.close()

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
