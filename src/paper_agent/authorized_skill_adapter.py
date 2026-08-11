"""Audited queue handoff for the optional authorized publisher skill.

The adapter never drives a browser or reads browser session state.  It prepares
the skill's immutable CSV queue, exposes the next small visible-browser batch,
and imports only a finalized ``article.pdf`` recorded by the skill ledger.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import csv
from dataclasses import dataclass
import errno
from hashlib import sha256
from io import StringIO
import json
import os
from pathlib import Path
import re
import secrets
import stat
import subprocess
import sys
import tempfile
from typing import Any
from urllib.parse import unquote, urlsplit

from .authorized_skill_runtime import AuthorizedSkillDoctorResult
from .domain import AccessLocationCandidate, DownloadResult, FetchDecision, FetchDecisionStatus, FetchRequest
from .download_providers import FetchContext, ProbeContext
from .downloads import DownloadService, HTTPResponse


FINAL_SKILL_STATUSES = frozenset({"complete", "complete_no_si"})
AUTHORIZED_PUBLISHER_HOSTS = {
    "wiley": frozenset({"onlinelibrary.wiley.com"}),
    "nature": frozenset({"nature.com", "www.nature.com"}),
    "acs": frozenset({"pubs.acs.org"}),
}


class AuthorizedSkillAdapterError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SkillQueueItem:
    paper_id: str
    doi: str
    url: str
    title: str = ""
    candidate_url: str | None = None


@dataclass(frozen=True, slots=True)
class _QueueSnapshot:
    payload: bytes
    rows: tuple[dict[str, str], ...]


class AuthorizedSkillQueue:
    """Read and write only the audited skill's documented queue contract."""

    def __init__(
        self,
        ready: AuthorizedSkillDoctorResult,
        csv_path: Path,
        output_dir: Path,
        *,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> None:
        if not ready.ready or ready.installed_path is None:
            raise AuthorizedSkillAdapterError("authorized skill runtime is not ready")
        self.skill_path = ready.installed_path
        self.script = self.skill_path / "scripts" / "paper_queue.py"
        if not self.script.is_file():
            raise AuthorizedSkillAdapterError("audited queue script is missing")
        self.csv_path = Path(csv_path)
        self.output_dir = Path(output_dir)
        self.runner = runner
        self._frozen_sha256: str | None = None
        self._candidate_url_to_row_url: dict[str, str] = {}

    def prepare(self, items: Sequence[SkillQueueItem]) -> None:
        ordered = tuple(sorted(
            (_validated_queue_item(item) for item in items),
            key=lambda item: item.paper_id,
        ))
        if not ordered or len({item.paper_id for item in ordered}) != len(ordered):
            raise AuthorizedSkillAdapterError("authorized queue requires unique papers")
        if len({item.doi.lower() for item in ordered}) != len(ordered):
            raise AuthorizedSkillAdapterError("authorized queue requires unique DOIs")
        if len({item.url for item in ordered}) != len(ordered):
            raise AuthorizedSkillAdapterError("authorized queue requires unique browser URLs")
        candidate_url_to_row_url = {
            item.candidate_url or item.url: item.url for item in ordered
        }
        if len(candidate_url_to_row_url) != len(ordered):
            raise AuthorizedSkillAdapterError("authorized queue requires unique candidate URLs")
        buffer = StringIO()
        writer = csv.DictWriter(buffer, fieldnames=("doi", "url", "title", "paper_id"), lineterminator="\n")
        writer.writeheader()
        for item in ordered:
            writer.writerow({
                "doi": item.doi, "url": item.url,
                "title": item.title, "paper_id": item.paper_id,
            })
        payload = buffer.getvalue().encode("utf-8")
        payload_hash = sha256(payload).hexdigest()
        try:
            existing = _read_regular_file_bytes(
                self.csv_path,
                description="authorized queue",
                lock_mode=0o444,
                expected_payload=payload,
            )
        except FileNotFoundError:
            _publish_readonly_bytes(self.csv_path, payload)
        else:
            if existing != payload:
                raise AuthorizedSkillAdapterError(
                    "authorized queue is immutable after creation"
                )
        self._frozen_sha256 = payload_hash
        self._candidate_url_to_row_url = candidate_url_to_row_url

    def frozen_items(self) -> tuple[SkillQueueItem, ...]:
        """Return the validated immutable queue in its canonical order."""

        snapshot = self._queue_snapshot(require_prepared=False)
        items = tuple(_validated_queue_item(SkillQueueItem(
            row["paper_id"], row["doi"], row["url"], row.get("title", "")
        )) for row in snapshot.rows)
        if len({item.paper_id for item in items}) != len(items):
            raise AuthorizedSkillAdapterError("authorized queue requires unique papers")
        if len({item.doi.lower() for item in items}) != len(items):
            raise AuthorizedSkillAdapterError("authorized queue requires unique DOIs")
        if len({item.url for item in items}) != len(items):
            raise AuthorizedSkillAdapterError(
                "authorized queue requires unique browser URLs"
            )
        if tuple(item.paper_id for item in items) != tuple(
            sorted(item.paper_id for item in items)
        ):
            raise AuthorizedSkillAdapterError(
                "authorized queue rows are not in canonical order"
            )
        return items

    def has_queue_file(self) -> bool:
        """Probe the queue path without following a final symbolic link."""

        try:
            _read_regular_file_bytes(self.csv_path, description="authorized queue")
        except FileNotFoundError:
            return False
        return True

    def plan(self) -> Mapping[str, Any]:
        return self._command("plan")

    def next_batch(self, *, limit: int = 2) -> tuple[Mapping[str, Any], ...]:
        if limit not in {1, 2}:
            raise AuthorizedSkillAdapterError("visible browser batches are limited to one or two papers")
        value = self._command("next", "--unscanned", "--limit", str(limit))
        if not isinstance(value, list) or not all(isinstance(item, Mapping) for item in value):
            raise AuthorizedSkillAdapterError("audited queue returned an invalid browser batch")
        return tuple(value)

    def state_for_url(self, url: str) -> tuple[str, str]:
        snapshot = self._queue_snapshot(require_prepared=True)
        row = self._row_for_url(url, snapshot.rows)
        if _platform(row["doi"]) == "unsupported":
            return "manual", "authorized_skill_publisher_unsupported"
        event = self._latest_ledger().get(row["doi"].lower())
        if event is None:
            return "handoff", "authorized_browser_handoff_required"
        if event.get("status") not in FINAL_SKILL_STATUSES:
            return "manual", _manual_reason(event)
        self._article_bytes(snapshot.rows, row, event)
        return "ready", "authorized_skill_article_staged"

    def fetch_response(self, url: str) -> HTTPResponse:
        snapshot = self._queue_snapshot(require_prepared=True)
        row = self._row_for_url(url, snapshot.rows)
        event = self._latest_ledger().get(row["doi"].lower())
        if event is None or event.get("status") not in FINAL_SKILL_STATUSES:
            raise AuthorizedSkillAdapterError("authorized browser handoff is not complete")
        article = self._article_bytes(snapshot.rows, row, event)
        return HTTPResponse(
            200, {"Content-Type": "application/pdf"}, article, final_url=url
        )

    def _command(self, command: str, *arguments: str) -> Any:
        snapshot = self._queue_snapshot(require_prepared=True)
        with tempfile.TemporaryDirectory(prefix="paper-agent-authorized-queue-") as directory:
            private_csv = Path(directory) / "papers.csv"
            _write_private_readonly_file(private_csv, snapshot.payload)
            try:
                completed = self.runner(
                    [
                        sys.executable, str(self.script), command,
                        "--csv", str(private_csv), "--output", str(self.output_dir),
                        *arguments,
                    ],
                    cwd=directory,
                    env={
                        key: os.environ[key]
                        for key in ("LANG", "LC_ALL", "PATH")
                        if key in os.environ
                    },
                    capture_output=True,
                    text=True,
                    check=False,
                )
            finally:
                # The skill consumes only the private snapshot.  Reprove the
                # durable queue before accepting any command result.
                self._queue_snapshot(require_prepared=True)
        if completed.returncode != 0:
            raise AuthorizedSkillAdapterError(f"authorized queue command failed: {command}")
        return json.loads(completed.stdout)

    def _queue_snapshot(self, *, require_prepared: bool) -> _QueueSnapshot:
        if require_prepared and self._frozen_sha256 is None:
            raise AuthorizedSkillAdapterError("authorized queue was not prepared")
        try:
            payload = _read_regular_file_bytes(
                self.csv_path, description="authorized queue"
            )
        except FileNotFoundError as error:
            raise AuthorizedSkillAdapterError(
                "authorized queue was not prepared"
            ) from error
        if (
            self._frozen_sha256 is not None
            and sha256(payload).hexdigest() != self._frozen_sha256
        ):
            raise AuthorizedSkillAdapterError("authorized queue changed after approval")
        return _QueueSnapshot(payload, _parse_queue_rows(payload))

    def _row_for_url(
        self, url: str, rows: Sequence[Mapping[str, str]]
    ) -> Mapping[str, str]:
        row_url = self._candidate_url_to_row_url.get(url)
        if row_url is None:
            raise AuthorizedSkillAdapterError(
                "candidate URL is outside the authorized skill queue"
            )
        matches = [row for row in rows if row["url"] == row_url]
        if len(matches) != 1:
            raise AuthorizedSkillAdapterError("candidate URL is outside the authorized skill queue")
        return matches[0]

    def _latest_ledger(self) -> dict[str, Mapping[str, Any]]:
        path = self.output_dir / "_state" / "ledger.jsonl"
        try:
            payload = _read_regular_file_bytes(path, description="skill ledger")
        except FileNotFoundError:
            return {}
        latest: dict[str, Mapping[str, Any]] = {}
        try:
            lines = payload.decode("utf-8").splitlines()
        except UnicodeDecodeError as error:
            raise AuthorizedSkillAdapterError("skill ledger is invalid") from error
        for line in lines:
            if line.strip():
                event = json.loads(line)
                if isinstance(event, Mapping) and isinstance(event.get("doi"), str):
                    latest[event["doi"].lower()] = event
        return latest

    def _article_bytes(
        self,
        rows: Sequence[Mapping[str, str]],
        row: Mapping[str, str],
        event: Mapping[str, Any],
    ) -> bytes:
        index = next(index for index, value in enumerate(rows, 1) if value["doi"].lower() == row["doi"].lower())
        directory = self.output_dir / _platform(row["doi"]) / _safe_dir_name(index, row["doi"])
        article = directory / "article.pdf"
        try:
            payload = _read_regular_file_bytes(article, description="article.pdf")
        except FileNotFoundError as error:
            raise AuthorizedSkillAdapterError(
                "final skill ledger has no article.pdf"
            ) from error
        files = event.get("files")
        metadata = next(
            (item for item in files if isinstance(item, Mapping) and item.get("name") == "article.pdf"),
            None,
        ) if isinstance(files, list) else None
        if metadata is None or metadata.get("sha256") != sha256(payload).hexdigest():
            raise AuthorizedSkillAdapterError("article.pdf does not match the final skill ledger")
        return payload


class AuditedAuthorizedSkillAdapter:
    """DownloadProvider adapter over a finalized audited-skill queue entry."""

    name = "authorized_skill"

    def __init__(self, service: DownloadService, queue: AuthorizedSkillQueue) -> None:
        self.service = service
        self.queue = queue

    def probe(self, candidate: AccessLocationCandidate, context: ProbeContext) -> FetchDecision:
        try:
            state, reason = self.queue.state_for_url(candidate.url)
        except AuthorizedSkillAdapterError:
            return FetchDecision(
                candidate.candidate_id, FetchDecisionStatus.MANUAL,
                "authorized_skill_queue_mismatch", self.service.policy.version,
            )
        if state != "ready":
            return FetchDecision(
                candidate.candidate_id, FetchDecisionStatus.MANUAL, reason, self.service.policy.version,
            )
        return self.service.probe(
            candidate,
            purpose=context.purpose,
            provider=self.name,
            now=context.now,
            authorization_grant_id=context.authorization_grant_id,
            mode=context.mode,
            skill_digest=context.skill_digest,
            dependency_digest=context.dependency_digest,
            collection_id=context.collection_id,
            collection_snapshot_hash=context.collection_snapshot_hash,
            selection_snapshot_hash=context.selection_snapshot_hash,
            run_id=context.run_id,
        )

    def fetch(self, request: FetchRequest, context: FetchContext) -> DownloadResult:
        if request.provider != self.name:
            raise AuthorizedSkillAdapterError("fetch request is bound to another provider")
        return self.service.fetch(
            request,
            run_id=context.run_id,
            now=context.now,
            authorization_context=context.authorization_context,
        )


def _canonical_doi(value: str) -> str:
    normalized = value.strip()
    normalized = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", normalized, flags=re.IGNORECASE)
    if not normalized.startswith("10.") or "/" not in normalized:
        raise AuthorizedSkillAdapterError(f"invalid DOI: {value}")
    return normalized


def _platform(doi: str) -> str:
    value = doi.lower()
    if value.startswith("10.1002/"):
        return "wiley"
    if value.startswith("10.1038/"):
        return "nature"
    if value.startswith("10.1021/"):
        return "acs"
    return "unsupported"


def authorized_publisher_host_matches(doi: str, host: str | None) -> bool:
    """Bind the optional queue URL to the skill's audited DOI publisher."""

    if host is None:
        return False
    return host.lower().rstrip(".") in AUTHORIZED_PUBLISHER_HOSTS.get(
        _platform(doi), frozenset()
    )


def _validated_queue_item(item: SkillQueueItem) -> SkillQueueItem:
    doi = _canonical_doi(item.doi)
    parsed = urlsplit(item.url)
    candidate_url = item.candidate_url or item.url
    candidate = urlsplit(candidate_url)
    try:
        same_endpoint = (
            candidate.hostname is not None
            and parsed.hostname is not None
            and candidate.hostname.lower().rstrip(".")
            == parsed.hostname.lower().rstrip(".")
            and (candidate.port or 443) == (parsed.port or 443)
            and (candidate.port or 443) == 443
        )
        unsafe_explicit_landing = item.candidate_url is not None and (
            _url_identity(parsed) == _url_identity(candidate)
        )
    except ValueError as error:
        raise AuthorizedSkillAdapterError(
            "authorized queue candidate or browser URL is invalid"
        ) from error
    if (
        not item.paper_id
        or parsed.scheme != "https"
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or not authorized_publisher_host_matches(doi, parsed.hostname)
        or candidate.scheme != "https"
        or candidate.hostname is None
        or candidate.username is not None
        or candidate.password is not None
        or not authorized_publisher_host_matches(doi, candidate.hostname)
        or not same_endpoint
        or unsafe_explicit_landing
        or _looks_like_pdf_url(parsed)
    ):
        raise AuthorizedSkillAdapterError(
            "authorized queue candidate and browser URLs do not match the DOI publisher"
        )
    return SkillQueueItem(item.paper_id, doi, item.url, item.title, candidate_url)


def _safe_dir_name(index: int, doi: str) -> str:
    return f"{index:04d}_{re.sub(r'[^A-Za-z0-9._-]+', '_', doi)}"


def _parse_queue_rows(payload: bytes) -> tuple[dict[str, str], ...]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise AuthorizedSkillAdapterError("authorized queue CSV is invalid") from error
    reader = csv.DictReader(StringIO(text, newline=""))
    if tuple(reader.fieldnames or ()) != ("doi", "url", "title", "paper_id"):
        raise AuthorizedSkillAdapterError("authorized queue CSV is invalid")
    rows = tuple(dict(row) for row in reader)
    if not rows or any(
        None in row
        or None in row.values()
        or not row.get("doi")
        or not row.get("url")
        or not row.get("paper_id")
        for row in rows
    ):
        raise AuthorizedSkillAdapterError("authorized queue CSV is invalid")
    validated = tuple(
        _validated_queue_item(SkillQueueItem(
            row["paper_id"], row["doi"], row["url"], row.get("title", "")
        ))
        for row in rows
    )
    if (
        len({item.paper_id for item in validated}) != len(validated)
        or len({item.doi.lower() for item in validated}) != len(validated)
        or len({item.url for item in validated}) != len(validated)
        or tuple(item.paper_id for item in validated)
        != tuple(sorted(item.paper_id for item in validated))
    ):
        raise AuthorizedSkillAdapterError(
            "authorized queue CSV has duplicate or non-canonical rows"
        )
    return rows


def _read_regular_file_bytes(
    path: Path,
    *,
    description: str,
    lock_mode: int | None = None,
    expected_payload: bytes | None = None,
) -> bytes:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        raise
    except OSError as error:
        raise AuthorizedSkillAdapterError(
            f"{description} must be a regular file and not a symbolic link"
        ) from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise AuthorizedSkillAdapterError(
                f"{description} must be a regular file and not a symbolic link"
            )
        blocks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            blocks.append(block)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise AuthorizedSkillAdapterError(
                f"{description} changed while it was being read"
            )
        payload = b"".join(blocks)
        if (
            lock_mode is not None
            and (expected_payload is None or payload == expected_payload)
        ):
            os.fchmod(descriptor, lock_mode)
            os.fsync(descriptor)
        return payload
    finally:
        os.close(descriptor)


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise AuthorizedSkillAdapterError("authorized queue write failed")
        view = view[written:]


def _write_private_readonly_file(path: Path, payload: bytes) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as error:
        raise AuthorizedSkillAdapterError(
            "could not create a private authorized queue snapshot"
        ) from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise AuthorizedSkillAdapterError(
                "private authorized queue snapshot is not a regular file"
            )
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_readonly_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        directory = os.open(path.parent, directory_flags)
    except OSError as error:
        raise AuthorizedSkillAdapterError(
            "authorized queue parent must be a real directory"
        ) from error
    temporary_name = (
        f".{path.name}.{os.getpid()}.{secrets.token_hex(12)}.part"
    )
    descriptor: int | None = None
    temporary_identity: tuple[int, int] | None = None
    try:
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(temporary_name, flags, 0o600, dir_fd=directory)
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise AuthorizedSkillAdapterError(
                "authorized queue temporary path is not a regular file"
            )
        temporary_identity = (file_stat.st_dev, file_stat.st_ino)
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        try:
            os.link(
                temporary_name,
                path.name,
                src_dir_fd=directory,
                dst_dir_fd=directory,
                follow_symlinks=False,
            )
        except FileExistsError as error:
            raise AuthorizedSkillAdapterError(
                "authorized queue target appeared during secure publication"
            ) from error
        os.fsync(directory)
    except OSError as error:
        if isinstance(error, AuthorizedSkillAdapterError):
            raise
        raise AuthorizedSkillAdapterError(
            "authorized queue could not be published securely"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            current = os.stat(
                temporary_name, dir_fd=directory, follow_symlinks=False
            )
        except FileNotFoundError:
            pass
        else:
            if temporary_identity == (current.st_dev, current.st_ino):
                os.unlink(temporary_name, dir_fd=directory)
                os.fsync(directory)
        os.close(directory)


def _url_identity(value: Any) -> tuple[str, str, int, str, str]:
    host = (value.hostname or "").lower().rstrip(".")
    return (
        value.scheme.lower(),
        host,
        value.port or 443,
        unquote(value.path or "/"),
        value.query,
    )


def _looks_like_pdf_url(value: Any) -> bool:
    path = unquote(value.path or "").lower()
    return path.endswith(".pdf") or bool(
        re.search(r"/doi/(?:pdf|epdf)(?:/|$)", path)
    )


def _manual_reason(event: Mapping[str, Any]) -> str:
    reason = str(event.get("reason") or "").lower()
    if any(value in reason for value in ("captcha", "403", "429", "access", "login", "authoriz")):
        return "authorized_session_repair_required"
    return "authorized_skill_manual_required"
