"""Audited queue handoff for the optional authorized publisher skill.

The adapter never drives a browser or reads browser session state.  It prepares
the skill's immutable CSV queue, exposes the next small visible-browser batch,
and imports only a finalized ``article.pdf`` recorded by the skill ledger.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import csv
from dataclasses import dataclass
from hashlib import sha256
from io import StringIO
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

from .authorized_skill_runtime import AuthorizedSkillDoctorResult
from .domain import AccessLocationCandidate, DownloadResult, FetchDecision, FetchDecisionStatus, FetchRequest
from .download_providers import FetchContext, ProbeContext
from .downloads import DownloadService, HTTPResponse


FINAL_SKILL_STATUSES = frozenset({"complete", "complete_no_si"})


class AuthorizedSkillAdapterError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SkillQueueItem:
    paper_id: str
    doi: str
    url: str
    title: str = ""


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

    def prepare(self, items: Sequence[SkillQueueItem]) -> None:
        ordered = tuple(sorted(items, key=lambda item: item.paper_id))
        if not ordered or len({item.paper_id for item in ordered}) != len(ordered):
            raise AuthorizedSkillAdapterError("authorized queue requires unique papers")
        if len({item.url for item in ordered}) != len(ordered):
            raise AuthorizedSkillAdapterError("authorized queue requires unique candidate URLs")
        buffer = StringIO()
        writer = csv.DictWriter(buffer, fieldnames=("doi", "url", "title", "paper_id"), lineterminator="\n")
        writer.writeheader()
        for item in ordered:
            writer.writerow({
                "doi": _canonical_doi(item.doi), "url": item.url,
                "title": item.title, "paper_id": item.paper_id,
            })
        payload = buffer.getvalue()
        if self.csv_path.is_file():
            if self.csv_path.read_text(encoding="utf-8") != payload:
                raise AuthorizedSkillAdapterError("authorized queue is immutable after creation")
            return
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        part = self.csv_path.with_suffix(self.csv_path.suffix + ".part")
        part.write_text(payload, encoding="utf-8")
        os.replace(part, self.csv_path)

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
        row = self._row_for_url(url)
        if _platform(row["doi"]) == "unsupported":
            return "manual", "authorized_skill_publisher_unsupported"
        event = self._latest_ledger().get(row["doi"].lower())
        if event is None:
            return "handoff", "authorized_browser_handoff_required"
        if event.get("status") not in FINAL_SKILL_STATUSES:
            return "manual", _manual_reason(event)
        self._article(row, event)
        return "ready", "authorized_skill_article_staged"

    def fetch_response(self, url: str) -> HTTPResponse:
        row = self._row_for_url(url)
        event = self._latest_ledger().get(row["doi"].lower())
        if event is None or event.get("status") not in FINAL_SKILL_STATUSES:
            raise AuthorizedSkillAdapterError("authorized browser handoff is not complete")
        article = self._article(row, event)
        return HTTPResponse(200, {"Content-Type": "application/pdf"}, article.read_bytes(), final_url=url)

    def _command(self, command: str, *arguments: str) -> Any:
        completed = self.runner(
            [
                sys.executable, str(self.script), command,
                "--csv", str(self.csv_path), "--output", str(self.output_dir), *arguments,
            ],
            cwd=str(self.csv_path.parent),
            env={key: os.environ[key] for key in ("LANG", "LC_ALL", "PATH") if key in os.environ},
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise AuthorizedSkillAdapterError(f"authorized queue command failed: {command}")
        return json.loads(completed.stdout)

    def _rows(self) -> tuple[dict[str, str], ...]:
        with self.csv_path.open("r", encoding="utf-8", newline="") as handle:
            rows = tuple(dict(row) for row in csv.DictReader(handle))
        if not rows or any(not row.get("doi") or not row.get("url") for row in rows):
            raise AuthorizedSkillAdapterError("authorized queue CSV is invalid")
        return rows

    def _row_for_url(self, url: str) -> dict[str, str]:
        matches = [row for row in self._rows() if row["url"] == url]
        if len(matches) != 1:
            raise AuthorizedSkillAdapterError("candidate URL is outside the authorized skill queue")
        return matches[0]

    def _latest_ledger(self) -> dict[str, Mapping[str, Any]]:
        path = self.output_dir / "_state" / "ledger.jsonl"
        if not path.is_file():
            return {}
        latest: dict[str, Mapping[str, Any]] = {}
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    event = json.loads(line)
                    if isinstance(event, Mapping) and isinstance(event.get("doi"), str):
                        latest[event["doi"].lower()] = event
        return latest

    def _article(self, row: Mapping[str, str], event: Mapping[str, Any]) -> Path:
        rows = self._rows()
        index = next(index for index, value in enumerate(rows, 1) if value["doi"].lower() == row["doi"].lower())
        directory = self.output_dir / _platform(row["doi"]) / _safe_dir_name(index, row["doi"])
        article = directory / "article.pdf"
        if not article.is_file():
            raise AuthorizedSkillAdapterError("final skill ledger has no article.pdf")
        files = event.get("files")
        metadata = next(
            (item for item in files if isinstance(item, Mapping) and item.get("name") == "article.pdf"),
            None,
        ) if isinstance(files, list) else None
        if metadata is None or metadata.get("sha256") != _file_hash(article):
            raise AuthorizedSkillAdapterError("article.pdf does not match the final skill ledger")
        return article


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


def _safe_dir_name(index: int, doi: str) -> str:
    return f"{index:04d}_{re.sub(r'[^A-Za-z0-9._-]+', '_', doi)}"


def _file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manual_reason(event: Mapping[str, Any]) -> str:
    reason = str(event.get("reason") or "").lower()
    if any(value in reason for value in ("captcha", "403", "429", "access", "login", "authoriz")):
        return "authorized_session_repair_required"
    return "authorized_skill_manual_required"
