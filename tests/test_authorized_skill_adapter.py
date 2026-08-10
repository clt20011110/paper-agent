from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

from pypdf import PdfWriter
import pytest

from paper_agent.authorized_skill_adapter import (
    AuditedAuthorizedSkillAdapter,
    AuthorizedSkillAdapterError,
    AuthorizedSkillQueue,
    SkillQueueItem,
)
from paper_agent.domain import (
    AccessBasis,
    AccessLocationCandidate,
    FetchDecision,
    FetchDecisionStatus,
    PublicationVersion,
)
from paper_agent.download_providers import ProbeContext


NOW = "2026-08-10T00:00:00Z"


def _pdf() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def _queue(tmp_path: Path, runner=subprocess.run) -> AuthorizedSkillQueue:
    skill = tmp_path / "skill"
    script = skill / "scripts" / "paper_queue.py"
    script.parent.mkdir(parents=True)
    script.write_text("# audited fixture\n", encoding="utf-8")
    ready = SimpleNamespace(ready=True, installed_path=skill)
    queue = AuthorizedSkillQueue(
        ready, tmp_path / "queue" / "papers.csv", tmp_path / "results", runner=runner,
    )
    queue.prepare((SkillQueueItem("paper-1", "10.1038/example", "https://nature.test/article", "Paper"),))
    return queue


def _stage_article(queue: AuthorizedSkillQueue, payload: bytes | None = None) -> Path:
    value = payload or _pdf()
    article = queue.output_dir / "nature" / "0001_10.1038_example" / "article.pdf"
    article.parent.mkdir(parents=True)
    article.write_bytes(value)
    ledger = queue.output_dir / "_state" / "ledger.jsonl"
    ledger.parent.mkdir(parents=True)
    ledger.write_text(json.dumps({
        "doi": "10.1038/example", "status": "complete_no_si",
        "files": [{"name": "article.pdf", "sha256": sha256(value).hexdigest()}],
    }) + "\n", encoding="utf-8")
    return article


def _candidate() -> AccessLocationCandidate:
    return AccessLocationCandidate(
        "candidate-1", "paper-1", "publisher_public", "https://nature.test/article",
        "https://nature.test/article", "nature.test", PublicationVersion.PUBLISHED,
        None, AccessBasis.USER_SUBSCRIPTION, NOW, "a" * 64, {},
    )


def test_queue_is_stable_and_emits_only_small_browser_handoffs(tmp_path: Path) -> None:
    calls = []

    def runner(argv, **kwargs):
        calls.append((argv, kwargs))
        output = {"total": 1} if "plan" in argv else [{"doi": "10.1038/example"}]
        return subprocess.CompletedProcess(argv, 0, json.dumps(output), "")

    queue = _queue(tmp_path, runner)

    assert queue.plan() == {"total": 1}
    assert queue.next_batch(limit=2) == ({"doi": "10.1038/example"},)
    assert calls[0][0][1].endswith("paper_queue.py")
    assert calls[0][0][2] == "plan"
    assert "--unscanned" in calls[1][0]
    with pytest.raises(AuthorizedSkillAdapterError, match="one or two"):
        queue.next_batch(limit=3)
    with pytest.raises(AuthorizedSkillAdapterError, match="immutable"):
        queue.prepare((SkillQueueItem("paper-1", "10.1038/changed", "https://nature.test/article"),))


def test_queue_imports_only_final_ledger_bound_article(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    assert queue.state_for_url("https://nature.test/article") == (
        "handoff", "authorized_browser_handoff_required",
    )
    article = _stage_article(queue)

    state = queue.state_for_url("https://nature.test/article")
    response = queue.fetch_response("https://nature.test/article")

    assert state == ("ready", "authorized_skill_article_staged")
    assert response.body == article.read_bytes()
    assert response.final_url == "https://nature.test/article"
    article.write_bytes(article.read_bytes() + b"drift")
    with pytest.raises(AuthorizedSkillAdapterError, match="ledger"):
        queue.fetch_response("https://nature.test/article")


def test_adapter_does_not_probe_download_service_before_visible_browser_handoff(tmp_path: Path) -> None:
    queue = _queue(tmp_path)

    class Service:
        policy = SimpleNamespace(version="download-access-v1")

        def __init__(self):
            self.calls = []

        def probe(self, candidate, **kwargs):
            self.calls.append((candidate, kwargs))
            return FetchDecision(
                candidate.candidate_id, FetchDecisionStatus.NEEDS_GRANT, "delegated_to_service",
                self.policy.version,
            )

    service = Service()
    adapter = AuditedAuthorizedSkillAdapter(service, queue)  # type: ignore[arg-type]
    context = ProbeContext("personal_research", NOW, authorization_grant_id="grant-1")

    waiting = adapter.probe(_candidate(), context)
    _stage_article(queue)
    ready = adapter.probe(_candidate(), context)

    assert waiting.status is FetchDecisionStatus.MANUAL
    assert waiting.reason_code == "authorized_browser_handoff_required"
    assert ready.status is FetchDecisionStatus.NEEDS_GRANT
    assert len(service.calls) == 1
