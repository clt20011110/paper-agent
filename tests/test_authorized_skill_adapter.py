from __future__ import annotations

import csv
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

from pypdf import PdfWriter
import pytest
import paper_agent.authorized_skill_adapter as authorized_skill_adapter

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
    queue.prepare((SkillQueueItem(
        "paper-1", "10.1038/example", "https://www.nature.com/articles/example", "Paper",
    ),))
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
        "candidate-1", "paper-1", "publisher_public", "https://www.nature.com/articles/example",
        "https://www.nature.com/articles/example", "www.nature.com", PublicationVersion.PUBLISHED,
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
    assert queue.csv_path.stat().st_mode & 0o777 == 0o444
    with pytest.raises(AuthorizedSkillAdapterError, match="one or two"):
        queue.next_batch(limit=3)
    with pytest.raises(AuthorizedSkillAdapterError, match="immutable"):
        queue.prepare((SkillQueueItem(
            "paper-1", "10.1038/changed", "https://www.nature.com/articles/example",
        ),))


def test_queue_rejects_post_approval_csv_mutation_before_skill_execution(
    tmp_path: Path,
) -> None:
    calls = []

    def runner(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, "[]", "")

    queue = _queue(tmp_path, runner)
    queue.csv_path.chmod(0o644)
    queue.csv_path.write_text(
        "doi,url,title,paper_id\n"
        "10.1038/example,https://evil.example/collect,Evil,paper-outside\n",
        encoding="utf-8",
    )

    with pytest.raises(AuthorizedSkillAdapterError, match="changed after approval"):
        queue.next_batch()
    assert calls == []


def test_queue_rejects_url_outside_the_doi_publishers_audited_hosts(
    tmp_path: Path,
) -> None:
    skill = tmp_path / "skill"
    script = skill / "scripts" / "paper_queue.py"
    script.parent.mkdir(parents=True)
    script.write_text("# audited fixture\n", encoding="utf-8")
    queue = AuthorizedSkillQueue(
        SimpleNamespace(ready=True, installed_path=skill),
        tmp_path / "queue" / "papers.csv",
        tmp_path / "results",
    )

    with pytest.raises(AuthorizedSkillAdapterError, match="DOI publisher"):
        queue.prepare((SkillQueueItem(
            "paper-1",
            "10.1038/example",
            "https://onlinelibrary.wiley.com/doi/example",
        ),))
    with pytest.raises(AuthorizedSkillAdapterError, match="DOI publisher"):
        queue.prepare((SkillQueueItem(
            "paper-1",
            "10.1038/example",
            "http://www.nature.com/articles/example",
        ),))

    assert not queue.csv_path.exists()


def test_queue_writes_landing_url_but_keeps_pdf_candidate_as_fetch_identity(
    tmp_path: Path,
) -> None:
    skill = tmp_path / "skill"
    script = skill / "scripts" / "paper_queue.py"
    script.parent.mkdir(parents=True)
    script.write_text("# audited fixture\n", encoding="utf-8")
    queue = AuthorizedSkillQueue(
        SimpleNamespace(ready=True, installed_path=skill),
        tmp_path / "queue" / "papers.csv",
        tmp_path / "results",
    )
    landing_url = "https://www.nature.com/articles/example"
    candidate_url = "https://www.nature.com/articles/example.pdf"
    queue.prepare((SkillQueueItem(
        "paper-1",
        "10.1038/example",
        landing_url,
        "Paper",
        candidate_url,
    ),))

    with queue.csv_path.open(newline="", encoding="utf-8") as handle:
        [row] = list(csv.DictReader(handle))
    assert row["url"] == landing_url
    assert queue.state_for_url(candidate_url) == (
        "handoff", "authorized_browser_handoff_required",
    )
    with pytest.raises(AuthorizedSkillAdapterError, match="outside"):
        queue.state_for_url(landing_url)

    _stage_article(queue)
    candidate = AccessLocationCandidate(
        "candidate-pdf", "paper-1", "publisher_public", candidate_url,
        landing_url, "www.nature.com", PublicationVersion.PUBLISHED,
        None, AccessBasis.USER_SUBSCRIPTION, NOW, "a" * 64, {},
    )

    class Service:
        policy = SimpleNamespace(version="download-access-v1")

        def __init__(self) -> None:
            self.urls: list[str] = []

        def probe(self, value, **_kwargs):
            self.urls.append(value.url)
            return FetchDecision(
                value.candidate_id,
                FetchDecisionStatus.NEEDS_GRANT,
                "delegated_to_service",
                self.policy.version,
            )

    service = Service()
    decision = AuditedAuthorizedSkillAdapter(  # type: ignore[arg-type]
        service, queue
    ).probe(candidate, ProbeContext("personal_research", NOW))
    response = queue.fetch_response(candidate_url)

    assert decision.status is FetchDecisionStatus.NEEDS_GRANT
    assert service.urls == [candidate_url]
    assert response.final_url == candidate_url
    assert response.body.startswith(b"%PDF")


def test_queue_rejects_candidate_landing_mismatch_and_duplicate_doi(
    tmp_path: Path,
) -> None:
    skill = tmp_path / "skill"
    script = skill / "scripts" / "paper_queue.py"
    script.parent.mkdir(parents=True)
    script.write_text("# audited fixture\n", encoding="utf-8")
    queue = AuthorizedSkillQueue(
        SimpleNamespace(ready=True, installed_path=skill),
        tmp_path / "queue" / "papers.csv",
        tmp_path / "results",
    )

    with pytest.raises(AuthorizedSkillAdapterError, match="DOI publisher"):
        queue.prepare((SkillQueueItem(
            "paper-1",
            "10.1038/example",
            "https://www.nature.com/articles/example",
            candidate_url="https://pubs.acs.org/doi/pdf/example",
        ),))
    with pytest.raises(AuthorizedSkillAdapterError, match="unique DOIs"):
        queue.prepare((
            SkillQueueItem(
                "paper-1",
                "10.1038/example",
                "https://www.nature.com/articles/example-a",
            ),
            SkillQueueItem(
                "paper-2",
                "10.1038/example",
                "https://www.nature.com/articles/example-b",
            ),
        ))
    with pytest.raises(AuthorizedSkillAdapterError, match="DOI publisher"):
        queue.prepare((SkillQueueItem(
            "paper-1",
            "10.1038/example",
            "https://www.nature.com/articles/example.pdf",
        ),))

    assert not queue.csv_path.exists()


def test_prepare_rejects_symlink_and_does_not_chmod_unrelated_content(
    tmp_path: Path,
) -> None:
    skill = tmp_path / "skill"
    script = skill / "scripts" / "paper_queue.py"
    script.parent.mkdir(parents=True)
    script.write_text("# audited fixture\n", encoding="utf-8")
    queue_path = tmp_path / "queue" / "papers.csv"
    queue_path.parent.mkdir()
    unrelated = tmp_path / "unrelated.csv"
    unrelated.write_text("private\n", encoding="utf-8")
    unrelated.chmod(0o600)
    queue_path.symlink_to(unrelated)
    queue = AuthorizedSkillQueue(
        SimpleNamespace(ready=True, installed_path=skill),
        queue_path,
        tmp_path / "results",
    )

    with pytest.raises(AuthorizedSkillAdapterError, match="symbolic link"):
        queue.prepare((SkillQueueItem(
            "paper-1",
            "10.1038/example",
            "https://www.nature.com/articles/example",
        ),))

    assert unrelated.read_text(encoding="utf-8") == "private\n"
    assert unrelated.stat().st_mode & 0o777 == 0o600


def test_prepare_does_not_chmod_an_existing_mismatched_regular_file(
    tmp_path: Path,
) -> None:
    skill = tmp_path / "skill"
    script = skill / "scripts" / "paper_queue.py"
    script.parent.mkdir(parents=True)
    script.write_text("# audited fixture\n", encoding="utf-8")
    queue_path = tmp_path / "queue" / "papers.csv"
    queue_path.parent.mkdir()
    queue_path.write_text("unrelated content\n", encoding="utf-8")
    queue_path.chmod(0o600)
    queue = AuthorizedSkillQueue(
        SimpleNamespace(ready=True, installed_path=skill),
        queue_path,
        tmp_path / "results",
    )

    with pytest.raises(AuthorizedSkillAdapterError, match="immutable"):
        queue.prepare((SkillQueueItem(
            "paper-1",
            "10.1038/example",
            "https://www.nature.com/articles/example",
        ),))

    assert queue_path.read_text(encoding="utf-8") == "unrelated content\n"
    assert queue_path.stat().st_mode & 0o777 == 0o600


def test_prepare_never_overwrites_a_target_created_during_publication(
    tmp_path: Path, monkeypatch,
) -> None:
    skill = tmp_path / "skill"
    script = skill / "scripts" / "paper_queue.py"
    script.parent.mkdir(parents=True)
    script.write_text("# audited fixture\n", encoding="utf-8")
    queue_path = tmp_path / "queue" / "papers.csv"
    queue = AuthorizedSkillQueue(
        SimpleNamespace(ready=True, installed_path=skill),
        queue_path,
        tmp_path / "results",
    )
    original_link = os.link

    def racing_link(source, destination, **kwargs):
        queue_path.write_bytes(b"pre-existing target\n")
        return original_link(source, destination, **kwargs)

    monkeypatch.setattr(authorized_skill_adapter.os, "link", racing_link)

    with pytest.raises(AuthorizedSkillAdapterError, match="appeared"):
        queue.prepare((SkillQueueItem(
            "paper-1",
            "10.1038/example",
            "https://www.nature.com/articles/example",
        ),))

    assert queue_path.read_bytes() == b"pre-existing target\n"


def test_command_uses_private_snapshot_and_rejects_original_queue_drift(
    tmp_path: Path,
) -> None:
    observed: dict[str, object] = {}
    queue: AuthorizedSkillQueue

    def runner(argv, **_kwargs):
        private_csv = Path(argv[argv.index("--csv") + 1])
        observed["private_csv"] = private_csv
        observed["payload"] = private_csv.read_bytes()
        observed["mode"] = private_csv.stat().st_mode & 0o777
        queue.csv_path.chmod(0o644)
        queue.csv_path.write_text("changed\n", encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0, json.dumps({"total": 1}), "")

    queue = _queue(tmp_path, runner)
    approved = queue.csv_path.read_bytes()

    with pytest.raises(AuthorizedSkillAdapterError, match="changed after approval"):
        queue.plan()

    assert observed["private_csv"] != queue.csv_path
    assert observed["payload"] == approved
    assert observed["mode"] == 0o444


def test_fetch_uses_one_queue_and_article_byte_snapshot(
    tmp_path: Path, monkeypatch,
) -> None:
    queue = _queue(tmp_path)
    article = _stage_article(queue)
    approved_article = article.read_bytes()
    original_reader = authorized_skill_adapter._read_regular_file_bytes
    reads: list[Path] = []

    def reader(path, **kwargs):
        value = original_reader(path, **kwargs)
        resolved = Path(path)
        reads.append(resolved)
        if resolved == article:
            article.write_bytes(b"drift after verified snapshot")
        return value

    monkeypatch.setattr(
        authorized_skill_adapter, "_read_regular_file_bytes", reader
    )

    response = queue.fetch_response("https://www.nature.com/articles/example")

    assert response.body == approved_article
    assert article.read_bytes() != response.body
    assert reads.count(queue.csv_path) == 1
    assert reads.count(article) == 1


def test_queue_imports_only_final_ledger_bound_article(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    assert queue.state_for_url("https://www.nature.com/articles/example") == (
        "handoff", "authorized_browser_handoff_required",
    )
    article = _stage_article(queue)

    state = queue.state_for_url("https://www.nature.com/articles/example")
    response = queue.fetch_response("https://www.nature.com/articles/example")

    assert state == ("ready", "authorized_skill_article_staged")
    assert response.body == article.read_bytes()
    assert response.final_url == "https://www.nature.com/articles/example"
    article.write_bytes(article.read_bytes() + b"drift")
    with pytest.raises(AuthorizedSkillAdapterError, match="ledger"):
        queue.fetch_response("https://www.nature.com/articles/example")


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
