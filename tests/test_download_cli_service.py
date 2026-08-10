from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from paper_agent.domain import AccessBasis, FilterStatus, Paper, PaperSource, PublicationVersion
from paper_agent.download_cli_service import (
    AuthorizedSkillHandoffOptions,
    Stage3DownloadService,
    load_provider_terms,
)
from paper_agent.download_providers import DEFAULT_PROVIDER_ORDER, DEFAULT_RESOLVER_ORDER
from paper_agent.downloads import HTTPResponse, ProviderTerms
from paper_agent.repository import PaperRepository
from paper_agent.storage import Database


NOW = "2026-08-10T00:00:00Z"


class Fetcher:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, url: str) -> HTTPResponse:
        self.calls.append(url)
        return HTTPResponse(200, {"Content-Type": "application/pdf"}, _pdf(), url)


@pytest.fixture
def database(tmp_path: Path):
    with Database(tmp_path / "papers.sqlite3") as database:
        database.migrate()
        yield database


def _service(
    tmp_path: Path,
    database: Database,
    fetcher: Fetcher,
    *,
    terms: ProviderTerms | None = None,
    authorized: bool = False,
    clock=None,
    planner=None,
) -> Stage3DownloadService:
    return Stage3DownloadService(
        database,
        {
            "download": {
                "resolvers": list(DEFAULT_RESOLVER_ORDER),
                "providers": list(DEFAULT_PROVIDER_ORDER),
                "purpose": "personal_research",
                "policy_matrix": "policies/download-access-v1.yaml",
                "authorized_skill": {"enabled": authorized},
            }
        },
        config_root=Path(__file__).parents[1],
        artifact_root=tmp_path / "output",
        provider_terms={"public_direct": terms} if terms else None,
        fetcher=fetcher,
        clock=clock or (lambda: datetime.fromisoformat(NOW.replace("Z", "+00:00"))),
        authorized_luna_planner=planner,
    )


@dataclass
class FakeAuthorizedLunaPlanner:
    selected: bool = True
    calls: list[object] | None = None

    def __call__(self, control):
        assert self.calls is not None
        self.calls.append(control)
        return SimpleNamespace(
            selected=self.selected,
            status="invoke_skill" if self.selected else "manual_queue",
            page_state="unknown",
            next_action="invoke_audited_skill" if self.selected else "manual_queue",
            reason_code="authorized_handoff_selected" if self.selected else "authorized_handoff_deferred",
            invocation_metadata={
                "invocation_id": f"fake-luna-{len(self.calls)}",
                "profile": "stage3_authorized_luna",
                "model": "gpt-5.6-luna",
                "reasoning_effort": "low",
                "actual_model": "gpt-5.6-luna",
                "actual_profile": "stage3_authorized_luna",
            },
        )


def _terms() -> ProviderTerms:
    return ProviderTerms(
        "public_direct", "test-v1", "https://publisher.example/terms", True, True, True,
        domain_allowlist=("publisher.example",),
    )


def _paper(
    database: Database,
    *,
    access_basis: AccessBasis,
    license: str | None,
    doi: str | None = None,
    paper_id: str = "paper-1",
    abstract: str | None = None,
) -> str:
    repository = PaperRepository(database)
    paper = repository.save_paper(
        Paper(paper_id, f"Title {paper_id}", doi=doi, abstract=abstract)
    )
    repository.upsert_source(PaperSource(
        f"source-{paper_id}", paper.paper_id, "publisher", paper_id,
        landing_url="https://publisher.example/paper",
        pdf_url="https://publisher.example/paper.pdf",
        publication_version=PublicationVersion.PUBLISHED,
        license=license,
        access_basis=access_basis,
        host_type="official",
    ))
    return paper.paper_id


def test_filter_selection_defaults_to_relevant_and_requires_explicit_review_opt_in(
    tmp_path: Path, database: Database,
) -> None:
    relevant = _paper(
        database,
        access_basis=AccessBasis.OPEN_LICENSE,
        license="CC-BY-4.0",
        paper_id="paper-relevant",
    )
    review = _paper(
        database,
        access_basis=AccessBasis.OPEN_LICENSE,
        license="CC-BY-4.0",
        paper_id="paper-review",
    )
    database.connection.execute(
        """INSERT INTO pipeline_runs(
               run_id, stage, status, input_hash, config_hash, implementation_version
           ) VALUES ('stage2-run', 'stage-2', 'complete', 'input', 'config', 'test')"""
    )
    database.connection.executemany(
        """INSERT INTO filter_decisions(
               filter_decision_id, run_id, paper_id, status, threshold_version,
               reason, input_hash, implementation_version
           ) VALUES (?, 'stage2-run', ?, ?, 'v1', 'selected', 'input', 'test')""",
        (
            ("decision-relevant", relevant, FilterStatus.RELEVANT.value),
            ("decision-review", review, FilterStatus.NEEDS_REVIEW.value),
        ),
    )
    database.connection.commit()
    service = _service(tmp_path, database, Fetcher(), terms=_terms())

    selected = service.select_papers(filter_run_id="stage2-run")
    expanded = service.select_papers(
        filter_run_id="stage2-run", include_needs_review=True
    )

    assert [item.paper.paper_id for item in selected] == [relevant]
    assert [item.paper.paper_id for item in expanded] == [relevant, review]


def test_filter_selection_rejects_non_complete_or_non_stage2_run(
    tmp_path: Path, database: Database,
) -> None:
    database.connection.execute(
        """INSERT INTO pipeline_runs(
               run_id, stage, status, input_hash, config_hash, implementation_version
           ) VALUES ('wrong-run', 'search', 'complete', 'input', 'config', 'test')"""
    )
    database.connection.commit()

    with pytest.raises(ValueError, match="complete Stage 2"):
        _service(tmp_path, database, Fetcher()).select_papers(
            filter_run_id="wrong-run"
        )


def test_complete_empty_filter_run_selects_no_stage3_papers(
    tmp_path: Path, database: Database,
) -> None:
    database.connection.execute(
        """INSERT INTO pipeline_runs(
               run_id, stage, status, input_hash, config_hash, implementation_version
           ) VALUES ('empty-stage2', 'stage-2', 'complete', 'input', 'config', 'test')"""
    )
    database.connection.commit()

    service = _service(tmp_path, database, Fetcher())
    assert service.select_papers(filter_run_id="empty-stage2") == ()
    result = service.run(filter_run_id="empty-stage2", run_id="empty-stage3")
    assert result.status == "complete"
    assert result.paper_ids == ()
    assert database.connection.execute(
        "SELECT stage, status FROM pipeline_runs WHERE run_id = 'empty-stage3'"
    ).fetchone()[:] == ("stage-3-download", "complete")


def test_no_grant_never_fetches_restricted_candidate(
    tmp_path: Path, database: Database,
) -> None:
    paper_id = _paper(database, access_basis=AccessBasis.PUBLIC_READ_ONLY, license=None)
    fetcher = Fetcher()

    result = _service(tmp_path, database, fetcher, terms=_terms()).run(
        paper_ids=[paper_id]
    )

    assert result.status == "manual_required"
    assert fetcher.calls == []
    assert database.connection.execute(
        "SELECT COUNT(*) FROM manual_queue WHERE queue_type = 'download'"
    ).fetchone()[0] == 1


def test_dry_run_probes_and_validates_without_persisting_or_fetching(
    tmp_path: Path, database: Database,
) -> None:
    paper_id = _paper(database, access_basis=AccessBasis.OPEN_LICENSE, license="CC-BY-4.0")
    fetcher = Fetcher()

    result = _service(tmp_path, database, fetcher, terms=_terms()).run(
        paper_ids=[paper_id], dry_run=True
    )

    assert result.status == "validated"
    assert result.planned_decisions == ((paper_id, "public_direct", "allow"),)
    assert fetcher.calls == []
    assert database.connection.execute("SELECT COUNT(*) FROM pipeline_runs").fetchone()[0] == 0
    assert database.connection.execute("SELECT COUNT(*) FROM download_candidates").fetchone()[0] == 0
    assert database.connection.execute("SELECT COUNT(*) FROM fetch_requests").fetchone()[0] == 0
    assert not (tmp_path / "output" / "artifacts").exists()


def test_repeat_run_reuses_consumed_fetch_request_and_stage2_selection(
    tmp_path: Path, database: Database,
) -> None:
    paper_id = _paper(database, access_basis=AccessBasis.OPEN_LICENSE, license="CC-BY-4.0")
    database.connection.execute(
        """INSERT INTO pipeline_runs(run_id, stage, status, input_hash, config_hash, implementation_version)
           VALUES ('stage2-run', 'stage-2', 'complete', 'input', 'config', 'test')"""
    )
    database.connection.execute(
        """INSERT INTO filter_decisions(
               filter_decision_id, run_id, paper_id, status, threshold_version, reason,
               input_hash, implementation_version
           ) VALUES ('decision-1', 'stage2-run', ?, ?, 'v1', 'included', 'input', 'test')""",
        (paper_id, FilterStatus.RELEVANT.value),
    )
    database.connection.commit()
    fetcher = Fetcher()
    service = _service(tmp_path, database, fetcher, terms=_terms())

    first = service.run(filter_run_id="stage2-run")
    second = service.run(filter_run_id="stage2-run")

    assert first.status == second.status == "complete"
    assert first.run_id == second.run_id
    assert fetcher.calls == ["https://publisher.example/paper.pdf"]


def test_resumed_download_backfills_missing_stage3_paper_checkpoint(
    tmp_path: Path, database: Database,
) -> None:
    paper_id = _paper(
        database,
        access_basis=AccessBasis.OPEN_LICENSE,
        license="CC-BY-4.0",
    )
    fetcher = Fetcher()
    service = _service(tmp_path, database, fetcher, terms=_terms())

    first = service.run(paper_ids=[paper_id])
    database.connection.execute(
        "DELETE FROM stage3_paper_results WHERE run_id = ? AND paper_id = ?",
        (first.run_id, paper_id),
    )
    database.connection.commit()

    resumed = service.run(paper_ids=[paper_id])

    assert resumed.status == "complete"
    assert resumed.run is not None
    assert resumed.run.for_paper(paper_id).resumed is True
    assert fetcher.calls == ["https://publisher.example/paper.pdf"]
    checkpoint = database.connection.execute(
        """SELECT status, reason_code FROM stage3_paper_results
           WHERE run_id = ? AND paper_id = ?""",
        (first.run_id, paper_id),
    ).fetchone()
    assert tuple(checkpoint) == ("downloaded", "downloaded")


@pytest.mark.parametrize(
    ("http_status", "paper_status"),
    (
        (404, "not_available"),
        (400, "failed_terminal"),
    ),
)
def test_terminal_no_pdf_result_completes_and_resumes_without_refetch(
    tmp_path: Path,
    database: Database,
    http_status: int,
    paper_status: str,
) -> None:
    paper_id = _paper(
        database,
        access_basis=AccessBasis.OPEN_LICENSE,
        license="CC-BY-4.0",
        abstract="Public abstract",
    )

    class TerminalFetcher:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, url: str) -> HTTPResponse:
            self.calls += 1
            return HTTPResponse(http_status, {"Content-Type": "text/plain"}, b"", url)

    fetcher = TerminalFetcher()
    service = _service(tmp_path, database, fetcher, terms=_terms())

    first = service.run(paper_ids=[paper_id])
    second = service.run(paper_ids=[paper_id])

    assert first.status == second.status == "complete"
    assert first.run is not None and second.run is not None
    assert first.run.for_paper(paper_id).status.value == paper_status
    assert second.run.for_paper(paper_id).status.value == paper_status
    assert second.run.for_paper(paper_id).resumed is True
    assert fetcher.calls == 1
    checkpoint = database.connection.execute(
        """SELECT status, reason_code FROM stage3_paper_results
           WHERE run_id = ? AND paper_id = ?""",
        (first.run_id, paper_id),
    ).fetchone()
    assert checkpoint["status"] == paper_status
    assert database.connection.execute(
        "SELECT status FROM pipeline_runs WHERE run_id = ?", (first.run_id,)
    ).fetchone()[0] == "complete"
    assert database.connection.execute(
        "SELECT implementation_version FROM pipeline_runs WHERE run_id = ?",
        (first.run_id,),
    ).fetchone()[0] == "stage3-cli-v2"
    assert database.connection.execute(
        "SELECT COUNT(*) FROM download_attempts WHERE run_id = ?",
        (first.run_id,),
    ).fetchone()[0] == 1


def test_retryable_pdf_failure_is_retried_in_same_run(
    tmp_path: Path, database: Database,
) -> None:
    paper_id = _paper(
        database,
        access_basis=AccessBasis.OPEN_LICENSE,
        license="CC-BY-4.0",
    )

    class RetryFetcher:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, url: str) -> HTTPResponse:
            self.calls += 1
            if self.calls == 1:
                return HTTPResponse(503, {}, b"", url)
            return HTTPResponse(
                200, {"Content-Type": "application/pdf"}, _pdf(), url
            )

    fetcher = RetryFetcher()
    service = _service(tmp_path, database, fetcher, terms=_terms())

    first = service.run(paper_ids=[paper_id])
    second = service.run(paper_ids=[paper_id])

    assert first.status == "incomplete"
    assert second.status == "complete"
    assert second.run is not None
    assert second.run.for_paper(paper_id).status.value == "downloaded"
    assert second.run.for_paper(paper_id).resumed is False
    assert fetcher.calls == 2


def test_authorized_skill_handoff_writes_queue_then_imports_only_staged_ledger(
    tmp_path: Path, database: Database, monkeypatch
) -> None:
    paper_id = _paper(
        database,
        access_basis=AccessBasis.PUBLIC_READ_ONLY,
        license=None,
        doi="10.1038/example",
    )
    _approved_authorized_grant(database, paper_id)
    installed = tmp_path / "installed-skill"
    (installed / "scripts").mkdir(parents=True)
    (installed / "scripts" / "paper_queue.py").write_text("# fixture\n", encoding="utf-8")
    ready = SimpleNamespace(
        ready=True,
        installed_path=installed,
        installed_content_sha256="a" * 64,
        dependency_lock_sha256="b" * 64,
    )

    class Runtime:
        def __init__(self, **_kwargs) -> None:
            pass

        def require_ready(self):
            return ready

    monkeypatch.setattr("paper_agent.download_cli_service.AuthorizedSkillRuntime", Runtime)
    terms = {"public_direct": _terms(), "authorized_skill": ProviderTerms(
        "authorized_skill", "test-v1", "https://publisher.example/terms", True, True, True,
        domain_allowlist=("publisher.example",),
    )}
    planner = FakeAuthorizedLunaPlanner(calls=[])
    service = _service(
        tmp_path, database, Fetcher(), terms=None, authorized=True, planner=planner,
    )
    service.provider_terms.update(terms)
    options = AuthorizedSkillHandoffOptions(
        queue_path=tmp_path / "handoff" / "papers.csv",
        output_dir=tmp_path / "handoff-results",
        skill_roots=(installed,),
    )

    waiting = service.run(
        paper_ids=[paper_id], authorization_grant_id="download-grant",
        authorized_skill=options,
    )

    assert waiting.status == "manual_required"
    assert len(planner.calls) == 1
    assert options.queue_path.is_file()
    assert "10.1038/example" in options.queue_path.read_text(encoding="utf-8")
    _stage_authorized_article(options.output_dir)

    resumed = service.run(
        paper_ids=[paper_id], authorization_grant_id="download-grant",
        authorized_skill=options,
    )

    assert resumed.status == "complete"
    assert database.connection.execute(
        "SELECT COUNT(*) FROM artifacts WHERE artifact_kind = 'pdf'"
    ).fetchone()[0] == 1
    assert len(planner.calls) == 1
    decision = database.connection.execute(
        """SELECT status, selected, reason_code, invocation_metadata_json
           FROM stage3_luna_decisions"""
    ).fetchone()
    assert tuple(decision[:3]) == ("complete", 1, "authorized_handoff_selected")
    metadata = json.loads(decision["invocation_metadata_json"])
    assert metadata["actual_model"] == "gpt-5.6-luna"
    assert metadata["actual_profile"] == "stage3_authorized_luna"
    attempt = database.connection.execute(
        "SELECT planner_decision_id FROM download_attempts WHERE result_status = 'downloaded'"
    ).fetchone()
    assert attempt["planner_decision_id"].startswith("stage3-luna-")


def test_authorized_luna_manual_decision_is_durable(tmp_path: Path, database: Database, monkeypatch) -> None:
    paper_id = _paper(
        database,
        access_basis=AccessBasis.PUBLIC_READ_ONLY,
        license=None,
        doi="10.1038/example",
    )
    _approved_authorized_grant(database, paper_id)
    installed = tmp_path / "installed-skill"
    (installed / "scripts").mkdir(parents=True)
    (installed / "scripts" / "paper_queue.py").write_text("# fixture\n", encoding="utf-8")
    ready = SimpleNamespace(
        ready=True,
        installed_path=installed,
        installed_content_sha256="a" * 64,
        dependency_lock_sha256="b" * 64,
    )

    class Runtime:
        def __init__(self, **_kwargs) -> None:
            pass

        def require_ready(self):
            return ready

    monkeypatch.setattr("paper_agent.download_cli_service.AuthorizedSkillRuntime", Runtime)
    planner = FakeAuthorizedLunaPlanner(selected=False, calls=[])
    service = _service(tmp_path, database, Fetcher(), terms=None, authorized=True, planner=planner)
    service.provider_terms["authorized_skill"] = ProviderTerms(
        "authorized_skill", "test-v1", "https://publisher.example/terms", True, True, True,
        domain_allowlist=("publisher.example",),
    )
    options = AuthorizedSkillHandoffOptions(
        queue_path=tmp_path / "handoff" / "papers.csv",
        output_dir=tmp_path / "handoff-results",
        skill_roots=(installed,),
    )

    first = service.run(
        paper_ids=[paper_id], authorization_grant_id="download-grant", authorized_skill=options,
    )
    second = service.run(
        paper_ids=[paper_id], authorization_grant_id="download-grant", authorized_skill=options,
    )

    assert first.status == second.status == "manual_required"
    assert len(planner.calls) == 1
    decision = database.connection.execute(
        "SELECT status, selected, reason_code FROM stage3_luna_decisions"
    ).fetchone()
    assert tuple(decision) == ("complete", 0, "authorized_handoff_deferred")


def test_download_service_uses_injected_clock_for_grant_expiry(
    tmp_path: Path, database: Database,
) -> None:
    paper_id = _paper(
        database, access_basis=AccessBasis.PUBLIC_READ_ONLY, license=None, doi="10.1038/example",
    )
    _approved_authorized_grant(database, paper_id)
    service = _service(
        tmp_path, database, Fetcher(), terms=_terms(),
        clock=lambda: datetime.fromisoformat("2026-08-12T00:00:00+00:00"),
    )
    with pytest.raises(ValueError, match="expired"):
        service.run(paper_ids=[paper_id], authorization_grant_id="download-grant")


@pytest.mark.parametrize("field, value", [
    ("machine_readable", "false"),
    ("allows_download", 1),
    ("allows_storage", "true"),
    ("allows_redistribution", []),
])
def test_provider_terms_rejects_coerced_boolean_values(
    tmp_path: Path, field: str, value: object,
) -> None:
    document = {
        "schema_version": "1",
        "providers": {
            "public_direct": {
                "terms_version": "v1",
                "evidence_url": "https://publisher.example/terms",
                "machine_readable": True,
                "allows_download": True,
                "allows_storage": True,
                "allows_redistribution": None,
                "domain_allowlist": ["publisher.example"],
            }
        },
    }
    document["providers"]["public_direct"][field] = value
    path = tmp_path / "provider-terms.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="boolean"):
        load_provider_terms(path)


def test_provider_terms_requires_explicit_nullable_permission_fields(tmp_path: Path) -> None:
    path = tmp_path / "provider-terms.json"
    path.write_text(json.dumps({
        "schema_version": "1",
        "providers": {
            "public_direct": {
                "terms_version": "v1",
                "evidence_url": None,
                "machine_readable": True,
                "allows_download": True,
                "allows_storage": True,
                "domain_allowlist": ["publisher.example"],
            }
        },
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        load_provider_terms(path)


def _approved_authorized_grant(database: Database, paper_id: str) -> None:
    from paper_agent.grants import GrantStore

    store = GrantStore(database)
    draft = store.create_draft(
        grant_id="download-grant",
        kind="download",
        actions=["download", "store"],
        purpose="personal_research",
        mode="attended",
        scope={
            "paper_ids": [paper_id], "artifact_hashes": [], "collection_ids": [],
            "collection_snapshot_hash": None, "selection_snapshot_hash": None,
            "domains": ["publisher.example"], "provider": "authorized_skill",
            "model": None, "data_categories": ["full_text"],
        },
        max_papers=1,
        expires_at="2026-08-11T00:00:00Z",
        skill_digest="a" * 64,
        dependency_digest="b" * 64,
    )
    store.approve(
        draft, draft["content_hash"], approved_by="owner", approved_at=NOW
    )


def _stage_authorized_article(output_dir: Path) -> None:
    payload = _pdf()
    article = output_dir / "nature" / "0001_10.1038_example" / "article.pdf"
    article.parent.mkdir(parents=True)
    article.write_bytes(payload)
    ledger = output_dir / "_state" / "ledger.jsonl"
    ledger.parent.mkdir(parents=True)
    ledger.write_text(
        '{"doi":"10.1038/example","status":"complete_no_si","files":['
        '{"name":"article.pdf","sha256":"' + sha256(payload).hexdigest() + '"}]}\n',
        encoding="utf-8",
    )


def _pdf() -> bytes:
    objects = (
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>",
    )
    payload = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, body in enumerate(objects, 1):
        offsets.append(len(payload))
        payload.extend(f"{number} 0 obj\n".encode())
        payload.extend(body + b"\nendobj\n")
    xref = len(payload)
    payload.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    payload.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        payload.extend(f"{offset:010d} 00000 n \n".encode())
    payload.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )
    return bytes(payload)
