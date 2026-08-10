from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
import csv
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
from paper_agent.grants import GrantError
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
    url: str = "https://publisher.example/paper.pdf",
) -> str:
    repository = PaperRepository(database)
    paper = repository.save_paper(
        Paper(paper_id, f"Title {paper_id}", doi=doi, abstract=abstract)
    )
    repository.upsert_source(PaperSource(
        f"source-{paper_id}", paper.paper_id, "publisher", paper_id,
        landing_url=url.removesuffix(".pdf"),
        pdf_url=url,
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


def test_public_download_grant_runs_only_after_the_oa_pass(
    tmp_path: Path, database: Database,
) -> None:
    paper_id = _paper(
        database,
        access_basis=AccessBasis.PUBLIC_READ_ONLY,
        license=None,
    )
    _approved_authorized_grant(
        database, paper_id, provider="public_direct",
    )
    fetcher = Fetcher()

    result = _service(tmp_path, database, fetcher, terms=_terms()).run(
        paper_ids=[paper_id], authorization_grant_id="download-grant",
    )

    assert result.status == "complete"
    assert fetcher.calls == ["https://publisher.example/paper.pdf"]
    assert database.connection.execute(
        "SELECT COUNT(*) FROM manual_queue WHERE queue_type = 'download'"
    ).fetchone()[0] == 0


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


def test_dry_run_rejects_a_supplied_unknown_authorized_grant(
    tmp_path: Path, database: Database,
) -> None:
    paper_id = _paper(
        database,
        access_basis=AccessBasis.PUBLIC_READ_ONLY,
        license=None,
        doi="10.1038/example",
        url="https://www.nature.com/articles/example.pdf",
    )
    options = AuthorizedSkillHandoffOptions(
        queue_path=tmp_path / "handoff" / "papers.csv",
        output_dir=tmp_path / "handoff-results",
        skill_roots=(tmp_path / "missing-skill",),
    )

    with pytest.raises(GrantError, match="not found"):
        _service(tmp_path, database, Fetcher(), authorized=True).run(
            paper_ids=[paper_id],
            authorization_grant_id="missing-grant",
            authorized_skill=options,
            dry_run=True,
        )
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
    ).fetchone()[0] == "stage3-cli-v3"
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
        url="https://www.nature.com/articles/example.pdf",
    )
    _approved_authorized_grant(database, paper_id, domain="www.nature.com")
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
    terms = {"public_direct": ProviderTerms(
        "public_direct", "test-v1", "https://www.nature.com/info/tandc.html",
        True, True, True, domain_allowlist=("www.nature.com",),
    ), "authorized_skill": ProviderTerms(
        "authorized_skill", "test-v1", "https://publisher.example/terms", True, True, True,
        domain_allowlist=("www.nature.com",),
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
        url="https://www.nature.com/articles/example.pdf",
    )
    _approved_authorized_grant(database, paper_id, domain="www.nature.com")
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
        domain_allowlist=("www.nature.com",),
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


def test_authorized_queue_is_created_after_public_pass_excludes_public_success_and_resumes(
    tmp_path: Path, database: Database, monkeypatch
) -> None:
    queue_path = tmp_path / "handoff" / "papers.csv"
    open_paper = _paper(
        database,
        access_basis=AccessBasis.OPEN_LICENSE,
        license="CC-BY-4.0",
        doi="10.1038/open",
        paper_id="paper-open",
        url="https://www.nature.com/articles/open.pdf",
    )
    restricted_paper = _paper(
        database,
        access_basis=AccessBasis.PUBLIC_READ_ONLY,
        license=None,
        doi="10.1038/restricted",
        paper_id="paper-restricted",
        url="https://www.nature.com/articles/restricted.pdf",
    )
    _approved_authorized_grant(
        database,
        (open_paper, restricted_paper),
        domain="www.nature.com",
        max_papers=2,
    )
    installed = _ready_authorized_runtime(tmp_path, monkeypatch)

    class QueueObservingFetcher(Fetcher):
        def __call__(self, url: str) -> HTTPResponse:
            assert not queue_path.exists()
            return super().__call__(url)

    fetcher = QueueObservingFetcher()
    planner = FakeAuthorizedLunaPlanner(calls=[])
    service = _service(
        tmp_path, database, fetcher, authorized=True, planner=planner,
    )
    service.provider_terms.update(_nature_terms())
    options = AuthorizedSkillHandoffOptions(
        queue_path=queue_path,
        output_dir=tmp_path / "handoff-results",
        skill_roots=(installed,),
    )

    waiting = service.run(
        paper_ids=[open_paper, restricted_paper],
        authorization_grant_id="download-grant",
        authorized_skill=options,
    )

    assert waiting.status == "manual_required"
    with queue_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["paper_id"] for row in rows] == [restricted_paper]
    assert fetcher.calls == ["https://www.nature.com/articles/open.pdf"]
    database.connection.execute(
        """INSERT INTO download_candidates(
               candidate_id, paper_id, resolver, url, landing_url,
               publication_version, host, license, access_basis, retrieved_at,
               raw_evidence_hash, provenance_json
           ) SELECT 'candidate-duplicate', paper_id, 'unpaywall', url, landing_url,
                    publication_version, host, license, access_basis, retrieved_at,
                    raw_evidence_hash, provenance_json
             FROM download_candidates
             WHERE candidate_id = (
                 SELECT candidate_id FROM authorized_download_queue_reservations
                 WHERE paper_id = ?
             )""",
        (restricted_paper,),
    )
    database.connection.commit()
    _stage_authorized_article(options.output_dir, doi="10.1038/restricted")

    resumed = service.run(
        paper_ids=[open_paper, restricted_paper],
        authorization_grant_id="download-grant",
        authorized_skill=options,
    )

    assert resumed.status == "complete"
    assert resumed.run is not None
    assert resumed.run.for_paper(open_paper).resumed is True
    assert resumed.run.for_paper(restricted_paper).status.value == "downloaded"
    assert fetcher.calls == ["https://www.nature.com/articles/open.pdf"]
    manual = database.connection.execute(
        "SELECT status, resolution_json FROM manual_queue WHERE paper_id = ?",
        (restricted_paper,),
    ).fetchone()
    assert manual["status"] == "resolved"
    assert json.loads(manual["resolution_json"])["status"] == "downloaded"


def test_authorized_queue_resume_keeps_completed_rows_in_the_immutable_csv(
    tmp_path: Path, database: Database, monkeypatch
) -> None:
    paper_a = _paper(
        database,
        access_basis=AccessBasis.PUBLIC_READ_ONLY,
        license=None,
        doi="10.1038/a",
        paper_id="paper-a",
        url="https://www.nature.com/articles/a.pdf",
    )
    paper_b = _paper(
        database,
        access_basis=AccessBasis.PUBLIC_READ_ONLY,
        license=None,
        doi="10.1038/b",
        paper_id="paper-b",
        url="https://www.nature.com/articles/b.pdf",
    )
    _approved_authorized_grant(
        database, (paper_a, paper_b), domain="www.nature.com", max_papers=2,
    )
    installed = _ready_authorized_runtime(tmp_path, monkeypatch)
    planner = FakeAuthorizedLunaPlanner(calls=[])
    service = _service(
        tmp_path, database, Fetcher(), authorized=True, planner=planner,
    )
    service.provider_terms.update(_nature_terms())
    options = AuthorizedSkillHandoffOptions(
        queue_path=tmp_path / "handoff" / "papers.csv",
        output_dir=tmp_path / "handoff-results",
        skill_roots=(installed,),
    )

    waiting = service.run(
        paper_ids=[paper_a, paper_b],
        authorization_grant_id="download-grant",
        authorized_skill=options,
    )
    frozen_csv = options.queue_path.read_text(encoding="utf-8")
    _stage_authorized_article(options.output_dir, doi="10.1038/a", index=1)

    partial = service.run(
        paper_ids=[paper_a, paper_b],
        authorization_grant_id="download-grant",
        authorized_skill=options,
    )
    _stage_authorized_article(
        options.output_dir, doi="10.1038/b", index=2, append=True,
    )
    completed = service.run(
        paper_ids=[paper_a, paper_b],
        authorization_grant_id="download-grant",
        authorized_skill=options,
    )

    assert waiting.status == partial.status == "manual_required"
    assert completed.status == "complete"
    assert completed.run is not None
    assert completed.run.for_paper(paper_a).resumed is True
    assert completed.run.for_paper(paper_b).status.value == "downloaded"
    assert options.queue_path.read_text(encoding="utf-8") == frozen_csv
    assert {
        row["paper_id"]: row["status"]
        for row in database.connection.execute(
            "SELECT paper_id, status FROM manual_queue"
        )
    } == {paper_a: "resolved", paper_b: "resolved"}


@pytest.mark.parametrize(
    ("grant_paper", "grant_domain", "url", "grant_provider"),
    (
        (
            "paper-outside",
            "www.nature.com",
            "https://www.nature.com/articles/example.pdf",
            "authorized_skill",
        ),
        (
            "paper-1",
            "pubs.acs.org",
            "https://www.nature.com/articles/example.pdf",
            "authorized_skill",
        ),
        (
            "paper-1",
            "onlinelibrary.wiley.com",
            "https://onlinelibrary.wiley.com/doi/pdf/10.1038/example",
            "authorized_skill",
        ),
        (
            "paper-1",
            "www.nature.com",
            "https://www.nature.com/articles/example.pdf",
            "unpaywall_location",
        ),
    ),
)
def test_authorized_queue_rejects_scope_domain_provider_and_doi_publisher_mismatches(
    tmp_path: Path,
    database: Database,
    monkeypatch,
    grant_paper: str,
    grant_domain: str,
    url: str,
    grant_provider: str,
) -> None:
    paper_id = _paper(
        database,
        access_basis=AccessBasis.PUBLIC_READ_ONLY,
        license=None,
        doi="10.1038/example",
        url=url,
    )
    _approved_authorized_grant(
        database,
        grant_paper,
        domain=grant_domain,
        provider=grant_provider,
    )
    installed = _ready_authorized_runtime(tmp_path, monkeypatch)
    planner = FakeAuthorizedLunaPlanner(calls=[])
    service = _service(
        tmp_path, database, Fetcher(), authorized=True, planner=planner,
    )
    host = "onlinelibrary.wiley.com" if "wiley.com" in url else "www.nature.com"
    service.provider_terms.update(_nature_terms(host=host))
    options = AuthorizedSkillHandoffOptions(
        queue_path=tmp_path / "handoff" / "papers.csv",
        output_dir=tmp_path / "handoff-results",
        skill_roots=(installed,),
    )

    result = service.run(
        paper_ids=[paper_id],
        authorization_grant_id="download-grant",
        authorized_skill=options,
    )

    assert result.status == "manual_required"
    assert not options.queue_path.exists()
    assert planner.calls == []


def test_authorized_queue_enforces_grant_max_papers_before_planning(
    tmp_path: Path, database: Database, monkeypatch
) -> None:
    paper_a = _paper(
        database,
        access_basis=AccessBasis.PUBLIC_READ_ONLY,
        license=None,
        doi="10.1038/a",
        paper_id="paper-a",
        url="https://www.nature.com/articles/a.pdf",
    )
    paper_b = _paper(
        database,
        access_basis=AccessBasis.PUBLIC_READ_ONLY,
        license=None,
        doi="10.1038/b",
        paper_id="paper-b",
        url="https://www.nature.com/articles/b.pdf",
    )
    _approved_authorized_grant(
        database, (paper_a, paper_b), domain="www.nature.com", max_papers=1,
    )
    installed = _ready_authorized_runtime(tmp_path, monkeypatch)
    planner = FakeAuthorizedLunaPlanner(selected=False, calls=[])
    service = _service(
        tmp_path, database, Fetcher(), authorized=True, planner=planner,
    )
    service.provider_terms.update(_nature_terms())
    options = AuthorizedSkillHandoffOptions(
        queue_path=tmp_path / "handoff" / "papers.csv",
        output_dir=tmp_path / "handoff-results",
        skill_roots=(installed,),
    )

    result = service.run(
        paper_ids=[paper_b, paper_a],
        authorization_grant_id="download-grant",
        authorized_skill=options,
    )

    assert result.status == "manual_required"
    with options.queue_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["paper_id"] for row in rows] == [paper_a]
    assert len(planner.calls) == 1


def test_authorized_queue_reserves_grant_capacity_across_runs(
    tmp_path: Path, database: Database, monkeypatch
) -> None:
    paper_a = _paper(
        database,
        access_basis=AccessBasis.PUBLIC_READ_ONLY,
        license=None,
        doi="10.1038/a",
        paper_id="paper-a",
        url="https://www.nature.com/articles/a.pdf",
    )
    paper_b = _paper(
        database,
        access_basis=AccessBasis.PUBLIC_READ_ONLY,
        license=None,
        doi="10.1038/b",
        paper_id="paper-b",
        url="https://www.nature.com/articles/b.pdf",
    )
    _approved_authorized_grant(
        database, (paper_a, paper_b), domain="www.nature.com", max_papers=1,
    )
    installed = _ready_authorized_runtime(tmp_path, monkeypatch)
    service = _service(
        tmp_path,
        database,
        Fetcher(),
        authorized=True,
        planner=FakeAuthorizedLunaPlanner(calls=[]),
    )
    service.provider_terms.update(_nature_terms())
    first_options = AuthorizedSkillHandoffOptions(
        queue_path=tmp_path / "handoff-a" / "papers.csv",
        output_dir=tmp_path / "results-a",
        skill_roots=(installed,),
    )
    second_options = AuthorizedSkillHandoffOptions(
        queue_path=tmp_path / "handoff-b" / "papers.csv",
        output_dir=tmp_path / "results-b",
        skill_roots=(installed,),
    )

    first = service.run(
        paper_ids=[paper_a],
        authorization_grant_id="download-grant",
        authorized_skill=first_options,
    )
    second = service.run(
        paper_ids=[paper_b],
        authorization_grant_id="download-grant",
        authorized_skill=second_options,
    )

    assert first.status == second.status == "manual_required"
    assert first_options.queue_path.is_file()
    assert not second_options.queue_path.exists()
    reservations = database.connection.execute(
        """SELECT paper_id FROM authorized_download_queue_reservations
           WHERE authorization_grant_id = 'download-grant'"""
    ).fetchall()
    assert [row["paper_id"] for row in reservations] == [paper_a]


def test_authorized_queue_requires_terms_permission_before_browser_handoff(
    tmp_path: Path, database: Database, monkeypatch
) -> None:
    paper_id = _paper(
        database,
        access_basis=AccessBasis.PUBLIC_READ_ONLY,
        license=None,
        doi="10.1038/example",
        url="https://www.nature.com/articles/example.pdf",
    )
    _approved_authorized_grant(
        database, paper_id, domain="www.nature.com",
    )
    installed = _ready_authorized_runtime(tmp_path, monkeypatch)
    service = _service(tmp_path, database, Fetcher(), authorized=True)
    service.provider_terms["authorized_skill"] = ProviderTerms(
        "authorized_skill",
        "test-v1",
        "https://www.nature.com/info/tandc.html",
        True,
        False,
        False,
        domain_allowlist=("www.nature.com",),
    )
    options = AuthorizedSkillHandoffOptions(
        queue_path=tmp_path / "handoff" / "papers.csv",
        output_dir=tmp_path / "handoff-results",
        skill_roots=(installed,),
    )

    result = service.run(
        paper_ids=[paper_id],
        authorization_grant_id="download-grant",
        authorized_skill=options,
    )

    assert result.status == "manual_required"
    assert not options.queue_path.exists()
    assert database.connection.execute(
        "SELECT COUNT(*) FROM authorized_download_queue_reservations"
    ).fetchone()[0] == 0


def test_authorized_queue_rejects_unproven_selection_snapshot_membership(
    tmp_path: Path, database: Database, monkeypatch
) -> None:
    paper_id = _paper(
        database,
        access_basis=AccessBasis.PUBLIC_READ_ONLY,
        license=None,
        doi="10.1038/example",
        url="https://www.nature.com/articles/example.pdf",
    )
    _approved_authorized_grant(
        database,
        None,
        domain="www.nature.com",
        selection_snapshot_hash="c" * 64,
    )
    installed = _ready_authorized_runtime(tmp_path, monkeypatch)
    service = _service(tmp_path, database, Fetcher(), authorized=True)
    service.provider_terms.update(_nature_terms())
    options = AuthorizedSkillHandoffOptions(
        queue_path=tmp_path / "handoff" / "papers.csv",
        output_dir=tmp_path / "handoff-results",
        skill_roots=(installed,),
    )

    result = service.run(
        paper_ids=[paper_id],
        authorization_grant_id="download-grant",
        authorized_skill=options,
    )

    assert result.status == "manual_required"
    assert not options.queue_path.exists()


def test_download_service_uses_injected_clock_for_grant_expiry(
    tmp_path: Path, database: Database, monkeypatch,
) -> None:
    paper_id = _paper(
        database,
        access_basis=AccessBasis.PUBLIC_READ_ONLY,
        license=None,
        doi="10.1038/example",
        url="https://www.nature.com/articles/example.pdf",
    )
    _approved_authorized_grant(database, paper_id, domain="www.nature.com")
    installed = _ready_authorized_runtime(tmp_path, monkeypatch)
    service = _service(
        tmp_path, database, Fetcher(), authorized=True,
        clock=lambda: datetime.fromisoformat("2026-08-12T00:00:00+00:00"),
    )
    service.provider_terms.update(_nature_terms())
    options = AuthorizedSkillHandoffOptions(
        queue_path=tmp_path / "handoff" / "papers.csv",
        output_dir=tmp_path / "handoff-results",
        skill_roots=(installed,),
    )
    result = service.run(
        paper_ids=[paper_id],
        authorization_grant_id="download-grant",
        authorized_skill=options,
    )

    assert result.status == "manual_required"
    assert not options.queue_path.exists()


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


def _ready_authorized_runtime(tmp_path: Path, monkeypatch) -> Path:
    installed = tmp_path / "installed-skill"
    (installed / "scripts").mkdir(parents=True)
    (installed / "scripts" / "paper_queue.py").write_text(
        "# fixture\n", encoding="utf-8"
    )
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

    monkeypatch.setattr(
        "paper_agent.download_cli_service.AuthorizedSkillRuntime", Runtime
    )
    return installed


def _nature_terms(
    *, host: str = "www.nature.com",
) -> dict[str, ProviderTerms]:
    return {
        provider: ProviderTerms(
            provider,
            "test-v1",
            f"https://{host}/terms",
            True,
            True,
            True,
            domain_allowlist=(host,),
        )
        for provider in ("public_direct", "authorized_skill")
    }


def _approved_authorized_grant(
    database: Database,
    paper_id: str | tuple[str, ...] | None,
    *,
    domain: str = "publisher.example",
    max_papers: int = 1,
    selection_snapshot_hash: str | None = None,
    provider: str = "authorized_skill",
) -> None:
    from paper_agent.grants import GrantStore

    store = GrantStore(database)
    draft = store.create_draft(
        grant_id="download-grant",
        kind="download",
        actions=["download", "store"],
        purpose="personal_research",
        mode="attended",
        scope={
            "paper_ids": (
                [paper_id]
                if isinstance(paper_id, str)
                else list(paper_id or ())
            ),
            "artifact_hashes": [], "collection_ids": [],
            "collection_snapshot_hash": None,
            "selection_snapshot_hash": selection_snapshot_hash,
            "domains": [domain], "provider": provider,
            "model": None, "data_categories": ["full_text"],
        },
        max_papers=max_papers,
        expires_at="2026-08-11T00:00:00Z",
        skill_digest="a" * 64 if provider == "authorized_skill" else None,
        dependency_digest="b" * 64 if provider == "authorized_skill" else None,
    )
    store.approve(
        draft, draft["content_hash"], approved_by="owner", approved_at=NOW
    )


def _stage_authorized_article(
    output_dir: Path,
    *,
    doi: str = "10.1038/example",
    index: int = 1,
    append: bool = False,
) -> None:
    payload = _pdf() + f"\n% staged {doi}\n".encode()
    article = (
        output_dir
        / "nature"
        / f"{index:04d}_{doi.replace('/', '_')}"
        / "article.pdf"
    )
    article.parent.mkdir(parents=True)
    article.write_bytes(payload)
    ledger = output_dir / "_state" / "ledger.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a" if append else "w", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "doi": doi,
            "status": "complete_no_si",
            "files": [{
                "name": "article.pdf",
                "sha256": sha256(payload).hexdigest(),
            }],
        }) + "\n")


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
