from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path

import pytest

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from paper_agent.analysis_cli_service import AnalysisCliService, AnalysisInputManifest
from paper_agent.artifacts import ArtifactStore
from paper_agent.codex_exec import CodexExecResult, InvocationMetadata
from paper_agent.processing import ArtifactProcessingPolicy
from paper_agent.storage import Database


ROOT = Path(__file__).parents[1]


def _pdf(text: str) -> bytes:
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"), NameObject("/BaseFont"): NameObject("/Helvetica")})
    reference = writer._add_object(font)
    page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): reference})})
    stream = DecodedStreamObject()
    stream.set_data(f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode())
    page[NameObject("/Contents")] = writer._add_object(stream)
    output = __import__("io").BytesIO()
    writer.write(output)
    return output.getvalue()


def _encrypted_pdf() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.encrypt("fixture-password")
    output = __import__("io").BytesIO()
    writer.write(output)
    return output.getvalue()


def _prepared(tmp_path, *, license: str | None, access_basis: str):
    database = Database(tmp_path / "papers.sqlite3")
    database.migrate()
    database.connection.execute("INSERT INTO papers(paper_id, title) VALUES ('paper-1', 'One')")
    store = ArtifactStore(tmp_path / "store")
    stored = store.put_bytes(_pdf("alpha " * 80), mime_type="application/pdf", metadata={"fixture": "stage3"})
    database.connection.execute(
        """INSERT INTO artifacts(artifact_id, paper_id, artifact_kind, relative_path, mime_type, byte_size, sha256, provenance_json)
           VALUES ('pdf-1', 'paper-1', 'pdf', ?, 'application/pdf', ?, ?, '{}')""",
        (stored.relative_path, stored.size_bytes, stored.artifact_hash),
    )
    database.connection.execute(
        """INSERT INTO pipeline_runs(run_id, stage, status, input_hash, config_hash, implementation_version)
           VALUES ('stage3-1', 'stage3', 'complete', ?, ?, 'fixture')""",
        ("1" * 64, "2" * 64),
    )
    database.connection.execute(
        """INSERT INTO download_candidates(candidate_id, paper_id, resolver, url, host, license, access_basis, retrieved_at, provenance_json)
           VALUES ('candidate-1', 'paper-1', 'fixture', 'https://example.test/one.pdf', 'example.test', ?, ?, '2026-08-10T00:00:00Z', '{}')""",
        (license, access_basis),
    )
    database.connection.execute(
        """INSERT INTO fetch_requests(request_id, candidate_id, policy_version, policy_hash, purpose, provider, created_at, expires_at, idempotency_key, fencing_token, status)
           VALUES ('fetch-1', 'candidate-1', 'fixture', ?, 'personal_research', 'public_direct', '2026-08-10T00:00:00Z', '2026-08-11T00:00:00Z', 'key-1', 0, 'consumed')""",
        ("3" * 64,),
    )
    database.connection.execute(
        """INSERT INTO download_attempts(download_attempt_id, run_id, candidate_id, provider, fetch_request_id, result_status, artifact_id)
           VALUES ('attempt-1', 'stage3-1', 'candidate-1', 'public_direct', 'fetch-1', 'downloaded', 'pdf-1')"""
    )
    database.connection.commit()
    return database, store


@dataclass
class FakeInvoker:
    prompt_hash: str
    schema_hash: str
    calls: list[object]

    def invoke(self, request):
        self.calls.append(request)
        payload = json.loads(request.prompt)
        output = {
            "paper_id": payload["paper_id"], "artifact_hash": payload["artifact_hash"], "input_scope": payload["input_scope"],
            "model": "gpt-5.6-luna", "model_revision": "fixture", "prompt_hash": self.prompt_hash,
            "schema_hash": self.schema_hash, "created_at": payload["output_binding"]["created_at"],
            "research_question_and_motivation": "Fixture.", "summary": "Fixture.", "methods": [], "key_techniques": [],
            "datasets": [], "experimental_setup": [], "metrics": [], "results": [], "limitations": [], "credibility": "Fixture.",
            "resources": [], "topic_relevance": "Relevant", "labels": {"subquestion": [], "theme": [], "method_family": [], "task": [], "dataset": [], "benchmark": [], "evidence_type": [], "publication_status": "unknown", "study_setting": "other"},
            "label_evidence": [], "evidence_units": [], "comparison_eligibility": "not_comparable",
            "missing_fields": (["full_text"] if payload["input_scope"] != "full_pdf" else ["comparison_evidence"]),
        }
        metadata = InvocationMetadata("fixture", "stage4_analysis_luna", "gpt-5.6-luna", "medium", "paper-analysis.schema.json", self.schema_hash, request.input_hash, "paper-analysis.md", self.prompt_hash, "rendered", None, 1, "gpt-5.6-luna", "stage4_analysis_luna")
        return CodexExecResult(output, metadata)


def _service(database, store, factory, *, clock=None):
    return AnalysisCliService(
        database, store,
        ArtifactProcessingPolicy.load(ROOT / "policies" / "artifact-processing-v1.yaml"),
        invoker_factory=factory, clock=clock,
    )


def _add_pipeline_run(
    database: Database, run_id: str, *, stage: str, status: str
) -> None:
    database.connection.execute(
        """INSERT INTO pipeline_runs(
               run_id, stage, status, input_hash, config_hash, implementation_version
           ) VALUES (?, ?, ?, ?, ?, 'fixture')""",
        (run_id, stage, status, "4" * 64, "5" * 64),
    )


def _add_paper(
    database: Database, paper_id: str, *, abstract: str | None = None
) -> None:
    database.connection.execute(
        "INSERT INTO papers(paper_id, title, abstract) VALUES (?, ?, ?)",
        (paper_id, f"Title for {paper_id}", abstract),
    )


def _add_downloaded_pdf(
    database: Database,
    store: ArtifactStore,
    *,
    run_id: str,
    paper_id: str,
    suffix: str,
    text: str,
) -> str:
    artifact_id = f"pdf-{suffix}"
    candidate_id = f"candidate-{suffix}"
    fetch_request_id = f"fetch-{suffix}"
    stored = store.put_bytes(
        _pdf((text + " ") * 80),
        mime_type="application/pdf",
        metadata={"fixture": suffix},
    )
    database.connection.execute(
        """INSERT INTO artifacts(
               artifact_id, paper_id, artifact_kind, relative_path, mime_type,
               byte_size, sha256, provenance_json
           ) VALUES (?, ?, 'pdf', ?, 'application/pdf', ?, ?, '{}')""",
        (
            artifact_id,
            paper_id,
            stored.relative_path,
            stored.size_bytes,
            stored.artifact_hash,
        ),
    )
    database.connection.execute(
        """INSERT INTO download_candidates(
               candidate_id, paper_id, resolver, url, host, license, access_basis,
               retrieved_at, provenance_json
           ) VALUES (?, ?, 'fixture', ?, 'example.test', 'CC-BY-4.0',
                     'open_license', '2026-08-10T00:00:00Z', '{}')""",
        (candidate_id, paper_id, f"https://example.test/{suffix}.pdf"),
    )
    database.connection.execute(
        """INSERT INTO fetch_requests(
               request_id, candidate_id, policy_version, policy_hash, purpose,
               provider, created_at, expires_at, idempotency_key, fencing_token,
               status
           ) VALUES (?, ?, 'fixture', ?, 'personal_research', 'public_direct',
                     '2026-08-10T00:00:00Z', '2026-08-11T00:00:00Z', ?, 0,
                     'consumed')""",
        (fetch_request_id, candidate_id, "6" * 64, f"key-{suffix}"),
    )
    database.connection.execute(
        """INSERT INTO download_attempts(
               download_attempt_id, run_id, candidate_id, provider,
               fetch_request_id, result_status, artifact_id
           ) VALUES (?, ?, ?, 'public_direct', ?, 'downloaded', ?)""",
        (f"attempt-{suffix}", run_id, candidate_id, fetch_request_id, artifact_id),
    )
    return artifact_id


def _add_old_text(
    database: Database,
    store: ArtifactStore,
    *,
    paper_id: str,
    source_artifact_id: str,
    suffix: str,
) -> None:
    stored = store.put_bytes(
        f"OLD EXTRACTED TEXT MUST NEVER BE SELECTED {suffix}".encode(),
        mime_type="text/plain",
        metadata={"fixture": suffix},
    )
    artifact_id = f"text-{suffix}"
    source_sha256 = database.connection.execute(
        "SELECT sha256 FROM artifacts WHERE artifact_id = ?", (source_artifact_id,)
    ).fetchone()[0]
    database.connection.execute(
        """INSERT INTO artifacts(
               artifact_id, paper_id, artifact_kind, relative_path, mime_type,
               byte_size, sha256, provenance_json
           ) VALUES (?, ?, 'text', ?, 'text/plain', ?, ?, '{}')""",
        (
            artifact_id,
            paper_id,
            stored.relative_path,
            stored.size_bytes,
            stored.artifact_hash,
        ),
    )
    database.connection.execute(
        """INSERT INTO text_extractions(
               extraction_id, paper_id, source_artifact_id, source_sha256,
               output_artifact_id, extractor_name, extractor_version, page_count,
               character_count, text_coverage, printable_ratio, status
           ) VALUES (?, ?, ?, ?, ?, 'fixture', '1', 1, 40, 1.0, 1.0,
                     'full_text_ready')""",
        (
            f"extraction-{suffix}",
            paper_id,
            source_artifact_id,
            source_sha256,
            artifact_id,
        ),
    )


def test_stage3_artifact_without_processing_grant_never_constructs_codex(tmp_path) -> None:
    database, store = _prepared(tmp_path, license=None, access_basis="user_subscription")
    try:
        service = _service(database, store, lambda: (_ for _ in ()).throw(AssertionError("Codex must not be constructed")))
        result = service.run("analysis-denied", AnalysisInputManifest(stage3_artifact_ids=("pdf-1",)))
        assert result.result is not None
        assert result.result.for_paper("paper-1").status == "incomplete"
        assert database.connection.execute("SELECT COUNT(*) FROM analysis_runs").fetchone()[0] == 1
    finally:
        database.close()


def test_disabled_abstract_only_is_manual_before_codex_construction(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "papers.sqlite3")
    database.migrate()
    database.connection.execute(
        """INSERT INTO papers(paper_id, title, abstract)
           VALUES ('paper-1', 'One', 'A public abstract')"""
    )
    database.connection.commit()
    store = ArtifactStore(tmp_path / "store")
    factory_calls = 0

    def forbidden_factory():
        nonlocal factory_calls
        factory_calls += 1
        raise AssertionError("disabled abstract-only input must not construct Codex")

    try:
        service = AnalysisCliService(
            database,
            store,
            ArtifactProcessingPolicy.load(
                ROOT / "policies" / "artifact-processing-v1.yaml"
            ),
            invoker_factory=forbidden_factory,
            allow_abstract_only=False,
        )
        result = service.run(
            "analysis-no-abstract", AnalysisInputManifest(paper_ids=("paper-1",))
        )

        assert result.result is not None
        paper = result.result.for_paper("paper-1")
        assert paper.status == "incomplete"
        assert paper.decision is not None
        assert paper.decision.reason_code == "abstract_only_disabled_by_analysis_config"
        assert factory_calls == 0
        assert database.connection.execute(
            "SELECT dispatch_count FROM analysis_dispatches"
        ).fetchone()[0] == 0
    finally:
        database.close()


def test_dry_run_previews_extraction_without_writes_or_codex(tmp_path, monkeypatch) -> None:
    database, store = _prepared(tmp_path, license=None, access_basis="user_subscription")
    try:
        service = _service(database, store, lambda: (_ for _ in ()).throw(AssertionError("Codex must not be constructed")))
        before = database.connection.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0]
        before_files = sorted(path.relative_to(store.root) for path in store.root.rglob("*"))
        requests = []
        decide = service.gate.decide

        def capture(request, **options):
            requests.append((request, options))
            return decide(request, **options)

        monkeypatch.setattr(service.gate, "decide", capture)
        result = service.run("analysis-dry", AnalysisInputManifest(stage3_artifact_ids=("pdf-1",)), dry_run=True)
        assert result.dry_run
        assert result.input_scopes == ("full_pdf",)
        assert requests[0][0].artifact == "normalized_text"
        assert database.connection.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0] == before
        assert database.connection.execute("SELECT COUNT(*) FROM analysis_runs").fetchone()[0] == 0
        assert database.connection.execute("SELECT COUNT(*) FROM text_extractions").fetchone()[0] == 0
        assert sorted(path.relative_to(store.root) for path in store.root.rglob("*")) == before_files
    finally:
        database.close()


@pytest.mark.parametrize(
    ("pdf_bytes", "extraction_status"),
    ((_pdf("scan"), "needs_ocr"), (_encrypted_pdf(), "extraction_failed")),
)
def test_unusable_pdf_falls_back_to_public_abstract_before_codex(
    tmp_path: Path, pdf_bytes: bytes, extraction_status: str,
) -> None:
    database, store = _prepared(
        tmp_path, license="CC-BY-4.0", access_basis="open_license"
    )
    replacement = store.put_bytes(
        pdf_bytes, mime_type="application/pdf", metadata={"fixture": extraction_status}
    )
    database.connection.execute(
        "UPDATE papers SET abstract = 'Public fallback abstract' WHERE paper_id = 'paper-1'"
    )
    database.connection.execute(
        """UPDATE artifacts
           SET relative_path = ?, byte_size = ?, sha256 = ?
           WHERE artifact_id = 'pdf-1'""",
        (replacement.relative_path, replacement.size_bytes, replacement.artifact_hash),
    )
    database.connection.commit()
    calls: list[object] = []

    try:
        from paper_agent.analysis import PaperAnalysisCoordinator

        template = PaperAnalysisCoordinator(
            database, store, _service(database, store, None).gate
        )
        service = _service(
            database,
            store,
            lambda: FakeInvoker(template.prompt_hash, template.schema_hash, calls),
        )
        result = service.run(
            f"analysis-{extraction_status}",
            AnalysisInputManifest(stage3_artifact_ids=("pdf-1",)),
        )

        assert result.input_scopes == ("abstract_only",)
        assert result.result is not None
        assert result.result.for_paper("paper-1").status == "complete"
        assert len(calls) == 1
        payload = json.loads(calls[0].prompt)
        assert payload["input_scope"] == "abstract_only"
        assert payload["content_encoding"] == "json"
        assert payload["content"]["abstract"] == "Public fallback abstract"
        row = database.connection.execute(
            "SELECT status FROM text_extractions WHERE source_artifact_id = 'pdf-1'"
        ).fetchone()
        assert row["status"] == extraction_status
        assert database.connection.execute(
            "SELECT artifact_id FROM analysis_runs"
        ).fetchone()["artifact_id"] is None
    finally:
        database.close()


def test_analysis_service_uses_injected_clock_and_rejects_call_time_override(tmp_path, monkeypatch) -> None:
    database, store = _prepared(tmp_path, license=None, access_basis="user_subscription")
    trusted = datetime(2026, 8, 10, 12, 0, tzinfo=UTC)
    try:
        service = _service(
            database, store, None, clock=lambda: trusted,
        )
        seen = []
        decide = service.gate.decide

        def capture(request, **options):
            seen.append(options["now"])
            return decide(request, **options)

        monkeypatch.setattr(service.gate, "decide", capture)
        service.run("analysis-clock", AnalysisInputManifest(stage3_artifact_ids=("pdf-1",)), dry_run=True)
        assert seen == [trusted]
        with pytest.raises(TypeError):
            service.run(  # type: ignore[call-arg]
                "analysis-clock", AnalysisInputManifest(stage3_artifact_ids=("pdf-1",)), now="2020-01-01T00:00:00Z"
            )
    finally:
        database.close()


def test_authorized_stage3_artifact_resume_does_not_repeat_codex(tmp_path) -> None:
    database, store = _prepared(tmp_path, license="CC-BY-4.0", access_basis="open_license")
    calls: list[object] = []
    try:
        from paper_agent.analysis import PaperAnalysisCoordinator
        template = PaperAnalysisCoordinator(database, store, _service(database, store, None).gate)
        service = _service(
            database, store,
            lambda: FakeInvoker(template.prompt_hash, template.schema_hash, calls),
        )
        first = service.run("analysis-ok", AnalysisInputManifest(stage3_artifact_ids=("pdf-1",)))
        second = service.run("analysis-ok", AnalysisInputManifest(stage3_artifact_ids=("pdf-1",)))
        assert first.result and first.result.for_paper("paper-1").status == "complete"
        assert second.result and second.result.for_paper("paper-1").resumed
        assert len(calls) == 1
    finally:
        database.close()


@pytest.mark.parametrize(
    ("stage", "status"),
    (("stage-3-download", "running"), ("stage4", "complete")),
)
def test_run_from_stage3_only_accepts_complete_download_runs(
    tmp_path: Path, stage: str, status: str
) -> None:
    database = Database(tmp_path / "papers.sqlite3")
    database.migrate()
    _add_pipeline_run(database, "stage3-source", stage=stage, status=status)
    database.connection.commit()
    factory_calls = 0

    def forbidden_factory():
        nonlocal factory_calls
        factory_calls += 1
        raise AssertionError("invalid Stage 3 lineage must fail before Codex")

    try:
        service = _service(
            database, ArtifactStore(tmp_path / "store"), forbidden_factory
        )
        with pytest.raises(
            ValueError, match="complete Stage 3 download run"
        ):
            service.run_from_stage3(
                "analysis-run", "stage3-source", expected_paper_ids=()
            )
        assert factory_calls == 0
        assert database.connection.execute(
            "SELECT COUNT(*) FROM analysis_runs"
        ).fetchone()[0] == 0
    finally:
        database.close()


def test_run_from_stage3_rejects_missing_expected_checkpoints_before_codex(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "papers.sqlite3")
    database.migrate()
    _add_paper(database, "paper-1", abstract="One")
    _add_paper(database, "paper-2", abstract="Two")
    _add_pipeline_run(
        database, "current-stage3", stage="stage-3-download", status="complete"
    )
    database.connection.execute(
        """INSERT INTO stage3_paper_results(
               run_id, paper_id, status, reason_code, updated_at
           ) VALUES ('current-stage3', 'paper-1', 'not_available',
                     'not_available', '2026-08-10T00:00:00Z')"""
    )
    database.connection.commit()
    factory_calls = 0

    def forbidden_factory():
        nonlocal factory_calls
        factory_calls += 1
        raise AssertionError("incomplete Stage 3 lineage must fail before Codex")

    try:
        service = _service(
            database, ArtifactStore(tmp_path / "store"), forbidden_factory
        )
        with pytest.raises(ValueError, match="checkpoints do not match"):
            service.run_from_stage3(
                "analysis-run",
                "current-stage3",
                expected_paper_ids=("paper-1", "paper-2"),
            )
        assert factory_calls == 0
        assert database.connection.execute(
            "SELECT COUNT(*) FROM analysis_runs"
        ).fetchone()[0] == 0
    finally:
        database.close()


def test_run_from_stage3_uses_only_this_runs_unique_downloaded_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = Database(tmp_path / "papers.sqlite3")
    database.migrate()
    store = ArtifactStore(tmp_path / "store")
    _add_paper(database, "paper-1", abstract="Fallback abstract")
    _add_pipeline_run(
        database, "old-stage3", stage="stage-3-download", status="complete"
    )
    _add_pipeline_run(
        database, "current-stage3", stage="stage-3-download", status="complete"
    )
    old_pdf = _add_downloaded_pdf(
        database,
        store,
        run_id="old-stage3",
        paper_id="paper-1",
        suffix="old",
        text="OLD PDF MUST NEVER BE SELECTED",
    )
    _add_old_text(
        database,
        store,
        paper_id="paper-1",
        source_artifact_id=old_pdf,
        suffix="old",
    )
    current_pdf = _add_downloaded_pdf(
        database,
        store,
        run_id="current-stage3",
        paper_id="paper-1",
        suffix="current",
        text="CURRENT RUN PDF",
    )
    _add_pipeline_run(
        database, "later-stage3", stage="stage-3-download", status="complete"
    )
    database.connection.execute(
        """INSERT INTO download_candidates(
               candidate_id, paper_id, resolver, url, host, license, access_basis,
               retrieved_at, provenance_json
           ) VALUES ('candidate-later', 'paper-1', 'fixture',
                     'https://later.example/paper.pdf', 'later.example', NULL,
                     'user_subscription', '2026-08-10T01:00:00Z', '{}')"""
    )
    database.connection.execute(
        """INSERT INTO fetch_requests(
               request_id, candidate_id, policy_version, policy_hash, purpose,
               provider, created_at, expires_at, idempotency_key, fencing_token,
               status
           ) VALUES ('fetch-later', 'candidate-later', 'fixture', ?,
                     'personal_research', 'public_direct',
                     '2026-08-10T01:00:00Z', '2026-08-11T01:00:00Z',
                     'key-later', 0, 'consumed')""",
        ("7" * 64,),
    )
    database.connection.execute(
        """INSERT INTO download_attempts(
               download_attempt_id, run_id, candidate_id, provider,
               fetch_request_id, result_status, artifact_id, attempted_at
           ) VALUES ('attempt-later', 'later-stage3', 'candidate-later',
                     'public_direct', 'fetch-later', 'downloaded', ?,
                     '2999-01-01T00:00:00Z')""",
        (current_pdf,),
    )
    database.connection.execute(
        """INSERT INTO stage3_paper_results(
               run_id, paper_id, status, reason_code, updated_at
           ) VALUES ('current-stage3', 'paper-1', 'downloaded', 'downloaded',
                     '2026-08-10T00:00:00Z')"""
    )
    database.connection.commit()
    captured = []

    try:
        service = _service(
            database,
            store,
            lambda: (_ for _ in ()).throw(
                AssertionError("dry-run must not construct Codex")
            ),
        )
        decide = service.gate.decide

        def capture(request, **options):
            captured.append(request)
            return decide(request, **options)

        monkeypatch.setattr(service.gate, "decide", capture)
        result = service.run_from_stage3(
            "analysis-run",
            "current-stage3",
            expected_paper_ids=("paper-1",),
            dry_run=True,
        )

        assert result.selected_paper_ids == ("paper-1",)
        assert result.input_scopes == ("full_pdf",)
        assert len(captured) == 1
        assert captured[0].artifact == "normalized_text"
        assert captured[0].license == "CC-BY-4.0"
        assert captured[0].access_basis == "open_license"
        assert captured[0].domain == "example.test"
        normalized = (captured[0].normalized_text_bytes or b"").decode()
        assert "CURRENT RUN PDF" in normalized
        assert "OLD PDF MUST NEVER BE SELECTED" not in normalized
        assert "OLD EXTRACTED TEXT MUST NEVER BE SELECTED" not in normalized
    finally:
        database.close()


def test_run_from_stage3_terminal_outcomes_force_abstract_or_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = Database(tmp_path / "papers.sqlite3")
    database.migrate()
    store = ArtifactStore(tmp_path / "store")
    _add_paper(database, "paper-abstract", abstract="Current public abstract")
    _add_paper(database, "paper-metadata")
    _add_pipeline_run(
        database, "old-stage3", stage="stage-3-download", status="complete"
    )
    _add_pipeline_run(
        database, "current-stage3", stage="stage-3-download", status="complete"
    )
    for paper_id, suffix in (
        ("paper-abstract", "old-abstract"),
        ("paper-metadata", "old-metadata"),
    ):
        old_pdf = _add_downloaded_pdf(
            database,
            store,
            run_id="old-stage3",
            paper_id=paper_id,
            suffix=suffix,
            text=f"OLD PDF FOR {paper_id}",
        )
        _add_old_text(
            database,
            store,
            paper_id=paper_id,
            source_artifact_id=old_pdf,
            suffix=suffix,
        )
    database.connection.executemany(
        """INSERT INTO stage3_paper_results(
               run_id, paper_id, status, reason_code, updated_at
           ) VALUES ('current-stage3', ?, ?, ?, '2026-08-10T00:00:00Z')""",
        (
            ("paper-abstract", "not_available", "not_available"),
            ("paper-metadata", "failed_terminal", "invalid_pdf"),
        ),
    )
    database.connection.commit()
    captured = []

    try:
        service = _service(
            database,
            store,
            lambda: (_ for _ in ()).throw(
                AssertionError("dry-run must not construct Codex")
            ),
        )
        decide = service.gate.decide

        def capture(request, **options):
            captured.append(request)
            return decide(request, **options)

        monkeypatch.setattr(service.gate, "decide", capture)
        result = service.run_from_stage3(
            "analysis-run",
            "current-stage3",
            expected_paper_ids=("paper-abstract", "paper-metadata"),
            dry_run=True,
        )

        assert result.selected_paper_ids == ("paper-abstract", "paper-metadata")
        assert result.input_scopes == ("abstract_only", "metadata_only")
        assert [(item.paper_id, item.artifact) for item in captured] == [
            ("paper-abstract", "abstract"),
            ("paper-metadata", "metadata"),
        ]
        assert all(item.normalized_text_bytes is None for item in captured)
        assert all(item.pdf_bytes is None for item in captured)
    finally:
        database.close()


def test_run_from_stage3_rejects_two_current_downloaded_artifacts_before_codex(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "papers.sqlite3")
    database.migrate()
    store = ArtifactStore(tmp_path / "store")
    _add_paper(database, "paper-1", abstract="Fallback abstract")
    _add_pipeline_run(
        database, "current-stage3", stage="stage-3-download", status="complete"
    )
    for suffix in ("current-a", "current-b"):
        _add_downloaded_pdf(
            database,
            store,
            run_id="current-stage3",
            paper_id="paper-1",
            suffix=suffix,
            text=suffix,
        )
    database.connection.execute(
        """INSERT INTO stage3_paper_results(
               run_id, paper_id, status, reason_code, updated_at
           ) VALUES ('current-stage3', 'paper-1', 'downloaded', 'downloaded',
                     '2026-08-10T00:00:00Z')"""
    )
    database.connection.commit()
    factory_calls = 0

    def forbidden_factory():
        nonlocal factory_calls
        factory_calls += 1
        raise AssertionError("ambiguous Stage 3 lineage must fail before Codex")

    try:
        service = _service(database, store, forbidden_factory)
        with pytest.raises(ValueError, match="exactly one available PDF artifact"):
            service.run_from_stage3(
                "analysis-run",
                "current-stage3",
                expected_paper_ids=("paper-1",),
            )
        assert factory_calls == 0
        assert database.connection.execute(
            "SELECT COUNT(*) FROM analysis_runs"
        ).fetchone()[0] == 0
    finally:
        database.close()
