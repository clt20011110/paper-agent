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
            "label_evidence": [], "evidence_units": [], "comparison_eligibility": "not_comparable", "missing_fields": ["comparison_evidence"],
        }
        metadata = InvocationMetadata("fixture", "stage4_analysis_luna", "gpt-5.6-luna", "medium", "paper-analysis.schema.json", self.schema_hash, request.input_hash, "paper-analysis.md", self.prompt_hash, "rendered", None, 1, "gpt-5.6-luna", "stage4_analysis_luna")
        return CodexExecResult(output, metadata)


def _service(database, store, factory, *, clock=None):
    return AnalysisCliService(
        database, store,
        ArtifactProcessingPolicy.load(ROOT / "policies" / "artifact-processing-v1.yaml"),
        invoker_factory=factory, clock=clock,
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
