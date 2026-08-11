from __future__ import annotations

from datetime import UTC, datetime, timedelta
import importlib.util
from io import BytesIO
import json
from pathlib import Path
import sys

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from paper_agent.artifacts import ArtifactStore
from paper_agent.canonical import content_hash
from paper_agent.codex_exec import CodexExecResult, DoctorReport, InvocationMetadata
from paper_agent.storage import Database


NOW = datetime(2026, 8, 12, 0, 0, tzinfo=UTC)
ROOT = Path(__file__).parents[1]
SPECIFICATION = importlib.util.spec_from_file_location(
    "run_venue_stage4_luna", ROOT / "scripts" / "run_venue_stage4_luna.py"
)
assert SPECIFICATION is not None and SPECIFICATION.loader is not None
runner = importlib.util.module_from_spec(SPECIFICATION)
sys.modules[SPECIFICATION.name] = runner
SPECIFICATION.loader.exec_module(runner)


def _pdf(text: str) -> bytes:
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    reference = writer._add_object(font)
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): reference})}
    )
    stream = DecodedStreamObject()
    stream.set_data(f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode())
    page[NameObject("/Contents")] = writer._add_object(stream)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def _prepared_run(tmp_path: Path, *, stage3_status: str = "downloaded") -> Path:
    run_dir = tmp_path / "icml-e2e"
    run_dir.mkdir()
    artifact_root = run_dir / "artifacts"
    store = ArtifactStore(artifact_root)
    database = Database(run_dir / "papers.sqlite3")
    database.migrate()
    stored = store.put_bytes(
        _pdf("normalized venue smoke full text " * 30),
        mime_type="application/pdf",
        metadata={"fixture": "stage4-runner"},
    )
    database.connection.execute(
        "INSERT INTO papers(paper_id, title) VALUES ('paper-1', 'Venue smoke paper')"
    )
    database.connection.execute(
        """INSERT INTO pipeline_runs(
               run_id, stage, status, input_hash, config_hash,
               implementation_version
           ) VALUES ('stage3-1', 'stage-3-download', 'complete', ?, ?, 'fixture')""",
        ("1" * 64, "2" * 64),
    )
    database.connection.execute(
        """INSERT INTO artifacts(
               artifact_id, paper_id, artifact_kind, relative_path, mime_type,
               byte_size, sha256, provenance_json
           ) VALUES ('pdf-1', 'paper-1', 'pdf', ?, 'application/pdf', ?, ?, '{}')""",
        (stored.relative_path, stored.size_bytes, stored.artifact_hash),
    )
    database.connection.execute(
        """INSERT INTO download_candidates(
               candidate_id, paper_id, resolver, url, host, license,
               access_basis, retrieved_at, provenance_json
           ) VALUES ('candidate-1', 'paper-1', 'fixture',
                     'https://example.test/paper.pdf', 'example.test', NULL,
                     'public_read_only', '2026-08-11T00:00:00Z', '{}')"""
    )
    database.connection.execute(
        """INSERT INTO fetch_requests(
               request_id, candidate_id, policy_version, policy_hash, purpose,
               provider, created_at, expires_at, idempotency_key,
               fencing_token, status
           ) VALUES ('fetch-1', 'candidate-1', 'fixture', ?,
                     'personal_research', 'public_direct',
                     '2026-08-11T00:00:00Z', '2026-08-13T00:00:00Z',
                     'fixture-key', 0, 'consumed')""",
        ("3" * 64,),
    )
    database.connection.execute(
        """INSERT INTO download_attempts(
               download_attempt_id, run_id, candidate_id, provider,
               fetch_request_id, result_status, artifact_id
           ) VALUES ('attempt-1', 'stage3-1', 'candidate-1', 'public_direct',
                     'fetch-1', 'downloaded', 'pdf-1')"""
    )
    database.connection.execute(
        """INSERT INTO stage3_paper_results(
               run_id, paper_id, status, reason_code, updated_at
           ) VALUES ('stage3-1', 'paper-1', ?, ?, '2026-08-11T00:00:00Z')""",
        (stage3_status, stage3_status),
    )
    database.connection.commit()
    database.close()
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "schema_version": "paper-agent.venue-e2e-matrix.v1",
                "run_id": "icml-e2e",
                "venue": "icml",
                "native_pipeline": {
                    "database": "papers.sqlite3",
                    "artifact_root": "artifacts",
                    "stage3_run_id": "stage3-1",
                    "stage4_run_id": None,
                },
                "stages": [
                    {
                        "stage": "stage4",
                        "status": "blocked_pending_explicit_execution",
                        "model": "gpt-5.6-luna",
                        "invocations": 0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return run_dir


class _LocalDoctor:
    def doctor(self, *, prove_model_availability: bool = False) -> DoctorReport:
        assert prove_model_availability is False
        return DoctorReport(
            executable="fixture-codex",
            version="fixture",
            authenticated=True,
            model_availability={"stage4_analysis_luna": "listed"},
        )


class _FakeLuna:
    def __init__(self, calls: list[object]) -> None:
        self.calls = calls

    def invoke(self, request: object) -> CodexExecResult:
        self.calls.append(request)
        payload = json.loads(request.prompt)
        binding = payload["output_binding"]
        output = {
            "paper_id": payload["paper_id"],
            "artifact_hash": payload["artifact_hash"],
            "input_scope": payload["input_scope"],
            "model": "gpt-5.6-luna",
            "model_revision": "fixture",
            "prompt_hash": binding["prompt_hash"],
            "schema_hash": binding["schema_hash"],
            "created_at": binding["created_at"],
            "research_question_and_motivation": "Fixture question.",
            "summary": "Fixture summary.",
            "methods": [],
            "key_techniques": [],
            "datasets": [],
            "experimental_setup": [],
            "metrics": [],
            "results": [],
            "limitations": [],
            "credibility": "Fixture credibility.",
            "resources": [],
            "topic_relevance": "Relevant to the venue smoke topic.",
            "labels": {
                "subquestion": [],
                "theme": [],
                "method_family": [],
                "task": [],
                "dataset": [],
                "benchmark": [],
                "evidence_type": [],
                "publication_status": "unknown",
                "study_setting": "other",
            },
            "label_evidence": [],
            "evidence_units": [],
            "comparison_eligibility": "not_comparable",
            "missing_fields": ["comparison_evidence"],
        }
        metadata = InvocationMetadata(
            invocation_id="fixture-luna-invocation",
            profile="stage4_analysis_luna",
            model="gpt-5.6-luna",
            reasoning_effort="medium",
            schema_name=request.schema_name,
            schema_hash=binding["schema_hash"],
            input_hash=request.input_hash,
            prompt_name=request.prompt_name,
            prompt_hash=binding["prompt_hash"],
            rendered_prompt_hash="4" * 64,
            call_kind=None,
            attempts=1,
            actual_model="gpt-5.6-luna",
            actual_profile="stage4_analysis_luna",
            output_hash=content_hash(output),
        )
        return CodexExecResult(output, metadata)


def test_default_preflight_extracts_and_approves_exact_scope_without_dispatch(
    tmp_path: Path,
) -> None:
    run_dir = _prepared_run(tmp_path)

    result = runner.run(
        run_dir,
        now=NOW,
        codex_factory=_LocalDoctor,
    )

    assert result["status"] == "preflight_complete"
    assert result["invocations"] == 0
    assert result["input_scopes"] == ["full_pdf"]
    normalized_hash = result["normalized_text_artifacts"][0]["sha256"]
    database = Database(run_dir / "papers.sqlite3")
    try:
        grant = json.loads(
            database.connection.execute(
                "SELECT scope_json FROM authorization_grants WHERE grant_id = ?",
                (result["grant_id"],),
            ).fetchone()[0]
        )
        assert grant["paper_ids"] == ["paper-1"]
        assert grant["artifact_hashes"] == [normalized_hash]
        assert grant["provider"] == "codex_cli"
        assert grant["model"] == "gpt-5.6-luna"
        assert grant["data_categories"] == ["normalized_text"]
        assert database.connection.execute(
            "SELECT COUNT(*) FROM pipeline_runs WHERE stage = 'stage4'"
        ).fetchone()[0] == 0
    finally:
        database.close()
    unchanged = json.loads((run_dir / "run.json").read_text())
    assert unchanged["native_pipeline"]["stage4_run_id"] is None
    assert unchanged["stages"][0]["invocations"] == 0


def test_explicit_execution_updates_run_json_and_resume_does_not_dispatch_twice(
    tmp_path: Path,
) -> None:
    run_dir = _prepared_run(tmp_path)
    calls: list[object] = []

    first = runner.run(
        run_dir,
        execute_model=True,
        now=NOW,
        codex_factory=_LocalDoctor,
        invoker_factory=lambda: _FakeLuna(calls),
    )
    resumed = runner.run(
        run_dir,
        execute_model=True,
        now=NOW + timedelta(hours=1),
        codex_factory=_LocalDoctor,
        invoker_factory=lambda: _FakeLuna(calls),
    )

    assert first["status"] == "complete"
    assert first["invocations"] == 1
    assert resumed["invocations"] == 1
    assert resumed["papers"][0]["resumed"] is True
    assert len(calls) == 1
    document = json.loads((run_dir / "run.json").read_text())
    assert document["native_pipeline"]["stage4_run_id"] == "icml-e2e-stage4-luna"
    assert document["stages"][0] == {
        "stage": "stage4",
        "status": "complete",
        "model": "gpt-5.6-luna",
        "invocations": 1,
    }
    stage4_result = json.loads((run_dir / "stage4" / "result.json").read_text())
    assert stage4_result["invocations"] == 1
    assert stage4_result["analysis_output_artifact_ids"]
