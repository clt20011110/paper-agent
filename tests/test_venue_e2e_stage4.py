from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from paper_agent.analysis import AnalysisInput
from paper_agent.artifacts import ArtifactStore
from paper_agent.report_config import ReportResources
from paper_agent.report_plan import REPORT_SECTION_IDS
from paper_agent.storage import Database


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "venue_e2e_stage4.py"


def _module():
    specification = importlib.util.spec_from_file_location("venue_e2e_stage4", SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _stage3_database(path: Path, *, status: str = "downloaded") -> None:
    with Database(path) as database:
        database.migrate()
        with database.transaction() as connection:
            connection.execute(
                """INSERT INTO papers(paper_id, title, authors_json, keywords_json)
                   VALUES ('paper-1', 'Fixture', '[]', '[]')"""
            )
            connection.execute(
                """INSERT INTO pipeline_runs(
                       run_id, stage, status, input_hash, config_hash,
                       implementation_version, started_at, completed_at
                   ) VALUES ('stage3-1', 'stage-3-download', 'complete',
                             'input', 'config', 'test', CURRENT_TIMESTAMP,
                             CURRENT_TIMESTAMP)"""
            )
            connection.execute(
                """INSERT INTO stage3_paper_results(
                       run_id, paper_id, status, reason_code, updated_at
                   ) VALUES ('stage3-1', 'paper-1', ?, 'fixture', CURRENT_TIMESTAMP)""",
                (status,),
            )


def test_model_free_stage4_preflight_creates_exact_grant_without_dispatch(
    tmp_path: Path, monkeypatch
) -> None:
    module = _module()
    run_dir = tmp_path / "icml-run"
    run_dir.mkdir()
    database_path = run_dir / "papers.sqlite3"
    _stage3_database(database_path)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": "icml-run",
                "venue": "icml",
                "native_pipeline": {
                    "database": "papers.sqlite3",
                    "artifact_root": "artifacts",
                    "crawl_run_id": "crawl-1",
                    "filter_run_id": "filter-1",
                    "stage3_run_id": "stage3-1",
                    "stage4_run_id": None,
                    "report_run_id": None,
                    "report_pipeline_run_id": None,
                },
                "stages": [],
            }
        ),
        encoding="utf-8",
    )

    def inputs(_service, _run_id, *, expected_paper_ids, preview):
        assert tuple(expected_paper_ids) == ("paper-1",)
        assert preview is False
        return (
            AnalysisInput(
                "paper-1",
                None,
                "user_supplied",
                normalized_text="deterministic full text",
            ),
        )

    monkeypatch.setattr(module.AnalysisCliService, "_stage3_inputs", inputs)
    result = module.execute_stage4_and_4b(
        run_dir,
        "icml",
        topic="分子生成",
        year=2024,
        through_stage="stage4",
        execute_models=False,
    )

    assert result["status"] == "ready_for_stage4"
    assert result["stage4"]["invocations"] == 0
    assert result["stage4"]["inputs"][0]["artifact_hash"]
    with Database(database_path, read_only=True) as database:
        assert database.connection.execute(
            "SELECT COUNT(*) FROM authorization_grants"
        ).fetchone()[0] == 1
        assert database.connection.execute(
            "SELECT COUNT(*) FROM analysis_dispatches"
        ).fetchone()[0] == 0
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert run["native_pipeline"]["stage4_run_id"] == "icml-run:stage4"
    assert next(item for item in run["stages"] if item["stage"] == "stage4") == {
        "stage": "stage4",
        "status": "ready_for_explicit_execution",
        "model": "gpt-5.6-luna",
        "invocations": 0,
    }


def test_stage3_preflight_rejects_any_non_downloaded_checkpoint(tmp_path: Path) -> None:
    module = _module()
    database_path = tmp_path / "papers.sqlite3"
    _stage3_database(database_path, status="not_available")
    with Database(database_path, read_only=True) as database:
        try:
            module._stage3_paper_ids(database, "stage3-1")
        except module.VenueE2ERuntimeError as error:
            assert "every Stage 3 checkpoint" in str(error)
        else:
            raise AssertionError("non-downloaded Stage 3 checkpoint was accepted")


def test_one_shot_draft_freezes_one_sol_call_and_all_required_sections() -> None:
    module = _module()
    draft = module.build_one_shot_report_draft(
        corpus_snapshot={
            "created_at": "2026-08-11T00:00:00Z",
            "papers": [{"paper_id": "paper-1"}, {"paper_id": "paper-2"}],
        },
        venue="icml",
        topic="分子生成",
        year=2024,
        recent_cutoff="2024-01-01",
        policy_hash="f" * 64,
        resources=ReportResources.defaults(),
        max_input_tokens=8_000_000,
    )

    assert draft["execution_strategy"] == "one_shot"
    assert draft["budget"] == {
        "max_sol_calls": 1,
        "max_input_tokens": 8_000_000,
        "max_retries": 0,
        "audit_calls": 0,
        "repair_calls": 0,
    }
    assert tuple(section["id"] for section in draft["sections"]) == REPORT_SECTION_IDS
    assert all(section["subquestion_ids"] == ["sq-main"] for section in draft["sections"])
    assert all(
        membership["section_ids"] == list(REPORT_SECTION_IDS)
        for membership in draft["paper_memberships"]
    )


def test_one_shot_summary_requires_exact_ledger_and_cas_binding(tmp_path: Path) -> None:
    module = _module()
    artifact_store = ArtifactStore(tmp_path / "artifacts-root")
    stored = artifact_store.put_bytes(
        b'{}',
        mime_type="application/json",
        metadata={"kind": "stage4b_one_shot_output"},
    )
    invocation_id = "sol-invocation-1"
    one_shot = {
        "status": "complete",
        "dispatch_count": 1,
        "budget_calls_reserved": 1,
        "profile": "stage4b_oneshot_sol",
        "model_id": "gpt-5.6-sol",
        "reasoning_effort": "high",
        "invocation_id": invocation_id,
        "output_artifact_id": "artifact-1",
        "output_hash": stored.artifact_hash,
        "artifact_id": "artifact-1",
        "paper_id": None,
        "artifact_kind": "report",
        "relative_path": stored.relative_path,
        "mime_type": stored.mime_type,
        "byte_size": stored.size_bytes,
        "sha256": stored.artifact_hash,
        "processing_status": "available",
    }
    ledger = [{
        "invocation_id": invocation_id,
        "phase": "reduce",
        "node_key": module.ONE_SHOT_NODE_ID,
    }]

    class Cursor:
        def __init__(self, rows):
            self.rows = rows

        def fetchone(self):
            return self.rows[0] if self.rows else None

        def fetchall(self):
            return self.rows

    class Connection:
        def execute(self, query, _parameters):
            return Cursor(ledger if "report_sol_invocations" in query else [one_shot])

    database = SimpleNamespace(connection=Connection())
    assert module._one_shot_summary(
        database, artifact_store, "report-1"
    ) == {
        "invocations": 1,
        "model": "gpt-5.6-sol",
        "strategy": "one_shot",
    }

    ledger.clear()
    with pytest.raises(module.VenueE2ERuntimeError, match="invocation-ledger"):
        module._one_shot_summary(database, artifact_store, "report-1")


def test_paid_tail_run_id_overrides_must_match_frozen_manifest_bindings() -> None:
    module = _module()
    native = {
        "stage4_run_id": "icml-e2e-stage4",
        "report_run_id": "icml-e2e:report",
        "report_pipeline_run_id": "icml-e2e:stage4b",
    }
    exact = {
        "stage4_run_id": native["stage4_run_id"],
        "report_run_id": native["report_run_id"],
        "report_pipeline_run_id": native["report_pipeline_run_id"],
    }

    assert module._pipeline_ids("icml-e2e", native, **exact) == tuple(
        native[name]
        for name in (
            "stage4_run_id",
            "report_run_id",
            "report_pipeline_run_id",
        )
    )
    for name in exact:
        conflicting = {**exact, name: f"different-{name}"}
        with pytest.raises(
            module.VenueE2ERuntimeError, match=f"explicit {name} conflicts"
        ):
            module._pipeline_ids("icml-e2e", native, **conflicting)

    assert module._pipeline_ids(
        "new-run",
        {
            "stage4_run_id": None,
            "report_run_id": None,
            "report_pipeline_run_id": None,
        },
        stage4_run_id="new-stage4",
        report_run_id="new-report",
        report_pipeline_run_id="new-stage4b",
    ) == ("new-stage4", "new-report", "new-stage4b")


@pytest.mark.parametrize("fallback", ("abstract", "metadata"))
def test_venue_e2e_rejects_non_pdf_stage4_fallbacks_before_any_grant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fallback: str
) -> None:
    module = _module()
    run_dir = tmp_path / f"{fallback}-run"
    run_dir.mkdir()
    database_path = run_dir / "papers.sqlite3"
    _stage3_database(database_path)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": f"{fallback}-run",
                "venue": "icml",
                "native_pipeline": {
                    "database": "papers.sqlite3",
                    "artifact_root": "artifacts",
                    "crawl_run_id": "crawl-1",
                    "filter_run_id": "filter-1",
                    "stage3_run_id": "stage3-1",
                    "stage4_run_id": None,
                    "report_run_id": None,
                    "report_pipeline_run_id": None,
                },
                "stages": [],
            }
        ),
        encoding="utf-8",
    )

    def inputs(_service, _run_id, *, expected_paper_ids, preview):
        assert preview is False
        if fallback == "abstract":
            selected = AnalysisInput(
                "paper-1",
                None,
                "public_read_only",
                abstract="Public abstract",
                metadata={"title": "Fixture"},
            )
        else:
            selected = AnalysisInput(
                "paper-1",
                None,
                "public_read_only",
                metadata={"title": "Fixture"},
            )
        return (selected,)

    monkeypatch.setattr(module.AnalysisCliService, "_stage3_inputs", inputs)
    with pytest.raises(module.VenueE2ERuntimeError, match="requires full_pdf"):
        module.execute_stage4_and_4b(
            run_dir,
            "icml",
            topic="分子生成",
            year=2024,
            through_stage="stage4",
            execute_models=False,
        )

    with Database(database_path, read_only=True) as database:
        assert database.connection.execute(
            "SELECT COUNT(*) FROM authorization_grants"
        ).fetchone()[0] == 0
        assert database.connection.execute(
            "SELECT COUNT(*) FROM analysis_dispatches"
        ).fetchone()[0] == 0


def test_completed_stage4_resume_rejects_input_drift_before_overwriting_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    run_dir = tmp_path / "icml-run"
    run_dir.mkdir()
    database_path = run_dir / "papers.sqlite3"
    _stage3_database(database_path)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": "icml-run",
                "venue": "icml",
                "native_pipeline": {
                    "database": "papers.sqlite3",
                    "artifact_root": "artifacts",
                    "crawl_run_id": "crawl-1",
                    "filter_run_id": "filter-1",
                    "stage3_run_id": "stage3-1",
                    "stage4_run_id": None,
                    "report_run_id": None,
                    "report_pipeline_run_id": None,
                },
                "stages": [],
            }
        ),
        encoding="utf-8",
    )

    def frozen_inputs(_service, _run_id, *, expected_paper_ids, preview):
        return (
            AnalysisInput(
                "paper-1",
                None,
                "user_supplied",
                normalized_text="frozen deterministic full text",
            ),
        )

    monkeypatch.setattr(
        module.AnalysisCliService, "_stage3_inputs", frozen_inputs
    )
    module.execute_stage4_and_4b(
        run_dir,
        "icml",
        topic="分子生成",
        year=2024,
        through_stage="stage4",
        execute_models=False,
    )
    state_path = run_dir / module.STATE_PATH
    frozen_state = state_path.read_bytes()
    with Database(database_path) as database:
        with database.transaction() as connection:
            connection.execute(
                """INSERT INTO pipeline_runs(
                       run_id, stage, status, input_hash, config_hash,
                       implementation_version, started_at, completed_at
                   ) VALUES ('icml-run:stage4', 'stage4', 'complete',
                             'stage4-input', 'stage4-config', 'test',
                             CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"""
            )

    def drifted_inputs(_service, _run_id, *, expected_paper_ids, preview):
        return (
            AnalysisInput(
                "paper-1",
                None,
                "user_supplied",
                normalized_text="different deterministic full text",
            ),
        )

    monkeypatch.setattr(
        module.AnalysisCliService, "_stage3_inputs", drifted_inputs
    )
    with pytest.raises(
        module.VenueE2ERuntimeError, match="differ from frozen inputs"
    ):
        module.execute_stage4_and_4b(
            run_dir,
            "icml",
            topic="分子生成",
            year=2024,
            stage4_run_id="icml-run:stage4",
            report_run_id="icml-run:report",
            report_pipeline_run_id="icml-run:stage4b",
            through_stage="stage4",
            resume=True,
            execute_models=False,
        )
    assert state_path.read_bytes() == frozen_state


@pytest.mark.parametrize(
    ("artifact_hash", "input_scope"),
    (("b" * 64, "full_pdf"), ("a" * 64, "abstract_only")),
)
def test_stage4_ledger_must_match_exact_full_pdf_input_binding(
    artifact_hash: str, input_scope: str
) -> None:
    module = _module()

    class Rows:
        def fetchall(self):
            return [
                {
                    "paper_id": "paper-1",
                    "artifact_hash": artifact_hash,
                    "input_scope": input_scope,
                    "status": "complete",
                    "dispatch_count": 1,
                    "profile": "stage4_analysis_luna",
                    "model_id": "gpt-5.6-luna",
                }
            ]

    class Connection:
        def execute(self, *_args, **_kwargs):
            return Rows()

    class FakeDatabase:
        connection = Connection()

    with pytest.raises(
        module.VenueE2ERuntimeError, match="artifact/input bindings"
    ):
        module._stage4_dispatch_summary(
            FakeDatabase(),
            "stage4-1",
            [
                {
                    "paper_id": "paper-1",
                    "artifact_hash": "a" * 64,
                    "input_scope": "full_pdf",
                    "data_category": "normalized_text",
                }
            ],
        )


def test_resume_loads_approved_report_inputs_instead_of_rebuilding_metadata(
    tmp_path: Path,
) -> None:
    module = _module()
    run_dir = tmp_path / "icml-run"
    plan_dir = run_dir / "reports" / "plans" / "report-plan-1"
    stage4b_dir = run_dir / "stage4b"
    plan_dir.mkdir(parents=True)
    stage4b_dir.mkdir()
    corpus = {
        "snapshot_hash": "c" * 64,
        "papers": [{"paper_id": "paper-1", "authors": []}],
    }
    audit = {"pack_hash": "s" * 64}
    plan = {
        "plan_id": "report-plan-1",
        "plan_hash": "p" * 64,
        "corpus_snapshot_hash": corpus["snapshot_hash"],
        "search_audit_pack_hash": audit["pack_hash"],
    }
    for path, document in (
        (plan_dir / "REPORT_PLAN.json", plan),
        (plan_dir / "CORPUS_SNAPSHOT.json", corpus),
        (plan_dir / "SEARCH_AUDIT.json", audit),
        (stage4b_dir / "REPORT_PLAN.json", plan),
    ):
        path.write_bytes(module.canonical_json(document))
    state = {
        "stage4b": {
            "report_plan_id": plan["plan_id"],
            "report_plan_hash": plan["plan_hash"],
            "report_input_bundle_id": "report-input-1",
        }
    }

    loaded = module._frozen_report_inputs(run_dir, state, resume=True)

    assert loaded is not None
    loaded_plan, loaded_corpus, loaded_audit, bundle_id = loaded
    assert loaded_plan == plan
    assert loaded_corpus["papers"][0]["authors"] == []
    assert loaded_audit == audit
    assert bundle_id == "report-input-1"
    assert module._frozen_report_inputs(run_dir, state, resume=False) is None
