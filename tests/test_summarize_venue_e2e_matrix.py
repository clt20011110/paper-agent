from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sqlite3

import pytest

from paper_agent.approval import approve, approved_content_hash
from paper_agent.canonical import content_hash
from paper_agent.report_artifacts import (
    ReportArtifactStore,
    audit_coverage_ledger,
    audit_rubric_hash,
    audit_search_limitations,
    report_artifact_hash,
)
from paper_agent.report_cli_service import verify_report_run
from paper_agent.reporting import stable_claim_id
from test_report_artifacts import _bundle


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "summarize_venue_e2e_matrix.py"
IMPORT_MANIFEST = ROOT / "configs" / "e2e" / "venue-e2e-acceptance-imports.json"
CURRENT_IMPORT_MANIFEST = (
    ROOT
    / "configs"
    / "e2e"
    / "venue-e2e-acceptance-imports-20260812-current.json"
)


def _module():
    specification = importlib.util.spec_from_file_location(
        "summarize_venue_e2e_matrix", SCRIPT
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_current_acceptance_manifest_has_no_historical_imports() -> None:
    module = _module()

    assert module._load_acceptance_imports(CURRENT_IMPORT_MANIFEST) == ()


def _two_claim_audit_fixture() -> tuple[dict, list[dict], dict]:
    bundle = _bundle()
    first = bundle["claims"][0]
    second = deepcopy(first)
    second["claim_key"] = {
        **second["claim_key"],
        "predicate_id": "confirms",
        "qualifier_context_hash": "d" * 64,
    }
    second["claim_id"] = stable_claim_id(
        second["claim_key"], report_run_id="report-1"
    )
    second["claim_text"] = "第二项证据在相同条件下同样达到 91%。"
    published_claims = sorted(
        [first, second], key=lambda item: str(item["claim_id"])
    )
    runtime_claims = list(reversed(published_claims))
    bundle["claims"] = published_claims
    bundle["document"]["blocks"].append({
        "block_id": "b3",
        "block_kind": "prose",
        "section_id": "evidence",
        "text": "第二项证据在相同条件下同样达到 91%。[@p1]",
        "claim_ids": [second["claim_id"]],
        "citation_paper_ids": ["p1"],
    })
    bundle["coverage"]["papers"][0]["evidence_claim_ids"] = sorted(
        item["claim_id"] for item in published_claims
    )
    legacy_hash = content_hash({
        "document": bundle["document"],
        "claims": runtime_claims,
        "coverage": bundle["coverage"],
        "comparison_groups": {},
        "claim_relations": [],
        "bibliography": bundle["bibliography"],
    })
    audit = {
        "audit_pass": "deterministic",
        "report_document_hash": content_hash(bundle["document"]),
        "report_artifact_hash": legacy_hash,
        "report_plan_hash": content_hash(bundle["plan"]),
        "rubric_hash": audit_rubric_hash(),
        "search_limitations_hash": content_hash(list(audit_search_limitations(
            bundle["search_audit"], bundle["corpus_snapshot"]
        ))),
        "coverage_complete": True,
        "coverage_ledger": audit_coverage_ledger(
            bundle["document"], published_claims
        ),
        "findings": [],
    }
    current_bundle = {
        "document": bundle["document"],
        "claims": runtime_claims,
        "coverage": {
            key: value
            for key, value in bundle["coverage"].items()
            if key != "complete"
        },
        "comparison_groups": {},
        "claim_relations": [],
        "bibliography": bundle["bibliography"],
    }
    return bundle, runtime_claims, {"audit": audit, "current": current_bundle}


def _legacy_audit_connection(
    audit_hash: str, current_bundle: dict
) -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute(
        """CREATE TABLE report_audit_runs(
               report_run_id TEXT, base_artifact_hash TEXT,
               current_artifact_hash TEXT, current_bundle_json TEXT
           )"""
    )
    connection.execute(
        "INSERT INTO report_audit_runs VALUES (?, ?, ?, ?)",
        (
            "report-1",
            audit_hash,
            audit_hash,
            json.dumps(current_bundle, ensure_ascii=False),
        ),
    )
    return connection


def _catalog(root: Path, *venue_ids: str) -> Path:
    catalog = root / "catalog"
    catalog.mkdir()
    for venue_id in venue_ids:
        (catalog / f"{venue_id}.yaml").write_text(
            f'schema_version: "1"\nvenue_id: {venue_id}\n', encoding="utf-8"
        )
    return catalog


def _cas_artifact(
    artifact_root: Path,
    payload: bytes,
    *,
    artifact_id: str,
    paper_id: str | None,
    artifact_kind: str,
    mime_type: str,
) -> tuple[tuple[object, ...], str, Path]:
    artifact_hash = sha256(payload).hexdigest()
    relative_path = f"artifacts/{artifact_hash[:2]}/{artifact_hash}"
    path = artifact_root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    row = (
        artifact_id,
        paper_id,
        artifact_kind,
        relative_path,
        mime_type,
        len(payload),
        artifact_hash,
        "available",
    )
    return row, artifact_hash, path


def _publish_report(
    run_dir: Path,
    analysis_hash: str,
    *,
    venue: str,
    query_plan_hash: str,
    stage4b_config_hash: str,
) -> Path:
    bundle = _bundle()
    bundle["corpus_snapshot"]["papers"][0]["analysis_artifact_hash"] = analysis_hash
    bundle["plan"]["scope"] = {"venues": [venue]}
    bundle["plan"]["query_plan_hash"] = query_plan_hash
    bundle["plan"]["stage4b_config_hash"] = stage4b_config_hash
    bundle["search_audit"]["query_plan_hash"] = query_plan_hash
    audit = {
        "audit_pass": "deterministic",
        "report_document_hash": content_hash(bundle["document"]),
        "report_artifact_hash": report_artifact_hash(
            document=bundle["document"],
            claims=bundle["claims"],
            coverage=bundle["coverage"],
            comparison_groups={},
            claim_relations=[],
            bibliography=bundle["bibliography"],
        ),
        "report_plan_hash": content_hash(bundle["plan"]),
        "rubric_hash": audit_rubric_hash(),
        "search_limitations_hash": content_hash(
            list(
                audit_search_limitations(
                    bundle["search_audit"], bundle["corpus_snapshot"]
                )
            )
        ),
        "coverage_complete": True,
        "coverage_ledger": audit_coverage_ledger(
            bundle["document"], bundle["claims"]
        ),
        "findings": [],
    }
    return ReportArtifactStore(run_dir).write(
        plan=bundle["plan"],
        search_audit=bundle["search_audit"],
        corpus_snapshot=bundle["corpus_snapshot"],
        claims=bundle["claims"],
        comparison_groups={},
        claim_relations=[],
        document=bundle["document"],
        coverage=bundle["coverage"],
        bibliography=bundle["bibliography"],
        audit=audit,
    )


def _venue_run(root: Path, venue: str = "icml") -> Path:
    run_dir = root / venue
    run_dir.mkdir(parents=True)
    artifact_root = run_dir / "artifacts"
    artifact_root.mkdir()

    pdf_row, pdf_hash, _ = _cas_artifact(
        artifact_root,
        b"%PDF-1.7\nvenue smoke fixture\n",
        artifact_id="pdf-1",
        paper_id="p1",
        artifact_kind="pdf",
        mime_type="application/pdf",
    )
    text_row, text_hash, _ = _cas_artifact(
        artifact_root,
        b"Normalized full text for the venue smoke.\n",
        artifact_id="text-1",
        paper_id="p1",
        artifact_kind="text",
        mime_type="text/plain; charset=utf-8",
    )
    analysis_row, analysis_hash, _ = _cas_artifact(
        artifact_root,
        b'{"analysis":"fixture"}',
        artifact_id="analysis-output-1",
        paper_id="p1",
        artifact_kind="analysis",
        mime_type="application/json",
    )
    markdown_row, _, _ = _cas_artifact(
        artifact_root,
        b"# Analysis fixture\n",
        artifact_id="analysis-markdown-1",
        paper_id="p1",
        artifact_kind="analysis",
        mime_type="text/markdown; charset=utf-8",
    )
    sol_row, sol_hash, _ = _cas_artifact(
        artifact_root,
        b'{"report":"fixture"}',
        artifact_id="sol-output-1",
        paper_id=None,
        artifact_kind="report",
        mime_type="application/json",
    )
    query_plan_draft = {
        "plan_id": "query-plan-1",
        "schema_version": "1",
        "scope": {"venues": [venue]},
        "status": "draft",
        "created_at": "2026-08-11T00:00:00Z",
    }
    query_plan = approve(
        query_plan_draft,
        approved_content_hash(query_plan_draft),
        approved_by="fixture",
        approved_at="2026-08-11T00:00:00Z",
        hash_field="plan_hash",
    )
    query_plan_hash = str(query_plan["plan_hash"])
    stage4b_config_hash = "2" * 64

    database = sqlite3.connect(run_dir / "papers.sqlite3")
    database.executescript(
        """
        CREATE TABLE pipeline_runs(
            run_id TEXT PRIMARY KEY, stage TEXT, status TEXT,
            input_hash TEXT, config_hash TEXT, implementation_version TEXT,
            started_at TEXT, completed_at TEXT, created_at TEXT
        );
        CREATE TABLE crawl_runs(
            crawl_run_id TEXT PRIMARY KEY, run_id TEXT, search_plan_id TEXT,
            status TEXT, started_at TEXT
        );
        CREATE TABLE search_plans(
            search_plan_id TEXT PRIMARY KEY, content_hash TEXT,
            schema_version TEXT, plan_json TEXT, approval_json TEXT, status TEXT
        );
        CREATE TABLE crawl_paper_snapshots(
            crawl_run_id TEXT, paper_id TEXT, metadata_hash TEXT,
            status_version_json TEXT
        );
        CREATE TABLE filter_decisions(
            paper_id TEXT, run_id TEXT, status TEXT, model_id TEXT, reason TEXT
        );
        CREATE TABLE stage3_paper_results(
            run_id TEXT, paper_id TEXT, status TEXT, reason_code TEXT
        );
        CREATE TABLE download_candidates(
            candidate_id TEXT PRIMARY KEY, paper_id TEXT
        );
        CREATE TABLE fetch_requests(
            request_id TEXT PRIMARY KEY, candidate_id TEXT
        );
        CREATE TABLE download_attempts(
            download_attempt_id TEXT PRIMARY KEY, run_id TEXT, candidate_id TEXT,
            fetch_request_id TEXT, result_status TEXT, artifact_id TEXT,
            attempted_at TEXT
        );
        CREATE TABLE artifacts(
            artifact_id TEXT PRIMARY KEY, paper_id TEXT, artifact_kind TEXT,
            relative_path TEXT, mime_type TEXT, byte_size INTEGER, sha256 TEXT,
            processing_status TEXT
        );
        CREATE TABLE text_extractions(
            extraction_id TEXT PRIMARY KEY, paper_id TEXT,
            source_artifact_id TEXT, source_sha256 TEXT,
            output_artifact_id TEXT, status TEXT
        );
        CREATE TABLE analysis_dispatches(
            dispatch_id TEXT PRIMARY KEY, run_id TEXT, paper_id TEXT,
            artifact_hash TEXT, artifact_id TEXT, input_scope TEXT,
            config_hash TEXT, implementation_version TEXT, profile TEXT,
            model_id TEXT, prompt_hash TEXT, schema_hash TEXT,
            prompt_input_hash TEXT, rendered_prompt_hash TEXT, status TEXT,
            dispatch_count INTEGER, invocation_id TEXT, analysis_run_id TEXT
        );
        CREATE TABLE analysis_runs(
            analysis_run_id TEXT PRIMARY KEY, run_id TEXT, paper_id TEXT,
            artifact_id TEXT, input_hash TEXT, input_scope TEXT, model_id TEXT,
            prompt_hash TEXT, schema_hash TEXT, implementation_version TEXT,
            status TEXT, output_artifact_id TEXT, markdown_artifact_id TEXT
        );
        CREATE TABLE report_runs(
            report_run_id TEXT PRIMARY KEY, run_id TEXT, status TEXT,
            output_relative_path TEXT, aggregation_tree_json TEXT
        );
        CREATE TABLE report_one_shot_runs(
            report_run_id TEXT PRIMARY KEY, input_artifact_hashes_json TEXT,
            input_hash TEXT, rendered_prompt_hash TEXT, status TEXT,
            dispatch_count INTEGER, budget_calls_reserved INTEGER,
            profile TEXT, model_id TEXT, reasoning_effort TEXT,
            invocation_id TEXT, output_hash TEXT, output_artifact_id TEXT
        );
        CREATE TABLE report_sol_invocations(
            report_run_id TEXT, invocation_id TEXT, phase TEXT,
            node_key TEXT, created_at TEXT
        );
        CREATE TABLE report_reduce_nodes(
            report_run_id TEXT, dispatch_count INTEGER
        );
        CREATE TABLE report_audit_steps(
            report_run_id TEXT, step_name TEXT, dispatch_count INTEGER
        );
        CREATE TABLE report_audit_shard_steps(
            report_run_id TEXT, dispatch_count INTEGER
        );
        """
    )
    runs = (
        ("search-1", "stage-1", "complete", "2026-08-11T00:00:00Z"),
        ("filter-1", "stage-2", "complete", "2026-08-11T00:01:00Z"),
        ("download-1", "stage-3-download", "complete", "2026-08-11T00:02:00Z"),
        ("analysis-1", "stage4", "complete", "2026-08-11T00:03:00Z"),
        ("report-pipeline-1", "stage4b", "complete", "2026-08-11T00:04:00Z"),
    )
    database.executemany(
        """INSERT INTO pipeline_runs(
               run_id, stage, status, input_hash, config_hash,
               implementation_version, started_at, completed_at, created_at
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            (
                run_id,
                stage,
                status,
                "1" * 64,
                "2" * 64,
                (
                    "analysis-v1"
                    if stage == "stage4"
                    else "stage4b-one-shot-v2"
                    if stage == "stage4b"
                    else "fixture-v1"
                ),
                stamp,
                stamp,
                stamp,
            )
            for run_id, stage, status, stamp in runs
        ],
    )
    database.execute(
        "INSERT INTO search_plans VALUES (?, ?, ?, ?, ?, ?)",
        (
            "query-plan-1",
            query_plan_hash,
            "1",
            json.dumps(query_plan, sort_keys=True),
            json.dumps(query_plan["approval"], sort_keys=True),
            "approved",
        ),
    )
    database.execute(
        "INSERT INTO crawl_runs VALUES (?, ?, ?, ?, ?)",
        (
            "crawl-1",
            "search-1",
            "query-plan-1",
            "complete",
            "2026-08-11T00:00:00Z",
        ),
    )
    database.execute(
        "INSERT INTO crawl_paper_snapshots VALUES (?, ?, ?, ?)",
        ("crawl-1", "p1", "9" * 64, "{}"),
    )
    database.execute(
        "INSERT INTO filter_decisions VALUES (?, ?, ?, ?, ?)",
        (
            "p1",
            "filter-1",
            "relevant",
            "TEST_ONLY/title-rule-adjudicator",
            json.dumps({"test_only": True}),
        ),
    )
    database.execute(
        "INSERT INTO stage3_paper_results VALUES (?, ?, ?, ?)",
        ("download-1", "p1", "downloaded", "pdf_valid"),
    )
    database.execute("INSERT INTO download_candidates VALUES (?, ?)", ("candidate-1", "p1"))
    database.execute("INSERT INTO fetch_requests VALUES (?, ?)", ("fetch-1", "candidate-1"))
    database.execute(
        "INSERT INTO download_attempts VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            "attempt-1",
            "download-1",
            "candidate-1",
            "fetch-1",
            "downloaded",
            "pdf-1",
            "2026-08-11T00:02:00Z",
        ),
    )
    database.executemany(
        "INSERT INTO artifacts VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (pdf_row, text_row, analysis_row, markdown_row, sol_row),
    )
    database.execute(
        "INSERT INTO text_extractions VALUES (?, ?, ?, ?, ?, ?)",
        ("extract-1", "p1", "pdf-1", pdf_hash, "text-1", "full_text_ready"),
    )
    database.execute(
        "INSERT INTO analysis_dispatches VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "dispatch-1",
            "analysis-1",
            "p1",
            text_hash,
            "text-1",
            "full_pdf",
            "2" * 64,
            "analysis-v1",
            "stage4_analysis_luna",
            "gpt-5.6-luna",
            "3" * 64,
            "4" * 64,
            "5" * 64,
            "6" * 64,
            "complete",
            1,
            "luna-invocation-1",
            "analysis-result-1",
        ),
    )
    database.execute(
        "INSERT INTO analysis_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "analysis-result-1",
            "analysis-1",
            "p1",
            "text-1",
            "5" * 64,
            "full_pdf",
            "gpt-5.6-luna",
            "3" * 64,
            "4" * 64,
            "analysis-v1",
            "complete",
            "analysis-output-1",
            "analysis-markdown-1",
        ),
    )
    database.execute(
        "INSERT INTO report_runs VALUES (?, ?, ?, ?, ?)",
        (
            "report-1",
            "report-pipeline-1",
            "complete",
            "reports/report-1",
            json.dumps({"strategy": "one_shot"}),
        ),
    )
    database.execute(
        "INSERT INTO report_one_shot_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "report-1",
            json.dumps([analysis_hash]),
            "7" * 64,
            "8" * 64,
            "complete",
            1,
            1,
            "stage4b_oneshot_sol",
            "gpt-5.6-sol",
            "high",
            "sol-invocation-1",
            sol_hash,
            "sol-output-1",
        ),
    )
    database.execute(
        "INSERT INTO report_sol_invocations VALUES (?, ?, ?, ?, ?)",
        (
            "report-1",
            "sol-invocation-1",
            "reduce",
            "one_shot:0001",
            "2026-08-11T00:04:00Z",
        ),
    )
    database.commit()
    database.close()

    report_dir = _publish_report(
        run_dir,
        analysis_hash,
        venue=venue,
        query_plan_hash=query_plan_hash,
        stage4b_config_hash=stage4b_config_hash,
    )
    assert report_dir == run_dir / "reports" / "report-1"
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "venue": venue,
                "native_pipeline": {
                    "database": "papers.sqlite3",
                    "artifact_root": "artifacts",
                    "search_run_id": "search-1",
                    "crawl_run_id": "crawl-1",
                    "filter_run_id": "filter-1",
                    "stage3_run_id": "download-1",
                    "stage4_run_id": "analysis-1",
                    "report_run_id": "report-1",
                    "report_pipeline_run_id": "report-pipeline-1",
                },
            }
        ),
        encoding="utf-8",
    )
    return run_dir


def _build_single(root: Path, venue: str = "icml"):
    _venue_run(root, venue)
    module = _module()
    return module, module.build_matrix(
        root,
        venue_catalog_root=_catalog(root, venue),
    )


def test_builds_passing_matrix_with_exact_cas_and_full_verifier_bindings(
    tmp_path: Path,
) -> None:
    module, matrix = _build_single(tmp_path)

    assert matrix["summary"] == {
        "venue_count": 1,
        "catalog_venue_count": 1,
        "passed": 1,
        "failed": 0,
        "coverage_complete": True,
        "missing_venues": [],
        "unexpected_venues": [],
        "duplicate_venues": [],
        "all_passed": True,
    }
    venue = matrix["venues"][0]
    assert venue["passed"] is True
    assert venue["checks"]["stage3_pdf_checkpoint_complete"]["attempt_count"] == 1
    stage4 = venue["checks"]["stage4_luna_invocation_one"]
    assert stage4["input_scopes"] == ["full_pdf"]
    assert stage4["stage3_lineage_match"] is True
    one_shot = venue["checks"]["stage4b_sol_one_shot_only"]
    assert one_shot["sol_invocation_ledger_count"] == 1
    assert one_shot["exact_stage4_inputs"] is True
    assert one_shot["reduce_nodes"] == one_shot["audit_steps"] == 0
    assert one_shot["implementation_version"] == "stage4b-one-shot-v2"
    assert one_shot["implementation_qualification"] == (
        "legacy_v2_reverified_by_current_verifier"
    )
    assert one_shot["config_hash"] == "2" * 64
    verified = venue["checks"]["verify_passed"]
    assert verified["verifier_matches_saved"] is True
    assert verified["exact_stage4_corpus_binding"] is True
    assert verified["audit_binding_mode"] == "canonical_claim_id_order"
    assert verified["audit_binding_database_row_bound"] is False
    assert len(verified["manifest_hash"]) == 64
    markdown = module.render_markdown(matrix)
    assert "1/1 catalog venues passed" in markdown
    assert "do not prove live provider transport" in markdown
    assert "hash-pinned historical attestations" in markdown
    assert "stage4b-one-shot-v2 (legacy_v2_reverified_by_current_verifier)" in markdown


def test_legacy_runtime_claim_order_is_strictly_database_bound() -> None:
    module = _module()
    bundle, _, evidence = _two_claim_audit_fixture()
    audit = evidence["audit"]
    connection = _legacy_audit_connection(
        audit["report_artifact_hash"], evidence["current"]
    )
    try:
        result = module._validate_report_audit_binding(
            connection,
            report_run_id="report-1",
            implementation_version="stage4b-one-shot-v2",
            audit=audit,
            plan=bundle["plan"],
            document=bundle["document"],
            claims=bundle["claims"],
            coverage=bundle["coverage"],
            comparison_groups={},
            claim_relations=[],
            bibliography=bundle["bibliography"],
            search_audit=bundle["search_audit"],
            corpus_snapshot=bundle["corpus_snapshot"],
        )
    finally:
        connection.close()

    assert result == {
        "mode": "legacy_runtime_claim_order_verified",
        "database_row_bound": True,
        "legacy_artifact_hash": audit["report_artifact_hash"],
    }

    assert module._implementation_qualification(
        "stage4b-one-shot-v2",
        verifier_passed=True,
        audit_binding_mode="legacy_runtime_claim_order_verified",
    ) == "legacy_v2_audit_order_reverified"
    assert module._implementation_qualification(
        "stage4b-one-shot-v3",
        verifier_passed=True,
        audit_binding_mode="canonical_claim_id_order",
    ) == "current_v3"

    rejected = _legacy_audit_connection(
        audit["report_artifact_hash"], evidence["current"]
    )
    try:
        with pytest.raises(module.MatrixError, match="requires stage4b-one-shot-v1 or v2"):
            module._validate_report_audit_binding(
                rejected,
                report_run_id="report-1",
                implementation_version="stage4b-one-shot-v3",
                audit=audit,
                plan=bundle["plan"],
                document=bundle["document"],
                claims=bundle["claims"],
                coverage=bundle["coverage"],
                comparison_groups={},
                claim_relations=[],
                bibliography=bundle["bibliography"],
                search_audit=bundle["search_audit"],
                corpus_snapshot=bundle["corpus_snapshot"],
            )
    finally:
        rejected.close()


@pytest.mark.parametrize(
    ("drift", "expected_error"),
    (
        ("base_hash", "base/current/artifact hashes"),
        ("claim_content", "claims differ from published"),
        ("duplicate_claim", "duplicate claim_id"),
        ("coverage", "coverage differs from the published"),
        ("bibliography", "bibliography binding has drifted"),
        ("report_run_id", "lacks one exact report_audit_runs row"),
    ),
)
def test_legacy_claim_order_rejects_any_non_ordering_drift(
    drift: str, expected_error: str
) -> None:
    module = _module()
    bundle, _, evidence = _two_claim_audit_fixture()
    audit = evidence["audit"]
    current = deepcopy(evidence["current"])
    if drift == "claim_content":
        current["claims"][0]["claim_text"] += "篡改"
    elif drift == "duplicate_claim":
        current["claims"][1]["claim_id"] = current["claims"][0]["claim_id"]
    elif drift == "coverage":
        current["coverage"]["uncovered_claim_ids"] = ["missing"]
    elif drift == "bibliography":
        current["bibliography"]["p1"]["title"] = "Tampered"
    connection = _legacy_audit_connection(
        audit["report_artifact_hash"], current
    )
    if drift == "base_hash":
        connection.execute(
            "UPDATE report_audit_runs SET base_artifact_hash = ?",
            ("f" * 64,),
        )
    elif drift == "report_run_id":
        connection.execute(
            "UPDATE report_audit_runs SET report_run_id = 'report-other'"
        )
    try:
        with pytest.raises(module.MatrixError, match=expected_error):
            module._validate_report_audit_binding(
                connection,
                report_run_id="report-1",
                implementation_version="stage4b-one-shot-v2",
                audit=audit,
                plan=bundle["plan"],
                document=bundle["document"],
                claims=bundle["claims"],
                coverage=bundle["coverage"],
                comparison_groups={},
                claim_relations=[],
                bibliography=bundle["bibliography"],
                search_audit=bundle["search_audit"],
                corpus_snapshot=bundle["corpus_snapshot"],
            )
    finally:
        connection.close()


def test_missing_run_json_fails_instead_of_downgrading_to_auto_discovery(
    tmp_path: Path,
) -> None:
    run_dir = _venue_run(tmp_path)
    (run_dir / "run.json").unlink()
    module = _module()

    matrix = module.build_matrix(
        tmp_path, venue_catalog_root=_catalog(tmp_path, "icml")
    )

    venue = matrix["venues"][0]
    assert venue["passed"] is False
    assert matrix["summary"]["all_passed"] is False
    assert "lacks required run.json" in venue["checks"]["stage1_complete"]["error"]


@pytest.mark.parametrize(
    ("binding", "escaped"),
    (
        ("database", "../shared.sqlite3"),
        ("artifact_root", "../shared-artifacts"),
        ("output_root", "../shared-output"),
    ),
)
def test_native_paths_must_resolve_within_their_venue_run(
    tmp_path: Path, binding: str, escaped: str
) -> None:
    run_dir = _venue_run(tmp_path)
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    run["native_pipeline"][binding] = escaped
    (run_dir / "run.json").write_text(json.dumps(run), encoding="utf-8")
    module = _module()

    matrix = module.build_matrix(
        tmp_path, venue_catalog_root=_catalog(tmp_path, "icml")
    )

    venue = matrix["venues"][0]
    assert venue["passed"] is False
    assert any(
        "escapes its venue run directory" in str(check.get("error"))
        for check in venue["checks"].values()
    )


def test_cross_venue_database_and_report_rebinding_is_rejected(tmp_path: Path) -> None:
    source = _venue_run(tmp_path, "icml")
    duplicate = tmp_path / "cvpr"
    duplicate.mkdir()
    run = json.loads((source / "run.json").read_text(encoding="utf-8"))
    run["venue"] = "cvpr"
    run["native_pipeline"]["database"] = str(source / "papers.sqlite3")
    run["native_pipeline"]["artifact_root"] = str(source / "artifacts")
    run["native_pipeline"]["output_root"] = str(source)
    (duplicate / "run.json").write_text(json.dumps(run), encoding="utf-8")
    module = _module()

    matrix = module.build_matrix(
        tmp_path, venue_catalog_root=_catalog(tmp_path, "icml", "cvpr")
    )

    rows = {row["venue"]: row for row in matrix["venues"]}
    assert rows["icml"]["passed"] is True
    assert rows["cvpr"]["passed"] is False
    assert matrix["summary"]["all_passed"] is False


@pytest.mark.parametrize(
    "duplicate_field",
    ("database_sha256", "manifest_hash"),
)
def test_duplicate_evidence_identity_across_venues_is_rejected(
    duplicate_field: str,
) -> None:
    module = _module()
    rows = [
        {
            "venue": "icml",
            "database": "/runs/icml/papers.sqlite3",
            "database_sha256": "a" * 64,
            "checks": {
                "verify_passed": {
                    "report_directory": "/runs/icml/reports/report-1",
                    "manifest_hash": "b" * 64,
                }
            },
        },
        {
            "venue": "cvpr",
            "database": "/runs/cvpr/papers.sqlite3",
            "database_sha256": "c" * 64,
            "checks": {
                "verify_passed": {
                    "report_directory": "/runs/cvpr/reports/report-1",
                    "manifest_hash": "d" * 64,
                }
            },
        },
    ]
    if duplicate_field == "database_sha256":
        rows[1]["database_sha256"] = rows[0]["database_sha256"]
    else:
        rows[1]["checks"]["verify_passed"]["manifest_hash"] = rows[0][
            "checks"
        ]["verify_passed"]["manifest_hash"]

    with pytest.raises(module.MatrixError, match="cross-venue"):
        module._reject_cross_venue_evidence_reuse(rows)


def test_run_venue_must_match_persisted_query_plan_and_report_scope(
    tmp_path: Path,
) -> None:
    run_dir = _venue_run(tmp_path)
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    run["venue"] = "cvpr"
    (run_dir / "run.json").write_text(json.dumps(run), encoding="utf-8")
    module = _module()

    matrix = module.build_matrix(
        tmp_path, venue_catalog_root=_catalog(tmp_path, "cvpr")
    )

    venue = matrix["venues"][0]
    assert venue["passed"] is False
    assert "venue scope does not match" in venue["checks"]["stage1_complete"]["error"]
    assert venue["checks"]["verify_passed"][
        "exact_stage1_query_plan_scope_binding"
    ] is False


def test_report_scope_cannot_drift_from_the_frozen_stage1_query_plan(
    tmp_path: Path,
) -> None:
    run_dir = _venue_run(tmp_path)
    report_dir = run_dir / "reports" / "report-1"
    plan_path = report_dir / "REPORT_PLAN.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["scope"]["venues"] = ["cvpr"]
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    audit_path = report_dir / "AUDIT.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["report_plan_hash"] = content_hash(plan)
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    verification_path = report_dir / "VERIFICATION.json"
    verification = verify_report_run(run_dir, "report-1")
    verification_path.write_text(json.dumps(verification), encoding="utf-8")
    manifest_path = report_dir / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["REPORT_PLAN.json"] = content_hash(plan)
    manifest["artifacts"]["AUDIT.json"] = content_hash(audit)
    manifest["artifacts"]["VERIFICATION.json"] = content_hash(verification)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    module = _module()

    matrix = module.build_matrix(
        tmp_path, venue_catalog_root=_catalog(tmp_path, "icml")
    )

    check = matrix["venues"][0]["checks"]["verify_passed"]
    assert check["passed"] is False
    assert check["exact_stage1_query_plan_scope_binding"] is False


def test_stage2_must_cover_the_frozen_stage1_membership(tmp_path: Path) -> None:
    run_dir = _venue_run(tmp_path)
    database = sqlite3.connect(run_dir / "papers.sqlite3")
    database.execute("DELETE FROM crawl_paper_snapshots")
    database.commit()
    database.close()
    module = _module()

    matrix = module.build_matrix(
        tmp_path, venue_catalog_root=_catalog(tmp_path, "icml")
    )

    venue = matrix["venues"][0]
    assert venue["passed"] is False
    assert venue["checks"]["stage1_complete"]["one_record_snapshot"] is False
    assert venue["checks"]["stage2_test_only_relevant_one"][
        "stage1_membership_closed"
    ] is False


def test_portable_matrix_replaces_machine_local_roots(tmp_path: Path) -> None:
    module = _module()
    run_root = tmp_path / "runs"
    repository_root = tmp_path / "checkout"
    matrix = {
        "run_root": str(run_root.resolve()),
        "database": str((run_root / "icml" / "papers.sqlite3").resolve()),
        "venue_catalog_root": str((repository_root / "venues").resolve()),
        "external": "https://example.org/paper.pdf",
        "error": f"cannot read {run_root.resolve()}/icml/AUDIT.json",
    }

    portable = module._portable_matrix(
        matrix, run_root=run_root, repository_root=repository_root
    )

    assert portable["run_root"] == "$RUN_ROOT"
    assert portable["database"] == "$RUN_ROOT/icml/papers.sqlite3"
    assert portable["venue_catalog_root"] == "$REPOSITORY_ROOT/venues"
    assert portable["external"] == "https://example.org/paper.pdf"
    assert portable["error"] == "cannot read $RUN_ROOT/icml/AUDIT.json"
    assert portable["portable_paths"]["$RUN_ROOT"]


def test_missing_bound_crawl_fails_instead_of_falling_back_to_latest(tmp_path: Path) -> None:
    run_dir = _venue_run(tmp_path)
    database = sqlite3.connect(run_dir / "papers.sqlite3")
    database.execute("DELETE FROM crawl_runs WHERE crawl_run_id = 'crawl-1'")
    database.execute(
        "INSERT INTO crawl_runs VALUES (?, ?, ?, ?, ?)",
        (
            "crawl-other",
            "search-1",
            "query-plan-1",
            "complete",
            "2026-08-11T01:00:00Z",
        ),
    )
    database.commit()
    database.close()
    module = _module()

    matrix = module.build_matrix(
        tmp_path, venue_catalog_root=_catalog(tmp_path, "icml")
    )

    check = matrix["venues"][0]["checks"]["stage1_complete"]
    assert check["passed"] is False
    assert "configured crawl_run_id does not exist" in check["error"]


def test_stage3_rehashes_the_exact_pdf_cas_payload(tmp_path: Path) -> None:
    run_dir = _venue_run(tmp_path)
    database = sqlite3.connect(run_dir / "papers.sqlite3")
    relative_path = database.execute(
        "SELECT relative_path FROM artifacts WHERE artifact_id = 'pdf-1'"
    ).fetchone()[0]
    database.close()
    (run_dir / "artifacts" / relative_path).write_bytes(b"%PDF-tampered payload")
    module = _module()

    matrix = module.build_matrix(
        tmp_path, venue_catalog_root=_catalog(tmp_path, "icml")
    )

    check = matrix["venues"][0]["checks"]["stage3_pdf_checkpoint_complete"]
    assert check["passed"] is False
    assert "does not match CAS" in check["artifact_error"]


@pytest.mark.parametrize(
    ("statement", "expected_error"),
    (
        (
            "UPDATE text_extractions SET source_sha256 = 'ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff'",
            "does not trace to the exact Stage 3 PDF",
        ),
        (
            "UPDATE analysis_dispatches SET artifact_hash = 'ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff'",
            "artifact_hash does not match",
        ),
    ),
)
def test_stage4_requires_exact_stage3_lineage_and_input_artifact(
    tmp_path: Path, statement: str, expected_error: str
) -> None:
    run_dir = _venue_run(tmp_path)
    database = sqlite3.connect(run_dir / "papers.sqlite3")
    database.execute(statement)
    database.commit()
    database.close()
    module = _module()

    matrix = module.build_matrix(
        tmp_path, venue_catalog_root=_catalog(tmp_path, "icml")
    )

    check = matrix["venues"][0]["checks"]["stage4_luna_invocation_one"]
    assert check["passed"] is False
    assert expected_error in check["artifact_error"]


def test_stage4b_rejects_a_sol_output_hash_not_backed_by_cas(tmp_path: Path) -> None:
    run_dir = _venue_run(tmp_path)
    database = sqlite3.connect(run_dir / "papers.sqlite3")
    relative_path = database.execute(
        "SELECT relative_path FROM artifacts WHERE artifact_id = 'sol-output-1'"
    ).fetchone()[0]
    database.close()
    (run_dir / "artifacts" / relative_path).write_bytes(b'{"tampered":true}')
    module = _module()

    matrix = module.build_matrix(
        tmp_path, venue_catalog_root=_catalog(tmp_path, "icml")
    )

    check = matrix["venues"][0]["checks"]["stage4b_sol_one_shot_only"]
    assert check["passed"] is False
    assert "does not match CAS" in check["output_artifact_error"]


def test_report_manifest_and_complete_verifier_are_both_required(tmp_path: Path) -> None:
    run_dir = _venue_run(tmp_path)
    report_dir = run_dir / "reports" / "report-1"
    verification_path = report_dir / "VERIFICATION.json"
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    verification["checks"]["citation_coverage"] = False
    verification_path.write_text(json.dumps(verification), encoding="utf-8")
    manifest_path = report_dir / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["VERIFICATION.json"] = content_hash(verification)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    module = _module()

    matrix = module.build_matrix(
        tmp_path, venue_catalog_root=_catalog(tmp_path, "icml")
    )

    check = matrix["venues"][0]["checks"]["verify_passed"]
    assert check["passed"] is False
    assert check["verifier_matches_saved"] is False


def test_report_audit_hash_bindings_are_recomputed_not_trusted(tmp_path: Path) -> None:
    run_dir = _venue_run(tmp_path)
    report_dir = run_dir / "reports" / "report-1"
    audit_path = report_dir / "AUDIT.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["report_document_hash"] = "0" * 64
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    manifest_path = report_dir / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["AUDIT.json"] = content_hash(audit)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    module = _module()

    matrix = module.build_matrix(
        tmp_path, venue_catalog_root=_catalog(tmp_path, "icml")
    )

    check = matrix["venues"][0]["checks"]["verify_passed"]
    assert check["passed"] is False
    assert "audit does not exhaustively cover" in check["error"]


def test_neurips_import_is_hash_pinned_and_disclosed_as_historical() -> None:
    module = _module()

    rows = module._load_acceptance_imports(IMPORT_MANIFEST)

    assert len(rows) == 1
    row = rows[0]
    assert row["venue"] == "neurips"
    assert row["passed"] is True
    assert row["binding_source"] == "acceptance-evidence-import"
    assert row["evidence_reuse"] == "reused_historical_evidence"
    assert row["checks"]["verify_passed"]["complete_verifier_status"] == (
        "historical_attestation_only"
    )


def test_acceptance_import_fails_closed_on_evidence_hash_drift(tmp_path: Path) -> None:
    manifest = json.loads(IMPORT_MANIFEST.read_text(encoding="utf-8"))
    manifest["repository_root"] = str(ROOT)
    manifest["imports"][0]["sha256"] = "0" * 64
    changed = tmp_path / "imports.json"
    changed.write_text(json.dumps(manifest), encoding="utf-8")
    module = _module()

    with pytest.raises(module.MatrixError, match="SHA-256 mismatch"):
        module._load_acceptance_imports(changed)


def test_acceptance_import_requires_an_explicit_supported_evidence_schema(
    tmp_path: Path,
) -> None:
    manifest = json.loads(IMPORT_MANIFEST.read_text(encoding="utf-8"))
    manifest["repository_root"] = str(ROOT)
    manifest["imports"][0].pop("evidence_schema_version")
    changed = tmp_path / "imports.json"
    changed.write_text(json.dumps(manifest), encoding="utf-8")
    module = _module()

    with pytest.raises(module.MatrixError, match="evidence schema_version"):
        module._load_acceptance_imports(changed)

    with pytest.raises(module.MatrixError, match="historical acceptance evidence"):
        module._historical_evidence_row(
            venue_id="neurips",
            evidence_path=tmp_path / "evidence.json",
            evidence_sha256="0" * 64,
            document={},
            repository_root=tmp_path,
        )


@pytest.mark.parametrize(
    ("implementation_version", "passed", "qualification"),
    (
        (
            "stage4b-one-shot-v1",
            True,
            "legacy_v1_reverified_by_current_verifier",
        ),
        ("unknown-stage4b", False, "unsupported"),
    ),
)
def test_stage4b_implementation_version_is_explicit_and_fail_closed(
    tmp_path: Path,
    implementation_version: str,
    passed: bool,
    qualification: str,
) -> None:
    run_dir = _venue_run(tmp_path)
    database = sqlite3.connect(run_dir / "papers.sqlite3")
    database.execute(
        "UPDATE pipeline_runs SET implementation_version = ? WHERE run_id = 'report-pipeline-1'",
        (implementation_version,),
    )
    database.commit()
    database.close()
    module = _module()

    matrix = module.build_matrix(
        tmp_path, venue_catalog_root=_catalog(tmp_path, "icml")
    )

    check = matrix["venues"][0]["checks"]["stage4b_sol_one_shot_only"]
    assert check["passed"] is passed
    assert check["implementation_version"] == implementation_version
    assert check["implementation_qualification"] == qualification
    assert matrix["summary"]["all_passed"] is passed


def test_real_catalog_reaches_20_of_20_only_with_explicit_neurips_import(
    tmp_path: Path,
) -> None:
    module = _module()
    catalog_ids = module._venue_catalog_ids(ROOT / "venues")
    assert len(catalog_ids) == 20
    for venue_id in catalog_ids:
        if venue_id != "neurips":
            _venue_run(tmp_path, venue_id)

    without_import = module.build_matrix(
        tmp_path, venue_catalog_root=ROOT / "venues"
    )
    with_import = module.build_matrix(
        tmp_path,
        venue_catalog_root=ROOT / "venues",
        acceptance_import_manifest=IMPORT_MANIFEST,
    )

    assert without_import["summary"]["all_passed"] is False
    assert without_import["summary"]["missing_venues"] == ["neurips"]
    assert with_import["summary"] == {
        "venue_count": 20,
        "catalog_venue_count": 20,
        "passed": 20,
        "failed": 0,
        "coverage_complete": True,
        "missing_venues": [],
        "unexpected_venues": [],
        "duplicate_venues": [],
        "all_passed": True,
    }


def test_extra_repair_dispatch_fails_the_sol_one_shot_gate(tmp_path: Path) -> None:
    run_dir = _venue_run(tmp_path)
    database = sqlite3.connect(run_dir / "papers.sqlite3")
    database.execute(
        "INSERT INTO report_audit_steps VALUES (?, ?, ?)",
        ("report-1", "repair", 1),
    )
    database.commit()
    database.close()
    module = _module()

    matrix = module.build_matrix(
        tmp_path, venue_catalog_root=_catalog(tmp_path, "icml")
    )

    one_shot = matrix["venues"][0]["checks"]["stage4b_sol_one_shot_only"]
    assert one_shot["passed"] is False
    assert one_shot["repair_steps"] == 1
    assert one_shot["repair_dispatches"] == 1


def test_main_writes_both_formats_and_returns_nonzero_for_incomplete_catalog(
    tmp_path: Path,
) -> None:
    _venue_run(tmp_path)
    module = _module()
    json_output = tmp_path / "matrix.json"
    markdown_output = tmp_path / "matrix.md"

    status = module.main(
        [
            "--run-root",
            str(tmp_path),
            "--json-output",
            str(json_output),
            "--markdown-output",
            str(markdown_output),
        ]
    )

    assert status == 1
    summary = json.loads(json_output.read_text())["summary"]
    assert summary["coverage_complete"] is False
    assert "## Catalog coverage" in markdown_output.read_text()
