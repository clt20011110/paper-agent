from __future__ import annotations

import json
from pathlib import Path

import pytest

from paper_agent import cli
from paper_agent.canonical import content_hash
from paper_agent.report_artifacts import (
    ReportArtifactStore,
    audit_coverage_ledger,
    audit_rubric_hash,
    report_artifact_hash,
)
from paper_agent.report_cli_service import (
    approve_report_plan_from_files,
    compile_report_plan_from_files,
    diff_report_runs,
    load_report_run_bundle,
    verify_report_run,
)
from paper_agent.report_config import ReportResources
from paper_agent.report_plan import ReportPlanError
from test_report_artifacts import _bundle
from test_report_plan import _draft, _inputs


def _write_json(path: Path, document: object) -> Path:
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_plan_only_and_approve_save_the_frozen_plan_bundle(tmp_path: Path) -> None:
    corpus, search_audit = _inputs()
    draft = _draft()
    draft["created_at"] = "2026-08-10T00:02:00Z"
    draft_path = _write_json(tmp_path / "draft.json", draft)
    corpus_path = _write_json(tmp_path / "corpus.json", corpus)
    audit_path = _write_json(tmp_path / "search-audit.json", search_audit)

    preview = compile_report_plan_from_files(
        draft_path, corpus_path, audit_path, tmp_path, save_draft=False
    )
    assert not preview.saved
    assert not preview.path.exists()

    compiled = compile_report_plan_from_files(
        draft_path, corpus_path, audit_path, tmp_path
    )

    assert compiled.saved
    assert compiled.path.is_file()
    assert json.loads(compiled.path.read_text(encoding="utf-8"))["plan_hash"] == compiled.plan["plan_hash"]

    approved = approve_report_plan_from_files(
        compiled.path,
        corpus_path,
        audit_path,
        tmp_path,
        expected_hash=str(compiled.plan["plan_hash"]),
        approved_by="owner",
        approved_at="2026-08-10T00:03:00Z",
    )

    assert approved.path.name == "REPORT_PLAN.json"
    assert approved.plan["approval"]["approved_hash"] == compiled.plan["plan_hash"]
    assert (approved.path.parent / "CORPUS_SNAPSHOT.json").is_file()
    assert (approved.path.parent / "SEARCH_AUDIT.json").is_file()


def test_approval_rejects_resources_that_differ_from_plan_compilation(
    tmp_path: Path,
) -> None:
    corpus, search_audit = _inputs()
    draft = _draft()
    draft["created_at"] = "2026-08-10T00:02:00Z"
    draft_path = _write_json(tmp_path / "draft.json", draft)
    corpus_path = _write_json(tmp_path / "corpus.json", corpus)
    audit_path = _write_json(tmp_path / "search-audit.json", search_audit)
    compiled = compile_report_plan_from_files(
        draft_path, corpus_path, audit_path, tmp_path
    )
    defaults = ReportResources.defaults()
    changed_prompt = tmp_path / "report-plan.md"
    changed_prompt.write_text(
        defaults.prompt("planning_assist") + "\nChanged after planning.\n",
        encoding="utf-8",
    )
    prompt_paths = dict(defaults.prompt_paths)
    prompt_paths["planning_assist"] = changed_prompt
    changed_resources = ReportResources(
        dict(defaults.schema_paths), prompt_paths, configured=True
    )

    with pytest.raises(ReportPlanError, match="prompt hashes"):
        approve_report_plan_from_files(
            compiled.path,
            corpus_path,
            audit_path,
            tmp_path,
            expected_hash=str(compiled.plan["plan_hash"]),
            approved_by="owner",
            approved_at="2026-08-10T00:03:00Z",
            resources=changed_resources,
        )


def test_approval_dry_run_still_validates_frozen_corpus_binding(
    tmp_path: Path,
) -> None:
    corpus, search_audit = _inputs()
    draft = _draft()
    draft["created_at"] = "2026-08-10T00:02:00Z"
    draft_path = _write_json(tmp_path / "draft.json", draft)
    corpus_path = _write_json(tmp_path / "corpus.json", corpus)
    audit_path = _write_json(tmp_path / "search-audit.json", search_audit)
    compiled = compile_report_plan_from_files(
        draft_path, corpus_path, audit_path, tmp_path
    )
    corpus_path.write_text(
        json.dumps({**corpus, "snapshot_hash": "b" * 64}), encoding="utf-8"
    )

    with pytest.raises(ReportPlanError, match="corpus snapshot hash has drifted"):
        approve_report_plan_from_files(
            compiled.path,
            corpus_path,
            audit_path,
            tmp_path,
            expected_hash=str(compiled.plan["plan_hash"]),
            approved_by="owner",
            approved_at="2026-08-10T00:03:00Z",
            save_bundle=False,
        )


def test_verify_report_run_loads_an_immutable_bundle(tmp_path: Path) -> None:
    bundle = _bundle()
    output = _publish(tmp_path, bundle)

    checklist = verify_report_run(tmp_path, output.name)

    assert checklist["coverage_complete"]


def test_diff_report_runs_loads_each_bundle_and_current_relations(tmp_path: Path) -> None:
    _write_minimal_run(tmp_path, "previous", claims=[], papers=[])
    _write_minimal_run(
        tmp_path,
        "current",
        claims=[{"claim_id": "claim-1", "report_section": "evidence"}],
        papers=[{"paper_id": "paper-1"}],
    )

    diff = diff_report_runs(tmp_path, "previous", "current")

    assert diff["added_claim_ids"] == ["claim-1"]
    assert diff["added_paper_ids"] == ["paper-1"]
    assert diff["unmapped_claim_ids"] == ["claim-1"]


def test_report_bundle_directory_cannot_impersonate_another_run(tmp_path: Path) -> None:
    _write_minimal_run(tmp_path, "expected", claims=[], papers=[])
    document_path = tmp_path / "reports/expected/REPORT_DOCUMENT.json"
    document_path.write_text(
        json.dumps({"report_run_id": "foreign"}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="does not match"):
        load_report_run_bundle(tmp_path, "expected")


def test_report_plan_approve_verify_and_diff_cli(tmp_path: Path, capsys) -> None:
    corpus, search_audit = _inputs()
    draft = _draft()
    draft["created_at"] = "2026-08-10T00:02:00Z"
    draft_path = _write_json(tmp_path / "draft.json", draft)
    corpus_path = _write_json(tmp_path / "corpus.json", corpus)
    audit_path = _write_json(tmp_path / "search-audit.json", search_audit)

    assert cli.main([
        "report", "--plan-only", "--draft", str(draft_path),
        "--corpus-snapshot", str(corpus_path), "--search-audit", str(audit_path),
        "--output-root", str(tmp_path), "--dry-run",
    ]) == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["status"] == "validated"
    assert not Path(preview["draft_path"]).exists()

    assert cli.main([
        "report", "--plan-only", "--draft", str(draft_path),
        "--corpus-snapshot", str(corpus_path), "--search-audit", str(audit_path),
        "--output-root", str(tmp_path),
    ]) == 0
    planned = json.loads(capsys.readouterr().out)
    assert Path(planned["draft_path"]).is_file()

    assert cli.main([
        "report", "approve", "--plan", planned["draft_path"],
        "--hash", planned["plan_hash"], "--approved-by", "owner",
        "--approved-at", "2026-08-10T00:03:00Z",
        "--corpus-snapshot", str(corpus_path), "--search-audit", str(audit_path),
        "--output-root", str(tmp_path), "--dry-run",
    ]) == 0
    approval_preview = json.loads(capsys.readouterr().out)
    assert approval_preview["write_performed"] is False
    assert not Path(approval_preview["path"]).exists()

    published = _publish(tmp_path, _bundle())
    assert cli.main([
        "verify-report", "--output-root", str(tmp_path),
        "--report-run-id", published.name,
    ]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "passed"

    _write_minimal_run(tmp_path, "previous", claims=[], papers=[])
    _write_minimal_run(
        tmp_path,
        "current",
        claims=[{"claim_id": "claim-1", "report_section": "evidence"}],
        papers=[{"paper_id": "paper-1"}],
    )
    assert cli.main([
        "report", "--diff-from", "previous", "--report-run-id", "current",
        "--output-root", str(tmp_path),
    ]) == 0
    diff = json.loads(capsys.readouterr().out)
    assert diff["diff"]["added_claim_ids"] == ["claim-1"]


def _publish(root: Path, bundle: dict) -> Path:
    audit = {
        "audit_pass": "A",
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
        "search_limitations_hash": content_hash([
            "抽取范围：full_pdf=1；全文、摘要和元数据证据已分层，缺失全文不作全文事实表述。"
        ]),
        "coverage_complete": True,
        "coverage_ledger": audit_coverage_ledger(bundle["document"], bundle["claims"]),
        "findings": [],
    }
    return ReportArtifactStore(root).write(
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


def _write_minimal_run(
    root: Path, run_id: str, *, claims: list[dict], papers: list[dict]
) -> None:
    directory = root / "reports" / run_id
    directory.mkdir(parents=True)
    documents = {
        "REPORT_PLAN.json": {"sections": [{"id": "evidence"}]},
        "SEARCH_AUDIT.json": {},
        "CORPUS_SNAPSHOT.json": {"papers": papers},
        "COMPARISON_GROUPS.json": {},
        "CLAIM_RELATIONS.json": [],
        "REPORT_DOCUMENT.json": {"report_run_id": run_id},
        "COVERAGE.json": {},
        "BIBLIOGRAPHY.json": {},
    }
    for name, document in documents.items():
        _write_json(directory / name, document)
    (directory / "CLAIMS_EVIDENCE.jsonl").write_text(
        "".join(json.dumps(claim) + "\n" for claim in claims), encoding="utf-8"
    )
