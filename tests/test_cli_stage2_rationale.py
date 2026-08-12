from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from paper_agent import cli
from test_stage2_rationale_workflow import _derived_source_inputs


def _examples_document() -> dict[str, object]:
    return {
        "schema_version": "1",
        "kind": "stage2_rationale_audit_examples",
        "corpus_hash": "c" * 64,
        "model_lock_hash": "d" * 64,
        "examples": [
            {
                "pair_id": f"pair-{stratum}-{language}-{index}",
                "stratum": stratum,
                "language": language,
                "rationale_artifact_hash": f"{offset + index:064x}",
                "evidence": f"Frozen {language} evidence {stratum} {index}",
                "rationale": f"Frozen rationale {stratum} {index}",
            }
            for stratum in ("relevant", "boundary")
            for language, offset in (("en", 0), ("zh", 100))
            for index in range(25)
        ],
    }


def test_rationale_cli_freezes_then_imports_only_explicit_human_labels(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    examples_path = tmp_path / "examples.json"
    examples_path.write_text(json.dumps(_examples_document()), encoding="utf-8")
    manifest_path = tmp_path / "audit" / "manifest.json"
    worklist_path = tmp_path / "audit" / "worklist.json"
    freeze_args = [
        "stage2-rationale", "freeze-worklist",
        "--examples", str(examples_path),
        "--reviewer-id", "reviewer-7",
        "--manifest-output", str(manifest_path),
        "--worklist-output", str(worklist_path),
    ]

    assert cli.main(["--dry-run", *freeze_args]) == 0
    dry_run = json.loads(capsys.readouterr().out)
    assert dry_run["row_count"] == 100
    assert dry_run["written"] is False
    assert not manifest_path.exists()
    assert not worklist_path.exists()

    assert cli.main(freeze_args) == 0
    frozen = json.loads(capsys.readouterr().out)
    assert frozen["status"] == "complete"
    assert manifest_path.exists()
    worklist = json.loads(worklist_path.read_text(encoding="utf-8"))
    assert all(row["evidence_supported"] is None for row in worklist["rows"])
    assert all(row["severe_fabrication"] is None for row in worklist["rows"])

    completed_path = tmp_path / "completed.json"
    worklist["rows"] = [
        {**row, "evidence_supported": True, "severe_fabrication": False}
        for row in worklist["rows"]
    ]
    completed_path.write_text(json.dumps(worklist), encoding="utf-8")
    records_path = tmp_path / "evidence" / "records.json"
    import_args = [
        "stage2-rationale", "import-worklist",
        "--manifest", str(manifest_path),
        "--worklist", str(completed_path),
        "--records-output", str(records_path),
    ]

    assert cli.main(["--dry-run", *import_args]) == 0
    imported_dry_run = json.loads(capsys.readouterr().out)
    assert imported_dry_run["record_count"] == 100
    assert imported_dry_run["written"] is False
    assert not records_path.exists()

    assert cli.main(import_args) == 0
    imported = json.loads(capsys.readouterr().out)
    assert imported["evidence_support_rate"] == 1
    assert imported["severe_fabrication_rate"] == 0
    assert imported["written"] is True
    records = json.loads(records_path.read_text(encoding="utf-8"))
    assert len(records["records"]) == 100


def test_rationale_cli_rejects_unfilled_human_labels(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    examples_path = tmp_path / "examples.json"
    examples_path.write_text(json.dumps(_examples_document()), encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    worklist_path = tmp_path / "worklist.json"
    assert cli.main([
        "stage2-rationale", "freeze-worklist",
        "--examples", str(examples_path),
        "--reviewer-id", "reviewer-7",
        "--manifest-output", str(manifest_path),
        "--worklist-output", str(worklist_path),
    ]) == 0
    capsys.readouterr()

    with pytest.raises(ValueError, match="unfilled human labels"):
        cli.main([
            "--dry-run", "stage2-rationale", "import-worklist",
            "--manifest", str(manifest_path),
            "--worklist", str(worklist_path),
            "--records-output", str(tmp_path / "records.json"),
        ])


def test_rationale_cli_derives_bound_examples_without_writing_in_dry_run(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, candidate, papers, metadata = _derived_source_inputs()
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text("candidate-bytes", encoding="utf-8")
    candidate_sha = sha256(candidate_path.read_bytes()).hexdigest()
    ledger["candidate"]["bundle_sha256"] = candidate_sha
    papers_path = tmp_path / "papers.json"
    papers_path.write_text(json.dumps(papers, sort_keys=True), encoding="utf-8")
    papers_sha = sha256(papers_path.read_bytes()).hexdigest()
    metadata["benchmark_papers_sha256"] = papers_sha
    metadata_path = tmp_path / "queries.json"
    metadata_path.write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")
    ledger["benchmark_papers_sha256"] = papers_sha
    ledger["query_metadata_sha256"] = sha256(metadata_path.read_bytes()).hexdigest()
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(json.dumps(ledger, sort_keys=True), encoding="utf-8")
    output = tmp_path / "derived.json"
    monkeypatch.setattr(cli, "load_stage2_benchmark_candidate", lambda _path: candidate)
    args = [
        "stage2-rationale", "derive-examples",
        "--stage2-candidate", str(candidate_path),
        "--benchmark-papers", str(papers_path),
        "--query-metadata", str(metadata_path),
        "--adjudication-ledger", str(ledger_path),
        "--output", str(output),
    ]

    assert cli.main(["--dry-run", *args]) == 0
    dry = json.loads(capsys.readouterr().out)
    assert dry["example_count"] == 100
    assert dry["written"] is False
    assert not output.exists()

    assert cli.main(args) == 0
    written = json.loads(capsys.readouterr().out)
    assert written["written"] is True
    output_document = json.loads(output.read_text(encoding="utf-8"))
    assert output_document["kind"] == "stage2_rationale_audit_derived_examples"
