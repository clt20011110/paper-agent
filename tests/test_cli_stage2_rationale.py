from __future__ import annotations

import json
from pathlib import Path

import pytest

from paper_agent import cli


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
