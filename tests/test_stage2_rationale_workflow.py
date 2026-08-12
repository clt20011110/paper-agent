from __future__ import annotations

import json

import pytest

from paper_agent.schema import validate
from paper_agent.stage2_evaluation import RationaleStratum, rationale_audit_gate
from paper_agent.stage2_rationale_workflow import (
    EVIDENCE_SUPPORT_RUBRIC_HASH,
    SEVERE_FABRICATION_RUBRIC_HASH,
    RationaleAuditExample,
    freeze_rationale_audit,
    import_completed_rationale_audit,
    rationale_audit_records_document,
    write_rationale_audit_artifacts,
    write_rationale_worklist_no_replace,
)


def _examples() -> tuple[RationaleAuditExample, ...]:
    return tuple(
        RationaleAuditExample(
            pair_id=f"pair-{stratum.value}-{language}-{index}",
            stratum=stratum,
            language=language,
            rationale_artifact_hash=f"{index + (0 if language == 'en' else 100):064x}",
            evidence=f"Frozen {language} evidence for {stratum.value} example {index}.",
            rationale=f"Frozen rationale for {stratum.value} example {index}.",
        )
        for stratum in RationaleStratum
        for language in ("en", "zh")
        for index in range(25)
    )


def _frozen():
    return freeze_rationale_audit(
        _examples(),
        corpus_hash="c" * 64,
        model_lock_hash="d" * 64,
        reviewer_id="reviewer-7",
    )


def test_freeze_creates_an_unlabelled_stratified_human_worklist() -> None:
    frozen = _frozen()

    assert len(frozen.manifest.cases) == 100
    assert frozen.manifest.evidence_rubric_hash == EVIDENCE_SUPPORT_RUBRIC_HASH
    assert frozen.manifest.fabrication_rubric_hash == SEVERE_FABRICATION_RUBRIC_HASH
    assert frozen.worklist["manifest_hash"] == frozen.manifest.hash()
    assert all(row["evidence_supported"] is None for row in frozen.worklist["rows"])
    assert all(row["severe_fabrication"] is None for row in frozen.worklist["rows"])
    assert all(row["content_hash"] for row in frozen.worklist["rows"])
    assert frozen.worklist["rows"][0]["evidence"].startswith("Frozen")
    with pytest.raises(ValueError, match="reviewer_id"):
        freeze_rationale_audit(
            _examples(), corpus_hash="c" * 64, model_lock_hash="d" * 64, reviewer_id="  "
        )


def test_import_requires_explicit_human_labels_and_emits_existing_schema(tmp_path) -> None:
    frozen = _frozen()
    with pytest.raises(ValueError, match="unfilled human labels"):
        import_completed_rationale_audit(frozen.worklist, manifest=frozen.manifest)

    completed = dict(frozen.worklist)
    completed["rows"] = [
        {**row, "evidence_supported": True, "severe_fabrication": False}
        for row in frozen.worklist["rows"]
    ]
    records = import_completed_rationale_audit(completed, manifest=frozen.manifest)
    document = rationale_audit_records_document(records)
    validate(frozen.manifest.document(), "stage2-rationale-audit-manifest.schema.json")
    validate(document, "stage2-rationale-audit-records.schema.json")
    assert rationale_audit_gate(frozen.manifest, records).passed

    manifest_path = tmp_path / "rationale-manifest.json"
    records_path = tmp_path / "rationale-records.json"
    write_rationale_audit_artifacts(
        frozen, records, manifest_path=manifest_path, records_path=records_path
    )
    assert json.loads(manifest_path.read_text()) == frozen.manifest.document()
    assert json.loads(records_path.read_text()) == document
    with pytest.raises(FileExistsError):
        write_rationale_audit_artifacts(
            frozen, records, manifest_path=manifest_path, records_path=records_path
        )
    assert json.loads(records_path.read_text()) == document

    worklist_path = tmp_path / "new-worklist-directory" / "rationale-worklist.json"
    write_rationale_worklist_no_replace(worklist_path, frozen.worklist)
    assert worklist_path.exists()
    with pytest.raises(FileExistsError):
        write_rationale_worklist_no_replace(worklist_path, frozen.worklist)


def test_import_rejects_a_changed_frozen_case_provenance() -> None:
    frozen = _frozen()
    completed = dict(frozen.worklist)
    completed["rows"] = [
        {**row, "evidence_supported": True, "severe_fabrication": False}
        for row in frozen.worklist["rows"]
    ]
    completed["rows"][0]["language"] = "fr"

    with pytest.raises(ValueError, match="changed frozen provenance"):
        import_completed_rationale_audit(completed, manifest=frozen.manifest)


def test_import_rejects_evidence_or_rationale_drift() -> None:
    frozen = _frozen()
    completed = dict(frozen.worklist)
    completed["rows"] = [
        {**row, "evidence_supported": True, "severe_fabrication": False}
        for row in frozen.worklist["rows"]
    ]
    completed["rows"][0]["evidence"] = "Changed after freezing."

    with pytest.raises(ValueError, match="content drifted"):
        import_completed_rationale_audit(completed, manifest=frozen.manifest)
