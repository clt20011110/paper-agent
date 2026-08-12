from __future__ import annotations

import json
from hashlib import sha256
from types import SimpleNamespace

import pytest

from paper_agent.schema import validate
from paper_agent.stage2_evaluation import RationaleStratum, rationale_audit_gate
from paper_agent.stage2_rationale_workflow import (
    EVIDENCE_SUPPORT_RUBRIC_HASH,
    SEVERE_FABRICATION_RUBRIC_HASH,
    RationaleAuditExample,
    derive_rationale_audit_examples,
    freeze_rationale_audit,
    import_completed_rationale_audit,
    rationale_audit_examples_from_document,
    rationale_audit_records_document,
    qwen_adjudication_ledger_document,
    write_rationale_audit_artifacts,
    write_rationale_worklist_no_replace,
)


def _derived_source_inputs() -> tuple[object, ...]:
    papers = []
    assignments = []
    records = []
    for language in ("en", "zh"):
        for decision in ("relevant", "needs_review"):
            for index in range(25):
                paper_id = f"{language}-{decision}-{index:02d}"
                papers.append({
                    "paper_id": paper_id,
                    "title": f"{language} title {index}",
                    "abstract": f"{language} abstract {index}",
                    "keywords": ["molecule", "generation"],
                })
                assignments.append({
                    "paper_id": paper_id,
                    "language": language,
                    "topic": "molecular_generation",
                    "query_version": "q-v1",
                    "query": "molecular generation",
                })
                records.append({
                    "paper_id": paper_id,
                    "decision": decision,
                    "score": 0.9 if decision == "relevant" else 0.5,
                    "rationale": f"ledger rationale {paper_id}",
                    "evidence_fields": ["title", "abstract", "keywords"],
                })
    papers_document = {
        "schema_version": "1", "kind": "stage2_benchmark_papers", "papers": papers,
    }
    papers_bytes = json.dumps(papers_document, sort_keys=True).encode()
    metadata = {
        "schema_version": "1",
        "kind": "stage2_rationale_query_metadata",
        "benchmark_papers_sha256": sha256(papers_bytes).hexdigest(),
        "primary_languages": ["en", "zh"],
        "assignments": assignments,
    }
    metadata_bytes = json.dumps(metadata, sort_keys=True).encode()
    candidate = SimpleNamespace(
        profile_name="candidate-v2",
        release_hash="a" * 64,
        profile=SimpleNamespace(
            adjudicator_model_id="Qwen/Qwen3.5-9B-MLX-8bit",
            adjudicator_lock_hash="b" * 64,
            prompt_version="stage2-adjudication-v1",
            schema_version="filter-decision.schema.json",
        ),
    )
    from paper_agent.stage2_benchmark_inputs import benchmark_corpus_hash, benchmark_papers_from_document

    ledger = {
        "schema_version": "1",
        "kind": "stage2_qwen_adjudication_ledger",
        "candidate": {
            "candidate_id": candidate.profile_name,
            "bundle_sha256": "c" * 64,
            "release_hash": candidate.release_hash,
            "adjudicator_model_id": candidate.profile.adjudicator_model_id,
            "adjudicator_model_lock_hash": candidate.profile.adjudicator_lock_hash,
            "prompt_version": candidate.profile.prompt_version,
            "response_schema": candidate.profile.schema_version,
        },
        "benchmark_papers_sha256": sha256(papers_bytes).hexdigest(),
        "corpus_hash": benchmark_corpus_hash(benchmark_papers_from_document(papers_document)),
        "query_metadata_sha256": sha256(metadata_bytes).hexdigest(),
        "records": records,
    }
    return ledger, candidate, papers_document, metadata


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
    document = rationale_audit_records_document(records, worklist_sha256="a" * 64)
    validate(frozen.manifest.document(), "stage2-rationale-audit-manifest.schema.json")
    validate(document, "stage2-rationale-audit-records.schema.json")
    assert rationale_audit_gate(frozen.manifest, records).passed

    manifest_path = tmp_path / "rationale-manifest.json"
    records_path = tmp_path / "rationale-records.json"
    write_rationale_audit_artifacts(
        frozen, records, manifest_path=manifest_path, records_path=records_path,
        worklist_sha256="a" * 64,
    )
    assert json.loads(manifest_path.read_text()) == frozen.manifest.document()
    assert json.loads(records_path.read_text()) == document
    with pytest.raises(FileExistsError):
        write_rationale_audit_artifacts(
            frozen, records, manifest_path=manifest_path, records_path=records_path,
            worklist_sha256="a" * 64,
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


def test_derivation_uses_only_bound_ledger_rationale_and_frozen_paper_fields() -> None:
    ledger, candidate, papers, metadata = _derived_source_inputs()
    ledger_bytes = json.dumps(ledger, sort_keys=True).encode()
    papers_bytes = json.dumps(papers, sort_keys=True).encode()
    metadata_bytes = json.dumps(metadata, sort_keys=True).encode()

    document = derive_rationale_audit_examples(
        ledger,
        source_ledger_sha256=sha256(ledger_bytes).hexdigest(),
        candidate=candidate,
        candidate_bundle_sha256="c" * 64,
        benchmark_papers_document=papers,
        benchmark_papers_sha256=sha256(papers_bytes).hexdigest(),
        query_metadata=metadata,
        query_metadata_sha256=sha256(metadata_bytes).hexdigest(),
    )

    assert len(document["examples"]) == 100
    assert all(row["rationale"].startswith("ledger rationale") for row in document["examples"])
    assert all(row["rationale_artifact_hash"] == sha256(ledger_bytes).hexdigest() for row in document["examples"])
    assert all("title:" in row["evidence"] and "abstract:" in row["evidence"] for row in document["examples"])
    examples, corpus_hash, model_lock_hash = rationale_audit_examples_from_document(document)
    frozen = freeze_rationale_audit(examples, corpus_hash=corpus_hash, model_lock_hash=model_lock_hash, reviewer_id="reviewer")
    assert len(frozen.manifest.cases) == 100
    drifted = json.loads(json.dumps(document))
    drifted["examples"][0]["rationale_artifact_hash"] = "0" * 64
    with pytest.raises(ValueError, match="source ledger"):
        rationale_audit_examples_from_document(drifted)

    ledger["candidate"]["bundle_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="frozen Stage 2 candidate"):
        derive_rationale_audit_examples(
            ledger,
            source_ledger_sha256=sha256(ledger_bytes).hexdigest(),
            candidate=candidate,
            candidate_bundle_sha256="c" * 64,
            benchmark_papers_document=papers,
            benchmark_papers_sha256=sha256(papers_bytes).hexdigest(),
            query_metadata=metadata,
            query_metadata_sha256=sha256(metadata_bytes).hexdigest(),
        )


def test_qwen_ledger_producer_accepts_only_typed_complete_stage2_decisions() -> None:
    _ledger, candidate, papers, _metadata = _derived_source_inputs()
    papers_bytes = json.dumps(papers, sort_keys=True).encode()
    decisions = tuple(
        SimpleNamespace(
            paper_id=paper["paper_id"],
            route=SimpleNamespace(value=("relevant" if index % 2 else "needs_review")),
            adjudicator_score=0.9 if index % 2 else 0.5,
            rationale=f"actual qwen rationale {index}",
            evidence_fields=("title", "abstract"),
            adjudicated=True,
        )
        for index, paper in enumerate(papers["papers"])
    )

    document = qwen_adjudication_ledger_document(
        decisions,
        candidate=candidate,
        candidate_bundle_sha256="c" * 64,
        benchmark_papers_document=papers,
        benchmark_papers_sha256=sha256(papers_bytes).hexdigest(),
        query_metadata_sha256="d" * 64,
    )
    assert document["kind"] == "stage2_qwen_adjudication_ledger"
    assert len(document["records"]) == 100

    with pytest.raises(ValueError, match="Stage2Decision"):
        qwen_adjudication_ledger_document(
            ({"rationale": "free text"},) * 100,
            candidate=candidate,
            candidate_bundle_sha256="c" * 64,
            benchmark_papers_document=papers,
            benchmark_papers_sha256=sha256(papers_bytes).hexdigest(),
            query_metadata_sha256="d" * 64,
        )
