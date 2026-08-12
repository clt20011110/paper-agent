from __future__ import annotations

import json
from hashlib import sha256
from threading import Lock
from types import SimpleNamespace

import pytest

from paper_agent.schema import validate
from paper_agent.canonical import content_hash
from paper_agent.stage2_backends import (
    AdjudicationDecision,
    RerankScore,
    StructuredOutputError,
)
from paper_agent.stage2_evaluation import RationaleStratum, rationale_audit_gate
from paper_agent.stage2_rationale_workflow import (
    EVIDENCE_SUPPORT_RUBRIC_HASH,
    SEVERE_FABRICATION_RUBRIC_HASH,
    RationaleAuditExample,
    collect_rationale_source_artifacts,
    derive_rationale_audit_examples,
    freeze_rationale_audit,
    import_completed_rationale_audit,
    rationale_audit_examples_from_document,
    rationale_audit_records_document,
    qwen_adjudication_ledger_document,
    write_rationale_audit_artifacts,
    write_rationale_source_artifacts_no_replace,
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
                    "pair_id": content_hash({
                        "kind": "stage2-rationale-pair-v1",
                        "source_paper_id": paper_id,
                        "topic": "molecular_generation",
                        "language": language,
                        "query_version": "q-v1",
                        "query": "molecular generation",
                    }),
                    "source_paper_id": paper_id,
                    "language": language,
                    "topic": "molecular_generation",
                    "query_version": "q-v1",
                    "query": "molecular generation",
                    "stratum": "relevant" if decision == "relevant" else "boundary",
                    "reranker_raw_score": 0.9 if decision == "relevant" else 0.5,
                    "reranker_probability": 0.9 if decision == "relevant" else 0.5,
                })
                records.append({
                    "pair_id": assignments[-1]["pair_id"],
                    "decision": decision,
                    "score": 0.9 if decision == "relevant" else 0.5,
                    "rationale": f"ledger rationale {paper_id}",
                    "evidence_fields": ["title", "abstract", "keywords"],
                })
    scores = []
    for language in ("en", "zh"):
        query = "molecular generation"
        for paper in papers:
            same_language = paper["paper_id"].startswith(f"{language}-")
            raw_score = (
                0.9 if same_language and "relevant" in paper["paper_id"]
                else 0.5 if same_language
                else 0.0
            )
            scores.append({
                "pair_id": content_hash({
                    "kind": "stage2-rationale-pair-v1",
                    "source_paper_id": paper["paper_id"],
                    "topic": "molecular_generation",
                    "language": language,
                    "query_version": "q-v1",
                    "query": query,
                }),
                "source_paper_id": paper["paper_id"],
                "language": language,
                "topic": "molecular_generation",
                "query_version": "q-v1",
                "query": query,
                "stratum": (
                    "relevant" if raw_score == 0.9
                    else "boundary" if raw_score == 0.5
                    else "irrelevant"
                ),
                "reranker_raw_score": raw_score,
                "reranker_probability": raw_score,
            })
    papers_document = {
        "schema_version": "1", "kind": "stage2_benchmark_papers", "papers": papers,
    }
    papers_bytes = json.dumps(papers_document, sort_keys=True).encode()
    metadata = {
        "schema_version": "3",
        "kind": "stage2_rationale_query_metadata",
        "candidate_id": "candidate-v2",
        "candidate_bundle_sha256": "c" * 64,
        "benchmark_papers_sha256": sha256(papers_bytes).hexdigest(),
        "primary_languages": ["en", "zh"],
        "scores": sorted(scores, key=lambda row: row["pair_id"]),
        "assignments": sorted(assignments, key=lambda row: row["pair_id"]),
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
            query_version="q-v1",
            evaluation_topic_queries=(
                ("molecular_generation", "en", "molecular generation"),
                ("molecular_generation", "zh", "molecular generation"),
            ),
            evaluation_topic_query_map={
                ("molecular_generation", "en"): "molecular generation",
                ("molecular_generation", "zh"): "molecular generation",
            },
            reranker_calibration=SimpleNamespace(
                calibrator=SimpleNamespace(predict=lambda score: score),
                threshold=SimpleNamespace(low=0.2, high=0.8),
            ),
        ),
    )
    from paper_agent.stage2_benchmark_inputs import benchmark_corpus_hash, benchmark_papers_from_document

    ledger = {
        "schema_version": "2",
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
    _ledger, candidate, papers, metadata = _derived_source_inputs()
    papers_bytes = json.dumps(papers, sort_keys=True).encode()
    decisions = tuple(
        AdjudicationDecision(
            paper_id=metadata["assignments"][index]["pair_id"],
            decision="relevant" if index % 2 else "needs_review",
            score=0.9 if index % 2 else 0.5,
            reason_codes=("semantic_match",),
            rationale=f"actual qwen rationale {index}",
            evidence_fields=("title", "abstract"),
        )
        for index, _paper in enumerate(papers["papers"])
    )

    document = qwen_adjudication_ledger_document(
        decisions,
        candidate=candidate,
        candidate_bundle_sha256="c" * 64,
        benchmark_papers_document=papers,
        benchmark_papers_sha256=sha256(papers_bytes).hexdigest(),
        query_metadata_document=metadata,
        query_metadata_sha256="d" * 64,
    )
    assert document["kind"] == "stage2_qwen_adjudication_ledger"
    assert len(document["records"]) == 100

    with pytest.raises(ValueError, match="typed Qwen decisions"):
        qwen_adjudication_ledger_document(
            ({"rationale": "free text"},) * 100,
            candidate=candidate,
            candidate_bundle_sha256="c" * 64,
            benchmark_papers_document=papers,
            benchmark_papers_sha256=sha256(papers_bytes).hexdigest(),
            query_metadata_document=metadata,
            query_metadata_sha256="d" * 64,
        )


def test_source_runner_selects_strata_and_publishes_one_atomic_bundle(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ledger, candidate, papers, _metadata = _derived_source_inputs()
    profile = candidate.profile
    profile.document_batch_size = 32
    profile.reranker_max_in_flight = 2
    profile.reranker_model_id = "bge"
    profile.adjudicator_concurrency = 4
    profile.adjudicator_seed = 42
    profile.adjudicator_max_context_window = 16_384
    profile.adjudicator_max_output_tokens = 256
    profile.evaluation_topic_queries = (
        ("molecular_generation", "en", "molecular generation"),
        ("molecular_generation", "zh", "分子生成"),
        ("protein_design", "en", "protein design"),
        ("protein_design", "zh", "蛋白质设计"),
    )
    profile.evaluation_topic_query_map = {
        (topic, language): query
        for topic, language, query in profile.evaluation_topic_queries
    }

    def rerank(_self, _query, documents):
        return tuple(
            RerankScore(
                document.paper_id,
                0.9 if "relevant" in document.paper_id else 0.5,
            )
            for document in documents
        )

    def adjudicate(_self, request):
        return AdjudicationDecision(
            request.paper_id,
            "relevant",
            0.9,
            ("semantic_match",),
            f"typed rationale {request.paper_id}",
            ("title", "abstract"),
        )

    monkeypatch.setattr(
        "paper_agent.stage2_rationale_workflow.OmlxRerankBackend.rerank", rerank
    )
    monkeypatch.setattr(
        "paper_agent.stage2_rationale_workflow.OmlxChatBackend.adjudicate", adjudicate
    )
    papers_bytes = json.dumps(papers, sort_keys=True).encode()
    artifacts = collect_rationale_source_artifacts(
        candidate,
        candidate_bundle_sha256="c" * 64,
        benchmark_papers_document=papers,
        benchmark_papers_sha256=sha256(papers_bytes).hexdigest(),
        transport=SimpleNamespace(),
    )

    assert len(artifacts.query_metadata["assignments"]) == 100
    assert len(artifacts.query_metadata["scores"]) == 400
    assert len(artifacts.source_ledger["records"]) == 100
    derived = derive_rationale_audit_examples(
        artifacts.source_ledger,
        source_ledger_sha256="e" * 64,
        candidate=candidate,
        candidate_bundle_sha256="c" * 64,
        benchmark_papers_document=papers,
        benchmark_papers_sha256=sha256(papers_bytes).hexdigest(),
        query_metadata=artifacts.query_metadata,
        query_metadata_sha256=artifacts.source_ledger["query_metadata_sha256"],
    )
    assert len(derived["examples"]) == 100
    assert {row["topic"] for row in artifacts.query_metadata["scores"]} == {
        "molecular_generation",
        "protein_design",
    }
    output = tmp_path / "source"
    metadata_path, ledger_path = write_rationale_source_artifacts_no_replace(
        artifacts, output_directory=output
    )
    assert metadata_path.is_file() and ledger_path.is_file()
    metadata_bytes = metadata_path.read_bytes()
    assert "分子生成" in metadata_bytes.decode("utf-8")
    assert json.loads(ledger_path.read_bytes())["query_metadata_sha256"] == sha256(
        metadata_bytes
    ).hexdigest()
    with pytest.raises(FileExistsError):
        write_rationale_source_artifacts_no_replace(
            artifacts, output_directory=output
        )


def test_derivation_replays_assignment_selection_from_every_reranker_score() -> None:
    ledger, candidate, papers, metadata = _derived_source_inputs()
    replacement = dict(metadata["scores"][0])
    replacement["reranker_raw_score"] = 1.0
    replacement["reranker_probability"] = 1.0
    replacement["stratum"] = "relevant"
    metadata["scores"][0] = replacement
    metadata["assignments"] = metadata["assignments"][1:] + [replacement]

    with pytest.raises(ValueError, match="deterministic score selection|at least 25 boundary"):
        derive_rationale_audit_examples(
            ledger,
            source_ledger_sha256="a" * 64,
            candidate=candidate,
            candidate_bundle_sha256="c" * 64,
            benchmark_papers_document=papers,
            benchmark_papers_sha256=metadata["benchmark_papers_sha256"],
            query_metadata=metadata,
            query_metadata_sha256=ledger["query_metadata_sha256"],
        )


def test_source_bundle_reserves_output_and_cleans_up_partial_publication(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, _candidate, _papers, metadata = _derived_source_inputs()
    artifacts = SimpleNamespace(query_metadata=metadata, source_ledger=ledger)
    output = tmp_path / "source"
    real_write = __import__(
        "paper_agent.stage2_rationale_workflow", fromlist=["_write_json_no_replace"]
    )._write_json_no_replace
    calls = 0

    def fail_second(path, document):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated second write failure")
        real_write(path, document)

    monkeypatch.setattr(
        "paper_agent.stage2_rationale_workflow._write_json_no_replace", fail_second
    )
    with pytest.raises(OSError, match="second write failure"):
        write_rationale_source_artifacts_no_replace(
            artifacts, output_directory=output
        )
    assert not output.exists()
    assert not tuple(tmp_path.glob(".source.*"))


def test_source_bundle_is_invisible_until_both_files_are_complete(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, _candidate, _papers, metadata = _derived_source_inputs()
    artifacts = SimpleNamespace(query_metadata=metadata, source_ledger=ledger)
    output = tmp_path / "source"
    real_write = __import__(
        "paper_agent.stage2_rationale_workflow", fromlist=["_write_json_no_replace"]
    )._write_json_no_replace
    writes = 0

    def observe_publish(path, document):
        nonlocal writes
        assert not output.exists()
        real_write(path, document)
        writes += 1
        assert not output.exists()

    monkeypatch.setattr(
        "paper_agent.stage2_rationale_workflow._write_json_no_replace", observe_publish
    )
    metadata_path, ledger_path = write_rationale_source_artifacts_no_replace(
        artifacts, output_directory=output
    )
    assert writes == 2
    assert metadata_path.is_file() and ledger_path.is_file()


def test_source_runner_retries_only_failed_qwen_items_after_first_wave(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ledger, candidate, papers, _metadata = _derived_source_inputs()
    profile = candidate.profile
    profile.document_batch_size = 32
    profile.reranker_max_in_flight = 2
    profile.reranker_model_id = "bge"
    profile.adjudicator_concurrency = 4
    profile.adjudicator_seed = 42
    profile.adjudicator_max_context_window = 16_384
    profile.adjudicator_max_output_tokens = 256
    failed_pair = None
    calls: dict[str, int] = {}
    first_wave_calls = 0
    lock = Lock()

    def rerank(_self, _query, documents):
        return tuple(
            RerankScore(
                document.paper_id,
                0.9 if "relevant" in document.paper_id else 0.5,
            )
            for document in documents
        )

    def adjudicate(_self, request):
        nonlocal failed_pair, first_wave_calls
        with lock:
            calls[request.paper_id] = calls.get(request.paper_id, 0) + 1
            attempt = calls[request.paper_id]
            if attempt == 1:
                first_wave_calls += 1
            if failed_pair is None:
                failed_pair = request.paper_id
            if request.paper_id == failed_pair and attempt == 2:
                assert first_wave_calls == 100
        if request.paper_id == failed_pair and attempt == 1:
            raise StructuredOutputError("retry once")
        return AdjudicationDecision(
            request.paper_id, "relevant", 0.9, ("semantic_match",),
            f"typed rationale {request.paper_id}", ("title", "abstract"),
        )

    monkeypatch.setattr(
        "paper_agent.stage2_rationale_workflow.OmlxRerankBackend.rerank", rerank
    )
    monkeypatch.setattr(
        "paper_agent.stage2_rationale_workflow.OmlxChatBackend.adjudicate", adjudicate
    )
    papers_bytes = json.dumps(papers, sort_keys=True).encode()
    artifacts = collect_rationale_source_artifacts(
        candidate,
        candidate_bundle_sha256="c" * 64,
        benchmark_papers_document=papers,
        benchmark_papers_sha256=sha256(papers_bytes).hexdigest(),
        transport=SimpleNamespace(),
    )

    assert len(artifacts.source_ledger["records"]) == 100
    assert sorted(calls.values()).count(2) == 1
    assert sum(calls.values()) == 101


def test_source_runner_terminal_qwen_failure_returns_no_artifacts(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ledger, candidate, papers, _metadata = _derived_source_inputs()
    profile = candidate.profile
    profile.document_batch_size = 32
    profile.reranker_max_in_flight = 2
    profile.reranker_model_id = "bge"
    profile.adjudicator_concurrency = 4
    profile.adjudicator_seed = 42
    profile.adjudicator_max_context_window = 16_384
    profile.adjudicator_max_output_tokens = 256

    monkeypatch.setattr(
        "paper_agent.stage2_rationale_workflow.OmlxRerankBackend.rerank",
        lambda _self, _query, documents: tuple(
            RerankScore(
                document.paper_id,
                0.9 if "relevant" in document.paper_id else 0.5,
            )
            for document in documents
        ),
    )
    monkeypatch.setattr(
        "paper_agent.stage2_rationale_workflow.OmlxChatBackend.adjudicate",
        lambda _self, _request: (_ for _ in ()).throw(
            StructuredOutputError("terminal failure")
        ),
    )
    output = tmp_path / "source"
    with pytest.raises(StructuredOutputError, match="terminal failure"):
        artifacts = collect_rationale_source_artifacts(
            candidate,
            candidate_bundle_sha256="c" * 64,
            benchmark_papers_document=papers,
            benchmark_papers_sha256="d" * 64,
            transport=SimpleNamespace(),
        )
        write_rationale_source_artifacts_no_replace(
            artifacts, output_directory=output
        )
    assert not output.exists()
