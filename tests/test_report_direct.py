from __future__ import annotations

from hashlib import sha256
import json
from threading import Thread
from time import sleep
from types import MappingProxyType

import paper_agent.report_direct as report_direct
import pytest
from paper_agent.canonical import content_hash
from paper_agent.codex_exec import CodexExecResult, InvocationMetadata
from paper_agent.report_artifacts import (
    LOCAL_REFERENCES_NOTE,
    ReportArtifactError,
    ReportArtifactStore,
    ReportVerificationError,
)
from paper_agent.report_config import ReportResources, ReportRuntimeConfig
from paper_agent.report_direct import (
    one_shot_config_hash,
    one_shot_validation_config_hash,
)
from paper_agent.report_execution_service import ReportExecutionService
from paper_agent.report_plan import (
    CorpusPaper,
    ReportPlanBundle,
    approve_report_plan,
    build_corpus_snapshot,
    build_search_audit_pack,
    compile_report_plan,
)
from paper_agent.reporting import stable_claim_id

from test_report_reduce import _claim, _draft, _fixture, _sol_grant


def _one_shot_bundle(
    fixture, *, max_input_tokens: int = 10_000_000
) -> ReportPlanBundle:
    resources = ReportResources.defaults()
    draft = _draft(max_input_tokens)
    draft["execution_strategy"] = "one_shot"
    draft["stage4b_config_hash"] = one_shot_config_hash(
        fixture.coordinator.gate.policy.hash,
        resources=resources,
    )
    draft["stage4b_audit_config_hash"] = one_shot_validation_config_hash()
    draft["budget"] = {
        "max_sol_calls": 1,
        "max_input_tokens": max_input_tokens,
        "max_retries": 0,
        "audit_calls": 0,
        "repair_calls": 0,
    }
    compiled = compile_report_plan(
        draft,
        corpus_snapshot=fixture.corpus,
        search_audit_pack=fixture.audit,
        created_at="2026-08-10T03:00:00Z",
        resources=resources,
    )
    approved = approve_report_plan(
        compiled,
        compiled["plan_hash"],
        approved_by="owner",
        approved_at="2026-08-10T03:01:00Z",
    )
    return ReportPlanBundle(approved, fixture.corpus, fixture.audit)


def _freeze_missing_authors_with_official_db_provenance(
    fixture, *, add_provenance: bool = True
) -> None:
    papers = tuple(
        CorpusPaper(
            **{
                **paper,
                "authors": (),
                "lineage_hashes": tuple(paper["lineage_hashes"]),
            }
        )
        for paper in fixture.corpus["papers"]
    )
    raw_audit = fixture.audit["source_round_audit"]
    fixture.corpus = build_corpus_snapshot(
        papers,
        query_plan_hash=fixture.corpus["query_plan_hash"],
        search_audit=raw_audit,
        created_at="2026-08-10T00:01:00Z",
    )
    fixture.audit = build_search_audit_pack(
        raw_audit,
        fixture.corpus,
        screening_flow={
            key: int(fixture.audit["flow"][key])
            for key in (
                "raw_discovered",
                "unique_after_dedup",
                "stage2_screened",
                "included",
            )
        },
        exclusion_reasons=fixture.audit["flow"]["excluded_by_reason"],
        created_at="2026-08-10T00:02:00Z",
    )
    if not add_provenance:
        return
    for paper in fixture.corpus["papers"]:
        paper_id = str(paper["paper_id"])
        source_id = f"official-metadata-{paper_id}"
        authors = ["Ada Lovelace"]
        fixture.database.connection.execute(
            """INSERT INTO paper_sources(
                   source_id, paper_id, provider, external_id, landing_url,
                   publication_version, host_type, access_basis, raw_metadata_json
               ) VALUES (?, ?, 'fixture-official', ?, ?, 'published', 'official',
                         'public_read_only', ?)""",
            (
                source_id,
                paper_id,
                f"official-{paper_id}",
                f"https://example.test/{paper_id}",
                json.dumps(
                    {
                        "kind": "canonical_metadata_fill",
                        "locator": "official landing page",
                    },
                    sort_keys=True,
                ),
            ),
        )
        fixture.database.connection.execute(
            """INSERT INTO paper_field_provenance(
                   provenance_id, paper_id, source_id, field_name, field_value_json
               ) VALUES (?, ?, ?, 'authors', ?)""",
            (
                f"provenance-{source_id}-authors",
                paper_id,
                source_id,
                json.dumps(authors, separators=(",", ":")),
            ),
        )
    fixture.database.connection.commit()


class OneShotSol:
    def __init__(self, fixture, mutate_output=None) -> None:
        self.fixture = fixture
        self.mutate_output = mutate_output
        self.calls = []

    def invoke(self, request):
        self.calls.append(request)
        payload = json.loads(request.prompt)
        records = {item.paper_id: item for item in self.fixture.records}
        claims = []
        blocks = []
        for index, section in enumerate(payload["report_plan"]["sections"]):
            paper_id = "p1" if index % 2 == 0 else "p2"
            claim = _claim(payload["report_run_id"], section["id"], records[paper_id])
            claim_ref = f"claim-{index + 1}"
            claims.append({
                "claim_ref": claim_ref,
                "subject_id": claim["claim_key"]["subject_id"],
                "predicate_id": claim["claim_key"]["predicate_id"],
                "object_or_scope_id": claim["claim_key"]["object_or_scope_id"],
                "qualifier_context": section["id"],
                "research_question_id": claim["research_question_id"],
                "report_section": claim["report_section"],
                "claim_text": claim["claim_text"],
                "claim_type": claim["claim_type"],
                "supporting_evidence": claim["supporting_evidence"],
                "contradicting_evidence": claim["contradicting_evidence"],
                "evidence_level": claim["evidence_level"],
                "confidence": claim["confidence"],
                "known_limitations": claim["known_limitations"],
                "status": claim["status"],
            })
            text = f"{claim['claim_text']} [@{paper_id}]"
            if section["id"] == "report_limitations":
                text += " " + " ".join(payload["required_disclosures"])
            blocks.append({
                "block_id": f"block-{index + 1}",
                "block_kind": "prose",
                "section_id": section["id"],
                "text": text,
                "claim_refs": [claim_ref],
                "citation_paper_ids": [paper_id],
            })
        output = {
            "claims": claims,
            "blocks": blocks,
            "unresolved_conflicts": [],
            "claim_relations": [],
        }
        if self.mutate_output is not None:
            self.mutate_output(output, payload)
        resources = ReportResources.defaults()
        rendered = (
            resources.prompt("one_shot_report").rstrip()
            + "\n\nThe authorized input follows as JSON data:\n"
            + json.dumps(
                {"authorized_input": request.prompt},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        )
        metadata = InvocationMetadata(
            invocation_id="one-shot-invocation-1",
            profile="stage4b_oneshot_sol",
            model="gpt-5.6-sol",
            reasoning_effort="high",
            schema_name=request.schema_name,
            schema_hash=content_hash(resources.schema("one_shot_report")),
            input_hash=request.input_hash,
            prompt_name=request.prompt_name,
            prompt_hash=sha256(
                resources.prompt_paths["one_shot_report"].read_bytes()
            ).hexdigest(),
            rendered_prompt_hash=sha256(rendered.encode()).hexdigest(),
            call_kind="one_shot_report",
            attempts=1,
            actual_model="gpt-5.6-sol",
            actual_profile="stage4b_oneshot_sol",
            schema_path=request.schema_path,
            prompt_path=request.prompt_path,
            output_hash=content_hash(output),
        )
        # Match the real CodexExec boundary, which freezes top-level output.
        return CodexExecResult(MappingProxyType(output), metadata)


def _one_shot_grants(fixture, prefix: str) -> dict[str, str]:
    return {
        artifact.artifact_hash: _sol_grant(
            fixture,
            artifact,
            f"{prefix}-{index}",
            expires_at="2026-09-11T00:00:00Z",
        )
        for index, artifact in enumerate(fixture.artifacts, start=1)
    }


def _one_shot_service(fixture, release, fake) -> ReportExecutionService:
    return ReportExecutionService(
        fixture.database,
        fixture.store,
        fixture.coordinator.gate,
        ReportArtifactStore(release),
        direct_invoker_factory=lambda: fake,
        runtime_config=ReportRuntimeConfig(
            True,
            ReportResources.defaults(),
            profile="stage4b_oneshot_sol",
            execution_strategy="one_shot",
        ),
    )


def _section_claim(output: dict, section_id: str) -> tuple[dict, dict]:
    block = next(
        item for item in output["blocks"] if item["section_id"] == section_id
    )
    claim_ref = str(block["claim_refs"][0])
    claim = next(
        item for item in output["claims"] if item["claim_ref"] == claim_ref
    )
    return claim, block


def _make_procedural_reference_note(output: dict, *, mixed: bool = False) -> None:
    claim, block = _section_claim(output, "references_and_appendices")
    claim.update({
        "subject_id": "report-renderer",
        "predicate_id": "should_generate_canonical_references",
        "object_or_scope_id": "local-reference-rendering",
        "qualifier_context": "do not generate bibliography entries from memory",
        "claim_text": "The coordinator should generate canonical references locally.",
        "claim_type": "recommendation",
        "supporting_evidence": [],
        "contradicting_evidence": [],
        "evidence_level": "corpus_stat",
        "confidence": "high",
        "known_limitations": ["This is a procedural note, not a paper claim."],
        "status": "insufficient",
    })
    block["text"] = "这段模型生成的程序文本不应进入最终报告。"
    block["citation_paper_ids"] = []
    if mixed:
        home_claim, _ = _section_claim(output, "executive_summary")
        block["claim_refs"].append(home_claim["claim_ref"])


def _make_semantic_procedural_reference_note(
    output: dict,
    *,
    subject_id: str = "report-bibliography",
    predicate_id: str = "is-generated-locally",
    object_or_scope_id: str = "canonical-reference-rendering",
) -> None:
    claim, block = _section_claim(output, "references_and_appendices")
    claim.update({
        "subject_id": subject_id,
        "predicate_id": predicate_id,
        "object_or_scope_id": object_or_scope_id,
        "qualifier_context": "canonical bibliography is generated locally",
        "claim_text": "规范参考文献由本地协调器确定性生成，不属于论文结论。",
        "claim_type": "recommendation",
        "supporting_evidence": [],
        "contradicting_evidence": [],
        "evidence_level": "metadata_only",
        "confidence": "high",
        "known_limitations": ["本次草稿不生成规范书目条目。"],
        "status": "supported",
    })
    block["text"] = "规范参考文献由本地协调器生成；这不是论文结论。"
    block["citation_paper_ids"] = []


def _make_unsupported_substantive_reference_recommendation(output: dict) -> None:
    claim, block = _section_claim(output, "references_and_appendices")
    claim.update({
        "subject_id": "future-molecular-model",
        "predicate_id": "should_improve_accuracy",
        "object_or_scope_id": "external-benchmark",
        "qualifier_context": "future deployment recommendation",
        "claim_text": "未来模型应在外部基准上将准确率提高到99%。",
        "claim_type": "recommendation",
        "supporting_evidence": [],
        "contradicting_evidence": [],
        "evidence_level": "metadata_only",
        "confidence": "low",
        "known_limitations": ["No direct evidence was provided."],
        "status": "insufficient",
    })
    block["text"] = "未来模型应在外部基准上将准确率提高到99%。"
    block["citation_paper_ids"] = []


@pytest.mark.parametrize(
    ("legacy_component", "expected_error"),
    (
        ("implementation", "Stage 4b configuration has drifted"),
        ("validation", "deterministic validation configuration has drifted"),
    ),
)
def test_legacy_one_shot_approved_plan_hashes_fail_closed_before_dispatch(
    tmp_path,
    monkeypatch,
    legacy_component,
    expected_error,
) -> None:
    fixture = _fixture(tmp_path, max_input_tokens=50_000_000)
    current_implementation = report_direct.IMPLEMENTATION_VERSION
    current_validation = report_direct.DETERMINISTIC_VALIDATION_VERSION
    if legacy_component == "implementation":
        monkeypatch.setattr(
            report_direct, "IMPLEMENTATION_VERSION", "stage4b-one-shot-v1"
        )
    else:
        monkeypatch.setattr(
            report_direct,
            "DETERMINISTIC_VALIDATION_VERSION",
            "deterministic-report-v1",
        )
    bundle = _one_shot_bundle(fixture)
    monkeypatch.setattr(
        report_direct, "IMPLEMENTATION_VERSION", current_implementation
    )
    monkeypatch.setattr(
        report_direct,
        "DETERMINISTIC_VALIDATION_VERSION",
        current_validation,
    )
    fake = OneShotSol(fixture)
    try:
        result = _one_shot_service(
            fixture, tmp_path / "release", fake
        ).run(
            f"legacy-{legacy_component}-report",
            f"legacy-{legacy_component}-pipeline",
            bundle,
            processing_grants={},
        )

        assert result.status == "incomplete"
        assert expected_error in str(result.error)
        assert fake.calls == []
        assert fixture.database.connection.execute(
            "SELECT COUNT(*) FROM report_one_shot_runs"
        ).fetchone()[0] == 0
        assert fixture.database.connection.execute(
            "SELECT COUNT(*) FROM report_sol_invocations"
        ).fetchone()[0] == 0
    finally:
        fixture.database.close()


def test_one_shot_report_uses_one_sol_call_and_resumes_without_another(tmp_path) -> None:
    fixture = _fixture(tmp_path, max_input_tokens=50_000_000)
    bundle = _one_shot_bundle(fixture)
    fake = OneShotSol(fixture)
    grants = {
        artifact.artifact_hash: _sol_grant(
            fixture,
            artifact,
            f"one-shot-grant-{index}",
            expires_at="2026-09-11T00:00:00Z",
        )
        for index, artifact in enumerate(fixture.artifacts, start=1)
    }
    try:
        service = ReportExecutionService(
            fixture.database,
            fixture.store,
            fixture.coordinator.gate,
            ReportArtifactStore(tmp_path / "release"),
            direct_invoker_factory=lambda: fake,
            runtime_config=ReportRuntimeConfig(
                True,
                ReportResources.defaults(),
                profile="stage4b_oneshot_sol",
                execution_strategy="one_shot",
            ),
        )
        first = service.run(
            "report-one-shot",
            "pipeline-one-shot",
            bundle,
            processing_grants=grants,
        )
        second = service.run(
            "report-one-shot",
            "pipeline-one-shot",
            bundle,
            processing_grants=grants,
        )

        assert first.status == second.status == "complete"
        assert len(fake.calls) == 1
        assert fake.calls[0].call_kind == "one_shot_report"
        prompt = json.loads(fake.calls[0].prompt)
        assert len(prompt["analyses"]) == len(fixture.artifacts)
        assert {
            item["analysis_artifact_hash"] for item in prompt["analyses"]
        } == {item.artifact_hash for item in fixture.artifacts}
        row = fixture.database.connection.execute(
            """SELECT status, dispatch_count, budget_calls_reserved
               FROM report_one_shot_runs WHERE report_run_id = ?""",
            ("report-one-shot",),
        ).fetchone()
        assert tuple(row) == ("complete", 1, 1)
        assert fixture.database.connection.execute(
            "SELECT COUNT(*) FROM report_sol_invocations WHERE report_run_id = ?",
            ("report-one-shot",),
        ).fetchone()[0] == 1
        assert fixture.database.connection.execute(
            "SELECT COUNT(*) FROM report_reduce_nodes WHERE report_run_id = ?",
            ("report-one-shot",),
        ).fetchone()[0] == 0
        assert fixture.database.connection.execute(
            "SELECT COUNT(*) FROM report_audit_steps WHERE report_run_id = ?",
            ("report-one-shot",),
        ).fetchone()[0] == 0
        assert fixture.database.connection.execute(
            "SELECT COUNT(*) FROM report_audit_shard_steps WHERE report_run_id = ?",
            ("report-one-shot",),
        ).fetchone()[0] == 0
        report = tmp_path / "release" / "reports" / "report-one-shot"
        assert (report / "REPORT.md").is_file()
        assert json.loads((report / "AUDIT.json").read_text())["audit_pass"] == "deterministic"
    finally:
        fixture.database.close()


def test_one_shot_oversized_input_stops_before_dispatch(tmp_path) -> None:
    fixture = _fixture(tmp_path, max_input_tokens=50_000_000)
    bundle = _one_shot_bundle(fixture, max_input_tokens=1)
    fake = OneShotSol(fixture)
    try:
        result = _one_shot_service(
            fixture, tmp_path / "release", fake
        ).run(
            "report-one-shot-oversized",
            "pipeline-one-shot-oversized",
            bundle,
            processing_grants={},
        )

        assert result.status == "incomplete"
        assert result.alarm_codes == ("report.codex_budget_exhausted",)
        assert result.error == {
            "type": "DirectReportBudgetError",
            "message": "one-shot Sol prompt exceeds the approved input budget",
            "event_code": "report.codex_budget_exhausted",
        }
        assert fake.calls == []
        assert fixture.database.connection.execute(
            "SELECT COUNT(*) FROM report_one_shot_runs"
        ).fetchone()[0] == 0
        assert fixture.database.connection.execute(
            "SELECT COUNT(*) FROM report_sol_invocations"
        ).fetchone()[0] == 0
    finally:
        fixture.database.close()


def test_bibliography_fill_only_overlay_keeps_frozen_one_shot_input(tmp_path) -> None:
    fixture = _fixture(tmp_path, max_input_tokens=50_000_000)
    _freeze_missing_authors_with_official_db_provenance(fixture)
    bundle = _one_shot_bundle(fixture)
    fake = OneShotSol(fixture)
    grants = _one_shot_grants(fixture, "bibliography-overlay-grant")
    report_run_id = "report-one-shot-bibliography-overlay"
    try:
        service = _one_shot_service(fixture, tmp_path / "release", fake)
        result = service.run(
            report_run_id,
            "pipeline-one-shot-bibliography-overlay",
            bundle,
            processing_grants=grants,
        )

        assert result.status == "complete", result.error
        assert len(fake.calls) == 1
        prompt = json.loads(fake.calls[0].prompt)
        assert prompt["corpus_summary"]["snapshot_hash"] == fixture.corpus["snapshot_hash"]
        assert all(
            "authors" not in paper
            for paper in prompt["corpus_summary"]["papers"]
        )
        bibliography = json.loads(
            (
                tmp_path
                / "release"
                / "reports"
                / report_run_id
                / "BIBLIOGRAPHY.json"
            ).read_text(encoding="utf-8")
        )
        for entry in bibliography.values():
            assert entry["authors"] == ["Ada Lovelace"]
            overlay = entry["canonical_metadata_overlay"]
            assert overlay["mode"] == "fill_only"
            assert overlay["frozen_snapshot_hash"] == fixture.corpus["snapshot_hash"]
            assert set(overlay["fields"]) == {"authors"}
            assert overlay["overlay_hash"] == content_hash(
                {key: value for key, value in overlay.items() if key != "overlay_hash"}
            )
    finally:
        fixture.database.close()


def test_missing_bibliography_metadata_stops_before_one_shot_dispatch(tmp_path) -> None:
    fixture = _fixture(tmp_path, max_input_tokens=50_000_000)
    _freeze_missing_authors_with_official_db_provenance(
        fixture, add_provenance=False
    )
    bundle = _one_shot_bundle(fixture)
    fake = OneShotSol(fixture)
    try:
        service = _one_shot_service(fixture, tmp_path / "release", fake)
        result = service.run(
            "report-one-shot-missing-bibliography",
            "pipeline-one-shot-missing-bibliography",
            bundle,
            processing_grants=_one_shot_grants(
                fixture, "missing-bibliography-grant"
            ),
        )

        assert result.status == "incomplete"
        assert "lacks official provenance" in str(result.error)
        assert fake.calls == []
        assert fixture.database.connection.execute(
            """SELECT COUNT(*) FROM report_one_shot_runs
               WHERE report_run_id = 'report-one-shot-missing-bibliography'"""
        ).fetchone()[0] == 0
    finally:
        fixture.database.close()


def test_post_validation_retry_publishes_persisted_output_without_redispatch(
    tmp_path, monkeypatch
) -> None:
    fixture = _fixture(tmp_path, max_input_tokens=50_000_000)
    bundle = _one_shot_bundle(fixture)
    fake = OneShotSol(fixture)
    grants = _one_shot_grants(fixture, "post-validation-grant")
    report_run_id = "report-one-shot-post-validation"
    pipeline_run_id = "pipeline-one-shot-post-validation"
    real_verify = report_direct._deterministic_verify
    try:
        service = _one_shot_service(fixture, tmp_path / "release", fake)
        monkeypatch.setattr(
            report_direct,
            "_deterministic_verify",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                ReportVerificationError("forced local post-validation failure")
            ),
        )
        first = service.run(
            report_run_id,
            pipeline_run_id,
            bundle,
            processing_grants=grants,
        )
        persisted = fixture.database.connection.execute(
            """SELECT status, dispatch_count, invocation_id, output_hash
               FROM report_one_shot_runs WHERE report_run_id = ?""",
            (report_run_id,),
        ).fetchone()

        monkeypatch.setattr(report_direct, "_deterministic_verify", real_verify)
        second = service.run(
            report_run_id,
            pipeline_run_id,
            bundle,
            processing_grants=grants,
        )
        replayed = fixture.database.connection.execute(
            """SELECT status, dispatch_count, invocation_id, output_hash
               FROM report_one_shot_runs WHERE report_run_id = ?""",
            (report_run_id,),
        ).fetchone()

        assert first.status == "failed"
        assert second.status == "complete", second.error
        assert len(fake.calls) == 1
        assert tuple(persisted) == tuple(replayed)
        assert tuple(replayed[:2]) == ("complete", 1)
        assert (tmp_path / "release" / "reports" / report_run_id / "REPORT.md").is_file()
    finally:
        fixture.database.close()


def test_cross_section_claim_is_derived_per_section_and_resume_does_not_redispatch(
    tmp_path,
) -> None:
    fixture = _fixture(tmp_path, max_input_tokens=50_000_000)
    bundle = _one_shot_bundle(fixture)

    def reuse_home_claim(output, _payload) -> None:
        claim, _ = _section_claim(output, "executive_summary")
        claim["report_section"] = "field_taxonomy"
        _, target = _section_claim(output, "scope_and_methods")
        target["claim_refs"].append(claim["claim_ref"])
        paper_id = claim["supporting_evidence"][0]["paper_id"]
        if paper_id not in target["citation_paper_ids"]:
            target["citation_paper_ids"].append(paper_id)
            target["text"] += f" [@{paper_id}]"

    fake = OneShotSol(fixture, reuse_home_claim)
    grants = _one_shot_grants(fixture, "cross-section-grant")
    report_run_id = "report-one-shot-cross-section"
    try:
        service = _one_shot_service(fixture, tmp_path / "release", fake)
        first = service.run(
            report_run_id,
            "pipeline-one-shot-cross-section",
            bundle,
            processing_grants=grants,
        )
        second = service.run(
            report_run_id,
            "pipeline-one-shot-cross-section",
            bundle,
            processing_grants=grants,
        )

        assert first.status == second.status == "complete"
        assert len(fake.calls) == 1
        report = tmp_path / "release" / "reports" / report_run_id
        claims = [
            json.loads(line)
            for line in (report / "CLAIMS_EVIDENCE.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        derived = [
            claim
            for claim in claims
            if claim["claim_key"]["subject_id"] == "p1"
            and claim["claim_key"]["object_or_scope_id"] == "executive_summary"
        ]
        assert {claim["report_section"] for claim in derived} == {
            "executive_summary",
            "scope_and_methods",
        }
        assert len({claim["claim_id"] for claim in derived}) == 2
        assert all(
            claim["claim_id"]
            == stable_claim_id(claim["claim_key"], report_run_id=report_run_id)
            for claim in derived
        )

        document = json.loads(
            (report / "REPORT_DOCUMENT.json").read_text(encoding="utf-8")
        )
        claims_by_id = {claim["claim_id"]: claim for claim in claims}
        for block in document["blocks"]:
            assert all(
                claims_by_id[claim_id]["report_section"] == block["section_id"]
                for claim_id in block["claim_ids"]
            )
        derived_by_section = {
            claim["report_section"]: claim["claim_id"] for claim in derived
        }
        blocks_by_section = {
            block["section_id"]: block for block in document["blocks"]
        }
        assert (
            derived_by_section["executive_summary"]
            in blocks_by_section["executive_summary"]["claim_ids"]
        )
        assert (
            derived_by_section["scope_and_methods"]
            in blocks_by_section["scope_and_methods"]["claim_ids"]
        )
        assert fixture.database.connection.execute(
            "SELECT dispatch_count FROM report_one_shot_runs WHERE report_run_id = ?",
            (report_run_id,),
        ).fetchone()[0] == 1
    finally:
        fixture.database.close()


def test_exclusive_procedural_reference_note_becomes_exact_local_block(
    tmp_path,
) -> None:
    fixture = _fixture(tmp_path, max_input_tokens=50_000_000)
    bundle = _one_shot_bundle(fixture)
    fake = OneShotSol(
        fixture,
        lambda output, _payload: _make_procedural_reference_note(output),
    )
    grants = _one_shot_grants(fixture, "procedural-reference-grant")
    report_run_id = "report-one-shot-procedural-reference"
    try:
        service = _one_shot_service(fixture, tmp_path / "release", fake)
        first = service.run(
            report_run_id,
            "pipeline-one-shot-procedural-reference",
            bundle,
            processing_grants=grants,
        )
        second = service.run(
            report_run_id,
            "pipeline-one-shot-procedural-reference",
            bundle,
            processing_grants=grants,
        )

        assert first.status == second.status == "complete"
        assert len(fake.calls) == 1
        report = tmp_path / "release" / "reports" / report_run_id
        document = json.loads(
            (report / "REPORT_DOCUMENT.json").read_text(encoding="utf-8")
        )
        reference_block = next(
            block
            for block in document["blocks"]
            if block["section_id"] == "references_and_appendices"
        )
        assert reference_block == {
            "block_id": reference_block["block_id"],
            "block_kind": "caption",
            "section_id": "references_and_appendices",
            "text": LOCAL_REFERENCES_NOTE,
            "claim_ids": [],
            "citation_paper_ids": [],
        }
        report_markdown = (report / "REPORT.md").read_text(encoding="utf-8")
        assert LOCAL_REFERENCES_NOTE in report_markdown
        assert "这段模型生成的程序文本不应进入最终报告。" not in report_markdown
        claims_text = (report / "CLAIMS_EVIDENCE.jsonl").read_text(encoding="utf-8")
        assert "should_generate_canonical_references" not in claims_text
    finally:
        fixture.database.close()


@pytest.mark.parametrize(
    ("subject_id", "predicate_id", "object_or_scope_id"),
    (
        (
            "report-bibliography",
            "is-generated-locally",
            "canonical-reference-rendering",
        ),
        (
            "local-report-coordinator",
            "should_be_generated_locally",
            "canonical-references-and-audit-appendices",
        ),
    ),
)
def test_semantic_bibliography_note_becomes_exact_local_block_without_redispatch(
    tmp_path, subject_id, predicate_id, object_or_scope_id
) -> None:
    fixture = _fixture(tmp_path, max_input_tokens=50_000_000)
    bundle = _one_shot_bundle(fixture)
    fake = OneShotSol(
        fixture,
        lambda output, _payload: _make_semantic_procedural_reference_note(
            output,
            subject_id=subject_id,
            predicate_id=predicate_id,
            object_or_scope_id=object_or_scope_id,
        ),
    )
    grants = _one_shot_grants(fixture, "semantic-reference-grant")
    report_run_id = "report-one-shot-semantic-reference"
    try:
        service = _one_shot_service(fixture, tmp_path / "release", fake)
        first = service.run(
            report_run_id,
            "pipeline-one-shot-semantic-reference",
            bundle,
            processing_grants=grants,
        )
        second = service.run(
            report_run_id,
            "pipeline-one-shot-semantic-reference",
            bundle,
            processing_grants=grants,
        )

        assert first.status == second.status == "complete"
        assert len(fake.calls) == 1
        report = tmp_path / "release" / "reports" / report_run_id
        document = json.loads(
            (report / "REPORT_DOCUMENT.json").read_text(encoding="utf-8")
        )
        reference_block = next(
            block
            for block in document["blocks"]
            if block["section_id"] == "references_and_appendices"
        )
        assert reference_block["text"] == LOCAL_REFERENCES_NOTE
        assert reference_block["claim_ids"] == []
        assert reference_block["citation_paper_ids"] == []
        assert fixture.database.connection.execute(
            """SELECT dispatch_count FROM report_one_shot_runs
               WHERE report_run_id = ?""",
            (report_run_id,),
        ).fetchone()[0] == 1
    finally:
        fixture.database.close()


def test_substantive_evidence_free_reference_recommendation_is_not_normalized_away(
    tmp_path,
) -> None:
    fixture = _fixture(tmp_path, max_input_tokens=50_000_000)
    bundle = _one_shot_bundle(fixture)
    fake = OneShotSol(
        fixture,
        lambda output, _payload: _make_unsupported_substantive_reference_recommendation(
            output
        ),
    )
    report_run_id = "report-one-shot-substantive-reference-recommendation"
    try:
        result = _one_shot_service(fixture, tmp_path / "release", fake).run(
            report_run_id,
            "pipeline-one-shot-substantive-reference-recommendation",
            bundle,
            processing_grants=_one_shot_grants(
                fixture, "substantive-reference-recommendation-grant"
            ),
        )

        assert result.status == "failed"
        assert len(fake.calls) == 1
        assert not (
            tmp_path / "release" / "reports" / report_run_id / "REPORT.md"
        ).exists()
        assert fixture.database.connection.execute(
            """SELECT dispatch_count FROM report_one_shot_runs
               WHERE report_run_id = ?""",
            (report_run_id,),
        ).fetchone()[0] == 1
    finally:
        fixture.database.close()


def test_evidence_free_claim_outside_references_still_fails_locally(tmp_path) -> None:
    fixture = _fixture(tmp_path, max_input_tokens=50_000_000)
    bundle = _one_shot_bundle(fixture)

    def remove_evidence(output, _payload) -> None:
        claim, block = _section_claim(output, "executive_summary")
        claim.update({
            "claim_type": "recommendation",
            "supporting_evidence": [],
            "contradicting_evidence": [],
            "evidence_level": "corpus_stat",
            "status": "insufficient",
        })
        block["text"] = "该章节中的无证据主张必须被拒绝。"
        block["citation_paper_ids"] = []

    fake = OneShotSol(fixture, remove_evidence)
    grants = _one_shot_grants(fixture, "unsupported-claim-grant")
    report_run_id = "report-one-shot-unsupported-claim"
    try:
        service = _one_shot_service(fixture, tmp_path / "release", fake)
        first = service.run(
            report_run_id,
            "pipeline-one-shot-unsupported-claim",
            bundle,
            processing_grants=grants,
        )
        second = service.run(
            report_run_id,
            "pipeline-one-shot-unsupported-claim",
            bundle,
            processing_grants=grants,
        )

        assert first.status == second.status == "failed"
        assert "every claim requires evidence" in str(first.direct.error)
        assert len(fake.calls) == 1
        assert not (
            tmp_path / "release" / "reports" / report_run_id
        ).exists()
    finally:
        fixture.database.close()


def test_procedural_reference_note_mixed_with_evidence_claim_still_fails(
    tmp_path,
) -> None:
    fixture = _fixture(tmp_path, max_input_tokens=50_000_000)
    bundle = _one_shot_bundle(fixture)
    fake = OneShotSol(
        fixture,
        lambda output, _payload: _make_procedural_reference_note(
            output, mixed=True
        ),
    )
    grants = _one_shot_grants(fixture, "mixed-procedural-reference-grant")
    report_run_id = "report-one-shot-mixed-procedural-reference"
    try:
        service = _one_shot_service(fixture, tmp_path / "release", fake)
        first = service.run(
            report_run_id,
            "pipeline-one-shot-mixed-procedural-reference",
            bundle,
            processing_grants=grants,
        )
        second = service.run(
            report_run_id,
            "pipeline-one-shot-mixed-procedural-reference",
            bundle,
            processing_grants=grants,
        )

        assert first.status == second.status == "failed"
        assert "must exclusively bind an uncited references block" in str(
            first.direct.error
        )
        assert len(fake.calls) == 1
        assert not (
            tmp_path / "release" / "reports" / report_run_id
        ).exists()
    finally:
        fixture.database.close()


def test_one_shot_missing_grant_makes_zero_calls(tmp_path) -> None:
    fixture = _fixture(tmp_path, max_input_tokens=50_000_000)
    bundle = _one_shot_bundle(fixture)
    fake = OneShotSol(fixture)
    try:
        service = ReportExecutionService(
            fixture.database,
            fixture.store,
            fixture.coordinator.gate,
            ReportArtifactStore(tmp_path / "release"),
            direct_invoker_factory=lambda: fake,
            runtime_config=ReportRuntimeConfig(
                True,
                ReportResources.defaults(),
                profile="stage4b_oneshot_sol",
                execution_strategy="one_shot",
            ),
        )
        result = service.run(
            "report-one-shot-manual",
            "pipeline-one-shot-manual",
            bundle,
            processing_grants={},
        )

        assert result.status == "manual_required"
        assert fake.calls == []
        assert fixture.database.connection.execute(
            "SELECT dispatch_count FROM report_one_shot_runs WHERE report_run_id = ?",
            ("report-one-shot-manual",),
        ).fetchone()[0] == 0
    finally:
        fixture.database.close()


def test_concurrent_resume_observes_the_sole_dispatch_without_failing_it(tmp_path) -> None:
    fixture = _fixture(tmp_path, max_input_tokens=50_000_000)
    bundle = _one_shot_bundle(fixture)
    grants = {
        artifact.artifact_hash: _sol_grant(
            fixture,
            artifact,
            f"one-shot-concurrent-grant-{index}",
            expires_at="2026-09-11T00:00:00Z",
        )
        for index, artifact in enumerate(fixture.artifacts, start=1)
    }
    nested_results = []
    service = None

    class ReentrantSol(OneShotSol):
        def invoke(self, request):
            assert service is not None
            nested_results.append(service.run(
                "report-one-shot-concurrent",
                "pipeline-one-shot-concurrent",
                bundle,
                processing_grants=grants,
            ))
            return super().invoke(request)

    fake = ReentrantSol(fixture)
    try:
        service = ReportExecutionService(
            fixture.database,
            fixture.store,
            fixture.coordinator.gate,
            ReportArtifactStore(tmp_path / "release"),
            direct_invoker_factory=lambda: fake,
            runtime_config=ReportRuntimeConfig(
                True,
                ReportResources.defaults(),
                profile="stage4b_oneshot_sol",
                execution_strategy="one_shot",
            ),
        )
        result = service.run(
            "report-one-shot-concurrent",
            "pipeline-one-shot-concurrent",
            bundle,
            processing_grants=grants,
        )

        assert result.status == "complete"
        assert [item.status for item in nested_results] == ["running"]
        assert len(fake.calls) == 1
        row = fixture.database.connection.execute(
            """SELECT status, dispatch_count FROM report_one_shot_runs
               WHERE report_run_id = ?""",
            ("report-one-shot-concurrent",),
        ).fetchone()
        assert tuple(row) == ("complete", 1)
    finally:
        fixture.database.close()


def test_expired_uncertain_dispatch_is_terminal_without_a_second_call(tmp_path) -> None:
    fixture = _fixture(tmp_path, max_input_tokens=50_000_000)
    bundle = _one_shot_bundle(fixture)
    fake = OneShotSol(fixture)
    try:
        service = ReportExecutionService(
            fixture.database,
            fixture.store,
            fixture.coordinator.gate,
            ReportArtifactStore(tmp_path / "release"),
            direct_invoker_factory=lambda: fake,
            runtime_config=ReportRuntimeConfig(
                True,
                ReportResources.defaults(),
                profile="stage4b_oneshot_sol",
                execution_strategy="one_shot",
            ),
        )
        pending = service.run(
            "report-one-shot-expired",
            "pipeline-one-shot-expired",
            bundle,
            processing_grants={},
        )
        assert pending.status == "manual_required"
        fixture.database.connection.execute(
            """UPDATE report_one_shot_runs
               SET status = 'running', dispatch_count = 1,
                   budget_calls_reserved = 1,
                   dispatch_expires_at = '2000-01-01T00:00:00.000000Z'
               WHERE report_run_id = ?""",
            ("report-one-shot-expired",),
        )
        fixture.database.connection.commit()

        result = service.run(
            "report-one-shot-expired",
            "pipeline-one-shot-expired",
            bundle,
            processing_grants={},
        )

        assert result.status == "failed"
        assert "will not be dispatched again" in str(result.error)
        assert fake.calls == []
        row = fixture.database.connection.execute(
            """SELECT status, dispatch_count FROM report_one_shot_runs
               WHERE report_run_id = ?""",
            ("report-one-shot-expired",),
        ).fetchone()
        assert tuple(row) == ("failed", 1)
    finally:
        fixture.database.close()


def test_concurrent_local_publish_reconciles_without_demoting_success(tmp_path) -> None:
    fixture = _fixture(tmp_path, max_input_tokens=50_000_000)
    bundle = _one_shot_bundle(fixture)
    fake = OneShotSol(fixture)
    grants = {
        artifact.artifact_hash: _sol_grant(
            fixture,
            artifact,
            f"one-shot-publish-grant-{index}",
            expires_at="2026-09-11T00:00:00Z",
        )
        for index, artifact in enumerate(fixture.artifacts, start=1)
    }
    nested_results = []
    service = None

    class ReentrantReportStore(ReportArtifactStore):
        entered = False

        def write(self, **kwargs):
            if not self.entered:
                self.entered = True
                assert service is not None
                nested_results.append(service.run(
                    "report-one-shot-publish-race",
                    "pipeline-one-shot-publish-race",
                    bundle,
                    processing_grants=grants,
                ))
            return super().write(**kwargs)

    try:
        report_store = ReentrantReportStore(tmp_path / "release")
        service = ReportExecutionService(
            fixture.database,
            fixture.store,
            fixture.coordinator.gate,
            report_store,
            direct_invoker_factory=lambda: fake,
            runtime_config=ReportRuntimeConfig(
                True,
                ReportResources.defaults(),
                profile="stage4b_oneshot_sol",
                execution_strategy="one_shot",
            ),
        )
        result = service.run(
            "report-one-shot-publish-race",
            "pipeline-one-shot-publish-race",
            bundle,
            processing_grants=grants,
        )

        assert result.status == "complete"
        assert [item.status for item in nested_results] == ["complete"]
        assert len(fake.calls) == 1
        row = fixture.database.connection.execute(
            """SELECT os.status, rr.status
               FROM report_one_shot_runs os
               JOIN report_runs rr USING(report_run_id)
               WHERE os.report_run_id = ?""",
            ("report-one-shot-publish-race",),
        ).fetchone()
        assert tuple(row) == ("complete", "complete")
    finally:
        fixture.database.close()


def test_reconcile_failure_does_not_overwrite_report_audit_attestation(
    tmp_path,
) -> None:
    fixture = _fixture(tmp_path, max_input_tokens=50_000_000)
    bundle = _one_shot_bundle(fixture)
    fake = OneShotSol(fixture)
    grants = {
        artifact.artifact_hash: _sol_grant(
            fixture,
            artifact,
            f"one-shot-reconcile-grant-{index}",
            expires_at="2026-09-11T00:00:00Z",
        )
        for index, artifact in enumerate(fixture.artifacts, start=1)
    }

    class RejectingReconcileStore(ReportArtifactStore):
        def reconcile(self, **kwargs):
            raise ReportArtifactError("frozen bundle uses an older contract")

    try:
        service = ReportExecutionService(
            fixture.database,
            fixture.store,
            fixture.coordinator.gate,
            RejectingReconcileStore(tmp_path / "release"),
            direct_invoker_factory=lambda: fake,
            runtime_config=ReportRuntimeConfig(
                True,
                ReportResources.defaults(),
                profile="stage4b_oneshot_sol",
                execution_strategy="one_shot",
            ),
        )

        result = service.run(
            "report-one-shot-reconcile-rejected",
            "pipeline-one-shot-reconcile-rejected",
            bundle,
            processing_grants=grants,
        )

        assert result.status == "failed"
        assert "older contract" in str(result.error)
        assert fixture.database.connection.execute(
            "SELECT COUNT(*) FROM report_audit_runs WHERE report_run_id = ?",
            ("report-one-shot-reconcile-rejected",),
        ).fetchone()[0] == 0
    finally:
        fixture.database.close()


def test_overlapping_temporary_publication_waits_for_atomic_winner(tmp_path) -> None:
    fixture = _fixture(tmp_path, max_input_tokens=50_000_000)
    bundle = _one_shot_bundle(fixture)
    fake = OneShotSol(fixture)
    grants = {
        artifact.artifact_hash: _sol_grant(
            fixture,
            artifact,
            f"one-shot-temp-race-grant-{index}",
            expires_at="2026-09-11T00:00:00Z",
        )
        for index, artifact in enumerate(fixture.artifacts, start=1)
    }

    class OverlappingReportStore(ReportArtifactStore):
        entered = False
        background_errors = []

        def write(self, **kwargs):
            if self.entered:
                return super().write(**kwargs)
            self.entered = True

            def publish() -> None:
                try:
                    ReportArtifactStore.write(self, **kwargs)
                except Exception as error:  # pragma: no cover - asserted below.
                    self.background_errors.append(error)

            worker = Thread(target=publish)
            worker.start()
            report_run_id = str(kwargs["document"]["report_run_id"])
            target = self.directory(report_run_id)
            temporary = target.with_name(f".{report_run_id}.tmp")
            for _ in range(100):
                if temporary.exists() or target.exists():
                    break
                sleep(0.01)
            try:
                return ReportArtifactStore.write(self, **kwargs)
            finally:
                worker.join()

    try:
        report_store = OverlappingReportStore(tmp_path / "release")
        service = ReportExecutionService(
            fixture.database,
            fixture.store,
            fixture.coordinator.gate,
            report_store,
            direct_invoker_factory=lambda: fake,
            runtime_config=ReportRuntimeConfig(
                True,
                ReportResources.defaults(),
                profile="stage4b_oneshot_sol",
                execution_strategy="one_shot",
            ),
        )
        result = service.run(
            "report-one-shot-temp-race",
            "pipeline-one-shot-temp-race",
            bundle,
            processing_grants=grants,
        )

        assert result.status == "complete"
        assert report_store.background_errors == []
        assert len(fake.calls) == 1
        row = fixture.database.connection.execute(
            """SELECT os.status, rr.status
               FROM report_one_shot_runs os
               JOIN report_runs rr USING(report_run_id)
               WHERE os.report_run_id = ?""",
            ("report-one-shot-temp-race",),
        ).fetchone()
        assert tuple(row) == ("complete", "complete")
    finally:
        fixture.database.close()
