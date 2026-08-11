from __future__ import annotations

from hashlib import sha256
import json
from threading import Thread
from time import sleep
from types import MappingProxyType

from paper_agent.canonical import content_hash
from paper_agent.codex_exec import CodexExecResult, InvocationMetadata
from paper_agent.report_artifacts import LOCAL_REFERENCES_NOTE, ReportArtifactStore
from paper_agent.report_config import ReportResources, ReportRuntimeConfig
from paper_agent.report_direct import (
    one_shot_config_hash,
    one_shot_validation_config_hash,
)
from paper_agent.report_execution_service import ReportExecutionService
from paper_agent.report_plan import (
    ReportPlanBundle,
    approve_report_plan,
    compile_report_plan,
)
from paper_agent.reporting import stable_claim_id

from test_report_reduce import _claim, _draft, _fixture, _sol_grant


def _one_shot_bundle(fixture) -> ReportPlanBundle:
    resources = ReportResources.defaults()
    draft = _draft(10_000_000)
    draft["execution_strategy"] = "one_shot"
    draft["stage4b_config_hash"] = one_shot_config_hash(
        fixture.coordinator.gate.policy.hash,
        resources=resources,
    )
    draft["stage4b_audit_config_hash"] = one_shot_validation_config_hash()
    draft["budget"] = {
        "max_sol_calls": 1,
        "max_input_tokens": 10_000_000,
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


def test_cross_section_claim_is_derived_per_section_and_resume_does_not_redispatch(
    tmp_path,
) -> None:
    fixture = _fixture(tmp_path, max_input_tokens=50_000_000)
    bundle = _one_shot_bundle(fixture)

    def reuse_home_claim(output, _payload) -> None:
        claim, _ = _section_claim(output, "executive_summary")
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
