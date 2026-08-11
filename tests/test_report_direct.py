from __future__ import annotations

from hashlib import sha256
import json
from threading import Thread
from time import sleep

from paper_agent.canonical import content_hash
from paper_agent.codex_exec import CodexExecResult, InvocationMetadata
from paper_agent.report_artifacts import ReportArtifactStore
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
    def __init__(self, fixture) -> None:
        self.fixture = fixture
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
        return CodexExecResult(output, metadata)


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
