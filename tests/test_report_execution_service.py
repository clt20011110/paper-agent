from __future__ import annotations

from copy import deepcopy

import pytest

from paper_agent.codex_exec import CodexExecResult
from paper_agent.report_artifacts import ReportArtifactStore
from paper_agent.report_execution_service import ReportExecutionService
from paper_agent.report_plan import ReportPlanBundle

from test_report_audit import DISCLOSURE, FakeAuditSol
from test_report_reduce import _fixture


def _service(fixture, tmp_path):
    holder = {}
    service = ReportExecutionService(
        fixture.database, fixture.store, fixture.coordinator.gate,
        ReportArtifactStore(tmp_path / "release"),
        reduce_invoker_factory=lambda: holder["reduce"],
        audit_invoker_factory=lambda: holder["audit"],
    )
    base = fixture.fake

    class DisclosureFinal:
        def invoke(self, request):
            result = base.invoke(request)
            if request.call_kind != "final_reduce":
                return result
            output = deepcopy(dict(result.output))
            output["blocks"][0]["text"] += " " + DISCLOSURE
            return CodexExecResult(output, result.metadata)

    holder["reduce"] = DisclosureFinal()
    audit_calls = []
    # The audit fake needs the actual coordinator only when a call is made.
    service.audit_invoker_factory = lambda: FakeAuditSol(service.last_audit, audit_calls)  # type: ignore[arg-type]
    return service


def _bundle(fixture):
    return ReportPlanBundle(fixture.plan, fixture.corpus, fixture.audit)


def test_report_execution_runs_reduce_then_audit_and_resumes(tmp_path) -> None:
    fixture = _fixture(tmp_path, max_input_tokens=50_000_000)
    try:
        service = _service(fixture, tmp_path)
        first = service.run("report-service", "pipeline-service", _bundle(fixture))
        second = service.run("report-service", "pipeline-service", _bundle(fixture))
        assert first.status == "complete"
        assert first.audit is not None and first.audit.audit_passes == ("A",)
        assert second.status == "complete"
        assert second.audit is not None and second.audit.resumed_steps == ("audit_a",)
    finally:
        fixture.database.close()


def test_report_execution_dry_run_is_pure_preflight(tmp_path) -> None:
    fixture = _fixture(tmp_path, max_input_tokens=50_000_000)
    try:
        service = _service(fixture, tmp_path)
        result = service.run("report-dry", "pipeline-dry", _bundle(fixture), dry_run=True)
        assert result.status == "validated" and result.dry_run
        assert fixture.database.connection.execute("SELECT COUNT(*) FROM report_runs").fetchone()[0] == 0
        assert fixture.database.connection.execute("SELECT COUNT(*) FROM report_plans").fetchone()[0] == 1
    finally:
        fixture.database.close()


def test_report_execution_dry_run_rejects_malformed_processing_grants(tmp_path) -> None:
    fixture = _fixture(tmp_path, max_input_tokens=50_000_000)
    try:
        service = _service(fixture, tmp_path)
        with pytest.raises(ValueError, match="artifact SHA-256"):
            service.run(
                "report-dry-invalid-grant",
                "pipeline-dry-invalid-grant",
                _bundle(fixture),
                processing_grants={"not-a-hash": "grant-1"},
                dry_run=True,
            )
        assert fixture.database.connection.execute("SELECT COUNT(*) FROM report_runs").fetchone()[0] == 0
    finally:
        fixture.database.close()
