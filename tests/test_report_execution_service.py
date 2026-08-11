from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import pytest

from paper_agent.codex_exec import CodexExecResult
from paper_agent.canonical import content_hash
from paper_agent.config import load_config
from paper_agent.report_artifacts import ReportArtifactStore
from paper_agent.report_audit import ReportAuditCoordinator
from paper_agent.report_config import ReportConfigError, ReportResources, ReportRuntimeConfig
from paper_agent.report_execution_service import ReportExecutionService
from paper_agent.report_plan import ReportPlanBundle
from paper_agent.report_reduce import SolReduceCoordinator

from test_report_audit import DISCLOSURE, FakeAuditSol
from test_report_reduce import _fixture


def _service(
    fixture,
    tmp_path,
    *,
    execution_mode="attended",
    runtime_config=None,
):
    holder = {}
    service = ReportExecutionService(
        fixture.database, fixture.store, fixture.coordinator.gate,
        ReportArtifactStore(tmp_path / "release"),
        reduce_invoker_factory=lambda: holder["reduce"],
        audit_invoker_factory=lambda: holder["audit"],
        execution_mode=execution_mode,
        runtime_config=runtime_config,
    )
    base = fixture.fake

    class DisclosureFinal:
        def invoke(self, request):
            result = base.invoke(request)
            if request.call_kind != "final_reduce":
                return result
            output = deepcopy(dict(result.output))
            output["blocks"][0]["text"] += " " + DISCLOSURE
            return CodexExecResult(
                output,
                replace(result.metadata, output_hash=content_hash(output)),
            )

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


def test_report_preflight_budget_exhaustion_is_observable_and_free(
    tmp_path,
) -> None:
    fixture = _fixture(
        tmp_path, max_input_tokens=10_000_000, summary_size=100_000
    )
    try:
        service = _service(fixture, tmp_path)
        first = service.run(
            "report-preflight-budget",
            "pipeline-preflight-budget",
            _bundle(fixture),
        )
        second = service.run(
            "report-preflight-budget",
            "pipeline-preflight-budget",
            _bundle(fixture),
        )

        assert first.status == second.status == "incomplete"
        assert first.alarm_codes == second.alarm_codes == (
            "report.codex_budget_exhausted",
        )
        assert first.error == second.error
        assert first.error is not None
        assert first.error["type"] == "SolBudgetError"
        assert first.codex_budget is not None
        assert first.codex_budget.calls_reserved == 0
        assert first.codex_budget.input_tokens_reserved == 0
        assert first.codex_budget.approved_call_limit == 300
        assert first.codex_budget.approved_input_token_limit == 10_000_000
        assert fixture.calls == []
        pipeline = fixture.database.connection.execute(
            "SELECT status, completed_at FROM pipeline_runs WHERE run_id = ?",
            ("pipeline-preflight-budget",),
        ).fetchone()
        report = fixture.database.connection.execute(
            "SELECT status, completed_at FROM report_runs WHERE report_run_id = ?",
            ("report-preflight-budget",),
        ).fetchone()
        nodes = fixture.database.connection.execute(
            """SELECT status, dispatch_count, budget_calls_reserved,
                      budget_tokens_reserved
                 FROM report_reduce_nodes WHERE report_run_id = ?""",
            ("report-preflight-budget",),
        ).fetchall()
        assert pipeline["status"] == report["status"] == "incomplete"
        assert pipeline["completed_at"] is not None
        assert report["completed_at"] is not None
        assert nodes
        assert {tuple(row) for row in nodes} == {("pending", 0, 0, 0)}
    finally:
        fixture.database.close()


def test_transaction_budget_exhaustion_is_incomplete_and_resume_is_free(
    tmp_path, monkeypatch
) -> None:
    fixture = _fixture(tmp_path, max_input_tokens=50_000_000)
    original = SolReduceCoordinator._reserve_budget_and_claim
    claims = 0

    def exhaust_second_claim(self, report_run_id, plan, *args, **kwargs):
        nonlocal claims
        claims += 1
        if claims == 2:
            audit_bounds = args[-1]
            plan = {
                **plan,
                "budget": {
                    **plan["budget"],
                    "max_sol_calls": 2 + int(audit_bounds.worst_case_calls),
                },
            }
        return original(self, report_run_id, plan, *args, **kwargs)

    monkeypatch.setattr(
        SolReduceCoordinator, "_reserve_budget_and_claim", exhaust_second_claim
    )
    try:
        service = _service(fixture, tmp_path)
        first = service.run(
            "report-transaction-budget",
            "pipeline-transaction-budget",
            _bundle(fixture),
        )
        paid_calls = len(fixture.calls)
        second = service.run(
            "report-transaction-budget",
            "pipeline-transaction-budget",
            _bundle(fixture),
        )

        assert first.status == second.status == "incomplete"
        assert first.error is not None
        assert first.error["type"] == "SolBudgetError"
        assert first.alarm_codes == ("report.codex_budget_exhausted",)
        assert paid_calls == 1
        assert len(fixture.calls) == paid_calls
        assert first.codex_budget == second.codex_budget
        assert first.codex_budget is not None
        assert first.codex_budget.calls_reserved == 2
        assert first.codex_budget.input_tokens_reserved > 0
        assert first.codex_budget.approved_call_limit == 300
        row = fixture.database.connection.execute(
            """SELECT status, error_json FROM report_reduce_nodes
               WHERE report_run_id = ? AND status = 'failed'""",
            ("report-transaction-budget",),
        ).fetchone()
        pipeline_status = fixture.database.connection.execute(
            "SELECT status FROM pipeline_runs WHERE run_id = ?",
            ("pipeline-transaction-budget",),
        ).fetchone()[0]
        assert json.loads(row["error_json"])["error"] == "SolBudgetError"
        assert pipeline_status == "incomplete"
    finally:
        fixture.database.close()


def test_audit_claim_budget_exhaustion_is_incomplete_and_resume_is_free(
    tmp_path, monkeypatch
) -> None:
    fixture = _fixture(tmp_path, max_input_tokens=50_000_000)
    original = ReportAuditCoordinator._claim_step
    injected = False

    def exhaust_claim(self, report_run_id, *args, **kwargs):
        nonlocal injected
        if not injected:
            with self.database.transaction() as connection:
                used = connection.execute(
                    """SELECT COALESCE(SUM(calls), 0) FROM (
                           SELECT budget_calls_reserved AS calls
                           FROM report_reduce_nodes WHERE report_run_id = ?
                           UNION ALL
                           SELECT budget_calls_reserved
                           FROM report_audit_steps WHERE report_run_id = ?
                       )""",
                    (report_run_id, report_run_id),
                ).fetchone()[0]
                connection.execute(
                    """UPDATE report_audit_steps
                       SET budget_calls_reserved = budget_calls_reserved + ?
                       WHERE report_run_id = ? AND step_name = 'audit_a'""",
                    (300 - int(used), report_run_id),
                )
            injected = True
        return original(self, report_run_id, *args, **kwargs)

    monkeypatch.setattr(ReportAuditCoordinator, "_claim_step", exhaust_claim)
    try:
        service = _service(fixture, tmp_path)
        first = service.run(
            "report-audit-budget", "pipeline-audit-budget", _bundle(fixture)
        )
        paid_calls = len(fixture.calls)
        with fixture.database.transaction() as connection:
            connection.execute(
                "UPDATE report_audit_runs SET status = 'failed' WHERE report_run_id = ?",
                ("report-audit-budget",),
            )
            connection.execute(
                "UPDATE report_runs SET status = 'failed' WHERE report_run_id = ?",
                ("report-audit-budget",),
            )
            connection.execute(
                "UPDATE pipeline_runs SET status = 'failed' WHERE run_id = ?",
                ("pipeline-audit-budget",),
            )
        second = service.run(
            "report-audit-budget", "pipeline-audit-budget", _bundle(fixture)
        )

        assert first.status == second.status == "incomplete"
        assert first.error is not None
        assert first.error["type"] == "ReportAuditBudgetError"
        assert first.alarm_codes == second.alarm_codes == (
            "report.codex_budget_exhausted",
        )
        assert first.codex_budget == second.codex_budget
        assert first.codex_budget is not None
        assert first.codex_budget.calls_reserved == 300
        assert len(fixture.calls) == paid_calls
        audit_step = fixture.database.connection.execute(
            """SELECT status, dispatch_count FROM report_audit_steps
               WHERE report_run_id = ? AND step_name = 'audit_a'""",
            ("report-audit-budget",),
        ).fetchone()
        audit_run = fixture.database.connection.execute(
            """SELECT status, error_json FROM report_audit_runs
               WHERE report_run_id = ?""",
            ("report-audit-budget",),
        ).fetchone()
        report_status = fixture.database.connection.execute(
            "SELECT status FROM report_runs WHERE report_run_id = ?",
            ("report-audit-budget",),
        ).fetchone()[0]
        pipeline_status = fixture.database.connection.execute(
            "SELECT status FROM pipeline_runs WHERE run_id = ?",
            ("pipeline-audit-budget",),
        ).fetchone()[0]
        assert tuple(audit_step) == ("pending", 0)
        assert audit_run["status"] == report_status == pipeline_status == "incomplete"
        assert json.loads(audit_run["error_json"]) == {
            "error": "ReportAuditBudgetError",
            "event_code": "report.codex_budget_exhausted",
            "message": "Sol call budget is exhausted before audit dispatch",
        }
    finally:
        fixture.database.close()


def test_configured_default_resources_resume_legacy_pathless_invocations(
    tmp_path,
) -> None:
    fixture = _fixture(tmp_path, max_input_tokens=50_000_000)
    try:
        first = _service(fixture, tmp_path).run(
            "report-legacy-paths", "pipeline-legacy-paths", _bundle(fixture)
        )
        defaults = ReportResources.defaults()
        configured_defaults = ReportResources(
            dict(defaults.schema_paths),
            dict(defaults.prompt_paths),
            configured=True,
        )
        second = _service(
            fixture,
            tmp_path,
            runtime_config=ReportRuntimeConfig(True, configured_defaults),
        ).run("report-legacy-paths", "pipeline-legacy-paths", _bundle(fixture))

        assert first.status == second.status == "complete"
        assert second.audit is not None
        assert second.audit.resumed_steps == ("audit_a",)
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


def test_disabled_summary_skips_inputs_database_and_model_work(tmp_path) -> None:
    fixture = _fixture(tmp_path, max_input_tokens=50_000_000)
    try:
        runtime = ReportRuntimeConfig(False, ReportResources.defaults())
        service = _service(fixture, tmp_path, runtime_config=runtime)
        before = fixture.database.connection.execute(
            "SELECT COUNT(*) FROM report_runs"
        ).fetchone()[0]

        statements = []
        fixture.database.connection.set_trace_callback(statements.append)
        try:
            result = service.run(
                "report-disabled", "pipeline-disabled", _bundle(fixture)
            )
        finally:
            fixture.database.connection.set_trace_callback(None)

        assert result.status == "complete" and result.skipped and not result.dry_run
        assert result.codex_budget is None
        assert service.last_reduce is None and service.last_audit is None
        assert fixture.calls == []
        assert statements == []
        assert fixture.database.connection.execute(
            "SELECT COUNT(*) FROM report_runs"
        ).fetchone()[0] == before
    finally:
        fixture.database.close()


def test_unattended_summary_requires_and_honors_exact_pinned_plan(tmp_path) -> None:
    fixture = _fixture(
        tmp_path, max_input_tokens=50_000_000, execution_mode="unattended"
    )
    pinned = tmp_path / "approved-report-plan.json"
    pinned.write_text(json.dumps(fixture.plan), encoding="utf-8")
    try:
        runtime = ReportRuntimeConfig(
            True,
            ReportResources.defaults(),
            report_plan_path=pinned,
            report_plan_hash=fixture.plan["plan_hash"],
        )
        service = _service(
            fixture,
            tmp_path,
            execution_mode="unattended",
            runtime_config=runtime,
        )

        result = service.run("report-pinned", "pipeline-pinned", _bundle(fixture))

        assert result.status == "complete"
    finally:
        fixture.database.close()


def test_unattended_summary_rejects_plan_file_drift_before_dispatch(tmp_path) -> None:
    fixture = _fixture(
        tmp_path, max_input_tokens=50_000_000, execution_mode="unattended"
    )
    pinned = tmp_path / "drifted-report-plan.json"
    drifted = deepcopy(fixture.plan)
    drifted["objective"] = "Changed after approval"
    pinned.write_text(json.dumps(drifted), encoding="utf-8")
    try:
        runtime = ReportRuntimeConfig(
            True,
            ReportResources.defaults(),
            report_plan_path=pinned,
            report_plan_hash=fixture.plan["plan_hash"],
        )
        service = _service(
            fixture,
            tmp_path,
            execution_mode="unattended",
            runtime_config=runtime,
        )

        with pytest.raises(ReportConfigError, match="differs from the pinned"):
            service.run("report-drifted", "pipeline-drifted", _bundle(fixture))

        assert fixture.calls == []
    finally:
        fixture.database.close()


def test_unattended_summary_rejects_missing_plan_pin(tmp_path) -> None:
    fixture = _fixture(
        tmp_path, max_input_tokens=50_000_000, execution_mode="unattended"
    )
    try:
        service = _service(
            fixture,
            tmp_path,
            execution_mode="unattended",
            runtime_config=ReportRuntimeConfig(True, ReportResources.defaults()),
        )

        with pytest.raises(ReportConfigError, match="requires a pinned"):
            service.run("report-unpinned", "pipeline-unpinned", _bundle(fixture))

        assert fixture.calls == []
    finally:
        fixture.database.close()


def test_custom_summary_resources_drive_hashes_and_invocation_audit(tmp_path) -> None:
    defaults = ReportResources.defaults()
    custom_schema = tmp_path / "custom-section-output.json"
    schema = defaults.schema("section_reduce")
    schema["title"] = "Project-specific section synthesis"
    custom_schema.write_text(json.dumps(schema), encoding="utf-8")
    custom_prompt = tmp_path / "custom-section-prompt.md"
    custom_prompt.write_text(
        defaults.prompt("section_reduce") + "\nFollow the project-specific synthesis rubric.\n",
        encoding="utf-8",
    )
    schemas = dict(defaults.schema_paths)
    schemas["section_reduce"] = custom_schema
    prompts = dict(defaults.prompt_paths)
    prompts["section_reduce"] = custom_prompt
    resources = ReportResources(schemas, prompts, configured=True)
    fixture = _fixture(
        tmp_path,
        max_input_tokens=50_000_000,
        resources=resources,
    )
    try:
        service = _service(
            fixture,
            tmp_path,
            runtime_config=ReportRuntimeConfig(True, resources),
        )

        result = service.run("report-custom", "pipeline-custom", _bundle(fixture))

        assert result.status == "complete"
        row = fixture.database.connection.execute(
            """SELECT prompt_hash, schema_hash, invocation_metadata_json
               FROM report_reduce_nodes
               WHERE report_run_id = ? AND call_kind = 'section_reduce'
               ORDER BY node_id LIMIT 1""",
            ("report-custom",),
        ).fetchone()
        metadata = json.loads(row["invocation_metadata_json"])
        assert row["prompt_hash"] == sha256(custom_prompt.read_bytes()).hexdigest()
        assert row["schema_hash"] == sha256(
            json.dumps(schema, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        assert metadata["prompt_path"] == str(custom_prompt)
        assert metadata["schema_path"] == str(custom_schema)
    finally:
        fixture.database.close()


def test_custom_audit_rubric_is_shared_by_reduce_budget_and_auditor(tmp_path) -> None:
    root = Path(__file__).resolve().parents[1]
    rubric = tmp_path / "custom-report-audit-rubric.yaml"
    rubric.write_text(
        (root / "policies" / "report-audit-rubric-v1.yaml").read_text(
            encoding="utf-8"
        )
        + "\nproject_note: custom frozen rubric\n",
        encoding="utf-8",
    )
    fixture = _fixture(
        tmp_path,
        max_input_tokens=50_000_000,
        rubric_path=rubric,
    )
    try:
        service = _service(
            fixture,
            tmp_path,
            runtime_config=ReportRuntimeConfig(
                True, ReportResources.defaults(), rubric_path=rubric
            ),
        )

        result = service.run(
            "report-custom-rubric", "pipeline-custom-rubric", _bundle(fixture)
        )

        assert result.status == "complete"
        assert service.last_reduce.audit_config_hash == fixture.plan[
            "stage4b_audit_config_hash"
        ]
        assert service.last_audit.config_hash == fixture.plan[
            "stage4b_audit_config_hash"
        ]
        assert service.last_audit.rubric_path == rubric
    finally:
        fixture.database.close()


def test_report_runtime_config_resolves_all_summary_resource_paths() -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = root / "example_config.yaml"
    config = load_config(config_path)

    runtime = ReportRuntimeConfig.from_config(config, config_path)

    assert runtime.enabled
    assert all(path.is_file() for path in runtime.resources.schema_paths.values())
    assert all(path.is_file() for path in runtime.resources.prompt_paths.values())
    assert runtime.rubric_path is not None and runtime.rubric_path.is_file()


def test_report_runtime_config_cannot_weaken_unattended_pinning() -> None:
    resources = ReportResources.defaults()

    with pytest.raises(ReportConfigError, match="must require a pinned"):
        ReportRuntimeConfig(
            True, resources, require_plan_for_unattended=False
        )
    with pytest.raises(ReportConfigError, match="lowercase SHA-256"):
        ReportRuntimeConfig(True, resources, report_plan_hash="not-a-hash")
    with pytest.raises(ReportConfigError, match="execution mode"):
        ReportRuntimeConfig.defaults().validate_for_run(
            {}, execution_mode="background"
        )


def test_report_resources_reject_shared_call_kind_schema() -> None:
    defaults = ReportResources.defaults()
    schemas = dict(defaults.schema_paths)
    schemas["quality_audit"] = schemas["planning_assist"]

    with pytest.raises(ReportConfigError, match="share one output schema"):
        ReportResources(
            schemas, dict(defaults.prompt_paths), configured=True
        ).validate_files()


@pytest.mark.parametrize(
    ("reference", "message"),
    [
        ("#/$defs/missing", "cannot be resolved"),
        ("missing-helper.schema.json", "is unavailable"),
        ("../outside/helper.schema.json", "must name a sibling"),
        (
            "https://example.test/helper.schema.json",
            "not a local frozen resource",
        ),
    ],
)
def test_report_resources_fail_startup_for_unresolvable_or_escaping_refs(
    tmp_path: Path, reference: str, message: str
) -> None:
    defaults = ReportResources.defaults()
    schema_path = tmp_path / "configured-section.schema.json"
    schema_path.write_text(
        json.dumps(
            {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "$id": "https://example.test/configured-section.schema.json",
                "type": "object",
                "additionalProperties": False,
                "required": ["value"],
                "properties": {"value": {"$ref": reference}},
            }
        ),
        encoding="utf-8",
    )
    schemas = dict(defaults.schema_paths)
    schemas["section_reduce"] = schema_path
    resources = ReportResources(
        schemas, dict(defaults.prompt_paths), configured=True
    )

    with pytest.raises(ReportConfigError, match=message):
        resources.validate_files()


def test_report_resources_reject_schema_outside_codex_strict_subset(
    tmp_path: Path,
) -> None:
    defaults = ReportResources.defaults()
    schema = defaults.schema("section_reduce")
    schema["required"] = schema["required"][:-1]
    schema_path = tmp_path / "non-strict-section.schema.json"
    schema_path.write_text(json.dumps(schema), encoding="utf-8")
    schemas = dict(defaults.schema_paths)
    schemas["section_reduce"] = schema_path

    with pytest.raises(ReportConfigError, match="not Codex-compatible"):
        ReportResources(
            schemas, dict(defaults.prompt_paths), configured=True
        ).validate_files()
