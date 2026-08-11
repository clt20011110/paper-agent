from __future__ import annotations

from inspect import signature
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from paper_agent.config import load_config
from paper_agent.report_config import (
    ReportConfigError,
    ReportResources,
    ReportRuntimeConfig,
)
from paper_agent.report_execution_service import ReportExecutionService


class _UnexpectedAccess:
    def __getattr__(self, name: str) -> object:
        raise AssertionError(f"legacy run touched {name}")


def test_report_execution_service_exposes_only_the_one_shot_invoker() -> None:
    parameters = signature(ReportExecutionService).parameters

    assert "direct_invoker_factory" in parameters
    assert "reduce_invoker_factory" not in parameters
    assert "audit_invoker_factory" not in parameters


def test_legacy_reduce_plan_is_rejected_before_io_or_dispatch() -> None:
    service = ReportExecutionService(
        _UnexpectedAccess(),
        _UnexpectedAccess(),
        _UnexpectedAccess(),
        _UnexpectedAccess(),
        direct_invoker_factory=lambda: (_ for _ in ()).throw(
            AssertionError("model dispatch was attempted")
        ),
    )
    bundle = SimpleNamespace(
        plan={"execution_strategy": "reduce_tree"},
        corpus_snapshot=_UnexpectedAccess(),
        search_audit=_UnexpectedAccess(),
    )

    with pytest.raises(ReportConfigError, match="does not match reduce_tree"):
        service.run("legacy-report", "legacy-pipeline", bundle)


def test_disabled_summary_skips_before_io_or_dispatch() -> None:
    service = ReportExecutionService(
        _UnexpectedAccess(),
        _UnexpectedAccess(),
        _UnexpectedAccess(),
        _UnexpectedAccess(),
        runtime_config=ReportRuntimeConfig(False, ReportResources.defaults()),
    )

    result = service.run(
        "disabled-report",
        "disabled-pipeline",
        SimpleNamespace(plan={}, corpus_snapshot=None, search_audit=None),
    )

    assert result.status == "complete"
    assert result.skipped
    assert result.codex_budget is None


def test_report_runtime_config_resolves_all_summary_resource_paths() -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = root / "example_config.yaml"
    config = load_config(config_path)

    runtime = ReportRuntimeConfig.from_config(config, config_path)

    assert runtime.enabled
    assert runtime.execution_strategy == "one_shot"
    assert runtime.profile == "stage4b_oneshot_sol"
    assert all(path.is_file() for path in runtime.resources.schema_paths.values())
    assert all(path.is_file() for path in runtime.resources.prompt_paths.values())
    assert runtime.rubric_path is not None and runtime.rubric_path.is_file()


def test_report_runtime_config_cannot_weaken_frozen_execution() -> None:
    resources = ReportResources.defaults()

    with pytest.raises(ReportConfigError, match="must require a pinned"):
        ReportRuntimeConfig(
            True, resources, require_plan_for_unattended=False
        )
    with pytest.raises(ReportConfigError, match="lowercase SHA-256"):
        ReportRuntimeConfig(True, resources, report_plan_hash="not-a-hash")
    with pytest.raises(ReportConfigError, match="execution_strategy=one_shot"):
        ReportRuntimeConfig(
            True,
            resources,
            profile="stage4b_summary_sol",
            execution_strategy="reduce_tree",
        )
    with pytest.raises(ReportConfigError, match="execution mode"):
        ReportRuntimeConfig.defaults().validate_for_run(
            {}, execution_mode="background"
        )


def test_one_shot_runtime_rejects_a_custom_audit_rubric(tmp_path: Path) -> None:
    rubric = tmp_path / "custom-rubric.yaml"
    rubric.write_text("version: custom\n", encoding="utf-8")
    runtime = ReportRuntimeConfig(
        True,
        ReportResources.defaults(),
        rubric_path=rubric,
    )

    with pytest.raises(ReportConfigError, match="frozen report audit rubric"):
        runtime.validate_for_run(
            {"execution_strategy": "one_shot"}, execution_mode="attended"
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
