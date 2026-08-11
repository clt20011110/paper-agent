from __future__ import annotations

import pytest

from paper_agent.report_config import (
    ReportConfigError,
    ReportResources,
    ReportRuntimeConfig,
)


def test_one_shot_resource_is_frozen_and_codex_compatible() -> None:
    resources = ReportResources.defaults()

    resources.validate_files()
    schema = resources.schema("one_shot_report")

    assert schema["required"] == [
        "claims",
        "blocks",
        "unresolved_conflicts",
        "claim_relations",
    ]
    claim = schema["$defs"]["draftClaim"]
    assert {
        "claim_ref",
        "subject_id",
        "predicate_id",
        "object_or_scope_id",
        "qualifier_context",
    }.issubset(claim["required"])
    assert not {
        "claim_id",
        "claim_key",
        "comparison_group_id",
        "mapping_status",
    }.intersection(claim["properties"])
    assert resources.service_schema_hash("one_shot_report")


def test_one_shot_runtime_requires_matching_profile_and_plan_strategy() -> None:
    resources = ReportResources.defaults()
    runtime = ReportRuntimeConfig(
        True,
        resources,
        profile="stage4b_oneshot_sol",
        execution_strategy="one_shot",
    )

    runtime.validate_for_run(
        {"execution_strategy": "one_shot"}, execution_mode="attended"
    )

    with pytest.raises(ReportConfigError, match="does not match"):
        runtime.validate_for_run(
            {"execution_strategy": "reduce_tree"}, execution_mode="attended"
        )
    with pytest.raises(ReportConfigError, match="requires profile"):
        ReportRuntimeConfig(
            True,
            resources,
            profile="stage4b_summary_sol",
            execution_strategy="one_shot",
        )
