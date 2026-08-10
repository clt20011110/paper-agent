"""Pure canonical Stage 4b budget reconstruction shared by reduce and audit."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


class CanonicalReportBudgetError(ValueError):
    """Canonical report budget inputs are incomplete or inconsistent."""


def canonical_report_budget(
    nodes: Sequence[Any],
    prompt_token_bounds: Mapping[str, int],
    audit_repair_bounds: Mapping[str, Any],
    *,
    max_retries: int,
) -> dict[str, int]:
    """Build the only trusted aggregation-tree budget from frozen bounds."""
    if max_retries < 0:
        raise CanonicalReportBudgetError("max_retries must be non-negative")
    node_ids = [str(node.node_id) for node in nodes]
    if (
        not node_ids
        or len(set(node_ids)) != len(node_ids)
        or set(prompt_token_bounds) != set(node_ids)
        or any(int(prompt_token_bounds[node_id]) < 1 for node_id in node_ids)
    ):
        raise CanonicalReportBudgetError(
            "canonical generation prompt bounds do not exactly match the reduce nodes"
        )
    required = {
        "audit_a_input_tokens",
        "repair_input_tokens",
        "audit_c_input_tokens",
        "worst_case_calls",
        "worst_case_input_tokens",
    }
    if any(key not in audit_repair_bounds for key in required):
        raise CanonicalReportBudgetError("canonical audit/repair bounds are incomplete")
    values = {key: int(audit_repair_bounds[key]) for key in required}
    if any(value < 1 for value in values.values()):
        raise CanonicalReportBudgetError("canonical audit/repair bounds must be positive")
    attempts = max_retries + 1
    generation_input_tokens = sum(int(prompt_token_bounds[node_id]) for node_id in node_ids)
    return {
        "generation_calls": len(node_ids),
        "audit_calls": 2,
        "repair_calls": 1,
        "worst_case_calls": len(node_ids) * attempts + values["worst_case_calls"],
        "generation_input_tokens": generation_input_tokens,
        "audit_input_tokens": (
            values["audit_a_input_tokens"] + values["audit_c_input_tokens"]
        ),
        "repair_input_tokens": values["repair_input_tokens"],
        "worst_case_input_tokens": (
            generation_input_tokens * attempts + values["worst_case_input_tokens"]
        ),
    }
