from __future__ import annotations

from dataclasses import dataclass

import pytest

from paper_agent.authorized_luna import AuthorizedLunaPlanner
from paper_agent.codex_exec import CodexExecResult, InvocationMetadata
from paper_agent.stage3_pipeline import LunaPlannerInput


@dataclass
class FakeInvoker:
    candidate_id: str = "candidate-1"
    calls: list[object] | None = None

    def invoke(self, request):
        assert self.calls is not None
        self.calls.append(request)
        output = {
            "schema_version": "1",
            "candidate_id": self.candidate_id,
            "status": "invoke_skill",
            "page_state": "unknown",
            "next_action": "invoke_audited_skill",
            "reason_code": "authorized_handoff_selected",
            "sensitive_data_included": False,
        }
        metadata = InvocationMetadata(
            "invoke-1", "stage3_authorized_luna", "gpt-5.6-luna", "low",
            "authorized-browser-result.schema.json", "a" * 64, request.input_hash,
            "authorized-browser.md", "b" * 64, "c" * 64, None, 1,
            "gpt-5.6-luna", "stage3_authorized_luna",
        )
        return CodexExecResult(output, metadata)


def test_planner_uses_frozen_luna_profile_and_only_sanitized_control_fields() -> None:
    calls: list[object] = []
    planner = AuthorizedLunaPlanner(FakeInvoker(calls=calls))
    control = LunaPlannerInput(
        "candidate-1", "paper-1", "nature.com", "needs_grant", "explicit_download_grant_required",
    )

    decision = planner(control)

    assert decision.selected
    assert decision.invocation_metadata["model"] == "gpt-5.6-luna"
    request = calls[0]
    assert request.profile == "stage3_authorized_luna"
    assert request.prompt_name == "authorized-browser.md"
    assert request.schema_name == "authorized-browser-result.schema.json"
    assert set(__import__("json").loads(request.prompt)) == {
        "candidate_id", "paper_id", "host", "status", "reason_code",
    }
    assert "url" not in request.prompt.lower()
    assert "cookie" not in request.prompt.lower()


def test_planner_rejects_candidate_substitution() -> None:
    planner = AuthorizedLunaPlanner(FakeInvoker(candidate_id="other", calls=[]))

    with pytest.raises(ValueError, match="binding mismatch"):
        planner(LunaPlannerInput("candidate-1", "paper-1", None, "manual", "reason"))
