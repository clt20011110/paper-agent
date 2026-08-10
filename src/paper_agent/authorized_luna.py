"""Frozen Luna planner for the optional authorized Stage 3 handoff."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Any, Protocol

from .codex_exec import CodexExec, CodexExecRequest, CodexExecResult
from .schema import schema_directory
from .stage3_pipeline import LunaPlannerInput


class PlannerInvoker(Protocol):
    def invoke(self, request: CodexExecRequest) -> CodexExecResult: ...


@dataclass(frozen=True, slots=True)
class AuthorizedLunaDecision:
    selected: bool
    status: str
    page_state: str
    next_action: str
    reason_code: str
    invocation_metadata: Mapping[str, Any]


class AuthorizedLunaPlanner:
    """Send only sanitized identifiers and control state to gpt-5.6-luna."""

    def __init__(self, invoker: PlannerInvoker | None = None) -> None:
        self.invoker = invoker or CodexExec()
        path = schema_directory() / "authorized-browser-result.schema.json"
        self.schema = json.loads(path.read_text(encoding="utf-8"))

    def __call__(self, control: LunaPlannerInput) -> AuthorizedLunaDecision:
        payload = json.dumps(asdict(control), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        result = self.invoker.invoke(CodexExecRequest(
            profile="stage3_authorized_luna",
            prompt=payload,
            output_schema=self.schema,
            schema_name="authorized-browser-result.schema.json",
            prompt_name="authorized-browser.md",
            input_hash=sha256(payload.encode("utf-8")).hexdigest(),
        ))
        metadata = result.metadata
        if (
            metadata.profile != "stage3_authorized_luna"
            or metadata.model != "gpt-5.6-luna"
            or metadata.reasoning_effort != "low"
            or metadata.actual_model != "gpt-5.6-luna"
            or metadata.actual_profile != "stage3_authorized_luna"
        ):
            raise ValueError(
                "Luna invocation metadata does not match the frozen authorized-download profile"
            )
        output = result.output
        if output["candidate_id"] != control.candidate_id:
            raise ValueError("Luna planner candidate binding mismatch")
        selected = (
            output["status"] == "invoke_skill"
            and output["next_action"] == "invoke_audited_skill"
            and output["sensitive_data_included"] is False
        )
        return AuthorizedLunaDecision(
            selected=selected,
            status=str(output["status"]),
            page_state=str(output["page_state"]),
            next_action=str(output["next_action"]),
            reason_code=str(output["reason_code"]),
            invocation_metadata=asdict(metadata),
        )
