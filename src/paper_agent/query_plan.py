"""Compilation, approval, persistence, and replay checks for QueryPlans."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from .approval import ApprovalError, approve, approved_content_hash, require_valid_approval
from .canonical import canonical_json
from .query_compilers import COMPILER_VERSION, compile_queries


class QueryPlanError(ValueError):
    pass


class QueryPlanDriftError(QueryPlanError):
    pass


_RUNTIME_PROVIDER_FIELDS = (
    "distribution",
    "version",
    "artifact_sha256",
    "manifest_hash",
    "mode",
    "snapshot_hash",
    "credentials_present",
    "query_compiler_version",
    "native_query_hashes",
)


def compile_query_plan(
    draft: Mapping[str, Any],
    *,
    providers: Sequence[Mapping[str, Any]] | None = None,
    plan_id: str | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Turn a plain mapping into a frozen, deterministic QueryPlan draft."""
    source = deepcopy(dict(draft))
    provider_specs = list(providers if providers is not None else source.pop("providers", ()))
    if not provider_specs:
        raise QueryPlanError("a QueryPlan needs at least one provider")
    if created_at is None:
        created_at = str(source.pop("created_at", ""))
    if not created_at:
        raise QueryPlanError("created_at is required to compile a QueryPlan")

    requirements = _requirements(source)
    requirements["required_providers"] = sorted(
        set(requirements["required_providers"])
        | {str(spec["provider"]) for spec in provider_specs if spec.get("exact_required") is True}
    )
    variants = list(source["query_variants"])
    scope = source["scope"]
    compiled_providers = [
        _compile_provider(spec, variants, scope, page_size=_page_size(source), requirements=requirements)
        for spec in provider_specs
    ]
    _validate_resolution(compiled_providers, requirements)

    plan = {
        "schema_version": "1",
        "plan_id": plan_id or "",
        "plan_hash": "",
        "status": "draft",
        "created_at": created_at,
        "research": source["research"],
        "scope": scope,
        "inclusion": source["inclusion"],
        "query_variants": variants,
        "providers": compiled_providers,
        "filter": source["filter"],
        "citation_snowball": source["citation_snowball"],
        "budgets": source["budgets"],
        "execution": requirements,
        "approval": None,
    }
    plan_hash = approved_content_hash(plan)
    plan["plan_id"] = plan_id or f"search-{plan_hash[:12]}"
    plan["plan_hash"] = plan_hash
    return plan


def approve_query_plan(
    plan: Mapping[str, Any],
    expected_hash: str,
    *,
    approved_by: str,
    approved_at: str,
) -> dict[str, Any]:
    return approve(
        plan,
        expected_hash,
        approved_by=approved_by,
        approved_at=approved_at,
        hash_field="plan_hash",
    )


def runtime_requirements(plan: Mapping[str, Any]) -> dict[str, Any]:
    return _requirements(plan)


def assert_runtime_matches(
    plan: Mapping[str, Any],
    runtime_providers: Sequence[Mapping[str, Any]],
    *,
    budgets: Mapping[str, Any] | None = None,
    policies: Mapping[str, Any] | None = None,
) -> None:
    """Reject every environment change that would alter replayed searches."""
    try:
        require_valid_approval(plan, "plan_hash")
    except ApprovalError as error:
        raise QueryPlanDriftError(str(error)) from error
    recorded = {str(provider["provider"]): provider for provider in plan["providers"]}
    observed = {str(provider["provider"]): provider for provider in runtime_providers}
    if set(recorded) != set(observed):
        raise QueryPlanDriftError("resolved provider set has drifted")
    for provider_name, expected in recorded.items():
        actual = observed[provider_name]
        for field in _RUNTIME_PROVIDER_FIELDS:
            if expected.get(field) != actual.get(field):
                raise QueryPlanDriftError(f"provider {provider_name} {field} has drifted")
    if budgets is not None and dict(plan["budgets"]) != dict(budgets):
        raise QueryPlanDriftError("request budgets have drifted")
    if policies is not None and runtime_requirements(plan) != _requirements(policies):
        raise QueryPlanDriftError("provider policy has drifted")


class QueryPlanStore:
    """Filesystem store with immutable approved plans and an atomic pointer."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def draft_path(self, plan_id: str) -> Path:
        return self.root / "search" / plan_id / "QUERY_PLAN.draft.json"

    def approved_path(self, plan_id: str) -> Path:
        return self.root / "search" / plan_id / "QUERY_PLAN.json"

    @property
    def latest_path(self) -> Path:
        return self.root / "search" / "latest-approved.json"

    def save_draft(self, plan: Mapping[str, Any]) -> Path:
        if plan.get("status") != "draft":
            raise QueryPlanError("save_draft only accepts draft plans")
        return self._write_replacing(self.draft_path(str(plan["plan_id"])), plan)

    def approve_and_save(
        self,
        plan: Mapping[str, Any],
        expected_hash: str,
        *,
        approved_by: str,
        approved_at: str,
    ) -> dict[str, Any]:
        approved = approve_query_plan(
            plan,
            expected_hash,
            approved_by=approved_by,
            approved_at=approved_at,
        )
        self.save_approved(approved)
        return approved

    def save_approved(self, plan: Mapping[str, Any]) -> Path:
        try:
            require_valid_approval(plan, "plan_hash")
        except ApprovalError as error:
            raise QueryPlanError(str(error)) from error
        path = self.approved_path(str(plan["plan_id"]))
        payload = canonical_json(dict(plan))
        if path.exists():
            if path.read_bytes() != payload:
                raise QueryPlanError("approved QueryPlan is immutable")
        else:
            self._atomic_write(path, payload)
        self._atomic_write(
            self.latest_path,
            canonical_json({"plan_id": plan["plan_id"], "plan_hash": plan["plan_hash"]}),
        )
        return path

    def load_approved(self, plan_id: str) -> dict[str, Any]:
        return json.loads(self.approved_path(plan_id).read_text(encoding="utf-8"))

    def _write_replacing(self, path: Path, document: Mapping[str, Any]) -> Path:
        self._atomic_write(path, canonical_json(dict(document)))
        return path

    @staticmethod
    def _atomic_write(path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_bytes(payload)
        os.replace(temporary, path)


def _compile_provider(
    specification: Mapping[str, Any],
    variants: list[Mapping[str, Any]],
    scope: Mapping[str, Any],
    *,
    page_size: int,
    requirements: Mapping[str, Any],
) -> dict[str, Any]:
    provider = str(specification["provider"])
    roles = sorted(str(role) for role in specification["roles"])
    credentials_present = bool(specification.get("credentials_present", False))
    credentials_required = bool(specification.get("credentials_required", False))
    enabled = _provider_enabled(specification.get("enabled", True), scope)
    manifest_trusted = bool(specification.get("manifest_trusted", True))
    snapshot_hash = specification.get("snapshot_hash")
    mode = str(specification.get("mode", "api"))
    resolved = enabled and manifest_trusted and (not credentials_required or credentials_present)
    if mode in {"snapshot", "bulk_snapshot"} and not snapshot_hash:
        resolved = False
    if provider in requirements["required_providers"] and not resolved:
        reason = "explicit_required_unavailable"
    elif not enabled:
        reason = "disabled"
    elif not manifest_trusted:
        reason = "untrusted_manifest"
    elif credentials_required and not credentials_present:
        reason = "credentials_unavailable"
    elif mode in {"snapshot", "bulk_snapshot"} and not snapshot_hash:
        reason = "snapshot_unavailable"
    else:
        reason = "resolved"
    declared_compiler_version = str(specification.get("query_compiler_version", COMPILER_VERSION))
    if declared_compiler_version != COMPILER_VERSION:
        raise QueryPlanError(f"provider {provider} query compiler version is unavailable")
    queries = (
        compile_queries(provider, variants, scope, page_size=page_size)
        if resolved and "search" in roles
        else ()
    )
    return {
        "provider": provider,
        "distribution": str(specification["distribution"]),
        "version": str(specification["version"]),
        "artifact_sha256": specification["artifact_sha256"],
        "manifest_hash": str(specification["manifest_hash"]),
        "roles": roles,
        "capabilities": sorted(str(capability) for capability in specification["capabilities"]),
        "resolved": resolved,
        "resolution_reason": reason,
        "mode": mode,
        "snapshot_hash": snapshot_hash,
        "credentials_present": credentials_present,
        "query_compiler_version": COMPILER_VERSION,
        "native_query_hashes": [query.query_hash for query in queries],
    }


def _requirements(document: Mapping[str, Any]) -> dict[str, Any]:
    execution = document.get("execution", document)
    return {
        "provider_policy": str(execution.get("provider_policy", "all_resolved")),
        "required_roles": sorted(str(role) for role in execution.get("required_roles", ("search",))),
        "required_providers": sorted(str(provider) for provider in execution.get("required_providers", ())),
    }


def _page_size(document: Mapping[str, Any]) -> int:
    return int(document.get("page_size", 100))


def _provider_enabled(setting: Any, scope: Mapping[str, Any]) -> bool:
    if isinstance(setting, bool):
        return setting
    fields = " ".join(str(field).casefold() for field in scope.get("fields", ()))
    if setting == "auto_for_cs":
        return any(term in fields for term in ("computer", "computing", "informatics", "ai", "machine learning"))
    if setting == "auto_for_biomed":
        return any(term in fields for term in ("bio", "med", "health", "life science", "chem"))
    raise QueryPlanError(f"unknown provider enabled policy {setting}")


def _validate_resolution(providers: Sequence[Mapping[str, Any]], requirements: Mapping[str, Any]) -> None:
    if requirements["provider_policy"] != "all_resolved":
        raise QueryPlanError("provider_policy must be all_resolved")
    resolved = [provider for provider in providers if provider["resolved"]]
    names = {str(provider["provider"]) for provider in resolved}
    missing_exact = set(requirements["required_providers"]) - names
    if missing_exact:
        raise QueryPlanError(f"explicit required providers unavailable: {', '.join(sorted(missing_exact))}")
    roles = {role for provider in resolved for role in provider["roles"]}
    missing_roles = set(requirements["required_roles"]) - roles
    if missing_roles:
        raise QueryPlanError(f"required provider roles unavailable: {', '.join(sorted(missing_roles))}")
