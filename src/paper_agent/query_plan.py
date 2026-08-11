"""Compilation, approval, persistence, and replay checks for QueryPlans."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from .approval import ApprovalError, approve, approved_content_hash, require_valid_approval
from .canonical import canonical_json, content_hash
from .query_compilers import COMPILER_VERSION, compile_queries
from .scope_filter import screening_scope_hash


class QueryPlanError(ValueError):
    pass


class QueryPlanDriftError(QueryPlanError):
    pass


QUERY_PLAN_SCHEMA_VERSION = "2"


_RUNTIME_PROVIDER_FIELDS = (
    "distribution",
    "version",
    "entry_point",
    "artifact_sha256",
    "manifest_hash",
    "enabled",
    "required",
    "roles",
    "capabilities",
    "authority",
    "credential_environment_variables",
    "credential_availability",
    "credentials_required",
    "rate_limit",
    "data_use",
    "terms_url",
    "independence_group",
    "upstream_families",
    "upstream_policies",
    "resolved",
    "resolution_reason",
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
    venue_specs: Sequence[Mapping[str, Any]] | None = None,
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
    venue_primary_providers = {
        str(item["descriptor"]["primary_provider"])
        for item in (venue_specs or ())
    }
    requirements["required_providers"] = sorted(
        set(requirements["required_providers"])
        | {str(spec["provider"]) for spec in provider_specs if spec.get("exact_required") is True}
        | venue_primary_providers
    )
    variants = list(source["query_variants"])
    scope = dict(source["scope"])
    scope.setdefault("include_arxiv_candidates", False)
    filter_config = dict(source["filter"])
    filter_config["screening_scope_hash"] = screening_scope_hash({
        "research": source["research"],
        "inclusion": source["inclusion"],
        "scope": scope,
    })
    compiled_providers = [
        _compile_provider(spec, variants, scope, page_size=_page_size(source), requirements=requirements)
        for spec in provider_specs
    ]
    _validate_terms_approvals(compiled_providers, requirements)
    _validate_resolution(compiled_providers, requirements)
    venue_operations = _compile_venue_operations(
        scope,
        variants,
        compiled_providers,
        venue_specs or (),
        page_size=_page_size(source),
    )

    plan = {
        "schema_version": QUERY_PLAN_SCHEMA_VERSION,
        "plan_id": plan_id or "",
        "plan_hash": "",
        "status": "draft",
        "created_at": created_at,
        "research": source["research"],
        "scope": scope,
        "inclusion": source["inclusion"],
        "query_variants": variants,
        "providers": compiled_providers,
        "venue_operations": venue_operations,
        "filter": filter_config,
        "citation_snowball": source["citation_snowball"],
        "budgets": source["budgets"],
        "execution": requirements,
        "approval": None,
    }
    plan_hash = approved_content_hash(plan)
    plan["plan_id"] = plan_id or f"search-{plan_hash[:12]}"
    plan["plan_hash"] = plan_hash
    return plan


def _compile_venue_operations(
    scope: Mapping[str, Any],
    variants: list[Mapping[str, Any]],
    providers: Sequence[Mapping[str, Any]],
    venue_specs: Sequence[Mapping[str, Any]],
    *,
    page_size: int,
) -> list[dict[str, Any]]:
    """Freeze descriptor and fallback decisions into the approved plan."""
    requested = tuple(str(venue_id) for venue_id in scope.get("venues", ()))
    if len(requested) != len(set(requested)):
        raise QueryPlanError("QueryPlan venue scope contains duplicates")
    specifications: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for item in venue_specs:
        descriptor = deepcopy(dict(item["descriptor"]))
        acceptance = deepcopy(dict(item["acceptance"]))
        venue_id = str(descriptor["venue_id"])
        if venue_id in specifications:
            raise QueryPlanError(f"venue specification repeats {venue_id}")
        if acceptance.get("venue_id") != venue_id:
            raise QueryPlanError(f"venue {venue_id} acceptance manifest does not match")
        specifications[venue_id] = (descriptor, acceptance)
    if set(specifications) != set(requested):
        missing = sorted(set(requested) - set(specifications))
        extra = sorted(set(specifications) - set(requested))
        detail = ", ".join((*[f"missing {name}" for name in missing], *[f"extra {name}" for name in extra]))
        raise QueryPlanError(f"venue specifications do not match scope: {detail}")

    provider_map = {str(provider["provider"]): provider for provider in providers}
    operations: list[dict[str, Any]] = []
    for venue_id in sorted(requested):
        descriptor, acceptance = specifications[venue_id]
        primary = str(descriptor["primary_provider"])
        if acceptance.get("primary_provider") != primary:
            raise QueryPlanError(f"venue {venue_id} primary provider does not match acceptance")
        if primary not in provider_map:
            raise QueryPlanError(f"venue {venue_id} primary provider is absent from QueryPlan")
        fallbacks = []
        venue_scope = {**scope, "venues": [venue_id]}
        for order, fallback in enumerate(acceptance.get("fallbacks", ()), start=1):
            provider_name = str(fallback["provider"])
            role = str(fallback["role"])
            provider = provider_map.get(provider_name)
            if provider is None:
                raise QueryPlanError(
                    f"venue {venue_id} fallback provider {provider_name} is absent from QueryPlan"
                )
            if role not in provider["roles"]:
                raise QueryPlanError(
                    f"venue {venue_id} fallback {provider_name} does not declare role {role}"
                )
            hashes = (
                [
                    query.query_hash
                    for query in compile_queries(
                        provider_name, variants, venue_scope, page_size=page_size
                    )
                ]
                if role == "search"
                else []
            )
            fallbacks.append(
                {
                    "order": order,
                    "provider": provider_name,
                    "role": role,
                    "native_query_hashes": hashes,
                }
            )
        operations.append(
            {
                "venue_id": venue_id,
                "name": str(descriptor["name"]),
                "venue_type": str(descriptor["venue_type"]),
                "descriptor": {
                    "schema_version": str(descriptor["schema_version"]),
                    "provider": primary,
                    "adapter": primary,
                    "parameters": deepcopy(dict(descriptor["provider_params"])),
                },
                "descriptor_hash": content_hash(descriptor),
                "acceptance_schema_version": str(acceptance["schema_version"]),
                "acceptance_manifest_hash": content_hash(acceptance),
                "fallbacks": fallbacks,
            }
        )
    return operations


def approve_query_plan(
    plan: Mapping[str, Any],
    expected_hash: str,
    *,
    approved_by: str,
    approved_at: str,
) -> dict[str, Any]:
    _require_schema_version(plan)
    assert_screening_scope_hash(plan)
    return approve(
        plan,
        expected_hash,
        approved_by=approved_by,
        approved_at=approved_at,
        hash_field="plan_hash",
    )


def runtime_requirements(plan: Mapping[str, Any]) -> dict[str, Any]:
    return _requirements(plan)


def assert_screening_scope_hash(plan: Mapping[str, Any]) -> str:
    """Require the frozen filter binding to match the plan's screening scope."""
    try:
        actual = plan["filter"]["screening_scope_hash"]
        expected = screening_scope_hash(plan)
    except (KeyError, TypeError) as error:
        raise QueryPlanError("QueryPlan screening scope hash is missing") from error
    if not _is_sha256(actual):
        raise QueryPlanError("QueryPlan screening scope hash must be a lowercase SHA-256")
    if actual != expected:
        raise QueryPlanError(
            "QueryPlan screening scope hash does not match research/inclusion/scope"
        )
    return actual


def compile_runtime_providers(
    plan: Mapping[str, Any], provider_specs: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], ...]:
    """Resolve current provider facts into the same frozen shape as a QueryPlan."""
    _require_schema_version(plan)
    requirements = runtime_requirements(plan)
    providers = tuple(
        _compile_provider(
            specification,
            list(plan["query_variants"]),
            plan["scope"],
            page_size=_page_size(plan),
            requirements=requirements,
        )
        for specification in provider_specs
    )
    _validate_terms_approvals(providers, requirements)
    _validate_resolution(providers, requirements)
    return providers


def assert_runtime_matches(
    plan: Mapping[str, Any],
    runtime_providers: Sequence[Mapping[str, Any]],
    *,
    budgets: Mapping[str, Any] | None = None,
    policies: Mapping[str, Any] | None = None,
    include_arxiv_candidates: bool | None = None,
) -> None:
    """Reject every environment change that would alter replayed searches."""
    try:
        _require_schema_version(plan)
    except QueryPlanError as error:
        raise QueryPlanDriftError(str(error)) from error
    try:
        require_valid_approval(plan, "plan_hash")
    except ApprovalError as error:
        raise QueryPlanDriftError(str(error)) from error
    try:
        assert_screening_scope_hash(plan)
    except QueryPlanError as error:
        raise QueryPlanDriftError(str(error)) from error
    frozen_arxiv_setting = plan.get("scope", {}).get("include_arxiv_candidates")
    if not isinstance(frozen_arxiv_setting, bool):
        raise QueryPlanDriftError("include_arxiv_candidates is not frozen in the QueryPlan")
    if include_arxiv_candidates is not None and include_arxiv_candidates != frozen_arxiv_setting:
        raise QueryPlanDriftError("include_arxiv_candidates has drifted")
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
        _require_schema_version(plan)
        try:
            require_valid_approval(plan, "plan_hash")
        except ApprovalError as error:
            raise QueryPlanError(str(error)) from error
        assert_screening_scope_hash(plan)
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
        plan = json.loads(self.approved_path(plan_id).read_text(encoding="utf-8"))
        _require_schema_version(plan)
        try:
            require_valid_approval(plan, "plan_hash")
        except ApprovalError as error:
            raise QueryPlanError(str(error)) from error
        assert_screening_scope_hash(plan)
        return plan

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
    credential_environment_variables = sorted(
        str(name) for name in specification.get("credential_environment_variables", ())
    )
    credential_availability = {
        name: bool(specification.get("credential_availability", {}).get(name, credentials_present))
        for name in credential_environment_variables
    }
    rate_limit = dict(
        specification.get(
            "rate_limit",
            {"global_qps": 1, "max_concurrency": 1, "cache_ttl_seconds": 0},
        )
    )
    return {
        "provider": provider,
        "distribution": str(specification["distribution"]),
        "version": str(specification["version"]),
        "entry_point": str(
            specification.get("entry_point", f"{specification['distribution']}:{provider}")
        ),
        "artifact_sha256": specification["artifact_sha256"],
        "manifest_hash": str(specification["manifest_hash"]),
        "roles": roles,
        "capabilities": sorted(str(capability) for capability in specification["capabilities"]),
        "enabled": enabled,
        "required": provider in requirements["required_providers"],
        "authority": str(specification.get("authority", "scholarly_graph")),
        "credential_environment_variables": credential_environment_variables,
        "credential_availability": credential_availability,
        "credentials_required": credentials_required,
        "rate_limit": {
            "global_qps": float(rate_limit["global_qps"]),
            "max_concurrency": int(rate_limit["max_concurrency"]),
            "cache_ttl_seconds": int(rate_limit["cache_ttl_seconds"]),
        },
        "data_use": str(specification.get("data_use", "permitted")),
        "terms_url": specification.get("terms_url"),
        "independence_group": str(specification.get("independence_group", provider)),
        "upstream_families": sorted(
            str(value) for value in specification.get("upstream_families", (provider,))
        ),
        "upstream_policies": deepcopy(dict(specification.get("upstream_policies", {}))),
        "resolved": resolved,
        "resolution_reason": reason,
        "mode": mode,
        "snapshot_hash": snapshot_hash,
        "credentials_present": credentials_present,
        "query_compiler_version": COMPILER_VERSION,
        "native_query_hashes": [query.query_hash for query in queries],
    }


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_schema_version(plan: Mapping[str, Any]) -> None:
    if plan.get("schema_version") != QUERY_PLAN_SCHEMA_VERSION:
        raise QueryPlanError(
            "QueryPlan schema_version is unsupported; recompile the plan as version "
            f"{QUERY_PLAN_SCHEMA_VERSION}"
        )


def _requirements(document: Mapping[str, Any]) -> dict[str, Any]:
    execution = document.get("execution", document)
    approvals = execution.get("terms_approvals", ())
    if not isinstance(approvals, (list, tuple)) or any(not isinstance(item, Mapping) for item in approvals):
        raise QueryPlanError("terms_approvals must be provider/terms_url objects")
    normalized_approvals = sorted(
        (
            {"provider": str(item.get("provider") or ""), "terms_url": str(item.get("terms_url") or "")}
            for item in approvals
        ),
        key=lambda item: (item["provider"], item["terms_url"]),
    )
    if any(not item["provider"] or not item["terms_url"] for item in normalized_approvals):
        raise QueryPlanError("terms approvals require provider and terms_url")
    if len({item["provider"] for item in normalized_approvals}) != len(normalized_approvals):
        raise QueryPlanError("terms approvals must name each provider at most once")
    return {
        "provider_policy": str(execution.get("provider_policy", "all_resolved")),
        "required_roles": sorted(str(role) for role in execution.get("required_roles", ("search",))),
        "required_providers": sorted(str(provider) for provider in execution.get("required_providers", ())),
        "terms_approvals": normalized_approvals,
    }


def _validate_terms_approvals(
    providers: Sequence[Mapping[str, Any]], requirements: Mapping[str, Any]
) -> None:
    available = {str(provider["provider"]): provider for provider in providers}
    for approval in requirements["terms_approvals"]:
        name = str(approval["provider"])
        root_name = name.split(":", 1)[0]
        if root_name not in available:
            raise QueryPlanError(f"terms approval names unavailable provider {name}")
        root = available[root_name]
        if name == root_name:
            expected_url = root.get("terms_url")
        else:
            upstream = root.get("upstream_policies", {}).get(name.split(":", 1)[1], {})
            expected_url = upstream.get("terms", {}).get("url") if isinstance(upstream, Mapping) else None
        if approval["terms_url"] != expected_url:
            raise QueryPlanError(f"terms approval URL for {name} does not match the frozen manifest")


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
