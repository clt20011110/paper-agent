"""Paper Agent command-line entry point."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import __version__
from .canonical import canonical_json, content_hash
from .citations import CitationRequest, SelectedSeed, schedule_requests
from .config import ConfigError, load_config, load_yaml
from .domain import CitationEdgeType
from .manifests import load_catalog
from .query_plan import QueryPlanStore, assert_runtime_matches, compile_query_plan
from .schema import schema_directory
from .search_execution import execute_search_plan, resolve_runtime_providers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="paper-agent")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--dry-run", action="store_true")
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("doctor", help="inspect the local runtime")

    search = subcommands.add_parser("search", help="compile, approve, and replay frozen searches")
    search_commands = search.add_subparsers(dest="search_command", required=True)
    plan = search_commands.add_parser("plan", help="compile a QueryPlan draft from YAML")
    plan.add_argument("--input", required=True, type=Path)
    plan.add_argument("--output-root", required=True, type=Path)
    approve = search_commands.add_parser("approve", help="approve a draft QueryPlan by content hash")
    approve.add_argument("--plan", required=True, type=Path)
    approve.add_argument("--hash", required=True)
    approve.add_argument("--approved-by", required=True)
    approve.add_argument("--approved-at")
    run = search_commands.add_parser("run", help="execute an approved frozen search")
    run.add_argument("--plan", required=True, type=Path)
    run.add_argument("--database", type=Path)
    run.add_argument("--contact")
    run.add_argument("--snapshot", action="append", default=[], metavar="PROVIDER=PATH")
    expand = search_commands.add_parser("expand-citations", help="plan a deterministic citation round")
    expand.add_argument("--plan", required=True, type=Path)
    expand.add_argument("--seeds", required=True, type=Path)
    expand.add_argument("--round-index", required=True, type=int)

    crawl = subcommands.add_parser("crawl", help="compatibility alias for venue descriptor discovery")
    crawl.add_argument("--venue", action="append", required=True)
    return parser


def doctor() -> dict[str, object]:
    schemas = sorted(schema_directory().glob("*.schema.json"))
    return {
        "paper_agent_version": __version__,
        "python": sys.version.split()[0],
        "python_supported": (3, 11) <= sys.version_info[:2] <= (3, 13),
        "schema_count": len(schemas),
        "codex_cli": shutil.which("codex"),
        "omlx_cli": shutil.which("omlx"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        _emit(doctor())
        return 0
    if args.command == "search" and args.search_command == "plan":
        _emit(_search_plan(args.input, args.output_root))
        return 0
    if args.command == "search" and args.search_command == "approve":
        _emit(_search_approve(args.plan, args.hash, args.approved_by, args.approved_at))
        return 0
    if args.command == "search" and args.search_command == "run":
        _emit(
            _search_run(
                args.plan,
                database_path=args.database,
                contact=args.contact,
                snapshot_values=args.snapshot,
                config_path=args.config,
                run_id=args.run_id,
                dry_run=args.dry_run,
            )
        )
        return 0
    if args.command == "search" and args.search_command == "expand-citations":
        _emit(_expand_citations(args.plan, args.seeds, args.round_index))
        return 0
    if args.command == "crawl":
        _emit(_crawl(args.venue))
        return 0
    raise AssertionError(args.command)


def _search_plan(input_path: Path, output_root: Path) -> dict[str, Any]:
    draft = load_yaml(input_path)
    providers = _provider_specs(
        draft.pop("providers"),
        input_path.parent,
        venue_ids=draft["scope"]["venues"],
    )
    plan = compile_query_plan(draft, providers=providers)
    path = QueryPlanStore(output_root).save_draft(plan)
    return {
        "command": "search.plan",
        "draft_path": str(path),
        "estimated_max_candidates": plan["budgets"]["max_candidates"],
        "estimated_max_requests": plan["budgets"]["max_requests"],
        "estimated_max_seconds": plan["budgets"]["max_seconds"],
        "plan_hash": plan["plan_hash"],
        "plan_id": plan["plan_id"],
        "status": plan["status"],
    }


def _search_approve(plan_path: Path, expected_hash: str, approved_by: str, approved_at: str | None) -> dict[str, Any]:
    plan = _load_json(plan_path)
    approved = QueryPlanStore(_store_root(plan_path)).approve_and_save(
        plan,
        expected_hash,
        approved_by=approved_by,
        approved_at=approved_at or datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    )
    return {
        "approved_path": str(QueryPlanStore(_store_root(plan_path)).approved_path(str(approved["plan_id"]))),
        "command": "search.approve",
        "latest_path": str(QueryPlanStore(_store_root(plan_path)).latest_path),
        "plan_hash": approved["plan_hash"],
        "plan_id": approved["plan_id"],
        "status": approved["status"],
    }


def _search_run(
    plan_path: Path,
    *,
    database_path: Path | None,
    contact: str | None,
    snapshot_values: Sequence[str],
    config_path: Path | None,
    run_id: str | None,
    dry_run: bool,
) -> dict[str, Any]:
    plan = _load_json(plan_path)
    config = load_config(config_path) if config_path else None
    if config is not None and config_path is not None:
        _assert_config_plan(config, config_path, plan_path, plan, require_hash=not dry_run)
    database = database_path or _configured_database(config, config_path) or (_store_root(plan_path) / "paper-agent.sqlite3")
    snapshots = _snapshot_paths(snapshot_values)
    operator_contact = (
        contact
        or os.environ.get("PAPER_AGENT_CONTACT")
        or os.environ.get("PAPER_AGENT_CONTACT_EMAIL")
    )
    if dry_run:
        runtime = resolve_runtime_providers(plan, snapshot_paths=snapshots)
        return {
            "command": "search.run",
            "database_path": str(database),
            "plan_hash": plan["plan_hash"],
            "plan_id": plan["plan_id"],
            "provider_invocation": "skipped_dry_run",
            "resolved_providers": sorted(
                provider["provider"] for provider in runtime if provider["resolved"]
            ),
            "status": "runtime_validated",
        }
    result, resolved_run_id, crawl_run_id = execute_search_plan(
        plan,
        database,
        run_id=run_id,
        contact=operator_contact,
        snapshot_paths=snapshots,
    )
    return {
        "command": "search.run",
        "crawl_run_id": crawl_run_id,
        "database_path": str(database),
        "plan_hash": plan["plan_hash"],
        "plan_id": plan["plan_id"],
        "provider_invocation": "completed",
        "provider_outcomes": {
            outcome.provider: outcome.status for outcome in result.fanout.outcomes
        },
        "paper_count": len(result.paper_ids),
        "arxiv_candidate_count": len(result.arxiv_candidate_ids),
        "run_id": resolved_run_id,
        "status": result.status,
    }


def _expand_citations(plan_path: Path, seeds_path: Path, round_index: int) -> dict[str, Any]:
    plan = _load_json(plan_path)
    runtime = {"providers": plan["providers"], "budgets": plan["budgets"], "execution": plan["execution"]}
    assert_runtime_matches(plan, runtime["providers"], budgets=runtime["budgets"], policies=runtime["execution"])
    seeds = _selected_seeds(_load_json(seeds_path))
    snowball = plan["citation_snowball"]
    requests = schedule_requests(
        seeds,
        providers=[
            provider["provider"]
            for provider in plan["providers"]
            if provider["resolved"] and "citation" in provider["roles"]
        ],
        directions=[CitationEdgeType(direction) for direction in snowball["directions"]],
        max_requests=int(plan["budgets"]["max_requests"]),
        max_candidates_per_request=int(snowball["max_per_seed_per_source"]),
    )
    manifest = _citation_manifest(plan, seeds, requests, round_index)
    return {"command": "search.expand-citations", **manifest}


def _crawl(venue_ids: Sequence[str]) -> dict[str, Any]:
    catalog = load_catalog()
    venues = [catalog.venue(venue_id) for venue_id in sorted(set(venue_ids))]
    return {
        "command": "crawl",
        "mode": "venue_descriptor_compatibility",
        "search_audit_intent": {
            "event": "venue_descriptor_discovery_planned",
            "providers": sorted({venue["primary_provider"] for venue in venues}),
            "venue_ids": [venue["venue_id"] for venue in venues],
        },
    }


def _provider_specs(value: Any, root: Path, *, venue_ids: Sequence[str]) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("providers must be a list")
    catalog = load_catalog(root if (root / "providers").exists() else None)
    requested_by_provider = {
        str(item if isinstance(item, str) else item["provider"]): (
            {"provider": item} if isinstance(item, str) else dict(item)
        )
        for item in value
    }
    exact_providers = {
        catalog.venue(Path(str(venue_id)).stem)["primary_provider"]
        for venue_id in venue_ids
    }
    for provider in exact_providers:
        requested_by_provider.setdefault(provider, {"provider": provider})

    specs: list[dict[str, Any]] = []
    for provider_name in sorted(requested_by_provider):
        requested = requested_by_provider[provider_name]
        manifest = catalog.provider(str(requested["provider"]))
        authentication = manifest["authentication"]
        credential_names = _credential_environment_variables(authentication)
        credentials_present = not credential_names or all(
            bool(os.environ.get(name)) for name in credential_names
        )
        credential_availability = {
            name: bool(os.environ.get(name)) for name in credential_names
        }
        spec = {
            "provider": manifest["provider"],
            "distribution": manifest["distribution"],
            "version": manifest["version"],
            "entry_point": manifest["entry_point"],
            "artifact_sha256": manifest["artifact_sha256"],
            "manifest_hash": content_hash(manifest),
            "roles": manifest["roles"],
            "capabilities": manifest["capabilities"],
            "enabled": manifest["enabled"],
            "authority": manifest["authority"],
            "credential_environment_variables": credential_names,
            "credential_availability": credential_availability,
            "rate_limit": manifest["rate_limit"],
            "data_use": manifest["terms"]["data_use"],
            "terms_url": manifest["terms"].get("url"),
            "independence_group": manifest["independence_group"],
            "upstream_families": manifest["upstream_families"],
            "mode": "api",
            "credentials_required": authentication["required"],
            "credentials_present": credentials_present,
            "manifest_trusted": manifest["builtin"],
            "exact_required": provider_name in exact_providers,
        }
        spec.update(requested)
        specs.append(spec)
    return specs


def _credential_environment_variables(authentication: Mapping[str, Any]) -> tuple[str, ...]:
    declared = authentication.get("credential_envs", ())
    names = declared.values() if isinstance(declared, Mapping) else declared
    if "credential_env" in authentication:
        names = (*names, authentication["credential_env"])
    return tuple(sorted(str(name) for name in names))


def _configured_database(config: Mapping[str, Any] | None, config_path: Path | None) -> Path | None:
    if config is None or config_path is None:
        return None
    path = Path(str(config["storage"]["sqlite_path"]))
    return path if path.is_absolute() else config_path.parent / path


def _assert_config_plan(
    config: Mapping[str, Any],
    config_path: Path,
    plan_path: Path,
    plan: Mapping[str, Any],
    *,
    require_hash: bool,
) -> None:
    approved = config["sources"]["approved_plan"]
    configured_path = Path(str(approved["input_path"]))
    configured_path = configured_path if configured_path.is_absolute() else config_path.parent / configured_path
    if configured_path.resolve() != plan_path.resolve():
        raise ConfigError("configured approved QueryPlan path does not match --plan")
    if require_hash and approved["content_hash"] is None:
        raise ConfigError("search execution requires an approved QueryPlan hash in config")
    if approved["content_hash"] is not None and approved["content_hash"] != plan["plan_hash"]:
        raise ConfigError("configured approved QueryPlan hash does not match --plan")


def _snapshot_paths(values: Sequence[str]) -> dict[str, Path]:
    snapshots = {}
    for value in values:
        provider, separator, path = value.partition("=")
        if not separator or not provider or not path:
            raise ValueError("--snapshot must be PROVIDER=PATH")
        snapshots[provider] = Path(path)
    return snapshots


def _selected_seeds(payload: Any) -> tuple[SelectedSeed, ...]:
    values = payload["seeds"] if isinstance(payload, dict) else payload
    return tuple(
        SelectedSeed(
            paper_id=str(seed["paper_id"]),
            seed_reason=str(seed["seed_reason"]),
            parent_round=int(seed["parent_round"]),
            depth=int(seed["depth"]),
            subquestion_id=seed.get("subquestion_id"),
            rank=int(seed["rank"]),
            selector_version=str(seed["selector_version"]),
            selector_config_hash=str(seed["selector_config_hash"]),
        )
        for seed in values
    )


def _citation_manifest(
    plan: Mapping[str, Any],
    seeds: Sequence[SelectedSeed],
    requests: Sequence[CitationRequest],
    round_index: int,
) -> dict[str, Any]:
    serialized_seeds = [
        {
            "paper_id": seed.paper_id,
            "seed_reason": seed.seed_reason,
            "parent_round": seed.parent_round,
            "depth": seed.depth,
            "subquestion_id": seed.subquestion_id,
            "rank": seed.rank,
            "selector_version": seed.selector_version,
            "selector_config_hash": seed.selector_config_hash,
        }
        for seed in seeds
    ]
    serialized_requests = [
        {
            "provider": request.provider,
            "direction": request.direction.value,
            "seed_paper_id": request.seed_paper_id,
            "depth": request.depth,
            "seed_rank": request.seed_rank,
            "schedule_order": request.schedule_order,
            "max_candidates": request.max_candidates,
        }
        for request in requests
    ]
    return {
        "plan_hash": plan["plan_hash"],
        "plan_id": plan["plan_id"],
        "request_schedule_hash": content_hash(serialized_requests),
        "requests": serialized_requests,
        "round_index": round_index,
        "seed_manifest_hash": content_hash(serialized_seeds),
        "seeds": serialized_seeds,
    }


def _store_root(plan_path: Path) -> Path:
    return plan_path.parent.parent.parent


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _emit(payload: Mapping[str, Any]) -> None:
    print(canonical_json(dict(payload)).decode("utf-8"))
