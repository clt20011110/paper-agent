"""Runtime resolution and execution for approved search plans."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from hashlib import sha256
import os
from pathlib import Path
import re
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from .canonical import content_hash
from .http_transport import ControlledHTTPTransport
from .manifests import ManifestCatalog, load_catalog
from .approved_snapshot import (
    MetadataSnapshotBundle,
    MetadataSnapshotError,
    MetadataSnapshotTransport,
    ProviderTransportRouter,
)
from .provider_runtime import ProviderRuntime, policy_from_manifest
from .providers.api import CrawlWindow, SeedInput
from .providers.builtin import create_builtin, manifest_from_document
from .query_plan import (
    QueryPlanDriftError,
    QueryPlanError,
    assert_runtime_matches,
    compile_runtime_providers,
)
from .search_pipeline import PipelineResult, SearchPipeline, VenueRun
from .storage import Database
from .verification import ProviderTrust, VenueContext


def resolve_runtime_providers(
    plan: Mapping[str, Any],
    *,
    catalog: ManifestCatalog | None = None,
    environment: Mapping[str, str] | None = None,
    snapshot_paths: Mapping[str, Path] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Re-read installed manifests, credential presence, and approved snapshots."""
    installed = catalog or load_catalog()
    values = environment if environment is not None else os.environ
    snapshots = snapshot_paths or {}
    approved_snapshot_providers = {
        str(provider["provider"])
        for provider in plan["providers"]
        if provider["mode"] in {"snapshot", "bulk_snapshot"}
    }
    if set(snapshots) - approved_snapshot_providers:
        raise QueryPlanDriftError("an unapproved snapshot path was supplied")
    specifications = []
    for expected in plan["providers"]:
        name = str(expected["provider"])
        try:
            manifest = installed.provider(name)
        except KeyError as error:
            raise QueryPlanDriftError(f"provider {name} is no longer installed") from error
        credential_names = _credential_environment_variables(manifest["authentication"])
        availability = {credential: bool(values.get(credential)) for credential in credential_names}
        mode = str(expected["mode"])
        snapshot_hash = _snapshot_hash(name, mode, snapshots)
        specifications.append(
            {
                "provider": name,
                "distribution": manifest["distribution"],
                "version": manifest["version"],
                "entry_point": manifest["entry_point"],
                "artifact_sha256": manifest["artifact_sha256"],
                "manifest_hash": content_hash(manifest),
                "roles": expected["roles"],
                "capabilities": expected["capabilities"],
                "enabled": bool(expected["enabled"]) and bool(manifest["enabled"]),
                "authority": manifest["authority"],
                "credential_environment_variables": credential_names,
                "credential_availability": availability,
                "credentials_required": manifest["authentication"]["required"],
                "credentials_present": not credential_names or all(availability.values()),
                "rate_limit": manifest["rate_limit"],
                "data_use": manifest["terms"]["data_use"],
                "terms_url": manifest["terms"].get("url"),
                "independence_group": manifest["independence_group"],
                "upstream_families": manifest["upstream_families"],
                "mode": mode,
                "snapshot_hash": snapshot_hash,
                "manifest_trusted": manifest["builtin"],
                "exact_required": bool(expected["required"]),
            }
        )
    try:
        runtime = compile_runtime_providers(plan, specifications)
    except QueryPlanError as error:
        raise QueryPlanDriftError(str(error)) from error
    assert_runtime_matches(plan, runtime, budgets=plan["budgets"], policies=plan["execution"])
    return runtime


def execute_search_plan(
    plan: Mapping[str, Any],
    database_path: Path,
    *,
    run_id: str | None = None,
    contact: str | None = None,
    catalog: ManifestCatalog | None = None,
    environment: Mapping[str, str] | None = None,
    snapshot_paths: Mapping[str, Path] | None = None,
    transport: Any | None = None,
    venue_only: bool = False,
    historical_replay: bool = False,
) -> tuple[PipelineResult, str, str]:
    """Execute one approved plan through the same single-writer pipeline as tests."""
    installed = catalog or load_catalog()
    runtime = resolve_runtime_providers(
        plan,
        catalog=installed,
        environment=environment,
        snapshot_paths=snapshot_paths,
    )
    active = tuple(provider for provider in runtime if provider["resolved"])
    if any(provider["mode"] == "bulk_snapshot" for provider in active):
        raise QueryPlanDriftError("bulk snapshots need a provider-specific reader")

    policies = {
        str(provider["provider"]): policy_from_manifest(
            manifest_from_document(installed.provider(str(provider["provider"]))),
            terms_accepted=provider["data_use"] == "permitted",
            robots_allowed=True,
        )
        for provider in active
    }
    provider_runtime = ProviderRuntime(policies)

    if transport is None:
        operator_contact = contact or ""
        needs_http = any(provider["mode"] == "api" and provider["provider"] != "user_library" for provider in active)
        if not operator_contact and needs_http:
            raise ValueError("search execution requires --contact or PAPER_AGENT_CONTACT")
        transport = (
            ControlledHTTPTransport(operator_contact, runtime=provider_runtime, environment=environment)
            if needs_http
            else _no_network_transport
        )

    snapshot_transports = {}
    paths = snapshot_paths or {}
    for provider in active:
        if provider["mode"] != "snapshot":
            continue
        name = str(provider["provider"])
        try:
            bundle = MetadataSnapshotBundle.load(paths[name], str(provider["snapshot_hash"]))
        except KeyError as error:
            raise QueryPlanDriftError(f"snapshot provider {name} has no approved bundle path") from error
        except MetadataSnapshotError as error:
            raise QueryPlanDriftError(f"snapshot provider {name}: {error}") from error
        if bundle.provider != name:
            raise QueryPlanDriftError(f"snapshot bundle provider {bundle.provider} does not match {name}")
        snapshot_transports[name] = MetadataSnapshotTransport(bundle, provider_runtime, environment)
    if snapshot_transports:
        transport = ProviderTransportRouter(snapshot_transports, transport)

    clients = {
        str(provider["provider"]): create_builtin(
            str(provider["provider"]),
            transport,
            manifest_from_document(installed.provider(str(provider["provider"]))),
        )
        for provider in active
    }
    trusts = {
        name: ProviderTrust.from_manifest(installed.provider(name))
        for name in clients
    }
    venues = _venue_runs(plan, installed, historical_replay=historical_replay)
    seeds = () if venue_only else tuple(seed_input(value) for value in plan["scope"].get("user_seeds", ()))
    citation_clients = {
        name: client
        for name, client in clients.items()
        if not venue_only
        and "citation" in next(provider["roles"] for provider in active if provider["provider"] == name)
    }
    stage_name = "crawl" if venue_only else "search"
    resolved_run_id = run_id or f"{stage_name}-run-{plan['plan_id']}"
    crawl_identity = f"{resolved_run_id}:{plan['plan_hash']}:{stage_name}"
    crawl_run_id = f"crawl-{uuid5(NAMESPACE_URL, crawl_identity).hex}"

    with Database(database_path) as database:
        database.migrate()
        row = database.connection.execute(
            "SELECT started_at FROM pipeline_runs WHERE run_id = ?", (resolved_run_id,)
        ).fetchone()
        observed_at = row["started_at"] if row else datetime.now(UTC).isoformat().replace("+00:00", "Z")
        pipeline = SearchPipeline(
            database,
            plan,
            runtime_providers=runtime,
            clients=clients,
            trusts=trusts,
            venue_runs=venues,
            seed_inputs=seeds,
            citation_clients=citation_clients,
            venue_only=venue_only,
        )
        result = pipeline.run(
            run_id=resolved_run_id,
            crawl_run_id=crawl_run_id,
            observed_at=observed_at,
        )
    return result, resolved_run_id, crawl_run_id


def seed_input(value: str) -> SeedInput:
    """Infer one authorized library seed without treating notes as metadata."""
    prefixes = {"doi", "arxiv", "arxiv_id", "url", "bibtex", "ris", "csl-json", "csl_json", "local_pdf", "pdf"}
    prefix, separator, payload = value.partition(":")
    if separator and prefix in prefixes:
        return SeedInput(prefix, payload)
    if value.startswith(("https://", "http://")):
        return SeedInput("url", value)
    if value.lstrip().startswith("@"):
        return SeedInput("bibtex", value)
    if value.startswith("TY  - "):
        return SeedInput("ris", value)
    if value.lstrip().startswith(("{", "[")):
        return SeedInput("csl-json", value)
    if Path(value).suffix.casefold() == ".pdf":
        return SeedInput("local_pdf", value)
    if re.fullmatch(r"10\.\d{4,9}/\S+", value, re.I):
        return SeedInput("doi", value)
    if re.fullmatch(r"(?:\d{4}\.\d{4,5}|[a-z-]+/\d{7})(?:v\d+)?", value, re.I):
        return SeedInput("arxiv", value)
    raise ValueError(f"seed needs an explicit supported kind: {value}")


def _venue_runs(
    plan: Mapping[str, Any], catalog: ManifestCatalog, *, historical_replay: bool = False
) -> tuple[VenueRun, ...]:
    start = str(plan["scope"]["date_from"])
    end = str(plan["scope"]["date_to"])
    year = int(start[:4]) if start[:4] == end[:4] else None
    runs = []
    for venue_id in plan["scope"].get("venues", ()):
        venue = catalog.venue(str(venue_id))
        runs.append(
            VenueRun(
                catalog.runtime_venue(str(venue_id)),
                CrawlWindow(date_from=start, date_to=end, year=year),
                VenueContext(
                    f"{venue_id}:{start}:{end}",
                    str(venue_id),
                    str(venue["name"]),
                    str(venue["venue_type"]),
                    str(venue["primary_provider"]),
                    venue,
                ),
                historical_replay=historical_replay,
            )
        )
    return tuple(runs)


def _credential_environment_variables(authentication: Mapping[str, Any]) -> tuple[str, ...]:
    declared = authentication.get("credential_envs", {})
    names = tuple(declared.values()) if isinstance(declared, Mapping) else tuple(declared)
    if authentication.get("credential_env"):
        names = (*names, authentication["credential_env"])
    return tuple(sorted(str(name) for name in names))


def _snapshot_hash(provider: str, mode: str, paths: Mapping[str, Path]) -> str | None:
    if mode == "api" or mode == "local":
        return None
    path = paths.get(provider)
    return sha256(path.read_bytes()).hexdigest() if path else None


def _no_network_transport(provider: str, operation: str, parameters: Mapping[str, Any]) -> Mapping[str, Any]:
    raise QueryPlanDriftError(f"{provider}:{operation} has no API transport in snapshot-only execution")
