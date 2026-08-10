"""Paper Agent command-line entry point."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

from . import __version__
from .authorized_skill_runtime import AuthorizedSkillRuntime, load_audit_record
from .canonical import canonical_json, content_hash
from .citations import CitationRequest, SelectedSeed, schedule_requests
from .config import ConfigError, load_config, load_yaml
from .doctor import DoctorPaths, SystemDoctor
from .domain import CitationEdgeType
from .exchange import export_csv, export_jsonl, validate_export
from .grants import (
    GrantStore,
    create_grant_draft,
    validate_grant_approval,
)
from .legacy import migrate_legacy_yaml, write_migrated
from .manifests import load_catalog
from .query_plan import (
    QueryPlanStore,
    approve_query_plan,
    assert_runtime_matches,
    compile_query_plan,
)
from .repository import PaperRepository
from .search_execution import execute_search_plan, resolve_runtime_providers, seed_input
from .stage2_search import Stage2ReleaseError, load_stage2_release
from .search_audit import search_audit
from .seed_import import import_seeds, inputs_from_files, validate_seed_inputs
from .storage import Database


class CliUsageError(ValueError):
    """A command-line usage error suitable for structured console output."""


class _StructuredArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise CliUsageError(message)


def build_parser(*, structured_errors: bool = False) -> argparse.ArgumentParser:
    parser_class = _StructuredArgumentParser if structured_errors else argparse.ArgumentParser
    parser = parser_class(prog="paper-agent")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--dry-run", action="store_true")
    subcommands = parser.add_subparsers(dest="command", required=True)
    doctor_command = subcommands.add_parser("doctor", help="inspect the local runtime")
    doctor_command.add_argument("--database", type=Path)
    doctor_command.add_argument("--model-lock", action="append", default=[], type=Path)
    doctor_command.add_argument("--query-plan", type=Path)
    doctor_command.add_argument("--stage2-release", type=Path)
    doctor_command.add_argument(
        "--authorized-skill-root", action="append", default=[], type=Path,
        help="root to inspect for the installed authorized download skill",
    )
    doctor_command.add_argument(
        "--authorized-skill-zip", type=Path,
        help="original audited authorized-skill archive",
    )
    doctor_command.add_argument(
        "--authorized-skill-audit", type=Path,
        help="override the checked-in authorized-skill audit manifest",
    )
    doctor_command.add_argument(
        "--prove-paid-models", "--prove-codex-models", dest="prove_codex_models", action="store_true",
        help="explicitly invoke the frozen Codex model profiles to prove availability",
    )
    doctor_command.add_argument(
        "--production-ready", action="store_true",
        help="also require production-readiness checks to pass",
    )

    grant = subcommands.add_parser("grant", help="create and administer immutable authorization grants")
    grant_commands = grant.add_subparsers(dest="grant_command", required=True)
    grant_create = grant_commands.add_parser("create", help="write an unapproved grant draft")
    grant_create.add_argument("--kind", required=True, choices=("download", "browser_data_sharing", "remote_model_processing"))
    grant_create.add_argument("--output", "--draft", dest="output", required=True, type=Path)
    grant_create.add_argument("--database", type=Path)
    grant_create.add_argument("--grant-id")
    grant_create.add_argument("--action", action="append", default=[])
    grant_create.add_argument("--purpose")
    grant_create.add_argument("--mode", choices=("attended", "unattended"))
    grant_create.add_argument("--allow-unattended", action="store_true", default=None)
    _add_grant_scope_arguments(grant_create)
    grant_create.add_argument("--max-papers", type=int)
    grant_create.add_argument("--expires-at")
    grant_create.add_argument("--skill-digest")
    grant_create.add_argument("--dependency-digest")
    grant_create.add_argument("--lineage-hash")
    grant_approve = grant_commands.add_parser("approve", help="approve and persist a draft grant")
    grant_approve.add_argument("--grant", required=True, type=Path)
    grant_approve.add_argument("--hash", required=True)
    grant_approve.add_argument("--approved-by", "--actor", dest="approved_by", required=True)
    grant_approve.add_argument("--approved-at", required=True)
    grant_approve.add_argument("--database", type=Path)
    grant_revoke = grant_commands.add_parser("revoke", help="append a grant revocation event")
    grant_revoke.add_argument("grant_id")
    grant_revoke.add_argument("--actor", required=True)
    grant_revoke.add_argument("--at", "--event-at", "--revoked-at", dest="event_at", required=True)
    grant_revoke.add_argument("--database", type=Path)

    export = subcommands.add_parser("export", help="export canonical papers from SQLite")
    export.add_argument("--database", type=Path)
    export.add_argument("--format", required=True, choices=("jsonl", "csv"))
    export.add_argument("--output", required=True, type=Path)

    migrate = subcommands.add_parser("migrate-config", help="convert a legacy YAML configuration to v2")
    migrate.add_argument("--input", required=True, type=Path)
    migrate.add_argument("--write", type=Path, help="write the converted v2 YAML after review")

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
    run.add_argument("--stage2-release", type=Path, help="passed local Stage 2 release manifest")
    run.add_argument("--historical-replay", action="store_true")
    audit = search_commands.add_parser("audit", help="read a persisted search audit")
    audit.add_argument("--database", required=True, type=Path)
    audit.add_argument("--crawl-run-id", required=True)
    expand = search_commands.add_parser("expand-citations", help="plan a deterministic citation round")
    expand.add_argument("--plan", required=True, type=Path)
    expand.add_argument("--seeds", required=True, type=Path)
    expand.add_argument("--round-index", required=True, type=int)

    crawl = subcommands.add_parser("crawl", help="compatibility alias for venue descriptor discovery")
    crawl.add_argument("--venue", action="append", required=True)
    crawl.add_argument("--plan", type=Path)
    crawl.add_argument("--database", type=Path)
    crawl.add_argument("--contact")
    crawl.add_argument("--snapshot", action="append", default=[], metavar="PROVIDER=PATH")
    crawl.add_argument("--stage2-release", type=Path, help="passed local Stage 2 release manifest")
    crawl.add_argument("--historical-replay", action="store_true")
    import_command = subcommands.add_parser("import-seeds", help="import authorized library seeds")
    import_command.add_argument("--database", required=True, type=Path)
    import_command.add_argument("--seed", action="append", default=[])
    import_command.add_argument("--input", action="append", default=[], type=Path)
    return parser


def main(
    argv: Sequence[str] | None = None, *, structured_errors: bool = False
) -> int:
    args = build_parser(structured_errors=structured_errors).parse_args(
        _runtime_argv(argv)
    )
    if args.command == "doctor":
        report = _doctor(args)
        exit_code = int(
            not (report.production_ready if args.production_ready else report.ready)
        )
        return _finish(
            args,
            {
                "command": "doctor",
                "paper_agent_version": __version__,
                "status": "ready" if exit_code == 0 else "blocked",
                **report.as_dict(),
            },
            exit_code,
        )
    if args.command == "grant" and args.grant_command == "create":
        return _finish(args, _grant_create(args))
    if args.command == "grant" and args.grant_command == "approve":
        return _finish(args, _grant_approve(args))
    if args.command == "grant" and args.grant_command == "revoke":
        return _finish(args, _grant_revoke(args))
    if args.command == "export":
        return _finish(args, _export(args))
    if args.command == "migrate-config":
        return _finish(args, _migrate_config(args))
    if args.command == "search" and args.search_command == "plan":
        return _finish(
            args,
            _search_plan(args.input, args.output_root, dry_run=args.dry_run),
        )
    if args.command == "search" and args.search_command == "approve":
        return _finish(
            args,
            _search_approve(
                args.plan,
                args.hash,
                args.approved_by,
                args.approved_at,
                dry_run=args.dry_run,
            ),
        )
    if args.command == "search" and args.search_command == "run":
        return _finish(
            args,
            _search_run(
                args.plan,
                database_path=args.database,
                contact=args.contact,
                snapshot_values=args.snapshot,
                stage2_release_path=args.stage2_release,
                config_path=args.config,
                run_id=args.run_id,
                dry_run=args.dry_run,
                historical_replay=args.historical_replay,
            )
        )
    if args.command == "search" and args.search_command == "audit":
        return _finish(
            args,
            {"command": "search.audit", **search_audit(args.database, args.crawl_run_id)},
        )
    if args.command == "search" and args.search_command == "expand-citations":
        return _finish(
            args, _expand_citations(args.plan, args.seeds, args.round_index)
        )
    if args.command == "crawl":
        return _finish(
            args,
            _crawl(
                args.venue,
                plan_path=args.plan,
                database_path=args.database,
                contact=args.contact,
                snapshot_values=args.snapshot,
                stage2_release_path=args.stage2_release,
                config_path=args.config,
                run_id=args.run_id,
                dry_run=args.dry_run,
                historical_replay=args.historical_replay,
            )
        )
    if args.command == "import-seeds":
        return _finish(
            args,
            _import_seeds(args.database, args.seed, args.input, args.run_id, args.dry_run),
        )
    raise AssertionError(args.command)


def entrypoint(argv: Sequence[str] | None = None) -> int:
    """Console boundary: emit one structured failure instead of a traceback."""
    try:
        return main(argv, structured_errors=True)
    except Exception as error:
        normalized = _runtime_argv(argv)
        command = _command_from_argv(normalized)
        _emit({
            "command": command,
            "error": str(error),
            "error_type": type(error).__name__,
            "event_code": f"{command}.failed",
            "run_id": _runtime_option(normalized, "--run-id"),
            "stage": _command_stage(command),
            "status": "failed",
        })
        return 1


def doctor() -> dict[str, object]:
    """Backward-compatible programmatic diagnostic entrypoint."""
    report = SystemDoctor().run()
    python_check = next(check for check in report.checks if check.name == "python")
    return {
        "paper_agent_version": __version__,
        "python_supported": python_check.status == "pass",
        **report.as_dict(),
    }


def _add_grant_scope_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--paper-id", action="append", default=[])
    parser.add_argument("--artifact-hash", action="append", default=[])
    parser.add_argument("--collection-id", action="append", default=[])
    parser.add_argument("--collection-snapshot-hash")
    parser.add_argument("--selection-snapshot-hash")
    parser.add_argument("--domain", action="append", default=[])
    parser.add_argument("--provider")
    parser.add_argument("--model")
    parser.add_argument("--data-category", action="append", default=[])


def _doctor(args: argparse.Namespace):
    try:
        config = load_config(args.config) if args.config else None
    except (OSError, ConfigError):
        # SystemDoctor owns the user-facing configuration diagnostic.
        config = None
    database = args.database or _configured_database(config, args.config)
    defaults = DoctorPaths.defaults()
    paths = DoctorPaths(
        repository_root=defaults.repository_root,
        config_path=args.config,
        database_path=database,
        model_lock_paths=tuple(args.model_lock) or defaults.model_lock_paths,
        stage2_release_path=args.stage2_release,
        query_plan_path=args.query_plan,
        authorized_skill_runtime=_authorized_skill_runtime(config, args),
    )
    return SystemDoctor(
        paths,
        http_probe=_local_http_probe,
        prove_codex_models=args.prove_codex_models,
    ).run()


def _authorized_skill_runtime(
    config: Mapping[str, Any] | None, args: argparse.Namespace
) -> AuthorizedSkillRuntime:
    enabled = False
    if config is not None:
        enabled = bool(config["download"]["authorized_skill"]["enabled"])
    roots = tuple(args.authorized_skill_root) or _default_authorized_skill_roots()
    archive_value = args.authorized_skill_zip or os.environ.get(
        "PAPER_AGENT_AUTHORIZED_SKILL_ZIP"
    )
    return AuthorizedSkillRuntime(
        enabled=enabled,
        skill_roots=roots,
        original_zip=Path(archive_value) if archive_value else None,
        audit_manifest=args.authorized_skill_audit,
    )


def _default_authorized_skill_roots() -> tuple[Path, ...]:
    configured = os.environ.get("PAPER_AGENT_AUTHORIZED_SKILL_ROOTS")
    if configured:
        return tuple(Path(item) for item in configured.split(os.pathsep) if item)
    codex_root = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
    return (codex_root / "skills",)


def _local_http_probe(url: str) -> tuple[int, str]:
    """Probe the already validated loopback oMLX endpoint with bounded I/O."""
    request = Request(url, headers={"Accept": "application/json"}, method="GET")
    opener = build_opener(ProxyHandler({}))
    try:
        with opener.open(request, timeout=3) as response:
            payload = response.read(2 * 1024 * 1024 + 1)
            status = int(response.status)
    except HTTPError as error:
        payload = error.read(2 * 1024 * 1024 + 1)
        status = int(error.code)
    if len(payload) > 2 * 1024 * 1024:
        raise OSError("local model inventory response exceeds 2 MiB")
    return status, payload.decode("utf-8", errors="replace")


def _grant_create(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config) if args.config else None
    defaults = _grant_defaults(config, args)
    scope = _grant_scope(args, defaults)
    actions = args.action or list(defaults.get("actions", (args.kind,)))
    purpose = args.purpose or str(defaults.get("purpose", ""))
    mode = args.mode or str(defaults.get("mode", "attended"))
    allow_unattended = args.allow_unattended
    if allow_unattended is None:
        allow_unattended = bool(defaults.get("allow_unattended", False))
    max_papers = args.max_papers if args.max_papers is not None else defaults.get("max_papers")
    expires_at = args.expires_at or defaults.get("authorization_expires_at")
    if not purpose or max_papers is None or expires_at is None:
        raise ValueError("grant create requires --purpose, --max-papers, and --expires-at (or download grant_defaults)")
    draft = create_grant_draft(
        grant_id=args.grant_id,
        kind=args.kind,
        actions=list(actions),
        purpose=purpose,
        mode=mode,
        allow_unattended=allow_unattended,
        scope=scope,
        max_papers=max_papers,
        expires_at=expires_at,
        skill_digest=args.skill_digest or defaults.get("installed_content_sha256"),
        dependency_digest=args.dependency_digest or defaults.get("dependency_lock_sha256"),
        lineage_hash=args.lineage_hash,
    )
    if not args.dry_run:
        if args.output.exists():
            raise FileExistsError(f"grant draft already exists: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(canonical_json(draft))
    return {
        "command": "grant.create",
        "content_hash": draft["content_hash"],
        "draft_path": str(args.output),
        "grant_id": draft["grant_id"],
        "status": "validated" if args.dry_run else "draft",
    }


def _grant_approve(args: argparse.Namespace) -> dict[str, Any]:
    draft = _load_json(args.grant)
    if args.dry_run:
        approved = validate_grant_approval(
            draft,
            args.hash,
            approved_by=args.approved_by,
            approved_at=args.approved_at,
        )
        return {"command": "grant.approve", "content_hash": approved["content_hash"], "grant_id": approved["grant_id"], "status": "validated"}
    database_path = _database_path(
        args.database,
        load_config(args.config) if args.config else None,
        args.config,
    )
    with Database(database_path) as database:
        database.migrate()
        approved = GrantStore(database).approve(
            draft, args.hash, approved_by=args.approved_by, approved_at=args.approved_at,
        )
    return {"command": "grant.approve", "content_hash": approved["content_hash"], "grant_id": approved["grant_id"], "status": "approved"}


def _grant_revoke(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config) if args.config else None
    database_path = _database_path(args.database, config, args.config)
    if not database_path.is_file():
        raise FileNotFoundError(f"database does not exist: {database_path}")
    if args.dry_run:
        with Database(database_path, read_only=True) as database:
            GrantStore(database).validate_revoke(
                args.grant_id, actor=args.actor, event_at=args.event_at
            )
    else:
        with Database(database_path) as database:
            database.migrate()
            GrantStore(database).revoke(args.grant_id, actor=args.actor, event_at=args.event_at)
    return {"command": "grant.revoke", "grant_id": args.grant_id, "status": "validated" if args.dry_run else "revoked"}


def _export(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config) if args.config else None
    database_path = _database_path(args.database, config, args.config)
    if not database_path.is_file():
        raise FileNotFoundError(f"database does not exist: {database_path}")
    _assert_safe_export_destination(database_path, args.output)
    with Database(database_path, read_only=True) as database:
        current_version = database.current_version()
        expected_version = max(migration.version for migration in Database.migrations())
        if current_version != expected_version:
            raise ConfigError(
                "export requires the current SQLite schema: "
                f"found {current_version}, expected {expected_version}"
            )
        inventory = validate_export(PaperRepository(database))
        if args.dry_run:
            return {
                "command": "export",
                "database_path": str(database_path),
                "format": args.format,
                "output_path": str(args.output),
                "planned_export_count": (
                    inventory["jsonl_rows"]
                    if args.format == "jsonl"
                    else inventory["papers"]
                ),
                "planned_paper_count": inventory["papers"],
                "status": "validated",
            }
        exporter = export_jsonl if args.format == "jsonl" else export_csv
        exported_count = exporter(PaperRepository(database), args.output)
    return {"command": "export", "database_path": str(database_path), "exported_count": exported_count, "format": args.format, "output_path": str(args.output), "status": "complete"}


def _assert_safe_export_destination(database_path: Path, output_path: Path) -> None:
    protected = (
        database_path,
        Path(f"{database_path}-wal"),
        Path(f"{database_path}-shm"),
        Path(f"{database_path}-journal"),
    )
    destination = output_path.resolve()
    for candidate in protected:
        if destination == candidate.resolve():
            raise ConfigError("export output must not overwrite the SQLite fact store")
        if output_path.exists() and candidate.exists() and output_path.samefile(candidate):
            raise ConfigError("export output must not alias the SQLite fact store")


def _migrate_config(args: argparse.Namespace) -> dict[str, Any]:
    report = migrate_legacy_yaml(args.input)
    if args.write is not None and not args.dry_run:
        write_migrated(report, args.write)
    return {
        "command": "migrate-config",
        "config": report.converted_config,
        "field_mappings": report.field_mappings,
        "input_path": str(args.input),
        "output_path": str(args.write) if args.write else None,
        "status": "validated" if args.write is None or args.dry_run else "written",
        "unmigrated": report.unmigrated,
        "warnings": report.warnings,
    }


def _database_path(database_path: Path | None, config: Mapping[str, Any] | None, config_path: Path | None) -> Path:
    database = database_path or _configured_database(config, config_path)
    if database is None:
        raise ConfigError("--database or a v2 --config with storage.sqlite_path is required")
    return database


def _grant_defaults(
    config: Mapping[str, Any] | None, args: argparse.Namespace
) -> Mapping[str, Any]:
    if args.kind != "download":
        return {}
    if config is None:
        raise ConfigError(
            "download grant drafts require --config and its frozen grant_defaults"
        )
    overrides = {
        "action": args.action,
        "purpose": args.purpose,
        "mode": args.mode,
        "allow_unattended": args.allow_unattended,
        "paper_id": args.paper_id,
        "artifact_hash": args.artifact_hash,
        "collection_id": args.collection_id,
        "collection_snapshot_hash": args.collection_snapshot_hash,
        "selection_snapshot_hash": args.selection_snapshot_hash,
        "domain": args.domain,
        "provider": args.provider,
        "model": args.model,
        "data_category": args.data_category,
        "max_papers": args.max_papers,
        "expires_at": args.expires_at,
        "skill_digest": args.skill_digest,
        "dependency_digest": args.dependency_digest,
        "lineage_hash": args.lineage_hash,
    }
    supplied = sorted(
        name for name, value in overrides.items() if value not in (None, [], ())
    )
    if supplied:
        raise ConfigError(
            "download grant content must come only from grant_defaults; remove CLI overrides: "
            + ", ".join(supplied)
        )
    defaults = config["download"]["authorized_skill"]["grant_defaults"]
    audit = load_audit_record()
    expected_digests = {
        "source_zip_sha256": audit.original_zip_sha256,
        "installed_content_sha256": audit.installed_content_sha256,
        "dependency_lock_sha256": audit.dependency_lock_sha256,
    }
    drifted = sorted(
        key for key, expected in expected_digests.items()
        if defaults.get(key) != expected
    )
    if drifted:
        raise ConfigError(
            "download grant_defaults differ from the checked-in skill audit: "
            + ", ".join(drifted)
        )
    return defaults


def _grant_scope(args: argparse.Namespace, defaults: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "paper_ids": args.paper_id or list(defaults.get("paper_ids", ())),
        "artifact_hashes": args.artifact_hash,
        "collection_ids": args.collection_id,
        "collection_snapshot_hash": args.collection_snapshot_hash or defaults.get("collection_snapshot_hash"),
        "selection_snapshot_hash": args.selection_snapshot_hash or defaults.get("selection_snapshot_hash"),
        "domains": args.domain or list(defaults.get("allowed_domains", ())),
        "provider": args.provider,
        "model": args.model,
        "data_categories": args.data_category,
    }


def _search_plan(
    input_path: Path, output_root: Path, *, dry_run: bool = False
) -> dict[str, Any]:
    draft = load_yaml(input_path)
    providers = _provider_specs(
        draft.pop("providers"),
        input_path.parent,
        venue_ids=draft["scope"]["venues"],
    )
    plan = compile_query_plan(draft, providers=providers)
    store = QueryPlanStore(output_root)
    path = store.draft_path(str(plan["plan_id"]))
    if not dry_run:
        store.save_draft(plan)
    return {
        "command": "search.plan",
        "draft_path": str(path),
        "estimated_max_candidates": plan["budgets"]["max_candidates"],
        "estimated_max_requests": plan["budgets"]["max_requests"],
        "estimated_max_seconds": plan["budgets"]["max_seconds"],
        "plan_hash": plan["plan_hash"],
        "plan_id": plan["plan_id"],
        "status": "validated" if dry_run else plan["status"],
        "write_performed": not dry_run,
    }


def _search_approve(
    plan_path: Path,
    expected_hash: str,
    approved_by: str,
    approved_at: str | None,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    plan = _load_json(plan_path)
    timestamp = approved_at or datetime.now(UTC).isoformat().replace("+00:00", "Z")
    store = QueryPlanStore(_store_root(plan_path))
    if dry_run:
        approved = approve_query_plan(
            plan,
            expected_hash,
            approved_by=approved_by,
            approved_at=timestamp,
        )
    else:
        approved = store.approve_and_save(
            plan,
            expected_hash,
            approved_by=approved_by,
            approved_at=timestamp,
        )
    return {
        "approved_path": str(store.approved_path(str(approved["plan_id"]))),
        "command": "search.approve",
        "latest_path": str(store.latest_path),
        "plan_hash": approved["plan_hash"],
        "plan_id": approved["plan_id"],
        "status": "validated" if dry_run else approved["status"],
        "write_performed": not dry_run,
    }


def _search_run(
    plan_path: Path,
    *,
    database_path: Path | None,
    contact: str | None,
    snapshot_values: Sequence[str],
    stage2_release_path: Path | None,
    config_path: Path | None,
    run_id: str | None,
    dry_run: bool,
    venue_only: bool = False,
    historical_replay: bool = False,
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
        release_path = stage2_release_path or _configured_stage2_release()
        if release_path is None:
            raise Stage2ReleaseError(
                "search startup requires --stage2-release or PAPER_AGENT_STAGE2_RELEASE"
            )
        released_stage2 = load_stage2_release(release_path, plan)
        return {
            "command": "search.run",
            "database_path": str(database),
            "plan_hash": plan["plan_hash"],
            "plan_id": plan["plan_id"],
            "provider_invocation": "skipped_dry_run",
            "resolved_providers": sorted(
                provider["provider"] for provider in runtime if provider["resolved"]
            ),
            "stage2_release_hash": released_stage2.release_hash,
            "status": "runtime_validated",
        }
    result, resolved_run_id, crawl_run_id = execute_search_plan(
        plan,
        database,
        run_id=run_id,
        contact=operator_contact,
        snapshot_paths=snapshots,
        stage2_release_path=stage2_release_path,
        venue_only=venue_only,
        historical_replay=historical_replay,
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
    assert_runtime_matches(
        plan,
        runtime["providers"],
        budgets=runtime["budgets"],
        policies=runtime["execution"],
        include_arxiv_candidates=plan["scope"]["include_arxiv_candidates"],
    )
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


def _crawl(
    venue_ids: Sequence[str],
    *,
    plan_path: Path | None = None,
    database_path: Path | None = None,
    contact: str | None = None,
    snapshot_values: Sequence[str] = (),
    stage2_release_path: Path | None = None,
    config_path: Path | None = None,
    run_id: str | None = None,
    dry_run: bool = False,
    historical_replay: bool = False,
) -> dict[str, Any]:
    catalog = load_catalog()
    normalized_ids = sorted(set(venue_ids))
    venues = [catalog.venue(venue_id) for venue_id in normalized_ids]
    if plan_path is not None:
        plan = _load_json(plan_path)
        _assert_venue_only_plan(plan, normalized_ids, venues)
        result = _search_run(
            plan_path,
            database_path=database_path,
            contact=contact,
            snapshot_values=snapshot_values,
            stage2_release_path=stage2_release_path,
            config_path=config_path,
            run_id=run_id,
            dry_run=dry_run,
            venue_only=True,
            historical_replay=historical_replay,
        )
        return {
            **result,
            "command": "crawl",
            "mode": "venue_descriptor_compatibility",
            "venue_ids": normalized_ids,
        }
    return {
        "command": "crawl",
        "mode": "venue_descriptor_compatibility",
        "search_audit_intent": {
            "event": "venue_descriptor_discovery_planned",
            "providers": sorted({venue["primary_provider"] for venue in venues}),
            "venue_ids": [venue["venue_id"] for venue in venues],
        },
    }


def _assert_venue_only_plan(
    plan: Mapping[str, Any],
    venue_ids: Sequence[str],
    venues: Sequence[Mapping[str, Any]],
) -> None:
    if sorted(set(plan["scope"]["venues"])) != list(venue_ids):
        raise ValueError("crawl venues must exactly match the approved QueryPlan scope")
    primary_providers = {str(venue["primary_provider"]) for venue in venues}
    resolved_providers = {
        str(provider["provider"]) for provider in plan["providers"] if provider["resolved"]
    }
    if resolved_providers != primary_providers:
        raise ValueError("crawl QueryPlan may resolve only the requested venue primary providers")
    if plan["scope"].get("user_seeds"):
        raise ValueError("crawl QueryPlan cannot contain user seeds")
    if plan["citation_snowball"]["enabled"]:
        raise ValueError("crawl QueryPlan must disable citation expansion")
    if set(plan["execution"]["required_roles"]) != {"venue_primary"}:
        raise ValueError("crawl QueryPlan must require only venue_primary")
    if set(plan["execution"]["required_providers"]) != primary_providers:
        raise ValueError("crawl QueryPlan must require every venue primary provider")


def _import_seeds(
    database_path: Path,
    seed_values: Sequence[str],
    input_paths: Sequence[Path],
    run_id: str | None,
    dry_run: bool,
) -> dict[str, Any]:
    inputs = (*tuple(seed_input(value) for value in seed_values), *inputs_from_files(input_paths))
    if not inputs:
        raise ValueError("import-seeds requires at least one --seed or --input")
    if dry_run:
        validate_seed_inputs(inputs)
        return {
            "command": "import-seeds",
            "database_path": str(database_path),
            "input_count": len(inputs),
            "status": "validated",
        }
    result = import_seeds(database_path, inputs, run_id=run_id)
    return {
        "command": "import-seeds",
        "database_path": str(database_path),
        "imported_count": result.imported_count,
        "input_count": result.input_count,
        "paper_ids": result.paper_ids,
        "run_id": result.run_id,
        "status": "complete",
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
            "upstream_policies": manifest.get("upstream_policies", {}),
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


def _configured_stage2_release() -> Path | None:
    value = os.environ.get("PAPER_AGENT_STAGE2_RELEASE")
    return Path(value) if value else None


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


def _runtime_argv(argv: Sequence[str] | None) -> list[str]:
    """Allow the three global runtime flags before or after any subcommand."""
    values = list(sys.argv[1:] if argv is None else argv)
    prefix: list[str] = []
    remainder: list[str] = []
    index = 0
    extract = True
    while index < len(values):
        value = values[index]
        if value == "--":
            extract = False
            remainder.append(value)
            index += 1
            continue
        if extract and value in {"--config", "--run-id"}:
            prefix.append(value)
            if index + 1 < len(values):
                prefix.append(values[index + 1])
                index += 2
            else:
                index += 1
            continue
        if extract and any(
            value.startswith(f"{option}=") for option in ("--config", "--run-id")
        ):
            prefix.append(value)
            index += 1
            continue
        if extract and value == "--dry-run":
            prefix.append(value)
            index += 1
            continue
        remainder.append(value)
        index += 1
    return [*prefix, *remainder]


def _runtime_option(argv: Sequence[str], option: str) -> str | None:
    result: str | None = None
    for index, value in enumerate(argv):
        if value == option and index + 1 < len(argv):
            result = argv[index + 1]
        elif value.startswith(f"{option}="):
            result = value.partition("=")[2]
    return result


def _command_from_argv(argv: Sequence[str]) -> str:
    index = 0
    while index < len(argv):
        value = argv[index]
        if value in {"--config", "--run-id"}:
            index += 2
            continue
        if value == "--dry-run" or value.startswith(("--config=", "--run-id=")):
            index += 1
            continue
        break
    if index >= len(argv):
        return "unknown"
    command = argv[index]
    if command in {"grant", "search"} and index + 1 < len(argv):
        command = f"{command}.{argv[index + 1]}"
    return command


def _command_stage(command: str) -> str:
    if command.startswith(("search", "crawl", "import-seeds")):
        return "stage1"
    if command.startswith("grant"):
        return "authorization"
    return "system"


def _finish(
    args: argparse.Namespace, payload: Mapping[str, Any], exit_code: int = 0
) -> int:
    document = dict(payload)
    command = str(document.get("command", args.command))
    document.setdefault("event_code", f"{command}.completed")
    document.setdefault("stage", _command_stage(command))
    document.setdefault("status", "complete" if exit_code == 0 else "blocked")
    if args.run_id is not None:
        document.setdefault("run_id", args.run_id)
    _emit(document)
    return exit_code


def _emit(payload: Mapping[str, Any]) -> None:
    print(canonical_json(dict(payload)).decode("utf-8"))
