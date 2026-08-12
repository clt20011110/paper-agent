"""Composable, offline-first runtime diagnostics for Paper Agent.

``SystemDoctor`` deliberately does not create a database, make HTTP requests,
download a model, or read credential values.  Callers that want to prove a
local service is responding can explicitly inject an HTTP probe.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from importlib import metadata
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys
from types import SimpleNamespace
from typing import Callable, Literal, Mapping, Sequence, cast
from urllib.parse import urlparse

from .approval import ApprovalError, require_valid_approval
from .authorized_skill_runtime import AuthorizedSkillRuntime
from .codex_exec import CodexExec, CodexExecError, FROZEN_PROFILES
from .config import ConfigError, load_config
from .canonical import content_hash
from .grants import GrantError, GrantStore
from .manifests import ManifestCatalog, ManifestError, load_catalog
from .providers.plugins import distribution_digest
from .resources import release_asset_root, stage2_model_lock_paths
from .schema import SchemaValidationError, validate
from .stage2_backends import ModelLock, load_model_lock
from .stage2_search import ReleasedStage2, Stage2ReleaseError, load_stage2_release
from .storage import Database


CheckStatus = Literal["pass", "warning", "blocker"]
CommandRunner = Callable[..., subprocess.CompletedProcess[str]]
HttpProbe = Callable[[str], tuple[int, str]]
ExecutableFinder = Callable[[str], str | None]
BrowserSessionProbe = Callable[[], tuple[bool, str]]


_SEMANTIC_VERSION = re.compile(r"(?<![0-9])([0-9]+)\.([0-9]+)\.([0-9]+)(?![0-9])")
_DIGEST = re.compile(r"^[a-f0-9]{64}$")


@dataclass(frozen=True, slots=True)
class DoctorCheck:
    name: str
    status: CheckStatus
    required: bool
    detail: str
    production_required: bool = False


@dataclass(frozen=True, slots=True)
class SystemDoctorReport:
    checks: tuple[DoctorCheck, ...]

    @property
    def ready(self) -> bool:
        return not any(item.required and item.status == "blocker" for item in self.checks)

    @property
    def blockers(self) -> tuple[DoctorCheck, ...]:
        return tuple(item for item in self.checks if item.status == "blocker")

    @property
    def production_ready(self) -> bool:
        return self.ready and not any(
            item.production_required and item.status != "pass" for item in self.checks
        )

    @property
    def production_blockers(self) -> tuple[DoctorCheck, ...]:
        return tuple(
            item
            for item in self.checks
            if (item.required and item.status == "blocker")
            or (item.production_required and item.status != "pass")
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "ready": self.ready,
            "production_ready": self.production_ready,
            "checks": [
                {
                    "name": item.name,
                    "status": item.status,
                    "required": item.required,
                    "production_required": item.production_required,
                    "detail": item.detail,
                }
                for item in self.checks
            ],
        }


@dataclass(frozen=True, slots=True)
class DoctorPaths:
    """All file-system inputs; no user home path is assumed by the core."""

    repository_root: Path
    config_path: Path | None = None
    database_path: Path | None = None
    manifest_root: Path | None = None
    model_lock_paths: tuple[Path, ...] = ()
    stage2_release_path: Path | None = None
    query_plan_path: Path | None = None
    authorized_skill_runtime: AuthorizedSkillRuntime | None = None
    minimum_free_bytes: int = 1_000_000_000

    @classmethod
    def defaults(cls, repository_root: Path | None = None) -> "DoctorPaths":
        root = repository_root or release_asset_root()
        return cls(
            repository_root=root,
            model_lock_paths=stage2_model_lock_paths(root),
        )


class SystemDoctor:
    """Run local diagnostics in a stable order suitable for CLI and tests."""

    def __init__(
        self,
        paths: DoctorPaths | None = None,
        *,
        environment: Mapping[str, str] | None = None,
        command_runner: CommandRunner | None = None,
        executable_finder: ExecutableFinder = shutil.which,
        http_probe: HttpProbe | None = None,
        browser_session_probe: BrowserSessionProbe | None = None,
        prove_codex_models: bool = False,
        now: datetime | None = None,
        python_version: tuple[int, int] | None = None,
        disk_usage: Callable[[str | Path], shutil._ntuple_diskusage] = shutil.disk_usage,
    ) -> None:
        self.paths = paths or DoctorPaths.defaults()
        self.environment = dict(os.environ if environment is None else environment)
        self.command_runner = command_runner or self._run_command
        self._default_command_runner = command_runner is None
        self.executable_finder = executable_finder
        self.http_probe = http_probe
        self.browser_session_probe = browser_session_probe
        self.prove_codex_models = prove_codex_models
        self.now = now or datetime.now(UTC)
        self.python_version = python_version or sys.version_info[:2]
        self.disk_usage = disk_usage

    def run(self) -> SystemDoctorReport:
        config = self._config()
        stage2_check, released_stage2, query_plan = self._stage2_release(config)
        checks = [
            self._python(), self._disk(), self._database(),
            *self._catalog_and_providers(config, query_plan),
            self._model_locks(config, released_stage2), stage2_check, self._omlx(released_stage2),
            self._codex(), *self._authorized_skill(config),
        ]
        return SystemDoctorReport(tuple(checks))

    def _python(self) -> DoctorCheck:
        supported = (3, 11) <= self.python_version <= (3, 13)
        return DoctorCheck("python", "pass" if supported else "blocker", True,
                           f"Python {self.python_version[0]}.{self.python_version[1]}" if supported else "requires Python 3.11 through 3.13")

    def _disk(self) -> DoctorCheck:
        target = self.paths.database_path or self.paths.repository_root
        probe = target
        while not probe.exists() and probe.parent != probe:
            probe = probe.parent
        try:
            available = self.disk_usage(probe).free
        except OSError as error:
            return DoctorCheck("disk", "blocker", True, str(error), True)
        status: CheckStatus = "pass" if available >= self.paths.minimum_free_bytes else "warning"
        return DoctorCheck("disk", status, False, f"{available} bytes free", True)

    def _config(self) -> Mapping[str, object] | None:
        if self.paths.config_path is None:
            return None
        try:
            return load_config(self.paths.config_path)
        except (OSError, ConfigError) as error:
            return {"_doctor_error": str(error)}

    def _database(self) -> DoctorCheck:
        path = self.paths.database_path
        if path is None:
            return DoctorCheck("database", "warning", False, "no SQLite path supplied", True)
        if not path.is_file():
            return DoctorCheck("database", "warning", False, f"database will be initialized at {path}", True)
        try:
            connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            try:
                row = connection.execute("SELECT COALESCE(MAX(version), 0) FROM schema_migrations").fetchone()
            finally:
                connection.close()
        except sqlite3.Error as error:
            return DoctorCheck("database", "blocker", True, f"SQLite read check failed: {error}", True)
        try:
            current = int(row[0])
        except (TypeError, ValueError) as error:
            return DoctorCheck(
                "database", "blocker", True,
                f"SQLite migration version is invalid: {error}", True,
            )
        expected = max(item.version for item in Database.migrations())
        if current > expected:
            return DoctorCheck("database", "blocker", True, f"database migration {current} is newer than installed {expected}", True)
        if current < expected:
            return DoctorCheck("database", "warning", False, f"{expected - current} migration(s) pending", True)
        return DoctorCheck("database", "pass", True, f"migration {current} is current", True)

    def _catalog_and_providers(
        self,
        config: Mapping[str, object] | None,
        query_plan: Mapping[str, object] | None = None,
    ) -> tuple[DoctorCheck, ...]:
        if config is not None and "_doctor_error" in config:
            return (DoctorCheck("config", "blocker", True, str(config["_doctor_error"]), True),)
        result: list[DoctorCheck] = []
        try:
            catalog = load_catalog(self.paths.manifest_root)
        except (OSError, ManifestError, ValueError) as error:
            return (DoctorCheck("provider_catalog", "blocker", True, str(error), True),)
        result.append(DoctorCheck(
            "provider_catalog", "pass", True,
            f"{len(catalog.providers)} providers; manifests and built-in digests trusted", True,
        ))
        result.append(DoctorCheck("config", "pass" if config is not None else "warning", config is not None,
                                  "configuration schema is valid" if config is not None else "no configuration supplied", True))
        configured = self._configured_providers(config, catalog) if query_plan is None else ()
        usable_roles: set[str] = set()
        conditional_roles: set[str] = set()
        for provider, enabled_state, mode, snapshot_path, configured_roles in configured:
            manifest = catalog.providers.get(provider)
            if manifest is None:
                result.append(DoctorCheck(
                    f"provider:{provider}", "blocker", enabled_state != "disabled",
                    "not present in trusted manifest catalog", enabled_state != "disabled",
                ))
                continue
            manifest_roles = set(map(str, manifest["roles"]))
            unknown_roles = set(configured_roles) - manifest_roles
            roles_valid = not unknown_roles
            if unknown_roles:
                result.append(DoctorCheck(
                    f"provider:{provider}:roles", "blocker", True,
                    "configured roles are not declared by the manifest: " + ", ".join(sorted(unknown_roles)),
                    True,
                ))
            if enabled_state != "disabled" and not manifest["enabled"]:
                result.append(DoctorCheck(
                    f"provider:{provider}:manifest", "blocker", enabled_state == "enabled",
                    "configuration enables a provider disabled by its trusted manifest", True,
                ))
                continue
            rate = manifest["rate_limit"]
            credentials = _credential_environment_variables(manifest["authentication"])
            authentication_required = manifest["authentication"]["required"] is True
            declared = bool(credentials) or not authentication_required
            available = declared and all(bool(self.environment.get(name)) for name in credentials)
            provider_ready = manifest["enabled"] is True and (not authentication_required or available)
            if enabled_state != "disabled" and authentication_required and not declared:
                result.append(DoctorCheck(
                    f"provider:{provider}:credentials", "blocker", enabled_state == "enabled",
                    "authentication is required but the manifest declares no credential environment variable", True,
                ))
                provider_ready = False
            elif enabled_state != "disabled" and authentication_required and not available:
                status: CheckStatus = "blocker" if enabled_state == "enabled" else "warning"
                result.append(DoctorCheck(
                    f"provider:{provider}:credentials", status, enabled_state == "enabled",
                    "declared credentials are absent", True,
                ))
                provider_ready = False
            else:
                status = "pass" if enabled_state != "conditional" else "warning"
                result.append(DoctorCheck(
                    f"provider:{provider}", status, enabled_state == "enabled",
                    f"state={enabled_state}; roles={','.join(manifest['roles'])}; "
                    f"qps={rate['global_qps']}; concurrency={rate['max_concurrency']}; "
                    f"credentials={'present' if authentication_required else 'not required'}",
                    enabled_state != "disabled",
                ))
            snapshot_ready = True
            if mode == "snapshot" and enabled_state != "disabled":
                snapshot = self._snapshot_check(provider, snapshot_path, enabled_state)
                result.append(snapshot)
                snapshot_ready = snapshot.status == "pass"
            if provider_ready and snapshot_ready and roles_valid:
                if enabled_state == "enabled":
                    usable_roles.update(configured_roles)
                elif enabled_state == "conditional":
                    conditional_roles.update(configured_roles)
        if config is not None and query_plan is None:
            required_roles = self._required_roles(config)
            missing = required_roles - usable_roles
            unresolved = missing.intersection(conditional_roles)
            hard_missing = missing - unresolved
            if hard_missing:
                result.append(DoctorCheck(
                    "provider_roles", "blocker", True,
                    "required roles have no enabled, usable provider: " + ", ".join(sorted(hard_missing)),
                    True,
                ))
            elif unresolved:
                result.append(DoctorCheck(
                    "provider_roles", "warning", False,
                    "required roles depend on unresolved auto provider policy: " + ", ".join(sorted(unresolved)),
                    True,
                ))
            else:
                result.append(DoctorCheck(
                    "provider_roles", "pass", True,
                    "all required roles have an enabled, usable manifest provider", True,
                ))
        if query_plan is not None:
            result.append(self._query_plan_provider_check(query_plan, config, catalog))
        result.extend(self._plugin_checks(config, catalog, query_plan))
        return tuple(result)

    def _query_plan_provider_check(
        self,
        plan: Mapping[str, object],
        config: Mapping[str, object] | None,
        catalog: ManifestCatalog,
    ) -> DoctorCheck:
        errors: list[str] = []
        providers = cast(Sequence[Mapping[str, object]], plan["providers"])
        resolved_roles: set[str] = set()
        resolved_names: set[str] = set()
        for frozen in providers:
            name = str(frozen["provider"])
            manifest = catalog.providers.get(name)
            if manifest is None:
                errors.append(f"{name}: no current trusted manifest")
                continue
            current = {
                "distribution": manifest["distribution"],
                "version": manifest["version"],
                "entry_point": manifest["entry_point"],
                "artifact_sha256": manifest["artifact_sha256"],
                "manifest_hash": content_hash(manifest),
                "roles": sorted(map(str, manifest["roles"])),
                "capabilities": sorted(map(str, manifest["capabilities"])),
                "authority": manifest["authority"],
                "rate_limit": manifest["rate_limit"],
                "data_use": manifest["terms"]["data_use"],
                "terms_url": manifest["terms"].get("url"),
                "independence_group": manifest["independence_group"],
                "upstream_families": sorted(map(str, manifest["upstream_families"])),
                "upstream_policies": manifest.get("upstream_policies", {}),
            }
            for field, expected in current.items():
                actual = frozen.get(field)
                if field in {"roles", "capabilities", "upstream_families"} and isinstance(actual, list):
                    actual = sorted(map(str, actual))
                if actual != expected:
                    errors.append(f"{name}: approved QueryPlan {field} has drifted")
            credential_names = _credential_environment_variables(manifest["authentication"])
            availability = {key: bool(self.environment.get(key)) for key in credential_names}
            if manifest["authentication"]["required"] is True and not credential_names:
                errors.append(f"{name}: authentication is required without declared credential variables")
            if frozen.get("credential_environment_variables") != list(credential_names):
                errors.append(f"{name}: credential declarations have drifted")
            if frozen.get("credentials_required") != manifest["authentication"]["required"]:
                errors.append(f"{name}: credentials_required has drifted")
            if frozen.get("credential_availability") != availability:
                errors.append(f"{name}: credential availability has drifted")
            if frozen.get("credentials_present") != (
                not credential_names or all(availability.values())
            ):
                errors.append(f"{name}: credentials_present has drifted")
            if frozen.get("resolved") is True:
                if not manifest["enabled"]:
                    errors.append(f"{name}: resolved provider is now disabled")
                resolved_names.add(name)
                resolved_roles.update(map(str, frozen["roles"]))
                if frozen.get("mode") in {"snapshot", "bulk_snapshot"}:
                    observed = self._configured_snapshot_hash(config, name)
                    if observed is None or observed != frozen.get("snapshot_hash"):
                        errors.append(f"{name}: approved bulk snapshot hash cannot be reproduced")
        execution = cast(Mapping[str, object], plan["execution"])
        missing_roles = set(map(str, cast(Sequence[object], execution["required_roles"]))) - resolved_roles
        missing_providers = set(map(str, cast(Sequence[object], execution["required_providers"]))) - resolved_names
        if missing_roles:
            errors.append("required roles are unresolved: " + ", ".join(sorted(missing_roles)))
        if missing_providers:
            errors.append("required providers are unresolved: " + ", ".join(sorted(missing_providers)))
        if errors:
            return DoctorCheck(
                "query_plan_providers", "blocker", True, "; ".join(errors), True,
            )
        return DoctorCheck(
            "query_plan_providers", "pass", True,
            "approved provider manifests, roles, capabilities, credentials, and snapshots match runtime", True,
        )

    def _configured_snapshot_hash(
        self,
        config: Mapping[str, object] | None,
        provider: str,
    ) -> str | None:
        if config is None or "_doctor_error" in config:
            return None
        sources = cast(Mapping[str, object], config["sources"])
        defaults = cast(Mapping[str, object], sources["plan_defaults"])
        providers = cast(Mapping[str, Mapping[str, object]], defaults["providers"])
        provider_config = providers.get(provider)
        if provider_config is None or not isinstance(provider_config.get("snapshot_path"), str):
            return None
        path = Path(cast(str, provider_config["snapshot_path"]))
        if not path.is_absolute():
            if self.paths.config_path is None:
                return None
            path = self.paths.config_path.parent / path
        return _sha256_file(path) if path.is_file() else None

    def _configured_providers(
        self,
        config: Mapping[str, object] | None,
        catalog: ManifestCatalog | None = None,
    ) -> tuple[tuple[str, Literal["enabled", "disabled", "conditional"], str, str | None, tuple[str, ...]], ...]:
        if config is None:
            return ()
        sources = config["sources"]
        assert isinstance(sources, Mapping)
        defaults = sources["plan_defaults"]
        assert isinstance(defaults, Mapping)
        raw = defaults["providers"]
        assert isinstance(raw, Mapping)
        values: dict[str, tuple[
            str, Literal["enabled", "disabled", "conditional"], str, str | None, tuple[str, ...],
        ]] = {}
        for name, entry in sorted(raw.items()):
            assert isinstance(entry, Mapping)
            setting = entry["enabled"]
            enabled_state: Literal["enabled", "disabled", "conditional"]
            if setting is True:
                enabled_state = "enabled"
            elif setting is False:
                enabled_state = "disabled"
            else:
                enabled_state = "conditional"
            values[str(name)] = (
                str(name), enabled_state, str(entry.get("mode", "api")),
                entry.get("snapshot_path") if isinstance(entry.get("snapshot_path"), str) else None,
                tuple(sorted(map(str, entry["roles"]))),
            )
        if catalog is not None:
            venues = defaults.get("venues", [])
            assert isinstance(venues, list)
            for configured_venue in venues:
                assert isinstance(configured_venue, Mapping)
                venue_id = Path(str(configured_venue["descriptor"])).stem
                venue = catalog.venues.get(venue_id)
                if venue is None:
                    continue
                provider = str(venue["primary_provider"])
                existing = values.get(provider)
                if existing is None:
                    values[provider] = (provider, "enabled", "api", None, ("venue_primary",))
                    continue
                name, state, mode, snapshot_path, roles = existing
                values[provider] = (
                    name, state, mode, snapshot_path,
                    tuple(sorted({*roles, "venue_primary"})),
                )
        return tuple(values[name] for name in sorted(values))

    @staticmethod
    def _required_roles(config: Mapping[str, object]) -> set[str]:
        sources = cast(Mapping[str, object], config["sources"])
        defaults = cast(Mapping[str, object], sources["plan_defaults"])
        return set(map(str, cast(Sequence[object], defaults["required_roles"])))

    def _snapshot_check(
        self,
        provider: str,
        raw_path: str | None,
        enabled_state: Literal["enabled", "disabled", "conditional"],
    ) -> DoctorCheck:
        required = enabled_state == "enabled"
        if raw_path is None:
            status: CheckStatus = "blocker" if required else "warning"
            return DoctorCheck(
                f"provider:{provider}:snapshot", status, required,
                "snapshot mode has no snapshot_path", True,
            )
        path = Path(raw_path)
        if not path.is_absolute():
            path = self.paths.config_path.parent / path if self.paths.config_path else self.paths.repository_root / path
        if not path.is_file():
            status = "blocker" if required else "warning"
            return DoctorCheck(
                f"provider:{provider}:snapshot", status, required,
                f"snapshot is missing: {path}", True,
            )
        return DoctorCheck(
            f"provider:{provider}:snapshot", "pass", required,
            f"sha256={_sha256_file(path)}; no approved-plan hash comparison performed", True,
        )

    def _plugin_checks(
        self,
        config: Mapping[str, object] | None,
        catalog: ManifestCatalog,
        query_plan: Mapping[str, object] | None = None,
    ) -> tuple[DoctorCheck, ...]:
        if config is None:
            return ()
        sources = config["sources"]
        assert isinstance(sources, Mapping)
        allowlist = sources["plugin_allowlist"]
        assert isinstance(allowlist, list)
        entries = [cast(Mapping[str, object], item) for item in allowlist if isinstance(item, Mapping)]
        names = [str(item["provider"]) for item in entries]
        if len(names) != len(set(names)):
            return (DoctorCheck("plugins", "blocker", True, "plugin allowlist repeats a provider", True),)
        required_plugins = self._required_third_party_plugins(config, catalog, query_plan)
        missing_allowlist = required_plugins - set(names)
        if missing_allowlist:
            return (DoctorCheck(
                "plugins", "blocker", True,
                "enabled third-party providers are not allowlisted: " + ", ".join(sorted(missing_allowlist)),
                True,
            ),)
        if not entries:
            return (DoctorCheck("plugins", "pass", True, "third-party plugins are disabled", True),)
        errors: list[str] = []
        for entry in entries:
            provider = str(entry["provider"])
            distribution_name = str(entry["distribution"])
            version = str(entry["version"])
            entry_point_value = str(entry["entry_point"])
            expected_digest = str(entry["artifact_sha256"])
            expected_signature = entry.get("signature")
            manifest = catalog.providers.get(provider)
            if manifest is None:
                errors.append(f"{provider}: no trusted provider manifest")
                continue
            if manifest["builtin"] or not manifest["enabled"]:
                errors.append(f"{provider}: trusted manifest is not an enabled third-party provider")
                continue
            expected_manifest = (
                str(manifest["distribution"]), str(manifest["version"]),
                str(manifest["entry_point"]), manifest["artifact_sha256"],
            )
            if expected_manifest != (distribution_name, version, entry_point_value, expected_digest):
                errors.append(f"{provider}: allowlist differs from trusted manifest")
                continue
            try:
                installed = metadata.distribution(distribution_name)
                installed_name = str(installed.metadata["Name"])
                installed_version = str(installed.version)
                installed_digest = distribution_digest(installed)
                installed_signature = installed.metadata.get("X-Paper-Agent-Signature")
                installed_entry_points = tuple(installed.entry_points)
            except (
                metadata.PackageNotFoundError, KeyError, OSError, TypeError, ValueError,
            ) as error:
                errors.append(f"{provider}: installed distribution metadata unavailable ({error})")
                continue
            if (installed_name, installed_version, installed_digest) != (
                distribution_name, version, expected_digest,
            ):
                errors.append(f"{provider}: installed distribution version or digest has drifted")
                continue
            if installed_signature != expected_signature:
                errors.append(f"{provider}: installed distribution signature has drifted")
                continue
            candidates = [
                point
                for point in installed_entry_points
                if point.group == "paper_agent.providers" and point.name == provider
            ]
            if len(candidates) != 1 or candidates[0].value != entry_point_value:
                errors.append(f"{provider}: installed entry point does not exactly match")
        if errors:
            return (DoctorCheck("plugins", "blocker", True, "; ".join(errors), True),)
        return (DoctorCheck(
            "plugins", "pass", True,
            f"{len(entries)} third-party plugin distribution(s) verified before import", True,
        ),)

    def _required_third_party_plugins(
        self,
        config: Mapping[str, object],
        catalog: ManifestCatalog,
        query_plan: Mapping[str, object] | None,
    ) -> set[str]:
        if query_plan is not None:
            names = {
                str(provider["provider"])
                for provider in cast(Sequence[Mapping[str, object]], query_plan["providers"])
                if provider.get("resolved") is True
            }
        else:
            sources = config.get("sources")
            if not isinstance(sources, Mapping) or "plan_defaults" not in sources:
                return set()
            names = {
                name
                for name, state, _, _, _ in self._configured_providers(config, catalog)
                if state != "disabled"
            }
        return {
            name
            for name in names
            if name in catalog.providers and catalog.providers[name]["builtin"] is False
        }

    def _model_locks(
        self,
        config: Mapping[str, object] | None,
        released: ReleasedStage2 | None,
    ) -> DoctorCheck:
        locks: list[ModelLock] = []
        try:
            if not self.paths.model_lock_paths:
                raise ValueError("no Stage 2 model locks were supplied")
            for path in self.paths.model_lock_paths:
                locks.append(load_model_lock(path))
        except (OSError, ValueError) as error:
            return DoctorCheck("stage2_model_locks", "blocker", True, str(error), True)
        backends = [lock.backend for lock in locks]
        if len(backends) != len(set(backends)):
            return DoctorCheck(
                "stage2_model_locks", "blocker", True,
                "Stage 2 model locks repeat a backend", True,
            )
        by_backend = {lock.backend: lock for lock in locks}
        paths_by_backend = {
            lock.backend: path for lock, path in zip(locks, self.paths.model_lock_paths, strict=True)
        }
        if set(by_backend) != {"omlx_rerank", "omlx_chat"}:
            return DoctorCheck(
                "stage2_model_locks", "blocker", True,
                "production needs exactly one omlx_rerank and one omlx_chat lock", True,
            )
        malformed = [
            lock.model_id
            for lock in locks
            if any(_DIGEST.fullmatch(str(digest)) is None for digest in lock.file_hashes.values())
        ]
        if malformed:
            return DoctorCheck(
                "stage2_model_locks", "blocker", True,
                "model locks contain non-SHA-256 file digests: " + ", ".join(malformed), True,
            )
        old = [
            lock.model_id
            for lock in locks
            if (_strict_version(lock.omlx_version) or (0, 0, 0)) < (0, 5, 7)
        ]
        if old:
            return DoctorCheck(
                "stage2_model_locks", "blocker", True,
                f"oMLX lock version below 0.5.7 or invalid: {', '.join(old)}", True,
            )
        mismatch = self._config_lock_mismatches(config, by_backend)
        if released is not None:
            mismatch.extend(self._release_lock_mismatches(released, by_backend, paths_by_backend))
        if mismatch:
            return DoctorCheck(
                "stage2_model_locks", "blocker", True, "; ".join(mismatch), True,
            )
        return DoctorCheck(
            "stage2_model_locks", "pass", True,
            f"{len(locks)} exact model revision locks valid and bound (all <=10B parameters)", True,
        )

    @staticmethod
    def _config_lock_mismatches(
        config: Mapping[str, object] | None,
        locks: Mapping[str, ModelLock],
    ) -> list[str]:
        if config is None or "_doctor_error" in config:
            return []
        filter_config = cast(Mapping[str, object], config["filter"])
        reranker = cast(Mapping[str, object], filter_config["reranker"])
        adjudicator = cast(Mapping[str, object], filter_config["adjudicator"])
        reranker_lock = locks["omlx_rerank"]
        adjudicator_lock = locks["omlx_chat"]
        mismatches: list[str] = []
        expected_reranker = {
            "backend": reranker_lock.backend,
            "model": reranker_lock.conversion_repo or reranker_lock.source_repo,
            "source_repo": reranker_lock.source_repo,
            "source_revision": reranker_lock.source_revision,
            "format": _config_model_format(reranker_lock),
        }
        for field, expected in expected_reranker.items():
            if reranker.get(field) != expected:
                mismatches.append(f"reranker {field} does not match its model lock")
        expected_adjudicator = {
            "backend": adjudicator_lock.backend,
            "model": adjudicator_lock.conversion_repo or adjudicator_lock.source_repo,
            "revision": adjudicator_lock.conversion_revision or adjudicator_lock.source_revision,
        }
        for field, expected in expected_adjudicator.items():
            if adjudicator.get(field) != expected:
                mismatches.append(f"adjudicator {field} does not match its model lock")
        return mismatches

    def _release_lock_mismatches(
        self,
        released: ReleasedStage2,
        locks: Mapping[str, ModelLock],
        paths: Mapping[str, Path],
    ) -> list[str]:
        reranker = locks["omlx_rerank"]
        adjudicator = locks["omlx_chat"]
        reranker_hash = _sha256_file(paths["omlx_rerank"])
        adjudicator_hash = _sha256_file(paths["omlx_chat"])
        expected = (
            released.profile.reranker_model_id,
            released.profile.reranker_revision,
            released.profile.reranker_lock_hash,
            released.profile.adjudicator_model_id,
            released.profile.adjudicator_revision,
            released.profile.adjudicator_lock_hash,
        )
        actual = (
            reranker.model_id,
            reranker.conversion_revision or reranker.source_revision,
            reranker_hash,
            adjudicator.model_id,
            adjudicator.conversion_revision or adjudicator.source_revision,
            adjudicator_hash,
        )
        return [] if expected == actual else ["Stage 2 release model IDs/revisions/digests differ from supplied locks"]

    def _stage2_release(
        self,
        config: Mapping[str, object] | None,
    ) -> tuple[DoctorCheck, ReleasedStage2 | None, Mapping[str, object] | None]:
        plan_path = self.paths.query_plan_path or self._configured_query_plan_path(config)
        release_path = self.paths.stage2_release_path
        if plan_path is None and release_path is None:
            return (
                DoctorCheck(
                    "stage2_release", "warning", False,
                    "no approved QueryPlan and Stage 2 release bundle supplied", True,
                ),
                None,
                None,
            )
        if plan_path is None or release_path is None:
            return (
                DoctorCheck(
                    "stage2_release", "blocker", True,
                    "QueryPlan and Stage 2 release must be supplied together", True,
                ),
                None,
                None,
            )
        try:
            plan_document = json.loads(plan_path.read_text(encoding="utf-8"))
            if not isinstance(plan_document, dict):
                raise ValueError("QueryPlan must be a JSON object")
            validate(plan_document, "query-plan.schema.json")
            require_valid_approval(plan_document, "plan_hash")
            self._require_configured_plan(config, plan_path, plan_document)
            released = load_stage2_release(release_path, plan_document)
        except (
            OSError, json.JSONDecodeError, ValueError, SchemaValidationError,
            ApprovalError, Stage2ReleaseError,
        ) as error:
            return (
                DoctorCheck("stage2_release", "blocker", True, str(error), True),
                None,
                None,
            )
        return (
            DoctorCheck(
                "stage2_release", "pass", True,
                f"approved QueryPlan and release {released.release_hash} are bound", True,
            ),
            released,
            plan_document,
        )

    def _configured_query_plan_path(self, config: Mapping[str, object] | None) -> Path | None:
        if config is None or "_doctor_error" in config or self.paths.config_path is None:
            return None
        sources = cast(Mapping[str, object], config["sources"])
        approved = cast(Mapping[str, object], sources["approved_plan"])
        if approved.get("content_hash") is None and approved.get("required") is not True:
            return None
        path = Path(str(approved["input_path"]))
        return path if path.is_absolute() else self.paths.config_path.parent / path

    def _require_configured_plan(
        self,
        config: Mapping[str, object] | None,
        plan_path: Path,
        plan: Mapping[str, object],
    ) -> None:
        if config is None or "_doctor_error" in config or self.paths.config_path is None:
            return
        sources = cast(Mapping[str, object], config["sources"])
        approved = cast(Mapping[str, object], sources["approved_plan"])
        configured = Path(str(approved["input_path"]))
        configured = configured if configured.is_absolute() else self.paths.config_path.parent / configured
        if configured.resolve() != plan_path.resolve():
            raise ValueError("configured QueryPlan path differs from doctor input")
        configured_hash = approved.get("content_hash")
        if configured_hash is not None and configured_hash != plan.get("plan_hash"):
            raise ValueError("configured QueryPlan hash differs from approved plan")

    def _omlx(self, released: ReleasedStage2 | None) -> DoctorCheck:
        executable = self.executable_finder("omlx")
        if not executable:
            return DoctorCheck(
                "omlx", "warning", False,
                "oMLX CLI not found; version cannot be proved", True,
            )
        try:
            completed = self.command_runner((executable, "--version"))
        except (OSError, subprocess.SubprocessError) as error:
            return DoctorCheck("omlx", "warning", False, str(error), True)
        if completed.returncode != 0:
            return DoctorCheck("omlx", "warning", False, "omlx --version failed", True)
        version_output = f"{completed.stdout}\n{completed.stderr}".strip()
        version = _strict_version(version_output)
        if version is None:
            return DoctorCheck(
                "omlx", "blocker", True,
                f"could not parse semantic oMLX version: {version_output}", True,
            )
        if version < (0, 5, 7):
            return DoctorCheck(
                "omlx", "blocker", True,
                f"requires oMLX >=0.5.7; found {version_output}", True,
            )
        if released is None:
            return DoctorCheck(
                "omlx", "warning", False,
                f"CLI {version_output}; no validated Stage 2 release endpoint", True,
            )
        endpoint = released.omlx_base_url
        if self.http_probe is None:
            return DoctorCheck(
                "omlx", "warning", False,
                f"CLI {version_output}; validated local endpoint was not probed", True,
            )
        parsed = urlparse(endpoint)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            return DoctorCheck("omlx", "blocker", True, "oMLX endpoint must be loopback HTTP", True)
        try:
            status, body = self.http_probe(f"{endpoint.rstrip('/')}/v1/models")
        except OSError as error:
            return DoctorCheck("omlx", "blocker", True, f"local endpoint unavailable: {error}", True)
        if not 200 <= status < 300:
            return DoctorCheck(
                "omlx", "blocker", True,
                f"local model inventory returned HTTP {status}", True,
            )
        listed = _model_ids(body)
        expected = {
            released.profile.reranker_model_id,
            released.profile.adjudicator_model_id,
        }
        if listed is None:
            return DoctorCheck(
                "omlx", "warning", False,
                "local endpoint responded but did not provide a verifiable model inventory", True,
            )
        missing = expected - listed
        if missing:
            return DoctorCheck(
                "omlx", "blocker", True,
                "released local models are not loaded: " + ", ".join(sorted(missing)), True,
            )
        return DoctorCheck(
            "omlx", "pass", True,
            f"CLI {version_output}; local endpoint lists both released models", True,
        )

    def _codex(self) -> DoctorCheck:
        executable = self.executable_finder("codex")
        if not executable:
            return DoctorCheck("codex", "warning", False, "codex CLI not found", True)
        diagnostics: dict[str, str] = {}

        def run(
            argv: Sequence[str], **options: object
        ) -> subprocess.CompletedProcess[str]:
            completed = (
                self.command_runner(argv, **options)
                if self._default_command_runner
                else self.command_runner(argv)
            )
            if tuple(argv[-2:]) == ("login", "status"):
                diagnostics["login"] = f"{completed.stdout}\n{completed.stderr}".strip()
            return completed

        try:
            adapter = CodexExec(
                executable=executable,
                runner=run,
                environment=self.environment,
            )
            report = adapter.doctor(prove_model_availability=self.prove_codex_models)
        except (CodexExecError, OSError, ValueError, subprocess.SubprocessError) as error:
            return DoctorCheck("codex", "blocker", True, str(error), True)
        if _strict_version(report.version) is None:
            return DoctorCheck(
                "codex", "blocker", True,
                f"could not parse semantic Codex CLI version: {report.version}", True,
            )
        login_status = diagnostics.get("login", "").casefold()
        if "not logged in" in login_status or "unauthenticated" in login_status:
            return DoctorCheck(
                "codex", "blocker", True,
                "Codex login status reports that authentication is absent", True,
            )
        missing = [profile.model for name, profile in FROZEN_PROFILES.items() if report.model_availability[name] == "unavailable"]
        if missing:
            return DoctorCheck(
                "codex", "blocker", True,
                "frozen model profiles unavailable: " + ", ".join(sorted(set(missing))), True,
            )
        if not self.prove_codex_models:
            return DoctorCheck(
                "codex", "warning", False,
                f"{report.version}; login command passed; frozen Luna/Sol slugs are listed but not invoked", True,
            )
        return DoctorCheck(
            "codex", "pass", True,
            f"{report.version}; login command passed; frozen Luna/Sol models were invoked successfully", True,
        )

    def _authorized_skill(self, config: Mapping[str, object] | None) -> tuple[DoctorCheck, ...]:
        runtime = self.paths.authorized_skill_runtime
        required = _authorized_enabled(config)
        if runtime is None:
            return (DoctorCheck(
                "authorized_download_skill", "blocker" if required else "warning", required,
                "authorized download skill runtime is not configured", required,
            ),)
        try:
            result = runtime.doctor()
        except (OSError, ValueError) as error:
            return (DoctorCheck(
                "authorized_download_skill", "blocker" if required else "warning", required,
                str(error), required,
            ),)
        if not result.ready:
            return (DoctorCheck(
                "authorized_download_skill", "blocker" if required else "warning", required,
                "; ".join(result.reasons), required,
            ),)
        checks = [DoctorCheck(
            "authorized_download_skill", "pass", required,
            "audited skill archive, installed content, and dependency locks match", required,
        )]
        if not required:
            return tuple(checks)
        checks.extend(self._authorization_grant_checks(config, result))
        if self.browser_session_probe is None:
            checks.append(DoctorCheck(
                "authorized_browser_session", "warning", False,
                "authorized browser login was not proved; attended handoff may require manual login", True,
            ))
        else:
            try:
                available, detail = self.browser_session_probe()
                if not isinstance(available, bool) or not isinstance(detail, str) or not detail:
                    raise ValueError("browser session probe returned an invalid result")
            except (OSError, ValueError) as error:
                available, detail = False, str(error)
            checks.append(DoctorCheck(
                "authorized_browser_session", "pass" if available else "warning", False,
                detail, True,
            ))
        return tuple(checks)

    def _authorization_grant_checks(
        self,
        config: Mapping[str, object] | None,
        skill_result: object,
    ) -> tuple[DoctorCheck, ...]:
        assert config is not None
        download = cast(Mapping[str, object], config["download"])
        authorized = cast(Mapping[str, object], download["authorized_skill"])
        grant_ids = (
            ("authorization_grant", authorized.get("authorization_grant_id"), "download", "download"),
            ("data_sharing_grant", authorized.get("data_sharing_grant_id"), "browser_data_sharing", "browser_data_sharing"),
        )
        checks: list[DoctorCheck] = []
        if not isinstance(grant_ids[0][1], str) or not grant_ids[0][1]:
            checks.append(DoctorCheck(
                "authorization_grant", "blocker", True,
                "enabled authorized downloads require an approved authorization_grant_id", True,
            ))
            return tuple(checks)
        database_path = self.paths.database_path
        if database_path is None or not database_path.is_file():
            return (DoctorCheck(
                "authorization_grants", "blocker", True,
                "authorized grant validation requires an existing SQLite database", True,
            ),)
        try:
            connection = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
            connection.row_factory = sqlite3.Row
        except sqlite3.Error as error:
            return (DoctorCheck(
                "authorization_grants", "blocker", True,
                f"cannot open authorization database read-only: {error}", True,
            ),)
        try:
            store = GrantStore(cast(Database, SimpleNamespace(connection=connection)))
            for name, grant_id, kind, action in grant_ids:
                if grant_id is None and name == "data_sharing_grant":
                    checks.append(DoctorCheck(
                        name, "pass", False,
                        "no browser page content is approved for model sharing", False,
                    ))
                    continue
                if not isinstance(grant_id, str) or not grant_id:
                    checks.append(DoctorCheck(name, "blocker", True, "configured grant ID is invalid", True))
                    continue
                try:
                    grant = store.load(grant_id, kind=kind, now=self.now)
                    if action not in grant.document["actions"]:
                        raise GrantError(f"grant does not allow {action}")
                    if grant.document["purpose"] != download["purpose"]:
                        raise GrantError("grant purpose differs from the configured download purpose")
                    installed_digest = getattr(skill_result, "installed_content_sha256", None)
                    dependency_digest = getattr(skill_result, "dependency_lock_sha256", None)
                    if grant.document["skill_digest"] != installed_digest:
                        raise GrantError("grant skill digest differs from the audited installation")
                    if grant.document["dependency_digest"] != dependency_digest:
                        raise GrantError("grant dependency digest differs from the audited installation")
                    approved_event = connection.execute(
                        "SELECT 1 FROM authorization_grant_events "
                        "WHERE grant_id = ? AND event_type = 'approved'",
                        (grant_id,),
                    ).fetchone()
                    if approved_event is None:
                        raise GrantError("grant has no immutable approved event")
                except (
                    GrantError, sqlite3.Error, SchemaValidationError,
                    json.JSONDecodeError, TypeError, ValueError,
                ) as error:
                    checks.append(DoctorCheck(name, "blocker", True, str(error), True))
                else:
                    checks.append(DoctorCheck(
                        name, "pass", True,
                        "approval hash, revocation, expiry, actions, and audited digests are valid", True,
                    ))
        finally:
            connection.close()
        return tuple(checks)

    @staticmethod
    def _run_command(
        argv: Sequence[str], **options: object
    ) -> subprocess.CompletedProcess[str]:
        if options:
            return subprocess.run(tuple(argv), **options)  # type: ignore[arg-type]
        return subprocess.run(tuple(argv), text=True, capture_output=True, check=False)


def _strict_version(value: str) -> tuple[int, int, int] | None:
    match = _SEMANTIC_VERSION.search(value)
    if match is None:
        return None
    return tuple(map(int, match.groups()))  # type: ignore[return-value]


def _config_model_format(lock: ModelLock) -> str:
    if lock.quantization == "none" and "fp32" in lock.format.casefold():
        return "fp32"
    if "bf16" in lock.format.casefold():
        return "bf16"
    if "4bit" in lock.quantization.casefold() or "4bit" in lock.format.casefold():
        return "4bit"
    return lock.format


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _model_ids(body: str) -> set[str] | None:
    try:
        document = json.loads(body)
    except json.JSONDecodeError:
        return None
    if not isinstance(document, Mapping):
        return None
    raw = document.get("data", document.get("models"))
    if not isinstance(raw, list):
        return None
    values: set[str] = set()
    for item in raw:
        if isinstance(item, str):
            values.add(item)
        elif isinstance(item, Mapping):
            value = item.get("id", item.get("slug"))
            if isinstance(value, str) and value:
                values.add(value)
    return values


def _authorized_enabled(config: Mapping[str, object] | None) -> bool:
    if config is None:
        return False
    download = config.get("download")
    if not isinstance(download, Mapping):
        return False
    authorized = download.get("authorized_skill")
    return isinstance(authorized, Mapping) and authorized.get("enabled") is True


def _credential_environment_variables(authentication: Mapping[str, object]) -> tuple[str, ...]:
    declared = authentication.get("credential_envs", {})
    names = tuple(declared.values()) if isinstance(declared, Mapping) else ()
    single = authentication.get("credential_env")
    if isinstance(single, str):
        names = (*names, single)
    return tuple(sorted(str(name) for name in names))
