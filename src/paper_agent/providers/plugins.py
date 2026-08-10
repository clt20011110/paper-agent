"""Pre-import verification and isolated execution for third-party providers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from hashlib import sha256
from importlib import metadata
from pathlib import Path
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
from typing import Any

from paper_agent.domain import (
    CitationBatch,
    CitationEdgeType,
    Paper,
    ProviderCapability,
    ProviderRole,
    QuerySpec,
    SourceBatch,
)

from .api import (
    CrawlWindow,
    ProviderManifest,
    SeedInput,
    VenueDescriptor,
    validate_citation_batch,
    validate_source_batch,
)
from .sandbox import SandboxPolicy, build_sandbox_command, interpreter_read_roots


class PluginRejected(ValueError):
    """Raised before a plugin import when trust metadata is incomplete or stale."""


class PluginExecutionError(RuntimeError):
    """Raised when an approved plugin fails inside its isolated process."""


_WORKER_BOOTSTRAP = (
    "import os,resource,runpy,sys;"
    "resource.setrlimit(resource.RLIMIT_FSIZE,(16777216,16777216));"
    "sys.path[:0]=[path for path in sys.argv[1].split(os.pathsep) if path];"
    "sys.argv=[sys.argv[0],*sys.argv[2:]];"
    "runpy.run_module('paper_agent.providers.worker',run_name='__main__')"
)
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_ENTRY_POINT_GROUP = "paper_agent.providers"
_PROTOCOL_VERSION = 1
_PLUGIN_ROOT_TOKEN = "__PAPER_AGENT_VERIFIED_PLUGIN_ROOT__"
_MAX_IPC_BYTES = 16 * 1024 * 1024
_MAX_BATCH_ENTRIES = 10_000


@dataclass(frozen=True, slots=True)
class PluginAllowlistEntry:
    distribution: str
    version: str
    provider: str
    entry_point: str
    artifact_sha256: str
    signature: str | None = None

    @property
    def content_digest(self) -> str:
        """Compatibility alias for the original helper API."""

        return self.artifact_sha256

    @classmethod
    def from_mapping(cls, document: Mapping[str, Any]) -> "PluginAllowlistEntry":
        required = ("distribution", "version", "provider", "entry_point", "artifact_sha256")
        missing = [field for field in required if not isinstance(document.get(field), str) or not document[field]]
        if missing:
            raise PluginRejected("plugin allowlist entry is missing " + ", ".join(missing))
        digest = str(document["artifact_sha256"])
        if not _SHA256.fullmatch(digest):
            raise PluginRejected("plugin allowlist artifact_sha256 must be lowercase SHA-256")
        signature = document.get("signature")
        if signature is not None and (not isinstance(signature, str) or not signature):
            raise PluginRejected("plugin allowlist signature must be a non-empty string")
        return cls(
            str(document["distribution"]),
            str(document["version"]),
            str(document["provider"]),
            str(document["entry_point"]),
            digest,
            signature,
        )


@dataclass(frozen=True, slots=True)
class PluginRegistration:
    manifest: ProviderManifest
    entry_point: str
    distribution: str
    version: str
    artifact_sha256: str
    signature: str | None = None
    third_party: bool = True

    @property
    def content_digest(self) -> str:
        """Compatibility alias for callers written against the original name."""

        return self.artifact_sha256


def plugin_allowlist_from_config(config: Mapping[str, Any] | None) -> tuple[PluginAllowlistEntry, ...]:
    """Read the already schema-validated v2 allowlist without importing plugins."""

    if config is None:
        return ()
    sources = config.get("sources")
    if sources is None:
        return ()
    if not isinstance(sources, Mapping):
        raise PluginRejected("configuration sources must be an object")
    documents = sources.get("plugin_allowlist", ())
    if not isinstance(documents, Sequence) or isinstance(documents, (str, bytes)):
        raise PluginRejected("plugin_allowlist must be an array")
    entries = tuple(
        PluginAllowlistEntry.from_mapping(document)
        if isinstance(document, Mapping)
        else _reject_allowlist_item()
        for document in documents
    )
    providers = [entry.provider for entry in entries]
    if len(providers) != len(set(providers)):
        raise PluginRejected("plugin allowlist repeats a provider")
    return entries


def _reject_allowlist_item() -> PluginAllowlistEntry:
    raise PluginRejected("plugin allowlist entries must be objects")


def distribution_digest(distribution: metadata.Distribution) -> str:
    """Hash every file named by installed distribution metadata deterministically."""

    inventory = _distribution_inventory(distribution)
    return _inventory_digest(inventory)


def _inventory_digest(inventory: Iterable[tuple[Path, Path]]) -> str:
    digest = sha256()
    for relative, path in inventory:
        encoded_path = str(relative).encode("utf-8")
        content = path.read_bytes()
        digest.update(len(encoded_path).to_bytes(8, "big"))
        digest.update(encoded_path)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _distribution_inventory(
    distribution: metadata.Distribution,
) -> tuple[tuple[Path, Path], ...]:
    """Resolve only regular, in-root, non-symlink files named by distribution metadata."""

    files = tuple(sorted(distribution.files or (), key=str))
    if not files:
        raise PluginRejected("installed distribution has no auditable file inventory")
    declared_root = Path(distribution.locate_file(".")).absolute()
    resolved_root = declared_root.resolve()
    inventory: list[tuple[Path, Path]] = []
    for file in files:
        relative = Path(file)
        if relative.is_absolute() or ".." in relative.parts:
            raise PluginRejected("installed distribution contains an unsafe file path")
        located = Path(distribution.locate_file(file)).absolute()
        try:
            located.relative_to(declared_root)
        except ValueError as error:
            raise PluginRejected("installed distribution file escapes its install root") from error
        cursor = declared_root
        for component in relative.parts:
            cursor /= component
            if cursor.is_symlink():
                raise PluginRejected("installed distribution inventory may not contain symlinks")
        try:
            resolved = located.resolve(strict=True)
            resolved.relative_to(resolved_root)
        except (FileNotFoundError, ValueError) as error:
            raise PluginRejected(f"installed distribution file is missing or escapes its root: {relative}") from error
        try:
            mode = resolved.stat().st_mode
        except OSError as error:
            raise PluginRejected(f"installed distribution file cannot be inspected: {relative}") from error
        if not stat.S_ISREG(mode):
            raise PluginRejected(f"installed distribution inventory is not a regular file: {relative}")
        inventory.append((relative, resolved))
    return tuple(inventory)


def _stage_distribution(
    registration: PluginRegistration,
    destination: Path,
) -> Path:
    """Copy the attested inventory into an invocation-local, sandbox-read-only tree."""

    distribution = attest_registration(registration)
    inventory = _distribution_inventory(distribution)
    destination.mkdir(parents=True)
    staged: list[tuple[Path, Path]] = []
    for relative, source in inventory:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        staged.append((relative, target))
    if _inventory_digest(staged) != registration.artifact_sha256:
        raise PluginRejected("installed provider content changed while staging")
    return destination


def _distribution_name(distribution: metadata.Distribution) -> str:
    try:
        name = distribution.metadata["Name"]
    except (KeyError, TypeError) as error:
        raise PluginRejected("installed distribution has no Name metadata") from error
    if not name:
        raise PluginRejected("installed distribution has no Name metadata")
    return str(name)


def _installed_entry_point(
    distribution: metadata.Distribution,
    *,
    provider: str,
    expected_value: str,
) -> metadata.EntryPoint:
    candidates = tuple(
        point
        for point in distribution.entry_points
        if point.group == _ENTRY_POINT_GROUP and point.name == provider
    )
    if len(candidates) != 1 or candidates[0].value != expected_value:
        raise PluginRejected("installed provider entry point does not exactly match")
    return candidates[0]


def _installed_distribution(name: str) -> metadata.Distribution:
    try:
        return metadata.distribution(name)
    except metadata.PackageNotFoundError as error:
        raise PluginRejected(f"plugin distribution is not installed: {name}") from error


def attest_registration(registration: PluginRegistration) -> metadata.Distribution:
    """Recheck all installed facts immediately before a subprocess import."""

    distribution = _installed_distribution(registration.distribution)
    facts = (
        _distribution_name(distribution),
        str(distribution.version),
        distribution_digest(distribution),
    )
    expected = (
        registration.distribution,
        registration.version,
        registration.artifact_sha256,
    )
    if facts != expected:
        raise PluginRejected("installed provider distribution version or content has drifted")
    _installed_entry_point(
        distribution,
        provider=registration.manifest.provider,
        expected_value=registration.entry_point,
    )
    declared_signature = distribution.metadata.get("X-Paper-Agent-Signature")
    if declared_signature != registration.signature:
        raise PluginRejected("installed provider signature has drifted")
    return distribution


class SubprocessProviderRunner:
    """Run one approved provider request in a network-denied JSON subprocess."""

    def __init__(
        self,
        command: tuple[str, ...],
        *,
        cwd: Path | None = None,
        environment: Mapping[str, str] | None = None,
        environment_names: Iterable[str] = (),
        read_roots: Iterable[Path] = (),
        timeout_seconds: float = 60,
        registration: PluginRegistration | None = None,
    ) -> None:
        supplied = dict(environment or {})
        names = tuple(dict.fromkeys(str(name) for name in environment_names))
        if set(supplied) - set(names):
            raise PluginRejected("plugin environment contains undeclared names")
        self.command = command
        roots = tuple(read_roots) or ((cwd,) if cwd is not None else ())
        self.read_roots = tuple(dict.fromkeys(Path(root).resolve() for root in roots))
        self.environment = {
            name: supplied.get(name, os.environ[name])
            for name in names
            if name in supplied or name in os.environ
        }
        self.timeout_seconds = timeout_seconds
        self.registration = registration

    @classmethod
    def for_registration(
        cls,
        registration: PluginRegistration,
        *,
        environment: Mapping[str, str] | None = None,
        timeout_seconds: float = 60,
    ) -> "SubprocessProviderRunner":
        package_root = Path(__file__).resolve().parents[2]
        credentials = registration.manifest.credential_policy.environment_variables
        return cls(
            (
                sys.executable,
                "-I",
                "-S",
                "-B",
                "-c",
                _WORKER_BOOTSTRAP,
                os.pathsep.join((str(package_root), _PLUGIN_ROOT_TOKEN)),
                registration.distribution,
                registration.version,
                registration.manifest.provider,
                registration.entry_point,
                registration.artifact_sha256,
                registration.signature or "",
            ),
            environment=environment,
            environment_names=credentials,
            read_roots=(package_root,),
            timeout_seconds=timeout_seconds,
            registration=registration,
        )

    def sandbox_command(
        self,
        work_root: Path,
        *,
        command: tuple[str, ...] | None = None,
        read_roots: Iterable[Path] = (),
    ) -> tuple[str, ...]:
        roots = (*interpreter_read_roots(), *self.read_roots, *read_roots)
        policy = SandboxPolicy(tuple(dict.fromkeys(roots)), work_root.resolve())
        return build_sandbox_command(command or self.command, policy)

    def run(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        request = json.dumps(dict(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
        if len(request) > _MAX_IPC_BYTES:
            raise PluginExecutionError("provider subprocess request exceeds the IPC limit")
        with tempfile.TemporaryDirectory(prefix="paper-agent-plugin-") as directory:
            temporary_root = Path(directory)
            work_root = temporary_root / "work"
            work_root.mkdir()
            command = self.command
            extra_read_roots: tuple[Path, ...] = ()
            if self.registration is not None:
                verified_root = _stage_distribution(
                    self.registration,
                    temporary_root / "verified-plugin",
                )
                command = tuple(
                    argument.replace(_PLUGIN_ROOT_TOKEN, str(verified_root))
                    for argument in command
                )
                extra_read_roots = (verified_root,)
            stdout_path = work_root / "stdout.json"
            stderr_path = work_root / "stderr.log"
            process: subprocess.Popen[bytes] | None = None
            with stdout_path.open("w+b") as stdout_file, stderr_path.open("wb") as stderr_file:
                try:
                    process = subprocess.Popen(
                        self.sandbox_command(
                            work_root,
                            command=command,
                            read_roots=extra_read_roots,
                        ),
                        stdin=subprocess.PIPE,
                        stdout=stdout_file,
                        stderr=stderr_file,
                        cwd=work_root,
                        env={**self.environment, "TMPDIR": str(work_root)},
                        start_new_session=True,
                    )
                    process.communicate(request, timeout=self.timeout_seconds)
                except subprocess.TimeoutExpired as error:
                    if process is not None:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        process.communicate()
                    raise PluginExecutionError("provider subprocess timed out") from error
                stdout_file.flush()
                if os.fstat(stdout_file.fileno()).st_size > _MAX_IPC_BYTES:
                    raise PluginExecutionError("provider subprocess response exceeds the IPC limit")
                stdout_file.seek(0)
                stdout = stdout_file.read()
            if process.returncode:
                # Plugin stderr may contain a credential; never reflect it into logs.
                raise PluginExecutionError(
                    f"provider subprocess failed with exit status {process.returncode}"
                )
            try:
                response = json.loads(stdout)
            except (json.JSONDecodeError, TypeError) as error:
                raise PluginExecutionError("provider subprocess returned invalid JSON") from error
            if not isinstance(response, dict):
                raise PluginExecutionError("provider subprocess response must be an object")
            return response


class PluginRegistry:
    """Trust registry that never imports a third-party entry point in-process."""

    def __init__(
        self,
        allowlist: Iterable[PluginAllowlistEntry] = (),
        *,
        third_party_enabled: bool | None = None,
    ) -> None:
        self.allowlist = tuple(allowlist)
        self.third_party_enabled = bool(self.allowlist) if third_party_enabled is None else third_party_enabled
        providers = [entry.provider for entry in self.allowlist]
        if len(providers) != len(set(providers)):
            raise PluginRejected("plugin allowlist repeats a provider")
        self.registrations: dict[str, PluginRegistration] = {}

    def register_builtin(self, manifest: ProviderManifest, entry_point: str) -> PluginRegistration:
        if not manifest.enabled or not manifest.builtin:
            raise PluginRejected("provider is not an enabled builtin")
        if manifest.entry_point and manifest.entry_point != entry_point:
            raise PluginRejected("builtin entry point differs from its manifest")
        registration = PluginRegistration(
            manifest=manifest,
            entry_point=entry_point,
            distribution="paper-agent",
            version=manifest.version,
            artifact_sha256=manifest.artifact_sha256 or "builtin",
            third_party=False,
        )
        self.registrations[manifest.provider] = registration
        return registration

    def register_third_party(
        self,
        manifest: ProviderManifest,
        entry_point: metadata.EntryPoint,
        distribution: metadata.Distribution,
    ) -> PluginRegistration:
        if not self.third_party_enabled:
            raise PluginRejected("third-party providers are disabled")
        name = _distribution_name(distribution)
        version = str(distribution.version)
        digest = distribution_digest(distribution)
        if manifest.builtin or not manifest.enabled:
            raise PluginRejected("provider manifest is not enabled for third-party loading")
        if (
            manifest.distribution != name
            or manifest.version != version
            or manifest.entry_point != entry_point.value
            or manifest.artifact_sha256 != digest
        ):
            raise PluginRejected("installed provider differs from its trusted manifest")
        candidates = tuple(
            item
            for item in self.allowlist
            if (
                item.distribution == name
                and item.version == version
                and item.provider == manifest.provider
                and item.entry_point == entry_point.value
                and item.artifact_sha256 == digest
            )
        )
        if len(candidates) != 1:
            raise PluginRejected("provider entry point is not exactly allowlisted")
        declared_signature = distribution.metadata.get("X-Paper-Agent-Signature")
        if candidates[0].signature != declared_signature:
            raise PluginRejected("provider signature does not match its allowlist")
        registration = PluginRegistration(
            manifest,
            entry_point.value,
            name,
            version,
            digest,
            declared_signature,
        )
        self.registrations[manifest.provider] = registration
        return registration

    def verify_requested(
        self,
        manifests: Mapping[str, ProviderManifest],
        providers: Iterable[str],
    ) -> tuple[PluginRegistration, ...]:
        """Verify allowlisted requested plugins using metadata only."""

        registrations: list[PluginRegistration] = []
        for provider in sorted(set(providers)):
            manifest = manifests.get(provider)
            if manifest is None or manifest.builtin:
                continue
            candidates = tuple(item for item in self.allowlist if item.provider == provider)
            if not candidates:
                # Default-disabled third parties remain unresolved in QueryPlan compilation.
                continue
            if len(candidates) != 1:
                raise PluginRejected(f"{provider}: plugin allowlist is ambiguous")
            allow = candidates[0]
            if manifest.distribution != allow.distribution:
                raise PluginRejected(f"{provider}: allowlist differs from trusted manifest")
            distribution = _installed_distribution(allow.distribution)
            point = _installed_entry_point(
                distribution,
                provider=provider,
                expected_value=allow.entry_point,
            )
            registrations.append(self.register_third_party(manifest, point, distribution))
        return tuple(registrations)

    def discover(
        self, manifests: Mapping[str, ProviderManifest]
    ) -> tuple[PluginRegistration, ...]:
        """Discover only explicitly allowlisted providers, using metadata only."""

        return self.verify_requested(
            manifests,
            (entry.provider for entry in self.allowlist),
        )

    def require_contract(
        self,
        provider: str,
        role: ProviderRole,
        capability: ProviderCapability,
    ) -> PluginRegistration:
        try:
            registration = self.registrations[provider]
        except KeyError as error:
            raise PluginRejected(f"third-party provider is not verified: {provider}") from error
        if role not in registration.manifest.roles:
            raise PluginRejected(f"{provider} does not declare role {role.value}")
        if not registration.manifest.supports(capability):
            raise PluginRejected(f"{provider} does not declare {capability.value}")
        return registration

    def require_capability(self, provider: str, capability: ProviderCapability) -> PluginRegistration:
        """Compatibility helper that checks capability without choosing an operation."""

        try:
            registration = self.registrations[provider]
        except KeyError as error:
            raise PluginRejected(f"provider is not registered: {provider}") from error
        if not registration.manifest.supports(capability):
            raise PluginRejected(f"{provider} does not declare {capability.value}")
        return registration

    def dispatch(
        self,
        provider: str,
        role: ProviderRole,
        capability: ProviderCapability,
        operation: str,
        arguments: Mapping[str, Any],
        *,
        environment: Mapping[str, str] | None = None,
        timeout_seconds: float = 60,
    ) -> dict[str, Any]:
        registration = self.require_contract(provider, role, capability)
        if not registration.third_party:
            raise PluginRejected("builtin providers must be dispatched by the coordinator")
        credentials = registration.manifest.credential_policy.environment_variables
        supplied_environment = (
            {name: str(environment[name]) for name in credentials if name in environment}
            if environment is not None
            else None
        )
        # for_registration repeats the distribution/version/entry-point/content
        # attestation immediately before the worker performs its own pre-import check.
        runner = SubprocessProviderRunner.for_registration(
            registration,
            environment=supplied_environment,
            timeout_seconds=timeout_seconds,
        )
        return runner.run(
            {
                "protocol_version": _PROTOCOL_VERSION,
                "provider": provider,
                "operation": operation,
                "arguments": dict(arguments),
            }
        )


class IsolatedProviderClient:
    """Search/citation/venue client backed by a verified plugin subprocess."""

    def __init__(
        self,
        registry: PluginRegistry,
        provider: str,
        *,
        environment: Mapping[str, str] | None = None,
        timeout_seconds: float = 60,
    ) -> None:
        self.registry = registry
        try:
            self.manifest = registry.registrations[provider].manifest
        except KeyError as error:
            raise PluginRejected(f"third-party provider is not verified: {provider}") from error
        self.provider = provider
        self.environment = environment
        self.timeout_seconds = timeout_seconds

    def _dispatch_source(
        self,
        role: ProviderRole,
        operation: str,
        arguments: Mapping[str, Any],
        *,
        query_hash: str | None,
    ) -> SourceBatch:
        response = self.registry.dispatch(
            self.provider,
            role,
            ProviderCapability.METADATA,
            operation,
            arguments,
            environment=self.environment,
            timeout_seconds=self.timeout_seconds,
        )
        if response.get("protocol_version") != _PROTOCOL_VERSION or response.get("result_type") != "source_batch":
            raise PluginExecutionError("provider returned the wrong IPC envelope")
        result = response.get("result")
        if not isinstance(result, Mapping):
            raise PluginExecutionError("provider source batch is missing")
        try:
            batch = validate_source_batch(SourceBatch.from_dict(result))
        except (KeyError, TypeError, ValueError) as error:
            raise PluginExecutionError("provider returned an invalid SourceBatch") from error
        if query_hash is not None and batch.query_hash != query_hash:
            raise PluginExecutionError("provider SourceBatch query hash does not match the request")
        if any(entry.provider != self.provider for entry in batch.entries):
            raise PluginExecutionError("provider SourceBatch contains foreign provenance")
        if len(batch.entries) > _MAX_BATCH_ENTRIES:
            raise PluginExecutionError("provider SourceBatch exceeds the entry limit")
        return replace(
            batch,
            source_run_id=f"{self.provider}:{operation}:{batch.query_hash[:12]}",
        )

    def search(self, query_spec: QuerySpec, cursor: str | None = None) -> SourceBatch:
        if not query_spec.native_query_hash:
            raise PluginRejected("third-party search requires a frozen native query hash")
        batch = self._dispatch_source(
            ProviderRole.SEARCH,
            "search",
            {"query_spec": query_spec.to_dict(), "cursor": cursor},
            query_hash=query_spec.native_query_hash,
        )
        if query_spec.page_size is not None and len(batch.entries) > query_spec.page_size:
            raise PluginExecutionError("provider SourceBatch exceeds the frozen page size")
        return batch

    def discover(
        self,
        descriptor: VenueDescriptor,
        window: CrawlWindow,
        cursor: str | None = None,
    ) -> SourceBatch:
        parameters = {
            **descriptor.parameters,
            "venue_id": descriptor.venue_id,
            "adapter": descriptor.adapter,
            "date_from": window.date_from,
            "date_to": window.date_to,
            "year": window.year,
            "volume": window.volume,
            "issue": window.issue,
            "cursor": cursor,
        }
        return self._dispatch_source(
            ProviderRole.VENUE_PRIMARY,
            "discover",
            {
                "descriptor": _venue_descriptor_document(descriptor),
                "window": _crawl_window_document(window),
                "cursor": cursor,
            },
            query_hash=sha256(
                json.dumps(parameters, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest(),
        )

    def import_seeds(self, input_spec: Sequence[SeedInput]) -> SourceBatch:
        return self._dispatch_source(
            ProviderRole.LIBRARY,
            "import_seeds",
            {"input_spec": [_seed_input_document(item) for item in input_spec]},
            query_hash=sha256(
                "|".join(item.value for item in input_spec).encode("utf-8")
            ).hexdigest(),
        )

    def _dispatch_citation(
        self,
        operation: str,
        seed: Paper,
        cursor: str | None,
        capability: ProviderCapability,
    ) -> CitationBatch:
        response = self.registry.dispatch(
            self.provider,
            ProviderRole.CITATION,
            capability,
            operation,
            {"seed": seed.to_dict(), "cursor": cursor},
            environment=self.environment,
            timeout_seconds=self.timeout_seconds,
        )
        if response.get("protocol_version") != _PROTOCOL_VERSION or response.get("result_type") != "citation_batch":
            raise PluginExecutionError("provider returned the wrong IPC envelope")
        result = response.get("result")
        if not isinstance(result, Mapping):
            raise PluginExecutionError("provider citation batch is missing")
        try:
            batch = validate_citation_batch(CitationBatch.from_dict(result))
        except (KeyError, TypeError, ValueError) as error:
            raise PluginExecutionError("provider returned an invalid CitationBatch") from error
        expected_direction = (
            CitationEdgeType.REFERENCES
            if operation == "references"
            else CitationEdgeType.CITATIONS
        )
        expected_query_hash = sha256(
            f"{seed.paper_id}:{operation}".encode("utf-8")
        ).hexdigest()
        if batch.query_hash != expected_query_hash:
            raise PluginExecutionError("provider CitationBatch query hash does not match the request")
        if any(
            edge.provider != self.provider
            or edge.edge_type is not expected_direction
            or (edge.candidate is not None and edge.candidate.provider != self.provider)
            for edge in batch.entries
        ):
            raise PluginExecutionError("provider CitationBatch contains foreign provenance")
        if len(batch.entries) > _MAX_BATCH_ENTRIES:
            raise PluginExecutionError("provider CitationBatch exceeds the entry limit")
        return replace(
            batch,
            source_run_id=f"{self.provider}:{operation}:{expected_query_hash[:12]}",
        )

    def references(self, seed: Paper, cursor: str | None = None) -> CitationBatch:
        return self._dispatch_citation(
            "references", seed, cursor, ProviderCapability.REFERENCES
        )

    def citations(self, seed: Paper, cursor: str | None = None) -> CitationBatch:
        return self._dispatch_citation(
            "citations", seed, cursor, ProviderCapability.CITATIONS
        )


def _venue_descriptor_document(value: VenueDescriptor) -> dict[str, Any]:
    return {
        "schema_version": value.schema_version,
        "venue_id": value.venue_id,
        "provider": value.provider,
        "adapter": value.adapter,
        "parameters": value.parameters,
    }


def _crawl_window_document(value: CrawlWindow) -> dict[str, Any]:
    return {
        "date_from": value.date_from,
        "date_to": value.date_to,
        "year": value.year,
        "volume": value.volume,
        "issue": value.issue,
    }


def _seed_input_document(value: SeedInput) -> dict[str, Any]:
    return {"kind": value.kind, "value": value.value, "source_name": value.source_name}
