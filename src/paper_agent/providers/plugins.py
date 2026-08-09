"""Pre-import verification and isolated execution for third-party providers."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from importlib import metadata
from pathlib import Path
import json
import os
import subprocess
import sys
from typing import Any, Iterable

from paper_agent.domain import ProviderCapability

from .api import ProviderManifest


class PluginRejected(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PluginAllowlistEntry:
    distribution: str
    version: str
    provider: str
    entry_point: str
    content_digest: str
    signature: str | None = None


@dataclass(frozen=True, slots=True)
class PluginRegistration:
    manifest: ProviderManifest
    entry_point: str
    distribution: str
    version: str
    content_digest: str
    third_party: bool = True


def distribution_digest(distribution: metadata.Distribution) -> str:
    digest = sha256()
    files = tuple(sorted(distribution.files or (), key=str))
    for file in files:
        path = Path(distribution.locate_file(file))
        if not path.is_file():
            continue
        digest.update(str(file).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


class SubprocessProviderRunner:
    """Runs an approved plugin through one JSON request/response process."""

    def __init__(self, command: tuple[str, ...], *, cwd: Path, environment: dict[str, str] | None = None) -> None:
        self.command = command
        self.cwd = cwd
        self.environment = {"PATH": os.defpath, **(environment or {})}

    @classmethod
    def for_entry_point(cls, entry_point: str, *, cwd: Path) -> "SubprocessProviderRunner":
        return cls(
            (sys.executable, "-m", "paper_agent.providers.worker", entry_point),
            cwd=cwd,
            environment={"PYTHONPATH": os.pathsep.join(sys.path)},
        )

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        completed = subprocess.run(
            self.command,
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            check=True,
            cwd=self.cwd,
            env=self.environment,
        )
        return json.loads(completed.stdout)


class PluginRegistry:
    """Trust registry that never imports a third-party entry point in-process."""

    def __init__(self, allowlist: Iterable[PluginAllowlistEntry] = (), *, third_party_enabled: bool = False) -> None:
        self.allowlist = tuple(allowlist)
        self.third_party_enabled = third_party_enabled
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
            content_digest="builtin",
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
        name = distribution.metadata["Name"]
        version = distribution.version
        digest = distribution_digest(distribution)
        if manifest.builtin or not manifest.enabled:
            raise PluginRejected("provider manifest is not enabled for third-party loading")
        if (
            manifest.distribution != name
            or manifest.version != version
            or manifest.entry_point != entry_point.value
            or manifest.artifact_sha256 != digest
        ):
            raise PluginRejected("installed provider differs from its manifest")
        candidates = [
            item
            for item in self.allowlist
            if (
                item.distribution == name
                and item.version == version
                and item.provider == manifest.provider
                and item.entry_point == entry_point.value
                and item.content_digest == digest
            )
        ]
        if not candidates:
            raise PluginRejected("provider entry point is not exactly allowlisted")
        declared_signature = distribution.metadata.get("X-Paper-Agent-Signature")
        if not any(item.signature == declared_signature for item in candidates):
            raise PluginRejected("provider signature does not match its allowlist")
        registration = PluginRegistration(manifest, entry_point.value, name, version, digest)
        self.registrations[manifest.provider] = registration
        return registration

    def discover(self, manifests: dict[str, ProviderManifest]) -> tuple[PluginRegistration, ...]:
        registrations: list[PluginRegistration] = []
        for entry_point in metadata.entry_points(group="paper_agent.providers"):
            distribution = entry_point.dist
            if distribution is None or entry_point.name not in manifests:
                continue
            registrations.append(self.register_third_party(manifests[entry_point.name], entry_point, distribution))
        return tuple(registrations)

    def require_capability(self, provider: str, capability: ProviderCapability) -> PluginRegistration:
        registration = self.registrations[provider]
        if not registration.manifest.supports(capability):
            raise PluginRejected(f"{provider} does not declare {capability.value}")
        return registration

    def dispatch(
        self,
        provider: str,
        capability: ProviderCapability,
        payload: dict[str, Any],
        *,
        cwd: Path,
    ) -> dict[str, Any]:
        registration = self.require_capability(provider, capability)
        if not registration.third_party:
            raise PluginRejected("builtin providers must be dispatched by the coordinator")
        return SubprocessProviderRunner.for_entry_point(registration.entry_point, cwd=cwd).run(payload)
