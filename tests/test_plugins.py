from __future__ import annotations

from email.message import Message
from importlib import metadata
from pathlib import Path
import platform
import shutil
import socket
import sys

import pytest

from paper_agent.domain import ProviderCapability, ProviderRole
from paper_agent.providers.api import ProviderManifest
from paper_agent.providers.plugins import (
    PluginAllowlistEntry,
    PluginRejected,
    PluginRegistry,
    SubprocessProviderRunner,
    distribution_digest,
)
from paper_agent.providers.sandbox import SandboxPolicy, SandboxUnavailable, build_sandbox_command, macos_profile


class FakeDistribution:
    def __init__(self, root: Path, version: str = "1.2.3") -> None:
        self.root = root
        self.version = version
        self.files = (Path("plugin.py"),)
        self.metadata = Message()
        self.metadata["Name"] = "example-plugin"

    def locate_file(self, path: Path) -> Path:
        return self.root / path


class FakeEntryPoint:
    def __init__(self) -> None:
        self.name = "example"
        self.value = "plugin:factory"

    def load(self) -> None:
        pytest.fail("entry point was imported")


def manifest(*, content_digest: str | None = None, third_party: bool = False) -> ProviderManifest:
    return ProviderManifest(
        provider="example",
        version="1.2.3" if third_party else "1.0",
        roles=(ProviderRole.SEARCH,),
        capabilities=(ProviderCapability.METADATA,),
        stable_identifier="example-id",
        distribution="example-plugin" if third_party else "paper-agent",
        entry_point="plugin:factory" if third_party else "paper_agent.providers.api:ProviderManifest",
        artifact_sha256=content_digest,
        builtin=not third_party,
    )


def test_untrusted_plugins_are_disabled_before_any_import(tmp_path: Path) -> None:
    (tmp_path / "plugin.py").write_text("not imported")
    entry_point = metadata.EntryPoint("example", "plugin:factory", "paper_agent.providers")
    with pytest.raises(PluginRejected, match="disabled"):
        PluginRegistry().register_third_party(manifest(), entry_point, FakeDistribution(tmp_path))


def test_mismatch_rejects_before_entry_point_load(tmp_path: Path) -> None:
    (tmp_path / "plugin.py").write_text("not imported")
    with pytest.raises(PluginRejected, match="exactly allowlisted"):
        PluginRegistry(third_party_enabled=True).register_third_party(
            manifest(content_digest=distribution_digest(FakeDistribution(tmp_path)), third_party=True),
            FakeEntryPoint(),
            FakeDistribution(tmp_path),
        )


def test_declared_capability_is_enforced() -> None:
    registry = PluginRegistry()
    registry.register_builtin(manifest(), "paper_agent.providers.api:ProviderManifest")
    assert registry.require_capability("example", ProviderCapability.METADATA).manifest.provider == "example"
    with pytest.raises(PluginRejected, match="does not declare"):
        registry.require_capability("example", ProviderCapability.CITATIONS)


def test_runner_exposes_only_declared_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DECLARED_TOKEN", "allowed")
    monkeypatch.setenv("UNDECLARED_TOKEN", "blocked")
    runner = SubprocessProviderRunner(
        (sys.executable, "-c", "pass"),
        cwd=tmp_path,
        environment_names=("DECLARED_TOKEN",),
    )
    assert runner.environment == {"DECLARED_TOKEN": "allowed"}


def test_runner_rejects_undeclared_environment(tmp_path: Path) -> None:
    with pytest.raises(PluginRejected, match="undeclared"):
        SubprocessProviderRunner(
            (sys.executable, "-c", "pass"),
            cwd=tmp_path,
            environment={"TOKEN": "secret"},
            environment_names=(),
        )


def test_macos_profile_denies_network_and_limits_writes(tmp_path: Path) -> None:
    profile = macos_profile(SandboxPolicy((tmp_path / "read",), tmp_path / "work"))
    assert "(deny default)" in profile
    assert "(deny network*)" in profile
    assert str(tmp_path / "read") in profile
    assert str(tmp_path / "work") in profile
    assert '(import "system.sb")' in profile
    assert "require-not" in profile


def test_subprocess_json_roundtrip_when_platform_sandbox_is_available(tmp_path: Path) -> None:
    system = platform.system()
    available = (system == "Darwin" and shutil.which("sandbox-exec")) or (
        system == "Linux" and shutil.which("bwrap")
    )
    if not available:
        pytest.skip("no platform sandbox is installed")
    declared = tmp_path / "declared.txt"
    declared.write_text("declared")
    secret = tmp_path.parent / "outside-secret.txt"
    secret.write_text("secret")
    (tmp_path / "plugin.py").write_text(
        "from pathlib import Path\n"
        "import socket\n\n"
        "def factory():\n"
        "    def handle(payload):\n"
        "        declared = Path(payload['declared']).read_text()\n"
        "        try:\n"
        "            Path(payload['secret']).read_text()\n"
        "            secret_blocked = False\n"
        "        except OSError:\n"
        "            secret_blocked = True\n"
        "        try:\n"
        "            socket.create_connection(('127.0.0.1', payload['port']), timeout=1).close()\n"
        "            network_blocked = False\n"
        "        except OSError:\n"
        "            network_blocked = True\n"
        "        return {'declared': declared, 'secret_blocked': secret_blocked, 'network_blocked': network_blocked}\n"
        "    return handle\n"
    )
    runner = SubprocessProviderRunner.for_entry_point("plugin:factory", cwd=tmp_path)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        result = runner.run({"declared": str(declared), "secret": str(secret), "port": listener.getsockname()[1]})
    assert result == {"declared": "declared", "secret_blocked": True, "network_blocked": True}


def test_sandbox_fails_closed_without_a_platform_enforcer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("paper_agent.providers.sandbox.platform.system", lambda: "Darwin")
    monkeypatch.setattr("paper_agent.providers.sandbox.shutil.which", lambda _: None)
    with pytest.raises(SandboxUnavailable, match="sandbox-exec"):
        build_sandbox_command((sys.executable, "-c", "pass"), SandboxPolicy((tmp_path,), tmp_path))


def test_linux_bubblewrap_has_network_and_filesystem_boundary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("paper_agent.providers.sandbox.platform.system", lambda: "Linux")
    monkeypatch.setattr("paper_agent.providers.sandbox.shutil.which", lambda _: "/usr/bin/bwrap")
    command = build_sandbox_command(
        (sys.executable, "-c", "pass"), SandboxPolicy((tmp_path / "read",), tmp_path / "work")
    )
    assert command[:4] == ("/usr/bin/bwrap", "--die-with-parent", "--new-session", "--unshare-net")
    assert ("--ro-bind", str(tmp_path / "read"), str(tmp_path / "read")) == command[8:11]
    assert command.count("--ro-bind") == 1
    assert ("--ro-bind", "/", "/") not in zip(command, command[1:], command[2:])
    assert "--bind" in command


def test_platform_command_has_enforceable_network_boundary(tmp_path: Path) -> None:
    policy = SandboxPolicy((tmp_path / "read",), tmp_path / "work")
    system = platform.system()
    if system == "Darwin" and shutil.which("sandbox-exec"):
        command = build_sandbox_command((sys.executable, "-c", "pass"), policy)
        assert command[1] == "-p"
        assert "(deny network*)" in command[2]
    elif system == "Linux" and shutil.which("bwrap"):
        command = build_sandbox_command((sys.executable, "-c", "pass"), policy)
        assert "--unshare-net" in command
        assert "--bind" in command
    else:
        pytest.skip("no platform sandbox is installed")


def test_exact_allowlist_registration(tmp_path: Path) -> None:
    (tmp_path / "plugin.py").write_text("not imported")
    distribution = FakeDistribution(tmp_path)
    entry_point = metadata.EntryPoint("example", "plugin:factory", "paper_agent.providers")
    allowlist = PluginAllowlistEntry(
        "example-plugin", "1.2.3", "example", "plugin:factory", distribution_digest(distribution)
    )
    registration = PluginRegistry((allowlist,), third_party_enabled=True).register_third_party(
        manifest(content_digest=distribution_digest(distribution), third_party=True), entry_point, distribution
    )
    assert registration.third_party is True


def test_declared_signature_must_be_allowlisted(tmp_path: Path) -> None:
    (tmp_path / "plugin.py").write_text("not imported")
    distribution = FakeDistribution(tmp_path)
    distribution.metadata["X-Paper-Agent-Signature"] = "signed-content"
    entry_point = metadata.EntryPoint("example", "plugin:factory", "paper_agent.providers")
    allowlist = PluginAllowlistEntry(
        "example-plugin", "1.2.3", "example", "plugin:factory", distribution_digest(distribution)
    )
    with pytest.raises(PluginRejected, match="signature"):
        PluginRegistry((allowlist,), third_party_enabled=True).register_third_party(
            manifest(content_digest=distribution_digest(distribution), third_party=True), entry_point, distribution
        )
