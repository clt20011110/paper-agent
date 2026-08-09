from __future__ import annotations

from email.message import Message
from importlib import metadata
from pathlib import Path
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


def manifest() -> ProviderManifest:
    return ProviderManifest(
        provider="example",
        version="1.0",
        roles=(ProviderRole.SEARCH,),
        capabilities=(ProviderCapability.METADATA,),
        stable_identifier="example-id",
    )


def test_untrusted_plugins_are_disabled_before_any_import(tmp_path: Path) -> None:
    (tmp_path / "plugin.py").write_text("not imported")
    entry_point = metadata.EntryPoint("example", "plugin:factory", "paper_agent.providers")
    with pytest.raises(PluginRejected, match="disabled"):
        PluginRegistry().register_third_party(manifest(), entry_point, FakeDistribution(tmp_path))


def test_mismatch_rejects_before_entry_point_load(tmp_path: Path) -> None:
    (tmp_path / "plugin.py").write_text("not imported")
    with pytest.raises(PluginRejected, match="exactly allowlisted"):
        PluginRegistry(third_party_enabled=True).register_third_party(manifest(), FakeEntryPoint(), FakeDistribution(tmp_path))


def test_declared_capability_is_enforced() -> None:
    registry = PluginRegistry()
    registry.register_builtin(manifest(), "paper_agent.providers.api:ProviderManifest")
    assert registry.require_capability("example", ProviderCapability.METADATA).manifest.provider == "example"
    with pytest.raises(PluginRejected, match="does not declare"):
        registry.require_capability("example", ProviderCapability.CITATIONS)


def test_subprocess_json_roundtrip(tmp_path: Path) -> None:
    runner = SubprocessProviderRunner(
        (sys.executable, "-c", "import json,sys; print(json.dumps(json.load(sys.stdin)))"),
        cwd=tmp_path,
    )
    assert runner.run({"source_run_id": "run-1", "entries": []}) == {"entries": [], "source_run_id": "run-1"}


def test_exact_allowlist_registration(tmp_path: Path) -> None:
    (tmp_path / "plugin.py").write_text("not imported")
    distribution = FakeDistribution(tmp_path)
    entry_point = metadata.EntryPoint("example", "plugin:factory", "paper_agent.providers")
    allowlist = PluginAllowlistEntry(
        "example-plugin", "1.2.3", "example", "plugin:factory", distribution_digest(distribution)
    )
    registration = PluginRegistry((allowlist,), third_party_enabled=True).register_third_party(
        manifest(), entry_point, distribution
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
        PluginRegistry((allowlist,), third_party_enabled=True).register_third_party(manifest(), entry_point, distribution)
