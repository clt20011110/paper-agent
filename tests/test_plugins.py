from __future__ import annotations

from dataclasses import replace
from email.message import Message
from hashlib import sha256
from importlib import metadata
from pathlib import Path
import platform
import shutil
import socket
import sys

import pytest

from paper_agent.domain import (
    CitationBatch,
    CitationEdge,
    CitationEdgeType,
    EnvelopeStatus,
    Paper,
    ProviderCapability,
    ProviderRole,
    QuerySpec,
    SourceEntry,
)
from paper_agent.providers.api import ProviderManifest
from paper_agent.providers.plugins import (
    IsolatedProviderClient,
    PluginAllowlistEntry,
    PluginExecutionError,
    PluginRegistration,
    PluginRejected,
    PluginRegistry,
    SubprocessProviderRunner,
    attest_registration,
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


@pytest.mark.enable_socket
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
    script = (
        "from pathlib import Path\n"
        "import json,socket,sys\n"
        "payload=json.load(sys.stdin)\n"
        "declared=Path(payload['declared']).read_text()\n"
        "try:\n Path(payload['secret']).read_text(); secret_blocked=False\n"
        "except OSError:\n secret_blocked=True\n"
        "try:\n socket.create_connection(('127.0.0.1',payload['port']),timeout=1).close(); network_blocked=False\n"
        "except OSError:\n network_blocked=True\n"
        "json.dump({'declared':declared,'secret_blocked':secret_blocked,'network_blocked':network_blocked},sys.stdout)\n"
    )
    runner = SubprocessProviderRunner((sys.executable, "-c", script), cwd=tmp_path)
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
    assert command[:7] == (
        "/usr/bin/bwrap",
        "--die-with-parent",
        "--new-session",
        "--unshare-all",
        "--cap-drop",
        "ALL",
        "--proc",
    )
    assert ("--ro-bind", str(tmp_path / "read"), str(tmp_path / "read")) == command[10:13]
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
        assert "--unshare-all" in command
        assert ("--cap-drop", "ALL") == command[4:6]
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


def _install_test_distribution(tmp_path: Path, source: str) -> metadata.Distribution:
    (tmp_path / "example_plugin.py").write_text(source, encoding="utf-8")
    metadata_root = tmp_path / "example_plugin-1.2.3.dist-info"
    metadata_root.mkdir()
    (metadata_root / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: example-plugin\nVersion: 1.2.3\n",
        encoding="utf-8",
    )
    (metadata_root / "entry_points.txt").write_text(
        "[paper_agent.providers]\nexample = example_plugin:factory\n",
        encoding="utf-8",
    )
    (metadata_root / "RECORD").write_text(
        "example_plugin.py,,\n"
        "example_plugin-1.2.3.dist-info/METADATA,,\n"
        "example_plugin-1.2.3.dist-info/entry_points.txt,,\n"
        "example_plugin-1.2.3.dist-info/RECORD,,\n",
        encoding="utf-8",
    )
    return next(metadata.distributions(path=[str(tmp_path)], name="example-plugin"))


def _installed_manifest(distribution: metadata.Distribution) -> ProviderManifest:
    return replace(
        manifest(
            content_digest=distribution_digest(distribution),
            third_party=True,
        ),
        entry_point="example_plugin:factory",
    )


def test_allowlist_version_and_content_drift_are_rejected_before_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = tmp_path / "imported"
    distribution = _install_test_distribution(
        tmp_path,
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('imported')\ndef factory(): return object()\n",
    )
    digest = distribution_digest(distribution)
    trusted = _installed_manifest(distribution)
    allow = PluginAllowlistEntry("example-plugin", "1.2.3", "example", "example_plugin:factory", digest)
    monkeypatch.setattr("paper_agent.providers.plugins.metadata.distribution", lambda _name: distribution)

    version_drift = PluginAllowlistEntry(
        "example-plugin", "1.2.4", "example", "example_plugin:factory", digest
    )
    with pytest.raises(PluginRejected, match="allowlisted|manifest"):
        PluginRegistry((version_drift,)).verify_requested({"example": trusted}, ("example",))
    assert not marker.exists()

    (tmp_path / "example_plugin.py").write_text("raise AssertionError('must not import')\n", encoding="utf-8")
    with pytest.raises(PluginRejected, match="manifest"):
        PluginRegistry((allow,)).verify_requested({"example": trusted}, ("example",))
    assert not marker.exists()


def test_attestation_rechecks_content_immediately_before_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    distribution = _install_test_distribution(tmp_path, "def factory(): return object()\n")
    digest = distribution_digest(distribution)
    registration = PluginRegistration(
        _installed_manifest(distribution),
        "example_plugin:factory",
        "example-plugin",
        "1.2.3",
        digest,
    )
    monkeypatch.setattr("paper_agent.providers.plugins.metadata.distribution", lambda _name: distribution)
    assert attest_registration(registration) is distribution

    (tmp_path / "example_plugin.py").write_text("def factory(): return 'changed'\n", encoding="utf-8")
    with pytest.raises(PluginRejected, match="drifted"):
        attest_registration(registration)


def test_distribution_inventory_rejects_symlinked_content(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-plugin.py"
    outside.write_text("def factory(): return object()\n", encoding="utf-8")
    distribution = _install_test_distribution(tmp_path, "def factory(): return object()\n")
    (tmp_path / "example_plugin.py").unlink()
    (tmp_path / "example_plugin.py").symlink_to(outside)

    with pytest.raises(PluginRejected, match="symlink"):
        distribution_digest(distribution)


def test_verified_staging_excludes_unlisted_adjacent_modules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = (
        "try:\n from adjacent_helper import TITLE\n"
        "except ModuleNotFoundError:\n TITLE='inventory-only'\n"
        "from paper_agent.domain import EnvelopeStatus,SourceBatch,SourceEntry\n"
        "class Provider:\n"
        " def search(self,spec,cursor):\n"
        "  return SourceBatch('plugin-run',spec.native_query_hash,(SourceEntry('example','x',TITLE),),None,EnvelopeStatus.SUCCESS)\n"
        "def factory(): return Provider()\n"
    )
    distribution = _install_test_distribution(tmp_path, source)
    helper = tmp_path / "adjacent_helper.py"
    helper.write_text("TITLE='unlisted-before'\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    digest = distribution_digest(distribution)
    registry = PluginRegistry(
        (PluginAllowlistEntry("example-plugin", "1.2.3", "example", "example_plugin:factory", digest),)
    )
    registry.verify_requested({"example": _installed_manifest(distribution)}, ("example",))
    monkeypatch.setattr(
        "paper_agent.providers.plugins.build_sandbox_command",
        lambda command, _policy: command,
    )
    client = IsolatedProviderClient(registry, "example")

    assert client.search(QuerySpec(1, "rq", "q", native_query_hash="a" * 64)).entries[0].title == "inventory-only"
    helper.write_text("TITLE='DRIFT EXECUTED'\n", encoding="utf-8")
    assert distribution_digest(distribution) == digest
    assert client.search(QuerySpec(1, "rq", "q", native_query_hash="a" * 64)).entries[0].title == "inventory-only"


def test_citation_ipc_binds_query_direction_and_nested_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    citation_manifest = replace(
        manifest(),
        roles=(ProviderRole.CITATION,),
        capabilities=(
            ProviderCapability.METADATA,
            ProviderCapability.REFERENCES,
            ProviderCapability.CITATIONS,
        ),
    )
    registry = PluginRegistry()
    registry.registrations["example"] = PluginRegistration(
        citation_manifest,
        "plugin:factory",
        "example-plugin",
        "1.2.3",
        "a" * 64,
    )
    client = IsolatedProviderClient(registry, "example")
    seed = Paper("seed", "Seed")
    query_hash = sha256(b"seed:references").hexdigest()
    forged = CitationBatch(
        "forged-run",
        query_hash,
        (
            CitationEdge(
                "seed",
                "candidate",
                CitationEdgeType.REFERENCES,
                "example",
                "",
                candidate=SourceEntry("FORGED_PROVIDER", "candidate", "Candidate"),
            ),
        ),
        None,
        EnvelopeStatus.SUCCESS,
    )
    monkeypatch.setattr(
        registry,
        "dispatch",
        lambda *_args, **_kwargs: {
            "protocol_version": 1,
            "result_type": "citation_batch",
            "result": forged.to_dict(),
        },
    )
    with pytest.raises(PluginExecutionError, match="foreign provenance"):
        client.references(seed)

    wrong_hash = replace(
        forged,
        query_hash="b" * 64,
        entries=(replace(forged.entries[0], candidate=SourceEntry("example", "candidate", "Candidate")),),
    )
    monkeypatch.setattr(
        registry,
        "dispatch",
        lambda *_args, **_kwargs: {
            "protocol_version": 1,
            "result_type": "citation_batch",
            "result": wrong_hash.to_dict(),
        },
    )
    with pytest.raises(PluginExecutionError, match="query hash"):
        client.references(seed)


def test_runner_rejects_oversized_ipc_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("paper_agent.providers.plugins._MAX_IPC_BYTES", 32)
    monkeypatch.setattr(
        "paper_agent.providers.plugins.build_sandbox_command",
        lambda command, _policy: command,
    )
    runner = SubprocessProviderRunner(
        (sys.executable, "-c", "import json; print(json.dumps({'value':'x'*64}))"),
        cwd=tmp_path,
    )
    with pytest.raises(PluginExecutionError, match="IPC limit"):
        runner.run({})


def test_isolated_search_provider_ipc_success_failure_and_revalidation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = (
        "from paper_agent.domain import EnvelopeStatus,SourceBatch,SourceEntry\n"
        "class Provider:\n"
        " def search(self,spec,cursor):\n"
        "  if spec.original_query == 'fail': raise RuntimeError('secret plugin detail')\n"
        "  entry=SourceEntry('example','external-1','Isolated result',('Ada',),year=2026)\n"
        "  return SourceBatch('plugin-run',spec.native_query_hash,(entry,),None,EnvelopeStatus.SUCCESS)\n"
        "def factory(): return Provider()\n"
    )
    distribution = _install_test_distribution(tmp_path, source)
    monkeypatch.syspath_prepend(str(tmp_path))
    digest = distribution_digest(distribution)
    allow = PluginAllowlistEntry("example-plugin", "1.2.3", "example", "example_plugin:factory", digest)
    registry = PluginRegistry((allow,))
    registry.verify_requested({"example": _installed_manifest(distribution)}, ("example",))
    monkeypatch.setattr(
        "paper_agent.providers.plugins.build_sandbox_command",
        lambda command, _policy: command,
    )
    client = IsolatedProviderClient(registry, "example")

    batch = client.search(QuerySpec(1, "rq", "ok", native_query_hash="a" * 64))
    assert batch.entries[0].title == "Isolated result"
    assert batch.entries[0].provider == "example"

    with pytest.raises(PluginExecutionError, match="exit status") as failure:
        client.search(QuerySpec(1, "rq", "fail", native_query_hash="b" * 64))
    assert "secret plugin detail" not in str(failure.value)

    (tmp_path / "example_plugin.py").write_text(source + "# drift\n", encoding="utf-8")
    with pytest.raises(PluginRejected, match="drifted"):
        client.search(QuerySpec(1, "rq", "ok", native_query_hash="c" * 64))


def test_isolated_provider_runs_in_the_real_platform_sandbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    system = platform.system()
    available = (system == "Darwin" and shutil.which("sandbox-exec")) or (
        system == "Linux" and shutil.which("bwrap")
    )
    if not available:
        pytest.skip("no platform sandbox is installed")
    source = (
        "from paper_agent.domain import EnvelopeStatus,SourceBatch,SourceEntry\n"
        "class Provider:\n"
        " def search(self,spec,cursor):\n"
        "  return SourceBatch('plugin-run',spec.native_query_hash,(SourceEntry('example','x','Sandboxed'),),None,EnvelopeStatus.SUCCESS)\n"
        "def factory(): return Provider()\n"
    )
    distribution = _install_test_distribution(tmp_path, source)
    monkeypatch.syspath_prepend(str(tmp_path))
    digest = distribution_digest(distribution)
    registry = PluginRegistry(
        (PluginAllowlistEntry("example-plugin", "1.2.3", "example", "example_plugin:factory", digest),)
    )
    registry.verify_requested({"example": _installed_manifest(distribution)}, ("example",))

    batch = IsolatedProviderClient(registry, "example").search(
        QuerySpec(1, "rq", "q", native_query_hash="d" * 64)
    )

    assert batch.entries[0].title == "Sandboxed"
