"""Verify an installed wheel without relying on the source checkout."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
from tempfile import TemporaryDirectory

from paper_agent import __version__
from paper_agent.analysis import ANALYSIS_PROMPT, PaperAnalysisCoordinator
from paper_agent.analysis_registry import registry_directory
from paper_agent.artifacts import ArtifactStore
from paper_agent.authorized_skill_runtime import audit_manifest_path
from paper_agent.codex_exec import prompt_directory
from paper_agent.domain import QuerySpec
from paper_agent.manifests import load_catalog, manifest_directory
from paper_agent.processing import ArtifactProcessingPolicy, ProcessingGate
from paper_agent.providers.builtin import FixtureTransport, create_builtin
from paper_agent.resources import (
    example_config_paths,
    paper_agent_skill_directory,
    public_oa_terms_path,
    release_asset_root,
    stage2_model_lock_paths,
)
from paper_agent.report_artifacts import audit_rubric_hash
from paper_agent.schema import schema_directory
from paper_agent.storage import Database


EXPECTED_SCHEMA_VERSION = 26


def main() -> None:
    assert not (Path.cwd() / "pyproject.toml").exists()
    console = Path(sys.executable).with_name("paper-agent")
    completed = subprocess.run(
        [str(console), "doctor"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    for arguments in (
        ("stage2-evaluator", "promote", "--help"),
        ("stage2-evaluator", "attest", "--help"),
        ("stage2-rationale", "run-source", "--help"),
        ("stage2-rationale", "derive-examples", "--help"),
        ("stage2-rationale", "freeze-worklist", "--help"),
        ("stage2-rationale", "import-worklist", "--help"),
        ("stage2-parity", "freeze-workload", "--help"),
        ("stage2-parity", "run", "--help"),
        ("stage2-tuning", "select", "--help"),
        ("stage2-release", "build-evidence", "--help"),
        ("stage2-release", "assemble", "--help"),
    ):
        assert subprocess.run([str(console), *arguments], check=False).returncode == 0
    diagnosis = json.loads(completed.stdout)
    catalog = load_catalog()
    manifest_root = manifest_directory().resolve()
    for acceptance in catalog.acceptances.values():
        for route in acceptance.get("transport_fixture_routes", ()):
            fixture = (manifest_root / route["path"]).resolve()
            fixture.relative_to(manifest_root)
            assert fixture.is_file()
            assert sha256(fixture.read_bytes()).hexdigest() == route["sha256"]
    provider = create_builtin(
        "crossref",
        FixtureTransport(
            {
                "crossref:search:first": {
                    "message": {
                        "items": [
                            {
                                "DOI": "10.1000/wheel-check",
                                "title": ["Wheel install check"],
                            }
                        ]
                    }
                }
            }
        ),
    )
    batch = provider.search(QuerySpec(1, "wheel-check", "fixture", page_size=1))

    checks = {item["name"]: item for item in diagnosis["checks"]}
    assert checks["python"]["status"] == "pass"
    assert checks["stage2_model_locks"]["status"] == "pass"
    assert diagnosis["status"] == "ready"
    assert len(catalog.providers) == 25
    assert len(catalog.venues) == 20
    assert batch.entries[0].external_id == "10.1000/wheel-check"
    schemas = schema_directory()
    for name in (
        "stage2-hidden-evaluator-attestation.schema.json",
        "stage2-parity-workload.schema.json",
        "stage2-parity-oracle-trust.schema.json",
        "stage2-rationale-query-metadata.schema.json",
        "stage2-rationale-source-ledger.schema.json",
        "stage2-release-evidence.schema.json",
        "stage2-tuning-selection-input.schema.json",
        "stage2-tuning-winner.schema.json",
    ):
        assert (schemas / name).is_file()
    assert len(tuple(prompt_directory().glob("*.md"))) >= 7
    assert (registry_directory() / "analysis-normalization-v1.yaml").is_file()
    assert audit_manifest_path().is_file()
    assert len(audit_rubric_hash()) == 64
    assets = release_asset_root()
    assert assets == (
        Path(sysconfig.get_path("data"))
        / "share"
        / "paper-agent"
        / __version__
    )
    locks = stage2_model_lock_paths()
    configs = example_config_paths()
    assert all(path.is_file() for path in locks)
    assert all(path.is_file() for path in configs)
    for path in (
        assets / "configs/stage2/challengers.json",
        assets / "configs/stage2/models/bge-reranker-v2-m3-mlx-bf16.lock.json",
    ):
        assert path.is_file()
    assert public_oa_terms_path().is_file()
    skill = paper_agent_skill_directory()
    assert (skill / "SKILL.md").is_file()
    assert (skill / "agents" / "openai.yaml").is_file()
    with TemporaryDirectory() as directory:
        root = Path(directory)
        source_config = next(path for path in configs if path.name == "smoke_supported.yaml")
        copied_config = shutil.copy2(source_config, root / "research.yaml")
        copied_skill = shutil.copytree(skill, root / "skills" / "paper-agent")
        assert Path(copied_config).read_bytes() == source_config.read_bytes()
        assert (copied_skill / "SKILL.md").read_bytes() == (skill / "SKILL.md").read_bytes()
        with Database(root / "wheel.sqlite3") as database:
            database.migrate()
            assert database.current_version() == EXPECTED_SCHEMA_VERSION
            policy_path = (
                Path(sysconfig.get_path("data"))
                / "share"
                / "paper-agent"
                / "policies"
                / "artifact-processing-v1.yaml"
            )
            coordinator = PaperAnalysisCoordinator(
                database,
                ArtifactStore(root / "stage4-artifacts"),
                ProcessingGate(ArtifactProcessingPolicy.load(policy_path)),
                invoker_factory=lambda: (_ for _ in ()).throw(
                    AssertionError("wheel verification must not dispatch Codex")
                ),
            )
            assert coordinator.prompt_hash == sha256(
                (prompt_directory() / ANALYSIS_PROMPT).read_bytes()
            ).hexdigest()
            assert len(coordinator.schema_hash) == 64


if __name__ == "__main__":
    main()
