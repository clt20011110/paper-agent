from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import tomllib

import pytest

import paper_agent.analysis as analysis_module
from paper_agent import __version__
from paper_agent.analysis import ANALYSIS_PROMPT, PaperAnalysisCoordinator
from paper_agent.artifacts import ArtifactStore
from paper_agent.cli import doctor
from paper_agent.domain import QuerySpec
from paper_agent.manifests import load_catalog
from paper_agent.processing import ArtifactProcessingPolicy, ProcessingGate
from paper_agent.providers.builtin import FixtureTransport, create_builtin
from paper_agent.resources import (
    example_config_paths,
    paper_agent_skill_directory,
    public_oa_terms_path,
    release_asset_root,
    stage2_model_lock_paths,
)
from paper_agent.storage import Database
from paper_agent.stage2_hidden_attestation import (
    HiddenPromotionAttestationError,
    hidden_evaluator_trust_from_document,
)


ROOT = Path(__file__).parents[1]


def test_runtime_data_and_builtin_work_outside_repository_cwd(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    catalog = load_catalog()
    provider = create_builtin(
        "crossref",
        FixtureTransport(
            {
                "crossref:search:first": {
                    "message": {
                        "items": [
                            {
                                "DOI": "10.1000/package-check",
                                "title": ["Installed package check"],
                            }
                        ]
                    }
                }
            }
        ),
    )
    batch = provider.search(QuerySpec(1, "package-check", "fixture", page_size=1))

    assert len(catalog.venues) == 20
    assert batch.entries[0].external_id == "10.1000/package-check"
    assert doctor()["python_supported"] is True


def test_console_script_uses_the_structured_error_boundary() -> None:
    project = tomllib.loads(
        (Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    )

    assert project["project"]["scripts"]["paper-agent"] == "paper_agent.cli:entrypoint"


def test_release_assets_use_one_versioned_source_and_wheel_layout() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    data_files = project["tool"]["setuptools"]["data-files"]
    installed_root = f"share/paper-agent/{__version__}"

    assert project["project"]["version"] == __version__
    assert data_files[installed_root] == ["example_config.yaml"]
    assert data_files[f"{installed_root}/configs"] == [
        "configs/*.yaml",
        "configs/*.json",
    ]
    assert data_files[f"{installed_root}/configs/stage2"] == [
        "configs/stage2/*.json"
    ]
    assert data_files[f"{installed_root}/configs/stage2/models"] == [
        "configs/stage2/models/*.json"
    ]
    assert data_files[f"{installed_root}/skills/paper-agent"] == [
        "skills/paper-agent/SKILL.md"
    ]
    assert data_files[f"{installed_root}/skills/paper-agent/agents"] == [
        "skills/paper-agent/agents/*.yaml"
    ]

    assert release_asset_root() == ROOT
    assert all(path.is_file() for path in stage2_model_lock_paths())
    trust_example = (
        release_asset_root() / "configs/stage2/hidden-evaluator-trust.example.json"
    )
    assert trust_example.is_file()
    with pytest.raises(HiddenPromotionAttestationError, match="no active key"):
        hidden_evaluator_trust_from_document(
            json.loads(trust_example.read_text(encoding="utf-8"))
        )
    assert all(path.is_file() for path in example_config_paths())
    assert public_oa_terms_path().is_file()
    assert (paper_agent_skill_directory() / "SKILL.md").is_file()
    assert (
        paper_agent_skill_directory() / "agents" / "openai.yaml"
    ).is_file()


def test_stage4_coordinator_uses_the_runtime_prompt_locator(
    tmp_path: Path, monkeypatch,
) -> None:
    prompt_root = tmp_path / "installed-data" / "prompts"
    prompt_root.mkdir(parents=True)
    prompt = b"installed wheel Stage 4 prompt\n"
    (prompt_root / ANALYSIS_PROMPT).write_bytes(prompt)
    monkeypatch.setattr(analysis_module, "prompt_directory", lambda: prompt_root)

    with Database(tmp_path / "papers.sqlite3") as database:
        database.migrate()
        coordinator = PaperAnalysisCoordinator(
            database,
            ArtifactStore(tmp_path / "artifacts"),
            ProcessingGate(
                ArtifactProcessingPolicy.load(
                    ROOT / "policies" / "artifact-processing-v1.yaml"
                )
            ),
            invoker_factory=lambda: (_ for _ in ()).throw(
                AssertionError("coordinator construction must not dispatch Codex")
            ),
        )

    assert coordinator.prompt_hash == sha256(prompt).hexdigest()
