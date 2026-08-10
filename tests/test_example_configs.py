from pathlib import Path

from paper_agent import cli
from paper_agent.config import load_config
from paper_agent.doctor import DoctorPaths, SystemDoctor
from paper_agent.download_cli_service import _policy_path as download_policy_path
from paper_agent.manifests import load_catalog


ROOT = Path(__file__).parents[1]
EXAMPLES = (
    ROOT / "example_config.yaml",
    ROOT / "configs" / "abstract_focus.yaml",
    ROOT / "configs" / "journal_smoke.yaml",
    ROOT / "configs" / "smoke_supported.yaml",
)


def test_v2_example_configs_validate_and_use_frozen_model_routes() -> None:
    for path in EXAMPLES:
        config = load_config(path)

        assert config["version"] == 2
        assert config["filter"]["reranker"]["backend"] == "omlx_rerank"
        assert config["filter"]["adjudicator"]["backend"] == "omlx_chat"
        assert config["download"]["authorized_skill"]["profile"] == "stage3_authorized_luna"
        assert config["download"]["authorized_skill"]["codex_model"] == "gpt-5.6-luna"
        assert config["analysis"]["profile"] == "stage4_analysis_luna"
        assert config["analysis"]["model"] == "gpt-5.6-luna"
        assert config["summary"]["profile"] == "stage4b_summary_sol"
        assert config["summary"]["model"] == "gpt-5.6-sol"


def test_example_configs_are_templates_without_credentials_or_retired_runtime_names() -> None:
    for path in EXAMPLES:
        text = path.read_text(encoding="utf-8").lower()

        assert "openrouter" not in text
        assert "opencode" not in text
        assert "api_key:" not in text
        assert "authorization_grant_id: null" in text
        assert "processing_grant_id: null" in text


def test_example_configs_reference_shipped_descriptors_and_contracts() -> None:
    for path in EXAMPLES:
        config = load_config(path)
        references = [
            *(venue["descriptor"] for venue in config["sources"]["plan_defaults"]["venues"]),
            config["download"]["policy_matrix"],
            config["analysis"]["output_schema"],
            config["analysis"]["remote_model_processing"]["policy_matrix"],
            config["summary"]["remote_model_processing"]["policy_matrix"],
            config["summary"]["final_audit"]["rubric"],
            *config["summary"]["schemas"].values(),
            *config["summary"]["prompts"].values(),
        ]

        assert all((ROOT / reference).is_file() for reference in references)


def test_example_configs_doctor_resolves_descriptor_primary_providers(tmp_path) -> None:
    catalog = load_catalog(ROOT)
    for path in EXAMPLES:
        config = load_config(path)
        report = SystemDoctor(
            DoctorPaths(
                repository_root=ROOT,
                config_path=path,
                database_path=tmp_path / f"{path.stem}.sqlite3",
            ),
            environment={},
            executable_finder=lambda _: None,
        ).run()

        primary_providers = {
            catalog.venue(Path(venue["descriptor"]).stem)["primary_provider"]
            for venue in config["sources"]["plan_defaults"]["venues"]
        }
        assert all(
            any(
                check.name == f"provider:{provider}"
                or check.name.startswith(f"provider:{provider}:")
                for check in report.checks
            )
            for provider in primary_providers
        )
        assert not any(
            check.status == "blocker"
            and (check.name == "provider_roles" or check.name.endswith(":roles"))
            for check in report.checks
        )


def test_query_draft_example_compiles_without_writing(tmp_path, capsys) -> None:
    assert cli.main([
        "search", "plan",
        "--input", str(ROOT / "configs" / "query_draft.example.yaml"),
        "--output-root", str(tmp_path / "output"),
        "--dry-run",
    ]) == 0
    assert '"status":"validated"' in capsys.readouterr().out
    assert not (tmp_path / "output").exists()


def test_relocated_config_resolves_shipped_processing_policies(tmp_path) -> None:
    source = ROOT / "configs" / "smoke_supported.yaml"
    relocated = tmp_path / "research.yaml"
    relocated.write_bytes(source.read_bytes())
    config = load_config(relocated)

    assert download_policy_path(tmp_path, config["download"]) == (
        ROOT / "policies" / "download-access-v1.yaml"
    )
    assert cli._analysis_policy_path(config, relocated) == (
        ROOT / "policies" / "artifact-processing-v1.yaml"
    )
    assert cli._report_policy_path(config, relocated) == (
        ROOT / "policies" / "artifact-processing-v1.yaml"
    )
