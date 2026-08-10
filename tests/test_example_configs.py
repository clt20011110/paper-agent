from pathlib import Path

from paper_agent.config import load_config


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
