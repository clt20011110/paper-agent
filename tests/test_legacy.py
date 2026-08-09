from pathlib import Path

from paper_agent.legacy import migrate_legacy_config, migrate_legacy_yaml, write_migrated
from paper_agent.schema import validate


SCHEMAS = Path(__file__).parents[1] / "schemas"


def test_migrates_legacy_configuration_to_schema_valid_v2(tmp_path: Path) -> None:
    report = migrate_legacy_config(
        {
            "output_dir": "./research",
            "topic": "Chip design",
            "sources": {
                "conferences": [
                    {"name": "ICLR", "years": [2024], "platform": "openreview"},
                    {"name": "TCAD", "years": [2024], "platform": "dblp_tcad"},
                ],
                "journals": [
                    {
                        "name": "Nature Computer Science",
                        "years": [2024],
                        "platform": "nature_computer_science",
                    }
                ],
                "arxiv": {
                    "enabled": True,
                    "categories": ["cs.AI"],
                    "date_range": "2024-01-01 to 2024-12-31",
                    "save_to_database": True,
                },
            },
            "database": {"format": "json", "path": "./research/papers.json"},
            "filter": {
                "mode": "hybrid",
                "regex": {"include_groups": [["chip", "design"]], "exclude": ["review"]},
                "semantic": {"model": "old"},
            },
            "analysis": {
                "model": "openrouter/legacy-model",
                "summary_model": "openrouter/legacy-summary",
                "workers": 7,
            },
        },
        SCHEMAS,
    )

    converted = report.converted_config
    validate(converted, "config-v2.schema.json", SCHEMAS)
    assert converted["analysis"]["model"] == "gpt-5.6-luna"
    assert converted["summary"]["model"] == "gpt-5.6-sol"
    assert converted["analysis"]["workers"] == 7
    assert converted["sources"]["plan_defaults"]["arxiv"]["include_arxiv_candidates"] is True
    assert converted["sources"]["plan_defaults"]["venues"][1] == {
        "descriptor": "venues/ieee_tcad.yaml",
        "date_from": "2024-01-01",
        "date_to": "2024-12-31",
    }
    assert converted["sources"]["plan_defaults"]["venues"][2]["descriptor"] == (
        "venues/nature_computational_science.yaml"
    )
    assert report.field_mappings["sources.arxiv.save_to_database"].endswith(
        "include_arxiv_candidates"
    )
    assert "filter.semantic" in report.unmigrated
    assert any("OpenRouter" in warning for warning in report.warnings)


def test_unversioned_config_can_be_written_after_dry_run(tmp_path: Path) -> None:
    source = tmp_path / "legacy.yaml"
    destination = tmp_path / "migrated.yaml"
    source.write_text("topic: Testing\noutput_dir: ./out\n", encoding="utf-8")

    report = migrate_legacy_yaml(source, SCHEMAS)
    write_migrated(report, destination)
    persisted = migrate_legacy_yaml(source, SCHEMAS)

    assert destination.exists()
    assert persisted.converted_config == report.converted_config
