import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from paper_agent.config import ConfigError, load_config
from paper_agent.schema import SchemaValidationError, validate


SCHEMAS = Path(__file__).parents[1] / "schemas"


def test_every_schema_is_valid_draft_2020_12() -> None:
    paths = sorted(SCHEMAS.glob("*.schema.json"))
    assert len(paths) == 38
    for path in paths:
        Draft202012Validator.check_schema(json.loads(path.read_text(encoding="utf-8")))


def test_schema_catalog_resolves_relative_references() -> None:
    approval = {
        "approved_hash": "a" * 64,
        "approved_by": "owner",
        "approved_at": "2026-08-09T00:00:00Z",
        "approval_method": "cli_hash",
    }
    validate(approval, "plan-approval.schema.json", SCHEMAS)


def test_schema_error_has_document_location() -> None:
    with pytest.raises(SchemaValidationError, match="approved_hash"):
        validate(
            {
                "approved_hash": "bad",
                "approved_by": "owner",
                "approved_at": "2026-08-09T00:00:00Z",
                "approval_method": "cli_hash",
            },
            "plan-approval.schema.json",
            SCHEMAS,
        )


def test_legacy_and_openrouter_configs_require_migration(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy.yaml"
    legacy.write_text("version: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="migrate-config"):
        load_config(legacy, SCHEMAS)

    openrouter = tmp_path / "openrouter.yaml"
    openrouter.write_text("version: 2\nprovider: openrouter\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="OpenRouter"):
        load_config(openrouter, SCHEMAS)
