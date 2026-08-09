"""Versioned YAML configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .schema import SchemaValidationError, validate


class ConfigError(ValueError):
    pass


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ConfigError("configuration root must be an object")
    return value


def load_config(path: Path, schema_root: Path | None = None) -> dict[str, Any]:
    document = load_yaml(path)
    if document.get("version") != 2:
        raise ConfigError("legacy configuration requires paper-agent migrate-config")
    if _contains_openrouter(document):
        raise ConfigError("OpenRouter is unsupported; use the frozen codex exec profiles")
    try:
        validate(document, "config-v2.schema.json", schema_root)
    except SchemaValidationError as error:
        raise ConfigError(str(error)) from error
    return document


def _contains_openrouter(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            "openrouter" in str(key).lower() or _contains_openrouter(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_contains_openrouter(child) for child in value)
    return isinstance(value, str) and "openrouter" in value.lower()
