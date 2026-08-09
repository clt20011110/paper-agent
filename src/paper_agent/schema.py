"""JSON Schema catalog and validation."""

from __future__ import annotations

import json
import sysconfig
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource


class SchemaValidationError(ValueError):
    pass


def schema_directory(override: Path | None = None) -> Path:
    if override is not None:
        return override
    repository_schemas = Path(__file__).resolve().parents[2] / "schemas"
    if repository_schemas.is_dir():
        return repository_schemas
    return Path(sysconfig.get_path("data")) / "share" / "paper-agent" / "schemas"


def schema_registry(root: Path | None = None) -> tuple[Path, Registry]:
    directory = schema_directory(root)
    registry = Registry()
    for path in sorted(directory.glob("*.schema.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        resource = Resource.from_contents(document)
        registry = registry.with_resource(document["$id"], resource)
    return directory, registry


def validate(document: Any, schema_name: str, root: Path | None = None) -> None:
    directory, registry = schema_registry(root)
    schema = json.loads((directory / schema_name).read_text(encoding="utf-8"))
    validator = Draft202012Validator(
        schema,
        registry=registry,
        format_checker=FormatChecker(),
    )
    errors = sorted(validator.iter_errors(document), key=lambda error: list(error.path))
    if errors:
        error = errors[0]
        location = ".".join(str(part) for part in error.path) or "$"
        raise SchemaValidationError(f"{location}: {error.message}")
