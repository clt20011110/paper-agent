"""Explicit trusted loading of configured Stage 1 implementations."""

import importlib
import re

from .errors import InputError

__all__ = ["load_adapter", "load_enricher", "load_enrichers"]


_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def _validate_path(value: object, prefix: str) -> tuple[str, str]:
    if not isinstance(value, str) or not value or value != value.strip():
        raise InputError("configured implementation path is invalid")
    module_name, separator, attribute_name = value.partition(":")
    parts = module_name.split(".")
    if (
        separator != ":"
        or len(parts) < 2
        or parts[0] != prefix
        or any(_IDENTIFIER_PATTERN.fullmatch(part) is None for part in parts[1:])
        or _IDENTIFIER_PATTERN.fullmatch(attribute_name) is None
    ):
        raise InputError("configured implementation path is invalid")
    return module_name, attribute_name


def _load(path: object, *, prefix: str, role: str) -> object:
    module_name, attribute_name = _validate_path(path, prefix)
    try:
        module = importlib.import_module(f"paper_agent_next.{module_name}")
        implementation = getattr(module, attribute_name)
        if not callable(implementation):
            raise TypeError("configured implementation is not callable")
        instance = implementation()
        source_name = getattr(instance, "source_name")
        if (
            not isinstance(source_name, str)
            or not source_name
            or source_name != source_name.strip()
            or any(character.isspace() for character in source_name)
        ):
            raise TypeError("configured implementation has an invalid source_name")
        method_name = "collect" if role == "adapter" else "enrich"
        if not callable(getattr(instance, method_name)):
            raise TypeError("configured implementation method is not callable")
        return instance
    except (ImportError, AttributeError, TypeError) as error:
        raise InputError(f"could not load configured {role}") from error


def load_adapter(path: str) -> object:
    """Load one catalog-configured adapter through an explicit import path."""

    return _load(path, prefix="adapters", role="adapter")


def load_enricher(path: str) -> object:
    """Load one catalog-configured enricher through an explicit import path."""

    return _load(path, prefix="enrichers", role="enricher")


def load_enrichers(paths: tuple[str, ...]) -> tuple[object, ...]:
    """Load configured enrichers in catalog order."""

    if not isinstance(paths, tuple):
        raise InputError("configured enrichers must be a tuple")
    return tuple(load_enricher(path) for path in paths)
