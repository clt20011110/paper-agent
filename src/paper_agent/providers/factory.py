"""Manifest-driven factory for core provider adapters."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Mapping

from paper_agent.providers.builtin import (
    BUILTIN_CLASSES,
    create_builtin,
    manifest_from_document,
)


def create_core_provider(
    provider: str,
    transport: Any,
    manifest_document: Mapping[str, Any],
) -> Any:
    """Instantiate an enabled core adapter from its frozen entry point."""

    manifest = manifest_from_document(manifest_document)
    if provider in BUILTIN_CLASSES:
        return create_builtin(provider, transport, manifest)
    if not manifest.enabled or not manifest.builtin:
        raise ValueError(f"{provider} is not an enabled core provider")
    module_name, separator, attribute = str(manifest.entry_point or "").partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError(f"{provider} has an invalid entry point")
    implementation = getattr(import_module(module_name), attribute)
    return implementation(provider, transport, manifest)
