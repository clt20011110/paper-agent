"""Strict JSON worker for an attested third-party provider entry point."""

from __future__ import annotations

from contextlib import redirect_stdout
from importlib import import_module, metadata
import json
import sys
from typing import Any, Mapping

from paper_agent.domain import CitationBatch, Paper, QuerySpec, SourceBatch

from .api import CrawlWindow, SeedInput, VenueDescriptor
from .plugins import PluginRegistration, attest_registration


_PROTOCOL_VERSION = 1


def _load_entry_point(value: str) -> Any:
    module_name, separator, attribute = value.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("entry point must use module:attribute syntax")
    target: Any = import_module(module_name)
    for component in attribute.split("."):
        target = getattr(target, component)
    return target


def _attested_registration(arguments: list[str]) -> PluginRegistration:
    if len(arguments) != 6:
        raise SystemExit(
            "usage: worker distribution version provider entry_point artifact_sha256 signature"
        )
    distribution_name, version, provider, entry_point, digest, signature = arguments
    # Construct only the immutable installed facts needed by attest_registration.
    # The manifest roles/capabilities are enforced by the parent registry before
    # this process is launched; no plugin module has been imported at this point.
    from paper_agent.domain import ProviderCapability, ProviderRole
    from .api import ProviderManifest

    registration = PluginRegistration(
        ProviderManifest(
            provider=provider,
            version=version,
            roles=(ProviderRole.SEARCH,),
            capabilities=(ProviderCapability.METADATA,),
            stable_identifier=f"{provider}:external_id",
            distribution=distribution_name,
            entry_point=entry_point,
            artifact_sha256=digest,
            builtin=False,
        ),
        entry_point,
        distribution_name,
        version,
        digest,
        signature or None,
    )
    attest_registration(registration)
    return registration


def _invoke(handler: Any, operation: str, arguments: Mapping[str, Any]) -> SourceBatch | CitationBatch:
    if operation == "search":
        return handler.search(
            QuerySpec.from_dict(_object(arguments, "query_spec")),
            _optional_string(arguments.get("cursor")),
        )
    if operation == "discover":
        descriptor = VenueDescriptor(**_object(arguments, "descriptor"))
        window = CrawlWindow(**_object(arguments, "window"))
        return handler.discover(descriptor, window, _optional_string(arguments.get("cursor")))
    if operation == "import_seeds":
        raw_inputs = arguments.get("input_spec")
        if not isinstance(raw_inputs, list) or any(not isinstance(item, Mapping) for item in raw_inputs):
            raise ValueError("input_spec must be an array of objects")
        return handler.import_seeds(tuple(SeedInput(**dict(item)) for item in raw_inputs))
    if operation in {"references", "citations"}:
        method = getattr(handler, operation)
        return method(
            Paper.from_dict(_object(arguments, "seed")),
            _optional_string(arguments.get("cursor")),
        )
    raise ValueError(f"unsupported provider operation: {operation}")


def _object(document: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = document.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be an object")
    return value


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("cursor must be a string or null")
    return value


def main(argv: list[str] | None = None) -> int:
    registration = _attested_registration(list(argv or sys.argv[1:]))
    payload = json.load(sys.stdin)
    if not isinstance(payload, Mapping):
        raise ValueError("provider request must be an object")
    if payload.get("protocol_version") != _PROTOCOL_VERSION:
        raise ValueError("unsupported provider IPC protocol")
    if payload.get("provider") != registration.manifest.provider:
        raise ValueError("provider request does not match the attested entry point")
    operation = payload.get("operation")
    arguments = payload.get("arguments")
    if not isinstance(operation, str) or not isinstance(arguments, Mapping):
        raise ValueError("provider request requires operation and arguments")

    # Plugin import and any accidental prints are kept away from the JSON stdout
    # channel. The target is imported only after the child independently repeats
    # distribution/version/entry-point/content attestation above.
    output = sys.stdout
    with redirect_stdout(sys.stderr):
        factory = _load_entry_point(registration.entry_point)
        handler = factory()
        result = _invoke(handler, operation, arguments)
    if not isinstance(result, (SourceBatch, CitationBatch)):
        raise TypeError("provider operation must return SourceBatch or CitationBatch")
    json.dump(
        {
            "protocol_version": _PROTOCOL_VERSION,
            "result_type": "source_batch" if isinstance(result, SourceBatch) else "citation_batch",
            "result": result.to_dict(),
        },
        output,
        sort_keys=True,
        separators=(",", ":"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
