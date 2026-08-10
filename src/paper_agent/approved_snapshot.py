"""Strict replay of approved, provider-specific metadata response bundles."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Callable, Mapping
from xml.etree import ElementTree

from .canonical import content_hash
from .http_transport import ControlledHTTPTransport, _access_payload, _enrichment_payload
from .provider_runtime import BulkSnapshot, ProviderRequestError, ProviderRuntime


BUNDLE_SCHEMA_VERSION = "1"


class MetadataSnapshotError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SnapshotResponse:
    operation: str
    parameters_hash: str
    cursor: str | None
    content_type: str
    body: bytes
    body_sha256: str


@dataclass(frozen=True, slots=True)
class MetadataSnapshotBundle:
    provider: str
    responses: Mapping[tuple[str, str, str | None], SnapshotResponse]

    @classmethod
    def load(cls, path: Path, expected_sha256: str) -> "MetadataSnapshotBundle":
        raw_bundle = path.read_bytes()
        if sha256(raw_bundle).hexdigest() != expected_sha256:
            raise MetadataSnapshotError("snapshot bundle hash does not match the approved QueryPlan")
        try:
            document = json.loads(raw_bundle)
        except json.JSONDecodeError as error:
            raise MetadataSnapshotError("snapshot bundle must be JSON") from error
        if (
            not isinstance(document, Mapping)
            or set(document) != {"schema_version", "provider", "responses"}
            or document.get("schema_version") != BUNDLE_SCHEMA_VERSION
        ):
            raise MetadataSnapshotError("snapshot bundle schema_version is unsupported")
        provider = document.get("provider")
        records = document.get("responses")
        if not isinstance(provider, str) or not isinstance(records, list):
            raise MetadataSnapshotError("snapshot bundle requires provider and responses")
        responses: dict[tuple[str, str, str | None], SnapshotResponse] = {}
        for record in records:
            response = _response(record)
            key = (response.operation, response.parameters_hash, response.cursor)
            if key in responses:
                raise MetadataSnapshotError("snapshot bundle has duplicate response keys")
            responses[key] = response
        return cls(provider, responses)


@dataclass(slots=True)
class MetadataSnapshotTransport:
    """Replay matching response bytes through the provider runtime, never HTTP."""

    bundle: MetadataSnapshotBundle
    runtime: ProviderRuntime
    environment: Mapping[str, str] | None = None

    def __call__(self, provider: str, operation: str, parameters: Mapping[str, Any]) -> Mapping[str, Any]:
        if provider != self.bundle.provider:
            raise MetadataSnapshotError(f"snapshot bundle belongs to {self.bundle.provider}, not {provider}")
        parameters_hash = frozen_parameters_hash(parameters)
        cursor = _cursor(parameters)
        try:
            response = self.bundle.responses[(operation, parameters_hash, cursor)]
        except KeyError as error:
            raise MetadataSnapshotError(
                f"snapshot bundle has no response for {provider}:{operation} parameters={parameters_hash} cursor={cursor}"
            ) from error
        content = self.runtime.request(
            provider,
            query_hash=parameters_hash,
            cursor=cursor,
            api_version=f"snapshot-bundle-v{BUNDLE_SCHEMA_VERSION}",
            mode="snapshot",
            snapshot=BulkSnapshot(response.body, response.body_sha256),
            expected_snapshot_hash=response.body_sha256,
            environment=self.environment,
        )
        payload = _decode(content, response.content_type)
        if not isinstance(payload, dict):
            raise ProviderRequestError(f"{provider}: snapshot response must be an object")
        if operation == "verify":
            payload = ControlledHTTPTransport._verification_payload(payload)
        elif operation == "enrich":
            payload = _enrichment_payload(provider, payload)
        elif operation == "resolve":
            payload = _access_payload(provider, payload)
        output = dict(payload)
        if output.get("status") == "ok":
            output["provider_status"] = "ok"
            output["status"] = "success"
        output["raw_response_artifact_hash"] = response.body_sha256
        return output


@dataclass(frozen=True, slots=True)
class ProviderTransportRouter:
    """Send snapshot-mode providers to exact replay and leave others unchanged."""

    snapshots: Mapping[str, MetadataSnapshotTransport]
    fallback: Callable[[str, str, Mapping[str, Any]], Mapping[str, Any]]

    def __call__(self, provider: str, operation: str, parameters: Mapping[str, Any]) -> Mapping[str, Any]:
        snapshot = self.snapshots.get(provider)
        if snapshot is not None:
            return snapshot(provider, operation, parameters)
        return self.fallback(provider, operation, parameters)


def frozen_parameters_hash(parameters: Mapping[str, Any]) -> str:
    """Hash exactly the frozen native parameters, with cursor represented separately."""
    return content_hash({key: value for key, value in parameters.items() if key != "cursor"})


def _response(value: object) -> SnapshotResponse:
    if not isinstance(value, Mapping):
        raise MetadataSnapshotError("snapshot response must be an object")
    required = {"operation", "parameters_hash", "cursor", "content_type", "body_base64", "body_sha256"}
    if set(value) != required:
        raise MetadataSnapshotError("snapshot response fields are invalid")
    operation = value["operation"]
    parameters_hash = value["parameters_hash"]
    cursor = value["cursor"]
    content_type = value["content_type"]
    body_sha256 = value["body_sha256"]
    if not isinstance(operation, str) or not isinstance(parameters_hash, str) or not isinstance(content_type, str):
        raise MetadataSnapshotError("snapshot response fields must be strings")
    if cursor is not None and not isinstance(cursor, str):
        raise MetadataSnapshotError("snapshot response cursor must be a string or null")
    if not _sha256(parameters_hash) or not _sha256(body_sha256):
        raise MetadataSnapshotError("snapshot response hashes are invalid")
    try:
        body = base64.b64decode(value["body_base64"], validate=True)
    except (ValueError, TypeError) as error:
        raise MetadataSnapshotError("snapshot response body_base64 is invalid") from error
    if sha256(body).hexdigest() != body_sha256:
        raise MetadataSnapshotError("snapshot response body hash has drifted")
    return SnapshotResponse(operation, parameters_hash, cursor, content_type, body, body_sha256)


def _cursor(parameters: Mapping[str, Any]) -> str | None:
    value = parameters.get("cursor")
    return str(value) if value is not None else None


def _sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _decode(body: bytes, content_type: str) -> Any:
    if "json" in content_type.casefold() or body.lstrip().startswith((b"{", b"[")):
        return json.loads(body)
    return _xml_object(ElementTree.fromstring(body))


def _xml_object(element: ElementTree.Element) -> dict[str, Any]:
    name = element.tag.rsplit("}", 1)[-1]
    children = list(element)
    if not children:
        return {name: (element.text or "").strip()}
    output: dict[str, Any] = {}
    for child in children:
        child_name, child_value = next(iter(_xml_object(child).items()))
        current = output.get(child_name)
        if current is None:
            output[child_name] = child_value
        elif isinstance(current, list):
            current.append(child_value)
        else:
            output[child_name] = [current, child_value]
    return {name: output}
