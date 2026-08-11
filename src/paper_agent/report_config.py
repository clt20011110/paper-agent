"""Resolved Stage 4b configuration and per-call model resources."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
import sysconfig
from types import MappingProxyType
from typing import Any
from urllib.parse import urldefrag, urljoin, urlsplit

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import SchemaError
from referencing import Registry, Resource
from referencing.exceptions import Unresolvable
from referencing.jsonschema import DRAFT202012

from .canonical import canonical_json, content_hash
from .codex_exec import (
    CALL_KIND_PROMPTS,
    CALL_KIND_SCHEMAS,
    CodexOutputError,
    prepare_service_schema,
    prompt_directory,
)
from .schema import SchemaValidationError, schema_directory


CALL_KINDS = tuple(CALL_KIND_SCHEMAS)


class ReportConfigError(ValueError):
    """A summary runtime setting is incomplete or does not match its input."""


@dataclass(frozen=True, slots=True)
class ReportResources:
    """Exact prompt and schema files selected for all Stage 4b call kinds."""

    schema_paths: Mapping[str, Path]
    prompt_paths: Mapping[str, Path]
    configured: bool = True

    @classmethod
    def defaults(
        cls, *, schema_root: Path | None = None, prompt_root: Path | None = None
    ) -> ReportResources:
        schemas = schema_directory(schema_root)
        prompts = prompt_directory() if prompt_root is None else prompt_root
        return cls(
            {kind: schemas / name for kind, name in CALL_KIND_SCHEMAS.items()},
            {kind: prompts / name for kind, name in CALL_KIND_PROMPTS.items()},
            configured=schema_root is not None or prompt_root is not None,
        )

    def __post_init__(self) -> None:
        expected = set(CALL_KINDS)
        if set(self.schema_paths) != expected or set(self.prompt_paths) != expected:
            raise ReportConfigError(
                "summary schemas and prompts must define every Stage 4b call kind"
            )
        object.__setattr__(
            self,
            "schema_paths",
            MappingProxyType({kind: Path(self.schema_paths[kind]) for kind in CALL_KINDS}),
        )
        object.__setattr__(
            self,
            "prompt_paths",
            MappingProxyType({kind: Path(self.prompt_paths[kind]) for kind in CALL_KINDS}),
        )

    def validate_files(self) -> None:
        identifiers: dict[str, Path] = {}
        schema_files: dict[Path, str] = {}
        schema_documents: dict[bytes, str] = {}
        for kind in CALL_KINDS:
            schema_path = self.schema_paths[kind]
            prompt_path = self.prompt_paths[kind]
            if not schema_path.is_file():
                raise ReportConfigError(f"summary schema is unavailable for {kind}: {schema_path}")
            if not prompt_path.is_file():
                raise ReportConfigError(f"summary prompt is unavailable for {kind}: {prompt_path}")
            schema = self.schema(kind)
            resolved = schema_path.resolve()
            previous_kind = schema_files.get(resolved)
            if previous_kind is not None:
                raise ReportConfigError(
                    f"summary call kinds {previous_kind} and {kind} share one output schema"
                )
            schema_files[resolved] = kind
            document = canonical_json(schema)
            previous_kind = schema_documents.get(document)
            if previous_kind is not None:
                raise ReportConfigError(
                    f"summary call kinds {previous_kind} and {kind} share one output schema"
                )
            schema_documents[document] = kind
            try:
                Draft202012Validator.check_schema(schema)
            except SchemaError as error:
                raise ReportConfigError(f"invalid summary schema for {kind}: {error}") from error
            identifier = schema.get("$id")
            if isinstance(identifier, str) and identifier:
                previous = identifiers.get(identifier)
                if previous is not None and previous.resolve() != schema_path.resolve():
                    raise ReportConfigError(
                        "summary schemas declare the same $id: "
                        f"{previous} and {schema_path}"
                    )
                identifiers[identifier] = schema_path
        for kind in CALL_KINDS:
            _schema_registry(self.schema_paths[kind], self.schema_paths)
            try:
                self.service_schema(kind)
            except CodexOutputError as error:
                raise ReportConfigError(
                    f"summary schema is not Codex-compatible for {kind}: {error}"
                ) from error

    def schema(self, call_kind: str) -> dict[str, Any]:
        path = self.schema_paths[call_kind]
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ReportConfigError(f"summary schema is not valid JSON: {path}") from error
        if not isinstance(value, dict):
            raise ReportConfigError(f"summary schema must be a JSON object: {path}")
        if call_kind == "planning_assist":
            properties = value.get("properties")
            required = value.get("required")
            if (
                isinstance(properties, dict)
                and isinstance(required, list)
            ):
                # Persisted plans may predate these fields, while every newly
                # compiled structured plan must remain a strict Codex schema.
                value = {
                    **value,
                    "required": list(properties),
                }
                prompt_hashes = value["properties"].get("prompt_hashes")
                if isinstance(prompt_hashes, dict) and isinstance(
                    prompt_hashes.get("properties"), dict
                ):
                    value["properties"]["prompt_hashes"] = {
                        **prompt_hashes,
                        "required": list(prompt_hashes["properties"]),
                    }
        return value

    def prompt(self, call_kind: str) -> str:
        return self.prompt_paths[call_kind].read_text(encoding="utf-8")

    def validate(self, document: Any, call_kind: str) -> None:
        schema = self.schema(call_kind)
        registry = _schema_registry(self.schema_paths[call_kind], self.schema_paths)
        errors = sorted(
            Draft202012Validator(
                schema, registry=registry, format_checker=FormatChecker()
            ).iter_errors(document),
            key=lambda error: list(error.path),
        )
        if errors:
            error = errors[0]
            location = ".".join(str(part) for part in error.path) or "$"
            raise SchemaValidationError(f"{location}: {error.message}")

    def schema_path(self, call_kind: str) -> str | None:
        return str(self.schema_paths[call_kind]) if self.configured else None

    def prompt_path(self, call_kind: str) -> str | None:
        return str(self.prompt_paths[call_kind]) if self.configured else None

    def configured_schema_resources(self) -> Mapping[str, str] | None:
        if not self.configured:
            return None
        return MappingProxyType({
            CALL_KIND_SCHEMAS[kind]: str(self.schema_paths[kind])
            for kind in CALL_KINDS
        })

    def service_schema(self, call_kind: str) -> dict[str, Any]:
        return prepare_service_schema(
            CALL_KIND_SCHEMAS[call_kind],
            self.schema(call_kind),
            schema_root=schema_directory(),
            resource_paths={
                CALL_KIND_SCHEMAS[kind]: str(self.schema_paths[kind])
                for kind in CALL_KINDS
            },
        )

    def service_schema_hash(self, call_kind: str) -> str:
        return content_hash(self.service_schema(call_kind))

    def accepts_metadata_paths(
        self,
        call_kind: str,
        schema_path: str | None,
        prompt_path: str | None,
    ) -> bool:
        """Accept an exact path binding or a legacy canonical-default binding."""
        if (schema_path, prompt_path) == (
            self.schema_path(call_kind),
            self.prompt_path(call_kind),
        ):
            return True
        if schema_path is not None or prompt_path is not None:
            return False
        defaults = ReportResources.defaults()
        return (
            canonical_json(self.schema(call_kind))
            == canonical_json(defaults.schema(call_kind))
            and self.prompt_paths[call_kind].read_bytes()
            == defaults.prompt_paths[call_kind].read_bytes()
        )


@dataclass(frozen=True, slots=True)
class ReportRuntimeConfig:
    """Summary enablement, pinned-plan policy, and resolved model resources."""

    enabled: bool
    resources: ReportResources
    report_plan_path: Path | None = None
    report_plan_hash: str | None = None
    require_plan_for_unattended: bool = True
    rubric_path: Path | None = None
    profile: str = "stage4b_summary_sol"
    execution_strategy: str = "reduce_tree"

    def __post_init__(self) -> None:
        if not self.require_plan_for_unattended:
            raise ReportConfigError(
                "unattended summary must require a pinned report plan"
            )
        if self.report_plan_hash is not None and not _is_sha256(
            self.report_plan_hash
        ):
            raise ReportConfigError(
                "summary report plan hash must be a lowercase SHA-256"
            )
        if self.profile not in {"stage4b_summary_sol", "stage4b_oneshot_sol"}:
            raise ReportConfigError("summary profile is unsupported")
        if self.execution_strategy not in {"reduce_tree", "one_shot"}:
            raise ReportConfigError("summary execution strategy is unsupported")
        expected_profile = (
            "stage4b_oneshot_sol"
            if self.execution_strategy == "one_shot"
            else "stage4b_summary_sol"
        )
        if self.profile != expected_profile:
            raise ReportConfigError(
                f"{self.execution_strategy} summary requires profile {expected_profile}"
            )

    @classmethod
    def defaults(cls) -> ReportRuntimeConfig:
        return cls(True, ReportResources.defaults())

    @classmethod
    def from_config(
        cls, config: Mapping[str, Any], config_path: Path
    ) -> ReportRuntimeConfig:
        summary = config.get("summary", config)
        if not isinstance(summary, Mapping):
            raise ReportConfigError("summary configuration must be an object")
        enabled = summary.get("enabled")
        if not isinstance(enabled, bool):
            raise ReportConfigError("summary.enabled must be boolean")
        schemas = _resource_mapping(summary.get("schemas"), "summary.schemas", config_path)
        prompts = _resource_mapping(summary.get("prompts"), "summary.prompts", config_path)
        plan = summary.get("report_plan")
        if not isinstance(plan, Mapping):
            raise ReportConfigError("summary.report_plan must be an object")
        raw_plan_path = plan.get("input_path")
        plan_path = (
            _configured_path(config_path, str(raw_plan_path))
            if raw_plan_path is not None
            else None
        )
        final_audit = summary.get("final_audit")
        if not isinstance(final_audit, Mapping):
            raise ReportConfigError("summary.final_audit must be an object")
        rubric = final_audit.get("rubric")
        rubric_path = (
            _resource_path(config_path, str(rubric)) if rubric is not None else None
        )
        runtime = cls(
            enabled=enabled,
            resources=ReportResources(schemas, prompts, configured=True),
            report_plan_path=plan_path,
            report_plan_hash=(
                str(plan["content_hash"]) if plan.get("content_hash") is not None else None
            ),
            require_plan_for_unattended=bool(plan.get("required_for_unattended", True)),
            rubric_path=rubric_path,
            profile=str(summary["profile"]),
            execution_strategy=str(summary["execution_strategy"]),
        )
        if runtime.enabled:
            runtime.resources.validate_files()
            if runtime.rubric_path is not None and not runtime.rubric_path.is_file():
                raise ReportConfigError(
                    f"summary audit rubric is unavailable: {runtime.rubric_path}"
                )
        return runtime

    def validate_for_run(
        self, plan: Mapping[str, Any], *, execution_mode: str
    ) -> None:
        if execution_mode not in {"attended", "unattended"}:
            raise ReportConfigError(
                "summary execution mode must be attended or unattended"
            )
        if not self.enabled:
            return
        strategy = str(plan.get("execution_strategy", "reduce_tree"))
        if self.execution_strategy != strategy:
            raise ReportConfigError(
                f"summary configuration strategy {self.execution_strategy} does not match {strategy} report plan"
            )
        self.resources.validate_files()
        if self.rubric_path is not None and not self.rubric_path.is_file():
            raise ReportConfigError(f"summary audit rubric is unavailable: {self.rubric_path}")
        if execution_mode != "unattended":
            return
        if self.report_plan_path is None or self.report_plan_hash is None:
            raise ReportConfigError(
                "unattended summary requires a pinned report plan path and content hash"
            )
        if not self.report_plan_path.is_file():
            raise ReportConfigError(
                f"pinned unattended report plan is unavailable: {self.report_plan_path}"
            )
        try:
            pinned = json.loads(self.report_plan_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ReportConfigError("pinned unattended report plan is not valid JSON") from error
        if not isinstance(pinned, Mapping):
            raise ReportConfigError("pinned unattended report plan must be an object")
        if pinned.get("plan_hash") != self.report_plan_hash:
            raise ReportConfigError("pinned unattended report plan hash differs from configuration")
        if plan.get("plan_hash") != self.report_plan_hash:
            raise ReportConfigError("runtime report plan hash differs from unattended pin")
        if canonical_json(dict(pinned)) != canonical_json(dict(plan)):
            raise ReportConfigError("runtime report plan differs from the pinned unattended file")


def _resource_mapping(
    value: object, label: str, config_path: Path
) -> dict[str, Path]:
    if not isinstance(value, Mapping) or set(value) != set(CALL_KINDS):
        raise ReportConfigError(f"{label} must define every Stage 4b call kind")
    return {
        kind: _resource_path(config_path, str(value[kind]))
        for kind in CALL_KINDS
    }


def _configured_path(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else config_path.parent / path


def _resource_path(config_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    configured = config_path.parent / path
    if configured.is_file():
        return configured
    repository = Path(__file__).resolve().parents[2] / path
    if repository.is_file():
        return repository
    installed = Path(sysconfig.get_path("data")) / "share" / "paper-agent" / path
    return installed if installed.is_file() else configured


def _schema_registry(
    main_path: Path, configured_paths: Mapping[str, Path]
) -> Registry:
    """Load only the selected schema and its local transitive references."""
    selected = {
        schema_name: Path(configured_paths[kind])
        for kind, schema_name in CALL_KIND_SCHEMAS.items()
    }
    default_root = schema_directory()
    registry = Registry()
    loaded: dict[Path, tuple[dict[str, Any], Resource]] = {}
    aliases: dict[str, Path] = {}
    lookups: list[tuple[str, str, Path]] = []

    def register(uri: str, path: Path, resource: Resource) -> None:
        nonlocal registry
        previous = aliases.get(uri)
        if previous is not None and previous != path:
            raise ReportConfigError(
                f"summary schema URI {uri} is declared by both {previous} and {path}"
            )
        aliases[uri] = path
        registry = registry.with_resource(uri, resource)

    def reference_path(reference: str, source: Path) -> Path:
        parsed = urlsplit(reference)
        if parsed.query:
            raise ReportConfigError(
                f"summary schema reference may not contain a query: {reference}"
            )
        name = Path(parsed.path).name
        if parsed.scheme or parsed.netloc:
            raise ReportConfigError(
                f"summary schema reference is not a local frozen resource: {reference}"
            )
        if not parsed.scheme:
            relative = Path(parsed.path)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or len(relative.parts) != 1
            ):
                raise ReportConfigError(
                    f"summary schema reference must name a sibling resource: {reference}"
                )
        if not name:
            raise ReportConfigError(f"summary schema reference is empty: {reference}")
        if name in selected:
            return selected[name]
        candidates = (default_root / name,)
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise ReportConfigError(
            f"referenced summary schema is unavailable from {source}: {reference}"
        )

    def references(value: Any) -> tuple[str, ...]:
        found: list[str] = []

        def visit(item: Any) -> None:
            if isinstance(item, Mapping):
                reference = item.get("$ref")
                if isinstance(reference, str) and reference:
                    found.append(reference)
                for child in item.values():
                    visit(child)
            elif isinstance(item, list):
                for child in item:
                    visit(child)

        visit(value)
        return tuple(found)

    def load(path: Path, requested_uri: str | None = None) -> None:
        resolved = path.resolve()
        if resolved in loaded:
            _, resource = loaded[resolved]
            if requested_uri is not None:
                register(requested_uri, resolved, resource)
            return
        try:
            value = json.loads(resolved.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ReportConfigError(
                f"referenced summary schema is not valid JSON: {resolved}"
            ) from error
        if not isinstance(value, dict):
            raise ReportConfigError(
                f"referenced summary schema must be a JSON object: {resolved}"
            )
        try:
            Draft202012Validator.check_schema(value)
        except SchemaError as error:
            raise ReportConfigError(
                f"invalid referenced summary schema {resolved}: {error}"
            ) from error
        resource = Resource.from_contents(
            value, default_specification=DRAFT202012
        )
        loaded[resolved] = (value, resource)
        identifier = value.get("$id")
        base_uri = requested_uri or resolved.as_uri()
        if isinstance(identifier, str) and identifier:
            base_uri = urljoin(base_uri, identifier)
        register(base_uri, resolved, resource)
        if requested_uri is not None and requested_uri != base_uri:
            register(requested_uri, resolved, resource)
        for reference in references(value):
            lookups.append((base_uri, reference, resolved))
            if not reference.startswith("#"):
                target_uri, _ = urldefrag(urljoin(base_uri, reference))
                load(reference_path(reference, resolved), target_uri)

    load(Path(main_path))
    for base_uri, reference, source in lookups:
        try:
            registry.resolver(base_uri).lookup(reference)
        except Unresolvable as error:
            raise ReportConfigError(
                f"summary schema reference cannot be resolved from {source}: {reference}"
            ) from error
    return registry


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
