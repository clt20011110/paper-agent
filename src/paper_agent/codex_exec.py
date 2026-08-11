"""Minimal, auditable adapter for isolated ``codex exec`` model calls.

The profiles in this module are deliberately code constants.  A configuration
file may select a profile, but it cannot replace its model, sandbox, reasoning
effort, retry budget, or network policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sysconfig
from tempfile import TemporaryDirectory
from types import MappingProxyType
from typing import Any, Callable, Literal, Mapping, Sequence
from urllib.parse import urlsplit
from uuid import uuid4

from jsonschema import Draft202012Validator

from .canonical import content_hash
from .schema import schema_directory, schema_registry


class CodexExecError(RuntimeError):
    """Base error for a failed isolated Codex invocation."""


class CodexAuthError(CodexExecError):
    """Codex CLI is not logged in."""


class CodexTimeoutError(CodexExecError):
    """A bounded Codex invocation exhausted its timeout/retry budget."""


class CodexProcessError(CodexExecError):
    """Codex CLI exited unsuccessfully."""


class CodexOutputError(CodexExecError):
    """Codex did not return the requested structured JSON output."""


class CodexModelMismatchError(CodexExecError):
    """Invocation metadata reported a model other than the frozen model."""


ProfileName = Literal[
    "stage3_authorized_luna",
    "stage4_analysis_luna",
    "stage4b_summary_sol",
]

CallKind = Literal[
    "planning_assist",
    "section_reduce",
    "cross_section_reduce",
    "final_reduce",
    "quality_audit",
    "repair",
]


@dataclass(frozen=True, slots=True)
class CodexExecProfile:
    name: ProfileName
    model: str
    reasoning_effort: Literal["low", "medium", "high"]
    sandbox: Literal["paper-agent-read"]
    network: Literal[False]
    timeout_seconds: int
    max_retries: int
    schema_name: str | None = None


STAGE3_AUTHORIZED_LUNA = CodexExecProfile(
    "stage3_authorized_luna", "gpt-5.6-luna", "low", "paper-agent-read", False, 120, 1,
    "authorized-browser-result.schema.json",
)
STAGE4_ANALYSIS_LUNA = CodexExecProfile(
    "stage4_analysis_luna", "gpt-5.6-luna", "medium", "paper-agent-read", False, 300, 1,
    "paper-analysis.schema.json",
)
STAGE4B_SUMMARY_SOL = CodexExecProfile(
    "stage4b_summary_sol", "gpt-5.6-sol", "high", "paper-agent-read", False, 300, 1,
)

FROZEN_PROFILES: Mapping[ProfileName, CodexExecProfile] = MappingProxyType({
    profile.name: profile
    for profile in (STAGE3_AUTHORIZED_LUNA, STAGE4_ANALYSIS_LUNA, STAGE4B_SUMMARY_SOL)
})

CALL_KIND_SCHEMAS: Mapping[CallKind, str] = MappingProxyType({
    "planning_assist": "report-plan.schema.json",
    "section_reduce": "section-synthesis.schema.json",
    "cross_section_reduce": "cross-section-synthesis.schema.json",
    "final_reduce": "report-document.schema.json",
    "quality_audit": "report-audit.schema.json",
    "repair": "report-repair.schema.json",
})

PROFILE_PROMPTS: Mapping[ProfileName, str | None] = MappingProxyType({
    "stage3_authorized_luna": "authorized-browser.md",
    "stage4_analysis_luna": "paper-analysis.md",
    "stage4b_summary_sol": None,
})

CALL_KIND_PROMPTS: Mapping[CallKind, str] = MappingProxyType({
    "planning_assist": "report-plan.md",
    "section_reduce": "section-synthesis.md",
    "cross_section_reduce": "cross-section-synthesis.md",
    "final_reduce": "final-report.md",
    "quality_audit": "report-audit.md",
    "repair": "report-repair.md",
})

# None of these names carry credentials.  In particular, do not inherit every
# parent variable: API keys, browser tokens and prompt-bearing tracing settings
# must not reach a child model process.
ENV_ALLOWLIST = frozenset({
    "CODEX_HOME", "HOME", "LANG", "LC_ALL", "LC_CTYPE", "LOGNAME", "PATH", "SYSTEMROOT", "TMP", "USER", "WINDIR",
})

USAGE_FIELDS = (
    "input_tokens",
    "cached_input_tokens",
    "cache_write_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
    "total_tokens",
)


def prompt_directory() -> Path:
    repository_prompts = Path(__file__).resolve().parents[2] / "prompts"
    if repository_prompts.is_dir():
        return repository_prompts
    return Path(sysconfig.get_path("data")) / "share" / "paper-agent" / "prompts"


def _digest(value: str | bytes | Mapping[str, Any]) -> str:
    if isinstance(value, Mapping):
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    elif isinstance(value, str):
        encoded = value.encode("utf-8")
    else:
        encoded = value
    return sha256(encoded).hexdigest()


def _validate_strict_schema(schema: Mapping[str, Any]) -> None:
    """Check constraints required by Codex structured outputs before a paid call."""
    def visit(value: Any, location: str) -> None:
        if isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{location}/{index}")
            return
        if not isinstance(value, Mapping):
            return
        if "properties" in value:
            properties = set(value["properties"])
            required = set(value.get("required", ()))
            if value.get("additionalProperties") is not False or required != properties:
                raise CodexOutputError(
                    f"Codex output schema object at {location} must forbid additional properties "
                    "and require every declared property"
                )
        if ("enum" in value or "const" in value) and "type" not in value:
            raise CodexOutputError(f"Codex output schema enum/const at {location} requires an explicit type")
        for key, item in value.items():
            visit(item, f"{location}/{key}")

    visit(schema, "$")


def prepare_service_schema(
    schema_name: str,
    schema: Mapping[str, Any],
    *,
    schema_root: Path,
    resource_paths: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Resolve frozen local refs and enforce Codex structured-output rules."""
    configured = {
        name: Path(path) for name, path in (resource_paths or {}).items()
    }
    documents = {schema_name: dict(schema)}
    resolving: set[tuple[str, str]] = set()

    def document(name: str) -> dict[str, Any]:
        if name not in documents:
            parsed = urlsplit(name)
            relative = Path(parsed.path)
            if (
                parsed.scheme
                or parsed.netloc
                or relative.is_absolute()
                or ".." in relative.parts
                or len(relative.parts) != 1
            ):
                raise CodexOutputError(
                    f"referenced frozen schema is not a local sibling: {name}"
                )
            path = configured.get(name, schema_root / name)
            if not path.is_file():
                raise CodexOutputError(
                    f"referenced frozen schema is missing: {name}"
                )
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as error:
                raise CodexOutputError(
                    f"referenced frozen schema is not valid JSON: {name}"
                ) from error
            if not isinstance(value, dict):
                raise CodexOutputError(
                    f"referenced schema must be a JSON object: {name}"
                )
            documents[name] = value
        return documents[name]

    def resolve_pointer(value: Any, fragment: str, reference: str) -> Any:
        try:
            for part in fragment.removeprefix("/").split("/") if fragment else ():
                value = value[part.replace("~1", "/").replace("~0", "~")]
        except (KeyError, IndexError, TypeError) as error:
            raise CodexOutputError(
                f"referenced frozen schema pointer is missing: {reference}"
            ) from error
        return value

    def expand(
        value: Any, current_name: str, current_document: Mapping[str, Any]
    ) -> Any:
        if isinstance(value, list):
            return [expand(item, current_name, current_document) for item in value]
        if not isinstance(value, Mapping):
            return value
        if "$ref" in value:
            if set(value) != {"$ref"}:
                raise CodexOutputError(
                    "Codex output schemas do not support sibling constraints beside $ref"
                )
            reference = str(value["$ref"])
            filename, _, fragment = reference.partition("#")
            target_name = filename or current_name
            target_document = document(target_name) if filename else current_document
            marker = (target_name, fragment)
            if marker in resolving:
                raise CodexOutputError(
                    f"recursive frozen schema reference is unsupported: {reference}"
                )
            resolving.add(marker)
            try:
                target = resolve_pointer(target_document, fragment, reference)
                return expand(target, target_name, target_document)
            finally:
                resolving.remove(marker)
        return {
            key: expand(item, current_name, current_document)
            for key, item in value.items()
            if key not in {"$schema", "$id", "$defs", "uniqueItems"}
        }

    result = expand(schema, schema_name, schema)
    _validate_strict_schema(result)
    return result


@dataclass(frozen=True, slots=True)
class CodexExecRequest:
    """One new, independent model invocation.

    ``prompt`` is intentionally not copied to the returned metadata.  The
    caller is responsible for authorizing its contents before constructing this
    request.
    """

    profile: ProfileName
    prompt: str
    output_schema: Mapping[str, Any]
    schema_name: str
    prompt_name: str
    input_hash: str
    call_kind: CallKind | None = None
    schema_path: str | None = None
    prompt_path: str | None = None
    expected_prompt_hash: str | None = None
    schema_resource_paths: Mapping[str, str] | None = None
    expected_service_schema_hash: str | None = None

    def __post_init__(self) -> None:
        if not self.prompt:
            raise ValueError("prompt is required")
        if len(self.input_hash) != 64 or any(character not in "0123456789abcdef" for character in self.input_hash):
            raise ValueError("input_hash must be a lowercase SHA-256 digest")
        Draft202012Validator.check_schema(dict(self.output_schema))
        if self.profile == "stage4b_summary_sol":
            if self.call_kind is None:
                raise ValueError("Stage 4b calls require call_kind")
            if self.schema_name != CALL_KIND_SCHEMAS[self.call_kind]:
                raise ValueError("Stage 4b call_kind requires its frozen output schema")
            if self.prompt_name != CALL_KIND_PROMPTS[self.call_kind]:
                raise ValueError("Stage 4b call_kind requires its frozen prompt")
        elif self.call_kind is not None:
            raise ValueError("only Stage 4b calls may set call_kind")
        elif self.schema_name != FROZEN_PROFILES[self.profile].schema_name:
            raise ValueError("profile requires its frozen output schema name")
        elif self.prompt_name != PROFILE_PROMPTS[self.profile]:
            raise ValueError("profile requires its frozen prompt")
        if (self.schema_path is None) != (self.prompt_path is None):
            raise ValueError("custom schema and prompt paths must be supplied together")
        if self.schema_path is not None and self.profile != "stage4b_summary_sol":
            raise ValueError("only Stage 4b calls may use configured resource paths")
        if self.schema_resource_paths is not None:
            if self.profile != "stage4b_summary_sol" or self.schema_path is None:
                raise ValueError(
                    "only configured Stage 4b calls may map schema resources"
                )
            if set(self.schema_resource_paths) != set(CALL_KIND_SCHEMAS.values()):
                raise ValueError(
                    "configured Stage 4b schema resources must map every call kind"
                )
            resources = {
                str(name): str(path)
                for name, path in self.schema_resource_paths.items()
            }
            if Path(resources[self.schema_name]) != Path(self.schema_path):
                raise ValueError(
                    "configured Stage 4b schema path differs from its resource map"
                )
            object.__setattr__(
                self, "schema_resource_paths", MappingProxyType(resources)
            )
            if self.expected_service_schema_hash is None:
                raise ValueError(
                    "configured Stage 4b resources require an effective schema SHA-256"
                )
        if self.expected_service_schema_hash is not None and (
            len(self.expected_service_schema_hash) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.expected_service_schema_hash
            )
        ):
            raise ValueError(
                "expected effective schema hash must be a lowercase SHA-256"
            )
        if self.schema_path is not None and (
            self.expected_prompt_hash is None
            or len(self.expected_prompt_hash) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.expected_prompt_hash
            )
        ):
            raise ValueError(
                "configured Stage 4b resources require an expected prompt SHA-256"
            )


@dataclass(frozen=True, slots=True)
class InvocationMetadata:
    invocation_id: str
    profile: ProfileName
    model: str
    reasoning_effort: str
    schema_name: str
    schema_hash: str
    input_hash: str
    prompt_name: str
    prompt_hash: str
    rendered_prompt_hash: str
    call_kind: CallKind | None
    attempts: int
    actual_model: str | None
    actual_profile: str | None
    schema_path: str | None = None
    prompt_path: str | None = None
    output_hash: str | None = None
    usage_available: bool = False
    input_tokens: int | None = None
    cached_input_tokens: int | None = None
    cache_write_input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_output_tokens: int | None = None
    total_tokens: int | None = None
    cost_usd: float | None = None


@dataclass(frozen=True, slots=True)
class CodexExecResult:
    output: Mapping[str, Any]
    metadata: InvocationMetadata


@dataclass(frozen=True, slots=True)
class DoctorReport:
    executable: str
    version: str
    authenticated: bool
    # "listed" proves only that the local Codex catalog exposes the slug.  An
    # opt-in paid probe is still required to prove account availability.
    model_availability: Mapping[ProfileName, Literal["listed", "available", "unavailable"]]


Runner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True, slots=True)
class _ExecutionResult:
    process: subprocess.CompletedProcess[str]
    attempts: int
    actual_model: str | None
    actual_profile: str | None
    usage: Mapping[str, int] | None


class CodexExec:
    """Run a frozen profile with an argv-only subprocess boundary."""

    def __init__(
        self,
        *,
        executable: str = "codex",
        runner: Runner = subprocess.run,
        environment: Mapping[str, str] | None = None,
        temporary_root: Path | None = None,
        schema_root: Path | None = None,
        prompt_root: Path | None = None,
    ) -> None:
        self._executable = executable
        self._runner = runner
        self._environment = dict(os.environ if environment is None else environment)
        self._temporary_root = temporary_root
        self._schema_root = schema_directory() if schema_root is None else schema_root
        self._prompt_root = prompt_directory() if prompt_root is None else prompt_root

    @staticmethod
    def profile(name: ProfileName) -> CodexExecProfile:
        return FROZEN_PROFILES[name]

    def doctor(self, *, prove_model_availability: bool = False) -> DoctorReport:
        executable = shutil.which(self._executable) if Path(self._executable).name == self._executable else self._executable
        if not executable:
            raise CodexExecError("codex executable was not found")
        version = self._run_diagnostic([executable, "--version"])
        try:
            self._run_diagnostic([executable, "login", "status"])
        except CodexProcessError as error:
            raise CodexAuthError("codex login status failed") from error
        catalog = json.loads(self._run_diagnostic([executable, "debug", "models"]))
        listed_models = {entry["slug"] for entry in catalog["models"]}
        availability: dict[ProfileName, Literal["listed", "available", "unavailable"]] = {
            name: "listed" if profile.model in listed_models else "unavailable"
            for name, profile in FROZEN_PROFILES.items()
        }
        if prove_model_availability:
            probed: dict[str, Literal["available", "unavailable"]] = {}
            for name, profile in FROZEN_PROFILES.items():
                if availability[name] == "unavailable":
                    continue
                if profile.model not in probed:
                    probed[profile.model] = self._probe_model(profile)
                availability[name] = probed[profile.model]
        return DoctorReport(
            executable=executable,
            version=version.strip(),
            authenticated=True,
            model_availability=MappingProxyType(availability),
        )

    def invoke(self, request: CodexExecRequest) -> CodexExecResult:
        profile = self.profile(request.profile)
        invocation_id = str(uuid4())
        schema_path = Path(request.schema_path) if request.schema_path is not None else None
        prompt_path = Path(request.prompt_path) if request.prompt_path is not None else None
        frozen_schema = self._frozen_schema(request.schema_name, schema_path)
        if _digest(request.output_schema) != _digest(frozen_schema):
            raise CodexOutputError(f"{request.schema_name} does not match the frozen repository schema")
        service_schema = self._service_schema(
            request.schema_name,
            frozen_schema,
            request.schema_resource_paths,
        )
        if (
            request.expected_service_schema_hash is not None
            and _digest(service_schema) != request.expected_service_schema_hash
        ):
            raise CodexOutputError(
                f"{request.schema_name} dependencies changed after approval"
            )
        schema_hash = _digest(frozen_schema)
        prompt_template = self._frozen_prompt(request.prompt_name, prompt_path)
        prompt_hash = _digest(prompt_template)
        if (
            request.expected_prompt_hash is not None
            and prompt_hash != request.expected_prompt_hash
        ):
            raise CodexOutputError(
                f"{request.prompt_name} changed after its caller approved the prompt"
            )
        rendered_prompt = self._render_prompt(prompt_template, request.prompt)
        rendered_prompt_hash = _digest(rendered_prompt)
        with TemporaryDirectory(prefix="paper-agent-codex-exec-", dir=self._temporary_root) as directory:
            workdir = Path(directory)
            schema_path = workdir / "output-schema.json"
            result_path = workdir / "last-message.json"
            schema_path.write_text(json.dumps(service_schema, ensure_ascii=False, sort_keys=True), encoding="utf-8")
            argv = self._argv(profile, workdir, schema_path, result_path)
            execution = self._run_with_retry(argv, workdir, profile, rendered_prompt)
            completed = execution.process
            if completed.returncode != 0:
                raise CodexProcessError(f"codex exec exited with status {completed.returncode}")
            output = self._read_output(result_path)
            self._validate_output(output, service_schema)
            actual_model = execution.actual_model or profile.model
            actual_profile = execution.actual_profile or profile.name
            usage = execution.usage
            metadata = InvocationMetadata(
                invocation_id=invocation_id,
                profile=profile.name,
                model=profile.model,
                reasoning_effort=profile.reasoning_effort,
                schema_name=request.schema_name,
                schema_hash=schema_hash,
                input_hash=request.input_hash,
                prompt_name=request.prompt_name,
                prompt_hash=prompt_hash,
                rendered_prompt_hash=rendered_prompt_hash,
                call_kind=request.call_kind,
                attempts=execution.attempts,
                actual_model=actual_model,
                actual_profile=actual_profile,
                schema_path=request.schema_path,
                prompt_path=request.prompt_path,
                output_hash=content_hash(output),
                usage_available=usage is not None,
                input_tokens=None if usage is None else usage.get("input_tokens"),
                cached_input_tokens=None if usage is None else usage.get("cached_input_tokens"),
                cache_write_input_tokens=None if usage is None else usage.get("cache_write_input_tokens"),
                output_tokens=None if usage is None else usage.get("output_tokens"),
                reasoning_output_tokens=None if usage is None else usage.get("reasoning_output_tokens"),
                total_tokens=None if usage is None else usage.get("total_tokens"),
            )
            return CodexExecResult(MappingProxyType(output), metadata)

    def _argv(self, profile: CodexExecProfile, workdir: Path, schema_path: Path, result_path: Path) -> list[str]:
        """Build the exact no-shell command used for every model invocation."""
        return [
            self._executable,
            "exec",
            "-m", profile.model,
            "-c", f'model_reasoning_effort="{profile.reasoning_effort}"',
            "-c", 'approval_policy="never"',
            "-c", f'default_permissions="{profile.sandbox}"',
            "-c", f'permissions.{profile.sandbox}.description="Read staged inputs only"',
            "-c", f'permissions.{profile.sandbox}.filesystem.:minimal="read"',
            "-c", f'permissions.{profile.sandbox}.filesystem.:workspace_roots={{"."="read"}}',
            "-c", f'permissions.{profile.sandbox}.network.enabled={str(profile.network).lower()}',
            "-C", str(workdir),
            "--skip-git-repo-check",
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--output-schema", str(schema_path),
            "--json",
            "-o", str(result_path),
            "-",
        ]

    def _run_with_retry(
        self, argv: Sequence[str], workdir: Path, profile: CodexExecProfile, prompt: str,
    ) -> _ExecutionResult:
        usages: list[dict[str, int] | None] = []
        for attempt in range(profile.max_retries + 1):
            try:
                completed = self._runner(
                    list(argv), cwd=str(workdir), env=self._safe_environment(workdir), timeout=profile.timeout_seconds,
                    input=prompt, capture_output=True, text=True, check=False,
                )
            except subprocess.TimeoutExpired as error:
                stdout = error.stdout
                if isinstance(stdout, bytes):
                    stdout = stdout.decode("utf-8")
                actual_model, actual_profile, usage = (
                    self._actual_invocation_metadata(stdout)
                )
                self._validate_actual_invocation(
                    profile, actual_model, actual_profile
                )
                usages.append(usage)
                if attempt == profile.max_retries:
                    raise CodexTimeoutError("codex exec timed out") from error
                continue
            actual_model, actual_profile, usage = self._actual_invocation_metadata(
                completed.stdout
            )
            self._validate_actual_invocation(profile, actual_model, actual_profile)
            usages.append(usage)
            result = _ExecutionResult(
                completed,
                attempt + 1,
                actual_model,
                actual_profile,
                self._aggregate_usage(usages),
            )
            if completed.returncode == 0:
                return result
            if attempt == profile.max_retries:
                return result
        raise AssertionError("unreachable retry state")

    @staticmethod
    def _validate_actual_invocation(
        profile: CodexExecProfile,
        actual_model: str | None,
        actual_profile: str | None,
    ) -> None:
        if actual_model is not None and actual_model != profile.model:
            raise CodexModelMismatchError(
                f"codex invocation model mismatch: expected {profile.model}, got {actual_model}"
            )
        if actual_profile is not None and actual_profile != profile.name:
            raise CodexModelMismatchError(
                f"codex invocation profile mismatch: expected {profile.name}, got {actual_profile}"
            )

    @staticmethod
    def _aggregate_usage(
        usages: Sequence[Mapping[str, int] | None],
    ) -> dict[str, int] | None:
        if not usages or any(usage is None for usage in usages):
            return None
        complete = tuple(usage for usage in usages if usage is not None)
        totals = {
            field: sum(usage[field] for usage in complete)
            for field in USAGE_FIELDS
            if all(field in usage for usage in complete)
        }
        return totals or None

    def _safe_environment(self, workdir: Path) -> dict[str, str]:
        environment = {name: self._environment[name] for name in ENV_ALLOWLIST if name in self._environment}
        environment["TMP"] = str(workdir)
        return environment

    def _run_diagnostic(self, argv: list[str]) -> str:
        completed = self._runner(
            argv, cwd=str(self._temporary_root or Path.cwd()), env=self._safe_environment(self._temporary_root or Path.cwd()),
            timeout=15, capture_output=True, text=True, check=False,
        )
        if completed.returncode != 0:
            raise CodexProcessError(f"{' '.join(argv[:3])} exited with status {completed.returncode}")
        return completed.stdout

    def _frozen_schema(
        self, schema_name: str, configured_path: Path | None = None
    ) -> dict[str, Any]:
        path = configured_path or self._schema_root / schema_name
        if not path.is_file() or (configured_path is None and path.parent != self._schema_root):
            raise CodexOutputError(f"frozen schema is missing: {schema_name}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise CodexOutputError(f"frozen schema must be a JSON object: {schema_name}")
        return value

    def _service_schema(
        self,
        schema_name: str,
        schema: Mapping[str, Any],
        resource_paths: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        return prepare_service_schema(
            schema_name,
            schema,
            schema_root=self._schema_root,
            resource_paths=resource_paths,
        )

    def _frozen_prompt(
        self, prompt_name: str, configured_path: Path | None = None
    ) -> str:
        path = configured_path or self._prompt_root / prompt_name
        if not path.is_file() or (configured_path is None and path.parent != self._prompt_root):
            raise CodexOutputError(f"frozen prompt is missing: {prompt_name}")
        return path.read_text(encoding="utf-8")

    @staticmethod
    def _render_prompt(template: str, payload: str) -> str:
        encoded = json.dumps({"authorized_input": payload}, ensure_ascii=False, separators=(",", ":"))
        return f"{template.rstrip()}\n\nThe authorized input follows as JSON data:\n{encoded}\n"

    def _probe_model(self, profile: CodexExecProfile) -> Literal["available", "unavailable"]:
        """Perform an opt-in real dry invocation; declarations alone prove nothing."""
        schema = {
            "type": "object", "additionalProperties": False,
            "required": ["ok"], "properties": {"ok": {"type": "boolean", "const": True}},
        }
        with TemporaryDirectory(prefix="paper-agent-codex-probe-", dir=self._temporary_root) as directory:
            workdir = Path(directory)
            schema_path = workdir / "output-schema.json"
            result_path = workdir / "last-message.json"
            schema_path.write_text(json.dumps(schema), encoding="utf-8")
            argv = self._argv(profile, workdir, schema_path, result_path)
            try:
                execution = self._run_with_retry(
                    argv, workdir, profile,
                    "Return only the structured confirmation requested by the output schema.",
                )
            except CodexExecError:
                return "unavailable"
            completed = execution.process
            if completed.returncode != 0:
                return "unavailable"
            try:
                output = self._read_output(result_path)
                self._validate_output(output, schema)
            except CodexOutputError:
                return "unavailable"
            return "available"

    @staticmethod
    def _read_output(path: Path) -> dict[str, Any]:
        if not path.is_file():
            raise CodexOutputError("codex did not write its final structured result")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise CodexOutputError("codex final result is not JSON") from error
        if not isinstance(value, dict):
            raise CodexOutputError("codex final result must be a JSON object")
        return value

    @staticmethod
    def _validate_output(output: Mapping[str, Any], schema: Mapping[str, Any]) -> None:
        _, registry = schema_registry()
        errors = sorted(
            Draft202012Validator(dict(schema), registry=registry).iter_errors(dict(output)),
            key=lambda error: list(error.path),
        )
        if errors:
            location = ".".join(str(part) for part in errors[0].path) or "$"
            raise CodexOutputError(f"codex final result violates output schema at {location}: {errors[0].message}")

    @staticmethod
    def _actual_invocation_metadata(
        stdout: str | None,
    ) -> tuple[str | None, str | None, dict[str, int] | None]:
        """Read exposed JSONL facts without retaining event contents."""
        if not stdout:
            return None, None, None
        models: set[str] = set()
        profiles: set[str] = set()
        usage: dict[str, int] | None = None
        completed_turns = 0
        for line in stdout.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                raise CodexOutputError("codex --json stdout contains malformed JSONL") from error
            if not isinstance(event, Mapping):
                raise CodexOutputError("codex --json stdout event must be a JSON object")
            for key in ("model", "model_slug", "model_name"):
                if isinstance(event.get(key), str):
                    models.add(event[key])
            for key in ("profile", "profile_name"):
                if isinstance(event.get(key), str):
                    profiles.add(event[key])
            if event.get("type") == "turn.completed":
                completed_turns += 1
                if completed_turns > 1:
                    raise CodexOutputError(
                        "codex --json stdout contains multiple turn.completed events"
                    )
                if event.get("usage") is None:
                    continue
                if not isinstance(event["usage"], Mapping):
                    raise CodexOutputError("codex invocation usage must be a JSON object")
                parsed: dict[str, int] = {}
                for key in USAGE_FIELDS:
                    value = event["usage"].get(key)
                    if value is None:
                        continue
                    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                        raise CodexOutputError(
                            f"codex invocation usage field is invalid: {key}"
                        )
                    parsed[key] = value
                usage = parsed or None
        if len(models) > 1:
            raise CodexModelMismatchError("codex invocation metadata reported multiple models")
        if len(profiles) > 1:
            raise CodexModelMismatchError("codex invocation metadata reported multiple profiles")
        return next(iter(models), None), next(iter(profiles), None), usage
