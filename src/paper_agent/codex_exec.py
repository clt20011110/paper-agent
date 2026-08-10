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
from uuid import uuid4

from jsonschema import Draft202012Validator

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
        frozen_schema = self._frozen_schema(request.schema_name)
        if _digest(request.output_schema) != _digest(frozen_schema):
            raise CodexOutputError(f"{request.schema_name} does not match the frozen repository schema")
        schema_hash = _digest(frozen_schema)
        prompt_template = self._frozen_prompt(request.prompt_name)
        prompt_hash = _digest(prompt_template)
        rendered_prompt = self._render_prompt(prompt_template, request.prompt)
        rendered_prompt_hash = _digest(rendered_prompt)
        with TemporaryDirectory(prefix="paper-agent-codex-exec-", dir=self._temporary_root) as directory:
            workdir = Path(directory)
            schema_path = workdir / "output-schema.json"
            result_path = workdir / "last-message.json"
            self._write_schema_bundle(schema_path, frozen_schema)
            argv = self._argv(profile, workdir, schema_path, result_path)
            completed = self._run_with_retry(argv, workdir, profile, rendered_prompt)
            if completed.returncode != 0:
                raise CodexProcessError(f"codex exec exited with status {completed.returncode}")
            output = self._read_output(result_path)
            self._validate_output(output, frozen_schema)
            actual_model, actual_profile = self._actual_invocation_metadata(completed.stdout)
            if actual_model is not None and actual_model != profile.model:
                raise CodexModelMismatchError(
                    f"codex invocation model mismatch: expected {profile.model}, got {actual_model}"
                )
            if actual_profile is not None and actual_profile != profile.name:
                raise CodexModelMismatchError(
                    f"codex invocation profile mismatch: expected {profile.name}, got {actual_profile}"
                )
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
                attempts=self._attempt_count(completed),
                actual_model=actual_model,
                actual_profile=actual_profile,
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
    ) -> subprocess.CompletedProcess[str]:
        for attempt in range(profile.max_retries + 1):
            try:
                completed = self._runner(
                    list(argv), cwd=str(workdir), env=self._safe_environment(workdir), timeout=profile.timeout_seconds,
                    input=prompt, capture_output=True, text=True, check=False,
                )
            except subprocess.TimeoutExpired as error:
                if attempt == profile.max_retries:
                    raise CodexTimeoutError("codex exec timed out") from error
                continue
            if completed.returncode == 0:
                return self._with_attempt(completed, attempt + 1)
            if attempt == profile.max_retries:
                return self._with_attempt(completed, attempt + 1)
        raise AssertionError("unreachable retry state")

    @staticmethod
    def _with_attempt(completed: subprocess.CompletedProcess[str], attempts: int) -> subprocess.CompletedProcess[str]:
        # CompletedProcess intentionally has no extension field.  The private
        # marker remains process-local and carries no prompt or response bytes.
        setattr(completed, "_paper_agent_attempts", attempts)
        return completed

    @staticmethod
    def _attempt_count(completed: subprocess.CompletedProcess[str]) -> int:
        return int(getattr(completed, "_paper_agent_attempts", 1))

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

    def _write_schema_bundle(self, path: Path, output_schema: Mapping[str, Any]) -> None:
        """Make local schema references resolvable without exposing project files."""
        for source in self._schema_root.glob("*.schema.json"):
            (path.parent / source.name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        path.write_text(json.dumps(output_schema, ensure_ascii=False, sort_keys=True), encoding="utf-8")

    def _frozen_schema(self, schema_name: str) -> dict[str, Any]:
        path = self._schema_root / schema_name
        if not path.is_file() or path.parent != self._schema_root:
            raise CodexOutputError(f"frozen schema is missing: {schema_name}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise CodexOutputError(f"frozen schema must be a JSON object: {schema_name}")
        return value

    def _frozen_prompt(self, prompt_name: str) -> str:
        path = self._prompt_root / prompt_name
        if not path.is_file() or path.parent != self._prompt_root:
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
            "required": ["ok"], "properties": {"ok": {"const": True}},
        }
        with TemporaryDirectory(prefix="paper-agent-codex-probe-", dir=self._temporary_root) as directory:
            workdir = Path(directory)
            schema_path = workdir / "output-schema.json"
            result_path = workdir / "last-message.json"
            schema_path.write_text(json.dumps(schema), encoding="utf-8")
            argv = self._argv(profile, workdir, schema_path, result_path)
            try:
                completed = self._run_with_retry(
                    argv, workdir, profile,
                    "Return only the structured confirmation requested by the output schema.",
                )
            except CodexExecError:
                return "unavailable"
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
    def _actual_invocation_metadata(stdout: str | None) -> tuple[str | None, str | None]:
        """Read an exposed JSONL model field without retaining event contents."""
        if not stdout:
            return None
        models: set[str] = set()
        profiles: set[str] = set()
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
        if len(models) > 1:
            raise CodexModelMismatchError("codex invocation metadata reported multiple models")
        if len(profiles) > 1:
            raise CodexModelMismatchError("codex invocation metadata reported multiple profiles")
        return next(iter(models), None), next(iter(profiles), None)
