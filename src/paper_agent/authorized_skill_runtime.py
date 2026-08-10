"""Offline, fail-closed trust boundary for an authorized browser skill.

This module deliberately does not invoke a browser, read browser state, or
persist anything.  Callers must opt in, provide a source archive and one or
more installation roots, then use :meth:`require_ready` before dispatching
any provider work.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import sysconfig
from typing import Iterable

import yaml


class AuthorizedSkillRuntimeError(ValueError):
    """Raised when an authorized skill cannot pass its immutable audit."""


@dataclass(frozen=True, slots=True)
class AuditRecord:
    """The checked-in, immutable facts to which an installed skill is bound."""

    schema_version: str
    skill_name: str
    skill_version: str | None
    audit_version: str
    original_zip_sha256: str
    installed_content_sha256: str
    dependency_lock_sha256: str
    dependency_lock_files: tuple[str, ...]
    files: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class AuthorizedSkillDoctorResult:
    """A non-sensitive, immutable result of an offline trust check."""

    enabled: bool
    ready: bool
    reasons: tuple[str, ...]
    audit: AuditRecord
    installed_path: Path | None
    discovered_name: str | None
    discovered_version: str | None
    original_zip_sha256: str | None
    installed_content_sha256: str | None
    dependency_lock_sha256: str | None


def audit_manifest_path(override: Path | None = None) -> Path:
    """Locate the checked-in audit record without relying on a user directory."""

    if override is not None:
        return override
    source_root = Path(__file__).resolve().parents[2]
    candidate = source_root / "policies" / "download-authorized-papers-skill-audit-v1.yaml"
    if candidate.is_file():
        return candidate
    return Path(sysconfig.get_path("data")) / "share" / "paper-agent" / "policies" / candidate.name


def load_audit_record(path: Path | None = None) -> AuditRecord:
    """Load and validate the compact, checked-in audit manifest."""

    manifest_path = audit_manifest_path(path)
    try:
        document = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise AuthorizedSkillRuntimeError(f"audit manifest is unavailable: {manifest_path}") from error
    if not isinstance(document, dict):
        raise AuthorizedSkillRuntimeError("audit manifest must be an object")

    expected_keys = {
        "schema_version", "skill_name", "skill_version", "audit_version", "original_zip_sha256",
        "installed_content_sha256", "dependency_lock_sha256", "dependency_lock_files", "files",
    }
    if set(document) != expected_keys:
        raise AuthorizedSkillRuntimeError("audit manifest has an unexpected shape")
    if document["schema_version"] != "1":
        raise AuthorizedSkillRuntimeError("unsupported audit manifest schema")
    if not isinstance(document["skill_name"], str) or not document["skill_name"]:
        raise AuthorizedSkillRuntimeError("audit manifest has no skill name")
    if document["skill_version"] is not None and not isinstance(document["skill_version"], str):
        raise AuthorizedSkillRuntimeError("audit manifest skill version is invalid")
    if not isinstance(document["audit_version"], str) or not document["audit_version"]:
        raise AuthorizedSkillRuntimeError("audit manifest has no audit version")
    for key in ("original_zip_sha256", "installed_content_sha256", "dependency_lock_sha256"):
        _require_digest(document[key], key)
    dependency_files = _validate_paths(document["dependency_lock_files"], "dependency lock files")
    file_digests = _validate_file_digests(document["files"])
    if "SKILL.md" not in dict(file_digests):
        raise AuthorizedSkillRuntimeError("audit manifest must include SKILL.md")
    return AuditRecord(
        schema_version=document["schema_version"],
        skill_name=document["skill_name"],
        skill_version=document["skill_version"],
        audit_version=document["audit_version"],
        original_zip_sha256=document["original_zip_sha256"],
        installed_content_sha256=document["installed_content_sha256"],
        dependency_lock_sha256=document["dependency_lock_sha256"],
        dependency_lock_files=dependency_files,
        files=file_digests,
    )


def discover_skill(roots: Iterable[Path], skill_name: str) -> Path:
    """Find exactly one installed SKILL.md by its declared frontmatter name."""

    matches: list[Path] = []
    for root in roots:
        if root.is_file() and root.name == "SKILL.md":
            candidates = (root,)
        elif root.is_dir():
            candidates = tuple(root.rglob("SKILL.md"))
        else:
            continue
        for candidate in candidates:
            if candidate.is_symlink() or not candidate.is_file():
                continue
            try:
                name, _ = _skill_metadata(candidate)
            except AuthorizedSkillRuntimeError:
                continue
            if name == skill_name:
                matches.append(candidate.parent)
    unique_matches = sorted({item.resolve() for item in matches})
    if not unique_matches:
        raise AuthorizedSkillRuntimeError(f"installed skill not found: {skill_name}")
    if len(unique_matches) != 1:
        raise AuthorizedSkillRuntimeError(f"multiple installed skills match: {skill_name}")
    return unique_matches[0]


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def content_digest(root: Path) -> tuple[str, tuple[tuple[str, str], ...]]:
    """Digest an installed tree as sorted path/NUL/content/NUL records."""

    if root.is_symlink() or not root.is_dir():
        raise AuthorizedSkillRuntimeError("installed skill root is not a real directory")
    records: list[tuple[str, str]] = []
    for candidate in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if candidate.is_symlink():
            raise AuthorizedSkillRuntimeError("installed skill contains a symlink")
        if candidate.is_dir():
            continue
        if not candidate.is_file():
            raise AuthorizedSkillRuntimeError("installed skill contains a non-regular file")
        relative = candidate.relative_to(root).as_posix()
        records.append((relative, sha256_file(candidate)))

    digest = sha256()
    for relative, _ in records:
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with (root / relative).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest(), tuple(records)


def dependency_lock_digest(root: Path, lock_files: Iterable[str]) -> str:
    """Digest only the explicitly audited lockfile set, in stable path order."""

    digest = sha256()
    for relative in sorted(lock_files):
        candidate = root / relative
        if candidate.is_symlink() or not candidate.is_file():
            raise AuthorizedSkillRuntimeError(f"dependency lockfile is missing: {relative}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with candidate.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


class AuthorizedSkillRuntime:
    """Validates a locally installed skill before an opt-in integration may run."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        skill_roots: Iterable[Path] = (),
        original_zip: Path | None = None,
        audit_manifest: Path | None = None,
    ) -> None:
        self.enabled = enabled
        self._skill_roots = tuple(Path(root) for root in skill_roots)
        self._original_zip = Path(original_zip) if original_zip is not None else None
        self._audit = load_audit_record(audit_manifest)

    @property
    def audit_record(self) -> AuditRecord:
        return self._audit

    def doctor(self) -> AuthorizedSkillDoctorResult:
        if not self.enabled:
            return AuthorizedSkillDoctorResult(
                enabled=False, ready=False, reasons=("authorized skill provider is disabled",), audit=self._audit,
                installed_path=None, discovered_name=None, discovered_version=None, original_zip_sha256=None,
                installed_content_sha256=None, dependency_lock_sha256=None,
            )

        reasons: list[str] = []
        archive_digest = self._verify_archive(reasons)
        installed_path, discovered_name, discovered_version = self._discover_installed(reasons)
        installed_digest: str | None = None
        lock_digest: str | None = None
        if installed_path is not None:
            try:
                installed_digest, installed_files = content_digest(installed_path)
                self._verify_installed_files(installed_files, reasons)
                if installed_digest != self._audit.installed_content_sha256:
                    reasons.append("installed content digest has drifted")
                lock_digest = dependency_lock_digest(installed_path, self._audit.dependency_lock_files)
                if lock_digest != self._audit.dependency_lock_sha256:
                    reasons.append("dependency lock digest has drifted")
            except (OSError, AuthorizedSkillRuntimeError) as error:
                reasons.append(str(error))
        return AuthorizedSkillDoctorResult(
            enabled=True, ready=not reasons, reasons=tuple(reasons), audit=self._audit,
            installed_path=installed_path, discovered_name=discovered_name, discovered_version=discovered_version,
            original_zip_sha256=archive_digest, installed_content_sha256=installed_digest,
            dependency_lock_sha256=lock_digest,
        )

    def require_ready(self) -> AuthorizedSkillDoctorResult:
        """Return a verified result or reject the integration before side effects."""

        result = self.doctor()
        if not result.ready:
            raise AuthorizedSkillRuntimeError("authorized skill is unavailable: " + "; ".join(result.reasons))
        return result

    def _verify_archive(self, reasons: list[str]) -> str | None:
        if self._original_zip is None or not self._original_zip.is_file() or self._original_zip.is_symlink():
            reasons.append("original skill archive is unavailable")
            return None
        digest = sha256_file(self._original_zip)
        if digest != self._audit.original_zip_sha256:
            reasons.append("original skill archive digest has drifted")
        return digest

    def _discover_installed(self, reasons: list[str]) -> tuple[Path | None, str | None, str | None]:
        try:
            installed_path = discover_skill(self._skill_roots, self._audit.skill_name)
            discovered_name, discovered_version = _skill_metadata(installed_path / "SKILL.md")
        except (OSError, AuthorizedSkillRuntimeError) as error:
            reasons.append(str(error))
            return None, None, None
        if discovered_name != self._audit.skill_name:
            reasons.append("installed skill name does not match audit record")
        if discovered_version != self._audit.skill_version:
            reasons.append("installed skill version does not match audit record")
        return installed_path, discovered_name, discovered_version

    def _verify_installed_files(self, installed_files: tuple[tuple[str, str], ...], reasons: list[str]) -> None:
        expected = dict(self._audit.files)
        actual = dict(installed_files)
        if set(actual) != set(expected):
            reasons.append("installed skill file set has drifted")
        for relative in sorted(set(actual).intersection(expected)):
            if actual[relative] != expected[relative]:
                reasons.append(f"installed skill file digest has drifted: {relative}")


def _skill_metadata(path: Path) -> tuple[str, str | None]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise AuthorizedSkillRuntimeError("installed SKILL.md is unreadable") from error
    if not text.startswith("---\n"):
        raise AuthorizedSkillRuntimeError("installed SKILL.md has no frontmatter")
    closing = text.find("\n---", 4)
    if closing < 0:
        raise AuthorizedSkillRuntimeError("installed SKILL.md has invalid frontmatter")
    try:
        frontmatter = yaml.safe_load(text[4:closing])
    except yaml.YAMLError as error:
        raise AuthorizedSkillRuntimeError("installed SKILL.md has invalid frontmatter") from error
    if not isinstance(frontmatter, dict) or not isinstance(frontmatter.get("name"), str):
        raise AuthorizedSkillRuntimeError("installed SKILL.md has no name")
    version = frontmatter.get("version")
    if version is not None and not isinstance(version, str):
        raise AuthorizedSkillRuntimeError("installed SKILL.md has invalid version")
    return frontmatter["name"], version


def _require_digest(value: object, label: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise AuthorizedSkillRuntimeError(f"audit manifest {label} is not a SHA-256 digest")


def _validate_paths(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) != len(set(value)):
        raise AuthorizedSkillRuntimeError(f"audit manifest {label} must be a unique list")
    values = tuple(value)
    if any(not isinstance(item, str) or not _safe_relative_path(item) for item in values):
        raise AuthorizedSkillRuntimeError(f"audit manifest {label} contains an invalid path")
    return values


def _validate_file_digests(value: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, dict) or not value:
        raise AuthorizedSkillRuntimeError("audit manifest files must be an object")
    entries = tuple(sorted(value.items()))
    for relative, digest in entries:
        if not isinstance(relative, str) or not _safe_relative_path(relative):
            raise AuthorizedSkillRuntimeError("audit manifest contains an invalid file path")
        _require_digest(digest, f"file digest for {relative}")
    return entries


def _safe_relative_path(value: str) -> bool:
    path = Path(value)
    return not path.is_absolute() and ".." not in path.parts and value not in {"", "."}
