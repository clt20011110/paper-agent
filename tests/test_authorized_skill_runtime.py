from __future__ import annotations

from dataclasses import FrozenInstanceError
from hashlib import sha256
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest
import yaml

from paper_agent.authorized_skill_runtime import (
    AuthorizedSkillRuntime,
    AuthorizedSkillRuntimeError,
    content_digest,
    load_audit_record,
)


def _write_skill(root: Path, *, name: str = "audited-skill", version: str | None = "1.0") -> Path:
    skill = root / "nested" / "install"
    (skill / "scripts").mkdir(parents=True)
    version_line = "" if version is None else f'version: "{version}"\n'
    (skill / "SKILL.md").write_text(f"---\nname: {name}\n{version_line}---\n\n# Skill\n", encoding="utf-8")
    (skill / "scripts" / "run.py").write_text("print('offline only')\n", encoding="utf-8")
    return skill


def _write_archive(path: Path) -> None:
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        archive.writestr("audited-skill/SKILL.md", "source archive only")


def _write_manifest(path: Path, skill: Path, archive: Path, *, name: str = "audited-skill") -> None:
    installed_digest, files = content_digest(skill)
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "1",
                "skill_name": name,
                "skill_version": "1.0",
                "audit_version": "test.1",
                "original_zip_sha256": sha256(archive.read_bytes()).hexdigest(),
                "installed_content_sha256": installed_digest,
                "dependency_lock_sha256": sha256(b"").hexdigest(),
                "dependency_lock_files": [],
                "files": dict(files),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _runtime(tmp_path: Path, *, enabled: bool = True) -> tuple[AuthorizedSkillRuntime, Path]:
    skill = _write_skill(tmp_path)
    archive = tmp_path / "source.zip"
    _write_archive(archive)
    manifest = tmp_path / "audit.yaml"
    _write_manifest(manifest, skill, archive)
    return AuthorizedSkillRuntime(
        enabled=enabled, skill_roots=(tmp_path,), original_zip=archive, audit_manifest=manifest
    ), skill


def test_disabled_by_default_does_not_require_an_archive_or_installed_skill(tmp_path: Path) -> None:
    manifest = tmp_path / "audit.yaml"
    skill = _write_skill(tmp_path)
    archive = tmp_path / "source.zip"
    _write_archive(archive)
    _write_manifest(manifest, skill, archive)

    result = AuthorizedSkillRuntime(audit_manifest=manifest).doctor()

    assert not result.enabled
    assert not result.ready
    assert result.reasons == ("authorized skill provider is disabled",)


def test_doctor_discovers_by_skill_name_and_returns_a_frozen_audit_record(tmp_path: Path) -> None:
    runtime, skill = _runtime(tmp_path)

    result = runtime.require_ready()

    assert result.ready
    assert result.installed_path == skill.resolve()
    assert result.discovered_name == "audited-skill"
    assert result.discovered_version == "1.0"
    assert result.installed_content_sha256 == runtime.audit_record.installed_content_sha256
    with pytest.raises(FrozenInstanceError):
        result.audit.skill_name = "other"  # type: ignore[misc]


def test_doctor_fails_closed_for_archive_or_installed_content_drift(tmp_path: Path) -> None:
    runtime, skill = _runtime(tmp_path)
    archive = tmp_path / "source.zip"
    archive.write_bytes(b"changed archive")
    (skill / "scripts" / "run.py").write_text("changed\n", encoding="utf-8")

    result = runtime.doctor()

    assert not result.ready
    assert "original skill archive digest has drifted" in result.reasons
    assert "installed content digest has drifted" in result.reasons
    assert any("scripts/run.py" in reason for reason in result.reasons)
    with pytest.raises(AuthorizedSkillRuntimeError, match="unavailable"):
        runtime.require_ready()


def test_doctor_rejects_missing_or_ambiguous_named_skill(tmp_path: Path) -> None:
    runtime, _ = _runtime(tmp_path)
    extra = _write_skill(tmp_path / "second")
    assert extra.exists()

    result = runtime.doctor()

    assert not result.ready
    assert "multiple installed skills match: audited-skill" in result.reasons


def test_doctor_fails_closed_when_the_archive_or_skill_is_missing(tmp_path: Path) -> None:
    runtime, _ = _runtime(tmp_path)
    missing_archive = AuthorizedSkillRuntime(
        enabled=True, skill_roots=(tmp_path,), original_zip=tmp_path / "missing.zip",
        audit_manifest=tmp_path / "audit.yaml",
    ).doctor()
    missing_skill = AuthorizedSkillRuntime(
        enabled=True, original_zip=tmp_path / "source.zip", audit_manifest=tmp_path / "audit.yaml"
    ).doctor()

    assert not missing_archive.ready
    assert "original skill archive is unavailable" in missing_archive.reasons
    assert not missing_skill.ready
    assert "installed skill not found: audited-skill" in missing_skill.reasons


def test_checked_in_record_matches_the_audited_archive_facts() -> None:
    record = load_audit_record()

    assert record.skill_name == "download-authorized-papers"
    assert record.audit_version == "2026-08-09.1"
    assert record.original_zip_sha256 == "ee69308c98ad8e564ee8098acc56628866c45657da259aa08bb29f8732874d5e"
    assert record.dependency_lock_files == ()
