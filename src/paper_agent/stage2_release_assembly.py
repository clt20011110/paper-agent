"""Fail-closed assembly of a Stage 2 schema-v3 release envelope."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import stat
from types import MappingProxyType
from typing import Any, Mapping

from .canonical import content_hash
from .stage2_release_evidence import (
    Stage2ReleaseEvidenceIndex,
    load_stage2_release_evidence_index_bytes,
)
from .stage2_search import (
    Stage2ReleaseError,
    _load_deployment_hidden_trust,
    _load_stage2_benchmark_candidate_bytes,
    verify_stage2_release_evidence_index,
)


class Stage2ReleaseAssemblyError(ValueError):
    """A candidate cannot safely be assembled into a Stage 2 v3 release."""


@dataclass(frozen=True, slots=True)
class AssembledStage2Release:
    """Public, non-secret provenance emitted after a successful assembly."""

    candidate_id: str
    evaluation_manifest_hash: str
    query_plan_config_hash: str
    query_plan_thresholds_hash: str
    release_path: Path
    release_sha256: str
    evidence_path: Path
    evidence_sha256: str
    throughput_runs: tuple[float, float, float]
    gate_hashes: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "gate_hashes", MappingProxyType(dict(self.gate_hashes)))

    def summary(self) -> dict[str, Any]:
        """Return only public provenance suitable for an operator log."""

        return {
            "candidate_id": self.candidate_id,
            "evaluation_manifest_hash": self.evaluation_manifest_hash,
            "expected_query_plan": {
                "config_hash": self.query_plan_config_hash,
                "thresholds_hash": self.query_plan_thresholds_hash,
            },
            "release": {
                "path": str(self.release_path),
                "sha256": self.release_sha256,
            },
            "evidence": {
                "path": str(self.evidence_path),
                "sha256": self.evidence_sha256,
            },
            "throughput_runs": list(self.throughput_runs),
            "gate_hashes": dict(self.gate_hashes),
        }


def assemble_stage2_release(
    candidate_path: Path,
    evidence_path: Path,
    hidden_trust_path: Path,
    output_path: Path,
) -> AssembledStage2Release:
    """Assemble a v3 release only after every public and hidden gate passes.

    The candidate and the final evidence index must be in one release bundle.
    The trust manifest is an explicit deployment-owned input and must live
    outside that bundle.  QueryPlan approval is intentionally not accepted
    here: its configuration hash includes the envelope built by this function.
    """

    return _verify_stage2_release_assembly(
        candidate_path,
        evidence_path,
        hidden_trust_path,
        output_path,
        write_output=True,
    )


def validate_stage2_release_assembly(
    candidate_path: Path,
    evidence_path: Path,
    hidden_trust_path: Path,
    output_path: Path,
) -> AssembledStage2Release:
    """Validate a prospective v3 release without creating any output.

    This follows the exact candidate, evidence, deployment-trust, and release
    gate verification path used by :func:`assemble_stage2_release`.  The
    prospective canonical release bytes and their digest are computed in
    memory, but the output path and its parent are never created.
    """

    return _verify_stage2_release_assembly(
        candidate_path,
        evidence_path,
        hidden_trust_path,
        output_path,
        write_output=False,
    )


def _verify_stage2_release_assembly(
    candidate_path: Path,
    evidence_path: Path,
    hidden_trust_path: Path,
    output_path: Path,
    *,
    write_output: bool,
) -> AssembledStage2Release:
    try:
        candidate_path = candidate_path.resolve(strict=True)
        evidence_path = evidence_path.resolve(strict=True)
    except OSError as error:
        raise Stage2ReleaseAssemblyError(
            f"Stage 2 release assembly input cannot be resolved: {error}"
        ) from error
    output_path = output_path.absolute()
    bundle_root = candidate_path.parent

    _require_exact_output_parent(output_path, bundle_root)
    _require_contained_file(evidence_path, bundle_root, "Stage 2 evidence index")
    bundle_fd = _open_bundle_dir(bundle_root)
    try:
        _require_missing_output(bundle_fd, output_path.name, output_path)
        candidate_bytes = _read_bytes(candidate_path, "Stage 2 benchmark candidate")
        evidence_bytes = _read_bytes(evidence_path, "Stage 2 evidence index")
        candidate_document = _json_object(candidate_bytes, "Stage 2 benchmark candidate")
        try:
            hidden_trust = _load_deployment_hidden_trust(
                hidden_trust_path,
                bundle_root=bundle_root,
            )
            candidate = _load_stage2_benchmark_candidate_bytes(
                candidate_path,
                candidate_bytes,
            )
            index = load_stage2_release_evidence_index_bytes(
                evidence_path,
                evidence_bytes,
            )
            _require_evidence_contained(index, bundle_root)
            gate = verify_stage2_release_evidence_index(
                index,
                candidate_id=candidate.profile_name,
                evaluation_manifest_hash=index.evaluation_manifest_hash,
                profile=candidate.profile,
                hidden_trust=hidden_trust,
            )
        except (OSError, Stage2ReleaseError, ValueError) as error:
            raise Stage2ReleaseAssemblyError(
                f"Stage 2 release assembly verification failed: {error}"
            ) from error

        evidence_sha256 = sha256(evidence_bytes).hexdigest()
        release_gate = {
            "candidate_id": candidate.profile_name,
            "evaluation_manifest_hash": index.evaluation_manifest_hash,
            "evidence": {
                "path": evidence_path.relative_to(bundle_root).as_posix(),
                "sha256": evidence_sha256,
            },
        }
        release_document = {
            **candidate_document,
            "schema_version": "3",
            "release_gate": release_gate,
        }
        release_bytes = _canonical_output_bytes(release_document)
        if write_output:
            _write_new(bundle_fd, output_path.name, output_path, release_bytes)
    finally:
        os.close(bundle_fd)

    released_profile = replace(
        candidate.profile,
        release_gate_hash=content_hash(release_gate),
    )
    return AssembledStage2Release(
        candidate_id=candidate.profile_name,
        evaluation_manifest_hash=index.evaluation_manifest_hash,
        query_plan_config_hash=released_profile.config_hash,
        query_plan_thresholds_hash=released_profile.threshold_hash,
        release_path=output_path,
        release_sha256=sha256(release_bytes).hexdigest(),
        evidence_path=evidence_path,
        evidence_sha256=evidence_sha256,
        throughput_runs=gate.throughput_runs,
        gate_hashes=gate.artifact_hashes,
    )


def _require_exact_output_parent(output_path: Path, bundle_root: Path) -> None:
    if output_path.parent != bundle_root:
        raise Stage2ReleaseAssemblyError(
            "Stage 2 release output must have exactly the benchmark candidate parent"
        )


def _require_missing_output(bundle_fd: int, name: str, path: Path) -> None:
    try:
        os.stat(name, dir_fd=bundle_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    raise Stage2ReleaseAssemblyError(f"Stage 2 release output already exists: {path}")


def _require_contained_file(path: Path, bundle_root: Path, label: str) -> None:
    if not path.is_relative_to(bundle_root):
        raise Stage2ReleaseAssemblyError(f"{label} must stay inside the release bundle")
    if not path.is_file():
        raise Stage2ReleaseAssemblyError(f"{label} is required: {path}")


def _require_evidence_contained(
    index: Stage2ReleaseEvidenceIndex,
    bundle_root: Path,
) -> None:
    if index.hidden_attestation is None:
        raise Stage2ReleaseAssemblyError(
            "Stage 2 release evidence requires a hidden attestation"
        )
    refs = [index.gold_manifest, index.hidden_attestation]
    for gate in index.public_gates.values():
        refs.append(gate.manifest)
        refs.extend(gate.records)
        if gate.papers is not None:
            refs.append(gate.papers)
    for ref in refs:
        if not ref.resolve(index.bundle_root).is_relative_to(bundle_root):
            raise Stage2ReleaseAssemblyError(
                "Stage 2 evidence artifacts must stay inside the release bundle"
            )


def _read_bytes(path: Path, label: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise Stage2ReleaseAssemblyError(f"{label} cannot be read: {error}") from error


def _json_object(payload: bytes, label: str) -> dict[str, Any]:
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Stage2ReleaseAssemblyError(f"{label} is not valid JSON: {error}") from error
    if not isinstance(document, dict):
        raise Stage2ReleaseAssemblyError(f"{label} must be an object")
    return document


def _canonical_output_bytes(document: Mapping[str, Any]) -> bytes:
    return (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _open_bundle_dir(path: Path) -> int:
    try:
        expected = path.stat()
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
    except OSError as error:
        raise Stage2ReleaseAssemblyError(
            f"cannot open Stage 2 release bundle directory: {error}"
        ) from error
    opened = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(opened.st_mode)
        or (opened.st_dev, opened.st_ino) != (expected.st_dev, expected.st_ino)
    ):
        os.close(descriptor)
        raise Stage2ReleaseAssemblyError(
            "Stage 2 release bundle directory changed while opening"
        )
    return descriptor


def _write_new(bundle_fd: int, name: str, path: Path, payload: bytes) -> None:
    descriptor: int | None = None
    created = False
    try:
        descriptor = os.open(
            name,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o666,
            dir_fd=bundle_fd,
        )
        created = True
        with os.fdopen(descriptor, "wb") as output:
            descriptor = None
            written = output.write(payload)
            if written != len(payload):
                raise OSError("short write while creating Stage 2 release")
            output.flush()
            os.fsync(output.fileno())
    except FileExistsError as error:
        raise Stage2ReleaseAssemblyError(
            f"Stage 2 release output already exists: {path}"
        ) from error
    except OSError as error:
        if created:
            try:
                os.unlink(name, dir_fd=bundle_fd)
            except OSError:
                pass
        raise Stage2ReleaseAssemblyError(f"cannot write Stage 2 release output: {error}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
