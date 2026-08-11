from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import errno
from hashlib import sha256
import json
from pathlib import Path
import shutil

import pytest

import paper_agent.stage2_release_assembly as stage2_release_assembly
from paper_agent.stage2_release_assembly import (
    Stage2ReleaseAssemblyError,
    assemble_stage2_release,
    validate_stage2_release_assembly,
)
from paper_agent.stage2_search import load_stage2_release

from test_stage2_release_v3 import V3Bundle, _build_v3_bundle


def _write_json(path: Path, document: object) -> None:
    path.write_text(json.dumps(document, sort_keys=True) + "\n", encoding="utf-8")


def _ref(path: Path) -> dict[str, str]:
    return {"path": path.name, "sha256": sha256(path.read_bytes()).hexdigest()}


@pytest.fixture(scope="module")
def assembly_template(tmp_path_factory: pytest.TempPathFactory) -> V3Bundle:
    return _build_v3_bundle(tmp_path_factory.mktemp("stage2-release-assembly"))


@pytest.fixture
def assembly_bundle(assembly_template: V3Bundle, tmp_path: Path) -> V3Bundle:
    root = tmp_path / "bundle"
    shutil.copytree(assembly_template.root, root)
    trust_path = tmp_path / "deployment-hidden-evaluator-trust.json"
    shutil.copy2(assembly_template.trust_path, trust_path)
    return replace(
        assembly_template,
        root=root,
        release_path=root / assembly_template.release_path.name,
        trust_path=trust_path,
        plan=deepcopy(assembly_template.plan),
    )


def test_assemble_v3_release_from_verified_candidate_and_evidence(
    assembly_bundle: V3Bundle,
) -> None:
    candidate_path = assembly_bundle.root / "candidate.json"
    evidence_path = assembly_bundle.root / "stage2-release-evidence.json"
    output_path = assembly_bundle.root / "assembled-stage2-release.json"

    result = assemble_stage2_release(
        candidate_path,
        evidence_path,
        assembly_bundle.trust_path,
        output_path,
    )

    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    released = json.loads(output_path.read_text(encoding="utf-8"))
    assert released == {
        **candidate,
        "schema_version": "3",
        "release_gate": {
            "candidate_id": candidate["profile"],
            "evaluation_manifest_hash": result.evaluation_manifest_hash,
            "evidence": _ref(evidence_path),
        },
    }
    assert result.release_path == output_path.resolve()
    assert result.evidence_path == evidence_path.resolve()
    assert set(result.gate_hashes) == {
        "promotion", "structured_replay", "rationale", "parity", "benchmark", "soak",
    }
    assert result.summary()["expected_query_plan"] == {
        "config_hash": result.query_plan_config_hash,
        "thresholds_hash": result.query_plan_thresholds_hash,
    }

    loaded = load_stage2_release(
        output_path,
        assembly_bundle.plan,
        hidden_trust_path=assembly_bundle.trust_path,
    )
    assert loaded.profile.config_hash == result.query_plan_config_hash
    assert loaded.profile.threshold_hash == result.query_plan_thresholds_hash


def test_validate_assembly_runs_all_gates_without_writing(
    assembly_bundle: V3Bundle,
) -> None:
    candidate_path = assembly_bundle.root / "candidate.json"
    evidence_path = assembly_bundle.root / "stage2-release-evidence.json"
    output_path = assembly_bundle.root / "assembled-stage2-release.json"

    validation = validate_stage2_release_assembly(
        candidate_path,
        evidence_path,
        assembly_bundle.trust_path,
        output_path,
    )

    assert not output_path.exists()
    assembled = assemble_stage2_release(
        candidate_path,
        evidence_path,
        assembly_bundle.trust_path,
        output_path,
    )
    assert validation.summary() == assembled.summary()
    assert sha256(output_path.read_bytes()).hexdigest() == validation.release_sha256


def test_validate_assembly_does_not_create_an_invalid_output_parent(
    assembly_bundle: V3Bundle,
) -> None:
    missing_parent = assembly_bundle.root / "missing"

    with pytest.raises(Stage2ReleaseAssemblyError, match="exactly the benchmark"):
        validate_stage2_release_assembly(
            assembly_bundle.root / "candidate.json",
            assembly_bundle.root / "stage2-release-evidence.json",
            assembly_bundle.trust_path,
            missing_parent / "assembled-stage2-release.json",
        )

    assert not missing_parent.exists()


def test_validate_assembly_fails_closed_on_untrusted_hidden_attestation(
    assembly_bundle: V3Bundle,
) -> None:
    trust = json.loads(assembly_bundle.trust_path.read_text(encoding="utf-8"))
    trust["keys"][0]["status"] = "revoked"
    _write_json(assembly_bundle.trust_path, trust)
    output_path = assembly_bundle.root / "assembled-stage2-release.json"

    with pytest.raises(Stage2ReleaseAssemblyError, match="verification failed"):
        validate_stage2_release_assembly(
            assembly_bundle.root / "candidate.json",
            assembly_bundle.root / "stage2-release-evidence.json",
            assembly_bundle.trust_path,
            output_path,
        )

    assert not output_path.exists()


@pytest.mark.parametrize(
    "kind",
    ("existing", "wrong_parent", "bundled_trust", "bundled_trust_symlink", "external_evidence"),
)
def test_assembly_rejects_unsafe_paths_before_writing(
    assembly_bundle: V3Bundle,
    tmp_path: Path,
    kind: str,
) -> None:
    candidate_path = assembly_bundle.root / "candidate.json"
    evidence_path = assembly_bundle.root / "stage2-release-evidence.json"
    trust_path = assembly_bundle.trust_path
    output_path = assembly_bundle.root / "assembled-stage2-release.json"
    if kind == "existing":
        output_path.write_text("do not replace", encoding="utf-8")
    elif kind == "wrong_parent":
        output_path = tmp_path / "assembled-stage2-release.json"
    elif kind == "bundled_trust":
        trust_path = assembly_bundle.root / "hidden-evaluator-trust.json"
    elif kind == "bundled_trust_symlink":
        trust_path = assembly_bundle.root / "deployment-trust-link.json"
        trust_path.symlink_to(assembly_bundle.trust_path)
    else:
        evidence_path = tmp_path / "stage2-release-evidence.json"
        evidence_path.write_text("{}", encoding="utf-8")

    with pytest.raises(Stage2ReleaseAssemblyError):
        assemble_stage2_release(candidate_path, evidence_path, trust_path, output_path)

    if kind != "existing":
        assert not output_path.exists()


def test_assembly_recomputes_the_full_public_evidence_gates(
    assembly_bundle: V3Bundle,
) -> None:
    evidence_path = assembly_bundle.root / "stage2-release-evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    records_path = assembly_bundle.root / "rationale-records.json"
    records = json.loads(records_path.read_text(encoding="utf-8"))
    for record in records["records"][:6]:
        record["evidence_supported"] = False
    _write_json(records_path, records)
    evidence["public_gates"]["rationale"]["records"] = _ref(records_path)
    _write_json(evidence_path, evidence)

    with pytest.raises(Stage2ReleaseAssemblyError, match="public release gates did not pass"):
        assemble_stage2_release(
            assembly_bundle.root / "candidate.json",
            evidence_path,
            assembly_bundle.trust_path,
            assembly_bundle.root / "assembled-stage2-release.json",
        )


def test_assembly_uses_one_candidate_and_evidence_index_snapshot(
    assembly_bundle: V3Bundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_path = assembly_bundle.root / "candidate.json"
    evidence_path = assembly_bundle.root / "stage2-release-evidence.json"
    candidate_document = json.loads(candidate_path.read_text(encoding="utf-8"))
    original_candidate_loader = (
        stage2_release_assembly._load_stage2_benchmark_candidate_bytes
    )
    original_evidence_loader = (
        stage2_release_assembly.load_stage2_release_evidence_index_bytes
    )

    def load_candidate_then_replace(path: Path, payload: bytes):
        candidate = original_candidate_loader(path, payload)
        path.write_text("{}\n", encoding="utf-8")
        return candidate

    def load_evidence_then_replace(path: Path, payload: bytes):
        index = original_evidence_loader(path, payload)
        path.write_text("{}\n", encoding="utf-8")
        return index

    monkeypatch.setattr(
        stage2_release_assembly,
        "_load_stage2_benchmark_candidate_bytes",
        load_candidate_then_replace,
    )
    monkeypatch.setattr(
        stage2_release_assembly,
        "load_stage2_release_evidence_index_bytes",
        load_evidence_then_replace,
    )

    output_path = assembly_bundle.root / "assembled-stage2-release.json"
    assemble_stage2_release(
        candidate_path,
        evidence_path,
        assembly_bundle.trust_path,
        output_path,
    )

    release = json.loads(output_path.read_text(encoding="utf-8"))
    assert {
        key: release[key] for key in candidate_document if key != "schema_version"
    } == {
        key: candidate_document[key]
        for key in candidate_document
        if key != "schema_version"
    }


def test_assembly_exclusive_write_rejects_a_late_leaf_symlink(
    assembly_bundle: V3Bundle,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = assembly_bundle.root / "assembled-stage2-release.json"
    victim = tmp_path / "victim.json"
    victim.write_text("unchanged", encoding="utf-8")
    original = stage2_release_assembly.verify_stage2_release_evidence_index

    def verify_then_link(*args, **kwargs):
        result = original(*args, **kwargs)
        output_path.symlink_to(victim)
        return result

    monkeypatch.setattr(
        stage2_release_assembly,
        "verify_stage2_release_evidence_index",
        verify_then_link,
    )

    with pytest.raises(Stage2ReleaseAssemblyError, match="already exists"):
        assemble_stage2_release(
            assembly_bundle.root / "candidate.json",
            assembly_bundle.root / "stage2-release-evidence.json",
            assembly_bundle.trust_path,
            output_path,
        )

    assert victim.read_text(encoding="utf-8") == "unchanged"


def test_assembly_writes_through_the_validated_dirfd_after_parent_swap(
    assembly_bundle: V3Bundle,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_name = "assembled-stage2-release.json"
    output_path = assembly_bundle.root / output_name
    validated_bundle = tmp_path / "validated-bundle"
    replacement_bundle = tmp_path / "replacement-bundle"
    replacement_bundle.mkdir()
    original = stage2_release_assembly.verify_stage2_release_evidence_index

    def verify_then_swap(*args, **kwargs):
        result = original(*args, **kwargs)
        assembly_bundle.root.rename(validated_bundle)
        assembly_bundle.root.symlink_to(replacement_bundle, target_is_directory=True)
        return result

    monkeypatch.setattr(
        stage2_release_assembly,
        "verify_stage2_release_evidence_index",
        verify_then_swap,
    )

    assemble_stage2_release(
        assembly_bundle.root / "candidate.json",
        assembly_bundle.root / "stage2-release-evidence.json",
        assembly_bundle.trust_path,
        output_path,
    )

    assert (validated_bundle / output_name).is_file()
    assert not (replacement_bundle / output_name).exists()


def test_assembly_removes_its_partial_output_after_disk_failure(
    assembly_bundle: V3Bundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = assembly_bundle.root / "assembled-stage2-release.json"

    def fail_fsync(_descriptor: int) -> None:
        raise OSError(errno.ENOSPC, "simulated disk full")

    monkeypatch.setattr(stage2_release_assembly.os, "fsync", fail_fsync)

    with pytest.raises(Stage2ReleaseAssemblyError, match="simulated disk full"):
        assemble_stage2_release(
            assembly_bundle.root / "candidate.json",
            assembly_bundle.root / "stage2-release-evidence.json",
            assembly_bundle.trust_path,
            output_path,
        )

    assert not output_path.exists()
