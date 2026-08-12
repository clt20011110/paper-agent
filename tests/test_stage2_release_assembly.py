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
from paper_agent.approval import approved_content_hash
from paper_agent.domain import FilterStatus
from paper_agent.query_plan import approve_query_plan
from paper_agent.stage2_backends import OmlxResponse
from paper_agent.stage2_release_assembly import (
    Stage2ReleaseAssemblyError,
    assemble_stage2_release,
    validate_stage2_release_assembly,
)
from paper_agent.stage2_search import Stage2ReleaseError, load_stage2_release
from paper_agent.storage import Database

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
    parity_trust_path = tmp_path / "deployment-parity-oracle-trust.json"
    shutil.copy2(assembly_template.trust_path, trust_path)
    shutil.copy2(assembly_template.parity_trust_path, parity_trust_path)
    return replace(
        assembly_template,
        root=root,
        release_path=root / assembly_template.release_path.name,
        trust_path=trust_path,
        parity_trust_path=parity_trust_path,
        plan=deepcopy(assembly_template.plan),
    )


@pytest.fixture(scope="module")
def fallback_assembly_template(
    tmp_path_factory: pytest.TempPathFactory,
) -> V3Bundle:
    template_root = tmp_path_factory.mktemp("stage2-fallback-assembly")
    root = template_root / "bundle"
    backup_root = root / "backup"
    backup_root.mkdir(parents=True)
    repository_root = Path(__file__).parents[1]
    backup_lock = json.loads(
        (repository_root / "configs/stage2/models/bge-reranker-v2-m3-mlx-bf16.lock.json")
        .read_text(encoding="utf-8")
    )
    backup_lock["model_id"] = "bge-reranker-v2-m3-mlx-backup"
    backup_lock_path = template_root / "backup-reranker.lock.json"
    _write_json(backup_lock_path, backup_lock)
    promotion_batch_hash = "9" * 64
    _build_v3_bundle(
        backup_root,
        reranker_lock_source=backup_lock_path,
        profile_name="local-backup",
        release_role="qualified_fallback",
        winner_candidate_id="local-winner",
        promotion_batch_hash=promotion_batch_hash,
    )
    return _build_v3_bundle(
        root,
        profile_name="local-winner",
        promotion_batch_hash=promotion_batch_hash,
    )


@pytest.fixture
def fallback_assembly_bundle(
    fallback_assembly_template: V3Bundle,
    tmp_path: Path,
) -> V3Bundle:
    root = tmp_path / "bundle"
    shutil.copytree(fallback_assembly_template.root, root)
    trust_path = tmp_path / "deployment-hidden-evaluator-trust.json"
    parity_trust_path = tmp_path / "deployment-parity-oracle-trust.json"
    shutil.copy2(fallback_assembly_template.trust_path, trust_path)
    shutil.copy2(fallback_assembly_template.parity_trust_path, parity_trust_path)
    return replace(
        fallback_assembly_template,
        root=root,
        release_path=root / fallback_assembly_template.release_path.name,
        trust_path=trust_path,
        parity_trust_path=parity_trust_path,
        plan=deepcopy(fallback_assembly_template.plan),
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
        parity_oracle_trust_path=assembly_bundle.parity_trust_path,
    )

    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    released = json.loads(output_path.read_text(encoding="utf-8"))
    assert released == {
        **candidate,
        "schema_version": "3",
        "release_gate": {
            "candidate_id": candidate["profile"],
            "candidate_bundle_sha256": sha256(
                candidate_path.read_bytes()
            ).hexdigest(),
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
        parity_oracle_trust_path=assembly_bundle.parity_trust_path,
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
        parity_oracle_trust_path=assembly_bundle.parity_trust_path,
    )

    assert not output_path.exists()
    assembled = assemble_stage2_release(
        candidate_path,
        evidence_path,
        assembly_bundle.trust_path,
        output_path,
        parity_oracle_trust_path=assembly_bundle.parity_trust_path,
    )
    assert validation.summary() == assembled.summary()
    assert sha256(output_path.read_bytes()).hexdigest() == validation.release_sha256


def test_assemble_load_and_screen_with_same_batch_qualified_fallback(
    fallback_assembly_bundle: V3Bundle,
    tmp_path: Path,
) -> None:
    """Exercise real v2 evidence, v3 assembly, load, and fallback execution."""

    root = fallback_assembly_bundle.root
    backup_root = root / "backup"
    trust_path = fallback_assembly_bundle.trust_path
    parity_trust_path = fallback_assembly_bundle.parity_trust_path

    output_path = root / "assembled-with-fallback.json"
    validation = validate_stage2_release_assembly(
        root / "candidate.json",
        root / "stage2-release-evidence.json",
        trust_path,
        output_path,
        parity_oracle_trust_path=parity_trust_path,
        fallback_candidate_path=backup_root / "candidate.json",
        fallback_evidence_path=backup_root / "stage2-release-evidence.json",
    )
    assert not output_path.exists()
    result = assemble_stage2_release(
        root / "candidate.json",
        root / "stage2-release-evidence.json",
        trust_path,
        output_path,
        parity_oracle_trust_path=parity_trust_path,
        fallback_candidate_path=backup_root / "candidate.json",
        fallback_evidence_path=backup_root / "stage2-release-evidence.json",
    )
    assert validation.summary() == result.summary()
    document = json.loads(output_path.read_text(encoding="utf-8"))
    assert document["reranker_fallback"]["candidate"]["path"] == (
        "backup/candidate.json"
    )
    assert result.fallback is not None

    with pytest.raises(Stage2ReleaseError, match="effective release configuration"):
        load_stage2_release(
            output_path,
            fallback_assembly_bundle.plan,
            hidden_trust_path=trust_path,
            parity_oracle_trust_path=parity_trust_path,
        )

    draft = deepcopy(fallback_assembly_bundle.plan)
    draft["status"] = "draft"
    draft["approval"] = None
    draft["filter"]["config_hash"] = result.query_plan_config_hash
    draft["filter"]["thresholds_hash"] = result.query_plan_thresholds_hash
    draft["plan_hash"] = approved_content_hash(draft)
    plan = approve_query_plan(
        draft,
        draft["plan_hash"],
        approved_by="test-owner",
        approved_at="2026-08-12T00:00:00Z",
    )
    released = load_stage2_release(
        output_path,
        plan,
        hidden_trust_path=trust_path,
        parity_oracle_trust_path=parity_trust_path,
    )
    assert released.reranker_fallback is not None
    assert released.effective_config_hash == result.query_plan_config_hash
    assert released.effective_config_hash != released.profile.config_hash

    class PrimaryFails:
        def __init__(self) -> None:
            self.models: list[str] = []

        def request(self, path: str, payload: dict) -> OmlxResponse:
            assert path == "/v1/rerank"
            model = payload["model"]
            self.models.append(model)
            if model == released.profile.reranker_model_id:
                return OmlxResponse(503, b"{}")
            assert model == "bge-reranker-v2-m3-mlx-backup"
            return OmlxResponse(
                200,
                json.dumps({
                    "model": model,
                    "results": [{"index": 0, "relevance_score": 20.0}],
                }).encode(),
            )

    transport = PrimaryFails()
    with Database(tmp_path / "papers.sqlite3") as database:
        database.migrate()
        database.connection.execute(
            "INSERT INTO papers(paper_id, title, abstract) VALUES (?, ?, ?)",
            ("paper-1", "Relevant molecular generation", "Graph learning methods"),
        )
        screener = released.screener(database, "fallback-crawl", transport=transport)
        assert screener.screen(("paper-1",)) == {
            "paper-1": FilterStatus.RELEVANT
        }
        row = database.connection.execute(
            "SELECT model_id, reason FROM filter_decisions WHERE paper_id = ?",
            ("paper-1",),
        ).fetchone()
        assert row["model_id"] == "bge-reranker-v2-m3-mlx-backup"
        reason = json.loads(row["reason"])
        assert reason["reranker_provenance"]["fallback_used"] is True
        run = database.connection.execute(
            "SELECT config_hash FROM pipeline_runs WHERE run_id = ?",
            (screener.run_ids[0],),
        ).fetchone()
        assert run["config_hash"] == released.effective_config_hash
    assert transport.models == [
        released.profile.reranker_model_id,
        "bge-reranker-v2-m3-mlx-backup",
    ]


def test_fallback_runtime_overrides_are_bound_into_the_effective_config(
    fallback_assembly_bundle: V3Bundle,
) -> None:
    root = fallback_assembly_bundle.root
    backup_root = root / "backup"
    common = dict(
        candidate_path=root / "candidate.json",
        evidence_path=root / "stage2-release-evidence.json",
        hidden_trust_path=fallback_assembly_bundle.trust_path,
        parity_oracle_trust_path=fallback_assembly_bundle.parity_trust_path,
        fallback_candidate_path=backup_root / "candidate.json",
        fallback_evidence_path=backup_root / "stage2-release-evidence.json",
    )
    default = validate_stage2_release_assembly(
        output_path=root / "default-release.json", **common
    )
    overridden = validate_stage2_release_assembly(
        output_path=root / "overridden-release.json",
        fallback_omlx_base_url="http://127.0.0.1:9000",
        fallback_api_key_env="BACKUP_OMLX_KEY",
        **common,
    )

    assert overridden.query_plan_config_hash != default.query_plan_config_hash
    assemble_stage2_release(
        output_path=root / "overridden-release.json",
        fallback_omlx_base_url="http://127.0.0.1:9000",
        fallback_api_key_env="BACKUP_OMLX_KEY",
        **common,
    )
    document = json.loads((root / "overridden-release.json").read_text())
    assert document["reranker_fallback"]["runtime"] == {
        "omlx_base_url": "http://127.0.0.1:9000",
        "api_key_env": "BACKUP_OMLX_KEY",
    }


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("evaluation_run_id", "different-run"),
        ("promotion_marker_hash", "f" * 64),
        ("promotion_batch_hash", "e" * 64),
        ("winner_candidate_id", "different-winner"),
        ("evaluator_id", "different-evaluator"),
        ("issued_at", "2026-08-12T01:00:00Z"),
    ),
)
def test_assembly_rejects_fallback_attestation_from_another_sealed_batch(
    fallback_assembly_bundle: V3Bundle,
    field: str,
    value: str,
) -> None:
    backup_root = fallback_assembly_bundle.root / "backup"
    attestation_path = backup_root / "attestation.json"
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    payload = attestation["payload"]
    payload[field] = value
    from paper_agent.stage2_hidden_attestation import issue_hidden_promotion_attestation

    _write_json(
        attestation_path,
        issue_hidden_promotion_attestation(
            payload,
            fallback_assembly_bundle.private_key,
        ),
    )
    evidence_path = backup_root / "stage2-release-evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["hidden_attestation"] = _ref(attestation_path)
    _write_json(evidence_path, evidence)

    with pytest.raises(Stage2ReleaseAssemblyError, match="sealed batch"):
        assemble_stage2_release(
            fallback_assembly_bundle.root / "candidate.json",
            fallback_assembly_bundle.root / "stage2-release-evidence.json",
            fallback_assembly_bundle.trust_path,
            fallback_assembly_bundle.root / "assembled-with-fallback.json",
            parity_oracle_trust_path=fallback_assembly_bundle.parity_trust_path,
            fallback_candidate_path=backup_root / "candidate.json",
            fallback_evidence_path=evidence_path,
        )


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
            parity_oracle_trust_path=assembly_bundle.parity_trust_path,
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
            parity_oracle_trust_path=assembly_bundle.parity_trust_path,
        )

    assert not output_path.exists()


@pytest.mark.parametrize(
    "kind",
    (
        "existing",
        "wrong_parent",
        "bundled_trust",
        "bundled_trust_symlink",
        "bundled_parity_trust",
        "bundled_parity_trust_symlink",
        "external_evidence",
    ),
)
def test_assembly_rejects_unsafe_paths_before_writing(
    assembly_bundle: V3Bundle,
    tmp_path: Path,
    kind: str,
) -> None:
    candidate_path = assembly_bundle.root / "candidate.json"
    evidence_path = assembly_bundle.root / "stage2-release-evidence.json"
    trust_path = assembly_bundle.trust_path
    parity_trust_path = assembly_bundle.parity_trust_path
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
    elif kind == "bundled_parity_trust":
        parity_trust_path = assembly_bundle.root / "parity-oracle-trust.json"
    elif kind == "bundled_parity_trust_symlink":
        parity_trust_path = assembly_bundle.root / "parity-trust-link.json"
        parity_trust_path.symlink_to(assembly_bundle.parity_trust_path)
    else:
        evidence_path = tmp_path / "stage2-release-evidence.json"
        evidence_path.write_text("{}", encoding="utf-8")

    with pytest.raises(Stage2ReleaseAssemblyError):
        assemble_stage2_release(
            candidate_path,
            evidence_path,
            trust_path,
            output_path,
            parity_oracle_trust_path=parity_trust_path,
        )

    if kind != "existing":
        assert not output_path.exists()


def test_assembly_recomputes_the_full_public_evidence_gates(
    assembly_bundle: V3Bundle,
) -> None:
    evidence_path = assembly_bundle.root / "stage2-release-evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    worklist_path = assembly_bundle.root / "rationale-worklist.json"
    worklist = json.loads(worklist_path.read_text(encoding="utf-8"))
    for row in worklist["rows"][:6]:
        row["evidence_supported"] = False
    _write_json(worklist_path, worklist)
    evidence["public_gates"]["rationale"]["worklist"] = _ref(worklist_path)
    records_path = assembly_bundle.root / "rationale-records.json"
    records = json.loads(records_path.read_text(encoding="utf-8"))
    for record in records["records"][:6]:
        record["evidence_supported"] = False
    records["worklist_sha256"] = evidence["public_gates"]["rationale"]["worklist"]["sha256"]
    _write_json(records_path, records)
    evidence["public_gates"]["rationale"]["records"] = _ref(records_path)
    _write_json(evidence_path, evidence)

    with pytest.raises(Stage2ReleaseAssemblyError, match="public release gates did not pass"):
        assemble_stage2_release(
            assembly_bundle.root / "candidate.json",
            evidence_path,
            assembly_bundle.trust_path,
            assembly_bundle.root / "assembled-stage2-release.json",
            parity_oracle_trust_path=assembly_bundle.parity_trust_path,
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
        parity_oracle_trust_path=assembly_bundle.parity_trust_path,
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
            parity_oracle_trust_path=assembly_bundle.parity_trust_path,
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
        parity_oracle_trust_path=assembly_bundle.parity_trust_path,
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
            parity_oracle_trust_path=assembly_bundle.parity_trust_path,
        )

    assert not output_path.exists()
