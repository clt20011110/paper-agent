from __future__ import annotations

from base64 import b64encode
from copy import deepcopy
from dataclasses import dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
import shutil

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import pytest

import paper_agent.stage2_search as stage2_search
from paper_agent.approval import approved_content_hash
from paper_agent.canonical import content_hash
from paper_agent.domain import FilterStatus
from paper_agent.query_plan import approve_query_plan
from paper_agent.stage2_evaluation import CalibrationPath, PathCalibrator
from paper_agent.stage2_hidden_attestation import (
    HIDDEN_PROMOTION_GATE_POLICY_HASH,
    issue_hidden_promotion_attestation,
)
from paper_agent.stage2_search import (
    Stage2ReleaseError,
    load_stage2_benchmark_candidate,
    load_stage2_release,
)
from paper_agent.storage import Database

import test_stage2_release_evidence as public_evidence
from test_stage2_search import LocalOmlxFixture, _release_bundle


@dataclass(frozen=True)
class V3Bundle:
    root: Path
    release_path: Path
    plan: dict
    trust_path: Path
    wrong_trust_path: Path
    private_key: Ed25519PrivateKey


def _write_json(path: Path, document: object) -> None:
    path.write_text(json.dumps(document, sort_keys=True) + "\n", encoding="utf-8")


def _ref(path: Path) -> dict[str, str]:
    return {"path": path.name, "sha256": sha256(path.read_bytes()).hexdigest()}


def _reapprove_release(bundle: V3Bundle) -> V3Bundle:
    release = json.loads(bundle.release_path.read_text(encoding="utf-8"))
    candidate_document = deepcopy(release)
    candidate_document.pop("release_gate")
    candidate_document["schema_version"] = "2"
    candidate_path = bundle.root / "candidate-for-plan.json"
    _write_json(candidate_path, candidate_document)
    candidate = load_stage2_benchmark_candidate(candidate_path)
    released_profile = replace(
        candidate.profile,
        release_gate_hash=content_hash(release["release_gate"]),
    )
    draft = deepcopy(bundle.plan)
    draft["status"] = "draft"
    draft["approval"] = None
    draft["filter"]["config_hash"] = released_profile.config_hash
    draft["filter"]["thresholds_hash"] = released_profile.threshold_hash
    draft["plan_hash"] = approved_content_hash(draft)
    plan = approve_query_plan(
        draft,
        draft["plan_hash"],
        approved_by="test-owner",
        approved_at="2026-08-11T00:00:00Z",
    )
    return replace(bundle, plan=plan)


def _refresh_outer_evidence_ref(bundle: V3Bundle) -> V3Bundle:
    release = json.loads(bundle.release_path.read_text(encoding="utf-8"))
    evidence_path = bundle.root / release["release_gate"]["evidence"]["path"]
    release["release_gate"]["evidence"] = _ref(evidence_path)
    _write_json(bundle.release_path, release)
    return _reapprove_release(bundle)


def _payload(
    *,
    trust_hash: str,
    candidate_id: str,
    evaluation_manifest_hash: str,
    stage2_config_hash: str,
    model_lock_hashes: dict[str, str],
    calibrator_hashes: dict[str, str],
    threshold_hashes: dict[str, str],
    gold: object,
) -> dict:
    return {
        "schema_version": "1",
        "attestation_type": "stage2-hidden-promotion",
        "evaluator_key_id": "test-hidden-evaluator",
        "evaluator_id": "test-evaluation-team",
        "trust_manifest_hash": trust_hash,
        "issued_at": "2026-08-11T00:00:00Z",
        "candidate_id": candidate_id,
        "evaluation_manifest_hash": evaluation_manifest_hash,
        "evaluation_run_id": "test-promotion-1",
        "stage2_config_hash": stage2_config_hash,
        "model_lock_hashes": model_lock_hashes,
        "calibrator_hashes": calibrator_hashes,
        "threshold_hashes": threshold_hashes,
        "hidden_pair_universe_hashes": {
            split.value: public_evidence.pair_universe_hash([
                pair.pair_id for pair in gold.pairs if pair.split is split
            ])
            for split in (
                public_evidence.GoldSplit.HIDDEN_HARD,
                public_evidence.GoldSplit.HIDDEN_REAL,
            )
        },
        "hidden_split_pair_counts": {
            "hidden_hard": 150,
            "hidden_real": 150,
        },
        "prediction_submission_hash": "1" * 64,
        "promotion_marker_hash": "2" * 64,
        "consumed_hidden_splits": ["hidden_hard", "hidden_real"],
        "gate_policy_hash": HIDDEN_PROMOTION_GATE_POLICY_HASH,
        "result_summary": {
            "passed": True,
            "failures": [],
            "gate_versions": {"promotion": "1", "determinism": "1"},
        },
    }


def _build_v3_bundle(root: Path) -> V3Bundle:
    release_path, plan = _release_bundle(root)
    release = json.loads(release_path.read_text(encoding="utf-8"))
    evidence_path, evidence = public_evidence._index(root)
    gold = public_evidence._gold_manifest()

    # The compact runtime fixture starts with a synthetic gold hash.  Rebind
    # its frozen calibration artefacts to the real, public gold manifest.
    for name in (CalibrationPath.RERANKER.value, CalibrationPath.QWEN.value):
        calibrator_path = root / release["calibration"][name]["calibrator"]["path"]
        calibrator = json.loads(calibrator_path.read_text(encoding="utf-8"))
        calibrator["gold_manifest_hash"] = gold.hash()
        _write_json(calibrator_path, calibrator)
        release["calibration"][name]["calibrator"] = _ref(calibrator_path)

        threshold_path = root / release["calibration"][name]["threshold"]["path"]
        threshold = json.loads(threshold_path.read_text(encoding="utf-8"))
        threshold["calibrator_hash"] = PathCalibrator(**calibrator).hash()
        _write_json(threshold_path, threshold)
        release["calibration"][name]["threshold"] = _ref(threshold_path)

    candidate_document = deepcopy(release)
    candidate_document.pop("release_gate")
    candidate_document["schema_version"] = "2"
    candidate_path = root / "candidate.json"
    _write_json(candidate_path, candidate_document)
    candidate = load_stage2_benchmark_candidate(candidate_path)
    model_lock_hashes = {
        "reranker": candidate.profile.reranker_lock_hash,
        "qwen": candidate.profile.adjudicator_lock_hash,
    }
    calibrator_hashes = {
        "reranker": candidate.profile.reranker_calibration.calibrator.hash(),
        "qwen": candidate.profile.adjudicator_calibration.calibrator.hash(),
    }
    threshold_hashes = {
        "reranker": candidate.profile.reranker_calibration.threshold.hash(),
        "qwen": candidate.profile.adjudicator_calibration.threshold.hash(),
    }
    evidence.update({
        "candidate_id": release["profile"],
        "evaluation_manifest_hash": gold.hash(),
        "stage2_config_hash": candidate.profile.base_runtime_config_hash,
        "model_lock_hashes": model_lock_hashes,
        "calibrator_hashes": calibrator_hashes,
        "threshold_hashes": threshold_hashes,
    })
    public_evidence._install_public_gate_evidence(root, evidence_path, evidence)

    private_key = Ed25519PrivateKey.from_private_bytes(bytes(range(1, 33)))
    public_key = private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    trust = {
        "schema_version": "1",
        "trust_manifest_type": "stage2-hidden-evaluator",
        "trust_manifest_id": "test-hidden-evaluator-v1",
        "keys": [{
            "key_id": "test-hidden-evaluator",
            "algorithm": "Ed25519",
            "purpose": "stage2-hidden-promotion",
            "public_key_b64": b64encode(public_key).decode("ascii"),
            "status": "active",
        }],
    }
    trust_path = root / "hidden-evaluator-trust.json"
    _write_json(trust_path, trust)
    wrong_trust = deepcopy(trust)
    wrong_trust["keys"][0]["public_key_b64"] = b64encode(
        Ed25519PrivateKey.from_private_bytes(bytes(range(2, 34))).public_key().public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        )
    ).decode("ascii")
    wrong_trust_path = root / "wrong-hidden-evaluator-trust.json"
    _write_json(wrong_trust_path, wrong_trust)

    attestation = issue_hidden_promotion_attestation(
        _payload(
            trust_hash=content_hash(trust),
            candidate_id=release["profile"],
            evaluation_manifest_hash=gold.hash(),
            stage2_config_hash=candidate.profile.base_runtime_config_hash,
            model_lock_hashes=model_lock_hashes,
            calibrator_hashes=calibrator_hashes,
            threshold_hashes=threshold_hashes,
            gold=gold,
        ),
        private_key,
    )
    attestation_path = root / "attestation.json"
    _write_json(attestation_path, attestation)
    evidence["hidden_attestation"] = _ref(attestation_path)
    _write_json(evidence_path, evidence)

    release["schema_version"] = "3"
    release["release_gate"] = {
        "candidate_id": release["profile"],
        "evaluation_manifest_hash": gold.hash(),
        "evidence": _ref(evidence_path),
    }
    _write_json(release_path, release)
    bundle = V3Bundle(root, release_path, plan, trust_path, wrong_trust_path, private_key)
    return _reapprove_release(bundle)


@pytest.fixture(scope="module")
def v3_template(tmp_path_factory: pytest.TempPathFactory) -> V3Bundle:
    return _build_v3_bundle(tmp_path_factory.mktemp("stage2-v3-release"))


@pytest.fixture
def v3_bundle(v3_template: V3Bundle, tmp_path: Path) -> V3Bundle:
    root = tmp_path / "bundle"
    shutil.copytree(v3_template.root, root)
    trust_path = tmp_path / "deployment-hidden-evaluator-trust.json"
    wrong_trust_path = tmp_path / "deployment-wrong-hidden-evaluator-trust.json"
    shutil.copy2(v3_template.trust_path, trust_path)
    shutil.copy2(v3_template.wrong_trust_path, wrong_trust_path)
    return replace(
        v3_template,
        root=root,
        release_path=root / v3_template.release_path.name,
        trust_path=trust_path,
        wrong_trust_path=wrong_trust_path,
        plan=deepcopy(v3_template.plan),
    )


def test_benchmark_candidate_remains_schema2_without_a_release_gate(tmp_path: Path) -> None:
    release_path, _ = _release_bundle(tmp_path)
    candidate = json.loads(release_path.read_text(encoding="utf-8"))
    candidate.pop("release_gate")
    candidate["schema_version"] = "2"
    candidate_path = tmp_path / "benchmark-candidate.json"
    _write_json(candidate_path, candidate)

    loaded = load_stage2_benchmark_candidate(candidate_path)

    assert loaded.profile.release_gate_hash is None


def test_production_rejects_schema2_generic_passed_shell(tmp_path: Path) -> None:
    release_path, plan = _release_bundle(tmp_path)
    release = json.loads(release_path.read_text(encoding="utf-8"))
    release["schema_version"] = "2"
    release["release_gate"] = {
        "candidate_id": release["profile"],
        "evaluation_manifest_hash": "a" * 64,
        "artifacts": {},
        "passed": True,
        "failures": [],
        "throughput_runs": [100.0, 100.0, 100.0],
    }
    _write_json(release_path, release)

    with pytest.raises(Stage2ReleaseError, match="schema_version 3"):
        load_stage2_release(release_path, plan)


def test_production_v3_rejects_claimed_gate_outcomes(v3_bundle: V3Bundle) -> None:
    release = json.loads(v3_bundle.release_path.read_text(encoding="utf-8"))
    release["release_gate"]["passed"] = True
    release["release_gate"]["failures"] = []
    _write_json(v3_bundle.release_path, release)
    v3_bundle = _reapprove_release(v3_bundle)

    with pytest.raises(Stage2ReleaseError, match="release gate fields are not exact"):
        load_stage2_release(
            v3_bundle.release_path,
            v3_bundle.plan,
            hidden_trust_path=v3_bundle.trust_path,
        )


def test_production_v3_requires_operator_controlled_trust(v3_bundle: V3Bundle) -> None:
    with pytest.raises(Stage2ReleaseError, match="deployment-controlled hidden evaluator trust"):
        load_stage2_release(v3_bundle.release_path, v3_bundle.plan, environment={})

    bundled_trust = v3_bundle.root / "hidden-evaluator-trust.json"
    with pytest.raises(Stage2ReleaseError, match="outside the release bundle"):
        load_stage2_release(
            v3_bundle.release_path,
            v3_bundle.plan,
            hidden_trust_path=bundled_trust,
        )

    bundled_symlink = v3_bundle.root / "deployment-trust-link.json"
    bundled_symlink.symlink_to(v3_bundle.trust_path)
    with pytest.raises(Stage2ReleaseError, match="outside the release bundle"):
        load_stage2_release(
            v3_bundle.release_path,
            v3_bundle.plan,
            hidden_trust_path=bundled_symlink,
        )

    bundle_alias = v3_bundle.root.parent / "bundle-alias"
    bundle_alias.symlink_to(v3_bundle.root, target_is_directory=True)
    with pytest.raises(Stage2ReleaseError, match="outside the release bundle"):
        load_stage2_release(
            v3_bundle.release_path,
            v3_bundle.plan,
            hidden_trust_path=bundle_alias / bundled_symlink.name,
        )


def test_v3_uses_one_loaded_trust_snapshot(
    v3_bundle: V3Bundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = stage2_search.load_hidden_evaluator_trust
    calls = 0

    def load_then_replace(path: Path):
        nonlocal calls
        calls += 1
        trust = original(path)
        shutil.copy2(v3_bundle.wrong_trust_path, path)
        return trust

    monkeypatch.setattr(stage2_search, "load_hidden_evaluator_trust", load_then_replace)

    load_stage2_release(
        v3_bundle.release_path,
        v3_bundle.plan,
        hidden_trust_path=v3_bundle.trust_path,
    )

    assert calls == 1


def test_v3_uses_outer_hash_verified_evidence_index_snapshot(
    v3_bundle: V3Bundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = stage2_search.load_stage2_release_evidence_index_bytes
    calls = 0

    def load_then_replace(path: Path, payload: bytes):
        nonlocal calls
        calls += 1
        index = original(path, payload)
        path.write_text("{}\n", encoding="utf-8")
        return index

    monkeypatch.setattr(
        stage2_search,
        "load_stage2_release_evidence_index_bytes",
        load_then_replace,
    )

    load_stage2_release(
        v3_bundle.release_path,
        v3_bundle.plan,
        hidden_trust_path=v3_bundle.trust_path,
    )

    assert calls == 1


def test_v3_release_verifies_raw_evidence_signature_then_screens(v3_bundle: V3Bundle) -> None:
    released = load_stage2_release(
        v3_bundle.release_path,
        v3_bundle.plan,
        hidden_trust_path=v3_bundle.trust_path,
    )
    transport = LocalOmlxFixture()

    with Database(v3_bundle.root / "papers.sqlite3") as database:
        database.migrate()
        database.connection.execute(
            "INSERT INTO papers(paper_id, title, abstract) VALUES (?, ?, ?)",
            ("relevant", "Relevant paper", "graph learning"),
        )
        decisions = released.screener(database, "v3-smoke", transport=transport).screen(("relevant",))

    assert decisions == {"relevant": FilterStatus.RELEVANT}
    assert transport.paths == ["/v1/rerank"]


def test_v3_rejects_evidence_ref_drift(v3_bundle: V3Bundle) -> None:
    evidence_path = v3_bundle.root / "stage2-release-evidence.json"
    evidence_path.write_text(evidence_path.read_text(encoding="utf-8") + " ", encoding="utf-8")

    with pytest.raises(Stage2ReleaseError, match="artifact drifted"):
        load_stage2_release(
            v3_bundle.release_path,
            v3_bundle.plan,
            hidden_trust_path=v3_bundle.trust_path,
        )


def test_v3_rejects_hash_valid_recomputed_public_gate_failure(v3_bundle: V3Bundle) -> None:
    evidence_path = v3_bundle.root / "stage2-release-evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    records_path = v3_bundle.root / "rationale-records.json"
    records = json.loads(records_path.read_text(encoding="utf-8"))
    for record in records["records"][:6]:
        record["evidence_supported"] = False
    _write_json(records_path, records)
    evidence["public_gates"]["rationale"]["records"] = _ref(records_path)
    _write_json(evidence_path, evidence)
    v3_bundle = _refresh_outer_evidence_ref(v3_bundle)

    with pytest.raises(Stage2ReleaseError, match="public release gates did not pass"):
        load_stage2_release(
            v3_bundle.release_path,
            v3_bundle.plan,
            hidden_trust_path=v3_bundle.trust_path,
        )


@pytest.mark.parametrize("mode", ("signature", "wrong_trust", "policy", "result_binding"))
def test_v3_rejects_untrusted_or_invalid_hidden_promotion(
    v3_bundle: V3Bundle,
    mode: str,
) -> None:
    if mode == "wrong_trust":
        trust_path = v3_bundle.wrong_trust_path
    else:
        trust_path = v3_bundle.trust_path
        evidence_path = v3_bundle.root / "stage2-release-evidence.json"
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        attestation_path = v3_bundle.root / "attestation.json"
        attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
        if mode == "signature":
            signature = attestation["signature_b64"]
            attestation["signature_b64"] = ("A" if signature[0] != "A" else "B") + signature[1:]
        else:
            if mode == "policy":
                attestation["payload"]["gate_policy_hash"] = "f" * 64
            else:
                attestation["payload"]["candidate_id"] = "different-candidate"
            attestation = issue_hidden_promotion_attestation(
                attestation["payload"], v3_bundle.private_key
            )
        _write_json(attestation_path, attestation)
        evidence["hidden_attestation"] = _ref(attestation_path)
        _write_json(evidence_path, evidence)
        v3_bundle = _refresh_outer_evidence_ref(v3_bundle)

    with pytest.raises(Stage2ReleaseError, match="release evidence verification failed"):
        load_stage2_release(
            v3_bundle.release_path,
            v3_bundle.plan,
            hidden_trust_path=trust_path,
        )
