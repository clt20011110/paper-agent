from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import paper_agent.stage2_promotion_artifacts as artifacts_io
from paper_agent.stage2_evaluation import (
    CalibrationPath,
    CandidateModelArtifacts,
    GoldLabelStore,
    GoldManifest,
    GoldPair,
    GoldSplit,
    GateResult,
    PathCalibrator,
    Prediction,
    Stage2Decision,
    ThresholdArtifact,
    write_gold_manifest,
)
from paper_agent.stage2_promotion_artifacts import (
    PrivatePromotionArtifactError,
    candidate_artifacts_from_v2_bundle,
    private_gold_labels_from_document,
    promotion_submission_document,
    promotion_submission_from_document,
    run_promotion_evaluation,
    validate_promotion_candidate_bundles,
)


def _hash(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _verified_public_gates(
    *, throughput: tuple[float, float, float] = (100, 100, 100),
) -> SimpleNamespace:
    return SimpleNamespace(
        gates={
            name: SimpleNamespace(
                evidence_hash="a" * 64,
                gate=GateResult(True, ()),
            )
            for name in ("structured_replay", "rationale", "parity", "benchmark", "soak")
        },
        throughput_runs=throughput,
    )


def _gold() -> tuple[GoldManifest, dict[str, object]]:
    pairs: list[GoldPair] = []
    for split, size in ((GoldSplit.DEV, 300), (GoldSplit.HIDDEN_HARD, 150), (GoldSplit.HIDDEN_REAL, 150)):
        for index in range(size):
            pairs.append(GoldPair(
                f"paper-{split.value}-{index}", f"topic-{index % 6}", "zh" if index % 2 else "en",
                "synthetic-corpus",
                0.2 if split is GoldSplit.HIDDEN_REAL else None,
                f"family-{split.value}-{index}", "a" * 64, split,
                abstract_incomplete=split is not GoldSplit.HIDDEN_REAL and index < size // 10,
                sampled_from_natural_distribution=split is GoldSplit.HIDDEN_REAL,
                cross_language_match=index % 20 == 0,
            ))
    manifest = GoldManifest(1, "a" * 64, tuple(pairs), ("en", "zh"))
    labels = {pair.pair_id: 2 if index % 3 else 0 for index, pair in enumerate(pairs)}
    dev = [pair for pair in pairs if pair.split is GoldSplit.DEV]
    hard = [pair for pair in pairs if pair.split is GoldSplit.HIDDEN_HARD]
    hard_negatives = {pair.pair_id for pair in dev[:60]} | {pair.pair_id for pair in hard[:30]}
    for pair_id in hard_negatives:
        labels[pair_id] = 0
    hard_positives = {dev[-1].pair_id, hard[-1].pair_id}
    for pair_id in hard_positives:
        labels[pair_id] = 3
    document = {
        "schema_version": "1",
        "gold_manifest_hash": manifest.hash(),
        "annotation_artifact_hash": "b" * 64,
        "labels": [{"pair_id": pair_id, "label": label} for pair_id, label in labels.items()],
        "hard_negative_pair_ids": sorted(hard_negatives),
        "hard_positive_pair_ids": sorted(hard_positives),
    }
    return manifest, document


def _candidate(manifest: GoldManifest, labels: GoldLabelStore, candidate_id: str = "candidate") -> CandidateModelArtifacts:
    dev_ids = tuple(sorted(pair.pair_id for pair in manifest.pairs if pair.split is GoldSplit.DEV))
    dev_labels = GoldLabelStore({pair_id: labels.labels[pair_id] for pair_id in dev_ids}, labels.annotation_artifact_hash)
    pair_ids_hash = _hash(dev_ids)
    calibrators = {
        path: PathCalibrator(
            1, path, 1.0, 0.0, manifest.dev_hash(), manifest.hash(),
            sha256(f"{candidate_id}:{path.value}:lock".encode()).hexdigest(), dev_labels.hash(),
            pair_ids_hash, len(dev_ids), dev_ids,
        )
        for path in CalibrationPath
    }
    thresholds = {
        path: ThresholdArtifact(
            1, path, 0.25, 0.75, calibrator.hash(), calibrator.model_lock_hash,
            manifest.dev_hash(), dev_labels.hash(), sha256(f"{candidate_id}:config".encode()).hexdigest(),
        )
        for path, calibrator in calibrators.items()
    }
    return CandidateModelArtifacts(candidate_id, calibrators, thresholds)


def _submission(manifest: GoldManifest, labels: GoldLabelStore, candidate: CandidateModelArtifacts, *, reject_all: bool = False):
    hidden = [pair for pair in manifest.pairs if pair.split is not GoldSplit.DEV]
    runs = []
    for run in range(3):
        predictions = []
        for pair in hidden:
            path = CalibrationPath.RERANKER
            calibrator = candidate.calibrators[path]
            threshold = candidate.thresholds[path]
            relevant = not reject_all and labels.labels[pair.pair_id] >= 2
            raw_score = 5.0 if relevant else -5.0
            predictions.append(Prediction(
                pair.pair_id, candidate.candidate_id,
                Stage2Decision.RELEVANT if relevant else Stage2Decision.IRRELEVANT,
                raw_score, calibrator.predict(raw_score), path, calibrator.hash(), threshold.hash(),
                calibrator.model_lock_hash, manifest.hash(), threshold.stage2_config_hash,
                sha256(f"{candidate.candidate_id}:{run}:{pair.pair_id}".encode()).hexdigest(),
            ))
        runs.append(tuple(predictions))
    from paper_agent.stage2_evaluation import PromotionSubmission

    return PromotionSubmission(candidate.candidate_id, tuple(runs))


def test_private_label_and_submission_parsers_are_strict_and_reconstruct_enums(tmp_path: Path) -> None:
    manifest, labels_document = _gold()
    labels = private_gold_labels_from_document(labels_document, manifest=manifest)
    candidate = _candidate(manifest, labels)
    submission_document = promotion_submission_document(_submission(manifest, labels, candidate))
    parsed = promotion_submission_from_document(submission_document, manifest=manifest)

    assert parsed.runs[0][0].decision is Stage2Decision.RELEVANT or parsed.runs[0][0].decision is Stage2Decision.IRRELEVANT
    assert parsed.runs[0][0].path is CalibrationPath.RERANKER
    assert parsed.candidate_id == "candidate" and len(parsed.runs) == 3

    invalid = json.loads(json.dumps(submission_document))
    invalid["runs"][0][0]["untrusted_artifacts"] = {"calibrators": "forged"}
    with pytest.raises(PrivatePromotionArtifactError, match="Additional properties"):
        promotion_submission_from_document(invalid, manifest=manifest)

    labels_path = tmp_path / "labels.json"
    labels_path.write_text('{"schema_version":"1","schema_version":"1"}', encoding="utf-8")
    with pytest.raises(PrivatePromotionArtifactError, match="cannot read private gold labels"):
        artifacts_io.load_private_gold_labels(labels_path, manifest=manifest)


def test_candidate_artifacts_are_derived_from_existing_v2_bundle(tmp_path: Path) -> None:
    from test_stage2_search import _release_bundle

    release_path, _ = _release_bundle(tmp_path)
    release = json.loads(release_path.read_text(encoding="utf-8"))
    release.pop("release_gate")
    release["schema_version"] = "2"
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(json.dumps(release), encoding="utf-8")

    derived = candidate_artifacts_from_v2_bundle(candidate_path)
    assert derived.candidate_id == "local-winner"
    assert set(derived.calibrators) == set(CalibrationPath)
    assert set(derived.thresholds) == set(CalibrationPath)


def test_candidate_preflight_requires_every_calibrator_to_bind_gold_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, labels_document = _gold()
    labels = private_gold_labels_from_document(labels_document, manifest=manifest)
    candidate = _candidate(manifest, labels)
    monkeypatch.setattr(
        artifacts_io, "candidate_artifacts_from_v2_bundle", lambda _path: candidate
    )

    validate_promotion_candidate_bundles(
        {candidate.candidate_id: tmp_path / "candidate.json"},
        expected_manifest_hash=manifest.hash(),
    )

    calibrators = dict(candidate.calibrators)
    calibrators[CalibrationPath.QWEN] = replace(
        calibrators[CalibrationPath.QWEN], gold_manifest_hash="f" * 64
    )
    mismatched = CandidateModelArtifacts(
        candidate.candidate_id, calibrators, candidate.thresholds
    )
    monkeypatch.setattr(
        artifacts_io, "candidate_artifacts_from_v2_bundle", lambda _path: mismatched
    )
    with pytest.raises(PrivatePromotionArtifactError, match="supplied gold manifest"):
        validate_promotion_candidate_bundles(
            {candidate.candidate_id: tmp_path / "candidate.json"},
            expected_manifest_hash=manifest.hash(),
        )


def test_public_promotion_verification_passes_the_candidate_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import paper_agent.stage2_public_gates as public_gates
    import paper_agent.stage2_release_evidence as evidence_io
    import paper_agent.stage2_search as stage2_search

    profile = object()
    candidate = SimpleNamespace(profile=profile)
    evidence = object()
    oracle_trust = object()
    observed: list[object] = []
    monkeypatch.setattr(
        stage2_search, "load_stage2_benchmark_candidate", lambda _path: candidate,
    )
    monkeypatch.setattr(
        stage2_search, "_validate_evidence_bindings", lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        stage2_search,
        "_load_deployment_parity_oracle_trust",
        lambda _path, *, bundle_root: oracle_trust,
    )
    monkeypatch.setattr(
        evidence_io, "load_stage2_release_evidence_index", lambda _path: evidence,
    )
    monkeypatch.setattr(
        public_gates,
        "verify_public_stage2_gates",
        lambda value, *, profile, oracle_trust: (
            observed.extend((value, profile, oracle_trust)) or "verified"
        ),
    )

    result = artifacts_io.validate_promotion_public_evidence(
        {"candidate": tmp_path / "candidate.json"},
        {"candidate": tmp_path / "evidence.json"},
        "a" * 64,
        parity_oracle_trust_path=tmp_path / "oracle-trust.json",
    )

    assert result["candidate"] == "verified"
    assert observed == [evidence, profile, oracle_trust]


def test_orchestration_returns_only_safe_hashes_and_consumes_marker_on_gate_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, labels_document = _gold()
    labels = private_gold_labels_from_document(labels_document, manifest=manifest)
    candidate = _candidate(manifest, labels)
    manifest_path = tmp_path / "gold.json"
    labels_path = tmp_path / "private-labels.json"
    submission_path = tmp_path / "private-submission.json"
    write_gold_manifest(manifest_path, manifest)
    labels_path.write_text(json.dumps(labels_document), encoding="utf-8")
    submission_path.write_text(
        json.dumps(promotion_submission_document(_submission(manifest, labels, candidate, reject_all=True))),
        encoding="utf-8",
    )
    monkeypatch.setattr(artifacts_io, "candidate_artifacts_from_v2_bundle", lambda _path: candidate)
    monkeypatch.setattr(
        artifacts_io,
        "_public_release_evidence",
        lambda *_args, **_kwargs: {"candidate": _verified_public_gates()},
    )

    result = run_promotion_evaluation(
        manifest_path=manifest_path,
        private_labels_path=labels_path,
        submission_paths={"candidate": submission_path},
        candidate_bundle_paths={"candidate": tmp_path / "candidate-v2.json"},
        public_evidence_paths={"candidate": tmp_path / "public-evidence.json"},
        evaluator_id="synthetic-evaluator",
        state_root=tmp_path / "state",
        incumbent_candidate_id="candidate",
        evaluation_run_id="synthetic-run",
        parity_oracle_trust_path=tmp_path / "oracle-trust.json",
        bootstrap_iterations=100,
    )

    unsigned = result.candidates["candidate"].document()
    assert unsigned["result_summary"]["passed"] is False
    assert "labels" not in json.dumps(unsigned) and "pair-" not in json.dumps(unsigned)
    payload = result.candidates["candidate"].attestation_payload(
        evaluator_key_id="synthetic-key",
        trust_manifest_hash="c" * 64,
        issued_at="2026-08-11T00:00:00Z",
    )
    assert payload["evaluator_id"] == "synthetic-evaluator"
    assert payload["prediction_submission_hash"] == result.candidates["candidate"].prediction_submission_hash
    assert (tmp_path / "state" / f"{manifest.hash()}.promotion.json").is_file()


def test_orchestration_derives_the_paired_hidden_winner_not_an_operator_choice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, labels_document = _gold()
    labels = private_gold_labels_from_document(labels_document, manifest=manifest)
    incumbent = _candidate(manifest, labels, "incumbent")
    challenger = _candidate(manifest, labels, "challenger")
    manifest_path = tmp_path / "gold.json"
    labels_path = tmp_path / "private-labels.json"
    incumbent_submission = tmp_path / "incumbent-submission.json"
    challenger_submission = tmp_path / "challenger-submission.json"
    write_gold_manifest(manifest_path, manifest)
    labels_path.write_text(json.dumps(labels_document), encoding="utf-8")
    incumbent_submission.write_text(
        json.dumps(promotion_submission_document(
            _submission(manifest, labels, incumbent, reject_all=True)
        )),
        encoding="utf-8",
    )
    challenger_submission.write_text(
        json.dumps(promotion_submission_document(_submission(manifest, labels, challenger))),
        encoding="utf-8",
    )
    candidate_paths = {
        "incumbent": tmp_path / "incumbent-v2.json",
        "challenger": tmp_path / "challenger-v2.json",
    }
    artifacts = {"incumbent": incumbent, "challenger": challenger}
    monkeypatch.setattr(
        artifacts_io,
        "candidate_artifacts_from_v2_bundle",
        lambda path: artifacts[path.stem.removesuffix("-v2")],
    )
    monkeypatch.setattr(
        artifacts_io,
        "_public_release_evidence",
        lambda *_args, **_kwargs: {
            "incumbent": _verified_public_gates(throughput=(100, 100, 100)),
            "challenger": _verified_public_gates(throughput=(120, 120, 120)),
        },
    )
    monkeypatch.setattr(
        artifacts_io, "promotion_gate", lambda _result: GateResult(True, ())
    )

    result = run_promotion_evaluation(
        manifest_path=manifest_path,
        private_labels_path=labels_path,
        submission_paths={
            "incumbent": incumbent_submission,
            "challenger": challenger_submission,
        },
        candidate_bundle_paths=candidate_paths,
        public_evidence_paths={
            "incumbent": tmp_path / "incumbent-public-evidence.json",
            "challenger": tmp_path / "challenger-public-evidence.json",
        },
        evaluator_id="synthetic-evaluator",
        state_root=tmp_path / "state",
        incumbent_candidate_id="incumbent",
        evaluation_run_id="synthetic-run",
        parity_oracle_trust_path=tmp_path / "oracle-trust.json",
        bootstrap_iterations=100,
    )

    assert result.winner_candidate_id == "challenger"
    winner = result.candidates[result.winner_candidate_id]
    assert winner.winner_candidate_id == winner.candidate_id
    assert winner.passed
