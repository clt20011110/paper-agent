from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

import paper_agent.stage2_candidate as stage2_candidate
from paper_agent.stage2_backends import load_model_lock
from paper_agent.stage2_candidate import build_stage2_candidate_bundle
from paper_agent.stage2_dev_calibration import FrozenDevRawScoreArtifact
from paper_agent.stage2_evaluation import CalibrationPath, GoldLabelStore, GoldSplit
from paper_agent.stage2_search import load_stage2_benchmark_candidate, stage2_base_profile

from test_stage2_dev_calibration import _inputs


ROOT = Path(__file__).parents[1]
RERANKER_LOCK = ROOT / "configs/stage2/models/bge-reranker-v2-m3-fp32.lock.json"
ADJUDICATOR_LOCK = ROOT / "configs/stage2/models/qwen3.5-9b-8bit.lock.json"


def _runtime() -> dict[str, object]:
    return {
        "query": "molecular generation",
        "query_version": "stage2-gold-query-v1",
        "screening_scope_hash": "0" * 64,
        "evaluation_topic_queries": [
            {
                "topic": f"topic-{topic}",
                "language": "zh" if topic % 2 == 0 else "en",
                "query": (
                    f"query for topic-{topic} in "
                    f"{'zh' if topic % 2 == 0 else 'en'}"
                ),
            }
            for topic in range(6)
        ],
        "include_document_types": [],
        "exclude_document_types": ["editorial", "retraction"],
        "token_bucket_width": 128,
        "document_batch_size": 32,
        "max_in_flight": 2,
        "adjudicator_concurrency": 4,
        "adjudicator_seed": 42,
        "max_context_window": 16_384,
        "max_tokens": 256,
        "omlx_base_url": "http://127.0.0.1:8000",
        "api_key_env": None,
        "prompt_version": "stage2-adjudication-v1",
        "schema_version": "filter-decision.schema.json",
    }


def _labels(manifest) -> GoldLabelStore:
    labels = {
        pair.pair_id: 0 if index % 3 == 0 else 3
        for index, pair in enumerate(manifest.pairs)
    }
    hard_negative_ids: set[str] = set()
    hard_positive_ids: set[str] = set()
    for split in (GoldSplit.DEV, GoldSplit.HIDDEN_HARD):
        split_pairs = [pair for pair in manifest.pairs if pair.split is split]
        hard_negative_ids.update(
            pair.pair_id for pair in split_pairs if labels[pair.pair_id] == 0
        )
        hard_positive_ids.add(
            next(pair.pair_id for pair in split_pairs if labels[pair.pair_id] == 3)
        )
    return GoldLabelStore(
        labels,
        "c" * 64,
        frozenset(hard_negative_ids),
        frozenset(hard_positive_ids),
    )


def _raw_scores(manifest, snapshot, labels, runtime) -> FrozenDevRawScoreArtifact:
    reranker_bytes = RERANKER_LOCK.read_bytes()
    adjudicator_bytes = ADJUDICATOR_LOCK.read_bytes()
    from hashlib import sha256

    lock_hashes = {
        CalibrationPath.RERANKER: sha256(reranker_bytes).hexdigest(),
        CalibrationPath.QWEN: sha256(adjudicator_bytes).hexdigest(),
    }
    profile = stage2_base_profile(
        runtime,
        load_model_lock(RERANKER_LOCK),
        load_model_lock(ADJUDICATOR_LOCK),
        reranker_lock_hash=lock_hashes[CalibrationPath.RERANKER],
        adjudicator_lock_hash=lock_hashes[CalibrationPath.QWEN],
    )
    dev = [pair for pair in manifest.pairs if pair.split is GoldSplit.DEV]
    return FrozenDevRawScoreArtifact(
        1,
        {
            CalibrationPath.RERANKER: {
                pair.pair_id: 4.0 if labels.labels[pair.pair_id] >= 2 else -4.0
                for pair in dev
            },
            CalibrationPath.QWEN: {
                pair.pair_id: 3.0 if labels.labels[pair.pair_id] >= 2 else -3.0
                for pair in dev
            },
        },
        lock_hashes,
        manifest.hash(),
        manifest.dev_hash(),
        snapshot.hash(),
        snapshot.corpus_hash,
        profile.base_runtime_config_hash,
        {
            (pair.topic, pair.language): f"query for {pair.topic} in {pair.language}"
            for pair in dev
        },
    )


def test_build_candidate_publishes_last_marker_and_loads_as_calibrated(tmp_path: Path) -> None:
    manifest, snapshot = _inputs()
    labels = _labels(manifest)
    runtime = _runtime()
    raw_scores = _raw_scores(manifest, snapshot, labels, runtime)
    output = tmp_path / "candidate"

    result = build_stage2_candidate_bundle(
        manifest=manifest,
        private_labels=labels,
        raw_scores=raw_scores,
        runtime=runtime,
        reranker_lock_path=RERANKER_LOCK,
        adjudicator_lock_path=ADJUDICATOR_LOCK,
        candidate_id="crossref-dev-v1",
        output_dir=output,
    )

    assert result.candidate_path == output / "stage2-candidate-v2.json"
    assert result.release.profile.production_calibrated
    assert load_stage2_benchmark_candidate(result.candidate_path).release_hash == result.release.release_hash
    assert {path.name for path in output.iterdir()} == {
        "reranker.lock.json",
        "adjudicator.lock.json",
        "reranker-calibrator.json",
        "reranker-threshold.json",
        "qwen-calibrator.json",
        "qwen-threshold.json",
        "stage2-candidate-v2.json",
    }
    assert (output / "reranker.lock.json").read_bytes() == RERANKER_LOCK.read_bytes()
    assert (output / "adjudicator.lock.json").read_bytes() == ADJUDICATOR_LOCK.read_bytes()
    candidate_document = json.loads(result.candidate_path.read_text(encoding="utf-8"))
    assert set(candidate_document) == {
        "schema_version", "profile", "reranker_lock", "adjudicator_lock",
        "calibration", "runtime",
    }
    serialized = "\n".join(path.read_text(encoding="utf-8") for path in output.iterdir())
    assert '"gold_label"' not in serialized
    assert '"labels"' not in serialized
    assert all(values["positive_retention"] == 1.0 for values in result.selections.values())

    with pytest.raises(FileExistsError, match="already exists"):
        build_stage2_candidate_bundle(
            manifest=manifest,
            private_labels=labels,
            raw_scores=raw_scores,
            runtime=runtime,
            reranker_lock_path=RERANKER_LOCK,
            adjudicator_lock_path=ADJUDICATOR_LOCK,
            candidate_id="crossref-dev-v1",
            output_dir=output,
        )


def test_build_candidate_rejects_runtime_and_snapshot_corpus_drift(tmp_path: Path) -> None:
    manifest, snapshot = _inputs()
    labels = _labels(manifest)
    runtime = _runtime()
    raw_scores = _raw_scores(manifest, snapshot, labels, runtime)

    with pytest.raises(ValueError, match="manifest or Stage 2 runtime"):
        build_stage2_candidate_bundle(
            manifest=manifest,
            private_labels=labels,
            raw_scores=replace(raw_scores, private_snapshot_corpus_hash="f" * 64),
            runtime=runtime,
            reranker_lock_path=RERANKER_LOCK,
            adjudicator_lock_path=ADJUDICATOR_LOCK,
            candidate_id="crossref-dev-v1",
            output_dir=tmp_path / "candidate-corpus-drift",
        )

    changed_runtime = json.loads(json.dumps(runtime))
    changed_runtime["adjudicator_seed"] = 7
    with pytest.raises(ValueError, match="manifest or Stage 2 runtime"):
        build_stage2_candidate_bundle(
            manifest=manifest,
            private_labels=labels,
            raw_scores=raw_scores,
            runtime=changed_runtime,
            reranker_lock_path=RERANKER_LOCK,
            adjudicator_lock_path=ADJUDICATOR_LOCK,
            candidate_id="crossref-dev-v1",
            output_dir=tmp_path / "candidate-runtime-drift",
        )
    assert not (tmp_path / "candidate-corpus-drift").exists()
    assert not (tmp_path / "candidate-runtime-drift").exists()


def test_build_candidate_rejects_topic_query_text_drift(tmp_path: Path) -> None:
    manifest, snapshot = _inputs()
    labels = _labels(manifest)
    runtime = _runtime()
    raw_scores = _raw_scores(manifest, snapshot, labels, runtime)
    changed_queries = dict(raw_scores.topic_queries)
    changed_queries[next(iter(changed_queries))] = "different query"

    with pytest.raises(ValueError, match="frozen Stage 2 runtime"):
        build_stage2_candidate_bundle(
            manifest=manifest,
            private_labels=labels,
            raw_scores=replace(raw_scores, topic_queries=changed_queries),
            runtime=runtime,
            reranker_lock_path=RERANKER_LOCK,
            adjudicator_lock_path=ADJUDICATOR_LOCK,
            candidate_id="crossref-dev-v1",
            output_dir=tmp_path / "candidate-query-drift",
        )

    assert not (tmp_path / "candidate-query-drift").exists()


def test_candidate_marker_is_absent_when_final_self_validation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, snapshot = _inputs()
    labels = _labels(manifest)
    runtime = _runtime()
    output = tmp_path / "candidate"
    monkeypatch.setattr(
        stage2_candidate,
        "load_stage2_benchmark_candidate",
        lambda _path: (_ for _ in ()).throw(ValueError("candidate invalid")),
    )

    with pytest.raises(ValueError, match="candidate invalid"):
        build_stage2_candidate_bundle(
            manifest=manifest,
            private_labels=labels,
            raw_scores=_raw_scores(manifest, snapshot, labels, runtime),
            runtime=runtime,
            reranker_lock_path=RERANKER_LOCK,
            adjudicator_lock_path=ADJUDICATOR_LOCK,
            candidate_id="crossref-dev-v1",
            output_dir=output,
        )

    assert output.is_dir()
    assert not (output / "stage2-candidate-v2.json").exists()
    assert not list(output.glob(".stage2-candidate-v2.json.*.tmp"))


@pytest.mark.parametrize("existing", ("directory", "file", "symlink"))
def test_candidate_output_claim_never_replaces_an_existing_target(
    tmp_path: Path, existing: str,
) -> None:
    manifest, snapshot = _inputs()
    labels = _labels(manifest)
    runtime = _runtime()
    output = tmp_path / "candidate"
    if existing == "directory":
        output.mkdir()
    elif existing == "file":
        output.write_text("keep", encoding="utf-8")
    else:
        output.symlink_to(tmp_path / "missing")

    with pytest.raises(FileExistsError, match="already exists"):
        build_stage2_candidate_bundle(
            manifest=manifest,
            private_labels=labels,
            raw_scores=_raw_scores(manifest, snapshot, labels, runtime),
            runtime=runtime,
            reranker_lock_path=RERANKER_LOCK,
            adjudicator_lock_path=ADJUDICATOR_LOCK,
            candidate_id="crossref-dev-v1",
            output_dir=output,
        )

    if existing == "file":
        assert output.read_text(encoding="utf-8") == "keep"
    elif existing == "symlink":
        assert output.is_symlink()
