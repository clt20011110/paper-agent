from __future__ import annotations

import pytest

from paper_agent.sharding import (
    ArtifactManifest,
    ArtifactReceipt,
    MergeRejected,
    MergedArtifact,
    WorkerResultManifest,
    build_shards,
    bump_epoch,
    freeze_snapshot,
    plan_merge,
)


HASH_A = "a" * 64
HASH_B = "b" * 64


def _snapshot(paper_ids: list[str] | None = None):
    return freeze_snapshot(
        "run-1", paper_ids or ["paper-c", "paper-a", "paper-b"], "c" * 64, {"model": "qwen"}, HASH_A
    )


def _artifact(paper_id: str, artifact_hash: str = HASH_A, path: str = "worker/output.json") -> ArtifactManifest:
    return ArtifactManifest(paper_id, "analysis", artifact_hash, "application/json", 12, path)


def _receipt(artifact_hash: str = HASH_A, size_bytes: int = 12) -> ArtifactReceipt:
    return ArtifactReceipt(artifact_hash, "application/json", size_bytes)


def _result(shard, artifacts: tuple[ArtifactManifest, ...]) -> WorkerResultManifest:
    return WorkerResultManifest(
        shard.run_id,
        shard.snapshot_hash,
        shard.shard_id,
        shard.epoch,
        shard.fencing_token,
        "stage4",
        shard.paper_ids,
        artifacts,
    )


def test_snapshot_and_shards_are_order_independent_and_mutually_exclusive() -> None:
    first = _snapshot(["paper-d", "paper-a", "paper-c", "paper-b"])
    second = _snapshot(["paper-b", "paper-d", "paper-a", "paper-c"])

    first_shards = build_shards(first, 3, "worker-results")
    second_shards = build_shards(second, 3, "worker-results")

    assert first.snapshot_hash == second.snapshot_hash
    assert first_shards == second_shards
    assigned = [paper_id for shard in first_shards for paper_id in shard.paper_ids]
    assert assigned == ["paper-a", "paper-b", "paper-c", "paper-d"]
    assert len(set(assigned)) == len(assigned)
    assert max(map(lambda shard: len(shard.paper_ids), first_shards)) - min(
        map(lambda shard: len(shard.paper_ids), first_shards)
    ) <= 1


def test_snapshot_rejects_empty_or_duplicate_paper_ids() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        freeze_snapshot("run-1", [], "c" * 64, {"model": "qwen"}, HASH_A)
    with pytest.raises(ValueError, match="unique"):
        _snapshot(["paper-a", "paper-a"])
    with pytest.raises(ValueError, match="relative"):
        build_shards(_snapshot(), 1, "/tmp/worker")


def test_old_epoch_result_is_fenced_off() -> None:
    shard = build_shards(_snapshot(), 1, "worker-results")[0]
    stale = _result(shard, tuple(_artifact(paper_id) for paper_id in shard.paper_ids))
    current = bump_epoch(shard)

    with pytest.raises(MergeRejected, match="fencing"):
        plan_merge([current], [stale], {HASH_A: _receipt()}, stage="stage4", output_kinds=["analysis"])


def test_same_key_same_hash_is_idempotent_and_does_not_keep_worker_path() -> None:
    shard = build_shards(_snapshot(["paper-a"]), 1, "worker-results")[0]
    artifact = _artifact("paper-a", path="machine-a/result.json")
    existing = MergedArtifact("run-1", "stage4", "paper-a", "analysis", HASH_A, "application/json", 12, "artifacts/old")

    merge = plan_merge(
        [shard],
        [_result(shard, (artifact,))],
        {HASH_A: _receipt()},
        stage="stage4",
        output_kinds=["analysis"],
        existing_outputs={existing.key: existing},
    )

    assert merge.complete
    assert merge.new_outputs == ()
    assert merge.idempotent_keys == (existing.key,)


def test_complete_merge_rewrites_artifact_to_the_coordinator_root() -> None:
    shard = build_shards(_snapshot(["paper-a"]), 1, "worker-results")[0]
    merge = plan_merge(
        [shard],
        [_result(shard, (_artifact("paper-a", path="machine-a/result.json"),))],
        {HASH_A: _receipt()},
        stage="stage4",
        output_kinds=["analysis"],
        coordinator_output_root="coordinator-artifacts",
    )

    assert merge.complete
    assert merge.new_outputs[0].relative_path == f"coordinator-artifacts/{HASH_A}"


def test_conflicting_hash_is_isolated_without_last_write_wins() -> None:
    shard = build_shards(_snapshot(["paper-a"]), 1, "worker-results")[0]
    existing = MergedArtifact("run-1", "stage4", "paper-a", "analysis", HASH_A, "application/json", 12, "artifacts/old")

    merge = plan_merge(
        [shard],
        [_result(shard, (_artifact("paper-a", HASH_B),))],
        {HASH_B: _receipt(HASH_B)},
        stage="stage4",
        output_kinds=["analysis"],
        existing_outputs={existing.key: existing},
    )

    assert not merge.complete
    assert merge.new_outputs == ()
    assert merge.conflicts[0].key == existing.key
    assert merge.conflicts[0].existing_hash == HASH_A
    assert merge.conflicts[0].incoming_hash == HASH_B


def test_missing_coverage_prevents_partial_merge() -> None:
    shard = build_shards(_snapshot(["paper-a", "paper-b"]), 1, "worker-results")[0]
    merge = plan_merge(
        [shard],
        [_result(shard, (_artifact("paper-a"),))],
        {HASH_A: _receipt()},
        stage="stage4",
        output_kinds=["analysis"],
    )

    assert not merge.complete
    assert merge.new_outputs == ()
    assert merge.missing_coverage == (("run-1", "stage4", "paper-b", "analysis"),)


def test_absolute_worker_path_and_receipt_mismatch_are_rejected() -> None:
    shard = build_shards(_snapshot(["paper-a"]), 1, "worker-results")[0]
    with pytest.raises(MergeRejected, match="relative"):
        plan_merge(
            [shard],
            [_result(shard, (_artifact("paper-a", path="/private/worker/result.json"),))],
            {HASH_A: _receipt()},
            stage="stage4",
            output_kinds=["analysis"],
        )
    with pytest.raises(MergeRejected, match="metadata"):
        plan_merge(
            [shard],
            [_result(shard, (_artifact("paper-a"),))],
            {HASH_A: _receipt(size_bytes=13)},
            stage="stage4",
            output_kinds=["analysis"],
        )
