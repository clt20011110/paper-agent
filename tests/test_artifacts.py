from __future__ import annotations

import hashlib

import pytest

from paper_agent.artifacts import ArtifactMetadataConflict, ArtifactStore, CoordinatorArtifactMerger
from paper_agent.sharding import ArtifactManifest, WorkerResultManifest, build_shards, bump_epoch, freeze_snapshot
from paper_agent.storage import Database


def _database(tmp_path) -> Database:
    database = Database(tmp_path / "papers.sqlite3")
    database.migrate()
    database.connection.execute(
        """INSERT INTO pipeline_runs(run_id, stage, status, input_hash, config_hash, implementation_version)
           VALUES ('run-1', 'stage4', 'running', 'input', 'config', 'test')"""
    )
    for paper_id in ("paper-a", "paper-b"):
        database.connection.execute("INSERT INTO papers(paper_id, title) VALUES (?, ?)", (paper_id, paper_id))
    database.connection.commit()
    return database


def _shard(paper_ids=("paper-a",)):
    snapshot = freeze_snapshot("run-1", paper_ids, "c" * 64, {"model": "local"}, "a" * 64)
    return build_shards(snapshot, 1, "worker-output")[0]


def _result(shard, artifact: ArtifactManifest) -> WorkerResultManifest:
    return WorkerResultManifest(
        shard.run_id, shard.snapshot_hash, shard.shard_id, shard.epoch, shard.fencing_token,
        "stage4", shard.paper_ids, (artifact,),
    )


def _bundle(tmp_path, payload: bytes, *, paper_id="paper-a", output_kind="analysis"):
    artifact_hash = hashlib.sha256(payload).hexdigest()
    filename = f"{artifact_hash}.json"
    path = tmp_path / "worker-output" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return ArtifactManifest(paper_id, output_kind, artifact_hash, "application/json", len(payload), f"worker-output/{filename}")


def test_store_is_atomic_content_addressed_and_reuses_identical_metadata(tmp_path) -> None:
    store = ArtifactStore(tmp_path)

    first = store.put_bytes(b'{"answer": 42}', mime_type="application/json", metadata={"kind": "analysis"})
    second = store.put_bytes(b'{"answer": 42}', mime_type="application/json", metadata={"kind": "analysis"})

    assert first == second
    assert first.path == tmp_path / "artifacts" / first.artifact_hash[:2] / first.artifact_hash
    assert store.read_bytes(first.artifact_hash) == b'{"answer": 42}'
    with pytest.raises(ArtifactMetadataConflict):
        store.put_bytes(b'{"answer": 42}', mime_type="text/plain", metadata={"kind": "analysis"})


def test_store_rejects_corrupted_content(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    stored = store.put_bytes(b"content")
    stored.path.write_bytes(b"damaged")

    with pytest.raises(ValueError, match="corrupted"):
        store.read_bytes(stored.artifact_hash)


def test_merge_reads_bundle_and_is_idempotent(tmp_path) -> None:
    database = _database(tmp_path)
    try:
        merger = CoordinatorArtifactMerger(database, ArtifactStore(tmp_path / "coordinator"))
        shard = _shard()
        artifact = _bundle(tmp_path, b'{"ok": true}')
        result = _result(shard, artifact)

        first = merger.apply([shard], [result], tmp_path, stage="stage4", output_kinds=["analysis"])
        second = merger.apply([shard], [result], tmp_path, stage="stage4", output_kinds=["analysis"])

        assert first.applied
        assert second.applied
        assert database.connection.execute("SELECT COUNT(*) FROM run_outputs").fetchone()[0] == 1
        assert database.connection.execute("SELECT relative_path FROM artifacts").fetchone()[0].startswith("artifacts/")
    finally:
        database.close()


def test_corrupt_or_hash_mismatched_bundle_is_not_committed(tmp_path) -> None:
    database = _database(tmp_path)
    try:
        merger = CoordinatorArtifactMerger(database, ArtifactStore(tmp_path / "coordinator"))
        shard = _shard()
        artifact = _bundle(tmp_path, b'{"ok": true}')
        bad = ArtifactManifest(
            artifact.paper_id, artifact.output_kind, "f" * 64, artifact.mime_type,
            artifact.size_bytes, artifact.relative_path,
        )

        applied = merger.apply([shard], [_result(shard, bad)], tmp_path, stage="stage4", output_kinds=["analysis"])

        assert not applied.applied
        assert database.connection.execute("SELECT COUNT(*) FROM run_outputs").fetchone()[0] == 0
        assert database.connection.execute("SELECT queue_type FROM manual_queue").fetchone()[0] == "merge_conflict"
    finally:
        database.close()


def test_conflict_and_old_epoch_are_isolated_in_manual_queue(tmp_path) -> None:
    database = _database(tmp_path)
    try:
        merger = CoordinatorArtifactMerger(database, ArtifactStore(tmp_path / "coordinator"))
        shard = _shard()
        first_artifact = _bundle(tmp_path, b'{"version": 1}')
        assert merger.apply([shard], [_result(shard, first_artifact)], tmp_path, stage="stage4", output_kinds=["analysis"]).applied

        changed_artifact = _bundle(tmp_path, b'{"version": 2}')
        conflict = merger.apply([shard], [_result(shard, changed_artifact)], tmp_path, stage="stage4", output_kinds=["analysis"])
        stale = merger.apply([bump_epoch(shard)], [_result(shard, first_artifact)], tmp_path, stage="stage4", output_kinds=["analysis"])

        assert not conflict.applied and conflict.manual_queue_ids
        assert not stale.applied and stale.manual_queue_ids
        assert database.connection.execute("SELECT COUNT(*) FROM run_outputs").fetchone()[0] == 1
        assert database.connection.execute("SELECT COUNT(*) FROM manual_queue").fetchone()[0] == 2
    finally:
        database.close()


def test_missing_coverage_does_not_commit_partial_outputs(tmp_path) -> None:
    database = _database(tmp_path)
    try:
        merger = CoordinatorArtifactMerger(database, ArtifactStore(tmp_path / "coordinator"))
        shard = _shard(("paper-a", "paper-b"))
        artifact = _bundle(tmp_path, b'{"only": "a"}')
        result = _result(shard, artifact)

        applied = merger.apply([shard], [result], tmp_path, stage="stage4", output_kinds=["analysis"])

        assert not applied.applied
        assert applied.plan and applied.plan.missing_coverage
        assert database.connection.execute("SELECT COUNT(*) FROM run_outputs").fetchone()[0] == 0
    finally:
        database.close()


def test_worker_bundle_symlink_cannot_escape_its_root(tmp_path) -> None:
    database = _database(tmp_path)
    try:
        merger = CoordinatorArtifactMerger(database, ArtifactStore(tmp_path / "coordinator"))
        shard = _shard()
        payload = b'{"outside": true}'
        outside = tmp_path.parent / f"{tmp_path.name}-outside.json"
        outside.write_bytes(payload)
        link = tmp_path / "worker-output" / "linked.json"
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(outside)
        artifact = ArtifactManifest(
            "paper-a",
            "analysis",
            hashlib.sha256(payload).hexdigest(),
            "application/json",
            len(payload),
            "worker-output/linked.json",
        )

        result = merger.apply(
            [shard], [_result(shard, artifact)], tmp_path, stage="stage4", output_kinds=["analysis"]
        )

        assert not result.applied and result.manual_queue_ids
        assert database.connection.execute("SELECT COUNT(*) FROM run_outputs").fetchone()[0] == 0
    finally:
        database.close()
