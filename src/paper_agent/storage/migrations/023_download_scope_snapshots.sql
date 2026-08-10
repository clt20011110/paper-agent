CREATE TABLE download_scope_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    snapshot_type TEXT NOT NULL CHECK (snapshot_type IN ('collection', 'selection')),
    snapshot_hash TEXT NOT NULL UNIQUE,
    collection_id TEXT REFERENCES collections(collection_id),
    paper_ids_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX idx_download_scope_snapshots_hash
ON download_scope_snapshots(snapshot_hash);

ALTER TABLE authorized_download_queue_reservations
ADD COLUMN collection_id TEXT REFERENCES collections(collection_id);

ALTER TABLE authorized_download_queue_reservations
ADD COLUMN collection_snapshot_hash TEXT;

ALTER TABLE authorized_download_queue_reservations
ADD COLUMN selection_snapshot_hash TEXT;
