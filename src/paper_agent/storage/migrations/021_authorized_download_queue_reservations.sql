CREATE TABLE authorized_download_queue_reservations (
    authorization_grant_id TEXT NOT NULL REFERENCES authorization_grants(grant_id),
    paper_id TEXT NOT NULL REFERENCES papers(paper_id),
    candidate_id TEXT NOT NULL REFERENCES download_candidates(candidate_id),
    run_id TEXT NOT NULL REFERENCES pipeline_runs(run_id),
    queue_path TEXT NOT NULL,
    queue_item_hash TEXT NOT NULL,
    reserved_at TEXT NOT NULL,
    PRIMARY KEY (authorization_grant_id, paper_id),
    UNIQUE (run_id, paper_id),
    UNIQUE (run_id, candidate_id)
);

CREATE INDEX idx_authorized_download_queue_reservations_run
ON authorized_download_queue_reservations(run_id);
