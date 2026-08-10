CREATE TABLE stage3_luna_decisions (
    planner_decision_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES pipeline_runs(run_id),
    candidate_id TEXT NOT NULL REFERENCES download_candidates(candidate_id),
    authorization_grant_id TEXT NOT NULL REFERENCES authorization_grants(grant_id),
    status TEXT NOT NULL CHECK (status IN ('pending', 'complete')),
    selected INTEGER CHECK (selected IN (0, 1)),
    planner_status TEXT,
    page_state TEXT,
    next_action TEXT,
    reason_code TEXT NOT NULL,
    invocation_metadata_json TEXT,
    decided_at TEXT NOT NULL,
    UNIQUE(run_id, candidate_id)
);

CREATE INDEX idx_stage3_luna_decisions_run
ON stage3_luna_decisions(run_id, candidate_id);

ALTER TABLE download_attempts
ADD COLUMN planner_decision_id TEXT REFERENCES stage3_luna_decisions(planner_decision_id);
