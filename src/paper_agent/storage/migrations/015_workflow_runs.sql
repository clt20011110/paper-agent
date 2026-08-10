CREATE TABLE workflow_runs (
    workflow_run_id TEXT PRIMARY KEY,
    manifest_hash TEXT NOT NULL,
    manifest_json TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN (
        'pending', 'running', 'complete', 'incomplete', 'blocked', 'failed'
    )),
    lease_owner TEXT,
    lease_token INTEGER NOT NULL DEFAULT 0 CHECK (lease_token >= 0),
    lease_expires_at TEXT,
    error_json TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE workflow_steps (
    workflow_run_id TEXT NOT NULL REFERENCES workflow_runs(workflow_run_id),
    step_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    stage TEXT NOT NULL CHECK (stage IN ('search', 'filter', 'download', 'analyze', 'report')),
    child_run_id TEXT NOT NULL,
    spec_hash TEXT NOT NULL,
    identity_hash TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN (
        'pending', 'running', 'complete', 'incomplete', 'blocked',
        'uncertain_terminal', 'failed'
    )),
    lease_owner TEXT,
    lease_token INTEGER NOT NULL DEFAULT 0 CHECK (lease_token >= 0),
    lease_expires_at TEXT,
    result_json TEXT,
    error_json TEXT,
    started_at TEXT,
    completed_at TEXT,
    PRIMARY KEY (workflow_run_id, step_id),
    UNIQUE (workflow_run_id, ordinal),
    UNIQUE (child_run_id)
);
