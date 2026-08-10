CREATE TABLE report_reduce_nodes (
    report_reduce_node_id TEXT PRIMARY KEY,
    report_run_id TEXT NOT NULL REFERENCES report_runs(report_run_id),
    node_id TEXT NOT NULL,
    call_kind TEXT NOT NULL CHECK (call_kind IN (
        'section_reduce', 'cross_section_reduce', 'final_reduce'
    )),
    section_ids_json TEXT NOT NULL,
    paper_ids_json TEXT NOT NULL,
    dependency_ids_json TEXT NOT NULL,
    planned_input_hash TEXT NOT NULL,
    actual_input_hash TEXT,
    rendered_prompt_hash TEXT,
    input_artifact_hashes_json TEXT NOT NULL DEFAULT '[]',
    input_tokens INTEGER NOT NULL CHECK (input_tokens >= 0),
    prompt_token_bound INTEGER NOT NULL CHECK (prompt_token_bound > 0),
    actual_input_tokens INTEGER CHECK (actual_input_tokens > 0),
    output_byte_limit INTEGER NOT NULL CHECK (output_byte_limit > 0),
    budget_calls_reserved INTEGER NOT NULL DEFAULT 0 CHECK (budget_calls_reserved >= 0),
    budget_tokens_reserved INTEGER NOT NULL DEFAULT 0 CHECK (budget_tokens_reserved >= 0),
    processing_decisions_json TEXT NOT NULL DEFAULT '[]',
    processing_grant_ids_json TEXT NOT NULL DEFAULT '[]',
    profile TEXT NOT NULL,
    model_id TEXT NOT NULL,
    reasoning_effort TEXT NOT NULL,
    prompt_name TEXT NOT NULL,
    prompt_hash TEXT NOT NULL,
    schema_name TEXT NOT NULL,
    schema_hash TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN (
        'pending', 'running', 'complete', 'manual_required', 'retryable', 'failed'
    )),
    dispatch_count INTEGER NOT NULL DEFAULT 0 CHECK (dispatch_count >= 0),
    lease_owner TEXT,
    lease_token INTEGER NOT NULL DEFAULT 0 CHECK (lease_token >= 0),
    lease_expires_at TEXT,
    invocation_metadata_json TEXT,
    invocation_id TEXT,
    output_artifact_id TEXT REFERENCES artifacts(artifact_id),
    output_hash TEXT,
    output_policy_json TEXT,
    error_json TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TEXT,
    CHECK (
        (status = 'running' AND lease_owner IS NOT NULL AND lease_expires_at IS NOT NULL)
        OR (status <> 'running' AND lease_owner IS NULL AND lease_expires_at IS NULL)
    ),
    UNIQUE(report_run_id, node_id),
    UNIQUE(report_run_id, invocation_id)
);

CREATE INDEX idx_report_reduce_nodes_run_status
    ON report_reduce_nodes(report_run_id, status, node_id);
