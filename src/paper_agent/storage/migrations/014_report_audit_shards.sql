CREATE TABLE report_audit_shard_steps (
    report_audit_shard_step_id TEXT PRIMARY KEY,
    report_run_id TEXT NOT NULL REFERENCES report_audit_runs(report_run_id),
    audit_pass TEXT NOT NULL CHECK (audit_pass IN ('A', 'C')),
    node_id TEXT NOT NULL,
    node_kind TEXT NOT NULL CHECK (node_kind IN ('shard', 'audit_reduce')),
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    source_node_ids_json TEXT NOT NULL,
    input_artifact_hash TEXT NOT NULL,
    input_bundle_hash TEXT NOT NULL,
    expected_coverage_hash TEXT NOT NULL,
    actual_input_hash TEXT,
    rendered_prompt_hash TEXT,
    actual_input_tokens INTEGER CHECK (actual_input_tokens > 0),
    input_token_limit INTEGER NOT NULL CHECK (input_token_limit > 0),
    output_byte_limit INTEGER NOT NULL CHECK (output_byte_limit > 0),
    budget_calls_reserved INTEGER NOT NULL DEFAULT 0 CHECK (budget_calls_reserved >= 0),
    budget_tokens_reserved INTEGER NOT NULL DEFAULT 0 CHECK (budget_tokens_reserved >= 0),
    profile TEXT NOT NULL CHECK (profile = 'stage4b_summary_sol'),
    model_id TEXT NOT NULL CHECK (model_id = 'gpt-5.6-sol'),
    reasoning_effort TEXT NOT NULL CHECK (reasoning_effort = 'high'),
    prompt_name TEXT NOT NULL,
    prompt_hash TEXT NOT NULL,
    schema_name TEXT NOT NULL,
    schema_hash TEXT NOT NULL,
    processing_facts_json TEXT NOT NULL,
    processing_decision_json TEXT,
    processing_grant_id TEXT,
    invocation_metadata_json TEXT,
    output_json TEXT,
    output_hash TEXT,
    status TEXT NOT NULL CHECK (status IN (
        'pending', 'running', 'complete', 'manual_required', 'failed'
    )),
    dispatch_count INTEGER NOT NULL DEFAULT 0 CHECK (dispatch_count >= 0),
    lease_owner TEXT,
    lease_token INTEGER NOT NULL DEFAULT 0 CHECK (lease_token >= 0),
    lease_expires_at TEXT,
    error_json TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TEXT,
    CHECK (
        (status = 'running' AND lease_owner IS NOT NULL AND lease_expires_at IS NOT NULL)
        OR (status <> 'running' AND lease_owner IS NULL AND lease_expires_at IS NULL)
    ),
    UNIQUE(report_run_id, audit_pass, node_id),
    UNIQUE(report_run_id, audit_pass, ordinal)
);

CREATE INDEX idx_report_audit_shards_run_status
    ON report_audit_shard_steps(report_run_id, audit_pass, status, ordinal);

CREATE TABLE report_sol_invocations (
    report_run_id TEXT NOT NULL REFERENCES report_runs(report_run_id),
    invocation_id TEXT NOT NULL,
    phase TEXT NOT NULL CHECK (phase IN ('reduce', 'audit_step', 'audit_shard')),
    node_key TEXT NOT NULL,
    metadata_hash TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY(report_run_id, invocation_id),
    UNIQUE(report_run_id, phase, node_key)
);

CREATE INDEX idx_report_sol_invocations_run_phase
    ON report_sol_invocations(report_run_id, phase, node_key);
