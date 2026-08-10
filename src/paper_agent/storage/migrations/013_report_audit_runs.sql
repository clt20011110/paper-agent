CREATE TABLE report_audit_runs (
    report_run_id TEXT PRIMARY KEY REFERENCES report_runs(report_run_id),
    input_snapshot_hash TEXT NOT NULL,
    base_artifact_hash TEXT NOT NULL,
    current_artifact_hash TEXT NOT NULL,
    current_bundle_json TEXT NOT NULL,
    rubric_hash TEXT NOT NULL,
    profile TEXT NOT NULL CHECK (profile = 'stage4b_summary_sol'),
    model_id TEXT NOT NULL CHECK (model_id = 'gpt-5.6-sol'),
    reasoning_effort TEXT NOT NULL CHECK (reasoning_effort = 'high'),
    config_hash TEXT NOT NULL,
    execution_mode TEXT NOT NULL CHECK (execution_mode IN ('attended', 'unattended')),
    worst_case_calls INTEGER NOT NULL CHECK (worst_case_calls > 0),
    worst_case_input_tokens INTEGER NOT NULL CHECK (worst_case_input_tokens > 0),
    repair_count INTEGER NOT NULL DEFAULT 0 CHECK (repair_count IN (0, 1)),
    status TEXT NOT NULL CHECK (status IN (
        'pending', 'running', 'complete', 'incomplete', 'manual_required', 'failed'
    )),
    final_audit_step TEXT CHECK (final_audit_step IN ('audit_a', 'audit_c')),
    published_relative_path TEXT,
    error_json TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TEXT
);

CREATE TABLE report_audit_steps (
    report_audit_step_id TEXT PRIMARY KEY,
    report_run_id TEXT NOT NULL REFERENCES report_audit_runs(report_run_id),
    step_name TEXT NOT NULL CHECK (step_name IN ('audit_a', 'repair', 'audit_c')),
    call_kind TEXT NOT NULL CHECK (call_kind IN ('quality_audit', 'repair')),
    input_artifact_hash TEXT NOT NULL,
    input_bundle_hash TEXT NOT NULL,
    expected_coverage_hash TEXT,
    actual_input_hash TEXT,
    rendered_prompt_hash TEXT,
    actual_input_tokens INTEGER CHECK (actual_input_tokens > 0),
    input_token_limit INTEGER NOT NULL CHECK (input_token_limit > 0),
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
        'pending', 'running', 'complete', 'manual_required', 'retryable', 'failed'
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
    CHECK (
        (step_name IN ('audit_a', 'audit_c') AND call_kind = 'quality_audit')
        OR (step_name = 'repair' AND call_kind = 'repair')
    ),
    UNIQUE(report_run_id, step_name)
);

CREATE INDEX idx_report_audit_steps_run_status
    ON report_audit_steps(report_run_id, status, step_name);
