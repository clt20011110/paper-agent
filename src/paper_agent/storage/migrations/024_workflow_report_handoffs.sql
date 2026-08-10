CREATE TABLE workflow_report_handoffs (
    handoff_id TEXT PRIMARY KEY,
    workflow_run_id TEXT NOT NULL REFERENCES workflow_runs(workflow_run_id),
    workflow_manifest_hash TEXT NOT NULL,
    workflow_binding_hash TEXT NOT NULL,
    request_hash TEXT NOT NULL,
    crawl_run_id TEXT NOT NULL REFERENCES crawl_runs(crawl_run_id),
    filter_run_id TEXT NOT NULL REFERENCES pipeline_runs(run_id),
    download_run_id TEXT NOT NULL REFERENCES pipeline_runs(run_id),
    stage4_run_id TEXT NOT NULL REFERENCES pipeline_runs(run_id),
    recent_cutoff TEXT NOT NULL,
    input_created_at TEXT NOT NULL,
    include_needs_review INTEGER NOT NULL CHECK (include_needs_review IN (0, 1)),
    artifact_root TEXT NOT NULL,
    output_root TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('preparing', 'complete')),
    bundle_id TEXT,
    bundle_hash TEXT,
    corpus_snapshot_hash TEXT,
    corpus_file_sha256 TEXT,
    corpus_snapshot_path TEXT,
    search_audit_pack_hash TEXT,
    search_audit_file_sha256 TEXT,
    search_audit_path TEXT,
    prepared_at TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (workflow_run_id, request_hash),
    CHECK (length(workflow_manifest_hash) = 64 AND workflow_manifest_hash NOT GLOB '*[^0-9a-f]*'),
    CHECK (length(workflow_binding_hash) = 64 AND workflow_binding_hash NOT GLOB '*[^0-9a-f]*'),
    CHECK (length(request_hash) = 64 AND request_hash NOT GLOB '*[^0-9a-f]*'),
    CHECK (
        (status = 'preparing'
         AND bundle_id IS NULL
         AND bundle_hash IS NULL
         AND corpus_snapshot_hash IS NULL
         AND corpus_file_sha256 IS NULL
         AND corpus_snapshot_path IS NULL
         AND search_audit_pack_hash IS NULL
         AND search_audit_file_sha256 IS NULL
         AND search_audit_path IS NULL
         AND prepared_at IS NULL)
        OR
        (status = 'complete'
         AND bundle_id IS NOT NULL
         AND length(bundle_hash) = 64 AND bundle_hash NOT GLOB '*[^0-9a-f]*'
         AND length(corpus_snapshot_hash) = 64 AND corpus_snapshot_hash NOT GLOB '*[^0-9a-f]*'
         AND length(corpus_file_sha256) = 64 AND corpus_file_sha256 NOT GLOB '*[^0-9a-f]*'
         AND corpus_snapshot_path IS NOT NULL
         AND length(search_audit_pack_hash) = 64 AND search_audit_pack_hash NOT GLOB '*[^0-9a-f]*'
         AND length(search_audit_file_sha256) = 64 AND search_audit_file_sha256 NOT GLOB '*[^0-9a-f]*'
         AND search_audit_path IS NOT NULL
         AND prepared_at IS NOT NULL)
    )
);

CREATE INDEX idx_workflow_report_handoffs_workflow
ON workflow_report_handoffs(workflow_run_id, status);

CREATE TRIGGER workflow_report_handoffs_complete_immutable
BEFORE UPDATE ON workflow_report_handoffs
WHEN OLD.status = 'complete'
BEGIN
    SELECT RAISE(ABORT, 'completed workflow report handoffs are immutable');
END;

CREATE TRIGGER workflow_report_handoffs_no_delete
BEFORE DELETE ON workflow_report_handoffs
BEGIN
    SELECT RAISE(ABORT, 'workflow report handoffs are immutable');
END;

CREATE TABLE workflow_report_executions (
    handoff_id TEXT PRIMARY KEY REFERENCES workflow_report_handoffs(handoff_id),
    report_plan_id TEXT NOT NULL UNIQUE REFERENCES report_plans(report_plan_id),
    report_plan_hash TEXT NOT NULL,
    report_plan_path TEXT NOT NULL,
    report_plan_file_sha256 TEXT NOT NULL,
    report_workflow_id TEXT NOT NULL,
    report_workflow_run_id TEXT NOT NULL UNIQUE,
    report_manifest_hash TEXT NOT NULL,
    report_manifest_json TEXT NOT NULL,
    report_manifest_path TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK (length(report_plan_hash) = 64 AND report_plan_hash NOT GLOB '*[^0-9a-f]*'),
    CHECK (length(report_plan_file_sha256) = 64 AND report_plan_file_sha256 NOT GLOB '*[^0-9a-f]*'),
    CHECK (length(report_manifest_hash) = 64 AND report_manifest_hash NOT GLOB '*[^0-9a-f]*')
);

CREATE TRIGGER workflow_report_executions_immutable
BEFORE UPDATE ON workflow_report_executions
BEGIN
    SELECT RAISE(ABORT, 'workflow report execution bindings are immutable');
END;

CREATE TRIGGER workflow_report_executions_no_delete
BEFORE DELETE ON workflow_report_executions
BEGIN
    SELECT RAISE(ABORT, 'workflow report execution bindings are immutable');
END;
