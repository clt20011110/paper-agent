CREATE TABLE stage3_paper_results (
    run_id TEXT NOT NULL REFERENCES pipeline_runs(run_id),
    paper_id TEXT NOT NULL REFERENCES papers(paper_id),
    status TEXT NOT NULL CHECK (status IN (
        'pending', 'downloaded', 'not_available', 'auth_required',
        'manual_required', 'failed_retryable', 'failed_terminal'
    )),
    reason_code TEXT NOT NULL CHECK (length(trim(reason_code)) > 0),
    updated_at TEXT NOT NULL,
    PRIMARY KEY (run_id, paper_id)
);

CREATE INDEX idx_stage3_paper_results_status
ON stage3_paper_results(run_id, status, paper_id);
