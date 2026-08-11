CREATE TABLE crawl_execution_attempts (
    crawl_attempt_id TEXT PRIMARY KEY,
    crawl_run_id TEXT NOT NULL REFERENCES crawl_runs(crawl_run_id),
    attempt_no INTEGER NOT NULL CHECK (attempt_no > 0),
    started_at_epoch REAL NOT NULL,
    completed_at_epoch REAL,
    elapsed_seconds REAL NOT NULL DEFAULT 0 CHECK (elapsed_seconds >= 0),
    status TEXT NOT NULL CHECK (status IN ('running', 'complete', 'failed')),
    UNIQUE(crawl_run_id, attempt_no)
);

CREATE TABLE provider_request_attempts (
    request_attempt_id TEXT PRIMARY KEY,
    crawl_run_id TEXT NOT NULL REFERENCES crawl_runs(crawl_run_id),
    source_run_id TEXT REFERENCES source_runs(source_run_id),
    citation_request_id TEXT REFERENCES citation_requests(citation_request_id),
    operation_key TEXT NOT NULL,
    attempt_no INTEGER NOT NULL CHECK (attempt_no > 0),
    provider TEXT NOT NULL,
    role TEXT NOT NULL,
    query_hash TEXT NOT NULL,
    requested_cursor TEXT,
    request_charged INTEGER NOT NULL CHECK (request_charged >= 0),
    accepted_count INTEGER NOT NULL CHECK (accepted_count >= 0),
    raw_returned_count INTEGER NOT NULL CHECK (raw_returned_count >= 0),
    status TEXT NOT NULL CHECK (status IN ('running', 'success', 'partial', 'failed')),
    error_json TEXT,
    response_hash TEXT,
    response_artifact_id TEXT REFERENCES artifacts(artifact_id),
    started_at TEXT NOT NULL,
    completed_at TEXT,
    UNIQUE(crawl_run_id, operation_key, attempt_no)
);

ALTER TABLE search_queries
ADD COLUMN request_attempt_id TEXT REFERENCES provider_request_attempts(request_attempt_id);

CREATE INDEX idx_provider_request_attempts_budget
    ON provider_request_attempts(crawl_run_id, request_charged, accepted_count);

CREATE INDEX idx_search_queries_request_attempt
    ON search_queries(request_attempt_id);
