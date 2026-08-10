ALTER TABLE download_attempts RENAME TO download_attempts_v1;

CREATE TABLE download_attempts (
    download_attempt_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES pipeline_runs(run_id),
    candidate_id TEXT NOT NULL REFERENCES download_candidates(candidate_id),
    provider TEXT NOT NULL,
    authorization_grant_id TEXT REFERENCES authorization_grants(grant_id),
    fetch_request_id TEXT NOT NULL REFERENCES fetch_requests(request_id),
    result_status TEXT NOT NULL CHECK (result_status IN (
        'pending', 'downloaded', 'not_available', 'auth_required',
        'manual_required', 'failed_retryable', 'failed_terminal'
    )),
    failure_category TEXT,
    http_status INTEGER,
    browser_result_json TEXT,
    artifact_id TEXT REFERENCES artifacts(artifact_id),
    attempted_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(run_id, fetch_request_id, provider)
);

INSERT INTO download_attempts(
    download_attempt_id, run_id, candidate_id, provider, authorization_grant_id,
    fetch_request_id, result_status, failure_category, http_status,
    browser_result_json, artifact_id, attempted_at
)
SELECT
    download_attempt_id, run_id, candidate_id, provider, authorization_grant_id,
    fetch_request_id,
    CASE
        WHEN result_status = 'manual_required' AND failure_category = 'auth_required'
            THEN 'auth_required'
        WHEN result_status = 'failed_terminal' AND failure_category = 'not_available'
            THEN 'not_available'
        ELSE result_status
    END,
    failure_category, http_status, browser_result_json, artifact_id, attempted_at
FROM download_attempts_v1;

DROP TABLE download_attempts_v1;

CREATE TABLE download_policy_decisions (
    decision_id TEXT PRIMARY KEY,
    candidate_id TEXT NOT NULL REFERENCES download_candidates(candidate_id),
    run_id TEXT REFERENCES pipeline_runs(run_id),
    provider TEXT NOT NULL,
    purpose TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    policy_hash TEXT NOT NULL,
    provider_terms_hash TEXT,
    authorization_grant_id TEXT REFERENCES authorization_grants(grant_id),
    decision TEXT NOT NULL CHECK (decision IN ('allow', 'needs_grant', 'manual', 'deny')),
    reason_code TEXT NOT NULL,
    decided_at TEXT NOT NULL
);

CREATE INDEX idx_download_policy_decisions_candidate
ON download_policy_decisions(candidate_id, decided_at);
