CREATE TABLE analysis_dispatches (
    dispatch_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES pipeline_runs(run_id),
    paper_id TEXT NOT NULL REFERENCES papers(paper_id),
    artifact_hash TEXT NOT NULL,
    artifact_id TEXT REFERENCES artifacts(artifact_id),
    input_scope TEXT NOT NULL CHECK (input_scope IN (
        'full_pdf', 'abstract_only', 'metadata_only'
    )),
    config_hash TEXT NOT NULL,
    implementation_version TEXT NOT NULL,
    profile TEXT NOT NULL CHECK (profile = 'stage4_analysis_luna'),
    model_id TEXT NOT NULL CHECK (model_id = 'gpt-5.6-luna'),
    prompt_hash TEXT NOT NULL,
    schema_hash TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    policy_hash TEXT NOT NULL,
    stable_created_at TEXT NOT NULL,
    prompt_input_hash TEXT,
    rendered_prompt_hash TEXT,
    processing_decision_json TEXT,
    processing_grant_id TEXT REFERENCES authorization_grants(grant_id),
    status TEXT NOT NULL CHECK (status IN (
        'prepared', 'running', 'complete', 'manual_required', 'failed_terminal'
    )),
    dispatch_count INTEGER NOT NULL DEFAULT 0 CHECK (dispatch_count BETWEEN 0 AND 1),
    lease_owner TEXT,
    lease_token INTEGER NOT NULL DEFAULT 0 CHECK (lease_token >= 0),
    lease_expires_at TEXT,
    invocation_id TEXT UNIQUE,
    invocation_metadata_json TEXT,
    analysis_run_id TEXT REFERENCES analysis_runs(analysis_run_id),
    error_json TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TEXT,
    UNIQUE (run_id, paper_id),
    CHECK (
        (status = 'running' AND lease_owner IS NOT NULL AND lease_expires_at IS NOT NULL)
        OR
        (status != 'running' AND lease_owner IS NULL AND lease_expires_at IS NULL)
    ),
    CHECK (
        (dispatch_count = 0 AND status IN ('prepared', 'manual_required'))
        OR
        (dispatch_count = 1 AND status IN ('running', 'complete', 'failed_terminal'))
    ),
    CHECK (status != 'running' OR prompt_input_hash IS NOT NULL),
    CHECK (status != 'complete' OR analysis_run_id IS NOT NULL),
    CHECK (status != 'failed_terminal' OR error_json IS NOT NULL)
);

CREATE INDEX analysis_dispatches_status_lease_idx
    ON analysis_dispatches(status, lease_expires_at);

CREATE INDEX analysis_dispatches_analysis_run_idx
    ON analysis_dispatches(analysis_run_id);

-- Before this ledger existed, a Stage 4 row could be left in `running` or
-- `failed` after a paid invocation had crossed the process boundary.  Treat
-- those outcomes conservatively: consume the one-call budget and make the
-- intent terminal.  A later coordinator may observe this tombstone, but must
-- never turn it back into a prepared dispatch.
--
-- Keep only the newest uncertain row per run/paper and do not shadow a known
-- complete result.  Older coordinators always persisted the selected artifact
-- hash in input_policy_facts; input_hash is a fail-closed digest fallback for
-- malformed/pre-policy rows.
INSERT INTO analysis_dispatches(
    dispatch_id, run_id, paper_id, artifact_hash, artifact_id, input_scope,
    config_hash, implementation_version, profile, model_id, prompt_hash,
    schema_hash, policy_version, policy_hash, stable_created_at,
    prompt_input_hash, rendered_prompt_hash, processing_decision_json,
    processing_grant_id, status, dispatch_count, invocation_id,
    invocation_metadata_json, analysis_run_id, error_json, created_at,
    updated_at, completed_at
)
SELECT
    'analysis-dispatch-legacy-' || ar.analysis_run_id,
    ar.run_id,
    ar.paper_id,
    COALESCE(
        CASE WHEN json_valid(ar.invocation_metadata_json)
             THEN json_extract(ar.invocation_metadata_json, '$.input_policy_facts.artifact_hash') END,
        ar.input_hash
    ),
    ar.artifact_id,
    ar.input_scope,
    pr.config_hash,
    ar.implementation_version,
    'stage4_analysis_luna',
    'gpt-5.6-luna',
    ar.prompt_hash,
    ar.schema_hash,
    COALESCE(
        CASE WHEN json_valid(ar.invocation_metadata_json)
             THEN json_extract(ar.invocation_metadata_json, '$.processing_decision.policy_version') END,
        ar.policy_version,
        'unavailable'
    ),
    COALESCE(
        CASE WHEN json_valid(ar.invocation_metadata_json)
             THEN json_extract(ar.invocation_metadata_json, '$.processing_decision.policy_hash') END,
        'legacy-unavailable'
    ),
    ar.created_at,
    ar.input_hash,
    CASE WHEN json_valid(ar.invocation_metadata_json)
         THEN json_extract(ar.invocation_metadata_json, '$.invocation.rendered_prompt_hash') END,
    CASE WHEN json_valid(ar.invocation_metadata_json)
         THEN json_extract(ar.invocation_metadata_json, '$.processing_decision') END,
    ar.authorization_grant_id,
    'failed_terminal',
    1,
    CASE WHEN json_valid(ar.invocation_metadata_json)
         THEN json_extract(ar.invocation_metadata_json, '$.invocation.invocation_id') END,
    CASE WHEN json_valid(ar.invocation_metadata_json)
         THEN json_extract(ar.invocation_metadata_json, '$.invocation') END,
    ar.analysis_run_id,
    json_object(
        'error', 'UncertainDispatch',
        'message', 'pre-migration analysis state may follow a remote invocation; outcome is uncertain',
        'reason', 'pre_migration_' || ar.status
    ),
    ar.created_at,
    CURRENT_TIMESTAMP,
    COALESCE(ar.completed_at, CURRENT_TIMESTAMP)
FROM analysis_runs AS ar
JOIN pipeline_runs AS pr ON pr.run_id = ar.run_id AND pr.stage = 'stage4'
WHERE ar.status IN ('failed', 'running')
  AND NOT EXISTS (
      SELECT 1
      FROM analysis_runs AS complete
      WHERE complete.run_id = ar.run_id
        AND complete.paper_id = ar.paper_id
        AND complete.status = 'complete'
  )
  AND NOT EXISTS (
      SELECT 1
      FROM analysis_runs AS newer
      WHERE newer.run_id = ar.run_id
        AND newer.paper_id = ar.paper_id
        AND newer.status IN ('failed', 'running')
        AND (
            newer.created_at > ar.created_at
            OR (newer.created_at = ar.created_at AND newer.analysis_run_id > ar.analysis_run_id)
        )
  );

UPDATE pipeline_runs
SET status = 'failed',
    completed_at = COALESCE(completed_at, CURRENT_TIMESTAMP)
WHERE run_id IN (
    SELECT DISTINCT run_id
    FROM analysis_dispatches
    WHERE status = 'failed_terminal'
      AND dispatch_id LIKE 'analysis-dispatch-legacy-%'
);
