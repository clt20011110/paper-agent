ALTER TABLE search_queries
ADD COLUMN response_artifact_id TEXT REFERENCES artifacts(artifact_id);

CREATE TABLE provider_response_cache (
    replay_scope TEXT NOT NULL REFERENCES crawl_runs(crawl_run_id),
    provider TEXT NOT NULL,
    query_hash TEXT NOT NULL,
    cursor_json TEXT NOT NULL,
    api_version TEXT NOT NULL,
    artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id),
    content_type TEXT NOT NULL DEFAULT '',
    etag TEXT,
    last_modified TEXT,
    recorded_at TEXT NOT NULL,
    PRIMARY KEY (replay_scope, provider, query_hash, cursor_json, api_version)
);

CREATE INDEX idx_provider_response_cache_artifact
ON provider_response_cache(artifact_id);
