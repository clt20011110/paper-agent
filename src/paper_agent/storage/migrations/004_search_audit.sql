ALTER TABLE source_runs ADD COLUMN raw_response_hash TEXT;

CREATE TABLE source_run_audits (
    source_run_id TEXT PRIMARY KEY REFERENCES source_runs(source_run_id),
    raw_discovered INTEGER NOT NULL DEFAULT 0 CHECK (raw_discovered >= 0),
    unique_after_dedup INTEGER NOT NULL DEFAULT 0 CHECK (unique_after_dedup >= 0),
    overlap INTEGER NOT NULL DEFAULT 0 CHECK (overlap >= 0),
    screened INTEGER NOT NULL DEFAULT 0 CHECK (screened >= 0),
    excluded INTEGER NOT NULL DEFAULT 0 CHECK (excluded >= 0),
    included INTEGER NOT NULL DEFAULT 0 CHECK (included >= 0),
    full_text_available INTEGER NOT NULL DEFAULT 0 CHECK (full_text_available >= 0),
    error_count INTEGER NOT NULL DEFAULT 0 CHECK (error_count >= 0),
    updated_at TEXT NOT NULL
);

CREATE TABLE provider_watermarks (
    provider TEXT NOT NULL,
    descriptor_key TEXT NOT NULL,
    watermark_json TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (provider, descriptor_key)
);

CREATE TABLE incremental_diffs (
    crawl_run_id TEXT PRIMARY KEY REFERENCES crawl_runs(crawl_run_id),
    new_count INTEGER NOT NULL DEFAULT 0 CHECK (new_count >= 0),
    removed_count INTEGER NOT NULL DEFAULT 0 CHECK (removed_count >= 0),
    retracted_count INTEGER NOT NULL DEFAULT 0 CHECK (retracted_count >= 0),
    metadata_changed_count INTEGER NOT NULL DEFAULT 0 CHECK (metadata_changed_count >= 0),
    preprint_replaced_count INTEGER NOT NULL DEFAULT 0 CHECK (preprint_replaced_count >= 0),
    recorded_at TEXT NOT NULL
);

CREATE TABLE incremental_diff_papers (
    crawl_run_id TEXT NOT NULL REFERENCES crawl_runs(crawl_run_id),
    paper_id TEXT NOT NULL REFERENCES papers(paper_id),
    change_kind TEXT NOT NULL CHECK (change_kind IN (
        'new', 'removed', 'retracted', 'metadata_changed', 'preprint_replaced'
    )),
    PRIMARY KEY (crawl_run_id, paper_id, change_kind)
);

CREATE INDEX idx_incremental_diff_papers_kind ON incremental_diff_papers(crawl_run_id, change_kind);
