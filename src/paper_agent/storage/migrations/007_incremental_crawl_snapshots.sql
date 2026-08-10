CREATE TABLE crawl_paper_snapshots (
    crawl_run_id TEXT NOT NULL REFERENCES crawl_runs(crawl_run_id),
    paper_id TEXT NOT NULL REFERENCES papers(paper_id),
    metadata_hash TEXT NOT NULL,
    status_version_json TEXT NOT NULL,
    PRIMARY KEY (crawl_run_id, paper_id)
);

CREATE TABLE crawl_paper_snapshot_sources (
    crawl_run_id TEXT NOT NULL REFERENCES crawl_runs(crawl_run_id),
    paper_id TEXT NOT NULL REFERENCES papers(paper_id),
    provider TEXT NOT NULL,
    descriptor_key TEXT NOT NULL,
    PRIMARY KEY (crawl_run_id, paper_id, provider, descriptor_key)
);

CREATE TABLE crawl_scope_statuses (
    crawl_run_id TEXT NOT NULL REFERENCES crawl_runs(crawl_run_id),
    provider TEXT NOT NULL,
    descriptor_key TEXT NOT NULL,
    cursor TEXT,
    complete INTEGER NOT NULL CHECK (complete IN (0, 1)),
    PRIMARY KEY (crawl_run_id, provider, descriptor_key)
);

CREATE INDEX idx_crawl_paper_snapshots_paper ON crawl_paper_snapshots(paper_id);
CREATE INDEX idx_crawl_paper_snapshot_sources_scope
    ON crawl_paper_snapshot_sources(crawl_run_id, provider, descriptor_key);
