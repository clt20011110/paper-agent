CREATE TABLE search_rounds (
    search_round_id TEXT PRIMARY KEY,
    crawl_run_id TEXT NOT NULL REFERENCES crawl_runs(crawl_run_id),
    round_index INTEGER NOT NULL CHECK (round_index >= 0),
    state TEXT NOT NULL CHECK (state IN (
        'planned', 'discovering', 'normalizing', 'screening', 'auditing', 'complete', 'stopped'
    )),
    seed_manifest_hash TEXT NOT NULL,
    request_schedule_hash TEXT NOT NULL,
    stop_reason TEXT CHECK (stop_reason IN (
        'sources_exhausted', 'saturated', 'saturated_with_unresolved', 'budget_exhausted', 'max_depth', 'max_rounds'
    )),
    limited_scope INTEGER NOT NULL DEFAULT 0 CHECK (limited_scope IN (0, 1)),
    stats_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TEXT,
    UNIQUE(crawl_run_id, round_index)
);

CREATE TABLE search_round_seeds (
    search_round_id TEXT NOT NULL REFERENCES search_rounds(search_round_id),
    paper_id TEXT NOT NULL REFERENCES papers(paper_id),
    seed_reason TEXT NOT NULL CHECK (seed_reason IN ('user_seed', 'relevant_topk')),
    parent_round INTEGER NOT NULL CHECK (parent_round >= 0),
    depth INTEGER NOT NULL CHECK (depth >= 0),
    subquestion_id TEXT,
    seed_rank INTEGER NOT NULL CHECK (seed_rank >= 0),
    selector_version TEXT NOT NULL,
    selector_config_hash TEXT NOT NULL,
    PRIMARY KEY (search_round_id, paper_id, seed_reason, subquestion_id)
);

CREATE TABLE citation_requests (
    citation_request_id TEXT PRIMARY KEY,
    search_round_id TEXT NOT NULL REFERENCES search_rounds(search_round_id),
    provider TEXT NOT NULL,
    direction TEXT NOT NULL CHECK (direction IN ('references', 'citations')),
    seed_paper_id TEXT NOT NULL REFERENCES papers(paper_id),
    depth INTEGER NOT NULL CHECK (depth >= 1),
    seed_rank INTEGER NOT NULL CHECK (seed_rank >= 0),
    schedule_order INTEGER NOT NULL CHECK (schedule_order >= 0),
    max_candidates INTEGER NOT NULL CHECK (max_candidates > 0),
    status TEXT NOT NULL CHECK (status IN ('planned', 'running', 'complete', 'failed', 'skipped_budget')),
    error_json TEXT,
    UNIQUE(search_round_id, provider, direction, seed_paper_id),
    UNIQUE(search_round_id, schedule_order)
);

CREATE TABLE search_round_papers (
    search_round_id TEXT NOT NULL REFERENCES search_rounds(search_round_id),
    paper_id TEXT NOT NULL REFERENCES papers(paper_id),
    depth INTEGER NOT NULL CHECK (depth >= 0),
    first_seen INTEGER NOT NULL CHECK (first_seen IN (0, 1)),
    screening_status TEXT CHECK (screening_status IN ('relevant', 'irrelevant', 'needs_review')),
    subquestion_id TEXT,
    PRIMARY KEY (search_round_id, paper_id)
);

CREATE TABLE search_round_audits (
    search_round_id TEXT PRIMARY KEY REFERENCES search_rounds(search_round_id),
    raw_discovered INTEGER NOT NULL CHECK (raw_discovered >= 0),
    unique_after_dedup INTEGER NOT NULL CHECK (unique_after_dedup >= 0),
    overlap INTEGER NOT NULL CHECK (overlap >= 0),
    screened_unique INTEGER NOT NULL CHECK (screened_unique >= 0),
    new_included_unique INTEGER NOT NULL CHECK (new_included_unique >= 0),
    needs_review INTEGER NOT NULL CHECK (needs_review >= 0),
    error_count INTEGER NOT NULL CHECK (error_count >= 0),
    edge_counts_json TEXT NOT NULL,
    source_stats_json TEXT NOT NULL,
    audited_at TEXT NOT NULL
);

CREATE INDEX idx_search_rounds_campaign ON search_rounds(crawl_run_id, round_index);
CREATE INDEX idx_citation_requests_schedule ON citation_requests(search_round_id, schedule_order);
CREATE INDEX idx_search_round_papers_depth ON search_round_papers(search_round_id, depth);
