CREATE TABLE schema_migrations (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    applied_by TEXT NOT NULL
);

CREATE TABLE pipeline_runs (
    run_id TEXT PRIMARY KEY,
    stage TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN (
        'draft', 'approved', 'running', 'complete', 'incomplete', 'failed', 'cancelled'
    )),
    input_hash TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    implementation_version TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE papers (
    paper_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    abstract TEXT,
    authors_json TEXT NOT NULL DEFAULT '[]',
    keywords_json TEXT NOT NULL DEFAULT '[]',
    publication_date TEXT,
    year INTEGER,
    venue_id TEXT,
    venue_name TEXT,
    venue_type TEXT CHECK (venue_type IN ('conference', 'journal', 'preprint', 'other')),
    doi TEXT UNIQUE,
    arxiv_id TEXT UNIQUE,
    canonical_url TEXT,
    volume TEXT,
    issue TEXT,
    pages TEXT,
    verification_status TEXT NOT NULL DEFAULT 'unverified' CHECK (verification_status IN (
        'verified', 'single_source', 'unverified', 'conflicted'
    )),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE paper_sources (
    source_id TEXT PRIMARY KEY,
    paper_id TEXT NOT NULL REFERENCES papers(paper_id),
    provider TEXT NOT NULL,
    external_id TEXT NOT NULL,
    landing_url TEXT,
    pdf_url TEXT,
    metadata_url TEXT,
    bibtex TEXT,
    citation_count INTEGER,
    citation_count_as_of TEXT,
    publication_version TEXT,
    license TEXT,
    host_type TEXT,
    access_basis TEXT NOT NULL DEFAULT 'unknown' CHECK (access_basis IN (
        'open_license', 'public_read_only', 'user_subscription', 'user_supplied', 'unknown'
    )),
    raw_metadata_json TEXT NOT NULL,
    metadata_capabilities_json TEXT NOT NULL DEFAULT '[]',
    download_capabilities_json TEXT NOT NULL DEFAULT '[]',
    first_seen_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_seen_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    source_updated_at TEXT,
    UNIQUE(provider, external_id)
);

CREATE TABLE paper_field_provenance (
    provenance_id TEXT PRIMARY KEY,
    paper_id TEXT NOT NULL REFERENCES papers(paper_id),
    source_id TEXT NOT NULL REFERENCES paper_sources(source_id),
    field_name TEXT NOT NULL,
    field_value_json TEXT NOT NULL,
    observed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(paper_id, source_id, field_name)
);

CREATE TABLE citation_counts (
    citation_count_id TEXT PRIMARY KEY,
    paper_id TEXT NOT NULL REFERENCES papers(paper_id),
    provider TEXT NOT NULL,
    count INTEGER NOT NULL CHECK (count >= 0),
    observed_at TEXT NOT NULL,
    UNIQUE(paper_id, provider, observed_at)
);

CREATE TABLE collections (
    collection_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    collection_type TEXT NOT NULL CHECK (collection_type IN ('conference', 'journal', 'arxiv', 'seed_set', 'other')),
    venue_id TEXT,
    descriptor_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE paper_collections (
    paper_id TEXT NOT NULL REFERENCES papers(paper_id),
    collection_id TEXT NOT NULL REFERENCES collections(collection_id),
    membership_status TEXT NOT NULL CHECK (membership_status IN (
        'official_confirmed', 'venue_candidate', 'not_member', 'conflicted'
    )),
    official_evidence_json TEXT,
    observed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (paper_id, collection_id)
);

CREATE TABLE artifacts (
    artifact_id TEXT PRIMARY KEY,
    paper_id TEXT REFERENCES papers(paper_id),
    artifact_kind TEXT NOT NULL CHECK (artifact_kind IN (
        'pdf', 'supplement', 'text', 'analysis', 'report', 'manifest', 'other'
    )),
    relative_path TEXT NOT NULL,
    mime_type TEXT NOT NULL,
    byte_size INTEGER NOT NULL CHECK (byte_size >= 0),
    sha256 TEXT NOT NULL,
    source_url TEXT,
    provenance_json TEXT NOT NULL DEFAULT '{}',
    processing_status TEXT NOT NULL DEFAULT 'available' CHECK (processing_status IN (
        'available', 'invalid', 'superseded', 'deleted'
    )),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(sha256),
    UNIQUE(relative_path)
);

CREATE TABLE search_plans (
    search_plan_id TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL UNIQUE,
    schema_version TEXT NOT NULL,
    plan_json TEXT NOT NULL,
    approval_json TEXT,
    status TEXT NOT NULL CHECK (status IN ('draft', 'approved', 'rejected', 'superseded')),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE crawl_runs (
    crawl_run_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL UNIQUE REFERENCES pipeline_runs(run_id),
    search_plan_id TEXT REFERENCES search_plans(search_plan_id),
    window_json TEXT NOT NULL DEFAULT '{}',
    cursor_json TEXT NOT NULL DEFAULT '{}',
    error_json TEXT,
    stats_json TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL CHECK (status IN ('running', 'complete', 'incomplete', 'failed')),
    started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TEXT
);

CREATE TABLE source_runs (
    source_run_id TEXT PRIMARY KEY,
    crawl_run_id TEXT NOT NULL REFERENCES crawl_runs(crawl_run_id),
    provider_registration_id TEXT REFERENCES provider_registrations(provider_registration_id),
    provider TEXT NOT NULL,
    provider_version TEXT NOT NULL,
    role TEXT NOT NULL,
    cursor_json TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'complete', 'incomplete', 'failed')),
    error_json TEXT,
    raw_response_artifact_id TEXT REFERENCES artifacts(artifact_id),
    started_at TEXT,
    completed_at TEXT,
    UNIQUE(crawl_run_id, provider, role)
);

CREATE TABLE search_queries (
    query_id TEXT PRIMARY KEY,
    search_plan_id TEXT NOT NULL REFERENCES search_plans(search_plan_id),
    source_run_id TEXT NOT NULL REFERENCES source_runs(source_run_id),
    provider TEXT NOT NULL,
    provider_version TEXT NOT NULL,
    query_compiler_version TEXT NOT NULL,
    role TEXT NOT NULL,
    query_text TEXT NOT NULL,
    provider_params_json TEXT NOT NULL DEFAULT '{}',
    alias_group TEXT,
    filters_json TEXT NOT NULL DEFAULT '{}',
    page TEXT,
    cursor TEXT,
    requested_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TEXT,
    query_hash TEXT NOT NULL,
    response_hash TEXT,
    returned_count INTEGER NOT NULL DEFAULT 0 CHECK (returned_count >= 0),
    status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'complete', 'failed')),
    error_json TEXT,
    UNIQUE(source_run_id, query_hash, page, cursor)
);

CREATE TABLE citation_edges (
    citation_edge_id TEXT PRIMARY KEY,
    source_paper_id TEXT NOT NULL REFERENCES papers(paper_id),
    target_paper_id TEXT NOT NULL REFERENCES papers(paper_id),
    edge_type TEXT NOT NULL CHECK (edge_type IN ('references', 'citations', 'version_of')),
    provider TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    raw_evidence_json TEXT NOT NULL,
    UNIQUE(source_paper_id, target_paper_id, edge_type, provider, observed_at)
);

CREATE TABLE screening_events (
    screening_event_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES pipeline_runs(run_id),
    paper_id TEXT NOT NULL REFERENCES papers(paper_id),
    criterion_id TEXT NOT NULL,
    decision TEXT NOT NULL CHECK (decision IN ('included', 'excluded', 'needs_review')),
    reason_code TEXT NOT NULL,
    input_hash TEXT NOT NULL,
    implementation_version TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(run_id, paper_id, criterion_id)
);

CREATE TABLE filter_decisions (
    filter_decision_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES pipeline_runs(run_id),
    paper_id TEXT NOT NULL REFERENCES papers(paper_id),
    status TEXT NOT NULL CHECK (status IN ('relevant', 'irrelevant', 'needs_review')),
    score REAL,
    threshold_version TEXT NOT NULL,
    reason TEXT NOT NULL,
    input_hash TEXT NOT NULL,
    implementation_version TEXT NOT NULL,
    model_id TEXT,
    model_revision TEXT,
    prompt_hash TEXT,
    schema_hash TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(run_id, paper_id)
);

CREATE TABLE provider_registrations (
    provider_registration_id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    distribution_name TEXT NOT NULL,
    distribution_version TEXT NOT NULL,
    entry_point TEXT NOT NULL,
    manifest_json TEXT NOT NULL,
    artifact_digest TEXT NOT NULL,
    audit_json TEXT NOT NULL DEFAULT '{}',
    trust_status TEXT NOT NULL CHECK (trust_status IN ('pending', 'trusted', 'disabled', 'revoked')),
    registered_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    disabled_at TEXT,
    UNIQUE(distribution_name, distribution_version, entry_point, artifact_digest)
);

CREATE TABLE download_candidates (
    candidate_id TEXT PRIMARY KEY,
    paper_id TEXT NOT NULL REFERENCES papers(paper_id),
    resolver TEXT NOT NULL,
    url TEXT NOT NULL,
    landing_url TEXT,
    publication_version TEXT,
    host TEXT NOT NULL,
    license TEXT,
    access_basis TEXT NOT NULL CHECK (access_basis IN (
        'open_license', 'public_read_only', 'user_subscription', 'user_supplied', 'unknown'
    )),
    retrieved_at TEXT NOT NULL,
    raw_evidence_hash TEXT,
    provenance_json TEXT NOT NULL,
    policy_version TEXT,
    policy_purpose TEXT,
    policy_decision TEXT CHECK (policy_decision IN ('allow', 'needs_grant', 'manual', 'deny')),
    policy_reason_code TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(paper_id, resolver, url, publication_version)
);

CREATE TABLE authorization_grants (
    grant_id TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL UNIQUE,
    approval_json TEXT NOT NULL,
    grant_kind TEXT NOT NULL CHECK (grant_kind IN (
        'download', 'browser_data_sharing', 'remote_model_processing'
    )),
    actions_json TEXT NOT NULL,
    purpose TEXT NOT NULL,
    mode TEXT NOT NULL CHECK (mode IN ('attended', 'unattended')),
    scope_json TEXT NOT NULL,
    selection_snapshot_hash TEXT,
    max_papers INTEGER CHECK (max_papers IS NULL OR max_papers > 0),
    artifact_hash TEXT,
    lineage_hash TEXT,
    provider TEXT,
    model_id TEXT,
    skill_digest TEXT,
    dependency_digest TEXT,
    expires_at TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE authorization_grant_events (
    grant_event_id TEXT PRIMARY KEY,
    grant_id TEXT NOT NULL REFERENCES authorization_grants(grant_id),
    event_type TEXT NOT NULL CHECK (event_type IN ('approved', 'revoked')),
    actor TEXT NOT NULL,
    event_at TEXT NOT NULL,
    event_json TEXT NOT NULL,
    UNIQUE(grant_id, event_type)
);

CREATE TABLE fetch_requests (
    request_id TEXT PRIMARY KEY,
    candidate_id TEXT NOT NULL REFERENCES download_candidates(candidate_id),
    policy_version TEXT NOT NULL,
    policy_hash TEXT NOT NULL,
    purpose TEXT NOT NULL,
    provider TEXT NOT NULL,
    authorization_grant_id TEXT REFERENCES authorization_grants(grant_id),
    authorization_grant_hash TEXT,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    fencing_token INTEGER NOT NULL CHECK (fencing_token >= 0),
    status TEXT NOT NULL CHECK (status IN ('ready', 'consumed', 'expired', 'revoked'))
);

CREATE TABLE download_attempts (
    download_attempt_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES pipeline_runs(run_id),
    candidate_id TEXT NOT NULL REFERENCES download_candidates(candidate_id),
    provider TEXT NOT NULL,
    authorization_grant_id TEXT REFERENCES authorization_grants(grant_id),
    fetch_request_id TEXT NOT NULL REFERENCES fetch_requests(request_id),
    result_status TEXT NOT NULL CHECK (result_status IN (
        'pending', 'downloaded', 'failed_retryable', 'failed_terminal', 'manual_required'
    )),
    failure_category TEXT,
    http_status INTEGER,
    browser_result_json TEXT,
    artifact_id TEXT REFERENCES artifacts(artifact_id),
    attempted_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(run_id, fetch_request_id, provider)
);

CREATE TABLE analysis_runs (
    analysis_run_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES pipeline_runs(run_id),
    paper_id TEXT NOT NULL REFERENCES papers(paper_id),
    artifact_id TEXT REFERENCES artifacts(artifact_id),
    input_hash TEXT NOT NULL,
    input_scope TEXT NOT NULL CHECK (input_scope IN ('full_pdf', 'abstract_only', 'metadata_only')),
    model_id TEXT NOT NULL,
    model_revision TEXT NOT NULL,
    prompt_hash TEXT NOT NULL,
    schema_hash TEXT NOT NULL,
    implementation_version TEXT NOT NULL,
    authorization_grant_id TEXT REFERENCES authorization_grants(grant_id),
    policy_version TEXT NOT NULL,
    policy_decision TEXT NOT NULL,
    invocation_metadata_json TEXT,
    status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'complete', 'incomplete', 'failed')),
    output_artifact_id TEXT REFERENCES artifacts(artifact_id),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TEXT,
    UNIQUE(run_id, paper_id, input_hash)
);

CREATE TABLE report_plans (
    report_plan_id TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL UNIQUE,
    schema_version TEXT NOT NULL,
    plan_json TEXT NOT NULL,
    approval_json TEXT,
    status TEXT NOT NULL CHECK (status IN ('draft', 'approved', 'rejected', 'superseded')),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE report_runs (
    report_run_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL UNIQUE REFERENCES pipeline_runs(run_id),
    report_plan_id TEXT NOT NULL REFERENCES report_plans(report_plan_id),
    corpus_snapshot_hash TEXT NOT NULL,
    aggregation_tree_json TEXT NOT NULL,
    model_id TEXT NOT NULL,
    model_revision TEXT NOT NULL,
    prompt_hash TEXT NOT NULL,
    schema_hash TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'complete', 'incomplete', 'failed')),
    output_relative_path TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TEXT
);

CREATE TABLE comparison_groups (
    comparison_group_id TEXT PRIMARY KEY,
    comparison_key TEXT NOT NULL UNIQUE,
    task_json TEXT NOT NULL,
    dataset_json TEXT NOT NULL,
    metric_json TEXT NOT NULL,
    protocol_json TEXT NOT NULL,
    baseline_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE report_claims (
    report_claim_id TEXT PRIMARY KEY,
    report_run_id TEXT NOT NULL REFERENCES report_runs(report_run_id),
    claim_id TEXT NOT NULL,
    claim_key TEXT NOT NULL,
    research_question_id TEXT NOT NULL,
    report_section TEXT NOT NULL,
    claim_text TEXT NOT NULL,
    claim_type TEXT NOT NULL CHECK (claim_type IN ('finding', 'trend', 'comparison', 'gap', 'recommendation', 'corpus_stat')),
    comparison_group_id TEXT REFERENCES comparison_groups(comparison_group_id),
    confidence REAL,
    known_limitations TEXT,
    status TEXT NOT NULL CHECK (status IN ('supported', 'mixed', 'insufficient')),
    UNIQUE(report_run_id, claim_id),
    UNIQUE(report_run_id, claim_key)
);

CREATE TABLE claim_evidence (
    claim_evidence_id TEXT PRIMARY KEY,
    report_claim_id TEXT NOT NULL REFERENCES report_claims(report_claim_id),
    evidence_kind TEXT NOT NULL CHECK (evidence_kind IN ('paper_evidence', 'corpus_evidence')),
    paper_id TEXT REFERENCES papers(paper_id),
    analysis_run_id TEXT REFERENCES analysis_runs(analysis_run_id),
    search_plan_id TEXT REFERENCES search_plans(search_plan_id),
    source_run_id TEXT REFERENCES source_runs(source_run_id),
    query_id TEXT REFERENCES search_queries(query_id),
    locator_json TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    direction TEXT NOT NULL CHECK (direction IN ('support', 'contradict', 'neutral')),
    evidence_level TEXT NOT NULL CHECK (evidence_level IN (
        'full_text_direct', 'full_text_inferred', 'abstract_direct', 'metadata_only', 'corpus_stat'
    )),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE claim_relations (
    claim_relation_id TEXT PRIMARY KEY,
    previous_report_claim_id TEXT NOT NULL REFERENCES report_claims(report_claim_id),
    current_report_claim_id TEXT NOT NULL REFERENCES report_claims(report_claim_id),
    relation_type TEXT NOT NULL CHECK (relation_type IN (
        'same', 'refined', 'split', 'merged', 'superseded', 'retired'
    )),
    reason TEXT NOT NULL,
    evidence_diff_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(previous_report_claim_id, current_report_claim_id, relation_type)
);

CREATE TABLE manual_queue (
    manual_queue_id TEXT PRIMARY KEY,
    queue_type TEXT NOT NULL CHECK (queue_type IN ('dedup', 'download', 'screening', 'merge_conflict', 'other')),
    dedup_key TEXT NOT NULL,
    paper_id TEXT REFERENCES papers(paper_id),
    run_id TEXT REFERENCES pipeline_runs(run_id),
    reason_json TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'resolved', 'dismissed')),
    resolution_json TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    resolved_at TEXT,
    UNIQUE(queue_type, dedup_key)
);

CREATE TABLE task_leases (
    task_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES pipeline_runs(run_id),
    stage TEXT NOT NULL,
    paper_id TEXT REFERENCES papers(paper_id),
    output_kind TEXT NOT NULL,
    input_hash TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN (
        'pending', 'running', 'complete', 'failed_retryable', 'failed_terminal', 'manual_required'
    )),
    worker_id TEXT,
    lease_expires_at TEXT,
    attempt INTEGER NOT NULL DEFAULT 0 CHECK (attempt >= 0),
    fencing_token INTEGER NOT NULL DEFAULT 0 CHECK (fencing_token >= 0),
    error_json TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(run_id, stage, paper_id, output_kind)
);

CREATE TABLE shard_manifests (
    shard_manifest_id TEXT PRIMARY KEY,
    shard_id TEXT NOT NULL,
    run_id TEXT NOT NULL REFERENCES pipeline_runs(run_id),
    epoch INTEGER NOT NULL CHECK (epoch >= 0),
    fencing_token INTEGER NOT NULL CHECK (fencing_token >= 0),
    paper_ids_json TEXT NOT NULL,
    input_artifact_hash TEXT,
    config_hash TEXT NOT NULL,
    model_profile_json TEXT NOT NULL,
    output_root TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('issued', 'running', 'complete', 'superseded')),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(run_id, shard_id, epoch)
);

CREATE TABLE shard_results (
    shard_result_id TEXT PRIMARY KEY,
    shard_manifest_id TEXT NOT NULL REFERENCES shard_manifests(shard_manifest_id),
    epoch INTEGER NOT NULL CHECK (epoch >= 0),
    result_manifest_json TEXT NOT NULL,
    received_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    status TEXT NOT NULL CHECK (status IN ('pending', 'accepted', 'rejected', 'conflict')),
    UNIQUE(shard_manifest_id, epoch)
);

CREATE TABLE run_outputs (
    run_output_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES pipeline_runs(run_id),
    stage TEXT NOT NULL,
    paper_id TEXT NOT NULL REFERENCES papers(paper_id),
    output_kind TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    artifact_id TEXT REFERENCES artifacts(artifact_id),
    shard_manifest_id TEXT REFERENCES shard_manifests(shard_manifest_id),
    shard_epoch INTEGER,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(run_id, stage, paper_id, output_kind)
);

CREATE INDEX idx_paper_sources_paper_id ON paper_sources(paper_id);
CREATE INDEX idx_artifacts_paper_id ON artifacts(paper_id);
CREATE INDEX idx_citation_edges_target ON citation_edges(target_paper_id);
CREATE INDEX idx_task_leases_claim ON task_leases(status, lease_expires_at);
CREATE INDEX idx_run_outputs_run ON run_outputs(run_id, stage);
