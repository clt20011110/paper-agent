-- Preserve the unused pre-materialization shapes for explicit inspection.
ALTER TABLE claim_relations RENAME TO claim_relations_legacy_v1;
ALTER TABLE claim_evidence RENAME TO claim_evidence_legacy_v1;
ALTER TABLE report_claims RENAME TO report_claims_legacy_v1;
ALTER TABLE comparison_groups RENAME TO comparison_groups_legacy_v1;

CREATE UNIQUE INDEX idx_analysis_runs_evidence_binding
ON analysis_runs(analysis_run_id, paper_id);

CREATE UNIQUE INDEX idx_search_queries_evidence_binding
ON search_queries(query_id, search_plan_id, source_run_id);

CREATE TABLE report_fact_sets (
    report_run_id TEXT PRIMARY KEY REFERENCES report_runs(report_run_id),
    report_document_hash TEXT NOT NULL,
    deterministic_verification_hash TEXT NOT NULL,
    facts_hash TEXT NOT NULL,
    claim_count INTEGER NOT NULL CHECK (claim_count >= 0),
    evidence_count INTEGER NOT NULL CHECK (evidence_count >= 0),
    comparison_group_count INTEGER NOT NULL CHECK (comparison_group_count >= 0),
    claim_relation_count INTEGER NOT NULL CHECK (claim_relation_count >= 0),
    sealed INTEGER NOT NULL DEFAULT 0 CHECK (sealed IN (0, 1)),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (report_run_id) REFERENCES report_audit_runs(report_run_id),
    CHECK (length(report_document_hash) = 64 AND report_document_hash NOT GLOB '*[^0-9a-f]*'),
    CHECK (length(deterministic_verification_hash) = 64 AND deterministic_verification_hash NOT GLOB '*[^0-9a-f]*'),
    CHECK (length(facts_hash) = 64 AND facts_hash NOT GLOB '*[^0-9a-f]*')
);

CREATE TABLE comparison_groups (
    comparison_group_id TEXT PRIMARY KEY,
    comparison_key_hash TEXT NOT NULL UNIQUE,
    comparison_key_json TEXT NOT NULL UNIQUE CHECK (json_valid(comparison_key_json)),
    task_id TEXT NOT NULL,
    dataset_id TEXT NOT NULL,
    dataset_version TEXT NOT NULL,
    split_id TEXT NOT NULL,
    metric_id TEXT NOT NULL,
    metric_definition_hash TEXT NOT NULL,
    unit TEXT NOT NULL,
    optimization_direction TEXT NOT NULL CHECK (
        optimization_direction IN ('maximize', 'minimize', 'not_applicable')
    ),
    protocol_id TEXT NOT NULL,
    protocol_hash TEXT NOT NULL,
    baseline_id TEXT NOT NULL,
    baseline_version TEXT NOT NULL,
    normalization_method TEXT NOT NULL,
    normalizer_version TEXT NOT NULL,
    conditions_json TEXT NOT NULL CHECK (json_valid(conditions_json)),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (comparison_group_id, comparison_key_hash),
    CHECK (length(comparison_key_hash) = 64 AND comparison_key_hash NOT GLOB '*[^0-9a-f]*'),
    CHECK (length(metric_definition_hash) = 64 AND metric_definition_hash NOT GLOB '*[^0-9a-f]*'),
    CHECK (length(protocol_hash) = 64 AND protocol_hash NOT GLOB '*[^0-9a-f]*')
);

CREATE TABLE report_comparison_groups (
    report_run_id TEXT NOT NULL REFERENCES report_fact_sets(report_run_id),
    comparison_group_id TEXT NOT NULL,
    comparison_key_hash TEXT NOT NULL,
    PRIMARY KEY (report_run_id, comparison_group_id),
    FOREIGN KEY (comparison_group_id, comparison_key_hash)
        REFERENCES comparison_groups(comparison_group_id, comparison_key_hash)
);

CREATE TABLE report_claims (
    report_run_id TEXT NOT NULL REFERENCES report_fact_sets(report_run_id),
    claim_id TEXT NOT NULL,
    claim_hash TEXT NOT NULL,
    claim_json TEXT NOT NULL CHECK (json_valid(claim_json)),
    claim_key_hash TEXT NOT NULL,
    claim_key_json TEXT NOT NULL CHECK (json_valid(claim_key_json)),
    subject_id TEXT NOT NULL,
    predicate_id TEXT NOT NULL,
    object_or_scope_id TEXT NOT NULL,
    qualifier_context_hash TEXT NOT NULL,
    research_question_id TEXT NOT NULL,
    report_section TEXT NOT NULL,
    claim_text TEXT NOT NULL,
    claim_type TEXT NOT NULL CHECK (
        claim_type IN ('finding', 'trend', 'comparison', 'gap', 'recommendation', 'corpus_stat')
    ),
    evidence_level TEXT NOT NULL CHECK (
        evidence_level IN (
            'full_text_direct', 'full_text_inferred', 'abstract_direct',
            'metadata_only', 'corpus_stat'
        )
    ),
    comparison_group_id TEXT,
    confidence TEXT NOT NULL CHECK (confidence IN ('high', 'medium', 'low')),
    known_limitations_json TEXT NOT NULL CHECK (json_valid(known_limitations_json)),
    status TEXT NOT NULL CHECK (status IN ('supported', 'mixed', 'insufficient')),
    mapping_status TEXT NOT NULL CHECK (mapping_status IN ('mapped', 'unmapped_new')),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (report_run_id, claim_id),
    UNIQUE (report_run_id, claim_key_hash),
    FOREIGN KEY (report_run_id, comparison_group_id)
        REFERENCES report_comparison_groups(report_run_id, comparison_group_id),
    CHECK (length(claim_hash) = 64 AND claim_hash NOT GLOB '*[^0-9a-f]*'),
    CHECK (length(claim_key_hash) = 64 AND claim_key_hash NOT GLOB '*[^0-9a-f]*'),
    CHECK (length(qualifier_context_hash) = 64 AND qualifier_context_hash NOT GLOB '*[^0-9a-f]*')
);

CREATE TABLE claim_evidence (
    report_run_id TEXT NOT NULL,
    claim_id TEXT NOT NULL,
    direction TEXT NOT NULL CHECK (direction IN ('support', 'contradict')),
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    evidence_ref_hash TEXT NOT NULL,
    evidence_ref_json TEXT NOT NULL CHECK (json_valid(evidence_ref_json)),
    evidence_kind TEXT NOT NULL CHECK (evidence_kind IN ('paper_evidence', 'corpus_evidence')),
    evidence_level TEXT NOT NULL CHECK (
        evidence_level IN (
            'full_text_direct', 'full_text_inferred', 'abstract_direct',
            'metadata_only', 'corpus_stat'
        )
    ),
    paper_id TEXT,
    analysis_run_id TEXT,
    search_plan_id TEXT,
    source_run_id TEXT,
    query_id TEXT,
    locator TEXT,
    evidence_unit_json TEXT CHECK (evidence_unit_json IS NULL OR json_valid(evidence_unit_json)),
    statistic TEXT,
    calculation TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (report_run_id, claim_id, direction, ordinal),
    UNIQUE (report_run_id, claim_id, evidence_ref_hash),
    FOREIGN KEY (report_run_id, claim_id)
        REFERENCES report_claims(report_run_id, claim_id),
    FOREIGN KEY (analysis_run_id, paper_id)
        REFERENCES analysis_runs(analysis_run_id, paper_id),
    FOREIGN KEY (query_id, search_plan_id, source_run_id)
        REFERENCES search_queries(query_id, search_plan_id, source_run_id),
    CHECK (length(evidence_ref_hash) = 64 AND evidence_ref_hash NOT GLOB '*[^0-9a-f]*'),
    CHECK (
        (
            evidence_kind = 'paper_evidence'
            AND evidence_level <> 'corpus_stat'
            AND paper_id IS NOT NULL
            AND analysis_run_id IS NOT NULL
            AND locator IS NOT NULL
            AND evidence_unit_json IS NOT NULL
            AND search_plan_id IS NULL
            AND source_run_id IS NULL
            AND query_id IS NULL
            AND statistic IS NULL
            AND calculation IS NULL
        )
        OR
        (
            evidence_kind = 'corpus_evidence'
            AND evidence_level = 'corpus_stat'
            AND paper_id IS NULL
            AND analysis_run_id IS NULL
            AND locator IS NULL
            AND evidence_unit_json IS NULL
            AND search_plan_id IS NOT NULL
            AND source_run_id IS NOT NULL
            AND query_id IS NOT NULL
            AND statistic IS NOT NULL
            AND calculation IS NOT NULL
        )
    )
);

CREATE TABLE claim_relations (
    current_report_run_id TEXT NOT NULL REFERENCES report_fact_sets(report_run_id),
    previous_report_run_id TEXT NOT NULL,
    previous_claim_id TEXT NOT NULL,
    current_claim_id TEXT NOT NULL,
    relation_type TEXT NOT NULL CHECK (
        relation_type IN ('same', 'refined', 'split', 'merged', 'superseded', 'retired')
    ),
    reason TEXT NOT NULL CHECK (length(trim(reason)) > 0),
    evidence_diff_json TEXT NOT NULL CHECK (json_valid(evidence_diff_json)),
    relation_hash TEXT NOT NULL,
    relation_json TEXT NOT NULL CHECK (json_valid(relation_json)),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (
        current_report_run_id, previous_report_run_id,
        previous_claim_id, current_claim_id
    ),
    UNIQUE (current_report_run_id, relation_hash),
    FOREIGN KEY (previous_report_run_id, previous_claim_id)
        REFERENCES report_claims(report_run_id, claim_id),
    FOREIGN KEY (current_report_run_id, current_claim_id)
        REFERENCES report_claims(report_run_id, claim_id),
    CHECK (current_report_run_id <> previous_report_run_id),
    CHECK (length(relation_hash) = 64 AND relation_hash NOT GLOB '*[^0-9a-f]*')
);

CREATE INDEX idx_report_claims_section
ON report_claims(report_run_id, report_section, claim_id);

CREATE INDEX idx_claim_evidence_paper
ON claim_evidence(paper_id, report_run_id, claim_id);

CREATE INDEX idx_claim_relations_previous
ON claim_relations(previous_report_run_id, previous_claim_id, current_report_run_id);

CREATE TRIGGER report_fact_sets_seal_only
BEFORE UPDATE ON report_fact_sets
WHEN NOT (
    OLD.sealed = 0
    AND NEW.sealed = 1
    AND OLD.report_run_id = NEW.report_run_id
    AND OLD.report_document_hash = NEW.report_document_hash
    AND OLD.deterministic_verification_hash = NEW.deterministic_verification_hash
    AND OLD.facts_hash = NEW.facts_hash
    AND OLD.claim_count = NEW.claim_count
    AND OLD.evidence_count = NEW.evidence_count
    AND OLD.comparison_group_count = NEW.comparison_group_count
    AND OLD.claim_relation_count = NEW.claim_relation_count
    AND OLD.created_at = NEW.created_at
)
BEGIN
    SELECT RAISE(ABORT, 'verified report fact sets are immutable');
END;

CREATE TRIGGER report_fact_sets_no_delete
BEFORE DELETE ON report_fact_sets
BEGIN
    SELECT RAISE(ABORT, 'verified report fact sets are immutable');
END;

CREATE TRIGGER comparison_groups_no_update
BEFORE UPDATE ON comparison_groups
BEGIN
    SELECT RAISE(ABORT, 'comparison groups are immutable');
END;

CREATE TRIGGER comparison_groups_no_delete
BEFORE DELETE ON comparison_groups
BEGIN
    SELECT RAISE(ABORT, 'comparison groups are immutable');
END;

CREATE TRIGGER report_comparison_groups_sealed_insert
BEFORE INSERT ON report_comparison_groups
WHEN EXISTS (
    SELECT 1 FROM report_fact_sets
    WHERE report_run_id = NEW.report_run_id AND sealed = 1
)
BEGIN
    SELECT RAISE(ABORT, 'verified report facts are immutable');
END;

CREATE TRIGGER report_comparison_groups_no_update
BEFORE UPDATE ON report_comparison_groups
BEGIN
    SELECT RAISE(ABORT, 'verified report facts are immutable');
END;

CREATE TRIGGER report_comparison_groups_no_delete
BEFORE DELETE ON report_comparison_groups
BEGIN
    SELECT RAISE(ABORT, 'verified report facts are immutable');
END;

CREATE TRIGGER report_claims_sealed_insert
BEFORE INSERT ON report_claims
WHEN EXISTS (
    SELECT 1 FROM report_fact_sets
    WHERE report_run_id = NEW.report_run_id AND sealed = 1
)
BEGIN
    SELECT RAISE(ABORT, 'verified report facts are immutable');
END;

CREATE TRIGGER report_claims_no_update
BEFORE UPDATE ON report_claims
BEGIN
    SELECT RAISE(ABORT, 'verified report facts are immutable');
END;

CREATE TRIGGER report_claims_no_delete
BEFORE DELETE ON report_claims
BEGIN
    SELECT RAISE(ABORT, 'verified report facts are immutable');
END;

CREATE TRIGGER claim_evidence_sealed_insert
BEFORE INSERT ON claim_evidence
WHEN EXISTS (
    SELECT 1 FROM report_fact_sets
    WHERE report_run_id = NEW.report_run_id AND sealed = 1
)
BEGIN
    SELECT RAISE(ABORT, 'verified report facts are immutable');
END;

CREATE TRIGGER claim_evidence_no_update
BEFORE UPDATE ON claim_evidence
BEGIN
    SELECT RAISE(ABORT, 'verified report facts are immutable');
END;

CREATE TRIGGER claim_evidence_no_delete
BEFORE DELETE ON claim_evidence
BEGIN
    SELECT RAISE(ABORT, 'verified report facts are immutable');
END;

CREATE TRIGGER claim_relations_sealed_insert
BEFORE INSERT ON claim_relations
WHEN EXISTS (
    SELECT 1 FROM report_fact_sets
    WHERE report_run_id = NEW.current_report_run_id AND sealed = 1
)
BEGIN
    SELECT RAISE(ABORT, 'verified report facts are immutable');
END;

CREATE TRIGGER claim_relations_no_update
BEFORE UPDATE ON claim_relations
BEGIN
    SELECT RAISE(ABORT, 'verified report facts are immutable');
END;

CREATE TRIGGER claim_relations_no_delete
BEFORE DELETE ON claim_relations
BEGIN
    SELECT RAISE(ABORT, 'verified report facts are immutable');
END;
