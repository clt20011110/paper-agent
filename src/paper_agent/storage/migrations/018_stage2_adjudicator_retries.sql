ALTER TABLE filter_decisions
ADD COLUMN adjudicator_attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (adjudicator_attempt_count >= 0);

ALTER TABLE filter_decisions
ADD COLUMN adjudicator_retry_reason TEXT;

ALTER TABLE filter_decisions
ADD COLUMN adjudicator_retry_outcome TEXT CHECK (
    adjudicator_retry_outcome IN ('succeeded', 'failed')
);
