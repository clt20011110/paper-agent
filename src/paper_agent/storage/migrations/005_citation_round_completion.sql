ALTER TABLE search_round_audits ADD COLUMN screening_complete INTEGER NOT NULL DEFAULT 1 CHECK (screening_complete IN (0, 1));
ALTER TABLE search_round_audits ADD COLUMN source_failed INTEGER NOT NULL DEFAULT 0 CHECK (source_failed IN (0, 1));
