CREATE TABLE text_extractions (
    extraction_id TEXT PRIMARY KEY,
    paper_id TEXT NOT NULL REFERENCES papers(paper_id),
    source_artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id),
    source_sha256 TEXT NOT NULL,
    output_artifact_id TEXT REFERENCES artifacts(artifact_id),
    extractor_name TEXT NOT NULL,
    extractor_version TEXT NOT NULL,
    page_count INTEGER NOT NULL CHECK (page_count >= 0),
    character_count INTEGER NOT NULL CHECK (character_count >= 0),
    text_coverage REAL NOT NULL CHECK (text_coverage >= 0 AND text_coverage <= 1),
    printable_ratio REAL NOT NULL CHECK (printable_ratio >= 0 AND printable_ratio <= 1),
    status TEXT NOT NULL CHECK (status IN ('full_text_ready', 'needs_ocr', 'extraction_failed')),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_artifact_id, extractor_name, extractor_version)
);
