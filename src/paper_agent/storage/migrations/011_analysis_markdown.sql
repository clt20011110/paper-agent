ALTER TABLE analysis_runs ADD COLUMN markdown_artifact_id TEXT REFERENCES artifacts(artifact_id);
