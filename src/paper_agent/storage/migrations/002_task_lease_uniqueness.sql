CREATE UNIQUE INDEX uq_task_leases_output
    ON task_leases(run_id, stage, COALESCE(paper_id, ''), output_kind);
