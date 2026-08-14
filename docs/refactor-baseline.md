# Refactor baseline

- Baseline commit SHA: `9c7f03c57276cf8f544d3a215425fcc4c50d3173`
- Original branch: `feature/crawler-adapters`
- New branch: `refactor/stage1-minimal`
- Backup tag: `backup/stage1-before-minimal-20260814`
- Python version: `Python 3.13.3`
- Current pytest command: `pytest` (unavailable: `command not found`)
- Current git status (before baseline setup):

  ```text
  ## feature/crawler-adapters...origin/feature/crawler-adapters
   M src/paper_agent/stage1.py
   M src/paper_agent/stage1_matrix.py
   M tests/test_stage1.py
   M tests/test_stage1_matrix.py
  ```

- Refactor target: Stage 1-only metadata collector.
- Runtime behavior: This stage does not change any runtime behavior.
