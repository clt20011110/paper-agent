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

## Reproducible Stage 1 test baseline

- Test Python path: `/Users/chenletian/Documents/Codex/2026-08-09/cl/paper-agent/.venv/bin/python`
- Test Python version: `Python 3.13.12`
- `.venv` created for this task: No; `.venv/bin/python` already existed.
- Development dependencies installed for this task: No.
- Executed pytest command:

  ```text
  .venv/bin/python -m pytest -q \
    tests/test_stage1.py \
    tests/test_stage1_matrix.py \
    tests/test_builtin_providers.py
  ```

- Result-reporting rerun with the same three test files:

  ```text
  .venv/bin/python -m pytest -q -rA \
    tests/test_stage1.py \
    tests/test_stage1_matrix.py \
    tests/test_builtin_providers.py
  ```

- Test exit code: `0`
- Test results: `67 passed`, `0 failed`, `0 skipped`, `0 error`
- Failed tests: None.
- `git stash list --date=iso` result:

  ```text
  stash@{0}: codex: preserve pre-existing changes before stage1 baseline
  ```

- The stash was not applied, popped, dropped, cleared, branched, or otherwise modified.
- Completed `git status --short --branch`:

  ```text
  ## refactor/stage1-minimal...origin/refactor/stage1-minimal
  ```
