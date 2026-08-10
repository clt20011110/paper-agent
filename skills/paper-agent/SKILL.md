---
name: paper-agent
description: Orchestrate the installed paper-agent CLI for auditable literature discovery, local Stage 2 screening, authorized PDF acquisition, per-paper analysis, and evidence-grounded Chinese review generation. Use when a user asks to plan, run, resume, inspect, audit, or export a paper-research workflow, or migrate an older paper-agent configuration.
---

# Paper Agent

Use the `paper-agent` console entry point as the only business implementation. Collect intent, obtain explicit approvals, invoke the CLI, and explain its structured results. Never import or run code under `.opencode`, duplicate crawler/model/database logic in this skill, or edit SQLite directly.

## Command contract

Before invoking a command, run its `--help` form from the installed version. The
supported workflow surface is:

```text
paper-agent doctor
paper-agent search plan | approve | run | expand-citations | audit
paper-agent import-seeds
paper-agent crawl
paper-agent filter
paper-agent grant create | approve | revoke
paper-agent download [--include-needs-review] [authorized handoff paths]
paper-agent analyze
paper-agent report prepare-inputs --crawl-run-id <ID> --filter-run-id <ID> --stage4-run-id <ID> --recent-cutoff <YYYY-MM-DD> --created-at <ISO-8601> --database <DB> --artifact-root <DIR> --output-root <DIR>
paper-agent report --plan-only
paper-agent report approve --plan <REPORT_PLAN.json> --hash <sha256> --approved-by <operator> --corpus-snapshot <CORPUS_SNAPSHOT.json> --search-audit <SEARCH_AUDIT.json> --output-root <DIR>
paper-agent report --plan <REPORT_PLAN.json> --database <DB> --output-root <DIR>
paper-agent report --diff-from <previous_report_run_id> --report-run-id <current_report_run_id> --output-root <DIR>
paper-agent verify-report
paper-agent run --workflow <WORKFLOW.json>
paper-agent resume --workflow <WORKFLOW.json>
paper-agent export
paper-agent migrate-config
paper-agent benchmark-stage2
```

Stage 3 的 `not_available/failed_terminal` 表示候选已得到确定的无 PDF 结论，可由 Stage 4
按实际摘要或元数据降级；`failed_retryable/auth_required/manual_required` 不计为完成。

Use `--config`, `--run-id`, and `--dry-run` where the command exposes them.
Commands emit one structured JSON result; preserve its `run_id`, status, event
code, and artifact paths in the task response. Do not emulate a missing CLI
command with Python imports, SQLite edits, shell pipelines, or a second
implementation. If a required command is unavailable or its help conflicts
with this contract, stop and report the installed version as a product gate.

## Locate and inspect the CLI

1. Prefer the active environment's `paper-agent` executable. In a source checkout, prefer `.venv/bin/paper-agent` when present.
2. Run `paper-agent --help` and the relevant subcommand `--help` before constructing commands. Do not assume options that the installed version does not expose.
3. Run `paper-agent doctor` before a new production campaign or when runtime state may have changed. Use `--production-ready` only when checking a real production run; it may correctly fail in an offline rehearsal.
4. Treat doctor failures as explicit gates. Do not invent credentials, model releases, approvals, grants, or provider access.

## Plan a campaign

Collect only decisions that affect the frozen plan:

- research question, subquestions, audience, and desired report;
- venues, date range, document types, languages, and user seeds;
- inclusion/exclusion criteria and whether arXiv-only candidates are allowed;
- search, citation-depth, Stage 2, download, Luna, and Sol budgets;
- existing database, approved plan, or run to resume.

Compile a draft with `paper-agent search plan`. Show the exact scope, resolved providers, request/candidate/time limits, Stage 2 release requirement, and expected downstream work. Require the user to approve the displayed content hash with `paper-agent search approve`; never approve a changed plan silently.

Use `--dry-run` before provider calls. A dry run is validation, not evidence that a live source, local model, remote model, or authorized publisher session will succeed.

## Execute and resume

Use the narrowest CLI command that satisfies the request. Preserve the same database and explicit `--run-id` where the CLI supports it. Prefer `paper-agent resume` for an interrupted immutable run; do not create a new run merely to bypass drift or a failed gate. `resume` must include the original `--workflow <WORKFLOW.json>` as well as its workflow run ID; it cannot recover a workflow from a bare `--run-id`. The manifest is typed JSON, not argv: every FileRef is a relative `path` plus a lowercase SHA-256, and any config/plan/release/selection/policy/report-input drift requires a new approved manifest. If a global `--config` is supplied, it must name the same file as the manifest config FileRef.

For a new typed workflow, inspect `paper-agent run --help`, then prepare the manifest using the per-stage field contract and FileRef schema in the repository README. Run `paper-agent --dry-run run --workflow <WORKFLOW.json> --workflow-run-id <ID>` before executing. A stop signal requests a checkpoint only at a stage boundary; never report the in-flight stage as cancelled before its structured result is returned.

A schema-version-2 workflow may bind Search → Filter → Download → Analyze with
`from_step` references. End that dynamic chain at Analyze. The ReportPlan requires exact
post-analysis corpus membership, so prepare report inputs, compile and approve the plan,
pin its path/hash in a new config, and run Report as a separate single-stage workflow.
Never append a guessed pre-crawl ReportPlan to the dynamic manifest.

For Stage 2, require a passed local release bundle and oMLX models. Never use a test fake, cloud fallback, unapproved model revision, or raw uncalibrated thresholds in production.

For Stage 3, exhaust public and authorized open-access providers first. Before a browser handoff, show the approved grant scope, domain allowlist, `max_papers`, expiry, and attended/unattended mode. The CLI can prepare an audited handoff only when `--authorized-skill-queue`, `--authorized-skill-output`, and at least one `--authorized-skill-root` are supplied together (with the approved `--grant-id` and enabled configuration). Read `authorized_queue_path` from the structured result, then invoke `$download-authorized-papers` only for that queue and only through the user's authorized visible browser session. The CLI does not operate the browser. Never request, inspect, copy, or log passwords, cookies, tokens, CAPTCHA contents, or session material. Stop the affected queue on login repair, CAPTCHA, 403, or 429 while allowing unrelated papers to continue.

For Stage 4 and Stage 4b, explain that `network=false` does not keep model payloads local. Require exact `remote_model_processing` policy allowance or approved artifact/model grants before sending full text or its restricted derivatives. Keep Stage 4 on `gpt-5.6-luna` and Stage 4b on `gpt-5.6-sol`; never upgrade, downgrade, or fall back silently.

## Generate a review

Run `paper-agent report --plan-only` first and show:

- frozen corpus and search-flow limitations;
- paper input scopes (`full_pdf`, `abstract_only`, `metadata_only`, or missing);
- semantic reduce-tree call/token estimate, including two audits and at most one repair;
- all incomplete sources and authorization-based evidence downgrades.

Require `paper-agent report approve --plan ... --hash ... --approved-by ... --corpus-snapshot ... --search-audit ... --output-root ...` before Sol calls. Execute with `report --plan <REPORT_PLAN.json> --database <DB> --output-root <DIR>` plus either `--policy` or a matching v2 `--config` that resolves the summary policy. Pass artifact-scoped grants only as `--processing-grant ARTIFACT_SHA256=GRANT_ID`, where the digest is 64 lowercase hexadecimal characters (or use the equivalent frozen mapping file). Publish only when `paper-agent verify-report` and the final independent Sol audit both pass with zero blocker and major findings. If repair is needed, allow one typed repair pass followed by full re-verification and a fresh audit. Do not patch rendered Markdown directly. For an incremental request, use `report --diff-from <previous> --report-run-id <current> --output-root <DIR>` instead of comparing rendered Markdown by hand.

## Deliver results

Read the CLI's JSON output and referenced artifacts. Report:

- run IDs, status, resumability, and exact blockers;
- source/search/Stage 2/download/analysis/report coverage counts;
- output paths for audits, paper analyses, report, sidecar, and diff;
- model/profile and approval/grant provenance without sensitive payloads.

Clearly separate completed evidence from external gates such as missing real gold-label release artifacts, unavailable credentials, Cloudflare/CAPTCHA, institutional login, or model authorization. Never label an incomplete run complete.
