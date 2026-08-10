---
name: paper-agent
description: Orchestrate the installed paper-agent CLI for auditable literature discovery, local Stage 2 screening, authorized PDF acquisition, per-paper analysis, and evidence-grounded Chinese review generation. Use when a user asks to plan, run, resume, inspect, audit, or export a paper-research workflow, or migrate an older paper-agent configuration.
---

# Paper Agent

Use the `paper-agent` console entry point as the only business implementation. Collect intent, obtain explicit approvals, invoke the CLI, and explain its structured results. Never import or run code under `.opencode`, duplicate crawler/model/database logic in this skill, or edit SQLite directly.

## Locate and inspect the CLI

1. Prefer the active environment's `paper-agent` executable. In a source checkout, prefer `.venv/bin/paper-agent` when present.
2. Run `paper-agent --help` and the relevant subcommand `--help` before constructing commands. Do not assume options that the installed version does not expose.
3. Run `paper-agent doctor` before a new production campaign or when runtime state may have changed.
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

Use the narrowest CLI command that satisfies the request. Preserve the same database and explicit `--run-id` where the CLI supports it. Prefer `resume` for an interrupted immutable run; do not create a new run merely to bypass drift or a failed gate.

For Stage 2, require a passed local release bundle and oMLX models. Never use a test fake, cloud fallback, unapproved model revision, or raw uncalibrated thresholds in production.

For Stage 3, exhaust public and authorized open-access providers first. If the CLI emits an attended publisher handoff, explain the target papers and batch size. Invoke `$download-authorized-papers` only for that audited handoff and only through the user's authorized visible browser session. Never request, inspect, copy, or log passwords, cookies, tokens, CAPTCHA contents, or session material. Stop the affected queue on login repair, CAPTCHA, 403, or 429 while allowing unrelated papers to continue.

For Stage 4 and Stage 4b, explain that `network=false` does not keep model payloads local. Require exact `remote_model_processing` policy allowance or approved artifact/model grants before sending full text or its restricted derivatives. Keep Stage 4 on `gpt-5.6-luna` and Stage 4b on `gpt-5.6-sol`; never upgrade, downgrade, or fall back silently.

## Generate a review

Run report planning first and show:

- frozen corpus and search-flow limitations;
- paper input scopes (`full_pdf`, `abstract_only`, `metadata_only`, or missing);
- semantic reduce-tree call/token estimate, including two audits and at most one repair;
- all incomplete sources and authorization-based evidence downgrades.

Require explicit ReportPlan hash approval before Sol calls. Publish only when the deterministic verifier and the final independent Sol audit both pass with zero blocker and major findings. If repair is needed, allow one typed repair pass followed by full re-verification and a fresh audit. Do not patch rendered Markdown directly.

## Deliver results

Read the CLI's JSON output and referenced artifacts. Report:

- run IDs, status, resumability, and exact blockers;
- source/search/Stage 2/download/analysis/report coverage counts;
- output paths for audits, paper analyses, report, sidecar, and diff;
- model/profile and approval/grant provenance without sensitive payloads.

Clearly separate completed evidence from external gates such as missing real gold-label release artifacts, unavailable credentials, Cloudflare/CAPTCHA, institutional login, or model authorization. Never label an incomplete run complete.
