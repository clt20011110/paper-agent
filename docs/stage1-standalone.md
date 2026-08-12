# Standalone Stage 1 metadata census

`paper-agent stage1` enumerates venue metadata without a QueryPlan, Stage 2,
PDF downloads, or model calls.

## Commands

List the registered identifiers:

```bash
paper-agent stage1 list-venues
paper-agent stage1 list-venues --type conference
```

Collect one or more venues over an inclusive year range:

```bash
paper-agent stage1 collect \
  --venue neurips \
  --venue icml \
  --year-from 2016 \
  --year-to 2025 \
  --contact operator@example.org \
  --max-workers 4 \
  --output build/stage1/papers.jsonl
```

The default is fail-closed. The receipt is always written to
`papers.jsonl.receipt.json`, but `papers.jsonl` is only published if every
venue-year is proven complete. `--allow-incomplete` is an explicit diagnostic
escape hatch; its receipt and command status remain `incomplete`.

## Python API

```python
from paper_agent.stage1 import Stage1Request, collect_stage1_metadata

request = Stage1Request(("icml", "neurips"), 2016, 2025, max_workers=4)
result = collect_stage1_metadata(
    request,
    catalog=catalog,
    adapter_factory=adapter_factory,
)
```

The injected factory keeps the core interface usable with live HTTP,
approved response snapshots, or deterministic test adapters.

## What `complete` means

Completeness is a membership claim, not a claim that every metadata field is
non-empty. Each venue-year receipt records pagination termination, the source
census, parser raw/rejected/explicitly-excluded counts, stable-ID duplicates,
field coverage, and
response hashes. Missing abstracts or DOIs remain null with an explicit field
status; they are never synthesized.

If a descriptor has an authoritative `date_range`, years before launch or
after closure are recorded as `not_applicable`. They do not contact a provider
and do not make an otherwise complete request fail.
