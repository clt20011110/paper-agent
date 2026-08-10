# Phase 2 controlled smoke

This gate uses a single, low-QPS request to Crossref's public `/works` metadata
API. It checks only that the documented JSON list maps to a stable identifier
and title. It never requests a landing page, full text, or PDF; it requires no
credentials and makes no assertion about changing result totals.

The transport identifies itself with an explicit contact, uses a finite timeout,
retains ETag/Last-Modified validators for conditional requests, and surfaces
`Retry-After` rate limits to the provider runtime.

Official sources:

- <https://api.crossref.org/>
- <https://www.crossref.org/documentation/retrieve-metadata/rest-api/>
- <https://github.com/CrossRef/rest-api-doc>

The historical evidence beside this document records an observed request and
its digest, but the corresponding raw bytes are not present in this workspace.
It is therefore marked `snapshot_status: absent` and must not be presented as a
replayable fixture.

To make a new manual, auditable run (never in ordinary test runs), set an
explicit opt-in flag, a Crossref contact, and an output directory outside the
repository. This makes exactly one request with page size 1 and no retry; it
writes `crossref-response.json` and `crossref-evidence.json` to that directory.

```sh
PAPER_AGENT_RUN_LIVE_SMOKE=1 \
PAPER_AGENT_SMOKE_CONTACT='mailto:you@example.org' \
PAPER_AGENT_SMOKE_OUTPUT_DIR=/secure/path/paper-agent-smoke \
uv run pytest -m live_smoke
```

`ApprovedSnapshotTransport` can replay an approved JSON or XML API response
without network access after verifying its SHA-256. It is only a response replay
mechanism; it does not claim support for Crossref or any other provider's bulk
snapshot format.
