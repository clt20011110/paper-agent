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

The evidence beside this document records a controlled request made on
2026-08-10 and binds the source commit, provider manifest digest, raw response
size, and response digest. The exact response bytes are retained as
`crossref-response.json.b64` so text encoding or formatting cannot change the
captured hash; the offline suite decodes and replays them. Changing either the
snapshot or provider manifest invalidates the committed evidence. Volatile
result totals are observed but never used as acceptance assertions.

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
