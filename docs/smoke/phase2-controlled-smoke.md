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

The run evidence is stored beside this document in
`phase2-controlled-smoke-evidence.json`; its response digest permits comparison
without committing the volatile response body.
