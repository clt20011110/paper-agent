# Stage 1 field enrichment acceptance — 2026-08-13

This acceptance run adds field hydration without changing the previously
proven venue membership census.  Hydrators receive the exact primary external
ID set and publication fails if they add, remove, or leave required fields
unresolved.

| Venue/year | Primary membership | Records | Abstract | DOI | Public PDF | Result |
|---|---|---:|---:|---:|---:|---|
| AAAI 2020 | OJS issues | 1,865 | 1,865 | 1,865 | 1,865 | complete |
| ICLR 2020 | DBLP conference TOC | 687 | 687 | 687 legitimately absent | 687 | complete |
| ICLR 2024 | DBLP conference TOC | 2,260 | 2,260 | 2,260 legitimately absent | 2,260 | complete |
| ICLR 2025 | DBLP conference TOC | 3,704 | 3,704 | 3,704 legitimately absent | 3,704 | complete |
| ICLR 2017 | DBLP conference TOC | 198 | 170 proven, 28 unresolved | 198 legitimately absent | 198 canonical URLs | incomplete |

AAAI used Crossref year pagination for the bulk registry join and official OJS
OAI-PMH `GetRecord` as the article-ID fallback.  OJS records 5591 and 6915 lack
a complete abstract; their official public PDF first pages were parsed
deterministically between the abstract and introduction headings.  No model was
used.  Both records retain field-level source provenance.

ICLR uses the OpenReview forum ID carried by each DBLP record.  The official
ICLR virtual JSON supplies abstracts in bulk for 2020–2025 and the canonical
public PDF URL is `https://openreview.net/pdf?id=<forum_id>`.  The conference
does not assign article-level proceedings DOI, so null plus
`legitimately_absent/not_assigned_by_venue` is the truthful complete state.

OpenReview currently returns a challenge to unattended API requests.  ICLR
2017 has no modern ICLR bulk JSON.  Exact arXiv title matching proves 170
abstracts, while 28 remain unresolved.  Strict publication correctly emits an
incomplete receipt; it does not synthesize or silently omit those papers.

Representative live artifacts were written outside the repository:

- `/tmp/stage1-aaai-2020-enriched.jsonl.receipt.json`
- `/tmp/stage1-iclr-2020-enriched.jsonl.receipt.json`
- `/tmp/stage1-iclr-2024-enriched.jsonl.receipt.json`
- `/tmp/stage1-iclr-2025-enriched.jsonl.receipt.json`
- `/tmp/stage1-iclr-2017-enriched.jsonl.receipt.json`

Automated regression command:

```bash
.venv/bin/python -m pytest \
  tests/test_stage1.py tests/test_stage1_hydration.py \
  tests/test_venue_http_transport.py tests/test_http_transport.py \
  tests/test_manifests.py -q
```
