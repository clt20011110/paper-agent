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
| IJCAI 2016 | IJCAI proceedings | 651 | 651 | 651 legitimately absent | 651 | complete |
| IJCAI 2024 | IJCAI proceedings | 1,048 | 1,048 | 1,048 | 1,048 | complete |
| ICML 2024 | PMLR volume 235 | 2,610 | 2,610 | 2,610 legitimately absent | 2,610 | complete |
| NeurIPS 2016 | NeurIPS proceedings | 569 | 569 | 569 legitimately absent | 569 | complete |
| NeurIPS 2024 | NeurIPS proceedings | 4,493 | 4,493 | 4,493 | 4,493 | complete |

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

IJCAI 2017 onward uses one cursor-paged Crossref prefix query per year and an
exact `10.24963/ijcai.<year>/<paper_id>` join.  The 2016 volume predates that
DOI series, so DOI is explicitly `legitimately_absent`; all 651 abstracts and
PDF URLs were recovered from the official legacy paper pages with a four-QPS
policy limit.  Both live runs completed without changing primary membership.

PMLR volumes use a single official `mlresearch/v<volume>` `gh-pages` archive.
The checked-in frontmatter gives stable ID, abstract and public PDF URL for
every paper.  ICML 2024 therefore enriched 2,610 papers using one 8-MB
decompressed metadata archive rather than 2,610 detail-page requests.  PMLR
frontmatter does not assign article DOI; those fields are preserved as null
with `legitimately_absent/not_assigned_by_venue`, never synthesized.

NeurIPS uses the conference's public annual JSON export to batch-join abstracts
by normalized title.  For 2016, only three legacy title variants required an
official detail-page fallback.  The 2016–2021 proceedings do not expose article
DOIs, while 2022 onward registers them: 2024 used ten cursor-paged Crossref
prefix pages and exact proceedings-container/title joins, with one official
`citation_doi` fallback for a registry title variant.  All 4,493 records retain
the public proceedings PDF URL.

Representative live artifacts were written outside the repository:

- `/tmp/stage1-aaai-2020-enriched.jsonl.receipt.json`
- `/tmp/stage1-iclr-2020-enriched.jsonl.receipt.json`
- `/tmp/stage1-iclr-2024-enriched.jsonl.receipt.json`
- `/tmp/stage1-iclr-2025-enriched.jsonl.receipt.json`
- `/tmp/stage1-iclr-2017-enriched.jsonl.receipt.json`
- `/tmp/stage1-ijcai-2016-enriched.receipt.json`
- `/tmp/stage1-ijcai-2024-enriched.jsonl.receipt.json`
- `/tmp/stage1-icml-2024-enriched.receipt.json`
- `/tmp/stage1-neurips-2016-enriched.receipt.json`
- `/tmp/stage1-neurips-2024-enriched.receipt.json`

Automated regression command:

```bash
.venv/bin/python -m pytest \
  tests/test_stage1.py tests/test_stage1_hydration.py \
  tests/test_venue_http_transport.py tests/test_http_transport.py \
  tests/test_manifests.py -q
```
