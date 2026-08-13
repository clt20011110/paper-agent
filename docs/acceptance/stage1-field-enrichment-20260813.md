# Stage 1 field enrichment acceptance — 2026-08-14

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
| JMLR 2024 | JMLR volume 25 | 422 | 422 | 422 legitimately absent | 422 | complete |
| ACL 2024 | ACL Anthology pinned XML | 984 | 984 | 984 | 984 | complete |
| COLING/LREC-COLING 2024 | ACL Anthology pinned XML | 1,567 | 1,567 | 1,567 legitimately absent | 1,567 | complete |
| CVPR 2024 | CVF Open Access annual index | 2,716 | 2,716 | 2,715 + 1 legitimately absent | 2,716 | complete |
| ICCV 2025 | CVF Open Access annual index | 2,701 | 2,701 | 2,700 + 1 legitimately absent | 2,701 | complete |
| DAC 2024 | DBLP conference TOC | 370 | 370 | 370 | 370 canonical endpoints | complete; one OpenAlex exact-title recovery |
| ICCAD 2024 | DBLP conference TOC | 239 | 234 | 239 | 239 canonical endpoints | incomplete (5 abstracts; upstream limits/challenge) |
| JCIM 2024 | Crossref ISSN registry | 805 | 740 + 65 legitimately absent | 805 | 805 | complete |
| Nature Machine Intelligence 2024 | Crossref ISSN registry + Nature article metadata | 184 | 166 + 18 legitimately absent | 184 | 184 | complete |
| Nature Biotechnology 2024 | Crossref ISSN registry + Nature article metadata | 468 | 333 + 135 legitimately absent | 468 | 468 | complete; publisher document types |
| JACS 2024 | Crossref ISSN registry | 3,783 | 3,611 | 3,783 | 3,783 | incomplete (172 abstracts) |
| Angewandte Chemie 2024 | Crossref ISSN registry | 5,170 | 4,817 + 353 legitimately absent | 5,170 | 5,170 | complete; recovery-aware DOI batching |
| COLT 2024 | PMLR volume | 169 | 169 | 169 legitimately absent | 169 | complete |
| CoRL 2024 | PMLR volume | 264 | 264 | 264 legitimately absent | 264 | complete |
| EMNLP 2024 | ACL Anthology pinned XML | 1,447 | 1,447 | 1,447 | 1,447 | complete |
| NAACL 2024 | ACL Anthology pinned XML | 632 | 632 | 632 | 632 | complete |
| UAI 2024 | PMLR volume | 201 | 201 | 201 legitimately absent | 201 | complete |
| AISTATS 2024 | PMLR volume | 547 | 547 | 547 legitimately absent | 547 | complete; descriptor detail fallback |
| Science 2024 | Crossref ISSN registry | 2,039 | 1,741 | 2,039 | 2,039 canonical endpoints | incomplete (298 abstracts) |

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

PMLR volumes prefer one official `mlresearch/v<volume>` `gh-pages` archive.
The checked-in frontmatter gives stable ID, abstract and public PDF URL for
every paper.  ICML 2024 therefore enriched 2,610 papers using one 8-MB
decompressed metadata archive rather than 2,610 detail-page requests.  When
the large AISTATS v238 archive returned `IncompleteRead`, the fallback retained
all primary PDF/landing URLs and fetched all 547 official detail pages under
the PMLR rate policy, yielding 547/547 abstracts.  PMLR frontmatter does not
assign article DOI; those fields are preserved as null with
`legitimately_absent/not_assigned_by_venue`, never synthesized.

NeurIPS uses the conference's public annual JSON export to batch-join abstracts
by normalized title.  For 2016, only three legacy title variants required an
official detail-page fallback.  The 2016–2021 proceedings do not expose article
DOIs, while 2022 onward registers them: 2024 used ten cursor-paged Crossref
prefix pages and exact proceedings-container/title joins, with one official
`citation_doi` fallback for a registry title variant.  All 4,493 records retain
the public proceedings PDF URL.

JMLR's public RSS only covers the current volume, so it is not treated as a
historical snapshot.  The 2024 acceptance run joined any RSS records and then
fetched the 422 official article detail pages with a four-QPS policy limit.
Every detail page supplied an abstract and public PDF meta tag.  The journal's
official citation metadata/BibTeX does not assign article DOI, so DOI remains
null with `legitimately_absent/not_assigned_by_journal` provenance.

ACL-family enrichment bulk-pages the `10.18653` Crossref prefix for the year
and joins only exact Anthology IDs.  ACL 2024 recovered the one XML-missing
abstract from its official public PDF and retained DOI for all 984 papers.
The joint LREC-COLING 2024 proceedings has no `10.18653` registrations for its
1,567 papers, so DOI is truthfully marked `legitimately_absent/not_registered`
after the registry audit; all abstracts and PDFs were already present in the
pinned official Anthology XML.

CVF enrichment reads each conference's single official Open Access annual
index for membership and public PDF URLs.  CVPR 2023 onward and ICCV 2025 use
the conference virtual site's annual JSON for abstracts; older years use a
four-QPS official-detail fallback.  DOI resolution uses one annual DBLP
proceedings snapshot, exact/fuzzy-unique title joins, and a Crossref title
audit only for residual misses.  CVPR 2024 has one paper (`SportsSloMo`) and
ICCV 2025 one paper (`DAViD`) with no registered proceedings DOI after that
audit; these remain null with `legitimately_absent/not_registered` rather than
receiving a fabricated identifier.

EDA enrichment preserves the complete DBLP proceedings membership and its DOI
registrations, then resolves abstracts and OA locations in Semantic Scholar
batch requests of at most 500 DOI each. Exact-title OpenAlex inverted-index
matching is attempted before arXiv title matching; recovered abstracts retain
the graph-specific provenance. DAC's one residual abstract was recovered this
way, bringing its strict receipt to 370/370. The registered ACM PDF endpoint
remains available for Stage 3 even when no OA mirror exists. ICCAD's five
residual abstracts remain incomplete because ACM is behind a Cloudflare
challenge, Semantic Scholar returned rate limits, and the unauthenticated
OpenAlex daily budget was exhausted; the bounded runtime fails fast and leaves
these records resumable rather than discarding papers or inventing text.

All 17 Crossref-journal descriptors now use the same membership-preserving DOI
batch hydrator.  Crossref publisher PDF links are retained directly by the
primary adapter, while Europe PMC DOI batches provide abstracts and OA mirrors
before the rate-limited Semantic Scholar fallback is considered.  Exact-title
arXiv matching is an optional residual batch layer; a rate-limit failure is a
warning rather than a reason to discard already hydrated fields.  JCIM 2024
proved 805/805 DOI and publisher PDF links plus 740 real abstracts.  The other
65 Crossref works are issue mastheads/publication information, corrections,
additions, editorials, or other explicitly frozen non-research document types;
their absent abstract is `legitimately_absent/not_applicable_to_document_type`.
The strict receipt therefore completed even while Semantic Scholar returned
HTTP 429, without generating text for documents that have no abstract.

Nature journals additionally use the public article page's `dc.description`
metadata after the DOI batch sources.  The route is a separately registered,
terms-gated provider with four-QPS concurrency and never fetches restricted
article body/PDF content.  Nature Machine Intelligence 2024 completed all 184
records: 166 public abstracts plus 18 publisher-proven absent abstracts (six
corrections and twelve documents typed by Nature as Correspondence, Matters
Arising, or Books & Arts), with DOI and PDF links on every record.

Nature Biotechnology 2024 completed all 468 records after the authorized
Nature article-page route was enabled. It returned 333 public abstracts and
classified the remaining 135 by publisher document type as News in Brief (45),
Research Highlight (36), Author Correction (16), Correspondence (15), Podcast
(10), News (8), Publisher Correction (3), or Data Page (2). Those records
retain DOI and publisher PDF endpoints and are marked
`legitimately_absent/not_applicable_to_document_type` rather than receiving
synthetic abstracts.

JACS 2024 reached 3,783 DOI/PDF-complete records and 3,611 abstracts. The
remaining 172 registrations are retained in the receipt as unresolved rather
than classified without publisher evidence; the run is therefore incomplete.

The Angewandte Chemie 2024 recovery-aware live probe returned 5,170 DOI/PDF-complete records.
Its 353 abstract-free records are exactly 300 covers/frontispieces/graphical
abstracts, 37 corrigenda, and 16 classifieds; frozen anchored patterns mark
only these publisher document classes as not applicable.  Europe PMC batch
timeouts are recorded as warnings without discarding already hydrated fields,
and the strict receipt completed.  Science 2024 has
2,039 DOI-complete records and now receives a deterministic canonical
`science.org/doi/pdf/<doi>` link for all records, but 298 abstract-free news,
letters, corrections, and other material remain unresolved because the
publisher page currently presents an unattended Cloudflare challenge.  The
receipt therefore remains incomplete rather than inventing those abstracts.

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
- `/tmp/stage1-jmlr-2024-enriched.receipt.json`
- `/tmp/stage1-acl-2024-enriched.receipt.json`
- `/tmp/stage1-coling-2024-enriched.receipt.json`
- `/tmp/stage1-cvpr-2024-enriched.receipt.json`
- `/tmp/stage1-iccv-2025-enriched.receipt.json`
- `/private/tmp/stage1-dac-2024-openalex-final2.receipt.json`
- `/private/tmp/stage1-dac-2024-openalex-final2.jsonl`
- `/tmp/stage1-iccad-2024-enriched-v3.receipt.json`
- `/private/tmp/stage1-iccad-2024-openalex-final.receipt.json`
- `/private/tmp/stage1-iccad-2024-openalex-final.jsonl`
- `/tmp/stage1-jcim-2024-epmc-v2.receipt.json`
- `/tmp/stage1-nmi-2024-final2.receipt.json`
- `/private/tmp/stage1-nature-biotechnology-2024-p1-classified.receipt.json`
- `/private/tmp/stage1-nature-biotechnology-2024-p1-classified.jsonl`
- `/private/tmp/stage1-jacs-2024-p1-final.receipt.json`
- `/private/tmp/stage1-jacs-2024-p1-final.jsonl`
- `/tmp/stage1-angew-2024.receipt.json`
- `/private/tmp/stage1-aistats-2024-p1-final.receipt.json`
- `/private/tmp/stage1-aistats-2024-p1-final.jsonl`
- `/tmp/stage1-science-2024.receipt.json`

Automated regression command:

```bash
.venv/bin/python -m pytest \
  tests/test_stage1.py tests/test_stage1_hydration.py \
  tests/test_venue_http_transport.py tests/test_http_transport.py \
  tests/test_manifests.py -q
```
