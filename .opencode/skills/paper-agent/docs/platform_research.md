# Platform Research Report: API and Scraping Approaches

**Generated:** March 15, 2026  
**Purpose:** Research conference and journal platforms for crawler implementation

---

## Summary Table

| Venue | Platform | Has API | API Type | PDF Access | Scraping Difficulty | Notes |
|-------|----------|---------|----------|------------|---------------------|-------|
| ACL | aclanthology.org | Yes (Python lib) | Python Package | Open Access | Low | Excellent Python library `acl-anthology` with full metadata |
| CVPR | openaccess.thecvf.com | No | N/A | Open Access | Low-Medium | HTML scraping required; predictable structure |
| ICCV | openaccess.thecvf.com | No | N/A | Open Access | Low-Medium | Same as CVPR; unified CVF platform |
| IJCAI | ijcai.org | No | N/A | Open Access | Medium | HTML-based proceedings; structured but manual scraping |
| DAC | IEEE Xplore / ACM DL | Yes | IEEE REST API | Mixed (subscription) | Medium | Requires IEEE API key; published via IEEE/ACM |
| TCAD | IEEE Xplore | Yes | IEEE REST API | Subscription | Medium | IEEE API for metadata; paywalled PDFs |
| ICCAD | IEEE Xplore / ACM DL | Yes | IEEE REST API | Mixed (subscription) | Medium | Similar to DAC; IEEE/ACM published |
| Nature MI | nature.com | Yes | Springer Nature API | Mixed (OA available) | Low | Springer Nature API; premium for higher limits |
| Nature Chemistry | nature.com | Yes | Springer Nature API | Mixed (OA available) | Low | Same as Nature MI |
| Nature Communications | nature.com | Yes | Springer Nature API | Open Access | Low | Fully OA journal |
| Cell | cell.com (Elsevier) | Yes | ScienceDirect API | Subscription | Medium-High | Elsevier API; paywalled content |
| Science | science.org | No | N/A | Subscription | High | No public API; strict paywall; TDM requires agreement |

---

## Detailed Platform Analysis

### 1. ACL (Association for Computational Linguistics)

**Website:** https://aclanthology.org

#### API Availability
- **Has API:** Yes (Python library, not REST API)
- **Documentation:** https://acl-anthology.readthedocs.io/
- **GitHub:** https://github.com/acl-org/acl-anthology

#### Python Library Details
```bash
pip install acl-anthology
```

The `acl-anthology` package provides:
- `Anthology` class for accessing all data
- `CollectionIndex` for collections, volumes, papers
- `PersonIndex` for authors and editors
- `VenueIndex` for venues
- `EventIndex` for events
- Full metadata: title, abstract, authors, year, pdf_url, doi, bibTeX

#### Metadata Available
- Title, abstract, authors
- Publication year, venue
- PDF URL (direct download)
- DOI
- BibTeX citation
- Editor information
- SIG affiliations

#### PDF Access
- **Status:** Fully Open Access
- All papers freely downloadable
- No authentication required

#### Recommended Approach
```python
from anthology import Anthology

anthology = Anthology()
# Access papers, collections, volumes directly
```

#### Rate Limits
- No explicit rate limits for the library
- Respects server when downloading PDFs (use delays)

---

### 2. CVPR (IEEE/CVF Conference on Computer Vision and Pattern Recognition)

**Website:** https://openaccess.thecvf.com

#### API Availability
- **Has API:** No official REST API
- **Method:** HTML scraping required

#### Scraping Approach
The CVF Open Access website has a predictable structure:
- Conference pages: `https://openaccess.thecvf.com/{CONF}{YEAR}?day=all`
- Paper listings in HTML tables
- Direct PDF links available

#### HTML Structure
```
Page: /CVPR2024?day=all
- Paper entries in HTML
- Each paper has: title, authors, PDF link, supplementary materials
- Pagination: "day" parameter (day1, day2, day3, all)
```

#### Metadata Available
- Title
- Authors
- PDF URL
- Supplementary materials URL
- Session/day information

#### PDF Access
- **Status:** Fully Open Access
- Direct PDF downloads available
- No authentication required

#### Recommended Libraries
- `requests` + `BeautifulSoup4` for HTML parsing
- `selenium` if JavaScript rendering needed (usually not)
- Existing scrapers: 
  - https://github.com/ElhamKhan859/CVF-Scrapper-Public
  - https://github.com/seanywang0408/Crawling-CV-Conference-Papers

#### Scraping Difficulty
- **Rating:** Low-Medium
- Structure is consistent across years
- No anti-scraping measures
- Rate limiting recommended (1-2 second delays)

---

### 3. ICCV (International Conference on Computer Vision)

**Website:** https://openaccess.thecvf.com

#### API Availability
- **Has API:** No (same platform as CVPR)
- **Method:** HTML scraping

#### Notes
- Same infrastructure as CVPR
- URLs: `https://openaccess.thecvf.com/ICCV{YEAR}`
- Identical scraping approach

---

### 4. IJCAI (International Joint Conference on AI)

**Website:** https://www.ijcai.org

#### API Availability
- **Has API:** No official API
- **Method:** HTML scraping from proceedings pages

#### Proceedings Structure
```
URL Pattern: https://www.ijcai.org/proceedings/{YEAR}/
- Paper pages: /proceedings/{YEAR}/{paper_id}
- PDF links: /proceedings/{YEAR}/{paper_id}.pdf
```

#### HTML Structure (Observed from 2025 proceedings)
- Papers organized by track/session
- Each paper has:
  - Title
  - Authors
  - PDF link (direct)
  - Details page with abstract

#### Metadata Available
- Title
- Authors (with affiliations)
- Abstract (on details page)
- PDF URL
- Session/track information

#### PDF Access
- **Status:** Open Access
- Direct PDF downloads available

#### Scraping Difficulty
- **Rating:** Medium
- Clean HTML structure
- May need to follow links to get abstracts
- No authentication required

---

### 5. DAC (Design Automation Conference)

**Website:** https://dac.com

#### Publication Platform
- Published via IEEE Xplore and ACM Digital Library
- Proceedings available at: https://dl.acm.org/doi/proceedings/10.1145/{proceedings-id}

#### API Availability
- **Has API:** Yes (via IEEE Xplore API)
- IEEE API can query DAC proceedings

#### IEEE API for DAC
- Use IEEE Metadata Search API
- Filter by conference name "Design Automation Conference"
- Requires API key registration

#### PDF Access
- **Status:** Mixed (Subscription/Institutional Access)
- IEEE Xplore: Requires subscription
- Some papers may be open access
- Authors often provide preprints

#### Recommended Approach
1. Use IEEE API for metadata
2. For PDFs: institutional access or author preprints

---

### 6. TCAD (IEEE Transactions on Computer-Aided Design)

**Platform:** IEEE Xplore

#### API Availability
- **Has API:** Yes (IEEE Xplore API)
- **Documentation:** https://developer.ieee.org/

#### IEEE Xplore API Details

**Available APIs:**
1. **Metadata Search API**
   - Query 6+ million documents
   - Boolean search support
   - Returns: title, abstract, authors, DOI, etc.

2. **DOI API**
   - Query up to 25 DOIs per request
   - Returns full metadata

3. **Open Access API**
   - Access open access articles
   - Full-text retrieval

4. **Full-Text Access API**
   - Chargeable full-text access
   - Requires subscription

#### Authentication
- API Key required
- Register at: https://developer.ieee.org/
- Keys issued during business hours (8am-5pm ET, Mon-Fri)

#### Rate Limits
- Documented in API terms of use
- SDKs available: Java, PHP, Python

#### PDF Access
- **Status:** Subscription required
- Institutional access typical
- Open access articles available via OA API

---

### 7. ICCAD (IEEE/ACM International Conference on Computer-Aided Design)

**Platform:** IEEE Xplore / ACM Digital Library

#### API Availability
- **Has API:** Yes (via IEEE and ACM APIs)
- Similar to DAC and TCAD

#### Publication Venues
- IEEE Xplore: https://ieeexplore.ieee.org/document/{id}
- ACM DL: https://dl.acm.org/doi/proceedings/10.1145/{id}

#### Notes
- Same IEEE API approach as TCAD
- ACM also has Digital Library API (separate)

---

### 8. Nature Machine Intelligence

**Website:** https://www.nature.com/natmachintell/

#### API Availability
- **Has API:** Yes (Springer Nature API)
- **Portal:** https://dev.springernature.com/

#### Springer Nature API Details

**Available APIs:**
1. **Metadata API**
   - 16+ million documents
   - Returns: title, abstract, DOI, authors, etc.

2. **Open Access API**
   - Free access to OA content
   - Rate: 100 hits/min (Basic), 300 hits/min (Premium)

3. **Meta API (v1)**
   - Enhanced metadata retrieval

4. **TDM API (Text & Data Mining)**
   - Bulk content access

#### Authentication
- API Key required
- Register at developer portal
- Free tier available

#### Python Client
```bash
pip install springernature-api-client
```

Official Python wrapper: https://github.com/springernature/springernature_api_client

#### Rate Limits
| Plan | Rate | Daily Limit |
|------|------|-------------|
| Basic | 100 hits/min | 500 hits/day |
| Premium | 300 hits/min | 10,000 hits/day |

#### PDF Access
- **Status:** Mixed (Open Access available)
- Open access articles: free PDFs
- Subscription articles: paywalled
- APC: $11,390 for OA (as of 2024)

---

### 9. Nature Chemistry

**Website:** https://www.nature.com/nchem/

#### API Availability
- Same Springer Nature API as above
- Journal-specific queries via API

#### PDF Access
- **Status:** Mixed
- Open access option available
- Subscription content paywalled

---

### 10. Nature Communications

**Website:** https://www.nature.com/ncomms/

#### API Availability
- Same Springer Nature API

#### PDF Access
- **Status:** Fully Open Access
- All articles freely available
- CC-BY license

#### Notes
- Easiest Nature journal for bulk access
- All content accessible via Open Access API

---

### 11. Cell

**Website:** https://www.cell.com (Elsevier ScienceDirect)

#### API Availability
- **Has API:** Yes (Elsevier ScienceDirect API)
- **Portal:** https://dev.elsevier.com/

#### Elsevier API Details

**Available APIs:**
1. **ScienceDirect Search API**
   - Search ScienceDirect content
   - Metadata retrieval

2. **ScienceDirect Object Retrieval**
   - Full-text retrieval (if entitled)

3. **Journals API (new)**
   - Full-text access by DOI
   - XOCS XML format output
   - Max 1000 articles per request

4. **Scopus API**
   - Abstract and citation data
   - Broader coverage than ScienceDirect

#### Authentication
- API Key required
- Institutional token for subscription content
- Register at: https://dev.elsevier.com/

#### Access Types
1. **Non-Commercial (Academic)**
   - Free for most APIs
   - Subject to usage limits

2. **Commercial**
   - Requires license and subscription
   - Contact Elsevier for pricing

#### PDF Access
- **Status:** Subscription required
- High APC for OA: ~$10,400
- Institutional subscription typical
- Requires institutional token + API key for full access

---

### 12. Science (AAAS)

**Website:** https://www.science.org

#### API Availability
- **Has API:** No public API
- Some institutional access via Crossref

#### Access Methods
1. **Manual Download**
   - Individual article access via website
   - Subscription required

2. **TDM (Text Data Mining)**
   - Requires agreement with AAAS
   - Contact for institutional access
   - See: https://www.science.org/content/page/terms-service

3. **Crossref API**
   - Metadata only (no full-text)
   - DOI lookup

#### PDF Access
- **Status:** Strictly Paywalled
- No open access content
- Individual/institutional subscription required
- No API for content retrieval

#### Scraping Difficulty
- **Rating:** High
- Strong anti-scraping measures
- Terms of service prohibit automated access
- Recommend: institutional TDM agreement

---

## Recommended Libraries by Platform

| Platform | Recommended Python Libraries |
|----------|----------------------------|
| ACL | `acl-anthology` |
| CVPR/ICCV | `requests`, `beautifulsoup4`, `lxml` |
| IJCAI | `requests`, `beautifulsoup4` |
| IEEE (DAC/TCAD/ICCAD) | `requests` (REST API), official SDK |
| Nature journals | `springernature-api-client` |
| Cell | `requests` (Elsevier API) |
| Science | Crossref API (metadata only) |

---

## Implementation Priority Recommendations

### High Priority (Easy Implementation)
1. **ACL** - Excellent Python library, open access
2. **Nature Communications** - Open access, good API
3. **CVPR/ICCV** - Open access, predictable scraping

### Medium Priority
4. **IJCAI** - Open access, manual scraping
5. **Nature MI/Chemistry** - API available, mixed access
6. **IEEE venues (DAC/TCAD/ICCAD)** - API available, subscription

### Low Priority (Challenging)
7. **Cell** - API exists but paywalled
8. **Science** - No API, strict paywall

---

## Notes on API Keys and Registration

### Free APIs (Immediate Access)
- ACL: No key needed (Python library)
- Springer Nature Basic: Free registration

### APIs Requiring Approval
- IEEE Xplore: Business hours approval (1-2 days)
- Elsevier: Instant registration, institutional token for full access

### Paywalled/Restricted
- Science: No API, requires TDM agreement

---

## Next Steps for Crawler Implementation

1. **Start with ACL** - Use existing `acl-anthology` library
2. **Implement CVPR/ICCV scraper** - Based on existing open-source scrapers
3. **Add Springer Nature support** - Use official Python client
4. **IEEE API integration** - Requires API key registration
5. **IJCAI HTML scraper** - Custom implementation

---

*End of Report*