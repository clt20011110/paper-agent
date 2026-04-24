#!/usr/bin/env python3
"""
DBLP Adapter for Computer Science Bibliography

Implements venue adapter for DBLP (dblp.org), providing access to:
- ICCAD (International Conference on Computer-Aided Design)
- DAC (Design Automation Conference)
- TCAD (IEEE Transactions on Computer-Aided Design)
- Other computer science conferences and journals

Features:
- NO API KEY REQUIRED - DBLP API is completely free and open
- Rate limit friendly - respects DBLP's fair use policy
- Complete metadata - title, authors, year, DOI, URL
- Note: DBLP does not provide abstracts

API Documentation: https://dblp.org/faq/How+to+use+dblp+search+API.html
"""

import requests
import time
import logging
import re
from html import unescape
from typing import List, Optional, Dict, Any, cast
from datetime import datetime
from pathlib import Path
import sys

# Add parent directory to path for imports
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from database import Paper
from .base import VenueAdapter, VenueConfig


logger = logging.getLogger(__name__)


class DBLPAdapter(VenueAdapter):
    """
    Adapter for DBLP (dblp.org) computer science bibliography.
    
    DBLP provides free, open access to publication metadata for computer science
    conferences and journals. No API key is required.
    
    Important Notes:
        - DBLP does NOT provide abstracts for papers
        - Focuses on bibliographic metadata (title, authors, venue, year, DOI)
        - All data is freely accessible without authentication
        - Please respect rate limits to avoid being blocked
        
    Supported Venues:
        - ICCAD: International Conference on Computer-Aided Design
        - DAC: Design Automation Conference
        - TCAD: IEEE Transactions on Computer-Aided Design (as journal)
        - Many other CS conferences via query parameter
    """
    
    API_BASE_URL = "https://dblp.org/search/publ/api"
    SEMANTIC_SCHOLAR_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
    _last_semantic_scholar_call = 0.0
    
    # Venue mappings for DBLP
    # Format: 'venue_code': ('dblp_prefix', 'venue_type', 'venue_name')
    VENUE_MAPPINGS: Dict[str, tuple] = {
        'ICCAD': ('conf/iccad', 'conference', 'International Conference on Computer-Aided Design'),
        'DAC': ('conf/dac', 'conference', 'Design Automation Conference'),
        'TCAD': ('journals/tcad', 'journal', 'IEEE Transactions on Computer-Aided Design'),
        'AAAI': ('conf/aaai', 'conference', 'AAAI Conference on Artificial Intelligence'),
        'ACL': ('conf/acl', 'conference', 'Annual Meeting of the ACL'),
        'CVPR': ('conf/cvpr', 'conference', 'Conference on Computer Vision and Pattern Recognition'),
        'ICCV': ('conf/iccv', 'conference', 'International Conference on Computer Vision'),
        'IJCAI': ('conf/ijcai', 'conference', 'International Joint Conference on Artificial Intelligence'),
    }
    
    def __init__(self, venue_code: str):
        """
        Initialize DBLP adapter for a specific venue.
        
        Args:
            venue_code: Venue code, e.g., 'ICCAD', 'DAC', 'TCAD'
        """
        self.venue_code = venue_code.upper()
        if self.venue_code not in self.VENUE_MAPPINGS:
            raise ValueError(
                f"Unknown venue: {venue_code}. "
                f"Supported: {list(self.VENUE_MAPPINGS.keys())}"
            )
        
        self.dblp_prefix, self._venue_type, self._venue_name = self.VENUE_MAPPINGS[self.venue_code]
    
    @property
    def platform_name(self) -> str:
        """Return platform identifier."""
        return self.venue_code.lower()
    
    @property
    def venue_type(self) -> str:
        """Return venue type ('conference' or 'journal')."""
        return self._venue_type
    
    @property
    def venue_name(self) -> str:
        """Return the full venue name."""
        return self._venue_name
    
    def crawl(self, config: VenueConfig) -> List[Paper]:
        """
        Crawl papers from DBLP for the configured venue and years.
        
        Args:
            config: VenueConfig with years
            
        Returns:
            List of Paper objects (without abstracts - DBLP limitation)
        """
        all_papers = []
        additional_params = config.additional_params or {}
        fetch_abstract = additional_params.get("fetch_abstract", True)
        fetch_pdf = additional_params.get("fetch_pdf", True)
        
        for year in config.years:
            logger.info(f"Crawling {self.venue_name} {year} from DBLP...")
            try:
                papers = self._crawl_year(year)
                if fetch_abstract or fetch_pdf:
                    self._enrich_with_semantic_scholar(
                        papers,
                        fetch_abstract=bool(fetch_abstract),
                        fetch_pdf=bool(fetch_pdf),
                        api_key=additional_params.get("semantic_scholar_api_key"),
                        min_delay=float(additional_params.get("semantic_scholar_delay", 2.0)),
                    )
                all_papers.extend(papers)
                abstract_count = sum(1 for paper in papers if paper.abstract)
                pdf_count = sum(1 for paper in papers if paper.pdf_url)
                logger.info(
                    f"Found {len(papers)} papers for {self.venue_name} {year} "
                    f"(abstracts={abstract_count}, pdf_urls={pdf_count})"
                )
            except requests.RequestException as e:
                logger.error(f"API error crawling {self.venue_name} {year}: {e}")
                raise
            except Exception as e:
                logger.error(f"Error crawling {self.venue_name} {year}: {e}")
                # Continue with other years
            
            # Rate limiting between year requests
            time.sleep(self.rate_limit_delay())
        
        return all_papers
    
    def _crawl_year(self, year: int) -> List[Paper]:
        """
        Crawl papers for a specific year using DBLP API.
        
        Uses pagination to retrieve all papers for the year.
        
        Args:
            year: Publication year
            
        Returns:
            List of Paper objects
        """
        papers = []
        
        # Build a DBLP stream query. The old key-prefix query
        # ("conf/dac/2024:*") is too broad because DBLP's full-text search can
        # match unrelated proceedings that merely contain similar tokens.
        query = f"stream:{self.dblp_prefix}:{year}"
        
        params = {
            'q': query,
            'h': 1000,  # Maximum results per request (DBLP limit)
            'format': 'json',
        }
        
        logger.debug(f"Querying DBLP: {query}")
        
        response = requests.get(
            self.API_BASE_URL,
            params=params,
            timeout=30
        )
        response.raise_for_status()
        
        data = response.json()
        
        # Parse results
        result = data.get('result', {})
        hits = result.get('hits', {})
        hit_list = hits.get('hit', [])
        if not isinstance(hit_list, list):
            hit_list = [hit_list] if hit_list else []
        
        total_records = int(hits.get('@total', 0))
        logger.info(f"  DBLP returned {len(hit_list)} of {total_records} records")

        # Some journal paths may not follow /{year}:* key layout on DBLP.
        if total_records == 0 and self._venue_type == 'journal':
            for alt_query in self._journal_fallback_queries(year):
                logger.info(f"  Retrying journal query with fallback pattern: {alt_query}")
                alt_params = {
                    'q': alt_query,
                    'h': 1000,
                    'format': 'json',
                }
                alt_resp = requests.get(self.API_BASE_URL, params=alt_params, timeout=30)
                alt_resp.raise_for_status()
                alt_data = alt_resp.json()
                alt_hits = alt_data.get('result', {}).get('hits', {})
                alt_list = alt_hits.get('hit', [])
                if not isinstance(alt_list, list):
                    alt_list = [alt_list] if alt_list else []

                filtered = self._filter_journal_hits(alt_list, year)
                logger.info(
                    f"  Journal fallback returned {len(alt_list)} raw / {len(filtered)} filtered records"
                )
                if filtered:
                    hit_list = filtered
                    total_records = len(filtered)
                    break
        
        for hit in hit_list:
            try:
                paper = self._parse_hit(hit, year)
                if paper:
                    papers.append(paper)
            except Exception as e:
                logger.warning(f"Error parsing hit: {e}")
                continue
        
        return papers

    def _enrich_with_semantic_scholar(
        self,
        papers: List[Paper],
        fetch_abstract: bool = True,
        fetch_pdf: bool = True,
        api_key: Optional[str] = None,
        min_delay: float = 2.0,
    ) -> None:
        """
        Enrich DBLP records with abstracts and open-access PDF links.

        DBLP is excellent for bibliographic coverage but does not expose
        abstracts. Semantic Scholar accepts DOI identifiers in batch, which lets
        us supplement most IEEE/ACM DBLP records without requiring an IEEE API
        key. Failures are non-fatal so DBLP crawling remains reliable.
        """
        doi_to_paper = {
            paper.doi.lower(): paper
            for paper in papers
            if paper.doi
        }
        if not doi_to_paper:
            return

        fields = ["title", "externalIds"]
        if fetch_abstract:
            fields.append("abstract")
        if fetch_pdf:
            fields.extend(["openAccessPdf", "isOpenAccess"])

        headers = {}
        if api_key:
            headers["x-api-key"] = api_key

        doi_items = list(doi_to_paper.items())
        batch_size = 100
        for idx in range(0, len(doi_items), batch_size):
            batch = doi_items[idx:idx + batch_size]
            ids = [f"DOI:{doi}" for doi, _paper in batch]
            try:
                response = self._semantic_scholar_batch_request(
                    ids=ids,
                    fields=fields,
                    headers=headers,
                    min_delay=min_delay,
                )
                response.raise_for_status()
                records = response.json()
            except requests.RequestException as e:
                logger.warning(f"Semantic Scholar enrichment failed: {e}")
                return

            if not isinstance(records, list):
                logger.warning("Semantic Scholar enrichment returned unexpected payload")
                return

            for (doi, paper), record in zip(batch, records):
                if not isinstance(record, dict):
                    continue

                returned_title = record.get("title") or ""
                if returned_title and self._title_similarity(paper.title, returned_title) < 0.45:
                    logger.debug(
                        "Skipping Semantic Scholar metadata due to title mismatch: "
                        f"{paper.title!r} != {returned_title!r}"
                    )
                    continue

                if fetch_abstract and not paper.abstract:
                    paper.abstract = self._clean_text(record.get("abstract") or "")

                if fetch_pdf:
                    self._apply_open_access_pdf(paper, record)

    def _semantic_scholar_batch_request(
        self,
        ids: List[str],
        fields: List[str],
        headers: Dict[str, str],
        min_delay: float,
    ) -> requests.Response:
        """Call Semantic Scholar with process-local throttling and one 429 retry."""
        for attempt in range(2):
            self._wait_for_semantic_scholar_slot(min_delay)
            response = requests.post(
                self.SEMANTIC_SCHOLAR_BATCH_URL,
                params={"fields": ",".join(fields)},
                json={"ids": ids},
                headers=headers,
                timeout=30,
            )
            if response.status_code != 429 or attempt == 1:
                return response

            retry_after = response.headers.get("Retry-After")
            wait_seconds = float(retry_after) if retry_after and retry_after.isdigit() else 10.0
            logger.info(f"Semantic Scholar rate limited; retrying after {wait_seconds:.1f}s")
            time.sleep(wait_seconds)

        return response

    def _wait_for_semantic_scholar_slot(self, min_delay: float) -> None:
        """Throttle Semantic Scholar requests across DBLP adapter instances."""
        now = time.monotonic()
        elapsed = now - DBLPAdapter._last_semantic_scholar_call
        if elapsed < min_delay:
            time.sleep(min_delay - elapsed)
        DBLPAdapter._last_semantic_scholar_call = time.monotonic()

    def _apply_open_access_pdf(self, paper: Paper, record: Dict[str, Any]) -> None:
        """Apply open-access PDF metadata from a Semantic Scholar record."""
        oa_pdf = record.get("openAccessPdf") or {}
        pdf_url = oa_pdf.get("url") if isinstance(oa_pdf, dict) else None
        if pdf_url:
            # Prefer a direct OA PDF over DBLP's publisher landing-page link.
            paper.pdf_url = pdf_url

        external_ids = record.get("externalIds") or {}
        arxiv_id = external_ids.get("ArXiv") if isinstance(external_ids, dict) else None
        if arxiv_id and not paper.arxiv_id:
            paper.arxiv_id = arxiv_id

        has_arxiv = bool(paper.arxiv_id or (paper.pdf_url and "arxiv.org" in paper.pdf_url.lower()))
        has_non_arxiv_pdf = bool(paper.pdf_url and "arxiv.org" not in paper.pdf_url.lower())
        if has_arxiv and has_non_arxiv_pdf:
            paper.download_available = "both"
        elif has_arxiv:
            paper.download_available = "arxiv"
            if not paper.pdf_url and paper.arxiv_id:
                paper.pdf_url = f"https://arxiv.org/pdf/{paper.arxiv_id}.pdf"
        elif has_non_arxiv_pdf:
            paper.download_available = "openreview"

    def _title_similarity(self, title1: str, title2: str) -> float:
        """Calculate a lightweight word-overlap similarity for title checks."""
        words1 = set(self._normalize_title(title1).split())
        words2 = set(self._normalize_title(title2).split())
        if not words1 or not words2:
            return 0.0
        return len(words1 & words2) / len(words1 | words2)

    def _normalize_title(self, title: str) -> str:
        """Normalize title text for fuzzy metadata matching."""
        title = unescape(title or "").lower()
        title = re.sub(r"[^a-z0-9\s]+", " ", title)
        return " ".join(title.split())

    def _journal_fallback_queries(self, year: int) -> List[str]:
        """
        Build fallback queries for journals when stream-style query returns zero.
        """
        queries = [f"{self.dblp_prefix} {year}"]
        if self.venue_code == 'TCAD':
            queries.extend(
                [
                    f"IEEE Trans. Comput. Aided Des. Integr. Circuits Syst. {year}",
                    f"Comput. Aided Des. Integr. Circuits Syst. {year}",
                ]
            )
        return queries

    def _filter_journal_hits(self, hits: List[Dict[str, Any]], year: int) -> List[Dict[str, Any]]:
        """
        Filter fallback journal results to the exact target venue/year.
        """
        filtered: List[Dict[str, Any]] = []
        for hit in hits:
            info = hit.get('info', {}) if isinstance(hit, dict) else {}
            info_year = info.get('year')
            if info_year and str(info_year) != str(year):
                continue

            if self.venue_code == 'TCAD':
                venue_text = (info.get('venue') or info.get('journal') or '').lower()
                url_text = (info.get('url') or '').lower()
                if 'comput. aided des. integr. circuits syst.' not in venue_text and '/journals/tcad/' not in url_text:
                    continue
            filtered.append(hit)
        return filtered
    
    def _parse_hit(self, hit: Dict[str, Any], year: int) -> Optional[Paper]:
        """
        Parse a DBLP hit into a Paper object.
        
        Args:
            hit: DBLP API hit
            year: Publication year
            
        Returns:
            Paper object or None if parsing fails
        """
        info = hit.get('info', {})
        if info.get('type') == 'Editorship':
            return None
        
        # Extract title
        title = info.get('title', '')
        if not title:
            return None
        
        # Clean title
        title = self._clean_text(title)
        
        # Extract authors
        authors = []
        authors_data = info.get('authors', {})
        if authors_data:
            author_list = authors_data.get('author', [])
            if not isinstance(author_list, list):
                author_list = [author_list]
            
            for author in author_list:
                if isinstance(author, dict):
                    author_name = author.get('text', '')
                    if author_name:
                        authors.append(author_name)
                elif isinstance(author, str):
                    authors.append(author)
        
        # Extract DOI
        doi = info.get('doi')
        
        # Extract URL (DBLP page)
        url = info.get('url', '')
        
        # Extract EE (electronic edition) - often points to publisher
        ee = info.get('ee', '')
        
        # Create paper ID
        if doi:
            paper_id = self._doi_to_paper_id(doi)
        else:
            # Use DBLP key as ID
            key = info.get('key', '')
            if key:
                paper_id = key.replace('/', '_')
            else:
                # Fallback to hash
                paper_id = f"dblp_{self.venue_code.lower()}_{year}_{hash(title) % 10000:04d}"
        
        # Determine download availability
        # DBLP doesn't provide direct PDF links, but provides links to publishers
        download_available = 'none'
        pdf_url = None
        
        if ee:
            # EE often points to publisher's page (IEEE, ACM, etc.)
            pdf_url = ee
            if 'doi.org' in ee or 'ieee' in ee.lower() or 'acm' in ee.lower():
                download_available = 'openreview'  # Available via publisher (may need subscription)
        
        return Paper(
            id=paper_id,
            title=title,
            abstract="",  # DBLP does not provide abstracts
            authors=authors,
            keywords=[],  # DBLP does not provide keywords
            year=year,
            venue=self.venue_code,
            venue_type=cast(
                Any,
                self._venue_type
            ),
            source_platform=self.platform_name,
            crawl_date=datetime.now().isoformat(),
            pdf_url=pdf_url,
            doi=doi,
            download_available=cast(Any, download_available)
        )
    
    def _doi_to_paper_id(self, doi: str) -> str:
        """Convert DOI to a safe paper ID."""
        return doi.replace('/', '_').replace('.', '_')
    
    def _clean_text(self, text: str) -> str:
        """Clean text by removing extra whitespace."""
        if not text:
            return ''
        return ' '.join(text.split()).strip()
    
    def get_pdf_url(self, paper_id: str, **kwargs) -> Optional[str]:
        """
        Get PDF URL for a paper.
        
        Note: DBLP does not provide direct PDF URLs. It provides links to publisher
        pages (IEEE, ACM, etc.) where PDFs may be available with subscription.
        
        Args:
            paper_id: Paper identifier
            **kwargs: Additional parameters including 'doi' or 'url'
            
        Returns:
            URL to publisher page (not direct PDF)
        """
        doi = kwargs.get('doi')
        url = kwargs.get('url')
        
        if doi:
            return f"https://doi.org/{doi}"
        elif url:
            return url
        
        return None
    
    def rate_limit_delay(self) -> float:
        """
        Return recommended delay between requests.
        
        DBLP does not specify strict rate limits, but we should be respectful.
        A delay of 1-2 seconds between requests is recommended.
        
        Returns:
            Delay in seconds
        """
        return 1.0
    
    def get_supported_venues(self) -> List[str]:
        """Return list of supported venue names."""
        return list(self.VENUE_MAPPINGS.keys())
    
    def check_availability(self) -> bool:
        """Check if the DBLP API is available."""
        try:
            response = requests.get(
                self.API_BASE_URL,
                params={'q': 'test', 'h': 1, 'format': 'json'},
                timeout=10
            )
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def validate_config(self, config: VenueConfig) -> bool:
        """Validate the configuration for this adapter."""
        current_year = datetime.now().year
        for year in config.years:
            if year < 1980 or year > current_year + 1:
                logger.warning(f"Year {year} may not have data available in DBLP")
        return True
    
    def supports_year(self, year: int) -> bool:
        """Check if year is supported."""
        current_year = datetime.now().year
        return 1980 <= year <= current_year + 1


# =============================================================================
# Specific Venue Adapters
# =============================================================================

class DBLP_ICCAD_Adapter(DBLPAdapter):
    """Adapter for ICCAD via DBLP."""
    
    def __init__(self):
        super().__init__('ICCAD')
    
    @property
    def platform_name(self) -> str:
        return "dblp_iccad"


class DBLP_DAC_Adapter(DBLPAdapter):
    """Adapter for DAC via DBLP."""
    
    def __init__(self):
        super().__init__('DAC')
    
    @property
    def platform_name(self) -> str:
        return "dblp_dac"


class DBLP_TCAD_Adapter(DBLPAdapter):
    """Adapter for TCAD via DBLP."""
    
    def __init__(self):
        super().__init__('TCAD')
    
    @property
    def platform_name(self) -> str:
        return "dblp_tcad"


class DBLP_AAAI_Adapter(DBLPAdapter):
    """Adapter for AAAI via DBLP."""

    def __init__(self):
        super().__init__('AAAI')

    @property
    def platform_name(self) -> str:
        return "dblp_aaai"


class DBLP_ACL_Adapter(DBLPAdapter):
    """Adapter for ACL via DBLP."""

    def __init__(self):
        super().__init__('ACL')

    @property
    def platform_name(self) -> str:
        return "dblp_acl"


class DBLP_CVPR_Adapter(DBLPAdapter):
    """Adapter for CVPR via DBLP."""

    def __init__(self):
        super().__init__('CVPR')

    @property
    def platform_name(self) -> str:
        return "dblp_cvpr"


class DBLP_ICCV_Adapter(DBLPAdapter):
    """Adapter for ICCV via DBLP."""

    def __init__(self):
        super().__init__('ICCV')

    @property
    def platform_name(self) -> str:
        return "dblp_iccv"


class DBLP_IJCAI_Adapter(DBLPAdapter):
    """Adapter for IJCAI via DBLP."""

    def __init__(self):
        super().__init__('IJCAI')

    @property
    def platform_name(self) -> str:
        return "dblp_ijcai"
