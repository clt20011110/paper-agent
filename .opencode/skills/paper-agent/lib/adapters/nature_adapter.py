#!/usr/bin/env python3
"""
Nature Journals Adapter using Springer Nature API

Implements venue adapter for Nature journals using the Springer Nature Metadata API.
Supports:
- Nature Machine Intelligence (journal code: 42256)
- Nature Chemistry (journal code: 41557)
- Nature Communications (journal code: 41467) - Fully Open Access
- Nature (journal code: 41586)
- Other Nature journals via journal code

API Documentation: https://dev.springernature.com/
"""

import requests
import time
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

from database import Paper
from .base import VenueAdapter, VenueConfig
from .registry import AdapterRegistry


logger = logging.getLogger(__name__)


class NatureAdapter(VenueAdapter):
    """
    Adapter for Nature journals using Springer Nature API.
    
    The Springer Nature Metadata API provides access to 16+ million documents
    including articles from Nature journals.
    
    API Key Required:
        Register at https://dev.springernature.com/ for a free API key.
        
    Rate Limits:
        - Basic (free): 100 hits/min, 500 hits/day
        - Premium: 300 hits/min, 10,000 hits/day
    
    PDF Access:
        - Nature Communications: Fully Open Access
        - Other Nature journals: Mixed (open access articles available)
    """
    
    API_BASE_URL = "https://api.springernature.com/meta/v2/json"
    
    # Mapping of journal codes to human-readable names
    JOURNAL_NAMES = {
        '41586': 'Nature',
        '42256': 'Nature Machine Intelligence',
        '41557': 'Nature Chemistry',
        '41467': 'Nature Communications',
        '41562': 'Nature Human Behaviour',
        '41551': 'Nature Biomedical Engineering',
        '41591': 'Nature Medicine',
        '41564': 'Nature Microbiology',
        '41570': 'Nature Catalysis',
        '41578': 'Nature Electronics',
    }
    
    # Journal codes that are fully open access
    OPEN_ACCESS_JOURNALS = {'41467'}  # Nature Communications is fully OA
    
    def __init__(self, journal_code: str):
        """
        Initialize the Nature adapter with a specific journal code.
        
        Args:
            journal_code: Springer Nature journal code (e.g., '41586' for Nature)
        """
        self.journal_code = journal_code
    
    @property
    def platform_name(self) -> str:
        """Return platform identifier."""
        return f"nature_{self.journal_code}"
    
    @property
    def venue_type(self) -> str:
        """Return venue type (Nature journals are journals)."""
        return "journal"
    
    @property
    def journal_name(self) -> str:
        """Return the human-readable journal name."""
        return self.JOURNAL_NAMES.get(self.journal_code, f"Nature Journal ({self.journal_code})")
    
    @property
    def is_open_access(self) -> bool:
        """Check if this journal is fully open access."""
        return self.journal_code in self.OPEN_ACCESS_JOURNALS
    
    def crawl(self, config: VenueConfig) -> List[Paper]:
        """
        Crawl Nature journal papers for given years.
        
        Args:
            config: VenueConfig with years and optional api_key in additional_params
            
        Returns:
            List of Paper objects
            
        Raises:
            ValueError: If API key is not provided
            requests.RequestException: If API request fails
        """
        api_key = config.additional_params.get('api_key') or config.additional_params.get('nature_api_key')
        if not api_key:
            raise ValueError(
                "Nature API key required. "
                "Get one at https://dev.springernature.com/ and pass as 'api_key' in additional_params"
            )
        
        all_papers = []
        
        for year in config.years:
            logger.info(f"Crawling {self.journal_name} {year}...")
            try:
                papers = self._crawl_year(year, api_key)
                all_papers.extend(papers)
                logger.info(f"Found {len(papers)} papers for {self.journal_name} {year}")
            except requests.RequestException as e:
                logger.error(f"API error crawling {self.journal_name} {year}: {e}")
                raise
            except Exception as e:
                logger.error(f"Error crawling {self.journal_name} {year}: {e}")
                # Continue with other years
            
            # Rate limiting between year requests
            time.sleep(self.rate_limit_delay())
        
        return all_papers
    
    def _crawl_year(self, year: int, api_key: str) -> List[Paper]:
        """
        Crawl papers for a specific year using Springer Nature API.
        
        Uses pagination to retrieve all papers for the year.
        
        Args:
            year: Publication year
            api_key: Springer Nature API key
            
        Returns:
            List of Paper objects
        """
        papers = []
        page = 1
        per_page = 25  # API limit is 25 per page
        
        while True:
            params = {
                'api_key': api_key,
                'q': f'journal:{self.journal_code} year:{year}',
                'p': per_page,
                's': (page - 1) * per_page + 1,  # Springer uses 1-based indexing for 's'
            }
            
            logger.debug(f"Fetching page {page} for {self.journal_name} {year}")
            
            response = requests.get(
                self.API_BASE_URL,
                params=params,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            records = data.get('records', [])
            
            if not records:
                break
            
            for record in records:
                try:
                    paper = self._parse_record(record, year)
                    if paper:
                        papers.append(paper)
                except Exception as e:
                    logger.warning(f"Error parsing record: {e}")
                    continue
            
            # Check if we've reached the end
            total = data.get('result', [{}])[0].get('total', 0)
            total = int(total) if isinstance(total, str) else total
            
            if page * per_page >= total:
                break
            
            page += 1
            time.sleep(0.5)  # Be nice to the API between pages
        
        return papers
    
    def _parse_record(self, record: Dict[str, Any], year: int) -> Optional[Paper]:
        """
        Parse a Springer API record into a Paper object.
        
        Args:
            record: API response record
            year: Publication year
            
        Returns:
            Paper object or None if parsing fails
        """
        # Extract DOI - required for paper ID
        doi = record.get('doi')
        if not doi:
            logger.debug("Skipping record without DOI")
            return None
        
        # Create paper ID from DOI
        paper_id = self._doi_to_paper_id(doi)
        
        # Extract title
        title = record.get('title', '')
        if not title:
            logger.debug(f"Skipping record without title: {doi}")
            return None
        
        # Extract abstract
        abstract = record.get('abstract', '')
        
        # Extract authors
        authors = []
        creators = record.get('creators', [])
        for creator in creators:
            if isinstance(creator, dict):
                author_name = creator.get('creator', '')
                if author_name:
                    authors.append(author_name)
            elif isinstance(creator, str):
                authors.append(creator)
        
        # Extract keywords
        keywords = []
        kw_data = record.get('keyword', [])
        if kw_data:
            if isinstance(kw_data, list):
                keywords = kw_data
            elif isinstance(kw_data, str):
                keywords = [k.strip() for k in kw_data.split(',')]
        
        # Determine PDF availability
        open_access = record.get('openaccess', 'false').lower() == 'true'
        pdf_url = None
        download_available = 'none'
        
        # Get URL from API response
        urls = record.get('url', [])
        if urls:
            # Find the PDF URL
            for url_entry in urls:
                if isinstance(url_entry, dict):
                    url_value = url_entry.get('value', '')
                    if url_value:
                        if open_access or self.is_open_access:
                            pdf_url = url_value
                            download_available = 'nature'
                            break
        
        # Construct Nature PDF URL if we have DOI
        if not pdf_url and doi:
            pdf_url = self.get_pdf_url(paper_id)
            if self.is_open_access or open_access:
                download_available = 'nature'
        
        return Paper(
            id=paper_id,
            title=title,
            abstract=abstract,
            authors=authors,
            keywords=keywords,
            year=year,
            venue=self.journal_name,
            venue_type='journal',
            source_platform=self.platform_name,
            crawl_date=datetime.now().isoformat(),
            pdf_url=pdf_url,
            doi=doi,
            download_available=download_available
        )
    
    def _doi_to_paper_id(self, doi: str) -> str:
        """
        Convert DOI to a safe paper ID.
        
        Replaces slashes and dots with underscores for use as ID.
        
        Args:
            doi: Digital Object Identifier
            
        Returns:
            Safe paper ID string
        """
        # Replace / and . with _ to create safe ID
        return doi.replace('/', '_').replace('.', '_')
    
    def _paper_id_to_doi(self, paper_id: str) -> str:
        """
        Convert paper ID back to DOI.
        
        This is a best-effort conversion since DOI format can be ambiguous.
        
        Args:
            paper_id: Paper identifier
            
        Returns:
            DOI string
        """
        # DOIs start with "10." so we can reconstruct
        # First underscore after "10" should be a dot
        if paper_id.startswith('10_'):
            # Convert first underscore after "10" back to dot
            rest = paper_id[3:]  # After "10_"
            # The next underscore should be "/"
            if '_' in rest:
                idx = rest.index('_')
                return f"10.{rest[:idx]}/{rest[idx+1:]}"
        
        # Fallback: try simple replacement
        return paper_id.replace('_', '/', 1).replace('_', '.', 1)
    
    def get_pdf_url(self, paper_id: str, **kwargs) -> Optional[str]:
        """
        Get PDF URL for a Nature paper.
        
        Nature PDFs follow the pattern:
        https://www.nature.com/articles/{doi}.pdf
        
        Note: Access depends on subscription or open access status.
        
        Args:
            paper_id: Paper identifier (derived from DOI)
            
        Returns:
            PDF URL if available
        """
        # Reconstruct DOI from paper_id
        doi = kwargs.get('doi') or self._paper_id_to_doi(paper_id)
        
        # Nature PDF URL format
        return f"https://www.nature.com/articles/{doi}.pdf"
    
    def rate_limit_delay(self) -> float:
        """
        Return recommended delay between requests.
        
        Springer Nature API rate limits:
        - Free tier: 100 hits/min (~0.6s between requests)
        - We use 1.0s to be safe
        
        Returns:
            Delay in seconds
        """
        return 1.0
    
    def get_supported_venues(self) -> List[str]:
        """
        Return list of supported venue names.
        
        Returns:
            List of journal names
        """
        return [self.journal_name]
    
    def check_availability(self) -> bool:
        """
        Check if the Nature API is available.
        
        Makes a simple request to check connectivity.
        
        Returns:
            True if API is reachable
        """
        try:
            # Make a simple request without API key to check if API is up
            response = requests.get(
                self.API_BASE_URL,
                params={'api_key': 'test', 'q': 'test', 'p': 1},
                timeout=10
            )
            # Even with invalid key, we should get a response (though possibly error)
            return response.status_code in (200, 401, 403)
        except requests.RequestException:
            return False
    
    def validate_config(self, config: VenueConfig) -> bool:
        """
        Validate the configuration for this adapter.
        
        Args:
            config: VenueConfig to validate
            
        Returns:
            True if configuration is valid
        """
        # Check for API key
        if not config.additional_params.get('api_key'):
            logger.warning(
                "API key not provided. "
                "Get one at https://dev.springernature.com/ and pass as 'api_key'"
            )
            return False
        
        # Check years
        current_year = datetime.now().year
        for year in config.years:
            if year < 1990 or year > current_year + 1:
                logger.warning(f"Year {year} may not have data available")
        
        return True


# =============================================================================
# Specific Journal Adapters
# =============================================================================

@AdapterRegistry.register
class NatureMachineIntelligenceAdapter(NatureAdapter):
    """
    Adapter for Nature Machine Intelligence (journal code: 42256).
    
    Nature Machine Intelligence publishes research on artificial intelligence
    and machine learning.
    
    PDF Access: Mixed (open access articles available, subscription for others)
    """
    
    def __init__(self):
        super().__init__('42256')
    
    @property
    def platform_name(self) -> str:
        """Return platform identifier."""
        return "nature_machine_intelligence"


@AdapterRegistry.register
class NatureChemistryAdapter(NatureAdapter):
    """
    Adapter for Nature Chemistry (journal code: 41557).
    
    Nature Chemistry publishes research across all areas of chemistry.
    
    PDF Access: Mixed (open access articles available, subscription for others)
    """
    
    def __init__(self):
        super().__init__('41557')
    
    @property
    def platform_name(self) -> str:
        """Return platform identifier."""
        return "nature_chemistry"


@AdapterRegistry.register
class NatureCommunicationsAdapter(NatureAdapter):
    """
    Adapter for Nature Communications (journal code: 41467).
    
    Nature Communications is a fully open access journal, making it ideal
    for bulk paper retrieval.
    
    PDF Access: Fully Open Access (CC-BY license)
    """
    
    def __init__(self):
        super().__init__('41467')
    
    @property
    def platform_name(self) -> str:
        """Return platform identifier."""
        return "nature_communications"


@AdapterRegistry.register
class NatureMainAdapter(NatureAdapter):
    """
    Adapter for Nature main journal (journal code: 41586).
    
    Nature is the flagship multidisciplinary science journal.
    
    PDF Access: Mixed (open access articles available, subscription for others)
    """
    
    def __init__(self):
        super().__init__('41586')
    
    @property
    def platform_name(self) -> str:
        """Return platform identifier."""
        return "nature_main"