#!/usr/bin/env python3
"""
IEEE Xplore Adapter for EDA Conferences and Journals

Implements venue adapter for IEEE Xplore platform, supporting:
- ICCAD (IEEE/ACM International Conference on Computer-Aided Design)
- DAC (Design Automation Conference)
- TCAD (IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems)

API Documentation: https://developer.ieee.org/
"""

import requests
import time
import logging
from typing import List, Optional, Dict, Any, Tuple, Literal, cast
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


class IEEEXploreAdapter(VenueAdapter):
    """
    Adapter for IEEE Xplore platform.
    
    Supports IEEE conferences and journals including:
    - ICCAD (International Conference on Computer-Aided Design)
    - DAC (Design Automation Conference)
    - TCAD (IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems)
    
    IEEE Xplore API Key Required:
        Register at https://developer.ieee.org/ for a free API key.
        
    Rate Limits:
        - Free tier: 100 calls/day
        - Paid tier: Higher limits available
        
    PDF Access:
        - Requires IEEE subscription for full PDF access
        - This adapter focuses on metadata (title, abstract, authors)
        - PDF URLs are provided when available but may require authentication
    """
    
    API_BASE_URL = "https://ieeexploreapi.ieee.org/api/v1/search/articles"
    
    # Conference and journal mappings
    # Format: 'venue_code': ('publication_number', 'venue_type', 'venue_name')
    VENUE_MAPPINGS: Dict[str, Tuple[str, str, str]] = {
        # Conferences
        'ICCAD': ('10008', 'conference', 'International Conference on Computer-Aided Design'),
        'DAC': ('10001', 'conference', 'Design Automation Conference'),
        # Journals  
        'TCAD': ('43', 'journal', 'IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems'),
    }
    
    def __init__(self, venue_code: str):
        """
        Initialize IEEE Xplore adapter for a specific venue.
        
        Args:
            venue_code: Venue code, e.g., 'ICCAD', 'DAC', 'TCAD'
        """
        self.venue_code = venue_code.upper()
        if self.venue_code not in self.VENUE_MAPPINGS:
            raise ValueError(
                f"Unknown venue: {venue_code}. "
                f"Supported: {list(self.VENUE_MAPPINGS.keys())}"
            )
        
        mapping = self.VENUE_MAPPINGS[self.venue_code]
        self.publication_number = mapping[0]
        self._venue_type = mapping[1]  # type: ignore  # Will be validated at runtime
        self._venue_name = mapping[2]
    
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
        Crawl papers from IEEE Xplore for the configured venue and years.
        
        Args:
            config: VenueConfig with years and optional api_key in additional_params
            
        Returns:
            List of Paper objects
            
        Raises:
            ValueError: If API key is not provided
            requests.RequestException: If API request fails
        """
        api_key = config.additional_params.get('api_key') or config.additional_params.get('ieee_api_key')
        if not api_key:
            raise ValueError(
                "IEEE Xplore API key required. "
                "Get one at https://developer.ieee.org/ and pass as 'api_key' in additional_params"
            )
        
        all_papers = []
        
        for year in config.years:
            logger.info(f"Crawling {self.venue_name} {year}...")
            try:
                papers = self._crawl_year(year, api_key)
                all_papers.extend(papers)
                logger.info(f"Found {len(papers)} papers for {self.venue_name} {year}")
            except requests.RequestException as e:
                logger.error(f"API error crawling {self.venue_name} {year}: {e}")
                raise
            except Exception as e:
                logger.error(f"Error crawling {self.venue_name} {year}: {e}")
                # Continue with other years
            
            # Rate limiting between year requests
            time.sleep(self.rate_limit_delay())
        
        return all_papers
    
    def _crawl_year(self, year: int, api_key: str) -> List[Paper]:
        """
        Crawl papers for a specific year using IEEE Xplore API.
        
        Uses pagination to retrieve all papers for the year.
        
        Args:
            year: Publication year
            api_key: IEEE Xplore API key
            
        Returns:
            List of Paper objects
        """
        papers = []
        start_record = 1
        max_records_per_call = 200  # IEEE Xplore API limit
        
        while True:
            params = {
                'apikey': api_key,
                'format': 'json',
                'publication_number': self.publication_number,
                'start_year': year,
                'end_year': year,
                'max_results': max_records_per_call,
                'start_record': start_record,
            }
            
            logger.debug(f"Fetching records starting from {start_record} for {self.venue_name} {year}")
            
            response = requests.get(
                self.API_BASE_URL,
                params=params,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            articles = data.get('articles', [])
            
            if not articles:
                break
            
            for article in articles:
                try:
                    paper = self._parse_article(article, year)
                    if paper:
                        papers.append(paper)
                except Exception as e:
                    logger.warning(f"Error parsing article: {e}")
                    continue
            
            # Check if we've retrieved all records
            total_records = data.get('total_records', 0)
            if start_record + len(articles) > total_records:
                break
            
            start_record += len(articles)
            time.sleep(0.5)  # Be nice to the API between pages
        
        return papers
    
    def _parse_article(self, article: Dict[str, Any], year: int) -> Optional[Paper]:
        """
        Parse an IEEE Xplore API article into a Paper object.
        
        Args:
            article: API response article
            year: Publication year
            
        Returns:
            Paper object or None if parsing fails
        """
        # Extract DOI - preferred for paper ID
        doi = article.get('doi')
        
        # Extract article number if no DOI
        article_number = article.get('article_number', '')
        
        # Create paper ID
        if doi:
            paper_id = self._doi_to_paper_id(doi)
        elif article_number:
            paper_id = f"{self.venue_code.lower()}_{year}_{article_number}"
        else:
            # Generate ID from title hash
            title = article.get('title', '')
            paper_id = f"{self.venue_code.lower()}_{year}_{hash(title) % 10000:04d}"
        
        # Extract title
        title = article.get('title', '')
        if not title:
            logger.debug(f"Skipping article without title")
            return None
        
        # Clean title (remove HTML tags if present)
        title = self._clean_text(title)
        
        # Extract abstract
        abstract = article.get('abstract', '')
        abstract = self._clean_text(abstract)
        
        # Extract authors
        authors = []
        authors_data = article.get('authors', {}).get('authors', [])
        for author in authors_data:
            if isinstance(author, dict):
                full_name = author.get('full_name', '')
                if full_name:
                    authors.append(full_name)
            elif isinstance(author, str):
                authors.append(author)
        
        # Extract keywords
        keywords = []
        keywords_data = article.get('keywords', [])
        if keywords_data:
            if isinstance(keywords_data, list):
                for kw_group in keywords_data:
                    if isinstance(kw_group, dict):
                        kwd = kw_group.get('kwd', [])
                        if isinstance(kwd, list):
                            keywords.extend(kwd)
                        elif isinstance(kwd, str):
                            keywords.append(kwd)
                    elif isinstance(kw_group, str):
                        keywords.append(kw_group)
        
        # Determine PDF availability
        pdf_url = None
        download_available: Literal['openreview', 'arxiv', 'both', 'none'] = 'none'
        
        # Check if PDF is available
        has_pdf = article.get('pdf_url') or article.get('html_url')
        if has_pdf:
            # Prefer the API-provided URL; otherwise construct the IEEE stamp
            # URL from article_number (more reliable than DOI parsing).
            pdf_url = article.get('pdf_url') or self.get_pdf_url(
                paper_id,
                doi=doi,
                article_number=article_number,
            )
            download_available = 'openreview'  # Mark as available but may need subscription
        
        return Paper(
            id=paper_id,
            title=title,
            abstract=abstract,
            authors=authors,
            keywords=keywords,
            year=year,
            venue=self.venue_code,
            venue_type=cast(Literal['conference', 'journal', 'preprint'], self._venue_type),
            source_platform=self.platform_name,
            crawl_date=datetime.now().isoformat(),
            pdf_url=pdf_url,
            doi=doi,
            download_available=download_available
        )
    
    def _doi_to_paper_id(self, doi: str) -> str:
        """
        Convert DOI to a safe paper ID.
        
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
        
        This is a best-effort conversion.
        
        Args:
            paper_id: Paper identifier
            
        Returns:
            DOI string
        """
        # IEEE DOIs start with "10.1109/"
        if paper_id.startswith('10_1109_'):
            # Reconstruct DOI
            rest = paper_id[8:]  # After "10_1109_"
            # The underscore after the last part should be a dot for the document number
            parts = rest.rsplit('_', 1)
            if len(parts) == 2:
                return f"10.1109/{parts[0]}.{parts[1]}"
        
        # Fallback: try simple replacement
        return paper_id.replace('_', '/', 1).replace('_', '.', 1)
    
    def _clean_text(self, text: str) -> str:
        """
        Clean text by removing HTML tags and extra whitespace.
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        import re
        if not text:
            return ''
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Replace HTML entities
        text = text.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')
        text = text.replace('&quot;', '"').replace('&apos;', "'")
        # Normalize whitespace
        text = ' '.join(text.split())
        return text.strip()
    
    def get_pdf_url(self, paper_id: str, **kwargs) -> Optional[str]:
        """
        Get PDF URL for an IEEE paper.
        
        IEEE Xplore PDF URLs follow the pattern:
        https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber={article_number}
        
        Note: Access requires IEEE subscription for most papers.
        
        Args:
            paper_id: Paper identifier
            **kwargs: Additional parameters including 'doi' or 'article_number'
            
        Returns:
            PDF URL if available (may require authentication)
        """
        doi = kwargs.get('doi')
        article_number = kwargs.get('article_number')
        
        if article_number:
            return f"https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber={article_number}"
        elif doi:
            # Extract article number from DOI if possible
            # IEEE DOI format: 10.1109/XX.YYYY.ZZZZZ
            if '.' in doi:
                parts = doi.split('.')
                if len(parts) >= 3:
                    return f"https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber={parts[-1]}"
        
        return None
    
    def rate_limit_delay(self) -> float:
        """
        Return recommended delay between requests.
        
        IEEE Xplore API rate limits:
        - Free tier: 100 calls/day (~864 seconds between calls if used evenly)
        - We use 5.0s to be conservative and not exceed daily limit
        
        Returns:
            Delay in seconds
        """
        return 5.0
    
    def get_supported_venues(self) -> List[str]:
        """
        Return list of supported venue names.
        
        Returns:
            List of venue codes
        """
        return list(self.VENUE_MAPPINGS.keys())
    
    def check_availability(self) -> bool:
        """
        Check if the IEEE Xplore API is available.
        
        Returns:
            True if API is reachable
        """
        try:
            # Make a simple request to check if API is up
            response = requests.get(
                self.API_BASE_URL,
                params={'apikey': 'test', 'format': 'json'},
                timeout=10
            )
            # With invalid key, should get 401, but API is reachable
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
        if not config.additional_params.get('api_key') and not config.additional_params.get('ieee_api_key'):
            logger.warning(
                "IEEE Xplore API key not provided. "
                "Get one at https://developer.ieee.org/ and pass as 'api_key'"
            )
            return False
        
        # Check years
        current_year = datetime.now().year
        for year in config.years:
            if year < 1980 or year > current_year + 1:
                logger.warning(f"Year {year} may not have data available in IEEE Xplore")
        
        return True
    
    def supports_year(self, year: int) -> bool:
        """
        Check if year is supported.
        
        IEEE Xplore has papers dating back to the 1980s for these venues.
        
        Args:
            year: Year to check
            
        Returns:
            True if year is supported
        """
        current_year = datetime.now().year
        return 1980 <= year <= current_year + 1


# =============================================================================
# Specific Venue Adapters
# =============================================================================

class ICCADAdapter(IEEEXploreAdapter):
    """
    Adapter for ICCAD (International Conference on Computer-Aided Design).
    
    ICCAD is a premier conference for computer-aided design of integrated circuits.
    
    Publication Number: 10008
    """
    
    def __init__(self):
        super().__init__('ICCAD')
    
    @property
    def platform_name(self) -> str:
        """Return platform identifier."""
        return "iccad"


class DACAdapter(IEEEXploreAdapter):
    """
    Adapter for DAC (Design Automation Conference).
    
    DAC is the premier conference for electronic design automation (EDA).
    
    Publication Number: 10001
    """
    
    def __init__(self):
        super().__init__('DAC')
    
    @property
    def platform_name(self) -> str:
        """Return platform identifier."""
        return "dac"


class TCADAdapter(IEEEXploreAdapter):
    """
    Adapter for TCAD (IEEE Transactions on Computer-Aided Design).
    
    TCAD is a monthly peer-reviewed scientific journal covering computer-aided design
    of integrated circuits and systems.
    
    Publication Number: 43
    """
    
    def __init__(self):
        super().__init__('TCAD')
    
    @property
    def venue_type(self) -> str:
        """Return venue type (TCAD is a journal)."""
        return "journal"
    
    @property
    def platform_name(self) -> str:
        """Return platform identifier."""
        return "tcad"
