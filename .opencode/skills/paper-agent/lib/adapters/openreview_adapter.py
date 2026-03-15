#!/usr/bin/env python3
"""
OpenReview venue adapter.

Implements the VenueAdapter interface for crawling papers from
OpenReview-hosted conferences (ICLR, NeurIPS, ICML, AAAI, etc.).
"""

import time
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple

from .base import VenueAdapter, VenueConfig
from database import Paper


# Venue ID mappings for OpenReview
OPENREVIEW_VENUE_IDS: Dict[str, Dict[int, str]] = {
    # ICLR
    'ICLR': {
        2024: 'ICLR.cc/2024/Conference',
        2023: 'ICLR.cc/2023/Conference',
        2022: 'ICLR.cc/2022/Conference',
        2021: 'ICLR.cc/2021/Conference',
        2020: 'ICLR.cc/2020/Conference',
        2019: 'ICLR.cc/2019/Conference',
    },
    # NeurIPS
    'NeurIPS': {
        2024: 'NeurIPS.cc/2024/Conference',
        2023: 'NeurIPS.cc/2023/Conference',
        2022: 'NeurIPS.cc/2022/Conference',
        2021: 'NeurIPS.cc/2021/Conference',
        2020: 'NeurIPS.cc/2020/Conference',
    },
    # ICML
    'ICML': {
        2024: 'ICML.cc/2024/Conference',
        2023: 'ICML.cc/2023/Conference',
        2022: 'ICML.cc/2022/Conference',
        2021: 'ICML.cc/2021/Conference',
        2020: 'ICML.cc/2020/Conference',
    },
    # AAAI (some years on OpenReview)
    'AAAI': {
        2024: 'AAAI.org/2024/Conference',
        2023: 'AAAI.org/2023/Conference',
    },
}


class OpenReviewAdapter(VenueAdapter):
    """
    Adapter for OpenReview-hosted conferences.
    
    Supports conferences like ICLR, NeurIPS, ICML, and AAAI that
    use the OpenReview platform for paper submission and review.
    
    Features:
    - Automatic API version detection (v1/v2)
    - PDF URL resolution
    - Acceptance status detection
    - arXiv link detection
    
    Example:
        config = VenueConfig(
            name='ICLR 2024',
            years=[2024],
            platform='openreview',
            venue_id='ICLR.cc/2024/Conference'
        )
        
        adapter = OpenReviewAdapter()
        papers = adapter.crawl(config)
    """
    
    @property
    def platform_name(self) -> str:
        """Return platform identifier."""
        return 'openreview'
    
    @property
    def venue_type(self) -> str:
        """Return venue type."""
        return 'conference'
    
    def supports_year(self, year: int) -> bool:
        """
        Check if year is supported for any venue.
        
        OpenReview supports papers from roughly 2019 onwards.
        
        Args:
            year: Year to check
            
        Returns:
            True if year >= 2019
        """
        return year >= 2019
    
    def rate_limit_delay(self) -> float:
        """
        Return recommended delay between requests.
        
        OpenReview requests a small delay between API calls.
        
        Returns:
            1.0 second delay
        """
        return 1.0
    
    def get_client(self) -> Tuple[Any, Optional[str]]:
        """
        Initialize OpenReview API client.
        
        Attempts to use API v2 first, falls back to v1.
        
        Returns:
            Tuple of (client, api_version)
            
        Raises:
            RuntimeError: If openreview-py is not installed
        """
        try:
            import openreview
            
            # Try API v2 first
            try:
                client = openreview.api.OpenReviewClient(
                    baseurl='https://api2.openreview.net'
                )
                return client, 'v2'
            except Exception:
                # Fallback to API v1
                client = openreview.Client(
                    baseurl='https://api.openreview.net'
                )
                return client, 'v1'
                
        except ImportError:
            raise RuntimeError(
                "openreview-py not installed. "
                "Install with: pip install openreview-py"
            )
    
    def crawl(self, config: VenueConfig) -> List[Paper]:
        """
        Crawl papers from OpenReview.
        
        Args:
            config: Venue configuration
            
        Returns:
            List of Paper objects
        """
        self.validate_config(config)
        
        client, api_version = self._get_or_init_client()
        
        all_papers: List[Paper] = []
        
        for year in config.years:
            # Determine venue ID
            venue_id = self._resolve_venue_id(config, year)
            if not venue_id:
                print(f"  Warning: No venue ID found for {config.name} {year}")
                continue
            
            print(f"  Crawling: {venue_id}")
            papers = self._crawl_venue_year(
                client, venue_id, year, api_version, config.accepted_only
            )
            all_papers.extend(papers)
            
            # Rate limiting between venues
            time.sleep(self.rate_limit_delay())
        
        return all_papers
    
    def get_pdf_url(self, paper_id: str, **kwargs) -> Optional[str]:
        """
        Get PDF URL for an OpenReview paper.
        
        Args:
            paper_id: OpenReview paper ID
            **kwargs: Ignored for OpenReview
            
        Returns:
            PDF URL
        """
        return f"https://openreview.net/pdf?id={paper_id}"
    
    def _resolve_venue_id(self, config: VenueConfig, year: int) -> Optional[str]:
        """
        Resolve venue ID from config or known mappings.
        
        Args:
            config: Venue configuration
            year: Year to resolve for
            
        Returns:
            Venue ID string or None if not found
        """
        # Use explicit venue_id if provided
        if config.venue_id:
            return config.venue_id
        
        # Look up in known mappings
        venue_name = config.name.split()[0] if config.name else ''
        
        if venue_name in OPENREVIEW_VENUE_IDS:
            year_mappings = OPENREVIEW_VENUE_IDS[venue_name]
            if year in year_mappings:
                return year_mappings[year]
        
        # Try to construct from name and year
        # Format: VENUE.cc/YEAR/Conference or VENUE.org/YEAR/Conference
        if venue_name in ['ICLR', 'NeurIPS', 'ICML']:
            return f"{venue_name}.cc/{year}/Conference"
        elif venue_name == 'AAAI':
            return f"{venue_name}.org/{year}/Conference"
        
        return None
    
    def _crawl_venue_year(
        self,
        client: Any,
        venue_id: str,
        year: int,
        api_version: str,
        accepted_only: bool
    ) -> List[Paper]:
        """
        Crawl a single venue/year combination.
        
        Args:
            client: OpenReview client
            venue_id: Venue identifier
            year: Publication year
            api_version: API version ('v1' or 'v2')
            accepted_only: Only return accepted papers
            
        Returns:
            List of Paper objects
        """
        papers: List[Paper] = []
        
        try:
            # Get venue group
            venue_group = client.get_group(venue_id)
            
            # Determine submission invitation name
            if api_version == 'v2':
                submission_name = venue_group.content.get(
                    'submission_name', {}
                ).get('value', 'Submission')
            else:
                submission_name = 'Submission'
            
            # Fetch all submissions
            invitation = f'{venue_id}/-/{submission_name}'
            submissions = client.get_all_notes(
                invitation=invitation,
                details='directReplies'
            )
            
            print(f"    Found {len(submissions)} submissions")
            
            # Process each submission
            for idx, note in enumerate(submissions, 1):
                if idx % 100 == 0:
                    print(f"    Processing: {idx}/{len(submissions)}")
                
                paper = self._convert_note_to_paper(note, venue_id, year, api_version)
                
                # Skip rejected papers if accepted_only
                if accepted_only:
                    venue_status = self._get_venue_status(note, api_version)
                    if self._is_rejected(venue_status):
                        continue
                
                papers.append(paper)
                
                # Small delay to avoid rate limiting
                time.sleep(0.01)
            
            print(f"    Processed: {len(papers)} papers")
            
        except Exception as e:
            print(f"    Error crawling {venue_id}: {e}")
            import traceback
            traceback.print_exc()
        
        return papers
    
    def _convert_note_to_paper(
        self,
        note: Any,
        venue_id: str,
        year: int,
        api_version: str
    ) -> Paper:
        """
        Convert an OpenReview note to a Paper object.
        
        Args:
            note: OpenReview note object
            venue_id: Venue identifier
            year: Publication year
            api_version: API version
            
        Returns:
            Paper object
        """
        paper_id = note.id
        
        # Extract fields based on API version
        if api_version == 'v2':
            content = note.content
            title = content.get('title', {}).get('value', '')
            abstract = content.get('abstract', {}).get('value', '')
            authors = content.get('authors', {}).get('value', [])
            keywords = content.get('keywords', {}).get('value', [])
            pdf_value = content.get('pdf', {}).get('value', '')
        else:
            content = note.content
            title = content.get('title', '')
            abstract = content.get('abstract', '')
            authors = content.get('authors', [])
            keywords = content.get('keywords', [])
            pdf_value = content.get('pdf', '')
        
        # Build PDF URL
        pdf_url = self._build_pdf_url(pdf_value, paper_id)
        
        # Extract venue name from venue_id
        venue_name = venue_id.split('/')[0].replace('.cc', '').replace('.org', '')
        
        # Determine download availability
        download_available = 'none'
        if pdf_url:
            if 'arxiv' in pdf_url.lower():
                download_available = 'arxiv'
            elif 'openreview' in pdf_url.lower():
                download_available = 'openreview'
        
        # Ensure keywords is a list
        if isinstance(keywords, str):
            keywords = [keywords] if keywords else []
        elif not isinstance(keywords, list):
            keywords = []
        
        # Ensure authors is a list
        if not isinstance(authors, list):
            authors = []
        
        return Paper(
            id=paper_id,
            title=title,
            abstract=abstract,
            authors=authors,
            keywords=keywords,
            year=year,
            venue=venue_name,
            venue_type='conference',
            source_platform='openreview',
            crawl_date=datetime.now().isoformat(),
            pdf_url=pdf_url,
            download_available=download_available,
        )
    
    def _get_venue_status(self, note: Any, api_version: str) -> str:
        """
        Get venue status from note.
        
        Args:
            note: OpenReview note
            api_version: API version
            
        Returns:
            Venue status string
        """
        if api_version == 'v2':
            return note.content.get('venue', {}).get('value', '')
        return note.content.get('venue', '')
    
    def _is_rejected(self, venue_status: str) -> bool:
        """
        Check if paper was rejected based on venue status.
        
        Args:
            venue_status: Venue status string
            
        Returns:
            True if paper was rejected
        """
        if not venue_status:
            return False
        
        status_lower = venue_status.lower()
        
        # Check for acceptance indicators
        if any(x in status_lower for x in ['accept', 'oral', 'poster', 'spotlight']):
            return False
        
        # Check for rejection
        if 'reject' in status_lower:
            return True
        
        return False
    
    def _build_pdf_url(self, pdf_value: str, paper_id: str) -> Optional[str]:
        """
        Build PDF URL from pdf field value.
        
        Args:
            pdf_value: Value from note's pdf field
            paper_id: Paper ID for fallback URL
            
        Returns:
            PDF URL or None
        """
        if not pdf_value:
            return None
        
        if pdf_value.startswith('/pdf'):
            return f"https://openreview.net{pdf_value}"
        elif pdf_value.startswith('http'):
            return pdf_value
        else:
            return f"https://openreview.net/pdf?id={paper_id}"