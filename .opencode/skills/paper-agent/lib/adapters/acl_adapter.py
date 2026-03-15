#!/usr/bin/env python3
"""
ACL Anthology Adapter using acl-anthology Python library

Implements venue adapter for ACL Anthology, supporting:
- ACL (Annual Meeting of the Association for Computational Linguistics)
- EMNLP (Conference on Empirical Methods in Natural Language Processing)
- NAACL (North American Chapter of the ACL)
- EACL (European Chapter of the ACL)
- COLING (International Conference on Computational Linguistics)
- CoNLL (Conference on Computational Natural Language Learning)
- TACL (Transactions of the Association for Computational Linguistics)
- CL (Computational Linguistics Journal)
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

from database import Paper
from .base import VenueAdapter, VenueConfig
from .registry import AdapterRegistry


logger = logging.getLogger(__name__)


@AdapterRegistry.register
class ACLAdapter(VenueAdapter):
    """
    Adapter for ACL Anthology (Association for Computational Linguistics).
    
    Uses the official acl-anthology Python library to access paper metadata.
    All papers are open access and freely downloadable.
    
    Documentation: https://acl-anthology.readthedocs.io/
    GitHub: https://github.com/acl-org/acl-anthology
    """
    
    # Mapping of common conference/journal names to ACL Anthology venue IDs
    VENUE_MAPPING = {
        'ACL': 'acl',
        'EMNLP': 'emnlp',
        'NAACL': 'naacl',
        'EACL': 'eacl',
        'COLING': 'coling',
        'CoNLL': 'conll',
        'TACL': 'tacl',  # Journal
        'CL': 'cl',      # Journal (Computational Linguistics)
        'AACL': 'aacl',
        'IJCNLP': 'ijcnlp',
        'LREC': 'lrec',
        'SEM': 'sem',
        'WS': 'ws',      # Workshops
    }
    
    @property
    def platform_name(self) -> str:
        """Return the platform identifier."""
        return "acl"
    
    @property
    def venue_type(self) -> str:
        """Return the venue type (can be conference or journal)."""
        return "conference"  # Default, overridden per venue
    
    def crawl(self, config: VenueConfig) -> List[Paper]:
        """
        Crawl ACL papers for given years.
        
        Args:
            config: VenueConfig with name (e.g., 'ACL', 'EMNLP', 'NAACL') and years
            
        Returns:
            List of Paper objects
            
        Raises:
            ImportError: If acl-anthology package is not installed
            ValueError: If venue name is not recognized
        """
        try:
            from anthology import Anthology
        except ImportError:
            raise ImportError(
                "acl-anthology package required. "
                "Install with: pip install acl-anthology"
            )
        
        papers = []
        anthology = Anthology()
        
        # Validate venue name
        venue_id = self._map_venue_name(config.name)
        if venue_id is None:
            raise ValueError(
                f"Unknown venue: {config.name}. "
                f"Supported venues: {list(self.VENUE_MAPPING.keys())}"
            )
        
        for year in config.years:
            logger.info(f"Crawling {config.name} {year}...")
            try:
                venue_papers = self._crawl_venue_year(
                    anthology, venue_id, year, config.name, config.additional_params
                )
                papers.extend(venue_papers)
                logger.info(f"Found {len(venue_papers)} papers for {config.name} {year}")
            except Exception as e:
                logger.error(f"Error crawling {config.name} {year}: {e}")
                # Continue with other years instead of failing completely
        
        return papers
    
    def _map_venue_name(self, name: str) -> Optional[str]:
        """
        Map common conference names to ACL Anthology venue IDs.
        
        Args:
            name: Venue name (e.g., 'ACL', 'EMNLP')
            
        Returns:
            ACL Anthology venue ID or None if not found
        """
        return self.VENUE_MAPPING.get(name.upper())
    
    def _crawl_venue_year(
        self,
        anthology,
        venue_id: str,
        year: int,
        venue_name: str,
        options: Dict[str, Any]
    ) -> List[Paper]:
        """
        Crawl papers for a specific venue and year.
        
        Args:
            anthology: Anthology instance
            venue_id: ACL Anthology venue ID
            year: Publication year
            venue_name: Display name for the venue
            options: Additional options (e.g., accepted_only)
            
        Returns:
            List of Paper objects
        """
        papers = []
        
        try:
            # Get venue from anthology
            # The anthology library provides access via venues index
            venues = anthology.venues
            
            # Find volumes for this venue and year
            # ACL IDs typically follow pattern: {year}.{venue}.{type}.{number}
            # e.g., 2024.acl-long.1, 2024.acl-short.5
            
            year_str = str(year)
            
            # Iterate through all papers and filter by venue and year
            for paper in anthology.papers:
                paper_id = paper.id
                
                # Check if paper matches venue and year
                if not paper_id.startswith(f"{year_str}."):
                    continue
                
                # Check venue match
                if not self._paper_matches_venue(paper_id, venue_id):
                    continue
                
                # Extract paper metadata
                paper_obj = self._create_paper_from_anthology(
                    paper, venue_name, year
                )
                
                if paper_obj:
                    papers.append(paper_obj)
                    
        except AttributeError:
            # If the anthology structure is different, try alternative approach
            papers = self._crawl_venue_year_alternative(
                anthology, venue_id, year, venue_name, options
            )
        
        return papers
    
    def _paper_matches_venue(self, paper_id: str, venue_id: str) -> bool:
        """
        Check if a paper ID matches the venue.
        
        ACL paper IDs follow pattern: {year}.{venue}-{type}.{number}
        e.g., 2024.acl-long.1
        
        Args:
            paper_id: ACL paper ID
            venue_id: Venue ID to match
            
        Returns:
            True if the paper matches the venue
        """
        parts = paper_id.split('.')
        if len(parts) >= 2:
            venue_part = parts[1].split('-')[0]  # Get venue before type
            return venue_part.lower() == venue_id.lower()
        return False
    
    def _crawl_venue_year_alternative(
        self,
        anthology,
        venue_id: str,
        year: int,
        venue_name: str,
        options: Dict[str, Any]
    ) -> List[Paper]:
        """
        Alternative crawling method using volume-based access.
        
        This is a fallback if the papers iterator is not available.
        
        Args:
            anthology: Anthology instance
            venue_id: ACL Anthology venue ID
            year: Publication year
            venue_name: Display name for the venue
            options: Additional options
            
        Returns:
            List of Paper objects
        """
        papers = []
        year_str = str(year)
        
        try:
            # Try to access via volumes
            for volume in anthology.volumes:
                volume_id = volume.id
                
                # Check if volume matches year and venue
                if not volume_id.startswith(f"{year_str}."):
                    continue
                
                if not self._paper_matches_venue(volume_id, venue_id):
                    continue
                
                # Get papers from this volume
                for paper in volume.papers:
                    paper_obj = self._create_paper_from_anthology(
                        paper, venue_name, year
                    )
                    if paper_obj:
                        papers.append(paper_obj)
                        
        except Exception as e:
            logger.warning(f"Alternative crawl also failed: {e}")
        
        return papers
    
    def _create_paper_from_anthology(
        self,
        paper,
        venue_name: str,
        year: int
    ) -> Optional[Paper]:
        """
        Create a Paper object from an ACL Anthology paper object.
        
        Args:
            paper: ACL Anthology paper object
            venue_name: Venue display name
            year: Publication year
            
        Returns:
            Paper object or None if creation fails
        """
        try:
            # Extract basic metadata
            paper_id = paper.id
            title = paper.title if hasattr(paper, 'title') else ""
            
            # Handle abstract (may not always be present)
            abstract = ""
            if hasattr(paper, 'abstract') and paper.abstract:
                abstract = paper.abstract
            
            # Extract authors
            authors = []
            if hasattr(paper, 'authors'):
                for author in paper.authors:
                    if hasattr(author, 'name'):
                        authors.append(author.name)
                    elif isinstance(author, str):
                        authors.append(author)
            
            # Get PDF URL
            pdf_url = self.get_pdf_url(paper_id)
            
            # Get DOI if available
            doi = None
            if hasattr(paper, 'doi') and paper.doi:
                doi = paper.doi
            
            # Get BibTeX
            bibtex = None
            if hasattr(paper, 'bibtex') and paper.bibtex:
                bibtex = paper.bibtex
            
            # Determine venue type
            venue_type = self._get_venue_type(venue_name)
            
            # Create Paper object
            return Paper(
                id=paper_id,
                title=title,
                abstract=abstract,
                authors=authors,
                keywords=[],  # ACL doesn't provide keywords directly
                year=year,
                venue=venue_name,
                venue_type=venue_type,
                source_platform='acl',
                crawl_date=datetime.now().isoformat(),
                pdf_url=pdf_url,
                doi=doi,
                bibtex=bibtex,
                download_available='acl',  # All ACL papers are open access
            )
            
        except Exception as e:
            logger.warning(f"Failed to create paper from anthology entry: {e}")
            return None
    
    def _get_venue_type(self, venue_name: str) -> str:
        """
        Determine if venue is a conference or journal.
        
        Args:
            venue_name: Venue name
            
        Returns:
            'conference' or 'journal'
        """
        journals = ['TACL', 'CL']  # Transactions of ACL and Computational Linguistics
        if venue_name.upper() in journals:
            return 'journal'
        return 'conference'
    
    def get_pdf_url(self, paper_id: str, **kwargs) -> Optional[str]:
        """
        Get PDF URL for an ACL paper.
        
        ACL papers follow pattern: https://aclanthology.org/{id}.pdf
        
        Args:
            paper_id: ACL Anthology ID (e.g., '2024.acl-long.1')
            
        Returns:
            Direct PDF URL
        """
        if not paper_id:
            return None
            
        # Remove .pdf suffix if already present
        if paper_id.endswith('.pdf'):
            paper_id = paper_id[:-4]
        
        return f"https://aclanthology.org/{paper_id}.pdf"
    
    def get_bibtex_url(self, paper_id: str) -> Optional[str]:
        """
        Get BibTeX URL for an ACL paper.
        
        Args:
            paper_id: ACL Anthology ID
            
        Returns:
            BibTeX URL
        """
        if not paper_id:
            return None
            
        if paper_id.endswith('.bib'):
            paper_id = paper_id[:-4]
        
        return f"https://aclanthology.org/{paper_id}.bib"
    
    def rate_limit_delay(self) -> float:
        """
        Return recommended delay between requests.
        
        ACL Anthology doesn't have strict rate limits, but recommends
        being courteous with delays.
        
        Returns:
            Delay in seconds
        """
        return 1.0
    
    def get_supported_venues(self) -> List[str]:
        """
        Return list of supported venue names.
        
        Returns:
            List of venue names
        """
        return list(self.VENUE_MAPPING.keys())
    
    def check_availability(self) -> bool:
        """
        Check if the acl-anthology library is available.
        
        Returns:
            True if the library can be imported
        """
        try:
            from anthology import Anthology
            return True
        except ImportError:
            return False
    
    def validate_config(self, config: VenueConfig) -> bool:
        """
        Validate the configuration for this adapter.
        
        Args:
            config: VenueConfig to validate
            
        Returns:
            True if configuration is valid
        """
        # Check venue name
        if config.name.upper() not in self.VENUE_MAPPING:
            logger.warning(
                f"Unknown venue: {config.name}. "
                f"Supported: {list(self.VENUE_MAPPING.keys())}"
            )
            return False
        
        # Check years (ACL Anthology has papers from 1965 onwards)
        for year in config.years:
            if year < 1965 or year > datetime.now().year + 1:
                logger.warning(f"Year {year} may not be available in ACL Anthology")
                # Still return True as it's just a warning
        
        return True