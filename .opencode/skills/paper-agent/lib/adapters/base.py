#!/usr/bin/env python3
"""
Base Adapter Classes for Paper Agent

Provides abstract base class for venue adapters and configuration dataclasses.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from database import Paper


@dataclass
class VenueConfig:
    """
    Configuration for a venue crawler.
    
    Attributes:
        name: Human-readable venue name (e.g., 'ICLR 2024')
        years: List of years to crawl
        platform: Platform identifier (e.g., 'openreview', 'acm', 'ieee', 'nature', 'arxiv')
        venue_id: Platform-specific venue identifier (e.g., 'ICLR.cc/2024/Conference')
        additional_params: Extra parameters specific to the platform
        accepted_only: Only crawl accepted papers (where applicable)
    """
    name: str
    years: List[int] = field(default_factory=list)
    platform: str = ''
    venue_id: Optional[str] = None
    additional_params: Dict[str, Any] = field(default_factory=dict)
    accepted_only: bool = True
    
    def __post_init__(self):
        """Validate and set defaults."""
        if not self.name:
            raise ValueError("Venue name is required")
        if not self.years:
            raise ValueError("At least one year is required")
        
        if self.additional_params is None:
            self.additional_params = {}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VenueConfig':
        """
        Create VenueConfig from dictionary.
        
        Args:
            data: Dictionary with configuration data
            
        Returns:
            VenueConfig instance
        """
        return cls(
            name=data.get('name', ''),
            years=data.get('years', []),
            platform=data.get('platform', ''),
            venue_id=data.get('venue_id'),
            additional_params=data.get('additional_params', {}),
            accepted_only=data.get('accepted_only', True)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'name': self.name,
            'years': self.years,
            'platform': self.platform,
            'venue_id': self.venue_id,
            'additional_params': self.additional_params,
            'accepted_only': self.accepted_only
        }


class VenueAdapter(ABC):
    """
    Abstract base class for all venue adapters.
    
    A venue adapter is responsible for:
    - Crawling papers from a specific platform (OpenReview, ACL, etc.)
    - Converting platform-specific data to the Paper model
    - Providing PDF download URLs
    
    Subclasses must implement:
    - platform_name property
    - venue_type property
    - crawl() method
    - get_pdf_url() method
    
    Optional overrides:
    - supports_year() for year validation
    - rate_limit_delay() for rate limiting
    - get_client() for platform-specific client initialization
    """
    
    # Platform-specific client (cached)
    _client: Optional[Any] = None
    _api_version: Optional[str] = None
    
    @property
    @abstractmethod
    def platform_name(self) -> str:
        """
        Return platform identifier.
        
        Should be lowercase, matching the 'platform' field in VenueConfig.
        Examples: 'openreview', 'acl', 'cvpr', 'ieee', 'nature', 'arxiv'
        
        Returns:
            Platform identifier string
        """
        pass
    
    @property
    @abstractmethod
    def venue_type(self) -> str:
        """
        Return venue type.
        
        Returns:
            'conference', 'journal', or 'preprint'
        """
        pass
    
    @abstractmethod
    def crawl(self, config: VenueConfig) -> List[Paper]:
        """
        Crawl papers from the venue.
        
        This is the main entry point for crawling. It should:
        1. Initialize platform client if needed
        2. Query papers for the specified venue/years
        3. Convert to Paper objects
        4. Apply accepted_only filter if configured
        
        Args:
            config: Venue configuration with name, years, and platform
            
        Returns:
            List of Paper objects
            
        Raises:
            RuntimeError: If platform client cannot be initialized
            ValueError: If configuration is invalid
        """
        pass
    
    @abstractmethod
    def get_pdf_url(self, paper_id: str, **kwargs) -> Optional[str]:
        """
        Get PDF download URL for a paper.
        
        Args:
            paper_id: Paper identifier (platform-specific)
            **kwargs: Additional parameters (e.g., arxiv_id, doi)
            
        Returns:
            PDF URL if available, None otherwise
        """
        pass
    
    def supports_year(self, year: int) -> bool:
        """
        Check if adapter supports crawling a specific year.
        
        Override this method to restrict which years can be crawled.
        Default implementation allows all years.
        
        Args:
            year: Year to check
            
        Returns:
            True if year is supported, False otherwise
        """
        return True
    
    def rate_limit_delay(self) -> float:
        """
        Return recommended delay between requests in seconds.
        
        Override this method to customize rate limiting.
        Default is 1.0 second.
        
        Returns:
            Delay in seconds between API calls
        """
        return 1.0
    
    def get_client(self) -> Tuple[Any, Optional[str]]:
        """
        Get or create platform-specific API client.
        
        Override this method to initialize the platform client.
        The client will be cached for reuse.
        
        Returns:
            Tuple of (client, api_version) where api_version is
            platform-specific (e.g., 'v1', 'v2' for OpenReview)
            
        Raises:
            RuntimeError: If client cannot be initialized
        """
        return None, None
    
    def _get_or_init_client(self) -> Tuple[Any, Optional[str]]:
        """
        Get cached client or initialize a new one.
        
        Internal method for client caching.
        
        Returns:
            Tuple of (client, api_version)
        """
        if self._client is None:
            self._client, self._api_version = self.get_client()
        return self._client, self._api_version
    
    def build_venue_id(self, venue_name: str, year: int) -> str:
        """
        Build platform-specific venue ID from venue name and year.
        
        Override this method to customize venue ID construction.
        
        Args:
            venue_name: Short venue name (e.g., 'ICLR', 'NeurIPS')
            year: Publication year
            
        Returns:
            Platform-specific venue ID
        """
        return f"{venue_name}.{year}"
    
    def validate_config(self, config: VenueConfig) -> bool:
        """
        Validate venue configuration.
        
        Args:
            config: Venue configuration to validate
            
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        if not config.name:
            raise ValueError("Venue name is required")
        if not config.years:
            raise ValueError("At least one year is required")
        
        # Check platform match if specified
        if config.platform and config.platform != self.platform_name:
            raise ValueError(
                f"Platform mismatch: expected '{self.platform_name}', "
                f"got '{config.platform}'"
            )
        
        # Check year support
        for year in config.years:
            if not self.supports_year(year):
                raise ValueError(f"Year {year} is not supported by {self.platform_name}")
        
        return True
    
    def get_supported_venues(self) -> List[str]:
        """
        Return list of supported venue names.
        
        Override in subclasses to return specific venues.
        """
        return []
    
    def check_availability(self) -> bool:
        """
        Check if the platform is available.
        
        Override to check API keys, network connectivity, etc.
        
        Returns:
            True if the adapter can be used
        """
        return True
    
    def __repr__(self) -> str:
        """String representation of the adapter."""
        return f"{self.__class__.__name__}(platform='{self.platform_name}', type='{self.venue_type}')"