#!/usr/bin/env python3
"""
Adapter Registry for Paper Agent

Provides a registry pattern for venue adapters with automatic registration.
"""

from typing import Dict, Type, Optional, List, Any
import sys
from pathlib import Path

# Add both lib directory and adapters directory to path for imports
lib_dir = str(Path(__file__).parent.parent)
adapters_dir = str(Path(__file__).parent)
if lib_dir not in sys.path:
    sys.path.insert(0, lib_dir)
if adapters_dir not in sys.path:
    sys.path.insert(0, adapters_dir)

from database import Paper
from base import VenueAdapter, VenueConfig


class AdapterRegistry:
    """
    Registry for venue adapters.
    
    Adapters are registered using the @AdapterRegistry.register decorator.
    The registry provides methods to:
        - Register adapters by platform name
        - Retrieve adapters by platform name
        - List all registered adapters
        - Find adapters that support a specific venue
        - Crawl venues directly using registered adapters
    
    Example:
        @AdapterRegistry.register
        class ACLAdapter(VenueAdapter):
            @property
            def platform_name(self) -> str:
                return "acl"
            # ...
        
        # Later:
        adapter = AdapterRegistry.get("acl")
    """
    
    _adapters: Dict[str, Type[VenueAdapter]] = {}
    _initialized: bool = False
    
    @classmethod
    def register(cls, adapter_class: Type[VenueAdapter]) -> Type[VenueAdapter]:
        """
        Decorator to register an adapter class.
        
        Args:
            adapter_class: The adapter class to register
            
        Returns:
            The same adapter class (for decorator chaining)
            
        Raises:
            ValueError: If platform_name is not defined or already registered
        """
        # Create a temporary instance to get platform_name
        instance = adapter_class()
        platform_name = instance.platform_name
        
        if not platform_name:
            raise ValueError(
                f"Adapter {adapter_class.__name__} must define platform_name property"
            )
        
        if platform_name in cls._adapters:
            raise ValueError(
                f"Platform '{platform_name}' is already registered by "
                f"{cls._adapters[platform_name].__name__}"
            )
        
        cls._adapters[platform_name] = adapter_class
        return adapter_class
    
    @classmethod
    def register_adapter(cls, adapter: VenueAdapter) -> None:
        """
        Register an adapter instance directly.
        
        Args:
            adapter: Adapter instance to register
        """
        cls._adapters[adapter.platform_name] = adapter.__class__
    
    @classmethod
    def unregister(cls, platform_name: str) -> bool:
        """
        Unregister an adapter by platform name.
        
        Args:
            platform_name: Platform name to unregister
            
        Returns:
            True if adapter was removed, False if not found
        """
        if platform_name in cls._adapters:
            del cls._adapters[platform_name]
            return True
        return False
    
    @classmethod
    def get(cls, platform_name: str) -> Optional[VenueAdapter]:
        """
        Get an adapter instance by platform name.
        
        Args:
            platform_name: The platform identifier (e.g., 'acl', 'openreview')
            
        Returns:
            Adapter instance or None if not found
        """
        cls._ensure_initialized()
        adapter_class = cls._adapters.get(platform_name)
        if adapter_class:
            return adapter_class()
        return None
    
    @classmethod
    def get_required(cls, platform_name: str) -> VenueAdapter:
        """
        Get adapter by platform name, raising error if not found.
        
        Args:
            platform_name: Platform identifier
            
        Returns:
            VenueAdapter instance
            
        Raises:
            ValueError: If no adapter registered for the platform
        """
        adapter = cls.get(platform_name)
        if not adapter:
            raise ValueError(f"No adapter registered for platform: {platform_name}")
        return adapter
    
    @classmethod
    def get_class(cls, platform_name: str) -> Optional[Type[VenueAdapter]]:
        """
        Get an adapter class by platform name.
        
        Args:
            platform_name: The platform identifier
            
        Returns:
            Adapter class or None if not found
        """
        return cls._adapters.get(platform_name)
    
    @classmethod
    def list_platforms(cls) -> List[str]:
        """
        List all registered platform names.
        
        Returns:
            List of platform identifiers
        """
        cls._ensure_initialized()
        return list(cls._adapters.keys())
    
    @classmethod
    def list_adapters(cls) -> List[Type[VenueAdapter]]:
        """
        List all registered adapter classes.
        
        Returns:
            List of adapter classes
        """
        cls._ensure_initialized()
        return list(cls._adapters.values())
    
    @classmethod
    def list_adapters_info(cls) -> List[Dict[str, Any]]:
        """
        Get detailed information about all registered adapters.
        
        Returns:
            List of dictionaries with adapter info
        """
        cls._ensure_initialized()
        info = []
        for adapter_class in cls._adapters.values():
            adapter = adapter_class()
            info.append({
                'platform': adapter.platform_name,
                'venue_type': adapter.venue_type,
                'supported_venues': adapter.get_supported_venues(),
                'rate_limit': adapter.rate_limit_delay(),
                'class_name': adapter_class.__name__,
            })
        return info
    
    @classmethod
    def list_by_type(cls, venue_type: str) -> List[VenueAdapter]:
        """
        List adapters by venue type.
        
        Args:
            venue_type: Type to filter by ('conference', 'journal', or 'preprint')
            
        Returns:
            List of adapters matching the type
        """
        cls._ensure_initialized()
        result = []
        for adapter_class in cls._adapters.values():
            adapter = adapter_class()
            if adapter.venue_type == venue_type:
                result.append(adapter)
        return result
    
    @classmethod
    def find_adapter_for_venue(cls, venue_name: str) -> Optional[VenueAdapter]:
        """
        Find an adapter that supports a specific venue.
        
        Checks each adapter's get_supported_venues() method.
        
        Args:
            venue_name: The venue name (e.g., 'ACL', 'EMNLP', 'ICLR')
            
        Returns:
            First adapter that supports the venue, or None
        """
        cls._ensure_initialized()
        venue_upper = venue_name.upper()
        
        for adapter_class in cls._adapters.values():
            adapter = adapter_class()
            supported = adapter.get_supported_venues()
            if venue_upper in [v.upper() for v in supported]:
                return adapter
        
        return None
    
    @classmethod
    def crawl_venue(cls, config: VenueConfig) -> List[Paper]:
        """
        Convenience method to crawl a venue using registered adapter.
        
        Args:
            config: Venue configuration with platform name
            
        Returns:
            List of Paper objects
            
        Raises:
            ValueError: If no adapter registered for the platform
        """
        adapter = cls.get_required(config.platform)
        return adapter.crawl(config)
    
    @classmethod
    def get_pdf_url(cls, platform: str, paper_id: str, **kwargs) -> Optional[str]:
        """
        Get PDF URL for a paper using the appropriate adapter.
        
        Args:
            platform: Platform identifier
            paper_id: Paper identifier
            **kwargs: Additional parameters
            
        Returns:
            PDF URL if available, None otherwise
            
        Raises:
            ValueError: If no adapter registered for the platform
        """
        adapter = cls.get_required(platform)
        return adapter.get_pdf_url(paper_id, **kwargs)
    
    @classmethod
    def supports_platform(cls, platform_name: str) -> bool:
        """
        Check if a platform is supported.
        
        Args:
            platform_name: Platform identifier to check
            
        Returns:
            True if platform has a registered adapter
        """
        cls._ensure_initialized()
        return platform_name in cls._adapters
    
    @classmethod
    def clear(cls) -> None:
        """
        Clear all registered adapters.
        
        Useful for testing.
        """
        cls._adapters.clear()
        cls._initialized = False
    
    @classmethod
    def is_registered(cls, platform_name: str) -> bool:
        """
        Check if a platform is registered.
        
        Args:
            platform_name: The platform identifier
            
        Returns:
            True if registered
        """
        return platform_name in cls._adapters
    
    @classmethod
    def get_adapter_info(cls) -> Dict[str, Dict]:
        """
        Get information about all registered adapters.
        
        Returns:
            Dictionary mapping platform names to adapter info
        """
        cls._ensure_initialized()
        info = {}
        for platform_name, adapter_class in cls._adapters.items():
            adapter = adapter_class()
            info[platform_name] = {
                'class_name': adapter_class.__name__,
                'venue_type': adapter.venue_type,
                'supported_venues': adapter.get_supported_venues(),
                'rate_limit': adapter.rate_limit_delay(),
            }
        return info
    
    @classmethod
    def _ensure_initialized(cls) -> None:
        """
        Ensure built-in adapters are registered.
        
        This lazy initialization avoids circular import issues.
        """
        if cls._initialized:
            return
        
        cls._initialized = True
        
        # Register built-in adapters lazily using relative imports
        try:
            from .openreview_adapter import OpenReviewAdapter
            cls._adapters[OpenReviewAdapter().platform_name] = OpenReviewAdapter
        except ImportError:
            pass
        
        # Register ACL adapter
        try:
            from .acl_adapter import ACLAdapter
            cls._adapters[ACLAdapter().platform_name] = ACLAdapter
        except ImportError:
            pass
        
        # Register Nature journal adapters
        try:
            from .nature_adapter import (
                NatureMachineIntelligenceAdapter,
                NatureChemistryAdapter,
                NatureCommunicationsAdapter,
                NatureMainAdapter
            )
            cls._adapters[NatureMachineIntelligenceAdapter().platform_name] = NatureMachineIntelligenceAdapter
            cls._adapters[NatureChemistryAdapter().platform_name] = NatureChemistryAdapter
            cls._adapters[NatureCommunicationsAdapter().platform_name] = NatureCommunicationsAdapter
            cls._adapters[NatureMainAdapter().platform_name] = NatureMainAdapter
        except ImportError:
            pass
        
        # Register CVF adapters (CVPR, ICCV)
        try:
            from .cvf_adapter import CVPRAdapter, ICCVAdapter
            cls._adapters[CVPRAdapter().platform_name] = CVPRAdapter
            cls._adapters[ICCVAdapter().platform_name] = ICCVAdapter
        except ImportError:
            pass
        
        # Register arXiv adapter
        try:
            from .arxiv_adapter import ArxivAdapter
            cls._adapters[ArxivAdapter().platform_name] = ArxivAdapter
        except ImportError:
            pass


def register_builtin_adapters() -> None:
    """
    Register all built-in adapters.
    
    This function imports adapter modules to trigger their registration.
    Call this before using AdapterRegistry.get() to ensure all adapters are loaded.
    """
    AdapterRegistry._ensure_initialized()