#!/usr/bin/env python3
"""
Venue Adapters Package

Provides a plugin-based architecture for crawling papers from different
academic platforms (OpenReview, ACL, IEEE, Nature, etc.).

Usage:
    from adapters import VenueConfig, VenueAdapter, AdapterRegistry
    
    # Get an adapter
    adapter = AdapterRegistry.get('openreview')
    
    # Crawl papers
    config = VenueConfig(
        name='ICLR 2024',
        years=[2024],
        platform='openreview'
    )
    papers = adapter.crawl(config)
    
    # Or use convenience method
    papers = AdapterRegistry.crawl_venue(config)
"""

from .base import VenueAdapter, VenueConfig
from .registry import AdapterRegistry, register_builtin_adapters

# Initialize built-in adapters on import
register_builtin_adapters()

__all__ = [
    # Base classes
    'VenueAdapter',
    'VenueConfig',
    
    # Registry
    'AdapterRegistry',
    'register_builtin_adapters',
]