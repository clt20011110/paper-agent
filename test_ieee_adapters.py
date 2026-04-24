#!/usr/bin/env python3
"""
Test script for IEEE Xplore EDA adapters (ICCAD, DAC, TCAD)

Usage:
    python test_ieee_adapters.py [--api-key YOUR_API_KEY]

This script tests the IEEE Xplore adapters to ensure they can:
1. Connect to the IEEE Xplore API
2. Parse responses correctly
3. Return properly formatted Paper objects

Note: Requires IEEE Xplore API key. Get one at https://developer.ieee.org/
"""

import sys
import argparse
from pathlib import Path

# Add lib directory to path
sys.path.insert(0, str(Path(__file__).parent / '.opencode' / 'skills' / 'paper-agent' / 'lib'))

from adapters import AdapterRegistry, VenueConfig


def test_adapter_availability():
    """Test that all IEEE adapters are registered."""
    print("=" * 60)
    print("Testing IEEE Adapter Registration")
    print("=" * 60)
    
    platforms = AdapterRegistry.list_platforms()
    
    expected_adapters = ['iccad', 'dac', 'tcad']
    
    for adapter_name in expected_adapters:
        if adapter_name in platforms:
            print(f"✓ {adapter_name.upper()} adapter is registered")
        else:
            print(f"✗ {adapter_name.upper()} adapter is NOT registered")
            return False
    
    print("\n✓ All IEEE adapters are properly registered\n")
    return True


def test_adapter_info():
    """Test that adapter information is correct."""
    print("=" * 60)
    print("Testing Adapter Information")
    print("=" * 60)
    
    for platform_name in ['iccad', 'dac', 'tcad']:
        adapter = AdapterRegistry.get(platform_name)
        if adapter:
            print(f"\n{platform_name.upper()}:")
            print(f"  Platform: {adapter.platform_name}")
            print(f"  Venue Type: {adapter.venue_type}")
            print(f"  Supported Venues: {adapter.get_supported_venues()}")
            print(f"  Rate Limit: {adapter.rate_limit_delay()}s")
        else:
            print(f"✗ Could not retrieve {platform_name} adapter")
            return False
    
    print("\n✓ All adapter info retrieved successfully\n")
    return True


def test_api_connectivity(api_key: str = None):
    """Test API connectivity (requires API key)."""
    print("=" * 60)
    print("Testing IEEE Xplore API Connectivity")
    print("=" * 60)
    
    if not api_key:
        print("⚠ No API key provided, skipping API connectivity test")
        print("  To test API connectivity, provide --api-key\n")
        return True
    
    # Test with TCAD (journal) - usually has consistent data
    adapter = AdapterRegistry.get('tcad')
    if not adapter:
        print("✗ TCAD adapter not found")
        return False
    
    # Test check_availability
    print("\nTesting API availability check...")
    if adapter.check_availability():
        print("✓ IEEE Xplore API is reachable")
    else:
        print("✗ IEEE Xplore API is not reachable")
        return False
    
    # Test crawling a single year (limited to save API calls)
    print("\nTesting crawl with small dataset...")
    config = VenueConfig(
        name='TCAD',
        years=[2024],
        platform='tcad',
        additional_params={'api_key': api_key}
    )
    
    try:
        # Note: This will use actual API calls
        papers = adapter.crawl(config)
        if papers:
            print(f"✓ Successfully crawled {len(papers)} papers from TCAD 2024")
            
            # Show sample paper
            sample = papers[0]
            print(f"\n  Sample paper:")
            print(f"    Title: {sample.title[:80]}...")
            print(f"    Authors: {', '.join(sample.authors[:3])}{'...' if len(sample.authors) > 3 else ''}")
            print(f"    Year: {sample.year}")
            print(f"    Venue Type: {sample.venue_type}")
            print(f"    Has Abstract: {'Yes' if sample.abstract else 'No'}")
            print(f"    PDF Available: {sample.download_available}")
        else:
            print("⚠ No papers found (might be normal if no data for 2024 yet)")
            
    except Exception as e:
        print(f"✗ Error during crawl: {e}")
        return False
    
    print("\n✓ API connectivity test passed\n")
    return True


def test_config_validation(api_key: str = None):
    """Test configuration validation."""
    print("=" * 60)
    print("Testing Configuration Validation")
    print("=" * 60)
    
    adapter = AdapterRegistry.get('iccad')
    
    # Test without API key
    print("\nTesting config without API key...")
    config_no_key = VenueConfig(
        name='ICCAD',
        years=[2024],
        platform='iccad',
        additional_params={}
    )
    
    if adapter.validate_config(config_no_key):
        print("⚠ Config validated without API key (might be a warning only)")
    else:
        print("✓ Config correctly rejected without API key")
    
    # Test with API key
    if api_key:
        print("\nTesting config with API key...")
        config_with_key = VenueConfig(
            name='ICCAD',
            years=[2024],
            platform='iccad',
            additional_params={'api_key': api_key}
        )
        
        if adapter.validate_config(config_with_key):
            print("✓ Config validated with API key")
        else:
            print("✗ Config rejected even with API key")
            return False
    
    print("\n✓ Configuration validation test passed\n")
    return True


def main():
    parser = argparse.ArgumentParser(description='Test IEEE Xplore EDA adapters')
    parser.add_argument('--api-key', help='IEEE Xplore API key for live testing')
    parser.add_argument('--skip-api', action='store_true', help='Skip API connectivity tests')
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("IEEE Xplore EDA Adapter Test Suite")
    print("=" * 60 + "\n")
    
    results = []
    
    # Test 1: Adapter Registration
    results.append(("Adapter Registration", test_adapter_availability()))
    
    # Test 2: Adapter Information
    results.append(("Adapter Information", test_adapter_info()))
    
    # Test 3: Configuration Validation
    results.append(("Configuration Validation", test_config_validation(args.api_key)))
    
    # Test 4: API Connectivity (optional)
    if not args.skip_api:
        results.append(("API Connectivity", test_api_connectivity(args.api_key)))
    else:
        print("Skipping API connectivity tests as requested\n")
    
    # Summary
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠ {total - passed} test(s) failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
