#!/usr/bin/env python3
"""
Test script for DBLP EDA adapters (ICCAD, DAC, TCAD)
NO API KEY REQUIRED - DBLP is completely free and open

Usage:
    python test_dblp_adapters.py

This script tests the DBLP adapters to verify:
1. Adapters are properly registered
2. Can connect to DBLP API without authentication
3. Can fetch and parse ICCAD/DAC paper metadata

Note: DBLP does not provide abstracts, only bibliographic metadata
"""

import sys
from pathlib import Path

# Add lib directory to path
sys.path.insert(0, str(Path(__file__).parent / '.opencode' / 'skills' / 'paper-agent' / 'lib'))

from adapters import AdapterRegistry, VenueConfig


def test_adapter_registration():
    """Test that all DBLP adapters are registered."""
    print("=" * 60)
    print("Testing DBLP Adapter Registration")
    print("=" * 60)
    
    platforms = AdapterRegistry.list_platforms()
    
    expected_adapters = ['dblp_iccad', 'dblp_dac', 'dblp_tcad']
    
    for adapter_name in expected_adapters:
        if adapter_name in platforms:
            print(f"✓ {adapter_name.upper()} adapter is registered")
        else:
            print(f"✗ {adapter_name.upper()} adapter is NOT registered")
            print(f"  Available platforms: {platforms}")
            return False
    
    print("\n✓ All DBLP adapters are properly registered\n")
    return True


def test_adapter_info():
    """Test that adapter information is correct."""
    print("=" * 60)
    print("Testing DBLP Adapter Information")
    print("=" * 60)
    
    for platform_name in ['dblp_iccad', 'dblp_dac', 'dblp_tcad']:
        adapter = AdapterRegistry.get(platform_name)
        if adapter:
            print(f"\n{platform_name.upper()}:")
            print(f"  Platform: {adapter.platform_name}")
            print(f"  Venue Type: {adapter.venue_type}")
            print(f"  Venue Name: {adapter.venue_name}")
            print(f"  Supported Venues: {adapter.get_supported_venues()}")
            print(f"  Rate Limit: {adapter.rate_limit_delay()}s")
            print(f"  No API Key Required: ✓")
        else:
            print(f"✗ Could not retrieve {platform_name} adapter")
            return False
    
    print("\n✓ All adapter info retrieved successfully\n")
    return True


def test_dblp_connectivity():
    """Test DBLP API connectivity (no key needed)."""
    print("=" * 60)
    print("Testing DBLP API Connectivity (NO API KEY)")
    print("=" * 60)
    
    adapter = AdapterRegistry.get('dblp_iccad')
    if not adapter:
        print("✗ DBLP ICCAD adapter not found")
        return False
    
    # Test check_availability
    print("\nTesting DBLP API availability...")
    if adapter.check_availability():
        print("✓ DBLP API is reachable")
    else:
        print("✗ DBLP API is not reachable")
        return False
    
    print("\n✓ DBLP connectivity test passed\n")
    return True


def test_iccad_crawl():
    """Test crawling ICCAD 2024 papers (small sample)."""
    print("=" * 60)
    print("Testing ICCAD 2024 Paper Crawl from DBLP")
    print("=" * 60)
    
    adapter = AdapterRegistry.get('dblp_iccad')
    if not adapter:
        print("✗ DBLP ICCAD adapter not found")
        return False
    
    config = VenueConfig(
        name='ICCAD',
        years=[2024],
        platform='dblp_iccad',
        additional_params={}  # NO API KEY NEEDED
    )
    
    print("\nCrawling ICCAD 2024 papers (this may take 10-20 seconds)...")
    print("Note: DBLP does not provide abstracts\n")
    
    try:
        papers = adapter.crawl(config)
        
        if papers:
            print(f"✓ Successfully crawled {len(papers)} papers from ICCAD 2024")
            
            # Show first 3 papers
            print(f"\n  First 30 papers:")
            for i, paper in enumerate(papers[:30], 1):
                print(f"\n  {i}. {paper.title}")
                print(f"     Authors: {', '.join(paper.authors[:3])}{'...' if len(paper.authors) > 3 else ''}")
                print(f"     Year: {paper.year}")
                print(f"     Venue: {paper.venue}")
                print(f"     DOI: {paper.doi or 'N/A'}")
                print(f"     Has Abstract: {'Yes' if paper.abstract else 'No (DBLP limitation)'}")
            
            # Show statistics
            print(f"\n  Statistics:")
            print(f"    Total papers: {len(papers)}")
            print(f"    Papers with DOI: {sum(1 for p in papers if p.doi)}")
            print(f"    Papers with URL: {sum(1 for p in papers if p.pdf_url)}")
        else:
            print("⚠ No papers found")
            return False
            
    except Exception as e:
        print(f"✗ Error during crawl: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n✓ ICCAD crawl test passed\n")
    return True


def test_dac_crawl():
    """Test crawling DAC 2024 papers (small sample)."""
    print("=" * 60)
    print("Testing DAC 2024 Paper Crawl from DBLP")
    print("=" * 60)
    
    adapter = AdapterRegistry.get('dblp_dac')
    if not adapter:
        print("✗ DBLP DAC adapter not found")
        return False
    
    config = VenueConfig(
        name='DAC',
        years=[2024],
        platform='dblp_dac',
        additional_params={}
    )
    
    print("\nCrawling DAC 2024 papers (this may take 10-20 seconds)...")
    
    try:
        papers = adapter.crawl(config)
        
        if papers:
            print(f"✓ Successfully crawled {len(papers)} papers from DAC 2024")
            
            # Show sample
            sample = papers[100]
            print(f"\n  Sample paper:")
            print(f"    Title: {sample.title}")
            print(f"    Authors: {', '.join(sample.authors[:3])}{'...' if len(sample.authors) > 3 else ''}")
            print(f"    Year: {sample.year}")
            print(f"    DOI: {sample.doi or 'N/A'}")
        else:
            print("⚠ No papers found")
            return False
            
    except Exception as e:
        print(f"✗ Error during crawl: {e}")
        return False
    
    print("\n✓ DAC crawl test passed\n")
    return True


def main():
    print("\n" + "=" * 60)
    print("DBLP EDA Adapter Test Suite (NO API KEY REQUIRED)")
    print("=" * 60 + "\n")
    
    results = []
    
    # Test 1: Adapter Registration
    results.append(("Adapter Registration", test_adapter_registration()))
    
    # Test 2: Adapter Information
    results.append(("Adapter Information", test_adapter_info()))
    
    # Test 3: DBLP Connectivity
    results.append(("DBLP Connectivity", test_dblp_connectivity()))
    
    # Test 4: ICCAD Crawl
    results.append(("ICCAD 2024 Crawl", test_iccad_crawl()))
    
    # Test 5: DAC Crawl
    results.append(("DAC 2024 Crawl", test_dac_crawl()))
    
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
        print("\nIMPORTANT NOTES:")
        print("- DBLP does NOT provide abstracts (bibliographic metadata only)")
        print("- DBLP is completely FREE - no API key required")
        print("- Use IEEE Xplore adapter if you need abstracts (requires API key)")
        return 0
    else:
        print(f"\n⚠ {total - passed} test(s) failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
