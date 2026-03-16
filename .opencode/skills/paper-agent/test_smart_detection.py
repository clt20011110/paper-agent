#!/usr/bin/env python3
"""
Quick test for smart day=all detection mechanism
Tests CVPR 2021 and ICCV 2019 per-day crawling
"""
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
lib_dir = os.path.join(script_dir, 'lib')
adapters_dir = os.path.join(lib_dir, 'adapters')
sys.path.insert(0, lib_dir)
sys.path.insert(0, adapters_dir)

from cvf_adapter import CVPRAdapter, ICCVAdapter


def test_day_all_detection():
    """Test day=all detection for different years"""
    print("=" * 60)
    print("Testing Day=All Detection Mechanism")
    print("=" * 60)
    
    results = []
    
    # Test cases: (adapter, year, expected_supported)
    test_cases = [
        (CVPRAdapter(), 2022, True, "CVPR 2022"),   # Should support day=all
        (CVPRAdapter(), 2021, False, "CVPR 2021"),  # Should NOT support day=all
        (CVPRAdapter(), 2020, False, "CVPR 2020"),  # Should NOT support day=all
        (ICCVAdapter(), 2019, False, "ICCV 2019"),  # Should NOT support day=all
    ]
    
    for adapter, year, expected_supported, name in test_cases:
        print(f"\n{name}:")
        url = f"{adapter.BASE_URL}/{adapter.venue_code}{year}?day=all"
        is_supported = adapter._is_day_all_supported(year, url)
        
        status = "✓" if is_supported == expected_supported else "✗"
        print(f"  {status} day=all supported: {is_supported} (expected: {expected_supported})")
        
        results.append((name, is_supported == expected_supported))
    
    return results


def test_day_url_extraction():
    """Test day URL extraction from main pages"""
    print("\n" + "=" * 60)
    print("Testing Day URL Extraction")
    print("=" * 60)
    
    results = []
    
    test_cases = [
        (CVPRAdapter(), 2021, 6, "CVPR 2021"),   # Should have 6 days
        (ICCVAdapter(), 2019, 4, "ICCV 2019"),   # Should have 4 days
    ]
    
    for adapter, year, expected_days, name in test_cases:
        print(f"\n{name}:")
        main_url = f"{adapter.BASE_URL}/{adapter.venue_code}{year}"
        
        try:
            import requests
            from bs4 import BeautifulSoup
            
            response = adapter._fetch_page(main_url)
            if response:
                soup = BeautifulSoup(response.text, 'html.parser')
                day_urls = adapter._get_day_urls(soup, year)
                
                print(f"  Found {len(day_urls)} day URLs (expected: {expected_days})")
                for url in day_urls[:3]:
                    print(f"    - {url}")
                if len(day_urls) > 3:
                    print(f"    ... and {len(day_urls)-3} more")
                
                success = len(day_urls) == expected_days
                status = "✓" if success else "✗"
                print(f"  {status} Day count matches expected")
                results.append((name, success))
            else:
                print(f"  ✗ Failed to fetch main page")
                results.append((name, False))
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append((name, False))
    
    return results


def test_sample_crawl():
    """Test crawling a small sample from CVPR 2021"""
    print("\n" + "=" * 60)
    print("Testing Sample Crawl from CVPR 2021 (first day only)")
    print("=" * 60)
    
    adapter = CVPRAdapter()
    
    try:
        # Crawl just first day
        day_url = "https://openaccess.thecvf.com/CVPR2021?day=2021-06-21"
        papers = adapter._crawl_single_url(day_url, 2021)
        
        print(f"\nCrawled {len(papers)} papers from first day")
        
        if papers:
            print("\nFirst 3 papers:")
            for i, paper in enumerate(papers[:3], 1):
                print(f"\n{i}. {paper.title}")
                print(f"   Authors: {', '.join(paper.authors[:5])}")
                print(f"   PDF: {paper.pdf_url if paper.pdf_url else 'N/A'}")
        
        success = len(papers) > 50
        status = "✓" if success else "✗"
        print(f"\n{status} Crawled {len(papers)} papers (>50 expected)")
        
        return [("CVPR 2021 Day 1 Crawl", success)]
    except Exception as e:
        print(f"\n✗ Crawl failed: {e}")
        import traceback
        traceback.print_exc()
        return [("CVPR 2021 Day 1 Crawl", False)]


if __name__ == "__main__":
    all_results = []
    
    # Test day=all detection
    all_results.extend(test_day_all_detection())
    
    # Test day URL extraction
    all_results.extend(test_day_url_extraction())
    
    # Test sample crawl
    all_results.extend(test_sample_crawl())
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for name, success in all_results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {name}")
    
    passed = sum(1 for _, success in all_results if success)
    total = len(all_results)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    sys.exit(0 if passed == total else 1)
