#!/usr/bin/env python3
"""
Test CVPR 2020 and ICCV 2019 paper crawling functionality
Tests the smart detection mechanism for day=all support
"""
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
lib_dir = os.path.join(script_dir, 'lib')
adapters_dir = os.path.join(lib_dir, 'adapters')
sys.path.insert(0, lib_dir)
sys.path.insert(0, adapters_dir)

from cvf_adapter import CVPRAdapter, ICCVAdapter
from base import VenueConfig


def test_cvpr_2020():
    """Test CVPR 2020 - Should use per-day crawling"""
    print("\n" + "=" * 60)
    print("Testing CVPR 2020 Paper Crawling")
    print("Expected: Day=all NOT supported, using per-day mode")
    print("=" * 60)

    adapter = CVPRAdapter()
    
    # First test if day=all is detected as unsupported
    url_with_all = f"{adapter.BASE_URL}/CVPR2020?day=all"
    is_supported = adapter._is_day_all_supported(2020, url_with_all)
    print(f"\nDay=all support check: {'✓ Supported' if is_supported else '✗ Not supported'}")
    print("Expected: Not supported (should use per-day crawling)")
    
    # Test crawling
    print("\nTesting crawl...")
    config = VenueConfig(name="CVPR", years=[2020])
    
    try:
        papers = adapter.crawl(config)
        print(f"\n✓ Successfully crawled {len(papers)} papers")
        
        if papers:
            print("\nSample papers:")
            for i, paper in enumerate(papers[:3], 1):
                print(f"  {i}. {paper.title[:70]}...")
                authors_str = ', '.join(paper.authors[:3])
                if len(paper.authors) > 3:
                    authors_str += f' +{len(paper.authors)-3} more'
                print(f"     Authors: {authors_str}")
                print(f"     PDF: {paper.pdf_url if paper.pdf_url else 'N/A'}")
                print(f"     Abstract length: {len(paper.abstract)} chars")
                print()
        
        success = len(papers) > 100  # CVPR 2020 should have hundreds of papers
        if success:
            print(f"✓ Crawled {len(papers)} papers (>100 expected)")
        else:
            print(f"✗ Only crawled {len(papers)} papers (<100 expected)")
        
        return success
        
    except Exception as e:
        print(f"\n✗ Crawl failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_iccv_2019():
    """Test ICCV 2019 - Should use per-day crawling"""
    print("\n" + "=" * 60)
    print("Testing ICCV 2019 Paper Crawling")
    print("Expected: Day=all NOT supported, using per-day mode")
    print("=" * 60)

    adapter = ICCVAdapter()
    
    # First test if day=all is detected as unsupported
    url_with_all = f"{adapter.BASE_URL}/ICCV2019?day=all"
    is_supported = adapter._is_day_all_supported(2019, url_with_all)
    print(f"\nDay=all support check: {'✓ Supported' if is_supported else '✗ Not supported'}")
    print("Expected: Not supported (should use per-day crawling)")
    
    # Test crawling
    print("\nTesting crawl...")
    config = VenueConfig(name="ICCV", years=[2019])
    
    try:
        papers = adapter.crawl(config)
        print(f"\n✓ Successfully crawled {len(papers)} papers")
        
        if papers:
            print("\nSample papers:")
            for i, paper in enumerate(papers[:3], 1):
                print(f"  {i}. {paper.title[:70]}...")
                authors_str = ', '.join(paper.authors[:3])
                if len(paper.authors) > 3:
                    authors_str += f' +{len(paper.authors)-3} more'
                print(f"     Authors: {authors_str}")
                print(f"     PDF: {paper.pdf_url if paper.pdf_url else 'N/A'}")
                print(f"     Abstract length: {len(paper.abstract)} chars")
                print()
        
        success = len(papers) > 50  # ICCV 2019 should have hundreds of papers
        if success:
            print(f"✓ Crawled {len(papers)} papers (>50 expected)")
        else:
            print(f"✗ Only crawled {len(papers)} papers (<50 expected)")
        
        return success
        
    except Exception as e:
        print(f"\n✗ Crawl failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cvpr_2022():
    """Test CVPR 2022 - Should use day=all mode"""
    print("\n" + "=" * 60)
    print("Testing CVPR 2022 Paper Crawling")
    print("Expected: Day=all IS supported")
    print("=" * 60)

    adapter = CVPRAdapter()
    
    # First test if day=all is detected as supported
    url_with_all = f"{adapter.BASE_URL}/CVPR2022?day=all"
    is_supported = adapter._is_day_all_supported(2022, url_with_all)
    print(f"\nDay=all support check: {'✓ Supported' if is_supported else '✗ Not supported'}")
    print("Expected: Supported (should use day=all mode)")
    
    # Test crawling
    print("\nTesting crawl...")
    config = VenueConfig(name="CVPR", years=[2022])
    
    try:
        papers = adapter.crawl(config)
        print(f"\n✓ Successfully crawled {len(papers)} papers")
        
        if papers:
            print("\nSample papers:")
            for i, paper in enumerate(papers[:3], 1):
                print(f"  {i}. {paper.title[:70]}...")
                authors_str = ', '.join(paper.authors[:3])
                if len(paper.authors) > 3:
                    authors_str += f' +{len(paper.authors)-3} more'
                print(f"     Authors: {authors_str}")
                print(f"     PDF: {paper.pdf_url if paper.pdf_url else 'N/A'}")
                print(f"     Abstract length: {len(paper.abstract)} chars")
                print()
        
        success = len(papers) > 100  # CVPR 2022 should have thousands of papers
        if success:
            print(f"✓ Crawled {len(papers)} papers (>100 expected)")
        else:
            print(f"✗ Only crawled {len(papers)} papers (<100 expected)")
        
        return success
        
    except Exception as e:
        print(f"\n✗ Crawl failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    results = []
    
    # Test CVPR 2020 (per-day mode)
    results.append(("CVPR 2020 (per-day mode)", test_cvpr_2020()))
    
    # Test ICCV 2019 (per-day mode)
    results.append(("ICCV 2019 (per-day mode)", test_iccv_2019()))
    
    # Test CVPR 2022 (day=all mode)
    results.append(("CVPR 2022 (day=all mode)", test_cvpr_2022()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {name}")
    
    all_passed = all(success for _, success in results)
    print(f"\nOverall: {'All tests passed!' if all_passed else 'Some tests failed!'}")
    
    sys.exit(0 if all_passed else 1)
