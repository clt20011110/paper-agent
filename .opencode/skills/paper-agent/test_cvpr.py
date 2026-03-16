#!/usr/bin/env python3
"""
Test CVPR 2020-2025 paper crawling functionality
"""
import sys
import os

# Add the library path
script_dir = os.path.dirname(os.path.abspath(__file__))
lib_dir = os.path.join(script_dir, 'lib')
adapters_dir = os.path.join(lib_dir, 'adapters')
sys.path.insert(0, lib_dir)
sys.path.insert(0, adapters_dir)

from cvf_adapter import CVPRAdapter
from base import VenueConfig


def test_cvpr_crawl():
    """Test CVPR 2020-2025 paper crawling"""
    print("=" * 60)
    print("Testing CVPR 2020-2025 Paper Crawling")
    print("=" * 60)

    # Create adapter
    adapter = CVPRAdapter()

    # Test year range
    test_years = [2020, 2021, 2022, 2023, 2024, 2025]

    print(f"\nAdapter: {adapter.venue_code}")
    print(f"Platform: {adapter.platform_name}")
    print(f"Base URL: {adapter.BASE_URL}")
    print()

    # Check year support
    print("Year support check:")
    for year in test_years:
        supported = adapter.supports_year(year)
        print(f"  CVPR {year}: {'✓ Supported' if supported else '✗ Not supported'}")

    # Check website availability
    print(f"\nChecking CVF website availability...")
    if adapter.check_availability():
        print("  ✓ CVF website is accessible")
    else:
        print("  ✗ CVF website is not accessible")
        return False

    # Test crawling
    print("\n" + "=" * 60)
    print("Starting crawl for CVPR 2020-2025")
    print("=" * 60)

    config = VenueConfig(name="CVPR", years=[2025])

    try:
        papers = adapter.crawl(config)
        print(f"\n✓ Successfully crawled {len(papers)} papers")

        if papers:
            print("\nFirst 5 papers:")
            for i, paper in enumerate(papers[:5], 1):
                print(f"  {i}. {paper.title[:80]}...")
                authors_str = ', '.join(paper.authors[:3])
                if len(paper.authors) > 3:
                    authors_str += '...'
                print(f"     Authors: {authors_str}")
                print(f"     PDF: {paper.pdf_url if paper.pdf_url else 'N/A'}")
                print(f"     Abstrct: {paper.abstract}")
                print()

        return len(papers) > 0

    except Exception as e:
        print(f"\n✗ Crawl failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_cvpr_crawl()
    sys.exit(0 if success else 1)
