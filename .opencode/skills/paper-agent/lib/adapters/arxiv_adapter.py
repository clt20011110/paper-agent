#!/usr/bin/env python3
"""
arXiv Adapter for searching and crawling arXiv papers.

Uses arXiv API (http://export.arxiv.org/api/query) for searching.
Supports category filtering, date ranges, and keyword search.

Features:
- Full-text and title search
- Category filtering (cs.AI, cs.LG, etc.)
- Date range filtering
- PDF download URL resolution
- 3-second rate limiting (per arXiv guidelines)

Example:
    config = VenueConfig(
        name='arXiv Search',
        years=[2024],
        platform='arxiv',
        additional_params={
            'keywords': 'diffusion models',
            'categories': ['cs.AI', 'cs.LG'],
            'max_results': 100
        }
    )
    
    adapter = ArxivAdapter()
    papers = adapter.crawl(config)
"""

import requests
import feedparser
import time
from typing import List, Optional, Dict, Any, Tuple, Set
from datetime import datetime
from urllib.parse import quote

from .base import VenueAdapter, VenueConfig
from .registry import AdapterRegistry
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from database import Paper


# @AdapterRegistry.register - disabled, use _ensure_initialized instead
class ArxivAdapter(VenueAdapter):
    """
    Adapter for arXiv search and crawling.
    
    Uses arXiv API (http://export.arxiv.org/api/query) for searching.
    Supports category filtering, date ranges, and keyword search.
    """
    
    API_BASE_URL = "http://export.arxiv.org/api/query"
    
    # Common arXiv categories
    CATEGORIES = {
        'cs': ['cs.AI', 'cs.LG', 'cs.CL', 'cs.CV', 'cs.NE', 'cs.RO', 'cs.DB', 
               'cs.DC', 'cs.HC', 'cs.IR', 'cs.MA', 'cs.MM', 'cs.SE'],
        'physics': ['astro-ph', 'cond-mat', 'gr-qc', 'hep-ex', 'hep-lat',
                    'hep-ph', 'hep-th', 'math-ph', 'nlin', 'nucl-ex',
                    'nucl-th', 'physics', 'quant-ph'],
        'math': ['math.AG', 'math.AT', 'math.AP', 'math.CA', 'math.CO',
                 'math.DG', 'math.DS', 'math.GT', 'math.NA', 'math.NT',
                 'math.PR', 'math.ST', 'math.SG'],
        'stats': ['stat.AP', 'stat.CO', 'stat.ML', 'stat.ME', 'stat.TH'],
        'q-bio': ['q-bio.BM', 'q-bio.CB', 'q-bio.GN', 'q-bio.MN', 'q-bio.NC',
                  'q-bio.QM', 'q-bio.SC', 'q-bio.TO'],
        'q-fin': ['q-fin.CP', 'q-fin.EC', 'q-fin.GN', 'q-fin.MF', 'q-fin.PM',
                  'q-fin.PR', 'q-fin.RM', 'q-fin.ST', 'q-fin.TR'],
    }
    
    @property
    def platform_name(self) -> str:
        """Return platform identifier."""
        return "arxiv"
    
    @property
    def venue_type(self) -> str:
        """Return venue type (arXiv is a preprint server)."""
        return "preprint"
    
    def crawl(self, config: VenueConfig) -> List[Paper]:
        """
        Search arXiv for papers.
        
        Args:
            config: VenueConfig with:
                - additional_params['categories']: List of arXiv categories ['cs.AI', 'cs.LG', ...]
                - additional_params['keywords']: Search keywords
                - additional_params['date_range']: Tuple of (start_date, end_date) in 'YYYY-MM-DD' format
                - additional_params['max_results']: Maximum results per year (default: 100)
                
        Returns:
            List of Paper objects
        """
        self.validate_config(config)
        
        categories = config.additional_params.get('categories', ['cs.AI', 'cs.LG'])
        keywords = config.additional_params.get('keywords', '')
        date_range = config.additional_params.get('date_range', None)
        max_results = config.additional_params.get('max_results', 100)
        
        all_papers = []
        seen_ids = set()  # Track unique papers across years
        
        for year in config.years:
            # Build date range for this year
            year_start = f"{year}-01-01"
            year_end = f"{year}-12-31"
            
            if date_range:
                start, end = date_range
                # Use intersection of year and provided date range
                start_year = int(start[:4])
                end_year = int(end[:4])
                if start_year <= year <= end_year:
                    if start_year == year:
                        start_dt = start
                    else:
                        start_dt = year_start
                    if end_year == year:
                        end_dt = end
                    else:
                        end_dt = year_end
                else:
                    continue
            else:
                start_dt = year_start
                end_dt = year_end
            
            print(f"  Searching arXiv for {year}...")
            papers = self._search_arxiv(
                keywords=keywords,
                categories=categories,
                start_date=start_dt,
                end_date=end_dt,
                max_results=max_results
            )
            
            # Filter out duplicates
            for paper in papers:
                if paper.id not in seen_ids:
                    seen_ids.add(paper.id)
                    all_papers.append(paper)
            
            print(f"    Found {len(papers)} papers for {year}")
            
            # Respect arXiv rate limits
            time.sleep(self.rate_limit_delay())
        
        return all_papers
    
    def _search_arxiv(
        self,
        keywords: str,
        categories: List[str],
        start_date: str,
        end_date: str,
        max_results: int = 100
    ) -> List[Paper]:
        """
        Search arXiv API.
        
        Args:
            keywords: Search keywords
            categories: arXiv categories
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            max_results: Maximum results to return
            
        Returns:
            List of Paper objects
        """
        # Build search query
        query_parts = []
        
        # Add category filter
        if categories:
            cat_query = ' OR '.join([f'cat:{cat}' for cat in categories])
            query_parts.append(f'({cat_query})')
        
        # Add keyword filter
        if keywords:
            # Search in title and abstract
            keyword_query = quote(keywords)
            query_parts.append(f'(ti:{keyword_query} OR abs:{keyword_query})')
        
        # Add date filter
        query_parts.append(f'submittedDate:[{start_date} TO {end_date}]')
        
        query = ' AND '.join(query_parts)
        
        # Build API URL
        params = {
            'search_query': query,
            'start': 0,
            'max_results': max_results,
            'sortBy': 'submittedDate',
            'sortOrder': 'descending'
        }
        
        try:
            response = requests.get(self.API_BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            
            # Parse Atom feed
            feed = feedparser.parse(response.content)
            
            papers = []
            for entry in feed.entries:
                try:
                    paper = self._parse_entry(entry)
                    if paper:
                        papers.append(paper)
                except Exception as e:
                    print(f"  Warning: Error parsing arXiv entry: {e}")
                    continue
            
            return papers
            
        except requests.exceptions.RequestException as e:
            print(f"  Error querying arXiv API: {e}")
            return []
    
    def _parse_entry(self, entry) -> Optional[Paper]:
        """
        Parse arXiv Atom entry to Paper.
        
        Args:
            entry: feedparser entry object
            
        Returns:
            Paper object or None if parsing fails
        """
        try:
            # Extract arXiv ID
            arxiv_id = entry.id.split('/abs/')[-1]
            
            # Remove version suffix if present (e.g., v1, v2)
            if 'v' in arxiv_id and arxiv_id[-2].isdigit():
                base_id = arxiv_id.rsplit('v', 1)[0]
            else:
                base_id = arxiv_id
            
            # Extract title
            title = entry.title.replace('\n', ' ').strip()
            
            # Extract abstract
            abstract = entry.summary.replace('\n', ' ').strip() if hasattr(entry, 'summary') else ''
            
            # Extract authors
            authors = []
            if hasattr(entry, 'authors'):
                authors = [author.name for author in entry.authors]
            
            # Extract publication date
            published = getattr(entry, 'published_parsed', None)
            if published:
                year = published.tm_year
            else:
                year = datetime.now().year
            
            # Extract categories
            categories = []
            if hasattr(entry, 'tags'):
                categories = [tag.term for tag in entry.tags]
            
            # Get PDF URL
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            
            # Build DOI (some arXiv papers have DOIs)
            doi = None
            links = getattr(entry, 'links', [])
            if links:
                for link in links:
                    if isinstance(link, dict) and link.get('type') == 'text/doi':
                        doi = link.get('href')
                        break
            
            # Build Paper object
            return Paper(
                id=f"arxiv:{arxiv_id}",
                title=title,
                abstract=abstract,
                authors=authors,
                keywords=categories,
                year=year,
                venue='arXiv',
                venue_type='preprint',
                source_platform='arxiv',
                arxiv_id=arxiv_id,
                pdf_url=pdf_url,
                doi=doi,
                download_available='arxiv'
            )
            
        except Exception as e:
            print(f"  Error parsing entry: {e}")
            return None
    
    def get_pdf_url(self, paper_id: str, **kwargs) -> Optional[str]:
        """
        Get PDF URL for arXiv paper.
        
        Args:
            paper_id: arXiv ID (e.g., 'arxiv:2401.12345' or just '2401.12345')
            
        Returns:
            Direct PDF URL
        """
        # Remove 'arxiv:' prefix if present
        if paper_id.startswith('arxiv:'):
            paper_id = paper_id[6:]
        
        return f"https://arxiv.org/pdf/{paper_id}.pdf"
    
    def rate_limit_delay(self) -> float:
        """
        Return recommended delay between requests.
        
        arXiv API recommends 3 seconds between requests:
        https://info.arxiv.org/help/api/tou.html
        
        Returns:
            3.0 seconds delay
        """
        return 3.0
    
    def get_supported_venues(self) -> List[str]:
        """
        Return list of supported venue names.
        
        arXiv supports multiple categories across domains.
        
        Returns:
            List of supported venue/category names
        """
        return ['arXiv', 'arxiv']
    
    def check_availability(self) -> bool:
        """
        Check if arXiv API is available.
        
        Returns:
            True if requests and feedparser are installed
        """
        try:
            import requests
            import feedparser
            return True
        except ImportError:
            return False
    
    def search_by_title(
        self,
        title: str,
        max_results: int = 5
    ) -> Optional[Paper]:
        """
        Search arXiv for a paper by title.
        
        Helper method for finding arXiv versions of conference papers.
        
        Args:
            title: Paper title to search for
            max_results: Maximum results to check
            
        Returns:
            Paper if found with high similarity, None otherwise
        """
        config = VenueConfig(
            name='arXiv Search',
            years=[datetime.now().year],
            platform='arxiv',
            additional_params={
                'keywords': title,
                'max_results': max_results
            }
        )
        
        papers = self.crawl(config)
        
        if not papers:
            return None
        
        # Find best match by title similarity
        best_match = None
        best_score = 0.0
        
        for paper in papers:
            score = self._title_similarity(title.lower(), paper.title.lower())
            if score > best_score and score > 0.7:  # 70% similarity threshold
                best_score = score
                best_match = paper
        
        return best_match
    
    def _title_similarity(self, title1: str, title2: str) -> float:
        """
        Calculate Jaccard similarity between two titles.
        
        Args:
            title1: First title
            title2: Second title
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        words1 = set(title1.split())
        words2 = set(title2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0


def search_arxiv_for_paper(
    title: str,
    authors: Optional[List[str]] = None,
    max_results: int = 3
) -> Optional[Paper]:
    """
    Search arXiv for a specific paper by title.
    
    Helper function for finding arXiv versions of conference papers.
    
    Args:
        title: Paper title
        authors: List of author names (optional, currently unused)
        max_results: Max results to check
        
    Returns:
        Paper if found, None otherwise
    """
    adapter = ArxivAdapter()
    
    config = VenueConfig(
        name='arXiv Search',
        years=[datetime.now().year],
        platform='arxiv',
        additional_params={
            'keywords': title,
            'max_results': max_results
        }
    )
    
    papers = adapter.crawl(config)
    
    if not papers:
        return None
    
    # Find best match by title similarity
    best_match = None
    best_score = 0.0
    
    for paper in papers:
        score = _title_similarity(title.lower(), paper.title.lower())
        if score > best_score and score > 0.8:  # 80% similarity threshold
            best_score = score
            best_match = paper
    
    return best_match


def _title_similarity(title1: str, title2: str) -> float:
    """Calculate similarity between two titles using Jaccard index."""
    words1 = set(title1.split())
    words2 = set(title2.split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1 & words2
    union = words1 | words2
    
    return len(intersection) / len(union) if union else 0.0