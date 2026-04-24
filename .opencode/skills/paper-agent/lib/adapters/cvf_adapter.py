#!/usr/bin/env python3
"""
CVF (Computer Vision Foundation) Adapter for CVPR/ICCV

Implements venue adapter for CVF Open Access platform, supporting:
- CVPR (Conference on Computer Vision and Pattern Recognition)
- ICCV (International Conference on Computer Vision)

Uses HTML scraping via requests + BeautifulSoup4.
All papers are open access and freely downloadable.

Website: https://openaccess.thecvf.com
"""

import re
import time
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path
import hashlib

import requests
from bs4 import BeautifulSoup

import sys
# Ensure parent directory is in path for imports
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from database import Paper
from base import VenueAdapter, VenueConfig
from registry import AdapterRegistry


logger = logging.getLogger(__name__)


class CVFAdapter(VenueAdapter):
    """
    Base adapter for CVF conferences (CVPR, ICCV).
    
    Uses HTML scraping to retrieve papers from the CVF Open Access website.
    The website has a predictable structure with papers listed in tables.
    
    Rate limiting: 2.0 seconds between requests (CVF recommendation: 1-2 seconds)
    """
    
    BASE_URL = "https://openaccess.thecvf.com"
    
    # Supported venues and their URL patterns
    VENUE_MAPPING = {
        'CVPR': 'CVPR',
        'ICCV': 'ICCV',
    }
    
    def __init__(self, venue_code: str):
        """
        Initialize CVF adapter.
        
        Args:
            venue_code: Venue identifier ('CVPR' or 'ICCV')
        """
        self.venue_code = venue_code.upper()
        self.fetch_abstract = False
        if self.venue_code not in self.VENUE_MAPPING:
            raise ValueError(
                f"Unknown venue: {venue_code}. "
                f"Supported: {list(self.VENUE_MAPPING.keys())}"
            )
    
    @property
    def platform_name(self) -> str:
        """Return platform identifier."""
        return self.venue_code.lower()
    
    @property
    def venue_type(self) -> str:
        """Return venue type (always conference for CVPR/ICCV)."""
        return "conference"
    
    def crawl(self, config: VenueConfig) -> List[Paper]:
        """
        Crawl CVF conference papers.
        
        Args:
            config: VenueConfig with years
            
        Returns:
            List of Paper objects
        """
        papers = []
        self.fetch_abstract = bool((config.additional_params or {}).get("fetch_abstract", False))
        
        for year in config.years:
            logger.info(f"Crawling {self.venue_code} {year}...")
            try:
                year_papers = self._crawl_year(year)
                papers.extend(year_papers)
                logger.info(f"Found {len(year_papers)} papers for {self.venue_code} {year}")
            except Exception as e:
                logger.error(f"Error crawling {self.venue_code} {year}: {e}")
                # Continue with other years
            
            # Rate limiting between years
            time.sleep(self.rate_limit_delay())
        
        return papers
    
    def _crawl_year(self, year: int) -> List[Paper]:
        """
        Crawl papers for a specific year.
        
        CVF uses different URL patterns across years:
        - 2013 and earlier: /{CONF}{YEAR} (e.g., /CVPR2013)
        - 2014-2020: /{CONF}{YEAR} with day parameter, may not support ?day=all
        - 2021 and later: /{CONF}{YEAR}?day=all typically supported
        
        This method intelligently detects the supported mode:
        1. First tries ?day=all (faster, single request)
        2. If not supported, fetches all available days and crawls each
        
        Args:
            year: Publication year
            
        Returns:
            List of Paper objects
        """
        papers = []
        
        # Build the day=all URL
        if year <= 2013:
            # 2013 and earlier use simple URL without day parameter
            url = f"{self.BASE_URL}/{self.venue_code}{year}"
            papers = self._crawl_single_url(url, year)
        else:
            # Try day=all first
            url_with_all = f"{self.BASE_URL}/{self.venue_code}{year}?day=all"
            
            if self._is_day_all_supported(year, url_with_all):
                # day=all is supported, crawl it
                logger.info(f"Using ?day=all mode for {self.venue_code} {year}")
                papers = self._crawl_single_url(url_with_all, year, timeout=120)
            else:
                # day=all not supported, crawl by days
                logger.info(f"Day=all not supported for {self.venue_code} {year}, using per-day mode")
                papers = self._crawl_by_days(year)
        
        return papers
    
    def _is_day_all_supported(self, year: int, url: str) -> bool:
        """
        Check if ?day=all is supported for a given year.
        """
        try:
            probe_timeout = 120 if "day=all" in url else 45
            response = self._fetch_page(url, timeout=probe_timeout)
            if response is None:
                return False
            
            # Check for error indicators in response text
            if "Error" in response.text or "error" in response.text.lower():
                if "1525" in response.text or "DATE value" in response.text:
                    return False
                if "Incorrect DATE" in response.text:
                    return False
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Check if page contains any paper elements
            paper_elements = (
                soup.find_all('dt', class_='ptitle') or
                soup.find_all('div', class_='ptitle') or
                soup.find_all('table')
            )
            
            has_papers = len(paper_elements) > 0
            
            if has_papers:
                logger.debug(f"day=all supported: found {len(paper_elements)} paper elements")
            else:
                logger.debug(f"day=all appears unsupported: no paper elements found")
            
            return has_papers
            
        except Exception as e:
            logger.warning(f"Failed to check day=all support for {url}: {e}")
            return False
    
    def _crawl_single_url(self, url: str, year: int, timeout: int = 45) -> List[Paper]:
        """
        Crawl papers from a single URL.
        """
        papers = []
        
        try:
            response = self._fetch_page(url, timeout=timeout)
            if response is None:
                return papers
            
            soup = BeautifulSoup(response.text, 'html.parser')
            papers = self._parse_papers_page(soup, year)
            
        except Exception as e:
            logger.error(f"Failed to crawl {url}: {e}")
        
        return papers
    
    def _crawl_by_days(self, year: int) -> List[Paper]:
        """
        Crawl papers by fetching all available days and crawling each.
        """
        papers = []
        main_url = f"{self.BASE_URL}/{self.venue_code}{year}"
        
        try:
            response = self._fetch_page(main_url, timeout=60)
            if response is None:
                return papers
            
            soup = BeautifulSoup(response.text, 'html.parser')
            day_urls = self._get_day_urls(soup, year)
            
            if not day_urls:
                logger.warning(f"No day links found for {self.venue_code} {year}, trying main page")
                return self._parse_papers_page(soup, year)
            
            logger.info(f"Found {len(day_urls)} day links for {self.venue_code} {year}")
            
            for day_url in day_urls:
                try:
                    day_papers = self._crawl_single_url(day_url, year, timeout=75)
                    papers.extend(day_papers)
                    logger.debug(f"Crawled {len(day_papers)} papers from {day_url}")
                    time.sleep(self.rate_limit_delay())
                except Exception as e:
                    logger.warning(f"Failed to crawl {day_url}: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"Failed to crawl {self.venue_code} {year} by days: {e}")
        
        return papers
    
    def _get_day_urls(self, soup: BeautifulSoup, year: int) -> List[str]:
        """
        Extract all day URLs from the main page.
        """
        day_urls = []
        
        for link in soup.find_all('a', href=True):
            href = link.get('href', '')
            
            if 'day=' in href:
                if href.startswith('http'):
                    day_url = href
                else:
                    if href.startswith('/'):
                        day_url = f"{self.BASE_URL}{href}"
                    else:
                        day_url = f"{self.BASE_URL}/{href}"
                
                day_urls.append(day_url)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_urls = []
        for url in day_urls:
            if url not in seen:
                seen.add(url)
                unique_urls.append(url)
        
        return unique_urls
    
    def _crawl_day_url(self, url: str, year: int) -> List[Paper]:
        """
        Crawl papers from a specific day URL.
        """
        return self._crawl_single_url(url, year)
    
    def _fetch_page(self, url: str, timeout: int = 30) -> Optional[requests.Response]:
        """
        Fetch a page with error handling.
        
        Args:
            url: URL to fetch
            timeout: Request timeout in seconds
            
        Returns:
            Response object or None on failure
        """
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to fetch {url}: {e}")
            return None
    
    def _parse_papers_page(self, soup: BeautifulSoup, year: int) -> List[Paper]:
        """
        Parse papers from CVF HTML page.
        
        CVF uses different HTML structures across years:
        1. Modern (2018+): Papers in dt.ptitle with sibling dd elements
        2. Legacy: Papers in dt/dd pairs or nested tables
        
        Args:
            soup: BeautifulSoup object of the page
            year: Publication year
            
        Returns:
            List of Paper objects
        """
        papers = []
        
        # Try legacy format first (dt.ptitle structure used by CVPR 2020+)
        legacy_papers = self._parse_legacy_format(soup, year)
        if legacy_papers:
            papers.extend(legacy_papers)
            logger.debug(f"Parsed {len(legacy_papers)} papers using legacy format")
            return papers
        
        # Try modern format (div.ptitle structure)
        modern_papers = self._parse_modern_format(soup, year)
        if modern_papers:
            papers.extend(modern_papers)
            logger.debug(f"Parsed {len(modern_papers)} papers using modern format")
            return papers
        
        # Fallback: try table-based parsing
        table_papers = self._parse_table_format(soup, year)
        if table_papers:
            papers.extend(table_papers)
            logger.debug(f"Parsed {len(table_papers)} papers using table format")
        
        return papers
    
    def _parse_modern_format(self, soup: BeautifulSoup, year: int) -> List[Paper]:
        """
        Parse papers in modern CVF format (2018+).
        
        Structure:
        - Papers are in <div class="ptitle"> elements
        - Title is in an <a> tag within the div
        - Authors are in the next sibling element
        
        Args:
            soup: BeautifulSoup object
            year: Publication year
            
        Returns:
            List of Paper objects
        """
        papers = []
        
        # Find all paper title divs
        title_divs = soup.find_all('div', class_='ptitle')
        
        for title_div in title_divs:
            try:
                paper = self._parse_paper_from_title_div(title_div, year)
                if paper:
                    papers.append(paper)
            except Exception as e:
                logger.debug(f"Failed to parse paper: {e}")
                continue
        
        return papers
    
    def _parse_paper_from_title_div(self, title_div, year: int) -> Optional[Paper]:
        """
        Parse a single paper from a title div.
        
        Args:
            title_div: BeautifulSoup element with class 'ptitle'
            year: Publication year
            
        Returns:
            Paper object or None
        """
        # Get title link
        title_link = title_div.find('a')
        if not title_link:
            return None
        
        title = title_link.get_text(strip=True)
        paper_href = title_link.get('href', '')
        
        # Build paper URL
        if paper_href and not paper_href.startswith('http'):
            paper_url = f"{self.BASE_URL}/{self.venue_code}{year}/{paper_href}"
        else:
            paper_url = paper_href
        
        # Find authors - usually in the next div with class 'authors' or similar
        authors = []
        authors_div = title_div.find_next_sibling('div', class_='authors')
        if authors_div:
            authors_text = authors_div.get_text(strip=True)
            # Authors are typically comma-separated
            authors = [a.strip() for a in authors_text.split(',') if a.strip()]
        else:
            # Try finding in parent container
            parent = title_div.parent
            if parent:
                text = parent.get_text()
                # Look for author patterns
                authors = self._extract_authors_from_text(text)
        
        # Find PDF link
        pdf_url = None
        # PDF link is usually in a link with text 'pdf' or ends with .pdf
        parent = title_div.parent
        if parent:
            pdf_link = parent.find('a', href=lambda x: x and '.pdf' in x.lower())
            if pdf_link:
                pdf_url = pdf_link.get('href', '')
                if pdf_url and not pdf_url.startswith('http'):
                    pdf_url = f"{self.BASE_URL}/{self.venue_code}{year}/{pdf_url}"
        
        # Fetch abstract if we have a paper page URL
        abstract = ""
        if paper_url and self.fetch_abstract:
            abstract = self._fetch_abstract(paper_url)
        
        # Generate unique paper ID
        paper_id = self._generate_paper_id(title, year, authors)
        
        return Paper(
            id=paper_id,
            title=title,
            abstract=abstract,
            authors=authors,
            keywords=[],
            year=year,
            venue=self.venue_code,
            venue_type='conference',
            source_platform=self.platform_name,
            crawl_date=datetime.now().isoformat(),
            pdf_url=pdf_url,
            download_available='openreview' if pdf_url else 'none',
        )
    
    def _parse_legacy_format(self, soup: BeautifulSoup, year: int) -> List[Paper]:
        """
        Parse papers in legacy CVF format (older years).
        
        Structure:
        - Papers are in <dt> (definition term) and <dd> (definition description) pairs
        - Title is in <dt>
        - Authors are in first <dd>
        - PDF and links are in second <dd>
        
        Args:
            soup: BeautifulSoup object
            year: Publication year
            
        Returns:
            List of Paper objects
        """
        papers = []
        
        dt_tags = soup.find_all('dt', class_='ptitle')
        if not dt_tags:
            # Older pages may use plain <dt> tags inside <dl> without class.
            dt_tags = [dt for dt in soup.find_all('dt') if dt.find('a')]
        
        for dt in dt_tags:
            try:
                paper = self._parse_dt_with_siblings(dt, year)
                if paper:
                    papers.append(paper)
            except Exception as e:
                logger.debug(f"Failed to parse dt: {e}")
                continue
        
        return papers
    
    def _parse_dt_with_siblings(self, dt, year: int) -> Optional[Paper]:
        """
        Parse a paper from dt element and its sibling dd elements.
        
        Args:
            dt: Definition term element with class 'ptitle' (contains title)
            year: Publication year
            
        Returns:
            Paper object or None
        """
        # Extract title from dt
        title_elem = dt.find('a')
        if title_elem:
            title = title_elem.get_text(strip=True)
            paper_href = title_elem.get('href', '')
        else:
            title = dt.get_text(strip=True)
            paper_href = ''
        
        if not title:
            return None
        
        # Build paper URL
        if paper_href and not paper_href.startswith('http'):
            paper_url = f"{self.BASE_URL}/{paper_href.lstrip('/')}"
        else:
            paper_url = paper_href if paper_href else None
        
        # Get the next two sibling dd elements
        next_siblings = []
        current = dt.next_sibling
        while current and len(next_siblings) < 2:
            if current.name == 'dd':
                next_siblings.append(current)
            current = current.next_sibling
        
        # Extract authors from first dd
        authors = []
        if len(next_siblings) >= 1:
            authors_dd = next_siblings[0]
            authors = self._extract_authors_from_dd(authors_dd)
        
        # Extract PDF from second dd, or from first dd when legacy markup
        # stores both authors and links in a single block.
        pdf_url = None
        if len(next_siblings) >= 2:
            links_dd = next_siblings[1]
            pdf_link = links_dd.find('a', href=lambda x: x and '.pdf' in x.lower())
            if pdf_link:
                pdf_href = pdf_link.get('href', '')
                if pdf_href and not pdf_href.startswith('http'):
                    pdf_url = f"{self.BASE_URL}/{pdf_href.lstrip('/')}"
                else:
                    pdf_url = pdf_href
        elif len(next_siblings) == 1:
            links_dd = next_siblings[0]
            pdf_link = links_dd.find('a', href=lambda x: x and '.pdf' in x.lower())
            if pdf_link:
                pdf_href = pdf_link.get('href', '')
                if pdf_href and not pdf_href.startswith('http'):
                    pdf_url = f"{self.BASE_URL}/{pdf_href.lstrip('/')}"
                else:
                    pdf_url = pdf_href
        
        # Generate paper ID
        paper_id = self._generate_paper_id(title, year, authors)
        
        # Fetch abstract from paper detail page
        abstract = ""
        if paper_url and self.fetch_abstract:
            abstract = self._fetch_abstract(paper_url)
        
        return Paper(
            id=paper_id,
            title=title,
            abstract=abstract,
            authors=authors,
            keywords=[],
            year=year,
            venue=self.venue_code,
            venue_type='conference',
            source_platform=self.platform_name,
            crawl_date=datetime.now().isoformat(),
            pdf_url=pdf_url,
            download_available='openreview' if pdf_url else 'none',
        )
    
    def _extract_authors_from_dd(self, dd) -> List[str]:
        """
        Extract author names from dd element.
        
        Args:
            dd: Definition description element containing authors
            
        Returns:
            List of author names
        """
        authors = []
        
        # Extract from form elements with class 'authsearch'
        forms = dd.find_all('form', class_='authsearch')
        for form in forms:
            input_field = form.find('input', {'name': 'query_author'})
            if input_field:
                author_name = input_field.get('value', '').strip()
                if author_name:
                    authors.append(author_name)

        # Some legacy pages expose authors as plain text:
        # "Authors: A, B"
        if not authors:
            text = dd.get_text(" ", strip=True)
            match = re.search(r'Authors?:\s*(.*?)(?:\bPDF\b|$)', text, flags=re.IGNORECASE)
            if match:
                raw_authors = match.group(1).strip().strip(':')
                for candidate in raw_authors.split(','):
                    name = candidate.strip()
                    if name:
                        authors.append(name)
        
        return authors
    
    def _parse_table_format(self, soup: BeautifulSoup, year: int) -> List[Paper]:
        """
        Parse papers from table-based layout.
        
        Fallback parser for pages using table layout.
        
        Args:
            soup: BeautifulSoup object
            year: Publication year
            
        Returns:
            List of Paper objects
        """
        papers = []
        
        # Find all tables
        tables = soup.find_all('table')
        
        for table in tables:
            rows = table.find_all('tr')
            for row in rows:
                try:
                    paper = self._parse_table_row(row, year)
                    if paper:
                        papers.append(paper)
                except Exception:
                    continue
        
        return papers
    
    def _parse_table_row(self, row, year: int) -> Optional[Paper]:
        """
        Parse a paper from a table row.
        
        Args:
            row: Table row element
            year: Publication year
            
        Returns:
            Paper object or None
        """
        cells = row.find_all(['td', 'th'])
        if not cells:
            return None
        
        # Look for links in the row
        links = row.find_all('a')
        title = None
        pdf_url = None
        
        for link in links:
            href = link.get('href', '')
            text = link.get_text(strip=True)
            
            # PDF link
            if '.pdf' in href.lower():
                pdf_url = href
                if pdf_url and not pdf_url.startswith('http'):
                    pdf_url = f"{self.BASE_URL}/{self.venue_code}{year}/{pdf_url}"
            # Title link (usually the longest text link)
            elif text and (not title or len(text) > len(title)):
                if len(text) > 10:  # Heuristic: titles are usually longer
                    title = text
        
        if not title:
            return None
        
        # Extract authors from row text
        row_text = row.get_text()
        authors = self._extract_authors_from_text(row_text)
        
        paper_id = self._generate_paper_id(title, year, authors)
        
        return Paper(
            id=paper_id,
            title=title,
            abstract="",
            authors=authors,
            keywords=[],
            year=year,
            venue=self.venue_code,
            venue_type='conference',
            source_platform=self.platform_name,
            crawl_date=datetime.now().isoformat(),
            pdf_url=pdf_url,
            download_available='openreview' if pdf_url else 'none',
        )
    
    def _extract_authors_from_text(self, text: str) -> List[str]:
        """
        Extract author names from text.
        
        Uses heuristics to find author names in text.
        
        Args:
            text: Text to extract authors from
            
        Returns:
            List of author names
        """
        authors = []
        
        # Remove common non-author text
        text = re.sub(r'\b(pdf|supplementary|video|code|poster|slides?)\b', '', text, flags=re.IGNORECASE)
        
        # Look for patterns like "Author Name, Another Name"
        # Author names typically start with uppercase and may contain spaces and hyphens
        author_pattern = r'([A-Z][a-zA-Z\-\']+ [A-Z][a-zA-Z\-\']+)'
        matches = re.findall(author_pattern, text)
        
        for match in matches:
            name = match.strip()
            # Filter out obvious non-names
            if name and len(name) > 3 and name not in authors:
                authors.append(name)
        
        return authors[:10]  # Limit to 10 authors
    
    def _generate_paper_id(self, title: str, year: int, authors: List[str]) -> str:
        """
        Generate a unique paper ID.
        
        Format: {venue}_{year}_{hash}
        
        Args:
            title: Paper title
            year: Publication year
            authors: List of authors
            
        Returns:
            Unique paper ID
        """
        # Create a hash from title + first author for uniqueness
        content = f"{title}_{authors[0] if authors else ''}"
        hash_part = hashlib.md5(content.encode()).hexdigest()[:8]
        
        # Clean title for readable ID
        clean_title = re.sub(r'[^\w\s]', '', title)
        clean_title = re.sub(r'\s+', '_', clean_title)[:30]
        
        return f"{self.venue_code.lower()}_{year}_{hash_part}"
    
    def _fetch_abstract(self, paper_url: str) -> str:
        """
        Fetch abstract from paper detail page.
        
        Args:
            paper_url: URL to paper page
            
        Returns:
            Abstract text or empty string
        """
        if not paper_url:
            return ""
        
        try:
            response = self._fetch_page(paper_url, timeout=10)
            if not response:
                return ""
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Try to find abstract in common locations
            # 1. div with id 'abstract'
            abstract_div = soup.find('div', id='abstract')
            if abstract_div:
                return abstract_div.get_text(strip=True)
            
            # 2. paragraph containing "Abstract"
            for p in soup.find_all('p'):
                text = p.get_text()
                if 'abstract' in text.lower()[:100]:
                    # Clean up the text
                    abstract = re.sub(r'^abstract[:\s]*', '', text, flags=re.IGNORECASE)
                    return abstract.strip()
            
            # 3. meta tag
            meta = soup.find('meta', attrs={'name': 'description'})
            if meta and meta.get('content'):
                return meta.get('content', '')
            
            return ""
            
        except Exception as e:
            logger.debug(f"Failed to fetch abstract from {paper_url}: {e}")
            return ""
    
    def get_pdf_url(self, paper_id: str, **kwargs) -> Optional[str]:
        """
        Get PDF URL for a paper.
        
        Note: CVF papers don't have a simple URL pattern from paper ID alone.
        This method returns None and relies on the pdf_url stored during crawl.
        
        Args:
            paper_id: Paper identifier
            
        Returns:
            None (CVF requires stored PDF URL)
        """
        # CVF URLs are stored during crawl, not generated from ID
        return None
    
    def rate_limit_delay(self) -> float:
        """
        Return recommended delay between requests.
        
        CVF recommends 1-2 seconds between requests.
        
        Returns:
            Delay in seconds
        """
        return 2.0
    
    def get_supported_venues(self) -> List[str]:
        """
        Return list of supported venue names.
        
        Returns:
            List of venue codes
        """
        return list(self.VENUE_MAPPING.keys())
    
    def check_availability(self) -> bool:
        """
        Check if the CVF website is accessible.
        
        Returns:
            True if the website responds
        """
        try:
            response = requests.get(self.BASE_URL, timeout=10)
            return response.status_code == 200
        except Exception:
            return False
    
    def supports_year(self, year: int) -> bool:
        """
        Check if year is supported.
        
        CVPR: 2000-present (annual)
        ICCV: 1987-present (biennial)
        
        Args:
            year: Year to check
            
        Returns:
            True if year is supported
        """
        current_year = datetime.now().year
        
        if self.venue_code == 'CVPR':
            return 2000 <= year <= current_year + 1
        elif self.venue_code == 'ICCV':
            # ICCV is biennial (odd years since 1987)
            return 1987 <= year <= current_year + 1
        
        return False


# @AdapterRegistry.register - disabled, use _ensure_initialized instead
class CVPRAdapter(CVFAdapter):
    """Adapter for CVPR (Conference on Computer Vision and Pattern Recognition)."""
    
    def __init__(self):
        """Initialize CVPR adapter."""
        super().__init__('CVPR')
    
    @property
    def platform_name(self) -> str:
        """Return platform identifier."""
        return "cvpr"


# @AdapterRegistry.register - disabled, use _ensure_initialized instead
class ICCVAdapter(CVFAdapter):
    """Adapter for ICCV (International Conference on Computer Vision)."""
    
    def __init__(self):
        """Initialize ICCV adapter."""
        super().__init__('ICCV')
    
    @property
    def platform_name(self) -> str:
        """Return platform identifier."""
        return "iccv"
