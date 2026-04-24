#!/usr/bin/env python3
"""
ACL Anthology adapter using the official ACL Anthology website.

Implements venue adapter for ACL Anthology, supporting:
- ACL (Annual Meeting of the Association for Computational Linguistics)
- EMNLP (Conference on Empirical Methods in Natural Language Processing)
- NAACL (North American Chapter of the ACL)
- EACL (European Chapter of the ACL)
- COLING (International Conference on Computational Linguistics)
- CoNLL (Conference on Computational Natural Language Learning)
- TACL (Transactions of the Association for Computational Linguistics)
- CL (Computational Linguistics Journal)
"""

import hashlib
from datetime import datetime
import logging
import re
import time
from typing import List, Optional, Dict, Any
from urllib.parse import urljoin
from tqdm import tqdm

import requests
from bs4 import BeautifulSoup

from database import Paper
from .base import VenueAdapter, VenueConfig
from .registry import AdapterRegistry


logger = logging.getLogger(__name__)


# @AdapterRegistry.register - disabled, use _ensure_initialized instead
class ACLAdapter(VenueAdapter):
    """
    Adapter for ACL Anthology (Association for Computational Linguistics).
    
    Uses the official ACL Anthology website to access paper metadata.
    All papers are open access and freely downloadable.
    
    Documentation: https://acl-anthology.readthedocs.io/
    GitHub: https://github.com/acl-org/acl-anthology
    """
    
    # Mapping of common conference/journal names to ACL Anthology venue IDs
    VENUE_MAPPING = {
        'ACL': 'acl',
        'EMNLP': 'emnlp',
        'NAACL': 'naacl',
        'EACL': 'eacl',
        'COLING': 'coling',
        'CoNLL': 'conll',
        'TACL': 'tacl',  # Journal
        'CL': 'cl',      # Journal (Computational Linguistics)
        'AACL': 'aacl',
        'IJCNLP': 'ijcnlp',
        'LREC': 'lrec',
        'SEM': 'sem',
        'WS': 'ws',      # Workshops
    }

    BASE_URL = "https://aclanthology.org"
    
    @property
    def platform_name(self) -> str:
        """Return the platform identifier."""
        return "acl"
    
    @property
    def venue_type(self) -> str:
        """Return the venue type (can be conference or journal)."""
        return "conference"  # Default, overridden per venue
    
    def crawl(self, config: VenueConfig) -> List[Paper]:
        """Crawl ACL papers from the website, with anthology fallback."""
        self.validate_config(config)
        venue_id = self._map_venue_name(config.name)
        if venue_id is None:
            raise ValueError(
                f"Unknown venue: {config.name}. "
                f"Supported venues: {list(self.VENUE_MAPPING.keys())}"
            )

        papers: List[Paper] = []
        for year in config.years:
            logger.info(f"ACL {config.name} {year}: starting web crawl")
            try:
                venue_papers = self._crawl_web_year(config.name, year)
                if not venue_papers:
                    logger.info(
                        f"ACL {config.name} {year}: web crawl returned 0 papers, trying anthology fallback"
                    )
                    venue_papers = self._crawl_venue_year_with_fallback(
                        config.name, year, config.additional_params
                    )
                papers.extend(venue_papers)
                abstract_count = sum(1 for p in venue_papers if p.abstract)
                logger.info(
                    f"ACL {config.name} {year}: papers={len(venue_papers)}, abstracts={abstract_count}"
                )
            except Exception as e:
                logger.error(f"Error crawling {config.name} {year}: {e}")
            time.sleep(self.rate_limit_delay())
        
        return papers

    def _crawl_venue_year_with_fallback(
        self,
        venue_name: str,
        year: int,
        options: Dict[str, Any],
    ) -> List[Paper]:
        venue_id = self._map_venue_name(venue_name)
        if venue_id is None:
            return []

        try:
            from acl_anthology.anthology import Anthology
            anthology = Anthology.from_repo(verbose=False)
        except Exception:
            try:
                from anthology import Anthology  # type: ignore
                anthology = Anthology()
            except ImportError:
                logger.warning(
                    "ACL Anthology library is not available for fallback; returning web results only"
                )
                return []

        return self._crawl_venue_year(
            anthology, venue_id, year, venue_name, options
        )

    def _crawl_web_year(self, venue_name: str, year: int) -> List[Paper]:
        event_url = f"{self.BASE_URL}/events/{venue_name.lower()}-{year}/"
        event_html = self._fetch_html(event_url, timeout=60)
        if not event_html:
            return []

        event_soup = BeautifulSoup(event_html, "html.parser")
        volume_urls = self._extract_volume_urls(event_soup, venue_name, year)
        logger.info(f"ACL {venue_name} {year}: found {len(volume_urls)} volume pages")

        papers: List[Paper] = []
        seen: set[str] = set()
        for volume_url in tqdm(volume_urls):
            volume_html = self._fetch_html(volume_url, timeout=60)
            if not volume_html:
                continue
            volume_soup = BeautifulSoup(volume_html, "html.parser")
            paper_urls = self._extract_paper_urls(volume_soup, venue_name, year)
            logger.info(
                f"ACL {venue_name} {year}: volume {self._volume_label(volume_url)} -> {len(paper_urls)} papers"
            )
            for paper_url in tqdm(paper_urls):
                if paper_url in seen:
                    continue
                seen.add(paper_url)
                detail_html = self._fetch_html(paper_url, timeout=60)
                if not detail_html:
                    continue
                paper = self._parse_detail_page(
                    BeautifulSoup(detail_html, "html.parser"),
                    paper_url,
                    venue_name,
                    year,
                )
                if paper:
                    papers.append(paper)

        return papers

    def _fetch_html(self, url: str, timeout: int = 30) -> Optional[str]:
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            logger.warning(f"Failed to fetch {url}: {e}")
            return None
    
    def _extract_volume_urls(self, soup: BeautifulSoup, venue_name: str, year: int) -> List[str]:
        volume_urls: List[str] = []
        volume_pattern = re.compile(rf"/volumes/{year}\.{re.escape(venue_name.lower())}-[a-z0-9-]+/?$")

        for link in soup.find_all("a", href=True):
            href = self._abs_url(link["href"])
            if volume_pattern.search(href):
                volume_urls.append(href.rstrip("/"))

        dedup: List[str] = []
        seen: set[str] = set()
        for url in volume_urls:
            if url not in seen:
                seen.add(url)
                dedup.append(url)
        return dedup

    def _extract_paper_urls(self, soup: BeautifulSoup, venue_name: str, year: int) -> List[str]:
        urls: List[str] = []
        paper_pattern = re.compile(rf"/{year}\.{re.escape(venue_name.lower())}-[a-z0-9-]+\.\d+/?$")

        for link in soup.find_all("a", href=True):
            href = self._abs_url(link["href"])
            if paper_pattern.search(href):
                tail = href.rstrip("/").split("/")[-1]
                if tail.endswith(".0"):
                    continue
                urls.append(href.rstrip("/"))

        dedup: List[str] = []
        seen: set[str] = set()
        for url in urls:
            if url not in seen:
                seen.add(url)
                dedup.append(url)
        return dedup

    def _parse_detail_page(
        self,
        soup: BeautifulSoup,
        paper_url: str,
        venue_name: str,
        year: int,
    ) -> Optional[Paper]:
        title = self._meta_content(soup, "citation_title")
        if not title:
            title_node = soup.select_one("h1, h2, h3, .title")
            if title_node:
                title = self._clean_text(title_node.get_text(" ", strip=True))
        if not title:
            return None

        abstract = self._extract_abstract(soup)
        authors = self._extract_authors(soup)
        pdf_url = self._extract_pdf_url(soup)
        doi = self._meta_content(soup, "citation_doi") or None

        paper_id = self._paper_id_from_url(paper_url, title, year)
        return Paper(
            id=paper_id,
            title=title,
            abstract=abstract,
            authors=authors,
            keywords=[],
            year=year,
            venue=venue_name,
            venue_type=self._get_venue_type(venue_name),
            source_platform=self.platform_name,
            crawl_date=datetime.now().isoformat(),
            pdf_url=pdf_url,
            doi=doi,
            download_available='acl' if pdf_url else 'none',
        )

    def _meta_content(self, soup: BeautifulSoup, name: str) -> str:
        meta = soup.find("meta", attrs={"name": name})
        if meta and meta.get("content"):
            return self._clean_text(meta["content"])
        return ""

    def _extract_authors(self, soup: BeautifulSoup) -> List[str]:
        authors: List[str] = []
        for meta in soup.find_all("meta", attrs={"name": "citation_author"}):
            value = self._clean_text(meta.get("content", ""))
            if value and value not in authors:
                authors.append(value)
        if authors:
            return authors

        author_node = soup.select_one(".authors, .author, .article-author")
        if author_node:
            raw = self._clean_text(author_node.get_text(" ", strip=True))
            if raw:
                authors = [x.strip() for x in re.split(r",| and ", raw) if x.strip()]
        return authors

    def _extract_abstract(self, soup: BeautifulSoup) -> str:
        abstract = self._meta_content(soup, "citation_abstract")
        if abstract:
            return abstract

        abstract_node = soup.select_one("div.abstract, section.abstract, #abstract, [class*='abstract']")
        if abstract_node:
            text = self._clean_text(abstract_node.get_text(" ", strip=True))
            return re.sub(r"^abstract[:\s]*", "", text, flags=re.IGNORECASE)
        return ""

    def _extract_pdf_url(self, soup: BeautifulSoup) -> Optional[str]:
        pdf = self._meta_content(soup, "citation_pdf_url")
        if pdf:
            return self._abs_url(pdf)

        for link in soup.find_all("a", href=True):
            href = self._abs_url(link["href"])
            if href.lower().endswith(".pdf") or "/pdf/" in href:
                return href
        return None

    def _abs_url(self, href: str) -> str:
        if not href:
            return ""
        if href.startswith("http://") or href.startswith("https://"):
            return href
        if href.startswith("/"):
            return f"{self.BASE_URL}{href}"
        return f"{self.BASE_URL}/{href}"

    def _clean_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", (text or "")).strip()

    def _paper_id_from_url(self, paper_url: str, title: str, year: int) -> str:
        tail = paper_url.rstrip("/").split("/")[-1]
        if tail and re.match(r"^\d+\.[a-z0-9-]+\.\d+$", tail):
            return tail
        if tail and re.match(r"^\d+\.[a-z0-9-]+$", tail):
            return tail
        digest = hashlib.md5(f"{title}_{year}".encode("utf-8")).hexdigest()[:10]
        return f"acl_{year}_{digest}"

    def _volume_label(self, volume_url: str) -> str:
        return volume_url.rstrip("/").split("/")[-1]

    def _map_venue_name(self, name: str) -> Optional[str]:
        """
        Map common conference names to ACL Anthology venue IDs.
        
        Args:
            name: Venue name (e.g., 'ACL', 'EMNLP')
            
        Returns:
            ACL Anthology venue ID or None if not found
        """
        return self.VENUE_MAPPING.get(name.upper())
    
    def _crawl_venue_year_with_library(
        self,
        anthology,
        venue_id: str,
        year: int,
        venue_name: str,
        options: Dict[str, Any]
    ) -> List[Paper]:
        """
        Crawl papers for a specific venue and year.
        
        Args:
            anthology: Anthology instance
            venue_id: ACL Anthology venue ID
            year: Publication year
            venue_name: Display name for the venue
            options: Additional options (e.g., accepted_only)
            
        Returns:
            List of Paper objects
        """
        papers = []
        
        try:
            # Get venue from anthology
            # The anthology library provides access via venues index
            venues = anthology.venues
            
            # Find volumes for this venue and year
            # ACL IDs typically follow pattern: {year}.{venue}.{type}.{number}
            # e.g., 2024.acl-long.1, 2024.acl-short.5
            
            year_str = str(year)
            
            # Iterate through all papers and filter by venue and year
            for paper in anthology.papers:
                paper_id = paper.id
                
                # Check if paper matches venue and year
                if not paper_id.startswith(f"{year_str}."):
                    continue
                
                # Check venue match
                if not self._paper_matches_venue(paper_id, venue_id):
                    continue
                
                # Extract paper metadata
                paper_obj = self._create_paper_from_anthology(
                    paper, venue_name, year
                )
                
                if paper_obj:
                    papers.append(paper_obj)
                    
        except AttributeError:
            # If the anthology structure is different, try alternative approach
            papers = self._crawl_venue_year_alternative(
                anthology, venue_id, year, venue_name, options
            )
        
        return papers

    def _crawl_venue_year(
        self,
        anthology,
        venue_id: str,
        year: int,
        venue_name: str,
        options: Dict[str, Any]
    ) -> List[Paper]:
        """Backward-compatible wrapper around the library-based crawl."""
        return self._crawl_venue_year_with_library(
            anthology, venue_id, year, venue_name, options
        )
    
    def _paper_matches_venue(self, paper_id: str, venue_id: str) -> bool:
        """
        Check if a paper ID matches the venue.
        
        ACL paper IDs follow pattern: {year}.{venue}-{type}.{number}
        e.g., 2024.acl-long.1
        
        Args:
            paper_id: ACL paper ID
            venue_id: Venue ID to match
            
        Returns:
            True if the paper matches the venue
        """
        parts = paper_id.split('.')
        if len(parts) >= 2:
            venue_part = parts[1].split('-')[0]  # Get venue before type
            return venue_part.lower() == venue_id.lower()
        return False
    
    def _crawl_venue_year_alternative(
        self,
        anthology,
        venue_id: str,
        year: int,
        venue_name: str,
        options: Dict[str, Any]
    ) -> List[Paper]:
        """
        Alternative crawling method using volume-based access.
        
        This is a fallback if the papers iterator is not available.
        
        Args:
            anthology: Anthology instance
            venue_id: ACL Anthology venue ID
            year: Publication year
            venue_name: Display name for the venue
            options: Additional options
            
        Returns:
            List of Paper objects
        """
        papers = []
        year_str = str(year)
        
        try:
            # Try to access via volumes
            for volume in anthology.volumes:
                volume_id = volume.id
                
                # Check if volume matches year and venue
                if not volume_id.startswith(f"{year_str}."):
                    continue
                
                if not self._paper_matches_venue(volume_id, venue_id):
                    continue
                
                # Get papers from this volume
                for paper in volume.papers:
                    paper_obj = self._create_paper_from_anthology(
                        paper, venue_name, year
                    )
                    if paper_obj:
                        papers.append(paper_obj)
                        
        except Exception as e:
            logger.warning(f"Alternative crawl also failed: {e}")
        
        return papers
    
    def _create_paper_from_anthology(
        self,
        paper,
        venue_name: str,
        year: int
    ) -> Optional[Paper]:
        """
        Create a Paper object from an ACL Anthology paper object.
        
        Args:
            paper: ACL Anthology paper object
            venue_name: Venue display name
            year: Publication year
            
        Returns:
            Paper object or None if creation fails
        """
        try:
            # Extract basic metadata
            paper_id = paper.id
            title = paper.title if hasattr(paper, 'title') else ""
            
            # Handle abstract (may not always be present)
            abstract = ""
            if hasattr(paper, 'abstract') and paper.abstract:
                abstract = paper.abstract
            
            # Extract authors
            authors = []
            if hasattr(paper, 'authors'):
                for author in paper.authors:
                    if hasattr(author, 'name'):
                        authors.append(author.name)
                    elif isinstance(author, str):
                        authors.append(author)
            
            # Get PDF URL
            pdf_url = self.get_pdf_url(paper_id)
            
            # Get DOI if available
            doi = None
            if hasattr(paper, 'doi') and paper.doi:
                doi = paper.doi
            
            # Get BibTeX
            bibtex = None
            if hasattr(paper, 'bibtex') and paper.bibtex:
                bibtex = paper.bibtex
            
            # Determine venue type
            venue_type = self._get_venue_type(venue_name)
            
            # Create Paper object
            return Paper(
                id=paper_id,
                title=title,
                abstract=abstract,
                authors=authors,
                keywords=[],  # ACL doesn't provide keywords directly
                year=year,
                venue=venue_name,
                venue_type=venue_type,
                source_platform='acl',
                crawl_date=datetime.now().isoformat(),
                pdf_url=pdf_url,
                doi=doi,
                bibtex=bibtex,
                download_available='acl',  # All ACL papers are open access
            )
            
        except Exception as e:
            logger.warning(f"Failed to create paper from anthology entry: {e}")
            return None
    
    def _get_venue_type(self, venue_name: str) -> str:
        """
        Determine if venue is a conference or journal.
        
        Args:
            venue_name: Venue name
            
        Returns:
            'conference' or 'journal'
        """
        journals = ['TACL', 'CL']  # Transactions of ACL and Computational Linguistics
        if venue_name.upper() in journals:
            return 'journal'
        return 'conference'
    
    def get_pdf_url(self, paper_id: str, **kwargs) -> Optional[str]:
        """
        Get PDF URL for an ACL paper.
        
        ACL papers follow pattern: https://aclanthology.org/{id}.pdf
        
        Args:
            paper_id: ACL Anthology ID (e.g., '2024.acl-long.1')
            
        Returns:
            Direct PDF URL
        """
        if not paper_id:
            return None
            
        # Remove .pdf suffix if already present
        if paper_id.endswith('.pdf'):
            paper_id = paper_id[:-4]
        
        return f"https://aclanthology.org/{paper_id}.pdf"
    
    def get_bibtex_url(self, paper_id: str) -> Optional[str]:
        """
        Get BibTeX URL for an ACL paper.
        
        Args:
            paper_id: ACL Anthology ID
            
        Returns:
            BibTeX URL
        """
        if not paper_id:
            return None
            
        if paper_id.endswith('.bib'):
            paper_id = paper_id[:-4]
        
        return f"https://aclanthology.org/{paper_id}.bib"
    
    def rate_limit_delay(self) -> float:
        """
        Return recommended delay between requests.
        
        ACL Anthology doesn't have strict rate limits, but recommends
        being courteous with delays.
        
        Returns:
            Delay in seconds
        """
        return 1.0
    
    def get_supported_venues(self) -> List[str]:
        """
        Return list of supported venue names.
        
        Returns:
            List of venue names
        """
        return list(self.VENUE_MAPPING.keys())
    
    def check_availability(self) -> bool:
        """
        Check if the ACL Anthology website is reachable.
        
        Returns:
            True if the website returns a successful response
        """
        try:
            response = requests.get(
                f"{self.BASE_URL}/events/acl-2025/",
                headers={
                    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                },
                timeout=20,
            )
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def validate_config(self, config: VenueConfig) -> bool:
        """
        Validate the configuration for this adapter.
        
        Args:
            config: VenueConfig to validate
            
        Returns:
            True if configuration is valid
        """
        # Check venue name
        if config.name.upper() not in self.VENUE_MAPPING:
            logger.warning(
                f"Unknown venue: {config.name}. "
                f"Supported: {list(self.VENUE_MAPPING.keys())}"
            )
            return False
        
        # Check years (ACL Anthology has papers from 1965 onwards)
        for year in config.years:
            if year < 1965 or year > datetime.now().year + 1:
                logger.warning(f"Year {year} may not be available in ACL Anthology")
                # Still return True as it's just a warning
        
        return True
