#!/usr/bin/env python3
"""IJCAI adapter using the official proceedings site.

The official IJCAI proceedings pages expose paper metadata and details pages
with abstract text. This adapter follows the proceedings index, resolves the
per-paper detail pages, and extracts title, authors, abstract, and PDF links.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from tqdm import tqdm

import requests
from bs4 import BeautifulSoup

from database import Paper
from .base import VenueAdapter, VenueConfig

logger = logging.getLogger(__name__)


class IJCAIAdapter(VenueAdapter):
    """Adapter for IJCAI papers via the official proceedings site."""

    BASE_URL = "https://www.ijcai.org"
    PROCEEDINGS_URL = f"{BASE_URL}/proceedings/{{year}}/"

    @property
    def platform_name(self) -> str:
        return "ijcai"

    @property
    def venue_type(self) -> str:
        return "conference"

    def get_supported_venues(self) -> List[str]:
        return ["IJCAI"]

    def supports_year(self, year: int) -> bool:
        current_year = datetime.now().year
        return 1969 <= year <= current_year + 1

    def crawl(self, config: VenueConfig) -> List[Paper]:
        self.validate_config(config)
        papers: List[Paper] = []

        for year in config.years:
            try:
                papers.extend(self._crawl_year(year))
            except Exception as e:
                logger.error(f"IJCAI {year} crawl failed: {e}")
            time.sleep(self.rate_limit_delay())

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

    def _crawl_year(self, year: int) -> List[Paper]:
        proceedings_url = self.PROCEEDINGS_URL.format(year=year)
        html = self._fetch_html(proceedings_url, timeout=120)
        if not html:
            return []

        soup = BeautifulSoup(html, "html.parser")
        detail_urls = self._extract_detail_urls(soup, year)

        papers: List[Paper] = []
        seen: set[str] = set()
        for detail_url in tqdm(detail_urls):
            # print(detail_url)
            if detail_url in seen:
                continue
            seen.add(detail_url)

            detail_html = self._fetch_html(detail_url, timeout=60)
            if not detail_html:
                continue

            detail_soup = BeautifulSoup(detail_html, "html.parser")
            paper = self._parse_detail_page(detail_soup, detail_url, year)
            if paper:
                papers.append(paper)

        logger.info(
            f"IJCAI {year}: papers={len(papers)}, abstracts={sum(1 for p in papers if p.abstract)}"
        )
        return papers

    def _extract_detail_urls(self, soup: BeautifulSoup, year: int) -> List[str]:
        urls: List[str] = []
        pattern = re.compile(rf"/proceedings/{year}/\d+/?$")

        for link in soup.find_all("a", href=True):
            href = self._abs_url(link["href"])
            if pattern.search(href):
                urls.append(href.rstrip("/"))

        # Deduplicate while preserving order.
        dedup: List[str] = []
        seen: set[str] = set()
        for url in urls:
            if url not in seen:
                seen.add(url)
                dedup.append(url)
        return dedup

    def _parse_detail_page(self, soup: BeautifulSoup, detail_url: str, year: int) -> Optional[Paper]:
        title = self._meta_content(soup, "citation_title")
        if not title:
            title_node = soup.select_one("h1, h2, h3, .title, .paper-title")
            if title_node:
                title = self._clean_text(title_node.get_text(" ", strip=True))
        if not title:
            return None

        authors = self._extract_authors(soup)
        abstract = self._extract_abstract(soup)
        pdf_url = self._extract_pdf_url(soup)
        doi = self._meta_content(soup, "citation_doi") or None

        paper_id = self._paper_id(doi, detail_url, title, year)

        return Paper(
            id=paper_id,
            title=title,
            abstract=abstract,
            authors=authors,
            keywords=[],
            year=year,
            venue="IJCAI",
            venue_type="conference",
            source_platform=self.platform_name,
            crawl_date=datetime.now().isoformat(),
            pdf_url=pdf_url,
            doi=doi,
            download_available="openreview" if pdf_url else "none",
        )

    def _meta_content(self, soup: BeautifulSoup, name: str) -> str:
        node = soup.find("meta", attrs={"name": name})
        if node and node.get("content"):
            return self._clean_text(node["content"])
        return ""

    def _extract_authors(self, soup: BeautifulSoup) -> List[str]:
        authors: List[str] = []
        for meta in soup.find_all("meta", attrs={"name": "citation_author"}):
            value = self._clean_text(meta.get("content", ""))
            if value and value not in authors:
                authors.append(value)

        if authors:
            return authors

        author_nodes = soup.select(
            ".authors, .item.authors, .paper-authors, [class*='author'], [data-testid*='author']"
        )
        for node in author_nodes:
            text = self._clean_text(node.get_text(" ", strip=True))
            if text and text not in authors:
                authors.append(text)

        if len(authors) == 1 and ("," in authors[0] or " and " in authors[0]):
            authors = [x.strip() for x in re.split(r",| and ", authors[0]) if x.strip()]

        return authors

    def _extract_abstract(self, soup: BeautifulSoup) -> str:
        abstract = self._meta_content(soup, "citation_abstract")
        if abstract:
            return abstract

        abstract_node = soup.select_one(
            "div.abstract, section.abstract, article .abstract, #abstract, [class*='abstract']"
        )
        if abstract_node:
            text = self._clean_text(abstract_node.get_text(" ", strip=True))
            return re.sub(r"^abstract[:\s]*", "", text, flags=re.IGNORECASE)

        detail = soup.select_one("div.proceedings-detail")
        if detail:
            rows = detail.find_all("div", class_="row", recursive=False)
            title_text = ""
            title_node = soup.select_one("h1")
            if title_node:
                title_text = self._clean_text(title_node.get_text(" ", strip=True))
            for row in rows:
                text = self._clean_text(row.get_text(" ", strip=True))
                if len(text) < 120:
                    continue
                if "PDF" in text and "BibTeX" in text:
                    continue
                if text.startswith("Proceedings of the"):
                    continue
                if title_text and text == title_text:
                    continue
                return text

        for heading in soup.find_all(["h2", "h3", "h4", "strong"]):
            heading_text = self._clean_text(heading.get_text(" ", strip=True))
            if heading_text.lower().startswith("abstract"):
                sibling = heading.find_next_sibling()
                while sibling and sibling.name in {"br", "hr"}:
                    sibling = sibling.find_next_sibling()
                if sibling:
                    text = self._clean_text(sibling.get_text(" ", strip=True))
                    if text:
                        return re.sub(r"^abstract[:\s]*", "", text, flags=re.IGNORECASE)

        return ""

    def _extract_pdf_url(self, soup: BeautifulSoup) -> Optional[str]:
        pdf = self._meta_content(soup, "citation_pdf_url")
        if pdf:
            return self._abs_url(pdf)

        for link in soup.find_all("a", href=True):
            href = self._abs_url(link["href"])
            if href.lower().endswith(".pdf"):
                return href
        return None

    def _paper_id(self, doi: Optional[str], detail_url: str, title: str, year: int) -> str:
        if doi:
            return doi.replace("/", "_").replace(".", "_")

        path_tail = detail_url.rstrip("/").split("/")[-1]
        if path_tail.isdigit():
            return f"ijcai_{year}_{path_tail}"

        digest = hashlib.md5(f"{title}_{year}".encode("utf-8")).hexdigest()[:10]
        return f"ijcai_{year}_{digest}"

    def _clean_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", (text or "")).strip()

    def _abs_url(self, href: str) -> str:
        if not href:
            return ""
        if href.startswith("http://") or href.startswith("https://"):
            return href
        if href.startswith("/"):
            return f"{self.BASE_URL}{href}"
        return f"{self.BASE_URL}/{href}"

    def get_pdf_url(self, paper_id: str, **kwargs) -> Optional[str]:
        doi = kwargs.get("doi")
        if doi:
            return f"https://doi.org/{doi}"
        return kwargs.get("url")

    def rate_limit_delay(self) -> float:
        return 1.0

    def check_availability(self) -> bool:
        try:
            response = requests.get(
                self.PROCEEDINGS_URL.format(year=datetime.now().year),
                timeout=20,
                headers={"User-Agent": "paper-agent/1.0 (+https://github.com/)"},
            )
            return response.status_code == 200
        except requests.RequestException:
            return False
