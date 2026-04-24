#!/usr/bin/env python3
"""
AAAI Proceedings Adapter (official AAAI OJS site).

Fetches paper list from AAAI official proceedings pages hosted on OJS:
https://ojs.aaai.org/index.php/AAAI
"""

import hashlib
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Set
import sys

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# Add parent directory to path for imports
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from database import Paper
from .base import VenueAdapter, VenueConfig


logger = logging.getLogger(__name__)


class AAAIAdapter(VenueAdapter):
    """Adapter for AAAI conference papers from the official AAAI OJS website."""

    BASE_URL = "https://ojs.aaai.org"
    ARCHIVE_URL = f"{BASE_URL}/index.php/AAAI/issue/archive"

    @property
    def platform_name(self) -> str:
        return "aaai"

    @property
    def venue_type(self) -> str:
        return "conference"

    def crawl(self, config: VenueConfig) -> List[Paper]:
        papers: List[Paper] = []
        seen_ids: Set[str] = set()

        for year in config.years:
            try:
                year_papers = self._crawl_year(year)
                for p in year_papers:
                    if p.id not in seen_ids:
                        seen_ids.add(p.id)
                        papers.append(p)
                abstract_count = sum(1 for p in year_papers if p.abstract)
                logger.info(
                    f"AAAI {year}: papers={len(year_papers)}, abstracts={abstract_count}"
                )
            except Exception as e:
                logger.error(f"AAAI {year} crawl failed: {e}")
            time.sleep(self.rate_limit_delay())

        return papers

    def _crawl_year(self, year: int) -> List[Paper]:
        archive_html = self._fetch_html(self.ARCHIVE_URL, timeout=60)
        if not archive_html:
            return []

        soup = BeautifulSoup(archive_html, "html.parser")
        issue_urls = self._extract_issue_urls_for_year(soup, year)
        logger.info(f"AAAI {year}: found {len(issue_urls)} issue pages")

        papers: List[Paper] = []
        seen: Set[str] = set()

        for issue_url in tqdm(issue_urls):
            issue_html = self._fetch_html(issue_url, timeout=60)
            if not issue_html:
                continue
            issue_soup = BeautifulSoup(issue_html, "html.parser")
            issue_papers = self._parse_issue_page(issue_soup, year)
            for p in issue_papers:
                if p.id not in seen:
                    seen.add(p.id)
                    papers.append(p)
            # time.sleep(self.rate_limit_delay())

        return papers

    def _extract_issue_urls_for_year(self, soup: BeautifulSoup, year: int) -> List[str]:
        issue_urls: List[str] = []
        year_text = str(year)
        short_year_text = year_text[-2:]

        for issue_block in soup.select("div.obj_issue_summary, div.issue-summary"):
            block_text = issue_block.get_text(" ", strip=True)
            if year_text not in block_text and f"AAAI-{short_year_text}" not in block_text:
                continue
            link = issue_block.find("a", href=True)
            if not link:
                continue
            url = self._abs_url(link["href"])
            if "/AAAI/issue/view/" in url:
                issue_urls.append(url)

        if not issue_urls:
            for link in soup.find_all("a", href=True):
                href = self._abs_url(link["href"])
                if "/AAAI/issue/view/" not in href:
                    continue
                context = " ".join(
                    [
                        link.get_text(" ", strip=True),
                        link.parent.get_text(" ", strip=True) if link.parent else "",
                    ]
                )
                if year_text in context or f"AAAI-{short_year_text}" in context:
                    issue_urls.append(href)

        # De-duplicate while preserving order.
        dedup: List[str] = []
        seen: Set[str] = set()
        for u in issue_urls:
            if u not in seen:
                seen.add(u)
                dedup.append(u)
        return dedup

    def _parse_issue_page(self, soup: BeautifulSoup, year: int) -> List[Paper]:
        papers: List[Paper] = []

        article_blocks = soup.select("div.obj_article_summary, article")
        article_urls: List[str] = []

        for block in article_blocks:
            article_url = self._extract_article_url(block)
            if article_url:
                article_urls.append(article_url)

        if not article_urls:
            # Fallback for pages that only provide plain links.
            for link in soup.find_all("a", href=True):
                href = self._abs_url(link["href"])
                if "/papers/" not in href and "/article/view/" not in href:
                    continue
                article_urls.append(href)

        seen: Set[str] = set()
        for article_url in tqdm(article_urls):
            if article_url in seen:
                continue
            seen.add(article_url)

            article_html = self._fetch_html(article_url, timeout=60)
            if article_html:
                article_soup = BeautifulSoup(article_html, "html.parser")
                paper = self._parse_article_page(article_soup, article_url, year)
                if paper:
                    papers.append(paper)
                    continue

            # If the detail page cannot be fetched, fall back to whatever
            # metadata was exposed directly on the listing page.
            fallback = self._parse_article_block(
                self._find_article_block_for_url(article_blocks, article_url), year
            )
            if fallback:
                papers.append(fallback)

        return papers

    def _parse_article_block(self, block, year: int) -> Optional[Paper]:
        if block is None:
            return None

        title_link = block.select_one(
            "h3.title a, h4.title a, a.title, a[href*='/AAAI/article/view/'], a[href*='/papers/']"
        )
        if not title_link:
            return None

        title = self._clean_text(title_link.get_text(" ", strip=True))
        if not title:
            return None

        article_url = self._abs_url(title_link.get("href", ""))
        if "/AAAI/article/view/" not in article_url:
            return None

        authors: List[str] = []
        authors_node = block.select_one("div.authors, p.authors, .meta .authors")
        if authors_node:
            raw = authors_node.get_text(" ", strip=True)
            raw = re.sub(r"\s+", " ", raw)
            authors = [x.strip() for x in re.split(r",| and ", raw) if x.strip()]

        abstract = ""
        abstract_node = block.select_one("div.abstract, p.abstract")
        if abstract_node:
            abstract = self._clean_text(abstract_node.get_text(" ", strip=True))

        pdf_url = None
        pdf_link = block.find("a", href=True, string=re.compile(r"pdf", re.IGNORECASE))
        if pdf_link:
            pdf_url = self._abs_url(pdf_link["href"])

        paper_id = self._paper_id_from_url(article_url, title, year)
        return Paper(
            id=paper_id,
            title=title,
            abstract=abstract,
            authors=authors,
            keywords=[],
            year=year,
            venue="AAAI",
            venue_type="conference",
            source_platform=self.platform_name,
            crawl_date=datetime.now().isoformat(),
            pdf_url=pdf_url,
            download_available="openreview" if pdf_url else "none",
        )

    def _parse_article_page(self, soup: BeautifulSoup, article_url: str, year: int) -> Optional[Paper]:
        title = self._meta_content(soup, "citation_title")
        if not title:
            title_node = soup.select_one("h1, h2, h3, .page_title, .pkp_title, .title")
            if title_node:
                title = self._clean_text(title_node.get_text(" ", strip=True))
        if not title:
            return None

        authors = self._extract_authors(soup)
        abstract = self._extract_abstract(soup)
        pdf_url = self._extract_pdf_url(soup)

        paper_id = self._paper_id_from_url(article_url, title, year)
        return Paper(
            id=paper_id,
            title=title,
            abstract=abstract,
            authors=authors,
            keywords=[],
            year=year,
            venue="AAAI",
            venue_type="conference",
            source_platform=self.platform_name,
            crawl_date=datetime.now().isoformat(),
            pdf_url=pdf_url,
            download_available="openreview" if pdf_url else "none",
        )

    def _extract_article_url(self, block) -> Optional[str]:
        if block is None:
            return None
        link = block.select_one("h3.title a, h4.title a, a.title, a[href*='/AAAI/article/view/'], a[href*='/papers/']")
        if link and link.get("href"):
            href = self._abs_url(link["href"])
            if "/papers/" in href or "/article/view/" in href:
                return href
        return None

    def _find_article_block_for_url(self, blocks, article_url: str):
        for block in blocks:
            link = block.select_one("a[href]")
            if link and self._abs_url(link.get("href", "")) == article_url:
                return block
        return None

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
            ".authors, .article-details .authors, .item.authors, [class*='author'], [data-testid*='author']"
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
            text = re.sub(r"^abstract[:\s]*", "", text, flags=re.IGNORECASE)
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

    def _fetch_html(self, url: str, timeout: int = 30) -> Optional[str]:
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as e:
            logger.warning(f"Failed to fetch {url}: {e}")
            return None

    def _paper_id_from_url(self, article_url: str, title: str, year: int) -> str:
        m = re.search(r"/article/view/(\d+)", article_url)
        if m:
            return f"aaai_{year}_{m.group(1)}"
        slug_match = re.search(r"/papers/([^/?#]+)/?", article_url)
        if slug_match:
            slug = slug_match.group(1).strip("/")
            if slug:
                return f"aaai_{year}_{slug}"
        digest = hashlib.md5(f"{title}_{year}".encode("utf-8")).hexdigest()[:10]
        return f"aaai_{year}_{digest}"

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

    def get_pdf_url(self, paper_id: str, **kwargs) -> Optional[str]:
        return kwargs.get("pdf_url")

    def rate_limit_delay(self) -> float:
        return 1.0

    def get_supported_venues(self) -> List[str]:
        return ["AAAI"]
