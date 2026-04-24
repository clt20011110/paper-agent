#!/usr/bin/env python3
"""Crossref journal adapters for Stage 1 expansion.

No API key is required. These adapters are metadata-first and may not always
return abstracts or direct PDF URLs depending on publisher metadata quality.
"""

from __future__ import annotations

import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests

from database import Paper
from .base import VenueAdapter, VenueConfig


class CrossrefJournalAdapter(VenueAdapter):
    API_BASE_URL = "https://api.crossref.org/works"

    def __init__(self, venue_name: str, issn: str, platform: str):
        self._venue_name = venue_name
        self._issn = issn
        self._platform = platform

    @property
    def platform_name(self) -> str:
        return self._platform

    @property
    def venue_type(self) -> str:
        return "journal"

    def get_supported_venues(self) -> List[str]:
        return [self._venue_name]

    def supports_year(self, year: int) -> bool:
        current_year = datetime.now().year
        return 1900 <= year <= current_year + 1

    def crawl(self, config: VenueConfig) -> List[Paper]:
        self.validate_config(config)
        papers: List[Paper] = []

        for year in config.years:
            papers.extend(self._crawl_year(year, config.additional_params))
            time.sleep(self.rate_limit_delay())

        return papers

    def _crawl_year(self, year: int, options: Dict[str, Any]) -> List[Paper]:
        rows = int(options.get("rows", 200))
        rows = max(20, min(rows, 1000))

        cursor = "*"
        papers: List[Paper] = []

        while True:
            params = {
                "filter": f"issn:{self._issn},from-pub-date:{year}-01-01,until-pub-date:{year}-12-31",
                "rows": rows,
                "cursor": cursor,
                "select": "DOI,title,abstract,author,published-print,published-online,issued,link,URL,subject,container-title",
                "mailto": options.get("mailto", "paper-agent@example.com"),
            }

            response = requests.get(self.API_BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            payload = response.json().get("message", {})

            items = payload.get("items", [])
            if not items:
                break

            for item in items:
                paper = self._parse_item(item, year)
                if paper:
                    papers.append(paper)

            next_cursor = payload.get("next-cursor")
            if not next_cursor or next_cursor == cursor:
                break
            cursor = next_cursor

        return papers

    def _parse_item(self, item: Dict[str, Any], fallback_year: int) -> Optional[Paper]:
        doi = item.get("DOI")
        titles = item.get("title", [])
        title = titles[0].strip() if titles else ""
        if not title:
            return None

        paper_id = self._paper_id(doi, title, fallback_year)

        abstract_raw = item.get("abstract", "")
        abstract = self._clean_abstract(abstract_raw)

        authors: List[str] = []
        for author in item.get("author", []):
            given = (author.get("given") or "").strip()
            family = (author.get("family") or "").strip()
            name = f"{given} {family}".strip()
            if name:
                authors.append(name)

        keywords = item.get("subject", [])
        if not isinstance(keywords, list):
            keywords = []

        year = self._extract_year(item) or fallback_year

        pdf_url = None
        for link in item.get("link", []):
            if isinstance(link, dict):
                content_type = (link.get("content-type") or "").lower()
                if "pdf" in content_type:
                    pdf_url = link.get("URL")
                    break

        if not pdf_url:
            url = item.get("URL")
            if isinstance(url, str):
                pdf_url = url

        return Paper(
            id=paper_id,
            title=title,
            abstract=abstract,
            authors=authors,
            keywords=keywords,
            year=year,
            venue=self._venue_name,
            venue_type="journal",
            source_platform=self.platform_name,
            crawl_date=datetime.now().isoformat(),
            pdf_url=pdf_url,
            doi=doi,
            download_available="openreview" if pdf_url else "none",
        )

    def _paper_id(self, doi: Optional[str], title: str, year: int) -> str:
        if doi:
            return doi.replace("/", "_").replace(".", "_")
        return f"{self.platform_name}_{year}_{abs(hash(title)) % 1000000:06d}"

    def _extract_year(self, item: Dict[str, Any]) -> Optional[int]:
        for key in ("published-print", "published-online", "issued"):
            block = item.get(key, {})
            parts = block.get("date-parts", [])
            if parts and isinstance(parts[0], list) and parts[0]:
                first = parts[0][0]
                if isinstance(first, int):
                    return first
        return None

    def _clean_abstract(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r"<[^>]+>", " ", text)
        text = text.replace("&lt;", "<").replace("&gt;", ">")
        text = text.replace("&amp;", "&")
        return " ".join(text.split()).strip()

    def get_pdf_url(self, paper_id: str, **kwargs) -> Optional[str]:
        return kwargs.get("url") or kwargs.get("pdf_url")

    def rate_limit_delay(self) -> float:
        return 1.0


class NatureComputerScienceAdapter(CrossrefJournalAdapter):
    def __init__(self):
        super().__init__("Nature Computer Science", "2662-8457", "nature_computer_science")


class NatureCatalysisAdapter(CrossrefJournalAdapter):
    def __init__(self):
        super().__init__("Nature Catalysis", "2520-1158", "nature_catalysis")


class NatureBiotechnologyAdapter(CrossrefJournalAdapter):
    def __init__(self):
        super().__init__("Nature Biotechnology", "1546-1696", "nature_biotechnology")


class NatureBiomedicalEngineeringXrefAdapter(CrossrefJournalAdapter):
    def __init__(self):
        super().__init__("Nature Biomedical Engineering", "2157-846X", "nature_biomedical_engineering")


class NatureMachineIntelligenceXrefAdapter(CrossrefJournalAdapter):
    def __init__(self):
        super().__init__("Nature Machine Intelligence", "2522-5839", "nature_machine_intelligence_xref")


class NatureChemistryXrefAdapter(CrossrefJournalAdapter):
    def __init__(self):
        super().__init__("Nature Chemistry", "1755-4349", "nature_chemistry_xref")


class NatureCommunicationsXrefAdapter(CrossrefJournalAdapter):
    def __init__(self):
        super().__init__("Nature Communications", "2041-1723", "nature_communications_xref")


class CellAdapter(CrossrefJournalAdapter):
    def __init__(self):
        super().__init__("Cell", "0092-8674", "cell")


class ScienceAdapter(CrossrefJournalAdapter):
    def __init__(self):
        super().__init__("Science", "0036-8075", "science")
