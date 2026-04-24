"""Tests for the AAAI adapter."""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "lib"))

from adapters import AdapterRegistry, VenueConfig
from adapters.aaai_adapter import AAAIAdapter


def _response(text: str) -> Mock:
    resp = Mock()
    resp.text = text
    resp.raise_for_status = Mock()
    return resp


class TestAAAIRegistration:
    def test_registered(self):
        assert AdapterRegistry.is_registered("aaai")

    def test_get_adapter(self):
        adapter = AdapterRegistry.get("aaai")
        assert isinstance(adapter, AAAIAdapter)


class TestAAAIExtraction:
    @pytest.fixture
    def adapter(self):
        return AAAIAdapter()

    def test_crawl_year_extracts_abstract_from_detail_page(self, adapter):
        archive_html = """
        <html><body>
          <div class="obj_issue_summary">
            <a href="https://aaai.org/proceeding/aaai-39-2025/AAAI/issue/view/1">AAAI-25 Technical Tracks 1</a>
          </div>
        </body></html>
        """
        issue_html = """
        <html><body>
          <article class="obj_article_summary">
            <h3 class="title">
              <a href="https://aaai.org/papers/12345-sample-paper/">Sample AAAI Paper</a>
            </h3>
          </article>
        </body></html>
        """
        detail_html = """
        <html><head>
          <meta name="citation_title" content="Sample AAAI Paper" />
          <meta name="citation_author" content="Alice Example" />
          <meta name="citation_author" content="Bob Example" />
          <meta name="citation_abstract" content="This is the AAAI abstract." />
          <meta name="citation_pdf_url" content="https://aaai.org/papers/12345-sample-paper.pdf" />
        </head><body></body></html>
        """

        def fake_get(url, headers=None, timeout=None):
            if url.endswith("/issue/archive"):
                return _response(archive_html)
            if "/AAAI/issue/view/1" in url:
                return _response(issue_html)
            if "/papers/12345-sample-paper/" in url:
                return _response(detail_html)
            raise AssertionError(f"unexpected url: {url}")

        with patch("adapters.aaai_adapter.requests.get", side_effect=fake_get):
            papers = adapter._crawl_year(2025)

        assert len(papers) == 1
        paper = papers[0]
        assert paper.title == "Sample AAAI Paper"
        assert paper.abstract == "This is the AAAI abstract."
        assert paper.authors == ["Alice Example", "Bob Example"]
        assert paper.pdf_url == "https://aaai.org/papers/12345-sample-paper.pdf"
        assert paper.source_platform == "aaai"
        assert paper.venue == "AAAI"

    def test_crawl_uses_public_config(self, adapter):
        config = VenueConfig(name="AAAI", years=[2025], platform="aaai")
        assert adapter.validate_config(config) is True
