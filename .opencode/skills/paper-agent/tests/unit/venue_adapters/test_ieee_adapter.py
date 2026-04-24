"""Tests for IEEE Xplore EDA adapters."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "lib"))

from adapters.ieee_adapter import DACAdapter


class TestIEEEDownloadLinks:
    @pytest.fixture
    def adapter(self):
        return DACAdapter()

    def test_parse_article_uses_article_number_for_pdf_url(self, adapter):
        article = {
            "title": "Sample DAC Paper",
            "abstract": "<p>Sample abstract.</p>",
            "doi": "10.1109/DAC1234.2024.9999999",
            "article_number": "9999999",
            "html_url": "https://ieeexplore.ieee.org/document/9999999",
            "authors": {"authors": [{"full_name": "Alice Example"}]},
        }

        paper = adapter._parse_article(article, 2024)

        assert paper is not None
        assert paper.abstract == "Sample abstract."
        assert paper.pdf_url == "https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9999999"
        assert paper.download_available == "openreview"

    def test_parse_article_prefers_api_pdf_url(self, adapter):
        article = {
            "title": "Sample DAC Paper",
            "doi": "10.1109/DAC1234.2024.9999999",
            "article_number": "9999999",
            "pdf_url": "https://example.org/direct.pdf",
            "authors": {"authors": []},
        }

        paper = adapter._parse_article(article, 2024)

        assert paper is not None
        assert paper.pdf_url == "https://example.org/direct.pdf"
