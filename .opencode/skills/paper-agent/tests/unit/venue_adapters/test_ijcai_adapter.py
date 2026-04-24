"""Tests for the IJCAI adapter."""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "lib"))

from adapters import AdapterRegistry, VenueConfig
from adapters.ijcai_adapter import IJCAIAdapter


def _response(text: str) -> Mock:
    resp = Mock()
    resp.text = text
    resp.raise_for_status = Mock()
    return resp


class TestIJCAIRegistration:
    def test_registered(self):
        assert AdapterRegistry.is_registered("ijcai")

    def test_get_adapter(self):
        adapter = AdapterRegistry.get("ijcai")
        assert isinstance(adapter, IJCAIAdapter)


class TestIJCAIExtraction:
    @pytest.fixture
    def adapter(self):
        return IJCAIAdapter()

    def test_crawl_year_extracts_abstract_from_detail_page(self, adapter):
        proceedings_html = """
        <html><body>
          <a href="/proceedings/2025/1231">Details</a>
        </body></html>
        """
        detail_html = """
        <html><head>
          <meta name="citation_title" content="Sample IJCAI Paper" />
          <meta name="citation_author" content="Alice Example" />
          <meta name="citation_author" content="Bob Example" />
          <meta name="citation_abstract" content="This is the IJCAI abstract." />
          <meta name="citation_pdf_url" content="https://www.ijcai.org/proceedings/2025/1231.pdf" />
          <meta name="citation_doi" content="10.24963/ijcai.2025/1231" />
        </head><body></body></html>
        """

        def fake_get(url, headers=None, timeout=None):
            if url.endswith("/proceedings/2025/"):
                return _response(proceedings_html)
            if url.endswith("/proceedings/2025/1231"):
                return _response(detail_html)
            raise AssertionError(f"unexpected url: {url}")

        with patch("adapters.ijcai_adapter.requests.get", side_effect=fake_get):
            papers = adapter._crawl_year(2025)

        assert len(papers) == 1
        paper = papers[0]
        assert paper.title == "Sample IJCAI Paper"
        assert paper.abstract == "This is the IJCAI abstract."
        assert paper.authors == ["Alice Example", "Bob Example"]
        assert paper.pdf_url == "https://www.ijcai.org/proceedings/2025/1231.pdf"
        assert paper.doi == "10.24963/ijcai.2025/1231"
        assert paper.source_platform == "ijcai"
        assert paper.venue == "IJCAI"

    def test_validate_config(self, adapter):
        config = VenueConfig(name="IJCAI", years=[2025], platform="ijcai")
        assert adapter.validate_config(config) is True
