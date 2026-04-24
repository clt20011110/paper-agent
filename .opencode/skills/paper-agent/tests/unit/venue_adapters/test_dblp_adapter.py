"""Tests for DBLP EDA adapter metadata enrichment."""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "lib"))

from adapters.dblp_adapter import DBLP_DAC_Adapter
from database import Paper


class TestDBLPEnrichment:
    @pytest.fixture
    def adapter(self):
        return DBLP_DAC_Adapter()

    def test_semantic_scholar_enrichment_adds_abstract_and_pdf(self, adapter):
        paper = Paper(
            id="10_1145_1234567_8901234",
            title="A Fast Timing Closure Method for Chip Design",
            abstract="",
            year=2024,
            venue="DAC",
            source_platform="dblp_dac",
            doi="10.1145/1234567.8901234",
            pdf_url="https://doi.org/10.1145/1234567.8901234",
            download_available="openreview",
        )

        response = Mock()
        response.raise_for_status = Mock()
        response.json.return_value = [
            {
                "title": "A Fast Timing Closure Method for Chip Design",
                "abstract": "This paper presents a fast timing closure method.",
                "externalIds": {"DOI": "10.1145/1234567.8901234", "ArXiv": "2401.12345"},
                "openAccessPdf": {"url": "https://arxiv.org/pdf/2401.12345.pdf"},
                "isOpenAccess": True,
            }
        ]

        with patch("adapters.dblp_adapter.requests.post", return_value=response) as mock_post:
            adapter._enrich_with_semantic_scholar([paper], min_delay=0)

        assert paper.abstract == "This paper presents a fast timing closure method."
        assert paper.arxiv_id == "2401.12345"
        assert paper.pdf_url == "https://arxiv.org/pdf/2401.12345.pdf"
        assert paper.download_available == "arxiv"
        mock_post.assert_called_once()
        payload = mock_post.call_args.kwargs["json"]
        assert payload == {"ids": ["DOI:10.1145/1234567.8901234"]}

    def test_semantic_scholar_enrichment_skips_title_mismatch(self, adapter):
        paper = Paper(
            id="10_1145_1234567_8901234",
            title="A Fast Timing Closure Method for Chip Design",
            doi="10.1145/1234567.8901234",
        )

        response = Mock()
        response.raise_for_status = Mock()
        response.json.return_value = [
            {
                "title": "Completely Different Biomedical Study",
                "abstract": "Wrong paper.",
                "openAccessPdf": {"url": "https://example.org/wrong.pdf"},
            }
        ]

        with patch("adapters.dblp_adapter.requests.post", return_value=response):
            adapter._enrich_with_semantic_scholar([paper], min_delay=0)

        assert paper.abstract == ""
        assert paper.pdf_url is None
        assert paper.download_available == "none"
