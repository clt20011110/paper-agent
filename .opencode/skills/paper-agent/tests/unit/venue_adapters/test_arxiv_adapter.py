#!/usr/bin/env python3
"""
Unit tests for arXiv Adapter.

Tests the ArxivAdapter class with mocked API responses.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
from pathlib import Path

# Add lib to path - same pattern as conftest.py
# test file is at: .../tests/unit/adapters/test_arxiv_adapter.py
# lib is at: .../lib
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'lib'))

# Import after path setup
from adapters.base import VenueAdapter, VenueConfig
from adapters.registry import AdapterRegistry
from adapters.arxiv_adapter import ArxivAdapter, search_arxiv_for_paper, _title_similarity
from database import Paper


class TestArxivAdapterRegistration:
    """Test arXiv adapter registration with the registry."""
    
    def test_adapter_is_registered(self):
        """Test that arXiv adapter is registered in AdapterRegistry."""
        # Ensure registry is initialized
        from adapters.registry import register_builtin_adapters
        register_builtin_adapters()
        
        assert AdapterRegistry.is_registered('arxiv')
    
    def test_get_adapter_from_registry(self):
        """Test retrieving arXiv adapter from registry."""
        # Ensure registry is initialized
        from adapters.registry import register_builtin_adapters
        register_builtin_adapters()
        
        adapter = AdapterRegistry.get('arxiv')
        
        assert adapter is not None
        assert isinstance(adapter, ArxivAdapter)
        assert adapter.platform_name == 'arxiv'


class TestArxivAdapterProperties:
    """Test arXiv adapter properties."""
    
    @pytest.fixture
    def adapter(self):
        """Create arXiv adapter instance."""
        return ArxivAdapter()
    
    def test_platform_name(self, adapter):
        """Test platform_name property."""
        assert adapter.platform_name == 'arxiv'
    
    def test_venue_type(self, adapter):
        """Test venue_type property."""
        assert adapter.venue_type == 'preprint'
    
    def test_rate_limit_delay(self, adapter):
        """Test rate_limit_delay method - arXiv requires 3 seconds."""
        assert adapter.rate_limit_delay() == 3.0
    
    def test_get_supported_venues(self, adapter):
        """Test get_supported_venues returns expected venues."""
        venues = adapter.get_supported_venues()
        
        assert 'arXiv' in venues
        assert 'arxiv' in venues


class TestPDFURLGeneration:
    """Test PDF URL generation."""
    
    @pytest.fixture
    def adapter(self):
        """Create arXiv adapter instance."""
        return ArxivAdapter()
    
    def test_get_pdf_url_standard(self, adapter):
        """Test PDF URL generation for standard arXiv ID."""
        paper_id = '2401.12345'
        url = adapter.get_pdf_url(paper_id)
        
        assert url == 'https://arxiv.org/pdf/2401.12345.pdf'
    
    def test_get_pdf_url_with_prefix(self, adapter):
        """Test PDF URL generation with arxiv: prefix."""
        paper_id = 'arxiv:2401.12345'
        url = adapter.get_pdf_url(paper_id)
        
        assert url == 'https://arxiv.org/pdf/2401.12345.pdf'
    
    def test_get_pdf_url_with_version(self, adapter):
        """Test PDF URL generation for versioned arXiv ID."""
        paper_id = '2401.12345v2'
        url = adapter.get_pdf_url(paper_id)
        
        # Should include version in URL
        assert '2401.12345v2' in url
        assert url.endswith('.pdf')


class TestEntryParsing:
    """Test arXiv Atom entry parsing."""
    
    @pytest.fixture
    def adapter(self):
        """Create arXiv adapter instance."""
        return ArxivAdapter()
    
    def test_parse_entry_full(self, adapter):
        """Test parsing a full arXiv entry."""
        entry = Mock()
        entry.id = 'http://arxiv.org/abs/2401.12345'
        entry.title = 'Test Paper Title'
        entry.summary = 'This is a test abstract.'
        # Create mock authors that have a 'name' attribute
        author1 = Mock()
        author1.name = 'John Doe'
        author2 = Mock()
        author2.name = 'Jane Smith'
        entry.authors = [author1, author2]
        # Create mock tags that have a 'term' attribute
        tag1 = Mock()
        tag1.term = 'cs.AI'
        tag2 = Mock()
        tag2.term = 'cs.LG'
        entry.tags = [tag1, tag2]
        entry.links = []
        entry.published_parsed = Mock(tm_year=2024)
        
        paper = adapter._parse_entry(entry)
        
        assert paper is not None
        assert paper.id == 'arxiv:2401.12345'
        assert paper.title == 'Test Paper Title'
        assert paper.abstract == 'This is a test abstract.'
        assert paper.authors == ['John Doe', 'Jane Smith']
        assert paper.year == 2024
        assert paper.venue == 'arXiv'
        assert paper.venue_type == 'preprint'
        assert paper.source_platform == 'arxiv'
        assert paper.arxiv_id == '2401.12345'
        assert paper.download_available == 'arxiv'
        assert 'arxiv.org/pdf/2401.12345' in paper.pdf_url
    
    def test_parse_entry_minimal(self, adapter):
        """Test parsing a minimal arXiv entry."""
        entry = Mock()
        entry.id = 'http://arxiv.org/abs/2401.12346'
        entry.title = 'Minimal Paper'
        entry.summary = ''
        entry.authors = []
        entry.tags = []
        entry.links = []
        # No published_parsed - should use current year
        del entry.published_parsed
        
        paper = adapter._parse_entry(entry)
        
        assert paper is not None
        assert paper.id == 'arxiv:2401.12346'
        assert paper.title == 'Minimal Paper'
        assert paper.authors == []
    
    def test_parse_entry_with_doi(self, adapter):
        """Test parsing entry with DOI link."""
        entry = Mock()
        entry.id = 'http://arxiv.org/abs/2401.12347'
        entry.title = 'Paper with DOI'
        entry.summary = 'Abstract'
        entry.authors = []
        entry.tags = []
        entry.links = [
            {'type': 'text/doi', 'href': 'https://doi.org/10.1234/test'}
        ]
        entry.published_parsed = Mock(tm_year=2024)
        
        paper = adapter._parse_entry(entry)
        
        assert paper is not None
        assert paper.doi == 'https://doi.org/10.1234/test'
    
    def test_parse_entry_with_version(self, adapter):
        """Test parsing entry with versioned arXiv ID."""
        entry = Mock()
        entry.id = 'http://arxiv.org/abs/2401.12348v2'
        entry.title = 'Versioned Paper'
        entry.summary = 'Abstract'
        entry.authors = []
        entry.tags = []
        entry.links = []
        entry.published_parsed = Mock(tm_year=2024)
        
        paper = adapter._parse_entry(entry)
        
        assert paper is not None
        assert '2401.12348' in paper.arxiv_id


class TestCrawling:
    """Test arXiv crawling functionality."""
    
    @pytest.fixture
    def adapter(self):
        """Create arXiv adapter instance."""
        return ArxivAdapter()
    
    @pytest.fixture
    def mock_feed_response(self):
        """Create a mock feedparser response."""
        mock_feed = Mock()
        mock_entry = Mock()
        mock_entry.id = 'http://arxiv.org/abs/2401.12345'
        mock_entry.title = 'Test Paper on Machine Learning'
        mock_entry.summary = 'This paper presents a novel approach to machine learning.'
        mock_entry.authors = [Mock(name='Test Author')]
        mock_entry.tags = [Mock(term='cs.LG')]
        mock_entry.links = []
        mock_entry.published_parsed = Mock(tm_year=2024)
        
        mock_feed.entries = [mock_entry]
        return mock_feed
    
    def test_crawl_returns_papers(self, adapter, mock_feed_response):
        """Test that crawl returns Paper objects."""
        config = VenueConfig(
            name='arXiv Search',
            years=[2024],
            platform='arxiv',
            additional_params={
                'keywords': 'machine learning',
                'categories': ['cs.LG'],
                'max_results': 10
            }
        )
        
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.content = b'<?xml version="1.0"?><feed></feed>'
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response
            
            with patch('feedparser.parse', return_value=mock_feed_response):
                papers = adapter.crawl(config)
                
                assert isinstance(papers, list)
                assert len(papers) >= 1
                assert isinstance(papers[0], Paper)
    
    def test_crawl_respects_rate_limit(self, adapter, mock_feed_response):
        """Test that crawl respects arXiv rate limit."""
        config = VenueConfig(
            name='arXiv Search',
            years=[2024],
            platform='arxiv',
            additional_params={
                'keywords': 'test',
                'max_results': 5
            }
        )
        
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.content = b'<?xml version="1.0"?><feed></feed>'
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response
            
            with patch('feedparser.parse', return_value=mock_feed_response):
                with patch('time.sleep') as mock_sleep:
                    adapter.crawl(config)
                    
                    # Should have called sleep after the request
                    assert mock_sleep.called


class TestTitleSimilarity:
    """Test title similarity calculation."""
    
    def test_identical_titles(self):
        """Test similarity of identical titles."""
        title = "Deep Learning for Natural Language Processing"
        score = _title_similarity(title, title)
        
        assert score == 1.0
    
    def test_similar_titles(self):
        """Test similarity of similar titles."""
        title1 = "Deep Learning for NLP"
        title2 = "Deep Learning for Natural Language Processing"
        score = _title_similarity(title1, title2)
        
        # "Deep", "Learning", "for" are common = 3 words
        # Total unique words: "Deep", "Learning", "for", "NLP", "Natural", "Language", "Processing" = 7
        # Jaccard similarity = 3/7 ≈ 0.43
        assert score > 0.3  # Lowered threshold for Jaccard similarity
    
    def test_different_titles(self):
        """Test similarity of different titles."""
        title1 = "Deep Learning for NLP"
        title2 = "Quantum Computing Applications"
        score = _title_similarity(title1, title2)
        
        assert score < 0.3
    
    def test_empty_titles(self):
        """Test similarity with empty titles."""
        assert _title_similarity("", "Some Title") == 0.0
        assert _title_similarity("Some Title", "") == 0.0
        assert _title_similarity("", "") == 0.0


class TestSearchByTitle:
    """Test search_by_title helper method."""
    
    @pytest.fixture
    def adapter(self):
        """Create arXiv adapter instance."""
        return ArxivAdapter()
    
    def test_search_by_title_finds_match(self, adapter):
        """Test search_by_title finds matching paper."""
        mock_paper = Paper(
            id='arxiv:2401.12345',
            title='Deep Learning for Natural Language Processing',
            abstract='Abstract',
            authors=[],
            keywords=[],
            year=2024
        )
        
        with patch.object(adapter, 'crawl', return_value=[mock_paper]):
            result = adapter.search_by_title('Deep Learning for NLP')
            
            # Should find the paper due to title similarity
            assert result is not None
    
    def test_search_by_title_no_match(self, adapter):
        """Test search_by_title with no matching paper."""
        mock_paper = Paper(
            id='arxiv:2401.12345',
            title='Quantum Computing Fundamentals',
            abstract='Abstract',
            authors=[],
            keywords=[],
            year=2024
        )
        
        with patch.object(adapter, 'crawl', return_value=[mock_paper]):
            result = adapter.search_by_title('Deep Learning for NLP')
            
            # Should not match due to low similarity
            assert result is None


class TestVenueConfig:
    """Test VenueConfig for arXiv."""
    
    def test_venue_config_creation(self):
        """Test creating VenueConfig for arXiv."""
        config = VenueConfig(
            name='arXiv Search',
            years=[2024, 2023],
            platform='arxiv',
            additional_params={
                'keywords': 'diffusion models',
                'categories': ['cs.AI', 'cs.LG'],
                'max_results': 100
            }
        )
        
        assert config.name == 'arXiv Search'
        assert config.years == [2024, 2023]
        assert config.platform == 'arxiv'
        assert config.additional_params['keywords'] == 'diffusion models'
    
    def test_venue_config_from_dict(self):
        """Test creating VenueConfig from dictionary."""
        config = VenueConfig.from_dict({
            'name': 'arXiv Search',
            'years': [2024],
            'platform': 'arxiv',
            'additional_params': {
                'keywords': 'test',
                'max_results': 50
            }
        })
        
        assert config.name == 'arXiv Search'
        assert config.years == [2024]
        assert config.additional_params['max_results'] == 50


class TestAvailabilityCheck:
    """Test availability checking."""
    
    def test_check_availability_with_libraries(self):
        """Test availability when required libraries are installed."""
        adapter = ArxivAdapter()
        
        with patch.dict('sys.modules', {
            'requests': MagicMock(),
            'feedparser': MagicMock()
        }):
            result = adapter.check_availability()
            assert result is True


class TestSearchArxivForPaperFunction:
    """Test the convenience function search_arxiv_for_paper."""
    
    def test_search_arxiv_for_paper(self):
        """Test the search_arxiv_for_paper function."""
        mock_paper = Paper(
            id='arxiv:2401.12345',
            title='Deep Learning for Natural Language Processing',
            abstract='Abstract',
            authors=[],
            keywords=[],
            year=2024
        )
        
        with patch('adapters.arxiv_adapter.ArxivAdapter') as MockAdapter:
            mock_adapter = Mock()
            mock_adapter.crawl.return_value = [mock_paper]
            MockAdapter.return_value = mock_adapter
            
            result = search_arxiv_for_paper('Deep Learning for NLP')
            
            assert mock_adapter.crawl.called


# Run tests if executed directly
if __name__ == '__main__':
    pytest.main([__file__, '-v'])