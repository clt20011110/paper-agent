#!/usr/bin/env python3
"""
Unit tests for ACL Anthology Adapter.

Tests the ACLAdapter class with mocked anthology library.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

# Import from lib - pytest.ini pythonpath should handle this
from adapters.base import VenueAdapter, VenueConfig
from adapters.registry import AdapterRegistry
from adapters.acl_adapter import ACLAdapter
from database import Paper


class TestACLAdapterRegistration:
    """Test ACL adapter registration with the registry."""
    
    def test_adapter_is_registered(self):
        """Test that ACL adapter is registered in AdapterRegistry."""
        # The adapter should already be registered at import time
        # Just verify it's registered
        assert AdapterRegistry.is_registered('acl')
    
    def test_get_adapter_from_registry(self):
        """Test retrieving ACL adapter from registry."""
        adapter = AdapterRegistry.get('acl')
        
        assert adapter is not None
        assert isinstance(adapter, ACLAdapter)
        assert adapter.platform_name == 'acl'


class TestACLAdapterProperties:
    """Test ACL adapter properties."""
    
    @pytest.fixture
    def adapter(self):
        """Create ACL adapter instance."""
        return ACLAdapter()
    
    def test_platform_name(self, adapter):
        """Test platform_name property."""
        assert adapter.platform_name == 'acl'
    
    def test_venue_type(self, adapter):
        """Test venue_type property."""
        assert adapter.venue_type == 'conference'
    
    def test_rate_limit_delay(self, adapter):
        """Test rate_limit_delay method."""
        assert adapter.rate_limit_delay() == 1.0
    
    def test_get_supported_venues(self, adapter):
        """Test get_supported_venues returns expected venues."""
        venues = adapter.get_supported_venues()
        
        assert 'ACL' in venues
        assert 'EMNLP' in venues
        assert 'NAACL' in venues
        assert 'COLING' in venues
        assert 'TACL' in venues


class TestVenueNameMapping:
    """Test venue name mapping functionality."""
    
    @pytest.fixture
    def adapter(self):
        """Create ACL adapter instance."""
        return ACLAdapter()
    
    def test_map_venue_name_acl(self, adapter):
        """Test mapping ACL venue name."""
        result = adapter._map_venue_name('ACL')
        assert result == 'acl'
    
    def test_map_venue_name_emnlp(self, adapter):
        """Test mapping EMNLP venue name."""
        result = adapter._map_venue_name('EMNLP')
        assert result == 'emnlp'
    
    def test_map_venue_name_naacl(self, adapter):
        """Test mapping NAACL venue name."""
        result = adapter._map_venue_name('NAACL')
        assert result == 'naacl'
    
    def test_map_venue_name_case_insensitive(self, adapter):
        """Test that venue name mapping is case insensitive."""
        assert adapter._map_venue_name('acl') == 'acl'
        assert adapter._map_venue_name('ACL') == 'acl'
        assert adapter._map_venue_name('Acl') == 'acl'
    
    def test_map_venue_name_unknown(self, adapter):
        """Test mapping unknown venue name."""
        result = adapter._map_venue_name('UNKNOWN_VENUE')
        assert result is None


class TestPDFURLGeneration:
    """Test PDF URL generation."""
    
    @pytest.fixture
    def adapter(self):
        """Create ACL adapter instance."""
        return ACLAdapter()
    
    def test_get_pdf_url_standard(self, adapter):
        """Test PDF URL generation for standard paper ID."""
        paper_id = '2024.acl-long.1'
        url = adapter.get_pdf_url(paper_id)
        
        assert url == 'https://aclanthology.org/2024.acl-long.1.pdf'
    
    def test_get_pdf_url_with_pdf_suffix(self, adapter):
        """Test PDF URL generation when ID already has .pdf suffix."""
        paper_id = '2024.acl-long.1.pdf'
        url = adapter.get_pdf_url(paper_id)
        
        # Should not double the .pdf suffix
        assert url == 'https://aclanthology.org/2024.acl-long.1.pdf'
    
    def test_get_pdf_url_none(self, adapter):
        """Test PDF URL generation with None input."""
        url = adapter.get_pdf_url(None)
        assert url is None
    
    def test_get_pdf_url_empty(self, adapter):
        """Test PDF URL generation with empty string."""
        url = adapter.get_pdf_url('')
        assert url is None


class TestBibTeXURLGeneration:
    """Test BibTeX URL generation."""
    
    @pytest.fixture
    def adapter(self):
        """Create ACL adapter instance."""
        return ACLAdapter()
    
    def test_get_bibtex_url(self, adapter):
        """Test BibTeX URL generation."""
        paper_id = '2024.acl-long.1'
        url = adapter.get_bibtex_url(paper_id)
        
        assert url == 'https://aclanthology.org/2024.acl-long.1.bib'
    
    def test_get_bibtex_url_with_bib_suffix(self, adapter):
        """Test BibTeX URL generation when ID has .bib suffix."""
        paper_id = '2024.acl-long.1.bib'
        url = adapter.get_bibtex_url(paper_id)
        
        assert url == 'https://aclanthology.org/2024.acl-long.1.bib'


class TestPaperMatchesVenue:
    """Test venue matching logic."""
    
    @pytest.fixture
    def adapter(self):
        """Create ACL adapter instance."""
        return ACLAdapter()
    
    def test_paper_matches_venue_acl(self, adapter):
        """Test that ACL papers match ACL venue."""
        paper_id = '2024.acl-long.1'
        assert adapter._paper_matches_venue(paper_id, 'acl') is True
    
    def test_paper_matches_venue_emnlp(self, adapter):
        """Test that EMNLP papers match EMNLP venue."""
        paper_id = '2024.emnlp-main.5'
        assert adapter._paper_matches_venue(paper_id, 'emnlp') is True
    
    def test_paper_does_not_match_different_venue(self, adapter):
        """Test that papers don't match different venues."""
        paper_id = '2024.acl-long.1'
        assert adapter._paper_matches_venue(paper_id, 'emnlp') is False
    
    def test_paper_matches_venue_case_insensitive(self, adapter):
        """Test case insensitive venue matching."""
        paper_id = '2024.acl-long.1'
        assert adapter._paper_matches_venue(paper_id, 'ACL') is True
        assert adapter._paper_matches_venue(paper_id, 'Acl') is True


class TestGetVenueType:
    """Test venue type determination."""
    
    @pytest.fixture
    def adapter(self):
        """Create ACL adapter instance."""
        return ACLAdapter()
    
    def test_get_venue_type_conference(self, adapter):
        """Test that ACL is a conference."""
        assert adapter._get_venue_type('ACL') == 'conference'
        assert adapter._get_venue_type('EMNLP') == 'conference'
    
    def test_get_venue_type_journal(self, adapter):
        """Test that TACL is a journal."""
        assert adapter._get_venue_type('TACL') == 'journal'
        assert adapter._get_venue_type('CL') == 'journal'


class TestConfigValidation:
    """Test configuration validation."""
    
    @pytest.fixture
    def adapter(self):
        """Create ACL adapter instance."""
        return ACLAdapter()
    
    def test_validate_config_valid(self, adapter):
        """Test validation with valid configuration."""
        config = VenueConfig(name='ACL', years=[2024])
        assert adapter.validate_config(config) is True
    
    def test_validate_config_invalid_venue(self, adapter):
        """Test validation with unknown venue."""
        config = VenueConfig(name='UNKNOWN', years=[2024])
        assert adapter.validate_config(config) is False
    
    def test_validate_config_year_warning(self, adapter):
        """Test validation with unusual year (should still pass)."""
        config = VenueConfig(name='ACL', years=[2000])  # ACL Anthology started around 1965
        # Should return True but may log a warning
        assert adapter.validate_config(config) is True


class TestCheckAvailability:
    """Test availability checking."""
    
    def test_check_availability_with_library(self):
        """Test availability when anthology library is installed."""
        adapter = ACLAdapter()
        
        with patch.dict('sys.modules', {'anthology': MagicMock()}):
            # This will try to import anthology
            result = adapter.check_availability()
            assert result is True
    
    def test_check_availability_without_library(self):
        """Test availability when anthology library is not installed."""
        adapter = ACLAdapter()
        
        # Mock the import to raise ImportError
        with patch.dict('sys.modules', {'anthology': None}):
            with patch('builtins.__import__', side_effect=ImportError):
                result = adapter.check_availability()
                assert result is False


class TestCrawlWithMockedAnthology:
    """Test crawling with mocked anthology library."""
    
    @pytest.fixture
    def adapter(self):
        """Create ACL adapter instance."""
        return ACLAdapter()
    
    @pytest.fixture
    def mock_paper(self):
        """Create a mock paper object."""
        paper = Mock()
        paper.id = '2024.acl-long.1'
        paper.title = 'Test Paper Title'
        paper.abstract = 'This is a test abstract.'
        paper.authors = [
            Mock(name='John Doe'),
            Mock(name='Jane Smith')
        ]
        paper.doi = '10.1234/test.001'
        paper.bibtex = '@article{test, title="Test"}'
        return paper
    
    def test_crawl_returns_papers(self, adapter, mock_paper):
        """Test that crawl returns Paper objects."""
        config = VenueConfig(name='ACL', years=[2024])
        
        # Create mock anthology
        mock_anthology = Mock()
        mock_anthology.papers = [mock_paper]
        mock_anthology_class = Mock(return_value=mock_anthology)
        
        # Mock the anthology module and Anthology class
        with patch.dict('sys.modules', {'anthology': Mock(Anthology=mock_anthology_class)}):
            papers = adapter.crawl(config)
            
            # Should have returned some papers
            assert isinstance(papers, list)
    
    def test_crawl_raises_import_error_without_library(self, adapter):
        """Test that crawl raises ImportError if library not installed."""
        config = VenueConfig(name='ACL', years=[2024])
        
        # Mock the import to raise ImportError
        with patch.dict('sys.modules', {'anthology': None}):
            with patch('builtins.__import__', side_effect=ImportError("No module named 'anthology'")):
                with pytest.raises(ImportError, match="acl-anthology package required"):
                    adapter.crawl(config)
    
    def test_crawl_raises_value_error_for_unknown_venue(self, adapter):
        """Test that crawl raises ValueError for unknown venue."""
        config = VenueConfig(name='UNKNOWN_VENUE', years=[2024])
        
        # Mock the anthology module so import succeeds
        mock_anthology_class = Mock(return_value=Mock())
        with patch.dict('sys.modules', {'anthology': Mock(Anthology=mock_anthology_class)}):
            with pytest.raises(ValueError, match="Unknown venue"):
                adapter.crawl(config)


class TestCreatePaperFromAnthology:
    """Test paper creation from anthology entries."""
    
    @pytest.fixture
    def adapter(self):
        """Create ACL adapter instance."""
        return ACLAdapter()
    
    def test_create_paper_from_anthology_full(self, adapter):
        """Test creating Paper from full anthology entry."""
        mock_paper = Mock()
        mock_paper.id = '2024.acl-long.1'
        mock_paper.title = 'Test Paper Title'
        mock_paper.abstract = 'Test abstract.'
        mock_paper.authors = [Mock(name='Author One'), Mock(name='Author Two')]
        mock_paper.doi = '10.1234/test'
        mock_paper.bibtex = '@article{test}'
        
        result = adapter._create_paper_from_anthology(mock_paper, 'ACL', 2024)
        
        assert result is not None
        assert result.id == '2024.acl-long.1'
        assert result.title == 'Test Paper Title'
        assert result.abstract == 'Test abstract.'
        assert result.year == 2024
        assert result.venue == 'ACL'
        assert result.source_platform == 'acl'
        assert result.download_available == 'acl'
    
    def test_create_paper_from_anthology_minimal(self, adapter):
        """Test creating Paper from minimal anthology entry."""
        mock_paper = Mock()
        mock_paper.id = '2024.acl-long.2'
        mock_paper.title = 'Minimal Paper'
        # Set abstract to empty string, authors to empty list
        mock_paper.abstract = ''
        mock_paper.authors = []
        
        # Handle missing optional attributes
        del mock_paper.doi
        del mock_paper.bibtex
        
        result = adapter._create_paper_from_anthology(mock_paper, 'ACL', 2024)
        
        assert result is not None
        assert result.id == '2024.acl-long.2'
        assert result.title == 'Minimal Paper'
    
    def test_create_paper_from_anthology_with_error(self, adapter):
        """Test that errors in paper creation are handled gracefully."""
        # Create a mock paper that will cause an error when accessing id
        mock_paper = Mock()
        mock_paper.id = Mock(side_effect=Exception("Test error"))
        
        result = adapter._create_paper_from_anthology(mock_paper, 'ACL', 2024)
        
        # Should return None on error
        assert result is None


class TestVenueConfig:
    """Test VenueConfig for ACL."""
    
    def test_venue_config_creation(self):
        """Test creating VenueConfig for ACL."""
        config = VenueConfig(
            name='ACL',
            years=[2024, 2023],
            platform='acl'
        )
        
        assert config.name == 'ACL'
        assert config.years == [2024, 2023]
        assert config.platform == 'acl'
    
    def test_venue_config_from_dict(self):
        """Test creating VenueConfig from dictionary."""
        config = VenueConfig.from_dict({
            'name': 'EMNLP',
            'years': [2024],
            'platform': 'acl',
            'accepted_only': True
        })
        
        assert config.name == 'EMNLP'
        assert config.years == [2024]
        assert config.accepted_only is True
    
    def test_venue_config_validation_missing_name(self):
        """Test that config validation fails without name."""
        with pytest.raises(ValueError, match="Venue name is required"):
            VenueConfig(name='', years=[2024])
    
    def test_venue_config_validation_missing_years(self):
        """Test that config validation fails without years."""
        with pytest.raises(ValueError, match="At least one year is required"):
            VenueConfig(name='ACL', years=[])


# Run tests if executed directly
if __name__ == '__main__':
    pytest.main([__file__, '-v'])