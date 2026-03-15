"""
Unit tests for the Nature Journals Adapter.

Tests cover:
- NatureAdapter initialization and properties
- API record parsing
- DOI/paper ID conversion
- Pagination handling
- Journal-specific adapters
"""

import sys
import os
from pathlib import Path

# Set up path before any other imports
# This test file is at: .../paper-agent/tests/unit/adapters/test_nature_adapter.py
# The lib directory is at: .../paper-agent/lib
_test_file_dir = Path(__file__).resolve()
_lib_path = _test_file_dir.parent.parent.parent.parent / 'lib'
if str(_lib_path) not in sys.path:
    sys.path.insert(0, str(_lib_path))

import pytest
from unittest.mock import Mock, patch, MagicMock
import requests

# Rely on pytest.ini pythonpath configuration
from adapters.base import VenueAdapter, VenueConfig
from adapters.registry import AdapterRegistry
from adapters.nature_adapter import (
    NatureAdapter,
    NatureMachineIntelligenceAdapter,
    NatureChemistryAdapter,
    NatureCommunicationsAdapter,
    NatureMainAdapter
)
from database import Paper


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def nature_adapter():
    """Basic Nature adapter instance."""
    return NatureAdapter('42256')


@pytest.fixture
def nature_communications_adapter():
    """Nature Communications adapter (fully open access)."""
    return NatureCommunicationsAdapter()


@pytest.fixture
def sample_api_record():
    """Sample Springer Nature API record."""
    return {
        'doi': '10.1038/s42256-024-00123-4',
        'title': 'A Novel Deep Learning Approach for Molecular Generation',
        'abstract': 'We present a novel deep learning method for generating molecular structures...',
        'creators': [
            {'creator': 'John Doe'},
            {'creator': 'Jane Smith'},
            {'creator': 'Alice Johnson'}
        ],
        'keyword': ['deep learning', 'molecular generation', 'drug discovery'],
        'openaccess': 'true',
        'url': [
            {'value': 'https://www.nature.com/articles/s42256-024-00123-4.pdf'}
        ],
        'journal': 'Nature Machine Intelligence',
        'publicationDate': '2024-03-15'
    }


@pytest.fixture
def sample_closed_access_record():
    """Sample closed access Springer Nature API record."""
    return {
        'doi': '10.1038/s41557-024-00123-4',
        'title': 'A Chemistry Paper',
        'abstract': 'This is a chemistry paper abstract...',
        'creators': [
            {'creator': 'Chemist One'},
            {'creator': 'Chemist Two'}
        ],
        'keyword': ['catalysis', 'organic chemistry'],
        'openaccess': 'false',
        'url': [],
        'journal': 'Nature Chemistry',
        'publicationDate': '2024-05-20'
    }


@pytest.fixture
def sample_api_response():
    """Sample Springer Nature API response."""
    return {
        'result': [{
            'total': '50',
            'recordsPerPage': '25',
            'start': '1'
        }],
        'records': [
            {
                'doi': '10.1038/s42256-024-00123-4',
                'title': 'Paper 1',
                'abstract': 'Abstract 1',
                'creators': [{'creator': 'Author 1'}],
                'openaccess': 'true',
                'url': [{'value': 'https://example.com/paper1.pdf'}]
            },
            {
                'doi': '10.1038/s42256-024-00124-1',
                'title': 'Paper 2',
                'abstract': 'Abstract 2',
                'creators': [{'creator': 'Author 2'}],
                'openaccess': 'false',
                'url': []
            }
        ]
    }


@pytest.fixture
def venue_config():
    """Sample venue configuration."""
    return VenueConfig(
        name='Nature Machine Intelligence',
        years=[2024],
        platform='nature',
        additional_params={'api_key': 'test-api-key'}
    )


# ============================================================================
# Test NatureAdapter Initialization
# ============================================================================

class TestNatureAdapterInit:
    """Test NatureAdapter initialization and properties."""

    def test_adapter_initialization(self, nature_adapter):
        """Test basic adapter initialization."""
        assert nature_adapter.journal_code == '42256'
        assert nature_adapter.venue_type == 'journal'

    def test_platform_name(self, nature_adapter):
        """Test platform name includes journal code."""
        assert nature_adapter.platform_name == 'nature_42256'

    def test_journal_name_known(self, nature_adapter):
        """Test journal name for known journal code."""
        assert nature_adapter.journal_name == 'Nature Machine Intelligence'

    def test_journal_name_unknown(self):
        """Test journal name for unknown journal code."""
        adapter = NatureAdapter('99999')
        assert 'Nature Journal' in adapter.journal_name
        assert '99999' in adapter.journal_name

    def test_is_open_access_false(self, nature_adapter):
        """Test open access status for non-OA journal."""
        assert nature_adapter.is_open_access is False

    def test_is_open_access_true(self, nature_communications_adapter):
        """Test open access status for fully OA journal."""
        assert nature_communications_adapter.is_open_access is True

    def test_rate_limit_delay(self, nature_adapter):
        """Test rate limit delay is appropriate."""
        assert nature_adapter.rate_limit_delay() == 1.0


# ============================================================================
# Test DOI/Paper ID Conversion
# ============================================================================

class TestDoiConversion:
    """Test DOI to paper ID conversion and vice versa."""

    def test_doi_to_paper_id(self, nature_adapter):
        """Test DOI to paper ID conversion."""
        doi = '10.1038/s42256-024-00123-4'
        paper_id = nature_adapter._doi_to_paper_id(doi)
        
        assert '/' not in paper_id
        assert paper_id == '10_1038_s42256-024-00123-4'

    def test_paper_id_to_doi(self, nature_adapter):
        """Test paper ID back to DOI conversion."""
        paper_id = '10_1038_s42256-024-00123-4'
        doi = nature_adapter._paper_id_to_doi(paper_id)
        
        assert doi.startswith('10.')
        assert '/' in doi

    def test_doi_roundtrip(self, nature_adapter):
        """Test DOI conversion roundtrip."""
        original_doi = '10.1038/s42256-024-00123-4'
        paper_id = nature_adapter._doi_to_paper_id(original_doi)
        recovered_doi = nature_adapter._paper_id_to_doi(paper_id)
        
        assert recovered_doi == original_doi


# ============================================================================
# Test Record Parsing
# ============================================================================

class TestRecordParsing:
    """Test API record parsing into Paper objects."""

    def test_parse_record_basic(self, nature_adapter, sample_api_record):
        """Test basic record parsing."""
        paper = nature_adapter._parse_record(sample_api_record, 2024)
        
        assert paper is not None
        assert paper.title == 'A Novel Deep Learning Approach for Molecular Generation'
        assert paper.year == 2024
        assert paper.venue_type == 'journal'

    def test_parse_record_doi(self, nature_adapter, sample_api_record):
        """Test DOI extraction and ID generation."""
        paper = nature_adapter._parse_record(sample_api_record, 2024)
        
        assert paper.doi == '10.1038/s42256-024-00123-4'
        assert paper.id == '10_1038_s42256-024-00123-4'

    def test_parse_record_authors(self, nature_adapter, sample_api_record):
        """Test author extraction."""
        paper = nature_adapter._parse_record(sample_api_record, 2024)
        
        assert len(paper.authors) == 3
        assert 'John Doe' in paper.authors
        assert 'Jane Smith' in paper.authors
        assert 'Alice Johnson' in paper.authors

    def test_parse_record_keywords(self, nature_adapter, sample_api_record):
        """Test keyword extraction."""
        paper = nature_adapter._parse_record(sample_api_record, 2024)
        
        assert len(paper.keywords) == 3
        assert 'deep learning' in paper.keywords

    def test_parse_record_open_access(self, nature_adapter, sample_api_record):
        """Test open access detection."""
        paper = nature_adapter._parse_record(sample_api_record, 2024)
        
        assert paper.download_available == 'nature'
        assert paper.pdf_url is not None

    def test_parse_record_closed_access(self, nature_adapter, sample_closed_access_record):
        """Test closed access handling."""
        paper = nature_adapter._parse_record(sample_closed_access_record, 2024)
        
        # Closed access should have PDF URL constructed but not flagged as available
        assert paper.doi is not None

    def test_parse_record_no_doi(self, nature_adapter):
        """Test record without DOI is skipped."""
        record = {
            'title': 'No DOI Paper',
            'abstract': 'Abstract'
        }
        
        paper = nature_adapter._parse_record(record, 2024)
        assert paper is None

    def test_parse_record_no_title(self, nature_adapter):
        """Test record without title is skipped."""
        record = {
            'doi': '10.1038/test',
            'abstract': 'Abstract'
        }
        
        paper = nature_adapter._parse_record(record, 2024)
        assert paper is None

    def test_parse_record_string_keywords(self, nature_adapter):
        """Test handling of comma-separated keyword string."""
        record = {
            'doi': '10.1038/test',
            'title': 'Test Paper',
            'keyword': 'keyword1, keyword2, keyword3'
        }
        
        paper = nature_adapter._parse_record(record, 2024)
        
        assert len(paper.keywords) == 3


# ============================================================================
# Test PDF URL Generation
# ============================================================================

class TestPdfUrl:
    """Test PDF URL generation."""

    def test_get_pdf_url(self, nature_adapter):
        """Test PDF URL generation."""
        paper_id = '10_1038_s42256-024-00123-4'
        pdf_url = nature_adapter.get_pdf_url(paper_id)
        
        assert 'nature.com' in pdf_url
        assert '.pdf' in pdf_url

    def test_get_pdf_url_with_doi_kwarg(self, nature_adapter):
        """Test PDF URL with explicit DOI."""
        paper_id = 'some-id'
        doi = '10.1038/s42256-024-00123-4'
        pdf_url = nature_adapter.get_pdf_url(paper_id, doi=doi)
        
        assert doi in pdf_url


# ============================================================================
# Test Crawling
# ============================================================================

class TestCrawling:
    """Test crawling functionality."""

    def test_crawl_requires_api_key(self, nature_adapter):
        """Test that crawl requires API key."""
        config = VenueConfig(
            name='Nature MI',
            years=[2024],
            platform='nature'
        )
        
        with pytest.raises(ValueError, match="API key required"):
            nature_adapter.crawl(config)

    def test_crawl_accepts_nature_api_key(self, nature_adapter):
        """Test that crawl accepts nature_api_key as well."""
        config = VenueConfig(
            name='Nature MI',
            years=[2024],
            platform='nature',
            additional_params={'nature_api_key': 'test-key'}
        )
        
        with patch.object(nature_adapter, '_crawl_year', return_value=[]):
            papers = nature_adapter.crawl(config)
            assert papers == []

    @patch('adapters.nature_adapter.requests.get')
    def test_crawl_year_pagination(self, mock_get, nature_adapter, sample_api_response):
        """Test pagination during year crawl."""
        # First response
        first_response = Mock()
        first_response.json.return_value = sample_api_response
        first_response.raise_for_status = Mock()
        
        # Second response (empty records to stop pagination)
        second_response = Mock()
        second_response.json.return_value = {'result': [{'total': '50'}], 'records': []}
        second_response.raise_for_status = Mock()
        
        mock_get.side_effect = [first_response, second_response]
        
        papers = nature_adapter._crawl_year(2024, 'test-key')
        
        assert len(papers) == 2
        assert mock_get.call_count == 2

    @patch('adapters.nature_adapter.requests.get')
    def test_crawl_year_api_error(self, mock_get, nature_adapter):
        """Test API error handling."""
        mock_get.side_effect = requests.RequestException("API Error")
        
        with pytest.raises(requests.RequestException):
            nature_adapter._crawl_year(2024, 'test-key')


# ============================================================================
# Test Journal-Specific Adapters
# ============================================================================

class TestJournalSpecificAdapters:
    """Test specific journal adapter classes."""

    def test_nature_machine_intelligence_adapter(self):
        """Test Nature Machine Intelligence adapter."""
        adapter = NatureMachineIntelligenceAdapter()
        
        assert adapter.journal_code == '42256'
        assert adapter.platform_name == 'nature_machine_intelligence'
        assert adapter.journal_name == 'Nature Machine Intelligence'
        assert adapter.is_open_access is False

    def test_nature_chemistry_adapter(self):
        """Test Nature Chemistry adapter."""
        adapter = NatureChemistryAdapter()
        
        assert adapter.journal_code == '41557'
        assert adapter.platform_name == 'nature_chemistry'
        assert adapter.journal_name == 'Nature Chemistry'
        assert adapter.is_open_access is False

    def test_nature_communications_adapter(self):
        """Test Nature Communications adapter."""
        adapter = NatureCommunicationsAdapter()
        
        assert adapter.journal_code == '41467'
        assert adapter.platform_name == 'nature_communications'
        assert adapter.journal_name == 'Nature Communications'
        assert adapter.is_open_access is True  # Fully OA

    def test_nature_main_adapter(self):
        """Test Nature main journal adapter."""
        adapter = NatureMainAdapter()
        
        assert adapter.journal_code == '41586'
        assert adapter.platform_name == 'nature_main'
        assert adapter.journal_name == 'Nature'
        assert adapter.is_open_access is False


# ============================================================================
# Test Availability and Validation
# ============================================================================

class TestAvailabilityAndValidation:
    """Test availability checks and config validation."""

    @patch('adapters.nature_adapter.requests.get')
    def test_check_availability_success(self, mock_get, nature_adapter):
        """Test availability check succeeds."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        assert nature_adapter.check_availability() is True

    @patch('adapters.nature_adapter.requests.get')
    def test_check_availability_forbidden(self, mock_get, nature_adapter):
        """Test availability check with 403."""
        mock_response = Mock()
        mock_response.status_code = 403
        mock_get.return_value = mock_response
        
        # 403 still means API is up
        assert nature_adapter.check_availability() is True

    @patch('adapters.nature_adapter.requests.get')
    def test_check_availability_error(self, mock_get, nature_adapter):
        """Test availability check with connection error."""
        mock_get.side_effect = requests.RequestException()
        
        assert nature_adapter.check_availability() is False

    def test_validate_config_no_api_key(self, nature_adapter, venue_config):
        """Test validation fails without API key."""
        config = VenueConfig(
            name='Nature MI',
            years=[2024],
            platform='nature'
        )
        
        assert nature_adapter.validate_config(config) is False

    def test_validate_config_success(self, nature_adapter, venue_config):
        """Test successful config validation."""
        assert nature_adapter.validate_config(venue_config) is True

    def test_get_supported_venues(self, nature_adapter):
        """Test get supported venues returns journal name."""
        venues = nature_adapter.get_supported_venues()
        
        assert 'Nature Machine Intelligence' in venues


# ============================================================================
# Test Integration with Registry
# ============================================================================

class TestRegistryIntegration:
    """Test that adapters are properly registered."""

    def test_nature_mi_registered(self):
        """Test Nature Machine Intelligence is registered."""
        from adapters.registry import AdapterRegistry
        
        # Clear and re-register
        AdapterRegistry.clear()
        from adapters import register_builtin_adapters
        register_builtin_adapters()
        
        # Try to import and register nature adapter
        try:
            from adapters import nature_adapter  # noqa: F401
        except ImportError:
            pass
        
        adapter = AdapterRegistry.get('nature_machine_intelligence')
        assert adapter is not None
        assert isinstance(adapter, NatureMachineIntelligenceAdapter)

    def test_nature_communications_registered(self):
        """Test Nature Communications is registered."""
        from adapters.registry import AdapterRegistry
        
        AdapterRegistry.clear()
        from adapters import register_builtin_adapters
        register_builtin_adapters()
        
        try:
            from adapters import nature_adapter  # noqa: F401
        except ImportError:
            pass
        
        adapter = AdapterRegistry.get('nature_communications')
        assert adapter is not None
        assert isinstance(adapter, NatureCommunicationsAdapter)