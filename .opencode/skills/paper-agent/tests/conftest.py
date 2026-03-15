"""
Shared fixtures for paper-agent database tests.
"""

import pytest
import json
import tempfile
import os
from pathlib import Path
from datetime import datetime

# Add lib to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'lib'))

from database import Paper, DatabaseManager


# ============================================================================
# Sample Paper Fixtures
# ============================================================================

@pytest.fixture
def sample_paper_dict():
    """Sample paper dictionary with all fields."""
    return {
        'id': 'test-paper-001',
        'title': 'A Novel Approach to Deep Learning',
        'abstract': 'This paper presents a novel approach to deep learning.',
        'authors': ['John Doe', 'Jane Smith'],
        'keywords': ['deep learning', 'neural networks'],
        'year': 2024,
        'venue': 'ICLR',
        'venue_type': 'conference',
        'source_platform': 'openreview',
        'crawl_date': '2024-01-15T10:30:00',
        'last_updated': '2024-01-15T10:30:00',
        'download_status': 'pending',
        'analysis_status': 'pending',
        'pdf_url': 'https://openreview.net/pdf?id=test',
        'doi': '10.1234/test.001',
        'bibtex': '@article{test, title="Test"}',
        'citation_count': 10,
        'arxiv_id': '2401.12345',
        'download_available': 'openreview',
        'analysis_file': None,
        'relevance_score': 0.85
    }


@pytest.fixture
def sample_paper(sample_paper_dict):
    """Sample Paper instance with all fields."""
    return Paper(**sample_paper_dict)


@pytest.fixture
def minimal_paper_dict():
    """Minimal paper dictionary with only required fields."""
    return {
        'id': 'minimal-paper-001',
        'title': 'Minimal Paper'
    }


@pytest.fixture
def minimal_paper(minimal_paper_dict):
    """Minimal Paper instance."""
    return Paper(**minimal_paper_dict)


@pytest.fixture
def legacy_paper_dict():
    """Legacy format paper dictionary (from old crawler)."""
    return {
        'id': 'legacy-paper-001',
        'title': 'Legacy Paper Title',
        'abstract': 'Abstract from legacy crawler',
        'authors': ['Author One', 'Author Two'],
        'keywords': ['keyword1', 'keyword2'],
        'venue_id': 'ICLR.cc/2024/Conference',
        'decision': 'accepted',
        'pdf_url': 'https://openreview.net/pdf?id=legacy'
    }


@pytest.fixture
def sample_papers_list():
    """List of sample papers for batch testing."""
    papers = []
    for i in range(5):
        papers.append({
            'id': f'paper-{i:03d}',
            'title': f'Sample Paper {i}',
            'abstract': f'Abstract for paper {i}',
            'authors': [f'Author {i}A', f'Author {i}B'],
            'keywords': ['ml', 'ai'],
            'year': 2024,
            'venue': 'ICLR' if i % 2 == 0 else 'ICML',
            'source_platform': 'openreview',
            'download_status': 'pending' if i % 2 == 0 else 'success',
            'analysis_status': 'pending',
            'relevance_score': 0.5 + i * 0.1
        })
    return papers


@pytest.fixture
def sample_papers(sample_papers_list):
    """List of Paper instances."""
    return [Paper.from_dict(d) for d in sample_papers_list]


# ============================================================================
# Database Fixtures
# ============================================================================

@pytest.fixture
def empty_db(tmp_path):
    """Empty database instance using JSON format."""
    db_path = tmp_path / 'test_db.json'
    return DatabaseManager(db_path, format='json')


@pytest.fixture
def populated_db(empty_db, sample_papers):
    """Database with sample papers pre-loaded."""
    empty_db.add_papers(sample_papers)
    return empty_db


@pytest.fixture
def csv_db(tmp_path):
    """Empty database instance using CSV format."""
    db_path = tmp_path / 'test_db.csv'
    return DatabaseManager(db_path, format='csv')


# ============================================================================
# Legacy Format Fixtures
# ============================================================================

@pytest.fixture
def legacy_json_file(tmp_path, legacy_paper_dict):
    """Create a legacy format JSON file."""
    legacy_file = tmp_path / 'legacy_papers.json'
    legacy_data = {
        'crawl_time': '2024-01-15T10:30:00',
        'venue': 'ICLR',
        'year': 2024,
        'papers': [legacy_paper_dict]
    }
    with open(legacy_file, 'w', encoding='utf-8') as f:
        json.dump(legacy_data, f)
    return legacy_file


@pytest.fixture
def legacy_json_list_file(tmp_path):
    """Create a legacy format JSON file with list of papers."""
    legacy_file = tmp_path / 'legacy_papers_list.json'
    papers_list = [
        {
            'id': f'legacy-{i}',
            'title': f'Legacy Paper {i}',
            'abstract': f'Abstract {i}',
            'authors': [f'Author {i}'],
            'venue_id': 'ICLR.cc/2024/Conference',
            'pdf_url': f'https://openreview.net/pdf?id={i}'
        }
        for i in range(3)
    ]
    with open(legacy_file, 'w', encoding='utf-8') as f:
        json.dump(papers_list, f)
    return legacy_file


# ============================================================================
# Helper Functions
# ============================================================================

def create_temp_json_file(tmp_path, data, filename='temp.json'):
    """Helper to create a temporary JSON file."""
    file_path = tmp_path / filename
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f)
    return file_path