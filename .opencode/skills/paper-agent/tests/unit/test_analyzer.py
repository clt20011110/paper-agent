#!/usr/bin/env python3
"""
Unit tests for analyzer module enhancements.

Tests cover:
- EnhancedAnalyzer class
- ResearchSummaryGenerator class
- Abstract-only analysis
- Summary generation
"""

import pytest
import json
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
import sys
import tempfile
import os

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'lib'))

from analyzer import (
    EnhancedAnalyzer,
    ResearchSummaryGenerator,
    generate_research_summary,
    ABSTRACT_ANALYSIS_PROMPT
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_paper():
    """Sample paper dictionary for testing."""
    return {
        'id': 'test-paper-001',
        'title': 'Deep Learning for Molecular Property Prediction',
        'abstract': 'This paper presents a novel deep learning approach for predicting molecular properties using graph neural networks.',
        'authors': ['John Doe', 'Jane Smith'],
        'keywords': ['deep learning', 'molecular'],
        'year': 2024,
        'venue': 'ICLR'
    }


@pytest.fixture
def sample_paper_no_abstract():
    """Sample paper without abstract."""
    return {
        'id': 'test-paper-002',
        'title': 'Test Paper Without Abstract',
        'abstract': '',
        'authors': ['Alice Brown'],
        'year': 2024,
        'venue': 'NeurIPS'
    }


@pytest.fixture
def sample_analyses():
    """Sample analysis results for summary generation."""
    return [
        {
            'title': 'Paper 1: Deep Learning for Molecules',
            'analysis': 'This paper proposes a GNN approach for molecular property prediction. Key contribution: novel attention mechanism.',
            'confidence': 'full_pdf'
        },
        {
            'title': 'Paper 2: Diffusion Models',
            'analysis': 'This paper applies diffusion models to molecular generation. Key contribution: conditional generation framework.',
            'confidence': 'abstract_only'
        }
    ]


@pytest.fixture
def mock_api_response():
    """Mock successful API response."""
    return {
        'choices': [
            {
                'message': {
                    'content': 'This is a mock analysis result.'
                }
            }
        ]
    }


# ============================================================================
# Test ABSTRACT_ANALYSIS_PROMPT
# ============================================================================

class TestAbstractAnalysisPrompt:
    """Tests for the abstract analysis prompt template."""
    
    def test_prompt_contains_required_fields(self):
        """Test that prompt template contains all required placeholders."""
        assert '{title}' in ABSTRACT_ANALYSIS_PROMPT
        assert '{authors}' in ABSTRACT_ANALYSIS_PROMPT
        assert '{venue}' in ABSTRACT_ANALYSIS_PROMPT
        assert '{year}' in ABSTRACT_ANALYSIS_PROMPT
        assert '{abstract}' in ABSTRACT_ANALYSIS_PROMPT
    
    def test_prompt_can_be_formatted(self, sample_paper):
        """Test that prompt can be formatted with paper data."""
        formatted = ABSTRACT_ANALYSIS_PROMPT.format(
            title=sample_paper['title'],
            authors=', '.join(sample_paper['authors']),
            venue=sample_paper['venue'],
            year=sample_paper['year'],
            abstract=sample_paper['abstract']
        )
        
        assert sample_paper['title'] in formatted
        assert 'John Doe, Jane Smith' in formatted
        assert 'ICLR' in formatted
        assert '2024' in formatted
        assert sample_paper['abstract'] in formatted


# ============================================================================
# Test EnhancedAnalyzer
# ============================================================================

class TestEnhancedAnalyzer:
    """Tests for EnhancedAnalyzer class."""
    
    def test_init(self):
        """Test EnhancedAnalyzer initialization."""
        analyzer = EnhancedAnalyzer()
        assert analyzer.logger is not None
    
    @patch('analyzer.analyze_pdf')
    def test_analyze_paper_with_pdf(self, mock_analyze_pdf, sample_paper):
        """Test analyzing paper with PDF available."""
        mock_analyze_pdf.return_value = 'Mock PDF analysis result'
        
        analyzer = EnhancedAnalyzer()
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            pdf_path = Path(f.name)
        
        try:
            result = analyzer.analyze_paper(
                paper=sample_paper,
                pdf_path=pdf_path,
                api_key='test-api-key',
                model='test-model'
            )
            
            assert result['confidence'] == 'full_pdf'
            assert result['method'] == 'full_pdf'
            assert result['analysis'] == 'Mock PDF analysis result'
            mock_analyze_pdf.assert_called_once()
        finally:
            os.unlink(pdf_path)
    
    @patch('analyzer.requests.post')
    def test_analyze_paper_without_pdf(self, mock_post, sample_paper):
        """Test analyzing paper without PDF (abstract-only)."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'choices': [{'message': {'content': 'Abstract analysis result'}}]
        }
        mock_post.return_value = mock_response
        
        analyzer = EnhancedAnalyzer()
        result = analyzer.analyze_paper(
            paper=sample_paper,
            pdf_path=None,
            api_key='test-api-key',
            model='test-model'
        )
        
        assert result['confidence'] == 'abstract_only'
        assert result['method'] == 'abstract_only'
        assert result['title'] == sample_paper['title']
    
    @patch('analyzer.requests.post')
    def test_analyze_paper_nonexistent_pdf(self, mock_post, sample_paper):
        """Test that non-existent PDF falls back to abstract analysis."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'choices': [{'message': {'content': 'Fallback abstract analysis'}}]
        }
        mock_post.return_value = mock_response
        
        analyzer = EnhancedAnalyzer()
        result = analyzer.analyze_paper(
            paper=sample_paper,
            pdf_path=Path('/nonexistent/path/file.pdf'),
            api_key='test-api-key',
            model='test-model'
        )
        
        assert result['confidence'] == 'abstract_only'
    
    @patch('analyzer.requests.post')
    def test_analyze_abstract_empty_paper(self, mock_post, sample_paper_no_abstract):
        """Test analyzing paper with empty abstract."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'choices': [{'message': {'content': 'Analysis of empty abstract'}}]
        }
        mock_post.return_value = mock_response
        
        analyzer = EnhancedAnalyzer()
        result = analyzer._analyze_abstract(
            paper=sample_paper_no_abstract,
            api_key='test-api-key',
            model='test-model'
        )
        
        assert result is not None
        assert result['method'] == 'abstract_only'
    
    def test_call_api_no_key(self):
        """Test that API call fails gracefully without API key."""
        analyzer = EnhancedAnalyzer()
        result = analyzer._call_api('test prompt', None, 'test-model')
        assert result is None
    
    @patch('analyzer.requests.post')
    def test_call_api_retry_on_failure(self, mock_post):
        """Test that API retries on failure."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response
        
        analyzer = EnhancedAnalyzer()
        
        with patch('analyzer.time.sleep'):
            result = analyzer._call_api(
                prompt='test prompt',
                api_key='test-key',
                model='test-model',
                max_retries=2
            )
        
        assert result is None
        assert mock_post.call_count == 2


# ============================================================================
# Test ResearchSummaryGenerator
# ============================================================================

class TestResearchSummaryGenerator:
    """Tests for ResearchSummaryGenerator class."""
    
    def test_init(self):
        """Test ResearchSummaryGenerator initialization."""
        generator = ResearchSummaryGenerator()
        assert generator.logger is not None
    
    @patch('analyzer.requests.post')
    def test_generate_summary(self, mock_post, sample_analyses):
        """Test generating summary from analyses."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'choices': [{'message': {'content': '# Research Summary\n\nThis is a mock summary.'}}]
        }
        mock_post.return_value = mock_response
        
        generator = ResearchSummaryGenerator()
        summary = generator.generate_summary(
            analyses=sample_analyses,
            topic='Machine Learning for Drug Discovery',
            api_key='test-api-key',
            model='anthropic/claude-3.5-sonnet'
        )
        
        assert '# Research Summary' in summary
        mock_post.assert_called_once()
    
    @patch('analyzer.requests.post')
    def test_generate_summary_truncates_large_input(self, mock_post):
        """Test that summary generation truncates large input."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'choices': [{'message': {'content': 'Summary'}}]
        }
        mock_post.return_value = mock_response
        
        generator = ResearchSummaryGenerator()
        
        large_analyses = [
            {'title': f'Paper {i}', 'analysis': 'x' * 1000}
            for i in range(60)
        ]
        
        summary = generator.generate_summary(
            analyses=large_analyses,
            topic='Test Topic',
            api_key='test-api-key'
        )
        
        assert summary is not None
    
    @patch('analyzer.requests.post')
    def test_generate_summary_handles_empty_analyses(self, mock_post):
        """Test summary generation with empty analyses list."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'choices': [{'message': {'content': 'No papers analyzed.'}}]
        }
        mock_post.return_value = mock_response
        
        generator = ResearchSummaryGenerator()
        summary = generator.generate_summary(
            analyses=[],
            topic='Test Topic',
            api_key='test-api-key'
        )
        
        assert summary is not None
    
    @patch('analyzer.requests.post')
    def test_generate_summary_api_failure(self, mock_post):
        """Test summary generation with API failure."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response
        
        generator = ResearchSummaryGenerator()
        
        with patch('analyzer.time.sleep'):
            summary = generator.generate_summary(
                analyses=[{'title': 'Test', 'analysis': 'Test'}],
                topic='Test',
                api_key='test-key'
            )
        
        assert 'Error' in summary


# ============================================================================
# Test generate_research_summary function
# ============================================================================

class TestGenerateResearchSummary:
    """Tests for the generate_research_summary function."""
    
    @patch('analyzer.requests.post')
    def test_generate_research_summary_success(self, mock_post):
        """Test successful research summary generation."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'choices': [{'message': {'content': '# Research Summary\n\nContent here.'}}]
        }
        mock_post.return_value = mock_response
        
        with tempfile.TemporaryDirectory() as tmpdir:
            analysis_dir = Path(tmpdir) / 'analysis'
            analysis_dir.mkdir()
            
            (analysis_dir / 'paper1_analysis.md').write_text('Analysis of paper 1')
            (analysis_dir / 'paper2_analysis.md').write_text('Analysis of paper 2')
            
            output_file = Path(tmpdir) / 'summary.md'
            
            result = generate_research_summary(
                analysis_dir=analysis_dir,
                output_file=output_file,
                topic='Test Topic',
                api_key='test-api-key'
            )
            
            assert result is True
            assert output_file.exists()
            content = output_file.read_text()
            assert '# Research Summary' in content
    
    def test_generate_research_summary_no_directory(self):
        """Test with non-existent directory."""
        result = generate_research_summary(
            analysis_dir=Path('/nonexistent/dir'),
            output_file=Path('/tmp/output.md'),
            topic='Test',
            api_key='test-key'
        )
        
        assert result is False
    
    def test_generate_research_summary_no_files(self):
        """Test with empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analysis_dir = Path(tmpdir) / 'empty'
            analysis_dir.mkdir()
            
            result = generate_research_summary(
                analysis_dir=analysis_dir,
                output_file=Path(tmpdir) / 'output.md',
                topic='Test',
                api_key='test-key'
            )
            
            assert result is False


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    @patch('analyzer.requests.post')
    def test_analyze_abstract_with_special_characters(self, mock_post):
        """Test analyzing abstract with special characters."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'choices': [{'message': {'content': 'Result'}}]
        }
        mock_post.return_value = mock_response
        
        paper = {
            'id': 'special',
            'title': 'Paper with "Quotes" and {Braces}',
            'abstract': 'Abstract with\nnewlines\tand\ttabs',
            'authors': ['Author'],
            'year': 2024,
            'venue': 'ICLR'
        }
        
        analyzer = EnhancedAnalyzer()
        result = analyzer._analyze_abstract(paper, 'test-key', 'test-model')
        
        assert result is not None
    
    @patch('analyzer.requests.post')
    def test_summary_with_unicode_content(self, mock_post):
        """Test summary generation with Unicode content."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'choices': [{'message': {'content': '# Summary 中文 日本語'}}]
        }
        mock_post.return_value = mock_response
        
        generator = ResearchSummaryGenerator()
        
        analyses = [
            {'title': '论文 标题', 'analysis': '分析内容'}
        ]
        
        summary = generator.generate_summary(
            analyses=analyses,
            topic='研究主题',
            api_key='test-key'
        )
        
        assert '中文' in summary


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])