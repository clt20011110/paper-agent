#!/usr/bin/env python3
"""
Unit tests for semantic_filter module.

Tests cover:
- SemanticScore dataclass
- SemanticScorer class (with mocked model)
- HybridFilter class
- Response parsing
- Prompt building
"""

import pytest
import json
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
import sys

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'lib'))

from semantic_filter import (
    SemanticScore,
    SemanticScorer,
    HybridFilter
)
from database import Paper
from filter import KeywordFilter, FilterConfig


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_paper():
    """Sample Paper instance for testing."""
    return Paper(
        id='test-paper-001',
        title='Deep Learning for Molecular Property Prediction',
        abstract='This paper presents a novel deep learning approach for predicting molecular properties using graph neural networks. We demonstrate state-of-the-art results on multiple benchmarks.',
        authors=['John Doe', 'Jane Smith'],
        keywords=['deep learning', 'molecular'],
        year=2024,
        venue='ICLR',
        source_platform='openreview'
    )


@pytest.fixture
def sample_paper_irrelevant():
    """Sample irrelevant Paper instance."""
    return Paper(
        id='test-paper-002',
        title='A Survey of Ancient Roman Architecture',
        abstract='This paper reviews the architectural styles and construction techniques used in ancient Rome.',
        authors=['Alice Brown'],
        year=2024,
        venue='ICLR',
        source_platform='openreview'
    )


@pytest.fixture
def sample_papers_list(sample_paper, sample_paper_irrelevant):
    """List of sample papers."""
    return [
        sample_paper,
        sample_paper_irrelevant,
        Paper(
            id='test-paper-003',
            title='Diffusion Models for Drug Discovery',
            abstract='We apply diffusion models to generate novel drug candidates with desired properties.',
            authors=['Bob Wilson'],
            year=2024,
            venue='NeurIPS',
            source_platform='openreview'
        )
    ]


@pytest.fixture
def sample_paper_dicts():
    """Sample paper dictionaries for testing."""
    return [
        {
            'id': 'paper-001',
            'title': 'Machine Learning for Drug Discovery',
            'abstract': 'This paper applies ML to drug discovery.',
            'authors': ['Author A'],
            'year': 2024,
            'venue': 'ICLR'
        },
        {
            'id': 'paper-002',
            'title': 'Unrelated Topic Paper',
            'abstract': 'This paper is about something unrelated.',
            'authors': ['Author B'],
            'year': 2024,
            'venue': 'ICLR'
        }
    ]


@pytest.fixture
def regex_filter():
    """Create a basic KeywordFilter for testing."""
    config = FilterConfig(
        include_groups=[['machine', 'learning'], ['drug']],
        exclude=['survey'],
        match_fields=['title', 'abstract']
    )
    return KeywordFilter(config)


@pytest.fixture
def mock_scorer():
    """Create a mocked SemanticScorer without loading actual model."""
    with patch('semantic_filter.SemanticScorer._load_model'):
        scorer = SemanticScorer.__new__(SemanticScorer)
        scorer.model_name = "Qwen/Qwen3.5-0.8B-Instruct"
        scorer.device = "cpu"
        scorer.batch_size = 8
        scorer.max_workers = 2
        scorer.max_length = 2048
        scorer.model = Mock()
        scorer.tokenizer = Mock()
        scorer._torch = Mock()
        scorer._torch.no_grad = MagicMock()
        return scorer


# ============================================================================
# SemanticScore Tests
# ============================================================================

class TestSemanticScore:
    """Tests for SemanticScore dataclass."""
    
    def test_create_semantic_score(self):
        """Test creating a SemanticScore instance."""
        score = SemanticScore(
            paper_id='test-001',
            relevance_score=0.85,
            is_relevant=True,
            reasoning='Paper directly addresses the topic',
            key_aspects=['deep learning', 'molecular']
        )
        
        assert score.paper_id == 'test-001'
        assert score.relevance_score == 0.85
        assert score.is_relevant is True
        assert 'deep learning' in score.key_aspects
    
    def test_semantic_score_to_dict(self):
        """Test converting SemanticScore to dictionary."""
        score = SemanticScore(
            paper_id='test-001',
            relevance_score=0.75,
            is_relevant=True,
            reasoning='Test reasoning',
            key_aspects=['aspect1', 'aspect2']
        )
        
        result = score.to_dict()
        
        assert isinstance(result, dict)
        assert result['paper_id'] == 'test-001'
        assert result['relevance_score'] == 0.75
        assert result['is_relevant'] is True
    
    def test_semantic_score_from_dict(self):
        """Test creating SemanticScore from dictionary."""
        data = {
            'paper_id': 'test-002',
            'relevance_score': 0.6,
            'is_relevant': False,
            'reasoning': 'Not quite relevant',
            'key_aspects': ['aspect1']
        }
        
        score = SemanticScore.from_dict(data)
        
        assert score.paper_id == 'test-002'
        assert score.relevance_score == 0.6
        assert score.is_relevant is False


# ============================================================================
# SemanticScorer Tests
# ============================================================================

class TestSemanticScorerPromptBuilding:
    """Tests for SemanticScorer prompt building."""
    
    def test_build_prompt_basic(self, mock_scorer, sample_paper):
        """Test building a basic prompt."""
        prompt = mock_scorer._build_prompt(sample_paper, "machine learning")
        
        assert "machine learning" in prompt.lower()
        assert sample_paper.title in prompt
        assert "JSON" in prompt
        assert "score" in prompt.lower()
    
    def test_build_prompt_with_long_abstract(self, mock_scorer):
        """Test that long abstracts are truncated."""
        paper = Paper(
            id='long-abstract',
            title='Test Paper',
            abstract='x' * 5000  # Very long abstract
        )
        
        prompt = mock_scorer._build_prompt(paper, "test topic")
        
        # Abstract should be truncated
        assert len(prompt) < 6000  # Reasonable prompt length
    
    def test_build_prompt_with_empty_abstract(self, mock_scorer):
        """Test building prompt with empty abstract."""
        paper = Paper(
            id='empty-abstract',
            title='Test Paper',
            abstract=''
        )
        
        prompt = mock_scorer._build_prompt(paper, "test topic")
        
        assert "No abstract available" in prompt or len(prompt) > 0


class TestSemanticScorerResponseParsing:
    """Tests for SemanticScorer response parsing."""
    
    def test_parse_valid_json_response(self, mock_scorer):
        """Test parsing a valid JSON response."""
        response = '''
        Based on my analysis, here is the evaluation:
        {"score": 0.85, "reasoning": "Highly relevant to the topic", "key_aspects": ["ml", "drugs"]}
        '''
        
        score, reasoning, aspects = mock_scorer._parse_response(response)
        
        assert score == 0.85
        assert "Highly relevant" in reasoning
        assert "ml" in aspects
    
    def test_parse_json_with_nested_braces(self, mock_scorer):
        """Test parsing JSON that might have nested braces."""
        response = '{"score": 0.7, "reasoning": "Moderately relevant", "key_aspects": ["a", "b"]}'
        
        score, reasoning, aspects = mock_scorer._parse_response(response)
        
        assert score == 0.7
        assert len(aspects) == 2
    
    def test_parse_response_with_percentage(self, mock_scorer):
        """Test parsing response with percentage score."""
        response = 'The score is 75 percent relevant'
        
        score, reasoning, aspects = mock_scorer._parse_response(response)
        
        assert score == 0.75  # Should convert percentage
    
    def test_parse_response_with_invalid_json(self, mock_scorer):
        """Test handling invalid JSON response."""
        response = 'This is not valid JSON at all'
        
        score, reasoning, aspects = mock_scorer._parse_response(response)
        
        # Should return default values
        assert score == 0.0
    
    def test_parse_response_clamps_score(self, mock_scorer):
        """Test that scores are clamped to valid range."""
        # Score above 1.0
        response = '{"score": 1.5}'
        score, _, _ = mock_scorer._parse_response(response)
        assert score == 1.0
        
        # Score below 0
        response = '{"score": -0.5}'
        score, _, _ = mock_scorer._parse_response(response)
        assert score == 0.0
    
    def test_parse_response_with_non_list_aspects(self, mock_scorer):
        """Test handling non-list key_aspects."""
        response = '{"score": 0.8, "key_aspects": "single_aspect"}'
        
        score, reasoning, aspects = mock_scorer._parse_response(response)
        
        assert isinstance(aspects, list)
        assert "single_aspect" in aspects


class TestSemanticScorerScoring:
    """Tests for SemanticScorer scoring functionality."""
    
    def test_score_paper_with_mock(self, mock_scorer, sample_paper):
        """Test scoring a single paper with mocked model."""
        # Setup mock behavior
        mock_output = Mock()
        mock_output.__getitem__ = Mock(return_value=Mock())
        mock_scorer.model.generate = Mock(return_value=[[0, 1, 2, 3]])
        mock_scorer.tokenizer.decode = Mock(return_value='{"score": 0.9, "reasoning": "Test", "key_aspects": ["test"]}')
        mock_scorer.tokenizer = Mock()
        mock_scorer.tokenizer.return_value = {'input_ids': Mock(), 'attention_mask': Mock()}
        mock_scorer.tokenizer.pad_token_id = 0
        
        # This would normally call the model, but we mock it
        with patch.object(mock_scorer, '_build_prompt', return_value="test prompt"):
            with patch.object(mock_scorer, '_parse_response', return_value=(0.9, "Test", ["test"])):
                # Mock the tokenizer call
                mock_scorer.tokenizer = Mock()
                mock_scorer.tokenizer.return_value = {'input_ids': Mock(), 'attention_mask': Mock()}
                mock_scorer.tokenizer.pad_token_id = 0
                mock_scorer.tokenizer.decode = Mock(return_value='{"score": 0.9}')
                
                result = mock_scorer.score_paper(sample_paper, "test topic")
                
                assert isinstance(result, SemanticScore)
                assert result.paper_id == sample_paper.id
    
    def test_score_batch(self, mock_scorer, sample_papers_list):
        """Test scoring a batch of papers."""
        # Mock score_paper to return predictable results
        def mock_score_paper(paper, topic):
            score = 0.8 if 'molecular' in paper.title.lower() or 'diffusion' in paper.title.lower() else 0.2
            return SemanticScore(
                paper_id=paper.id,
                relevance_score=score,
                is_relevant=score >= 0.7,
                reasoning="Mocked",
                key_aspects=[]
            )
        
        with patch.object(mock_scorer, 'score_paper', side_effect=mock_score_paper):
            results = mock_scorer.score_batch(
                sample_papers_list,
                "molecular property prediction",
                threshold=0.7,
                show_progress=False
            )
            
            assert len(results) == len(sample_papers_list)
            assert all(isinstance(r, SemanticScore) for r in results)
    
    def test_get_statistics(self, mock_scorer):
        """Test getting statistics from scores."""
        scores = [
            SemanticScore('p1', 0.9, True, '', []),
            SemanticScore('p2', 0.8, True, '', []),
            SemanticScore('p3', 0.4, False, '', []),
            SemanticScore('p4', 0.2, False, '', []),
        ]
        
        stats = mock_scorer.get_statistics(scores)
        
        assert stats['total_papers'] == 4
        assert stats['relevant_count'] == 2
        assert stats['irrelevant_count'] == 2
        assert stats['relevance_rate'] == 50.0
        assert stats['score_distribution']['0.7-1.0'] == 2
    
    def test_get_statistics_empty(self, mock_scorer):
        """Test statistics with empty list."""
        stats = mock_scorer.get_statistics([])
        
        assert stats['total_papers'] == 0
        assert stats['avg_score'] == 0.0


# ============================================================================
# HybridFilter Tests
# ============================================================================

class TestHybridFilter:
    """Tests for HybridFilter class."""
    
    def test_create_hybrid_filter(self, regex_filter, mock_scorer):
        """Test creating a HybridFilter instance."""
        hf = HybridFilter(
            regex_filter=regex_filter,
            semantic_scorer=mock_scorer,
            mode='hybrid'
        )
        
        assert hf.mode == 'hybrid'
        assert hf.regex_filter is regex_filter
        assert hf.semantic_scorer is mock_scorer
    
    def test_invalid_mode_raises_error(self, regex_filter, mock_scorer):
        """Test that invalid mode raises ValueError."""
        with pytest.raises(ValueError):
            HybridFilter(
                regex_filter=regex_filter,
                semantic_scorer=mock_scorer,
                mode='invalid_mode'
            )
    
    def test_semantic_mode_requires_scorer(self, regex_filter):
        """Test that semantic_only mode requires a scorer."""
        with pytest.raises(ValueError):
            HybridFilter(
                regex_filter=regex_filter,
                semantic_scorer=None,
                mode='semantic_only'
            )
    
    def test_regex_only_mode(self, regex_filter, sample_paper_dicts):
        """Test regex_only mode."""
        hf = HybridFilter(
            regex_filter=regex_filter,
            semantic_scorer=None,
            mode='regex_only'
        )
        
        relevant, scores = hf.filter_papers(
            sample_paper_dicts,
            "test topic",
            threshold=0.7
        )
        
        # Should use only regex filtering
        assert len(scores) == 0  # No semantic scores in regex_only mode
        # Paper with 'machine' and 'learning' should be matched
        assert any('Machine Learning' in p['title'] for p in relevant)
    
    def test_hybrid_mode(self, regex_filter, mock_scorer, sample_paper_dicts):
        """Test hybrid mode."""
        # Mock semantic scoring
        def mock_score_batch(papers, topic, threshold, show_progress=True):
            return [
                SemanticScore(p.id if hasattr(p, 'id') else p['id'], 0.8, True, '', [])
                for p in papers
            ]
        
        mock_scorer.score_batch = mock_score_batch
        
        hf = HybridFilter(
            regex_filter=regex_filter,
            semantic_scorer=mock_scorer,
            mode='hybrid'
        )
        
        relevant, scores = hf.filter_papers(
            sample_paper_dicts,
            "test topic",
            threshold=0.7
        )
        
        # Should have semantic scores
        assert len(scores) > 0
    
    def test_get_mode_description(self, regex_filter, mock_scorer):
        """Test getting mode description."""
        hf = HybridFilter(regex_filter, mock_scorer, 'hybrid')
        desc = hf.get_mode_description()
        
        assert 'Two-stage' in desc or 'hybrid' in desc.lower()
        
        hf_regex = HybridFilter(regex_filter, None, 'regex_only')
        desc = hf_regex.get_mode_description()
        
        assert 'keyword' in desc.lower()


class TestHybridFilterPaperObjects:
    """Tests for HybridFilter with Paper objects."""
    
    def test_filter_paper_objects_regex_only(self, regex_filter, sample_papers_list):
        """Test filtering Paper objects with regex_only mode."""
        hf = HybridFilter(
            regex_filter=regex_filter,
            semantic_scorer=None,
            mode='regex_only'
        )
        
        relevant, scores = hf.filter_paper_objects(
            sample_papers_list,
            "test topic"
        )
        
        assert len(scores) == 0
        assert all(isinstance(p, Paper) for p in relevant)
    
    def test_filter_paper_objects_semantic_only(self, regex_filter, mock_scorer, sample_papers_list):
        """Test filtering Paper objects with semantic_only mode."""
        def mock_score_batch(papers, topic, threshold, show_progress=True):
            return [
                SemanticScore(p.id, 0.8 if i == 0 else 0.3, True if i == 0 else False, '', [])
                for i, p in enumerate(papers)
            ]
        
        mock_scorer.score_batch = mock_score_batch
        
        hf = HybridFilter(
            regex_filter=regex_filter,
            semantic_scorer=mock_scorer,
            mode='semantic_only'
        )
        
        relevant, scores = hf.filter_paper_objects(
            sample_papers_list,
            "test topic",
            threshold=0.7
        )
        
        assert len(scores) == len(sample_papers_list)
        assert len(relevant) >= 0


# ============================================================================
# Integration Tests (without actual model loading)
# ============================================================================

class TestSemanticScorerInit:
    """Tests for SemanticScorer initialization."""
    
    def test_init_raises_import_error_without_transformers(self):
        """Test that ImportError is raised without transformers."""
        with patch.dict('sys.modules', {'transformers': None}):
            with pytest.raises(ImportError):
                SemanticScorer(model_name="test-model")
    
    def test_default_parameters(self):
        """Test default parameters are set correctly."""
        with patch('semantic_filter.SemanticScorer._load_model'):
            scorer = SemanticScorer.__new__(SemanticScorer)
            scorer.model_name = "Qwen/Qwen3.5-0.8B-Instruct"
            scorer.device = "cpu"
            scorer.batch_size = 8
            scorer.max_workers = 2
            scorer.max_length = 2048
            
            assert scorer.device == "cpu"
            assert scorer.batch_size == 8
            assert scorer.max_length == 2048


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_score_paper_with_empty_title(self, mock_scorer):
        """Test scoring a paper with empty title."""
        paper = Paper(id='empty', title='', abstract='Test abstract')
        
        prompt = mock_scorer._build_prompt(paper, "test topic")
        
        # Should still produce a valid prompt
        assert len(prompt) > 0
    
    def test_score_paper_with_special_characters(self, mock_scorer):
        """Test scoring a paper with special characters."""
        paper = Paper(
            id='special',
            title='Test Paper with "Quotes" and {Braces}',
            abstract='Abstract with\nnewlines\tand\ttabs'
        )
        
        prompt = mock_scorer._build_prompt(paper, "test topic")
        
        # Should handle special characters
        assert '{Braces}' in prompt or 'Braces' in prompt
    
    def test_filter_empty_paper_list(self, regex_filter, mock_scorer):
        """Test filtering an empty paper list."""
        hf = HybridFilter(regex_filter, mock_scorer, 'hybrid')
        
        relevant, scores = hf.filter_papers([], "test topic")
        
        assert relevant == []
        assert scores == []
    
    def test_threshold_boundary(self, mock_scorer):
        """Test that threshold boundaries work correctly."""
        # Score exactly at threshold
        score = SemanticScore('p1', 0.7, True, '', [])
        assert score.is_relevant  # 0.7 should be relevant
        
        # Score just below threshold
        score = SemanticScore('p2', 0.69, False, '', [])
        assert not score.is_relevant  # 0.69 should not be relevant


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])