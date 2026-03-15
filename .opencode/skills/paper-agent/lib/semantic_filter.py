#!/usr/bin/env python3
"""
Semantic Filtering using Local LLM (Qwen3.5)

Provides CPU-optimized semantic relevance scoring for academic papers.
Supports three filtering modes: regex_only, semantic_only, and hybrid.
"""

from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import json
import re
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))

from database import Paper


@dataclass
class SemanticScore:
    """
    Result of semantic relevance scoring.
    
    Attributes:
        paper_id: Unique identifier of the paper
        relevance_score: Float between 0.0 and 1.0
        is_relevant: Boolean indicating if score meets threshold
        reasoning: Brief explanation of relevance assessment
        key_aspects: List of key aspects identified in the paper
    """
    paper_id: str
    relevance_score: float  # 0.0 to 1.0
    is_relevant: bool
    reasoning: str
    key_aspects: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'paper_id': self.paper_id,
            'relevance_score': self.relevance_score,
            'is_relevant': self.is_relevant,
            'reasoning': self.reasoning,
            'key_aspects': self.key_aspects
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SemanticScore':
        """Create from dictionary."""
        return cls(
            paper_id=data['paper_id'],
            relevance_score=data['relevance_score'],
            is_relevant=data['is_relevant'],
            reasoning=data['reasoning'],
            key_aspects=data.get('key_aspects', [])
        )


class SemanticScorer:
    """
    CPU-optimized semantic relevance scorer using local LLM.
    
    Uses Qwen3.5-0.8B or Qwen3.5-2B for CPU inference.
    Optimized for batch processing and memory efficiency.
    
    Example:
        >>> scorer = SemanticScorer(model_name="Qwen/Qwen3.5-0.8B-Instruct")
        >>> paper = Paper(id="test", title="AI Paper", abstract="...")
        >>> score = scorer.score_paper(paper, topic="machine learning")
        >>> print(score.relevance_score)
        0.85
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3.5-0.8B-Instruct",
        device: str = "cpu",
        batch_size: int = 8,
        max_workers: int = 2,
        max_length: int = 2048
    ):
        """
        Initialize semantic scorer.
        
        Args:
            model_name: HuggingFace model name (default: Qwen3.5-0.8B-Instruct)
            device: Device for inference, 'cpu' only in Phase 1
            batch_size: Batch size for inference (8 recommended for CPU)
            max_workers: Max parallel workers (2 recommended for CPU)
            max_length: Max token length for model input
            
        Raises:
            ImportError: If transformers library not installed
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.max_length = max_length
        
        self.model = None
        self.tokenizer = None
        self._torch = None
        self._load_model()
    
    def _load_model(self):
        """Load model and tokenizer from HuggingFace Hub."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            self._torch = torch
            
            print(f"Loading model {self.model_name} on {self.device}...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Ensure pad token is set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,  # CPU requires float32
                device_map=self.device,
                trust_remote_code=True
            )
            self.model.eval()
            
            print(f"Model loaded successfully. Device: {self.device}")
            
        except ImportError as e:
            raise ImportError(
                "transformers library required for semantic filtering. "
                "Install with: pip install transformers>=4.35.0"
            ) from e
    
    def _build_prompt(self, paper: Paper, topic: str) -> str:
        """
        Build structured prompt for relevance scoring.
        
        Args:
            paper: Paper instance to score
            topic: Research topic description
            
        Returns:
            Formatted prompt string
        """
        # Truncate abstract to fit context window
        abstract_text = paper.abstract[:1500] if paper.abstract else "No abstract available."
        
        return f"""Analyze whether this research paper is relevant to the research topic.

Research Topic: {topic}

Paper Title: {paper.title}

Abstract: {abstract_text}

Respond in JSON format:
{{
    "score": <float between 0.0 and 1.0>,
    "reasoning": "<brief explanation of relevance>",
    "key_aspects": ["<aspect1>", "<aspect2>"]
}}

The score should be:
- 0.0-0.3: Not relevant
- 0.3-0.5: Weakly relevant  
- 0.5-0.7: Moderately relevant
- 0.7-1.0: Highly relevant

JSON response:"""
    
    def _parse_response(self, response: str) -> Tuple[float, str, List[str]]:
        """
        Parse model response to extract score and reasoning.
        
        Args:
            response: Raw model output string
            
        Returns:
            Tuple of (score, reasoning, key_aspects)
        """
        # Try to find JSON in response
        try:
            # Look for JSON block
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                
                score = float(data.get('score', 0.0))
                # Clamp score to valid range
                score = max(0.0, min(1.0, score))
                
                reasoning = str(data.get('reasoning', 'No reasoning provided'))
                key_aspects = data.get('key_aspects', [])
                
                if not isinstance(key_aspects, list):
                    key_aspects = [str(key_aspects)]
                
                return score, reasoning, key_aspects
                
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            pass
        
        # Fallback: try to extract score with regex
        score_match = re.search(r'score[:\s]+(\d+\.?\d*)', response.lower())
        if score_match:
            try:
                score = float(score_match.group(1))
                if score > 1.0:  # Handle percentage
                    score = score / 100.0
                score = max(0.0, min(1.0, score))
                return score, response, []
            except ValueError:
                pass
        
        # Default: not relevant
        return 0.0, response, []
    
    def score_paper(self, paper: Paper, topic: str) -> SemanticScore:
        """
        Score a single paper for relevance.
        
        Args:
            paper: Paper to score
            topic: Research topic description
            
        Returns:
            SemanticScore with relevance information
        """
        prompt = self._build_prompt(paper, topic)
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate response
        with self._torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,  # Low temperature for consistent scoring
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Decode and extract only the new tokens
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract response after the prompt
        if "JSON response:" in full_response:
            response = full_response.split("JSON response:")[-1].strip()
        else:
            response = full_response
        
        # Parse response
        score, reasoning, key_aspects = self._parse_response(response)
        
        return SemanticScore(
            paper_id=paper.id,
            relevance_score=score,
            is_relevant=score >= 0.7,  # Default threshold
            reasoning=reasoning,
            key_aspects=key_aspects
        )
    
    def score_batch(
        self,
        papers: List[Paper],
        topic: str,
        threshold: float = 0.7,
        show_progress: bool = True
    ) -> List[SemanticScore]:
        """
        Score a batch of papers.
        
        Args:
            papers: List of papers to score
            topic: Research topic
            threshold: Relevance threshold (0.0-1.0)
            show_progress: Show progress bar
            
        Returns:
            List of SemanticScore objects
        """
        results = []
        
        # Setup progress bar if available
        iterator = papers
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(papers, desc="Scoring papers semantically")
            except ImportError:
                pass
        
        for i, paper in enumerate(iterator):
            try:
                score = self.score_paper(paper, topic)
                score.is_relevant = score.relevance_score >= threshold
                results.append(score)
            except Exception as e:
                # On error, create a default score
                results.append(SemanticScore(
                    paper_id=paper.id,
                    relevance_score=0.0,
                    is_relevant=False,
                    reasoning=f"Error during scoring: {str(e)}",
                    key_aspects=[]
                ))
            
            # Simple rate limiting between papers
            if i < len(papers) - 1:
                import time
                time.sleep(0.1)
        
        return results
    
    def get_statistics(self, scores: List[SemanticScore]) -> Dict[str, Any]:
        """
        Get statistics from semantic scores.
        
        Args:
            scores: List of SemanticScore objects
            
        Returns:
            Dictionary with statistics
        """
        if not scores:
            return {
                'total_papers': 0,
                'relevant_count': 0,
                'irrelevant_count': 0,
                'avg_score': 0.0,
                'score_distribution': {}
            }
        
        relevant = [s for s in scores if s.is_relevant]
        
        # Score distribution
        score_ranges = {
            '0.0-0.3': 0,
            '0.3-0.5': 0,
            '0.5-0.7': 0,
            '0.7-1.0': 0
        }
        
        for s in scores:
            if s.relevance_score < 0.3:
                score_ranges['0.0-0.3'] += 1
            elif s.relevance_score < 0.5:
                score_ranges['0.3-0.5'] += 1
            elif s.relevance_score < 0.7:
                score_ranges['0.5-0.7'] += 1
            else:
                score_ranges['0.7-1.0'] += 1
        
        return {
            'total_papers': len(scores),
            'relevant_count': len(relevant),
            'irrelevant_count': len(scores) - len(relevant),
            'relevance_rate': len(relevant) / len(scores) * 100 if scores else 0,
            'avg_score': sum(s.relevance_score for s in scores) / len(scores),
            'score_distribution': score_ranges
        }


class HybridFilter:
    """
    Combines regex and semantic filtering for optimal results.
    
    Supports three modes:
    - regex_only: Fast keyword filtering only
    - semantic_only: Accurate semantic scoring only
    - hybrid: Regex pre-filter + semantic fine-filter (recommended)
    
    Example:
        >>> from filter import KeywordFilter, FilterConfig
        >>> config = FilterConfig(include_groups=[['ai', 'ml']])
        >>> regex_filter = KeywordFilter(config)
        >>> scorer = SemanticScorer()
        >>> hybrid = HybridFilter(regex_filter, scorer, mode='hybrid')
        >>> relevant, scores = hybrid.filter_papers(papers, 'machine learning')
    """
    
    VALID_MODES = ('regex_only', 'semantic_only', 'hybrid')
    
    def __init__(
        self,
        regex_filter: 'KeywordFilter',
        semantic_scorer: Optional[SemanticScorer] = None,
        mode: str = "hybrid"
    ):
        """
        Initialize hybrid filter.
        
        Args:
            regex_filter: Existing KeywordFilter instance
            semantic_scorer: SemanticScorer instance (optional for regex_only mode)
            mode: Filtering mode - 'regex_only', 'semantic_only', 'hybrid'
            
        Raises:
            ValueError: If mode is invalid or semantic_scorer required but not provided
        """
        self.regex_filter = regex_filter
        self.semantic_scorer = semantic_scorer
        self.mode = mode
        
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Invalid mode: {mode}. Must be one of {self.VALID_MODES}"
            )
        
        if mode in ('semantic_only', 'hybrid') and semantic_scorer is None:
            raise ValueError(
                f"Semantic scorer required for mode '{mode}'"
            )
    
    def filter_papers(
        self,
        papers: List[Dict],
        topic: str,
        threshold: float = 0.7
    ) -> Tuple[List[Dict], List[SemanticScore]]:
        """
        Filter papers using selected mode.
        
        Args:
            papers: List of paper dictionaries to filter
            topic: Research topic description
            threshold: Relevance threshold for semantic filter
            
        Returns:
            Tuple of (filtered_papers, all_scores)
            - filtered_papers: List of relevant papers
            - all_scores: List of SemanticScore objects (empty for regex_only mode)
        """
        if self.mode == "regex_only":
            # Use existing regex filter only
            relevant, _ = self.regex_filter.filter_papers(papers)
            return relevant, []
        
        elif self.mode == "semantic_only":
            # Use only semantic filter
            # Convert dicts to Paper objects if needed
            paper_objects = []
            for p in papers:
                if isinstance(p, Paper):
                    paper_objects.append(p)
                else:
                    paper_objects.append(Paper.from_dict(p))
            
            scores = self.semantic_scorer.score_batch(
                paper_objects, topic, threshold
            )
            
            relevant = []
            for p, s in zip(papers, scores):
                if s.is_relevant:
                    paper_copy = p.copy() if isinstance(p, dict) else p.to_dict()
                    paper_copy['relevance_score'] = s.relevance_score
                    paper_copy['semantic_reasoning'] = s.reasoning
                    relevant.append(paper_copy)
            
            return relevant, scores
        
        else:  # hybrid mode
            # Step 1: Regex pre-filter (fast)
            pre_filtered, _ = self.regex_filter.filter_papers(papers)
            print(f"Regex pre-filter: {len(pre_filtered)}/{len(papers)} papers passed")
            
            # Step 2: Semantic fine-filter (accurate)
            if self.semantic_scorer and pre_filtered:
                # Convert dicts to Paper objects
                paper_objects = []
                for p in pre_filtered:
                    if isinstance(p, Paper):
                        paper_objects.append(p)
                    else:
                        paper_objects.append(Paper.from_dict(p))
                
                scores = self.semantic_scorer.score_batch(
                    paper_objects, topic, threshold
                )
                
                relevant = []
                for p, s in zip(pre_filtered, scores):
                    if s.is_relevant:
                        paper_copy = p.copy() if isinstance(p, dict) else p.to_dict()
                        paper_copy['relevance_score'] = s.relevance_score
                        paper_copy['semantic_reasoning'] = s.reasoning
                        relevant.append(paper_copy)
                
                print(f"Semantic fine-filter: {len(relevant)}/{len(pre_filtered)} papers relevant")
                return relevant, scores
            
            return pre_filtered, []
    
    def filter_paper_objects(
        self,
        papers: List[Paper],
        topic: str,
        threshold: float = 0.7
    ) -> Tuple[List[Paper], List[SemanticScore]]:
        """
        Filter Paper objects using selected mode.
        
        Args:
            papers: List of Paper objects to filter
            topic: Research topic description
            threshold: Relevance threshold for semantic filter
            
        Returns:
            Tuple of (filtered_papers, all_scores)
        """
        if self.mode == "regex_only":
            # Convert to dict for regex filter
            paper_dicts = [p.to_dict() for p in papers]
            relevant_dicts, _ = self.regex_filter.filter_papers(paper_dicts)
            relevant_papers = [Paper.from_dict(d) for d in relevant_dicts]
            return relevant_papers, []
        
        elif self.mode == "semantic_only":
            scores = self.semantic_scorer.score_batch(papers, topic, threshold)
            relevant = [p for p, s in zip(papers, scores) if s.is_relevant]
            
            # Update relevance scores on papers
            for p, s in zip(papers, scores):
                p.relevance_score = s.relevance_score
            
            return relevant, scores
        
        else:  # hybrid mode
            # Step 1: Regex pre-filter
            paper_dicts = [p.to_dict() for p in papers]
            pre_filtered_dicts, _ = self.regex_filter.filter_papers(paper_dicts)
            pre_filtered = [Paper.from_dict(d) for d in pre_filtered_dicts]
            
            print(f"Regex pre-filter: {len(pre_filtered)}/{len(papers)} papers passed")
            
            # Step 2: Semantic fine-filter
            if self.semantic_scorer and pre_filtered:
                scores = self.semantic_scorer.score_batch(
                    pre_filtered, topic, threshold
                )
                
                relevant = [p for p, s in zip(pre_filtered, scores) if s.is_relevant]
                
                # Update relevance scores
                for p, s in zip(pre_filtered, scores):
                    p.relevance_score = s.relevance_score
                
                print(f"Semantic fine-filter: {len(relevant)}/{len(pre_filtered)} papers relevant")
                return relevant, scores
            
            return pre_filtered, []
    
    def get_mode_description(self) -> str:
        """Get description of current filtering mode."""
        descriptions = {
            'regex_only': "Fast keyword-based filtering without semantic understanding",
            'semantic_only': "Accurate semantic relevance scoring for all papers",
            'hybrid': "Two-stage: regex pre-filter followed by semantic fine-filter"
        }
        return descriptions.get(self.mode, "Unknown mode")


def filter_papers_semantic(
    papers: List[Dict],
    topic: str,
    config: 'FilterConfig',
    model_name: str = "Qwen/Qwen3.5-0.8B-Instruct",
    mode: str = "hybrid",
    threshold: float = 0.7,
    show_progress: bool = True
) -> Tuple[List[Dict], List[SemanticScore]]:
    """
    Convenience function for semantic filtering.
    
    Args:
        papers: List of paper dictionaries
        topic: Research topic description
        config: FilterConfig for regex filtering
        model_name: HuggingFace model name
        mode: Filtering mode ('regex_only', 'semantic_only', 'hybrid')
        threshold: Relevance threshold
        show_progress: Show progress bar
        
    Returns:
        Tuple of (filtered_papers, scores)
    """
    from filter import KeywordFilter
    
    # Create regex filter
    regex_filter = KeywordFilter(config)
    
    # Create semantic scorer if needed
    semantic_scorer = None
    if mode in ('semantic_only', 'hybrid'):
        semantic_scorer = SemanticScorer(model_name=model_name)
    
    # Create hybrid filter and run
    hybrid_filter = HybridFilter(
        regex_filter=regex_filter,
        semantic_scorer=semantic_scorer,
        mode=mode
    )
    
    return hybrid_filter.filter_papers(papers, topic, threshold)