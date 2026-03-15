# Paper Agent Enhancement - Detailed Work Plan

## Executive Summary

This document outlines a comprehensive 6-phase enhancement plan for the paper-agent project, transforming it from a basic OpenReview crawler into a full-featured academic paper research system with multi-source support, semantic filtering, and intelligent analysis.

---

## Current Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    paper_agent.py (CLI Entry)                   │
└─────────────────────────────────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌───────────────┐      ┌───────────────┐      ┌───────────────┐
│   Stage 1     │      │   Stage 2     │      │   Stage 3-4   │
│   crawler.py  │──────▶│   filter.py   │──────▶│ downloader/   │
│               │      │               │      │ analyzer.py   │
│ OpenReview    │      │ Regex-based   │      │               │
│ Only          │      │ keyword filter│      │ OpenRouter API│
└───────────────┘      └───────────────┘      └───────────────┘
```

**Current Limitations:**
- Only 4 conferences (ICLR, NeurIPS, ICML, AAAI) via OpenReview
- Simple regex-based filtering
- No persistent database (JSON files only)
- No semantic understanding
- No arXiv search integration
- PDF required for analysis

---

## Phase 1: Database Layer & Storage (Foundation)

**Duration:** 5-7 days  
**Branch:** `feature/database-layer`  
**Priority:** CRITICAL (blocks Phase 2-5)

### 1.1 Tasks Breakdown

| Task ID | Task Name | Description | Est. Time | Dependencies | Category |
|---------|-----------|-------------|-----------|--------------|----------|
| P1-001 | Design unified schema | Define Paper schema with all required fields | 4h | None | design |
| P1-002 | Implement database module | Create `lib/database.py` with CRUD operations | 8h | P1-001 | backend |
| P1-003 | Implement JSON backend | JSON file read/write with atomic updates | 4h | P1-002 | backend |
| P1-004 | Implement CSV backend | CSV file read/write with pandas | 4h | P1-002 | backend |
| P1-005 | Implement merge function | Merge multiple database files | 4h | P1-002 | backend |
| P1-006 | Implement incremental update | Detect duplicates and update existing records | 6h | P1-002 | backend |
| P1-007 | Create migration utility | Convert old JSON format to new schema | 3h | P1-003 | backend |
| P1-008 | Refactor crawler output | Update crawler.py to use database module | 4h | P1-003, P1-004 | backend |
| P1-009 | Update paper_agent.py | Integrate database module into CLI | 4h | P1-008 | backend |
| P1-010 | Unit tests for database | Test all CRUD operations | 4h | P1-002 to P1-006 | testing |
| P1-011 | Integration test | End-to-end test with crawler | 3h | P1-008, P1-009 | testing |

### 1.2 Database Schema

```python
@dataclass
class Paper:
    # Basic Information
    id: str                    # Unique identifier (OpenReview ID / DOI / arXiv ID)
    title: str                 # Paper title
    abstract: str              # Paper abstract
    authors: List[str]         # Author list
    keywords: List[str]        # Keywords/tags
    
    # Venue Information
    year: int                  # Publication year
    venue: str                 # Conference/journal name
    venue_type: str            # 'conference' or 'journal'
    source_platform: str       # 'openreview', 'arxiv', 'acm', 'ieee', 'nature', etc.
    
    # Status Tracking
    crawl_date: str            # ISO datetime when crawled
    last_updated: str          # ISO datetime of last update
    download_status: str       # 'pending', 'success', 'failed', 'skipped'
    analysis_status: str       # 'pending', 'success', 'failed', 'skipped'
    
    # Metadata
    pdf_url: Optional[str]     # PDF download URL
    doi: Optional[str]         # DOI
    bibtex: Optional[str]      # BibTeX citation
    citation_count: Optional[int]  # Citation count if available
    arxiv_id: Optional[str]    # arXiv ID if available
    download_available: str    # 'openreview', 'arxiv', 'both', 'none'
    
    # Analysis Results
    analysis_file: Optional[str]   # Path to analysis file
    relevance_score: Optional[float]  # Semantic relevance score
```

### 1.3 Parallel Execution Opportunities

```
P1-001 ──────┬─── P1-002 ──┬─── P1-003 ──┐
             │             │             │
             │             └─── P1-004 ──┼─── P1-005
             │                           │
             └───────────────────────────┴─── P1-006
                                          │
                                          ▼
P1-007 (parallel with P1-005, P1-006) ───┤
                                          │
                                          ▼
P1-008 ───────────────────────────────────┴─── P1-009 ── P1-010 ── P1-011
```

**Parallel tracks:**
- Track A: P1-003 (JSON) and P1-004 (CSV) can be developed in parallel
- Track B: P1-005 (merge) and P1-006 (incremental) can start after P1-002
- Track C: P1-007 (migration) can proceed in parallel with P1-005/P1-006

### 1.4 Test Plan

| Test ID | Test Name | Description | Priority |
|---------|-----------|-------------|----------|
| T1-001 | CRUD operations | Test create, read, update, delete for single paper | High |
| T1-002 | Batch operations | Test bulk insert and batch updates | High |
| T1-003 | Duplicate detection | Test ID-based deduplication | High |
| T1-004 | Merge function | Test merging two database files | Medium |
| T1-005 | Schema validation | Test all required fields are present | High |
| T1-006 | Backward compatibility | Test migration from old format | High |
| T1-007 | JSON serialization | Test JSON file read/write | High |
| T1-008 | CSV serialization | Test CSV file read/write | Medium |
| T1-009 | Incremental update | Test updating existing records | High |
| T1-010 | Concurrent access | Test thread-safe operations | Medium |

### 1.5 Success Criteria

- [ ] Database module supports both JSON and CSV formats
- [ ] All CRUD operations pass unit tests
- [ ] Incremental update correctly identifies and updates duplicates
- [ ] Migration utility successfully converts old data
- [ ] Existing crawler output integrates with new database
- [ ] Backward compatibility maintained (old config files work)

---

## Phase 2: Crawler Expansion

**Duration:** 10-14 days  
**Branch:** `feature/crawler-expansion`  
**Priority:** HIGH  
**Depends on:** Phase 1 (P1-008)

### 2.1 Tasks Breakdown

| Task ID | Task Name | Description | Est. Time | Dependencies | Category |
|---------|-----------|-------------|-----------|--------------|----------|
| P2-001 | Research platform APIs | Investigate ACL, CVPR, ICCV platforms | 8h | None | research |
| P2-002 | Design plugin architecture | Create base VenueAdapter class | 6h | None | design |
| P2-003 | Implement OpenReview adapter | Refactor existing crawler into adapter | 4h | P2-002 | backend |
| P2-004 | Implement ACL adapter | ACL anthology crawler | 8h | P2-002 | backend |
| P2-005 | Implement CVPR adapter | CVF conference crawler | 8h | P2-002 | backend |
| P2-006 | Implement ICCV adapter | CVF conference crawler (similar to CVPR) | 4h | P2-005 | backend |
| P2-007 | Implement IJCAI adapter | IJCAI website crawler | 8h | P2-002 | backend |
| P2-008 | Implement DAC adapter | IEEE/ACM DAC crawler | 8h | P2-002 | backend |
| P2-009 | Implement TCAD adapter | IEEE TCAD journal crawler | 6h | P2-002 | backend |
| P2-010 | Implement ICCAD adapter | IEEE/ACM ICCAD crawler | 6h | P2-002 | backend |
| P2-011 | Design journal adapter base | Create JournalAdapter base class | 4h | P2-002 | design |
| P2-012 | Implement Nature adapter | Nature journals crawler via API/scraper | 12h | P2-011 | backend |
| P2-013 | Implement Cell adapter | Cell journal crawler | 8h | P2-011 | backend |
| P2-014 | Implement Science adapter | Science journal crawler | 8h | P2-011 | backend |
| P2-015 | Create venue registry | Plugin registration mechanism | 4h | P2-002 | backend |
| P2-016 | Update config schema | Add sources configuration section | 3h | P2-015 | backend |
| P2-017 | Update paper_agent.py | Add venue selection to CLI | 4h | P2-016 | backend |
| P2-018 | Integration tests | Test each adapter end-to-end | 8h | P2-003 to P2-014 | testing |
| P2-019 | Documentation | Document each adapter's capabilities | 4h | P2-003 to P2-014 | docs |

### 2.2 Plugin Architecture Design

```python
# lib/adapters/base.py
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class VenueConfig:
    name: str
    years: List[int]
    platform: str
    additional_params: Dict = None

class VenueAdapter(ABC):
    """Base class for all venue adapters"""
    
    @property
    @abstractmethod
    def platform_name(self) -> str:
        """Return platform identifier"""
        pass
    
    @abstractmethod
    def crawl(self, config: VenueConfig) -> List[Dict]:
        """Crawl papers from the venue"""
        pass
    
    @abstractmethod
    def get_pdf_url(self, paper_id: str) -> Optional[str]:
        """Get PDF download URL for a paper"""
        pass
    
    def supports_journal(self) -> bool:
        """Whether this adapter supports journals"""
        return False


# lib/adapters/registry.py
class VenueRegistry:
    """Plugin registry for venue adapters"""
    
    _adapters = {}
    
    @classmethod
    def register(cls, adapter_class):
        """Register an adapter class"""
        instance = adapter_class()
        cls._adapters[instance.platform_name] = instance
        return adapter_class
    
    @classmethod
    def get(cls, platform: str) -> Optional[VenueAdapter]:
        """Get adapter by platform name"""
        return cls._adapters.get(platform)
    
    @classmethod
    def list_available(cls) -> List[str]:
        """List all registered platforms"""
        return list(cls._adapters.keys())
```

### 2.3 Platform Research Summary

| Venue | Platform | API Available | PDF Access | Notes |
|-------|----------|---------------|------------|-------|
| ACL | ACL Anthology | Yes (REST) | Yes | Open access |
| CVPR | CVF Open Access | No (scraping) | Yes | Need rate limiting |
| ICCV | CVF Open Access | No (scraping) | Yes | Similar to CVPR |
| IJCAI | IJCAI.org | No (scraping) | Varies | May need subscription |
| DAC | IEEE Xplore | Yes (IEEE API) | Subscription | Title/abstract only |
| TCAD | IEEE Xplore | Yes (IEEE API) | Subscription | Title/abstract only |
| ICCAD | IEEE Xplore | Yes (IEEE API) | Subscription | Title/abstract only |
| Nature | Nature.com API | Yes (REST) | Subscription | Use CrossRef for metadata |
| Cell | Cell.com | No (scraping) | Subscription | Use PubMed for metadata |
| Science | Science.org | No (scraping) | Subscription | Use CrossRef for metadata |

### 2.4 Parallel Execution Opportunities

```
P2-001 (research) ─────────────────────────────────────────────┐
                                                                │
P2-002 ─── P2-003 ─────────────────────────────────────────────┤
    │                                                          │
    ├── P2-004 (ACL) ──────────────────────────────────────────┤
    │                                                          │
    ├── P2-005 (CVPR) ─── P2-006 (ICCV) ───────────────────────┤  PARALLEL
    │                                                          │  DEVELOPMENT
    ├── P2-007 (IJCAI) ────────────────────────────────────────┤  (multiple
    │                                                          │  adapters)
    ├── P2-008 (DAC) ──────────────────────────────────────────┤
    │                                                          │
    ├── P2-009 (TCAD) ─────────────────────────────────────────┤
    │                                                          │
    ├── P2-010 (ICCAD) ────────────────────────────────────────┤
    │                                                          │
    └── P2-011 ─── P2-012 (Nature) ────────────────────────────┤
                 │                                              │
                 ├── P2-013 (Cell) ─────────────────────────────┤
                 │                                              │
                 └── P2-014 (Science) ──────────────────────────┘
                                                                │
P2-015 (registry) ─────────────────────────────────────────────┤
                                                                │
P2-016 ─── P2-017 ─── P2-018 ─── P2-019 ───────────────────────┘
```

**Recommended parallel teams:**
- Team A: OpenReview adapter refactor + ACL + IJCAI
- Team B: CVPR + ICCV (similar architecture)
- Team C: DAC + TCAD + ICCAD (EDA conferences)
- Team D: Nature + Cell + Science (journals)

### 2.5 Test Plan

| Test ID | Test Name | Description | Priority |
|---------|-----------|-------------|----------|
| T2-001 | Adapter registration | Test venue registry functionality | High |
| T2-002 | ACL adapter | Crawl ACL papers (small test set) | High |
| T2-003 | CVPR adapter | Crawl CVPR papers | High |
| T2-004 | ICCV adapter | Crawl ICCV papers | Medium |
| T2-005 | IJCAI adapter | Crawl IJCAI papers | Medium |
| T2-006 | DAC adapter | Crawl DAC papers | Medium |
| T2-007 | TCAD adapter | Crawl TCAD papers | Low |
| T2-008 | ICCAD adapter | Crawl ICCAD papers | Low |
| T2-009 | Nature adapter | Crawl Nature MI papers | High |
| T2-010 | Cell adapter | Crawl Cell papers | Medium |
| T2-011 | Science adapter | Crawl Science papers | Medium |
| T2-012 | PDF URL retrieval | Test get_pdf_url for each adapter | High |
| T2-013 | Error handling | Test graceful failure on invalid venues | High |
| T2-014 | Rate limiting | Test rate limiting compliance | Medium |
| T2-015 | Data format consistency | Test all adapters produce same schema | Critical |

### 2.6 Success Criteria

- [ ] At least 5 new CCF-A conferences supported
- [ ] At least 5 Nature sub-journals supported
- [ ] Plugin architecture allows easy addition of new venues
- [ ] All adapters produce consistent data format
- [ ] Configuration supports dynamic venue addition
- [ ] Each adapter handles errors gracefully
- [ ] Rate limiting implemented for all scrapers

---

## Phase 3: Semantic Filtering

**Duration:** 7-10 days  
**Branch:** `feature/semantic-filter`  
**Priority:** HIGH  
**Depends on:** Phase 1 (database module)

### 3.1 Tasks Breakdown

| Task ID | Task Name | Description | Est. Time | Dependencies | Category |
|---------|-----------|-------------|-----------|--------------|----------|
| P3-001 | Research model requirements | Evaluate Qwen models for CPU inference | 4h | None | research |
| P3-002 | Implement model loader | Load Qwen3.5-0.8B/2B with transformers | 6h | P3-001 | backend |
| P3-003 | Implement CPU optimization | Batch processing, memory management | 6h | P3-002 | backend |
| P3-004 | Design scoring prompt | Create structured prompt for relevance scoring | 4h | None | design |
| P3-005 | Implement semantic scorer | Score papers with reasoning | 8h | P3-002, P3-004 | backend |
| P3-006 | Create SemanticFilter class | Integrate scorer into filter module | 4h | P3-005 | backend |
| P3-007 | Implement hybrid filter | Combine regex + semantic filtering | 6h | P3-006 | backend |
| P3-008 | Update config schema | Add semantic filter configuration | 2h | P3-007 | backend |
| P3-009 | Update paper_agent.py | Add filter mode selection to CLI | 3h | P3-008 | backend |
| P3-010 | Performance benchmarking | Test CPU inference performance | 4h | P3-003 | testing |
| P3-011 | Unit tests | Test semantic filter components | 4h | P3-006 | testing |
| P3-012 | Integration tests | Test hybrid filter end-to-end | 3h | P3-007 | testing |

### 3.2 Semantic Filter Architecture

```python
# lib/semantic_filter.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Optional
import torch

class SemanticScorer:
    """CPU-optimized semantic relevance scorer"""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3.5-0.8B-Instruct",
        device: str = "cpu",
        batch_size: int = 8,
        max_workers: int = 2
    ):
        self.device = device
        self.batch_size = batch_size
        self.max_workers = max_workers
        
        # Load model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # CPU requires float32
            device_map=device
        )
        self.model.eval()
    
    def score_batch(
        self,
        papers: List[Dict],
        topic: str,
        threshold: float = 0.7
    ) -> List[Dict]:
        """Score a batch of papers for relevance"""
        results = []
        
        for paper in papers:
            prompt = self._build_prompt(paper, topic)
            response = self._generate(prompt)
            score, reasoning = self._parse_response(response)
            
            results.append({
                'paper_id': paper['id'],
                'relevance_score': score,
                'is_relevant': score >= threshold,
                'reasoning': reasoning
            })
        
        return results
    
    def _build_prompt(self, paper: Dict, topic: str) -> str:
        """Build structured prompt for relevance scoring"""
        return f"""Analyze whether this paper is relevant to the research topic.

Topic: {topic}

Paper Title: {paper['title']}

Abstract: {paper['abstract'][:1000]}

Respond in JSON format:
{{
    "score": <float 0-1>,
    "reasoning": "<brief explanation>",
    "key_aspects": ["aspect1", "aspect2"]
}}"""


class HybridFilter:
    """Combines regex and semantic filtering"""
    
    def __init__(
        self,
        regex_filter: KeywordFilter,
        semantic_filter: Optional[SemanticScorer],
        mode: str = "hybrid"  # "regex_only", "semantic_only", "hybrid"
    ):
        self.regex_filter = regex_filter
        self.semantic_filter = semantic_filter
        self.mode = mode
    
    def filter(self, papers: List[Dict], topic: str, threshold: float = 0.7) -> List[Dict]:
        """Filter papers based on mode"""
        if self.mode == "regex_only":
            relevant, _ = self.regex_filter.filter_papers(papers)
            return relevant
        
        elif self.mode == "semantic_only":
            scores = self.semantic_filter.score_batch(papers, topic, threshold)
            return [p for p, s in zip(papers, scores) if s['is_relevant']]
        
        else:  # hybrid
            # Step 1: Regex pre-filter (fast)
            pre_filtered, _ = self.regex_filter.filter_papers(papers)
            
            # Step 2: Semantic fine-filter (accurate)
            if self.semantic_filter:
                scores = self.semantic_filter.score_batch(pre_filtered, topic, threshold)
                return [p for p, s in zip(pre_filtered, scores) if s['is_relevant']]
            
            return pre_filtered
```

### 3.3 CPU Optimization Strategy

```yaml
# Recommended CPU settings
semantic:
  model: "Qwen/Qwen3.5-0.8B-Instruct"  # Smaller model for CPU
  device: "cpu"
  batch_size: 8         # Small batches for memory efficiency
  max_workers: 2        # Limited parallelism on CPU
  quantization: null    # No quantization for CPU (Phase 1)
  
# Memory estimation:
# Qwen3.5-0.8B: ~1.6GB (float32)
# Qwen3.5-2B:   ~4GB (float32)
```

### 3.4 Parallel Execution Opportunities

```
P3-001 ────────────────────────────────────────────────────────┐
                                                                │
P3-002 ─── P3-003 ───┐                                         │
                      │                                         │
P3-004 ──────────────┼─── P3-005 ─── P3-006 ─── P3-007 ────────┤
                      │                                         │
                      │                                         │
                      └─────────────────────────────────────────┤
                                                                │
P3-008 ─── P3-009 ─── P3-010 ─── P3-011 ─── P3-012 ────────────┘
```

**Parallel tracks:**
- P3-001 and P3-004 can run in parallel (research + prompt design)
- P3-010 (benchmarking) can start after P3-003
- P3-011 (unit tests) can be written in parallel with P3-006

### 3.5 Test Plan

| Test ID | Test Name | Description | Priority |
|---------|-----------|-------------|----------|
| T3-001 | Model loading | Test model loads successfully on CPU | Critical |
| T3-002 | Batch processing | Test batch scoring functionality | High |
| T3-003 | Prompt parsing | Test JSON response parsing | High |
| T3-004 | Score accuracy | Test scoring with known papers | High |
| T3-005 | Memory management | Test memory doesn't exceed limits | High |
| T3-006 | Hybrid filter regex | Test regex-only mode | High |
| T3-007 | Hybrid filter semantic | Test semantic-only mode | High |
| T3-008 | Hybrid filter combined | Test hybrid mode | Critical |
| T3-009 | Threshold configuration | Test different thresholds | Medium |
| T3-010 | Performance benchmark | Measure papers/second on CPU | High |
| T3-011 | Large batch handling | Test with 100+ papers | Medium |

### 3.6 Success Criteria

- [ ] Qwen3.5-0.8B or 2B loads successfully on CPU
- [ ] Semantic scoring produces 0-1 scores with reasoning
- [ ] Hybrid mode combines regex pre-filter + semantic fine-filter
- [ ] CPU inference processes at least 1 paper/second
- [ ] Memory usage stays under 4GB for 0.8B model
- [ ] Filter mode configurable via YAML
- [ ] All three modes (regex_only, semantic_only, hybrid) work correctly

---

## Phase 4: arXiv Integration

**Duration:** 5-7 days  
**Branch:** `feature/arxiv-integration`  
**Priority:** MEDIUM  
**Depends on:** Phase 1 (database), Phase 2 (adapters)

### 4.1 Tasks Breakdown

| Task ID | Task Name | Description | Est. Time | Dependencies | Category |
|---------|-----------|-------------|-----------|--------------|----------|
| P4-001 | Implement arXiv searcher | Search arXiv using keywords | 6h | None | backend |
| P4-002 | Implement category filter | Filter by cs.AI, cs.LG, etc. | 3h | P4-001 | backend |
| P4-003 | Implement date filter | Filter by date range | 3h | P4-001 | backend |
| P4-004 | Create arXiv adapter | Wrap as VenueAdapter | 4h | P4-001, P4-002, P4-003 | backend |
| P4-005 | Implement download tagging | Mark download_available field | 4h | P4-004 | backend |
| P4-006 | Update downloader logic | Support multi-source downloads | 6h | P4-005 | backend |
| P4-007 | Update config schema | Add arxiv configuration section | 2h | P4-004 | backend |
| P4-008 | Update paper_agent.py | Add arXiv search to CLI | 3h | P4-007 | backend |
| P4-009 | Unit tests | Test arXiv search functionality | 4h | P4-004 | testing |
| P4-010 | Integration tests | Test download fallback chain | 4h | P4-006 | testing |

### 4.2 arXiv Integration Architecture

```python
# lib/adapters/arxiv_adapter.py
import feedparser
import requests
from datetime import datetime
from typing import List, Dict, Optional

class ArxivAdapter(VenueAdapter):
    """arXiv search adapter"""
    
    @property
    def platform_name(self) -> str:
        return "arxiv"
    
    def crawl(self, config: VenueConfig) -> List[Dict]:
        """Search arXiv by keywords"""
        papers = []
        
        query = self._build_query(
            keywords=config.additional_params.get('keywords', ''),
            categories=config.additional_params.get('categories', []),
            date_range=config.additional_params.get('date_range', None)
        )
        
        url = f"http://export.arxiv.org/api/query?{query}&max_results=100"
        response = requests.get(url, timeout=30)
        feed = feedparser.parse(response.content)
        
        for entry in feed.entries:
            papers.append(self._parse_entry(entry))
        
        return papers
    
    def _parse_entry(self, entry) -> Dict:
        """Parse arXiv entry to paper dict"""
        arxiv_id = entry.id.split('/abs/')[-1]
        
        return {
            'id': f"arxiv:{arxiv_id}",
            'title': entry.title,
            'abstract': entry.summary,
            'authors': [a.name for a in entry.authors],
            'year': entry.published_parsed.tm_year,
            'venue': 'arXiv',
            'venue_type': 'preprint',
            'source_platform': 'arxiv',
            'arxiv_id': arxiv_id,
            'pdf_url': f"https://arxiv.org/pdf/{arxiv_id}.pdf",
            'download_available': 'arxiv',
            'categories': [tag.term for tag in entry.tags]
        }


# lib/downloader.py (enhanced)
class MultiSourceDownloader:
    """Downloads from multiple sources with fallback"""
    
    DOWNLOAD_PRIORITY = ['openreview', 'arxiv']
    
    def download(self, paper: Dict) -> Optional[Path]:
        """Try each available source in priority order"""
        available = paper.get('download_available', 'none')
        
        if available == 'both':
            sources = self.DOWNLOAD_PRIORITY
        elif available == 'openreview':
            sources = ['openreview']
        elif available == 'arxiv':
            sources = ['arxiv']
        else:
            return None
        
        for source in sources:
            path = self._try_source(source, paper)
            if path:
                return path
        
        return None
```

### 4.3 Parallel Execution Opportunities

```
P4-001 ───┬── P4-002 ───┐
          │             │
          └── P4-003 ───┼── P4-004 ─── P4-005 ─── P4-006 ─── P4-007 ─── P4-008 ─── P4-009 ─── P4-010
                        │
                        └───────────────────────────────────────────────────────────────────────────┘
```

**Parallel tracks:**
- P4-002 and P4-003 can run in parallel (category + date filters)

### 4.4 Test Plan

| Test ID | Test Name | Description | Priority |
|---------|-----------|-------------|----------|
| T4-001 | arXiv search | Test keyword search returns results | High |
| T4-002 | Category filter | Test cs.AI, cs.LG filtering | High |
| T4-003 | Date range filter | Test date range filtering | High |
| T4-004 | Result parsing | Test entry to Paper conversion | High |
| T4-005 | Download tagging | Test download_available marking | High |
| T4-006 | Multi-source download | Test fallback chain | Critical |
| T4-007 | OpenReview priority | Test OpenReview tried first when available | Medium |
| T4-008 | arXiv fallback | Test arXiv fallback when OpenReview fails | Medium |
| T4-009 | API rate limiting | Test arXiv API compliance | Medium |
| T4-010 | Empty results | Test handling of no results | Low |

### 4.5 Success Criteria

- [ ] arXiv search returns papers matching keywords
- [ ] Category and date filters work correctly
- [ ] Papers tagged with correct download_available value
- [ ] Multi-source download tries sources in priority order
- [ ] Fallback to arXiv when OpenReview fails
- [ ] arXiv results don't pollute main database (optional save)
- [ ] Rate limiting respects arXiv API guidelines

---

## Phase 5: Analysis Enhancement

**Duration:** 5-7 days  
**Branch:** `feature/analysis-enhancement`  
**Priority:** MEDIUM  
**Depends on:** Phase 1 (database), Phase 4 (download tagging)

### 5.1 Tasks Breakdown

| Task ID | Task Name | Description | Est. Time | Dependencies | Category |
|---------|-----------|-------------|-----------|--------------|----------|
| P5-001 | Design abstract analysis prompt | Prompt for title+abstract analysis | 3h | None | design |
| P5-002 | Implement fallback analyzer | Analyze without PDF | 6h | P5-001 | backend |
| P5-003 | Add confidence marking | Mark full_pdf vs abstract_only | 2h | P5-002 | backend |
| P5-004 | Design summary prompt | Prompt for research summary generation | 4h | None | design |
| P5-005 | Implement paper aggregator | Aggregate analysis results | 4h | None | backend |
| P5-006 | Implement summary generator | Generate research summary report | 8h | P5-004, P5-005 | backend |
| P5-007 | Update config schema | Add summary model configuration | 2h | P5-006 | backend |
| P5-008 | Update paper_agent.py | Add stage4b command | 3h | P5-007 | backend |
| P5-009 | Unit tests | Test analysis components | 4h | P5-002, P5-006 | testing |
| P5-010 | Integration tests | Test full analysis pipeline | 4h | P5-008 | testing |

### 5.2 Analysis Enhancement Architecture

```python
# lib/analyzer.py (enhanced)

ABSTRACT_ANALYSIS_PROMPT = """Analyze this paper based on title and abstract only.
Note: Full PDF was not available, so analysis is limited.

## Title
{title}

## Abstract
{abstract}

Please provide analysis in Markdown format:
## Research Question
## Main Approach
## Key Contributions (estimated)
## Relevance Assessment
## Confidence: abstract_only (limited analysis)
"""


class EnhancedAnalyzer:
    """Analyzer with PDF fallback and summary generation"""
    
    def analyze_paper(
        self,
        paper: Dict,
        pdf_path: Optional[Path] = None,
        api_key: str = None
    ) -> Dict:
        """Analyze paper with optional PDF fallback"""
        
        if pdf_path and pdf_path.exists():
            # Full PDF analysis
            result = self._analyze_pdf(pdf_path, api_key)
            result['confidence'] = 'full_pdf'
        else:
            # Fallback to abstract analysis
            result = self._analyze_abstract(paper, api_key)
            result['confidence'] = 'abstract_only'
        
        return result
    
    def _analyze_abstract(self, paper: Dict, api_key: str) -> Dict:
        """Analyze paper using only title and abstract"""
        prompt = ABSTRACT_ANALYSIS_PROMPT.format(
            title=paper['title'],
            abstract=paper['abstract']
        )
        return self._call_api(prompt, api_key)


class ResearchSummaryGenerator:
    """Generate comprehensive research summary"""
    
    SUMMARY_PROMPT = """Based on the following paper analyses, generate a comprehensive research summary.

## Topic: {topic}

## Papers Analyzed:
{paper_summaries}

Generate a research summary with the following sections:
1. Research Trends Overview
2. Main Method Categories
3. Key Technical Approaches
4. Dataset & Benchmark Analysis
5. Open Source Resources
6. Research Gaps and Future Directions

Format as Markdown."""

    def generate(
        self,
        analyses: List[Dict],
        topic: str,
        api_key: str,
        model: str = "anthropic/claude-3.5-sonnet"
    ) -> str:
        """Generate research summary from analyses"""
        
        # Aggregate analyses
        paper_summaries = self._aggregate(analyses)
        
        # Build prompt
        prompt = self.SUMMARY_PROMPT.format(
            topic=topic,
            paper_summaries=paper_summaries
        )
        
        # Call high-capability model
        return self._call_api(prompt, api_key, model)
```

### 5.3 Parallel Execution Opportunities

```
P5-001 ─── P5-002 ─── P5-003 ───────────────────────────────────┐
                                                                 │
P5-004 ─── P5-005 ─── P5-006 ───────────────────────────────────┤
                                                                 │
                              P5-007 ─── P5-008 ─── P5-009 ─── P5-010
```

**Parallel tracks:**
- P5-001/P5-002 and P5-004/P5-005 can run in parallel (different features)

### 5.4 Test Plan

| Test ID | Test Name | Description | Priority |
|---------|-----------|-------------|----------|
| T5-001 | Abstract analysis | Test analysis without PDF | High |
| T5-002 | Confidence marking | Test full_pdf vs abstract_only marking | High |
| T5-003 | PDF fallback | Test automatic fallback when PDF missing | High |
| T5-004 | Summary generation | Test research summary output | High |
| T5-005 | Large paper set | Test summary with 50+ papers | Medium |
| T5-006 | Summary quality | Test summary contains required sections | High |
| T5-007 | Model selection | Test different summary models | Medium |
| T5-008 | Error handling | Test graceful failure on API errors | High |

### 5.5 Success Criteria

- [ ] Fallback analysis works when PDF unavailable
- [ ] Analysis results marked with confidence level
- [ ] Stage 4b generates comprehensive research summary
- [ ] Summary includes all required sections
- [ ] High-capability model used for summary generation
- [ ] Large paper sets handled with chunking if needed

---

## Phase 6: Testing & Documentation

**Duration:** 5-7 days  
**Branch:** `feature/testing-documentation`  
**Priority:** MEDIUM  
**Depends on:** All previous phases

### 6.1 Tasks Breakdown

| Task ID | Task Name | Description | Est. Time | Dependencies | Category |
|---------|-----------|-------------|-----------|--------------|----------|
| P6-001 | Setup pytest framework | Configure pytest with coverage | 2h | None | testing |
| P6-002 | Database module tests | Comprehensive database tests | 4h | P6-001 | testing |
| P6-003 | Adapter tests | All venue adapter tests | 6h | P6-001 | testing |
| P6-004 | Filter tests | Regex and semantic filter tests | 4h | P6-001 | testing |
| P6-005 | Downloader tests | Download functionality tests | 4h | P6-001 | testing |
| P6-006 | Analyzer tests | Analysis functionality tests | 4h | P6-001 | testing |
| P6-007 | Integration tests | End-to-end pipeline tests | 8h | P6-002 to P6-006 | testing |
| P6-008 | Performance tests | Benchmark all modules | 4h | P6-007 | testing |
| P6-009 | Update README.md | Document new features | 3h | None | docs |
| P6-010 | Write API documentation | Document all public APIs | 6h | None | docs |
| P6-011 | Create example configs | Example configs for different use cases | 3h | None | docs |
| P6-012 | Write troubleshooting guide | Common issues and solutions | 2h | None | docs |
| P6-013 | Create CONTRIBUTING.md | Contribution guidelines | 2h | None | docs |

### 6.2 Test Directory Structure

```
tests/
├── __init__.py
├── conftest.py              # Shared fixtures
├── unit/
│   ├── test_database.py     # Database module tests
│   ├── test_adapters/
│   │   ├── test_openreview.py
│   │   ├── test_acl.py
│   │   ├── test_cvpr.py
│   │   ├── test_arxiv.py
│   │   └── test_nature.py
│   ├── test_filter.py       # Filter tests
│   ├── test_semantic.py     # Semantic filter tests
│   ├── test_downloader.py   # Downloader tests
│   └── test_analyzer.py     # Analyzer tests
├── integration/
│   ├── test_pipeline.py     # Full pipeline tests
│   ├── test_crawl_filter.py
│   └── test_download_analyze.py
└── fixtures/
    ├── sample_papers.json
    ├── sample_pdfs/
    └── mock_api_responses/
```

### 6.3 Parallel Execution Opportunities

```
P6-001 ──────────────────────────────────────────────────────────┐
                                                                  │
├── P6-002 ───────────────────────────────────────────────────────┤
│                                                                 │
├── P6-003 ───────────────────────────────────────────────────────┤  PARALLEL
│                                                                 │  TEST
├── P6-004 ───────────────────────────────────────────────────────┤  DEVELOPMENT
│                                                                 │
├── P6-005 ───────────────────────────────────────────────────────┤
│                                                                 │
├── P6-006 ───────────────────────────────────────────────────────┤
                                                                  │
├── P6-007 ─── P6-008 ────────────────────────────────────────────┤
                                                                  │
├── P6-009 ───────────────────────────────────────────────────────┤
│                                                                 │
├── P6-010 ───────────────────────────────────────────────────────┤  PARALLEL
│                                                                 │  DOCS
├── P6-011 ───────────────────────────────────────────────────────┤
│                                                                 │
├── P6-012 ───────────────────────────────────────────────────────┤
│                                                                 │
└── P6-013 ───────────────────────────────────────────────────────┘
```

### 6.4 Test Coverage Goals

| Module | Target Coverage | Notes |
|--------|-----------------|-------|
| database.py | 90% | Critical foundation |
| adapters/* | 80% | External API mocking |
| filter.py | 90% | Core functionality |
| semantic_filter.py | 85% | CPU-dependent tests |
| downloader.py | 85% | Network mocking |
| analyzer.py | 80% | API mocking |

### 6.5 Success Criteria

- [ ] Overall test coverage >= 85%
- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] Documentation covers all public APIs
- [ ] Example configs for common use cases
- [ ] CONTRIBUTING.md with development setup
- [ ] CI/CD pipeline configured (optional)

---

## Branch Naming Strategy

```
main (stable)
  │
  ├── feature/database-layer        # Phase 1
  │     └── feature/database-layer-json
  │     └── feature/database-layer-csv
  │
  ├── feature/crawler-expansion     # Phase 2
  │     └── feature/crawler-acl
  │     └── feature/crawler-cvpr
  │     └── feature/crawler-nature
  │     └── feature/crawler-eda
  │
  ├── feature/semantic-filter       # Phase 3
  │     └── feature/semantic-cpu-opt
  │     └── feature/semantic-hybrid
  │
  ├── feature/arxiv-integration     # Phase 4
  │
  ├── feature/analysis-enhancement  # Phase 5
  │     └── feature/analysis-fallback
  │     └── feature/analysis-summary
  │
  └── feature/testing-documentation # Phase 6
```

**Branch Naming Convention:**
- Feature branches: `feature/<phase-name>-<sub-feature>`
- Bug fixes: `fix/<issue-description>`
- Hotfixes: `hotfix/<critical-issue>`
- Releases: `release/v<major>.<minor>.<patch>`

**Merge Strategy:**
1. Create feature branch from `main`
2. Develop and test on feature branch
3. Create PR to `main`
4. Require code review + passing tests
5. Squash merge to `main`

---

## Dependency Graph (All Phases)

```
                    ┌─────────────────┐
                    │   Phase 1       │
                    │   Database      │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
     ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
     │   Phase 2   │ │   Phase 3   │ │   Phase 4   │
     │   Crawler   │ │  Semantic   │ │   arXiv     │
     │  Expansion  │ │  Filtering  │ │Integration  │
     └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
            │               │               │
            └───────────────┼───────────────┘
                            │
                            ▼
                   ┌─────────────┐
                   │   Phase 5   │
                   │  Analysis   │
                   │ Enhancement │
                   └──────┬──────┘
                          │
                          ▼
                   ┌─────────────┐
                   │   Phase 6   │
                   │  Testing &  │
                   │    Docs     │
                   └─────────────┘
```

**Parallel Development Possible:**
- Phase 2, 3, 4 can proceed in parallel after Phase 1
- Within Phase 2, multiple adapters can be developed in parallel
- Phase 5 depends on Phase 4 (download tagging)
- Phase 6 depends on all previous phases

---

## Timeline Estimate

| Phase | Duration | Start | End | Dependencies |
|-------|----------|-------|-----|--------------|
| Phase 1 | 5-7 days | Day 1 | Day 7 | None |
| Phase 2 | 10-14 days | Day 8 | Day 21 | Phase 1 |
| Phase 3 | 7-10 days | Day 8 | Day 17 | Phase 1 |
| Phase 4 | 5-7 days | Day 8 | Day 14 | Phase 1 |
| Phase 5 | 5-7 days | Day 15 | Day 21 | Phase 4 |
| Phase 6 | 5-7 days | Day 22 | Day 28 | All |

**Total Estimated Duration:** 4-5 weeks

**Critical Path:** Phase 1 → Phase 2 → Phase 5 → Phase 6

**Parallel Opportunities:**
- Weeks 2-3: Phase 2, 3, 4 in parallel (different teams)
- Phase 2 adapters can be developed by multiple teams

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| API rate limiting | High | Medium | Implement backoff, caching |
| Platform changes | Medium | High | Abstract adapters, mock tests |
| CPU memory limits | Medium | High | Strict batch size, model selection |
| PDF access restrictions | High | Medium | Fallback to abstract analysis |
| Model download size | Low | Medium | Pre-download, cache locally |
| Test data availability | Medium | Low | Use mock data, small test sets |

---

## Category + Skills Assignment

| Phase | Category | Recommended Skills | Notes |
|-------|----------|-------------------|-------|
| Phase 1 | backend | git-master | Database design, Python development |
| Phase 2 | backend, research | git-master, playwright | Web scraping, API integration |
| Phase 3 | ml, backend | git-master | Transformers, CPU optimization |
| Phase 4 | backend | git-master | API integration |
| Phase 5 | ml, backend | git-master | Prompt engineering, API integration |
| Phase 6 | testing, docs | git-master | pytest, documentation |

---

## Quick Reference: Task Summary by Category

### Backend Tasks
- P1-002 to P1-009: Database implementation
- P2-003 to P2-017: Crawler adapters and registry
- P3-002, P3-003, P3-005 to P3-009: Semantic filter
- P4-001 to P4-008: arXiv integration
- P5-002, P5-003, P5-005 to P5-008: Analysis enhancement

### Research Tasks
- P2-001: Platform API research
- P3-001: Model requirements research

### Design Tasks
- P1-001: Database schema design
- P2-002, P2-011: Adapter architecture design
- P3-004: Scoring prompt design
- P5-001, P5-004: Analysis prompts design

### Testing Tasks
- P1-010, P1-011: Database tests
- P2-018: Adapter integration tests
- P3-010, P3-011, P3-012: Semantic filter tests
- P4-009, P4-010: arXiv tests
- P5-009, P5-010: Analysis tests
- P6-001 to P6-008: All testing

### Documentation Tasks
- P2-019: Adapter documentation
- P6-009 to P6-013: All documentation

---

## Next Steps

1. **Review this plan** with the team
2. **Assign teams** to parallel tracks
3. **Create Phase 1 branch** and begin database implementation
4. **Setup CI/CD** for automated testing
5. **Schedule weekly syncs** for cross-phase coordination