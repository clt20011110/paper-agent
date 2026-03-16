#!/usr/bin/env python3
"""
Database Module for Paper Agent
Provides unified storage layer supporting JSON and CSV formats with incremental updates.
"""

import json
import csv
import tempfile
import os
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any, Union, Literal


@dataclass
class Paper:
    """
    Represents a single academic paper with comprehensive metadata.
    
    Attributes:
        id: Unique identifier (OpenReview ID / DOI / arXiv ID)
        title: Paper title
        abstract: Paper abstract
        authors: List of author names
        keywords: List of keywords/tags
        year: Publication year
        venue: Conference/journal name
        venue_type: 'conference' or 'journal'
        source_platform: Platform where paper was found
        crawl_date: ISO datetime when crawled
        last_updated: ISO datetime of last update
        download_status: Status of PDF download
        analysis_status: Status of analysis
        pdf_url: PDF download URL
        doi: Digital Object Identifier
        bibtex: BibTeX citation string
        citation_count: Number of citations if available
        arxiv_id: arXiv identifier
        download_available: Which platforms have PDF available
        analysis_file: Path to analysis results file
        relevance_score: Semantic relevance score (0.0-1.0)
    """
    # Basic Information
    id: str
    title: str
    abstract: str = ""
    authors: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    
    # Venue Information
    year: int = 0
    venue: str = ""
    venue_type: Literal['conference', 'journal', 'preprint'] = 'conference'
    source_platform: str = 'openreview'  # 'openreview', 'arxiv', 'acm', 'ieee', 'nature', etc.
    
    # Status Tracking
    crawl_date: str = ""  # ISO datetime
    last_updated: str = ""  # ISO datetime
    download_status: Literal['pending', 'success', 'failed', 'skipped'] = 'pending'
    analysis_status: Literal['pending', 'success', 'failed', 'skipped'] = 'pending'
    
    # Metadata
    pdf_url: Optional[str] = None
    doi: Optional[str] = None
    bibtex: Optional[str] = None
    citation_count: Optional[int] = None
    arxiv_id: Optional[str] = None
    download_available: Literal['openreview', 'arxiv', 'both', 'none'] = 'none'
    
    # Analysis Results
    analysis_file: Optional[str] = None
    relevance_score: Optional[float] = None
    
    def __post_init__(self):
        """Initialize timestamps if not provided."""
        if not self.crawl_date:
            self.crawl_date = datetime.now().isoformat()
        if not self.last_updated:
            self.last_updated = self.crawl_date
    
    def update_timestamp(self):
        """Update the last_updated timestamp to now."""
        self.last_updated = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert paper to dictionary representation."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Paper':
        """
        Create Paper instance from dictionary.
        
        Handles backward compatibility with old format by filling missing fields
        with defaults.
        """
        # Handle old format compatibility
        # Map old 'venue_id' to venue if venue is empty
        if 'venue_id' in data and not data.get('venue'):
            data['venue'] = data['venue_id']
        
        # Map old 'decision' to download_status hint
        if 'decision' in data and data.get('download_status') is None:
            decision = data.get('decision', '').lower()
            if decision == 'accepted':
                data['download_status'] = 'pending'
            elif decision == 'rejected':
                data['download_status'] = 'skipped'
        
        # Ensure all list fields are lists
        for list_field in ['authors', 'keywords']:
            if data.get(list_field) is None:
                data[list_field] = []
            elif isinstance(data[list_field], str):
                data[list_field] = [data[list_field]]
        
        # Create paper with only valid fields
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        
        return cls(**filtered_data)
    
    @classmethod
    def from_legacy_dict(cls, data: Dict[str, Any], venue: str = "", year: int = 0) -> 'Paper':
        """
        Create Paper from legacy crawler output format.
        
        Args:
            data: Dictionary from legacy JSON format
            venue: Venue name (extracted from venue_id if not provided)
            year: Publication year
            
        Returns:
            Paper instance with mapped fields
        """
        # Extract venue info from venue_id if present
        venue_id = data.get('venue_id', '')
        if venue_id and not venue:
            # Extract venue name from ID like "ICLR.cc/2024/Conference"
            parts = venue_id.split('/')
            if len(parts) >= 1:
                venue = parts[0].replace('.cc', '').replace('.org', '')
            if len(parts) >= 2 and parts[1].isdigit():
                year = int(parts[1])
        
        # Determine download_available from pdf_url
        pdf_url = data.get('pdf_url')
        download_available = 'none'
        if pdf_url:
            if 'arxiv' in pdf_url.lower():
                download_available = 'arxiv'
            elif 'openreview' in pdf_url.lower():
                download_available = 'openreview'
        
        paper_data = {
            'id': data.get('id', ''),
            'title': data.get('title', ''),
            'abstract': data.get('abstract', ''),
            'authors': data.get('authors', []),
            'keywords': data.get('keywords', []),
            'year': year,
            'venue': venue,
            'venue_type': 'conference',
            'source_platform': 'openreview',
            'pdf_url': pdf_url,
            'download_available': download_available,
        }
        
        return cls.from_dict(paper_data)


class DatabaseManager:
    """
    Manages paper database with support for JSON and CSV formats.
    
    Provides CRUD operations, incremental updates, and atomic file operations
    for safe concurrent access.
    
    Attributes:
        db_path: Path to database file
        format: Storage format ('json' or 'csv')
        papers: Dictionary of papers indexed by ID
    """
    
    def __init__(self, db_path: Union[str, Path], format: str = 'json'):
        """
        Initialize database manager.
        
        Args:
            db_path: Path to database file (extension determines format if not specified)
            format: Storage format ('json' or 'csv'). If None, auto-detected from path.
        """
        self.db_path = Path(db_path)
        
        # Auto-detect format from file extension if not specified
        if format == 'json' and self.db_path.suffix.lower() == '.csv':
            self.format = 'csv'
        elif format == 'csv' and self.db_path.suffix.lower() == '.json':
            self.format = 'json'
        else:
            self.format = format.lower()
        
        if self.format not in ('json', 'csv'):
            raise ValueError(f"Unsupported format: {format}. Use 'json' or 'csv'.")
        
        # In-memory storage
        self._papers: Dict[str, Paper] = {}
        
        # Load existing database if it exists
        if self.db_path.exists():
            self._load()
    
    def _load(self) -> None:
        """Load database from disk."""
        if self.format == 'json':
            self._load_json()
        else:
            self._load_csv()
    
    def _load_json(self) -> None:
        """Load database from JSON file."""
        try:
            with open(self.db_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle both old and new format
            if isinstance(data, dict) and 'papers' in data:
                # Old format: {"papers": [...]}
                papers_list = data['papers']
            elif isinstance(data, list):
                # Simple list of papers
                papers_list = data
            else:
                papers_list = []
            
            for paper_dict in papers_list:
                paper = Paper.from_dict(paper_dict)
                self._papers[paper.id] = paper
                
        except json.JSONDecodeError:
            # File exists but is empty or corrupted
            self._papers = {}
    
    def _load_csv(self) -> None:
        """Load database from CSV file."""
        try:
            import pandas as pd
            df = pd.read_csv(self.db_path)
            
            for _, row in df.iterrows():
                paper_dict = row.to_dict()
                # Handle list fields stored as strings
                for field_name in ['authors', 'keywords']:
                    if field_name in paper_dict and isinstance(paper_dict[field_name], str):
                        try:
                            paper_dict[field_name] = json.loads(paper_dict[field_name])
                        except json.JSONDecodeError:
                            paper_dict[field_name] = paper_dict[field_name].split(';')
                
                paper = Paper.from_dict(paper_dict)
                self._papers[paper.id] = paper
                
        except Exception:
            # File exists but is empty or corrupted
            self._papers = {}
    
    def _save_json_atomic(self) -> None:
        """
        Save database to JSON file atomically.
        
        Uses temp file + rename for safe concurrent writes.
        """
        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data
        papers_list = [p.to_dict() for p in self._papers.values()]
        
        # Write to temp file first
        temp_fd, temp_path = tempfile.mkstemp(
            suffix='.json',
            dir=self.db_path.parent,
            prefix=f'.{self.db_path.stem}_tmp_'
        )
        
        try:
            with os.fdopen(temp_fd, 'w', encoding='utf-8') as f:
                json.dump(papers_list, f, ensure_ascii=False, indent=2)
            
            # Atomic rename
            shutil.move(temp_path, self.db_path)
            
        except Exception:
            # Clean up temp file on error
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise
    
    def _save_csv_atomic(self) -> None:
        """
        Save database to CSV file atomically.
        
        Uses temp file + rename for safe concurrent writes.
        """
        try:
            import pandas as pd
            
            # Ensure parent directory exists
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert papers to DataFrame
            papers_list = [p.to_dict() for p in self._papers.values()]
            df = pd.DataFrame(papers_list)
            
            # Handle list fields - convert to JSON strings
            for col in ['authors', 'keywords']:
                if col in df.columns:
                    df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, list) else x)
            
            # Write to temp file first
            temp_fd, temp_path = tempfile.mkstemp(
                suffix='.csv',
                dir=self.db_path.parent,
                prefix=f'.{self.db_path.stem}_tmp_'
            )
            
            try:
                with os.fdopen(temp_fd, 'w', encoding='utf-8', newline='') as f:
                    df.to_csv(f, index=False)
                
                # Atomic rename
                shutil.move(temp_path, self.db_path)
                
            except Exception:
                # Clean up temp file on error
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                raise
                
        except ImportError:
            # Fallback to built-in csv module if pandas not available
            self._save_csv_native()
    
    def _save_csv_native(self) -> None:
        """Save to CSV using built-in csv module (no pandas dependency)."""
        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not self._papers:
            # Write empty file
            self.db_path.touch()
            return
        
        papers_list = [p.to_dict() for p in self._papers.values()]
        fieldnames = list(papers_list[0].keys())
        
        temp_fd, temp_path = tempfile.mkstemp(
            suffix='.csv',
            dir=self.db_path.parent,
            prefix=f'.{self.db_path.stem}_tmp_'
        )
        
        try:
            with os.fdopen(temp_fd, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for paper_dict in papers_list:
                    # Convert list fields to strings
                    row = paper_dict.copy()
                    for key in ['authors', 'keywords']:
                        if isinstance(row.get(key), list):
                            row[key] = json.dumps(row[key])
                    writer.writerow(row)
            
            # Atomic rename
            shutil.move(temp_path, self.db_path)
            
        except Exception:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise
    
    def save(self) -> None:
        """
        Persist database to disk.
        
        Uses atomic write operations for safe concurrent access.
        """
        if self.format == 'json':
            self._save_json_atomic()
        else:
            self._save_csv_atomic()
    
    def add_paper(self, paper: Paper) -> bool:
        """
        Add a single paper to the database.
        
        Args:
            paper: Paper instance to add
            
        Returns:
            True if paper was added, False if duplicate (same ID exists)
        """
        if paper.id in self._papers:
            return False
        
        paper.update_timestamp()
        self._papers[paper.id] = paper
        return True
    
    def add_papers(self, papers: List[Paper]) -> int:
        """
        Add multiple papers to the database.
        
        Args:
            papers: List of Paper instances to add
            
        Returns:
            Number of papers actually added (excludes duplicates)
        """
        added = 0
        for paper in papers:
            if self.add_paper(paper):
                added += 1
        return added
    
    def get_paper(self, paper_id: str) -> Optional[Paper]:
        """
        Retrieve a paper by ID.
        
        Args:
            paper_id: Unique paper identifier
            
        Returns:
            Paper instance if found, None otherwise
        """
        return self._papers.get(paper_id)
    
    def get_all_papers(self) -> List[Paper]:
        """
        Retrieve all papers in the database.
        
        Returns:
            List of all Paper instances
        """
        return list(self._papers.values())
    
    def update_paper(self, paper: Paper) -> bool:
        """
        Update an existing paper.
        
        Args:
            paper: Paper instance with updated data
            
        Returns:
            True if paper was updated, False if not found
        """
        if paper.id not in self._papers:
            return False
        
        paper.update_timestamp()
        self._papers[paper.id] = paper
        return True
    
    def delete_paper(self, paper_id: str) -> bool:
        """
        Delete a paper from the database.
        
        Args:
            paper_id: ID of paper to delete
            
        Returns:
            True if paper was deleted, False if not found
        """
        if paper_id not in self._papers:
            return False
        
        del self._papers[paper_id]
        return True
    
    def merge(self, other_db_path: Union[str, Path]) -> Tuple[int, int]:
        """
        Merge another database into this one.
        
        Args:
            other_db_path: Path to database file to merge
            
        Returns:
            Tuple of (papers_added, papers_updated)
        """
        other_path = Path(other_db_path)
        
        # Detect format from other file
        other_format = 'json' if other_path.suffix.lower() == '.json' else 'csv'
        
        # Create temporary manager for other database
        other_db = DatabaseManager(other_path, format=other_format)
        
        added = 0
        updated = 0
        
        for paper in other_db.get_all_papers():
            if paper.id in self._papers:
                # Update existing paper
                self._papers[paper.id] = paper
                updated += 1
            else:
                # Add new paper
                self._papers[paper.id] = paper
                added += 1
        
        return added, updated
    
    def incremental_update(self, new_papers: List[Paper]) -> Tuple[int, int]:
        """
        Incrementally update database with new papers.
        
        Detects duplicates based on paper ID. Updates existing papers
        with new data, adds new papers.
        
        Args:
            new_papers: List of papers to add/update
            
        Returns:
            Tuple of (papers_added, papers_updated)
        """
        added = 0
        updated = 0
        
        for paper in new_papers:
            if paper.id in self._papers:
                # Update existing paper
                paper.update_timestamp()
                self._papers[paper.id] = paper
                updated += 1
            else:
                # Add new paper
                paper.update_timestamp()
                self._papers[paper.id] = paper
                added += 1
        
        return added, updated
    
    def filter_papers(
        self,
        venue: Optional[str] = None,
        year: Optional[int] = None,
        source_platform: Optional[str] = None,
        download_status: Optional[str] = None,
        analysis_status: Optional[str] = None,
        min_relevance_score: Optional[float] = None
    ) -> List[Paper]:
        """
        Filter papers by various criteria.
        
        Args:
            venue: Filter by venue name
            year: Filter by publication year
            source_platform: Filter by source platform
            download_status: Filter by download status
            analysis_status: Filter by analysis status
            min_relevance_score: Minimum relevance score
            
        Returns:
            List of papers matching all criteria
        """
        results = []
        
        for paper in self._papers.values():
            match = True
            
            if venue and paper.venue != venue:
                match = False
            if year and paper.year != year:
                match = False
            if source_platform and paper.source_platform != source_platform:
                match = False
            if download_status and paper.download_status != download_status:
                match = False
            if analysis_status and paper.analysis_status != analysis_status:
                match = False
            if min_relevance_score is not None:
                if paper.relevance_score is None or paper.relevance_score < min_relevance_score:
                    match = False
            
            if match:
                results.append(paper)
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dictionary with various statistics about the database
        """
        papers = self.get_all_papers()
        
        stats = {
            'total_papers': len(papers),
            'by_venue': {},
            'by_year': {},
            'by_source_platform': {},
            'by_download_status': {
                'pending': 0,
                'success': 0,
                'failed': 0,
                'skipped': 0
            },
            'by_analysis_status': {
                'pending': 0,
                'success': 0,
                'failed': 0,
                'skipped': 0
            },
            'with_pdf_url': 0,
            'with_doi': 0,
            'analyzed': 0
        }
        
        for paper in papers:
            # Count by venue
            venue = paper.venue or 'unknown'
            stats['by_venue'][venue] = stats['by_venue'].get(venue, 0) + 1
            
            # Count by year
            year = paper.year or 'unknown'
            stats['by_year'][year] = stats['by_year'].get(year, 0) + 1
            
            # Count by source platform
            platform = paper.source_platform or 'unknown'
            stats['by_source_platform'][platform] = stats['by_source_platform'].get(platform, 0) + 1
            
            # Count by download status
            if paper.download_status in stats['by_download_status']:
                stats['by_download_status'][paper.download_status] += 1
            
            # Count by analysis status
            if paper.analysis_status in stats['by_analysis_status']:
                stats['by_analysis_status'][paper.analysis_status] += 1
            
            # Count optional fields
            if paper.pdf_url:
                stats['with_pdf_url'] += 1
            if paper.doi:
                stats['with_doi'] += 1
            if paper.analysis_file:
                stats['analyzed'] += 1
        
        return stats
    
    def __len__(self) -> int:
        """Return number of papers in database."""
        return len(self._papers)
    
    def __contains__(self, paper_id: str) -> bool:
        """Check if paper ID exists in database."""
        return paper_id in self._papers
    
    def __iter__(self):
        """Iterate over papers in database."""
        return iter(self._papers.values())


def convert_legacy_json(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    format: str = 'json'
) -> Path:
    """
    Convert legacy crawler JSON output to new database format.
    
    Args:
        input_path: Path to legacy JSON file
        output_path: Output path (defaults to input_path with new extension)
        format: Output format ('json' or 'csv')
        
    Returns:
        Path to converted database file
    """
    input_path = Path(input_path)
    
    if output_path is None:
        suffix = '.json' if format == 'json' else '.csv'
        output_path = input_path.with_suffix(suffix)
    else:
        output_path = Path(output_path)
    
    # Read legacy format
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle different legacy formats
    if isinstance(data, dict) and 'papers' in data:
        # Format: {"crawl_time": ..., "papers": [...]}
        papers_list = data['papers']
        venue = data.get('venue', '')
        year = data.get('year', 0)
    elif isinstance(data, list):
        # Format: [...]
        papers_list = data
        venue = ''
        year = 0
    else:
        papers_list = []
        venue = ''
        year = 0
    
    # Create new database
    db = DatabaseManager(output_path, format=format)
    
    # Convert and add papers
    for paper_dict in papers_list:
        paper = Paper.from_legacy_dict(paper_dict, venue, year)
        db.add_paper(paper)
    
    # Save
    db.save()
    
    return output_path


def create_database_from_papers(
    papers: List[Dict[str, Any]],
    output_path: Union[str, Path],
    format: str = 'json'
) -> DatabaseManager:
    """
    Create a new database from a list of paper dictionaries.
    
    Args:
        papers: List of paper dictionaries (legacy or new format)
        output_path: Path for the database file
        format: Storage format ('json' or 'csv')
        
    Returns:
        DatabaseManager instance with papers loaded
    """
    output_path = Path(output_path)
    db = DatabaseManager(output_path, format=format)
    
    for paper_dict in papers:
        paper = Paper.from_dict(paper_dict)
        db.add_paper(paper)
    
    db.save()
    return db