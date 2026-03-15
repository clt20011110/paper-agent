#!/usr/bin/env python3
"""
Lib: OpenReview Crawler
爬取指定会议和年份的所有论文信息
"""

import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any

from database import DatabaseManager, Paper


def get_openreview_client() -> Tuple[Any, str]:
    """获取OpenReview客户端"""
    try:
        import openreview
        # 尝试使用API V2
        try:
            client = openreview.api.OpenReviewClient(
                baseurl='https://api2.openreview.net'
            )
            return client, 'v2'
        except:
            # 回退到API V1
            client = openreview.Client(
                baseurl='https://api.openreview.net'
            )
            return client, 'v1'
    except ImportError:
        print("错误: 请安装openreview-py: pip install openreview-py")
        raise RuntimeError("openreview-py not installed")


def convert_openreview_note_to_paper(note, venue_id: str, year: int, api_version: str) -> Paper:
    """
    Convert OpenReview note to Paper object.
    
    Args:
        note: OpenReview note object
        venue_id: Venue ID (e.g., "ICLR.cc/2024/Conference")
        year: Publication year
        api_version: API version ('v1' or 'v2')
    
    Returns:
        Paper object with extracted metadata
    """
    paper_id = note.id
    
    # Extract fields based on API version
    if api_version == 'v2':
        title = note.content.get('title', {}).get('value', '')
        abstract = note.content.get('abstract', {}).get('value', '')
        authors = note.content.get('authors', {}).get('value', [])
        keywords = note.content.get('keywords', {}).get('value', [])
        venue_status = note.content.get('venue', {}).get('value', '')
        pdf_value = note.content.get('pdf', {}).get('value', '')
    else:
        title = note.content.get('title', '')
        abstract = note.content.get('abstract', '')
        authors = note.content.get('authors', [])
        keywords = note.content.get('keywords', [])
        venue_status = note.content.get('venue', '')
        pdf_value = note.content.get('pdf', '')
    
    # Build PDF URL
    pdf_url = None
    if pdf_value:
        if pdf_value.startswith('/pdf'):
            pdf_url = f"https://openreview.net{pdf_value}"
        elif pdf_value.startswith('http'):
            pdf_url = pdf_value
        else:
            pdf_url = f"https://openreview.net/pdf?id={paper_id}"
    
    # Determine acceptance status
    decision = 'unknown'
    if venue_status:
        if any(x in venue_status.lower() for x in ['accept', 'oral', 'poster', 'spotlight']):
            decision = 'accepted'
        elif 'reject' in venue_status.lower():
            decision = 'rejected'
    
    # Determine download_available
    download_available = 'none'
    if pdf_url:
        if 'arxiv' in pdf_url.lower():
            download_available = 'arxiv'
        elif 'openreview' in pdf_url.lower():
            download_available = 'openreview'
    
    # Extract venue name from venue_id
    venue_name = venue_id.split('/')[0].replace('.cc', '').replace('.org', '') if venue_id else ''
    
    # Ensure keywords is a list
    if isinstance(keywords, str):
        keywords = [keywords] if keywords else []
    elif not isinstance(keywords, list):
        keywords = []
    
    # Create Paper object
    paper = Paper(
        id=paper_id,
        title=title,
        abstract=abstract,
        authors=authors if isinstance(authors, list) else [],
        keywords=keywords,
        year=year,
        venue=venue_name,
        venue_type='conference',
        source_platform='openreview',
        crawl_date=datetime.now().isoformat(),
        pdf_url=pdf_url,
        download_available=download_available,
    )
    
    return paper


def crawl_conference(client, venue_id: str, api_version: str, accepted_only: bool = True) -> List[Paper]:
    """
    爬取单个会议的所有论文
    
    Args:
        client: OpenReview客户端
        venue_id: 会议ID, 如 "ICLR.cc/2024/Conference"
        api_version: API版本 'v1' 或 'v2'
        accepted_only: 是否只获取已接受论文
    
    Returns:
        论文列表 (List[Paper])
    """
    papers = []
    
    try:
        print(f"  获取会议信息: {venue_id}")
        venue_group = client.get_group(venue_id)
        
        # 获取提交名称
        if api_version == 'v2':
            submission_name = venue_group.content.get('submission_name', {}).get('value', 'Submission')
        else:
            submission_name = 'Submission'
        
        print(f"  获取所有提交论文...")
        submissions = client.get_all_notes(
            invitation=f'{venue_id}/-/{submission_name}',
            details='directReplies'
        )
        
        print(f"  找到 {len(submissions)} 篇论文")
        
        # Extract year from venue_id
        year = 2024  # default
        for part in venue_id.split('/'):
            if part.isdigit():
                year = int(part)
                break
        
        for idx, paper in enumerate(submissions, 1):
            if idx % 50 == 0:
                print(f"    处理中... {idx}/{len(submissions)}")
            
            # Convert to Paper object
            paper_obj = convert_openreview_note_to_paper(paper, venue_id, year, api_version)
            
            # Check decision from venue_status if available
            venue_status = paper.content.get('venue', {}).get('value', '') if api_version == 'v2' else paper.content.get('venue', '')
            decision = 'unknown'
            if venue_status:
                if any(x in venue_status.lower() for x in ['accept', 'oral', 'poster', 'spotlight']):
                    decision = 'accepted'
                elif 'reject' in venue_status.lower():
                    decision = 'rejected'
            
            # If accepted_only, skip rejected papers
            if accepted_only and decision == 'rejected':
                continue
            
            papers.append(paper_obj)
            
            # Avoid rate limiting
            time.sleep(0.01)
        
        return papers
        
    except Exception as e:
        print(f"  错误: {e}")
        import traceback
        traceback.print_exc()
        return []


def crawl_venues(
    venues: List[str], 
    output_dir: Path, 
    accepted_only: bool = True,
    database_path: Optional[Path] = None
) -> Tuple[List[Paper], Optional[Path]]:
    """
    爬取多个会议的论文
    
    Args:
        venues: List of venue IDs to crawl
        output_dir: Output directory for JSON files (legacy mode)
        accepted_only: Only fetch accepted papers
        database_path: Optional path to database file. If provided, uses DatabaseManager.
    
    Returns:
        (所有论文列表, 汇总文件路径或数据库路径)
    """
    client, api_version = get_openreview_client()
    if not client:
        return [], None
    
    print(f"\n{'='*60}")
    print(f"🕷️ 阶段1: OpenReview爬虫")
    print(f"{'='*60}")
    print(f"API版本: {api_version}")
    if database_path:
        print(f"数据库模式: {database_path}")
    else:
        print(f"输出目录: {output_dir}")
    print(f"只接受已接受论文: {accepted_only}")
    print(f"{'='*60}\n")
    
    all_papers: List[Paper] = []
    
    # Determine mode: database or legacy JSON
    use_database = database_path is not None
    
    # Initialize database variables (needed for both paths)
    db: Optional[DatabaseManager] = None
    total_added = 0
    total_updated = 0
    
    if use_database:
        # Database mode
        db = DatabaseManager(database_path, format='json')
    else:
        # Legacy mode - create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
    
    for venue in venues:
        print(f"\n📚 正在爬取: {venue}")
        papers = crawl_conference(client, venue, api_version, accepted_only)
        
        if papers:
            # Extract year
            year = 2024  # default
            for part in venue.split('/'):
                if part.isdigit():
                    year = int(part)
                    break
            
            if use_database and db is not None:
                # Database mode: use incremental update
                added, updated = db.incremental_update(papers)
                total_added += added
                total_updated += updated
                print(f"  ✓ 已保存到数据库: +{added} 新增, ~{updated} 更新")
            else:
                # Legacy mode: save to JSON files
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                venue_name = venue.replace('/', '_').replace('.', '_')
                filename = f"{venue_name}_{year}_{timestamp}.json"
                output_path = output_dir / filename
                
                # Convert Paper objects to dicts for JSON serialization
                papers_dict = [p.to_dict() for p in papers]
                
                data = {
                    'venue': venue,
                    'year': year,
                    'crawl_time': datetime.now().isoformat(),
                    'total_papers': len(papers),
                    'papers': papers_dict
                }
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                
                print(f"  ✓ 已保存: {output_path} ({len(papers)}篇)")
            
            all_papers.extend(papers)
        else:
            print(f"  ⚠️ 未找到论文或爬取失败")
        
        # Delay between venues
        time.sleep(2)
    
    # Final save and return
    if use_database and db is not None:
        # Save database
        db.save()
        print(f"\n{'='*60}")
        print(f"✅ 爬取完成!")
        print(f"📊 总计: {len(all_papers)} 篇论文")
        print(f"📈 数据库更新: +{total_added} 新增, ~{total_updated} 更新")
        print(f"📁 数据库文件: {database_path}")
        print(f"{'='*60}\n")
        return all_papers, database_path
    else:
        # Legacy mode: save summary JSON
        if all_papers:
            summary_path = output_dir / f"all_papers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Convert Paper objects to dicts for JSON serialization
            papers_dict = [p.to_dict() for p in all_papers]
            
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'crawl_time': datetime.now().isoformat(),
                    'total_papers': len(all_papers),
                    'venues': venues,
                    'papers': papers_dict
                }, f, ensure_ascii=False, indent=2)
            
            print(f"\n{'='*60}")
            print(f"✅ 爬取完成!")
            print(f"📊 总计: {len(all_papers)} 篇论文")
            print(f"📁 汇总文件: {summary_path}")
            print(f"{'='*60}\n")
            
            return all_papers, summary_path
    
    return [], None