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


def crawl_conference(client, venue_id: str, api_version: str, accepted_only: bool = True) -> List[Dict]:
    """
    爬取单个会议的所有论文
    
    Args:
        client: OpenReview客户端
        venue_id: 会议ID, 如 "ICLR.cc/2024/Conference"
        api_version: API版本 'v1' 或 'v2'
        accepted_only: 是否只获取已接受论文
    
    Returns:
        论文列表
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
        
        for idx, paper in enumerate(submissions, 1):
            if idx % 50 == 0:
                print(f"    处理中... {idx}/{len(submissions)}")
            
            # 提取基本信息
            paper_id = paper.id
            
            if api_version == 'v2':
                title = paper.content.get('title', {}).get('value', '')
                abstract = paper.content.get('abstract', {}).get('value', '')
                authors = paper.content.get('authors', {}).get('value', [])
                keywords = paper.content.get('keywords', {}).get('value', [])
                venue_status = paper.content.get('venue', {}).get('value', '')
                pdf_value = paper.content.get('pdf', {}).get('value', '')
            else:
                title = paper.content.get('title', '')
                abstract = paper.content.get('abstract', '')
                authors = paper.content.get('authors', [])
                keywords = paper.content.get('keywords', [])
                venue_status = paper.content.get('venue', '')
                pdf_value = paper.content.get('pdf', '')
            
            # 构建PDF URL
            pdf_url = None
            if pdf_value:
                if pdf_value.startswith('/pdf'):
                    pdf_url = f"https://openreview.net{pdf_value}"
                elif pdf_value.startswith('http'):
                    pdf_url = pdf_value
                else:
                    pdf_url = f"https://openreview.net/pdf?id={paper_id}"
            
            # 判断接受状态
            decision = 'unknown'
            if venue_status:
                if any(x in venue_status.lower() for x in ['accept', 'oral', 'poster', 'spotlight']):
                    decision = 'accepted'
                elif 'reject' in venue_status.lower():
                    decision = 'rejected'
            
            # 如果只接受已接受论文，跳过拒绝的
            if accepted_only and decision == 'rejected':
                continue
            
            papers.append({
                'id': paper_id,
                'title': title,
                'abstract': abstract,
                'authors': authors,
                'keywords': keywords if isinstance(keywords, list) else [keywords] if keywords else [],
                'venue_id': venue_id,
                'venue_status': venue_status,
                'decision': decision,
                'pdf_url': pdf_url,
                'forum': paper.forum,
                'cdate': paper.cdate,
                'mdate': paper.mdate
            })
            
            # 避免请求过快
            time.sleep(0.01)
        
        return papers
        
    except Exception as e:
        print(f"  错误: {e}")
        import traceback
        traceback.print_exc()
        return []


def crawl_venues(venues: List[str], output_dir: Path, accepted_only: bool = True) -> Tuple[List[Dict], Optional[Path]]:
    """
    爬取多个会议的论文
    
    Returns:
        (所有论文列表, 汇总文件路径)
    """
    client, api_version = get_openreview_client()
    if not client:
        return [], None
    
    print(f"\n{'='*60}")
    print(f"🕷️ 阶段1: OpenReview爬虫")
    print(f"{'='*60}")
    print(f"API版本: {api_version}")
    print(f"输出目录: {output_dir}")
    print(f"只接受已接受论文: {accepted_only}")
    print(f"{'='*60}\n")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    all_papers = []
    
    for venue in venues:
        print(f"\n📚 正在爬取: {venue}")
        papers = crawl_conference(client, venue, api_version, accepted_only)
        
        if papers:
            # 提取年份
            year = 2024  # 默认
            for part in venue.split('/'):
                if part.isdigit():
                    year = int(part)
                    break
            
            # 单独保存每个会议
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            venue_name = venue.replace('/', '_').replace('.', '_')
            filename = f"{venue_name}_{year}_{timestamp}.json"
            output_path = output_dir / filename
            
            data = {
                'venue': venue,
                'year': year,
                'crawl_time': datetime.now().isoformat(),
                'total_papers': len(papers),
                'papers': papers
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print(f"  ✓ 已保存: {output_path} ({len(papers)}篇)")
            all_papers.extend(papers)
        else:
            print(f"  ⚠️ 未找到论文或爬取失败")
        
        # 会议间延迟
        time.sleep(2)
    
    # 保存汇总文件
    if all_papers:
        summary_path = output_dir / f"all_papers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump({
                'crawl_time': datetime.now().isoformat(),
                'total_papers': len(all_papers),
                'venues': venues,
                'papers': all_papers
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\n{'='*60}")
        print(f"✅ 爬取完成!")
        print(f"📊 总计: {len(all_papers)} 篇论文")
        print(f"📁 汇总文件: {summary_path}")
        print(f"{'='*60}\n")
        
        return all_papers, summary_path
    
    return [], None
