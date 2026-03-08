#!/usr/bin/env python3
"""
Lib: PDF Downloader
基于过滤结果下载PDF文件
支持OpenReview和arXiv双通道
"""

import json
import time
import requests
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime


class PDFDownloader:
    """PDF下载器"""
    
    def __init__(self, output_dir: Path, timeout: int = 60):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'skipped': 0
        }
    
    def download(self, paper: Dict, paper_idx: int, total: int) -> Optional[Path]:
        """下载单篇论文的PDF"""
        paper_id = paper.get('id', 'unknown')
        title = paper.get('title', 'Unknown')
        pdf_url = paper.get('pdf_url')
        
        print(f"\n[{paper_idx}/{total}] {title[:70]}...")
        self.stats['total'] += 1
        
        # 生成文件名
        year = paper.get('venue_id', 2024).split('/')[1] if 'venue_id' in paper else '2024'
        authors = paper.get('authors', [])
        first_author = authors[0].split()[-1] if authors else 'Unknown'
        safe_title = self._sanitize_filename(title)[:50]
        filename = f"{year}_{first_author}_{safe_title}.pdf"
        
        output_path = self.output_dir / filename
        
        # 检查是否已存在
        if output_path.exists():
            print(f"  ⏭️  已存在，跳过")
            self.stats['skipped'] += 1
            return output_path
        
        # 尝试下载
        success = False
        
        # 1. 尝试OpenReview PDF
        if pdf_url and not success:
            print(f"  📥 尝试OpenReview下载...")
            success = self._try_download(pdf_url, output_path)
            if success:
                print(f"  ✓ OpenReview下载成功")
        
        # 2. 尝试arXiv搜索
        if not success:
            print(f"  🔍 尝试搜索arXiv...")
            arxiv_url = self._search_arxiv(title)
            if arxiv_url:
                success = self._try_download(arxiv_url, output_path)
                if success:
                    print(f"  ✓ arXiv下载成功")
        
        if success:
            self.stats['success'] += 1
            return output_path
        else:
            print(f"  ✗ 所有下载方式均失败")
            self.stats['failed'] += 1
            return None
    
    def _try_download(self, url: str, output_path: Path) -> bool:
        """尝试下载URL到指定路径"""
        try:
            response = self.session.get(url, timeout=self.timeout, stream=True)
            response.raise_for_status()
            
            # 检查内容类型
            content_type = response.headers.get('content-type', '')
            if 'pdf' not in content_type.lower() and not url.endswith('.pdf'):
                # 可能是HTML页面，检查内容
                content = response.content[:100]
                if b'%PDF' not in content:
                    return False
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            return True
            
        except Exception as e:
            return False
    
    def _search_arxiv(self, title: str) -> Optional[str]:
        """根据标题搜索arXiv"""
        try:
            import feedparser
            
            # 简化标题用于搜索
            search_title = title.replace(' ', '+')[:100]
            url = f"http://export.arxiv.org/api/query?search_query=ti:{search_title}&max_results=1"
            
            response = requests.get(url, timeout=10)
            feed = feedparser.parse(response.content)
            
            if feed.entries:
                entry = feed.entries[0]
                # 检查标题相似度
                arxiv_title = str(entry.title).lower()
                query_title = title.lower()
                
                # 简单相似度检查
                if self._title_similarity(arxiv_title, query_title) > 0.6:
                    for link in entry.links:
                        if link.get('title') == 'pdf':
                            return str(link.href)
            
            return None
            
        except Exception as e:
            return None
    
    def _title_similarity(self, title1: str, title2: str) -> float:
        """计算标题相似度"""
        words1 = set(title1.lower().split())
        words2 = set(title2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0
    
    def _sanitize_filename(self, filename: str) -> str:
        """清理文件名"""
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        return filename.strip()
    
    def get_stats(self) -> Dict:
        """获取下载统计"""
        return self.stats.copy()


def generate_bibtex(papers: List[Dict]) -> str:
    """生成BibTeX引用"""
    bibtex_entries = []
    
    for paper in papers:
        title = paper.get('title', '')
        authors = paper.get('authors', [])
        year = paper.get('year', 2024)
        venue = paper.get('venue_id', '')
        paper_id = paper.get('id', '')
        
        # 生成引用键
        first_author = authors[0].split()[-1] if authors else 'Unknown'
        cite_key = f"{first_author.lower()}{year}{title.split()[0].lower()}"
        cite_key = ''.join(c for c in cite_key if c.isalnum())
        
        # 格式化作者
        author_str = ' and '.join(authors) if authors else 'Unknown'
        
        # 提取会议名称
        venue_name = venue.split('/')[0] if '/' in venue else venue
        
        entry = f"""@inproceedings{{{cite_key},
  title = {{{title}}},
  author = {{{author_str}}},
  booktitle = {{{venue_name}}},
  year = {{{year}}},
  url = {{https://openreview.net/forum?id={paper_id}}}
}}"""
        
        bibtex_entries.append(entry)
    
    return '\n\n'.join(bibtex_entries)


def download_papers_from_file(
    input_file: Path,
    output_dir: Path,
    delay: float = 1.0
) -> Tuple[List[Dict], Path]:
    """
    从文件下载论文PDF
    
    Returns:
        (已下载论文列表, 输出目录)
    """
    print(f"\n{'='*60}")
    print(f"📥 阶段3: PDF下载")
    print(f"{'='*60}")
    print(f"输入文件: {input_file}")
    print(f"输出目录: {output_dir}")
    print(f"下载延迟: {delay}秒")
    print(f"{'='*60}\n")
    
    # 读取输入文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 获取论文列表
    if 'relevant_papers' in data:
        papers = data['relevant_papers']
    elif 'papers' in data:
        papers = data['papers']
    else:
        papers = []
    
    # 过滤出需要下载的论文
    papers_to_download = [p for p in papers if p.get('relevant', True)]
    
    print(f"📚 找到 {len(papers_to_download)} 篇相关论文")
    
    if not papers_to_download:
        print("⚠️ 没有需要下载的论文")
        return [], output_dir
    
    # 创建下载器
    output_dir.mkdir(parents=True, exist_ok=True)
    downloader = PDFDownloader(output_dir)
    
    # 下载论文
    downloaded_papers = []
    for idx, paper in enumerate(papers_to_download, 1):
        pdf_path = downloader.download(paper, idx, len(papers_to_download))
        
        if pdf_path:
            paper['local_pdf'] = str(pdf_path)
            downloaded_papers.append(paper)
        
        # 延迟
        if idx < len(papers_to_download):
            time.sleep(delay)
    
    # 生成BibTeX
    print(f"\n📝 生成BibTeX...")
    bibtex_content = generate_bibtex(downloaded_papers)
    bibtex_path = output_dir / "references.bib"
    with open(bibtex_path, 'w', encoding='utf-8') as f:
        f.write(bibtex_content)
    
    # 保存下载记录
    record_path = output_dir / "download_record.json"
    with open(record_path, 'w', encoding='utf-8') as f:
        json.dump({
            'download_time': datetime.now().isoformat(),
            'source_file': str(input_file),
            'total_papers': len(papers_to_download),
            'downloaded': len(downloaded_papers),
            'stats': downloader.get_stats(),
            'papers': downloaded_papers
        }, f, ensure_ascii=False, indent=2)
    
    # 打印统计
    stats = downloader.get_stats()
    print(f"\n{'='*60}")
    print(f"✅ 下载完成!")
    print(f"📊 统计:")
    print(f"   总计: {stats['total']}")
    print(f"   成功: {stats['success']}")
    print(f"   失败: {stats['failed']}")
    print(f"   跳过: {stats['skipped']}")
    print(f"📁 输出目录: {output_dir}")
    print(f"📚 BibTeX: {bibtex_path}")
    print(f"{'='*60}\n")
    
    return downloaded_papers, output_dir
