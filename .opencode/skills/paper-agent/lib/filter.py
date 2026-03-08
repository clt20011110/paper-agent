#!/usr/bin/env python3
"""
Lib: Keyword Filter
基于关键词的论文过滤器
支持同义词扩展、AND/OR/NOT逻辑
"""

import json
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass, field


# 内置同义词库
BUILTIN_SYNONYMS = {
    # # 分子/药物相关
    # 'molecular': ['molecule', 'molecules', 'chemical', 'chemistry'],
    # 'drug': ['pharmaceutical', 'medication', 'medicine', 'therapeutic'],
    # 'compound': ['chemical compound', 'small molecule'],
    # 'generation': ['generative', 'generating', 'synthesis', 'synthesizing'],
    # 'design': ['designing', 'discovery', 'screening'],
    
    # # AI/ML相关
    # 'diffusion': ['diffusion model', 'diffusion-based'],
    # 'transformer': ['attention', 'self-attention'],
    # 'graph': ['graph neural network', 'gnn', 'graph-based'],
    # 'learning': ['machine learning', 'ml', 'deep learning'],
    
    # # 生物相关
    # 'protein': ['proteins', 'peptide', 'amino acid'],
    # 'binding': ['affinity', 'docking', 'interaction'],
    # 'target': ['receptor', 'enzyme', 'protein target'],
}


@dataclass
class FilterConfig:
    """过滤配置"""
    include_groups: List[List[str]] = field(default_factory=list)  # OR of AND groups
    exclude: List[str] = field(default_factory=list)
    synonyms: Dict[str, List[str]] = field(default_factory=dict)
    match_fields: List[str] = field(default_factory=lambda: ['title', 'abstract'])
    case_sensitive: bool = False
    whole_word: bool = False
    
    def __post_init__(self):
        """初始化后合并内置同义词"""
        # 合并用户同义词和内置同义词
        merged = BUILTIN_SYNONYMS.copy()
        merged.update(self.synonyms)
        self.synonyms = merged


class KeywordFilter:
    """关键词过滤器"""
    
    def __init__(self, config: FilterConfig):
        self.config = config
        self._compile_patterns()
    
    def _compile_patterns(self):
        """预编译正则表达式模式以提高性能"""
        self.patterns = {}
        
        for word in self._get_all_keywords():
            flags = 0 if self.config.case_sensitive else re.IGNORECASE
            
            if self.config.whole_word:
                pattern = rf'\b{re.escape(word)}\b'
            else:
                pattern = rf'{re.escape(word)}'
            
            self.patterns[word] = re.compile(pattern, flags)
    
    def _get_all_keywords(self) -> Set[str]:
        """获取所有关键词（包括同义词）"""
        keywords = set()
        
        # 从include_groups收集
        for group in self.config.include_groups:
            for word in group:
                keywords.add(word.lower() if not self.config.case_sensitive else word)
                # 添加同义词
                if word in self.config.synonyms:
                    for syn in self.config.synonyms[word]:
                        keywords.add(syn.lower() if not self.config.case_sensitive else syn)
        
        # 从exclude收集
        for word in self.config.exclude:
            keywords.add(word.lower() if not self.config.case_sensitive else word)
        
        return keywords
    
    def _preprocess_text(self, text: str) -> str:
        """预处理文本"""
        if not text:
            return ""
        
        if not self.config.case_sensitive:
            text = text.lower()
        
        # 移除多余空白，但保留单词边界
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def _contains_keyword(self, text: str, keyword: str) -> bool:
        """检查文本是否包含关键词"""
        pattern = self.patterns.get(keyword)
        if not pattern:
            return False
        
        return bool(pattern.search(text))
    
    def _get_paper_text(self, paper: Dict) -> str:
        """获取论文的待匹配文本"""
        texts = []
        for field in self.config.match_fields:
            value = paper.get(field, '')
            if isinstance(value, str):
                texts.append(value)
            elif isinstance(value, list):
                texts.append(' '.join(str(v) for v in value))
        return self._preprocess_text(' '.join(texts))
    
    def filter_papers(self, papers: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        过滤论文列表
        
        Returns:
            (相关论文列表, 不相关论文列表)
        """
        relevant = []
        irrelevant = []
        
        for paper in papers:
            paper_text = self._get_paper_text(paper)
            match_info = self._check_paper(paper_text)
            
            # 添加匹配信息到论文
            paper_copy = paper.copy()
            paper_copy['relevant'] = match_info['is_relevant']
            paper_copy['match_info'] = match_info
            
            if match_info['is_relevant']:
                relevant.append(paper_copy)
            else:
                irrelevant.append(paper_copy)
        
        return relevant, irrelevant
    
    def _check_paper(self, text: str) -> Dict:
        """检查单篇论文的匹配情况"""
        match_info = {
            'is_relevant': False,
            'matched_groups': [],
            'matched_keywords': [],
            'excluded': False,
            'exclude_reason': None
        }
        
        # 首先检查排除词
        for exclude_word in self.config.exclude:
            # 检查词本身
            if self._contains_keyword(text, exclude_word):
                match_info['is_relevant'] = False
                match_info['excluded'] = True
                match_info['exclude_reason'] = f"包含排除词: {exclude_word}"
                return match_info
            
            # 检查同义词
            if exclude_word in self.config.synonyms:
                for syn in self.config.synonyms[exclude_word]:
                    if self._contains_keyword(text, syn):
                        match_info['is_relevant'] = False
                        match_info['excluded'] = True
                        match_info['exclude_reason'] = f"包含排除词同义词: {syn} (来自 {exclude_word})"
                        return match_info
        
        # 检查包含组（OR逻辑）
        for group_idx, group in enumerate(self.config.include_groups):
            group_matches = []
            all_match = True
            
            for keyword in group:
                keyword_matched = False
                matched_words = []
                
                # 检查关键词本身
                if self._contains_keyword(text, keyword):
                    keyword_matched = True
                    matched_words.append(keyword)
                
                # 检查同义词
                if keyword in self.config.synonyms:
                    for syn in self.config.synonyms[keyword]:
                        if self._contains_keyword(text, syn):
                            keyword_matched = True
                            matched_words.append(f"{syn}({keyword})")
                
                if keyword_matched:
                    group_matches.extend(matched_words)
                else:
                    all_match = False
                    break
            
            if all_match:
                match_info['is_relevant'] = True
                match_info['matched_groups'].append({
                    'group_index': group_idx,
                    'group': group,
                    'matched_keywords': group_matches
                })
                match_info['matched_keywords'].extend(group_matches)
        
        # 去重匹配的关键词
        match_info['matched_keywords'] = list(set(match_info['matched_keywords']))
        
        return match_info
    
    def get_statistics(self, relevant: List[Dict], irrelevant: List[Dict]) -> Dict:
        """获取过滤统计信息"""
        total = len(relevant) + len(irrelevant)
        
        # 统计匹配组使用情况
        group_counts = {}
        for paper in relevant:
            for group_match in paper.get('match_info', {}).get('matched_groups', []):
                group_key = tuple(group_match['group'])
                group_counts[group_key] = group_counts.get(group_key, 0) + 1
        
        # 统计排除原因
        exclude_reasons = {}
        for paper in irrelevant:
            reason = paper.get('match_info', {}).get('exclude_reason', '未匹配任何包含组')
            exclude_reasons[reason] = exclude_reasons.get(reason, 0) + 1
        
        return {
            'total_papers': total,
            'relevant_count': len(relevant),
            'irrelevant_count': len(irrelevant),
            'relevance_rate': len(relevant) / total * 100 if total > 0 else 0,
            'group_matches': {str(k): v for k, v in group_counts.items()},
            'exclusion_reasons': exclude_reasons
        }


def filter_papers_from_file(
    input_file: Path, 
    output_file: Path,
    config: FilterConfig
) -> Tuple[List[Dict], Dict]:
    """
    从文件过滤论文
    
    Returns:
        (相关论文列表, 统计信息)
    """
    print(f"\n{'='*60}")
    print(f"🔍 阶段2: 关键词过滤")
    print(f"{'='*60}")
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print(f"{'='*60}\n")
    
    # 读取论文
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    papers = data.get('papers', [])
    print(f"📚 读取到 {len(papers)} 篇论文")
    
    # 创建过滤器并执行过滤
    filter_obj = KeywordFilter(config)
    relevant, irrelevant = filter_obj.filter_papers(papers)
    
    # 获取统计
    stats = filter_obj.get_statistics(relevant, irrelevant)
    
    # 保存结果
    output_data = {
        'filter_time': datetime.now().isoformat(),
        'source_file': str(input_file),
        'config': {
            'include_groups': config.include_groups,
            'exclude': config.exclude,
            'synonyms_used': list(config.synonyms.keys()),
            'match_fields': config.match_fields
        },
        'statistics': stats,
        'total_papers': len(papers),
        'relevant_count': len(relevant),
        'irrelevant_count': len(irrelevant),
        'relevant_papers': relevant,
        'irrelevant_papers': irrelevant
    }
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    # 打印统计
    print(f"\n📊 过滤统计:")
    print(f"   总论文数: {stats['total_papers']}")
    print(f"   相关论文: {stats['relevant_count']} ({stats['relevance_rate']:.1f}%)")
    print(f"   不相关论文: {stats['irrelevant_count']}")
    
    if stats['group_matches']:
        print(f"\n🏷️  匹配组统计:")
        for group, count in stats['group_matches'].items():
            print(f"   {group}: {count}篇")
    
    if stats['exclusion_reasons']:
        print(f"\n🚫 排除统计:")
        for reason, count in list(stats['exclusion_reasons'].items())[:5]:
            print(f"   {reason}: {count}篇")
    
    print(f"\n✅ 结果已保存: {output_file}")
    
    return relevant, stats
