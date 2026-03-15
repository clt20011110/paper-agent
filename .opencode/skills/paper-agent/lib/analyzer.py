#!/usr/bin/env python3
"""
Lib: PDF Analyzer
使用OpenRouter API深度分析PDF论文
"""

import os
import json
import base64
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('paper_agent_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 论文分析的prompt
ANALYSIS_PROMPT = """请作为资深研究员分析这篇学术论文，从PDF中提取关键信息，按以下结构输出Markdown格式报告：

## 基本信息
- **标题**: （论文完整标题）
- **作者**: （作者列表）
- **会议/期刊**: （发表场所）
- **年份**: （发表年份）

## 研究背景
- **研究问题**: （论文要解决的核心问题）
- **现有问题**: （当前方法的局限性）
- **核心挑战**: （技术难点）

## 核心方法
- **主要技术**: （1-2句话概括主要技术）
- **关键技术组件**: （2-4个技术要点）
- **创新点**: （与现有方法的区别）
- **架构图**: （如有，描述整体流程）

## 实验结果
- **数据集**: （使用的所有数据集名称）
- **评价指标**: （主要评估指标）
- **主要结果**: （关键性能数字）
- **与SOTA对比**: （相对提升幅度）

## 主要贡献
- **理论贡献**: （新的理论/算法）
- **实践贡献**: （实际应用价值）
- **开源贡献**: （代码/数据是否开源）

## 开源信息
- **代码开源**: （是/否/未提及）
- **代码链接**: （GitHub或其他仓库URL）
- **预训练模型**: （是否提供）
- **数据集**: （是否公开可用）

请基于PDF内容准确提取信息，如果某些信息未在论文中明确提及，请标注"未提及"。不要添加额外的评价或个人观点。"""


# Add new prompt for abstract-only analysis
ABSTRACT_ANALYSIS_PROMPT = """Analyze this paper based on title and abstract only.
Note: Full PDF was not available, so analysis is limited to abstract content.

## Paper Information
- **Title**: {title}
- **Authors**: {authors}
- **Venue**: {venue}
- **Year**: {year}

## Abstract
{abstract}

## Analysis Structure

### Research Problem
What is the main research question or problem addressed?

### Proposed Approach
What is the high-level approach or methodology?

### Key Claims
What are the main claims or contributions (as stated in abstract)?

### Limitations of this Analysis
- Analysis based on abstract only (typically 150-250 words)
- Full methodology, experiments, and results not available
- Detailed technical contributions may be missing

**Confidence Level**: abstract_only (limited)
"""


def encode_pdf_to_base64(pdf_path: str) -> str:
    """将PDF文件编码为base64字符串"""
    with open(pdf_path, "rb") as pdf_file:
        return base64.b64encode(pdf_file.read()).decode('utf-8')


def analyze_pdf(
    pdf_path: str,
    api_key: str,
    model: str = "stepfun/step-3.5-flash:free",
    max_retries: int = 3,
    timeout: int = 120
) -> Optional[str]:
    """分析单篇PDF论文"""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        # 编码PDF
        base64_pdf = encode_pdf_to_base64(pdf_path)
        data_url = f"data:application/pdf;base64,{base64_pdf}"
        
        # 构建消息
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": ANALYSIS_PROMPT},
                    {
                        "type": "file",
                        "file": {
                            "filename": os.path.basename(pdf_path),
                            "file_data": data_url
                        }
                    },
                ]
            }
        ]
        
        # 插件配置
        plugins = [{"id": "file-parser", "pdf": {"engine": "pdf-text"}}]
        
        payload = {
            "model": model,
            "messages": messages,
            "plugins": plugins,
            "stream": False
        }
        
        # 重试机制
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    url, headers=headers, json=payload, timeout=timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if 'choices' in result and len(result['choices']) > 0:
                        return result['choices'][0]['message']['content']
                    else:
                        logger.warning(f"API返回格式异常: {result}")
                else:
                    logger.warning(f"API请求失败 (尝试 {attempt + 1}/{max_retries}): HTTP {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"请求异常 (尝试 {attempt + 1}/{max_retries}): {e}")
            
            # 重试前等待
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # 指数退避
        
        logger.error(f"分析失败，已达到最大重试次数: {pdf_path}")
        return None
        
    except Exception as e:
        logger.error(f"处理PDF时发生错误 {pdf_path}: {e}")
        return None


def process_single_pdf(
    pdf_path: Path,
    output_dir: Path,
    api_key: str,
    model: str,
    overwrite: bool = False
) -> Tuple[bool, str]:
    """处理单篇PDF并保存结果"""
    try:
        # 生成输出文件名
        output_file = output_dir / f"{pdf_path.stem}_analysis.md"
        
        # 检查是否已存在
        if not overwrite and output_file.exists():
            return True, f"跳过（已存在）: {pdf_path.name}"
        
        # 分析PDF
        analysis = analyze_pdf(str(pdf_path), api_key, model)
        
        if analysis is None:
            return False, f"分析失败: {pdf_path.name}"
        
        # 保存结果
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(analysis)
        
        return True, f"完成: {pdf_path.name}"
        
    except Exception as e:
        logger.error(f"处理文件失败 {pdf_path}: {e}")
        return False, f"处理失败: {pdf_path.name} - {e}"


def find_pdf_files(input_dir: Path) -> List[Path]:
    """查找所有PDF文件"""
    return sorted(input_dir.rglob("*.pdf"))


def analyze_papers(
    input_dir: Path,
    output_dir: Path,
    api_key: str,
    model: str = "stepfun/step-3.5-flash:free",
    max_workers: int = 4,
    overwrite: bool = False
) -> Dict:
    """
    批量分析PDF文件
    
    Returns:
        统计信息字典
    """
    print(f"\n{'='*60}")
    print(f"🔬 阶段4: PDF深度分析")
    print(f"{'='*60}")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"并行数: {max_workers}")
    print(f"模型: {model}")
    print(f"{'='*60}\n")
    
    # 验证输入目录
    if not input_dir.exists():
        logger.error(f"输入目录不存在: {input_dir}")
        return {'success': 0, 'failed': 0, 'total': 0}
    
    # 查找所有PDF
    pdf_files = find_pdf_files(input_dir)
    
    if not pdf_files:
        logger.warning(f"在 {input_dir} 中未找到PDF文件")
        return {'success': 0, 'failed': 0, 'total': 0}
    
    logger.info(f"找到 {len(pdf_files)} 篇PDF论文待分析")
    
    # 统计
    success_count = 0
    fail_count = 0
    total = len(pdf_files)
    for i, pdf_path in enumerate(pdf_files[25:], 26):
        success,message=process_single_pdf(pdf_path, output_dir, api_key, model, overwrite)
        if success:
            success_count += 1
            print(f"[{i}/{total}] ✅ {message}")
        else:
            fail_count += 1
            print(f"[{i}/{total}] ❌ {message}")

    
    # 使用ThreadPoolExecutor并行处理
    # with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务

        # future_to_pdf = {
        #     executor.submit(
        #         process_single_pdf, pdf_path, output_dir, api_key, model, overwrite
        #     ): pdf_path for pdf_path in pdf_files
        # }
        
        # 显示进度
        # total = len(pdf_files)
        # for i, future in enumerate(future_to_pdf, 1):
        #     pdf_path = future_to_pdf[future]
        #     try:
        #         success, message = future.result()
        #         if success:
        #             success_count += 1
        #             print(f"[{i}/{total}] ✅ {message}")
        #         else:
        #             fail_count += 1
        #             print(f"[{i}/{total}] ❌ {message}")
                    
        #     except Exception as e:
        #         fail_count += 1
        #         print(f"[{i}/{total}] ❌ 任务异常 {pdf_path.name}: {e}")
    
    # 输出统计
    print(f"\n{'='*60}")
    print(f"✅ 分析完成！")
    print(f"📊 统计:")
    print(f"   总论文数: {len(pdf_files)}")
    print(f"   成功: {success_count}")
    print(f"   失败: {fail_count}")
    print(f"📁 输出目录: {output_dir}")
    print(f"{'='*60}\n")
    
    if fail_count > 0:
        print(f"查看错误日志: paper_agent_analysis.log")
    
    return {
        'total': len(pdf_files),
        'success': success_count,
        'failed': fail_count
    }


class EnhancedAnalyzer:
    """Enhanced analyzer with PDF fallback and summary generation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_paper(
        self,
        paper: Dict,
        pdf_path: Optional[Path] = None,
        api_key: Optional[str] = None,
        model: str = "stepfun/step-3.5-flash:free"
    ) -> Dict:
        """
        Analyze paper with automatic PDF fallback.
        
        Args:
            paper: Paper dictionary with title, abstract, etc.
            pdf_path: Path to PDF file (optional)
            api_key: OpenRouter API key
            model: Model to use for analysis
            
        Returns:
            Analysis result dictionary with 'confidence' field
        """
        if pdf_path and Path(pdf_path).exists():
            result = self._analyze_pdf(pdf_path, api_key, model)
            result['confidence'] = 'full_pdf'
        else:
            result = self._analyze_abstract(paper, api_key, model)
            result['confidence'] = 'abstract_only'
        
        return result
    
    def _analyze_pdf(self, pdf_path: Path, api_key: Optional[str], model: str) -> Dict:
        """Analyze using full PDF content"""
        analysis = analyze_pdf(str(pdf_path), api_key or '', model)
        
        return {
            'analysis': analysis,
            'pdf_path': str(pdf_path),
            'method': 'full_pdf'
        }
    
    def _analyze_abstract(self, paper: Dict, api_key: Optional[str], model: str) -> Dict:
        """Analyze using only title and abstract"""
        prompt = ABSTRACT_ANALYSIS_PROMPT.format(
            title=paper.get('title', ''),
            authors=', '.join(paper.get('authors', [])),
            venue=paper.get('venue', ''),
            year=paper.get('year', ''),
            abstract=paper.get('abstract', '')
        )
        
        response = self._call_api(prompt, api_key, model)
        
        return {
            'analysis': response,
            'paper_id': paper.get('id'),
            'title': paper.get('title'),
            'method': 'abstract_only'
        }
    
    def _call_api(self, prompt: str, api_key: Optional[str], model: str, max_retries: int = 3, timeout: int = 120) -> Optional[str]:
        """Call OpenRouter API with text prompt"""
        if not api_key:
            self.logger.error("API key is required")
            return None
        
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": False
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    url, headers=headers, json=payload, timeout=timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if 'choices' in result and len(result['choices']) > 0:
                        return result['choices'][0]['message']['content']
                    else:
                        self.logger.warning(f"API returned unexpected format: {result}")
                else:
                    self.logger.warning(f"API request failed (attempt {attempt + 1}/{max_retries}): HTTP {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Request exception (attempt {attempt + 1}/{max_retries}): {e}")
            
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
        
        self.logger.error("Analysis failed after max retries")
        return None


class ResearchSummaryGenerator:
    """Generate comprehensive research summary from multiple analyses"""
    
    SUMMARY_PROMPT = """Based on the following paper analyses, generate a comprehensive research summary report.

## Research Topic: {topic}

## Number of Papers Analyzed: {paper_count}

## Individual Paper Summaries:
{paper_summaries}

---

Please generate a research summary report with the following sections:

# Research Summary Report

## 1. Overview and Trends
- What are the main research directions in this topic?
- How has the field evolved based on the papers analyzed?

## 2. Methodological Approaches
- What are the dominant methods/techniques?
- Categorize papers by their methodological approach

## 3. Key Findings and Contributions
- What are the most significant contributions?
- Any consensus or disagreements in the field?

## 4. Datasets and Benchmarks
- What datasets are commonly used?
- What evaluation metrics are standard?

## 5. Open Source Resources
- List available code repositories
- List public datasets mentioned

## 6. Research Gaps and Future Directions
- What problems remain unsolved?
- What are promising future directions?

Format as Markdown with clear headings and bullet points.
"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def generate_summary(
        self,
        analyses: List[Dict],
        topic: str,
        api_key: str,
        model: str = "anthropic/claude-3.5-sonnet"
    ) -> str:
        """
        Generate research summary from paper analyses.
        
        Args:
            analyses: List of analysis results from EnhancedAnalyzer
            topic: Research topic description
            api_key: OpenRouter API key
            model: High-capability model for summary generation
            
        Returns:
            Markdown formatted summary report
        """
        max_papers = 50
        if len(analyses) > max_papers:
            self.logger.info(f"Truncating {len(analyses)} analyses to {max_papers}")
            analyses = analyses[:max_papers]
        
        paper_summaries = []
        for i, analysis in enumerate(analyses, 1):
            summary = f"### Paper {i}: {analysis.get('title', 'Unknown')}\n"
            analysis_text = analysis.get('analysis', '')
            summary += analysis_text[:500] if analysis_text else 'No analysis available'
            summary += "\n\n"
            paper_summaries.append(summary)
        
        prompt = self.SUMMARY_PROMPT.format(
            topic=topic,
            paper_count=len(analyses),
            paper_summaries='\n'.join(paper_summaries)
        )
        
        return self._call_api(prompt, api_key, model)
    
    def _call_api(self, prompt: str, api_key: str, model: str, max_retries: int = 3, timeout: int = 180) -> str:
        """Call OpenRouter API with text prompt"""
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": False
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    url, headers=headers, json=payload, timeout=timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if 'choices' in result and len(result['choices']) > 0:
                        return result['choices'][0]['message']['content']
                    else:
                        self.logger.warning(f"API returned unexpected format: {result}")
                else:
                    self.logger.warning(f"API request failed (attempt {attempt + 1}/{max_retries}): HTTP {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Request exception (attempt {attempt + 1}/{max_retries}): {e}")
            
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
        
        self.logger.error("Summary generation failed after max retries")
        return "# Error: Summary generation failed\n\nPlease check API key and try again."


def generate_research_summary(
    analysis_dir: Path,
    output_file: Path,
    topic: str,
    api_key: str,
    model: str = "anthropic/claude-3.5-sonnet"
) -> bool:
    """
    Generate research summary from analysis results.
    
    Args:
        analysis_dir: Directory containing analysis markdown files
        output_file: Output file path for summary
        topic: Research topic description
        api_key: OpenRouter API key
        model: Model for summary generation
        
    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"📊 阶段4B: 研究总结生成")
    print(f"{'='*60}")
    print(f"输入目录: {analysis_dir}")
    print(f"输出文件: {output_file}")
    print(f"主题: {topic}")
    print(f"模型: {model}")
    print(f"{'='*60}\n")
    
    if not analysis_dir.exists():
        print(f"ERROR: 分析目录不存在: {analysis_dir}")
        return False
    
    analysis_files = sorted(analysis_dir.glob("*.md"))
    
    if not analysis_files:
        print(f"ERROR: 未找到分析文件")
        return False
    
    print(f"找到 {len(analysis_files)} 个分析文件")
    
    analyses = []
    for md_file in analysis_files:
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        analyses.append({
            'title': md_file.stem.replace('_analysis', ''),
            'analysis': content
        })
    
    generator = ResearchSummaryGenerator()
    summary = generator.generate_summary(
        analyses=analyses,
        topic=topic,
        api_key=api_key,
        model=model
    )
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(f"\n✅ 研究总结已生成: {output_file}")
    return True
