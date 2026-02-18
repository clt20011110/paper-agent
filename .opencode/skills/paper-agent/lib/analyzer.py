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
    
    # 使用ThreadPoolExecutor并行处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_pdf = {
            executor.submit(
                process_single_pdf, pdf_path, output_dir, api_key, model, overwrite
            ): pdf_path for pdf_path in pdf_files
        }
        
        # 显示进度
        total = len(pdf_files)
        for i, future in enumerate(future_to_pdf, 1):
            pdf_path = future_to_pdf[future]
            try:
                success, message = future.result()
                if success:
                    success_count += 1
                    print(f"[{i}/{total}] ✅ {message}")
                else:
                    fail_count += 1
                    print(f"[{i}/{total}] ❌ {message}")
                    
            except Exception as e:
                fail_count += 1
                print(f"[{i}/{total}] ❌ 任务异常 {pdf_path.name}: {e}")
    
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
