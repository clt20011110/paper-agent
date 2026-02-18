#!/usr/bin/env python3
"""
Paper Agent - 非交互式命令行工具
由OpenCode Agent负责交互，本脚本仅执行操作

Usage:
    # 环境检查
    python paper_agent.py check
    
    # 阶段1: 爬取
    python paper_agent.py stage1 --config config.yaml
    
    # 阶段2: 关键词过滤
    python paper_agent.py stage2 --input all_papers.json --config config.yaml
    
    # 阶段3: 下载PDF
    python paper_agent.py stage3 --input filtered_papers.json --output ./papers
    
    # 阶段4: 分析PDF
    python paper_agent.py stage4 --input ./papers --output ./analysis --api-key xxx
    
    # 全流程
    python paper_agent.py all --config config.yaml --api-key xxx
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

# 添加lib目录到路径
script_dir = Path(__file__).parent
lib_dir = script_dir.parent / 'lib'
sys.path.insert(0, str(lib_dir))

try:
    from crawler import crawl_venues
    from filter import FilterConfig, filter_papers_from_file
    from downloader import download_papers_from_file
    from analyzer import analyze_papers
except ImportError as e:
    print(f"ERROR:IMPORT_FAILED:{e}")
    sys.exit(1)


def check_environment() -> Dict:
    """检查运行环境，返回机器可解析的结果"""
    results = {
        'python_ok': sys.version_info >= (3, 7),
        'missing_packages': [],
        'api_configured': False,
        'ready': False
    }
    
    # 检查依赖包
    required = ['openreview', 'requests', 'feedparser', 'yaml']
    for module in required:
        try:
            __import__(module)
        except ImportError:
            results['missing_packages'].append(module)
    
    # 检查API密钥
    api_key = os.environ.get('OPENROUTER_API_KEY')
    if api_key:
        results['api_configured'] = True
        results['api_key'] = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
    
    # 判断是否就绪
    results['ready'] = len(results['missing_packages']) == 0
    
    return results


def load_config(config_path: Path) -> Dict:
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    # 重建FilterConfig
    filter_dict = config_dict.get('filter', {})
    filter_config = FilterConfig(
        include_groups=filter_dict.get('include_groups', []),
        exclude=filter_dict.get('exclude', []),
        match_fields=filter_dict.get('match_fields', ['title', 'abstract']),
        case_sensitive=filter_dict.get('case_sensitive', False),
        whole_word=filter_dict.get('whole_word', False)
    )
    
    config = {
        'output_dir': config_dict.get('output_dir', './paper_research'),
        'conferences': config_dict.get('conferences', ['ICLR']),
        'years': config_dict.get('years', [2024]),
        'topic': config_dict.get('topic', 'Research'),
        'keywords': config_dict.get('keywords', ''),
        'filter_config': filter_config
    }
    
    options = config_dict.get('options', {})
    config['workers'] = options.get('workers', 4)
    config['delay'] = options.get('delay', 1.0)
    config['accepted_only'] = options.get('accepted_only', True)
    
    return config


def save_config(config: Dict, filepath: Path):
    """保存配置到YAML"""
    config_dict = {
        'output_dir': config['output_dir'],
        'conferences': config['conferences'],
        'years': config['years'],
        'topic': config['topic'],
        'keywords': config['keywords'],
        'filter': {
            'include_groups': config['filter_config'].include_groups,
            'exclude': config['filter_config'].exclude,
            'match_fields': config['filter_config'].match_fields,
            'case_sensitive': config['filter_config'].case_sensitive,
            'whole_word': config['filter_config'].whole_word
        },
        'options': {
            'workers': config.get('workers', 4),
            'delay': config.get('delay', 1.0),
            'accepted_only': config.get('accepted_only', True)
        }
    }
    
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        yaml.dump(config_dict, f, allow_unicode=True, default_flow_style=False)


def run_stage1(config: Dict) -> Optional[Path]:
    """运行阶段1: 爬取论文"""
    conference_dict = {'ICLR':'ICLR.cc',
                        'NeurIPS':'NeurIPS.cc',
                        'ICML':'ICML.cc',
                        'AAAI':'AAAI.org'}
    venues = []
    for conf in config['conferences']:
        for year in config['years']:
            venues.append(f"{conference_dict[conf]}/{year}/Conference")
    
    output_dir = Path(config['output_dir']) / 'data'
    papers, summary_path = crawl_venues(
        venues=venues,
        output_dir=output_dir,
        accepted_only=config.get('accepted_only', True)
    )
    
    return summary_path


def run_stage2(config: Dict, input_file: Path, output_file: Optional[Path] = None) -> Optional[Path]:
    """运行阶段2: 关键词过滤"""
    if output_file is None:
        output_dir = Path(config['output_dir']) / 'data'
        output_file = output_dir / f"filtered_papers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    else:
        output_file = Path(output_file)
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    relevant, stats = filter_papers_from_file(
        input_file=input_file,
        output_file=output_file,
        config=config['filter_config']
    )
    
    # 输出机器可解析的统计信息
    print(f"\nSTAGE2_STATS:total={stats['total_papers']},relevant={stats['relevant_count']},rate={stats['relevance_rate']:.2f}")
    
    if stats['relevant_count'] == 0:
        print("ERROR:NO_PAPERS_MATCHED")
        return None
    
    # 输出匹配信息供OpenCode查看
    print(f"\nMATCHING_SUMMARY:")
    for i, paper in enumerate(relevant[:10], 1):
        match_info = paper.get('match_info', {})
        matched = ', '.join(match_info.get('matched_keywords', [])[:3])
        print(f"  {i}. {paper['title'][:70]}...")
        print(f"     Match: {matched}")
    if len(relevant) > 10:
        print(f"     ... and {len(relevant) - 10} more")
    
    return output_file


def run_stage3(config: Dict, input_file: Path, output_dir: Optional[Path] = None) -> Optional[Path]:
    """运行阶段3: 下载PDF"""
    if output_dir is None:
        output_dir = Path(config['output_dir']) / 'papers'
    
    papers, papers_dir = download_papers_from_file(
        input_file=input_file,
        output_dir=output_dir,
        delay=config.get('delay', 1.0)
    )
    
    return papers_dir


def run_stage4(config: Dict, input_dir: Path, output_dir: Optional[Path] = None, api_key: Optional[str] = None) -> bool:
    """运行阶段4: 分析论文"""
    if api_key is None:
        api_key = os.environ.get('OPENROUTER_API_KEY')
    
    if not api_key:
        print("ERROR:NO_API_KEY")
        return False
    
    if output_dir is None:
        output_dir = Path(config['output_dir']) / 'analysis'
    
    stats = analyze_papers(
        input_dir=Path(input_dir),
        output_dir=output_dir,
        api_key=api_key,
        max_workers=config.get('workers', 4)
    )
    
    print(f"\nSTAGE4_STATS:total={stats['total']},success={stats['success']},failed={stats['failed']}")
    
    return stats['success'] > 0


def main():
    parser = argparse.ArgumentParser(
        description='Paper Agent - 非交互式论文搜索分析工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
命令:
  check       检查运行环境，输出JSON格式结果
  stage1      阶段1: 爬取论文
  stage2      阶段2: 关键词过滤
  stage3      阶段3: 下载PDF
  stage4      阶段4: 深度分析
  all         运行所有阶段

输出格式:
  check命令输出JSON格式:
    {"ready": true, "missing_packages": [], "api_configured": true}
  
  stage2输出统计:
    STAGE2_STATS:total=100,relevant=23,rate=23.00
    
  stage4输出统计:
    STAGE4_STATS:total=23,success=20,failed=3

示例:
  python paper_agent.py check
  python paper_agent.py stage1 --config config.yaml
  python paper_agent.py stage2 --input data/all_papers.json --config config.yaml
  python paper_agent.py all --config config.yaml --api-key xxx
        """
    )
    
    parser.add_argument('command', choices=['check', 'stage1', 'stage2', 'stage3', 'stage4', 'all'],
                       help='要执行的命令')
    parser.add_argument('--config', '-c', type=str,
                       help='配置文件路径 (YAML格式)')
    parser.add_argument('--input', '-i', type=str,
                       help='输入文件/目录路径')
    parser.add_argument('--output', '-o', type=str,
                       help='输出文件/目录路径')
    parser.add_argument('--api-key', type=str,
                       help='OpenRouter API密钥（覆盖环境变量）')
    
    args = parser.parse_args()
    
    # 环境检查 - 输出JSON
    if args.command == 'check':
        import json
        results = check_environment()
        print(json.dumps(results))
        sys.exit(0 if results['ready'] else 1)
    
    # 加载配置
    if not args.config:
        print("ERROR:CONFIG_REQUIRED")
        sys.exit(1)
    
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR:CONFIG_NOT_FOUND:{config_path}")
        sys.exit(1)
    
    config = load_config(config_path)
    
    # 执行命令
    try:
        if args.command == 'stage1':
            result = run_stage1(config)
            if result:
                print(f"\nOUTPUT:{result}")
            else:
                print("ERROR:STAGE1_FAILED")
                sys.exit(1)
        
        elif args.command == 'stage2':
            if not args.input:
                print("ERROR:INPUT_REQUIRED")
                sys.exit(1)
            
            result = run_stage2(config, Path(args.input), Path(args.output) if args.output else None)
            if result:
                print(f"\nOUTPUT:{result}")
            else:
                sys.exit(1)
        
        elif args.command == 'stage3':
            if not args.input:
                print("ERROR:INPUT_REQUIRED")
                sys.exit(1)
            
            result = run_stage3(config, Path(args.input), Path(args.output) if args.output else None)
            if result:
                print(f"\nOUTPUT:{result}")
            else:
                print("ERROR:STAGE3_FAILED")
                sys.exit(1)
        
        elif args.command == 'stage4':
            if not args.input:
                print("ERROR:INPUT_REQUIRED")
                sys.exit(1)
            
            success = run_stage4(config, Path(args.input), Path(args.output) if args.output else None, args.api_key)
            if not success:
                print("ERROR:STAGE4_FAILED")
                sys.exit(1)
        
        elif args.command == 'all':
            # 阶段1
            print("[STAGE1] 开始爬取...")
            stage1_result = run_stage1(config)
            if not stage1_result:
                print("ERROR:STAGE1_FAILED")
                sys.exit(1)
            print(f"OUTPUT:{stage1_result}")
            
            # 阶段2
            print("\n[STAGE2] 开始过滤...")
            stage2_result = run_stage2(config, stage1_result)
            if not stage2_result:
                sys.exit(1)
            print(f"OUTPUT:{stage2_result}")
            
            # 阶段3
            print("\n[STAGE3] 开始下载...")
            stage3_result = run_stage3(config, stage2_result)
            if not stage3_result:
                print("ERROR:STAGE3_FAILED")
                sys.exit(1)
            print(f"OUTPUT:{stage3_result}")
            
            # 阶段4
            print("\n[STAGE4] 开始分析...")
            success = run_stage4(config, stage3_result, api_key=args.api_key)
            if not success:
                print("ERROR:STAGE4_FAILED")
                sys.exit(1)
            
            print("\n[COMPLETE] 全流程执行完成")
    
    except Exception as e:
        print(f"ERROR:EXCEPTION:{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
