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
import json
import re
from pathlib import Path
from typing import Dict, Optional, List, Any, Tuple
from datetime import datetime

# 添加lib目录到路径
script_dir = Path(__file__).parent
lib_dir = script_dir.parent / 'lib'
sys.path.insert(0, str(lib_dir))

try:
    from crawler import crawl_venues
    from filter import FilterConfig, filter_papers_from_file
    from downloader import download_papers_from_file
    from analyzer import analyze_papers, generate_research_summary
    from database import DatabaseManager, Paper
    from adapters import AdapterRegistry, VenueConfig
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

    # Support both legacy and new filter schema.
    raw_filter = config_dict.get('filter', {})
    filter_dict = raw_filter.get('regex', raw_filter) if isinstance(raw_filter, dict) else {}
    filter_config = FilterConfig(
        include_groups=filter_dict.get('include_groups', []),
        exclude=filter_dict.get('exclude', []),
        match_fields=filter_dict.get('match_fields', ['title', 'abstract']),
        case_sensitive=filter_dict.get('case_sensitive', False),
        whole_word=filter_dict.get('whole_word', False)
    )
    
    output_dir = config_dict.get('output_dir', './paper_research')
    
    config = {
        'output_dir': output_dir,
        'conferences': config_dict.get('conferences', ['ICLR']),
        'years': config_dict.get('years', [2024]),
        'topic': config_dict.get('topic', 'Research'),
        'keywords': config_dict.get('keywords', ''),
        'filter_config': filter_config,
        'sources': config_dict.get('sources', {}),
        'venues': config_dict.get('venues', {}),
    }
    
    options = config_dict.get('options', {})
    config['workers'] = options.get('workers', 4)
    config['delay'] = options.get('delay', 1.0)
    config['accepted_only'] = options.get('accepted_only', True)
    
    # Parse database configuration (optional)
    database_config = config_dict.get('database', {})
    if database_config:
        config['database'] = {
            'format': database_config.get('format', 'json'),
            'path': database_config.get('path', f"{output_dir}/papers.json"),
            'incremental': database_config.get('incremental', True),
            'backup': database_config.get('backup', True)
        }
    else:
        config['database'] = None

    # Ensure new-style source config has stable defaults.
    sources = config.get('sources', {}) if isinstance(config.get('sources', {}), dict) else {}
    sources.setdefault('conferences', [])
    sources.setdefault('journals', [])
    sources.setdefault('arxiv', {})
    config['sources'] = sources
    
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


def _normalize_title(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\\s]+", " ", text)
    return " ".join(text.split())


def _title_similarity(t1: str, t2: str) -> float:
    a = set(_normalize_title(t1).split())
    b = set(_normalize_title(t2).split())
    if not a or not b:
        return 0.0
    inter = a & b
    union = a | b
    return len(inter) / len(union)


def _parse_date_range(value: Any) -> Optional[Tuple[str, str]]:
    if not value:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return str(value[0]), str(value[1])
    if isinstance(value, str):
        parts = [p.strip() for p in value.split("to")]
        if len(parts) == 2 and parts[0] and parts[1]:
            return parts[0], parts[1]
    return None


def _default_platform_for_conference(name: str) -> str:
    n = name.upper()
    if n in {"ICLR", "NEURIPS", "ICML"}:
        return "openreview"
    if n == "AAAI":
        return "aaai"
    if n == "ACL":
        return "acl"
    if n == "CVPR":
        return "cvpr"
    if n == "ICCV":
        return "iccv"
    if n == "IJCAI":
        return "ijcai"
    if n == "DAC":
        return "dblp_dac"
    if n == "ICCAD":
        return "dblp_iccad"
    if n == "TCAD":
        return "dblp_tcad"
    return "openreview"


def _default_platform_for_journal(name: str) -> str:
    mapping = {
        "nature machine intelligence": "nature_machine_intelligence_xref",
        "nature chemistry": "nature_chemistry_xref",
        "nature computer science": "nature_computer_science",
        "nature communications": "nature_communications_xref",
        "nature catalysis": "nature_catalysis",
        "nature biotechnology": "nature_biotechnology",
        "nature biomedical engineering": "nature_biomedical_engineering",
        "cell": "cell",
        "science": "science",
    }
    return mapping.get(name.strip().lower(), name.strip().lower().replace(" ", "_"))


def _resolve_platform(
    venue_name: str,
    declared_platform: Optional[str],
    venue_type: str,
    additional_params: Dict[str, Any],
) -> str:
    if declared_platform:
        p = declared_platform.strip().lower()

        # AAAI recent proceedings are more reliable from official OJS pages.
        if venue_type == "conference" and venue_name.upper() == "AAAI" and p == "openreview":
            if AdapterRegistry.supports_platform("aaai"):
                return "aaai"

        # Direct adapter platform name.
        if AdapterRegistry.supports_platform(p):
            return p

        # Convenience aliases for EDA platforms.
        upper_name = venue_name.upper()
        if venue_type == "conference" and p == "ieee":
            if upper_name in {"DAC", "ICCAD", "TCAD"}:
                return upper_name.lower()
        if venue_type == "conference" and p == "dblp":
            if upper_name in {"DAC", "ICCAD", "TCAD"}:
                return f"dblp_{upper_name.lower()}"

        # "nature" can use paid Springer adapters (if key exists), otherwise
        # fallback to Crossref-backed adapters.
        if venue_type == "journal" and p == "nature":
            has_key = bool(additional_params.get("api_key") or additional_params.get("nature_api_key"))
            lower_name = venue_name.strip().lower()
            if has_key:
                native = {
                    "nature machine intelligence": "nature_machine_intelligence",
                    "nature chemistry": "nature_chemistry",
                    "nature communications": "nature_communications",
                    "nature": "nature_main",
                }.get(lower_name)
                if native and AdapterRegistry.supports_platform(native):
                    return native
            return _default_platform_for_journal(venue_name)

    if venue_type == "conference":
        return _default_platform_for_conference(venue_name)
    return _default_platform_for_journal(venue_name)


def _legacy_sources_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    sources = {"conferences": [], "journals": [], "arxiv": {}}
    venue_overrides = config.get("venues", {}) or {}
    years = config.get("years", [datetime.now().year])

    for conf in config.get("conferences", []):
        conf_override = venue_overrides.get(conf, {})
        sources["conferences"].append(
            {
                "name": conf,
                "years": conf_override.get("years", years),
                "platform": conf_override.get("platform"),
                "additional_params": conf_override.get("additional_params", {}),
            }
        )

    return sources


def _merge_arxiv_tags(main_papers: List[Paper], arxiv_pool: List[Paper]) -> None:
    if not main_papers:
        return

    arxiv_by_norm_title = {}
    for p in arxiv_pool:
        key = _normalize_title(p.title)
        if key and key not in arxiv_by_norm_title:
            arxiv_by_norm_title[key] = p

    for paper in main_papers:
        has_open = bool(paper.pdf_url and "arxiv.org" not in (paper.pdf_url or ""))

        matched = None
        norm_title = _normalize_title(paper.title)
        if norm_title in arxiv_by_norm_title:
            matched = arxiv_by_norm_title[norm_title]
        else:
            # Cheap fuzzy fallback for near-identical titles.
            for k, candidate in arxiv_by_norm_title.items():
                if abs(len(k) - len(norm_title)) > 30:
                    continue
                if _title_similarity(norm_title, k) >= 0.8:
                    matched = candidate
                    break

        has_arxiv = matched is not None or bool(paper.arxiv_id)
        if matched and not paper.arxiv_id:
            paper.arxiv_id = matched.arxiv_id

        if has_open and has_arxiv:
            paper.download_available = "both"
        elif has_arxiv:
            paper.download_available = "arxiv"
        elif has_open:
            paper.download_available = "openreview"
        else:
            paper.download_available = "none"


def _collect_source_papers(
    sources: List[Dict[str, Any]],
    venue_type: str,
    accepted_only: bool = True,
) -> Tuple[List[Paper], List[Dict[str, str]]]:
    def _fallback_chain(source_name: str, primary_platform: str) -> List[str]:
        name = (source_name or "").upper()
        chain: List[str] = []

        by_platform = {
            "cvpr": ["dblp_cvpr"],
            "iccv": ["dblp_iccv"],
            "tcad": ["dblp_tcad"],
        }
        chain.extend(by_platform.get(primary_platform, []))

        # Venue-aware fallback, useful when user explicitly configures platform.
        by_venue = {
            "CVPR": ["dblp_cvpr"],
            "ICCV": ["dblp_iccv"],
            "TCAD": ["dblp_tcad"],
        }
        chain.extend(by_venue.get(name, []))

        # Dedupe and keep valid adapters only.
        dedup: List[str] = []
        for p in chain:
            if p == primary_platform:
                continue
            if p not in dedup and AdapterRegistry.supports_platform(p):
                dedup.append(p)
        return dedup

    papers: List[Paper] = []
    failures: List[Dict[str, str]] = []

    for source in sources:
        name = source.get("name")
        years = source.get("years") or [datetime.now().year]
        additional_params = source.get("additional_params", {}) or {}
        platform = _resolve_platform(name, source.get("platform"), venue_type, additional_params)
        fallback_candidates = _fallback_chain(name, platform)

        try:
            adapter = AdapterRegistry.get_required(platform)
            venue_cfg = VenueConfig(
                name=name,
                years=years,
                platform=platform,
                venue_id=source.get("venue_id"),
                additional_params=additional_params,
                accepted_only=source.get("accepted_only", accepted_only),
            )
            venue_papers = adapter.crawl(venue_cfg)
            if (not venue_papers) and fallback_candidates:
                for fallback in fallback_candidates:
                    print(f"[STAGE1] {name} ({platform}) -> 0, retrying with {fallback}")
                    fallback_adapter = AdapterRegistry.get_required(fallback)
                    fallback_cfg = VenueConfig(
                        name=name,
                        years=years,
                        platform=fallback,
                        additional_params=additional_params,
                        accepted_only=False,
                    )
                    venue_papers = fallback_adapter.crawl(fallback_cfg)
                    if venue_papers:
                        platform = fallback
                        break
            if not venue_papers:
                failures.append({"name": name, "platform": platform, "error": "0 papers fetched"})
            print(f"[STAGE1] {name} ({platform}) -> {len(venue_papers)} papers")
            papers.extend(venue_papers)
        except Exception as e:
            msg = str(e)
            print(f"[STAGE1] WARN {name} ({platform}) failed: {msg}")
            if fallback_candidates:
                try:
                    for fallback in fallback_candidates:
                        print(f"[STAGE1] retry {name} with fallback {fallback}")
                        fallback_adapter = AdapterRegistry.get_required(fallback)
                        fallback_cfg = VenueConfig(
                            name=name,
                            years=years,
                            platform=fallback,
                            additional_params=additional_params,
                            accepted_only=False,
                        )
                        venue_papers = fallback_adapter.crawl(fallback_cfg)
                        print(f"[STAGE1] {name} ({fallback}) -> {len(venue_papers)} papers")
                        if venue_papers:
                            papers.extend(venue_papers)
                            break
                    else:
                        failures.append({"name": name, "platform": fallback_candidates[-1], "error": "fallback returned 0 papers"})
                        continue
                    continue
                except Exception as e2:
                    failures.append({"name": name, "platform": fallback_candidates[-1], "error": str(e2)})
            failures.append({"name": name, "platform": platform, "error": msg})

    return papers, failures


def run_stage1(config: Dict) -> Optional[Path]:
    """运行阶段1增强版: 多源爬取 + 增量数据库 + arXiv补充标记。"""
    output_dir = Path(config['output_dir']) / 'data'
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = config.get("sources") or {}
    has_new_schema = bool(sources.get("conferences") or sources.get("journals") or sources.get("arxiv"))
    if not has_new_schema:
        sources = _legacy_sources_from_config(config)

    conferences = sources.get("conferences", []) or []
    journals = sources.get("journals", []) or []
    arxiv_cfg = sources.get("arxiv", {}) or {}

    conference_papers, conf_failures = _collect_source_papers(
        conferences,
        venue_type="conference",
        accepted_only=config.get("accepted_only", True),
    )
    journal_papers, journal_failures = _collect_source_papers(
        journals,
        venue_type="journal",
        accepted_only=True,
    )

    main_papers = conference_papers + journal_papers

    # arXiv integration: supplement download availability (default no DB write).
    arxiv_papers: List[Paper] = []
    if arxiv_cfg.get("enabled"):
        date_range = _parse_date_range(arxiv_cfg.get("date_range"))
        if date_range:
            start_year = int(date_range[0][:4])
            end_year = int(date_range[1][:4])
            arxiv_years = list(range(start_year, end_year + 1))
        else:
            arxiv_years = config.get("years", [datetime.now().year])

        keywords = arxiv_cfg.get("keywords") or config.get("topic") or config.get("keywords", "")
        arxiv_venue = VenueConfig(
            name="arXiv Search",
            years=arxiv_years,
            platform="arxiv",
            additional_params={
                "categories": arxiv_cfg.get("categories", ["cs.AI", "cs.LG"]),
                "keywords": keywords,
                "date_range": date_range,
                "max_results": arxiv_cfg.get("max_results", 200),
            },
            accepted_only=False,
        )
        try:
            arxiv_papers = AdapterRegistry.get_required("arxiv").crawl(arxiv_venue)
            print(f"[STAGE1] arXiv supplement -> {len(arxiv_papers)} papers")
        except Exception as e:
            print(f"[STAGE1] WARN arXiv supplement failed: {e}")

        _merge_arxiv_tags(main_papers, arxiv_papers)

        if arxiv_cfg.get("save_to_database", False):
            main_papers.extend(arxiv_papers)

    # Database-backed storage with incremental updates.
    db_cfg = config.get("database") or {}
    db_format = db_cfg.get("format", "json")
    default_db_path = Path(config["output_dir"]) / "database" / f"papers.{db_format}"
    db_path = Path(db_cfg.get("path", default_db_path))
    incremental = db_cfg.get("incremental", True)

    if not incremental and db_path.exists():
        db_path.unlink()

    db = DatabaseManager(db_path, format=db_format)
    if incremental:
        added, updated = db.incremental_update(main_papers)
    else:
        added = db.add_papers(main_papers)
        updated = 0
    db.save()

    stats = db.get_statistics()
    summary_path = output_dir / f"stage1_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "created_at": datetime.now().isoformat(),
                "database_path": str(db_path),
                "database_format": db_format,
                "incremental": incremental,
                "added": added,
                "updated": updated,
                "crawled": {
                    "conference_papers": len(conference_papers),
                    "journal_papers": len(journal_papers),
                    "arxiv_supplement": len(arxiv_papers),
                },
                "failures": conf_failures + journal_failures,
                "database_stats": stats,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nDATABASE_INCREMENTAL:added={added},updated={updated}")
    print(f"DATABASE_PATH:{db_path}")
    print(f"STAGE1_SUMMARY:{summary_path}")
    print(f"DATABASE_TOTAL:{stats['total_papers']}")

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
        max_workers=config.get('workers', 1)
    )
    
    print(f"\nSTAGE4_STATS:total={stats['total']},success={stats['success']},failed={stats['failed']}")
    
    return stats['success'] > 0


def run_stage4b(config: Dict, input_dir: Path, output_file: Optional[Path] = None, 
                topic: Optional[str] = None, api_key: Optional[str] = None,
                model: str = "anthropic/claude-3.5-sonnet") -> bool:
    """运行阶段4B: 生成研究总结"""
    if api_key is None:
        api_key = os.environ.get('OPENROUTER_API_KEY')
    
    if not api_key:
        print("ERROR:NO_API_KEY")
        return False
    
    if output_file is None:
        output_file = Path(config['output_dir']) / 'research_summary.md'
    
    if topic is None:
        topic = config.get('topic', 'Research Topic')
    
    success = generate_research_summary(
        analysis_dir=Path(input_dir),
        output_file=output_file,
        topic=topic,
        api_key=api_key,
        model=model
    )
    
    return success


def run_db_convert(input_path: Path, output_path: Optional[Path] = None, 
                   format: str = 'json') -> Optional[Path]:
    """Convert legacy JSON to new database format.
    
    Args:
        input_path: Path to legacy JSON file
        output_path: Output path for new database file
        format: Output format ('json' or 'csv')
        
    Returns:
        Path to converted database file, or None if failed
    """
    from database import convert_legacy_json
    
    try:
        result_path = convert_legacy_json(input_path, output_path, format)
        print(f"CONVERT_SUCCESS:{result_path}")
        return result_path
    except Exception as e:
        print(f"ERROR:CONVERT_FAILED:{e}")
        return None


def run_db_stats(database_path: Path) -> bool:
    """Show database statistics.
    
    Args:
        database_path: Path to database file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        db = DatabaseManager(database_path)
        stats = db.get_statistics()
        
        import json
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        return True
    except Exception as e:
        print(f"ERROR:DB_STATS_FAILED:{e}")
        return False


def run_db_merge(source_path: Path, target_path: Path) -> bool:
    """Merge two databases.
    
    Args:
        source_path: Path to source database file
        target_path: Path to target database file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load target database
        target_db = DatabaseManager(target_path)
        
        # Merge source into target
        added, updated = target_db.merge(source_path)
        
        # Save merged database
        target_db.save()
        
        print(f"MERGE_SUCCESS:added={added},updated={updated},total={len(target_db)}")
        return True
    except Exception as e:
        print(f"ERROR:MERGE_FAILED:{e}")
        return False


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
  stage4b     阶段4B: 生成研究总结
  all         运行所有阶段
  db-convert  转换旧版JSON到新版数据库格式
  db-stats    显示数据库统计信息
  db-merge    合并两个数据库

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
  python paper_agent.py stage4b --input ./analysis --output summary.md --config config.yaml
  python paper_agent.py all --config config.yaml --api-key xxx
  python paper_agent.py db-convert --input old_papers.json --output papers.json
  python paper_agent.py db-stats --database papers.json
  python paper_agent.py db-merge --source source.json --target target.json
        """
    )
    
    parser.add_argument('command', choices=['check', 'stage1', 'stage2', 'stage3', 'stage4', 'stage4b', 'all', 
                                            'db-convert', 'db-stats', 'db-merge'],
                       help='要执行的命令')
    parser.add_argument('--config', '-c', type=str,
                       help='配置文件路径 (YAML格式)')
    parser.add_argument('--input', '-i', type=str,
                       help='输入文件/目录路径')
    parser.add_argument('--output', '-o', type=str,
                       help='输出文件/目录路径')
    parser.add_argument('--api-key', type=str,
                       help='OpenRouter API密钥（覆盖环境变量）')
    parser.add_argument('--database', '-d', type=str,
                       help='数据库文件路径 (用于db-stats命令)')
    parser.add_argument('--source', '-s', type=str,
                       help='源数据库路径 (用于db-merge命令)')
    parser.add_argument('--target', '-t', type=str,
                       help='目标数据库路径 (用于db-merge命令)')
    parser.add_argument('--format', '-f', type=str, default='json',
                       choices=['json', 'csv'],
                       help='输出格式 (json或csv, 默认json)')
    parser.add_argument('--topic', type=str,
                       help='研究主题 (用于stage4b命令)')
    parser.add_argument('--model', type=str, default='anthropic/claude-3.5-sonnet',
                       help='模型名称 (用于stage4b命令)')
    
    args = parser.parse_args()
    
    # 环境检查 - 输出JSON
    if args.command == 'check':
        import json
        results = check_environment()
        print(json.dumps(results))
        sys.exit(0 if results['ready'] else 1)
    
    # 数据库命令 - 不需要配置文件
    if args.command == 'db-convert':
        if not args.input:
            print("ERROR:INPUT_REQUIRED")
            sys.exit(1)
        
        result = run_db_convert(
            Path(args.input),
            Path(args.output) if args.output else None,
            args.format
        )
        if result:
            sys.exit(0)
        else:
            sys.exit(1)
    
    elif args.command == 'db-stats':
        if not args.database:
            print("ERROR:DATABASE_REQUIRED")
            sys.exit(1)
        
        success = run_db_stats(Path(args.database))
        sys.exit(0 if success else 1)
    
    elif args.command == 'db-merge':
        if not args.source or not args.target:
            print("ERROR:SOURCE_AND_TARGET_REQUIRED")
            sys.exit(1)
        
        success = run_db_merge(Path(args.source), Path(args.target))
        sys.exit(0 if success else 1)
    
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
        
        elif args.command == 'stage4b':
            if not args.input:
                print("ERROR:INPUT_REQUIRED")
                sys.exit(1)
            
            success = run_stage4b(
                config, 
                Path(args.input), 
                Path(args.output) if args.output else None,
                args.topic,
                args.api_key,
                args.model
            )
            if not success:
                print("ERROR:STAGE4B_FAILED")
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
