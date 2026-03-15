# Paper Agent Enhancement Task

## Project Overview
增强 paper-agent 的功能，使其支持更广泛的会议/期刊来源、更智能的过滤机制、更完善的数据库管理和更强大的分析能力。

## Current Architecture
- **Stage 1 (Crawler)**: OpenReview爬取 (仅支持NeurIPS/ICML/ICLR/AAAI)
- **Stage 2 (Filter)**: 基于正则表达式的关键词过滤
- **Stage 3 (Downloader)**: PDF下载 (OpenReview + arXiv搜索)
- **Stage 4 (Analyzer)**: 使用OpenRouter API分析PDF

---

## Enhancement Requirements

### 1. Crawler Expansion (Stage 1 Enhancement)

#### 1.1 Support Additional CCF-A Conferences
扩展支持的会议列表，除了现有的会议，新增：

**AI/ML Conferences:**
- AAAI (已支持，需验证)
- ACL (Association for Computational Linguistics)
- CVPR (IEEE/CVF Conference on Computer Vision and Pattern Recognition)
- ICCV (IEEE/CVF International Conference on Computer Vision)
- IJCAI (International Joint Conference on Artificial Intelligence)

**EDA Conferences:**
- DAC (Design Automation Conference)
- TCAD (IEEE Transactions on Computer-Aided Design)
- ICCAD (IEEE/ACM International Conference on Computer-Aided Design)

**Implementation Requirements:**
- [ ] 为每个新增会议实现专门的爬取适配器
- [ ] 研究各会议的论文发布平台（OpenReview/ACM DL/IEEE Xplore/会议官网等）
- [ ] 统一数据格式，确保所有来源的论文数据结构一致
- [ ] 处理不同平台的API差异和限制
- [ ] 不要死磕如何获取pdf下载链接，一旦发现做不出来，请只考虑爬取论文标题和摘要即可

#### 1.2 Support Journal Crawling
新增期刊支持，实现期刊论文爬取：

**Nature Series:**
- Nature Machine Intelligence
- Nature Chemistry
- Nature Computer Science
- Nature Communications
- Nature Catalysis
- Nature Biotechnology
- Nature Biomedical Engineering

**Other Journals:**
- Cell
- Science

**Implementation Requirements:**
- [ ] 研究期刊网站的爬取方式（可能涉及PubMed, CrossRef, 期刊官网等）
- [ ] 处理期刊与会议不同的元数据结构（volume, issue, pages等）
- [ ] 实现期刊特有的PDF获取方式
- [ ] 考虑期刊的更新频率和回溯范围

#### 1.3 Database-Backed Storage with Incremental Updates
重构数据存储层，支持可复用和增量更新：

**Database Format:**
- [ ] 支持人类可读的格式：JSON 和 CSV
- [ ] 设计统一的数据库schema，包含字段：
  - 基础信息：id, title, abstract, authors, keywords, year
  - 来源信息：venue, venue_type (conference/journal), source_platform
  - 状态信息：crawl_date, last_updated, download_status, analysis_status
  - 元数据：pdf_url, doi, bibtex, citation_count (如有)

**Incremental Update:**
- [ ] 实现基于论文ID的重复检测机制
- [ ] 新爬取时只添加新论文，更新已有论文的元数据
- [ ] 支持按时间范围筛选（只爬取某日期之后的论文）
- [ ] 提供数据库合并功能（合并多次爬取的结果）

**Extensibility:**
- [ ] 设计插件式架构，方便后续添加新的会议/期刊源
- [ ] 配置文件中支持动态添加新的venue配置

#### 1.4 arXiv Integration
增强arXiv支持：

**Search & Crawl:**
- [ ] 使用宽泛关键词（用户研究主题）搜索arXiv
- [ ] 支持按日期范围、类别(cs.AI, cs.LG等)筛选
- [ ] 搜索结果不保存到主数据库，仅用于补充下载

**Download Tagging:**
- [ ] 为每篇论文添加 `download_available` 标签
- [ ] 标记来源：openreview_available, arxiv_available, both
- [ ] Stage 3下载时优先选择可获取的来源

---

### 2. Semantic Filtering (Stage 2 Enhancement)

新增基于本地LLM的智能语义过滤功能，作为正则过滤的补充/替代。

#### 2.1 Local Model Integration
**Model Support:**
- [ ] 支持 Qwen3.5-0.8B 或 Qwen3.5-2B (轻量级模型，适合CPU推理)
- [ ] 使用 transformers 实现本地推理
- [ ] **Phase 1 仅支持CPU**，GPU支持作为后续优化

**CPU Optimization:**
- [ ] 实现批处理推理，最大化CPU利用率
- [ ] 使用 ONNX Runtime 或 Intel OpenVINO 加速CPU推理（可选优化）
- [ ] 提供进度显示和内存管理，控制内存占用

#### 2.2 Semantic Relevance Scoring
**Filtering Logic:**
- [ ] 输入：用户主题描述 + 论文（标题/摘要/关键词）
- [ ] 输出：相关性分数（0-1）+ 判断理由
- [ ] 支持阈值配置（如只保留score > 0.7的论文）

**Prompt Design:**
- [ ] 设计结构化prompt，要求模型输出JSON格式
- [ ] 包含few-shot示例提高判断准确性
- [ ] 支持领域特定的判断标准

#### 2.3 Hybrid Filtering Mode
**Integration:**
- [ ] 保留原有的正则过滤作为预过滤（快速筛除明显不相关）
- [ ] 语义过滤作为精过滤（处理边界情况）
- [ ] 支持纯正则、纯语义、混合三种模式

---

### 3. Stage 3 (Downloader) - Minor Updates
保持现有功能，新增：
- [ ] 根据Stage 1的`download_available`标签选择下载源
- [ ] 下载失败时尝试备用源（OpenReview失败则试arXiv）
- [ ] 更新下载记录的数据库状态

---

### 4. Stage 4 (Analyzer) Enhancement

#### 4.1 Fallback Analysis
**No-PDF Handling:**
- [ ] 如果PDF下载失败，使用标题+摘要进行分析
- [ ] 调整prompt，说明只能基于摘要分析
- [ ] 标记分析结果的置信度（full_pdf / abstract_only）

#### 4.2 Summary Generation
**新增Stage 4b: 综合分析报告生成**
- [ ] 读取所有Stage 4生成的分析报告
- [ ] 调用高能力模型（通过OpenRouter）生成领域综述
- [ ] 报告内容：
  - 研究趋势总结
  - 主流方法分类
  - 关键技术对比
  - 开源资源汇总
  - 研究空白识别

**Implementation:**
- [ ] 设计新的prompt模板用于综述生成
- [ ] 实现报告聚合逻辑（处理大量论文时的分段处理）
- [ ] 输出Markdown格式的综述报告

---

### 5. Configuration Schema Update

更新YAML配置格式，支持新功能：

```yaml
# 基础配置
output_dir: ./paper_research
topic: "Diffusion Models for Molecular Generation"

# 数据来源配置（支持动态扩展）
sources:
  conferences:
    - name: ICLR
      years: [2024, 2025]
      platform: openreview
    - name: AAAI
      years: [2024]
      platform: openreview
    - name: CVPR
      years: [2024]
      platform: openreview  # 或其他平台
    - name: DAC
      years: [2024]
      platform: ieee  # EDA会议可能需要不同平台
  
  journals:
    - name: "Nature Machine Intelligence"
      years: [2023, 2024]
      platform: nature
    - name: "Nature Chemistry"
      years: [2023, 2024]
      platform: nature
  
  arxiv:
    enabled: true
    categories: [cs.AI, cs.LG, cs.CL]
    keywords: "diffusion model, molecular generation"
    date_range: "2023-01-01 to 2024-12-31"
    save_to_database: false  # arxiv结果不保存

# 数据库配置
database:
  format: json  # 或 csv
  path: ./database/papers.json
  incremental: true
  backup: true

# 过滤配置（新增语义过滤）
filter:
  mode: hybrid  # regex_only / semantic_only / hybrid
  
  # 正则过滤（保留原有功能）
  regex:
    include_groups:
      - [diffusion, molecular]
      - [diffusion, generation]
    exclude:
      - survey
      - review
    match_fields: [title, abstract]
    case_sensitive: false
    whole_word: false
  
  # 语义过滤（新增）
  semantic:
    enabled: true
    model: "Qwen/Qwen3.5-0.8B-Instruct"  # 或 "Qwen/Qwen3.5-2B-Instruct"
    device: "cpu"  # Phase 1 仅支持CPU
    batch_size: 8  # CPU建议使用较小batch
    threshold: 0.7
    max_workers: 2  # CPU限制并行度

# 分析配置
analysis:
  model: "stepfun/step-3.5-flash:free"  # PDF分析模型
  summary_model: "anthropic/claude-3.5-sonnet"  # 综述生成模型（高能力）
  generate_summary: true  # 是否生成综述
  workers: 4

# 下载配置
download:
  delay: 1.0
  timeout: 60
  preferred_source: "openreview"  # openreview / arxiv / auto
```

---

## Implementation Phases

### Phase 1: Database Layer & Storage (Foundation)
1. 设计数据库schema（JSON/CSV格式）
2. 实现数据库管理模块（增删改查、增量更新、合并）
3. 重构现有crawler，使其输出到新数据库格式
4. 保持向后兼容（支持从旧格式迁移）

### Phase 2: Crawler Expansion
1. 实现ACL/CVPR/ICCV/IJCAI的爬取适配器
2. 实现Nature系列期刊的爬取适配器
3. 实现EDA会议（DAC/TCAD/ICCAD）的爬取适配器
4. 设计插件式venue注册机制

### Phase 3: Semantic Filtering
1. 集成transformers，实现本地模型加载（Qwen3.5-0.8B/2B）
2. 针对CPU优化：调整batch size和并行度
3. 实现语义相关性判断逻辑
4. 集成到Stage 2，支持混合过滤模式

### Phase 4: arXiv Integration
1. 实现arXiv搜索模块
2. 实现下载可用性标记
3. 更新Stage 3下载逻辑支持多源

### Phase 5: Analysis Enhancement
1. 实现无PDF情况下的摘要分析
2. 实现Stage 4b综述报告生成
3. 集成到主流程

### Phase 6: Testing & Documentation
1. 各模块单元测试
2. 端到端集成测试
3. 更新文档和示例配置

---

## Technical Considerations

### Dependencies
新增依赖（预估）：
```
transformers>=4.35.0
# accelerate>=0.24.0  # Phase 1 暂不需要（CPU only）
sentencepiece>=0.1.99
protobuf>=4.24.0
feedparser>=6.0.10  # 可能已有
pandas>=2.0.0  # CSV支持
# onnxruntime>=1.16.0  # 可选：CPU加速
```

### Performance Considerations
- **CPU推理**: Qwen3.5-0.8B/2B 在CPU上运行，需优化batch size和并行度
- **内存管理**: 本地LLM可能占用较多内存，需实现批处理和清理机制
- **API限流**: OpenRouter和arXiv都有速率限制，需实现退避策略
- **并发安全**: 数据库操作需考虑并发写入的情况

### Error Handling
- 网络失败自动重试（指数退避）
- 模型加载失败提供友好的错误提示
- 单个venue爬取失败不影响其他venue
- 部分PDF下载失败不影响整体流程

---

## Acceptance Criteria

- [ ] 支持至少5个新的CCF-A会议和5个Nature子刊
- [ ] 数据库支持JSON和CSV格式，支持增量更新
- [ ] 语义过滤支持 Qwen3.5-0.8B 或 Qwen3.5-2B 模型（CPU推理）
- [ ] arXiv搜索集成，支持下载源标记
- [ ] Stage 4支持无PDF时的摘要分析
- [ ] Stage 4b能生成合格的领域综述报告
- [ ] 配置向后兼容，旧配置文件仍可用
- [ ] 完整的错误处理和日志记录

---

## Notes

1. **Conference Platform Research**: 需要先调研每个新增会议/期刊的论文发布平台，不同会议可能使用不同系统
2. **Model Selection**: Qwen3.5-0.8B 是推荐起点（内存占用小，推理快），如效果不佳可升级到2B。两者均可在CPU上运行。
3. **CPU Optimization**: 使用较小的batch_size（如8），限制max_workers（如2），避免内存溢出
4. **Incremental Logic**: 以论文ID为主键，标题/作者变化视为同一篇论文的更新
5. **Extensibility**: 设计时应考虑未来添加更多venue的便利性
