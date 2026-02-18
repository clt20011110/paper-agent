# Paper Agent - 四阶段论文搜索分析系统

一个强大的多阶段论文搜索、筛选和分析系统，专门设计用于从OpenReview高效获取和分析学术会议论文(目前仅限NIPS,ICML,ICLR和AAAI)。

## 核心特性

- **阶段1 - 智能爬虫**: 从OpenReview批量爬取会议论文，获取完整的标题、摘要和元数据
- **阶段2 - 语义筛选**: 精准判断与主题的相关性
- **阶段3 - PDF下载**: 双通道下载（OpenReview + arXiv）
- **阶段4 - 深度分析**: 使用OpenRouter API批量分析PDF，提取方法、贡献、实验结果等关键信息


## 快速开始

opencode安装该skill可直接使用，也可以自己手动配置，流程如下：

### 1. 安装依赖

```bash
pip install openreview-py requests feedparser pdfplumber tqdm
#配置openrouter api-key
export OPENROUTER_API_KEY="sk-xx"
```

### 2. 使用流程

```bash
python .opencode/skills/paper-agent/scripts/paper_agent.py all --config xxx.yaml
```

### 3. 配置示例

```yaml
output_dir: ./paper_research
conferences:
  - ICLR
years:
  - 2024
topic: Diffusion Models for Molecular Generation
keywords: diffusion, molecular, generation, score-based

filter:
  include_groups:
    - [diffusion, molecular]      # AND logic
    - [diffusion, generation]     # AND logic
    - [score, based]              # AND logic
  exclude:
    - survey
    - review
  match_fields:
    - title
    - abstract
  case_sensitive: false
  whole_word: false

options:
  workers: 4
  delay: 1.0
  accepted_only: true
```

### Keyword Filter Logic

- **include_groups**: OR of AND groups
  - 每组内 AND 关系
  - 组间 OR 关系
  
- **exclude**: NOT logic
  - 包含排除词的论文会被过滤
