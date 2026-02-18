---
name: paper-agent
description: 用户想要研究某个主题的最新论文，涉及爬取、过滤、下载和分析。通过 OpenCode Agent 交互式配置和执行。
---

## 执行流程（严格按照此流程执行，不能跳步！！！）

1. 根据用户情况确定需要搜寻的会议和年份以及主题
2. 生成配置文件并展示给用户
3. 用户确认配置，若不满意，则反复交流更改直至满意(这一步很关键，不能省略，一定要把会议搜索的范围，关键词的设计以及最后的输出路径明确的展现出来)。写配置文件前注意文件夹不要与之前目录中已有的结果冲突，或者在配置文件里指定新的输出文件夹路径。
4. 询问用户是否需要检查环境配置和API key，如果需要则执行 `python .opencode/skills/scripts/paper_agent.py check`，展示检查结果，并指导用户安装缺失的包或配置API key。
5. 询问用户选择执行模式：
   - 全自动模式：直接执行所有阶段，若用户选择全自动模式，则询问是否代为执行，若用户需要代为执行，则执行 `python .opencode/skills/scripts/paper_agent.py all --config xxx.yaml`，等待执行完成后展示结果，并说明不一定能在规定时间内执行出来，如果超时则请用户自行运行代码。若不需要代为执行，则展示命令行指令，指导用户自己执行（包括配置环境和api key）。
   - 交互过滤模式：Stage 2 后询问用户是否满意过滤结果，不满意则修改关键词重新过滤（不重新爬取）,直到用户满意或放弃，然后继续执行后续阶段。
6. 由于用户有可能在之前已经爬取过了，所以在用户提出这一需求时，先询问用户是否已经有爬取结果，如果有，则上述两种模式均直接进入 Stage 2 过滤阶段（并据此更改执行指令），如果没有，则从 Stage 1 开始执行。
7. Stage4执行完毕后，通过浏览阅读结果文件夹（如果是用户执行则在用户说结果运行出来之后做这一步）给出一份综合分析总结。



## 配置示例

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

## Architecture

```
User <-> OpenCode Agent <-> paper_agent.py (CLI tool)
                              |
              +---------------+---------------+---------------+
              |               |               |               |
           crawler         filter        downloader      analyzer
         (Stage 1)       (Stage 2)       (Stage 3)       (Stage 4)
```

**设计原则**:
- `paper_agent.py` 是**纯命令行工具**，无任何交互式输入
- 所有用户交互由 **OpenCode Agent** 负责
- 配置通过 YAML 文件传递
- 输出机器可解析的格式（JSON、关键字标记行）

## 两种模式补充说明

### Mode 1: 全自动模式

如何检查环境配置和api key: `python paper_agent.py check`, 输出 JSON 格式的环境检查结果，包含是否准备就绪、缺失的包、API 配置状态等。帮助/指导用户安装或配置环境


### Mode 2: 交互式过滤

Stage 2 完成后，对结果进行总结，并展示给用户，询问是否满意过滤结果。如果不满意，请根据用户反馈修改关键词配置（通过 OpenCode Agent 交互式界面），然后重新执行 Stage 2 过滤（不重新爬取）。这个过程可以循环进行，直到用户满意或选择放弃。

## CLI Commands

### Environment Check

```bash
python paper_agent.py check
```

**输出** (JSON):
```json
{
  "ready": true,
  "python_ok": true,
  "missing_packages": [],
  "api_configured": true,
  "api_key": "sk-or-v1-...xxx"
}
```

### Stage 1: Crawl

```bash
python paper_agent.py stage1 --config config.yaml
```

### Stage 2: Filter

```bash
python paper_agent.py stage2 \
  --input data/all_papers_xxx.json \
  --config config.yaml
```

### Stage 3: Download

```bash
python paper_agent.py stage3 \
  --input data/filtered_papers_xxx.json \
  --output ./papers
  --config config.yaml
```

### Stage 4: Analyze

```bash
python paper_agent.py stage4 \
  --input ./papers \
  --output ./analysis \
  --api-key $OPENROUTER_API_KEY
  --config config.yaml
```
