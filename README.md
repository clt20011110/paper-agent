# Paper Agent v2

Paper Agent 是一套本地优先、可恢复、可审计的文献检索与综述系统。它把来源发现、语义筛选、合法 PDF 获取、逐篇分析和领域综述放在同一个 Python 服务层与 SQLite 事实源中；CLI 与 Codex skill 共用这套实现。

## 核心设计

- Stage 1：通过可扩展的 `VenueProvider`、`SearchProvider`、`CitationProvider`、`MetadataProvider` 和 `DownloadProvider` 接口组合官方来源、学术图谱、快照与本地种子。运行前冻结并批准 `QueryPlan`，所有来源、查询、轮次和不完整状态均可审计。
- Stage 2：在 Apple Silicon 上通过 oMLX 批量运行 reranker，并用 Qwen3.5-9B 处理灰区样本。生产运行必须绑定真实金标晋级门、路径级校准器和阈值；不允许静默使用测试 fake 或云端回退。
- Stage 3：依次尝试官方公开 PDF、Europe PMC/PMC、Unpaywall、arXiv、用户授权的可见浏览器会话和人工队列。下载、保存、浏览器数据共享均受显式 policy/grant 控制。
- Stage 4：用固定的 `gpt-5.6-luna` profile 分析获准的 normalized text、摘要或元数据，输出带定位的结构化 evidence units。dataset/metric/baseline/protocol 映射由版本化本地 registry 校验。
- Stage 4b：用固定的 `gpt-5.6-sol` profile 按语义 section 做稳定分层 reduce，生成 Claims-Evidence Matrix、ReportDocument AST、确定性 Markdown、sidecar、审计和增量 diff。

系统不依赖 OpenRouter 或 OpenCode，也不会把 `network=false` 误认为远程模型载荷留在本地。全文及其受限派生物进入 Luna/Sol 前，必须通过精确 artifact/model 处理授权。

## 安装

要求 Python 3.11–3.13；macOS Apple Silicon + oMLX 是首要部署目标。

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/paper-agent doctor
```

查看当前安装版本支持的命令与参数：

```bash
.venv/bin/paper-agent --help
.venv/bin/paper-agent search --help
```

## 冻结检索流程

先从 v2 YAML 编译草案并检查预算，再按显示的内容 hash 显式批准：

```bash
.venv/bin/paper-agent search plan \
  --input /absolute/path/to/research.yaml \
  --output-root /absolute/path/to/paper-research

.venv/bin/paper-agent search approve \
  --plan /absolute/path/to/paper-research/search/<plan-id>/QUERY_PLAN.draft.json \
  --hash <displayed-plan-hash> \
  --approved-by <operator>
```

生产检索还需要通过全部质量门的本地 Stage 2 release bundle。先做不会调用 provider 或模型的运行时校验，再执行：

```bash
.venv/bin/paper-agent --dry-run search run \
  --plan /absolute/path/to/QUERY_PLAN.json \
  --database /absolute/path/to/papers.sqlite3 \
  --stage2-release /absolute/path/to/stage2-release.json

.venv/bin/paper-agent search run \
  --plan /absolute/path/to/QUERY_PLAN.json \
  --database /absolute/path/to/papers.sqlite3 \
  --stage2-release /absolute/path/to/stage2-release.json
```

`--dry-run` 只证明配置、计划和本地 release 可以装载，不证明实时来源、学校订阅或模型服务可用。

## Codex skill

仓库内的 `skills/paper-agent` 是薄编排层：它只收集用户意图、展示计划/成本、调用同一 CLI 并解释结构化结果，不包含第二套爬虫或数据库实现。可复制或链接到个人 Codex skills 目录；安装后用 `$paper-agent` 触发。

涉及订阅站点时，系统只会为已批准论文生成 attended handoff。`download-authorized-papers` skill 使用用户自己的可见、已登录浏览器会话；它不会读取或保存密码、cookie、token，也不会绕过 CAPTCHA、403、429、付费墙或访问控制。

## 测试

```bash
.venv/bin/python -m pytest
```

CI 完全离线：不会访问实时站点、下载大模型、消耗 Codex 配额或使用订阅登录。真实金标晋级、oMLX 性能/浸泡测试、实时 venue smoke 和学校授权下载属于明确的外部门禁，缺少证据时生产运行会停止而不是伪造通过结果。

完整可执行规格见 `task.md`；安全边界、状态机和数据库设计见 `docs/`。
