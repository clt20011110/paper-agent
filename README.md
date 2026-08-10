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

## Typed workflow 与恢复

`run` 和 `resume` 只接受冻结的 typed workflow manifest；两者都必须提供
`--workflow`。清单中的每个 `FileRef` 都是相对清单目录的 `path` 加小写 SHA-256：任何
文件内容或路径漂移都会被拒绝，不能用新的 `--run-id` 绕过。`--config` 是可选的全局参数，
但若提供，必须与清单中 `config` 的 FileRef 指向同一个文件。

最小的单阶段搜索清单如下。`<...>` 必须替换为对应文件的实际摘要，不能原样执行：

```json
{
  "schema_version": "1",
  "workflow_id": "literature-2026-08",
  "config": {"path": "research.yaml", "sha256": "<sha256>"},
  "steps": [
    {
      "id": "search",
      "stage": "search",
      "plan": {"path": "QUERY_PLAN.json", "sha256": "<sha256>"},
      "stage2_release": {"path": "stage2-release.json", "sha256": "<sha256>"},
      "snapshots": [],
      "historical_replay": false
    }
  ]
}
```

在清单目录内生成摘要（macOS/Linux 均可用）并填入 JSON：

```sh
python3 - <<'PY'
from hashlib import sha256
from pathlib import Path
for name in ("research.yaml", "QUERY_PLAN.json", "stage2-release.json"):
    print(name, sha256(Path(name).read_bytes()).hexdigest())
PY
```

先验证，再执行；恢复时要再次带上完全相同的清单和 run ID：

```sh
paper-agent --dry-run run --workflow /absolute/path/to/workflow.json \
  --workflow-run-id literature-2026-08
paper-agent run --workflow /absolute/path/to/workflow.json \
  --database /absolute/path/to/papers.sqlite3 \
  --workflow-run-id literature-2026-08
paper-agent resume --workflow /absolute/path/to/workflow.json \
  --database /absolute/path/to/papers.sqlite3 \
  --workflow-run-id literature-2026-08
```

`schema_version: "1"` 只允许单阶段清单。多阶段必须使用 version 2，并通过
`{"from_step":"..."}` 绑定本次 workflow 的上游结果；静态 selection 不能替代这一绑定。
当前可连续运行 `search → filter → download`，例如 Filter 的 `selection` 写
`{"from_step":"search"}`，Download 的 `selection` 写 `{"from_step":"filter"}`。
Download 还必须显式冻结 `include_needs_review`；默认建议为 `false`。空 Search 结果会保持
为空，不会退化成筛选全库。`run --help` 只列命令参数；字段合同如下，未列字段会被拒绝：

| stage | typed step 字段（另含 `id`、`stage`） |
| --- | --- |
| `search` | `plan`、`stage2_release`、`snapshots`、`historical_replay` |
| `filter` | `plan`、`stage2_release`、`selection`（单阶段为 `FileRef`；v2 链为 Search output ref） |
| `download` | `selection`、`authorization_grant_id`、`provider_terms`；v2 另需 `include_needs_review` |
| `analyze` | `selection`、`processing_grant_id`、`policy` |
| `report` | `plan`、`corpus_snapshot`、`search_audit`、`processing_grants`、`previous_report_run_id`、`policy` |

Analyze 与 Report 的动态上游绑定仍受 ReportPlan 人工审批边界约束，暂时使用各自的单阶段
version 1 清单。所有文件字段均为 `FileRef`，`snapshots` 元素为
`{"provider":"...","file":<FileRef>}`；可选字段也必须显式写成 `null`。收到 SIGINT/SIGTERM
时，已在运行的 step 会先返回到自己的安全点，工作流仅在 step 边界 checkpoint 为可恢复状态。

## 授权下载与报告执行

`download` 默认只走公开来源。只有配置已启用授权 skill、有效 download grant 已批准，且下列
三个 handoff 参数同时给出时，CLI 才会生成审计过的浏览器队列；它不会自行登录或驱动浏览器：
按 Stage 2 run 选择时默认只处理 `relevant`；只有用户显式传入 `--include-needs-review`
才会把人工复核队列一并送入 Stage 3。

```sh
paper-agent --config /absolute/path/to/research.yaml download \
  --database /absolute/path/to/papers.sqlite3 \
  --filter-run-id <stage2-run-id> --grant-id <download-grant-id> \
  --authorized-skill-queue /absolute/path/to/authorized-queue.csv \
  --authorized-skill-output /absolute/path/to/authorized-output \
  --authorized-skill-root /absolute/path/to/download-authorized-papers
```

JSON 中的 `authorized_queue_path` 才是交给 `$download-authorized-papers` 的输入；该 skill
只通过用户可见、已登录的浏览器会话处理它。CAPTCHA、403、429 或登录修复会停止受影响队列，
不应把 cookie、密码、OTP 或 session 信息交给 CLI、skill 或日志。

报告执行不是 `report --plan-only` 的别名：先生成并批准 ReportPlan，再以其 bundle 执行。
执行时必须提供可解析的 policy（`--policy` 或同一 v2 config 的 summary policy）、数据库和
输出根；`--processing-grant` 的格式是 `小写十六进制_ARTIFACT_SHA256=GRANT_ID`。

```sh
paper-agent report prepare-inputs \
  --database /absolute/path/to/papers.sqlite3 \
  --artifact-root /absolute/path/to/artifacts \
  --output-root /absolute/path/to/reports \
  --crawl-run-id <crawl-run-id> --filter-run-id <stage2-run-id> \
  --stage4-run-id <stage4-run-id> --recent-cutoff 2024-01-01 \
  --created-at 2026-08-10T00:00:00Z
paper-agent report --plan-only --draft /absolute/path/to/REPORT_DRAFT.json \
  --corpus-snapshot /absolute/path/to/CORPUS_SNAPSHOT.json \
  --search-audit /absolute/path/to/SEARCH_AUDIT.json \
  --output-root /absolute/path/to/reports
paper-agent report approve --plan /absolute/path/to/REPORT_PLAN.json \
  --hash <plan-hash> --approved-by <operator> \
  --corpus-snapshot /absolute/path/to/CORPUS_SNAPSHOT.json \
  --search-audit /absolute/path/to/SEARCH_AUDIT.json \
  --output-root /absolute/path/to/reports
paper-agent --config /absolute/path/to/research.yaml report \
  --plan /absolute/path/to/REPORT_PLAN.json \
  --database /absolute/path/to/papers.sqlite3 \
  --output-root /absolute/path/to/reports \
  --processing-grant <artifact-sha256>=<grant-id>
```

## Codex skill

仓库内的 `skills/paper-agent` 是薄编排层：它只收集用户意图、展示计划/成本、调用同一 CLI 并解释结构化结果，不包含第二套爬虫或数据库实现。可复制或链接到个人 Codex skills 目录；安装后用 `$paper-agent` 触发。

涉及订阅站点时，系统只会为已批准论文生成 attended handoff。`download-authorized-papers` skill 使用用户自己的可见、已登录浏览器会话；它不会读取或保存密码、cookie、token，也不会绕过 CAPTCHA、403、429、付费墙或访问控制。

## 测试

```bash
.venv/bin/python -m pytest
```

CI 在按 lockfile 安装依赖后，以禁用外部 socket 的方式执行测试；测试过程不会访问实时站点、下载大模型、消耗 Codex 配额或使用订阅登录。冷缓存 runner 的依赖安装仍会访问 Python package index，不宣称整个 CI bootstrap 物理断网。真实金标晋级、oMLX 性能/浸泡测试、实时 venue smoke 和学校授权下载属于明确的外部门禁，缺少证据时生产运行会停止而不是伪造通过结果。

完整可执行规格见 `task.md`；安全边界、状态机和数据库设计见 `docs/`。

新机器配置和离线首跑见 [docs/getting-started.md](docs/getting-started.md)；备份、恢复、
授权故障与升级见 [docs/operations.md](docs/operations.md)；配置升级见
[docs/migration-v1-to-v2.md](docs/migration-v1-to-v2.md)，发布前门禁见
[docs/release-checklist.md](docs/release-checklist.md)。
