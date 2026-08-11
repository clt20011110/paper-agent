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

已有事实库可无损导出并回灌到新 SQLite；正式导入前先 dry-run 校验整个文件：

```bash
.venv/bin/paper-agent export --database /absolute/path/to/source.sqlite3 \
  --format jsonl --output /absolute/path/to/papers.jsonl
.venv/bin/paper-agent --dry-run import --database /absolute/path/to/new.sqlite3 \
  --format jsonl --input /absolute/path/to/papers.jsonl
.venv/bin/paper-agent import --database /absolute/path/to/new.sqlite3 \
  --format jsonl --input /absolute/path/to/papers.jsonl
```

`--format csv` 只接受本 CLI 导出的规范 CSV；旧版论文 JSON 使用
`--format legacy-json`，结果会报告字段映射、警告和未迁移路径。

## 冻结检索流程

先从 v2 YAML 编译草案并检查预算，再按显示的内容 hash 显式批准：

```bash
.venv/bin/paper-agent search plan \
  --input configs/query_draft.example.yaml \
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
当前可连续运行 `search → filter → download → analyze`。Filter、Download 与
Analyze 的 `selection` 依次写 `{"from_step":"search"}`、
`{"from_step":"filter"}` 和 `{"from_step":"download"}`。
Download 还必须显式冻结 `include_needs_review`；默认建议为 `false`。空 Search 结果会保持
为空，不会退化成筛选全库。Download 的静态 `paper_ids: []` 也会 fail-close；Stage 3 不再把
空 ID 列表解释成“读取全局最新 Stage 2 结果”。`run --help` 只列命令参数；字段合同如下，
未列字段会被拒绝：

| stage | typed step 字段（另含 `id`、`stage`） |
| --- | --- |
| `search` | `plan`、`stage2_release`、`snapshots`、`historical_replay` |
| `filter` | `plan`、`stage2_release`、`selection`（单阶段为 `FileRef`；v2 链为 Search output ref） |
| `download` | `selection`、`authorization_grant_id`、`provider_terms`、可选 `scope_snapshots`；v2 另需 `include_needs_review` |
| `analyze` | `selection`（单阶段为 `FileRef`；v2 链为 Download output ref）、`processing_grant_id`、`policy` |
| `report` | `plan`、`corpus_snapshot`、`search_audit`、`processing_grants`、`previous_report_run_id`、`policy`；handoff 生成的 manifest 另冻结 `artifact_root` DirectoryRef |

ReportPlan 必须在实际 corpus 和 Stage 4 结果产生后编译、人工批准；它的 paper membership
不能在 crawler 运行前猜测。因此完整分析 workflow 在 Analyze 结束，随后用
`report prepare-inputs --workflow-run-id` 从已完成 workflow 生成持久 handoff，基于该 handoff
编译并批准 ReportPlan，再生成冻结 approved plan、corpus、search audit 和已 pin plan path/hash
config 的独立单阶段 Report workflow。handoff 会冻结原 workflow 的 manifest、四个 child run、
Stage 3/4 数据库论文集合、artifact root 和输入文件 hash；漂移后拒绝计划、执行或恢复。多阶段
manifest 中直接追加 Report 会被拒绝。所有文件字段均为 `FileRef`，artifact root 使用受 manifest
目录约束的 `DirectoryRef`，`snapshots` 元素为
`{"provider":"...","file":<FileRef>}`；可选字段也必须显式写成 `null`。收到 SIGINT/SIGTERM
时，已在运行的 step 会先返回到自己的安全点，工作流仅在 step 边界 checkpoint 为可恢复状态。

使用 snapshot-scoped download grant 时，Download step 的 `scope_snapshots` 必须逐项冻结
`snapshot_type`、`snapshot_id`、`snapshot_hash`、`collection_id` 和可选 `file`。首次从文件加载时
使用 FileRef；已经校验并写入同一 SQLite 后可令 `file: null`，按 snapshot ID 恢复。adapter 会在
任何 provider probe/fetch 前重算文件或数据库内容、核对 grant 的精确 snapshot hash，并确认本次
选择的每篇论文都属于所有声明的 scope。旧清单仍可读取，但使用 snapshot-scoped grant 的旧
Download step 必须补齐这些绑定后才能执行或恢复。

当前 typed Download workflow 只支持 public provider grant。它没有冻结浏览器 queue/output、skill
root、原始 ZIP 或 audit manifest，因而 `provider: authorized_skill` 或绑定 skill/dependency digest 的
grant 会在 service/provider 构造前拒绝；请改用下述独立 `paper-agent download` authorized handoff。
动态 Filter 输出会先解析成 exact paper IDs，再把这些 ID 直接交给 Stage 3，同时单独保留来源
`filter_run_id` lineage；执行期间 Filter 表新增行不能扩大下载集合。

## 授权下载与报告执行

`download` 默认只走公开来源。只有配置已启用授权 skill、有效 download grant 已批准，且下列
三个 handoff 参数同时给出时，CLI 才会生成审计过的浏览器队列；它不会自行登录或驱动浏览器：
按 Stage 2 run 选择时默认只处理 `relevant`；只有用户显式传入 `--include-needs-review`
才会把人工复核队列一并送入 Stage 3。

```sh
paper-agent --config /absolute/path/to/research.yaml download \
  --database /absolute/path/to/papers.sqlite3 \
  --filter-run-id <stage2-run-id> --grant-id <download-grant-id> \
  --selection-snapshot /absolute/path/to/download-selection-snapshot.json \
  --authorized-skill-queue /absolute/path/to/authorized-queue.csv \
  --authorized-skill-output /absolute/path/to/authorized-output \
  --authorized-skill-root /absolute/path/to/download-authorized-papers
```

当 grant 使用 `collection_snapshot_hash` 或 `selection_snapshot_hash` 时，download 命令必须提供
与该哈希一致的 `--collection-snapshot` / `--selection-snapshot`。快照采用
`download-scope-snapshot.schema.json`，其按 `snapshot_type`、`collection_id` 和排序后的
`paper_ids` 重算 `snapshot_hash`；首次成功校验后写入 SQLite，恢复时可改用
`--collection-snapshot-id` / `--selection-snapshot-id`。collection scope 还要传入或由快照提供
同一个 `--collection-id`。运行、授权队列 reservation 和 fetch request 都绑定这些值，快照漂移
或 grant 撤销会在浏览器/网络副作用前拒绝继续。

可用内置 builder 生成规范快照；它以 RFC 8785 canonical JSON 对
`schema_version`、`snapshot_type`、`collection_id` 和排序后的 `paper_ids` 求 SHA-256，
`snapshot_id` 与 `created_at` 不进入 membership hash：

```sh
python3 - <<'PY'
from datetime import datetime, timezone
import json
from pathlib import Path

from paper_agent.downloads import build_download_scope_snapshot

snapshot = build_download_scope_snapshot(
    "selection",
    ["paper-1", "paper-2"],
    created_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
)
Path("download-selection-snapshot.json").write_text(
    json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(snapshot["snapshot_hash"])
PY
```

构建 collection snapshot 时把第一个参数改为 `"collection"`，并传入
`collection_id="<已存在的 collection ID>"`。CLI 加载时还会确认每个 paper ID 存在，且
collection snapshot 的全部论文在该 collection 中仍有非 `not_member` membership。

JSON 中的 `authorized_queue_path` 才是交给 `$download-authorized-papers` 的输入；该 skill
只通过用户可见、已登录的浏览器会话处理它。CAPTCHA、403、429 或登录修复会停止受影响队列，
不应把 cookie、密码、OTP 或 session 信息交给 CLI、skill 或日志。

报告执行不是 `report --plan-only` 的别名。推荐的 workflow-bound 链路要求原四阶段 workflow
已经完整结束（包括成功完成 Stage 4）；未完成或失败的 Stage 4 应先恢复，不能交给报告阶段。
下列命令输出的 `handoff_id`、输入路径、`plan_hash`、`manifest_path` 和
`report_workflow_run_id` 由后续命令逐项复用。批准前，先在新 config 中启用 summary，令 database
和 output root 与 handoff 相同，并把预期 `REPORT_PLAN.json` 路径及 `plan_hash` 固定到
`summary.report_plan`。

```sh
paper-agent report prepare-inputs \
  --database /absolute/path/to/papers.sqlite3 \
  --artifact-root /absolute/path/to/artifacts \
  --output-root /absolute/path/to/reports \
  --workflow-run-id <completed-analysis-workflow-run-id> \
  --recent-cutoff 2024-01-01 \
  --created-at 2026-08-10T00:00:00Z
paper-agent report --plan-only --handoff-id <handoff-id> \
  --draft /absolute/path/to/REPORT_DRAFT.json \
  --database /absolute/path/to/papers.sqlite3 \
  --artifact-root /absolute/path/to/artifacts \
  --output-root /absolute/path/to/reports
paper-agent report approve --plan /absolute/path/to/REPORT_PLAN.json \
  --hash <plan-hash> --approved-by <operator> \
  --corpus-snapshot <prepare-output-corpus-snapshot-path> \
  --search-audit <prepare-output-search-audit-path> \
  --output-root /absolute/path/to/reports \
  --handoff-id <handoff-id> --database /absolute/path/to/papers.sqlite3 \
  --artifact-root /absolute/path/to/artifacts \
  --workflow-config /absolute/path/to/report-workflow.yaml \
  --workflow-manifest /absolute/path/to/report-workflow.json \
  --workflow-policy /absolute/path/to/artifact-processing-v1.yaml
paper-agent run --workflow /absolute/path/to/report-workflow.json \
  --database /absolute/path/to/papers.sqlite3 \
  --workflow-run-id <report-workflow-run-id>
paper-agent resume --workflow /absolute/path/to/report-workflow.json \
  --database /absolute/path/to/papers.sqlite3 \
  --workflow-run-id <report-workflow-run-id>
```

需要远程处理授权时，把规范的 grant 映射文件传给 approve 的
`--workflow-processing-grants`。直接 `report --plan` 不能绕过已登记 handoff 的独立 workflow
绑定；保留的非 handoff 兼容命令仍使用
`--processing-grant <ARTIFACT_SHA256>=<GRANT_ID>`。

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
