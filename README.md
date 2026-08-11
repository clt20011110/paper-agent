# Paper Agent v2

Paper Agent 是一套本地优先、可恢复、可审计的文献检索与综述系统。它把来源发现、语义筛选、合法 PDF 获取、逐篇分析和领域综述放在同一个 Python 服务层与 SQLite 事实源中；CLI 与 Codex skill 共用这套实现。

## 核心设计

- Stage 1：通过可扩展的 `VenueProvider`、`SearchProvider`、`CitationProvider`、`MetadataProvider` 和 `DownloadProvider` 接口组合官方来源、学术图谱、快照与本地种子。运行前冻结并批准 `QueryPlan`，所有来源、查询、轮次和不完整状态均可审计。
- Stage 2：在 Apple Silicon 上通过 oMLX 批量运行 reranker，并用 Qwen3.5-9B 处理灰区样本。生产运行必须绑定真实金标晋级门、路径级校准器和阈值；不允许静默使用测试 fake 或云端回退。
- Stage 3：依次尝试官方公开 PDF、Europe PMC/PMC、Unpaywall、arXiv、用户授权的可见浏览器会话和人工队列。下载、保存、浏览器数据共享均受显式 policy/grant 控制。
- Stage 4：用固定的 `gpt-5.6-luna` profile 分析获准的 normalized text、摘要或元数据，输出带定位的结构化 evidence units。dataset/metric/baseline/protocol 映射由版本化本地 registry 校验。
- Stage 4b：把全部 Luna 逐篇报告确定性排序并一次完整打包，用固定的 `gpt-5.6-sol` profile 严格调用一次 `one_shot_report`；随后在本地生成 Claims-Evidence Matrix、ReportDocument AST、确定性 Markdown、sidecar、审计和增量 diff，不再调用 Sol。

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

Stage 2 production release 现为 schema v3：公共门禁从原始 evidence 重算，hidden promotion
必须由部署侧 `PAPER_AGENT_STAGE2_HIDDEN_TRUST` 指向的 Ed25519 trust manifest 验签；release
bundle 不能携带或选择这个 trust root。先阅读
[hidden evaluator custody runbook](docs/security/stage2-hidden-evaluator-custody.md)，再配置该环境变量。

Stage 2 金标先抽样、后标注：evaluator 从 private snapshot 的完整自然语料框按冻结 seed 抽取 150 条
HIDDEN_REAL（不读取 curated labels，记录真实纳入概率 150/N），再从剩余 paper-family 的 curated pool 构建 DEV/HIDDEN_HARD。
curated annotations 是只用于配额与难例分层的临时 curation 标签，不是 gold；完整 snapshot 无需全量预标。
选中 600 个 pair 后，全部样本才进入权威的双人独立标注和第三人仲裁。构建命令（`--dry-run` 是全局选项）为：

```sh
python scripts/freeze_stage2_crossref_snapshot.py \
  --query-spec configs/stage2/real-sampling-crossref-v1.json \
  --contact operator@example.org \
  --output /secure/evaluator/private-snapshot.json \
  --capture-directory /secure/evaluator/crossref-raw \
  --capture-manifest /secure/evaluator/crossref-captures.json

paper-agent --dry-run stage2-sampling freeze-frame \
  --private-snapshot /secure/evaluator/private-snapshot.json \
  --output /secure/evaluator/hidden-real-freeze-frame.json

paper-agent --dry-run stage2-sampling curation-worklist \
  --private-snapshot /secure/evaluator/private-snapshot.json \
  --hidden-real-freeze-frame /secure/evaluator/hidden-real-freeze-frame.json \
  --output /secure/evaluator/curation-worklist.json

# Fill curation-decisions.json privately; provisional labels are never gold.
paper-agent --dry-run stage2-sampling curation-import \
  --private-snapshot /secure/evaluator/private-snapshot.json \
  --hidden-real-freeze-frame /secure/evaluator/hidden-real-freeze-frame.json \
  --worklist /secure/evaluator/curation-worklist.json \
  --decisions /secure/evaluator/curation-decisions.json \
  --curated-annotations-output /secure/evaluator/curated-annotations.json \
  --receipt-output /secure/evaluator/curation-receipt.json

paper-agent --dry-run stage2-sampling build \
  --private-snapshot /secure/evaluator/private-snapshot.json \
  --hidden-real-freeze-frame /secure/evaluator/hidden-real-freeze-frame.json \
  --curated-annotations /secure/evaluator/curated-annotations.json \
  --curation-receipt /secure/evaluator/curation-receipt.json \
  --gold-manifest-output /secure/evaluator-transfer/gold-manifest.json \
  --provenance-output /secure/evaluator/provenance.json
```

manifest 只给 HIDDEN_REAL 记录可解释的 `sampling_probability=150/N`；DEV/HIDDEN_HARD 受配额、权重和
paper-family 约束，字段为 `null`，不会用错误概率污染逆概率指标。

无标签的 600-pair gold manifest 可放入 release；private snapshot、HIDDEN_REAL freeze frame、curated annotations、provenance 与原始标注 ledger
均留在 evaluator custody（当前 provenance 也不公开）。`--private-labels` 只包含 manifest 的精确 600 条标签，不是
snapshot 全量标签。

当前真实语料已完成上述抽样并冻结无标签 600-pair manifest；公开安全的聚合记录见
[stage2-real-curation-frame-20260812.json](docs/smoke/stage2-real-curation-frame-20260812.json)。其中本地
9B 输出仅是抽样临时标签，人工双标与第三人仲裁仍未完成，不能作为 production release。

两位标注者和第三人完成 ledger 后，先验证再生成 promotion 私有输入：

```sh
paper-agent --dry-run stage2-sampling finalize-annotations \
  --gold-manifest /secure/evaluator-transfer/gold-manifest.json \
  --annotation-ledger /secure/evaluator/annotation-ledger.json \
  --private-labels-output /secure/evaluator/private-gold-labels.json
```

该命令要求每个 pair 恰好两份独立标注、所有分歧恰好一次第三人仲裁、仲裁前 quadratic-weighted kappa
至少为 0.75，并重新验证语言正例与 hard-case 配额。移除 `--dry-run` 后只写 label artifact，不复制标注者身份或原始仲裁记录，且拒绝覆盖已有输出。

生产 release 的顺序是 `promote → assemble → load`。优先在隔离 evaluator 中使用一次性
`promote`；`--candidate` 与 `--submission` 是可重复的 `ID=PATH`，两边 ID 集合必须完全一致：

```sh
paper-agent --dry-run stage2-evaluator promote \
  --manifest /secure/evaluator/gold-manifest.json \
  --private-labels /secure/evaluator/private-labels.json \
  --candidate incumbent=/secure/evaluator/incumbent-candidate-v2.json \
  --candidate challenger=/secure/evaluator/challenger-candidate-v2.json \
  --submission incumbent=/secure/evaluator/incumbent-submission.json \
  --submission challenger=/secure/evaluator/challenger-submission.json \
  --public-evidence incumbent=/secure/evaluator/incumbent-public-evidence.json \
  --public-evidence challenger=/secure/evaluator/challenger-public-evidence.json \
  --incumbent-candidate-id incumbent \
  --evaluator-id evaluator-team-1 \
  --evaluation-run-id promotion-2026-08-11 \
  --state-root /secure/evaluator/state \
  --evaluator-key-id evaluator-key-2026-08 \
  --issued-at 2026-08-11T08:00:00Z \
  --trust-manifest /secure/deployment/hidden-evaluator-trust.json \
  --signing-key-file /secure/evaluator/hidden-promotion-key.pem \
  --output /secure/transfer/hidden-promotion-attestation.json
```

dry-run 不读取 private labels、submission 或私钥，不消费 marker，也不创建 output；确认
`status: "validated"` 后，只移除 `--dry-run`，以完全相同参数执行一次。真实执行无论 gate
通过还是失败都会消费该 holdout；失败也会生成签名失败证明，不能组装为 release，更不能删除
`<state-root>/<manifest-hash>.promotion.json` 后重试。私钥必须是当前用户拥有、非 symlink、精确
`0600`、不超过 16 KiB 的 canonical unencrypted Ed25519 PKCS#8 PEM。当前 CLI 不直接支持 HSM
或 secret-manager signing API。

release builder 将 public-safe attestation 放入 evidence index 后，先 dry-run，再用同一参数组装：

```sh
paper-agent --dry-run stage2-release assemble \
  --candidate /absolute/path/to/release-bundle/stage2-candidate-v2.json \
  --evidence /absolute/path/to/release-bundle/stage2-release-evidence.json \
  --trust-manifest /secure/deployment/hidden-evaluator-trust.json \
  --output /absolute/path/to/release-bundle/stage2-release.json
```

candidate 与 output 必须具有同一 parent，evidence 与所有 evidence 引用必须留在该 bundle root；
trust manifest 必须在 root 外，output 必须不存在。dry-run 重算同样的门禁但不写 output；验证后只移除 `--dry-run`
执行组装。`stage2-evaluator attest --payload --signing-key-file --trust-manifest --output` 只适用于
另一个已审阅的一次性 evaluator 已经生成 public-safe payload 的场景；它本身不运行 hidden
evaluation，也不创建 marker，不能用来绕过 `promote`。组装完成后，`PAPER_AGENT_STAGE2_HIDDEN_TRUST`
才用于 `doctor`、search/filter/workflow 等生产 release 加载；evaluator 与 assembler 始终使用显式
`--trust-manifest`。完整的 dry-run、失败、轮换和事故语义见上述 custody runbook。

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

`download` 默认只走公开来源。只有配置已启用授权 skill、有效 attended download grant 已批准，且
下列 queue、output、skill root、原始 ZIP 四个运行时输入同时给出时，CLI 才会生成审计过的浏览器
队列；它不会自行登录或驱动浏览器：
按 Stage 2 run 选择时默认只处理 `relevant`；只有用户显式传入 `--include-needs-review`
才会把人工复核队列一并送入 Stage 3。

```sh
paper-agent --config /absolute/path/to/research.yaml --run-id <stage3-run-id> download \
  --database /absolute/path/to/papers.sqlite3 \
  --filter-run-id <stage2-run-id> --grant-id <download-grant-id> \
  --selection-snapshot /absolute/path/to/download-selection-snapshot.json \
  --authorized-skill-queue /absolute/path/to/authorized-queue.csv \
  --authorized-skill-output /absolute/path/to/authorized-output \
  --authorized-skill-root /absolute/path/to/download-authorized-papers \
  --authorized-skill-zip /absolute/path/to/download-authorized-papers-skill.zip
```

`--authorized-skill-audit` 只用于显式选择另一个已经审阅的 audit manifest；省略时使用 wheel
内置的版本化 audit。原始 ZIP、安装内容或 dependency digest 任一不匹配都会 fail closed。download
命令不会从 doctor 的环境默认值推断 ZIP，应显式传入上述文件。

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

第一次运行通常返回 `manual_required` 和 `authorized_queue_path`。这表示队列已冻结并等待 attended
browser handoff，不是下载成功。完整闭环如下：

1. 把 JSON 中的 `authorized_queue_path` 原样交给 `$download-authorized-papers`；队列是只读、不可变的
   grant/run/selection 绑定，不能手工增删行、替换 URL 或重排。
2. 用 skill 的 `paper_queue.py plan`（传入 `--csv <authorized_queue_path>` 和
   `--output <authorized_output>`）检查队列，再用相同 CSV/output 调用小批量
   `next --unscanned --limit 2`。在用户可见、已登录的浏览器中运行固定 publisher pass；每篇基准
   延迟为 30 秒、jitter 为 5 秒，并为每批使用全新的 browser event JSONL。
3. 每批下载完成后用 `paper_queue.py stage`，传入同一 `--csv`、`--output`，以及
   `--downloads <browser-download-directory> --events <fresh-events.jsonl> --wait-seconds 30` 做校验、
   复制和哈希；完成整个 fixed pass 后再以同一 CSV/output 运行 `paper_queue.py audit`，不能把 UI
   fallback 穿插在 fixed pass 中。
4. CAPTCHA、403、429、access denied、登录失效、缺少或歧义的授权 PDF 链接、`stopQueue: true`
   都必须立即停止，由用户修复同一可见会话；不得读取 cookie、密码、OTP、token，猜测 selector，
   或绕过访问控制。
5. 对将要导入的每个 DOI，ledger 必须是 `complete` 或 `complete_no_si`，article 与所有已发现 PDF SI
   均已通过校验并记录 SHA-256；`manual_required`、缺 article 或缺已发现 SI 都不能作为成功。
6. 最后以完全相同的 config、`--run-id`、database、selection、grant、queue、output、skill root、ZIP
   和 audit 参数重新运行原 download 命令。CLI 会再次验证 immutable queue、grant 与全部 digest，
   只从 final ledger 导入匹配 SHA-256 的 `article.pdf`；它不会在这一步启动浏览器。

队列或输入有任何漂移时必须保留旧审计证据并创建新的 run/handoff；scope 或 digest 改变时还必须
创建新 grant，不能改旧 CSV 后继续。
skill 自身的 `plan/next/stage/audit/recover` 细节以已安装、digest 匹配的 `SKILL.md` 为准。

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

受控 venue E2E 使用冻结的单记录 Stage 1 snapshot 与 `TEST_ONLY` Stage 2，随后下载真实公开 PDF；
Luna 和 Sol 只有带显式执行参数时才会启动。先以单个 venue 验证边界：

```bash
.venv/bin/python scripts/run_venue_e2e_matrix.py \
  --venue icml --output-root /absolute/run/root --run-id icml-e2e
.venv/bin/python scripts/venue_e2e_stage4.py \
  --run-dir /absolute/run/root/icml-e2e --through-stage stage4b
.venv/bin/python scripts/venue_e2e_stage4.py \
  --run-dir /absolute/run/root/icml-e2e --through-stage stage4b \
  --resume --execute-models
.venv/bin/python scripts/summarize_venue_e2e_matrix.py \
  --run-root /absolute/run/root --venue-catalog-root venues
```

最后一个汇总命令不调用模型，并对 run lineage、PDF/analysis/report CAS、一次性 Sol ledger、
报告 manifest 与完整本地 verifier 做只读核验。受控矩阵不替代逐 provider 的 live transport 验收，
也不替代 Stage 2 production release gate。最新的 20 个 venue 全当前运行结果见
[docs/acceptance/venue-e2e-matrix-20260812.md](docs/acceptance/venue-e2e-matrix-20260812.md)，
其验收 manifest 不导入历史行；2026-08-11 的“19 个当前运行 + NeurIPS 历史复用”记录仍保留供审计。
需要提交机器可读证据时加 `--portable-paths --json-output /absolute/evidence.json`，本机 run/repository
前缀会替换为 `$RUN_ROOT`/`$REPOSITORY_ROOT`。

CI 在按 lockfile 安装依赖后，以禁用外部 socket 的方式执行测试；测试过程不会访问实时站点、下载大模型、消耗 Codex 配额或使用订阅登录。冷缓存 runner 的依赖安装仍会访问 Python package index，不宣称整个 CI bootstrap 物理断网。真实金标晋级、oMLX 性能/浸泡测试、实时 venue smoke 和学校授权下载属于明确的外部门禁，缺少证据时生产运行会停止而不是伪造通过结果。

完整可执行规格见 `task.md`；安全边界、状态机和数据库设计见 `docs/`。

新机器配置和离线首跑见 [docs/getting-started.md](docs/getting-started.md)；备份、恢复、
授权故障与升级见 [docs/operations.md](docs/operations.md)；配置升级见
[docs/migration-v1-to-v2.md](docs/migration-v1-to-v2.md)，发布前门禁见
[docs/release-checklist.md](docs/release-checklist.md)。
