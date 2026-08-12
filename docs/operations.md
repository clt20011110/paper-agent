# 运行与故障处理

Paper Agent 的 SQLite 数据库、冻结计划、grant、模型 release 和报告产物共同构成一次
运行的审计记录。它们必须位于受访问控制、可备份的项目目录中；不要把临时目录或共享
网络盘当作并发写入数据库。

## 每次生产运行前

1. 保存配置副本，确认其与将要批准的 QueryPlan 相符。
2. 运行 `paper-agent doctor --production-ready`，处理 database、模型 lock/release、Codex、
   磁盘、provider、plugin 与授权 skill 的 blocker。
3. 检查每个 grant 的 paper/collection scope、域名、purpose、`max_papers`、expiry 和 digest。
   过期、撤销或内容漂移的 grant 不能修补后继续使用，必须创建新 draft 并重新批准。
4. 记录操作者、配置 hash、plan hash、release hash 和预期预算。`--dry-run` 只验证本地绑定，
   不能证明远程站点、学校订阅、模型或浏览器会话可用。
5. 使用 schema-v3 Stage 2 release 时，确认 `PAPER_AGENT_STAGE2_HIDDEN_TRUST` 已指向部署控制的
   evaluator public-key manifest。该路径不能由 release bundle 指定；密钥、签名、轮换与事故边界按
   [hidden evaluator custody runbook](security/stage2-hidden-evaluator-custody.md) 执行。

## Stage 2 晋级与 release 组装

金标采样在隔离 evaluator 内先于标注完成。private snapshot 中标记的完整自然语料框不要求全量预标：冻结 seed
先从其中抽取 150 条 HIDDEN_REAL，记录真实纳入概率 150/N，且该步不得读取 curated labels；随后从剩余 paper-family 的 curated
pool 构建 DEV/HIDDEN_HARD。curated annotations 中的标签和难例标记只用于抽样分层，不是 gold label。
600 个 pair 选定后，全部样本才进行权威的两位独立标注和第三人仲裁：

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

HIDDEN_REAL 在 manifest 中记录真实 `sampling_probability=150/N`；DEV/HIDDEN_HARD 的配额、权重与 family
约束使其不存在单一纳入概率，因此记录 `null` 且不得用于逆概率加权。

无标签的 600-pair gold manifest 可进入 release；private snapshot、curated annotations、抽样 provenance（当前设计为 private）和原始标注 ledger 必须留在 evaluator custody。传给 promotion 的 `--private-labels` 精确覆盖这 600 条 manifest pair，不得以 snapshot 全量标签代替。

完成原始 ledger 后，在同一 evaluator custody 内运行：

```sh
paper-agent --dry-run stage2-sampling finalize-annotations \
  --gold-manifest /secure/evaluator-transfer/gold-manifest.json \
  --annotation-ledger /secure/evaluator/annotation-ledger.json \
  --private-labels-output /secure/evaluator/private-gold-labels.json
```

只有每个 pair 恰好两份固定标注者记录、每个分歧恰好一次固定第三人仲裁、仲裁前 QWK ≥ 0.75，且最终 gold 配额全部通过时，dry-run 才返回 `validated`。正式执行生成 no-replace 私有 labels；控制台和输出文件均不复制 annotator 身份或原始仲裁明细。

隔离 evaluator 优先运行 `stage2-evaluator promote`，而不是手工构造 hidden gate 结论。先在全局
`--dry-run` 下提供完整的 `--manifest`、`--private-labels`、重复的 `--candidate ID=PATH` 与
`--submission ID=PATH`、重复的 `--public-evidence ID=PATH`、incumbent/evaluator/run IDs、`--state-root`、key ID、RFC 3339
`--issued-at`、`--trust-manifest`、`--signing-key-file` 和不存在的 `--output`。dry-run 只读取并验证
公共 sampling manifest、schema-v2 candidates、公共质量/性能 evidence、trust 和 output 边界；它不读取 private labels、
submissions 或私钥，不创建 output，也不消费 marker。确认结构化 `status: "validated"` 后，只移除
`--dry-run` 并执行一次完全相同的命令。

私钥文件必须是当前用户拥有的 non-symlink regular file，精确模式 `0600`，不超过 16 KiB，内容为
一个 canonical unencrypted Ed25519 PKCS#8 PEM。当前 CLI 不直接调用 HSM/secret-manager signing
API。真实 promotion 在 `<state-root>/<gold-manifest-hash>.promotion.json` 原子记录一次性消费；gate
failure 也会生成 signed failure 并消费 holdout。如果 marker 已存在，即使 attestation 因后续签名或
写盘错误缺失，也不得删除 marker 或以新 evaluator/run ID 重试同一 holdout。

`stage2-evaluator attest --payload --signing-key-file --trust-manifest --output` 只签已有 public-safe
payload，不运行 hidden evaluation、不管理 marker；只有另一个已审阅的一次性 evaluator 已完成这些
职责时才可使用。release builder 只接收 public-safe attestation，并在 bundle 外显式提供 trust：

```sh
paper-agent --dry-run stage2-release assemble \
  --candidate /absolute/path/to/release-bundle/stage2-candidate-v2.json \
  --evidence /absolute/path/to/release-bundle/stage2-release-evidence.json \
  --trust-manifest /secure/deployment/hidden-evaluator-trust.json \
  --parity-oracle-trust /secure/deployment/parity-oracle-trust.json \
  --output /absolute/path/to/release-bundle/stage2-release.json
```

candidate 与 output 必须具有同一 parent，evidence 与其引用必须留在该 bundle root，trust 必须在
root 外，output 必须不存在。assembly dry-run 重算全部 public gates、验证 hidden attestation 和相同路径边界，但不写
output；成功后只移除 `--dry-run` 执行真实组装。private labels、raw submissions、私钥和 marker state
不能进入 bundle。组装后的生产加载才使用 `PAPER_AGENT_STAGE2_HIDDEN_TRUST`；它不替代 evaluator
或 assembler 的显式 `--trust-manifest`。

组装前的 public-gate 工件必须是原始 no-replace outputs：rationale 依次使用
`stage2-rationale derive-examples`、`freeze-worklist` 与人工完成后的 `import-worklist`；
`stage2-parity freeze-workload/run` 固定并运行
10,000-pair 数值 parity；`stage2-tuning select` 选择完整 3×3 的实测 batch/concurrency winner；最后
`stage2-release build-evidence` 绑定 gold、structured replay、rationale、10 个 parity artifacts、
benchmark（恰好六次 records）和 soak，带 hidden attestation 时才形成 final evidence。任何正式命令都先
用全局 `--dry-run` 完整验证且不写文件；输出已存在即停止。人工标签、真实本地模型测量和 sealed hidden
promotion 均不可由此绕过。

## 运行、观察和恢复

为一次 campaign 固定数据库和 `--run-id`。读取每条 JSON 输出的 `status`、`event_code`、
`run_id` 和 artifact path；不要根据终端文本判断完成。`incomplete`、`manual_required`、
`blocked` 和 `failed` 等非成功状态会返回非零退出码，调度器必须保留产物并进入人工检查或恢复流程。

```sh
paper-agent search audit --database /absolute/path/to/papers.sqlite3 \
  --crawl-run-id <crawl-run-id>
paper-agent resume --workflow /absolute/path/to/workflow.json \
  --database /absolute/path/to/papers.sqlite3 --workflow-run-id <workflow-run-id>
paper-agent verify-report --run-id <report-run-id> --config /absolute/path/to/research.yaml
```

迁移事实库时，先用 `paper-agent --dry-run import --format jsonl|csv|legacy-json
--input <FILE> --database <DB>` 验证全部记录，再去掉 `--dry-run`。正式导入在单一事务内
提交，重复导入相同规范 JSONL/CSV 不会制造重复 paper、source 或 membership。

`resume` 不接受单独的 `--run-id`：必须重传原始 `--workflow` manifest。多阶段清单必须使用
schema version 2 的 `from_step` 绑定；version 1 多阶段清单会被拒绝。其 FileRef（包括
config、plan、release、selection、policy 和 report inputs）必须仍在原清单目录中、摘要完全一致；
全局 `--config` 若显式传入，也必须匹配清单的 config FileRef。只恢复同一冻结输入的 recoverable
工作。plan、provider resolution、模型 revision、prompt/schema、grant、release 或 artifact hash
不匹配时，停止恢复并重新计划/批准；不要通过新 run ID 绕过门禁。

Download grant 若绑定 collection/selection snapshot，workflow Download step 也必须声明相同的
`scope_snapshots` type、ID、hash 和 collection ID。恢复会在观察既有 child run 前从 FileRef 或
SQLite 重新构造 snapshot，并重新检查 grant、成员关系、撤销和有效期；任何漂移都会在 provider
probe/fetch 或浏览器队列副作用前停止。未声明该字段的旧 manifest 仍可读取，但不能借此运行一个
snapshot-scoped grant。

typed Download workflow 当前不承载 authorized browser handoff 文件，因此只允许 public provider
grant；遇到 `provider=authorized_skill` 或 skill/dependency digest 绑定会提示改用独立
`paper-agent download`。Filter 来源在 adapter 内先解析为 exact paper IDs，Stage 3 只接收该集合，
并把原 `filter_run_id` 作为 lineage 保存；两者之间的 Filter 行变化不能扩大 scope。空静态 ID
列表同样直接拒绝，不读取其他 Stage 2 run。

Stage 4 在真正构造 Codex 调用前会持久领取一次性 dispatch。领取后若进程丢失、租约过期或
返回结果无法确认，该论文与 workflow step 会进入 `failed_terminal`/`uncertain_terminal`；
同一 run 的 `resume` 只返回该终态，绝不会再次付费调用。先检查调用审计和已有产物，再由操作者
明确创建新 run；不要反复 resume 期待它自动重发。Stage 3/4 的 grant 有效期只使用运行主机的
UTC 时钟，CLI 不接受可回拨的 `--now`。

收到 SIGINT 或 SIGTERM 时，typed workflow 不会中断正在运行的 stage adapter；它在该 stage 返回后
于下一 stage 边界写入可恢复 checkpoint。等待 JSON 结果中的 `status` 再决定是否调用相同
`--workflow` 和 `--workflow-run-id` 的 `resume`。

schema version 2 可把 Analyze 精确绑定到当前 Download step，但完整动态链在 Analyze 结束。
随后以完成态原 workflow 的 ID 执行 `report prepare-inputs --workflow-run-id`；保存返回的
`handoff_id`，用 `report --plan-only --handoff-id` 编译计划，再用
`report approve --handoff-id --workflow-config --workflow-manifest` 同时批准计划并登记独立的单阶段
Report workflow。新 config 必须启用 summary，固定 approved ReportPlan path/hash，并与 handoff 使用
相同 database/output root；prepare 时使用的 artifact root 会作为 manifest-relative DirectoryRef
冻结，Report adapter 不会把 report output root 猜成 analysis artifact root。最后以 approve 返回的
`manifest_path` 和 `report_workflow_run_id` 调用
普通 `run` 或 `resume`。handoff 只接受完整成功的 Search → Filter → Download → Analyze workflow；
Stage 4 未完成或失败时必须先恢复上游。ReportPlan 要求 membership 与实际 corpus 完全一致，因此
不能在 crawler 运行前把 Report 塞进同一 manifest；解析器会拒绝这种清单。Stage 4b 通过
`report_one_shot_runs` 为 report run 原子预约唯一 dispatch。尚未预约时可在相同冻结输入上恢复
preflight；预约后并发 worker 或 `resume` 只能观察状态或复用已证明持久化的结果。若 Codex 已启动后
发生 timeout、连接中断或结果不确定，旧 run 进入终态且不得重发；再次调用必须新建并批准 ReportPlan
与 report run。

授权下载只在 CLI 已输出 `authorized_queue_path` 后交给 `download-authorized-papers`。生成该队列时
需要有效 attended grant，以及 `--authorized-skill-queue`、`--authorized-skill-output`、至少一个
`--authorized-skill-root` 和 `--authorized-skill-zip`。ZIP、安装内容、dependency lock 或 grant digest
漂移会 fail closed；`--authorized-skill-audit` 只用于已审阅的显式覆盖，省略时使用内置 audit。
CLI 不启动或接管浏览器会话。

第一次 download 的 `manual_required` 表示 immutable CSV 已冻结等待 handoff，不是完成。不得编辑、
重排或追加该 CSV。使用 digest 匹配的 skill 依次执行 `plan`、小批量 `next --unscanned --limit 2`、
fixed browser pass、每批 `stage`，最后统一 `audit`；fixed pass 每篇使用 30 秒延迟与 5 秒 jitter，
每批使用新 event JSONL。UI fallback 只能在完整 fixed pass 和 audit 后作为第二轮处理。CAPTCHA、403、
429、access denied、登录失效、缺少/歧义授权 PDF link 或 `stopQueue: true` 必须立即停止，让用户在
同一可见浏览器修复，不能猜 selector、读取会话材料或绕过限制。

每个待导入 DOI 必须有 `complete`/`complete_no_si` final ledger，article 与全部已发现 PDF SI 通过
校验并记录 SHA-256。然后以完全相同的 config、run ID、database、selection、grant、queue、output、
root、ZIP 和 audit 参数重跑原 download 命令；CLI 重验全部绑定后只导入 ledger 匹配的
`article.pdf`。队列漂移、`manual_required`、缺 article 或缺已发现 SI 都必须保留为人工状态，不能
宣称完成或用新参数覆盖原 run。

报告执行则使用已批准的 ReportPlan bundle：
`report --plan` 还须有 database、output root 以及 `--policy` 或相同 v2 config 的 summary policy。
`--processing-grant` 始终写作 `ARTIFACT_SHA256=GRANT_ID`。
全部 Luna 报告必须在唯一输入包中各出现一次；full-payload 预算、授权或 coverage 门禁失败时必须在
dispatch 前以 0 次 Sol 调用保留 `incomplete`。门禁通过后只允许一次 `one_shot_report`，其后仅运行
本地 deterministic normalize、verifier 和 audit；blocker/major、无效输出或不确定调用结果均不得
触发 Sol retry/repair/reaudit，也不得更新 `latest`。

## 授权与安全事件

- 遇到 CAPTCHA、403、429、登录修复或未知下载链接：停止该 handoff，不读取 cookie、密码、
  OTP、token 或页面正文；记录 paper/candidate ID 和非敏感错误，再由用户在可见浏览器处理。
- 发现 provider/plugin/skill/model digest 漂移：禁用该组件和绑定旧 digest 的授权，重新审计后
  新建 grant/release。不得在运行中升级模型或自动云回退。
- 怀疑凭据泄露：立即在供应商侧撤销/轮换，清理环境注入点与 CI secrets，审计日志是否只含名称
  和 hash；不要把密钥粘贴到 issue、报告或聊天记录。

## SQLite 备份、恢复和保留

保持 WAL 文件与数据库位于本地、可用空间充足的磁盘。优先在暂停 writer 后使用 SQLite backup
API 创建一致快照，例如：

```sh
python - <<'PY'
import sqlite3
source = sqlite3.connect('/absolute/path/to/papers.sqlite3')
target = sqlite3.connect('/absolute/path/to/backups/papers.sqlite3')
with target:
    source.backup(target)
target.close()
source.close()
PY
```

备份后在隔离位置以只读方式打开并运行 audit/导出校验。恢复时先保留当前目录的完整副本，再以
一致备份替换数据库和关联产物；不要手工编辑 SQLite 表、approval 或 grant。保留不可变
QueryPlan、ReportPlan、grant 事件、source/search audit、模型调用 metadata、report sidecar、
coverage ledger、diff 与对应配置/release，直至组织的审计保留期结束。

## 容量、升级和告警

在 Stage 3/4/4b 前预留 PDF、文本、artifact 和 SQLite WAL 的空间，并将 doctor 的磁盘检查
视为硬门。对请求数、错误率、Stage 2 adjudicator 比例、下载失败、Codex 调用/令牌预算和未完成
coverage 设置告警。升级依赖、provider manifest、模型、prompt、schema、policy 或 skill 后，运行
离线测试和 `doctor`；任何 hash 漂移都需要新的计划/release/grant，而不是原地覆盖历史 run。

### 稳定告警码

命令单行 JSON 中的 `alarm_codes` 是稳定机器码；解释告警时同时读取 `status`、对应指标和
typed `error`，不能只匹配终端文本。

| 告警码 | 触发与状态 | 操作员动作 |
| --- | --- | --- |
| `stage2.adjudicator_share_exceeded` | 任一 Stage 2 子 run 的 adjudicator share `> 15%`；`> 30%` 为 `severe`。该容量告警本身不改变筛选决定，也不单独把 run 置为 `incomplete`。 | 检查 `stage2.max_run_adjudicator_share`、`stage2.run_details` 和路由原因；不得扩大自动拒绝区间。长期超过 30% 时重校 query/阈值或扩容，并创建新 release/plan。 |
| `stage2.error_rate_exceeded` | filter 单 run 或 search campaign 的终态技术错误率 `>= 0.5%`；search 顶层按 `error_count / screened_count` 聚合。benchmark 的终态 case 错误率或 oMLX service-request 失败率任一 `>= 0.5%` 也触发。相关命令返回 `incomplete`。 | 检查 `error_count`、`error_rate`、`max_run_error_rate` 和 `run_details`，benchmark 另查两类 request failure rate。成功重试不计终态错误，但失败 service attempt 仍计入 service rate。修复服务、schema 或校准后创建新 run；同一 run 的 terminal decision 不会因 `resume` 重发。 |
| `stage2.memory_watermark_exceeded` | `benchmark-stage2 measure` 的峰值 `> 28 GiB`、macOS memory-pressure critical，或热身后出现无界增长；返回 `incomplete`。恰好 28 GiB 不越线。 | 确认 RSS 覆盖 runner 和全部 oMLX PID；按 64 → 32 → 16 下调 batch，保存失败 record，并用新 benchmark run 重测。 |
| `report.codex_budget_exhausted` | 全量 one-shot prompt 会越过批准的 ReportPlan 输入上限，或恢复时发现已持久化 budget error；preflight 返回 `incomplete` 且 dispatch=0。 | 对比 `codex_budget.calls_reserved`、`input_tokens_reserved` 与两个 approved limits，并读取 typed `error`。已预约或已 dispatch 的 run 不因 `resume` 或扩预算重发；扩预算必须批准新 ReportPlan 并创建新 report run，禁止修改 SQLite ledger。 |

### Provider rate/credit 审计

`paper-agent search audit --database <DB> --crawl-run-id <ID>` 的
`provider_request_attempts[]` 记录实际 request reservation、charge、status 和 error；
`totals.requests_made`、`candidates_returned` 与 `candidates_accepted` 是 campaign 汇总。
`provider_rate_limits[]` 只展开实际持久化的 allowlist response headers，包含
`provider`、`query_id`、`request_index`、`query_hash` 和小写 `rate_limit` 字段。

`totals.provider_rate_limit_observations = 0` 只表示 provider、cache/snapshot 或 replay 没有提供
可持久化的额度 header，不表示额度无限。遇到 429/5xx 时同时检查 request attempt 的 error；观察到
remaining/credit 接近耗尽或 `Retry-After` 时，停止扩大 fan-out，按冻结的退避和限流配置恢复，或改用
用户预先批准的 snapshot。不得通过提高并发绕过全局 provider 限流。审计只允许 rate/quota/credit
header；`Authorization`、`Cookie`、`Set-Cookie` 等凭据和会话信息不得进入产物。
