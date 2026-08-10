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

Stage 4 在真正构造 Codex 调用前会持久领取一次性 dispatch。领取后若进程丢失、租约过期或
返回结果无法确认，该论文与 workflow step 会进入 `failed_terminal`/`uncertain_terminal`；
同一 run 的 `resume` 只返回该终态，绝不会再次付费调用。先检查调用审计和已有产物，再由操作者
明确创建新 run；不要反复 resume 期待它自动重发。Stage 3/4 的 grant 有效期只使用运行主机的
UTC 时钟，CLI 不接受可回拨的 `--now`。

收到 SIGINT 或 SIGTERM 时，typed workflow 不会中断正在运行的 stage adapter；它在该 stage 返回后
于下一 stage 边界写入可恢复 checkpoint。等待 JSON 结果中的 `status` 再决定是否调用相同
`--workflow` 和 `--workflow-run-id` 的 `resume`。

schema version 2 可把 Analyze 精确绑定到当前 Download step，但完整动态链在 Analyze 结束。
随后用这些子运行的精确 ID 执行 `report prepare-inputs`，人工完成 plan-only 与 approve，并将
approved ReportPlan path/hash 固定到新 config，再启动独立的单阶段 Report workflow。ReportPlan
要求 membership 与实际 corpus 完全一致，因此不能在 crawler 运行前把 Report 塞进同一 manifest；
解析器会拒绝这种清单。Stage 4b 若在 `pipeline_runs=running` 时崩溃，workflow 会检查 reduce、
audit 与 audit-shard 的子租约：仍有效时保持 blocked，无有效租约时允许相同冻结输入进入协调器的
故障恢复逻辑。

授权下载只在 CLI 已输出 `authorized_queue_path` 后交给 `download-authorized-papers`。生成该队列时
需要有效 grant 及 `--authorized-skill-queue`、`--authorized-skill-output`、至少一个
`--authorized-skill-root`；CLI 不启动或接管浏览器会话。报告执行则使用已批准的 ReportPlan bundle：
`report --plan` 还须有 database、output root 以及 `--policy` 或相同 v2 config 的 summary policy。
`--processing-grant` 始终写作 `ARTIFACT_SHA256=GRANT_ID`。
单个论文、可选来源或可重试模型请求失败可以保留并让其他工作继续，但 required source 失败、
预算耗尽、coverage 缺失和 report audit blocker/major 必须保留 `incomplete`。

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
