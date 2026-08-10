# 从 v1 配置迁移到 v2

v2 是唯一可执行的配置格式。它把检索计划、下载授权、模型处理授权和报告计划分开冻结；示例文件都不含密钥、grant ID 或批准后的内容 hash。

## 先做只读迁移

先通过受支持的 CLI 查看迁移结果。它读取旧 YAML 并输出结构化 JSON，但不写入文件：

```sh
paper-agent migrate-config --input /absolute/path/to/old.yaml
```

检查 JSON 中的 `field_mappings`、`warnings` 和 `unmigrated`。迁移不猜测关键词、旧正则的
细节、下载超时或模型选择；所有 `unmigrated` 都必须由操作者处理。确认后才写入**新路径**：

```sh
paper-agent migrate-config \
  --input /absolute/path/to/old.yaml \
  --write /absolute/path/to/new-v2.yaml

paper-agent --config /absolute/path/to/new-v2.yaml doctor
cp configs/query_draft.example.yaml /absolute/path/to/query-draft.yaml
# 编译前审阅 query-draft.yaml，并替换其中的 Stage 2 hashes。
paper-agent search plan \
  --input /absolute/path/to/query-draft.yaml \
  --output-root /absolute/path/to/paper-research
```

保留原 YAML 和输出 JSON 作为迁移审计记录。不要覆盖旧文件，也不要把 `content_hash: null`
模板直接用于无人值守运行；必须检查并批准新生成的 QueryPlan。`paper_agent.legacy` API 是供
集成代码使用的等价接口，不是日常运维入口。

迁移后的文档会经过 `config-v2.schema.json` 验证。请逐项处理 `unmigrated`；迁移不会猜测关键词、旧的正则匹配细节、下载超时，或把旧的模型选择照搬到 v2。

## 字段与行为变化

| v1 | v2 | 处理 |
| --- | --- | --- |
| `topic`、`output_dir` | `project.*`、`storage.*` | SQLite/WAL 成为运行时事实源；JSONL/CSV 为导出。 |
| conference/journal 名称 | `sources.plan_defaults.venues[].descriptor` | 使用版本化 venue descriptor；TCAD 以期刊日期范围表示。 |
| `arxiv.save_to_database` | `include_arxiv_candidates` | 该开关决定 arXiv-only 候选是否可进入后续阶段。 |
| hybrid/semantic filter | 固定 cascade | 本地 Stage 2 决定性规则、reranker 和灰区裁决替代旧路径。 |
| download settings | `download` policy 与 grant defaults | 默认不启用授权浏览器；配置草稿不是实际授权。 |
| analysis/summary model | 固定 Stage 4/4b profile | 模型、schema、prompt 和输入 hash 由 run 记录绑定。 |

## 模型与环境约定

示例把模型路由写进 v2 可验证字段，不能被默认模型覆盖：

- Stage 2 完全在本机 oMLX：`omlx_rerank` 使用已验收的 reranker，`omlx_chat` 使用锁定 revision 的 9B Qwen 裁决器。运行前应按 `configs/stage2/models/*.lock.json` 安装并验证发布工件、校准器和阈值工件；这些运行时 release 输入不在 schema 中，因此不能只凭示例配置视为已验收。
- Stage 3 的可选授权下载使用 `stage3_authorized_luna`、`gpt-5.6-luna` 和 low reasoning。必须先安装并审计下载 skill，创建精确的 download/data-sharing grant，再把 `authorized_skill.enabled` 设为 true；浏览器授权默认为 attended。
- Stage 4 分析固定为 `stage4_analysis_luna`、`gpt-5.6-luna` 和 medium reasoning。全文进入远程处理前必须有 artifact-scoped processing grant；否则只能使用另行允许的摘要。
- Stage 4b 报告固定为 `stage4b_summary_sol`、`gpt-5.6-sol` 和 high reasoning。派生材料需要独立 lineage-scoped grant，且必须有冻结并批准的报告计划。

服务凭据只通过示例中列出的环境变量名称提供，例如 `CROSSREF_MAILTO`、`SEMANTIC_SCHOLAR_API_KEY`、`OPENALEX_API_KEY`、`NCBI_API_KEY`、`NCBI_EMAIL` 和 `UNPAYWALL_EMAIL`。不要把值写入 YAML、报告、SQLite 导出或版本库。

## 数据库升级与 Stage 4 恢复

升级前先按运维文档备份 SQLite 与 artifact 目录。migration 16 会为 Stage 4 建立一次性付费调用
账本；旧库中状态为 `failed` 或 `running`、且无法证明调用未发生的分析会保守迁移为
`failed_terminal`，不会在 `resume` 时重新调用 Codex。旧的完整结果仍可只读恢复；旧的
`incomplete` 记录只有在原 processing policy hash 可核验一致时才可继续。policy、prompt、schema
或输入发生漂移时必须建立新 run，不能借数据库升级覆盖原审计身份。

Migration 19 增加 `stage3_paper_results`，供 `stage3-cli-v2` 保存逐论文聚合状态并恢复
`downloaded/not_available/failed_terminal` 终态。迁移不会从旧 `download_attempts` 回填
`not_available/failed_terminal`，因为单个 URL 的失败不能安全代表整篇论文所有候选均已耗尽。
升级后默认选择会创建新的 v2 run；显式要求复用冻结为 v1 的 run ID 会因输入或实现版本不匹配
而拒绝。旧的已下载 artifact 仍可作为只读 Stage 4 输入。

## 选择示例

`example_config.yaml` 是跨会议、期刊和 arXiv 的完整示例，报告默认开启。`configs/abstract_focus.yaml`、`configs/journal_smoke.yaml` 和 `configs/smoke_supported.yaml` 保留其原有的窄范围 smoke 场景，并关闭报告生成。每一个示例都要求先生成并批准 QueryPlan；`content_hash: null` 表示它只是初始模板，不能用于无人值守执行。
