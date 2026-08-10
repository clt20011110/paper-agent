# 从 v1 配置迁移到 v2

v2 是唯一可执行的配置格式。它把检索计划、下载授权、模型处理授权和报告计划分开冻结；示例文件都不含密钥、grant ID 或批准后的内容 hash。

## 先做只读迁移

迁移逻辑在 `paper_agent.legacy.migrate_legacy_yaml`：它读取旧 YAML，返回 `MigrationReport`，但不会写入文件。报告包含 `converted_config`、字段映射、警告和无法自动表示的字段。确认结果后，才调用 `paper_agent.legacy.write_migrated` 写入一个新路径。

```python
from pathlib import Path

from paper_agent.legacy import migrate_legacy_yaml, write_migrated

report = migrate_legacy_yaml(Path("old.yaml"))
print(report.field_mappings)
print(report.warnings)
print(report.unmigrated)
write_migrated(report, Path("new-v2.yaml"))
```

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

## 选择示例

`example_config.yaml` 是跨会议、期刊和 arXiv 的完整示例，报告默认开启。`configs/abstract_focus.yaml`、`configs/journal_smoke.yaml` 和 `configs/smoke_supported.yaml` 保留其原有的窄范围 smoke 场景，并关闭报告生成。每一个示例都要求先生成并批准 QueryPlan；`content_hash: null` 表示它只是初始模板，不能用于无人值守执行。
