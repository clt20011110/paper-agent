# Stage 1 当前进展与未完成工作

更新日期：2026-08-14  
代码基线：`feature/crawler-adapters`（2026-08-14 Stage 1 enrichment 更新）
范围：Stage 1 独立接口、2016–2025 venue membership census，以及标题、作者、摘要、DOI、PDF URL 的字段补全。

## 结论先行

Stage 1 已经具备可独立运行和审计的生产骨架，但还没有达到“目录内每个 venue 在 2016–2025 每一篇论文的字段全部闭合”的最终门槛。

- **原有论文集合（membership）已闭合**：此前 35 个 venue、2016–2025 全部适用年份均完成 terminal cursor、稳定 ID 去重和 parser 收支校验；共 374,784 条唯一记录。新接入的 8 个 venue 已完成 manifest/fixture contract，但尚未把 fixture 当作 live decade census 证据。
- **独立接口已完成**：可以输入一个或多个 venue、起止年份，输出 JSONL metadata 和 receipt；默认 fail-closed，不会把缺字段的结果误标为 complete。
- **可扩展架构已完成**：venue descriptor、provider manifest、primary census 和 enrichment provider 已解耦；新增同类来源主要通过 YAML descriptor 扩展。
- **代表年份字段验收大部分已完成**：会议类已有多组真实 complete receipt；部分 EDA、期刊和历史 ICLR/Science 仍有明确缺口。
- **无人值守回归已通过**：完整 pytest 回归通过；本次工作区变更在最终回归后提交并推送到当前分支。
- **全目录字段矩阵审计接口已完成**：`stage1 matrix` 会枚举每个 venue×year，显式报告 `complete`、`not_applicable`、`missing_receipt`、`failed`、`unproven` 和 `conflict`，并绑定输入 receipt 哈希。

因此，当前可以把 Stage 1 当作“可用的独立 metadata census 接口”使用；不能把整个十年字段矩阵宣称为已完成。

## 1. 已完成的能力

### 1.1 独立 Stage 1 接口

CLI：

```bash
paper-agent stage1 list-venues

paper-agent stage1 collect \
  --venue neurips \
  --venue icml \
  --year-from 2016 \
  --year-to 2025 \
  --contact operator@example.org \
  --max-workers 4 \
  --output build/stage1/papers.jsonl
```

Python API：

```python
from paper_agent.stage1 import Stage1Request, collect_stage1_metadata

request = Stage1Request(("icml", "neurips"), 2016, 2025, max_workers=4)
result = collect_stage1_metadata(request, catalog=catalog,
                                 adapter_factory=adapter_factory)
```

接口特征：

- `venue` 可重复传入；年份范围为闭区间。
- 默认模式只有在所有请求单元满足严格门禁时才发布 JSONL；receipt 始终写出。
- `--allow-incomplete` 仅用于诊断，receipt 仍保持 `incomplete`。
- receipt 保存 membership 数量、terminal cursor、稳定 ID 重复、parser 收支、字段 coverage、字段状态、provenance、provider warning 和响应哈希。
- enrichment 只允许对 primary census 已冻结的 stable ID 回填字段，不得增加或删除论文。
- `pdf_url` 代表候选位置，不代表已授权下载；Stage 3 负责实际访问、授权和 PDF 校验。

### 1.2 当前 venue 目录与来源架构

当前目录共 43 个 venue：20 个会议、23 个期刊。新增的 8 个目录项为 TMLR、TPAMI、DATE、ASP-DAC、ISPD、TODAES、TVLSI、JSSC；它们已接入现有 Crossref serial/DBLP TOC 通用适配器并通过离线及受控 transport fixture contract。

- 会议：AAAI、ACL、AISTATS、COLING、COLT、CoRL、CVPR、DAC、EMNLP、ICCAD、ICCV、ICLR、ICML、IJCAI、NAACL、NeurIPS、UAI。
- 期刊：ACS Central Science、Angewandte Chemie、Cell、Chemical Science、JACS、JCIM、JCTC、JMLR、Nature Biomedical Engineering、Nature Biotechnology、Nature Catalysis、Nature Chemistry、Nature Communications、Nature Computational Science、Nature Machine Intelligence、Nature Synthesis、Science、TCAD。

已接入或验证的来源族包括：

- AAAI OJS、ACL Anthology、PMLR、NeurIPS Proceedings、OpenReview/ICLR、IJCAI Proceedings、CVF Open Access。
- DBLP 完整 proceedings TOC，用于 ICLR、DAC、ICCAD 等会议的 membership census。
- Crossref ISSN serial census，用于化学、Nature、Cell、Science、TCAD 等期刊。
- Crossref、Europe PMC、Semantic Scholar、OpenAlex、arXiv、DBLP 年度快照和 Nature 官方文章页等字段 enrichment 来源。

PMLR 已支持 bulk frontmatter 失败时的逐篇官方 detail fallback；期刊 DOI batch 失败时会保留已取得字段；Semantic Scholar/OpenAlex/arXiv 的临时限流不会清空 primary membership 或已有字段；超过 30 秒的 `Retry-After` 会快速失败并留下可恢复 warning，避免公共配额耗尽时挂起数小时。

## 2. 十年 membership 验收

统一证据：[`docs/acceptance/stage1-decade-census-20260812.json`](acceptance/stage1-decade-census-20260812.json)（仅覆盖接入新 venue 前的 35 个 venue）。新增矩阵审计的本机运行证据见 `/private/tmp/stage1-field-matrix-20260814-v4.json`，其 machine hash 为 `1fd8599da7330099ea81edf4f8a2ca9c37c5a16ba94fd2a1f40940fa0ffbaed4`。

| 门禁 | 当前结果 | 说明 |
|---|---:|---|
| 已有 decade census venue 数量 | 35/35 | 新增 8 个 venue 尚未完成 live decade census |
| 当前目录 venue 数量 | 43 | 20 个会议、23 个期刊；全部有 descriptor/acceptance |
| 适用年份 | 2016–2025 | 每个适用 `venue × year` 到达 terminal cursor |
| 已验证唯一记录 | 374,784 | 仅对应已完成的 35-venue census |
| parser 收支 | 完整 | raw、rejected、explicitly excluded 可审计 |
| 字段全覆盖 | 未完成 | membership 完整不等于摘要/DOI/PDF 全部存在 |

这项验收证明“论文集合没有被分页截断、重复或静默丢弃”，不证明每条记录的可选字段都已补齐。

## 3. 已完成的代表年份字段验收

下表是已经跑过真实严格字段流程的代表年份；`complete` 只对该行的该年度成立，不代表该 venue 的十年字段矩阵已经完成。

| Venue/year | Records | 摘要 | DOI | PDF URL | 当前结果 |
|---|---:|---:|---:|---:|---|
| AAAI 2020 | 1,865 | 1,865 | 1,865 | 1,865 | complete |
| ICLR 2020 | 687 | 687 | 依法未分配 | 687 | complete |
| ICLR 2024 | 2,260 | 2,260 | 依法未分配 | 2,260 | complete |
| ICLR 2025 | 3,704 | 3,704 | 依法未分配 | 3,704 | complete |
| IJCAI 2016 | 651 | 651 | 依法未分配 | 651 | complete |
| IJCAI 2024 | 1,048 | 1,048 | 1,048 | 1,048 | complete |
| ICML 2024 | 2,610 | 2,610 | 依法未分配 | 2,610 | complete |
| NeurIPS 2016 | 569 | 569 | 依法未分配 | 569 | complete |
| NeurIPS 2024 | 4,493 | 4,493 | 4,493 | 4,493 | complete |
| JMLR 2024 | 422 | 422 | 依法未分配 | 422 | complete |
| ACL 2024 | 984 | 984 | 984 | 984 | complete |
| COLING/LREC-COLING 2024 | 1,567 | 1,567 | 未注册 | 1,567 | complete |
| CVPR 2024 | 2,716 | 2,716 | 2,715 + 1 未注册 | 2,716 | complete |
| ICCV 2025 | 2,701 | 2,701 | 2,700 + 1 未注册 | 2,701 | complete |
| DAC 2024 | 370 | 370 | 370 | 370 | complete |
| JCIM 2024 | 805 | 740 + 65 不适用 | 805 | 805 | complete |
| Nature Machine Intelligence 2024 | 184 | 166 + 18 不适用 | 184 | 184 | complete |
| Nature Biotechnology 2024 | 468 | 333 + 135 不适用 | 468 | 468 | complete |
| ACS Central Science 2024 | 278 | 201 + 77 不适用 | 278 | 278 | complete |
| Nature Biomedical Engineering 2024 | 159 | 151 + 8 不适用 | 159 | 159 | complete |
| Nature Catalysis 2024 | 193 | 158 + 35 不适用 | 193 | 193 | complete |
| Nature Chemistry 2024 | 291 | 279 + 12 不适用 | 291 | 291 | complete |
| Nature Communications 2024 | 10,926 | 10,434 + 492 不适用 | 10,926 | 10,926 | complete |
| Nature Computational Science 2024 | 167 | 151 + 16 不适用 | 167 | 167 | complete |
| Nature Synthesis 2024 | 257 | 211 + 46 不适用 | 257 | 257 | complete |
| Chemical Science 2024 | 2,091 | 1,874 + 217 不适用 | 2,091 | 2,091 | complete |
| JCTC 2024 | 909 | 850 + 59 不适用 | 909 | 909 | complete |
| Cell 2024 | 548 | 520 + 27 不适用，1 未解决 | 548 | 548 | incomplete |
| TCAD 2024 | 429 | 385 + 37 不适用，7 未解决 | 429 | 429 | incomplete |
| Angewandte Chemie 2024 | 5,170 | 4,817 + 353 不适用 | 5,170 | 5,170 | complete |
| COLT 2024 | 169 | 169 | 依法未分配 | 169 | complete |
| CoRL 2024 | 264 | 264 | 依法未分配 | 264 | complete |
| EMNLP 2024 | 1,447 | 1,447 | 1,447 | 1,447 | complete |
| NAACL 2024 | 632 | 632 | 632 | 632 | complete |
| UAI 2024 | 201 | 201 | 依法未分配 | 201 | complete |
| AISTATS 2024 | 547 | 547 | 依法未分配 | 547 | recovery complete |

“依法未分配”“未注册”“不适用”都以 `null + legitimately_absent + provenance` 表达，不会填入虚构 DOI 或摘要。

## 4. 当前未完成与受阻项目

### P0：已有明确字段缺口

| 项目 | 现状 | 缺口 | 主要原因 | 正确处理 |
|---|---|---:|---|---|
| ICLR 2017 | 198 条 membership/PDF 已确认 | 28 条摘要 | 无现代年度 bulk JSON；无人值守 OpenReview 返回 challenge | 保持 `incomplete`；寻找可审计官方 metadata/export，不绕过 challenge |
| ICCAD 2024 | 239 条 DOI/PDF 已确认 | 5 条摘要 | ACM Cloudflare、Semantic Scholar 429、OpenAlex 公共配额耗尽 | 配额恢复或取得合规的官方批量来源后逐条补齐 |
| Science 2024 | 2,039 条 DOI/PDF 已确认 | 298 条摘要 | publisher 页面出现无人值守 Cloudflare challenge | 优先寻找 AAAS 官方 metadata/export；不逐篇绕过挑战 |

ICLR 2017 的官方 OpenReview 页面在已登录浏览器中可见，并不等于无人值守 Stage 1 receipt 已完成；人工可见结果尚未作为自动化字段快照接入，因此不能把这 28 条直接标成 complete。

### P1：代表年份矩阵仍不完整

以下 venue 已完成十年 membership，但尚未完成本文要求的代表年份严格字段验收，或尚有代表年份缺少统一 receipt 留证：

- Cell、TCAD：PDF 候选已经分别达到 548/548、429/429；Cell 仍有 1 条、TCAD 仍有 7 条普通记录缺摘要。
- ACS Central Science、Chemical Science、JCTC、Nature Biomedical Engineering、Nature Catalysis、Nature Chemistry、Nature Communications、Nature Computational Science、Nature Synthesis 和 JACS 2024 代表年份均已通过；非研究/期刊元数据类条目被可审计规则标记为摘要不适用。

每个 venue 至少需要一个高产年份的全量 strict run，并将记录数、摘要、DOI、PDF、legitimately absent 分类、receipt hash 写入统一验收矩阵；Cell/TCAD 的 2024 代表年份已跑过，但仍保持 incomplete。

新增目录的第一轮真实联网验证如下；fixture contract 通过不等于下表的 live 结果：

| 新 venue/year | membership | 字段结果 | receipt |
|---|---:|---|---|
| DATE 2024 | 379/379 | 333 摘要，340 条摘要/PDF enrichment 未闭合 | `/private/tmp/stage1-date-2024-live.receipt.json` |
| ASP-DAC 2024 | 157/157 | 155 摘要，133 条字段未闭合 | `/private/tmp/stage1-aspdac-ispd-2024-live.receipt.json` |
| ISPD 2024 | 49/49 | 49/49 complete | 同上 |
| ISPD 2016–2025 | 355/355（10/10 年） | 2021–2023、2025 complete；其余年份因 timeout/429 或明确缺口保持 incomplete | `/private/tmp/stage1-ispd-2016-2025-live.receipt.json` |
| TMLR 2024 | 0（OpenReview challenge） | Crossref 404 已排除；已切换官方 `TMLR/-/Submission` rolling invitation，当前无人值守 API 返回 challenge | `/private/tmp/stage1-tmlr-2024-openreview.receipt.json` |
| TPAMI 2024 | 739/739 | 737 摘要，739 DOI/作者/PDF；2 条摘要仍未闭合 | `/private/tmp/stage1-tpami-2024-live.receipt.json` |
| TODAES 2024 | 99/99 | 98 摘要、99 DOI/PDF；Europe PMC/后备批次超时导致 1 条未闭合 | `/private/tmp/stage1-todaes-tvlsi-jssc-2024-live.receipt.json` |
| TVLSI 2024 | 269/269 | 232 作者、269 DOI；辅助批次超时，摘要/PDF 大量未闭合 | 同上 |
| JSSC 2024 | 409/409 | 347 作者、409 DOI；辅助批次超时，摘要/PDF 未闭合 | 同上 |

### P2：2016–2025 全字段矩阵

代表年份完成后，还需按全部适用 `venue × year` 跑十年字段矩阵。每个单元必须同时满足：

1. primary membership 与冻结的 decade census 完全一致；
2. enrichment 不增加、不删除 membership；
3. 摘要、DOI、PDF URL 要么 present，要么有来源证据的 `legitimately_absent`；
4. terminal cursor、parser 收支、provenance、请求/响应 hash 完整；
5. 429/timeout 可断点恢复，不要求从头重跑整年；
6. 使用 `paper-agent stage1 matrix` 输出机器可读总矩阵，至少区分 `complete`、`incomplete`、`not_applicable`、`missing_receipt` 和 `conflict`；当前 43-venue 运行结果为 202 complete、36 not_applicable、29 conflict、33 failed、10 unproven、120 missing_receipt。

### P3/P4：可靠性和发布收尾

- 持久化 journal DOI batch 的每个批次响应，支持失败批次重试。
- Europe PMC、Semantic Scholar、OpenAlex 和 arXiv 的超时/断路器错误现在会降级为可审计 warning，并继续尝试后备源；primary membership 不会因 enrichment 单点失败而丢失。
- 分 provider 统计成功率、429、timeout、重试次数和配额状态。
- 已由 document type 证明摘要不适用的条目不再调用辅助 API。
- 为 publisher HTML fallback 设置 venue 级开关和明确条款门禁。
- 增加大年份并行 soak，验证 QPS、连接池、缓存和恢复行为。
- 将字段验收矩阵变成单一机器可读 artifact，并由它生成 Markdown，避免手工数字漂移。
- 为 Stage 1 output/receipt schema 建立版本化兼容测试。

## 5. 证据与可复核入口

- 十年 membership：[`docs/acceptance/stage1-decade-census-20260812.json`](stage1-decade-census-20260812.json)
- 代表年份字段验收：[`docs/acceptance/stage1-field-enrichment-20260813.md`](acceptance/stage1-field-enrichment-20260813.md)
- 独立接口用法：[`docs/stage1-standalone.md`](stage1-standalone.md)
- 总规格与发布门禁：[`task.md`](../task.md)
- DAC receipt：`/private/tmp/stage1-dac-2024-openalex-final2.receipt.json`
- ICCAD receipt：`/private/tmp/stage1-iccad-2024-openalex-final.receipt.json`
- ICLR 2017 receipt：`/private/tmp/stage1-iclr-2017-reprobe.receipt.json`
- Nature Biotechnology receipt：`/private/tmp/stage1-nature-biotechnology-2024-p1-classified.receipt.json`
- JACS receipt：`/private/tmp/stage1-jacs-2024-p1-final.receipt.json`
- JACS classified receipt：`/private/tmp/stage1-jacs-2024-classified.receipt.json`
- ACS Central Science receipt：`/private/tmp/stage1-acs-central-science-2024-classified.receipt.json`
- Cell/TCAD receipt：`/private/tmp/stage1-cell-tcad-2024-classified2.receipt.json`
- AISTATS receipt：`/private/tmp/stage1-aistats-2024-p1-final.receipt.json`

`/private/tmp` 中的 receipt 是本机运行产物，不随 Git 提交；提交到仓库的 acceptance 文档记录了其摘要、状态和限制。

## 6. 建议执行顺序

1. 在 OpenAlex 配额恢复或取得合规的 Semantic Scholar/ACM 访问后，关闭 ICCAD 5 条摘要缺口。
2. 为 ICLR 2017 取得官方、可复现的批量 metadata/export；若只能浏览器人工访问，先设计带 hash 和 source URL 的显式 evidence snapshot，不得隐式混入 strict receipt。
3. 为 Science 找 AAAS 可批量审计的官方 metadata 路径。
4. 关闭 Cell 的 1 条、TCAD 的 7 条普通摘要缺口，并核实 RSC/Cell/IEEE 候选 PDF 在 Stage 3 的实际可访问性。
5. 扩展至全部 2016–2025 字段矩阵，生成统一 JSON/Markdown 报告。
6. 只有上述适用单元全部通过，才勾选 `task.md` 中的“EDA 与 Crossref 期刊字段补全”十年完成项。

## 7. 运行与验收命令

```bash
paper-agent stage1 list-venues

paper-agent stage1 collect \
  --venue <venue> \
  --year-from <start> \
  --year-to <end> \
  --contact <email> \
  --max-workers 4 \
  --output build/stage1/<venue>-<start>-<end>.jsonl

.venv/bin/pytest -q
```

正式发布前应同时检查 JSONL、`.receipt.json`、provider warning、field coverage 和 `git diff`；任何 unresolved required field 都必须保留为 `incomplete`，而不是通过 `--allow-incomplete` 改写状态。
