# Stage 1 当前进展与剩余工作

更新日期：2026-08-14  
范围：独立 Stage 1 接口、2016–2025 venue membership census、标题/作者/摘要/DOI/PDF URL 等字段补全。  
不在本文范围：PDF 实际下载与授权（Stage 3）、论文语义筛选（Stage 2）、Luna/Sol 分析与综述（Stage 4/4b）。

## 1. 结论摘要

Stage 1 的“收录论文集合是否完整”与“每条论文的字段是否完整”是两个独立门禁，当前状态如下：

| 门禁 | 当前状态 | 结论 |
|---|---|---|
| 2016–2025 membership census | 已完成 | 35/35 个 venue 已验证；374,784 条唯一记录；所有适用年份到达 terminal cursor；稳定 ID 无重复；parser 收支闭合 |
| 独立接口 | 已完成 | 可输入一个或多个 venue 和年份范围，输出 JSONL metadata 与审计 receipt；无需进入 Stage 2–4 |
| 可扩展 provider/descriptor 架构 | 已完成 | 新增同平台 venue 主要通过 YAML descriptor；官方来源、注册表和补全来源按角色解耦 |
| 基础字段输出 | 已完成 | 标题、作者、日期、venue、landing URL、DOI、PDF URL、摘要及逐字段状态/provenance 均有统一结构 |
| 全部 venue × 2016–2025 字段全覆盖 | 未完成 | membership 已闭合；字段矩阵仍有 ICLR 2017、ICCAD 2024、Science 2024 及尚未留证的期刊年份缺口 |

核心原则保持不变：找不到的字段不伪造；确实不存在的 DOI/摘要以 `legitimately_absent` 加证据表达；辅助 enrichment 失败不得改变 primary membership 集合。

## 2. 已完成的系统能力

### 2.1 独立 Stage 1 接口

CLI 已支持：

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

默认 fail-closed：只有所有请求单元均满足严格门禁时才发布 JSONL；receipt 始终生成。`--allow-incomplete` 只用于诊断，不会把不完整结果标为 complete。

Python API 也已提供：

```python
from paper_agent.stage1 import Stage1Request, collect_stage1_metadata

request = Stage1Request(("icml", "neurips"), 2016, 2025, max_workers=4)
result = collect_stage1_metadata(
    request,
    catalog=catalog,
    adapter_factory=adapter_factory,
)
```

### 2.2 可扩展来源架构

当前 catalog 共 35 个 venue：17 个会议、18 个期刊。

- 会议：`aaai`、`acl`、`aistats`、`coling`、`colt`、`corl`、`cvpr`、`dac`、`emnlp`、`iccad`、`iccv`、`iclr`、`icml`、`ijcai`、`naacl`、`neurips`、`uai`。
- 期刊：`acs_central_science`、`angewandte_chemie`、`cell`、`chemical_science`、`jacs`、`jcim`、`jctc`、`jmlr`、`nature_biomedical_engineering`、`nature_biotechnology`、`nature_catalysis`、`nature_chemistry`、`nature_communications`、`nature_computational_science`、`nature_machine_intelligence`、`nature_synthesis`、`science`、`tcad`。

已实现的主要 primary adapter/family：

- AAAI OJS、ACL Anthology、PMLR、NeurIPS Proceedings、OpenReview/ICLR、IJCAI Proceedings、CVF Open Access。
- DBLP 完整 proceedings TOC，用于 ICLR、DAC、ICCAD 等声明的领域权威 membership census。
- Crossref ISSN serial census，用于化学、Nature、Cell、Science、TCAD 等期刊。
- 独立 enrichment：Crossref、Europe PMC、Semantic Scholar、arXiv、DBLP 年度快照、Nature 官方文章页等。

新增 venue 时，原则上只需新增/修改 descriptor、acceptance 和 provider manifest；只有出现全新上游协议时才需要新增 adapter。

### 2.3 完整性与审计

每个 `venue × year` receipt 已记录：

- expected/observed count、terminal cursor、稳定 ID 重复情况；
- parser raw/rejected/explicitly excluded 收支；
- 字段覆盖率与 `present`、`legitimately_absent`、`unavailable_at_primary` 等状态；
- provider warning、请求/响应 SHA-256、字段 provenance；
- enrichment 是否在不增加、不删除 primary membership 的前提下完成。

2016–2025 membership census 的统一证据为 `docs/acceptance/stage1-decade-census-20260812.json`：35/35 venue complete，共 374,784 条唯一记录，重复稳定 ID 为 0。

## 3. 字段补全的真实验收进展

下表只列已经进行过真实全量字段验证的代表年份；“complete”表示该次严格运行完成，不等价于该 venue 的十年字段均已验证。

| Venue/year | Records | 摘要 | DOI | PDF URL | 状态 |
|---|---:|---:|---:|---:|---|
| AAAI 2020 | 1,865 | 1,865 | 1,865 | 1,865 | complete |
| ICLR 2020 | 687 | 687 | 687 不分配 | 687 | complete |
| ICLR 2024 | 2,260 | 2,260 | 2,260 不分配 | 2,260 | complete |
| ICLR 2025 | 3,704 | 3,704 | 3,704 不分配 | 3,704 | complete |
| ICLR 2017 | 198 | 170，缺 28 | 198 不分配 | 198 | incomplete |
| IJCAI 2016 | 651 | 651 | 651 不分配 | 651 | complete |
| IJCAI 2024 | 1,048 | 1,048 | 1,048 | 1,048 | complete |
| ICML 2024 | 2,610 | 2,610 | 2,610 不分配 | 2,610 | complete |
| NeurIPS 2016 | 569 | 569 | 569 不分配 | 569 | complete |
| NeurIPS 2024 | 4,493 | 4,493 | 4,493 | 4,493 | complete |
| JMLR 2024 | 422 | 422 | 422 不分配 | 422 | complete |
| ACL 2024 | 984 | 984 | 984 | 984 | complete |
| COLING/LREC-COLING 2024 | 1,567 | 1,567 | 1,567 未注册 | 1,567 | complete |
| CVPR 2024 | 2,716 | 2,716 | 2,715 + 1 未注册 | 2,716 | complete |
| ICCV 2025 | 2,701 | 2,701 | 2,700 + 1 未注册 | 2,701 | complete |
| DAC 2024 | 370 | 370 | 370 | 370 | complete；OpenAlex 精确 DOI/标题回填 1 条 |
| ICCAD 2024 | 239 | 234 | 239 | 239 | incomplete；5 条受 ACM Cloudflare、Semantic Scholar 429、OpenAlex 配额影响 |
| JCIM 2024 | 805 | 740 + 65 不适用 | 805 | 805 | complete |
| Nature Machine Intelligence 2024 | 184 | 166 + 18 不适用 | 184 | 184 | complete |
| Nature Biomedical Engineering 2024 | 159 | 151 + 8 不适用 | 159 | 159 | complete |
| Nature Catalysis 2024 | 193 | 158 + 35 不适用 | 193 | 193 | complete |
| Nature Chemistry 2024 | 291 | 279 + 12 不适用 | 291 | 291 | complete |
| Nature Communications 2024 | 10,926 | 10,434 + 492 不适用 | 10,926 | 10,926 | complete |
| Nature Computational Science 2024 | 167 | 151 + 16 不适用 | 167 | 167 | complete |
| Nature Synthesis 2024 | 257 | 211 + 46 不适用 | 257 | 257 | complete |
| ACS Central Science 2024 | 278 | 201 + 77 不适用 | 278 | 278 | complete |
| Chemical Science 2024 | 2,091 | 1,874 + 217 不适用 | 2,091 | 2,091 | complete |
| JCTC 2024 | 909 | 850 + 59 不适用 | 909 | 909 | complete |
| Cell 2024 | 548 | 520 + 27 不适用，1 未解决 | 548 | 548 | incomplete |
| TCAD 2024 | 429 | 385 + 37 不适用，7 未解决 | 429 | 429 | incomplete |
| Angewandte Chemie 2024 | 5,170 | 4,817 + 353 不适用 | 5,170 | 5,170 | complete；Europe PMC 批次失败可恢复 |
| COLT 2024 | 169 | 169 | 169 不分配 | 169 | complete |
| CoRL 2024 | 264 | 264 | 264 不分配 | 264 | complete |
| EMNLP 2024 | 1,447 | 1,447 | 1,447 | 1,447 | complete |
| NAACL 2024 | 632 | 632 | 632 | 632 | complete |
| UAI 2024 | 201 | 201 | 201 不分配 | 201 | complete |
| AISTATS 2024 | 547 | 547 | 547 不分配 | 547 | recovery complete; bulk snapshot interrupted |
| Science 2024 | 2,039 | 1,741，缺 298 | 2,039 | 2,039 | incomplete |

“不分配/未注册/不适用”均保存为 null + `legitimately_absent`，不是填入虚构 DOI 或摘要。

### 3.1 会议来源的已完成优化

- AAAI：Crossref 年度批量 DOI join；OJS OAI-PMH 回填；极少数 OJS 无完整摘要时确定性解析官方 PDF 首页。
- ICLR：现代年份使用官网年度 virtual JSON 批量摘要和 OpenReview PDF；DOI 按主办方实际不分配处理。2017 受无人值守 OpenReview challenge 影响，仍缺 28 个摘要。
- IJCAI：2017+ 用 Crossref prefix 年度游标批量 join；2016 用官方 legacy 页面；PDF 为官网公开链接。
- PMLR：优先使用每个 volume 的官方 GitHub frontmatter 快照；若大压缩包发生 `IncompleteRead`，保留 primary 的官方 PDF/landing URL，并以受控并发逐篇读取官方详情页补摘要，不改变 membership。
- NeurIPS：官网年度 JSON 批量摘要；2022+ 通过 Crossref 年度注册表补 DOI；历史不分配 DOI 的年份如实标记。
- ACL family：pinned Anthology XML + 年度 `10.18653` 注册表；只对残余摘要读取官方 PDF 首页。
- CVF：年度 Open Access 索引给 PDF；新年份用官网 virtual JSON 批量摘要；DBLP 年度快照 + Crossref 残余审计补 DOI。

### 3.2 期刊来源的已完成优化

- Crossref serial adapter 已保留注册记录中的 publisher PDF link。
- Europe PMC 按最多 100 个 DOI 批量补摘要/OA PDF。
- Semantic Scholar 按 DOI batch 补摘要/OA PDF；匿名 429 现在只告警，不清空其他来源已取得的结果。
- 精确标题 arXiv 查询可作为残余回填；429/超时不会使整个 membership census 丢失。
- Nature 系列新增独立 `nature_articles` 官方元数据源：按 4 QPS 受控并发读取公开文章页 `dc.description`，并读取出版商文档类型判断摘要是否不适用。该路径不抓受限正文、不下载 PDF，也不绕过认证或访问控制。
- Science DOI 可确定性构造 publisher PDF endpoint；这只表示 Stage 1 有候选链接，实际访问与授权仍由 Stage 3 判定。
- Angewandte 2024 的 353 个无摘要条目已分类为 300 个 cover/frontispiece/graphical abstract、37 个 corrigendum、16 个 classifieds，并以锚定规则标记不适用。

## 4. 尚未完成的工作

### P0：关闭已知字段缺口

1. ICLR 2017：补齐 28/198 个摘要，或取得可审计的官方“无摘要/不可用”证据。不得绕过 OpenReview challenge。
2. ICCAD 2024：补齐 5 个摘要，或取得逐条可审计的官方“无摘要/不可用”证据；DAC 2024 已在 2026-08-14 receipt 中闭合。
3. Science 2024：处理 298 个缺摘要记录。当前无人值守访问遇到 Cloudflare challenge；应优先寻找 AAAS 可批量使用的官方 metadata/export，而不是浏览器逐篇抓取或绕过挑战。
4. Angewandte 2024：已通过批次级告警与恢复完成严格 receipt；后续将批次响应持久化，减少网络波动时的重复请求。

### P1：完成 35 个 venue 的代表年份字段矩阵

以下 venue 已完成十年 membership，但尚无本文所需的代表年份严格字段验收记录，或尚未在统一验收表中留证：

- 会议：AISTATS、COLT、CoRL、EMNLP、NAACL、UAI 的 2024 代表年份均有真实 complete receipt；AISTATS 通过 descriptor 级 detail fallback 跳过约 1 GB bulk snapshot，547 篇逐篇官方详情摘要与 primary PDF URL 全部恢复。
- 期刊：Cell、TCAD 已完成代表年份诊断但各有 1/7 条普通记录缺摘要；ACS Central Science、Chemical Science、JCTC、Nature Biomedical Engineering、Nature Catalysis、Nature Chemistry、Nature Communications、Nature Computational Science、Nature Synthesis、JACS 2024 已完成代表年份验收。

每个 venue 至少选择一个高产年份做全量严格运行，并将记录数、摘要、DOI、PDF URL、legitimately absent 分类和 receipt hash 写入统一验收文档。

### P2：完成 2016–2025 全字段矩阵

代表年份通过后，按 `venue × applicable year` 跑完整十年矩阵。每个单元必须满足：

- primary membership 与已冻结 decade census 一致；
- enrichment 不增加、不删除 membership；
- 摘要、DOI、PDF URL 为 present 或有来源证据的 legitimately absent；
- terminal cursor、parser 收支、字段 provenance、请求/响应 hash 完整；
- 临时 429/timeout 可断点续跑，不能要求从头重跑整年；
- 输出一份机器可读总矩阵，明确 complete、incomplete、not_applicable。

### P3：可靠性与性能

1. 将 journal DOI enrichment 的每个 batch 单独持久化，支持失败批次重试。
2. 为 Europe PMC、Semantic Scholar、arXiv、Nature article page 分别记录成功率、429、timeout 和重试次数。
3. 避免对已由 title/document type 证明摘要不适用的条目调用辅助 API。
4. 对 publisher HTML fallback 设置 venue 级开关；仅在批量元数据不足时启用。
5. 增加大年份并行 soak，验证 QPS、连接池、缓存和恢复行为，而不只验证功能正确性。
6. 对超过 30 秒的 Retry-After 直接返回可恢复错误，避免公共配额耗尽时把整个批处理挂起数小时；短暂退避仍按 provider policy 执行。

### P4：文档与发布收尾

1. 将字段验收矩阵合并为单一机器可读 artifact，并从其生成 Markdown，避免手工数字漂移。
2. 更新 `docs/stage1-standalone.md`，加入 Nature terms acceptance 示例、字段状态说明和严格/诊断模式示例。
3. 在 `task.md` 中只有当全部适用 `venue × year` 字段门禁通过后，才勾选 EDA/期刊十年字段完成项。
4. 为 Stage 1 输出 schema 和 receipt schema 建立版本化兼容测试。

## 5. 明确边界

- Stage 1 的 `pdf_url` 是候选位置，不代表已获授权或已成功下载。
- 完全开放的会议 PDF 可由 Stage 3 直接下载；Wiley、Nature/Springer Nature、ACS 等受限内容必须由 Stage 3 的授权策略和用户会话处理。
- `download-authorized-papers` skill 支持合法授权下载，不用于补写 Stage 1 摘要，也不用于绕过 paywall、CAPTCHA、Cloudflare 或其他访问控制。
- membership complete 不等于 field complete；代表年份 complete 也不等于十年 field complete。
- 辅助 scholarly graph 只能补字段/校验身份，不能把 venue candidate 提升为 official membership。

## 6. 建议执行顺序

1. 将 journal DOI batch 和 PMLR detail fallback 的成功响应持久化，减少重试成本。
2. 在 OpenAlex 配额恢复或取得授权 Semantic Scholar/ACM 访问后关闭 ICCAD 5 个摘要缺口；DAC 已关闭。
3. 为 Science 找可批量、可审计的官方 metadata 路径；找不到时保持 incomplete。
4. 关闭 Cell 的 1 条、TCAD 的 7 条普通摘要缺口，并补齐剩余代表年份证据。
5. 扩展至 2016–2025 全字段矩阵，生成统一 JSON + Markdown 报告。

## 7. 现有证据入口

- Membership 十年总证据：`docs/acceptance/stage1-decade-census-20260812.json`
- 字段补全验收：`docs/acceptance/stage1-field-enrichment-20260813.md`
- 独立接口说明：`docs/stage1-standalone.md`
- 总体规格与验收清单：`task.md`
