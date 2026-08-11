# Paper Agent v2 — 可执行任务规格

> 状态：核心实现与真实 Stage 1→4b 小规模验收已完成；Stage 2 已冻结无标签 600-pair manifest，人工双标、隐藏晋级与性能门禁待完成
> 规格日期：2026-08-09
> 实施基线：feature/crawler-adapters
> 本文用途：后续实现、验收和回归测试的唯一任务依据

## 1. 目标与交付边界

将现有 paper-agent 完善为一套可恢复、可审计、可扩展的论文研究流水线。系统需要覆盖会议和期刊元数据抓取、语义筛选、合法 PDF 获取、逐篇分析与领域综述，并同时提供：

1. 独立 Python CLI，适合本地、定时任务和批量运行。
2. Codex skill，负责对话式参数收集、任务编排和结果交付。
3. 同一套核心 Python 服务层；skill 不得复制 CLI 业务逻辑。

本次改造必须移除 OpenRouter 和 OpenCode 运行时依赖。需要模型推理的远程阶段统一通过 codex exec；Stage 2 统一使用本机开源模型，不允许静默回退到云模型。

### 1.1 非目标

- 不绕过付费墙、访问控制、验证码或站点授权。
- 不承诺所有论文都能自动获得 PDF；无法合法自动获取的条目进入人工队列。
- 本阶段不建设 Web UI。
- 本阶段不建设独立分布式控制平面，但任务模型必须支持多进程和多机器分片。
- 不在 CI 中下载大模型、访问真实订阅站点或消耗 Codex 配额。

## 2. 已冻结的关键决策

| 项目 | 决策 |
|---|---|
| 事实源 | SQLite |
| 交换格式 | JSONL 和 CSV；兼容导入旧 JSON/YAML |
| Stage 1 | VenueAdapter + SearchProvider + CitationProvider + LibraryProvider + MetadataEnricher |
| Stage 2 | 批量 reranker 粗筛 + Qwen3.5-9B 疑难裁决 |
| Stage 3 | DownloadProvider 链；浏览器授权流程使用 gpt-5.6-luna |
| Stage 4 | codex exec -m gpt-5.6-luna |
| Stage 4b | 全部 Luna 逐篇报告一次打包，codex exec -m gpt-5.6-sol 严格调用一次 |
| 报告语言 | 中文 Markdown |
| 默认 arXiv 行为 | 独立候选集，默认不进入最终报告 |
| 错误策略 | 通用阶段可恢复、可重试、fail-open；Stage 4b approved run 为一次性 dispatch，超时或结果不确定时禁止同 run 重发；禁止静默丢论文 |

整体数据流：

~~~text
Research questions / user seeds / venue descriptors
                 |
                 v
 frozen QueryPlan -> read-only source fan-out -> verify + deduplicate + citation snowballing
                 |
                 v
        Stage 1: discover + enrich
                 |
                 v
        SQLite canonical store
                 |
                 v
 deterministic rules -> batched reranker -> uncertain band -> Qwen3.5-9B
                 |
                 v
 public PDF -> arXiv -> authorized browser skill -> manual queue
                 |
                 v
       Luna per-paper analysis
                 |
                 v
 freeze every Luna report into one complete payload
                 |
                 v
       exactly one Sol one_shot_report
                 |
                 v
 deterministic local normalize -> verify -> audit -> publish
~~~

## 3. 统一领域模型与存储

### 3.1 SQLite 是单节点运行的唯一事实源

JSONL 和 CSV 只用于导入、导出和人工检查，不能作为单节点并发运行时数据库。SQLite 必须启用 WAL、外键和 schema migration。写入采用短事务；同一节点的并行 worker 通过租约领取任务，不能依赖进程内锁。禁止多台机器通过网络文件系统并发写同一个 SQLite 文件。

至少包含以下逻辑实体：

- papers：规范化论文实体。
- paper_sources：同一论文在 OpenReview、Crossref、arXiv、出版社等平台的来源记录。
- collections / paper_collections：会议、期刊和 arXiv 候选集成员关系、membership_status 与官方证据。
- artifacts：PDF、补充材料、抽取文本和分析文件。
- crawl_runs：抓取运行、窗口、游标、错误和统计。
- filter_decisions：规则、模型分数、阈值版本、理由和最终状态。
- download_attempts：provider 尝试顺序、HTTP/浏览器结果和失败类别。
- analysis_runs：Stage 4 输入范围、模型、prompt/schema hash 和输出。
- report_runs / report_one_shot_runs：Stage 4b 覆盖清单、唯一 dispatch 预约、模型输入输出 hash、本地验证和发布状态。
- search_plans：已冻结的研究问题、范围、查询族、来源选择、预算和停止条件。
- source_runs / search_queries：逐来源请求、游标、原始查询、过滤条件、时间、结果数、错误和 query hash。
- citation_edges：引用、被引和版本关系及其来源 provenance。
- screening_events：候选论文的纳入、排除、待复核状态和标准化原因。
- report_plans：报告受众、研究子问题、章节、分类轴、篇幅和证据要求。
- report_claims / claim_evidence：最终论点、支持/反对证据、定位、适用条件、强度与置信状态。
- comparison_groups：规范化任务/数据/指标/协议/baseline 条件和跨 run 稳定 comparison key。
- claim_relations：跨 report run 的 same/refined/split/merged/superseded/retired claim lineage。
- provider_registrations：provider/skill 的 distribution、version、entry point、能力 manifest、artifact digest、审计与信任状态。
- download_candidates：逐 URL 的版本、host、license、access_basis、resolver provenance 和策略决定。
- authorization_grants：download/store/extract/browser_data_sharing/remote_model_processing 等 action 的 paper/artifact/collection/domain/provider/model/purpose/mode scope、selection snapshot、max_papers、有效期、skill digest 和批准记录。
- manual_queue：待人工处理的去重、下载和筛选条目。
- schema_migrations：数据库版本。

### 3.2 论文规范字段

papers 至少保存：

- paper_id：系统生成且稳定的内部 ID。
- title、abstract、authors、keywords。
- publication_date、year。
- venue_id、venue_name、venue_type。
- doi、arxiv_id、canonical_url。
- volume、issue、pages。
- created_at、updated_at。

paper_sources 至少保存：

- source_id、paper_id、provider、external_id。
- landing_url、pdf_url、metadata_url。
- bibtex、citation_count、citation_count_as_of（来源提供时）。
- publication_version、license、host_type、access_basis（open_license/public_read_only/user_subscription/user_supplied/unknown）。
- raw_metadata_json。
- first_seen_at、last_seen_at、source_updated_at。
- metadata_capabilities、download_capabilities。

search_queries 至少保存：

- search_plan_id、source_run_id、provider、provider_version、query_compiler_version、role、query_text 和 provider_params_json。
- alias_group、date/venue/field/language/document_type filters、page/cursor。
- requested_at、completed_at、query_hash、response_hash、returned_count 和状态/错误。

citation_edges 至少保存 source_paper_id、target_paper_id、edge_type、provider、observed_at 和 raw evidence；引用方向必须有测试，不能因 provider 字段命名不同而颠倒。

规范论文另保存 verification_status，枚举为 verified、single_source、unverified、conflicted。verified 必须满足“官方 venue/publisher 记录加可解析稳定标识符”或“至少两个相互独立的可信来源在 DOI/arXiv ID、标题、作者和年份核心字段上一致”；独立性由 provider manifest 的 independence_group/upstream_families 判定，共同转引 Crossref 等同一上游的数据不能算两票。无法满足时保留论文并降级状态，禁止补造字段。citation_count 必须按 provider 和 as_of 分开保存，不能合并为一个无来源的数字。

论文身份验证与 venue membership 分开。paper_collections.membership_status 枚举为 official_confirmed、venue_candidate、not_member、conflicted；只有 official proceedings/publisher/accepted-decision evidence 才能产生 official_confirmed。聚合搜索或两份书目记录最多产生 venue_candidate；primary venue provider 失败时这些候选不得计入正式 venue 集，search run 必须 incomplete。

任何筛选、下载和分析结论都必须记录 run_id、paper_id、输入 hash、实现版本、模型 ID/revision、prompt/schema hash、时间戳和状态。原始来源数据不能被规范化结果覆盖。

### 3.3 去重与合并

去重优先级：

1. 规范化 DOI 精确匹配。
2. arXiv ID 精确匹配。
3. 同一 provider 的 external_id 精确匹配。
4. 标题、首作者、年份的规范化候选匹配。

前三类可自动合并来源；第四类只能产生候选，达到高阈值仍需保留审计记录。存在明显冲突时进入 manual_queue，不得自动覆盖。合并操作必须幂等，并保留每个来源的字段级 provenance。

### 3.4 导入导出与兼容

- 支持完整 JSONL 导入导出。
- 支持平面 CSV 导出；嵌套字段使用明确的 JSON 字符串编码。
- 支持旧数据库 JSON 和旧 YAML 配置迁移。
- 迁移必须可 dry-run，输出字段映射、警告和无法迁移项。
- 导入同一文件两次不能制造重复论文或重复来源。

### 3.5 多机分片与合并协议

多机仅用于 Stage 2–4b 的既有论文处理；Stage 1 与 canonical ID 分配由协调端完成。

- 协调端在只读事务中冻结输入 snapshot，生成全局 run_id、snapshot hash 和按稳定 paper_id 排序的互斥 shard manifest。
- manifest 包含 shard_id、epoch/fencing token、paper_id 列表、输入 artifact hash、配置 hash、模型 profile 和相对输出根目录。
- worker 不得创建或改写 canonical paper_id；每个 worker 使用本地 SQLite 和本地 content-addressed artifact 目录。
- 论文和 run ID 由协调端分配；重派 shard 时递增 epoch，旧 epoch 的迟到结果不得进入主库。
- artifact 以 SHA-256 寻址并随结果 manifest 传输；协调端校验 hash、MIME 和大小后重写为主库相对路径，不能导入 worker 绝对路径。
- 只有协调端拥有合并写权限。唯一键至少覆盖 run_id、stage、paper_id 和 output_kind。
- 同键同 hash 视为幂等重放；同键不同 hash 进入 merge_conflict/manual_queue，不以最后写入覆盖。
- 合并完成后核对 shard/paper/artifact coverage，缺失项可按新 epoch 精确重派。

## 4. Stage 1：可扩展爬虫

### 4.1 插件边界

核心接口至少包括：

~~~text
VenueAdapter.discover(descriptor, window, cursor) -> SourceBatch
SearchProvider.search(query_spec, cursor) -> SourceBatch
CitationProvider.references(seed, cursor) -> CitationBatch
CitationProvider.citations(seed, cursor) -> CitationBatch
LibraryProvider.import_seeds(input_spec) -> SourceBatch
MetadataEnricher.enrich(raw_paper) -> EnrichmentResult
MetadataVerifier.verify(identity_candidate, evidence[]) -> VerificationResult
OpenAccessResolver.resolve(paper, policy) -> AccessLocationCandidate[]
DownloadProvider.probe(candidate, policy) -> FetchDecision
DownloadProvider.fetch(request: FetchRequest, authorization_context) -> DownloadResult
~~~

要求：

- VenueAdapter 只处理平台协议和分页，不写业务筛选逻辑；SearchProvider 面向主题查询，CitationProvider 面向引用图扩展，两者不能伪装成 venue 列表抓取。
- VenueDescriptor 使用版本化 YAML 描述 venue、年份/日期范围、provider 和 provider 参数。
- QuerySpec 是版本化对象，至少包含 research_question_id、原始查询、同义词组、日期、venue/field/language/document type filters、排序、页大小和预算。
- provider role 枚举冻结为 venue_primary、search、citation、library、metadata_enricher、metadata_verifier、oa_resolver、download；capability 枚举与 role 分离，至少包含 stable_id、metadata、abstract、date_filter、references、citations、oa_locations、full_text、supplement 和 bulk_snapshot。
- 同一平台新增 venue 原则上只增加 YAML 和 fixture。
- 新平台通过 Python entry point 注册，但发现 entry point 时只读取 distribution metadata，不得先 import 再判断是否可信。
- 第三方插件默认禁用。allowlist 必须绑定 distribution name、精确 version、provider、entry point 和 wheel/sdist 或安装内容 SHA-256（有签名时同时校验）；任一版本/digest 漂移立即禁用并要求重新审计/授权，不能只按包名或 provider 名放行。
- 内置 provider 可在主进程加载；第三方 provider 即使已 allowlist，也默认在网络/文件/环境变量最小权限子进程运行并只通过 SourceBatch IPC 返回。只有单独完成信任提升和记录 ADR 后才允许 in-process import。
- 每个 provider 必须声明 role、稳定标识符、metadata、abstract、references、citations、OA/full-text、日期过滤、认证和限流等能力；调度器只能调用已声明能力。
- versioned provider manifest 是 role/capability、认证、限流、terms、independence_group 和 upstream_families 信息的唯一事实源；正文映射表、示例配置和 schema fixture 必须由同一 manifest 校验，禁止三处手工定义后漂移。
- SourceBatch/CitationBatch 统一携带 source_run_id、query_hash、entries、next_cursor、status、error 和原始响应 artifact hash。provider 任务只读返回结果，只有协调端负责规范化、去重和 SQLite 写入。
- 单个 venue 失败不得终止其他 venue；失败必须进入本次 run 的结构化摘要。

### 4.2 来源组合与职责

检索采用“用户已有知识优先、官方 venue/publisher 为主、开放学术图补召回、权威元数据源做校验”的组合，不允许把任一聚合站当作唯一事实源：

| 层级 | 默认来源 | 职责与约束 |
|---|---|---|
| 用户种子 | DOI、arXiv ID、URL、BibTeX、RIS、CSL-JSON、Zotero 导出/显式授权 API、本地 PDF | 最高优先级种子；笔记和本地分析可提供 query term，但不能覆盖外部书目事实 |
| 官方 venue/publisher | OpenReview、PMLR、ACL Anthology、CVF Open Access、AAAI/IJCAI/NeurIPS 官方 proceedings、ACM DL、IEEE Xplore、Springer Nature、Cell Press、AAAS | 确认 venue 身份、正式版本、卷期页码和官方落地页；遵守各站 API/robots/条款 |
| 广域发现 | arXiv、Semantic Scholar、OpenAlex、PubMed、Europe PMC | 补充跨 venue 召回、主题搜索和引用图；结果仍需去重与元数据校验 |
| 书目校验 | Crossref、DBLP、arXiv、Semantic Scholar、OpenAlex、PubMed/Europe PMC | 校验 DOI/arXiv ID、标题、作者、年份和出版状态；保留字段级 provenance 与冲突 |
| 公开访问 / OA 解析 | 官方公开 PDF、arXiv、Unpaywall、PMC/Europe PMC | 只返回公开可读或明确开放许可的候选位置，随后由 Stage 3 独立判断当前用途的下载/保存权限，不能自动提升许可证 |
| 可选发现/阅读增强 | Exa、Gemini Search、DeepXiv、AlphaXiv | 默认关闭，不属于 Phase 2 核心依赖；只有显式配置原生凭据、成本和条款后加载，不得经 OpenRouter，不能作为权威书目来源，候选必须经上述来源交叉验证 |

Google Scholar 不作为无人值守核心 provider：没有冻结的官方自动化 API 时，只允许人工/attended 发现并把 DOI、标题或 URL 作为种子导入，禁止依赖页面抓取完成验收。

### 4.3 首批 venue 与冻结 provider 映射

会议：

- NeurIPS、ICML、ICLR、AAAI。
- ACL、CVPR、ICCV、IJCAI。
- DAC、ICCAD。

期刊：

- IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems（TCAD）。
- Nature Machine Intelligence。
- Nature Chemistry。
- Nature Computational Science。
- Nature Communications。
- Nature Catalysis。
- Nature Biotechnology。
- Nature Biomedical Engineering。
- Cell。
- Science。

首批内置映射如下；实现时 adapter 名、入口、能力和 fallback 顺序写入 versioned acceptance manifest，不得在运行中按 venue 名称猜测：

| Venue | Primary adapter | Metadata/discovery fallback |
|---|---|---|
| NeurIPS | neurips_proceedings | OpenReview（存在对应记录时）、Crossref、DBLP、Semantic Scholar、OpenAlex |
| ICML | pmlr | OpenReview（存在对应记录时）、Crossref、DBLP、Semantic Scholar、OpenAlex |
| ICLR | openreview | arXiv、DBLP、Semantic Scholar、OpenAlex；按 accepted decision/venue 字段筛选，不能把整组 submission 当录用论文 |
| AAAI | aaai_ojs | Crossref、DBLP、Semantic Scholar、OpenAlex |
| ACL | acl_anthology | Crossref、DBLP、Semantic Scholar、OpenAlex；descriptor 区分 main、Findings 和 workshop |
| CVPR / ICCV | cvf_open_access | IEEE Xplore、Crossref、DBLP、Semantic Scholar、OpenAlex；严格区分 main proceedings 与 workshop |
| IJCAI | ijcai_proceedings | Crossref、DBLP、Semantic Scholar、OpenAlex |
| DAC / ICCAD | eda_proceedings（逐年解析 IEEE Xplore + ACM DL） | 会议官方 program/proceedings、Crossref、DBLP、Semantic Scholar、OpenAlex；按 DOI 去重，不能冻结单一出版平台覆盖所有年份 |
| IEEE TCAD | ieee_xplore（publication number 43；ISSN 0278-0070） | Crossref、DBLP、Semantic Scholar、OpenAlex |
| 指定 Nature 系列期刊 | springer_nature | Crossref；生命科学论文再用 PubMed/Europe PMC 校验和 OA 解析 |
| Cell | cell_press | Crossref、PubMed、Europe PMC、Semantic Scholar、OpenAlex |
| Science | aaas_science | Crossref、PubMed、Europe PMC、Semantic Scholar、OpenAlex |

表中 fallback 只用于发现、字段补全和身份校验；它们不能在 primary/official accepted evidence 缺失时把 venue_candidate 提升为 official_confirmed。报告可单列候选，但不能混入该会议/期刊的正式收录统计。

Nature 系列 acceptance manifest 必须冻结准确 journal slug、print/electronic ISSN 和允许的 article types：natmachintell（2522-5839）、nchem（1755-4330/1755-4349）、natcomputsci（2662-8457）、ncomms（2041-1723）、natcatal（2520-1158）、nbt（1087-0156/1546-1696）和 natbiomedeng（2157-846X）。Cell flagship 冻结 ISSN 0092-8674；Science 冻结 ISSN 0036-8075/1095-9203。禁止仅用标题前缀或 publisher 域名判定 venue。

注意：

- Nature Computer Science 是错误名称，统一修正为 Nature Computational Science。
- TCAD 是期刊，不得按会议年份页建模。
- 来源平台必须通过调研和 fixture 固定，不能根据 venue 名称硬猜。
- OpenReview 从 venue group 动态解析 invitation 并兼容 API v1/v2；PMLR volume ID 从会议/PMLR 官方链接解析；AAAI 同年所有相关 OJS issue 必须遍历，均不得只靠年份拼 URL。
- ACL Anthology 优先使用官方数据/API 并冻结数据 commit/version；静态官方站采用低 QPS、缓存、条件请求和分页断点。
- 元数据抓取与 PDF 能力分离；无法获取 PDF 不能阻塞论文元数据入库。
- 每个内置 venue 提交 versioned acceptance manifest，固定 venue_id、provider/adapter、测试窗口、fixture hash、预期 stable IDs、最少字段与能力标记。
- fixture 契约测试使用精确预期；受控 live smoke 保存响应快照和时间戳，只验证映射、最小结果与字段下限，不把易变化的实时总数写成永久断言。

### 4.4 冻结 QueryPlan 与可复现检索

任何批量检索开始前必须先编译并批准 QueryPlan。它由 CLI 配置或 Codex skill 辅助形成，不要求额外模型调用；YAML 只用于生成 draft，运行时以 approved compiled plan 为准。用户修改研究问题、范围、provider resolution、纳入标准或预算时必须产生新版本，不能原地改写旧 run。

QueryPlan 的已确认版本写入不可变 search/<search_plan_id>/QUERY_PLAN.json，并在原子校验后更新 search/latest-approved.json；运行时复制到 search run artifact 中。latest 只是入口，SQLite 中的 plan_id/hash 才是引用依据，历史版本不删除。

content hash 使用 RFC 8785 canonical JSON + SHA-256；被哈希内容排除 plan_id/plan_hash、status、created/updated timestamps 和 detached approval record，避免自引用/时间戳改变内容身份。approval record 单独引用 content_hash，任何批准后业务字段修改都会使批准失效。

QueryPlan 至少包含：

- 研究目标、受众、主问题和可独立回答的子问题。
- 时间范围、学科/venue、语言、文献类型，以及明确的纳入与排除标准。
- 核心概念、缩写、旧称/新称、相邻术语、方法/任务/数据集/benchmark 变体；每个 query variant 绑定子问题和 alias_group。
- 用户种子、来源 allowlist/required/optional 策略、每来源字段/日期过滤和正式版优先规则。
- 已解析的 provider distribution/version/artifact digest、manifest hash、roles/capabilities、resolved 原因、API/snapshot mode、snapshot hash、凭据存在性（不含 secret）、query compiler version 和原生 query hash。
- Stage 2 filter profile/config/threshold artifact hash、seed selector version/config，以及 round state machine version；这些都是饱和判定输入。
- 引用扩展深度、每个 seed/round/source 上限、请求/候选/时间预算和饱和停止条件。
- plan hash、schema version、创建时间、状态（draft/approved/superseded）和 approval record（approved_hash、approved_by、approved_at、approval_method）。

paper-agent search plan 生成 draft；paper-agent search approve --plan <path> --hash <sha256> 写入显式 approval record；paper-agent search run --plan <path> 在发请求前重新解析运行环境并逐字段比对 compiled plan。YAML、provider/version/digest、mode/snapshot、原生 query、预算或策略有任一偏差即拒绝运行并要求重新编译/批准，不能用运行时覆盖继续。

每个 provider 使用独立 query compiler 生成原生查询，不得用一个字符串冒充所有平台查询。所有实际查询、参数、分页/游标、时间戳、命中数和 query hash 写入 search_queries，使同一快照可重放和审计。

### 4.5 并行 fan-out、合并与校验

- provider_policy=all_resolved 表示调用 compiled QueryPlan 中所有 resolved=true 的 provider。resolved 是在 plan 编译时依据 enabled 条件、QueryPlan 领域枚举、受信 manifest、凭据可用性、snapshot hash 和 capability 计算出的冻结布尔值；运行时不得重新解释 auto_for_cs/auto_for_biomed。
- schema 必须验证每个 required role 至少解析到一个 resolved provider，并验证正文 fallback 使用的 role 已在该 provider manifest 声明；每个显式 VenueDescriptor 的 primary adapter 自动成为 exact required provider，exact required provider 与 required role 是两个独立约束。
- 对 compiled QueryPlan 中的全部 resolved provider 并行发起只读任务；每个来源独立分页、缓存、限流、重试和熔断。
- 每个来源成功或失败都必须记录。只要至少一个来源贡献结果就可继续处理，但任一显式 required provider 失败，或任一 required role 没有至少一个成功 provider，都会把搜索/最终报告标记为 incomplete；显式请求但不可用的 provider 不得静默跳过。
- 协调端先按 DOI、arXiv ID、provider stable ID 做机械去重，再做标题/作者/年份候选匹配；昂贵的模型判断只能用于剩余歧义项。
- 正式出版版本优先于 preprint，但两者以 version_of 关系保留；preprint、workshop、peer_reviewed 状态必须显式标注。
- 存在字段冲突时保留全部来源值并按“官方 venue/publisher > DOI 注册机构/领域权威库 > 聚合图 > AI/网页发现”选择显示值；无法裁决则标记 conflicted 并进入 manual_queue。
- 来源缓存键至少包含 provider、query hash、cursor 和 API/version；每个 provider 单独配置凭据、QPS、并发、缓存 TTL 和数据使用条款标记。
- 报告逐来源 raw_discovered、unique_after_dedup、screened、excluded_by_reason、included、full_text_available、error_count 和 overlap；不能只给总论文数。

### 4.6 引文滚雪球扩展与饱和停止

- search campaign 使用冻结状态机：round 0 执行 user seeds + venue/topic search -> normalize/verify -> Stage 2；每个 round r≥1 执行 select_seeds -> references/citations -> normalize/verify -> Stage 2 -> audit -> stop/next。每个 round 的输入集合和状态转换写入 manifest，可逐轮 resume。
- root seeds 是 QueryPlan 中的用户种子和 round 0 的 Stage 2 relevant 论文；citation depth 是从任一 root 沿规范引用边的最短距离，root depth=0。相同论文从不同路径到达时保留边 provenance，但只筛选一次。
- seed selector 必须冻结 algorithm/version/config。默认选择所有仍在范围内的用户种子，再按每个 subquestion 的 Stage 2 relevant 论文依 `reranker_score DESC, paper_id ASC` 取前 20；needs_review、metadata conflicted 和明确排除项不能成为自动 seed。每轮只扩展此前未扩展过且 depth < max_depth 的 seed。
- 每轮在发请求前冻结 seed manifest，记录 seed paper_id、seed_reason、parent_round、depth、subquestion、rank、selector/config hash；按 provider、direction、depth、seed rank、paper_id 生成稳定 request schedule 并预留预算槽位，使并发完成顺序不会改变哪些请求获准执行。
- CitationProvider 同时执行 backward references 和 forward cited-by；扩展结果回到相同去重、校验和 Stage 2 筛选链，不能绕过纳入标准。
- 默认 max_depth=2；每个 seed、round 和 provider 都有硬上限，引用环通过 stable paper ID 去重。
- 同时保留支持、反对、复现和后续修正论文，不按引用数或多数票删除反例。
- screened_unique 定义为本轮首次出现、完成规范化且获得 Stage 2 最终决策的 canonical papers；new_included_unique 是其中首次进入累计 relevant 集合的论文，needs_review 不计入分子。若本轮有未完成筛选，不能用该轮触发饱和；仍有 needs_review 时最多标 saturated_with_unresolved，最终报告按 limited_scope 处理，不能标 complete saturation。
- 当所有查询/分页耗尽，或连续两轮 `new_included_unique / max(1, screened_unique) < 5%` 时可判定经验饱和；任何请求、时间或候选硬预算先耗尽则停止原因为 budget_exhausted，最终报告必须标 limited_scope，不能宣称检索饱和或完备。
- search audit 保存每轮新增/重复/纳入/排除数、边类型、深度、停止原因和未完成来源。

### 4.7 增量抓取

- 支持日期窗口、年份、卷期和 provider cursor。
- 每次运行记录水位线，但允许显式回放历史窗口。
- 更新已有论文时只追加或更新对应 source，不破坏其他来源。
- 网络请求使用有上限的指数退避、抖动和 Retry-After。
- 对 robots、站点条款和 API 限流采取保守策略。
- fixture 测试不得依赖实时站点 HTML。
- 增量 run 必须输出新增、删除/撤稿、书目字段变更、正式版替换 preprint 和受影响下游 paper_id；水位线不能掩盖更正、撤稿或回填记录。

### 4.8 arXiv 候选集

- 支持主题关键词、日期范围和类别过滤。
- 结果写入同一规范数据库，但 membership 标记为 arxiv_candidates。
- include_arxiv_candidates 是控制 arXiv-only 论文参与后续流水线的唯一开关，默认 false；关闭时不参与筛选、下载或报告。
- 用户显式开启 include_arxiv_candidates 后，arXiv-only 论文才进入 Stage 2；其中被保留的论文才能进入 Stage 3/4/4b。
- 与主集合论文匹配时增加 arXiv source，不复制 paper。
- use_matched_arxiv_as_download_source 默认 true，只允许把已匹配的 arXiv 版本作为主 venue 论文的下载备选，不等同于启用 arXiv-only 候选。
- arXiv 元数据许可与每篇 PDF/source 的许可分开记录；“可公开读取”不能自动标为可再分发，Stage 3 仍按 access_basis/license/purpose 判定。

## 5. Stage 2：高吞吐语义筛选

### 5.1 处理顺序

Stage 2 使用以下级联：

1. 确定性元数据规则。
2. 全量批量 reranker 主题相关性评分。
3. 根据已校准的 T_low 和 T_high 分流。
4. 灰区、异常和信息不足样本交给 Qwen3.5-9B。
5. 强冲突或模型失败进入 needs_review。

正则规则只能：

- 排除用户明确声明的确定性类型，例如撤稿、社论或特定文献类型。
- 对明确命中的样本快速通过。
- 作为特征或审计理由。

正则未命中不能直接判为无关。保留 regex_only、semantic_only 和 cascade 模式用于调试与回归，但生产默认是 cascade。

### 5.2 批量 reranker

稳定基线：

- 模型：BAAI/bge-reranker-v2-m3。
- 官方 oracle revision：953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e。
- oMLX 原生路径：POST /v1/rerank。
- 官方 FP32 权重作为数值 oracle。
- 初始 MLX BF16 候选为 soichisumi/bge-reranker-v2-m3-mlx@b4577f49e18adb53ed9e557192094f69f3dc2c1c；它属于第三方转换，完成供应链审计并通过 parity gate 后才能进入生产。

质量/速度挑战者：

- Querit/Querit-4B。
- Qwen/Qwen3-Reranker-0.6B。
- Qwen/Qwen3-Reranker-4B。

Querit-4B 截至 2026-06-20 的模型卡自报为公开模型中 MTEB Multilingual v2 reranking 平均分最高，但这不等同于本项目论文主题筛选效果。所有候选必须使用本项目隐藏金标集盲测；最终默认模型是质量门通过后吞吐最优者。

实现要求：

- Stage 2 的每个生产或候选模型都必须以权威 config/model card 证明总参数量 ≤ 10B；按总参数而不是活跃参数计算，超限模型不得进入评测。
- 模型后端必须可替换，至少抽象 omlx_rerank、omlx_chat 和受控的 mlx_native 实验后端。
- 固定 API model ID、上游 source repo/revision、转换 revision、格式、量化、oMLX/MLX 版本和模型文件 hash；不得用一个含糊 revision 混合表示上游与转换权重。
- 生产模型及转换权重必须记录许可证；默认只接受允许本地部署和项目用途的许可证。
- query 简短、稳定且版本化；document 固定为 title + abstract + keywords。
- oMLX v0.5.7 的 /v1/rerank 没有暴露 max_length；XLM-R 路径默认 max_length=512，表示 query、document 与特殊 token 组成的单个 pair 总长度。请求 schema 中的 max_chunks_per_doc 尚未实现，不得假定额外传入 max_length 或分块参数会生效。
- 缺摘要、可能被 512-token 上限截断、多条件冲突或语言异常的样本不得自动拒绝，直接交给 Qwen。
- raw reranker score 不是概率，禁止硬编码 0.5 或 0.7。
- T_low/T_high 由开发集以优先满足 recall 的方式校准并版本化。

批量策略：

- 大吞吐依赖一个请求携带多个 documents，而不是大量并发 HTTP 请求。
- 按 token 长度分桶，return_documents=false。
- 初始 32 documents/request，基准测试 16、32、64。
- OOM、峰值内存越过 28 GB，或相对下一档小 batch 的 p95 增幅 > 25% 且吞吐提升 < 10% 时，按 64 → 32 → 16 回退；基准 winner 写入运行配置，生产中只允许因资源保护向下回退。
- reranker 客户端默认最多 2 个 in-flight 请求。
- oMLX 的非流式 reranker forward 经单 worker executor 调度；2 个 in-flight 仅用于填充队列和隐藏 HTTP 开销，不代表两个 Metal forward 并行。
- 记录 papers/s、pair tokens/s、p50、p95、峰值内存和回退次数。

### 5.3 Qwen 疑难裁决

默认候选：

- mlx-community/Qwen3.5-9B-8bit。
- 固定已验收 revision；初始候选 revision 为 16daa4818c54ce5f5436f929d52542eb65bbed9d。
- 4-bit 版本只作为显式低内存配置，必须单独通过质量回归，禁止自动切换。

运行要求：

- oMLX 至少升级到 v0.5.7；当前本机 0.2.7 不满足要求。
- 使用 OpenAI-compatible chat endpoint。
- 请求必须设置 chat_template_kwargs.enable_thinking=false；若迁移旧配置中的 thinking=false，adapter 必须映射到该字段，不得把非标准 thinking 字段原样发送后假定生效。
- temperature=0、固定 seed、stream=false、max_tokens=256。
- 每模型 max_context_window 初始固定为 16384；只有压测通过后才能提高，硬上限为 32768，不继承模型声明的超大原生上下文。
- 使用 extra_body.structured_outputs.json 传递 logit-level JSON schema；grammar 编译失败必须以 400 fail-closed。
- 不能只依赖可能退化为 prompt 注入的 response_format；出现 Warning header 视为结构化约束失败。
- 模型返回后仍由 Pydantic/jsonschema 二次验证。
- 一个 paper 对应一个逻辑请求；依靠 continuous batching，不把多篇论文塞进一个巨大 JSON。
- Qwen worker 初始并发 4；对客户端/服务并发组合 4、8、16 做压测后显式固定 winner，禁止运行时无条件自动升到 16。
- schema、超时、paper_id 错配或模型冲突必须重试或 needs_review，不能判为无关。

Qwen 返回至少：

~~~text
paper_id
decision: relevant | irrelevant | needs_review
score: 0..1
reason_codes[]
rationale
evidence_fields[]
~~~

容量目标：

- reranker 消化约 80%–90% 的样本。
- Qwen 裁决比例默认预期不超过 15%，超过时报警并记录成本原因。
- 15% 是容量预警，不是牺牲召回率的硬配额。
- 灰区长期超过 30% 时必须重校 query/阈值或扩容，禁止扩大自动拒绝区间掩盖问题。
- 明确淘汰项只保存 score、reason_code 和 provenance；只有保留项及边界项生成自然语言理由。

### 5.4 金标集与模型晋级门

建立 600 个 topic-paper pair：

- 300 个开发/校准样本。
- 150 个隐藏 hard-case 样本。
- 150 个隐藏真实爬取分布样本。
- 覆盖 6–8 个主题、中英文与跨语言匹配；每个需要单独报告 recall 的主要语言切片至少包含 30 个正例。
- 开发集和 hard-case 集至少包含 20% hard negatives、10% 缺失或截断摘要，并覆盖 hard positives。
- 真实分布集必须从目标爬虫输出按已记录的抽样概率随机抽取，不得为了平衡标签而改写自然基率。
- 同一论文的 arXiv、会议、期刊或预印本家族必须使用 paper-family 分组切分，不能跨 dev/hidden 泄漏。
- 冻结 sampling manifest，记录 topic、language、source、paper-family 和语料 hash；HIDDEN_REAL 记录真实纳入概率 150/N，受配额、权重和 family 约束的 DEV/HIDDEN_HARD 没有可解释的单一纳入概率，必须记录 `null`，不得借用 HIDDEN_REAL 概率。

抽样与权威 gold 标注必须分两阶段：先由 evaluator 保管的完整自然语料框生成 private snapshot（不要求对全量 snapshot 预标）。使用冻结 seed，在**不读取 curated labels**的情况下从完整自然框抽取 150 条 HIDDEN_REAL，冻结为 evaluator-only freeze frame，并记录真实纳入概率 150/N；再从剩余 paper-family 的 curated pool 构建 DEV 与 HIDDEN_HARD。curated annotations 的临时标签和难例标记只用于抽样配额与分层，绝不是 gold label。仅当这 600 个 pair 已被选中后，全部样本才由两位标注者独立标注，并交由第三人仲裁。构建命令为：

```sh
paper-agent --dry-run stage2-sampling freeze-frame \
  --private-snapshot /secure/evaluator/private-snapshot.json \
  --output /secure/evaluator/hidden-real-freeze-frame.json
```

先从同一 snapshot 和 freeze frame 导出 curation worklist。它排除全部 HIDDEN_REAL pair 及其整个 paper-family；在 evaluator 内人工或本地模型辅助填写 `curation-decisions.json`。其中 `provisional_label`、hard flags 和 `source`（`human_provisional`、`model_provisional` 或 `human_reviewed_model_suggestion`）只服务于抽样，绝不是人工 gold：

```sh
paper-agent --dry-run stage2-sampling curation-worklist \
  --private-snapshot /secure/evaluator/private-snapshot.json \
  --hidden-real-freeze-frame /secure/evaluator/hidden-real-freeze-frame.json \
  --output /secure/evaluator/curation-worklist.json

paper-agent --dry-run stage2-sampling curation-import \
  --private-snapshot /secure/evaluator/private-snapshot.json \
  --hidden-real-freeze-frame /secure/evaluator/hidden-real-freeze-frame.json \
  --worklist /secure/evaluator/curation-worklist.json \
  --decisions /secure/evaluator/curation-decisions.json \
  --curated-annotations-output /secure/evaluator/curated-annotations.json \
  --receipt-output /secure/evaluator/curation-receipt.json
```

```sh
paper-agent --dry-run stage2-sampling build \
  --private-snapshot /secure/evaluator/private-snapshot.json \
  --hidden-real-freeze-frame /secure/evaluator/hidden-real-freeze-frame.json \
  --curated-annotations /secure/evaluator/curated-annotations.json \
  --curation-receipt /secure/evaluator/curation-receipt.json \
  --gold-manifest-output /secure/evaluator-transfer/gold-manifest.json \
  --provenance-output /secure/evaluator/provenance.json
```

2026-08-12 已从真实 Crossref 自然语料框排除 HIDDEN_REAL paper-family 后生成 802 条 curation
worklist，并用锁定的本地 Qwen3.5-9B 以 101 个一次性结构化批请求生成仅供抽样的
`model_provisional` 标签；随后正式冻结 300 DEV、150 HIDDEN_HARD、150 HIDDEN_REAL 的无标签
600-pair manifest。聚合证据见 `docs/smoke/stage2-real-curation-frame-20260812.json`。这些临时标签不是
人工 gold，不得用于质量得分或勾选生产 release gate；下一步仍是对已选 600 pair 做两人独立标注和
第三人分歧仲裁。

`gold-manifest` 是不含标签的 600-pair 公共清单，可进入 release；private snapshot、HIDDEN_REAL freeze frame、curated annotations、抽样 provenance 和原始标注 ledger 均由 evaluator 托管（当前设计下 provenance 也不公开）。`--private-labels` 必须精确覆盖该 600-pair manifest，绝不是完整 snapshot 的全量标签。

不得手工拼装原始 ledger。先为两位标注者分别生成盲表；盲表只包含 topic、title、abstract、language
和稳定 pair_id，不暴露 split、paper-family、抽样概率、临时标签或 hard flags。每位标注者独立把自己的
`label: null` 填成 0..3：

```sh
paper-agent stage2-sampling annotation-worklist \
  --gold-manifest /secure/evaluator-transfer/gold-manifest.json \
  --private-snapshot /secure/evaluator/private-snapshot.json \
  --participant-id annotator-a \
  --output /secure/evaluator/annotation-a.json

paper-agent stage2-sampling annotation-worklist \
  --gold-manifest /secure/evaluator-transfer/gold-manifest.json \
  --private-snapshot /secure/evaluator/private-snapshot.json \
  --participant-id annotator-b \
  --output /secure/evaluator/annotation-b.json
```

两份盲表填完后生成只含分歧项且不显示两人原标签的第三人盲表；该盲表绑定两份已完成输入，仲裁后
替换任一标注文件会失败。QWK 低于 0.75 时该命令直接失败，不能以仲裁掩盖一致性不足：

```sh
paper-agent stage2-sampling adjudication-worklist \
  --gold-manifest /secure/evaluator-transfer/gold-manifest.json \
  --private-snapshot /secure/evaluator/private-snapshot.json \
  --annotation-a /secure/evaluator/annotation-a.json \
  --annotation-b /secure/evaluator/annotation-b.json \
  --participant-id adjudicator-c \
  --output /secure/evaluator/adjudication.json
```

第三人填完所有分歧后，由受控命令组装并验证 ledger。sampling provenance 必须绑定生成 manifest 时
使用的 curated hard flags；只有与最终人工 label 兼容的候选才进入 private difficulty strata，最终配额仍由
`GoldManifest.validate` fail-closed：

```sh
paper-agent --dry-run stage2-sampling assemble-annotation-ledger \
  --gold-manifest /secure/evaluator-transfer/gold-manifest.json \
  --private-snapshot /secure/evaluator/private-snapshot.json \
  --curated-annotations /secure/evaluator/curated-annotations.json \
  --sampling-provenance /secure/evaluator/provenance.json \
  --annotation-a /secure/evaluator/annotation-a.json \
  --annotation-b /secure/evaluator/annotation-b.json \
  --adjudication /secure/evaluator/adjudication.json \
  --output /secure/evaluator/annotation-ledger.json
```

dry-run 通过后移除 `--dry-run` 创建 no-replace ledger，再通过受控转换生成 promotion 私有标签：

```sh
paper-agent --dry-run stage2-sampling finalize-annotations \
  --gold-manifest /secure/evaluator-transfer/gold-manifest.json \
  --annotation-ledger /secure/evaluator/annotation-ledger.json \
  --private-labels-output /secure/evaluator/private-gold-labels.json
```

dry-run 必须重算双人完整覆盖、第三人分歧仲裁、仲裁前 QWK ≥ 0.75 与最终 gold 配额；正式执行只生成 no-replace 私有 label artifact，不复制 annotator 身份或原始标注/仲裁行。

标注 rubric：

- 0：明确无关。
- 1：弱相关、仅背景提及或证据不足。
- 2：与主题直接相关，应保留。
- 3：主题核心论文，必须保留。

2/3 为正例。两位标注者在 600-pair 选样后独立标注，分歧由第三人仲裁。quadratic-weighted kappa 在仲裁前计算且必须 ≥ 0.75。隐藏集标签由一次性 promotion evaluator 持有；候选不得读取标签或逐次调参。隐藏结果一旦暴露，该集合转为 regression set，下一轮模型晋级必须使用新 holdout。

级联分数必须定义为 P(gold_label ≥ 2)。reranker 与 Qwen 分别在 dev 上拟合并冻结 path-specific calibrator，不能直接比较或混合原始分数。ECE 使用 10 个 equal-frequency bins。

needs_review 的指标语义：

- retention recall：needs_review 视为未丢弃，用于衡量召回保护。
- 定义 TP_auto 为自动 relevant 且 gold positive，FP_auto 为自动 relevant 且 gold negative，P_gold 为全部 gold positive，P_review 为 gold positive 且 needs_review。
- automatic precision = TP_auto / (TP_auto + FP_auto)；needs_review 不进入分母。
- automatic recall = TP_auto / P_gold；positive-class F1 使用 automatic precision 与 automatic recall。
- retention recall = (TP_auto + P_review) / P_gold。
- 真实分布集的 operational precision 与 automatic precision 使用同一定义，表示默认会自动进入下游的 relevant 集合精度。
- automatic coverage：自动 relevant/irrelevant 占全部样本的比例，必须 ≥ 95%。
- 因 schema、超时或服务错误产生的 needs_review 必须 ≤ 0.5%。
- needs_review 默认停在 Stage 2 人工队列，不自动进入 Stage 3/4；只有用户显式 include_needs_review 才继续。
- 容量与潜在下游成本估算必须把 needs_review 按可能保留项计入，不能用复核队列隐藏成本。

隐藏集分别验收，禁止无权重混算：

- hard-case 集只报告并约束 retention recall、positive-class F1、topic-macro F1 和错误类型。
- 真实分布集单独报告 operational precision、retention recall、Brier score、每千篇误保留/误拒绝数和 automatic coverage。
- 只有保存抽样概率并使用逆概率权重时，才能给出跨集合总体指标。

发布硬门：

- hard-case 与真实分布集的 relevant retention recall 均 ≥ 0.95。
- 每个隐藏集内 gold label=3 的核心论文 retention recall ≥ 0.97；某集合核心样本不足 30 条时该项只报告点估计与 Wilson interval，并由两个隐藏集的核心样本合并门补充约束。
- 真实分布集 operational precision ≥ 0.80。
- hard-case 集 positive-class F1 ≥ 0.88，按 topic 计算的 macro positive F1 ≥ 0.82。
- 每个隐藏集中样本门槛达标的主要语言切片 retention recall ≥ 0.90。
- 真实分布集 Brier score ≤ 0.15。
- ECE 目标 ≤ 0.08，但在自然 holdout 少于 500 条时只作为带置信区间的报告项，不作为硬门。
- 固定配置三次完整隐藏集运行中，三次决策完全相同的 pair 比例 ≥ 99%，needs_review 也视为一种决策。

结构化输出另设至少 1,000 个 adjudicator 请求的固定专项回放：

- 首次响应 JSON/schema 合法率 ≥ 99.5%。
- paper_id 保真率 100%，schema 外文本或 think 标签泄漏为 0。
- timeout/服务错误与 schema 错误分开统计。
- 确定性解析修复与模型重试分开统计；任何最终无效响应 100% 路由 needs_review，绝不自动拒绝。
- 固定专项回放在允许的一次重试后必须全部得到合法结果或正确进入 needs_review。

rationale 使用至少 100 条按 relevant/边界/语言分层的人工审计样本。预先冻结“证据支持”和“严重编造”rubric；证据支持率 ≥ 95%，严重编造率 ≤ 1%。

模型选择规则：

- 先满足全部质量硬门，再比较速度、内存和稳定性。
- 晋级比较完整 cascade，而不是只比较 reranker；任何主要切片破门都不得晋级。
- recall 点估计相差不超过 1 个百分点且 F1 相差不超过 2 个百分点只定义为 tie band，不宣称统计显著；同时报告 Wilson interval 和 paired bootstrap。
- 质量进入 tie band 后，新方案的三次吞吐中位数必须至少快 20% 才替换 incumbent，否则保留现有方案。
- 第三方量化/转换使用相同 tokenizer、preprocess 和至少 10,000 个固定 pair；Kendall tau-b ≥ 0.995 作为数值诊断。
- 量化模型必须重新校准阈值，并通过完整 pipeline 质量门。关键阈值两侧分类一致率 ≥ 99.5%，阈值窗口和分母写入 parity manifest。
- 所谓“市面最强”在 task 中定义为本项目隐藏金标 winner，不能仅引用厂商榜单。

### 5.5 性能与恢复

目标机器：Apple Silicon M4 Max、36 GB unified memory。

- 冻结带 hash 的 1,000 篇性能回放集，固定输入 token 分布、10% 缺摘要、输出上限和所有阈值。
- normal 场景固定 15% Qwen 裁决；stress 场景固定 30%。两者使用冻结的 performance-only routing manifest 指定进入 Qwen 的 paper_id，不改质量阈值，也不能通过缩小灰区规避 Qwen 工作量。
- “完整 Stage 2”包含规则、rerank、Qwen、schema 校验和 SQLite 提交；只排除首次模型加载。
- 每个场景预热后独立运行三次。normal 场景三次均 ≤ 15 分钟；stress 场景三次均 ≤ 25 分钟，并报告中位数、p50 和 p95。
- 使用固定 10,000 篇语料进行 soak：OOM、进程崩溃、丢失结果和重复结果均为 0；所有请求失败必须落入 needs_review。
- 峰值进程/模型内存 ≤ 28 GB，不得出现 macOS memory-pressure critical；服务请求失败率 < 0.5%，热身后无持续无界内存增长。
- 基准记录机器型号、内存、macOS、oMLX/MLX、模型 hash、电源模式、后台负载和全部 batch/concurrency 参数。
- 每种模型最多常驻一个实例，禁止同模型多副本争抢 unified memory。根据实测结果选择同时 pin BGE 与 Qwen，或先完成全量 BGE 再批量处理灰区，不能因 LRU 频繁换模造成吞吐抖动。
- 单机多进程使用带 worker_id、expires_at、attempt 和 fencing token 的 SQLite 任务租约，只有当前 token 可以提交完成状态。
- 多机器严格遵循 3.5 的 snapshot、互斥分片、本地 SQLite、artifact bundle 和协调端幂等合并协议。
- 预留 QueueBackend 接口供未来接入外部队列，但本阶段不新增外部基础设施。
- 相同 paper/run 的结果必须具有唯一约束，重放分片不得重复提交。
- 模型失败回退顺序：主模型 → 具有自己已校准 threshold artifact 且通过同一质量门的备用本地模型 → needs_review。
- 批次部分失败必须拆回 paper 级状态并可 resume；fail-open 仅表示不判 irrelevant、不丢记录，默认停在人工队列。
- 禁止自动回退云端。

## 6. Stage 3：合法 PDF 获取

### 6.1 Provider 顺序

默认下载链：

1. 官方 venue/publisher 明确公开的直链。
2. PMC / Europe PMC Open Access subset。
3. Unpaywall 解析出的 OA location。
4. 已匹配的 arXiv 版本，或用户显式开启的 arXiv-only 候选。
5. 用户已获授权的浏览器会话与 download-authorized-papers skill。
6. manual_queue。

OpenAccessResolver 只返回不可信 AccessLocationCandidate 并逐 URL 写入 download_candidates；DownloadProvider.probe 结合策略生成 FetchDecision，状态闭集为 allow、needs_grant、manual、deny。缺少 grant 时返回 needs_grant；用户提供有效 grant 后必须重新 probe，只有新的 allow decision 才包含不可变 FetchRequest；manual 表示无法可靠自动裁决，deny 是当前 policy/purpose 下的终态。DownloadProvider.fetch 只能消费 FetchRequest。candidate 至少保存 candidate_id、paper_id、resolver、URL/landing URL、host、publication_version、license、access_basis、retrieved_at、raw evidence hash 和 provenance；同一论文的不同位置/版本不得挤进单个 paper_source 字段。

FetchRequest 由协调端持久化并绑定 request_id、candidate/policy/purpose/grant hashes、provider、created/expires_at 和幂等键；fetch 在产生任何网络副作用前回查该记录和当前 fencing token，字段不匹配、过期、撤销或手工构造的请求一律拒绝。

Crossref/OpenAlex/Semantic Scholar 返回 URL 不等于已获得下载、保存或再分发授权；Unpaywall 的 bronze 或 license=null 也不能自动提升为 open_license。fetch 前必须通过版本化 policy matrix，对 purpose（personal_research/internal_analysis/redistribution）× access_basis × license × publication_version × provider terms 产生 allow/needs_grant/manual/deny 及 reason code。默认只有与 purpose 相容且有条款证据的 open_license 可自动 allow；public_read_only、bronze/license=null、unknown 和 user_subscription 均不得自动保存，personal/internal 用途进入 needs_grant，条款无法机器判定时进入 manual，redistribution 在无兼容明确许可时 deny；user_supplied 不隐含再分发权。unknown/missing 不能被 require_access_basis=true 这种“非空检查”放行。

Europe PMC core API 的 `cc by` 许可证族标签在 machine-readable OA 记录与 provider terms 同时成立时，仅对 personal_research/internal_analysis 兼容；未提供具体版本时不自动通过 redistribution。

每个 DownloadProvider 必须声明：

- 是否需要认证。
- 是否支持主文、补充材料和版本选择。
- 是否允许无人值守。
- 可处理的域名/provider。
- 失败是否可重试。
- probe 与 fetch 的输入/输出 schema、幂等键和副作用边界；probe 禁止下载正文。由 needs_grant 经有效 grant 重新 probe 转为 allow 时，生成的 FetchRequest 必须绑定 candidate_id、policy version 和 authorization_grant_id；needs_grant 状态本身不得调用 fetch。

### 6.2 授权下载技能

用户提供的 download-authorized-papers-skill(1).zip 作为可选运行时能力：

- 实现前先审计压缩包内容、SKILL.md、脚本、依赖和权限。
- 审计记录必须绑定原始 ZIP SHA-256、安装后内容 digest、skill name/version 和依赖 lock/hash；任一内容或依赖漂移立即禁用，必须重新审计并使旧 authorization grant 失效。
- 安装后的实际 skill 名称由 doctor 探测，不在核心代码中硬编码用户目录。
- 只使用用户已有的合法访问权限和现有登录会话。
- 配置默认关闭；每次运行必须由用户创建 authorization grant，显式绑定 paper IDs 或 collection/selection snapshot hash、max_papers、域名 allowlist、purpose、confirmed_at、有效期、attended/unattended 模式和已审计 skill digest。scope 外或超上限论文直接 manual_required，不能借同一浏览器会话扩大范围。
- paper-agent grant create --kind download 只用 grant_defaults 生成 draft；paper-agent grant approve --grant <path> --hash <sha256> 以 4.4 的 canonical JSON + detached approval 规则写入不可变 SQLite grant；paper-agent grant revoke <grant_id> 只追加撤销事件。任何 scope 变化必须新建 grant。
- download 运行时只接受 authorization_grant_id/data_sharing_grant_id，从 SQLite 读取已批准内容并校验 content/approval hash、revocation、expiry、skill/dependency digests、selection scope 和 action；严禁读取 YAML grant_defaults 覆盖或扩张 grant，ID 与运行时内联 scope 并存时 schema 直接失败。
- 默认 attended 且 allow_unattended=false；“浏览器中已登录”本身不构成批量无人值守下载授权。
- unattended 只有在 skill 声明支持、站点允许且用户对本次 scope 明确授权后才能开启。
- 不记录 cookie、token、账号或页面中的敏感内容。
- authorization grant 只保存 grant_id、非敏感 scope、approved_by/at、过期时间、skill digest 和 approval record，不复制或持久化浏览器会话材料；每次 download_attempt 必须引用实际使用的 grant_id。
- Luna 只接收净化后的控制状态、目标 paper/candidate ID 和完成任务所需的最小页面内容；订阅页正文、截图、账户标识或其他敏感页面数据若要进入模型上下文，必须有独立 data-sharing grant，默认不传输、不落日志。
- 不绕过验证码、DRM、访问控制或站点限制。
- skill 缺失、未登录或站点不支持时进入 manual_queue，不阻塞其他论文。

普通公开 HTTP 下载不调用模型。只有需要浏览器代理判断或 skill 编排时才使用：

~~~text
codex exec -m gpt-5.6-luna
~~~

该调用使用冻结的 stage3_authorized_luna profile：reasoning_effort=low，只开放授权浏览器 skill 所需能力、专用临时工作目录和结构化结果 schema；禁止修改仓库或读取无关目录。禁止自动升级到 Sol。

### 6.3 文件校验与状态

- 主论文 PDF 默认开启；supplement 默认关闭，用户显式开启。
- 使用 MIME、魔数、文件大小和可解析性校验，不能把 HTML 错误页保存为 PDF。
- 保存内容 hash、来源 URL、下载时间、provider 和授权类别。
- 临时文件使用原子重命名，重试不会产生多个损坏副本。
- 状态至少包含 pending、downloaded、not_available、auth_required、manual_required、failed_retryable、failed_terminal。
- 下载中断后可从数据库状态恢复。

## 7. Stage 4：逐篇分析

### 7.1 模型路由

每篇论文分析明确调用：

~~~text
codex exec -m gpt-5.6-luna
~~~

不得使用 OpenRouter，不得静默切换模型。调用前 doctor 检查 Codex CLI、登录状态、模型可用性和版本。命令参数使用 argv 构造，禁止拼接未经转义的 shell 字符串。

CodexExecProfile 必须冻结并随 run 保存：

- provider=codex_cli、准确 model slug、reasoning 配置（CLI 支持时）。
- sandbox、网络策略、工具 allowlist、允许读取的工作目录。
- output schema、timeout、最大重试、环境变量 allowlist。
- 独立临时会话，不继承其他 Codex 对话上下文。

stage4_analysis_luna profile 固定 model=gpt-5.6-luna、reasoning_effort=medium、只读论文工作目录、网络关闭和分析输出 schema。YAML 中该模型字段使用 const 校验；用户级 Codex 默认模型或旧配置不能覆盖 CLI 的 model 参数。运行后核对实际 invocation metadata，模型不符即失败。

### 7.2 远程模型处理授权边界

codex exec 的 network=false 只限制 agent 工具联网，不表示 prompt/附件不会发送给远程模型 provider。任何 PDF bytes、抽取正文、订阅页面内容、用户提供文件或受限内部材料进入 Luna 前，都必须针对 remote_model_processing action 通过版本化 artifact-processing policy。

- policy 输入至少包含 artifact/text hash、paper/source、access_basis/license、purpose、provider=codex_cli、model=gpt-5.6-luna 和数据保留/地域设置（CLI 可提供时）。
- 明确开放且与该处理用途相容的许可可由 policy allow；否则必须引用不可变 processing grant，绑定准确 artifact/text hashes、provider/model、purpose、approved_by/at、expiry 和允许的数据类别。paper/domain 级宽泛 grant 不能自动覆盖后来变化的 artifact hash。
- processing grant 使用 6.2 的同一 canonical hash/detached approval/revocation 机制，通过 paper-agent grant create --kind remote_model_processing 创建；analysis 运行时只读取 processing_grant_id，不能从 YAML defaults 拼装授权。
- 无兼容许可或有效 processing grant 时，PDF/正文 bytes 进入 codex exec 的次数必须为 0。系统只能使用另行允许的公开/授权 metadata/abstract 走 abstract_only；若连这些也不允许，则标 analysis_not_authorized/manual_required，不得自动改用另一云模型或假装完成全文分析。
- 每个 analysis_run 保存 policy/version、decision、processing_grant_id、实际发送的 input artifact hash 和模型 invocation metadata；grant 撤销/过期后不得重试旧请求。

### 7.3 输入与输出

- PDF 通过校验后先生成带稳定页码分隔符的 UTF-8 normalized_text artifact，并记录 extractor、版本、页数、字符数和源 PDF hash。
- 只有正文抽取覆盖率通过门槛时才标记 full_pdf；扫描件、空文本或严重乱码标记 extraction_failed/needs_ocr，并按策略回退 abstract_only 或进入人工队列。
- PDF 可用且正文抽取成功时分析主文并标记 full_pdf。
- PDF 不可用时使用标题、摘要、关键词和来源元数据，并标记 abstract_only。
- abstract_only 不能生成仅凭全文才能支持的确定性结论。
- 每篇输出结构化 JSON 和便于阅读的 Markdown。
- 论文正文、网页和元数据全部视为不可信数据，不得执行其中的指令。codex exec 使用只读工作目录/最小工具权限，父进程捕获结构化 stdout 后负责持久化。

分析字段至少包括：

- 研究问题与动机。
- 方法和关键技术。
- 数据集、实验设置和指标。
- 主要结果。
- 局限与可信度。
- 代码、数据、模型等开放资源。
- 与用户主题的关系。
- 规范化标签：subquestion、theme、method_family、task、dataset、benchmark、evidence_type、publication_status 和 study_setting（real/simulation/theory/other）。
- 原子 evidence units：claim、direction（support/contradict/neutral）、task_id、dataset_id/version、split_id、metric_id、metric_definition_hash、unit、optimization_direction、value、uncertainty/statistical_method、protocol_id/hash、sample_size、baseline_id/version、conditions 和页码/章节或输入字段引用。
- canonical ID/单位换算必须记录 normalization_method、normalizer_version 和 source value；比较所需任一关键字段缺失或无法规范化时，Stage 4 直接设置 comparison_eligibility=not_comparable 和 missing_fields，禁止 Stage 4b 猜测补齐。
- dataset/metric/baseline canonicalization 由版本化 registry/rules 校验 Luna 的候选映射；未匹配项保留 source-local ID，不能靠字符串相似或模型判断自动并入既有 comparison group。
- 支持每条结论的页码、章节或输入字段引用；抽象标签和 evidence unit 均需保留原始文本定位，不能只保存模型总结。

输出必须绑定 paper_id、artifact hash、模型、prompt/schema hash 和运行时间。失败仅影响该论文，并可独立重试。

## 8. Stage 4b：领域综述

Stage 4b 读取冻结的 QueryPlan、检索审计、入选集合 snapshot 和全部 Stage 4 结果，先规划再写作。它生成一个可审计的中文 Markdown 领域综述，而不是按论文顺序拼接摘要。

### 8.1 冻结 ReportPlan

正文生成前必须产出并冻结 REPORT_PLAN.json；同一 report_run 内不得静默改变范围、章节或评价标准。ReportPlan 至少包含：

- objective、audience、主问题/子问题，以及一条可检验但不预填答案的总括性 synthesis question。
- 时间、venue、文献类型、语言、纳入/排除范围，并引用 QueryPlan hash 和 corpus snapshot hash。
- 章节顺序、每章服务的子问题、目标字数/token、关键证据要求和允许的 evidence level。
- 分类轴：subquestion、theme、method_family、task、dataset、benchmark、时间、venue/出版状态、证据类型和 study_setting。
- cohort rules：recent cutoff、foundational paper 判定/显式种子、peer-review 状态和 real/simulation/theory 分类规则；不能事后为配合结论改阈值。
- 每篇论文的预定 section membership；允许一篇进入多章，但必须指定 primary section。
- 计划生成的对比表、趋势统计、资源表和附录。
- execution_strategy=one_shot，以及冻结预算 max_sol_calls=1、max_retries=0、audit_calls=0、repair_calls=0 和完整输入 token 上限。
- plan hash、schema/prompt hash、状态（draft/approved/superseded）和 approval record（approved_hash、approved_by、approved_at、approval_method）。

CLI 必须提供 report --plan-only、report approve --plan <path> --hash <sha256> 和 report --plan <path>。有人值守运行也必须写入显式 approval record；无人值守运行必须同时配置 approved plan path/hash。执行前重算 corpus、QueryPlan、schema/prompt 和配置 hash，任一偏差即拒绝运行并生成新 draft。改变研究问题或纳入范围会创建新 report_run，不能覆盖旧计划。

ReportPlan 使用与 4.4 相同的 canonical JSON/content hash 与 detached approval 规则；approval metadata、时间戳和 plan ID 不进入 content hash，corpus/QueryPlan/prompt/schema/config hashes 必须进入。

### 8.2 检索流与语料清单

报告必须包含 PRISMA-style 检索流量说明，但不得宣称符合临床 PRISMA 标准：

- 每个 source/query round 的 raw discovered、去重后唯一候选、Stage 2 筛选、按原因排除、纳入、full_pdf、abstract_only 和缺失数量。
- user_library、newly_discovered、citation_snowball 三类来源分别统计。
- foundational/recent、peer_reviewed/workshop/preprint、full_pdf/abstract_only、real/simulation/theory 分层统计。
- 附录引用 search audit 和全部 query variants；required provider 失败或 budget_exhausted 必须在执行摘要和限制章节显式出现。

### 8.3 全量输入冻结与单次聚合

- Stage 4 分析是唯一的逐篇模型 map artifact；Stage 4b 不重复逐篇调用。
- 协调端按 stable paper_id 确定性排序，把冻结语料中每篇论文的完整 Luna 报告、canonical metadata、evidence locator、publication/full-text 状态、比较条件、ReportPlan、corpus snapshot 和 search audit 一次打包为唯一 `one_shot_report` 输入；每篇 Luna 报告必须且只能出现一次。
- 输入包同时携带支持、反对和未知证据，不允许为节省 token 丢弃冲突、限制、abstract_only 或 not_comparable 信息。
- 单篇论文可以服务多个子问题，但 coverage ledger 只按 stable paper_id 计一次总体覆盖，并记录所有消费节点。
- dispatch 前必须使用冻结 tokenizer/估算器对完整 prompt 做预算门禁；若完整输入超过 ReportPlan 或模型上下文上限，则状态为 incomplete/manual_required，记录预算证据并保持 Sol 调用数为 0。
- 禁止 shard、section/cross-section/final reduce、抽样、取前 N 篇、截断单篇 Luna 报告或静默缩小 approved corpus。需要缩小范围时必须创建并批准新的 ReportPlan 和新的 report_run。

### 8.4 Claims-Evidence Matrix

在生成最终正文前维护 CLAIMS_EVIDENCE.jsonl。每条记录至少包含：

- claim_id、claim_key、research_question_id、report_section、claim_text、claim_type（finding/trend/comparison/gap/recommendation/corpus_stat）。claim_key 至少结构化 subject_id、predicate_id、object_or_scope_id、qualifier/context hash 和 comparison_group_id（适用时）；claim_id 使用固定 namespace 对 canonical claim_key 生成 UUIDv5，不依赖自然语言措辞。
- supporting_evidence[] 与 contradicting_evidence[]；每项是 typed evidence ref。paper_evidence 绑定 paper_id、analysis_run_id、locator、原始 evidence unit 和条件；corpus_evidence 绑定 search_plan/source/query/统计单元及计算方式。
- evidence_level：full_text_direct、full_text_inferred、abstract_direct、metadata_only 或 corpus_stat。
- comparison_group_id、confidence、known_limitations 和 status（supported/mixed/insufficient）。

唯一一次 `one_shot_report` 必须返回结构化 one-shot draft：章节 blocks、draft claim records、supporting/contradicting evidence refs、未解决冲突和 claim relation 候选。Sol 只能引用输入 allowlist 中已经存在的 stable paper_id、analysis/evidence ref 和 ReportPlan section/subquestion；不得凭标题、记忆或常识新增论文、定位、数值、claim ID、hash 或书目。

模型返回的 claim/block 引用只是草稿引用；协调端在本地通过 schema、controlled vocabulary/claim-key registry 和 paper/citation/evidence allowlist 后，确定性生成 stable claim UUID、comparison group、block ID、Claims-Evidence Matrix 与 ReportDocument AST。block_kind 枚举冻结为 prose、list_item、table_cell、caption；每个实质性 block 必须具有 section_id、至少一个 claim_id 和 citation paper_ids。heading、布局、目录、参考文献等非实质性块只允许协调端从 ReportPlan/metadata 生成。协调端由 AST 确定性渲染 Markdown，并在机器可读 sidecar 保存 block↔claim↔citation 绑定；任一模型文本/表格单元没有有效 claim、正文与 sidecar 不一致，或绑定到不存在 claim 的 block 都使 verifier 失败。

增量 run 先按 stable claim_id 对齐；措辞变化不改变 ID。split/merged/refined/superseded/retired 必须写入 claim_relations，绑定 previous/current claim IDs、reason 和 evidence diff；无法确认的映射保持 unmapped，不得仅靠文本相似度静默继承历史结论。

最终报告的实质性事实、趋势、对比、空白和建议都必须绑定 claim_id；没有足够证据的内容只能写成待验证问题。每篇入选论文必须至少贡献一条 evidence、进入一张资源/背景表，或被显式标为 background_only 并说明原因；仅“进入过某个 token 分块”不算有效覆盖。

### 8.5 矛盾、证据强度与可比性

- 不用多数票抹平冲突。对于相反结论，同时呈现双方论文，并优先检查 dataset、split、metric definition、protocol、sample、baseline/version、assumptions、年份和出版状态差异。
- comparison_groups 持久化 canonical task、dataset/version、split、metric definition/unit/direction、protocol、关键 baseline/version、normalizer/registry versions 和 comparison_key；comparison_group_id 由固定 namespace + canonical comparison_key 生成 UUIDv5，跨 run 相同条件保持稳定，模型不能自由命名后宣称可比。
- 数值只能在同一 comparison_group 且 comparison_eligibility=comparable 时并列或排序；关键字段缺失/冲突或 normalization 无 provenance 时强制标注“不可直接比较”，不得计算统一胜率或平均提升。
- full_pdf 与 abstract_only、peer-reviewed 与 preprint/workshop、真实测量与 simulation/theory 必须分层表达；引用数只能作为来源覆盖信号，不能替代证据质量判断。
- “研究空白”必须同时说明检索范围和未发现证据的边界；不能把未检索到等同于不存在。

### 8.6 最终报告章节合同

最终中文 Markdown 至少包含：

1. 执行摘要：一条总括结论、3–7 条主要发现、适用边界和不确定性。
2. 研究范围与方法：问题、时间/来源、纳入排除规则、模型与证据限制。
3. 检索流与语料画像：来源覆盖、去重/筛选流程和分层统计。
4. 领域图景与分类体系：趋势、主题、方法族及时间演化。
5. 按子问题/主题的证据综合：不是逐篇摘要列表。
6. 方法、benchmark、数据集、代码和模型资源对比矩阵。
7. 矛盾结论、不可比项与证据局限。
8. 研究空白、可检验机会与未来方向。
9. 面向目标受众的实践结论/建议及适用条件。
10. 本报告限制、未完成来源和更新状态。
11. 参考文献与附录：query manifest、排除原因、coverage ledger 和 claim ledger 链接。

### 8.7 Sol 路由、审计与发布

最终聚合明确调用：

~~~text
codex exec -m gpt-5.6-sol
~~~

approved Stage 4b 执行使用冻结的 `stage4b_oneshot_sol` security/model profile，并且只允许一个模型 call_kind：

| call_kind | 输出 schema |
|---|---|
| one_shot_report（approved run 必须且只能一次） | one-shot-report.schema.json |

`planning_assist` 仅可在 ReportPlan 批准前作为可选的规划辅助，不能接收 Stage 4 analysis/evidence，不属于 approved Stage 4b 执行，也不能在执行期间或失败后用于追加调用。旧 `section_reduce`、`cross_section_reduce`、`final_reduce`、`quality_audit` 和 `repair` 资源只允许为 legacy 配置迁移保留；`execution_strategy=one_shot` 时调度器必须拒绝调用它们。

Stage 4 analysis/evidence 仍是论文内容的派生数据。任何 analysis、evidence pack、claim ledger 或报告草稿进入 Sol 前，必须用其 lineage/source artifact hashes 对 remote_model_processing action 重新评估 provider=codex_cli、model=gpt-5.6-sol；Luna processing grant 不自动覆盖 Sol。无兼容许可或精确授权的论文不能把其全文派生内容发送给 Sol；若只允许 metadata/abstract，则降到相应 evidence level 并在 ReportPlan/corpus snapshot 中冻结，否则 report 为 incomplete。network=false 同样不代表模型载荷留在本机。

要求：

- 不得以降低成本为由静默改成 Luna。
- Stage 4 已生成的全部逐篇 Luna 分析就是唯一模型输入语料；approved run 只做确定性全量打包，并严格 dispatch 一次 `one_shot_report`。不得新增模型 map、reduce、audit 或 repair 调用。
- `stage4b_oneshot_sol` 固定 model=gpt-5.6-sol、reasoning_effort=high、只读输入、网络关闭、max_retries=0 和 one-shot output schema；模型字段使用 const 校验并核对实际 invocation metadata。
- ReportPlan 和运行时都必须冻结 `max_sol_calls=1`、`max_retries=0`、`audit_calls=0`、`repair_calls=0`。完整输入超过 token/context 预算、缺少任一输入或授权不完整时，必须在 dispatch 前停止，Sol 调用数为 0；禁止通过 shard、reduce、抽样、截断、降级模型或省略论文来凑预算。
- 数据库必须先以唯一约束原子预约该 report_run 的唯一 dispatch，再启动 codex exec。并发 worker/resume 只能观察并复用 `pending/running/complete/manual_required/failed_terminal` 状态，不能获得第二个 dispatch 权限。
- Codex 启动后的 timeout、连接中断、进程丢失或无法证明“尚未发送”的 uncertain outcome 一律终止该 report_run，不得自动或人工 resume 重发。若确需再次调用，必须创建新的 report_run 并重新批准其不可变输入；旧 run 保留完整审计记录。
- Sol 只能发出 allowlist 中的 `[@paper_id]` 引用标记；协调端从已校验 canonical metadata 按冻结 CSL/style 确定性生成参考文献，一个 canonical paper 只生成一条，未知/重复/缺字段引用均使校验失败，禁止让模型自由生成书目。
- 唯一 Sol 输出落盘后，全部后处理在本地完成：schema 校验、draft ref→stable ID normalize、Claims-Evidence/AST/comparison/coverage 构建、确定性 renderer、report verifier、确定性 audit 和 publish；禁止把失败输出再次发送给 Sol 修复。
- 本地 report verifier 检查 plan 章节、paper/claim coverage、claim-evidence、stable paper_id、citation target、重复引用、evidence level、comparison group、冲突/不可比披露和未完成来源。本地 deterministic audit 使用冻结 rubric 产生机器可读 finding，severity 枚举为 blocker、major、minor、note。
- complete 必须 schema、normalize、verifier 和 deterministic audit 全部通过且 blocker=0、major=0；任何失败都保留原始输出和本地审计，状态为 incomplete/failed_terminal，不更新 latest，也不触发第二次 Sol 调用。
- 最终事实陈述使用 stable paper_id/claim_id，并链接对应分析、证据定位或来源；不能把摘要推断表述成全文事实。
- 输出写入不可变 reports/<report_run_id>/，至少包含 REPORT_PLAN.json、SEARCH_AUDIT.json、CLAIMS_EVIDENCE.jsonl、COMPARISON_GROUPS.json、CLAIM_RELATIONS.json、REPORT_DOCUMENT.json、COVERAGE.json、REPORT.md、AUDIT.json 和 artifact manifest；校验完成后才原子更新 reports/latest.md。历史 run 不删除。
- 增量报告另生成 REPORT_DIFF.md/json，列出 query/范围变化、added/removed/changed papers、正式版/撤稿变化、按 stable claim_id 对齐的 added/changed/retired/split/merged claims、evidence diff、受影响 section 和未变化章节；无法映射的 claim 显式列为 unmapped，禁止无说明地整篇覆盖。

## 9. 配置规范

采用版本化 YAML。示例：

~~~yaml
version: 2

project:
  topic: "Diffusion Models for Molecular Generation"
  output_dir: "./paper_research"
  report_language: "zh-CN"

storage:
  sqlite_path: "./paper_research/papers.sqlite3"
  wal: true
  exports:
    - format: jsonl
      path: "./paper_research/exports/papers.jsonl"
    - format: csv
      path: "./paper_research/exports/papers.csv"

sources:
  approved_plan:
    input_path: "./paper_research/search/latest-approved.json"
    content_hash: null  # 运行前填写 search approve 输出的 SHA-256
    required: true
  plan_defaults:  # 只用于编译 draft；approved plan 生成后不再是运行时覆盖项
    required_roles: [venue_primary, search, citation, metadata_verifier]
    provider_policy: all_resolved
    user_seeds:
      inputs: []  # DOI/arXiv/URL/BibTeX/RIS/CSL-JSON/Zotero export/local PDF
    venues:
      - descriptor: "venues/iclr.yaml"
        years: [2024, 2025]
      - descriptor: "venues/nature_machine_intelligence.yaml"
        date_from: "2024-01-01"
    arxiv:
      enabled: true
      roles: [search, metadata_enricher, metadata_verifier, oa_resolver]
      categories: [cs.AI, cs.LG, cs.CL]
      date_from: "2024-01-01"
      include_arxiv_candidates: false
      use_matched_arxiv_as_download_source: true
      global_min_interval_seconds: 3
    providers:
      crossref:
        enabled: true
        roles: [search, metadata_enricher, metadata_verifier]
        mailto_env: CROSSREF_MAILTO
        rate_policy: response_headers
      dblp:
        enabled: auto_for_cs
        roles: [search, metadata_enricher, metadata_verifier]
        bulk_snapshot_preferred: true
      semantic_scholar:
        enabled: true
        roles: [search, citation, metadata_enricher, metadata_verifier]
        api_key_env: SEMANTIC_SCHOLAR_API_KEY
        use_batch_endpoints: true
      openalex:
        enabled: true
        roles: [search, citation, metadata_enricher, metadata_verifier]
        api_key_env: OPENALEX_API_KEY
        mode: api
        snapshot_path: null  # 仅使用用户预先提供并校验的 snapshot，禁止自动下载全量快照
      pubmed:
        enabled: auto_for_biomed
        roles: [search, metadata_enricher, metadata_verifier]
        api_key_env: NCBI_API_KEY
        tool: paper-agent
        email_env: NCBI_EMAIL
      europe_pmc:
        enabled: auto_for_biomed
        roles: [search, citation, metadata_enricher, metadata_verifier, oa_resolver]
      unpaywall:
        enabled: true
        roles: [oa_resolver]
        email_env: UNPAYWALL_EMAIL
      exa: {enabled: false, roles: [search]}
      gemini_search: {enabled: false, roles: [search]}
      deepxiv: {enabled: false, roles: [metadata_enricher]}
      alphaxiv: {enabled: false, roles: [metadata_enricher]}
    verification:
      prefer_formal_version: true
      require_two_sources_when_feasible: true
      preserve_conflicts: true
    citation_snowball:
      enabled: true
      directions: [references, citations]
      max_depth: 2
      max_rounds: 2
      seed_selector: relevant_topk_by_subquestion_v1
      seeds_per_subquestion: 20
      max_per_seed_per_source: 500
      max_candidates_per_round: 50000
      max_seconds_per_round: 14400
    saturation:
      min_unique_included_yield: 0.05
      consecutive_low_yield_rounds: 2
      max_candidates: 200000
      max_requests: 20000
  plugin_allowlist: []

filter:
  mode: cascade
  deterministic:
    include: []
    exclude_document_types: [editorial, retraction]
  reranker:
    backend: omlx_rerank
    model: "BAAI/bge-reranker-v2-m3"
    source_repo: "BAAI/bge-reranker-v2-m3"
    source_revision: "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
    format: "fp32"
    document_batch_size: 32
    candidate_batch_sizes: [16, 32, 64]
    max_in_flight: 2
    thresholds_artifact: "./paper_research/models/stage2-thresholds.json"
  adjudicator:
    backend: omlx_chat
    model: "mlx-community/Qwen3.5-9B-8bit"
    revision: "16daa4818c54ce5f5436f929d52542eb65bbed9d"
    chat_template_kwargs:
      enable_thinking: false
    temperature: 0
    seed: 42
    stream: false
    max_tokens: 256
    max_context_window: 16384
    structured_output:
      transport: "extra_body.structured_outputs.json"
      schema: "./schemas/filter-decision.schema.json"
    client_concurrency: 4
    server_max_concurrent_requests: 8
    benchmark_concurrency_pairs:
      - [4, 8]
      - [8, 8]
      - [8, 16]
      - [16, 16]
    expected_max_share: 0.15
  fail_open: true

download:
  include_supplements: false
  resolvers: [publisher_public, europe_pmc, unpaywall, arxiv]
  providers: [public_direct, europe_pmc, unpaywall_location, arxiv, authorized_skill, manual]
  purpose: personal_research
  policy_matrix: "./policies/download-access-v2.yaml"
  require_access_basis: true
  treat_unknown_license_as_open: false
  authorized_skill:
    enabled: false
    skill_name: "download-authorized-papers"
    authorization_grant_id: null
    data_sharing_grant_id: null
    profile: "stage3_authorized_luna"
    codex_model: "gpt-5.6-luna"
    reasoning_effort: low
    grant_defaults:  # 仅供 grant create --kind download 生成 draft，download 运行时禁止读取这些字段扩权
      source_zip_sha256: null
      installed_content_sha256: null
      dependency_lock_sha256: null
      allowed_domains: []
      paper_ids: []
      collection_snapshot_hash: null
      selection_snapshot_hash: null
      max_papers: null
      actions: [download, store, extract]
      purpose: personal_research
      mode: attended
      allow_unattended: false
      authorization_expires_at: null

analysis:
  profile: "stage4_analysis_luna"
  provider: codex_exec
  model: "gpt-5.6-luna"
  reasoning_effort: medium
  sandbox: read_only
  network: false
  output_schema: "./schemas/paper-analysis.schema.json"
  workers: 4
  allow_abstract_only: true
  remote_model_processing:
    policy_matrix: "./policies/artifact-processing-v1.yaml"
    processing_grant_id: null
    require_artifact_hash_scope: true

summary:
  enabled: true
  execution_strategy: one_shot
  profile: "stage4b_oneshot_sol"
  provider: codex_exec
  model: "gpt-5.6-sol"
  reasoning_effort: high
  sandbox: read_only
  network: false
  schemas:
    planning_assist: "./schemas/report-plan.schema.json"
    section_reduce: "./schemas/section-synthesis.schema.json"
    cross_section_reduce: "./schemas/cross-section-synthesis.schema.json"
    final_reduce: "./schemas/report-document.schema.json"
    quality_audit: "./schemas/report-audit.schema.json"
    repair: "./schemas/report-repair.schema.json"
    one_shot_report: "./schemas/one-shot-report.schema.json"
  prompts:
    planning_assist: "./prompts/report-plan.md"
    section_reduce: "./prompts/section-synthesis.md"
    cross_section_reduce: "./prompts/cross-section-synthesis.md"
    final_reduce: "./prompts/final-report.md"
    quality_audit: "./prompts/report-audit.md"
    repair: "./prompts/report-repair.md"
    one_shot_report: "./prompts/one-shot-report.md"
  format: markdown
  language: "zh-CN"
  report_plan:
    input_path: null  # unattended 时必须指向已批准的不可变 plan
    content_hash: null
    required_for_unattended: true
    classification_axes: [subquestion, theme, method_family, task, dataset, benchmark, time, publication_status, evidence_type, study_setting]
  require_search_audit: true
  require_complete_coverage: true
  require_claim_evidence: true
  semantic_chunking: false
  remote_model_processing:
    policy_matrix: "./policies/artifact-processing-v1.yaml"
    processing_grant_id: null
    require_lineage_hash_scope: true
  citations:
    marker: stable_paper_id
    style: ieee
    bibliography_from_canonical_metadata: true
  final_audit:
    deterministic: true
    independent_sol_session: false
    rubric: "./policies/report-audit-rubric-v1.yaml"
    max_blocker_findings: 0
    max_major_findings: 0
    max_repair_calls: 0
    reverify_and_reaudit_after_repair: false
  immutable_run_directories: true
  update_latest_after_pass: true
  emit_incremental_diff: true
~~~

要求：

- schema 校验失败必须在启动前报错。
- 配置使用 command-specific 条件校验：plan/dry-run 允许 approval/grant ID 为 null；search run 和 unattended report 要求匹配的 content hash/approval record；enabled authorized_skill 只允许通过非空 grant IDs 引用 SQLite 中完整、有效、未撤销的授权，禁止把 grant_defaults 当运行时 scope；Stage 4 full_pdf 还必须有 artifact-processing allow decision。
- sources.plan_defaults 只参与 search plan 编译；approved_plan.content_hash 一旦存在，运行时不得再读取 defaults 覆盖 venue/provider/role/mode/query/budget。任何 default/runtime resolution 差异只能生成新 draft。
- plugin_allowlist 条目必须是 distribution/version/provider/entry_point/artifact_sha256 的结构化对象，禁止字符串包名；download/browser authorization grant 必须引用已审计 skill digest 和批准 hash，remote-processing grant 必须引用 artifact/model/provider hashes。
- provider 的 enabled/role/capability、required/optional、API/snapshot 模式、QPS/concurrency/cache、credential env、许可策略和 query compiler version 都属于运行快照；secret 的值只做存在性检查，永不写入快照。
- OpenAlex API 模式必须预估当前 cost/credit；若用户已预置固定日期和 hash 的 CC0 snapshot，大规模运行可显式改用，但系统不得自动下载全量快照或新增基础设施。Semantic Scholar 优先 batch/bulk endpoint，DBLP/PubMed/Europe PMC 大规模回放优先用户已配置的官方 bulk snapshot/FTP；不得用并发绕过全局 provider 限流。
- arXiv legacy API/OAI/RSS 的全局限流按用户控制下所有 worker/机器汇总执行，默认单连接且请求间隔至少 3 秒；多机分片不能各自独立放大速率。
- QueryPlan、ReportPlan、provider acceptance manifest 和 corpus snapshot 都必须保存 schema version 与 hash；resume 时 hash 不一致必须创建新 run 或显式迁移。
- 新配置默认 `execution_strategy=one_shot` 和 `profile=stage4b_oneshot_sol`；approved run 只解析并调用独立的 `one_shot_report` prompt/schema。legacy reduce 资源即使为迁移兼容而存在，也不得被 one-shot 调度器调用。
- v2 schema 将 authorized_skill.codex_model、analysis.model 和 summary.model 分别约束为 gpt-5.6-luna、gpt-5.6-luna 和 gpt-5.6-sol；环境变量和用户默认配置不得覆盖。
- 环境变量只允许覆盖 secret、端点和机器相关参数；业务配置仍写入运行快照。
- 每次 run 保存解析后的完整配置及其 hash。
- 旧配置迁移需覆盖原有 database.format、filter.semantic、analysis.model、summary_model 等字段，并明确报告废弃项。
- OpenRouter 相关配置一律报迁移错误或给出 codex exec 替代说明，不能继续生效。

## 10. CLI 与 Codex skill

目标代码布局：

~~~text
src/paper_agent/          # 唯一核心实现
tests/                    # 单元、契约和集成测试
skills/paper-agent/       # 可安装的薄 Codex skill
schemas/                  # 配置与模型输出 schema
prompts/                  # 冻结的各 call_kind prompt
policies/                 # 下载授权矩阵与报告审计 rubric
venues/                   # 内置 VenueDescriptor
providers/                # 受版本控制的 provider role/capability manifests
~~~

- 核心包使用标准 pyproject.toml 构建并提供 paper-agent console entry point。
- 支持 Python 3.11–3.13；macOS Apple Silicon + oMLX 是首要部署目标。
- .opencode 下的代码和 pytest 路径迁出；完成迁移后不得存在对 .opencode 的运行时 import 或测试依赖。
- Codex skill 的安装包可以复制/链接到用户 Codex skills 目录，但业务代码始终从已安装的 paper_agent Python 包调用。

CLI 至少提供：

~~~text
paper-agent doctor
paper-agent search plan
paper-agent search approve --plan <QUERY_PLAN.json> --hash <sha256>
paper-agent search run --plan <QUERY_PLAN.json>
paper-agent search expand-citations
paper-agent search audit
paper-agent import-seeds
paper-agent crawl
paper-agent filter
paper-agent grant create --kind <download|browser_data_sharing|remote_model_processing>
paper-agent grant approve --grant <GRANT.json> --hash <sha256>
paper-agent grant revoke <grant_id>
paper-agent download
paper-agent analyze
paper-agent report --plan-only
paper-agent report approve --plan <REPORT_PLAN.json> --hash <sha256>
paper-agent report --plan <REPORT_PLAN.json>
paper-agent report --diff-from <report_run_id>
paper-agent verify-report
paper-agent run
paper-agent resume
paper-agent export
paper-agent migrate-config
paper-agent benchmark-stage2
~~~

通用要求：

- 支持 --config、--run-id、--dry-run 和结构化日志。
- search plan 输出 compiled QueryPlan 草案和 provider/request/cost 上限预估；search run 只接受带匹配 approval record 的 plan 且拒绝运行时漂移；crawl 保留为仅运行 venue descriptors 的兼容别名并同样产生 search audit。
- import-seeds 支持 DOI/arXiv/URL/BibTeX/RIS/CSL-JSON/Zotero export/local PDF；外部笔记仅作为 query/evidence hint，不直接覆盖 canonical metadata。
- 每个阶段可单独运行，也可从数据库断点续跑。
- 重复执行相同阶段必须幂等。
- SIGINT/SIGTERM 后安全保存状态。
- doctor 检查数据库、配置、插件、provider capability/credential/限流、bulk snapshot hash、oMLX、模型 revision、Codex CLI、登录状态、磁盘空间及授权下载 skill。

Codex skill：

- 只负责收集用户意图、生成/确认配置、调用同一 CLI 和解释结果。
- 不包含另一套爬虫、过滤或数据库实现。
- 对高成本 Stage 4b、授权浏览器操作和大规模回放给出明确预估。
- 不在日志或 prompt 中泄露凭据。

## 11. 可观测性、安全与成本控制

- 所有日志包含 run_id、stage、paper_id/provider（适用时）和事件码。
- 提供每阶段及每来源 raw_discovered、unique_after_dedup、overlap、screened、excluded_by_reason、included、citation-expanded、downloaded、full_pdf/abstract_only、failed、token/cost 等统计。
- 提供 query/source/round 可下钻的 search audit、provider credit/限流观测，以及 report 的 paper/claim/section coverage；引文数始终带 provider 和采集时间。
- 记录模型输入输出 hash；默认不把完整 PDF 或敏感浏览器内容写入日志。
- 重试次数、并发、最大样本量和 Codex 调用预算均可配置。
- 任何超过 adjudicator 比例、调用预算、错误率或内存水位的行为必须报警。
- 禁止自动模型升级、自动云回退和无限重试。
- 外部插件、技能压缩包和下载文件均按不可信输入处理。
- 所有输出路径需防止目录穿越和文件名冲突。

## 12. 测试策略

### 12.1 单元与契约测试

- 每个 VenueAdapter/SearchProvider/CitationProvider/LibraryProvider/MetadataEnricher/MetadataVerifier/OpenAccessResolver 使用固定 fixture 测试分页、日期、字段映射、capability 和统一 SourceBatch envelope。
- VenueDescriptor/provider manifest schema；entry point 在 import 前校验 distribution/version/digest；hash drift 禁用；第三方子进程权限/IPC/超时测试。
- QueryPlan/QuerySpec schema、source-specific query compiler、compiled provider resolution、approve/hash、运行时漂移拒绝、冻结/重放和 plan 变更新建 run 测试。
- 全 resolved source fan-out、单源失败隔离、required source 失败导致 incomplete，以及显式请求不可用 source 不得静默跳过测试。
- 去重优先级、冲突和幂等合并测试。
- 正式版/preprint 关系、双源元数据校验、字段 provenance、verification_status 和逐来源 citation_count 测试。
- provider independence/upstream-family 反例，以及 primary 失败但聚合源命中时只能 venue_candidate、不能 official_confirmed 的 membership 测试。
- backward/forward 引用方向、root/depth、seed selector/tie-break、环路去重、逐轮分母、每 seed 上限、未完成轮不得饱和、低收益饱和和 budget_exhausted 测试。
- SQLite migration、WAL 并发、任务租约和崩溃恢复测试。
- 多机 snapshot/shard epoch、迟到结果 fencing、artifact 校验、幂等合并和冲突隔离测试。
- AccessLocationCandidate 持久化、probe 无副作用、FetchRequest 不可伪造、fetch 幂等、provider 顺序、失败分类和文件校验测试。
- OpenAccessResolver 与 DownloadProvider 分离，purpose×access_basis×license×version×terms policy matrix；allow/needs_grant/manual/deny 全状态与 grant 后 re-probe 转换；URL/非空 access_basis 不自动等于授权测试。
- 授权下载默认关闭；grant canonical hash/approval/revocation、paper/collection/domain/purpose/action scope、skill ZIP/installed digest 漂移、attended/unattended、过期，以及篡改 YAML grant_defaults 不能扩大已批准 scope 的测试。
- Stage 4/4b artifact-processing policy/processing grant 测试；无兼容许可/grant 时 spy codex exec 断言 PDF/normalized text 或其受限派生 analysis/evidence 进入 Luna/Sol 调用次数为 0，只能走获准 abstract_only/metadata-only 或 analysis/report incomplete。
- Codex/oMLX 客户端使用 fake server 验证超时、schema、profile 常量和实际模型核对；Stage 4b 另须证明 `max_retries=0`，timeout/uncertain outcome 不重发。
- ReportPlan approval/hash/运行时漂移、完整 Luna 报告一次打包、`one_shot_report` 唯一 prompt/schema、预算或授权失败时 0 dispatch、唯一 dispatch 数据库约束、并发 resume 不重复调用、Claims-Evidence Matrix、ReportDocument block_kind 与 AST block↔claim↔citation 绑定、renderer/sidecar hash、stable claim UUID/lineage、持久 comparison key/not_comparable、paper/claim coverage、本地 deterministic normalize/verifier/audit、不可变版本和增量 diff 测试。

### 12.2 集成测试

- 从 user seed + 多源 fixture search/crawl 到 SQLite、citation expansion、filter、mock download、mock analysis、mock report 的完整流水线。
- 同一配置运行两次不重复数据。
- 中途终止后 resume 得到与一次完成相同结果；Stage 4b 只有在尚未 dispatch 时可继续，已 dispatch 的 running/timeout/uncertain 状态只能观察或终止，不能再次调用 Sol。
- 一个 venue、PDF 或模型请求失败不影响其他工作项。
- arXiv 默认不进入最终报告，显式开启后才进入。
- search audit 的逐来源/逐轮数字能与原始 fixture、去重、排除原因和最终集合对账。
- 报告 paper coverage 与最终入选集合完全一致；每个实质性 claim 都有有效 evidence，漏论文/漏证据/坏引用/不可比数值会使 verifier 失败。
- 同一 corpus 和 ReportPlan 在打乱输入顺序后仍产生相同 one-shot input hash、确定性 section/claim normalization 和 coverage ledger，且 fake Codex 计数严格为 1。
- 冲突证据、abstract_only 和 incomplete source 在报告中按合同披露；不得被 one-shot 输出或本地 normalize 抹去。
- 增量 run 正确输出 added/removed/changed papers 与受影响 claim/section，旧报告仍可读取。

### 12.3 CI

- CI 禁止实时爬站、真实浏览器授权、模型下载和 Codex 消耗。
- 使用小型 fixtures、mock oMLX、mock codex exec 和临时 SQLite。
- 网络型 smoke test 单独标记，只有人工显式触发。

## 13. 实施阶段与门禁

### Phase 0：基线与设计冻结

1. 确认 feature/crawler-adapters 为实现基线。
2. 生成 v2 配置、compiled QueryPlan/approval、VenueDescriptor/provider manifest、ReportPlan/approval、paper analysis、claim-evidence、Stage 4b `one_shot_report` 与 report schema，以及 SQLite ERD、唯一 dispatch 状态机和 migration 设计。
3. 审计 download-authorized-papers skill 压缩包。
4. 建立 ADR，记录模型路由、插件 trust/digest/子进程边界、下载 policy matrix、报告 audit gate 和 fail-open 决策。

门禁：schema、接口、状态机和迁移方案通过评审。

### Phase 1：存储与迁移

1. 实现 SQLite repository 和 migrations。
2. 实现 canonical paper/source/collection/artifact/run、search plan/query/source run/citation edge/screening、provider registration、download candidate/authorization grant、report plan/claim/evidence 模型。
3. 实现旧 JSON/YAML 导入和 JSONL/CSV 导出。
4. 实现任务租约、幂等和 resume。
5. 实现多机 snapshot/shard/artifact bundle/merge 协议。

门禁：迁移、去重、并发和恢复测试通过。

### Phase 2：爬虫扩展

1. 抽取 VenueAdapter、SearchProvider、CitationProvider、LibraryProvider、MetadataEnricher/Verifier 和 OpenAccessResolver。
2. 实现 QueryPlan、source-specific query compiler、只读 fan-out、统一 envelope 和单写者合并。
3. 将现有来源迁入新接口，实现首批会议/期刊的冻结 provider manifest 和 arXiv descriptor。
4. 实现 Crossref、DBLP、Semantic Scholar、OpenAlex、PubMed、Europe PMC 与 Unpaywall 的默认/条件 provider；可选 Exa/Gemini/DeepXiv/AlphaXiv 只保留禁用默认和插件契约。
5. 实现用户 seeds 导入、正式版/preprint 关系、字段级 provenance、provider independence、双源身份校验、official venue membership 和冲突队列。
6. 实现 backward/forward citation primitives、round 状态机、硬预算和逐来源 search audit；本阶段用 deterministic fake screener 验证接口，不声称完成真实饱和。
7. 实现 entry point 注册前的 distribution/version/digest 校验、第三方最小权限子进程、allowlist、provider credential/限流/缓存和 bulk snapshot 模式。

门禁：全部命名 venue 通过 fixture 和一次受控 smoke test；多源检索可重放，required source 失败显式 incomplete，引用方向/depth/loop/cap 和 fake-screening round 契约通过，PDF 缺失不影响元数据入库；共享 Crossref 上游的两条记录不能算独立双源，primary 失败时两份 fallback 仍只能 venue_candidate、不得进入正式 venue 统计且 run 必须 incomplete。

### Phase 3：Stage 2 模型评测与上线

1. 升级 oMLX 至至少 v0.5.7。
2. 从完整 private snapshot 按冻结 seed 先抽 HIDDEN_REAL、再以剩余 family 构建 DEV/HIDDEN_HARD；选中 600 条后完成双人独立标注与仲裁，并保管 sampling provenance。
3. 实现 BGE 基线、候选 backend 和 Qwen 裁决器。
4. 校准阈值并运行隐藏集。
5. 运行 1,000/10,000 篇性能与 soak 测试。
6. 固定 winner 的 revision、量化、prompt/schema 和阈值 artifact。
7. 用冻结 seed selector 运行完整 search↔Stage 2 round 状态机，验证新增候选仍完整经过 Stage 2、饱和/预算判断和 search audit。

门禁：质量、结构化输出、吞吐、内存和稳定性硬门全部通过；真实筛选闭环的 depth、逐轮分母、经验饱和/budget_exhausted 和 resume 行为可重放。

### Phase 4：Stage 3 下载链

1. 实现 official public、PMC/Europe PMC、Unpaywall location、arXiv、authorized skill 和 manual resolvers/providers，以及 per-URL candidate -> probe -> fetch 链。
2. 集成 Luna 浏览器代理编排。
3. 实现 PDF 校验、原子落盘、状态和恢复。
4. 实现 purpose×access_basis×license×version×terms policy matrix，以及绑定 skill digest 的显式 grant、paper/collection/domain scope、时限和 attended/unattended 策略。

门禁：权限边界、下载顺序、损坏文件、skill 缺失和人工队列测试通过。

### Phase 5：Stage 4 分析

1. 实现 codex exec Luna adapter。
2. 实现 artifact-processing policy、remote_model_processing grant 和禁止未授权 bytes 进入 Codex 的 pre-dispatch gate。
3. 实现 full_pdf/abstract_only 输入、规范标签、原子 evidence units 和结构化输出。
4. 实现逐篇恢复、引用和 provenance。

门禁：固定样本的字段完整性、引用正确性、失败隔离和 resume 通过；无许可/grant 样本的全文远程调用为 0，授权样本绑定准确 artifact/model/purpose hash。

### Phase 6：Stage 4b 综述

1. 实现 ReportPlan 冻结、plan-only/approved hash 和 search/corpus audit pack。
2. 实现派生 analysis/evidence lineage 的 Sol remote_model_processing policy/grant pre-dispatch gate。
3. 实现稳定输入排序、完整 Luna 报告单包、Claims-Evidence Matrix、结构化 comparison groups/not_comparable 和 paper/claim coverage ledger。
4. 实现 `stage4b_oneshot_sol` 与独立 `one_shot_report` prompt/schema；approved run 严格一次 dispatch，0 retry、0 Sol audit、0 repair，并以数据库唯一约束阻止并发或 resume 重发。
5. 实现矛盾/不可比/evidence-level 规则及固定中文章节合同。
6. 实现全本地 deterministic normalize、renderer、report verifier 和 audit；失败保留输出并停止发布，不调用 Sol repair 或 reaudit。
7. 实现不可变报告目录、原子 latest 和增量 REPORT_DIFF。

门禁：全部入选论文的完整 Luna 报告在唯一输入包中各出现一次，全部入选论文有效覆盖率 100%，全部实质性 claim 有证据，矛盾与不可比项未被抹平，搜索限制完整披露，无 shard/抽样/截断；预算或许可/grant 门禁失败时 Sol 调用为 0，通过门禁时严格为 1，且仅本地 verifier/audit 通过后才标 complete。

### Phase 7：产品化

1. 完成 CLI、Codex skill、doctor、示例配置和迁移指南。
2. 完成离线 CI 和运维文档。
3. 完成端到端验收与发布说明。

门禁：新环境可按文档完成配置，mock 流水线通过，真实小规模 smoke run 通过。

## 14. 最终验收清单

- [x] 所有命名会议和期刊均由 descriptor/provider 架构支持。
- [x] 新增同平台 venue 不需要修改核心调度代码。
- [x] 第三方 entry point 在 import 前绑定 distribution/version/digest 校验并默认隔离执行；skill/provider 内容漂移会禁用旧信任与授权。
- [x] 用户种子、官方 venue、Crossref/DBLP、Semantic Scholar/OpenAlex、PubMed/Europe PMC、arXiv 和 OA resolver 按 role 解耦；可选发现源默认关闭且不充当权威书目来源。
- [x] compiled QueryPlan 与 ReportPlan 有显式 approval record/hash；运行时配置/provider/corpus/prompt 漂移会拒绝执行并要求新 plan。
- [x] 实际 query/filters/cursors、source errors、query hash、provider 版本与逐来源结果均可重放和审计；每个 required role 均由 manifest 中可用 provider 满足。
- [x] backward/forward citation snowballing 有方向测试、深度/预算/饱和门，并且新增候选不绕过 Stage 2。
- [x] 正式版/preprint 关系、字段 provenance、verification_status 和逐来源 citation count 可追溯；冲突不被静默覆盖。
- [x] provider independence/upstream family 可审计；共享上游不算两票，聚合源只能产生 venue_candidate，正式 venue membership 必须有官方收录证据，primary 缺失时 run 为 incomplete。
- [x] SQLite 是唯一事实源，JSONL/CSV 导入导出和旧配置迁移可用。
- [x] 多机分片不共享写 SQLite，snapshot、fencing、artifact bundle 和幂等合并测试通过。
- [x] 增量抓取、去重、合并和断点续跑幂等。
- [x] arXiv 独立候选集默认不进入最终报告。
- [ ] Stage 2 已以冻结真实样本完成 §5.4–5.5 全部 release gate：600-pair 金标、至少 1,000 次 structured replay、normal/stress 各三次 1,000-case 性能回放及 10,000-case soak，并保存模型/release、manifest、隐藏 evaluator、环境和 benchmark record hash；fixture/mock 不得勾选本项。
- [x] 正则未命中不会导致论文被静默丢弃。
- [x] 模型、revision、量化、prompt/schema 和阈值均可追溯。
- [x] PDF 下载遵循 per-URL candidate→probe→fetch 链；purpose×access_basis×license×version×terms 通过 policy matrix，能发现 URL/非空字段不被视为自动授权，失败进入人工队列。
- [x] 授权浏览器下载默认关闭，只有显式域名/scope/时限和 unattended 授权才能启用。
- [x] 不可变 grant/approval hash 是下载与数据共享的唯一授权事实源；YAML grant_defaults 不能覆盖 scope，撤销/过期/digest 漂移即时生效。
- [x] Stage 4 固定使用 gpt-5.6-luna，区分 full_pdf/abstract_only，并为数值比较输出规范化 dataset/split/metric/protocol/baseline 字段或 not_comparable。
- [x] PDF/正文进入远程 Luna、受限派生 analysis/evidence 进入 Sol 前，分别通过 remote_model_processing policy 或 artifact/lineage-hash-scoped grant；无授权内容进入对应远程调用次数为 0。
- [x] Stage 4b 固定使用 gpt-5.6-sol；approved run 将全部 Luna 报告一次打包并严格调用一次 `one_shot_report`，ReportPlan、search audit、Claims-Evidence Matrix、comparison groups 和 paper/claim coverage 完整。
- [x] ReportDocument AST 的每个实质性 block 可机检绑定 claim/citation；stable claim/comparison IDs 与 split/merge/retire lineage 支持可信增量 diff。
- [x] 中文报告按语义主题综合而非论文顺序堆叠；冲突、不可比、abstract_only、preprint 和未完成来源均显式披露。
- [x] Stage 4b 的 `one_shot_report` prompt/schema 独立冻结；预算/授权失败为 0 调用，已 dispatch 后并发、resume、超时或 uncertain outcome 不会重发；normalize/verifier/audit/publish 全部在本地确定性完成。
- [x] 报告 run 不可变，latest 原子更新，增量 diff 可定位受影响 claim/section。
- [x] Stage 3/4/4b 的 CodexExecProfile、reasoning、sandbox、网络及 call-specific prompt/output schema 均被冻结并核对实际调用。
- [x] OpenRouter/OpenCode 运行时依赖和配置已移除或迁移。
- [x] 无自动云回退、模型升级、无限重试或静默截断。
- [x] CI 完全离线且不需要真实模型、Codex 配额或订阅登录。
- [x] Codex skill 与 CLI 共用同一核心实现。

### 14.1 真实联网与授权外部门禁

以下项目按当前 source commit 验收；离线 fixture、历史 snapshot、`--dry-run` 和 `doctor` 不得替代。

- [x] 受控小预算真实 provider smoke 已通过并保存 QueryPlan、provider manifest、response、search-audit 和 rate/credit evidence；2026-08-11 的 Crossref `search → enrich → verify` 三请求完整链路证据见 `docs/smoke/crossref-full-pipeline-20260811.md`，未返回的 quota/credit header 已明确记录为 `unavailable`。其中 Stage 2 使用 fake screener，只证明 provider 链路，不得用于勾选 §14 的 Stage 2 release gate。
- [x] public OA 真实 PDF smoke 已使用默认生产传输走完 candidate → probe → fetch → PDF validation；2026-08-11 从 NeurIPS 官方 proceedings 以 `public_direct` 完成 15/15 篇公开 PDF 下载和校验，未调用 authorized browser/download skill，证据见 `docs/acceptance/neurips-2025-molecular-e2e-20260811-evidence.json`。此前 Europe PMC 的默认 SSRF fail-closed 诊断保留于 `docs/smoke/public-oa-20260811-blocker.md`，但不再阻塞本门禁。
- [x] authorized browser 真实 PDF smoke 已使用 exact grant 与用户可见的已登录授权会话成功下载允许域名和 selection scope 内的一篇论文；2026-08-11 的 Edge/Nature 运行、Luna 决策、skill audit、artifact 与 Stage 3 数据库证据见 `docs/smoke/authorized-browser-20260811.md`。
- [x] NeurIPS 2025 分子生成主题的真实 Stage 1→4b E2E 已将全部 15 篇入选论文的 Luna 全文报告一次打包，并完成一次且仅一次 `one_shot_report`；dispatch=1、Sol invocation ledger=1、reduce/audit/repair=0，本地 deterministic verifier/audit 通过。完整证据见 `docs/acceptance/neurips-2025-molecular-e2e-20260811-evidence.json`，最终中文报告见 `docs/smoke/neurips-2025-molecular-one-shot-report-20260811.md`。本次 Stage 2 明确使用 test-only selector，只验证 E2E 数据流，不满足也不影响上方 Stage 2 生产 release gate 的未勾选状态。
- [x] 全部 20 个 venue descriptor 已各选一个主题完成全当前运行的受控 Stage 1→4b 功能矩阵，不再导入历史 NeurIPS 行：每个 venue 均下载并校验一篇真实公开 PDF，以 `full_pdf` 调用一次 Luna，最终绑定报告再调用一次且仅一次 one-shot Sol；严格汇总为 20/20，全部使用 `stage4b-one-shot-v3/current_v3`、canonical `claim_id` audit ordering，并通过 PDF→Luna→Sol lineage、CAS/manifest/hash 和当前 deterministic verifier，最终绑定报告的 reduce/audit/repair/retry 调用均为 0。为完整披露成本，本轮连同保留的失败/预检尝试共调用 21 次 Luna、22 次 Sol：ICML 首次预检在 Sol 前停止；Cell 与 Nature Chemistry 的旧错误 plan 各消耗一次诊断 Sol，修复 planner 后复用原 Luna 并以新 `report-v2` 各调用一次 Sol；IJCAI、Nature Machine Intelligence 和 Nature Biotechnology 仅做确定性本地恢复，未追加模型调用。该矩阵 Stage 1 为 approved one-record provider-shaped snapshot replay、Stage 2 为 `TEST_ONLY`，因此不冒充逐 provider 的 live transport、oMLX 筛选或 Stage 2 production release 证据；范围、主题、论文及哈希见 `docs/acceptance/venue-e2e-matrix-20260812.md` 与 `docs/acceptance/venue-e2e-matrix-20260812-evidence.json`，2026-08-11 的历史复用矩阵仍保留供审计。

## 15. 实施停止条件

遇到以下情况必须停止相关阶段并明确报告，不能自行扩大权限或改变产品语义：

- 需要绕过站点访问控制才能继续。
- 用户提供的 skill 或插件要求超出预期的敏感权限。
- 金标质量门与吞吐门无法同时满足，需要改变模型或召回策略。
- Codex 模型 slug、CLI 能力或授权状态与规格不一致。
- 数据迁移存在不可逆丢失风险。
- 需要新增云服务、付费 API 或外部基础设施。

其余单条论文、单个来源或单次模型请求失败均应隔离、记录并继续处理其他工作项。

## 16. 实施参考基线

以下资料在 2026-08-09 制定本规格时用于确认模型与运行时能力。真正下载或升级前仍需重新核对上游 revision、许可证和兼容性：

### 16.1 ARIS 检索与综合设计参考

- [Auto-claude-code-research-in-sleep / ARIS](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep)
- [research-lit：多来源文献发现与去重](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep/blob/main/skills/skills-codex/research-lit/SKILL.md)
- [comm-lit-review：来源分层与按技术轴综合](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep/blob/main/skills/skills-codex/comm-lit-review/SKILL.md)
- [paper-plan：先建立 claim/evidence 计划再写作](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep/blob/main/skills/skills-codex/paper-plan/SKILL.md)
- [fan-out pattern：只读分片、机械去重、单写者合并](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep/blob/main/skills/skills-codex/shared-references/fan-out-pattern.md)
- [citation discipline：书目与 claim-context 双重校验](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep/blob/main/skills/skills-codex/shared-references/citation-discipline.md)
- [research-wiki：持久论文节点、关系和 query pack](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep/blob/main/skills/skills-codex/research-wiki/SKILL.md)

### 16.2 学术图、书目与合法 OA 来源

- [Semantic Scholar Academic Graph API](https://www.semanticscholar.org/product/api)
- [OpenAlex API](https://developers.openalex.org/api-reference/introduction) 与 [CC0 snapshot](https://developers.openalex.org/download/overview)
- [Crossref REST API](https://www.crossref.org/documentation/retrieve-metadata/rest-api/)
- [DBLP Search API](https://dblp.org/faq/How%2Bto%2Buse%2Bthe%2Bdblp%2Bsearch%2BAPI.html) 与 [月度 snapshot](https://dblp.org/xml/release/)
- [NCBI E-utilities](https://www.ncbi.nlm.nih.gov/books/NBK25497/)
- [Europe PMC REST API](https://europepmc.org/RestfulWebService)
- [Unpaywall API](https://unpaywall.org/api) 与 [字段/许可语义](https://unpaywall.org/data-format)
- [arXiv API](https://info.arxiv.org/help/api/index.html) 与 [API terms/rate limits](https://info.arxiv.org/help/api/tou.html)
- [Google Scholar 使用说明](https://scholar.google.com/intl/us/scholar/help.html)

### 16.3 首批 venue 官方入口

- [NeurIPS Proceedings](https://proceedings.neurips.cc/) · [PMLR](https://proceedings.mlr.press/) · [OpenReview venues](https://openreview.net/venues)
- [AAAI OJS archive](https://ojs.aaai.org/index.php/AAAI/issue/archive) · [ACL Anthology data/API](https://aclanthology.org/info/development/) · [IJCAI proceedings](https://www.ijcai.org/all_proceedings)
- [CVF Open Access](https://openaccess.thecvf.com/) · [IEEE Xplore API](https://developer.ieee.org/)
- [DAC publication instructions](https://dac.com/2026/research-manuscript-submissions) · [ICCAD publication instructions](https://iccad.com/2026/final-author-instructions) · [IEEE CEDA TCAD](https://ieee-ceda.org/publications/tcad)
- [Springer Nature Metadata API](https://dev.springernature.com/docs/api-endpoints/metadata-api/) · [Elsevier ScienceDirect API](https://dev.elsevier.com/sd_api_spec.html) · [Science eTOC](https://www.science.org/action/showFeed?type=etoc&feed=rss&jc=science)

### 16.4 模型与运行时

- [oMLX README：模型类型、continuous batching 与 API](https://github.com/jundot/omlx/blob/main/README.md)
- [oMLX v0.5.7 release](https://github.com/jundot/omlx/releases/tag/v0.5.7)
- [Qwen3.5-9B model card](https://huggingface.co/Qwen/Qwen3.5-9B)
- [Qwen3.5-9B MLX 8-bit](https://huggingface.co/mlx-community/Qwen3.5-9B-8bit)
- [Qwen3 Reranker series](https://huggingface.co/Qwen/Qwen3-Reranker-8B)
- [BGE reranker v2 m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- [BGE reranker v2 m3 MLX BF16 候选](https://huggingface.co/soichisumi/bge-reranker-v2-m3-mlx)
- [Querit-4B](https://huggingface.co/Querit/Querit-4B)
- [Codex recommended models](https://learn.chatgpt.com/docs/models#recommended-models)
- [Codex non-interactive mode](https://learn.chatgpt.com/docs/non-interactive-mode.md)
