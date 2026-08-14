# Minimal Stage 1 architecture

本文是 Stage 1-only 新实现的规范性内部架构。外部行为由 `docs/stage1-contract.md` 定义；内部实现必须同时遵守 contract 和 architecture。本文定义目标架构，不声明当前旧代码已经符合，也不得被用于保留 Stage 2、Stage 3、Stage 4 或旧兼容层。

## 1. Architecture goals

目标是：单次只处理一个 venue-year；由 authoritative source 决定 membership；enrichment 不得改变 membership；core 与具体来源彻底解耦；新增已有来源族的 venue 不修改 Python core；新增来源族只增加独立模块、venue spec 和测试；不下载完整 PDF、不运行 Stage 2–4、不保留旧 API 兼容层；新实现可独立测试，不依赖旧 package。

## 2. Design principles

架构坚持 small core、source-family adapters、explicit configuration、dependency injection、pure normalization、deterministic output、boundary validation、no speculative abstractions、no global mutable registry、no provider-specific branches in core、no raw provider metadata in public output。

必须防御外部 HTTP 失败、非法外部响应、cursor cycle、total mismatch、parser loss、duplicate source ID、enrichment identity drift、非 PDF 响应和输出文件发布失败。不得为可信 adapter import、内部 dataclass 互转、函数间重复 schema validation、第三方不可信插件执行、分布式任务、多阶段授权或 artifact attestation 引入大型框架。

## 3. Target package layout

最终切换后的目标结构为：

```text
src/paper_agent/
├── __init__.py
├── __main__.py
├── cli.py
├── errors.py
├── models.py
├── normalize.py
├── http.py
├── catalog.py
├── loading.py
├── collector.py
├── access.py
├── output.py
├── adapters/
│   ├── __init__.py
│   ├── base.py
│   └── <source_family>.py
├── enrichers/
│   ├── __init__.py
│   ├── base.py
│   └── <metadata_source>.py
└── venue_specs/
    ├── __init__.py
    └── <venue_id>.toml

tests/
└── stage1/
    ├── unit/
    ├── contract/
    ├── integration/
    ├── live/
    └── fixtures/
```

不创建 `services/`、`managers/`、`factories/`、`runtime/` 等通用层；只有出现两个以上真实且不同的职责时才允许新增模块。不创建 `legacy/` 保存旧代码；Git 历史和 backup tag 已足够恢复旧实现。

## 4. Migration isolation

最终切换前，新实现临时位于：

```text
src/paper_agent_next/
tests/stage1_next/
```

其内部布局与最终 `paper_agent` 基本相同。`paper_agent_next` 不得 import 旧 `paper_agent` 的任何模块；新测试不得依赖旧 `domain.py`、`providers/api.py`、`providers/builtin.py`、`provider_runtime.py`、`stage1.py`、`stage1_hydration.py` 或 `venue_transport.py`。旧 console entry point 在新 CLI 通过切换门禁前保持不变；不得用 wrapper、compatibility adapter 或 re-export 复用旧架构。

可以人工移植已验证的 parser 小函数、DOI 清理逻辑、cursor 规则和 fixture，但移植后必须符合新接口并由新测试覆盖；不得整文件复制旧 monolith。新 package 模块名不得包含 `legacy`、`compat`、`v2`、`manager` 或 `service_factory`。

最终切换时删除旧 `src/paper_agent`，将 `paper_agent_next` 重命名为 `paper_agent`，将 `tests/stage1_next` 重命名或整理为 `tests/stage1`，更新 console entry point，不保留旧 API wrapper。

## 5. Module responsibilities

每个模块只承担下表职责，具体类名由实现任务定义：

| Module | Sole responsibility |
|---|---|
| `errors.py` | 小型 typed exceptions；不含业务流程或大型错误层次。异常控制在 input/catalog、provider/collection、contract/validation、publication 四类附近。 |
| `models.py` | Stage 1 数据值；使用 frozen/slotted dataclass 或等价值对象；不网络、不文件、不 import adapter/enricher，不含 Stage 2–4、ORM 或通用序列化框架。 |
| `normalize.py` | 纯 DOI、文本、title、author、abstract、URL normalization 和 stable comparison key；不网络、不配置、不文件、不依赖 venue。 |
| `http.py` | 单一同步 HTTP client、timeout、有限重试、User-Agent/contact、简单 host 限速；支持 fixture/mock transport 注入；不含 parser、venue URL 或 PDF policy；adapter/enricher 不得自建 session。 |
| `catalog.py` | 用标准库 `tomllib` 读取 venue specs，验证通用字段，解析适用年份和 year override；不 import adapter、不网络、不含 manifest、terms state 或 artifact hash。 |
| `loading.py` | 用 `importlib` 加载明确配置的可信仓库内模块；不环境扫描、entry points、第三方 plugin system、sandbox、signature、attestation 或中央 registry。 |
| `adapters/base.py` | CorpusAdapter 边界和小型共享类型；决定 authoritative membership、source-native pagination、raw item accounting；不写文件、不验证 PDF、不调用其他 adapter，只用注入 HTTP client。 |
| `adapters/<source_family>.py` | 一个来源族一个模块，同族服务多个 venue；只放 source-local URL、分页、解析和分类规则；不得改 collector 接入。复杂 parser 可放同族小子包。 |
| `enrichers/base.py` | MetadataEnricher 边界和 patch 语义；patch 以冻结 source identity 为 key；不做 membership discovery。 |
| `enrichers/<metadata_source>.py` | 只补已有论文的字段或 PDF candidate；不得增删替换论文或覆盖 primary 非空强字段；只用注入 HTTP client。 |
| `access.py` | 验证匿名 direct PDF candidates，选择稳定可复用 URL，设置 direct_pdf/doi_only；不 publisher-specific 拼 URL、不完整下载、不绕过访问控制。 |
| `collector.py` | 固定编排；不解析来源、不按 provider 分支、不构造来源 URL、不写正式文件；返回符合 contract 的内存结果。 |
| `output.py` | 将内存结果转为三个 contract artifacts，确定性 JSON/JSONL、原子发布、run.json 最后发布；不网络、不 membership/enrichment 规则。 |
| `cli.py` | 解析参数，调用 catalog/loading/http/collector/output，映射退出码，最外层呈现友好错误；不 source-specific、不 parser、不 dedup。 |
| `__main__.py` | 只调用 CLI entry point，不含其他逻辑。 |

## 6. Dependency direction

下图箭头表示“可以依赖”，是概念层次，不要求每层依赖所有上层模块：

```text
errors, models, normalize
        ↓
http, catalog
        ↓
adapters/base, enrichers/base
        ↓
concrete adapters, concrete enrichers, access
        ↓
loading, collector
        ↓
output
        ↓
cli, __main__
```

强制规则：`errors.py`、`models.py`、`normalize.py` 不依赖高层；`http.py` 不依赖 adapter、enricher、collector、output、CLI；concrete adapter/enricher 不依赖 collector、output、CLI；`access.py` 不 import 具体 adapter/enricher；`collector.py` 只依赖协议和已加载实例，不 import `adapters.<specific_source>` 或 `enrichers.<specific_source>`；`output.py` 不依赖具体来源；任何模块不得 import 旧 `paper_agent`；禁止循环 import，禁止 import 时网络请求或全局 mutable state。`loading.py` 可以用 importlib 加载配置路径，但不构成 collector 的静态实现依赖。

## 7. Extension boundaries

只保留两个可扩展角色：`CorpusAdapter` 和 `MetadataEnricher`。不得增加 SearchProvider、CitationProvider、LibraryProvider、MetadataVerifier、DownloadProvider、WorkflowProvider、ReportProvider 或 plugin distribution role。

### CorpusAdapter

概念伪签名：

```text
collect(venue_spec, year, http_client) -> CollectionResult
```

必须完整枚举 authoritative membership，返回 stable source IDs、初始 metadata、PDF candidates（不是 verified `pdf_url`）、pagination 和 census，并核算每个 raw item 的 included/excluded/duplicate/parse-reject；不执行 enrichment fallback chain、不写文件。

### MetadataEnricher

概念伪签名：

```text
enrich(frozen_papers, http_client) -> patches_by_source_identity
```

只返回 patch，key 必须属于冻结 membership；未知 identity 必须拒绝；不得清空已有值；可以增加 PDF candidate；不验证最终 direct PDF、不写文件。精确 Python Protocol 和 dataclass 字段由后续任务定义，后续实现不得突破这些边界。

## 8. Venue catalog

新实现使用 TOML，不使用旧 YAML manifest。每个 venue 一个文件 `venue_specs/<venue_id>.toml`，由 Python 3.11+ 标准库 `tomllib` 读取：

```toml
schema_version = 1
id = "icml"
name = "International Conference on Machine Learning"
venue_type = "conference"
adapter = "adapters.pmlr:PmlrAdapter"
enrichers = ["enrichers.openalex:OpenAlexEnricher"]
start_year = 1980

[source]
series = "ICML"

[year_overrides."2024"]
volume = "v235"
```

示例只表示结构，不声明数值事实正确。通用字段为 `schema_version`、`id`、`name`、`venue_type`、`adapter`、`enrichers`、`start_year`、可选 `end_year`、`held_years`、`source` 和 `year_overrides`。

规则：`id` 与文件名一致；`venue_type` 只允许 conference/journal；`adapter` 是受信任相对实现路径；`enrichers` 按执行顺序排列；迁移期路径相对 `paper_agent_next`，切换后相对 `paper_agent`；adapter 路径只以 `adapters.` 开头，enricher 只以 `enrichers.` 开头；路径为 `module:attribute`，不允许绝对 Python package、distribution entry point、文件系统路径或 URL。`source` 是 adapter-specific 只读 mapping；`year_overrides.<year>` 只覆盖 source 参数，不执行 Python 函数或包含 parser 代码；`held_years` 支持不规则举办年份；catalog 必须在 provider 前决定 not_applicable；第一版不支持 alias；旧 `venues/*.yaml` 不得读取。

config 不得包含 field_enrichment 函数名、fallback graph、fixture hash、implementation hash、terms acceptance state 或 arbitrary Python expression。

## 9. Trusted implementation loading

采用 explicit configured import path；不使用中央 registry、package scanning、decorator registration 或 Python distribution plugin discovery。流程是：catalog 读取字符串路径 → `loading.py` 验证允许前缀 → importlib 导入 module → 读取 attribute → 创建或取得实例 → 检查符合对应 protocol → 注入 collector。

不得维护 `BUILTIN_CLASSES`、`ADAPTERS`/`PROVIDERS` dict、handler decorator registry、manifest-driven provider factory 或 plugin allowlist。

## 10. Collection flow

只允许以下固定顺序：

```text
parse CLI input
→ load venue spec
→ decide applicable/not_applicable
→ load adapter and enrichers
→ create one shared HTTP client
→ authoritative membership collection
→ normalize raw fields
→ deterministic initial deduplication
→ freeze membership identities
→ run configured enrichers in order
→ reject unknown enrichment identities
→ resolve and verify PDF candidates
→ validate complete/incomplete papers
→ construct run counts and diagnostics
→ publish papers.jsonl
→ publish issues.jsonl
→ publish run.json last
→ return contract exit code
```

collector 不允许 adapter 启动另一个 workflow；enricher 顺序来自 venue spec；一个 enricher 失败后是否继续由后续错误策略任务定义，但不得清空数据或 membership；access verification 必须在最终 record validation 前；output publication 不得在 membership/enrichment 仍被修改时发生。

## 11. Error boundaries

外部边界必须严格验证；内部已验证 dataclass 不重复大规模 schema validation；只捕获可解释可处理的异常，低层禁止裸 `except Exception`。CLI 最外层可捕获未知异常以输出简短错误并返回 exit code 4，但不得伪造 run.json。adapter 必须将 provider、parser、pagination 问题转为有限 typed error/diagnostic；enrichment 单点失败不得删除 membership；artifact publication error 不得成为 ordinary warning；公开错误不得含 credential、token、cookie 或完整 HTML 响应。

不实现 circuit-breaker framework、distributed retry state、signed receipt 或 response hash ledger。

## 12. Test architecture

迁移期新测试位于 `tests/stage1_next/`，最终位于 `tests/stage1/`：

```text
tests/stage1/
├── unit/
├── contract/
├── integration/
├── live/
└── fixtures/
```

分层规则：unit 覆盖 models、normalization、dedup、access classification、serialization、catalog validation、import-path validation；contract 用固定 fixture 验证 source IDs、pagination terminal、raw item accounting、excluded/parse reject、adapter 不写 verified `pdf_url`、enricher 不改 membership；integration 用 fake adapter/enricher/fixture HTTP client 覆盖 complete、partial、failed、not_applicable、atomic publication、exit mapping；live 默认不运行，仅由明确环境变量启用，每个 source family 选少量稳定历史年份，不把当前年度精确数量作为普通 CI 断言，live failure 不改变离线 contract 结果。

普通 pytest 不访问网络；新测试不 import 旧 `paper_agent`；fixture 保存 source-native 响应而非预加工最终 record；新增 source family 必须新增 contract fixture，新增同族 venue 至少新增 venue-spec validation case。必须有 import-boundary test 阻止 `paper_agent_next` import `paper_agent`、collector import concrete source、adapter import CLI/output/collector。旧 67 个基线测试在切换前继续作为旧代码回归基线，新代码完成度由新测试判断。

## 13. Complexity budget

建议上限：`cli.py` 200 行、`collector.py` 400 行、`models.py` 350 行、`http.py` 300 行、`access.py` 350 行、`catalog.py`+`loading.py` 合计 350 行、单个 adapter/enricher 450 行；任何运行时 Python 文件超过 550 行必须拆分或在 review 给出具体理由；core runtime（不含 concrete adapters/enrichers）目标约 2,000 行。不为行数机械压缩可读性，不用一行多语句规避门禁。

依赖优先标准库；HTTP 只允许一个共享客户端库；HTML parsing 最多一个成熟 parser。新增 runtime dependency 必须单独 review；不引入 Scrapy、pandas、SQLAlchemy、Pydantic、workflow engine、dependency injection framework、plugin framework 或 retry framework，除非后续有独立明确获准的决策。当前旧依赖不在本任务删除，pyproject 精简属于后续切换任务。

## 14. Extension recipes

已有 source family 的新 venue 只需新增 venue TOML、fixture（或复用已有 fixture）和 venue-spec/contract test；不得改 collector、models、access、output、cli、loading 或中央 registry（中央 registry 不存在）。

新 source family 只需新增 `adapters/<source_family>.py` 或同名小子包、venue TOML、source-native fixture、adapter contract test，并复用或新增独立 enricher；不得在 core 加入 `if source_name == ...`。新 metadata source 只需新增 `enrichers/<source>.py`、在 venue spec 加 import path、fixture 和 contract test；不得修改中央 hydration dispatcher（它不存在）。

## 15. Cutover gates

切换默认 package/CLI 前至少满足：

1. 新 core unit tests 通过；
2. complete/partial/failed/not_applicable integration tests 通过；
3. output contract tests 和 import-boundary tests 通过；
4. 至少三个不同来源类型通过离线端到端测试：官方会议 proceedings（如 PMLR）、conference API（如 OpenReview）、journal registry/API（如 Crossref journal）；
5. PDF candidate 验证和新 CLI fixture smoke 通过；
6. 新 package 不 import 旧 package；
7. 至少一次获准 live smoke 已运行并记录，但不替代 fixture contract tests；
8. 已列出最终保留和删除的 venue 支持范围。

最终切换必须切换 console entry point，删除旧运行时代码、旧非 Stage 1 测试和旧 provider/plugin/authorization/workflow/report 框架，重命名 `paper_agent_next`，精简依赖和 package data，不保留兼容 wrapper，不把旧代码移入 `legacy/`。

## 16. Forbidden patterns

禁止 core 中出现 `if provider == ...`、`if venue_id == ...`、中央 adapter/enricher 名称列表、decorator import-time registration、package auto-scan、Python distribution plugin entry point、plugin sandbox、implementation/fixture SHA 运行时条件、provider manifest role/capability graph、fallback graph YAML、raw metadata catch-all、中央 hydration 类、中央 transport 文件、adapter 自建 HTTP session、adapter 写正式输出、enricher 改 membership、core 猜测 publisher PDF URL、title-only fuzzy merge、LLM 生成摘要、学校账号/cookie/SSO/CAPTCHA 绕过、Stage 2–4 compatibility layer、GenericManager/CoordinatorFactory/ProviderRuntime 等无必要抽象，以及将旧代码复制到 `legacy/`。

## 17. Architecture acceptance

后续 review 清单：新 core 不知道具体 source 名称；venue spec 明确选择实现；adapter/enricher 只用注入 HTTP client；membership 冻结后不可改变；PDF candidate 与 verified `pdf_url` 分离；output 与 collection 分离；普通测试完全离线；新 venue/source 不改 core；无中央 registry、旧 package import 或 Stage 2–4 概念；文件规模符合预算；外部行为符合 `stage1-contract.md`。

## 18. Scope limits

本文不授权新增 Python/TOML/测试文件，不修改 `src/**`、`tests/**`、`pyproject.toml` 或旧 venue YAML，不实现 Protocol、dataclass、loader、adapter、enricher，不删除旧代码、不切换 CLI、不操作 stash。实现任务开始前，tracked diff 只能包含本架构文档和 contract 的两处指定修正。
