# Stage 1 metadata collection contract

本文是规范性外部产品契约，后续实现和测试必须遵守本文。本文已经定义 complete paper JSON representation、issue JSON representation、run JSON representation、output artifacts、run status、CLI 参数和退出码；仍不定义 Python 类、正式 JSON Schema 文件、adapter/enricher API 或内部实现细节。JSON representation 已定义不等于正式 JSON Schema 文件已定义。不得声明当前代码已经实现本文。

## 1. Goal

唯一目标是：给定一个规范化 `venue_id` 和一个年份，完整枚举该目标期刊或会议在该年份范围内的正式研究论文，并为每篇论文取得：

- `title`
- `authors`
- `abstract`
- 可直接匿名访问的 PDF URL，或者 DOI

`title`、`authors`、`abstract` 是每篇纳入论文的强制字段。若没有可直接访问的 PDF，则 DOI 是强制字段。有直接 PDF 但没有 DOI 的会议论文可以合法完成。不得生成、推测或由 LLM 编写缺失摘要。

## 2. Unit of work

第一版的单个工作单元必须是：

- 一个 `venue_id`；
- 一个四位数 `year`；
- 一次运行只处理一个 venue-year。

多 venue 或多年批量调度不属于第一版内核，可以由未来外层工具实现。

venue-year 的定义如下：

- `venue_id` 必须由本地 venue catalog 解析；
- `year` 必须是该 venue 配置所声明的出版或会议年份；
- 不适用年份必须被识别为 `not_applicable`，不得返回虚假的空成功结果。

## 3. Terminology

### Authoritative membership source

能够确定该 venue-year 正式论文集合的主要来源，例如：

- 官方 proceedings；
- 官方 conference API；
- 官方期刊或出版社 API；
- 明确配置并验收过的 registry/TOC fallback。

通用学术搜索结果不能默认被视为完整 membership。

### Included paper

属于目标 venue-year 范围、被正式发表或接受的研究论文。

### Excluded non-paper item

不是研究论文的项目，例如：

- editorial
- correction
- erratum
- withdrawal
- preface
- table of contents
- proceedings front matter
- reviewer acknowledgement
- news
- podcast
- supplementary-only item

排除必须有明确类型或来源证据，不能因为缺摘要就把一条记录自动排除。

### Direct PDF

无需以下任何内容即可通过普通匿名 HTTP 客户端直接取得 PDF 内容的 URL：

- 学校账号；
- 出版社个人账号；
- 浏览器 cookie；
- SSO；
- CAPTCHA；
- 付费墙绕过；
- 人工交互。

仅仅以 `.pdf` 结尾、由字符串拼接生成、或重定向到登录页的 URL 不是 direct PDF。

### Metadata enrichment

只能补齐已经属于 primary membership 集合的论文字段，不得增加、删除或替换 membership。

## 4. Conference scope

第一版默认规则如下：

- 包含配置中指定的主会议正式 accepted papers；
- oral、spotlight、poster 若都属于主会议正式论文，必须包含；
- rejected、withdrawn、desk-rejected submission 不包含；
- workshop、tutorial、demo、doctoral consortium、competition 等默认不包含；
- 若将来需要 workshop，必须作为独立 venue 配置接入；
- proceedings 中的 front matter 等非论文条目需要显式分类并统计。

不得把所有 conference 的特殊规则写入核心契约。具体 invitation、track、volume 和 route 属于 venue 配置或 adapter。

## 5. Journal scope

第一版默认规则如下：

- 包含目标期刊在请求年份内的正式 research articles；
- 年份判定必须由 venue 配置选择稳定的日期口径；
- correction、editorial、news、front matter 等非研究条目不包含；
- online-first 与 issue year 冲突时，必须使用配置声明的单一口径，不能在运行中临时猜测；
- adapter 必须统计原始条目、排除条目和纳入论文；
- 不得因为 Crossref、OpenAlex 或出版社之间数量不同而静默选择数量较大的集合。

## 6. Completeness

### membership_complete

只有同时满足以下条件时，`membership_complete` 才能为 true：

1. authoritative membership source 正常访问并成功解析；
2. 分页或游标到达明确终点；
3. 检测到的 cursor cycle、请求中断或 parser failure 均必须使其为 false；
4. source 提供 expected total 时，数量必须核对一致；
5. 每个原始条目都必须归入 included paper、excluded non-paper item、duplicate occurrence 或 rejected parse item；
6. 不得存在未解释的静默丢失。

**规则 A：rejected parse item 必须阻止 complete。**

- `rejected parse item` 表示 adapter 无法可靠判断或解析该原始条目；
- `rejected parse item` 可以被保留在运行统计和诊断输出中；
- 任何尚未解决的 `rejected parse item` 都必须使 `membership_complete` 为 false；
- 只有该条目后来被可靠地重新分类为 included paper、excluded non-paper item 或 duplicate occurrence，才算解决；
- 不得仅因为 `rejected parse item` 已被计数，就把它视为 membership 已闭合。

**规则 B：适用 venue-year 的空结果需要明确证明。**

- 对于 `not_applicable` 年份，继续使用独立的 `not_applicable` 状态；
- 对于被 catalog 判定为适用的 venue-year，如果 included paper 数量为零，只有 authoritative source 明确证明该年份的正式论文总数确实为零时，`membership_complete` 才可以为 true；
- 明确证明可以是 source 声明的 expected total = 0，或语义等价的 authoritative empty census；
- 普通空 HTTP 响应、空搜索结果、空 HTML selector 结果、缺少 total 的空列表，都不得被视为零论文证明；
- 无法证明时 `membership_complete` 必须为 false。

### metadata_complete

只有以下条件对每一篇 included paper 都成立时，`metadata_complete` 才能为 true：

1. normalized title 非空；
2. 至少有一个 normalized author；
3. normalized abstract 非空；
4. 存在经过验证的 direct PDF，或存在 normalized DOI。

`title`、`authors`、`abstract` 不允许使用 `legitimately_absent` 绕过。非研究条目应在 membership 分类时排除，而不是作为缺摘要论文留在最终集合。enrichment 服务失败而留下必填字段缺失时，`metadata_complete` 必须为 false。

### complete

`complete = membership_complete AND metadata_complete`。

不得将 membership complete 单独对用户报告为完整成功。

## 7. Complete paper record

本章节定义的是已经通过单篇 metadata completeness 验证、可以写入 `papers.jsonl` 的记录。一条 complete paper record 代表一篇 included paper；partial run 中也可以存在 complete paper records。unresolved、parse reject 和 excluded item 不使用该记录格式。complete paper record 不允许使用 `unknown`、`unresolved`、`legitimately_absent` 等状态规避强制字段。schema version 1 中所有顶层字段必须始终存在；可空字段无值时必须显式输出 JSON `null`，不能省略。

schema version 1 的顶层字段恰好为：`schema_version`、`venue_id`、`venue_name`、`venue_type`、`year`、`source_name`、`source_id`、`title`、`authors`、`abstract`、`doi`、`landing_url`、`pdf_url`、`access_status`、`field_sources`。不得增加其他顶层字段。

- `schema_version`：JSON integer，第一版固定为 `1`。
- `venue_id`：非空 JSON string，必须是 venue catalog 中的 canonical ID，不得输出未规范化别名。
- `venue_name`：来自 venue catalog 的非空 JSON string。
- `venue_type`：JSON string，只允许 `conference` 或 `journal`。
- `year`：请求的四位数 venue-year 的 JSON integer，不得被 enrichment 来源中的其他年份静默替换。
- `source_name`：表示 authoritative membership source 的稳定规范化标识的非空 JSON string，例如 `pmlr`、`openreview`、`acl_anthology`、`crossref_journal`；不得使用 Python 类路径、随机运行 ID、临时 URL 或响应哈希。
- `source_id`：source 提供或 adapter 确定性生成的非空稳定 JSON string。core 必须将其视为 opaque value，不得解析或依赖内部格式；不得使用随机 UUID；在同一个 `venue_id`、`year`、`source_name` 范围内必须唯一。
- `title`：非空 normalized JSON string，不得为 null。
- `authors`：见下文，必须是非空字符串数组，不得为 null。
- `abstract`：非空 normalized plain-text JSON string，不得为 null。
- `doi`：normalized bare DOI 或 JSON null，规则见下文。
- `landing_url`：官方或 authoritative source 落地页的绝对 HTTP/HTTPS URL 或 JSON null。
- `pdf_url`：经过验证的 direct PDF 绝对 HTTP/HTTPS URL 或 JSON null。
- `access_status`：只允许 `direct_pdf` 或 `doi_only`。
- `field_sources`：固定结构的 JSON object，规则见下文。

`authors` 第一版必须是 JSON array of strings。数组至少一项；每项必须是非空 normalized author name；必须保留来源作者顺序，不得按字母排序，也不得因为两个作者姓名字符串相同而自动去重。第一版不输出 affiliation、email、ORCID 或 author ID。

`abstract` 必须是 normalized plain-text abstract，不得由 LLM、PDF 正文自动总结、标题扩写、关键词拼接或其他生成式补写产生。

`doi` 非 null 时必须是小写 normalized bare DOI，不得包含 `https://doi.org/`、`http://doi.org/`、`http://dx.doi.org/`、`doi:` 或前后空白。有 verified direct PDF 时 DOI 可以为 null；没有 verified direct PDF 时 DOI 不得为 null。本任务不定义完整 DOI 正则表达式或 normalization 代码。

`landing_url` 非 null 时必须是论文官方或 authoritative source 的绝对 HTTP/HTTPS 落地页；它不等于 direct PDF，不得被写入 `pdf_url` 作为替代。`pdf_url` 是调用者未来应再次请求的稳定 URL；系统必须已用普通匿名 HTTP 客户端、允许正常跟随重定向并取得 PDF 内容验证它。它可以是稳定的重定向入口，不要求等于最终 response URL；应优先保存稳定、canonical、可重复使用的候选 URL。不得保存短期签名 CDN URL、认证 token、session credential 或预期很快失效的最终重定向 URL；未验证候选、登录页、SSO 页面、付费墙页面、DOI landing page、CAPTCHA 页面和 HTML 错误页都不得写入。本任务不定义具体 PDF probe 算法。

如果匿名请求当前能取得 PDF，但唯一可保存地址是短期临时 URL，该候选不得作为 durable direct PDF 输出；没有稳定 direct PDF 时，有 DOI 则使用 `doi_only`，没有 DOI 则该论文 incomplete。`field_sources.pdf_url` 记录稳定 URL 候选的来源，不是通用 probe 实现名称。

`access_status` 的真值表为：

| access_status | pdf_url | doi |
|---|---|---|
| `direct_pdf` | 必须为 verified URL | 可以为 normalized DOI 或 null |
| `doi_only` | 必须为 null | 必须为 normalized DOI |

不变量：`pdf_url` 非 null 时 `access_status` 必须为 `direct_pdf`；`direct_pdf` 下 `pdf_url` 不得为 null；`doi_only` 下 `pdf_url` 必须为 null 且 `doi` 不得为 null；两者同时为 null 的记录不是 complete paper record。同时拥有 verified PDF 和 DOI 时必须使用 `direct_pdf` 且保留 DOI，不能为了输出 `doi_only` 丢弃已验证 PDF。`access_status` 不允许 `unresolved`、`unknown`、`auth_required`、`manual_required` 或其他值。

`field_sources` 必须始终包含且只包含以下 key：`title`、`authors`、`abstract`、`doi`、`landing_url`、`pdf_url`。每个 value 是 JSON string 或 JSON null；非 null 时必须是稳定规范化来源标识，例如 `pmlr`、`openreview`、`crossref`、`openalex`、`semantic_scholar`、`unpaywall`，不得使用 Python 类路径、响应哈希、临时文件路径或随机运行 ID。最终字段为 null 时对应 value 必须为 null，最终字段非 null 时对应 value 必须非 null；membership source 提供的字段可以使用与 `source_name` 相同的值。schema version 1 禁止 `raw_metadata`、`source_metadata`、`metadata`、`extra`、`arbitrary attributes` 或任意 catch-all JSON object；adapter 原始响应不得原样写入 complete paper record。将来增加标准化字段必须显式更新契约版本。

### Complete paper record examples

以下两个语法有效的 JSON 示例使用明显的虚构数据，且不含额外字段。

**direct_pdf conference paper**

```json
{"schema_version":1,"venue_id":"example-conf","venue_name":"Example Conference","venue_type":"conference","year":2024,"source_name":"example_proceedings","source_id":"paper-001","title":"An Example Paper","authors":["Alice Example","Bob Example"],"abstract":"This is an example abstract.","doi":null,"landing_url":"https://example.org/papers/001","pdf_url":"https://example.org/papers/001.pdf","access_status":"direct_pdf","field_sources":{"title":"example_proceedings","authors":"example_proceedings","abstract":"example_proceedings","doi":null,"landing_url":"example_proceedings","pdf_url":"example_proceedings"}}
```

**doi_only journal paper**

```json
{"schema_version":1,"venue_id":"example-journal","venue_name":"Example Journal","venue_type":"journal","year":2024,"source_name":"example_journal","source_id":"article-001","title":"Another Example Paper","authors":["Casey Example"],"abstract":"This abstract was supplied by an enrichment source.","doi":"10.1234/example.2024.001","landing_url":"https://example.org/articles/001","pdf_url":null,"access_status":"doi_only","field_sources":{"title":"example_journal","authors":"example_journal","abstract":"crossref","doi":"crossref","landing_url":"example_journal","pdf_url":null}}
```

## 8. Field normalization

schema version 1 使用保守规范化：必须清理格式噪声，但不得改变学术内容。

### 通用文本

对 title、author name、abstract，必须解码 HTML entities、移除 HTML/XML 标签、移除首尾空白、规范化连续普通空白、移除不可见控制字符，并使用 Unicode NFC normalization。不得 ASCII transliteration、自动翻译、自动改写大小写或语言模型纠错；不得删除有语义的数学符号、希腊字母或标点。

### title normalization

必须保留来源标题的语言、大小写和有语义标点，折叠换行和重复空白，解码实体并移除标签；不得自动改成 title case，也不得仅为去重删除全部标点。输出 title 可以与用于比较的内部 comparison key 不同，comparison key 不属于输出 schema。

### author normalization

每个作者名独立规范化并保留作者顺序；移除空作者项，规范化后没有作者即记录不完整。不得自动互转 `Family, Given` 与 `Given Family`，不得猜测 given name/family name，不得根据姓名字符串合并作者；来源将 consortium 或 group author 作为作者时，必须保留为一个字符串。

### abstract normalization

输出必须是 plain text；解码 HTML/XML/JATS 标签，去除明显标签噪声，可以保留有意义的段落边界，不得包含纯标签、脚本或样式内容。只有空白、标题重复或版权模板的内容不算有效 abstract；不得从 PDF 正文生成或总结 abstract。

### DOI normalization

逻辑语义是去除 DOI URL/prefix、去除前后空白并输出小写；非法或空 DOI 归一化为 null。不得根据 title 或 publisher 猜测 DOI，也不得通过字符串模板制造 DOI。本任务仍不定义实现正则。

### URL normalization

`landing_url` 和 `pdf_url` 必须是绝对 HTTP/HTTPS URL，禁止用户名或密码；去除首尾空白，fragment 可以移除，query string 只有服务实际需要时才能保留。`pdf_url` 必须使用已验证且可重复使用的稳定 URL 候选，不表示最终 response URL；不得通过给 `landing_url` 拼接 `.pdf`、`/pdf` 或类似后缀猜测 PDF，也不得保存短期最终重定向 URL。

## 9. Identity, deduplication, and enrichment

### Stable record identity

schema version 1 不增加派生 `paper_id`。单次输出中的稳定身份由四元组 `(venue_id, year, source_name, source_id)` 确定；该四元组在一次 venue-year 输出中必须唯一。DOI 是跨来源标识和去重信号但不是唯一允许的记录身份，因为部分合法会议论文没有 DOI；title 不得单独作为记录身份，不得使用随机 UUID。

### Automatic deduplication

第一版只允许两种自动去重：完全相同的稳定身份四元组；同一 venue-year 中完全相同的 normalized DOI 且候选记录不存在明显身份冲突。重复稳定身份的后续出现计为 duplicate occurrence。相同 DOI 若 title、venue 或年份存在明显冲突，不能静默合并。title-only 或 fuzzy-title 匹配不得自动去重；`(title, first author, year)` 只能用于诊断候选，不得自动合并；不得使用编辑距离、embedding、LLM 或搜索排名自动合并。冲突无法可靠解决时必须产生 issue，并使运行不能 complete。

### Membership freeze

authoritative adapter 完成 membership 枚举和初始去重后，membership 集合必须冻结。enricher 不得增加、删除、替换论文或改变 source identity，只能为已有记录提供字段 patch；enrichment 返回未知 source identity 必须被拒绝并记录错误。

### Enrichment matching

匹配优先级为：normalized DOI 精确匹配；adapter 明确支持的稳定 source-specific identifier；只有没有 DOI 时才可使用严格 metadata match。严格 match 必须同时满足 normalized title 精确一致、year 一致、normalized first author 精确一致且候选唯一。不得仅按模糊标题、搜索结果第一名、citation count、embedding 相似度或 LLM 判断选择候选。

### Field update rule

primary source 的非空字段默认不得被 enricher 覆盖；enricher 默认只能填补 null 或空缺字段。每个最终字段必须更新对应 `field_sources`；enricher 返回空值不得清空已有值。DOI 冲突、identity 冲突或无法解释的强字段冲突必须产生 issue；普通格式差异可以产生 warning，但不能静默覆盖。enrichment 失败不得删除 primary membership；失败导致必填字段缺失时，`metadata_complete` 必须为 false，对应论文必须进入 `issues.jsonl`，run status 不得为 complete。

## 10. Output artifacts

第一版一次运行输出到一个目录，目录中固定包含 `papers.jsonl`、`issues.jsonl`、`run.json`。对任何成功完成原子发布的 run outcome，正式输出目录必须包含这三个文件，允许 JSONL 文件为空；`run.json` 最后发布，只有最终 `run.json` 存在时调用者才可认为该 run 已正式发布。

### papers.jsonl

`papers.jsonl` 必须是 UTF-8、无 BOM、每行一个完整 JSON object、每行符合 Complete paper record，并以换行结束。不得写入 incomplete paper、parse reject、excluded item 或 unverified PDF candidate。partial run 中可以包含已经完整的论文，但文件存在不代表 run complete，必须查看 `run.json`。输出必须按 `(venue_id, year, source_name, source_id)` 确定性排序。

### issues.jsonl

`issues.jsonl` 只包含运行结束时仍未解决的 blocking issues；每行必须始终包含且只包含以下顶层字段：`schema_version`、`issue_kind`、`venue_id`、`year`、`source_name`、`source_id`、`source_locator`、`title`、`authors`、`abstract`、`doi`、`landing_url`、`missing_fields`、`reason_codes`、`message`。

字段规则：`schema_version` 是固定为 1 的 integer；`issue_kind` 只允许 `incomplete_paper`、`parse_reject`、`identity_conflict`、`field_conflict`；`venue_id` 是非空 canonical ID；`year` 是请求的四位数 integer；`source_name` 是 string 或 null，已知 authoritative source 时必须非 null；`source_id` 是 string 或 null，无可靠 ID 时可为 null；`source_locator` 是用于定位原始条目、页面、cursor 或来源记录的 string 或 null，不得含密码、cookie、token 或学校认证信息且不得用作 complete paper identity；`title` 是 normalized string 或 null；`authors` 是 normalized string array，无法解析时可以为空数组但不得为 null；`abstract` 是 normalized string 或 null；`doi` 是 normalized bare DOI 或 null；`landing_url` 是绝对 HTTP/HTTPS URL 或 null；`missing_fields` 是 string array，只允许 `title`、`authors`、`abstract`、`access_locator`，可以为空；`reason_codes` 是非空 snake_case string array，至少一个机器可读原因；`message` 是非空 human-readable string。

`reason_codes` 示例包括 `missing_abstract`、`missing_authors`、`no_verified_pdf_or_doi`、`parse_failed`、`identity_conflict`、`doi_conflict`、`field_conflict`。incomplete included paper 必须有对应 issue；每个 unresolved parse item 必须有对应 `parse_reject` issue；excluded non-paper item 不进入 `issues.jsonl`，只进入 `run.json` 统计。不得写入任意 raw_metadata catch-all 或完整原始 provider response；message 也不得包含 credential、cookie 或 token。

issue_kind 到 completeness 的强制映射为：

| issue_kind | completeness effect |
|---|---|
| `incomplete_paper` | `metadata_complete` 必须为 false |
| `parse_reject` | `membership_complete` 必须为 false |
| `identity_conflict` | `membership_complete` 必须为 false |
| `field_conflict` | `metadata_complete` 必须为 false |

任何 issue record 都必须使至少一个 completeness boolean 为 false，因此 `issue_records > 0 => complete = false`，不得出现两个 completeness boolean 都为 true 且 `issue_records > 0`。可靠解决的身份或字段差异不得继续作为 issue record，可以作为 warning；warning 不改变 completeness。DOI conflict 按 field conflict 处理，可靠解决前必须使 `metadata_complete` 为 false。

### Blocking diagnostics

Paper-level blockers 必须进入 `issues.jsonl`：incomplete paper、unresolved parse item、identity conflict、field/DOI conflict。Run-level membership blockers 必须进入 `run.json.errors`，不得伪造成 paper issue：cursor cycle、terminal cursor 未到达、authoritative request 中断、source total mismatch、parser census mismatch、applicable empty result 缺少 authoritative zero proof，以及其他无法绑定到单条论文的 membership failure。

以下是缺少 abstract 但保留其他已取得字段的语法有效 `incomplete_paper` 示例：

```json
{"schema_version":1,"issue_kind":"incomplete_paper","venue_id":"example-conf","year":2024,"source_name":"example_proceedings","source_id":"paper-002","source_locator":"https://example.org/papers/002","title":"Incomplete Example","authors":["Dana Example"],"abstract":null,"doi":"10.1234/example.2024.002","landing_url":"https://example.org/papers/002","missing_fields":["abstract"],"reason_codes":["missing_abstract"],"message":"Authoritative metadata did not provide an abstract."}
```

### run.json

`run.json` 是 UTF-8、无 BOM、以换行结束的 JSON object；每次运行只有一个，必须最后原子发布。其存在表示此次运行已经结束并完成结果发布；不使用随机 run ID，第一版不要求时间戳，也不存储完整 provider response。

`run.json` 必须始终包含且只包含以下顶层字段：`schema_version`、`status`、`venue_id`、`venue_name`、`venue_type`、`year`、`source_name`、`membership_complete`、`metadata_complete`、`complete`、`counts`、`pagination`、`warnings`、`errors`。

字段规则：`schema_version` 是固定为 1 的 integer；`status` 只允许 `complete`、`partial`、`failed`、`not_applicable`；`venue_id`、`venue_name`、`venue_type`、`year` 与 Complete paper record 语义一致；`source_name` 是 string 或 null，not_applicable 或 primary source 尚未成功解析时可为 null；`membership_complete`、`metadata_complete`、`complete` 是 boolean，必须遵守本文 Completeness 章节，且 applicable run 满足 `complete = membership_complete AND metadata_complete`。`status = complete` 时 complete 必须 true，`partial` 或 `failed` 时必须 false，`not_applicable` 时也必须 false。

`counts` 固定包含 `raw_items`、`included_papers`、`complete_papers`、`incomplete_papers`、`excluded_non_papers`、`duplicate_occurrences`、`parse_rejects`、`issue_records`，全部是大于等于 0 的 JSON integer，并满足：`raw_items = included_papers + excluded_non_papers + duplicate_occurrences + parse_rejects`；`included_papers = complete_papers + incomplete_papers`；`complete_papers` 等于 `papers.jsonl` 行数；`issue_records` 等于 `issues.jsonl` 行数。`incomplete_papers > 0` 时 `metadata_complete` 必须 false；`parse_rejects > 0` 时 `membership_complete` 必须 false；`status = complete` 时 incomplete_papers、parse_rejects、issue_records 必须均为 0，warnings 可以非空，errors 必须为空。

`pagination` 可以是 JSON object 或 null。进行了 authoritative membership 请求时必须是 object，包含 `pages_fetched`、`terminal_reached`、`source_total`；前者为非负 integer，次者为 boolean。`source_total` 为 object 或 null，非 null 时包含非负 integer `value` 和只允许 `raw_items` 或 `included_papers` 的 `scope`。adapter 必须声明 total 语义；scope 为 raw_items 时与 raw_items 核对，scope 为 included_papers 时与 included_papers 核对；不一致使 `membership_complete` 为 false。未进行网络请求的 not_applicable run 的 pagination 必须为 null，primary source 请求前失败时可以为 null。

`warnings` 和 `errors` 都是非空 string array 项的 JSON array；warning 不得隐藏 required-field failure，可以存在于 complete run；status = failed 时 errors 必须非空，status = complete 时 errors 必须为空；errors 不得包含 credential、token、cookie 或完整 HTML 响应。

以下是满足计数方程的语法有效 partial `run.json` 示例：

```json
{"schema_version":1,"status":"partial","venue_id":"example-conf","venue_name":"Example Conference","venue_type":"conference","year":2024,"source_name":"example_proceedings","membership_complete":true,"metadata_complete":false,"complete":false,"counts":{"raw_items":2,"included_papers":2,"complete_papers":1,"incomplete_papers":1,"excluded_non_papers":0,"duplicate_occurrences":0,"parse_rejects":0,"issue_records":1},"pagination":{"pages_fetched":1,"terminal_reached":true,"source_total":{"value":2,"scope":"included_papers"}},"warnings":[],"errors":[]}
```

### Atomic publication

每个文件必须先写到同一输出目录中的临时文件，完成写入后使用原子 replace，且 `run.json` 必须最后发布。调用者只有看到最终 `run.json` 后才可以认为运行结束；不得先发布 status=complete 的 run.json 再继续写 papers.jsonl。中途中断留下的临时文件不属于正式输出。若写临时文件、flush、replace 或发布 run.json 失败，命令退出码为 4，不保证存在有效 run.json，也不得声称已发布 status=failed 的 run；实现应 best-effort 清理本次创建的临时文件，但不得删除或覆盖运行前已有的用户文件。若失败留下不完整正式文件，下一次运行仍按已有文件前置条件拒绝，由用户明确清理；第一版不提供 `--force`。

## 11. Run status and statistics

### complete

只有以下全部成立时才是 complete：applicable venue-year；`membership_complete = true`；`metadata_complete = true`；`complete = true`；`status = complete`；errors 为空；`incomplete_papers = 0`；`parse_rejects = 0`；`issue_records = 0`。

### partial

partial 表示已经取得部分可信结果或诊断，但 membership_complete 或 metadata_complete 至少一个为 false。`papers.jsonl` 可以包含完整论文，`issues.jsonl` 必须解释未完成论文或 parse reject，complete 必须 false，且不得对用户显示为完整成功。partial 必须满足 `issue_records > 0 OR errors 非空`；可以同时有 issue records 和 errors，errors 可以非空。warnings 只用于非阻塞信息，单独存在不得使 run 变成 partial，也不得替代 blocking issue 或 error。必填摘要缺失、无 direct PDF 且无 DOI、cursor cycle、expected total mismatch、unresolved parse reject、identity conflict、enrichment 导致必填字段缺失，或 primary membership 有部分有效结果但请求中途中断，都必须是 partial。

### failed

failed 表示无法取得可信 authoritative membership 集合，或 primary adapter 在产生可用 membership 前失败；`membership_complete`、`metadata_complete`、`complete` 必须均为 false，errors 必须非空，`papers.jsonl` 必须为空，`issues.jsonl` 可以为空或含有限诊断。普通参数错误不使用 failed，而由 CLI 退出码单独处理。

### not_applicable

not_applicable 表示 venue catalog 明确证明该 venue 在请求年份不存在、未举办、尚未创刊或已经终止；不得访问 provider，counts 全部为 0，pagination 为 null，三个 completeness boolean 均为 false，errors 必须为空，status 为 not_applicable。这不是“权威证明论文总数为零”的 applicable empty census；有明确 total=0 证明的 applicable empty census 可以是 complete，二者必须严格区分。

## 12. Command-line contract

第一版核心命令为：

```text
paper-agent collect \
  --venue <canonical-venue-id> \
  --year <four-digit-year> \
  --output <new-output-directory> \
  --contact <email>
```

- `--venue` 必填，只接受 canonical venue ID；不要求 alias 支持。未知 venue ID 属于参数/catalog 错误，输出使用同一个 canonical ID。
- `--year` 必填，是四位数 integer，必须在合理出版年份范围内；精确范围由后续实现决定。格式合法但不适用时返回 not_applicable，而不是参数错误。
- `--output` 必填，指定不存在或不含正式输出文件的目录；若已有 papers.jsonl、issues.jsonl 或 run.json，必须在访问 provider 前失败，不得静默覆盖；第一版不提供 `--force`。
- `--contact` 必填，是非空 email-like string，用于 API polite pool、User-Agent 或服务要求；不得写入 papers.jsonl、issues.jsonl 或第一版 run.json，不得作为 provider credential。

第一版不要求定义 list-venues；后续增加不得改变 collect 契约。

固定退出码：

| Exit code | Meaning |
|---:|---|
| 0 | status 为 complete 或 not_applicable |
| 2 | CLI 参数、catalog、输出目录前置条件错误 |
| 3 | status 为 partial |
| 4 | 已成功发布 run.json 且 status = failed；或不可恢复的运行时或 artifact publication failure，导致无法形成有效 run.json |

exit code 4 表示以下任一情况：已成功发布 `run.json` 且 `status = failed`；或发生不可恢复的运行时或 artifact publication failure，导致无法形成有效 `run.json`。参数、catalog 和运行前输出目录前置条件错误使用 2；partial 使用 3；complete/not_applicable 使用 0。exit code 0 不代表一定产生论文；not_applicable 使用 0 但 run.json.status 必须明确；不得因 papers.jsonl 有部分有效论文就把 partial 改为 0。

## 13. Non-goals

第一版不做：

- 下载或保存完整 PDF；
- 使用学校或出版社账号登录；
- 绕过付费墙、CAPTCHA、Cloudflare 或访问控制；
- 主题关键词搜索；
- 引用网络扩展；
- PDF 正文解析；
- 文献筛选、reranking 或 embedding；
- LLM 摘要生成；
- 论文分析；
- 报告生成；
- SQLite 或其他数据库持久化；
- 第三方不可信插件执行；
- 通用 workflow/task engine；
- Stage 2、Stage 3 或 Stage 4 的兼容层。

- schema version 1 不输出 raw provider responses；
- schema version 1 不输出任意 provider-specific metadata；
- schema version 1 不支持覆盖已有输出目录；
- schema version 1 不进行模糊论文身份合并。

## 14. Deferred decisions

以下内容不在本任务中定义：

- Python dataclass 的具体实现；
- adapter Protocol；
- enricher Protocol；
- HTTP client 和 retry 参数；
- host rate limiting 实现；
- PDF probe 的具体请求算法；
- venue catalog 的文件格式；
- adapter 自动发现方式；
- 具体 normalization 函数实现；
- 具体 logging 实现。

Deferred decision 不得推翻本契约已经确定的外部行为。后续实现可以选择内部结构，但必须满足本文 JSON 和状态不变量。
