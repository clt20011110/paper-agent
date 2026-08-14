# Stage 1 metadata collection contract

本文是 Stage 1-only 重构的规范性产品契约。后续实现和测试必须遵守本文。本阶段不定义 Python 类、JSON schema、CLI 退出码或 adapter API；这些内容将在后续小任务中定义。本文定义目标，不声明当前实现已经满足本文。

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

## 7. Non-goals

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

## 8. Deferred decisions

以下内容不在本任务中定义：

- `PaperRecord` 的精确字段和 JSON 表示；
- `authors` 的最终 JSON 结构；
- output directory 文件名；
- partial 与 failed 的输出规则；
- CLI 参数和退出码；
- Python dataclass；
- adapter/enricher Protocol；
- HTTP retry 参数；
- PDF probe 的具体算法；
- venue catalog 文件格式。
