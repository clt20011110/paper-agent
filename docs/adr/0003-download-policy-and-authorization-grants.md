# ADR 0003：下载策略矩阵与授权 grant

状态：已接受（设计冻结）

下载顺序固定为：官方明确公开直链、PMC/Europe PMC OA、Unpaywall OA location、匹配 arXiv、已授权浏览器 skill、`manual_queue`。resolver 只产生不可信的逐 URL `AccessLocationCandidate`；`probe` 依据版本化矩阵返回 `allow/needs_grant/manual/deny`，不得下载正文。URL 存在、`access_basis` 非空、bronze 或 `license=null` 都不等于授权。

矩阵按 `purpose × access_basis × license × publication_version × provider terms` 决策：有明确兼容许可与条款证据的开放内容才可自动 `allow`；`public_read_only`、bronze/null license、unknown、user subscription 不能自动保存，个人研究/内部分析通常是 `needs_grant`，无法机器判定为 `manual`，没有兼容许可的再分发为 `deny`。`user_supplied` 不赋予再分发权。

`download-access-v2` 把 Europe PMC core API 的受控标签 `cc by`/`cc-by` 归一为未版本化的 CC BY 许可证族。只有 candidate 已是 `open_license`、provider terms 明确允许下载与保存、purpose 为 `personal_research` 或 `internal_analysis` 时才兼容；它不会被提升为 CC BY 4.0，缺少具体版本时也不会自动通过 `redistribution`。`download-access-v1` 保留不变，便于审计旧 run。

grant 采用 canonical JSON content hash 与 detached approval；SQLite 是运行时唯一授权来源。它精确约束 action、purpose、paper/collection/selection snapshot、domain、max papers、模式、expiry，以及 download skill digest 或 remote-model artifact/lineage/provider/model hash。撤销、过期和 digest 漂移即时失效。YAML `grant_defaults` 只可生成草稿，不能扩大获批范围。重新获得有效 grant 后必须重新 probe，新的 allow 才能生成持久化 `FetchRequest`；fetch 拒绝手工构造、过期、撤销或字段不匹配请求。

浏览器默认 attended，登录状态本身不是批量无人值守许可。除非有独立 data-sharing grant，Luna 仅可接收净化控制状态、目标 ID 与最小必要页面内容，不能接收订阅正文、截图或账户标识。系统不绕过验证码、DRM、访问控制或站点限制。
