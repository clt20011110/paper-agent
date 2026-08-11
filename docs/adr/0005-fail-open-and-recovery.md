# ADR 0005：fail-open 保护论文记录，不放宽权限或质量门

状态：已接受（设计冻结）

`fail-open` 的含义是单个来源、论文、模型或可重试请求失败时，保留原始记录、结构化错误和可恢复状态，让独立工作继续；它绝不表示自动纳入、自动下载、自动发送全文给远程模型、自动标 complete，或静默忽略 required source。

筛选失败、schema 不合法、超时、冲突或信息不足进入 `needs_review`；不会被判 `irrelevant`。下载授权/条款不确定进入 `needs_grant` 或 `manual_queue`；无处理授权时 full text 不会进入 Luna/Sol。报告的 required source 失败、预算耗尽、coverage 缺失或本地 deterministic audit gate 不通过均为 `incomplete`，但已有的不可变产物仍保留。

恢复依赖 SQLite 的 WAL、短事务、租约和 fencing token，而不是进程内锁。对相同冻结输入的重放是幂等的；不同 hash 的同键结果隔离为冲突。通用重试有配置上限并记录，不无限重试；Stage 4b one-shot 的重试上限固定为 0，唯一 dispatch 后的 timeout 或 uncertain outcome 终止旧 run，并发/resume 只能观察或复用已证明持久化的结果。没有自动云模型回退、模型升级或配置漂移覆盖。多机只处理协调端已冻结的 shard，协调端是唯一合并写者，迟到 epoch 结果被拒绝。
