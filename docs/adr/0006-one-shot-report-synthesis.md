# ADR 0006：Stage 4b 采用单次 Sol 综合与本地发布门

状态：已接受（取代 ADR 0004，并修订 ADR 0001 的 Stage 4b 路由）

approved Stage 4b 从已批准 `ReportPlan`、冻结 QueryPlan、search audit、corpus snapshot 和全部 Stage 4 Luna 逐篇报告构造唯一输入。协调端按 stable paper ID 确定性排序，每篇完整 Luna 报告恰好出现一次。预算、上下文、许可、grant 或输入完整性门禁在 dispatch 前失败时，Sol 调用数为 0；禁止 shard、section/cross-section/final reduce、抽样、截断或静默缩小 corpus。

门禁通过后，`report_one_shot_runs` 先为 report run 原子预约唯一 dispatch，再以 `stage4b_oneshot_sol`、`gpt-5.6-sol`、high reasoning、`max_retries=0` 调用一次且仅一次 `one_shot_report`。并发 worker 和 `resume` 不能获得第二次 dispatch；Codex 启动后的 timeout、连接中断或结果不确定使旧 run 终止，再次调用必须创建并重新批准新的不可变 ReportPlan/report run。

Sol 只返回带 draft claim/evidence refs 和章节 blocks 的结构化草稿。协调端在本地确定性完成 stable claim/section normalization、Claims-Evidence Matrix、comparison groups、coverage ledger、`ReportDocument` AST、canonical bibliography、Markdown renderer、verifier、audit 和 publish。每个实质性 block 必须绑定合法 claim 与 citation；只有协调端生成的固定参考文献说明块可不绑定 claim。schema、normalize、verifier 或 local audit 任一失败，或 blocker/major 非零，均保留不可变输出并标 `incomplete`/`failed_terminal`，不更新 `latest`，也不调用 Sol repair、reaudit 或 retry。
