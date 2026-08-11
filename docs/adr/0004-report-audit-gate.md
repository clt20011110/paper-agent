# ADR 0004：报告必须经确定性校验和独立 Sol 审计后发布

状态：已取代（由 ADR 0006 取代；以下保留历史决定）

报告从已批准 `ReportPlan`、冻结 QueryPlan、search audit、corpus snapshot 和 Stage 4 map artifacts 构造。先按语义 section 与稳定 paper ID 分块，再 section → cross-section → final reduce；不可用“前 N 篇”或随机抽样替代覆盖。

模型的 final 输出是带 `block_id/section_id/text/claim_id/citation paper_ids` 的 `ReportDocument` AST。协调端以 AST 确定性渲染 Markdown、生成 sidecar 与参考文献；每个实质性 block 必须绑定存在的 claim 和合法 paper citation。Claims-Evidence Matrix、comparison group、coverage ledger、source limitations 都必须通过确定性 verifier。

发布门为：verifier 通过，独立 Sol audit A 的 blocker=0、major=0，且审计覆盖全部实质性 block/claim。若 A 发现 blocker/major，只允许独立会话 B 作一次有界 typed repair。repair 生成新结构化 hash 后重新渲染、完整 verify，并以全新 Sol 会话 C 复审。第二次仍有 blocker/major 或覆盖不完整则 `incomplete`，保存不可变草稿与审计，但不得更新 `reports/latest.md`。Luna 的 grant 不自动授权 Sol；进入 Sol 的派生内容须按 lineage hash 单独通过处理策略或精确 grant。
