# ADR 0001：模型路由固定为本地 Stage 2 与 Codex Stage 3–4b

状态：已接受（设计冻结；Stage 4b 路由由 ADR 0006 修订）

Stage 2 在本机 oMLX 执行：批量 reranker 为 `BAAI/bge-reranker-v2-m3`，疑难裁决固定为已验收 revision 的 `mlx-community/Qwen3.5-9B-8bit`。不超过 10B 参数的本地模型满足大规模并发筛选的成本与吞吐目标；Qwen 只处理灰区、异常和信息不足，失败进入 `needs_review`，不回退云端。

浏览器授权编排使用 `codex exec -m gpt-5.6-luna`，`stage3_authorized_luna` 为 low reasoning、最小能力、专用临时目录。逐篇分析使用 `stage4_analysis_luna`，固定 `gpt-5.6-luna`、medium reasoning、只读工作目录、网络关闭。approved Stage 4b 报告使用 `stage4b_oneshot_sol`，固定 `gpt-5.6-sol`、high reasoning、只读输入、网络关闭和 `max_retries=0`：协调端将全部 Luna 逐篇报告确定性排序后一次完整打包，只允许一次 `one_shot_report` 调用；normalize、verify、audit 与 publish 均在本地完成，不再调用 Sol。旧 `stage4b_summary_sol` 仅保留给 legacy reduce-tree run 只读恢复，不能被 one-shot run 调用。

模型 slug、profile、revision、量化、阈值、prompt/schema、输入 hash 与实际 invocation metadata 都是 run 记录。配置 schema 以 const 限制 Luna/Sol；不能由环境变量、用户默认模型或旧配置覆盖。移除 OpenRouter/OpenCode 运行时路径，不自动升级模型、不自动云回退。
