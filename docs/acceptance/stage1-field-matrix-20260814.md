# Stage 1 venue/year field matrix audit

更新日期：2026-08-14  
代码基线：`feature/crawler-adapters`

本文件记录新的 `paper-agent stage1 matrix` 审计接口及其第一次真实运行。该运行递归读取 `/private/tmp` 中已有的 Stage 1 receipt；它不是新的 live census，也没有把 fixture 计入结果。

运行命令：

```bash
paper-agent stage1 matrix \
  --receipts-root /private/tmp \
  --year-from 2016 --year-to 2025 \
  --output /private/tmp/stage1-field-matrix-20260814-v4.json \
  --markdown-output /private/tmp/stage1-field-matrix-20260814-v4.md
```

矩阵哈希：`1fd8599da7330099ea81edf4f8a2ca9c37c5a16ba94fd2a1f40940fa0ffbaed4`

## 结果

| 指标 | 数值 |
|---|---:|
| 当前目录 venue | 43 |
| 矩阵 cell | 430 |
| 适用 cell | 394 |
| `complete` | 202 |
| `not_applicable` | 36 |
| `conflict` | 29 |
| `failed` | 33 |
| `missing_receipt` | 120 |
| `unproven` | 10 |
| input receipt 解析错误 | 0 |
| 当前 proven cell 的记录数合计 | 237,319 |

### 解释

- `complete` 只表示该 cell 的某份 receipt 通过了 Stage 1 收集器的严格门禁。
- 同一 cell 有多个同等强度但 census/审计指纹不同的 receipt 时标为 `conflict`。例如同一年度的旧版诊断 receipt 和后续修复 receipt 不能被文件名自动合并。
- `missing_receipt` 表示当前证据目录中没有覆盖该 cell 的持久化 receipt，不表示该年度没有论文。
- `failed`/`unproven` 保留 provider 失败、限流、Cloudflare 或字段 enrichment 缺口；不降级为 complete。
- 新增 TMLR、TPAMI、DATE、ASP-DAC、ISPD、TODAES、TVLSI、JSSC 已有 descriptor、acceptance、fixture 和通用适配器 contract，但本次 `/private/tmp` 运行没有它们的 live decade receipt，因此不能把它们记为完成。

后续只有在每个适用 cell 都有唯一、无冲突且字段门禁通过的 live receipt 后，矩阵才会返回 `status=complete`；当前应继续保持 `incomplete`。
