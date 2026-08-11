# Crossref 全链路受控 smoke（2026-08-11）

本次真实联网 smoke 针对 source commit
`64ee6c0e434fe1e7a75ec6397d4af0d807ce63b9`，使用已批准 QueryPlan 和必需的
Crossref provider，完整执行 `search → metadata_enricher → metadata_verifier`。最终
SearchPipeline 状态为 `complete`。

网络边界固定为三次 `https://api.crossref.org/works` 元数据请求：page size 为 1、每个
operation 最多一次尝试、禁止重定向、禁止 PDF、关闭 citation snowball。三次响应均成功并
保存了原始响应摘要、CAS 关联、request attempt、search audit 和限流 header。Crossref 没有
返回 allowlist 中的 quota/credit header，因此证据将二者明确记为 `unavailable`。本次没有使用
凭据、cookie 或全文。

Stage 2 使用 `DeterministicFakeScreener`，只为使 provider 链路可重复验收；这份证据不是
Stage 2 release evidence，也不能替代 600-pair 金标、structured replay、性能回放或 soak。

当前会话共发生七次 Crossref outbound：一次 page-size 缺陷诊断、三次因
`scope_field_unverified` 正确停止的诊断，以及本次三次成功请求。失败尝试没有被隐藏，详情在
`network_history` 中。

证据文件：

- `crossref-full-pipeline-20260811-evidence.json`：总证明与历史尝试。
- `crossref-full-pipeline-20260811-query-plan.json`：批准后的冻结计划。
- `crossref-full-pipeline-20260811-search-audit.json`：数据库导出的完整 search audit。
- `crossref-full-pipeline-20260811-transport-audit.json`：operation 与 HTTP 请求审计。
- `crossref-full-pipeline-20260811-response-manifest.json`：三份响应的摘要与快照映射。
- `crossref-full-pipeline-20260811-{search,enrich,verify}.json.b64`：原始公开元数据响应。

`tests/test_committed_provider_smoke.py` 完全离线校验计划 approval、当前 provider manifest、
所有文件与响应 SHA-256、三次 request-attempt 关联、URL 边界和最终 complete 状态。
