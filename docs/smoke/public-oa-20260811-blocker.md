# Public OA 默认下载 smoke blocker（2026-08-11）

本记录只描述一次安全停止，不是成功的 PDF release evidence。

## 已验证范围

- source implementation：`64ee6c0e434fe1e7a75ec6397d4af0d807ce63b9`
- run：`public-oa-smoke-20260811`
- DOI：`10.3758/s13421-020-01060-2`
- PMCID：`PMC7683441`
- Europe PMC candidate：`https://europepmc.org/articles/PMC7683441?pdf=render`
- publication version：published
- access basis：open license
- license：CC-BY-4.0
- purpose：personal research

Europe PMC `resultType=core` 元数据成功解析出同 host 的 PDF candidate。冻结的 provider terms
与下载 policy 对该 candidate 产生 `allow / compatible_open_license`，fetch request 在网络副作用前
被持久化并消费。

实际默认 `urllib_fetch` 没有发出 HTTP 请求。`socket.getaddrinfo("europepmc.org")` 在当前桌面
网络返回 `198.18.0.66`；该地址属于 `198.18.0.0/15`，Python 将其判为 non-global，因此
`_require_public_dns` 抛出 `OSError`。下载 attempt 持久化为
`failed_retryable / network_error / OSError`，没有创建 PDF artifact。

系统 `curl` 的 HEAD 诊断能通过本机代理到达 Europe PMC，并观察到同 host 的
`/api/getPdf?pmcid=PMC7683441` 跳转和 PDF 响应。这只能证明另一套网络栈可达，不能证明默认
生产传输、DNS/SSRF guard、redirect 或读取上限可用。没有把 curl fetcher 接入系统，也没有
下载或提交全文。

## 门禁结论

public OA release smoke 保持未通过。不得放宽 SSRF guard，也不得把 curl 注入或 HEAD 诊断记为
成功。解除门禁需在 DNS 为 `europepmc.org` 返回真实 global IP 的环境中，用默认
`urllib_fetch` 对一篇合法 public OA 论文重新执行 candidate → probe → fetch → PDF validation，
并保存 request、attempt、artifact hash、MIME、大小和解析结果。

学校订阅或出版社浏览器授权下载是独立门禁，仍需用户可见登录会话与 exact grant；本次诊断未
使用任何凭据、cookie 或访问控制绕过。
