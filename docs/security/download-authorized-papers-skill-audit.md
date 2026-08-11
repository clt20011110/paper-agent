# download-authorized-papers skill 审计

状态：可作为 Stage 3 可选 provider 集成，但默认禁用并受 authorization grant 约束。

审计日期：2026-08-09  
原始文件：`download-authorized-papers-skill(1).zip`  
ZIP SHA-256：`ee69308c98ad8e564ee8098acc56628866c45657da259aa08bb29f8732874d5e`  
skill name：`download-authorized-papers`  
审计版本：`2026-08-09.1`（上游未提供版本）

## 内容与依赖

ZIP 测试通过，共 6 个文件、46,080 bytes 解压内容；没有绝对路径、`..` 路径、加密项或异常压缩比。压缩包不含许可证、依赖清单或 lockfile。Python 脚本仅使用标准库；浏览器脚本只使用 Node `fs/promises`、`path` 和宿主提供的 Playwright 风格浏览器接口。

| 文件 | SHA-256 |
|---|---|
| `SKILL.md` | `8fd451d6bb27f5f356b889e1a0041905c1d8ebafc70350d2ad63e36e624664d5` |
| `scripts/paper_queue.py` | `cceb01780236be40a50d26e3d2e2bb99c7666d58505f48f850749c4f306813e4` |
| `scripts/publisher-workflows.mjs` | `28ee3b9ee5876266ad2e36e86b3ff75e0966b44610b549831d810fe82079f671` |
| `references/data-contract.md` | `f766dc1f3d853d01d1aab1d5744a0dd0f2e5b8b2d6fe90e15ed7455ec6852501` |
| `references/browser-pass.md` | `b57e6d58e6a746d9a5860d8a9a19734fc27c0fb4d85847714370790f38d83969` |
| `agents/openai.yaml` | `7914671e946f6f1fe73b0cc3a2b4dd7ead9fdc54789fdb0394371da2109d3c20` |

安装后必须重新计算目录内容 digest，并与 ZIP digest、上述逐文件 digest、审计版本一起登记。任一文件、依赖或 digest 漂移会禁用 provider，并使绑定旧 digest 的 grant 失效。

## 能力边界

该 skill 只覆盖 Wiley（DOI `10.1002/`）、Nature/Springer Nature（`10.1038/`）和 ACS（`10.1021/`）的正文 PDF 与 PDF 补充材料。其他出版社进入 `manual_queue`，除非新增并审计对应适配器。

skill 使用用户已有浏览器登录态和可见的普通下载控件；不会读取 cookie、密码、OTP、local storage 或会话文件。遇 CAPTCHA、403、429、access denied、缺少已授权 PDF 链接或安全警告时停止，不绕过访问控制。默认约 30 秒一篇，最短间隔 15 秒。下载文件经 `.part` 原子复制、PDF 魔数/HTML/大小校验并记录 SHA-256。

## 集成约束

- CSV 的可选 `url` 在进入浏览器前，必须按 DOI publisher 和 grant 的精确域名 allowlist 校验；禁止任意 URL。
- Nature/ACS 页面出现多个候选链接时必须返回 `manual_required`，不能选择第一个可见链接。
- 自动路径不暴露 `record --accept-scan-error`；只有 attended 模式的明确人工决定可以接受异常，并保存审计事件。
- Luna 默认只接收净化后的状态、paper/candidate ID 和最小控制信息。页面正文、截图或账户标识需要独立 `browser_data_sharing` grant，且不得写日志。
- 下载 grant 不授权把 PDF 或派生正文发送到 Luna/Sol；远程处理需要独立、绑定准确 artifact/lineage hash 的 `remote_model_processing` grant。

## 最小 grant

下载 grant 至少冻结：

- actions：`download`、`store`；需要本地抽取时另加 `extract`；
- purpose：`personal_research`；
- 精确 `paper_ids`，或不可变 collection/selection snapshot hash；
- `max_papers`、有效期、批准与撤销记录；
- skill name、审计版本、ZIP digest 和安装内容 digest；
- 本次候选实际需要的精确域名；
- mode 默认 `attended`。只有 skill 声明支持、站点允许且用户对同一冻结 scope 显式批准时才允许 `unattended`。

本次审计未声明 unattended 支持，provider contract 因而固定 `allows_unattended=false`。合法的
unattended public-provider grant 不受此限制；但该 browser skill 收到 unattended grant 时必须保持
`manual_required`，不得创建队列或把 grant 伪装成 attended。

## 需要用户参与

用户需要在选定浏览器中完成 Wiley、Nature/Springer 或 ACS 的机构/个人登录，并以目标页面实际出现正常 PDF 链接确认权限。CAPTCHA、403、429、缺授权链接和浏览器安全警告只能由用户处理；系统不会自动绕过。Chrome/Computer Use 回退只处理首轮固定流程留下的逐项失败，并保持相同 scope 与限速。
