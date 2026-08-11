# Authorized browser 真实 PDF smoke（2026-08-11）

本门禁已通过。source commit `79fabcd798024509a908d285201100140934eb2c` 使用 Microsoft Edge 中用户可见的已登录授权会话，对 DOI `10.1038/s42256-025-01108-5` 完成了 exact grant → Luna 控制决策 → 固定浏览器下载 → skill staging/audit → Stage 3 恢复导入。

授权 grant `grant-authorized-nature-smoke-20260811-v5` 的 canonical hash 为 `e396ae55ad59a0ab8d97063363697c36a3af02f6af0daf0d05b16337de14559f`。它只允许 `www.nature.com`、一个 exact paper ID、`download/store`、`personal_research`、`max_papers=1`，采用 attended 模式且 `allow_unattended=false`。没有 collection/filter selection，因此 selection scope 由 grant 中唯一的 `paper_ids` 冻结，而不是空泛扩大到登录账户可访问的其他内容。

页面显示正常 full-access 状态和唯一的 `Download PDF` 控件；未出现 CAPTCHA、403、429、access denied 或安全警告。固定 Edge pass 从 `08:50:28.408Z` 开始，在 `08:50:59.178Z` 触发一个 article PDF，使用 30 秒间隔和 5 秒 jitter。Edge 下载栏随后要求可见的“保存”动作，因此只对该已触发项使用 Computer Use 点击 Save；没有读取 cookie、密码、OTP、local storage 或浏览器 profile。

Luna 使用 `stage3_authorized_luna / gpt-5.6-luna / low`，一次调用得到 `invoke_skill / invoke_audited_skill`。planner decision 为 `stage3-luna-ae626091-46dd-4fe6-a15c-f5d84a1a96ed`。下载 skill 的 ZIP、安装内容和 dependency digests 均与审计 manifest 一致；最终 audit 为 1 个唯一终态、1 个 article、0 个预期 SI、0 个缺失项、0 个 manual-required。

导入 artifact 为未加密、2 页的 PDF 1.4，MIME `application/pdf`，大小 `686261` bytes，SHA-256 `a110d36d8bf3b87710454698fde970fa3f7095a95116542dfbe7f0b4a1074c4b`。SQLite 中 run `authorized-smoke-nature-s42256-025-01108-5-v5` 最终为 `complete`，paper 为 `downloaded`，provider 为 `authorized_skill`，fetch request 为 `consumed`。PDF、正文、cookies 和账户标识均未提交仓库，也未声称任何再分发权限。

真实运行同时暴露并修复了两个问题：Codex CLI 0.147 的 JSONL 不回显 model/profile，`1fb9bb6` 在保留显式冲突拒绝的前提下从冻结调用绑定补齐标签；旧 prompt 把公开来源 `provider_terms_unmachineable` 错当成浏览器停止条件，`79fabcd` 明确了 planner 调用前置条件并由 skill 独立处理登录、挑战和下载结果。v2–v4 诊断运行均保持 incomplete，没有伪装成成功或修改数据库结果。

机器可读的净化证据见 `authorized-browser-20260811-evidence.json`。
