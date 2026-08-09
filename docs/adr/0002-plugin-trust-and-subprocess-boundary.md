# ADR 0002：第三方 provider/skill 的信任与隔离边界

状态：已接受（设计冻结）

外部 provider、skill ZIP 和下载文件一律是不可信输入。发现 Python entry point 时，先仅读取 distribution metadata；allowlist 必须精确绑定 distribution name、version、provider、entry point 和 wheel/sdist 或安装内容 SHA-256。manifest 是 role、capability、认证、限流、条款、`independence_group` 与 `upstream_families` 的唯一事实源。版本或 digest 漂移立即禁用注册，必须重新审计；旧 skill 相关 grant 同时失效。

内置 provider 可由主进程加载。已 allowlist 的第三方 provider 仍默认在最小网络、文件和环境权限子进程中运行，只通过版本化 IPC 返回 `SourceBatch`/`CitationBatch`；子进程不写 SQLite。提升为 in-process 是单独的信任决定，必须新增 ADR 和审计记录。

`download-authorized-papers` 仅为可选、默认关闭的运行时能力。安装前审计 ZIP、`SKILL.md`、脚本、依赖和权限；记录 ZIP hash、安装内容 digest、名称/version 与依赖 lock/hash。doctor 从安装环境发现实际名称，核心代码不硬编码用户目录。不会记录 cookie、token、账号或敏感页面内容。
