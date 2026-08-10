# 发布检查清单

每次发布从干净 checkout 开始，所有项目均须有可追溯证据。未通过项应阻止发布，而不是在 release
note 中声明已知可用。

## 构建前

- [ ] 确认版本号、`uv.lock`、Python 3.11–3.13 支持范围和变更说明一致。
- [ ] 检查 `git status --short`，审阅变更范围；不把本地 `dist/` 旧 wheel 当候选产物。
- [ ] 更新 README、[getting-started.md](getting-started.md)、[operations.md](operations.md)、
  [migration-v1-to-v2.md](migration-v1-to-v2.md) 和 Skill，确保命令合同一致。
- [ ] 核对 typed workflow 示例：`run` 与 `resume` 都带 `--workflow`，每个 FileRef 都是相对路径和
  小写 SHA-256；不得保留 argv 型 workflow 示例或声称裸 `--run-id` 可恢复。
- [ ] 审核 provider/venue manifest、prompt、schema、policy、model lock 与授权 skill audit 的
  版本/digest 变化；必要时创建新的审计记录和 release 输入。

## 自动离线门禁

```sh
uv sync --locked --extra dev
uv run --no-sync pytest --disable-socket --allow-unix-socket \
  --allow-hosts=localhost,127.0.0.1,::1
uv build --wheel
uv venv /tmp/paper-agent-wheel
uv export --locked --no-dev --no-emit-project --output-file /tmp/paper-agent-runtime.txt
uv pip install --python /tmp/paper-agent-wheel/bin/python --requirements /tmp/paper-agent-runtime.txt
uv pip install --offline --no-deps --python /tmp/paper-agent-wheel/bin/python dist/*.whl
(cd /tmp && /tmp/paper-agent-wheel/bin/paper-agent --version)
```

`uv sync` 是 CI bootstrap，在冷缓存 runner 上会从 package index 取得 lockfile 指定依赖；
“离线门禁”指随后 pytest 进程禁用外部 socket，不能把 bootstrap 描述成物理断网安装。

- [ ] 全部单元、契约、mock pipeline 和 skill/CLI contract 测试在 socket 禁用模式下通过；测试不会
  访问真实站点、浏览器、模型或 Codex。依赖安装仅按 lockfile 解析，不是一次真实 smoke。
- [ ] 在隔离环境以 `uv pip install --offline` 实际安装刚构建的 wheel，并从源码树外运行
  console version、包数据载入与 SQLite migration 验证；这证明打包内容可用，不证明任何外部
  provider 或生产 `doctor` gate 已通过。
- [ ] 检查 wheel 中的 migrations、schemas、providers、venues、acceptance、policies、prompts、
  registries 等运行时数据；每个 acceptance manifest 的主 fixture 和全部原生传输 route fixture
  必须存在且摘要匹配。配置模板与 Codex skill 必须随同一 release 明确分发。

## 人工生产门禁

- [ ] 在新环境按 getting-started 文档安装，复制同版本模板，并以 `doctor --production-ready`
  验证预期门禁。
- [ ] 完成一次受控、小预算的真实 smoke：已批准 QueryPlan、最少 provider/论文、准确 Stage 2
  release、审计输出和明确 operator/contact。它不能进入普通 PR CI。
- [ ] 如 smoke 使用授权下载或远程模型，逐项确认独立 grant、scope、expiry、artifact/lineage hash，
  并确认没有凭据、cookie 或全文写入日志。
- [ ] 如发布报告工作流，保存 ReportPlan、deterministic verifier、独立 Sol audit 和 coverage
  evidence；blocker 或 major 非零则不发布 report/latest。

## 发布与回滚

- [ ] 记录 source commit、wheel SHA-256、测试/CI run、人工 smoke evidence、已知限制和升级说明。
- [ ] 仅在上述证据齐全后打 tag、上传 wheel/skill/templates，并创建 release note。
- [ ] 保留上一稳定 wheel、模板、数据库备份和迁移说明。回滚应用版本不会倒改 SQLite 或不可变
  artifacts；若 schema/manifest 已漂移，按迁移和恢复流程建立新 run。
