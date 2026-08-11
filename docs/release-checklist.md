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
(cd /tmp && /tmp/paper-agent-wheel/bin/paper-agent stage2-evaluator promote --help)
(cd /tmp && /tmp/paper-agent-wheel/bin/paper-agent stage2-evaluator attest --help)
(cd /tmp && /tmp/paper-agent-wheel/bin/paper-agent stage2-release assemble --help)
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
  必须存在且摘要匹配。
- [ ] 从隔离 wheel 环境调用 `paper_agent.resources.release_asset_root()`；路径必须以当前版本结尾，
  其中 Stage 2 model locks、全部示例 config、`configs/stage2/hidden-evaluator-trust.example.json`、
  `skills/paper-agent/SKILL.md` 和 agent metadata 必须存在。trust example 必须保持 retired、无 active
  key，实际交给 loader 时以 `no active key` fail closed，不能把 fixture trust 当生产默认值。
  隔离环境的普通 `paper-agent doctor` 必须返回成功，且 `stage2_model_locks=pass`；不允许仅接受
  非零退出码后跳过该检查。实际把 config 和 skill 从该路径复制到临时目标，确认无需源码 checkout。
- [ ] CLI contract 回归必须覆盖 `stage2-evaluator promote/attest` 与 `stage2-release assemble` 的完整
  required options、结构化状态、existing-output 拒绝和 global `--dry-run`。dry-run 不得读取 evaluator
  private labels/submissions/key、消费 promotion marker 或创建 attestation/release output；assembly
  dry-run 必须执行与真实组装相同的 gate/trust/path 验证。
- [ ] 告警契约回归覆盖 Stage 2 的 15%/30%、0.5%、28 GiB 边界、resume 等价和 report Codex
  budget exhaustion；search audit 只保留 allowlist rate/quota/credit headers，凭据与 cookie fixture
  不得出现在产物中。

## 人工生产门禁

以下项目必须针对当前 source commit 留存真实运行证据；mock、fixture、snapshot replay、`--dry-run`、
`doctor` 和旧 commit 的 smoke evidence 均不能代替。

- [ ] 在新环境按 getting-started 文档安装，复制同版本模板，并以 `doctor --production-ready`
  验证预期门禁。
- [ ] 以冻结真实样本完成 Stage 2 release gate：600 个 topic-paper pair（300 DEV、150 hidden
  hard-case、150 hidden real-distribution）、至少 1,000 次 adjudicator structured replay、normal/stress
  各三次 1,000-case 性能回放和 10,000-case soak；同时保存 label-custody evaluator、rationale/parity
  gate、manifest、release/model hash、环境和 benchmark records。fixture 规模记录不能替代。
- [ ] 在隔离 evaluator 中按
  [hidden evaluator custody runbook](security/stage2-hidden-evaluator-custody.md) 运行完整
  `stage2-evaluator promote` dry-run，确认只验证 public manifest/candidates、trust 和 output 边界；
  private labels、submissions、key、state root、marker 和 output 均未被读取或修改。记录 public input
  hashes 和结构化 `status: validated`，但不得把 dry-run 当真实晋级证据。
- [ ] 真实 promotion 只执行一次。私钥是 current-owner、non-symlink regular、精确 `0600`、≤16 KiB
  的 canonical unencrypted Ed25519 PKCS#8 PEM，并匹配当前 active trust key。保存 attestation、marker
  hash、candidate/evaluation IDs 和结果；gate failure 也消费 holdout 并保留 signed failure，不得删除
  marker、重试同一 holdout 或将失败结果组装为 release。HSM/secret-manager API 未经独立集成不得
  声称由当前 CLI 直接支持。
- [ ] 若使用 `stage2-evaluator attest`，证明 public-safe payload 来自另一个已审阅、已实施一次性
  marker 的 sealed evaluator；该命令本身不执行 hidden evaluation，不能替代 `promote` 的 custody。
- [ ] 先以 global `--dry-run`、再以完全相同的 `--candidate --evidence --trust-manifest --output`
  参数运行 `stage2-release assemble`。确认 candidate/output 具有同一 parent，evidence 及其全部引用
  留在该 bundle root，trust manifest 在 root 外且 output 原先不存在；assembly 重算 public gates、
  验证 hidden signature 与 candidate/model/calibrator/threshold/manifest/split bindings。bundle 不含 private labels、raw
  submissions、私钥或 marker state。
- [ ] 对组装后的 schema-v3 release 配置部署控制的 `PAPER_AGENT_STAGE2_HIDDEN_TRUST`，从隔离 wheel
  环境实际运行 `doctor` 和至少一个 production release loader。该环境变量只供已组装 release 加载，
  不替代 evaluator/assembler 的显式 `--trust-manifest`；不得用 release 内 `passed`、fixture key、
  私钥或未激活 key 代替验签。
- [ ] 完成受控、小预算的真实 provider smoke：使用已批准 QueryPlan、至少一个实际启用的 required
  provider 和明确 operator/contact；保存当前 source commit、provider manifest/response hash、search
  audit、request attempts，以及 rate/credit 值或明确的 `unavailable`。它不能进入普通 PR CI。
- [ ] 完成真实 PDF smoke：至少一篇 public OA 论文走完 candidate → probe → fetch → PDF validation；
  若发布 authorized browser 能力，另以用户可见的已登录会话和 exact attended grant 完成一篇允许
  域名与 selection scope 内的成功下载。保存原始 ZIP、installed skill、dependency、grant、selection、
  immutable queue、final ledger 和 imported artifact hashes；第一次 download 生成 queue 后，按 skill
  完成 30 秒 + 5 秒 jitter 的 fixed pass、stage 和 clean audit，再以完全相同 run/grant/queue/output/
  root/ZIP/audit 参数重跑 download 导入。CAPTCHA、403、429、access denied、missing link、
  `stopQueue` 或 `manual_required` 只证明安全停止，不算成功 smoke；不得记录凭据、cookie 或全文。
- [ ] 如发布报告工作流，保存 ReportPlan、deterministic verifier、独立 Sol audit 和 coverage
  evidence；blocker 或 major 非零则不发布 report/latest。

## 发布与回滚

- [ ] 记录 source commit、wheel SHA-256、测试/CI run、人工 smoke evidence、已知限制和升级说明。
- [ ] 仅在上述证据齐全后打 tag、上传已内含版本化 skill/templates/model-lock assets 的 wheel，并创建
  release note；发布记录须写明 `release_asset_root` 的版本路径和 wheel SHA-256。
- [ ] 保留上一稳定 wheel、模板、数据库备份和迁移说明。回滚应用版本不会倒改 SQLite 或不可变
  artifacts；若 schema/manifest 已漂移，按迁移和恢复流程建立新 run。
