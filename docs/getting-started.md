# 新环境配置与首个运行

本文把 Paper Agent 安装到新机器，并先完成一个完全离线的验证。生产检索、模型推理和授权下载是后续明确选择的门禁；不要把离线验证视为它们已经可用。

## 1. 安装

支持 Python 3.11–3.13。macOS Apple Silicon 是本地 Stage 2（oMLX）的首要部署目标。

```sh
git clone <repository-url> paper-agent
cd paper-agent
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/paper-agent --version
.venv/bin/paper-agent --help
```

也可以在空目录中安装同一 release 的 wheel：

```sh
python3 -m venv .venv
.venv/bin/pip install /absolute/path/to/paper_agent-2.0.0a0-py3-none-any.whl
```

源码和 wheel 都通过同一个只读 locator 暴露与包版本绑定的模板、Stage 2 model locks
和 Codex skill。源码态返回 checkout 根目录；wheel 态返回
`<venv>/share/paper-agent/<paper-agent-version>`。先记录准确路径：

```sh
ASSET_ROOT="$(.venv/bin/python -c \
  'from paper_agent.resources import release_asset_root; print(release_asset_root())')"
test -f "$ASSET_ROOT/configs/stage2/models/bge-reranker-v2-m3-fp32.lock.json"
test -f "$ASSET_ROOT/configs/stage2/models/qwen3.5-9b-8bit.lock.json"
```

安装 Codex skill 时，将整个版本绑定目录复制到个人 Codex skills 目录。下面的目标路径
必须替换成当前 Codex 安装实际使用的位置；skill 只调用已安装的 `paper-agent`，不携带
第二套实现：

```sh
cp -R "$ASSET_ROOT/skills/paper-agent" \
  /absolute/path/to/codex-home/skills/paper-agent
```

## 2. 选择配置并运行 doctor

从窄范围模板开始，复制到自己的受保护项目目录；不要原地修改安装资源。wheel 已把
venue/provider manifests、schema、prompt、policy 和 registry 安装到同一 Python 环境。运行时
locator 在源码态读取 checkout，在 wheel 态读取当前 Python 环境中的冻结版本；无需把这些只读目录
复制进项目，也不要从另一个 release 混入资源。

```sh
mkdir -p /absolute/path/to/paper-research
cp "$ASSET_ROOT/configs/smoke_supported.yaml" \
  /absolute/path/to/paper-research/research.yaml
.venv/bin/paper-agent --config /absolute/path/to/paper-research/research.yaml doctor
```

配置中的输出、SQLite、provider snapshot 和模型验收产物等可变路径仍以 `research.yaml` 所在目录
为根。核心只读 manifests/schema/prompt/policy 由安装 locator 提供，Stage 2 默认 model locks 则由
上述版本化 asset root 提供。也可以把配置留在完整 checkout 中，只把 output/database 路径改成
绝对项目路径。

普通 `doctor` 可以在未安装模型、未配置账号或没有批准计划时报告 warning/blocker。
在真实生产 campaign 前再运行：

```sh
.venv/bin/paper-agent --config /absolute/path/to/paper-research/research.yaml \
  doctor --production-ready
```

只在你明确同意用冻结的 Codex profile 实际探测付费模型时使用
`--prove-paid-models`；它不是离线检查的一部分。

## 3. 本地 Stage 2 与外部凭据

Stage 2 只能使用已经验收的本地 oMLX release。安装并校验与配置匹配的模型 lock、
calibrator、threshold 和 release bundle；`configs/stage2/models/*.lock.json` 仅是
锁定输入，不等于 release 已通过。随后用同一 approved QueryPlan 和
`--stage2-release` 运行 `doctor`，确认模型 ID、revision、digest 和本地 endpoint。

schema-v3 release 不能手写或仅修改 schema version。隔离 evaluator 优先用一次性的
`stage2-evaluator promote` 同时执行 hidden gate、创建持久 marker 并签名。先做 public-only
dry-run；虽然 private/key/state/output 参数仍为必填，但这一步不读取 private labels、submission
或私钥，不接触 marker，也不创建 output：

```sh
paper-agent --dry-run stage2-evaluator promote \
  --manifest /secure/evaluator/gold-manifest.json \
  --private-labels /secure/evaluator/private-labels.json \
  --candidate incumbent=/secure/evaluator/incumbent-candidate-v2.json \
  --candidate challenger=/secure/evaluator/challenger-candidate-v2.json \
  --submission incumbent=/secure/evaluator/incumbent-submission.json \
  --submission challenger=/secure/evaluator/challenger-submission.json \
  --incumbent-candidate-id incumbent \
  --selected-candidate-id challenger \
  --evaluator-id evaluator-team-1 \
  --evaluation-run-id promotion-2026-08-11 \
  --state-root /secure/evaluator/state \
  --evaluator-key-id evaluator-key-2026-08 \
  --issued-at 2026-08-11T08:00:00Z \
  --trust-manifest /secure/deployment/hidden-evaluator-trust.json \
  --signing-key-file /secure/evaluator/hidden-promotion-key.pem \
  --output /secure/transfer/hidden-promotion-attestation.json
```

确认 `status: "validated"` 后只移除 `--dry-run`，以相同参数执行一次。真实 promotion 的通过和失败
都会消费 `<state-root>/<gold-manifest-hash>.promotion.json`；失败会保留签名失败证明，但不能用于
production release，也不能删除 marker 后重试。私钥必须是当前用户拥有、非 symlink、精确 `0600`、
不超过 16 KiB 的 canonical unencrypted Ed25519 PKCS#8 PEM；当前 CLI 不直接调用 HSM 或
secret-manager signing API。`stage2-evaluator attest` 的必需选项是 `--payload`、
`--signing-key-file`、`--trust-manifest` 和 `--output`；它只用于已有独立一次性 evaluator 的
public-safe payload，本身不执行 hidden evaluation 或 marker 管理。

release builder 只接收 public-safe attestation，并将它放入完整 evidence index。candidate 与 output
具有同一 parent，evidence 和全部引用留在该 bundle root 内，trust manifest 位于 root 外。先验证，
再组装：

```sh
paper-agent --dry-run stage2-release assemble \
  --candidate /absolute/path/to/release-bundle/stage2-candidate-v2.json \
  --evidence /absolute/path/to/release-bundle/stage2-release-evidence.json \
  --trust-manifest /secure/deployment/hidden-evaluator-trust.json \
  --output /absolute/path/to/release-bundle/stage2-release.json
```

dry-run 重算相同 public gates、验证 hidden attestation 与路径边界，但不写 output；成功后只移除
`--dry-run` 执行一次真实 assembly。output 已存在时不会覆盖。private labels、raw submissions、
私钥和 marker state 永远不能进入 release bundle。

生产 release 使用 schema v3，并在加载时重算公共 gate evidence、验证 hidden evaluator 的
Ed25519 promotion attestation。先将版本化 asset root 内的
`configs/stage2/hidden-evaluator-trust.example.json` 复制到受保护的部署路径，替换其中退役示例
公钥为已审阅的 active evaluator 公钥；该文件不应放入 release bundle。再设置：

```sh
export PAPER_AGENT_STAGE2_HIDDEN_TRUST=/absolute/path/to/deployment/hidden-evaluator-trust.json
```

没有该变量（或 embedding application 明确传入等价 `hidden_trust_path`）时，生产 release 会
fail closed。该变量只供已组装 release 的 `doctor`、search/filter/workflow 等加载路径使用；
evaluator 与 assembler 使用各自显式的 `--trust-manifest`，不能从 release bundle 或环境隐式选择。
完整的密钥托管、dry-run、marker、attestation transfer、rotation 和 compromise 响应见
[Stage 2 hidden evaluator custody runbook](security/stage2-hidden-evaluator-custody.md)。

服务凭据只通过配置声明的环境变量提供，例如 `CROSSREF_MAILTO`、
`SEMANTIC_SCHOLAR_API_KEY`、`OPENALEX_API_KEY`、`NCBI_API_KEY`、`NCBI_EMAIL` 和
`UNPAYWALL_EMAIL`。不要把值写入 YAML、shell history、SQLite 导出、日志或报告。
先用最少来源和最小预算验证账号；provider 不可用必须在结果中显式显示，不能被静默跳过。

## 4. 冻结并批准检索

```sh
.venv/bin/paper-agent search plan \
  --input /absolute/path/to/paper-research/query-draft.yaml \
  --output-root /absolute/path/to/paper-research

.venv/bin/paper-agent search approve \
  --plan /absolute/path/to/paper-research/search/<plan-id>/QUERY_PLAN.draft.json \
  --hash <displayed-plan-hash> \
  --approved-by <operator>
```

`search plan --input` 接受研究问题、query variants、纳入范围与冻结 Stage 2 hashes 组成的
QueryPlan draft，不接受 v2 runtime config；同版本起点见仓库
`configs/query_draft.example.yaml`。检查 JSON 输出中的 resolved providers、请求/候选/时间预算和 content hash。范围、来源、
纳入条件、预算或配置发生变化时，重新生成并批准新计划；不要改写已批准计划。

在调用 provider 或模型前，可以做无写入的本地检查：

```sh
.venv/bin/paper-agent --dry-run search run \
  --plan /absolute/path/to/QUERY_PLAN.json \
  --database /absolute/path/to/papers.sqlite3 \
  --stage2-release /absolute/path/to/stage2-release.json
```

## 5. 离线验收与后续阶段

CI 在 lockfile bootstrap 之后使用固定 fixture、mock oMLX、mock Codex 和临时 SQLite，并在
测试进程中禁用外部 socket；测试不会下载模型、访问站点、使用 Codex 配额或浏览器登录。开发者可运行：

```sh
.venv/bin/python -m pytest --disable-socket --allow-unix-socket \
  tests/test_search_cli.py tests/test_phase7_productization.py
```

真实运行前，分别执行 `paper-agent filter`、`download`、`analyze`、`report` 和
`verify-report` 的 `--help`，只使用当前安装版本显示的参数。下载前创建并批准精确的
grant；全文及其受限派生物进入 Luna/Sol 前必须另有匹配的 processing grant。报告必须先
`--plan-only`、再批准 ReportPlan，且 deterministic verifier 与独立 Sol audit 都通过后
才能发布。完整故障处理、备份和恢复见 [operations.md](operations.md)。

## 6. 可恢复 typed workflow（可选）

当要把多个阶段串成一次可恢复运行时，创建 schema version 2 的 JSON workflow manifest，
而不是把 CLI argv 写入文件。Filter、Download 与 Analyze 的 `selection` 必须分别使用
`{"from_step":"search"}`、`{"from_step":"filter"}` 与 `{"from_step":"download"}`，Download 还要显式冻结
`include_needs_review`。version 1 仅用于单阶段清单。manifest 必须有 `schema_version`、
`workflow_id`、config FileRef 和 typed steps；每个 FileRef
只包含相对 manifest 目录的 `path` 与该文件的小写 SHA-256。最小 search step 的 JSON 和摘要生成
命令见根目录 [README](../README.md#typed-workflow-与恢复)。先运行：

若 Download 使用 collection/selection snapshot grant，在该 step 增加 `scope_snapshots`。每项写入
`snapshot_type`、`snapshot_id`、`snapshot_hash`、`collection_id`，首次运行还应以 `file` FileRef
固定原始 JSON；仅在 snapshot 已存在于同一 SQLite 时才可使用 `file: null`。workflow dry-run 会
只读重验这些绑定，不写入 snapshot、run 或 artifact。

typed workflow 当前只接受 public provider download grant；authorized-skill/digest-bound grant 必须
使用独立 `paper-agent download` 命令并显式提供 queue、output、至少一个 skill root 和原始 ZIP；
`--authorized-skill-audit` 只是对内置 audit manifest 的可选、已审阅覆盖。第一次 download 冻结
immutable CSV 并返回 `manual_required`；随后按匹配 digest 的 skill 完成 fixed browser pass、
`stage` 和 `audit`，最后以完全相同 run/grant/queue/output/root/ZIP/audit 参数重跑 download，CLI
才会从 final ledger 导入已验证 article。固定 pass 使用每篇 30 秒和 5 秒 jitter；CAPTCHA、403、
429、access denied 或缺少授权 PDF 链接必须立即停止。完整命令与恢复闭环见根目录
[README](../README.md#授权下载与报告执行)。静态 `paper_ids: []` 会被拒绝，绝不会回退到全局最新
Stage 2 选择。

```sh
paper-agent --dry-run run --workflow /absolute/path/to/workflow.json \
  --workflow-run-id <workflow-run-id>
```

然后使用同一个 `--workflow`、数据库和 workflow run ID 执行或恢复。`resume` 也要求
`--workflow`；它不会从一个裸 `--run-id` 推断输入。修改任何被引用文件后，旧 manifest 和旧 run
都不再可恢复，必须生成并批准新的冻结输入。

报告不能直接追加到这条动态链：ReportPlan 的 paper membership 只有在实际 corpus 和 Stage 4
结果产生后才能冻结。Analyze 完成后，先运行
`report prepare-inputs --workflow-run-id <completed-workflow-run-id>`，再用返回的 `handoff_id` 执行
plan-only 与人工批准；approve 同时接收固定了 approved plan path/hash 的新 config 和目标 workflow
manifest 路径，登记独立的单阶段 Report workflow。随后用返回的 manifest 与 report workflow run ID
调用普通 `run`/`resume`。manifest 同时冻结 prepare 使用的 artifact root；执行不会依赖 report
output root 与 analysis artifact root 恰好相同。这样不会为了恢复报告而修改原 workflow 的 FileRef，
也不会把一次人工批准伪装成无人值守执行。
