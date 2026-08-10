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

从源码 checkout 安装会保留 `example_config.yaml`、`configs/` 和
`skills/paper-agent/`。若只安装 wheel，请另外取得同一 release 的配置模板和
Codex skill；不要从另一个版本的仓库混用 YAML、prompt、schema 或 policy。

安装 Codex skill 时，将整个 `skills/paper-agent/` 目录复制或链接到个人 Codex
skills 目录。skill 只调用已安装的 `paper-agent`，不携带第二套实现。

## 2. 选择配置并运行 doctor

从窄范围模板开始，复制到自己的受保护项目目录；不要原地修改仓库示例。不要只复制一个
YAML：模板还引用同一 release 的 venue/provider manifests、schema、prompt、policy 和 registry。
源码 release 可把下面的只读运行时资源一并复制；wheel 用户应从同一 release artifact 取得
等价模板/资源包。

```sh
mkdir -p /absolute/path/to/paper-research
cp configs/smoke_supported.yaml /absolute/path/to/paper-research/research.yaml
cp -R acceptance policies prompts providers registries schemas venues \
  /absolute/path/to/paper-research/
.venv/bin/paper-agent --config /absolute/path/to/paper-research/research.yaml doctor
```

配置中的输出、SQLite、模型验收产物等相对路径以 `research.yaml` 所在目录为根；上述只读
资源必须保持相同目录关系。也可以把配置留在完整 checkout 中，只把 output/database 路径改成
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

当要把多个阶段串成一次可恢复运行时，创建 JSON workflow manifest，而不是把 CLI argv 写入
文件。manifest 必须有 `schema_version`、`workflow_id`、config FileRef 和 typed steps；每个 FileRef
只包含相对 manifest 目录的 `path` 与该文件的小写 SHA-256。最小 search step 的 JSON 和摘要生成
命令见根目录 [README](../README.md#typed-workflow-与恢复)。先运行：

```sh
paper-agent --dry-run run --workflow /absolute/path/to/workflow.json \
  --workflow-run-id <workflow-run-id>
```

然后使用同一个 `--workflow`、数据库和 workflow run ID 执行或恢复。`resume` 也要求
`--workflow`；它不会从一个裸 `--run-id` 推断输入。修改任何被引用文件后，旧 manifest 和旧 run
都不再可恢复，必须生成并批准新的冻结输入。
