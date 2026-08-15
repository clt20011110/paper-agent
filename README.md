# Paper Agent Stage 1

Paper Agent Stage 1 的目标是：给定一个规范化的 venue ID 和一个年份，枚举该 venue-year 的正式论文，生成可审计的元数据结果。一次 CLI 运行只处理一个 venue-year；当前实现使用标准库运行时和包内 TOML venue catalog。

## 当前 catalog

会议：

- `icml` — International Conference on Machine Learning
- `aspdac` — Asia and South Pacific Design Automation Conference
- `dac` — Design Automation Conference
- `date` — Design, Automation and Test in Europe Conference
- `iccad` — IEEE/ACM International Conference on Computer-Aided Design
- `ispd` — International Symposium on Physical Design

期刊：

- `jssc` — IEEE Journal of Solid-State Circuits
- `tcad` — IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems
- `todaes` — ACM Transactions on Design Automation of Electronic Systems
- `tvlsi` — IEEE Transactions on Very Large Scale Integration (VLSI) Systems

## 安装与 CLI

要求 Python 3.11–3.13。安装项目和开发测试依赖：

```bash
uv sync --extra dev
```

查看帮助：

```bash
.venv/bin/paper-agent --help
```

收集一个 venue-year：

```bash
.venv/bin/paper-agent collect \
  --venue icml \
  --year 2024 \
  --output /absolute/path/to/icml-2024 \
  --contact researcher@example.org
```

## Artifacts 与 exit status

每次运行的输出目录包含三个正式 artifacts：

- `papers.jsonl`：通过完整论文记录门禁的论文。
- `issues.jsonl`：排除项、未闭合字段和运行诊断。
- `run.json`：该 venue-year 的状态、计数和运行摘要。

退出码如下：

- `0`：`complete` 或 `not_applicable`。
- `2`：输入或 catalog 参数无效。
- `3`：`partial`，结果未达到完整门禁。
- `4`：`failed` 或运行/发布错误。

## 离线测试

```bash
.venv/bin/pytest --disable-socket \
  --allow-unix-socket \
  --allow-hosts=localhost,127.0.0.1,::1
```

live smoke 测试不会被默认测试运行；只有显式设置 `PAPER_AGENT_RUN_LIVE_SMOKE=1` 时才启用。
