# 20-Venue Current-Run Controlled E2E Acceptance — 2026-08-12

## Result

All 20 venue descriptors in `venues/` passed a current-run controlled Stage 1
through Stage 4b acceptance flow. No venue row is imported from historical
evidence.

- Each venue selected one paper for one frozen topic.
- Stage 3 downloaded and validated one real public PDF per venue.
- Each final bound run dispatched `gpt-5.6-luna` exactly once over the full PDF.
- Each final bound report dispatched `gpt-5.6-sol` exactly once with the
  one-shot strategy and all Luna output visible in that single prompt.
- All 20 reports use `stage4b-one-shot-v3` / `current_v3`, pass the current
  deterministic verifier, and use canonical `claim_id` audit ordering.
- Across the 20 final bound reports, reduce, audit, audit-shard, repair, and
  retry dispatch counts are all zero.

The portable machine-readable result is
`venue-e2e-matrix-20260812-evidence.json` (SHA-256
`0a90def35b343466cd403bc3e62ec6a4c56ecdd70ae7db136d6f8f6249c10b13`).

## Qualification

This is a controlled functional E2E, not a live-network acceptance for every
external provider. Stage 1 replays one approved provider-shaped record through
the native adapter and SQLite persistence path. Stage 2 uses the deterministic
`TEST_ONLY` selector and does not invoke oMLX, so this matrix does not satisfy
the production Stage 2 release gate.

Stage 3 is not mocked. Every case used actual public PDF bytes through the
`public_direct` path and passed PDF/CAS validation. Official open-access URLs
were preferred; legal author or institutional-repository copies were used when
appropriate. The authorized-browser download skill was not needed. Stage 4 and
Stage 4b are real model executions using `gpt-5.6-luna` and `gpt-5.6-sol`.

The acceptance import manifest is intentionally empty:
`configs/e2e/venue-e2e-acceptance-imports-20260812-current.json`. The verifier
therefore fails if any catalog venue is missing, duplicated, unexpected, or
silently replaced by a historical attestation.

## Cases

Hash columns are 12-character prefixes of the full SHA-256 values stored in the
machine-readable evidence.

| Venue | Topic | Selected paper | PDF SHA | Luna SHA | Final Sol SHA | Result |
| --- | --- | --- | --- | --- | --- | --- |
| AAAI | 蛋白口袋条件的三维分子设计 | BindGPT: A Scalable Framework for 3D Molecular Design via Language Modeling and Reinforcement Learning | `082b8ccd1009` | `356e6552ff5c` | `162e7f735340` | PASS |
| ACL | LLM 分子结构解析 | Boosting LLM’s Molecular Structure Elucidation with Knowledge Enhanced Tree Search Reasoning | `f783a664c48a` | `d972082943cd` | `71504d43eb58` | PASS |
| Cell | 果蝇突触位点电子显微镜神经递质分类 | Neurotransmitter classification from electron microscopy images at synaptic sites in Drosophila melanogaster | `59a7e1e8054f` | `2473ee019462` | `012fdabaf2e3` | PASS |
| CVPR | 分子弱监督伪标注 | Molecular Data Programming: Towards Molecule Pseudo-labeling with Systematic Weak Supervision | `aa5d82ba2c17` | `ca99ec338627` | `f39b27cebe9d` | PASS |
| DAC | 端侧 LLM 个性化 | Enabling On-Device Large Language Model Personalization with Self-Supervised Data Selection and Synthesis | `8a357551eca7` | `ebffb0d93335` | `c4c94d29eb58` | PASS |
| ICCAD | 无注意力 LLM 的 PIM 加速 | Towards Floating Point-Based Attention-Free LLM: Hybrid PIM with Non-Uniform Data Format and Reduced Multiplications | `ad482a7c9d91` | `20642cdbc2a5` | `b69b95208a09` | PASS |
| ICCV | 化学结构图片识别 | MolParser: End-to-end Visual Recognition of Molecule Structures in the Wild | `eb6e2fb8c1ba` | `82c2aabf494e` | `c651ccd7ec77` | PASS |
| ICLR | 统一分子生成与性质预测 | UniGEM: A Unified Approach to Generation and Property Prediction for Molecules | `f5cb5024bfba` | `6497234a1884` | `2edda049c8e6` | PASS |
| ICML | 分子生成与可解释图语法 | Representing Molecules as Random Walks Over Interpretable Grammars | `974d56038f70` | `0ab48535b4d9` | `d9c468d68bbe` | PASS |
| IJCAI | 条件生成模型检索 | CGI: Identifying Conditional Generative Models with Example Images | `2f435b704f41` | `d70aed2944d4` | `d951835c78d8` | PASS |
| Nature Biomedical Engineering | 大基因整合的 prime editing 与重组酶 | Efficient site-specific integration of large genes in mammalian cells via continuously evolved recombinases and prime editing | `df4f126177a7` | `60bdd06fb225` | `bf6d6a96651c` | PASS |
| Nature Biotechnology | 纳米载体全鼠单细胞深度学习成像 | Nanocarrier imaging at single-cell resolution across entire mouse bodies with deep learning | `ef36e734ef58` | `ba6220d14eb6` | `f2502e0eff2f` | PASS |
| Nature Catalysis | 钙钛矿太阳能 CO2 C2 烃合成 | Perovskite-driven solar C2 hydrocarbon synthesis from CO2 | `79938a9db44c` | `a2033a8fa493` | `834c0e9cf59a` | PASS |
| Nature Chemistry | 三金属铋(I)烯丙基阳离子 | A trimetallic bismuth(I)-based allyl cation | `f378e746d29f` | `496467dc097f` | `eec46a9c2857` | PASS |
| Nature Communications | GNN 分子生成器 | Using GNN property predictors as molecule generators | `b88d3e3ec29c` | `e7563d49964b` | `0d30e1c2e0db` | PASS |
| Nature Computational Science | 复杂系统深度主动优化 | Deep active optimization for complex systems | `bdc2355e554a` | `e0c3ce2163fe` | `7436a71312ee` | PASS |
| Nature Machine Intelligence | LLM 置信度校准 | What large language models know and what people think they know | `048824333721` | `da1acef2a410` | `ea99d7c0d953` | PASS |
| NeurIPS | 低成本模板化分子生成 | Scalable and Cost-Efficient de Novo Template-Based Molecular Generation | `cf6d51c43171` | `8872d85ca973` | `08ba6bcef44f` | PASS |
| Science | 蛋白质宇宙的结构域百科 | Exploring structural diversity across the protein universe with The Encyclopedia of Domains | `a4661b74deaa` | `1839129e3649` | `d74357a483fa` | PASS |
| TCAD | LLM 驱动 EDA Agent | ChatEDA: A Large Language Model Powered Autonomous Agent for EDA | `2d7dc713cb57` | `6a1106610cf5` | `6585f8828344` | PASS |

## Recovery and cost disclosure

The final matrix counts only the model calls bound to each accepted run. The
whole campaign, including retained failed or preflight attempts, used 21 Luna
calls and 22 Sol calls:

- The first ICML preflight reached Luna but stopped before Sol because its
  bibliography identity was incomplete. The corrected `v2` run made one Luna
  and one Sol call and is the only ICML run in the acceptance root.
- Cell and Nature Chemistry each exposed an impossible ReportPlan: their Stage
  4 output contained no directional evidence, but the planner still marked the
  paper as evidence. Each first report attempt consumed one diagnostic Sol
  call. After the planner fix, a new `report-v2` reused the existing Luna output
  and called Sol once. The matrix binds only those final `report-v2` runs, while
  both earlier Sol ledger rows remain preserved in their databases.
- IJCAI accepted the already-cited canonical publication year after a local
  verifier fix. Nature Machine Intelligence received deterministic
  section-local claim/reference normalization. Nature Biotechnology received a
  deterministic per-block incomparability disclosure. These three recoveries
  reused the original raw Sol output and did not dispatch another model call.

No failed output was rewritten or rebound to a different approved plan.

## Reproduction and verification

The case definitions are frozen in `configs/e2e/venue-smoke-matrix.yaml`. Run
the read-only verifier against the retained current-run root with:

```bash
python scripts/summarize_venue_e2e_matrix.py \
  --run-root "$VENUE_E2E_RUN_ROOT" \
  --venue-catalog-root venues \
  --acceptance-import-manifest \
    configs/e2e/venue-e2e-acceptance-imports-20260812-current.json \
  --portable-paths \
  --json-output docs/acceptance/venue-e2e-matrix-20260812-evidence.json
```

The evidence binds verifier version `paper-agent.venue-e2e-verifier.v2` and
verifier script SHA-256
`8009bf61ca3e62c6cfe12b3a71ffeb0e1fb16615771249d89adcc17bf24ff854`.
It replaces local checkout and run-root prefixes with `$REPOSITORY_ROOT` and
`$RUN_ROOT` and contains no user-specific absolute path.

The JSON is a hash-rich attestation, not a self-contained replay bundle. The
20 SQLite databases, CAS directories, PDFs, Luna analyses, and report bundles
remain in the external run root and are not committed to Git. Full independent
revalidation therefore requires that retained root. The verifier fails closed
on run-binding drift, missing or altered PDF/CAS bytes, non-full-PDF Luna input,
duplicate final Sol calls, any reduce/audit/repair dispatch, report-manifest or
saved-verification drift, and incomplete venue-catalog coverage.
