# 20-Venue Controlled E2E Acceptance — 2026-08-11

## Result

All 20 venue descriptors in `venues/` passed the controlled Stage 1 through
Stage 4b acceptance matrix.

- 19 venues were executed in the current isolated matrix: one selected paper,
  one real public PDF, one full-PDF Luna analysis, and exactly one one-shot Sol
  synthesis per venue.
- NeurIPS reuses the hash-pinned 2025 molecular-generation acceptance run:
  15 real public PDFs, 15 full-PDF Luna analyses, and exactly one one-shot Sol
  synthesis over all 15 Luna reports.
- Every current report passed the current deterministic content verifier,
  immutable manifest/hash checks, Stage 3 PDF-to-Stage 4 lineage checks, exact
  Stage 4 corpus-to-Stage 4b binding, and the strict legacy audit-order proof
  described below.
- Every Stage 4b run has `dispatch_count=1`, one Sol invocation-ledger row, and
  zero reduce-tree, audit, audit-shard, repair, or retry calls.

## Qualification

This is a controlled functional E2E, not a live-crawler acceptance for every
external provider. The 19 current runs replay one approved provider-shaped
Stage 1 record through the native adapter and persistence path, then use a
deterministic `TEST_ONLY` Stage 2 selector. They therefore do not prove live
provider transport or the production Stage 2 model release gate.

Stage 3 is not mocked: all 19 current runs downloaded and validated actual
public PDF bytes through `public_direct`. Official OA URLs were preferred;
legal author or institutional-repository copies were used where appropriate.
No authorized-browser paper-download workflow was used. Stage 4 and Stage 4b
are also real model executions using `gpt-5.6-luna` and `gpt-5.6-sol`.

The NeurIPS row is explicitly marked as reused historical evidence. Its
acceptance JSON and committed report are independently hash-pinned by
`configs/e2e/venue-e2e-acceptance-imports.json`; it is not represented as a
current-revision re-execution.

Seven current-runtime reports (AAAI, ACL, Cell, CVPR, ICLR, ICML, and Nature
Chemistry) crossed the single Sol boundary before the one-shot implementation
marker was raised from `stage4b-one-shot-v1` to `stage4b-one-shot-v2` during
this acceptance campaign. Their frozen v1 plans were not rewritten and Sol was
not called again. Their final immutable artifacts were rechecked with the
current complete verifier. The other 12 current-runtime reports use v2 plans.
Because all 19 reports predate the claim-order fix below, the evidence labels
those 12 rows `legacy_v2_audit_order_reverified`. The checked-in producer is
now `stage4b-one-shot-v3`, so the old audit serialization contract cannot be
mistaken for current behavior.

The campaign also exposed and fixed a producer/serializer mismatch shared by
all 19 current runs: the audit originally hashed claims in runtime Sol order,
while `CLAIMS_EVIDENCE.jsonl` persisted the same claims in `claim_id` order.
The checked-in implementation now uses `claim_id` order for both hashing and
persistence. The existing immutable bundles were not rewritten. Instead, the
acceptance verifier proves `legacy_runtime_claim_order_verified` by binding the
exact `report_audit_runs` database row, its base/current/audit hashes, and the
saved runtime bundle to the published bundle, permitting only claim ordering
and the derived `coverage.complete` representation to differ.

## Cases

Hash columns are 12-character prefixes of the full SHA-256 values preserved in
the strict machine-readable runtime matrix.

| Venue | Topic | Selected paper | PDF SHA | Luna output SHA | Sol output SHA | Result |
| --- | --- | --- | --- | --- | --- | --- |
| AAAI | 蛋白口袋条件的三维分子设计 | BindGPT: A Scalable Framework for 3D Molecular Design via Language Modeling and Reinforcement Learning | `082b8ccd1009` | `82d2893154c5` | `29b0529afc52` | PASS |
| ACL | LLM 分子结构解析 | Boosting LLM’s Molecular Structure Elucidation with Knowledge Enhanced Tree Search Reasoning | `f783a664c48a` | `c8cec2402be1` | `bfa3a742ac23` | PASS |
| Cell | 果蝇突触位点电子显微镜神经递质分类 | Neurotransmitter classification from electron microscopy images at synaptic sites in Drosophila melanogaster | `59a7e1e8054f` | `44f5cfc109c1` | `6da4bb8e1e96` | PASS |
| CVPR | 分子弱监督伪标注 | Molecular Data Programming: Towards Molecule Pseudo-labeling with Systematic Weak Supervision | `aa5d82ba2c17` | `2ed8dbe305a7` | `bb4c8aa7a654` | PASS |
| DAC | 端侧 LLM 个性化 | Enabling On-Device Large Language Model Personalization with Self-Supervised Data Selection and Synthesis | `8a357551eca7` | `5d6b8bafa61e` | `7b483da10d7c` | PASS |
| ICCAD | 无注意力 LLM 的 PIM 加速 | Towards Floating Point-Based Attention-Free LLM: Hybrid PIM with Non-Uniform Data Format and Reduced Multiplications | `ad482a7c9d91` | `62761ec2ec89` | `989a648e94a8` | PASS |
| ICCV | 化学结构图片识别 | MolParser: End-to-end Visual Recognition of Molecule Structures in the Wild | `eb6e2fb8c1ba` | `e110d2823795` | `e692d0da6db4` | PASS |
| ICLR | 统一分子生成与性质预测 | UniGEM: A Unified Approach to Generation and Property Prediction for Molecules | `f5cb5024bfba` | `81188dd1d46a` | `ec2247ad83c7` | PASS |
| ICML | 分子生成与可解释图语法 | Representing Molecules as Random Walks Over Interpretable Grammars | `974d56038f70` | `892eeb8b3bc1` | `8d752c0bcbcc` | PASS |
| IJCAI | 条件生成模型检索 | CGI: Identifying Conditional Generative Models with Example Images | `2f435b704f41` | `5ad4e4b0ac98` | `1731d8448ec8` | PASS |
| Nature Biomedical Engineering | 大基因整合的 prime editing 与重组酶 | Efficient site-specific integration of large genes in mammalian cells via continuously evolved recombinases and prime editing | `df4f126177a7` | `53d97284e56b` | `2beb4004a915` | PASS |
| Nature Biotechnology | 纳米载体全鼠单细胞深度学习成像 | Nanocarrier imaging at single-cell resolution across entire mouse bodies with deep learning | `ef36e734ef58` | `d5eaba035d04` | `f31186fcc2a5` | PASS |
| Nature Catalysis | 钙钛矿太阳能 CO2 C2 烃合成 | Perovskite-driven solar C2 hydrocarbon synthesis from CO2 | `79938a9db44c` | `84874a25c1fa` | `b6225ec30029` | PASS |
| Nature Chemistry | 三金属铋(I)烯丙基阳离子 | A trimetallic bismuth(I)-based allyl cation | `f378e746d29f` | `aefb197be4bb` | `5a408c03f585` | PASS |
| Nature Communications | GNN 分子生成器 | Using GNN property predictors as molecule generators | `b88d3e3ec29c` | `75e447f1c61c` | `305002d6eba7` | PASS |
| Nature Computational Science | 复杂系统深度主动优化 | Deep active optimization for complex systems | `bdc2355e554a` | `28121a75c6ff` | `1b26462aa53c` | PASS |
| Nature Machine Intelligence | LLM 置信度校准 | What large language models know and what people think they know | `048824333721` | `12a3187c2285` | `d172af9998d3` | PASS |
| NeurIPS | 分子生成与分子设计 | 15-paper 2025 molecular-generation corpus | historical | historical | `6cf6799196ae` | PASS |
| Science | 蛋白质宇宙的结构域百科 | Exploring structural diversity across the protein universe with The Encyclopedia of Domains | `a4661b74deaa` | `4a61bd567807` | `80daaf602bf6` | PASS |
| TCAD | LLM 驱动 EDA Agent | ChatEDA: A Large Language Model Powered Autonomous Agent for EDA | `2d7dc713cb57` | `c56c4b05ba25` | `8016741b740c` | PASS |

## Recovery disclosure

ICML, Cell, and CVPR each retained their original single Sol invocation and raw
output hash while deterministic local normalization/republication repaired a
post-dispatch validation issue. No model was called again. The remaining 16
current venue runs completed directly. The NeurIPS historical evidence records
its own previously disclosed local post-dispatch recovery and zero additional
model calls. Separately, all 19 current bundles use the read-only legacy audit
ordering proof above; this did not rewrite a report or dispatch a model.

## Reproduction and verification

The controlled case definitions are frozen in
`configs/e2e/venue-smoke-matrix.yaml`. The read-only verifier is:

```bash
python scripts/summarize_venue_e2e_matrix.py \
  --run-root "$VENUE_E2E_RUN_ROOT" \
  --venue-catalog-root venues \
  --acceptance-import-manifest configs/e2e/venue-e2e-acceptance-imports.json
```

The portable machine-readable result is
`docs/acceptance/venue-e2e-matrix-20260811-evidence.json` (raw SHA-256
`bdbae4d2f8f119c9d8942806ace659355b80e80e2f9a7643e81795fa7b56c628`).
It replaces local checkout and run-root prefixes with `$REPOSITORY_ROOT` and
`$RUN_ROOT`; regenerate that form with `--portable-paths`. The evidence also
binds verifier version `paper-agent.venue-e2e-verifier.v2` and verifier script
SHA-256 `8009bf61ca3e62c6cfe12b3a71ffeb0e1fb16615771249d89adcc17bf24ff854`.

This JSON is a hash-rich attestation, not a self-contained replay bundle: the
19 current databases, CAS directories, PDFs, and report directories remain in
the external `$RUN_ROOT` and are not checked into Git. Independent full
revalidation therefore requires that original run root; the repository alone
can verify the attestation structure, verifier hash, imported NeurIPS evidence,
and the committed hashes, but cannot reconstruct the 19 external runs.

The verifier fails closed on missing or substituted run IDs, PDF CAS bytes,
non-`full_pdf` Stage 4 inputs, output-artifact drift, duplicate Sol calls,
reduce/audit/repair calls, report-manifest drift, saved-verification drift, or
incomplete venue-catalog coverage. The legacy audit path additionally fails on
database bundle, base/current hash, report-run, claim content/set, coverage,
bibliography, or any other audited-input drift.
