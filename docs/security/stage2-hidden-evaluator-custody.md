# Stage 2 hidden evaluator custody and promotion

Production Stage 2 release schema v3 has two independently verified inputs: public raw gate evidence and a signed statement from the team that holds the hidden labels. The signed statement does not carry labels, pair IDs, annotations, raw predictions, scores, or private marker state. Every statement has an explicit `release_role`: the primary is `winner`; an optional non-winner backup is `qualified_fallback` only after it independently passes every Phase 3 gate in the same sealed promotion batch.

## Sampling and annotation custody

Build the 600-pair set before creating authoritative gold labels. The complete natural frame marked inside the private snapshot does not require exhaustive pre-labeling. Under a frozen seed, draw 150 HIDDEN_REAL rows from that frame **without reading curated labels** and record their true inclusion probability, 150/N; then build DEV and HIDDEN_HARD from the curated pool of remaining paper families. The curated labels and difficulty flags are provisional sampling strata only, not gold. After all 600 pairs are selected, two annotators independently label every pair and a third adjudicates disagreements:

`curation-worklist` excludes each frozen HIDDEN_REAL pair and every row in its paper family. `curation-import` requires one private decision for every remaining worklist row and writes the existing strict curated-annotations artifact plus a hash-bound receipt. A decision source may be `human_provisional`, `model_provisional`, or `human_reviewed_model_suggestion`; none is a human gold label or may enter the later annotation ledger.

```sh
paper-agent --dry-run stage2-sampling freeze-frame \
  --private-snapshot /secure/evaluator/private-snapshot.json \
  --output /secure/evaluator/hidden-real-freeze-frame.json

paper-agent --dry-run stage2-sampling curation-worklist \
  --private-snapshot /secure/evaluator/private-snapshot.json \
  --hidden-real-freeze-frame /secure/evaluator/hidden-real-freeze-frame.json \
  --output /secure/evaluator/curation-worklist.json

# curation-decisions.json remains private and uses provisional, never-gold labels.
paper-agent --dry-run stage2-sampling curation-import \
  --private-snapshot /secure/evaluator/private-snapshot.json \
  --hidden-real-freeze-frame /secure/evaluator/hidden-real-freeze-frame.json \
  --worklist /secure/evaluator/curation-worklist.json \
  --decisions /secure/evaluator/curation-decisions.json \
  --curated-annotations-output /secure/evaluator/curated-annotations.json \
  --receipt-output /secure/evaluator/curation-receipt.json

paper-agent --dry-run stage2-sampling build \
  --private-snapshot /secure/evaluator/private-snapshot.json \
  --hidden-real-freeze-frame /secure/evaluator/hidden-real-freeze-frame.json \
  --curated-annotations /secure/evaluator/curated-annotations.json \
  --curation-receipt /secure/evaluator/curation-receipt.json \
  --gold-manifest-output /secure/evaluator-transfer/gold-manifest.json \
  --provenance-output /secure/evaluator/provenance.json
```

Only HIDDEN_REAL has an interpretable `sampling_probability=150/N`. DEV and HIDDEN_HARD are constrained by quotas, weights, and paper-family grouping, so their field is `null` and they must not enter inverse-probability metrics.

The label-free 600-pair gold manifest may enter the release bundle. The snapshot, curated annotations, sampling provenance (private in the current design), and raw annotation ledger remain in evaluator custody. `--private-labels` is exactly the labels for those 600 manifest pairs, never a full-snapshot label export.

Do not assemble the raw ledger by hand. Generate one blind worklist for each annotator; each person independently
fills the `label` fields. The files omit split membership, sampling strata, provisional labels, and hard flags:

```sh
paper-agent stage2-sampling annotation-worklist \
  --gold-manifest /secure/evaluator-transfer/gold-manifest.json \
  --private-snapshot /secure/evaluator/private-snapshot.json \
  --participant-id annotator-a \
  --output /secure/evaluator/annotation-a.json

paper-agent stage2-sampling annotation-worklist \
  --gold-manifest /secure/evaluator-transfer/gold-manifest.json \
  --private-snapshot /secure/evaluator/private-snapshot.json \
  --participant-id annotator-b \
  --output /secure/evaluator/annotation-b.json
```

After both are complete, create the third-person worklist. It contains only disagreements and does not show either
annotator's label. It binds the exact two completed inputs, so replacing either annotation after handoff is rejected.
The command refuses to proceed when pre-adjudication QWK is below 0.75:

```sh
paper-agent stage2-sampling adjudication-worklist \
  --gold-manifest /secure/evaluator-transfer/gold-manifest.json \
  --private-snapshot /secure/evaluator/private-snapshot.json \
  --annotation-a /secure/evaluator/annotation-a.json \
  --annotation-b /secure/evaluator/annotation-b.json \
  --participant-id adjudicator-c \
  --output /secure/evaluator/adjudication.json
```

Once the adjudicator fills every disagreement, assemble the ledger under evaluator custody:

```sh
paper-agent --dry-run stage2-sampling assemble-annotation-ledger \
  --gold-manifest /secure/evaluator-transfer/gold-manifest.json \
  --private-snapshot /secure/evaluator/private-snapshot.json \
  --curated-annotations /secure/evaluator/curated-annotations.json \
  --sampling-provenance /secure/evaluator/provenance.json \
  --annotation-a /secure/evaluator/annotation-a.json \
  --annotation-b /secure/evaluator/annotation-b.json \
  --adjudication /secure/evaluator/adjudication.json \
  --output /secure/evaluator/annotation-ledger.json
```

The sampling provenance binds the curated hard-flag candidates used to build the public manifest. Assembly retains a
candidate only when it is compatible with the final human label, then applies the existing final gold quotas
fail-closed. Participant IDs are custody labels, not identity authentication; the evaluator remains responsible for
ensuring that they name two independent humans and a distinct third person. Remove `--dry-run` only after validation
passes. Finally, derive the private promotion input inside the same custody boundary:

```sh
paper-agent --dry-run stage2-sampling finalize-annotations \
  --gold-manifest /secure/evaluator-transfer/gold-manifest.json \
  --annotation-ledger /secure/evaluator/annotation-ledger.json \
  --private-labels-output /secure/evaluator/private-gold-labels.json
```

Validation requires exactly two fixed annotators per pair, exactly one fixed third-person ruling for every disagreement, pre-adjudication quadratic-weighted kappa of at least 0.75, and all final gold quotas. The non-dry run creates the private label artifact without replacement; it omits annotator identities and raw annotation/adjudication rows.

The evaluator and the deployment verifier share only a reviewed Ed25519 public-key trust manifest. A release bundle cannot select or replace that trust root. The evaluator uses an explicit `--trust-manifest` for `stage2-evaluator attest`, `stage2-evaluator promote`, and `stage2-release assemble`. Commands that load an already assembled production release instead use the deployment environment variable `PAPER_AGENT_STAGE2_HIDDEN_TRUST`, or the equivalent `hidden_trust_path` Python argument. The environment variable does not replace the explicit evaluator or assembler option.

## Trust-root deployment

The shipped [`hidden-evaluator-trust.example.json`](../../configs/stage2/hidden-evaluator-trust.example.json) is deliberately unusable: its only key is a retired all-zero placeholder, it has no active key, and it contains no private key. Copy it to a protected deployment configuration location, replace its identity and key list with the evaluator's real Ed25519 public key, and activate only the intended key. Do not put the resulting manifest inside a release bundle, source checkout, artifact directory, or user-writable shared path.

The manifest is validated against `stage2-hidden-evaluator-trust.schema.json`. It must contain an active Ed25519 key whose purpose is `stage2-hidden-promotion`; its content hash is bound into each promotion attestation. Record the reviewed manifest hash, key ID, evaluator identity, activation time, and approver in the custody record.

## Private-key file contract

Paper Agent does not generate evaluator keys. Generate and retain the Ed25519 key with organization-approved tooling in the isolated evaluator environment. The private key must never enter a release bundle, Git repository, CI log, ticket, chat, test fixture, database export, or report artifact.

Both evaluator CLI commands accept only `--signing-key-file`. The file must be:

- a canonical, unencrypted PKCS#8 PEM containing one Ed25519 private key;
- a regular file, not a symbolic link;
- owned by the current effective user;
- exactly mode `0600`; and
- no larger than 16 KiB.

Set the mode before use and verify ownership in the isolated environment:

```sh
chmod 600 /secure/evaluator/hidden-promotion-key.pem
```

The current CLI cannot address an HSM or secret-manager signing API directly. A secret manager may materialize a short-lived file only if the organization permits it and the file satisfies the contract above. A hardware-backed or callback-based signer requires a separately reviewed integration; do not export an HSM key merely to make the file CLI work.

Transfer only the corresponding canonical padded-base64 32-byte public key to the deployment trust-manifest maintainer.

## DEV calibration custody

`stage2-calibration freeze-dev-scores` runs before opening labels. Its output freezes only the exact DEV 300-pair
reranker/Qwen raw scores and the topic-query/runtime/model provenance. Keep that file under evaluator custody: it is
an input to calibration, not a release artifact. After the verified human annotation workflow creates the complete
private label artifact, `stage2-calibration build-candidate` validates all 600 labels but joins only the DEV subset to
those raw scores. Neither raw scores nor private labels may be copied into the schema-v2 candidate directory.

The candidate directory is claimed without replacement. Its six dependency files are not a published candidate by
themselves; only the final `stage2-candidate-v2.json` leaf is the commit marker. If a failed build leaves a directory
without that marker, preserve it for diagnosis or move it aside before choosing a new output directory. Do not hand
assemble the missing marker.

## Preferred sealed promotion

`stage2-evaluator promote` is the preferred production path. It validates the public 600-pair sampling manifest and every schema-v2 benchmark candidate before opening private labels or submissions. Candidate, submission, and public-evidence mappings are repeatable `ID=PATH` arguments and must name exactly the same candidate IDs. Optional `--qualified-fallback-output ID=PATH` mappings name candidates for which the same sealed batch may emit an additional fallback attestation.

Each `--public-evidence` document uses `evidence_type: stage2_public_promotion_evidence`: it contains the same candidate binding, gold manifest, and raw public gate references as release evidence, but deliberately has no `hidden_attestation`. This lets the evaluator recompute quality, benchmark, and soak gates before the one-shot hidden comparison without a circular dependency.

First run the public-only validation. The paths for private labels, submissions, key, state, and output are still syntactically required, but dry-run does not read the private labels, submissions, or key, does not touch the state root or consume a marker, and does not create the output or its parent:

```sh
paper-agent --dry-run stage2-evaluator promote \
  --manifest /secure/evaluator/gold-manifest.json \
  --private-labels /secure/evaluator/private-labels.json \
  --candidate incumbent=/secure/evaluator/incumbent-candidate-v2.json \
  --candidate challenger=/secure/evaluator/challenger-candidate-v2.json \
  --candidate backup=/secure/evaluator/backup-candidate-v2.json \
  --submission incumbent=/secure/evaluator/incumbent-submission.json \
  --submission challenger=/secure/evaluator/challenger-submission.json \
  --submission backup=/secure/evaluator/backup-submission.json \
  --public-evidence incumbent=/secure/evaluator/incumbent-public-evidence.json \
  --public-evidence challenger=/secure/evaluator/challenger-public-evidence.json \
  --public-evidence backup=/secure/evaluator/backup-public-evidence.json \
  --incumbent-candidate-id incumbent \
  --evaluator-id evaluator-team-1 \
  --evaluation-run-id promotion-2026-08-11 \
  --state-root /secure/evaluator/state \
  --evaluator-key-id evaluator-key-2026-08 \
  --issued-at 2026-08-11T08:00:00Z \
  --trust-manifest /secure/deployment/hidden-evaluator-trust.json \
  --parity-oracle-trust /secure/deployment/parity-oracle-trust.json \
  --signing-key-file /secure/evaluator/hidden-promotion-key.pem \
  --output /secure/transfer/winner-attestation.json \
  --qualified-fallback-output backup=/secure/transfer/backup-attestation.json
```

Review the structured `status: "validated"` result and the frozen public inputs. Then remove only `--dry-run` and run the exact same command once. `--issued-at` must be an RFC 3339 timestamp. The optional `--bootstrap-iterations` and `--bootstrap-seed` default to `2000` and `0`; if overridden, record them before opening the holdout.

Public manifest/candidate errors, invalid trust, a missing or mismatched key, and a pre-existing output are rejected before private evaluation and marker consumption. Once a valid sealed evaluation completes, the evaluator atomically creates:

```text
<state-root>/<gold-manifest-hash>.promotion.json
```

That marker is the authoritative one-shot state. A passing or failing gate consumes the same holdout. The evaluator derives the candidate from the frozen paired hidden comparison and recomputed public quality/performance evidence; an operator cannot select a lower-ranked candidate. The main output always signs the unique `winner`. A requested non-winner is signed as `qualified_fallback` only when the winner is valid and that candidate independently passes every Phase 3 public, hidden, parity, performance, and soak gate. A requested fallback that becomes the winner or fails a gate gets no fallback file and is listed in `unqualified_fallback_candidate_ids`; the valid winner attestation is still published. Every attestation also carries the same `promotion_batch_hash`, committing the complete candidate set, exact candidate bytes, prediction submissions, public-gate artifact hashes, throughput runs, winner, policy, and one-shot marker. Assembly compares that signed commitment, so independently constructed attestations cannot imitate one sealed batch.

The signed winner binding includes the public gate artifacts and throughput, and must equal the primary release candidate. Release assembly independently recomputes the same gates. A failed winner still produces a signed public-safe failure attestation and the command reports `status: "failed"`; preserve both the attestation and marker for audit, but never assemble that result into a production release. If signing or output persistence fails after the marker was claimed, the marker can exist without a transferable attestation. Do not delete it or retry against the same holdout. Investigate the failure and use a new frozen holdout for a new promotion. Never infer reusability from terminal output alone; inspect the protected marker state.

## Advanced payload-only attestation

`stage2-evaluator attest` validates and signs an already prepared public-safe schema-v1 payload. It does not evaluate hidden data and does not create or consume a promotion marker. Use it only when a separately reviewed sealed evaluator has already enforced equivalent one-shot holdout custody; it is not a way to bypass `promote` marker semantics.

Validate the payload/trust/output binding without reading the private key or creating output:

```sh
paper-agent --dry-run stage2-evaluator attest \
  --payload /secure/transfer/hidden-promotion-signing-input.json \
  --signing-key-file /secure/evaluator/hidden-promotion-key.pem \
  --trust-manifest /secure/deployment/hidden-evaluator-trust.json \
  --output /secure/transfer/hidden-promotion-attestation.json
```

After review, remove only `--dry-run` and run the same command. The real command requires the private key to match the payload's active `evaluator_key_id` in the exact trust manifest. The output path must not already exist.

## Release assembly

Transfer only the signed public-safe attestations to the release builder. Build one final schema-v3 evidence index per candidate: the primary index contains the `winner` attestation and an optional backup index contains its `qualified_fallback` attestation. Each index binds the exact candidate bytes through `candidate_bundle_sha256`. Its rationale chain has seven FileRefs—manifest, worklist, records, source ledger, query metadata, derived examples, and papers—so the verifier can replay the deterministic selection from every topic-query×paper score, then verify typed Qwen source ledger → derived examples → human worklist/records. It also binds the gold-manifest commitment and all structured-replay, parity, benchmark, and soak raw evidence. Private labels, raw hidden submissions, the private key, and evaluator marker state must remain outside the release bundle.

The schema-v2 benchmark candidate and output must have the same parent directory. The evidence index and every referenced evidence artifact must stay inside that release-bundle root. The deployment trust manifest must be outside the root. Validate the complete assembly without writing output:

```sh
paper-agent --dry-run stage2-release assemble \
  --candidate /absolute/path/to/release-bundle/stage2-candidate-v2.json \
  --evidence /absolute/path/to/release-bundle/stage2-release-evidence.json \
  --fallback-candidate /absolute/path/to/release-bundle/backup-candidate-v2.json \
  --fallback-evidence /absolute/path/to/release-bundle/backup-release-evidence.json \
  --trust-manifest /secure/deployment/hidden-evaluator-trust.json \
  --parity-oracle-trust /secure/deployment/parity-oracle-trust.json \
  --output /absolute/path/to/release-bundle/stage2-release.json
```

Omit both fallback artifact options when no backup is deployed; they are all-or-none. Optional `--fallback-omlx-base-url` and `--fallback-api-key-env` bind different local deployment coordinates without changing the already evaluated candidate. Dry-run reads and verifies the same public evidence and hidden attestations as the real assembly, enforces the path and absent-output rules, and does not create the output or its parent. After reviewing the structured summary, remove only `--dry-run` and run the same command. Assembly never overwrites an existing destination.

Assembly recomputes all public gates, verifies the primary `winner` and backup `qualified_fallback` signatures, requires the same evaluation manifest, promotion gate policy, query/Qwen/shared runtime semantics, and a distinct reranker lock, then injects the fallback only into the final schema-v3 envelope. It never edits either schema-v2 candidate, which prevents a candidate/evidence hash cycle. The summary's `expected_query_plan.config_hash` binds the primary profile plus fallback candidate, evidence, runtime, and release binding; use that exact effective hash in QueryPlan approval.

## Production verification

A production bundle has `schema_version: "3"` and, without a backup, exactly these top-level fields: `schema_version`, `profile`, `reranker_lock`, `adjudicator_lock`, `calibration`, `release_gate`, and `runtime`. A release with a qualified backup adds `reranker_fallback`. Its `release_gate` has exactly:

```json
{
  "candidate_id": "<same as profile>",
  "candidate_bundle_sha256": "<exact schema-v2 candidate sha256>",
  "evaluation_manifest_hash": "<lowercase sha256>",
  "evidence": {
    "path": "stage2-release-evidence.json",
    "sha256": "<lowercase sha256>"
  }
}
```

Before a production command loads this completed release, set the deployment-controlled trust path:

```sh
export PAPER_AGENT_STAGE2_HIDDEN_TRUST=/secure/deployment/hidden-evaluator-trust.json
```

`load_stage2_release(..., hidden_trust_path=...)` is the equivalent Python API for an embedding application. If neither input is supplied, a v3 release fails closed. The loader again recomputes public gates and verifies the attestation; it does not trust a release or evidence `passed` field. A schema-v2 benchmark candidate remains useful before throughput gating, but production loading rejects it.

The signature proves that a currently trusted evaluator key attested to the exact frozen bindings. It does not protect against a stolen evaluator private key, compromise of the deployment trust root, or a modified verifier/wheel; those are separate security incidents.

## Rotation and compromise

For rotation, add the new public key as `active` in a newly reviewed deployment trust manifest, distribute that manifest atomically, then issue new attestations whose `trust_manifest_hash` matches it. Retain the old key as `retired` only for historical audit; it cannot sign a new promotion. A release attested under the old trust-manifest hash must be re-evaluated and re-signed before it can load against the new trust root.

On suspected private-key, evaluator-host, or trust-manifest compromise: remove the affected key from active use, replace the deployment trust manifest, invalidate pending promotions, and run hidden evaluation with a newly generated key and holdout. Preserve only non-sensitive hashes, timestamps, evaluator identity, marker hash, and incident references in audit records. Do not repair a signed release in place, delete a consumed marker, or disclose hidden data while investigating.
