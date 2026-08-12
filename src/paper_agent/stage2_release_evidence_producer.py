"""Write immutable schema-v3 Stage 2 release-evidence indexes."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any

from .schema import validate
from .stage2_release_evidence import (
    Stage2EvidenceError,
    load_stage2_release_evidence_index_bytes,
)
from .stage2_search import load_stage2_benchmark_candidate


class Stage2EvidenceProducerError(ValueError):
    """An evidence-index input cannot form a safe, bound release artifact."""


_PARITY_NAMES = (
    "manifest",
    "workload",
    "selection_receipt",
    "scores",
    "oracle_model_lock",
    "candidate_model_lock",
    "oracle_calibrator",
    "candidate_calibrator",
    "oracle_threshold",
    "candidate_threshold",
)


def write_stage2_release_evidence_index(
    *,
    output_path: Path,
    candidate_bundle_path: Path,
    gold_manifest_path: Path,
    structured_manifest_path: Path,
    structured_records_path: Path,
    structured_papers_path: Path,
    rationale_manifest_path: Path,
    rationale_worklist_path: Path,
    rationale_records_path: Path,
    rationale_source_ledger_path: Path,
    rationale_query_metadata_path: Path,
    rationale_derived_examples_path: Path,
    rationale_papers_path: Path,
    parity_manifest_path: Path,
    parity_workload_path: Path,
    parity_selection_receipt_path: Path,
    parity_scores_path: Path,
    parity_oracle_model_lock_path: Path,
    parity_candidate_model_lock_path: Path,
    parity_oracle_calibrator_path: Path,
    parity_candidate_calibrator_path: Path,
    parity_oracle_threshold_path: Path,
    parity_candidate_threshold_path: Path,
    benchmark_manifest_path: Path,
    benchmark_papers_path: Path,
    benchmark_record_paths: tuple[Path, Path, Path, Path, Path, Path],
    soak_manifest_path: Path,
    soak_papers_path: Path,
    soak_record_path: Path,
    hidden_attestation_path: Path | None = None,
) -> Path:
    """Write one public-promotion or final-release evidence index, never replacing it.

    All referenced files must already exist beneath ``output_path.parent``.
    Candidate identity and hashes are always derived from the frozen v2 bundle.
    Providing ``hidden_attestation_path`` selects final-release mode; omitting it
    selects public-promotion mode.
    """

    payload = build_stage2_release_evidence_index_bytes(
        output_path=output_path,
        candidate_bundle_path=candidate_bundle_path,
        gold_manifest_path=gold_manifest_path,
        structured_manifest_path=structured_manifest_path,
        structured_records_path=structured_records_path,
        structured_papers_path=structured_papers_path,
        rationale_manifest_path=rationale_manifest_path,
        rationale_worklist_path=rationale_worklist_path,
        rationale_records_path=rationale_records_path,
        rationale_source_ledger_path=rationale_source_ledger_path,
        rationale_query_metadata_path=rationale_query_metadata_path,
        rationale_derived_examples_path=rationale_derived_examples_path,
        rationale_papers_path=rationale_papers_path,
        parity_manifest_path=parity_manifest_path,
        parity_workload_path=parity_workload_path,
        parity_selection_receipt_path=parity_selection_receipt_path,
        parity_scores_path=parity_scores_path,
        parity_oracle_model_lock_path=parity_oracle_model_lock_path,
        parity_candidate_model_lock_path=parity_candidate_model_lock_path,
        parity_oracle_calibrator_path=parity_oracle_calibrator_path,
        parity_candidate_calibrator_path=parity_candidate_calibrator_path,
        parity_oracle_threshold_path=parity_oracle_threshold_path,
        parity_candidate_threshold_path=parity_candidate_threshold_path,
        benchmark_manifest_path=benchmark_manifest_path,
        benchmark_papers_path=benchmark_papers_path,
        benchmark_record_paths=benchmark_record_paths,
        soak_manifest_path=soak_manifest_path,
        soak_papers_path=soak_papers_path,
        soak_record_path=soak_record_path,
        hidden_attestation_path=hidden_attestation_path,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return output_path


def build_stage2_release_evidence_index_bytes(
    *,
    output_path: Path,
    candidate_bundle_path: Path,
    gold_manifest_path: Path,
    structured_manifest_path: Path,
    structured_records_path: Path,
    structured_papers_path: Path,
    rationale_manifest_path: Path,
    rationale_worklist_path: Path,
    rationale_records_path: Path,
    rationale_source_ledger_path: Path,
    rationale_query_metadata_path: Path,
    rationale_derived_examples_path: Path,
    rationale_papers_path: Path,
    parity_manifest_path: Path,
    parity_workload_path: Path,
    parity_selection_receipt_path: Path,
    parity_scores_path: Path,
    parity_oracle_model_lock_path: Path,
    parity_candidate_model_lock_path: Path,
    parity_oracle_calibrator_path: Path,
    parity_candidate_calibrator_path: Path,
    parity_oracle_threshold_path: Path,
    parity_candidate_threshold_path: Path,
    benchmark_manifest_path: Path,
    benchmark_papers_path: Path,
    benchmark_record_paths: tuple[Path, Path, Path, Path, Path, Path],
    soak_manifest_path: Path,
    soak_papers_path: Path,
    soak_record_path: Path,
    hidden_attestation_path: Path | None = None,
) -> bytes:
    """Build and fully validate immutable release-evidence bytes without writing."""

    if os.path.lexists(output_path):
        raise FileExistsError(f"Stage 2 release evidence output already exists: {output_path}")
    release = load_stage2_benchmark_candidate(candidate_bundle_path)
    profile = release.profile
    reranker = profile.reranker_calibration
    qwen = profile.adjudicator_calibration
    if reranker is None or qwen is None:
        raise Stage2EvidenceProducerError("v2 candidate bundle has no probability calibrations")

    root = output_path.parent.resolve()
    refs = {
        "gold_manifest": _artifact_ref(gold_manifest_path, root),
        "structured_manifest": _artifact_ref(structured_manifest_path, root),
        "structured_records": _artifact_ref(structured_records_path, root),
        "structured_papers": _artifact_ref(structured_papers_path, root),
        "rationale_manifest": _artifact_ref(rationale_manifest_path, root),
        "rationale_worklist": _artifact_ref(rationale_worklist_path, root),
        "rationale_records": _artifact_ref(rationale_records_path, root),
        "rationale_source_ledger": _artifact_ref(rationale_source_ledger_path, root),
        "rationale_query_metadata": _artifact_ref(rationale_query_metadata_path, root),
        "rationale_derived_examples": _artifact_ref(rationale_derived_examples_path, root),
        "rationale_papers": _artifact_ref(rationale_papers_path, root),
        "benchmark_manifest": _artifact_ref(benchmark_manifest_path, root),
        "benchmark_papers": _artifact_ref(benchmark_papers_path, root),
        "soak_manifest": _artifact_ref(soak_manifest_path, root),
        "soak_papers": _artifact_ref(soak_papers_path, root),
        "soak_record": _artifact_ref(soak_record_path, root),
    }
    parity_paths = (
        parity_manifest_path, parity_workload_path, parity_selection_receipt_path,
        parity_scores_path, parity_oracle_model_lock_path,
        parity_candidate_model_lock_path, parity_oracle_calibrator_path,
        parity_candidate_calibrator_path, parity_oracle_threshold_path,
        parity_candidate_threshold_path,
    )
    parity_refs = {
        name: _artifact_ref(path, root) for name, path in zip(_PARITY_NAMES, parity_paths, strict=True)
    }
    benchmark_records = [_artifact_ref(path, root) for path in benchmark_record_paths]
    if len(benchmark_records) != 6:
        raise Stage2EvidenceProducerError("benchmark requires exactly six record paths")

    evaluation_manifest_hash = _gold_manifest_hash(refs["gold_manifest"], root)
    if any(
        binding.calibrator.gold_manifest_hash != evaluation_manifest_hash
        for binding in (reranker, qwen)
    ):
        raise Stage2EvidenceProducerError("candidate calibrators do not bind the supplied gold manifest")
    document: dict[str, Any] = {
        "schema_version": "3",
        "evidence_type": (
            "stage2_release_evidence" if hidden_attestation_path is not None
            else "stage2_public_promotion_evidence"
        ),
        "candidate_id": release.profile_name,
        "candidate_bundle_sha256": release.release_hash,
        "evaluation_manifest_hash": evaluation_manifest_hash,
        "stage2_config_hash": profile.base_runtime_config_hash,
        "model_lock_hashes": {"reranker": profile.reranker_lock_hash, "qwen": profile.adjudicator_lock_hash},
        "calibrator_hashes": {"reranker": reranker.calibrator.hash(), "qwen": qwen.calibrator.hash()},
        "threshold_hashes": {"reranker": reranker.threshold.hash(), "qwen": qwen.threshold.hash()},
        "gold_manifest": refs["gold_manifest"],
        "public_gates": {
            "structured_replay": {
                "manifest": refs["structured_manifest"],
                "records": refs["structured_records"],
                "papers": refs["structured_papers"],
            },
            "rationale": {
                "manifest": refs["rationale_manifest"],
                "worklist": refs["rationale_worklist"],
                "records": refs["rationale_records"],
                "source_ledger": refs["rationale_source_ledger"],
                "query_metadata": refs["rationale_query_metadata"],
                "derived_examples": refs["rationale_derived_examples"],
                "papers": refs["rationale_papers"],
            },
            "parity": parity_refs,
            "benchmark": {"manifest": refs["benchmark_manifest"], "papers": refs["benchmark_papers"], "records": benchmark_records},
            "soak": {"manifest": refs["soak_manifest"], "papers": refs["soak_papers"], "record": refs["soak_record"]},
        },
    }
    if hidden_attestation_path is not None:
        document["hidden_attestation"] = _artifact_ref(hidden_attestation_path, root)

    validate(document, "stage2-release-evidence.schema.json")
    payload = (json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    try:
        load_stage2_release_evidence_index_bytes(output_path.resolve(), payload)
    except (OSError, Stage2EvidenceError, ValueError) as error:
        raise Stage2EvidenceProducerError(f"release evidence inputs are invalid: {error}") from error
    return payload


def _artifact_ref(path: Path, root: Path) -> dict[str, str]:
    try:
        resolved = path.resolve(strict=True)
        relative = resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise Stage2EvidenceProducerError(
            f"evidence artifact must exist inside the output bundle: {path}"
        ) from error
    if not resolved.is_file():
        raise Stage2EvidenceProducerError(f"evidence artifact is not a file: {path}")
    return {"path": relative.as_posix(), "sha256": sha256(resolved.read_bytes()).hexdigest()}


def _gold_manifest_hash(ref: dict[str, str], root: Path) -> str:
    from .stage2_evaluation import gold_manifest_from_document

    try:
        manifest = gold_manifest_from_document(json.loads((root / ref["path"]).read_bytes()))
        manifest.validate_sampling_structure()
        return manifest.hash()
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise Stage2EvidenceProducerError(f"gold manifest is invalid: {error}") from error
