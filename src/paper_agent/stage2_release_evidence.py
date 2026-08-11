"""Versioned file bindings for independently verifiable Stage 2 evidence."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .canonical import content_hash
from .schema import validate
from .stage2_evaluation import (
    GoldManifest,
    GoldSplit,
    gold_manifest_from_document,
    pair_universe_hash,
)


PUBLIC_GATE_NAMES = (
    "structured_replay",
    "rationale",
    "parity",
    "benchmark",
    "soak",
)
STAGE2_PATH_NAMES = frozenset({"reranker", "qwen"})


class Stage2EvidenceError(ValueError):
    """A Stage 2 evidence index or one of its bound files is invalid."""


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    path: str
    sha256: str

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> "ArtifactRef":
        return cls(str(document["path"]), str(document["sha256"]))

    def resolve(self, bundle_root: Path) -> Path:
        relative = Path(self.path)
        if relative.is_absolute() or ".." in relative.parts:
            raise Stage2EvidenceError("Stage 2 evidence paths must stay inside the bundle")
        root = bundle_root.resolve()
        resolved = (root / relative).resolve()
        if not resolved.is_relative_to(root):
            raise Stage2EvidenceError("Stage 2 evidence paths must stay inside the bundle")
        return resolved

    def read_bytes(self, bundle_root: Path) -> bytes:
        path = self.resolve(bundle_root)
        if not path.is_file():
            raise Stage2EvidenceError(f"Stage 2 evidence file does not exist: {self.path}")
        value = path.read_bytes()
        if sha256(value).hexdigest() != self.sha256:
            raise Stage2EvidenceError(f"Stage 2 evidence file drifted: {self.path}")
        return value

    def read_json(self, bundle_root: Path) -> Any:
        return json.loads(self.read_bytes(bundle_root))


@dataclass(frozen=True, slots=True)
class GateEvidenceRefs:
    manifest: ArtifactRef
    records: tuple[ArtifactRef, ...]
    papers: ArtifactRef | None = None


@dataclass(frozen=True, slots=True)
class Stage2ReleaseEvidenceIndex:
    """Hash-bound inputs only; gate outcomes still require recomputation/signature."""

    index_path: Path
    candidate_id: str
    evaluation_manifest_hash: str
    stage2_config_hash: str
    model_lock_hashes: Mapping[str, str]
    calibrator_hashes: Mapping[str, str]
    threshold_hashes: Mapping[str, str]
    gold_manifest: ArtifactRef
    hidden_attestation: ArtifactRef
    public_gates: Mapping[str, GateEvidenceRefs]

    @property
    def bundle_root(self) -> Path:
        return self.index_path.parent


def load_stage2_release_evidence_index(path: Path) -> Stage2ReleaseEvidenceIndex:
    """Load one strict index and verify its file and public-manifest bindings."""

    document = json.loads(path.read_text(encoding="utf-8"))
    validate(document, "stage2-release-evidence.schema.json")
    public = document["public_gates"]
    gates = {
        "structured_replay": _single_records(public["structured_replay"], "records"),
        "rationale": _single_records(public["rationale"], "records"),
        "parity": _single_records(public["parity"], "scores"),
        "benchmark": GateEvidenceRefs(
            ArtifactRef.from_document(public["benchmark"]["manifest"]),
            tuple(
                ArtifactRef.from_document(item)
                for item in public["benchmark"]["records"]
            ),
            ArtifactRef.from_document(public["benchmark"]["papers"]),
        ),
        "soak": GateEvidenceRefs(
            ArtifactRef.from_document(public["soak"]["manifest"]),
            (ArtifactRef.from_document(public["soak"]["record"]),),
            ArtifactRef.from_document(public["soak"]["papers"]),
        ),
    }
    index = Stage2ReleaseEvidenceIndex(
        index_path=path.resolve(),
        candidate_id=str(document["candidate_id"]),
        evaluation_manifest_hash=str(document["evaluation_manifest_hash"]),
        stage2_config_hash=str(document["stage2_config_hash"]),
        model_lock_hashes=_hash_binding(document["model_lock_hashes"]),
        calibrator_hashes=_hash_binding(document["calibrator_hashes"]),
        threshold_hashes=_hash_binding(document["threshold_hashes"]),
        gold_manifest=ArtifactRef.from_document(document["gold_manifest"]),
        hidden_attestation=ArtifactRef.from_document(document["hidden_attestation"]),
        public_gates=MappingProxyType(gates),
    )
    _verify_all_refs(index)
    manifest = _verify_gold_manifest(index)
    _validate_attestation_shape_and_binding(index, manifest)
    return index


def _single_records(document: Mapping[str, Any], field: str) -> GateEvidenceRefs:
    value = document[field]
    refs = value if isinstance(value, list) else [value]
    return GateEvidenceRefs(
        ArtifactRef.from_document(document["manifest"]),
        tuple(ArtifactRef.from_document(item) for item in refs),
    )


def _hash_binding(document: Mapping[str, Any]) -> Mapping[str, str]:
    if set(document) != STAGE2_PATH_NAMES:
        raise Stage2EvidenceError("Stage 2 evidence must bind reranker and qwen hashes")
    return MappingProxyType({str(key): str(value) for key, value in document.items()})


def _verify_all_refs(index: Stage2ReleaseEvidenceIndex) -> None:
    refs = [index.gold_manifest, index.hidden_attestation]
    for gate in index.public_gates.values():
        refs.append(gate.manifest)
        refs.extend(gate.records)
        if gate.papers is not None:
            refs.append(gate.papers)
    for ref in refs:
        ref.read_bytes(index.bundle_root)


def _verify_gold_manifest(index: Stage2ReleaseEvidenceIndex) -> GoldManifest:
    document = index.gold_manifest.read_json(index.bundle_root)
    manifest = gold_manifest_from_document(document)
    manifest.validate_sampling_structure()
    if manifest.hash() != index.evaluation_manifest_hash:
        raise Stage2EvidenceError(
            "Stage 2 gold manifest does not match evaluation_manifest_hash"
        )
    return manifest


def _validate_attestation_shape_and_binding(
    index: Stage2ReleaseEvidenceIndex,
    manifest: GoldManifest,
) -> None:
    document = index.hidden_attestation.read_json(index.bundle_root)
    validate(document, "stage2-hidden-evaluator-attestation.schema.json")
    payload = document["payload"]
    if document["payload_sha256"] != content_hash(payload):
        raise Stage2EvidenceError("Stage 2 hidden attestation payload hash is invalid")
    expected = {
        "candidate_id": index.candidate_id,
        "evaluation_manifest_hash": index.evaluation_manifest_hash,
        "stage2_config_hash": index.stage2_config_hash,
        "model_lock_hashes": dict(index.model_lock_hashes),
        "calibrator_hashes": dict(index.calibrator_hashes),
        "threshold_hashes": dict(index.threshold_hashes),
    }
    for field, value in expected.items():
        if payload[field] != value:
            raise Stage2EvidenceError(
                f"Stage 2 hidden attestation {field} does not match the evidence index"
            )
    hidden_pairs = {
        split.value: [
            pair.pair_id for pair in manifest.pairs if pair.split is split
        ]
        for split in (GoldSplit.HIDDEN_HARD, GoldSplit.HIDDEN_REAL)
    }
    expected_counts = {name: len(pair_ids) for name, pair_ids in hidden_pairs.items()}
    expected_universes = {
        name: pair_universe_hash(pair_ids) for name, pair_ids in hidden_pairs.items()
    }
    if payload["hidden_split_pair_counts"] != expected_counts:
        raise Stage2EvidenceError(
            "Stage 2 hidden attestation split counts do not match the gold manifest"
        )
    if payload["hidden_pair_universe_hashes"] != expected_universes:
        raise Stage2EvidenceError(
            "Stage 2 hidden attestation pair universes do not match the gold manifest"
        )
