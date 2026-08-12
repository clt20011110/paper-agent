from __future__ import annotations

from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from paper_agent.stage2_backends import load_model_lock


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "verify_stage2_bf16_derivation.py"
SPECIFICATION = importlib.util.spec_from_file_location(
    "verify_stage2_bf16_derivation",
    SCRIPT,
)
assert SPECIFICATION and SPECIFICATION.loader
audit = importlib.util.module_from_spec(SPECIFICATION)
sys.modules[SPECIFICATION.name] = audit
SPECIFICATION.loader.exec_module(audit)


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _expected_file_set_hash(files: dict[str, tuple[str, int]]) -> str:
    manifest = [
        {"path": path, "sha256": digest, "size": size}
        for path, (digest, size) in sorted(files.items())
    ]
    return audit._file_set_hash(manifest)


def test_pinned_file_sets_match_the_published_audit() -> None:
    assert _expected_file_set_hash(audit.ORACLE_FILES) == (
        "139e20c3a3078f2548ee92da9cef92160ad01ca36f6d4a650dea2935986301a9"
    )
    assert _expected_file_set_hash(audit.CANDIDATE_FILES) == (
        "cb633870ea1ab751c4285238ffa88a1de6137b70351933b5f148ccccacdc2fdc"
    )
    assert _expected_file_set_hash(audit.RUNTIME_FILES) == (
        "298fe67e028d1eea45ceb57f47ab87269acce6111d5b4326f3005e9121f78a88"
    )
    assert audit._candidate_key("roberta.encoder.layer.0") == "encoder.layer.0"
    assert audit._candidate_key("classifier.out_proj.weight") == (
        "classifier.out_proj.weight"
    )


def test_bf16_model_lock_binds_the_verified_runtime() -> None:
    lock_path = (
        ROOT
        / "configs/stage2/models/bge-reranker-v2-m3-mlx-bf16.lock.json"
    )
    lock = load_model_lock(lock_path)
    summary = json.loads(
        (
            ROOT
            / "docs/smoke/stage2-bf16-derivation-audit-20260812.json"
        ).read_text(encoding="utf-8")
    )

    assert lock.source_repo == audit.ORACLE_REPO
    assert lock.source_revision == audit.ORACLE_REVISION
    assert lock.conversion_repo == audit.CANDIDATE_REPO
    assert lock.conversion_revision == audit.CANDIDATE_REVISION
    assert lock.format == "safetensors-bf16"
    assert lock.quantization == "none"
    assert lock.license == "apache-2.0"
    assert lock.parameter_count == 567_755_777
    assert lock.omlx_version == "0.5.7"
    assert lock.mlx_version == "0.32.0"
    assert lock.file_hashes == {
        path: digest for path, (digest, _size) in audit.RUNTIME_FILES.items()
    }
    assert summary["model_lock"]["sha256"] == _sha256(lock_path)
    assert "/Users/" not in json.dumps(summary)


def test_file_verification_is_exact_and_output_is_no_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "weights.bin").write_bytes(b"weights")
    expected = {"weights.bin": (sha256(b"weights").hexdigest(), 7)}

    verified = audit._verify_files(model_dir, expected)
    assert verified["file_count"] == 1
    (model_dir / "unexpected.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="file set mismatch"):
        audit._verify_files(model_dir, expected)

    output = tmp_path / "nested" / "audit.json"
    monkeypatch.setattr(audit, "build_audit", lambda *_: {"result": "pass"})
    arguments = [
        "--oracle-dir",
        str(tmp_path / "oracle"),
        "--candidate-dir",
        str(tmp_path / "candidate"),
        "--runtime-dir",
        str(tmp_path / "runtime"),
        "--output",
        str(output),
    ]
    assert audit.main(arguments) == 0
    assert json.loads(output.read_text(encoding="utf-8")) == {"result": "pass"}
    with pytest.raises(FileExistsError, match="already exists"):
        audit.main(arguments)
