from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys


SCRIPT = Path(__file__).parents[1] / "scripts" / "verify_stage2_model_locks.py"


def _digest(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def test_model_lock_verifier_writes_self_contained_offline_evidence(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model":"fixture"}\n', encoding="utf-8")
    (model_dir / "weights.bin").write_text("fixture weights\n", encoding="utf-8")
    lock_path = tmp_path / "fixture.lock.json"
    lock_path.write_text(json.dumps({
        "model_id": "fixture-model",
        "file_hashes": {
            "config.json": _digest('{"model":"fixture"}\n'),
            "weights.bin": _digest("fixture weights\n"),
        },
    }), encoding="utf-8")
    output = tmp_path / "evidence.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--binding", str(lock_path), str(model_dir),
            "--output", str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    evidence = json.loads(completed.stdout)
    assert json.loads(output.read_text(encoding="utf-8")) == evidence
    assert evidence == {
        "schema_version": 1,
        "stage": "stage2",
        "status": "pass",
        "bindings": [{
            "lock_path": str(lock_path.resolve()),
            "lock_sha256": sha256(lock_path.read_bytes()).hexdigest(),
            "model_dir": str(model_dir.resolve()),
            "model_id": "fixture-model",
            "status": "pass",
            "files": [
                {
                    "path": "config.json",
                    "expected_sha256": _digest('{"model":"fixture"}\n'),
                    "actual_sha256": _digest('{"model":"fixture"}\n'),
                    "matches": True,
                },
                {
                    "path": "weights.bin",
                    "expected_sha256": _digest("fixture weights\n"),
                    "actual_sha256": _digest("fixture weights\n"),
                    "matches": True,
                },
            ],
        }],
    }
