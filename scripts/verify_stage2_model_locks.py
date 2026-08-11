#!/usr/bin/env python3
"""Verify locally cached Stage 2 model files against their lock files."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(lock_path: Path, model_dir: Path) -> dict[str, Any]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    files = []
    for relative_path, expected in sorted(lock["file_hashes"].items()):
        candidate = model_dir / relative_path
        actual = _sha256(candidate) if candidate.is_file() else None
        files.append({
            "path": relative_path,
            "expected_sha256": expected,
            "actual_sha256": actual,
            "matches": actual == expected,
        })
    return {
        "lock_path": str(lock_path.resolve()),
        "lock_sha256": _sha256(lock_path),
        "model_dir": str(model_dir.resolve()),
        "model_id": lock.get("model_id"),
        "files": files,
        "status": "pass" if all(item["matches"] for item in files) else "fail",
    }


def build_evidence(bindings: list[tuple[Path, Path]]) -> dict[str, Any]:
    verified = [_verify(lock_path, model_dir) for lock_path, model_dir in bindings]
    return {
        "schema_version": 1,
        "stage": "stage2",
        "bindings": verified,
        "status": "pass" if all(item["status"] == "pass" for item in verified) else "fail",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binding",
        action="append",
        nargs=2,
        metavar=("LOCK", "MODEL_DIR"),
        required=True,
        help="lock file and its local model directory; repeat for each model",
    )
    parser.add_argument("--output", type=Path, help="write the JSON evidence to this path")
    arguments = parser.parse_args(argv)
    evidence = build_evidence([(Path(lock), Path(model_dir)) for lock, model_dir in arguments.binding])
    payload = json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    if arguments.output:
        arguments.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if evidence["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
