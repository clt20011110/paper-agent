#!/usr/bin/env python3
"""Verify the pinned Stage 2 BGE FP32-to-BF16 conversion and runtime."""

from __future__ import annotations

import argparse
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform
from typing import Any, Mapping


ORACLE_REPO = "BAAI/bge-reranker-v2-m3"
ORACLE_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
CANDIDATE_REPO = "soichisumi/bge-reranker-v2-m3-mlx"
CANDIDATE_REVISION = "b4577f49e18adb53ed9e557192094f69f3dc2c1c"

ORACLE_FILES = {
    ".gitattributes": ("34448b82c17d60fec9b65b1f093c115ddbaadc04beb1b0140b6bfed2e012a930", 1_570),
    "README.md": ("c887aa6dd2598f908bf0582ca7068cc816585c7a1b6a07df305b631ede0cb174", 17_229),
    "assets/BEIR-bge-en-v1.5.png": ("c1e57bfc0bb87408ac0d5084acce36a26914a2ea19f2f9f3560e71be9d4daef6", 56_374),
    "assets/BEIR-e5-mistral.png": ("5119f2d1b364f79eaa2954bd5efb4a7e56364ee8748571964d800f85683280c4", 40_223),
    "assets/CMTEB-retrieval-bge-zh-v1.5.png": ("ce02a58c566da5f733b6928cc968865e4d472c8a276546bfce5d0f4452050ed4", 51_484),
    "assets/llama-index.png": ("62c4fbdeeb44296da80bdd2f0a7a6b5e44f3492072af84fdf3ed99d01a53e596", 106_473),
    "assets/miracl-bge-m3.png": ("98f40bf0ba104f3efa52ef76da1434fb776834cdc732e6b29d2550672ea2df1b", 52_028),
    "config.json": ("13dcd6c31d9fec9d1d8e158702072f62d7fa7d312a64b9fe057bec9a08cfe41a", 795),
    "model.safetensors": ("d9e3e081faff1eefb84019509b2f5558fd74c1a05a2c7db22f74174fcedb5286", 2_271_071_852),
    "sentencepiece.bpe.model": ("cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865", 5_069_051),
    "special_tokens_map.json": ("8c785abebea9ae3257b61681b4e6fd8365ceafde980c21970d001e834cf10835", 964),
    "tokenizer.json": ("69564b696052886ed0ac63fa393e928384e0f8caada38c1f4864a9bfbf379c15", 17_098_273),
    "tokenizer_config.json": ("7e4c1cc848840aeccdd763458c18dd525eb0f795c992e00ebe9c28554e7db2d4", 1_173),
}

CANDIDATE_FILES = {
    ".gitattributes": ("34448b82c17d60fec9b65b1f093c115ddbaadc04beb1b0140b6bfed2e012a930", 1_570),
    "README.md": ("4eae846c8fb064ef2433afb6546cbc19b4d2d18c55e32550cec7ad89eb17b0e2", 553),
    "config.json": ("f6af7b8ee660a78a5a6f129766823712a7a0a5f93994014dbbb17c79cc2ed2b8", 829),
    "model.safetensors": ("80be6e38dfd2156d865a5068cdd78774f29b4b91ce100acc9f331c382e2b18b4", 1_135_556_833),
    "model.safetensors.index.json": ("1792b5cbb59f52c8d68eea2c6be07d6ad07af77ffe812fa5fab583a71e206c4f", 29_278),
    "special_tokens_map.json": ("8c785abebea9ae3257b61681b4e6fd8365ceafde980c21970d001e834cf10835", 964),
    "tokenizer.json": ("5df1f55d60c9705a501ab9a75550728625740741fe4be308dac4806c16b7d51d", 17_098_085),
    "tokenizer_config.json": ("44951e38f3060da047ee671e2ce4aac84e210fa771366110a1e7ba3ac870b2b7", 408),
}

RUNTIME_FILES = {
    "config.json": CANDIDATE_FILES["config.json"],
    "model.safetensors": CANDIDATE_FILES["model.safetensors"],
    "model.safetensors.index.json": CANDIDATE_FILES["model.safetensors.index.json"],
    "sentencepiece.bpe.model": ORACLE_FILES["sentencepiece.bpe.model"],
    "special_tokens_map.json": ORACLE_FILES["special_tokens_map.json"],
    "tokenizer.json": ORACLE_FILES["tokenizer.json"],
    "tokenizer_config.json": ORACLE_FILES["tokenizer_config.json"],
}


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _file_set_hash(files: list[dict[str, Any]]) -> str:
    payload = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return sha256(payload).hexdigest()


def _verify_files(
    directory: Path,
    expected: Mapping[str, tuple[str, int]],
) -> dict[str, Any]:
    actual_paths = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and ".cache" not in path.relative_to(directory).parts
    }
    if actual_paths != set(expected):
        raise ValueError(
            f"file set mismatch for {directory}: expected {sorted(expected)}, "
            f"observed {sorted(actual_paths)}"
        )
    files = []
    for relative, (expected_sha256, expected_size) in sorted(expected.items()):
        path = directory / relative
        observed_sha256 = _sha256(path)
        observed_size = path.stat().st_size
        if (observed_sha256, observed_size) != (expected_sha256, expected_size):
            raise ValueError(f"hash or size mismatch: {path}")
        files.append({"path": relative, "sha256": observed_sha256, "size": observed_size})
    return {
        "file_count": len(files),
        "file_set_hash": _file_set_hash(files),
        "files": files,
    }


def _verify_revision_manifest(
    directory: Path,
    revision: str,
    expected: Mapping[str, tuple[str, int]],
) -> None:
    path = directory / ".cache" / "huggingface" / "trees" / f"{revision}.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    files = manifest["files"]
    if set(files) != set(expected):
        raise ValueError(f"pinned tree file set mismatch: {path}")
    for relative, (expected_sha256, expected_size) in expected.items():
        entry = files[relative]
        if entry["size"] != expected_size:
            raise ValueError(f"pinned tree size mismatch: {relative}")
        if "lfs_sha256" in entry and entry["lfs_sha256"] != expected_sha256:
            raise ValueError(f"pinned tree LFS hash mismatch: {relative}")


def _candidate_key(oracle_key: str) -> str:
    return oracle_key.removeprefix("roberta.")


def _verify_tensor_cast(oracle_path: Path, candidate_path: Path) -> dict[str, Any]:
    import mlx.core as mx

    oracle = mx.load(str(oracle_path))
    candidate = mx.load(str(candidate_path))
    mapped = {_candidate_key(key): key for key in oracle}
    if len(mapped) != len(oracle) or set(mapped) != set(candidate):
        raise ValueError("oracle-to-candidate tensor mapping is not bijective")

    element_count = 0
    mismatch_count = 0
    for candidate_key, oracle_key in sorted(mapped.items()):
        oracle_tensor = oracle[oracle_key]
        candidate_tensor = candidate[candidate_key]
        if oracle_tensor.dtype != mx.float32 or candidate_tensor.dtype != mx.bfloat16:
            raise ValueError(f"unexpected tensor dtype: {oracle_key} -> {candidate_key}")
        if oracle_tensor.shape != candidate_tensor.shape:
            raise ValueError(f"tensor shape mismatch: {oracle_key} -> {candidate_key}")
        element_count += oracle_tensor.size
        mismatch_count += not bool(
            mx.array_equal(oracle_tensor.astype(mx.bfloat16), candidate_tensor)
        )

    if mismatch_count:
        raise ValueError(f"{mismatch_count} tensors are not exact MLX FP32-to-BF16 casts")
    return {
        "mapping_rule": (
            "strip the exact roberta. prefix from oracle keys; keep all other keys"
        ),
        "mapping_is_bijective": True,
        "oracle_tensor_count": len(oracle),
        "candidate_tensor_count": len(candidate),
        "oracle_dtype_counts": {"F32": len(oracle)},
        "candidate_dtype_counts": {"BF16": len(candidate)},
        "element_count": element_count,
        "mismatch_count": mismatch_count,
    }


def build_audit(oracle_dir: Path, candidate_dir: Path, runtime_dir: Path) -> dict[str, Any]:
    _verify_revision_manifest(oracle_dir, ORACLE_REVISION, ORACLE_FILES)
    _verify_revision_manifest(candidate_dir, CANDIDATE_REVISION, CANDIDATE_FILES)
    oracle = _verify_files(oracle_dir, ORACLE_FILES)
    candidate = _verify_files(candidate_dir, CANDIDATE_FILES)
    runtime = _verify_files(runtime_dir, RUNTIME_FILES)
    tensor_contract = _verify_tensor_cast(
        oracle_dir / "model.safetensors",
        candidate_dir / "model.safetensors",
    )
    return {
        "schema_version": 1,
        "kind": "bge_reranker_v2_m3_fp32_to_bf16_exact_cast_audit",
        "oracle": {
            "repo_id": ORACLE_REPO,
            "revision": ORACLE_REVISION,
            **oracle,
        },
        "candidate": {
            "repo_id": CANDIDATE_REPO,
            "revision": CANDIDATE_REVISION,
            **candidate,
        },
        "runtime": runtime,
        "tensor_contract": tensor_contract,
        "toolchain": {
            "method": (
                "mlx.core.array.astype(mlx.core.bfloat16), then "
                "mlx.core.array_equal, for every tensor"
            ),
            "python": platform.python_version(),
            "mlx": version("mlx"),
            "omlx": version("omlx"),
            "safetensors": version("safetensors"),
        },
        "result": "pass",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-dir", required=True, type=Path)
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--runtime-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    if arguments.output and arguments.output.exists():
        raise FileExistsError(f"output already exists: {arguments.output}")

    audit = build_audit(
        arguments.oracle_dir,
        arguments.candidate_dir,
        arguments.runtime_dir,
    )
    payload = json.dumps(audit, indent=2, sort_keys=True) + "\n"
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        with arguments.output.open("x", encoding="utf-8") as handle:
            handle.write(payload)
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
