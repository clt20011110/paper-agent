from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from paper_agent.schema import validate
from paper_agent.stage2_benchmark_freeze import (
    BenchmarkFreezeError,
    freeze_candidate_benchmark_manifests,
    publish_candidate_benchmark_manifests,
)
from paper_agent.stage2_benchmark_inputs import benchmark_corpus_hash
from paper_agent.stage2_pipeline import Stage2Paper
from paper_agent.stage2_prompt_contract import (
    OMLX_CHAT_INPUT_TOKEN_PROXY_ESTIMATOR,
    adjudication_messages,
    estimate_omlx_chat_input_token_proxy,
)


def _papers(count: int, *, missing: int = 0) -> tuple[Stage2Paper, ...]:
    return tuple(
        Stage2Paper(
            f"paper-{index:05d}",
            f"Title {index}",
            None if index < missing else f"Abstract {index}",
            ("keyword",),
        )
        for index in range(count)
    )


def _receipt(performance: tuple[Stage2Paper, ...], soak: tuple[Stage2Paper, ...]) -> dict[str, object]:
    missing = [paper.paper_id for paper in performance if paper.abstract is None]
    present = [paper.paper_id for paper in performance if paper.abstract is not None]
    normal = sorted(missing + present[:50])
    stress = sorted(normal + present[50:200])
    return {
        "schema_version": 1,
        "candidate_independent": True,
        "omitted_bindings": ["stage2_config_hash", "threshold_artifact_hashes", "model_lock_hashes"],
        "performance": {
            "paper_ids": sorted(paper.paper_id for paper in performance),
            "papers_corpus_hash": benchmark_corpus_hash(performance),
            "abstract_present_count": 900,
            "abstract_missing_count": 100,
            "normal_qwen_ids": normal,
            "stress_qwen_ids": stress,
        },
        "soak": {
            "paper_ids": sorted(paper.paper_id for paper in soak),
            "papers_corpus_hash": benchmark_corpus_hash(soak),
        },
    }


def _release() -> SimpleNamespace:
    return SimpleNamespace(profile=SimpleNamespace(
        production_calibrated=True,
        reranker_lock_hash="a" * 64,
        adjudicator_lock_hash="b" * 64,
        base_runtime_config_hash="c" * 64,
        query="frozen molecular generation query",
        query_version="screening-query-v1",
        adjudicator_max_output_tokens=256,
        reranker_calibration=SimpleNamespace(threshold=SimpleNamespace(hash=lambda: "d" * 64)),
        adjudicator_calibration=SimpleNamespace(threshold=SimpleNamespace(hash=lambda: "e" * 64)),
    ))


def test_freezer_binds_only_candidate_and_frozen_workloads(tmp_path: Path) -> None:
    performance = _papers(1_000, missing=100)
    soak = _papers(10_000)
    manifests = freeze_candidate_benchmark_manifests(
        _release(),
        performance_papers=tuple(reversed(performance)),
        soak_papers=tuple(reversed(soak)),
        selection_receipt=_receipt(performance, soak),
    )

    assert manifests.performance.corpus_hash == benchmark_corpus_hash(performance)
    assert manifests.soak.corpus_hash == benchmark_corpus_hash(soak)
    assert manifests.performance.model_lock_hashes == ("a" * 64, "b" * 64)
    assert manifests.performance.threshold_artifact_hashes == ("d" * 64, "e" * 64)
    assert manifests.performance.output_token_limit == 256
    assert manifests.soak.output_token_limit == 256
    assert manifests.performance.input_token_estimator == OMLX_CHAT_INPUT_TOKEN_PROXY_ESTIMATOR
    assert manifests.soak.input_token_estimator == OMLX_CHAT_INPUT_TOKEN_PROXY_ESTIMATOR
    assert len(manifests.performance.normal_qwen_ids) == 150
    assert len(manifests.performance.stress_qwen_ids) == 300
    assert sum(case.abstract_missing for case in manifests.performance.cases) == 100
    assert all(case.input_tokens > 0 for case in manifests.soak.cases)
    first = performance[0]
    assert manifests.performance.cases[0].input_tokens == estimate_omlx_chat_input_token_proxy(
        adjudication_messages(
            query_version=_release().profile.query_version,
            query=_release().profile.query,
            paper=first,
        )
    )
    validate(manifests.performance.document(), "stage2-performance-manifest.schema.json")
    validate(manifests.soak.document(), "stage2-soak-manifest.schema.json")

    performance_output = tmp_path / "performance-manifest.json"
    soak_output = tmp_path / "soak-manifest.json"
    publish_candidate_benchmark_manifests(
        manifests,
        performance_output=performance_output,
        soak_output=soak_output,
    )
    assert json.loads(performance_output.read_text()) == manifests.performance.document()
    assert json.loads(soak_output.read_text()) == manifests.soak.document()
    with pytest.raises(FileExistsError):
        publish_candidate_benchmark_manifests(
            manifests,
            performance_output=performance_output,
            soak_output=soak_output,
        )


def test_freezer_rejects_receipt_and_candidate_drift() -> None:
    performance = _papers(1_000, missing=100)
    soak = _papers(10_000)
    receipt = _receipt(performance, soak)
    receipt["performance"]["normal_qwen_ids"] = receipt["performance"]["normal_qwen_ids"][1:]
    with pytest.raises(BenchmarkFreezeError, match="15% normal"):
        freeze_candidate_benchmark_manifests(
            _release(),
            performance_papers=performance,
            soak_papers=soak,
            selection_receipt=receipt,
        )

    release = _release()
    release.profile.production_calibrated = False
    with pytest.raises(BenchmarkFreezeError, match="calibrated schema-v2"):
        freeze_candidate_benchmark_manifests(
            release,
            performance_papers=performance,
            soak_papers=soak,
            selection_receipt=_receipt(performance, soak),
        )
