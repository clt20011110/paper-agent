from __future__ import annotations

from copy import deepcopy

from paper_agent.analysis_registry import AnalysisNormalizationRegistry


METRIC_HASH = "677b87be65f571cd1027701cdc332cb607d8558e258f5de6431e34433742fab0"
PROTOCOL_HASH = "aa15debef3444b8ee215dd6043eaf89a00af76145ff2cfbf2a5f01bd6b67a9c3"


def _unit() -> dict:
    return {
        "claim": "ResNet-50 reached 91% accuracy.",
        "direction": "support",
        "task_id": "image_classification",
        "dataset_id": "mnist",
        "dataset_version": "original",
        "split_id": "test",
        "metric_id": "accuracy",
        "metric_definition_hash": METRIC_HASH,
        "unit": "ratio",
        "optimization_direction": "maximize",
        "value": 91.0,
        "uncertainty": None,
        "statistical_method": None,
        "protocol_id": "official_test",
        "protocol_hash": PROTOCOL_HASH,
        "sample_size": 10000,
        "baseline_id": "resnet50",
        "baseline_version": "torchvision",
        "conditions": [
            "source_task=image classification",
            "source_dataset=MNIST",
            "source_metric=accuracy",
            "source_baseline=ResNet-50",
            "source_protocol=official test split",
            "source_unit=percent",
        ],
        "locator": {"kind": "page", "value": "7"},
        "normalization_method": "model_candidate",
        "normalizer_version": "model_candidate",
        "source_value": 91.0,
        "comparison_eligibility": "comparable",
        "missing_fields": [],
    }


def test_exact_registry_mapping_and_unit_conversion_are_deterministic() -> None:
    registry = AnalysisNormalizationRegistry.load()

    normalized = registry.normalize_evidence_unit(_unit())

    assert normalized["comparison_eligibility"] == "comparable"
    assert normalized["value"] == 0.91
    assert normalized["normalization_method"] == "registry_alias+unit_conversion"
    assert normalized["normalizer_version"] == "analysis-normalization-v1"
    assert normalized["source_value"] == 91.0


def test_unknown_mapping_keeps_source_local_id_and_prevents_comparison() -> None:
    registry = AnalysisNormalizationRegistry.load()
    unit = _unit()
    unit["dataset_id"] = "private-benchmark-2026"
    unit["conditions"][1] = "source_dataset=Private Benchmark 2026"

    normalized = registry.normalize_evidence_unit(unit)

    assert normalized["dataset_id"] == "private-benchmark-2026"
    assert normalized["comparison_eligibility"] == "not_comparable"
    assert "registry_mapping:dataset_id" in normalized["missing_fields"]
    assert "source_local" in normalized["normalization_method"]


def test_candidate_id_cannot_claim_a_different_registered_source_alias() -> None:
    registry = AnalysisNormalizationRegistry.load()
    unit = deepcopy(_unit())
    unit["conditions"][1] = "source_dataset=CIFAR-10"

    normalized = registry.normalize_evidence_unit(unit)

    assert normalized["dataset_id"] == "mnist"
    assert normalized["comparison_eligibility"] == "not_comparable"
    assert "registry_mapping:dataset_id" in normalized["missing_fields"]
