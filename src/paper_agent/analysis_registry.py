"""Versioned, exact-match normalization for Stage 4 comparison evidence."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import re
import sysconfig
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from .canonical import content_hash


ENTITY_FIELDS = MappingProxyType({
    "task_id": ("tasks", "source_task"),
    "dataset_id": ("datasets", "source_dataset"),
    "metric_id": ("metrics", "source_metric"),
    "baseline_id": ("baselines", "source_baseline"),
    "protocol_id": ("protocols", "source_protocol"),
})
COMPARISON_FIELDS = (
    "task_id",
    "dataset_id",
    "dataset_version",
    "split_id",
    "metric_id",
    "metric_definition_hash",
    "unit",
    "value",
    "protocol_id",
    "protocol_hash",
    "baseline_id",
    "baseline_version",
    "source_value",
)


class AnalysisRegistryError(ValueError):
    pass


def registry_directory(override: Path | None = None) -> Path:
    if override is not None:
        return override
    repository = Path(__file__).resolve().parents[2] / "registries"
    if repository.is_dir():
        return repository
    return Path(sysconfig.get_path("data")) / "share" / "paper-agent" / "registries"


@dataclass(frozen=True, slots=True)
class AnalysisNormalizationRegistry:
    version: str
    registry_hash: str
    entities: Mapping[str, Mapping[str, Mapping[str, Any]]]
    units: Mapping[str, Mapping[str, Any]]

    @classmethod
    def load(cls, path: str | Path | None = None) -> "AnalysisNormalizationRegistry":
        source = Path(path) if path is not None else registry_directory() / "analysis-normalization-v1.yaml"
        document = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(document, Mapping) or document.get("schema_version") != "1":
            raise AnalysisRegistryError("analysis registry schema_version must be 1")
        version = document.get("registry_version")
        if not isinstance(version, str) or not version:
            raise AnalysisRegistryError("analysis registry_version is required")
        entities: dict[str, Mapping[str, Mapping[str, Any]]] = {}
        for group in ("tasks", "datasets", "metrics", "baselines", "protocols"):
            entries = document.get(group)
            if not isinstance(entries, Mapping):
                raise AnalysisRegistryError(f"analysis registry {group} must be a mapping")
            entities[group] = MappingProxyType({str(key): MappingProxyType(dict(value)) for key, value in entries.items()})
        units = document.get("units")
        if not isinstance(units, Mapping):
            raise AnalysisRegistryError("analysis registry units must be a mapping")
        return cls(
            version,
            content_hash(document),
            MappingProxyType(entities),
            MappingProxyType({str(key): MappingProxyType(dict(value)) for key, value in units.items()}),
        )

    def normalize_analysis(self, output: Mapping[str, Any]) -> dict[str, Any]:
        normalized = deepcopy(dict(output))
        normalized["evidence_units"] = [
            self.normalize_evidence_unit(unit) for unit in output["evidence_units"]
        ]
        cited_labels = {
            (item["axis"], item["value"])
            for item in normalized["label_evidence"]
        }
        normalized["labels"] = {
            axis: [
                value for value in values if (axis, value) in cited_labels
            ]
            if isinstance(values, list)
            else values
            for axis, values in normalized["labels"].items()
        }
        if normalized["comparison_eligibility"] == "comparable" and (
            not normalized["evidence_units"]
            or any(unit["comparison_eligibility"] != "comparable" for unit in normalized["evidence_units"])
        ):
            normalized["comparison_eligibility"] = "not_comparable"
            normalized["missing_fields"] = _merge(normalized["missing_fields"], ("comparable_evidence_unit",))
        return normalized

    def normalize_evidence_unit(self, unit: Mapping[str, Any]) -> dict[str, Any]:
        normalized = deepcopy(dict(unit))
        source_tags = _source_tags(normalized["conditions"])
        issues: list[str] = []
        used_alias = False

        for field, (group, source_tag) in ENTITY_FIELDS.items():
            candidate = normalized[field]
            if candidate is None:
                continue
            entry = self.entities[group].get(str(candidate))
            source_value = source_tags.get(source_tag, str(candidate))
            if entry is None:
                issues.append(f"registry_mapping:{field}")
                continue
            match = _match_kind(str(candidate), source_value, entry)
            if match is None:
                issues.append(f"registry_mapping:{field}")
            elif match == "alias":
                used_alias = True

        dataset = self.entities["datasets"].get(str(normalized["dataset_id"]))
        if dataset is not None and normalized["dataset_version"] not in dataset.get("versions", ()):
            issues.append("dataset_version")
        baseline = self.entities["baselines"].get(str(normalized["baseline_id"]))
        if baseline is not None and normalized["baseline_version"] not in baseline.get("versions", ()):
            issues.append("baseline_version")

        metric = self.entities["metrics"].get(str(normalized["metric_id"]))
        if metric is not None:
            if normalized["metric_definition_hash"] != metric.get("definition_hash"):
                issues.append("metric_definition_hash")
            if normalized["optimization_direction"] != metric.get("optimization_direction"):
                issues.append("optimization_direction")
        protocol = self.entities["protocols"].get(str(normalized["protocol_id"]))
        if protocol is not None and normalized["protocol_hash"] != protocol.get("protocol_hash"):
            issues.append("protocol_hash")

        unit_kind = self._normalize_unit(normalized, source_tags.get("source_unit"))
        if unit_kind is None:
            issues.append("unit")
        missing = [field for field in COMPARISON_FIELDS if normalized[field] is None]
        if not isinstance(normalized["value"], (int, float)):
            missing.append("value")
        issues.extend(missing)

        if issues:
            normalized["comparison_eligibility"] = "not_comparable"
            normalized["missing_fields"] = _merge(normalized["missing_fields"], issues)
        methods = ["registry_alias" if used_alias else "registry_exact"]
        if unit_kind == "converted":
            methods.append("unit_conversion")
        if any(item.startswith("registry_mapping:") for item in issues):
            methods.append("source_local")
        normalized["normalization_method"] = "+".join(methods)
        normalized["normalizer_version"] = self.version
        return normalized

    def _normalize_unit(self, unit: dict[str, Any], source_unit: str | None) -> str | None:
        candidate = str(source_unit or unit["unit"] or "")
        matched = next(
            (
                value
                for unit_id, value in self.units.items()
                if _key(candidate) in {_key(unit_id), *(_key(alias) for alias in value.get("aliases", ()))}
            ),
            None,
        )
        if matched is None or unit["unit"] != matched.get("canonical_unit"):
            return None
        scale = float(matched.get("scale", 1.0))
        offset = float(matched.get("offset", 0.0))
        source_value = unit["source_value"]
        if scale == 1.0 and offset == 0.0:
            if isinstance(source_value, (int, float)):
                unit["value"] = source_value
            return "exact"
        if not isinstance(source_value, (int, float)):
            return None
        unit["value"] = source_value * scale + offset
        return "converted"


def _source_tags(conditions: object) -> dict[str, str]:
    tags: dict[str, str] = {}
    for condition in conditions if isinstance(conditions, list) else ():
        if isinstance(condition, str) and "=" in condition:
            key, value = condition.split("=", 1)
            if key.startswith("source_") and value:
                tags[key] = value
    return tags


def _match_kind(canonical_id: str, source_value: str, entry: Mapping[str, Any]) -> str | None:
    if _key(source_value) == _key(canonical_id):
        return "exact"
    if _key(source_value) in {_key(alias) for alias in entry.get("aliases", ())}:
        return "alias"
    return None


def _key(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).strip().casefold())


def _merge(existing: object, additions: object) -> list[str]:
    values = [str(item) for item in existing if isinstance(item, str)] if isinstance(existing, list) else []
    return list(dict.fromkeys([*values, *(str(item) for item in additions)]))
