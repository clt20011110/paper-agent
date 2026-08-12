from __future__ import annotations

import json
from pathlib import Path

import pytest

from paper_agent.stage2_challengers import (
    ChallengerRegistryError,
    evaluation_challengers,
    load_stage2_challenger_registry,
)


REGISTRY = Path(__file__).parents[1] / "configs" / "stage2" / "challengers.json"


def test_registry_records_all_requested_challengers_without_production_approval() -> None:
    candidates = load_stage2_challenger_registry(REGISTRY)

    assert {candidate.source_repo for candidate in candidates} == {
        "Querit/Querit-4B",
        "Qwen/Qwen3-Reranker-0.6B",
        "Qwen/Qwen3-Reranker-4B",
    }
    assert all(candidate.parameter_evidence_url.startswith("https://huggingface.co/") for candidate in candidates)
    assert all(candidate.is_within_parameter_bound for candidate in candidates)
    assert not any(candidate.production_approved for candidate in candidates)


def test_preflight_rejects_candidates_until_local_backend_is_verified() -> None:
    eligible = evaluation_challengers(REGISTRY)

    assert eligible == ()


def test_preflight_admits_a_pinned_licensed_backend_verified_candidate(tmp_path: Path) -> None:
    document = json.loads(REGISTRY.read_text(encoding="utf-8"))
    document["candidates"][2]["backend_capability_status"] = "verified"
    path = tmp_path / "challengers.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    eligible = evaluation_challengers(path)

    assert [candidate.id for candidate in eligible] == ["qwen3-reranker-4b"]
    assert eligible[0].production_approved is False


def test_preflight_rejects_short_revision(tmp_path: Path) -> None:
    document = json.loads(REGISTRY.read_text(encoding="utf-8"))
    document["candidates"][2]["source_revision"] = "25f203a0"
    path = tmp_path / "challengers.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ChallengerRegistryError, match="immutable 40-character SHA"):
        load_stage2_challenger_registry(path)


def test_registry_cannot_grant_production_approval(tmp_path: Path) -> None:
    document = json.loads(REGISTRY.read_text(encoding="utf-8"))
    document["candidates"][2]["production_approved"] = True
    path = tmp_path / "challengers.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ChallengerRegistryError, match="cannot production-approve"):
        load_stage2_challenger_registry(path)
