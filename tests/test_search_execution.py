from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest

from paper_agent.cli import _provider_specs
from paper_agent.query_plan import QueryPlanDriftError, approve_query_plan, compile_query_plan
from paper_agent.search_execution import resolve_runtime_providers, seed_input

from test_query_plan import draft


ROOT = Path(__file__).parents[1]
NOW = "2026-08-09T01:00:00Z"


def _approved(provider: str, monkeypatch, **overrides):
    if provider == "openalex":
        monkeypatch.setenv("OPENALEX_API_KEY", "test-key")
    document = draft()
    document["providers"] = [provider]
    document["required_providers"] = [provider]
    specs = _provider_specs(
        [{"provider": provider, **overrides}],
        ROOT,
        venue_ids=(),
    )
    plan = compile_query_plan(document, providers=specs)
    return approve_query_plan(plan, plan["plan_hash"], approved_by="owner", approved_at=NOW)


def test_runtime_reresolves_declared_credential_presence(monkeypatch) -> None:
    plan = _approved("openalex", monkeypatch)

    runtime = resolve_runtime_providers(plan)
    assert runtime[0]["credential_availability"] == {"OPENALEX_API_KEY": True}

    monkeypatch.delenv("OPENALEX_API_KEY")
    with pytest.raises(QueryPlanDriftError, match="credential|unavailable"):
        resolve_runtime_providers(plan)


def test_snapshot_runtime_requires_the_exact_approved_file(tmp_path, monkeypatch) -> None:
    snapshot = tmp_path / "crossref.json"
    snapshot.write_bytes(b'{"message":{"items":[]}}')
    digest = sha256(snapshot.read_bytes()).hexdigest()
    plan = _approved("crossref", monkeypatch, mode="snapshot", snapshot_hash=digest)

    runtime = resolve_runtime_providers(plan, snapshot_paths={"crossref": snapshot})
    assert runtime[0]["snapshot_hash"] == digest

    snapshot.write_bytes(b'{"message":{"items":[{"changed":true}]}}')
    with pytest.raises(QueryPlanDriftError, match="snapshot_hash"):
        resolve_runtime_providers(plan, snapshot_paths={"crossref": snapshot})


@pytest.mark.parametrize(
    ("value", "kind"),
    [
        ("10.1000/example", "doi"),
        ("2501.01234v2", "arxiv"),
        ("https://example.test/paper", "url"),
        ("@article{x,title={Example}}", "bibtex"),
        ("TY  - JOUR\nTI  - Example", "ris"),
        ('{"id":"x","title":"Example"}', "csl-json"),
        ("paper.pdf", "local_pdf"),
    ],
)
def test_seed_input_infers_supported_formats(value: str, kind: str) -> None:
    assert seed_input(value).kind == kind
