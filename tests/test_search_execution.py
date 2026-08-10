from __future__ import annotations

import base64
from contextlib import closing
from hashlib import sha256
import json
from pathlib import Path
import sqlite3

import pytest

from paper_agent.cli import _provider_specs
from paper_agent.citations import DeterministicFakeScreener
from paper_agent.approved_snapshot import frozen_parameters_hash
from paper_agent.query_plan import QueryPlanDriftError, approve_query_plan, compile_query_plan
from paper_agent.providers.builtin import FixtureTransport
from paper_agent.query_compilers import compile_queries
from paper_agent.search_execution import execute_search_plan, resolve_runtime_providers, seed_input
from paper_agent.stage2_search import Stage2ReleaseError

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


def test_search_startup_requires_a_released_local_stage2_before_provider_contact(tmp_path, monkeypatch) -> None:
    plan = _approved("openalex", monkeypatch)
    provider_calls = []

    def transport(*args):
        provider_calls.append(args)
        return {"results": []}

    with pytest.raises(Stage2ReleaseError, match="requires --stage2-release"):
        execute_search_plan(plan, tmp_path / "papers.sqlite3", transport=transport)

    assert provider_calls == []


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


def _snapshot_bundle(path: Path, provider: str, responses: list[tuple[str, dict, bytes, str]]) -> str:
    document = {
        "schema_version": "1",
        "provider": provider,
        "responses": [
            {
                "operation": operation,
                "parameters_hash": frozen_parameters_hash(parameters),
                "cursor": parameters.get("cursor"),
                "content_type": content_type,
                "body_base64": base64.b64encode(body).decode("ascii"),
                "body_sha256": sha256(body).hexdigest(),
            }
            for operation, parameters, body, content_type in responses
        ],
    }
    path.write_text(json.dumps(document, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    return sha256(path.read_bytes()).hexdigest()


def _crossref_parameters() -> dict:
    query = compile_queries("crossref", draft()["query_variants"], draft()["scope"])[0]
    return dict(query.parameters)


def test_search_execution_replays_paginated_snapshot_into_sqlite_without_contact(tmp_path, monkeypatch) -> None:
    first_parameters = {**_crossref_parameters(), "cursor": None}
    second_parameters = {**_crossref_parameters(), "cursor": "page-2"}
    first = b'{"status":"ok","message":{"next-cursor":"page-2","items":[{"DOI":"10.1/first","title":["First"]}]}}'
    second = b'{"status":"ok","message":{"items":[{"DOI":"10.1/second","title":["Second"]}]}}'
    bundle = tmp_path / "crossref.snapshot.json"
    digest = _snapshot_bundle(
        bundle,
        "crossref",
        [("search", first_parameters, first, "application/json"), ("search", second_parameters, second, "application/json")],
    )
    plan = _approved("crossref", monkeypatch, mode="snapshot", snapshot_hash=digest)
    database = tmp_path / "papers.sqlite3"

    result, _, _ = execute_search_plan(
        plan,
        database,
        snapshot_paths={"crossref": bundle},
        stage2_screener=DeterministicFakeScreener(frozenset()),
    )

    assert (result.fanout.incomplete, len(result.paper_ids)) == (False, 2)
    with closing(sqlite3.connect(database)) as connection:
        assert connection.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 2
        assert connection.execute("SELECT COUNT(*) FROM search_queries WHERE role = 'search'").fetchone()[0] == 2


def test_snapshot_bundle_drift_and_missing_page_are_rejected_without_transport_fallback(tmp_path, monkeypatch) -> None:
    first_parameters = {**_crossref_parameters(), "cursor": None}
    first = b'{"status":"ok","message":{"next-cursor":"page-2","items":[{"DOI":"10.1/first","title":["First"]}]}}'
    bundle = tmp_path / "crossref.snapshot.json"
    digest = _snapshot_bundle(bundle, "crossref", [("search", first_parameters, first, "application/json")])
    plan = _approved("crossref", monkeypatch, mode="snapshot", snapshot_hash=digest)
    fallback_calls = []

    def fallback(*args):
        fallback_calls.append(args)
        raise AssertionError("snapshot request must not fall back to API transport")

    result, _, _ = execute_search_plan(
        plan,
        tmp_path / "missing-page.sqlite3",
        snapshot_paths={"crossref": bundle},
        transport=fallback,
        stage2_screener=DeterministicFakeScreener(frozenset()),
    )
    assert result.status == "incomplete"
    assert fallback_calls == []

    bundle.write_bytes(bundle.read_bytes() + b"\n")
    with pytest.raises(QueryPlanDriftError, match="snapshot_hash"):
        execute_search_plan(
            plan,
            tmp_path / "drift.sqlite3",
            snapshot_paths={"crossref": bundle},
            stage2_screener=DeterministicFakeScreener(frozenset()),
        )


def test_snapshot_and_api_providers_use_their_respective_transports(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OPENALEX_API_KEY", "test-key")
    parameters = {**_crossref_parameters(), "cursor": None}
    body = b'{"status":"ok","message":{"items":[{"DOI":"10.1/snapshot","title":["Snapshot"]}]}}'
    bundle = tmp_path / "crossref.snapshot.json"
    digest = _snapshot_bundle(bundle, "crossref", [("search", parameters, body, "application/json")])
    document = draft()
    document["providers"] = ["crossref", "openalex"]
    document["required_providers"] = ["crossref", "openalex"]
    specs = _provider_specs(
        [
            {"provider": "crossref", "mode": "snapshot", "snapshot_hash": digest},
            "openalex",
        ],
        ROOT,
        venue_ids=(),
    )
    plan = compile_query_plan(document, providers=specs)
    plan = approve_query_plan(plan, plan["plan_hash"], approved_by="owner", approved_at=NOW)
    fallback_calls = []

    def api_transport(provider, operation, parameters):
        fallback_calls.append((provider, operation, parameters))
        assert provider == "openalex"
        return {
            "results": [
                {"id": "https://openalex.org/W1", "title": "API", "authorships": [], "publication_year": 2024}
            ]
        }

    result, _, _ = execute_search_plan(
        plan,
        tmp_path / "mixed.sqlite3",
        snapshot_paths={"crossref": bundle},
        transport=api_transport,
        stage2_screener=DeterministicFakeScreener(frozenset()),
    )
    assert (result.fanout.incomplete, len(result.paper_ids)) == (False, 2)
    assert fallback_calls[0][:2] == ("openalex", "search")
    assert {call[0] for call in fallback_calls} == {"openalex"}


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


def test_venue_only_execution_runs_descriptors_without_topic_search(tmp_path) -> None:
    document = draft()
    document["scope"]["venues"] = ["neurips"]
    document["citation_snowball"]["enabled"] = False
    document["required_roles"] = ["venue_primary"]
    document["required_providers"] = []
    specs = _provider_specs(["neurips_proceedings"], ROOT, venue_ids=("neurips",))
    plan = compile_query_plan(document, providers=specs)
    approved = approve_query_plan(plan, plan["plan_hash"], approved_by="owner", approved_at=NOW)
    responses = {
        "neurips_proceedings:discover:first": json.loads(
            (ROOT / "tests/fixtures/providers/venue-neurips.json").read_text(encoding="utf-8")
        ),
        "neurips_proceedings:discover:neurips:page-2": {"entries": []},
    }
    database = tmp_path / "crawl.sqlite3"

    result, _, crawl_run_id = execute_search_plan(
        approved,
        database,
        transport=FixtureTransport(responses),
        stage2_screener=DeterministicFakeScreener(frozenset()),
        venue_only=True,
    )

    assert (result.status, len(result.paper_ids)) == ("complete", 1)
    with closing(sqlite3.connect(database)) as connection:
        assert connection.execute(
            "SELECT DISTINCT role FROM source_runs WHERE crawl_run_id = ?", (crawl_run_id,)
        ).fetchall() == [("venue_primary",)]
