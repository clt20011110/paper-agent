from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import pytest

from paper_agent.canonical import content_hash
from paper_agent.domain import DownloadStatus, Paper
from paper_agent.download_cli_service import Stage3DownloadService
from paper_agent.download_providers import (
    DEFAULT_PROVIDER_ORDER,
    DEFAULT_RESOLVER_ORDER,
    EuropePMCOpenAccessResolver,
    ResolverContext,
)
from paper_agent.provider_runtime import ProviderRequestError
from paper_agent.repository import PaperRepository
from paper_agent.stage3_metadata_lookup import (
    CONTROLLED_HTTP_TRANSPORT_IMPLEMENTATION_VERSION,
    MetadataLookupDescriptor,
    MetadataLookupRegistry,
    Stage3MetadataLookup,
    default_metadata_lookup_registry,
)
from paper_agent.storage import Database


NOW = datetime(2026, 8, 10, tzinfo=UTC)


class FixtureMetadataTransport:
    def __init__(self, *, fail: bool = False, version: str = "fixture-transport-v1") -> None:
        self.fail = fail
        self.version = version
        self.calls: list[tuple[str, str, Mapping[str, Any]]] = []

    def canonical_identity(self) -> Mapping[str, Any]:
        return {"implementation_version": self.version}

    def __call__(self, provider: str, operation: str, parameters: Mapping[str, Any]) -> Mapping[str, Any]:
        self.calls.append((provider, operation, dict(parameters)))
        if self.fail:
            raise ProviderRequestError("fixture metadata failure")
        payloads = {
            "europe_pmc": {"resultList": {"result": []}},
            "unpaywall": {"is_oa": False},
            "arxiv": {"feed": {"entry": []}},
        }
        return {**payloads[provider], "raw_response_artifact_hash": f"{provider}-raw"}


def _config(*, metadata_lookup: bool) -> dict[str, Any]:
    download: dict[str, Any] = {
        "resolvers": list(DEFAULT_RESOLVER_ORDER),
        "providers": list(DEFAULT_PROVIDER_ORDER),
        "purpose": "personal_research",
        "policy_matrix": "policies/download-access-v1.yaml",
        "authorized_skill": {"enabled": False},
    }
    if metadata_lookup:
        download["metadata_lookup"] = {
            "enabled": True,
            "contact": "mailto:operator@example.test",
            "user_agent": "paper-agent-test/stage3",
            "timeout_seconds": 7,
            "unpaywall_email": "operator@example.test",
        }
    return {"download": download}


def _service(
    tmp_path: Path,
    database: Database,
    transport: FixtureMetadataTransport,
    *,
    metadata_lookup: bool,
) -> Stage3DownloadService:
    return Stage3DownloadService(
        database,
        _config(metadata_lookup=metadata_lookup),
        config_root=Path(__file__).parents[1],
        artifact_root=tmp_path / "output",
        metadata_transport=transport,
        clock=lambda: NOW,
    )


def _save_paper(database: Database) -> str:
    return PaperRepository(database).save_paper(
        Paper("paper-metadata", "Metadata Lookup", doi="10.1000/Stage3", arxiv_id="2501.01234v2")
    ).paper_id


@pytest.fixture
def database(tmp_path: Path):
    with Database(tmp_path / "papers.sqlite3") as value:
        value.migrate()
        yield value


def test_service_uses_descriptor_lookup_order_and_identifier_bindings(
    tmp_path: Path, database: Database
) -> None:
    transport = FixtureMetadataTransport()
    paper_id = _save_paper(database)
    service = _service(tmp_path, database, transport, metadata_lookup=True)

    result = service.run(paper_ids=[paper_id])

    assert result.run is not None
    assert result.run.for_paper(paper_id).status is DownloadStatus.MANUAL_REQUIRED
    assert [(provider, operation) for provider, operation, _ in transport.calls] == [
        ("europe_pmc", "search"),
        ("unpaywall", "resolve"),
        ("arxiv", "search"),
    ]
    assert transport.calls[0][2] == {
        "doi": "10.1000/stage3",
        "resultType": "core",
    }
    assert transport.calls[1][2]["doi"] == "10.1000/stage3"
    assert transport.calls[2][2] == {"query": "id:2501.01234"}


def test_lookup_retains_raw_evidence_hash_and_retrieval_time_for_resolver_candidates() -> None:
    class EuropeFixture:
        def __call__(self, provider: str, operation: str, parameters: Mapping[str, Any]) -> Mapping[str, Any]:
            assert (provider, operation, dict(parameters)) == (
                "europe_pmc",
                "search",
                {"doi": "10.1000/stage3", "resultType": "core"},
            )
            return {
                "raw_response_artifact_hash": "europe-pmc-response-sha256",
                "resultList": {"result": [{
                    "isOpenAccess": "Y",
                    "fullTextUrlList": {"fullTextUrl": [{
                        "availability": "Open access",
                        "documentStyle": "pdf",
                        "url": "https://europepmc.example/paper.pdf",
                    }]},
                }]},
            }

    paper = Paper("paper-metadata", "Metadata Lookup", doi="10.1000/stage3")
    lookup = Stage3MetadataLookup(EuropeFixture(), lambda: NOW, default_metadata_lookup_registry())
    candidates = EuropePMCOpenAccessResolver().resolve(ResolverContext(paper=paper, lookup=lookup))

    assert candidates[0].raw_evidence_hash == "europe-pmc-response-sha256"
    assert candidates[0].retrieved_at == "2026-08-10T00:00:00Z"


def test_default_europe_pmc_lookup_freezes_the_core_result_contract() -> None:
    registry = default_metadata_lookup_registry()
    descriptor = registry.get("europe_pmc")

    assert descriptor is not None
    assert descriptor.parameters(Paper("paper-metadata", "Metadata Lookup", doi="10.1000/stage3")) == {
        "doi": "10.1000/stage3",
        "resultType": "core",
    }
    assert registry.canonical_identity()["descriptors"][0] == {
        "resolver": "europe_pmc",
        "provider": "europe_pmc",
        "operation": "search",
        "implementation_version": "europe-pmc-doi-parameters-v2",
        "parameter_contract": {
            "paper_field": "doi",
            "parameter": "doi",
            "fixed_parameters": {"resultType": "core"},
            "missing": "skip",
        },
    }


def test_lookup_identity_comes_from_the_actual_registry_contract() -> None:
    def parameters(paper: Paper) -> Mapping[str, Any] | None:
        return {"doi": paper.doi} if paper.doi else None

    search = MetadataLookupRegistry((MetadataLookupDescriptor(
        "europe_pmc",
        "europe_pmc",
        "search",
        "fixture-parameters-v1",
        {"paper_field": "doi", "parameter": "doi", "missing": "skip"},
        parameters,
    ),))
    resolve = MetadataLookupRegistry((MetadataLookupDescriptor(
        "europe_pmc",
        "europe_pmc",
        "resolve",
        "fixture-parameters-v1",
        {"paper_field": "doi", "parameter": "doi", "missing": "skip"},
        parameters,
    ),))
    transport = FixtureMetadataTransport()
    first = Stage3MetadataLookup(transport, lambda: NOW, search)
    second = Stage3MetadataLookup(transport, lambda: NOW, resolve)

    assert content_hash(first.canonical_identity()) != content_hash(
        second.canonical_identity()
    )
    assert first.canonical_identity()["registry"]["descriptors"][0][
        "operation"
    ] == "search"


def test_default_metadata_transport_has_a_stable_implementation_identity(
    tmp_path: Path, database: Database,
) -> None:
    service = Stage3DownloadService(
        database,
        _config(metadata_lookup=True),
        config_root=Path(__file__).parents[1],
        artifact_root=tmp_path / "output",
        clock=lambda: NOW,
    )

    assert service.lookup is not None
    assert service.lookup.canonical_identity()["transport"] == {
        "implementation_version": CONTROLLED_HTTP_TRANSPORT_IMPLEMENTATION_VERSION
    }


def test_injected_metadata_transport_requires_identity_before_network(
    tmp_path: Path, database: Database,
) -> None:
    paper_id = _save_paper(database)

    class AnonymousTransport:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, provider, operation, parameters):
            self.calls += 1
            return {}

    transport = AnonymousTransport()
    service = Stage3DownloadService(
        database,
        _config(metadata_lookup=True),
        config_root=Path(__file__).parents[1],
        artifact_root=tmp_path / "output",
        metadata_transport=transport,
        clock=lambda: NOW,
    )

    with pytest.raises(ValueError, match="transport must expose canonical_identity"):
        service.run(paper_ids=[paper_id], run_id="anonymous-transport")

    assert transport.calls == 0
    assert database.connection.execute("SELECT COUNT(*) FROM pipeline_runs").fetchone()[0] == 0
    assert database.connection.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0] == 0


def test_metadata_transport_version_drift_refuses_resume_before_network(
    tmp_path: Path, database: Database,
) -> None:
    paper_id = _save_paper(database)
    original = FixtureMetadataTransport(version="fixture-transport-v1")
    _service(tmp_path, database, original, metadata_lookup=True).run(
        paper_ids=[paper_id], run_id="transport-drift"
    )
    drifted = FixtureMetadataTransport(version="fixture-transport-v2")

    with pytest.raises(ValueError, match="different frozen inputs"):
        _service(tmp_path, database, drifted, metadata_lookup=True).run(
            paper_ids=[paper_id], run_id="transport-drift"
        )

    assert drifted.calls == []


def test_stage3_snapshot_binds_the_injected_lookup_registry_and_dry_run_writes_nothing(
    tmp_path: Path, database: Database,
) -> None:
    paper_id = _save_paper(database)

    def parameters(paper: Paper) -> Mapping[str, Any] | None:
        return {"doi": paper.doi} if paper.doi else None

    def service(operation: str, transport: FixtureMetadataTransport):
        registry = MetadataLookupRegistry((MetadataLookupDescriptor(
            "europe_pmc",
            "europe_pmc",
            operation,
            "fixture-parameters-v1",
            {"paper_field": "doi", "parameter": "doi", "missing": "skip"},
            parameters,
        ),))
        return Stage3DownloadService(
            database,
            _config(metadata_lookup=True),
            config_root=Path(__file__).parents[1],
            artifact_root=tmp_path / "output",
            lookup=Stage3MetadataLookup(transport, lambda: NOW, registry),
            clock=lambda: NOW,
        )

    search_transport = FixtureMetadataTransport()
    resolve_transport = FixtureMetadataTransport()
    search = service("search", search_transport).run(
        paper_ids=[paper_id], run_id="lookup-search", dry_run=True
    )
    resolve = service("resolve", resolve_transport).run(
        paper_ids=[paper_id], run_id="lookup-resolve", dry_run=True
    )

    assert search.resolver_snapshot is not None
    assert resolve.resolver_snapshot is not None
    assert search.resolver_snapshot.snapshot_hash != resolve.resolver_snapshot.snapshot_hash
    assert search_transport.calls[0][1] == "search"
    assert resolve_transport.calls[0][1] == "resolve"
    assert database.connection.execute("SELECT COUNT(*) FROM pipeline_runs").fetchone()[0] == 0
    assert database.connection.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0] == 0
    assert not (tmp_path / "output" / "artifacts").exists()


def test_absent_metadata_configuration_keeps_existing_manual_queue_semantics(
    tmp_path: Path, database: Database
) -> None:
    transport = FixtureMetadataTransport()
    paper_id = _save_paper(database)

    result = _service(tmp_path, database, transport, metadata_lookup=False).run(paper_ids=[paper_id])

    paper = result.run.for_paper(paper_id) if result.run else None
    assert paper is not None
    assert paper.status is DownloadStatus.MANUAL_REQUIRED
    assert paper.reason_code == "no_access_location_candidates"
    assert transport.calls == []


def test_metadata_provider_failure_degrades_to_existing_manual_queue(
    tmp_path: Path, database: Database
) -> None:
    transport = FixtureMetadataTransport(fail=True)
    paper_id = _save_paper(database)

    result = _service(tmp_path, database, transport, metadata_lookup=True).run(paper_ids=[paper_id])

    paper = result.run.for_paper(paper_id) if result.run else None
    assert paper is not None
    assert paper.status is DownloadStatus.MANUAL_REQUIRED
    assert paper.reason_code == "no_access_location_candidates"
    assert [call[:2] for call in transport.calls] == [
        ("europe_pmc", "search"), ("unpaywall", "resolve"), ("arxiv", "search"),
    ]
