from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import pytest

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
from paper_agent.stage3_metadata_lookup import Stage3MetadataLookup, default_metadata_lookup_registry
from paper_agent.storage import Database


NOW = datetime(2026, 8, 10, tzinfo=UTC)


class FixtureMetadataTransport:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[str, str, Mapping[str, Any]]] = []

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
    assert transport.calls[0][2] == {"doi": "10.1000/stage3"}
    assert transport.calls[1][2]["doi"] == "10.1000/stage3"
    assert transport.calls[2][2] == {"query": "id:2501.01234"}


def test_lookup_retains_raw_evidence_hash_and_retrieval_time_for_resolver_candidates() -> None:
    class EuropeFixture:
        def __call__(self, provider: str, operation: str, parameters: Mapping[str, Any]) -> Mapping[str, Any]:
            assert (provider, operation, parameters["doi"]) == ("europe_pmc", "search", "10.1000/stage3")
            return {
                "raw_response_artifact_hash": "europe-pmc-response-sha256",
                "resultList": {"result": [{
                    "isOpenAccess": "Y",
                    "fullTextUrlList": {"fullTextUrl": [{
                        "availability": "Open access", "url": "https://europepmc.example/paper.pdf",
                    }]},
                }]},
            }

    paper = Paper("paper-metadata", "Metadata Lookup", doi="10.1000/stage3")
    lookup = Stage3MetadataLookup(EuropeFixture(), lambda: NOW, default_metadata_lookup_registry())
    candidates = EuropePMCOpenAccessResolver().resolve(ResolverContext(paper=paper, lookup=lookup))

    assert candidates[0].raw_evidence_hash == "europe-pmc-response-sha256"
    assert candidates[0].retrieved_at == "2026-08-10T00:00:00Z"


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
