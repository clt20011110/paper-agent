from __future__ import annotations

import json
import importlib.util
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
import sys

import pytest
from pypdf import PdfWriter

import paper_agent.smoke as smoke
from paper_agent.config import load_config
from paper_agent.domain import Paper
from paper_agent.download_cli_service import (
    Stage3DownloadResult,
    Stage3DownloadService,
    load_provider_terms,
)
from paper_agent.downloads import HTTPResponse, urllib_fetch
from paper_agent.repository import PaperRepository
from paper_agent.resources import public_oa_terms_path, release_asset_root
from paper_agent.stage3_metadata_lookup import (
    Stage3MetadataLookup,
    default_metadata_lookup_registry,
)
from paper_agent.storage import Database


class _Transport:
    request_audit = [{
        "provider": "unpaywall",
        "status": "success",
        "url": "https://api.unpaywall.org/v2/example?email=operator@example.test",
        "content_type": "application/json",
        "response_size_bytes": 12,
        "rate_limit": {},
    }]

    def __call__(self, *_args, **_kwargs):
        return {}


def test_public_oa_smoke_uses_default_stage3_boundaries_and_sanitizes_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    calls: list[dict[str, object]] = []

    class Service:
        def __init__(self, *_args, **kwargs) -> None:
            calls.append(kwargs)
            self.lookup = Stage3MetadataLookup(
                _Transport(),
                retrieved_at=lambda: datetime.now(UTC),
                registry=default_metadata_lookup_registry(),
                transport_identity={"test": "metadata"},
            )

        def run(self, *, paper_ids, run_id):
            assert paper_ids == [smoke.PUBLIC_OA_SMOKE_PAPER_ID]
            return Stage3DownloadResult(run_id, tuple(paper_ids), "incomplete", False)

    monkeypatch.setattr(smoke, "Stage3DownloadService", Service)
    monkeypatch.setattr(smoke, "ControlledHTTPTransport", _Transport)
    output = tmp_path / "new-output"

    result = smoke.run_public_oa_download_smoke(
        output,
        contact="mailto:operator@example.test",
        unpaywall_email="operator@example.test",
        source_commit="a" * 40,
    )

    assert result.success is False
    assert result.evidence_path == output / "public-oa-evidence.json"
    assert not {"fetcher", "lookup", "metadata_transport", "resolver_registry"} & calls[0].keys()
    evidence = json.loads(result.evidence_path.read_text())
    assert evidence["fixed_paper"] == {
        "doi": "10.3758/s13421-020-01060-2", "pmcid": "PMC7683441"
    }
    assert evidence["production_path"]["pdf_fetcher"] == "urllib_fetch"
    assert evidence["source_commit"] == "a" * 40
    assert evidence["metadata_requests"][0]["operation"] == "resolve"
    assert evidence["metadata_requests"][0]["url"] == "https://api.unpaywall.org/v2/example"
    assert "operator@example.test" not in result.evidence_path.read_text()
    assert "%PDF" not in result.evidence_path.read_text()
    assert all(
        b"operator@example.test" not in path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    )

    with pytest.raises(ValueError, match="must not already exist"):
        smoke.run_public_oa_download_smoke(
            output,
            contact="mailto:operator@example.test",
            unpaywall_email="operator@example.test",
            source_commit="a" * 40,
        )


def test_public_oa_terms_are_a_versioned_release_asset() -> None:
    terms = json.loads(public_oa_terms_path().read_text(encoding="utf-8"))

    assert terms["providers"]["europe_pmc"] == {
        "terms_version": "europe-pmc-openaccess-2026-08-11",
        "evidence_url": "https://europepmc.org/downloads/openaccess",
        "machine_readable": True,
        "allows_download": True,
        "allows_storage": True,
        "allows_redistribution": None,
        "domain_allowlist": ["europepmc.org"],
    }


def test_public_oa_evidence_requires_a_persisted_pdf_artifact(tmp_path: Path) -> None:
    class EuropePMCTransport:
        def __init__(self) -> None:
            self.request_audit: list[dict[str, object]] = []

        def canonical_identity(self):
            return {"implementation_version": "fixture-europe-pmc-v1"}

        def __call__(self, provider, operation, parameters):
            self.request_audit.append({
                "provider": provider,
                "status": "success",
                "url": (
                    "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
                    if provider == "europe_pmc"
                    else "https://api.unpaywall.org/v2/example?email=operator@example.test"
                ),
                "content_type": "application/json",
                "response_size_bytes": 1,
                "rate_limit": {},
            })
            if (provider, operation) == ("unpaywall", "resolve"):
                return {
                    "raw_response_artifact_hash": "f" * 64,
                    "is_oa": False,
                }
            assert (provider, operation) == ("europe_pmc", "search")
            return {
                "raw_response_artifact_hash": "e" * 64,
                "resultList": {"result": [{
                    "isOpenAccess": "Y",
                    "pmcid": smoke.PUBLIC_OA_SMOKE_PMCID,
                    "license": "CC-BY-4.0",
                    "fullTextUrlList": {"fullTextUrl": [{
                        "availability": "Open access",
                        "documentStyle": "pdf",
                        "site": "Europe_PMC",
                        "url": (
                            "https://europepmc.org/articles/"
                            f"{smoke.PUBLIC_OA_SMOKE_PMCID}?pdf=render"
                        ),
                    }]},
                }]},
            }

    pdf = BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.write(pdf)
    root = release_asset_root()
    config = load_config(root / "configs" / "smoke_supported.yaml")
    output = tmp_path / "success"
    output.mkdir()
    transport = EuropePMCTransport()
    timestamp = datetime(2026, 8, 11, tzinfo=UTC)

    with Database(output / "papers.sqlite3") as database:
        database.migrate()
        PaperRepository(database).save_paper(Paper(
            smoke.PUBLIC_OA_SMOKE_PAPER_ID,
            "Public OA smoke paper",
            doi=smoke.PUBLIC_OA_SMOKE_DOI,
        ))
        service = Stage3DownloadService(
            database,
            config,
            config_root=root,
            artifact_root=output / "artifacts",
            provider_terms=load_provider_terms(public_oa_terms_path(root)),
            fetcher=lambda url: HTTPResponse(
                200, {"Content-Type": "application/pdf"}, pdf.getvalue(), url
            ),
            metadata_transport=transport,
            clock=lambda: timestamp,
        )
        result = service.run(
            paper_ids=[smoke.PUBLIC_OA_SMOKE_PAPER_ID],
            run_id="public-oa-success",
        )
        evidence = smoke._public_oa_evidence(
            database,
            service,
            result,
            "public-oa-success",
            timestamp,
            "a" * 40,
        )

        assert evidence["success"] is True
        assert evidence["artifact"]["byte_size"] == len(pdf.getvalue())
        artifact_path = service.artifact_root / evidence["artifact"]["relative_path"]
        artifact_path.write_bytes(b"")
        assert smoke._public_oa_evidence(
            database,
            service,
            result,
            "public-oa-success",
            timestamp,
            "a" * 40,
        )["success"] is False


def test_default_pdf_fetcher_rejects_fake_ip_dns_before_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "paper_agent.downloads.socket.getaddrinfo",
        lambda *_args, **_kwargs: [(2, 1, 6, "", ("198.18.0.66", 0))],
    )

    with pytest.raises(OSError, match="private or local address"):
        urllib_fetch("https://europepmc.org/articles/PMC7683441?pdf=render")


def test_public_oa_script_returns_nonzero_for_a_failed_smoke(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    script = Path(__file__).parents[1] / "scripts" / "run_public_oa_smoke.py"
    spec = importlib.util.spec_from_file_location("public_oa_smoke_script", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module,
        "run_public_oa_download_smoke",
        lambda *_args, **_kwargs: smoke.PublicOASmokeResult(
            tmp_path / "public-oa-evidence.json", "failed", False
        ),
    )
    monkeypatch.setenv("PAPER_AGENT_RUN_LIVE_SMOKE", "1")
    monkeypatch.setenv("PAPER_AGENT_SMOKE_CONTACT", "mailto:operator@example.test")
    monkeypatch.setenv("PAPER_AGENT_SMOKE_UNPAYWALL_EMAIL", "operator@example.test")
    monkeypatch.setattr(sys, "argv", [str(script), "--output-dir", str(tmp_path / "output")])

    assert module.main() == 1
