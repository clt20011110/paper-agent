from __future__ import annotations

from base64 import b64decode
from hashlib import sha256
import json
import os
from pathlib import Path

import pytest

from paper_agent.http_transport import ApprovedMetadataSnapshot, ApprovedSnapshotTransport
from paper_agent.domain import QuerySpec
from paper_agent.provider_runtime import ProviderRuntime, ProviderRuntimePolicy, SnapshotDriftError
from paper_agent.providers.builtin import create_builtin
from paper_agent.smoke import SmokeEvidence, run_crossref_smoke, run_venue_smoke, write_smoke_evidence


def test_smoke_evidence_writer_omits_volatile_result_content(tmp_path) -> None:
    evidence = SmokeEvidence("2026-08-09T00:00:00+00:00", "crossref", "https://api.crossref.org/works?rows=1", ("DOI", "title"), "a" * 64, 1)
    destination = tmp_path / "evidence.json"
    write_smoke_evidence(evidence, destination)
    rendered = destination.read_text()
    assert "no volatile totals" in rendered
    assert "Example Paper" not in rendered


def test_crossref_smoke_maps_minimum_fields_and_persists_raw_snapshot(monkeypatch, tmp_path) -> None:
    import paper_agent.smoke as smoke

    response_body = b'{"message":{"items":[{"DOI":"10.1/example","title":["Example"]}]}}'

    class Transport:
        last_request_url = "https://api.crossref.org/works?rows=1"
        last_response_body = response_body

        def __call__(self, provider, operation, parameters):
            return {
                "message": {"items": [{"DOI": "10.1/example", "title": ["Example"]}]},
                "raw_response_artifact_hash": sha256(response_body).hexdigest(),
            }

    monkeypatch.setattr(smoke, "ControlledHTTPTransport", lambda **_: Transport())
    snapshot = tmp_path / "crossref-response.json"
    evidence_path = tmp_path / "evidence.json"
    evidence = run_crossref_smoke("https://example.test/contact", snapshot_path=snapshot, evidence_path=evidence_path)

    assert evidence.provider == "crossref"
    assert evidence.schema_minimum == ("DOI", "title")
    assert snapshot.read_bytes() == response_body
    assert evidence.snapshot_file == "crossref-response.json"
    assert evidence.snapshot_bytes == len(response_body)
    assert '"snapshot_file": "crossref-response.json"' in evidence_path.read_text()


def test_venue_smoke_persists_each_native_request_and_audit(monkeypatch, tmp_path) -> None:
    import paper_agent.smoke as smoke

    response_body = b"<html>official metadata</html>"
    response_hash = sha256(response_body).hexdigest()

    class Transport:
        request_snapshots = [response_body]

        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, provider, operation, parameters):
            return {
                "entries": [{"external_id": "NeurIPS-2024-one", "title": "One", "year": 2024}],
                "raw_response_artifact_hash": response_hash,
                "_request_audit": [
                    {
                        "provider": provider,
                        "url": "https://proceedings.neurips.cc/paper_files/paper/2024",
                        "query_hash": "query-hash",
                        "cursor": None,
                        "api_version": "neurips-proceedings-html-v1",
                        "requested_at": "2026-08-10T00:00:00+00:00",
                        "completed_at": "2026-08-10T00:00:01+00:00",
                        "status": "success",
                        "response_sha256": response_hash,
                        "content_type": "text/html",
                    }
                ],
            }

    monkeypatch.setattr(smoke, "ControlledHTTPTransport", Transport)
    evidence = run_venue_smoke(
        "neurips", 2024, "operator@example.test", tmp_path
    )

    assert evidence.mapped_entries == 1
    assert evidence.snapshot_files == ("neurips-2024-response-01.bin",)
    assert (tmp_path / evidence.snapshot_files[0]).read_bytes() == response_body
    assert '"request_audit"' in (tmp_path / "neurips-2024-evidence.json").read_text()


def test_approved_json_snapshot_replays_without_network() -> None:
    body = b'{"status":"ok","message":{"items":[]}}'
    runtime = ProviderRuntime({"crossref": ProviderRuntimePolicy("crossref")})
    transport = ApprovedSnapshotTransport(
        {("crossref", "search"): ApprovedMetadataSnapshot(body, sha256(body).hexdigest(), "application/json")}, runtime
    )

    payload = transport("crossref", "search", {"query": "fixture", "cursor": "page-2"})

    assert payload["status"] == "success"
    assert payload["provider_status"] == "ok"
    assert payload["raw_response_artifact_hash"] == sha256(body).hexdigest()


def test_approved_xml_snapshot_replays_without_network() -> None:
    body = b"<root><record>one</record></root>"
    runtime = ProviderRuntime({"crossref": ProviderRuntimePolicy("crossref")})
    transport = ApprovedSnapshotTransport(
        {("crossref", "search"): ApprovedMetadataSnapshot(body, sha256(body).hexdigest(), "application/xml")}, runtime
    )

    assert transport("crossref", "search", {"query": "fixture"})["root"]["record"] == "one"


def test_approved_snapshot_rejects_digest_drift() -> None:
    runtime = ProviderRuntime({"crossref": ProviderRuntimePolicy("crossref")})
    transport = ApprovedSnapshotTransport(
        {("crossref", "search"): ApprovedMetadataSnapshot(b"{}", "0" * 64, "application/json")}, runtime
    )
    with pytest.raises(SnapshotDriftError):
        transport("crossref", "search", {"query": "fixture"})


def test_committed_phase2_smoke_snapshot_is_replayable() -> None:
    root = Path(__file__).parents[1]
    evidence = json.loads(
        (root / "docs" / "smoke" / "phase2-controlled-smoke-evidence.json").read_text()
    )
    snapshot = root / "docs" / "smoke" / evidence["evidence"]["snapshot_file"]
    assert evidence["evidence"]["snapshot_encoding"] == "base64"
    body = b64decode(snapshot.read_bytes().strip(), validate=True)

    assert evidence["snapshot_status"] == "present"
    assert len(body) == evidence["evidence"]["snapshot_bytes"]
    assert sha256(body).hexdigest() == evidence["evidence"]["response_sha256"]
    manifest = root / evidence["provider_manifest"]["path"]
    assert sha256(manifest.read_bytes()).hexdigest() == evidence["provider_manifest"]["sha256"]

    runtime = ProviderRuntime({"crossref": ProviderRuntimePolicy("crossref")})
    transport = ApprovedSnapshotTransport(
        {
            ("crossref", "search"): ApprovedMetadataSnapshot(
                body, evidence["evidence"]["response_sha256"], "application/json"
            )
        },
        runtime,
    )
    batch = create_builtin("crossref", transport).search(
        QuerySpec(1, "phase2-smoke", "machine learning", page_size=1)
    )
    assert len(batch.entries) == evidence["evidence"]["mapped_entries"]
    assert batch.entries[0].external_id
    assert batch.entries[0].title


@pytest.mark.live_smoke
@pytest.mark.skipif(
    os.environ.get("PAPER_AGENT_RUN_LIVE_SMOKE") != "1",
    reason="set PAPER_AGENT_RUN_LIVE_SMOKE=1 to permit the single Crossref request",
)
def test_live_crossref_smoke_writes_auditable_snapshot() -> None:
    """Manual-only check; excluded by default and intentionally not run in CI."""
    contact = os.environ.get("PAPER_AGENT_SMOKE_CONTACT")
    output = os.environ.get("PAPER_AGENT_SMOKE_OUTPUT_DIR")
    if not contact or not output:
        pytest.fail("live smoke requires PAPER_AGENT_SMOKE_CONTACT and PAPER_AGENT_SMOKE_OUTPUT_DIR")
    directory = Path(output)
    evidence = run_crossref_smoke(
        contact,
        snapshot_path=directory / "crossref-response.json",
        evidence_path=directory / "crossref-evidence.json",
    )
    assert evidence.mapped_entries >= 1
