from __future__ import annotations

from hashlib import sha256
import os
from pathlib import Path

import pytest

from paper_agent.http_transport import ApprovedMetadataSnapshot, ApprovedSnapshotTransport
from paper_agent.provider_runtime import ProviderRuntime, ProviderRuntimePolicy, SnapshotDriftError
from paper_agent.smoke import SmokeEvidence, run_crossref_smoke, write_smoke_evidence


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
