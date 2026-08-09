from __future__ import annotations

from paper_agent.domain import QuerySpec
from paper_agent.smoke import SmokeEvidence, run_crossref_smoke, write_smoke_evidence


def test_smoke_evidence_writer_omits_volatile_result_content(tmp_path) -> None:
    evidence = SmokeEvidence("2026-08-09T00:00:00+00:00", "crossref", "https://api.crossref.org/works?rows=1", ("DOI", "title"), "a" * 64, 1)
    destination = tmp_path / "evidence.json"
    write_smoke_evidence(evidence, destination)
    rendered = destination.read_text()
    assert "no volatile totals" in rendered
    assert "Example Paper" not in rendered


def test_crossref_smoke_maps_minimum_fields(monkeypatch) -> None:
    import paper_agent.smoke as smoke

    class Transport:
        last_request_url = "https://api.crossref.org/works?rows=1"

        def __call__(self, provider, operation, parameters):
            return {"message": {"items": [{"DOI": "10.1/example", "title": ["Example"]}]}, "raw_response_artifact_hash": "b" * 64}

    monkeypatch.setattr(smoke, "ControlledHTTPTransport", lambda **_: Transport())
    evidence = run_crossref_smoke("https://example.test/contact")
    assert evidence.provider == "crossref"
    assert evidence.schema_minimum == ("DOI", "title")
