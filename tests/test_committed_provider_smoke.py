from __future__ import annotations

from base64 import b64decode
from hashlib import sha256
import json
from pathlib import Path
from urllib.parse import urlsplit

from paper_agent.approval import require_valid_approval
from paper_agent.canonical import content_hash
from paper_agent.manifests import load_catalog
from paper_agent.query_plan import assert_screening_scope_hash
from paper_agent.schema import validate


PREFIX = "crossref-full-pipeline-20260811"


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_committed_crossref_full_pipeline_smoke_evidence() -> None:
    root = Path(__file__).parents[1]
    smoke = root / "docs" / "smoke"
    evidence = _json(smoke / f"{PREFIX}-evidence.json")

    assert evidence["source_commit"] == "64ee6c0e434fe1e7a75ec6397d4af0d807ce63b9"
    assert evidence["pipeline"]["status"] == "complete"
    assert evidence["pipeline"]["fanout_incomplete"] is False
    assert evidence["pipeline"]["budget_exhausted"] is False
    assert "not Stage 2 release evidence" in evidence["pipeline"]["stage2_screener"]

    constraints = evidence["constraints"]
    assert constraints["actual_outbound_requests"] == 3
    assert constraints["outbound_request_limit"] == 3
    assert constraints["expected_operations"] == ["search", "enrich", "verify"]
    assert constraints["transport_retry_attempts"] == 1
    assert constraints["redirects_followed"] is False
    assert constraints["pdf_requested"] is False
    assert evidence["safety"]["no_pdf_request_observed"] is True
    history = evidence["network_history"]
    assert history["session_cumulative_outbound_requests"] == (
        history["current_run_outbound_requests"]
        + sum(item["actual_outbound_requests"] for item in history["prior_attempts"])
    )

    plan_record = evidence["query_plan"]
    plan_path = smoke / plan_record["path"]
    assert _sha256(plan_path) == plan_record["file_sha256"]
    plan = _json(plan_path)
    validate(plan, "query-plan.schema.json")
    require_valid_approval(plan, "plan_hash")
    assert_screening_scope_hash(plan)
    assert plan["plan_hash"] == plan_record["plan_hash"]
    assert plan["page_size"] == 1
    assert plan["scope"]["fields"] == []

    manifest_record = evidence["provider_manifest"]
    manifest_path = root / manifest_record["path"]
    assert _sha256(manifest_path) == manifest_record["raw_file_sha256"]
    assert content_hash(load_catalog(root).provider("crossref")) == manifest_record[
        "canonical_manifest_hash"
    ]

    for record_name in ("response_manifest",):
        path = smoke / evidence["network"][f"{record_name}_path"]
        assert _sha256(path) == evidence["network"][f"{record_name}_sha256"]
    search_audit_path = smoke / evidence["search_audit"]["path"]
    transport_audit_path = smoke / evidence["transport_audit"]["path"]
    assert _sha256(search_audit_path) == evidence["search_audit"]["file_sha256"]
    assert _sha256(transport_audit_path) == evidence["transport_audit"]["file_sha256"]

    response_manifest = _json(
        smoke / evidence["network"]["response_manifest_path"]
    )
    responses = response_manifest["responses"]
    assert responses == evidence["network"]["responses"]
    assert [item["operation"] for item in responses] == ["search", "enrich", "verify"]
    for response in responses:
        body = b64decode(
            (smoke / response["committed_snapshot_file"]).read_bytes().strip(),
            validate=True,
        )
        assert len(body) == response["response_size_bytes"]
        assert sha256(body).hexdigest() == response["response_sha256"]
        assert json.loads(body)["status"] == "ok"
        url = urlsplit(response["url"])
        assert (url.scheme, url.hostname) == ("https", "api.crossref.org")
        assert url.path == "/works" or url.path.startswith("/works/")
        assert "pdf" not in response["url"].casefold()

    search_audit = _json(search_audit_path)
    assert search_audit["status"] == "complete"
    assert search_audit["totals"]["provider_request_attempts"] == 3
    assert search_audit["totals"]["requests_made"] == 3
    assert all(item["join_valid"] for item in evidence["search_audit"]["attempt_response_join"])

    transport_audit = _json(transport_audit_path)
    assert [item["operation"] for item in transport_audit["operations"]] == [
        "search",
        "enrich",
        "verify",
    ]
    assert len(transport_audit["requests"]) == 3
    assert all(item["status"] == "success" for item in transport_audit["requests"])


def test_committed_public_oa_default_transport_failure_evidence() -> None:
    evidence = _json(
        Path(__file__).parents[1]
        / "docs"
        / "smoke"
        / "public-oa-default-20260811-failed.json"
    )

    assert evidence["source_commit"] == "570a28a7e3e357ad9bf8d0484e0bcb6e3d667e89"
    assert evidence["success"] is False
    assert evidence["production_path"] == {
        "metadata_transport": "ControlledHTTPTransport",
        "pdf_fetcher": "urllib_fetch",
        "resolver_registry": "default",
    }
    assert evidence["candidate"] == {
        "access_basis": "open_license",
        "candidate_id": "location-76145a707867addc3a7c94ea230b64e6",
        "host": "europepmc.org",
        "license": "cc by",
        "pmcid": "PMC7683441",
        "policy_decision": "allow",
        "policy_reason_code": "compatible_open_license",
        "resolver": "europe_pmc",
    }
    assert evidence["fetch_request"]["status"] == "consumed"
    assert evidence["attempt"] == {
        "failure_category": "network_error",
        "http_status": None,
        "status": "failed_retryable",
    }
    assert evidence["run"] == {
        "paper_reason_code": "manual_queue_required",
        "paper_status": "manual_required",
        "run_id": "public-oa-smoke-20260811T040226Z",
        "stage3_status": "manual_required",
    }
    assert evidence["artifact"] is None
    assert [item["provider"] for item in evidence["metadata_requests"]] == [
        "europe_pmc",
        "unpaywall",
    ]
    for request in evidence["metadata_requests"]:
        assert urlsplit(request["url"]).query == ""
        assert "@" not in request["url"]
