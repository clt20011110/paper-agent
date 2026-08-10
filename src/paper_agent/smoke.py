"""Explicit, bounded live smoke checks for public metadata integrations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

from paper_agent.domain import QuerySpec
from paper_agent.http_transport import ControlledHTTPTransport
from paper_agent.provider_runtime import ProviderRuntime, policy_from_manifest
from paper_agent.providers.builtin import create_builtin, load_builtin_manifest


@dataclass(frozen=True, slots=True)
class SmokeEvidence:
    timestamp: str
    provider: str
    api_url: str
    schema_minimum: tuple[str, ...]
    response_sha256: str
    mapped_entries: int
    snapshot_file: str | None = None
    snapshot_bytes: int | None = None


def run_crossref_smoke(
    contact: str,
    *,
    snapshot_path: Path | None = None,
    evidence_path: Path | None = None,
) -> SmokeEvidence:
    """Make exactly one Crossref metadata request and optionally persist it.

    The dedicated runtime disables retries so an opt-in smoke invocation makes
    one outbound request even when that request fails.  The returned metadata
    is never dereferenced as HTML, full text, or PDF.
    """
    if (snapshot_path is None) != (evidence_path is None):
        raise ValueError("snapshot_path and evidence_path must be provided together")
    manifest = load_builtin_manifest("crossref")
    policy = replace(
        policy_from_manifest(manifest, terms_accepted=True, robots_allowed=True), retry_attempts=1
    )
    transport = ControlledHTTPTransport(
        contact=contact,
        timeout_seconds=15,
        runtime=ProviderRuntime({"crossref": policy}),
    )
    provider = create_builtin("crossref", transport)
    batch = provider.search(QuerySpec(1, "phase2-smoke", "machine learning", page_size=1))
    if not batch.entries:
        raise AssertionError("Crossref smoke returned no metadata entries")
    entry = batch.entries[0]
    if not entry.external_id or not entry.title:
        raise AssertionError("Crossref smoke entry lacks stable ID or title")
    if not batch.raw_response_artifact_hash or not transport.last_request_url or transport.last_response_body is None:
        raise AssertionError("Crossref smoke lacks response artifact evidence")
    if sha256(transport.last_response_body).hexdigest() != batch.raw_response_artifact_hash:
        raise AssertionError("Crossref response body digest differs from batch evidence")

    evidence = SmokeEvidence(
        timestamp=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        provider="crossref",
        api_url=transport.last_request_url,
        schema_minimum=("DOI", "title"),
        response_sha256=batch.raw_response_artifact_hash,
        mapped_entries=len(batch.entries),
        snapshot_file=snapshot_path.name if snapshot_path else None,
        snapshot_bytes=len(transport.last_response_body) if snapshot_path else None,
    )
    if snapshot_path and evidence_path:
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_bytes(transport.last_response_body)
        write_smoke_evidence(evidence, evidence_path)
    return evidence


def write_smoke_evidence(evidence: SmokeEvidence, path: Path) -> None:
    """Write evidence metadata, never the potentially volatile response body."""
    document = {
        "phase": 2,
        "purpose": "controlled public metadata smoke",
        "constraints": ["one request", "page_size=1", "no credentials", "no PDF", "no volatile totals"],
        "evidence": asdict(evidence),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
