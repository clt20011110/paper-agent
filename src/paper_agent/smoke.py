"""Explicit, bounded live smoke checks for public metadata integrations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

from paper_agent.domain import QuerySpec
from paper_agent.http_transport import ControlledHTTPTransport
from paper_agent.manifests import load_catalog
from paper_agent.provider_runtime import ProviderRuntime, policy_from_manifest
from paper_agent.providers.api import CrawlWindow
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


@dataclass(frozen=True, slots=True)
class VenueSmokeEvidence:
    timestamp: str
    venue_id: str
    provider: str
    year: int
    mapped_entries: int
    request_audit: tuple[Mapping[str, Any], ...]
    snapshot_files: tuple[str, ...]


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


def run_venue_smoke(
    venue_id: str,
    year: int,
    contact: str,
    output_dir: Path,
    *,
    accepted_terms: Mapping[str, str] | None = None,
    environment: Mapping[str, str] | None = None,
) -> VenueSmokeEvidence:
    """Run one exact venue descriptor and retain every metadata response."""
    catalog = load_catalog()
    descriptor = catalog.runtime_venue(venue_id)
    acceptance = catalog.acceptance(venue_id)
    transport = ControlledHTTPTransport(
        contact,
        timeout_seconds=30,
        environment=environment,
        accepted_terms=accepted_terms,
    )
    batch = create_builtin(descriptor.provider, transport).discover(
        descriptor,
        CrawlWindow(
            date_from=f"{year:04d}-01-01",
            date_to=f"{year:04d}-12-31",
            year=year,
        ),
    )
    if not batch.entries:
        raise AssertionError(f"{venue_id} smoke returned no metadata entries")
    for entry in batch.entries:
        if not entry.external_id or not entry.title:
            raise AssertionError(f"{venue_id} smoke entry lacks stable ID or title")
        if "abstract" in acceptance["required_fields"] and not entry.abstract:
            raise AssertionError(f"{venue_id} smoke entry lacks required abstract")
        if "date_filter" in acceptance["required_fields"] and not entry.publication_date:
            raise AssertionError(f"{venue_id} smoke entry lacks required publication date")
    if len(batch.request_audit) != len(transport.request_snapshots):
        raise AssertionError("venue smoke request audit and raw snapshots differ")

    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_files = []
    for index, (audit, body) in enumerate(zip(batch.request_audit, transport.request_snapshots), 1):
        if sha256(body).hexdigest() != audit["response_sha256"]:
            raise AssertionError("venue smoke response digest differs from request audit")
        path = output_dir / f"{venue_id}-{year}-response-{index:02d}.bin"
        path.write_bytes(body)
        snapshot_files.append(path.name)
    evidence = VenueSmokeEvidence(
        timestamp=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        venue_id=venue_id,
        provider=descriptor.provider,
        year=year,
        mapped_entries=len(batch.entries),
        request_audit=batch.request_audit,
        snapshot_files=tuple(snapshot_files),
    )
    (output_dir / f"{venue_id}-{year}-evidence.json").write_text(
        json.dumps(
            {"phase": 2, "purpose": "controlled venue metadata smoke", "evidence": asdict(evidence)},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return evidence
