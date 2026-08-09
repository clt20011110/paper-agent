"""Bounded live smoke checks for public metadata integrations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

from paper_agent.domain import QuerySpec
from paper_agent.http_transport import ControlledHTTPTransport
from paper_agent.providers.builtin import create_builtin


@dataclass(frozen=True, slots=True)
class SmokeEvidence:
    timestamp: str
    provider: str
    api_url: str
    schema_minimum: tuple[str, ...]
    response_sha256: str
    mapped_entries: int


def run_crossref_smoke(contact: str) -> SmokeEvidence:
    """Issue exactly one low-QPS Crossref metadata request and validate mapping."""

    transport = ControlledHTTPTransport(contact=contact, timeout_seconds=15)
    provider = create_builtin("crossref", transport)
    batch = provider.search(QuerySpec(1, "phase2-smoke", "machine learning", page_size=1))
    if not batch.entries:
        raise AssertionError("Crossref smoke returned no metadata entries")
    entry = batch.entries[0]
    if not entry.external_id or not entry.title:
        raise AssertionError("Crossref smoke entry lacks stable ID or title")
    if not batch.raw_response_artifact_hash or not transport.last_request_url:
        raise AssertionError("Crossref smoke lacks response artifact evidence")
    return SmokeEvidence(
        timestamp=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        provider="crossref",
        api_url=transport.last_request_url,
        schema_minimum=("DOI", "title"),
        response_sha256=batch.raw_response_artifact_hash,
        mapped_entries=len(batch.entries),
    )


def write_smoke_evidence(evidence: SmokeEvidence, path: Path) -> None:
    """Write only stable smoke facts; live totals and result titles are omitted."""

    document = {
        "phase": 2,
        "purpose": "controlled public metadata smoke",
        "constraints": ["one request", "page_size=1", "no credentials", "no PDF", "no volatile totals"],
        "evidence": asdict(evidence),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
