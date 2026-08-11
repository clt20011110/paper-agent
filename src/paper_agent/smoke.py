"""Explicit, bounded live smoke checks for public metadata integrations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

from paper_agent.config import load_config
from paper_agent.download_cli_service import (
    Stage3DownloadResult,
    Stage3DownloadService,
    load_provider_terms,
)
from paper_agent.domain import Paper, QuerySpec
from paper_agent.http_transport import ControlledHTTPTransport
from paper_agent.manifests import load_catalog
from paper_agent.provider_runtime import ProviderRuntime, policy_from_manifest
from paper_agent.providers.api import CrawlWindow
from paper_agent.providers.builtin import create_builtin, load_builtin_manifest
from paper_agent.repository import PaperRepository
from paper_agent.resources import public_oa_terms_path, release_asset_root
from paper_agent.stage3_metadata_lookup import Stage3MetadataLookup
from paper_agent.storage import Database


PUBLIC_OA_SMOKE_DOI = "10.3758/s13421-020-01060-2"
PUBLIC_OA_SMOKE_PMCID = "PMC7683441"
PUBLIC_OA_SMOKE_PAPER_ID = "paper-pmc7683441"


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


@dataclass(frozen=True, slots=True)
class PublicOASmokeResult:
    evidence_path: Path
    run_id: str
    success: bool


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


def run_public_oa_download_smoke(
    output_dir: Path,
    *,
    contact: str,
    unpaywall_email: str,
    source_commit: str,
) -> PublicOASmokeResult:
    """Run one fixed Europe PMC public-OA download through the production path.

    The output directory must be new so a failed fetch can never resume an old
    request.  The service receives no injected fetcher, lookup, or resolver
    registry: metadata lookup and PDF retrieval therefore use the normal
    ``ControlledHTTPTransport`` and ``urllib_fetch`` implementations.
    """
    if output_dir.exists():
        raise ValueError("public OA smoke output directory must not already exist")
    if not contact:
        raise ValueError("public OA smoke requires a metadata contact")
    if "@" not in unpaywall_email:
        raise ValueError("public OA smoke requires an Unpaywall email")
    if not source_commit:
        raise ValueError("public OA smoke requires the source commit")

    root = release_asset_root()
    config = load_config(root / "configs" / "smoke_supported.yaml")
    metadata = config["download"]["metadata_lookup"]
    metadata["contact"] = contact
    metadata["unpaywall_email"] = unpaywall_email
    output_dir.mkdir(parents=True)
    timestamp = datetime.now(timezone.utc).replace(microsecond=0)
    run_id = "public-oa-smoke-" + timestamp.strftime("%Y%m%dT%H%M%SZ")

    with Database(output_dir / "papers.sqlite3") as database:
        database.migrate()
        PaperRepository(database).save_paper(
            Paper(
                PUBLIC_OA_SMOKE_PAPER_ID,
                "Public OA smoke paper",
                doi=PUBLIC_OA_SMOKE_DOI,
            )
        )
        service = Stage3DownloadService(
            database,
            config,
            config_root=root,
            artifact_root=output_dir / "artifacts",
            provider_terms=load_provider_terms(public_oa_terms_path(root)),
        )
        if not isinstance(service.lookup, Stage3MetadataLookup):
            raise AssertionError("public OA smoke requires the production metadata lookup")
        if not isinstance(service.lookup.transport, ControlledHTTPTransport):
            raise AssertionError("public OA smoke requires ControlledHTTPTransport")
        result = service.run(paper_ids=[PUBLIC_OA_SMOKE_PAPER_ID], run_id=run_id)
        evidence = _public_oa_evidence(
            database, service, result, run_id, timestamp, source_commit
        )

    evidence_path = output_dir / "public-oa-evidence.json"
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return PublicOASmokeResult(evidence_path, run_id, bool(evidence["success"]))


def _public_oa_evidence(
    database: Database,
    service: Stage3DownloadService,
    result: Stage3DownloadResult,
    run_id: str,
    timestamp: datetime,
    source_commit: str,
) -> dict[str, Any]:
    row = database.connection.execute(
        """SELECT candidate_id, host, license, access_basis, provenance_json,
                  policy_decision, policy_reason_code
           FROM download_candidates
           WHERE paper_id = ? AND resolver = 'europe_pmc'""",
        (PUBLIC_OA_SMOKE_PAPER_ID,),
    ).fetchone()
    candidate = None if row is None else {
        "candidate_id": row["candidate_id"],
        "resolver": "europe_pmc",
        "host": row["host"],
        "license": row["license"],
        "access_basis": row["access_basis"],
        "pmcid": json.loads(row["provenance_json"]).get("pmcid"),
        "policy_decision": row["policy_decision"],
        "policy_reason_code": row["policy_reason_code"],
    }
    request = None if row is None else database.connection.execute(
        """SELECT request_id, status FROM fetch_requests
           WHERE candidate_id = ? AND provider = 'europe_pmc'""",
        (row["candidate_id"],),
    ).fetchone()
    attempt = None if request is None else database.connection.execute(
        """SELECT result_status, failure_category, http_status, artifact_id
           FROM download_attempts WHERE fetch_request_id = ?""",
        (request["request_id"],),
    ).fetchone()
    artifact = None if attempt is None or attempt["artifact_id"] is None else database.connection.execute(
        """SELECT artifact_id, relative_path, sha256, mime_type, byte_size
           FROM artifacts WHERE artifact_id = ?""",
        (attempt["artifact_id"],),
    ).fetchone()
    paper = result.run.for_paper(PUBLIC_OA_SMOKE_PAPER_ID) if result.run else None
    artifact_path = (
        service.artifact_root / artifact["relative_path"]
        if artifact is not None else None
    )
    success = bool(
        result.status == "complete"
        and paper is not None
        and paper.status.value == "downloaded"
        and candidate is not None
        and candidate["resolver"] == "europe_pmc"
        and candidate["host"] == "europepmc.org"
        and candidate["pmcid"] == PUBLIC_OA_SMOKE_PMCID
        and candidate["access_basis"] == "open_license"
        and candidate["policy_decision"] == "allow"
        and candidate["policy_reason_code"] == "compatible_open_license"
        and request is not None
        and request["status"] == "consumed"
        and attempt is not None
        and attempt["result_status"] == "downloaded"
        and artifact is not None
        and artifact["mime_type"] == "application/pdf"
        and len(artifact["sha256"]) == 64
        and all(character in "0123456789abcdef" for character in artifact["sha256"])
        and artifact["byte_size"] > 0
        and artifact_path is not None
        and artifact_path.is_file()
        and artifact_path.stat().st_size == artifact["byte_size"]
    )
    transport = service.lookup.transport
    return {
        "purpose": "manual public OA PDF release smoke",
        "source_commit": source_commit,
        "timestamp": timestamp.isoformat(),
        "fixed_paper": {"doi": PUBLIC_OA_SMOKE_DOI, "pmcid": PUBLIC_OA_SMOKE_PMCID},
        "production_path": {
            "metadata_transport": "ControlledHTTPTransport",
            "resolver_registry": "default",
            "pdf_fetcher": "urllib_fetch",
        },
        "metadata_requests": _sanitized_metadata_audit(transport.request_audit),
        "run": {
            "run_id": run_id,
            "stage3_status": result.status,
            "paper_status": paper.status.value if paper else None,
            "paper_reason_code": paper.reason_code if paper else None,
        },
        "candidate": candidate,
        "fetch_request": (
            {"request_id": request["request_id"], "status": request["status"]}
            if request is not None else None
        ),
        "attempt": (
            {
                "status": attempt["result_status"],
                "failure_category": attempt["failure_category"],
                "http_status": attempt["http_status"],
            }
            if attempt is not None else None
        ),
        "artifact": (
            {
                "artifact_id": artifact["artifact_id"],
                "relative_path": artifact["relative_path"],
                "sha256": artifact["sha256"],
                "mime_type": artifact["mime_type"],
                "byte_size": artifact["byte_size"],
            }
            if artifact is not None else None
        ),
        "success": success,
    }


def _sanitized_metadata_audit(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    operations = {"europe_pmc": "search", "unpaywall": "resolve", "arxiv": "search"}
    output = []
    for record in records:
        parsed = urlsplit(str(record["url"]))
        provider = str(record["provider"])
        output.append({
            "provider": provider,
            "operation": record.get("operation", operations[provider]),
            "status": record["status"],
            "url": urlunsplit((parsed.scheme, parsed.hostname or "", parsed.path, "", "")),
            "content_type": record.get("content_type"),
            "response_size_bytes": record.get("response_size_bytes"),
            "rate_limit": record.get("rate_limit", {}),
        })
    return output


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
