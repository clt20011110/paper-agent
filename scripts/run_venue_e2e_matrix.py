"""Run a minimal, native Stage 1--3 smoke test for one venue in the matrix.

The input paper is an approved, one-record metadata snapshot.  Stage 1 replays
that snapshot through the installed venue adapter into SQLite; Stage 2 then
uses an explicit deterministic TEST_ONLY profile; Stage 3 exercises the normal
public-direct resolver/download pipeline.  No Stage 4 or Stage 4b model is
ever dispatched by this command.
"""

from __future__ import annotations

import argparse
import base64
from datetime import UTC, datetime, timedelta
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

from jsonschema import Draft202012Validator
import yaml

from paper_agent.approved_snapshot import frozen_parameters_hash
from paper_agent.canonical import content_hash
from paper_agent.cli import _provider_specs
from paper_agent.download_cli_service import Stage3DownloadService
from paper_agent.downloads import DownloadScopeBinding, DownloadScopeSnapshotStore, ProviderTerms, build_download_scope_snapshot
from paper_agent.domain import AccessBasis, FilterStatus, PaperSource, PublicationVersion
from paper_agent.grants import GrantStore
from paper_agent.identity import source_id_for
from paper_agent.query_plan import QueryPlanStore, approve_query_plan, compile_query_plan
from paper_agent.repository import PaperRepository
from paper_agent.search_execution import execute_search_plan
from paper_agent.stage2_backends import AdjudicationDecision, RerankInput, RerankScore, ThresholdArtifact
from paper_agent.stage2_pipeline import Stage2Paper, Stage2Pipeline, Stage2Profile
from paper_agent.storage import Database


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "e2e" / "venue-smoke-matrix.yaml"
MATRIX_SCHEMA = ROOT / "schemas" / "venue-e2e-matrix.schema.json"
SCHEMA_VERSION = "paper-agent.venue-e2e-matrix.v1"
NOW = "2026-08-11T00:00:00Z"
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


class MatrixConfigError(ValueError):
    """Raised when a venue smoke configuration cannot be reproduced safely."""


class _AllRelevantTestScreener:
    """SearchPipeline-compatible test double; it never contacts a model."""

    def screen(self, paper_ids: Sequence[str]) -> Mapping[str, FilterStatus]:
        return {paper_id: FilterStatus.RELEVANT for paper_id in paper_ids}

    def reranker_score(self, paper_id: str) -> float:
        return 1.0


class _FixedReranker:
    backend_name = "venue_e2e_test_reranker"

    def rerank(self, query: str, documents: Sequence[RerankInput]) -> tuple[RerankScore, ...]:
        return tuple(RerankScore(item.paper_id, 3.0) for item in documents)


class _AllRelevantTestAdjudicator:
    backend_name = "venue_e2e_test_adjudicator"

    def adjudicate(self, request: Any) -> AdjudicationDecision:
        return AdjudicationDecision(
            request.paper_id, "relevant", 1.0, ("TEST_ONLY",),
            "Deterministic smoke-test adjudication; no model was invoked.",
            ("title",),
        )


def _canonical(document: Mapping[str, Any]) -> bytes:
    return json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256(document: Mapping[str, Any]) -> str:
    return sha256(_canonical(document)).hexdigest()


def _timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _stringify_mapping_keys(value: Any) -> Any:
    """YAML permits integer keys, but frozen JSON identities deliberately do not."""
    if isinstance(value, Mapping):
        return {str(key): _stringify_mapping_keys(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_stringify_mapping_keys(item) for item in value]
    return value


def load_matrix(path: Path) -> dict[str, Any]:
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise MatrixConfigError(f"cannot read matrix config: {path}") from error
    except yaml.YAMLError as error:
        raise MatrixConfigError(f"invalid YAML matrix config: {path}") from error
    try:
        schema = json.loads(MATRIX_SCHEMA.read_text(encoding="utf-8"))
    except OSError as error:
        raise MatrixConfigError(f"cannot read matrix schema: {MATRIX_SCHEMA}") from error
    errors = sorted(Draft202012Validator(schema).iter_errors(document), key=lambda error: list(error.path))
    if errors:
        location = ".".join(str(part) for part in errors[0].path) or "root"
        raise MatrixConfigError(f"matrix schema violation at {location}: {errors[0].message}")
    if not isinstance(document, dict) or document.get("schema_version") != SCHEMA_VERSION:
        raise MatrixConfigError(f"matrix config must use {SCHEMA_VERSION}")
    _validate_stage_contract(document["stage_contract"])
    venues = document.get("venues")
    if not isinstance(venues, list) or not venues:
        raise MatrixConfigError("matrix config must contain at least one venue")
    seen: set[str] = set()
    for venue in venues:
        _validate_venue(venue, seen)
    return document


def _validate_stage_contract(contract: Mapping[str, Any]) -> None:
    expected = {
        "stage1": {"mode": "approved_snapshot"},
        "stage2": {"mode": "TEST_ONLY", "model_invoked": False},
        "stage3": {"provider": "public_direct", "authorized_skill_allowed": False},
        "stage4": {"model": "gpt-5.6-luna", "invoke_by_default": False},
        "stage4b": {"model": "gpt-5.6-sol", "execution_strategy": "one_shot", "invoke_by_default": False},
    }
    for stage, required in expected.items():
        if any(contract.get(stage, {}).get(key) != value for key, value in required.items()):
            raise MatrixConfigError(f"matrix stage contract is invalid for {stage}")


def _validate_venue(venue: object, seen: set[str]) -> None:
    if not isinstance(venue, dict):
        raise MatrixConfigError("venue entries must be mappings")
    venue_id = venue.get("id")
    if not isinstance(venue_id, str) or not venue_id or venue_id in seen:
        raise MatrixConfigError("venue ids must be unique non-empty strings")
    seen.add(venue_id)
    if not isinstance(venue.get("year"), int):
        raise MatrixConfigError(f"{venue_id}: year must be an integer")
    for field in ("topic", "descriptor", "official_identity_url"):
        if not isinstance(venue.get(field), str) or not venue[field]:
            raise MatrixConfigError(f"{venue_id}: {field} is required")
    paper = venue.get("paper")
    if not isinstance(paper, dict):
        raise MatrixConfigError(f"{venue_id}: paper is required")
    if not isinstance(paper.get("title"), str) or not paper["title"]:
        raise MatrixConfigError(f"{venue_id}: paper.title is required")
    authors = paper.get("authors")
    if (
        not isinstance(authors, list)
        or not authors
        or any(not isinstance(author, str) or not author.strip() for author in authors)
        or len(authors) != len(set(authors))
    ):
        raise MatrixConfigError(f"{venue_id}: paper.authors must contain unique non-empty names")
    doi = paper.get("doi")
    if doi is not None and (not isinstance(doi, str) or not doi.startswith("10.") or "/" not in doi):
        raise MatrixConfigError(f"{venue_id}: paper.doi must be a bare DOI")
    arxiv_id = paper.get("arxiv_id")
    if arxiv_id is not None and (not isinstance(arxiv_id, str) or not arxiv_id):
        raise MatrixConfigError(f"{venue_id}: paper.arxiv_id must be a non-empty string")
    for field in ("metadata_source_url", "landing_url", "pdf_url"):
        value = paper.get(field)
        if not isinstance(value, str) or not value.startswith("http"):
            raise MatrixConfigError(f"{venue_id}: paper.{field} must be an HTTP URL")
    if paper.get("access_basis") not in {"official_open_access", "official_pdf_url", "legal_author_copy"}:
        raise MatrixConfigError(f"{venue_id}: unsupported public access basis")


def _venue_by_id(matrix: Mapping[str, Any], venue_id: str) -> dict[str, Any]:
    for venue in matrix["venues"]:
        if venue["id"] == venue_id:
            return venue
    raise MatrixConfigError(f"unknown venue: {venue_id}")


def _run_id(venue_id: str, supplied: str | None) -> str:
    selected = supplied or (
        f"{venue_id}-venue-smoke-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    )
    if RUN_ID_PATTERN.fullmatch(selected) is None or selected in {".", ".."}:
        raise MatrixConfigError(
            "run_id must be a plain 1-128 character identifier"
        )
    return selected


def _discovery_parameters(venue: Mapping[str, Any], descriptor: Mapping[str, Any]) -> tuple[list[tuple[str, dict[str, Any], dict[str, Any]]], dict[str, Any]]:
    """Freeze the exact adapter requests before QueryPlan binds the bundle hash."""
    provider = str(descriptor["primary_provider"])
    parameters = dict(descriptor["provider_params"])
    year = int(venue["year"])
    date_from, date_to = f"{year}-01-01", f"{year}-12-31"
    paper = venue["paper"]
    record = {
        "external_id": sha256(str(paper["landing_url"]).encode("utf-8")).hexdigest(),
        "title": paper["title"],
        "authors": list(paper["authors"]),
        "doi": paper.get("doi"),
        "arxiv_id": paper.get("arxiv_id"),
        "metadata_source_url": paper["metadata_source_url"],
        "abstract": None,
        "publication_date": f"{year}-01-01",
        "year": year,
        "venue": descriptor["name"],
        "landing_url": paper["landing_url"],
        "pdf_url": paper["pdf_url"],
        "publication_version": "published",
        "access_basis": "public_read_only",
        "license": None,
        "host_type": "official",
        "language": "en",
        "document_type": "article",
    }
    discovery = {
        **parameters,
        "venue_id": venue["id"], "adapter": provider,
        "date_from": date_from, "date_to": date_to, "year": year,
        "volume": None, "issue": None, "cursor": None,
    }
    if provider == "dblp_toc":
        # The external core adapter freezes its exact public contract rather
        # than the compatibility-only adapter/volume/issue keys used by the
        # legacy VenueBuiltinAdapter.
        discovery = {
            **parameters,
            "venue_id": venue["id"],
            "date_from": date_from,
            "date_to": date_to,
            "year": year,
            "cursor": None,
        }
    responses: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    if provider == "pmlr":
        responses.append(("resolve_volume", {"series": parameters.get("series"), "year": year}, {"status": "success", "official_url": "https://proceedings.mlr.press/v235/"}))
        discovery["volume_id"] = "v235"
    if provider == "openreview":
        responses.append(("resolve_invitation", {"venue_group": parameters["venue_group"], "year": year, "decision": "accepted"}, {"status": "success", "invitation": f"{parameters['venue_group']}/{year}/Conference/-/Blind_Submission", "api_version": "v2", "accepted_venue_ids": [f"{parameters['venue_group']}/{year}/Conference"]}))
        discovery["invitation"] = f"{parameters['venue_group']}/{year}/Conference/-/Blind_Submission"
        discovery["api_version"] = "v2"
        discovery["accepted_venue_ids"] = (f"{parameters['venue_group']}/{year}/Conference",)
        record["decision"] = "Accept"
        record["venueid"] = f"{parameters['venue_group']}/{year}/Conference"
    payload: dict[str, Any] = {"status": "success", "source_run_id": f"snapshot:{venue['id']}:{year}", "entries": [record]}
    if provider == "aaai_ojs":
        payload = {**payload, "issues": [{"id": f"{year}-smoke", "articles": [record]}]}
    elif provider == "cvf_open_access":
        payload = {**payload, "main": [record]}
    elif provider == "eda_proceedings":
        upstreams = {str(item): {"entries": [record] if index == 0 else []} for index, item in enumerate(parameters["upstreams"])}
        payload = {**payload, "upstreams": upstreams}
    elif provider == "ieee_xplore":
        payload = {**payload, "articles": [record]}
    elif provider == "springer_nature":
        payload = {**payload, "records": [record], "result": {"records": [record]}}
    elif provider == "cell_press":
        payload = {**payload, "search-results": {"entry": [record]}}
    responses.append(("discover", discovery, payload))
    return responses, record


def _snapshot_bundle(venue: Mapping[str, Any], descriptor: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    responses, record = _discovery_parameters(venue, descriptor)
    frozen = []
    for operation, parameters, body_document in responses:
        body = _canonical(body_document)
        frozen.append({
            "operation": operation,
            "parameters_hash": frozen_parameters_hash(parameters),
            "cursor": parameters.get("cursor"),
            "content_type": "application/json",
            "body_base64": base64.b64encode(body).decode("ascii"),
            "body_sha256": sha256(body).hexdigest(),
        })
    return {"schema_version": "1", "provider": descriptor["primary_provider"], "responses": frozen}, record


def _venue_spec(descriptor: Mapping[str, Any]) -> dict[str, Any]:
    """Keep native descriptor fields while intentionally disabling fallback HTTP."""
    return {
        "descriptor": dict(descriptor),
        "acceptance": {
            "schema_version": "2", "venue_id": descriptor["venue_id"],
            "primary_provider": descriptor["primary_provider"], "fallbacks": [],
        },
    }


def _terms_approvals(descriptor: Mapping[str, Any]) -> list[dict[str, str]]:
    """Freeze every restricted primary/upstream terms URL used by this replay."""
    provider = str(descriptor["primary_provider"])
    manifest = yaml.safe_load((ROOT / "providers" / f"{provider}.yaml").read_text(encoding="utf-8"))
    approvals: list[dict[str, str]] = []
    terms = manifest.get("terms", {})
    if terms.get("data_use") != "permitted" and terms.get("url"):
        approvals.append({"provider": provider, "terms_url": str(terms["url"])})
    if provider == "eda_proceedings":
        active_upstreams = {str(item) for item in descriptor["provider_params"].get("upstreams", ())}
        for name, upstream in manifest.get("upstream_policies", {}).items():
            if name not in active_upstreams:
                continue
            upstream_terms = upstream.get("terms", {})
            if upstream_terms.get("data_use") != "permitted" and upstream_terms.get("url"):
                approvals.append({"provider": f"{provider}:{name}", "terms_url": str(upstream_terms["url"])})
    return approvals


def _query_draft(venue: Mapping[str, Any], descriptor: Mapping[str, Any]) -> dict[str, Any]:
    year = int(venue["year"])
    return {
        "created_at": NOW,
        "research": {"objective": f"Smoke test: {venue['topic']}", "audience": "paper-agent engineering", "primary_question": f"Can the native {venue['id']} adapter retrieve the approved fixture?", "subquestions": []},
        "scope": {"date_from": f"{year}-01-01", "date_to": f"{year}-12-31", "venues": [venue["id"]], "fields": [], "languages": ["en"], "document_types": ["article"], "user_seeds": [], "include_arxiv_candidates": False},
        "inclusion": {"criteria": [f"Official {venue['id']} record about {venue['topic']}"], "exclusion_criteria": []},
        "query_variants": [{"id": "venue-smoke", "subquestion_id": "venue-smoke", "alias_group": "venue-smoke", "raw_query": venue["topic"], "synonyms": []}],
        "filter": {"profile": "venue-e2e-TEST_ONLY-v1", "config_hash": "1" * 64, "thresholds_hash": "2" * 64, "seed_selector_version": "venue-smoke-v1", "seed_selector_config_hash": "3" * 64, "round_state_machine_version": "venue-smoke-v1"},
        "citation_snowball": {"enabled": False, "directions": ["references", "citations"], "max_depth": 0, "max_rounds": 0, "max_per_seed_per_source": 1},
        "budgets": {"max_requests": 4, "max_candidates": 4, "max_seconds": 60, "saturation": {"min_unique_included_yield": 0.01, "consecutive_low_yield_rounds": 1}},
        "provider_policy": "all_resolved", "required_roles": ["venue_primary"], "required_providers": [],
        "terms_approvals": _terms_approvals(descriptor),
    }


def _environment_for_snapshot(descriptor: Mapping[str, Any]) -> dict[str, str]:
    manifest = yaml.safe_load((ROOT / "providers" / f"{descriptor['primary_provider']}.yaml").read_text(encoding="utf-8"))
    auth = manifest["authentication"]
    names = [auth.get("credential_env")] if auth.get("credential_env") else []
    values = auth.get("credential_envs", {})
    names.extend(values.values() if isinstance(values, Mapping) else values)
    return {str(name): "snapshot-only-no-network" for name in names if name}


def _stage2(database_path: Path, paper_ids: Sequence[str], venue: Mapping[str, Any], run_id: str, scope_hash: str) -> dict[str, Any]:
    with Database(database_path) as database:
        database.migrate()
        rows = database.connection.execute("SELECT paper_id, title, abstract FROM papers WHERE paper_id IN ({}) ORDER BY paper_id".format(",".join("?" for _ in paper_ids)), tuple(paper_ids)).fetchall()
        papers = tuple(Stage2Paper(str(row["paper_id"]), str(row["title"]), row["abstract"], document_type="article") for row in rows)
        profile = Stage2Profile(
            query=str(venue["topic"]), query_version="venue-e2e-test-v1",
            thresholds=ThresholdArtifact("venue-e2e-test-thresholds-v1", "none", "raw_reranker_score", -1.0, 2.0),
            reranker_model_id="TEST_ONLY", reranker_revision="deterministic-v1",
            adjudicator_model_id="TEST_ONLY", adjudicator_revision="not-invoked-v1",
            screening_scope_hash=scope_hash,
        )
        result = Stage2Pipeline(database, _FixedReranker(), _AllRelevantTestAdjudicator(), profile).run(run_id, papers)
    return {"mode": "TEST_ONLY", "model_invoked": False, "filter_run_id": run_id, "paper_ids": list(paper_ids), "statuses": {item.paper_id: item.status.value for item in result.decisions}, "profile_hash": profile.full_profile_hash}


def _stage3_config() -> dict[str, Any]:
    return {"download": {"resolvers": ["publisher_public", "europe_pmc", "unpaywall", "arxiv"], "providers": ["public_direct", "europe_pmc", "unpaywall_location", "arxiv", "authorized_skill", "manual"], "purpose": "personal_research", "policy_matrix": str(ROOT / "policies" / "download-access-v2.yaml"), "metadata_lookup": {"enabled": False}, "allow_rfc2544_fake_ip_dns": True}}


def _ensure_public_pdf_source(
    database_path: Path, paper_ids: Sequence[str], venue: Mapping[str, Any], descriptor: Mapping[str, Any], record: Mapping[str, Any]
) -> None:
    """Keep a direct public PDF source even when a native journal mapper drops it."""
    provider = str(descriptor["primary_provider"])
    native_external_id = str(record["external_id"])
    pdf_url = str(venue["paper"]["pdf_url"])
    # A distinct external ID preserves the immutable native provider response
    # instead of overwriting its raw metadata and capability provenance.
    external_id = f"{native_external_id}:public-pdf:{sha256(pdf_url.encode('utf-8')).hexdigest()[:16]}"
    with Database(database_path) as database:
        database.migrate()
        repository = PaperRepository(database)
        for paper_id in paper_ids:
            canonical = repository.get_paper(str(paper_id))
            if canonical is None:
                raise MatrixConfigError(f"{venue['id']}: Stage 1 returned an unknown paper ID")
            source = repository.upsert_source(PaperSource(
                source_id=source_id_for(provider, external_id), paper_id=str(paper_id),
                provider=provider, external_id=external_id,
                landing_url=str(venue["paper"]["landing_url"]), pdf_url=pdf_url,
                metadata_url=str(venue["paper"]["metadata_source_url"]),
                publication_version=PublicationVersion.PUBLISHED,
                host_type="official", access_basis=AccessBasis.PUBLIC_READ_ONLY,
                raw_metadata={
                    "source": "approved_venue_e2e_matrix",
                    "source_role": "public_pdf_locator_supplement",
                    "canonical_field_provenance": "copied_from_stage1_canonical_paper",
                    "native_external_id": native_external_id,
                    "official_identity_url": str(venue["official_identity_url"]),
                    "metadata_source_url": str(venue["paper"]["metadata_source_url"]),
                },
            ))
            # MetadataCoordinator reconstructs every paper source from field
            # provenance.  Although this record primarily supplies a public
            # PDF locator, it must remain a complete, auditable SourceEntry.
            # Copying the already-ingested canonical fields avoids inventing
            # values and the raw marker above makes that provenance basis
            # explicit.
            repository.record_field_provenance(
                canonical.paper_id,
                source.source_id,
                {
                    "title": canonical.title,
                    "abstract": canonical.abstract,
                    "authors": canonical.authors,
                    "publication_date": canonical.publication_date,
                    "year": canonical.year,
                    "venue_name": canonical.venue_name,
                    "doi": canonical.doi,
                    "arxiv_id": canonical.arxiv_id,
                    "canonical_url": canonical.canonical_url,
                },
            )


def _stage3(database_path: Path, run_dir: Path, paper_ids: Sequence[str], venue: Mapping[str, Any], filter_run_id: str, run_id: str) -> dict[str, Any]:
    host = urlsplit(str(venue["paper"]["pdf_url"])).hostname
    if not host:
        raise MatrixConfigError(f"{venue['id']}: PDF URL has no host")
    selection = build_download_scope_snapshot("selection", paper_ids, created_at=_timestamp(), snapshot_id=f"{run_id}-selection")
    selection_path = run_dir / "stage3" / "selection-snapshot.json"
    _write_json(selection_path, selection)
    with Database(database_path) as database:
        database.migrate()
        snapshot = DownloadScopeSnapshotStore(database).load_file(selection_path, expected_type="selection")
        grant_store = GrantStore(database)
        grant_draft = grant_store.create_draft(
            grant_id=f"{run_id}-public-direct-grant", kind="download", actions=["download", "store"],
            purpose="personal_research", mode="unattended", allow_unattended=True,
            scope={"paper_ids": list(paper_ids), "artifact_hashes": [], "collection_ids": [], "collection_snapshot_hash": None, "selection_snapshot_hash": snapshot.snapshot_hash, "domains": [host], "provider": "public_direct", "model": None, "data_categories": ["full_text"]},
            max_papers=len(paper_ids), expires_at=(datetime.now(UTC) + timedelta(days=1)).isoformat().replace("+00:00", "Z"),
        )
        grant = grant_store.approve(grant_draft, grant_draft["content_hash"], approved_by="venue-e2e-runner", approved_at=_timestamp())
        terms_evidence = "https://info.arxiv.org/help/license.html" if host == "arxiv.org" else str(venue["official_identity_url"])
        terms = {"public_direct": ProviderTerms("public_direct", "venue-e2e-approved-public-direct-v1", terms_evidence, True, True, True, False, (host,))}
        service = Stage3DownloadService(database, _stage3_config(), config_root=ROOT, artifact_root=run_dir / "artifacts", provider_terms=terms, scope_membership=DownloadScopeSnapshotStore(database).contains)
        result = service.run(filter_run_id=filter_run_id, authorization_grant_id=str(grant["grant_id"]), run_id=f"{run_id}-stage3", authorization_scope=DownloadScopeBinding(selection_snapshot_hash=snapshot.snapshot_hash))
    downloads = [{"paper_id": item.paper_id, "status": item.status.value, "reason_code": item.reason_code} for item in result.run.papers] if result.run else []
    expected_ids = {str(paper_id) for paper_id in paper_ids}
    observed_ids = {str(item["paper_id"]) for item in downloads}
    downloaded = (
        len(downloads) == len(expected_ids)
        and observed_ids == expected_ids
        and all(item["status"] == "downloaded" for item in downloads)
    )
    return {"provider": "public_direct", "authorized_skill_allowed": False, "stage3_run_id": result.run_id, "status": "complete" if downloaded else "failed", "paper_ids": list(result.paper_ids), "selection_snapshot_hash": snapshot.snapshot_hash, "grant_id": str(grant["grant_id"]), "downloads": downloads}


def _write_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(document) + b"\n")


def _prepared_result(venue: Mapping[str, Any], run_id: str, run_dir: Path, *, dry_run: bool) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "run_id": run_id, "run_dir": str(run_dir), "venue": venue["id"], "preflight": {"config_valid": True, "venue_descriptor": str(ROOT / venue["descriptor"]), "models_dispatched": 0, "metadata_network_requests": 0}, "dry_run": dry_run, "status": "validated" if dry_run else "prepared"}


def run(matrix: Mapping[str, Any], venue_id: str, output_root: Path, run_id: str, *, dry_run: bool, prepare_only: bool, through_stage: int) -> dict[str, Any]:
    venue = _venue_by_id(matrix, venue_id)
    descriptor_path = ROOT / venue["descriptor"]
    if not descriptor_path.is_file():
        raise MatrixConfigError(f"{venue_id}: descriptor does not exist: {descriptor_path}")
    descriptor = _stringify_mapping_keys(yaml.safe_load(descriptor_path.read_text(encoding="utf-8")))
    resolved_output_root = output_root.expanduser().resolve()
    run_dir = (resolved_output_root / run_id).resolve()
    try:
        run_dir.relative_to(resolved_output_root)
    except ValueError as error:
        raise MatrixConfigError("run directory escapes output_root") from error
    result = _prepared_result(venue, run_id, run_dir, dry_run=dry_run)
    if dry_run:
        return result
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    if prepare_only:
        result["status"] = "prepared"
        _write_json(run_dir / "run.json", result)
        return result
    bundle, record = _snapshot_bundle(venue, descriptor)
    snapshot_path = run_dir / "stage1" / "metadata-snapshot.json"
    _write_json(snapshot_path, bundle)
    digest = sha256(snapshot_path.read_bytes()).hexdigest()
    environment = _environment_for_snapshot(descriptor)
    specs = _provider_specs([{"provider": descriptor["primary_provider"], "mode": "snapshot", "snapshot_hash": digest}], ROOT, venue_ids=())
    for spec in specs:
        # This runner is deliberately venue-only.  Some primary providers also
        # advertise topic search (notably OpenReview); freezing only the role
        # exercised here prevents an unapproved search request from leaking
        # into the native venue replay.
        spec["roles"] = ["venue_primary"]
        spec["credentials_present"] = True
        spec["credential_availability"] = {name: True for name in spec["credential_environment_variables"]}
    plan = compile_query_plan(_query_draft(venue, descriptor), providers=specs, venue_specs=[_venue_spec(descriptor)], plan_id=f"{run_id}-stage1", created_at=NOW)
    approved = approve_query_plan(plan, plan["plan_hash"], approved_by="venue-e2e-runner", approved_at=NOW)
    plan_store = QueryPlanStore(run_dir)
    plan_store.save_draft(plan)
    approved_path = plan_store.save_approved(approved)
    database_path = run_dir / "papers.sqlite3"
    search_result, search_run_id, crawl_run_id = execute_search_plan(approved, database_path, run_id=f"{run_id}-stage1", snapshot_paths={descriptor["primary_provider"]: snapshot_path}, stage2_screener=_AllRelevantTestScreener(), venue_only=True, environment=environment)
    if search_result.status != "complete":
        raise MatrixConfigError(f"{venue_id}: Stage 1 replay finished with status {search_result.status}")
    if not search_result.paper_ids:
        raise MatrixConfigError(f"{venue_id}: Stage 1 replay returned no canonical papers")
    _ensure_public_pdf_source(database_path, search_result.paper_ids, venue, descriptor, record)
    stage1 = {"mode": "approved_snapshot", "snapshot_path": str(snapshot_path), "snapshot_sha256": digest, "approved_query_plan": str(approved_path), "search_run_id": search_run_id, "crawl_run_id": crawl_run_id, "paper_ids": list(search_result.paper_ids), "fixture_title": record["title"]}
    _write_json(run_dir / "stage1" / "result.json", stage1)
    result["native_pipeline"] = {
        "database": "papers.sqlite3", "artifact_root": "artifacts",
        "search_run_id": search_run_id, "crawl_run_id": crawl_run_id,
        "filter_run_id": None, "stage3_run_id": None, "stage4_run_id": None,
        "report_run_id": None, "report_pipeline_run_id": None,
    }
    if through_stage >= 2:
        stage2_run_id = f"{run_id}-stage2-test-only"
        stage2 = _stage2(database_path, search_result.paper_ids, venue, stage2_run_id, approved["filter"]["screening_scope_hash"])
        _write_json(run_dir / "stage2" / "result.json", stage2)
        result["native_pipeline"]["filter_run_id"] = stage2_run_id
    if through_stage >= 3:
        stage3 = _stage3(database_path, run_dir, search_result.paper_ids, venue, stage2_run_id, run_id)
        _write_json(run_dir / "stage3" / "result.json", stage3)
        result["native_pipeline"]["stage3_run_id"] = stage3["stage3_run_id"]
    result["stages"] = [
        {"stage": "stage1", "status": "complete"},
        {"stage": "stage2", "status": "complete" if through_stage >= 2 else "not_requested", "mode": "TEST_ONLY"},
        {"stage": "stage3", "status": stage3["status"] if through_stage >= 3 else "not_requested", "provider": "public_direct"},
        {"stage": "stage4", "status": "blocked_pending_explicit_execution", "model": "gpt-5.6-luna", "invocations": 0},
        {"stage": "stage4b", "status": "blocked_pending_explicit_execution", "model": "gpt-5.6-sol", "invocations": 0, "execution_strategy": "one_shot"},
    ]
    result["status"] = "failed" if through_stage >= 3 and stage3["status"] != "complete" else "complete"
    _write_json(run_dir / "run.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--venue", required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "paper_research_venue_e2e")
    parser.add_argument("--run-id")
    parser.add_argument("--through-stage", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--prepare-only", action="store_true", help="write only preflight metadata; do not execute native stages")
    parser.add_argument("--dry-run", action="store_true", help="validate and print the plan without writing a run directory")
    args = parser.parse_args(argv)
    try:
        matrix = load_matrix(args.config)
        run_id = _run_id(args.venue, args.run_id)
        result = run(matrix, args.venue, args.output_root, run_id, dry_run=args.dry_run, prepare_only=args.prepare_only, through_stage=args.through_stage)
    except (MatrixConfigError, FileExistsError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 1 if result.get("status") == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
