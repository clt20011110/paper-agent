from __future__ import annotations

from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path

import pytest

import paper_agent.downloads as downloads_module
from paper_agent.artifacts import ArtifactStore
from paper_agent.domain import (
    AccessBasis,
    AccessLocationCandidate,
    DownloadStatus,
    FetchDecisionStatus,
    FetchRequest,
    PublicationVersion,
)
from paper_agent.downloads import (
    AuthorizationContext,
    DownloadAccessPolicy,
    DownloadError,
    DownloadScopeSnapshotStore,
    DownloadService,
    FetchRejected,
    HTTPResponse,
    ProviderTerms,
    build_download_scope_snapshot,
)
from paper_agent.grants import GrantStore
from paper_agent.storage import Database


ROOT = Path(__file__).parents[1]
NOW = "2026-08-10T00:00:00Z"
HASH = "a" * 64


class FakeFetcher:
    def __init__(self, *responses: HTTPResponse | BaseException) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    def __call__(self, url: str) -> HTTPResponse:
        self.calls.append(url)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class SimulatedCrash(BaseException):
    pass


@pytest.fixture
def database(tmp_path) -> Database:
    value = Database(tmp_path / "papers.sqlite3")
    value.migrate()
    value.connection.execute("INSERT INTO papers(paper_id, title) VALUES ('paper-1', 'Paper')")
    value.connection.execute("INSERT INTO papers(paper_id, title) VALUES ('paper-2', 'Paper 2')")
    value.connection.execute(
        """INSERT INTO pipeline_runs(run_id, stage, status, input_hash, config_hash, implementation_version)
           VALUES ('run-1', 'stage3', 'running', 'input', 'config', 'test')"""
    )
    value.connection.commit()
    yield value
    value.close()


@pytest.fixture
def policy() -> DownloadAccessPolicy:
    return DownloadAccessPolicy.load(ROOT / "policies" / "download-access-v1.yaml")


def terms(
    *,
    machine_readable: bool = True,
    download: bool | None = True,
    storage: bool | None = True,
    redistribution: bool | None = True,
    version: str = "2026-08-10",
    domains: tuple[str, ...] = ("example.test",),
) -> ProviderTerms:
    return ProviderTerms(
        provider="public_http",
        terms_version=version,
        evidence_url="https://example.test/terms" if machine_readable else None,
        machine_readable=machine_readable,
        allows_download=download,
        allows_storage=storage,
        allows_redistribution=redistribution,
        domain_allowlist=domains,
    )


def candidate(
    suffix: str = "1",
    *,
    access_basis: AccessBasis = AccessBasis.OPEN_LICENSE,
    license: str | None = "CC-BY-4.0",
    version: PublicationVersion = PublicationVersion.PUBLISHED,
    paper_id: str = "paper-1",
) -> AccessLocationCandidate:
    return AccessLocationCandidate(
        candidate_id=f"candidate-{suffix}",
        paper_id=paper_id,
        resolver="fixture",
        url=f"https://example.test/paper-{suffix}.pdf",
        landing_url=f"https://example.test/paper-{suffix}",
        host="example.test",
        publication_version=version,
        license=license,
        access_basis=access_basis,
        retrieved_at=NOW,
        raw_evidence_hash=HASH,
        provenance={"fixture": suffix},
    )


def service(
    database: Database,
    tmp_path: Path,
    policy: DownloadAccessPolicy,
    fetcher: FakeFetcher,
    *,
    provider_terms: ProviderTerms | None = None,
    scope_membership=None,
    clock=None,
    provider_fetchers=None,
) -> DownloadService:
    registry = {} if provider_terms is None else {provider_terms.provider: provider_terms}
    return DownloadService(
        database,
        ArtifactStore(tmp_path / "store"),
        policy,
        registry,
        fetcher,
        scope_membership,
        clock or (lambda: datetime.fromisoformat(NOW.replace("Z", "+00:00"))),
        provider_fetchers,
    )


def test_provider_specific_fetcher_keeps_authorized_skill_off_public_http(
    database: Database, tmp_path: Path, policy: DownloadAccessPolicy,
) -> None:
    public = FakeFetcher(AssertionError("public fetcher must not receive authorized skill work"))
    authorized = FakeFetcher(HTTPResponse(200, {"Content-Type": "application/pdf"}, valid_pdf(), "https://example.test/paper-1.pdf"))
    authorized_terms = replace(terms(), provider="authorized_skill")
    downloader = service(
        database, tmp_path, policy, public, provider_terms=authorized_terms,
        provider_fetchers={"authorized_skill": authorized},
    )

    decision = downloader.probe(
        candidate(), purpose="personal_research", provider="authorized_skill", now=NOW,
    )
    result = downloader.fetch(decision.fetch_request, run_id="run-1", now=NOW)

    assert result.status is DownloadStatus.DOWNLOADED
    assert public.calls == []
    assert authorized.calls == ["https://example.test/paper-1.pdf"]


def valid_pdf() -> bytes:
    objects = (
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>",
    )
    payload = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, body in enumerate(objects, 1):
        offsets.append(len(payload))
        payload.extend(f"{number} 0 obj\n".encode())
        payload.extend(body + b"\nendobj\n")
    xref = len(payload)
    payload.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    payload.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        payload.extend(f"{offset:010d} 00000 n \n".encode())
    payload.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )
    return bytes(payload)


@pytest.mark.parametrize(
    ("purpose", "basis", "license_id", "provider_terms", "has_grant", "status", "reason"),
    [
        (
            "internal_analysis", AccessBasis.OPEN_LICENSE, "https://creativecommons.org/licenses/by/4.0/",
            terms(), False, FetchDecisionStatus.ALLOW, "compatible_open_license",
        ),
        (
            "personal_research", AccessBasis.PUBLIC_READ_ONLY, None,
            terms(), False, FetchDecisionStatus.NEEDS_GRANT, "explicit_download_grant_required",
        ),
        (
            "internal_analysis", AccessBasis.USER_SUBSCRIPTION, None,
            terms(), False, FetchDecisionStatus.NEEDS_GRANT, "explicit_download_grant_required",
        ),
        (
            "internal_analysis", AccessBasis.UNKNOWN, "bronze",
            terms(), False, FetchDecisionStatus.NEEDS_GRANT, "explicit_download_grant_required",
        ),
        (
            "internal_analysis", AccessBasis.USER_SUPPLIED, None,
            terms(), True, FetchDecisionStatus.ALLOW, "authorized_by_grant",
        ),
        (
            "personal_research", AccessBasis.OPEN_LICENSE, "CC-BY-4.0",
            terms(machine_readable=False), False, FetchDecisionStatus.MANUAL, "provider_terms_unmachineable",
        ),
        (
            "internal_analysis", AccessBasis.OPEN_LICENSE, "CC-BY-4.0",
            terms(download=None), False, FetchDecisionStatus.MANUAL, "provider_terms_permission_unknown",
        ),
        (
            "internal_analysis", AccessBasis.OPEN_LICENSE, "CC-BY-4.0",
            terms(storage=False), False, FetchDecisionStatus.DENY, "provider_terms_forbid_download_or_storage",
        ),
        (
            "redistribution", AccessBasis.OPEN_LICENSE, "CC-BY-NC-4.0",
            terms(), False, FetchDecisionStatus.DENY, "redistribution_requires_compatible_license",
        ),
        (
            "redistribution", AccessBasis.PUBLIC_READ_ONLY, "CC-BY-4.0",
            terms(machine_readable=False), False, FetchDecisionStatus.DENY,
            "redistribution_requires_compatible_license",
        ),
        (
            "redistribution", AccessBasis.OPEN_LICENSE, "CC-BY-4.0",
            terms(redistribution=False), False, FetchDecisionStatus.DENY,
            "provider_terms_forbid_redistribution",
        ),
        (
            "redistribution", AccessBasis.OPEN_LICENSE, "CC-BY-4.0",
            terms(), False, FetchDecisionStatus.ALLOW, "compatible_open_license",
        ),
    ],
)
def test_policy_matrix_covers_purpose_access_license_version_and_terms(
    policy: DownloadAccessPolicy,
    purpose: str,
    basis: AccessBasis,
    license_id: str | None,
    provider_terms: ProviderTerms,
    has_grant: bool,
    status: FetchDecisionStatus,
    reason: str,
) -> None:
    outcome = policy.decide(
        candidate(access_basis=basis, license=license_id),
        purpose,
        provider_terms,
        has_grant=has_grant,
    )

    assert (outcome.status, outcome.reason_code) == (status, reason)


def test_policy_classifies_every_publication_version(policy: DownloadAccessPolicy) -> None:
    outcomes = {
        version: policy.decide(
            candidate(version=version), "internal_analysis", terms(), has_grant=False
        ).status
        for version in PublicationVersion
    }

    assert outcomes == {version: FetchDecisionStatus.ALLOW for version in PublicationVersion}


def test_unversioned_creative_commons_label_is_not_promoted_to_a_compatible_license(
    policy: DownloadAccessPolicy,
) -> None:
    outcome = policy.decide(
        candidate(license="CC-BY"), "internal_analysis", terms(), has_grant=False
    )

    assert outcome.status is FetchDecisionStatus.NEEDS_GRANT


def test_probe_persists_candidate_decision_and_all_request_binding_hashes(
    database: Database, tmp_path: Path, policy: DownloadAccessPolicy
) -> None:
    fetcher = FakeFetcher(HTTPResponse(200, {"Content-Type": "application/pdf"}, valid_pdf()))
    downloads = service(database, tmp_path, policy, fetcher, provider_terms=terms())

    first = downloads.probe(
        candidate(), purpose="internal_analysis", provider="public_http", now=NOW, run_id="run-1"
    )
    second = downloads.probe(
        candidate(), purpose="internal_analysis", provider="public_http", now=NOW, run_id="run-1"
    )

    assert first.status is FetchDecisionStatus.ALLOW
    assert first.fetch_request == second.fetch_request
    assert fetcher.calls == []
    saved = database.connection.execute(
        """SELECT dc.policy_decision, dc.policy_reason_code, fr.policy_hash, fr.status
           FROM download_candidates dc JOIN fetch_requests fr USING(candidate_id)"""
    ).fetchone()
    assert (saved["policy_decision"], saved["policy_reason_code"], saved["status"]) == (
        "allow", "compatible_open_license", "ready",
    )
    assert saved["policy_hash"] == policy.hash
    assert len(first.fetch_request.idempotency_key.removeprefix("download:")) == 64
    audit = database.connection.execute("SELECT * FROM download_policy_decisions").fetchone()
    assert (audit["run_id"], audit["provider"], audit["decision"], audit["policy_hash"]) == (
        "run-1", "public_http", "allow", policy.hash,
    )


def test_policy_implementation_upgrade_revokes_ready_request_and_issues_new_fence(
    database: Database,
    tmp_path: Path,
    policy: DownloadAccessPolicy,
    monkeypatch,
) -> None:
    downloads = service(
        database, tmp_path, policy, FakeFetcher(), provider_terms=terms()
    )
    monkeypatch.setattr(
        downloads_module,
        "POLICY_IMPLEMENTATION_VERSION",
        "download-policy-evaluator-v1",
    )
    legacy = downloads.probe(
        candidate(), purpose="internal_analysis", provider="public_http", now=NOW
    ).fetch_request
    assert legacy is not None

    monkeypatch.setattr(
        downloads_module,
        "POLICY_IMPLEMENTATION_VERSION",
        "download-policy-evaluator-v2",
    )
    current = downloads.probe(
        candidate(), purpose="internal_analysis", provider="public_http", now=NOW
    ).fetch_request
    assert current is not None

    assert current.request_id != legacy.request_id
    assert current.fencing_token == legacy.fencing_token + 1
    rows = database.connection.execute(
        "SELECT request_id, status FROM fetch_requests ORDER BY fencing_token"
    ).fetchall()
    assert [tuple(row) for row in rows] == [
        (legacy.request_id, "revoked"),
        (current.request_id, "ready"),
    ]


def test_candidate_identity_and_url_host_are_immutable(
    database: Database, tmp_path: Path, policy: DownloadAccessPolicy
) -> None:
    downloads = service(database, tmp_path, policy, FakeFetcher(), provider_terms=terms())
    original = downloads.persist_candidate(candidate(), now=NOW)
    without_time = replace(candidate("2"), retrieved_at=None)
    first_without_time = downloads.persist_candidate(without_time, now=NOW)
    second_without_time = downloads.persist_candidate(
        without_time, now="2026-08-10T01:00:00Z"
    )

    assert original.host == "example.test"
    assert first_without_time == second_without_time
    with pytest.raises(DownloadError, match="immutable"):
        downloads.persist_candidate(replace(candidate(), url="https://example.test/changed.pdf"), now=NOW)
    with pytest.raises(DownloadError, match="host"):
        downloads.persist_candidate(replace(candidate("3"), host="other.test"), now=NOW)
    with pytest.raises(DownloadError, match="private or local"):
        downloads.persist_candidate(
            replace(candidate("4"), url="http://127.0.0.1/paper.pdf", host="127.0.0.1"),
            now=NOW,
        )


def test_provider_terms_must_explicitly_cover_candidate_host(
    database: Database, tmp_path: Path, policy: DownloadAccessPolicy
) -> None:
    downloads = service(
        database,
        tmp_path,
        policy,
        FakeFetcher(),
        provider_terms=terms(domains=("other.test",)),
    )

    decision = downloads.probe(
        candidate(), purpose="internal_analysis", provider="public_http", now=NOW
    )

    assert decision.status is FetchDecisionStatus.MANUAL
    assert decision.reason_code == "provider_terms_host_uncovered"


def _approved_download_grant(
    database: Database,
    *,
    grant_id: str = "grant-1",
    domain: str = "example.test",
    expires_at: str = "2026-08-10T00:10:00Z",
    paper_ids: tuple[str, ...] = ("paper-1",),
    max_papers: int = 1,
    mode: str = "attended",
    skill_digest: str | None = None,
    dependency_digest: str | None = None,
    selection_snapshot_hash: str | None = None,
) -> dict[str, object]:
    grants = GrantStore(database)
    draft = grants.create_draft(
        grant_id=grant_id,
        kind="download",
        actions=["download", "store"],
        purpose="internal_analysis",
        mode=mode,
        allow_unattended=mode == "unattended",
        scope={
            "paper_ids": list(paper_ids),
            "artifact_hashes": [],
            "collection_ids": [],
            "collection_snapshot_hash": None,
            "selection_snapshot_hash": selection_snapshot_hash,
            "domains": [domain],
            "provider": "public_http",
            "model": None,
            "data_categories": ["full_text"],
        },
        max_papers=max_papers,
        expires_at=expires_at,
        skill_digest=skill_digest,
        dependency_digest=dependency_digest,
    )
    return grants.approve(draft, draft["content_hash"], approved_by="owner", approved_at=NOW)


def test_valid_grant_requires_reprobe_and_is_hash_bound(
    database: Database, tmp_path: Path, policy: DownloadAccessPolicy
) -> None:
    fetcher = FakeFetcher(HTTPResponse(200, {"Content-Type": "application/pdf"}, valid_pdf()))
    downloads = service(database, tmp_path, policy, fetcher, provider_terms=terms())
    location = candidate(access_basis=AccessBasis.PUBLIC_READ_ONLY, license=None)
    initial = downloads.probe(
        location, purpose="internal_analysis", provider="public_http", now=NOW
    )
    approved = _approved_download_grant(database)
    allowed = downloads.probe(
        location,
        purpose="internal_analysis",
        provider="public_http",
        now=NOW,
        authorization_grant_id="grant-1",
    )

    assert initial.status is FetchDecisionStatus.NEEDS_GRANT and initial.fetch_request is None
    assert allowed.status is FetchDecisionStatus.ALLOW
    assert allowed.reason_code == "authorized_by_grant"
    assert allowed.fetch_request is not None
    assert allowed.fetch_request.authorization_grant_id == "grant-1"
    assert allowed.fetch_request.authorization_grant_hash == approved["content_hash"]
    assert allowed.fetch_request.expires_at == approved["expires_at"]
    assert fetcher.calls == []


def test_invalid_grant_stays_needs_grant_without_a_request(
    database: Database, tmp_path: Path, policy: DownloadAccessPolicy
) -> None:
    _approved_download_grant(database, domain="other.test")
    downloads = service(database, tmp_path, policy, FakeFetcher(), provider_terms=terms())

    decision = downloads.probe(
        candidate(access_basis=AccessBasis.USER_SUBSCRIPTION, license=None),
        purpose="internal_analysis",
        provider="public_http",
        now=NOW,
        authorization_grant_id="grant-1",
    )

    assert decision.status is FetchDecisionStatus.NEEDS_GRANT
    assert decision.reason_code == "download_grant_invalid"
    assert decision.fetch_request is None
    assert database.connection.execute("SELECT COUNT(*) FROM fetch_requests").fetchone()[0] == 0


def test_grant_max_papers_is_cumulative_across_issued_requests(
    database: Database, tmp_path: Path, policy: DownloadAccessPolicy
) -> None:
    _approved_download_grant(database, paper_ids=("paper-1", "paper-2"), max_papers=1)
    downloads = service(database, tmp_path, policy, FakeFetcher(), provider_terms=terms())
    first = downloads.probe(
        candidate("one", access_basis=AccessBasis.USER_SUBSCRIPTION, license=None),
        purpose="internal_analysis",
        provider="public_http",
        now=NOW,
        authorization_grant_id="grant-1",
    )
    second = downloads.probe(
        candidate(
            "two", access_basis=AccessBasis.USER_SUBSCRIPTION, license=None, paper_id="paper-2"
        ),
        purpose="internal_analysis",
        provider="public_http",
        now=NOW,
        authorization_grant_id="grant-1",
    )

    assert first.status is FetchDecisionStatus.ALLOW
    assert second.status is FetchDecisionStatus.NEEDS_GRANT
    assert second.reason_code == "download_grant_invalid"
    assert database.connection.execute(
        "SELECT COUNT(DISTINCT dc.paper_id) FROM fetch_requests fr JOIN download_candidates dc USING(candidate_id)"
    ).fetchone()[0] == 1


def test_unattended_requires_and_honors_the_grants_explicit_allow_signal(
    database: Database, tmp_path: Path, policy: DownloadAccessPolicy
) -> None:
    _approved_download_grant(database, mode="unattended")
    downloads = service(database, tmp_path, policy, FakeFetcher(), provider_terms=terms())

    decision = downloads.probe(
        candidate(access_basis=AccessBasis.USER_SUBSCRIPTION, license=None),
        purpose="internal_analysis",
        provider="public_http",
        now=NOW,
        authorization_grant_id="grant-1",
        mode="unattended",
    )

    assert decision.status is FetchDecisionStatus.ALLOW
    assert decision.fetch_request is not None


def test_selection_snapshot_grant_requires_proven_candidate_membership(
    database: Database, tmp_path: Path, policy: DownloadAccessPolicy
) -> None:
    grants = GrantStore(database)
    draft = grants.create_draft(
        grant_id="selection-grant",
        kind="download",
        actions=["download", "store"],
        purpose="internal_analysis",
        mode="attended",
        scope={
            "paper_ids": [],
            "artifact_hashes": [],
            "collection_ids": [],
            "collection_snapshot_hash": None,
            "selection_snapshot_hash": HASH,
            "domains": ["example.test"],
            "provider": "public_http",
            "model": None,
            "data_categories": ["full_text"],
        },
        max_papers=1,
        expires_at="2026-08-10T00:10:00Z",
    )
    grants.approve(draft, draft["content_hash"], approved_by="owner", approved_at=NOW)
    location = candidate(access_basis=AccessBasis.USER_SUBSCRIPTION, license=None)
    denied = service(
        database, tmp_path, policy, FakeFetcher(), provider_terms=terms()
    ).probe(
        location,
        purpose="internal_analysis",
        provider="public_http",
        now=NOW,
        authorization_grant_id="selection-grant",
        selection_snapshot_hash=HASH,
    )
    allowed = service(
        database,
        tmp_path,
        policy,
        FakeFetcher(),
        provider_terms=terms(),
        scope_membership=lambda snapshot, paper, snapshot_type, collection: (
            (snapshot, paper, snapshot_type, collection)
            == (HASH, "paper-1", "selection", None)
        ),
    ).probe(
        location,
        purpose="internal_analysis",
        provider="public_http",
        now=NOW,
        authorization_grant_id="selection-grant",
        selection_snapshot_hash=HASH,
    )

    assert denied.status is FetchDecisionStatus.NEEDS_GRANT
    assert allowed.status is FetchDecisionStatus.ALLOW


def test_every_declared_selection_scope_must_cover_the_candidate(
    database: Database, tmp_path: Path, policy: DownloadAccessPolicy
) -> None:
    _approved_download_grant(database, selection_snapshot_hash=HASH)
    downloads = service(
        database,
        tmp_path,
        policy,
        FakeFetcher(),
        provider_terms=terms(),
        scope_membership=lambda _snapshot, _paper, _type, _collection: False,
    )

    decision = downloads.probe(
        candidate(access_basis=AccessBasis.USER_SUBSCRIPTION, license=None),
        purpose="internal_analysis",
        provider="public_http",
        now=NOW,
        authorization_grant_id="grant-1",
        selection_snapshot_hash=HASH,
    )

    assert decision.status is FetchDecisionStatus.NEEDS_GRANT
    assert decision.reason_code == "download_grant_invalid"


@pytest.mark.parametrize(
    ("snapshot_type", "runtime_collection_id", "expected_status"),
    (
        ("collection", "collection-1", FetchDecisionStatus.ALLOW),
        ("collection", "collection-2", FetchDecisionStatus.NEEDS_GRANT),
        ("selection", "collection-1", FetchDecisionStatus.NEEDS_GRANT),
    ),
)
def test_collection_snapshot_only_grant_binds_type_and_runtime_collection(
    database: Database,
    tmp_path: Path,
    policy: DownloadAccessPolicy,
    snapshot_type: str,
    runtime_collection_id: str,
    expected_status: FetchDecisionStatus,
) -> None:
    database.connection.execute(
        """INSERT INTO collections(collection_id, name, collection_type)
           VALUES ('collection-1', 'Collection 1', 'seed_set')"""
    )
    database.connection.execute(
        """INSERT INTO paper_collections(
               paper_id, collection_id, membership_status
           ) VALUES ('paper-1', 'collection-1', 'official_confirmed')"""
    )
    database.connection.commit()
    document = build_download_scope_snapshot(
        snapshot_type,
        ["paper-1"],
        collection_id="collection-1" if snapshot_type == "collection" else None,
        created_at=NOW,
    )
    snapshot_path = tmp_path / f"{snapshot_type}-snapshot.json"
    snapshot_path.write_text(json.dumps(document), encoding="utf-8")
    snapshot_store = DownloadScopeSnapshotStore(database)
    snapshot = snapshot_store.load_file(
        snapshot_path, expected_type=snapshot_type
    )
    grants = GrantStore(database)
    draft = grants.create_draft(
        grant_id="collection-snapshot-grant",
        kind="download",
        actions=["download", "store"],
        purpose="internal_analysis",
        mode="attended",
        scope={
            "paper_ids": [],
            "artifact_hashes": [],
            "collection_ids": [],
            "collection_snapshot_hash": snapshot.snapshot_hash,
            "selection_snapshot_hash": None,
            "domains": ["example.test"],
            "provider": "public_http",
            "model": None,
            "data_categories": ["full_text"],
        },
        max_papers=1,
        expires_at="2026-08-10T00:10:00Z",
    )
    grants.approve(draft, draft["content_hash"], approved_by="owner", approved_at=NOW)
    decision = service(
        database,
        tmp_path,
        policy,
        FakeFetcher(),
        provider_terms=terms(),
        scope_membership=snapshot_store.contains,
    ).probe(
        candidate(access_basis=AccessBasis.USER_SUBSCRIPTION, license=None),
        purpose="internal_analysis",
        provider="public_http",
        now=NOW,
        authorization_grant_id="collection-snapshot-grant",
        collection_id=runtime_collection_id,
        collection_snapshot_hash=snapshot.snapshot_hash,
    )

    assert decision.status is expected_status


@pytest.mark.parametrize(
    ("snapshot_type", "collection_id"),
    (("collection", None), ("selection", "collection-1")),
)
def test_download_scope_snapshot_schema_binds_type_to_collection_id(
    snapshot_type: str, collection_id: str | None
) -> None:
    with pytest.raises(ValueError):
        build_download_scope_snapshot(
            snapshot_type,
            ["paper-1"],
            collection_id=collection_id,
            created_at=NOW,
        )


def test_fetch_reproves_runtime_skill_and_dependency_digests(
    database: Database, tmp_path: Path, policy: DownloadAccessPolicy
) -> None:
    dependency = "b" * 64
    _approved_download_grant(database, skill_digest=HASH, dependency_digest=dependency)
    fetcher = FakeFetcher(HTTPResponse(200, {"Content-Type": "application/pdf"}, valid_pdf()))
    downloads = service(database, tmp_path, policy, fetcher, provider_terms=terms())
    request = downloads.probe(
        candidate(access_basis=AccessBasis.USER_SUBSCRIPTION, license=None),
        purpose="internal_analysis",
        provider="public_http",
        now=NOW,
        authorization_grant_id="grant-1",
        skill_digest=HASH,
        dependency_digest=dependency,
    ).fetch_request
    assert request is not None

    with pytest.raises(FetchRejected, match="hash drifted"):
        downloads.fetch(request, run_id="run-1", now=NOW)
    result = downloads.fetch(
        request,
        run_id="run-1",
        now=NOW,
        authorization_context=AuthorizationContext(
            skill_digest=HASH, dependency_digest=dependency
        ),
    )

    assert result.status is DownloadStatus.DOWNLOADED
    assert fetcher.calls == [candidate().url]


def test_fetch_rejects_constructed_mismatched_expired_and_hash_drift_before_network(
    database: Database, tmp_path: Path, policy: DownloadAccessPolicy
) -> None:
    fetcher = FakeFetcher(HTTPResponse(200, {"Content-Type": "application/pdf"}, valid_pdf()))
    downloads = service(database, tmp_path, policy, fetcher, provider_terms=terms())
    request = downloads.probe(
        candidate(), purpose="internal_analysis", provider="public_http", now=NOW
    ).fetch_request
    assert request is not None

    constructed = FetchRequest(
        request_id="fetch-constructed",
        candidate_id=request.candidate_id,
        policy_version=request.policy_version,
        purpose=request.purpose,
        provider=request.provider,
        created_at=request.created_at,
        expires_at=request.expires_at,
        idempotency_key="constructed",
        fencing_token=request.fencing_token,
    )
    with pytest.raises(FetchRejected, match="not issued"):
        downloads.fetch(constructed, run_id="run-1", now=NOW)
    with pytest.raises(FetchRejected, match="fields"):
        downloads.fetch(replace(request, purpose="personal_research"), run_id="run-1", now=NOW)

    database.connection.execute(
        "UPDATE download_candidates SET url = 'https://example.test/drifted.pdf' WHERE candidate_id = 'candidate-1'"
    )
    with pytest.raises(FetchRejected, match="hash drifted"):
        downloads.fetch(request, run_id="run-1", now=NOW)
    database.connection.execute(
        "UPDATE download_candidates SET url = 'https://example.test/paper-1.pdf' WHERE candidate_id = 'candidate-1'"
    )
    database.connection.commit()
    with pytest.raises(FetchRejected, match="expired"):
        downloads.fetch(request, run_id="run-1", now="2026-08-10T00:15:01Z")
    assert fetcher.calls == []


def test_terms_or_grant_revocation_invalidates_ready_request_before_network(
    database: Database, tmp_path: Path, policy: DownloadAccessPolicy
) -> None:
    fetcher = FakeFetcher(HTTPResponse(200, {"Content-Type": "application/pdf"}, valid_pdf()))
    downloads = service(database, tmp_path, policy, fetcher, provider_terms=terms())
    open_request = downloads.probe(
        candidate("open"), purpose="internal_analysis", provider="public_http", now=NOW
    ).fetch_request
    assert open_request is not None
    downloads.provider_terms["public_http"] = terms(version="2026-08-11")
    with pytest.raises(FetchRejected, match="terms hash drifted"):
        downloads.fetch(open_request, run_id="run-1", now=NOW)

    downloads.provider_terms["public_http"] = terms()
    _approved_download_grant(database)
    granted_request = downloads.probe(
        candidate("grant", access_basis=AccessBasis.PUBLIC_READ_ONLY, license=None),
        purpose="internal_analysis",
        provider="public_http",
        now=NOW,
        authorization_grant_id="grant-1",
    ).fetch_request
    assert granted_request is not None
    GrantStore(database).revoke("grant-1", actor="owner", event_at="2026-08-10T00:01:00Z")
    with pytest.raises(FetchRejected, match="revoked"):
        downloads.fetch(granted_request, run_id="run-1", now="2026-08-10T00:02:00Z")
    assert database.connection.execute(
        "SELECT status FROM fetch_requests WHERE request_id = ?", (granted_request.request_id,)
    ).fetchone()[0] == "revoked"
    assert fetcher.calls == []


def test_successful_fetch_validates_saves_atomically_and_is_idempotent(
    database: Database, tmp_path: Path, policy: DownloadAccessPolicy
) -> None:
    payload = valid_pdf()
    fetcher = FakeFetcher(HTTPResponse(200, {"content-type": "application/pdf; charset=binary"}, payload))
    downloads = service(database, tmp_path, policy, fetcher, provider_terms=terms())
    request = downloads.probe(
        candidate(), purpose="internal_analysis", provider="public_http", now=NOW
    ).fetch_request
    assert request is not None

    first = downloads.fetch(request, run_id="run-1", now=NOW)
    second = downloads.fetch(request, run_id="run-1", now="2026-08-10T00:01:00Z")

    assert first.status is DownloadStatus.DOWNLOADED
    assert second == first
    assert fetcher.calls == [candidate().url]
    artifact = database.connection.execute(
        "SELECT * FROM artifacts WHERE artifact_id = ?", (first.artifact_id,)
    ).fetchone()
    assert (artifact["mime_type"], artifact["byte_size"], artifact["source_url"]) == (
        "application/pdf", len(payload), candidate().url,
    )
    assert (tmp_path / "store" / artifact["relative_path"]).read_bytes() == payload
    assert database.connection.execute(
        "SELECT result_status FROM download_attempts"
    ).fetchone()[0] == "downloaded"
    assert database.connection.execute(
        "SELECT status FROM fetch_requests WHERE request_id = ?", (request.request_id,)
    ).fetchone()[0] == "consumed"


def test_a_later_probe_for_another_purpose_does_not_invalidate_bound_request(
    database: Database, tmp_path: Path, policy: DownloadAccessPolicy
) -> None:
    fetcher = FakeFetcher(HTTPResponse(200, {"Content-Type": "application/pdf"}, valid_pdf()))
    downloads = service(database, tmp_path, policy, fetcher, provider_terms=terms())
    location = candidate(license="CC-BY-NC-4.0")
    internal = downloads.probe(
        location, purpose="internal_analysis", provider="public_http", now=NOW
    )
    redistribution = downloads.probe(
        location, purpose="redistribution", provider="public_http", now=NOW
    )
    assert internal.fetch_request is not None
    assert redistribution.status is FetchDecisionStatus.DENY

    result = downloads.fetch(internal.fetch_request, run_id="run-1", now=NOW)

    assert result.status is DownloadStatus.DOWNLOADED


def test_cross_host_redirect_result_is_rejected_before_artifact_storage(
    database: Database, tmp_path: Path, policy: DownloadAccessPolicy
) -> None:
    response = HTTPResponse(
        200,
        {"Content-Type": "application/pdf"},
        valid_pdf(),
        "https://cdn.other.test/paper.pdf",
    )
    downloads = service(database, tmp_path, policy, FakeFetcher(response), provider_terms=terms())
    request = downloads.probe(
        candidate(), purpose="internal_analysis", provider="public_http", now=NOW
    ).fetch_request
    assert request is not None

    result = downloads.fetch(request, run_id="run-1", now=NOW)

    assert (result.status, result.error_code) == (
        DownloadStatus.FAILED_TERMINAL, "cross_host_redirect_denied",
    )
    assert database.connection.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0] == 0


@pytest.mark.parametrize(
    ("response", "error_code"),
    [
        (HTTPResponse(200, {"Content-Type": "text/html"}, b"<html>denied</html>" * 20), "invalid_pdf_mime"),
        (HTTPResponse(200, {"Content-Type": "application/pdf"}, b"not a pdf" * 20), "invalid_pdf_magic"),
        (HTTPResponse(200, {"Content-Type": "application/pdf"}, b"%PDF-1.7\n" + b"x" * 150), "invalid_pdf_structure"),
        (HTTPResponse(200, {"Content-Type": "application/pdf"}, b"%PDF-1.7\n%%EOF"), "pdf_too_small"),
    ],
)
def test_mime_magic_size_and_parseability_fail_terminal_without_artifact(
    database: Database,
    tmp_path: Path,
    policy: DownloadAccessPolicy,
    response: HTTPResponse,
    error_code: str,
) -> None:
    downloads = service(database, tmp_path, policy, FakeFetcher(response), provider_terms=terms())
    request = downloads.probe(
        candidate(), purpose="internal_analysis", provider="public_http", now=NOW
    ).fetch_request
    assert request is not None

    result = downloads.fetch(request, run_id="run-1", now=NOW)

    assert (result.status, result.error_code) == (DownloadStatus.FAILED_TERMINAL, error_code)
    assert database.connection.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0] == 0
    assert database.connection.execute(
        "SELECT result_status, failure_category FROM download_attempts"
    ).fetchone()[:] == ("failed_terminal", error_code)


@pytest.mark.parametrize(
    ("status_code", "status", "attempt_status", "category"),
    [
        (503, DownloadStatus.FAILED_RETRYABLE, "failed_retryable", "http_503"),
        (429, DownloadStatus.FAILED_RETRYABLE, "failed_retryable", "http_429"),
        (206, DownloadStatus.FAILED_RETRYABLE, "failed_retryable", "partial_http_response"),
        (403, DownloadStatus.AUTH_REQUIRED, "auth_required", "auth_required"),
        (404, DownloadStatus.NOT_AVAILABLE, "not_available", "not_available"),
        (400, DownloadStatus.FAILED_TERMINAL, "failed_terminal", "http_400"),
    ],
)
def test_http_failures_have_explicit_retryable_terminal_auth_and_unavailable_states(
    database: Database,
    tmp_path: Path,
    policy: DownloadAccessPolicy,
    status_code: int,
    status: DownloadStatus,
    attempt_status: str,
    category: str,
) -> None:
    downloads = service(
        database, tmp_path, policy, FakeFetcher(HTTPResponse(status_code, {}, b"")),
        provider_terms=terms(),
    )
    request = downloads.probe(
        candidate(), purpose="internal_analysis", provider="public_http", now=NOW
    ).fetch_request
    assert request is not None

    result = downloads.fetch(request, run_id="run-1", now=NOW)

    assert (result.status, result.error_code) == (status, category)
    saved = database.connection.execute(
        "SELECT result_status, failure_category, http_status FROM download_attempts"
    ).fetchone()
    assert tuple(saved) == (attempt_status, category, status_code)


def test_network_failure_can_be_resumed_with_a_new_fencing_token(
    database: Database, tmp_path: Path, policy: DownloadAccessPolicy
) -> None:
    fetcher = FakeFetcher(
        TimeoutError("timed out"),
        HTTPResponse(200, {"Content-Type": "application/pdf"}, valid_pdf()),
    )
    downloads = service(database, tmp_path, policy, fetcher, provider_terms=terms())
    first_request = downloads.probe(
        candidate(), purpose="internal_analysis", provider="public_http", now=NOW
    ).fetch_request
    assert first_request is not None

    failed = downloads.fetch(first_request, run_id="run-1", now=NOW)
    replay = downloads.fetch(first_request, run_id="run-1", now="2026-08-10T00:00:30Z")
    second_request = downloads.reissue_retryable(
        first_request.request_id, now="2026-08-10T00:01:00Z"
    )
    duplicate_reissue = downloads.reissue_retryable(
        first_request.request_id, now="2026-08-10T00:01:00Z"
    )
    completed = downloads.fetch(second_request, run_id="run-1", now="2026-08-10T00:01:00Z")

    assert failed.status is DownloadStatus.FAILED_RETRYABLE
    assert replay == failed
    assert second_request.fencing_token == first_request.fencing_token + 1
    assert duplicate_reissue == second_request
    assert second_request.idempotency_key != first_request.idempotency_key
    assert completed.status is DownloadStatus.DOWNLOADED
    assert fetcher.calls == [candidate().url, candidate().url]
    assert database.connection.execute("SELECT COUNT(*) FROM fetch_requests").fetchone()[0] == 2


def test_expired_pending_attempt_is_recoverable_after_process_interruption(
    database: Database, tmp_path: Path, policy: DownloadAccessPolicy
) -> None:
    fetcher = FakeFetcher(
        SimulatedCrash(),
        HTTPResponse(200, {"Content-Type": "application/pdf"}, valid_pdf()),
    )
    downloads = service(database, tmp_path, policy, fetcher, provider_terms=terms())
    first_request = downloads.probe(
        candidate(), purpose="internal_analysis", provider="public_http", now=NOW
    ).fetch_request
    assert first_request is not None

    with pytest.raises(SimulatedCrash):
        downloads.fetch(first_request, run_id="run-1", now=NOW)
    with pytest.raises(FetchRejected, match="execution window"):
        downloads.reissue_retryable(first_request.request_id, now="2026-08-10T00:14:59Z")
    second_request = downloads.reissue_retryable(
        first_request.request_id, now="2026-08-10T00:15:00Z"
    )
    result = downloads.fetch(second_request, run_id="run-1", now="2026-08-10T00:15:00Z")

    assert second_request.fencing_token == first_request.fencing_token + 1
    assert result.status is DownloadStatus.DOWNLOADED
    interrupted = database.connection.execute(
        """SELECT result_status, failure_category FROM download_attempts
           WHERE fetch_request_id = ?""",
        (first_request.request_id,),
    ).fetchone()
    assert tuple(interrupted) == ("failed_retryable", "interrupted")


def test_late_worker_cannot_commit_after_a_new_fencing_token_is_issued(
    database: Database, tmp_path: Path, policy: DownloadAccessPolicy
) -> None:
    holder = {}
    issued = {}

    def superseding_fetch(url: str) -> HTTPResponse:
        database.connection.execute(
            """UPDATE download_attempts
               SET result_status = 'failed_retryable', failure_category = 'interrupted'
               WHERE fetch_request_id = ?""",
            (holder["request"].request_id,),
        )
        database.connection.commit()
        issued["request"] = holder["service"].reissue_retryable(
            holder["request"].request_id, now="2026-08-10T00:01:00Z"
        )
        return HTTPResponse(200, {"Content-Type": "application/pdf"}, valid_pdf())

    downloads = service(database, tmp_path, policy, superseding_fetch, provider_terms=terms())
    request = downloads.probe(
        candidate(), purpose="internal_analysis", provider="public_http", now=NOW
    ).fetch_request
    assert request is not None
    holder.update(service=downloads, request=request)

    result = downloads.fetch(request, run_id="run-1", now=NOW)

    assert (result.status, result.error_code) == (
        DownloadStatus.FAILED_TERMINAL, "stale_fencing_token",
    )
    assert issued["request"].fencing_token == request.fencing_token + 1
    assert database.connection.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0] == 0
    interrupted = database.connection.execute(
        "SELECT result_status, failure_category FROM download_attempts"
    ).fetchone()
    assert tuple(interrupted) == ("failed_retryable", "interrupted")


def test_expiry_during_artifact_validation_is_checked_with_a_fresh_clock(
    database: Database, tmp_path: Path, policy: DownloadAccessPolicy
) -> None:
    moments = iter(
        (
            datetime.fromisoformat(NOW.replace("Z", "+00:00")),
            datetime.fromisoformat("2026-08-10T00:15:00+00:00"),
        )
    )
    downloads = service(
        database,
        tmp_path,
        policy,
        FakeFetcher(HTTPResponse(200, {"Content-Type": "application/pdf"}, valid_pdf())),
        provider_terms=terms(),
        clock=lambda: next(moments),
    )
    request = downloads.probe(
        candidate(), purpose="internal_analysis", provider="public_http", now=NOW
    ).fetch_request
    assert request is not None

    result = downloads.fetch(request, run_id="run-1", now=NOW)

    assert (result.status, result.error_code) == (
        DownloadStatus.FAILED_RETRYABLE,
        "fetch_request_expired_during_fetch",
    )
    assert database.connection.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0] == 0


def test_revocation_during_network_call_prevents_artifact_commit(
    database: Database, tmp_path: Path, policy: DownloadAccessPolicy
) -> None:
    _approved_download_grant(database)

    def revoking_fetch(url: str) -> HTTPResponse:
        GrantStore(database).revoke(
            "grant-1", actor="owner", event_at="2026-08-10T00:00:01Z"
        )
        return HTTPResponse(200, {"Content-Type": "application/pdf"}, valid_pdf())

    downloads = service(database, tmp_path, policy, revoking_fetch, provider_terms=terms())
    request = downloads.probe(
        candidate(access_basis=AccessBasis.USER_SUBSCRIPTION, license=None),
        purpose="internal_analysis",
        provider="public_http",
        now=NOW,
        authorization_grant_id="grant-1",
    ).fetch_request
    assert request is not None

    result = downloads.fetch(request, run_id="run-1", now=NOW)

    assert (result.status, result.error_code) == (
        DownloadStatus.MANUAL_REQUIRED, "authorization_changed_during_fetch",
    )
    assert database.connection.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0] == 0


def test_unmachineable_or_missing_terms_never_create_fetch_request(
    database: Database, tmp_path: Path, policy: DownloadAccessPolicy
) -> None:
    no_terms = service(database, tmp_path, policy, FakeFetcher())
    missing = no_terms.probe(
        candidate("missing"), purpose="internal_analysis", provider="public_http", now=NOW
    )
    opaque = service(
        database, tmp_path, policy, FakeFetcher(), provider_terms=terms(machine_readable=False)
    ).probe(candidate("opaque"), purpose="internal_analysis", provider="public_http", now=NOW)

    assert missing.status is FetchDecisionStatus.MANUAL and missing.fetch_request is None
    assert opaque.status is FetchDecisionStatus.MANUAL and opaque.fetch_request is None
    assert database.connection.execute("SELECT COUNT(*) FROM fetch_requests").fetchone()[0] == 0
