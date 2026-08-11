"""Policy-gated, SQLite-backed public PDF downloads.

Resolvers only create :class:`AccessLocationCandidate` values.  This module
persists those untrusted locations, probes the versioned access policy without
fetching content, and issues an immutable request only after an ``allow``
decision.  Fetching consumes that persisted request before making a network
call and writes validated PDFs to the content-addressed artifact store.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.message import Message
from io import BytesIO
import ipaddress
import json
from pathlib import Path
import re
import socket
from types import MappingProxyType
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

import yaml
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from .artifacts import ArtifactStore
from .canonical import content_hash
from .domain import (
    AccessBasis,
    AccessLocationCandidate,
    DownloadResult,
    DownloadStatus,
    FetchDecision,
    FetchDecisionStatus,
    FetchRequest,
    PublicationVersion,
)
from .grants import ActiveGrant, GrantError, GrantStore
from .schema import validate
from .storage import Database


POLICY_IMPLEMENTATION_VERSION = "download-policy-evaluator-v2"


class DownloadError(ValueError):
    """Base error for invalid download state or configuration."""


class PolicyError(DownloadError):
    """The download policy is invalid or cannot evaluate an input."""


class FetchRejected(DownloadError):
    """A fetch request failed its pre-network persisted-state checks."""


@dataclass(frozen=True, slots=True)
class DownloadScopeSnapshot:
    """An immutable set of repository paper IDs used by a download grant."""

    snapshot_id: str
    snapshot_type: str
    snapshot_hash: str
    collection_id: str | None
    paper_ids: tuple[str, ...]
    created_at: str

    @property
    def core(self) -> dict[str, object]:
        return {
            "schema_version": "1",
            "snapshot_type": self.snapshot_type,
            "collection_id": self.collection_id,
            "paper_ids": list(self.paper_ids),
        }


@dataclass(frozen=True, slots=True)
class DownloadScopeBinding:
    """Exact runtime selection facts carried through Stage 3 authorization."""

    collection_id: str | None = None
    collection_snapshot_hash: str | None = None
    selection_snapshot_hash: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "collection_id": self.collection_id,
            "collection_snapshot_hash": self.collection_snapshot_hash,
            "selection_snapshot_hash": self.selection_snapshot_hash,
        }

    def authorization_context(
        self,
        *,
        mode: str = "attended",
        skill_digest: str | None = None,
        dependency_digest: str | None = None,
        planner_decision_id: str | None = None,
    ) -> AuthorizationContext:
        return AuthorizationContext(
            mode=mode,
            skill_digest=skill_digest,
            dependency_digest=dependency_digest,
            collection_id=self.collection_id,
            collection_snapshot_hash=self.collection_snapshot_hash,
            selection_snapshot_hash=self.selection_snapshot_hash,
            planner_decision_id=planner_decision_id,
        )


class DownloadScopeSnapshotStore:
    """Validate snapshots against the repository and prove membership by hash."""

    def __init__(self, database: Database) -> None:
        self.database = database

    def load_file(
        self,
        path: str | Path,
        *,
        expected_type: str,
        persist: bool = True,
    ) -> DownloadScopeSnapshot:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        snapshot = self._from_document(document, expected_type=expected_type)
        if persist:
            self._persist(snapshot)
        else:
            self._validate_existing(snapshot)
        return snapshot

    def load_id(
        self, snapshot_id: str, *, expected_type: str
    ) -> DownloadScopeSnapshot:
        row = self.database.connection.execute(
            "SELECT * FROM download_scope_snapshots WHERE snapshot_id = ?",
            (snapshot_id,),
        ).fetchone()
        if row is None:
            raise DownloadError(f"download scope snapshot not found: {snapshot_id}")
        snapshot = self._from_row(row)
        if snapshot.snapshot_type != expected_type:
            raise DownloadError(
                f"download scope snapshot type must be {expected_type}"
            )
        return snapshot

    def contains(
        self,
        snapshot_hash: str,
        paper_id: str,
        expected_type: str,
        expected_collection_id: str | None,
    ) -> bool:
        row = self.database.connection.execute(
            "SELECT * FROM download_scope_snapshots WHERE snapshot_hash = ?",
            (snapshot_hash,),
        ).fetchone()
        try:
            if row is None:
                return False
            snapshot = self._from_row(row)
        except (DownloadError, TypeError, ValueError):
            return False
        return (
            snapshot.snapshot_type == expected_type
            and snapshot.collection_id == expected_collection_id
            and paper_id in snapshot.paper_ids
        )

    def _from_document(
        self, document: object, *, expected_type: str | None = None
    ) -> DownloadScopeSnapshot:
        validate(document, "download-scope-snapshot.schema.json")
        if not isinstance(document, Mapping):
            raise DownloadError("download scope snapshot must be an object")
        paper_ids = tuple(document["paper_ids"])
        if paper_ids != tuple(sorted(paper_ids)):
            raise DownloadError("download scope snapshot paper_ids must be sorted")
        snapshot = DownloadScopeSnapshot(
            snapshot_id=str(document["snapshot_id"]),
            snapshot_type=str(document["snapshot_type"]),
            snapshot_hash=str(document["snapshot_hash"]),
            collection_id=document["collection_id"],
            paper_ids=paper_ids,
            created_at=str(document["created_at"]),
        )
        if expected_type is not None and snapshot.snapshot_type != expected_type:
            raise DownloadError(
                f"download scope snapshot type must be {expected_type}"
            )
        if snapshot.snapshot_type == "collection" and snapshot.collection_id is None:
            raise DownloadError("collection snapshot requires collection_id")
        if snapshot.snapshot_type == "selection" and snapshot.collection_id is not None:
            raise DownloadError("selection snapshot cannot set collection_id")
        self._validate_repository(snapshot)
        return snapshot

    def _from_row(self, row: Mapping[str, Any]) -> DownloadScopeSnapshot:
        paper_ids = json.loads(row["paper_ids_json"])
        return self._from_document({
            "schema_version": "1",
            "snapshot_id": row["snapshot_id"],
            "snapshot_type": row["snapshot_type"],
            "snapshot_hash": row["snapshot_hash"],
            "collection_id": row["collection_id"],
            "paper_ids": paper_ids,
            "created_at": row["created_at"],
        })

    def _validate_repository(self, snapshot: DownloadScopeSnapshot) -> None:
        if content_hash(snapshot.core) != snapshot.snapshot_hash:
            raise DownloadError("download scope snapshot hash has drifted")
        placeholders = ",".join("?" for _ in snapshot.paper_ids)
        rows = self.database.connection.execute(
            f"SELECT paper_id FROM papers WHERE paper_id IN ({placeholders})",
            snapshot.paper_ids,
        ).fetchall()
        if {str(row["paper_id"]) for row in rows} != set(snapshot.paper_ids):
            raise DownloadError("download scope snapshot contains an unknown paper")
        if snapshot.collection_id is not None:
            collection = self.database.connection.execute(
                "SELECT 1 FROM collections WHERE collection_id = ?",
                (snapshot.collection_id,),
            ).fetchone()
            if collection is None:
                raise DownloadError("download scope snapshot collection does not exist")
            members = self.database.connection.execute(
                f"""SELECT paper_id FROM paper_collections
                    WHERE collection_id = ? AND paper_id IN ({placeholders})
                      AND membership_status != 'not_member'""",
                (snapshot.collection_id, *snapshot.paper_ids),
            ).fetchall()
            if {str(row["paper_id"]) for row in members} != set(snapshot.paper_ids):
                raise DownloadError(
                    "download scope snapshot contains a paper outside its collection"
                )

    def _persist(self, snapshot: DownloadScopeSnapshot) -> None:
        with self.database.transaction() as connection:
            if self._validate_existing(snapshot):
                return
            connection.execute(
                """INSERT INTO download_scope_snapshots(
                       snapshot_id, snapshot_type, snapshot_hash, collection_id,
                       paper_ids_json, created_at
                   ) VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    snapshot.snapshot_id,
                    snapshot.snapshot_type,
                    snapshot.snapshot_hash,
                    snapshot.collection_id,
                    _json(snapshot.paper_ids),
                    snapshot.created_at,
                ),
            )

    def _validate_existing(self, snapshot: DownloadScopeSnapshot) -> bool:
        by_id = self.database.connection.execute(
            "SELECT * FROM download_scope_snapshots WHERE snapshot_id = ?",
            (snapshot.snapshot_id,),
        ).fetchone()
        by_hash = self.database.connection.execute(
            "SELECT * FROM download_scope_snapshots WHERE snapshot_hash = ?",
            (snapshot.snapshot_hash,),
        ).fetchone()
        found = False
        for existing in (by_id, by_hash):
            if existing is None:
                continue
            found = True
            if self._from_row(existing) != snapshot:
                raise DownloadError(
                    "download scope snapshot conflicts with persisted content"
                )
        return found


def build_download_scope_snapshot(
    snapshot_type: str,
    paper_ids: Iterable[str],
    *,
    created_at: str,
    collection_id: str | None = None,
    snapshot_id: str | None = None,
) -> dict[str, object]:
    """Build a canonical hash-bound snapshot document for review or persistence."""

    ordered = tuple(sorted(paper_ids))
    core = {
        "schema_version": "1",
        "snapshot_type": snapshot_type,
        "collection_id": collection_id,
        "paper_ids": list(ordered),
    }
    snapshot_hash = content_hash(core)
    document = {
        **core,
        "snapshot_id": snapshot_id or f"download-{snapshot_type}-{snapshot_hash[:16]}",
        "snapshot_hash": snapshot_hash,
        "created_at": created_at,
    }
    validate(document, "download-scope-snapshot.schema.json")
    return document


@dataclass(frozen=True, slots=True)
class ProviderTerms:
    """Machine-readable evidence about the download provider's current terms."""

    provider: str
    terms_version: str
    evidence_url: str | None
    machine_readable: bool
    allows_download: bool | None
    allows_storage: bool | None
    allows_redistribution: bool | None = None
    domain_allowlist: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.provider or not self.terms_version:
            raise ValueError("provider and terms_version are required")

    @property
    def hash(self) -> str:
        return content_hash(
            {
                "provider": self.provider,
                "terms_version": self.terms_version,
                "evidence_url": self.evidence_url,
                "machine_readable": self.machine_readable,
                "allows_download": self.allows_download,
                "allows_storage": self.allows_storage,
                "allows_redistribution": self.allows_redistribution,
                "domain_allowlist": sorted(host.lower() for host in self.domain_allowlist),
            }
        )

    def covers(self, host: str | None) -> bool:
        return host is not None and host.lower() in {
            value.lower() for value in self.domain_allowlist
        }


@dataclass(frozen=True, slots=True)
class HTTPResponse:
    status_code: int
    headers: Mapping[str, str]
    body: bytes
    final_url: str | None = None


@dataclass(frozen=True, slots=True)
class PolicyOutcome:
    status: FetchDecisionStatus
    reason_code: str


@dataclass(frozen=True, slots=True)
class AuthorizationContext:
    """Runtime facts that must still match the approved grant at fetch time."""

    mode: str = "attended"
    skill_digest: str | None = None
    dependency_digest: str | None = None
    collection_id: str | None = None
    collection_snapshot_hash: str | None = None
    selection_snapshot_hash: str | None = None
    planner_decision_id: str | None = None

    @property
    def hash(self) -> str:
        return content_hash(
            {
                "mode": self.mode,
                "skill_digest": self.skill_digest,
                "dependency_digest": self.dependency_digest,
                "collection_id": self.collection_id,
                "collection_snapshot_hash": self.collection_snapshot_hash,
                "selection_snapshot_hash": self.selection_snapshot_hash,
            }
        )


class DownloadAccessPolicy:
    """A small evaluator whose decision sets and license aliases live in YAML."""

    def __init__(self, document: Mapping[str, Any]) -> None:
        self.document = MappingProxyType(dict(document))
        self.version = _required_text(document, "policy_version")
        if document.get("schema_version") != "1":
            raise PolicyError("download policy schema_version must be 1")
        self.hash = content_hash(document)
        self.purposes = frozenset(_text_list(document, "purposes"))
        self.restricted_access_bases = frozenset(_text_list(document, "restricted_access_bases"))
        self.publication_versions = frozenset(_text_list(document, "publication_versions"))
        aliases = document.get("license_aliases")
        compatible = document.get("compatible_licenses")
        states = document.get("states")
        limits = document.get("fetch_limits")
        if not isinstance(aliases, dict) or not isinstance(compatible, dict):
            raise PolicyError("license_aliases and compatible_licenses must be mappings")
        if not isinstance(states, dict) or not isinstance(limits, dict):
            raise PolicyError("states and fetch_limits must be mappings")
        self.license_aliases = MappingProxyType(
            {_license_key(str(key)): _license_key(str(value)) for key, value in aliases.items()}
        )
        self.compatible_licenses = MappingProxyType(
            {
                purpose: frozenset(_license_key(str(value)) for value in values)
                for purpose, values in compatible.items()
                if isinstance(values, list)
            }
        )
        required_states = {"allow", "needs_grant", "manual", "deny"}
        if set(states) != required_states or any(states[key] != key for key in required_states):
            raise PolicyError("policy states must freeze allow, needs_grant, manual, and deny")
        self.request_ttl_seconds = _positive_int(limits, "request_ttl_seconds")
        self.min_pdf_bytes = _positive_int(limits, "min_pdf_bytes")
        self.max_pdf_bytes = _positive_int(limits, "max_pdf_bytes")
        if self.min_pdf_bytes >= self.max_pdf_bytes:
            raise PolicyError("min_pdf_bytes must be smaller than max_pdf_bytes")
        if self.purposes != set(self.compatible_licenses):
            raise PolicyError("every purpose needs an explicit compatible license set")

    @classmethod
    def load(cls, path: str | Path) -> DownloadAccessPolicy:
        document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise PolicyError("download policy must be a mapping")
        return cls(document)

    def normalize_license(self, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        normalized = _license_key(value)
        return self.license_aliases.get(normalized, normalized)

    def decide(
        self,
        candidate: AccessLocationCandidate,
        purpose: str,
        terms: ProviderTerms | None,
        *,
        has_grant: bool,
    ) -> PolicyOutcome:
        if purpose not in self.purposes:
            raise PolicyError(f"unsupported download purpose: {purpose}")
        if candidate.publication_version.value not in self.publication_versions:
            return PolicyOutcome(FetchDecisionStatus.MANUAL, "publication_version_unclassified")

        license_id = self.normalize_license(candidate.license)
        compatible = license_id in self.compatible_licenses[purpose]
        clearly_open = candidate.access_basis is AccessBasis.OPEN_LICENSE and compatible

        if purpose == "redistribution" and not clearly_open:
            return PolicyOutcome(FetchDecisionStatus.DENY, "redistribution_requires_compatible_license")
        if terms is None or not terms.machine_readable or not terms.evidence_url:
            return PolicyOutcome(FetchDecisionStatus.MANUAL, "provider_terms_unmachineable")
        if not terms.covers(candidate.host):
            return PolicyOutcome(FetchDecisionStatus.MANUAL, "provider_terms_host_uncovered")
        if terms.allows_download is None or terms.allows_storage is None:
            return PolicyOutcome(FetchDecisionStatus.MANUAL, "provider_terms_permission_unknown")
        if not terms.allows_download or not terms.allows_storage:
            return PolicyOutcome(FetchDecisionStatus.DENY, "provider_terms_forbid_download_or_storage")
        if purpose == "redistribution":
            if terms.allows_redistribution is None:
                return PolicyOutcome(FetchDecisionStatus.MANUAL, "provider_terms_redistribution_unknown")
            if not terms.allows_redistribution:
                return PolicyOutcome(FetchDecisionStatus.DENY, "provider_terms_forbid_redistribution")
            return PolicyOutcome(FetchDecisionStatus.ALLOW, "compatible_open_license")
        if clearly_open:
            return PolicyOutcome(FetchDecisionStatus.ALLOW, "compatible_open_license")
        if candidate.access_basis.value in self.restricted_access_bases or not compatible:
            if has_grant:
                return PolicyOutcome(FetchDecisionStatus.ALLOW, "authorized_by_grant")
            return PolicyOutcome(FetchDecisionStatus.NEEDS_GRANT, "explicit_download_grant_required")
        return PolicyOutcome(FetchDecisionStatus.MANUAL, "access_combination_unclassified")


class DownloadService:
    """Persist candidates, issue policy-bound requests, and fetch public PDFs."""

    def __init__(
        self,
        database: Database,
        artifact_store: ArtifactStore,
        policy: DownloadAccessPolicy,
        provider_terms: Mapping[str, ProviderTerms],
        fetcher: Callable[[str], HTTPResponse],
        scope_membership: Callable[[str, str, str, str | None], bool] | None = None,
        clock: Callable[[], datetime] | None = None,
        provider_fetchers: Mapping[str, Callable[[str], HTTPResponse]] | None = None,
    ) -> None:
        self.database = database
        self.artifact_store = artifact_store
        self.policy = policy
        self.provider_terms = dict(provider_terms)
        self.fetcher = (
            (lambda url: urllib_fetch(url, max_bytes=policy.max_pdf_bytes))
            if fetcher is urllib_fetch
            else fetcher
        )
        self.scope_membership = scope_membership
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.provider_fetchers = dict(provider_fetchers or {})
        for provider, terms in self.provider_terms.items():
            if provider != terms.provider:
                raise ValueError("provider terms registry keys must match provider names")

    def persist_candidate(
        self, candidate: AccessLocationCandidate, *, now: datetime | str | None = None
    ) -> AccessLocationCandidate:
        with self.database.transaction() as connection:
            existing = connection.execute(
                "SELECT * FROM download_candidates WHERE candidate_id = ?", (candidate.candidate_id,)
            ).fetchone()
            persisted_time = existing["retrieved_at"] if existing is not None else now
            normalized = _normalize_candidate(candidate, now=persisted_time)
            if existing is not None:
                if _candidate_from_row(existing) != normalized:
                    raise DownloadError("persisted candidate content is immutable")
                return normalized
            duplicate = connection.execute(
                """SELECT candidate_id FROM download_candidates
                   WHERE paper_id = ? AND resolver = ? AND url = ? AND publication_version = ?""",
                (
                    normalized.paper_id,
                    normalized.resolver,
                    normalized.url,
                    normalized.publication_version.value,
                ),
            ).fetchone()
            if duplicate is not None:
                raise DownloadError(
                    f"download location already has candidate_id {duplicate['candidate_id']}"
                )
            connection.execute(
                """INSERT INTO download_candidates(
                       candidate_id, paper_id, resolver, url, landing_url, publication_version,
                       host, license, access_basis, retrieved_at, raw_evidence_hash, provenance_json
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                _candidate_row_values(normalized),
            )
        return normalized

    def require_authorized_handoff(
        self,
        grant_id: str,
        candidate: AccessLocationCandidate,
        *,
        purpose: str,
        provider: str,
        mode: str,
        now: datetime | str,
        skill_digest: str,
        dependency_digest: str,
        reserved_paper_ids: Iterable[str] = (),
        collection_id: str | None = None,
        collection_snapshot_hash: str | None = None,
        selection_snapshot_hash: str | None = None,
    ) -> ActiveGrant:
        """Reprove an exact grant before exposing a candidate to a browser queue.

        This is deliberately read-only.  It reuses the same selection, domain,
        purpose, provider, digest, and cumulative-capacity checks used when an
        immutable fetch request is issued later.
        """

        at = _as_datetime(now)
        terms = self.provider_terms.get(provider)
        policy = self.policy.decide(candidate, purpose, terms, has_grant=True)
        if policy.status is not FetchDecisionStatus.ALLOW:
            raise FetchRejected(
                f"authorized handoff is not allowed by provider policy: {policy.reason_code}"
            )
        requested_papers = (
            self._grant_issued_papers(self.database.connection, grant_id)
            | self._grant_reserved_papers(self.database.connection, grant_id)
            | set(reserved_paper_ids)
            | {candidate.paper_id}
        )
        active = self._require_grant(
            grant_id,
            candidate,
            purpose=purpose,
            provider=provider,
            mode=mode,
            now=at,
            skill_digest=skill_digest,
            dependency_digest=dependency_digest,
            collection_id=collection_id,
            collection_snapshot_hash=collection_snapshot_hash,
            selection_snapshot_hash=selection_snapshot_hash,
            paper_count=len(requested_papers),
        )
        if active.document["scope"]["provider"] != provider:
            raise GrantError("authorized handoff grant must bind its exact provider")
        return active

    def reserve_authorized_handoff(
        self,
        grant_id: str,
        candidate: AccessLocationCandidate,
        *,
        run_id: str,
        queue_path: str,
        queue_item_hash: str,
        purpose: str,
        provider: str,
        mode: str,
        now: datetime | str,
        skill_digest: str,
        dependency_digest: str,
        reserved_paper_ids: Iterable[str] = (),
        collection_id: str | None = None,
        collection_snapshot_hash: str | None = None,
        selection_snapshot_hash: str | None = None,
    ) -> ActiveGrant:
        """Persist one exact browser-queue capacity claim before exposing its CSV."""

        at = _as_datetime(now)
        with self.database.transaction() as connection:
            active = self.require_authorized_handoff(
                grant_id,
                candidate,
                purpose=purpose,
                provider=provider,
                mode=mode,
                now=at,
                skill_digest=skill_digest,
                dependency_digest=dependency_digest,
                reserved_paper_ids=reserved_paper_ids,
                collection_id=collection_id,
                collection_snapshot_hash=collection_snapshot_hash,
                selection_snapshot_hash=selection_snapshot_hash,
            )
            expected = (
                candidate.candidate_id,
                run_id,
                queue_path,
                queue_item_hash,
                collection_id,
                collection_snapshot_hash,
                selection_snapshot_hash,
            )
            existing = connection.execute(
                """SELECT candidate_id, run_id, queue_path, queue_item_hash,
                          collection_id, collection_snapshot_hash,
                          selection_snapshot_hash
                   FROM authorized_download_queue_reservations
                   WHERE authorization_grant_id = ? AND paper_id = ?""",
                (grant_id, candidate.paper_id),
            ).fetchone()
            if existing is not None:
                if tuple(existing) != expected:
                    raise GrantError(
                        "authorized handoff reservation has different immutable inputs"
                    )
                return active
            connection.execute(
                """INSERT INTO authorized_download_queue_reservations(
                       authorization_grant_id, paper_id, candidate_id, run_id,
                       queue_path, queue_item_hash, reserved_at, collection_id,
                       collection_snapshot_hash, selection_snapshot_hash
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    grant_id,
                    candidate.paper_id,
                    candidate.candidate_id,
                    run_id,
                    queue_path,
                    queue_item_hash,
                    _iso(at),
                    collection_id,
                    collection_snapshot_hash,
                    selection_snapshot_hash,
                ),
            )
            return active

    def load_candidate(self, candidate_id: str) -> AccessLocationCandidate:
        """Load the immutable candidate persisted by the public probe pass."""

        return self._load_candidate(candidate_id)

    def load_candidate_for_handoff(
        self, paper_id: str, url: str
    ) -> AccessLocationCandidate:
        """Resolve one frozen queue row back to exactly one persisted candidate."""

        rows = self.database.connection.execute(
            """SELECT candidate_id FROM download_candidates
               WHERE paper_id = ? AND url = ? ORDER BY candidate_id""",
            (paper_id, url),
        ).fetchall()
        if len(rows) != 1:
            raise FetchRejected(
                "authorized queue row does not bind exactly one persisted candidate"
            )
        return self._load_candidate(str(rows[0]["candidate_id"]))

    def load_reserved_handoff_candidate(
        self,
        grant_id: str,
        *,
        run_id: str,
        paper_id: str,
        queue_path: str,
        queue_item_hash: str,
        collection_id: str | None = None,
        collection_snapshot_hash: str | None = None,
        selection_snapshot_hash: str | None = None,
    ) -> AccessLocationCandidate:
        candidate, stored_queue_item_hash = self.load_reserved_handoff_binding(
            grant_id,
            run_id=run_id,
            paper_id=paper_id,
            queue_path=queue_path,
            collection_id=collection_id,
            collection_snapshot_hash=collection_snapshot_hash,
            selection_snapshot_hash=selection_snapshot_hash,
        )
        if stored_queue_item_hash != queue_item_hash:
            raise FetchRejected(
                "authorized queue row has no matching durable reservation"
            )
        return candidate

    def load_reserved_handoff_binding(
        self,
        grant_id: str,
        *,
        run_id: str,
        paper_id: str,
        queue_path: str,
        collection_id: str | None = None,
        collection_snapshot_hash: str | None = None,
        selection_snapshot_hash: str | None = None,
    ) -> tuple[AccessLocationCandidate, str]:
        """Load the exact reserved candidate before reproving its row hash.

        The durable reservation supplies the candidate identity on resume.  A
        caller must still recompute and compare ``queue_item_hash`` from that
        candidate before exposing the queue.
        """

        row = self.database.connection.execute(
            """SELECT candidate_id, run_id, queue_path, queue_item_hash,
                      collection_id, collection_snapshot_hash,
                      selection_snapshot_hash
               FROM authorized_download_queue_reservations
               WHERE authorization_grant_id = ? AND paper_id = ?""",
            (grant_id, paper_id),
        ).fetchone()
        expected = (
            run_id,
            queue_path,
            collection_id,
            collection_snapshot_hash,
            selection_snapshot_hash,
        )
        if row is None or tuple(row[key] for key in (
            "run_id", "queue_path", "collection_id", "collection_snapshot_hash",
            "selection_snapshot_hash",
        )) != expected:
            raise FetchRejected(
                "authorized queue row has no matching durable reservation"
            )
        return (
            self._load_candidate(str(row["candidate_id"])),
            str(row["queue_item_hash"]),
        )

    def list_authorized_handoff_reservations(
        self, *, run_id: str
    ) -> tuple[dict[str, str | None], ...]:
        """Return the complete durable reservation set for one Stage 3 run."""

        rows = self.database.connection.execute(
            """SELECT authorization_grant_id, paper_id, candidate_id, run_id,
                      queue_path, queue_item_hash, collection_id,
                      collection_snapshot_hash, selection_snapshot_hash
               FROM authorized_download_queue_reservations
               WHERE run_id = ? ORDER BY paper_id, authorization_grant_id""",
            (run_id,),
        ).fetchall()
        return tuple({key: row[key] for key in row.keys()} for row in rows)

    def probe(
        self,
        candidate: AccessLocationCandidate,
        *,
        purpose: str,
        provider: str,
        now: datetime | str,
        authorization_grant_id: str | None = None,
        mode: str = "attended",
        skill_digest: str | None = None,
        dependency_digest: str | None = None,
        collection_id: str | None = None,
        collection_snapshot_hash: str | None = None,
        selection_snapshot_hash: str | None = None,
        run_id: str | None = None,
    ) -> FetchDecision:
        """Evaluate policy and persist its decision without fetching the URL."""

        at = _as_datetime(now)
        persisted = self.persist_candidate(candidate, now=at)
        terms = self.provider_terms.get(provider)
        authorization_context = AuthorizationContext(
            mode=mode,
            skill_digest=skill_digest,
            dependency_digest=dependency_digest,
            collection_id=collection_id,
            collection_snapshot_hash=collection_snapshot_hash,
            selection_snapshot_hash=selection_snapshot_hash,
        )
        outcome = self.policy.decide(persisted, purpose, terms, has_grant=False)
        grant: ActiveGrant | None = None
        if outcome.status is FetchDecisionStatus.NEEDS_GRANT and authorization_grant_id:
            try:
                grant = self._require_grant(
                    authorization_grant_id,
                    persisted,
                    purpose=purpose,
                    provider=provider,
                    mode=mode,
                    now=at,
                    skill_digest=skill_digest,
                    dependency_digest=dependency_digest,
                    collection_id=collection_id,
                    collection_snapshot_hash=collection_snapshot_hash,
                    selection_snapshot_hash=selection_snapshot_hash,
                )
            except GrantError:
                outcome = PolicyOutcome(FetchDecisionStatus.NEEDS_GRANT, "download_grant_invalid")
            else:
                outcome = self.policy.decide(persisted, purpose, terms, has_grant=True)

        self._save_policy_decision(
            persisted.candidate_id,
            purpose,
            provider,
            terms,
            outcome,
            at,
            run_id=run_id,
            grant=grant,
        )
        if outcome.status is not FetchDecisionStatus.ALLOW:
            return FetchDecision(
                candidate_id=persisted.candidate_id,
                status=outcome.status,
                reason_code=outcome.reason_code,
                policy_version=self.policy.version,
                authorization_grant_id=grant.grant_id if grant else None,
            )
        assert terms is not None
        try:
            request = self._issue_request(
                persisted,
                purpose=purpose,
                provider=provider,
                terms=terms,
                grant=grant,
                authorization_context=authorization_context if grant else None,
                now=at,
            )
        except GrantError:
            outcome = PolicyOutcome(FetchDecisionStatus.NEEDS_GRANT, "download_grant_exhausted")
            self._save_policy_decision(
                persisted.candidate_id,
                purpose,
                provider,
                terms,
                outcome,
                at,
                run_id=run_id,
                grant=grant,
            )
            return FetchDecision(
                candidate_id=persisted.candidate_id,
                status=outcome.status,
                reason_code=outcome.reason_code,
                policy_version=self.policy.version,
            )
        return FetchDecision(
            candidate_id=persisted.candidate_id,
            status=outcome.status,
            reason_code=outcome.reason_code,
            policy_version=self.policy.version,
            fetch_request=request,
            authorization_grant_id=grant.grant_id if grant else None,
        )

    def fetch(
        self,
        request: FetchRequest,
        *,
        run_id: str,
        now: datetime | str,
        authorization_context: AuthorizationContext | None = None,
    ) -> DownloadResult:
        """Consume a persisted request before the first network side effect."""

        at = _as_datetime(now)
        row = self.database.connection.execute(
            "SELECT * FROM fetch_requests WHERE request_id = ?", (request.request_id,)
        ).fetchone()
        if row is None:
            raise FetchRejected("fetch request was not issued by this coordinator")
        self._match_request(row, request)
        if row["status"] == "consumed":
            return self._existing_result(request.request_id)
        if row["status"] != "ready":
            raise FetchRejected(f"fetch request is {row['status']}")
        if at >= _as_datetime(row["expires_at"]):
            self._set_request_status(request.request_id, "expired", expected="ready")
            raise FetchRejected("fetch request is expired")

        candidate = self._load_candidate(request.candidate_id)
        terms = self.provider_terms.get(request.provider)
        runtime_context = authorization_context or AuthorizationContext()
        expected_identity = self._request_identity_hash(
            candidate,
            request.purpose,
            terms,
            request.authorization_grant_hash,
            request.fencing_token,
            runtime_context if request.authorization_grant_id else None,
        )
        if (
            row["policy_hash"] != self.policy.hash
            or request.request_id != "fetch-" + expected_identity[:32]
            or request.idempotency_key != "download:" + expected_identity
        ):
            raise FetchRejected("candidate, policy, purpose, or provider terms hash drifted")
        if (
            row["policy_version"] != self.policy.version
            or candidate.candidate_id != request.candidate_id
            or request.provider not in self.provider_terms
        ):
            raise FetchRejected("fetch request no longer matches current policy state")
        has_grant = False
        if request.authorization_grant_id:
            try:
                grant = self._require_grant_from_request(
                    request, candidate, at, runtime_context
                )
            except GrantError as error:
                self._set_request_status(request.request_id, "revoked", expected="ready")
                raise FetchRejected(str(error)) from error
            if grant.content_hash != request.authorization_grant_hash:
                raise FetchRejected("authorization grant hash drifted")
            has_grant = True
        elif request.authorization_grant_hash is not None:
            raise FetchRejected("authorization grant binding is incomplete")
        current_outcome = self.policy.decide(
            candidate, request.purpose, terms, has_grant=has_grant
        )
        if current_outcome.status is not FetchDecisionStatus.ALLOW:
            raise FetchRejected("current policy no longer allows this fetch request")

        attempt_id = "attempt-" + content_hash(
            {"run_id": run_id, "request_id": request.request_id, "provider": request.provider}
        )[:32]
        with self.database.transaction() as connection:
            if runtime_context.planner_decision_id is not None:
                planner = connection.execute(
                    """SELECT run_id, candidate_id, authorization_grant_id, status, selected
                       FROM stage3_luna_decisions WHERE planner_decision_id = ?""",
                    (runtime_context.planner_decision_id,),
                ).fetchone()
                if (
                    planner is None
                    or planner["run_id"] != run_id
                    or planner["candidate_id"] != candidate.candidate_id
                    or planner["authorization_grant_id"] != request.authorization_grant_id
                    or planner["status"] != "complete"
                    or planner["selected"] != 1
                ):
                    raise FetchRejected("authorized Luna planner decision is not bound to this fetch")
            consumed = connection.execute(
                """UPDATE fetch_requests SET status = 'consumed'
                   WHERE request_id = ? AND status = 'ready' AND fencing_token = ?
                     AND fencing_token = (
                         SELECT MAX(fencing_token) FROM fetch_requests
                         WHERE candidate_id = ? AND provider = ? AND policy_hash = ?
                           AND purpose = ? AND authorization_grant_hash IS ?
                     )""",
                (
                    request.request_id,
                    request.fencing_token,
                    request.candidate_id,
                    request.provider,
                    row["policy_hash"],
                    request.purpose,
                    request.authorization_grant_hash,
                ),
            )
            if consumed.rowcount != 1:
                raise FetchRejected("fetch request has a stale or consumed fencing token")
            connection.execute(
                """INSERT INTO download_attempts(
                       download_attempt_id, run_id, candidate_id, provider,
                       authorization_grant_id, fetch_request_id, planner_decision_id,
                       result_status, attempted_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?)""",
                (
                    attempt_id,
                    run_id,
                    candidate.candidate_id,
                    request.provider,
                    request.authorization_grant_id,
                    request.request_id,
                    runtime_context.planner_decision_id,
                    _iso(at),
                ),
            )

        try:
            response = self.provider_fetchers.get(request.provider, self.fetcher)(candidate.url)
        except (TimeoutError, OSError) as error:
            return self._finish_failure(
                attempt_id,
                request,
                candidate,
                DownloadStatus.FAILED_RETRYABLE,
                "network_error",
                detail=type(error).__name__,
            )
        final_url_error = _final_url_error(candidate, response)
        if final_url_error:
            return self._finish_failure(
                attempt_id,
                request,
                candidate,
                DownloadStatus.FAILED_TERMINAL,
                final_url_error,
                http_status=response.status_code,
            )
        completed_at = max(at, _as_datetime(self.clock()))
        completion_failure = self._completion_failure(
            request, candidate, terms, runtime_context, completed_at
        )
        if completion_failure:
            status, category = completion_failure
            return self._finish_failure(
                attempt_id, request, candidate, status, category, http_status=response.status_code
            )
        status_result = _http_failure_status(response.status_code)
        if status_result is not None:
            status, attempt_status, category = status_result
            return self._finish_failure(
                attempt_id,
                request,
                candidate,
                status,
                category,
                attempt_status=attempt_status,
                http_status=response.status_code,
            )
        validation_error = self._validate_pdf_response(response)
        if validation_error:
            return self._finish_failure(
                attempt_id,
                request,
                candidate,
                DownloadStatus.FAILED_TERMINAL,
                validation_error,
                http_status=response.status_code,
            )

        try:
            stored = self.artifact_store.put_bytes(response.body, mime_type="application/pdf")
        except OSError as error:
            return self._finish_failure(
                attempt_id,
                request,
                candidate,
                DownloadStatus.FAILED_RETRYABLE,
                "artifact_store_io_error",
                detail=type(error).__name__,
            )
        except ValueError as error:
            return self._finish_failure(
                attempt_id,
                request,
                candidate,
                DownloadStatus.FAILED_TERMINAL,
                "artifact_store_invalid",
                detail=type(error).__name__,
            )
        artifact_id = "artifact-" + stored.artifact_hash
        provenance = {
            "provider": request.provider,
            "candidate_id": candidate.candidate_id,
            "access_basis": candidate.access_basis.value,
            "authorization_grant_id": request.authorization_grant_id,
        }
        late_failure: tuple[DownloadStatus, str] | None = None
        with self.database.transaction() as connection:
            late_completed_at = max(completed_at, _as_datetime(self.clock()))
            late_failure = self._completion_failure(
                request, candidate, terms, runtime_context, late_completed_at, connection=connection
            )
            if late_failure:
                return self._finish_failure(
                    attempt_id,
                    request,
                    candidate,
                    late_failure[0],
                    late_failure[1],
                    http_status=response.status_code,
                )
            existing = connection.execute(
                """SELECT artifact_id, paper_id, mime_type, byte_size, relative_path
                   FROM artifacts WHERE sha256 = ?""",
                (stored.artifact_hash,),
            ).fetchone()
            if existing is None:
                connection.execute(
                    """INSERT INTO artifacts(
                           artifact_id, paper_id, artifact_kind, relative_path, mime_type,
                           byte_size, sha256, source_url, provenance_json
                       ) VALUES (?, ?, 'pdf', ?, 'application/pdf', ?, ?, ?, ?)""",
                    (
                        artifact_id,
                        candidate.paper_id,
                        stored.relative_path,
                        stored.size_bytes,
                        stored.artifact_hash,
                        candidate.url,
                        _json(provenance),
                    ),
                )
            else:
                if (
                    existing["paper_id"] != candidate.paper_id
                    or existing["mime_type"] != "application/pdf"
                    or existing["byte_size"] != stored.size_bytes
                    or existing["relative_path"] != stored.relative_path
                ):
                    return self._finish_failure(
                        attempt_id,
                        request,
                        candidate,
                        DownloadStatus.FAILED_TERMINAL,
                        "artifact_metadata_conflict",
                        http_status=response.status_code,
                    )
                artifact_id = existing["artifact_id"]
            connection.execute(
                """UPDATE download_attempts
                   SET result_status = 'downloaded', artifact_id = ?, http_status = ?
                   WHERE download_attempt_id = ?""",
                (artifact_id, response.status_code, attempt_id),
            )
        return DownloadResult(
            request_id=request.request_id,
            paper_id=candidate.paper_id,
            status=DownloadStatus.DOWNLOADED,
            provider=request.provider,
            artifact_id=artifact_id,
            content_hash=stored.artifact_hash,
            source_url=candidate.url,
            downloaded_at=_iso(at),
            authorization_grant_id=request.authorization_grant_id,
        )

    def reissue_retryable(
        self,
        request_id: str,
        *,
        now: datetime | str,
        authorization_context: AuthorizationContext | None = None,
    ) -> FetchRequest:
        """Issue the next fence after a retryable result or an expired interrupted attempt."""

        at = _as_datetime(now)
        row = self.database.connection.execute(
            "SELECT * FROM fetch_requests WHERE request_id = ?", (request_id,)
        ).fetchone()
        if row is None:
            raise FetchRejected("fetch request not found")
        attempt = self.database.connection.execute(
            """SELECT result_status FROM download_attempts
               WHERE fetch_request_id = ? ORDER BY attempted_at DESC LIMIT 1""",
            (request_id,),
        ).fetchone()
        if row["status"] != "consumed" or attempt is None:
            raise FetchRejected("only a retryable consumed request can be reissued")
        if attempt["result_status"] == "pending":
            if at < _as_datetime(row["expires_at"]):
                raise FetchRejected("pending attempt is still inside its execution window")
            with self.database.transaction() as connection:
                connection.execute(
                    """UPDATE download_attempts
                       SET result_status = 'failed_retryable', failure_category = 'interrupted'
                       WHERE fetch_request_id = ? AND result_status = 'pending'""",
                    (request_id,),
                )
        elif attempt["result_status"] != "failed_retryable":
            raise FetchRejected("only a retryable consumed request can be reissued")
        candidate = self._load_candidate(row["candidate_id"])
        terms = self.provider_terms.get(row["provider"])
        runtime_context = authorization_context or AuthorizationContext()
        expected_identity = self._request_identity_hash(
            candidate,
            row["purpose"],
            terms,
            row["authorization_grant_hash"],
            row["fencing_token"],
            runtime_context if row["authorization_grant_id"] else None,
        )
        if (
            row["policy_hash"] != self.policy.hash
            or row["request_id"] != "fetch-" + expected_identity[:32]
            or row["idempotency_key"] != "download:" + expected_identity
        ):
            raise FetchRejected("retry binding no longer matches current policy state")
        grant = None
        if row["authorization_grant_id"]:
            request = _request_from_row(row)
            grant = self._require_grant_from_request(request, candidate, at, runtime_context)
            if grant.content_hash != row["authorization_grant_hash"]:
                raise FetchRejected("authorization grant hash drifted")
        assert terms is not None
        outcome = self.policy.decide(
            candidate, row["purpose"], terms, has_grant=grant is not None
        )
        if outcome.status is not FetchDecisionStatus.ALLOW:
            raise FetchRejected("current policy no longer permits retry")
        return self._issue_request(
            candidate,
            purpose=row["purpose"],
            provider=row["provider"],
            terms=terms,
            grant=grant,
            authorization_context=runtime_context if grant else None,
            now=at,
        )

    def _save_policy_decision(
        self,
        candidate_id: str,
        purpose: str,
        provider: str,
        terms: ProviderTerms | None,
        outcome: PolicyOutcome,
        decided_at: datetime,
        *,
        run_id: str | None,
        grant: ActiveGrant | None,
    ) -> None:
        decision_document = {
            "candidate_id": candidate_id,
            "run_id": run_id,
            "provider": provider,
            "purpose": purpose,
            "policy_hash": self.policy.hash,
            "provider_terms_hash": terms.hash if terms else None,
            "authorization_grant_id": grant.grant_id if grant else None,
            "decision": outcome.status.value,
            "reason_code": outcome.reason_code,
            "decided_at": _iso(decided_at),
        }
        with self.database.transaction() as connection:
            connection.execute(
                """UPDATE download_candidates
                   SET policy_version = ?, policy_purpose = ?, policy_decision = ?, policy_reason_code = ?
                   WHERE candidate_id = ?""",
                (self.policy.version, purpose, outcome.status.value, outcome.reason_code, candidate_id),
            )
            connection.execute(
                """INSERT OR IGNORE INTO download_policy_decisions(
                       decision_id, candidate_id, run_id, provider, purpose, policy_version,
                       policy_hash, provider_terms_hash, authorization_grant_id, decision,
                       reason_code, decided_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    "decision-" + content_hash(decision_document)[:32],
                    candidate_id,
                    run_id,
                    provider,
                    purpose,
                    self.policy.version,
                    self.policy.hash,
                    terms.hash if terms else None,
                    grant.grant_id if grant else None,
                    outcome.status.value,
                    outcome.reason_code,
                    _iso(decided_at),
                ),
            )

    def _issue_request(
        self,
        candidate: AccessLocationCandidate,
        *,
        purpose: str,
        provider: str,
        terms: ProviderTerms,
        grant: ActiveGrant | None,
        authorization_context: AuthorizationContext | None,
        now: datetime,
    ) -> FetchRequest:
        grant_id = grant.grant_id if grant else None
        grant_hash = grant.content_hash if grant else None
        with self.database.transaction() as connection:
            existing = connection.execute(
                """SELECT * FROM fetch_requests
                   WHERE candidate_id = ? AND provider = ? AND purpose = ? AND policy_hash = ?
                     AND authorization_grant_id IS ? AND authorization_grant_hash IS ? AND status = 'ready'
                   ORDER BY fencing_token DESC LIMIT 1""",
                (candidate.candidate_id, provider, purpose, self.policy.hash, grant_id, grant_hash),
            ).fetchone()
            if existing is not None:
                existing_identity = self._request_identity_hash(
                    candidate,
                    purpose,
                    terms,
                    grant_hash,
                    existing["fencing_token"],
                    authorization_context,
                )
                identity_matches = (
                    existing["request_id"] == "fetch-" + existing_identity[:32]
                    and existing["idempotency_key"] == "download:" + existing_identity
                )
                if identity_matches and _as_datetime(existing["expires_at"]) > now:
                    return _request_from_row(existing)
                status = "expired" if _as_datetime(existing["expires_at"]) <= now else "revoked"
                connection.execute(
                    "UPDATE fetch_requests SET status = ? WHERE request_id = ? AND status = 'ready'",
                    (status, existing["request_id"]),
                )
            token_row = connection.execute(
                """SELECT COALESCE(MAX(fencing_token), 0) AS token FROM fetch_requests
                   WHERE candidate_id = ? AND provider = ?""",
                (candidate.candidate_id, provider),
            ).fetchone()
            fencing_token = int(token_row["token"]) + 1
            expires_at = now + timedelta(seconds=self.policy.request_ttl_seconds)
            if grant:
                expires_at = min(expires_at, _as_datetime(str(grant.document["expires_at"])))
                self._require_grant_capacity(connection, grant, candidate.paper_id)
            if expires_at <= now:
                raise FetchRejected("authorization expires before the fetch request")
            identity_hash = self._request_identity_hash(
                candidate,
                purpose,
                terms,
                grant_hash,
                fencing_token,
                authorization_context,
            )
            request = FetchRequest(
                request_id="fetch-" + identity_hash[:32],
                candidate_id=candidate.candidate_id,
                policy_version=self.policy.version,
                purpose=purpose,
                provider=provider,
                created_at=_iso(now),
                expires_at=_iso(expires_at),
                idempotency_key="download:" + identity_hash,
                authorization_grant_id=grant_id,
                authorization_grant_hash=grant_hash,
                fencing_token=fencing_token,
            )
            connection.execute(
                """INSERT INTO fetch_requests(
                       request_id, candidate_id, policy_version, policy_hash, purpose, provider,
                       authorization_grant_id, authorization_grant_hash, created_at, expires_at,
                       idempotency_key, fencing_token, status
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'ready')""",
                (
                    request.request_id,
                    request.candidate_id,
                    request.policy_version,
                    self.policy.hash,
                    request.purpose,
                    request.provider,
                    request.authorization_grant_id,
                    request.authorization_grant_hash,
                    request.created_at,
                    request.expires_at,
                    request.idempotency_key,
                    request.fencing_token,
                ),
            )
        return request

    @staticmethod
    def _require_grant_capacity(
        connection: Any,
        grant: ActiveGrant,
        paper_id: str,
        *,
        reserved_paper_ids: Iterable[str] = (),
    ) -> None:
        issued_papers = DownloadService._grant_issued_papers(
            connection, grant.grant_id
        )
        reserved_papers = DownloadService._grant_reserved_papers(
            connection, grant.grant_id
        )
        requested_papers = (
            issued_papers | reserved_papers | set(reserved_paper_ids) | {paper_id}
        )
        if len(requested_papers) > grant.document["max_papers"]:
            raise GrantError("download grant max_papers has been exhausted")

    @staticmethod
    def _grant_issued_papers(connection: Any, grant_id: str) -> set[str]:
        rows = connection.execute(
            """SELECT DISTINCT dc.paper_id
               FROM fetch_requests fr
               JOIN download_candidates dc ON dc.candidate_id = fr.candidate_id
               WHERE fr.authorization_grant_id = ?""",
            (grant_id,),
        ).fetchall()
        return {str(row["paper_id"]) for row in rows}

    @staticmethod
    def _grant_reserved_papers(connection: Any, grant_id: str) -> set[str]:
        rows = connection.execute(
            """SELECT paper_id FROM authorized_download_queue_reservations
               WHERE authorization_grant_id = ?""",
            (grant_id,),
        ).fetchall()
        return {str(row["paper_id"]) for row in rows}

    def _binding(
        self,
        candidate: AccessLocationCandidate,
        purpose: str,
        terms: ProviderTerms | None,
    ) -> dict[str, str]:
        return {
            "candidate_sha256": content_hash(candidate.to_dict()),
            "policy_sha256": self.policy.hash,
            "implementation_sha256": content_hash(
                {"implementation_version": POLICY_IMPLEMENTATION_VERSION}
            ),
            "purpose_sha256": content_hash({"purpose": purpose}),
            "provider_terms_sha256": terms.hash if terms else content_hash({"provider_terms": None}),
        }

    def _request_identity_hash(
        self,
        candidate: AccessLocationCandidate,
        purpose: str,
        terms: ProviderTerms | None,
        grant_hash: str | None,
        fencing_token: int | None,
        authorization_context: AuthorizationContext | None,
    ) -> str:
        if fencing_token is None:
            raise FetchRejected("fetch request requires a fencing token")
        return content_hash(
            {
                "binding": self._binding(candidate, purpose, terms),
                "grant_hash": grant_hash,
                "authorization_context_sha256": (
                    authorization_context.hash if authorization_context else None
                ),
                "fencing_token": fencing_token,
            }
        )

    def _require_grant(
        self,
        grant_id: str,
        candidate: AccessLocationCandidate,
        *,
        purpose: str,
        provider: str,
        mode: str,
        now: datetime,
        skill_digest: str | None,
        dependency_digest: str | None,
        collection_id: str | None,
        collection_snapshot_hash: str | None,
        selection_snapshot_hash: str | None,
        paper_count: int = 1,
    ) -> ActiveGrant:
        store = GrantStore(self.database)
        grant = store.load(grant_id, kind="download", now=now)
        document = grant.document
        scope = document["scope"]
        if mode == "unattended" and document.get("allow_unattended") is not True:
            raise GrantError("unattended download requires an explicit allow_unattended grant signal")
        if "store" not in document["actions"]:
            raise GrantError("download grant must also allow store")
        if candidate.host not in scope["domains"]:
            raise GrantError("download grant does not cover candidate domain")
        scoped_paper = candidate.paper_id if scope["paper_ids"] else None
        scoped_collection = collection_id if scope["collection_ids"] else None
        has_selection = False
        if scope["paper_ids"]:
            has_selection = True
            if candidate.paper_id not in scope["paper_ids"]:
                raise GrantError("download grant does not cover paper")
        if scope["collection_ids"]:
            has_selection = True
            if collection_id not in scope["collection_ids"]:
                raise GrantError("download grant does not cover collection")
            membership = self.database.connection.execute(
                """SELECT 1 FROM paper_collections
                   WHERE paper_id = ? AND collection_id = ? AND membership_status != 'not_member'""",
                (candidate.paper_id, collection_id),
            ).fetchone()
            if membership is None:
                raise GrantError("paper is outside the authorized collection")
        if scope["collection_snapshot_hash"]:
            has_selection = True
            if (
                self.scope_membership is None
                or not self.scope_membership(
                    scope["collection_snapshot_hash"],
                    candidate.paper_id,
                    "collection",
                    collection_id,
                )
            ):
                raise GrantError("paper is outside the frozen collection snapshot")
        if scope["selection_snapshot_hash"]:
            has_selection = True
            if selection_snapshot_hash != scope["selection_snapshot_hash"]:
                raise GrantError("download grant selection snapshot does not match")
            if self.scope_membership is None or not self.scope_membership(
                scope["selection_snapshot_hash"],
                candidate.paper_id,
                "selection",
                None,
            ):
                raise GrantError("paper is outside the frozen selection snapshot")
        if not has_selection:
            raise GrantError("download grant has no usable paper selection")
        data_category = "full_text" if scope["data_categories"] else None
        if scope["data_categories"] and "full_text" not in scope["data_categories"]:
            raise GrantError("download grant does not cover full text")
        active = store.require_active(
            grant_id,
            kind="download",
            action="download",
            purpose=purpose,
            mode=mode,
            now=now,
            paper_id=scoped_paper,
            collection_id=scoped_collection,
            collection_snapshot_hash=collection_snapshot_hash if scope["collection_snapshot_hash"] else None,
            selection_snapshot_hash=selection_snapshot_hash if scope["selection_snapshot_hash"] else None,
            domain=candidate.host,
            provider=provider if scope["provider"] is not None else None,
            data_category=data_category,
            skill_digest=skill_digest if document["skill_digest"] is not None else None,
            dependency_digest=dependency_digest if document["dependency_digest"] is not None else None,
            lineage_hash=document["lineage_hash"],
            paper_count=paper_count,
        )
        self._require_grant_capacity(self.database.connection, active, candidate.paper_id)
        return active

    def _require_grant_from_request(
        self,
        request: FetchRequest,
        candidate: AccessLocationCandidate,
        now: datetime,
        authorization_context: AuthorizationContext,
    ) -> ActiveGrant:
        assert request.authorization_grant_id is not None
        return self._require_grant(
            request.authorization_grant_id,
            candidate,
            purpose=request.purpose,
            provider=request.provider,
            mode=authorization_context.mode,
            now=now,
            skill_digest=authorization_context.skill_digest,
            dependency_digest=authorization_context.dependency_digest,
            collection_id=authorization_context.collection_id,
            collection_snapshot_hash=authorization_context.collection_snapshot_hash,
            selection_snapshot_hash=authorization_context.selection_snapshot_hash,
        )

    def _match_request(self, row: Mapping[str, Any], request: FetchRequest) -> None:
        persisted = _request_from_row(row)
        if persisted != request:
            raise FetchRejected("fetch request fields do not match the persisted immutable request")

    def _load_candidate(self, candidate_id: str) -> AccessLocationCandidate:
        row = self.database.connection.execute(
            "SELECT * FROM download_candidates WHERE candidate_id = ?", (candidate_id,)
        ).fetchone()
        if row is None:
            raise FetchRejected("download candidate no longer exists")
        return _candidate_from_row(row)

    def _completion_failure(
        self,
        request: FetchRequest,
        candidate: AccessLocationCandidate,
        terms: ProviderTerms | None,
        authorization_context: AuthorizationContext,
        completed_at: datetime,
        *,
        connection: Any | None = None,
    ) -> tuple[DownloadStatus, str] | None:
        reader = connection or self.database.connection
        row = reader.execute(
            "SELECT * FROM fetch_requests WHERE request_id = ?", (request.request_id,)
        ).fetchone()
        if row is None or row["status"] != "consumed":
            return DownloadStatus.FAILED_TERMINAL, "stale_fencing_token"
        if completed_at >= _as_datetime(row["expires_at"]):
            return DownloadStatus.FAILED_RETRYABLE, "fetch_request_expired_during_fetch"
        current = reader.execute(
            """SELECT MAX(fencing_token) FROM fetch_requests
               WHERE candidate_id = ? AND provider = ? AND policy_hash = ?
                 AND purpose = ? AND authorization_grant_hash IS ?""",
            (
                request.candidate_id,
                request.provider,
                row["policy_hash"],
                request.purpose,
                request.authorization_grant_hash,
            ),
        ).fetchone()[0]
        if current != request.fencing_token:
            return DownloadStatus.FAILED_TERMINAL, "stale_fencing_token"
        expected_identity = self._request_identity_hash(
            candidate,
            request.purpose,
            terms,
            request.authorization_grant_hash,
            request.fencing_token,
            authorization_context if request.authorization_grant_id else None,
        )
        if (
            row["policy_hash"] != self.policy.hash
            or request.request_id != "fetch-" + expected_identity[:32]
            or request.idempotency_key != "download:" + expected_identity
        ):
            return DownloadStatus.FAILED_TERMINAL, "policy_binding_changed_during_fetch"
        has_grant = False
        if request.authorization_grant_id:
            try:
                grant = self._require_grant_from_request(
                    request, candidate, completed_at, authorization_context
                )
            except GrantError:
                return DownloadStatus.MANUAL_REQUIRED, "authorization_changed_during_fetch"
            if grant.content_hash != request.authorization_grant_hash:
                return DownloadStatus.MANUAL_REQUIRED, "authorization_changed_during_fetch"
            has_grant = True
        outcome = self.policy.decide(
            candidate, request.purpose, terms, has_grant=has_grant
        )
        if outcome.status is not FetchDecisionStatus.ALLOW:
            return DownloadStatus.FAILED_TERMINAL, "policy_changed_during_fetch"
        return None

    def _validate_pdf_response(self, response: HTTPResponse) -> str | None:
        mime = _response_mime(response.headers)
        if mime != "application/pdf":
            return "invalid_pdf_mime"
        if len(response.body) < self.policy.min_pdf_bytes:
            return "pdf_too_small"
        if len(response.body) > self.policy.max_pdf_bytes:
            return "pdf_too_large"
        if not response.body.startswith(b"%PDF-"):
            return "invalid_pdf_magic"
        if not _pdf_is_parseable(response.body):
            return "invalid_pdf_structure"
        return None

    def _finish_failure(
        self,
        attempt_id: str,
        request: FetchRequest,
        candidate: AccessLocationCandidate,
        status: DownloadStatus,
        category: str,
        *,
        attempt_status: str | None = None,
        http_status: int | None = None,
        detail: str | None = None,
    ) -> DownloadResult:
        persisted_status = attempt_status or status.value
        with self.database.transaction() as connection:
            connection.execute(
                """UPDATE download_attempts
                   SET result_status = ?, failure_category = ?, http_status = ?, browser_result_json = ?
                   WHERE download_attempt_id = ? AND result_status = 'pending'""",
                (
                    persisted_status,
                    category,
                    http_status,
                    _json({"detail": detail}) if detail else None,
                    attempt_id,
                ),
            )
        return DownloadResult(
            request_id=request.request_id,
            paper_id=candidate.paper_id,
            status=status,
            provider=request.provider,
            source_url=candidate.url,
            error_code=category,
            authorization_grant_id=request.authorization_grant_id,
        )

    def _existing_result(self, request_id: str) -> DownloadResult:
        row = self.database.connection.execute(
            """SELECT da.*, dc.paper_id, dc.url, a.sha256
               FROM download_attempts da
               JOIN download_candidates dc ON dc.candidate_id = da.candidate_id
               LEFT JOIN artifacts a ON a.artifact_id = da.artifact_id
               WHERE da.fetch_request_id = ? ORDER BY da.attempted_at DESC LIMIT 1""",
            (request_id,),
        ).fetchone()
        if row is None or row["result_status"] == "pending":
            raise FetchRejected("consumed fetch request has no completed result")
        status = _domain_status(row["result_status"], row["failure_category"])
        return DownloadResult(
            request_id=request_id,
            paper_id=row["paper_id"],
            status=status,
            provider=row["provider"],
            artifact_id=row["artifact_id"],
            content_hash=row["sha256"],
            source_url=row["url"],
            downloaded_at=row["attempted_at"] if status is DownloadStatus.DOWNLOADED else None,
            error_code=row["failure_category"],
            authorization_grant_id=row["authorization_grant_id"],
        )

    def _set_request_status(self, request_id: str, status: str, *, expected: str) -> None:
        with self.database.transaction() as connection:
            connection.execute(
                "UPDATE fetch_requests SET status = ? WHERE request_id = ? AND status = ?",
                (status, request_id, expected),
            )


class _SameHostRedirects(HTTPRedirectHandler):
    def __init__(self, host: str) -> None:
        super().__init__()
        self.host = host

    def redirect_request(
        self,
        req: Any,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> Any:
        if urlsplit(newurl).hostname != self.host:
            raise URLError("cross-host redirects are disabled")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def urllib_fetch(
    url: str, *, timeout: float = 30.0, max_bytes: int = 209_715_200
) -> HTTPResponse:
    """Fetch one public URL without browser state, cookies, or authentication."""

    split = urlsplit(url)
    if not split.hostname:
        raise OSError("download URL has no host")
    _require_public_dns(split.hostname)
    request = Request(url, headers={"Accept": "application/pdf", "User-Agent": "paper-agent/2"})
    opener = build_opener(_SameHostRedirects(split.hostname))
    try:
        with opener.open(request, timeout=timeout) as response:
            return HTTPResponse(
                response.status,
                dict(response.headers.items()),
                response.read(max_bytes + 1),
                response.geturl(),
            )
    except HTTPError as error:
        return HTTPResponse(
            error.code,
            dict(error.headers.items()),
            error.read(max_bytes + 1),
            error.geturl(),
        )


def _require_public_dns(host: str) -> None:
    addresses = {
        ipaddress.ip_address(item[4][0])
        for item in socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    }
    if not addresses or any(not address.is_global for address in addresses):
        raise OSError("download host resolves to a private or local address")


def _normalize_candidate(
    candidate: AccessLocationCandidate, *, now: datetime | str | None
) -> AccessLocationCandidate:
    if not candidate.candidate_id or not candidate.paper_id or not candidate.resolver:
        raise DownloadError("candidate_id, paper_id, and resolver are required")
    if (
        candidate.raw_evidence_hash is None
        or re.fullmatch(r"[a-f0-9]{64}", candidate.raw_evidence_hash) is None
    ):
        raise DownloadError("candidate raw_evidence_hash must be a lowercase SHA-256 digest")
    if not candidate.provenance:
        raise DownloadError("candidate provenance is required")
    split = urlsplit(candidate.url)
    if split.scheme not in {"http", "https"} or not split.hostname:
        raise DownloadError("candidate URL must be an absolute HTTP(S) URL")
    host = split.hostname.lower()
    if _host_is_obviously_private(host):
        raise DownloadError("candidate URL must not target a private or local host")
    if candidate.host and candidate.host.lower() != host:
        raise DownloadError("candidate host does not match its URL")
    retrieved_at = candidate.retrieved_at
    if retrieved_at is None:
        if now is None:
            raise DownloadError("candidate retrieved_at or persist time is required")
        retrieved_at = _iso(_as_datetime(now))
    else:
        retrieved_at = _iso(_as_datetime(retrieved_at))
    return AccessLocationCandidate(
        candidate_id=candidate.candidate_id,
        paper_id=candidate.paper_id,
        resolver=candidate.resolver,
        url=candidate.url,
        landing_url=candidate.landing_url,
        host=host,
        publication_version=candidate.publication_version,
        license=candidate.license,
        access_basis=candidate.access_basis,
        retrieved_at=retrieved_at,
        raw_evidence_hash=candidate.raw_evidence_hash,
        provenance=dict(candidate.provenance),
    )


def _candidate_row_values(candidate: AccessLocationCandidate) -> tuple[object, ...]:
    return (
        candidate.candidate_id,
        candidate.paper_id,
        candidate.resolver,
        candidate.url,
        candidate.landing_url,
        candidate.publication_version.value,
        candidate.host,
        candidate.license,
        candidate.access_basis.value,
        candidate.retrieved_at,
        candidate.raw_evidence_hash,
        _json(candidate.provenance),
    )


def _candidate_from_row(row: Mapping[str, Any]) -> AccessLocationCandidate:
    return AccessLocationCandidate(
        candidate_id=row["candidate_id"],
        paper_id=row["paper_id"],
        resolver=row["resolver"],
        url=row["url"],
        landing_url=row["landing_url"],
        host=row["host"],
        publication_version=PublicationVersion(row["publication_version"] or "unknown"),
        license=row["license"],
        access_basis=AccessBasis(row["access_basis"]),
        retrieved_at=row["retrieved_at"],
        raw_evidence_hash=row["raw_evidence_hash"],
        provenance=json.loads(row["provenance_json"]),
    )


def _request_from_row(row: Mapping[str, Any]) -> FetchRequest:
    return FetchRequest(
        request_id=row["request_id"],
        candidate_id=row["candidate_id"],
        policy_version=row["policy_version"],
        purpose=row["purpose"],
        provider=row["provider"],
        created_at=row["created_at"],
        expires_at=row["expires_at"],
        idempotency_key=row["idempotency_key"],
        authorization_grant_id=row["authorization_grant_id"],
        authorization_grant_hash=row["authorization_grant_hash"],
        fencing_token=row["fencing_token"],
    )


def _http_failure_status(status: int) -> tuple[DownloadStatus, str, str] | None:
    if status == 200:
        return None
    if status == 206:
        return DownloadStatus.FAILED_RETRYABLE, "failed_retryable", "partial_http_response"
    if status in {408, 425, 429} or 500 <= status < 600:
        return DownloadStatus.FAILED_RETRYABLE, "failed_retryable", f"http_{status}"
    if status in {401, 403}:
        return DownloadStatus.AUTH_REQUIRED, "auth_required", "auth_required"
    if status in {404, 410}:
        return DownloadStatus.NOT_AVAILABLE, "not_available", "not_available"
    return DownloadStatus.FAILED_TERMINAL, "failed_terminal", f"http_{status}"


def _domain_status(result_status: str, category: str | None) -> DownloadStatus:
    if result_status == "downloaded":
        return DownloadStatus.DOWNLOADED
    if result_status == "failed_retryable":
        return DownloadStatus.FAILED_RETRYABLE
    if result_status == "auth_required":
        return DownloadStatus.AUTH_REQUIRED
    if result_status == "not_available":
        return DownloadStatus.NOT_AVAILABLE
    if result_status == "manual_required":
        return DownloadStatus.MANUAL_REQUIRED
    return DownloadStatus.FAILED_TERMINAL


def _response_mime(headers: Mapping[str, str]) -> str | None:
    content_type = next(
        (value for key, value in headers.items() if key.lower() == "content-type"), None
    )
    if not content_type:
        return None
    message = Message()
    message["content-type"] = content_type
    return message.get_content_type().lower()


def _final_url_error(
    candidate: AccessLocationCandidate, response: HTTPResponse
) -> str | None:
    if response.final_url is None:
        return None
    final = urlsplit(response.final_url)
    if final.scheme not in {"http", "https"} or not final.hostname:
        return "unsafe_final_url"
    host = final.hostname.lower()
    if _host_is_obviously_private(host):
        return "unsafe_final_url"
    if host != candidate.host:
        return "cross_host_redirect_denied"
    return None


def _host_is_obviously_private(host: str) -> bool:
    if host == "localhost" or host.endswith((".localhost", ".local")):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return not address.is_global


def _pdf_is_parseable(payload: bytes) -> bool:
    try:
        reader = PdfReader(BytesIO(payload), strict=False)
        return len(reader.pages) > 0
    except PdfReadError:
        return False


def _required_text(document: Mapping[str, Any], key: str) -> str:
    value = document.get(key)
    if not isinstance(value, str) or not value:
        raise PolicyError(f"{key} must be a non-empty string")
    return value


def _text_list(document: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = document.get(key)
    if not isinstance(value, list) or not value or any(not isinstance(item, str) for item in value):
        raise PolicyError(f"{key} must be a non-empty string list")
    return tuple(value)


def _positive_int(document: Mapping[str, Any], key: str) -> int:
    value = document.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise PolicyError(f"{key} must be a positive integer")
    return value


def _license_key(value: str) -> str:
    return value.strip().lower().rstrip("/")


def _as_datetime(value: datetime | str) -> datetime:
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamps must include a timezone")
    return parsed.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
