"""Thin CLI-facing orchestration for the Stage 3 download chain.

This module deliberately does not drive an authorized browser skill.  An
unavailable authenticated provider is left to the existing Stage 3 manual
queue path; public fetches remain governed by ``DownloadService``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
import json
from pathlib import Path
import sysconfig
from typing import Any

from .artifacts import ArtifactStore
from .authorized_skill_adapter import (
    AuditedAuthorizedSkillAdapter,
    AuthorizedSkillQueue,
    SkillQueueItem,
)
from .authorized_skill_runtime import AuthorizedSkillRuntime
from .authorized_luna import AuthorizedLunaPlanner
from .canonical import content_hash
from .domain import DownloadResult, DownloadStatus, FilterStatus, PaperSource
from .download_providers import (
    DEFAULT_PROVIDER_ORDER,
    DEFAULT_RESOLVER_ORDER,
    MetadataResolverTransport,
    ProbeContext,
    ResolverContext,
    default_download_provider_registry,
    default_resolver_registry,
)
from .downloads import (
    DownloadAccessPolicy,
    DownloadService,
    HTTPResponse,
    ProviderTerms,
    urllib_fetch,
)
from .grants import GrantStore
from .http_transport import ControlledHTTPTransport
from .repository import PaperRepository
from .runs import RunStatus, RunStore
from .stage3_pipeline import (
    AuthorizedSkillOptions,
    LunaPlanner,
    Stage3Paper,
    Stage3PaperResult,
    Stage3Pipeline,
    Stage3RunResult,
)
from .stage3_luna_decisions import Stage3LunaDecisionStore
from .stage3_metadata_lookup import (
    PublicMetadataTransport,
    Stage3MetadataLookup,
    default_metadata_lookup_registry,
)
from .storage import Database


@dataclass(frozen=True, slots=True)
class Stage3DownloadResult:
    run_id: str
    paper_ids: tuple[str, ...]
    status: str
    dry_run: bool
    run: Stage3RunResult | None = None
    planned_decisions: tuple[tuple[str, str, str], ...] = ()
    authorized_queue_path: Path | None = None


@dataclass(frozen=True, slots=True)
class AuthorizedSkillHandoffOptions:
    """Explicit local paths for an audited, attended browser handoff."""

    queue_path: Path
    output_dir: Path
    skill_roots: tuple[Path, ...]
    original_zip: Path | None = None
    audit_manifest: Path | None = None


class Stage3DownloadService:
    """Select canonical papers and delegate Stage 3 work to existing services."""

    def __init__(
        self,
        database: Database,
        config: Mapping[str, Any],
        *,
        config_root: str | Path,
        artifact_root: str | Path,
        provider_terms: Mapping[str, ProviderTerms] | None = None,
        fetcher: Callable[[str], HTTPResponse] = urllib_fetch,
        lookup: MetadataResolverTransport | None = None,
        metadata_transport: PublicMetadataTransport | None = None,
        clock: Callable[[], datetime] | None = None,
        authorized_luna_planner: LunaPlanner | None = None,
    ) -> None:
        self.database = database
        self.config_root = Path(config_root)
        self.artifact_root = Path(artifact_root)
        self.provider_terms = {**_safe_provider_terms(), **dict(provider_terms or {})}
        self.fetcher = fetcher
        self.clock = _trusted_clock(clock)
        self.authorized_luna_planner = authorized_luna_planner
        self.download_config = _download_config(config)
        _require_frozen_routing(self.download_config)
        self.lookup = lookup or _configured_metadata_lookup(
            self.download_config,
            transport=metadata_transport,
            clock=self.clock,
        )

    def select_papers(
        self,
        *,
        paper_ids: Sequence[str] = (),
        filter_run_id: str | None = None,
    ) -> tuple[Stage3Paper, ...]:
        """Select explicit IDs, or the latest relevant/needs-review Stage 2 rows."""
        selected = _selected_ids(
            self.database,
            paper_ids=paper_ids,
            filter_run_id=filter_run_id,
        )
        repository = PaperRepository(self.database)
        papers: list[Stage3Paper] = []
        for paper_id in selected:
            paper = repository.get_paper(paper_id)
            if paper is None:
                raise ValueError(f"paper does not exist: {paper_id}")
            sources = _sources_for(self.database, paper_id)
            papers.append(Stage3Paper(
                paper=paper,
                official_sources=tuple(
                    source for source in sources
                    if source.host_type in {"official", "publisher", "venue"}
                ),
                lookup=self.lookup,
                matched_arxiv=paper.arxiv_id is not None,
            ))
        return tuple(papers)

    def run(
        self,
        *,
        paper_ids: Sequence[str] = (),
        filter_run_id: str | None = None,
        authorization_grant_id: str | None = None,
        run_id: str | None = None,
        dry_run: bool = False,
        authorized_skill: AuthorizedSkillHandoffOptions | None = None,
    ) -> Stage3DownloadResult:
        """Run or safely validate the frozen public-download chain.

        ``authorization_grant_id`` is the only authorization input.  This
        adapter never accepts or derives an inline scope from configuration.
        """
        timestamp = _timestamp(self.clock())
        papers = _normalize_source_timestamps(
            self.select_papers(paper_ids=paper_ids, filter_run_id=filter_run_id)
        )
        selected_ids = tuple(item.paper.paper_id for item in papers)
        policy = DownloadAccessPolicy.load(_policy_path(self.config_root, self.download_config))
        identity = {
            "paper_ids": selected_ids,
            "filter_run_id": filter_run_id,
            "authorization_grant_id": authorization_grant_id,
            "download_config": self.download_config,
        }
        input_hash = content_hash(identity)
        config_hash = content_hash(self.download_config)
        resolved_run_id = run_id or f"stage3-{input_hash[:16]}"
        if authorization_grant_id is not None:
            GrantStore(self.database).load(
                authorization_grant_id, kind="download", now=timestamp
            )
        resolvers = default_resolver_registry()
        handoff = self._authorized_handoff(
            papers,
            resolvers=resolvers,
            now=timestamp,
            authorization_grant_id=authorization_grant_id,
            options=authorized_skill,
            prepare_queue=not dry_run,
        )
        service = DownloadService(
            self.database,
            ArtifactStore(self.artifact_root),
            policy,
            self.provider_terms,
            self.fetcher,
            clock=self.clock,
            provider_fetchers=(
                {"authorized_skill": handoff.queue.fetch_response}
                if handoff is not None else None
            ),
        )
        adapter = AuditedAuthorizedSkillAdapter(service, handoff.queue) if handoff else None
        providers = default_download_provider_registry(service, authorized_skill=adapter)
        if dry_run:
            decisions = self._validate_without_writes(
                papers,
                resolvers=resolvers,
                providers=providers,
                purpose=str(self.download_config["purpose"]),
                now=timestamp,
                authorization_grant_id=authorization_grant_id,
            )
            return Stage3DownloadResult(
                resolved_run_id, selected_ids, "validated", True,
                planned_decisions=decisions,
                authorized_queue_path=(handoff.queue.csv_path if handoff else None),
            )

        runs = RunStore(self.database)
        run = runs.create(
            run_id=resolved_run_id,
            stage="stage-3-download",
            input_hash=input_hash,
            config_hash=config_hash,
            implementation_version="stage3-cli-v1",
        )
        if run.status is RunStatus.DRAFT:
            runs.transition(resolved_run_id, RunStatus.APPROVED, at=timestamp)
            run = runs.transition(resolved_run_id, RunStatus.RUNNING, at=timestamp)
        elif run.status in {RunStatus.INCOMPLETE, RunStatus.FAILED}:
            run = runs.transition(resolved_run_id, RunStatus.RUNNING, at=timestamp)

        pipeline = Stage3Pipeline(
            resolvers=resolvers,
            providers=providers,
            purpose=str(self.download_config["purpose"]),
            now=timestamp,
            run_id=resolved_run_id,
            manual_queue=PaperRepository(self.database),
            authorized=AuthorizedSkillOptions(
                enabled=handoff is not None,
                runtime=handoff.runtime if handoff else None,
                grant_store=GrantStore(self.database) if handoff else None,
                authorization_grant_id=authorization_grant_id if handoff else None,
                planner=(
                    _DurableAuthorizedLunaPlanner(
                        Stage3LunaDecisionStore(
                            self.database, resolved_run_id, authorization_grant_id,
                        ),
                        self.authorized_luna_planner or AuthorizedLunaPlanner(),
                        timestamp,
                    )
                    if handoff is not None and authorization_grant_id is not None
                    else None
                ),
            ),
            public_authorization_grant_id=authorization_grant_id,
        )
        result = pipeline.run(
            papers, completed=_completed_downloads(self.database, resolved_run_id)
        )
        if run.status is RunStatus.RUNNING:
            status = (
                RunStatus.COMPLETE
                if all(item.status.value == "downloaded" for item in result.papers)
                else RunStatus.INCOMPLETE
            )
            runs.transition(resolved_run_id, status, at=timestamp)
        return Stage3DownloadResult(
            resolved_run_id,
            selected_ids,
            "complete" if all(item.status.value == "downloaded" for item in result.papers) else "incomplete",
            False,
            result,
            authorized_queue_path=(handoff.queue.csv_path if handoff else None),
        )

    def _authorized_handoff(
        self,
        papers: Sequence[Stage3Paper],
        *,
        resolvers,
        now: str,
        authorization_grant_id: str | None,
        options: AuthorizedSkillHandoffOptions | None,
        prepare_queue: bool,
    ) -> _AuthorizedHandoff | None:
        configured = self.download_config.get("authorized_skill", {})
        if not isinstance(configured, Mapping) or not configured.get("enabled"):
            return None
        if authorization_grant_id is None or options is None:
            return None
        runtime = AuthorizedSkillRuntime(
            enabled=True,
            skill_roots=options.skill_roots,
            original_zip=options.original_zip,
            audit_manifest=options.audit_manifest,
        )
        ready = runtime.require_ready()
        queue = AuthorizedSkillQueue(ready, options.queue_path, options.output_dir)
        if prepare_queue:
            items = _queue_items(papers, resolvers, now)
            if items:
                queue.prepare(items)
            else:
                return None
        return _AuthorizedHandoff(runtime, queue)

    def _validate_without_writes(
        self,
        papers: Sequence[Stage3Paper],
        *,
        resolvers,
        providers,
        purpose: str,
        now: str,
        authorization_grant_id: str | None,
    ) -> tuple[tuple[str, str, str], ...]:
        """Exercise exact probe/grant validation and roll back every database change."""
        decisions: list[tuple[str, str, str]] = []
        try:
            with self.database.transaction():
                for item in papers:
                    candidates = resolvers.resolve(
                        ResolverContext(
                            paper=item.paper,
                            official_sources=item.official_sources,
                            lookup=item.lookup,
                            matched_arxiv=item.matched_arxiv,
                            include_arxiv_candidates=item.include_arxiv_candidates,
                            retrieved_at=now,
                        )
                    )
                    for candidate in candidates:
                        attempt = providers.probe(candidate, ProbeContext(
                            purpose, now, authorization_grant_id=authorization_grant_id,
                        ))
                        decisions.append((item.paper.paper_id, attempt.provider, attempt.decision.status.value))
                raise _RollbackDryRun
        except _RollbackDryRun:
            return tuple(decisions)


class _RollbackDryRun(Exception):
    pass


@dataclass(frozen=True, slots=True)
class _AuthorizedHandoff:
    runtime: AuthorizedSkillRuntime
    queue: AuthorizedSkillQueue


@dataclass(frozen=True, slots=True)
class _DurableAuthorizedLunaPlanner:
    store: Stage3LunaDecisionStore
    planner: LunaPlanner
    decided_at: str

    def __call__(self, control):
        return self.store.decide(control, self.planner, decided_at=self.decided_at)


def load_provider_terms(path: str | Path) -> dict[str, ProviderTerms]:
    """Load an explicit, reviewable provider-terms snapshot."""
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or set(value) != {"schema_version", "providers"}:
        raise ValueError("provider terms snapshot has unexpected or missing fields")
    if value["schema_version"] != "1" or not isinstance(value["providers"], Mapping):
        raise ValueError("provider terms snapshot must use schema_version 1")
    terms: dict[str, ProviderTerms] = {}
    for provider, document in value["providers"].items():
        if not isinstance(provider, str) or not isinstance(document, Mapping):
            raise ValueError("provider terms entries must be named objects")
        _validate_provider_terms(provider, document)
        terms[provider] = ProviderTerms(
            provider=provider,
            terms_version=document["terms_version"],
            evidence_url=document.get("evidence_url"),
            machine_readable=document["machine_readable"],
            allows_download=document.get("allows_download"),
            allows_storage=document.get("allows_storage"),
            allows_redistribution=document.get("allows_redistribution"),
            domain_allowlist=tuple(document["domain_allowlist"]),
        )
    return terms


def _validate_provider_terms(provider: str, document: Mapping[str, Any]) -> None:
    expected = {
        "terms_version", "evidence_url", "machine_readable", "allows_download",
        "allows_storage", "allows_redistribution", "domain_allowlist",
    }
    if set(document) != expected:
        raise ValueError(f"provider terms {provider} has unexpected or missing fields")
    if not isinstance(document["terms_version"], str) or not document["terms_version"]:
        raise ValueError(f"provider terms {provider} terms_version must be a non-empty string")
    if document["evidence_url"] is not None and not isinstance(document["evidence_url"], str):
        raise ValueError(f"provider terms {provider} evidence_url must be a string or null")
    if type(document["machine_readable"]) is not bool:
        raise ValueError(f"provider terms {provider} machine_readable must be a boolean")
    for field in ("allows_download", "allows_storage", "allows_redistribution"):
        if document[field] is not None and type(document[field]) is not bool:
            raise ValueError(f"provider terms {provider} {field} must be a boolean or null")
    domains = document["domain_allowlist"]
    if not isinstance(domains, list) or not all(isinstance(item, str) and item for item in domains):
        raise ValueError(f"provider terms {provider} domain_allowlist must be a string list")


def _trusted_clock(clock: Callable[[], datetime] | None) -> Callable[[], datetime]:
    source = clock or (lambda: datetime.now(UTC))

    def current() -> datetime:
        value = source()
        if not isinstance(value, datetime) or value.tzinfo is None:
            raise ValueError("Stage 3 clock must return a timezone-aware datetime")
        return value.astimezone(UTC)

    return current


def _timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _download_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    value = config.get("download")
    if not isinstance(value, Mapping):
        raise ValueError("configuration requires a download section")
    return value


def _require_frozen_routing(config: Mapping[str, Any]) -> None:
    if tuple(config.get("resolvers", ())) != DEFAULT_RESOLVER_ORDER:
        raise ValueError("download resolvers must use the frozen default order")
    if tuple(config.get("providers", ())) != DEFAULT_PROVIDER_ORDER:
        raise ValueError("download providers must use the frozen default order")


def _configured_metadata_lookup(
    config: Mapping[str, Any],
    *,
    transport: PublicMetadataTransport | None,
    clock: Callable[[], datetime],
) -> MetadataResolverTransport | None:
    value = config.get("metadata_lookup")
    if not isinstance(value, Mapping) or not value.get("enabled"):
        return None
    contact = value.get("contact")
    user_agent = value.get("user_agent")
    timeout_seconds = value.get("timeout_seconds")
    unpaywall_email = value.get("unpaywall_email")
    if not isinstance(contact, str) or not contact:
        raise ValueError("download metadata lookup requires a contact")
    if not isinstance(user_agent, str) or not user_agent:
        raise ValueError("download metadata lookup requires a user_agent")
    if not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
        raise ValueError("download metadata lookup requires a positive timeout_seconds")
    if not isinstance(unpaywall_email, str) or "@" not in unpaywall_email:
        raise ValueError("download metadata lookup requires an Unpaywall email")
    controlled = transport or ControlledHTTPTransport(
        contact=contact,
        user_agent=user_agent,
        timeout_seconds=float(timeout_seconds),
        environment={"UNPAYWALL_EMAIL": unpaywall_email},
    )
    return Stage3MetadataLookup(
        controlled,
        retrieved_at=clock,
        registry=default_metadata_lookup_registry(),
    )


def _policy_path(config_root: Path, config: Mapping[str, Any]) -> Path:
    path = Path(str(config["policy_matrix"]))
    if path.is_absolute():
        return path
    configured = config_root / path
    if configured.is_file():
        return configured
    repository = Path(__file__).resolve().parents[2] / path
    if repository.is_file():
        return repository
    return (
        Path(sysconfig.get_path("data"))
        / "share"
        / "paper-agent"
        / "policies"
        / path.name
    )


def _safe_provider_terms() -> dict[str, ProviderTerms]:
    return {
        provider: ProviderTerms(provider, "unconfigured", None, False, None, None)
        for provider in DEFAULT_PROVIDER_ORDER
    }


def _selected_ids(
    database: Database,
    *,
    paper_ids: Sequence[str],
    filter_run_id: str | None,
) -> tuple[str, ...]:
    explicit = tuple(sorted(set(paper_ids)))
    if explicit:
        if filter_run_id is not None:
            raise ValueError("paper_ids and filter_run_id are mutually exclusive")
        return explicit
    statuses = tuple(status.value for status in (FilterStatus.RELEVANT, FilterStatus.NEEDS_REVIEW))
    if filter_run_id is not None:
        rows = database.connection.execute(
            """SELECT paper_id FROM filter_decisions
               WHERE run_id = ? AND status IN (?, ?) ORDER BY paper_id""",
            (filter_run_id, *statuses),
        ).fetchall()
    else:
        rows = database.connection.execute(
            """SELECT decision.paper_id FROM filter_decisions AS decision
               WHERE decision.status IN (?, ?)
                 AND decision.rowid = (
                   SELECT latest.rowid FROM filter_decisions AS latest
                   WHERE latest.paper_id = decision.paper_id
                   ORDER BY latest.created_at DESC, latest.rowid DESC LIMIT 1
                 )
               ORDER BY decision.paper_id""",
            statuses,
        ).fetchall()
    selected = tuple(str(row["paper_id"]) for row in rows)
    if not selected:
        raise ValueError("no relevant or needs_review Stage 2 papers were selected")
    return selected


def _sources_for(database: Database, paper_id: str) -> tuple[PaperSource, ...]:
    rows = database.connection.execute(
        "SELECT * FROM paper_sources WHERE paper_id = ? ORDER BY source_id", (paper_id,)
    ).fetchall()
    return tuple(PaperSource.from_dict({
        **dict(row),
        "raw_metadata": json.loads(row["raw_metadata_json"]),
        "metadata_capabilities": json.loads(row["metadata_capabilities_json"]),
        "download_capabilities": json.loads(row["download_capabilities_json"]),
    }) for row in rows)


def _queue_items(
    papers: Sequence[Stage3Paper], resolvers, now: str
) -> tuple[SkillQueueItem, ...]:
    """Freeze at most one attended browser URL per DOI-backed paper."""
    items: list[SkillQueueItem] = []
    urls: set[str] = set()
    for item in papers:
        doi = item.paper.doi
        if doi is None or not doi.startswith("10.") or "/" not in doi:
            continue
        candidates = resolvers.resolve(ResolverContext(
            paper=item.paper,
            official_sources=item.official_sources,
            lookup=item.lookup,
            matched_arxiv=item.matched_arxiv,
            include_arxiv_candidates=item.include_arxiv_candidates,
            retrieved_at=now,
        ))
        candidate = next((value for value in candidates if value.url not in urls), None)
        if candidate is None:
            continue
        items.append(SkillQueueItem(item.paper.paper_id, doi, candidate.url, item.paper.title))
        urls.add(candidate.url)
    return tuple(items)


def _completed_downloads(
    database: Database, run_id: str
) -> dict[str, Stage3PaperResult]:
    """Resume only successful immutable fetches; failures stay retryable."""
    rows = database.connection.execute(
        """SELECT da.fetch_request_id, da.provider, da.authorization_grant_id,
                  da.attempted_at, dc.paper_id, dc.url, da.artifact_id, a.sha256
           FROM download_attempts AS da
           JOIN download_candidates AS dc ON dc.candidate_id = da.candidate_id
           LEFT JOIN artifacts AS a ON a.artifact_id = da.artifact_id
           WHERE da.run_id = ? AND da.result_status = 'downloaded'
           ORDER BY da.attempted_at DESC, da.download_attempt_id DESC""",
        (run_id,),
    ).fetchall()
    completed: dict[str, Stage3PaperResult] = {}
    for row in rows:
        paper_id = str(row["paper_id"])
        if paper_id in completed:
            continue
        result = DownloadResult(
            str(row["fetch_request_id"]),
            paper_id,
            DownloadStatus.DOWNLOADED,
            str(row["provider"]),
            artifact_id=row["artifact_id"],
            content_hash=row["sha256"],
            source_url=str(row["url"]),
            downloaded_at=row["attempted_at"],
            authorization_grant_id=row["authorization_grant_id"],
        )
        completed[paper_id] = Stage3PaperResult(
            paper_id, DownloadStatus.DOWNLOADED, "downloaded", (), result
        )
    return completed


def _normalize_source_timestamps(
    papers: Sequence[Stage3Paper],
) -> tuple[Stage3Paper, ...]:
    """Normalize SQLite's timezone-less source timestamp without changing its instant."""
    return tuple(replace(
        item,
        official_sources=tuple(
            source if _has_timezone(source.last_seen_at) else replace(
                source, last_seen_at=_as_utc(source.last_seen_at)
            )
            for source in item.official_sources
        ),
    ) for item in papers)


def _has_timezone(value: str | None) -> bool:
    if value is None:
        return False
    return value.endswith("Z") or "+" in value[10:] or "-" in value[10:]


def _as_utc(value: str | None) -> str | None:
    return f"{value}Z" if value is not None else None
