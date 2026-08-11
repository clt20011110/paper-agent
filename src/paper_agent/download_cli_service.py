"""Thin CLI-facing orchestration for the Stage 3 download chain.

This module deliberately does not drive an authorized browser skill.  An
unavailable authenticated provider is left to the existing Stage 3 manual
queue path; public fetches remain governed by ``DownloadService``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
import json
from pathlib import Path
import re
import sysconfig
from typing import Any
from urllib.parse import unquote, urlsplit

from .artifacts import ArtifactStore
from .authorized_skill_adapter import (
    AuditedAuthorizedSkillAdapter,
    AuthorizedSkillAdapterError,
    AuthorizedSkillQueue,
    SkillQueueItem,
    authorized_publisher_host_matches,
)
from .authorized_skill_runtime import AuthorizedSkillRuntime, AuthorizedSkillRuntimeError
from .authorized_luna import AuthorizedLunaPlanner
from .canonical import canonical_json, content_hash
from .domain import (
    AccessLocationCandidate,
    DownloadResult,
    DownloadStatus,
    FetchDecisionStatus,
    FilterStatus,
    PaperSource,
)
from .download_providers import (
    DEFAULT_PROVIDER_ORDER,
    DEFAULT_RESOLVER_ORDER,
    MetadataResolverTransport,
    ProbeContext,
    ResolverContext,
    ResolverRegistry,
    ResolverSnapshot,
    default_download_provider_registry,
    default_resolver_registry,
)
from .downloads import (
    AuthorizationContext,
    DownloadAccessPolicy,
    DownloadScopeBinding,
    DownloadService,
    FetchRejected,
    HTTPResponse,
    ProviderTerms,
    urllib_fetch,
)
from .grants import GrantError, GrantStore
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
    CONTROLLED_HTTP_TRANSPORT_IMPLEMENTATION_VERSION,
    PublicMetadataTransport,
    Stage3MetadataLookup,
    default_metadata_lookup_registry,
)
from .storage import Database


IMPLEMENTATION_VERSION = "stage3-cli-v6"


@dataclass(frozen=True, slots=True)
class Stage3DownloadResult:
    run_id: str
    paper_ids: tuple[str, ...]
    status: str
    dry_run: bool
    run: Stage3RunResult | None = None
    planned_decisions: tuple[tuple[str, str, str], ...] = ()
    authorized_queue_path: Path | None = None
    authorization_scope: DownloadScopeBinding = DownloadScopeBinding()
    resolver_snapshot: ResolverSnapshot | None = None


@dataclass(frozen=True, slots=True)
class AuthorizedSkillHandoffOptions:
    """Explicit local paths for an audited, attended browser handoff."""

    queue_path: Path
    output_dir: Path
    skill_roots: tuple[Path, ...]
    original_zip: Path | None = None
    audit_manifest: Path | None = None


_TERMINAL_DOWNLOAD_STATUSES = frozenset({
    DownloadStatus.DOWNLOADED,
    DownloadStatus.NOT_AVAILABLE,
    DownloadStatus.FAILED_TERMINAL,
})
_AUTHORIZED_SKILL_EXECUTION_MODE = "attended"


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
        scope_membership: Callable[[str, str, str, str | None], bool] | None = None,
        resolver_registry: ResolverRegistry | None = None,
    ) -> None:
        self.database = database
        self.config_root = Path(config_root)
        self.artifact_root = Path(artifact_root)
        self.provider_terms = {**_safe_provider_terms(), **dict(provider_terms or {})}
        self.fetcher = fetcher
        self.clock = _trusted_clock(clock)
        self.authorized_luna_planner = authorized_luna_planner
        self.scope_membership = scope_membership
        self.resolver_registry = resolver_registry
        self.download_config = _download_config(config)
        _require_frozen_routing(self.download_config)
        self.lookup = (
            lookup
            if lookup is not None
            else _configured_metadata_lookup(
                self.download_config,
                transport=metadata_transport,
                clock=self.clock,
            )
        )

    def select_papers(
        self,
        *,
        paper_ids: Sequence[str] | None = None,
        filter_run_id: str | None = None,
        include_needs_review: bool = False,
    ) -> tuple[Stage3Paper, ...]:
        """Select explicit IDs or Stage 2 rows approved for downstream work."""
        selected = select_stage3_paper_ids(
            self.database,
            paper_ids=paper_ids,
            filter_run_id=filter_run_id,
            include_needs_review=include_needs_review,
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
        paper_ids: Sequence[str] | None = None,
        filter_run_id: str | None = None,
        source_filter_run_id: str | None = None,
        include_needs_review: bool = False,
        authorization_grant_id: str | None = None,
        run_id: str | None = None,
        dry_run: bool = False,
        authorized_skill: AuthorizedSkillHandoffOptions | None = None,
        authorization_scope: DownloadScopeBinding = DownloadScopeBinding(),
    ) -> Stage3DownloadResult:
        """Run or safely validate the frozen public-download chain.

        ``authorization_grant_id`` is the only authorization input.  This
        adapter never accepts or derives an inline scope from configuration.
        """
        timestamp = _timestamp(self.clock())
        authorization_context: AuthorizationContext | None = None
        if authorization_grant_id is not None:
            grant = GrantStore(self.database).load(
                authorization_grant_id,
                kind="download",
                now=timestamp if dry_run else None,
            )
            authorization_context = authorization_scope.authorization_context(
                mode=str(grant.document["mode"])
            )
        papers = _normalize_source_timestamps(
            self.select_papers(
                paper_ids=paper_ids,
                filter_run_id=filter_run_id,
                include_needs_review=include_needs_review,
            )
        )
        selected_ids = tuple(item.paper.paper_id for item in papers)
        policy = DownloadAccessPolicy.load(_policy_path(self.config_root, self.download_config))
        resolvers = self.resolver_registry or default_resolver_registry()
        configured_resolver_order = tuple(self.download_config["resolvers"])
        resolver_snapshot = resolvers.freeze(
            configured_order=configured_resolver_order,
            runtime_config=_resolver_runtime_config(
                resolvers.names, self.download_config, self.lookup
            ),
            download_config_hash=content_hash(self.download_config),
        )
        identity = {
            "paper_ids": selected_ids,
            "per_paper_resolver_inputs": _per_paper_resolver_inputs(papers),
            "filter_run_id": filter_run_id,
            "source_filter_run_id": source_filter_run_id,
            "include_needs_review": include_needs_review,
            "authorization_grant_id": authorization_grant_id,
            "authorization_scope": authorization_scope.to_dict(),
            "resolver_snapshot_hash": resolver_snapshot.snapshot_hash,
            "authorized_handoff": _authorized_handoff_identity(authorized_skill),
        }
        input_hash = content_hash(identity)
        config_hash = resolver_snapshot.snapshot_hash
        resolved_run_id = run_id or f"stage3-{input_hash[:16]}"
        artifact_store = ArtifactStore(self.artifact_root)
        fetcher = self.fetcher
        if fetcher is urllib_fetch:
            trusted_cidrs = self.download_config.get("trusted_egress_proxy_cidrs", ())
            fetcher = lambda url: urllib_fetch(
                url,
                max_bytes=policy.max_pdf_bytes,
                trusted_egress_proxy_cidrs=trusted_cidrs,
            )
        service = DownloadService(
            self.database,
            artifact_store,
            policy,
            self.provider_terms,
            fetcher,
            scope_membership=self.scope_membership,
            clock=self.clock,
        )
        providers = default_download_provider_registry(service)
        if dry_run:
            decisions = self._validate_without_writes(
                papers,
                resolvers=resolvers,
                providers=providers,
                service=service,
                purpose=str(self.download_config["purpose"]),
                now=timestamp,
                run_id=resolved_run_id,
                authorization_grant_id=authorization_grant_id,
                authorization_context=authorization_context,
                authorized_skill=authorized_skill,
                authorization_scope=authorization_scope,
            )
            return Stage3DownloadResult(
                resolved_run_id, selected_ids, "validated", True,
                planned_decisions=decisions,
                authorization_scope=authorization_scope,
                resolver_snapshot=resolver_snapshot,
            )

        runs = RunStore(self.database)
        create_run = {
            "run_id": resolved_run_id,
            "stage": "stage-3-download",
            "input_hash": input_hash,
            "config_hash": config_hash,
            "implementation_version": IMPLEMENTATION_VERSION,
        }
        if runs.get(resolved_run_id) is not None:
            # Reject resume drift before creating a new global manifest.
            run = runs.create(**create_run)
            _persist_resolver_snapshot(
                self.database, artifact_store, resolver_snapshot
            )
        else:
            # A manifest persistence failure must not leave a draft run behind.
            artifact_hash = _persist_resolver_snapshot(
                self.database, artifact_store, resolver_snapshot
            )
            if artifact_hash != config_hash:
                raise ValueError("resolver snapshot artifact does not match config hash")
            run = runs.create(**create_run)
        if run.status is RunStatus.DRAFT:
            runs.transition(resolved_run_id, RunStatus.APPROVED, at=timestamp)
            run = runs.transition(resolved_run_id, RunStatus.RUNNING, at=timestamp)
        elif run.status in {RunStatus.INCOMPLETE, RunStatus.FAILED}:
            run = runs.transition(resolved_run_id, RunStatus.RUNNING, at=timestamp)

        manual_queue = _DeferredManualQueue(resolved_run_id, timestamp)
        # Phase one is intentionally public-only.  The authorized CSV does not
        # exist while public/OA candidates are being tried.
        public_pipeline = Stage3Pipeline(
            resolvers=resolvers,
            providers=providers,
            purpose=str(self.download_config["purpose"]),
            now=timestamp,
            run_id=resolved_run_id,
            manual_queue=manual_queue,
        )
        result = public_pipeline.run(
            papers,
            completed=_resume_checkpoints(self.database, resolved_run_id),
            checkpoint=lambda item: _save_checkpoint(
                self.database, resolved_run_id, item, timestamp
            ),
        )
        for item in result.papers:
            _save_checkpoint(self.database, resolved_run_id, item, timestamp)

        # Preserve non-browser download grants as a separate post-OA probe.
        # An authorized-skill-bound grant cannot authorize these providers and
        # therefore remains unresolved for the audited browser handoff below.
        if authorization_grant_id is not None:
            assert authorization_context is not None
            granted_public_pipeline = Stage3Pipeline(
                resolvers=resolvers,
                providers=providers,
                purpose=str(self.download_config["purpose"]),
                now=timestamp,
                run_id=resolved_run_id,
                manual_queue=manual_queue,
                public_authorization_grant_id=authorization_grant_id,
                public_authorization_context=authorization_context,
            )
            result = granted_public_pipeline.run(
                papers,
                completed=_resume_checkpoints(self.database, resolved_run_id),
                checkpoint=lambda item: _save_checkpoint(
                    self.database, resolved_run_id, item, timestamp
                ),
            )
            for item in result.papers:
                _save_checkpoint(self.database, resolved_run_id, item, timestamp)

        handoff = self._authorized_handoff(
            papers,
            public_result=result,
            service=service,
            run_id=resolved_run_id,
            now=timestamp,
            authorization_grant_id=authorization_grant_id,
            options=authorized_skill,
            authorization_scope=authorization_scope,
        )
        if handoff is not None:
            service.provider_fetchers["authorized_skill"] = handoff.queue.fetch_response
            adapter = AuditedAuthorizedSkillAdapter(service, handoff.queue)
            authorized_providers = default_download_provider_registry(
                service, authorized_skill=adapter
            )
            pipeline = Stage3Pipeline(
                resolvers=resolvers,
                providers=authorized_providers,
                purpose=str(self.download_config["purpose"]),
                now=timestamp,
                run_id=resolved_run_id,
                manual_queue=manual_queue,
                authorized=AuthorizedSkillOptions(
                    enabled=True,
                    runtime=handoff.runtime,
                    grant_store=GrantStore(self.database),
                    authorization_grant_id=authorization_grant_id,
                    mode=_AUTHORIZED_SKILL_EXECUTION_MODE,
                    collection_id=authorization_scope.collection_id,
                    collection_snapshot_hash=authorization_scope.collection_snapshot_hash,
                    selection_snapshot_hash=authorization_scope.selection_snapshot_hash,
                    planner=(
                        _DurableAuthorizedLunaPlanner(
                            Stage3LunaDecisionStore(
                                self.database,
                                resolved_run_id,
                                authorization_grant_id,
                            ),
                            self.authorized_luna_planner or AuthorizedLunaPlanner(),
                            timestamp,
                        )
                        if authorization_grant_id is not None
                        else None
                    ),
                    candidate_ids=handoff.candidate_ids,
                ),
            )
            result = pipeline.run(
                papers,
                completed=_resume_checkpoints(self.database, resolved_run_id),
                checkpoint=lambda item: _save_checkpoint(
                    self.database, resolved_run_id, item, timestamp
                ),
            )
            for item in result.papers:
                _save_checkpoint(self.database, resolved_run_id, item, timestamp)
        manual_queue.flush(PaperRepository(self.database), result)
        complete = all(
            item.status in _TERMINAL_DOWNLOAD_STATUSES for item in result.papers
        )
        if run.status is RunStatus.RUNNING:
            status = RunStatus.COMPLETE if complete else RunStatus.INCOMPLETE
            runs.transition(resolved_run_id, status, at=timestamp)
        public_status = _stage3_result_status(result)
        return Stage3DownloadResult(
            resolved_run_id,
            selected_ids,
            public_status,
            False,
            result,
            authorized_queue_path=(handoff.queue.csv_path if handoff else None),
            authorization_scope=authorization_scope,
            resolver_snapshot=resolver_snapshot,
        )

    def _authorized_handoff(
        self,
        papers: Sequence[Stage3Paper],
        *,
        public_result: Stage3RunResult,
        service: DownloadService,
        run_id: str,
        now: str,
        authorization_grant_id: str | None,
        options: AuthorizedSkillHandoffOptions | None,
        authorization_scope: DownloadScopeBinding,
    ) -> _AuthorizedHandoff | None:
        configured = self.download_config.get("authorized_skill", {})
        if not isinstance(configured, Mapping) or not configured.get("enabled"):
            return None
        if authorization_grant_id is None or options is None:
            return None
        grant = GrantStore(self.database).load(
            authorization_grant_id, kind="download"
        )
        if grant.document["mode"] != _AUTHORIZED_SKILL_EXECUTION_MODE:
            # The audited browser skill currently declares attended-only
            # operation.  An unattended grant remains manual and never creates
            # a queue or browser side effect.
            return None
        try:
            runtime = AuthorizedSkillRuntime(
                enabled=True,
                skill_roots=options.skill_roots,
                original_zip=options.original_zip,
                audit_manifest=options.audit_manifest,
            )
            ready = runtime.require_ready()
        except AuthorizedSkillRuntimeError:
            return None
        if (
            ready.installed_content_sha256 is None
            or ready.dependency_lock_sha256 is None
        ):
            return None
        queue = AuthorizedSkillQueue(ready, options.queue_path, options.output_dir)
        plan = _queue_items(
            papers,
            public_result,
            service=service,
            authorization_grant_id=authorization_grant_id,
            purpose=str(self.download_config["purpose"]),
            now=now,
            skill_digest=ready.installed_content_sha256,
            dependency_digest=ready.dependency_lock_sha256,
            authorization_scope=authorization_scope,
        )
        if not plan.items:
            return None
        if queue.has_queue_file():
            frozen_items = queue.frozen_items()
            frozen_plan = _validate_frozen_queue(
                papers,
                frozen_items,
                service=service,
                run_id=run_id,
                queue_path=queue.csv_path,
                authorization_grant_id=authorization_grant_id,
                purpose=str(self.download_config["purpose"]),
                now=now,
                skill_digest=ready.installed_content_sha256,
                dependency_digest=ready.dependency_lock_sha256,
                authorization_scope=authorization_scope,
            )
            _reserve_queue_plan(
                service,
                frozen_plan,
                run_id=run_id,
                queue_path=queue.csv_path,
                authorization_grant_id=authorization_grant_id,
                purpose=str(self.download_config["purpose"]),
                now=now,
                skill_digest=ready.installed_content_sha256,
                dependency_digest=ready.dependency_lock_sha256,
            )
            queue.prepare(frozen_plan.items)
            frozen_keys = {
                (item.paper_id, item.doi.lower(), item.url, item.candidate_url)
                for item in frozen_plan.items
            }
            planned_keys = {
                (item.paper_id, item.doi.lower(), item.url, item.candidate_url)
                for item in plan.items
            }
            if not planned_keys <= frozen_keys:
                raise AuthorizedSkillAdapterError(
                    "authorized queue is immutable after creation"
                )
            planned_papers = {item.paper_id for item in plan.items}
            if any(
                item.paper_id not in planned_papers
                and not _has_authorized_attempt(
                    self.database,
                    run_id=run_id,
                    authorization_grant_id=authorization_grant_id,
                    paper_id=item.paper_id,
                    url=item.candidate_url,
                )
                for item in frozen_plan.items
            ):
                raise AuthorizedSkillAdapterError(
                    "authorized queue contains a paper outside the public fallback set"
                )
        else:
            with service.database.transaction():
                _reserve_queue_plan(
                    service,
                    plan,
                    run_id=run_id,
                    queue_path=queue.csv_path,
                    authorization_grant_id=authorization_grant_id,
                    purpose=str(self.download_config["purpose"]),
                    now=now,
                    skill_digest=ready.installed_content_sha256,
                    dependency_digest=ready.dependency_lock_sha256,
                )
                queue.prepare(plan.items)
        return _AuthorizedHandoff(runtime, queue, frozenset(plan.candidate_ids))

    def _validate_without_writes(
        self,
        papers: Sequence[Stage3Paper],
        *,
        resolvers,
        providers,
        service: DownloadService,
        purpose: str,
        now: str,
        run_id: str,
        authorization_grant_id: str | None,
        authorization_context: AuthorizationContext | None,
        authorized_skill: AuthorizedSkillHandoffOptions | None,
        authorization_scope: DownloadScopeBinding,
    ) -> tuple[tuple[str, str, str], ...]:
        """Exercise exact probe/grant validation and roll back every database change."""
        decisions: list[tuple[str, str, str]] = []
        authorized_candidates: dict[str, list[str]] = {}
        runtime_authorization = (
            authorization_context or authorization_scope.authorization_context()
        )
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
                            mode=runtime_authorization.mode,
                            skill_digest=runtime_authorization.skill_digest,
                            dependency_digest=runtime_authorization.dependency_digest,
                            collection_id=runtime_authorization.collection_id,
                            collection_snapshot_hash=runtime_authorization.collection_snapshot_hash,
                            selection_snapshot_hash=runtime_authorization.selection_snapshot_hash,
                        ))
                        decisions.append((item.paper.paper_id, attempt.provider, attempt.decision.status.value))
                        if attempt.decision.status in {
                            FetchDecisionStatus.NEEDS_GRANT,
                            FetchDecisionStatus.MANUAL,
                        }:
                            authorized_candidates.setdefault(
                                item.paper.paper_id, []
                            ).append(candidate.candidate_id)
                self._validate_dry_handoff(
                    papers,
                    authorized_candidates,
                    service=service,
                    purpose=purpose,
                    now=now,
                    run_id=run_id,
                    authorization_grant_id=authorization_grant_id,
                    options=authorized_skill,
                    decisions=decisions,
                    authorization_scope=authorization_scope,
                )
                raise _RollbackDryRun
        except _RollbackDryRun:
            return tuple(decisions)

    def _validate_dry_handoff(
        self,
        papers: Sequence[Stage3Paper],
        authorized_candidates: Mapping[str, Sequence[str]],
        *,
        service: DownloadService,
        purpose: str,
        now: str,
        run_id: str,
        authorization_grant_id: str | None,
        options: AuthorizedSkillHandoffOptions | None,
        decisions: list[tuple[str, str, str]],
        authorization_scope: DownloadScopeBinding,
    ) -> None:
        configured = self.download_config.get("authorized_skill", {})
        if (
            not isinstance(configured, Mapping)
            or not configured.get("enabled")
            or authorization_grant_id is None
            or options is None
        ):
            return
        grant = GrantStore(self.database).load(
            authorization_grant_id, kind="download", now=now
        )
        if grant.document["scope"]["provider"] != "authorized_skill":
            return
        if grant.document["mode"] != _AUTHORIZED_SKILL_EXECUTION_MODE:
            decisions.extend(
                (paper_id, "authorized_skill", "manual")
                for paper_id in sorted(authorized_candidates)
            )
            return
        runtime = AuthorizedSkillRuntime(
            enabled=True,
            skill_roots=options.skill_roots,
            original_zip=options.original_zip,
            audit_manifest=options.audit_manifest,
        )
        ready = runtime.require_ready()
        if (
            ready.installed_content_sha256 is None
            or ready.dependency_lock_sha256 is None
        ):
            raise AuthorizedSkillRuntimeError(
                "authorized skill audit has no frozen content or dependency digest"
            )
        dry_result = Stage3RunResult(tuple(
            Stage3PaperResult(
                item.paper.paper_id,
                DownloadStatus.MANUAL_REQUIRED,
                "dry_run_authorized_candidate",
                (),
                authorized_candidate_ids=tuple(
                    authorized_candidates.get(item.paper.paper_id, ())
                ),
            )
            for item in papers
        ))
        plan = _queue_items(
            papers,
            dry_result,
            service=service,
            authorization_grant_id=authorization_grant_id,
            purpose=purpose,
            now=now,
            skill_digest=ready.installed_content_sha256,
            dependency_digest=ready.dependency_lock_sha256,
            authorization_scope=authorization_scope,
        )
        planned = {item.paper_id for item in plan.items}
        decisions.extend(
            (
                paper_id,
                "authorized_skill",
                "allow" if paper_id in planned else "manual",
            )
            for paper_id in sorted(authorized_candidates)
        )
        queue = AuthorizedSkillQueue(ready, options.queue_path, options.output_dir)
        if queue.csv_path.is_file():
            _validate_frozen_queue(
                papers,
                queue.frozen_items(),
                service=service,
                run_id=run_id,
                queue_path=queue.csv_path,
                authorization_grant_id=authorization_grant_id,
                purpose=purpose,
                now=now,
                skill_digest=ready.installed_content_sha256,
                dependency_digest=ready.dependency_lock_sha256,
                authorization_scope=authorization_scope,
            )


class _RollbackDryRun(Exception):
    pass


@dataclass(frozen=True, slots=True)
class _AuthorizedHandoff:
    runtime: AuthorizedSkillRuntime
    queue: AuthorizedSkillQueue
    candidate_ids: frozenset[str]


@dataclass(frozen=True, slots=True)
class _AuthorizedQueuePlan:
    items: tuple[SkillQueueItem, ...]
    candidate_ids: tuple[str, ...]
    authorization_scope: DownloadScopeBinding


@dataclass(frozen=True, slots=True)
class _DeferredManualItem:
    queue_type: str
    dedup_key: str
    paper_id: str | None
    reason: Mapping[str, object]


@dataclass(slots=True)
class _DeferredManualQueue:
    run_id: str
    resolved_at: str
    items: dict[str, _DeferredManualItem] = field(default_factory=dict)

    def enqueue_manual(
        self,
        queue_type: str,
        dedup_key: str,
        paper_id: str | None,
        reason: Mapping[str, object],
    ) -> None:
        if paper_id is not None:
            self.items[paper_id] = _DeferredManualItem(
                queue_type, dedup_key, paper_id, reason
            )

    def flush(
        self, repository: PaperRepository, result: Stage3RunResult
    ) -> None:
        manual_ids = {
            item.paper_id
            for item in result.papers
            if item.status in {
                DownloadStatus.AUTH_REQUIRED,
                DownloadStatus.MANUAL_REQUIRED,
            }
        }
        for paper_id in sorted(manual_ids):
            item = self.items.get(paper_id)
            if item is not None:
                repository.enqueue_manual(
                    item.queue_type,
                    item.dedup_key,
                    item.paper_id,
                    item.reason,
                )
        for item in result.papers:
            if item.paper_id not in manual_ids:
                repository.resolve_manual(
                    "download",
                    f"{self.run_id}:{item.paper_id}",
                    {
                        "run_id": self.run_id,
                        "status": item.status.value,
                        "reason_code": item.reason_code,
                    },
                    resolved_at=self.resolved_at,
                )


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


def _resolver_runtime_config(
    resolver_names: Sequence[str],
    download_config: Mapping[str, Any],
    lookup: MetadataResolverTransport | None,
) -> dict[str, Mapping[str, Any]]:
    configured_lookup = download_config.get("metadata_lookup")
    lookup_config = (
        {
            "enabled": configured_lookup.get("enabled"),
            "user_agent": configured_lookup.get("user_agent"),
            "timeout_seconds": configured_lookup.get("timeout_seconds"),
            "contact_configured": bool(configured_lookup.get("contact")),
            "unpaywall_email_configured": bool(
                configured_lookup.get("unpaywall_email")
            ),
        }
        if isinstance(configured_lookup, Mapping)
        else {"enabled": False}
    )
    lookup_identity: Mapping[str, Any] | None = None
    if lookup is not None:
        identity = getattr(lookup, "canonical_identity", None)
        if not callable(identity):
            raise ValueError(
                "metadata resolver transport must expose canonical_identity"
            )
        current = identity()
        if not isinstance(current, Mapping):
            raise ValueError("metadata resolver canonical_identity must be a mapping")
        lookup_identity = dict(current)
        content_hash(lookup_identity)
    lookup_runtime = {
        "available": lookup is not None,
        "configuration": lookup_config,
        "identity": lookup_identity,
    }
    runtime: dict[str, Mapping[str, Any]] = {str(name): {} for name in resolver_names}
    if "publisher_public" in runtime:
        runtime["publisher_public"] = {
            "metadata_lookup": {"available": False}
        }
    for name in ("europe_pmc", "unpaywall"):
        if name in runtime:
            runtime[name] = {"metadata_lookup": lookup_runtime}
    if "arxiv" in runtime:
        runtime["arxiv"] = {
            "metadata_lookup": lookup_runtime,
            "matched_arxiv_source": "paper.arxiv_id",
        }
    return runtime


def _per_paper_resolver_inputs(
    papers: Sequence[Stage3Paper],
) -> list[dict[str, Any]]:
    """Freeze dynamic candidate controls separately from registry configuration."""

    return [
        {
            "paper_id": item.paper.paper_id,
            "doi": item.paper.doi,
            "arxiv_id": item.paper.arxiv_id,
            "matched_arxiv": item.matched_arxiv,
            "include_arxiv_candidates": item.include_arxiv_candidates,
            "official_sources": [
                source.to_dict()
                for source in sorted(
                    item.official_sources, key=lambda value: value.source_id
                )
            ],
        }
        for item in sorted(papers, key=lambda value: value.paper.paper_id)
    ]


def _persist_resolver_snapshot(
    database: Database,
    artifact_store: ArtifactStore,
    snapshot: ResolverSnapshot,
) -> str:
    payload = canonical_json(snapshot.to_dict())
    stored = artifact_store.put_bytes(
        payload,
        mime_type="application/json",
        metadata={"kind": "stage3_resolver_snapshot", "schema_version": "1"},
    )
    if stored.artifact_hash != snapshot.snapshot_hash:
        raise ValueError("resolver snapshot artifact hash does not match its identity")
    artifact_id = "artifact-" + stored.artifact_hash
    provenance = json.dumps(
        {"kind": "stage3_resolver_snapshot", "schema_version": "1"},
        sort_keys=True,
        separators=(",", ":"),
    )
    expected = (
        artifact_id,
        None,
        "manifest",
        stored.relative_path,
        stored.mime_type,
        stored.size_bytes,
        stored.artifact_hash,
        provenance,
        "available",
    )
    with database.transaction() as connection:
        connection.execute(
            """INSERT OR IGNORE INTO artifacts(
                   artifact_id, paper_id, artifact_kind, relative_path, mime_type,
                   byte_size, sha256, provenance_json, processing_status
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            expected,
        )
        row = connection.execute(
            """SELECT artifact_id, paper_id, artifact_kind, relative_path,
                      mime_type, byte_size, sha256, provenance_json,
                      processing_status
               FROM artifacts WHERE sha256 = ?""",
            (stored.artifact_hash,),
        ).fetchone()
        if row is None or tuple(row) != expected:
            raise ValueError("resolver snapshot conflicts with persisted artifact metadata")
    return stored.artifact_hash


def _authorized_handoff_identity(
    options: AuthorizedSkillHandoffOptions | None,
) -> Mapping[str, object] | None:
    if options is None:
        return None
    return {
        "queue_path": str(options.queue_path.resolve()),
        "output_dir": str(options.output_dir.resolve()),
        "skill_roots": [str(path.resolve()) for path in options.skill_roots],
        "original_zip": (
            str(options.original_zip.resolve()) if options.original_zip else None
        ),
        "audit_manifest": (
            str(options.audit_manifest.resolve()) if options.audit_manifest else None
        ),
    }


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
    controlled = transport
    transport_identity: Mapping[str, Any] | None = None
    if controlled is None:
        controlled = ControlledHTTPTransport(
            contact=contact,
            user_agent=user_agent,
            timeout_seconds=float(timeout_seconds),
            environment={"UNPAYWALL_EMAIL": unpaywall_email},
        )
        transport_identity = {
            "implementation_version": (
                CONTROLLED_HTTP_TRANSPORT_IMPLEMENTATION_VERSION
            )
        }
    return Stage3MetadataLookup(
        controlled,
        retrieved_at=clock,
        registry=default_metadata_lookup_registry(),
        transport_identity=transport_identity,
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


def select_stage3_paper_ids(
    database: Database,
    *,
    paper_ids: Sequence[str] | None,
    filter_run_id: str | None,
    include_needs_review: bool,
) -> tuple[str, ...]:
    explicit = tuple(sorted(set(paper_ids or ())))
    if explicit:
        if filter_run_id is not None:
            raise ValueError("paper_ids and filter_run_id are mutually exclusive")
        return explicit
    if filter_run_id is None:
        if paper_ids is None:
            raise ValueError("paper_ids or filter_run_id is required for Stage 3")
        return explicit
    statuses = (FilterStatus.RELEVANT.value,)
    if include_needs_review:
        statuses += (FilterStatus.NEEDS_REVIEW.value,)
    placeholders = ", ".join("?" for _ in statuses)
    run = database.connection.execute(
        "SELECT stage, status FROM pipeline_runs WHERE run_id = ?",
        (filter_run_id,),
    ).fetchone()
    if run is None or tuple(run) != ("stage-2", "complete"):
        raise ValueError("filter_run_id must name a complete Stage 2 run")
    rows = database.connection.execute(
        f"""SELECT paper_id FROM filter_decisions
            WHERE run_id = ? AND status IN ({placeholders}) ORDER BY paper_id""",
        (filter_run_id, *statuses),
    ).fetchall()
    return tuple(str(row["paper_id"]) for row in rows)


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
    papers: Sequence[Stage3Paper],
    public_result: Stage3RunResult,
    *,
    service: DownloadService,
    authorization_grant_id: str,
    purpose: str,
    now: str,
    skill_digest: str,
    dependency_digest: str,
    authorization_scope: DownloadScopeBinding,
) -> _AuthorizedQueuePlan:
    """Freeze only unresolved, publisher-bound, exactly authorized candidates."""

    items: list[SkillQueueItem] = []
    candidate_ids: list[str] = []
    candidate_urls: set[str] = set()
    browser_urls: set[str] = set()
    queue_dois: set[str] = set()
    reserved_paper_ids: set[str] = set()
    results = {item.paper_id: item for item in public_result.papers}
    for item in papers:
        result = results.get(item.paper.paper_id)
        if (
            result is None
            or result.status is DownloadStatus.DOWNLOADED
            or not result.authorized_candidate_ids
        ):
            continue
        doi = item.paper.doi
        if doi is None or not doi.startswith("10.") or "/" not in doi:
            continue
        for candidate_id in result.authorized_candidate_ids:
            try:
                candidate = service.load_candidate(candidate_id)
            except FetchRejected:
                continue
            if (
                candidate.url in candidate_urls
                or not authorized_publisher_host_matches(doi, candidate.host)
            ):
                continue
            try:
                browser_url = _authorized_browser_queue_url(candidate, doi)
            except AuthorizedSkillAdapterError:
                continue
            if browser_url in browser_urls:
                continue
            if doi.lower() in queue_dois:
                raise AuthorizedSkillAdapterError(
                    "authorized queue requires unique DOIs"
                )
            try:
                service.require_authorized_handoff(
                    authorization_grant_id,
                    candidate,
                    purpose=purpose,
                    provider="authorized_skill",
                    mode=_AUTHORIZED_SKILL_EXECUTION_MODE,
                    now=now,
                    skill_digest=skill_digest,
                    dependency_digest=dependency_digest,
                    reserved_paper_ids=reserved_paper_ids,
                    collection_id=authorization_scope.collection_id,
                    collection_snapshot_hash=authorization_scope.collection_snapshot_hash,
                    selection_snapshot_hash=authorization_scope.selection_snapshot_hash,
                )
            except (FetchRejected, GrantError):
                continue
            items.append(
                SkillQueueItem(
                    item.paper.paper_id,
                    doi,
                    browser_url,
                    item.paper.title,
                    candidate.url,
                )
            )
            candidate_urls.add(candidate.url)
            browser_urls.add(browser_url)
            queue_dois.add(doi.lower())
            reserved_paper_ids.add(item.paper.paper_id)
            candidate_ids.append(candidate.candidate_id)
            break
    ordered = tuple(sorted(
        zip(items, candidate_ids, strict=True),
        key=lambda value: value[0].paper_id,
    ))
    return _AuthorizedQueuePlan(
        tuple(item for item, _candidate_id in ordered),
        tuple(candidate_id for _item, candidate_id in ordered),
        authorization_scope,
    )


def _validate_frozen_queue(
    papers: Sequence[Stage3Paper],
    items: Sequence[SkillQueueItem],
    *,
    service: DownloadService,
    run_id: str,
    queue_path: Path,
    authorization_grant_id: str,
    purpose: str,
    now: str,
    skill_digest: str,
    dependency_digest: str,
    authorization_scope: DownloadScopeBinding,
) -> _AuthorizedQueuePlan:
    """Reprove every immutable row before a resumed browser handoff is exposed."""

    selected = {item.paper.paper_id: item.paper for item in papers}
    reserved_paper_ids: set[str] = set()
    expected: list[SkillQueueItem] = []
    candidate_ids: list[str] = []
    resolved_queue_path = str(queue_path.absolute())
    reservations = _authorized_reservation_map(
        service,
        items,
        run_id=run_id,
        queue_path=resolved_queue_path,
        authorization_grant_id=authorization_grant_id,
        authorization_scope=authorization_scope,
    )
    for queue_index, item in enumerate(items, 1):
        paper = selected.get(item.paper_id)
        if (
            paper is None
            or paper.doi is None
            or paper.doi.strip().lower() != item.doi.lower()
            or paper.title != item.title
        ):
            raise AuthorizedSkillAdapterError(
                "authorized queue does not match the selected papers"
            )
        reservation = reservations[item.paper_id]
        try:
            candidate = service.load_candidate(str(reservation["candidate_id"]))
        except FetchRejected as error:
            raise AuthorizedSkillAdapterError(
                "authorized queue row has no matching durable reservation"
            ) from error
        stored_item_hash = str(reservation["queue_item_hash"])
        if candidate.paper_id != item.paper_id:
            raise AuthorizedSkillAdapterError(
                "authorized queue candidate differs from its durable reservation"
            )
        try:
            browser_url = _authorized_browser_queue_url(candidate, item.doi)
        except AuthorizedSkillAdapterError as error:
            raise AuthorizedSkillAdapterError(
                "authorized queue landing URL differs from its durable reservation"
            ) from error
        if browser_url != item.url:
            raise AuthorizedSkillAdapterError(
                "authorized queue landing URL differs from its durable reservation"
            )
        expected_item = SkillQueueItem(
            item.paper_id, item.doi, item.url, item.title, candidate.url
        )
        if _queue_item_hash(
            expected_item,
            authorization_scope,
            candidate=candidate,
            queue_index=queue_index,
        ) != stored_item_hash:
            raise AuthorizedSkillAdapterError(
                "authorized queue row differs from its durable reservation"
            )
        service.require_authorized_handoff(
            authorization_grant_id,
            candidate,
            purpose=purpose,
            provider="authorized_skill",
            mode=_AUTHORIZED_SKILL_EXECUTION_MODE,
            now=now,
            skill_digest=skill_digest,
            dependency_digest=dependency_digest,
            reserved_paper_ids=reserved_paper_ids,
            collection_id=authorization_scope.collection_id,
            collection_snapshot_hash=authorization_scope.collection_snapshot_hash,
            selection_snapshot_hash=authorization_scope.selection_snapshot_hash,
        )
        reserved_paper_ids.add(item.paper_id)
        expected.append(expected_item)
        candidate_ids.append(candidate.candidate_id)
    return _AuthorizedQueuePlan(
        tuple(expected), tuple(candidate_ids), authorization_scope
    )


def _reserve_queue_plan(
    service: DownloadService,
    plan: _AuthorizedQueuePlan,
    *,
    run_id: str,
    queue_path: Path,
    authorization_grant_id: str,
    purpose: str,
    now: str,
    skill_digest: str,
    dependency_digest: str,
) -> None:
    resolved_queue_path = str(queue_path.absolute())
    reserved_paper_ids: set[str] = set()
    with service.database.transaction():
        for queue_index, (item, candidate_id) in enumerate(
            zip(plan.items, plan.candidate_ids, strict=True), 1
        ):
            candidate = service.load_candidate(candidate_id)
            if (
                candidate.paper_id != item.paper_id
                or item.candidate_url != candidate.url
                or _authorized_browser_queue_url(candidate, item.doi) != item.url
            ):
                raise AuthorizedSkillAdapterError(
                    "authorized queue item differs from its persisted candidate"
                )
            service.reserve_authorized_handoff(
                authorization_grant_id,
                candidate,
                run_id=run_id,
                queue_path=resolved_queue_path,
                queue_item_hash=_queue_item_hash(
                    item,
                    plan.authorization_scope,
                    candidate=candidate,
                    queue_index=queue_index,
                ),
                purpose=purpose,
                provider="authorized_skill",
                mode=_AUTHORIZED_SKILL_EXECUTION_MODE,
                now=now,
                skill_digest=skill_digest,
                dependency_digest=dependency_digest,
                reserved_paper_ids=reserved_paper_ids,
                collection_id=plan.authorization_scope.collection_id,
                collection_snapshot_hash=plan.authorization_scope.collection_snapshot_hash,
                selection_snapshot_hash=plan.authorization_scope.selection_snapshot_hash,
            )
            reserved_paper_ids.add(item.paper_id)
        reservations = _authorized_reservation_map(
            service,
            plan.items,
            run_id=run_id,
            queue_path=resolved_queue_path,
            authorization_grant_id=authorization_grant_id,
            authorization_scope=plan.authorization_scope,
        )
        for queue_index, (item, candidate_id) in enumerate(
            zip(plan.items, plan.candidate_ids, strict=True), 1
        ):
            candidate = service.load_candidate(candidate_id)
            reservation = reservations[item.paper_id]
            if (
                reservation["candidate_id"] != candidate_id
                or reservation["queue_item_hash"]
                != _queue_item_hash(
                    item,
                    plan.authorization_scope,
                    candidate=candidate,
                    queue_index=queue_index,
                )
            ):
                raise AuthorizedSkillAdapterError(
                    "authorized queue reservations differ from the complete plan"
                )


def _queue_item_hash(
    item: SkillQueueItem,
    authorization_scope: DownloadScopeBinding,
    *,
    candidate: AccessLocationCandidate,
    queue_index: int,
) -> str:
    if (
        not item.candidate_url
        or candidate.candidate_id == ""
        or candidate.paper_id != item.paper_id
        or candidate.url != item.candidate_url
        or queue_index < 1
    ):
        raise AuthorizedSkillAdapterError(
            "authorized queue item is not bound to its complete candidate"
        )
    return content_hash({
        "schema_version": "2",
        "queue_index": queue_index,
        "paper_id": item.paper_id,
        "doi": item.doi,
        "landing_url": item.url,
        "title": item.title,
        "candidate_id": candidate.candidate_id,
        "candidate_url": item.candidate_url,
        "candidate_sha256": content_hash(candidate.to_dict()),
        "authorization_scope": authorization_scope.to_dict(),
    })


def _authorized_reservation_map(
    service: DownloadService,
    items: Sequence[SkillQueueItem],
    *,
    run_id: str,
    queue_path: str,
    authorization_grant_id: str,
    authorization_scope: DownloadScopeBinding,
) -> dict[str, Mapping[str, str | None]]:
    rows = service.list_authorized_handoff_reservations(run_id=run_id)
    expected_paper_ids = {item.paper_id for item in items}
    if (
        len(rows) != len(items)
        or len({str(row["paper_id"]) for row in rows}) != len(rows)
        or {str(row["paper_id"]) for row in rows} != expected_paper_ids
        or any(
            row["authorization_grant_id"] != authorization_grant_id
            or row["run_id"] != run_id
            or row["queue_path"] != queue_path
            or row["collection_id"] != authorization_scope.collection_id
            or row["collection_snapshot_hash"]
            != authorization_scope.collection_snapshot_hash
            or row["selection_snapshot_hash"]
            != authorization_scope.selection_snapshot_hash
            for row in rows
        )
    ):
        raise AuthorizedSkillAdapterError(
            "authorized queue reservations do not exactly match the CSV"
        )
    return {str(row["paper_id"]): row for row in rows}


def _authorized_browser_queue_url(
    candidate: AccessLocationCandidate, doi: str
) -> str:
    """Choose the audited browser landing URL without weakening candidate binding."""

    if not candidate.landing_url:
        raise AuthorizedSkillAdapterError(
            "authorized queue requires an explicit publisher landing URL"
        )
    candidate_parts = urlsplit(candidate.url)
    landing_parts = urlsplit(candidate.landing_url)
    try:
        same_endpoint = (
            candidate_parts.scheme == "https"
            and landing_parts.scheme == "https"
            and candidate_parts.hostname is not None
            and landing_parts.hostname is not None
            and candidate_parts.username is None
            and candidate_parts.password is None
            and landing_parts.username is None
            and landing_parts.password is None
            and candidate.host is not None
            and candidate.host.lower().rstrip(".")
            == candidate_parts.hostname.lower().rstrip(".")
            and candidate_parts.hostname.lower().rstrip(".")
            == landing_parts.hostname.lower().rstrip(".")
            and (candidate_parts.port or 443) == (landing_parts.port or 443)
            and (candidate_parts.port or 443) == 443
            and authorized_publisher_host_matches(doi, candidate.host)
            and authorized_publisher_host_matches(doi, candidate_parts.hostname)
            and authorized_publisher_host_matches(doi, landing_parts.hostname)
            and _normalized_url_identity(candidate_parts)
            != _normalized_url_identity(landing_parts)
            and not _looks_like_pdf_landing(landing_parts)
        )
    except ValueError as error:
        raise AuthorizedSkillAdapterError(
            "authorized queue landing URL is invalid"
        ) from error
    if not same_endpoint:
        raise AuthorizedSkillAdapterError(
            "authorized queue landing URL must match the candidate domain"
        )
    return candidate.landing_url


def _normalized_url_identity(value: Any) -> tuple[str, str, int, str, str]:
    return (
        value.scheme.lower(),
        (value.hostname or "").lower().rstrip("."),
        value.port or 443,
        unquote(value.path or "/"),
        value.query,
    )


def _looks_like_pdf_landing(value: Any) -> bool:
    path = unquote(value.path or "").lower()
    return path.endswith(".pdf") or bool(
        re.search(r"/doi/(?:pdf|epdf)(?:/|$)", path)
    )


def _has_authorized_attempt(
    database: Database,
    *,
    run_id: str,
    authorization_grant_id: str,
    paper_id: str,
    url: str,
) -> bool:
    row = database.connection.execute(
        """SELECT 1 FROM download_attempts AS attempt
           JOIN download_candidates AS candidate
             ON candidate.candidate_id = attempt.candidate_id
           WHERE attempt.run_id = ?
             AND attempt.authorization_grant_id = ?
             AND attempt.provider = 'authorized_skill'
             AND candidate.paper_id = ?
             AND candidate.url = ?
           LIMIT 1""",
        (run_id, authorization_grant_id, paper_id, url),
    ).fetchone()
    return row is not None


def _resume_checkpoints(
    database: Database, run_id: str
) -> dict[str, Stage3PaperResult]:
    """Resume immutable downloads and terminal no-PDF outcomes."""
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
    terminal = database.connection.execute(
        """SELECT paper_id, status, reason_code FROM stage3_paper_results
           WHERE run_id = ? AND status IN ('not_available', 'failed_terminal')
           ORDER BY paper_id""",
        (run_id,),
    ).fetchall()
    for row in terminal:
        paper_id = str(row["paper_id"])
        completed.setdefault(
            paper_id,
            Stage3PaperResult(
                paper_id,
                DownloadStatus(str(row["status"])),
                str(row["reason_code"]),
                (),
            ),
        )
    return completed


def _save_checkpoint(
    database: Database,
    run_id: str,
    result: Stage3PaperResult,
    updated_at: str,
) -> None:
    with database.transaction() as connection:
        connection.execute(
            """INSERT INTO stage3_paper_results(
                   run_id, paper_id, status, reason_code, updated_at
               ) VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(run_id, paper_id) DO UPDATE SET
                   status = excluded.status,
                   reason_code = excluded.reason_code,
                   updated_at = excluded.updated_at
               WHERE stage3_paper_results.status NOT IN (
                   'downloaded', 'not_available', 'failed_terminal'
               )""",
            (
                run_id,
                result.paper_id,
                result.status.value,
                result.reason_code,
                updated_at,
            ),
        )


def _stage3_result_status(result: Stage3RunResult) -> str:
    statuses = {item.status for item in result.papers}
    if statuses <= _TERMINAL_DOWNLOAD_STATUSES:
        return "complete"
    if statuses & {DownloadStatus.AUTH_REQUIRED, DownloadStatus.MANUAL_REQUIRED}:
        return "manual_required"
    return "incomplete"


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
