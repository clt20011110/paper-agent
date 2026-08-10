"""Bounded, injectable coordinator for Stage 3 access and download work.

The coordinator owns ordering and fail-closed control flow.  Resolvers,
providers, grants, and the optional planner are injected boundaries, making
this module safe to exercise without a browser, network, or model runtime.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from .authorized_skill_runtime import (
    AuthorizedSkillDoctorResult,
    AuthorizedSkillRuntime,
    AuthorizedSkillRuntimeError,
)
from .codex_exec import CodexExecError
from .downloads import AuthorizationContext, FetchRejected
from .domain import (
    AccessLocationCandidate,
    DownloadResult,
    DownloadStatus,
    FetchDecisionStatus,
    FetchRequest,
    Paper,
    PaperSource,
)
from .grants import GrantError
from .download_providers import (
    DEFAULT_RESOLVER_ORDER,
    DownloadProviderError,
    DownloadProviderRegistry,
    FetchContext,
    MetadataResolverTransport,
    ProbeContext,
    ResolverContext,
    ResolverRegistry,
)


class Stage3GrantAuthorizer(Protocol):
    """The narrow subset of ``GrantStore`` needed before a skill dispatch."""

    def load(self, grant_id: str, **kwargs: object) -> object: ...

    def require_active(self, grant_id: str, **kwargs: object) -> object: ...


class Stage3ManualQueue(Protocol):
    """Persist one deduplicated manual item after the provider chain is exhausted."""

    def enqueue_manual(
        self,
        queue_type: str,
        dedup_key: str,
        paper_id: str | None,
        reason: Mapping[str, object],
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class Stage3Paper:
    paper: Paper
    official_sources: tuple[PaperSource, ...] = ()
    lookup: MetadataResolverTransport | None = None
    matched_arxiv: bool = False
    include_arxiv_candidates: bool = False


@dataclass(frozen=True, slots=True)
class LunaPlannerInput:
    """The only data an optional Stage 3 planner may receive.

    This deliberately has no URL, page/body, account, cookie, or token field.
    """

    candidate_id: str
    paper_id: str
    host: str | None
    status: str
    reason_code: str


class LunaPlannerResult(Protocol):
    selected: bool
    status: str
    page_state: str
    next_action: str
    reason_code: str
    invocation_metadata: Mapping[str, Any]
    planner_decision_id: str | None


LunaPlanner = Callable[[LunaPlannerInput], bool | LunaPlannerResult]


@dataclass(frozen=True, slots=True)
class AuthorizedSkillOptions:
    """Explicit opt-in configuration for an audited skill provider."""

    enabled: bool = False
    runtime: AuthorizedSkillRuntime | None = None
    grant_store: Stage3GrantAuthorizer | None = None
    authorization_grant_id: str | None = None
    mode: str = "attended"
    collection_id: str | None = None
    collection_snapshot_hash: str | None = None
    selection_snapshot_hash: str | None = None
    planner: LunaPlanner | None = None


@dataclass(frozen=True, slots=True)
class Stage3Attempt:
    candidate_id: str | None
    provider: str
    status: str
    reason_code: str
    download_status: DownloadStatus | None = None
    fetch_request: FetchRequest | None = field(default=None, repr=False)
    planner_metadata: Mapping[str, Any] | None = field(default=None, repr=False)
    planner_decision_id: str | None = None


@dataclass(frozen=True, slots=True)
class Stage3PaperResult:
    paper_id: str
    status: DownloadStatus
    reason_code: str
    attempts: tuple[Stage3Attempt, ...]
    download: DownloadResult | None = None
    resumed: bool = False


@dataclass(frozen=True, slots=True)
class Stage3RunResult:
    papers: tuple[Stage3PaperResult, ...]

    def for_paper(self, paper_id: str) -> Stage3PaperResult:
        for result in self.papers:
            if result.paper_id == paper_id:
                return result
        raise KeyError(paper_id)


@dataclass(slots=True)
class Stage3Pipeline:
    """Resolve in frozen order and try only persisted allow requests.

    A caller may pass prior results to :meth:`run` to resume individual papers;
    this intentionally carries no hidden database or network dependency.
    """

    resolvers: ResolverRegistry
    providers: DownloadProviderRegistry
    purpose: str
    now: str
    run_id: str
    manual_queue: Stage3ManualQueue
    resolver_order: tuple[str, ...] = DEFAULT_RESOLVER_ORDER
    authorized: AuthorizedSkillOptions = AuthorizedSkillOptions()
    public_authorization_grant_id: str | None = None

    def __post_init__(self) -> None:
        if not self.purpose or not self.now or not self.run_id:
            raise ValueError("Stage 3 purpose, now, and run_id are required")
        if tuple(self.resolvers.names) != self.resolver_order:
            raise ValueError("resolver registry must match the frozen resolver order")

    def run(
        self,
        papers: Sequence[Stage3Paper],
        *,
        completed: Mapping[str, Stage3PaperResult] | None = None,
        checkpoint: Callable[[Stage3PaperResult], None] | None = None,
    ) -> Stage3RunResult:
        """Process paper inputs in caller order, skipping explicit checkpoints."""

        if len({item.paper.paper_id for item in papers}) != len(papers):
            raise ValueError("a Stage 3 call cannot contain duplicate paper_ids")
        checkpoints = completed or {}
        skill_state = self._prepare_skill() if self.authorized.enabled else None
        output: list[Stage3PaperResult] = []
        for item in papers:
            existing = checkpoints.get(item.paper.paper_id)
            if existing is not None:
                output.append(replace(existing, resumed=True))
                continue
            result = self._run_paper(item, skill_state)
            if checkpoint is not None:
                checkpoint(result)
            output.append(result)
        return Stage3RunResult(tuple(output))

    def _run_paper(
        self, item: Stage3Paper, skill_state: _SkillState | None
    ) -> Stage3PaperResult:
        candidates = self._resolve(item)
        attempts: list[Stage3Attempt] = []
        authorized_candidates: list[tuple[AccessLocationCandidate, Stage3Attempt]] = []
        fetch_results: list[DownloadResult] = []
        manual_required = False
        denied_candidates = 0
        for candidate in candidates:
            public = self._probe(candidate, provider=None, context=self._public_context())
            attempts.append(public)
            if public.status == FetchDecisionStatus.DENY.value:
                denied_candidates += 1
                continue
            result = self._fetch_if_allowed(candidate, public, self._public_fetch_context())
            if result is not None:
                fetch_results.append(result)
                fetch_attempt = _download_attempt(candidate, public.provider, result)
                attempts.append(fetch_attempt)
                if result.status is DownloadStatus.DOWNLOADED:
                    return Stage3PaperResult(item.paper.paper_id, result.status, "downloaded", tuple(attempts), result)
                if result.status is DownloadStatus.AUTH_REQUIRED:
                    authorized_candidates.append((candidate, fetch_attempt))
                elif result.status is DownloadStatus.MANUAL_REQUIRED:
                    manual_required = True
            elif public.status in {
                FetchDecisionStatus.NEEDS_GRANT.value,
                FetchDecisionStatus.MANUAL.value,
            }:
                authorized_candidates.append((candidate, public))

        # Authorized browser work is a fallback provider stage.  Do not enter
        # it until every public/OA candidate has been tried in frozen order.
        if self.authorized.enabled:
            for candidate, public in authorized_candidates:
                assert skill_state is not None
                skill_attempt = self._authorized_attempt(candidate, public, skill_state)
                attempts.append(skill_attempt)
                if skill_attempt.status != FetchDecisionStatus.ALLOW.value:
                    manual_required = True
                    continue
                skill_result = self._fetch_if_allowed(
                    candidate, skill_attempt,
                    self._skill_fetch_context(skill_state, skill_attempt.planner_decision_id),
                )
                if skill_result is not None:
                    fetch_results.append(skill_result)
                    attempts.append(_download_attempt(candidate, "authorized_skill", skill_result))
                    if skill_result.status is DownloadStatus.DOWNLOADED:
                        return Stage3PaperResult(item.paper.paper_id, skill_result.status, "downloaded", tuple(attempts), skill_result)
                    if skill_result.status in {
                        DownloadStatus.AUTH_REQUIRED,
                        DownloadStatus.MANUAL_REQUIRED,
                    }:
                        manual_required = True
                else:
                    manual_required = True
        elif authorized_candidates:
            manual_required = True

        if candidates and denied_candidates == len(candidates):
            return Stage3PaperResult(
                item.paper.paper_id,
                DownloadStatus.FAILED_TERMINAL,
                "all_access_locations_denied",
                tuple(attempts),
            )
        if not manual_required and fetch_results:
            result = _final_fetch_result(fetch_results)
            if (
                denied_candidates
                and result.status is DownloadStatus.NOT_AVAILABLE
            ):
                return Stage3PaperResult(
                    item.paper.paper_id,
                    DownloadStatus.FAILED_TERMINAL,
                    "access_location_denied",
                    tuple(attempts),
                )
            return Stage3PaperResult(
                item.paper.paper_id,
                result.status,
                result.error_code or result.status.value,
                tuple(attempts),
                result,
            )
        reason = "no_access_location_candidates" if not candidates else "manual_queue_required"
        attempts.append(Stage3Attempt(None, "manual", FetchDecisionStatus.MANUAL.value, reason))
        self.manual_queue.enqueue_manual(
            "download",
            f"{self.run_id}:{item.paper.paper_id}",
            item.paper.paper_id,
            {
                "run_id": self.run_id,
                "reason_code": reason,
                "attempts": [
                    {
                        "candidate_id": attempt.candidate_id,
                        "provider": attempt.provider,
                        "status": attempt.status,
                        "reason_code": attempt.reason_code,
                    }
                    for attempt in attempts
                ],
            },
        )
        return Stage3PaperResult(
            item.paper.paper_id, DownloadStatus.MANUAL_REQUIRED, reason, tuple(attempts)
        )

    def _resolve(self, item: Stage3Paper) -> tuple[AccessLocationCandidate, ...]:
        context = ResolverContext(
            paper=item.paper,
            official_sources=item.official_sources,
            lookup=item.lookup,
            matched_arxiv=item.matched_arxiv,
            include_arxiv_candidates=item.include_arxiv_candidates,
            retrieved_at=self.now,
        )
        resolved = self.resolvers.resolve(context)
        index = {name: number for number, name in enumerate(self.resolver_order)}
        return tuple(candidate for _, candidate in sorted(
            ((position, candidate) for position, candidate in enumerate(resolved)
             if candidate.paper_id == item.paper.paper_id),
            key=lambda pair: (index[pair[1].resolver], pair[0]),
        ))

    def _public_context(self) -> ProbeContext:
        return ProbeContext(
            self.purpose,
            self.now,
            authorization_grant_id=self.public_authorization_grant_id,
            run_id=self.run_id,
        )

    def _public_fetch_context(self) -> FetchContext:
        return FetchContext(self.run_id, self.now)

    def _probe(self, candidate: AccessLocationCandidate, *, provider: str | None, context: ProbeContext) -> Stage3Attempt:
        try:
            attempt = self.providers.probe_with(provider, candidate, context) if provider else self.providers.probe(candidate, context)
        except (DownloadProviderError, OSError, ValueError) as error:
            return Stage3Attempt(candidate.candidate_id, provider or "public", FetchDecisionStatus.MANUAL.value, _reason(error))
        return Stage3Attempt(
            candidate.candidate_id, attempt.provider, attempt.decision.status.value,
            attempt.decision.reason_code, fetch_request=attempt.decision.fetch_request,
        )

    def _fetch_if_allowed(
        self, candidate: AccessLocationCandidate, attempt: Stage3Attempt, context: FetchContext
    ) -> DownloadResult | None:
        # Never synthesize a request or re-probe.  A provider sees only the
        # immutable FetchRequest attached to this exact allow decision.
        if attempt.status != FetchDecisionStatus.ALLOW.value or attempt.fetch_request is None:
            return None
        try:
            return self.providers.fetch(attempt.fetch_request, context)
        except FetchRejected:
            return DownloadResult(
                attempt.fetch_request.request_id, candidate.paper_id, DownloadStatus.MANUAL_REQUIRED,
                attempt.provider, error_code="fetch_request_rejected",
            )
        except (DownloadProviderError, OSError, ValueError):
            return DownloadResult(
                attempt.fetch_request.request_id, candidate.paper_id, DownloadStatus.FAILED_RETRYABLE,
                attempt.provider, error_code="provider_fetch_failed",
            )

    def _prepare_skill(self) -> _SkillState:
        options = self.authorized
        if options.runtime is None:
            return _SkillState(None, "authorized_skill_not_configured")
        try:
            ready = options.runtime.require_ready()
        except (AuthorizedSkillRuntimeError, OSError, ValueError) as error:
            return _SkillState(None, "authorized_skill_unavailable:" + _reason(error))
        return _SkillState(ready, None)

    def _authorized_attempt(
        self, candidate: AccessLocationCandidate, public: Stage3Attempt, state: _SkillState
    ) -> Stage3Attempt:
        if state.reason:
            return Stage3Attempt(candidate.candidate_id, "authorized_skill", FetchDecisionStatus.MANUAL.value, state.reason)
        assert state.ready is not None
        options = self.authorized
        if options.grant_store is None or not options.authorization_grant_id:
            return Stage3Attempt(candidate.candidate_id, "authorized_skill", FetchDecisionStatus.MANUAL.value, "authorized_grant_not_configured")
        try:
            loaded = options.grant_store.load(
                options.authorization_grant_id, kind="download", now=self.now
            )
            document = loaded.document  # type: ignore[attr-defined]
            scope = document["scope"]
            if "store" not in document["actions"]:
                raise GrantError("authorized skill grant must allow store")
            if document["skill_digest"] != state.ready.installed_content_sha256:
                raise GrantError("authorized skill digest does not match the grant")
            if document["dependency_digest"] != state.ready.dependency_lock_sha256:
                raise GrantError("authorized skill dependency digest does not match the grant")
            if scope["provider"] != "authorized_skill":
                raise GrantError("authorized skill grant must bind its provider")
            options.grant_store.require_active(
                options.authorization_grant_id, kind="download", action="download", purpose=self.purpose,
                mode=options.mode, now=self.now,
                paper_id=candidate.paper_id if scope["paper_ids"] else None,
                domain=candidate.host if scope["domains"] else None,
                provider="authorized_skill",
                collection_id=options.collection_id if scope["collection_ids"] else None,
                collection_snapshot_hash=(
                    options.collection_snapshot_hash if scope["collection_snapshot_hash"] else None
                ),
                selection_snapshot_hash=(
                    options.selection_snapshot_hash if scope["selection_snapshot_hash"] else None
                ),
                data_category="full_text" if scope["data_categories"] else None,
                skill_digest=state.ready.installed_content_sha256,
                dependency_digest=state.ready.dependency_lock_sha256,
            )
        except (GrantError, OSError, ValueError) as error:
            return Stage3Attempt(candidate.candidate_id, "authorized_skill", FetchDecisionStatus.MANUAL.value, "authorized_grant_invalid:" + _reason(error))
        control = LunaPlannerInput(candidate.candidate_id, candidate.paper_id, candidate.host, public.status, public.reason_code)
        planner_metadata = None
        planner_decision_id = None
        if options.planner is not None:
            try:
                planned = options.planner(control)
                if isinstance(planned, bool):
                    selected, planner_reason = planned, "authorized_skill_not_selected"
                else:
                    selected = planned.selected
                    planner_reason = planned.reason_code
                    planner_metadata = planned.invocation_metadata
                    planner_decision_id = planned.planner_decision_id
                if not selected:
                    return Stage3Attempt(
                        candidate.candidate_id, "authorized_skill", FetchDecisionStatus.MANUAL.value,
                        planner_reason, planner_metadata=planner_metadata,
                        planner_decision_id=planner_decision_id,
                    )
            except (CodexExecError, OSError, ValueError) as error:
                return Stage3Attempt(candidate.candidate_id, "authorized_skill", FetchDecisionStatus.MANUAL.value, "authorized_planner_failed:" + _reason(error))
        context = ProbeContext(
            self.purpose, self.now, options.authorization_grant_id, options.mode,
            state.ready.installed_content_sha256, state.ready.dependency_lock_sha256,
            options.collection_id, options.collection_snapshot_hash, options.selection_snapshot_hash, self.run_id,
        )
        return replace(
            self._probe(candidate, provider="authorized_skill", context=context),
            planner_metadata=planner_metadata,
            planner_decision_id=planner_decision_id,
        )

    def _skill_fetch_context(
        self, state: _SkillState, planner_decision_id: str | None,
    ) -> FetchContext:
        if state.ready is None:
            return self._public_fetch_context()
        options = self.authorized
        return FetchContext(self.run_id, self.now, AuthorizationContext(
            options.mode, state.ready.installed_content_sha256, state.ready.dependency_lock_sha256,
            options.collection_id, options.collection_snapshot_hash, options.selection_snapshot_hash,
            planner_decision_id,
        ))


@dataclass(frozen=True, slots=True)
class _SkillState:
    ready: AuthorizedSkillDoctorResult | None
    reason: str | None


def _download_attempt(candidate: AccessLocationCandidate, provider: str, result: DownloadResult) -> Stage3Attempt:
    return Stage3Attempt(candidate.candidate_id, provider, "fetch", result.error_code or result.status.value, result.status)


def _final_fetch_result(results: Sequence[DownloadResult]) -> DownloadResult:
    priority = (
        DownloadStatus.FAILED_RETRYABLE,
        DownloadStatus.FAILED_TERMINAL,
        DownloadStatus.NOT_AVAILABLE,
        DownloadStatus.AUTH_REQUIRED,
        DownloadStatus.MANUAL_REQUIRED,
        DownloadStatus.PENDING,
    )
    return next(result for status in priority for result in results if result.status is status)


def _reason(error: Exception) -> str:
    return type(error).__name__.lower()
