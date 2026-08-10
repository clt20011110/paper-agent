from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from paper_agent.authorized_luna import AuthorizedLunaDecision
from paper_agent.authorized_skill_runtime import AuthorizedSkillRuntimeError
from paper_agent.domain import (
    AccessLocationCandidate,
    DownloadResult,
    DownloadStatus,
    FetchDecision,
    FetchDecisionStatus,
    FetchRequest,
    Paper,
)
from paper_agent.download_providers import (
    DownloadProviderDescriptor,
    DownloadProviderRegistry,
    FetchContext,
    ProbeContext,
    ResolverDescriptor,
    ResolverRegistry,
    RoutedDownloadProvider,
    provider_contract,
)
from paper_agent.downloads import FetchRejected
from paper_agent.stage3_pipeline import (
    AuthorizedSkillOptions,
    Stage3Paper,
    Stage3Pipeline,
)


NOW = "2026-08-10T00:00:00Z"


def candidate(identifier: str, resolver: str, paper_id: str = "paper-1") -> AccessLocationCandidate:
    return AccessLocationCandidate(identifier, paper_id, resolver, f"https://{identifier}.example/paper.pdf", host=f"{identifier}.example")


def request(identifier: str, provider: str) -> FetchRequest:
    return FetchRequest(f"request-{identifier}", identifier, "policy-v1", "research", provider, NOW, "2026-08-11T00:00:00Z", identifier)


def descriptor(name: str, provider: RoutedDownloadProvider, handles) -> DownloadProviderDescriptor:
    return DownloadProviderDescriptor(name, provider, handles, provider_contract())


@dataclass
class Resolver:
    name: str
    values: tuple[AccessLocationCandidate, ...]

    def resolve(self, _context):
        return self.values


@dataclass
class Provider:
    name: str
    decisions: dict[str, FetchDecision]
    probes: list[tuple[str, str]]
    fetches: list[FetchRequest]
    fetch_error: Exception | None = None

    def probe(self, value, _context: ProbeContext) -> FetchDecision:
        self.probes.append((self.name, value.candidate_id))
        return self.decisions[value.candidate_id]

    def fetch(self, value: FetchRequest, _context: FetchContext) -> DownloadResult:
        self.fetches.append(value)
        if self.fetch_error is not None:
            raise self.fetch_error
        return DownloadResult(value.request_id, "paper-1", DownloadStatus.DOWNLOADED, self.name)


@dataclass
class ManualQueue:
    calls: list[tuple[str, str, str | None, dict[str, object]]]

    def enqueue_manual(
        self,
        queue_type: str,
        dedup_key: str,
        paper_id: str | None,
        reason: dict[str, object],
    ) -> None:
        self.calls.append((queue_type, dedup_key, paper_id, reason))


def pipeline(
    *,
    resolvers: ResolverRegistry,
    providers: DownloadProviderRegistry,
    authorized=AuthorizedSkillOptions(),
    manual_queue: ManualQueue | None = None,
) -> Stage3Pipeline:
    return Stage3Pipeline(
        resolvers,
        providers,
        "research",
        NOW,
        "run-1",
        manual_queue or ManualQueue([]),
        resolvers.names,
        authorized,
    )


def test_resolves_in_frozen_order_continues_after_unavailable_and_fetches_only_allow_request() -> None:
    first, second = candidate("first", "first"), candidate("second", "second")
    resolver_registry = ResolverRegistry((
        ResolverDescriptor("first", Resolver("first", (first,))),
        ResolverDescriptor("second", Resolver("second", (second,))),
    ))
    provider = Provider("public_direct", {
        "first": FetchDecision("first", FetchDecisionStatus.MANUAL, "unavailable", "policy-v1"),
        "second": FetchDecision("second", FetchDecisionStatus.ALLOW, "open", "policy-v1", request("second", "public_direct")),
    }, [], [])
    registry = DownloadProviderRegistry((descriptor("public_direct", provider, lambda _value: True),))

    result = pipeline(resolvers=resolver_registry, providers=registry).run((Stage3Paper(Paper("paper-1", "One")),)).for_paper("paper-1")

    assert [item.candidate_id for item in result.attempts[:2]] == ["first", "second"]
    assert provider.probes == [("public_direct", "first"), ("public_direct", "second")]
    assert [item.request_id for item in provider.fetches] == ["request-second"]
    assert result.status is DownloadStatus.DOWNLOADED


def test_explicit_manual_outcome_never_fetches_a_non_allow_probe() -> None:
    value = candidate("only", "only")
    resolver_registry = ResolverRegistry((ResolverDescriptor("only", Resolver("only", (value,))),))
    provider = Provider("public_direct", {
        "only": FetchDecision("only", FetchDecisionStatus.NEEDS_GRANT, "grant_required", "policy-v1"),
    }, [], [])
    registry = DownloadProviderRegistry((descriptor("public_direct", provider, lambda _value: True),))

    manual_queue = ManualQueue([])
    result = pipeline(
        resolvers=resolver_registry, providers=registry, manual_queue=manual_queue
    ).run((Stage3Paper(Paper("paper-1", "One")),)).for_paper("paper-1")

    assert provider.fetches == []
    assert result.status is DownloadStatus.MANUAL_REQUIRED
    assert result.attempts[-1].provider == "manual"
    assert result.attempts[-1].status == "manual"
    assert manual_queue.calls[0][:3] == ("download", "run-1:paper-1", "paper-1")
    assert manual_queue.calls[0][3]["reason_code"] == "manual_queue_required"


def test_policy_deny_is_terminal_and_never_escalates_to_the_authorized_skill() -> None:
    value = candidate("denied", "publisher")
    resolver_registry = ResolverRegistry((
        ResolverDescriptor("publisher", Resolver("publisher", (value,))),
    ))
    public = Provider("public_direct", {
        "denied": FetchDecision(
            "denied", FetchDecisionStatus.DENY, "redistribution_forbidden", "policy-v1"
        ),
    }, [], [])
    skill = Provider("authorized_skill", {
        "denied": FetchDecision(
            "denied", FetchDecisionStatus.ALLOW, "granted", "policy-v1",
            request("denied", "authorized_skill"),
        ),
    }, [], [])
    registry = DownloadProviderRegistry((
        descriptor("public_direct", public, lambda _value: True),
        descriptor("authorized_skill", skill, lambda _value: False),
    ))
    options = AuthorizedSkillOptions(
        enabled=True,
        runtime=Runtime(SimpleNamespace(installed_content_sha256="skill", dependency_lock_sha256="deps")),  # type: ignore[arg-type]
    )

    result = pipeline(
        resolvers=resolver_registry, providers=registry, authorized=options
    ).run((Stage3Paper(Paper("paper-1", "One")),)).for_paper("paper-1")

    assert result.status is DownloadStatus.FAILED_TERMINAL
    assert result.reason_code == "all_access_locations_denied"
    assert skill.probes == []


def test_all_public_locations_are_tried_before_authorized_browser_fallback() -> None:
    private = candidate("private", "publisher")
    public_copy = candidate("public-copy", "arxiv")
    resolver_registry = ResolverRegistry((
        ResolverDescriptor("publisher", Resolver("publisher", (private,))),
        ResolverDescriptor("arxiv", Resolver("arxiv", (public_copy,))),
    ))
    public = Provider(
        "public_direct",
        {
            "private": FetchDecision(
                "private", FetchDecisionStatus.NEEDS_GRANT, "grant_required", "policy-v1"
            ),
            "public-copy": FetchDecision(
                "public-copy", FetchDecisionStatus.ALLOW, "open", "policy-v1",
                request("public-copy", "public_direct"),
            ),
        },
        [],
        [],
    )
    skill = Provider(
        "authorized_skill",
        {
            "private": FetchDecision(
                "private", FetchDecisionStatus.ALLOW, "granted", "policy-v1",
                request("private", "authorized_skill"),
            ),
        },
        [],
        [],
    )
    registry = DownloadProviderRegistry((
        descriptor("public_direct", public, lambda _value: True),
        descriptor("authorized_skill", skill, lambda _value: False),
    ))
    options = AuthorizedSkillOptions(
        enabled=True,
        runtime=Runtime(
            SimpleNamespace(
                installed_content_sha256="skill", dependency_lock_sha256="deps"
            )
        ),  # type: ignore[arg-type]
        grant_store=Grants([]),
        authorization_grant_id="grant-1",
    )

    result = pipeline(
        resolvers=resolver_registry, providers=registry, authorized=options
    ).run((Stage3Paper(Paper("paper-1", "One")),)).for_paper("paper-1")

    assert result.status is DownloadStatus.DOWNLOADED
    assert [value.candidate_id for value in public.fetches] == ["public-copy"]
    assert skill.probes == []


def test_rejected_persisted_request_is_manual_not_blindly_retryable() -> None:
    value = candidate("rejected", "publisher")
    resolver_registry = ResolverRegistry((
        ResolverDescriptor("publisher", Resolver("publisher", (value,))),
    ))
    provider = Provider(
        "public_direct",
        {
            "rejected": FetchDecision(
                "rejected", FetchDecisionStatus.ALLOW, "open", "policy-v1",
                request("rejected", "public_direct"),
            )
        },
        [],
        [],
        FetchRejected("authorization grant is revoked"),
    )
    registry = DownloadProviderRegistry((
        descriptor("public_direct", provider, lambda _value: True),
    ))

    result = pipeline(
        resolvers=resolver_registry, providers=registry
    ).run((Stage3Paper(Paper("paper-1", "One")),)).for_paper("paper-1")

    fetch_attempt = next(item for item in result.attempts if item.status == "fetch")
    assert fetch_attempt.download_status is DownloadStatus.MANUAL_REQUIRED
    assert fetch_attempt.reason_code == "fetch_request_rejected"


@dataclass
class Runtime:
    ready: object
    calls: int = 0

    def require_ready(self):
        self.calls += 1
        if isinstance(self.ready, Exception):
            raise self.ready
        return self.ready


@dataclass
class Grants:
    calls: list[dict]

    def load(self, _grant_id: str, **_kwargs: object) -> object:
        return SimpleNamespace(document={
            "actions": ["download", "store"],
            "skill_digest": "skill",
            "dependency_digest": "deps",
            "scope": {
                "paper_ids": ["paper-1"], "collection_ids": [],
                "collection_snapshot_hash": None, "selection_snapshot_hash": None,
                "domains": ["private.example"], "provider": "authorized_skill",
                "data_categories": ["full_text"],
            },
        })

    def require_active(self, _grant_id: str, **kwargs: object) -> object:
        self.calls.append(dict(kwargs))
        return object()


def test_skill_is_opt_in_and_planner_receives_only_sanitized_control_fields() -> None:
    value = candidate("private", "publisher")
    resolver_registry = ResolverRegistry((ResolverDescriptor("publisher", Resolver("publisher", (value,))),))
    public = Provider("public_direct", {
        "private": FetchDecision("private", FetchDecisionStatus.MANUAL, "not_public", "policy-v1"),
    }, [], [])
    skill = Provider("authorized_skill", {
        "private": FetchDecision("private", FetchDecisionStatus.ALLOW, "granted", "policy-v1", request("private", "authorized_skill")),
    }, [], [])
    registry = DownloadProviderRegistry((
        descriptor("public_direct", public, lambda _value: True),
        descriptor("authorized_skill", skill, lambda _value: False),
    ))
    planner_inputs = []
    grants = Grants([])

    def planner(control):
        planner_inputs.append(control)
        return AuthorizedLunaDecision(
            True,
            "invoke_skill",
            "unknown",
            "invoke_audited_skill",
            "authorized_handoff_selected",
            {},
        )

    options = AuthorizedSkillOptions(
        enabled=True,
        runtime=Runtime(SimpleNamespace(installed_content_sha256="skill", dependency_lock_sha256="deps")),  # type: ignore[arg-type]
        grant_store=grants,
        authorization_grant_id="grant-1",
        planner=planner,
    )

    result = pipeline(resolvers=resolver_registry, providers=registry, authorized=options).run((Stage3Paper(Paper("paper-1", "One")),)).for_paper("paper-1")

    assert result.status is DownloadStatus.DOWNLOADED
    assert skill.probes == [("authorized_skill", "private")]
    assert grants.calls[0]["skill_digest"] == "skill"
    assert grants.calls[0]["dependency_digest"] == "deps"
    control = planner_inputs[0]
    assert control.candidate_id == "private" and control.paper_id == "paper-1" and control.host == "private.example"
    assert set(control.__dataclass_fields__) == {"candidate_id", "paper_id", "host", "status", "reason_code"}


def test_skill_drift_is_manual_and_does_not_block_other_papers_or_resumption() -> None:
    first, second = candidate("first", "publisher", "paper-1"), candidate("second", "publisher", "paper-2")
    resolver_registry = ResolverRegistry((ResolverDescriptor("publisher", Resolver("publisher", (first, second))),))
    public = Provider("public_direct", {
        "first": FetchDecision("first", FetchDecisionStatus.MANUAL, "private", "policy-v1"),
        "second": FetchDecision("second", FetchDecisionStatus.ALLOW, "open", "policy-v1", request("second", "public_direct")),
    }, [], [])
    registry = DownloadProviderRegistry((descriptor("public_direct", public, lambda _value: True),))
    runtime = Runtime(AuthorizedSkillRuntimeError("drift"))
    options = AuthorizedSkillOptions(enabled=True, runtime=runtime)  # type: ignore[arg-type]
    stage = pipeline(resolvers=resolver_registry, providers=registry, authorized=options)

    run = stage.run((Stage3Paper(Paper("paper-1", "One")), Stage3Paper(Paper("paper-2", "Two"))))

    assert run.for_paper("paper-1").status is DownloadStatus.MANUAL_REQUIRED
    assert run.for_paper("paper-2").status is DownloadStatus.DOWNLOADED
    assert runtime.calls == 1
    assert any(item.reason_code.startswith("authorized_skill_unavailable") for item in run.for_paper("paper-1").attempts)
    resumed = stage.run((Stage3Paper(Paper("paper-1", "One")),), completed={"paper-1": run.for_paper("paper-1")})
    assert resumed.for_paper("paper-1").resumed
