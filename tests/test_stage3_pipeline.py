from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

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
)
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

    def probe(self, value, _context: ProbeContext) -> FetchDecision:
        self.probes.append((self.name, value.candidate_id))
        return self.decisions[value.candidate_id]

    def fetch(self, value: FetchRequest, _context: FetchContext) -> DownloadResult:
        self.fetches.append(value)
        return DownloadResult(value.request_id, "paper-1", DownloadStatus.DOWNLOADED, self.name)


def pipeline(*, resolvers: ResolverRegistry, providers: DownloadProviderRegistry, authorized=AuthorizedSkillOptions()) -> Stage3Pipeline:
    return Stage3Pipeline(resolvers, providers, "research", NOW, "run-1", resolvers.names, authorized)


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
    registry = DownloadProviderRegistry((DownloadProviderDescriptor("public_direct", provider, lambda _value: True),))

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
    registry = DownloadProviderRegistry((DownloadProviderDescriptor("public_direct", provider, lambda _value: True),))

    result = pipeline(resolvers=resolver_registry, providers=registry).run((Stage3Paper(Paper("paper-1", "One")),)).for_paper("paper-1")

    assert provider.fetches == []
    assert result.status is DownloadStatus.MANUAL_REQUIRED
    assert result.attempts[-1].provider == "manual"
    assert result.attempts[-1].status == "manual"


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
        DownloadProviderDescriptor("public_direct", public, lambda _value: True),
        DownloadProviderDescriptor("authorized_skill", skill, lambda _value: False),
    ))
    planner_inputs = []
    grants = Grants([])
    options = AuthorizedSkillOptions(
        enabled=True,
        runtime=Runtime(SimpleNamespace(installed_content_sha256="skill", dependency_lock_sha256="deps")),  # type: ignore[arg-type]
        grant_store=grants,
        authorization_grant_id="grant-1",
        planner=lambda control: planner_inputs.append(control) is None,
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
    registry = DownloadProviderRegistry((DownloadProviderDescriptor("public_direct", public, lambda _value: True),))
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
