from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import threading
import time

import pytest

from paper_agent.provider_runtime import (
    BulkSnapshot,
    CircuitOpenError,
    CircuitState,
    ProviderPolicyDenied,
    ProviderRuntime,
    ProviderRuntimePolicy,
    RetryableProviderError,
    SnapshotDriftError,
    policy_from_manifest,
)
from paper_agent.providers.builtin import load_builtin_manifest


class Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds


def runtime(*policies: ProviderRuntimePolicy, clock: Clock | None = None) -> ProviderRuntime:
    active_clock = clock or Clock()
    return ProviderRuntime(
        {policy.provider: policy for policy in policies},
        clock=active_clock,
        sleeper=active_clock.sleep,
        random_value=lambda: 0.5,
    )


def test_provider_qps_is_global_across_worker_threads() -> None:
    current = Clock()
    client = runtime(ProviderRuntimePolicy("crossref", queries_per_second=2), clock=current)
    starts: list[float] = []

    def send() -> str:
        starts.append(current())
        return "ok"

    with ThreadPoolExecutor(max_workers=3) as workers:
        list(
            workers.map(
                lambda number: client.request(
                    "crossref", query_hash=str(number), cursor=None, api_version="v1", send=send
                ),
                range(3),
            )
        )

    assert starts == [0.0, 0.5, 1.0]


def test_provider_concurrency_is_global_across_worker_threads() -> None:
    client = ProviderRuntime({"openalex": ProviderRuntimePolicy("openalex", max_concurrency=2)})
    entered = 0
    peak = 0
    lock = threading.Lock()

    def send() -> str:
        nonlocal entered, peak
        with lock:
            entered += 1
            peak = max(peak, entered)
        time.sleep(0.02)
        with lock:
            entered -= 1
        return "ok"

    with ThreadPoolExecutor(max_workers=6) as workers:
        list(
            workers.map(
                lambda number: client.request(
                    "openalex", query_hash=str(number), cursor=None, api_version="v1", send=send
                ),
                range(6),
            )
        )

    assert peak == 2


def test_cache_isolated_by_provider_query_cursor_and_version_and_replays_within_ttl() -> None:
    current = Clock()
    client = runtime(
        ProviderRuntimePolicy("crossref", cache_ttl_seconds=10),
        ProviderRuntimePolicy("openalex", cache_ttl_seconds=10),
        clock=current,
    )
    calls: list[str] = []

    def send() -> str:
        calls.append("network")
        return str(len(calls))

    assert client.request("crossref", query_hash="q", cursor=None, api_version="v1", send=send) == "1"
    assert client.request("crossref", query_hash="q", cursor=None, api_version="v1", send=send) == "1"
    assert client.request("crossref", query_hash="q", cursor="next", api_version="v1", send=send) == "2"
    assert client.request("crossref", query_hash="q", cursor=None, api_version="v2", send=send) == "3"
    assert client.request("openalex", query_hash="q", cursor=None, api_version="v1", send=send) == "4"
    current.value = 10
    assert client.request("crossref", query_hash="q", cursor=None, api_version="v1", send=send) == "5"


def test_retry_after_is_honored_before_retry() -> None:
    current = Clock()
    client = runtime(
        ProviderRuntimePolicy("crossref", retry_attempts=2, initial_backoff_seconds=0.1, max_backoff_seconds=1),
        clock=current,
    )
    attempts = 0

    def send() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RetryableProviderError("limited", retry_after=3)
        return "ok"

    assert client.request("crossref", query_hash="q", cursor=None, api_version="v1", send=send) == "ok"
    assert attempts == 2
    assert current() == 3


def test_credentials_return_availability_without_retaining_secret_values() -> None:
    class DeclaredOnly(dict[str, str]):
        reads: list[str] = []

        def get(self, key: str, default: str | None = None) -> str | None:
            self.reads.append(key)
            if key == "UNDECLARED_SECRET":
                pytest.fail("runtime read an undeclared secret")
            return super().get(key, default)

    environment = DeclaredOnly(API_TOKEN="secret", UNDECLARED_SECRET="never read")
    client = runtime(ProviderRuntimePolicy("semantic", credential_environment_variables=("API_TOKEN",)))
    assert client.credentials_available("semantic", environment) is True
    assert environment.reads == ["API_TOKEN"]
    assert "secret" not in repr(client.__dict__)


def test_required_credentials_use_the_injected_declared_environment() -> None:
    client = runtime(
        ProviderRuntimePolicy(
            "semantic", credentials_required=True, credential_environment_variables=("DECLARED_TOKEN",)
        )
    )

    assert client.request(
        "semantic",
        query_hash="q",
        cursor=None,
        api_version="v1",
        send=lambda: "ok",
        environment={"DECLARED_TOKEN": "secret", "UNDECLARED_TOKEN": "never read"},
    ) == "ok"
    with pytest.raises(ProviderPolicyDenied, match="credentials"):
        client.request(
            "semantic", query_hash="q2", cursor=None, api_version="v1", send=lambda: "never", environment={}
        )


@pytest.mark.parametrize("mode", ["snapshot", "bulk_snapshot"])
def test_snapshot_mode_verifies_exact_content_and_never_calls_network(mode: str) -> None:
    content = b'{"records": []}'
    snapshot = BulkSnapshot(content, sha256(content).hexdigest())
    client = runtime(ProviderRuntimePolicy("openalex", cache_ttl_seconds=10))

    def network() -> bytes:
        pytest.fail("snapshot mode must not invoke network")

    assert client.request(
        "openalex",
        query_hash="bulk",
        cursor=None,
        api_version="snapshot-v1",
        send=network,
        mode=mode,
        snapshot=snapshot,
        expected_snapshot_hash=snapshot.content_hash,
    ) == content
    with pytest.raises(SnapshotDriftError):
        client.request(
            "openalex",
            query_hash="bulk-2",
            cursor=None,
            api_version="snapshot-v1",
            mode=mode,
            snapshot=snapshot,
            expected_snapshot_hash="0" * 64,
        )


def test_terms_and_robots_gate_prevent_request() -> None:
    client = runtime(ProviderRuntimePolicy("site", terms_accepted=False))
    with pytest.raises(ProviderPolicyDenied, match="terms"):
        client.request("site", query_hash="q", cursor=None, api_version="v1", send=lambda: "never")
    robots = runtime(ProviderRuntimePolicy("site", robots_allowed=False))
    with pytest.raises(ProviderPolicyDenied, match="robots"):
        robots.request("site", query_hash="q", cursor=None, api_version="v1", send=lambda: "never")


def test_runtime_policy_is_derived_from_the_versioned_manifest() -> None:
    manifest = load_builtin_manifest("crossref")
    policy = policy_from_manifest(manifest, terms_accepted=True, robots_allowed=True)

    assert policy.provider == "crossref"
    assert policy.queries_per_second == 5
    assert policy.max_concurrency == 4
    assert policy.cache_ttl_seconds == 3600


def test_circuit_breaker_is_independent_and_recovers_with_one_probe() -> None:
    current = Clock()
    client = runtime(
        ProviderRuntimePolicy("a", failure_threshold=1, recovery_seconds=2),
        ProviderRuntimePolicy("b", failure_threshold=1, recovery_seconds=2),
        clock=current,
    )

    with pytest.raises(RetryableProviderError):
        client.request(
            "a",
            query_hash="q",
            cursor=None,
            api_version="v1",
            send=lambda: (_ for _ in ()).throw(RetryableProviderError("down")),
        )
    assert client.circuit_state("a") is CircuitState.OPEN
    assert client.request("b", query_hash="q", cursor=None, api_version="v1", send=lambda: "ok") == "ok"
    with pytest.raises(CircuitOpenError):
        client.request("a", query_hash="q2", cursor=None, api_version="v1", send=lambda: "never")
    current.value += 2
    assert client.request("a", query_hash="q2", cursor=None, api_version="v1", send=lambda: "recovered") == "recovered"
    assert client.circuit_state("a") is CircuitState.CLOSED
