"""Thread-safe runtime policy enforcement for Stage 1 providers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import os
import random as random_module
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any, TypeVar

from .providers.api import ProviderManifest


T = TypeVar("T")


class ProviderPolicyDenied(ValueError):
    pass


class SnapshotDriftError(ValueError):
    pass


class CircuitOpenError(RuntimeError):
    pass


class ProviderRequestError(RuntimeError):
    """A provider request failed with optional HTTP retry metadata."""

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class RetryableProviderError(ProviderRequestError):
    pass


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass(frozen=True, slots=True)
class ProviderRuntimePolicy:
    provider: str
    queries_per_second: float | None = None
    max_concurrency: int = 1
    cache_ttl_seconds: float | None = None
    credentials_required: bool = False
    credential_environment_variables: tuple[str, ...] = ()
    terms_accepted: bool = True
    robots_allowed: bool = True
    retry_attempts: int = 3
    initial_backoff_seconds: float = 0.25
    max_backoff_seconds: float = 8.0
    jitter_seconds: float = 0.1
    failure_threshold: int = 3
    recovery_seconds: float = 30.0

    def __post_init__(self) -> None:
        if self.queries_per_second is not None and self.queries_per_second <= 0:
            raise ValueError("queries_per_second must be positive")
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be at least one")
        if self.cache_ttl_seconds is not None and self.cache_ttl_seconds < 0:
            raise ValueError("cache_ttl_seconds cannot be negative")
        if self.retry_attempts < 1:
            raise ValueError("retry_attempts must be at least one")
        if self.initial_backoff_seconds < 0 or self.max_backoff_seconds < self.initial_backoff_seconds:
            raise ValueError("backoff bounds are invalid")
        if self.jitter_seconds < 0:
            raise ValueError("jitter_seconds cannot be negative")
        if self.failure_threshold < 1 or self.recovery_seconds < 0:
            raise ValueError("circuit breaker bounds are invalid")


def policy_from_manifest(
    manifest: ProviderManifest,
    *,
    terms_accepted: bool,
    robots_allowed: bool,
) -> ProviderRuntimePolicy:
    return ProviderRuntimePolicy(
        provider=manifest.provider,
        queries_per_second=manifest.rate_limit_policy.queries_per_second,
        max_concurrency=manifest.rate_limit_policy.max_concurrency,
        cache_ttl_seconds=manifest.rate_limit_policy.cache_ttl_seconds,
        credentials_required=manifest.credential_policy.required,
        credential_environment_variables=manifest.credential_policy.environment_variables,
        terms_accepted=terms_accepted,
        robots_allowed=robots_allowed,
    )


@dataclass(frozen=True, slots=True)
class CacheKey:
    provider: str
    query_hash: str
    cursor: str | None
    api_version: str


@dataclass(frozen=True, slots=True)
class BulkSnapshot:
    """A user-supplied bulk response; it is never fetched by this runtime."""

    content: bytes
    content_hash: str

    def verify(self, expected_hash: str) -> bytes:
        actual = sha256(self.content).hexdigest()
        if actual != self.content_hash or actual != expected_hash:
            raise SnapshotDriftError("bulk snapshot content hash has drifted")
        return self.content


@dataclass(slots=True)
class _CachedValue:
    value: Any
    expires_at: float


@dataclass(slots=True)
class _Circuit:
    state: CircuitState = CircuitState.CLOSED
    failures: int = 0
    opened_at: float | None = None
    probe_in_flight: bool = False


@dataclass(slots=True)
class _ProviderState:
    policy: ProviderRuntimePolicy
    semaphore: threading.BoundedSemaphore
    rate_lock: threading.Lock
    circuit_lock: threading.Lock
    last_request_at: float | None = None
    circuit: _Circuit | None = None


class ProviderRuntime:
    """Applies frozen provider policies to every API or snapshot request.

    One runtime instance is shared by fan-out workers, making QPS and concurrency
    provider-global within the process rather than worker-local.
    """

    # A provider can advertise an hours-long Retry-After window when a public
    # quota is exhausted.  A batch metadata run must fail fast and leave the
    # item resumable instead of parking every worker for that entire window.
    _MAX_RETRY_AFTER_SECONDS = 30.0

    def __init__(
        self,
        policies: Mapping[str, ProviderRuntimePolicy],
        *,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
        random_value: Callable[[], float] = random_module.random,
    ) -> None:
        self._policies = dict(policies)
        if set(self._policies) != {policy.provider for policy in self._policies.values()}:
            raise ValueError("policy mapping keys must match provider names")
        self._states = {
            name: _ProviderState(
                policy=policy,
                semaphore=threading.BoundedSemaphore(policy.max_concurrency),
                rate_lock=threading.Lock(),
                circuit_lock=threading.Lock(),
                circuit=_Circuit(),
            )
            for name, policy in self._policies.items()
        }
        self._cache: dict[CacheKey, _CachedValue] = {}
        self._cache_lock = threading.Lock()
        self._clock = clock
        self._sleeper = sleeper
        self._random_value = random_value

    def credentials_available(self, provider: str, environment: Mapping[str, str] | None = None) -> bool:
        """Return only availability, reading only env variables declared by policy."""
        policy = self._policy(provider)
        values = environment if environment is not None else os.environ
        return all(bool(values.get(name)) for name in policy.credential_environment_variables)

    def assert_allowed(self, provider: str, environment: Mapping[str, str] | None = None) -> None:
        policy = self._policy(provider)
        if not policy.terms_accepted:
            raise ProviderPolicyDenied(f"{provider}: terms are not accepted")
        if not policy.robots_allowed:
            raise ProviderPolicyDenied(f"{provider}: robots policy forbids automated access")
        if policy.credentials_required and not self.credentials_available(provider, environment):
            raise ProviderPolicyDenied(f"{provider}: declared credentials are unavailable")

    def circuit_state(self, provider: str) -> CircuitState:
        state = self._state(provider)
        with state.circuit_lock:
            return state.circuit.state

    def request(
        self,
        provider: str,
        *,
        query_hash: str,
        cursor: str | None,
        api_version: str,
        send: Callable[[], T] | None = None,
        mode: str = "api",
        snapshot: BulkSnapshot | None = None,
        expected_snapshot_hash: str | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> T | bytes:
        """Return a cached/API value or a verified user-supplied snapshot.

        Snapshot mode intentionally has no network fallback: callers must provide
        the exact approved content and hash.
        """
        self.assert_allowed(provider, environment)
        if mode in {"snapshot", "bulk_snapshot"}:
            if snapshot is None or expected_snapshot_hash is None:
                raise ProviderPolicyDenied("snapshot mode requires approved snapshot content and hash")
            return snapshot.verify(expected_snapshot_hash)
        if mode != "api":
            raise ProviderPolicyDenied(f"unknown provider mode: {mode}")
        if send is None:
            raise ProviderPolicyDenied("API mode requires a request callable")
        key = CacheKey(provider, query_hash, cursor, api_version)
        cached = self._cached(key)
        if cached is not None:
            return cached

        state = self._state(provider)
        self._before_request(provider, state)
        self._wait_for_rate_limit(state)
        state.semaphore.acquire()
        try:
            value = self._send_with_retries(send, state.policy)
        except ProviderRequestError:
            self._record_failure(state)
            raise
        else:
            self._record_success(state)
            self._cache_value(key, value, provider)
            return value
        finally:
            state.semaphore.release()

    def _policy(self, provider: str) -> ProviderRuntimePolicy:
        try:
            return self._policies[provider]
        except KeyError as error:
            raise ProviderPolicyDenied(f"unknown provider: {provider}") from error

    def _state(self, provider: str) -> _ProviderState:
        self._policy(provider)
        return self._states[provider]

    def _cached(self, key: CacheKey) -> Any | None:
        with self._cache_lock:
            cached = self._cache.get(key)
            if cached is None:
                return None
            if cached.expires_at <= self._clock():
                del self._cache[key]
                return None
            return cached.value

    def _cache_value(self, key: CacheKey, value: Any, provider: str) -> None:
        ttl = self._policy(provider).cache_ttl_seconds
        if ttl is None or ttl == 0:
            return
        with self._cache_lock:
            self._cache[key] = _CachedValue(value=value, expires_at=self._clock() + ttl)

    def _before_request(self, provider: str, state: _ProviderState) -> None:
        now = self._clock()
        with state.circuit_lock:
            circuit = state.circuit
            if circuit.state is CircuitState.OPEN:
                assert circuit.opened_at is not None
                if now - circuit.opened_at < state.policy.recovery_seconds:
                    raise CircuitOpenError(f"{provider}: circuit is open")
                circuit.state = CircuitState.HALF_OPEN
            if circuit.state is CircuitState.HALF_OPEN:
                if circuit.probe_in_flight:
                    raise CircuitOpenError(f"{provider}: circuit recovery probe is active")
                circuit.probe_in_flight = True

    def _record_success(self, state: _ProviderState) -> None:
        with state.circuit_lock:
            state.circuit = _Circuit()

    def _record_failure(self, state: _ProviderState) -> None:
        with state.circuit_lock:
            circuit = state.circuit
            if circuit.state is CircuitState.HALF_OPEN:
                circuit.state = CircuitState.OPEN
                circuit.opened_at = self._clock()
                circuit.probe_in_flight = False
                return
            circuit.failures += 1
            if circuit.failures >= state.policy.failure_threshold:
                circuit.state = CircuitState.OPEN
                circuit.opened_at = self._clock()

    def _wait_for_rate_limit(self, state: _ProviderState) -> None:
        qps = state.policy.queries_per_second
        if qps is None:
            return
        interval = 1 / qps
        with state.rate_lock:
            if state.last_request_at is not None:
                delay = state.last_request_at + interval - self._clock()
                if delay > 0:
                    self._sleeper(delay)
            state.last_request_at = self._clock()

    def _send_with_retries(self, send: Callable[[], T], policy: ProviderRuntimePolicy) -> T:
        for attempt in range(policy.retry_attempts):
            try:
                return send()
            except RetryableProviderError as error:
                if attempt + 1 == policy.retry_attempts:
                    raise
                if (
                    error.retry_after is not None
                    and error.retry_after > self._MAX_RETRY_AFTER_SECONDS
                ):
                    raise
                self._sleeper(self._retry_delay(error, attempt, policy))
        raise AssertionError("retry loop always returns or raises")

    def _retry_delay(self, error: RetryableProviderError, attempt: int, policy: ProviderRuntimePolicy) -> float:
        if error.retry_after is not None:
            return error.retry_after
        backoff = min(policy.max_backoff_seconds, policy.initial_backoff_seconds * (2**attempt))
        available_jitter = min(policy.jitter_seconds, policy.max_backoff_seconds - backoff)
        return backoff + self._random_value() * available_jitter
