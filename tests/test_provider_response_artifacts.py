from __future__ import annotations

from email.message import Message

from paper_agent.artifacts import ArtifactStore
from paper_agent.domain import EnvelopeStatus, SourceBatch
from paper_agent.http_transport import ControlledHTTPTransport
from paper_agent.provider_response_artifacts import ProviderResponseArtifactService
from paper_agent.provider_runtime import ProviderRuntime, ProviderRuntimePolicy
from paper_agent.search_runs import SearchRunCoordinator
from paper_agent.storage import Database


NOW = "2026-08-11T00:00:00Z"


class Response:
    def __init__(self, body: bytes, headers: dict[str, str]) -> None:
        self._body = body
        self.headers = Message()
        for key, value in headers.items():
            self.headers[key] = value

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> Response:
        return self

    def __exit__(self, *_: object) -> None:
        return None


def _runtime() -> ProviderRuntime:
    return ProviderRuntime(
        {
            "crossref": ProviderRuntimePolicy(
                "crossref",
                cache_ttl_seconds=3600,
            )
        }
    )


def test_provider_response_replays_exact_bytes_after_process_restart(tmp_path) -> None:
    database_path = tmp_path / "papers.sqlite3"
    artifact_store = ArtifactStore(tmp_path)
    body = b'{"status":"ok","message":{"items":[]}}'
    provider_calls = 0
    first_runtime = _runtime()

    def first_opener(request, timeout):
        nonlocal provider_calls
        provider_calls += 1
        return Response(
            body,
            {
                "Content-Type": "application/json",
                "ETag": '"frozen-response"',
                "RateLimit-Remaining": "9",
            },
        )

    with Database(database_path) as database:
        database.migrate()
        database.connection.execute(
            """INSERT INTO pipeline_runs(
                   run_id, stage, status, input_hash, config_hash, implementation_version
               ) VALUES ('run-1', 'stage-1', 'running', 'input', 'config', 'test')"""
        )
        database.connection.execute(
            """INSERT INTO search_plans(
                   search_plan_id, content_hash, schema_version, plan_json, status
               ) VALUES ('plan-1', 'plan-hash', '1', '{}', 'approved')"""
        )
        database.connection.commit()
        coordinator = SearchRunCoordinator(database)
        coordinator.start_crawl(
            crawl_run_id="crawl-1",
            run_id="run-1",
            search_plan_id="plan-1",
            window={"date_from": "2024-01-01", "date_to": "2024-12-31"},
        )
        transport = ControlledHTTPTransport(
            "https://example.test/contact",
            opener=first_opener,
            runtime=first_runtime,
            response_artifacts=ProviderResponseArtifactService(
                database_path,
                artifact_store,
            ),
            replay_scope="crawl-1",
        )
        parameters = {"query": "durable replay", "page_size": 1}
        first = transport("crossref", "search", parameters)
        batch = SourceBatch(
            source_run_id="source-1",
            query_hash="compiled-query-hash",
            entries=(),
            next_cursor=None,
            status=EnvelopeStatus.SUCCESS,
            raw_response_artifact_hash=str(first["raw_response_artifact_hash"]),
            request_audit=tuple(first["_request_audit"]),
        )
        coordinator.record_batch(
            crawl_run_id="crawl-1",
            provider="crossref",
            provider_version="2026.08",
            role="search",
            query_text="durable replay",
            provider_params=parameters,
            query_compiler_version="1",
            batch=batch,
            requested_at=NOW,
            completed_at=NOW,
            page="1",
        )

    def forbidden_opener(request, timeout):
        nonlocal provider_calls
        provider_calls += 1
        raise AssertionError("persisted replay must not contact the provider")

    restarted_transport = ControlledHTTPTransport(
        "https://example.test/contact",
        opener=forbidden_opener,
        runtime=_runtime(),
        response_artifacts=ProviderResponseArtifactService(
            database_path,
            artifact_store,
        ),
        replay_scope="crawl-1",
    )
    replayed = restarted_transport("crossref", "search", parameters)

    assert provider_calls == 1
    assert restarted_transport.last_response_body == body
    assert replayed["message"] == first["message"]
    assert replayed["raw_response_artifact_hash"] == first["raw_response_artifact_hash"]
    assert replayed["_request_audit"][0]["cache_source"] == "persistent"
    assert replayed["_request_audit"][0]["replay_scope"] == "crawl-1"
    assert first["_request_audit"][0]["rate_limit"] == {"ratelimit-remaining": "9"}
    assert replayed["_request_audit"][0]["rate_limit"] == {}

    fresh_body = b'{"status":"ok","message":{"items":[{"DOI":"10.1/fresh"}]}}'

    def fresh_scope_opener(request, timeout):
        nonlocal provider_calls
        provider_calls += 1
        return Response(fresh_body, {"Content-Type": "application/json"})

    fresh_transport = ControlledHTTPTransport(
        "https://example.test/contact",
        opener=fresh_scope_opener,
        runtime=first_runtime,
        response_artifacts=ProviderResponseArtifactService(
            database_path,
            artifact_store,
        ),
        replay_scope="crawl-2",
    )
    fresh = fresh_transport("crossref", "search", parameters)
    assert provider_calls == 2
    assert fresh_transport.last_response_body == fresh_body
    assert fresh["_request_audit"][0]["cache_source"] == "provider_or_runtime"
    assert fresh["_request_audit"][0]["replay_scope"] == "crawl-2"

    with Database(database_path) as database:
        database.connection.execute(
            """INSERT INTO pipeline_runs(
                   run_id, stage, status, input_hash, config_hash, implementation_version
               ) VALUES ('run-2', 'stage-1', 'running', 'input-2', 'config', 'test')"""
        )
        database.connection.commit()
        coordinator = SearchRunCoordinator(database)
        coordinator.start_crawl(
            crawl_run_id="crawl-2",
            run_id="run-2",
            search_plan_id="plan-1",
            window={"date_from": "2024-01-01", "date_to": "2024-12-31"},
        )
        coordinator.record_batch(
            crawl_run_id="crawl-2",
            provider="crossref",
            provider_version="2026.08",
            role="search",
            query_text="durable replay",
            provider_params=parameters,
            query_compiler_version="1",
            batch=SourceBatch(
                source_run_id="source-2",
                query_hash="compiled-query-hash",
                entries=(),
                next_cursor=None,
                status=EnvelopeStatus.SUCCESS,
                raw_response_artifact_hash=str(fresh["raw_response_artifact_hash"]),
                request_audit=tuple(fresh["_request_audit"]),
            ),
            requested_at=NOW,
            completed_at=NOW,
            page="1",
        )

    with Database(database_path, read_only=True) as database:
        sources = {
            row["source_run_id"]: row["raw_response_artifact_id"]
            for row in database.connection.execute(
                "SELECT source_run_id, raw_response_artifact_id FROM source_runs"
            )
        }
        queries = {
            row["source_run_id"]: row["response_artifact_id"]
            for row in database.connection.execute(
                "SELECT source_run_id, response_artifact_id FROM search_queries"
            )
        }
        cache = database.connection.execute(
            """SELECT c.replay_scope, c.artifact_id, a.sha256
               FROM provider_response_cache c
               JOIN artifacts a ON a.artifact_id = c.artifact_id
               ORDER BY c.replay_scope"""
        ).fetchall()
        assert sources == queries
        assert [row["replay_scope"] for row in cache] == ["crawl-1", "crawl-2"]
        assert cache[0]["artifact_id"] == sources["source-1"]
        assert cache[0]["sha256"] == first["raw_response_artifact_hash"]
        assert cache[1]["artifact_id"] == sources["source-2"]
        assert cache[1]["sha256"] == fresh["raw_response_artifact_hash"]
    assert artifact_store.read_bytes(str(first["raw_response_artifact_hash"])) == body
    assert artifact_store.read_bytes(str(fresh["raw_response_artifact_hash"])) == fresh_body
