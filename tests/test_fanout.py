from paper_agent.domain import EnvelopeStatus, SourceBatch, SourceEntry
from paper_agent.fanout import ProviderPage, fan_out
from paper_agent.query_plan import approve_query_plan, compile_query_plan

from test_query_plan import draft, provider


def _plan() -> dict[str, object]:
    document = draft()
    document["required_providers"] = ["openalex"]
    document["required_roles"] = ["search"]
    plan = compile_query_plan(document, providers=[provider("openalex"), provider("crossref")])
    return approve_query_plan(plan, plan["plan_hash"], approved_by="owner", approved_at="2026-08-09T01:00:00Z")


def test_all_resolved_sources_are_invoked_and_optional_failure_continues() -> None:
    calls: list[str] = []

    def okay(provider, queries):
        calls.append(provider["provider"])
        return {"query_hashes": [query.query_hash for query in queries]}

    def broken(provider, queries):
        calls.append(provider["provider"])
        raise RuntimeError("upstream unavailable")

    result = fan_out(_plan(), {"openalex": okay, "crossref": broken})

    assert sorted(calls) == ["crossref", "openalex"]
    assert result.successful_providers == ("openalex",)
    assert result.incomplete is False


def test_required_provider_failure_marks_fanout_incomplete() -> None:
    def broken(provider, queries):
        raise RuntimeError("upstream unavailable")

    result = fan_out(_plan(), {"openalex": broken, "crossref": broken})

    assert result.incomplete is True
    assert [outcome.status for outcome in result.outcomes] == ["failed", "failed"]


def test_provider_protocol_clients_receive_one_query_spec_per_variant() -> None:
    calls: list[tuple[str, object, str, dict[str, object]]] = []
    scopes: list[tuple[object, ...]] = []

    class Client:
        def search(self, query_spec, cursor):
            calls.append(
                (
                    query_spec.research_question_id,
                    cursor,
                    query_spec.native_query_hash,
                    dict(query_spec.native_parameters),
                )
            )
            scopes.append(
                (
                    query_spec.date_from,
                    query_spec.date_to,
                    query_spec.venue_ids,
                    query_spec.fields,
                    query_spec.languages,
                    query_spec.document_types,
                )
            )
            return {"query": query_spec.original_query}

    result = fan_out(_plan(), {"openalex": Client(), "crossref": Client()})

    assert result.incomplete is False
    assert [call[:2] for call in sorted(calls)] == [("q1", None), ("q1", None)]
    assert all(len(call[2]) == 64 for call in calls)
    assert {next(iter(call[3])) for call in calls} == {"query.bibliographic", "search"}
    assert scopes == [
        (
            "2020-01-01",
            "2024-12-31",
            (),
            ("computer science",),
            ("en",),
            ("article",),
        )
    ] * 2


def test_protocol_client_paginates_until_cursor_is_empty() -> None:
    calls: list[str | None] = []

    class Client:
        def search(self, query_spec, cursor):
            calls.append(cursor)
            return SourceBatch(
                "source",
                query_spec.native_query_hash,
                (),
                "next" if cursor is None else None,
                EnvelopeStatus.SUCCESS,
            )

    result = fan_out(_plan(), {"openalex": Client(), "crossref": Client()})

    assert result.incomplete is False
    assert calls == [None, "next", None, "next"] or calls == [None, None, "next", "next"]
    assert all(isinstance(page, ProviderPage) for outcome in result.outcomes for page in outcome.result)


def test_non_search_provider_failure_is_isolated_in_outcome() -> None:
    plan = _plan()
    plan["providers"][0]["roles"] = ["venue_primary"]
    plan["providers"][0]["native_query_hashes"] = []

    result = fan_out(plan, {"openalex": object(), "crossref": lambda *_: ()})

    failed = next(outcome for outcome in result.outcomes if outcome.provider == "openalex")
    assert failed.status == "failed"
    assert "invocation adapter" in failed.error


def test_later_page_failure_preserves_earlier_page_and_marks_partial() -> None:
    class Client:
        def search(self, query_spec, cursor):
            if cursor:
                raise RuntimeError("page two failed")
            return SourceBatch(
                "source",
                query_spec.native_query_hash,
                (),
                "page-2",
                EnvelopeStatus.SUCCESS,
            )

    result = fan_out(_plan(), {"openalex": Client(), "crossref": Client()})

    assert result.incomplete is True
    assert [outcome.status for outcome in result.outcomes] == ["partial", "partial"]
    assert all(len(outcome.result) == 2 for outcome in result.outcomes)


def test_tight_request_budget_selects_pages_by_frozen_provider_order() -> None:
    plan = _plan()
    plan["budgets"] = {**plan["budgets"], "max_requests": 1, "max_candidates": 100}
    calls: list[str] = []

    class Client:
        def __init__(self, name):
            self.name = name

        def search(self, query_spec, cursor):
            calls.append(self.name)
            return SourceBatch(self.name, query_spec.native_query_hash, (), None, EnvelopeStatus.SUCCESS)

    result = fan_out(plan, {"openalex": Client("openalex"), "crossref": Client("crossref")})

    assert calls == ["crossref"]
    assert [outcome.status for outcome in result.outcomes] == ["success", "skipped_budget"]
    assert (result.incomplete, result.budget_exhausted, result.requests_made) == (True, True, 1)


def test_candidate_cap_is_deterministic_across_a_parallel_page_wave() -> None:
    plan = _plan()
    plan["budgets"] = {**plan["budgets"], "max_requests": 10, "max_candidates": 1}

    class Client:
        def __init__(self, name):
            self.name = name

        def search(self, query_spec, cursor):
            return SourceBatch(
                self.name,
                query_spec.native_query_hash,
                (SourceEntry(self.name, "one", "One"),),
                None,
                EnvelopeStatus.SUCCESS,
            )

    result = fan_out(plan, {"openalex": Client("openalex"), "crossref": Client("crossref")})
    pages = {outcome.provider: outcome.result[0].batch for outcome in result.outcomes}

    assert len(pages["crossref"].entries) == 1
    assert (pages["openalex"].status, len(pages["openalex"].entries)) == (EnvelopeStatus.PARTIAL, 0)
    assert (result.candidates_returned, result.budget_exhausted) == (1, True)


def test_pagination_budget_replay_and_roomy_budget_parallelism() -> None:
    plan = _plan()
    plan["budgets"] = {**plan["budgets"], "max_requests": 3, "max_candidates": 100}
    calls: list[tuple[str, str | None]] = []

    class Paginated:
        def __init__(self, name):
            self.name = name

        def search(self, query_spec, cursor):
            calls.append((self.name, cursor))
            return SourceBatch(self.name, query_spec.native_query_hash, (), "next" if cursor is None else None, EnvelopeStatus.SUCCESS)

    result = fan_out(plan, {"openalex": Paginated("openalex"), "crossref": Paginated("crossref")})
    assert calls == [("crossref", None), ("openalex", None), ("crossref", "next")]
    assert result.budget_exhausted is True

    roomy = _plan()
    roomy["budgets"] = {**roomy["budgets"], "max_requests": 10, "max_candidates": 100}
    from threading import Barrier

    barrier = Barrier(2)

    class Concurrent:
        def search(self, query_spec, cursor):
            barrier.wait(timeout=1)
            return SourceBatch("source", query_spec.native_query_hash, (), None, EnvelopeStatus.SUCCESS)

    result = fan_out(roomy, {"openalex": Concurrent(), "crossref": Concurrent()}, max_workers=2)
    assert result.budget_exhausted is False


def test_final_page_that_runs_past_deadline_is_limited(monkeypatch) -> None:
    plan = _plan()
    plan["budgets"] = {**plan["budgets"], "max_seconds": 1}
    clock = iter((0.0, 0.0, 2.0))
    monkeypatch.setattr("paper_agent.fanout.time.monotonic", lambda: next(clock))

    class Client:
        def search(self, query_spec, cursor):
            return SourceBatch("source", query_spec.native_query_hash, (), None, EnvelopeStatus.SUCCESS)

    result = fan_out(plan, {"openalex": Client(), "crossref": Client()})
    assert result.budget_exhausted is True


def test_repeated_cursor_stops_its_stream_without_spending_the_budget() -> None:
    calls: list[str | None] = []

    class Client:
        def search(self, query_spec, cursor):
            calls.append(cursor)
            return SourceBatch("source", query_spec.native_query_hash, (), "again", EnvelopeStatus.SUCCESS)

    result = fan_out(_plan(), {"openalex": Client(), "crossref": Client()})

    assert calls == [None, None, "again", "again"]
    assert [outcome.status for outcome in result.outcomes] == ["partial", "partial"]
    assert result.budget_exhausted is False
