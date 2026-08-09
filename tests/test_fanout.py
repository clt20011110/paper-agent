from paper_agent.fanout import fan_out
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
    calls: list[tuple[str, object]] = []

    class Client:
        def search(self, query_spec, cursor):
            calls.append((query_spec.research_question_id, cursor))
            return {"query": query_spec.original_query}

    result = fan_out(_plan(), {"openalex": Client(), "crossref": Client()})

    assert result.incomplete is False
    assert sorted(calls) == [("q1", None), ("q1", None)]
