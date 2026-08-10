from paper_agent.query_compilers import compile_queries


SCOPE = {
    "date_from": "2020-01-01",
    "date_to": "2024-12-31",
    "venues": ["venue-1"],
    "fields": ["computer science"],
    "languages": ["en"],
    "document_types": ["article"],
}
VARIANT = {"id": "q1", "raw_query": "graph learning", "synonyms": ["GNN"]}


def test_built_in_broad_sources_have_distinct_deterministic_native_queries() -> None:
    providers = (
        "crossref",
        "dblp",
        "semantic_scholar",
        "openalex",
        "pubmed",
        "europe_pmc",
        "arxiv",
    )
    rendered = [compile_queries(provider, [VARIANT], SCOPE)[0] for provider in providers]

    assert len({str(query.parameters) for query in rendered}) == len(providers)
    assert [query.query_hash for query in rendered] == [
        compile_queries(provider, [VARIANT], SCOPE)[0].query_hash for provider in providers
    ]


def test_native_query_hash_and_audit_bind_every_requested_filter() -> None:
    openalex = compile_queries("openalex", [VARIANT], SCOPE)[0]
    dblp = compile_queries("dblp", [VARIANT], SCOPE)[0]
    shifted = compile_queries(
        "dblp",
        [VARIANT],
        {**SCOPE, "date_from": "2021-01-01"},
    )[0]

    assert openalex.requested_filters == SCOPE
    assert openalex.native_applied_filters == {
        "date_from": "2020-01-01",
        "date_to": "2024-12-31",
    }
    assert openalex.post_filters == {
        "venues": ["venue-1"],
        "fields": ["computer science"],
        "languages": ["en"],
        "document_types": ["article"],
    }
    assert dblp.native_applied_filters == {}
    assert dblp.post_filters == SCOPE
    assert dblp.parameters == shifted.parameters
    assert dblp.query_hash != shifted.query_hash


def test_coarse_native_year_filter_keeps_exact_dates_for_post_filtering() -> None:
    query = compile_queries("semantic_scholar", [VARIANT], SCOPE)[0]

    assert query.native_applied_filters == {"year_from": "2020", "year_to": "2024"}
    assert query.post_filters["date_from"] == "2020-01-01"
    assert query.post_filters["date_to"] == "2024-12-31"
