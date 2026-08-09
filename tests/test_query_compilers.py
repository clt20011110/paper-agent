from paper_agent.query_compilers import compile_queries


SCOPE = {
    "date_from": "2020-01-01",
    "date_to": "2024-12-31",
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
