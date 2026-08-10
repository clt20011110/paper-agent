from __future__ import annotations

import json
from pathlib import Path

import pytest

from paper_agent.domain import AccessBasis, CitationEdgeType, Paper, ProviderRole, PublicationVersion, QuerySpec
from paper_agent.providers.api import AccessPolicy, CrawlWindow, SeedInput, VenueDescriptor
from paper_agent.providers.builtin import (
    AAASScienceAdapter,
    AAAIOJSAdapter,
    ACLAnthologyAdapter,
    ArXivProvider,
    BUILTIN_CLASSES,
    CVFOpenAccessAdapter,
    EDAProceedingsAdapter,
    EuropePMCProvider,
    FixtureTransport,
    IEEEXploreAdapter,
    IJCAIAdapter,
    LibrarySeedImporter,
    NeurIPSProceedingsAdapter,
    OpenAlexProvider,
    OpenReviewAdapter,
    PMLRAdapter,
    SemanticScholarProvider,
    SpringerNatureAdapter,
    CellPressAdapter,
    create_builtin,
    load_builtin_manifest,
)


FIXTURES = Path(__file__).parent / "fixtures" / "providers"


def fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def adapter(provider: str, response: dict | None = None):
    return create_builtin(provider, FixtureTransport({f"{provider}:discover:first": response or fixture("official-page-1.json")}))


@pytest.mark.parametrize(
    ("provider", "klass", "parameters"),
    [
        ("neurips_proceedings", NeurIPSProceedingsAdapter, {}),
        ("pmlr", PMLRAdapter, {"volume_id": "v235"}),
        ("openreview", OpenReviewAdapter, {"api_version": "v2", "invitation": "ICLR.cc/2025/Conference/-/Decision", "accepted_decision_required": False}),
        ("aaai_ojs", AAAIOJSAdapter, {"issue_ids": [1, 2]}),
        ("acl_anthology", ACLAnthologyAdapter, {"snapshot_version": "1941968b51805719b418a0b0919e335662cdd172", "track": "main"}),
        ("cvf_open_access", CVFOpenAccessAdapter, {"track": "main"}),
        ("ijcai_proceedings", IJCAIAdapter, {}),
        ("eda_proceedings", EDAProceedingsAdapter, {"upstreams": ["ieee", "acm"]}),
        ("ieee_xplore", IEEEXploreAdapter, {"publication_number": 43, "issn": "0278-0070"}),
        ("springer_nature", SpringerNatureAdapter, {"journal_slug": "natmachintell", "issns": ["2522-5839"], "article_types": ["Article"]}),
        ("cell_press", CellPressAdapter, {"issn": "0092-8674"}),
        ("aaas_science", AAASScienceAdapter, {"issns": ["0036-8075", "1095-9203"]}),
    ],
)
def test_official_adapter_families_use_exact_fixture(provider: str, klass: type, parameters: dict) -> None:
    instance = adapter(provider)
    assert isinstance(instance, klass)
    batch = instance.discover(VenueDescriptor(1, provider, provider, provider, parameters), CrawlWindow(year=2025), None)
    assert batch.entries[0].external_id == "official-001"
    assert batch.entries[0].doi == "10.1000/fixture"
    assert batch.next_cursor == "page-2"
    assert batch.raw_response_artifact_hash == "fixture-response"


@pytest.mark.parametrize("version,fixture_name,expected", [("v1", "openreview-v1.json", "note-v1"), ("v2", "openreview-v2.json", "note-v2")])
def test_openreview_uses_dynamic_invitation_and_v1_v2_payloads(version: str, fixture_name: str, expected: str) -> None:
    payload = fixture(fixture_name)
    transport = FixtureTransport({"openreview:discover:first": payload})
    instance = OpenReviewAdapter("openreview", transport)
    descriptor = VenueDescriptor(1, "iclr", "openreview", "openreview", {"api_version": version, "invitation": "ICLR.cc/2025/Conference/-/Decision", "accepted_decision_required": False})
    assert instance.discover(descriptor, CrawlWindow(year=2025), None).entries[0].external_id == expected
    assert transport.calls[0][2]["invitation"].endswith("Decision")


def test_openreview_resolves_invitation_from_group_and_excludes_rejected_records() -> None:
    payload = {
        "notes": [
            {"id": "accepted", "title": "Accepted", "content": {"venueid": {"value": "ICLR.cc/2025/Conference"}}},
            {"id": "rejected", "title": "Rejected", "decision": "Reject"},
            {"id": "submission", "title": "Submission", "venue": "ICLR 2025 Conference Submission"},
        ]
    }
    transport = FixtureTransport(
        {
            "openreview:resolve_invitation:first": {
                "invitation": "ICLR.cc/2025/Conference/-/Decision",
                "api_version": "v2",
                "accepted_venue_ids": ["ICLR.cc/2025/Conference"],
            },
            "openreview:discover:first": payload,
        }
    )
    descriptor = VenueDescriptor(
        1,
        "iclr",
        "openreview",
        "openreview",
        {"venue_group": "ICLR.cc", "accepted_decision_required": True},
    )
    batch = OpenReviewAdapter("openreview", transport).discover(descriptor, CrawlWindow(year=2025))

    assert [entry.external_id for entry in batch.entries] == ["accepted"]
    assert transport.calls[0][1] == "resolve_invitation"


def test_openreview_acceptance_requires_exact_venue_id_or_accept_decision() -> None:
    payload = {
        "notes": [
            {"id": "exact", "title": "Exact", "content": {"venueid": {"value": "ICLR.cc/2025/Conference"}}},
            {"id": "decision", "title": "Decision", "decision": "Accept (Poster)"},
            {"id": "submission", "title": "Submission", "venue": "ICLR 2025 Conference Submission"},
            {"id": "other", "title": "Other", "content": {"venueid": {"value": "ICLR.cc/2025/Conference/Submission"}}},
        ]
    }
    descriptor = VenueDescriptor(
        1,
        "iclr",
        "openreview",
        "openreview",
        {
            "api_version": "v2",
            "invitation": "ICLR.cc/2025/Conference/-/Submission",
            "accepted_venue_ids": ["ICLR.cc/2025/Conference"],
        },
    )
    batch = OpenReviewAdapter("openreview", FixtureTransport({"openreview:discover:first": payload})).discover(
        descriptor, CrawlWindow(year=2025)
    )

    assert [entry.external_id for entry in batch.entries] == ["exact", "decision"]


def test_cursor_and_window_mapping_are_transport_visible() -> None:
    transport = FixtureTransport({"neurips_proceedings:discover:cursor-2": fixture("official-page-1.json")})
    instance = NeurIPSProceedingsAdapter("neurips_proceedings", transport)
    instance.discover(VenueDescriptor(1, "neurips", "neurips_proceedings", "neurips_proceedings"), CrawlWindow(date_from="2024-01-01", date_to="2024-12-31", year=2024), "cursor-2")
    assert transport.calls == [("neurips_proceedings", "discover", {"venue_id": "neurips", "adapter": "neurips_proceedings", "date_from": "2024-01-01", "date_to": "2024-12-31", "year": 2024, "volume": None, "issue": None, "cursor": "cursor-2"})]


def test_platform_specific_response_shapes_remain_protocol_only() -> None:
    transport = FixtureTransport(
        {
            "aaai_ojs:discover:first": {"issues": [{"id": 7, "articles": [{"id": "aaai-7", "title": "AAAI issue paper"}]}]},
            "cvf_open_access:discover:first": {"main": [{"id": "cvf-main", "title": "Main"}], "workshop": [{"id": "cvf-workshop", "title": "Workshop"}]},
            "eda_proceedings:discover:first": {"upstreams": {"ieee": {"entries": [{"id": "ieee-1", "title": "IEEE"}]}, "acm": {"entries": [{"id": "acm-1", "title": "ACM"}]}}},
        }
    )
    aaai = AAAIOJSAdapter("aaai_ojs", transport).discover(VenueDescriptor(1, "aaai", "aaai_ojs", "aaai_ojs", {"issue_ids": [7]}), CrawlWindow(year=2025))
    cvf = CVFOpenAccessAdapter("cvf_open_access", transport).discover(VenueDescriptor(1, "cvpr", "cvf_open_access", "cvf_open_access", {"track": "workshop"}), CrawlWindow(year=2025))
    eda = EDAProceedingsAdapter("eda_proceedings", transport).discover(VenueDescriptor(1, "dac", "eda_proceedings", "eda_proceedings", {"upstreams": ["ieee", "acm"]}), CrawlWindow(year=2025))
    assert aaai.entries[0].metadata["ojs_issue_id"] == 7
    assert cvf.entries[0].external_id == "cvf-workshop"
    assert {entry.metadata["upstream"] for entry in eda.entries} == {"ieee", "acm"}


def test_pmlr_extracts_volume_id_from_official_volume_url() -> None:
    transport = FixtureTransport({"pmlr:discover:first": fixture("official-page-1.json")})
    PMLRAdapter("pmlr", transport).discover(VenueDescriptor(1, "icml", "pmlr", "pmlr", {"volume_url": "https://proceedings.mlr.press/v235/"}), CrawlWindow(year=2024))
    assert transport.calls[0][2]["volume_id"] == "v235"


def test_broad_providers_are_active_and_optional_products_are_not() -> None:
    active = {"arxiv", "crossref", "dblp", "semantic_scholar", "openalex", "pubmed", "europe_pmc", "unpaywall"}
    assert active.issubset(BUILTIN_CLASSES)
    assert not {"exa", "gemini_search", "deepxiv", "alphaxiv"}.intersection(BUILTIN_CLASSES)


@pytest.mark.parametrize("provider", ["arxiv", "crossref", "dblp", "semantic_scholar", "openalex", "pubmed", "europe_pmc"])
def test_broad_search_providers_map_fixture_records(provider: str) -> None:
    transport = FixtureTransport({f"{provider}:search:first": fixture("official-page-1.json")})
    batch = create_builtin(provider, transport).search(QuerySpec(1, "rq", "fixture"))
    assert batch.entries[0].external_id == "official-001"


def test_crossref_native_message_shape_maps_date_and_author() -> None:
    payload = {"message": {"items": [{"DOI": "10.1000/crossref", "title": ["Crossref Fixture"], "author": [{"given": "Ada", "family": "Lovelace"}], "published": {"date-parts": [[2025, 2, 3]]}}]}}
    transport = FixtureTransport({"crossref:search:first": payload})
    entry = create_builtin("crossref", transport).search(QuerySpec(1, "rq", "fixture")).entries[0]
    assert (entry.external_id, entry.authors, entry.publication_date) == ("10.1000/crossref", ("Ada Lovelace",), "2025-02-03")


def test_search_uses_frozen_native_parameters_and_hash() -> None:
    transport = FixtureTransport({"crossref:search:first": fixture("crossref-native.json")})
    spec = QuerySpec(
        1,
        "rq",
        "uncompiled query",
        native_parameters={"query.bibliographic": "native query", "rows": 25},
        native_query_hash="frozen-native-hash",
    )
    batch = create_builtin("crossref", transport).search(spec)

    assert transport.calls == [
        ("crossref", "search", {"query.bibliographic": "native query", "rows": 25, "cursor": None})
    ]
    assert batch.query_hash == "frozen-native-hash"
    assert batch.next_cursor == "DnF1ZXJ5VGhlbkZldGNoBQAAAAA="


def test_dblp_native_hit_info_mapping_and_offset() -> None:
    batch = create_builtin(
        "dblp", FixtureTransport({"dblp:search:first": fixture("dblp-native.json")})
    ).search(QuerySpec(1, "rq", "fixture"))
    entry = batch.entries[0]

    assert (entry.external_id, entry.title) == ("conf/icml/native-2025", "Native DBLP Fixture")
    assert entry.authors == ("Ada Lovelace", "Grace Hopper")
    assert (entry.doi, entry.year, entry.venue_name) == ("10.1000/dblp.native", 2025, "ICML")
    assert batch.next_cursor == "1"


def test_semantic_scholar_native_external_ids_and_pagination() -> None:
    batch = create_builtin(
        "semantic_scholar",
        FixtureTransport({"semantic_scholar:search:first": fixture("semantic-scholar-search.json")}),
    ).search(QuerySpec(1, "rq", "fixture"))
    entry = batch.entries[0]

    assert entry.external_id == "s2-paper-1"
    assert (entry.doi, entry.arxiv_id) == ("10.1000/s2.native", "2501.01234")
    assert entry.authors == ("Ada Lovelace", "Grace Hopper")
    assert batch.next_cursor == "1"


def test_semantic_scholar_native_citation_wrappers_preserve_direction() -> None:
    transport = FixtureTransport(
        {
            "semantic_scholar:citations:first": fixture("semantic-scholar-citations.json"),
            "semantic_scholar:references:first": fixture("semantic-scholar-references.json"),
        }
    )
    provider = create_builtin("semantic_scholar", transport)
    seed = Paper("canonical-seed", "Seed")

    cited = provider.references(seed)
    citing = provider.citations(seed)
    assert (cited.entries[0].source_paper_id, cited.entries[0].target_paper_id) == (
        "canonical-seed",
        "s2-cited-paper",
    )
    assert (citing.entries[0].source_paper_id, citing.entries[0].target_paper_id) == (
        "s2-citing-paper",
        "canonical-seed",
    )
    assert citing.next_cursor == "100"
    assert citing.entries[0].raw_evidence["contexts"] == ["Prior context"]
    assert cited.entries[0].candidate.external_id == "s2-cited-paper"
    assert citing.entries[0].candidate.external_id == "s2-citing-paper"


def test_openalex_native_work_mapping_and_cursor() -> None:
    batch = create_builtin(
        "openalex", FixtureTransport({"openalex:search:first": fixture("openalex-native.json")})
    ).search(QuerySpec(1, "rq", "fixture"))
    entry = batch.entries[0]

    assert entry.external_id == "https://openalex.org/W1234567890"
    assert entry.authors == ("Ada Lovelace", "Grace Hopper")
    assert entry.doi == "10.1000/openalex.native"
    assert entry.abstract == "Graph retrieval works"
    assert entry.venue_name == "Proceedings of Machine Learning Research"
    assert batch.next_cursor == "IlsxMDAuMCwgJ2h0dHBzOi8vb3BlbmFsZXgub3JnL1cyJ10i"


def test_pubmed_esummary_uses_uid_order_and_article_ids() -> None:
    batch = create_builtin(
        "pubmed", FixtureTransport({"pubmed:search:first": fixture("pubmed-esummary.json")})
    ).search(QuerySpec(1, "rq", "fixture"))
    entry = batch.entries[0]

    assert (entry.external_id, entry.title) == ("39900001", "Native PubMed ESummary Fixture")
    assert entry.authors == ("Lovelace A", "Hopper G")
    assert (entry.doi, entry.year, entry.venue_name) == ("10.1000/pubmed.native", 2025, "Nature")
    assert entry.publication_date == "2025-01-17"


def test_europe_pmc_native_result_list_and_cursor() -> None:
    batch = create_builtin(
        "europe_pmc", FixtureTransport({"europe_pmc:search:first": fixture("europe-pmc-native.json")})
    ).search(QuerySpec(1, "rq", "fixture"))
    entry = batch.entries[0]

    assert (entry.external_id, entry.doi) == ("39900001", "10.1000/epmc.native")
    assert entry.authors == ("Ada Lovelace", "Grace Hopper")
    assert (entry.publication_date, entry.year) == ("2025-01-17", 2025)
    assert batch.next_cursor == "AoIIP_4r0ig1NTQ0NTA0OA=="


def test_arxiv_decoded_atom_feed_entry_mapping() -> None:
    batch = create_builtin(
        "arxiv", FixtureTransport({"arxiv:search:first": fixture("arxiv-atom.json")})
    ).search(QuerySpec(1, "rq", "fixture"))
    entry = batch.entries[0]

    assert (entry.external_id, entry.arxiv_id) == ("2501.01234v2", "2501.01234v2")
    assert entry.title == "Native arXiv Atom Fixture"
    assert entry.authors == ("Ada Lovelace", "Grace Hopper")
    assert (entry.abstract, entry.doi) == ("A decoded Atom summary.", "10.1000/arxiv.native")
    assert batch.next_cursor == "1"


def test_openalex_native_reference_and_citation_shapes() -> None:
    transport = FixtureTransport(
        {
            "openalex:references:first": {
                "referenced_works": ["https://openalex.org/W2"],
                "observed_at": "2025-03-01",
            },
            "openalex:citations:first": {
                "results": [{"id": "https://openalex.org/W3"}],
                "meta": {"next_cursor": "next-page"},
                "observed_at": "2025-03-01",
            },
        }
    )
    provider = create_builtin("openalex", transport)
    seed = Paper("canonical-seed", "Seed")

    reference = provider.references(seed)
    citation = provider.citations(seed)

    assert (reference.entries[0].source_paper_id, reference.entries[0].target_paper_id) == (
        "canonical-seed",
        "https://openalex.org/W2",
    )
    assert (citation.entries[0].source_paper_id, citation.entries[0].target_paper_id) == (
        "https://openalex.org/W3",
        "canonical-seed",
    )
    assert citation.next_cursor == "next-page"


def test_unpaywall_native_best_and_other_oa_locations() -> None:
    provider = create_builtin(
        "unpaywall", FixtureTransport({"unpaywall:resolve:first": fixture("unpaywall-native.json")})
    )
    candidates = provider.resolve(Paper("paper-1", "Fixture", doi="10.1000/unpaywall.native"), AccessPolicy("research"))

    assert [candidate.url for candidate in candidates] == [
        "https://example.test/article.pdf",
        "https://repository.test/item/1",
    ]
    assert candidates[0].publication_version is PublicationVersion.PUBLISHED
    assert candidates[0].access_basis is AccessBasis.OPEN_LICENSE
    assert candidates[1].candidate_id == "repo-1"
    assert candidates[1].publication_version is PublicationVersion.ACCEPTED_MANUSCRIPT
    assert candidates[1].access_basis is AccessBasis.PUBLIC_READ_ONLY


def test_builtin_manifest_uses_install_catalog_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from paper_agent import manifests

    catalog = tmp_path / "share" / "paper-agent"
    (catalog / "providers").mkdir(parents=True)
    source = Path(__file__).parents[1] / "providers" / "crossref.yaml"
    (catalog / "providers" / "crossref.yaml").write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(manifests, "manifest_directory", lambda override=None: catalog)

    assert load_builtin_manifest("crossref").provider == "crossref"


def test_role_and_capability_gates_are_manifest_driven() -> None:
    pmlr = PMLRAdapter("pmlr", FixtureTransport({}))
    spec = QuerySpec(1, "rq", "test")
    with pytest.raises(ValueError, match="role search"):
        pmlr.search(spec)
    unpaywall = create_builtin("unpaywall", FixtureTransport({}))
    with pytest.raises(ValueError, match="role citation"):
        unpaywall.references(Paper("p", "paper"))
    assert ProviderRole.SEARCH in ArXivProvider("arxiv", FixtureTransport({})).manifest.roles


@pytest.mark.parametrize("provider,klass", [("semantic_scholar", SemanticScholarProvider), ("openalex", OpenAlexProvider)])
def test_citation_direction_uses_provider_contract(provider: str, klass: type) -> None:
    transport = FixtureTransport(
        {
            f"{provider}:references:first": {"entries": [{"id": "backward", "observed_at": "2025-01-01"}]},
            f"{provider}:citations:first": {"entries": [{"id": "forward", "observed_at": "2025-01-01"}]},
        }
    )
    instance = klass(provider, transport)
    seed = Paper("paper-1", "Seed")
    backward = instance.references(seed).entries[0]
    forward = instance.citations(seed).entries[0]
    assert (backward.source_paper_id, backward.target_paper_id, backward.edge_type) == (
        "paper-1",
        "backward",
        CitationEdgeType.REFERENCES,
    )
    assert (forward.source_paper_id, forward.target_paper_id, forward.edge_type) == (
        "forward",
        "paper-1",
        CitationEdgeType.CITATIONS,
    )


def test_metadata_can_succeed_when_fixture_has_no_pdf() -> None:
    batch = create_builtin("arxiv", FixtureTransport({"arxiv:search:first": fixture("official-page-1.json")})).search(QuerySpec(1, "rq", "fixture"))
    assert batch.entries[0].title == "A Fixture Paper"
    assert "pdf" not in batch.entries[0].metadata


@pytest.mark.parametrize(
    "kind,value",
    [
        ("doi", "10.1000/fixture"),
        ("arxiv", "arXiv:2401.00001"),
        ("url", "https://example.test/paper"),
        ("bibtex", "@article{x, title={Bib Fixture}, author={Ada Lovelace}, year={2025}, doi={10.1000/bib}}"),
        ("ris", "TY  - JOUR\nTI  - RIS Fixture\nAU  - Ada Lovelace\nPY  - 2025\nDO  - 10.1000/ris\nER  -"),
        ("csl-json", '{"id":"csl", "title":"CSL Fixture", "DOI":"10.1000/csl", "issued":{"date-parts":[[2025]]}}'),
        ("local_pdf", "/tmp/fixture-paper.pdf"),
    ],
)
def test_library_seed_import_supports_every_authorized_seed(kind: str, value: str) -> None:
    entry = create_builtin("user_library", FixtureTransport({})).import_seeds([SeedInput(kind, value)]).entries[0]
    assert entry.external_id
