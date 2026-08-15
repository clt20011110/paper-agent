"""Offline contract coverage for the minimal Stage 1 enrichment boundary."""

from dataclasses import FrozenInstanceError

import pytest

from paper_agent_next.adapters.base import CollectedPaper, CollectionResult
from paper_agent_next.catalog import load_venue_spec
from paper_agent_next.collector import collect_venue_year
from paper_agent_next.enrichers.base import EnrichmentPatch, FrozenPaper
from paper_agent_next.errors import ContractError, EnrichmentError
from paper_agent_next.models import MissingField, Pagination, SourceIdentity
from paper_agent_next.http import PrefixResponse


def _identity(source_id: str = "paper-1") -> SourceIdentity:
    return SourceIdentity("dac", 2024, "primary", source_id)


def _view(
    source_id: str = "paper-1",
    *,
    abstract: str | None = None,
    doi: str | None = "10.1234/example.1",
    pdf_candidates: tuple[str, ...] = (),
) -> FrozenPaper:
    return FrozenPaper(
        identity=_identity(source_id),
        title="A paper title",
        authors=("Ada Lovelace",),
        abstract=abstract,
        doi=doi,
        landing_url=f"https://example.test/{source_id}",
        pdf_candidates=pdf_candidates,
    )


def test_frozen_view_and_patch_reject_mutable_or_empty_boundary_values() -> None:
    with pytest.raises(ContractError):
        FrozenPaper(
            identity=_identity(),
            title="A paper title",
            authors=["Ada Lovelace"],  # type: ignore[arg-type]
            abstract=None,
            doi=None,
            landing_url="https://example.test/paper-1",
            pdf_candidates=(),
        )
    with pytest.raises(ContractError):
        EnrichmentPatch(_identity())
    with pytest.raises(ContractError):
        EnrichmentPatch(_identity(), pdf_candidates=["https://example.test/paper.pdf"])  # type: ignore[arg-type]

    view = _view()
    patch = EnrichmentPatch(_identity(), abstract="An abstract.")
    assert not hasattr(view, "__dict__")
    assert not hasattr(patch, "__dict__")
    with pytest.raises(FrozenInstanceError):
        view.abstract = "changed"  # type: ignore[misc]


class _JsonClient:
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, object]] = []

    def post_json(self, url: str, payload: object) -> object:
        self.calls.append((url, payload))
        return self.responses.pop(0)


def _s2_result(doi: str, *, abstract: str | None = "An abstract.", url: str | None = None) -> dict[str, object]:
    return {
        "abstract": abstract,
        "externalIds": {"DOI": doi},
        "openAccessPdf": None if url is None else {"url": url},
    }


def test_semantic_scholar_batches_at_fixed_500_and_matches_doi_not_position() -> None:
    papers = tuple(
        _view(f"paper-{index}", doi=f"10.1234/example.{index}")
        for index in range(501)
    )
    first = list(reversed([
        _s2_result(f"10.1234/example.{index}", abstract=f"Abstract {index}")
        for index in range(500)
    ]))
    second = [_s2_result("10.1234/example.500", abstract="Abstract 500")]
    client = _JsonClient([first, second])

    from paper_agent_next.enrichers.semantic_scholar import SemanticScholarEnricher

    patches = SemanticScholarEnricher().enrich(papers, client)

    assert len(client.calls) == 2
    assert client.calls[0][0] == (
        "https://api.semanticscholar.org/graph/v1/paper/batch"
        "?fields=abstract,externalIds,openAccessPdf"
    )
    first_ids = client.calls[0][1]["ids"]  # type: ignore[index]
    second_ids = client.calls[1][1]["ids"]  # type: ignore[index]
    assert len(first_ids) == 500
    assert first_ids[0] == "DOI:10.1234/example.0"
    assert first_ids[-1] == "DOI:10.1234/example.499"
    assert second_ids == ["DOI:10.1234/example.500"]
    assert len(patches) == 501
    first_patch = next(patch for patch in patches if patch.identity.source_id == "paper-0")
    assert first_patch.identity == _identity("paper-0")
    assert first_patch.abstract == "Abstract 0"


@pytest.mark.parametrize(
    "response_doi",
    ["10.1234/EXAMPLE.1", "DOI:10.1234/EXAMPLE.1"],
    ids=["case-insensitive", "doi-prefix"],
)
def test_semantic_scholar_normalizes_response_doi_before_exact_binding(
    response_doi: str,
) -> None:
    from paper_agent_next.enrichers.semantic_scholar import SemanticScholarEnricher

    papers = (
        _view("paper-1", doi="10.1234/example.1"),
        _view("paper-2", doi="10.1234/example.2"),
    )
    response = [
        _s2_result("DOI:10.1234/EXAMPLE.2", abstract="Abstract 2"),
        _s2_result(response_doi, abstract="Abstract 1"),
    ]

    patches = SemanticScholarEnricher().enrich(papers, _JsonClient([response]))

    assert {
        patch.identity.source_id: patch.abstract
        for patch in patches
    } == {"paper-1": "Abstract 1", "paper-2": "Abstract 2"}


def test_semantic_scholar_null_no_result_and_no_doi_do_not_request() -> None:
    from paper_agent_next.enrichers.semantic_scholar import SemanticScholarEnricher

    client = _JsonClient([[None]])
    assert SemanticScholarEnricher().enrich((_view(doi=None),), client) == ()
    assert client.calls == []

    existing = _view(abstract="Already present")
    assert SemanticScholarEnricher().enrich((existing,), client) == ()
    assert client.calls == []

    assert SemanticScholarEnricher().enrich((_view(),), client) == ()
    assert len(client.calls) == 1


@pytest.mark.parametrize(
    "response",
    [
        {"not": "a list"},
        [],
        [{"externalIds": {"DOI": "10.1234/other"}}],
        [{"externalIds": {"DOI": "10.1234/example.1"}}, {"externalIds": {"DOI": "10.1234/example.1"}}],
        [{"externalIds": {"DOI": "10.1234/example.1"}, "abstract": 3}],
        [{"externalIds": {"DOI": "10.1234/example.1"}, "openAccessPdf": []}],
        [{"externalIds": {"DOI": "not-a-doi"}}],
    ],
    ids=[
        "wrong-root",
        "wrong-count",
        "doi-mismatch",
        "duplicate",
        "bad-abstract",
        "bad-pdf",
        "invalid-doi",
    ],
)
def test_semantic_scholar_bad_schema_is_typed_failure(response: object) -> None:
    from paper_agent_next.enrichers.semantic_scholar import SemanticScholarEnricher

    with pytest.raises(EnrichmentError):
        SemanticScholarEnricher().enrich((_view(),), _JsonClient([response]))


def test_semantic_scholar_strips_nonempty_open_access_pdf_url() -> None:
    from paper_agent_next.enrichers.semantic_scholar import SemanticScholarEnricher

    patches = SemanticScholarEnricher().enrich(
        (_view(),),
        _JsonClient(
            [
                [
                    _s2_result(
                        "10.1234/example.1",
                        abstract=None,
                        url="  https://example.test/paper.pdf  ",
                    )
                ]
            ]
        ),
    )

    assert patches[0].abstract is None
    assert patches[0].pdf_candidates == ("https://example.test/paper.pdf",)


class _Adapter:
    source_name = "primary"

    def __init__(self, paper: CollectedPaper) -> None:
        self.paper = paper

    def collect(self, venue_spec, year, http_client) -> CollectionResult:
        return CollectionResult(
            source_name=self.source_name,
            papers=(self.paper,),
            raw_items=1,
            excluded_non_papers=0,
            duplicate_occurrences=0,
            parse_rejects=(),
            pagination=Pagination(1, True, None),
        )


class _PrefixClient:
    def __init__(self, responses: dict[str, PrefixResponse]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    def get_prefix(self, url: str, max_bytes: int) -> PrefixResponse:
        self.calls.append(url)
        return self.responses.get(url, PrefixResponse("text/html", b"login"))


class _SemanticScholarClient(_PrefixClient):
    def __init__(self, batch: list[object], responses: dict[str, PrefixResponse]) -> None:
        super().__init__(responses)
        self.batch = batch

    def post_json(self, url: str, payload: object) -> object:
        return self.batch


class _PatchingEnricher:
    source_name = "enricher"

    def __init__(self, patch: EnrichmentPatch) -> None:
        self.patch = patch
        self.views: tuple[FrozenPaper, ...] | None = None
        self.access_calls_at_enrich: list[str] | None = None

    def enrich(self, papers, http_client):
        self.views = papers
        self.access_calls_at_enrich = list(http_client.calls)
        return (self.patch,)


def _collected(
    *,
    abstract: str | None,
    doi: str | None = None,
    pdf_candidates: tuple[str, ...] = ("https://primary.test/paper.pdf",),
) -> CollectedPaper:
    return CollectedPaper(
        source_id="paper-1",
        title="A paper title",
        authors=("Ada Lovelace",),
        abstract=abstract,
        doi=doi,
        landing_url="https://example.test/paper-1",
        pdf_candidates=pdf_candidates,
    )


def test_collector_freezes_identity_and_merges_candidates_with_actual_provenance() -> None:
    patch = EnrichmentPatch(
        _identity(),
        abstract="Enriched abstract.",
        pdf_candidates=("https://enricher.test/paper.pdf",),
    )
    enricher = _PatchingEnricher(patch)
    client = _PrefixClient(
        {
            "https://primary.test/paper.pdf": PrefixResponse("text/html", b"login"),
            "https://enricher.test/paper.pdf": PrefixResponse("application/pdf", b"%PDF-1.7"),
        }
    )

    outcome = collect_venue_year(
        load_venue_spec("dac"),
        2024,
        _Adapter(_collected(abstract=None)),
        client,  # type: ignore[arg-type]
        enrichers=(enricher,),
    )

    assert enricher.views is not None
    assert [paper.identity.as_tuple() for paper in enricher.views] == [
        ("dac", 2024, "primary", "paper-1")
    ]
    assert enricher.access_calls_at_enrich == []
    assert client.calls == [
        "https://primary.test/paper.pdf",
        "https://enricher.test/paper.pdf",
    ]
    assert outcome.run.complete is True
    assert outcome.papers[0].identity.as_tuple() == (
        "dac", 2024, "primary", "paper-1"
    )
    assert outcome.papers[0].abstract == "Enriched abstract."
    assert outcome.papers[0].field_sources.abstract == "enricher"
    assert outcome.papers[0].field_sources.pdf_url == "enricher"


def test_collector_never_overwrites_primary_abstract() -> None:
    patch = EnrichmentPatch(_identity(), abstract="Should not replace")
    enricher = _PatchingEnricher(patch)
    client = _PrefixClient({})

    outcome = collect_venue_year(
        load_venue_spec("dac"),
        2024,
        _Adapter(_collected(abstract="Primary abstract", doi="10.1234/example.1")),
        client,  # type: ignore[arg-type]
        enrichers=(enricher,),
    )

    assert outcome.papers[0].abstract == "Primary abstract"
    assert outcome.papers[0].field_sources.abstract == "primary"
    assert outcome.run.complete is True


def test_collector_discards_enriched_abstract_that_normalizes_to_title() -> None:
    patch = EnrichmentPatch(
        _identity(),
        abstract="<div> A paper   title </div>",
    )
    enricher = _PatchingEnricher(patch)
    client = _PrefixClient(
        {"https://primary.test/paper.pdf": PrefixResponse("application/pdf", b"%PDF-1.7")}
    )

    outcome = collect_venue_year(
        load_venue_spec("dac"),
        2024,
        _Adapter(_collected(abstract=None)),
        client,  # type: ignore[arg-type]
        enrichers=(enricher,),
    )

    assert outcome.papers == ()
    assert len(outcome.issues) == 1
    issue = outcome.issues[0]
    assert issue.source_name == "primary"
    assert issue.abstract is None
    assert issue.missing_fields == (MissingField.ABSTRACT,)
    assert issue.reason_codes == ("missing_abstract",)
    assert outcome.run.complete is False


def test_semantic_scholar_pdf_candidate_is_stripped_before_access_verification() -> None:
    from paper_agent_next.enrichers.semantic_scholar import SemanticScholarEnricher

    candidate = "https://enricher.test/paper.pdf"
    client = _SemanticScholarClient(
        [
            _s2_result(
                "10.1234/example.1",
                abstract="Enriched abstract",
                url=f"  {candidate}  ",
            )
        ],
        {candidate: PrefixResponse("application/pdf", b"%PDF-1.7")},
    )

    outcome = collect_venue_year(
        load_venue_spec("dac"),
        2024,
        _Adapter(_collected(abstract=None, doi="10.1234/example.1", pdf_candidates=())),
        client,  # type: ignore[arg-type]
        enrichers=(SemanticScholarEnricher(),),
    )

    assert outcome.run.complete is True
    assert outcome.papers[0].pdf_url == candidate
    assert client.calls == [candidate]


@pytest.mark.parametrize("kind", ["unknown", "duplicate", "failure"])
def test_invalid_or_failed_enrichment_preserves_membership_and_is_partial(kind: str) -> None:
    paper = _collected(abstract=None, doi="10.1234/example.1")
    identity = _identity()
    if kind == "unknown":
        patch = EnrichmentPatch(
            SourceIdentity("dac", 2024, "primary", "not-in-membership"),
            abstract="Enriched",
        )
        enricher = _PatchingEnricher(patch)
    elif kind == "duplicate":
        class DuplicateEnricher:
            source_name = "duplicate"

            def enrich(self, papers, http_client):
                return (
                    EnrichmentPatch(identity, abstract="One"),
                    EnrichmentPatch(identity, abstract="Two"),
                )

        enricher = DuplicateEnricher()
    else:
        class FailingEnricher:
            source_name = "failing"

            def enrich(self, papers, http_client):
                raise EnrichmentError("credential=secret")

        enricher = FailingEnricher()

    outcome = collect_venue_year(
        load_venue_spec("dac"),
        2024,
        _Adapter(paper),
        _PrefixClient({}),  # type: ignore[arg-type]
        enrichers=(enricher,),
    )

    assert outcome.papers == ()
    assert outcome.issues[0].source_id == "paper-1"
    assert outcome.run.membership_complete is True
    assert outcome.run.metadata_complete is False
    assert outcome.run.complete is False
    assert len(outcome.run.errors) == 1
    assert "secret" not in " ".join(outcome.run.errors)
