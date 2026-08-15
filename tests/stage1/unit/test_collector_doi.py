"""Offline coverage for DOI enrichment patch application and provenance."""

from dataclasses import dataclass

from paper_agent.adapters.base import CollectedPaper, CollectionResult
from paper_agent.catalog import load_venue_spec
from paper_agent.collector import collect_venue_year
from paper_agent.enrichers.base import EnrichmentPatch, FrozenPaper
from paper_agent.models import IssueKind, Pagination, SourceIdentity


IDENTITY = SourceIdentity("dac", 2024, "primary", "paper-1")


@dataclass
class _Adapter:
    paper: CollectedPaper

    source_name = "primary"

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


class _Client:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def get_prefix(self, url: str, max_bytes: int):
        self.calls.append(url)
        raise AssertionError("DOI-only tests must not verify a PDF")


class _Enricher:
    source_name = "openalex"

    def __init__(self, patch: EnrichmentPatch) -> None:
        self.patch = patch
        self.views: tuple[FrozenPaper, ...] | None = None

    def enrich(self, papers, http_client):
        self.views = papers
        return (self.patch,)


def _paper(*, doi: str | None = None, abstract: str | None = "Primary abstract") -> CollectedPaper:
    return CollectedPaper(
        source_id="paper-1",
        title="A paper title",
        authors=("Ada Lovelace",),
        abstract=abstract,
        doi=doi,
        landing_url="https://example.test/paper-1",
        pdf_candidates=(),
    )


def _collect(paper: CollectedPaper, patch: EnrichmentPatch):
    return collect_venue_year(
        load_venue_spec("dac"),
        2024,
        _Adapter(paper),
        _Client(),
        enrichers=(_Enricher(patch),),
    )


def test_valid_new_doi_is_filled_with_enricher_provenance_and_completes_doi_only() -> None:
    outcome = _collect(
        _paper(doi=None, abstract=None),
        EnrichmentPatch(IDENTITY, abstract="Enriched abstract", doi="DOI:10.1234/NEW"),
    )

    assert len(outcome.papers) == 1
    record = outcome.papers[0]
    assert record.doi == "10.1234/new"
    assert record.field_sources.doi == "openalex"
    assert record.abstract == "Enriched abstract"
    assert record.field_sources.abstract == "openalex"
    assert record.access_status.value == "doi_only"
    assert outcome.issues == ()
    assert outcome.run.complete is True


def test_same_doi_does_not_change_primary_doi_provenance() -> None:
    outcome = _collect(
        _paper(doi="10.1234/same"),
        EnrichmentPatch(IDENTITY, doi="https://doi.org/10.1234/SAME"),
    )

    record = outcome.papers[0]
    assert record.doi == "10.1234/same"
    assert record.field_sources.doi == "primary"
    assert outcome.issues == ()
    assert outcome.run.complete is True


def test_conflicting_doi_retains_primary_paper_and_adds_blocking_field_issue() -> None:
    outcome = _collect(
        _paper(doi="10.1234/primary"),
        EnrichmentPatch(IDENTITY, doi="10.1234/other"),
    )

    assert len(outcome.papers) == 1
    assert outcome.papers[0].doi == "10.1234/primary"
    assert outcome.papers[0].field_sources.doi == "primary"
    assert len(outcome.issues) == 1
    issue = outcome.issues[0]
    assert issue.issue_kind is IssueKind.FIELD_CONFLICT
    assert issue.source_id == "paper-1"
    assert issue.missing_fields == ()
    assert issue.reason_codes == ("doi_conflict",)
    assert issue.message == (
        "enrichment DOI conflicts with primary DOI; primary DOI was retained"
    )
    assert outcome.run.counts.complete_papers == 1
    assert outcome.run.counts.incomplete_papers == 0
    assert outcome.run.metadata_complete is False
    assert outcome.run.complete is False


def test_invalid_patch_doi_cannot_become_body_doi() -> None:
    outcome = _collect(
        _paper(doi=None, abstract="Primary abstract"),
        EnrichmentPatch(IDENTITY, doi="not-a-doi"),
    )

    assert outcome.papers == ()
    assert outcome.issues[0].doi is None
    assert outcome.issues[0].reason_codes == (
        "no_verified_pdf_or_doi",
    )
