"""In-memory composition of authoritative Stage 1 collection results."""

from dataclasses import dataclass, replace

from .access import resolve_access
from .adapters.base import CorpusAdapter
from .catalog import VenueSpec
from .enrichers.base import EnrichmentPatch, FrozenPaper, MetadataEnricher
from .errors import CollectionError, ContractError, EnrichmentError
from .http import HttpClient
from .models import (
    AccessStatus,
    FieldSources,
    IssueKind,
    IssueRecord,
    MissingField,
    PaperRecord,
    RunCounts,
    RunRecord,
    RunStatus,
    SourceIdentity,
    SourceTotalScope,
)
from .normalize import normalize_doi, normalize_text

__all__ = [
    "CollectionOutcome",
    "not_applicable_outcome",
    "collect_venue_year",
]


@dataclass(frozen=True, slots=True)
class CollectionOutcome:
    papers: tuple[PaperRecord, ...]
    issues: tuple[IssueRecord, ...]
    run: RunRecord


def _empty_counts() -> RunCounts:
    return RunCounts(0, 0, 0, 0, 0, 0, 0, 0)


def not_applicable_outcome(
    venue_spec: VenueSpec, year: int
) -> CollectionOutcome | None:
    """Return the contract outcome for a catalog-excluded year, if applicable."""

    if venue_spec.is_applicable(year):
        return None
    run = RunRecord(
        status=RunStatus.NOT_APPLICABLE,
        venue_id=venue_spec.venue_id,
        venue_name=venue_spec.name,
        venue_type=venue_spec.venue_type,
        year=year,
        source_name=None,
        membership_complete=False,
        metadata_complete=False,
        complete=False,
        counts=_empty_counts(),
        pagination=None,
        warnings=(),
        errors=(),
    )
    return CollectionOutcome((), (), run)


def _initial_view(
    venue_spec: VenueSpec,
    year: int,
    source_name: str,
    paper,
) -> tuple[FrozenPaper, dict[str, str]]:
    title = normalize_text(paper.title)
    authors = tuple(
        author
        for raw_author in paper.authors
        if (author := normalize_text(raw_author)) is not None
    )
    abstract = normalize_text(paper.abstract)
    if title is not None and abstract is not None and title == abstract:
        abstract = None
    doi = normalize_doi(paper.doi)
    identity = SourceIdentity(venue_spec.venue_id, year, source_name, paper.source_id)

    ordered_candidates: list[str] = []
    candidate_sources: dict[str, str] = {}
    for candidate in paper.pdf_candidates:
        if candidate not in candidate_sources:
            ordered_candidates.append(candidate)
            candidate_sources[candidate] = source_name

    view = FrozenPaper(
        identity=identity,
        title=title,
        authors=authors,
        abstract=abstract,
        doi=doi,
        landing_url=paper.landing_url,
        pdf_candidates=tuple(ordered_candidates),
    )
    return view, candidate_sources


def _validate_patches(
    patches: object,
    identity_index: dict[SourceIdentity, int],
) -> tuple[tuple[EnrichmentPatch, ...], str | None]:
    if not isinstance(patches, tuple):
        raise ContractError("enrichment patches: must be a tuple")
    seen: set[SourceIdentity] = set()
    for index, patch in enumerate(patches):
        if not isinstance(patch, EnrichmentPatch):
            raise ContractError(f"enrichment patches[{index}]: must be an EnrichmentPatch")
        if patch.identity not in identity_index:
            return (), "returned unknown identity"
        if patch.identity in seen:
            return (), "returned duplicate identity"
        seen.add(patch.identity)
    return patches, None


def collect_venue_year(
    venue_spec: VenueSpec,
    year: int,
    adapter: CorpusAdapter,
    http_client: HttpClient,
    enrichers: tuple[MetadataEnricher, ...] = (),
) -> CollectionOutcome:
    """Collect one venue-year into contract records without publishing files."""
    not_applicable = not_applicable_outcome(venue_spec, year)
    if not_applicable is not None:
        return not_applicable
    if not isinstance(enrichers, tuple):
        raise ContractError("enrichers: must be a tuple")

    try:
        result = adapter.collect(venue_spec, year, http_client)
    except CollectionError:
        counts = _empty_counts()
        run = RunRecord(
            status=RunStatus.FAILED,
            venue_id=venue_spec.venue_id,
            venue_name=venue_spec.name,
            venue_type=venue_spec.venue_type,
            year=year,
            source_name=adapter.source_name,
            membership_complete=False,
            metadata_complete=False,
            complete=False,
            counts=counts,
            pagination=None,
            warnings=(),
            errors=("authoritative membership collection failed",),
        )
        return CollectionOutcome((), (), run)

    papers: list[PaperRecord] = []
    issues: list[IssueRecord] = []

    for reject in result.parse_rejects:
        issue_kind = (
            IssueKind.IDENTITY_CONFLICT
            if reject.reason_code == "identity_conflict"
            else IssueKind.PARSE_REJECT
        )
        issues.append(
            IssueRecord(
                issue_kind=issue_kind,
                venue_id=venue_spec.venue_id,
                year=year,
                source_name=result.source_name,
                source_id=None,
                source_locator=reject.source_locator,
                title=None,
                authors=(),
                abstract=None,
                doi=None,
                landing_url=None,
                missing_fields=(),
                reason_codes=(reject.reason_code,),
                message=reject.message,
            )
        )

    views: list[FrozenPaper] = []
    candidate_sources: list[dict[str, str]] = []
    abstract_sources: list[str] = []
    for paper in result.papers:
        view, sources = _initial_view(venue_spec, year, result.source_name, paper)
        views.append(view)
        candidate_sources.append(sources)
        abstract_sources.append(result.source_name)

    identity_index = {view.identity: index for index, view in enumerate(views)}
    enrichment_errors: list[str] = []
    for enricher in enrichers:
        source_name = getattr(enricher, "source_name", None)
        enrich = getattr(enricher, "enrich", None)
        if (
            not isinstance(source_name, str)
            or not source_name
            or source_name != source_name.strip()
            or any(character.isspace() for character in source_name)
            or not callable(enrich)
        ):
            raise ContractError("enricher: must provide source_name and callable enrich")

        try:
            raw_patches = enrich(tuple(views), http_client)
        except EnrichmentError:
            enrichment_errors.append(f"enrichment {source_name} failed")
            continue

        patches_for_enricher, validation_error = _validate_patches(
            raw_patches, identity_index
        )
        if validation_error is not None:
            enrichment_errors.append(f"enrichment {source_name} {validation_error}")
            continue

        for patch in patches_for_enricher:
            paper_index = identity_index[patch.identity]
            view = views[paper_index]
            if patch.abstract is not None and view.abstract is None:
                abstract = normalize_text(patch.abstract)
                if abstract is not None and abstract != normalize_text(view.title):
                    view = replace(view, abstract=abstract)
                    abstract_sources[paper_index] = source_name

            ordered_candidates = list(view.pdf_candidates)
            sources = candidate_sources[paper_index]
            for candidate in patch.pdf_candidates:
                if candidate in sources:
                    continue
                ordered_candidates.append(candidate)
                sources[candidate] = source_name
            if tuple(ordered_candidates) != view.pdf_candidates:
                view = replace(view, pdf_candidates=tuple(ordered_candidates))
            views[paper_index] = view

    for paper_index, view in enumerate(views):
        access = resolve_access(view.pdf_candidates, view.doi, http_client)
        if (
            view.title is not None
            and view.authors
            and view.abstract is not None
            and access.access_status in {AccessStatus.DIRECT_PDF, AccessStatus.DOI_ONLY}
        ):
            pdf_source = (
                candidate_sources[paper_index].get(access.pdf_url)
                if access.pdf_url is not None
                else None
            )
            papers.append(
                PaperRecord(
                    venue_id=venue_spec.venue_id,
                    venue_name=venue_spec.name,
                    venue_type=venue_spec.venue_type,
                    year=year,
                    source_name=result.source_name,
                    source_id=view.identity.source_id,
                    title=view.title,
                    authors=view.authors,
                    abstract=view.abstract,
                    doi=view.doi,
                    landing_url=view.landing_url,
                    pdf_url=access.pdf_url,
                    access_status=access.access_status,
                    field_sources=FieldSources(
                        title=result.source_name,
                        authors=result.source_name,
                        abstract=abstract_sources[paper_index],
                        doi=result.source_name if view.doi is not None else None,
                        landing_url=result.source_name if view.landing_url is not None else None,
                        pdf_url=pdf_source,
                    ),
                )
            )
            continue

        missing_fields: list[MissingField] = []
        reason_codes: list[str] = []
        if view.title is None:
            missing_fields.append(MissingField.TITLE)
            reason_codes.append("missing_title")
        if not view.authors:
            missing_fields.append(MissingField.AUTHORS)
            reason_codes.append("missing_authors")
        if view.abstract is None:
            missing_fields.append(MissingField.ABSTRACT)
            reason_codes.append("missing_abstract")
        if access.access_status not in {AccessStatus.DIRECT_PDF, AccessStatus.DOI_ONLY}:
            missing_fields.append(MissingField.ACCESS_LOCATOR)
            reason_codes.append(access.reason_code)
        issues.append(
            IssueRecord(
                issue_kind=IssueKind.INCOMPLETE_PAPER,
                venue_id=venue_spec.venue_id,
                year=year,
                source_name=result.source_name,
                source_id=view.identity.source_id,
                source_locator=view.landing_url,
                title=view.title,
                authors=view.authors,
                abstract=view.abstract,
                doi=view.doi,
                landing_url=view.landing_url,
                missing_fields=tuple(missing_fields),
                reason_codes=tuple(reason_codes),
                message="required metadata or access locator is missing",
            )
        )

    counts = RunCounts(
        raw_items=result.raw_items,
        included_papers=len(result.papers),
        complete_papers=len(papers),
        incomplete_papers=len(result.papers) - len(papers),
        excluded_non_papers=result.excluded_non_papers,
        duplicate_occurrences=result.duplicate_occurrences,
        parse_rejects=len(result.parse_rejects),
        issue_records=len(issues),
    )
    source_total = result.pagination.source_total
    if source_total is None:
        total_matches = True
    else:
        expected_total = (
            counts.raw_items
            if source_total.scope is SourceTotalScope.RAW_ITEMS
            else counts.included_papers
        )
        total_matches = source_total.value == expected_total
    zero_paper_proof = (
        counts.included_papers != 0
        or (source_total is not None and total_matches and source_total.value == 0)
    )

    errors: list[str] = []
    if not result.pagination.terminal_reached:
        errors.append("authoritative pagination did not reach a terminal state")
    if not total_matches:
        errors.append("source total does not match collected counts")
    if counts.included_papers == 0 and not zero_paper_proof:
        errors.append("applicable venue-year has no authoritative zero-paper proof")
    errors.extend(enrichment_errors)

    membership_complete = (
        result.pagination.terminal_reached
        and not result.parse_rejects
        and total_matches
        and zero_paper_proof
    )
    metadata_complete = counts.incomplete_papers == 0 and not enrichment_errors
    complete = membership_complete and metadata_complete
    run = RunRecord(
        status=RunStatus.COMPLETE if complete else RunStatus.PARTIAL,
        venue_id=venue_spec.venue_id,
        venue_name=venue_spec.name,
        venue_type=venue_spec.venue_type,
        year=year,
        source_name=result.source_name,
        membership_complete=membership_complete,
        metadata_complete=metadata_complete,
        complete=complete,
        counts=counts,
        pagination=result.pagination,
        warnings=(),
        errors=tuple(errors),
    )
    return CollectionOutcome(tuple(papers), tuple(issues), run)
