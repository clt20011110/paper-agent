"""In-memory composition of authoritative Stage 1 collection results."""

from dataclasses import dataclass

from .access import resolve_access
from .adapters.base import CorpusAdapter
from .catalog import VenueSpec
from .errors import CollectionError
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
    SourceTotalScope,
)
from .normalize import normalize_text

__all__ = ["CollectionOutcome", "collect_venue_year"]


@dataclass(frozen=True, slots=True)
class CollectionOutcome:
    papers: tuple[PaperRecord, ...]
    issues: tuple[IssueRecord, ...]
    run: RunRecord


def collect_venue_year(
    venue_spec: VenueSpec,
    year: int,
    adapter: CorpusAdapter,
    http_client: HttpClient,
) -> CollectionOutcome:
    """Collect one venue-year into contract records without publishing files."""
    if not venue_spec.is_applicable(year):
        counts = RunCounts(0, 0, 0, 0, 0, 0, 0, 0)
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
            counts=counts,
            pagination=None,
            warnings=(),
            errors=(),
        )
        return CollectionOutcome((), (), run)

    try:
        result = adapter.collect(venue_spec, year, http_client)
    except CollectionError:
        counts = RunCounts(0, 0, 0, 0, 0, 0, 0, 0)
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

    for paper in result.papers:
        title = normalize_text(paper.title)
        authors = tuple(
            author
            for raw_author in paper.authors
            if (author := normalize_text(raw_author)) is not None
        )
        abstract = normalize_text(paper.abstract)
        access = resolve_access(paper.pdf_candidates, None, http_client)

        if (
            title is not None
            and authors
            and abstract is not None
            and access.access_status is AccessStatus.DIRECT_PDF
        ):
            papers.append(
                PaperRecord(
                    venue_id=venue_spec.venue_id,
                    venue_name=venue_spec.name,
                    venue_type=venue_spec.venue_type,
                    year=year,
                    source_name=result.source_name,
                    source_id=paper.source_id,
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    doi=None,
                    landing_url=paper.landing_url,
                    pdf_url=access.pdf_url,
                    access_status=access.access_status,
                    field_sources=FieldSources(
                        title=result.source_name,
                        authors=result.source_name,
                        abstract=result.source_name,
                        doi=None,
                        landing_url=result.source_name,
                        pdf_url=result.source_name,
                    ),
                )
            )
            continue

        missing_fields: list[MissingField] = []
        reason_codes: list[str] = []
        if title is None:
            missing_fields.append(MissingField.TITLE)
            reason_codes.append("missing_title")
        if not authors:
            missing_fields.append(MissingField.AUTHORS)
            reason_codes.append("missing_authors")
        if abstract is None:
            missing_fields.append(MissingField.ABSTRACT)
            reason_codes.append("missing_abstract")
        if access.access_status is not AccessStatus.DIRECT_PDF:
            missing_fields.append(MissingField.ACCESS_LOCATOR)
            reason_codes.append(access.reason_code)
        issues.append(
            IssueRecord(
                issue_kind=IssueKind.INCOMPLETE_PAPER,
                venue_id=venue_spec.venue_id,
                year=year,
                source_name=result.source_name,
                source_id=paper.source_id,
                source_locator=paper.landing_url,
                title=title,
                authors=authors,
                abstract=abstract,
                doi=None,
                landing_url=paper.landing_url,
                missing_fields=tuple(missing_fields),
                reason_codes=tuple(reason_codes),
                message="required metadata or direct PDF access is missing",
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

    membership_complete = (
        result.pagination.terminal_reached
        and not result.parse_rejects
        and total_matches
        and zero_paper_proof
    )
    metadata_complete = counts.incomplete_papers == 0
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
