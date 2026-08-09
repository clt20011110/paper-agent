"""Typed, serializable values shared by every Paper Agent stage.

These objects deliberately contain data only.  Repositories and services own
validation that needs database or policy context.
"""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field, fields
from enum import StrEnum
from types import UnionType
from typing import Any, ClassVar, Mapping, Self, TypeVar, Union, get_args, get_origin, get_type_hints


class ProviderRole(StrEnum):
    VENUE_PRIMARY = "venue_primary"
    SEARCH = "search"
    CITATION = "citation"
    LIBRARY = "library"
    METADATA_ENRICHER = "metadata_enricher"
    METADATA_VERIFIER = "metadata_verifier"
    OA_RESOLVER = "oa_resolver"
    DOWNLOAD = "download"


class ProviderCapability(StrEnum):
    STABLE_ID = "stable_id"
    METADATA = "metadata"
    ABSTRACT = "abstract"
    DATE_FILTER = "date_filter"
    REFERENCES = "references"
    CITATIONS = "citations"
    OA_LOCATIONS = "oa_locations"
    FULL_TEXT = "full_text"
    SUPPLEMENT = "supplement"
    BULK_SNAPSHOT = "bulk_snapshot"


class VerificationStatus(StrEnum):
    VERIFIED = "verified"
    SINGLE_SOURCE = "single_source"
    UNVERIFIED = "unverified"
    CONFLICTED = "conflicted"


class MembershipStatus(StrEnum):
    OFFICIAL_CONFIRMED = "official_confirmed"
    VENUE_CANDIDATE = "venue_candidate"
    NOT_MEMBER = "not_member"
    CONFLICTED = "conflicted"


class PublicationVersion(StrEnum):
    PREPRINT = "preprint"
    ACCEPTED_MANUSCRIPT = "accepted_manuscript"
    PUBLISHED = "published"
    UNKNOWN = "unknown"


class AccessBasis(StrEnum):
    OPEN_LICENSE = "open_license"
    PUBLIC_READ_ONLY = "public_read_only"
    USER_SUBSCRIPTION = "user_subscription"
    USER_SUPPLIED = "user_supplied"
    UNKNOWN = "unknown"


class PlanStatus(StrEnum):
    DRAFT = "draft"
    APPROVED = "approved"
    SUPERSEDED = "superseded"


class EnvelopeStatus(StrEnum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


class CitationEdgeType(StrEnum):
    REFERENCES = "references"
    CITATIONS = "citations"
    VERSION_OF = "version_of"


class FetchDecisionStatus(StrEnum):
    ALLOW = "allow"
    NEEDS_GRANT = "needs_grant"
    MANUAL = "manual"
    DENY = "deny"


class DownloadStatus(StrEnum):
    PENDING = "pending"
    DOWNLOADED = "downloaded"
    NOT_AVAILABLE = "not_available"
    AUTH_REQUIRED = "auth_required"
    MANUAL_REQUIRED = "manual_required"
    FAILED_RETRYABLE = "failed_retryable"
    FAILED_TERMINAL = "failed_terminal"


class GrantKind(StrEnum):
    DOWNLOAD = "download"
    BROWSER_DATA_SHARING = "browser_data_sharing"
    REMOTE_MODEL_PROCESSING = "remote_model_processing"


class GrantMode(StrEnum):
    ATTENDED = "attended"
    UNATTENDED = "unattended"


class FilterStatus(StrEnum):
    RELEVANT = "relevant"
    IRRELEVANT = "irrelevant"
    NEEDS_REVIEW = "needs_review"


class AnalysisInputKind(StrEnum):
    FULL_PDF = "full_pdf"
    ABSTRACT_ONLY = "abstract_only"


class AnalysisStatus(StrEnum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    ANALYSIS_NOT_AUTHORIZED = "analysis_not_authorized"
    MANUAL_REQUIRED = "manual_required"


class StudySetting(StrEnum):
    REAL = "real"
    SIMULATION = "simulation"
    THEORY = "theory"
    OTHER = "other"


class EvidenceDirection(StrEnum):
    SUPPORT = "support"
    CONTRADICT = "contradict"
    NEUTRAL = "neutral"


class ComparisonEligibility(StrEnum):
    COMPARABLE = "comparable"
    NOT_COMPARABLE = "not_comparable"


class EvidenceLevel(StrEnum):
    FULL_TEXT_DIRECT = "full_text_direct"
    FULL_TEXT_INFERRED = "full_text_inferred"
    ABSTRACT_DIRECT = "abstract_direct"
    METADATA_ONLY = "metadata_only"
    CORPUS_STAT = "corpus_stat"


class ClaimType(StrEnum):
    FINDING = "finding"
    TREND = "trend"
    COMPARISON = "comparison"
    GAP = "gap"
    RECOMMENDATION = "recommendation"
    CORPUS_STAT = "corpus_stat"


class ClaimStatus(StrEnum):
    SUPPORTED = "supported"
    MIXED = "mixed"
    INSUFFICIENT = "insufficient"


class ClaimRelationType(StrEnum):
    SAME = "same"
    REFINED = "refined"
    SPLIT = "split"
    MERGED = "merged"
    SUPERSEDED = "superseded"
    RETIRED = "retired"


class ReportBlockKind(StrEnum):
    PROSE = "prose"
    LIST_ITEM = "list_item"
    TABLE_CELL = "table_cell"
    CAPTION = "caption"


class AuditSeverity(StrEnum):
    BLOCKER = "blocker"
    MAJOR = "major"
    MINOR = "minor"
    NOTE = "note"


class TaskState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED_RETRYABLE = "failed_retryable"
    FAILED_TERMINAL = "failed_terminal"
    MANUAL_REQUIRED = "manual_required"


T = TypeVar("T", bound="Model")


def _encode(value: Any) -> Any:
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, Model):
        return value.to_dict()
    if isinstance(value, tuple):
        return [_encode(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _encode(item) for key, item in value.items()}
    return value


def _decode(value: Any, annotation: Any) -> Any:
    if value is None:
        return None
    origin = get_origin(annotation)
    if origin in (Union, UnionType):
        choices = [choice for choice in get_args(annotation) if choice is not type(None)]
        return _decode(value, choices[0]) if choices else value
    if origin is tuple:
        item_types = get_args(annotation)
        item_type = item_types[0] if item_types else Any
        return tuple(_decode(item, item_type) for item in value)
    if origin in (dict, Mapping, MappingABC):
        item_types = get_args(annotation)
        value_type = item_types[1] if len(item_types) == 2 else Any
        return {key: _decode(item, value_type) for key, item in value.items()}
    if isinstance(annotation, type) and issubclass(annotation, StrEnum):
        return annotation(value)
    if isinstance(annotation, type) and issubclass(annotation, Model):
        return annotation.from_dict(value)
    return value


@dataclass(frozen=True, slots=True)
class Model:
    """Small JSON-friendly base class for frozen domain values."""

    _ignored_fields: ClassVar[frozenset[str]] = frozenset()

    def to_dict(self) -> dict[str, Any]:
        return {
            item.name: _encode(getattr(self, item.name))
            for item in fields(self)
            if item.name not in self._ignored_fields
        }

    @classmethod
    def from_dict(cls: type[T], value: Mapping[str, Any]) -> T:
        hints = get_type_hints(cls)
        return cls(
            **{
                item.name: _decode(value[item.name], hints[item.name])
                for item in fields(cls)
                if item.name in value
            }
        )


@dataclass(frozen=True, slots=True)
class Paper(Model):
    paper_id: str
    title: str
    abstract: str | None = None
    authors: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    publication_date: str | None = None
    year: int | None = None
    venue_id: str | None = None
    venue_name: str | None = None
    venue_type: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    canonical_url: str | None = None
    volume: str | None = None
    issue: str | None = None
    pages: str | None = None
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED
    created_at: str | None = None
    updated_at: str | None = None


@dataclass(frozen=True, slots=True)
class PaperSource(Model):
    source_id: str
    paper_id: str
    provider: str
    external_id: str
    landing_url: str | None = None
    pdf_url: str | None = None
    metadata_url: str | None = None
    bibtex: str | None = None
    citation_count: int | None = None
    citation_count_as_of: str | None = None
    publication_version: PublicationVersion = PublicationVersion.UNKNOWN
    license: str | None = None
    host_type: str | None = None
    access_basis: AccessBasis = AccessBasis.UNKNOWN
    raw_metadata: Mapping[str, Any] = field(default_factory=dict)
    first_seen_at: str | None = None
    last_seen_at: str | None = None
    source_updated_at: str | None = None
    metadata_capabilities: tuple[ProviderCapability, ...] = ()
    download_capabilities: tuple[ProviderCapability, ...] = ()


@dataclass(frozen=True, slots=True)
class CollectionMembership(Model):
    collection_id: str
    paper_id: str
    membership_status: MembershipStatus
    official_evidence: tuple[str, ...] = ()
    source_id: str | None = None
    observed_at: str | None = None


@dataclass(frozen=True, slots=True)
class Artifact(Model):
    artifact_id: str
    paper_id: str | None
    kind: str
    relative_path: str
    content_hash: str
    media_type: str | None = None
    size_bytes: int | None = None
    created_at: str | None = None
    source_artifact_hash: str | None = None


@dataclass(frozen=True, slots=True)
class QuerySpec(Model):
    schema_version: int
    research_question_id: str
    original_query: str
    synonym_groups: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    alias_group: str | None = None
    date_from: str | None = None
    date_to: str | None = None
    venue_ids: tuple[str, ...] = ()
    fields: tuple[str, ...] = ()
    languages: tuple[str, ...] = ()
    document_types: tuple[str, ...] = ()
    sort: str | None = None
    page_size: int | None = None
    budget: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ProviderResolution(Model):
    provider: str
    role: ProviderRole
    resolved: bool
    reason: str
    distribution_name: str | None = None
    distribution_version: str | None = None
    artifact_digest: str | None = None
    manifest_hash: str | None = None
    capabilities: tuple[ProviderCapability, ...] = ()
    api_mode: str | None = None
    snapshot_hash: str | None = None
    credentials_available: bool = False
    query_compiler_version: str | None = None
    native_query_hash: str | None = None


@dataclass(frozen=True, slots=True)
class SourceEntry(Model):
    provider: str
    external_id: str
    title: str
    authors: tuple[str, ...] = ()
    abstract: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    publication_date: str | None = None
    year: int | None = None
    venue_name: str | None = None
    landing_url: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SourceBatch(Model):
    source_run_id: str
    query_hash: str
    entries: tuple[SourceEntry, ...]
    next_cursor: str | None
    status: EnvelopeStatus
    error: str | None = None
    raw_response_artifact_hash: str | None = None


@dataclass(frozen=True, slots=True)
class CitationEdge(Model):
    source_paper_id: str
    target_paper_id: str
    edge_type: CitationEdgeType
    provider: str
    observed_at: str
    raw_evidence: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CitationBatch(Model):
    source_run_id: str
    query_hash: str
    entries: tuple[CitationEdge, ...]
    next_cursor: str | None
    status: EnvelopeStatus
    error: str | None = None
    raw_response_artifact_hash: str | None = None


@dataclass(frozen=True, slots=True)
class AccessLocationCandidate(Model):
    candidate_id: str
    paper_id: str
    resolver: str
    url: str
    landing_url: str | None = None
    host: str | None = None
    publication_version: PublicationVersion = PublicationVersion.UNKNOWN
    license: str | None = None
    access_basis: AccessBasis = AccessBasis.UNKNOWN
    retrieved_at: str | None = None
    raw_evidence_hash: str | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FetchRequest(Model):
    request_id: str
    candidate_id: str
    policy_version: str
    purpose: str
    provider: str
    created_at: str
    expires_at: str
    idempotency_key: str
    authorization_grant_id: str | None = None
    authorization_grant_hash: str | None = None
    fencing_token: int | None = None


@dataclass(frozen=True, slots=True)
class FetchDecision(Model):
    candidate_id: str
    status: FetchDecisionStatus
    reason_code: str
    policy_version: str
    fetch_request: FetchRequest | None = None
    authorization_grant_id: str | None = None

    def __post_init__(self) -> None:
        if self.status is FetchDecisionStatus.ALLOW and self.fetch_request is None:
            raise ValueError("allow decisions require a fetch request")
        if self.status is not FetchDecisionStatus.ALLOW and self.fetch_request is not None:
            raise ValueError("only allow decisions may carry a fetch request")


@dataclass(frozen=True, slots=True)
class DownloadResult(Model):
    request_id: str
    paper_id: str
    status: DownloadStatus
    provider: str
    artifact_id: str | None = None
    content_hash: str | None = None
    source_url: str | None = None
    downloaded_at: str | None = None
    error_code: str | None = None
    authorization_grant_id: str | None = None


@dataclass(frozen=True, slots=True)
class AuthorizationGrant(Model):
    grant_id: str
    kind: GrantKind
    actions: tuple[str, ...]
    purpose: str
    paper_ids: tuple[str, ...] = ()
    collection_ids: tuple[str, ...] = ()
    selection_snapshot_hash: str | None = None
    domain_allowlist: tuple[str, ...] = ()
    artifact_hashes: tuple[str, ...] = ()
    provider: str | None = None
    model: str | None = None
    max_papers: int | None = None
    mode: GrantMode = GrantMode.ATTENDED
    allow_unattended: bool = False
    skill_digest: str | None = None
    dependency_digest: str | None = None
    approved_hash: str | None = None
    approved_by: str | None = None
    approved_at: str | None = None
    expires_at: str | None = None
    revoked_at: str | None = None


@dataclass(frozen=True, slots=True)
class FilterDecision(Model):
    run_id: str
    paper_id: str
    status: FilterStatus
    reason: str
    rule_version: str | None = None
    reranker_score: float | None = None
    adjudicator_score: float | None = None
    threshold_version: str | None = None
    input_hash: str | None = None
    model_id: str | None = None
    model_revision: str | None = None
    created_at: str | None = None


@dataclass(frozen=True, slots=True)
class EvidenceUnit(Model):
    claim: str
    direction: EvidenceDirection
    task_id: str | None = None
    dataset_id: str | None = None
    dataset_version: str | None = None
    split_id: str | None = None
    metric_id: str | None = None
    metric_definition_hash: str | None = None
    unit: str | None = None
    optimization_direction: str | None = None
    value: str | None = None
    uncertainty: str | None = None
    statistical_method: str | None = None
    protocol_id: str | None = None
    protocol_hash: str | None = None
    sample_size: int | None = None
    baseline_id: str | None = None
    baseline_version: str | None = None
    conditions: Mapping[str, Any] = field(default_factory=dict)
    locator: str | None = None
    source_value: str | None = None
    normalization_method: str | None = None
    normalizer_version: str | None = None


@dataclass(frozen=True, slots=True)
class PaperAnalysis(Model):
    analysis_run_id: str
    paper_id: str
    status: AnalysisStatus
    input_kind: AnalysisInputKind
    input_artifact_hash: str | None
    model: str
    prompt_hash: str
    schema_hash: str
    research_question: str | None = None
    motivation: str | None = None
    methods: str | None = None
    experimental_setup: str | None = None
    main_results: str | None = None
    limitations: str | None = None
    open_resources: str | None = None
    topic_relation: str | None = None
    subquestion: tuple[str, ...] = ()
    theme: tuple[str, ...] = ()
    method_family: tuple[str, ...] = ()
    task: tuple[str, ...] = ()
    dataset: tuple[str, ...] = ()
    benchmark: tuple[str, ...] = ()
    evidence_type: tuple[str, ...] = ()
    publication_status: str | None = None
    study_setting: StudySetting = StudySetting.OTHER
    evidence_units: tuple[EvidenceUnit, ...] = ()
    comparison_eligibility: ComparisonEligibility = ComparisonEligibility.NOT_COMPARABLE
    missing_fields: tuple[str, ...] = ()
    processing_grant_id: str | None = None
    created_at: str | None = None


@dataclass(frozen=True, slots=True)
class ClaimEvidence(Model):
    evidence_kind: str
    paper_id: str | None = None
    analysis_run_id: str | None = None
    locator: str | None = None
    evidence_unit: EvidenceUnit | None = None
    conditions: Mapping[str, Any] = field(default_factory=dict)
    search_plan_id: str | None = None
    source_run_id: str | None = None
    query_hash: str | None = None
    statistic: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ReportClaim(Model):
    claim_id: str
    claim_key: Mapping[str, str]
    research_question_id: str
    report_section: str
    claim_text: str
    claim_type: ClaimType
    supporting_evidence: tuple[ClaimEvidence, ...] = ()
    contradicting_evidence: tuple[ClaimEvidence, ...] = ()
    evidence_level: EvidenceLevel = EvidenceLevel.METADATA_ONLY
    comparison_group_id: str | None = None
    confidence: str | None = None
    known_limitations: tuple[str, ...] = ()
    status: ClaimStatus = ClaimStatus.INSUFFICIENT
