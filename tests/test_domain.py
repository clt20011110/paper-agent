from dataclasses import FrozenInstanceError

import pytest

from paper_agent.domain import (
    AccessBasis,
    AccessLocationCandidate,
    CitationBatch,
    CitationEdge,
    CitationEdgeType,
    EnvelopeStatus,
    FetchDecision,
    FetchDecisionStatus,
    FetchRequest,
    AnalysisInputKind,
    AnalysisStatus,
    Paper,
    PaperAnalysis,
    ProviderRole,
    SourceBatch,
    SourceEntry,
)


def test_paper_is_frozen_and_round_trips() -> None:
    paper = Paper(paper_id="paper-1", title="A Paper", authors=("Ada",))

    with pytest.raises(FrozenInstanceError):
        paper.title = "Changed"  # type: ignore[misc]

    assert Paper.from_dict(paper.to_dict()) == paper


def test_str_enums_have_the_frozen_wire_values() -> None:
    assert ProviderRole.VENUE_PRIMARY == "venue_primary"
    assert CitationEdgeType.REFERENCES == "references"
    assert FetchDecisionStatus.NEEDS_GRANT == "needs_grant"


def test_source_and_citation_envelopes_keep_required_audit_fields() -> None:
    entry = SourceEntry(provider="crossref", external_id="doi:1", title="Paper")
    source_batch = SourceBatch(
        source_run_id="source-run-1",
        query_hash="query-hash",
        entries=(entry,),
        next_cursor="page-2",
        status=EnvelopeStatus.PARTIAL,
        raw_response_artifact_hash="raw-hash",
    )
    citation_batch = CitationBatch(
        source_run_id="source-run-1",
        query_hash="query-hash",
        entries=(
            CitationEdge(
                source_paper_id="source",
                target_paper_id="target",
                edge_type=CitationEdgeType.REFERENCES,
                provider="openalex",
                observed_at="2026-08-09T00:00:00Z",
            ),
        ),
        next_cursor=None,
        status=EnvelopeStatus.SUCCESS,
    )

    assert SourceBatch.from_dict(source_batch.to_dict()) == source_batch
    assert CitationBatch.from_dict(citation_batch.to_dict()) == citation_batch


def test_fetch_decision_only_carries_request_after_allow() -> None:
    request = FetchRequest(
        request_id="request-1",
        candidate_id="candidate-1",
        policy_version="v1",
        purpose="internal_analysis",
        provider="official",
        created_at="2026-08-09T00:00:00Z",
        expires_at="2026-08-10T00:00:00Z",
        idempotency_key="download:1",
    )
    decision = FetchDecision(
        candidate_id="candidate-1",
        status=FetchDecisionStatus.ALLOW,
        reason_code="open_license",
        policy_version="v1",
        fetch_request=request,
    )

    assert FetchDecision.from_dict(decision.to_dict()) == decision
    assert AccessLocationCandidate(
        candidate_id="candidate-1",
        paper_id="paper-1",
        resolver="unpaywall",
        url="https://example.test/paper.pdf",
        access_basis=AccessBasis.OPEN_LICENSE,
    ).to_dict()["access_basis"] == "open_license"


def test_analysis_serializes_nested_evidence_envelope_fields() -> None:
    analysis = PaperAnalysis(
        analysis_run_id="analysis-1",
        paper_id="paper-1",
        status=AnalysisStatus.COMPLETED,
        input_kind=AnalysisInputKind.ABSTRACT_ONLY,
        input_artifact_hash=None,
        model="gpt-5.6-luna",
        prompt_hash="prompt",
        schema_hash="schema",
    )

    serialized = analysis.to_dict()
    assert serialized["model"] == "gpt-5.6-luna"
    assert PaperAnalysis.from_dict(serialized) == analysis
