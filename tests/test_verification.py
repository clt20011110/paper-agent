from __future__ import annotations

from paper_agent.domain import (
    EnvelopeStatus,
    MembershipStatus,
    ProviderRole,
    SourceBatch,
    SourceEntry,
    VerificationStatus,
)
from paper_agent.repository import PaperRepository
from paper_agent.storage import Database
from paper_agent.verification import (
    MetadataCoordinator,
    MetadataVerification,
    ProviderTrust,
    VenueContext,
    providers_are_independent,
)


def trust(
    provider: str,
    *,
    roles: frozenset[ProviderRole] = frozenset({ProviderRole.METADATA_VERIFIER}),
    group: str | None = None,
    upstreams: frozenset[str] | None = None,
    authority_rank: int = 3,
) -> ProviderTrust:
    return ProviderTrust(provider, roles, group or provider, upstreams or frozenset({provider}), authority_rank)


def entry(provider: str, **metadata) -> SourceEntry:
    return SourceEntry(
        provider=provider,
        external_id=f"{provider}-1",
        title="A Verified Paper",
        authors=("Ada Lovelace",),
        year=2025,
        doi="10.1000/verified",
        venue_name="Example Venue",
        metadata=metadata,
    )


def batch(source: SourceEntry) -> SourceBatch:
    return SourceBatch(
        source_run_id=f"run-{source.provider}",
        query_hash=f"query-{source.provider}",
        entries=(source,),
        next_cursor=None,
        status=EnvelopeStatus.SUCCESS,
    )


def test_shared_upstream_cannot_supply_two_independent_votes() -> None:
    semantic = trust("semantic_scholar", upstreams=frozenset({"crossref"}))
    graph = trust("another_graph", upstreams=frozenset({"crossref"}))
    verifier = MetadataVerification({item.provider: item for item in (semantic, graph)})

    assert not providers_are_independent(semantic, graph)
    assert verifier.verify((entry("semantic_scholar"), entry("another_graph"))).status is VerificationStatus.SINGLE_SOURCE


def test_two_independent_sources_or_official_record_verify_identity() -> None:
    first = trust("crossref", upstreams=frozenset({"crossref"}))
    second = trust("openalex", upstreams=frozenset({"openalex"}))
    official = trust(
        "pmlr",
        roles=frozenset({ProviderRole.VENUE_PRIMARY}),
        upstreams=frozenset({"pmlr"}),
    )
    verifier = MetadataVerification({item.provider: item for item in (first, second, official)})

    independent = verifier.verify((entry("crossref"), entry("openalex")))
    authoritative = verifier.verify((entry("pmlr", official_membership=True),))

    assert independent.status is VerificationStatus.VERIFIED
    assert independent.supporting_providers == ("crossref", "openalex")
    assert authoritative.status is VerificationStatus.VERIFIED


def test_core_metadata_conflicts_are_not_silently_resolved() -> None:
    trusts = {name: trust(name) for name in ("crossref", "openalex")}
    conflicting = SourceEntry(
        provider="openalex",
        external_id="openalex-1",
        title="A Different Paper",
        authors=("Grace Hopper",),
        year=2024,
        doi="10.1000/verified",
    )

    result = MetadataVerification(trusts).verify((entry("crossref"), conflicting))

    assert result.status is VerificationStatus.CONFLICTED
    assert result.conflict_fields == ("title", "first_author", "year")


def test_coordinator_keeps_fallback_as_candidate_then_promotes_exact_primary(tmp_path) -> None:
    path = tmp_path / "papers.sqlite3"
    with Database(path) as database:
        database.migrate()
        repository = PaperRepository(database)
        trusts = {
            "crossref": trust("crossref", upstreams=frozenset({"crossref"}), authority_rank=1),
            "openalex": trust("openalex", upstreams=frozenset({"openalex"}), authority_rank=2),
            "pmlr": trust(
                "pmlr",
                roles=frozenset({ProviderRole.VENUE_PRIMARY}),
                upstreams=frozenset({"pmlr"}),
                authority_rank=0,
            ),
        }
        coordinator = MetadataCoordinator(repository, trusts)
        venue = VenueContext("venue-icml", "icml", "ICML", "conference", "pmlr", {"version": "1"})

        paper = coordinator.merge_batch(batch(entry("crossref", venue_id="icml")), venue)[0]
        coordinator.merge_batch(
            batch(
                entry(
                    "openalex",
                    venue_id="icml",
                    citation_count=7,
                    citation_count_as_of="2026-08-09T00:00:00Z",
                )
            ),
            venue,
        )

        membership = database.connection.execute(
            "SELECT membership_status FROM paper_collections WHERE paper_id = ?",
            (paper.paper_id,),
        ).fetchone()[0]
        assert membership == MembershipStatus.VENUE_CANDIDATE
        assert repository.get_paper(paper.paper_id).verification_status is VerificationStatus.VERIFIED

        coordinator.merge_batch(
            batch(entry("pmlr", venue_id="icml", official_membership=True, publication_version="published")),
            venue,
        )
        membership = database.connection.execute(
            "SELECT membership_status, official_evidence_json FROM paper_collections WHERE paper_id = ?",
            (paper.paper_id,),
        ).fetchone()
        assert membership["membership_status"] == MembershipStatus.OFFICIAL_CONFIRMED
        assert "pmlr:pmlr-1" in membership["official_evidence_json"]
        coordinator.merge_batch(batch(entry("crossref", venue_id="icml")), venue)
        official_evidence = database.connection.execute(
            "SELECT official_evidence_json FROM paper_collections WHERE paper_id = ?",
            (paper.paper_id,),
        ).fetchone()[0]
        assert "pmlr:pmlr-1" in official_evidence
        citation_count = database.connection.execute(
            "SELECT provider, count, observed_at FROM citation_counts"
        ).fetchone()
        assert tuple(citation_count) == ("openalex", 7, "2026-08-09T00:00:00Z")


def test_authority_precedence_selects_display_value_without_losing_provenance(tmp_path) -> None:
    with Database(tmp_path / "papers.sqlite3") as database:
        database.migrate()
        repository = PaperRepository(database)
        trusts = {
            "openalex": trust("openalex", authority_rank=2),
            "pmlr": trust(
                "pmlr",
                roles=frozenset({ProviderRole.VENUE_PRIMARY}),
                authority_rank=0,
            ),
        }
        coordinator = MetadataCoordinator(repository, trusts)
        graph = entry("openalex")
        official = SourceEntry(
            provider="pmlr",
            external_id="pmlr-1",
            title="The Official Published Title",
            authors=("Ada Lovelace",),
            year=2025,
            doi="10.1000/verified",
            metadata={"official_membership": True},
        )

        paper = coordinator.merge_batch(batch(graph))[0]
        coordinator.merge_batch(batch(official))

        assert repository.get_paper(paper.paper_id).title == "The Official Published Title"
        assert repository.get_paper(paper.paper_id).verification_status is VerificationStatus.VERIFIED
        values = database.connection.execute(
            "SELECT field_value_json FROM paper_field_provenance WHERE paper_id = ? AND field_name = 'title' ORDER BY field_value_json",
            (paper.paper_id,),
        ).fetchall()
        assert [row[0] for row in values] == ['"A Verified Paper"', '"The Official Published Title"']
