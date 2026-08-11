import pytest

from paper_agent.domain import (
    AccessBasis,
    CollectionMembership,
    MembershipStatus,
    Paper,
    PaperSource,
    PublicationVersion,
    SourceEntry,
)
from paper_agent.repository import PaperRepository
from paper_agent.storage import Database


def _repository(tmp_path):
    database = Database(tmp_path / "papers.sqlite3")
    database.migrate()
    return database, PaperRepository(database)


def test_identity_priority_is_doi_then_arxiv_then_provider_id(tmp_path) -> None:
    database, repository = _repository(tmp_path)
    first = repository.ingest(
        SourceEntry("crossref", "doi-1", "First", authors=("Ada",), year=2025, doi="https://doi.org/10.1/ABC.")
    )
    second = repository.ingest(
        SourceEntry("arxiv", "2401.00001", "Renamed", arxiv_id="2401.00001", doi="doi:10.1/abc")
    )
    assert second.paper_id == first.paper_id
    assert repository.find_paper(doi="10.1/abc").paper_id == first.paper_id
    assert repository.find_paper(arxiv_id="2401.00001").paper_id == first.paper_id
    database.close()


def test_same_provider_source_is_idempotent_and_never_rebinds(tmp_path) -> None:
    database, repository = _repository(tmp_path)
    paper = repository.ingest(SourceEntry("openalex", "W1", "One"))
    assert repository.ingest(SourceEntry("openalex", "W1", "One updated")) == paper
    other = repository.save_paper(Paper("paper-other", "Other"))
    source = PaperSource("source-other", other.paper_id, "openalex", "W1", raw_metadata={})

    with pytest.raises(ValueError, match="already bound"):
        repository.upsert_source(source)

    assert repository.find_paper(provider="openalex", external_id="W1").paper_id == paper.paper_id
    assert database.connection.execute("SELECT queue_type FROM manual_queue").fetchone()[0] == "merge_conflict"
    database.close()


def test_ingest_preserves_typed_official_public_pdf_metadata(tmp_path) -> None:
    database, repository = _repository(tmp_path)
    paper = repository.ingest(SourceEntry(
        "neurips_proceedings",
        "NeurIPS-2024-abc123",
        "Public conference paper",
        landing_url="https://proceedings.neurips.cc/paper_files/paper/2024/hash/abc123-Abstract-Conference.html",
        pdf_url="https://proceedings.neurips.cc/paper_files/paper/2024/hash/abc123-Paper-Conference.pdf",
        publication_version=PublicationVersion.PUBLISHED,
        host_type="official",
        access_basis=AccessBasis.PUBLIC_READ_ONLY,
    ))

    row = database.connection.execute(
        """SELECT pdf_url, publication_version, license, host_type, access_basis
           FROM paper_sources WHERE paper_id = ?""",
        (paper.paper_id,),
    ).fetchone()

    assert tuple(row) == (
        "https://proceedings.neurips.cc/paper_files/paper/2024/hash/abc123-Paper-Conference.pdf",
        "published",
        None,
        "official",
        "public_read_only",
    )
    database.close()


def test_two_ingests_preserve_sources_provenance_and_citations_without_duplicates(tmp_path) -> None:
    database, repository = _repository(tmp_path)
    paper = repository.ingest(SourceEntry("crossref", "1", "A paper", authors=("Ada",), year=2024, doi="10.1/a"))
    repository.ingest(SourceEntry("openalex", "W1", "A paper", authors=("Ada",), year=2024, doi="10.1/a"))
    repository.ingest(SourceEntry("openalex", "W1", "A paper", authors=("Ada",), year=2024, doi="10.1/a"))
    repository.record_citation_count(paper.paper_id, "openalex", 4, "2026-08-09T00:00:00Z")
    repository.record_citation_count(paper.paper_id, "semantic_scholar", 5, "2026-08-09T00:00:00Z")

    assert database.connection.execute("SELECT COUNT(*) FROM paper_sources").fetchone()[0] == 2
    assert database.connection.execute("SELECT COUNT(*) FROM paper_field_provenance").fetchone()[0] == 8
    assert [tuple(row) for row in database.connection.execute("SELECT provider, count FROM citation_counts ORDER BY provider").fetchall()] == [
        ("openalex", 4), ("semantic_scholar", 5)
    ]
    database.close()


def test_title_author_year_only_adds_manual_candidate_without_merging(tmp_path) -> None:
    database, repository = _repository(tmp_path)
    one = repository.ingest(SourceEntry("first", "1", "Same Title", authors=("Ada Lovelace",), year=2024))
    two = repository.ingest(SourceEntry("second", "2", "same-title", authors=("Ada Lovelace",), year=2024))

    assert one.paper_id != two.paper_id
    queue = database.connection.execute("SELECT queue_type, paper_id FROM manual_queue").fetchone()
    assert tuple(queue) == ("dedup", two.paper_id)
    database.close()


def test_membership_and_arxiv_candidates_are_stored_separately(tmp_path) -> None:
    database, repository = _repository(tmp_path)
    paper = repository.ingest(SourceEntry("arxiv", "2401.00001", "Candidate", arxiv_id="2401.00001v2"))
    repository.save_collection("arxiv-2024", "arXiv 2024", "arxiv")
    repository.upsert_membership(
        CollectionMembership("arxiv-2024", paper.paper_id, MembershipStatus.VENUE_CANDIDATE)
    )

    assert repository.arxiv_candidates() == (paper,)
    assert database.connection.execute("SELECT membership_status FROM paper_collections").fetchone()[0] == "venue_candidate"
    database.close()
