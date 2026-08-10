import csv
import json

from paper_agent.domain import CollectionMembership, MembershipStatus, SourceEntry
from paper_agent.exchange import (
    export_csv,
    export_jsonl,
    import_csv,
    import_jsonl,
    import_legacy_json,
)
from paper_agent.repository import PaperRepository
from paper_agent.storage import Database


def _repository(tmp_path):
    database = Database(tmp_path / "papers.sqlite3")
    database.migrate()
    return database, PaperRepository(database)


def _seed(repository: PaperRepository):
    paper = repository.ingest(
        SourceEntry(
            "openalex", "W1", "A paper", abstract="Abstract", authors=("Ada",),
            year=2025, doi="10.1/a", landing_url="https://example.test/paper", metadata={"kind": "article"},
        )
    )
    repository.save_collection("iclr-2025", "ICLR 2025", "conference", venue_id="iclr")
    repository.upsert_membership(
        CollectionMembership("iclr-2025", paper.paper_id, MembershipStatus.OFFICIAL_CONFIRMED, ("https://example.test",))
    )
    return paper


def test_jsonl_round_trip_preserves_papers_sources_and_memberships(tmp_path) -> None:
    database, repository = _repository(tmp_path)
    paper = _seed(repository)
    exported = tmp_path / "papers.jsonl"
    assert export_jsonl(repository, exported) == 3

    destination_database, destination = _repository(tmp_path / "destination")
    report = import_jsonl(destination, exported)
    assert report.counts == {"papers": 1, "sources": 1, "memberships": 1}
    assert destination.get_paper(paper.paper_id).title == "A paper"
    assert destination.connection.execute("SELECT COUNT(*) FROM paper_sources").fetchone()[0] == 1
    assert destination.connection.execute("SELECT COUNT(*) FROM paper_collections").fetchone()[0] == 1
    database.close()
    destination_database.close()


def test_csv_export_encodes_nested_values_as_json_strings(tmp_path) -> None:
    database, repository = _repository(tmp_path)
    _seed(repository)
    exported = tmp_path / "papers.csv"
    assert export_csv(repository, exported) == 1
    with exported.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert json.loads(row["authors"]) == ["Ada"]
    assert json.loads(row["sources_json"])[0]["raw_metadata"] == {"kind": "article"}
    assert json.loads(row["memberships_json"])[0]["collection"]["collection_id"] == "iclr-2025"
    database.close()


def test_csv_round_trip_preserves_papers_sources_and_memberships(tmp_path) -> None:
    database, repository = _repository(tmp_path)
    paper = _seed(repository)
    exported = tmp_path / "papers.csv"
    export_csv(repository, exported)

    destination_database, destination = _repository(tmp_path / "destination")
    report = import_csv(destination, exported)

    assert report.counts == {"papers": 1, "sources": 1, "memberships": 1}
    assert destination.get_paper(paper.paper_id).authors == ("Ada",)
    assert destination.connection.execute(
        "SELECT COUNT(*) FROM paper_sources"
    ).fetchone()[0] == 1
    assert destination.connection.execute(
        "SELECT COUNT(*) FROM paper_collections"
    ).fetchone()[0] == 1
    database.close()
    destination_database.close()


def test_csv_import_is_idempotent_and_dry_run_does_not_write(tmp_path) -> None:
    source_database, source = _repository(tmp_path / "source")
    _seed(source)
    exported = tmp_path / "papers.csv"
    export_csv(source, exported)
    database, repository = _repository(tmp_path / "destination")

    assert import_csv(repository, exported, dry_run=True).counts["papers"] == 1
    assert repository.connection.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 0
    import_csv(repository, exported)
    import_csv(repository, exported)
    assert repository.connection.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 1
    assert repository.connection.execute("SELECT COUNT(*) FROM paper_sources").fetchone()[0] == 1
    source_database.close()
    database.close()


def test_legacy_json_dry_run_then_commit_uses_stable_canonical_ids(tmp_path) -> None:
    legacy = tmp_path / "legacy.json"
    legacy.write_text(
        json.dumps({"papers": [{
            "id": "old-1", "title": "Legacy paper", "abstract": "Old abstract", "authors": ["Ada"],
            "keywords": ["ML"], "year": 2024, "venue": "ICLR", "venue_type": "conference",
            "source_platform": "OpenReview", "pdf_url": "https://example.test/p.pdf", "doi": "10.1/legacy",
            "bibtex": "@article{legacy}", "citation_count": 3, "arxiv_id": "2401.00001v2",
        }]}),
        encoding="utf-8",
    )
    database, repository = _repository(tmp_path)
    report = import_legacy_json(repository, legacy, dry_run=True)
    assert report.counts == {"papers": 1, "sources": 1, "memberships": 0}
    assert repository.connection.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 0
    assert report.mappings["pdf_url"] == "source.pdf_url"

    import_legacy_json(repository, legacy)
    paper = repository.find_paper(doi="10.1/legacy")
    source = repository.connection.execute("SELECT provider, pdf_url, citation_count FROM paper_sources").fetchone()
    assert paper.arxiv_id == "2401.00001"
    assert tuple(source) == ("openreview", "https://example.test/p.pdf", 3)
    database.close()


def test_importing_same_legacy_file_twice_is_idempotent(tmp_path) -> None:
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps([{"id": "old-1", "title": "Legacy paper", "source_platform": "legacy"}]), encoding="utf-8")
    database, repository = _repository(tmp_path)
    import_legacy_json(repository, legacy)
    import_legacy_json(repository, legacy)
    assert repository.connection.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 1
    assert repository.connection.execute("SELECT COUNT(*) FROM paper_sources").fetchone()[0] == 1
    database.close()
