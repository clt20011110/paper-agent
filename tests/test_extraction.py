from __future__ import annotations

from io import BytesIO
import json

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from paper_agent.artifacts import ArtifactStore
from paper_agent.extraction import (
    PAGE_SEPARATOR,
    ExtractionConflict,
    ExtractionStatus,
    PdfTextExtractor,
)
from paper_agent.storage import Database


def _pdf_with_text(text: str, *, producer: str = "fixture") -> bytes:
    """Produce a small valid PDF whose text pypdf can deterministically read."""
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_ref = writer._add_object(font)
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref})}
    )
    stream = DecodedStreamObject()
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream.set_data(f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("ascii"))
    page[NameObject("/Contents")] = writer._add_object(stream)
    writer.add_metadata({"/Producer": producer})
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


@pytest.fixture
def database(tmp_path) -> Database:
    value = Database(tmp_path / "papers.sqlite3")
    value.migrate()
    value.connection.executemany(
        "INSERT INTO papers(paper_id, title) VALUES (?, ?)",
        (("paper-1", "One"), ("paper-2", "Two")),
    )
    value.connection.commit()
    yield value
    value.close()


def _source_artifact(
    database: Database, store: ArtifactStore, paper_id: str, payload: bytes, *, suffix: str
) -> str:
    stored = store.put_bytes(payload, mime_type="application/pdf", metadata={"fixture": suffix})
    artifact_id = f"pdf-{suffix}"
    database.connection.execute(
        """INSERT INTO artifacts(
               artifact_id, paper_id, artifact_kind, relative_path, mime_type, byte_size, sha256, provenance_json
           ) VALUES (?, ?, 'pdf', ?, 'application/pdf', ?, ?, '{}')""",
        (artifact_id, paper_id, stored.relative_path, stored.size_bytes, stored.artifact_hash),
    )
    database.connection.commit()
    return artifact_id


def test_extracts_stable_normalized_text_and_persists_auditable_rows(database: Database, tmp_path) -> None:
    store = ArtifactStore(tmp_path / "store")
    source_id = _source_artifact(database, store, "paper-1", _pdf_with_text("alpha " * 80), suffix="one")

    result = PdfTextExtractor(database, store, extractor_version="test-v1").extract("paper-1", source_id)

    assert result.status is ExtractionStatus.FULL_TEXT_READY
    assert result.page_count == 1
    assert result.character_count == len(("alpha " * 80).strip())
    assert result.text_coverage == 1.0
    assert result.output_artifact_id == f"artifact-{result.normalized_text_sha256}"
    text = store.read_bytes(result.normalized_text_sha256).decode("utf-8")
    assert text == PAGE_SEPARATOR.format(page_number=1) + ("alpha " * 80).strip()
    row = database.connection.execute(
        """SELECT a.artifact_kind, a.provenance_json, te.source_sha256, te.status
           FROM text_extractions te JOIN artifacts a ON a.artifact_id = te.output_artifact_id
           WHERE te.extraction_id = ?""",
        (result.extraction_id,),
    ).fetchone()
    assert row["artifact_kind"] == "text"
    assert row["source_sha256"]
    assert row["status"] == "full_text_ready"
    assert json.loads(row["provenance_json"])["source_artifact_id"] == source_id

    resumed = PdfTextExtractor(database, store, extractor_version="test-v1").extract("paper-1", source_id)
    assert resumed == result
    assert database.connection.execute("SELECT COUNT(*) FROM text_extractions").fetchone()[0] == 1


def test_blank_pdf_is_persisted_as_needing_ocr(database: Database, tmp_path) -> None:
    store = ArtifactStore(tmp_path / "store")
    source_id = _source_artifact(database, store, "paper-1", _pdf_with_text(""), suffix="blank")

    result = PdfTextExtractor(database, store, extractor_version="test-v1").extract("paper-1", source_id)

    assert result.status is ExtractionStatus.NEEDS_OCR
    assert result.character_count == 0
    assert result.text_coverage == 0.0
    assert store.read_bytes(result.normalized_text_sha256) == PAGE_SEPARATOR.format(page_number=1).encode()


def test_unparseable_pdf_records_deterministic_extraction_failure(database: Database, tmp_path) -> None:
    store = ArtifactStore(tmp_path / "store")
    source_id = _source_artifact(database, store, "paper-1", b"not a PDF", suffix="broken")

    result = PdfTextExtractor(database, store, extractor_version="test-v1").extract("paper-1", source_id)

    assert result.status is ExtractionStatus.EXTRACTION_FAILED
    assert result.output_artifact_id is None
    assert result.normalized_text_sha256 is None
    assert database.connection.execute("SELECT COUNT(*) FROM artifacts WHERE artifact_kind = 'text'").fetchone()[0] == 0


def test_encrypted_pdf_records_deterministic_extraction_failure(database: Database, tmp_path) -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.encrypt("secret")
    encrypted = BytesIO()
    writer.write(encrypted)
    store = ArtifactStore(tmp_path / "store")
    source_id = _source_artifact(database, store, "paper-1", encrypted.getvalue(), suffix="encrypted")

    result = PdfTextExtractor(database, store, extractor_version="test-v1").extract("paper-1", source_id)

    assert result.status is ExtractionStatus.EXTRACTION_FAILED
    assert result.output_artifact_id is None


def test_severe_garble_is_an_extraction_failure(database: Database, tmp_path, monkeypatch) -> None:
    class GarbledPage:
        def extract_text(self) -> str:
            return "\x00" * 250

    class GarbledReader:
        is_encrypted = False
        pages = (GarbledPage(),)

    monkeypatch.setattr("paper_agent.extraction.PdfReader", lambda *_args, **_kwargs: GarbledReader())
    store = ArtifactStore(tmp_path / "store")
    source_id = _source_artifact(database, store, "paper-1", _pdf_with_text("fixture"), suffix="garbled")

    result = PdfTextExtractor(database, store, extractor_version="test-v1").extract("paper-1", source_id)

    assert result.status is ExtractionStatus.EXTRACTION_FAILED
    assert result.printable_ratio == 0.0
    assert result.output_artifact_id is not None


def test_rejects_normalized_text_hash_reuse_between_papers(database: Database, tmp_path) -> None:
    store = ArtifactStore(tmp_path / "store")
    text = "same content " * 40
    first = _source_artifact(database, store, "paper-1", _pdf_with_text(text, producer="one"), suffix="one")
    second = _source_artifact(database, store, "paper-2", _pdf_with_text(text, producer="two"), suffix="two")
    extractor = PdfTextExtractor(database, store, extractor_version="test-v1")
    extractor.extract("paper-1", first)

    with pytest.raises(ExtractionConflict, match="belongs to another paper"):
        extractor.extract("paper-2", second)
