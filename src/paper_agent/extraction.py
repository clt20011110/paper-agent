"""Deterministic, local PDF-to-text extraction.

This module deliberately only turns a previously validated PDF artifact into a
normalized UTF-8 text artifact.  It neither performs OCR nor sends document
content to a model or a network service.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import hashlib
from io import BytesIO
import json
from typing import Final
import unicodedata

import pypdf
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from .artifacts import ArtifactStore, StoredArtifact
from .canonical import content_hash
from .storage import Database


EXTRACTOR_NAME: Final = "pypdf"
EXTRACTOR_VERSION: Final = pypdf.__version__
NORMALIZATION_VERSION: Final = "utf8-nfc-lines-v1"
PAGE_SEPARATOR: Final = "\n\n===== PAGE {page_number} =====\n\n"

# A paper needs enough readable text on most pages before it is safe to call it
# full text.  These deliberately small, explicit thresholds make the result
# deterministic and easy to audit; OCR remains a separate, opt-in phase.
MIN_CHARACTERS_PER_COVERED_PAGE: Final = 80
MIN_TOTAL_CHARACTERS: Final = 200
MIN_TEXT_COVERAGE: Final = 0.60
MIN_PRINTABLE_RATIO: Final = 0.95
MAX_REPLACEMENT_RATIO: Final = 0.01


class ExtractionStatus(StrEnum):
    """The deterministic extraction outcome, not an analysis outcome."""

    FULL_TEXT_READY = "full_text_ready"
    NEEDS_OCR = "needs_ocr"
    EXTRACTION_FAILED = "extraction_failed"


class ExtractionError(ValueError):
    """The requested artifact cannot safely participate in an extraction."""


class ExtractionConflict(ExtractionError):
    """An immutable artifact or extraction record has incompatible ownership."""


@dataclass(frozen=True, slots=True)
class TextExtractionResult:
    """Persisted evidence for one deterministic PDF extraction."""

    extraction_id: str
    paper_id: str
    source_artifact_id: str
    source_sha256: str
    output_artifact_id: str | None
    normalized_text_sha256: str | None
    extractor_name: str
    extractor_version: str
    page_count: int
    character_count: int
    text_coverage: float
    printable_ratio: float
    status: ExtractionStatus


@dataclass(frozen=True, slots=True)
class _ExtractedText:
    normalized_text: str
    page_count: int
    character_count: int
    text_coverage: float
    printable_ratio: float
    replacement_ratio: float
    status: ExtractionStatus


class PdfTextExtractor:
    """Extract and persist normalized text for PDF artifacts owned by one paper.

    The unique source/extractor/version key in ``text_extractions`` is the
    resume key.  A completed attempt (including a deterministic failure) is
    returned without re-reading the PDF.
    """

    def __init__(
        self,
        database: Database,
        artifact_store: ArtifactStore,
        *,
        extractor_name: str = EXTRACTOR_NAME,
        extractor_version: str = EXTRACTOR_VERSION,
    ) -> None:
        if not extractor_name or not extractor_version:
            raise ValueError("extractor_name and extractor_version are required")
        self.database = database
        self.artifact_store = artifact_store
        self.extractor_name = extractor_name
        self.extractor_version = extractor_version

    def extract(self, paper_id: str, source_artifact_id: str) -> TextExtractionResult:
        """Extract ``source_artifact_id`` after validating its database ownership.

        Files are written by :class:`ArtifactStore` with its atomic replace
        protocol; rows for the text artifact and extraction are committed in a
        single SQLite transaction.  An orphaned immutable file after a process
        crash is harmless and is reused only when its provenance agrees.
        """
        source = self._source_artifact(paper_id, source_artifact_id)
        existing = self._existing(source_artifact_id)
        if existing is not None:
            self._validate_existing(existing, paper_id, source["sha256"])
            return existing

        payload = self.artifact_store.read_bytes(source["sha256"])
        extracted = self._extract(payload)
        stored: StoredArtifact | None = None
        if extracted.status is not ExtractionStatus.EXTRACTION_FAILED or extracted.normalized_text:
            stored = self._store_text(source, extracted)

        with self.database.transaction() as connection:
            # Another resume worker may have committed while this worker was
            # reading.  Preserve the first immutable result in that case.
            existing = self._existing(source_artifact_id, connection=connection)
            if existing is not None:
                self._validate_existing(existing, paper_id, source["sha256"])
                return existing
            output_artifact_id = self._persist_artifact(connection, paper_id, source, stored)
            extraction_id = "text-extraction-" + content_hash(
                {
                    "source_artifact_id": source_artifact_id,
                    "extractor_name": self.extractor_name,
                    "extractor_version": self.extractor_version,
                }
            )
            connection.execute(
                """INSERT INTO text_extractions(
                       extraction_id, paper_id, source_artifact_id, source_sha256,
                       output_artifact_id, extractor_name, extractor_version,
                       page_count, character_count, text_coverage, printable_ratio, status
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    extraction_id,
                    paper_id,
                    source_artifact_id,
                    source["sha256"],
                    output_artifact_id,
                    self.extractor_name,
                    self.extractor_version,
                    extracted.page_count,
                    extracted.character_count,
                    extracted.text_coverage,
                    extracted.printable_ratio,
                    extracted.status.value,
                ),
            )
        return TextExtractionResult(
            extraction_id=extraction_id,
            paper_id=paper_id,
            source_artifact_id=source_artifact_id,
            source_sha256=source["sha256"],
            output_artifact_id=output_artifact_id,
            normalized_text_sha256=stored.artifact_hash if stored is not None else None,
            extractor_name=self.extractor_name,
            extractor_version=self.extractor_version,
            page_count=extracted.page_count,
            character_count=extracted.character_count,
            text_coverage=extracted.text_coverage,
            printable_ratio=extracted.printable_ratio,
            status=extracted.status,
        )

    def _source_artifact(self, paper_id: str, source_artifact_id: str):
        source = self.database.connection.execute(
            """SELECT artifact_id, paper_id, artifact_kind, mime_type, sha256
               FROM artifacts WHERE artifact_id = ?""",
            (source_artifact_id,),
        ).fetchone()
        if source is None:
            raise ExtractionError("source PDF artifact does not exist")
        if source["paper_id"] != paper_id:
            raise ExtractionConflict("source PDF artifact belongs to another paper")
        if source["artifact_kind"] != "pdf" or source["mime_type"] != "application/pdf":
            raise ExtractionError("source artifact is not a validated PDF")
        return source

    def _existing(self, source_artifact_id: str, *, connection=None) -> TextExtractionResult | None:
        database = connection if connection is not None else self.database.connection
        row = database.execute(
            """SELECT te.*, output.sha256 AS output_sha256
               FROM text_extractions te
               LEFT JOIN artifacts output ON output.artifact_id = te.output_artifact_id
               WHERE te.source_artifact_id = ?
                 AND te.extractor_name = ? AND te.extractor_version = ?""",
            (source_artifact_id, self.extractor_name, self.extractor_version),
        ).fetchone()
        if row is None:
            return None
        if row["output_artifact_id"] is not None and row["output_sha256"] is None:
            raise ExtractionConflict("text extraction refers to a missing output artifact")
        return TextExtractionResult(
            extraction_id=row["extraction_id"],
            paper_id=row["paper_id"],
            source_artifact_id=row["source_artifact_id"],
            source_sha256=row["source_sha256"],
            output_artifact_id=row["output_artifact_id"],
            normalized_text_sha256=row["output_sha256"],
            extractor_name=row["extractor_name"],
            extractor_version=row["extractor_version"],
            page_count=row["page_count"],
            character_count=row["character_count"],
            text_coverage=row["text_coverage"],
            printable_ratio=row["printable_ratio"],
            status=ExtractionStatus(row["status"]),
        )

    @staticmethod
    def _validate_existing(result: TextExtractionResult, paper_id: str, source_sha256: str) -> None:
        if result.paper_id != paper_id or result.source_sha256 != source_sha256:
            raise ExtractionConflict("existing text extraction has incompatible source binding")

    def _store_text(self, source, extracted: _ExtractedText) -> StoredArtifact:
        text_bytes = extracted.normalized_text.encode("utf-8")
        text_hash = hashlib.sha256(text_bytes).hexdigest()
        existing = self.database.connection.execute(
            "SELECT paper_id, artifact_kind FROM artifacts WHERE sha256 = ?", (text_hash,)
        ).fetchone()
        if existing is not None:
            if existing["paper_id"] != source["paper_id"]:
                raise ExtractionConflict("normalized text hash already belongs to another paper")
            raise ExtractionConflict("normalized text hash already has incompatible source provenance")
        return self.artifact_store.put_bytes(
            text_bytes,
            mime_type="text/plain; charset=utf-8",
            metadata={
                "artifact_kind": "text",
                "source_pdf_sha256": source["sha256"],
                "extractor_name": self.extractor_name,
                "extractor_version": self.extractor_version,
                "normalization": NORMALIZATION_VERSION,
            },
        )

    def _persist_artifact(self, connection, paper_id: str, source, stored: StoredArtifact | None) -> str | None:
        if stored is None:
            return None
        existing = connection.execute(
            "SELECT artifact_id, paper_id, artifact_kind, mime_type, byte_size, relative_path "
            "FROM artifacts WHERE sha256 = ?",
            (stored.artifact_hash,),
        ).fetchone()
        if existing is not None:
            if (
                existing["paper_id"] != paper_id
                or existing["artifact_kind"] != "text"
                or existing["mime_type"] != stored.mime_type
                or existing["byte_size"] != stored.size_bytes
                or existing["relative_path"] != stored.relative_path
            ):
                raise ExtractionConflict("normalized text hash has incompatible artifact metadata")
            return str(existing["artifact_id"])
        artifact_id = "artifact-" + stored.artifact_hash
        provenance = json.dumps(
            {
                "source_artifact_id": source["artifact_id"],
                "source_pdf_sha256": source["sha256"],
                "extractor_name": self.extractor_name,
                "extractor_version": self.extractor_version,
                "normalization": NORMALIZATION_VERSION,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        connection.execute(
            """INSERT INTO artifacts(
                   artifact_id, paper_id, artifact_kind, relative_path, mime_type,
                   byte_size, sha256, provenance_json
               ) VALUES (?, ?, 'text', ?, ?, ?, ?, ?)""",
            (
                artifact_id,
                paper_id,
                stored.relative_path,
                stored.mime_type,
                stored.size_bytes,
                stored.artifact_hash,
                provenance,
            ),
        )
        return artifact_id

    @staticmethod
    def _extract(payload: bytes) -> _ExtractedText:
        try:
            reader = PdfReader(BytesIO(payload), strict=False)
            if reader.is_encrypted:
                return _failed_text()
            pages = tuple(_normalize_page(page.extract_text() or "") for page in reader.pages)
        except PdfReadError:
            return _failed_text()

        normalized_text = "".join(
            PAGE_SEPARATOR.format(page_number=index) + page_text
            for index, page_text in enumerate(pages, start=1)
        )
        character_count = sum(len(page) for page in pages)
        page_count = len(pages)
        coverage = (
            sum(len(page) >= MIN_CHARACTERS_PER_COVERED_PAGE for page in pages) / page_count
            if page_count
            else 0.0
        )
        printable_ratio = _printable_ratio("".join(pages))
        replacement_ratio = (
            sum(page.count("\ufffd") for page in pages) / character_count if character_count else 0.0
        )
        if character_count and (
            printable_ratio < MIN_PRINTABLE_RATIO or replacement_ratio > MAX_REPLACEMENT_RATIO
        ):
            status = ExtractionStatus.EXTRACTION_FAILED
        elif (
            character_count < MIN_TOTAL_CHARACTERS
            or coverage < MIN_TEXT_COVERAGE
        ):
            status = ExtractionStatus.NEEDS_OCR
        else:
            status = ExtractionStatus.FULL_TEXT_READY
        return _ExtractedText(
            normalized_text=normalized_text,
            page_count=page_count,
            character_count=character_count,
            text_coverage=coverage,
            printable_ratio=printable_ratio,
            replacement_ratio=replacement_ratio,
            status=status,
        )


def _normalize_page(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in normalized.split("\n")).strip()


def _printable_ratio(text: str) -> float:
    if not text:
        return 0.0
    printable = sum(character.isprintable() or character in "\n\t" for character in text)
    return printable / len(text)


def _failed_text() -> _ExtractedText:
    return _ExtractedText(
        normalized_text="",
        page_count=0,
        character_count=0,
        text_coverage=0.0,
        printable_ratio=0.0,
        replacement_ratio=0.0,
        status=ExtractionStatus.EXTRACTION_FAILED,
    )
