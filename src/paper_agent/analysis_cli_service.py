"""Thin SQLite/manifest adapter for the Stage 4 ``analyze`` CLI command.

Argument parsing deliberately stays out of this module.  The adapter resolves
only canonical paper IDs or persisted Stage 3 PDF artifacts, prepares local
text through :class:`PdfTextExtractor`, and delegates every remote boundary to
``PaperAnalysisCoordinator`` and its ``ProcessingGate``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

from .analysis import AnalysisInput, AnalysisRunResult, AnalysisInvoker, PaperAnalysisCoordinator
from .artifacts import ArtifactStore
from .extraction import ExtractionStatus, PdfTextExtractor
from .grants import GrantStore
from .processing import ArtifactProcessingPolicy, ProcessingGate
from .storage import Database


@dataclass(frozen=True, slots=True)
class AnalysisInputManifest:
    """Explicit, reviewable selection for one Stage 4 run."""

    paper_ids: tuple[str, ...] = ()
    stage3_artifact_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        selected = (*self.paper_ids, *self.stage3_artifact_ids)
        if not selected:
            raise ValueError("analysis input manifest selects no papers or Stage 3 artifacts")
        if any(not item for item in selected):
            raise ValueError("analysis input manifest IDs must be non-empty")
        if len(set(self.paper_ids)) != len(self.paper_ids):
            raise ValueError("analysis input manifest has duplicate paper_ids")
        if len(set(self.stage3_artifact_ids)) != len(self.stage3_artifact_ids):
            raise ValueError("analysis input manifest has duplicate stage3_artifact_ids")


@dataclass(frozen=True, slots=True)
class AnalysisServiceResult:
    """JSON-friendly result of a dry-run or a persisted analysis run."""

    run_id: str
    dry_run: bool
    selected_paper_ids: tuple[str, ...]
    input_scopes: tuple[str, ...]
    result: AnalysisRunResult | None = None


def load_analysis_input_manifest(path: str | Path) -> AnalysisInputManifest:
    """Load the small, explicit JSON selection consumed by the CLI adapter."""
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version", "paper_ids", "stage3_artifact_ids",
    }:
        raise ValueError("analysis input manifest must contain schema_version, paper_ids, and stage3_artifact_ids")
    if value["schema_version"] != "1":
        raise ValueError("analysis input manifest must use schema_version 1")
    paper_ids = _string_list(value["paper_ids"], "paper_ids")
    artifact_ids = _string_list(value["stage3_artifact_ids"], "stage3_artifact_ids")
    return AnalysisInputManifest(paper_ids, artifact_ids)


class AnalysisCliService:
    """Resolve Stage 4 inputs from SQLite without adding a second pipeline."""

    def __init__(
        self,
        database: Database,
        artifact_store: ArtifactStore,
        policy: ArtifactProcessingPolicy,
        *,
        grants: GrantStore | None = None,
        invoker_factory: Callable[[], AnalysisInvoker] | None = None,
        extractor: PdfTextExtractor | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.database = database
        self.artifact_store = artifact_store
        self.gate = ProcessingGate(policy, grants)
        self.invoker_factory = invoker_factory
        self.extractor = extractor or PdfTextExtractor(database, artifact_store)
        self.clock = _trusted_clock(clock)

    def run(
        self,
        run_id: str,
        manifest: AnalysisInputManifest,
        *,
        processing_grant_id: str | None = None,
        dry_run: bool = False,
    ) -> AnalysisServiceResult:
        """Resolve inputs and run Stage 4, or validate them without side effects.

        A dry run never invokes the extractor/coordinator, so it makes neither
        SQLite/artifact writes nor Codex calls.  Normal runs reuse completed
        extraction and analysis rows through their own immutable resume keys.
        """
        if not run_id:
            raise ValueError("run_id is required")
        now = self.clock()
        inputs = self._inputs(manifest, extract=True, preview=dry_run)
        if dry_run:
            # The preview runs the same local extraction selection as execution
            # without persisting its derived artifact or constructing Codex.
            for item in inputs:
                self.gate.decide(
                    item.processing_request(),
                    processing_grant_id=processing_grant_id,
                    now=now,
                )
            return AnalysisServiceResult(
                run_id, True, tuple(item.paper_id for item in inputs),
                tuple(item.processing_request().input_scope for item in inputs),
            )
        options: dict[str, Any] = {}
        if self.invoker_factory is not None:
            options["invoker_factory"] = self.invoker_factory
        coordinator = PaperAnalysisCoordinator(
            self.database, self.artifact_store, self.gate, clock=self.clock, **options,
        )
        result = coordinator.run(
            run_id, inputs, now=now, processing_grant_id=processing_grant_id,
        )
        return AnalysisServiceResult(
            run_id, False, tuple(item.paper_id for item in inputs),
            tuple(item.processing_request().input_scope for item in inputs), result,
        )

    def _inputs(
        self, manifest: AnalysisInputManifest, *, extract: bool, preview: bool = False,
    ) -> tuple[AnalysisInput, ...]:
        selected: dict[str, Any] = {}
        for paper_id in manifest.paper_ids:
            selected[paper_id] = self._best_artifact(paper_id)
        for artifact_id in manifest.stage3_artifact_ids:
            row = self._stage3_artifact(artifact_id)
            previous = selected.setdefault(row["paper_id"], row)
            if previous is not row and previous["artifact_id"] != row["artifact_id"]:
                raise ValueError("analysis manifest selects multiple artifacts for one paper")
        return tuple(
            self._input_for(row, extract=extract, preview=preview)
            for _, row in sorted(selected.items())
        )

    def _best_artifact(self, paper_id: str):
        paper = self._paper(paper_id)
        row = self.database.connection.execute(
            """SELECT a.* FROM text_extractions te
               JOIN artifacts a ON a.artifact_id = te.output_artifact_id
               WHERE te.paper_id = ? AND te.status = 'full_text_ready'
                 AND a.processing_status = 'available'
               ORDER BY te.created_at DESC LIMIT 1""",
            (paper_id,),
        ).fetchone()
        if row is not None:
            return row
        row = self.database.connection.execute(
            """SELECT * FROM artifacts WHERE paper_id = ? AND artifact_kind = 'pdf'
                 AND mime_type = 'application/pdf' AND processing_status = 'available'
               ORDER BY created_at DESC, artifact_id DESC LIMIT 1""",
            (paper_id,),
        ).fetchone()
        return row if row is not None else paper

    def _stage3_artifact(self, artifact_id: str):
        row = self.database.connection.execute(
            """SELECT a.* FROM artifacts a JOIN download_attempts da ON da.artifact_id = a.artifact_id
               WHERE a.artifact_id = ? AND a.artifact_kind = 'pdf'
                 AND a.mime_type = 'application/pdf' AND a.processing_status = 'available'
                 AND da.result_status = 'downloaded'
               ORDER BY da.attempted_at DESC LIMIT 1""",
            (artifact_id,),
        ).fetchone()
        if row is None:
            raise ValueError("stage3_artifact_id must reference an available downloaded PDF")
        return row

    def _input_for(self, source: Any, *, extract: bool, preview: bool = False) -> AnalysisInput:
        paper = self._paper(str(source["paper_id"]))
        facts = self._facts(str(paper["paper_id"]), source)
        if "artifact_id" not in source.keys():
            metadata = _metadata(paper)
            if paper["abstract"]:
                return AnalysisInput(paper["paper_id"], facts["license"], "public_read_only",
                                     abstract=paper["abstract"], metadata=metadata, domain=facts["domain"])
            return AnalysisInput(paper["paper_id"], facts["license"], "public_read_only",
                                 metadata=metadata, domain=facts["domain"])
        if source["artifact_kind"] == "text":
            return AnalysisInput(paper["paper_id"], facts["license"], facts["access_basis"],
                                 normalized_text=self.artifact_store.read_bytes(source["sha256"]),
                                 artifact_id=source["artifact_id"], domain=facts["domain"])
        if extract:
            if preview:
                return self._preview_extracted_input(paper, source, facts)
            extracted = self.extractor.extract(str(paper["paper_id"]), str(source["artifact_id"]))
            if extracted.status is ExtractionStatus.FULL_TEXT_READY and extracted.output_artifact_id:
                return AnalysisInput(paper["paper_id"], facts["license"], facts["access_basis"],
                                     normalized_text=self.artifact_store.read_bytes(extracted.normalized_text_sha256 or ""),
                                     artifact_id=extracted.output_artifact_id, domain=facts["domain"])
        return AnalysisInput(paper["paper_id"], facts["license"], facts["access_basis"],
                             full_pdf=self.artifact_store.read_bytes(source["sha256"]),
                             artifact_id=source["artifact_id"], domain=facts["domain"])

    def _preview_extracted_input(
        self, paper: Any, source: Any, facts: Mapping[str, str | None],
    ) -> AnalysisInput:
        """Run the extractor's deterministic algorithm without artifact/SQLite writes."""
        extracted = self.extractor._extract(  # noqa: SLF001 - dry-run needs the exact extraction algorithm.
            self.artifact_store.read_bytes(source["sha256"])
        )
        if extracted.status is ExtractionStatus.FULL_TEXT_READY:
            return AnalysisInput(
                paper["paper_id"], facts["license"], facts["access_basis"],
                normalized_text=extracted.normalized_text,
                artifact_id=source["artifact_id"], domain=facts["domain"],
            )
        return AnalysisInput(
            paper["paper_id"], facts["license"], facts["access_basis"],
            full_pdf=self.artifact_store.read_bytes(source["sha256"]),
            artifact_id=source["artifact_id"], domain=facts["domain"],
        )

    def _paper(self, paper_id: str):
        row = self.database.connection.execute("SELECT * FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()
        if row is None:
            raise ValueError(f"unknown canonical paper_id: {paper_id}")
        return row

    def _facts(self, paper_id: str, source: Any) -> Mapping[str, str | None]:
        artifact_id = source["artifact_id"] if "artifact_id" in source.keys() else None
        row = self.database.connection.execute(
            """SELECT dc.license, dc.access_basis, dc.host FROM artifacts a
               LEFT JOIN text_extractions te ON te.output_artifact_id = a.artifact_id
               JOIN download_attempts da ON da.artifact_id = COALESCE(te.source_artifact_id, a.artifact_id)
               JOIN download_candidates dc ON dc.candidate_id = da.candidate_id
               WHERE a.artifact_id = ? AND da.result_status = 'downloaded'
               ORDER BY da.attempted_at DESC LIMIT 1""",
            (artifact_id,),
        ).fetchone() if artifact_id else None
        return {
            "license": row["license"] if row else None,
            "access_basis": row["access_basis"] if row else "unknown",
            "domain": row["host"] if row else None,
        }


def _string_list(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"analysis input manifest {name} must be a string list")
    return tuple(value)


def _trusted_clock(clock: Callable[[], datetime] | None) -> Callable[[], datetime]:
    source = clock or (lambda: datetime.now(UTC))

    def current() -> datetime:
        value = source()
        if not isinstance(value, datetime) or value.tzinfo is None:
            raise ValueError("Stage 4 clock must return a timezone-aware datetime")
        return value.astimezone(UTC)

    return current


def _metadata(paper: Any) -> dict[str, object]:
    return {
        key: paper[key] for key in (
            "title", "authors_json", "keywords_json", "publication_date", "year", "venue_id",
            "venue_name", "doi", "arxiv_id", "canonical_url", "verification_status",
        ) if paper[key] is not None
    }
