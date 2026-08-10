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

from .analysis import (
    AnalysisInput,
    AnalysisInvoker,
    AnalysisRunResult,
    PaperAnalysisCoordinator,
    analysis_configuration_denial,
    load_analysis_output_schema,
)
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
        workers: int = 1,
        allow_abstract_only: bool = True,
        output_schema_path: str | Path | None = None,
    ) -> None:
        if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
            raise ValueError("analysis workers must be a positive integer")
        if not isinstance(allow_abstract_only, bool):
            raise ValueError("allow_abstract_only must be a boolean")
        self.database = database
        self.artifact_store = artifact_store
        self.gate = ProcessingGate(policy, grants)
        self.invoker_factory = invoker_factory
        self.extractor = extractor or PdfTextExtractor(database, artifact_store)
        self.clock = _trusted_clock(clock)
        self.workers = workers
        self.allow_abstract_only = allow_abstract_only
        self.output_schema_path = load_analysis_output_schema(output_schema_path)[0]

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
        inputs = self._inputs(manifest, extract=True, preview=dry_run)
        return self._run_inputs(
            run_id,
            inputs,
            processing_grant_id=processing_grant_id,
            dry_run=dry_run,
        )

    def run_from_stage3(
        self,
        run_id: str,
        stage3_run_id: str,
        *,
        expected_paper_ids: Sequence[str],
        processing_grant_id: str | None = None,
        dry_run: bool = False,
    ) -> AnalysisServiceResult:
        """Analyze one Stage 3 run only when all expected checkpoints are present."""
        if not run_id or not stage3_run_id:
            raise ValueError("run_id and stage3_run_id are required")
        if isinstance(expected_paper_ids, (str, bytes)):
            raise ValueError("expected Stage 3 paper IDs must be a sequence")
        expected = tuple(expected_paper_ids)
        if any(
            not isinstance(paper_id, str) or not paper_id
            for paper_id in expected
        ):
            raise ValueError("expected Stage 3 paper IDs must be non-empty strings")
        if len(set(expected)) != len(expected):
            raise ValueError("expected Stage 3 paper IDs must be unique")
        inputs = self._stage3_inputs(
            stage3_run_id,
            expected_paper_ids=expected,
            preview=dry_run,
        )
        return self._run_inputs(
            run_id,
            inputs,
            processing_grant_id=processing_grant_id,
            dry_run=dry_run,
        )

    def _run_inputs(
        self,
        run_id: str,
        inputs: tuple[AnalysisInput, ...],
        *,
        processing_grant_id: str | None,
        dry_run: bool,
    ) -> AnalysisServiceResult:
        now = self.clock()
        if dry_run:
            # The preview runs the same local extraction selection as execution
            # without persisting its derived artifact or constructing Codex.
            for item in inputs:
                request = item.processing_request()
                if analysis_configuration_denial(
                    self.gate,
                    request,
                    allow_abstract_only=self.allow_abstract_only,
                ) is None:
                    self.gate.decide(
                        request,
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
            self.database,
            self.artifact_store,
            self.gate,
            clock=self.clock,
            workers=self.workers,
            allow_abstract_only=self.allow_abstract_only,
            output_schema_path=self.output_schema_path,
            **options,
        )
        result = coordinator.run(
            run_id, inputs, now=now, processing_grant_id=processing_grant_id,
        )
        return AnalysisServiceResult(
            run_id, False, tuple(item.paper_id for item in inputs),
            tuple(item.processing_request().input_scope for item in inputs), result,
        )

    def _stage3_inputs(
        self,
        stage3_run_id: str,
        *,
        expected_paper_ids: Sequence[str],
        preview: bool,
    ) -> tuple[AnalysisInput, ...]:
        run = self.database.connection.execute(
            "SELECT stage, status FROM pipeline_runs WHERE run_id = ?",
            (stage3_run_id,),
        ).fetchone()
        if run is None or tuple(run) != ("stage-3-download", "complete"):
            raise ValueError("stage3_run_id must name a complete Stage 3 download run")
        results = self.database.connection.execute(
            """SELECT paper_id, status FROM stage3_paper_results
               WHERE run_id = ? ORDER BY paper_id""",
            (stage3_run_id,),
        ).fetchall()
        actual_paper_ids = tuple(str(result["paper_id"]) for result in results)
        if actual_paper_ids != tuple(sorted(expected_paper_ids)):
            raise ValueError(
                "Stage 3 paper checkpoints do not match the expected selection"
            )
        inputs: list[AnalysisInput] = []
        for result in results:
            paper_id = str(result["paper_id"])
            if result["status"] == "downloaded":
                source = self._stage3_run_artifact(stage3_run_id, paper_id)
            elif result["status"] in {"not_available", "failed_terminal"}:
                source = self._paper(paper_id)
            else:
                raise ValueError("complete Stage 3 run contains a non-terminal paper result")
            inputs.append(self._input_for(
                source,
                extract=True,
                preview=preview,
                download_run_id=(
                    stage3_run_id if result["status"] == "downloaded" else None
                ),
            ))
        return tuple(inputs)

    def _stage3_run_artifact(self, stage3_run_id: str, paper_id: str):
        rows = self.database.connection.execute(
            """SELECT DISTINCT a.* FROM download_attempts da
               JOIN download_candidates dc ON dc.candidate_id = da.candidate_id
               JOIN artifacts a ON a.artifact_id = da.artifact_id
               WHERE da.run_id = ? AND dc.paper_id = ?
                 AND da.result_status = 'downloaded'
                 AND a.paper_id = dc.paper_id AND a.artifact_kind = 'pdf'
                 AND a.mime_type = 'application/pdf'
                 AND a.processing_status = 'available'
               ORDER BY a.artifact_id""",
            (stage3_run_id, paper_id),
        ).fetchall()
        if len(rows) != 1:
            raise ValueError(
                "downloaded Stage 3 paper must bind exactly one available PDF artifact"
            )
        return rows[0]

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

    def _input_for(
        self,
        source: Any,
        *,
        extract: bool,
        preview: bool = False,
        download_run_id: str | None = None,
    ) -> AnalysisInput:
        paper = self._paper(str(source["paper_id"]))
        facts = self._facts(
            str(paper["paper_id"]), source, download_run_id=download_run_id
        )
        if "artifact_id" not in source.keys():
            return self._public_fallback_input(paper, facts)
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
            return self._public_fallback_input(paper, facts)
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
        return self._public_fallback_input(paper, facts)

    @staticmethod
    def _public_fallback_input(
        paper: Any, facts: Mapping[str, str | None],
    ) -> AnalysisInput:
        """Use only public abstract/metadata when a PDF has no usable local text."""
        metadata = _metadata(paper)
        if paper["abstract"]:
            return AnalysisInput(
                paper["paper_id"], facts["license"], "public_read_only",
                abstract=paper["abstract"], metadata=metadata, domain=facts["domain"],
            )
        return AnalysisInput(
            paper["paper_id"], facts["license"], "public_read_only",
            metadata=metadata, domain=facts["domain"],
        )

    def _paper(self, paper_id: str):
        row = self.database.connection.execute("SELECT * FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()
        if row is None:
            raise ValueError(f"unknown canonical paper_id: {paper_id}")
        return row

    def _facts(
        self,
        paper_id: str,
        source: Any,
        *,
        download_run_id: str | None = None,
    ) -> Mapping[str, str | None]:
        artifact_id = source["artifact_id"] if "artifact_id" in source.keys() else None
        row = self.database.connection.execute(
            """SELECT dc.license, dc.access_basis, dc.host FROM artifacts a
               LEFT JOIN text_extractions te ON te.output_artifact_id = a.artifact_id
               JOIN download_attempts da ON da.artifact_id = COALESCE(te.source_artifact_id, a.artifact_id)
               JOIN download_candidates dc ON dc.candidate_id = da.candidate_id
               WHERE a.artifact_id = ? AND da.result_status = 'downloaded'
                 AND (? IS NULL OR da.run_id = ?)
               ORDER BY da.attempted_at DESC LIMIT 1""",
            (artifact_id, download_run_id, download_run_id),
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
