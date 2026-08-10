"""Build immutable Stage 4b inputs from persisted search, screening, and analysis state."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from .approval import ApprovalError, require_valid_approval
from .artifacts import ArtifactStore
from .canonical import canonical_json, content_hash
from .report_plan import CorpusPaper, build_corpus_snapshot, build_search_audit_pack
from .schema import SchemaValidationError, validate
from .search_audit import search_audit
from .storage import Database


class ReportInputError(ValueError):
    """Persisted upstream state cannot form a trustworthy Stage 4b input bundle."""


@dataclass(frozen=True, slots=True)
class ReportInputRequest:
    crawl_run_id: str
    filter_run_id: str
    stage4_run_id: str
    recent_cutoff: str
    created_at: str
    include_needs_review: bool = False

    def __post_init__(self) -> None:
        if not all((self.crawl_run_id, self.filter_run_id, self.stage4_run_id)):
            raise ReportInputError("crawl, filter, and Stage 4 run IDs are required")
        try:
            date.fromisoformat(self.recent_cutoff)
        except ValueError as error:
            raise ReportInputError("recent_cutoff must be an ISO date") from error
        try:
            timestamp = datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))
        except ValueError as error:
            raise ReportInputError("created_at must be an ISO date-time") from error
        if timestamp.tzinfo is None:
            raise ReportInputError("created_at must include a timezone")
        if not isinstance(self.include_needs_review, bool):
            raise ReportInputError("include_needs_review must be a boolean")


@dataclass(frozen=True, slots=True)
class ReportInputResult:
    bundle_id: str
    directory: Path
    corpus_snapshot_path: Path
    search_audit_path: Path
    corpus_snapshot: Mapping[str, Any]
    search_audit: Mapping[str, Any]
    saved: bool


class ReportInputService:
    """Freeze report inputs without asking an operator to assemble trusted JSON."""

    def __init__(
        self,
        database: Database,
        artifact_store: ArtifactStore,
        output_root: str | Path,
    ) -> None:
        self.database = database
        self.artifact_store = artifact_store
        self.output_root = Path(output_root)

    def build(
        self, request: ReportInputRequest, *, save_bundle: bool = True
    ) -> ReportInputResult:
        self._require_run(request.filter_run_id, "stage-2", ("complete",))
        stage4_run = self._require_run(
            request.stage4_run_id, "stage4", ("complete", "incomplete", "failed")
        )
        raw_audit = search_audit(self.database.path, request.crawl_run_id)
        query_plan = self._query_plan(request.crawl_run_id, raw_audit)
        decisions = self._filter_decisions(request.filter_run_id)
        unique_after_dedup, seed_only_count = self._require_search_membership(
            request.crawl_run_id, decisions
        )

        included_statuses = {"relevant"}
        if request.include_needs_review:
            included_statuses.add("needs_review")
        included = tuple(row for row in decisions if row["status"] in included_statuses)
        if not included:
            raise ReportInputError("the selected Stage 2 run contains no included papers")

        categories = self._source_categories(
            request.crawl_run_id,
            tuple(str(row["paper_id"]) for row in included),
            query_plan,
        )
        papers = tuple(
            self._corpus_paper(
                str(row["paper_id"]),
                request.stage4_run_id,
                stage4_run,
                categories[str(row["paper_id"])],
                request.recent_cutoff,
            )
            for row in included
        )
        corpus = build_corpus_snapshot(
            papers,
            query_plan_hash=str(raw_audit["plan_hash"]),
            search_audit=raw_audit,
            created_at=request.created_at,
        )
        audit_pack = build_search_audit_pack(
            raw_audit,
            corpus,
            screening_flow=self._screening_flow(
                raw_audit,
                decisions,
                len(included),
                unique_after_dedup,
                seed_only_count,
            ),
            exclusion_reasons=self._exclusion_reasons(decisions, included_statuses),
            required_providers=self._required_providers(query_plan),
            created_at=request.created_at,
        )
        return self._write(corpus, audit_pack, save_bundle=save_bundle)

    def _require_run(
        self, run_id: str, stage: str, statuses: tuple[str, ...]
    ) -> Mapping[str, Any]:
        row = self.database.connection.execute(
            "SELECT * FROM pipeline_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            raise ReportInputError(f"unknown pipeline run: {run_id}")
        if row["stage"] != stage:
            raise ReportInputError(f"pipeline run {run_id} is not a {stage} run")
        if row["status"] not in statuses:
            raise ReportInputError(
                f"pipeline run {run_id} has unusable status: {row['status']}"
            )
        return dict(row)

    def _query_plan(
        self, crawl_run_id: str, raw_audit: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        row = self.database.connection.execute(
            """SELECT sp.content_hash, sp.plan_json
               FROM crawl_runs cr
               JOIN search_plans sp ON sp.search_plan_id = cr.search_plan_id
               WHERE cr.crawl_run_id = ?""",
            (crawl_run_id,),
        ).fetchone()
        if row is None:
            raise ReportInputError("crawl run has no persisted QueryPlan")
        if row["content_hash"] != raw_audit.get("plan_hash"):
            raise ReportInputError("crawl audit and persisted QueryPlan hashes disagree")
        plan = _mapping(row["plan_json"], "persisted QueryPlan")
        try:
            require_valid_approval(plan, "plan_hash")
        except ApprovalError as error:
            raise ReportInputError(
                f"persisted QueryPlan approval is invalid: {error}"
            ) from error
        if plan.get("plan_hash") != row["content_hash"]:
            raise ReportInputError("persisted QueryPlan content hash has drifted")
        return plan

    def _filter_decisions(self, run_id: str) -> tuple[Mapping[str, Any], ...]:
        rows = self.database.connection.execute(
            """SELECT fd.*, p.paper_id AS canonical_paper_id
               FROM filter_decisions fd
               JOIN papers p ON p.paper_id = fd.paper_id
               WHERE fd.run_id = ? ORDER BY fd.paper_id""",
            (run_id,),
        ).fetchall()
        if not rows:
            raise ReportInputError("the selected Stage 2 run has no persisted decisions")
        return tuple(dict(row) for row in rows)

    def _require_search_membership(
        self, crawl_run_id: str, decisions: tuple[Mapping[str, Any], ...]
    ) -> tuple[int, int]:
        discovered_rows = self.database.connection.execute(
            """SELECT paper_id FROM crawl_paper_snapshots WHERE crawl_run_id = ?
               UNION
               SELECT srp.paper_id FROM search_round_papers srp
               JOIN search_rounds sr ON sr.search_round_id = srp.search_round_id
               WHERE sr.crawl_run_id = ?""",
            (crawl_run_id, crawl_run_id),
        ).fetchall()
        seed_rows = self.database.connection.execute(
            """SELECT DISTINCT s.paper_id FROM search_round_seeds s
               JOIN search_rounds r ON r.search_round_id = s.search_round_id
               WHERE r.crawl_run_id = ?""",
            (crawl_run_id,),
        ).fetchall()
        discovered = {str(row["paper_id"]) for row in discovered_rows}
        seeds = {str(row["paper_id"]) for row in seed_rows}
        members = discovered | seeds
        foreign = sorted(str(row["paper_id"]) for row in decisions if row["paper_id"] not in members)
        if foreign:
            raise ReportInputError(
                f"Stage 2 decisions are outside the selected crawl: {foreign}"
            )
        return len(members), len(seeds - discovered)

    def _source_categories(
        self,
        crawl_run_id: str,
        paper_ids: tuple[str, ...],
        query_plan: Mapping[str, Any],
    ) -> dict[str, str]:
        seed_rows = self.database.connection.execute(
            """SELECT DISTINCT s.paper_id
               FROM search_round_seeds s
               JOIN search_rounds r ON r.search_round_id = s.search_round_id
               WHERE r.crawl_run_id = ? AND s.seed_reason = 'user_seed'""",
            (crawl_run_id,),
        ).fetchall()
        user_seeds = {str(row["paper_id"]) for row in seed_rows}
        scope = query_plan.get("scope")
        if isinstance(scope, Mapping):
            planned_seeds = scope.get("user_seeds", ())
            if isinstance(planned_seeds, list):
                user_seeds.update(str(value) for value in planned_seeds)
        snowball_rows = self.database.connection.execute(
            """SELECT DISTINCT srp.paper_id
               FROM search_round_papers srp
               JOIN search_rounds sr ON sr.search_round_id = srp.search_round_id
               WHERE sr.crawl_run_id = ? AND srp.depth > 0""",
            (crawl_run_id,),
        ).fetchall()
        snowball = {str(row["paper_id"]) for row in snowball_rows}
        return {
            paper_id: (
                "user_library"
                if paper_id in user_seeds
                else "citation_snowball"
                if paper_id in snowball
                else "newly_discovered"
            )
            for paper_id in paper_ids
        }

    def _corpus_paper(
        self,
        paper_id: str,
        stage4_run_id: str,
        stage4_run: Mapping[str, Any],
        source_category: str,
        recent_cutoff: str,
    ) -> CorpusPaper:
        paper = self.database.connection.execute(
            "SELECT * FROM papers WHERE paper_id = ?", (paper_id,)
        ).fetchone()
        assert paper is not None
        common = self._paper_metadata(paper, source_category, recent_cutoff)
        analysis = self._complete_analysis(stage4_run_id, paper_id)
        if analysis is None:
            reason, lineage = self._missing_analysis(stage4_run_id, paper_id)
            return CorpusPaper(
                paper_id=paper_id,
                analysis_run_id=None,
                analysis_artifact_hash=None,
                lineage_hashes=lineage,
                publication_status="unknown",
                study_setting="other",
                input_scope="missing",
                evidence_level="metadata_only",
                incomplete_reason=reason,
                **common,
            )

        document, detail, invocation, policy_facts, lineage = self._analysis_documents(
            analysis, stage4_run
        )
        labels = document["labels"]
        evidence_level = {
            "full_pdf": "full_text_direct",
            "abstract_only": "abstract_direct",
            "metadata_only": "metadata_only",
        }[str(analysis["input_scope"])]
        return CorpusPaper(
            paper_id=paper_id,
            analysis_run_id=str(analysis["analysis_run_id"]),
            analysis_artifact_hash=str(analysis["output_sha256"]),
            lineage_hashes=lineage,
            publication_status=str(labels["publication_status"]),
            study_setting=str(labels["study_setting"]),
            input_scope=str(analysis["input_scope"]),
            evidence_level=evidence_level,
            incomplete_reason=None,
            analysis_input_tokens=int(detail["report_input_tokens"]),
            analysis_pipeline_input_hash=str(stage4_run["input_hash"]),
            analysis_config_hash=str(stage4_run["config_hash"]),
            analysis_implementation_version=str(stage4_run["implementation_version"]),
            analysis_prompt_input_hash=str(analysis["input_hash"]),
            analysis_rendered_prompt_hash=str(invocation["rendered_prompt_hash"]),
            analysis_invocation_id=str(invocation["invocation_id"]),
            analysis_policy_facts_hash=content_hash(dict(policy_facts)),
            **common,
        )

    def _complete_analysis(self, run_id: str, paper_id: str) -> Mapping[str, Any] | None:
        row = self.database.connection.execute(
            """SELECT ar.*, d.dispatch_id, d.artifact_hash AS dispatch_artifact_hash,
                      d.artifact_id AS dispatch_artifact_id,
                      d.input_scope AS dispatch_input_scope,
                      d.config_hash AS dispatch_config_hash,
                      d.implementation_version AS dispatch_implementation_version,
                      d.profile AS dispatch_profile, d.model_id AS dispatch_model_id,
                      d.prompt_hash AS dispatch_prompt_hash,
                      d.schema_hash AS dispatch_schema_hash,
                      d.prompt_input_hash AS dispatch_prompt_input_hash,
                      d.rendered_prompt_hash AS dispatch_rendered_prompt_hash,
                      d.invocation_id AS dispatch_invocation_id,
                      d.processing_decision_json AS dispatch_processing_decision_json,
                      d.invocation_metadata_json AS dispatch_invocation_metadata_json,
                      output.sha256 AS output_sha256,
                      output.relative_path AS output_relative_path,
                      output.mime_type AS output_mime_type,
                      output.byte_size AS output_byte_size,
                      output.artifact_kind AS output_artifact_kind,
                      output.paper_id AS output_paper_id,
                      output.processing_status AS output_processing_status,
                      output.provenance_json AS output_provenance_json
               FROM analysis_dispatches d
               JOIN analysis_runs ar ON ar.analysis_run_id = d.analysis_run_id
               JOIN artifacts output ON output.artifact_id = ar.output_artifact_id
               WHERE d.run_id = ? AND d.paper_id = ?
                 AND d.status = 'complete' AND ar.status = 'complete'""",
            (run_id, paper_id),
        ).fetchone()
        return dict(row) if row is not None else None

    def _analysis_documents(
        self, analysis: Mapping[str, Any], stage4_run: Mapping[str, Any]
    ) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], tuple[str, ...]]:
        output_hash = str(analysis["output_sha256"])
        payload = self.artifact_store.read_bytes(output_hash)
        if (
            analysis["output_artifact_kind"] != "analysis"
            or analysis["output_mime_type"] != "application/json"
            or analysis["output_processing_status"] != "available"
            or analysis["output_paper_id"] != analysis["paper_id"]
            or analysis["output_relative_path"] != self.artifact_store.relative_path(output_hash)
            or int(analysis["output_byte_size"]) != len(payload)
        ):
            raise ReportInputError("Stage 4 output artifact metadata has drifted")
        document = _mapping(payload, "Stage 4 analysis artifact")
        try:
            validate(document, "paper-analysis.schema.json")
        except SchemaValidationError as error:
            raise ReportInputError(str(error)) from error
        detail = _mapping(analysis["invocation_metadata_json"], "Stage 4 invocation metadata")
        invocation = _object(detail.get("invocation"), "Stage 4 invocation")
        policy_facts = _object(detail.get("input_policy_facts"), "Stage 4 input policy facts")
        decision = _object(detail.get("processing_decision"), "Stage 4 processing decision")
        dispatch_decision = _mapping(
            analysis["dispatch_processing_decision_json"],
            "Stage 4 dispatch processing decision",
        )
        dispatch_invocation = _mapping(
            analysis["dispatch_invocation_metadata_json"],
            "Stage 4 dispatch invocation metadata",
        )
        provenance = _mapping(analysis["output_provenance_json"], "Stage 4 output provenance")
        source_hash = str(policy_facts.get("artifact_hash") or "")
        expected = (
            analysis["paper_id"],
            analysis["dispatch_artifact_hash"],
            analysis["input_scope"],
            analysis["model_id"],
            analysis["prompt_hash"],
            analysis["schema_hash"],
            analysis["implementation_version"],
            analysis["input_hash"],
            invocation.get("rendered_prompt_hash"),
            invocation.get("invocation_id"),
        )
        actual = (
            document.get("paper_id"),
            document.get("artifact_hash"),
            document.get("input_scope"),
            document.get("model"),
            document.get("prompt_hash"),
            document.get("schema_hash"),
            analysis["dispatch_implementation_version"],
            analysis["dispatch_prompt_input_hash"],
            analysis["dispatch_rendered_prompt_hash"],
            analysis["dispatch_invocation_id"],
        )
        if expected != actual:
            raise ReportInputError("Stage 4 analysis and dispatch bindings disagree")
        if (
            source_hash != document["artifact_hash"]
            or decision.get("input_artifact_hash") != source_hash
            or policy_facts.get("paper_id") != analysis["paper_id"]
            or policy_facts.get("input_scope") != analysis["input_scope"]
            or analysis["dispatch_artifact_id"] != analysis["artifact_id"]
            or analysis["dispatch_input_scope"] != analysis["input_scope"]
            or analysis["dispatch_config_hash"] != stage4_run["config_hash"]
            or analysis["dispatch_implementation_version"] != stage4_run["implementation_version"]
            or analysis["dispatch_profile"] != "stage4_analysis_luna"
            or analysis["dispatch_model_id"] != analysis["model_id"]
            or analysis["dispatch_prompt_hash"] != analysis["prompt_hash"]
            or analysis["dispatch_schema_hash"] != analysis["schema_hash"]
            or invocation.get("input_hash") != analysis["input_hash"]
            or invocation.get("profile") != "stage4_analysis_luna"
            or invocation.get("model") != "gpt-5.6-luna"
            or invocation.get("actual_model") != "gpt-5.6-luna"
            or invocation.get("actual_profile") != "stage4_analysis_luna"
            or invocation.get("prompt_hash") != analysis["prompt_hash"]
            or invocation.get("schema_hash") != analysis["schema_hash"]
            or document.get("model_revision") != analysis["model_revision"]
            or dict(dispatch_invocation) != dict(invocation)
            or dict(dispatch_decision) != dict(decision)
            or provenance.get("analysis_run_id") != analysis["analysis_run_id"]
            or provenance.get("stage") != "stage4"
            or not isinstance(detail.get("report_input_tokens"), int)
            or detail["report_input_tokens"] < 1
        ):
            raise ReportInputError("Stage 4 runtime provenance is incomplete or has drifted")
        lineage = self._lineage(
            str(analysis["paper_id"]), analysis["artifact_id"], source_hash
        )
        return document, detail, invocation, policy_facts, lineage

    def _missing_analysis(self, run_id: str, paper_id: str) -> tuple[str, tuple[str, ...]]:
        dispatch = self.database.connection.execute(
            """SELECT status, artifact_id, artifact_hash FROM analysis_dispatches
               WHERE run_id = ? AND paper_id = ?""",
            (run_id, paper_id),
        ).fetchone()
        if dispatch is not None:
            if dispatch["status"] == "complete":
                raise ReportInputError(
                    "complete Stage 4 dispatch lacks its complete analysis artifact"
                )
            return (
                f"stage4_{dispatch['status']}",
                self._lineage(paper_id, dispatch["artifact_id"], str(dispatch["artifact_hash"])),
            )
        analysis = self.database.connection.execute(
            """SELECT status FROM analysis_runs WHERE run_id = ? AND paper_id = ?
               ORDER BY created_at DESC, analysis_run_id DESC LIMIT 1""",
            (run_id, paper_id),
        ).fetchone()
        reason = (
            "stage4_complete_dispatch_missing"
            if analysis is not None and analysis["status"] == "complete"
            else f"stage4_{analysis['status']}"
            if analysis is not None
            else "stage4_analysis_missing"
        )
        return reason, ()

    def _lineage(
        self, paper_id: str, artifact_id: str | None, source_hash: str
    ) -> tuple[str, ...]:
        if not _is_sha256(source_hash):
            raise ReportInputError("Stage 4 source lineage lacks a SHA-256 hash")
        hashes = {source_hash}
        if artifact_id is None:
            return tuple(sorted(hashes))
        artifact = self.database.connection.execute(
            """SELECT paper_id, artifact_kind, relative_path, mime_type,
                      byte_size, sha256, processing_status
               FROM artifacts WHERE artifact_id = ?""",
            (artifact_id,),
        ).fetchone()
        if (
            artifact is None
            or artifact["paper_id"] != paper_id
            or artifact["sha256"] != source_hash
            or artifact["processing_status"] != "available"
        ):
            raise ReportInputError("Stage 4 source artifact binding has drifted")
        payload = self.artifact_store.read_bytes(source_hash)
        if (
            artifact["relative_path"] != self.artifact_store.relative_path(source_hash)
            or int(artifact["byte_size"]) != len(payload)
        ):
            raise ReportInputError("Stage 4 source artifact metadata has drifted")
        if artifact["artifact_kind"] == "text":
            rows = self.database.connection.execute(
                """SELECT te.source_sha256, te.source_artifact_id,
                          source.paper_id, source.artifact_kind, source.relative_path,
                          source.mime_type, source.byte_size, source.sha256,
                          source.processing_status
                   FROM text_extractions te
                   JOIN artifacts source ON source.artifact_id = te.source_artifact_id
                   WHERE te.paper_id = ? AND te.output_artifact_id = ?
                     AND te.status = 'full_text_ready'""",
                (paper_id, artifact_id),
            ).fetchall()
            source_hashes = {str(row["source_sha256"]) for row in rows}
            if len(source_hashes) != 1 or not all(_is_sha256(value) for value in source_hashes):
                raise ReportInputError("normalized text lacks one exact source PDF lineage")
            source = rows[0]
            pdf_hash = str(source["source_sha256"])
            pdf_payload = self.artifact_store.read_bytes(pdf_hash)
            if (
                source["paper_id"] != paper_id
                or source["artifact_kind"] != "pdf"
                or source["mime_type"] != "application/pdf"
                or source["sha256"] != pdf_hash
                or source["processing_status"] != "available"
                or source["relative_path"] != self.artifact_store.relative_path(pdf_hash)
                or int(source["byte_size"]) != len(pdf_payload)
            ):
                raise ReportInputError("normalized text source PDF lineage has drifted")
            hashes.update(source_hashes)
        return tuple(sorted(hashes))

    @staticmethod
    def _paper_metadata(
        paper: Mapping[str, Any], source_category: str, recent_cutoff: str
    ) -> dict[str, Any]:
        authors = json.loads(str(paper["authors_json"]))
        if not isinstance(authors, list) or not all(isinstance(author, str) for author in authors):
            raise ReportInputError("canonical paper authors are malformed")
        return {
            "source_category": source_category,
            "foundational": source_category == "user_library",
            "recent": _is_recent(paper["publication_date"], paper["year"], recent_cutoff),
            "publication_date": paper["publication_date"],
            "publication_year": paper["year"],
            "venue_id": paper["venue_id"],
            "venue_name": paper["venue_name"],
            "title": paper["title"],
            "authors": tuple(authors),
            "doi": paper["doi"],
            "canonical_url": paper["canonical_url"],
            "verification_status": paper["verification_status"],
        }

    @staticmethod
    def _screening_flow(
        raw_audit: Mapping[str, Any],
        decisions: tuple[Mapping[str, Any], ...],
        included: int,
        unique_after_dedup: int,
        seed_only_count: int,
    ) -> dict[str, int]:
        totals = _object(raw_audit.get("totals"), "search audit totals")
        sources = _object(totals.get("sources"), "search source totals")
        rounds = _object(totals.get("citation_rounds"), "citation round totals")
        raw = (
            int(sources.get("raw_discovered", 0))
            + int(rounds.get("raw_discovered", 0))
            + seed_only_count
        )
        screened = len(decisions)
        if raw < unique_after_dedup or unique_after_dedup < screened:
            raise ReportInputError("search audit totals do not reconcile with Stage 2 decisions")
        return {
            "raw_discovered": raw,
            "unique_after_dedup": unique_after_dedup,
            "stage2_screened": screened,
            "included": included,
        }

    @staticmethod
    def _exclusion_reasons(
        decisions: tuple[Mapping[str, Any], ...], included_statuses: set[str]
    ) -> dict[str, int]:
        reasons: Counter[str] = Counter()
        for decision in decisions:
            if decision["status"] in included_statuses:
                continue
            detail = _mapping(decision["reason"], "Stage 2 decision reason")
            reason = detail.get("reason_code")
            if not isinstance(reason, str) or not reason:
                raise ReportInputError("Stage 2 exclusion lacks a reason_code")
            reasons[reason] += 1
        return dict(sorted(reasons.items()))

    @staticmethod
    def _required_providers(query_plan: Mapping[str, Any]) -> tuple[str, ...]:
        execution = _object(query_plan.get("execution"), "QueryPlan execution")
        values = execution.get("required_providers")
        if not isinstance(values, list) or not all(isinstance(value, str) and value for value in values):
            raise ReportInputError("QueryPlan required_providers is malformed")
        return tuple(sorted(set(values)))

    def _write(
        self,
        corpus: Mapping[str, Any],
        audit: Mapping[str, Any],
        *,
        save_bundle: bool,
    ) -> ReportInputResult:
        bundle_hash = content_hash(
            {
                "corpus_snapshot_hash": corpus["snapshot_hash"],
                "search_audit_pack_hash": audit["pack_hash"],
            }
        )
        bundle_id = f"report-input-{bundle_hash[:12]}"
        directory = self.output_root / "reports" / "inputs" / bundle_id
        corpus_path = directory / "CORPUS_SNAPSHOT.json"
        audit_path = directory / "SEARCH_AUDIT.json"
        documents = {
            corpus_path: canonical_json(dict(corpus)),
            audit_path: canonical_json(dict(audit)),
        }
        for path, payload in documents.items():
            if path.exists() and path.read_bytes() != payload:
                raise ReportInputError(f"report input is immutable: {path.name}")
        for path, payload in documents.items():
            if save_bundle and not path.exists():
                _atomic_write(path, payload)
        return ReportInputResult(
            bundle_id,
            directory,
            corpus_path,
            audit_path,
            corpus,
            audit,
            save_bundle,
        )


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise ReportInputError(f"{name} is not valid JSON") from error
    if not isinstance(value, Mapping):
        raise ReportInputError(f"{name} must be a JSON object")
    return value


def _object(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReportInputError(f"{name} must be an object")
    return value


def _is_recent(publication_date: object, year: object, cutoff: str) -> bool:
    threshold = date.fromisoformat(cutoff)
    if publication_date:
        try:
            return date.fromisoformat(str(publication_date)[:10]) >= threshold
        except ValueError:
            pass
    if isinstance(year, int):
        return date(year, 1, 1) >= threshold
    return False


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temporary.write(payload)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)
