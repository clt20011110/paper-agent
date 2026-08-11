"""Stage 4's policy-gated, per-paper analysis coordinator.

This module owns the boundary between prepared paper inputs and ``codex
exec``.  In particular, an executor is made only inside the callback passed to
``ProcessingGate.dispatch``: a policy denial therefore cannot create an
executor, let alone make a model call.
"""

from __future__ import annotations

from base64 import b64encode
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re
from typing import Any, Protocol
from uuid import uuid4

from pypdf import PdfReader

from .analysis_dispatches import (
    AnalysisDispatchBinding,
    AnalysisDispatchClaim,
    AnalysisDispatchRecord,
    AnalysisDispatchStatus,
    AnalysisDispatchStore,
)
from .analysis_registry import AnalysisNormalizationRegistry
from .artifacts import ArtifactStore
from .canonical import content_hash
from .codex_exec import (
    CodexExec,
    CodexExecRequest,
    CodexExecResult,
    InvocationMetadata,
    prepare_service_schema,
    prompt_directory,
)
from .grants import GrantStore
from .processing import (
    ModelInvocation,
    ProcessingDecision,
    ProcessingGate,
    ProcessingOutcome,
    ProcessingRequest,
)
from .schema import SchemaValidationError, schema_directory, validate
from .storage import Database


ANALYSIS_PROFILE = "stage4_analysis_luna"
ANALYSIS_SCHEMA = "paper-analysis.schema.json"
ANALYSIS_PROMPT = "paper-analysis.md"
IMPLEMENTATION_VERSION = "phase5-stage4-v3"
ANALYSIS_DISPATCH_LEASE_SECONDS = 900
PAGE_MARKER = re.compile(r"(?:^|\n\n)===== PAGE ([1-9][0-9]*) =====\n\n")


class AnalysisValidationError(ValueError):
    """A model result is valid JSON but is not a valid bound paper analysis."""


class _DispatchClaimLost(RuntimeError):
    """Another coordinator already consumed the single dispatch claim."""


class AnalysisInvoker(Protocol):
    def invoke(self, request: CodexExecRequest) -> CodexExecResult: ...


def load_analysis_output_schema(
    output_schema_path: str | Path | None = None,
) -> tuple[Path, dict[str, Any], str]:
    """Load the configured Stage 4 schema and prove it is the frozen schema.

    The configured path is part of the Stage 4 run identity.  Its content must
    still match the schema shipped with Paper Agent, otherwise the run fails
    before an executor can be constructed.
    """
    frozen_path = (schema_directory() / ANALYSIS_SCHEMA).resolve()
    configured_path = Path(output_schema_path) if output_schema_path is not None else frozen_path
    configured_path = configured_path.resolve()
    if not configured_path.is_file():
        raise AnalysisValidationError(
            f"configured analysis output schema is missing: {configured_path}"
        )
    try:
        configured = json.loads(configured_path.read_text(encoding="utf-8"))
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AnalysisValidationError("configured analysis output schema is unreadable") from error
    if not isinstance(configured, dict) or not isinstance(frozen, dict):
        raise AnalysisValidationError("analysis output schema must be a JSON object")
    schema_hash = _digest_json(configured)
    if schema_hash != _digest_json(frozen):
        raise AnalysisValidationError(
            "configured analysis output schema does not match the frozen schema"
        )
    return configured_path, configured, schema_hash


def analysis_configuration_denial(
    gate: ProcessingGate,
    request: ProcessingRequest,
    *,
    allow_abstract_only: bool,
) -> ProcessingDecision | None:
    """Return the configuration-level denial that precedes policy dispatch."""
    if allow_abstract_only or request.input_scope != "abstract_only":
        return None
    return ProcessingDecision(
        policy_version=gate.policy.version,
        policy_hash=gate.policy.hash,
        outcome=ProcessingOutcome.MANUAL,
        reason_code="abstract_only_disabled_by_analysis_config",
        input_artifact_hash=request.artifact_hash,
        provider=request.provider,
        model=request.model,
        purpose=request.purpose,
        data_category=request.data_category,
        processing_grant_id=None,
        authorized_by=None,
    )


@dataclass(frozen=True, slots=True)
class AnalysisInput:
    """One selected input scope for a paper.

    Exactly one content source is selected.  Metadata accompanying an abstract
    is folded into the abstract payload before the gate, so the gate still
    exposes only one authorized abstract artifact to the model.
    """

    paper_id: str
    license: str | None
    access_basis: str
    full_pdf: bytes | None = None
    normalized_text: str | bytes | None = None
    abstract: str | None = None
    metadata: Mapping[str, Any] | None = None
    artifact_id: str | None = None
    domain: str | None = None
    collection_id: str | None = None
    collection_snapshot_hash: str | None = None
    selection_snapshot_hash: str | None = None
    mode: str = "attended"

    def __post_init__(self) -> None:
        if not self.paper_id:
            raise ValueError("paper_id is required")
        selected = sum(value is not None for value in (self.full_pdf, self.normalized_text, self.abstract))
        if selected > 1 or (selected == 0 and self.metadata is None):
            raise ValueError("select exactly one of full_pdf, normalized_text, abstract, or metadata")

    def processing_request(self) -> ProcessingRequest:
        common = {
            "license": self.license, "access_basis": self.access_basis,
            "purpose": "internal_analysis", "paper_id": self.paper_id,
            "domain": self.domain, "mode": self.mode, "collection_id": self.collection_id,
            "collection_snapshot_hash": self.collection_snapshot_hash,
            "selection_snapshot_hash": self.selection_snapshot_hash,
        }
        if self.full_pdf is not None:
            return ProcessingRequest(
                artifact_hash=sha256(self.full_pdf).hexdigest(), artifact="pdf", input_scope="full_pdf",
                data_category="full_text", pdf_bytes=self.full_pdf, **common,
            )
        if self.normalized_text is not None:
            payload = self.normalized_text.encode("utf-8") if isinstance(self.normalized_text, str) else self.normalized_text
            return ProcessingRequest(
                artifact_hash=sha256(payload).hexdigest(), artifact="normalized_text", input_scope="full_pdf",
                data_category="normalized_text", normalized_text_bytes=payload, **common,
            )
        if self.abstract is not None:
            # Abstract-only policy must still cover title/keywords supplied as
            # metadata.  The content-addressed abstract artifact is this exact
            # canonical wrapper, not an unbound side channel.
            payload = _json_bytes({"abstract": self.abstract, "metadata": dict(self.metadata or {})})
            return ProcessingRequest(
                artifact_hash=sha256(payload).hexdigest(), artifact="abstract", input_scope="abstract_only",
                data_category="abstract", abstract_bytes=payload, **common,
            )
        assert self.metadata is not None
        frozen_metadata = dict(self.metadata)
        return ProcessingRequest(
            artifact_hash=content_hash(frozen_metadata), artifact="metadata", input_scope="metadata_only",
            data_category="metadata", metadata=frozen_metadata, **common,
        )


@dataclass(frozen=True, slots=True)
class AnalysisPaperResult:
    paper_id: str
    analysis_run_id: str
    status: str
    input_hash: str
    input_scope: str
    decision: ProcessingDecision | None = None
    output: Mapping[str, Any] | None = None
    resumed: bool = False
    error: str | None = None
    markdown_artifact_id: str | None = None


@dataclass(frozen=True, slots=True)
class AnalysisRunResult:
    papers: tuple[AnalysisPaperResult, ...]

    def for_paper(self, paper_id: str) -> AnalysisPaperResult:
        for result in self.papers:
            if result.paper_id == paper_id:
                return result
        raise KeyError(paper_id)


class PaperAnalysisCoordinator:
    """Persist isolated Stage 4 work while keeping input/config bindings frozen."""

    def __init__(
        self,
        database: Database,
        artifact_store: ArtifactStore,
        gate: ProcessingGate,
        *,
        invoker_factory: Callable[[], AnalysisInvoker] = CodexExec,
        normalization_registry: AnalysisNormalizationRegistry | None = None,
        implementation_version: str = IMPLEMENTATION_VERSION,
        dispatch_store: AnalysisDispatchStore | None = None,
        clock: Callable[[], datetime] | None = None,
        dispatch_lease_seconds: int = ANALYSIS_DISPATCH_LEASE_SECONDS,
        workers: int = 1,
        allow_abstract_only: bool = True,
        output_schema_path: str | Path | None = None,
    ) -> None:
        if dispatch_lease_seconds <= 0:
            raise ValueError("dispatch_lease_seconds must be positive")
        if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
            raise ValueError("analysis workers must be a positive integer")
        if not isinstance(allow_abstract_only, bool):
            raise ValueError("allow_abstract_only must be a boolean")
        if workers > 1 and dispatch_store is not None:
            raise ValueError("custom analysis dispatch stores require workers=1")
        self.database = database
        self.artifact_store = artifact_store
        self.gate = gate
        self.invoker_factory = invoker_factory
        self.normalization_registry = normalization_registry or AnalysisNormalizationRegistry.load()
        self.implementation_version = implementation_version
        self.dispatch_store = dispatch_store or AnalysisDispatchStore(database)
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.dispatch_lease_seconds = dispatch_lease_seconds
        self.workers = workers
        self.allow_abstract_only = allow_abstract_only
        self.schema_path, self.schema, self.schema_hash = load_analysis_output_schema(
            output_schema_path
        )
        self.service_schema_hash = content_hash(prepare_service_schema(
            ANALYSIS_SCHEMA,
            self.schema,
            schema_root=schema_directory(),
        ))
        self.prompt_hash = sha256(
            (prompt_directory() / ANALYSIS_PROMPT).read_bytes()
        ).hexdigest()
        legacy_config = {
            "profile": ANALYSIS_PROFILE, "model": "gpt-5.6-luna", "prompt_hash": self.prompt_hash,
            "schema_hash": self.schema_hash, "implementation_version": implementation_version,
            "normalization_registry": self.normalization_registry.registry_hash,
        }
        # Preserve the exact pre-ledger identity solely for migration-016
        # compatibility checks.  Every newly created/resumed dispatch freezes
        # the policy as part of both the pipeline and paid-call identities.
        self.legacy_config_hash = content_hash(legacy_config)
        policy_config = {
            **legacy_config,
            "service_schema_hash": self.service_schema_hash,
            "processing_policy_version": self.gate.policy.version,
            "processing_policy_hash": self.gate.policy.hash,
        }
        self.pre_analysis_config_hash = content_hash(policy_config)
        self.config_hash = content_hash({
            **policy_config,
            "allow_abstract_only": self.allow_abstract_only,
            "output_schema_path": str(self.schema_path),
        })

    def run(
        self,
        run_id: str,
        papers: Sequence[AnalysisInput],
        *,
        now: datetime | str | None = None,
        processing_grant_id: str | None = None,
    ) -> AnalysisRunResult:
        if not run_id:
            raise ValueError("run_id is required")
        if len({paper.paper_id for paper in papers}) != len(papers):
            raise ValueError("a Stage 4 run cannot contain duplicate paper_ids")
        requests = tuple(paper.processing_request() for paper in papers)
        self._ensure_run(run_id, requests)
        grant_paper_count = 1
        if processing_grant_id is not None:
            grant_paper_count = max(1, sum(
                analysis_configuration_denial(
                    self.gate,
                    request,
                    allow_abstract_only=self.allow_abstract_only,
                ) is None
                and not self.gate.decide(request).is_authorized
                for request in requests
            ))
        work = tuple(zip(papers, requests, strict=True))
        if self.workers == 1 or len(work) < 2:
            results = [
                self._run_one_isolated(
                    run_id,
                    paper,
                    request,
                    now,
                    processing_grant_id,
                    grant_paper_count,
                )
                for paper, request in work
            ]
        else:
            # Futures are consumed in submission order so completion timing
            # cannot reorder the frozen paper list.
            with ThreadPoolExecutor(
                max_workers=min(self.workers, len(work)),
                thread_name_prefix="paper-agent-stage4",
            ) as executor:
                futures = [
                    executor.submit(
                        self._run_one_in_worker,
                        run_id,
                        paper,
                        request,
                        now,
                        processing_grant_id,
                        grant_paper_count,
                    )
                    for paper, request in work
                ]
                results = [future.result() for future in futures]
        status = "complete" if all(result.status == "complete" for result in results) else "incomplete"
        with self.database.transaction() as connection:
            terminal = connection.execute(
                """SELECT 1 FROM analysis_dispatches
                   WHERE run_id = ? AND status = 'failed_terminal' LIMIT 1""",
                (run_id,),
            ).fetchone()
            if terminal is not None:
                status = "failed"
            connection.execute(
                """UPDATE pipeline_runs
                   SET status = CASE
                           WHEN ? = 'failed' THEN 'failed'
                           WHEN status IN ('complete', 'failed') THEN status
                           ELSE ?
                       END,
                       completed_at = CURRENT_TIMESTAMP
                   WHERE run_id = ?""",
                (status, status, run_id),
            )
        return AnalysisRunResult(tuple(results))

    def _run_one_isolated(
        self,
        run_id: str,
        paper: AnalysisInput,
        request: ProcessingRequest,
        now: datetime | str | None,
        processing_grant_id: str | None,
        grant_paper_count: int,
    ) -> AnalysisPaperResult:
        try:
            return self._run_one(
                run_id,
                paper,
                request,
                now,
                processing_grant_id,
                grant_paper_count,
            )
        except Exception as error:  # One malformed/model-failed paper cannot stop the batch.
            return self._record_unexpected_failure(run_id, paper, request, error)

    def _run_one_in_worker(
        self,
        run_id: str,
        paper: AnalysisInput,
        request: ProcessingRequest,
        now: datetime | str | None,
        processing_grant_id: str | None,
        grant_paper_count: int,
    ) -> AnalysisPaperResult:
        """Run one paper with a thread-local SQLite connection."""
        with Database(self.database.path) as database:
            grants = GrantStore(database) if self.gate.grants is not None else None
            worker = PaperAnalysisCoordinator(
                database,
                self.artifact_store,
                ProcessingGate(self.gate.policy, grants),
                invoker_factory=self.invoker_factory,
                normalization_registry=self.normalization_registry,
                implementation_version=self.implementation_version,
                clock=self.clock,
                dispatch_lease_seconds=self.dispatch_lease_seconds,
                workers=self.workers,
                allow_abstract_only=self.allow_abstract_only,
                output_schema_path=self.schema_path,
            )
            if worker.config_hash != self.config_hash:
                raise AnalysisValidationError("Stage 4 worker configuration drifted")
            return worker._run_one_isolated(
                run_id,
                paper,
                request,
                now,
                processing_grant_id,
                grant_paper_count,
            )

    def _ensure_run(self, run_id: str, requests: Sequence[ProcessingRequest]) -> None:
        input_hash = content_hash([{
            "paper_id": request.paper_id, "artifact_hash": request.artifact_hash,
            "input_scope": request.input_scope, "artifact": request.artifact,
            "license": request.license, "access_basis": request.access_basis,
            "data_category": request.data_category, "mode": request.mode,
        } for request in sorted(requests, key=lambda item: item.paper_id)])
        with self.database.transaction() as connection:
            row = connection.execute(
                "SELECT stage, input_hash, config_hash, implementation_version FROM pipeline_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            expected = ("stage4", input_hash, self.config_hash, self.implementation_version)
            if row is not None:
                actual = tuple(row[key] for key in ("stage", "input_hash", "config_hash", "implementation_version"))
                if actual == expected:
                    return
                previous = (
                    "stage4", input_hash, self.pre_analysis_config_hash,
                    self.implementation_version,
                )
                frozen_schema_path = (schema_directory() / ANALYSIS_SCHEMA).resolve()
                if (
                    actual == previous
                    and self.allow_abstract_only
                    and self.schema_path == frozen_schema_path
                ):
                    connection.execute(
                        "UPDATE pipeline_runs SET config_hash = ? WHERE run_id = ?",
                        (self.config_hash, run_id),
                    )
                    return
                legacy = ("stage4", input_hash, self.legacy_config_hash, self.implementation_version)
                if actual != legacy or not self._legacy_policy_compatible(
                    connection, run_id, requests,
                ):
                    raise ValueError("Stage 4 run input or configuration is immutable")
                connection.execute(
                    """UPDATE pipeline_runs SET config_hash = ?
                       WHERE run_id = ? AND config_hash = ?""",
                    (self.config_hash, run_id, self.legacy_config_hash),
                )
                return
            connection.execute(
                """INSERT INTO pipeline_runs(run_id, stage, status, input_hash, config_hash, implementation_version, started_at)
                   VALUES (?, 'stage4', 'running', ?, ?, ?, CURRENT_TIMESTAMP)""",
                (run_id, input_hash, self.config_hash, self.implementation_version),
            )

    def _legacy_policy_compatible(
        self,
        connection: Any,
        run_id: str,
        requests: Sequence[ProcessingRequest],
    ) -> bool:
        """Adopt only verifiably same-policy v15 work or terminal tombstones."""
        for request in requests:
            assert request.paper_id is not None
            dispatch_row = connection.execute(
                """SELECT * FROM analysis_dispatches
                   WHERE run_id = ? AND paper_id = ?""",
                (run_id, request.paper_id),
            ).fetchone()
            if dispatch_row is not None:
                dispatch = self.dispatch_store.find(
                    run_id, request.paper_id, connection=connection,
                )
                assert dispatch is not None
                if (
                    dispatch.status is AnalysisDispatchStatus.FAILED_TERMINAL
                    and dispatch.dispatch_id.startswith("analysis-dispatch-legacy-")
                    and dispatch.artifact_hash == request.artifact_hash
                ):
                    if dispatch.processing_decision is None:
                        # No policy proof exists, so the migration consumed the
                        # dispatch budget permanently.  This is safe to adopt
                        # only because no execution path can reopen it.
                        continue
                    if self._decision_matches_current_policy(dispatch.processing_decision):
                        continue
                return False

            row = self._legacy_analysis_row(connection, run_id, request)
            if row is None:
                return False
            detail = json.loads(row["invocation_metadata_json"] or "{}")
            decision = detail.get("processing_decision")
            if not isinstance(decision, Mapping) or not self._decision_matches_current_policy(decision):
                return False
        return True

    def _legacy_analysis_row(
        self,
        connection: Any,
        run_id: str,
        request: ProcessingRequest,
    ) -> Any | None:
        rows = connection.execute(
            """SELECT * FROM analysis_runs WHERE run_id = ? AND paper_id = ?
               ORDER BY CASE status WHEN 'complete' THEN 0 ELSE 1 END,
                        created_at DESC, analysis_run_id DESC""",
            (run_id, request.paper_id),
        ).fetchall()
        for row in rows:
            detail = json.loads(row["invocation_metadata_json"] or "{}")
            facts = detail.get("input_policy_facts", {})
            decision = detail.get("processing_decision", {})
            if (
                isinstance(facts, Mapping)
                and facts.get("artifact_hash") == request.artifact_hash
            ) or (
                isinstance(decision, Mapping)
                and decision.get("input_artifact_hash") == request.artifact_hash
            ):
                return row
        return None

    def _decision_matches_current_policy(self, decision: Mapping[str, Any]) -> bool:
        return (
            decision.get("policy_version") == self.gate.policy.version
            and decision.get("policy_hash") == self.gate.policy.hash
        )

    def _run_one(
        self, run_id: str, paper: AnalysisInput, request: ProcessingRequest,
        now: datetime | str | None, processing_grant_id: str | None,
        grant_paper_count: int,
    ) -> AnalysisPaperResult:
        existing = self._existing(run_id, paper.paper_id, request.artifact_hash)
        if existing is not None:
            self._assert_existing_binding(existing, paper, request)
            if existing["status"] == "complete":
                ledger = self.database.connection.execute(
                    "SELECT 1 FROM analysis_dispatches WHERE run_id = ? AND paper_id = ?",
                    (run_id, paper.paper_id),
                ).fetchone()
                if ledger is None:  # Completed before migration 016; preserve free legacy resume.
                    return AnalysisPaperResult(
                        paper.paper_id, existing["analysis_run_id"], "complete", existing["input_hash"],
                        existing["input_scope"], output=self._load_output(existing), resumed=True,
                        markdown_artifact_id=existing["markdown_artifact_id"],
                    )

        binding = AnalysisDispatchBinding(
            run_id=run_id,
            paper_id=paper.paper_id,
            artifact_hash=request.artifact_hash,
            artifact_id=paper.artifact_id,
            input_scope=request.input_scope,
            config_hash=self.config_hash,
            implementation_version=self.implementation_version,
            profile=ANALYSIS_PROFILE,
            model_id="gpt-5.6-luna",
            prompt_hash=self.prompt_hash,
            schema_hash=self.schema_hash,
            policy_version=self.gate.policy.version,
            policy_hash=self.gate.policy.hash,
        )
        persisted = self.dispatch_store.find(run_id, paper.paper_id)
        if persisted is not None and persisted.status is AnalysisDispatchStatus.FAILED_TERMINAL:
            self.dispatch_store.assert_binding(
                persisted,
                binding,
                allow_legacy_terminal=True,
                legacy_config_hash=self.legacy_config_hash,
            )
            current, terminal_result = self._observe_dispatch(persisted, paper, request)
            if terminal_result is None:  # Defensive: failed_terminal is immutable.
                raise AnalysisValidationError("terminal analysis dispatch was reopened")
            return terminal_result
        dispatch = self.dispatch_store.prepare(
            binding,
            stable_created_at=now if now is not None else self.clock(),
        )
        dispatch, terminal_result = self._observe_dispatch(dispatch, paper, request)
        if terminal_result is not None:
            return terminal_result
        if existing is not None and existing["status"] == "complete" and (
            dispatch.status is not AnalysisDispatchStatus.COMPLETE
            or dispatch.analysis_run_id != existing["analysis_run_id"]
        ):
            raise AnalysisValidationError("completed analysis run conflicts with its dispatch ledger")
        if dispatch.status is AnalysisDispatchStatus.COMPLETE:
            return self._result_from_dispatch(dispatch, paper, request)
        if dispatch.status is AnalysisDispatchStatus.RUNNING:
            return self._result_from_dispatch(dispatch, paper, request)

        configuration_denial = analysis_configuration_denial(
            self.gate,
            request,
            allow_abstract_only=self.allow_abstract_only,
        )
        if configuration_denial is not None:
            return self._persist_not_authorized(
                run_id, paper, request, configuration_denial, dispatch
            )

        captured: ModelInvocation | None = None
        sent_hash: str | None = None
        metadata: InvocationMetadata | None = None
        analysis_output: Mapping[str, Any] | None = None
        claim: AnalysisDispatchClaim | None = None
        created_at = dispatch.stable_created_at
        owner = f"analysis-worker-{uuid4()}"

        def invoke(invocation: ModelInvocation) -> CodexExecResult:
            nonlocal captured, sent_hash, metadata, analysis_output, claim
            captured = invocation
            payload = _authorized_payload(
                paper.paper_id,
                request.artifact_hash,
                request.input_scope,
                invocation,
                {
                    "paper_id": paper.paper_id,
                    "artifact_hash": request.artifact_hash,
                    "input_scope": request.input_scope,
                    "model": "gpt-5.6-luna",
                    "prompt_hash": self.prompt_hash,
                    "schema_hash": self.schema_hash,
                    "created_at": created_at,
                },
            )
            prompt = _json_text(payload)
            sent_hash = sha256(prompt.encode("utf-8")).hexdigest()
            claim = self.dispatch_store.claim(
                dispatch.dispatch_id,
                invocation.decision,
                owner=owner,
                prompt_input_hash=sent_hash,
                now=self.clock(),
                lease_seconds=self.dispatch_lease_seconds,
            )
            if claim is None:
                raise _DispatchClaimLost("analysis dispatch was already claimed")
            # This factory call is intentionally within ProcessingGate.dispatch.
            result = self.invoker_factory().invoke(CodexExecRequest(
                profile=ANALYSIS_PROFILE, prompt=prompt, output_schema=self.schema,
                schema_name=ANALYSIS_SCHEMA, prompt_name=ANALYSIS_PROMPT, input_hash=sent_hash,
                expected_prompt_hash=self.prompt_hash,
                expected_service_schema_hash=self.service_schema_hash,
            ))
            metadata = result.metadata
            analysis_output = self._validate_output(
                result.output, paper.paper_id, request, created_at, metadata,
            )
            return CodexExecResult(analysis_output, metadata)

        try:
            dispatched = self.gate.dispatch(
                request,
                invoke,
                processing_grant_id=processing_grant_id,
                now=now,
                paper_count=grant_paper_count,
            )
            decision = dispatched.decision
            if not decision.is_authorized:
                return self._persist_not_authorized(run_id, paper, request, decision, dispatch)
            assert claim is not None and sent_hash is not None and metadata is not None
            assert analysis_output is not None
            return self._persist_complete(
                run_id, paper, request, decision, sent_hash, metadata, analysis_output, claim,
            )
        except _DispatchClaimLost:
            current, terminal_result = self._observe_dispatch(
                self.dispatch_store.get(dispatch.dispatch_id), paper, request,
            )
            return terminal_result or self._result_from_dispatch(current, paper, request)
        except Exception as error:
            decision = captured.decision if captured is not None else None
            if claim is not None:
                return self._persist_uncertain_failure(
                    run_id, paper, request, decision, sent_hash, metadata, error, claim,
                )
            return self._persist_failure(run_id, paper, request, decision, sent_hash, metadata, error)

    def _validate_output(
        self, output: Mapping[str, Any], paper_id: str, request: ProcessingRequest,
        created_at: str, metadata: InvocationMetadata,
    ) -> Mapping[str, Any]:
        artifact_hash = request.artifact_hash
        input_scope = request.input_scope
        if (
            metadata.profile != ANALYSIS_PROFILE
            or metadata.model != "gpt-5.6-luna"
            or metadata.reasoning_effort != "medium"
            or metadata.actual_model != "gpt-5.6-luna"
            or metadata.actual_profile != ANALYSIS_PROFILE
            or metadata.output_hash != content_hash(dict(output))
        ):
            raise AnalysisValidationError(
                "Luna invocation metadata does not match the frozen analysis profile"
            )
        try:
            validate(dict(output), ANALYSIS_SCHEMA)
        except SchemaValidationError as error:
            raise AnalysisValidationError(str(error)) from error
        bindings = {
            "paper_id": paper_id, "artifact_hash": artifact_hash, "model": metadata.model,
            "prompt_hash": metadata.prompt_hash, "schema_hash": metadata.schema_hash,
            "input_scope": input_scope, "created_at": created_at,
        }
        if any(output[key] != value for key, value in bindings.items()):
            raise AnalysisValidationError("analysis output does not match its paper/artifact/model/prompt/schema binding")
        normalized = self.normalization_registry.normalize_analysis(output)
        normalized = _prune_unverifiable_label_evidence(normalized, request)
        try:
            validate(normalized, ANALYSIS_SCHEMA)
        except SchemaValidationError as error:
            raise AnalysisValidationError(str(error)) from error
        for unit in normalized["evidence_units"]:
            self._validate_evidence_unit(unit, input_scope)
        self._validate_label_evidence(normalized, input_scope)
        self._validate_source_locators(normalized, request)
        if input_scope != "full_pdf":
            if normalized["comparison_eligibility"] != "not_comparable" or "full_text" not in normalized["missing_fields"]:
                raise AnalysisValidationError("abstract_only and metadata_only analyses must disclose missing full text")
        return normalized

    @staticmethod
    def _validate_source_locators(
        output: Mapping[str, Any], request: ProcessingRequest,
    ) -> None:
        evidence = tuple(output["evidence_units"])
        labels = tuple(output["label_evidence"])
        if not evidence and not labels:
            return
        located = (*evidence, *labels)
        if request.input_scope == "full_pdf":
            pages = _authorized_pages(request)
            document = "\n".join(pages.values())
            for item in located:
                locator = item["locator"]
                if locator["kind"] == "page":
                    if not locator["value"].isdigit() or int(locator["value"]) not in pages:
                        raise AnalysisValidationError(
                            "full-text page locator does not exist in the authorized input"
                        )
                    source = pages[int(locator["value"])]
                elif locator["kind"] == "section":
                    if not _contains_text(document, locator["value"]):
                        raise AnalysisValidationError(
                            "full-text section locator does not exist in the authorized input"
                        )
                    source = document
                else:
                    raise AnalysisValidationError(
                        "full-text evidence must use page or section locators"
                    )
                if "source_text" in item and not _contains_text(source, item["source_text"]):
                    raise AnalysisValidationError(
                        "label source_text does not occur at its full-text locator"
                    )
            return

        fields = _authorized_input_fields(request)
        for item in located:
            locator = item["locator"]
            value = locator["value"]
            if locator["kind"] != "input_field" or value not in fields:
                raise AnalysisValidationError(
                    "input_field locator does not name a field in the authorized input"
                )
            if "source_text" in item and not _contains_text(fields[value], item["source_text"]):
                raise AnalysisValidationError(
                    "label source_text does not occur in its authorized input field"
                )

    @staticmethod
    def _validate_evidence_unit(unit: Mapping[str, Any], input_scope: str) -> None:
        needed = (
            "task_id", "dataset_id", "dataset_version", "split_id", "metric_id",
            "metric_definition_hash", "unit", "protocol_id", "protocol_hash", "baseline_id",
            "baseline_version", "source_value", "normalization_method", "normalizer_version",
        )
        absent = [name for name in needed if unit[name] is None]
        if unit["comparison_eligibility"] == "comparable":
            if absent or unit["missing_fields"] or not isinstance(unit["value"], (int, float)):
                raise AnalysisValidationError("comparable evidence units require complete normalized comparison fields")
        elif absent and not set(absent).issubset(set(unit["missing_fields"])):
            raise AnalysisValidationError("not_comparable evidence units must identify missing comparison fields")
        if input_scope != "full_pdf":
            if unit["locator"]["kind"] != "input_field":
                raise AnalysisValidationError("abstract_only and metadata_only evidence must use input_field locators")
            if unit["comparison_eligibility"] != "not_comparable" or "full_text" not in unit["missing_fields"]:
                raise AnalysisValidationError("abstract_only and metadata_only evidence is not directly comparable")

    @staticmethod
    def _validate_label_evidence(output: Mapping[str, Any], input_scope: str) -> None:
        cited = {(item["axis"], item["value"]) for item in output["label_evidence"]}
        for axis, values in output["labels"].items():
            if isinstance(values, list):
                missing = {(axis, value) for value in values} - cited
                if missing:
                    raise AnalysisValidationError("every generated label requires source evidence")
        if input_scope != "full_pdf" and any(
            item["locator"]["kind"] != "input_field" for item in output["label_evidence"]
        ):
            raise AnalysisValidationError("abstract_only and metadata_only labels must use input_field locators")

    def _observe_dispatch(
        self,
        dispatch: AnalysisDispatchRecord,
        paper: AnalysisInput,
        request: ProcessingRequest,
    ) -> tuple[AnalysisDispatchRecord, AnalysisPaperResult | None]:
        """Expire stale claims and materialize their terminal audit row atomically."""
        with self.database.transaction() as connection:
            current = self.dispatch_store.expire_stale(
                dispatch.dispatch_id, now=self.clock(), connection=connection,
            )
            if (
                current.status is AnalysisDispatchStatus.FAILED_TERMINAL
                and current.analysis_run_id is None
            ):
                failure = dict(current.error or {
                    "error": "UncertainDispatch",
                    "message": "analysis dispatch outcome is uncertain",
                })
                decision = _processing_decision(current.processing_decision)
                result = self._upsert(
                    current.run_id,
                    paper,
                    request,
                    current.prompt_input_hash or current.artifact_hash,
                    decision,
                    "failed",
                    None,
                    None,
                    failure,
                    connection=connection,
                )
                current = self.dispatch_store.link_analysis_run(
                    current.dispatch_id, result.analysis_run_id, connection=connection,
                )
        if current.status is AnalysisDispatchStatus.FAILED_TERMINAL:
            return current, self._result_from_dispatch(current, paper, request)
        return current, None

    def _result_from_dispatch(
        self,
        dispatch: AnalysisDispatchRecord,
        paper: AnalysisInput,
        request: ProcessingRequest,
    ) -> AnalysisPaperResult:
        decision = _processing_decision(dispatch.processing_decision)
        input_hash = dispatch.prompt_input_hash or dispatch.artifact_hash
        analysis_run_id = dispatch.analysis_run_id or (
            "analysis-" + content_hash([dispatch.run_id, dispatch.paper_id, input_hash])
        )
        row = None
        if dispatch.analysis_run_id is not None:
            row = self.database.connection.execute(
                "SELECT * FROM analysis_runs WHERE analysis_run_id = ?",
                (dispatch.analysis_run_id,),
            ).fetchone()
            if row is None:
                raise AnalysisValidationError("analysis dispatch references a missing analysis run")
            self._assert_existing_binding(row, paper, request)
            input_hash = row["input_hash"]

        if dispatch.status is AnalysisDispatchStatus.COMPLETE:
            if row is None or row["status"] != "complete":
                raise AnalysisValidationError("completed analysis dispatch has no completed analysis run")
            return AnalysisPaperResult(
                paper.paper_id,
                analysis_run_id,
                "complete",
                input_hash,
                request.input_scope,
                decision,
                self._load_output(row),
                resumed=True,
                markdown_artifact_id=row["markdown_artifact_id"],
            )

        if dispatch.status is AnalysisDispatchStatus.FAILED_TERMINAL:
            return AnalysisPaperResult(
                paper.paper_id,
                analysis_run_id,
                "failed",
                input_hash,
                request.input_scope,
                decision,
                resumed=True,
                error=_dispatch_error(dispatch),
                markdown_artifact_id=row["markdown_artifact_id"] if row is not None else None,
            )

        return AnalysisPaperResult(
            paper.paper_id,
            analysis_run_id,
            "incomplete",
            input_hash,
            request.input_scope,
            decision,
            resumed=True,
            error="analysis dispatch is already running" if dispatch.status is AnalysisDispatchStatus.RUNNING else None,
            markdown_artifact_id=row["markdown_artifact_id"] if row is not None else None,
        )

    def _persist_not_authorized(
        self,
        run_id: str,
        paper: AnalysisInput,
        request: ProcessingRequest,
        decision: ProcessingDecision,
        dispatch: AnalysisDispatchRecord,
    ) -> AnalysisPaperResult:
        # No prompt was sent; use the selected artifact hash solely as the
        # auditable attempted-input identity, never as a claimed sent prompt.
        result: AnalysisPaperResult | None = None
        with self.database.transaction() as connection:
            current = self.dispatch_store.record_manual(
                dispatch.dispatch_id, decision, connection=connection,
            )
            if current.status is AnalysisDispatchStatus.MANUAL_REQUIRED:
                result = self._upsert(
                    run_id,
                    paper,
                    request,
                    request.artifact_hash,
                    decision,
                    "incomplete",
                    None,
                    None,
                    None,
                    connection=connection,
                )
                current = self.dispatch_store.link_analysis_run(
                    current.dispatch_id, result.analysis_run_id, connection=connection,
                )
        return result or self._result_from_dispatch(current, paper, request)

    def _persist_failure(
        self, run_id: str, paper: AnalysisInput, request: ProcessingRequest, decision: ProcessingDecision | None,
        sent_hash: str | None, metadata: InvocationMetadata | None, error: Exception,
    ) -> AnalysisPaperResult:
        return self._upsert(
            run_id, paper, request, sent_hash or request.artifact_hash, decision, "failed", metadata, None,
            {"error": type(error).__name__, "message": str(error)},
        )

    def _record_unexpected_failure(
        self, run_id: str, paper: AnalysisInput, request: ProcessingRequest, error: Exception,
    ) -> AnalysisPaperResult:
        return self._persist_failure(run_id, paper, request, None, None, None, error)

    def _persist_uncertain_failure(
        self,
        run_id: str,
        paper: AnalysisInput,
        request: ProcessingRequest,
        decision: ProcessingDecision | None,
        sent_hash: str | None,
        metadata: InvocationMetadata | None,
        error: Exception,
        claim: AnalysisDispatchClaim,
    ) -> AnalysisPaperResult:
        failure: dict[str, Any] = {
            "error": "UncertainDispatch",
            "message": f"UncertainDispatch: {type(error).__name__}: {error}",
            "cause": {"error": type(error).__name__, "message": str(error)},
        }
        result: AnalysisPaperResult | None = None
        with self.database.transaction() as connection:
            current = self.dispatch_store.fail_terminal(
                claim,
                error=failure,
                now=self.clock(),
                invocation_id=metadata.invocation_id if metadata is not None else None,
                rendered_prompt_hash=metadata.rendered_prompt_hash if metadata is not None else None,
                invocation_metadata=asdict(metadata) if metadata is not None else None,
                connection=connection,
            )
            if (
                current.status is AnalysisDispatchStatus.FAILED_TERMINAL
                and current.analysis_run_id is None
            ):
                frozen_decision = _processing_decision(current.processing_decision) or decision
                result = self._upsert(
                    run_id,
                    paper,
                    request,
                    current.prompt_input_hash or sent_hash or request.artifact_hash,
                    frozen_decision,
                    "failed",
                    metadata,
                    None,
                    dict(current.error or failure),
                    connection=connection,
                )
                current = self.dispatch_store.link_analysis_run(
                    current.dispatch_id, result.analysis_run_id, connection=connection,
                )
        return result or self._result_from_dispatch(current, paper, request)

    def _persist_complete(
        self, run_id: str, paper: AnalysisInput, request: ProcessingRequest, decision: ProcessingDecision,
        sent_hash: str, metadata: InvocationMetadata, output: Mapping[str, Any],
        claim: AnalysisDispatchClaim,
    ) -> AnalysisPaperResult:
        payload = _json_bytes(output)
        stored = self.artifact_store.put_bytes(payload, mime_type="application/json", metadata={"kind": "analysis"})
        markdown = render_analysis_markdown(output).encode("utf-8")
        markdown_stored = self.artifact_store.put_bytes(
            markdown, mime_type="text/markdown; charset=utf-8", metadata={"kind": "analysis_markdown"},
        )
        with self.database.transaction() as connection:
            result = self._upsert(
                run_id,
                paper,
                request,
                sent_hash,
                decision,
                "complete",
                metadata,
                stored,
                None,
                output,
                markdown_stored=markdown_stored,
                connection=connection,
            )
            self.dispatch_store.complete(
                claim,
                analysis_run_id=result.analysis_run_id,
                invocation_id=metadata.invocation_id,
                rendered_prompt_hash=metadata.rendered_prompt_hash,
                invocation_metadata=asdict(metadata),
                now=self.clock(),
                connection=connection,
            )
        return result

    def _upsert(
        self, run_id: str, paper: AnalysisInput, request: ProcessingRequest, input_hash: str,
        decision: ProcessingDecision | None, status: str, metadata: InvocationMetadata | None,
        stored: Any | None, error: Mapping[str, Any] | None, output: Mapping[str, Any] | None = None,
        *, markdown_stored: Any | None = None, connection: Any | None = None,
    ) -> AnalysisPaperResult:
        if connection is None:
            with self.database.transaction() as active:
                return self._upsert(
                    run_id,
                    paper,
                    request,
                    input_hash,
                    decision,
                    status,
                    metadata,
                    stored,
                    error,
                    output,
                    markdown_stored=markdown_stored,
                    connection=active,
                )
        analysis_run_id = "analysis-" + content_hash([run_id, paper.paper_id, input_hash])
        metadata_document: dict[str, Any] = {}
        if output is not None:
            metadata_document["report_input_tokens"] = max(1, len(_json_bytes(output)))
        metadata_document["input_policy_facts"] = {
            "paper_id": request.paper_id,
            "artifact_hash": request.artifact_hash,
            "artifact": request.artifact,
            "input_scope": request.input_scope,
            "license": request.license,
            "access_basis": request.access_basis,
            "domain": request.domain,
            "mode": request.mode,
            "collection_id": request.collection_id,
            "collection_snapshot_hash": request.collection_snapshot_hash,
            "selection_snapshot_hash": request.selection_snapshot_hash,
            "data_category": request.data_category,
        }
        if decision is not None:
            metadata_document["processing_decision"] = _decision_json(decision)
        if metadata is not None:
            metadata_document["invocation"] = asdict(metadata)
        metadata_document["normalization_registry"] = {
            "version": self.normalization_registry.version,
            "registry_hash": self.normalization_registry.registry_hash,
        }
        metadata_document["analysis_configuration"] = {
            "workers": self.workers,
            "allow_abstract_only": self.allow_abstract_only,
            "output_schema_path": str(self.schema_path),
            "output_schema_hash": self.schema_hash,
        }
        if error is not None:
            metadata_document["failure"] = dict(error)
        artifact_id = None
        markdown_artifact_id = None
        if stored is not None:
            artifact_id = self._save_output_artifact(
                connection, paper.paper_id, stored, analysis_run_id, output_format="json",
            )
        if markdown_stored is not None:
            markdown_artifact_id = self._save_output_artifact(
                connection, paper.paper_id, markdown_stored, analysis_run_id, output_format="markdown",
            )
        connection.execute(
            """INSERT INTO analysis_runs(
                analysis_run_id, run_id, paper_id, artifact_id, input_hash, input_scope, model_id, model_revision,
                prompt_hash, schema_hash, implementation_version, authorization_grant_id, policy_version,
                policy_decision, invocation_metadata_json, status, output_artifact_id,
                markdown_artifact_id, completed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CASE WHEN ? IN ('complete', 'failed', 'incomplete') THEN CURRENT_TIMESTAMP END)
            ON CONFLICT(run_id, paper_id, input_hash) DO UPDATE SET
                artifact_id=excluded.artifact_id, authorization_grant_id=excluded.authorization_grant_id,
                policy_version=excluded.policy_version, policy_decision=excluded.policy_decision,
                invocation_metadata_json=excluded.invocation_metadata_json, status=excluded.status,
                output_artifact_id=excluded.output_artifact_id,
                markdown_artifact_id=excluded.markdown_artifact_id, completed_at=excluded.completed_at""",
            (
                analysis_run_id, run_id, paper.paper_id, paper.artifact_id, input_hash, request.input_scope,
                "gpt-5.6-luna", str(output["model_revision"]) if output else "unavailable",
                self.prompt_hash, self.schema_hash, self.implementation_version,
                decision.processing_grant_id if decision is not None and decision.is_authorized else None,
                decision.policy_version if decision else "unavailable",
                decision.outcome.value if decision else "failed_before_policy",
                _json_text(metadata_document), status, artifact_id, markdown_artifact_id, status,
            ),
        )
        return AnalysisPaperResult(
            paper.paper_id, analysis_run_id, status, input_hash, request.input_scope, decision, output,
            error=str(error.get("message", error.get("error"))) if error else None,
            markdown_artifact_id=markdown_artifact_id,
        )

    def _save_output_artifact(
        self, connection: Any, paper_id: str, stored: Any, analysis_run_id: str, *, output_format: str,
    ) -> str:
        artifact_id = "artifact-" + stored.artifact_hash
        existing = connection.execute("SELECT paper_id, artifact_kind, relative_path, mime_type, byte_size FROM artifacts WHERE sha256 = ?", (stored.artifact_hash,)).fetchone()
        if existing is None:
            connection.execute(
                """INSERT INTO artifacts(artifact_id, paper_id, artifact_kind, relative_path, mime_type, byte_size, sha256, provenance_json)
                   VALUES (?, ?, 'analysis', ?, ?, ?, ?, ?)""",
                (artifact_id, paper_id, stored.relative_path, stored.mime_type, stored.size_bytes, stored.artifact_hash,
                 _json_text({"analysis_run_id": analysis_run_id, "stage": "stage4", "format": output_format})),
            )
        elif tuple(existing[key] for key in ("paper_id", "artifact_kind", "relative_path", "mime_type", "byte_size")) != (
            paper_id, "analysis", stored.relative_path, stored.mime_type, stored.size_bytes,
        ):
            raise AnalysisValidationError("analysis artifact metadata conflicts with existing content")
        return artifact_id

    def _existing(self, run_id: str, paper_id: str, artifact_hash: str):
        # Completed rows may use a sent-prompt hash, so locate them by the
        # immutable selected artifact binding retained in the decision JSON.
        rows = self.database.connection.execute(
            """SELECT * FROM analysis_runs WHERE run_id = ? AND paper_id = ?
               ORDER BY CASE status WHEN 'complete' THEN 0 ELSE 1 END, created_at DESC, analysis_run_id DESC""",
            (run_id, paper_id),
        ).fetchall()
        for row in rows:
            detail = json.loads(row["invocation_metadata_json"] or "{}")
            decision = detail.get("processing_decision", {})
            if decision.get("input_artifact_hash") == artifact_hash:
                return row
        return None

    def _load_output(self, row: Any) -> Mapping[str, Any]:
        artifact = self.database.connection.execute(
            "SELECT sha256 FROM artifacts WHERE artifact_id = ?", (row["output_artifact_id"],),
        ).fetchone()
        if artifact is None:
            raise AnalysisValidationError("completed analysis output artifact is missing")
        value = json.loads(self.artifact_store.read_bytes(artifact["sha256"]))
        if not isinstance(value, Mapping):
            raise AnalysisValidationError("completed analysis output is not an object")
        return value

    def _assert_existing_binding(self, row: Any, paper: AnalysisInput, request: ProcessingRequest) -> None:
        expected = (request.input_scope, "gpt-5.6-luna", self.prompt_hash, self.schema_hash, self.implementation_version)
        actual = tuple(row[key] for key in ("input_scope", "model_id", "prompt_hash", "schema_hash", "implementation_version"))
        if actual != expected or (row["artifact_id"] is not None and row["artifact_id"] != paper.artifact_id):
            raise ValueError("analysis input or configuration is immutable")


def _authorized_payload(
    paper_id: str,
    artifact_hash: str,
    input_scope: str,
    invocation: ModelInvocation,
    output_binding: Mapping[str, str],
) -> dict[str, Any]:
    if invocation.pdf_bytes is not None:
        content: Any = b64encode(invocation.pdf_bytes).decode("ascii")
        encoding = "base64"
    elif invocation.normalized_text_bytes is not None:
        content = invocation.normalized_text_bytes.decode("utf-8")
        encoding = "utf-8"
    elif invocation.abstract_bytes is not None:
        content = json.loads(invocation.abstract_bytes.decode("utf-8"))
        encoding = "json"
    else:
        content = dict(invocation.metadata or {})
        encoding = "json"
    return {
        "paper_id": paper_id, "artifact_hash": artifact_hash, "input_scope": input_scope,
        "output_binding": dict(output_binding), "content_encoding": encoding, "content": content,
    }


def _authorized_pages(request: ProcessingRequest) -> dict[int, str]:
    if request.normalized_text_bytes is not None:
        text = request.normalized_text_bytes.decode("utf-8")
        markers = tuple(PAGE_MARKER.finditer(text))
        if not markers:
            raise AnalysisValidationError(
                "full-text citations require normalized page markers"
            )
        return {
            int(match.group(1)): text[
                match.end() : markers[index + 1].start()
                if index + 1 < len(markers)
                else len(text)
            ]
            for index, match in enumerate(markers)
        }
    if request.pdf_bytes is not None:
        reader = PdfReader(BytesIO(request.pdf_bytes), strict=False)
        if reader.is_encrypted:
            raise AnalysisValidationError(
                "encrypted PDF cannot validate full-text locators"
            )
        return {
            index: page.extract_text() or ""
            for index, page in enumerate(reader.pages, start=1)
        }
    raise AnalysisValidationError("full-text locator has no authorized full-text input")


def _authorized_input_fields(request: ProcessingRequest) -> dict[str, str]:
    if request.abstract_bytes is not None:
        wrapper = json.loads(request.abstract_bytes.decode("utf-8"))
        fields = {"abstract": str(wrapper["abstract"])}
        fields.update(
            {
                f"metadata.{key}": _field_text(value)
                for key, value in wrapper["metadata"].items()
            }
        )
        return fields
    if request.metadata is not None:
        return {str(key): _field_text(value) for key, value in request.metadata.items()}
    raise AnalysisValidationError("input_field locator has no authorized structured input")


def _field_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _contains_text(source: str, cited: str) -> bool:
    return " ".join(cited.split()).casefold() in " ".join(source.split()).casefold()


def _prune_unverifiable_label_evidence(
    output: Mapping[str, Any], request: ProcessingRequest,
) -> dict[str, Any]:
    normalized = dict(output)
    if request.input_scope != "full_pdf" or not normalized["label_evidence"]:
        return normalized
    pages = _authorized_pages(request)
    document = "\n".join(pages.values())

    def supported(item: Mapping[str, Any]) -> bool:
        locator = item["locator"]
        if locator["kind"] == "page":
            if not locator["value"].isdigit():
                return False
            source = pages.get(int(locator["value"]))
        elif locator["kind"] == "section":
            source = document if _contains_text(document, locator["value"]) else None
        else:
            source = None
        return source is not None and _contains_text(source, item["source_text"])

    evidence = [item for item in normalized["label_evidence"] if supported(item)]
    cited = {(item["axis"], item["value"]) for item in evidence}
    normalized["label_evidence"] = evidence
    normalized["labels"] = {
        axis: [value for value in values if (axis, value) in cited]
        if isinstance(values, list)
        else values
        for axis, values in normalized["labels"].items()
    }
    return normalized


def _timestamp(value: datetime | str | None) -> str:
    if isinstance(value, str):
        return value
    moment = value or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def render_analysis_markdown(output: Mapping[str, Any]) -> str:
    """Render the validated analysis JSON into a deterministic readable view."""
    lines = [
        f"# 论文分析：{output['paper_id']}",
        "",
        f"- 输入范围：`{output['input_scope']}`",
        f"- 模型：`{output['model']}`",
        f"- 分析时间：{output['created_at']}",
        "",
        "## 研究问题与动机",
        "",
        str(output["research_question_and_motivation"]),
        "",
        "## 摘要",
        "",
        str(output["summary"]),
    ]
    for title, key in (
        ("方法", "methods"),
        ("关键技术", "key_techniques"),
        ("数据集", "datasets"),
        ("实验设置", "experimental_setup"),
        ("指标", "metrics"),
        ("主要结果", "results"),
        ("局限", "limitations"),
    ):
        lines.extend(["", f"## {title}", ""])
        values = output[key]
        lines.extend(f"- {value}" for value in values)
        if not values:
            lines.append("- 未提供")
    lines.extend([
        "", "## 可信度", "", str(output["credibility"]),
        "", "## 与主题的关系", "", str(output["topic_relevance"]),
        "", "## 证据单元", "",
    ])
    for unit in output["evidence_units"]:
        locator = unit["locator"]
        lines.append(f"- {unit['claim']}（{locator['kind']}: {locator['value']}）")
    if not output["evidence_units"]:
        lines.append("- 未提供")
    return "\n".join(lines) + "\n"


def _processing_decision(document: Mapping[str, Any] | None) -> ProcessingDecision | None:
    if document is None:
        return None
    try:
        return ProcessingDecision(
            policy_version=str(document["policy_version"]),
            policy_hash=str(document["policy_hash"]),
            outcome=ProcessingOutcome(str(document["outcome"])),
            reason_code=str(document["reason_code"]),
            input_artifact_hash=str(document["input_artifact_hash"]),
            provider=str(document["provider"]),
            model=str(document["model"]),
            purpose=str(document["purpose"]),
            data_category=str(document["data_category"]),
            processing_grant_id=(
                str(document["processing_grant_id"])
                if document.get("processing_grant_id") is not None
                else None
            ),
            authorized_by=(
                str(document["authorized_by"])
                if document.get("authorized_by") is not None
                else None
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise AnalysisValidationError("analysis dispatch processing decision is malformed") from error


def _dispatch_error(dispatch: AnalysisDispatchRecord) -> str:
    if dispatch.error is None:
        return "UncertainDispatch: remote invocation outcome is uncertain"
    message = str(dispatch.error.get("message", "remote invocation outcome is uncertain"))
    if "UncertainDispatch" in message:
        return message
    return f"UncertainDispatch: {message}"


def _decision_json(decision: ProcessingDecision) -> dict[str, Any]:
    document = asdict(decision)
    document["outcome"] = decision.outcome.value
    return document


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _json_bytes(value: Any) -> bytes:
    return _json_text(value).encode("utf-8")


def _digest_json(value: Mapping[str, Any]) -> str:
    return sha256(_json_bytes(value)).hexdigest()
