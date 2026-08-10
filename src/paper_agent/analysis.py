"""Stage 4's policy-gated, per-paper analysis coordinator.

This module owns the boundary between prepared paper inputs and ``codex
exec``.  In particular, an executor is made only inside the callback passed to
``ProcessingGate.dispatch``: a policy denial therefore cannot create an
executor, let alone make a model call.
"""

from __future__ import annotations

from base64 import b64encode
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Protocol

from .analysis_registry import AnalysisNormalizationRegistry
from .artifacts import ArtifactStore
from .canonical import content_hash
from .codex_exec import CodexExec, CodexExecRequest, CodexExecResult, InvocationMetadata
from .processing import ModelInvocation, ProcessingDecision, ProcessingGate, ProcessingRequest
from .schema import SchemaValidationError, schema_directory, validate
from .storage import Database


ANALYSIS_PROFILE = "stage4_analysis_luna"
ANALYSIS_SCHEMA = "paper-analysis.schema.json"
ANALYSIS_PROMPT = "paper-analysis.md"
IMPLEMENTATION_VERSION = "phase5-stage4-v2"


class AnalysisValidationError(ValueError):
    """A model result is valid JSON but is not a valid bound paper analysis."""


class AnalysisInvoker(Protocol):
    def invoke(self, request: CodexExecRequest) -> CodexExecResult: ...


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
    ) -> None:
        self.database = database
        self.artifact_store = artifact_store
        self.gate = gate
        self.invoker_factory = invoker_factory
        self.normalization_registry = normalization_registry or AnalysisNormalizationRegistry.load()
        self.implementation_version = implementation_version
        root = schema_directory()
        self.schema = json.loads((root / ANALYSIS_SCHEMA).read_text(encoding="utf-8"))
        self.schema_hash = _digest_json(self.schema)
        self.prompt_hash = sha256((Path(__file__).resolve().parents[2] / "prompts" / ANALYSIS_PROMPT).read_bytes()).hexdigest()
        self.config_hash = content_hash({
            "profile": ANALYSIS_PROFILE, "model": "gpt-5.6-luna", "prompt_hash": self.prompt_hash,
            "schema_hash": self.schema_hash, "implementation_version": implementation_version,
            "normalization_registry": self.normalization_registry.registry_hash,
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
        results: list[AnalysisPaperResult] = []
        for paper, request in zip(papers, requests, strict=True):
            try:
                results.append(self._run_one(run_id, paper, request, now, processing_grant_id))
            except Exception as error:  # One malformed/model-failed paper cannot stop the batch.
                results.append(self._record_unexpected_failure(run_id, paper, request, error))
        status = "complete" if all(result.status == "complete" for result in results) else "incomplete"
        with self.database.transaction() as connection:
            connection.execute(
                "UPDATE pipeline_runs SET status = ?, completed_at = CURRENT_TIMESTAMP WHERE run_id = ?",
                (status, run_id),
            )
        return AnalysisRunResult(tuple(results))

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
                if actual != expected:
                    raise ValueError("Stage 4 run input or configuration is immutable")
                return
            connection.execute(
                """INSERT INTO pipeline_runs(run_id, stage, status, input_hash, config_hash, implementation_version, started_at)
                   VALUES (?, 'stage4', 'running', ?, ?, ?, CURRENT_TIMESTAMP)""",
                (run_id, input_hash, self.config_hash, self.implementation_version),
            )

    def _run_one(
        self, run_id: str, paper: AnalysisInput, request: ProcessingRequest,
        now: datetime | str | None, processing_grant_id: str | None,
    ) -> AnalysisPaperResult:
        existing = self._existing(run_id, paper.paper_id, request.artifact_hash)
        if existing is not None:
            self._assert_existing_binding(existing, paper, request)
            if existing["status"] == "complete":
                return AnalysisPaperResult(
                    paper.paper_id, existing["analysis_run_id"], "complete", existing["input_hash"],
                    existing["input_scope"], output=self._load_output(existing), resumed=True,
                    markdown_artifact_id=existing["markdown_artifact_id"],
                )

        captured: ModelInvocation | None = None
        sent_hash: str | None = None
        metadata: InvocationMetadata | None = None
        analysis_output: Mapping[str, Any] | None = None
        created_at = _timestamp(now)

        def invoke(invocation: ModelInvocation) -> CodexExecResult:
            nonlocal captured, sent_hash, metadata, analysis_output
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
            # This factory call is intentionally within ProcessingGate.dispatch.
            result = self.invoker_factory().invoke(CodexExecRequest(
                profile=ANALYSIS_PROFILE, prompt=prompt, output_schema=self.schema,
                schema_name=ANALYSIS_SCHEMA, prompt_name=ANALYSIS_PROMPT, input_hash=sent_hash,
            ))
            metadata = result.metadata
            analysis_output = self._validate_output(
                result.output, paper.paper_id, request.artifact_hash, request.input_scope, created_at, metadata,
            )
            return CodexExecResult(analysis_output, metadata)

        try:
            dispatched = self.gate.dispatch(
                request, invoke, processing_grant_id=processing_grant_id, now=now,
            )
        except Exception as error:
            decision = captured.decision if captured is not None else None
            return self._persist_failure(run_id, paper, request, decision, sent_hash, metadata, error)

        decision = dispatched.decision
        if not decision.is_authorized:
            return self._persist_not_authorized(run_id, paper, request, decision)
        assert sent_hash is not None and metadata is not None and analysis_output is not None
        return self._persist_complete(run_id, paper, request, decision, sent_hash, metadata, analysis_output)

    def _validate_output(
        self, output: Mapping[str, Any], paper_id: str, artifact_hash: str, input_scope: str,
        created_at: str, metadata: InvocationMetadata,
    ) -> Mapping[str, Any]:
        try:
            validate(output, ANALYSIS_SCHEMA)
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
        try:
            validate(normalized, ANALYSIS_SCHEMA)
        except SchemaValidationError as error:
            raise AnalysisValidationError(str(error)) from error
        for unit in normalized["evidence_units"]:
            self._validate_evidence_unit(unit, input_scope)
        self._validate_label_evidence(normalized, input_scope)
        if input_scope != "full_pdf":
            if normalized["comparison_eligibility"] != "not_comparable" or "full_text" not in normalized["missing_fields"]:
                raise AnalysisValidationError("abstract_only and metadata_only analyses must disclose missing full text")
        return normalized

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

    def _persist_not_authorized(
        self, run_id: str, paper: AnalysisInput, request: ProcessingRequest, decision: ProcessingDecision,
    ) -> AnalysisPaperResult:
        # No prompt was sent; use the selected artifact hash solely as the
        # auditable attempted-input identity, never as a claimed sent prompt.
        return self._upsert(
            run_id, paper, request, request.artifact_hash, decision, "incomplete", None, None, None,
        )

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

    def _persist_complete(
        self, run_id: str, paper: AnalysisInput, request: ProcessingRequest, decision: ProcessingDecision,
        sent_hash: str, metadata: InvocationMetadata, output: Mapping[str, Any],
    ) -> AnalysisPaperResult:
        payload = _json_bytes(output)
        stored = self.artifact_store.put_bytes(payload, mime_type="application/json", metadata={"kind": "analysis"})
        markdown = render_analysis_markdown(output).encode("utf-8")
        markdown_stored = self.artifact_store.put_bytes(
            markdown, mime_type="text/markdown; charset=utf-8", metadata={"kind": "analysis_markdown"},
        )
        return self._upsert(
            run_id, paper, request, sent_hash, decision, "complete", metadata, stored, None, output,
            markdown_stored=markdown_stored,
        )

    def _upsert(
        self, run_id: str, paper: AnalysisInput, request: ProcessingRequest, input_hash: str,
        decision: ProcessingDecision | None, status: str, metadata: InvocationMetadata | None,
        stored: Any | None, error: Mapping[str, str] | None, output: Mapping[str, Any] | None = None,
        *, markdown_stored: Any | None = None,
    ) -> AnalysisPaperResult:
        analysis_run_id = "analysis-" + content_hash([run_id, paper.paper_id, input_hash])
        metadata_document: dict[str, Any] = {}
        if decision is not None:
            metadata_document["processing_decision"] = _decision_json(decision)
        if metadata is not None:
            metadata_document["invocation"] = asdict(metadata)
        metadata_document["normalization_registry"] = {
            "version": self.normalization_registry.version,
            "registry_hash": self.normalization_registry.registry_hash,
        }
        if error is not None:
            metadata_document["failure"] = dict(error)
        artifact_id = None
        markdown_artifact_id = None
        with self.database.transaction() as connection:
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
                    decision.processing_grant_id if decision else None,
                    decision.policy_version if decision else "unavailable",
                    decision.outcome.value if decision else "failed_before_policy",
                    _json_text(metadata_document), status, artifact_id, markdown_artifact_id, status,
                ),
            )
        return AnalysisPaperResult(
            paper.paper_id, analysis_run_id, status, input_hash, request.input_scope, decision, output,
            error=error["message"] if error else None, markdown_artifact_id=markdown_artifact_id,
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
