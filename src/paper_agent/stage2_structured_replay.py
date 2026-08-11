"""Measured runner for the public Stage 2 structured-output replay gate."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import mkstemp
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator, ValidationError

from .canonical import content_hash
from .schema import schema_directory, validate
from .stage2_backends import OmlxResponse, OmlxTransport, Stage2BackendError
from .stage2_evaluation import (
    ReplayError,
    Stage2Decision,
    StructuredReplayManifest,
    StructuredReplayRecord,
    StructuredReplayResult,
    structured_replay_gate,
)
from .stage2_pipeline import (
    ADJUDICATION_SYSTEM_PROMPT,
    ADJUDICATION_USER_TEMPLATE,
    Stage2Paper,
    Stage2Profile,
)


@dataclass(frozen=True, slots=True)
class StructuredReplayRun:
    """Frozen replay evidence and the gate recomputed from that evidence."""

    manifest: StructuredReplayManifest
    records: tuple[StructuredReplayRecord, ...]
    result: StructuredReplayResult


@dataclass(frozen=True, slots=True)
class _Attempt:
    error: ReplayError
    returned_pair_id: str | None
    schema_outside_text: bool
    think_tag_leak: bool
    valid: bool
    decision: Stage2Decision


def freeze_structured_replay_manifest(
    papers: Sequence[Stage2Paper], profile: Stage2Profile,
) -> StructuredReplayManifest:
    """Freeze the exact papers and Qwen provenance consumed by a replay."""

    _validate_papers(papers)
    return _frozen_manifest(
        papers,
        profile,
        stage2_config_hash=profile.base_runtime_config_hash,
        schema_hash=profile.schema_hash,
    )


def _frozen_manifest(
    papers: Sequence[Stage2Paper],
    profile: Stage2Profile,
    *,
    stage2_config_hash: str,
    schema_hash: str,
) -> StructuredReplayManifest:
    return StructuredReplayManifest(
        version=1,
        pair_ids=tuple(paper.paper_id for paper in papers),
        corpus_hash=content_hash({
            "schema_version": 1,
            "papers": [_paper_document(paper) for paper in papers],
        }),
        stage2_config_hash=stage2_config_hash,
        model_lock_hash=_model_lock_hash(profile),
        prompt_hash=profile.prompt_hash,
        schema_hash=schema_hash,
    )


@dataclass(slots=True)
class StructuredReplayRunner:
    """Run one chat completion per frozen paper, with at most one retry."""

    profile: Stage2Profile
    transport: OmlxTransport

    def run(
        self,
        papers: Sequence[Stage2Paper],
        *,
        manifest: StructuredReplayManifest | None = None,
        manifest_path: Path | None = None,
        records_path: Path | None = None,
    ) -> StructuredReplayRun:
        """Execute a frozen universe and optionally publish no-replace artifacts."""

        _validate_papers(papers)
        if (manifest_path is None) != (records_path is None):
            raise ValueError("structured replay requires both manifest and records paths")
        if manifest_path is not None and records_path is not None:
            _validate_output_paths(manifest_path, records_path)
        schema, schema_hash = _load_schema(self.profile.schema_version)
        stage2_config_hash = self.profile.base_runtime_config_hash
        if self.profile.schema_hash != schema_hash:
            raise ValueError("structured replay schema changed while freezing the run")
        expected = _frozen_manifest(
            papers,
            self.profile,
            stage2_config_hash=stage2_config_hash,
            schema_hash=schema_hash,
        )
        if manifest is not None and manifest != expected:
            raise ValueError("structured replay manifest does not match the supplied papers and profile")
        frozen = expected
        manifest_hash = frozen.hash()
        validator = Draft202012Validator(schema)
        with ThreadPoolExecutor(max_workers=self.profile.adjudicator_concurrency) as executor:
            records = tuple(executor.map(
                lambda paper: self._run_one(paper, manifest_hash, schema, validator), papers,
            ))
        result = structured_replay_gate(frozen, records)
        run = StructuredReplayRun(frozen, records, result)
        if manifest_path is not None and records_path is not None:
            write_structured_replay_artifacts(run, manifest_path, records_path)
        return run

    def _run_one(
        self,
        paper: Stage2Paper,
        manifest_hash: str,
        schema: Mapping[str, Any],
        validator: Draft202012Validator,
    ) -> StructuredReplayRecord:
        first = self._request(paper, schema, validator)
        if first.valid:
            return StructuredReplayRecord(
                paper.paper_id, manifest_hash, first.error, first.returned_pair_id,
                first.schema_outside_text, first.think_tag_leak, 0, 0, None,
                True, first.returned_pair_id, first.schema_outside_text,
                first.think_tag_leak, first.decision,
            )
        retry = self._request(paper, schema, validator)
        return StructuredReplayRecord(
            paper.paper_id, manifest_hash, first.error, first.returned_pair_id,
            first.schema_outside_text, first.think_tag_leak, 0, 1, retry.error,
            retry.valid, retry.returned_pair_id, retry.schema_outside_text,
            retry.think_tag_leak,
            retry.decision if retry.valid else Stage2Decision.NEEDS_REVIEW,
        )

    def _request(
        self,
        paper: Stage2Paper,
        schema: Mapping[str, Any],
        validator: Draft202012Validator,
    ) -> _Attempt:
        payload = {
            "model": self.profile.adjudicator_model_id,
            "messages": [
                {"role": "system", "content": ADJUDICATION_SYSTEM_PROMPT},
                {"role": "user", "content": ADJUDICATION_USER_TEMPLATE.format(
                    query_version=self.profile.query_version,
                    query=self.profile.query,
                    paper_id=paper.paper_id,
                    document=_paper_text(paper),
                )},
            ],
            "temperature": 0,
            "seed": self.profile.adjudicator_seed,
            "stream": False,
            "max_tokens": 256,
            "chat_template_kwargs": {"enable_thinking": False},
            "structured_outputs": {"json": dict(schema)},
        }
        try:
            response = self.transport.request("/v1/chat/completions", payload)
        except Exception as error:
            return _failed_attempt(_exception_error(error))
        if any(name.lower() == "warning" for name in response.headers):
            return _failed_attempt(ReplayError.SCHEMA)
        if response.status_code != 200:
            return _failed_attempt(
                ReplayError.SCHEMA
                if response.status_code == 400
                else ReplayError.TIMEOUT
                if response.status_code in {408, 504}
                else ReplayError.SERVICE
            )
        return _inspect_response(
            response,
            paper.paper_id,
            self.profile.adjudicator_model_id,
            validator,
        )


def write_structured_replay_artifacts(
    run: StructuredReplayRun, manifest_path: Path, records_path: Path,
) -> None:
    """Create no-replace evidence, publishing the manifest after the records."""

    if manifest_path == records_path:
        raise ValueError("structured replay manifest and records paths must differ")
    documents = (
        (manifest_path, run.manifest.document(), "stage2-structured-replay-manifest.schema.json"),
        (records_path, {
            "schema_version": "1",
            "kind": "stage2_structured_replay_records",
            "records": [record.document() for record in run.records],
        }, "stage2-structured-replay-records.schema.json"),
    )
    for path, document, schema_name in documents:
        validate(document, schema_name)
        if path.exists():
            raise FileExistsError(f"refusing to replace structured replay artifact: {path}")
    for path, document, _ in reversed(documents):
        _write_json_no_replace(path, document)


def _inspect_response(
    response: OmlxResponse,
    paper_id: str,
    model_id: str,
    validator: Draft202012Validator,
) -> _Attempt:
    try:
        response_document = response.json()
        content = response_document["choices"][0]["message"]["content"]
    except (
        KeyError,
        IndexError,
        TypeError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        Stage2BackendError,
    ):
        return _failed_attempt(ReplayError.SCHEMA)
    if response_document.get("model") != model_id or not isinstance(content, str):
        return _failed_attempt(ReplayError.SCHEMA)
    think_tag_leak = "<think" in content.casefold() or "</think" in content.casefold()
    stripped = content.lstrip()
    prefix = False
    try:
        value, end = json.JSONDecoder().raw_decode(stripped)
    except json.JSONDecodeError:
        start = content.find("{")
        if start < 0:
            return _Attempt(ReplayError.SCHEMA, None, False, think_tag_leak, False, Stage2Decision.NEEDS_REVIEW)
        try:
            value, end = json.JSONDecoder().raw_decode(content[start:])
        except json.JSONDecodeError:
            return _Attempt(ReplayError.SCHEMA, None, False, think_tag_leak, False, Stage2Decision.NEEDS_REVIEW)
        stripped = content[start:]
        prefix = bool(content[:start].strip())
    outside_text = prefix or bool(stripped[end:].strip())
    returned_pair_id = value.get("paper_id") if isinstance(value, dict) and isinstance(value.get("paper_id"), str) else None
    try:
        validator.validate(value)
        decision = Stage2Decision(value["decision"])
    except (ValidationError, TypeError, KeyError, ValueError):
        return _Attempt(ReplayError.SCHEMA, returned_pair_id, outside_text, think_tag_leak, False, Stage2Decision.NEEDS_REVIEW)
    valid = not outside_text and not think_tag_leak and returned_pair_id == paper_id
    return _Attempt(
        ReplayError.NONE if valid else ReplayError.SCHEMA,
        returned_pair_id,
        outside_text,
        think_tag_leak,
        valid,
        decision if valid else Stage2Decision.NEEDS_REVIEW,
    )


def _failed_attempt(error: ReplayError) -> _Attempt:
    return _Attempt(error, None, False, False, False, Stage2Decision.NEEDS_REVIEW)


def _exception_error(error: Exception) -> ReplayError:
    if isinstance(error, TimeoutError) or "timeout" in str(error).casefold() or "timed out" in str(error).casefold():
        return ReplayError.TIMEOUT
    return ReplayError.SERVICE


def _load_schema(schema_version: str) -> tuple[Mapping[str, Any], str]:
    payload = (schema_directory() / schema_version).read_bytes()
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("structured replay schema must be an object")
    Draft202012Validator.check_schema(value)
    return value, sha256(payload).hexdigest()


def _validate_papers(papers: Sequence[Stage2Paper]) -> None:
    if len(papers) < 1_000:
        raise ValueError("structured replay requires at least 1,000 papers")
    if len({paper.paper_id for paper in papers}) != len(papers):
        raise ValueError("structured replay paper_ids must be unique")


def _model_lock_hash(profile: Stage2Profile) -> str:
    lock = profile.adjudicator_lock_hash
    if lock is not None and len(lock) == 64 and all(character in "0123456789abcdef" for character in lock):
        return lock
    return content_hash({
        "model_lock": lock,
        "model_id": profile.adjudicator_model_id,
        "revision": profile.adjudicator_revision,
    })


def _paper_text(paper: Stage2Paper) -> str:
    return f"Title: {paper.title}\nAbstract: {paper.abstract or ''}\nKeywords: {', '.join(paper.keywords)}"


def _paper_document(paper: Stage2Paper) -> dict[str, object]:
    return {
        "paper_id": paper.paper_id,
        "title": paper.title,
        "abstract": paper.abstract,
        "keywords": list(paper.keywords),
        "document_type": paper.document_type,
        "possibly_truncated": paper.possibly_truncated,
        "multi_condition_conflict": paper.multi_condition_conflict,
        "language_anomaly": paper.language_anomaly,
    }


def _validate_output_paths(manifest_path: Path, records_path: Path) -> None:
    if manifest_path == records_path:
        raise ValueError("structured replay manifest and records paths must differ")
    for path in (manifest_path, records_path):
        if path.exists():
            raise FileExistsError(f"refusing to replace structured replay artifact: {path}")


def _write_json_no_replace(path: Path, document: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
