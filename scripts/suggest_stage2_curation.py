#!/usr/bin/env python3
"""Create evaluator-private, model-provisional Stage 2 curation suggestions.

The output is deliberately only a sampling aid.  It is not a gold-label
generator and does not send any result back to the model for repair or retry.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.request import Request, urlopen

from paper_agent.canonical import content_hash
from paper_agent.schema import SchemaValidationError, validate
from paper_agent.stage2_sampling import (
    CurationDecision,
    CurationDecisions,
    CurationWorklist,
    CurationWorklistRow,
)


PROMPT_VERSION = "stage2-curation-suggestion-v1"
SYSTEM_PROMPT = """You assign a provisional relevance label for literature sampling.
Use this fixed rubric: 0 = clearly irrelevant; 1 = weakly related, background-only,
or evidence is insufficient; 2 = directly related and should be retained; 3 = a core
paper for the topic and must be retained. Return exactly one label for every supplied
topic and paper_id pair. Do not include a rationale, analysis, or any extra fields."""
Transport = Callable[[Mapping[str, Any]], Mapping[str, Any]]


def _sha256_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json_object(path: Path, name: str) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} must be a readable JSON object") from error
    if not isinstance(document, dict):
        raise ValueError(f"{name} must be a JSON object")
    return document


def load_worklist(path: Path) -> CurationWorklist:
    document = _load_json_object(path, "curation worklist")
    try:
        validate(document, "stage2-curation-worklist.schema.json")
    except SchemaValidationError as error:
        raise ValueError(str(error)) from error
    return CurationWorklist(
        snapshot_hash=document["snapshot_hash"],
        hidden_real_freeze_frame_hash=document["hidden_real_freeze_frame_hash"],
        rows=tuple(CurationWorklistRow(**row) for row in document["rows"]),
    )


def load_model_lock(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()
    lock = _load_json_object(path, "model lock")
    model_id = lock.get("model_id")
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("model lock requires model_id")
    return model_id, _sha256_bytes(raw)


def _response_schema(rows: list[CurationWorklistRow]) -> dict[str, Any]:
    topics = sorted({row.topic for row in rows})
    paper_ids = sorted({row.paper_id for row in rows})
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["decisions"],
        "properties": {
            "decisions": {
                "type": "array",
                "minItems": len(rows),
                "maxItems": len(rows),
                "uniqueItems": True,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["topic", "paper_id", "provisional_label"],
                    "properties": {
                        "topic": {"type": "string", "enum": topics},
                        "paper_id": {"type": "string", "enum": paper_ids},
                        "provisional_label": {"type": "integer", "minimum": 0, "maximum": 3},
                    },
                },
            },
        },
    }


def _request_for_batch(rows: list[CurationWorklistRow], model_id: str) -> dict[str, Any]:
    papers = [{
        "topic": row.topic,
        "paper_id": row.paper_id,
        "title": row.title,
        "abstract": row.abstract,
        "language": row.language,
    } for row in rows]
    return {
        "model": model_id,
        "temperature": 0,
        "seed": 0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps({"papers": papers}, ensure_ascii=False, sort_keys=True)},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "stage2_curation_suggestions",
                "strict": True,
                "schema": _response_schema(rows),
            },
        },
    }


def _extract_decisions(
    response: Mapping[str, Any], expected_keys: set[tuple[str, str]]
) -> dict[tuple[str, str], int]:
    try:
        content = response["choices"][0]["message"]["content"]
        document = json.loads(content)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("oMLX response is not a structured decision object") from error
    if not isinstance(document, dict) or set(document) != {"decisions"}:
        raise ValueError("oMLX response has unsupported fields")
    decisions = document["decisions"]
    if not isinstance(decisions, list):
        raise ValueError("oMLX decisions must be an array")
    result: dict[tuple[str, str], int] = {}
    for item in decisions:
        if not isinstance(item, dict) or set(item) != {"topic", "paper_id", "provisional_label"}:
            raise ValueError("oMLX decision has unsupported fields")
        topic, paper_id, label = item["topic"], item["paper_id"], item["provisional_label"]
        if not isinstance(topic, str) or not isinstance(paper_id, str) or type(label) is not int or label not in range(4):
            raise ValueError("oMLX decision has an invalid topic, paper_id, or label")
        key = topic, paper_id
        if key in result:
            raise ValueError("oMLX response repeats topic-paper_id pair")
        result[key] = label
    if set(result) != expected_keys:
        raise ValueError("oMLX response does not exactly preserve batch topic-paper_id pairs")
    return result


def _http_transport(endpoint: str, api_key: str | None, timeout_seconds: float) -> Transport:
    def post(payload: Mapping[str, Any]) -> Mapping[str, Any]:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = Request(
            endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urlopen(request, timeout=timeout_seconds) as response:
            document = json.loads(response.read().decode("utf-8"))
        if not isinstance(document, dict):
            raise ValueError("oMLX response must be a JSON object")
        return document
    return post


def _write_new(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(document, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def suggest(
    *,
    worklist_path: Path,
    model_lock_path: Path,
    output_path: Path,
    evidence_path: Path,
    transport: Transport,
    batch_size: int = 8,
    concurrency: int = 4,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run all batches once, then publish exact-cover results and redacted evidence."""

    if not 6 <= batch_size <= 8:
        raise ValueError("batch_size must be between 6 and 8")
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    if output_path.resolve() == evidence_path.resolve():
        raise ValueError("output and evidence paths must differ")
    for path in (output_path, evidence_path):
        if path.exists():
            raise FileExistsError(f"no-replace output already exists: {path}")

    worklist = load_worklist(worklist_path)
    model_id, model_lock_sha256 = load_model_lock(model_lock_path)
    rows = sorted(worklist.rows, key=lambda row: row.key)
    batches = [rows[index:index + batch_size] for index in range(0, len(rows), batch_size)]
    started_at = _utc_now()

    def call_batch(batch: list[CurationWorklistRow]) -> dict[tuple[str, str], int]:
        return _extract_decisions(transport(_request_for_batch(batch, model_id)), {row.key for row in batch})

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_by_index = {index: executor.submit(call_batch, batch) for index, batch in enumerate(batches)}
        labels_by_key = {
            key: label
            for index in range(len(batches))
            for key, label in future_by_index[index].result().items()
        }
    if set(labels_by_key) != {row.key for row in rows}:
        raise ValueError("batch responses do not exactly cover the worklist")

    decisions = CurationDecisions(
        worklist_hash=worklist.hash(),
        rows=tuple(
            CurationDecision(
                topic=row.topic,
                paper_id=row.paper_id,
                provisional_label=labels_by_key[row.key],
                hard_negative=labels_by_key[row.key] <= 1,
                hard_positive=(labels_by_key[row.key] == 3 and (row.abstract_incomplete or row.cross_language_match)),
                source="model_provisional",
            )
            for row in rows
        ),
    )
    decisions_document = decisions.document()
    try:
        validate(decisions_document, "stage2-curation-decisions.schema.json")
    except SchemaValidationError as error:
        raise ValueError(str(error)) from error
    completed_at = _utc_now()
    label_counts = {str(label): sum(row.provisional_label == label for row in decisions.rows) for label in range(4)}
    language_counts = {language: sum(row.language == language for row in rows) for language in sorted({row.language for row in rows})}
    evidence = {
        "schema_version": 1,
        "kind": "stage2_curation_suggestion_evidence",
        "worklist_hash": worklist.hash(),
        "model_lock_sha256": model_lock_sha256,
        "model_id": model_id,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": _sha256_bytes(SYSTEM_PROMPT.encode("utf-8")),
        "request_count": len(batches),
        "case_count": len(rows),
        "label_counts": label_counts,
        "hard_negative_count": sum(row.hard_negative for row in decisions.rows),
        "hard_positive_count": sum(row.hard_positive for row in decisions.rows),
        "language_counts": language_counts,
        "started_at": started_at,
        "completed_at": completed_at,
    }
    _write_new(output_path, decisions_document)
    _write_new(evidence_path, evidence)
    return decisions_document, evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worklist", required=True, type=Path)
    parser.add_argument("--model-lock", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--evidence-output", required=True, type=Path)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8080/v1/chat/completions")
    parser.add_argument("--api-key-env", default="PAPER_AGENT_OMLX_API_KEY")
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--batch-size", type=int, choices=range(6, 9), default=8)
    parser.add_argument("--concurrency", type=int, default=4)
    arguments = parser.parse_args(argv)
    api_key = os.environ.get(arguments.api_key_env) if arguments.api_key_env else None
    _, evidence = suggest(
        worklist_path=arguments.worklist,
        model_lock_path=arguments.model_lock,
        output_path=arguments.output,
        evidence_path=arguments.evidence_output,
        transport=_http_transport(arguments.endpoint, api_key, arguments.timeout_seconds),
        batch_size=arguments.batch_size,
        concurrency=arguments.concurrency,
    )
    print(json.dumps({"status": "complete", "case_count": evidence["case_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
