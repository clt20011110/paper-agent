#!/usr/bin/env python3
"""Freeze one official NeurIPS listing into normalized, label-free papers."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import mkstemp
from typing import Any, Mapping, Protocol

from paper_agent.canonical import content_hash
from paper_agent.http_transport import ControlledHTTPTransport
from paper_agent.identity import paper_id_for
from paper_agent.provider_runtime import ProviderRuntime, ProviderRuntimePolicy


class Transport(Protocol):
    last_response_body: bytes | None
    last_response_sha256: str | None

    def __call__(self, provider: str, operation: str, parameters: Mapping[str, Any]) -> Mapping[str, Any]: ...


def freeze_year(year: int, transport: Transport) -> tuple[dict[str, Any], dict[str, Any], bytes]:
    """Fetch every normalized page while retaining one exact official HTML body."""

    if year < 1987:
        raise ValueError("NeurIPS year must be 1987 or later")
    cursor: str | None = None
    raw_html: bytes | None = None
    raw_hash: str | None = None
    entries: dict[str, dict[str, Any]] = {}
    page_count = 0
    while True:
        payload = transport(
            "neurips_proceedings",
            "discover",
            {"series": "NeurIPS", "year": year, "page_size": 1_000, "cursor": cursor},
        )
        body = transport.last_response_body
        if not isinstance(body, bytes):
            raise ValueError("NeurIPS transport did not retain official HTML bytes")
        body_hash = sha256(body).hexdigest()
        if raw_html is None:
            raw_html, raw_hash = body, body_hash
        elif body_hash != raw_hash:
            raise ValueError("NeurIPS listing changed while paginating")
        values = payload.get("entries")
        if not isinstance(values, list):
            raise ValueError("NeurIPS discovery response has no entries list")
        for entry in values:
            if not isinstance(entry, dict):
                raise ValueError("NeurIPS discovery entry is invalid")
            external_id = entry.get("external_id")
            title = entry.get("title")
            if not isinstance(external_id, str) or not isinstance(title, str) or not title:
                raise ValueError("NeurIPS discovery entry lacks external_id or title")
            paper_id = paper_id_for(provider="neurips_proceedings", external_id=external_id)
            entries[paper_id] = {
                "paper_id": paper_id,
                "title": title,
                "abstract": None,
                "keywords": [],
                "document_type": entry.get("document_type"),
                "possibly_truncated": False,
                "multi_condition_conflict": False,
                "language_anomaly": False,
            }
        page_count += 1
        cursor_value = payload.get("next_cursor")
        cursor = cursor_value if isinstance(cursor_value, str) and cursor_value else None
        if cursor is None:
            break
    if raw_html is None or raw_hash is None or not entries:
        raise ValueError("NeurIPS listing yielded no papers")
    papers = {
        "schema_version": "1",
        "kind": "stage2_benchmark_papers",
        "papers": [entries[key] for key in sorted(entries)],
    }
    manifest = {
        "schema_version": 1,
        "kind": "stage2_neurips_capture",
        "provider": "neurips_proceedings",
        "year": year,
        "captured_at": datetime.now(UTC).isoformat(),
        "page_count": page_count,
        "paper_count": len(entries),
        "raw_html": {"sha256": raw_hash, "size_bytes": len(raw_html)},
        "papers_corpus_hash": content_hash(papers),
    }
    return papers, manifest, raw_html


def _write_no_replace(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def publish(*, papers: Mapping[str, Any], manifest: Mapping[str, Any], raw_html: bytes, papers_output: Path, manifest_output: Path, raw_html_output: Path) -> None:
    targets = (papers_output, manifest_output, raw_html_output)
    if len({path.resolve() for path in targets}) != len(targets):
        raise ValueError("all output paths must differ")
    existing = next((path for path in targets if os.path.lexists(path)), None)
    if existing is not None:
        raise FileExistsError(f"output already exists: {existing}")
    _write_no_replace(raw_html_output, raw_html)
    _write_no_replace(
        papers_output,
        (json.dumps(papers, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(),
    )
    _write_no_replace(
        manifest_output,
        (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(),
    )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", required=True, type=int)
    parser.add_argument("--papers-output", required=True, type=Path)
    parser.add_argument("--capture-manifest-output", required=True, type=Path)
    parser.add_argument("--raw-html-output", required=True, type=Path)
    parser.add_argument("--contact", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    runtime = ProviderRuntime({
        "neurips_proceedings": ProviderRuntimePolicy("neurips_proceedings", queries_per_second=1.0, max_concurrency=1)
    })
    transport = ControlledHTTPTransport(args.contact, timeout_seconds=args.timeout_seconds, runtime=runtime)
    papers, manifest, raw_html = freeze_year(args.year, transport)
    publish(
        papers=papers,
        manifest=manifest,
        raw_html=raw_html,
        papers_output=args.papers_output,
        manifest_output=args.capture_manifest_output,
        raw_html_output=args.raw_html_output,
    )
    print(json.dumps({"status": "complete", "year": args.year, "paper_count": len(papers["papers"]), "papers_corpus_hash": manifest["papers_corpus_hash"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
