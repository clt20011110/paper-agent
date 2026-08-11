#!/usr/bin/env python3
"""Freeze a real, evaluator-private Crossref Stage 2 sampling population.

The script deliberately does not create labels.  It records all raw Crossref
responses beside the private corpus snapshot, then uses the production loader
and HIDDEN_REAL selector to prove the frozen natural frame is usable.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
import html
import json
import os
from pathlib import Path
import re
from tempfile import mkdtemp, mkstemp
from typing import Any, Mapping, Protocol, Sequence

from paper_agent.canonical import content_hash
from paper_agent.http_transport import ControlledHTTPTransport
from paper_agent.identity import normalize_doi as identity_normalize_doi, paper_id_for
from paper_agent.provider_runtime import ProviderRuntime, ProviderRuntimePolicy
from paper_agent.stage2_sampling import (
    CorpusPaper,
    PrivateCorpusSnapshot,
    SamplingPolicy,
    load_private_corpus_snapshot,
    private_corpus_snapshot_from_document,
    select_hidden_real,
    write_private_corpus_snapshot,
)


_TAG = re.compile(r"<[^>]+>")
_SPACE = re.compile(r"\s+")
_HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_TOPIC_ID = re.compile(r"^[a-z0-9_]+$")


class CrossrefTransport(Protocol):
    last_response_body: bytes | None

    def __call__(self, provider: str, operation: str, parameters: Mapping[str, Any]) -> Mapping[str, Any]: ...


@dataclass(frozen=True, slots=True)
class Capture:
    topic: str
    query_language: str
    query: str
    parameters: Mapping[str, Any]
    body: bytes

    @property
    def filename(self) -> str:
        return f"{self.topic}-{self.query_language}.json"


@dataclass(frozen=True, slots=True)
class FrozenSnapshot:
    snapshot: PrivateCorpusSnapshot
    captures: tuple[Capture, ...]
    query_spec_hash: str


def _text(value: object) -> str | None:
    if isinstance(value, list):
        value = value[0] if value else None
    if value is None:
        return None
    cleaned = _SPACE.sub(" ", _TAG.sub(" ", html.unescape(str(value)))).strip()
    return cleaned or None


def normalize_doi(value: str | None) -> str | None:
    doi = identity_normalize_doi(value)
    return doi if doi and doi.startswith("10.") and "/" in doi else None


def _crossref_items(payload: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    message = payload.get("message")
    items = message.get("items") if isinstance(message, Mapping) else None
    if not isinstance(items, list):
        raise ValueError("Crossref search response has no message.items list")
    return tuple(item for item in items if isinstance(item, Mapping))


def _relation_dois(record: Mapping[str, Any]) -> set[str]:
    """Return DOI endpoints explicitly declared in a Crossref relation."""

    output: set[str] = set()

    def visit(value: object) -> None:
        if isinstance(value, Mapping):
            kind = value.get("id-type") or value.get("id_type")
            identifier = value.get("id")
            if str(kind).casefold() == "doi" and isinstance(identifier, str):
                normalized = normalize_doi(identifier)
                if normalized:
                    output.add(normalized)
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    visit(record.get("relation"))
    return output


def _family_by_doi(relations: Mapping[str, set[str]]) -> Mapping[str, str]:
    parent = {doi: doi for doi in relations}

    def find(doi: str) -> str:
        while parent[doi] != doi:
            parent[doi] = parent[parent[doi]]
            doi = parent[doi]
        return doi

    for doi, related in relations.items():
        for other in related:
            parent.setdefault(other, other)
            left, right = find(doi), find(other)
            if left != right:
                parent[right] = left
    members: dict[str, set[str]] = {}
    for doi in parent:
        members.setdefault(find(doi), set()).add(doi)
    return {
        doi: f"doi:{next(iter(group))}" if len(group) == 1 else f"doi-family:{content_hash(sorted(group))}"
        for doi in relations
        for group in (members[find(doi)],)
    }


def _query_spec(document: object) -> tuple[str, int, str, str, str, tuple[tuple[str, str, str, int], ...]]:
    if not isinstance(document, dict) or set(document) != {
        "schema_version", "provider", "sampling_policy", "scope", "topics"
    }:
        raise ValueError("query spec has unsupported fields")
    if document["schema_version"] != 1 or document["provider"] != "crossref":
        raise ValueError("query spec must be Crossref schema version 1")
    policy = document["sampling_policy"]
    scope = document["scope"]
    topics = document["topics"]
    if not isinstance(policy, dict) or set(policy) != {"version", "seed"}:
        raise ValueError("query spec sampling_policy is invalid")
    if not isinstance(scope, dict) or set(scope) != {"from_pub_date", "until_pub_date", "type"}:
        raise ValueError("query spec scope is invalid")
    if not isinstance(scope["type"], str) or not scope["type"].strip():
        raise ValueError("query spec must declare a Crossref work type")
    if not isinstance(topics, list) or not 6 <= len(topics) <= 8:
        raise ValueError("query spec must contain six to eight bilingual topics")
    queries: list[tuple[str, str, str, int]] = []
    for topic in topics:
        if (
            not isinstance(topic, dict)
            or set(topic) != {"id", "queries"}
            or not isinstance(topic["id"], str)
            or not _TOPIC_ID.fullmatch(topic["id"])
        ):
            raise ValueError("query spec topic is invalid")
        rows = topic["queries"]
        if not isinstance(rows, list) or len(rows) != 2:
            raise ValueError("each topic must have English and Chinese queries")
        expected = {"en", "zh"}
        seen: set[str] = set()
        for item in rows:
            if not isinstance(item, dict) or set(item) != {"language", "query", "rows"}:
                raise ValueError("query spec query is invalid")
            language, query, count = item["language"], item["query"], item["rows"]
            if (
                language not in expected
                or language in seen
                or not isinstance(query, str)
                or not query.strip()
                or not isinstance(count, int)
                or count <= 0
            ):
                raise ValueError("query spec must use unique EN/ZH queries with positive rows")
            seen.add(language)
            queries.append((topic["id"], language, query, count))
    if len({topic for topic, _, _, _ in queries}) != len(topics):
        raise ValueError("query spec topic ids must be unique")
    return (
        str(policy["version"]),
        int(policy["seed"]),
        str(scope["from_pub_date"]),
        str(scope["until_pub_date"]),
        str(scope["type"]),
        tuple(queries),
    )


def freeze_snapshot(document: object, transport: CrossrefTransport) -> FrozenSnapshot:
    """Fetch all spec queries, construct a natural corpus, and validate it."""

    policy_version, seed, date_from, date_to, work_type, queries = _query_spec(document)
    captures: list[Capture] = []
    by_topic: dict[str, dict[str, dict[str, Any]]] = {}
    relations: dict[str, set[str]] = {}
    for topic, query_language, query, rows in queries:
        parameters = {
            "query.title": query,
            "page_size": rows,
            "date_from": date_from,
            "date_to": date_to,
            "filter": f"type:{work_type},from-pub-date:{date_from},until-pub-date:{date_to}",
        }
        payload = transport("crossref", "search", parameters)
        body = transport.last_response_body
        if not isinstance(body, bytes):
            raise ValueError("Crossref transport did not retain raw response bytes")
        captures.append(Capture(topic, query_language, query, parameters, body))
        raw_response_sha256 = sha256(body).hexdigest()
        topic_rows = by_topic.setdefault(topic, {})
        for item in _crossref_items(payload):
            doi_value = item.get("DOI")
            doi = normalize_doi(doi_value) if isinstance(doi_value, str) else None
            title = _text(item.get("title"))
            if doi is None or title is None:
                continue
            relations.setdefault(doi, set()).update(_relation_dois(item))
            title_language = "zh" if _HAN.search(title) else "en"
            abstract = _text(item.get("abstract"))
            topic_rows.setdefault(
                doi,
                {
                    "topic": topic,
                    "doi": doi,
                    "title": title,
                    "abstract": abstract,
                    "query": query,
                    "query_language": query_language,
                    "title_language": title_language,
                    "raw_response_sha256": raw_response_sha256,
                    "crossref_record": dict(item),
                },
            )
    family_by_doi = _family_by_doi(relations)
    provisional = [
        CorpusPaper(
            topic=row["topic"],
            paper_id=paper_id_for(doi=row["doi"]),
            title=row["title"],
            abstract=row["abstract"],
            metadata={
                "topic": row["topic"],
                "query": row["query"],
                "query_language": row["query_language"],
                "title_language": row["title_language"],
                "raw_response_sha256": row["raw_response_sha256"],
                "doi": row["doi"],
                "crossref_record": row["crossref_record"],
            },
            source="crossref",
            language=row["title_language"],
            paper_family=family_by_doi[row["doi"]],
            sampling_weight=1.0,
            sampling_probability=1.0,
            abstract_incomplete=row["abstract"] is None,
            natural_crawler_population=True,
            cross_language_match=row["query_language"] != row["title_language"],
        )
        for topic in sorted(by_topic)
        for row in by_topic[topic].values()
    ]
    if len(provisional) < 150:
        raise ValueError("Crossref sampling frame has fewer than 150 natural rows")
    probability = 150 / len(provisional)
    snapshot = PrivateCorpusSnapshot(
        1,
        policy_version,
        seed,
        tuple(
            CorpusPaper(**{**paper.document(), "sampling_probability": probability})
            for paper in sorted(provisional, key=lambda paper: paper.key)
        ),
    )
    # The production verifier proves schema/hash compatibility and the pre-label draw.
    restored = private_corpus_snapshot_from_document(snapshot.document())
    select_hidden_real(restored, SamplingPolicy(policy_version, seed))
    return FrozenSnapshot(snapshot, tuple(captures), content_hash(document))


def _write_bytes_no_replace(path: Path, content: bytes) -> None:
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


def _capture_manifest(frozen: FrozenSnapshot) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "provider": "crossref",
        "query_spec_hash": frozen.query_spec_hash,
        "snapshot_hash": frozen.snapshot.hash(),
        "captured_at": datetime.now(UTC).isoformat(),
        "responses": [
            {
                "topic": capture.topic,
                "query_language": capture.query_language,
                "query": capture.query,
                "parameters": dict(capture.parameters),
                "filename": capture.filename,
                "sha256": sha256(capture.body).hexdigest(),
                "size_bytes": len(capture.body),
            }
            for capture in frozen.captures
        ],
    }


def publish_snapshot(
    frozen: FrozenSnapshot,
    *,
    output: Path,
    capture_directory: Path,
    capture_manifest: Path,
) -> None:
    """Publish a completed snapshot and raw captures without replacing files."""

    targets = (output, capture_manifest, capture_directory)
    if any(os.path.lexists(path) for path in targets):
        existing = next(path for path in targets if os.path.lexists(path))
        raise FileExistsError(f"output already exists: {existing}")
    if len({path.resolve() for path in targets}) != len(targets):
        raise ValueError("snapshot, capture manifest, and capture directory must differ")
    staging_parent = capture_directory.parent
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(mkdtemp(dir=staging_parent, prefix=f".{capture_directory.name}."))
    for capture in frozen.captures:
        _write_bytes_no_replace(staging / capture.filename, capture.body)
    staged_snapshot = staging / "private-snapshot.json"
    write_private_corpus_snapshot(staged_snapshot, frozen.snapshot)
    restored = load_private_corpus_snapshot(staged_snapshot)
    select_hidden_real(restored, SamplingPolicy(restored.sampling_policy_version, restored.sampling_seed))
    _write_bytes_no_replace(output, staged_snapshot.read_bytes())
    staged_snapshot.unlink()
    _write_bytes_no_replace(
        capture_manifest,
        (json.dumps(_capture_manifest(frozen), ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"),
    )
    os.replace(staging, capture_directory)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-spec", type=Path, default=Path("configs/stage2/real-sampling-crossref-v1.json"))
    parser.add_argument("--output", required=True, type=Path, help="new private corpus snapshot path")
    parser.add_argument("--capture-directory", required=True, type=Path, help="new directory for raw Crossref response bytes")
    parser.add_argument("--capture-manifest", required=True, type=Path, help="new raw-capture manifest path")
    parser.add_argument("--contact", required=True, help="Crossref mailto address or contact URL")
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    document = json.loads(args.query_spec.read_text(encoding="utf-8"))
    runtime = ProviderRuntime({
        "crossref": ProviderRuntimePolicy("crossref", queries_per_second=1.0, max_concurrency=1)
    })
    transport = ControlledHTTPTransport(args.contact, timeout_seconds=args.timeout_seconds, runtime=runtime)
    frozen = freeze_snapshot(document, transport)
    publish_snapshot(
        frozen,
        output=args.output,
        capture_directory=args.capture_directory,
        capture_manifest=args.capture_manifest,
    )
    print(json.dumps({
        "status": "complete",
        "snapshot_hash": frozen.snapshot.hash(),
        "corpus_hash": frozen.snapshot.corpus_hash,
        "natural_rows": len(frozen.snapshot.papers),
        "sampling_probability": 150 / len(frozen.snapshot.papers),
        "raw_responses": len(frozen.captures),
        "query_spec_hash": frozen.query_spec_hash,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
