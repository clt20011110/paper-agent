"""Deterministic import of user-authorized library seeds."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from .canonical import canonical_json, content_hash
from .domain import SourceBatch
from .manifests import load_catalog
from .providers.api import SeedInput
from .providers.builtin import FixtureTransport, create_builtin, manifest_from_document
from .repository import PaperRepository
from .runs import RunStatus, RunStore
from .search_execution import seed_input
from .storage import Database
from .verification import MetadataCoordinator, ProviderTrust


@dataclass(frozen=True, slots=True)
class SeedImportResult:
    run_id: str
    input_count: int
    imported_count: int
    paper_ids: tuple[str, ...]


def inputs_from_files(paths: Sequence[Path]) -> tuple[SeedInput, ...]:
    return tuple(seed for path in paths for seed in _inputs_from_file(path))


def import_seeds(
    database_path: Path,
    inputs: Sequence[SeedInput],
    *,
    run_id: str | None = None,
    observed_at: str | None = None,
) -> SeedImportResult:
    if not inputs:
        raise ValueError("import-seeds requires at least one --seed or --input")
    frozen_inputs = tuple(inputs)
    manifest, batch = _validated_seed_batch(frozen_inputs)
    input_hash = content_hash(
        [
            {"kind": item.kind, "value": item.value, "source_name": item.source_name}
            for item in frozen_inputs
        ]
    )
    resolved_run_id = run_id or f"seed-import-{input_hash[:16]}"
    timestamp = observed_at or datetime.now(UTC).isoformat().replace("+00:00", "Z")
    with Database(database_path) as database:
        database.migrate()
        runs = RunStore(database)
        run = runs.create(
            run_id=resolved_run_id,
            stage="stage-1-import-seeds",
            input_hash=input_hash,
            config_hash=content_hash(manifest),
            implementation_version="phase2-library-v1",
        )
        if run.status is RunStatus.DRAFT:
            runs.transition(resolved_run_id, RunStatus.APPROVED, at=timestamp)
            runs.transition(resolved_run_id, RunStatus.RUNNING, at=timestamp)
        coordinator = MetadataCoordinator(
            PaperRepository(database),
            {"user_library": ProviderTrust.from_manifest(manifest)},
        )
        papers = coordinator.merge_batch(batch)
        if runs.get(resolved_run_id).status is RunStatus.RUNNING:  # type: ignore[union-attr]
            runs.transition(resolved_run_id, RunStatus.COMPLETE, at=timestamp)

    paper_ids = tuple(sorted(paper.paper_id for paper in papers))
    return SeedImportResult(resolved_run_id, len(frozen_inputs), len(paper_ids), paper_ids)


def validate_seed_inputs(inputs: Sequence[SeedInput]) -> None:
    """Run the exact parser/provider contract without creating pipeline storage."""
    if not inputs:
        raise ValueError("import-seeds requires at least one --seed or --input")
    _validated_seed_batch(tuple(inputs))


def _validated_seed_batch(
    inputs: tuple[SeedInput, ...],
) -> tuple[Mapping[str, Any], SourceBatch]:
    catalog = load_catalog()
    manifest = catalog.provider("user_library")
    provider = create_builtin(
        "user_library",
        FixtureTransport({}),
        manifest_from_document(manifest),
    )
    return manifest, provider.import_seeds(inputs)


def _inputs_from_file(path: Path) -> tuple[SeedInput, ...]:
    suffix = path.suffix.casefold()
    if suffix == ".pdf":
        return (SeedInput("local_pdf", str(path.resolve()), path.name),)
    text = path.read_text(encoding="utf-8")
    if suffix == ".bib":
        records = tuple(record.strip() for record in re.split(r"(?m)(?=^@\w+\s*[({])", text) if record.strip())
        return tuple(SeedInput("bibtex", record, path.name) for record in records)
    if suffix == ".ris":
        records = tuple(record.strip() for record in re.findall(r"(?ms)^TY  - .*?^ER  -.*?$", text))
        return tuple(SeedInput("ris", record, path.name) for record in records)
    if suffix == ".json":
        payload = json.loads(text)
        records = payload.get("items", ()) if isinstance(payload, Mapping) and "items" in payload else payload
        values = records if isinstance(records, list) else [records]
        return tuple(
            SeedInput("csl-json", canonical_json(_csl_record(value)).decode("utf-8"), path.name)
            for value in values
        )
    return tuple(
        SeedInput(item.kind, item.value, path.name)
        for line in text.splitlines()
        if (value := line.strip()) and not value.startswith("#")
        for item in (seed_input(value),)
    )


def _csl_record(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("JSON bibliography records must be objects")
    record = value.get("data", value)
    if not isinstance(record, Mapping):
        raise ValueError("Zotero data must be an object")
    if "itemType" not in record:
        return record
    creators = record.get("creators", ())
    authors = [
        {
            "given": creator.get("firstName", ""),
            "family": creator.get("lastName", ""),
            "literal": creator.get("name", ""),
        }
        for creator in creators
        if isinstance(creator, Mapping) and creator.get("creatorType", "author") == "author"
    ]
    year = re.search(r"\b(\d{4})\b", str(record.get("date", "")))
    return {
        "id": value.get("key") or record.get("key") or record.get("DOI") or record.get("title"),
        "title": record["title"],
        "author": authors,
        "DOI": record.get("DOI"),
        "issued": {"date-parts": [[int(year.group(1))]]} if year else {},
    }
