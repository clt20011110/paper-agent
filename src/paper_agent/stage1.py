"""Standalone, fail-closed Stage 1 venue metadata collection.

This service deliberately does not depend on QueryPlan approval, Stage 2, the
canonical database, PDF acquisition, or model execution.  Its unit of work is
one venue and one publication year.  A result is only called complete when the
primary source supplies an auditable census and every page reaches a terminal
cursor without unexplained parser loss.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, replace
from hashlib import sha256
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Callable, Mapping, Sequence
from uuid import NAMESPACE_URL, uuid5

from .canonical import canonical_json
from .domain import EnvelopeStatus, SourceBatch, SourceEntry
from .manifests import ManifestCatalog, load_catalog
from .providers.api import CrawlWindow, VenueAdapter, VenueDescriptor


STAGE1_INTERFACE_VERSION = "stage1-standalone-v1"
_OUTPUT_FIELDS = (
    "title",
    "abstract",
    "authors",
    "doi",
    "publication_date",
    "year",
    "venue_name",
    "landing_url",
    "pdf_url",
    "volume",
    "issue",
    "pages",
    "keywords",
)


class Stage1RequestError(ValueError):
    """The requested venue/year collection cannot be executed as specified."""


class Stage1IncompleteError(RuntimeError):
    """Strict publication was refused because completeness was not proven."""

    def __init__(self, result: "Stage1Result") -> None:
        self.result = result
        units = ", ".join(
            f"{unit.venue_id}:{unit.year}={unit.status}"
            for unit in result.receipt.units
            if unit.status != "complete"
        )
        super().__init__(f"Stage 1 completeness is not proven: {units}")


@dataclass(frozen=True, slots=True)
class Stage1Request:
    venue_ids: tuple[str, ...]
    year_from: int
    year_to: int
    page_size: int = 500
    max_workers: int = 4
    strict: bool = True

    def __post_init__(self) -> None:
        normalized = tuple(sorted(set(self.venue_ids)))
        if not normalized or any(not value.strip() for value in normalized):
            raise Stage1RequestError("at least one non-empty venue_id is required")
        if normalized != self.venue_ids:
            object.__setattr__(self, "venue_ids", normalized)
        if self.year_from < 1900 or self.year_to > 2200 or self.year_from > self.year_to:
            raise Stage1RequestError("year range is invalid")
        if not 1 <= self.page_size <= 1000:
            raise Stage1RequestError("page_size must be between 1 and 1000")
        if not 1 <= self.max_workers <= 64:
            raise Stage1RequestError("max_workers must be between 1 and 64")


@dataclass(frozen=True, slots=True)
class Stage1UnitReceipt:
    venue_id: str
    venue_name: str
    venue_type: str
    provider: str
    year: int
    status: str
    pages_fetched: int
    terminal_cursor_reached: bool
    returned_records: int
    unique_records: int
    expected_total: int | None
    parser_raw_records: int | None
    parser_rejected_records: int | None
    parser_excluded_records: int | None = None
    duplicate_external_ids: tuple[str, ...] = ()
    response_hashes: tuple[str, ...] = ()
    request_hashes: tuple[str, ...] = ()
    field_coverage: Mapping[str, int] = field(default_factory=dict)
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Stage1CompletenessReceipt:
    schema_version: str
    interface_version: str
    run_id: str
    request: Mapping[str, Any]
    status: str
    units: tuple[Stage1UnitReceipt, ...]
    metadata_sha256: str

    def document(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Stage1Result:
    run_id: str
    status: str
    records: tuple[Mapping[str, Any], ...]
    receipt: Stage1CompletenessReceipt
    output_path: Path | None = None
    receipt_path: Path | None = None

    @property
    def complete(self) -> bool:
        return self.status == "complete"


AdapterFactory = Callable[[VenueDescriptor], VenueAdapter]


@dataclass(slots=True)
class CensusCapturingAdapter:
    """Attach transport-native census data without changing provider ABI.

    Existing third-party and frozen built-in adapters continue returning the
    versioned ``SourceBatch`` contract.  The standalone service may wrap an
    adapter with a transport that exposes its last native payload.
    """

    delegate: VenueAdapter
    transport: Any

    @property
    def manifest(self):
        return self.delegate.manifest

    def discover(
        self,
        descriptor: VenueDescriptor,
        window: CrawlWindow,
        cursor: str | None = None,
    ) -> SourceBatch:
        batch = self.delegate.discover(descriptor, window, cursor)
        payload = getattr(self.transport, "last_payload", None)
        census = payload.get("census") if isinstance(payload, Mapping) else None
        return replace(
            batch,
            census=dict(census) if isinstance(census, Mapping) else batch.census,
        )


def collect_stage1_metadata(
    request: Stage1Request,
    *,
    adapter_factory: AdapterFactory,
    catalog: ManifestCatalog | None = None,
    run_id: str | None = None,
) -> Stage1Result:
    """Collect every requested venue/year and return deterministic metadata.

    ``strict`` controls publication by :func:`write_stage1_result`; collection
    itself always returns its complete audit so callers can inspect failures.
    """

    resolved_catalog = catalog or load_catalog()
    unknown = sorted(set(request.venue_ids) - set(resolved_catalog.venues))
    if unknown:
        raise Stage1RequestError(f"unknown venue_id(s): {', '.join(unknown)}")
    request_document = {
        "venue_ids": list(request.venue_ids),
        "year_from": request.year_from,
        "year_to": request.year_to,
        "page_size": request.page_size,
        "max_workers": request.max_workers,
        "strict": request.strict,
    }
    resolved_run_id = run_id or str(
        uuid5(NAMESPACE_URL, canonical_json(request_document).decode("utf-8"))
    )
    jobs = [
        (venue_id, year)
        for venue_id in request.venue_ids
        for year in range(request.year_from, request.year_to + 1)
    ]
    completed: dict[tuple[str, int], tuple[tuple[Mapping[str, Any], ...], Stage1UnitReceipt]] = {}
    workers = min(request.max_workers, len(jobs))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="stage1") as pool:
        future_jobs = {
            pool.submit(
                _collect_unit,
                venue_id,
                year,
                request.page_size,
                resolved_catalog,
                adapter_factory,
                resolved_run_id,
            ): (venue_id, year)
            for venue_id, year in jobs
        }
        for future in as_completed(future_jobs):
            key = future_jobs[future]
            try:
                completed[key] = future.result()
            except Exception as error:  # one venue-year must not cancel siblings
                venue = resolved_catalog.venue(key[0])
                completed[key] = (
                    (),
                    Stage1UnitReceipt(
                        venue_id=key[0],
                        venue_name=str(venue["name"]),
                        venue_type=str(venue["venue_type"]),
                        provider=str(venue["primary_provider"]),
                        year=key[1],
                        status="failed",
                        pages_fetched=0,
                        terminal_cursor_reached=False,
                        returned_records=0,
                        unique_records=0,
                        expected_total=None,
                        parser_raw_records=None,
                        parser_rejected_records=None,
                        reasons=(f"{type(error).__name__}: {error}",),
                    ),
                )

    ordered_units = tuple(completed[key][1] for key in sorted(completed))
    records = tuple(
        record
        for key in sorted(completed)
        for record in completed[key][0]
    )
    metadata_bytes = b"".join(canonical_json(dict(record)) + b"\n" for record in records)
    status = (
        "complete"
        if all(unit.status in {"complete", "not_applicable"} for unit in ordered_units)
        else "incomplete"
    )
    receipt = Stage1CompletenessReceipt(
        schema_version="1",
        interface_version=STAGE1_INTERFACE_VERSION,
        run_id=resolved_run_id,
        request=request_document,
        status=status,
        units=ordered_units,
        metadata_sha256=sha256(metadata_bytes).hexdigest(),
    )
    return Stage1Result(resolved_run_id, status, records, receipt)


def write_stage1_result(
    result: Stage1Result,
    *,
    output_path: Path,
    receipt_path: Path | None = None,
    allow_incomplete: bool = False,
) -> Stage1Result:
    """Atomically publish JSONL plus receipt, refusing unproven data by default."""

    resolved_receipt = receipt_path or output_path.with_suffix(output_path.suffix + ".receipt.json")
    _atomic_write(resolved_receipt, canonical_json(result.receipt.document()) + b"\n")
    if not result.complete and not allow_incomplete:
        raise Stage1IncompleteError(
            Stage1Result(
                result.run_id,
                result.status,
                result.records,
                result.receipt,
                None,
                resolved_receipt,
            )
        )
    payload = b"".join(canonical_json(dict(record)) + b"\n" for record in result.records)
    _atomic_write(output_path, payload)
    return Stage1Result(
        result.run_id,
        result.status,
        result.records,
        result.receipt,
        output_path,
        resolved_receipt,
    )


def venue_catalog_document(catalog: ManifestCatalog | None = None) -> list[dict[str, Any]]:
    resolved = catalog or load_catalog()
    return [
        {
            "venue_id": venue_id,
            "name": document["name"],
            "venue_type": document["venue_type"],
            "primary_provider": document["primary_provider"],
            "official_url": document.get("official_url"),
            "date_range": document.get("date_range"),
        }
        for venue_id, document in sorted(resolved.venues.items())
    ]


def _collect_unit(
    venue_id: str,
    year: int,
    page_size: int,
    catalog: ManifestCatalog,
    adapter_factory: AdapterFactory,
    run_id: str,
) -> tuple[tuple[Mapping[str, Any], ...], Stage1UnitReceipt]:
    venue = catalog.venue(venue_id)
    date_range = venue.get("date_range")
    if isinstance(date_range, Mapping):
        start_year = int(str(date_range["start"])[:4])
        end_year = int(str(date_range["end"])[:4])
        if year < start_year or year > end_year:
            return (), Stage1UnitReceipt(
                venue_id=venue_id,
                venue_name=str(venue["name"]),
                venue_type=str(venue["venue_type"]),
                provider=str(venue["primary_provider"]),
                year=year,
                status="not_applicable",
                pages_fetched=0,
                terminal_cursor_reached=True,
                returned_records=0,
                unique_records=0,
                expected_total=0,
                parser_raw_records=0,
                parser_rejected_records=0,
                parser_excluded_records=0,
                field_coverage={field: 0 for field in _OUTPUT_FIELDS},
                reasons=(
                    f"venue date_range is {date_range['start']} through {date_range['end']}",
                ),
            )
    descriptor = catalog.runtime_venue(venue_id)
    descriptor = VenueDescriptor(
        descriptor.schema_version,
        descriptor.venue_id,
        descriptor.provider,
        descriptor.adapter,
        {**descriptor.parameters, "page_size": page_size, "stage1_run_id": run_id},
    )
    adapter = adapter_factory(descriptor)
    window = CrawlWindow(
        date_from=f"{year:04d}-01-01",
        date_to=f"{year:04d}-12-31",
        year=year,
    )
    batches: list[SourceBatch] = []
    cursor: str | None = None
    seen_cursors: set[str | None] = set()
    reasons: list[str] = []
    while True:
        if cursor in seen_cursors:
            reasons.append(f"cursor cycle detected at {cursor!r}")
            break
        seen_cursors.add(cursor)
        batch = adapter.discover(descriptor, window, cursor)
        batches.append(batch)
        if batch.status is not EnvelopeStatus.SUCCESS:
            reasons.append(batch.error or f"provider returned {batch.status.value}")
            break
        if batch.next_cursor is None:
            break
        cursor = batch.next_cursor

    entries = [entry for batch in batches for entry in batch.entries]
    duplicates = _duplicates(entry.external_id for entry in entries)
    unique: dict[str, SourceEntry] = {}
    for entry in entries:
        unique.setdefault(entry.external_id, entry)
    for entry in unique.values():
        if entry.year != year:
            reasons.append(
                f"record {entry.external_id} has year {entry.year!r}, expected {year}"
            )

    censuses = [dict(batch.census) for batch in batches if batch.census]
    expected_total = _consistent_integer(censuses, "expected_total", reasons)
    parser_raw = _consistent_integer(censuses, "parser_raw_records", reasons)
    parser_rejected = _consistent_integer(censuses, "parser_rejected_records", reasons)
    parser_excluded = _consistent_integer(censuses, "parser_excluded_records", reasons)
    if expected_total is None:
        reasons.append("primary source did not provide an expected_total census")
    elif expected_total != len(unique):
        reasons.append(f"expected_total={expected_total}, unique_records={len(unique)}")
    if parser_raw is None or parser_rejected is None or parser_excluded is None:
        reasons.append("primary parser did not provide raw/rejected/excluded record counts")
    elif parser_raw - parser_rejected - parser_excluded != expected_total:
        reasons.append(
            "parser census does not reconcile: "
            f"raw={parser_raw}, rejected={parser_rejected}, excluded={parser_excluded}, "
            f"expected={expected_total}"
        )
    if parser_rejected:
        reasons.append(f"primary parser rejected {parser_rejected} record(s)")
    if duplicates:
        reasons.append(f"duplicate official external IDs: {', '.join(duplicates)}")
    terminal = bool(batches) and batches[-1].next_cursor is None
    if not terminal:
        reasons.append("terminal cursor was not reached")
    response_hashes = tuple(
        value
        for batch in batches
        for value in (batch.raw_response_artifact_hash,)
        if value
    )
    if any(not batch.raw_response_artifact_hash for batch in batches):
        reasons.append("one or more response artifact hashes are missing")
    request_hashes = tuple(
        str(item["response_sha256"])
        for batch in batches
        for item in batch.request_audit
        if item.get("response_sha256")
    )
    field_coverage = {
        field: sum(_field_present(entry, field) for entry in unique.values())
        for field in _OUTPUT_FIELDS
    }
    records = tuple(
        _record_document(entry, venue, venue_id, year, batches)
        for entry in sorted(unique.values(), key=lambda item: (item.title.casefold(), item.external_id))
    )
    status = "complete" if not reasons else (
        "failed" if any(batch.status is EnvelopeStatus.FAILED for batch in batches) else "unproven"
    )
    return records, Stage1UnitReceipt(
        venue_id=venue_id,
        venue_name=str(venue["name"]),
        venue_type=str(venue["venue_type"]),
        provider=descriptor.provider,
        year=year,
        status=status,
        pages_fetched=len(batches),
        terminal_cursor_reached=terminal,
        returned_records=len(entries),
        unique_records=len(unique),
        expected_total=expected_total,
        parser_raw_records=parser_raw,
        parser_rejected_records=parser_rejected,
        parser_excluded_records=parser_excluded,
        duplicate_external_ids=duplicates,
        response_hashes=response_hashes,
        request_hashes=request_hashes,
        field_coverage=field_coverage,
        reasons=tuple(dict.fromkeys(reasons)),
    )


def _record_document(
    entry: SourceEntry,
    venue: Mapping[str, Any],
    venue_id: str,
    year: int,
    batches: Sequence[SourceBatch],
) -> dict[str, Any]:
    metadata = dict(entry.metadata)
    values: dict[str, Any] = {
        "schema_version": "1",
        "venue_id": venue_id,
        "venue_name": venue["name"],
        "venue_type": venue["venue_type"],
        "membership_status": "official_confirmed",
        "provider": entry.provider,
        "external_id": entry.external_id,
        "title": entry.title,
        "abstract": entry.abstract,
        "authors": list(entry.authors),
        "doi": entry.doi,
        "publication_date": entry.publication_date,
        "year": entry.year,
        "landing_url": entry.landing_url,
        "pdf_url": entry.pdf_url,
        "volume": metadata.get("volume"),
        "issue": metadata.get("issue"),
        "pages": metadata.get("pages") or metadata.get("page"),
        "keywords": metadata.get("keywords") or [],
        "publication_version": entry.publication_version.value,
        "license": entry.license,
        "host_type": entry.host_type,
        "access_basis": entry.access_basis.value,
        "field_status": {},
        "provenance": {
            "source_run_ids": sorted({batch.source_run_id for batch in batches}),
            "query_hashes": sorted({batch.query_hash for batch in batches}),
            "official_url": venue.get("official_url"),
        },
        "source_metadata": metadata,
    }
    for field in _OUTPUT_FIELDS:
        value = values.get(field)
        values["field_status"][field] = "present" if value not in (None, "", [], ()) else "unavailable_at_primary"
    if values["year"] is None:
        values["year"] = year
    return values


def _consistent_integer(
    censuses: Sequence[Mapping[str, Any]], key: str, reasons: list[str]
) -> int | None:
    values: set[int] = set()
    for census in censuses:
        value = census.get(key)
        if value is None:
            continue
        try:
            values.add(int(value))
        except (TypeError, ValueError):
            reasons.append(f"invalid census {key}={value!r}")
    if len(values) > 1:
        reasons.append(f"inconsistent census {key}: {sorted(values)}")
        return None
    return next(iter(values), None)


def _duplicates(values: Sequence[str] | Any) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return tuple(sorted(duplicates))


def _field_present(entry: SourceEntry, field: str) -> int:
    if field in {"volume", "issue", "pages", "keywords"}:
        key = "page" if field == "pages" and "pages" not in entry.metadata else field
        value = entry.metadata.get(key)
    else:
        value = getattr(entry, field)
    return int(value not in (None, "", [], ()))


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
    temporary.replace(path)
