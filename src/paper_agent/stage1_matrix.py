"""Evidence matrix for the standalone Stage 1 venue/year census.

The live collector produces one receipt per run and can be resumed or split
across many processes.  This module deliberately does not infer completeness
from filenames or from a successful process exit.  It enumerates every
requested venue/year cell, selects the strongest compatible receipt evidence,
and leaves cells without proof visible as ``missing_receipt``.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .canonical import canonical_json
from .manifests import ManifestCatalog, load_catalog
from .stage1 import STAGE1_INTERFACE_VERSION


STAGE1_MATRIX_SCHEMA_VERSION = "stage1-field-matrix-v1"
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
_STATUS_RANK = {
    "complete": 4,
    "not_applicable": 4,
    "unproven": 3,
    "failed": 2,
    "incomplete": 1,
}
_PASSING_STATUSES = frozenset({"complete", "not_applicable"})


class Stage1MatrixError(ValueError):
    """The receipt evidence cannot be interpreted as a Stage 1 matrix."""


def build_stage1_field_matrix(
    receipt_paths: Sequence[Path] = (),
    *,
    receipts_root: Path | None = None,
    venue_ids: Sequence[str] | None = None,
    year_from: int = 2016,
    year_to: int = 2025,
    catalog: ManifestCatalog | None = None,
) -> dict[str, Any]:
    """Build a deterministic, fail-closed matrix from persisted receipts.

    ``receipt_paths`` may contain multi-year or multi-venue receipts.  A
    ``receipts_root`` recursively discovers ``*.receipt.json`` files, which
    makes the command useful for a resumable batch.  Duplicate evidence is
    retained in each cell; if equally strong receipts disagree, the cell is
    marked ``conflict`` instead of silently choosing one.
    """

    _validate_year_range(year_from, year_to)
    resolved_catalog = catalog or load_catalog()
    selected_venues = _normalize_venues(venue_ids, resolved_catalog)
    paths = _resolve_receipt_paths(receipt_paths, receipts_root)
    documents, input_errors = _load_receipts(paths)

    candidates: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for source in documents:
        document = source["document"]
        units = document.get("units")
        if not isinstance(units, list):
            input_errors.append(
                f"{source['path']}: receipt units must be a list"
            )
            continue
        for unit in units:
            if not isinstance(unit, Mapping):
                input_errors.append(
                    f"{source['path']}: receipt contains a non-object unit"
                )
                continue
            try:
                key = (str(unit["venue_id"]), int(unit["year"]))
            except (KeyError, TypeError, ValueError) as error:
                input_errors.append(
                    f"{source['path']}: invalid unit key ({type(error).__name__})"
                )
                continue
            if key[0] not in selected_venues or not year_from <= key[1] <= year_to:
                continue
            candidates.setdefault(key, []).append(
                {
                    "path": source["path"],
                    "sha256": source["sha256"],
                    "receipt_status": str(document.get("status", "unknown")),
                    "run_id": document.get("run_id"),
                    "unit": dict(unit),
                }
            )

    rows: list[dict[str, Any]] = []
    for venue_id in selected_venues:
        venue = resolved_catalog.venue(venue_id)
        for year in range(year_from, year_to + 1):
            key = (venue_id, year)
            source_candidates = candidates.get(key, [])
            applicable, applicability_reason = _venue_year_applicability(venue, year)
            if not applicable and source_candidates and all(
                str(candidate["unit"].get("status")) == "not_applicable"
                for candidate in source_candidates
            ):
                rows.append(
                    _not_applicable_with_sources(
                        venue_id,
                        venue,
                        year,
                        applicability_reason,
                        source_candidates,
                    )
                )
                continue
            if not source_candidates:
                if not applicable:
                    row = _empty_row(
                        venue_id,
                        venue,
                        year,
                        status="not_applicable",
                        reasons=(applicability_reason,),
                    )
                else:
                    row = _empty_row(
                        venue_id,
                        venue,
                        year,
                        status="missing_receipt",
                        reasons=(
                            "no persisted Stage 1 receipt covers this venue/year",
                        ),
                    )
                rows.append(row)
                continue
            rows.append(_select_cell(venue_id, venue, year, source_candidates))

    status_counts: dict[str, int] = {}
    field_coverage: dict[str, int] = {field: 0 for field in _OUTPUT_FIELDS}
    total_records = 0
    for row in rows:
        status = str(row["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
        if status in _PASSING_STATUSES:
            total_records += int(row["unique_records"] or 0)
            for field in _OUTPUT_FIELDS:
                field_coverage[field] += int(row["field_coverage"].get(field, 0) or 0)

    base: dict[str, Any] = {
        "schema_version": STAGE1_MATRIX_SCHEMA_VERSION,
        "interface_version": STAGE1_INTERFACE_VERSION,
        "request": {
            "venue_ids": list(selected_venues),
            "year_from": year_from,
            "year_to": year_to,
        },
        "status": (
            "complete"
            if not input_errors and all(row["status"] in _PASSING_STATUSES for row in rows)
            else "incomplete"
        ),
        "summary": {
            "total_cells": len(rows),
            "eligible_cells": sum(row["status"] != "not_applicable" for row in rows),
            "status_counts": dict(sorted(status_counts.items())),
            "total_records": total_records,
            "field_coverage": field_coverage,
            "input_error_count": len(input_errors),
            "all_cells_proven": not input_errors
            and all(row["status"] in _PASSING_STATUSES for row in rows),
        },
        "inputs": [
            {"path": item["path"], "sha256": item["sha256"]}
            for item in documents
        ],
        "input_errors": sorted(dict.fromkeys(input_errors)),
        "cells": rows,
    }
    base["matrix_sha256"] = sha256(canonical_json(base)).hexdigest()
    return base


def write_stage1_field_matrix(
    matrix: Mapping[str, Any],
    *,
    output_path: Path,
    markdown_path: Path | None = None,
) -> None:
    """Write JSON and optional human-readable Markdown atomically enough for CLI use."""

    _write_bytes(output_path, canonical_json(dict(matrix)) + b"\n")
    if markdown_path is not None:
        _write_bytes(
            markdown_path,
            render_stage1_field_matrix_markdown(matrix).encode("utf-8"),
        )


def render_stage1_field_matrix_markdown(matrix: Mapping[str, Any]) -> str:
    """Render the matrix without dropping failed or missing cells."""

    request = matrix.get("request", {})
    summary = matrix.get("summary", {})
    lines = [
        "# Stage 1 field matrix",
        "",
        f"- Status: **{matrix.get('status', 'unknown')}**",
        f"- Scope: {', '.join(str(value) for value in request.get('venue_ids', []))} "
        f"× {request.get('year_from')}–{request.get('year_to')}",
        f"- Cells: {summary.get('total_cells', 0)}; "
        f"eligible: {summary.get('eligible_cells', 0)}; "
        f"records in proven cells: {summary.get('total_records', 0)}",
        "",
        "| Venue | Year | Status | Records | Abstract | DOI | PDF | Receipt evidence | Reasons |",
        "|---|---:|---|---:|---:|---:|---:|---|---|",
    ]
    for row in matrix.get("cells", []):
        coverage = row.get("field_coverage", {})
        sources = row.get("receipt_sources", [])
        receipt = ", ".join(Path(str(item.get("path", ""))).name for item in sources)
        reasons = "; ".join(str(reason) for reason in row.get("reasons", []))
        lines.append(
            "| "
            + " | ".join(
                (
                    _md(row.get("venue_id")),
                    str(row.get("year", "")),
                    _md(row.get("status")),
                    str(row.get("unique_records", 0)),
                    str(coverage.get("abstract", 0)),
                    str(coverage.get("doi", 0)),
                    str(coverage.get("pdf_url", 0)),
                    _md(receipt),
                    _md(reasons),
                )
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _normalize_venues(
    venue_ids: Sequence[str] | None,
    catalog: ManifestCatalog,
) -> tuple[str, ...]:
    values = tuple(sorted(set(venue_ids or catalog.venues)))
    unknown = sorted(set(values) - set(catalog.venues))
    if unknown:
        raise Stage1MatrixError(f"unknown venue_id(s): {', '.join(unknown)}")
    if not values:
        raise Stage1MatrixError("at least one venue_id is required")
    return values


def _validate_year_range(year_from: int, year_to: int) -> None:
    if year_from < 1900 or year_to > 2200 or year_from > year_to:
        raise Stage1MatrixError("year range is invalid")


def _resolve_receipt_paths(
    receipt_paths: Sequence[Path], receipts_root: Path | None
) -> tuple[Path, ...]:
    resolved: set[Path] = set()
    for path in receipt_paths:
        candidate = Path(path)
        if candidate.is_dir():
            raise Stage1MatrixError(f"receipt path is a directory: {candidate}")
        resolved.add(candidate)
    if receipts_root is not None:
        root = Path(receipts_root)
        if not root.exists():
            raise Stage1MatrixError(f"receipt root does not exist: {root}")
        resolved.update(path for path in root.rglob("*.receipt.json") if path.is_file())
    return tuple(sorted(resolved, key=lambda path: str(path)))


def _load_receipts(paths: Iterable[Path]) -> tuple[list[dict[str, Any]], list[str]]:
    documents: list[dict[str, Any]] = []
    errors: list[str] = []
    for path in paths:
        try:
            payload = path.read_bytes()
            document = json.loads(payload)
            if not isinstance(document, Mapping):
                raise ValueError("receipt root must be an object")
            documents.append(
                {
                    "path": str(path),
                    "sha256": sha256(payload).hexdigest(),
                    "document": dict(document),
                }
            )
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as error:
            errors.append(f"{path}: {type(error).__name__}: {error}")
    return documents, errors


def _select_cell(
    venue_id: str,
    venue: Mapping[str, Any],
    year: int,
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    ranked = sorted(
        candidates,
        key=lambda candidate: (
            -_STATUS_RANK.get(str(candidate["unit"].get("status")), 0),
            str(candidate["path"]),
        ),
    )
    best_rank = _STATUS_RANK.get(str(ranked[0]["unit"].get("status")), 0)
    strongest = [
        candidate
        for candidate in ranked
        if _STATUS_RANK.get(str(candidate["unit"].get("status")), 0) == best_rank
    ]
    fingerprints = {_unit_fingerprint(candidate["unit"]) for candidate in strongest}
    conflict = len(fingerprints) > 1
    selected = dict(ranked[0]["unit"])
    status = "conflict" if conflict else str(selected.get("status", "unknown"))
    reasons = [str(value) for value in selected.get("reasons", [])]
    if conflict:
        reasons.append(
            "equally strong receipts disagree; inspect receipt_sources before publishing"
        )
    fields = _field_coverage(selected)
    return {
        "venue_id": venue_id,
        "venue_name": selected.get("venue_name", venue.get("name")),
        "venue_type": selected.get("venue_type", venue.get("venue_type")),
        "year": year,
        "status": status,
        "unique_records": _int_or_none(selected.get("unique_records")),
        "returned_records": _int_or_none(selected.get("returned_records")),
        "expected_total": _int_or_none(selected.get("expected_total")),
        "pages_fetched": _int_or_none(selected.get("pages_fetched")),
        "terminal_cursor_reached": bool(selected.get("terminal_cursor_reached", False)),
        "parser_raw_records": _int_or_none(selected.get("parser_raw_records")),
        "parser_rejected_records": _int_or_none(selected.get("parser_rejected_records")),
        "parser_excluded_records": _int_or_none(selected.get("parser_excluded_records")),
        "field_coverage": fields,
        "receipt_sources": [
            {
                "path": str(candidate["path"]),
                "sha256": str(candidate["sha256"]),
                "receipt_status": str(candidate["receipt_status"]),
                "run_id": candidate.get("run_id"),
                "selected": candidate is ranked[0],
            }
            for candidate in ranked
        ],
        "reasons": list(dict.fromkeys(reasons)),
    }


def _empty_row(
    venue_id: str,
    venue: Mapping[str, Any],
    year: int,
    *,
    status: str,
    reasons: Sequence[str],
) -> dict[str, Any]:
    return {
        "venue_id": venue_id,
        "venue_name": venue.get("name"),
        "venue_type": venue.get("venue_type"),
        "year": year,
        "status": status,
        "unique_records": 0,
        "returned_records": 0,
        "expected_total": 0 if status == "not_applicable" else None,
        "pages_fetched": 0,
        "terminal_cursor_reached": status == "not_applicable",
        "parser_raw_records": 0 if status == "not_applicable" else None,
        "parser_rejected_records": 0 if status == "not_applicable" else None,
        "parser_excluded_records": 0 if status == "not_applicable" else None,
        "field_coverage": {field: 0 for field in _OUTPUT_FIELDS},
        "receipt_sources": [],
        "reasons": list(reasons),
    }


def _not_applicable_with_sources(
    venue_id: str,
    venue: Mapping[str, Any],
    year: int,
    reason: str,
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    row = _empty_row(
        venue_id,
        venue,
        year,
        status="not_applicable",
        reasons=(reason,),
    )
    row["receipt_sources"] = [
        {
            "path": str(candidate["path"]),
            "sha256": str(candidate["sha256"]),
            "receipt_status": str(candidate["receipt_status"]),
            "run_id": candidate.get("run_id"),
            "selected": True,
        }
        for candidate in sorted(candidates, key=lambda item: str(item["path"]))
    ]
    return row


def _venue_year_applicability(venue: Mapping[str, Any], year: int) -> tuple[bool, str]:
    parameters = venue.get("provider_params", {})
    held_years = parameters.get("held_years", []) if isinstance(parameters, Mapping) else []
    if held_years and year not in {int(value) for value in held_years}:
        return False, f"venue was not held in {year}"
    date_range = venue.get("date_range")
    if isinstance(date_range, Mapping):
        start_year = int(str(date_range["start"])[:4])
        end_year = int(str(date_range["end"])[:4])
        if year < start_year or year > end_year:
            return False, f"venue date_range is {date_range['start']} through {date_range['end']}"
    return True, ""


def _field_coverage(unit: Mapping[str, Any]) -> dict[str, int]:
    value = unit.get("field_coverage")
    coverage = value if isinstance(value, Mapping) else {}
    return {field: _int_or_zero(coverage.get(field)) for field in _OUTPUT_FIELDS}


def _unit_fingerprint(unit: Mapping[str, Any]) -> bytes:
    selected = {
        key: unit.get(key)
        for key in (
            "status",
            "pages_fetched",
            "terminal_cursor_reached",
            "returned_records",
            "unique_records",
            "expected_total",
            "parser_raw_records",
            "parser_rejected_records",
            "parser_excluded_records",
            "duplicate_external_ids",
            "response_hashes",
            "request_hashes",
            "field_coverage",
            "reasons",
        )
    }
    return canonical_json(selected)


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _int_or_zero(value: Any) -> int:
    return _int_or_none(value) or 0


def _md(value: Any) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ")


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
