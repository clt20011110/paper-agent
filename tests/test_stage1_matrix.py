from __future__ import annotations

import json
from pathlib import Path

from paper_agent.canonical import canonical_json
from paper_agent.cli import build_parser
from paper_agent.manifests import ManifestCatalog
from paper_agent.stage1_matrix import (
    build_stage1_field_matrix,
    render_stage1_field_matrix_markdown,
    write_stage1_field_matrix,
)


def _catalog() -> ManifestCatalog:
    return ManifestCatalog(
        providers={"provider": {}},
        venues={
            "alpha": {
                "schema_version": "1",
                "venue_id": "alpha",
                "name": "Alpha",
                "venue_type": "journal",
                "primary_provider": "provider",
                "provider_params": {},
                "date_range": {"start": "2020-01-01", "end": "2025-12-31"},
            },
            "periodic": {
                "schema_version": "1",
                "venue_id": "periodic",
                "name": "Periodic",
                "venue_type": "conference",
                "primary_provider": "provider",
                "provider_params": {"held_years": [2024]},
            },
        },
        acceptances={"alpha": {}, "periodic": {}},
    )


def _unit(venue_id: str, year: int, *, status: str = "complete", count: int = 2):
    return {
        "venue_id": venue_id,
        "venue_name": venue_id.title(),
        "venue_type": "journal",
        "provider": "provider",
        "year": year,
        "status": status,
        "pages_fetched": 1,
        "terminal_cursor_reached": True,
        "returned_records": count,
        "unique_records": count,
        "expected_total": count,
        "parser_raw_records": count,
        "parser_rejected_records": 0,
        "parser_excluded_records": 0,
        "field_coverage": {
            "title": count,
            "abstract": count,
            "authors": count,
            "doi": count,
            "publication_date": count,
            "year": count,
            "venue_name": count,
            "landing_url": count,
            "pdf_url": count,
            "volume": 0,
            "issue": 0,
            "pages": 0,
            "keywords": 0,
        },
        "response_hashes": ["a" * 64],
        "request_hashes": ["b" * 64],
        "reasons": [],
    }


def _receipt(path: Path, units: list[dict]) -> None:
    path.write_bytes(
        canonical_json(
            {
                "schema_version": "1",
                "interface_version": "stage1-standalone-v1",
                "run_id": path.stem,
                "request": {},
                "status": "complete"
                if all(unit["status"] == "complete" for unit in units)
                else "incomplete",
                "units": units,
                "metadata_sha256": "c" * 64,
            }
        )
        + b"\n"
    )


def test_matrix_enumerates_missing_and_not_applicable_cells(tmp_path: Path) -> None:
    receipt = tmp_path / "run.receipt.json"
    _receipt(receipt, [_unit("alpha", 2024)])

    matrix = build_stage1_field_matrix(
        [receipt],
        venue_ids=("alpha", "periodic"),
        year_from=2023,
        year_to=2025,
        catalog=_catalog(),
    )

    assert matrix["status"] == "incomplete"
    assert matrix["summary"]["status_counts"] == {
        "complete": 1,
        "missing_receipt": 3,
        "not_applicable": 2,
    }
    rows = {(row["venue_id"], row["year"]): row for row in matrix["cells"]}
    assert rows[("alpha", 2024)]["field_coverage"]["abstract"] == 2
    assert rows[("alpha", 2023)]["status"] == "missing_receipt"
    assert rows[("periodic", 2023)]["status"] == "not_applicable"


def test_equally_strong_disagreement_is_conflict(tmp_path: Path) -> None:
    first = tmp_path / "first.receipt.json"
    second = tmp_path / "second.receipt.json"
    _receipt(first, [_unit("alpha", 2024, count=2)])
    _receipt(second, [_unit("alpha", 2024, count=3)])

    matrix = build_stage1_field_matrix(
        [first, second],
        venue_ids=("alpha",),
        year_from=2024,
        year_to=2024,
        catalog=_catalog(),
    )

    row = matrix["cells"][0]
    assert matrix["status"] == "incomplete"
    assert row["status"] == "conflict"
    assert len(row["receipt_sources"]) == 2
    assert "equally strong receipts disagree" in row["reasons"][-1]


def test_matrix_writes_machine_and_markdown_outputs(tmp_path: Path) -> None:
    receipt = tmp_path / "run.receipt.json"
    _receipt(receipt, [_unit("alpha", 2024)])
    matrix = build_stage1_field_matrix(
        receipts_root=tmp_path,
        venue_ids=("alpha",),
        year_from=2024,
        year_to=2024,
        catalog=_catalog(),
    )
    output = tmp_path / "matrix.json"
    markdown = tmp_path / "matrix.md"
    write_stage1_field_matrix(matrix, output_path=output, markdown_path=markdown)

    assert json.loads(output.read_text())["matrix_sha256"] == matrix["matrix_sha256"]
    assert "| alpha | 2024 | complete |" in markdown.read_text()
    assert render_stage1_field_matrix_markdown(matrix).endswith("\n")


def test_stage1_matrix_cli_parser_accepts_receipt_root() -> None:
    args = build_parser().parse_args(
        [
            "stage1",
            "matrix",
            "--venue",
            "alpha",
            "--year-from",
            "2020",
            "--year-to",
            "2025",
            "--receipts-root",
            "/tmp/receipts",
            "--output",
            "/tmp/matrix.json",
        ]
    )
    assert args.stage1_command == "matrix"
    assert args.year_from == 2020
    assert args.receipts_root == Path("/tmp/receipts")
