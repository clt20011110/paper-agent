from __future__ import annotations

import json
from pathlib import Path

import pytest

from paper_agent.domain import EnvelopeStatus, SourceBatch, SourceEntry
from paper_agent.manifests import ManifestCatalog
from paper_agent.providers.api import VenueDescriptor
from paper_agent.stage1 import (
    Stage1IncompleteError,
    Stage1Request,
    collect_stage1_metadata,
    venue_catalog_document,
    write_stage1_result,
)


def _catalog() -> ManifestCatalog:
    venue = {
        "schema_version": "1",
        "venue_id": "example",
        "name": "Example Conference",
        "venue_type": "conference",
        "primary_provider": "example_provider",
        "provider_params": {},
        "official_url": "https://example.test/",
    }
    return ManifestCatalog(
        providers={"example_provider": {}},
        venues={"example": venue},
        acceptances={"example": {}},
    )


class _Adapter:
    def __init__(self, *, census: bool = True, cycle: bool = False) -> None:
        self.census = census
        self.cycle = cycle

    def discover(self, descriptor, window, cursor=None):
        offset = int(cursor or 0)
        year = int(window.year)
        entry = SourceEntry(
            provider=descriptor.provider,
            external_id=f"{year}-{offset}",
            title=f"Paper {year}-{offset}",
            authors=("A. Author",),
            abstract="Abstract",
            doi=f"10.1234/{year}.{offset}",
            publication_date=f"{year}-06-01",
            year=year,
            venue_name="Example Conference",
            landing_url=f"https://example.test/{year}/{offset}",
            metadata={"volume": str(year - 2000), "pages": f"{offset + 1}-2"},
        )
        next_cursor = "1" if offset == 0 else ("1" if self.cycle else None)
        return SourceBatch(
            source_run_id=f"example:{year}",
            query_hash=f"query:{year}",
            entries=(entry,),
            next_cursor=next_cursor,
            status=EnvelopeStatus.SUCCESS,
            raw_response_artifact_hash=f"{'a' if offset == 0 else 'b'}" * 64,
            request_audit=({"response_sha256": f"{'a' if offset == 0 else 'b'}" * 64},),
            census=(
                {
                    "expected_total": 2,
                    "parser_raw_records": 2,
                    "parser_rejected_records": 0,
                    "parser_excluded_records": 0,
                }
                if self.census
                else {}
            ),
        )


def test_collect_expands_venue_year_and_proves_each_census() -> None:
    result = collect_stage1_metadata(
        Stage1Request(("example",), 2023, 2024, page_size=1, max_workers=2),
        catalog=_catalog(),
        adapter_factory=lambda _: _Adapter(),
    )

    assert result.complete
    assert len(result.records) == 4
    assert [(unit.year, unit.pages_fetched, unit.expected_total) for unit in result.receipt.units] == [
        (2023, 2, 2),
        (2024, 2, 2),
    ]
    assert all(unit.field_coverage["abstract"] == 2 for unit in result.receipt.units)
    assert result.records[0]["membership_status"] == "official_confirmed"
    assert result.records[0]["volume"] == "23"


def test_strict_publication_writes_receipt_but_not_unproven_metadata(tmp_path: Path) -> None:
    result = collect_stage1_metadata(
        Stage1Request(("example",), 2024, 2024),
        catalog=_catalog(),
        adapter_factory=lambda _: _Adapter(census=False),
    )
    output = tmp_path / "papers.jsonl"

    with pytest.raises(Stage1IncompleteError) as raised:
        write_stage1_result(result, output_path=output)

    assert not output.exists()
    receipt = output.with_suffix(".jsonl.receipt.json")
    assert receipt.exists()
    document = json.loads(receipt.read_text())
    assert document["status"] == "incomplete"
    assert "expected_total" in " ".join(document["units"][0]["reasons"])
    assert raised.value.result.receipt_path == receipt


def test_allow_incomplete_explicitly_publishes_records(tmp_path: Path) -> None:
    result = collect_stage1_metadata(
        Stage1Request(("example",), 2024, 2024, strict=False),
        catalog=_catalog(),
        adapter_factory=lambda _: _Adapter(census=False),
    )
    output = tmp_path / "papers.jsonl"

    published = write_stage1_result(result, output_path=output, allow_incomplete=True)

    assert published.status == "incomplete"
    assert len(output.read_text().splitlines()) == 2


def test_cursor_cycle_is_fail_closed() -> None:
    result = collect_stage1_metadata(
        Stage1Request(("example",), 2024, 2024),
        catalog=_catalog(),
        adapter_factory=lambda _: _Adapter(cycle=True),
    )

    assert not result.complete
    assert result.receipt.units[0].status == "unproven"
    assert any("cursor cycle" in reason for reason in result.receipt.units[0].reasons)


def test_request_normalizes_duplicate_venues_and_catalog_is_sorted() -> None:
    request = Stage1Request(("example", "example"), 2024, 2024)
    assert request.venue_ids == ("example",)
    assert venue_catalog_document(_catalog())[0]["venue_id"] == "example"


def test_unknown_venue_is_rejected_before_work_starts() -> None:
    with pytest.raises(ValueError, match="unknown venue_id"):
        collect_stage1_metadata(
            Stage1Request(("missing",), 2024, 2024),
            catalog=_catalog(),
            adapter_factory=lambda _: _Adapter(),
        )


def test_year_before_venue_launch_is_not_applicable_and_does_not_call_adapter() -> None:
    catalog = _catalog()
    catalog.venues["example"]["date_range"] = {
        "start": "2024-01-01",
        "end": "2025-12-31",
    }

    result = collect_stage1_metadata(
        Stage1Request(("example",), 2023, 2024),
        catalog=catalog,
        adapter_factory=lambda _: _Adapter(),
    )

    assert result.complete
    assert [unit.status for unit in result.receipt.units] == [
        "not_applicable",
        "complete",
    ]
    assert len(result.records) == 2
