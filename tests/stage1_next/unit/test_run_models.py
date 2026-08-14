"""Offline unit tests for Stage 1 run and collection statistics models."""

from dataclasses import FrozenInstanceError

import pytest

from paper_agent_next.errors import ContractError
from paper_agent_next.models import (
    Pagination,
    RunCounts,
    RunRecord,
    RunStatus,
    SourceTotal,
    SourceTotalScope,
    VenueType,
)


def _counts(**overrides) -> RunCounts:
    values = {
        "raw_items": 3,
        "included_papers": 3,
        "complete_papers": 3,
        "incomplete_papers": 0,
        "excluded_non_papers": 0,
        "duplicate_occurrences": 0,
        "parse_rejects": 0,
        "issue_records": 0,
    }
    values.update(overrides)
    return RunCounts(**values)


def _run(
    *,
    status=RunStatus.COMPLETE,
    source_name="proceedings",
    membership_complete=True,
    metadata_complete=True,
    complete=True,
    counts=None,
    pagination=Pagination(1, True, SourceTotal(3, SourceTotalScope.RAW_ITEMS)),
    warnings=("one warning",),
    errors=(),
    **overrides,
) -> RunRecord:
    values = {
        "status": status,
        "venue_id": "example-conf",
        "venue_name": "Example Conference",
        "venue_type": VenueType.CONFERENCE,
        "year": 2024,
        "source_name": source_name,
        "membership_complete": membership_complete,
        "metadata_complete": metadata_complete,
        "complete": complete,
        "counts": _counts() if counts is None else counts,
        "pagination": pagination,
        "warnings": warnings,
        "errors": errors,
    }
    values.update(overrides)
    return RunRecord(**values)


def _partial_counts() -> RunCounts:
    return _counts(
        raw_items=4,
        included_papers=3,
        complete_papers=2,
        incomplete_papers=1,
        excluded_non_papers=1,
        issue_records=1,
    )


def test_new_enums_have_exact_values() -> None:
    assert list(RunStatus) == [
        RunStatus.COMPLETE,
        RunStatus.PARTIAL,
        RunStatus.FAILED,
        RunStatus.NOT_APPLICABLE,
    ]
    assert [item.value for item in RunStatus] == ["complete", "partial", "failed", "not_applicable"]
    assert list(SourceTotalScope) == [SourceTotalScope.RAW_ITEMS, SourceTotalScope.INCLUDED_PAPERS]
    assert [item.value for item in SourceTotalScope] == ["raw_items", "included_papers"]


def test_run_counts_accept_zero_and_nonzero_equation_consistent_values() -> None:
    zero = _counts(
        raw_items=0,
        included_papers=0,
        complete_papers=0,
    )
    nonzero = _partial_counts()

    assert zero.to_dict()["raw_items"] == 0
    assert nonzero.raw_items == 4
    assert nonzero.included_papers == nonzero.complete_papers + nonzero.incomplete_papers
    assert nonzero.raw_items == (
        nonzero.included_papers
        + nonzero.excluded_non_papers
        + nonzero.duplicate_occurrences
        + nonzero.parse_rejects
    )


def test_run_counts_serializes_exact_keys_and_is_frozen_slotted() -> None:
    counts = _partial_counts()

    assert list(counts.to_dict()) == [
        "raw_items",
        "included_papers",
        "complete_papers",
        "incomplete_papers",
        "excluded_non_papers",
        "duplicate_occurrences",
        "parse_rejects",
        "issue_records",
    ]
    assert not hasattr(counts, "__dict__")
    with pytest.raises(FrozenInstanceError):
        counts.raw_items = 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"raw_items": -1},
        {"raw_items": True},
        {"included_papers": 1.0},
        {"complete_papers": "3"},
        {"incomplete_papers": -1},
        {"excluded_non_papers": True},
        {"duplicate_occurrences": -1},
        {"parse_rejects": False},
        {"issue_records": "0"},
        {"raw_items": 2},
        {"included_papers": 2},
    ],
)
def test_run_counts_rejects_invalid_types_values_and_equations(kwargs) -> None:
    with pytest.raises(ContractError):
        _counts(**kwargs)


@pytest.mark.parametrize("scope", [SourceTotalScope.RAW_ITEMS, SourceTotalScope.INCLUDED_PAPERS])
def test_source_total_serializes_scope_value(scope) -> None:
    total = SourceTotal(3, scope)

    assert list(total.to_dict()) == ["value", "scope"]
    assert total.to_dict() == {"value": 3, "scope": scope.value}
    assert not hasattr(total, "__dict__")
    with pytest.raises(FrozenInstanceError):
        total.value = 4


@pytest.mark.parametrize("value", [-1, True, 1.0, "3"])
def test_source_total_rejects_invalid_value(value) -> None:
    with pytest.raises(ContractError, match="value"):
        SourceTotal(value, SourceTotalScope.RAW_ITEMS)


def test_source_total_rejects_plain_scope_string() -> None:
    with pytest.raises(ContractError, match="scope"):
        SourceTotal(3, "raw_items")


def test_pagination_supports_null_and_nested_source_total() -> None:
    empty_total = Pagination(0, False, None)
    nested_total = Pagination(2, True, SourceTotal(3, SourceTotalScope.INCLUDED_PAPERS))

    assert empty_total.to_dict() == {
        "pages_fetched": 0,
        "terminal_reached": False,
        "source_total": None,
    }
    assert list(nested_total.to_dict()) == ["pages_fetched", "terminal_reached", "source_total"]
    assert nested_total.to_dict()["source_total"] == {"value": 3, "scope": "included_papers"}
    assert not hasattr(nested_total, "__dict__")
    with pytest.raises(FrozenInstanceError):
        nested_total.pages_fetched = 3


@pytest.mark.parametrize(
    "kwargs",
    [
        {"pages_fetched": -1},
        {"pages_fetched": True},
        {"pages_fetched": "2"},
        {"terminal_reached": 1},
        {"terminal_reached": "true"},
        {"source_total": {}},
    ],
)
def test_pagination_rejects_invalid_values(kwargs) -> None:
    values = {"pages_fetched": 1, "terminal_reached": True, "source_total": None}
    values.update(kwargs)
    with pytest.raises(ContractError):
        Pagination(**values)


def test_run_record_valid_complete() -> None:
    record = _run()

    assert record.status is RunStatus.COMPLETE
    assert record.complete is True
    assert record.membership_complete is True
    assert record.metadata_complete is True
    assert record.warnings == ("one warning",)
    assert record.errors == ()


def test_run_record_valid_partial() -> None:
    record = _run(
        status=RunStatus.PARTIAL,
        membership_complete=True,
        metadata_complete=False,
        complete=False,
        counts=_partial_counts(),
        pagination=Pagination(2, True, SourceTotal(4, SourceTotalScope.RAW_ITEMS)),
        warnings=(),
        errors=(),
    )

    assert record.status is RunStatus.PARTIAL
    assert record.complete is False
    assert record.counts.incomplete_papers == 1


def test_run_record_valid_failed() -> None:
    record = _run(
        status=RunStatus.FAILED,
        source_name=None,
        membership_complete=False,
        metadata_complete=False,
        complete=False,
        counts=_counts(raw_items=0, included_papers=0, complete_papers=0),
        pagination=None,
        warnings=(),
        errors=("request_failed",),
    )

    assert record.status is RunStatus.FAILED
    assert record.pagination is None
    assert record.errors == ("request_failed",)


def test_run_record_valid_not_applicable() -> None:
    record = _run(
        status=RunStatus.NOT_APPLICABLE,
        source_name=None,
        membership_complete=False,
        metadata_complete=False,
        complete=False,
        counts=_counts(raw_items=0, included_papers=0, complete_papers=0),
        pagination=None,
        warnings=("year not held",),
        errors=(),
    )

    assert record.status is RunStatus.NOT_APPLICABLE
    assert record.complete is False
    assert record.warnings == ("year not held",)


def test_run_record_serializes_exact_nested_schema() -> None:
    record = _run()
    serialized = record.to_dict()

    assert list(serialized) == [
        "schema_version",
        "status",
        "venue_id",
        "venue_name",
        "venue_type",
        "year",
        "source_name",
        "membership_complete",
        "metadata_complete",
        "complete",
        "counts",
        "pagination",
        "warnings",
        "errors",
    ]
    assert serialized["schema_version"] == 1
    assert serialized["status"] == "complete"
    assert serialized["venue_type"] == "conference"
    assert list(serialized["counts"]) == [
        "raw_items",
        "included_papers",
        "complete_papers",
        "incomplete_papers",
        "excluded_non_papers",
        "duplicate_occurrences",
        "parse_rejects",
        "issue_records",
    ]
    assert list(serialized["pagination"]) == ["pages_fetched", "terminal_reached", "source_total"]
    assert serialized["pagination"]["source_total"] == {"value": 3, "scope": "raw_items"}
    assert serialized["warnings"] == ["one warning"]
    assert serialized["errors"] == []
    assert not hasattr(record, "__dict__")
    with pytest.raises(FrozenInstanceError):
        record.status = RunStatus.FAILED


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "complete"),
        ("venue_type", "conference"),
        ("membership_complete", 1),
        ("metadata_complete", 0),
        ("complete", 1),
        ("counts", {}),
        ("pagination", {}),
        ("warnings", ["warning"]),
        ("errors", ["error"]),
    ],
)
def test_run_record_rejects_plain_enums_wrong_types_and_lists(field, value) -> None:
    with pytest.raises(ContractError, match=field):
        _run(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("warnings", ("",)),
        ("warnings", (" warning",)),
        ("warnings", ("warning ",)),
        ("errors", ("",)),
        ("errors", (" error",)),
        ("errors", ("error ",)),
    ],
)
def test_run_record_rejects_empty_or_surrounding_warning_error_text(field, value) -> None:
    with pytest.raises(ContractError, match=field):
        _run(**{field: value})


def test_run_record_rejects_complete_not_equal_to_completeness_and() -> None:
    with pytest.raises(ContractError, match="complete"):
        _run(
            status=RunStatus.PARTIAL,
            membership_complete=True,
            metadata_complete=False,
            complete=True,
            errors=("metadata incomplete",),
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"counts": _partial_counts(), "metadata_complete": True, "complete": False, "status": RunStatus.PARTIAL},
        {
            "counts": _counts(raw_items=4, parse_rejects=1),
            "membership_complete": True,
            "metadata_complete": False,
            "complete": False,
            "status": RunStatus.PARTIAL,
            "pagination": Pagination(1, True, SourceTotal(4, SourceTotalScope.RAW_ITEMS)),
            "errors": ("parse rejected",),
        },
        {"counts": _counts(issue_records=1), "status": RunStatus.COMPLETE},
        {"status": RunStatus.COMPLETE, "errors": ("unexpected error",)},
    ],
)
def test_run_record_rejects_completeness_count_effects(kwargs) -> None:
    with pytest.raises(ContractError):
        _run(**kwargs)


def test_run_record_rejects_complete_status_with_incomplete_paper() -> None:
    with pytest.raises(ContractError):
        _run(
            status=RunStatus.COMPLETE,
            counts=_partial_counts(),
            pagination=Pagination(2, True, SourceTotal(4, SourceTotalScope.RAW_ITEMS)),
        )


def test_run_record_rejects_partial_with_only_warning() -> None:
    with pytest.raises(ContractError, match="partial"):
        _run(
            status=RunStatus.PARTIAL,
            membership_complete=False,
            metadata_complete=False,
            complete=False,
            counts=_counts(raw_items=0, included_papers=0, complete_papers=0),
            pagination=None,
            warnings=("warning",),
            errors=(),
        )


def test_run_record_rejects_failed_without_error() -> None:
    with pytest.raises(ContractError, match="errors"):
        _run(
            status=RunStatus.FAILED,
            membership_complete=False,
            metadata_complete=False,
            complete=False,
            counts=_counts(raw_items=0, included_papers=0, complete_papers=0),
            pagination=None,
            errors=(),
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "counts": _counts(raw_items=1, included_papers=1, complete_papers=1),
        },
        {
            "pagination": Pagination(1, True, None),
        },
        {
            "errors": ("not applicable failed",),
        },
    ],
)
def test_run_record_rejects_not_applicable_incompatible_fields(kwargs) -> None:
    with pytest.raises(ContractError):
        values = {
            "status": RunStatus.NOT_APPLICABLE,
            "source_name": None,
            "membership_complete": False,
            "metadata_complete": False,
            "complete": False,
            "counts": _counts(raw_items=0, included_papers=0, complete_papers=0),
            "pagination": None,
            "warnings": (),
            "errors": (),
        }
        values.update(kwargs)
        _run(
            **values,
        )


def test_membership_complete_requires_terminal_pagination_and_source() -> None:
    with pytest.raises(ContractError, match="pagination"):
        _run(
            status=RunStatus.PARTIAL,
            membership_complete=True,
            metadata_complete=False,
            complete=False,
            counts=_partial_counts(),
            pagination=None,
            errors=("incomplete pagination",),
        )
    with pytest.raises(ContractError, match="terminal"):
        _run(
            status=RunStatus.PARTIAL,
            membership_complete=True,
            metadata_complete=False,
            complete=False,
            counts=_partial_counts(),
            pagination=Pagination(1, False, None),
            errors=("incomplete pagination",),
        )
    with pytest.raises(ContractError, match="source_name"):
        _run(
            status=RunStatus.PARTIAL,
            source_name=None,
            membership_complete=True,
            metadata_complete=False,
            complete=False,
            counts=_partial_counts(),
            pagination=Pagination(1, True, None),
            errors=("missing source",),
        )


def test_membership_complete_rejects_source_total_mismatch() -> None:
    with pytest.raises(ContractError, match="source_total"):
        _run(
            status=RunStatus.PARTIAL,
            membership_complete=True,
            metadata_complete=False,
            complete=False,
            counts=_partial_counts(),
            pagination=Pagination(2, True, SourceTotal(99, SourceTotalScope.RAW_ITEMS)),
            errors=("total mismatch",),
        )


def test_source_total_mismatch_is_allowed_for_incomplete_membership_diagnostic() -> None:
    record = _run(
        status=RunStatus.PARTIAL,
        membership_complete=False,
        metadata_complete=False,
        complete=False,
        counts=_partial_counts(),
        pagination=Pagination(2, True, SourceTotal(99, SourceTotalScope.RAW_ITEMS)),
        errors=(),
    )

    assert record.pagination.source_total.value == 99
