"""Contract tests for the isolated Crossref serial adapter."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from paper_agent.adapters.crossref import CrossrefSerialAdapter
from paper_agent.catalog import load_venue_spec
from paper_agent.errors import CollectionError
from paper_agent.models import Pagination, SourceTotal, SourceTotalScope


FIXTURES = Path(__file__).parents[1] / "fixtures" / "crossref"
ISSN = "1234-567X"
YEAR = 2024


class QueueTextClient:
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    def get_text(self, url: str) -> object:
        self.calls.append(url)
        if not self.responses:
            raise AssertionError(f"unexpected request: {url}")
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def _fixture(name: str) -> dict[str, object]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _response(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _spec(source: dict[str, object] | None = None):
    return replace(
        load_venue_spec("tcad"),
        source=source if source is not None else {"issn": ISSN},
        year_overrides={},
    )


def _base_item() -> dict[str, object]:
    payload = _fixture("page-full.json")
    message = payload["message"]
    assert isinstance(message, dict)
    item = message["items"][0]
    assert isinstance(item, dict)
    return deepcopy(item)


def _full_page(*, total: int = 1001, next_cursor: str = "cursor / + =") -> dict[str, object]:
    payload = _fixture("page-full.json")
    message = payload["message"]
    assert isinstance(message, dict)
    item = _base_item()
    message["total-results"] = total
    message["items"] = [deepcopy(item) for _ in range(1000)]
    message["next-cursor"] = next_cursor
    return payload


def _last_page(*, total: int = 1001, next_cursor: str = "ignored") -> dict[str, object]:
    payload = _fixture("page-last.json")
    message = payload["message"]
    assert isinstance(message, dict)
    message["total-results"] = total
    message["next-cursor"] = next_cursor
    return payload


def test_two_pages_exact_total_stop_and_encode_filter_and_cursor() -> None:
    first = _full_page()
    last = _last_page()
    client = QueueTextClient([_response(first), _response(last)])

    result = CrossrefSerialAdapter().collect(_spec(), YEAR, client)

    assert result.source_name == "crossref_serial"
    assert [paper.source_id for paper in result.papers] == [
        "10.5555/crossref.first",
        "10.5555/crossref.second",
    ]
    assert result.raw_items == 1001
    assert result.duplicate_occurrences == 999
    assert result.parse_rejects == ()
    assert result.pagination == Pagination(
        2, True, SourceTotal(1001, SourceTotalScope.RAW_ITEMS)
    )
    assert result.raw_items == (
        len(result.papers)
        + result.excluded_non_papers
        + result.duplicate_occurrences
        + len(result.parse_rejects)
    )

    first_query = parse_qs(urlsplit(client.calls[0]).query, keep_blank_values=True)
    second_query = parse_qs(urlsplit(client.calls[1]).query, keep_blank_values=True)
    assert urlsplit(client.calls[0]).path == "/journals/1234-567X/works"
    assert first_query["filter"] == [
        "from-pub-date:2024-01-01,until-pub-date:2024-12-31,type:journal-article"
    ]
    assert first_query["rows"] == ["1000"]
    assert first_query["cursor"] == ["*"]
    assert second_query["cursor"] == ["cursor / + ="]
    assert "%2F" in client.calls[1]
    assert "%2B" in client.calls[1]
    assert "%3D" in client.calls[1]
    assert len(client.calls) == 2


def test_zero_authoritative_total_is_terminal_without_a_cursor() -> None:
    client = QueueTextClient([_response(_fixture("zero.json"))])

    result = CrossrefSerialAdapter().collect(_spec(), YEAR, client)

    assert result.papers == ()
    assert result.raw_items == 0
    assert result.parse_rejects == ()
    assert result.pagination == Pagination(
        1, True, SourceTotal(0, SourceTotalScope.RAW_ITEMS)
    )
    assert len(client.calls) == 1


def test_exact_full_page_total_stops_even_when_crossref_returns_a_cursor() -> None:
    client = QueueTextClient([_response(_full_page(total=1000))])

    result = CrossrefSerialAdapter().collect(_spec(), YEAR, client)

    assert result.raw_items == 1000
    assert result.pagination == Pagination(
        1, True, SourceTotal(1000, SourceTotalScope.RAW_ITEMS)
    )
    assert len(client.calls) == 1


def test_crossref_fields_use_only_explicit_metadata_and_normalize_text() -> None:
    payload = _fixture("page-full.json")
    message = payload["message"]
    assert isinstance(message, dict)
    message["total-results"] = 1
    message["items"] = [message["items"][0]]
    message.pop("next-cursor")
    client = QueueTextClient([_response(payload)])

    result = CrossrefSerialAdapter().collect(_spec(), YEAR, client)

    paper = result.papers[0]
    assert paper.title == "First & Article"
    assert paper.authors == ("Named Group", "Ada Lovelace", "Hopper")
    assert paper.abstract == "First JATS abstract."
    assert paper.doi == "10.5555/crossref.first"
    assert paper.landing_url == "https://doi.org/10.5555/crossref.first"
    assert paper.pdf_candidates == (
        "https://publisher.test/first.pdf",
        "https://publisher.test/resource.pdf",
    )


def test_each_invalid_raw_item_becomes_a_parse_reject() -> None:
    client = QueueTextClient([_response(_fixture("invalid-items.json"))])

    result = CrossrefSerialAdapter().collect(_spec(), YEAR, client)

    assert result.papers == ()
    assert result.excluded_non_papers == 1
    assert result.raw_items == 8
    assert [reject.reason_code for reject in result.parse_rejects] == [
        "item_not_object",
        "missing_doi",
        "issn_mismatch",
        "publication_year_mismatch",
        "ambiguous_title",
        "invalid_author_name",
        "invalid_abstract",
    ]
    assert result.pagination.terminal_reached is True
    assert result.raw_items == result.excluded_non_papers + len(result.parse_rejects)


def test_explicit_crossref_non_paper_evidence_preserves_near_miss_research_items() -> None:
    client = QueueTextClient([_response(_fixture("eda-mixed.json"))])

    result = CrossrefSerialAdapter().collect(_spec(), YEAR, client)

    assert [paper.source_id for paper in result.papers] == [
        "10.5555/eda.research",
        "10.5555/eda.automatic-correction",
        "10.5555/eda.index-structure",
    ]
    assert result.excluded_non_papers == 9
    assert result.duplicate_occurrences == 1
    assert [reject.reason_code for reject in result.parse_rejects] == [
        "missing_type",
        "invalid_type",
        "invalid_issn",
        "missing_published_date",
        "invalid_title",
    ]
    assert result.raw_items == 18
    assert result.raw_items == (
        len(result.papers)
        + result.excluded_non_papers
        + result.duplicate_occurrences
        + len(result.parse_rejects)
    )
    assert result.pagination.source_total == SourceTotal(18, SourceTotalScope.RAW_ITEMS)
    assert result.pagination.terminal_reached is True
    assert len(client.calls) == 1


def test_duplicate_occurrence_and_identity_conflict_preserve_first_paper() -> None:
    client = QueueTextClient([_response(_fixture("duplicate-conflict.json"))])

    result = CrossrefSerialAdapter().collect(_spec(), YEAR, client)

    assert len(result.papers) == 1
    assert result.papers[0].title == "Stable duplicate"
    assert result.duplicate_occurrences == 1
    assert [reject.reason_code for reject in result.parse_rejects] == [
        "identity_conflict"
    ]
    assert result.raw_items == 3
    assert result.raw_items == (
        len(result.papers)
        + result.duplicate_occurrences
        + len(result.parse_rejects)
    )


@pytest.mark.parametrize(
    "payload",
    [
        "not json",
        [],
        {"status": "error", "message-type": "work-list", "message": {}},
        {"status": "ok", "message-type": "other", "message": {}},
        {"status": "ok", "message-type": "work-list", "message": []},
        {
            "status": "ok",
            "message-type": "work-list",
            "message": {"total-results": True, "items": []},
        },
        {
            "status": "ok",
            "message-type": "work-list",
            "message": {"total-results": 0, "items": {}},
        },
    ],
)
def test_first_transport_json_or_outer_schema_failure_is_typed(payload: object) -> None:
    response = payload if isinstance(payload, str) else _response(payload)
    client = QueueTextClient([response])

    with pytest.raises(CollectionError) as raised:
        CrossrefSerialAdapter().collect(_spec(), YEAR, client)

    assert raised.value.__cause__ is not None
    assert len(client.calls) == 1


def test_total_drift_keeps_only_pre_drift_trusted_pages() -> None:
    first = _full_page()
    second = _last_page(total=1000)
    client = QueueTextClient([_response(first), _response(second)])

    result = CrossrefSerialAdapter().collect(_spec(), YEAR, client)

    assert result.raw_items == 1000
    assert len(result.papers) == 1
    assert result.duplicate_occurrences == 999
    assert result.pagination == Pagination(
        1, False, SourceTotal(1001, SourceTotalScope.RAW_ITEMS)
    )
    assert len(client.calls) == 2


def test_page_items_over_remaining_total_are_not_accepted_as_terminal() -> None:
    first = _full_page()
    second = _last_page()
    second_message = second["message"]
    assert isinstance(second_message, dict)
    second_message["items"] = [deepcopy(_base_item()), deepcopy(_base_item())]
    client = QueueTextClient([_response(first), _response(second)])

    result = CrossrefSerialAdapter().collect(_spec(), YEAR, client)

    assert result.raw_items == 1000
    assert result.pagination == Pagination(
        1, False, SourceTotal(1001, SourceTotalScope.RAW_ITEMS)
    )
    assert len(client.calls) == 2


def test_short_page_with_unsettled_total_is_partial_and_not_terminal() -> None:
    payload = _full_page(total=1001)
    message = payload["message"]
    assert isinstance(message, dict)
    message["items"] = [message["items"][0]]
    client = QueueTextClient([_response(payload)])

    result = CrossrefSerialAdapter().collect(_spec(), YEAR, client)

    assert result.raw_items == 1
    assert len(result.papers) == 1
    assert result.pagination == Pagination(
        1, False, SourceTotal(1001, SourceTotalScope.RAW_ITEMS)
    )
    assert len(client.calls) == 1


def test_missing_cursor_keeps_received_full_page_without_claiming_terminal() -> None:
    payload = _full_page(total=1001)
    message = payload["message"]
    assert isinstance(message, dict)
    message.pop("next-cursor")
    client = QueueTextClient([_response(payload)])

    result = CrossrefSerialAdapter().collect(_spec(), YEAR, client)

    assert result.raw_items == 1000
    assert result.pagination == Pagination(
        1, False, SourceTotal(1001, SourceTotalScope.RAW_ITEMS)
    )
    assert len(client.calls) == 1


def test_repeated_cursor_keeps_received_page_without_following_cycle() -> None:
    payload = _full_page(total=2000, next_cursor="*")
    client = QueueTextClient([_response(payload)])

    result = CrossrefSerialAdapter().collect(_spec(), YEAR, client)

    assert result.raw_items == 1000
    assert result.pagination == Pagination(
        1, False, SourceTotal(2000, SourceTotalScope.RAW_ITEMS)
    )
    assert len(client.calls) == 1


def test_second_page_transport_failure_returns_partial_membership() -> None:
    first = _full_page()
    client = QueueTextClient([_response(first), CollectionError("connection reset")])

    result = CrossrefSerialAdapter().collect(_spec(), YEAR, client)

    assert result.raw_items == 1000
    assert len(result.papers) == 1
    assert result.pagination == Pagination(
        1, False, SourceTotal(1001, SourceTotalScope.RAW_ITEMS)
    )
    assert len(client.calls) == 2


def test_first_page_transport_failure_raises_collection_error_with_cause() -> None:
    client = QueueTextClient([CollectionError("connection reset")])

    with pytest.raises(CollectionError) as raised:
        CrossrefSerialAdapter().collect(_spec(), YEAR, client)

    assert raised.value.__cause__ is not None
    assert len(client.calls) == 1


def test_first_page_unexpected_assertion_error_propagates() -> None:
    client = QueueTextClient([AssertionError("adapter bug")])

    with pytest.raises(AssertionError, match="adapter bug"):
        CrossrefSerialAdapter().collect(_spec(), YEAR, client)

    assert len(client.calls) == 1


def test_second_page_unexpected_assertion_error_is_not_silent_partial() -> None:
    first = _full_page()
    client = QueueTextClient([_response(first), AssertionError("adapter bug")])

    with pytest.raises(AssertionError, match="adapter bug"):
        CrossrefSerialAdapter().collect(_spec(), YEAR, client)

    assert len(client.calls) == 2


@pytest.mark.parametrize(
    "source",
    [
        {},
        {"issn": "1234-567x"},
        {"issn": "1234-567"},
        {"issn": "12345-678X"},
        {"issn": "1234-567X", "issns": ["1234-567X"]},
        {"issn": "1234-567X", "registry_issn": "1234-567X"},
        {"issn": "1234-567X", "rows": 1000},
        {"issn": "1234-567X", "type": "journal-article"},
    ],
)
def test_source_accepts_only_one_strict_issn_parameter(source: dict[str, object]) -> None:
    client = QueueTextClient([])

    with pytest.raises(CollectionError):
        CrossrefSerialAdapter().collect(_spec(source), YEAR, client)

    assert client.calls == []
