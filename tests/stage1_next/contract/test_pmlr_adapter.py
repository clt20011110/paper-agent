from dataclasses import replace
import ast
from pathlib import Path

import pytest

from paper_agent_next.adapters.base import CollectedPaper, CollectionResult, ParseReject
from paper_agent_next.adapters.pmlr import PmlrAdapter
from paper_agent_next.catalog import load_venue_spec
from paper_agent_next.errors import CollectionError, ContractError
from paper_agent_next.models import Pagination


FIXTURES = Path(__file__).parents[1] / "fixtures" / "pmlr"
VOLUME_URL = "https://proceedings.mlr.press/v235/"
ADA_URL = "https://proceedings.mlr.press/v235/lovelace24a.html"
TURING_URL = "https://proceedings.mlr.press/v235/turing24a.html"


class FakeTextClient:
    def __init__(self, responses: dict[str, str]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    def get_text(self, url: str) -> str:
        self.calls.append(url)
        if url not in self.responses:
            raise AssertionError(f"unexpected offline URL: {url}")
        if url.casefold().endswith(".pdf"):
            raise AssertionError("the adapter must not request PDF bytes")
        return self.responses[url]


def _fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def _client(*, volume: str | None = None, details: dict[str, str] | None = None) -> FakeTextClient:
    responses = {
        VOLUME_URL: _fixture("volume-v235.html"),
        ADA_URL: _fixture("lovelace24a.html"),
        TURING_URL: _fixture("turing24a.html"),
    }
    if volume is not None:
        responses[VOLUME_URL] = volume
    if details:
        responses.update(details)
    return FakeTextClient(responses)


def test_collects_authoritative_membership_and_reconciles_every_occurrence() -> None:
    client = _client()

    result = PmlrAdapter().collect(load_venue_spec("icml"), 2024, client)

    assert result.source_name == "pmlr"
    assert [paper.source_id for paper in result.papers] == [
        "v235/lovelace24a",
        "v235/turing24a",
    ]
    ada, turing = result.papers
    assert ada.title == "Reliable Small Models & Graphs"
    assert ada.authors == ("Ada Lovelace", "Grace Hopper")
    assert ada.abstract == "Reliable small models & graphs for reproducible experiments."
    assert ada.landing_url == ADA_URL
    assert ada.pdf_candidates == (
        "https://proceedings.mlr.press/v235/lovelace24a/lovelace24a.pdf",
    )
    assert turing.title == "Parallel Inference"
    assert turing.authors == ("Alan Turing", "Grace Hopper")
    assert turing.abstract == "Parallel inference reduces latency. It also preserves & checks accuracy."
    assert turing.landing_url == TURING_URL
    assert turing.pdf_candidates == ()
    assert result.raw_items == 5
    assert result.excluded_non_papers == 1
    assert result.duplicate_occurrences == 1
    assert len(result.parse_rejects) == 1
    assert result.raw_items == (
        len(result.papers)
        + result.excluded_non_papers
        + result.duplicate_occurrences
        + len(result.parse_rejects)
    )
    assert result.pagination == Pagination(1, True, None)
    assert client.calls == [VOLUME_URL, ADA_URL, TURING_URL]


def test_conflicting_duplicate_is_a_structured_parse_reject() -> None:
    volume = """\
    <html><body>
      <div class="paper"><p class="title">First title</p>
        <span class="authors"><a>Ada Lovelace</a></span>
        <a href="lovelace24a.html">abs</a>
      </div>
      <div class="paper"><p class="title">Conflicting title</p>
        <span class="authors"><a>Ada Lovelace</a></span>
        <a href="lovelace24a.html">abs</a>
      </div>
    </body></html>
    """
    client = _client(
        volume=volume,
        details={ADA_URL: '<div id="abstract">One abstract.</div>'},
    )

    result = PmlrAdapter().collect(load_venue_spec("icml"), 2024, client)

    assert len(result.papers) == 1
    assert result.duplicate_occurrences == 0
    assert len(result.parse_rejects) == 1
    reject = result.parse_rejects[0]
    assert reject.reason_code == "identity_conflict"
    assert "conflicting duplicate metadata" in reject.message
    assert reject.source_locator == f"{VOLUME_URL}#paper-2"
    assert client.calls == [VOLUME_URL, ADA_URL]


def test_missing_or_invalid_volume_fails_before_any_http_call() -> None:
    spec = load_venue_spec("icml")
    for overrides in (
        {},
        {"2024": {"volume": "235"}},
        {"2024": {"volume": "v235/"}},
        {"2024": {"volume": True}},
    ):
        candidate = replace(spec, year_overrides=overrides)
        client = FakeTextClient({})

        with pytest.raises(CollectionError, match="explicit volume"):
            PmlrAdapter().collect(candidate, 2024, client)
        assert client.calls == []


def test_empty_volume_page_is_not_an_authoritative_empty_collection() -> None:
    client = _client(volume="<html><body><div class='not-paper'>Nothing</div></body></html>")

    with pytest.raises(CollectionError, match="no div.paper"):
        PmlrAdapter().collect(load_venue_spec("icml"), 2024, client)

    assert client.calls == [VOLUME_URL]


def test_valid_identity_keeps_missing_initial_fields_and_missing_pdf() -> None:
    volume = '<html><body><div class="paper"><a href="minimal24a.html">abs</a></div></body></html>'
    client = _client(
        volume=volume,
        details={
            "https://proceedings.mlr.press/v235/minimal24a.html":
                "<html><body><p>No abstract here.</p></body></html>"
        },
    )

    result = PmlrAdapter().collect(load_venue_spec("icml"), 2024, client)

    assert result.papers[0].source_id == "v235/minimal24a"
    assert result.papers[0].title is None
    assert result.papers[0].authors == ()
    assert result.papers[0].abstract is None
    assert result.papers[0].pdf_candidates == ()
    assert result.parse_rejects == ()
    assert client.calls == [VOLUME_URL, "https://proceedings.mlr.press/v235/minimal24a.html"]


def test_collection_result_rejects_census_mismatch_and_mutable_collections() -> None:
    paper = CollectedPaper(
        source_id="v1/paper",
        title=None,
        authors=(),
        abstract=None,
        landing_url="https://proceedings.mlr.press/v1/paper.html",
        pdf_candidates=(),
    )
    reject = ParseReject("https://example.test/#paper-1", "missing_landing_url", "missing")
    valid = CollectionResult(
        source_name="pmlr",
        papers=(paper,),
        raw_items=2,
        excluded_non_papers=0,
        duplicate_occurrences=0,
        parse_rejects=(reject,),
        pagination=Pagination(1, True, None),
    )
    assert valid.raw_items == 2
    with pytest.raises(ContractError, match="raw_items"):
        replace(valid, raw_items=1)
    with pytest.raises(ContractError, match="papers"):
        CollectionResult(
            source_name="pmlr",
            papers=[paper],  # type: ignore[arg-type]
            raw_items=1,
            excluded_non_papers=0,
            duplicate_occurrences=0,
            parse_rejects=(),
            pagination=Pagination(1, True, None),
        )


def test_new_adapter_modules_do_not_import_the_legacy_package() -> None:
    root = Path(__file__).resolve().parents[3] / "src" / "paper_agent_next" / "adapters"
    violations: list[str] = []
    for path in sorted(root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                if any(alias.name == "paper_agent" or alias.name.startswith("paper_agent.") for alias in node.names):
                    violations.append(f"{path}:{node.lineno}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module == "paper_agent" or module.startswith("paper_agent."):
                    violations.append(f"{path}:{node.lineno}")
    assert violations == []
