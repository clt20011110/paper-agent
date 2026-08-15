"""Contract coverage for explicit trusted implementation loading."""

from types import SimpleNamespace

import pytest

from paper_agent import loading
from paper_agent.errors import InputError


def test_loading_loads_adapter_and_enricher_without_a_registry() -> None:
    adapter = loading.load_adapter("adapters.dblp:DblpTocAdapter")
    enricher = loading.load_enricher(
        "enrichers.semantic_scholar:SemanticScholarEnricher"
    )

    assert adapter.source_name == "dblp_toc"
    assert callable(adapter.collect)
    assert enricher.source_name == "semantic_scholar"
    assert callable(enricher.enrich)
    assert loading.load_enrichers((
        "enrichers.semantic_scholar:SemanticScholarEnricher",
    ))[0].source_name == "semantic_scholar"


@pytest.mark.parametrize(
    ("path", "loader"),
    [
        ("paper_agent.adapters.dblp:DblpTocAdapter", loading.load_adapter),
        ("adapters.dblp", loading.load_adapter),
        ("adapters.dblp:DblpTocAdapter.extra", loading.load_adapter),
        ("enrichers.semantic_scholar:SemanticScholarEnricher", loading.load_adapter),
    ],
)
def test_loading_repeats_path_trust_boundary(path: str, loader) -> None:
    with pytest.raises(InputError):
        loader(path)


def test_loading_checks_instance_contract_and_keeps_error_safe(monkeypatch) -> None:
    def importer(module_name: str):
        assert module_name == "paper_agent.adapters.fake"
        return SimpleNamespace(Fake=object())

    monkeypatch.setattr(loading.importlib, "import_module", importer)

    with pytest.raises(InputError, match="could not load configured adapter") as raised:
        loading.load_adapter("adapters.fake:Fake")
    assert isinstance(raised.value.__cause__, TypeError)
    assert "secret" not in str(raised.value)


def test_loading_enrichers_requires_catalog_tuple() -> None:
    with pytest.raises(InputError):
        loading.load_enrichers([])  # type: ignore[arg-type]
