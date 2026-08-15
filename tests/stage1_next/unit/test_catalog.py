"""Unit tests for the minimal Stage 1 venue catalog."""

from dataclasses import FrozenInstanceError
import importlib
from pathlib import Path
import tomllib
from types import MappingProxyType
import urllib.request

import pytest

from paper_agent_next import catalog as catalog_module
from paper_agent_next.catalog import VenueSpec, load_venue_spec
from paper_agent_next.errors import InputError
from paper_agent_next.models import VenueType


_BASE_SPEC = """\
schema_version = 1
id = "example"
name = "Example Conference"
venue_type = "conference"
adapter = "adapters.pmlr:PmlrAdapter"
enrichers = ["enrichers.openalex:OpenAlexEnricher"]
start_year = 2020
end_year = 2024
held_years = [2020, 2022, 2024]

[source]
series = "EXAMPLE"
enabled = true

[year_overrides."2022"]
volume = "v1"
"""


def _load_temporary_spec(monkeypatch, tmp_path: Path, document: str, venue_id: str = "example") -> VenueSpec:
    spec_dir = tmp_path / "venue-specs"
    spec_dir.mkdir()
    (spec_dir / f"{venue_id}.toml").write_text(document, encoding="utf-8")
    monkeypatch.setattr(catalog_module, "_VENUE_SPECS_DIR", spec_dir)
    return load_venue_spec(venue_id)


def test_public_surface_and_icml_fields() -> None:
    spec = load_venue_spec("icml")

    assert catalog_module.__all__ == ["VenueSpec", "load_venue_spec"]
    assert isinstance(spec, VenueSpec)
    assert spec.venue_id == "icml"
    assert spec.name == "International Conference on Machine Learning"
    assert spec.venue_type is VenueType.CONFERENCE
    assert spec.adapter == "adapters.pmlr:PmlrAdapter"
    assert spec.enrichers == ()
    assert spec.start_year == 1980
    assert spec.end_year == 2026
    assert spec.held_years == (1980, 1983, 1985, *range(1987, 2027))
    assert dict(spec.source) == {"series": "ICML"}
    assert dict(spec.year_overrides["2024"]) == {"volume": "v235"}


def test_venue_spec_is_frozen_slotted_and_nested_values_are_read_only() -> None:
    spec = load_venue_spec("icml")

    assert isinstance(spec.source, MappingProxyType)
    assert isinstance(spec.year_overrides, MappingProxyType)
    assert isinstance(spec.year_overrides["2024"], MappingProxyType)
    assert not hasattr(spec, "__dict__")
    with pytest.raises((FrozenInstanceError, AttributeError)):
        spec.name = "Changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        spec.source["series"] = "changed"  # type: ignore[index]
    with pytest.raises(TypeError):
        spec.year_overrides["2024"]["volume"] = "changed"  # type: ignore[index]

    resolved = spec.source_for_year(2024)
    assert isinstance(resolved, MappingProxyType)
    assert resolved is not spec.source
    assert dict(resolved) == {"series": "ICML", "volume": "v235"}
    with pytest.raises(TypeError):
        resolved["series"] = "changed"  # type: ignore[index]
    assert dict(spec.source) == {"series": "ICML"}


def test_source_resolution_without_override_is_a_fresh_base_copy() -> None:
    spec = load_venue_spec("icml")

    resolved = spec.source_for_year(2026)
    assert resolved is not spec.source
    assert dict(resolved) == {"series": "ICML"}
    assert dict(spec.source_for_year(2026)) == dict(spec.source)


@pytest.mark.parametrize(
    ("year", "expected"),
    [
        (1980, True),
        (1983, True),
        (1985, True),
        (1987, True),
        (2024, True),
        (2026, True),
        (1979, False),
        (1981, False),
        (1982, False),
        (1984, False),
        (1986, False),
        (2027, False),
    ],
)
def test_icml_applicability(year: int, expected: bool) -> None:
    assert load_venue_spec("icml").is_applicable(year) is expected


def test_source_for_year_rejects_inapplicable_year() -> None:
    with pytest.raises(InputError, match="not applicable"):
        load_venue_spec("icml").source_for_year(1981)


@pytest.mark.parametrize(
    "venue_id",
    [
        "ICML",
        "icml ",
        " icml",
        "international-conference-on-machine-learning",
        "icml.toml",
        "../icml",
        "icml/../icml",
        "/tmp/icml",
    ],
)
def test_noncanonical_venue_ids_are_rejected(venue_id: str) -> None:
    with pytest.raises(InputError):
        load_venue_spec(venue_id)


def test_unknown_venue_is_an_input_error() -> None:
    with pytest.raises(InputError, match="unknown venue"):
        load_venue_spec("does-not-exist")


def test_read_failure_is_an_input_error(monkeypatch, tmp_path: Path) -> None:
    spec_dir = tmp_path / "venue-specs"
    spec_dir.mkdir()
    (spec_dir / "broken.toml").mkdir()
    monkeypatch.setattr(catalog_module, "_VENUE_SPECS_DIR", spec_dir)

    with pytest.raises(InputError, match="could not read venue spec"):
        load_venue_spec("broken")


@pytest.mark.parametrize(
    ("label", "document"),
    [
        ("malformed TOML", "schema_version = [\n"),
        ("missing field", "schema_version = 1\n"),
        ("bool schema version", _BASE_SPEC.replace("schema_version = 1", "schema_version = true")),
        ("wrong schema version", _BASE_SPEC.replace("schema_version = 1", "schema_version = 2")),
        ("id mismatch", _BASE_SPEC.replace('id = "example"', 'id = "other"')),
        (
            "unknown top-level field",
            _BASE_SPEC.replace("\n[source]", "\nunknown = 1\n\n[source]"),
        ),
        (
            "invalid adapter path",
            _BASE_SPEC.replace('adapter = "adapters.pmlr:PmlrAdapter"', 'adapter = "paper_agent.pmlr:PmlrAdapter"'),
        ),
        (
            "invalid enricher path",
            _BASE_SPEC.replace('enrichers.openalex:OpenAlexEnricher', 'enrichers.openalex'),
        ),
        (
            "duplicate enricher",
            _BASE_SPEC.replace(
                'enrichers = ["enrichers.openalex:OpenAlexEnricher"]',
                'enrichers = ["enrichers.openalex:OpenAlexEnricher", "enrichers.openalex:OpenAlexEnricher"]',
            ),
        ),
        ("invalid start year", _BASE_SPEC.replace("start_year = 2020", "start_year = true")),
        (
            "reversed year range",
            _BASE_SPEC.replace("start_year = 2020", "start_year = 2024").replace("end_year = 2024", "end_year = 2020"),
        ),
        ("empty held years", _BASE_SPEC.replace("held_years = [2020, 2022, 2024]", "held_years = []")),
        (
            "unordered held years",
            _BASE_SPEC.replace("held_years = [2020, 2022, 2024]", "held_years = [2022, 2020, 2024]"),
        ),
        (
            "out of range held year",
            _BASE_SPEC.replace("held_years = [2020, 2022, 2024]", "held_years = [2019, 2020, 2024]"),
        ),
        (
            "nonapplicable override year",
            _BASE_SPEC.replace('[year_overrides."2022"]', '[year_overrides."2021"]'),
        ),
        (
            "invalid override key",
            _BASE_SPEC.replace('[year_overrides."2022"]', '[year_overrides."20x2"]'),
        ),
        (
            "non-scalar source value",
            _BASE_SPEC.replace('series = "EXAMPLE"', 'series = ["EXAMPLE"]'),
        ),
    ],
)
def test_malformed_specs_become_input_errors(
    label: str, document: str, monkeypatch, tmp_path: Path
) -> None:
    with pytest.raises(InputError, match="invalid") as caught:
        _load_temporary_spec(monkeypatch, tmp_path, document)
    assert label
    assert caught.value.__cause__ is None or isinstance(caught.value.__cause__, tomllib.TOMLDecodeError)


def test_optional_catalog_sections_default_to_empty_or_unbounded(monkeypatch, tmp_path: Path) -> None:
    document = """\
schema_version = 1
id = "example"
name = "Example Journal"
venue_type = "journal"
adapter = "adapters.crossref:CrossrefAdapter"
enrichers = []
start_year = 2020
"""
    spec = _load_temporary_spec(monkeypatch, tmp_path, document)

    assert spec.end_year is None
    assert spec.held_years is None
    assert dict(spec.source) == {}
    assert dict(spec.year_overrides) == {}
    assert spec.is_applicable(9999)


def test_loading_does_not_import_configured_classes_or_access_network(monkeypatch) -> None:
    def unexpected_import(*args, **kwargs):
        raise AssertionError(f"unexpected implementation import: {args!r} {kwargs!r}")

    def unexpected_network(*args, **kwargs):
        raise AssertionError(f"unexpected network access: {args!r} {kwargs!r}")

    monkeypatch.setattr(importlib, "import_module", unexpected_import)
    monkeypatch.setattr(urllib.request, "urlopen", unexpected_network)

    spec = load_venue_spec("icml")
    assert spec.adapter == "adapters.pmlr:PmlrAdapter"
    assert spec.enrichers == ()
