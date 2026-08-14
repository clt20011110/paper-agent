"""Offline unit tests for DOI normalization."""

import pytest

import paper_agent_next.normalize as normalize_module
from paper_agent_next.errors import ContractError


@pytest.mark.parametrize("value", [None, "", "   ", "\t\n "])
def test_empty_values_return_none(value) -> None:
    assert normalize_module.normalize_doi(value) is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("10.1000/ABC.1", "10.1000/abc.1"),
        ("doi:10.1000/ABC.1", "10.1000/abc.1"),
        (" DOI: 10.1000/ABC.1 ", "10.1000/abc.1"),
        ("https://doi.org/10.1000/ABC.1", "10.1000/abc.1"),
        ("http://doi.org/10.1000/ABC.1", "10.1000/abc.1"),
        ("HTTPS://DX.DOI.ORG/10.1000/ABC.1", "10.1000/abc.1"),
        ("http://dx.doi.org/10.1000/ABC.1", "10.1000/abc.1"),
    ],
)
def test_known_wrappers_and_bare_doi_normalize_to_lowercase(value, expected) -> None:
    assert normalize_module.normalize_doi(value) == expected


def test_doi_url_query_and_fragment_are_not_part_of_result() -> None:
    value = "https://doi.org/10.1000/ABC.1?utm_source=example#details"

    assert normalize_module.normalize_doi(value) == "10.1000/abc.1"


def test_suffix_punctuation_is_preserved() -> None:
    assert normalize_module.normalize_doi("10.1000/ABC.1).") == "10.1000/abc.1)."


@pytest.mark.parametrize(
    "value",
    [
        "not a DOI",
        "https://example.org/10.1000/abc",
        "10./abc",
        "10.1000",
        "10.1000/",
        "10.1000/abc def",
        "doi:https://doi.org/10.1000/abc",
    ],
)
def test_invalid_doi_strings_return_none(value) -> None:
    assert normalize_module.normalize_doi(value) is None


@pytest.mark.parametrize("value", [True, 7, b"10.1000/abc", ["10.1000/abc"]])
def test_non_string_values_raise_contract_error(value) -> None:
    with pytest.raises(ContractError):
        normalize_module.normalize_doi(value)


@pytest.mark.parametrize(
    "value",
    [
        "10.1000/ABC.1",
        "doi:10.1000/ABC.1",
        "https://dx.doi.org/10.1000/ABC.1?query#fragment",
    ],
)
def test_normalization_is_idempotent(value) -> None:
    result = normalize_module.normalize_doi(value)

    assert result is not None
    assert normalize_module.normalize_doi(result) == result


def test_public_surface_is_exact() -> None:
    assert normalize_module.__all__ == ["normalize_doi"]
