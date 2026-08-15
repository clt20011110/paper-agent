"""Offline unit tests for DOI normalization."""

import pytest
from unicodedata import normalize as normalize_unicode

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
    assert normalize_module.__all__ == ["normalize_doi", "normalize_text"]


@pytest.mark.parametrize("value", [None, "", " \t\n", "<div><br></div>", "<!-- comment -->"])
def test_text_empty_or_markup_only_returns_none(value) -> None:
    assert normalize_module.normalize_text(value) is None


def test_text_decodes_named_numeric_entities_once() -> None:
    value = "R&amp;D &#945; &#x2211; &amp;lt;"

    assert normalize_module.normalize_text(value) == "R&D α ∑ &lt;"


def test_text_removes_tags_and_preserves_block_boundaries() -> None:
    value = "<p>First</p><br><div>Second</div><li>Third</li><jats:p>Fourth</jats:p>"

    assert normalize_module.normalize_text(value) == "First Second Third Fourth"


def test_text_ignores_comments_declarations_and_processing_instructions() -> None:
    value = "A<!-- hidden --><!DOCTYPE note><?pi data?>B"

    assert normalize_module.normalize_text(value) == "AB"


def test_text_drops_script_and_style_content() -> None:
    value = "<p>Keep</p><script>drop &amp; this</script><style>.x { color: red; }</style><p>Text</p>"

    assert normalize_module.normalize_text(value) == "Keep Text"


def test_text_does_not_add_space_for_inline_markup() -> None:
    assert normalize_module.normalize_text("H<sub>2</sub>O") == "H2O"


def test_text_collapses_unicode_whitespace_and_noise_to_separators() -> None:
    value = "A\u00a0B\x00C\u200bD\ufeffE\tF"

    assert normalize_module.normalize_text(value) == "A B C D E F"


def test_text_normalizes_to_nfc() -> None:
    value = "Cafe\u0301"

    result = normalize_module.normalize_text(value)

    assert result == "Café"
    assert result == normalize_unicode("NFC", result)


def test_text_preserves_content_and_comparison_symbols() -> None:
    value = "中文  Αβ ∑ 2 < 3 > 1 — “Quoted”"

    assert normalize_module.normalize_text(value) == "中文 Αβ ∑ 2 < 3 > 1 — “Quoted”"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("x<y and y>z", "x<y and y>z"),
        ("A<B>C", "A<B>C"),
        ("x<y && y>z", "x<y && y>z"),
        ("<foo>bar</foo>", "bar"),
        ("<jats:p>First</jats:p><p>Second</p>", "First Second"),
    ],
)
def test_unmatched_angle_brackets_are_preserved_but_matched_tags_are_removed(
    value, expected
) -> None:
    assert normalize_module.normalize_text(value) == expected


@pytest.mark.parametrize("value", [True, 7, b"text", ["text"]])
def test_text_non_string_values_raise_contract_error(value) -> None:
    with pytest.raises(ContractError):
        normalize_module.normalize_text(value)


@pytest.mark.parametrize("value", ["<p> Café </p>", "H<sub>2</sub>O", "中文\u00a0Αβ"])
def test_text_normalization_is_idempotent(value) -> None:
    result = normalize_module.normalize_text(value)

    assert result is not None
    assert normalize_module.normalize_text(result) == result
