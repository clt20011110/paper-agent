from paper_agent.domain import FilterStatus, Paper
from paper_agent.scope_filter import evaluate_scope, screening_scope_hash


SCOPE = {
    "date_from": "2024-01-01",
    "date_to": "2024-12-31",
    "venues": ["ICML"],
    "fields": ["computer science"],
    "languages": ["en"],
    "document_types": ["article"],
    "user_seeds": [],
    "include_arxiv_candidates": False,
}


def _paper(publication_date: str | None = "2024-07-10") -> Paper:
    return Paper("paper-1", "A Paper", publication_date=publication_date, year=2024)


def _metadata(**overrides):
    return {
        "venue": "ICML",
        "fields_of_study": ["Computer Science"],
        "language": "eng",
        "type": "journal-article",
        **overrides,
    }


def test_scope_filter_normalizes_known_metadata_and_includes_match() -> None:
    decision = evaluate_scope(_paper(), (_metadata(),), SCOPE)

    assert decision.status is FilterStatus.RELEVANT
    assert decision.reason_code == "scope_match"
    assert len(decision.input_hash) == 64


def test_scope_filter_excludes_a_known_mismatch_before_missing_values() -> None:
    decision = evaluate_scope(
        _paper(),
        ({"venue": "ICML", "fields": ["computer science"], "type": "editorial"},),
        SCOPE,
    )

    assert decision.status is FilterStatus.IRRELEVANT
    assert decision.reason_code == "scope_document_type_mismatch"


def test_scope_filter_sends_missing_required_metadata_to_review() -> None:
    metadata = _metadata()
    del metadata["language"]

    decision = evaluate_scope(_paper(), (metadata,), SCOPE)

    assert decision.status is FilterStatus.NEEDS_REVIEW
    assert decision.reason_code == "scope_language_unverified"


def test_partial_date_overlapping_a_boundary_needs_review() -> None:
    scope = {**SCOPE, "date_from": "2024-06-15"}

    decision = evaluate_scope(_paper("2024"), (_metadata(),), scope)

    assert decision.status is FilterStatus.NEEDS_REVIEW
    assert decision.reason_code == "scope_date_unverified"


def test_screening_scope_hash_excludes_execution_budgets() -> None:
    plan = {
        "research": {"objective": "map methods"},
        "inclusion": {"criteria": ["empirical"], "exclusion_criteria": []},
        "scope": SCOPE,
        "budgets": {"max_requests": 10},
    }

    assert screening_scope_hash(plan) == screening_scope_hash(
        {**plan, "budgets": {"max_requests": 20}}
    )
    assert screening_scope_hash(plan) != screening_scope_hash(
        {**plan, "inclusion": {"criteria": ["theory"], "exclusion_criteria": []}}
    )
