import pytest

from paper_agent.approval import (
    ApprovalError,
    approve,
    approved_content_hash,
    require_valid_approval,
)


def draft() -> dict[str, object]:
    return {
        "plan_id": "plan-1",
        "plan_hash": "0" * 64,
        "status": "draft",
        "created_at": "2026-08-09T00:00:00Z",
        "approval": None,
        "research": {"question": "What works?"},
        "budget": {"max_requests": 10},
    }


def test_approval_is_detached_from_content_identity() -> None:
    document = draft()
    expected_hash = approved_content_hash(document)
    approved = approve(
        document,
        expected_hash,
        approved_by="owner",
        approved_at="2026-08-09T01:00:00Z",
        hash_field="plan_hash",
    )

    require_valid_approval(approved, "plan_hash")
    assert approved["approval"]["approved_hash"] == expected_hash  # type: ignore[index]
    assert document["status"] == "draft"


def test_business_field_drift_invalidates_approval() -> None:
    document = draft()
    approved = approve(
        document,
        approved_content_hash(document),
        approved_by="owner",
        approved_at="2026-08-09T01:00:00Z",
        hash_field="plan_hash",
    )
    approved["budget"] = {"max_requests": 11}

    with pytest.raises(ApprovalError, match="drifted"):
        require_valid_approval(approved, "plan_hash")


def test_wrong_explicit_hash_is_rejected() -> None:
    with pytest.raises(ApprovalError, match="content hash mismatch"):
        approve(
            draft(),
            "f" * 64,
            approved_by="owner",
            approved_at="2026-08-09T01:00:00Z",
            hash_field="plan_hash",
        )
