"""Detached approval records for plans and grants."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from .canonical import content_hash


IDENTITY_METADATA = {
    "plan_id",
    "grant_id",
    "plan_hash",
    "content_hash",
    "status",
    "created_at",
    "updated_at",
    "approval",
}


class ApprovalError(ValueError):
    pass


def approved_content(document: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in document.items() if key not in IDENTITY_METADATA}


def approved_content_hash(document: Mapping[str, Any]) -> str:
    return content_hash(approved_content(document))


def approve(
    document: Mapping[str, Any],
    expected_hash: str,
    *,
    approved_by: str,
    approved_at: str,
    hash_field: str,
) -> dict[str, Any]:
    actual_hash = approved_content_hash(document)
    if actual_hash != expected_hash:
        raise ApprovalError(f"content hash mismatch: expected {expected_hash}, got {actual_hash}")
    if document.get("status") != "draft":
        raise ApprovalError("only draft documents can be approved")

    approved = deepcopy(dict(document))
    approved[hash_field] = actual_hash
    approved["status"] = "approved"
    approved["approval"] = {
        "approved_hash": actual_hash,
        "approved_by": approved_by,
        "approved_at": approved_at,
        "approval_method": "cli_hash",
    }
    return approved


def require_valid_approval(document: Mapping[str, Any], hash_field: str) -> None:
    if document.get("status") != "approved":
        raise ApprovalError("document is not approved")
    actual_hash = approved_content_hash(document)
    approval = document.get("approval")
    if not isinstance(approval, dict):
        raise ApprovalError("approval record is missing")
    if document.get(hash_field) != actual_hash or approval.get("approved_hash") != actual_hash:
        raise ApprovalError("approved document content has drifted")
