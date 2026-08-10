from __future__ import annotations

from dataclasses import FrozenInstanceError
from hashlib import sha256
from pathlib import Path

import pytest

from paper_agent.grants import GrantStore
from paper_agent.canonical import content_hash
from paper_agent.processing import (
    ArtifactProcessingPolicy,
    PROCESSING_MODEL,
    PROCESSING_PROVIDER,
    ProcessingGate,
    ProcessingOutcome,
    ProcessingRequest,
)
from paper_agent.storage import Database


ROOT = Path(__file__).resolve().parents[1]
NOW = "2026-08-09T12:00:00Z"
FUTURE = "2026-08-10T00:00:00Z"
PDF_BYTES = b"%PDF-restricted-content"
HASH = sha256(PDF_BYTES).hexdigest()


@pytest.fixture
def gate() -> ProcessingGate:
    return ProcessingGate(ArtifactProcessingPolicy.load(ROOT / "policies" / "artifact-processing-v1.yaml"))


def request(**changes: object) -> ProcessingRequest:
    values: dict[str, object] = {
        "artifact_hash": HASH, "artifact": "pdf", "input_scope": "full_pdf",
        "license": "CC-BY-4.0", "access_basis": "open_license",
        "purpose": "internal_analysis", "data_category": "full_text",
        "pdf_bytes": PDF_BYTES, "normalized_text_bytes": b"full text",
    }
    values.update(changes)
    if "artifact_hash" not in changes:
        payload = {
            "pdf": values.get("pdf_bytes"),
            "normalized_text": values.get("normalized_text_bytes"),
            "abstract": values.get("abstract_bytes"),
        }.get(str(values["artifact"]))
        values["artifact_hash"] = (
            content_hash(dict(values["metadata"]))
            if values["artifact"] == "metadata"
            else sha256(payload).hexdigest()  # type: ignore[arg-type]
        )
    return ProcessingRequest(**values)  # type: ignore[arg-type]


def test_open_full_pdf_is_dispatched_and_decision_is_auditable(gate: ProcessingGate) -> None:
    calls = []
    result = gate.dispatch(request(), lambda invocation: calls.append(invocation) or "ok")

    assert result.result == "ok"
    assert result.decision.outcome is ProcessingOutcome.FULL_PDF
    assert result.decision.authorized_by == "policy"
    assert result.decision.input_artifact_hash == HASH
    assert len(result.decision.audit_hash) == 64
    assert calls[0].pdf_bytes == PDF_BYTES
    assert calls[0].normalized_text_bytes is None
    with pytest.raises(FrozenInstanceError):
        result.decision.reason_code = "changed"  # type: ignore[misc]


def test_denied_full_content_never_reaches_supplied_callback(gate: ProcessingGate) -> None:
    calls = []
    result = gate.dispatch(
        request(access_basis="user_subscription", license=None),
        lambda invocation: calls.append(invocation),
    )

    assert result.decision.outcome is ProcessingOutcome.MANUAL
    assert result.result is None
    assert calls == []


def test_unversioned_license_is_not_promoted_to_an_open_processing_license(gate: ProcessingGate) -> None:
    result = gate.dispatch(request(license="CC-BY"), lambda _invocation: pytest.fail("must not call"))

    assert result.decision.outcome is ProcessingOutcome.ANALYSIS_NOT_AUTHORIZED


def test_abstract_only_strips_pdf_and_text_bytes_before_callback(gate: ProcessingGate) -> None:
    calls = []
    result = gate.dispatch(
        request(
            artifact="abstract", input_scope="abstract_only", data_category="abstract",
            access_basis="public_read_only", license=None, abstract_bytes=b"safe abstract",
        ),
        calls.append,
    )

    assert result.decision.outcome is ProcessingOutcome.ABSTRACT_ONLY
    assert len(calls) == 1
    assert calls[0].abstract_bytes == b"safe abstract"
    assert calls[0].pdf_bytes is None
    assert calls[0].normalized_text_bytes is None


def test_normalized_text_hash_authorizes_only_the_bound_text_payload(gate: ProcessingGate) -> None:
    calls = []
    result = gate.dispatch(
        request(
            artifact="normalized_text", data_category="normalized_text",
            pdf_bytes=PDF_BYTES, normalized_text_bytes=b"normalized paper text",
        ),
        calls.append,
    )

    assert result.decision.outcome is ProcessingOutcome.FULL_PDF
    assert calls[0].pdf_bytes is None
    assert calls[0].normalized_text_bytes == b"normalized paper text"


def test_declared_hash_must_match_selected_payload() -> None:
    with pytest.raises(ValueError, match="does not match"):
        request(artifact_hash="b" * 64)


def _approved_grant(store: GrantStore, *, artifact_hash: str = HASH) -> None:
    scope = {
        "paper_ids": ["paper-1"], "artifact_hashes": [artifact_hash],
        "collection_ids": [], "collection_snapshot_hash": None,
        "selection_snapshot_hash": None, "domains": [],
        "provider": PROCESSING_PROVIDER, "model": PROCESSING_MODEL,
        "data_categories": ["full_text"],
    }
    draft = store.create_draft(
        grant_id="processing-grant", kind="remote_model_processing",
        actions=["remote_model_processing"], purpose="internal_analysis", mode="attended",
        scope=scope, max_papers=1, expires_at=FUTURE,
    )
    store.approve(draft, draft["content_hash"], approved_by="owner", approved_at=NOW)


@pytest.fixture
def grant_gate(tmp_path) -> tuple[ProcessingGate, GrantStore]:
    database = Database(tmp_path / "papers.sqlite")
    database.migrate()
    store = GrantStore(database)
    yield ProcessingGate(ArtifactProcessingPolicy.load(ROOT / "policies" / "artifact-processing-v1.yaml"), store), store
    database.close()


def test_exact_artifact_bound_grant_can_authorize_full_pdf(grant_gate: tuple[ProcessingGate, GrantStore]) -> None:
    gate, store = grant_gate
    _approved_grant(store)
    calls = []
    result = gate.dispatch(
        request(access_basis="user_subscription", license=None, paper_id="paper-1"), calls.append,
        processing_grant_id="processing-grant", now=NOW,
    )

    assert result.decision.outcome is ProcessingOutcome.FULL_PDF
    assert result.decision.authorized_by == "grant"
    assert len(calls) == 1


@pytest.mark.parametrize("changes", [
    {"pdf_bytes": b"%PDF-another-artifact"},
    {"provider": "another_provider"},
    {"model": "gpt-5.6-sol"},
])
def test_incompatible_grant_scope_or_remote_target_means_zero_calls(
    grant_gate: tuple[ProcessingGate, GrantStore], changes: dict[str, object]
) -> None:
    gate, store = grant_gate
    _approved_grant(store)
    calls = []
    result = gate.dispatch(
        request(access_basis="user_subscription", license=None, paper_id="paper-1", **changes), calls.append,
        processing_grant_id="processing-grant", now=NOW,
    )

    assert not result.decision.is_authorized
    assert calls == []


@pytest.mark.parametrize("at, revoke", [("2026-08-11T00:00:00Z", False), (NOW, True)])
def test_expired_or_revoked_processing_grant_cannot_dispatch(
    grant_gate: tuple[ProcessingGate, GrantStore], at: str, revoke: bool
) -> None:
    gate, store = grant_gate
    _approved_grant(store)
    if revoke:
        store.revoke("processing-grant", actor="owner", event_at=NOW)
    calls = []
    result = gate.dispatch(
        request(access_basis="user_subscription", license=None, paper_id="paper-1"), calls.append,
        processing_grant_id="processing-grant", now=at,
    )

    assert result.decision.outcome is ProcessingOutcome.MANUAL
    assert calls == []
