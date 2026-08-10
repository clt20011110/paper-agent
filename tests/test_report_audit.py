from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from hashlib import sha256
import json
from pathlib import Path

import pytest

from paper_agent.canonical import canonical_json, content_hash
from paper_agent.codex_exec import CodexExecResult, InvocationMetadata
from paper_agent.report_artifacts import ReportArtifactStore
from paper_agent.report_audit import (
    MODEL,
    PROFILE,
    REASONING_EFFORT,
    ReportAuditCoordinator,
    ReportAuditError,
    ReportBundle,
)
from test_report_reduce import _fixture


DISCLOSURE = (
    "抽取范围：full_pdf=2；全文、摘要和元数据证据已分层，"
    "缺失全文不作全文事实表述。"
)


@dataclass
class FakeAuditSol:
    coordinator: ReportAuditCoordinator
    calls: list
    severe_a: bool = False
    severe_c: bool = False
    incomplete_coverage: bool = False
    wrong_actual_model: bool = False
    arbitrary_repair: bool = False
    reuse_invocation: bool = False

    def invoke(self, request):
        self.calls.append(request)
        payload = json.loads(request.prompt)
        if request.call_kind == "quality_audit":
            audit_pass = payload["audit_pass"]
            severe = self.severe_a if audit_pass == "A" else self.severe_c
            findings = []
            if severe:
                block = payload["report_document"]["blocks"][0]
                findings.append({
                    "finding_id": f"major-{audit_pass.lower()}",
                    "severity": "major",
                    "category": "overgeneralization",
                    "description": "The boundary needs an explicit qualification.",
                    "block_ids": [block["block_id"]],
                    "claim_ids": [block["claim_ids"][0]],
                    "paper_ids": [block["citation_paper_ids"][0]],
                    "recommendation": "Add the frozen boundary without changing evidence.",
                })
            output = {
                "audit_pass": audit_pass,
                "report_document_hash": payload["report_document_hash"],
                "report_artifact_hash": payload["report_artifact_hash"],
                "report_plan_hash": payload["report_plan_hash"],
                "rubric_hash": payload["rubric_hash"],
                "search_limitations_hash": payload["search_limitations_hash"],
                "coverage_complete": not self.incomplete_coverage,
                "coverage_ledger": payload["expected_coverage_ledger"],
                "findings": findings,
            }
        else:
            block = deepcopy(payload["report_document"]["blocks"][0])
            block["text"] += " 已明确该结论仅适用于冻结条件。"
            patch = {
                    "target": "REPORT_DOCUMENT",
                    "operation": "replace_block",
                    "block_id": block["block_id"],
                    "precondition_hash": content_hash(payload["report_document"]["blocks"][0]),
                    "value": block,
            }
            if self.arbitrary_repair:
                patch = {
                    "target": "REPORT_DOCUMENT", "operation": "replace",
                    "path": "/blocks/0/text", "precondition_hash": "0" * 64,
                    "value": "untyped replacement",
                }
            output = {
                "base_artifact_hash": payload["base_artifact_hash"],
                "patches": [patch],
            }
        rendered = self.coordinator._rendered_prompt(request.call_kind, request.prompt)
        actual_model = "gpt-5.6-luna" if self.wrong_actual_model else MODEL
        self.wrong_actual_model = False
        metadata = InvocationMetadata(
            invocation_id=(
                "audit-invocation-reused"
                if self.reuse_invocation
                else f"audit-invocation-{len(self.calls)}"
            ),
            profile=PROFILE,
            model=MODEL,
            reasoning_effort=REASONING_EFFORT,
            schema_name=request.schema_name,
            schema_hash=self.coordinator.schema_hashes[request.call_kind],
            input_hash=request.input_hash,
            prompt_name=request.prompt_name,
            prompt_hash=self.coordinator.prompt_hashes[request.call_kind],
            rendered_prompt_hash=sha256(rendered.encode("utf-8")).hexdigest(),
            call_kind=request.call_kind,
            attempts=1,
            actual_model=actual_model,
            actual_profile=PROFILE,
            schema_path=request.schema_path,
            prompt_path=request.prompt_path,
        )
        return CodexExecResult(output, metadata)


@dataclass
class AuditFixture:
    reduce: object
    coordinator: ReportAuditCoordinator
    fake: FakeAuditSol
    bundle: ReportBundle
    constructions: list
    root: Path

    def run(self, **kwargs):
        return self.coordinator.run(
            "report-1",
            self.bundle,
            now="2026-08-10T02:00:00Z",
            **kwargs,
        )


def _audit_fixture(tmp_path: Path) -> AuditFixture:
    reduce = _fixture(tmp_path, max_input_tokens=50_000_000)
    base_fake = reduce.fake

    class DisclosureFinal:
        def invoke(self, request):
            result = base_fake.invoke(request)
            if request.call_kind != "final_reduce":
                return result
            output = deepcopy(dict(result.output))
            output["blocks"][0]["text"] += " " + DISCLOSURE
            return CodexExecResult(output, result.metadata)

    reduce.database.connection.execute(
        """INSERT OR IGNORE INTO report_plans(
               report_plan_id, content_hash, schema_version, plan_json,
               approval_json, status
           ) VALUES (?, ?, ?, ?, ?, 'approved')""",
        (
            reduce.plan["plan_id"],
            reduce.plan["plan_hash"],
            reduce.plan["schema_version"],
            json.dumps(reduce.plan, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            json.dumps(reduce.plan["approval"], ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        ),
    )
    reduce.database.connection.commit()
    reduce.coordinator.invoker_factory = DisclosureFinal
    generated = reduce.run()
    assert generated.status == "generation_complete"
    # Older reducer fixtures predate the authoritative download-attempt
    # lineage. Add the same structured chain production Stage 3 persists;
    # report_audit never reads free-form artifact provenance for authority.
    for paper_id in ("p1", "p2"):
        artifact_id = f"source-artifact-{paper_id}"
        exists = reduce.database.connection.execute(
            """SELECT 1 FROM download_attempts
               WHERE artifact_id = ? AND result_status = 'downloaded'""",
            (artifact_id,),
        ).fetchone()
        if exists is not None:
            continue
        request_id = f"fetch-audit-{paper_id}"
        reduce.database.connection.execute(
            """INSERT INTO fetch_requests(
                   request_id, candidate_id, policy_version, policy_hash, purpose,
                   provider, created_at, expires_at, idempotency_key, fencing_token, status
               ) VALUES (?, ?, ?, ?, 'personal_research', 'fixture',
                         '2026-08-10T00:00:00Z', '2026-08-11T00:00:00Z', ?, 1, 'consumed')""",
            (
                request_id,
                f"candidate-{paper_id}",
                reduce.coordinator.gate.policy.version,
                reduce.coordinator.gate.policy.hash,
                f"audit-fetch-{paper_id}",
            ),
        )
        reduce.database.connection.execute(
            """INSERT INTO download_attempts(
                   download_attempt_id, run_id, candidate_id, provider,
                   fetch_request_id, result_status, artifact_id
               ) VALUES (?, 'stage4-fixture', ?, 'fixture', ?, 'downloaded', ?)""",
            (
                f"download-audit-{paper_id}",
                f"candidate-{paper_id}",
                request_id,
                artifact_id,
            ),
        )
    reduce.database.connection.commit()
    final_hash = generated.final_output_hash
    assert final_hash is not None
    document = json.loads(reduce.store.read_bytes(final_hash))
    final_row = reduce.database.connection.execute(
        """SELECT dependency_ids_json FROM report_reduce_nodes
           WHERE report_run_id = 'report-1' AND call_kind = 'final_reduce'"""
    ).fetchone()
    dependency = json.loads(final_row["dependency_ids_json"])[0]
    synthesis_row = reduce.database.connection.execute(
        """SELECT output_hash FROM report_reduce_nodes
           WHERE report_run_id = 'report-1' AND node_id = ?""",
        (dependency,),
    ).fetchone()
    synthesis = json.loads(reduce.store.read_bytes(synthesis_row["output_hash"]))
    claims = synthesis["claims"]
    paper_claims = {"p1": set(), "p2": set()}
    consumed = {"p1": set(), "p2": set()}
    section_rows = reduce.database.connection.execute(
        """SELECT node_id, paper_ids_json, output_hash FROM report_reduce_nodes
           WHERE report_run_id = 'report-1' AND call_kind = 'section_reduce'"""
    ).fetchall()
    for row in section_rows:
        for paper_id in json.loads(row["paper_ids_json"]):
            consumed[paper_id].add(row["node_id"])
        output = json.loads(reduce.store.read_bytes(row["output_hash"]))
        for claim in output["claims"]:
            for field in ("supporting_evidence", "contradicting_evidence"):
                for reference in claim[field]:
                    if reference["kind"] == "paper_evidence":
                        paper_claims[reference["paper_id"]].add(claim["claim_id"])
    coverage = {
        "papers": [{
            "paper_id": paper_id,
            "evidence_claim_ids": sorted(paper_claims[paper_id]),
            "consumed_node_ids": sorted(consumed[paper_id]),
            "disposition": "evidence",
            "reason": None,
        } for paper_id in ("p1", "p2")],
        "missing_paper_ids": [],
        "uncovered_claim_ids": [],
        "complete": True,
    }
    bibliography = {
        paper_id: {
            "title": f"Paper {index}",
            "authors": ["Ada Lovelace"],
            "year": 2022 + index,
            "venue_name": f"Venue {index}",
            "doi": f"10.1000/{paper_id}",
        }
        for index, paper_id in enumerate(("p1", "p2"), 1)
    }
    bundle = ReportBundle(
        plan=reduce.plan,
        search_audit=reduce.audit,
        corpus_snapshot=reduce.corpus,
        claims=claims,
        comparison_groups={},
        claim_relations=(),
        document=document,
        coverage=coverage,
        bibliography=bibliography,
    )
    calls = []
    constructions = []
    holder = {}

    def factory():
        constructions.append(object())
        return holder["fake"]

    coordinator = ReportAuditCoordinator(
        reduce.database,
        reduce.store,
        reduce.coordinator.gate,
        ReportArtifactStore(tmp_path / "release"),
        invoker_factory=factory,
    )
    fake = FakeAuditSol(coordinator, calls)
    holder["fake"] = fake
    return AuditFixture(reduce, coordinator, fake, bundle, constructions, tmp_path / "release")


def test_passing_audit_publishes_once_and_resume_is_free(tmp_path: Path) -> None:
    fixture = _audit_fixture(tmp_path)
    try:
        first = fixture.run()
        (fixture.root / "reports/latest.md").unlink()
        second = fixture.run()

        assert first.status == "complete"
        assert first.audit_passes == ("A",)
        assert second.status == "complete"
        assert len(fixture.fake.calls) == 1
        assert len(fixture.constructions) == 1
        assert second.resumed_steps == ("audit_a",)
        assert (fixture.root / "reports/report-1/REPORT.md").is_file()
        assert (fixture.root / "reports/latest.md").is_file()
        step = fixture.reduce.database.connection.execute(
            """SELECT profile, model_id, reasoning_effort, actual_input_hash,
                      rendered_prompt_hash, actual_input_tokens, dispatch_count,
                      processing_facts_json
               FROM report_audit_steps WHERE step_name = 'audit_a'"""
        ).fetchone()
        assert (step["profile"], step["model_id"], step["reasoning_effort"]) == (
            PROFILE, MODEL, REASONING_EFFORT,
        )
        assert step["actual_input_hash"] and step["rendered_prompt_hash"]
        assert step["actual_input_tokens"] > 0 and step["dispatch_count"] == 1
        assert json.loads(step["processing_facts_json"])["execution_mode"] == "attended"
        assert fixture.reduce.database.connection.execute(
            "SELECT execution_mode FROM report_audit_runs WHERE report_run_id = 'report-1'"
        ).fetchone()[0] == "attended"
    finally:
        fixture.reduce.database.close()


@pytest.mark.parametrize(
    "search_fault, expected_field",
    (
        ("search_status", "search_status"),
        ("required_provider", "required_provider_failures"),
        ("budget_exhausted", "budget_exhausted"),
    ),
)
def test_incomplete_search_is_terminal_before_sol_and_resume_is_free(
    tmp_path: Path, monkeypatch, search_fault: str, expected_field: str
) -> None:
    import test_report_reduce

    original = test_report_reduce._raw_search_audit

    def incomplete_search_audit() -> dict:
        audit = original()
        if search_fault == "search_status":
            audit["status"] = "incomplete"
        elif search_fault == "required_provider":
            audit["sources"][0]["status"] = "failed"
        else:
            audit["rounds"][0]["stop_reason"] = "budget_exhausted"
        return audit

    monkeypatch.setattr(test_report_reduce, "_raw_search_audit", incomplete_search_audit)
    fixture = _audit_fixture(tmp_path)
    assert fixture.bundle.search_audit[expected_field]
    try:
        first = fixture.run()
        second = fixture.run()

        assert first.status == second.status == "incomplete"
        assert first.audit_passes == second.audit_passes == ()
        assert first.error == second.error
        assert "not publication-ready" in first.error
        assert fixture.fake.calls == []
        assert fixture.constructions == []
        assert not (fixture.root / "reports/latest.md").exists()
        assert not (fixture.root / "reports/report-1").exists()
        audit_run = fixture.reduce.database.connection.execute(
            """SELECT status, worst_case_calls, worst_case_input_tokens
               FROM report_audit_runs WHERE report_run_id = 'report-1'"""
        ).fetchone()
        assert tuple(audit_run) == ("incomplete", 1, 1)
        assert fixture.reduce.database.connection.execute(
            "SELECT COUNT(*) FROM report_audit_steps WHERE report_run_id = 'report-1'"
        ).fetchone()[0] == 0
        statuses = fixture.reduce.database.connection.execute(
            """SELECT rr.status, pr.status
               FROM report_runs rr JOIN pipeline_runs pr ON pr.run_id = rr.run_id
               WHERE rr.report_run_id = 'report-1'"""
        ).fetchone()
        assert tuple(statuses) == ("incomplete", "incomplete")
    finally:
        fixture.reduce.database.close()


def test_major_finding_gets_one_typed_repair_reverify_and_fresh_reaudit(tmp_path: Path) -> None:
    fixture = _audit_fixture(tmp_path)
    fixture.fake.severe_a = True
    before_hash = content_hash(fixture.bundle.document)
    try:
        result = fixture.run()
        resumed_result = fixture.run()

        assert result.status == "complete"
        assert result.audit_passes == ("A", "C")
        assert result.repair_count == 1
        assert result.report_document_hash != before_hash
        assert [item.call_kind for item in fixture.fake.calls] == [
            "quality_audit", "repair", "quality_audit",
        ]
        assert resumed_result.status == "complete"
        assert resumed_result.resumed_steps == ("audit_a", "audit_c", "repair")
        metadata = fixture.reduce.database.connection.execute(
            """SELECT step_name, invocation_metadata_json FROM report_audit_steps
               ORDER BY CASE step_name WHEN 'audit_a' THEN 1 WHEN 'repair' THEN 2 ELSE 3 END"""
        ).fetchall()
        invocation_ids = [json.loads(item["invocation_metadata_json"])["invocation_id"] for item in metadata]
        assert len(set(invocation_ids)) == 3
        assert json.loads((fixture.root / "reports/report-1/AUDIT.json").read_text())["audit_pass"] == "C"
    finally:
        fixture.reduce.database.close()


def test_failed_fresh_reaudit_is_incomplete_and_never_updates_latest(tmp_path: Path) -> None:
    fixture = _audit_fixture(tmp_path)
    fixture.fake.severe_a = True
    fixture.fake.severe_c = True
    try:
        result = fixture.run()

        assert result.status == "incomplete"
        assert result.audit_passes == ("A", "C")
        assert len(fixture.fake.calls) == 3
        assert not (fixture.root / "reports/latest.md").exists()
        assert not (fixture.root / "reports/report-1").exists()
        assert fixture.reduce.database.connection.execute(
            "SELECT status FROM report_runs WHERE report_run_id = 'report-1'"
        ).fetchone()[0] == "incomplete"
    finally:
        fixture.reduce.database.close()


def test_repair_must_use_a_fresh_invocation(tmp_path: Path) -> None:
    fixture = _audit_fixture(tmp_path)
    fixture.fake.severe_a = True
    fixture.fake.reuse_invocation = True
    try:
        result = fixture.run()

        assert result.status == "failed"
        assert [item.call_kind for item in fixture.fake.calls] == [
            "quality_audit", "repair",
        ]
        assert not (fixture.root / "reports/latest.md").exists()
    finally:
        fixture.reduce.database.close()


def test_restricted_provenance_reaches_no_audit_or_repair_invocation(tmp_path: Path) -> None:
    fixture = _audit_fixture(tmp_path)
    fixture.reduce.database.connection.execute(
        """UPDATE download_candidates SET license = NULL, access_basis = 'user_subscription'
           WHERE paper_id = 'p1'"""
    )
    fixture.reduce.database.connection.execute(
        """INSERT INTO download_candidates(
               candidate_id, paper_id, resolver, url, landing_url,
               publication_version, host, license, access_basis,
               retrieved_at, provenance_json
           ) VALUES ('decoy-open-p1', 'p1', 'decoy', 'https://decoy.test/p1.pdf',
                     NULL, 'published', 'decoy.test', 'CC-BY-4.0', 'open_license',
                     '2026-08-10T00:00:00Z', '{}')"""
    )
    fixture.reduce.database.connection.execute(
        """UPDATE artifacts SET provenance_json = ? WHERE artifact_id = 'source-artifact-p1'""",
        (json.dumps({"candidate_id": "decoy-open-p1", "access_basis": "open_license"}),),
    )
    fixture.reduce.database.connection.commit()
    try:
        result = fixture.run()

        assert result.status == "manual_required"
        assert fixture.fake.calls == []
        assert not (fixture.root / "reports/latest.md").exists()
        step = fixture.reduce.database.connection.execute(
            """SELECT status, dispatch_count, budget_calls_reserved
               FROM report_audit_steps WHERE step_name = 'audit_a'"""
        ).fetchone()
        assert tuple(step) == ("manual_required", 0, 0)
    finally:
        fixture.reduce.database.close()


def test_forged_reduce_invocation_metadata_reaches_no_audit_call(tmp_path: Path) -> None:
    fixture = _audit_fixture(tmp_path)
    row = fixture.reduce.database.connection.execute(
        """SELECT report_reduce_node_id, invocation_metadata_json
           FROM report_reduce_nodes
           WHERE report_run_id = 'report-1' AND call_kind = 'final_reduce'"""
    ).fetchone()
    metadata = json.loads(row["invocation_metadata_json"])
    metadata["actual_model"] = "gpt-5.6-luna"
    fixture.reduce.database.connection.execute(
        """UPDATE report_reduce_nodes SET invocation_metadata_json = ?
           WHERE report_reduce_node_id = ?""",
        (json.dumps(metadata, sort_keys=True, separators=(",", ":")), row["report_reduce_node_id"]),
    )
    fixture.reduce.database.connection.commit()
    try:
        with pytest.raises(ReportAuditError, match="invocation binding"):
            fixture.run()

        assert fixture.fake.calls == []
    finally:
        fixture.reduce.database.close()


def test_forged_bibliography_reaches_no_audit_call(tmp_path: Path) -> None:
    fixture = _audit_fixture(tmp_path)
    fixture.bundle.bibliography["p1"]["title"] = "Forged title"
    try:
        with pytest.raises(ReportAuditError, match="canonical metadata"):
            fixture.run()

        assert fixture.fake.calls == []
    finally:
        fixture.reduce.database.close()


def test_oversized_comparison_groups_reach_no_paid_audit(tmp_path: Path) -> None:
    fixture = _audit_fixture(tmp_path)
    fixture.bundle = replace(
        fixture.bundle,
        comparison_groups={"forged": {"blob": "x" * 2_000_000}},
    )
    try:
        with pytest.raises(ReportAuditError, match="deterministic evidence-derived"):
            fixture.run()
        assert fixture.fake.calls == []
        assert fixture.constructions == []
        assert fixture.reduce.database.connection.execute(
            "SELECT 1 FROM report_audit_runs WHERE report_run_id = 'report-1'"
        ).fetchone() is None
    finally:
        fixture.reduce.database.close()


def test_non_incremental_claim_relations_reach_no_paid_audit(tmp_path: Path) -> None:
    fixture = _audit_fixture(tmp_path)
    fixture.bundle = replace(
        fixture.bundle,
        claim_relations=({"garbage": "x" * 1_000},),
    )
    try:
        with pytest.raises(ReportAuditError, match="non-incremental"):
            fixture.run()
        assert fixture.fake.calls == []
        assert fixture.constructions == []
        assert fixture.reduce.database.connection.execute(
            "SELECT 1 FROM report_audit_runs WHERE report_run_id = 'report-1'"
        ).fetchone() is None
    finally:
        fixture.reduce.database.close()


def test_missing_canonical_reduce_node_reaches_no_paid_audit(tmp_path: Path) -> None:
    fixture = _audit_fixture(tmp_path)
    row = fixture.reduce.database.connection.execute(
        """SELECT report_reduce_node_id FROM report_reduce_nodes
           WHERE report_run_id = 'report-1' AND call_kind = 'section_reduce'
           ORDER BY node_id LIMIT 1"""
    ).fetchone()
    fixture.reduce.database.connection.execute(
        "DELETE FROM report_reduce_nodes WHERE report_reduce_node_id = ?",
        (row["report_reduce_node_id"],),
    )
    fixture.reduce.database.connection.commit()
    try:
        with pytest.raises(ReportAuditError, match="exactly match the canonical tree"):
            fixture.run()
        assert fixture.fake.calls == []
        assert fixture.constructions == []
    finally:
        fixture.reduce.database.close()


def test_orphaned_run_invocation_registry_reaches_no_paid_audit(tmp_path: Path) -> None:
    fixture = _audit_fixture(tmp_path)
    fixture.reduce.database.connection.execute(
        """INSERT INTO report_sol_invocations(
               report_run_id, invocation_id, phase, node_key, metadata_hash
           ) VALUES ('report-1', 'orphaned-invocation', 'audit_step', 'repair', ?)""",
        ("0" * 64,),
    )
    fixture.reduce.database.connection.commit()
    try:
        with pytest.raises(ReportAuditError, match="orphaned"):
            fixture.run()
        assert fixture.fake.calls == []
        assert fixture.constructions == []
    finally:
        fixture.reduce.database.close()


def test_persisted_section_omission_cannot_be_hidden_by_complete_caller_ledger(tmp_path: Path) -> None:
    fixture = _audit_fixture(tmp_path)
    rows = fixture.reduce.database.connection.execute(
        """SELECT report_reduce_node_id, output_artifact_id, output_hash
           FROM report_reduce_nodes
           WHERE report_run_id = 'report-1' AND call_kind = 'section_reduce'"""
    ).fetchall()
    for index, row in enumerate(rows):
        original_payload = fixture.reduce.store.read_bytes(row["output_hash"])
        output = json.loads(original_payload)
        output["claims"] = [
            claim for claim in output["claims"]
            if all(
                reference.get("paper_id") != "p2"
                for field in ("supporting_evidence", "contradicting_evidence")
                for reference in claim[field]
            )
        ]
        output["citation_paper_ids"] = [item for item in output["citation_paper_ids"] if item != "p2"]
        output["draft"] = output["draft"].replace(" Evidence from [@p2].", "").replace(
            "Evidence from [@p2].", ""
        )
        modified_payload = canonical_json(output)
        if modified_payload == original_payload:
            continue
        stored = fixture.reduce.store.put_bytes(
            modified_payload,
            mime_type="application/json",
            metadata={"kind": "stage4b_reduce_output"},
        )
        existing = fixture.reduce.database.connection.execute(
            "SELECT artifact_id FROM artifacts WHERE sha256 = ?",
            (stored.artifact_hash,),
        ).fetchone()
        artifact_id = existing["artifact_id"] if existing is not None else f"omitted-section-{index}"
        if existing is None:
            fixture.reduce.database.connection.execute(
                """INSERT INTO artifacts(
                       artifact_id, paper_id, artifact_kind, relative_path, mime_type,
                       byte_size, sha256, provenance_json
                   ) VALUES (?, NULL, 'report', ?, 'application/json', ?, ?, ?)""",
                (
                    artifact_id,
                    stored.relative_path,
                    stored.size_bytes,
                    stored.artifact_hash,
                    json.dumps({"stage": "stage4b", "content_hash": stored.artifact_hash}),
                ),
            )
        fixture.reduce.database.connection.execute(
            """UPDATE report_reduce_nodes SET output_artifact_id = ?, output_hash = ?
               WHERE report_reduce_node_id = ?""",
            (artifact_id, stored.artifact_hash, row["report_reduce_node_id"]),
        )
    fixture.reduce.database.connection.commit()
    try:
        with pytest.raises(ReportAuditError, match="drifted"):
            fixture.run()

        assert fixture.fake.calls == []
        assert not (fixture.root / "reports/latest.md").exists()
    finally:
        fixture.reduce.database.close()


def test_audit_rejects_forged_reduce_tree_token_estimates(tmp_path: Path) -> None:
    fixture = _audit_fixture(tmp_path)
    row = fixture.reduce.database.connection.execute(
        "SELECT aggregation_tree_json FROM report_runs WHERE report_run_id = 'report-1'"
    ).fetchone()
    tree = json.loads(row["aggregation_tree_json"])
    tree["budget"]["audit_input_tokens"] = 1
    tree["budget"]["repair_input_tokens"] = 1
    attempts = fixture.bundle.plan["budget"]["max_retries"] + 1
    tree["budget"]["worst_case_input_tokens"] = (
        tree["budget"]["generation_input_tokens"] * attempts + 2
    )
    prompt_bounds = {
        str(item["node_id"]): int(item["prompt_token_bound"])
        for item in fixture.reduce.database.connection.execute(
            """SELECT node_id, prompt_token_bound FROM report_reduce_nodes
               WHERE report_run_id = 'report-1'"""
        ).fetchall()
    }
    output_limits = {
        str(item["node_id"]): int(item["output_byte_limit"])
        for item in fixture.reduce.database.connection.execute(
            """SELECT node_id, output_byte_limit FROM report_reduce_nodes
               WHERE report_run_id = 'report-1'"""
        ).fetchall()
    }
    forged_input_hash = content_hash({
        "plan_hash": fixture.bundle.plan["plan_hash"],
        "corpus_snapshot_hash": fixture.bundle.corpus_snapshot["snapshot_hash"],
        "search_audit_pack_hash": fixture.bundle.search_audit["pack_hash"],
        "tree": tree,
        "prompt_token_bounds": prompt_bounds,
        "output_byte_limits": output_limits,
        "audit_repair_budget_bounds": tree["audit_repair_budget_bounds"],
    })
    fixture.reduce.database.connection.execute(
        "UPDATE report_runs SET aggregation_tree_json = ? WHERE report_run_id = 'report-1'",
        (json.dumps(tree, sort_keys=True, separators=(",", ":")),),
    )
    fixture.reduce.database.connection.execute(
        """UPDATE pipeline_runs SET input_hash = ?
           WHERE run_id = (SELECT run_id FROM report_runs WHERE report_run_id = 'report-1')""",
        (forged_input_hash,),
    )
    fixture.reduce.database.connection.commit()
    try:
        with pytest.raises(ReportAuditError, match="canonical approved tree"):
            fixture.run()
        assert fixture.fake.calls == []
        assert fixture.reduce.database.connection.execute(
            "SELECT 1 FROM report_audit_runs WHERE report_run_id = 'report-1'"
        ).fetchone() is None
    finally:
        fixture.reduce.database.close()


def test_audit_mode_must_match_the_frozen_reduce_execution_mode(tmp_path: Path) -> None:
    fixture = _audit_fixture(tmp_path)
    coordinator = ReportAuditCoordinator(
        fixture.reduce.database,
        fixture.reduce.store,
        fixture.reduce.coordinator.gate,
        ReportArtifactStore(fixture.root),
        invoker_factory=lambda: fixture.fake,
        execution_mode="unattended",
    )
    try:
        with pytest.raises(ReportAuditError, match="audit configuration"):
            coordinator.run(
                "report-1",
                fixture.bundle,
                now="2026-08-10T02:00:00Z",
            )

        assert fixture.fake.calls == []
    finally:
        fixture.reduce.database.close()


def test_bad_actual_model_is_terminal_and_resume_never_repays(tmp_path: Path) -> None:
    fixture = _audit_fixture(tmp_path)
    fixture.fake.wrong_actual_model = True
    try:
        first = fixture.run()
        second = fixture.run()

        assert first.status == second.status == "failed"
        assert len(fixture.fake.calls) == 1
        assert not (fixture.root / "reports/latest.md").exists()
        step = fixture.reduce.database.connection.execute(
            """SELECT status, dispatch_count, budget_calls_reserved
               FROM report_audit_steps WHERE step_name = 'audit_a'"""
        ).fetchone()
        assert step["status"] == "failed"
        assert step["dispatch_count"] == 1
        assert step["budget_calls_reserved"] == 2
    finally:
        fixture.reduce.database.close()


def test_repair_rejects_arbitrary_json_pointer_and_never_publishes(tmp_path: Path) -> None:
    fixture = _audit_fixture(tmp_path)
    fixture.fake.severe_a = True
    fixture.fake.arbitrary_repair = True
    try:
        result = fixture.run()

        assert result.status == "failed"
        assert [item.call_kind for item in fixture.fake.calls] == ["quality_audit", "repair"]
        assert not (fixture.root / "reports/latest.md").exists()
        assert fixture.reduce.database.connection.execute(
            "SELECT status FROM report_audit_steps WHERE step_name = 'repair'"
        ).fetchone()[0] == "failed"
    finally:
        fixture.reduce.database.close()
