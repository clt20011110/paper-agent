from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from uuid import UUID

import pytest

import paper_agent.report_audit as report_audit
from paper_agent.canonical import content_hash
from paper_agent.codex_exec import prompt_directory
from paper_agent.report_artifacts import audit_coverage_ledger
from paper_agent.report_audit import (
    AUDIT_HARD_CONTEXT_TOKENS,
    AUDIT_MAX_INPUT_TOKENS,
    AUDIT_OUTPUT_TOKEN_RESERVE,
    MAX_OUTPUT_BYTES,
    REPAIR_MAX_INPUT_TOKENS,
    REPAIR_OUTPUT_TOKEN_RESERVE,
    ReportAuditCoordinator,
    ReportAuditError,
    ReportAuditOutputError,
    _coverage_union,
    stage4b_audit_config_hash,
    stage4b_audit_repair_budget_bounds,
)
from paper_agent.report_config import ReportResources
from paper_agent.storage import Database
import test_report_reduce
from test_report_audit import _audit_fixture


def _claim(index: int, section_id: str, paper_id: str) -> dict:
    claim_id = str(UUID(int=index + 1))
    return {
        "claim_id": claim_id,
        "claim_key": {
            "subject_id": f"subject-{index}",
            "predicate_id": "supports",
            "object_or_scope_id": section_id,
            "qualifier_context_hash": content_hash([section_id, index]),
            "comparison_group_id": None,
        },
        "research_question_id": "rq1",
        "report_section": section_id,
        "claim_text": f"Claim {index}",
        "claim_type": "finding",
        "supporting_evidence": [{
            "kind": "paper_evidence",
            "paper_id": paper_id,
            "analysis_run_id": f"analysis-{paper_id}",
            "locator": f"page {index}",
            "evidence_unit": {"direction": "support", "text": f"evidence-{index}"},
        }],
        "contradicting_evidence": [],
        "evidence_level": "full_text_direct",
        "comparison_group_id": None,
    }


def _bundle() -> dict:
    sections = ("section-a", "section-b")
    papers = ("p1", "p2", "p3", "p4")
    claims = [
        _claim(index, sections[index // 2], paper_id)
        for index, paper_id in enumerate(papers)
    ]
    blocks = [
        {
            "block_id": f"block-{index}",
            "block_kind": "prose",
            "section_id": claim["report_section"],
            "text": (f"Substantive block {index}. " + "x" * 10_000),
            "claim_ids": [claim["claim_id"]],
            "citation_paper_ids": [papers[index]],
        }
        for index, claim in enumerate(claims)
    ]
    return {
        "plan": {
            "plan_hash": "a" * 64,
            "report_language": "zh-CN",
            "sections": [
                {"id": section_id, "title": section_id} for section_id in sections
            ],
        },
        "search_audit": {"limitations": ["database coverage ends at the frozen date"]},
        "corpus_snapshot": {
            "snapshot_hash": "b" * 64,
            "input_scope": {"full_pdf": 4},
            "papers": [
                {"paper_id": paper_id, "input_scope": "full_pdf"}
                for paper_id in papers
            ],
        },
        "claims": claims,
        "comparison_groups": {},
        "claim_relations": [],
        "document": {"report_run_id": "report-1", "blocks": blocks},
        "coverage": {
            "papers": [
                {
                    "paper_id": paper_id,
                    "evidence_claim_ids": [claims[index]["claim_id"]],
                    "consumed_node_ids": [f"node-{index}"],
                    "disposition": "evidence",
                    "reason": None,
                }
                for index, paper_id in enumerate(papers)
            ],
            "missing_paper_ids": [],
            "uncovered_claim_ids": [],
            "complete": True,
        },
        "bibliography": {
            paper_id: {"title": f"Paper {paper_id}"} for paper_id in papers
        },
    }


def _planner() -> ReportAuditCoordinator:
    coordinator = object.__new__(ReportAuditCoordinator)
    coordinator.prompt_root = prompt_directory()
    coordinator.resources = ReportResources.defaults()
    coordinator.rubric = {"version": 1, "rules": ["audit everything"]}
    coordinator.rubric_hash = content_hash(coordinator.rubric)
    return coordinator


def _fixture_audit_config_hash() -> str:
    policy = test_report_reduce.ArtifactProcessingPolicy.load(
        test_report_reduce.ROOT / "policies" / "artifact-processing-v1.yaml"
    )
    return stage4b_audit_config_hash(policy.hash)


def test_hard_context_is_frozen_with_an_explicit_output_reserve() -> None:
    assert AUDIT_HARD_CONTEXT_TOKENS == 1_179_648
    assert AUDIT_OUTPUT_TOKEN_RESERVE == 131_072
    assert AUDIT_MAX_INPUT_TOKENS == (
        AUDIT_HARD_CONTEXT_TOKENS - AUDIT_OUTPUT_TOKEN_RESERVE
    )
    assert REPAIR_OUTPUT_TOKEN_RESERVE == MAX_OUTPUT_BYTES
    assert REPAIR_MAX_INPUT_TOKENS + REPAIR_OUTPUT_TOKEN_RESERVE == (
        AUDIT_HARD_CONTEXT_TOKENS
    )
    assert stage4b_audit_config_hash("a" * 64) != stage4b_audit_config_hash(
        "b" * 64
    )
    assert stage4b_audit_config_hash(
        "a" * 64, execution_mode="attended"
    ) != stage4b_audit_config_hash("a" * 64, execution_mode="unattended")


def test_stable_section_claim_shards_cover_every_substantive_unit(monkeypatch) -> None:
    coordinator = _planner()
    bundle = _bundle()
    coverage = audit_coverage_ledger(bundle["document"], bundle["claims"])
    direct = coordinator._audit_payload(
        "report-1", "A", bundle, {"complete": True}, coverage
    )
    monkeypatch.setattr(report_audit, "AUDIT_MAX_INPUT_TOKENS", 28_000)

    first = coordinator._audit_pass_plan(
        "report-1", "A", bundle, {"complete": True}, coverage, direct
    )
    second = coordinator._audit_pass_plan(
        "report-1", "A", bundle, {"complete": True}, coverage, direct
    )

    assert first.direct_payload is None
    assert len(first.shards) > 1
    assert [item.node_id for item in first.shards] == [
        item.node_id for item in second.shards
    ]
    assert _coverage_union(coverage, [item.coverage for item in first.shards]) == coverage
    assert sum(len(item.coverage["block_ids"]) for item in first.shards) == 4
    assert sum(len(item.coverage["claim_ids"]) for item in first.shards) == 4
    for shard in first.shards:
        assert shard.payload["report_plan"] == bundle["plan"]
        assert shard.payload["search_limitations"] == [
            "database coverage ends at the frozen date",
            "抽取范围：full_pdf=4；全文、摘要和元数据证据已分层，缺失全文不作全文事实表述。",
        ]
        prompt = report_audit.canonical_json(shard.payload).decode("utf-8")
        rendered = coordinator._rendered_prompt("quality_audit", prompt)
        assert report_audit._token_upper_bound(rendered) <= 28_000


def test_audit_reduce_rejects_any_sampled_finding_or_coverage() -> None:
    coordinator = _planner()
    bundle = _bundle()
    coverage = audit_coverage_ledger(bundle["document"], bundle["claims"])
    first_coverage = report_audit._coverage_subset(
        coverage, ["block-0", "block-1"], coverage["claim_ids"][:2]
    )
    second_coverage = report_audit._coverage_subset(
        coverage, ["block-2", "block-3"], coverage["claim_ids"][2:]
    )
    finding = {
        "finding_id": "audit-shard-one:major-1",
        "severity": "major",
        "category": "coverage_gap",
        "description": "A required boundary is absent.",
        "block_ids": ["block-0"],
        "claim_ids": [coverage["claim_ids"][0]],
        "paper_ids": ["p1"],
        "recommendation": "Restore the boundary.",
    }

    def source(part, findings):
        return {
            "audit_pass": "A",
            "coverage_complete": True,
            "coverage_ledger": part,
            "findings": findings,
        }

    sources = [
        ("one", source(first_coverage, [finding])),
        ("two", source(second_coverage, [])),
    ]
    payload = coordinator._audit_reduce_payload(
        "report-1", "A", "root", bundle, coverage, sources
    )
    output = {
        "audit_pass": "A",
        "coverage_complete": True,
        "coverage_ledger": coverage,
        "findings": [finding],
    }
    coordinator._validate_audit_reduce(output, coverage, payload)

    sampled = deepcopy(output)
    sampled["findings"] = []
    with pytest.raises(ReportAuditOutputError, match="exact union"):
        coordinator._validate_audit_reduce(sampled, coverage, payload)

    missing_source = deepcopy(payload)
    missing_source["source_audits"] = missing_source["source_audits"][:1]
    missing_source["audit_scope"]["source_node_ids"] = ["one"]
    with pytest.raises(ReportAuditOutputError, match="source coverage"):
        coordinator._validate_audit_reduce(output, coverage, missing_source)


def test_shared_budget_rejects_an_unshardable_repair_before_any_call() -> None:
    with pytest.raises(ReportAuditError, match="repair prompt exceeds"):
        stage4b_audit_repair_budget_bounds(
            {"plan_hash": "a" * 64},
            {"snapshot_hash": "b" * 64, "papers": []},
            {"pack_hash": "c" * 64},
            final_output_byte_limit=262_144,
            synthesis_output_byte_limit=262_144,
        )


def test_shared_preflight_budget_includes_frozen_bibliography_bytes() -> None:
    plan = {
        "plan_hash": "a" * 64,
        "report_language": "zh-CN",
        "sections": [{"id": "evidence"}],
        "paper_memberships": [{"paper_id": "p1", "section_ids": ["evidence"]}],
    }
    base_corpus = {"snapshot_hash": "b" * 64, "papers": [{"paper_id": "p1"}]}
    large_corpus = deepcopy(base_corpus)
    large_corpus["papers"][0].update({
        "title": "t" * 10_000,
        "authors": ["a" * 10_000],
        "publication_year": 2026,
        "venue_name": "Venue",
        "doi": "10.1000/frozen",
    })
    search = {"pack_hash": "c" * 64}

    baseline = stage4b_audit_repair_budget_bounds(
        plan,
        base_corpus,
        search,
        final_output_byte_limit=65_536,
        synthesis_output_byte_limit=65_536,
    )
    expanded = stage4b_audit_repair_budget_bounds(
        plan,
        large_corpus,
        search,
        final_output_byte_limit=65_536,
        synthesis_output_byte_limit=65_536,
    )

    assert expanded.worst_case_input_tokens > baseline.worst_case_input_tokens


def test_shared_preflight_repeats_custom_rubric_in_every_audit_shard(
    tmp_path: Path,
) -> None:
    plan = {
        "plan_hash": "a" * 64,
        "report_language": "zh-CN",
        "sections": [{"id": "evidence"}],
        "paper_memberships": [{"paper_id": "p1", "section_ids": ["evidence"]}],
    }
    corpus = {"snapshot_hash": "b" * 64, "papers": [{"paper_id": "p1"}]}
    search = {"pack_hash": "c" * 64}
    rubric = tmp_path / "large-rubric.yaml"
    rubric.write_text(
        "version: 1\nproject_note: " + json.dumps("x" * 400_000) + "\n",
        encoding="utf-8",
    )

    baseline = stage4b_audit_repair_budget_bounds(
        plan,
        corpus,
        search,
        final_output_byte_limit=65_536,
        synthesis_output_byte_limit=65_536,
    )
    expanded = stage4b_audit_repair_budget_bounds(
        plan,
        corpus,
        search,
        final_output_byte_limit=65_536,
        synthesis_output_byte_limit=65_536,
        rubric_path=rubric,
    )

    assert expanded.audit_shards_per_pass > baseline.audit_shards_per_pass
    assert expanded.worst_case_calls > baseline.worst_case_calls


def test_migration_014_persists_recoverable_audit_shard_steps(tmp_path: Path) -> None:
    database = Database(tmp_path / "audit-shards.sqlite3")
    try:
        database.migrate()
        columns = {
            row["name"]
            for row in database.connection.execute(
                "PRAGMA table_info(report_audit_shard_steps)"
            )
        }
        assert {
            "audit_pass",
            "node_id",
            "node_kind",
            "source_node_ids_json",
            "expected_coverage_hash",
            "dispatch_count",
            "lease_token",
            "lease_expires_at",
            "budget_calls_reserved",
            "budget_tokens_reserved",
        }.issubset(columns)
    finally:
        database.close()


def test_sharded_sol_audit_is_independent_persisted_and_free_on_resume(
    tmp_path: Path, monkeypatch
) -> None:
    original_draft = test_report_reduce._draft

    def roomy_draft(_limit: int) -> dict:
        value = original_draft(250_000_000)
        value["budget"] = {
            **value["budget"],
            "max_sol_calls": 2_500,
            "max_input_tokens": 250_000_000,
        }
        value["stage4b_audit_config_hash"] = _fixture_audit_config_hash()
        return value

    monkeypatch.setattr(test_report_reduce, "_draft", roomy_draft)
    monkeypatch.setattr(report_audit, "AUDIT_MAX_INPUT_TOKENS", 63_000)
    fixture = _audit_fixture(tmp_path)
    clock_reads = []

    def clock():
        value = datetime(2026, 8, 10, 2, tzinfo=timezone.utc) + timedelta(
            seconds=len(clock_reads)
        )
        clock_reads.append(value)
        return value

    fixture.coordinator.clock = clock
    try:
        first = fixture.run()
        second = fixture.run()

        assert first.status == second.status == "complete"
        assert len(fixture.fake.calls) >= 3
        assert len(fixture.constructions) == len(fixture.fake.calls)
        assert len(fixture.fake.calls) == len({
            json.loads(row["invocation_metadata_json"])["invocation_id"]
            for row in fixture.reduce.database.connection.execute(
                """SELECT invocation_metadata_json FROM report_audit_shard_steps
                   WHERE report_run_id = 'report-1'
                   UNION ALL
                   SELECT invocation_metadata_json FROM report_audit_steps
                   WHERE report_run_id = 'report-1' AND step_name = 'audit_a'"""
            )
        })
        rows = fixture.reduce.database.connection.execute(
            """SELECT node_kind, status, dispatch_count, actual_input_tokens
               FROM report_audit_shard_steps
               WHERE report_run_id = 'report-1' ORDER BY ordinal"""
        ).fetchall()
        assert len(rows) >= 2
        assert all(row["status"] == "complete" for row in rows)
        assert all(row["dispatch_count"] == 1 for row in rows)
        assert all(row["actual_input_tokens"] <= 63_000 for row in rows)
        assert second.resumed_steps == ("audit_a",)
        assert len(clock_reads) >= 2 * len(fixture.fake.calls)
        assert clock_reads == sorted(set(clock_reads))
    finally:
        fixture.reduce.database.close()


def test_sharded_audit_invocation_cannot_be_reused_by_repair(
    tmp_path: Path, monkeypatch
) -> None:
    original_draft = test_report_reduce._draft

    def roomy_draft(_limit: int) -> dict:
        value = original_draft(250_000_000)
        value["budget"] = {
            **value["budget"],
            "max_sol_calls": 2_500,
            "max_input_tokens": 250_000_000,
        }
        value["stage4b_audit_config_hash"] = _fixture_audit_config_hash()
        return value

    monkeypatch.setattr(test_report_reduce, "_draft", roomy_draft)
    monkeypatch.setattr(report_audit, "AUDIT_MAX_INPUT_TOKENS", 63_000)
    fixture = _audit_fixture(tmp_path)
    base = fixture.fake

    class ReusingRepair:
        first_shard_invocation_id: str | None = None
        finding_injected = False

        def invoke(self, request):
            result = base.invoke(request)
            payload = json.loads(request.prompt)
            output = deepcopy(dict(result.output))
            metadata = result.metadata
            scope = payload.get("audit_scope", {})
            if request.call_kind == "quality_audit" and payload["audit_pass"] == "A":
                if scope.get("kind") == "stable_section_claim_shard":
                    if self.first_shard_invocation_id is None:
                        self.first_shard_invocation_id = metadata.invocation_id
                    if not self.finding_injected:
                        block = payload["report_document"]["blocks"][0]
                        output["findings"] = [{
                            "finding_id": f"{scope['node_id']}:major-1",
                            "severity": "major",
                            "category": "overgeneralization",
                            "description": "A frozen boundary is missing.",
                            "block_ids": [block["block_id"]],
                            "claim_ids": [block["claim_ids"][0]],
                            "paper_ids": [block["citation_paper_ids"][0]],
                            "recommendation": "Add the frozen boundary.",
                        }]
                        self.finding_injected = True
                elif scope.get("kind") == "exhaustive_audit_reduce":
                    output["findings"] = [
                        finding
                        for source in payload["source_audits"]
                        for finding in source["audit"]["findings"]
                    ]
            elif request.call_kind == "repair":
                assert self.first_shard_invocation_id is not None
                metadata = replace(
                    metadata,
                    invocation_id=self.first_shard_invocation_id,
                )
            metadata = replace(metadata, output_hash=content_hash(output))
            return report_audit.CodexExecResult(output, metadata)

    reusing = ReusingRepair()
    fixture.coordinator.invoker_factory = lambda: reusing
    try:
        result = fixture.run()

        assert result.status == "failed"
        repair = fixture.reduce.database.connection.execute(
            """SELECT status FROM report_audit_steps
               WHERE report_run_id = 'report-1' AND step_name = 'repair'"""
        ).fetchone()
        assert repair["status"] == "failed"
        assert fixture.reduce.database.connection.execute(
            """SELECT COUNT(*) FROM report_sol_invocations
               WHERE report_run_id = 'report-1' AND invocation_id = ?""",
            (reusing.first_shard_invocation_id,),
        ).fetchone()[0] == 1
        assert not (fixture.root / "reports/latest.md").exists()
    finally:
        fixture.reduce.database.close()


def test_uncertain_shard_dispatch_is_terminal_and_never_repaid(
    tmp_path: Path, monkeypatch
) -> None:
    original_draft = test_report_reduce._draft

    def roomy_draft(_limit: int) -> dict:
        value = original_draft(250_000_000)
        value["budget"] = {
            **value["budget"],
            "max_sol_calls": 2_500,
            "max_input_tokens": 250_000_000,
        }
        value["stage4b_audit_config_hash"] = _fixture_audit_config_hash()
        return value

    monkeypatch.setattr(test_report_reduce, "_draft", roomy_draft)
    monkeypatch.setattr(report_audit, "AUDIT_MAX_INPUT_TOKENS", 63_000)
    fixture = _audit_fixture(tmp_path)
    calls = []

    class UncertainSol:
        def invoke(self, request):
            calls.append(request)
            raise TimeoutError("connection ended after dispatch")

    fixture.coordinator.invoker_factory = UncertainSol
    try:
        first = fixture.run()
        second = fixture.run()

        assert first.status == second.status == "failed"
        assert len(calls) == 1
        row = fixture.reduce.database.connection.execute(
            """SELECT status, dispatch_count, budget_calls_reserved
               FROM report_audit_shard_steps
               WHERE report_run_id = 'report-1' ORDER BY ordinal LIMIT 1"""
        ).fetchone()
        assert tuple(row) == ("failed", 1, 2)
        assert not (fixture.root / "reports/latest.md").exists()
    finally:
        fixture.reduce.database.close()


def test_audit_rejects_reduce_artifact_metadata_drift_before_sol(
    tmp_path: Path, monkeypatch
) -> None:
    original_draft = test_report_reduce._draft

    def roomy_draft(limit: int) -> dict:
        value = original_draft(limit)
        value["budget"] = {**value["budget"], "max_sol_calls": 300}
        value["stage4b_audit_config_hash"] = _fixture_audit_config_hash()
        return value

    monkeypatch.setattr(test_report_reduce, "_draft", roomy_draft)
    fixture = _audit_fixture(tmp_path)
    final = fixture.reduce.database.connection.execute(
        """SELECT output_artifact_id FROM report_reduce_nodes
           WHERE report_run_id = 'report-1' AND call_kind = 'final_reduce'"""
    ).fetchone()
    fixture.reduce.database.connection.execute(
        "UPDATE artifacts SET mime_type = 'text/plain' WHERE artifact_id = ?",
        (final["output_artifact_id"],),
    )
    fixture.reduce.database.connection.commit()
    try:
        with pytest.raises(ReportAuditError, match="reduce output artifact"):
            fixture.run()
        assert fixture.fake.calls == []
    finally:
        fixture.reduce.database.close()


def test_audit_rejects_foreign_stage4_luna_invocation_before_sol(
    tmp_path: Path, monkeypatch
) -> None:
    original_draft = test_report_reduce._draft

    def roomy_draft(limit: int) -> dict:
        value = original_draft(limit)
        value["budget"] = {**value["budget"], "max_sol_calls": 300}
        value["stage4b_audit_config_hash"] = _fixture_audit_config_hash()
        return value

    monkeypatch.setattr(test_report_reduce, "_draft", roomy_draft)
    fixture = _audit_fixture(tmp_path)
    row = fixture.reduce.database.connection.execute(
        "SELECT analysis_run_id, invocation_metadata_json FROM analysis_runs LIMIT 1"
    ).fetchone()
    detail = json.loads(row["invocation_metadata_json"])
    detail["invocation"]["actual_model"] = "foreign-model"
    fixture.reduce.database.connection.execute(
        "UPDATE analysis_runs SET invocation_metadata_json = ? WHERE analysis_run_id = ?",
        (json.dumps(detail), row["analysis_run_id"]),
    )
    fixture.reduce.database.connection.commit()
    try:
        with pytest.raises(ReportAuditError, match="frozen Luna profile"):
            fixture.run()
        assert fixture.fake.calls == []
    finally:
        fixture.reduce.database.close()


def test_decoy_download_attempt_cannot_authorize_a_foreign_pdf_candidate(
    tmp_path: Path, monkeypatch
) -> None:
    original_draft = test_report_reduce._draft

    def roomy_draft(limit: int) -> dict:
        value = original_draft(limit)
        value["budget"] = {**value["budget"], "max_sol_calls": 300}
        value["stage4b_audit_config_hash"] = _fixture_audit_config_hash()
        return value

    monkeypatch.setattr(test_report_reduce, "_draft", roomy_draft)
    fixture = _audit_fixture(tmp_path)
    database = fixture.reduce.database.connection
    database.execute(
        "DELETE FROM download_attempts WHERE artifact_id = 'source-artifact-p1'"
    )
    database.execute(
        """UPDATE download_candidates SET license = NULL,
                  access_basis = 'user_subscription'
           WHERE candidate_id = 'candidate-p1'"""
    )
    database.execute(
        """INSERT INTO download_candidates(
               candidate_id, paper_id, resolver, url, landing_url,
               publication_version, host, license, access_basis,
               retrieved_at, provenance_json
           ) VALUES ('decoy-open-p1', 'p1', 'decoy', 'https://decoy.test/p1.pdf',
                     NULL, 'published', 'decoy.test', 'CC-BY-4.0', 'open_license',
                     '2026-08-10T00:00:00Z', '{}')"""
    )
    database.execute(
        """INSERT INTO fetch_requests(
               request_id, candidate_id, policy_version, policy_hash, purpose,
               provider, created_at, expires_at, idempotency_key, fencing_token, status
           ) VALUES ('fetch-decoy-p1', 'decoy-open-p1', ?, ?, 'personal_research',
                     'fixture', '2026-08-10T00:00:00Z', '2026-08-11T00:00:00Z',
                     'fetch-decoy-p1', 1, 'consumed')""",
        (fixture.reduce.coordinator.gate.policy.version,
         fixture.reduce.coordinator.gate.policy.hash),
    )
    database.execute(
        """INSERT INTO download_attempts(
               download_attempt_id, run_id, candidate_id, provider,
               fetch_request_id, result_status, artifact_id
           ) VALUES ('attempt-decoy-p1', 'stage4-fixture', 'decoy-open-p1',
                     'fixture', 'fetch-decoy-p1', 'downloaded', 'source-artifact-p1')"""
    )
    database.commit()
    try:
        result = fixture.run()
        assert result.status == "manual_required"
        assert fixture.fake.calls == []
    finally:
        fixture.reduce.database.close()


def test_audit_rejects_jointly_rewritten_stage4_invocation_bindings_before_sol(
    tmp_path: Path,
) -> None:
    fixture = _audit_fixture(tmp_path)
    row = fixture.reduce.database.connection.execute(
        "SELECT analysis_run_id, invocation_metadata_json FROM analysis_runs LIMIT 1"
    ).fetchone()
    detail = json.loads(row["invocation_metadata_json"])
    detail["invocation"]["input_hash"] = "0" * 64
    detail["invocation"]["rendered_prompt_hash"] = "1" * 64
    detail["invocation"]["invocation_id"] = "jointly-rewritten"
    fixture.reduce.database.connection.execute(
        """UPDATE analysis_runs SET input_hash = ?, invocation_metadata_json = ?
           WHERE analysis_run_id = ?""",
        ("0" * 64, json.dumps(detail), row["analysis_run_id"]),
    )
    fixture.reduce.database.connection.commit()
    try:
        with pytest.raises(ReportAuditError, match="frozen Luna profile"):
            fixture.run()

        assert fixture.fake.calls == []
    finally:
        fixture.reduce.database.close()
