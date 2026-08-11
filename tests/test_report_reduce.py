from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path

import pytest

from paper_agent.approval import approve
from paper_agent.artifacts import ArtifactStore
from paper_agent.canonical import canonical_json, content_hash
from paper_agent.codex_exec import CodexExecResult, InvocationMetadata
from paper_agent.grants import GrantStore
from paper_agent.processing import (
    ArtifactProcessingPolicy,
    PROCESSING_MODEL,
    PROCESSING_PROVIDER,
    ProcessingGate,
)
from paper_agent.report_plan import (
    CLASSIFICATION_AXES,
    REPORT_SECTION_IDS,
    CorpusPaper,
    build_corpus_snapshot,
    build_search_audit_pack,
    compile_report_plan,
    persist_approved_report_plan,
)
from paper_agent.report_artifacts import ReportArtifactStore
from paper_agent.report_audit import ReportAuditCoordinator, stage4b_audit_config_hash
from paper_agent.report_config import ReportResources
from paper_agent.report_invocations import report_invocation_metadata_hash
from paper_agent.report_reduce import (
    PROFILE,
    REASONING_EFFORT,
    FrozenDerivedArtifact,
    ReportReduceError,
    SolBudgetError,
    SolReduceCoordinator,
    stage4b_reduce_config_hash,
)
from paper_agent.reporting import (
    AnalysisRecord,
    ReportPlanner,
    SectionRule,
    corpus_evidence_allowlist,
    stable_claim_id,
)
from paper_agent.storage import Database


ROOT = Path(__file__).resolve().parents[1]
QUERY_PLAN_HASH = "a" * 64


def _unit() -> dict:
    return {
        "claim": "The method produced a measured result.",
        "direction": "support",
        "task_id": "task-a",
        "dataset_id": "dataset-a",
        "dataset_version": "v1",
        "split_id": "test",
        "metric_id": "accuracy",
        "metric_definition_hash": "b" * 64,
        "unit": "percent",
        "optimization_direction": "maximize",
        "value": 82.0,
        "uncertainty": "not reported",
        "statistical_method": "point estimate",
        "protocol_id": "protocol-a",
        "protocol_hash": "c" * 64,
        "sample_size": 100,
        "baseline_id": "baseline-a",
        "baseline_version": "v1",
        "conditions": ["frozen protocol"],
        "locator": {"kind": "page", "value": "4"},
        "normalization_method": "registry_exact",
        "normalizer_version": "registry-v1",
        "source_value": 82.0,
        "comparison_eligibility": "comparable",
        "missing_fields": [],
    }


def _analysis_document(
    paper_id: str,
    source_hash: str,
    *,
    prompt_hash: str,
    schema_hash: str,
    summary_size: int = 0,
) -> dict:
    return {
        "paper_id": paper_id,
        "artifact_hash": source_hash,
        "input_scope": "full_pdf",
        "model": PROCESSING_MODEL,
        "model_revision": "fixture-revision",
        "prompt_hash": prompt_hash,
        "schema_hash": schema_hash,
        "created_at": "2026-08-10T00:00:00Z",
        "research_question_and_motivation": "Test the frozen evidence.",
        "summary": "Measured result. " + ("x" * summary_size),
        "methods": ["method-a"],
        "key_techniques": ["technique-a"],
        "datasets": ["dataset-a"],
        "experimental_setup": ["frozen protocol"],
        "metrics": ["accuracy"],
        "results": ["82 percent"],
        "limitations": ["one protocol"],
        "credibility": "The result has a page locator.",
        "resources": [],
        "topic_relevance": "Directly relevant.",
        "labels": {
            "subquestion": ["rq1"],
            "theme": ["shared-theme"],
            "method_family": ["method-a"],
            "task": ["task-a"],
            "dataset": ["dataset-a"],
            "benchmark": ["benchmark-a"],
            "evidence_type": ["measurement"],
            "publication_status": "peer_reviewed",
            "study_setting": "real",
        },
        "label_evidence": [],
        "evidence_units": [_unit()],
        "comparison_eligibility": "comparable",
        "missing_fields": [],
    }


def _raw_search_audit() -> dict:
    return {
        "schema_version": "1",
        "crawl_run_id": "crawl-report-fixture",
        "run_id": "search-report-fixture",
        "search_plan_id": "search-plan-fixture",
        "plan_hash": QUERY_PLAN_HASH,
        "status": "complete",
        "incomplete_sources": [],
        "sources": [
            {"source_run_id": "source-fixture", "provider": "fixture", "status": "complete"}
        ],
        "queries": [{"query_id": "query-fixture", "provider": "fixture", "native_query": "test"}],
        "rounds": [{"round_index": 0, "stop_reason": "exhausted"}],
        "totals": {"sources": {"raw_discovered": 2, "unique_after_dedup": 2}},
    }


def _draft(max_input_tokens: int) -> dict:
    sections = [
        {
            "id": section_id,
            "title": f"中文章节：{section_id.replace('_', ' ')}",
            "subquestion_ids": ["rq1"],
            "target_words": 300,
            "evidence_requirements": ["Every claim has evidence"],
            "allowed_evidence_levels": ["full_text_direct", "corpus_stat"],
        }
        for section_id in REPORT_SECTION_IDS
    ]
    all_sections = list(REPORT_SECTION_IDS)
    return {
        "objective": "综合冻结证据，不增加论文。",
        "report_language": "zh-CN",
        "execution_strategy": "reduce_tree",
        "audience": "Researchers",
        "primary_question": "What does the frozen evidence support?",
        "subquestions": [{"id": "rq1", "question": "What does the evidence support?"}],
        "synthesis_question": "Under which frozen conditions do findings differ?",
        "scope": {
            "date_from": "2020-01-01",
            "date_to": "2026-08-10",
            "venues": ["fixture-venue"],
            "document_types": ["article"],
            "languages": ["en"],
            "inclusion_criteria": ["Included in the frozen corpus"],
            "exclusion_criteria": ["Outside the frozen corpus"],
        },
        "stage4b_config_hash": "f" * 64,
        "stage4b_audit_config_hash": "e" * 64,
        "aggregation": {
            "max_chunk_input_tokens": 500_000,
            "reduce_output_tokens": 5_000,
        },
        "sections": sections,
        "classification_axes": list(CLASSIFICATION_AXES),
        "cohort_rules": {
            "recent_cutoff": "2024-01-01",
            "foundational_rule": "Frozen seed",
            "peer_review_rule": "Canonical status",
            "study_setting_rule": "Frozen labels",
        },
        "paper_memberships": [
            {
                "paper_id": paper_id,
                "section_ids": all_sections,
                "primary_section_id": "evidence_synthesis",
                "coverage_disposition": "evidence",
                "coverage_reason": None,
                "resource_table_ids": [],
            }
            for paper_id in ("p1", "p2")
        ],
        "artifacts": {
            "comparison_tables": [],
            "trend_statistics": [],
            "resource_tables": [],
            "appendices": ["coverage ledger"],
        },
        "budget": {
            "max_sol_calls": 300,
            "max_input_tokens": max_input_tokens,
            "max_retries": 1,
            "audit_calls": 2,
            "repair_calls": 1,
        },
    }


def _claim(report_run_id: str, section_id: str, record: AnalysisRecord) -> dict:
    key = {
        "subject_id": record.paper_id,
        "predicate_id": "has_measured_result",
        "object_or_scope_id": section_id,
        "qualifier_context_hash": content_hash({"section": section_id}),
        "comparison_group_id": None,
    }
    reference = {
        "kind": "paper_evidence",
        "evidence_level": "full_text_direct",
        "paper_id": record.paper_id,
        "analysis_run_id": record.analysis_run_id,
        "evidence_unit": record.evidence_units[0],
        "locator": "page 4",
        "search_plan_id": None,
        "source_run_id": None,
        "query_id": None,
        "statistic": None,
        "calculation": None,
    }
    return {
        "claim_id": stable_claim_id(key, report_run_id=report_run_id),
        "claim_key": key,
        "research_question_id": "rq1",
        "report_section": section_id,
        "claim_text": f"论文 {record.paper_id} 具有测量结果。",
        "claim_type": "finding",
        "supporting_evidence": [reference],
        "contradicting_evidence": [],
        "evidence_level": "full_text_direct",
        "comparison_group_id": None,
        "confidence": "medium",
        "known_limitations": ["One frozen protocol"],
        "status": "supported",
        "mapping_status": "mapped",
    }


def _corpus_claim(
    report_run_id: str,
    section_id: str,
    corpus_evidence: dict,
    *,
    calculation: str | None = None,
) -> dict:
    statistic = next(
        item for item in corpus_evidence["statistics"]
        if item["statistic"] == "flow.included"
    )
    key = {
        "subject_id": "frozen-corpus",
        "predicate_id": "included_count",
        "object_or_scope_id": section_id,
        "qualifier_context_hash": content_hash({"statistic": "flow.included"}),
        "comparison_group_id": None,
    }
    reference = {
        "kind": "corpus_evidence",
        "evidence_level": "corpus_stat",
        "paper_id": None,
        "analysis_run_id": None,
        "evidence_unit": None,
        "locator": None,
        "search_plan_id": corpus_evidence["search_plan_ids"][0],
        "source_run_id": corpus_evidence["source_run_ids"][0],
        "query_id": corpus_evidence["query_ids"][0],
        "statistic": statistic["statistic"],
        "calculation": calculation or statistic["calculation"],
    }
    return {
        "claim_id": stable_claim_id(key, report_run_id=report_run_id),
        "claim_key": key,
        "research_question_id": "rq1",
        "report_section": section_id,
        "claim_text": "冻结检索流程纳入了两篇论文。",
        "claim_type": "corpus_stat",
        "supporting_evidence": [reference],
        "contradicting_evidence": [],
        "evidence_level": "corpus_stat",
        "comparison_group_id": None,
        "confidence": "high",
        "known_limitations": ["This count applies only to the frozen search audit."],
        "status": "supported",
        "mapping_status": "mapped",
    }


@dataclass
class FakeSol:
    coordinator: SolReduceCoordinator
    records: dict[str, AnalysisRecord]
    calls: list
    fail_once: bool = False
    missing_actual_once: bool = False
    wrong_model_once: bool = False
    omit_final_markers: bool = False
    omit_final_conflicts: bool = False
    duplicate_section_marker: bool = False
    add_conflict: bool = False
    oversized_once: bool = False
    misplace_final_claim: bool = False
    omit_paper: str | None = None
    reuse_invocation_id: str | None = None
    add_corpus_stat: bool = False
    corpus_calculation: str | None = None
    wrong_output_hash_once: bool = False
    missing_output_hash_once: bool = False

    def invoke(self, request):
        self.calls.append(request)
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError("temporary Sol transport failure")
        payload = json.loads(request.prompt)
        node = payload["node"]
        if request.call_kind == "section_reduce":
            section_id = node["section_ids"][0]
            paper_ids = [
                paper_id for paper_id in node["paper_ids"] if paper_id != self.omit_paper
            ]
            conflicts = ["CONFLICT-P1-P2"] if self.add_conflict and section_id == REPORT_SECTION_IDS[0] else []
            draft = " ".join(f"Evidence from [@{paper_id}]." for paper_id in paper_ids)
            if self.duplicate_section_marker and paper_ids:
                draft += f" Duplicate [@{paper_ids[0]}]."
            if self.oversized_once:
                self.oversized_once = False
                draft += "x" * 300_000
            claims = [
                _claim(payload["report_run_id"], section_id, self.records[paper_id])
                for paper_id in paper_ids
            ]
            if self.add_corpus_stat and section_id == REPORT_SECTION_IDS[0]:
                claims.append(_corpus_claim(
                    payload["report_run_id"],
                    section_id,
                    payload["corpus_evidence"],
                    calculation=self.corpus_calculation,
                ))
            output = {
                "section_id": section_id,
                "draft": draft,
                "claims": claims,
                "citation_paper_ids": paper_ids,
                "unresolved_conflicts": conflicts,
            }
        elif request.call_kind == "cross_section_reduce":
            documents = [item["document"] for item in payload["inputs"]]
            citations = sorted(
                {paper for item in documents for paper in item["citation_paper_ids"]}
            )
            output = {
                "section_ids": sorted({section for item in documents for section in (item.get("section_ids") or [item["section_id"]])}),
                "draft": "Cross-section evidence " + " ".join(f"[@{paper}]" for paper in citations),
                "claims": [claim for item in documents for claim in item["claims"]],
                "citation_paper_ids": citations,
                "unresolved_conflicts": sorted({conflict for item in documents for conflict in item["unresolved_conflicts"]}),
            }
        else:
            synthesis = payload["inputs"][0]["document"]
            blocks = []
            for index, claim in enumerate(synthesis["claims"], start=1):
                papers = sorted({
                    ref["paper_id"]
                    for ref in claim["supporting_evidence"] + claim["contradicting_evidence"]
                    if ref["kind"] == "paper_evidence"
                })
                markers = "" if self.omit_final_markers else " " + " ".join(f"[@{paper}]" for paper in papers)
                blocks.append({
                    "block_id": f"block-{index}",
                    "block_kind": "prose",
                    "section_id": claim["report_section"],
                    "text": claim["claim_text"] + markers,
                    "claim_ids": [claim["claim_id"]],
                    "citation_paper_ids": papers,
                })
            if synthesis["unresolved_conflicts"] and not self.omit_final_conflicts:
                blocks[0]["text"] += " " + " ".join(synthesis["unresolved_conflicts"])
            if self.misplace_final_claim:
                blocks[0]["section_id"] = REPORT_SECTION_IDS[1]
            output = {"report_run_id": payload["report_run_id"], "blocks": blocks}

        model = "gpt-5.6-luna" if self.wrong_model_once else "gpt-5.6-sol"
        self.wrong_model_once = False
        actual_model = None if self.missing_actual_once else model
        actual_profile = None if self.missing_actual_once else PROFILE
        self.missing_actual_once = False
        rendered = self.coordinator._rendered_prompt(request.call_kind, request.prompt)
        metadata = InvocationMetadata(
            invocation_id=self.reuse_invocation_id or f"invocation-{len(self.calls)}",
            profile=PROFILE,
            model=model,
            reasoning_effort=REASONING_EFFORT,
            schema_name=request.schema_name,
            schema_hash=self.coordinator.schema_hashes[request.call_kind],
            input_hash=request.input_hash,
            prompt_name=request.prompt_name,
            prompt_hash=self.coordinator.prompt_hashes[request.call_kind],
            rendered_prompt_hash=sha256(rendered.encode()).hexdigest(),
            call_kind=request.call_kind,
            attempts=1,
            actual_model=actual_model,
            actual_profile=actual_profile,
            schema_path=request.schema_path,
            prompt_path=request.prompt_path,
            output_hash=(
                None
                if self.missing_output_hash_once
                else "0" * 64
                if self.wrong_output_hash_once
                else content_hash(output)
            ),
        )
        self.wrong_output_hash_once = False
        self.missing_output_hash_once = False
        return CodexExecResult(output, metadata)


@dataclass
class Fixture:
    database: Database
    store: ArtifactStore
    coordinator: SolReduceCoordinator
    plan: dict
    corpus: dict
    audit: dict
    reduce_plan: object
    records: tuple[AnalysisRecord, ...]
    artifacts: tuple[FrozenDerivedArtifact, ...]
    calls: list
    constructions: list
    fake: FakeSol
    grants: GrantStore
    clock: list[datetime]

    def run(self, report_run_id: str = "report-1", pipeline_run_id: str = "pipeline-report-1", **kwargs):
        supplied_now = kwargs.pop("now", None)
        if supplied_now is not None:
            self.clock[0] = datetime.fromisoformat(
                str(supplied_now).replace("Z", "+00:00")
            ).astimezone(timezone.utc)
        return self.coordinator.run(
            report_run_id,
            pipeline_run_id,
            self.plan,
            self.reduce_plan,
            self.artifacts,
            corpus_snapshot=self.corpus,
            search_audit_pack=self.audit,
            **kwargs,
        )


def _fixture(
    tmp_path: Path,
    *,
    max_input_tokens: int = 50_000_000,
    summary_size: int = 0,
    normalized_text: bool = False,
    execution_mode: str = "attended",
    resources: ReportResources | None = None,
    rubric_path: Path | None = None,
    coverage_dispositions: dict[str, tuple[str, str | None, tuple[str, ...]]] | None = None,
) -> Fixture:
    database = Database(tmp_path / "papers.sqlite")
    database.migrate()
    store = ArtifactStore(tmp_path / "store")
    policy = ArtifactProcessingPolicy.load(ROOT / "policies" / "artifact-processing-v1.yaml")
    grants = GrantStore(database)
    gate = ProcessingGate(policy, grants)
    records: list[AnalysisRecord] = []
    artifacts: list[FrozenDerivedArtifact] = []
    corpus_papers: list[CorpusPaper] = []
    stage4_schema_hash = sha256(
        canonical_json(json.loads((ROOT / "schemas" / "paper-analysis.schema.json").read_text()))
    ).hexdigest()
    stage4_prompt_hash = sha256((ROOT / "prompts" / "paper-analysis.md").read_bytes()).hexdigest()

    database.connection.execute(
        """INSERT INTO pipeline_runs(
               run_id, stage, status, input_hash, config_hash, implementation_version, started_at
           ) VALUES ('stage4-fixture', 'stage4', 'complete', ?, ?, 'fixture', CURRENT_TIMESTAMP)""",
        ("1" * 64, "2" * 64),
    )
    database.connection.execute(
        """INSERT INTO pipeline_runs(
               run_id, stage, status, input_hash, config_hash, implementation_version, started_at
           ) VALUES ('download-fixture', 'stage3', 'complete', ?, ?, 'fixture', CURRENT_TIMESTAMP)""",
        ("3" * 64, "4" * 64),
    )
    for index, paper_id in enumerate(("p1", "p2"), start=1):
        database.connection.execute(
            """INSERT INTO papers(
                   paper_id, title, authors_json, publication_date, year,
                   venue_id, venue_name, doi, verification_status
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'verified')""",
            (
                paper_id,
                f"Paper {index}",
                json.dumps(["Ada Lovelace"]),
                f"202{index + 2}-01-01",
                2022 + index,
                f"venue-{paper_id}",
                f"Venue {index}",
                f"10.1000/{paper_id}",
            ),
        )
        source_payload = f"%PDF-1.4 fixture source {paper_id}".encode()
        source_stored = store.put_bytes(source_payload, mime_type="application/pdf")
        candidate_id = f"candidate-{paper_id}"
        database.connection.execute(
            """INSERT INTO download_candidates(
                   candidate_id, paper_id, resolver, url, landing_url, publication_version,
                   host, license, access_basis, retrieved_at, provenance_json
               ) VALUES (?, ?, 'fixture', ?, NULL, 'published', 'example.test',
                         'CC-BY-4.0', 'open_license', '2026-08-10T00:00:00Z', '{}')""",
            (candidate_id, paper_id, f"https://example.test/{paper_id}.pdf"),
        )
        source_artifact_id = f"source-artifact-{paper_id}"
        database.connection.execute(
            """INSERT INTO artifacts(
                   artifact_id, paper_id, artifact_kind, relative_path, mime_type,
                   byte_size, sha256, provenance_json
               ) VALUES (?, ?, 'pdf', ?, 'application/pdf', ?, ?, ?)""",
            (
                source_artifact_id,
                paper_id,
                source_stored.relative_path,
                source_stored.size_bytes,
                source_stored.artifact_hash,
                json.dumps({"candidate_id": candidate_id, "access_basis": "open_license"}),
            ),
        )
        request_id = f"fetch-{paper_id}"
        database.connection.execute(
            """INSERT INTO fetch_requests(
                   request_id, candidate_id, policy_version, policy_hash, purpose, provider,
                   created_at, expires_at, idempotency_key, fencing_token, status
               ) VALUES (?, ?, 'fixture-v1', ?, 'research_download', 'fixture',
                         '2026-08-10T00:00:00Z', '2026-08-11T00:00:00Z', ?, 1, 'consumed')""",
            (request_id, candidate_id, "5" * 64, f"idempotency-{paper_id}"),
        )
        database.connection.execute(
            """INSERT INTO download_attempts(
                   download_attempt_id, run_id, candidate_id, provider, fetch_request_id,
                   result_status, http_status, artifact_id
               ) VALUES (?, 'download-fixture', ?, 'fixture', ?, 'downloaded', 200, ?)""",
            (f"attempt-{paper_id}", candidate_id, request_id, source_artifact_id),
        )
        selected_artifact_id = source_artifact_id
        selected_hash = source_stored.artifact_hash
        selected_category = "full_text"
        if normalized_text:
            text_stored = store.put_bytes(
                f"normalized full text for {paper_id}".encode(),
                mime_type="text/plain; charset=utf-8",
            )
            selected_artifact_id = f"text-artifact-{paper_id}"
            selected_hash = text_stored.artifact_hash
            selected_category = "normalized_text"
            database.connection.execute(
                """INSERT INTO artifacts(
                       artifact_id, paper_id, artifact_kind, relative_path, mime_type,
                       byte_size, sha256, provenance_json
                   ) VALUES (?, ?, 'text', ?, 'text/plain; charset=utf-8', ?, ?, ?)""",
                (
                    selected_artifact_id,
                    paper_id,
                    text_stored.relative_path,
                    text_stored.size_bytes,
                    text_stored.artifact_hash,
                    json.dumps({"source_artifact_id": "decoy-not-authoritative"}),
                ),
            )
            database.connection.execute(
                """INSERT INTO text_extractions(
                       extraction_id, paper_id, source_artifact_id, source_sha256,
                       output_artifact_id, extractor_name, extractor_version,
                       page_count, character_count, text_coverage, printable_ratio, status
                   ) VALUES (?, ?, ?, ?, ?, 'fixture', 'v1', 1, ?, 1.0, 1.0,
                             'full_text_ready')""",
                (
                    f"extraction-{paper_id}",
                    paper_id,
                    source_artifact_id,
                    source_stored.artifact_hash,
                    selected_artifact_id,
                    text_stored.size_bytes,
                ),
            )
        document = _analysis_document(
            paper_id,
            selected_hash,
            prompt_hash=stage4_prompt_hash,
            schema_hash=stage4_schema_hash,
            summary_size=summary_size,
        )
        analysis_payload = canonical_json(document)
        analysis_stored = store.put_bytes(
            analysis_payload, mime_type="application/json", metadata={"kind": "analysis"}
        )
        analysis_run_id = f"analysis-{paper_id}"
        output_artifact_id = f"analysis-artifact-{paper_id}"
        database.connection.execute(
            """INSERT INTO artifacts(
                   artifact_id, paper_id, artifact_kind, relative_path, mime_type,
                   byte_size, sha256, provenance_json
               ) VALUES (?, ?, 'analysis', ?, 'application/json', ?, ?, ?)""",
            (
                output_artifact_id,
                paper_id,
                analysis_stored.relative_path,
                analysis_stored.size_bytes,
                analysis_stored.artifact_hash,
                json.dumps({"analysis_run_id": analysis_run_id, "stage": "stage4", "format": "json"}),
            ),
        )
        decision = {
            "policy_version": policy.version,
            "policy_hash": policy.hash,
            "outcome": "full_pdf",
            "reason_code": "policy_rule_0",
            "input_artifact_hash": selected_hash,
            "provider": PROCESSING_PROVIDER,
            "model": PROCESSING_MODEL,
            "purpose": "internal_analysis",
            "data_category": selected_category,
            "processing_grant_id": None,
            "authorized_by": "policy",
        }
        stage4_input_hash = sha256(f"sent-{paper_id}".encode()).hexdigest()
        stage4_rendered_hash = sha256(f"rendered-{paper_id}".encode()).hexdigest()
        input_policy_facts = {
            "paper_id": paper_id,
            "artifact_hash": selected_hash,
            "artifact": "normalized_text" if normalized_text else "pdf",
            "input_scope": "full_pdf",
            "license": "CC-BY-4.0",
            "access_basis": "open_license",
            "domain": "example.test",
            "mode": "attended",
            "collection_id": None,
            "collection_snapshot_hash": None,
            "selection_snapshot_hash": None,
            "data_category": selected_category,
        }
        invocation = {
            "invocation_id": f"stage4-invocation-{paper_id}",
            "profile": "stage4_analysis_luna",
            "model": PROCESSING_MODEL,
            "reasoning_effort": "medium",
            "schema_name": "paper-analysis.schema.json",
            "schema_hash": stage4_schema_hash,
            "input_hash": stage4_input_hash,
            "prompt_name": "paper-analysis.md",
            "prompt_hash": stage4_prompt_hash,
            "rendered_prompt_hash": stage4_rendered_hash,
            "call_kind": None,
            "attempts": 1,
            "actual_model": PROCESSING_MODEL,
            "actual_profile": "stage4_analysis_luna",
        }
        database.connection.execute(
            """INSERT INTO analysis_runs(
                   analysis_run_id, run_id, paper_id, artifact_id, input_hash, input_scope,
                   model_id, model_revision, prompt_hash, schema_hash, implementation_version,
                   authorization_grant_id, policy_version, policy_decision,
                   invocation_metadata_json, status, output_artifact_id, completed_at
               ) VALUES (?, 'stage4-fixture', ?, ?, ?, 'full_pdf', ?, 'fixture-revision',
                         ?, ?, 'fixture', NULL, ?, 'full_pdf', ?, 'complete', ?, CURRENT_TIMESTAMP)""",
            (
                analysis_run_id,
                paper_id,
                selected_artifact_id,
                stage4_input_hash,
                PROCESSING_MODEL,
                document["prompt_hash"],
                document["schema_hash"],
                policy.version,
                json.dumps({
                    "report_input_tokens": len(analysis_payload),
                    "input_policy_facts": input_policy_facts,
                    "processing_decision": decision,
                    "invocation": invocation,
                }),
                output_artifact_id,
            ),
        )
        record = AnalysisRecord(
            paper_id=paper_id,
            analysis_run_id=analysis_run_id,
            analysis_hash=analysis_stored.artifact_hash,
            input_scope="full_pdf",
            input_tokens=len(analysis_payload),
            classifications={
                "subquestion": ("rq1",),
                "theme": ("shared-theme",),
                "method_family": ("method-a",),
                "task": ("task-a",),
                "dataset": ("dataset-a",),
                "benchmark": ("benchmark-a",),
                "time": (str(2022 + index),),
                "venue": (f"venue-{paper_id}",),
                "publication_status": ("peer_reviewed",),
                "evidence_type": ("measurement",),
                "study_setting": ("real",),
            },
            evidence_units=(_unit(),),
        )
        records.append(record)
        # These policy fields are intentionally untrusted.  The coordinator
        # must replace them from the persisted candidate/source provenance.
        artifacts.append(FrozenDerivedArtifact(
            artifact_hash=analysis_stored.artifact_hash,
            payload=analysis_payload,
            artifact_kind="analysis",
            input_scope="full_pdf",
            license=None,
            access_basis="unknown",
            lineage_hash=selected_hash,
            source_lineage_hashes=(selected_hash,),
            paper_id=paper_id,
        ))
        corpus_papers.append(CorpusPaper(
            paper_id,
            analysis_run_id,
            analysis_stored.artifact_hash,
            (selected_hash,),
            "user_library" if paper_id == "p1" else "newly_discovered",
            "peer_reviewed",
            "real",
            "full_pdf",
            "full_text_direct",
            paper_id == "p1",
            paper_id == "p2",
            analysis_input_tokens=len(analysis_payload),
            analysis_pipeline_input_hash="1" * 64,
            analysis_config_hash="2" * 64,
            analysis_implementation_version="fixture",
            analysis_prompt_input_hash=stage4_input_hash,
            analysis_rendered_prompt_hash=stage4_rendered_hash,
            analysis_invocation_id=f"stage4-invocation-{paper_id}",
            analysis_policy_facts_hash=content_hash(input_policy_facts),
            publication_date=f"202{index + 2}-01-01",
            publication_year=2022 + index,
            venue_id=f"venue-{paper_id}",
            venue_name=f"Venue {index}",
            title=f"Paper {index}",
            authors=("Ada Lovelace",),
            doi=f"10.1000/{paper_id}",
            canonical_url=None,
            verification_status="verified",
        ))
    database.connection.commit()

    raw_audit = _raw_search_audit()
    corpus = build_corpus_snapshot(
        tuple(corpus_papers),
        query_plan_hash=QUERY_PLAN_HASH,
        search_audit=raw_audit,
        created_at="2026-08-10T00:01:00Z",
    )
    audit = build_search_audit_pack(
        raw_audit,
        corpus,
        screening_flow={
            "raw_discovered": 2,
            "unique_after_dedup": 2,
            "stage2_screened": 2,
            "included": 2,
        },
        exclusion_reasons={},
        required_providers=("fixture",),
        created_at="2026-08-10T00:02:00Z",
    )
    draft_document = _draft(max_input_tokens)
    if coverage_dispositions:
        resource_tables = {
            table_id
            for _, _, table_ids in coverage_dispositions.values()
            for table_id in table_ids
        }
        draft_document["artifacts"]["resource_tables"] = sorted(resource_tables)
        for membership in draft_document["paper_memberships"]:
            configured = coverage_dispositions.get(str(membership["paper_id"]))
            if configured is None:
                continue
            disposition, reason, table_ids = configured
            membership["coverage_disposition"] = disposition
            membership["coverage_reason"] = reason
            membership["resource_table_ids"] = list(table_ids)
    draft_document["stage4b_config_hash"] = stage4b_reduce_config_hash(
        policy.hash,
        execution_mode=execution_mode,
        resources=resources,
    )
    draft_document["stage4b_audit_config_hash"] = stage4b_audit_config_hash(
        policy.hash,
        execution_mode=execution_mode,
        resources=resources,
        rubric_path=rubric_path,
    )
    draft = compile_report_plan(
        draft_document,
        corpus_snapshot=corpus,
        search_audit_pack=audit,
        created_at="2026-08-10T00:03:00Z",
        resources=resources,
        _legacy_read_only=True,
    )
    plan = approve(
        draft,
        draft["plan_hash"],
        approved_by="owner",
        approved_at="2026-08-10T00:04:00Z",
        hash_field="plan_hash",
    )
    persist_approved_report_plan(database, plan)
    sections = tuple(
        SectionRule(
            str(section["id"]),
            frozenset(str(item) for item in section["subquestion_ids"]),
            frozenset(str(item) for item in section["allowed_evidence_levels"]),
        )
        for section in plan["sections"]
    )
    memberships = {
        str(item["paper_id"]): tuple(str(section) for section in item["section_ids"])
        for item in plan["paper_memberships"]
    }
    calls: list = []
    constructions: list = []
    holder: dict[str, object] = {}
    clock = [datetime(2026, 8, 10, 1, tzinfo=timezone.utc)]

    def factory():
        constructions.append(object())
        return holder["fake"]

    coordinator = SolReduceCoordinator(
        database,
        store,
        gate,
        records,
        sections,
        memberships,
        invoker_factory=factory,
        execution_mode=execution_mode,
        resources=resources,
        rubric_path=rubric_path,
        clock=lambda: clock[0],
    )
    reduce_plan = ReportPlanner(
        plan,
        records,
        max_chunk_input_tokens=500_000,
        reduce_output_tokens=5_000,
        audit_input_tokens=1_000,
        repair_input_tokens=1_000,
    ).build()
    fake = FakeSol(coordinator, {item.paper_id: item for item in records}, calls)
    holder["fake"] = fake
    return Fixture(
        database,
        store,
        coordinator,
        plan,
        corpus,
        audit,
        reduce_plan,
        tuple(records),
        tuple(artifacts),
        calls,
        constructions,
        fake,
        grants,
        clock,
    )


def _restrict_candidate(fixture: Fixture, paper_id: str = "p1") -> None:
    fixture.database.connection.execute(
        "UPDATE download_candidates SET license = NULL, access_basis = 'user_subscription' WHERE paper_id = ?",
        (paper_id,),
    )
    fixture.database.connection.execute(
        "UPDATE artifacts SET provenance_json = ? WHERE artifact_id = ?",
        (json.dumps({"candidate_id": f"candidate-{paper_id}", "access_basis": "user_subscription"}), f"source-artifact-{paper_id}"),
    )
    fixture.database.connection.commit()


def _sol_grant(
    fixture: Fixture,
    artifact: FrozenDerivedArtifact,
    grant_id: str,
    *,
    expires_at: str = "2026-08-11T00:00:00Z",
) -> str:
    lineage_hash = content_hash(tuple(sorted(artifact.source_lineage_hashes)))
    draft = fixture.grants.create_draft(
        grant_id=grant_id,
        kind="remote_model_processing",
        actions=["remote_model_processing"],
        purpose="research_synthesis",
        mode="attended",
        scope={
            "paper_ids": [artifact.paper_id],
            "artifact_hashes": [artifact.artifact_hash],
            "collection_ids": [],
            "collection_snapshot_hash": None,
            "selection_snapshot_hash": None,
            "domains": [],
            "provider": "codex_cli",
            "model": "gpt-5.6-sol",
            "data_categories": ["analysis"],
        },
        max_papers=1,
        expires_at=expires_at,
        lineage_hash=lineage_hash,
    )
    fixture.grants.approve(
        draft,
        draft["content_hash"],
        approved_by="owner",
        approved_at="2026-08-10T00:00:00Z",
    )
    return grant_id


def test_sol_tree_completes_and_resume_revalidates_every_node(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    try:
        first = fixture.run()
        second = fixture.run()

        assert first.status == "generation_complete"
        assert second.status == "generation_complete"
        assert len(fixture.calls) == len(fixture.reduce_plan.nodes)
        assert len(fixture.constructions) == len(fixture.reduce_plan.nodes)
        assert all(request.profile == PROFILE for request in fixture.calls)
        assert all(result.resumed for result in second.nodes)
        assert first.final_output_hash == second.final_output_hash
        rows = fixture.database.connection.execute(
            """SELECT status, profile, model_id, reasoning_effort, actual_input_tokens,
                      rendered_prompt_hash, prompt_token_bound, output_byte_limit,
                      budget_calls_reserved, budget_tokens_reserved, output_hash
               FROM report_reduce_nodes ORDER BY node_id"""
        ).fetchall()
        assert len(rows) == len(fixture.reduce_plan.nodes)
        assert {row["status"] for row in rows} == {"complete"}
        assert {(row["profile"], row["model_id"], row["reasoning_effort"]) for row in rows} == {
            ("stage4b_summary_sol", "gpt-5.6-sol", "high")
        }
        assert all(0 < row["actual_input_tokens"] <= row["prompt_token_bound"] for row in rows)
        assert all(row["rendered_prompt_hash"] and row["output_hash"] for row in rows)
        assert sum(row["budget_calls_reserved"] for row in rows) == len(rows) * 2
        assert fixture.database.connection.execute(
            "SELECT status FROM report_runs WHERE report_run_id = 'report-1'"
        ).fetchone()[0] == "running"
    finally:
        fixture.database.close()


def test_reduce_accepts_only_frozen_corpus_stat_and_supplies_its_inputs(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    fixture.fake.add_corpus_stat = True
    try:
        result = fixture.run()

        assert result.status == "generation_complete"
        expected = corpus_evidence_allowlist(fixture.audit).document()
        section_payloads = [
            json.loads(request.prompt)
            for request in fixture.calls
            if request.call_kind == "section_reduce"
        ]
        assert section_payloads
        assert all(payload["corpus_evidence"] == expected for payload in section_payloads)
    finally:
        fixture.database.close()


def test_reduce_rejects_corpus_stat_not_recomputed_from_frozen_audit(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    fixture.fake.add_corpus_stat = True
    fixture.fake.corpus_calculation = "999"
    try:
        result = fixture.run()

        assert result.status == "failed"
        assert any(
            "frozen search audit" in (node.error or "")
            for node in result.nodes
        )
    finally:
        fixture.database.close()


def test_reduce_plan_must_be_one_complete_binary_tree(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    try:
        dependencies = tuple(
            node for node in fixture.reduce_plan.nodes if node.call_kind == "section_reduce"
        )
        root = fixture.reduce_plan.nodes[-1]
        malformed_root = replace(
            root,
            section_ids=tuple(
                dict.fromkeys(
                    section for node in dependencies for section in node.section_ids
                )
            ),
            paper_ids=tuple(
                sorted({paper for node in dependencies for paper in node.paper_ids})
            ),
            dependency_ids=tuple(node.node_id for node in dependencies),
            planned_input_hash=content_hash({
                "call_kind": root.call_kind,
                "dependencies": [
                    {
                        "node_id": node.node_id,
                        "planned_input_hash": node.planned_input_hash,
                    }
                    for node in dependencies
                ],
            }),
        )
        malformed = replace(
            fixture.reduce_plan,
            nodes=(*fixture.reduce_plan.nodes[:-1], malformed_root),
        )

        with pytest.raises(ReportReduceError, match="one cross-section root"):
            fixture.coordinator._verify_tree_bindings(fixture.plan, malformed)
    finally:
        fixture.database.close()


def test_reduce_plan_order_must_match_the_approved_canonical_tree(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture.reduce_plan = replace(
        fixture.reduce_plan,
        chunks=tuple(reversed(fixture.reduce_plan.chunks)),
    )
    try:
        with pytest.raises(ReportReduceError, match="canonical"):
            fixture.run("report-reordered", "pipeline-reordered")

        assert fixture.calls == []
    finally:
        fixture.database.close()


def test_audit_config_drift_is_rejected_before_any_generation(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture.coordinator.audit_config_hash = "0" * 64
    try:
        with pytest.raises(ReportReduceError, match="audit configuration"):
            fixture.run("report-audit-config", "pipeline-audit-config")

        assert fixture.calls == []
    finally:
        fixture.database.close()


def test_reduce_tree_cannot_omit_a_chunk_that_remains_in_the_frozen_plan(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    try:
        split_plan = deepcopy(fixture.plan)
        split_plan["budget"] = {**split_plan["budget"], "max_sol_calls": 200}
        full = ReportPlanner(
            split_plan,
            fixture.records,
            max_chunk_input_tokens=500_000,
            reduce_output_tokens=5_000,
            audit_input_tokens=1_000,
            repair_input_tokens=1_000,
        ).build()
        p1_plan = deepcopy(split_plan)
        p1_plan["paper_memberships"] = [
            item for item in p1_plan["paper_memberships"] if item["paper_id"] == "p1"
        ]
        p1_tree = ReportPlanner(
            p1_plan,
            fixture.records[:1],
            max_chunk_input_tokens=500_000,
            reduce_output_tokens=5_000,
            audit_input_tokens=1_000,
            repair_input_tokens=1_000,
        ).build()
        malformed = replace(p1_tree, chunks=full.chunks)

        with pytest.raises(ReportReduceError, match="every semantic chunk"):
            fixture.coordinator._verify_tree_bindings(fixture.plan, malformed)
    finally:
        fixture.database.close()


def test_completed_report_status_is_not_reopened_by_reduce_resume(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    try:
        fixture.run()
        fixture.database.connection.execute(
            "UPDATE report_runs SET status = 'complete', completed_at = '2026-08-10T02:00:00Z'"
        )
        fixture.database.connection.execute(
            "UPDATE pipeline_runs SET status = 'complete', completed_at = '2026-08-10T02:00:00Z' "
            "WHERE run_id = 'pipeline-report-1'"
        )
        fixture.database.connection.commit()

        resumed = fixture.run()

        assert resumed.status == "generation_complete"
        assert fixture.database.connection.execute(
            "SELECT status FROM report_runs WHERE report_run_id = 'report-1'"
        ).fetchone()[0] == "complete"
        assert fixture.database.connection.execute(
            "SELECT status FROM pipeline_runs WHERE run_id = 'pipeline-report-1'"
        ).fetchone()[0] == "complete"
    finally:
        fixture.database.close()


def test_reduce_requires_a_separately_persisted_approved_plan(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture.database.connection.execute(
        "DELETE FROM report_plans WHERE report_plan_id = ?", (fixture.plan["plan_id"],)
    )
    fixture.database.connection.commit()
    try:
        with pytest.raises(ReportReduceError, match="must be persisted"):
            fixture.run("report-no-plan", "pipeline-no-plan")

        assert fixture.calls == []
        assert fixture.database.connection.execute(
            "SELECT COUNT(*) FROM pipeline_runs WHERE run_id = 'pipeline-no-plan'"
        ).fetchone()[0] == 0
    finally:
        fixture.database.close()


def test_caller_cannot_relabel_restricted_stage4_provenance_as_open(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _restrict_candidate(fixture)
    fixture.database.connection.execute(
        """INSERT INTO download_candidates(
               candidate_id, paper_id, resolver, url, publication_version, host,
               license, access_basis, retrieved_at, provenance_json
           ) VALUES ('decoy-open-p1', 'p1', 'decoy', 'https://decoy.test/p1.pdf',
                     'published', 'decoy.test', 'CC-BY-4.0', 'open_license',
                     '2026-08-10T00:00:00Z', '{}')"""
    )
    fixture.database.connection.execute(
        "UPDATE artifacts SET provenance_json = ? WHERE artifact_id = 'source-artifact-p1'",
        (json.dumps({"candidate_id": "decoy-open-p1", "access_basis": "open_license"}),),
    )
    fixture.database.connection.commit()
    # The supplied object already claims unknown; replacing it with an open
    # label must make no difference because the coordinator reloads DB facts.
    fixture.artifacts = (
        replace(fixture.artifacts[0], license="CC-BY-4.0", access_basis="open_license"),
        fixture.artifacts[1],
    )
    try:
        result = fixture.run("report-denied", "pipeline-denied")

        assert result.status == "incomplete"
        assert fixture.calls
        assert {
            tuple(json.loads(call.prompt)["node"]["paper_ids"])
            for call in fixture.calls
        } == {("p2",)}
        rows = fixture.database.connection.execute(
            """SELECT status, paper_ids_json, budget_calls_reserved
               FROM report_reduce_nodes WHERE report_run_id = 'report-denied'"""
        ).fetchall()
        assert "manual_required" in {row["status"] for row in rows}
        assert all(
            row["budget_calls_reserved"] == 0
            for row in rows
            if row["status"] == "manual_required"
            and json.loads(row["paper_ids_json"]) == ["p1"]
        )
    finally:
        fixture.database.close()


@pytest.mark.parametrize("fault", ["missing", "profile", "input_hash"])
def test_stage4_luna_invocation_metadata_is_required_and_exact(
    tmp_path: Path, fault: str
) -> None:
    fixture = _fixture(tmp_path)
    row = fixture.database.connection.execute(
        "SELECT invocation_metadata_json FROM analysis_runs WHERE analysis_run_id = 'analysis-p1'"
    ).fetchone()
    detail = json.loads(row["invocation_metadata_json"])
    if fault == "missing":
        detail.pop("invocation")
    elif fault == "profile":
        detail["invocation"]["actual_profile"] = "stage4b_summary_sol"
    else:
        detail["invocation"]["input_hash"] = "0" * 64
    fixture.database.connection.execute(
        "UPDATE analysis_runs SET invocation_metadata_json = ? WHERE analysis_run_id = 'analysis-p1'",
        (json.dumps(detail),),
    )
    fixture.database.connection.commit()
    try:
        with pytest.raises(ReportReduceError, match="invocation metadata"):
            fixture.run(f"report-stage4-{fault}", f"pipeline-stage4-{fault}")

        assert fixture.calls == []
        assert fixture.database.connection.execute(
            "SELECT COUNT(*) FROM report_runs"
        ).fetchone()[0] == 0
    finally:
        fixture.database.close()


def test_analysis_record_cannot_omit_persisted_classification_axes(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    records = tuple(
        replace(record, classifications={"theme": ("shared-theme",)})
        for record in fixture.records
    )
    coordinator = SolReduceCoordinator(
        fixture.database,
        fixture.store,
        fixture.coordinator.gate,
        records,
        fixture.coordinator.sections,
        fixture.coordinator.memberships,
        invoker_factory=lambda: fixture.fake,
        clock=lambda: fixture.clock[0],
    )
    fixture.coordinator = coordinator
    fixture.fake.coordinator = coordinator
    fixture.records = records
    fixture.reduce_plan = ReportPlanner(
        fixture.plan,
        records,
        max_chunk_input_tokens=500_000,
        reduce_output_tokens=5_000,
        audit_input_tokens=1_000,
        repair_input_tokens=1_000,
    ).build()
    try:
        with pytest.raises(ReportReduceError, match="trusted frozen labels"):
            fixture.run("report-classification-drift", "pipeline-classification-drift")

        assert fixture.calls == []
    finally:
        fixture.database.close()


def test_normalized_text_uses_the_exact_extraction_and_download_chain(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, normalized_text=True)
    try:
        result = fixture.run("report-normalized-text", "pipeline-normalized-text")

        assert result.status == "generation_complete"
        assert len(fixture.calls) == len(fixture.reduce_plan.nodes)
    finally:
        fixture.database.close()


def test_normalized_text_rejects_source_pdf_hash_drift_before_sol(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, normalized_text=True)
    fixture.database.connection.execute(
        "UPDATE text_extractions SET source_sha256 = ? WHERE paper_id = 'p1'",
        ("0" * 64,),
    )
    fixture.database.connection.commit()
    try:
        with pytest.raises(ReportReduceError, match="source PDF binding has drifted"):
            fixture.run("report-normalized-drift", "pipeline-normalized-drift")

        assert fixture.calls == []
    finally:
        fixture.database.close()


def test_exact_leaf_grant_does_not_authorize_derived_reduce_outputs(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _restrict_candidate(fixture)
    grant_id = _sol_grant(fixture, fixture.artifacts[0], "leaf-sol-grant")
    try:
        result = fixture.run(
            "report-granted-leaf",
            "pipeline-granted-leaf",
            processing_grants={fixture.artifacts[0].artifact_hash: grant_id},
            now="2026-08-10T01:00:00Z",
        )

        assert result.status == "incomplete"
        expected_leaf_calls = sum(
            not node.dependency_ids
            for node in fixture.reduce_plan.nodes_for("section_reduce")
        )
        assert len(fixture.calls) == expected_leaf_calls
        parent_rows = fixture.database.connection.execute(
            """SELECT status, input_artifact_hashes_json FROM report_reduce_nodes
               WHERE report_run_id = 'report-granted-leaf'
                 AND dependency_ids_json <> '[]'"""
        ).fetchall()
        assert "manual_required" in {row["status"] for row in parent_rows}
        assert all(
            fixture.artifacts[0].artifact_hash not in json.loads(row["input_artifact_hashes_json"])
            for row in parent_rows if row["status"] == "manual_required"
        )
    finally:
        fixture.database.close()


def test_unattended_mode_is_preserved_across_all_derived_outputs(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, execution_mode="unattended")
    try:
        result = fixture.run("report-unattended-open", "pipeline-unattended-open")

        assert result.status == "generation_complete"
        rows = fixture.database.connection.execute(
            """SELECT output_policy_json FROM report_reduce_nodes
               WHERE report_run_id = 'report-unattended-open'"""
        ).fetchall()
        assert rows
        assert {json.loads(row["output_policy_json"])["mode"] for row in rows} == {
            "unattended"
        }
    finally:
        fixture.database.close()


def test_attended_grant_cannot_authorize_unattended_reduce(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, execution_mode="unattended")
    _restrict_candidate(fixture)
    grant_id = _sol_grant(fixture, fixture.artifacts[0], "attended-only-sol-grant")
    try:
        result = fixture.run(
            "report-unattended-denied",
            "pipeline-unattended-denied",
            processing_grants={fixture.artifacts[0].artifact_hash: grant_id},
            now="2026-08-10T01:00:00Z",
        )

        assert result.status == "incomplete"
        assert fixture.calls
        assert {
            tuple(json.loads(call.prompt)["node"]["paper_ids"])
            for call in fixture.calls
        } == {("p2",)}
        assert "manual_required" in {item.status for item in result.nodes}
    finally:
        fixture.database.close()


def test_each_node_rechecks_grant_expiry_with_a_fresh_clock_value(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _restrict_candidate(fixture)
    grant_id = _sol_grant(
        fixture,
        fixture.artifacts[0],
        "short-lived-sol-grant",
        expires_at="2026-08-10T01:00:02Z",
    )
    clock_calls: list[datetime] = []

    def ticking_clock() -> datetime:
        value = datetime(2026, 8, 10, 1, tzinfo=timezone.utc) + timedelta(
            seconds=len(clock_calls)
        )
        clock_calls.append(value)
        return value

    fixture.coordinator.clock = ticking_clock
    try:
        result = fixture.run(
            "report-expiring-grant",
            "pipeline-expiring-grant",
            processing_grants={fixture.artifacts[0].artifact_hash: grant_id},
        )

        assert result.status == "incomplete"
        call_papers = [tuple(json.loads(call.prompt)["node"]["paper_ids"]) for call in fixture.calls]
        assert call_papers.count(("p1",)) == 1
        assert call_papers.count(("p2",)) == len(REPORT_SECTION_IDS)
        assert len(clock_calls) >= 3
        assert "manual_required" in {item.status for item in result.nodes}
    finally:
        fixture.database.close()


@pytest.mark.parametrize("metadata_fault", ["wrong", "missing"])
def test_actual_sol_metadata_must_be_present_and_exact(tmp_path: Path, metadata_fault: str) -> None:
    fixture = _fixture(tmp_path)
    if metadata_fault == "wrong":
        fixture.fake.wrong_model_once = True
    else:
        fixture.fake.missing_actual_once = True
    try:
        result = fixture.run(f"report-{metadata_fault}", f"pipeline-{metadata_fault}")

        assert result.status == "failed"
        failed = fixture.database.connection.execute(
            """SELECT status, output_artifact_id, error_json FROM report_reduce_nodes
               WHERE report_run_id = ? AND status = 'failed' LIMIT 1""",
            (f"report-{metadata_fault}",),
        ).fetchone()
        assert failed["output_artifact_id"] is None
        assert "frozen node binding" in failed["error_json"]
    finally:
        fixture.database.close()


@pytest.mark.parametrize("fault", ("missing", "wrong"))
def test_actual_sol_output_hash_must_match_the_result(
    tmp_path: Path, fault: str
) -> None:
    fixture = _fixture(tmp_path)
    fixture.fake.wrong_output_hash_once = fault == "wrong"
    fixture.fake.missing_output_hash_once = fault == "missing"
    try:
        result = fixture.run("report-output-hash", "pipeline-output-hash")

        assert result.status == "failed"
        assert "output hash" in fixture.database.connection.execute(
            """SELECT error_json FROM report_reduce_nodes
               WHERE report_run_id = 'report-output-hash' AND status = 'failed'"""
        ).fetchone()[0]
    finally:
        fixture.database.close()


def test_legacy_completed_node_without_metadata_output_hash_still_replays(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    try:
        first = fixture.run("report-legacy-hash", "pipeline-legacy-hash")
        calls = len(fixture.calls)
        row = fixture.database.connection.execute(
            """SELECT report_reduce_node_id, node_id, invocation_id,
                      invocation_metadata_json
               FROM report_reduce_nodes
               WHERE report_run_id = 'report-legacy-hash'
               ORDER BY node_id LIMIT 1"""
        ).fetchone()
        metadata = json.loads(row["invocation_metadata_json"])
        metadata.pop("output_hash")
        fixture.database.connection.execute(
            """UPDATE report_reduce_nodes SET invocation_metadata_json = ?
               WHERE report_reduce_node_id = ?""",
            (json.dumps(metadata), row["report_reduce_node_id"]),
        )
        fixture.database.connection.execute(
            """UPDATE report_sol_invocations SET metadata_hash = ?
               WHERE report_run_id = 'report-legacy-hash' AND phase = 'reduce'
                 AND node_key = ? AND invocation_id = ?""",
            (
                report_invocation_metadata_hash(metadata),
                row["node_id"],
                row["invocation_id"],
            ),
        )
        fixture.database.connection.commit()

        resumed = fixture.run("report-legacy-hash", "pipeline-legacy-hash")

        assert first.status == resumed.status == "generation_complete"
        assert len(fixture.calls) == calls
    finally:
        fixture.database.close()


def test_bundle_and_persisted_stage4_drift_stop_before_dispatch(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    changed_corpus = deepcopy(fixture.corpus)
    changed_corpus["papers"][0]["publication_status"] = "preprint"
    try:
        with pytest.raises(ReportReduceError, match="hash has drifted"):
            fixture.coordinator.run(
                "report-corpus-drift",
                "pipeline-corpus-drift",
                fixture.plan,
                fixture.reduce_plan,
                fixture.artifacts,
                corpus_snapshot=changed_corpus,
                search_audit_pack=fixture.audit,
            )
        fixture.database.connection.execute(
            "UPDATE analysis_runs SET status = 'failed' WHERE analysis_run_id = 'analysis-p1'"
        )
        fixture.database.connection.commit()
        with pytest.raises(ReportReduceError, match="output binding has drifted"):
            fixture.run("report-stage4-drift", "pipeline-stage4-drift")

        assert fixture.calls == []
        assert fixture.database.connection.execute("SELECT COUNT(*) FROM report_runs").fetchone()[0] == 0
    finally:
        fixture.database.close()


def test_rendered_prompt_preflight_catches_cost_hidden_by_declared_input_tokens(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, max_input_tokens=10_000_000, summary_size=100_000)
    try:
        with pytest.raises(SolBudgetError, match="rendered Sol prompt upper bound"):
            fixture.run("report-budget", "pipeline-budget")

        assert fixture.calls == []
        report = fixture.database.connection.execute(
            "SELECT status FROM report_runs WHERE report_run_id = 'report-budget'"
        ).fetchone()
        pipeline = fixture.database.connection.execute(
            "SELECT status FROM pipeline_runs WHERE run_id = 'pipeline-budget'"
        ).fetchone()
        nodes = fixture.database.connection.execute(
            """SELECT status, dispatch_count, budget_calls_reserved
               FROM report_reduce_nodes WHERE report_run_id = 'report-budget'"""
        ).fetchall()
        assert report["status"] == pipeline["status"] == "incomplete"
        assert nodes
        assert {tuple(row) for row in nodes} == {("pending", 0, 0)}
    finally:
        fixture.database.close()


def test_caller_audit_token_estimates_cannot_reduce_the_shared_reserve(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    budget = replace(
        fixture.reduce_plan.budget,
        audit_input_tokens=1,
        repair_input_tokens=1,
        worst_case_input_tokens=1,
    )
    fixture.reduce_plan = replace(fixture.reduce_plan, budget=budget)
    try:
        result = fixture.run("report-trusted-budget", "pipeline-trusted-budget")

        assert result.status == "generation_complete"
        tree = json.loads(
            fixture.database.connection.execute(
                """SELECT aggregation_tree_json FROM report_runs
                   WHERE report_run_id = 'report-trusted-budget'"""
            ).fetchone()[0]
        )
        assert tree["audit_repair_budget_bounds"]["worst_case_input_tokens"] > 2
        assert tree["audit_repair_budget_bounds"]["worst_case_calls"] >= 6
        assert tree["budget"]["audit_input_tokens"] == (
            tree["audit_repair_budget_bounds"]["audit_a_input_tokens"]
            + tree["audit_repair_budget_bounds"]["audit_c_input_tokens"]
        )
        assert tree["budget"]["repair_input_tokens"] == tree[
            "audit_repair_budget_bounds"
        ]["repair_input_tokens"]
    finally:
        fixture.database.close()


def test_unshardable_repair_bound_stops_before_reduce_calls(
    tmp_path: Path, monkeypatch
) -> None:
    import paper_agent.report_reduce as report_reduce

    monkeypatch.setattr(report_reduce, "OUTPUT_BYTES_PER_ESTIMATED_TOKEN", 100)
    fixture = _fixture(tmp_path, max_input_tokens=200_000_000)
    try:
        with pytest.raises(SolBudgetError, match="repair prompt exceeds"):
            fixture.run("report-repair-context", "pipeline-repair-context")
        assert fixture.calls == []
        assert fixture.database.connection.execute(
            "SELECT 1 FROM report_runs WHERE report_run_id = 'report-repair-context'"
        ).fetchone() is None
    finally:
        fixture.database.close()


def test_output_byte_limit_is_a_hard_node_failure(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture.fake.oversized_once = True
    try:
        result = fixture.run("report-oversized", "pipeline-oversized")

        assert result.status == "failed"
        assert len(fixture.calls) >= 1
        assert "byte limit" in next(item.error for item in result.nodes if item.status == "failed")
    finally:
        fixture.database.close()


def test_duplicate_citation_marker_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture.fake.duplicate_section_marker = True
    try:
        result = fixture.run("report-duplicate-marker", "pipeline-duplicate-marker")

        assert result.status == "failed"
        failed = next(item for item in result.nodes if item.status == "failed")
        assert failed.node_id.startswith("section:")
        assert "citation binding" in failed.error
    finally:
        fixture.database.close()


@pytest.mark.parametrize("fault", ["markers", "conflicts"])
def test_final_reduce_cannot_erase_citation_markers_or_conflicts(tmp_path: Path, fault: str) -> None:
    fixture = _fixture(tmp_path)
    if fault == "markers":
        fixture.fake.omit_final_markers = True
    else:
        fixture.fake.add_conflict = True
        fixture.fake.omit_final_conflicts = True
    try:
        result = fixture.run(f"report-final-{fault}", f"pipeline-final-{fault}")

        assert result.status == "failed"
        final = next(item for item in result.nodes if item.node_id.startswith("final_reduce:"))
        assert final.status == "failed"
        assert "citation binding" in final.error if fault == "markers" else "conflict" in final.error
    finally:
        fixture.database.close()


def test_final_block_claims_cannot_move_between_frozen_sections(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture.fake.misplace_final_claim = True
    try:
        result = fixture.run("report-final-section", "pipeline-final-section")

        final = next(item for item in result.nodes if item.node_id.startswith("final_reduce:"))
        assert final.status == "failed"
        assert "different frozen section" in final.error
    finally:
        fixture.database.close()


def test_final_report_cannot_silently_omit_a_selected_paper(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture.fake.omit_paper = "p2"
    try:
        result = fixture.run("report-paper-coverage", "pipeline-paper-coverage")

        final = next(item for item in result.nodes if item.node_id.startswith("final_reduce:"))
        assert final.status == "failed"
        assert "frozen evidence dispositions" in final.error
    finally:
        fixture.database.close()


@pytest.mark.parametrize(
    ("disposition", "reason", "table_ids"),
    [
        ("resource_or_background_table", None, ("resource-inventory",)),
        ("background_only", "Context only; no extractable evidence.", ()),
    ],
)
def test_frozen_non_evidence_dispositions_are_complete_without_claim_fabrication(
    tmp_path: Path,
    disposition: str,
    reason: str | None,
    table_ids: tuple[str, ...],
) -> None:
    fixture = _fixture(
        tmp_path,
        coverage_dispositions={"p2": (disposition, reason, table_ids)},
    )
    fixture.fake.omit_paper = "p2"
    try:
        result = fixture.run("report-non-evidence", "pipeline-non-evidence")

        assert result.status == "generation_complete"
        final_row = fixture.database.connection.execute(
            """SELECT dependency_ids_json FROM report_reduce_nodes
               WHERE report_run_id = 'report-non-evidence'
                 AND call_kind = 'final_reduce'"""
        ).fetchone()
        synthesis_id = json.loads(final_row["dependency_ids_json"])[0]
        synthesis_hash = fixture.database.connection.execute(
            """SELECT output_hash FROM report_reduce_nodes
               WHERE report_run_id = 'report-non-evidence' AND node_id = ?""",
            (synthesis_id,),
        ).fetchone()[0]
        claims = json.loads(fixture.store.read_bytes(synthesis_hash))["claims"]
        audit = ReportAuditCoordinator(
            fixture.database,
            fixture.store,
            fixture.coordinator.gate,
            ReportArtifactStore(tmp_path / "published"),
        )
        coverage = audit._rebuild_persisted_coverage(
            "report-non-evidence",
            {
                "plan": fixture.plan,
                "corpus_snapshot": fixture.corpus,
                "claims": claims,
            },
        )
        p2 = next(item for item in coverage["papers"] if item["paper_id"] == "p2")
        assert p2["disposition"] == disposition
        assert p2["reason"] == reason
        assert p2["evidence_claim_ids"] == []
        assert coverage["complete"]
    finally:
        fixture.database.close()


def test_dispatch_failure_is_terminal_and_resume_never_repeats_paid_node(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture.fake.fail_once = True
    try:
        first = fixture.run("report-retry", "pipeline-retry", now="2026-08-10T01:00:00Z")
        calls_after_first = len(fixture.calls)
        second = fixture.run("report-retry", "pipeline-retry", now="2026-08-10T01:01:00Z")

        assert first.status == "failed"
        assert any(item.status == "failed" for item in first.nodes)
        assert second.status == "failed"
        assert len(fixture.calls) == calls_after_first
        failed = fixture.database.connection.execute(
            """SELECT dispatch_count, budget_calls_reserved FROM report_reduce_nodes
               WHERE report_run_id = 'report-retry' AND status = 'failed'"""
        ).fetchone()
        assert (failed["dispatch_count"], failed["budget_calls_reserved"]) == (1, 2)
    finally:
        fixture.database.close()


def test_sol_invocation_ids_must_be_nonempty_and_unique_for_the_tree(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture.fake.reuse_invocation_id = "reused-invocation"
    try:
        first = fixture.run("report-reused-invocation", "pipeline-reused-invocation")
        paid_calls = len(fixture.calls)
        second = fixture.run("report-reused-invocation", "pipeline-reused-invocation")

        assert first.status == second.status == "failed"
        assert paid_calls == 2
        assert len(fixture.calls) == paid_calls
        invocation_rows = fixture.database.connection.execute(
            """SELECT invocation_id FROM report_reduce_nodes
               WHERE report_run_id = 'report-reused-invocation'
                 AND invocation_id IS NOT NULL"""
        ).fetchall()
        assert [row["invocation_id"] for row in invocation_rows] == [
            "reused-invocation"
        ]
    finally:
        fixture.database.close()


def test_nonstale_lease_is_not_stolen_and_stale_dispatch_fails_uncertain(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _restrict_candidate(fixture)
    try:
        fixture.run("report-lease", "pipeline-lease", now="2026-08-10T01:00:00Z")
        target = fixture.reduce_plan.nodes[0].node_id
        fixture.database.connection.execute(
            """UPDATE report_reduce_nodes SET status = 'running', lease_owner = 'worker-a',
                      lease_token = 1, lease_expires_at = '2026-08-10T02:00:00Z'
               WHERE report_run_id = 'report-lease' AND node_id = ?""",
            (target,),
        )
        fixture.database.connection.commit()
        before = len(fixture.calls)
        live = fixture.run("report-lease", "pipeline-lease", now="2026-08-10T01:30:00Z")
        row = fixture.database.connection.execute(
            "SELECT status, lease_owner, lease_token FROM report_reduce_nodes WHERE report_run_id = 'report-lease' AND node_id = ?",
            (target,),
        ).fetchone()
        assert any(item.node_id == target and item.status == "running" for item in live.nodes)
        assert (row["status"], row["lease_owner"], row["lease_token"]) == ("running", "worker-a", 1)
        assert len(fixture.calls) == before

        fixture.database.connection.execute(
            "UPDATE report_reduce_nodes SET lease_expires_at = '2026-08-10T01:00:00Z' WHERE report_run_id = 'report-lease' AND node_id = ?",
            (target,),
        )
        fixture.database.connection.commit()
        stale = fixture.run("report-lease", "pipeline-lease", now="2026-08-10T01:30:00Z")
        assert any(item.node_id == target and item.status == "failed" for item in stale.nodes)
        recovered = fixture.database.connection.execute(
            "SELECT status, lease_owner, lease_token, error_json FROM report_reduce_nodes WHERE report_run_id = 'report-lease' AND node_id = ?",
            (target,),
        ).fetchone()
        assert recovered["status"] == "failed"
        assert recovered["lease_owner"] is None
        assert recovered["lease_token"] == 1
        assert "InterruptedDispatch" in recovered["error_json"]
    finally:
        fixture.database.close()


@pytest.mark.parametrize("field", ["output_policy_json", "invocation_metadata_json", "actual_input_hash"])
def test_resume_rejects_policy_model_or_input_binding_drift(tmp_path: Path, field: str) -> None:
    fixture = _fixture(tmp_path)
    try:
        fixture.run("report-resume-drift", "pipeline-resume-drift")
        row = fixture.database.connection.execute(
            "SELECT * FROM report_reduce_nodes WHERE report_run_id = 'report-resume-drift' ORDER BY node_id LIMIT 1"
        ).fetchone()
        if field == "output_policy_json":
            value = json.loads(row[field])
            value["access_basis"] = "unknown"
            changed = json.dumps(value)
        elif field == "invocation_metadata_json":
            value = json.loads(row[field])
            value["actual_model"] = None
            changed = json.dumps(value)
        else:
            changed = "0" * 64
        fixture.database.connection.execute(
            f"UPDATE report_reduce_nodes SET {field} = ? WHERE report_reduce_node_id = ?",
            (changed, row["report_reduce_node_id"]),
        )
        fixture.database.connection.commit()

        with pytest.raises(ReportReduceError, match="drifted|binding"):
            fixture.run("report-resume-drift", "pipeline-resume-drift")
    finally:
        fixture.database.close()


@pytest.mark.parametrize("field", ["relative_path", "mime_type", "byte_size"])
def test_resume_rejects_output_artifact_metadata_drift(tmp_path: Path, field: str) -> None:
    fixture = _fixture(tmp_path)
    try:
        fixture.run("report-artifact-drift", "pipeline-artifact-drift")
        row = fixture.database.connection.execute(
            """SELECT a.artifact_id, a.relative_path, a.mime_type, a.byte_size
               FROM report_reduce_nodes rrn
               JOIN artifacts a ON a.artifact_id = rrn.output_artifact_id
               WHERE rrn.report_run_id = 'report-artifact-drift'
               ORDER BY rrn.node_id LIMIT 1"""
        ).fetchone()
        changed = {
            "relative_path": "drifted/report-output.json",
            "mime_type": "text/plain",
            "byte_size": int(row["byte_size"]) + 1,
        }[field]
        fixture.database.connection.execute(
            f"UPDATE artifacts SET {field} = ? WHERE artifact_id = ?",
            (changed, row["artifact_id"]),
        )
        fixture.database.connection.commit()

        with pytest.raises(ReportReduceError, match="artifact metadata has drifted"):
            fixture.run("report-artifact-drift", "pipeline-artifact-drift")
    finally:
        fixture.database.close()
