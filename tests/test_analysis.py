from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from paper_agent.analysis import AnalysisInput, PaperAnalysisCoordinator
from paper_agent.artifacts import ArtifactStore
from paper_agent.codex_exec import CodexExecResult, InvocationMetadata
from paper_agent.grants import GrantStore
from paper_agent.processing import (
    PROCESSING_MODEL,
    PROCESSING_PROVIDER,
    ArtifactProcessingPolicy,
    ProcessingGate,
)
from paper_agent.storage import Database


ROOT = Path(__file__).resolve().parents[1]


def _database(tmp_path: Path) -> Database:
    database = Database(tmp_path / "papers.sqlite")
    database.migrate()
    database.connection.executemany(
        "INSERT INTO papers(paper_id, title) VALUES (?, ?)", (("one", "One"), ("two", "Two")),
    )
    database.connection.commit()
    return database


@dataclass
class FakeInvoker:
    coordinator: PaperAnalysisCoordinator
    fail: bool = False
    calls: list[object] | None = None

    def invoke(self, request):
        assert self.calls is not None
        self.calls.append(request)
        if self.fail:
            raise RuntimeError("fake model failure")
        payload = json.loads(request.prompt)
        output = {
            "paper_id": payload["paper_id"], "artifact_hash": payload["artifact_hash"],
            "input_scope": payload["input_scope"],
            "model": "gpt-5.6-luna", "model_revision": "fake-revision",
            "prompt_hash": self.coordinator.prompt_hash, "schema_hash": self.coordinator.schema_hash,
            "created_at": payload["output_binding"]["created_at"],
            "research_question_and_motivation": "Synthetic fixture motivation.",
            "summary": "Bounded summary.", "methods": [], "key_techniques": [], "datasets": [],
            "experimental_setup": [], "metrics": [], "results": [], "limitations": [],
            "credibility": "Not assessed in the fixture.", "resources": [],
            "topic_relevance": "Relevant", "labels": {
                "subquestion": [], "theme": [], "method_family": [], "task": [], "dataset": [], "benchmark": [],
                "evidence_type": [], "publication_status": "unknown", "study_setting": "other",
            }, "label_evidence": [], "evidence_units": [],
            "comparison_eligibility": "not_comparable",
            "missing_fields": ["full_text"] if payload["input_scope"] != "full_pdf" else ["comparison_evidence"],
        }
        metadata = InvocationMetadata(
            "fake-id", "stage4_analysis_luna", "gpt-5.6-luna", "medium",
            "paper-analysis.schema.json", self.coordinator.schema_hash, request.input_hash,
            "paper-analysis.md", self.coordinator.prompt_hash, "rendered", None, 1,
            "gpt-5.6-luna", "stage4_analysis_luna",
        )
        return CodexExecResult(output, metadata)


def _coordinator(tmp_path: Path, factory):
    database = _database(tmp_path)
    gate = ProcessingGate(ArtifactProcessingPolicy.load(ROOT / "policies" / "artifact-processing-v1.yaml"))
    return database, PaperAnalysisCoordinator(database, ArtifactStore(tmp_path / "store"), gate, invoker_factory=factory)


def test_denial_precedes_executor_construction_and_persists_policy_decision(tmp_path: Path) -> None:
    calls: list[object] = []
    database, coordinator = _coordinator(tmp_path, lambda: (_ for _ in ()).throw(AssertionError("must not construct")))
    try:
        result = coordinator.run("run-denied", [AnalysisInput(
            "one", None, "user_subscription", full_pdf=b"%PDF-restricted",
        )])

        assert result.for_paper("one").status == "incomplete"
        assert calls == []
        row = database.connection.execute("SELECT policy_version, policy_decision, invocation_metadata_json FROM analysis_runs").fetchone()
        assert row["policy_version"] == "artifact-processing-v1"
        assert row["policy_decision"] == "manual"
        assert json.loads(row["invocation_metadata_json"])["processing_decision"]["processing_grant_id"] is None
    finally:
        database.close()


def test_authorized_output_is_bound_persisted_and_resume_skips_model(tmp_path: Path) -> None:
    calls: list[object] = []
    holder: dict[str, PaperAnalysisCoordinator] = {}
    database, coordinator = _coordinator(tmp_path, lambda: FakeInvoker(holder["coordinator"], calls=calls))
    holder["coordinator"] = coordinator
    paper = AnalysisInput("one", "CC-BY-4.0", "open_license", normalized_text="a normalized paper")
    try:
        first = coordinator.run("run-ok", [paper])
        second = coordinator.run("run-ok", [paper])

        assert first.for_paper("one").status == "complete"
        assert second.for_paper("one").resumed
        assert second.for_paper("one").output == first.for_paper("one").output
        assert len(calls) == 1
        row = database.connection.execute("SELECT * FROM analysis_runs").fetchone()
        assert row["input_hash"] == calls[0].input_hash
        detail = json.loads(row["invocation_metadata_json"])
        assert detail["processing_decision"]["input_artifact_hash"] == first.for_paper("one").output["artifact_hash"]
        assert detail["invocation"]["input_hash"] == calls[0].input_hash
        assert row["output_artifact_id"]
        assert row["markdown_artifact_id"] == first.for_paper("one").markdown_artifact_id
        markdown_sha = database.connection.execute(
            "SELECT sha256 FROM artifacts WHERE artifact_id = ?", (row["markdown_artifact_id"],),
        ).fetchone()[0]
        assert "# 论文分析：one" in coordinator.artifact_store.read_bytes(markdown_sha).decode("utf-8")
        assert database.connection.execute("SELECT status FROM pipeline_runs").fetchone()[0] == "complete"
    finally:
        database.close()


def test_one_model_failure_does_not_prevent_other_papers(tmp_path: Path) -> None:
    calls: list[object] = []
    database = _database(tmp_path)
    gate = ProcessingGate(ArtifactProcessingPolicy.load(ROOT / "policies" / "artifact-processing-v1.yaml"))
    count = 0
    coordinator: PaperAnalysisCoordinator

    def factory():
        nonlocal count
        count += 1
        return FakeInvoker(coordinator, fail=count == 1, calls=calls)

    coordinator = PaperAnalysisCoordinator(database, ArtifactStore(tmp_path / "store"), gate, invoker_factory=factory)
    try:
        result = coordinator.run("run-isolated", [
            AnalysisInput("one", "CC-BY-4.0", "open_license", normalized_text="one"),
            AnalysisInput("two", "CC-BY-4.0", "open_license", normalized_text="two"),
        ])

        assert result.for_paper("one").status == "failed"
        assert result.for_paper("two").status == "complete"
        assert count == 2
    finally:
        database.close()


def test_abstract_scope_sends_only_the_bound_abstract_wrapper(tmp_path: Path) -> None:
    calls: list[object] = []
    holder: dict[str, PaperAnalysisCoordinator] = {}
    database, coordinator = _coordinator(tmp_path, lambda: FakeInvoker(holder["coordinator"], calls=calls))
    holder["coordinator"] = coordinator
    try:
        result = coordinator.run("run-abstract", [AnalysisInput(
            "one", None, "public_read_only", abstract="Public abstract", metadata={"title": "One"},
        )])

        assert result.for_paper("one").status == "complete"
        payload = json.loads(calls[0].prompt)
        assert payload["input_scope"] == "abstract_only"
        assert payload["content"] == {"abstract": "Public abstract", "metadata": {"title": "One"}}
        assert payload["output_binding"]["schema_hash"] == coordinator.schema_hash
        assert result.for_paper("one").output["missing_fields"] == ["full_text"]
    finally:
        database.close()


def test_grant_after_denial_completes_once_and_then_resumes(tmp_path: Path) -> None:
    calls: list[object] = []
    database = _database(tmp_path)
    grants = GrantStore(database)
    gate = ProcessingGate(
        ArtifactProcessingPolicy.load(ROOT / "policies" / "artifact-processing-v1.yaml"), grants,
    )
    holder: dict[str, PaperAnalysisCoordinator] = {}
    coordinator = PaperAnalysisCoordinator(
        database, ArtifactStore(tmp_path / "store"), gate,
        invoker_factory=lambda: FakeInvoker(holder["coordinator"], calls=calls),
    )
    holder["coordinator"] = coordinator
    text = "subscription text"
    artifact_hash = sha256(text.encode()).hexdigest()
    paper = AnalysisInput("one", None, "user_subscription", normalized_text=text)
    try:
        denied = coordinator.run("run-granted", [paper], now="2026-08-10T00:00:00Z")
        scope = {
            "paper_ids": ["one"], "artifact_hashes": [artifact_hash], "collection_ids": [],
            "collection_snapshot_hash": None, "selection_snapshot_hash": None, "domains": [],
            "provider": PROCESSING_PROVIDER, "model": PROCESSING_MODEL,
            "data_categories": ["normalized_text"],
        }
        draft = grants.create_draft(
            grant_id="analysis-grant", kind="remote_model_processing",
            actions=["remote_model_processing"], purpose="internal_analysis", mode="attended",
            scope=scope, max_papers=1, expires_at="2026-08-12T00:00:00Z",
        )
        grants.approve(
            draft, draft["content_hash"], approved_by="owner", approved_at="2026-08-10T00:00:00Z",
        )
        completed = coordinator.run(
            "run-granted", [paper], processing_grant_id="analysis-grant", now="2026-08-10T01:00:00Z",
        )
        resumed = coordinator.run(
            "run-granted", [paper], processing_grant_id="analysis-grant", now="2026-08-10T02:00:00Z",
        )

        assert denied.for_paper("one").status == "incomplete"
        assert completed.for_paper("one").status == "complete"
        assert resumed.for_paper("one").resumed
        assert len(calls) == 1
        assert database.connection.execute(
            "SELECT COUNT(*) FROM analysis_runs WHERE run_id = 'run-granted'",
        ).fetchone()[0] == 2
    finally:
        database.close()
