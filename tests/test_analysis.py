from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path

import pytest
import yaml

from paper_agent.analysis import AnalysisInput, PaperAnalysisCoordinator
from paper_agent.artifacts import ArtifactStore
from paper_agent.canonical import content_hash
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


def _database_at_version(tmp_path: Path, version: int) -> Database:
    database = Database(tmp_path / "papers.sqlite")
    for migration in database.migrations():
        if migration.version > version:
            break
        database.connection.executescript(migration.sql)
        database.connection.execute(
            "INSERT INTO schema_migrations(version, name, applied_by) VALUES (?, ?, 'test')",
            (migration.version, migration.name),
        )
        database.connection.commit()
    return database


def _pipeline_input_hash(requests) -> str:
    return content_hash([{
        "paper_id": request.paper_id,
        "artifact_hash": request.artifact_hash,
        "input_scope": request.input_scope,
        "artifact": request.artifact,
        "license": request.license,
        "access_basis": request.access_basis,
        "data_category": request.data_category,
        "mode": request.mode,
    } for request in sorted(requests, key=lambda item: item.paper_id)])


def _decision_document(decision) -> dict:
    document = asdict(decision)
    document["outcome"] = decision.outcome.value
    return document


@dataclass
class FakeInvoker:
    coordinator: PaperAnalysisCoordinator
    fail: bool = False
    calls: list[object] | None = None
    evidence_unit: dict | None = None
    actual_model: str | None = "gpt-5.6-luna"
    actual_profile: str | None = "stage4_analysis_luna"

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
            }, "label_evidence": [], "evidence_units": [self.evidence_unit] if self.evidence_unit else [],
            "comparison_eligibility": "comparable" if self.evidence_unit else "not_comparable",
            "missing_fields": ["full_text"] if payload["input_scope"] != "full_pdf" else ([] if self.evidence_unit else ["comparison_evidence"]),
        }
        metadata = InvocationMetadata(
            "fake-id", "stage4_analysis_luna", "gpt-5.6-luna", "medium",
            "paper-analysis.schema.json", self.coordinator.schema_hash, request.input_hash,
            "paper-analysis.md", self.coordinator.prompt_hash, "rendered", None, 1,
            self.actual_model, self.actual_profile,
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


@pytest.mark.parametrize(
    ("actual_model", "actual_profile"),
    (
        (None, "stage4_analysis_luna"),
        ("gpt-5.6-sol", "stage4_analysis_luna"),
        ("gpt-5.6-luna", None),
        ("gpt-5.6-luna", "stage4b_summary_sol"),
    ),
)
def test_actual_luna_metadata_must_be_present_and_exact(
    tmp_path: Path,
    actual_model: str | None,
    actual_profile: str | None,
) -> None:
    calls: list[object] = []
    holder: dict[str, PaperAnalysisCoordinator] = {}
    database, coordinator = _coordinator(
        tmp_path,
        lambda: FakeInvoker(
            holder["coordinator"],
            calls=calls,
            actual_model=actual_model,
            actual_profile=actual_profile,
        ),
    )
    holder["coordinator"] = coordinator
    try:
        result = coordinator.run(
            "run-actual-metadata",
            [AnalysisInput("one", "CC-BY-4.0", "open_license", normalized_text="paper")],
        )

        paper = result.for_paper("one")
        assert paper.status == "failed"
        assert paper.output is None
        assert len(calls) == 1
        row = database.connection.execute(
            "SELECT status, output_artifact_id FROM analysis_runs"
        ).fetchone()
        assert (row["status"], row["output_artifact_id"]) == ("failed", None)
        dispatch = database.connection.execute(
            "SELECT status, error_json FROM analysis_dispatches"
        ).fetchone()
        assert dispatch["status"] == "failed_terminal"
        assert json.loads(dispatch["error_json"])["cause"]["error"] == "AnalysisValidationError"
    finally:
        database.close()


def test_authorized_output_is_registry_normalized_before_persistence(tmp_path: Path) -> None:
    calls: list[object] = []
    holder: dict[str, PaperAnalysisCoordinator] = {}
    evidence = {
        "claim": "ResNet-50 reached 91% accuracy.", "direction": "support",
        "task_id": "image_classification", "dataset_id": "mnist", "dataset_version": "original",
        "split_id": "test", "metric_id": "accuracy",
        "metric_definition_hash": "677b87be65f571cd1027701cdc332cb607d8558e258f5de6431e34433742fab0",
        "unit": "ratio", "optimization_direction": "maximize", "value": 91.0,
        "uncertainty": None, "statistical_method": None, "protocol_id": "official_test",
        "protocol_hash": "aa15debef3444b8ee215dd6043eaf89a00af76145ff2cfbf2a5f01bd6b67a9c3",
        "sample_size": 10000, "baseline_id": "resnet50", "baseline_version": "torchvision",
        "conditions": [
            "source_task=image classification", "source_dataset=MNIST", "source_metric=accuracy",
            "source_baseline=ResNet-50", "source_protocol=official test split", "source_unit=percent",
        ],
        "locator": {"kind": "page", "value": "7"},
        "normalization_method": "model_candidate", "normalizer_version": "model_candidate",
        "source_value": 91.0, "comparison_eligibility": "comparable", "missing_fields": [],
    }
    database, coordinator = _coordinator(
        tmp_path, lambda: FakeInvoker(holder["coordinator"], calls=calls, evidence_unit=evidence),
    )
    holder["coordinator"] = coordinator
    try:
        result = coordinator.run("run-normalized", [AnalysisInput(
            "one", "CC-BY-4.0", "open_license", normalized_text="normalized paper",
        )])

        unit = result.for_paper("one").output["evidence_units"][0]
        assert unit["value"] == 0.91
        assert unit["normalizer_version"] == "analysis-normalization-v1"
        metadata = json.loads(database.connection.execute(
            "SELECT invocation_metadata_json FROM analysis_runs",
        ).fetchone()[0])
        assert metadata["normalization_registry"]["registry_hash"] == coordinator.normalization_registry.registry_hash
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

        resumed = coordinator.run("run-isolated", [
            AnalysisInput("one", "CC-BY-4.0", "open_license", normalized_text="one"),
            AnalysisInput("two", "CC-BY-4.0", "open_license", normalized_text="two"),
        ])
        assert resumed.for_paper("one").status == "failed"
        assert resumed.for_paper("one").resumed
        assert "UncertainDispatch" in resumed.for_paper("one").error
        assert resumed.for_paper("two").resumed
        assert count == 2
        dispatch = database.connection.execute(
            "SELECT status, dispatch_count, error_json FROM analysis_dispatches WHERE paper_id = 'one'",
        ).fetchone()
        assert (dispatch["status"], dispatch["dispatch_count"]) == ("failed_terminal", 1)
        assert json.loads(dispatch["error_json"])["error"] == "UncertainDispatch"
        assert database.connection.execute(
            "SELECT status FROM pipeline_runs WHERE run_id = 'run-isolated'",
        ).fetchone()[0] == "failed"
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
        dispatch = database.connection.execute(
            "SELECT status, dispatch_count, stable_created_at FROM analysis_dispatches",
        ).fetchone()
        assert (dispatch["status"], dispatch["dispatch_count"]) == ("complete", 1)
        assert dispatch["stable_created_at"] == "2026-08-10T00:00:00Z"
        assert completed.for_paper("one").output["created_at"] == "2026-08-10T00:00:00Z"
    finally:
        database.close()


def test_concurrent_coordinator_observes_claim_without_constructing_a_second_executor(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path)
    second_database = Database(database.path)
    second_database.migrate()
    policy = ArtifactProcessingPolicy.load(ROOT / "policies" / "artifact-processing-v1.yaml")
    paper = AnalysisInput("one", "CC-BY-4.0", "open_license", normalized_text="shared")
    calls: list[object] = []
    forbidden_factory_calls = 0
    concurrent_results = []
    first: PaperAnalysisCoordinator

    def forbidden_factory():
        nonlocal forbidden_factory_calls
        forbidden_factory_calls += 1
        raise AssertionError("a live analysis dispatch must not construct another executor")

    second = PaperAnalysisCoordinator(
        second_database,
        ArtifactStore(tmp_path / "store"),
        ProcessingGate(policy),
        invoker_factory=forbidden_factory,
    )

    class ReentrantInvoker:
        def invoke(self, request):
            calls.append(request)
            concurrent_results.append(second.run("run-concurrent", [paper]))
            return FakeInvoker(first, calls=[]).invoke(request)

    first = PaperAnalysisCoordinator(
        database,
        ArtifactStore(tmp_path / "store"),
        ProcessingGate(policy),
        invoker_factory=ReentrantInvoker,
    )
    try:
        completed = first.run("run-concurrent", [paper])
        resumed = second.run("run-concurrent", [paper])

        assert completed.for_paper("one").status == "complete"
        assert concurrent_results[0].for_paper("one").status == "incomplete"
        assert concurrent_results[0].for_paper("one").resumed
        assert resumed.for_paper("one").status == "complete"
        assert resumed.for_paper("one").resumed
        assert len(calls) == 1
        assert forbidden_factory_calls == 0
        assert database.connection.execute(
            "SELECT dispatch_count FROM analysis_dispatches",
        ).fetchone()[0] == 1
    finally:
        second_database.close()
        database.close()


def test_expired_claim_becomes_uncertain_terminal_and_resume_never_reinvokes(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path)
    gate = ProcessingGate(ArtifactProcessingPolicy.load(ROOT / "policies" / "artifact-processing-v1.yaml"))
    moment = [datetime(2026, 8, 10, tzinfo=timezone.utc)]
    calls: list[object] = []
    factory_calls = 0

    class CrashingInvoker:
        def invoke(self, request):
            calls.append(request)
            raise KeyboardInterrupt("simulated worker loss after paid dispatch")

    def factory():
        nonlocal factory_calls
        factory_calls += 1
        if factory_calls > 1:
            raise AssertionError("an expired claimed dispatch must never reconstruct an executor")
        return CrashingInvoker()

    coordinator = PaperAnalysisCoordinator(
        database,
        ArtifactStore(tmp_path / "store"),
        gate,
        invoker_factory=factory,
        clock=lambda: moment[0],
        dispatch_lease_seconds=10,
    )
    paper = AnalysisInput("one", "CC-BY-4.0", "open_license", normalized_text="crash")
    try:
        with pytest.raises(KeyboardInterrupt, match="simulated worker loss"):
            coordinator.run("run-expired", [paper])
        running = database.connection.execute(
            "SELECT status, dispatch_count FROM analysis_dispatches",
        ).fetchone()
        assert (running["status"], running["dispatch_count"]) == ("running", 1)

        moment[0] += timedelta(seconds=11)
        expired = coordinator.run("run-expired", [paper])
        resumed = coordinator.run("run-expired", [paper])

        assert expired.for_paper("one").status == "failed"
        assert expired.for_paper("one").resumed
        assert "UncertainDispatch" in expired.for_paper("one").error
        assert resumed.for_paper("one").status == "failed"
        assert resumed.for_paper("one").resumed
        assert factory_calls == 1
        assert len(calls) == 1
        terminal = database.connection.execute(
            "SELECT status, dispatch_count, lease_owner, analysis_run_id FROM analysis_dispatches",
        ).fetchone()
        assert (terminal["status"], terminal["dispatch_count"], terminal["lease_owner"]) == (
            "failed_terminal", 1, None,
        )
        assert terminal["analysis_run_id"]
        assert database.connection.execute(
            "SELECT status FROM analysis_runs WHERE analysis_run_id = ?",
            (terminal["analysis_run_id"],),
        ).fetchone()[0] == "failed"
        assert database.connection.execute(
            "SELECT status FROM pipeline_runs WHERE run_id = 'run-expired'",
        ).fetchone()[0] == "failed"
    finally:
        database.close()


@pytest.mark.parametrize(
    ("legacy_status", "with_decision"),
    (("failed", False), ("running", True)),
)
def test_migration_adopts_legacy_uncertain_analysis_as_terminal_without_reinvocation(
    tmp_path: Path,
    legacy_status: str,
    with_decision: bool,
) -> None:
    database = _database_at_version(tmp_path, 15)
    policy = ArtifactProcessingPolicy.load(ROOT / "policies" / "artifact-processing-v1.yaml")
    gate = ProcessingGate(policy)
    factory_calls = 0

    def forbidden_factory():
        nonlocal factory_calls
        factory_calls += 1
        raise AssertionError("a pre-migration uncertain analysis must never be dispatched again")

    coordinator = PaperAnalysisCoordinator(
        database,
        ArtifactStore(tmp_path / "store"),
        gate,
        invoker_factory=forbidden_factory,
    )
    paper = AnalysisInput(
        "one", "CC-BY-4.0", "open_license", normalized_text=f"legacy-{legacy_status}",
    )
    request = paper.processing_request()
    decision = gate.decide(request)
    metadata = {
        "input_policy_facts": {
            "paper_id": "one",
            "artifact_hash": request.artifact_hash,
            "artifact": request.artifact,
            "input_scope": request.input_scope,
        },
    }
    if with_decision:
        metadata["processing_decision"] = _decision_document(decision)
    try:
        database.connection.execute("INSERT INTO papers(paper_id, title) VALUES ('one', 'One')")
        database.connection.execute(
            """INSERT INTO pipeline_runs(
                   run_id, stage, status, input_hash, config_hash,
                   implementation_version, started_at
               ) VALUES (?, 'stage4', ?, ?, ?, ?, '2026-08-09T00:00:00Z')""",
            (
                "legacy-run",
                "running" if legacy_status == "running" else "incomplete",
                _pipeline_input_hash((request,)),
                coordinator.legacy_config_hash,
                coordinator.implementation_version,
            ),
        )
        database.connection.execute(
            """INSERT INTO analysis_runs(
                   analysis_run_id, run_id, paper_id, artifact_id, input_hash,
                   input_scope, model_id, model_revision, prompt_hash, schema_hash,
                   implementation_version, authorization_grant_id, policy_version,
                   policy_decision, invocation_metadata_json, status, created_at,
                   completed_at
               ) VALUES (
                   ?, 'legacy-run', 'one', NULL, ?, ?, 'gpt-5.6-luna',
                   'legacy-revision', ?, ?, ?, NULL, ?, ?, ?, ?,
                   '2026-08-09T00:01:00Z', ?
               )""",
            (
                f"legacy-analysis-{legacy_status}",
                request.artifact_hash,
                request.input_scope,
                coordinator.prompt_hash,
                coordinator.schema_hash,
                coordinator.implementation_version,
                decision.policy_version if with_decision else "unavailable",
                decision.outcome.value if with_decision else "failed_before_policy",
                json.dumps(metadata, sort_keys=True),
                legacy_status,
                None if legacy_status == "running" else "2026-08-09T00:02:00Z",
            ),
        )
        database.connection.commit()

        assert [migration.version for migration in database.migrate()] == [16, 17]
        adopted = database.connection.execute(
            """SELECT status, dispatch_count, policy_version, policy_hash,
                      analysis_run_id, error_json
               FROM analysis_dispatches WHERE run_id = 'legacy-run'""",
        ).fetchone()
        assert (adopted["status"], adopted["dispatch_count"]) == ("failed_terminal", 1)
        assert adopted["analysis_run_id"] == f"legacy-analysis-{legacy_status}"
        assert json.loads(adopted["error_json"])["reason"] == f"pre_migration_{legacy_status}"
        if with_decision:
            assert (adopted["policy_version"], adopted["policy_hash"]) == (
                policy.version,
                policy.hash,
            )
        else:
            assert (adopted["policy_version"], adopted["policy_hash"]) == (
                "unavailable",
                "legacy-unavailable",
            )
        assert database.connection.execute(
            "SELECT status FROM pipeline_runs WHERE run_id = 'legacy-run'",
        ).fetchone()[0] == "failed"

        resumed = coordinator.run("legacy-run", [paper])

        assert resumed.for_paper("one").status == "failed"
        assert resumed.for_paper("one").resumed
        assert "UncertainDispatch" in resumed.for_paper("one").error
        assert factory_calls == 0
        assert database.connection.execute(
            "SELECT dispatch_count FROM analysis_dispatches WHERE run_id = 'legacy-run'",
        ).fetchone()[0] == 1
        assert database.connection.execute(
            "SELECT config_hash FROM pipeline_runs WHERE run_id = 'legacy-run'",
        ).fetchone()[0] == coordinator.config_hash
    finally:
        database.close()


def test_stage4_policy_hash_is_frozen_in_pipeline_and_dispatch_binding(tmp_path: Path) -> None:
    database = _database(tmp_path)
    original = ArtifactProcessingPolicy.load(ROOT / "policies" / "artifact-processing-v1.yaml")
    paper = AnalysisInput("one", None, "user_subscription", normalized_text="restricted")
    first = PaperAnalysisCoordinator(
        database,
        ArtifactStore(tmp_path / "store"),
        ProcessingGate(original),
        invoker_factory=lambda: (_ for _ in ()).throw(AssertionError("denial must not dispatch")),
    )
    widened_document = yaml.safe_load(
        (ROOT / "policies" / "artifact-processing-v1.yaml").read_text(encoding="utf-8")
    )
    widened_document["matrix"][5]["outcome"] = "full_pdf"
    widened = ArtifactProcessingPolicy(widened_document)
    factory_calls = 0

    def forbidden_factory():
        nonlocal factory_calls
        factory_calls += 1
        raise AssertionError("policy drift must be rejected before dispatch")

    try:
        denied = first.run("policy-frozen", [paper])
        assert denied.for_paper("one").status == "incomplete"
        dispatch = database.connection.execute(
            """SELECT config_hash, policy_version, policy_hash
               FROM analysis_dispatches WHERE run_id = 'policy-frozen'""",
        ).fetchone()
        assert (dispatch["policy_version"], dispatch["policy_hash"]) == (
            original.version,
            original.hash,
        )
        assert database.connection.execute(
            "SELECT config_hash FROM pipeline_runs WHERE run_id = 'policy-frozen'",
        ).fetchone()[0] == first.config_hash

        changed = PaperAnalysisCoordinator(
            database,
            ArtifactStore(tmp_path / "store"),
            ProcessingGate(widened),
            invoker_factory=forbidden_factory,
        )
        with pytest.raises(ValueError, match="configuration is immutable"):
            changed.run("policy-frozen", [paper])

        assert changed.config_hash != first.config_hash
        assert widened.version == original.version
        assert widened.hash != original.hash
        assert factory_calls == 0
        unchanged = database.connection.execute(
            "SELECT policy_version, policy_hash, dispatch_count FROM analysis_dispatches",
        ).fetchone()
        assert (unchanged["policy_version"], unchanged["policy_hash"], unchanged["dispatch_count"]) == (
            original.version,
            original.hash,
            0,
        )
    finally:
        database.close()
