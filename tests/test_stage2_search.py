from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import pytest

import paper_agent.stage2_search as stage2_search
from paper_agent.approval import approved_content_hash
from paper_agent.canonical import content_hash
from paper_agent.domain import FilterStatus
from paper_agent.query_plan import approve_query_plan, compile_query_plan
from paper_agent.scope_filter import screening_scope_hash
from paper_agent.stage2_backends import (
    OmlxResponse,
    load_model_lock,
    write_model_lock,
)
from paper_agent.stage2_evaluation import (
    CalibrationPath,
    GateResult,
    PathCalibrator,
    ThresholdArtifact,
    phase3_release_gate,
    write_path_calibrator,
)
from paper_agent.stage2_pipeline import (
    ADJUDICATOR_SHARE_ALARM,
    ERROR_RATE_ALARM,
    PathCalibration,
    Stage2Profile,
    Stage2Summary,
)
from paper_agent.stage2_search import (
    Stage2ReleaseError,
    Stage2SearchScreener,
    load_stage2_benchmark_candidate,
    load_stage2_release as _load_stage2_release,
)
from paper_agent.storage import Database


ROOT = Path(__file__).parents[1]
GATE_NAMES = ("promotion", "structured_replay", "rationale", "parity", "benchmark", "soak")


def load_stage2_release(path: Path, plan: dict):
    """Use the test-only fast evidence verifier for runtime/configuration tests.

    The full raw-evidence, trust-root, and Ed25519 path is exercised in
    ``test_stage2_release_v3.py``.  Keeping this test module focused avoids
    rebuilding the production-size 1k/10k evidence corpus for every ordinary
    runtime validation assertion.
    """

    return _load_stage2_release(
        path,
        plan,
        hidden_trust_path=path.parent.parent / f"{path.parent.name}-test-hidden-evaluator-trust.json",
        parity_oracle_trust_path=(
            path.parent.parent / f"{path.parent.name}-test-parity-oracle-trust.json"
        ),
    )


@pytest.fixture(autouse=True)
def _fast_release_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        stage2_search,
        "_load_deployment_hidden_trust",
        lambda _path, *, bundle_root: object(),
    )
    monkeypatch.setattr(
        stage2_search,
        "_load_deployment_parity_oracle_trust",
        lambda _path, *, bundle_root: object(),
    )

    def verified_gate(
        _release_path: Path,
        document: dict,
        _profile_name: str,
        _profile: Stage2Profile,
        _hidden_trust: object,
        _oracle_trust: object,
    ):
        return phase3_release_gate(
            candidate_id=document["candidate_id"],
            evaluation_manifest_hash=document["evaluation_manifest_hash"],
            artifacts={
                name: (f"{index:x}" * 64, GateResult(True, ()))
                for index, name in enumerate(GATE_NAMES, start=1)
            },
            throughput_runs=(99.0, 100.0, 101.0),
        ), None

    monkeypatch.setattr(stage2_search, "_release_gate", verified_gate)


def _release_bundle(
    tmp_path: Path,
    *,
    base_url: str = "http://127.0.0.1:8000",
    reranker_lock_source: Path | None = None,
    profile_name: str = "local-winner",
) -> tuple[Path, dict]:
    reranker = load_model_lock(
        reranker_lock_source
        or ROOT / "configs/stage2/models/bge-reranker-v2-m3-fp32.lock.json"
    )
    adjudicator = load_model_lock(ROOT / "configs/stage2/models/qwen3.5-9b-8bit.lock.json")
    reranker_path = tmp_path / "reranker.lock.json"
    adjudicator_path = tmp_path / "adjudicator.lock.json"
    write_model_lock(reranker_path, reranker)
    write_model_lock(adjudicator_path, adjudicator)
    reranker_hash = sha256(reranker_path.read_bytes()).hexdigest()
    adjudicator_hash = sha256(adjudicator_path.read_bytes()).hexdigest()

    evaluation_manifest_hash = "a" * 64
    research = {
        "objective": "screen graph learning papers",
        "audience": "researchers",
        "primary_question": "Which graph learning methods are relevant?",
        "subquestions": [{"id": "sq1", "question": "Which methods are relevant?"}],
    }
    scope = {
        "date_from": "2020-01-01",
        "date_to": "2026-12-31",
        "venues": [],
        "fields": ["computer science"],
        "languages": ["en"],
        "document_types": ["article"],
        "user_seeds": [],
        "include_arxiv_candidates": False,
    }
    inclusion = {"criteria": ["topic match"], "exclusion_criteria": ["unrelated"]}
    scope_hash = screening_scope_hash({
        "research": research,
        "inclusion": inclusion,
        "scope": scope,
    })
    runtime = {
        "query": "graph learning methods",
        "query_version": "screening-query-v1",
        "screening_scope_hash": scope_hash,
        "evaluation_topic_queries": [
            {"topic": "molecular_generation", "language": "en", "query": "molecular generation"},
            {"topic": "molecular_generation", "language": "zh", "query": "分子生成"},
        ],
        "include_document_types": [],
        "exclude_document_types": ["editorial", "retraction"],
        "token_bucket_width": 128,
        "document_batch_size": 32,
        "max_in_flight": 2,
        "adjudicator_concurrency": 4,
        "adjudicator_seed": 42,
        "max_context_window": 16_384,
        "max_tokens": 256,
        "omlx_base_url": base_url,
        "api_key_env": None,
        "prompt_version": "stage2-adjudication-v1",
        "schema_version": "filter-decision.schema.json",
    }
    base_profile = Stage2Profile(
        query=runtime["query"],
        query_version=runtime["query_version"],
        thresholds=None,
        reranker_model_id=reranker.model_id,
        reranker_revision=reranker.conversion_revision or reranker.source_revision,
        adjudicator_model_id=adjudicator.model_id,
        adjudicator_revision=adjudicator.conversion_revision or adjudicator.source_revision,
        screening_scope_hash=runtime["screening_scope_hash"],
        reranker_lock_hash=reranker_hash,
        adjudicator_lock_hash=adjudicator_hash,
        release_gate_hash=None,
        evaluation_topic_queries=tuple(
            (item["topic"], item["language"], item["query"])
            for item in runtime["evaluation_topic_queries"]
        ),
        include_document_types=frozenset(runtime["include_document_types"]),
        exclude_document_types=frozenset(runtime["exclude_document_types"]),
        token_bucket_width=runtime["token_bucket_width"],
        document_batch_size=runtime["document_batch_size"],
        reranker_max_in_flight=runtime["max_in_flight"],
        adjudicator_concurrency=runtime["adjudicator_concurrency"],
        adjudicator_seed=runtime["adjudicator_seed"],
        adjudicator_max_context_window=runtime["max_context_window"],
        adjudicator_max_output_tokens=runtime["max_tokens"],
        omlx_base_url=runtime["omlx_base_url"],
        api_key_env=runtime["api_key_env"],
        prompt_version=runtime["prompt_version"],
        schema_version=runtime["schema_version"],
    )
    pair_ids = ("pair-dev-negative", "pair-dev-positive")
    pair_ids_hash = _evaluation_hash(sorted(pair_ids))
    calibrators = {
        CalibrationPath.RERANKER: PathCalibrator(
            1,
            CalibrationPath.RERANKER,
            8.0,
            -4.0,
            "b" * 64,
            evaluation_manifest_hash,
            reranker_hash,
            "c" * 64,
            pair_ids_hash,
            len(pair_ids),
            pair_ids,
        ),
        CalibrationPath.QWEN: PathCalibrator(
            1,
            CalibrationPath.QWEN,
            8.0,
            -4.0,
            "b" * 64,
            evaluation_manifest_hash,
            adjudicator_hash,
            "c" * 64,
            pair_ids_hash,
            len(pair_ids),
            pair_ids,
        ),
    }
    thresholds = {
        path: ThresholdArtifact(
            1,
            path,
            0.2,
            0.8,
            calibrator.hash(),
            calibrator.model_lock_hash,
            calibrator.dev_manifest_hash,
            calibrator.dev_label_hash,
            base_profile.base_runtime_config_hash,
        )
        for path, calibrator in calibrators.items()
    }
    calibration_documents: dict[str, dict[str, dict[str, str]]] = {}
    bindings: dict[CalibrationPath, PathCalibration] = {}
    for calibration_path in CalibrationPath:
        calibrator = calibrators[calibration_path]
        threshold = thresholds[calibration_path]
        calibrator_path = tmp_path / f"{calibration_path.value}-calibrator.json"
        threshold_path = tmp_path / f"{calibration_path.value}-threshold.json"
        write_path_calibrator(calibrator_path, calibrator)
        threshold_path.write_text(
            json.dumps(threshold.document(), sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        calibration_documents[calibration_path.value] = {
            "calibrator": {
                "path": calibrator_path.name,
                "sha256": sha256(calibrator_path.read_bytes()).hexdigest(),
            },
            "threshold": {
                "path": threshold_path.name,
                "sha256": sha256(threshold_path.read_bytes()).hexdigest(),
            },
        }
        bindings[calibration_path] = PathCalibration(calibrator, threshold)
    gate_bindings = {
        "model_lock_hashes": {
            CalibrationPath.RERANKER.value: reranker_hash,
            CalibrationPath.QWEN.value: adjudicator_hash,
        },
        "calibrator_hashes": {
            path.value: calibrator.hash() for path, calibrator in calibrators.items()
        },
        "threshold_hashes": {
            path.value: threshold.hash() for path, threshold in thresholds.items()
        },
    }
    gate = {
        "candidate_id": profile_name,
        "candidate_bundle_sha256": "9" * 64,
        "evaluation_manifest_hash": evaluation_manifest_hash,
        "evidence": {"path": "stage2-release-evidence.json", "sha256": "0" * 64},
    }
    base_profile = replace(base_profile, release_gate_hash=content_hash(gate))
    profile = replace(
        base_profile,
        reranker_calibration=bindings[CalibrationPath.RERANKER],
        adjudicator_calibration=bindings[CalibrationPath.QWEN],
    )
    plan_draft = {
        "created_at": "2026-08-09T00:00:00Z",
        "research": research,
        "scope": scope,
        "inclusion": inclusion,
        "query_variants": [{
            "id": "q1",
            "subquestion_id": "sq1",
            "alias_group": "graph-learning",
            "raw_query": "graph learning methods",
            "synonyms": [],
        }],
        "filter": {
            "profile": profile_name,
            "config_hash": profile.config_hash,
            "thresholds_hash": profile.threshold_hash,
            "seed_selector_version": "1",
            "seed_selector_config_hash": "d" * 64,
            "round_state_machine_version": "1",
        },
        "citation_snowball": {
            "enabled": False,
            "directions": ["references", "citations"],
            "max_depth": 1,
            "max_rounds": 1,
            "max_per_seed_per_source": 10,
        },
        "budgets": {
            "max_requests": 10,
            "max_candidates": 100,
            "max_seconds": 60,
            "saturation": {"min_unique_included_yield": 0.05, "consecutive_low_yield_rounds": 2},
        },
        "provider_policy": "all_resolved",
        "required_roles": ["search"],
        "required_providers": ["openalex"],
    }
    provider = {
        "provider": "openalex",
        "distribution": "paper-agent-openalex",
        "version": "1.0.0",
        "artifact_sha256": "e" * 64,
        "manifest_hash": "f" * 64,
        "roles": ["search"],
        "capabilities": ["stable_id", "metadata", "date_filter"],
        "enabled": True,
        "mode": "api",
        "credentials_present": True,
    }
    draft_plan = compile_query_plan(plan_draft, providers=[provider])
    plan = approve_query_plan(
        draft_plan,
        draft_plan["plan_hash"],
        approved_by="owner",
        approved_at="2026-08-09T01:00:00Z",
    )
    release = {
        "schema_version": "3",
        "profile": profile_name,
        "reranker_lock": {"path": reranker_path.name, "sha256": reranker_hash},
        "adjudicator_lock": {"path": adjudicator_path.name, "sha256": adjudicator_hash},
        "calibration": calibration_documents,
        "release_gate": gate,
        "runtime": runtime,
    }
    release_path = tmp_path / "stage2-release.json"
    release_path.write_text(json.dumps(release, sort_keys=True), encoding="utf-8")
    return release_path, plan


def _evaluation_hash(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()


def _approved_plan_with_screening_change(
    plan: dict,
    section: str,
    field: str,
    value: object,
) -> dict:
    changed = deepcopy(plan)
    changed["status"] = "draft"
    changed["approval"] = None
    changed[section][field] = value
    changed["filter"]["screening_scope_hash"] = screening_scope_hash(changed)
    changed["plan_hash"] = approved_content_hash(changed)
    return approve_query_plan(
        changed,
        changed["plan_hash"],
        approved_by="owner",
        approved_at="2026-08-09T02:00:00Z",
    )


class LocalOmlxFixture:
    def __init__(
        self,
        model_id: str = "bge-reranker-v2-m3",
        *,
        relevant_score: float = 0.7,
        irrelevant_score: float = 0.3,
    ) -> None:
        self.paths: list[str] = []
        self.model_id = model_id
        self.relevant_score = relevant_score
        self.irrelevant_score = irrelevant_score

    def request(self, path, payload):
        self.paths.append(path)
        assert payload["model"] == self.model_id
        scores = [
            self.relevant_score if "Relevant" in document else self.irrelevant_score
            for document in payload["documents"]
        ]
        body = {"model": payload["model"], "results": [{"index": index, "relevance_score": score} for index, score in enumerate(scores)]}
        return OmlxResponse(200, json.dumps(body).encode())


class AdjudicatingOmlxFixture:
    def __init__(self) -> None:
        self.paths: list[str] = []

    def request(self, path, payload):
        self.paths.append(path)
        if path == "/v1/rerank":
            body = {
                "model": payload["model"],
                "results": [
                    {"index": index, "relevance_score": 0.5}
                    for index, _ in enumerate(payload["documents"])
                ]
            }
            return OmlxResponse(200, json.dumps(body).encode())
        prompt = payload["messages"][1]["content"]
        paper_id = "consistent" if "Paper ID: consistent" in prompt else "conflict"
        decision = {
            "paper_id": paper_id,
            "decision": "relevant" if paper_id == "consistent" else "irrelevant",
            "score": 0.9,
            "reason_codes": ["topic_match"],
            "rationale": "The abstract addresses the query.",
            "evidence_fields": ["abstract"],
        }
        body = {
            "model": payload["model"],
            "choices": [{"message": {"content": json.dumps(decision)}}],
        }
        return OmlxResponse(200, json.dumps(body).encode())


def test_benchmark_candidate_loads_before_throughput_release_gate_exists(
    tmp_path: Path,
) -> None:
    release_path, plan = _release_bundle(tmp_path)
    document = json.loads(release_path.read_text(encoding="utf-8"))
    document.pop("release_gate")
    document["schema_version"] = "2"
    candidate_path = tmp_path / "stage2-candidate.json"
    candidate_path.write_text(json.dumps(document), encoding="utf-8")

    candidate = load_stage2_benchmark_candidate(candidate_path)

    assert candidate.profile.release_gate_hash is None
    assert candidate.profile.screening_scope_hash == plan["filter"][
        "screening_scope_hash"
    ]
    assert candidate.profile.reranker_calibration is not None
    assert candidate.profile.adjudicator_calibration is not None


def test_release_requires_deployment_controlled_parity_oracle_trust(tmp_path) -> None:
    release_path, plan = _release_bundle(tmp_path)

    with pytest.raises(Stage2ReleaseError, match="deployment-controlled parity oracle"):
        _load_stage2_release(
            release_path,
            plan,
            hidden_trust_path=tmp_path.parent / "hidden-trust.json",
            environment={},
        )


def test_release_reads_parity_oracle_trust_path_from_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release_path, plan = _release_bundle(tmp_path)
    oracle_path = tmp_path.parent / "parity-oracle-trust.json"
    loaded: list[Path] = []

    def load_oracle(path: Path, *, bundle_root: Path) -> object:
        loaded.append(path)
        return object()

    monkeypatch.setattr(
        stage2_search,
        "_load_deployment_parity_oracle_trust",
        load_oracle,
    )

    _load_stage2_release(
        release_path,
        plan,
        hidden_trust_path=tmp_path.parent / "hidden-trust.json",
        environment={"PAPER_AGENT_STAGE2_PARITY_ORACLE_TRUST": str(oracle_path)},
    )

    assert loaded == [oracle_path]


def test_parity_oracle_trust_cannot_come_from_release_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.undo()
    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir()
    bundled_trust = bundle_root / "parity-oracle-trust.json"
    bundled_trust.write_text("{}", encoding="utf-8")

    with pytest.raises(Stage2ReleaseError, match="must stay outside"):
        stage2_search._load_deployment_parity_oracle_trust(
            bundled_trust,
            bundle_root=bundle_root.resolve(),
        )


def test_released_stage2_screens_database_papers_and_persists_decisions(tmp_path) -> None:
    release_path, plan = _release_bundle(tmp_path)
    released = load_stage2_release(release_path, plan)
    transport = LocalOmlxFixture()

    with Database(tmp_path / "papers.sqlite3") as database:
        database.migrate()
        database.connection.executemany(
            "INSERT INTO papers(paper_id, title, abstract) VALUES (?, ?, ?)",
            (("relevant", "Relevant paper", "graph learning"), ("irrelevant", "Other paper", "cooking")),
        )
        screener = released.screener(database, "crawl-1", transport=transport)

        decisions = screener.screen(("irrelevant", "relevant"))

        assert decisions == {
            "irrelevant": FilterStatus.IRRELEVANT,
            "relevant": FilterStatus.RELEVANT,
        }
        assert transport.paths == ["/v1/rerank"]
        rows = database.connection.execute(
            "SELECT paper_id, status, model_id, reason FROM filter_decisions ORDER BY paper_id"
        ).fetchall()
        assert [tuple(row[:3]) for row in rows] == [
            ("irrelevant", "irrelevant", "bge-reranker-v2-m3"),
            ("relevant", "relevant", "bge-reranker-v2-m3"),
        ]
        provenance = json.loads(rows[0]["reason"])
        assert provenance["screening_scope_hash"] == plan["filter"][
            "screening_scope_hash"
        ]
        assert provenance["release_gate_hash"] == released.profile.release_gate_hash
        assert provenance["reranker_lock_hash"] == released.profile.reranker_lock_hash
        assert provenance["reranker_score"] == 0.3
        assert provenance["reranker_probability"] < 0.2
        assert provenance["reranker_calibrator_hash"] == released.profile.reranker_calibration.calibrator.hash()
        assert provenance["reranker_threshold_hash"] == released.profile.reranker_calibration.threshold.hash()
        assert provenance["base_runtime_config_hash"] == released.profile.base_runtime_config_hash
        assert provenance["threshold_bundle_hash"] == released.profile.threshold_bundle_hash
        assert provenance["full_profile_hash"] == released.profile.full_profile_hash
        assert released.profile.reranker_calibration.threshold.stage2_config_hash == (
            released.profile.base_runtime_config_hash
        )
        assert len({
            released.profile.base_runtime_config_hash,
            released.profile.threshold_bundle_hash,
            released.profile.full_profile_hash,
        }) == 3
        stored_score = database.connection.execute(
            "SELECT score FROM filter_decisions WHERE paper_id = 'irrelevant'"
        ).fetchone()[0]
        assert stored_score == provenance["reranker_probability"]
        stage2_run = database.connection.execute(
            "SELECT stage, status, config_hash FROM pipeline_runs WHERE run_id = ?",
            (screener.run_ids[0],),
        ).fetchone()
        assert tuple(stage2_run) == ("stage-2", "complete", released.profile.config_hash)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    (
        ("research", "objective", "changed research objective"),
        ("inclusion", "criteria", ["replicated evidence"]),
        ("scope", "languages", ["zh"]),
    ),
)
def test_release_rejects_approved_plan_for_a_different_screening_scope(
    tmp_path,
    section,
    field,
    value,
) -> None:
    release_path, plan = _release_bundle(tmp_path)
    changed_plan = _approved_plan_with_screening_change(
        plan,
        section,
        field,
        value,
    )

    with pytest.raises(Stage2ReleaseError, match="screening scope does not match"):
        load_stage2_release(release_path, changed_plan)


def test_release_runtime_requires_an_exact_lowercase_screening_scope_hash(tmp_path) -> None:
    release_path, plan = _release_bundle(tmp_path / "missing")
    document = json.loads(release_path.read_text())
    document["runtime"].pop("screening_scope_hash")
    release_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(Stage2ReleaseError, match="runtime fields are not exact"):
        load_stage2_release(release_path, plan)

    for index, invalid in enumerate(("a" * 63, "A" * 64)):
        release_path, plan = _release_bundle(tmp_path / f"invalid-{index}")
        document = json.loads(release_path.read_text())
        document["runtime"]["screening_scope_hash"] = invalid
        release_path.write_text(json.dumps(document), encoding="utf-8")
        with pytest.raises(Stage2ReleaseError, match="lowercase SHA-256"):
            load_stage2_release(release_path, plan)


def test_qwen_uses_its_calibrator_and_conflicts_fail_open(tmp_path) -> None:
    release_path, plan = _release_bundle(tmp_path)
    released = load_stage2_release(release_path, plan)
    transport = AdjudicatingOmlxFixture()

    with Database(tmp_path / "papers.sqlite3") as database:
        database.migrate()
        database.connection.executemany(
            "INSERT INTO papers(paper_id, title, abstract) VALUES (?, ?, ?)",
            (
                ("consistent", "Consistent", "graph learning"),
                ("conflict", "Conflict", "graph learning"),
            ),
        )
        screener = released.screener(database, "crawl-qwen", transport=transport)

        decisions = screener.screen(("conflict", "consistent"))

        assert decisions == {
            "conflict": FilterStatus.NEEDS_REVIEW,
            "consistent": FilterStatus.RELEVANT,
        }
        assert transport.paths == ["/v1/rerank", "/v1/chat/completions", "/v1/chat/completions"]
        persisted = {
            row["paper_id"]: json.loads(row["reason"])
            for row in database.connection.execute(
                "SELECT paper_id, reason FROM filter_decisions ORDER BY paper_id"
            )
        }
        assert "qwen_calibration_conflict" in persisted["conflict"]["reason_code"]
        assert persisted["conflict"]["adjudicator_score"] == 0.9
        assert persisted["conflict"]["adjudicator_probability"] > 0.8
        assert persisted["conflict"]["evidence_fields"] == ["abstract"]
        assert persisted["conflict"]["qwen_calibrator_hash"] == (
            released.profile.adjudicator_calibration.calibrator.hash()
        )
        assert persisted["conflict"]["qwen_threshold_hash"] == (
            released.profile.adjudicator_calibration.threshold.hash()
        )


def test_release_rejects_artifact_drift_failed_gate_and_nonlocal_endpoint(tmp_path) -> None:
    release_path, plan = _release_bundle(tmp_path)
    document = json.loads(release_path.read_text())
    document["release_gate"]["passed"] = False
    release_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(Stage2ReleaseError, match="configuration does not match QueryPlan"):
        load_stage2_release(release_path, plan)

    release_path, plan = _release_bundle(tmp_path / "remote", base_url="https://models.example.test")
    with pytest.raises(Stage2ReleaseError, match="local|cloud"):
        load_stage2_release(release_path, plan)

    release_path, plan = _release_bundle(tmp_path / "drift")
    (release_path.parent / "reranker-threshold.json").write_text("{}", encoding="utf-8")
    with pytest.raises(Stage2ReleaseError, match="drifted"):
        load_stage2_release(release_path, plan)


def test_release_rejects_legacy_raw_score_threshold_manifest(tmp_path) -> None:
    release_path, plan = _release_bundle(tmp_path)
    document = json.loads(release_path.read_text())
    document["thresholds"] = {"path": "legacy-thresholds.json", "sha256": "0" * 64}
    release_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(Stage2ReleaseError, match="legacy raw-score"):
        load_stage2_release(release_path, plan)


def test_release_rejects_hash_only_gate_claims_and_extra_fields(tmp_path) -> None:
    release_path, plan = _release_bundle(tmp_path / "hash-only")
    document = json.loads(release_path.read_text())
    document["release_gate"]["artifact_hashes"] = {"promotion": "a" * 64}
    release_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(Stage2ReleaseError, match="configuration does not match QueryPlan"):
        load_stage2_release(release_path, plan)

    release_path, plan = _release_bundle(tmp_path / "top-extra")
    document = json.loads(release_path.read_text())
    document["unbound_override"] = True
    release_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(Stage2ReleaseError, match="fields are not exact"):
        load_stage2_release(release_path, plan)

    release_path, plan = _release_bundle(tmp_path / "runtime-extra")
    document = json.loads(release_path.read_text())
    document["runtime"]["cloud_fallback"] = True
    release_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(Stage2ReleaseError, match="runtime fields are not exact"):
        load_stage2_release(release_path, plan)

    release_path, plan = _release_bundle(tmp_path / "ref-extra")
    document = json.loads(release_path.read_text())
    document["calibration"]["reranker"]["calibrator"]["download_url"] = "https://example.test"
    release_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(Stage2ReleaseError, match="artifact reference fields are not exact"):
        load_stage2_release(release_path, plan)


def test_release_rejects_rehashed_gate_with_mismatched_provenance_or_extra_fields(tmp_path) -> None:
    release_path, plan = _release_bundle(tmp_path / "manifest-mismatch")
    document = json.loads(release_path.read_text())
    document["release_gate"]["candidate_id"] = "different-candidate"
    release_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(Stage2ReleaseError, match="configuration does not match QueryPlan"):
        load_stage2_release(release_path, plan)

    release_path, plan = _release_bundle(tmp_path / "gate-extra")
    document = json.loads(release_path.read_text())
    document["release_gate"]["unbound_metric"] = 1
    release_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(Stage2ReleaseError, match="configuration does not match QueryPlan"):
        load_stage2_release(release_path, plan)


def test_release_requires_an_untampered_approved_query_plan(tmp_path) -> None:
    release_path, plan = _release_bundle(tmp_path)
    tampered = json.loads(json.dumps(plan))
    tampered["filter"]["profile"] = "unapproved-profile"

    with pytest.raises(Stage2ReleaseError, match="exact approved QueryPlan"):
        load_stage2_release(release_path, tampered)


def test_gate_bytes_are_bound_by_plan_approval_and_symlinks_cannot_escape_bundle(tmp_path) -> None:
    release_path, plan = _release_bundle(tmp_path / "reformatted")
    document = json.loads(release_path.read_text())
    document["release_gate"]["evidence"]["sha256"] = "1" * 64
    release_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(Stage2ReleaseError, match="configuration does not match QueryPlan"):
        load_stage2_release(release_path, plan)


@pytest.mark.parametrize("disguised_version", ("oMLX 0.2.7", "0.5", "0.5.7-rc1", 0.57))
def test_release_rejects_disguised_or_noncanonical_omlx_versions(tmp_path, disguised_version) -> None:
    release_path, plan = _release_bundle(tmp_path)
    document = json.loads(release_path.read_text())
    lock_path = release_path.parent / document["reranker_lock"]["path"]
    lock = json.loads(lock_path.read_text())
    lock["omlx_version"] = disguised_version
    lock_path.write_text(json.dumps(lock, sort_keys=True), encoding="utf-8")
    document["reranker_lock"]["sha256"] = sha256(lock_path.read_bytes()).hexdigest()
    release_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(Stage2ReleaseError, match="strict MAJOR.MINOR.PATCH"):
        load_stage2_release(release_path, plan)


def test_stage2_search_telemetry_aggregates_counts_and_keeps_run_peaks(tmp_path) -> None:
    class RerankedDecision:
        reason_code = "reranker_threshold"

    class DeterministicDecision:
        reason_code = "document_type_included:article"

    def decisions(reranked: int):
        return tuple(RerankedDecision() for _ in range(reranked)) + tuple(
            DeterministicDecision() for _ in range(100 - reranked)
        )

    with Database(tmp_path / "telemetry.sqlite3") as database:
        database.migrate()
        screener = Stage2SearchScreener(database, object(), "campaign")
        screener.run_ids.extend(("stage2-a", "stage2-b"))
        screener.summaries.update({
            "stage2-a": Stage2Summary(
                decisions(80),
                80, 15, 0.15, (), 0, 0.0,
            ),
            "stage2-b": Stage2Summary(
                decisions(90),
                90, 31, 0.31,
                (ADJUDICATOR_SHARE_ALARM,), 1, 0.01,
            ),
        })

        telemetry = screener.telemetry()

    assert telemetry["stage2_run_ids"] == ["stage2-a", "stage2-b"]
    assert telemetry["screened_count"] == 200
    assert telemetry["reranked_count"] == 170
    assert telemetry["adjudicator_count"] == 46
    assert telemetry["adjudicator_share"] == 0.23
    assert telemetry["max_run_adjudicator_share"] == 0.31
    assert telemetry["adjudicator_capacity"] == "severe"
    assert telemetry["error_count"] == 1
    assert telemetry["error_rate"] == 0.005
    assert telemetry["alarm_codes"] == [
        ADJUDICATOR_SHARE_ALARM,
        ERROR_RATE_ALARM,
    ]


def test_stage2_search_error_alarm_uses_campaign_aggregate_rate(tmp_path) -> None:
    class TechnicalDecision:
        reason_code = "reranker_backend_failure"

    class SuccessfulDecision:
        reason_code = "reranker_threshold"

    with Database(tmp_path / "aggregate-errors.sqlite3") as database:
        database.migrate()
        screener = Stage2SearchScreener(database, object(), "campaign")
        screener.run_ids.extend(("stage2-small", "stage2-large"))
        screener.summaries.update({
            "stage2-small": Stage2Summary(
                (TechnicalDecision(),), 1, 0, 0.0, (), 1, 1.0,
            ),
            "stage2-large": Stage2Summary(
                tuple(SuccessfulDecision() for _ in range(1000)),
                1000, 0, 0.0, (), 0, 0.0,
            ),
        })

        telemetry = screener.telemetry()

    assert telemetry["error_count"] == 1
    assert telemetry["error_rate"] == pytest.approx(1 / 1001)
    assert telemetry["max_run_error_rate"] == 1.0
    assert ERROR_RATE_ALARM not in telemetry["alarm_codes"]
    assert ERROR_RATE_ALARM in telemetry["run_details"][0]["alarm_codes"]
