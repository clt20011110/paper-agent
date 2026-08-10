from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import subprocess
from types import SimpleNamespace

import pytest

import paper_agent.resources as resources_module
from paper_agent import __version__
from paper_agent.canonical import content_hash
from paper_agent.doctor import DoctorCheck, DoctorPaths, SystemDoctor, SystemDoctorReport
from paper_agent.grants import GrantStore
from paper_agent.manifests import load_catalog
from paper_agent.storage import Database


ROOT = Path(__file__).resolve().parents[1]


def _runner(argv):
    command = tuple(argv)
    if command[-1] == "--version":
        version = "0.5.7\n" if Path(command[0]).name == "omlx" else "codex 1.2.3\n"
        return subprocess.CompletedProcess(command, 0, version, "")
    if command[-2:] == ("login", "status"):
        return subprocess.CompletedProcess(command, 0, "logged in\n", "")
    if command[-2:] == ("debug", "models"):
        return subprocess.CompletedProcess(command, 0, '{"models":[{"slug":"gpt-5.6-luna"},{"slug":"gpt-5.6-sol"}]}', "")
    raise AssertionError(command)


def _executable(name: str) -> str:
    return f"/fake/bin/{name}"


def _paths(tmp_path: Path, **changes) -> DoctorPaths:
    values = {
        "repository_root": ROOT,
        "database_path": tmp_path / "papers.sqlite",
        "model_lock_paths": (
            ROOT / "configs/stage2/models/bge-reranker-v2-m3-fp32.lock.json",
            ROOT / "configs/stage2/models/qwen3.5-9b-8bit.lock.json",
        ),
    }
    values.update(changes)
    return DoctorPaths(**values)


def _doctor(tmp_path: Path, **changes) -> SystemDoctor:
    return SystemDoctor(
        _paths(tmp_path, **changes),
        command_runner=_runner,
        executable_finder=_executable,
        environment={},
        disk_usage=lambda _: type("Disk", (), {"free": 2_000_000_000})(),
    )


def _check(report, name: str):
    return next(item for item in report.checks if item.name == name)


def test_production_ready_requires_positive_evidence_not_only_no_blocker() -> None:
    report = SystemDoctorReport((
        DoctorCheck("verified", "pass", True, "ok", True),
        DoctorCheck("optional", "warning", False, "not needed", False),
    ))
    assert report.ready
    assert report.production_ready

    unproved = SystemDoctorReport((
        DoctorCheck("model", "warning", False, "listed only", True),
    ))
    assert unproved.ready
    assert not unproved.production_ready
    assert unproved.as_dict()["production_ready"] is False


def test_default_doctor_is_offline_and_reports_optional_uninitialized_database(tmp_path: Path) -> None:
    report = _doctor(tmp_path).run()

    assert report.ready
    assert _check(report, "database").status == "warning"
    assert _check(report, "provider_catalog").status == "pass"
    assert _check(report, "stage2_model_locks").status == "pass"
    assert _check(report, "codex").status == "warning"
    assert report.ready
    assert not report.production_ready
    assert not (tmp_path / "papers.sqlite").exists()


def test_installed_doctor_defaults_to_versioned_packaged_model_locks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "installed"
    asset_root = data_root / "share" / "paper-agent" / __version__
    installed_locks = (
        asset_root
        / "configs/stage2/models/bge-reranker-v2-m3-fp32.lock.json",
        asset_root
        / "configs/stage2/models/qwen3.5-9b-8bit.lock.json",
    )
    for source, installed in zip(_paths(tmp_path).model_lock_paths, installed_locks):
        installed.parent.mkdir(parents=True, exist_ok=True)
        installed.write_bytes(source.read_bytes())
    monkeypatch.setattr(resources_module, "_source_checkout_root", lambda: None)
    monkeypatch.setattr(
        resources_module.sysconfig,
        "get_path",
        lambda name: str(data_root) if name == "data" else None,
    )

    paths = DoctorPaths.defaults()
    check = SystemDoctor(paths)._model_locks(None, None)

    assert paths.repository_root == asset_root
    assert paths.model_lock_paths == installed_locks
    assert check.status == "pass"


def test_disk_check_uses_nearest_existing_ancestor_for_nested_database(tmp_path: Path) -> None:
    database_path = tmp_path / "missing" / "nested" / "papers.sqlite"
    probes: list[Path] = []
    doctor = SystemDoctor(
        _paths(tmp_path, database_path=database_path),
        disk_usage=lambda path: (
            probes.append(Path(path))
            or type("Disk", (), {"free": 2_000_000_000})()
        ),
    )

    check = doctor._disk()

    assert check.status == "pass"
    assert probes == [tmp_path]
    assert not database_path.parent.exists()


def test_disk_check_preserves_real_disk_usage_errors(tmp_path: Path) -> None:
    def unavailable(_path: str | Path):
        raise OSError("filesystem probe failed")

    check = SystemDoctor(_paths(tmp_path), disk_usage=unavailable)._disk()

    assert check.status == "blocker"
    assert check.required
    assert check.detail == "filesystem probe failed"


def test_database_version_is_checked_read_only(tmp_path: Path) -> None:
    database = tmp_path / "papers.sqlite"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE schema_migrations(version INTEGER)")
    connection.execute("INSERT INTO schema_migrations VALUES (999)")
    connection.commit()
    connection.close()

    check = _check(_doctor(tmp_path).run(), "database")

    assert check.status == "blocker"
    assert "newer" in check.detail


def test_config_provider_credentials_and_snapshot_are_explicit_blockers(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text("{}")
    config = tmp_path / "config.yaml"
    config.write_text(
        """version: 2
project: {topic: x, output_dir: out, report_language: zh-CN}
storage: {sqlite_path: papers.sqlite, wal: true, exports: []}
sources:
  approved_plan: {input_path: plan.json, content_hash: null, required: false}
  plugin_allowlist: []
  plan_defaults:
    required_roles: [search]
    provider_policy: all_resolved
    user_seeds: {inputs: []}
    venues: []
    arxiv: {enabled: false, roles: [], categories: [], include_arxiv_candidates: false, use_matched_arxiv_as_download_source: false, global_min_interval_seconds: 3}
    providers:
      ieee_xplore: {enabled: true, roles: [search], mode: snapshot, snapshot_path: missing.json}
    verification: {prefer_formal_version: true, require_two_sources_when_feasible: true, preserve_conflicts: true}
    citation_snowball: {enabled: false, directions: [], max_depth: 0, max_rounds: 0, seed_selector: x, seeds_per_subquestion: 1, max_per_seed_per_source: 1, max_candidates_per_round: 1, max_seconds_per_round: 1}
    saturation: {min_unique_included_yield: 0, consecutive_low_yield_rounds: 1, max_candidates: 1, max_requests: 1}
filter:
  mode: cascade
  deterministic: {include: [], exclude_document_types: []}
  reranker: {backend: omlx_rerank, model: bge, source_repo: bge, source_revision: abc, format: fp32, document_batch_size: 32, candidate_batch_sizes: [32], max_in_flight: 2, thresholds_artifact: thresholds.json}
  adjudicator: {backend: omlx_chat, model: qwen, revision: abc, chat_template_kwargs: {enable_thinking: false}, temperature: 0, seed: 1, stream: false, max_tokens: 1, max_context_window: 1, structured_output: {transport: extra_body.structured_outputs.json, schema: filter-decision.schema.json}, client_concurrency: 1, server_max_concurrent_requests: 1, benchmark_concurrency_pairs: [[1, 1]], expected_max_share: 1}
  fail_open: true
download:
  include_supplements: false
  resolvers: []
  providers: []
  purpose: x
  policy_matrix: policies/download-access-v1.yaml
  require_access_basis: true
  treat_unknown_license_as_open: false
  authorized_skill: {enabled: false, skill_name: download-authorized-papers, authorization_grant_id: null, data_sharing_grant_id: null, profile: stage3_authorized_luna, codex_model: gpt-5.6-luna, reasoning_effort: low, grant_defaults: {source_zip_sha256: null, installed_content_sha256: null, dependency_lock_sha256: null, allowed_domains: [], paper_ids: [], collection_snapshot_hash: null, selection_snapshot_hash: null, max_papers: null, actions: [], purpose: x, mode: attended, allow_unattended: false, authorization_expires_at: null}}
analysis: {profile: stage4_analysis_luna, provider: codex_exec, model: gpt-5.6-luna, reasoning_effort: medium, sandbox: read_only, network: false, output_schema: paper-analysis.schema.json, workers: 1, allow_abstract_only: true, remote_model_processing: {policy_matrix: policies/artifact-processing-v1.yaml, processing_grant_id: null}}
summary:
  enabled: false
  profile: stage4b_summary_sol
  provider: codex_exec
  model: gpt-5.6-sol
  reasoning_effort: high
  sandbox: read_only
  network: false
  schemas: {planning_assist: a, section_reduce: a, cross_section_reduce: a, final_reduce: a, quality_audit: a, repair: a}
  prompts: {planning_assist: a, section_reduce: a, cross_section_reduce: a, final_reduce: a, quality_audit: a, repair: a}
  format: markdown
  language: zh-CN
  report_plan: {input_path: null, content_hash: null, required_for_unattended: true, classification_axes: [x]}
  require_search_audit: true
  require_complete_coverage: true
  require_claim_evidence: true
  semantic_chunking: true
  remote_model_processing: {policy_matrix: policies/artifact-processing-v1.yaml, processing_grant_id: null}
  citations: {marker: stable_paper_id, style: x, bibliography_from_canonical_metadata: true}
  final_audit: {deterministic: true, independent_sol_session: true, rubric: policies/report-audit-rubric-v1.yaml, max_blocker_findings: 0, max_major_findings: 0, max_repair_calls: 1, reverify_and_reaudit_after_repair: true}
  immutable_run_directories: true
  update_latest_after_pass: true
  emit_incremental_diff: true
"""
    )

    report = _doctor(tmp_path, config_path=config).run()

    assert _check(report, "provider:ieee_xplore:credentials").status == "blocker"
    assert _check(report, "provider:ieee_xplore:snapshot").status == "blocker"


def test_omlx_endpoint_is_not_invented_without_a_validated_release(tmp_path: Path) -> None:
    calls: list[str] = []
    doctor = SystemDoctor(
        _paths(tmp_path), command_runner=_runner, executable_finder=_executable,
        http_probe=lambda endpoint: (calls.append(endpoint) or (200, "ok")),
        disk_usage=lambda _: type("Disk", (), {"free": 2_000_000_000})(),
    )
    report = doctor.run()

    assert _check(report, "omlx").status == "warning"
    assert calls == []


@pytest.mark.parametrize("version", ["oMLX 0.2.7\n", "development\n"])
def test_omlx_old_or_unparseable_version_is_a_blocker(tmp_path: Path, version: str) -> None:
    doctor = SystemDoctor(
        _paths(tmp_path),
        command_runner=lambda argv: subprocess.CompletedProcess(argv, 0, version, ""),
        executable_finder=_executable,
    )

    check = doctor._omlx(None)

    assert check.status == "blocker"
    assert check.production_required


def test_omlx_requires_2xx_and_a_matching_model_inventory(tmp_path: Path) -> None:
    released = SimpleNamespace(
        omlx_base_url="http://127.0.0.1:8000",
        profile=SimpleNamespace(
            reranker_model_id="bge-reranker-v2-m3",
            adjudicator_model_id="qwen3.5-9b-8bit",
        ),
    )
    denied = SystemDoctor(
        _paths(tmp_path), command_runner=_runner, executable_finder=_executable,
        http_probe=lambda _: (401, "unauthorized"),
    )
    assert denied._omlx(released).status == "blocker"  # type: ignore[arg-type]

    incomplete = SystemDoctor(
        _paths(tmp_path), command_runner=_runner, executable_finder=_executable,
        http_probe=lambda _: (200, json.dumps({"data": [{"id": "bge-reranker-v2-m3"}]})),
    )
    assert incomplete._omlx(released).status == "blocker"  # type: ignore[arg-type]

    available = SystemDoctor(
        _paths(tmp_path), command_runner=_runner, executable_finder=_executable,
        http_probe=lambda _: (
            200,
            json.dumps({"data": [
                {"id": "bge-reranker-v2-m3"}, {"id": "qwen3.5-9b-8bit"},
            ]}),
        ),
    )
    assert available._omlx(released).status == "pass"  # type: ignore[arg-type]


def test_empty_or_malformed_model_locks_fail_closed(tmp_path: Path) -> None:
    empty = _doctor(tmp_path, model_lock_paths=())._model_locks(None, None)
    assert empty.status == "blocker"

    documents = []
    for source in _paths(tmp_path).model_lock_paths:
        document = json.loads(source.read_text())
        documents.append(document)
    documents[0]["file_hashes"]["model.safetensors"] = "not-a-digest"
    paths = []
    for index, document in enumerate(documents):
        path = tmp_path / f"model-{index}.lock.json"
        path.write_text(json.dumps(document))
        paths.append(path)

    malformed = _doctor(tmp_path, model_lock_paths=tuple(paths))._model_locks(None, None)
    assert malformed.status == "blocker"
    assert "non-SHA-256" in malformed.detail


def test_configured_model_ids_and_revisions_must_match_exact_locks(tmp_path: Path) -> None:
    config = {
        "filter": {
            "reranker": {
                "backend": "omlx_rerank",
                "model": "BAAI/bge-reranker-v2-m3",
                "source_repo": "BAAI/bge-reranker-v2-m3",
                "source_revision": "wrong",
                "format": "fp32",
            },
            "adjudicator": {
                "backend": "omlx_chat",
                "model": "mlx-community/Qwen3.5-9B-8bit",
                "revision": "16daa4818c54ce5f5436f929d52542eb65bbed9d",
            },
        },
    }

    check = _doctor(tmp_path)._model_locks(config, None)

    assert check.status == "blocker"
    assert "source_revision" in check.detail


def test_configured_roles_must_be_declared_and_required_roles_must_resolve(tmp_path: Path) -> None:
    config = {
        "sources": {
            "plan_defaults": {
                "required_roles": ["citation"],
                "providers": {
                    "crossref": {"enabled": True, "roles": ["citation"], "mode": "api"},
                },
            },
            "plugin_allowlist": [],
        },
    }

    checks = _doctor(tmp_path)._catalog_and_providers(config)

    assert next(check for check in checks if check.name == "provider:crossref:roles").status == "blocker"
    assert next(check for check in checks if check.name == "provider_roles").status == "blocker"


def test_auto_provider_policy_is_unresolved_not_silently_disabled(tmp_path: Path) -> None:
    config = {
        "sources": {
            "plan_defaults": {
                "required_roles": ["search"],
                "providers": {
                    "dblp": {"enabled": "auto_for_cs", "roles": ["search"], "mode": "api"},
                },
            },
            "plugin_allowlist": [],
        },
    }

    checks = _doctor(tmp_path)._catalog_and_providers(config)

    assert next(check for check in checks if check.name == "provider:dblp").status == "warning"
    assert next(check for check in checks if check.name == "provider_roles").status == "warning"


class _NoImportEntryPoint:
    group = "paper_agent.providers"
    name = "example"
    value = "example_plugin:factory"

    def load(self):
        raise AssertionError("doctor must never import a provider entry point")


class _Distribution:
    def __init__(self, root: Path) -> None:
        self.metadata = {"Name": "example-plugin"}
        self.version = "1.2.3"
        self.files = (Path("plugin.py"),)
        self.entry_points = (_NoImportEntryPoint(),)
        self.root = root

    def locate_file(self, path: Path) -> Path:
        return self.root / path


def test_plugin_allowlist_verifies_installed_metadata_digest_before_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = tmp_path / "plugin.py"
    plugin.write_text("PLUGIN = True\n")
    distribution = _Distribution(tmp_path)
    digest = sha256(b"plugin.py" + plugin.read_bytes()).hexdigest()
    catalog = load_catalog()
    catalog.providers["example"] = {
        **catalog.providers["alphaxiv"],
        "provider": "example",
        "distribution": "example-plugin",
        "version": "1.2.3",
        "entry_point": "example_plugin:factory",
        "artifact_sha256": digest,
        "enabled": True,
        "builtin": False,
    }
    config = {
        "sources": {
            "plugin_allowlist": [{
                "distribution": "example-plugin", "version": "1.2.3", "provider": "example",
                "entry_point": "example_plugin:factory", "artifact_sha256": digest,
            }],
        },
    }
    monkeypatch.setattr("paper_agent.doctor.metadata.distribution", lambda _: distribution)

    check = _doctor(tmp_path)._plugin_checks(config, catalog)[0]
    assert check.status == "pass"

    distribution.version = "1.2.4"
    drifted = _doctor(tmp_path)._plugin_checks(config, catalog)[0]
    assert drifted.status == "blocker"
    assert "drifted" in drifted.detail


def test_plugin_allowlist_rejects_missing_distribution(tmp_path: Path) -> None:
    catalog = load_catalog()
    config = {
        "sources": {
            "plugin_allowlist": [{
                "distribution": "missing", "version": "1", "provider": "ghost",
                "entry_point": "ghost:factory", "artifact_sha256": "a" * 64,
            }],
        },
    }

    check = _doctor(tmp_path)._plugin_checks(config, catalog)[0]

    assert check.status == "blocker"
    assert "no trusted provider manifest" in check.detail


def test_codex_catalog_listing_is_not_production_availability(tmp_path: Path) -> None:
    listed = _doctor(tmp_path)._codex()
    assert listed.status == "warning"
    assert listed.production_required

    def proving_runner(argv):
        command = tuple(argv)
        if command[-1] == "--version":
            return subprocess.CompletedProcess(command, 0, "codex 1.2.3\n", "")
        if command[-2:] == ("login", "status"):
            return subprocess.CompletedProcess(command, 0, "logged in\n", "")
        if command[-2:] == ("debug", "models"):
            return subprocess.CompletedProcess(
                command, 0,
                '{"models":[{"slug":"gpt-5.6-luna"},{"slug":"gpt-5.6-sol"}]}', "",
            )
        output_path = Path(command[command.index("-o") + 1])
        output_path.write_text('{"ok":true}')
        return subprocess.CompletedProcess(command, 0, json.dumps({"model": command[3]}) + "\n", "")

    proved = SystemDoctor(
        _paths(tmp_path), command_runner=proving_runner, executable_finder=_executable,
        prove_codex_models=True,
    )._codex()
    assert proved.status == "pass"

    def misleading_auth(argv):
        command = tuple(argv)
        if command[-1] == "--version":
            return subprocess.CompletedProcess(command, 0, "codex 1.2.3\n", "")
        if command[-2:] == ("login", "status"):
            return subprocess.CompletedProcess(command, 0, "Not logged in\n", "")
        return subprocess.CompletedProcess(
            command, 0,
            '{"models":[{"slug":"gpt-5.6-luna"},{"slug":"gpt-5.6-sol"}]}', "",
        )

    auth = SystemDoctor(
        _paths(tmp_path), command_runner=misleading_auth, executable_finder=_executable,
    )._codex()
    assert auth.status == "blocker"


@dataclass
class _SkillResult:
    ready: bool = True
    reasons: tuple[str, ...] = ()
    installed_content_sha256: str = "a" * 64
    dependency_lock_sha256: str = "b" * 64


class _SkillRuntime:
    def doctor(self) -> _SkillResult:
        return _SkillResult()


def _authorized_config(grant_id: str | None) -> dict[str, object]:
    return {
        "download": {
            "purpose": "personal_research",
            "authorized_skill": {
                "enabled": True,
                "authorization_grant_id": grant_id,
                "data_sharing_grant_id": None,
            },
        },
    }


def test_enabled_authorized_skill_requires_active_digest_bound_grant_and_session(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "papers.sqlite"
    database = Database(database_path)
    database.migrate()
    store = GrantStore(database)
    draft = store.create_draft(
        grant_id="download-grant",
        kind="download",
        actions=["download", "store"],
        purpose="personal_research",
        mode="attended",
        scope={
            "paper_ids": ["paper-1"], "artifact_hashes": [], "collection_ids": [],
            "collection_snapshot_hash": None, "selection_snapshot_hash": None,
            "domains": ["publisher.example"], "provider": None, "model": None,
            "data_categories": [],
        },
        max_papers=1,
        expires_at="2099-01-01T00:00:00Z",
        skill_digest="a" * 64,
        dependency_digest="b" * 64,
    )
    store.approve(draft, draft["content_hash"], approved_by="owner", approved_at="2026-08-10T00:00:00Z")
    database.close()

    paths = _paths(
        tmp_path,
        database_path=database_path,
        authorized_skill_runtime=_SkillRuntime(),  # type: ignore[arg-type]
    )
    doctor = SystemDoctor(
        paths, browser_session_probe=lambda: (True, "visible authorized session available"),
        now=datetime(2026, 8, 10, tzinfo=UTC),
    )
    checks = doctor._authorized_skill(_authorized_config("download-grant"))

    assert next(check for check in checks if check.name == "authorization_grant").status == "pass"
    assert next(check for check in checks if check.name == "authorized_browser_session").status == "pass"

    no_session = SystemDoctor(paths, now=doctor.now)._authorized_skill(_authorized_config("download-grant"))
    session = next(check for check in no_session if check.name == "authorized_browser_session")
    assert session.status == "warning"
    assert session.production_required

    reopened = Database(database_path)
    GrantStore(reopened).revoke(
        "download-grant", actor="owner", event_at="2026-08-10T00:01:00Z",
    )
    reopened.close()
    revoked = SystemDoctor(paths, now=doctor.now)._authorized_skill(
        _authorized_config("download-grant")
    )
    assert next(check for check in revoked if check.name == "authorization_grant").status == "blocker"

    missing = SystemDoctor(paths, now=doctor.now)._authorized_skill(_authorized_config(None))
    assert next(check for check in missing if check.name == "authorization_grant").status == "blocker"


def test_stage2_query_plan_and_release_must_be_supplied_together(tmp_path: Path) -> None:
    doctor = _doctor(tmp_path, query_plan_path=tmp_path / "QUERY_PLAN.json")

    check, released, plan = doctor._stage2_release(None)

    assert check.status == "blocker"
    assert released is plan is None
