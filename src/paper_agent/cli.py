"""Paper Agent command-line entry point."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
import re
import sqlite3
import stat
import sys
import sysconfig
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from . import __version__
from .authorized_skill_runtime import AuthorizedSkillRuntime, load_audit_record
from .analysis import ANALYSIS_PROFILE
from .analysis_cli_service import AnalysisCliService, load_analysis_input_manifest
from .artifacts import ArtifactStore
from .canonical import canonical_json, content_hash
from .citations import CitationRequest, SelectedSeed, schedule_requests
from .config import ConfigError, load_config, load_yaml
from .codex_exec import CodexExec
from .doctor import DoctorPaths, SystemDoctor
from .download_cli_service import (
    AuthorizedSkillHandoffOptions,
    Stage3DownloadResult,
    Stage3DownloadService,
    load_provider_terms,
)
from .downloads import DownloadScopeBinding, DownloadScopeSnapshotStore
from .domain import CitationEdgeType
from .exchange import (
    export_csv,
    export_jsonl,
    import_csv,
    import_jsonl,
    import_legacy_json,
    validate_export,
)
from .grants import (
    GrantStore,
    create_grant_draft,
    validate_grant_approval,
)
from .legacy import migrate_legacy_yaml, write_migrated
from .manifests import load_catalog
from .query_plan import (
    QueryPlanStore,
    approve_query_plan,
    assert_runtime_matches,
    compile_query_plan,
)
from .processing import ArtifactProcessingPolicy, ProcessingGate
from .providers.builtin import manifest_from_document
from .providers.builtin import create_builtin
from .providers.plugins import (
    PluginAllowlistEntry,
    PluginRegistry,
    plugin_allowlist_from_config,
)
from .report_artifacts import ReportArtifactStore
from .report_cli_service import (
    approve_report_plan_from_files,
    assert_report_plan_resource_binding,
    compile_report_plan_from_files,
    diff_report_runs,
    load_report_run_bundle,
    verify_report_run,
)
from .report_config import ReportRuntimeConfig
from .report_execution_service import ReportExecutionService
from .report_input_service import ReportInputRequest, ReportInputService
from .report_plan import ReportPlanBundle
from .repository import PaperRepository
from .http_transport import ControlledHTTPTransport
from .stage1 import (
    CensusCapturingAdapter,
    Stage1IncompleteError,
    Stage1Request,
    collect_stage1_metadata,
    venue_catalog_document,
    write_stage1_result,
)
from .search_execution import execute_search_plan, resolve_runtime_providers, seed_input
from .stage2_search import (
    Stage2ReleaseError,
    _load_stage2_benchmark_candidate_bytes,
    load_stage2_release,
)
from .stage2_annotation_artifacts import (
    STAGE2_ANNOTATION_RUBRIC_HASH,
    assemble_annotation_ledger,
    load_annotation_ledger,
    load_human_annotation_worklist,
    make_adjudication_worklist,
    make_human_annotation_worklist,
    write_annotation_ledger,
    write_human_annotation_worklist,
    write_private_gold_labels,
)
from .stage2_backends import UrlLibOmlxTransport
from .stage2_evaluation import load_gold_manifest, rationale_audit_gate
from .stage2_evaluator import (
    issue_hidden_promotion_from_payload,
    load_hidden_evaluator_private_key,
    validate_hidden_evaluator_private_key_trust,
    validate_hidden_evaluator_signing_trust,
    validate_hidden_promotion_payload,
)
from .stage2_hidden_attestation import load_hidden_evaluator_trust
from .stage2_promotion_artifacts import (
    run_promotion_evaluation,
    validate_promotion_public_evidence,
    validate_promotion_candidate_bundles,
)
from .stage2_rationale_workflow import (
    collect_rationale_source_artifacts,
    derive_rationale_audit_examples,
    freeze_rationale_audit,
    import_completed_rationale_audit,
    load_rationale_audit_manifest,
    load_rationale_worklist,
    rationale_audit_examples_from_document,
    rationale_audit_records_document,
    rationale_source_plan,
    write_frozen_rationale_audit,
    write_derived_rationale_examples_no_replace,
    write_rationale_records_no_replace,
    write_rationale_source_artifacts_no_replace,
)
from .stage2_release_assembly import (
    assemble_stage2_release,
    validate_stage2_release_assembly,
)
from .stage2_release_evidence_producer import (
    build_stage2_release_evidence_index_bytes,
    write_stage2_release_evidence_index,
)
from .stage2_tuning import (
    build_stage2_tuning_winner_document,
    write_stage2_tuning_winner,
)
from .stage2_sampling import (
    SamplingPolicy,
    build_gold_sampling,
    curated_annotations_from_decisions,
    load_curation_decisions,
    load_curation_receipt,
    load_curation_worklist,
    load_hidden_real_selection,
    load_gold_sampling_provenance,
    load_private_corpus_snapshot,
    load_private_sampling_annotations,
    make_curation_receipt,
    make_curation_worklist,
    select_hidden_real,
    write_curation_receipt,
    write_curation_worklist,
    write_gold_sampling_manifest,
    write_gold_sampling_provenance,
    write_hidden_real_selection,
    write_private_sampling_annotations,
)
from .stage2_commands import (
    build_hidden_promotion_submission,
    build_stage2_candidate,
    evaluate_benchmark_artifacts,
    filter_database,
    freeze_stage2_benchmark_manifests,
    freeze_stage2_dev_scores,
    measure_stage2_benchmark,
    freeze_stage2_parity_workload,
    run_stage2_parity,
    run_structured_replay,
)
from .search_audit import search_audit
from .seed_import import import_seeds, inputs_from_files, validate_seed_inputs
from .storage import Database
from .workflow import SequentialWorkflowOrchestrator, StopToken, load_workflow_manifest
from .workflow_adapters import default_stage_adapters
from .workflow_report_handoff import (
    WorkflowReportExecutionRequest,
    WorkflowReportHandoffRequest,
    WorkflowReportHandoffService,
)


class CliUsageError(ValueError):
    """A command-line usage error suitable for structured console output."""


_SUCCESS_STATUSES = frozenset({
    "approved",
    "complete",
    "draft",
    "passed",
    "ready",
    "revoked",
    "runtime_validated",
    "validated",
    "written",
})

_NON_SUCCESS_EVENT_STATUSES = frozenset({
    "blocked",
    "cancelled",
    "failed",
    "failed_terminal",
    "incomplete",
    "manual_required",
    "pending",
    "retryable",
    "running",
    "uncertain_terminal",
})

_STAGE2_PROMOTION_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_STAGE2_PROMOTION_DATE_TIME = re.compile(
    r"^\d{4}-\d{2}-\d{2}[Tt]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[Zz]|[+-]\d{2}:\d{2})$"
)


class _StructuredArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise CliUsageError(message)


def build_parser(*, structured_errors: bool = False) -> argparse.ArgumentParser:
    parser_class = _StructuredArgumentParser if structured_errors else argparse.ArgumentParser
    parser = parser_class(prog="paper-agent")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--dry-run", action="store_true")
    subcommands = parser.add_subparsers(dest="command", required=True)
    doctor_command = subcommands.add_parser("doctor", help="inspect the local runtime")
    doctor_command.add_argument("--database", type=Path)
    doctor_command.add_argument("--model-lock", action="append", default=[], type=Path)
    doctor_command.add_argument("--query-plan", type=Path)
    doctor_command.add_argument("--stage2-release", type=Path)
    doctor_command.add_argument(
        "--authorized-skill-root", action="append", default=[], type=Path,
        help="root to inspect for the installed authorized download skill",
    )
    doctor_command.add_argument(
        "--authorized-skill-zip", type=Path,
        help="original audited authorized-skill archive",
    )
    doctor_command.add_argument(
        "--authorized-skill-audit", type=Path,
        help="override the checked-in authorized-skill audit manifest",
    )
    doctor_command.add_argument(
        "--prove-paid-models", "--prove-codex-models", dest="prove_codex_models", action="store_true",
        help="explicitly invoke the frozen Codex model profiles to prove availability",
    )
    doctor_command.add_argument(
        "--production-ready", action="store_true",
        help="also require production-readiness checks to pass",
    )

    grant = subcommands.add_parser("grant", help="create and administer immutable authorization grants")
    grant_commands = grant.add_subparsers(dest="grant_command", required=True)
    grant_create = grant_commands.add_parser("create", help="write an unapproved grant draft")
    grant_create.add_argument("--kind", required=True, choices=("download", "browser_data_sharing", "remote_model_processing"))
    grant_create.add_argument("--output", "--draft", dest="output", required=True, type=Path)
    grant_create.add_argument("--database", type=Path)
    grant_create.add_argument("--grant-id")
    grant_create.add_argument("--action", action="append", default=[])
    grant_create.add_argument("--purpose")
    grant_create.add_argument("--mode", choices=("attended", "unattended"))
    grant_create.add_argument("--allow-unattended", action="store_true", default=None)
    _add_grant_scope_arguments(grant_create)
    grant_create.add_argument("--max-papers", type=int)
    grant_create.add_argument("--expires-at")
    grant_create.add_argument("--skill-digest")
    grant_create.add_argument("--dependency-digest")
    grant_create.add_argument("--lineage-hash")
    grant_approve = grant_commands.add_parser("approve", help="approve and persist a draft grant")
    grant_approve.add_argument("--grant", required=True, type=Path)
    grant_approve.add_argument("--hash", required=True)
    grant_approve.add_argument("--approved-by", "--actor", dest="approved_by", required=True)
    grant_approve.add_argument("--approved-at", required=True)
    grant_approve.add_argument("--database", type=Path)
    grant_revoke = grant_commands.add_parser("revoke", help="append a grant revocation event")
    grant_revoke.add_argument("grant_id")
    grant_revoke.add_argument("--actor", required=True)
    grant_revoke.add_argument("--at", "--event-at", "--revoked-at", dest="event_at", required=True)
    grant_revoke.add_argument("--database", type=Path)

    export = subcommands.add_parser("export", help="export canonical papers from SQLite")
    export.add_argument("--database", type=Path)
    export.add_argument("--format", required=True, choices=("jsonl", "csv"))
    export.add_argument("--output", required=True, type=Path)

    import_data = subcommands.add_parser(
        "import", help="import canonical JSONL/CSV or legacy JSON into SQLite"
    )
    import_data.add_argument("--database", type=Path)
    import_data.add_argument(
        "--format", required=True, choices=("jsonl", "csv", "legacy-json")
    )
    import_data.add_argument("--input", required=True, type=Path)

    migrate = subcommands.add_parser("migrate-config", help="convert a legacy YAML configuration to v2")
    migrate.add_argument("--input", required=True, type=Path)
    migrate.add_argument("--write", type=Path, help="write the converted v2 YAML after review")

    search = subcommands.add_parser("search", help="compile, approve, and replay frozen searches")
    search_commands = search.add_subparsers(dest="search_command", required=True)
    plan = search_commands.add_parser("plan", help="compile a QueryPlan draft from YAML")
    plan.add_argument("--input", required=True, type=Path)
    plan.add_argument("--output-root", required=True, type=Path)
    approve = search_commands.add_parser("approve", help="approve a draft QueryPlan by content hash")
    approve.add_argument("--plan", required=True, type=Path)
    approve.add_argument("--hash", required=True)
    approve.add_argument("--approved-by", required=True)
    approve.add_argument("--approved-at")
    run = search_commands.add_parser("run", help="execute an approved frozen search")
    run.add_argument("--plan", required=True, type=Path)
    run.add_argument("--database", type=Path)
    run.add_argument("--contact")
    run.add_argument("--snapshot", action="append", default=[], metavar="PROVIDER=PATH")
    run.add_argument("--stage2-release", type=Path, help="passed local Stage 2 release manifest")
    run.add_argument("--historical-replay", action="store_true")
    audit = search_commands.add_parser("audit", help="read a persisted search audit")
    audit.add_argument("--database", required=True, type=Path)
    audit.add_argument("--crawl-run-id", required=True)
    expand = search_commands.add_parser("expand-citations", help="plan a deterministic citation round")
    expand.add_argument("--plan", required=True, type=Path)
    expand.add_argument("--seeds", required=True, type=Path)
    expand.add_argument("--round-index", required=True, type=int)

    crawl = subcommands.add_parser("crawl", help="compatibility alias for venue descriptor discovery")
    crawl.add_argument("--venue", action="append", required=True)
    crawl.add_argument("--plan", type=Path)
    crawl.add_argument("--database", type=Path)
    crawl.add_argument("--contact")
    crawl.add_argument("--snapshot", action="append", default=[], metavar="PROVIDER=PATH")
    crawl.add_argument("--stage2-release", type=Path, help="passed local Stage 2 release manifest")
    crawl.add_argument("--historical-replay", action="store_true")
    stage1 = subcommands.add_parser(
        "stage1", help="collect complete venue metadata without Stage 2 or a QueryPlan"
    )
    stage1_commands = stage1.add_subparsers(dest="stage1_command", required=True)
    stage1_list = stage1_commands.add_parser(
        "list-venues", help="list registered venue identifiers and source boundaries"
    )
    stage1_list.add_argument("--type", choices=("conference", "journal", "repository"))
    stage1_collect = stage1_commands.add_parser(
        "collect", help="collect a venue-by-year metadata census"
    )
    stage1_collect.add_argument("--venue", action="append", required=True)
    stage1_collect.add_argument("--year-from", required=True, type=int)
    stage1_collect.add_argument("--year-to", required=True, type=int)
    stage1_collect.add_argument("--output", required=True, type=Path)
    stage1_collect.add_argument("--receipt", type=Path)
    stage1_collect.add_argument("--contact")
    stage1_collect.add_argument("--page-size", type=int, default=500)
    stage1_collect.add_argument("--max-workers", type=int, default=4)
    stage1_collect.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="publish records even when the receipt cannot prove completeness",
    )
    filter_command = subcommands.add_parser(
        "filter", help="screen canonical papers with an approved local Stage 2 release"
    )
    filter_command.add_argument("--plan", required=True, type=Path)
    filter_command.add_argument("--stage2-release", type=Path)
    filter_command.add_argument("--database", type=Path)
    filter_command.add_argument("--campaign-id")
    filter_command.add_argument("--paper-id", action="append", default=[])

    stage2_sampling = subcommands.add_parser(
        "stage2-sampling",
        help="build the frozen public Stage 2 gold manifest inside evaluator custody",
    )
    stage2_sampling_commands = stage2_sampling.add_subparsers(
        dest="stage2_sampling_command", required=True
    )
    stage2_sampling_freeze = stage2_sampling_commands.add_parser(
        "freeze-frame", help="freeze HIDDEN_REAL before opening curated annotations"
    )
    stage2_sampling_freeze.add_argument(
        "--private-snapshot", required=True, type=Path
    )
    stage2_sampling_freeze.add_argument("--output", required=True, type=Path)
    stage2_sampling_worklist = stage2_sampling_commands.add_parser(
        "curation-worklist",
        help="export evaluator-private curation candidates outside HIDDEN_REAL families",
    )
    stage2_sampling_worklist.add_argument("--private-snapshot", required=True, type=Path)
    stage2_sampling_worklist.add_argument("--hidden-real-freeze-frame", required=True, type=Path)
    stage2_sampling_worklist.add_argument("--output", required=True, type=Path)
    stage2_sampling_import = stage2_sampling_commands.add_parser(
        "curation-import",
        help="validate provisional curation decisions and write strict private annotations",
    )
    stage2_sampling_import.add_argument("--private-snapshot", required=True, type=Path)
    stage2_sampling_import.add_argument("--hidden-real-freeze-frame", required=True, type=Path)
    stage2_sampling_import.add_argument("--worklist", required=True, type=Path)
    stage2_sampling_import.add_argument("--decisions", required=True, type=Path)
    stage2_sampling_import.add_argument("--curated-annotations-output", required=True, type=Path)
    stage2_sampling_import.add_argument("--receipt-output", required=True, type=Path)
    stage2_sampling_build = stage2_sampling_commands.add_parser(
        "build", help="sample 600 pairs from frozen private inputs"
    )
    stage2_sampling_build.add_argument(
        "--private-snapshot", required=True, type=Path
    )
    stage2_sampling_build.add_argument(
        "--hidden-real-freeze-frame", required=True, type=Path
    )
    stage2_sampling_build.add_argument(
        "--curated-annotations", required=True, type=Path
    )
    stage2_sampling_build.add_argument("--curation-receipt", required=True, type=Path)
    stage2_sampling_build.add_argument(
        "--gold-manifest-output", required=True, type=Path
    )
    stage2_sampling_build.add_argument(
        "--provenance-output", required=True, type=Path
    )
    stage2_annotation_worklist = stage2_sampling_commands.add_parser(
        "annotation-worklist",
        help="export one blind 600-pair worklist for an independent annotator",
    )
    stage2_annotation_worklist.add_argument("--gold-manifest", required=True, type=Path)
    stage2_annotation_worklist.add_argument("--private-snapshot", required=True, type=Path)
    stage2_annotation_worklist.add_argument("--participant-id", required=True)
    stage2_annotation_worklist.add_argument("--output", required=True, type=Path)
    stage2_adjudication_worklist = stage2_sampling_commands.add_parser(
        "adjudication-worklist",
        help="export a blind worklist containing only completed-annotation disagreements",
    )
    stage2_adjudication_worklist.add_argument("--gold-manifest", required=True, type=Path)
    stage2_adjudication_worklist.add_argument("--private-snapshot", required=True, type=Path)
    stage2_adjudication_worklist.add_argument("--annotation-a", required=True, type=Path)
    stage2_adjudication_worklist.add_argument("--annotation-b", required=True, type=Path)
    stage2_adjudication_worklist.add_argument("--participant-id", required=True)
    stage2_adjudication_worklist.add_argument("--output", required=True, type=Path)
    stage2_assemble_annotations = stage2_sampling_commands.add_parser(
        "assemble-annotation-ledger",
        help="assemble human annotation handoffs into the private verified ledger",
    )
    stage2_assemble_annotations.add_argument("--gold-manifest", required=True, type=Path)
    stage2_assemble_annotations.add_argument("--private-snapshot", required=True, type=Path)
    stage2_assemble_annotations.add_argument("--curated-annotations", required=True, type=Path)
    stage2_assemble_annotations.add_argument("--sampling-provenance", required=True, type=Path)
    stage2_assemble_annotations.add_argument("--annotation-a", required=True, type=Path)
    stage2_assemble_annotations.add_argument("--annotation-b", required=True, type=Path)
    stage2_assemble_annotations.add_argument("--adjudication", required=True, type=Path)
    stage2_assemble_annotations.add_argument("--output", required=True, type=Path)
    stage2_sampling_finalize = stage2_sampling_commands.add_parser(
        "finalize-annotations",
        help="validate a double-annotation ledger and create private promotion labels",
    )
    stage2_sampling_finalize.add_argument(
        "--gold-manifest", required=True, type=Path
    )
    stage2_sampling_finalize.add_argument(
        "--annotation-ledger", required=True, type=Path
    )
    stage2_sampling_finalize.add_argument(
        "--private-labels-output", required=True, type=Path
    )

    stage2_rationale = subcommands.add_parser(
        "stage2-rationale",
        help="freeze and import an explicitly human-labelled rationale audit",
    )
    stage2_rationale_commands = stage2_rationale.add_subparsers(
        dest="stage2_rationale_command", required=True
    )
    rationale_source = stage2_rationale_commands.add_parser(
        "run-source",
        help="select rationale strata and run the candidate's local Qwen adjudicator",
    )
    rationale_source.add_argument("--stage2-candidate", required=True, type=Path)
    rationale_source.add_argument("--benchmark-papers", required=True, type=Path)
    rationale_source.add_argument("--output-dir", required=True, type=Path)
    rationale_freeze = stage2_rationale_commands.add_parser(
        "freeze-worklist",
        help="freeze already-selected rationale examples before human review",
    )
    rationale_freeze.add_argument("--examples", required=True, type=Path)
    rationale_freeze.add_argument("--reviewer-id", required=True)
    rationale_freeze.add_argument("--manifest-output", required=True, type=Path)
    rationale_freeze.add_argument("--worklist-output", required=True, type=Path)
    rationale_derive = stage2_rationale_commands.add_parser(
        "derive-examples",
        help="derive auditable rationale examples from a bound frozen Qwen ledger",
    )
    rationale_derive.add_argument("--stage2-candidate", required=True, type=Path)
    rationale_derive.add_argument("--benchmark-papers", required=True, type=Path)
    rationale_derive.add_argument("--query-metadata", required=True, type=Path)
    rationale_derive.add_argument("--adjudication-ledger", required=True, type=Path)
    rationale_derive.add_argument("--output", required=True, type=Path)
    rationale_import = stage2_rationale_commands.add_parser(
        "import-worklist",
        help="validate explicit human booleans and emit raw rationale audit records",
    )
    rationale_import.add_argument("--manifest", required=True, type=Path)
    rationale_import.add_argument("--worklist", required=True, type=Path)
    rationale_import.add_argument("--records-output", required=True, type=Path)

    stage2_calibration = subcommands.add_parser(
        "stage2-calibration",
        help="freeze DEV model scores and build a calibrated benchmark candidate",
    )
    stage2_calibration_commands = stage2_calibration.add_subparsers(
        dest="stage2_calibration_command", required=True
    )
    stage2_freeze_scores = stage2_calibration_commands.add_parser(
        "freeze-dev-scores",
        help="score the exact unlabelled 300-pair DEV split with both local models",
    )
    stage2_freeze_scores.add_argument("--gold-manifest", required=True, type=Path)
    stage2_freeze_scores.add_argument("--private-snapshot", required=True, type=Path)
    stage2_freeze_scores.add_argument("--topic-queries", required=True, type=Path)
    stage2_freeze_scores.add_argument("--runtime", required=True, type=Path)
    stage2_freeze_scores.add_argument("--reranker-lock", required=True, type=Path)
    stage2_freeze_scores.add_argument("--adjudicator-lock", required=True, type=Path)
    stage2_freeze_scores.add_argument("--output", required=True, type=Path)
    stage2_build_candidate = stage2_calibration_commands.add_parser(
        "build-candidate",
        help="join frozen DEV scores to authoritative human labels and calibrate",
    )
    stage2_build_candidate.add_argument("--gold-manifest", required=True, type=Path)
    stage2_build_candidate.add_argument("--private-labels", required=True, type=Path)
    stage2_build_candidate.add_argument("--raw-scores", required=True, type=Path)
    stage2_build_candidate.add_argument("--runtime", required=True, type=Path)
    stage2_build_candidate.add_argument("--reranker-lock", required=True, type=Path)
    stage2_build_candidate.add_argument("--adjudicator-lock", required=True, type=Path)
    stage2_build_candidate.add_argument("--candidate-id", required=True)
    stage2_build_candidate.add_argument("--output-dir", required=True, type=Path)

    parity = subcommands.add_parser(
        "stage2-parity", help="freeze and execute the trusted 10,000-pair FP32/BF16 parity gate"
    )
    parity_commands = parity.add_subparsers(dest="stage2_parity_command", required=True)
    parity_freeze = parity_commands.add_parser(
        "freeze-workload", help="freeze one fixed query assignment over exactly 10,000 papers"
    )
    parity_freeze.add_argument("--papers", required=True, type=Path)
    parity_freeze.add_argument("--selection-receipt", required=True, type=Path)
    parity_freeze.add_argument("--topic", required=True)
    parity_freeze.add_argument("--language", required=True)
    parity_freeze.add_argument("--query-version", required=True)
    parity_freeze.add_argument("--query", required=True)
    parity_freeze.add_argument("--output", required=True, type=Path)
    parity_run = parity_commands.add_parser(
        "run", help="run the frozen workload with the official FP32 oracle and schema-v2 BF16 candidate"
    )
    parity_run.add_argument("--workload", required=True, type=Path)
    parity_run.add_argument("--selection-receipt", required=True, type=Path)
    parity_run.add_argument("--oracle-stage2-candidate", required=True, type=Path)
    parity_run.add_argument("--oracle-model-lock", required=True, type=Path)
    parity_run.add_argument("--stage2-candidate", required=True, type=Path)
    parity_run.add_argument("--candidate-model-lock", required=True, type=Path)
    parity_run.add_argument("--manifest-output", required=True, type=Path)
    parity_run.add_argument("--scores-output", required=True, type=Path)

    replay = subcommands.add_parser(
        "stage2-replay",
        help="run the frozen 1,000-request structured-output gate against local oMLX",
    )
    replay.add_argument("--papers", required=True, type=Path)
    replay.add_argument(
        "--stage2-candidate",
        "--stage2-config",
        dest="stage2_candidate",
        required=True,
        type=Path,
    )
    replay.add_argument("--manifest-output", required=True, type=Path)
    replay.add_argument("--records-output", required=True, type=Path)

    benchmark = subcommands.add_parser(
        "benchmark-stage2", help="measure or evaluate frozen Stage 2 benchmark records"
    )
    benchmark.add_argument("--manifest", type=Path)
    benchmark.add_argument("--record", action="append", type=Path)
    benchmark.add_argument("--soak-manifest", type=Path)
    benchmark.add_argument("--soak-record", type=Path)
    benchmark_modes = benchmark.add_subparsers(dest="benchmark_command")
    measure = benchmark_modes.add_parser(
        "measure", help="execute one production-scale workload against local oMLX"
    )
    measure.add_argument("--manifest", required=True, type=Path)
    measure.add_argument("--papers", required=True, type=Path)
    measure.add_argument(
        "--stage2-candidate",
        "--stage2-config",
        dest="stage2_candidate",
        required=True,
        type=Path,
        help="frozen pre-throughput Stage 2 models, calibrations, thresholds, and runtime",
    )
    measure.add_argument(
        "--environment",
        required=True,
        type=Path,
        help="frozen environment metadata; hardware and macOS are verified locally",
    )
    measure.add_argument("--database", required=True, type=Path)
    measure.add_argument("--output", required=True, type=Path)
    measure.add_argument(
        "--scenario", required=True, choices=("normal", "stress", "soak")
    )
    measure.add_argument(
        "--omlx-pid",
        required=True,
        action="append",
        type=int,
        help="oMLX server or worker PID to include in current RSS; repeat for every process",
    )
    measure.add_argument("--sample-interval-seconds", type=float, default=0.25)
    freeze_benchmarks = benchmark_modes.add_parser(
        "freeze-manifests",
        help="bind frozen 1k/10k workloads to one schema-v2 candidate",
    )
    freeze_benchmarks.add_argument(
        "--stage2-candidate", required=True, type=Path
    )
    freeze_benchmarks.add_argument(
        "--performance-papers", required=True, type=Path
    )
    freeze_benchmarks.add_argument("--soak-papers", required=True, type=Path)
    freeze_benchmarks.add_argument(
        "--selection-receipt", required=True, type=Path
    )
    freeze_benchmarks.add_argument(
        "--performance-output", required=True, type=Path
    )
    freeze_benchmarks.add_argument("--soak-output", required=True, type=Path)

    evaluator = subcommands.add_parser(
        "stage2-evaluator",
        help="issue public-safe attestations from the isolated Stage 2 evaluator",
    )
    evaluator_commands = evaluator.add_subparsers(
        dest="stage2_evaluator_command", required=True
    )
    attest = evaluator_commands.add_parser(
        "attest",
        help="validate and Ed25519-sign a public-safe hidden-promotion payload",
    )
    attest.add_argument("--payload", required=True, type=Path)
    attest.add_argument("--signing-key-file", required=True, type=Path)
    attest.add_argument("--trust-manifest", required=True, type=Path)
    attest.add_argument("--output", required=True, type=Path)

    predict_hidden = evaluator_commands.add_parser(
        "predict-hidden",
        help="run one schema-v2 candidate three times over the sealed hidden set",
    )
    predict_hidden.add_argument("--manifest", required=True, type=Path)
    predict_hidden.add_argument("--private-snapshot", required=True, type=Path)
    predict_hidden.add_argument("--stage2-candidate", required=True, type=Path)
    predict_hidden.add_argument("--output", required=True, type=Path)

    promote = evaluator_commands.add_parser(
        "promote",
        help="run one sealed hidden promotion batch and sign one public-safe attestation",
    )
    promote.add_argument("--manifest", required=True, type=Path)
    promote.add_argument("--private-labels", required=True, type=Path)
    promote.add_argument("--candidate", action="append", required=True, metavar="ID=PATH")
    promote.add_argument("--submission", action="append", required=True, metavar="ID=PATH")
    promote.add_argument(
        "--public-evidence",
        action="append",
        required=True,
        metavar="ID=PATH",
        help="public-only Stage 2 evidence with evidence_type=stage2_public_promotion_evidence",
    )
    promote.add_argument("--incumbent-candidate-id", required=True)
    promote.add_argument("--evaluator-id", required=True)
    promote.add_argument("--evaluation-run-id", required=True)
    promote.add_argument("--state-root", required=True, type=Path)
    promote.add_argument("--evaluator-key-id", required=True)
    promote.add_argument("--issued-at", required=True)
    promote.add_argument("--trust-manifest", required=True, type=Path)
    promote.add_argument("--parity-oracle-trust", required=True, type=Path)
    promote.add_argument("--signing-key-file", required=True, type=Path)
    promote.add_argument("--output", required=True, type=Path)
    promote.add_argument(
        "--qualified-fallback-output",
        action="append",
        default=[],
        metavar="ID=PATH",
        help=(
            "sign an additional non-winner candidate that passed every release "
            "gate as a qualified fallback; repeat for multiple fallbacks"
        ),
    )
    promote.add_argument("--bootstrap-iterations", type=int, default=2_000)
    promote.add_argument("--bootstrap-seed", type=int, default=0)

    stage2_release = subcommands.add_parser(
        "stage2-release",
        help="validate or assemble a deployment-ready Stage 2 release",
    )
    stage2_release_commands = stage2_release.add_subparsers(
        dest="stage2_release_command", required=True
    )
    assemble = stage2_release_commands.add_parser(
        "assemble",
        help="verify every public and hidden gate and write a schema-v3 release",
    )
    assemble.add_argument("--candidate", required=True, type=Path)
    assemble.add_argument("--evidence", required=True, type=Path)
    assemble.add_argument("--trust-manifest", required=True, type=Path)
    assemble.add_argument("--parity-oracle-trust", required=True, type=Path)
    assemble.add_argument("--fallback-candidate", type=Path)
    assemble.add_argument("--fallback-evidence", type=Path)
    assemble.add_argument("--fallback-omlx-base-url")
    assemble.add_argument("--fallback-api-key-env")
    assemble.add_argument("--output", required=True, type=Path)
    evidence = stage2_release_commands.add_parser(
        "build-evidence",
        help="validate and write an immutable Stage 2 public or final evidence index",
    )
    evidence.add_argument("--candidate", required=True, type=Path)
    evidence.add_argument("--gold-manifest", required=True, type=Path)
    evidence.add_argument("--structured-manifest", required=True, type=Path)
    evidence.add_argument("--structured-records", required=True, type=Path)
    evidence.add_argument("--structured-papers", required=True, type=Path)
    evidence.add_argument("--rationale-manifest", required=True, type=Path)
    evidence.add_argument("--rationale-worklist", required=True, type=Path)
    evidence.add_argument("--rationale-records", required=True, type=Path)
    evidence.add_argument("--rationale-source-ledger", required=True, type=Path)
    evidence.add_argument("--rationale-query-metadata", required=True, type=Path)
    evidence.add_argument("--rationale-derived-examples", required=True, type=Path)
    evidence.add_argument("--rationale-papers", required=True, type=Path)
    evidence.add_argument("--parity-manifest", required=True, type=Path)
    evidence.add_argument("--parity-workload", required=True, type=Path)
    evidence.add_argument("--parity-selection-receipt", required=True, type=Path)
    evidence.add_argument("--parity-scores", required=True, type=Path)
    evidence.add_argument("--parity-oracle-model-lock", required=True, type=Path)
    evidence.add_argument("--parity-candidate-model-lock", required=True, type=Path)
    evidence.add_argument("--parity-oracle-calibrator", required=True, type=Path)
    evidence.add_argument("--parity-candidate-calibrator", required=True, type=Path)
    evidence.add_argument("--parity-oracle-threshold", required=True, type=Path)
    evidence.add_argument("--parity-candidate-threshold", required=True, type=Path)
    evidence.add_argument("--benchmark-manifest", required=True, type=Path)
    evidence.add_argument("--benchmark-papers", required=True, type=Path)
    evidence.add_argument("--benchmark-record", action="append", default=[], type=Path)
    evidence.add_argument("--soak-manifest", required=True, type=Path)
    evidence.add_argument("--soak-papers", required=True, type=Path)
    evidence.add_argument("--soak-record", required=True, type=Path)
    evidence.add_argument("--hidden-attestation", type=Path)
    evidence.add_argument("--output", required=True, type=Path)

    stage2_tuning = subcommands.add_parser(
        "stage2-tuning", help="select one production Stage 2 batch and concurrency configuration",
    )
    tuning_commands = stage2_tuning.add_subparsers(dest="stage2_tuning_command", required=True)
    tuning_select = tuning_commands.add_parser(
        "select", help="validate a frozen 3x3 measurement grid and write its winner",
    )
    tuning_select.add_argument("--input", required=True, type=Path)
    tuning_select.add_argument("--output", required=True, type=Path)

    report = subcommands.add_parser(
        "report", help="plan, approve, run, or compare Stage 4b reports"
    )
    report_mode = report.add_mutually_exclusive_group()
    report_mode.add_argument("--plan-only", action="store_true")
    report_mode.add_argument("--plan", type=Path)
    report_mode.add_argument("--diff-from")
    report.add_argument("--draft", type=Path)
    report.add_argument("--corpus-snapshot", type=Path)
    report.add_argument("--search-audit", type=Path)
    report.add_argument("--handoff-id")
    report.add_argument("--output-root", type=Path)
    report.add_argument("--report-run-id")
    report.add_argument("--pipeline-run-id")
    report.add_argument("--database", type=Path)
    report.add_argument("--artifact-root", type=Path)
    report.add_argument("--policy", type=Path)
    report.add_argument(
        "--processing-grant",
        action="append",
        default=[],
        metavar="ARTIFACT_HASH=GRANT_ID",
    )
    report.add_argument("--processing-grants", type=Path)
    report.add_argument("--previous-report-run-id")
    report.add_argument(
        "--execution-mode", choices=("attended", "unattended"), default="attended"
    )
    report_commands = report.add_subparsers(dest="report_command")
    report_inputs = report_commands.add_parser(
        "prepare-inputs",
        help="freeze Stage 4b inputs from persisted Stage 1, 2, and 4 runs",
    )
    report_inputs.add_argument("--database", type=Path)
    report_inputs.add_argument("--artifact-root", type=Path)
    report_inputs.add_argument("--output-root", type=Path)
    report_inputs.add_argument("--workflow-run-id")
    report_inputs.add_argument("--crawl-run-id")
    report_inputs.add_argument("--filter-run-id")
    report_inputs.add_argument("--stage4-run-id")
    report_inputs.add_argument("--recent-cutoff", required=True)
    report_inputs.add_argument("--created-at", required=True)
    report_inputs.add_argument("--include-needs-review", action="store_true")
    report_approve = report_commands.add_parser(
        "approve", help="approve and persist a frozen ReportPlan bundle"
    )
    report_approve.add_argument("--plan", required=True, type=Path)
    report_approve.add_argument("--hash", required=True)
    report_approve.add_argument("--approved-by", required=True)
    report_approve.add_argument("--approved-at")
    report_approve.add_argument("--corpus-snapshot", required=True, type=Path)
    report_approve.add_argument("--search-audit", required=True, type=Path)
    report_approve.add_argument("--output-root", required=True, type=Path)
    report_approve.add_argument("--handoff-id")
    report_approve.add_argument("--database", type=Path)
    report_approve.add_argument("--artifact-root", type=Path)
    report_approve.add_argument("--workflow-config", type=Path)
    report_approve.add_argument("--workflow-manifest", type=Path)
    report_approve.add_argument("--report-workflow-id")
    report_approve.add_argument("--report-workflow-run-id")
    report_approve.add_argument("--workflow-processing-grants", type=Path)
    report_approve.add_argument("--workflow-policy", type=Path)
    report_approve.add_argument("--previous-report-run-id")

    verify = subcommands.add_parser(
        "verify-report", help="run deterministic gates over a published report"
    )
    verify.add_argument("--output-root", required=True, type=Path)
    verify.add_argument("--report-run-id")

    download = subcommands.add_parser(
        "download", help="run the frozen Stage 3 resolver/probe/fetch chain"
    )
    download.add_argument("--database", type=Path)
    download.add_argument("--artifact-root", type=Path)
    download.add_argument("--paper-id", action="append", default=[])
    download.add_argument("--filter-run-id")
    download.add_argument("--include-needs-review", action="store_true")
    download.add_argument("--grant-id")
    download.add_argument("--collection-id")
    collection_snapshot = download.add_mutually_exclusive_group()
    collection_snapshot.add_argument("--collection-snapshot", type=Path)
    collection_snapshot.add_argument("--collection-snapshot-id")
    selection_snapshot = download.add_mutually_exclusive_group()
    selection_snapshot.add_argument("--selection-snapshot", type=Path)
    selection_snapshot.add_argument("--selection-snapshot-id")
    download.add_argument("--provider-terms", type=Path)
    download.add_argument("--authorized-skill-queue", type=Path)
    download.add_argument("--authorized-skill-output", type=Path)
    download.add_argument("--authorized-skill-root", action="append", default=[], type=Path)
    download.add_argument("--authorized-skill-zip", type=Path)
    download.add_argument("--authorized-skill-audit", type=Path)

    analyze = subcommands.add_parser(
        "analyze", help="run policy-gated Stage 4 Luna analysis"
    )
    analyze.add_argument("--database", type=Path)
    analyze.add_argument("--artifact-root", type=Path)
    analyze.add_argument("--input", required=True, type=Path)
    analyze.add_argument("--processing-grant-id")
    analyze.add_argument("--policy", type=Path)

    for command_name, help_text in (
        ("run", "execute a frozen multi-stage workflow manifest"),
        ("resume", "resume an interrupted frozen workflow manifest"),
    ):
        workflow = subcommands.add_parser(command_name, help=help_text)
        workflow.add_argument("--workflow", required=True, type=Path)
        workflow.add_argument("--database", type=Path)
        workflow.add_argument("--workflow-run-id")
    import_command = subcommands.add_parser("import-seeds", help="import authorized library seeds")
    import_command.add_argument("--database", required=True, type=Path)
    import_command.add_argument("--seed", action="append", default=[])
    import_command.add_argument("--input", action="append", default=[], type=Path)
    return parser


def main(
    argv: Sequence[str] | None = None, *, structured_errors: bool = False
) -> int:
    args = build_parser(structured_errors=structured_errors).parse_args(
        _runtime_argv(argv)
    )
    if args.command == "doctor":
        report = _doctor(args)
        ready = report.production_ready if args.production_ready else report.ready
        return _finish(
            args,
            {
                "command": "doctor",
                "paper_agent_version": __version__,
                "status": "ready" if ready else "blocked",
                **report.as_dict(),
            },
        )
    if args.command == "grant" and args.grant_command == "create":
        return _finish(args, _grant_create(args))
    if args.command == "grant" and args.grant_command == "approve":
        return _finish(args, _grant_approve(args))
    if args.command == "grant" and args.grant_command == "revoke":
        return _finish(args, _grant_revoke(args))
    if args.command == "export":
        return _finish(args, _export(args))
    if args.command == "import":
        return _finish(args, _import_data(args))
    if args.command == "migrate-config":
        return _finish(args, _migrate_config(args))
    if args.command == "search" and args.search_command == "plan":
        return _finish(
            args,
            _search_plan(
                args.input,
                args.output_root,
                config_path=args.config,
                dry_run=args.dry_run,
            ),
        )
    if args.command == "search" and args.search_command == "approve":
        return _finish(
            args,
            _search_approve(
                args.plan,
                args.hash,
                args.approved_by,
                args.approved_at,
                dry_run=args.dry_run,
            ),
        )
    if args.command == "search" and args.search_command == "run":
        return _finish(
            args,
            _search_run(
                args.plan,
                database_path=args.database,
                contact=args.contact,
                snapshot_values=args.snapshot,
                stage2_release_path=args.stage2_release,
                config_path=args.config,
                run_id=args.run_id,
                dry_run=args.dry_run,
                historical_replay=args.historical_replay,
            )
        )
    if args.command == "search" and args.search_command == "audit":
        return _finish(
            args,
            {"command": "search.audit", **search_audit(args.database, args.crawl_run_id)},
        )
    if args.command == "search" and args.search_command == "expand-citations":
        return _finish(
            args, _expand_citations(args.plan, args.seeds, args.round_index)
        )
    if args.command == "crawl":
        return _finish(
            args,
            _crawl(
                args.venue,
                plan_path=args.plan,
                database_path=args.database,
                contact=args.contact,
                snapshot_values=args.snapshot,
                stage2_release_path=args.stage2_release,
                config_path=args.config,
                run_id=args.run_id,
                dry_run=args.dry_run,
                historical_replay=args.historical_replay,
            )
        )
    if args.command == "stage1" and args.stage1_command == "list-venues":
        venues = venue_catalog_document()
        if args.type:
            venues = [venue for venue in venues if venue["venue_type"] == args.type]
        return _finish(
            args,
            {"command": "stage1.list-venues", "status": "complete", "venues": venues},
        )
    if args.command == "stage1" and args.stage1_command == "collect":
        return _finish(args, _stage1_collect(args))
    if args.command == "filter":
        return _finish(args, _filter(args))
    if (
        args.command == "stage2-sampling"
        and args.stage2_sampling_command == "freeze-frame"
    ):
        return _finish(args, _stage2_sampling_freeze_frame(args))
    if (
        args.command == "stage2-sampling"
        and args.stage2_sampling_command == "curation-worklist"
    ):
        return _finish(args, _stage2_curation_worklist(args))
    if (
        args.command == "stage2-sampling"
        and args.stage2_sampling_command == "curation-import"
    ):
        return _finish(args, _stage2_curation_import(args))
    if args.command == "stage2-sampling" and args.stage2_sampling_command == "build":
        return _finish(args, _stage2_sampling_build(args))
    if (
        args.command == "stage2-sampling"
        and args.stage2_sampling_command == "annotation-worklist"
    ):
        return _finish(args, _stage2_annotation_worklist(args))
    if (
        args.command == "stage2-sampling"
        and args.stage2_sampling_command == "adjudication-worklist"
    ):
        return _finish(args, _stage2_adjudication_worklist(args))
    if (
        args.command == "stage2-sampling"
        and args.stage2_sampling_command == "assemble-annotation-ledger"
    ):
        return _finish(args, _stage2_assemble_annotation_ledger(args))
    if (
        args.command == "stage2-sampling"
        and args.stage2_sampling_command == "finalize-annotations"
    ):
        return _finish(args, _stage2_annotations_finalize(args))
    if (
        args.command == "stage2-calibration"
        and args.stage2_calibration_command == "freeze-dev-scores"
    ):
        return _finish(args, freeze_stage2_dev_scores(
            manifest_path=args.gold_manifest,
            snapshot_path=args.private_snapshot,
            topic_queries_path=args.topic_queries,
            runtime_path=args.runtime,
            reranker_lock_path=args.reranker_lock,
            adjudicator_lock_path=args.adjudicator_lock,
            output_path=args.output,
            dry_run=args.dry_run,
        ))
    if args.command == "stage2-parity" and args.stage2_parity_command == "freeze-workload":
        return _finish(args, freeze_stage2_parity_workload(
            papers_path=args.papers,
            selection_receipt_path=args.selection_receipt,
            topic=args.topic,
            language=args.language,
            query_version=args.query_version,
            query=args.query,
            output_path=args.output,
            dry_run=args.dry_run,
        ))
    if args.command == "stage2-parity" and args.stage2_parity_command == "run":
        return _finish(args, run_stage2_parity(
            workload_path=args.workload,
            selection_receipt_path=args.selection_receipt,
            oracle_candidate_path=args.oracle_stage2_candidate,
            oracle_model_lock_path=args.oracle_model_lock,
            candidate_path=args.stage2_candidate,
            candidate_model_lock_path=args.candidate_model_lock,
            manifest_output_path=args.manifest_output,
            scores_output_path=args.scores_output,
            dry_run=args.dry_run,
        ))
    if (
        args.command == "stage2-calibration"
        and args.stage2_calibration_command == "build-candidate"
    ):
        return _finish(args, build_stage2_candidate(
            manifest_path=args.gold_manifest,
            private_labels_path=args.private_labels,
            raw_scores_path=args.raw_scores,
            runtime_path=args.runtime,
            reranker_lock_path=args.reranker_lock,
            adjudicator_lock_path=args.adjudicator_lock,
            candidate_id=args.candidate_id,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
        ))
    if (
        args.command == "stage2-rationale"
        and args.stage2_rationale_command == "run-source"
    ):
        return _finish(args, _stage2_rationale_run_source(args))
    if (
        args.command == "stage2-rationale"
        and args.stage2_rationale_command == "derive-examples"
    ):
        return _finish(args, _stage2_rationale_derive(args))
    if (
        args.command == "stage2-rationale"
        and args.stage2_rationale_command == "freeze-worklist"
    ):
        return _finish(args, _stage2_rationale_freeze(args))
    if (
        args.command == "stage2-rationale"
        and args.stage2_rationale_command == "import-worklist"
    ):
        return _finish(args, _stage2_rationale_import(args))
    if args.command == "stage2-replay":
        return _finish(args, run_structured_replay(
            papers_path=args.papers,
            candidate_path=args.stage2_candidate,
            manifest_output=args.manifest_output,
            records_output=args.records_output,
            dry_run=args.dry_run,
        ))
    if args.command == "benchmark-stage2":
        if args.benchmark_command == "freeze-manifests":
            return _finish(args, freeze_stage2_benchmark_manifests(
                candidate_path=args.stage2_candidate,
                performance_papers_path=args.performance_papers,
                soak_papers_path=args.soak_papers,
                selection_receipt_path=args.selection_receipt,
                performance_output=args.performance_output,
                soak_output=args.soak_output,
                dry_run=args.dry_run,
            ))
        if args.benchmark_command == "measure":
            result = measure_stage2_benchmark(
                manifest_path=args.manifest,
                papers_path=args.papers,
                candidate_path=args.stage2_candidate,
                environment_path=args.environment,
                database_path=args.database,
                output_path=args.output,
                scenario=args.scenario,
                run_id=args.run_id or "",
                omlx_pids=args.omlx_pid,
                sample_interval_seconds=args.sample_interval_seconds,
                dry_run=args.dry_run,
            )
            return _finish(args, result)
        if args.manifest is None or not args.record:
            raise CliUsageError(
                "benchmark-stage2 evaluation requires --manifest and at least one --record"
            )
        result = evaluate_benchmark_artifacts(
            manifest_path=args.manifest,
            record_paths=args.record,
            soak_manifest_path=args.soak_manifest,
            soak_record_path=args.soak_record,
        )
        return _finish(args, result)
    if args.command == "stage2-evaluator" and args.stage2_evaluator_command == "attest":
        return _finish(args, _stage2_evaluator_attest(args))
    if (
        args.command == "stage2-evaluator"
        and args.stage2_evaluator_command == "predict-hidden"
    ):
        return _finish(args, build_hidden_promotion_submission(
            manifest_path=args.manifest,
            snapshot_path=args.private_snapshot,
            candidate_path=args.stage2_candidate,
            output_path=args.output,
            dry_run=args.dry_run,
        ))
    if args.command == "stage2-evaluator" and args.stage2_evaluator_command == "promote":
        return _finish(args, _stage2_evaluator_promote(args))
    if args.command == "stage2-release" and args.stage2_release_command == "assemble":
        return _finish(args, _stage2_release_assemble(args))
    if args.command == "stage2-release" and args.stage2_release_command == "build-evidence":
        return _finish(args, _stage2_release_build_evidence(args))
    if args.command == "stage2-tuning" and args.stage2_tuning_command == "select":
        return _finish(args, _stage2_tuning_select(args))
    if args.command == "report":
        return _finish(args, _report(args))
    if args.command == "verify-report":
        report_run_id = args.report_run_id or args.run_id
        if report_run_id is None:
            raise CliUsageError("verify-report requires --report-run-id or --run-id")
        checklist = verify_report_run(args.output_root, report_run_id)
        return _finish(args, {
            "checks": checklist,
            "command": "verify-report",
            "report_run_id": report_run_id,
            "status": "passed",
        })
    if args.command == "download":
        return _finish(args, _download(args))
    if args.command == "analyze":
        return _finish(args, _analyze(args))
    if args.command in {"run", "resume"}:
        result = _workflow(args)
        return _finish(args, result)
    if args.command == "import-seeds":
        return _finish(
            args,
            _import_seeds(args.database, args.seed, args.input, args.run_id, args.dry_run),
        )
    raise AssertionError(args.command)


def _filter(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config) if args.config else None
    database = _database_path(args.database, config, args.config)
    release = args.stage2_release or _configured_stage2_release()
    if release is None:
        raise Stage2ReleaseError(
            "filter requires --stage2-release or PAPER_AGENT_STAGE2_RELEASE"
        )
    plan = _load_json(args.plan)
    campaign_id = args.campaign_id or args.run_id or f"filter-{plan['plan_id']}"
    return filter_database(
        plan_path=args.plan,
        release_path=release,
        database_path=database,
        campaign_id=campaign_id,
        paper_ids=args.paper_id,
        dry_run=args.dry_run,
    )


def _stage2_sampling_build(args: argparse.Namespace) -> dict[str, Any]:
    outputs = (args.gold_manifest_output, args.provenance_output)
    if outputs[0].resolve() == outputs[1].resolve():
        raise CliUsageError("Stage 2 sampling outputs must use different paths")
    existing = [path for path in outputs if os.path.lexists(path)]
    if existing:
        raise FileExistsError(f"Stage 2 sampling output already exists: {existing[0]}")

    snapshot = load_private_corpus_snapshot(args.private_snapshot)
    policy = SamplingPolicy(
        snapshot.sampling_policy_version,
        snapshot.sampling_seed,
    )
    hidden_real_selection = load_hidden_real_selection(
        args.hidden_real_freeze_frame,
        snapshot=snapshot,
        policy=policy,
    )
    receipt = load_curation_receipt(
        args.curation_receipt,
        snapshot=snapshot,
        hidden_real_selection=hidden_real_selection,
    )
    annotations = load_private_sampling_annotations(
        args.curated_annotations, snapshot=snapshot
    )
    if receipt.curated_annotations_hash != annotations.hash():
        raise ValueError("curation receipt does not bind the supplied curated annotations")
    result = build_gold_sampling(
        snapshot,
        annotations,
        policy,
        hidden_real_selection=hidden_real_selection,
    )

    if not args.dry_run:
        write_gold_sampling_provenance(args.provenance_output, result.provenance)
        write_gold_sampling_manifest(args.gold_manifest_output, result.manifest)

    split_counts = {
        split: sum(pair.split.value == split for pair in result.manifest.pairs)
        for split in ("dev", "hidden_hard", "hidden_real")
    }
    return {
        "command": "stage2-sampling.build",
        "corpus_hash": snapshot.corpus_hash,
        "curation_receipt_hash": receipt.hash(),
        "dry_run": args.dry_run,
        "gold_manifest_hash": result.manifest.hash(),
        "gold_manifest_output": str(args.gold_manifest_output),
        "hidden_real_freeze_frame_hash": hidden_real_selection.hash(),
        "provenance_hash": result.provenance.hash(),
        "provenance_output": str(args.provenance_output),
        "sampling_annotations_hash": annotations.hash(),
        "snapshot_hash": snapshot.hash(),
        "split_counts": split_counts,
        "status": "validated" if args.dry_run else "complete",
        "written": not args.dry_run,
    }


def _stage2_sampling_freeze_frame(args: argparse.Namespace) -> dict[str, Any]:
    if os.path.lexists(args.output):
        raise FileExistsError(f"Stage 2 hidden_real freeze-frame output already exists: {args.output}")
    snapshot = load_private_corpus_snapshot(args.private_snapshot)
    policy = SamplingPolicy(snapshot.sampling_policy_version, snapshot.sampling_seed)
    selection = select_hidden_real(snapshot, policy)
    if not args.dry_run:
        write_hidden_real_selection(args.output, selection)
    return {
        "command": "stage2-sampling.freeze-frame",
        "corpus_hash": snapshot.corpus_hash,
        "dry_run": args.dry_run,
        "hidden_real_count": len(selection.pair_keys),
        "hidden_real_freeze_frame_hash": selection.hash(),
        "output": str(args.output),
        "sampling_probability": selection.sampling_probability,
        "snapshot_hash": snapshot.hash(),
        "status": "validated" if args.dry_run else "complete",
        "written": not args.dry_run,
    }


def _stage2_curation_worklist(args: argparse.Namespace) -> dict[str, Any]:
    if os.path.lexists(args.output):
        raise FileExistsError(f"Stage 2 curation worklist output already exists: {args.output}")
    snapshot = load_private_corpus_snapshot(args.private_snapshot)
    policy = SamplingPolicy(snapshot.sampling_policy_version, snapshot.sampling_seed)
    hidden_real_selection = load_hidden_real_selection(
        args.hidden_real_freeze_frame,
        snapshot=snapshot,
        policy=policy,
    )
    worklist = make_curation_worklist(snapshot, hidden_real_selection)
    if not args.dry_run:
        write_curation_worklist(args.output, worklist)
    return {
        "command": "stage2-sampling.curation-worklist",
        "dry_run": args.dry_run,
        "hidden_real_freeze_frame_hash": hidden_real_selection.hash(),
        "output": str(args.output),
        "row_count": len(worklist.rows),
        "snapshot_hash": snapshot.hash(),
        "status": "validated" if args.dry_run else "complete",
        "worklist_hash": worklist.hash(),
        "written": not args.dry_run,
    }


def _stage2_curation_import(args: argparse.Namespace) -> dict[str, Any]:
    outputs = (args.curated_annotations_output, args.receipt_output)
    if outputs[0].resolve() == outputs[1].resolve():
        raise CliUsageError("Stage 2 curation import outputs must use different paths")
    existing = [path for path in outputs if os.path.lexists(path)]
    if existing:
        raise FileExistsError(f"Stage 2 curation import output already exists: {existing[0]}")

    snapshot = load_private_corpus_snapshot(args.private_snapshot)
    policy = SamplingPolicy(snapshot.sampling_policy_version, snapshot.sampling_seed)
    hidden_real_selection = load_hidden_real_selection(
        args.hidden_real_freeze_frame,
        snapshot=snapshot,
        policy=policy,
    )
    worklist = load_curation_worklist(
        args.worklist,
        snapshot=snapshot,
        hidden_real_selection=hidden_real_selection,
    )
    decisions = load_curation_decisions(args.decisions, worklist=worklist)
    annotations = curated_annotations_from_decisions(
        decisions,
        worklist=worklist,
        snapshot=snapshot,
    )
    receipt = make_curation_receipt(
        snapshot,
        hidden_real_selection,
        worklist,
        decisions,
        annotations,
    )
    if not args.dry_run:
        write_private_sampling_annotations(
            args.curated_annotations_output,
            annotations,
            snapshot=snapshot,
        )
        write_curation_receipt(args.receipt_output, receipt)
    return {
        "command": "stage2-sampling.curation-import",
        "curated_annotations_hash": annotations.hash(),
        "curated_annotations_output": str(args.curated_annotations_output),
        "dry_run": args.dry_run,
        "receipt_hash": receipt.hash(),
        "receipt_output": str(args.receipt_output),
        "row_count": len(annotations.rows),
        "source_counts": decisions.source_counts(),
        "status": "validated" if args.dry_run else "complete",
        "worklist_hash": worklist.hash(),
        "written": not args.dry_run,
    }


def _stage2_annotation_worklist(args: argparse.Namespace) -> dict[str, Any]:
    if os.path.lexists(args.output):
        raise FileExistsError(f"Stage 2 annotation worklist output already exists: {args.output}")
    manifest = load_gold_manifest(args.gold_manifest)
    snapshot = load_private_corpus_snapshot(args.private_snapshot)
    worklist = make_human_annotation_worklist(
        manifest,
        snapshot,
        participant_id=args.participant_id,
    )
    if not args.dry_run:
        write_human_annotation_worklist(args.output, worklist)
    return {
        "command": "stage2-sampling.annotation-worklist",
        "dry_run": args.dry_run,
        "gold_manifest_hash": worklist["gold_manifest_hash"],
        "output": str(args.output),
        "row_count": len(worklist["rows"]),
        "rubric_hash": STAGE2_ANNOTATION_RUBRIC_HASH,
        "status": "validated" if args.dry_run else "complete",
        "written": not args.dry_run,
    }


def _stage2_adjudication_worklist(args: argparse.Namespace) -> dict[str, Any]:
    if os.path.lexists(args.output):
        raise FileExistsError(f"Stage 2 adjudication worklist output already exists: {args.output}")
    manifest = load_gold_manifest(args.gold_manifest)
    snapshot = load_private_corpus_snapshot(args.private_snapshot)
    first, second = _load_completed_annotators(args, manifest, snapshot)
    worklist, kappa = make_adjudication_worklist(
        manifest,
        snapshot,
        first,
        second,
        participant_id=args.participant_id,
    )
    if not args.dry_run:
        write_human_annotation_worklist(args.output, worklist)
    return {
        "command": "stage2-sampling.adjudication-worklist",
        "disagreement_count": len(worklist["rows"]),
        "dry_run": args.dry_run,
        "gold_manifest_hash": worklist["gold_manifest_hash"],
        "output": str(args.output),
        "pre_adjudication_quadratic_weighted_kappa": kappa,
        "rubric_hash": STAGE2_ANNOTATION_RUBRIC_HASH,
        "status": "validated" if args.dry_run else "complete",
        "written": not args.dry_run,
    }


def _stage2_assemble_annotation_ledger(args: argparse.Namespace) -> dict[str, Any]:
    if os.path.lexists(args.output):
        raise FileExistsError(f"Stage 2 annotation ledger output already exists: {args.output}")
    manifest = load_gold_manifest(args.gold_manifest)
    snapshot = load_private_corpus_snapshot(args.private_snapshot)
    first, second = _load_completed_annotators(args, manifest, snapshot)
    adjudication = load_human_annotation_worklist(
        args.adjudication,
        manifest=manifest,
        snapshot=snapshot,
        role="adjudicator",
        require_complete=True,
    )
    curated_annotations = load_private_sampling_annotations(
        args.curated_annotations,
        snapshot=snapshot,
    )
    sampling_provenance = load_gold_sampling_provenance(
        args.sampling_provenance,
        snapshot=snapshot,
        annotations=curated_annotations,
        manifest=manifest,
    )
    document, ledger = assemble_annotation_ledger(
        manifest,
        first,
        second,
        adjudication,
        curated_annotations,
        sampling_provenance_hash=sampling_provenance.hash(),
    )
    if not args.dry_run:
        write_annotation_ledger(args.output, document)
    return {
        "annotation_artifact_hash": ledger.summary.annotation_artifact_hash,
        "command": "stage2-sampling.assemble-annotation-ledger",
        "disagreement_count": len(ledger.summary.disagreement_pair_ids),
        "dry_run": args.dry_run,
        "gold_manifest_hash": document["gold_manifest_hash"],
        "hard_negative_count": len(ledger.gold_labels.hard_negative_pair_ids),
        "hard_positive_count": len(ledger.gold_labels.hard_positive_pair_ids),
        "label_count": len(ledger.gold_labels.labels),
        "output": str(args.output),
        "pre_adjudication_quadratic_weighted_kappa": ledger.summary.quadratic_weighted_kappa,
        "rubric_hash": STAGE2_ANNOTATION_RUBRIC_HASH,
        "status": "validated" if args.dry_run else "complete",
        "written": not args.dry_run,
    }


def _load_completed_annotators(
    args: argparse.Namespace,
    manifest: Any,
    snapshot: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        load_human_annotation_worklist(
            args.annotation_a,
            manifest=manifest,
            snapshot=snapshot,
            role="annotator",
            require_complete=True,
        ),
        load_human_annotation_worklist(
            args.annotation_b,
            manifest=manifest,
            snapshot=snapshot,
            role="annotator",
            require_complete=True,
        ),
    )


def _stage2_annotations_finalize(args: argparse.Namespace) -> dict[str, Any]:
    if os.path.lexists(args.private_labels_output):
        raise FileExistsError(
            f"Stage 2 private labels output already exists: {args.private_labels_output}"
        )

    manifest = load_gold_manifest(args.gold_manifest)
    manifest.validate_sampling_structure()
    ledger = load_annotation_ledger(args.annotation_ledger, manifest=manifest)
    if not args.dry_run:
        write_private_gold_labels(
            args.private_labels_output,
            ledger,
            manifest=manifest,
        )
    return {
        "annotation_artifact_hash": ledger.summary.annotation_artifact_hash,
        "command": "stage2-sampling.finalize-annotations",
        "dry_run": args.dry_run,
        "gold_label_store_hash": ledger.gold_labels.hash(),
        "gold_manifest_hash": manifest.hash(),
        "label_count": len(ledger.gold_labels.labels),
        "pre_adjudication_quadratic_weighted_kappa": (
            ledger.summary.quadratic_weighted_kappa
        ),
        "private_labels_output": str(args.private_labels_output),
        "status": "validated" if args.dry_run else "complete",
        "written": not args.dry_run,
    }


def _stage2_rationale_run_source(args: argparse.Namespace) -> dict[str, Any]:
    if os.path.lexists(args.output_dir):
        raise FileExistsError(
            f"Stage 2 rationale source output already exists: {args.output_dir}"
        )
    candidate_path = args.stage2_candidate.resolve(strict=True)
    candidate_bytes = candidate_path.read_bytes()
    papers_bytes = args.benchmark_papers.read_bytes()
    candidate = _load_stage2_benchmark_candidate_bytes(candidate_path, candidate_bytes)
    papers_document = json.loads(papers_bytes)
    plan = rationale_source_plan(
        candidate,
        benchmark_papers_document=papers_document,
    )
    payload: dict[str, Any] = {
        "command": "stage2-rationale.run-source",
        "dry_run": args.dry_run,
        "candidate_id": plan.candidate_id,
        "paper_count": plan.paper_count,
        "primary_languages": list(plan.primary_languages),
        "qwen_pair_count": plan.qwen_pair_count,
        "reranker_pair_count": plan.reranker_pair_count,
        "status": "validated" if args.dry_run else "complete",
        "topic_query_count": plan.topic_query_count,
        "written": not args.dry_run,
    }
    if args.dry_run:
        return payload
    transport = UrlLibOmlxTransport(
        candidate.omlx_base_url,
        api_key=(
            os.environ.get(candidate.api_key_env)
            if candidate.api_key_env is not None
            else None
        ),
    )
    artifacts = collect_rationale_source_artifacts(
        candidate,
        candidate_bundle_sha256=sha256(candidate_bytes).hexdigest(),
        benchmark_papers_document=papers_document,
        benchmark_papers_sha256=sha256(papers_bytes).hexdigest(),
        transport=transport,
    )
    query_metadata_path, source_ledger_path = write_rationale_source_artifacts_no_replace(
        artifacts,
        output_directory=args.output_dir,
    )
    payload.update({
        "query_metadata_output": str(query_metadata_path),
        "query_metadata_sha256": sha256(query_metadata_path.read_bytes()).hexdigest(),
        "adjudication_ledger_output": str(source_ledger_path),
        "adjudication_ledger_sha256": sha256(source_ledger_path.read_bytes()).hexdigest(),
    })
    return payload


def _stage2_rationale_freeze(args: argparse.Namespace) -> dict[str, Any]:
    outputs = (args.manifest_output, args.worklist_output)
    if outputs[0].absolute() == outputs[1].absolute():
        raise CliUsageError("Stage 2 rationale outputs must use different paths")
    existing = next((path for path in outputs if path.exists()), None)
    if existing is not None:
        raise FileExistsError(f"Stage 2 rationale output already exists: {existing}")
    examples, corpus_hash, model_lock_hash = rationale_audit_examples_from_document(
        _load_json(args.examples), require_derived=True
    )
    frozen = freeze_rationale_audit(
        examples,
        corpus_hash=corpus_hash,
        model_lock_hash=model_lock_hash,
        reviewer_id=args.reviewer_id,
    )
    if not args.dry_run:
        write_frozen_rationale_audit(
            frozen,
            manifest_path=args.manifest_output,
            worklist_path=args.worklist_output,
        )
    return {
        "command": "stage2-rationale.freeze-worklist",
        "dry_run": args.dry_run,
        "manifest_hash": frozen.manifest.hash(),
        "manifest_output": str(args.manifest_output),
        "reviewer_id": args.reviewer_id,
        "row_count": len(frozen.manifest.cases),
        "status": "validated" if args.dry_run else "complete",
        "worklist_output": str(args.worklist_output),
        "written": not args.dry_run,
    }


def _stage2_rationale_derive(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError(f"Stage 2 rationale examples output already exists: {args.output}")
    candidate_path = args.stage2_candidate.resolve(strict=True)
    candidate_bytes = candidate_path.read_bytes()
    benchmark_papers_bytes = args.benchmark_papers.read_bytes()
    query_metadata_bytes = args.query_metadata.read_bytes()
    ledger_bytes = args.adjudication_ledger.read_bytes()
    candidate = _load_stage2_benchmark_candidate_bytes(candidate_path, candidate_bytes)
    document = derive_rationale_audit_examples(
        json.loads(ledger_bytes),
        source_ledger_sha256=sha256(ledger_bytes).hexdigest(),
        candidate=candidate,
        candidate_bundle_sha256=sha256(candidate_bytes).hexdigest(),
        benchmark_papers_document=json.loads(benchmark_papers_bytes),
        benchmark_papers_sha256=sha256(benchmark_papers_bytes).hexdigest(),
        query_metadata=json.loads(query_metadata_bytes),
        query_metadata_sha256=sha256(query_metadata_bytes).hexdigest(),
    )
    if not args.dry_run:
        write_derived_rationale_examples_no_replace(args.output, document)
    return {
        "command": "stage2-rationale.derive-examples",
        "dry_run": args.dry_run,
        "candidate_id": candidate.profile_name,
        "corpus_hash": document["corpus_hash"],
        "example_count": len(document["examples"]),
        "output": str(args.output),
        "source_ledger_sha256": document["source_ledger_sha256"],
        "status": "validated" if args.dry_run else "complete",
        "written": not args.dry_run,
    }


def _stage2_rationale_import(args: argparse.Namespace) -> dict[str, Any]:
    if args.records_output.exists():
        raise FileExistsError(
            f"Stage 2 rationale records output already exists: {args.records_output}"
        )
    manifest = load_rationale_audit_manifest(args.manifest)
    worklist = load_rationale_worklist(args.worklist)
    records = import_completed_rationale_audit(worklist, manifest=manifest)
    gate = rationale_audit_gate(manifest, records)
    worklist_sha256 = sha256(args.worklist.read_bytes()).hexdigest()
    document = rationale_audit_records_document(records, worklist_sha256=worklist_sha256)
    if not args.dry_run:
        write_rationale_records_no_replace(
            args.records_output, records, worklist_sha256=worklist_sha256
        )
    return {
        "command": "stage2-rationale.import-worklist",
        "dry_run": args.dry_run,
        "evidence_support_rate": sum(row.evidence_supported for row in records) / len(records),
        "failures": list(gate.failures),
        "manifest_hash": manifest.hash(),
        "record_count": len(records),
        "records_hash": content_hash(document),
        "records_output": str(args.records_output),
        "severe_fabrication_rate": sum(row.severe_fabrication for row in records) / len(records),
        "status": "validated" if args.dry_run else "complete",
        "written": not args.dry_run,
    }


def _stage2_release_assemble(args: argparse.Namespace) -> dict[str, Any]:
    """Verify and optionally write one public-safe Stage 2 release."""

    operation = (
        validate_stage2_release_assembly
        if args.dry_run
        else assemble_stage2_release
    )
    try:
        result = operation(
            args.candidate,
            args.evidence,
            args.trust_manifest,
            args.output,
            parity_oracle_trust_path=args.parity_oracle_trust,
            fallback_candidate_path=args.fallback_candidate,
            fallback_evidence_path=args.fallback_evidence,
            fallback_omlx_base_url=args.fallback_omlx_base_url,
            fallback_api_key_env=args.fallback_api_key_env,
        )
    except Exception:
        # Inputs include deployment-owned trust.  Keep the console boundary
        # useful without reflecting paths, schema instances, or verifier
        # internals that could disclose evaluator material.
        raise CliUsageError(
            "Stage 2 release assembly verification failed"
        ) from None
    return {
        "command": "stage2-release.assemble",
        "dry_run": args.dry_run,
        "written": not args.dry_run,
        **result.summary(),
        "status": "validated" if args.dry_run else "complete",
    }


def _stage2_release_build_evidence(args: argparse.Namespace) -> dict[str, Any]:
    if len(args.benchmark_record) != 6:
        raise CliUsageError("stage2-release build-evidence requires exactly six --benchmark-record values")
    values = {
        "output_path": args.output,
        "candidate_bundle_path": args.candidate,
        "gold_manifest_path": args.gold_manifest,
        "structured_manifest_path": args.structured_manifest,
        "structured_records_path": args.structured_records,
        "structured_papers_path": args.structured_papers,
        "rationale_manifest_path": args.rationale_manifest,
        "rationale_worklist_path": args.rationale_worklist,
        "rationale_records_path": args.rationale_records,
        "rationale_source_ledger_path": args.rationale_source_ledger,
        "rationale_query_metadata_path": args.rationale_query_metadata,
        "rationale_derived_examples_path": args.rationale_derived_examples,
        "rationale_papers_path": args.rationale_papers,
        "parity_manifest_path": args.parity_manifest,
        "parity_workload_path": args.parity_workload,
        "parity_selection_receipt_path": args.parity_selection_receipt,
        "parity_scores_path": args.parity_scores,
        "parity_oracle_model_lock_path": args.parity_oracle_model_lock,
        "parity_candidate_model_lock_path": args.parity_candidate_model_lock,
        "parity_oracle_calibrator_path": args.parity_oracle_calibrator,
        "parity_candidate_calibrator_path": args.parity_candidate_calibrator,
        "parity_oracle_threshold_path": args.parity_oracle_threshold,
        "parity_candidate_threshold_path": args.parity_candidate_threshold,
        "benchmark_manifest_path": args.benchmark_manifest,
        "benchmark_papers_path": args.benchmark_papers,
        "benchmark_record_paths": tuple(args.benchmark_record),
        "soak_manifest_path": args.soak_manifest,
        "soak_papers_path": args.soak_papers,
        "soak_record_path": args.soak_record,
        "hidden_attestation_path": args.hidden_attestation,
    }
    if args.dry_run:
        build_stage2_release_evidence_index_bytes(**values)
    else:
        write_stage2_release_evidence_index(**values)
    return {
        "command": "stage2-release.build-evidence",
        "dry_run": args.dry_run,
        "evidence_type": (
            "stage2_release_evidence" if args.hidden_attestation is not None
            else "stage2_public_promotion_evidence"
        ),
        "output": str(args.output),
        "status": "validated" if args.dry_run else "complete",
        "written": not args.dry_run,
    }


def _stage2_tuning_select(args: argparse.Namespace) -> dict[str, Any]:
    if args.dry_run and os.path.lexists(args.output):
        raise FileExistsError(f"Stage 2 tuning winner output already exists: {args.output}")
    document = (
        build_stage2_tuning_winner_document(args.input)
        if args.dry_run
        else write_stage2_tuning_winner(args.input, args.output)
    )
    return {
        "command": "stage2-tuning.select",
        "dry_run": args.dry_run,
        "document_batch_size": document["document_batch_size"],
        "adjudicator_concurrency": document["adjudicator_concurrency"],
        "input_record_count": len(document["input_record_hashes"]),
        "output": str(args.output),
        "selection_hash": document["selection_hash"],
        "status": "validated" if args.dry_run else "complete",
        "written": not args.dry_run,
    }


def _stage2_evaluator_attest(args: argparse.Namespace) -> dict[str, Any]:
    """Sign one schema-validated public-safe hidden-promotion payload."""

    payload = _load_json(args.payload)
    if not isinstance(payload, Mapping):
        raise CliUsageError("stage2-evaluator attest payload must be a JSON object")
    validate_hidden_promotion_payload(payload)
    _assert_attestation_output_absent(args.output)
    trust = load_hidden_evaluator_trust(args.trust_manifest)
    validate_hidden_evaluator_signing_trust(payload, trust)

    result = {
        "command": "stage2-evaluator.attest",
        "candidate_id": payload["candidate_id"],
        "evaluation_run_id": payload["evaluation_run_id"],
        "evaluator_key_id": payload["evaluator_key_id"],
        "passed": payload["result_summary"]["passed"],
        "payload_sha256": content_hash(payload),
        "output": str(args.output),
    }
    if args.dry_run:
        return {**result, "signed": False, "status": "validated"}

    private_key = load_hidden_evaluator_private_key(args.signing_key_file)
    validate_hidden_evaluator_private_key_trust(
        private_key,
        evaluator_key_id=payload["evaluator_key_id"],
        trust=trust,
    )
    attestation = issue_hidden_promotion_from_payload(payload, private_key)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as output:
        output.write(canonical_json(attestation))
    return {
        **result,
        "attestation_sha256": content_hash(attestation),
        "signed": True,
        "status": "complete",
    }


def _assert_attestation_output_absent(path: Path) -> None:
    """Reject every pre-existing destination, including a dangling symlink."""

    try:
        path.lstat()
    except FileNotFoundError:
        return
    except OSError as error:
        raise OSError("cannot inspect hidden promotion attestation output") from error
    raise FileExistsError("hidden promotion attestation already exists")


def _stage2_evaluator_promote(args: argparse.Namespace) -> dict[str, Any]:
    """Run one sealed batch and sign its winner plus requested qualified backups.

    All public bindings, trust material, private-key custody, and the empty
    output reservation are checked before private labels or predictions are
    opened.  Once the evaluator starts, its hidden marker may be consumed even
    when the selected candidate fails a gate; that signed failure is useful
    audit evidence and never becomes a production release.
    """

    candidate_paths = _stage2_promotion_paths(args.candidate, "candidate")
    submission_paths = _stage2_promotion_paths(args.submission, "submission")
    public_evidence_paths = _stage2_promotion_paths(args.public_evidence, "public evidence")
    fallback_output_paths = _stage2_promotion_paths(
        args.qualified_fallback_output,
        "qualified fallback output",
    )
    _validate_stage2_promotion_controls(
        args,
        candidate_paths,
        submission_paths,
        public_evidence_paths,
        fallback_output_paths,
    )
    try:
        # A gold manifest and v2 bundles are public inputs.  Do this before
        # touching labels or submissions so malformed mappings cannot consume
        # the hidden holdout.
        from .stage2_evaluation import load_gold_manifest as load_promotion_manifest

        manifest = load_promotion_manifest(args.manifest)
        manifest.validate_sampling_structure()
        validate_promotion_candidate_bundles(
            candidate_paths, expected_manifest_hash=manifest.hash()
        )
        validate_promotion_public_evidence(
            candidate_paths,
            public_evidence_paths,
            manifest.hash(),
            parity_oracle_trust_path=args.parity_oracle_trust,
        )
    except (OSError, ValueError, TypeError, KeyError):
        raise CliUsageError("Stage 2 promotion public inputs are invalid") from None

    output_paths = [args.output, *fallback_output_paths.values()]
    for output_path in output_paths:
        _assert_attestation_output_absent(output_path)
    try:
        trust = load_hidden_evaluator_trust(args.trust_manifest)
        if args.evaluator_key_id not in trust.keys:
            raise ValueError("selected evaluator key is not active")
    except (OSError, ValueError, TypeError, KeyError):
        raise CliUsageError("Stage 2 promotion deployment trust is invalid") from None
    if args.dry_run:
        return {
            "command": "stage2-evaluator.promote",
            "candidate_id": None,
            "evaluation_manifest_hash": manifest.hash(),
            "evaluation_run_id": args.evaluation_run_id,
            "evaluated": False,
            "signed": False,
            "requested_fallback_candidate_ids": sorted(fallback_output_paths),
            "status": "validated",
        }

    # Reading the key precedes private evaluator input and marker consumption.
    try:
        private_key = load_hidden_evaluator_private_key(args.signing_key_file)
        validate_hidden_evaluator_private_key_trust(
            private_key, evaluator_key_id=args.evaluator_key_id, trust=trust
        )
    except (OSError, ValueError, TypeError):
        raise CliUsageError("Stage 2 promotion signing prerequisites are invalid") from None
    reservations: list[
        tuple[Path, int, int, os.stat_result, os.stat_result]
    ] = []
    try:
        for output_path in output_paths:
            descriptor, directory, parent_reservation, reservation = (
                _reserve_hidden_promotion_output(output_path)
            )
            reservations.append(
                (
                    output_path,
                    descriptor,
                    directory,
                    parent_reservation,
                    reservation,
                )
            )
    except FileExistsError:
        _remove_hidden_promotion_reservations(reservations)
        raise
    except OSError:
        _remove_hidden_promotion_reservations(reservations)
        raise CliUsageError("Stage 2 promotion attestation output failed") from None
    published = False
    try:
        try:
            evaluation = run_promotion_evaluation(
                manifest_path=args.manifest,
                private_labels_path=args.private_labels,
                submission_paths=submission_paths,
                candidate_bundle_paths=candidate_paths,
                public_evidence_paths=public_evidence_paths,
                evaluator_id=args.evaluator_id,
                state_root=args.state_root,
                incumbent_candidate_id=args.incumbent_candidate_id,
                evaluation_run_id=args.evaluation_run_id,
                parity_oracle_trust_path=args.parity_oracle_trust,
                bootstrap_iterations=args.bootstrap_iterations,
                bootstrap_seed=args.bootstrap_seed,
            )
            signing = evaluation.candidates[evaluation.winner_candidate_id]
            payload = signing.attestation_payload(
                evaluator_key_id=args.evaluator_key_id,
                trust_manifest_hash=trust.manifest_hash,
                issued_at=args.issued_at,
            )
            attestation = issue_hidden_promotion_from_payload(payload, private_key)
            fallback_attestations: dict[str, tuple[dict[str, Any], Mapping[str, Any]]] = {}
            unqualified_fallback_candidate_ids: list[str] = []
            for candidate_id in fallback_output_paths:
                fallback_signing = evaluation.candidates[candidate_id]
                if (
                    fallback_signing.release_role != "qualified_fallback"
                    or not fallback_signing.passed
                    or fallback_signing.failures
                ):
                    unqualified_fallback_candidate_ids.append(candidate_id)
                    continue
                fallback_payload = fallback_signing.attestation_payload(
                    evaluator_key_id=args.evaluator_key_id,
                    trust_manifest_hash=trust.manifest_hash,
                    issued_at=args.issued_at,
                )
                fallback_attestations[candidate_id] = (
                    fallback_payload,
                    issue_hidden_promotion_from_payload(fallback_payload, private_key),
                )
        except Exception:
            raise CliUsageError("sealed Stage 2 promotion evaluation failed") from None

        try:
            requested_outputs = [
                (args.output, attestation),
                *((fallback_output_paths[candidate_id], fallback_attestation)
                  for candidate_id, (_, fallback_attestation)
                  in fallback_attestations.items()),
            ]
            reservations_by_path = {reserved[0]: reserved for reserved in reservations}
            unqualified_paths = {
                fallback_output_paths[candidate_id]
                for candidate_id in unqualified_fallback_candidate_ids
            }
            for output_path in unqualified_paths:
                reserved = reservations_by_path.pop(output_path)
                _remove_hidden_promotion_reservation(
                    reserved[2], output_path.name, reserved[1], reserved[4]
                )
                os.close(reserved[2])
                reservations.remove(reserved)
                if os.path.lexists(output_path):
                    raise OSError(
                        "cannot discard unqualified fallback attestation output"
                    )
            for output_path, output_attestation in requested_outputs:
                reserved = reservations_by_path[output_path]
                _write_reserved_hidden_promotion_output(
                    reserved[1], output_attestation
                )
            for output_path, output_attestation in requested_outputs:
                reserved = reservations_by_path[output_path]
                _verify_reserved_hidden_promotion_output(
                    reserved[0].parent,
                    reserved[0].name,
                    reserved[3],
                    reserved[4],
                    canonical_json(dict(output_attestation)),
                )
        except OSError:
            raise CliUsageError("Stage 2 promotion attestation output failed") from None
        published = True
    finally:
        if published:
            for _, descriptor, directory, _, _ in reservations:
                os.close(descriptor)
                os.close(directory)
        else:
            _remove_hidden_promotion_reservations(reservations)

    passed = bool(payload["result_summary"]["passed"])
    return {
        "command": "stage2-evaluator.promote",
        "candidate_id": evaluation.winner_candidate_id,
        "evaluation_manifest_hash": evaluation.evaluation_manifest_hash,
        "evaluation_run_id": evaluation.evaluation_run_id,
        "promotion_marker_hash": evaluation.promotion_marker_hash,
        "promotion_batch_hash": evaluation.promotion_batch_hash,
        "payload_sha256": content_hash(payload),
        "attestation_sha256": content_hash(attestation),
        "qualified_fallback_attestation_sha256": {
            candidate_id: content_hash(fallback_attestation)
            for candidate_id, (_, fallback_attestation)
            in fallback_attestations.items()
        },
        "requested_fallback_candidate_ids": sorted(fallback_output_paths),
        "unqualified_fallback_candidate_ids": sorted(
            unqualified_fallback_candidate_ids
        ),
        "evaluated": True,
        "signed": True,
        "passed": passed,
        "status": "complete" if passed else "failed",
    }


def _stage2_promotion_paths(values: Sequence[str], label: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        candidate_id, separator, path = value.partition("=")
        if (
            not separator
            or not _STAGE2_PROMOTION_IDENTIFIER.fullmatch(candidate_id)
            or not path
            or candidate_id in result
        ):
            raise CliUsageError(
                f"Stage 2 promotion {label} mappings must be unique ID=PATH values"
            )
        result[candidate_id] = Path(path)
    return result


def _validate_stage2_promotion_controls(
    args: argparse.Namespace,
    candidate_paths: Mapping[str, Path],
    submission_paths: Mapping[str, Path],
    public_evidence_paths: Mapping[str, Path],
    fallback_output_paths: Mapping[str, Path],
) -> None:
    identifiers = (
        *candidate_paths,
        *submission_paths,
        args.incumbent_candidate_id,
        args.evaluator_id,
        args.evaluation_run_id,
        args.evaluator_key_id,
    )
    for value in identifiers:
        if not _STAGE2_PROMOTION_IDENTIFIER.fullmatch(value):
            raise CliUsageError("Stage 2 promotion identifiers are invalid")
    if set(candidate_paths) != set(submission_paths) or set(candidate_paths) != set(public_evidence_paths):
        raise CliUsageError("Stage 2 promotion candidate, submission, and public evidence mappings must match")
    if not set(fallback_output_paths).issubset(candidate_paths):
        raise CliUsageError(
            "Stage 2 qualified fallback outputs must name submitted candidates"
        )
    if (
        args.incumbent_candidate_id not in candidate_paths
    ):
        raise CliUsageError("Stage 2 promotion incumbent candidate must be submitted")
    if not _stage2_promotion_date_time(args.issued_at):
        raise CliUsageError("Stage 2 promotion issued-at is not a schema-valid date-time")
    if args.bootstrap_iterations <= 0:
        raise CliUsageError("Stage 2 promotion bootstrap iterations must be positive")
    canonical_outputs = {
        path.resolve(strict=False) for path in (args.output, *fallback_output_paths.values())
    }
    if len(canonical_outputs) != 1 + len(fallback_output_paths):
        raise CliUsageError("Stage 2 promotion output paths must be unique")


def _stage2_promotion_date_time(value: str) -> bool:
    """Apply the RFC 3339 shape required by JSON Schema ``date-time``."""

    if not _STAGE2_PROMOTION_DATE_TIME.fullmatch(value):
        return False
    normalized = value.replace("t", "T").replace("z", "Z")
    try:
        parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


def _hidden_promotion_directory_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _reserve_hidden_promotion_output(
    path: Path,
) -> tuple[int, int, os.stat_result, os.stat_result]:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        directory = os.open(path.parent, _hidden_promotion_directory_flags())
    except OSError:
        raise OSError("cannot open hidden promotion attestation output directory") from None
    try:
        parent_reservation = os.fstat(directory)
    except OSError:
        os.close(directory)
        raise OSError("cannot inspect hidden promotion attestation output directory") from None
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor: int | None = None
    reservation: os.stat_result | None = None
    try:
        descriptor = os.open(path.name, flags, 0o600, dir_fd=directory)
        reservation = os.fstat(descriptor)
        os.fchmod(descriptor, 0o600)
        reservation = os.fstat(descriptor)
        if not stat.S_ISREG(reservation.st_mode):
            raise OSError("hidden promotion attestation output is not a regular file")
    except FileExistsError:
        os.close(directory)
        raise FileExistsError("hidden promotion attestation already exists") from None
    except OSError:
        if descriptor is not None:
            if reservation is None:
                os.close(descriptor)
            else:
                _remove_hidden_promotion_reservation(
                    directory, path.name, descriptor, reservation
                )
        os.close(directory)
        raise
    return descriptor, directory, parent_reservation, reservation


def _write_reserved_hidden_promotion_output(
    descriptor: int, attestation: Mapping[str, Any]
) -> None:
    remaining = memoryview(canonical_json(dict(attestation)))
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise OSError("cannot write hidden promotion attestation output")
        remaining = remaining[written:]
    os.fsync(descriptor)


def _verify_reserved_hidden_promotion_output(
    parent: Path,
    name: str,
    parent_reservation: os.stat_result,
    reservation: os.stat_result,
    expected: bytes,
) -> None:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    directory = os.open(parent, _hidden_promotion_directory_flags())
    try:
        current_parent = os.fstat(directory)
        if (current_parent.st_dev, current_parent.st_ino) != (
            parent_reservation.st_dev,
            parent_reservation.st_ino,
        ):
            raise OSError("hidden promotion attestation output directory changed")
        descriptor = os.open(name, flags, dir_fd=directory)
        try:
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISREG(opened.st_mode)
                or stat.S_IMODE(opened.st_mode) != 0o600
                or (opened.st_dev, opened.st_ino)
                != (reservation.st_dev, reservation.st_ino)
            ):
                raise OSError("hidden promotion attestation output reservation changed")
            actual = bytearray()
            while len(actual) <= len(expected):
                block = os.read(
                    descriptor, min(65_536, len(expected) + 1 - len(actual))
                )
                if not block:
                    break
                actual.extend(block)
            final = os.fstat(descriptor)
            current = os.stat(name, dir_fd=directory, follow_symlinks=False)
            if (
                bytes(actual) != expected
                or (
                    opened.st_dev,
                    opened.st_ino,
                    opened.st_mode,
                    opened.st_size,
                    opened.st_mtime_ns,
                    opened.st_ctime_ns,
                )
                != (
                    final.st_dev,
                    final.st_ino,
                    final.st_mode,
                    final.st_size,
                    final.st_mtime_ns,
                    final.st_ctime_ns,
                )
                or not stat.S_ISREG(current.st_mode)
                or stat.S_IMODE(current.st_mode) != 0o600
                or (current.st_dev, current.st_ino, current.st_size)
                != (reservation.st_dev, reservation.st_ino, len(expected))
            ):
                raise OSError("hidden promotion attestation output verification failed")
            os.fsync(directory)
        finally:
            os.close(descriptor)
    finally:
        os.close(directory)


def _remove_hidden_promotion_reservation(
    directory: int, name: str, descriptor: int, reservation: os.stat_result
) -> None:
    try:
        os.close(descriptor)
    except OSError:
        pass
    try:
        current = os.stat(name, dir_fd=directory, follow_symlinks=False)
    except OSError:
        return
    if (
        current.st_dev == reservation.st_dev
        and current.st_ino == reservation.st_ino
    ):
        try:
            os.unlink(name, dir_fd=directory)
        except OSError:
            pass


def _remove_hidden_promotion_reservations(
    reservations: Sequence[
        tuple[Path, int, int, os.stat_result, os.stat_result]
    ],
) -> None:
    """Remove only the exact still-reserved inodes and close their directories."""

    for path, descriptor, directory, _, reservation in reversed(reservations):
        _remove_hidden_promotion_reservation(
            directory, path.name, descriptor, reservation
        )
        try:
            os.close(directory)
        except OSError:
            pass


def _report(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config) if args.config else None
    runtime = _report_runtime_config(config, args.config)
    generation_requested = (
        args.report_command in {"approve", "prepare-inputs"}
        or args.plan_only
        or args.plan is not None
    )
    if generation_requested and not runtime.enabled:
        return {
            "alarm_codes": [],
            "command": "report",
            "codex_budget": None,
            "dry_run": args.dry_run,
            "error": None,
            "skipped": True,
            "status": "complete",
        }
    if args.report_command == "prepare-inputs":
        return _report_prepare_inputs(args, config=config)
    if args.report_command == "approve":
        database_path: Path | None = None
        if args.handoff_id is not None:
            required = {
                "--database": args.database,
                "--workflow-config": args.workflow_config,
                "--workflow-manifest": args.workflow_manifest,
            }
            missing = [name for name, value in required.items() if value is None]
            if missing:
                raise CliUsageError(
                    "report approve --handoff-id requires " + ", ".join(missing)
                )
            database_path = _database_path(args.database, config, args.config)
            if not database_path.is_file():
                raise FileNotFoundError(f"database does not exist: {database_path}")
        approved_at = args.approved_at or datetime.now(UTC).isoformat().replace(
            "+00:00", "Z"
        )
        result = approve_report_plan_from_files(
            args.plan,
            args.corpus_snapshot,
            args.search_audit,
            args.output_root,
            expected_hash=args.hash,
            approved_by=args.approved_by,
            approved_at=approved_at,
            save_bundle=not args.dry_run and args.handoff_id is None,
            resources=runtime.resources,
        )
        response: dict[str, Any] = {
            "command": "report.approve",
            "path": str(result.path),
            "plan_hash": result.plan["plan_hash"],
            "plan_id": result.plan["plan_id"],
            "status": "validated" if args.dry_run else "approved",
            "write_performed": result.saved,
        }
        if args.handoff_id is not None:
            assert database_path is not None
            with Database(database_path, read_only=args.dry_run) as database:
                if not args.dry_run:
                    database.migrate()
                execution = WorkflowReportHandoffService(
                    database,
                    ArtifactStore(args.artifact_root or args.output_root),
                    args.output_root,
                ).prepare_report_workflow(
                    WorkflowReportExecutionRequest(
                        args.handoff_id,
                        result.path,
                        args.workflow_config,
                        args.workflow_manifest,
                        processing_grants_path=args.workflow_processing_grants,
                        previous_report_run_id=args.previous_report_run_id,
                        policy_path=args.workflow_policy,
                        workflow_id=args.report_workflow_id,
                        workflow_run_id=args.report_workflow_run_id,
                    ),
                    save_manifest=not args.dry_run,
                    approved_bundle=ReportPlanBundle(
                        result.plan,
                        _load_mapping(args.corpus_snapshot, "corpus snapshot"),
                        _load_mapping(args.search_audit, "search audit"),
                    ),
                )
            response["report_workflow"] = execution.document()
            response["write_performed"] = execution.write_performed
        return response
    if args.plan_only:
        required = {"--draft": args.draft, "--output-root": args.output_root}
        if args.handoff_id is None:
            required.update({
                "--corpus-snapshot": args.corpus_snapshot,
                "--search-audit": args.search_audit,
            })
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise CliUsageError(
                "report --plan-only requires " + ", ".join(missing)
            )
        if args.handoff_id is not None:
            if args.corpus_snapshot is not None or args.search_audit is not None:
                raise CliUsageError(
                    "handoff-bound report planning derives corpus/search audit from SQLite"
                )
            database_path = _database_path(args.database, config, args.config)
            if not database_path.is_file():
                raise FileNotFoundError(f"database does not exist: {database_path}")
            with Database(database_path, read_only=True) as database:
                result = WorkflowReportHandoffService(
                    database,
                    ArtifactStore(args.artifact_root or args.output_root),
                    args.output_root,
                ).compile_plan(
                    args.handoff_id,
                    args.draft,
                    save_draft=not args.dry_run,
                    resources=runtime.resources,
                )
        else:
            result = compile_report_plan_from_files(
                args.draft,
                args.corpus_snapshot,
                args.search_audit,
                args.output_root,
                save_draft=not args.dry_run,
                resources=runtime.resources,
            )
        return {
            "command": "report.plan",
            "draft_path": str(result.path),
            "plan_hash": result.plan["plan_hash"],
            "plan_id": result.plan["plan_id"],
            "status": "validated" if args.dry_run else "draft",
            "write_performed": result.saved,
        }
    if args.diff_from is not None:
        if args.output_root is None:
            raise CliUsageError("report --diff-from requires --output-root")
        report_run_id = args.report_run_id or args.run_id
        if report_run_id is None:
            raise CliUsageError(
                "report --diff-from requires --report-run-id or --run-id"
            )
        return {
            "command": "report.diff",
            "diff": diff_report_runs(
                args.output_root, args.diff_from, report_run_id
            ),
            "previous_report_run_id": args.diff_from,
            "report_run_id": report_run_id,
            "status": "complete",
        }
    if args.plan is not None:
        return _report_execute(args, config=config, runtime=runtime)
    raise CliUsageError(
        "report requires prepare-inputs, --plan-only, approve, --plan, or --diff-from"
    )


def _report_prepare_inputs(
    args: argparse.Namespace, *, config: Mapping[str, Any] | None
) -> dict[str, Any]:
    database_path = _database_path(args.database, config, args.config)
    if not database_path.is_file():
        raise FileNotFoundError(f"database does not exist: {database_path}")
    output_root = args.output_root or _configured_output_root(config, args.config)
    if output_root is None:
        raise ConfigError(
            "report prepare-inputs requires --output-root or a v2 --config"
        )
    if args.workflow_run_id is not None:
        if any(
            value is not None
            for value in (args.crawl_run_id, args.filter_run_id, args.stage4_run_id)
        ) or args.include_needs_review:
            raise CliUsageError(
                "workflow report handoff derives run IDs and include_needs_review "
                "from the completed workflow"
            )
        with Database(database_path, read_only=args.dry_run) as database:
            if not args.dry_run:
                database.migrate()
            result = WorkflowReportHandoffService(
                database,
                ArtifactStore(args.artifact_root or output_root),
                output_root,
            ).prepare(
                WorkflowReportHandoffRequest(
                    args.workflow_run_id,
                    args.recent_cutoff,
                    args.created_at,
                ),
                save_bundle=not args.dry_run,
            )
        document = result.document()
        return {
            **document,
            "command": "report.prepare-workflow-inputs",
            "dry_run": args.dry_run,
        }
    required = {
        "--crawl-run-id": args.crawl_run_id,
        "--filter-run-id": args.filter_run_id,
        "--stage4-run-id": args.stage4_run_id,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise CliUsageError(
            "report prepare-inputs requires --workflow-run-id or "
            + ", ".join(missing)
        )
    with Database(database_path, read_only=True) as database:
        result = ReportInputService(
            database,
            ArtifactStore(args.artifact_root or output_root),
            output_root,
        ).build(
            ReportInputRequest(
                crawl_run_id=args.crawl_run_id,
                filter_run_id=args.filter_run_id,
                stage4_run_id=args.stage4_run_id,
                recent_cutoff=args.recent_cutoff,
                created_at=args.created_at,
                include_needs_review=args.include_needs_review,
            ),
            save_bundle=not args.dry_run,
        )
    return {
        "bundle_id": result.bundle_id,
        "command": "report.prepare-inputs",
        "corpus_snapshot_hash": result.corpus_snapshot["snapshot_hash"],
        "corpus_snapshot_path": str(result.corpus_snapshot_path),
        "directory": str(result.directory),
        "dry_run": args.dry_run,
        "search_audit_hash": result.search_audit["pack_hash"],
        "search_audit_path": str(result.search_audit_path),
        "status": "validated" if args.dry_run else "complete",
        "write_performed": result.saved,
    }


def _report_execute(
    args: argparse.Namespace,
    *,
    config: Mapping[str, Any] | None = None,
    runtime: ReportRuntimeConfig | None = None,
) -> dict[str, Any]:
    if config is None and args.config is not None:
        config = load_config(args.config)
    runtime = runtime or _report_runtime_config(config, args.config)
    if not runtime.enabled:
        return {
            "alarm_codes": [],
            "command": "report",
            "codex_budget": None,
            "dry_run": args.dry_run,
            "error": None,
            "skipped": True,
            "status": "complete",
        }
    bundle = _load_report_plan_bundle(args.plan)
    runtime.validate_for_run(bundle.plan, execution_mode=args.execution_mode)
    assert_report_plan_resource_binding(bundle.plan, runtime.resources)
    database_path = _database_path(args.database, config, args.config)
    if not database_path.is_file():
        raise FileNotFoundError(f"database does not exist: {database_path}")
    output_root = args.output_root or _configured_output_root(config, args.config)
    if output_root is None:
        raise ConfigError("report execution requires --output-root or a v2 --config")
    policy_path = args.policy or _report_policy_path(config, args.config)
    if policy_path is None:
        raise ConfigError("report execution requires --policy or a v2 --config")
    report_run_id = args.report_run_id or args.run_id
    if report_run_id is None:
        report_run_id = f"report-{content_hash(bundle.plan)[:16]}"
    pipeline_run_id = args.pipeline_run_id or f"{report_run_id}:stage4b"
    grants = _processing_grants(args.processing_grants, args.processing_grant)
    previous = None
    if args.previous_report_run_id is not None:
        previous = load_report_run_bundle(
            output_root, args.previous_report_run_id
        ).diff_input()
    with Database(database_path, read_only=args.dry_run) as database:
        if not args.dry_run:
            database.migrate()
        gate = ProcessingGate(
            ArtifactProcessingPolicy.load(policy_path), GrantStore(database)
        )
        service = ReportExecutionService(
            database,
            ArtifactStore(args.artifact_root or output_root),
            gate,
            ReportArtifactStore(output_root),
            execution_mode=args.execution_mode,
            runtime_config=runtime,
        )
        result = service.run(
            report_run_id,
            pipeline_run_id,
            bundle,
            processing_grants=grants,
            previous=previous,
            dry_run=args.dry_run,
        )
    budget = result.codex_budget
    return {
        "alarm_codes": list(result.alarm_codes),
        "command": "report",
        "codex_budget": (
            {
                "approved_call_limit": budget.approved_call_limit,
                "approved_input_token_limit": budget.approved_input_token_limit,
                "calls_reserved": budget.calls_reserved,
                "input_tokens_reserved": budget.input_tokens_reserved,
            }
            if budget is not None
            else None
        ),
        "dry_run": result.dry_run,
        "error": dict(result.error) if result.error is not None else None,
        "pipeline_run_id": pipeline_run_id,
        "published_path": (
            str(result.direct.published_path)
            if result.direct and result.direct.published_path
            else None
        ),
        "report_run_id": result.report_run_id,
        "skipped": bool(getattr(result, "skipped", False)),
        "status": result.status,
    }


def _workflow(args: argparse.Namespace) -> dict[str, Any]:
    manifest = load_workflow_manifest(args.workflow)
    if (
        args.config is not None
        and args.config.resolve() != manifest.config.resolved_path
    ):
        raise ConfigError("--config must match the frozen workflow config FileRef")
    config_path = manifest.config.resolved_path
    config = load_config(config_path)
    database_path = _database_path(args.database, config, config_path)
    workflow_run_id = args.workflow_run_id or args.run_id or manifest.workflow_id
    if args.command == "resume" and not database_path.is_file():
        raise FileNotFoundError(f"database does not exist: {database_path}")
    stop_token = StopToken()
    if args.dry_run and not database_path.is_file():
        orchestrator = SequentialWorkflowOrchestrator(
            None, manifest, default_stage_adapters(), stop_token=stop_token
        )
        result = orchestrator.run(workflow_run_id, dry_run=True)
    else:
        with Database(database_path, read_only=args.dry_run) as database:
            if not args.dry_run:
                database.migrate()
            orchestrator = SequentialWorkflowOrchestrator(
                database, manifest, default_stage_adapters(), stop_token=stop_token
            )
            with stop_token.install_signal_handlers():
                operation = (
                    orchestrator.resume if args.command == "resume" else orchestrator.run
                )
                result = operation(workflow_run_id, dry_run=args.dry_run)
    result["command"] = args.command
    return result


def _download(args: argparse.Namespace) -> dict[str, Any]:
    if args.config is None:
        raise ConfigError("download requires a v2 --config")
    config = load_config(args.config)
    database_path = _database_path(args.database, config, args.config)
    if not database_path.is_file():
        raise FileNotFoundError(f"database does not exist: {database_path}")
    terms = load_provider_terms(args.provider_terms) if args.provider_terms else None
    authorized_skill = _authorized_skill_options(args)
    if args.dry_run:
        # Probe against a consistent disposable clone.  Stage 3's exact dry
        # run deliberately exercises persisted probe code and rolls it back;
        # using the original database would still create WAL/SHM side effects.
        with TemporaryDirectory(prefix="paper-agent-download-dry-") as directory:
            clone_path = Path(directory) / "papers.sqlite3"
            with Database(clone_path) as database:
                _backup_for_download_dry_run(database_path, database)
                expected = Database.migrations()[-1].version
                if database.current_version() != expected:
                    raise ValueError("download dry-run requires a fully migrated database")
                result = _run_download_service(
                    args,
                    config,
                    database,
                    Path(directory) / "artifacts",
                    terms,
                    authorized_skill,
                )
    else:
        with Database(database_path) as database:
            database.migrate()
            result = _run_download_service(
                args,
                config,
                database,
                args.artifact_root or database_path.parent,
                terms,
                authorized_skill,
            )
    papers = [] if result.run is None else [
        {
            "paper_id": item.paper_id,
            "reason_code": item.reason_code,
            "resumed": item.resumed,
            "status": item.status.value,
        }
        for item in result.run.papers
    ]
    return {
        "command": "download",
        "dry_run": result.dry_run,
        "paper_ids": list(result.paper_ids),
        "papers": papers,
        "planned_decisions": [list(item) for item in result.planned_decisions],
        "authorized_queue_path": (
            str(result.authorized_queue_path)
            if result.authorized_queue_path is not None
            else None
        ),
        "authorization_scope": result.authorization_scope.to_dict(),
        "run_id": result.run_id,
        "status": result.status,
    }


def _run_download_service(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    database: Database,
    artifact_root: Path,
    terms: Mapping[str, Any] | None,
    authorized_skill: AuthorizedSkillHandoffOptions | None,
) -> Stage3DownloadResult:
    snapshot_store = DownloadScopeSnapshotStore(database)
    authorization_scope = _download_scope_binding(args, snapshot_store)
    service = Stage3DownloadService(
        database,
        config,
        config_root=args.config.parent,
        artifact_root=artifact_root,
        provider_terms=terms,
        scope_membership=snapshot_store.contains,
    )
    return service.run(
        paper_ids=args.paper_id,
        filter_run_id=args.filter_run_id,
        include_needs_review=args.include_needs_review,
        authorization_grant_id=args.grant_id,
        run_id=args.run_id,
        dry_run=args.dry_run,
        authorized_skill=authorized_skill,
        authorization_scope=authorization_scope,
    )


def _download_scope_binding(
    args: argparse.Namespace, snapshot_store: DownloadScopeSnapshotStore
) -> DownloadScopeBinding:
    collection = None
    if args.collection_snapshot is not None:
        collection = snapshot_store.load_file(
            args.collection_snapshot, expected_type="collection"
        )
    elif args.collection_snapshot_id is not None:
        collection = snapshot_store.load_id(
            args.collection_snapshot_id, expected_type="collection"
        )
    selection = None
    if args.selection_snapshot is not None:
        selection = snapshot_store.load_file(
            args.selection_snapshot, expected_type="selection"
        )
    elif args.selection_snapshot_id is not None:
        selection = snapshot_store.load_id(
            args.selection_snapshot_id, expected_type="selection"
        )
    if (
        collection is not None
        and args.collection_id is not None
        and args.collection_id != collection.collection_id
    ):
        raise ValueError("--collection-id does not match the collection snapshot")
    return DownloadScopeBinding(
        collection_id=(
            args.collection_id
            if args.collection_id is not None
            else collection.collection_id if collection is not None else None
        ),
        collection_snapshot_hash=(
            collection.snapshot_hash if collection is not None else None
        ),
        selection_snapshot_hash=(
            selection.snapshot_hash if selection is not None else None
        ),
    )


def _backup_for_download_dry_run(source_path: Path, destination: Database) -> None:
    """Copy a consistent source snapshot without touching a quiescent WAL database."""
    wal_path = source_path.with_name(f"{source_path.name}-wal")
    if wal_path.exists() and wal_path.stat().st_size > 0:
        raise ValueError(
            "download dry-run requires a quiescent database; close active writers "
            "and checkpoint the WAL first"
        )
    uri = f"{source_path.resolve().as_uri()}?mode=ro&immutable=1"
    source = sqlite3.connect(uri, uri=True)
    try:
        source.backup(destination.connection)
    finally:
        source.close()


def _analyze(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config) if args.config else None
    database_path = _database_path(args.database, config, args.config)
    if not database_path.is_file():
        raise FileNotFoundError(f"database does not exist: {database_path}")
    policy_path = args.policy or _analysis_policy_path(config, args.config)
    if policy_path is None:
        raise ConfigError("analyze requires --policy or a v2 --config")
    grant_id = args.processing_grant_id
    if grant_id is None and config is not None:
        grant_id = config["analysis"]["remote_model_processing"][
            "processing_grant_id"
        ]
    manifest = load_analysis_input_manifest(args.input)
    manifest_hash = content_hash({
        "paper_ids": manifest.paper_ids,
        "stage3_artifact_ids": manifest.stage3_artifact_ids,
    })
    resolved_run_id = args.run_id or f"analysis-{manifest_hash[:16]}"
    analysis_config = config["analysis"] if config is not None else None
    workers = int(analysis_config["workers"]) if analysis_config is not None else 1
    allow_abstract_only = (
        bool(analysis_config["allow_abstract_only"])
        if analysis_config is not None
        else True
    )
    output_schema_path = (
        _config_resource_path(
            args.config,
            Path(str(analysis_config["output_schema"])),
        )
        if analysis_config is not None and args.config is not None
        else None
    )
    with Database(database_path, read_only=args.dry_run) as database:
        if not args.dry_run:
            database.migrate()
        service = AnalysisCliService(
            database,
            ArtifactStore(args.artifact_root or database_path.parent),
            ArtifactProcessingPolicy.load(policy_path),
            grants=GrantStore(database),
            workers=workers,
            allow_abstract_only=allow_abstract_only,
            output_schema_path=output_schema_path,
        )
        if not args.dry_run:
            _analysis_codex_preflight()
        result = service.run(
            resolved_run_id,
            manifest,
            processing_grant_id=grant_id,
            dry_run=args.dry_run,
        )
    papers = [] if result.result is None else [
        {
            "error": item.error,
            "input_scope": item.input_scope,
            "paper_id": item.paper_id,
            "resumed": item.resumed,
            "status": item.status,
        }
        for item in result.result.papers
    ]
    status = (
        "validated"
        if result.dry_run
        else "complete"
        if papers and all(item["status"] == "complete" for item in papers)
        else "incomplete"
    )
    return {
        "command": "analyze",
        "dry_run": result.dry_run,
        "input_scopes": list(result.input_scopes),
        "paper_ids": list(result.selected_paper_ids),
        "papers": papers,
        "run_id": result.run_id,
        "status": status,
    }


def _analysis_codex_preflight() -> None:
    """Check the Stage 4 Codex runtime without making a paid model call."""
    report = CodexExec().doctor(prove_model_availability=False)
    if report.model_availability[ANALYSIS_PROFILE] == "unavailable":
        raise ConfigError("Stage 4 Luna model is unavailable in the Codex catalog")


def _analysis_policy_path(
    config: Mapping[str, Any] | None, config_path: Path | None
) -> Path | None:
    if config is None or config_path is None:
        return None
    value = Path(
        str(config["analysis"]["remote_model_processing"]["policy_matrix"])
    )
    return _config_resource_path(config_path, value)


def _report_runtime_config(
    config: Mapping[str, Any] | None, config_path: Path | None
) -> ReportRuntimeConfig:
    if config is None:
        return ReportRuntimeConfig.defaults()
    if config_path is None:
        raise ConfigError("report configuration path is required")
    return ReportRuntimeConfig.from_config(config, config_path)


def _report_policy_path(
    config: Mapping[str, Any] | None, config_path: Path | None
) -> Path | None:
    if config is None or config_path is None:
        return None
    value = Path(
        str(config["summary"]["remote_model_processing"]["policy_matrix"])
    )
    return _config_resource_path(config_path, value)


def _configured_output_root(
    config: Mapping[str, Any] | None, config_path: Path | None
) -> Path | None:
    if config is None or config_path is None:
        return None
    value = Path(str(config["project"]["output_dir"]))
    return value if value.is_absolute() else config_path.parent / value


def _config_resource_path(config_path: Path, value: Path) -> Path:
    if value.is_absolute():
        return value
    configured = config_path.parent / value
    if configured.is_file():
        return configured
    repository = Path(__file__).resolve().parents[2] / value
    if repository.is_file():
        return repository
    installed = Path(sysconfig.get_path("data")) / "share" / "paper-agent" / value
    return installed if installed.is_file() else configured


def _load_report_plan_bundle(path: Path) -> ReportPlanBundle:
    directory = path.parent
    return ReportPlanBundle(
        _load_mapping(path, "ReportPlan"),
        _load_mapping(directory / "CORPUS_SNAPSHOT.json", "corpus snapshot"),
        _load_mapping(directory / "SEARCH_AUDIT.json", "search audit"),
    )


def _load_mapping(path: Path, label: str) -> Mapping[str, Any]:
    value = _load_json(path)
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _processing_grants(path: Path | None, values: Sequence[str]) -> dict[str, str]:
    grants: dict[str, str] = {}
    if path is not None:
        document = _load_mapping(path, "processing grants")
        if set(document) == {"schema_version", "grants"}:
            if document["schema_version"] != "1" or not isinstance(
                document["grants"], Mapping
            ):
                raise ValueError("processing grants must use schema_version 1")
            document = document["grants"]
        if not all(
            isinstance(artifact_hash, str)
            and len(artifact_hash) == 64
            and all(character in "0123456789abcdef" for character in artifact_hash)
            and isinstance(grant_id, str)
            and grant_id
            for artifact_hash, grant_id in document.items()
        ):
            raise ValueError("processing grants must map artifact SHA-256 hashes to grant IDs")
        grants.update(document)
    for value in values:
        artifact_hash, separator, grant_id = value.partition("=")
        if (
            not separator
            or len(artifact_hash) != 64
            or any(character not in "0123456789abcdef" for character in artifact_hash)
            or not grant_id
        ):
            raise ValueError("--processing-grant must be ARTIFACT_HASH=GRANT_ID")
        grants[artifact_hash] = grant_id
    return grants


def _authorized_skill_options(
    args: argparse.Namespace,
) -> AuthorizedSkillHandoffOptions | None:
    supplied = (
        args.authorized_skill_queue,
        args.authorized_skill_output,
        *args.authorized_skill_root,
        args.authorized_skill_zip,
        args.authorized_skill_audit,
    )
    if not any(value is not None for value in supplied):
        return None
    if (
        args.authorized_skill_queue is None
        or args.authorized_skill_output is None
        or not args.authorized_skill_root
    ):
        raise CliUsageError(
            "authorized skill handoff requires --authorized-skill-queue, "
            "--authorized-skill-output, and --authorized-skill-root"
        )
    return AuthorizedSkillHandoffOptions(
        queue_path=args.authorized_skill_queue,
        output_dir=args.authorized_skill_output,
        skill_roots=tuple(args.authorized_skill_root),
        original_zip=args.authorized_skill_zip,
        audit_manifest=args.authorized_skill_audit,
    )


def entrypoint(argv: Sequence[str] | None = None) -> int:
    """Console boundary: emit one structured failure instead of a traceback."""
    try:
        return main(argv, structured_errors=True)
    except Exception as error:
        normalized = _runtime_argv(argv)
        command = _command_from_argv(normalized)
        _emit({
            "command": command,
            "error": str(error),
            "error_type": type(error).__name__,
            "event_code": f"{command}.failed",
            "run_id": _local_run_id_from_argv(normalized),
            "stage": _command_stage(command),
            "status": "failed",
        })
        return 1


def doctor() -> dict[str, object]:
    """Backward-compatible programmatic diagnostic entrypoint."""
    report = SystemDoctor().run()
    python_check = next(check for check in report.checks if check.name == "python")
    return {
        "paper_agent_version": __version__,
        "python_supported": python_check.status == "pass",
        **report.as_dict(),
    }


def _add_grant_scope_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--paper-id", action="append", default=[])
    parser.add_argument("--artifact-hash", action="append", default=[])
    parser.add_argument("--collection-id", action="append", default=[])
    parser.add_argument("--collection-snapshot-hash")
    parser.add_argument("--selection-snapshot-hash")
    parser.add_argument("--domain", action="append", default=[])
    parser.add_argument("--provider")
    parser.add_argument("--model")
    parser.add_argument("--data-category", action="append", default=[])


def _doctor(args: argparse.Namespace):
    try:
        config = load_config(args.config) if args.config else None
    except (OSError, ConfigError):
        # SystemDoctor owns the user-facing configuration diagnostic.
        config = None
    database = args.database or _configured_database(config, args.config)
    defaults = DoctorPaths.defaults()
    paths = DoctorPaths(
        repository_root=defaults.repository_root,
        config_path=args.config,
        database_path=database,
        model_lock_paths=tuple(args.model_lock) or defaults.model_lock_paths,
        stage2_release_path=args.stage2_release,
        query_plan_path=args.query_plan,
        authorized_skill_runtime=_authorized_skill_runtime(config, args),
    )
    return SystemDoctor(
        paths,
        http_probe=_local_http_probe,
        prove_codex_models=args.prove_codex_models,
    ).run()


def _authorized_skill_runtime(
    config: Mapping[str, Any] | None, args: argparse.Namespace
) -> AuthorizedSkillRuntime:
    enabled = False
    if config is not None:
        enabled = bool(config["download"]["authorized_skill"]["enabled"])
    roots = tuple(args.authorized_skill_root) or _default_authorized_skill_roots()
    archive_value = args.authorized_skill_zip or os.environ.get(
        "PAPER_AGENT_AUTHORIZED_SKILL_ZIP"
    )
    return AuthorizedSkillRuntime(
        enabled=enabled,
        skill_roots=roots,
        original_zip=Path(archive_value) if archive_value else None,
        audit_manifest=args.authorized_skill_audit,
    )


def _default_authorized_skill_roots() -> tuple[Path, ...]:
    configured = os.environ.get("PAPER_AGENT_AUTHORIZED_SKILL_ROOTS")
    if configured:
        return tuple(Path(item) for item in configured.split(os.pathsep) if item)
    codex_root = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
    return (codex_root / "skills",)


class _NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        return None


def _local_http_probe(url: str, headers: Mapping[str, str]) -> tuple[int, str]:
    """Probe the already validated loopback oMLX endpoint with bounded I/O."""
    request = Request(url, headers=dict(headers), method="GET")
    opener = build_opener(ProxyHandler({}), _NoRedirectHandler())
    try:
        with opener.open(request, timeout=3) as response:
            payload = response.read(2 * 1024 * 1024 + 1)
            status = int(response.status)
    except HTTPError as error:
        payload = error.read(2 * 1024 * 1024 + 1)
        status = int(error.code)
    if len(payload) > 2 * 1024 * 1024:
        raise OSError("local model inventory response exceeds 2 MiB")
    return status, payload.decode("utf-8", errors="replace")


def _grant_create(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config) if args.config else None
    defaults = _grant_defaults(config, args)
    scope = _grant_scope(args, defaults)
    actions = args.action or list(defaults.get("actions", (args.kind,)))
    purpose = args.purpose or str(defaults.get("purpose", ""))
    mode = args.mode or str(defaults.get("mode", "attended"))
    allow_unattended = args.allow_unattended
    if allow_unattended is None:
        allow_unattended = bool(defaults.get("allow_unattended", False))
    max_papers = args.max_papers if args.max_papers is not None else defaults.get("max_papers")
    expires_at = args.expires_at or defaults.get("authorization_expires_at")
    if not purpose or max_papers is None or expires_at is None:
        raise ValueError("grant create requires --purpose, --max-papers, and --expires-at (or download grant_defaults)")
    draft = create_grant_draft(
        grant_id=args.grant_id,
        kind=args.kind,
        actions=list(actions),
        purpose=purpose,
        mode=mode,
        allow_unattended=allow_unattended,
        scope=scope,
        max_papers=max_papers,
        expires_at=expires_at,
        skill_digest=args.skill_digest or defaults.get("installed_content_sha256"),
        dependency_digest=args.dependency_digest or defaults.get("dependency_lock_sha256"),
        lineage_hash=args.lineage_hash,
    )
    if not args.dry_run:
        if args.output.exists():
            raise FileExistsError(f"grant draft already exists: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(canonical_json(draft))
    return {
        "command": "grant.create",
        "content_hash": draft["content_hash"],
        "draft_path": str(args.output),
        "grant_id": draft["grant_id"],
        "status": "validated" if args.dry_run else "draft",
    }


def _grant_approve(args: argparse.Namespace) -> dict[str, Any]:
    draft = _load_json(args.grant)
    if args.dry_run:
        approved = validate_grant_approval(
            draft,
            args.hash,
            approved_by=args.approved_by,
            approved_at=args.approved_at,
        )
        return {"command": "grant.approve", "content_hash": approved["content_hash"], "grant_id": approved["grant_id"], "status": "validated"}
    database_path = _database_path(
        args.database,
        load_config(args.config) if args.config else None,
        args.config,
    )
    with Database(database_path) as database:
        database.migrate()
        approved = GrantStore(database).approve(
            draft, args.hash, approved_by=args.approved_by, approved_at=args.approved_at,
        )
    return {"command": "grant.approve", "content_hash": approved["content_hash"], "grant_id": approved["grant_id"], "status": "approved"}


def _grant_revoke(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config) if args.config else None
    database_path = _database_path(args.database, config, args.config)
    if not database_path.is_file():
        raise FileNotFoundError(f"database does not exist: {database_path}")
    if args.dry_run:
        with Database(database_path, read_only=True) as database:
            GrantStore(database).validate_revoke(
                args.grant_id, actor=args.actor, event_at=args.event_at
            )
    else:
        with Database(database_path) as database:
            database.migrate()
            GrantStore(database).revoke(args.grant_id, actor=args.actor, event_at=args.event_at)
    return {"command": "grant.revoke", "grant_id": args.grant_id, "status": "validated" if args.dry_run else "revoked"}


def _export(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config) if args.config else None
    database_path = _database_path(args.database, config, args.config)
    if not database_path.is_file():
        raise FileNotFoundError(f"database does not exist: {database_path}")
    _assert_safe_export_destination(database_path, args.output)
    with Database(database_path, read_only=True) as database:
        current_version = database.current_version()
        expected_version = max(migration.version for migration in Database.migrations())
        if current_version != expected_version:
            raise ConfigError(
                "export requires the current SQLite schema: "
                f"found {current_version}, expected {expected_version}"
            )
        inventory = validate_export(PaperRepository(database))
        if args.dry_run:
            return {
                "command": "export",
                "database_path": str(database_path),
                "format": args.format,
                "output_path": str(args.output),
                "planned_export_count": (
                    inventory["jsonl_rows"]
                    if args.format == "jsonl"
                    else inventory["papers"]
                ),
                "planned_paper_count": inventory["papers"],
                "status": "validated",
            }
        exporter = export_jsonl if args.format == "jsonl" else export_csv
        exported_count = exporter(PaperRepository(database), args.output)
    return {"command": "export", "database_path": str(database_path), "exported_count": exported_count, "format": args.format, "output_path": str(args.output), "status": "complete"}


def _assert_safe_export_destination(database_path: Path, output_path: Path) -> None:
    protected = (
        database_path,
        Path(f"{database_path}-wal"),
        Path(f"{database_path}-shm"),
        Path(f"{database_path}-journal"),
    )
    destination = output_path.resolve()
    for candidate in protected:
        if destination == candidate.resolve():
            raise ConfigError("export output must not overwrite the SQLite fact store")
        if output_path.exists() and candidate.exists() and output_path.samefile(candidate):
            raise ConfigError("export output must not alias the SQLite fact store")


def _import_data(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config) if args.config else None
    database_path = _database_path(args.database, config, args.config)
    if not args.input.is_file():
        raise FileNotFoundError(f"import input does not exist: {args.input}")
    _assert_safe_import_source(database_path, args.input)
    importer = {
        "jsonl": import_jsonl,
        "csv": import_csv,
        "legacy-json": import_legacy_json,
    }[args.format]
    if args.dry_run and not database_path.is_file():
        with TemporaryDirectory(prefix="paper-agent-import-") as temporary:
            with Database(Path(temporary) / "validation.sqlite3") as database:
                database.migrate()
                report = importer(PaperRepository(database), args.input, dry_run=True)
    else:
        with Database(database_path, read_only=args.dry_run) as database:
            if not args.dry_run:
                database.migrate()
            report = importer(
                PaperRepository(database), args.input, dry_run=args.dry_run
            )
    return {
        "command": "import",
        "database_path": str(database_path),
        "format": args.format,
        "input_path": str(args.input),
        "counts": dict(report.counts),
        "field_mappings": dict(report.mappings),
        "warnings": list(report.warnings),
        "unmigrated": list(report.unmigrated),
        "status": "validated" if args.dry_run else "complete",
    }


def _assert_safe_import_source(database_path: Path, input_path: Path) -> None:
    source = input_path.resolve()
    for candidate in (
        database_path,
        Path(f"{database_path}-wal"),
        Path(f"{database_path}-shm"),
        Path(f"{database_path}-journal"),
    ):
        if source == candidate.resolve() or (
            candidate.exists() and input_path.samefile(candidate)
        ):
            raise ConfigError("import input must not alias the SQLite fact store")


def _migrate_config(args: argparse.Namespace) -> dict[str, Any]:
    report = migrate_legacy_yaml(args.input)
    if args.write is not None and not args.dry_run:
        write_migrated(report, args.write)
    return {
        "command": "migrate-config",
        "config": report.converted_config,
        "field_mappings": report.field_mappings,
        "input_path": str(args.input),
        "output_path": str(args.write) if args.write else None,
        "status": "validated" if args.write is None or args.dry_run else "written",
        "unmigrated": report.unmigrated,
        "warnings": report.warnings,
    }


def _database_path(database_path: Path | None, config: Mapping[str, Any] | None, config_path: Path | None) -> Path:
    database = database_path or _configured_database(config, config_path)
    if database is None:
        raise ConfigError("--database or a v2 --config with storage.sqlite_path is required")
    return database


def _grant_defaults(
    config: Mapping[str, Any] | None, args: argparse.Namespace
) -> Mapping[str, Any]:
    if args.kind != "download":
        return {}
    if config is None:
        raise ConfigError(
            "download grant drafts require --config and its frozen grant_defaults"
        )
    overrides = {
        "action": args.action,
        "purpose": args.purpose,
        "mode": args.mode,
        "allow_unattended": args.allow_unattended,
        "paper_id": args.paper_id,
        "artifact_hash": args.artifact_hash,
        "collection_id": args.collection_id,
        "collection_snapshot_hash": args.collection_snapshot_hash,
        "selection_snapshot_hash": args.selection_snapshot_hash,
        "domain": args.domain,
        "provider": args.provider,
        "model": args.model,
        "data_category": args.data_category,
        "max_papers": args.max_papers,
        "expires_at": args.expires_at,
        "skill_digest": args.skill_digest,
        "dependency_digest": args.dependency_digest,
        "lineage_hash": args.lineage_hash,
    }
    supplied = sorted(
        name for name, value in overrides.items() if value not in (None, [], ())
    )
    if supplied:
        raise ConfigError(
            "download grant content must come only from grant_defaults; remove CLI overrides: "
            + ", ".join(supplied)
        )
    download = config["download"]
    defaults = download.get("grant_defaults")
    if defaults is None:
        defaults = download["authorized_skill"]["grant_defaults"]
    if defaults["provider"] == "authorized_skill":
        audit = load_audit_record()
        expected_digests = {
            "source_zip_sha256": audit.original_zip_sha256,
            "installed_content_sha256": audit.installed_content_sha256,
            "dependency_lock_sha256": audit.dependency_lock_sha256,
        }
        drifted = sorted(
            key for key, expected in expected_digests.items()
            if defaults.get(key) != expected
        )
        if drifted:
            raise ConfigError(
                "download grant_defaults differ from the checked-in skill audit: "
                + ", ".join(drifted)
            )
    return defaults


def _grant_scope(args: argparse.Namespace, defaults: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "paper_ids": args.paper_id or list(defaults.get("paper_ids", ())),
        "artifact_hashes": args.artifact_hash,
        "collection_ids": args.collection_id,
        "collection_snapshot_hash": args.collection_snapshot_hash or defaults.get("collection_snapshot_hash"),
        "selection_snapshot_hash": args.selection_snapshot_hash or defaults.get("selection_snapshot_hash"),
        "domains": args.domain or list(defaults.get("allowed_domains", ())),
        "provider": defaults["provider"] if args.kind == "download" else args.provider,
        "model": args.model,
        "data_categories": args.data_category,
    }


def _search_plan(
    input_path: Path,
    output_root: Path,
    *,
    config_path: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    draft = load_yaml(input_path)
    config = load_config(config_path) if config_path is not None else None
    venue_specs = _venue_specs(input_path.parent, draft["scope"]["venues"])
    providers = _provider_specs(
        draft.pop("providers"),
        input_path.parent,
        venue_ids=draft["scope"]["venues"],
        plugin_allowlist=plugin_allowlist_from_config(config),
    )
    plan = compile_query_plan(draft, providers=providers, venue_specs=venue_specs)
    store = QueryPlanStore(output_root)
    path = store.draft_path(str(plan["plan_id"]))
    if not dry_run:
        store.save_draft(plan)
    return {
        "command": "search.plan",
        "draft_path": str(path),
        "estimated_max_candidates": plan["budgets"]["max_candidates"],
        "estimated_max_requests": plan["budgets"]["max_requests"],
        "estimated_max_seconds": plan["budgets"]["max_seconds"],
        "plan_hash": plan["plan_hash"],
        "plan_id": plan["plan_id"],
        "status": "validated" if dry_run else plan["status"],
        "write_performed": not dry_run,
    }


def _search_approve(
    plan_path: Path,
    expected_hash: str,
    approved_by: str,
    approved_at: str | None,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    plan = _load_json(plan_path)
    timestamp = approved_at or datetime.now(UTC).isoformat().replace("+00:00", "Z")
    store = QueryPlanStore(_store_root(plan_path))
    if dry_run:
        approved = approve_query_plan(
            plan,
            expected_hash,
            approved_by=approved_by,
            approved_at=timestamp,
        )
    else:
        approved = store.approve_and_save(
            plan,
            expected_hash,
            approved_by=approved_by,
            approved_at=timestamp,
        )
    return {
        "approved_path": str(store.approved_path(str(approved["plan_id"]))),
        "command": "search.approve",
        "latest_path": str(store.latest_path),
        "plan_hash": approved["plan_hash"],
        "plan_id": approved["plan_id"],
        "status": "validated" if dry_run else approved["status"],
        "write_performed": not dry_run,
    }


def _search_run(
    plan_path: Path,
    *,
    database_path: Path | None,
    contact: str | None,
    snapshot_values: Sequence[str],
    stage2_release_path: Path | None,
    config_path: Path | None,
    run_id: str | None,
    dry_run: bool,
    venue_only: bool = False,
    historical_replay: bool = False,
) -> dict[str, Any]:
    plan = _load_json(plan_path)
    config = load_config(config_path) if config_path else None
    plugin_allowlist = plugin_allowlist_from_config(config)
    if config is not None and config_path is not None:
        _assert_config_plan(config, config_path, plan_path, plan, require_hash=not dry_run)
    database = database_path or _configured_database(config, config_path) or (_store_root(plan_path) / "paper-agent.sqlite3")
    snapshots = _snapshot_paths(snapshot_values)
    operator_contact = (
        contact
        or os.environ.get("PAPER_AGENT_CONTACT")
        or os.environ.get("PAPER_AGENT_CONTACT_EMAIL")
    )
    if dry_run:
        runtime = resolve_runtime_providers(
            plan,
            snapshot_paths=snapshots,
            plugin_allowlist=plugin_allowlist,
        )
        release_path = stage2_release_path or _configured_stage2_release()
        if release_path is None:
            raise Stage2ReleaseError(
                "search startup requires --stage2-release or PAPER_AGENT_STAGE2_RELEASE"
            )
        released_stage2 = load_stage2_release(release_path, plan)
        return {
            "command": "search.run",
            "database_path": str(database),
            "plan_hash": plan["plan_hash"],
            "plan_id": plan["plan_id"],
            "provider_invocation": "skipped_dry_run",
            "resolved_providers": sorted(
                provider["provider"] for provider in runtime if provider["resolved"]
            ),
            "stage2_release_hash": released_stage2.release_hash,
            "status": "runtime_validated",
        }
    result, resolved_run_id, crawl_run_id = execute_search_plan(
        plan,
        database,
        run_id=run_id,
        contact=operator_contact,
        snapshot_paths=snapshots,
        stage2_release_path=stage2_release_path,
        venue_only=venue_only,
        historical_replay=historical_replay,
        plugin_allowlist=plugin_allowlist,
    )
    return {
        "alarm_codes": list(getattr(result, "alarm_codes", ())),
        "command": "search.run",
        "crawl_run_id": crawl_run_id,
        "database_path": str(database),
        "plan_hash": plan["plan_hash"],
        "plan_id": plan["plan_id"],
        "provider_invocation": "completed",
        "provider_outcomes": {
            outcome.provider: outcome.status for outcome in result.fanout.outcomes
        },
        "paper_count": len(result.paper_ids),
        "arxiv_candidate_count": len(result.arxiv_candidate_ids),
        "run_id": resolved_run_id,
        "stage2": dict(getattr(result, "stage2_metrics", {})),
        "status": result.status,
    }


def _expand_citations(plan_path: Path, seeds_path: Path, round_index: int) -> dict[str, Any]:
    plan = _load_json(plan_path)
    runtime = {"providers": plan["providers"], "budgets": plan["budgets"], "execution": plan["execution"]}
    assert_runtime_matches(
        plan,
        runtime["providers"],
        budgets=runtime["budgets"],
        policies=runtime["execution"],
        include_arxiv_candidates=plan["scope"]["include_arxiv_candidates"],
    )
    seeds = _selected_seeds(_load_json(seeds_path))
    snowball = plan["citation_snowball"]
    requests = schedule_requests(
        seeds,
        providers=[
            provider["provider"]
            for provider in plan["providers"]
            if provider["resolved"] and "citation" in provider["roles"]
        ],
        directions=[CitationEdgeType(direction) for direction in snowball["directions"]],
        max_requests=int(plan["budgets"]["max_requests"]),
        max_candidates_per_request=int(snowball["max_per_seed_per_source"]),
    )
    manifest = _citation_manifest(plan, seeds, requests, round_index)
    return {"command": "search.expand-citations", **manifest}


def _crawl(
    venue_ids: Sequence[str],
    *,
    plan_path: Path | None = None,
    database_path: Path | None = None,
    contact: str | None = None,
    snapshot_values: Sequence[str] = (),
    stage2_release_path: Path | None = None,
    config_path: Path | None = None,
    run_id: str | None = None,
    dry_run: bool = False,
    historical_replay: bool = False,
) -> dict[str, Any]:
    catalog = load_catalog()
    normalized_ids = sorted(set(venue_ids))
    venues = [catalog.venue(venue_id) for venue_id in normalized_ids]
    if plan_path is not None:
        plan = _load_json(plan_path)
        _assert_venue_only_plan(plan, normalized_ids)
        result = _search_run(
            plan_path,
            database_path=database_path,
            contact=contact,
            snapshot_values=snapshot_values,
            stage2_release_path=stage2_release_path,
            config_path=config_path,
            run_id=run_id,
            dry_run=dry_run,
            venue_only=True,
            historical_replay=historical_replay,
        )
        return {
            **result,
            "command": "crawl",
            "mode": "venue_descriptor_compatibility",
            "venue_ids": normalized_ids,
        }
    return {
        "command": "crawl",
        "mode": "venue_descriptor_compatibility",
        "search_audit_intent": {
            "event": "venue_descriptor_discovery_planned",
            "providers": sorted({venue["primary_provider"] for venue in venues}),
            "venue_ids": [venue["venue_id"] for venue in venues],
        },
    }


def _stage1_collect(args: argparse.Namespace) -> dict[str, Any]:
    contact = str(args.contact or os.environ.get("PAPER_AGENT_CONTACT") or "").strip()
    if not contact:
        raise ValueError("stage1 collect requires --contact or PAPER_AGENT_CONTACT")
    catalog = load_catalog()
    transport = ControlledHTTPTransport(contact, timeout_seconds=30)
    request = Stage1Request(
        venue_ids=tuple(args.venue),
        year_from=args.year_from,
        year_to=args.year_to,
        page_size=args.page_size,
        max_workers=args.max_workers,
        strict=not args.allow_incomplete,
    )
    result = collect_stage1_metadata(
        request,
        catalog=catalog,
        run_id=args.run_id,
        # Each venue-year owns its audit buffer/cache while sharing the same
        # ProviderRuntime, so source QPS/concurrency limits remain global and
        # response-audit slicing remains thread-safe.
        adapter_factory=lambda descriptor: _stage1_live_adapter(
            descriptor,
            contact=contact,
            runtime=transport.runtime,
        ),
    )
    try:
        published = write_stage1_result(
            result,
            output_path=args.output,
            receipt_path=args.receipt,
            allow_incomplete=args.allow_incomplete,
        )
    except Stage1IncompleteError as error:
        blocked = error.result
        return {
            "command": "stage1.collect",
            "run_id": blocked.run_id,
            "status": "incomplete",
            "record_count": len(blocked.records),
            "output": None,
            "receipt": str(blocked.receipt_path),
            "failed_units": [
                {
                    "venue_id": unit.venue_id,
                    "year": unit.year,
                    "status": unit.status,
                    "reasons": list(unit.reasons),
                }
                for unit in blocked.receipt.units
                if unit.status not in {"complete", "not_applicable"}
            ],
            "not_applicable_units": sum(
                unit.status == "not_applicable" for unit in blocked.receipt.units
            ),
        }
    return {
        "command": "stage1.collect",
        "run_id": published.run_id,
        "status": published.status,
        "record_count": len(published.records),
        "output": str(published.output_path),
        "receipt": str(published.receipt_path),
        "metadata_sha256": published.receipt.metadata_sha256,
    }


def _stage1_live_adapter(
    descriptor: Any, *, contact: str, runtime: Any
) -> CensusCapturingAdapter:
    unit_transport = ControlledHTTPTransport(
        contact,
        timeout_seconds=30,
        runtime=runtime,
    )
    return CensusCapturingAdapter(
        create_builtin(descriptor.provider, unit_transport),
        unit_transport,
    )


def _assert_venue_only_plan(
    plan: Mapping[str, Any],
    venue_ids: Sequence[str],
) -> None:
    if sorted(set(plan["scope"]["venues"])) != list(venue_ids):
        raise ValueError("crawl venues must exactly match the approved QueryPlan scope")
    operations = tuple(plan.get("venue_operations", ()))
    if {str(operation["venue_id"]) for operation in operations} != set(venue_ids):
        raise ValueError("crawl QueryPlan must freeze every requested venue operation")
    primary_providers = {
        str(operation["descriptor"]["provider"]) for operation in operations
    }
    graph_providers = primary_providers | {
        str(fallback["provider"])
        for operation in operations
        for fallback in operation["fallbacks"]
    }
    resolved_providers = {
        str(provider["provider"]) for provider in plan["providers"] if provider["resolved"]
    }
    if not primary_providers.issubset(resolved_providers):
        raise ValueError("crawl QueryPlan must resolve every frozen venue primary provider")
    if not resolved_providers.issubset(graph_providers):
        raise ValueError("crawl QueryPlan may resolve only providers in the frozen venue graph")
    if plan["scope"].get("user_seeds"):
        raise ValueError("crawl QueryPlan cannot contain user seeds")
    if plan["citation_snowball"]["enabled"]:
        raise ValueError("crawl QueryPlan must disable citation expansion")
    if set(plan["execution"]["required_roles"]) != {"venue_primary"}:
        raise ValueError("crawl QueryPlan must require only venue_primary")
    if set(plan["execution"]["required_providers"]) != primary_providers:
        raise ValueError("crawl QueryPlan must require every venue primary provider")


def _import_seeds(
    database_path: Path,
    seed_values: Sequence[str],
    input_paths: Sequence[Path],
    run_id: str | None,
    dry_run: bool,
) -> dict[str, Any]:
    inputs = (*tuple(seed_input(value) for value in seed_values), *inputs_from_files(input_paths))
    if not inputs:
        raise ValueError("import-seeds requires at least one --seed or --input")
    if dry_run:
        validate_seed_inputs(inputs)
        return {
            "command": "import-seeds",
            "database_path": str(database_path),
            "input_count": len(inputs),
            "status": "validated",
        }
    result = import_seeds(database_path, inputs, run_id=run_id)
    return {
        "command": "import-seeds",
        "database_path": str(database_path),
        "imported_count": result.imported_count,
        "input_count": result.input_count,
        "paper_ids": result.paper_ids,
        "run_id": result.run_id,
        "status": "complete",
    }


def _provider_specs(
    value: Any,
    root: Path,
    *,
    venue_ids: Sequence[str],
    plugin_allowlist: tuple[PluginAllowlistEntry, ...] = (),
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("providers must be a list")
    catalog = load_catalog(root if (root / "providers").exists() else None)
    requested_by_provider: dict[str, dict[str, Any]] = {}
    for item in value:
        requested = {"provider": item} if isinstance(item, str) else dict(item)
        unexpected = set(requested) - {"provider", "mode", "snapshot_hash"}
        if unexpected:
            names = ", ".join(sorted(str(name) for name in unexpected))
            raise ValueError(f"provider draft contains protected fields: {names}")
        provider_name = str(requested["provider"])
        if provider_name in requested_by_provider:
            raise ValueError(f"provider draft repeats {provider_name}")
        requested_by_provider[provider_name] = requested
    exact_providers = {
        catalog.venue(Path(str(venue_id)).stem)["primary_provider"]
        for venue_id in venue_ids
    }
    fallback_providers = {
        fallback["provider"]
        for venue_id in venue_ids
        for fallback in catalog.acceptance(Path(str(venue_id)).stem)["fallbacks"]
    }
    for provider in exact_providers | fallback_providers:
        requested_by_provider.setdefault(provider, {"provider": provider})

    plugin_registry = PluginRegistry(plugin_allowlist)
    plugin_registry.verify_requested(
        {
            provider: manifest_from_document(catalog.provider(provider))
            for provider in requested_by_provider
        },
        requested_by_provider,
    )

    specs: list[dict[str, Any]] = []
    for provider_name in sorted(requested_by_provider):
        requested = requested_by_provider[provider_name]
        manifest = catalog.provider(str(requested["provider"]))
        authentication = manifest["authentication"]
        credential_names = _credential_environment_variables(authentication)
        credentials_present = not credential_names or all(
            bool(os.environ.get(name)) for name in credential_names
        )
        credential_availability = {
            name: bool(os.environ.get(name)) for name in credential_names
        }
        spec = {
            "provider": manifest["provider"],
            "distribution": manifest["distribution"],
            "version": manifest["version"],
            "entry_point": manifest["entry_point"],
            "artifact_sha256": manifest["artifact_sha256"],
            "manifest_hash": content_hash(manifest),
            "roles": manifest["roles"],
            "capabilities": manifest["capabilities"],
            "enabled": manifest["enabled"],
            "authority": manifest["authority"],
            "credential_environment_variables": credential_names,
            "credential_availability": credential_availability,
            "rate_limit": manifest["rate_limit"],
            "data_use": manifest["terms"]["data_use"],
            "terms_url": manifest["terms"].get("url"),
            "independence_group": manifest["independence_group"],
            "upstream_families": manifest["upstream_families"],
            "upstream_policies": manifest.get("upstream_policies", {}),
            "mode": "api",
            "credentials_required": authentication["required"],
            "credentials_present": credentials_present,
            "manifest_trusted": bool(
                manifest["builtin"] or provider_name in plugin_registry.registrations
            ),
            "exact_required": provider_name in exact_providers,
        }
        spec.update(requested)
        specs.append(spec)
    return specs


def _venue_specs(root: Path, venue_ids: Sequence[str]) -> list[dict[str, Any]]:
    catalog = load_catalog(root if (root / "providers").exists() else None)
    return [
        {
            "descriptor": catalog.venue(Path(str(venue_id)).stem),
            "acceptance": catalog.acceptance(Path(str(venue_id)).stem),
        }
        for venue_id in sorted(set(venue_ids))
    ]


def _credential_environment_variables(authentication: Mapping[str, Any]) -> tuple[str, ...]:
    declared = authentication.get("credential_envs", ())
    names = declared.values() if isinstance(declared, Mapping) else declared
    if "credential_env" in authentication:
        names = (*names, authentication["credential_env"])
    return tuple(sorted(str(name) for name in names))


def _configured_database(config: Mapping[str, Any] | None, config_path: Path | None) -> Path | None:
    if config is None or config_path is None:
        return None
    path = Path(str(config["storage"]["sqlite_path"]))
    return path if path.is_absolute() else config_path.parent / path


def _configured_stage2_release() -> Path | None:
    value = os.environ.get("PAPER_AGENT_STAGE2_RELEASE")
    return Path(value) if value else None


def _assert_config_plan(
    config: Mapping[str, Any],
    config_path: Path,
    plan_path: Path,
    plan: Mapping[str, Any],
    *,
    require_hash: bool,
) -> None:
    approved = config["sources"]["approved_plan"]
    configured_path = Path(str(approved["input_path"]))
    configured_path = configured_path if configured_path.is_absolute() else config_path.parent / configured_path
    if configured_path.resolve() != plan_path.resolve():
        raise ConfigError("configured approved QueryPlan path does not match --plan")
    if require_hash and approved["content_hash"] is None:
        raise ConfigError("search execution requires an approved QueryPlan hash in config")
    if approved["content_hash"] is not None and approved["content_hash"] != plan["plan_hash"]:
        raise ConfigError("configured approved QueryPlan hash does not match --plan")


def _snapshot_paths(values: Sequence[str]) -> dict[str, Path]:
    snapshots = {}
    for value in values:
        provider, separator, path = value.partition("=")
        if not separator or not provider or not path:
            raise ValueError("--snapshot must be PROVIDER=PATH")
        snapshots[provider] = Path(path)
    return snapshots


def _selected_seeds(payload: Any) -> tuple[SelectedSeed, ...]:
    values = payload["seeds"] if isinstance(payload, dict) else payload
    return tuple(
        SelectedSeed(
            paper_id=str(seed["paper_id"]),
            seed_reason=str(seed["seed_reason"]),
            parent_round=int(seed["parent_round"]),
            depth=int(seed["depth"]),
            subquestion_id=seed.get("subquestion_id"),
            rank=int(seed["rank"]),
            selector_version=str(seed["selector_version"]),
            selector_config_hash=str(seed["selector_config_hash"]),
        )
        for seed in values
    )


def _citation_manifest(
    plan: Mapping[str, Any],
    seeds: Sequence[SelectedSeed],
    requests: Sequence[CitationRequest],
    round_index: int,
) -> dict[str, Any]:
    serialized_seeds = [
        {
            "paper_id": seed.paper_id,
            "seed_reason": seed.seed_reason,
            "parent_round": seed.parent_round,
            "depth": seed.depth,
            "subquestion_id": seed.subquestion_id,
            "rank": seed.rank,
            "selector_version": seed.selector_version,
            "selector_config_hash": seed.selector_config_hash,
        }
        for seed in seeds
    ]
    serialized_requests = [
        {
            "provider": request.provider,
            "direction": request.direction.value,
            "seed_paper_id": request.seed_paper_id,
            "depth": request.depth,
            "seed_rank": request.seed_rank,
            "schedule_order": request.schedule_order,
            "max_candidates": request.max_candidates,
        }
        for request in requests
    ]
    return {
        "plan_hash": plan["plan_hash"],
        "plan_id": plan["plan_id"],
        "request_schedule_hash": content_hash(serialized_requests),
        "requests": serialized_requests,
        "round_index": round_index,
        "seed_manifest_hash": content_hash(serialized_seeds),
        "seeds": serialized_seeds,
    }


def _store_root(plan_path: Path) -> Path:
    return plan_path.parent.parent.parent


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _runtime_argv(argv: Sequence[str] | None) -> list[str]:
    """Allow the three global runtime flags before or after any subcommand."""
    values = list(sys.argv[1:] if argv is None else argv)
    prefix: list[str] = []
    remainder: list[str] = []
    index = 0
    extract = True
    while index < len(values):
        value = values[index]
        if value == "--":
            extract = False
            remainder.append(value)
            index += 1
            continue
        if extract and value in {"--config", "--run-id"}:
            prefix.append(value)
            if index + 1 < len(values):
                prefix.append(values[index + 1])
                index += 2
            else:
                index += 1
            continue
        if extract and any(
            value.startswith(f"{option}=") for option in ("--config", "--run-id")
        ):
            prefix.append(value)
            index += 1
            continue
        if extract and value == "--dry-run":
            prefix.append(value)
            index += 1
            continue
        remainder.append(value)
        index += 1
    return [*prefix, *remainder]


def _runtime_option(argv: Sequence[str], option: str) -> str | None:
    result: str | None = None
    for index, value in enumerate(argv):
        if value == option and index + 1 < len(argv):
            result = argv[index + 1]
        elif value.startswith(f"{option}="):
            result = value.partition("=")[2]
    return result


def _local_run_id_from_argv(argv: Sequence[str]) -> str | None:
    for option in (
        "--run-id",
        "--workflow-run-id",
        "--report-run-id",
        "--campaign-id",
        "--crawl-run-id",
    ):
        value = _runtime_option(argv, option)
        if value is not None:
            return value
    return None


def _command_from_argv(argv: Sequence[str]) -> str:
    index = 0
    while index < len(argv):
        value = argv[index]
        if value in {"--config", "--run-id"}:
            index += 2
            continue
        if value == "--dry-run" or value.startswith(("--config=", "--run-id=")):
            index += 1
            continue
        break
    if index >= len(argv):
        return "unknown"
    command = argv[index]
    if command in {
        "grant",
        "search",
        "report",
        "stage2-calibration",
        "stage2-evaluator",
        "stage2-rationale",
        "stage2-release",
        "stage2-parity",
        "stage2-tuning",
    } and index + 1 < len(argv):
        if argv[index + 1].startswith("-"):
            return command
        command = f"{command}.{argv[index + 1]}"
    return command


def _command_stage(command: str) -> str:
    if command.startswith(("stage1", "search", "crawl", "import-seeds", "import")):
        return "stage1"
    if command.startswith(("filter", "benchmark-stage2", "stage2-")):
        return "stage2"
    if command.startswith(("report", "verify-report")):
        return "stage4b"
    if command == "download":
        return "stage3"
    if command == "analyze":
        return "stage4"
    if command in {"run", "resume"}:
        return "workflow"
    if command.startswith("grant"):
        return "authorization"
    return "system"


def _finish(
    args: argparse.Namespace,
    payload: Mapping[str, Any],
) -> int:
    document = dict(payload)
    command = str(document.get("command", args.command))
    status = str(document.get("status", "complete"))
    success = status in _SUCCESS_STATUSES
    event = (
        "completed"
        if success
        else status
        if status in _NON_SUCCESS_EVENT_STATUSES
        else "failed"
    )
    document["event_code"] = f"{command}.{event}"
    document.setdefault("stage", _command_stage(command))
    document["status"] = status
    for field in (
        "run_id",
        "workflow_run_id",
        "report_run_id",
        "campaign_id",
        "crawl_run_id",
    ):
        if document.get(field) is not None:
            document["run_id"] = document[field]
            break
    else:
        if args.run_id is not None:
            document["run_id"] = args.run_id
    _emit(document)
    return int(not success)


def _emit(payload: Mapping[str, Any]) -> None:
    print(canonical_json(dict(payload)).decode("utf-8"))
