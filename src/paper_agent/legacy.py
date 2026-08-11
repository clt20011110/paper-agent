"""One-way migration from the retired v1 YAML shape to configuration v2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .schema import validate


@dataclass(frozen=True)
class MigrationReport:
    """The complete result of a dry-run legacy configuration migration."""

    converted_config: dict[str, Any]
    field_mappings: dict[str, str]
    warnings: list[str]
    unmigrated: list[str]

    @property
    def config(self) -> dict[str, Any]:
        return self.converted_config


_VENUE_DESCRIPTORS = {
    "NeurIPS": "venues/neurips.yaml",
    "ICML": "venues/icml.yaml",
    "ICLR": "venues/iclr.yaml",
    "AAAI": "venues/aaai.yaml",
    "ACL": "venues/acl.yaml",
    "CVPR": "venues/cvpr.yaml",
    "ICCV": "venues/iccv.yaml",
    "IJCAI": "venues/ijcai.yaml",
    "DAC": "venues/dac.yaml",
    "ICCAD": "venues/iccad.yaml",
    "TCAD": "venues/tcad.yaml",
    "Nature Machine Intelligence": "venues/nature_machine_intelligence.yaml",
    "Nature Chemistry": "venues/nature_chemistry.yaml",
    "Nature Computational Science": "venues/nature_computational_science.yaml",
    "Nature Communications": "venues/nature_communications.yaml",
    "Nature Catalysis": "venues/nature_catalysis.yaml",
    "Nature Biotechnology": "venues/nature_biotechnology.yaml",
    "Nature Biomedical Engineering": "venues/nature_biomedical_engineering.yaml",
    "Cell": "venues/cell.yaml",
    "Science": "venues/science.yaml",
}


def migrate_legacy_config(
    legacy: dict[str, Any], schema_root: Path | None = None
) -> MigrationReport:
    """Convert a v1 (or unversioned) document without writing to disk."""
    if legacy.get("version") == 2:
        raise ValueError("configuration is already version 2")

    mappings: dict[str, str] = {
        "version": "version",
        "topic": "project.topic",
        "output_dir": "project.output_dir",
        "database.path": "storage.sqlite_path",
        "database.format": "storage.exports",
        "filter.mode": "filter.mode",
        "analysis.workers": "analysis.workers",
        "analysis.generate_summary": "summary.enabled",
        "analysis.model": "analysis.model",
        "analysis.summary_model": "summary.model",
    }
    warnings: list[str] = []
    unmigrated: list[str] = []

    output_dir = str(legacy.get("output_dir", "./paper_research"))
    topic = str(legacy.get("topic", "Literature research"))
    database = _mapping(legacy.get("database"))
    old_database_path = database.get("path")
    if old_database_path:
        warnings.append(
            "database.path is replaced with a SQLite database under project.output_dir"
        )
    if database.get("format") not in (None, "json", "csv", "sqlite"):
        unmigrated.append("database.format")
    for key in ("incremental", "backup"):
        if key in database:
            unmigrated.append(f"database.{key}")

    sources, source_mappings, source_warnings, source_unmigrated = _sources(
        _mapping(legacy.get("sources")), output_dir
    )
    mappings.update(source_mappings)
    warnings.extend(source_warnings)
    unmigrated.extend(source_unmigrated)

    filter_config, filter_mappings, filter_warnings, filter_unmigrated = _filter(
        _mapping(legacy.get("filter")), output_dir
    )
    mappings.update(filter_mappings)
    warnings.extend(filter_warnings)
    unmigrated.extend(filter_unmigrated)

    analysis = _mapping(legacy.get("analysis"))
    if analysis.get("model") is not None:
        warnings.append(
            "analysis.model is retired and replaced by frozen codex exec model gpt-5.6-luna"
        )
    if analysis.get("summary_model") is not None:
        warnings.append(
            "analysis.summary_model is retired and replaced by frozen codex exec model gpt-5.6-sol"
        )
    if _contains_openrouter(legacy):
        warnings.append(
            "OpenRouter configuration is not migrated; frozen codex exec Luna/Sol profiles replace it"
        )
    for key in analysis:
        if key not in {"model", "summary_model", "workers", "generate_summary"}:
            unmigrated.append(f"analysis.{key}")
    download = _mapping(legacy.get("download"))
    unmigrated.extend(f"download.{key}" for key in download)

    converted = {
        "version": 2,
        "project": {
            "topic": topic,
            "output_dir": output_dir,
            "report_language": "zh-CN",
        },
        "storage": {
            "sqlite_path": f"{output_dir}/papers.sqlite3",
            "wal": True,
            "exports": [
                {"format": "jsonl", "path": f"{output_dir}/exports/papers.jsonl"},
                {"format": "csv", "path": f"{output_dir}/exports/papers.csv"},
            ],
        },
        "sources": sources,
        "filter": filter_config,
        "download": _download(),
        "analysis": _analysis(analysis),
        "summary": _summary(analysis),
    }
    validate(converted, "config-v2.schema.json", schema_root)
    return MigrationReport(converted, mappings, warnings, unmigrated)


def migrate_legacy_yaml(path: Path, schema_root: Path | None = None) -> MigrationReport:
    """Load and dry-run a v1 YAML migration."""
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("legacy configuration root must be an object")
    return migrate_legacy_config(document, schema_root)


def write_migrated(report: MigrationReport, path: Path) -> None:
    """Write the converted v2 document selected by a dry-run report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(report.converted_config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _sources(
    legacy: dict[str, Any], output_dir: str
) -> tuple[dict[str, Any], dict[str, str], list[str], list[str]]:
    mappings: dict[str, str] = {}
    warnings: list[str] = []
    unmigrated: list[str] = []
    venues: list[dict[str, Any]] = []
    for group in ("conferences", "journals"):
        entries = legacy.get(group, [])
        if not isinstance(entries, list):
            unmigrated.append(f"sources.{group}")
            continue
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                unmigrated.append(f"sources.{group}[{index}]")
                continue
            name = str(entry.get("name", ""))
            original_name = name
            if name == "Nature Computer Science":
                name = "Nature Computational Science"
                warnings.append(
                    "Nature Computer Science is corrected to Nature Computational Science"
                )
            descriptor = _VENUE_DESCRIPTORS.get(name)
            if descriptor is None:
                descriptor = _fallback_descriptor(name)
                unmigrated.append(f"sources.{group}[{index}].name")
                warnings.append(f"venue {name!r} needs a reviewed v2 descriptor")
            venue: dict[str, Any] = {"descriptor": descriptor}
            years = entry.get("years")
            if name == "TCAD":
                warnings.append(
                    "TCAD is migrated as the IEEE TCAD journal descriptor, not a conference year page"
                )
                if isinstance(years, list) and years:
                    venue["date_from"] = f"{min(years)}-01-01"
                    venue["date_to"] = f"{max(years)}-12-31"
            elif isinstance(years, list):
                venue["years"] = years
            venues.append(venue)
            mappings[
                f"sources.{group}[{index}]"
            ] = f"sources.plan_defaults.venues[{len(venues) - 1}]"
            if entry.get("platform") is not None:
                mappings[
                    f"sources.{group}[{index}].platform"
                ] = f"sources.plan_defaults.venues[{len(venues) - 1}].descriptor"
            if original_name != name:
                mappings[f"sources.{group}[{index}].name"] = (
                    f"sources.plan_defaults.venues[{len(venues) - 1}].descriptor"
                )
            for key in entry:
                if key not in {"name", "platform", "years"}:
                    unmigrated.append(f"sources.{group}[{index}].{key}")

    old_arxiv = _mapping(legacy.get("arxiv"))
    arxiv_enabled = bool(old_arxiv.get("enabled", False))
    include_candidates = bool(old_arxiv.get("save_to_database", False))
    mappings["sources.arxiv.enabled"] = "sources.plan_defaults.arxiv.enabled"
    if "save_to_database" in old_arxiv:
        mappings[
            "sources.arxiv.save_to_database"
        ] = "sources.plan_defaults.arxiv.include_arxiv_candidates"
        warnings.append(
            "arxiv.save_to_database now means arXiv-only papers may enter later stages"
        )
    arxiv: dict[str, Any] = {
        "enabled": arxiv_enabled,
        "roles": ["search", "metadata_enricher", "metadata_verifier", "oa_resolver"],
        "categories": list(old_arxiv.get("categories", [])),
        "include_arxiv_candidates": include_candidates,
        "use_matched_arxiv_as_download_source": True,
        "global_min_interval_seconds": 3,
    }
    date_from = _date_from(old_arxiv.get("date_range"))
    if date_from is not None:
        arxiv["date_from"] = date_from
        mappings["sources.arxiv.date_range"] = "sources.plan_defaults.arxiv.date_from"
        if " to " in str(old_arxiv.get("date_range")):
            unmigrated.append("sources.arxiv.date_range.end")
            warnings.append("arxiv.date_range end is not represented by the v2 arXiv plan default")
    elif old_arxiv.get("date_range") is not None:
        unmigrated.append("sources.arxiv.date_range")
    for key in ("keywords", "max_results"):
        if key in old_arxiv:
            unmigrated.append(f"sources.arxiv.{key}")

    return (
        {
            "approved_plan": {
                "input_path": f"{output_dir}/search/latest-approved.json",
                "content_hash": None,
                "required": True,
            },
            "plan_defaults": {
                "required_roles": [
                    "venue_primary",
                    "search",
                    "citation",
                    "metadata_verifier",
                ],
                "provider_policy": "all_resolved",
                "user_seeds": {"inputs": []},
                "venues": venues,
                "arxiv": arxiv,
                "providers": _providers(),
                "verification": {
                    "prefer_formal_version": True,
                    "require_two_sources_when_feasible": True,
                    "preserve_conflicts": True,
                },
                "citation_snowball": {
                    "enabled": True,
                    "directions": ["references", "citations"],
                    "max_depth": 2,
                    "max_rounds": 2,
                    "seed_selector": "relevant_topk_by_subquestion_v1",
                    "seeds_per_subquestion": 20,
                    "max_per_seed_per_source": 500,
                    "max_candidates_per_round": 50000,
                    "max_seconds_per_round": 14400,
                },
                "saturation": {
                    "min_unique_included_yield": 0.05,
                    "consecutive_low_yield_rounds": 2,
                    "max_candidates": 200000,
                    "max_requests": 20000,
                },
            },
            "plugin_allowlist": [],
        },
        mappings,
        warnings,
        unmigrated,
    )


def _filter(
    legacy: dict[str, Any], output_dir: str
) -> tuple[dict[str, Any], dict[str, str], list[str], list[str]]:
    mappings = {
        "filter.mode": "filter.mode",
        "filter.regex.include_groups": "filter.deterministic.include",
        "filter.regex.exclude": "filter.deterministic.exclude_document_types",
    }
    warnings: list[str] = []
    unmigrated: list[str] = []
    regex = _mapping(legacy.get("regex"))
    include_groups = regex.get("include_groups", [])
    include = [" AND ".join(map(str, group)) for group in include_groups if isinstance(group, list)]
    if include_groups and len(include) != len(include_groups):
        unmigrated.append("filter.regex.include_groups")
    exclude = list(regex.get("exclude", []))
    if legacy.get("mode") not in (None, "cascade"):
        warnings.append("filter.mode is replaced by the frozen v2 cascade")
    if "semantic" in legacy:
        unmigrated.append("filter.semantic")
        warnings.append("filter.semantic is retired; v2 uses the frozen oMLX cascade")
    for key in ("match_fields", "case_sensitive", "whole_word"):
        if key in regex:
            unmigrated.append(f"filter.regex.{key}")
    return (
        {
            "mode": "cascade",
            "deterministic": {
                "include": include,
                "exclude_document_types": ["editorial", "retraction", *exclude],
            },
            "reranker": {
                "backend": "omlx_rerank",
                "model": "BAAI/bge-reranker-v2-m3",
                "source_repo": "BAAI/bge-reranker-v2-m3",
                "source_revision": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
                "format": "fp32",
                "document_batch_size": 32,
                "candidate_batch_sizes": [16, 32, 64],
                "max_in_flight": 2,
                "thresholds_artifact": f"{output_dir}/models/stage2-thresholds.json",
            },
            "adjudicator": _adjudicator(),
            "fail_open": True,
        },
        mappings,
        warnings,
        unmigrated,
    )


def _providers() -> dict[str, Any]:
    return {
        "crossref": {"enabled": True, "roles": ["search", "metadata_enricher", "metadata_verifier"], "mailto_env": "CROSSREF_MAILTO", "rate_policy": "response_headers"},
        "dblp": {"enabled": "auto_for_cs", "roles": ["search", "metadata_enricher", "metadata_verifier"], "bulk_snapshot_preferred": True},
        "semantic_scholar": {"enabled": True, "roles": ["search", "citation", "metadata_enricher", "metadata_verifier"], "api_key_env": "SEMANTIC_SCHOLAR_API_KEY", "use_batch_endpoints": True},
        "openalex": {"enabled": True, "roles": ["search", "citation", "metadata_enricher", "metadata_verifier"], "api_key_env": "OPENALEX_API_KEY", "mode": "api", "snapshot_path": None},
        "pubmed": {"enabled": "auto_for_biomed", "roles": ["search", "metadata_enricher", "metadata_verifier"], "api_key_env": "NCBI_API_KEY", "tool": "paper-agent", "email_env": "NCBI_EMAIL"},
        "europe_pmc": {"enabled": "auto_for_biomed", "roles": ["search", "citation", "metadata_enricher", "metadata_verifier", "oa_resolver"]},
        "unpaywall": {"enabled": True, "roles": ["oa_resolver"], "email_env": "UNPAYWALL_EMAIL"},
    }


def _adjudicator() -> dict[str, Any]:
    return {
        "backend": "omlx_chat",
        "model": "mlx-community/Qwen3.5-9B-8bit",
        "revision": "16daa4818c54ce5f5436f929d52542eb65bbed9d",
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0,
        "seed": 42,
        "stream": False,
        "max_tokens": 256,
        "max_context_window": 16384,
        "structured_output": {
            "transport": "extra_body.structured_outputs.json",
            "schema": "./schemas/filter-decision.schema.json",
        },
        "client_concurrency": 4,
        "server_max_concurrent_requests": 8,
        "benchmark_concurrency_pairs": [[4, 8], [8, 8], [8, 16], [16, 16]],
        "expected_max_share": 0.15,
    }


def _download() -> dict[str, Any]:
    return {
        "include_supplements": False,
        "resolvers": ["publisher_public", "europe_pmc", "unpaywall", "arxiv"],
        "providers": ["public_direct", "europe_pmc", "unpaywall_location", "arxiv", "authorized_skill", "manual"],
        "purpose": "personal_research",
        "policy_matrix": "./policies/download-access-v2.yaml",
        "require_access_basis": True,
        "treat_unknown_license_as_open": False,
        "authorized_skill": {
            "enabled": False,
            "skill_name": "download-authorized-papers",
            "authorization_grant_id": None,
            "data_sharing_grant_id": None,
            "profile": "stage3_authorized_luna",
            "codex_model": "gpt-5.6-luna",
            "reasoning_effort": "low",
            "grant_defaults": {
                "source_zip_sha256": None,
                "installed_content_sha256": None,
                "dependency_lock_sha256": None,
                "allowed_domains": [],
                "paper_ids": [],
                "collection_snapshot_hash": None,
                "selection_snapshot_hash": None,
                "max_papers": None,
                "actions": ["download", "store", "extract"],
                "purpose": "personal_research",
                "mode": "attended",
                "allow_unattended": False,
                "authorization_expires_at": None,
            },
        },
    }


def _analysis(legacy: dict[str, Any]) -> dict[str, Any]:
    return {
        "profile": "stage4_analysis_luna",
        "provider": "codex_exec",
        "model": "gpt-5.6-luna",
        "reasoning_effort": "medium",
        "sandbox": "read_only",
        "network": False,
        "output_schema": "./schemas/paper-analysis.schema.json",
        "workers": int(legacy.get("workers", 4)),
        "allow_abstract_only": True,
        "remote_model_processing": {
            "policy_matrix": "./policies/artifact-processing-v1.yaml",
            "processing_grant_id": None,
            "require_artifact_hash_scope": True,
        },
    }


def _summary(legacy: dict[str, Any]) -> dict[str, Any]:
    return {
        "enabled": bool(legacy.get("generate_summary", True)),
        "profile": "stage4b_summary_sol",
        "provider": "codex_exec",
        "model": "gpt-5.6-sol",
        "reasoning_effort": "high",
        "sandbox": "read_only",
        "network": False,
        "schemas": {
            "planning_assist": "./schemas/report-plan.schema.json",
            "section_reduce": "./schemas/section-synthesis.schema.json",
            "cross_section_reduce": "./schemas/cross-section-synthesis.schema.json",
            "final_reduce": "./schemas/report-document.schema.json",
            "quality_audit": "./schemas/report-audit.schema.json",
            "repair": "./schemas/report-repair.schema.json",
        },
        "prompts": {
            "planning_assist": "./prompts/report-plan.md",
            "section_reduce": "./prompts/section-synthesis.md",
            "cross_section_reduce": "./prompts/cross-section-synthesis.md",
            "final_reduce": "./prompts/final-report.md",
            "quality_audit": "./prompts/report-audit.md",
            "repair": "./prompts/report-repair.md",
        },
        "format": "markdown",
        "language": "zh-CN",
        "report_plan": {
            "input_path": None,
            "content_hash": None,
            "required_for_unattended": True,
            "classification_axes": ["subquestion", "theme", "method_family", "task", "dataset", "benchmark", "time", "publication_status", "evidence_type", "study_setting"],
        },
        "require_search_audit": True,
        "require_complete_coverage": True,
        "require_claim_evidence": True,
        "semantic_chunking": True,
        "remote_model_processing": {
            "policy_matrix": "./policies/artifact-processing-v1.yaml",
            "processing_grant_id": None,
            "require_lineage_hash_scope": True,
        },
        "citations": {"marker": "stable_paper_id", "style": "ieee", "bibliography_from_canonical_metadata": True},
        "final_audit": {
            "deterministic": True,
            "independent_sol_session": True,
            "rubric": "./policies/report-audit-rubric-v1.yaml",
            "max_blocker_findings": 0,
            "max_major_findings": 0,
            "max_repair_calls": 1,
            "reverify_and_reaudit_after_repair": True,
        },
        "immutable_run_directories": True,
        "update_latest_after_pass": True,
        "emit_incremental_diff": True,
    }


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _fallback_descriptor(name: str) -> str:
    slug = "".join(character.lower() if character.isalnum() else "_" for character in name)
    return f"venues/{slug.strip('_') or 'unrecognized'}.yaml"


def _date_from(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    candidate = value.split(" to ", 1)[0]
    return candidate if len(candidate) == 10 else None


def _contains_openrouter(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            "openrouter" in str(key).lower() or _contains_openrouter(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_contains_openrouter(child) for child in value)
    return isinstance(value, str) and "openrouter" in value.lower()
