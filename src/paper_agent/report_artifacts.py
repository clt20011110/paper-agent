"""Deterministic Stage 4b report artifacts and release gates.

The Sol final-reduce response is deliberately kept as a small AST.  This
module is the only place that turns it into Markdown, adds bibliography and
disclosures, and publishes an immutable report run.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import sysconfig
from typing import Any

import yaml

from .canonical import canonical_json, content_hash
from .reporting import (
    CoverageLedger,
    EVIDENCE_LEVEL_RANK,
    EvidenceValidationError,
    INPUT_SCOPE_LEVELS,
    comparison_assessment,
    require_exact_comparison_groups,
    stable_claim_id,
    validate_evidence_reference_shape,
)
from .schema import SchemaValidationError, validate


RENDERER_VERSION = "report-markdown-v1"
SUBSTANTIVE_KINDS = frozenset({"prose", "list_item", "table_cell", "caption"})
LOCAL_REFERENCES_NOTE = (
    "规范参考文献由本地协调器根据冻结的 canonical metadata 生成；"
    "附录保留查询清单、排除原因、覆盖台账与主张台账。"
)
PAPER_MARKER = re.compile(r"@([A-Za-z0-9._:-]+)")
NUMBER = re.compile(r"(?<![A-Za-z0-9._:-])\d+(?:\.\d+)?(?:\s*[%×x])?")
CJK_TEXT = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
CONFLICT_DISCLOSURE = re.compile(r"冲突|矛盾|不一致|相反|分歧")
INCOMPARABLE_DISCLOSURE = re.compile(r"不可(?:直接)?比较|不具可比性")


class ReportArtifactError(ValueError):
    pass


class ReportVerificationError(ReportArtifactError):
    pass


def search_publication_blockers(
    search_audit: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return frozen search conditions that prohibit a final publication."""
    blockers = []
    if search_audit.get("search_status") != "complete":
        blockers.append("search_status is not complete")
    required_failures = search_audit.get("required_provider_failures")
    if (
        not isinstance(required_failures, Sequence)
        or isinstance(required_failures, (str, bytes))
    ):
        blockers.append("required provider status is missing or invalid")
    elif required_failures:
        blockers.append(
            "required providers failed: "
            + ", ".join(sorted(str(item) for item in required_failures))
        )
    if search_audit.get("budget_exhausted") is not False:
        blockers.append("search budget is exhausted")
    return tuple(blockers)


def _json(value: Any) -> str:
    return canonical_json(value).decode("utf-8") + "\n"


def _claim_map(claims: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    mapped = {str(claim["claim_id"]): claim for claim in claims}
    if len(mapped) != len(claims):
        raise ReportVerificationError("CLAIMS_EVIDENCE contains duplicate claim_id")
    return mapped


def _paper_ids(claim: Mapping[str, Any]) -> set[str]:
    return {
        str(ref["paper_id"])
        for field in ("supporting_evidence", "contradicting_evidence")
        for ref in claim[field]
        if ref["kind"] == "paper_evidence" and ref["paper_id"] is not None
    }


def _numeric_evidence(claim: Mapping[str, Any]) -> bool:
    for field in ("supporting_evidence", "contradicting_evidence"):
        for ref in claim[field]:
            if ref["kind"] == "corpus_evidence" and ref.get("statistic"):
                return True
            unit = ref.get("evidence_unit")
            if isinstance(unit, Mapping):
                for key in ("value", "source_value"):
                    value = unit.get(key)
                    if (
                        isinstance(value, (int, float))
                        and not isinstance(value, bool)
                    ) or (
                        isinstance(value, str) and NUMBER.search(value) is not None
                    ):
                        return True
    return False


def is_local_references_block(block: Mapping[str, Any]) -> bool:
    """Recognize the sole coordinator-owned block allowed without claims."""
    return (
        block.get("block_kind") == "caption"
        and block.get("section_id") == "references_and_appendices"
        and block.get("text") == LOCAL_REFERENCES_NOTE
        and list(block.get("claim_ids", ())) == []
        and list(block.get("citation_paper_ids", ())) == []
    )


def _coverage_dict(coverage: CoverageLedger | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(coverage, CoverageLedger):
        return {
            "papers": [
                {
                    "paper_id": item.paper_id,
                    "evidence_claim_ids": list(item.evidence_claim_ids),
                    "consumed_node_ids": list(item.consumed_node_ids),
                    "disposition": item.disposition,
                    "reason": item.reason,
                }
                for item in coverage.papers
            ],
            "missing_paper_ids": list(coverage.missing_paper_ids),
            "uncovered_claim_ids": list(coverage.uncovered_claim_ids),
            "complete": coverage.complete,
        }
    return deepcopy(dict(coverage))


def _bibliography_entry(paper_id: str, metadata: Mapping[str, Any]) -> str:
    title = str(metadata.get("title", "")).strip()
    authors = metadata.get("authors")
    year = metadata.get("year")
    venue = str(metadata.get("venue_name") or metadata.get("venue") or "").strip()
    locator = str(metadata.get("doi") or metadata.get("canonical_url") or "").strip()
    if not title or not isinstance(authors, Sequence) or isinstance(authors, str) or not authors:
        raise ReportVerificationError(f"canonical metadata is incomplete for citation {paper_id}")
    if not isinstance(year, int) or not venue or not locator:
        raise ReportVerificationError(f"canonical metadata is incomplete for citation {paper_id}")
    author_text = ", ".join(str(author).strip() for author in authors if str(author).strip())
    if not author_text:
        raise ReportVerificationError(f"canonical metadata is incomplete for citation {paper_id}")
    return f"[@{paper_id}] {author_text}. {title}. {venue}, {year}. {locator}"


def _sections(plan: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    sections = tuple((str(item["id"]), str(item["title"])) for item in plan["sections"])
    if not sections or len({item[0] for item in sections}) != len(sections):
        raise ReportVerificationError("ReportPlan sections are missing or duplicate")
    return sections


def _validate_report_language(
    plan: Mapping[str, Any], claims: Sequence[Mapping[str, Any]]
) -> None:
    if plan.get("report_language") != "zh-CN":
        raise ReportVerificationError("ReportPlan report_language must be zh-CN")
    if CJK_TEXT.search(str(plan.get("objective") or "")) is None:
        raise ReportVerificationError("zh-CN ReportPlan objective must contain CJK text")
    for section_id, title in _sections(plan):
        if CJK_TEXT.search(title) is None:
            raise ReportVerificationError(
                f"zh-CN section title must contain CJK text: {section_id}"
            )
    for claim in claims:
        if CJK_TEXT.search(str(claim.get("claim_text") or "")) is None:
            raise ReportVerificationError(
                "zh-CN claim_text must contain CJK text: "
                + str(claim.get("claim_id") or "<missing>")
            )


def _publication_status_disclosure(
    corpus_snapshot: Mapping[str, Any],
) -> str | None:
    counts = {"preprint": 0, "workshop": 0}
    for paper in corpus_snapshot.get("papers", ()):
        status = str(paper.get("publication_status") or "")
        if status in counts:
            counts[status] += 1
    cohorts = tuple(
        (label, counts[status])
        for status, label in (("preprint", "预印本"), ("workshop", "研讨会论文"))
        if counts[status]
    )
    if not cohorts:
        return None
    return (
        "出版状态分层："
        + "、".join(f"{label}={count}" for label, count in cohorts)
        + "；预印本和研讨会论文与正式同行评审论文分层呈现，不视为同等证据。"
    )


def _disclosures(search_audit: Mapping[str, Any], corpus_snapshot: Mapping[str, Any] | None) -> tuple[str, ...]:
    limitations = tuple(str(item) for item in search_audit.get("limitations", ()))
    snapshot = corpus_snapshot or {}
    scope = dict(snapshot.get("input_scope", {}))
    if not scope and snapshot.get("papers"):
        for paper in snapshot["papers"]:
            input_scope = str(paper.get("input_scope", "missing"))
            scope[input_scope] = int(scope.get(input_scope, 0)) + 1
    parts = list(limitations)
    if scope:
        parts.append(
            "抽取范围："
            + "、".join(f"{key}={scope[key]}" for key in sorted(scope))
            + "；全文、摘要和元数据证据已分层，缺失全文不作全文事实表述。"
        )
    publication_status = _publication_status_disclosure(snapshot)
    if publication_status is not None:
        parts.append(publication_status)
    return tuple(dict.fromkeys(parts))


def audit_coverage_ledger(
    document: Mapping[str, Any], claims: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Return the exact substantive units an exhaustive Sol audit must enumerate."""
    evidence_refs = []
    for claim in sorted(claims, key=lambda item: str(item["claim_id"])):
        claim_id = str(claim["claim_id"])
        for field, direction in (
            ("supporting_evidence", "support"),
            ("contradicting_evidence", "contradict"),
        ):
            for ordinal, reference in enumerate(claim[field]):
                evidence_refs.append({
                    "claim_id": claim_id,
                    "direction": direction,
                    "ordinal": ordinal,
                    "evidence_ref_hash": content_hash(reference),
                })
    return {
        "block_ids": sorted(str(item["block_id"]) for item in document["blocks"]),
        "claim_ids": sorted(str(item["claim_id"]) for item in claims),
        "evidence_refs": evidence_refs,
    }


def report_artifact_hash(
    *,
    document: Mapping[str, Any],
    claims: Sequence[Mapping[str, Any]],
    coverage: CoverageLedger | Mapping[str, Any],
    comparison_groups: Mapping[str, Mapping[str, Any]],
    claim_relations: Sequence[Mapping[str, Any]],
    bibliography: Mapping[str, Mapping[str, Any]],
) -> str:
    """Bind every mutable or generated structured input to one audit digest."""
    return content_hash({
        "document": document,
        "claims": list(claims),
        "coverage": _coverage_dict(coverage),
        "comparison_groups": comparison_groups,
        "claim_relations": list(claim_relations),
        "bibliography": bibliography,
    })


def audit_search_limitations(
    search_audit: Mapping[str, Any], corpus_snapshot: Mapping[str, Any]
) -> tuple[str, ...]:
    """Return the exact limitation units bound into audit prompts and outputs."""
    return _disclosures(search_audit, corpus_snapshot)


def audit_rubric_hash(path: str | Path | None = None) -> str:
    if path is not None:
        rubric_path = Path(path)
    else:
        repository = Path(__file__).resolve().parents[2] / "policies" / "report-audit-rubric-v1.yaml"
        rubric_path = repository if repository.is_file() else (
            Path(sysconfig.get_path("data"))
            / "share" / "paper-agent" / "policies" / "report-audit-rubric-v1.yaml"
        )
    try:
        document = yaml.safe_load(rubric_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ReportVerificationError("frozen report audit rubric is unavailable") from error
    if not isinstance(document, Mapping):
        raise ReportVerificationError("frozen report audit rubric must be an object")
    return content_hash(document)


def _validate_audit_binding(
    audit: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    document: Mapping[str, Any],
    claims: Sequence[Mapping[str, Any]],
    coverage: CoverageLedger | Mapping[str, Any],
    comparison_groups: Mapping[str, Mapping[str, Any]],
    claim_relations: Sequence[Mapping[str, Any]],
    bibliography: Mapping[str, Mapping[str, Any]],
    search_audit: Mapping[str, Any],
    corpus_snapshot: Mapping[str, Any],
    rubric_path: str | Path | None = None,
) -> None:
    expected = (
        content_hash(document),
        report_artifact_hash(
            document=document,
            claims=claims,
            coverage=coverage,
            comparison_groups=comparison_groups,
            claim_relations=claim_relations,
            bibliography=bibliography,
        ),
        str(plan.get("plan_hash") or content_hash(plan)),
        audit_rubric_hash(rubric_path),
        content_hash(list(_disclosures(search_audit, corpus_snapshot))),
        canonical_json(audit_coverage_ledger(document, claims)),
    )
    actual = (
        audit.get("report_document_hash"),
        audit.get("report_artifact_hash"),
        audit.get("report_plan_hash"),
        audit.get("rubric_hash"),
        audit.get("search_limitations_hash"),
        canonical_json(audit.get("coverage_ledger", {})),
    )
    if actual != expected or not audit.get("coverage_complete"):
        raise ReportVerificationError("audit does not exhaustively cover the verified report inputs")
    block_ids = {str(item["block_id"]) for item in document["blocks"]}
    claim_ids = {str(item["claim_id"]) for item in claims}
    paper_ids = {
        str(item["paper_id"]) for item in corpus_snapshot.get("papers", ())
    }
    finding_ids: set[str] = set()
    for finding in audit.get("findings", ()):
        finding_id = str(finding.get("finding_id") or "")
        if not finding_id or finding_id in finding_ids:
            raise ReportVerificationError("audit contains a missing or duplicate finding ID")
        finding_ids.add(finding_id)
        if (
            not {str(value) for value in finding.get("block_ids", ())}.issubset(block_ids)
            or not {str(value) for value in finding.get("claim_ids", ())}.issubset(claim_ids)
            or not {str(value) for value in finding.get("paper_ids", ())}.issubset(paper_ids)
        ):
            raise ReportVerificationError("audit finding references an unknown report unit")


def verify_report(
    *,
    plan: Mapping[str, Any],
    document: Mapping[str, Any],
    claims: Sequence[Mapping[str, Any]],
    coverage: CoverageLedger | Mapping[str, Any],
    bibliography: Mapping[str, Mapping[str, Any]],
    comparison_groups: Mapping[str, Mapping[str, Any]] = {},
    search_audit: Mapping[str, Any] = {},
    corpus_snapshot: Mapping[str, Any] | None = None,
    previous: Mapping[str, Any] | None = None,
    claim_relations: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Validate all hard release gates and return the deterministic checklist."""
    try:
        validate(document, "report-document.schema.json")
        for claim in claims:
            validate(claim, "claim-evidence.schema.json")
    except SchemaValidationError as error:
        raise ReportVerificationError(str(error)) from error
    validate_claim_relations(previous, claims, claim_relations)
    sections = _sections(plan)
    section_ids = {section_id for section_id, _ in sections}
    claim_by_id = _claim_map(claims)
    _validate_report_language(plan, claims)
    try:
        require_exact_comparison_groups(claims, comparison_groups)
    except EvidenceValidationError as error:
        raise ReportVerificationError(str(error)) from error
    for claim in claims:
        if claim.get("comparison_group_id") != claim["claim_key"]["comparison_group_id"]:
            raise ReportVerificationError("claim comparison group bindings disagree")
        try:
            expected_claim_id = stable_claim_id(
                claim["claim_key"],
                report_run_id=str(document["report_run_id"]),
                mapping_status=str(claim["mapping_status"]),
            )
        except (EvidenceValidationError, TypeError, ValueError) as error:
            raise ReportVerificationError(str(error)) from error
        if claim["claim_id"] != expected_claim_id:
            raise ReportVerificationError("claim_id does not match its stable claim key")
    if str(document["report_run_id"]) != str(plan.get("report_run_id", document["report_run_id"])):
        raise ReportVerificationError("ReportDocument report_run_id does not match the report plan")

    blocks = tuple(document["blocks"])
    block_ids = [str(block["block_id"]) for block in blocks]
    if len(set(block_ids)) != len(block_ids):
        raise ReportVerificationError("ReportDocument contains duplicate block_id")
    claims_to_blocks: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    cited_papers: set[str] = set()
    table_blocks: list[str] = []
    for block in blocks:
        if block["block_kind"] not in SUBSTANTIVE_KINDS or block["section_id"] not in section_ids:
            raise ReportVerificationError("ReportDocument block is outside the frozen ReportPlan")
        if (
            block["block_kind"] in {"prose", "list_item"}
            and CJK_TEXT.search(str(block["text"])) is None
        ):
            raise ReportVerificationError(
                "zh-CN prose/list_item block must contain CJK text: "
                + str(block["block_id"])
            )
        claim_ids = tuple(str(item) for item in block["claim_ids"])
        citations = tuple(str(item) for item in block["citation_paper_ids"])
        if len(set(claim_ids)) != len(claim_ids) or len(set(citations)) != len(citations):
            raise ReportVerificationError("ReportDocument block has duplicate claim or citation bindings")
        if not claim_ids and not is_local_references_block(block):
            raise ReportVerificationError(
                "claim-free ReportDocument block is not the deterministic references note"
            )
        markers = set(PAPER_MARKER.findall(str(block["text"])))
        if markers != set(citations):
            raise ReportVerificationError("block citation sidecar does not exactly match its Markdown markers")
        for claim_id in claim_ids:
            if claim_id not in claim_by_id:
                raise ReportVerificationError(f"block binds an unknown claim: {claim_id}")
            claims_to_blocks[claim_id].append(block)
        allowed_citations = set().union(*(_paper_ids(claim_by_id[claim_id]) for claim_id in claim_ids))
        if not set(citations).issubset(allowed_citations):
            raise ReportVerificationError("block cites a paper not bound to its claim evidence")
        if NUMBER.search(str(block["text"])) and not any(
            _numeric_evidence(claim_by_id[claim_id]) for claim_id in claim_ids
        ):
            raise ReportVerificationError("numeric report text has no bound numeric evidence")
        if block["block_kind"] == "table_cell":
            table_blocks.append(str(block["block_id"]))
            for claim_id in claim_ids:
                paper_refs = [
                    ref
                    for field in ("supporting_evidence", "contradicting_evidence")
                    for ref in claim_by_id[claim_id][field]
                    if ref["kind"] == "paper_evidence"
                ]
                if not paper_refs or any(not ref.get("locator") for ref in paper_refs):
                    raise ReportVerificationError("table cells require paper-backed provenance")
        cited_papers.update(citations)
    missing_claims = sorted(set(claim_by_id) - set(claims_to_blocks))
    if missing_claims:
        raise ReportVerificationError(f"claims are absent from ReportDocument: {missing_claims}")
    for claim_id, claim_blocks in claims_to_blocks.items():
        if (
            not any(
                block["block_kind"] in {"prose", "list_item"}
                for block in claim_blocks
            )
            and not any(CJK_TEXT.search(str(block["text"])) for block in claim_blocks)
        ):
            raise ReportVerificationError(
                "zh-CN claim without prose/list_item requires CJK text in a table/caption block: "
                + claim_id
            )
        evidence_papers = _paper_ids(claim_by_id[claim_id])
        citation_coverage = set().union(*(set(block["citation_paper_ids"]) for block in claim_blocks))
        if not evidence_papers.issubset(citation_coverage):
            raise ReportVerificationError(f"claim {claim_id} has uncited paper evidence")
    if set(bibliography) != cited_papers:
        raise ReportVerificationError("bibliography must contain every and only cited canonical paper")
    for paper_id in sorted(cited_papers):
        _bibliography_entry(paper_id, bibliography[paper_id])

    coverage_value = _coverage_dict(coverage)
    if not coverage_value.get("complete") or coverage_value.get("missing_paper_ids") or coverage_value.get("uncovered_claim_ids"):
        raise ReportVerificationError("paper/claim coverage is incomplete")
    corpus_papers = {
        str(item["paper_id"]): item for item in (corpus_snapshot or {}).get("papers", ())
    }
    if corpus_snapshot is not None:
        covered_papers = {str(item["paper_id"]): item for item in coverage_value.get("papers", ())}
        memberships = {
            str(value["paper_id"]): value
            for value in plan.get("paper_memberships", ())
        }
        if len(covered_papers) != len(coverage_value.get("papers", ())):
            raise ReportVerificationError("coverage ledger contains duplicate paper IDs")
        if set(covered_papers) != set(corpus_papers):
            raise ReportVerificationError("coverage ledger does not exactly match the frozen corpus")
        for paper_id, item in covered_papers.items():
            membership = memberships.get(paper_id)
            if memberships and membership is None:
                raise ReportVerificationError(
                    f"coverage paper is absent from the approved plan: {paper_id}"
                )
            if item.get("disposition") == "background_only" and not str(item.get("reason") or "").strip():
                raise ReportVerificationError(f"background-only paper lacks a reason: {paper_id}")
            claim_ids = sorted(
                claim_id for claim_id, claim in claim_by_id.items()
                if paper_id in _paper_ids(claim)
            )
            recorded = sorted(str(value) for value in item.get("evidence_claim_ids", ()))
            if recorded != claim_ids:
                raise ReportVerificationError(
                    f"coverage evidence claims do not match the report claims for {paper_id}"
                )
            if item.get("disposition") == "evidence" and not claim_ids:
                raise ReportVerificationError(f"evidence disposition has no evidence claim: {paper_id}")
            if membership is not None and (
                item.get("disposition") != membership["coverage_disposition"]
                or item.get("reason") != membership["coverage_reason"]
            ):
                raise ReportVerificationError(
                    f"coverage disposition differs from the approved plan: {paper_id}"
                )
            if membership is not None and membership["coverage_disposition"] != "evidence" and claim_ids:
                raise ReportVerificationError(
                    f"non-evidence coverage paper appears in report claims: {paper_id}"
                )
        if memberships and set(memberships) != set(covered_papers):
            raise ReportVerificationError(
                "approved paper memberships do not exactly match coverage"
            )
    for claim in claims:
        support = tuple(claim["supporting_evidence"])
        contradict = tuple(claim["contradicting_evidence"])
        try:
            for reference in (*support, *contradict):
                validate_evidence_reference_shape(reference)
        except EvidenceValidationError as error:
            raise ReportVerificationError(str(error)) from error
        signatures = [content_hash(ref) for ref in (*support, *contradict)]
        if len(set(signatures)) != len(signatures):
            raise ReportVerificationError("claim repeats the same evidence reference")
        if not support and not contradict:
            raise ReportVerificationError("every report claim requires evidence")
        if claim["status"] == "supported" and not support:
            raise ReportVerificationError("supported claims require supporting evidence")
        if contradict and claim["status"] != "mixed":
            raise ReportVerificationError("contradicting evidence must remain a mixed claim")
        if claim["status"] == "mixed" and (not support or not contradict):
            raise ReportVerificationError("mixed claims require both sides of the conflict")
        paper_levels = [
            str(ref["evidence_level"])
            for ref in (*support, *contradict)
            if ref["kind"] == "paper_evidence"
        ]
        claim_level = str(claim["evidence_level"])
        if paper_levels:
            if (
                claim_level not in EVIDENCE_LEVEL_RANK
                or any(level not in EVIDENCE_LEVEL_RANK for level in paper_levels)
                or EVIDENCE_LEVEL_RANK[claim_level]
                > min(EVIDENCE_LEVEL_RANK[level] for level in paper_levels)
            ):
                raise ReportVerificationError(
                    "claim evidence level overstates its weakest paper evidence"
                )
        elif claim_level != "corpus_stat":
            raise ReportVerificationError("corpus-only claim must use corpus_stat evidence level")
        for expected_direction, references in (
            ("support", support),
            ("contradict", contradict),
        ):
            for ref in references:
                if ref["kind"] != "paper_evidence":
                    continue
                unit = ref.get("evidence_unit")
                if (
                    not str(ref.get("locator") or "").strip()
                    or not isinstance(unit, Mapping)
                    or unit.get("direction") != expected_direction
                ):
                    raise ReportVerificationError(
                        "paper evidence locator or direction is invalid"
                    )
        claim_text = "\n".join(
            str(block["text"]) for block in claims_to_blocks[str(claim["claim_id"])]
        )
        if claim["status"] == "mixed" and CONFLICT_DISCLOSURE.search(claim_text) is None:
            raise ReportVerificationError("mixed claim text erases its evidence conflict")
        group_id = claim.get("comparison_group_id")
        if group_id is not None and str(group_id) not in comparison_groups:
            raise ReportVerificationError(f"claim references an unknown comparison group: {group_id}")
        if claim["claim_type"] == "comparison" and group_id is None:
            if not claim["known_limitations"] or INCOMPARABLE_DISCLOSURE.search(claim_text) is None:
                raise ReportVerificationError("ungrouped comparison must be labeled as not directly comparable")
        for ref in (*support, *contradict):
            if ref["kind"] != "paper_evidence":
                continue
            paper_id = str(ref["paper_id"])
            if corpus_snapshot is not None:
                paper = corpus_papers.get(paper_id)
                if paper is None or ref["evidence_level"] not in INPUT_SCOPE_LEVELS.get(
                    str(paper["input_scope"]), frozenset()
                ):
                    raise ReportVerificationError("claim evidence level overstates its frozen paper input scope")
            try:
                assessment = comparison_assessment(ref["evidence_unit"])
            except (EvidenceValidationError, TypeError) as error:
                raise ReportVerificationError(str(error)) from error
            if group_id is not None and assessment.comparison_group_id != group_id:
                raise ReportVerificationError("claim evidence is outside its comparison group")
    limitations = _disclosures(search_audit, corpus_snapshot)
    document_text = "\n".join(str(block["text"]) for block in blocks)
    missing_limitations = [item for item in limitations if item not in document_text]
    if missing_limitations:
        raise ReportVerificationError(
            "search, extraction, or publication-status limitations are absent from the report body"
        )
    return {
        "renderer_version": RENDERER_VERSION,
        "plan_sections": [section_id for section_id, _ in sections],
        "block_count": len(blocks),
        "claim_count": len(claims),
        "cited_paper_ids": sorted(cited_papers),
        "table_blocks": table_blocks,
        "coverage_complete": True,
        "search_limitations_disclosed": list(limitations),
        "checks": {
            "no_unsupported_claims": True,
            "citation_coverage": True,
            "table_provenance": True,
            "search_limitations": True,
            "extraction_scope": True,
            "no_fabricated_statistics": True,
        },
    }


def render_markdown(
    *,
    plan: Mapping[str, Any],
    document: Mapping[str, Any],
    claims: Sequence[Mapping[str, Any]],
    bibliography: Mapping[str, Mapping[str, Any]],
    search_audit: Mapping[str, Any] = {},
    corpus_snapshot: Mapping[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Render the verified AST without letting model text control layout."""
    section_blocks: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for block in document["blocks"]:
        section_blocks[str(block["section_id"])].append(block)
    lines = [f"# {plan['objective']}", ""]
    sidecar_blocks: list[dict[str, Any]] = []
    for section_id, title in _sections(plan):
        lines.extend((f"## {title}", ""))
        table_open = False
        for block in section_blocks[section_id]:
            text = str(block["text"])
            if block["block_kind"] == "prose":
                lines.extend((text, ""))
            elif block["block_kind"] == "list_item":
                lines.extend((f"- {text}", ""))
            elif block["block_kind"] == "table_cell":
                if not table_open:
                    lines.extend(("| 内容 |", "| --- |"))
                    table_open = True
                escaped_text = text.replace("|", "\\|")
                lines.append(f"| {escaped_text} |")
            else:
                lines.extend((f"*{text}*", ""))
            sidecar_blocks.append({
                "block_id": block["block_id"],
                "section_id": section_id,
                "block_kind": block["block_kind"],
                "claim_ids": list(block["claim_ids"]),
                "citation_paper_ids": list(block["citation_paper_ids"]),
                "text_hash": content_hash(text),
            })
        if table_open:
            lines.append("")
    lines.extend(("## 参考文献", ""))
    cited = sorted({paper for block in document["blocks"] for paper in block["citation_paper_ids"]})
    lines.extend(_bibliography_entry(paper_id, bibliography[paper_id]) for paper_id in cited)
    lines.append("")
    sidecar = {
        "renderer_version": RENDERER_VERSION,
        "report_run_id": document["report_run_id"],
        "report_document_hash": content_hash(document),
        "blocks": sidecar_blocks,
        "claims": [
            {
                "claim_id": claim["claim_id"],
                "claim_hash": content_hash(claim),
                "evidence_ref_hashes": sorted(
                    content_hash(ref)
                    for field in ("supporting_evidence", "contradicting_evidence")
                    for ref in claim[field]
                ),
            }
            for claim in sorted(claims, key=lambda item: str(item["claim_id"]))
        ],
        "bibliography_paper_ids": cited,
        "search_limitations": list(_disclosures(search_audit, corpus_snapshot)),
    }
    return "\n".join(lines), sidecar


def report_diff(
    previous: Mapping[str, Any], current: Mapping[str, Any], *, claim_relations: Sequence[Mapping[str, Any]] = ()
) -> dict[str, Any]:
    """Stable ID based incremental report diff; never infer claim lineage from text."""
    before_claims = _claim_map(previous.get("claims", ()))
    after_claims = _claim_map(current.get("claims", ()))
    added = sorted(set(after_claims) - set(before_claims))
    retired = sorted(set(before_claims) - set(after_claims))
    changed = sorted(
        claim_id for claim_id in set(before_claims) & set(after_claims)
        if canonical_json(before_claims[claim_id]) != canonical_json(after_claims[claim_id])
    )
    before_papers = {str(item["paper_id"]): item for item in previous.get("corpus_snapshot", {}).get("papers", ())}
    after_papers = {str(item["paper_id"]): item for item in current.get("corpus_snapshot", {}).get("papers", ())}
    changed_papers = sorted(
        paper_id for paper_id in set(before_papers) & set(after_papers)
        if canonical_json(before_papers[paper_id]) != canonical_json(after_papers[paper_id])
    )
    relations = validate_claim_relations(previous, tuple(after_claims.values()), claim_relations)
    classified = {
        "split": [],
        "merged": [],
        "refined": [],
        "superseded": [],
        "retired": [],
        "same": [],
    }
    for relation in relations:
        classified[str(relation["relation_type"])].append(relation)
    mapped_previous = {str(item["previous_claim_id"]) for item in relations}
    mapped_current = {str(item["current_claim_id"]) for item in relations}
    unmapped = sorted((set(added) | set(retired)) - mapped_previous - mapped_current)
    evidence_diff = {
        claim_id: _claim_evidence_diff(before_claims[claim_id], after_claims[claim_id])
        for claim_id in changed
    }
    affected_sections = sorted({
        str(after_claims[claim_id]["report_section"])
        for claim_id in added + changed if claim_id in after_claims
    } | {
        str(before_claims[claim_id]["report_section"])
        for claim_id in retired if claim_id in before_claims
    })
    plan_before = previous.get("plan", {})
    plan_after = current.get("plan", {})
    all_sections = {
        str(item["id"])
        for item in (*plan_before.get("sections", ()), *plan_after.get("sections", ()))
    }
    publication_changes = []
    for paper_id in sorted(set(before_papers) & set(after_papers)):
        before_status = _publication_state(before_papers[paper_id])
        after_status = _publication_state(after_papers[paper_id])
        if before_status != after_status:
            publication_changes.append({"paper_id": paper_id, "before": before_status, "after": after_status})
    return {
        "query_or_scope_changed": content_hash(_scope_identity(plan_before)) != content_hash(_scope_identity(plan_after)),
        "added_paper_ids": sorted(set(after_papers) - set(before_papers)),
        "removed_paper_ids": sorted(set(before_papers) - set(after_papers)),
        "changed_paper_ids": changed_papers,
        "publication_or_retraction_changes": publication_changes,
        "added_claim_ids": added,
        "changed_claim_ids": changed,
        "retired_claim_ids": retired,
        "evidence_diff": evidence_diff,
        "relations": classified,
        "unmapped_claim_ids": unmapped,
        "affected_sections": affected_sections,
        "unchanged_sections": sorted(all_sections - set(affected_sections)),
    }


def validate_claim_relations(
    previous: Mapping[str, Any] | None,
    current_claims: Sequence[Mapping[str, Any]],
    relations: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Validate and canonicalize explicit cross-run claim lineage."""
    if previous is None:
        if len(relations) != 0:
            raise ReportVerificationError(
                "a non-incremental report must not contain claim relations"
            )
        return ()
    before = _claim_map(previous.get("claims", ()))
    after = _claim_map(current_claims)
    if len(relations) > len(before) + len(after):
        raise ReportVerificationError("claim relation cardinality exceeds its endpoints")
    required = {
        "previous_claim_id",
        "current_claim_id",
        "relation_type",
        "reason",
        "evidence_diff",
    }
    allowed_types = {"same", "refined", "split", "merged", "superseded", "retired"}
    canonical: list[dict[str, Any]] = []
    pairs: set[tuple[str, str]] = set()
    previous_types: dict[str, str] = {}
    current_types: dict[str, str] = {}
    previous_degree: dict[tuple[str, str], set[str]] = defaultdict(set)
    current_degree: dict[tuple[str, str], set[str]] = defaultdict(set)
    for relation in relations:
        if (
            not isinstance(relation, Mapping)
            or len(relation) != len(required)
            or any(field not in relation for field in required)
        ):
            raise ReportVerificationError("claim relation has unexpected or missing fields")
        previous_id = relation["previous_claim_id"]
        current_id = relation["current_claim_id"]
        relation_type = relation["relation_type"]
        reason = relation["reason"]
        if (
            not isinstance(previous_id, str)
            or previous_id not in before
            or not isinstance(current_id, str)
            or current_id not in after
            or not isinstance(relation_type, str)
            or relation_type not in allowed_types
            or not isinstance(reason, str)
            or reason != reason.strip()
            or not reason
            or len(reason) > 2_048
            or len(reason.encode("utf-8")) > 2_048
            or after[current_id].get("mapping_status") == "unmapped_new"
        ):
            raise ReportVerificationError(
                "claim relation lacks valid typed endpoints, type, or reason"
            )
        pair = (previous_id, current_id)
        if pair in pairs:
            raise ReportVerificationError("claim relation repeats one endpoint pair")
        pairs.add(pair)
        if relation_type == "same" and previous_id != current_id:
            raise ReportVerificationError("same claim relation must preserve the stable claim ID")
        if relation_type != "same" and previous_id == current_id:
            raise ReportVerificationError(
                "changed claim relation must not reuse the same stable claim ID"
            )
        if previous_id in previous_types and previous_types[previous_id] != relation_type:
            raise ReportVerificationError("one previous claim belongs to conflicting relation types")
        if current_id in current_types and current_types[current_id] != relation_type:
            raise ReportVerificationError("one current claim belongs to conflicting relation types")
        previous_types[previous_id] = str(relation_type)
        current_types[current_id] = str(relation_type)
        previous_degree[(str(relation_type), previous_id)].add(current_id)
        current_degree[(str(relation_type), current_id)].add(previous_id)
        expected_diff = _claim_evidence_diff(before[previous_id], after[current_id])
        supplied_diff = relation["evidence_diff"]
        if not isinstance(supplied_diff, Mapping) or len(supplied_diff) != len(expected_diff):
            raise ReportVerificationError("claim relation evidence diff is not exact")
        for field, expected_values in expected_diff.items():
            actual_values = supplied_diff.get(field)
            if (
                not isinstance(actual_values, list)
                or len(actual_values) != len(expected_values)
                or any(
                    left != right
                    for left, right in zip(actual_values, expected_values, strict=True)
                )
            ):
                raise ReportVerificationError("claim relation evidence diff is not exact")
        canonical.append({
            "previous_claim_id": previous_id,
            "current_claim_id": current_id,
            "relation_type": relation_type,
            "reason": reason,
            "evidence_diff": expected_diff,
        })
    for (relation_type, _), targets in previous_degree.items():
        if relation_type == "split" and len(targets) < 2:
            raise ReportVerificationError("split relation requires one-to-many cardinality")
        if relation_type != "split" and len(targets) != 1:
            raise ReportVerificationError("claim relation violates previous endpoint cardinality")
    for (relation_type, _), sources in current_degree.items():
        if relation_type == "merged" and len(sources) < 2:
            raise ReportVerificationError("merged relation requires many-to-one cardinality")
        if relation_type != "merged" and len(sources) != 1:
            raise ReportVerificationError("claim relation violates current endpoint cardinality")
    return tuple(sorted(
        canonical,
        key=lambda item: (
            str(item["previous_claim_id"]),
            str(item["current_claim_id"]),
            str(item["relation_type"]),
        ),
    ))


def _scope_identity(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: plan.get(key)
        for key in ("query_plan_hash", "primary_question", "subquestions", "scope")
    }


def _publication_state(paper: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: paper.get(key)
        for key in ("publication_status", "publication_version", "retraction_status", "is_retracted")
    }


def _claim_evidence_diff(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, list[str]]:
    def identities(claim: Mapping[str, Any], field: str) -> set[str]:
        return {content_hash(item) for item in claim[field]}

    before_support = identities(before, "supporting_evidence")
    after_support = identities(after, "supporting_evidence")
    before_contradict = identities(before, "contradicting_evidence")
    after_contradict = identities(after, "contradicting_evidence")
    return {
        "added_support": sorted(after_support - before_support),
        "removed_support": sorted(before_support - after_support),
        "added_contradiction": sorted(after_contradict - before_contradict),
        "removed_contradiction": sorted(before_contradict - after_contradict),
    }


def render_report_diff(diff: Mapping[str, Any]) -> str:
    lines = [
        "# 增量报告差异",
        "",
        f"- 检索问题或范围变化：{'是' if diff.get('query_or_scope_changed') else '否'}",
        "",
    ]
    for key in (
        "added_paper_ids", "removed_paper_ids", "changed_paper_ids", "added_claim_ids",
        "changed_claim_ids", "retired_claim_ids", "unmapped_claim_ids", "affected_sections",
        "unchanged_sections",
    ):
        lines.append(f"## {key}")
        values = diff.get(key, ())
        lines.extend(f"- {value}" for value in values) if values else lines.append("- 无")
        lines.append("")
    lines.extend(("## publication_or_retraction_changes", ""))
    publication = diff.get("publication_or_retraction_changes", ())
    lines.extend(
        f"- {item['paper_id']}: {item['before']} -> {item['after']}" for item in publication
    ) if publication else lines.append("- 无")
    lines.extend(("", "## evidence_diff", ""))
    evidence = diff.get("evidence_diff", {})
    lines.extend(f"- {claim_id}: {value}" for claim_id, value in sorted(evidence.items())) if evidence else lines.append("- 无")
    lines.extend(("", "## claim_relations", ""))
    relation_lines = [
        f"- {relation_type}: {item['previous_claim_id']} -> {item['current_claim_id']} ({item['reason']})"
        for relation_type, values in diff.get("relations", {}).items()
        for item in values
    ]
    lines.extend(relation_lines or ["- 无"])
    lines.append("")
    return "\n".join(lines)


class ReportArtifactStore:
    """Write immutable report bundles and atomically advance ``reports/latest.md``."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def directory(self, report_run_id: str) -> Path:
        component = str(report_run_id)
        if (
            not component
            or component in {".", ".."}
            or "/" in component
            or "\\" in component
            or "\x00" in component
            or Path(component).name != component
        ):
            raise ReportArtifactError("report_run_id is not a safe path component")
        return self.root / "reports" / component

    @property
    def latest_path(self) -> Path:
        return self.root / "reports" / "latest.md"

    def write(
        self,
        *,
        plan: Mapping[str, Any],
        search_audit: Mapping[str, Any],
        corpus_snapshot: Mapping[str, Any],
        claims: Sequence[Mapping[str, Any]],
        comparison_groups: Mapping[str, Mapping[str, Any]],
        claim_relations: Sequence[Mapping[str, Any]],
        document: Mapping[str, Any],
        coverage: CoverageLedger | Mapping[str, Any],
        bibliography: Mapping[str, Mapping[str, Any]],
        audit: Mapping[str, Any],
        previous: Mapping[str, Any] | None = None,
        rubric_path: str | Path | None = None,
        advance_latest: bool = True,
    ) -> Path:
        contents, markdown = self._bundle_contents(
            plan=plan,
            search_audit=search_audit,
            corpus_snapshot=corpus_snapshot,
            claims=claims,
            comparison_groups=comparison_groups,
            claim_relations=claim_relations,
            document=document,
            coverage=coverage,
            bibliography=bibliography,
            audit=audit,
            previous=previous,
            rubric_path=rubric_path,
        )
        report_run_id = str(document["report_run_id"])
        target = self.directory(report_run_id)
        if target.exists():
            raise ReportArtifactError(f"immutable report run already exists: {report_run_id}")
        temporary = target.with_name(f".{report_run_id}.tmp")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            if temporary.exists():
                raise ReportArtifactError(f"unfinished report bundle exists: {temporary.name}")
            temporary.mkdir()
            for name, text in contents.items():
                (temporary / name).write_text(text, encoding="utf-8")
            os.replace(temporary, target)
            if advance_latest:
                self._atomic_write(self.latest_path, markdown)
        except OSError as error:
            raise ReportArtifactError("immutable report bundle could not be written atomically") from error
        return target

    def reconcile(
        self,
        *,
        plan: Mapping[str, Any],
        search_audit: Mapping[str, Any],
        corpus_snapshot: Mapping[str, Any],
        claims: Sequence[Mapping[str, Any]],
        comparison_groups: Mapping[str, Mapping[str, Any]],
        claim_relations: Sequence[Mapping[str, Any]],
        document: Mapping[str, Any],
        coverage: CoverageLedger | Mapping[str, Any],
        bibliography: Mapping[str, Mapping[str, Any]],
        audit: Mapping[str, Any],
        previous: Mapping[str, Any] | None = None,
        rubric_path: str | Path | None = None,
        advance_latest: bool = True,
    ) -> Path:
        """Verify a crash-left bundle and optionally restore ``latest.md``."""
        contents, markdown = self._bundle_contents(
            plan=plan,
            search_audit=search_audit,
            corpus_snapshot=corpus_snapshot,
            claims=claims,
            comparison_groups=comparison_groups,
            claim_relations=claim_relations,
            document=document,
            coverage=coverage,
            bibliography=bibliography,
            audit=audit,
            previous=previous,
            rubric_path=rubric_path,
        )
        target = self.directory(str(document["report_run_id"]))
        try:
            items = tuple(target.iterdir())
        except OSError as error:
            raise ReportArtifactError("existing immutable report bundle is unavailable") from error
        if (
            target.is_symlink()
            or any(item.is_symlink() or not item.is_file() for item in items)
            or {item.name for item in items} != set(contents)
        ):
            raise ReportArtifactError("existing immutable report bundle has an unexpected file set")
        for name, expected in contents.items():
            try:
                actual = (target / name).read_text(encoding="utf-8")
            except (OSError, UnicodeError) as error:
                raise ReportArtifactError(
                    f"existing immutable report artifact is unreadable: {name}"
                ) from error
            if actual != expected:
                raise ReportArtifactError(
                    f"existing immutable report artifact conflicts with final state: {name}"
                )
        if advance_latest:
            try:
                self._atomic_write(self.latest_path, markdown)
            except OSError as error:
                raise ReportArtifactError(
                    "reports/latest.md could not be restored atomically"
                ) from error
        return target

    @staticmethod
    def _bundle_contents(
        *,
        plan: Mapping[str, Any],
        search_audit: Mapping[str, Any],
        corpus_snapshot: Mapping[str, Any],
        claims: Sequence[Mapping[str, Any]],
        comparison_groups: Mapping[str, Mapping[str, Any]],
        claim_relations: Sequence[Mapping[str, Any]],
        document: Mapping[str, Any],
        coverage: CoverageLedger | Mapping[str, Any],
        bibliography: Mapping[str, Mapping[str, Any]],
        audit: Mapping[str, Any],
        previous: Mapping[str, Any] | None,
        rubric_path: str | Path | None = None,
    ) -> tuple[dict[str, str], str]:
        blockers = search_publication_blockers(search_audit)
        if blockers:
            raise ReportVerificationError(
                "search audit is not publication-ready: " + "; ".join(blockers)
            )
        try:
            canonical_comparison_groups = require_exact_comparison_groups(
                claims, comparison_groups
            )
        except EvidenceValidationError as error:
            raise ReportVerificationError(str(error)) from error
        canonical_claim_relations = validate_claim_relations(
            previous, claims, claim_relations
        )
        checklist = verify_report(
            plan=plan, document=document, claims=claims, coverage=coverage,
            bibliography=bibliography, comparison_groups=comparison_groups,
            search_audit=search_audit, corpus_snapshot=corpus_snapshot,
            previous=previous, claim_relations=canonical_claim_relations,
        )
        try:
            validate(audit, "report-audit.schema.json")
        except SchemaValidationError as error:
            raise ReportVerificationError(str(error)) from error
        _validate_audit_binding(
            audit,
            plan=plan,
            document=document,
            claims=claims,
            coverage=coverage,
            comparison_groups=canonical_comparison_groups,
            claim_relations=canonical_claim_relations,
            bibliography=bibliography,
            search_audit=search_audit,
            corpus_snapshot=corpus_snapshot,
            rubric_path=rubric_path,
        )
        severe = [item for item in audit.get("findings", ()) if item.get("severity") in {"blocker", "major"}]
        if severe:
            raise ReportVerificationError("report audit contains blocker or major findings")
        markdown, sidecar = render_markdown(
            plan=plan, document=document, claims=claims, bibliography=bibliography,
            search_audit=search_audit, corpus_snapshot=corpus_snapshot,
        )
        diff = None
        rendered_diff = None
        if previous is not None:
            diff = report_diff(
                previous,
                {"plan": plan, "claims": claims, "corpus_snapshot": corpus_snapshot},
                claim_relations=canonical_claim_relations,
            )
            rendered_diff = render_report_diff(diff)
        report_run_id = str(document["report_run_id"])
        ordered_claims = tuple(sorted(claims, key=lambda item: str(item["claim_id"])))
        files: dict[str, Any] = {
            "REPORT_PLAN.json": plan,
            "SEARCH_AUDIT.json": search_audit,
            "CORPUS_SNAPSHOT.json": corpus_snapshot,
            "COMPARISON_GROUPS.json": canonical_comparison_groups,
            "CLAIM_RELATIONS.json": list(canonical_claim_relations),
            "REPORT_DOCUMENT.json": document,
            "COVERAGE.json": _coverage_dict(coverage),
            "RESOURCE_TABLES.json": _resource_tables_document(plan, corpus_snapshot),
            "BIBLIOGRAPHY.json": bibliography,
            "REPORT_SIDECAR.json": sidecar,
            "AUDIT.json": audit,
            "VERIFICATION.json": checklist,
        }
        if diff is not None:
            files["REPORT_DIFF.json"] = diff
        text_artifacts = {"REPORT.md": markdown}
        if rendered_diff is not None:
            text_artifacts["REPORT_DIFF.md"] = rendered_diff
        manifest = {name: content_hash(value) for name, value in files.items()}
        manifest.update({
            "CLAIMS_EVIDENCE.jsonl": content_hash(list(ordered_claims)),
            **{name: content_hash(value) for name, value in text_artifacts.items()},
        })
        contents = {name: _json(value) for name, value in files.items()}
        contents["CLAIMS_EVIDENCE.jsonl"] = "".join(
            _json(claim) for claim in ordered_claims
        )
        contents.update(text_artifacts)
        contents["MANIFEST.json"] = _json({
            "report_run_id": report_run_id,
            "artifacts": manifest,
        })
        return contents, markdown

    @staticmethod
    def _atomic_write(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)


def _resource_tables_document(
    plan: Mapping[str, Any], corpus_snapshot: Mapping[str, Any]
) -> dict[str, Any]:
    table_ids = tuple(str(item) for item in plan.get("artifacts", {}).get("resource_tables", ()))
    papers = {
        str(item["paper_id"]): item for item in corpus_snapshot.get("papers", ())
    }
    assigned: dict[str, list[dict[str, Any]]] = {table_id: [] for table_id in table_ids}
    fields = (
        "paper_id",
        "origin",
        "publication_date",
        "publication_year",
        "venue_id",
        "venue_name",
        "publication_status",
        "study_setting",
        "input_scope",
        "evidence_level",
    )
    for membership in plan.get("paper_memberships", ()):
        if membership.get("coverage_disposition") != "resource_or_background_table":
            continue
        paper_id = str(membership["paper_id"])
        paper = papers.get(paper_id)
        if paper is None:
            raise ReportVerificationError(
                f"resource table paper is absent from the frozen corpus: {paper_id}"
            )
        row = {field: paper.get(field) for field in fields}
        for table_id in membership["resource_table_ids"]:
            if table_id not in assigned:
                raise ReportVerificationError(
                    f"coverage references an unknown resource table: {table_id}"
                )
            assigned[table_id].append(row)
    return {
        "tables": [
            {
                "table_id": table_id,
                "rows": sorted(assigned[table_id], key=lambda item: str(item["paper_id"])),
            }
            for table_id in table_ids
        ]
    }
