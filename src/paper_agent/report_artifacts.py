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
from typing import Any

from .canonical import canonical_json, content_hash
from .reporting import CoverageLedger, EvidenceValidationError, INPUT_SCOPE_LEVELS, comparison_assessment
from .schema import SchemaValidationError, validate


RENDERER_VERSION = "report-markdown-v1"
SUBSTANTIVE_KINDS = frozenset({"prose", "list_item", "table_cell", "caption"})
PAPER_MARKER = re.compile(r"@([A-Za-z0-9._:-]+)")
NUMBER = re.compile(r"(?<![A-Za-z0-9._:-])\d+(?:\.\d+)?(?:\s*[%×x])?")
CONFLICT_DISCLOSURE = re.compile(r"冲突|矛盾|不一致|相反|分歧")
INCOMPARABLE_DISCLOSURE = re.compile(r"不可(?:直接)?比较|不具可比性")


class ReportArtifactError(ValueError):
    pass


class ReportVerificationError(ReportArtifactError):
    pass


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
            if isinstance(unit, Mapping) and isinstance(unit.get("value"), (int, float)):
                return True
    return False


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
    return tuple(dict.fromkeys(parts))


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
) -> dict[str, Any]:
    """Validate all hard release gates and return the deterministic checklist."""
    try:
        validate(document, "report-document.schema.json")
        for claim in claims:
            validate(claim, "claim-evidence.schema.json")
    except SchemaValidationError as error:
        raise ReportVerificationError(str(error)) from error
    sections = _sections(plan)
    section_ids = {section_id for section_id, _ in sections}
    claim_by_id = _claim_map(claims)
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
        claim_ids = tuple(str(item) for item in block["claim_ids"])
        citations = tuple(str(item) for item in block["citation_paper_ids"])
        if len(set(claim_ids)) != len(claim_ids) or len(set(citations)) != len(citations):
            raise ReportVerificationError("ReportDocument block has duplicate claim or citation bindings")
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
        if set(covered_papers) != set(corpus_papers):
            raise ReportVerificationError("coverage ledger does not exactly match the frozen corpus")
        for paper_id, item in covered_papers.items():
            if item.get("disposition") == "background_only" and not str(item.get("reason") or "").strip():
                raise ReportVerificationError(f"background-only paper lacks a reason: {paper_id}")
    for claim in claims:
        support = tuple(claim["supporting_evidence"])
        contradict = tuple(claim["contradicting_evidence"])
        if not support and not contradict:
            raise ReportVerificationError("every report claim requires evidence")
        if claim["status"] == "supported" and not support:
            raise ReportVerificationError("supported claims require supporting evidence")
        if contradict and claim["status"] != "mixed":
            raise ReportVerificationError("contradicting evidence must remain a mixed claim")
        if claim["status"] == "mixed" and (not support or not contradict):
            raise ReportVerificationError("mixed claims require both sides of the conflict")
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
        raise ReportVerificationError("search or extraction limitations are absent from the report body")
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
    relations = tuple(deepcopy(dict(item)) for item in claim_relations)
    classified = {"split": [], "merged": [], "refined": [], "superseded": [], "same": []}
    for relation in relations:
        relation_type = str(relation.get("relation_type", ""))
        previous_id = str(relation.get("previous_claim_id") or "")
        current_id = str(relation.get("current_claim_id") or "")
        if (
            relation_type not in classified
            or previous_id not in before_claims
            or current_id not in after_claims
            or not str(relation.get("reason") or "").strip()
            or not isinstance(relation.get("evidence_diff"), Mapping)
        ):
            raise ReportVerificationError("claim relation lacks its typed IDs, reason, or evidence diff")
        classified[relation_type].append(relation)
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
        return self.root / "reports" / report_run_id

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
    ) -> Path:
        checklist = verify_report(
            plan=plan, document=document, claims=claims, coverage=coverage,
            bibliography=bibliography, comparison_groups=comparison_groups,
            search_audit=search_audit, corpus_snapshot=corpus_snapshot,
        )
        report_hash = content_hash(document)
        try:
            validate(audit, "report-audit.schema.json")
        except SchemaValidationError as error:
            raise ReportVerificationError(str(error)) from error
        if audit.get("report_document_hash") != report_hash or not audit.get("coverage_complete"):
            raise ReportVerificationError("audit does not cover the verified report document")
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
                claim_relations=claim_relations,
            )
            rendered_diff = render_report_diff(diff)
        report_run_id = str(document["report_run_id"])
        target = self.directory(report_run_id)
        if target.exists():
            raise ReportArtifactError(f"immutable report run already exists: {report_run_id}")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{report_run_id}.tmp")
        if temporary.exists():
            raise ReportArtifactError(f"unfinished report bundle exists: {temporary.name}")
        temporary.mkdir()
        ordered_claims = tuple(sorted(claims, key=lambda item: str(item["claim_id"])))
        files: dict[str, Any] = {
            "REPORT_PLAN.json": plan,
            "SEARCH_AUDIT.json": search_audit,
            "CORPUS_SNAPSHOT.json": corpus_snapshot,
            "COMPARISON_GROUPS.json": comparison_groups,
            "CLAIM_RELATIONS.json": list(claim_relations),
            "REPORT_DOCUMENT.json": document,
            "COVERAGE.json": _coverage_dict(coverage),
            "REPORT_SIDECAR.json": sidecar,
            "AUDIT.json": audit,
            "VERIFICATION.json": checklist,
        }
        if diff is not None:
            files["REPORT_DIFF.json"] = diff
        for name, value in files.items():
            (temporary / name).write_text(_json(value), encoding="utf-8")
        (temporary / "CLAIMS_EVIDENCE.jsonl").write_text(
            "".join(_json(claim) for claim in ordered_claims), encoding="utf-8"
        )
        (temporary / "REPORT.md").write_text(markdown, encoding="utf-8")
        text_artifacts = {"REPORT.md": markdown}
        if rendered_diff is not None:
            text_artifacts["REPORT_DIFF.md"] = rendered_diff
            (temporary / "REPORT_DIFF.md").write_text(rendered_diff, encoding="utf-8")
        manifest = {name: content_hash(value) for name, value in files.items()}
        manifest.update({
            "CLAIMS_EVIDENCE.jsonl": content_hash(list(ordered_claims)),
            **{name: content_hash(value) for name, value in text_artifacts.items()},
        })
        (temporary / "MANIFEST.json").write_text(
            _json({"report_run_id": report_run_id, "artifacts": manifest}), encoding="utf-8"
        )
        os.replace(temporary, target)
        self._atomic_write(self.latest_path, markdown)
        return target

    @staticmethod
    def _atomic_write(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
