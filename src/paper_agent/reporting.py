"""Deterministic Stage 4b planning and evidence validation.

The model writes synthesis text.  The coordinator owns membership, stable
identifiers, comparison eligibility, coverage, and the reduce budget so a
model cannot silently change the report corpus or invent evidence.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import groupby
import re
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

from .canonical import canonical_json, content_hash


CLAIM_NAMESPACE = uuid5(NAMESPACE_URL, "https://paper-agent.dev/namespaces/report-claim-v1")
COMPARISON_NAMESPACE = uuid5(NAMESPACE_URL, "https://paper-agent.dev/namespaces/comparison-group-v1")
CLAIM_KEY_FIELDS = (
    "subject_id",
    "predicate_id",
    "object_or_scope_id",
    "qualifier_context_hash",
    "comparison_group_id",
)
COMPARISON_FIELDS = (
    "task_id",
    "dataset_id",
    "dataset_version",
    "split_id",
    "metric_id",
    "metric_definition_hash",
    "unit",
    "optimization_direction",
    "protocol_id",
    "protocol_hash",
    "baseline_id",
    "baseline_version",
    "normalization_method",
    "normalizer_version",
    "source_value",
)
EVIDENCE_LEVEL_RANK = {
    "metadata_only": 0,
    "abstract_direct": 1,
    "full_text_inferred": 2,
    "full_text_direct": 3,
}
INPUT_SCOPE_LEVELS = {
    "metadata_only": frozenset({"metadata_only"}),
    "abstract_only": frozenset({"abstract_direct", "metadata_only"}),
    "full_pdf": frozenset(EVIDENCE_LEVEL_RANK),
}
PAPER_MARKER = re.compile(r"@([A-Za-z0-9._:-]+)")


class ReportPlanningError(ValueError):
    pass


class EvidenceValidationError(ValueError):
    pass


class BudgetExceeded(ReportPlanningError):
    pass


@dataclass(frozen=True, slots=True)
class AnalysisRecord:
    paper_id: str
    analysis_run_id: str
    analysis_hash: str
    input_scope: str
    input_tokens: int
    classifications: Mapping[str, tuple[str, ...]]
    evidence_units: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        if not self.paper_id or not self.analysis_run_id or not self.analysis_hash:
            raise ValueError("analysis identity is required")
        if self.input_scope not in INPUT_SCOPE_LEVELS:
            raise ValueError(f"unsupported analysis input scope: {self.input_scope}")
        if self.input_tokens < 1:
            raise ValueError("analysis input_tokens must be positive")


@dataclass(frozen=True, slots=True)
class SectionRule:
    section_id: str
    subquestion_ids: frozenset[str]
    allowed_evidence_levels: frozenset[str]


@dataclass(frozen=True, slots=True)
class SemanticChunk:
    node_id: str
    section_id: str
    classification_key: tuple[tuple[str, tuple[str, ...]], ...]
    paper_ids: tuple[str, ...]
    analysis_hashes: tuple[str, ...]
    input_tokens: int


@dataclass(frozen=True, slots=True)
class ReduceNode:
    node_id: str
    call_kind: str
    section_ids: tuple[str, ...]
    paper_ids: tuple[str, ...]
    dependency_ids: tuple[str, ...]
    planned_input_hash: str
    input_tokens: int


@dataclass(frozen=True, slots=True)
class BudgetEstimate:
    generation_calls: int
    audit_calls: int
    repair_calls: int
    worst_case_calls: int
    generation_input_tokens: int
    audit_input_tokens: int
    repair_input_tokens: int
    worst_case_input_tokens: int


@dataclass(frozen=True, slots=True)
class ReducePlan:
    chunks: tuple[SemanticChunk, ...]
    nodes: tuple[ReduceNode, ...]
    budget: BudgetEstimate

    def nodes_for(self, call_kind: str) -> tuple[ReduceNode, ...]:
        return tuple(node for node in self.nodes if node.call_kind == call_kind)


class ReportPlanner:
    """Compile frozen plan membership into a stable semantic reduce tree."""

    def __init__(
        self,
        plan: Mapping[str, Any],
        analyses: Sequence[AnalysisRecord],
        *,
        max_chunk_input_tokens: int,
        reduce_output_tokens: int,
        audit_input_tokens: int,
        repair_input_tokens: int,
    ) -> None:
        self.plan = plan
        self.analyses = {item.paper_id: item for item in analyses}
        if len(self.analyses) != len(analyses):
            raise ReportPlanningError("analysis records contain duplicate paper_ids")
        if min(max_chunk_input_tokens, reduce_output_tokens, audit_input_tokens, repair_input_tokens) < 1:
            raise ReportPlanningError("all token estimates must be positive")
        self.max_chunk_input_tokens = max_chunk_input_tokens
        self.reduce_output_tokens = reduce_output_tokens
        self.audit_input_tokens = audit_input_tokens
        self.repair_input_tokens = repair_input_tokens
        self.axes = tuple(str(axis) for axis in plan["classification_axes"])
        if not self.axes or len(set(self.axes)) != len(self.axes):
            raise ReportPlanningError("classification_axes must be non-empty and unique")
        self.sections = self._sections()
        self.memberships = self._memberships()

    def build(self) -> ReducePlan:
        chunks = self._chunks()
        nodes = self._tree(chunks)
        estimate = self._estimate(nodes)
        budget = self.plan["budget"]
        if estimate.worst_case_calls > int(budget["max_sol_calls"]):
            raise BudgetExceeded(
                f"reduce plan needs {estimate.worst_case_calls} Sol calls; budget allows {budget['max_sol_calls']}"
            )
        if estimate.worst_case_input_tokens > int(budget["max_input_tokens"]):
            raise BudgetExceeded(
                f"reduce plan needs {estimate.worst_case_input_tokens} input tokens; "
                f"budget allows {budget['max_input_tokens']}"
            )
        return ReducePlan(chunks, nodes, estimate)

    def _sections(self) -> tuple[SectionRule, ...]:
        rules = tuple(
            SectionRule(
                str(section["id"]),
                frozenset(str(item) for item in section["subquestion_ids"]),
                frozenset(str(item) for item in section["allowed_evidence_levels"]),
            )
            for section in self.plan["sections"]
        )
        if not rules or len({item.section_id for item in rules}) != len(rules):
            raise ReportPlanningError("plan sections must be non-empty and uniquely identified")
        return rules

    def _memberships(self) -> Mapping[str, tuple[str, ...]]:
        section_ids = {item.section_id for item in self.sections}
        memberships: dict[str, tuple[str, ...]] = {}
        for value in self.plan["paper_memberships"]:
            paper_id = str(value["paper_id"])
            assigned = tuple(str(item) for item in value["section_ids"])
            primary = str(value["primary_section_id"])
            if paper_id in memberships:
                raise ReportPlanningError(f"duplicate paper membership: {paper_id}")
            if not assigned or len(set(assigned)) != len(assigned) or primary not in assigned:
                raise ReportPlanningError(f"invalid section membership for {paper_id}")
            unknown = set(assigned) - section_ids
            if unknown:
                raise ReportPlanningError(f"unknown sections for {paper_id}: {sorted(unknown)}")
            memberships[paper_id] = assigned
        if set(memberships) != set(self.analyses):
            missing = sorted(set(self.analyses) - set(memberships))
            extra = sorted(set(memberships) - set(self.analyses))
            raise ReportPlanningError(f"plan/corpus membership mismatch: missing={missing}, extra={extra}")
        return memberships

    def _semantic_key(self, analysis: AnalysisRecord) -> tuple[tuple[str, tuple[str, ...]], ...]:
        return tuple(
            (axis, tuple(sorted(str(value) for value in analysis.classifications.get(axis, ()))))
            for axis in self.axes
        )

    def _chunks(self) -> tuple[SemanticChunk, ...]:
        chunks: list[SemanticChunk] = []
        for section in self.sections:
            assigned = [
                analysis
                for paper_id, analysis in self.analyses.items()
                if section.section_id in self.memberships[paper_id]
            ]
            assigned.sort(key=lambda item: (self._semantic_key(item), item.paper_id))
            if not assigned:
                raise ReportPlanningError(f"section has no assigned papers: {section.section_id}")
            section_index = 0
            for semantic_key, values in groupby(assigned, key=self._semantic_key):
                current: list[AnalysisRecord] = []
                current_tokens = 0
                for analysis in values:
                    if analysis.input_tokens > self.max_chunk_input_tokens:
                        raise BudgetExceeded(
                            f"paper {analysis.paper_id} needs {analysis.input_tokens} tokens; "
                            f"chunk limit is {self.max_chunk_input_tokens}; truncation is forbidden"
                        )
                    if current and current_tokens + analysis.input_tokens > self.max_chunk_input_tokens:
                        section_index += 1
                        chunks.append(self._chunk(section, semantic_key, section_index, current, current_tokens))
                        current = []
                        current_tokens = 0
                    current.append(analysis)
                    current_tokens += analysis.input_tokens
                if current:
                    section_index += 1
                    chunks.append(self._chunk(section, semantic_key, section_index, current, current_tokens))
        return tuple(chunks)

    @staticmethod
    def _chunk(
        section: SectionRule,
        semantic_key: tuple[tuple[str, tuple[str, ...]], ...],
        index: int,
        analyses: Sequence[AnalysisRecord],
        input_tokens: int,
    ) -> SemanticChunk:
        return SemanticChunk(
            node_id=f"section:{section.section_id}:{index:04d}",
            section_id=section.section_id,
            classification_key=semantic_key,
            paper_ids=tuple(item.paper_id for item in analyses),
            analysis_hashes=tuple(item.analysis_hash for item in analyses),
            input_tokens=input_tokens,
        )

    def _tree(self, chunks: Sequence[SemanticChunk]) -> tuple[ReduceNode, ...]:
        nodes: list[ReduceNode] = []
        roots: list[ReduceNode] = []
        for section in self.sections:
            leaves: list[ReduceNode] = []
            for chunk in (item for item in chunks if item.section_id == section.section_id):
                leaf = ReduceNode(
                    chunk.node_id,
                    "section_reduce",
                    (chunk.section_id,),
                    chunk.paper_ids,
                    (),
                    content_hash({
                        "section_id": chunk.section_id,
                        "classification_key": chunk.classification_key,
                        "analysis_hashes": chunk.analysis_hashes,
                    }),
                    chunk.input_tokens,
                )
                leaves.append(leaf)
                nodes.append(leaf)
            roots.append(self._reduce_layer(leaves, "section_reduce", nodes, section.section_id))
        cross_root = self._reduce_layer(roots, "cross_section_reduce", nodes, "cross")
        final = self._node("final_reduce", (cross_root,), "final", len(nodes) + 1)
        nodes.append(final)
        return tuple(nodes)

    def _reduce_layer(
        self,
        values: Sequence[ReduceNode],
        call_kind: str,
        nodes: list[ReduceNode],
        label: str,
    ) -> ReduceNode:
        layer = list(values)
        depth = 0
        while len(layer) > 1:
            next_layer: list[ReduceNode] = []
            for offset in range(0, len(layer), 2):
                pair = layer[offset : offset + 2]
                if len(pair) == 1:
                    next_layer.append(pair[0])
                    continue
                node = self._node(call_kind, pair, f"{label}:{depth}:{offset // 2}", len(nodes) + 1)
                nodes.append(node)
                next_layer.append(node)
            layer = next_layer
            depth += 1
        return layer[0]

    def _node(
        self, call_kind: str, dependencies: Sequence[ReduceNode], label: str, ordinal: int
    ) -> ReduceNode:
        section_ids = tuple(dict.fromkeys(section for node in dependencies for section in node.section_ids))
        paper_ids = tuple(sorted({paper for node in dependencies for paper in node.paper_ids}))
        dependency_ids = tuple(node.node_id for node in dependencies)
        return ReduceNode(
            f"{call_kind}:{label}:{ordinal:04d}",
            call_kind,
            section_ids,
            paper_ids,
            dependency_ids,
            content_hash({
                "call_kind": call_kind,
                "dependencies": [
                    {"node_id": node.node_id, "planned_input_hash": node.planned_input_hash}
                    for node in dependencies
                ],
            }),
            len(dependencies) * self.reduce_output_tokens,
        )

    def _estimate(self, nodes: Sequence[ReduceNode]) -> BudgetEstimate:
        budget = self.plan["budget"]
        retries = int(budget["max_retries"])
        audit_calls = int(budget["audit_calls"])
        repair_calls = int(budget["repair_calls"])
        if retries < 0 or audit_calls < 2 or repair_calls != 1:
            raise ReportPlanningError("budget must reserve non-negative retries, two audits, and one repair")
        generation_tokens = sum(node.input_tokens for node in nodes)
        audit_tokens = audit_calls * self.audit_input_tokens
        repair_tokens = repair_calls * self.repair_input_tokens
        base_calls = len(nodes) + audit_calls + repair_calls
        base_tokens = generation_tokens + audit_tokens + repair_tokens
        return BudgetEstimate(
            generation_calls=len(nodes),
            audit_calls=audit_calls,
            repair_calls=repair_calls,
            worst_case_calls=base_calls * (retries + 1),
            generation_input_tokens=generation_tokens,
            audit_input_tokens=audit_tokens,
            repair_input_tokens=repair_tokens,
            worst_case_input_tokens=base_tokens * (retries + 1),
        )


@dataclass(frozen=True, slots=True)
class ComparisonAssessment:
    eligibility: str
    comparison_group_id: str | None
    comparison_key: Mapping[str, Any] | None
    missing_fields: tuple[str, ...]


def comparison_assessment(unit: Mapping[str, Any]) -> ComparisonAssessment:
    absent = tuple(field for field in COMPARISON_FIELDS if unit.get(field) is None)
    declared_missing = tuple(sorted(set(str(item) for item in unit.get("missing_fields", ()))))
    eligibility = str(unit["comparison_eligibility"])
    if eligibility == "not_comparable":
        if not absent and not declared_missing:
            raise EvidenceValidationError("not_comparable evidence must state missing_fields")
        if not set(absent).issubset(declared_missing):
            raise EvidenceValidationError("missing_fields does not cover absent comparison fields")
        return ComparisonAssessment(eligibility, None, None, declared_missing)
    if eligibility != "comparable":
        raise EvidenceValidationError(f"unknown comparison eligibility: {eligibility}")
    if absent or declared_missing:
        raise EvidenceValidationError("comparable evidence requires every comparison field and no missing_fields")
    if not isinstance(unit["value"], (int, float)) or isinstance(unit["value"], bool):
        raise EvidenceValidationError("comparable evidence requires a numeric value")
    comparison_key = {
        field: unit[field]
        for field in COMPARISON_FIELDS
        if field != "source_value"
    }
    comparison_key["conditions"] = tuple(sorted(str(item) for item in unit.get("conditions", ())))
    group_id = str(uuid5(COMPARISON_NAMESPACE, content_hash(comparison_key)))
    return ComparisonAssessment(eligibility, group_id, comparison_key, ())


def require_parallel_comparison(units: Sequence[Mapping[str, Any]]) -> str:
    if len(units) < 2:
        raise EvidenceValidationError("a parallel comparison needs at least two evidence units")
    assessments = tuple(comparison_assessment(unit) for unit in units)
    if any(item.eligibility != "comparable" for item in assessments):
        raise EvidenceValidationError("not_comparable evidence cannot be ranked or aggregated")
    group_ids = {item.comparison_group_id for item in assessments}
    if len(group_ids) != 1:
        raise EvidenceValidationError("numeric evidence belongs to different comparison groups")
    return next(iter(group_ids))  # type: ignore[return-value]


def stable_claim_id(
    claim_key: Mapping[str, Any], *, report_run_id: str, mapping_status: str = "mapped"
) -> str:
    _validate_claim_key(claim_key)
    if mapping_status == "mapped":
        namespace: UUID = CLAIM_NAMESPACE
    elif mapping_status == "unmapped_new":
        namespace = uuid5(CLAIM_NAMESPACE, report_run_id)
    else:
        raise EvidenceValidationError(f"unknown mapping status: {mapping_status}")
    return str(uuid5(namespace, content_hash(dict(claim_key))))


def _validate_claim_key(claim_key: Mapping[str, Any]) -> None:
    if set(claim_key) != set(CLAIM_KEY_FIELDS):
        raise EvidenceValidationError("claim_key has the wrong fields")
    for field in CLAIM_KEY_FIELDS[:3]:
        if not isinstance(claim_key[field], str) or not claim_key[field]:
            raise EvidenceValidationError(f"claim_key.{field} is required")
    qualifier_hash = claim_key["qualifier_context_hash"]
    if not isinstance(qualifier_hash, str) or re.fullmatch(r"[a-f0-9]{64}", qualifier_hash) is None:
        raise EvidenceValidationError("claim_key.qualifier_context_hash must be a sha256")
    group_id = claim_key["comparison_group_id"]
    if group_id is not None:
        UUID(str(group_id))


@dataclass(frozen=True, slots=True)
class CorpusEvidenceAllowlist:
    search_plan_ids: frozenset[str] = frozenset()
    source_run_ids: frozenset[str] = frozenset()
    query_ids: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class ValidatedSection:
    section_id: str
    claims: tuple[Mapping[str, Any], ...]
    citation_paper_ids: tuple[str, ...]
    unresolved_conflicts: tuple[str, ...]
    document: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ValidatedCrossSection:
    section_ids: tuple[str, ...]
    claims: tuple[Mapping[str, Any], ...]
    citation_paper_ids: tuple[str, ...]
    unresolved_conflicts: tuple[str, ...]
    document: Mapping[str, Any]


class SynthesisValidator:
    def __init__(
        self,
        *,
        report_run_id: str,
        analyses: Sequence[AnalysisRecord],
        sections: Sequence[SectionRule],
        memberships: Mapping[str, Sequence[str]],
        corpus_evidence: CorpusEvidenceAllowlist = CorpusEvidenceAllowlist(),
    ) -> None:
        self.report_run_id = report_run_id
        self.analyses = {item.paper_id: item for item in analyses}
        self.sections = {item.section_id: item for item in sections}
        self.memberships = {paper: frozenset(section_ids) for paper, section_ids in memberships.items()}
        self.corpus_evidence = corpus_evidence

    def validate_section(self, document: Mapping[str, Any]) -> ValidatedSection:
        section_id = str(document["section_id"])
        if section_id not in self.sections:
            raise EvidenceValidationError(f"unknown section: {section_id}")
        allowed_papers = {
            paper_id for paper_id, sections in self.memberships.items() if section_id in sections
        }
        citations = tuple(str(item) for item in document["citation_paper_ids"])
        if len(set(citations)) != len(citations) or not set(citations).issubset(allowed_papers):
            raise EvidenceValidationError("section contains duplicate or non-allowlisted citations")
        claims = tuple(document["claims"])
        evidence_papers: set[str] = set()
        claim_ids: set[str] = set()
        mixed = False
        for claim in claims:
            if claim["claim_id"] in claim_ids:
                raise EvidenceValidationError(f"duplicate claim_id: {claim['claim_id']}")
            claim_ids.add(str(claim["claim_id"]))
            evidence_papers.update(self._validate_claim(claim, section_id, allowed_papers))
            mixed = mixed or claim["status"] == "mixed"
        if set(citations) != evidence_papers:
            raise EvidenceValidationError("section citation_paper_ids must exactly match paper evidence")
        markers = set(PAPER_MARKER.findall(str(document["draft"])))
        if not markers.issubset(set(citations)):
            raise EvidenceValidationError("section draft contains a non-allowlisted paper marker")
        conflicts = tuple(str(item) for item in document["unresolved_conflicts"])
        if mixed and not conflicts:
            raise EvidenceValidationError("mixed claims require an unresolved conflict disclosure")
        return ValidatedSection(section_id, claims, citations, conflicts, document)

    def validate_cross_section(
        self,
        document: Mapping[str, Any],
        sections: Sequence[ValidatedSection],
    ) -> ValidatedCrossSection:
        expected_sections = tuple(sorted(item.section_id for item in sections))
        actual_sections = tuple(str(item) for item in document["section_ids"])
        if actual_sections != expected_sections:
            raise EvidenceValidationError("cross-section output must cover the exact sorted section set")
        expected_claims: dict[str, Mapping[str, Any]] = {}
        for section in sections:
            for claim in section.claims:
                claim_id = str(claim["claim_id"])
                existing = expected_claims.get(claim_id)
                if existing is not None and canonical_json(existing) != canonical_json(claim):
                    raise EvidenceValidationError(f"conflicting records for stable claim {claim_id}")
                expected_claims[claim_id] = claim
        actual_claims = {str(claim["claim_id"]): claim for claim in document["claims"]}
        if len(actual_claims) != len(document["claims"]) or set(actual_claims) != set(expected_claims):
            raise EvidenceValidationError("cross-section output added or dropped claims")
        if any(canonical_json(actual_claims[key]) != canonical_json(value) for key, value in expected_claims.items()):
            raise EvidenceValidationError("cross-section output changed a validated claim")
        expected_citations = tuple(sorted({paper for section in sections for paper in section.citation_paper_ids}))
        actual_citations = tuple(str(item) for item in document["citation_paper_ids"])
        if actual_citations != expected_citations:
            raise EvidenceValidationError("cross-section output added or dropped citations")
        markers = set(PAPER_MARKER.findall(str(document["draft"])))
        if not markers.issubset(set(actual_citations)):
            raise EvidenceValidationError("cross-section draft contains a non-allowlisted paper marker")
        expected_conflicts = {value for section in sections for value in section.unresolved_conflicts}
        conflicts = tuple(str(item) for item in document["unresolved_conflicts"])
        if not expected_conflicts.issubset(conflicts):
            raise EvidenceValidationError("cross-section output erased an unresolved conflict")
        return ValidatedCrossSection(
            actual_sections,
            tuple(actual_claims[key] for key in sorted(actual_claims)),
            actual_citations,
            conflicts,
            document,
        )

    def _validate_claim(
        self, claim: Mapping[str, Any], section_id: str, allowed_papers: set[str]
    ) -> set[str]:
        if claim["report_section"] != section_id:
            raise EvidenceValidationError("claim report_section does not match its section")
        section = self.sections[section_id]
        if claim["research_question_id"] not in section.subquestion_ids:
            raise EvidenceValidationError("claim research_question_id is outside the section plan")
        mapping_status = str(claim.get("mapping_status", "mapped"))
        expected_id = stable_claim_id(
            claim["claim_key"], report_run_id=self.report_run_id, mapping_status=mapping_status
        )
        if claim["claim_id"] != expected_id:
            raise EvidenceValidationError("claim_id does not match its canonical claim_key")
        if claim.get("comparison_group_id") != claim["claim_key"]["comparison_group_id"]:
            raise EvidenceValidationError("claim comparison_group_id bindings disagree")

        support = tuple(claim["supporting_evidence"])
        contradict = tuple(claim["contradicting_evidence"])
        if not support and not contradict:
            raise EvidenceValidationError("every claim requires evidence")
        if claim["status"] == "supported" and not support:
            raise EvidenceValidationError("supported claims require supporting evidence")
        if claim["status"] == "mixed" and (not support or not contradict):
            raise EvidenceValidationError("mixed claims require supporting and contradicting evidence")

        paper_ids: set[str] = set()
        paper_units: list[Mapping[str, Any]] = []
        levels: list[str] = []
        signatures: set[str] = set()
        for direction, refs in (("support", support), ("contradict", contradict)):
            for ref in refs:
                signature = content_hash(ref)
                if signature in signatures:
                    raise EvidenceValidationError("claim repeats the same evidence reference")
                signatures.add(signature)
                level = str(ref["evidence_level"])
                if level not in section.allowed_evidence_levels:
                    raise EvidenceValidationError(f"evidence level {level} is not allowed in {section_id}")
                levels.append(level)
                if ref["kind"] == "paper_evidence":
                    paper_id, unit = self._validate_paper_ref(ref, direction, level, allowed_papers)
                    paper_ids.add(paper_id)
                    paper_units.append(unit)
                elif ref["kind"] == "corpus_evidence":
                    self._validate_corpus_ref(ref, level)
                else:
                    raise EvidenceValidationError(f"unknown evidence kind: {ref['kind']}")
        self._validate_claim_level(str(claim["evidence_level"]), levels)
        self._validate_claim_comparison(claim, paper_units)
        if claim["claim_type"] == "corpus_stat" and not any(
            ref["kind"] == "corpus_evidence" for ref in support + contradict
        ):
            raise EvidenceValidationError("corpus_stat claims require corpus evidence")
        return paper_ids

    def _validate_paper_ref(
        self,
        ref: Mapping[str, Any],
        direction: str,
        level: str,
        allowed_papers: set[str],
    ) -> tuple[str, Mapping[str, Any]]:
        paper_id = str(ref["paper_id"])
        if paper_id not in allowed_papers or paper_id not in self.analyses:
            raise EvidenceValidationError(f"paper evidence is outside the section allowlist: {paper_id}")
        analysis = self.analyses[paper_id]
        if ref["analysis_run_id"] != analysis.analysis_run_id:
            raise EvidenceValidationError("paper evidence uses the wrong analysis run")
        if level not in INPUT_SCOPE_LEVELS[analysis.input_scope]:
            raise EvidenceValidationError("paper evidence overstates its analysis input scope")
        if not ref.get("locator"):
            raise EvidenceValidationError("paper evidence requires a source locator")
        unit = ref["evidence_unit"]
        allowed_hashes = {content_hash(value) for value in analysis.evidence_units}
        if content_hash(unit) not in allowed_hashes:
            raise EvidenceValidationError("paper evidence unit is not in the bound analysis")
        if unit["direction"] != direction:
            raise EvidenceValidationError("evidence direction disagrees with its claim ledger column")
        return paper_id, unit

    def _validate_corpus_ref(self, ref: Mapping[str, Any], level: str) -> None:
        if level != "corpus_stat":
            raise EvidenceValidationError("corpus evidence must use corpus_stat evidence level")
        bindings = (
            ("search_plan_id", self.corpus_evidence.search_plan_ids),
            ("source_run_id", self.corpus_evidence.source_run_ids),
            ("query_id", self.corpus_evidence.query_ids),
        )
        for field, allowlist in bindings:
            if ref[field] not in allowlist:
                raise EvidenceValidationError(f"corpus evidence has a non-allowlisted {field}")
        if not ref.get("statistic") or not ref.get("calculation"):
            raise EvidenceValidationError("corpus evidence requires statistic and calculation")

    @staticmethod
    def _validate_claim_level(claim_level: str, evidence_levels: Sequence[str]) -> None:
        paper_levels = [level for level in evidence_levels if level != "corpus_stat"]
        if not paper_levels:
            if claim_level != "corpus_stat":
                raise EvidenceValidationError("corpus-only claims must use corpus_stat evidence level")
            return
        if claim_level == "corpus_stat" or claim_level not in EVIDENCE_LEVEL_RANK:
            raise EvidenceValidationError("paper-backed claim has an invalid evidence level")
        weakest = min(EVIDENCE_LEVEL_RANK[level] for level in paper_levels)
        if EVIDENCE_LEVEL_RANK[claim_level] > weakest:
            raise EvidenceValidationError("claim evidence level overstates its weakest paper evidence")

    @staticmethod
    def _validate_claim_comparison(
        claim: Mapping[str, Any], units: Sequence[Mapping[str, Any]]
    ) -> None:
        group_id = claim.get("comparison_group_id")
        assessments = tuple(comparison_assessment(unit) for unit in units)
        if group_id is not None:
            if not assessments or any(item.comparison_group_id != group_id for item in assessments):
                raise EvidenceValidationError("claim mixes evidence outside its comparison group")
        if claim["claim_type"] != "comparison":
            return
        comparable = tuple(item for item in assessments if item.eligibility == "comparable")
        if group_id is None:
            if comparable or not claim["known_limitations"]:
                raise EvidenceValidationError(
                    "an ungrouped comparison must disclose that its evidence is not directly comparable"
                )
        elif len(comparable) < 2:
            raise EvidenceValidationError("a grouped comparison needs at least two comparable evidence units")


@dataclass(frozen=True, slots=True)
class PaperCoverage:
    paper_id: str
    evidence_claim_ids: tuple[str, ...]
    consumed_node_ids: tuple[str, ...]
    disposition: str
    reason: str | None


@dataclass(frozen=True, slots=True)
class CoverageLedger:
    papers: tuple[PaperCoverage, ...]
    missing_paper_ids: tuple[str, ...]
    uncovered_claim_ids: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not self.missing_paper_ids and not self.uncovered_claim_ids

    def require_complete(self) -> None:
        if not self.complete:
            raise EvidenceValidationError(
                f"coverage incomplete: papers={list(self.missing_paper_ids)}, "
                f"claims={list(self.uncovered_claim_ids)}"
            )


def build_coverage_ledger(
    selected_paper_ids: Iterable[str],
    sections: Sequence[ValidatedSection],
    chunks: Sequence[SemanticChunk],
    *,
    resource_paper_ids: Iterable[str] = (),
    background_only: Mapping[str, str] = {},
) -> CoverageLedger:
    selected = frozenset(str(item) for item in selected_paper_ids)
    resources = frozenset(str(item) for item in resource_paper_ids)
    if not resources.issubset(selected) or not set(background_only).issubset(selected):
        raise EvidenceValidationError("coverage declarations contain a paper outside the corpus")
    if any(not reason.strip() for reason in background_only.values()):
        raise EvidenceValidationError("background_only coverage requires a reason")

    paper_claims: dict[str, set[str]] = defaultdict(set)
    claim_has_evidence: dict[str, bool] = {}
    for section in sections:
        for claim in section.claims:
            claim_id = str(claim["claim_id"])
            refs = tuple(claim["supporting_evidence"]) + tuple(claim["contradicting_evidence"])
            claim_has_evidence[claim_id] = claim_has_evidence.get(claim_id, False) or bool(refs)
            for ref in refs:
                if ref["kind"] == "paper_evidence":
                    paper_claims[str(ref["paper_id"])].add(claim_id)
    consumed: dict[str, set[str]] = defaultdict(set)
    for chunk in chunks:
        for paper_id in chunk.paper_ids:
            consumed[paper_id].add(chunk.node_id)

    papers: list[PaperCoverage] = []
    missing: list[str] = []
    for paper_id in sorted(selected):
        claims = tuple(sorted(paper_claims[paper_id]))
        if claims:
            disposition = "evidence"
            reason = None
        elif paper_id in resources:
            disposition = "resource_or_background_table"
            reason = None
        elif paper_id in background_only:
            disposition = "background_only"
            reason = background_only[paper_id]
        else:
            disposition = "uncovered"
            reason = None
            missing.append(paper_id)
        papers.append(PaperCoverage(
            paper_id,
            claims,
            tuple(sorted(consumed[paper_id])),
            disposition,
            reason,
        ))
    uncovered_claims = tuple(sorted(claim_id for claim_id, covered in claim_has_evidence.items() if not covered))
    return CoverageLedger(tuple(papers), tuple(missing), uncovered_claims)
