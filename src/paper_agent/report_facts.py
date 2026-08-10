"""Immutable SQLite materialization for deterministically verified reports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import sqlite3
from typing import Any

from .canonical import canonical_json, content_hash


class ReportFactError(RuntimeError):
    """Verified report facts are missing, malformed, or conflict with SQLite."""


_FACT_SET_COLUMNS = (
    "report_run_id",
    "report_document_hash",
    "deterministic_verification_hash",
    "facts_hash",
    "claim_count",
    "evidence_count",
    "comparison_group_count",
    "claim_relation_count",
    "sealed",
)
_COMPARISON_COLUMNS = (
    "comparison_group_id",
    "comparison_key_hash",
    "comparison_key_json",
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
    "conditions_json",
)
_REPORT_COMPARISON_COLUMNS = (
    "report_run_id",
    "comparison_group_id",
    "comparison_key_hash",
)
_CLAIM_COLUMNS = (
    "report_run_id",
    "claim_id",
    "claim_hash",
    "claim_json",
    "claim_key_hash",
    "claim_key_json",
    "subject_id",
    "predicate_id",
    "object_or_scope_id",
    "qualifier_context_hash",
    "research_question_id",
    "report_section",
    "claim_text",
    "claim_type",
    "evidence_level",
    "comparison_group_id",
    "confidence",
    "known_limitations_json",
    "status",
    "mapping_status",
)
_EVIDENCE_COLUMNS = (
    "report_run_id",
    "claim_id",
    "direction",
    "ordinal",
    "evidence_ref_hash",
    "evidence_ref_json",
    "evidence_kind",
    "evidence_level",
    "paper_id",
    "analysis_run_id",
    "search_plan_id",
    "source_run_id",
    "query_id",
    "locator",
    "evidence_unit_json",
    "statistic",
    "calculation",
)
_RELATION_COLUMNS = (
    "current_report_run_id",
    "previous_report_run_id",
    "previous_claim_id",
    "current_claim_id",
    "relation_type",
    "reason",
    "evidence_diff_json",
    "relation_hash",
    "relation_json",
)
_REQUIRED_VERIFICATION_CHECKS = frozenset({
    "no_unsupported_claims",
    "citation_coverage",
    "table_provenance",
    "search_limitations",
    "extraction_scope",
    "no_fabricated_statistics",
})


@dataclass(frozen=True, slots=True)
class _CompiledReportFacts:
    fact_set: tuple[object, ...]
    comparison_groups: tuple[tuple[object, ...], ...]
    report_comparison_groups: tuple[tuple[object, ...], ...]
    claims: tuple[tuple[object, ...], ...]
    evidence: tuple[tuple[object, ...], ...]
    relations: tuple[tuple[object, ...], ...]


def materialize_verified_report_facts(
    connection: sqlite3.Connection,
    *,
    report_run_id: str,
    bundle: Mapping[str, Any],
    deterministic_verification: Mapping[str, Any],
    previous_report_run_id: str | None = None,
) -> None:
    """Insert one sealed fact set, or prove an identical set already exists.

    The audit coordinator must call this while holding its short SQLite write
    transaction.  A fact-set header is inserted unsealed, all normalized rows
    are written and compared, and the header is sealed before commit.  Sealed
    runs accept only byte-identical resume checks.
    """
    if not connection.in_transaction:
        raise ReportFactError(
            "verified report facts require the coordinator write transaction"
        )
    compiled = _compile(
        report_run_id,
        bundle,
        deterministic_verification,
        previous_report_run_id,
    )
    existing = _fact_set_row(connection, report_run_id)
    if existing is not None:
        if tuple(existing) == (*compiled.fact_set[:-1], 1):
            _require_compiled(connection, compiled)
            return
        if tuple(existing) != compiled.fact_set:
            raise ReportFactError(
                "verified report fact-set key is bound to different values"
            )
    try:
        if existing is None:
            _insert(
                connection,
                "report_fact_sets",
                _FACT_SET_COLUMNS,
                (compiled.fact_set,),
            )
        _insert(
            connection,
            "comparison_groups",
            _COMPARISON_COLUMNS,
            compiled.comparison_groups,
        )
        _require_global_comparison_groups(connection, compiled.comparison_groups)
        _insert(
            connection,
            "report_comparison_groups",
            _REPORT_COMPARISON_COLUMNS,
            compiled.report_comparison_groups,
        )
        _insert(connection, "report_claims", _CLAIM_COLUMNS, compiled.claims)
        _insert(connection, "claim_evidence", _EVIDENCE_COLUMNS, compiled.evidence)
        _insert(connection, "claim_relations", _RELATION_COLUMNS, compiled.relations)
        _require_run_rows(connection, compiled, sealed=0)
        sealed = connection.execute(
            """UPDATE report_fact_sets SET sealed = 1
               WHERE report_run_id = ? AND sealed = 0""",
            (report_run_id,),
        )
        if sealed.rowcount != 1:
            raise ReportFactError("verified report fact set could not be sealed")
        _require_compiled(connection, compiled)
    except sqlite3.IntegrityError as error:
        raise ReportFactError(
            "verified report fact key is missing its run binding or has different values"
        ) from error


def require_verified_report_facts(
    connection: sqlite3.Connection,
    *,
    report_run_id: str,
    bundle: Mapping[str, Any],
    deterministic_verification: Mapping[str, Any],
    previous_report_run_id: str | None = None,
) -> None:
    """Require the complete sealed SQLite projection for a verified bundle."""
    compiled = _compile(
        report_run_id,
        bundle,
        deterministic_verification,
        previous_report_run_id,
    )
    _require_compiled(connection, compiled)


def require_verified_report_claims(
    connection: sqlite3.Connection,
    *,
    report_run_id: str,
    claims: Sequence[Mapping[str, Any]],
) -> None:
    """Bind incremental lineage input to every sealed claim in a prior run."""
    if (
        not report_run_id
        or not isinstance(claims, Sequence)
        or isinstance(claims, (str, bytes))
    ):
        raise ReportFactError("previous report claims require a sealed run binding")
    try:
        expected = tuple(sorted(
            (
                str(claim["claim_id"]),
                content_hash(claim),
                _json(claim),
            )
            for claim in (_mapping(value, "previous report claim") for value in claims)
        ))
    except KeyError as error:
        raise ReportFactError("previous report claim is missing its claim_id") from error
    header = connection.execute(
        """SELECT fs.sealed, fs.claim_count,
                  rr.status AS report_status, rar.status AS audit_status
           FROM report_fact_sets fs
           JOIN report_runs rr ON rr.report_run_id = fs.report_run_id
           JOIN report_audit_runs rar ON rar.report_run_id = fs.report_run_id
           WHERE fs.report_run_id = ?""",
        (report_run_id,),
    ).fetchone()
    if header is None or tuple(header) != (1, len(expected), "complete", "complete"):
        raise ReportFactError(
            "previous report facts are not sealed and complete; resume that run first"
        )
    actual = tuple(
        tuple(row)
        for row in connection.execute(
            """SELECT claim_id, claim_hash, claim_json
               FROM report_claims WHERE report_run_id = ? ORDER BY claim_id""",
            (report_run_id,),
        ).fetchall()
    )
    if actual != expected:
        raise ReportFactError(
            "previous report claims differ from their sealed SQLite facts"
        )


def _compile(
    report_run_id: str,
    bundle: Mapping[str, Any],
    verification: Mapping[str, Any],
    previous_report_run_id: str | None,
) -> _CompiledReportFacts:
    document = bundle.get("document")
    claims_value = bundle.get("claims")
    groups_value = bundle.get("comparison_groups")
    relations_value = bundle.get("claim_relations")
    checks = verification.get("checks")
    if (
        not report_run_id
        or not isinstance(document, Mapping)
        or str(document.get("report_run_id")) != report_run_id
        or verification.get("report_document_hash") != content_hash(document)
        or verification.get("coverage_complete") is not True
        or not isinstance(checks, Mapping)
        or not _REQUIRED_VERIFICATION_CHECKS.issubset(checks)
        or any(value is not True for value in checks.values())
        or not isinstance(claims_value, Sequence)
        or isinstance(claims_value, (str, bytes))
        or not isinstance(groups_value, Mapping)
        or not isinstance(relations_value, Sequence)
        or isinstance(relations_value, (str, bytes))
    ):
        raise ReportFactError(
            "report facts require one successful deterministic verification result"
        )
    if verification.get("claim_count") != len(claims_value):
        raise ReportFactError(
            "deterministic verification claim count differs from report facts"
        )

    claims = tuple(sorted(
        (_mapping(value, "claim") for value in claims_value),
        key=lambda value: str(value["claim_id"]),
    ))
    groups = tuple(
        (str(group_id), _mapping(value, "comparison group"))
        for group_id, value in sorted(groups_value.items(), key=lambda item: str(item[0]))
    )
    relations = tuple(sorted(
        (_mapping(value, "claim relation") for value in relations_value),
        key=lambda value: (
            str(value["previous_claim_id"]),
            str(value["current_claim_id"]),
            str(value["relation_type"]),
        ),
    ))
    if relations and (
        not previous_report_run_id or previous_report_run_id == report_run_id
    ):
        raise ReportFactError(
            "claim relations require a distinct previous report run binding"
        )

    comparison_rows: list[tuple[object, ...]] = []
    report_comparison_rows: list[tuple[object, ...]] = []
    for group_id, key in groups:
        key_hash = content_hash(key)
        comparison_rows.append((
            group_id,
            key_hash,
            _json(key),
            *_comparison_values(key),
            _json(key["conditions"]),
        ))
        report_comparison_rows.append((report_run_id, group_id, key_hash))

    claim_rows: list[tuple[object, ...]] = []
    evidence_rows: list[tuple[object, ...]] = []
    for claim in claims:
        claim_key = _mapping(claim.get("claim_key"), "claim key")
        claim_id = str(claim["claim_id"])
        comparison_group_id = claim.get("comparison_group_id")
        claim_rows.append((
            report_run_id,
            claim_id,
            content_hash(claim),
            _json(claim),
            content_hash(claim_key),
            _json(claim_key),
            str(claim_key["subject_id"]),
            str(claim_key["predicate_id"]),
            str(claim_key["object_or_scope_id"]),
            str(claim_key["qualifier_context_hash"]),
            str(claim["research_question_id"]),
            str(claim["report_section"]),
            str(claim["claim_text"]),
            str(claim["claim_type"]),
            str(claim["evidence_level"]),
            str(comparison_group_id) if comparison_group_id is not None else None,
            str(claim["confidence"]),
            _json(claim["known_limitations"]),
            str(claim["status"]),
            str(claim["mapping_status"]),
        ))
        for field, direction in (
            ("supporting_evidence", "support"),
            ("contradicting_evidence", "contradict"),
        ):
            references = claim[field]
            if not isinstance(references, Sequence) or isinstance(references, (str, bytes)):
                raise ReportFactError("claim evidence must be an ordered sequence")
            for ordinal, raw_reference in enumerate(references):
                reference = _mapping(raw_reference, "claim evidence")
                unit = reference.get("evidence_unit")
                evidence_rows.append((
                    report_run_id,
                    claim_id,
                    direction,
                    ordinal,
                    content_hash(reference),
                    _json(reference),
                    str(reference["kind"]),
                    str(reference["evidence_level"]),
                    _optional_text(reference.get("paper_id")),
                    _optional_text(reference.get("analysis_run_id")),
                    _optional_text(reference.get("search_plan_id")),
                    _optional_text(reference.get("source_run_id")),
                    _optional_text(reference.get("query_id")),
                    _optional_text(reference.get("locator")),
                    _json(unit) if unit is not None else None,
                    _optional_text(reference.get("statistic")),
                    _optional_text(reference.get("calculation")),
                ))

    relation_rows = tuple((
        report_run_id,
        str(previous_report_run_id),
        str(relation["previous_claim_id"]),
        str(relation["current_claim_id"]),
        str(relation["relation_type"]),
        str(relation["reason"]),
        _json(relation["evidence_diff"]),
        content_hash(relation),
        _json(relation),
    ) for relation in relations)
    facts_document = {
        "report_run_id": report_run_id,
        "previous_report_run_id": previous_report_run_id if relations else None,
        "claims": list(claims),
        "comparison_groups": {group_id: key for group_id, key in groups},
        "claim_relations": list(relations),
    }
    fact_set = (
        report_run_id,
        content_hash(document),
        content_hash(verification),
        content_hash(facts_document),
        len(claim_rows),
        len(evidence_rows),
        len(comparison_rows),
        len(relation_rows),
        0,
    )
    return _CompiledReportFacts(
        fact_set,
        tuple(comparison_rows),
        tuple(report_comparison_rows),
        tuple(claim_rows),
        tuple(sorted(evidence_rows, key=lambda row: (str(row[1]), str(row[2]), int(row[3])))),
        tuple(relation_rows),
    )


def _comparison_values(key: Mapping[str, Any]) -> tuple[str, ...]:
    fields = (
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
    )
    try:
        values = tuple(key[field] for field in fields)
    except KeyError as error:
        raise ReportFactError("comparison group is missing a normalized field") from error
    if any(not isinstance(value, str) or not value for value in values):
        raise ReportFactError("comparison group normalized fields must be non-empty strings")
    conditions = key.get("conditions")
    if not isinstance(conditions, Sequence) or isinstance(conditions, (str, bytes)):
        raise ReportFactError("comparison group conditions must be an ordered sequence")
    return tuple(str(value) for value in values)


def _require_compiled(
    connection: sqlite3.Connection, compiled: _CompiledReportFacts
) -> None:
    _require_global_comparison_groups(connection, compiled.comparison_groups)
    _require_run_rows(connection, compiled, sealed=1)


def _require_run_rows(
    connection: sqlite3.Connection,
    compiled: _CompiledReportFacts,
    *,
    sealed: int,
) -> None:
    report_run_id = str(compiled.fact_set[0])
    fact_set = _fact_set_row(connection, report_run_id)
    if fact_set is None or tuple(fact_set) != (*compiled.fact_set[:-1], sealed):
        raise ReportFactError("verified report fact-set binding has drifted")
    _require_rows(
        connection,
        "report_comparison_groups",
        _REPORT_COMPARISON_COLUMNS,
        compiled.report_comparison_groups,
        "report_run_id = ?",
        (report_run_id,),
        "comparison_group_id",
    )
    _require_rows(
        connection,
        "report_claims",
        _CLAIM_COLUMNS,
        compiled.claims,
        "report_run_id = ?",
        (report_run_id,),
        "claim_id",
    )
    _require_rows(
        connection,
        "claim_evidence",
        _EVIDENCE_COLUMNS,
        compiled.evidence,
        "report_run_id = ?",
        (report_run_id,),
        "claim_id, direction, ordinal",
    )
    _require_rows(
        connection,
        "claim_relations",
        _RELATION_COLUMNS,
        compiled.relations,
        "current_report_run_id = ?",
        (report_run_id,),
        "previous_report_run_id, previous_claim_id, current_claim_id",
    )


def _require_global_comparison_groups(
    connection: sqlite3.Connection,
    expected: tuple[tuple[object, ...], ...],
) -> None:
    columns = ", ".join(_COMPARISON_COLUMNS)
    for row in expected:
        actual = connection.execute(
            f"SELECT {columns} FROM comparison_groups WHERE comparison_group_id = ?",
            (row[0],),
        ).fetchone()
        if actual is None or tuple(actual) != row:
            raise ReportFactError(
                "comparison-group key is bound to different normalized values"
            )


def _fact_set_row(
    connection: sqlite3.Connection, report_run_id: str
) -> sqlite3.Row | tuple[object, ...] | None:
    return connection.execute(
        f"SELECT {', '.join(_FACT_SET_COLUMNS)} FROM report_fact_sets "
        "WHERE report_run_id = ?",
        (report_run_id,),
    ).fetchone()


def _insert(
    connection: sqlite3.Connection,
    table: str,
    columns: tuple[str, ...],
    rows: tuple[tuple[object, ...], ...],
) -> None:
    if not rows:
        return
    placeholders = ", ".join("?" for _ in columns)
    connection.executemany(
        f"INSERT OR IGNORE INTO {table}({', '.join(columns)}) VALUES ({placeholders})",
        rows,
    )


def _require_rows(
    connection: sqlite3.Connection,
    table: str,
    columns: tuple[str, ...],
    expected: tuple[tuple[object, ...], ...],
    where: str,
    parameters: tuple[object, ...],
    order_by: str,
) -> None:
    actual = tuple(
        tuple(row)
        for row in connection.execute(
            f"SELECT {', '.join(columns)} FROM {table} "
            f"WHERE {where} ORDER BY {order_by}",
            parameters,
        ).fetchall()
    )
    if actual != expected:
        raise ReportFactError(f"verified {table} rows have drifted")


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReportFactError(f"{label} must be an object")
    return value


def _optional_text(value: object) -> str | None:
    return None if value is None else str(value)


def _json(value: object) -> str:
    return canonical_json(value).decode("utf-8")
