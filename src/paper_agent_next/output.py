"""Deterministic, atomic publication of the Stage 1 contract artifacts."""

import json
import os
from pathlib import Path
import tempfile
from typing import Sequence

from .errors import ContractError, InputError, PublicationError
from .models import IssueRecord, PaperRecord, RunRecord

__all__ = ["publish_artifacts"]


_ARTIFACT_NAMES = ("papers.jsonl", "issues.jsonl", "run.json")
_JSON_KWARGS = {"ensure_ascii": False, "separators": (",", ":")}


def _json_line(record: PaperRecord | IssueRecord | RunRecord) -> str:
    return json.dumps(record.to_dict(), **_JSON_KWARGS) + "\n"


def _jsonl_payload(records: Sequence[PaperRecord | IssueRecord]) -> str:
    return "".join(_json_line(record) for record in records)


def _validate_inputs(
    papers: Sequence[PaperRecord],
    issues: Sequence[IssueRecord],
    run: RunRecord,
) -> tuple[tuple[PaperRecord, ...], tuple[IssueRecord, ...]]:
    if not isinstance(run, RunRecord):
        raise ContractError("run: must be a RunRecord")

    try:
        paper_records = tuple(papers)
        issue_records = tuple(issues)
    except TypeError as error:
        raise ContractError("papers and issues: must be sequences of contract records") from error

    if len(paper_records) != run.counts.complete_papers:
        raise ContractError(
            "papers: length must equal run.counts.complete_papers"
        )
    if len(issue_records) != run.counts.issue_records:
        raise ContractError("issues: length must equal run.counts.issue_records")

    for index, paper in enumerate(paper_records):
        if not isinstance(paper, PaperRecord):
            raise ContractError(f"papers[{index}]: must be a PaperRecord")
        if (paper.venue_id, paper.year) != (run.venue_id, run.year):
            raise ContractError(f"papers[{index}]: venue_id and year must match run")

    for index, issue in enumerate(issue_records):
        if not isinstance(issue, IssueRecord):
            raise ContractError(f"issues[{index}]: must be an IssueRecord")
        if (issue.venue_id, issue.year) != (run.venue_id, run.year):
            raise ContractError(f"issues[{index}]: venue_id and year must match run")

    return paper_records, issue_records


def _sorted_papers(papers: Sequence[PaperRecord]) -> tuple[PaperRecord, ...]:
    return tuple(
        sorted(
            papers,
            key=lambda paper: (
                paper.venue_id,
                paper.year,
                paper.source_name,
                paper.source_id,
            ),
        )
    )


def _prepare_output_dir(output_dir: Path) -> None:
    try:
        if os.path.lexists(output_dir):
            if not output_dir.is_dir():
                raise InputError("output_dir: exists but is not a directory")
            existing = [
                name for name in _ARTIFACT_NAMES if os.path.lexists(output_dir / name)
            ]
            if existing:
                joined = ", ".join(existing)
                raise InputError(f"output_dir: formal artifact already exists: {joined}")
            return

        output_dir.mkdir(parents=True)
    except InputError:
        raise
    except OSError as error:
        raise PublicationError("failed to prepare output directory") from error


def _write_temp_file(
    output_dir: Path,
    payload: str,
    unpublished: list[Path],
) -> Path:
    file_descriptor = -1
    temp_path: Path | None = None
    try:
        file_descriptor, raw_path = tempfile.mkstemp(
            prefix=".paper-agent-",
            suffix=".tmp",
            dir=output_dir,
        )
        temp_path = Path(raw_path)
        unpublished.append(temp_path)
        stream = os.fdopen(file_descriptor, "w", encoding="utf-8", newline="")
        file_descriptor = -1
        with stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as error:
        if file_descriptor != -1:
            try:
                os.close(file_descriptor)
            except OSError:
                pass
        raise PublicationError("failed to stage an output artifact") from error

    assert temp_path is not None
    return temp_path


def _cleanup_unpublished(paths: Sequence[Path]) -> None:
    for path in paths:
        try:
            os.unlink(path)
        except OSError:
            pass


def publish_artifacts(
    output_dir: Path,
    papers: Sequence[PaperRecord],
    issues: Sequence[IssueRecord],
    run: RunRecord,
) -> None:
    """Publish the three Stage 1 artifacts as one ordered publication."""
    paper_records, issue_records = _validate_inputs(papers, issues, run)
    ordered_papers = _sorted_papers(paper_records)
    payloads = (
        _jsonl_payload(ordered_papers),
        _jsonl_payload(issue_records),
        _json_line(run),
    )

    output_dir = Path(output_dir)
    _prepare_output_dir(output_dir)

    unpublished: list[Path] = []
    try:
        temp_paths = [
            _write_temp_file(output_dir, payload, unpublished) for payload in payloads
        ]
        for temp_path, artifact_name in zip(temp_paths, _ARTIFACT_NAMES):
            try:
                os.replace(temp_path, output_dir / artifact_name)
            except OSError as error:
                raise PublicationError(
                    f"failed to publish {artifact_name}"
                ) from error
            unpublished.remove(temp_path)
    except PublicationError:
        _cleanup_unpublished(unpublished)
        raise
