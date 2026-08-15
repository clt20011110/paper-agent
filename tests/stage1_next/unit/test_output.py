"""Unit tests for the fixed Stage 1 artifact publication boundary."""

import json
import os
from pathlib import Path

import pytest

from paper_agent_next import output as output_module
from paper_agent_next.errors import ContractError, InputError, PublicationError
from paper_agent_next.models import (
    AccessStatus,
    FieldSources,
    IssueKind,
    IssueRecord,
    MissingField,
    PaperRecord,
    Pagination,
    RunCounts,
    RunRecord,
    RunStatus,
    SourceTotal,
    SourceTotalScope,
    VenueType,
)


def _paper(
    source_id: str = "paper-1",
    *,
    venue_id: str = "example-conf",
    year: int = 2024,
    title: str = "Café paper",
) -> PaperRecord:
    return PaperRecord(
        venue_id=venue_id,
        venue_name="Example Conference",
        venue_type=VenueType.CONFERENCE,
        year=year,
        source_name="proceedings",
        source_id=source_id,
        title=title,
        authors=("Zoë Example",),
        abstract="Résumé abstract.",
        doi=None,
        landing_url=f"https://example.org/{source_id}",
        pdf_url=f"https://example.org/{source_id}.pdf",
        access_status=AccessStatus.DIRECT_PDF,
        field_sources=FieldSources(
            "proceedings",
            "proceedings",
            "proceedings",
            None,
            "proceedings",
            "proceedings",
        ),
    )


def _issue(
    source_id: str = "issue-1",
    *,
    venue_id: str = "example-conf",
    year: int = 2024,
) -> IssueRecord:
    return IssueRecord(
        issue_kind=IssueKind.INCOMPLETE_PAPER,
        venue_id=venue_id,
        year=year,
        source_name="proceedings",
        source_id=source_id,
        source_locator=f"https://example.org/{source_id}",
        title="Café incomplete paper",
        authors=("Zoë Example",),
        abstract=None,
        doi="10.1234/example.1",
        landing_url=f"https://example.org/{source_id}",
        missing_fields=(MissingField.ABSTRACT,),
        reason_codes=("missing_abstract",),
        message="The abstract is missing.",
    )


def _counts(
    *,
    complete_papers: int = 0,
    incomplete_papers: int = 0,
    issue_records: int = 0,
) -> RunCounts:
    included_papers = complete_papers + incomplete_papers
    return RunCounts(
        raw_items=included_papers,
        included_papers=included_papers,
        complete_papers=complete_papers,
        incomplete_papers=incomplete_papers,
        excluded_non_papers=0,
        duplicate_occurrences=0,
        parse_rejects=0,
        issue_records=issue_records,
    )


def _complete_run(complete_papers: int = 0) -> RunRecord:
    return RunRecord(
        status=RunStatus.COMPLETE,
        venue_id="example-conf",
        venue_name="Example Conference",
        venue_type=VenueType.CONFERENCE,
        year=2024,
        source_name="proceedings",
        membership_complete=True,
        metadata_complete=True,
        complete=True,
        counts=_counts(complete_papers=complete_papers),
        pagination=Pagination(
            1,
            True,
            SourceTotal(complete_papers, SourceTotalScope.INCLUDED_PAPERS),
        ),
        warnings=(),
        errors=(),
    )


def _partial_run(
    *, complete_papers: int, incomplete_papers: int, issue_records: int
) -> RunRecord:
    included_papers = complete_papers + incomplete_papers
    return RunRecord(
        status=RunStatus.PARTIAL,
        venue_id="example-conf",
        venue_name="Example Conference",
        venue_type=VenueType.CONFERENCE,
        year=2024,
        source_name="proceedings",
        membership_complete=True,
        metadata_complete=False,
        complete=False,
        counts=_counts(
            complete_papers=complete_papers,
            incomplete_papers=incomplete_papers,
            issue_records=issue_records,
        ),
        pagination=Pagination(
            1,
            True,
            SourceTotal(included_papers, SourceTotalScope.INCLUDED_PAPERS),
        ),
        warnings=(),
        errors=(),
    )


def _empty_run(status: RunStatus) -> RunRecord:
    if status is RunStatus.COMPLETE:
        return _complete_run()
    if status is RunStatus.PARTIAL:
        return RunRecord(
            status=status,
            venue_id="example-conf",
            venue_name="Example Conference",
            venue_type=VenueType.CONFERENCE,
            year=2024,
            source_name="proceedings",
            membership_complete=False,
            metadata_complete=False,
            complete=False,
            counts=_counts(),
            pagination=None,
            warnings=(),
            errors=("membership interrupted",),
        )
    if status is RunStatus.FAILED:
        return RunRecord(
            status=status,
            venue_id="example-conf",
            venue_name="Example Conference",
            venue_type=VenueType.CONFERENCE,
            year=2024,
            source_name="proceedings",
            membership_complete=False,
            metadata_complete=False,
            complete=False,
            counts=_counts(),
            pagination=None,
            warnings=(),
            errors=("membership failed",),
        )
    return RunRecord(
        status=RunStatus.NOT_APPLICABLE,
        venue_id="example-conf",
        venue_name="Example Conference",
        venue_type=VenueType.CONFERENCE,
        year=2024,
        source_name=None,
        membership_complete=False,
        metadata_complete=False,
        complete=False,
        counts=_counts(),
        pagination=None,
        warnings=(),
        errors=(),
    )


def _compact(record: PaperRecord | IssueRecord | RunRecord) -> str:
    return json.dumps(record.to_dict(), ensure_ascii=False, separators=(",", ":")) + "\n"


def test_publish_writes_exact_unicode_payloads_and_deterministic_order(tmp_path: Path) -> None:
    first = _paper("z", title="Zèbre paper")
    second = _paper("a", title="Álgebra paper")
    issue = _issue("issue-z")
    run = _partial_run(complete_papers=2, incomplete_papers=1, issue_records=1)

    assert output_module.publish_artifacts(tmp_path / "artifacts", [first, second], [issue], run) is None

    output_dir = tmp_path / "artifacts"
    assert (output_dir / "papers.jsonl").read_text(encoding="utf-8") == _compact(second) + _compact(first)
    assert (output_dir / "issues.jsonl").read_text(encoding="utf-8") == _compact(issue)
    assert (output_dir / "run.json").read_text(encoding="utf-8") == _compact(run)
    for name in ("papers.jsonl", "issues.jsonl", "run.json"):
        data = (output_dir / name).read_bytes()
        assert not data.startswith(b"\xef\xbb\xbf")
        assert data.endswith(b"\n")
    assert "Álgebra" in (output_dir / "papers.jsonl").read_text(encoding="utf-8")
    assert "\\u" not in (output_dir / "papers.jsonl").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "status",
    [RunStatus.COMPLETE, RunStatus.PARTIAL, RunStatus.FAILED, RunStatus.NOT_APPLICABLE],
)
def test_all_run_statuses_publish_fixed_artifacts_and_empty_jsonl(
    tmp_path: Path, status: RunStatus
) -> None:
    output_dir = tmp_path / status.value

    output_module.publish_artifacts(output_dir, [], [], _empty_run(status))

    assert sorted(path.name for path in output_dir.iterdir()) == [
        "issues.jsonl",
        "papers.jsonl",
        "run.json",
    ]
    assert (output_dir / "papers.jsonl").read_bytes() == b""
    assert (output_dir / "issues.jsonl").read_bytes() == b""
    assert json.loads((output_dir / "run.json").read_text(encoding="utf-8"))["status"] == status.value


def test_contract_validation_happens_before_directory_creation(tmp_path: Path) -> None:
    count_dir = tmp_path / "count-mismatch"
    with pytest.raises(ContractError):
        output_module.publish_artifacts(count_dir, [_paper()], [], _complete_run())
    assert not count_dir.exists()

    venue_dir = tmp_path / "venue-mismatch"
    run = _partial_run(complete_papers=1, incomplete_papers=0, issue_records=1)
    with pytest.raises(ContractError):
        output_module.publish_artifacts(
            venue_dir,
            [_paper()],
            [_issue(venue_id="other-conf")],
            run,
        )
    assert not venue_dir.exists()


def test_prepare_output_dir_is_public_and_idempotent_for_clean_directories(tmp_path: Path) -> None:
    assert output_module.__all__ == ["prepare_output_dir", "publish_artifacts"]
    output_dir = tmp_path / "clean"

    output_module.prepare_output_dir(output_dir)
    output_module.prepare_output_dir(output_dir)

    assert output_dir.is_dir()
    assert list(output_dir.iterdir()) == []


@pytest.mark.parametrize("artifact_name", ["papers.jsonl", "issues.jsonl", "run.json"])
def test_prepare_output_dir_rejects_existing_formal_artifacts(
    tmp_path: Path, artifact_name: str
) -> None:
    output_dir = tmp_path / artifact_name
    output_dir.mkdir()
    (output_dir / artifact_name).write_bytes(b"keep")

    with pytest.raises(InputError):
        output_module.prepare_output_dir(output_dir)


def test_publish_artifacts_reuses_public_prepare_output_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "published"
    calls: list[Path] = []
    real_prepare = output_module.prepare_output_dir

    def recording_prepare(path: Path) -> None:
        calls.append(path)
        real_prepare(path)

    monkeypatch.setattr(output_module, "prepare_output_dir", recording_prepare)
    output_module.publish_artifacts(output_dir, [], [], _complete_run())

    assert calls == [output_dir]


def test_existing_artifact_and_non_directory_are_input_errors_without_overwrite(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    original = b"existing content\n"
    (output_dir / "papers.jsonl").write_bytes(original)
    (output_dir / "other.txt").write_bytes(b"keep me")

    with pytest.raises(InputError):
        output_module.publish_artifacts(output_dir, [], [], _complete_run())

    assert (output_dir / "papers.jsonl").read_bytes() == original
    assert (output_dir / "other.txt").read_bytes() == b"keep me"

    non_directory = tmp_path / "file"
    non_directory.write_bytes(b"do not touch")
    with pytest.raises(InputError):
        output_module.publish_artifacts(non_directory, [], [], _complete_run())
    assert non_directory.read_bytes() == b"do not touch"


def test_staging_is_fsynced_before_ordered_replacement_and_run_is_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "ordered"
    events: list[tuple[str, str | None, Path | None]] = []
    real_fsync = output_module.os.fsync
    real_replace = output_module.os.replace

    def recording_fsync(file_descriptor: int) -> None:
        events.append(("fsync", None, None))
        real_fsync(file_descriptor)

    def recording_replace(source: str | os.PathLike[str], destination: str | os.PathLike[str]) -> None:
        destination_path = Path(destination)
        events.append(("replace", destination_path.name, Path(source).resolve().parent))
        if destination_path.name == "run.json":
            assert not (output_dir / "run.json").exists()
            assert (output_dir / "papers.jsonl").exists()
            assert (output_dir / "issues.jsonl").exists()
        real_replace(source, destination)

    monkeypatch.setattr(output_module.os, "fsync", recording_fsync)
    monkeypatch.setattr(output_module.os, "replace", recording_replace)
    output_module.publish_artifacts(output_dir, [_paper()], [], _complete_run(1))

    assert [event[0] for event in events] == ["fsync", "fsync", "fsync", "replace", "replace", "replace"]
    assert [event[1] for event in events[3:]] == ["papers.jsonl", "issues.jsonl", "run.json"]
    assert all(event[2] == output_dir.resolve() for event in events[3:])


class _FailingStream:
    def __init__(self, stream, failure: str) -> None:
        self._stream = stream
        self._failure = failure

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return self._stream.__exit__(exc_type, exc_value, traceback)

    def write(self, payload: str) -> int:
        if self._failure == "write":
            raise OSError("simulated write failure")
        return self._stream.write(payload)

    def flush(self) -> None:
        if self._failure == "flush":
            raise OSError("simulated flush failure")
        self._stream.flush()

    def fileno(self) -> int:
        return self._stream.fileno()


@pytest.mark.parametrize("failure", ["write", "flush"])
def test_staging_write_errors_are_chained_and_cleaned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    output_dir = tmp_path / failure
    real_fdopen = output_module.os.fdopen

    def failing_fdopen(file_descriptor: int, *args, **kwargs):
        stream = real_fdopen(file_descriptor, *args, **kwargs)
        return _FailingStream(stream, failure)

    monkeypatch.setattr(output_module.os, "fdopen", failing_fdopen)
    with pytest.raises(PublicationError) as raised:
        output_module.publish_artifacts(output_dir, [_paper()], [], _complete_run(1))

    assert isinstance(raised.value.__cause__, OSError)
    assert not (output_dir / "run.json").exists()
    assert not list(output_dir.glob(".paper-agent-*.tmp"))


def test_replace_error_leaves_no_run_and_cleans_unpublished_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "replace-failure"
    real_replace = output_module.os.replace

    def failing_run_replace(source, destination):
        if Path(destination).name == "run.json":
            raise OSError("simulated replace failure")
        real_replace(source, destination)

    monkeypatch.setattr(output_module.os, "replace", failing_run_replace)
    with pytest.raises(PublicationError) as raised:
        output_module.publish_artifacts(output_dir, [_paper()], [], _complete_run(1))

    assert isinstance(raised.value.__cause__, OSError)
    assert (output_dir / "papers.jsonl").exists()
    assert (output_dir / "issues.jsonl").exists()
    assert not (output_dir / "run.json").exists()
    assert not list(output_dir.glob(".paper-agent-*.tmp"))
