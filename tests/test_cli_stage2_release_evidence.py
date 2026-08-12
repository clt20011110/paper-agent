from __future__ import annotations

import json
from pathlib import Path

import pytest

from paper_agent import cli


def _arguments(output: Path) -> list[str]:
    result = [
        "stage2-release", "build-evidence",
        "--candidate", "candidate.json",
        "--gold-manifest", "gold.json",
        "--structured-manifest", "structured-manifest.json",
        "--structured-records", "structured-records.json",
        "--structured-papers", "structured-papers.json",
        "--rationale-manifest", "rationale-manifest.json",
        "--rationale-worklist", "rationale-worklist.json",
        "--rationale-records", "rationale-records.json",
        "--parity-manifest", "parity-manifest.json",
        "--parity-workload", "parity-workload.json",
        "--parity-selection-receipt", "parity-receipt.json",
        "--parity-scores", "parity-scores.json",
        "--parity-oracle-model-lock", "oracle-lock.json",
        "--parity-candidate-model-lock", "candidate-lock.json",
        "--parity-oracle-calibrator", "oracle-calibrator.json",
        "--parity-candidate-calibrator", "candidate-calibrator.json",
        "--parity-oracle-threshold", "oracle-threshold.json",
        "--parity-candidate-threshold", "candidate-threshold.json",
        "--benchmark-manifest", "benchmark-manifest.json",
        "--benchmark-papers", "benchmark-papers.json",
        "--soak-manifest", "soak-manifest.json",
        "--soak-papers", "soak-papers.json",
        "--soak-record", "soak-record.json",
        "--output", str(output),
    ]
    for number in range(6):
        result.extend(("--benchmark-record", f"benchmark-{number}.json"))
    return result


def _stdout(capsys: pytest.CaptureFixture[str]) -> dict[str, object]:
    return json.loads(capsys.readouterr().out)


def test_build_evidence_help_lists_bound_artifacts(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as stopped:
        cli.main(["stage2-release", "build-evidence", "--help"])

    assert stopped.value.code == 0
    help_text = capsys.readouterr().out
    for option in ("--gold-manifest", "--rationale-worklist", "--parity-workload", "--benchmark-record", "--hidden-attestation", "--output"):
        assert option in help_text


def test_build_evidence_dry_run_validates_without_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "evidence.json"
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        cli, "build_stage2_release_evidence_index_bytes",
        lambda **values: captured.update(values) or b"validated",
    )

    assert cli.main(["--dry-run", *_arguments(output)]) == 0

    payload = _stdout(capsys)
    assert payload["status"] == "validated"
    assert payload["written"] is False
    assert not output.exists()
    assert len(captured["benchmark_record_paths"]) == 6


def test_build_evidence_writes_final_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "evidence.json"

    def write(**values: object) -> Path:
        path = values["output_path"]
        assert isinstance(path, Path)
        path.write_bytes(b"evidence")
        return path

    monkeypatch.setattr(cli, "write_stage2_release_evidence_index", write)
    args = _arguments(output)
    args[args.index("--output"):args.index("--output") + 2] = ["--hidden-attestation", "attestation.json", "--output", str(output)]

    assert cli.main(args) == 0

    payload = _stdout(capsys)
    assert output.read_bytes() == b"evidence"
    assert payload["status"] == "complete"
    assert payload["evidence_type"] == "stage2_release_evidence"


def test_build_evidence_requires_six_records_without_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "evidence.json"
    monkeypatch.setattr(cli, "write_stage2_release_evidence_index", lambda **_values: pytest.fail("must not write"))
    args = _arguments(output)
    index = args.index("--benchmark-record")
    del args[index:index + 2]

    assert cli.entrypoint(args) == 1

    payload = _stdout(capsys)
    assert "exactly six" in str(payload["error"])
    assert not output.exists()


def test_build_evidence_preserves_existing_output_on_writer_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "evidence.json"
    output.write_text("sentinel", encoding="utf-8")
    monkeypatch.setattr(
        cli, "write_stage2_release_evidence_index",
        lambda **_values: (_ for _ in ()).throw(FileExistsError("already exists")),
    )

    assert cli.entrypoint(_arguments(output)) == 1

    payload = _stdout(capsys)
    assert payload["error_type"] == "FileExistsError"
    assert output.read_text(encoding="utf-8") == "sentinel"
