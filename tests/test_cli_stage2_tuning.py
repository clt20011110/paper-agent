from __future__ import annotations

import json
from pathlib import Path

import pytest

from paper_agent import cli


def test_stage2_tuning_select_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as stopped:
        cli.main(["stage2-tuning", "select", "--help"])
    assert stopped.value.code == 0
    assert "--input" in capsys.readouterr().out


def test_stage2_tuning_select_dry_run_has_no_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "winner.json"
    document = {"document_batch_size": 32, "adjudicator_concurrency": 8, "input_record_hashes": ["a" * 64] * 63, "selection_hash": "b" * 64}
    monkeypatch.setattr(cli, "build_stage2_tuning_winner_document", lambda _input: document)

    assert cli.main(["--dry-run", "stage2-tuning", "select", "--input", str(tmp_path / "input.json"), "--output", str(output)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "validated"
    assert payload["written"] is False
    assert not output.exists()


def test_stage2_tuning_select_writes_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "winner.json"
    document = {"document_batch_size": 16, "adjudicator_concurrency": 4, "input_record_hashes": ["a" * 64] * 63, "selection_hash": "b" * 64}
    def write(_input: Path, path: Path) -> dict[str, object]:
        path.write_text("winner")
        return document

    monkeypatch.setattr(cli, "write_stage2_tuning_winner", write)

    assert cli.main(["stage2-tuning", "select", "--input", str(tmp_path / "input.json"), "--output", str(output)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert output.read_text() == "winner"
    assert payload["input_record_count"] == 63
