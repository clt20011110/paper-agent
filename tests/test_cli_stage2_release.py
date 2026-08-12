from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import shutil

import pytest

from paper_agent import cli

from test_stage2_release_v3 import V3Bundle, _build_v3_bundle


@pytest.fixture(scope="module")
def release_template(tmp_path_factory: pytest.TempPathFactory) -> V3Bundle:
    return _build_v3_bundle(tmp_path_factory.mktemp("stage2-release-cli"))


@pytest.fixture
def release_bundle(release_template: V3Bundle, tmp_path: Path) -> V3Bundle:
    root = tmp_path / "bundle"
    shutil.copytree(release_template.root, root)
    trust_path = tmp_path / "deployment-hidden-evaluator-trust.json"
    parity_trust_path = tmp_path / "deployment-parity-oracle-trust.json"
    shutil.copy2(release_template.trust_path, trust_path)
    shutil.copy2(release_template.parity_trust_path, parity_trust_path)
    return replace(
        release_template,
        root=root,
        release_path=root / release_template.release_path.name,
        trust_path=trust_path,
        parity_trust_path=parity_trust_path,
        plan=deepcopy(release_template.plan),
    )


def _arguments(bundle: V3Bundle, output: Path) -> list[str]:
    return [
        "stage2-release",
        "assemble",
        "--candidate",
        str(bundle.root / "candidate.json"),
        "--evidence",
        str(bundle.root / "stage2-release-evidence.json"),
        "--trust-manifest",
        str(bundle.trust_path),
        "--parity-oracle-trust",
        str(bundle.parity_trust_path),
        "--output",
        str(output),
    ]


def _stdout(capsys: pytest.CaptureFixture[str]) -> dict[str, object]:
    return json.loads(capsys.readouterr().out)


def test_stage2_release_assemble_help_lists_required_inputs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as stopped:
        cli.main(["stage2-release", "assemble", "--help"])

    assert stopped.value.code == 0
    help_text = capsys.readouterr().out
    for option in (
        "--candidate",
        "--evidence",
        "--trust-manifest",
        "--parity-oracle-trust",
        "--output",
    ):
        assert option in help_text


def test_stage2_release_assemble_writes_release_and_emits_public_summary(
    release_bundle: V3Bundle,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = release_bundle.root / "assembled-stage2-release.json"

    assert cli.main(_arguments(release_bundle, output)) == 0

    payload = _stdout(capsys)
    assert output.is_file()
    assert payload["command"] == "stage2-release.assemble"
    assert payload["status"] == "complete"
    assert payload["stage"] == "stage2"
    assert payload["event_code"] == "stage2-release.assemble.completed"
    assert payload["dry_run"] is False
    assert payload["written"] is True
    candidate = json.loads(
        (release_bundle.root / "candidate.json").read_text(encoding="utf-8")
    )
    assert payload["candidate_id"] == candidate["profile"]
    assert payload["release"]["path"] == str(output.resolve())
    assert set(payload["gate_hashes"]) == {
        "promotion",
        "structured_replay",
        "rationale",
        "parity",
        "benchmark",
        "soak",
    }


def test_stage2_release_assemble_rejects_existing_output_without_replacing_it(
    release_bundle: V3Bundle,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = release_bundle.root / "assembled-stage2-release.json"
    output.write_text("sentinel", encoding="utf-8")

    assert cli.entrypoint(_arguments(release_bundle, output)) == 1

    payload = _stdout(capsys)
    assert payload["command"] == "stage2-release.assemble"
    assert payload["event_code"] == "stage2-release.assemble.failed"
    assert payload["error"] == "Stage 2 release assembly verification failed"
    assert output.read_text(encoding="utf-8") == "sentinel"


def test_stage2_release_assemble_dry_run_validates_without_output(
    release_bundle: V3Bundle,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = release_bundle.root / "assembled-stage2-release.json"

    assert cli.main([*_arguments(release_bundle, output), "--dry-run"]) == 0

    payload = _stdout(capsys)
    assert payload["status"] == "validated"
    assert payload["stage"] == "stage2"
    assert payload["event_code"] == "stage2-release.assemble.completed"
    assert payload["dry_run"] is True
    assert payload["written"] is False
    assert payload["release"]["path"] == str(output.resolve())
    assert not output.exists()


def test_stage2_release_dry_run_never_creates_an_invalid_output_parent(
    release_bundle: V3Bundle,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing_parent = release_bundle.root / "missing"
    output = missing_parent / "assembled-stage2-release.json"

    assert cli.entrypoint([*_arguments(release_bundle, output), "--dry-run"]) == 1

    payload = _stdout(capsys)
    assert payload["command"] == "stage2-release.assemble"
    assert payload["stage"] == "stage2"
    assert payload["event_code"] == "stage2-release.assemble.failed"
    assert payload["error"] == "Stage 2 release assembly verification failed"
    assert not missing_parent.exists()
    assert not output.exists()


def test_stage2_release_assemble_trust_failure_is_redacted_and_fail_closed(
    release_bundle: V3Bundle,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "PRIVATE-HIDDEN-MATERIAL-MUST-NOT-LEAK"
    release_bundle.trust_path.write_text(
        json.dumps({"private_material": secret}) + "\n",
        encoding="utf-8",
    )
    output = release_bundle.root / "assembled-stage2-release.json"

    assert cli.entrypoint(_arguments(release_bundle, output)) == 1

    payload = _stdout(capsys)
    assert payload["status"] == "failed"
    assert payload["error"] == "Stage 2 release assembly verification failed"
    assert secret not in json.dumps(payload)
    assert not output.exists()
