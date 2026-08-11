from __future__ import annotations

from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from paper_agent.stage2_sampling import CurationWorklist, CurationWorklistRow


SCRIPT = Path(__file__).parents[1] / "scripts" / "suggest_stage2_curation.py"
SPEC = importlib.util.spec_from_file_location("suggest_stage2_curation", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _worklist(path: Path, count: int = 8) -> CurationWorklist:
    worklist = CurationWorklist(
        snapshot_hash="a" * 64,
        hidden_real_freeze_frame_hash="b" * 64,
        rows=tuple(
            CurationWorklistRow(
                topic="drug discovery" if index == 8 else "molecular generation",
                paper_id="paper-00" if index == 8 else f"paper-{index:02d}",
                title=f"Title {index}",
                abstract=None if index == 1 else f"Abstract {index}",
                source="crossref",
                language="zh" if index == 2 else "en",
                paper_family=f"family-{index}",
                abstract_incomplete=index == 3,
                cross_language_match=index == 7,
            )
            for index in range(count)
        ),
    )
    path.write_text(json.dumps(worklist.document()), encoding="utf-8")
    return worklist


def _lock(path: Path) -> str:
    path.write_text(json.dumps({"model_id": "qwen-test", "file_hashes": {}}), encoding="utf-8")
    return sha256(path.read_bytes()).hexdigest()


def test_suggestions_batch_exact_ids_and_write_redacted_evidence(tmp_path: Path) -> None:
    worklist_path = tmp_path / "worklist.json"
    worklist = _worklist(worklist_path, count=14)
    lock_path = tmp_path / "model.lock.json"
    lock_hash = _lock(lock_path)
    output_path, evidence_path = tmp_path / "decisions.json", tmp_path / "evidence.json"
    requests: list[dict[str, object]] = []

    def fake_transport(request: dict[str, object]) -> dict[str, object]:
        requests.append(request)
        papers = json.loads(request["messages"][1]["content"])["papers"]  # type: ignore[index]
        return {"choices": [{"message": {"content": json.dumps({"decisions": [
            {
                "topic": paper["topic"],
                "paper_id": paper["paper_id"],
                "provisional_label": 3 if paper["topic"] == "drug discovery" else int(paper["paper_id"].split("-")[-1]) % 4,
            }
            for paper in reversed(papers)
        ]})}}]}

    decisions, evidence = MODULE.suggest(
        worklist_path=worklist_path,
        model_lock_path=lock_path,
        output_path=output_path,
        evidence_path=evidence_path,
        transport=fake_transport,
        batch_size=6,
        concurrency=2,
    )

    assert len(requests) == 3
    assert all(request["model"] == "qwen-test" for request in requests)
    assert all(request["temperature"] == 0 and request["seed"] == 0 for request in requests)
    assert all(request["response_format"]["json_schema"]["strict"] is True for request in requests)  # type: ignore[index]
    duplicate_request = next(
        request for request in requests
        if {(paper["topic"], paper["paper_id"])
            for paper in json.loads(request["messages"][1]["content"])["papers"]} >= {  # type: ignore[index]
                ("molecular generation", "paper-00"),
                ("drug discovery", "paper-00"),
            }
    )
    duplicate_papers = json.loads(duplicate_request["messages"][1]["content"])["papers"]  # type: ignore[index]
    duplicate_schema = duplicate_request["response_format"]["json_schema"]["schema"]  # type: ignore[index]
    assert len(duplicate_papers) == 6
    assert {(paper["topic"], paper["paper_id"]) for paper in duplicate_papers} >= {
        ("molecular generation", "paper-00"),
        ("drug discovery", "paper-00"),
    }
    assert duplicate_schema["properties"]["decisions"]["minItems"] == len(duplicate_papers)
    assert duplicate_schema["properties"]["decisions"]["maxItems"] == len(duplicate_papers)
    assert decisions == json.loads(output_path.read_text(encoding="utf-8"))
    assert evidence == json.loads(evidence_path.read_text(encoding="utf-8"))
    assert output_path.stat().st_mode & 0o777 == 0o600
    assert evidence_path.stat().st_mode & 0o777 == 0o600
    assert decisions["worklist_hash"] == worklist.hash()
    assert {(row["topic"], row["paper_id"]) for row in decisions["rows"]} == {row.key for row in worklist.rows}
    by_key = {(row["topic"], row["paper_id"]): row for row in decisions["rows"]}
    assert by_key[("molecular generation", "paper-00")]["hard_negative"] is True
    assert by_key[("drug discovery", "paper-00")]["provisional_label"] == 3
    assert by_key[("drug discovery", "paper-00")]["hard_negative"] is False
    assert by_key[("molecular generation", "paper-03")]["hard_positive"] is True
    assert by_key[("molecular generation", "paper-07")]["hard_positive"] is True
    assert evidence["model_lock_sha256"] == lock_hash
    assert evidence["case_count"] == 14 and evidence["request_count"] == 3
    assert evidence["label_counts"] == {"0": 3, "1": 4, "2": 3, "3": 4}
    serialized_evidence = json.dumps(evidence)
    assert "Title" not in serialized_evidence and "Abstract" not in serialized_evidence and "paper-00" not in serialized_evidence


def test_failure_or_existing_output_never_replaces_results(tmp_path: Path) -> None:
    worklist_path = tmp_path / "worklist.json"
    _worklist(worklist_path)
    lock_path = tmp_path / "model.lock.json"
    _lock(lock_path)
    output_path, evidence_path = tmp_path / "decisions.json", tmp_path / "evidence.json"

    def invalid_transport(_: dict[str, object]) -> dict[str, object]:
        return {"choices": [{"message": {"content": '{"decisions": []}'}}]}

    with pytest.raises(ValueError, match="exactly preserve"):
        MODULE.suggest(
            worklist_path=worklist_path,
            model_lock_path=lock_path,
            output_path=output_path,
            evidence_path=evidence_path,
            transport=invalid_transport,
        )
    assert not output_path.exists() and not evidence_path.exists()

    output_path.write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError, match="no-replace"):
        MODULE.suggest(
            worklist_path=worklist_path,
            model_lock_path=lock_path,
            output_path=output_path,
            evidence_path=evidence_path,
            transport=invalid_transport,
        )
    assert output_path.read_text(encoding="utf-8") == "existing"


def test_pair_validation_rejects_nonexistent_topic_paper_combination() -> None:
    response = {"choices": [{"message": {"content": json.dumps({"decisions": [
        {"topic": "topic-a", "paper_id": "paper-b", "provisional_label": 1},
        {"topic": "topic-b", "paper_id": "paper-a", "provisional_label": 2},
    ]})}}]}

    with pytest.raises(ValueError, match="exactly preserve"):
        MODULE._extract_decisions(response, {("topic-a", "paper-a"), ("topic-b", "paper-b")})
