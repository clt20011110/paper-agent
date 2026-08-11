from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sqlite3
import sys

import pytest


ROOT = Path(__file__).parents[1]


def _load(name: str, filename: str):
    specification = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    assert specification and specification.loader
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


neurips = _load("freeze_stage2_neurips_corpus", "freeze_stage2_neurips_corpus.py")
workload = _load("build_stage2_workload_frame", "build_stage2_workload_frame.py")


class FakeNeuripsTransport:
    def __init__(self) -> None:
        self.last_response_body: bytes | None = None
        self.last_response_sha256: str | None = None
        self.calls: list[dict[str, object]] = []

    def __call__(self, provider: str, operation: str, parameters: dict[str, object]) -> dict[str, object]:
        assert (provider, operation) == ("neurips_proceedings", "discover")
        self.calls.append(dict(parameters))
        cursor = int(parameters["cursor"] or 0)
        entries = [
            {"external_id": f"NeurIPS-2024-{index}", "title": f"Title {index}", "document_type": "proceedings-article"}
            for index in range(cursor, min(cursor + 1_000, 1_201))
        ]
        self.last_response_body = b"<html>official NeurIPS listing</html>"
        self.last_response_sha256 = neurips.sha256(self.last_response_body).hexdigest()
        next_cursor = str(cursor + len(entries)) if cursor + len(entries) < 1_201 else None
        return {"entries": entries, "next_cursor": next_cursor}


def test_neurips_freeze_paginates_one_listing_and_never_replaces_outputs(tmp_path: Path) -> None:
    papers, manifest, raw_html = neurips.freeze_year(2024, FakeNeuripsTransport())
    assert len(papers["papers"]) == 1_201
    assert manifest["page_count"] == 2
    assert manifest["raw_html"] == {
        "sha256": neurips.sha256(raw_html).hexdigest(),
        "size_bytes": len(raw_html),
    }
    assert all(item["abstract"] is None for item in papers["papers"])

    papers_path = tmp_path / "neurips-papers.json"
    capture_path = tmp_path / "capture.json"
    raw_path = tmp_path / "listing.html"
    neurips.publish(
        papers=papers,
        manifest=manifest,
        raw_html=raw_html,
        papers_output=papers_path,
        manifest_output=capture_path,
        raw_html_output=raw_path,
    )
    assert json.loads(papers_path.read_text())["papers"] == papers["papers"]
    assert raw_path.read_bytes() == raw_html
    with pytest.raises(FileExistsError):
        neurips.publish(
            papers=papers,
            manifest=manifest,
            raw_html=raw_html,
            papers_output=papers_path,
            manifest_output=capture_path,
            raw_html_output=raw_path,
        )


def _sqlite_corpus(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "CREATE TABLE papers (paper_id TEXT PRIMARY KEY, title TEXT NOT NULL, abstract TEXT, keywords_json TEXT NOT NULL)"
        )
        rows = [
            (f"paper-{index:05d}", f"Title {index}", f"Abstract {index}" if index < 950 else None, "[]")
            for index in range(10_000)
        ]
        connection.executemany("INSERT INTO papers VALUES (?, ?, ?, ?)", rows)
        connection.commit()
    finally:
        connection.close()


def test_workload_frame_uses_sqlite_exact_counts_hashes_and_no_replace(tmp_path: Path) -> None:
    database = tmp_path / "papers.sqlite3"
    _sqlite_corpus(database)
    papers = workload._deduplicate([workload._load_sqlite_papers(database)])
    performance, soak, receipt = workload.build_workload_frame(papers, seed=17)
    again = workload.build_workload_frame(papers, seed=17)

    assert len(performance["papers"]) == 1_000
    assert sum(item["abstract"] is None for item in performance["papers"]) == 100
    assert len(soak["papers"]) == 10_000
    assert receipt["performance"]["papers_corpus_hash"] == again[2]["performance"]["papers_corpus_hash"]
    assert len(receipt["performance"]["normal_qwen_ids"]) == 150
    assert len(receipt["performance"]["stress_qwen_ids"]) == 300
    missing_ids = {
        item["paper_id"] for item in performance["papers"] if item["abstract"] is None
    }
    normal_ids = set(receipt["performance"]["normal_qwen_ids"])
    stress_ids = set(receipt["performance"]["stress_qwen_ids"])
    assert missing_ids <= normal_ids <= stress_ids
    assert receipt["omitted_bindings"] == ["stage2_config_hash", "threshold_artifact_hashes", "model_lock_hashes"]

    performance_path = tmp_path / "performance-papers.json"
    soak_path = tmp_path / "soak-papers.json"
    receipt_path = tmp_path / "receipt.json"
    workload.publish(
        performance=performance,
        soak=soak,
        receipt=receipt,
        performance_output=performance_path,
        soak_output=soak_path,
        receipt_output=receipt_path,
    )
    assert json.loads(receipt_path.read_text())["soak"]["paper_count"] == 10_000
    with pytest.raises(FileExistsError):
        workload.publish(
            performance=performance,
            soak=soak,
            receipt=receipt,
            performance_output=performance_path,
            soak_output=soak_path,
            receipt_output=receipt_path,
        )
