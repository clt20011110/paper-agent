from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

from paper_agent.stage2_sampling import load_private_corpus_snapshot, select_hidden_real, SamplingPolicy
from paper_agent.identity import paper_id_for


SCRIPT = Path(__file__).parents[1] / "scripts" / "freeze_stage2_crossref_snapshot.py"
SPEC = Path(__file__).parents[1] / "configs" / "stage2" / "real-sampling-crossref-v1.json"
module_spec = importlib.util.spec_from_file_location("freeze_stage2_crossref_snapshot", SCRIPT)
assert module_spec and module_spec.loader
freeze = importlib.util.module_from_spec(module_spec)
sys.modules[module_spec.name] = freeze
module_spec.loader.exec_module(freeze)


class FakeTransport:
    def __init__(self) -> None:
        self.last_response_body: bytes | None = None
        self.calls: list[dict[str, object]] = []

    def __call__(self, provider: str, operation: str, parameters: dict[str, object]) -> dict[str, object]:
        assert (provider, operation) == ("crossref", "search")
        assert parameters["filter"] == "type:journal-article,from-pub-date:2024-01-01,until-pub-date:2026-08-12"
        assert "query.title" in parameters
        call = len(self.calls)
        self.calls.append(dict(parameters))
        language = "zh" if "分子" in str(parameters["query.title"]) or any(
            character in str(parameters["query.title"])
            for character in "蛋白人工材料科学药物"
        ) else "en"
        items = [
            {
                "DOI": "https://doi.org/10.1000/shared.",
                "title": ["跨语言标题" if call == 0 else "Shared paper"],
                "abstract": "<jats:p>Shared &amp; clean</jats:p>",
                "type": "journal-article",
                "container-title": ["Journal"],
                "published": {"date-parts": [[2025, 1, 2]]},
                "URL": "https://doi.org/10.1000/shared",
            }
        ]
        items.extend(
            {
                "DOI": f"10.1000/{call}-{index}",
                "title": [f"中文标题 {call}-{index}" if language == "zh" else f"English title {call}-{index}"],
                "abstract": None if index == 0 else f"<p>Abstract {call} {index}</p>",
                "type": "journal-article",
                "container-title": ["Journal"],
                "published-online": {"date-parts": [[2025, 2, 3]]},
                "URL": f"https://doi.org/10.1000/{call}-{index}",
                "relation": (
                    {"is-preprint-of": [{"id-type": "doi", "id": "10.1000/0-2"}]}
                    if (call, index) == (0, 1)
                    else {}
                ),
            }
            for index in range(20)
        )
        payload = {"message": {"items": items}}
        self.last_response_body = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return payload


def test_freezes_deduplicated_valid_crossref_frame_and_raw_captures(tmp_path: Path) -> None:
    document = json.loads(SPEC.read_text(encoding="utf-8"))
    first = freeze.freeze_snapshot(document, FakeTransport())
    second = freeze.freeze_snapshot(document, FakeTransport())

    assert first.snapshot.hash() == second.snapshot.hash()
    assert len(first.captures) == 12
    assert {paper.sampling_probability for paper in first.snapshot.papers} == {150 / len(first.snapshot.papers)}
    assert all(paper.paper_id.startswith("paper-") for paper in first.snapshot.papers)
    assert all(paper.paper_id == paper_id_for(doi=paper.metadata["doi"]) for paper in first.snapshot.papers)
    assert all(paper.source == "crossref" for paper in first.snapshot.papers)
    assert all(paper.abstract is None or "<" not in paper.abstract for paper in first.snapshot.papers)
    shared = [paper for paper in first.snapshot.papers if paper.metadata["doi"] == "10.1000/shared"]
    assert len(shared) == 6  # one topic-local row, but its family remains visible across topics
    assert any(paper.cross_language_match for paper in first.snapshot.papers)
    related = {paper.metadata["doi"]: paper for paper in first.snapshot.papers if paper.metadata["doi"] in {"10.1000/0-1", "10.1000/0-2"}}
    assert related["10.1000/0-1"].paper_family == related["10.1000/0-2"].paper_family
    sampled = next(paper for paper in first.snapshot.papers if paper.metadata["doi"] == "10.1000/0-1")
    assert sampled.metadata["topic"] == "molecular_generation"
    assert sampled.metadata["topic_language"] == "en"
    assert sampled.metadata["query_language"] == "en"
    assert sampled.metadata["raw_response_sha256"] == freeze.sha256(first.captures[0].body).hexdigest()
    assert sampled.metadata["crossref_record"]["DOI"] == "10.1000/0-1"

    snapshot_path = tmp_path / "private-snapshot.json"
    raw_directory = tmp_path / "raw-crossref"
    manifest_path = tmp_path / "capture-manifest.json"
    freeze.publish_snapshot(
        first,
        output=snapshot_path,
        capture_directory=raw_directory,
        capture_manifest=manifest_path,
    )

    restored = load_private_corpus_snapshot(snapshot_path)
    selection = select_hidden_real(restored, SamplingPolicy(restored.sampling_policy_version, restored.sampling_seed))
    assert restored.hash() == first.snapshot.hash()
    assert len(selection.pair_keys) == 150
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["snapshot_hash"] == restored.hash()
    assert len(manifest["responses"]) == 12
    assert {item["filename"] for item in manifest["responses"]} == {path.name for path in raw_directory.iterdir()}
    assert all(len((raw_directory / item["filename"]).read_bytes()) == item["size_bytes"] for item in manifest["responses"])

    with pytest.raises(FileExistsError):
        freeze.publish_snapshot(
            first,
            output=snapshot_path,
            capture_directory=raw_directory,
            capture_manifest=manifest_path,
        )


def test_normalize_doi_rejects_non_doi_values() -> None:
    assert freeze.normalize_doi(" DOI:10.1000/A/B. ") == "10.1000/a/b"
    assert freeze.normalize_doi("not-a-doi") is None
