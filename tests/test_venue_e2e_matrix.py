from __future__ import annotations

import json
import base64
import importlib.util
from pathlib import Path
import sqlite3
import subprocess
import sys

import yaml

from paper_agent.domain import EnvelopeStatus, SourceBatch, SourceEntry
from paper_agent.repository import PaperRepository
from paper_agent.storage import Database
from paper_agent.verification import MetadataCoordinator


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "run_venue_e2e_matrix.py"


def _runner_module():
    specification = importlib.util.spec_from_file_location("venue_e2e_matrix", SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_icml_native_stage1_and_test_only_stage2_persist_sqlite_checkpoints(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable, str(SCRIPT), "--venue", "icml", "--output-root", str(tmp_path),
            "--run-id", "icml-test", "--through-stage", "2",
        ],
        check=True, capture_output=True, text=True,
    )
    result = json.loads(completed.stdout)
    run_dir = tmp_path / "icml-test"
    database = run_dir / "papers.sqlite3"
    assert result["status"] == "complete"
    assert result["preflight"]["models_dispatched"] == 0
    assert result["native_pipeline"]["database"] == "papers.sqlite3"
    assert (run_dir / "stage1" / "metadata-snapshot.json").is_file()
    assert (run_dir / "stage1" / "result.json").is_file()
    stage2 = json.loads((run_dir / "stage2" / "result.json").read_text())
    assert stage2["mode"] == "TEST_ONLY"
    assert set(stage2["statuses"].values()) == {"relevant"}
    connection = sqlite3.connect(database)
    try:
        assert connection.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 1
        assert connection.execute(
            "SELECT verification_status FROM papers"
        ).fetchone()[0] == "verified"
        authors = json.loads(connection.execute("SELECT authors_json FROM papers").fetchone()[0])
        assert connection.execute("SELECT COUNT(*) FROM filter_decisions").fetchone()[0] == 1
        sources = connection.execute(
            "SELECT external_id, raw_metadata_json FROM paper_sources ORDER BY external_id"
        ).fetchall()
        stages = dict(connection.execute("SELECT run_id, stage FROM pipeline_runs"))
    finally:
        connection.close()
    assert stages["icml-test-stage2-test-only"] == "stage-2"
    assert authors == [
        "Michael Sun", "Minghao Guo", "Weize Yuan", "Veronika Thost",
        "Crystal Elaine Owens", "Aristotle Franklin Grosz", "Sharvaa Selvan",
        "Katelyn Zhou", "Hassan Mohiuddin", "Benjamin J Pedretti",
        "Zachary P Smith", "Jie Chen", "Wojciech Matusik",
    ]
    assert len(sources) == 2
    assert sources[0][0] != sources[1][0]
    assert sum("approved_venue_e2e_matrix" in row[1] for row in sources) == 1


def test_iclr_venue_only_replay_does_not_dispatch_topic_search(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable, str(SCRIPT), "--venue", "iclr", "--output-root", str(tmp_path),
            "--run-id", "iclr-test", "--through-stage", "1",
        ],
        check=True, capture_output=True, text=True,
    )
    assert json.loads(completed.stdout)["status"] == "complete"
    plan = json.loads(
        (tmp_path / "iclr-test" / "search" / "iclr-test-stage1" / "QUERY_PLAN.json").read_text()
    )
    assert plan["providers"][0]["roles"] == ["venue_primary"]
    assert plan["providers"][0]["native_query_hashes"] == []
    connection = sqlite3.connect(tmp_path / "iclr-test" / "papers.sqlite3")
    try:
        sources = connection.execute(
            "SELECT source_id, raw_metadata_json FROM paper_sources ORDER BY external_id"
        ).fetchall()
        provenance = {
            source_id: {
                row[0]
                for row in connection.execute(
                    "SELECT field_name FROM paper_field_provenance WHERE source_id = ?",
                    (source_id,),
                )
            }
            for source_id, _ in sources
        }
    finally:
        connection.close()
    assert len(sources) == 2
    assert all({"title", "authors"}.issubset(provenance[source_id]) for source_id, _ in sources)
    supplemental = next(
        (source_id, json.loads(raw_metadata))
        for source_id, raw_metadata in sources
        if json.loads(raw_metadata).get("source_role") == "public_pdf_locator_supplement"
    )
    assert supplemental[1]["canonical_field_provenance"] == "copied_from_stage1_canonical_paper"
    assert supplemental[1]["metadata_source_url"].startswith("https://proceedings.iclr.cc/")

    # Reproduce the compatibility path that originally failed: merging another
    # native batch reconstructs every existing source before verification.
    with Database(tmp_path / "iclr-test" / "papers.sqlite3") as database:
        repository = PaperRepository(database)
        canonical = repository.connection.execute("SELECT * FROM papers").fetchone()
        native = repository.connection.execute(
            """SELECT provider, external_id, landing_url
               FROM paper_sources
               WHERE json_extract(raw_metadata_json, '$.source_role') IS NULL"""
        ).fetchone()
        coordinator = MetadataCoordinator(repository, {})
        merged = coordinator.merge_batch(SourceBatch(
            source_run_id="iclr-regression-replay",
            query_hash="iclr-regression-replay",
            entries=(SourceEntry(
                provider=native["provider"],
                external_id=native["external_id"],
                title=canonical["title"],
                authors=tuple(json.loads(canonical["authors_json"])),
                publication_date=canonical["publication_date"],
                year=canonical["year"],
                venue_name=canonical["venue_name"],
                landing_url=native["landing_url"],
            ),),
            next_cursor=None,
            status=EnvelopeStatus.SUCCESS,
        ))
        evidence = coordinator._entries_for_paper(merged[0].paper_id)
    assert len(evidence) == 2
    assert all(entry.title == "UniGEM: A Unified Approach to Generation and Property Prediction for Molecules" for entry in evidence)


def test_dry_run_never_creates_a_run_directory(tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--venue", "icml", "--output-root", str(tmp_path), "--run-id", "dry", "--dry-run"],
        check=True, capture_output=True, text=True,
    )
    assert json.loads(completed.stdout)["status"] == "validated"
    assert not (tmp_path / "dry").exists()


def test_run_id_cannot_escape_output_root(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--venue",
            "icml",
            "--output-root",
            str(tmp_path),
            "--run-id",
            "../escape",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "run_id must be a plain" in completed.stderr
    assert not (tmp_path.parent / "escape").exists()


def test_matrix_schema_accepts_future_venue_year(tmp_path: Path) -> None:
    module = _runner_module()
    document = yaml.safe_load(
        (ROOT / "configs" / "e2e" / "venue-smoke-matrix.yaml").read_text()
    )
    document["venues"][0]["year"] = 2027
    config = tmp_path / "future-matrix.yaml"
    config.write_text(yaml.safe_dump(document), encoding="utf-8")

    loaded = module.load_matrix(config)

    assert loaded["venues"][0]["year"] == 2027


def test_all_configured_venues_have_a_valid_model_free_preflight() -> None:
    module = _runner_module()
    matrix = module.load_matrix(ROOT / "configs" / "e2e" / "venue-smoke-matrix.yaml")
    assert len(matrix["venues"]) == 20
    assert all(venue["paper"]["authors"] for venue in matrix["venues"])
    assert all(venue["paper"]["metadata_source_url"].startswith("https://") for venue in matrix["venues"])


def test_frozen_stage1_records_preserve_audited_authors_and_dois() -> None:
    module = _runner_module()
    matrix = module.load_matrix(ROOT / "configs" / "e2e" / "venue-smoke-matrix.yaml")
    for venue in matrix["venues"]:
        descriptor = module._stringify_mapping_keys(
            yaml.safe_load((ROOT / venue["descriptor"]).read_text())
        )
        _, record = module._snapshot_bundle(venue, descriptor)
        assert record["authors"] == venue["paper"]["authors"]
        assert record["doi"] == venue["paper"].get("doi")
        assert record["arxiv_id"] == venue["paper"].get("arxiv_id")
        assert record["metadata_source_url"] == venue["paper"]["metadata_source_url"]

    aaai = module._venue_by_id(matrix, "aaai")
    assert aaai["paper"]["doi"] == "10.1609/aaai.v39i24.34804"
    assert aaai["paper"]["authors"][0] == "Artem Zholus"


def test_restricted_primary_terms_and_journal_native_envelopes_are_frozen() -> None:
    module = _runner_module()
    matrix = module.load_matrix(ROOT / "configs" / "e2e" / "venue-smoke-matrix.yaml")
    expectations = {
        "tcad": ("ieee_xplore", "articles"),
        "nature_communications": ("springer_nature", "records"),
        "cell": ("cell_press", "search-results"),
        "science": ("aaas_science", "entries"),
    }
    for venue_id, (provider, envelope_key) in expectations.items():
        venue = module._venue_by_id(matrix, venue_id)
        descriptor = yaml.safe_load((ROOT / venue["descriptor"]).read_text())
        draft = module._query_draft(venue, descriptor)
        assert {item["provider"] for item in draft["terms_approvals"]} == {provider}
        bundle, _ = module._snapshot_bundle(venue, descriptor)
        discover = next(item for item in bundle["responses"] if item["operation"] == "discover")
        payload = json.loads(base64.b64decode(discover["body_base64"]))
        assert envelope_key in payload
