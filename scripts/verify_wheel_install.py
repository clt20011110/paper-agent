"""Verify an installed wheel without relying on the source checkout."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from paper_agent.analysis_registry import registry_directory
from paper_agent.authorized_skill_runtime import audit_manifest_path
from paper_agent.codex_exec import prompt_directory
from paper_agent.domain import QuerySpec
from paper_agent.manifests import load_catalog
from paper_agent.providers.builtin import FixtureTransport, create_builtin
from paper_agent.report_artifacts import audit_rubric_hash
from paper_agent.schema import schema_directory
from paper_agent.storage import Database


def main() -> None:
    assert not (Path.cwd() / "pyproject.toml").exists()
    console = Path(sys.executable).with_name("paper-agent")
    completed = subprocess.run(
        [str(console), "doctor"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode in {0, 1}
    diagnosis = json.loads(completed.stdout)
    catalog = load_catalog()
    provider = create_builtin(
        "crossref",
        FixtureTransport(
            {
                "crossref:search:first": {
                    "message": {
                        "items": [
                            {
                                "DOI": "10.1000/wheel-check",
                                "title": ["Wheel install check"],
                            }
                        ]
                    }
                }
            }
        ),
    )
    batch = provider.search(QuerySpec(1, "wheel-check", "fixture", page_size=1))

    checks = {item["name"]: item for item in diagnosis["checks"]}
    assert checks["python"]["status"] == "pass"
    assert len(catalog.providers) == 25
    assert len(catalog.venues) == 20
    assert batch.entries[0].external_id == "10.1000/wheel-check"
    assert len(tuple(schema_directory().glob("*.json"))) >= 19
    assert len(tuple(prompt_directory().glob("*.md"))) >= 7
    assert (registry_directory() / "analysis-normalization-v1.yaml").is_file()
    assert audit_manifest_path().is_file()
    assert len(audit_rubric_hash()) == 64
    with TemporaryDirectory() as directory:
        with Database(Path(directory) / "wheel.sqlite3") as database:
            database.migrate()
            assert database.current_version() >= 16


if __name__ == "__main__":
    main()
