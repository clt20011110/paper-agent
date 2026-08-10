"""Verify an installed wheel without relying on the source checkout."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from paper_agent.domain import QuerySpec
from paper_agent.manifests import load_catalog
from paper_agent.providers.builtin import FixtureTransport, create_builtin


def main() -> None:
    assert not (Path.cwd() / "pyproject.toml").exists()
    console = Path(sys.executable).with_name("paper-agent")
    completed = subprocess.run(
        [str(console), "doctor"],
        check=True,
        capture_output=True,
        text=True,
    )
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

    assert diagnosis["python_supported"] is True
    assert diagnosis["schema_count"] == 19
    assert len(catalog.providers) == 25
    assert len(catalog.venues) == 20
    assert batch.entries[0].external_id == "10.1000/wheel-check"


if __name__ == "__main__":
    main()
