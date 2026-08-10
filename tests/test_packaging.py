from __future__ import annotations

from pathlib import Path
import tomllib

from paper_agent.cli import doctor
from paper_agent.domain import QuerySpec
from paper_agent.manifests import load_catalog
from paper_agent.providers.builtin import FixtureTransport, create_builtin


def test_runtime_data_and_builtin_work_outside_repository_cwd(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    catalog = load_catalog()
    provider = create_builtin(
        "crossref",
        FixtureTransport(
            {
                "crossref:search:first": {
                    "message": {
                        "items": [
                            {
                                "DOI": "10.1000/package-check",
                                "title": ["Installed package check"],
                            }
                        ]
                    }
                }
            }
        ),
    )
    batch = provider.search(QuerySpec(1, "package-check", "fixture", page_size=1))

    assert len(catalog.venues) == 20
    assert batch.entries[0].external_id == "10.1000/package-check"
    assert doctor()["python_supported"] is True


def test_console_script_uses_the_structured_error_boundary() -> None:
    project = tomllib.loads(
        (Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    )

    assert project["project"]["scripts"]["paper-agent"] == "paper_agent.cli:entrypoint"
