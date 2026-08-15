"""Verify the installed Stage 1 wheel without relying on the source checkout."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def _run(*command: str) -> None:
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr or result.stdout


def main() -> None:
    source_root = Path(__file__).resolve().parents[1]
    assert not (Path.cwd() / "pyproject.toml").exists()

    console = Path(sys.executable).with_name("paper-agent")
    _run(str(console), "--help")
    _run(sys.executable, "-I", "-m", "paper_agent", "--help")

    import paper_agent
    from paper_agent.catalog import load_venue_spec
    from paper_agent.loading import load_adapter, load_enrichers

    package_root = Path(paper_agent.__file__).resolve().parent
    assert not package_root.is_relative_to(source_root / "src")
    venue_specs = package_root / "venue_specs"
    assert venue_specs.is_dir()

    for venue_id in ("icml", "tcad"):
        spec = load_venue_spec(venue_id)
        assert (venue_specs / f"{venue_id}.toml").is_file()
        assert spec.adapter.startswith("adapters.")
        assert callable(load_adapter(spec.adapter).collect)
        enrichers = load_enrichers(spec.enrichers)
        assert all(callable(enricher.enrich) for enricher in enrichers)


if __name__ == "__main__":
    main()
