"""Locate version-bound installation assets in a checkout or installed wheel."""

from __future__ import annotations

from pathlib import Path
import sysconfig

from . import __version__


MODEL_LOCK_RELATIVE_PATHS = (
    Path("configs/stage2/models/bge-reranker-v2-m3-fp32.lock.json"),
    Path("configs/stage2/models/qwen3.5-9b-8bit.lock.json"),
)
EXAMPLE_CONFIG_RELATIVE_PATHS = (
    Path("example_config.yaml"),
    Path("configs/abstract_focus.yaml"),
    Path("configs/journal_smoke.yaml"),
    Path("configs/query_draft.example.yaml"),
    Path("configs/smoke_supported.yaml"),
)
PAPER_AGENT_SKILL_RELATIVE_PATH = Path("skills/paper-agent")


def release_asset_root(override: str | Path | None = None) -> Path:
    """Return the immutable asset root for the running package version.

    A source checkout already has the release layout at its repository root.
    Wheels install the same relative files below a versioned data directory so
    multiple application versions cannot silently share templates or skills.
    """
    if override is not None:
        return Path(override)
    source = _source_checkout_root()
    if source is not None:
        return source
    return (
        Path(sysconfig.get_path("data"))
        / "share"
        / "paper-agent"
        / __version__
    )


def stage2_model_lock_paths(root: str | Path | None = None) -> tuple[Path, ...]:
    base = release_asset_root(root)
    return tuple(base / relative for relative in MODEL_LOCK_RELATIVE_PATHS)


def example_config_paths(root: str | Path | None = None) -> tuple[Path, ...]:
    base = release_asset_root(root)
    return tuple(base / relative for relative in EXAMPLE_CONFIG_RELATIVE_PATHS)


def paper_agent_skill_directory(root: str | Path | None = None) -> Path:
    return release_asset_root(root) / PAPER_AGENT_SKILL_RELATIVE_PATH


def _source_checkout_root() -> Path | None:
    candidate = Path(__file__).resolve().parents[2]
    required = (
        candidate / "pyproject.toml",
        *(candidate / relative for relative in MODEL_LOCK_RELATIVE_PATHS),
        candidate / PAPER_AGENT_SKILL_RELATIVE_PATH / "SKILL.md",
    )
    return candidate if all(path.is_file() for path in required) else None
