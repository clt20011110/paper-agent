"""Paper Agent command-line entry point."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Sequence

from . import __version__
from .schema import schema_directory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="paper-agent")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--dry-run", action="store_true")
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("doctor", help="inspect the local runtime")
    return parser


def doctor() -> dict[str, object]:
    schemas = sorted(schema_directory().glob("*.schema.json"))
    return {
        "paper_agent_version": __version__,
        "python": sys.version.split()[0],
        "python_supported": (3, 11) <= sys.version_info[:2] <= (3, 13),
        "schema_count": len(schemas),
        "codex_cli": shutil.which("codex"),
        "omlx_cli": shutil.which("omlx"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        print(json.dumps(doctor(), ensure_ascii=False, sort_keys=True))
        return 0
    raise AssertionError(args.command)
