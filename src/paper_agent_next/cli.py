"""Minimal command-line composition root for Stage 1 collection."""

import argparse
from collections.abc import Sequence
import importlib
from pathlib import Path
import sys

from .catalog import load_venue_spec
from .collector import collect_venue_year
from .errors import InputError, PublicationError, Stage1Error
from .http import HttpClient
from .models import RunStatus
from .output import prepare_output_dir, publish_artifacts

__all__ = ["main"]


_EXIT_CODES = {
    RunStatus.COMPLETE: 0,
    RunStatus.NOT_APPLICABLE: 0,
    RunStatus.PARTIAL: 3,
    RunStatus.FAILED: 4,
}


def _parse_year(value: str) -> int:
    if len(value) != 4 or any(character < "0" or character > "9" for character in value):
        raise argparse.ArgumentTypeError("year must be exactly four ASCII digits")
    year = int(value)
    if not 1000 <= year <= 9999:
        raise argparse.ArgumentTypeError("year must be between 1000 and 9999")
    return year


def _parse_contact(value: str) -> str:
    if (
        not value
        or any(character.isspace() for character in value)
        or value.count("@") != 1
    ):
        raise argparse.ArgumentTypeError("contact must be an email-like value")
    local, domain = value.split("@")
    if not local or not domain or not any(
        character == "." and index not in {0, len(domain) - 1}
        for index, character in enumerate(domain)
    ):
        raise argparse.ArgumentTypeError("contact must be an email-like value")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paper-agent",
        allow_abbrev=False,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect_parser = subparsers.add_parser(
        "collect",
        allow_abbrev=False,
        help="collect one venue-year",
    )
    collect_parser.add_argument("--venue", required=True)
    collect_parser.add_argument("--year", required=True, type=_parse_year)
    collect_parser.add_argument("--output", required=True, type=Path)
    collect_parser.add_argument("--contact", required=True, type=_parse_contact)
    return parser


def _load_adapter(adapter_path: str) -> object:
    module_name, _, attribute_name = adapter_path.partition(":")
    try:
        module = importlib.import_module(f"paper_agent_next.{module_name}")
        adapter_factory = getattr(module, attribute_name)
        if not callable(adapter_factory):
            raise TypeError("configured adapter is not callable")
        return adapter_factory()
    except (ImportError, AttributeError, TypeError) as error:
        raise InputError("could not load configured adapter") from error


def _collect(args: argparse.Namespace) -> int:
    venue_spec = load_venue_spec(args.venue)
    prepare_output_dir(args.output)
    adapter = _load_adapter(venue_spec.adapter)
    http_client = HttpClient(args.contact, 30.0)
    outcome = collect_venue_year(venue_spec, args.year, adapter, http_client)
    publish_artifacts(args.output, outcome.papers, outcome.issues, outcome.run)
    return _EXIT_CODES[outcome.run.status]


def _report_error(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _build_parser().parse_args(argv)
        return _collect(args)
    except InputError as error:
        _report_error(str(error))
        return 2
    except PublicationError:
        _report_error("publication failed")
        return 4
    except Stage1Error:
        _report_error("runtime failure")
        return 4
    except Exception:
        _report_error("unexpected runtime failure")
        return 4
