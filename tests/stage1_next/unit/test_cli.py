"""Offline tests for the minimal Stage 1 CLI composition root."""

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from paper_agent_next import cli
from paper_agent_next import http as http_module
from paper_agent_next.catalog import load_venue_spec
from paper_agent_next.collector import CollectionOutcome
from paper_agent_next.errors import CollectionError, InputError, PublicationError
from paper_agent_next.models import (
    RunCounts,
    RunRecord,
    RunStatus,
    VenueType,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = REPO_ROOT / "src"


def _valid_args(
    output_dir: Path,
    *,
    year: str = "2024",
    contact: str = "researcher@example.org",
) -> list[str]:
    return [
        "collect",
        "--venue",
        "icml",
        "--year",
        year,
        "--output",
        str(output_dir),
        "--contact",
        contact,
    ]


def _empty_counts() -> RunCounts:
    return RunCounts(0, 0, 0, 0, 0, 0, 0, 0)


def _outcome(status: RunStatus) -> CollectionOutcome:
    if status is RunStatus.COMPLETE:
        raise ValueError("use _valid_complete_outcome for complete status")
    common = {
        "venue_id": "icml",
        "venue_name": "International Conference on Machine Learning",
        "venue_type": VenueType.CONFERENCE,
        "year": 2024,
        "membership_complete": False,
        "metadata_complete": False,
        "complete": False,
        "counts": _empty_counts(),
        "pagination": None,
        "warnings": (),
        "errors": () if status is RunStatus.NOT_APPLICABLE else ("run failed",),
    }
    common["source_name"] = None
    return CollectionOutcome(
        (),
        (),
        RunRecord(status=status, **common),
    )


def _valid_complete_outcome() -> CollectionOutcome:
    from paper_agent_next.models import Pagination, SourceTotal, SourceTotalScope

    return CollectionOutcome(
        (),
        (),
        RunRecord(
            status=RunStatus.COMPLETE,
            venue_id="icml",
            venue_name="International Conference on Machine Learning",
            venue_type=VenueType.CONFERENCE,
            year=2024,
            source_name="pmlr",
            membership_complete=True,
            metadata_complete=True,
            complete=True,
            counts=_empty_counts(),
            pagination=Pagination(1, True, SourceTotal(0, SourceTotalScope.INCLUDED_PAPERS)),
            warnings=(),
            errors=(),
        ),
    )


def _outcome_for_status(status: RunStatus) -> CollectionOutcome:
    return _valid_complete_outcome() if status is RunStatus.COMPLETE else _outcome(status)


def _run_module(*arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    paths = [str(SOURCE_ROOT)]
    if existing_pythonpath:
        paths.extend(existing_pythonpath.split(os.pathsep))
    environment["PYTHONPATH"] = os.pathsep.join(paths)
    return subprocess.run(
        [sys.executable, "-m", "paper_agent_next", *arguments],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def test_module_and_collect_help_are_offline_and_expose_only_collect_surface() -> None:
    root_help = _run_module("--help")
    collect_help = _run_module("collect", "--help")

    assert root_help.returncode == 0
    assert collect_help.returncode == 0
    assert root_help.stdout.startswith("usage: paper-agent ")
    assert collect_help.stdout.startswith("usage: paper-agent collect ")
    assert "collect" in root_help.stdout
    for option in ("--venue", "--year", "--output", "--contact"):
        assert option in collect_help.stdout
    assert all(option not in collect_help.stdout for option in ("--force", "--timeout", "--config"))
    assert root_help.stderr == ""
    assert collect_help.stderr == ""


@pytest.mark.parametrize(
    "arguments",
    [
        ["unknown"],
        ["collect"],
        ["collect", "--venue", "icml", "--year", "2024", "--output", "out"],
        _valid_args(Path("out")) + ["--unknown"],
    ],
)
def test_missing_or_unknown_command_arguments_are_argparse_errors(arguments: list[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        cli.main(arguments)
    assert raised.value.code == 2


@pytest.mark.parametrize("year", ["+2024", "02024", " 2024", "20a4", "0999", "0000", ""])
def test_year_requires_four_ascii_digits_in_range(year: str, tmp_path: Path) -> None:
    arguments = _valid_args(tmp_path / "out")
    arguments[arguments.index("2024")] = year

    with pytest.raises(SystemExit) as raised:
        cli.main(arguments)

    assert raised.value.code == 2


@pytest.mark.parametrize(
    "contact",
    ["", "researcher", "researcher@example", "researcher@.org", "researcher@example.",
     " researcher@example.org", "researcher@example.org ", "researcher @example.org",
     "researcher@example .org", "researcher@@example.org"],
)
def test_contact_uses_the_minimal_email_like_boundary(contact: str, tmp_path: Path) -> None:
    arguments = _valid_args(tmp_path / "out", contact=contact)

    with pytest.raises(SystemExit) as raised:
        cli.main(arguments)

    assert raised.value.code == 2


def test_valid_contact_is_preserved_only_for_the_shared_http_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = load_venue_spec("icml")
    contact = "Researcher+stage1@example.org"
    adapter = object()
    client = object()
    outcome = _outcome_for_status(RunStatus.NOT_APPLICABLE)
    calls: dict[str, object] = {}

    monkeypatch.setattr(cli, "load_venue_spec", lambda venue: spec)
    monkeypatch.setattr(cli, "prepare_output_dir", lambda output: None)
    monkeypatch.setattr(cli, "_load_adapter", lambda path: adapter)

    def fake_http(value: str, timeout: float) -> object:
        calls["http"] = (value, timeout)
        return client

    monkeypatch.setattr(cli, "HttpClient", fake_http)

    def fake_collect(venue, year, loaded_adapter, http_client):
        calls["collector"] = (venue, year, loaded_adapter, http_client)
        return outcome

    monkeypatch.setattr(cli, "collect_venue_year", fake_collect)
    published: list[object] = []
    monkeypatch.setattr(cli, "publish_artifacts", lambda *values: published.append(values))

    assert cli.main(_valid_args(tmp_path / "out", contact=contact)) == 0
    assert calls["http"] == (contact, 30.0)
    assert calls["collector"] == (spec, 2024, adapter, client)
    assert published == [(tmp_path / "out", (), (), outcome.run)]
    assert contact not in json.dumps(outcome.run.to_dict())


def test_composition_order_and_shared_client_are_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = load_venue_spec("icml")
    adapter = object()
    outcome = _outcome_for_status(RunStatus.NOT_APPLICABLE)
    events: list[tuple[str, object]] = []
    client_instances: list[object] = []

    def fake_catalog(venue):
        events.append(("catalog", venue))
        return spec

    def fake_preflight(output):
        events.append(("preflight", output))

    def fake_loader(path):
        events.append(("loader", path))
        return adapter

    class FakeHttpClient:
        def __init__(self, contact, timeout):
            events.append(("client", (contact, timeout)))
            client_instances.append(self)

    def fake_collect(venue, year, loaded_adapter, http_client):
        events.append(("collector", (venue, year, loaded_adapter, http_client)))
        return outcome

    def fake_publish(output, papers, issues, run):
        events.append(("publisher", (output, papers, issues, run)))

    monkeypatch.setattr(cli, "load_venue_spec", fake_catalog)
    monkeypatch.setattr(cli, "prepare_output_dir", fake_preflight)
    monkeypatch.setattr(cli, "_load_adapter", fake_loader)
    monkeypatch.setattr(cli, "HttpClient", FakeHttpClient)
    monkeypatch.setattr(cli, "collect_venue_year", fake_collect)
    monkeypatch.setattr(cli, "publish_artifacts", fake_publish)

    assert cli.main(_valid_args(tmp_path / "out")) == 0
    assert [name for name, _ in events] == [
        "catalog",
        "preflight",
        "loader",
        "client",
        "collector",
        "publisher",
    ]
    assert len(client_instances) == 1
    assert events[4][1][3] is client_instances[0]
    assert events[5][1] == (tmp_path / "out", outcome.papers, outcome.issues, outcome.run)


def test_preflight_input_error_stops_before_loader_client_collector_publisher_or_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    spec = load_venue_spec("icml")
    calls: list[str] = []

    monkeypatch.setattr(cli, "load_venue_spec", lambda venue: calls.append("catalog") or spec)

    def fail_preflight(output):
        calls.append("preflight")
        raise InputError("output preflight failed")

    monkeypatch.setattr(cli, "prepare_output_dir", fail_preflight)
    monkeypatch.setattr(cli, "_load_adapter", lambda path: calls.append("loader"))
    monkeypatch.setattr(cli, "HttpClient", lambda *args: calls.append("client"))
    monkeypatch.setattr(cli, "collect_venue_year", lambda *args: calls.append("collector"))
    monkeypatch.setattr(cli, "publish_artifacts", lambda *args: calls.append("publisher"))
    monkeypatch.setattr(http_module, "urlopen", lambda *args, **kwargs: pytest.fail("network access"))

    assert cli.main(_valid_args(tmp_path / "out")) == 2
    assert calls == ["catalog", "preflight"]
    assert capsys.readouterr().err == "error: output preflight failed\n"


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (RunStatus.COMPLETE, 0),
        (RunStatus.NOT_APPLICABLE, 0),
        (RunStatus.PARTIAL, 3),
        (RunStatus.FAILED, 4),
    ],
)
def test_status_exit_codes_are_applied_after_publishing(
    status: RunStatus, expected: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = load_venue_spec("icml")
    outcome = _outcome_for_status(status)
    published: list[tuple[object, ...]] = []

    monkeypatch.setattr(cli, "load_venue_spec", lambda venue: spec)
    monkeypatch.setattr(cli, "prepare_output_dir", lambda output: None)
    monkeypatch.setattr(cli, "_load_adapter", lambda path: object())
    monkeypatch.setattr(cli, "HttpClient", lambda contact, timeout: object())
    monkeypatch.setattr(cli, "collect_venue_year", lambda *args: outcome)
    monkeypatch.setattr(cli, "publish_artifacts", lambda *args: published.append(args))

    assert cli.main(_valid_args(tmp_path / status.value)) == expected
    assert published == [(tmp_path / status.value, outcome.papers, outcome.issues, outcome.run)]


@pytest.mark.parametrize(
    ("error", "expected_code", "expected_stderr"),
    [
        (InputError("bad input"), 2, "error: bad input\n"),
        (PublicationError("secret publication details"), 4, "error: publication failed\n"),
        (CollectionError("provider HTML credential=secret"), 4, "error: runtime failure\n"),
        (RuntimeError("secret unexpected details"), 4, "error: unexpected runtime failure\n"),
    ],
)
def test_exception_mapping_is_short_and_safe(
    error: BaseException,
    expected_code: int,
    expected_stderr: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "_collect", lambda args: (_ for _ in ()).throw(error))

    assert cli.main(_valid_args(tmp_path / "out")) == expected_code
    stderr = capsys.readouterr().err
    assert stderr == expected_stderr
    assert "secret" not in stderr


def test_trusted_loader_loads_current_pmlr_adapter() -> None:
    adapter = cli._load_adapter("adapters.pmlr:PmlrAdapter")

    assert adapter.__class__.__name__ == "PmlrAdapter"
    assert adapter.__class__.__module__ == "paper_agent_next.adapters.pmlr"


@pytest.mark.parametrize("mode", ["missing_module", "missing_attribute", "non_callable", "type_error"])
def test_trusted_loader_errors_have_a_cause_without_scanning_or_leaking(
    mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def importer(module_name: str):
        calls.append(module_name)
        if mode == "missing_module":
            raise ModuleNotFoundError("secret internal module")
        if mode == "missing_attribute":
            return SimpleNamespace()
        if mode == "non_callable":
            return SimpleNamespace(PmlrAdapter=object())

        def constructor():
            raise TypeError("secret constructor details")

        return SimpleNamespace(PmlrAdapter=constructor)

    monkeypatch.setattr(cli.importlib, "import_module", importer)

    with pytest.raises(InputError) as raised:
        cli._load_adapter("adapters.pmlr:PmlrAdapter")

    assert calls == ["paper_agent_next.adapters.pmlr"]
    assert isinstance(raised.value.__cause__, (ImportError, AttributeError, TypeError))
    assert str(raised.value) == "could not load configured adapter"
    assert "secret" not in str(raised.value)


def test_real_not_applicable_composition_publishes_three_artifacts_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[object] = []

    def fail_network(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("network access")

    monkeypatch.setattr(http_module, "urlopen", fail_network)
    output_dir = tmp_path / "artifacts"

    assert cli.main(_valid_args(output_dir, year="1981", contact="researcher@example.org")) == 0
    assert calls == []
    assert sorted(path.name for path in output_dir.iterdir()) == [
        "issues.jsonl",
        "papers.jsonl",
        "run.json",
    ]
    assert (output_dir / "papers.jsonl").read_bytes() == b""
    assert (output_dir / "issues.jsonl").read_bytes() == b""
    run = json.loads((output_dir / "run.json").read_text(encoding="utf-8"))
    assert run["status"] == "not_applicable"
    assert all(value == 0 for value in run["counts"].values())
