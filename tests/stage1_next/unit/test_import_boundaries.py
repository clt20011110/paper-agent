"""Import-boundary tests for the isolated Stage 1 package."""

import ast
import os
from pathlib import Path
import subprocess
import sys
import tomllib


def _package_root() -> Path:
    return Path(__file__).resolve().parents[3] / "src" / "paper_agent_next"


def _is_old_package(module: str | None) -> bool:
    return module == "paper_agent" or bool(module and module.startswith("paper_agent."))


def test_stage1_console_entry_point_and_venue_specs_are_packaged() -> None:
    project = tomllib.loads(
        (Path(__file__).resolve().parents[3] / "pyproject.toml").read_text(
            encoding="utf-8"
        )
    )

    assert project["project"]["scripts"]["paper-agent"] == "paper_agent_next.cli:main"
    assert project["tool"]["setuptools"]["package-data"] == {
        "paper_agent": ["storage/migrations/*.sql"],
        "paper_agent_next": ["venue_specs/*.toml"],
    }


def test_new_package_does_not_import_old_package() -> None:
    package_root = _package_root()
    violations: list[str] = []

    for path in sorted(package_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if _is_old_package(alias.name):
                        violations.append(f"{path}:{node.lineno}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom) and _is_old_package(node.module):
                violations.append(f"{path}:{node.lineno}: from {node.module} import ...")

    assert not violations, "\n".join(violations)


def test_importing_models_does_not_load_old_package_in_isolated_subprocess() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    source_root = repo_root / "src"
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    paths = [str(source_root)]
    if existing_pythonpath:
        paths.extend(part for part in existing_pythonpath.split(os.pathsep) if part)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    code = """
import sys
import paper_agent_next.models

old_modules = sorted(
    name for name in sys.modules
    if name == 'paper_agent' or name.startswith('paper_agent.')
)
if old_modules:
    print('old package modules loaded: ' + ', '.join(old_modules), file=sys.stderr)
    raise SystemExit(1)
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout


def test_collector_does_not_import_concrete_adapters_output_loading_cli_or_old_package() -> None:
    collector_path = _package_root() / "collector.py"
    tree = ast.parse(collector_path.read_text(encoding="utf-8"), filename=str(collector_path))
    violations: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            prefix = "." * node.level
            modules = [prefix + (node.module or "")]
        else:
            continue

        for module in modules:
            relative = module.lstrip(".")
            if _is_old_package(module):
                violations.append(f"{collector_path}:{node.lineno}: import {module}")
            elif relative.startswith("adapters.") and relative != "adapters.base":
                violations.append(f"{collector_path}:{node.lineno}: concrete adapter import {module}")
            elif relative in {"output", "loading", "cli"}:
                violations.append(f"{collector_path}:{node.lineno}: forbidden layer import {module}")

    assert not violations, "\n".join(violations)


def test_cli_has_no_static_concrete_adapter_or_old_package_import() -> None:
    cli_path = _package_root() / "cli.py"
    tree = ast.parse(cli_path.read_text(encoding="utf-8"), filename=str(cli_path))
    violations: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [("." * node.level) + (node.module or "")]
        else:
            continue
        for module in modules:
            relative = module.lstrip(".")
            if _is_old_package(module):
                violations.append(f"{cli_path}:{node.lineno}: forbidden import {module}")
            elif relative.startswith("adapters.") and relative != "adapters.base":
                violations.append(f"{cli_path}:{node.lineno}: concrete adapter import {module}")

    assert not violations, "\n".join(violations)


def test_loading_has_no_static_concrete_adapter_or_enricher_import() -> None:
    loading_path = _package_root() / "loading.py"
    tree = ast.parse(loading_path.read_text(encoding="utf-8"), filename=str(loading_path))
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [("." * node.level) + (node.module or "")]
        else:
            continue
        for module in modules:
            relative = module.lstrip(".")
            if _is_old_package(module):
                violations.append(f"{loading_path}:{node.lineno}: forbidden import {module}")
            elif relative.startswith("adapters.") or relative.startswith("enrichers."):
                violations.append(f"{loading_path}:{node.lineno}: concrete import {module}")
    assert not violations, "\n".join(violations)


def test_main_module_only_imports_cli_main() -> None:
    main_path = _package_root() / "__main__.py"
    tree = ast.parse(main_path.read_text(encoding="utf-8"), filename=str(main_path))
    imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]

    assert len(imports) == 1
    import_node = imports[0]
    assert isinstance(import_node, ast.ImportFrom)
    assert import_node.level == 1
    assert import_node.module == "cli"
    assert [alias.name for alias in import_node.names] == ["main"]
