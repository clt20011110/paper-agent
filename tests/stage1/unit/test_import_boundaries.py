"""Import-boundary tests for the cutover Stage 1 package."""

import ast
import os
from pathlib import Path
import subprocess
import sys
import tomllib


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _package_root() -> Path:
    return _repo_root() / "src" / "paper_agent"


def _legacy_package_name() -> str:
    return "paper_agent" + "_next"


def _legacy_tests_dir() -> str:
    return "stage1" + "_next"


def _is_legacy_package(module: str | None) -> bool:
    legacy_package = _legacy_package_name()
    return module == legacy_package or bool(
        module and module.startswith(legacy_package + ".")
    )


def _import_modules(tree: ast.AST) -> list[str]:
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.append("." * node.level + (node.module or ""))
    return modules


def test_cutover_roots_are_canonical_and_legacy_roots_are_absent() -> None:
    assert _package_root().is_dir()
    assert not (_repo_root() / "src" / _legacy_package_name()).exists()
    assert not (_repo_root() / "tests" / _legacy_tests_dir()).exists()


def test_stage1_console_entry_point_and_venue_specs_are_packaged() -> None:
    project = tomllib.loads(
        (_repo_root() / "pyproject.toml").read_text(encoding="utf-8")
    )

    assert project["project"]["scripts"]["paper-agent"] == "paper_agent.cli:main"
    assert project["tool"]["setuptools"]["package-data"] == {
        "paper_agent": ["venue_specs/*.toml"],
    }


def test_formal_package_does_not_import_legacy_package() -> None:
    violations: list[str] = []

    for path in sorted(_package_root().rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for module in _import_modules(tree):
            if _is_legacy_package(module):
                violations.append(f"{path}: import {module}")

    assert not violations, "\n".join(violations)


def test_importing_models_does_not_load_legacy_package_in_isolated_subprocess() -> None:
    source_root = _repo_root() / "src"
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    paths = [str(source_root)]
    if existing_pythonpath:
        paths.extend(part for part in existing_pythonpath.split(os.pathsep) if part)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    legacy_package = _legacy_package_name()
    code = f"""
import sys
import paper_agent.models

legacy_modules = sorted(
    name for name in sys.modules
    if name == {legacy_package!r} or name.startswith({(legacy_package + '.')!r})
)
if legacy_modules:
    print('legacy package modules loaded: ' + ', '.join(legacy_modules), file=sys.stderr)
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


def test_collector_does_not_import_concrete_adapters_output_loading_cli_or_legacy_package() -> None:
    collector_path = _package_root() / "collector.py"
    tree = ast.parse(collector_path.read_text(encoding="utf-8"), filename=str(collector_path))
    violations: list[str] = []

    for module in _import_modules(tree):
        relative = module.lstrip(".")
        if _is_legacy_package(module):
            violations.append(f"{collector_path}: import {module}")
        elif relative.startswith("adapters.") and relative != "adapters.base":
            violations.append(f"{collector_path}: concrete adapter import {module}")
        elif relative in {"output", "loading", "cli"}:
            violations.append(f"{collector_path}: forbidden layer import {module}")

    assert not violations, "\n".join(violations)


def test_cli_has_no_static_concrete_adapter_or_legacy_package_import() -> None:
    cli_path = _package_root() / "cli.py"
    tree = ast.parse(cli_path.read_text(encoding="utf-8"), filename=str(cli_path))
    violations: list[str] = []

    for module in _import_modules(tree):
        relative = module.lstrip(".")
        if _is_legacy_package(module):
            violations.append(f"{cli_path}: forbidden import {module}")
        elif relative.startswith("adapters."):
            violations.append(f"{cli_path}: concrete adapter import {module}")

    assert not violations, "\n".join(violations)


def test_loading_has_only_trusted_dynamic_stage1_imports() -> None:
    loading_path = _package_root() / "loading.py"
    source = loading_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(loading_path))
    violations: list[str] = []

    for module in _import_modules(tree):
        relative = module.lstrip(".")
        if _is_legacy_package(module):
            violations.append(f"{loading_path}: forbidden import {module}")
        elif relative.startswith("adapters.") or relative.startswith("enrichers."):
            violations.append(f"{loading_path}: concrete import {module}")

    assert not violations, "\n".join(violations)
    assert "importlib.import_module(f\"paper_agent.{module_name}\")" in source
    assert _legacy_package_name() not in source


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
