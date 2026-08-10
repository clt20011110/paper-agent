"""Least-privilege command construction for third-party providers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import platform
import shutil
import sys
import sysconfig


class SandboxUnavailable(RuntimeError):
    """Raised when the host cannot enforce the plugin sandbox."""


@dataclass(frozen=True, slots=True)
class SandboxPolicy:
    """Filesystem and network boundary for one plugin invocation."""

    read_roots: tuple[Path, ...]
    work_root: Path


def interpreter_read_roots() -> tuple[Path, ...]:
    """Return interpreter-owned roots without exposing the caller's working tree."""

    paths = sysconfig.get_paths()
    library_name = sysconfig.get_config_var("LDLIBRARY")
    library_dir = sysconfig.get_config_var("LIBDIR")
    candidates = [
        Path(sys.executable).resolve().parent,
        *(Path(paths[name]).resolve() for name in ("stdlib",) if paths.get(name)),
        *(
            ((Path(str(library_dir)) / str(library_name)).resolve(),)
            if library_name and library_dir
            else ()
        ),
        *(
            tuple(
                root
                for root in map(Path, ("/lib", "/lib64", "/usr/lib", "/usr/lib64"))
                if root.exists()
            )
            if platform.system() == "Linux"
            else ()
        ),
    ]
    roots: list[Path] = []
    for candidate in candidates:
        if candidate.exists() and candidate not in roots:
            roots.append(candidate)
    return tuple(roots)


def build_sandbox_command(command: tuple[str, ...], policy: SandboxPolicy) -> tuple[str, ...]:
    """Wrap a command in an OS sandbox, or fail before executing the plugin."""

    system = platform.system()
    if system == "Darwin":
        executable = shutil.which("sandbox-exec")
        if executable is None:
            raise SandboxUnavailable("sandbox-exec is required to run third-party providers on macOS")
        resolved_command = (str(Path(command[0]).resolve()), *command[1:])
        profile = macos_profile(policy, executable_path=Path(resolved_command[0]))
        return (executable, "-p", profile, "--", *resolved_command)
    if system == "Linux":
        executable = shutil.which("bwrap")
        if executable is None:
            raise SandboxUnavailable("bubblewrap is required to run third-party providers on Linux")
        resolved_command = (str(Path(command[0]).resolve()), *command[1:])
        return linux_command(executable, resolved_command, policy)
    raise SandboxUnavailable(f"no supported third-party provider sandbox on {system}")


def macos_profile(policy: SandboxPolicy, *, executable_path: Path | None = None) -> str:
    """A sandbox-exec profile with no network and one writable directory."""

    rules = [
        "(version 1)",
        "(deny default)",
        '(import "system.sb")',
        "(deny network*)",
        f"(deny file-write* (require-not (subpath {json.dumps(str(policy.work_root))})))",
    ]
    if executable_path is not None:
        rules.append(f"(allow process-exec (literal {json.dumps(str(executable_path.absolute()))}))")
    resolved_roots = tuple(root.resolve() for root in policy.read_roots)
    ancestors = tuple(
        dict.fromkeys(
            ancestor
            for root in (*resolved_roots, policy.work_root.resolve())
            for ancestor in root.parents
        )
    )
    for ancestor in ancestors:
        rules.append(
            f"(allow file-read-metadata file-test-existence (literal {json.dumps(str(ancestor))}))"
        )
    for resolved in resolved_roots:
        rules.append(f"(allow file-read* file-test-existence (literal {json.dumps(str(resolved))}))")
        rules.append(f"(allow file-read* file-test-existence (subpath {json.dumps(str(resolved))}))")
    rules.append(
        f"(allow file-read* file-test-existence file-write* (literal {json.dumps(str(policy.work_root))}))"
    )
    rules.append(f"(allow file-read* file-test-existence file-write* (subpath {json.dumps(str(policy.work_root))}))")
    return "\n".join(rules)


def linux_command(executable: str, command: tuple[str, ...], policy: SandboxPolicy) -> tuple[str, ...]:
    """Build a bubblewrap command with an empty mount namespace and no network."""

    result: list[str] = [
        executable,
        "--die-with-parent",
        "--new-session",
        "--unshare-all",
        "--cap-drop",
        "ALL",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
    ]
    for root in policy.read_roots:
        resolved = root.resolve()
        result.extend(("--ro-bind", str(resolved), str(resolved)))
    result.extend(("--bind", str(policy.work_root), str(policy.work_root), "--chdir", str(policy.work_root), "--"))
    result.extend(command)
    return tuple(result)
