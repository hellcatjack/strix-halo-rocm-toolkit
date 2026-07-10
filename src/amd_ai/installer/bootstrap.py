from __future__ import annotations

import argparse
import hashlib
import os
import re
import secrets
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from amd_ai import __version__


RUNTIME_PATHS = (
    "src/amd_ai",
    "profiles",
    "templates",
    "images",
    "bin",
    "pyproject.toml",
)
REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
VERSION_PATTERN = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+(?:[+.-][0-9A-Za-z.-]+)?")
IGNORED_NAMES = frozenset({"__pycache__"})
IGNORED_SUFFIXES = (".pyc", ".pyo")

LAUNCHER_CONTENT = """#!/usr/bin/env bash
set -euo pipefail
ROOT="${HOME}/.local/share/strix-halo-rocm-toolkit/current"
export AMD_AI_TOOLKIT_ROOT="${ROOT}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec python3.12 -m amd_ai.cli "$@"
"""


class BootstrapError(RuntimeError):
    pass


@dataclass(frozen=True)
class BootstrapResult:
    runtime: Path
    current: Path
    launcher: Path
    source_root: Path
    installer_source_revision: str


def install_user_runtime(
    *,
    source_root: Path,
    home: Path,
    version: str,
    installer_source_revision: str,
) -> BootstrapResult:
    source_root = _source_root(source_root)
    home = _home_path(home)
    if VERSION_PATTERN.fullmatch(version) is None:
        raise BootstrapError("installer version is invalid")
    if REVISION_PATTERN.fullmatch(installer_source_revision) is None:
        raise BootstrapError("installer source revision is invalid")
    _validate_source_payload(source_root)

    local = _ensure_control_directory(home / ".local", parents=True)
    share = _ensure_control_directory(local / "share")
    toolkit = _ensure_control_directory(
        share / "strix-halo-rocm-toolkit", private=True
    )
    releases = _ensure_control_directory(toolkit / "releases", private=True)
    bin_directory = _ensure_control_directory(local / "bin")

    release_name = f"{version}-{installer_source_revision[:12]}"
    runtime = releases / release_name
    staging = Path(
        tempfile.mkdtemp(prefix=".staging-", dir=releases)
    )
    staging.chmod(0o700)
    try:
        _copy_payload(source_root, staging)
        _fsync_tree(staging)
        if _lexists(runtime):
            if runtime.is_symlink() or not runtime.is_dir():
                raise BootstrapError(
                    f"runtime destination is not a regular directory: {runtime}"
                )
            _reject_tree_symlinks(runtime)
            if _payload_manifest(source_root) != _payload_manifest(runtime):
                raise BootstrapError(
                    f"existing versioned runtime does not match source: {runtime}"
                )
        else:
            try:
                os.replace(staging, runtime)
                _fsync_directory(releases)
            except OSError as error:
                raise BootstrapError(
                    f"cannot install versioned runtime: {error}"
                ) from error
    except BootstrapError:
        raise
    except (OSError, shutil.Error) as error:
        raise BootstrapError(f"cannot copy runtime payload: {error}") from error
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)

    current = toolkit / "current"
    _switch_current(current, f"releases/{release_name}")
    launcher = bin_directory / "strix-halo-rocm"
    _install_launcher(launcher)
    return BootstrapResult(
        runtime=runtime,
        current=current,
        launcher=launcher,
        source_root=source_root,
        installer_source_revision=installer_source_revision,
    )


def discover_installer_source_revision(source_root: Path) -> str:
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(source_root),
                "rev-parse",
                "--verify",
                "HEAD",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise BootstrapError(
            f"cannot determine installer source revision: {error}"
        ) from error
    revision = result.stdout.strip()
    if result.returncode != 0 or REVISION_PATTERN.fullmatch(revision) is None:
        evidence = result.stderr.strip() or result.stdout.strip() or "not a Git checkout"
        raise BootstrapError(
            f"cannot determine installer source revision: {evidence}"
        )
    return revision


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--source-root", type=Path, required=True)
    namespace, remaining = parser.parse_known_args(argv)
    source_root = _source_root(namespace.source_root)
    revision = discover_installer_source_revision(source_root)
    result = install_user_runtime(
        source_root=source_root,
        home=Path(os.environ.get("HOME", str(Path.home()))),
        version=__version__,
        installer_source_revision=revision,
    )
    os.environ["AMD_AI_TOOLKIT_ROOT"] = str(result.current)
    os.environ["AMD_AI_INSTALLER_SOURCE_REVISION"] = revision
    return _forward_install(
        ["--source-root", str(result.source_root), *remaining]
    )


def _forward_install(arguments: list[str]) -> int:
    from amd_ai.cli import main as cli_main

    return cli_main(["install", *arguments])


def _source_root(path: Path) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink():
        raise BootstrapError(f"source root must not be a symlink: {raw}")
    try:
        resolved = raw.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise BootstrapError(f"cannot resolve source root {raw}: {error}") from error
    if not resolved.is_dir():
        raise BootstrapError(f"source root is not a directory: {resolved}")
    return resolved


def _home_path(path: Path) -> Path:
    raw = Path(path).expanduser()
    if not raw.is_absolute():
        raw = Path.cwd() / raw
    if raw.is_symlink():
        raise BootstrapError(f"home must not be a symlink: {raw}")
    if _lexists(raw) and not raw.is_dir():
        raise BootstrapError(f"home is not a directory: {raw}")
    try:
        raw.mkdir(parents=True, mode=0o700, exist_ok=True)
        return raw.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise BootstrapError(f"cannot prepare home directory {raw}: {error}") from error


def _validate_source_payload(source_root: Path) -> None:
    for relative in RUNTIME_PATHS:
        path = source_root / relative
        if not _lexists(path):
            raise BootstrapError(f"runtime payload is missing: {relative}")
        if path.is_symlink():
            raise BootstrapError(f"runtime payload contains a symlink: {relative}")
        if relative == "pyproject.toml":
            if not path.is_file():
                raise BootstrapError(f"runtime payload is not a file: {relative}")
        elif not path.is_dir():
            raise BootstrapError(f"runtime payload is not a directory: {relative}")
        if path.is_dir():
            _reject_tree_symlinks(path)


def _reject_tree_symlinks(root: Path) -> None:
    for directory, names, files in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in [*names, *files]:
            candidate = base / name
            if candidate.is_symlink():
                raise BootstrapError(
                    f"runtime payload contains a symlink: {candidate}"
                )


def _ensure_control_directory(
    path: Path, *, parents: bool = False, private: bool = False
) -> Path:
    if path.is_symlink():
        raise BootstrapError(f"destination control directory is a symlink: {path}")
    if _lexists(path) and not path.is_dir():
        raise BootstrapError(
            f"destination control path is not a directory: {path}"
        )
    try:
        path.mkdir(parents=parents, mode=0o700 if private else 0o755, exist_ok=True)
        if private:
            path.chmod(0o700)
    except OSError as error:
        raise BootstrapError(
            f"cannot prepare destination control directory {path}: {error}"
        ) from error
    if path.is_symlink():
        raise BootstrapError(f"destination control directory is a symlink: {path}")
    return path


def _copy_payload(source_root: Path, destination: Path) -> None:
    for relative in RUNTIME_PATHS:
        source = source_root / relative
        target = destination / relative
        if source.is_dir():
            shutil.copytree(
                source,
                target,
                copy_function=_copy_file,
                ignore=_ignore_generated,
            )
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            _copy_file(source, target)


def _copy_file(source: str | Path, target: str | Path) -> str:
    copied = Path(shutil.copy2(source, target))
    descriptor = os.open(copied, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return str(copied)


def _ignore_generated(directory: str, names: list[str]) -> set[str]:
    del directory
    return {
        name
        for name in names
        if name in IGNORED_NAMES or name.endswith(IGNORED_SUFFIXES)
    }


def _switch_current(current: Path, relative_target: str) -> None:
    if _lexists(current) and not current.is_symlink():
        raise BootstrapError(f"current runtime path is not a symlink: {current}")
    temporary = current.with_name(
        f".current-{os.getpid()}-{secrets.token_hex(6)}"
    )
    try:
        os.symlink(relative_target, temporary, target_is_directory=True)
        os.replace(temporary, current)
        _fsync_directory(current.parent)
    except OSError as error:
        raise BootstrapError(f"cannot switch current runtime: {error}") from error
    finally:
        temporary.unlink(missing_ok=True)


def _install_launcher(path: Path) -> None:
    if path.is_symlink():
        raise BootstrapError(f"launcher destination is a symlink: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o755)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(LAUNCHER_CONTENT.encode("ascii"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except OSError as error:
        raise BootstrapError(f"cannot install launcher: {error}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _payload_manifest(root: Path) -> tuple[tuple[str, str, str], ...]:
    records: list[tuple[str, str, str]] = []
    for relative in RUNTIME_PATHS:
        base = root / relative
        if not base.exists():
            return ()
        if base.is_file():
            records.append(_file_record(root, base))
            continue
        records.append(("directory", relative, ""))
        for directory, names, files in os.walk(base):
            names[:] = sorted(
                name for name in names if name not in IGNORED_NAMES
            )
            files = sorted(
                name
                for name in files
                if not name.endswith(IGNORED_SUFFIXES)
            )
            current = Path(directory)
            if current != base:
                records.append(
                    ("directory", str(current.relative_to(root)), "")
                )
            for name in files:
                records.append(_file_record(root, current / name))
    return tuple(sorted(records))


def _file_record(root: Path, path: Path) -> tuple[str, str, str]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    executable = "x" if path.stat().st_mode & 0o111 else "-"
    return ("file", str(path.relative_to(root)), f"{digest}:{executable}")


def _fsync_tree(root: Path) -> None:
    for directory, names, files in os.walk(root, topdown=False):
        base = Path(directory)
        for name in files:
            descriptor = os.open(base / name, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        for name in names:
            _fsync_directory(base / name)
        _fsync_directory(base)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)


if __name__ == "__main__":
    raise SystemExit(main())
