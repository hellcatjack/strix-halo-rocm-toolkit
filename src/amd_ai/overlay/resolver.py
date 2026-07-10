from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import stat
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import unquote, urlsplit

from amd_ai.overlay.models import (
    PROTECTED_DISTRIBUTIONS,
    ProtectedProfile,
    canonicalize_protected_name,
)
from amd_ai.overlay.packaging_compat import (
    InvalidWheelFilename,
    Requirement,
    Version,
    canonicalize_name,
    parse_wheel_filename,
)
from amd_ai.overlay.requirements import InspectedRequirements
from amd_ai.runner import CommandResult


class ResolverError(RuntimeError):
    pass


@dataclass(frozen=True)
class ReportItem:
    name: str
    version: str
    url: str
    sha256: str
    requested: bool


@dataclass(frozen=True)
class WheelArtifact:
    name: str
    version: str
    sha256: str
    path: Path
    requested: bool


class ProcessRunner(Protocol):
    def run(
        self,
        args: list[str],
        *,
        environment: dict[str, str],
        cwd: Path | None = None,
    ) -> CommandResult:
        pass


class SubprocessProcessRunner:
    def run(
        self,
        args: list[str],
        *,
        environment: dict[str, str],
        cwd: Path | None = None,
    ) -> CommandResult:
        completed = subprocess.run(
            args,
            check=False,
            cwd=cwd,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return CommandResult(
            args=tuple(args),
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )


def resolver_argv(
    *,
    input_path: Path,
    constraints_path: Path,
    report_path: Path,
    resolver_options: tuple[str, ...],
) -> tuple[str, ...]:
    return (
        "/opt/venv/bin/python",
        "-m",
        "pip",
        "install",
        "--dry-run",
        "--disable-pip-version-check",
        "--report",
        str(report_path),
        "--constraint",
        str(constraints_path),
        *resolver_options,
        "--requirement",
        str(input_path),
    )


def render_constraints(profile: ProtectedProfile) -> str:
    return "".join(
        f"{component.name}=={component.version}\n"
        for component in sorted(profile.components, key=lambda value: value.name)
    )


def parse_pip_report(text: str) -> tuple[ReportItem, ...]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise ResolverError(f"cannot parse pip report: {error}") from error
    if not isinstance(payload, dict) or payload.get("version") != "1":
        raise ResolverError("unsupported pip report schema")
    if not isinstance(payload.get("pip_version"), str):
        raise ResolverError("pip report has no pip version")
    records = payload.get("install")
    if not isinstance(records, list):
        raise ResolverError("pip report install records are invalid")

    items: list[ReportItem] = []
    names: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ResolverError("pip report install record is invalid")
        metadata = record.get("metadata")
        download = record.get("download_info")
        requested = record.get("requested")
        if (
            not isinstance(metadata, dict)
            or not isinstance(download, dict)
            or not isinstance(requested, bool)
        ):
            raise ResolverError("pip report artifact metadata is invalid")
        raw_name = metadata.get("name")
        raw_version = metadata.get("version")
        if not isinstance(raw_name, str) or not raw_name:
            raise ResolverError("pip report artifact name is invalid")
        if not isinstance(raw_version, str) or not raw_version:
            raise ResolverError("pip report artifact version is invalid")
        try:
            Version(raw_version)
        except Exception as error:
            raise ResolverError(
                f"pip report artifact version is invalid: {raw_version}"
            ) from error
        name = canonicalize_name(raw_name)
        protected = canonicalize_protected_name(name)
        if protected in PROTECTED_DISTRIBUTIONS:
            raise ResolverError(
                f"pip resolver returned protected distribution: {protected}"
            )
        if name in names:
            raise ResolverError(f"pip report contains duplicate distribution: {name}")

        url = download.get("url")
        archive = download.get("archive_info")
        if not isinstance(url, str) or not isinstance(archive, dict):
            raise ResolverError("pip report download identity is invalid")
        parsed_url = urlsplit(url)
        if (
            parsed_url.scheme not in {"https", "file"}
            or parsed_url.username is not None
            or parsed_url.password is not None
            or (parsed_url.scheme == "https" and not parsed_url.hostname)
            or (parsed_url.scheme == "file" and parsed_url.hostname not in {None, "", "localhost"})
        ):
            raise ResolverError("pip report artifact URL is unsafe")
        hashes = archive.get("hashes")
        sha256 = hashes.get("sha256") if isinstance(hashes, dict) else None
        if (
            not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
        ):
            raise ResolverError("pip report artifact has no valid SHA-256")
        names.add(name)
        items.append(
            ReportItem(
                name=name,
                version=raw_version,
                url=url,
                sha256=sha256,
                requested=requested,
            )
        )
    return tuple(items)


def resolve_report(
    *,
    input_lines: tuple[str, ...],
    profile: ProtectedProfile,
    transaction_dir: Path,
    resolver_options: tuple[str, ...],
    runner: ProcessRunner,
    base_environment: Mapping[str, str] | None = None,
) -> tuple[ReportItem, ...]:
    _ensure_directory(transaction_dir)
    input_path = transaction_dir / "resolver.requirements.in"
    constraints_path = transaction_dir / "protected-constraints.txt"
    report_path = transaction_dir / "pip-report.json"
    cache_dir = transaction_dir / "pip-cache"
    _write_private_text(input_path, "\n".join(input_lines) + "\n")
    _write_private_text(constraints_path, render_constraints(profile))
    _ensure_directory(cache_dir)
    environment = dict(os.environ if base_environment is None else base_environment)
    environment.update(
        {
            "PYTHONPATH": "/opt/amd-ai/src",
            "PYTHONNOUSERSITE": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_CACHE_DIR": str(cache_dir),
        }
    )
    argv = resolver_argv(
        input_path=input_path,
        constraints_path=constraints_path,
        report_path=report_path,
        resolver_options=resolver_options,
    )
    try:
        result = runner.run(
            list(argv),
            environment=environment,
            cwd=Path("/workspace"),
        )
        if result.returncode != 0:
            raise ResolverError(
                "pip resolution failed: "
                + (result.stderr.strip() or result.stdout.strip() or "no output")
            )
        try:
            report_text = report_path.read_text(encoding="utf-8")
        except OSError as error:
            raise ResolverError(f"cannot read pip resolution report: {error}") from error
        return parse_pip_report(report_text)
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


def materialize_report(
    items: tuple[ReportItem, ...],
    *,
    artifacts_root: Path,
    transaction_dir: Path,
    runner: ProcessRunner,
    base_environment: Mapping[str, str] | None = None,
) -> tuple[WheelArtifact, ...]:
    downloads_root = transaction_dir / "downloads"
    cache_dir = transaction_dir / "pip-cache"
    _ensure_directory(transaction_dir)
    _ensure_directory(downloads_root)
    _ensure_directory(cache_dir)
    environment = dict(os.environ if base_environment is None else base_environment)
    environment.update(
        {
            "PYTHONPATH": "/opt/amd-ai/src",
            "PYTHONNOUSERSITE": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_CACHE_DIR": str(cache_dir),
        }
    )
    artifacts: list[WheelArtifact] = []
    try:
        for index, item in enumerate(items):
            parsed = urlsplit(item.url)
            if parsed.scheme == "file":
                source = Path(unquote(parsed.path)).resolve(strict=True)
            else:
                item_dir = downloads_root / f"{index:04d}-{item.name}"
                _ensure_directory(item_dir)
                requirement_path = item_dir / "artifact.requirement.txt"
                _write_private_text(
                    requirement_path,
                    f"{item.name} @ {item.url} \\"
                    f"\n    --hash=sha256:{item.sha256}\n",
                )
                argv = _download_argv(requirement_path, item_dir)
                result = runner.run(
                    list(argv),
                    environment=environment,
                    cwd=Path("/workspace"),
                )
                if result.returncode != 0:
                    raise ResolverError(
                        f"artifact download failed for {item.name}: "
                        + (
                            result.stderr.strip()
                            or result.stdout.strip()
                            or "no output"
                        )
                    )
                candidates = tuple(
                    path
                    for path in item_dir.iterdir()
                    if path.name != requirement_path.name and path.is_file()
                )
                if len(candidates) != 1:
                    raise ResolverError(
                        f"artifact download for {item.name} produced {len(candidates)} files"
                    )
                source = candidates[0]
            if _hash_file(source) != item.sha256:
                raise ResolverError(
                    f"downloaded artifact hash changed for {item.name}"
                )
            wheel = source
            if source.suffix.lower() != ".whl":
                wheel_dir = downloads_root / f"{index:04d}-{item.name}-wheel"
                _ensure_directory(wheel_dir)
                result = runner.run(
                    list(_wheel_argv(source, wheel_dir)),
                    environment=environment,
                    cwd=Path("/workspace"),
                )
                if result.returncode != 0:
                    raise ResolverError(
                        f"wheel build failed for {item.name}: "
                        + (
                            result.stderr.strip()
                            or result.stdout.strip()
                            or "no output"
                        )
                    )
                wheels = tuple(wheel_dir.glob("*.whl"))
                if len(wheels) != 1:
                    raise ResolverError(
                        f"wheel build for {item.name} produced {len(wheels)} wheels"
                    )
                wheel = wheels[0]
            artifacts.append(
                store_wheel(
                    wheel,
                    artifacts_root=artifacts_root,
                    expected_name=item.name,
                    expected_version=item.version,
                    requested=item.requested,
                )
            )
        return tuple(artifacts)
    finally:
        shutil.rmtree(downloads_root, ignore_errors=True)
        shutil.rmtree(cache_dir, ignore_errors=True)


def prepare_direct_wheels(
    inspected: InspectedRequirements,
    *,
    transaction_dir: Path,
    runner: ProcessRunner,
    base_environment: Mapping[str, str] | None = None,
) -> tuple[Path, ...]:
    output_root = transaction_dir / "direct-wheels"
    cache_dir = transaction_dir / "pip-cache"
    _ensure_directory(transaction_dir)
    _ensure_directory(output_root)
    _ensure_directory(cache_dir)
    environment = dict(os.environ if base_environment is None else base_environment)
    environment.update(
        {
            "PYTHONPATH": "/opt/amd-ai/src",
            "PYTHONNOUSERSITE": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_CACHE_DIR": str(cache_dir),
        }
    )
    wheels: list[Path] = []
    try:
        inputs: list[tuple[str, str | Path, str | None]] = []
        for local in inspected.local_inputs:
            if local.is_file() and local.suffix.lower() == ".whl":
                _wheel_identity(local)
                wheels.append(local)
            else:
                inputs.append(("local", local, None))
        for vcs in inspected.vcs_inputs:
            requirement = Requirement(vcs)
            inputs.append(("vcs", vcs, canonicalize_name(requirement.name)))

        for index, (kind, source, expected_name) in enumerate(inputs):
            destination = output_root / f"{index:04d}-{kind}"
            _ensure_directory(destination)
            result = runner.run(
                list(_wheel_argv_value(str(source), destination)),
                environment=environment,
                cwd=Path("/workspace"),
            )
            if result.returncode != 0:
                raise ResolverError(
                    f"wheel build failed for {kind} input: "
                    + (
                        result.stderr.strip()
                        or result.stdout.strip()
                        or "no output"
                    )
                )
            built = tuple(destination.glob("*.whl"))
            if len(built) != 1:
                raise ResolverError(
                    f"wheel build for {kind} input produced {len(built)} wheels"
                )
            name, _ = _wheel_identity(built[0])
            if expected_name is not None and name != expected_name:
                raise ResolverError(
                    f"VCS wheel identity mismatch: expected {expected_name}, got {name}"
                )
            wheels.append(built[0])
        return tuple(wheels)
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


def direct_wheel_requirements(wheels: tuple[Path, ...]) -> tuple[str, ...]:
    requirements: list[str] = []
    for wheel in wheels:
        name, _ = _wheel_identity(wheel)
        requirements.append(f"{name} @ {wheel.resolve(strict=True).as_uri()}")
    return tuple(requirements)


def resolve_and_materialize(
    inspected: InspectedRequirements,
    *,
    profile: ProtectedProfile,
    artifacts_root: Path,
    transaction_dir: Path,
    resolver_options: tuple[str, ...],
    runner: ProcessRunner,
    base_environment: Mapping[str, str] | None = None,
) -> tuple[WheelArtifact, ...]:
    direct_wheels = prepare_direct_wheels(
        inspected,
        transaction_dir=transaction_dir,
        runner=runner,
        base_environment=base_environment,
    )
    input_lines = (
        *inspected.resolver_inputs,
        *direct_wheel_requirements(direct_wheels),
    )
    if not input_lines:
        return ()
    try:
        report = resolve_report(
            input_lines=tuple(input_lines),
            profile=profile,
            transaction_dir=transaction_dir,
            resolver_options=resolver_options,
            runner=runner,
            base_environment=base_environment,
        )
        return materialize_report(
            report,
            artifacts_root=artifacts_root,
            transaction_dir=transaction_dir,
            runner=runner,
            base_environment=base_environment,
        )
    finally:
        shutil.rmtree(transaction_dir / "direct-wheels", ignore_errors=True)


def store_wheel(
    source: Path,
    *,
    artifacts_root: Path,
    expected_name: str,
    expected_version: str,
    requested: bool,
) -> WheelArtifact:
    if source.is_symlink():
        raise ResolverError(f"wheel artifact must not be a symlink: {source}")
    source = source.resolve(strict=True)
    if not source.is_file():
        raise ResolverError(f"wheel artifact is not a regular file: {source}")
    try:
        parsed_name, parsed_version, _, _ = parse_wheel_filename(source.name)
    except InvalidWheelFilename as error:
        raise ResolverError(f"invalid wheel filename: {source.name}") from error
    name = canonicalize_name(str(parsed_name))
    version = str(parsed_version)
    if (
        name != canonicalize_name(expected_name)
        or version != str(Version(expected_version))
    ):
        raise ResolverError(
            "wheel identity does not match resolver metadata: "
            f"wheel={name}=={version}, expected={expected_name}=={expected_version}"
        )
    protected = canonicalize_protected_name(name)
    if protected in PROTECTED_DISTRIBUTIONS:
        raise ResolverError(f"wheel artifact is protected: {protected}")

    digest = _hash_file(source)
    destination_dir = artifacts_root / digest
    destination = destination_dir / source.name
    _ensure_directory(artifacts_root)
    _ensure_directory(destination_dir)
    if destination.exists():
        if (
            destination.is_symlink()
            or not destination.is_file()
            or _hash_file(destination) != digest
        ):
            raise ResolverError(
                f"existing wheel artifact is invalid: {destination}"
            )
    else:
        temporary = destination_dir / (
            f".{source.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
        )
        try:
            with source.open("rb") as input_stream, temporary.open("xb") as output_stream:
                shutil.copyfileobj(input_stream, output_stream, 1024 * 1024)
                output_stream.flush()
                os.fsync(output_stream.fileno())
            temporary.chmod(0o444)
            os.replace(temporary, destination)
            _fsync_directory(destination_dir)
        finally:
            temporary.unlink(missing_ok=True)
    destination.chmod(0o444)
    return WheelArtifact(
        name=name,
        version=version,
        sha256=digest,
        path=destination,
        requested=requested,
    )


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _download_argv(
    requirement_path: Path, destination: Path
) -> tuple[str, ...]:
    return (
        "/opt/venv/bin/python",
        "-m",
        "pip",
        "download",
        "--disable-pip-version-check",
        "--no-deps",
        "--require-hashes",
        "--dest",
        str(destination),
        "--requirement",
        str(requirement_path),
    )


def _wheel_argv(source: Path, destination: Path) -> tuple[str, ...]:
    return _wheel_argv_value(str(source), destination)


def _wheel_argv_value(source: str, destination: Path) -> tuple[str, ...]:
    return (
        "/opt/venv/bin/python",
        "-m",
        "pip",
        "wheel",
        "--disable-pip-version-check",
        "--no-deps",
        "--wheel-dir",
        str(destination),
        source,
    )


def _wheel_identity(path: Path) -> tuple[str, str]:
    try:
        parsed_name, parsed_version, _, _ = parse_wheel_filename(path.name)
    except InvalidWheelFilename as error:
        raise ResolverError(f"invalid wheel filename: {path.name}") from error
    name = canonicalize_name(str(parsed_name))
    protected = canonicalize_protected_name(name)
    if protected in PROTECTED_DISTRIBUTIONS:
        raise ResolverError(f"wheel artifact is protected: {protected}")
    return name, str(parsed_version)


def _write_private_text(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o600)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _ensure_directory(path: Path) -> None:
    if path.is_symlink():
        raise ResolverError(f"artifact directory must not be a symlink: {path}")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    metadata = path.stat()
    if not stat.S_ISDIR(metadata.st_mode):
        raise ResolverError(f"artifact path is not a directory: {path}")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
