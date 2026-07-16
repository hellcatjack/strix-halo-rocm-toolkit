from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from amd_ai.host.models import AptSourceFile, HostSnapshot, PlannedAction, PreparePlan
from amd_ai.host.parsers import parse_apt_candidate
from amd_ai.host.prepare import old_rocm_source_paths
from amd_ai.runner import CommandError, CommandResult, Runner


DOCKER_GPG_FINGERPRINT = "9DC858229FC7DD38854AE2D88D81803C0EBFCD88"


class ApplyRefused(RuntimeError):
    pass


class ApplyError(RuntimeError):
    pass


@dataclass(frozen=True)
class ApplyResult:
    backup_path: Path
    executed_codes: tuple[str, ...]
    skipped_codes: tuple[str, ...]


BACKUP_FILES = (
    "etc/os-release",
    "etc/default/grub",
    "etc/modprobe.d/ttm.conf",
    "etc/apt/sources.list",
    "proc/cmdline",
    "sys/module/ttm/parameters/pages_limit",
    "sys/module/amdttm/parameters/pages_limit",
)
BACKUP_GLOBS = (
    "etc/apt/sources.list.d/*",
    "etc/apt/preferences.d/*",
)
BACKUP_COMMANDS = (
    (
        "packages",
        ("dpkg-query", "-W", "-f=${binary:Package}\\t${Version}\\n"),
    ),
    ("dkms", ("dkms", "status")),
    ("kernel", ("uname", "-a")),
    ("gpu", ("lspci", "-Dnnk", "-d", "1002:1586")),
    ("docker", ("docker", "version")),
)


def backup_host_state(
    *,
    snapshot: HostSnapshot,
    runner: Runner,
    destination: Path = Path("/var/backups/amd-ai"),
    root: Path = Path("/"),
    timestamp: str | None = None,
) -> Path:
    timestamp = timestamp or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    destination.mkdir(parents=True, exist_ok=True)
    os.chmod(destination, 0o700)
    backup = destination / timestamp
    if backup.exists():
        raise ApplyError(f"backup destination already exists: {backup}")
    backup.mkdir(mode=0o700)
    os.chmod(backup, 0o700)

    copied: list[str] = []
    candidates = [root / relative for relative in BACKUP_FILES]
    for pattern in BACKUP_GLOBS:
        candidates.extend(sorted(root.glob(pattern)))
    for source in candidates:
        if not source.is_file():
            continue
        relative = source.relative_to(root)
        target = backup / relative
        _copy_private(source, target, backup)
        copied.append(str(relative))

    command_records: list[dict[str, object]] = []
    for name, argv in BACKUP_COMMANDS:
        result = _run_optional(runner, argv)
        command_records.append(
            {
                "name": name,
                "argv": list(argv),
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        )

    manifest = {
        "schema_version": 1,
        "created_at": timestamp,
        "host": {
            "os_id": snapshot.os_id,
            "os_version": snapshot.os_version,
            "architecture": snapshot.architecture,
            "kernel": snapshot.kernel,
            "gpu_pci_id": snapshot.gpu.pci_id,
            "gpu_driver": snapshot.gpu.driver,
        },
        "copied_files": sorted(set(copied)),
        "commands": command_records,
    }
    _write_text_atomic(
        backup / "manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        mode=0o600,
    )
    return backup


def execute_plan(
    plan: PreparePlan,
    runner: Runner,
    *,
    effective_uid: int,
    confirmed: bool,
    reboot: bool = False,
    snapshot: HostSnapshot | None = None,
    root: Path = Path("/"),
    backup_destination: Path | None = None,
    timestamp: str | None = None,
) -> ApplyResult:
    if effective_uid != 0:
        raise ApplyRefused("host preparation apply requires root")
    if not confirmed:
        raise ApplyRefused("host preparation requires exact confirmation")
    if not plan.supported:
        raise ApplyRefused("the preparation plan is not supported")

    backup_path: Path | None = None
    executed: list[str] = []
    skipped: list[str] = []
    for action in plan.actions:
        if action.code == "HOST.REBOOT" and not reboot:
            skipped.append(action.code)
            continue

        if action.code == "BACKUP.SNAPSHOT":
            if backup_path is not None:
                raise ApplyError("the plan contains more than one backup action")
            if snapshot is None:
                raise ApplyError("a host snapshot is required for backup")
            destination = backup_destination or _rooted(root, "/var/backups/amd-ai")
            backup_path = backup_host_state(
                snapshot=snapshot,
                runner=runner,
                destination=destination,
                root=root,
                timestamp=timestamp,
            )
            executed.append(action.code)
            continue

        if backup_path is None or not (backup_path / "manifest.json").is_file():
            raise ApplyError("no successful backup exists before the first host change")

        if action.code == "APT.DISABLE_OLD_ROCM_SOURCES":
            _disable_old_sources(action, root)
        elif action.code == "APT.INSTALL_OEM_617":
            _install_oem_617(runner)
        elif action.code == "DOCKER.INSTALL_IF_MISSING":
            _install_docker(runner, root)
        elif action.code == "DOCKER.INSTALL_BUILDX_PLUGIN":
            _install_docker_buildx_plugin(runner, root)
        elif action.code == "DOCKER.INSTALL_UBUNTU_BUILDX":
            _install_ubuntu_buildx(runner)
        elif action.argv:
            _run_required(runner, action.argv, input_text=action.input_text)
        else:
            raise ApplyError(f"unknown internal action: {action.code}")
        executed.append(action.code)

    if backup_path is None:
        raise ApplyError("the plan completed without a successful backup")
    return ApplyResult(backup_path, tuple(executed), tuple(skipped))


def parse_gpg_fingerprints(colon_listing: str) -> tuple[str, ...]:
    fingerprints: list[str] = []
    for line in colon_listing.splitlines():
        fields = line.split(":")
        if len(fields) > 9 and fields[0] == "fpr":
            fingerprint = fields[9].upper()
            if re.fullmatch(r"[0-9A-F]{40}", fingerprint):
                fingerprints.append(fingerprint)
    return tuple(fingerprints)


def _disable_old_sources(action: PlannedAction, root: Path) -> None:
    try:
        raw_paths = json.loads(action.input_text or "")
    except json.JSONDecodeError as error:
        raise ApplyError("invalid old-source action metadata") from error
    if not isinstance(raw_paths, list) or not all(
        isinstance(path, str) for path in raw_paths
    ):
        raise ApplyError("invalid old-source action path list")

    for raw_path in raw_paths:
        relative = Path(raw_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ApplyError(f"unsafe APT source path: {raw_path}")
        if not (
            raw_path == "etc/apt/sources.list"
            or raw_path.startswith("etc/apt/sources.list.d/")
        ):
            raise ApplyError(f"path is not an APT source: {raw_path}")
        source = root / relative
        disabled = source.with_name(source.name + ".amd-ai-disabled")
        if not source.exists() and disabled.is_file():
            continue
        if not source.is_file():
            raise ApplyError(f"planned APT source is missing: {source}")
        if disabled.exists():
            raise ApplyError(f"disabled APT source already exists: {disabled}")
        record = AptSourceFile(raw_path, source.read_text(encoding="utf-8"))
        if old_rocm_source_paths((record,)) != (raw_path,):
            raise ApplyError(f"APT source no longer contains a ROCm 6.4 URL: {source}")
        source.rename(disabled)


def _install_oem_617(runner: Runner) -> None:
    _run_required(runner, ("apt-get", "update"))
    policy = _run_required(
        runner,
        ("apt-cache", "policy", "linux-oem-6.17"),
    )
    candidate = parse_apt_candidate(policy.stdout)
    if candidate is None or not candidate.startswith("6.17."):
        rendered = candidate or "none"
        raise ApplyError(
            "linux-oem-6.17 no longer has a valid 6.17 candidate "
            f"after apt-get update (candidate: {rendered})"
        )
    _run_required(
        runner,
        (
            "apt-get",
            "install",
            "-y",
            "linux-oem-6.17",
            "linux-firmware",
        ),
    )


def _install_docker(runner: Runner, root: Path) -> None:
    _configure_docker_repository(runner, root)
    _run_required(runner, ("apt-get", "update"))
    _run_required(
        runner,
        (
            "apt-get",
            "install",
            "-y",
            "docker-ce",
            "docker-ce-cli",
            "containerd.io",
            "docker-buildx-plugin",
            "docker-compose-plugin",
        ),
    )
    _run_required(runner, ("systemctl", "enable", "--now", "docker"))


def _install_docker_buildx_plugin(runner: Runner, root: Path) -> None:
    _configure_docker_repository(runner, root)
    _run_required(runner, ("apt-get", "update"))
    _run_required(runner, ("apt-get", "install", "-y", "docker-buildx-plugin"))
    _verify_docker_runtime_and_buildx(runner)


def _install_ubuntu_buildx(runner: Runner) -> None:
    _run_required(runner, ("apt-get", "update"))
    _run_required(runner, ("apt-get", "install", "-y", "docker-buildx"))
    _verify_docker_runtime_and_buildx(runner)


def _configure_docker_repository(runner: Runner, root: Path) -> None:
    keyring = _rooted(root, "/etc/apt/keyrings/docker.asc")
    temporary_key = keyring.with_name("docker.asc.amd-ai.tmp")
    keyring.parent.mkdir(parents=True, exist_ok=True)
    temporary_key.unlink(missing_ok=True)
    _run_required(
        runner,
        (
            "curl",
            "--fail",
            "--silent",
            "--show-error",
            "--location",
            "https://download.docker.com/linux/ubuntu/gpg",
            "--output",
            str(temporary_key),
        ),
    )
    listing = _run_required(
        runner,
        ("gpg", "--batch", "--show-keys", "--with-colons", str(temporary_key)),
    ).stdout
    if DOCKER_GPG_FINGERPRINT not in parse_gpg_fingerprints(listing):
        temporary_key.unlink(missing_ok=True)
        raise ApplyError("Docker signing-key fingerprint verification failed")
    if not temporary_key.is_file():
        raise ApplyError("Docker signing key download did not create a file")
    os.replace(temporary_key, keyring)
    os.chmod(keyring, 0o644)

    source = _rooted(root, "/etc/apt/sources.list.d/docker.sources")
    _write_text_atomic(
        source,
        "Types: deb\n"
        "URIs: https://download.docker.com/linux/ubuntu\n"
        "Suites: noble\n"
        "Components: stable\n"
        "Architectures: amd64\n"
        "Signed-By: /etc/apt/keyrings/docker.asc\n",
        mode=0o644,
    )


def _verify_docker_runtime_and_buildx(runner: Runner) -> None:
    probes = (
        ("Docker daemon", ("docker", "version", "--format", "{{.Server.Version}}")),
        ("Docker Buildx", ("docker", "buildx", "version")),
    )
    for label, argv in probes:
        result = _run_optional(runner, argv)
        if result.returncode == 0 and result.stdout.strip():
            continue
        evidence = (result.stderr or result.stdout).strip() or "no command output"
        raise ApplyError(f"{label} verification failed after Buildx repair: {evidence}")


def _run_required(
    runner: Runner,
    argv: tuple[str, ...],
    *,
    input_text: str | None = None,
) -> CommandResult:
    try:
        return runner.run(list(argv), check=True, input_text=input_text)
    except CommandError as error:
        evidence = (error.result.stderr or error.result.stdout).strip()
        raise ApplyError(
            f"command failed ({error.result.returncode}): {' '.join(argv)}: {evidence}"
        ) from error
    except OSError as error:
        raise ApplyError(f"could not execute {' '.join(argv)}: {error}") from error


def _run_optional(runner: Runner, argv: tuple[str, ...]) -> CommandResult:
    try:
        return runner.run(list(argv), check=False)
    except OSError as error:
        return CommandResult(argv, 127, "", str(error))


def _copy_private(source: Path, target: Path, backup_root: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    current = target.parent
    while True:
        os.chmod(current, 0o700)
        if current == backup_root:
            break
        if backup_root not in current.parents:
            raise ApplyError(f"backup target escaped its root: {target}")
        current = current.parent
    with source.open("rb") as input_stream, target.open("wb") as output_stream:
        shutil.copyfileobj(input_stream, output_stream)
    os.chmod(target, 0o600)


def _write_text_atomic(path: Path, content: str, *, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _rooted(root: Path, absolute_path: str) -> Path:
    return root / absolute_path.removeprefix("/")
