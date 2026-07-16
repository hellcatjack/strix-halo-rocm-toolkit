from __future__ import annotations

import re
import shlex
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class GpuPciInfo:
    pci_id: str | None
    driver: str | None
    raw: str


def parse_os_release(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        parsed = shlex.split(raw_value, comments=False, posix=True)
        values[key] = parsed[0] if parsed else ""
    return values


def parse_meminfo(text: str) -> dict[str, int]:
    values: dict[str, int] = {}
    for line in text.splitlines():
        match = re.fullmatch(r"([^:]+):\s+(\d+)\s+kB", line.strip())
        if match:
            values[match.group(1)] = int(match.group(2))
    return values


def parse_cmdline(text: str) -> dict[str, str | None]:
    parsed: dict[str, str | None] = {}
    for token in shlex.split(text.strip()):
        key, separator, value = token.partition("=")
        parsed[key] = value if separator else None
    return parsed


def parse_dmi_memory_bytes(text: str) -> int | None:
    total = 0
    matched = False
    multipliers = {"MB": 1024**2, "GB": 1024**3, "TB": 1024**4}
    for size, unit in re.findall(
        r"^\s*Size:\s+(\d+)\s+(MB|GB|TB)\s*$",
        text,
        re.MULTILINE,
    ):
        total += int(size) * multipliers[unit]
        matched = True
    return total if matched else None


def parse_lspci_gpu(text: str) -> GpuPciInfo:
    pci_match = re.search(r"\[([0-9a-fA-F]{4}:[0-9a-fA-F]{4})\]", text)
    driver_match = re.search(r"Kernel driver in use:\s+(\S+)", text)
    return GpuPciInfo(
        pci_id=pci_match.group(1).lower() if pci_match else None,
        driver=driver_match.group(1) if driver_match else None,
        raw=text,
    )


def parse_vram_mib(text: str) -> int | None:
    matches = re.findall(
        r"VRAM:\s*(\d+)M|(?:amdgpu:\s*)?(\d+)M(?:iB)?\s+of VRAM",
        text,
        re.IGNORECASE,
    )
    values = [first or second for first, second in matches]
    return int(values[-1]) if values else None


def parse_dpkg_packages(text: str) -> tuple[tuple[str, str], ...]:
    packages: list[tuple[str, str]] = []
    for line in text.splitlines():
        name, separator, version = line.partition("\t")
        if not separator or not name or not version:
            continue
        packages.append((name.split(":", 1)[0], version.strip()))
    return tuple(packages)


def parse_apt_policy_origin(text: str) -> str | None:
    match = re.search(r"https://repo\.radeon\.com/[^\s]+", text)
    return match.group(0).rstrip("/") if match else None


def parse_apt_candidate(text: str) -> str | None:
    match = re.search(r"^\s*Candidate:\s*(\S+)\s*$", text, re.MULTILINE)
    if match is None or match.group(1) == "(none)":
        return None
    return match.group(1)


def classify_docker_distribution(
    packages: Sequence[object],
    *,
    runtime_available: bool,
) -> "DockerDistribution":
    from amd_ai.host.models import DockerDistribution

    if not runtime_available:
        return DockerDistribution.MISSING
    names = {getattr(package, "name", None) for package in packages}
    docker_ce = bool({"docker-ce", "docker-ce-cli"}.intersection(names))
    docker_io = "docker.io" in names
    if docker_ce and docker_io:
        return DockerDistribution.MIXED
    if docker_ce:
        return DockerDistribution.DOCKER_CE
    if docker_io:
        return DockerDistribution.UBUNTU_DOCKER_IO
    return DockerDistribution.EXTERNAL
