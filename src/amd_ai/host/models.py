from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from amd_ai.host.parsers import GpuPciInfo


class DockerDistribution(StrEnum):
    DOCKER_CE = "docker-ce"
    UBUNTU_DOCKER_IO = "ubuntu-docker-io"
    MIXED = "mixed"
    EXTERNAL = "external"
    MISSING = "missing"


@dataclass(frozen=True)
class InstalledPackage:
    name: str
    version: str
    origin: str | None = None


@dataclass(frozen=True)
class AptSourceFile:
    path: str
    content: str


@dataclass(frozen=True)
class HostSnapshot:
    os_id: str
    os_version: str
    architecture: str
    kernel: str
    gpu: GpuPciInfo
    mem_total_kib: int
    swap_total_kib: int
    page_size: int
    kernel_args: dict[str, str | None]
    ttm_pages_limit: int | None
    dmi_memory_bytes: int | None
    device_gids: dict[str, int]
    current_group_ids: tuple[int, ...]
    packages: tuple[InstalledPackage, ...]
    apt_sources: tuple[AptSourceFile, ...]
    dkms_status: str
    docker_version: str | None
    docker_buildx_version: str | None
    docker_buildx_error: str | None
    docker_distribution: DockerDistribution
    kernel_oem_617_candidate: str | None
    display_manager_loaded: bool
    display_manager_active: bool
    dmesg: str
    dmesg_available: bool
    dedicated_vram_mib: int | None


@dataclass(frozen=True)
class PlannedAction:
    code: str
    summary: str
    argv: tuple[str, ...]
    privileged: bool
    input_text: str | None = None


@dataclass(frozen=True)
class PreparePlan:
    supported: bool
    target_user: str
    actions: tuple[PlannedAction, ...]
    reboot_required: bool
