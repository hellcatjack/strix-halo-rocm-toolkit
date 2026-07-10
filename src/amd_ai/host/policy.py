from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from amd_ai.host.models import HostSnapshot
from amd_ai.report import Finding, Report, Severity, Status


TARGET_PCI_ID = "1002:1586"
MINIMUM_OEM_KERNEL = (6, 14, 0, 1018)


@dataclass(frozen=True)
class _ClassifiedFinding:
    finding: Finding
    status: Status


def evaluate_preflight(
    snapshot: HostSnapshot,
    *,
    tested_kernels_path: Path | None = None,
) -> Report:
    classified: list[_ClassifiedFinding] = []

    def add(
        code: str,
        severity: Severity,
        status: Status,
        summary: str,
        evidence: str,
        remediation: str,
    ) -> None:
        classified.append(
            _ClassifiedFinding(
                Finding(code, severity, summary, evidence, remediation),
                status,
            )
        )

    if snapshot.os_id != "ubuntu" or not snapshot.os_version.startswith("24.04"):
        add(
            "HOST.UNSUPPORTED_OS",
            Severity.ERROR,
            Status.BLOCKED,
            "No write adapter is available for this operating system",
            f"ID={snapshot.os_id!r}, VERSION_ID={snapshot.os_version!r}",
            "Use Ubuntu 24.04.x for host preparation; other systems are audit-only.",
        )

    if snapshot.architecture != "x86_64":
        add(
            "HOST.UNSUPPORTED_ARCH",
            Severity.ERROR,
            Status.BLOCKED,
            "The host architecture is unsupported",
            f"uname -m returned {snapshot.architecture!r}",
            "Use an AMD64/x86_64 host.",
        )

    kernel_version = _parse_oem_kernel(snapshot.kernel)
    if kernel_version is None or kernel_version < MINIMUM_OEM_KERNEL:
        add(
            "HOST.OEM_KERNEL",
            Severity.ERROR,
            Status.BLOCKED,
            "A supported Ubuntu OEM kernel is not running",
            f"running kernel: {snapshot.kernel or '<unknown>'}",
            "Install linux-oem-24.04 and boot kernel 6.14.0-1018-oem or newer.",
        )

    if snapshot.gpu.pci_id != TARGET_PCI_ID:
        add(
            "GPU.NOT_FOUND",
            Severity.ERROR,
            Status.BLOCKED,
            "The Radeon 8060S target PCI device was not found",
            f"detected PCI ID: {snapshot.gpu.pci_id or '<none>'}",
            "Confirm the Ryzen AI Max+ 395 GPU is enabled and visible to the host.",
        )
    elif snapshot.gpu.driver != "amdgpu":
        add(
            "GPU.WRONG_DRIVER",
            Severity.ERROR,
            Status.BLOCKED,
            "The target GPU is not using the inbox amdgpu driver",
            f"kernel driver: {snapshot.gpu.driver or '<none>'}",
            "Remove conflicting GPU DKMS/VFIO binding and use the kernel inbox amdgpu module.",
        )

    has_kfd = "/dev/kfd" in snapshot.device_gids
    render_nodes = {
        path: gid
        for path, gid in snapshot.device_gids.items()
        if path.startswith("/dev/dri/render")
    }
    if not has_kfd:
        add(
            "GPU.KFD_MISSING",
            Severity.WARNING,
            Status.CHANGE_REQUIRED,
            "/dev/kfd is missing",
            "No KFD device node was collected",
            "Load the inbox amdgpu driver, then reboot and rerun preflight.",
        )
    if not render_nodes:
        add(
            "GPU.RENDER_MISSING",
            Severity.WARNING,
            Status.CHANGE_REQUIRED,
            "No DRM render node is available",
            "No /dev/dri/render* node was collected",
            "Check amdgpu initialization and udev, then reboot and rerun preflight.",
        )

    required_gids = set(snapshot.device_gids.values())
    missing_gids = sorted(required_gids.difference(snapshot.current_group_ids))
    if required_gids and missing_gids:
        add(
            "GPU.PERMISSION",
            Severity.WARNING,
            Status.CHANGE_REQUIRED,
            "The current user lacks one or more GPU device groups",
            "missing GIDs: " + ", ".join(str(gid) for gid in missing_gids),
            "Add the target user to the device groups and start a new login session.",
        )

    if snapshot.swap_total_kib == 0:
        add(
            "HOST.SWAP_DISABLED",
            Severity.WARNING,
            Status.PASS,
            "Swap is disabled",
            "SwapTotal is 0 KiB",
            "For memory-heavy video workloads, consider a bounded swap file or zram policy.",
        )

    if (
        snapshot.dedicated_vram_mib is not None
        and snapshot.dedicated_vram_mib > 512
    ):
        add(
            "GPU.BIOS_VRAM_HIGH",
            Severity.WARNING,
            Status.CHANGE_REQUIRED,
            "Dedicated UMA memory is above the 512 MiB AI Max baseline",
            f"reported dedicated VRAM: {snapshot.dedicated_vram_mib} MiB",
            "In firmware setup, set the UMA frame buffer to 512 MiB; this cannot be changed by the host tool.",
        )

    if kernel_version is not None and kernel_version >= MINIMUM_OEM_KERNEL:
        tested_kernels = _load_tested_kernels(tested_kernels_path)
        if snapshot.kernel not in tested_kernels:
            add(
                "HOST.UPSTREAM_UNVERIFIED",
                Severity.WARNING,
                Status.UNVERIFIED,
                "The OEM kernel meets the minimum but is not in the recorded AMD-tested set",
                f"running kernel: {snapshot.kernel}",
                "Run the full hardware qualification before promoting this kernel.",
            )

    return Report(
        command="host-preflight",
        status=_overall_status(classified),
        generated_at=datetime.now(UTC).isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        ),
        facts=_snapshot_facts(snapshot),
        findings=tuple(item.finding for item in classified),
    )


def _parse_oem_kernel(kernel: str) -> tuple[int, int, int, int] | None:
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)-(\d+)-oem", kernel)
    if match is None:
        return None
    return tuple(int(part) for part in match.groups())  # type: ignore[return-value]


def _load_tested_kernels(path: Path | None) -> frozenset[str]:
    if path is None:
        path = Path(__file__).resolve().parents[3] / "profiles/host/tested-kernels.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("kernels"), list):
        raise ValueError(f"invalid tested-kernel data: {path}")
    return frozenset(str(kernel) for kernel in payload["kernels"])


def _overall_status(findings: list[_ClassifiedFinding]) -> Status:
    precedence = (
        Status.BLOCKED,
        Status.REBOOT_REQUIRED,
        Status.CHANGE_REQUIRED,
        Status.UNVERIFIED,
    )
    statuses = {item.status for item in findings}
    return next((status for status in precedence if status in statuses), Status.PASS)


def _snapshot_facts(snapshot: HostSnapshot) -> dict[str, object]:
    return {
        "os": {"id": snapshot.os_id, "version": snapshot.os_version},
        "architecture": snapshot.architecture,
        "kernel": snapshot.kernel,
        "gpu": {
            "pci_id": snapshot.gpu.pci_id,
            "driver": snapshot.gpu.driver,
        },
        "memory": {
            "mem_total_kib": snapshot.mem_total_kib,
            "swap_total_kib": snapshot.swap_total_kib,
            "dmi_bytes": snapshot.dmi_memory_bytes,
            "page_size": snapshot.page_size,
        },
        "kernel_args": snapshot.kernel_args,
        "ttm_pages_limit": snapshot.ttm_pages_limit,
        "dedicated_vram_mib": snapshot.dedicated_vram_mib,
        "device_gids": snapshot.device_gids,
        "current_group_ids": list(snapshot.current_group_ids),
        "packages": [asdict(package) for package in snapshot.packages],
        "apt_source_paths": [source.path for source in snapshot.apt_sources],
        "dkms_status": snapshot.dkms_status,
        "docker_version": snapshot.docker_version,
    }

