from __future__ import annotations

import grp
import json
import re
from dataclasses import replace
from urllib.parse import urlparse

from amd_ai.host.adapters.base import select_adapter
from amd_ai.host.models import (
    AptSourceFile,
    HostSnapshot,
    InstalledPackage,
    PlannedAction,
    PreparePlan,
)
from amd_ai.host.ttm import compute_ttm_plan
from amd_ai.host.policy import evaluate_preflight


ROCM_PACKAGE_PREFIXES = (
    "rocm",
    "hip-",
    "hsa-",
    "hsakmt",
    "comgr",
    "miopen",
    "rocblas",
    "rocfft",
    "rocrand",
    "rocsolver",
    "rocsparse",
    "rccl",
)
PROTECTED_DKMS_PACKAGES = {"dkms", "zfs-dkms", "virtualbox-dkms"}
NON_REMEDIABLE_PREFLIGHT_CODES = {
    "HOST.UNSUPPORTED_OS",
    "HOST.UNSUPPORTED_ARCH",
    "GPU.NOT_FOUND",
    "GPU.WRONG_DRIVER",
}


class UnsupportedHostError(RuntimeError):
    pass


class HostPlanningError(RuntimeError):
    pass


def cleanup_candidates(packages: tuple[InstalledPackage, ...]) -> tuple[str, ...]:
    selected: set[str] = set()
    for package in packages:
        if package.name in PROTECTED_DKMS_PACKAGES:
            continue
        if package.name == "amdgpu-dkms":
            selected.add(package.name)
            continue
        if (
            package.version.startswith("6.4")
            and package.origin is not None
            and "repo.radeon.com" in package.origin
            and package.name.startswith(ROCM_PACKAGE_PREFIXES)
        ):
            selected.add(package.name)
    return tuple(sorted(selected))


def create_prepare_plan(
    snapshot: HostSnapshot,
    *,
    target_user: str,
    memory_gib: int | None = None,
) -> PreparePlan:
    adapter = select_adapter(snapshot)
    if adapter is None:
        raise UnsupportedHostError(
            f"no host write adapter for {snapshot.os_id} {snapshot.os_version} "
            f"on {snapshot.architecture}"
        )
    preflight = evaluate_preflight(snapshot)
    blocking_codes = sorted(
        finding.code
        for finding in preflight.findings
        if finding.code in NON_REMEDIABLE_PREFLIGHT_CODES
    )
    if blocking_codes:
        raise HostPlanningError(
            "host preparation is blocked by preflight: " + ", ".join(blocking_codes)
        )
    return adapter.create_prepare_plan(snapshot, target_user, memory_gib)


def with_docker_group_action(plan: PreparePlan) -> PreparePlan:
    if plan.target_user == "root" or any(
        action.code == "DOCKER.ADD_USER_TO_GROUP" for action in plan.actions
    ):
        return plan
    action = PlannedAction(
        code="DOCKER.ADD_USER_TO_GROUP",
        summary="Grant the target user access to the Docker daemon",
        argv=("usermod", "-a", "-G", "docker", plan.target_user),
        privileged=True,
    )
    actions = list(plan.actions)
    insertion = next(
        (
            index + 1
            for index, existing in enumerate(actions)
            if existing.code == "DOCKER.INSTALL_IF_MISSING"
        ),
        next(
            (
                index + 1
                for index, existing in enumerate(actions)
                if existing.code == "APT.INSTALL_HOST_TOOLS"
            ),
            1,
        ),
    )
    actions.insert(insertion, action)
    return replace(plan, actions=tuple(actions))


def create_ubuntu_prepare_plan(
    snapshot: HostSnapshot,
    target_user: str,
    memory_gib: int | None,
) -> PreparePlan:
    ttm = compute_ttm_plan(
        mem_total_kib=snapshot.mem_total_kib,
        page_size=snapshot.page_size,
        dmi_memory_bytes=snapshot.dmi_memory_bytes,
        explicit_gib=memory_gib,
    )
    actions: list[PlannedAction] = [
        _internal_action("BACKUP.SNAPSHOT", "Back up the current host state")
    ]

    mixed_sources = mixed_old_rocm_source_paths(snapshot.apt_sources)
    if mixed_sources:
        raise HostPlanningError(
            "mixed APT source files require manual line/stanza cleanup: "
            + ", ".join(mixed_sources)
        )
    source_paths = old_rocm_source_paths(snapshot.apt_sources)
    if source_paths:
        actions.append(
            _internal_action(
                "APT.DISABLE_OLD_ROCM_SOURCES",
                "Disable confirmed ROCm 6.4 APT sources",
                input_text=json.dumps(source_paths),
            )
        )

    packages = cleanup_candidates(snapshot.packages)
    if packages:
        actions.append(
            PlannedAction(
                code="APT.REMOVE_OLD_ROCM_PACKAGES",
                summary="Remove confirmed ROCm 6.4 packages and amdgpu-dkms",
                argv=("apt-get", "remove", "--purge", "-y", *packages),
                privileged=True,
            )
        )

    actions.extend(
        (
            PlannedAction(
                code="APT.INSTALL_OEM_KERNEL",
                summary="Install the Ubuntu 24.04 OEM kernel, headers, and firmware",
                argv=(
                    "apt-get",
                    "install",
                    "-y",
                    "linux-oem-24.04",
                    "linux-headers-oem-24.04",
                    "linux-firmware",
                ),
                privileged=True,
            ),
            PlannedAction(
                code="APT.INSTALL_HOST_TOOLS",
                summary="Install host preparation and diagnostic tools",
                argv=(
                    "apt-get",
                    "install",
                    "-y",
                    "ca-certificates",
                    "curl",
                    "gnupg",
                    "pciutils",
                    "python3-pip",
                    "pipx",
                ),
                privileged=True,
            ),
        )
    )

    if snapshot.docker_version is None:
        actions.append(
            _internal_action(
                "DOCKER.INSTALL_IF_MISSING",
                "Install Docker Engine from the official Ubuntu repository",
            )
        )

    group_names = _missing_device_group_names(snapshot)
    if group_names:
        actions.append(
            PlannedAction(
                code="GROUPS.ADD_DEVICE_GROUPS",
                summary="Add the target user to GPU device groups",
                argv=(
                    "usermod",
                    "-a",
                    "-G",
                    ",".join(group_names),
                    target_user,
                ),
                privileged=True,
            )
        )

    actions.append(
        _internal_action(
            "TTM.INSTALL_AMD_DEBUG_TOOLS",
            "Install the pinned amd-debug-tools package after hash verification",
        )
    )
    if snapshot.ttm_pages_limit != ttm.pages_limit:
        actions.append(
            PlannedAction(
                code="TTM.SET_AI_MAX",
                summary=(
                    f"Set the AI Max TTM limit to {ttm.nominal_gib} GiB "
                    f"({ttm.pages_limit} pages)"
                ),
                argv=(
                    "/usr/local/bin/amd-ttm",
                    "--set",
                    str(ttm.nominal_gib),
                ),
                privileged=True,
            )
        )

    actions.append(
        PlannedAction(
            code="HOST.REBOOT",
            summary="Reboot to activate host kernel and TTM changes",
            argv=("systemctl", "reboot"),
            privileged=True,
        )
    )
    return PreparePlan(
        supported=True,
        target_user=target_user,
        actions=tuple(actions),
        reboot_required=True,
    )


def old_rocm_source_paths(sources: tuple[AptSourceFile, ...]) -> tuple[str, ...]:
    return tuple(
        sorted(
            source.path
            for source in sources
            if _old_rocm_source_state(source) == "exclusive"
        )
    )


def mixed_old_rocm_source_paths(
    sources: tuple[AptSourceFile, ...],
) -> tuple[str, ...]:
    return tuple(
        sorted(
            source.path
            for source in sources
            if _old_rocm_source_state(source) == "mixed"
        )
    )


def _old_rocm_source_state(source: AptSourceFile) -> str:
    urls: list[str] = []
    for line in source.content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        urls.extend(re.findall(r"https?://[^\s#]+", stripped))
    old_flags = [_is_old_rocm_url(url) for url in urls]
    if not any(old_flags):
        return "none"
    return "exclusive" if all(old_flags) else "mixed"


def _is_old_rocm_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.hostname == "repo.radeon.com" and re.search(
        r"/6\.4(?:[./]|$)", parsed.path
    ) is not None


def _missing_device_group_names(snapshot: HostSnapshot) -> tuple[str, ...]:
    missing_gids = sorted(
        set(snapshot.device_gids.values()).difference(snapshot.current_group_ids)
    )
    names: list[str] = []
    for gid in missing_gids:
        try:
            name = grp.getgrgid(gid).gr_name
        except KeyError as error:
            raise HostPlanningError(f"device GID {gid} has no group name") from error
        if name not in names:
            names.append(name)
    return tuple(names)


def _internal_action(
    code: str,
    summary: str,
    *,
    input_text: str | None = None,
) -> PlannedAction:
    return PlannedAction(
        code=code,
        summary=summary,
        argv=(),
        privileged=True,
        input_text=input_text,
    )
