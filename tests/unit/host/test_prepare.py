import json
from dataclasses import replace

import pytest

from amd_ai.host.models import AptSourceFile, InstalledPackage
from amd_ai.host.parsers import GpuPciInfo
from amd_ai.host.prepare import (
    HostPlanningError,
    UnsupportedHostError,
    cleanup_candidates,
    create_prepare_plan,
)
from tests.unit.host.fakes import healthy_snapshot


def action_codes(plan):
    return [action.code for action in plan.actions]


def test_cleanup_only_selects_rocm_64_and_amdgpu_dkms():
    packages = (
        InstalledPackage(
            "rocm-core",
            "6.4.43483-1",
            "https://repo.radeon.com/rocm/apt/6.4",
        ),
        InstalledPackage(
            "hip-runtime-amd",
            "6.4.43483-1",
            "https://repo.radeon.com/rocm/apt/6.4",
        ),
        InstalledPackage(
            "amdgpu-dkms",
            "6.12.12.60402-1",
            "https://repo.radeon.com/graphics/6.4",
        ),
        InstalledPackage("dkms", "3.0.11-1ubuntu13"),
        InstalledPackage("zfs-dkms", "2.2.2-0ubuntu9"),
        InstalledPackage("virtualbox-dkms", "7.0.16-dfsg-2ubuntu1"),
        InstalledPackage("not-amd-runtime", "6.4.1", "https://repo.radeon.com/x/6.4"),
        InstalledPackage("rocblas", "7.2.1", "https://repo.radeon.com/rocm/apt/7.2.1"),
        InstalledPackage("rocm-no-origin", "6.4.1"),
    )

    assert cleanup_candidates(packages) == (
        "amdgpu-dkms",
        "hip-runtime-amd",
        "rocm-core",
    )


def test_plan_never_removes_generic_dkms_and_has_stable_order():
    packages = (
        InstalledPackage(
            "rocm-core",
            "6.4.43483-1",
            "https://repo.radeon.com/rocm/apt/6.4",
        ),
        InstalledPackage(
            "amdgpu-dkms",
            "6.12.12.60402-1",
            "https://repo.radeon.com/graphics/6.4",
        ),
        InstalledPackage("dkms", "3.0.11-1ubuntu13"),
        InstalledPackage("zfs-dkms", "2.2.2-0ubuntu9"),
    )
    snapshot = replace(healthy_snapshot(), packages=packages)

    plan = create_prepare_plan(snapshot, target_user="customer")

    flattened = " ".join(arg for action in plan.actions for arg in action.argv)
    assert "amdgpu-dkms" in flattened
    assert " zfs-dkms " not in f" {flattened} "
    assert " autoremove " not in f" {flattened} "
    assert action_codes(plan) == [
        "BACKUP.SNAPSHOT",
        "APT.REMOVE_OLD_ROCM_PACKAGES",
        "APT.INSTALL_OEM_KERNEL",
        "APT.INSTALL_HOST_TOOLS",
        "TTM.INSTALL_AMD_DEBUG_TOOLS",
        "HOST.REBOOT",
    ]
    assert plan.reboot_required is True


def test_old_radeon_source_detection_ignores_comments_and_new_repositories():
    sources = (
        AptSourceFile(
            "etc/apt/sources.list.d/rocm64.list",
            "deb https://repo.radeon.com/rocm/apt/6.4 noble main\n",
        ),
        AptSourceFile(
            "etc/apt/sources.list.d/commented.list",
            "# deb https://repo.radeon.com/rocm/apt/6.4 noble main\n",
        ),
        AptSourceFile(
            "etc/apt/sources.list.d/rocm72.sources",
            "URIs: https://repo.radeon.com/rocm/apt/7.2.1\n",
        ),
        AptSourceFile(
            "etc/apt/sources.list.d/substring.list",
            "deb https://repo.radeon.com/rocm/apt/6.40 noble main\n",
        ),
    )
    plan = create_prepare_plan(
        replace(healthy_snapshot(), apt_sources=sources),
        target_user="customer",
    )
    action = next(
        action
        for action in plan.actions
        if action.code == "APT.DISABLE_OLD_ROCM_SOURCES"
    )

    assert action.argv == ()
    assert json.loads(action.input_text) == ["etc/apt/sources.list.d/rocm64.list"]


def test_mixed_apt_source_file_is_never_disabled_wholesale():
    mixed = AptSourceFile(
        "etc/apt/sources.list",
        "deb http://archive.ubuntu.com/ubuntu noble main\n"
        "deb https://repo.radeon.com/rocm/apt/6.4 noble main\n",
    )

    with pytest.raises(HostPlanningError, match="mixed"):
        create_prepare_plan(
            replace(healthy_snapshot(), apt_sources=(mixed,)),
            target_user="customer",
        )


def test_plan_uses_exact_kernel_tools_and_ttm_commands():
    snapshot = healthy_snapshot(
        docker_version=None,
        ttm_pages_limit=1,
        kernel_args={"quiet": None, "amdgpu.gttsize": "131072"},
    )

    plan = create_prepare_plan(snapshot, target_user="customer")
    actions = {action.code: action for action in plan.actions}

    assert actions["APT.INSTALL_OEM_KERNEL"].argv == (
        "apt-get",
        "install",
        "-y",
        "linux-oem-24.04",
        "linux-headers-oem-24.04",
        "linux-firmware",
    )
    assert actions["APT.INSTALL_HOST_TOOLS"].argv == (
        "apt-get",
        "install",
        "-y",
        "ca-certificates",
        "curl",
        "gnupg",
        "pciutils",
        "python3-pip",
        "pipx",
    )
    assert actions["DOCKER.INSTALL_IF_MISSING"].argv == ()
    assert actions["TTM.SET_AI_MAX"].argv == (
        "/usr/local/bin/amd-ttm",
        "--set",
        "128",
    )


def test_missing_device_groups_are_derived_by_gid(monkeypatch):
    class Group:
        def __init__(self, name):
            self.gr_name = name

    names = {109: "video", 110: "render"}
    monkeypatch.setattr(
        "amd_ai.host.prepare.grp.getgrgid",
        lambda gid: Group(names[gid]),
    )
    snapshot = healthy_snapshot(current_group_ids=())

    plan = create_prepare_plan(snapshot, target_user="customer")
    action = next(
        action for action in plan.actions if action.code == "GROUPS.ADD_DEVICE_GROUPS"
    )

    assert action.argv == ("usermod", "-a", "-G", "video,render", "customer")


def test_unknown_distribution_refuses_to_create_a_write_plan():
    with pytest.raises(UnsupportedHostError, match="no host write adapter"):
        create_prepare_plan(
            healthy_snapshot(os_id="fedora", os_version="42"),
            target_user="customer",
        )


def test_missing_target_gpu_refuses_to_create_a_write_plan():
    with pytest.raises(HostPlanningError, match="GPU.NOT_FOUND"):
        create_prepare_plan(
            replace(
                healthy_snapshot(),
                gpu=GpuPciInfo(None, None, ""),
            ),
            target_user="customer",
        )
