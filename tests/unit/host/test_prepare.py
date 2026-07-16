import json
from dataclasses import replace

import pytest

from amd_ai.host.models import (
    AptSourceFile,
    DockerDistribution,
    HostPlanPhase,
    InstalledPackage,
)
from amd_ai.host.parsers import GpuPciInfo
from amd_ai.host.prepare import (
    HostPlanningError,
    UnsupportedHostError,
    cleanup_candidates,
    create_prepare_plan,
    with_docker_group_action,
)
from tests.unit.host.fakes import healthy_snapshot


def action_codes(plan):
    return [action.code for action in plan.actions]


def test_kernel_plan_contains_no_ttm_or_docker_actions():
    plan = create_prepare_plan(
        healthy_snapshot(kernel="6.14.0-1020-oem"),
        target_user="customer",
        phase=HostPlanPhase.KERNEL,
    )

    codes = action_codes(plan)
    assert plan.phase is HostPlanPhase.KERNEL
    assert "APT.INSTALL_OEM_617" in codes
    assert "HOST.REBOOT" in codes
    assert not any(code.startswith("TTM.") for code in codes)
    assert not any(code.startswith("DOCKER.") for code in codes)


def test_tuning_plan_never_installs_a_kernel():
    plan = create_prepare_plan(
        healthy_snapshot(kernel="6.17.0-1025-oem"),
        target_user="customer",
        phase=HostPlanPhase.TUNING,
    )

    assert plan.phase is HostPlanPhase.TUNING
    assert "APT.INSTALL_OEM_617" not in action_codes(plan)


def test_platform_plan_is_read_only_for_gtt_and_never_reboots():
    plan = create_prepare_plan(
        healthy_snapshot(ttm_pages_limit=1),
        target_user="customer",
        phase=HostPlanPhase.TUNING,
    )

    codes = action_codes(plan)
    assert not any(code.startswith("TTM.") for code in codes)
    assert "HOST.REBOOT" not in codes
    assert plan.reboot_required is False
    assert "amd-ttm" not in " ".join(
        argument for action in plan.actions for argument in action.argv
    )


@pytest.mark.parametrize(
    ("docker_version", "buildx_version", "distribution", "expected"),
    [
        (None, None, DockerDistribution.MISSING, "DOCKER.INSTALL_IF_MISSING"),
        (
            "27.5.1",
            None,
            DockerDistribution.DOCKER_CE,
            "DOCKER.INSTALL_BUILDX_PLUGIN",
        ),
        (
            "27.5.1",
            None,
            DockerDistribution.UBUNTU_DOCKER_IO,
            "DOCKER.INSTALL_UBUNTU_BUILDX",
        ),
        ("27.5.1", "v0.30.1", DockerDistribution.DOCKER_CE, None),
    ],
)
def test_tuning_plan_selects_package_matched_docker_action(
    docker_version,
    buildx_version,
    distribution,
    expected,
):
    snapshot = healthy_snapshot(
        docker_version=docker_version,
        docker_buildx_version=buildx_version,
        docker_distribution=distribution,
    )

    plan = create_prepare_plan(
        snapshot,
        target_user="customer",
        phase=HostPlanPhase.TUNING,
    )
    docker_actions = [
        code for code in action_codes(plan) if code.startswith("DOCKER.INSTALL")
    ]

    assert docker_actions == ([] if expected is None else [expected])


@pytest.mark.parametrize(
    "distribution",
    [DockerDistribution.MIXED, DockerDistribution.EXTERNAL],
)
def test_tuning_plan_refuses_unsafe_buildx_distribution(distribution):
    snapshot = healthy_snapshot(
        docker_version="27.5.1",
        docker_buildx_version=None,
        docker_distribution=distribution,
    )

    with pytest.raises(HostPlanningError):
        create_prepare_plan(
            snapshot,
            target_user="customer",
            phase=HostPlanPhase.TUNING,
        )


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

    plan = create_prepare_plan(
        snapshot, target_user="customer", phase=HostPlanPhase.KERNEL
    )

    flattened = " ".join(arg for action in plan.actions for arg in action.argv)
    assert "amdgpu-dkms" in flattened
    assert " zfs-dkms " not in f" {flattened} "
    assert " autoremove " not in f" {flattened} "
    assert action_codes(plan) == [
        "BACKUP.SNAPSHOT",
        "APT.REMOVE_OLD_ROCM_PACKAGES",
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
        phase=HostPlanPhase.KERNEL,
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
            phase=HostPlanPhase.KERNEL,
        )


def test_platform_plan_uses_only_docker_and_pci_host_tools():
    snapshot = healthy_snapshot(
        docker_version=None,
        ttm_pages_limit=1,
        kernel_args={"quiet": None, "amdgpu.gttsize": "131072"},
    )

    plan = create_prepare_plan(
        snapshot, target_user="customer", phase=HostPlanPhase.TUNING
    )
    actions = {action.code: action for action in plan.actions}

    assert "APT.INSTALL_OEM_617" not in actions
    assert actions["APT.INSTALL_HOST_TOOLS"].argv == (
        "apt-get",
        "install",
        "-y",
        "ca-certificates",
        "curl",
        "gnupg",
        "pciutils",
    )
    assert actions["DOCKER.INSTALL_IF_MISSING"].argv == ()
    assert not any(code.startswith("TTM.") for code in actions)


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


def test_docker_group_action_is_pure_explicit_and_idempotent():
    plan = create_prepare_plan(
        healthy_snapshot(docker_version=None), target_user="customer"
    )

    extended = with_docker_group_action(plan)
    repeated = with_docker_group_action(extended)

    assert "DOCKER.ADD_USER_TO_GROUP" not in action_codes(plan)
    assert action_codes(extended).count("DOCKER.ADD_USER_TO_GROUP") == 1
    assert repeated == extended
    docker_index = action_codes(extended).index("DOCKER.INSTALL_IF_MISSING")
    assert action_codes(extended)[docker_index + 1] == "DOCKER.ADD_USER_TO_GROUP"
    action = extended.actions[docker_index + 1]
    assert action.argv == ("usermod", "-a", "-G", "docker", "customer")


def test_docker_group_action_is_not_added_for_root():
    plan = create_prepare_plan(healthy_snapshot(), target_user="root")

    assert with_docker_group_action(plan) == plan
