import pytest

from amd_ai.host.models import DockerDistribution, InstalledPackage
from amd_ai.host.parsers import (
    classify_docker_distribution,
    parse_apt_candidate,
    parse_apt_policy_origin,
    parse_cmdline,
    parse_dmi_memory_bytes,
    parse_dpkg_packages,
    parse_lspci_gpu,
    parse_meminfo,
    parse_os_release,
    parse_vram_mib,
)


def test_parse_apt_candidate():
    assert (
        parse_apt_candidate("  Candidate: 6.17.0-1028.28\n")
        == "6.17.0-1028.28"
    )
    assert parse_apt_candidate("  Candidate: (none)\n") is None


@pytest.mark.parametrize(
    ("names", "expected"),
    [
        ({"docker-ce-cli"}, DockerDistribution.DOCKER_CE),
        ({"docker.io"}, DockerDistribution.UBUNTU_DOCKER_IO),
        ({"docker.io", "docker-ce"}, DockerDistribution.MIXED),
        (set(), DockerDistribution.EXTERNAL),
    ],
)
def test_classify_docker_distribution(names, expected):
    packages = tuple(InstalledPackage(name, "1") for name in sorted(names))

    assert (
        classify_docker_distribution(packages, runtime_available=True)
        is expected
    )


def test_missing_runtime_takes_precedence_over_installed_package():
    packages = (InstalledPackage("docker-ce-cli", "1"),)

    assert (
        classify_docker_distribution(packages, runtime_available=False)
        is DockerDistribution.MISSING
    )


def test_parse_target_host_facts():
    assert parse_os_release('ID=ubuntu\nVERSION_ID="24.04"\n') == {
        "ID": "ubuntu",
        "VERSION_ID": "24.04",
    }
    assert parse_meminfo(
        "MemTotal:       131015488 kB\nSwapTotal:              0 kB\n"
    ) == {
        "MemTotal": 131015488,
        "SwapTotal": 0,
    }
    args = parse_cmdline(
        "quiet splash amdgpu.gttsize=131072 ttm.pages_limit=33554432 "
        "amdgpu.mcbp=0 amdgpu.gpu_recovery=1 amdgpu.cwsr_enable=0"
    )
    assert args["ttm.pages_limit"] == "33554432"
    assert args["amdgpu.gttsize"] == "131072"


def test_parse_dmi_gpu_and_vram():
    dmi = "Size: 64 GB\nSize: 64 GB\nSize: No Module Installed\n"
    assert parse_dmi_memory_bytes(dmi) == 128 * 1024**3
    gpu = parse_lspci_gpu(
        "0000:c5:00.0 VGA compatible controller [0300]: Advanced Micro Devices, Inc. "
        "Device [1002:1586]\n\tKernel driver in use: amdgpu\n"
    )
    assert gpu.pci_id == "1002:1586"
    assert gpu.driver == "amdgpu"
    assert parse_vram_mib("amdgpu: 512M of VRAM memory ready") == 512
    assert parse_vram_mib("amdgpu: VRAM: 1024M 0x0 - 0x1") == 1024


def test_parse_packages_strips_architecture_and_finds_radeon_origin():
    assert parse_dpkg_packages(
        "rocm-core:amd64\t6.4.43483-1\nlinux-firmware\t20240318.git3b128b60-0ubuntu2\n"
    ) == (
        ("rocm-core", "6.4.43483-1"),
        ("linux-firmware", "20240318.git3b128b60-0ubuntu2"),
    )
    assert parse_apt_policy_origin(
        " 500 https://repo.radeon.com/rocm/apt/6.4 noble/main amd64 Packages\n"
    ) == "https://repo.radeon.com/rocm/apt/6.4"
    assert parse_apt_policy_origin(" 500 http://archive.ubuntu.com/ubuntu noble/main\n") is None
