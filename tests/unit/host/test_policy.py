import json
from dataclasses import replace
from pathlib import Path

import pytest

from amd_ai.host.adapters.base import select_adapter
from amd_ai.host.parsers import GpuPciInfo
from amd_ai.host.policy import evaluate_preflight
from amd_ai.report import Status
from tests.unit.host.fakes import healthy_snapshot


def finding_codes(report):
    return {finding.code for finding in report.findings}


def test_report_exposes_docker_kernel_candidate_and_display_facts():
    report = evaluate_preflight(healthy_snapshot())

    assert report.facts["docker_runtime_version"] == "27.5.1"
    assert report.facts["docker_buildx_version"] == (
        "github.com/docker/buildx v0.30.1"
    )
    assert report.facts["docker_distribution"] == "docker-ce"
    assert report.facts["kernel_oem_617_candidate"] == "6.17.0-1028.28"
    assert report.facts["display_manager"] == {
        "loaded": True,
        "active": True,
    }


def _tested_kernel_file(tmp_path: Path) -> Path:
    path = tmp_path / "tested-kernels.json"
    path.write_text(
        json.dumps(
            {"schema_version": 1, "kernels": ["6.17.0-1028-oem"]}
        ),
        encoding="utf-8",
    )
    return path


def test_healthy_gfx1151_oem_617_host_passes(tmp_path):
    report = evaluate_preflight(
        healthy_snapshot(kernel="6.17.0-1028-oem"),
        tested_kernels_path=_tested_kernel_file(tmp_path),
    )

    assert report.status is Status.PASS
    assert not [
        finding for finding in report.findings if finding.severity.value == "error"
    ]
    assert "HOST.SWAP_DISABLED" in finding_codes(report)


def test_newer_unrecorded_oem_kernel_is_explicitly_unverified():
    report = evaluate_preflight(healthy_snapshot(kernel="6.17.0-1025-oem"))

    assert report.status.value == "unverified"
    assert "HOST.UPSTREAM_UNVERIFIED" in finding_codes(report)


def test_unknown_distribution_is_blocked_and_has_no_write_adapter():
    snapshot = healthy_snapshot(os_id="fedora", os_version="42")

    report = evaluate_preflight(snapshot)

    assert report.status.value == "blocked"
    assert "HOST.UNSUPPORTED_OS" in finding_codes(report)
    assert select_adapter(snapshot) is None


@pytest.mark.parametrize(
    "kernel",
    ["6.8.0-31-generic", "6.14.0-1020-oem", "6.17.0-generic"],
)
def test_old_or_non_oem_kernel_requires_oem_617(kernel):
    report = evaluate_preflight(healthy_snapshot(kernel=kernel))

    assert report.status is Status.CHANGE_REQUIRED
    assert "HOST.OEM_617_REQUIRED" in finding_codes(report)


@pytest.mark.parametrize("kernel", ["6.17.0-1029-oem", "6.19.0-1001-oem"])
def test_unlisted_or_future_oem_kernel_is_unverified(kernel, tmp_path):
    report = evaluate_preflight(
        healthy_snapshot(kernel=kernel),
        tested_kernels_path=_tested_kernel_file(tmp_path),
    )

    assert report.status is Status.UNVERIFIED
    assert "HOST.UPSTREAM_UNVERIFIED" in finding_codes(report)


def test_old_kernel_without_oem_617_candidate_is_blocked():
    report = evaluate_preflight(
        healthy_snapshot(
            kernel="6.14.0-1020-oem",
            kernel_oem_617_candidate=None,
        )
    )

    assert report.status is Status.BLOCKED
    assert "HOST.OEM_617_CANDIDATE" in finding_codes(report)


def test_non_amd64_host_is_blocked():
    report = evaluate_preflight(healthy_snapshot(architecture="aarch64"))

    assert report.status.value == "blocked"
    assert "HOST.UNSUPPORTED_ARCH" in finding_codes(report)


@pytest.mark.parametrize(
    ("snapshot_changes", "expected_code"),
    [
        ({"device_gids": {"/dev/dri/renderD128": 110}}, "GPU.KFD_MISSING"),
        ({"device_gids": {"/dev/kfd": 109}}, "GPU.RENDER_MISSING"),
        ({"current_group_ids": (109,)}, "GPU.PERMISSION"),
        ({"dedicated_vram_mib": 4096}, "GPU.BIOS_VRAM_HIGH"),
    ],
)
def test_repairable_gpu_host_issues_require_change(snapshot_changes, expected_code):
    report = evaluate_preflight(
        healthy_snapshot(kernel="6.17.0-1025-oem", **snapshot_changes)
    )

    assert report.status.value == "change-required"
    assert expected_code in finding_codes(report)


def test_missing_target_gpu_and_wrong_driver_are_blocked():
    missing_gpu = replace(
        healthy_snapshot(),
        gpu=GpuPciInfo(pci_id=None, driver=None, raw=""),
    )
    wrong_driver = replace(
        healthy_snapshot(),
        gpu=GpuPciInfo(pci_id="1002:1586", driver="vfio-pci", raw="vfio-pci"),
    )

    missing_report = evaluate_preflight(missing_gpu)
    driver_report = evaluate_preflight(wrong_driver)

    assert missing_report.status.value == "blocked"
    assert "GPU.NOT_FOUND" in finding_codes(missing_report)
    assert driver_report.status.value == "blocked"
    assert "GPU.WRONG_DRIVER" in finding_codes(driver_report)


def test_blocked_status_takes_precedence_over_change_and_unverified():
    report = evaluate_preflight(
        healthy_snapshot(
            os_id="fedora",
            os_version="42",
            device_gids={},
            dedicated_vram_mib=4096,
        )
    )

    assert report.status.value == "blocked"


def test_ubuntu_2404_amd64_selects_the_write_adapter():
    adapter = select_adapter(healthy_snapshot())

    assert adapter is not None
    assert adapter.adapter_id == "ubuntu-24.04"
