import shutil
from pathlib import Path

from amd_ai.host.probe import HostProbe
from tests.unit.host.fakes import FakeRunner


def test_probe_collects_target_snapshot():
    snapshot = HostProbe(
        root=Path("tests/fixtures/host/healthy"),
        runner=FakeRunner.healthy_target(),
        device_gids={"/dev/kfd": 0, "/dev/dri/renderD128": 128},
        current_group_ids=(0, 128),
    ).collect()

    assert snapshot.os_id == "ubuntu"
    assert snapshot.os_version == "24.04"
    assert snapshot.architecture == "x86_64"
    assert snapshot.kernel == "6.17.0-1025-oem"
    assert snapshot.gpu.pci_id == "1002:1586"
    assert snapshot.gpu.driver == "amdgpu"
    assert snapshot.mem_total_kib == 131015488
    assert snapshot.swap_total_kib == 0
    assert snapshot.ttm_pages_limit == 33554432
    assert snapshot.dedicated_vram_mib == 512
    assert snapshot.device_gids == {"/dev/kfd": 0, "/dev/dri/renderD128": 128}
    assert snapshot.docker_version == "27.5.1"


def test_probe_records_radeon_origin_for_old_rocm_package():
    snapshot = HostProbe(
        root=Path("tests/fixtures/host/healthy"),
        runner=FakeRunner.healthy_target(with_rocm64=True),
        device_gids={},
        current_group_ids=(),
    ).collect()

    rocm = next(package for package in snapshot.packages if package.name == "rocm-core")
    assert rocm.version == "6.4.43483-1"
    assert rocm.origin == "https://repo.radeon.com/rocm/apt/6.4"
    assert any("archive.ubuntu.com" in source.content for source in snapshot.apt_sources)


def test_probe_reads_legacy_amdttm_live_page_limit(tmp_path):
    root = tmp_path / "host"
    shutil.copytree("tests/fixtures/host/healthy", root)
    (root / "sys/module/ttm/parameters/pages_limit").unlink()
    legacy = root / "sys/module/amdttm/parameters/pages_limit"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("33554432\n", encoding="utf-8")

    snapshot = HostProbe(
        root=root,
        runner=FakeRunner.healthy_target(),
        device_gids={},
        current_group_ids=(),
    ).collect()

    assert snapshot.ttm_pages_limit == 33554432
