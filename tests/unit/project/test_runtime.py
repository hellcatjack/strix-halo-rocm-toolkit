from pathlib import Path, PurePosixPath

import pytest

from amd_ai.project.config import MountConfig
from amd_ai.project.runtime import (
    RuntimePolicyError,
    compute_shm_gib,
    discover_gpu_access,
    mount_argv,
)


def test_nominal_memory_uses_bounded_shared_memory():
    assert compute_shm_gib(mem_total_kib=131015488) == 16
    assert compute_shm_gib(mem_total_kib=31 * 1024**2) == 4
    assert compute_shm_gib(mem_total_kib=7 * 1024**2) == 4


def test_device_groups_are_sorted_and_deduplicated():
    access = discover_gpu_access(
        kfd=Path("/dev/kfd"),
        dri=Path("/dev/dri"),
        stat_gids={
            "/dev/kfd": 109,
            "/dev/dri/renderD128": 110,
            "/dev/dri/renderD129": 110,
        },
    )

    assert access.devices == (Path("/dev/kfd"), Path("/dev/dri"))
    assert access.render_nodes == (
        Path("/dev/dri/renderD128"),
        Path("/dev/dri/renderD129"),
    )
    assert access.group_ids == (109, 110)


def test_missing_kfd_or_render_node_is_reported_before_docker():
    with pytest.raises(RuntimePolicyError, match="/dev/kfd"):
        discover_gpu_access(
            kfd=Path("/dev/kfd"),
            dri=Path("/dev/dri"),
            stat_gids={},
        )
    with pytest.raises(RuntimePolicyError, match="render"):
        discover_gpu_access(
            kfd=Path("/dev/kfd"),
            dri=Path("/dev/dri"),
            stat_gids={"/dev/kfd": 109},
        )


def test_mount_argv_preserves_explicit_access_mode(tmp_path):
    source = tmp_path / "models"
    source.mkdir()
    mounts = (
        MountConfig(source, PurePosixPath("/models"), True),
        MountConfig(tmp_path, PurePosixPath("/outputs"), False),
    )

    assert mount_argv(mounts) == (
        "--mount",
        f"type=bind,src={source},dst=/models,readonly",
        "--mount",
        f"type=bind,src={tmp_path},dst=/outputs",
    )


def test_missing_mount_source_is_not_created(tmp_path):
    source = tmp_path / "missing"

    with pytest.raises(RuntimePolicyError, match="does not exist"):
        mount_argv((MountConfig(source, PurePosixPath("/models"), True),))

    assert not source.exists()
