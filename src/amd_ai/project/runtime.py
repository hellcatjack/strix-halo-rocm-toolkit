from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from amd_ai.host.ttm import compute_ttm_plan
from amd_ai.project.config import MountConfig


class RuntimePolicyError(RuntimeError):
    pass


@dataclass(frozen=True)
class GpuAccess:
    devices: tuple[Path, Path]
    render_nodes: tuple[Path, ...]
    group_ids: tuple[int, ...]


def compute_shm_gib(*, mem_total_kib: int) -> int:
    if mem_total_kib <= 0:
        raise RuntimePolicyError("MemTotal must be positive")
    nominal_gib = compute_ttm_plan(
        mem_total_kib=mem_total_kib,
        page_size=4096,
    ).nominal_gib
    return max(4, min(16, nominal_gib // 8))


def read_mem_total_kib(path: Path = Path("/proc/meminfo")) -> int:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise RuntimePolicyError(f"cannot read {path}: {error}") from error
    match = re.search(r"^MemTotal:\s+(\d+)\s+kB$", text, re.MULTILINE)
    if match is None:
        raise RuntimePolicyError(f"MemTotal is missing from {path}")
    return int(match.group(1))


def discover_gpu_access(
    *,
    kfd: Path = Path("/dev/kfd"),
    dri: Path = Path("/dev/dri"),
    stat_gids: Mapping[str | Path, int] | None = None,
) -> GpuAccess:
    if stat_gids is None:
        if not kfd.exists():
            raise RuntimePolicyError(f"required GPU device is missing: {kfd}")
        if not dri.is_dir():
            raise RuntimePolicyError(f"required GPU device directory is missing: {dri}")
        render_nodes = tuple(sorted(dri.glob("renderD*"), key=lambda path: path.name))
        if not render_nodes:
            raise RuntimePolicyError(f"no render nodes exist under {dri}")
        gids = {os.stat(kfd).st_gid}
        gids.update(os.stat(node).st_gid for node in render_nodes)
    else:
        normalized = {str(path): gid for path, gid in stat_gids.items()}
        if str(kfd) not in normalized:
            raise RuntimePolicyError(f"required GPU device is missing: {kfd}")
        render_nodes = tuple(
            sorted(
                (
                    Path(path)
                    for path in normalized
                    if Path(path).parent == dri
                    and re.fullmatch(r"renderD\d+", Path(path).name)
                ),
                key=lambda path: path.name,
            )
        )
        if not render_nodes:
            raise RuntimePolicyError(f"no render nodes exist under {dri}")
        gids = {normalized[str(kfd)]}
        gids.update(normalized[str(node)] for node in render_nodes)
    if any(not isinstance(gid, int) or gid < 0 for gid in gids):
        raise RuntimePolicyError("GPU device GIDs must be nonnegative integers")
    return GpuAccess(
        devices=(kfd, dri),
        render_nodes=render_nodes,
        group_ids=tuple(sorted(gids)),
    )


def mount_argv(mounts: Sequence[MountConfig]) -> tuple[str, ...]:
    argv: list[str] = []
    for mount in mounts:
        if not mount.source.exists():
            raise RuntimePolicyError(f"mount source does not exist: {mount.source}")
        specification = f"type=bind,src={mount.source},dst={mount.target}"
        if mount.read_only:
            specification += ",readonly"
        argv.extend(("--mount", specification))
    return tuple(argv)
