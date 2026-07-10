from __future__ import annotations

import math
from dataclasses import dataclass


GIB = 1024**3
KIB = 1024


class MemoryConflict(ValueError):
    pass


@dataclass(frozen=True)
class TtmPlan:
    nominal_gib: int
    page_size: int
    pages_limit: int
    legacy_gttsize_mib: int
    source: str


def _normalize_gib(byte_count: int) -> int:
    return math.ceil((byte_count / GIB) / 8) * 8


def compute_ttm_plan(
    *,
    mem_total_kib: int,
    page_size: int,
    dmi_memory_bytes: int | None = None,
    explicit_gib: int | None = None,
) -> TtmPlan:
    if page_size <= 0 or page_size & (page_size - 1):
        raise ValueError("page size must be a positive power of two")

    meminfo_gib = _normalize_gib(mem_total_kib * KIB)
    if explicit_gib is not None:
        if explicit_gib <= 0 or explicit_gib % 8:
            raise ValueError("explicit capacity must be a positive multiple of 8 GiB")
        nominal_gib, source = explicit_gib, "explicit"
    elif dmi_memory_bytes is not None:
        dmi_gib = _normalize_gib(dmi_memory_bytes)
        if abs(dmi_gib - meminfo_gib) > 8:
            raise MemoryConflict(f"DMI={dmi_gib} GiB, MemTotal={meminfo_gib} GiB")
        nominal_gib, source = dmi_gib, "dmi"
    else:
        nominal_gib, source = meminfo_gib, "meminfo"

    return TtmPlan(
        nominal_gib=nominal_gib,
        page_size=page_size,
        pages_limit=(nominal_gib * GIB) // page_size,
        legacy_gttsize_mib=nominal_gib * 1024,
        source=source,
    )
