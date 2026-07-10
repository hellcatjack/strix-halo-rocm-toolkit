import pytest

from amd_ai.host.ttm import MemoryConflict, compute_ttm_plan


def test_target_host_maps_124_95_gib_to_128_gib():
    plan = compute_ttm_plan(mem_total_kib=131015488, page_size=4096)

    assert plan.nominal_gib == 128
    assert plan.pages_limit == 33554432
    assert plan.legacy_gttsize_mib == 131072
    assert plan.source == "meminfo"


def test_dmi_is_preferred_when_consistent():
    plan = compute_ttm_plan(
        mem_total_kib=131015488,
        page_size=4096,
        dmi_memory_bytes=128 * 1024**3,
    )

    assert plan.source == "dmi"


def test_large_dmi_disagreement_requires_override():
    with pytest.raises(MemoryConflict):
        compute_ttm_plan(
            mem_total_kib=63 * 1024**2,
            page_size=4096,
            dmi_memory_bytes=128 * 1024**3,
        )


def test_explicit_capacity_is_auditable():
    plan = compute_ttm_plan(
        mem_total_kib=63 * 1024**2,
        page_size=4096,
        explicit_gib=64,
    )

    assert plan.nominal_gib == 64
    assert plan.source == "explicit"


@pytest.mark.parametrize("page_size", [0, 3000])
def test_page_size_must_be_positive_power_of_two(page_size):
    with pytest.raises(ValueError, match="power of two"):
        compute_ttm_plan(mem_total_kib=131015488, page_size=page_size)
