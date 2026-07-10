from __future__ import annotations

import pytest

from tests.gpu.torch_smoke import validate_result


def test_validate_result_accepts_gfx1151_and_small_errors():
    validate_result(
        {
            "available": True,
            "arch": "gfx1151:sramecc-:xnack-",
            "matmul_max_error": 0.01,
            "conv_max_error": 0.01,
        }
    )


def test_validate_result_rejects_cpu_fallback():
    with pytest.raises(AssertionError, match="GPU unavailable"):
        validate_result({"available": False})


def test_validate_result_rejects_wrong_arch_or_large_error():
    with pytest.raises(AssertionError):
        validate_result(
            {
                "available": True,
                "arch": "gfx1100",
                "matmul_max_error": 0.01,
                "conv_max_error": 0.01,
            }
        )
    with pytest.raises(AssertionError):
        validate_result(
            {
                "available": True,
                "arch": "gfx1151",
                "matmul_max_error": 0.21,
                "conv_max_error": 0.01,
            }
        )
