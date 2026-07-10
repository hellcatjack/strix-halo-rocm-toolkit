from __future__ import annotations

import json
import math


def validate_result(result: dict[str, object]) -> None:
    assert result.get("available") is True, (
        "GPU unavailable; CPU fallback is forbidden"
    )
    architecture = str(result.get("arch", ""))
    assert architecture.startswith("gfx1151"), architecture
    for name in ("matmul_max_error", "conv_max_error"):
        error = float(result[name])
        assert math.isfinite(error) and 0 <= error <= 0.2, f"{name}={error}"


def run() -> dict[str, object]:
    import torch
    import torch.nn.functional as functional

    result: dict[str, object] = {
        "available": torch.cuda.is_available(),
        "torch": torch.__version__,
        "hip": torch.version.hip,
    }
    if not torch.cuda.is_available():
        validate_result(result)
    properties = torch.cuda.get_device_properties(0)
    architecture = str(getattr(properties, "gcnArchName", ""))
    if not architecture:
        architecture = next(
            (
                value
                for value in torch.cuda.get_arch_list()
                if value.startswith("gfx1151")
            ),
            "",
        )
    torch.manual_seed(7)
    left_cpu = torch.randn((1024, 1024), dtype=torch.float32)
    right_cpu = torch.randn((1024, 1024), dtype=torch.float32)
    expected_mm = left_cpu @ right_cpu
    actual_mm = (left_cpu.half().cuda() @ right_cpu.half().cuda()).float().cpu()
    image_cpu = torch.randn((2, 4, 64, 64), dtype=torch.float32)
    kernel_cpu = torch.randn((8, 4, 3, 3), dtype=torch.float32)
    expected_conv = functional.conv2d(image_cpu, kernel_cpu, padding=1)
    actual_conv = functional.conv2d(
        image_cpu.half().cuda(),
        kernel_cpu.half().cuda(),
        padding=1,
    ).float().cpu()
    torch.cuda.synchronize()
    result.update(
        {
            "device": torch.cuda.get_device_name(0),
            "arch": architecture,
            "matmul_max_error": (expected_mm - actual_mm).abs().max().item(),
            "conv_max_error": (expected_conv - actual_conv).abs().max().item(),
        }
    )
    validate_result(result)
    return result


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True))
