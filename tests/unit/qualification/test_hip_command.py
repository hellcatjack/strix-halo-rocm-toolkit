from __future__ import annotations

from tests.gpu.run_hip_smoke import compile_argv


def test_hip_compile_targets_native_gfx1151(tmp_path):
    argv = compile_argv(
        tmp_path / "hip_vector_add.cpp",
        tmp_path / "hip-vector-add",
    )

    assert argv[0] == "/opt/rocm/bin/hipcc"
    assert "--offload-arch=gfx1151" in argv
    assert "-O2" in argv
    assert argv[-2:] == ("-o", str(tmp_path / "hip-vector-add"))
