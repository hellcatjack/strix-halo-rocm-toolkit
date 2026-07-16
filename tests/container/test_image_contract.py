from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest


BASE_IMAGE = "rocm-python:7.2.1-py3.12"
TORCH_IMAGE = "rocm-pytorch:7.2.1-py3.12-torch2.9.1"


def test_operator_documentation_contains_required_contract_anchors() -> None:
    required = {
        "README.md": ["./install.sh", "ROCm 7.2.1", "PyTorch 2.9.1"],
        "docs/install.md": [
            "--mode container",
            "--non-interactive",
            "KERNEL_REBOOT_PENDING",
        ],
        "docs/protected-pip.md": [
            "pip install",
            "--target",
            "overlay.requirements.lock",
        ],
        "docs/doctor-repair.md": [
            "TORCH.SHADOWED",
            "quarantine",
            "docker system prune",
        ],
        "docs/release-chain.md": [
            "manifest digest",
            "config digest",
            "anonymous",
        ],
    }
    for filename, anchors in required.items():
        text = Path(filename).read_text(encoding="utf-8")
        for anchor in anchors:
            assert anchor in text, f"{filename} is missing {anchor!r}"


def test_readme_quick_start_is_ordered_complete_and_safe() -> None:
    text = Path("README.md").read_text(encoding="utf-8")
    required_headings = (
        "## 快速开始",
        "## 目录",
        "## 项目解决什么问题",
    )
    for heading in required_headings:
        assert heading in text, f"README.md is missing {heading!r}"

    quick_start = text.index("## 快速开始")
    contents = text.index("## 目录")
    rationale = text.index("## 项目解决什么问题")
    assert quick_start < contents < rationale

    quick_text = text[quick_start:contents]
    required_steps = (
        "--mode full",
        '--project-dir "$PROJECT"',
        "sudo reboot",
        'strix-halo-rocm project run "$PROJECT"',
        "assert torch.version.hip",
        "torch.cuda.is_available()",
        'torch.device("cuda:0")',
        "torch.cuda.synchronize()",
        "pip install transformers safetensors",
        "strix-halo-rocm project lock",
        'strix-halo-rocm doctor "$PROJECT"',
        "**宿主机：**",
        "**项目容器内：**",
    )
    for step in required_steps:
        assert step in quick_text, f"quick start is missing {step!r}"

    assert re.search(r"(?m)^\s*sudo strix-halo-rocm(?:\s|$)", quick_text) is None


def test_readme_bash_examples_are_valid_shell() -> None:
    text = Path("README.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```bash\n(.*?)\n```", text, flags=re.DOTALL)
    assert blocks, "README.md has no Bash examples"

    for index, block in enumerate(blocks, start=1):
        result = subprocess.run(
            ("bash", "-n"),
            input=block,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        assert result.returncode == 0, f"Bash block {index}: {result.stderr}"


def _completed(prefix, args):
    return subprocess.run(
        (*prefix, *args),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


@pytest.fixture(scope="module")
def docker():
    for prefix in (("docker",), ("sudo", "-n", "docker")):
        if (
            _completed(prefix, ("info", "--format", "{{.ServerVersion}}")).returncode
            == 0
        ):
            for image in (BASE_IMAGE, TORCH_IMAGE):
                result = _completed(prefix, ("image", "inspect", image))
                assert result.returncode == 0, (
                    f"required contract image is missing: {image}: {result.stderr}"
                )
            return prefix
    pytest.skip("Docker daemon is unavailable to the user and sudo -n")


def _run(docker, image, script, *, user=None):
    args = ["run", "--rm"]
    if user is not None:
        args.extend(("--user", user))
    args.extend((image, "sh", "-eu", "-c", script))
    result = _completed(docker, tuple(args))
    assert result.returncode == 0, result.stderr or result.stdout
    return result.stdout


@pytest.mark.container
def test_base_has_no_torch_and_both_images_have_one_venv(docker):
    _run(
        docker,
        BASE_IMAGE,
        "python - <<'PY'\n"
        "import importlib.metadata as md\n"
        "import importlib.util\n"
        "assert importlib.util.find_spec('torch') is None\n"
        "try:\n"
        "    md.version('torch')\n"
        "except md.PackageNotFoundError:\n"
        "    pass\n"
        "else:\n"
        "    raise AssertionError('torch distribution is installed')\n"
        "PY",
    )
    for image in (BASE_IMAGE, TORCH_IMAGE):
        activations = _run(
            docker,
            image,
            "find / -xdev -type f -path '*/bin/activate' 2>/dev/null",
            user="0",
        ).splitlines()
        assert activations == ["/opt/venv/bin/activate"]


@pytest.mark.container
def test_images_are_non_root_without_forced_model_or_compiler_caches(docker):
    forbidden = {
        "HF_HOME",
        "HF_HUB_CACHE",
        "COMFYUI_MODELS",
        "MIOPEN_USER_DB_PATH",
        "TRITON_CACHE_DIR",
        "TORCHINDUCTOR_CACHE_DIR",
    }
    for image in (BASE_IMAGE, TORCH_IMAGE):
        result = _completed(
            docker,
            ("image", "inspect", "--format", "{{json .Config}}", image),
        )
        assert result.returncode == 0, result.stderr
        config = json.loads(result.stdout)
        assert config["User"] == "developer"
        names = {entry.partition("=")[0] for entry in config["Env"]}
        assert names.isdisjoint(forbidden)
        _run(
            docker,
            image,
            "python - <<'PY'\n"
            "import importlib.util\n"
            "from pathlib import Path\n"
            "assert importlib.util.find_spec('comfy') is None\n"
            "assert not Path('/workspace/ComfyUI').exists()\n"
            "PY",
        )


@pytest.mark.container
def test_base_contains_full_hip_cpp_development_toolchain(docker):
    _run(
        docker,
        BASE_IMAGE,
        "command -v hipcc cmake ninja g++ >/dev/null\n"
        "test -f /usr/include/python3.12/Python.h\n"
        "test -f /opt/rocm/include/hip/hip_runtime.h\n"
        "test -e /opt/rocm/lib/libroctx64.so.4",
    )


@pytest.mark.container
def test_torch_image_has_verified_labels_and_reuses_base_layers(docker):
    result = _completed(docker, ("image", "inspect", BASE_IMAGE, TORCH_IMAGE))
    assert result.returncode == 0, result.stderr
    base, torch = json.loads(result.stdout)
    labels = torch["Config"]["Labels"]
    assert labels["org.amd-ai.profile.id"] == "rocm-7.2.1-py3.12-torch-2.9.1"
    assert labels["org.amd-ai.profile.status"] == "verified"
    assert labels["org.amd-ai.rocm.version"] == "7.2.1"
    assert labels["org.amd-ai.torch.version"] == "2.9.1"
    base_layers = base["RootFS"]["Layers"]
    assert torch["RootFS"]["Layers"][: len(base_layers)] == base_layers


@pytest.mark.container
def test_torch_history_has_no_retained_wheel_copy_layer(docker):
    result = _completed(
        docker,
        ("history", "--no-trunc", "--format", "{{.CreatedBy}}", TORCH_IMAGE),
    )
    assert result.returncode == 0, result.stderr
    for command in result.stdout.splitlines():
        assert re.search(r"\b(?:COPY|ADD)\b.*(?:\.whl|wheels)", command, re.I) is None
