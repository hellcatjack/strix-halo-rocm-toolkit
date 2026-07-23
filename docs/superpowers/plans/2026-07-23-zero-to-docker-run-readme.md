# Zero-to-Docker-Run README Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the existing README quick start into the single linear path from a new Ubuntu 24.04 host to an interactive `video-lab:runtime` container launched with standard `docker run`.

**Architecture:** Keep `快速开始` as the only quick-start section and make it own the only complete direct-run command and manual Torch GPU probe. Shorten `极简 Docker 直启` into parameter and risk documentation that links back to the canonical quick-start steps, while leaving the managed project workflow in its existing detailed section. Update the existing README contract test first so the intended documentation behavior is executable and regression-resistant.

**Tech Stack:** Markdown, Bash examples, Python 3.10+ documentation contract tests, pytest 8.

---

## File Map

- Modify `tests/container/test_image_contract.py`: replace the old managed-run quick-start contract with assertions for one direct-run path, required GPU arguments, immutable SWR parent identity, and forbidden unsafe flags.
- Modify `README.md`: replace steps 5-7 of the existing quick start and reduce the detailed direct-run section without creating another quick start.
- Read only `profiles/releases/stable.json`: source of the stable Torch manifest digest asserted by the test.
- Read only `templates/project/.dockerignore`: confirms `.cache` remains outside generated project build contexts.

### Task 1: Encode The New README Contract

**Files:**
- Modify: `tests/container/test_image_contract.py:45-80`
- Test: `tests/container/test_image_contract.py`

- [ ] **Step 1: Replace the old quick-start assertions**

Replace `test_readme_quick_start_is_ordered_complete_and_safe` with:

```python
def test_readme_quick_start_is_ordered_complete_and_safe() -> None:
    text = Path("README.md").read_text(encoding="utf-8")
    required_headings = (
        "## 快速开始",
        "## 目录",
        "## 项目解决什么问题",
    )
    for heading in required_headings:
        assert heading in text, f"README.md is missing {heading!r}"

    assert text.count("## 快速开始\n") == 1
    quick_start = text.index("## 快速开始")
    contents = text.index("## 目录")
    rationale = text.index("## 项目解决什么问题")
    assert quick_start < contents < rationale

    quick_text = text[quick_start:contents]
    required_steps = (
        "--mode full",
        '--project-dir "$PROJECT"',
        "sudo reboot",
        "ls -l /dev/kfd /dev/dri/render*",
        "docker image inspect video-lab:runtime",
        "docker run --rm -it",
        "assert torch.version.hip",
        "torch.cuda.is_available()",
        'torch.device("cuda:0")',
        "torch.cuda.synchronize()",
        "python -m pip install transformers safetensors",
        "**宿主机：**",
        "**项目容器内：**",
    )
    for step in required_steps:
        assert step in quick_text, f"quick start is missing {step!r}"

    assert "strix-halo-rocm project run" not in quick_text
    assert re.search(r"(?m)^\s*sudo strix-halo-rocm(?:\s|$)", quick_text) is None

    bash_blocks = re.findall(r"```bash\n(.*?)\n```", text, flags=re.DOTALL)
    docker_blocks = [block for block in bash_blocks if "docker run --rm -it" in block]
    assert len(docker_blocks) == 1, "README must contain one canonical direct-run command"
    docker_block = docker_blocks[0]

    required_arguments = (
        "--device /dev/kfd",
        "--device /dev/dri",
        "--group-add \"$(stat -c '%g' /dev/kfd)\"",
        "--group-add \"$(stat -c '%g' \"$RENDER_NODE\")\"",
        "--user \"$(id -u):$(id -g)\"",
        "--ipc=private",
        "--shm-size=16g",
        "PIP_TARGET=/workspace/.cache/python-site",
        "PYTHONPATH=/workspace/.cache/python-site:/workspace",
        '--mount "type=bind,src=$PROJECT,dst=/workspace"',
        "--workdir /workspace",
        "--entrypoint /bin/bash",
        "video-lab:runtime",
    )
    for argument in required_arguments:
        assert argument in docker_block, f"direct-run command is missing {argument!r}"

    forbidden_arguments = (
        "--privileged",
        "--ipc=host",
        "--cap-add",
        "PIP_USER",
        "PYTHONUSERBASE",
    )
    for argument in forbidden_arguments:
        assert argument not in docker_block, f"direct-run command contains {argument!r}"

    release = json.loads(Path("profiles/releases/stable.json").read_text())
    swr_torch = (
        "swr.cn-east-3.myhuaweicloud.com/hellcat-home/"
        "strix-halo-rocm-pytorch@" + release["torch"]["manifest_digest"]
    )
    assert swr_torch in text
    assert ".cache" in Path("templates/project/.dockerignore").read_text()
```

- [ ] **Step 2: Run the updated contract and confirm it fails against the old README**

Run:

```bash
PYENV_VERSION=3.10.19 pytest -q \
  tests/container/test_image_contract.py::test_readme_quick_start_is_ordered_complete_and_safe
```

Expected: FAIL because the old quick start has no `docker image inspect video-lab:runtime` and still invokes `strix-halo-rocm project run`.

### Task 2: Make Direct Docker The Single Quick-Start Exit

**Files:**
- Modify: `README.md:10-180`
- Modify: `README.md:636-720`
- Test: `tests/container/test_image_contract.py`

- [ ] **Step 1: Replace the quick-start introduction and steps 5-7**

Keep current steps 1-4, but replace the sentence immediately below `## 快速开始` with:

```markdown
本路径从一台全新的 **Ubuntu 24.04.x AMD64 + AMD Ryzen AI Max+ 395 / Radeon 8060S** 主机开始。完成后将得到项目派生镜像 `video-lab:runtime`，并只用标准 `docker run` 挂载一个业务目录、映射 GPU 和进入 Bash。
```

Replace quick-start steps 5-7 with the following content:

````markdown
### 5. 确认 GPU 设备和项目镜像

安装器全部阶段通过后，确认宿主 GPU 设备节点和项目派生镜像都存在：

**宿主机：**

```bash
PROJECT="$HOME/ai-projects/video-lab"

ls -l /dev/kfd /dev/dri/render*
docker image inspect video-lab:runtime \
  --format 'image={{join .RepoTags ","}} id={{.Id}}'
```

必须看到 `/dev/kfd`、至少一个 `/dev/dri/renderD*`，以及
`video-lab:runtime` 的镜像 ID。若 Docker 报权限错误，先注销并重新登录桌面或
SSH，让安装器添加的 `docker` 组生效，再从本步骤继续；不要改用
`sudo strix-halo-rocm`。

### 6. 使用标准 Docker run 启动

以下是本文唯一的完整直启命令。它只挂载 `$PROJECT` 到 `/workspace`，不会默认
共享模型、Hugging Face 缓存或其他项目数据：

**宿主机：**

```bash
PROJECT="$(realpath "$HOME/ai-projects/video-lab")"
RENDER_NODE="$(find /dev/dri -maxdepth 1 -type c -name 'renderD*' | sort | head -n 1)"

if [[ ! -c /dev/kfd || -z "$RENDER_NODE" ]]; then
  echo "未发现可用的 AMD GPU 设备节点，请先完成宿主机安装与重启。" >&2
else
  docker run --rm -it \
    --device /dev/kfd \
    --device /dev/dri \
    --group-add "$(stat -c '%g' /dev/kfd)" \
    --group-add "$(stat -c '%g' "$RENDER_NODE")" \
    --user "$(id -u):$(id -g)" \
    --ipc=private \
    --shm-size=16g \
    --env HOME=/workspace \
    --env PIP_TARGET=/workspace/.cache/python-site \
    --env PYTHONPATH=/workspace/.cache/python-site:/workspace \
    --env PATH=/workspace/.cache/python-site/bin:/opt/venv/bin:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    --mount "type=bind,src=$PROJECT,dst=/workspace" \
    --workdir /workspace \
    --entrypoint /bin/bash \
    video-lab:runtime
fi
```

看到容器中的 Bash 提示符即表示标准 Docker 启动成功。参数含义、公开父镜像
替换方式和直启模式边界见[极简 Docker 直启](#极简-docker-直启)。

### 7. 验证 GPU 并安装 Python 依赖

ROCm 版 PyTorch 仍使用 `torch.cuda` API；这里的 `cuda:0` 是 PyTorch 设备
接口，不表示安装了 NVIDIA CUDA。

**项目容器内：**

```bash
python - <<'PY'
import torch

print(f"torch={torch.__version__}")
print(f"hip={torch.version.hip}")
assert torch.version.hip, "PyTorch is not a ROCm build"
assert torch.cuda.is_available(), "ROCm GPU is not available to PyTorch"

device = torch.device("cuda:0")
print(f"gpu={torch.cuda.get_device_name(0)}")
left = torch.randn((1024, 1024), device=device)
right = torch.randn((1024, 1024), device=device)
result = left @ right
torch.cuda.synchronize()
print(f"GPU_OK device={result.device} mean={result.mean().item():.6f}")
PY

python -m pip install transformers safetensors
python -m pip check
```

必须看到非空 HIP 版本、Radeon GPU 名称和 `GPU_OK device=cuda:0`。普通包
保存在业务目录的 `.cache/python-site`，删除并重新创建容器后仍然存在；项目
模板已从 Docker 构建上下文排除 `.cache`。

直启模式允许 `pip` 覆盖任何 Python 包，包括 Torch。这里只安装普通业务
依赖，不要把 `torch`、`torchvision`、`torchaudio` 或 `triton` 加入这条命令。
需要启动前自动 GPU 检查、Torch 保护、overlay、依赖锁或修复能力时，改用
[创建和运行项目](#创建和运行项目)中的托管工作流。

至此已经从全新 Ubuntu 24.04 主机完成 ROCm 7.2.1 / PyTorch 2.9.1 平台安装、
项目派生镜像创建、标准 Docker GPU 映射、真实矩阵运算验证和普通 Python 依赖
安装。
````

- [ ] **Step 2: Replace the detailed direct-run section with explanations only**

Replace everything from `## 极简 Docker 直启` through the line before `## 创建和运行项目` with:

````markdown
## 极简 Docker 直启

完整命令见[快速开始第 6 步](#6-使用标准-docker-run-启动)，真实 GPU 探针与
普通依赖安装见[第 7 步](#7-验证-gpu-并安装-python-依赖)。本节只解释参数和
适用边界，避免维护第二份可能漂移的启动命令。

| 参数 | 作用 |
| --- | --- |
| `--device /dev/kfd` | 映射 ROCm 计算设备 |
| `--device /dev/dri` | 映射 DRM render 设备目录 |
| 两个 `--group-add` | 按宿主设备的数字 GID 授予访问权限，不依赖容器内组名 |
| `--user` | 让容器以当前宿主 UID/GID 写业务目录 |
| `--ipc=private`、`--shm-size=16g` | 保持私有 IPC，并为大型 AI 任务提供明确共享内存 |
| `--mount`、`--workdir` | 只把业务目录挂载到 `/workspace` 并作为工作目录 |
| `PIP_TARGET`、`PYTHONPATH` | 把普通 Python 包持久化到 `.cache/python-site` 并立即加入导入路径 |
| `PATH` | 优先使用业务目录脚本、镜像 Python venv 和 ROCm 工具链 |
| `--entrypoint /bin/bash` | 绕过项目策略入口，直接进入交互 Bash |

快速开始使用安装器生成的 `video-lab:runtime`，其中包含该项目构建时的业务
依赖。只需要正式 ROCm/PyTorch 父环境时，把完整命令最后一行的镜像替换为：

```text
swr.cn-east-3.myhuaweicloud.com/hellcat-home/strix-halo-rocm-pytorch@sha256:dc0bb217474cfd4f602423bd3bf4fe8714b03e900cf3c6b4417b99e622ebcf8b
```

该公开 SWR 引用锁定 stable manifest digest，匿名即可拉取；父镜像不包含项目
派生镜像中的业务依赖。

直启模式有意不提供以下保证：

- 不初始化或切换项目 overlay；
- 不验证项目镜像是否需要重建；
- 不保护 Torch、TorchVision、TorchAudio 或 Triton；
- 不在进入 Bash 前执行 GPU 运算门禁；
- 不自动诊断、修复或生成跨主机部署状态。

镜像本身保持不可变，但挂载目录中的 `.cache/python-site` 可以遮蔽镜像包。
需要上述保证时使用[创建和运行项目](#创建和运行项目)中的
`strix-halo-rocm project run`；直启命令不需要 `--privileged`、`--ipc=host`
或额外 capability。
````

- [ ] **Step 3: Run the focused README contract tests**

Run:

```bash
PYENV_VERSION=3.10.19 pytest -q \
  tests/container/test_image_contract.py::test_operator_documentation_contains_required_contract_anchors \
  tests/container/test_image_contract.py::test_readme_quick_start_is_ordered_complete_and_safe \
  tests/container/test_image_contract.py::test_readme_bash_examples_are_valid_shell
```

Expected: `3 passed`.

- [ ] **Step 4: Commit the documentation behavior and contract together**

```bash
git add README.md tests/container/test_image_contract.py
git commit -m "docs: make Docker run the primary quick start"
```

### Task 3: Verify Structure, Identity, And Scope

**Files:**
- Verify: `README.md`
- Verify: `tests/container/test_image_contract.py`
- Verify: `profiles/releases/stable.json`
- Verify: `templates/project/.dockerignore`

- [ ] **Step 1: Check heading and command uniqueness**

Run:

```bash
test "$(rg -c '^## 快速开始$' README.md)" -eq 1
test "$(rg -c 'docker run --rm -it' README.md)" -eq 1
```

Expected: both commands exit 0 with no output.

- [ ] **Step 2: Check the quick start does not invoke the managed runner**

Run:

```bash
sed -n '/^## 快速开始$/,/^## 目录$/p' README.md | \
  rg 'strix-halo-rocm project run' && exit 1 || true
```

Expected: no output.

- [ ] **Step 3: Check the exact SWR Torch identity matches the release manifest**

Run:

```bash
EXPECTED="$(python3 -c 'import json; print(json.load(open("profiles/releases/stable.json"))["torch"]["manifest_digest"])')"
ACTUAL="$(rg -o 'swr\.cn-east-3\.myhuaweicloud\.com/hellcat-home/strix-halo-rocm-pytorch@sha256:[0-9a-f]{64}' README.md | sort -u)"
test "$ACTUAL" = "swr.cn-east-3.myhuaweicloud.com/hellcat-home/strix-halo-rocm-pytorch@$EXPECTED"
```

Expected: exit 0 with no output.

- [ ] **Step 4: Check Markdown diff cleanliness and change scope**

Run:

```bash
git diff --check HEAD^..HEAD
git show --stat --oneline HEAD
git status --short
```

Expected: `git diff --check` succeeds; the commit contains only `README.md` and `tests/container/test_image_contract.py`; worktree status is clean.

- [ ] **Step 5: Review the final README flow as rendered source**

Run:

```bash
sed -n '1,220p' README.md
sed -n '/^## 极简 Docker 直启$/,/^## 创建和运行项目$/p' README.md
```

Expected: one seven-step zero-to-Docker flow, one full direct-run command, no duplicate GPU probe in the direct-run detail section, and the managed workflow still begins under `创建和运行项目`.
