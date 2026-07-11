# README Quick Start Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize README so a first-time Ryzen AI Max+ 395 user can install the toolkit, resume after reboot, start the managed Docker project, prove PyTorch GPU execution, and begin installing protected project dependencies without leaving the quick-start path.

**Architecture:** Add one canonical, task-oriented quick start immediately after the release baseline and move the table of contents below it. Keep detailed reference sections, but replace the old partial quick-install section with supplemental interactive-installer and launcher reference. Add a documentation contract test for ordering, required commands, execution-context labels, and the non-root launcher boundary.

**Tech Stack:** Markdown, Bash command examples, Python 3.12, pytest 8.4, markdownlint-cli 0.45.0

---

## Implementation Tasks

### Task 1: Define the README onboarding contract

**Files:**

- Modify: `tests/container/test_image_contract.py`
- Test: `tests/container/test_image_contract.py`

- [x] **Step 1: Add the failing quick-start contract test**

Add this test below `test_operator_documentation_contains_required_contract_anchors`:

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

    assert re.search(
        r"(?m)^\s*sudo strix-halo-rocm(?:\s|$)", quick_text
    ) is None
```

- [x] **Step 2: Run the new test and verify the current README fails the contract**

Run:

```bash
PYTHONPATH=src /app/imgMaker/.venv/bin/python -m pytest \
  tests/container/test_image_contract.py::test_readme_quick_start_is_ordered_complete_and_safe \
  -q
```

Expected: `FAIL` with `README.md is missing '## 快速开始'`.

### Task 2: Build the canonical end-to-end quick start

**Files:**

- Modify: `README.md`
- Test: `tests/container/test_image_contract.py`

- [x] **Step 1: Move the table of contents below the new quick start**

Keep the title, product description, and four-line release baseline at the top. Remove the current `## 目录` block from that position and reinsert it after the complete quick-start section. Update its installation entries to:

```markdown
- [快速开始](#快速开始)
- [项目解决什么问题](#项目解决什么问题)
- [支持范围与固定基线](#支持范围与固定基线)
- [安装前准备](#安装前准备)
- [交互安装与 launcher](#交互安装与-launcher)
- [安装模式与自动化](#安装模式与自动化)
```

Keep the remaining existing entries in their current order.

- [x] **Step 2: Insert the exact quick-start workflow before the table of contents**

Add a `## 快速开始` section containing these seven focused subsections and commands.

Start with the scope and prerequisites:

````markdown
本路径适用于 **Ubuntu 24.04.x AMD64 + AMD Ryzen AI Max+ 395 / Radeon 8060S**。完成后将得到一个已经通过真实 GPU 运算检查的独立项目容器，并可直接安装自己的 Python 依赖。

### 1. 开始前确认

- BIOS/UEFI 的 UMA Frame Buffer 使用主板允许的最小值，建议 512 MiB；
- 当前用户具有 `sudo` 权限，网络可访问 GitHub 和 GHCR；
- Docker 数据目录建议至少有 40 GiB 可用空间；
- 宿主具有 `git` 和 `python3.12`。

缺少基础工具时执行：

**宿主机：**

```bash
sudo apt update
sudo apt install -y git python3.12
```
````

Continue with fixed checkout paths and the auditable clone:

````markdown
### 2. 获取固定版本

**宿主机：**

```bash
TOOLKIT="$HOME/src/strix-halo-rocm-toolkit"
PROJECT="$HOME/ai-projects/video-lab"

mkdir -p "$(dirname "$TOOLKIT")"
git clone --branch v0.2.3 --depth 1 \
  https://github.com/hellcatjack/strix-halo-rocm-toolkit.git \
  "$TOOLKIT"
cd "$TOOLKIT"
```

不要通过 `curl | sudo bash` 执行安装器。固定 Git checkout 让安装来源和恢复状态可以审计。
````

Use one explicit and repeatable full-install command:

````markdown
### 3. 安装平台并创建第一个项目

**宿主机：**

```bash
./install.sh --mode full \
  --project-dir "$PROJECT" \
  --project-name video-lab \
  --image-source pull
```

审阅 host plan 后，只有精确输入 `APPLY` 才会修改宿主。Docker 组授权会单独询问。安装器从公开 GHCR 匿名拉取固定 digest，并实时显示阶段、下载进度和私有日志路径。
````

Show the reboot and retry branch without changing mode or state:

````markdown
### 4. 按提示重启或恢复

安装器显示 `ACTION` 和 `RESUME` 要求重启时执行：

**宿主机：**

```bash
sudo reboot
```

重启后重新打开终端，恢复路径变量并原样重跑安装命令：

```bash
TOOLKIT="$HOME/src/strix-halo-rocm-toolkit"
PROJECT="$HOME/ai-projects/video-lab"
cd "$TOOLKIT"

./install.sh --mode full \
  --project-dir "$PROJECT" \
  --project-name video-lab \
  --image-source pull
```

不要删除安装状态，也不要切换为 `container` 模式。普通失败先读取 `CAUSE`、`LOG` 和 `RESUME`，修复原因后仍执行同一条命令；可信阶段会显示为 `SKIP`。
````

Start the managed Docker project as the ordinary user:

````markdown
### 5. 启动项目 Docker

**宿主机：**

```bash
export PATH="$HOME/.local/bin:$PATH"
strix-halo-rocm --version
strix-halo-rocm project run "$PROJECT"
```

如果当前用户不能直接访问 Docker，先运行 `sudo -v` 刷新凭据，再执行同一条 `project run`；不要使用 `sudo strix-halo-rocm`。

`project run` 会按需构建项目镜像并映射 GPU 设备。进入 Bash 前，项目入口会自动校验 Torch manifest、项目 overlay、ROCm GPU 识别和真实 GPU 运算；任何一项失败都不会进入项目 shell。

如果自动检查失败，不要继续安装 Python 依赖。回到宿主机运行 `strix-halo-rocm doctor "$PROJECT"`，按报告修复 GPU、镜像或 overlay 问题后再启动项目。
````

Prove PyTorch is executing on the GPU, with an explicit note about ROCm API naming:

````markdown
### 6. 验证 PyTorch GPU

看到项目 Bash 提示符后运行以下探针。ROCm 版 PyTorch 仍使用 `torch.cuda` API；这里的 `cuda:0` 表示 PyTorch 的设备接口，不表示安装了 NVIDIA CUDA。

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
```

必须看到非空 HIP 版本、Radeon GPU 名称和 `GPU_OK device=cuda:0`。仅能 `import torch` 不算 GPU 验证通过。
````

End at an immediately usable protected pip environment and show the reproducible promotion path:

````markdown
### 7. 开始构建 Python 环境

GPU 验证通过后，可以直接试装普通项目依赖：

**项目容器内：**

```bash
pip install transformers safetensors
pip check
```

直接安装的依赖保存在当前项目的持久 overlay 中，退出临时容器后仍然存在。受保护的 `torch`、`torchvision`、`torchaudio` 和 `triton` 不能被项目安装覆盖；不要再创建一份包含 Torch 的 venv。确实需要其他 Torch 组合时，使用[自选 PyTorch 版本](#自选-pytorch-版本)创建完整父镜像 profile。

试装稳定后，把直接依赖写入项目的 `requirements.in`，然后退出容器：

```bash
printf '%s\n' 'transformers' 'safetensors' > requirements.in
exit
```

**宿主机：**

```bash
strix-halo-rocm project lock "$PROJECT"
strix-halo-rocm project run "$PROJECT" --build
```

至此已经具备可运行的 ROCm 7.2.1 / PyTorch 2.9.1 GPU 容器、独立项目目录、受保护的 Torch 基线和可继续扩展的 Python 环境。依赖试装、卸载和固化的完整规则见[Python 依赖与受保护 pip](#python-依赖与受保护-pip)。
````

- [x] **Step 3: Replace the old partial quick-install section with supplemental reference**

Rename `## 快速安装` to `## 交互安装与 launcher`. Remove its duplicate fixed-version clone subsection. Keep the existing interactive menu behavior, stable image identity explanation, runtime installation paths, PATH export, non-root launcher warning, and installed-runtime `cd` fallback under these two headings:

```markdown
### 交互式安装器

### launcher 与安装运行时
```

The canonical full-install command must remain only in the top quick start and `## 安装模式与自动化`; this supplemental section must not introduce a second abbreviated installation path.

- [x] **Step 4: Run the contract test and Markdown lint**

Run:

```bash
PYTHONPATH=src /app/imgMaker/.venv/bin/python -m pytest \
  tests/container/test_image_contract.py::test_operator_documentation_contains_required_contract_anchors \
  tests/container/test_image_contract.py::test_readme_quick_start_is_ordered_complete_and_safe \
  -q
npx --yes markdownlint-cli@0.45.0 README.md --disable MD013
```

Expected: `2 passed`; markdownlint exits `0` with no output.

- [x] **Step 5: Commit the README workflow and contract**

```bash
git add README.md tests/container/test_image_contract.py
git commit -m "docs: add end-to-end GPU quick start"
```

### Task 3: Audit commands, links, and repository contracts

**Files:**

- Modify: `README.md`
- Modify: `tests/container/test_image_contract.py`
- Verify: `install.sh`
- Verify: `src/amd_ai/cli.py`
- Verify: `templates/project/project-entrypoint`
- Verify: `docs/superpowers/specs/2026-07-10-readme-quickstart-design.md`

- [x] **Step 1: Verify local README links resolve to repository files**

Run:

```bash
python3.12 - <<'PY'
import re
from pathlib import Path

root = Path.cwd()
text = (root / "README.md").read_text(encoding="utf-8")
targets = re.findall(r"\[[^]]+\]\(([^)#]+)(?:#[^)]+)?\)", text)
missing = sorted(
    target
    for target in targets
    if "://" not in target and not (root / target).exists()
)
assert not missing, f"missing local README links: {missing}"
print(f"README_LINKS_OK checked={len(targets)}")
PY
```

Expected: `README_LINKS_OK` and no assertion failure.

- [x] **Step 2: Verify documented parser commands and documentation contracts**

Run:

```bash
PYTHONPATH=src /app/imgMaker/.venv/bin/python -m amd_ai.cli --help >/dev/null
PYTHONPATH=src /app/imgMaker/.venv/bin/python -m amd_ai.cli project run --help >/dev/null
PYTHONPATH=src /app/imgMaker/.venv/bin/python -m pytest \
  tests/cli/test_installer_commands.py \
  tests/test_version.py \
  tests/container/test_image_contract.py::test_operator_documentation_contains_required_contract_anchors \
  tests/container/test_image_contract.py::test_readme_quick_start_is_ordered_complete_and_safe \
  tests/container/test_image_contract.py::test_readme_bash_examples_are_valid_shell \
  -q
```

Expected: both help commands exit `0`; pytest reports `15 passed`.

- [x] **Step 3: Run final formatting and whitespace checks**

Run:

```bash
npx --yes markdownlint-cli@0.45.0 \
  README.md \
  docs/superpowers/specs/2026-07-10-readme-quickstart-design.md \
  docs/superpowers/plans/2026-07-10-readme-quickstart.md \
  --disable MD013
git diff --check main...HEAD
git status --short
```

Expected: markdownlint and `git diff --check` exit `0`; status contains no unexpected files.

- [x] **Step 4: Persist the README Bash syntax audit**

Add a contract test that extracts every `bash` fence and passes it to `bash -n` without executing it:

```python
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
```

The RED run must identify the invalid `<自定义PROFILE_ID>` and `<新发布标签>` shell redirections. Replace them with quoted `CUSTOM_PROFILE_ID` and `RELEASE_TAG` variables, then rerun the test and expect `1 passed`.

- [x] **Step 5: Review the final diff against the approved design**

Run:

```bash
git diff --stat main...HEAD
git diff main...HEAD -- README.md tests/container/test_image_contract.py
```

Confirm all of the following before completion:

- the quick start appears before the table of contents;
- every code block is labeled as host or project-container context;
- reboot recovery repeats the exact full-mode command;
- project startup explains the mandatory automatic GPU check;
- the manual probe performs a synchronized GPU matrix multiplication;
- ordinary pip usage and protected Torch behavior are both explicit;
- no image digest, release manifest, installer behavior, or project template changed.
