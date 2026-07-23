# Simple Direct Docker Run Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a copyable, local-only standard `docker run` workflow for the stable PyTorch parent image and existing project-derived images.

**Architecture:** Documentation only. README will present one canonical Bash command that maps the Radeon devices and one business directory, bypasses the derived-image policy entrypoint, and persists ordinary pip user packages under that directory; managed `project run` remains documented as the protected alternative.

**Tech Stack:** Markdown, Bash, Docker CLI, Python 3.12 standard library for documentation checks.

---

### Task 1: Document the simple direct Docker workflow

**Files:**
- Modify: `README.md:170-185`
- Modify: `README.md:596-635`
- Modify: `README.md:970-985`

- [ ] **Step 1: Confirm the direct-run section is absent**

Run:

```bash
if rg -q '^## 极简 Docker 直启$' README.md; then
  echo 'unexpected existing section' >&2
  exit 1
fi
```

Expected: exit code `0` with no output.

- [ ] **Step 2: Add the table-of-contents entry**

Add this item after `安装后验证`:

```markdown
- [极简 Docker 直启](#极简-docker-直启)
```

- [ ] **Step 3: Add the complete direct-run section**

Insert the following section after the canonical SWR/GHCR image references and
before `创建和运行项目`:

````markdown
## 极简 Docker 直启

本节面向只需要在本机启动容器、挂载一个业务目录并直接使用 GPU 的用户。
它不使用项目 overlay、受保护 pip、启动时 Torch 校验、自动修复或跨主机部署。
需要这些能力时继续使用 [`strix-halo-rocm project run`](#创建和运行项目)。

宿主必须已经存在 `/dev/kfd`、至少一个 `/dev/dri/renderD*`，并且当前用户
可以执行 Docker。派生镜像还必须已经通过安装器或 `project run --build`
构建完成。

### 直接进入项目派生镜像

把 `PROJECT` 和 `IMAGE` 改成实际项目目录与派生镜像名：

```bash
PROJECT="$(realpath "$HOME/ai-projects/video-lab")"
IMAGE="video-lab:runtime"
RENDER_NODE="$(find /dev/dri -maxdepth 1 -type c -name 'renderD*' | sort | head -n 1)"

test -c /dev/kfd
test -n "$RENDER_NODE"

docker run --rm -it \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add "$(stat -c '%g' /dev/kfd)" \
  --group-add "$(stat -c '%g' "$RENDER_NODE")" \
  --user "$(id -u):$(id -g)" \
  --ipc=private \
  --shm-size=16g \
  --env HOME=/workspace \
  --env PYTHONUSERBASE=/workspace/.python-user \
  --env PIP_USER=1 \
  --env PATH=/workspace/.python-user/bin:/opt/venv/bin:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  --mount "type=bind,src=$PROJECT,dst=/workspace" \
  --workdir /workspace \
  --entrypoint /bin/bash \
  "$IMAGE"
```

`--entrypoint /bin/bash` 会绕过派生镜像的策略入口点，这是本模式的预期行为。
容器不会自动检查 Torch、GPU 或项目状态，也不会阻止用户替换 Torch。

### 直接进入公开 PyTorch 父镜像

使用同一条命令，只替换镜像变量：

```bash
IMAGE="swr.cn-east-3.myhuaweicloud.com/hellcat-home/strix-halo-rocm-pytorch@sha256:dc0bb217474cfd4f602423bd3bf4fe8714b03e900cf3c6b4417b99e622ebcf8b"
```

父镜像只预装 ROCm、Python 和锁定的 PyTorch 栈，不包含项目派生镜像中的
业务依赖。

### 安装普通 Python 包

进入容器后直接执行：

```bash
pip install transformers safetensors
```

普通包写入业务目录的 `.python-user`，重新创建容器后仍然存在。该目录可以
覆盖镜像中的 Python 包，包括 Torch；需要受保护基线时不要使用本模式。

### 手工验证 GPU

```bash
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("hip:", torch.version.hip)
print("available:", torch.cuda.is_available())

x = torch.randn((1024, 1024), device="cuda")
y = x @ x
torch.cuda.synchronize()
print("GPU_OK:", y.device)
PY
```

成功时必须看到非空 HIP 版本、`available: True` 和 `GPU_OK: cuda:0`。
这条命令不需要 `--privileged`、`--ipc=host` 或额外 capability。
````

- [ ] **Step 4: Add focused troubleshooting entries**

Add these rows to the existing troubleshooting table:

```markdown
| 极简 Docker 直启时 `rocminfo` 或 Torch 无权访问 GPU | 确认命令同时包含 `/dev/kfd`、`/dev/dri` 和由 `stat -c '%g'` 得到的两个数字 `--group-add`；不要用固定的 `video`/`render` 组名代替宿主实际 GID |
| 极简模式中的 `pip` 显示 protected pip 或 overlay 错误 | 使用文档中的完整 `PATH`、`PYTHONUSERBASE` 和 `PIP_USER` 参数重新创建容器；该组合会选择 `/opt/venv/bin/pip` 并写入业务目录 `.python-user` |
| 极简模式创建的文件无法由宿主修改 | 保留 `--user "$(id -u):$(id -g)"`，并确认业务目录允许当前宿主用户写入 |
```

- [ ] **Step 5: Review the rendered section boundaries**

Run:

```bash
sed -n '/^## 极简 Docker 直启$/,/^## 创建和运行项目$/p' README.md
```

Expected: one complete direct-run section followed immediately by the
`创建和运行项目` heading.

### Task 2: Validate and publish the documentation change

**Files:**
- Verify: `README.md`
- Verify: `profiles/releases/stable.json`

- [ ] **Step 1: Parse-check the documented Bash command**

Run:

```bash
python3 - <<'PY'
import re
import subprocess
from pathlib import Path

readme = Path("README.md").read_text(encoding="utf-8")
section = readme.split("## 极简 Docker 直启", 1)[1].split(
    "## 创建和运行项目", 1
)[0]
blocks = re.findall(r"```bash\n(.*?)\n```", section, flags=re.DOTALL)
assert blocks, "direct-run Bash block is missing"
subprocess.run(["bash", "-n"], input=blocks[0], text=True, check=True)
PY
```

Expected: exit code `0` with no output.

- [ ] **Step 2: Verify the SWR digest matches the stable manifest**

Run:

```bash
python3 - <<'PY'
import json
from pathlib import Path

readme = Path("README.md").read_text(encoding="utf-8")
release = json.loads(Path("profiles/releases/stable.json").read_text())
reference = (
    "swr.cn-east-3.myhuaweicloud.com/hellcat-home/"
    "strix-halo-rocm-pytorch@" + release["torch"]["manifest_digest"]
)
assert reference in readme
PY
```

Expected: exit code `0` with no output.

- [ ] **Step 3: Verify required and forbidden Docker flags**

Run:

```bash
python3 - <<'PY'
from pathlib import Path

readme = Path("README.md").read_text(encoding="utf-8")
section = readme.split("## 极简 Docker 直启", 1)[1].split(
    "## 创建和运行项目", 1
)[0]
for required in (
    "--device /dev/kfd",
    "--device /dev/dri",
    "--group-add",
    '--user "$(id -u):$(id -g)"',
    "--shm-size=16g",
    "--entrypoint /bin/bash",
):
    assert required in section, required
for forbidden in ("--privileged", "--ipc=host", "--cap-add"):
    assert forbidden not in section.split("这条命令不需要", 1)[0], forbidden
PY
```

Expected: exit code `0` with no output.

- [ ] **Step 4: Check Markdown diff hygiene**

Run:

```bash
git diff --check
git diff -- README.md
```

Expected: `git diff --check` exits `0`; the diff contains only the approved
README documentation changes.

- [ ] **Step 5: Commit the README update**

```bash
git add README.md
git commit -m "docs: add simple direct Docker workflow"
```
