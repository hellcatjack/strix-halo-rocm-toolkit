# Strix Halo ROCm Toolkit

面向 **AMD Ryzen AI Max+ 395 / Radeon 8060S (`gfx1151`)** 的可恢复 ROCm 容器开发平台。它提供 Ubuntu 宿主机准备、公开且内容寻址的 ROCm/PyTorch 镜像、独立项目容器、受保护的 `pip` 工作流，以及可审计的 GPU 验证和精确修复。

> - 当前工具包版本：`v0.2.1`
> - 正式软件基线：ROCm 7.2.1、Python 3.12、PyTorch 2.9.1
> - 正式宿主适配器：Ubuntu 24.04.x AMD64
> - Stable 镜像 release ID：`0.2.0`（`v0.2.1` 不重建镜像）

## 目录

- [项目解决什么问题](#项目解决什么问题)
- [支持范围与固定基线](#支持范围与固定基线)
- [安装前准备](#安装前准备)
- [快速安装](#快速安装)
- [安装模式与自动化](#安装模式与自动化)
- [安装后验证](#安装后验证)
- [创建和运行项目](#创建和运行项目)
- [Python 依赖与受保护 pip](#python-依赖与受保护-pip)
- [模型、缓存和数据挂载](#模型缓存和数据挂载)
- [ComfyUI 与大型视频应用](#comfyui-与大型视频应用)
- [自选 PyTorch 版本](#自选-pytorch-版本)
- [Doctor 与 Repair](#doctor-与-repair)
- [升级、磁盘与清理](#升级磁盘与清理)
- [常见故障](#常见故障)
- [安全边界](#安全边界)
- [命令索引](#命令索引)
- [专项文档](#专项文档)
- [开发与验证](#开发与验证)

## 项目解决什么问题

Strix Halo 使用统一物理内存，宿主内核驱动、TTM/GTT、容器设备权限和 ROCm 用户态必须一起正确配置。直接在宿主机反复安装 ROCm，或让每个 Python venv 各自安装数 GB 的 Torch wheel，容易造成版本漂移、磁盘浪费和难以恢复的环境。

本项目采用以下边界：

- 宿主机只使用 Ubuntu OEM 内核自带的 `amdgpu`，不安装完整 ROCm 用户态，也不安装 `amdgpu-dkms`。
- ROCm、Python 开发环境和 PyTorch 都位于容器镜像中。
- 所有项目共享同一个不可变 PyTorch 父镜像层，但每个项目拥有独立源码、依赖层、配置和容器。
- 项目不创建第二份包含 Torch 的 venv；Docker 内容寻址存储只保存一份相同父层。
- 运行中的项目容器允许直接 `pip install` 普通依赖，但不能替换受保护的 Torch 组合。
- 依赖试装写入项目私有、可事务切换的 overlay；失败不会破坏上一个可用 generation。
- Stable 镜像按公开 GHCR 的 exact digest 拉取并校验，不依赖维护者登录状态。
- 项目不预装 ComfyUI，也不默认共享模型、Hugging Face、输入、输出或编译缓存。

镜像关系如下：

```text
Ubuntu 24.04 + ROCm 7.2.1 + Python 3.12
  └── PyTorch 2.9.1 + TorchVision/TorchAudio/Triton
        ├── project-a:runtime（项目 A 依赖与源码）
        ├── project-b:runtime（项目 B 依赖与源码）
        └── project-c:runtime（项目 C 依赖与源码）
```

## 支持范围与固定基线

| 项目 | 当前状态 |
| --- | --- |
| CPU / GPU | AMD Ryzen AI Max+ 395 / Radeon 8060S |
| GPU 架构 | `gfx1151` |
| 宿主自动写入 | Ubuntu 24.04.x AMD64 |
| 其他 Linux | 可执行只读预检；没有专用适配器时拒绝修改宿主 |
| 宿主 GPU 驱动 | Ubuntu OEM 内核内置 `amdgpu` |
| 容器用户态 | Ubuntu 24.04、ROCm 7.2.1 |
| Python | 3.12；安装器同样要求宿主存在 `python3.12` |
| PyTorch | 2.9.1 |
| TorchVision | 0.24.0 |
| TorchAudio | 2.9.0 |
| Triton | 3.5.1 |
| 容器运行时 | Docker Engine，支持 Buildx；完整模式可安装 |
| Stable release ID | `0.2.0` |

OCI 镜像可以运行在满足内核、`/dev/kfd`、DRM render node 和容器设备映射要求的其他 Linux 主机上，但这不等于宿主自动配置已获得支持。当前只有 Ubuntu 24.04.x AMD64 可以执行正式的 `host-prepare apply`。

## 安装前准备

### 1. BIOS/UEFI

将 **UMA Frame Buffer / Dedicated VRAM 设置为主板允许的最小值，建议 512 MiB（0.5 GiB）**。这样只保留必要的固定帧缓冲，其余物理内存由系统和 GPU 通过 TTM/GTT 按需使用。

脚本不能修改 BIOS，也不会在启动时永久预留全部 GTT 上限。完整模式会根据 DMI 和 `/proc/meminfo` 识别物理内存，计算 `ttm.pages_limit`；检测结果冲突时停止自动应用并要求人工确认。

### 2. 宿主条件

正式完整安装需要：

- Ubuntu 24.04.x Desktop 或 Server，AMD64；
- Ryzen AI Max+ 395 / Radeon 8060S；
- 可用的 `sudo` 权限；
- `git`、`bash`、`python3.12` 和互联网连接；
- Docker 数据目录有足够空间。

Ubuntu 24.04 默认提供 Python 3.12。若缺失：

```bash
sudo apt update
sudo apt install -y git python3.12
```

首次拉取和展开完整镜像时，建议 Docker 数据目录至少预留 40 GiB。安装器会按缺失层计算实际需求，并额外要求 5 GiB 安全余量；空间不足时会在拉取或构建前停止。

### 3. 已有 ROCm 6.4 或失败驱动安装

不要继续在宿主 Python 中叠加 ROCm 或 Torch。完整模式会识别并处理确认属于 `repo.radeon.com` 的 ROCm 6.4 软件源、白名单用户态包和 `amdgpu-dkms`，但不会执行 `apt autoremove`，也不会删除无法确认来源的 DKMS 模块或混合 APT 源。

在任何写入前先执行只读检查：

```bash
git clone --branch v0.2.1 --depth 1 \
  https://github.com/hellcatjack/strix-halo-rocm-toolkit.git
cd strix-halo-rocm-toolkit
mkdir -p reports
./bin/host-preflight --json reports/preflight.json
```

预检退出码：

| 退出码 | 含义 |
| --- | --- |
| `0` | 通过，或明确标记为尚未验证但没有阻断 |
| `1` | 需要修改或重启 |
| `2` | 不支持的系统、错误 GPU/驱动等阻断状态 |

## 快速安装

### 1. 获取固定版本

```bash
git clone --branch v0.2.1 --depth 1 \
  https://github.com/hellcatjack/strix-halo-rocm-toolkit.git
cd strix-halo-rocm-toolkit
git rev-parse --verify HEAD
```

不要通过 `curl | sudo bash` 运行安装器。安装脚本必须来自本地 Git checkout，以便记录源码 revision、复制版本化运行时并校验本地构建输入。

### 2. 启动交互式安装器

```bash
./install.sh
```

交互首页提供：

1. **完整工作站安装**：检查并准备 Ubuntu 内核、旧 ROCm 残留、Docker、设备组和 TTM/GTT，然后部署镜像与第一个项目。
2. **仅容器平台**：不修改内核、APT、固件、用户组或 Docker，只校验现有宿主并部署镜像与项目。
3. **Doctor/Repair**：检查现有平台或项目，并在明确确认后执行精确修复。

Stable 默认从公开 GHCR 匿名拉取固定 digest。只有网络获取失败或镜像不存在时，交互模式才会询问是否从当前干净 checkout 本地构建。digest、config ID、OCI 标签或内嵌锁不一致时直接阻断，不会静默回退。

### 3. 让 launcher 进入 PATH

安装器把版本化运行时安装到：

```text
~/.local/share/strix-halo-rocm-toolkit/releases/
```

并创建：

```text
~/.local/bin/strix-halo-rocm
```

若当前 shell 找不到命令：

```bash
export PATH="$HOME/.local/bin:$PATH"
strix-halo-rocm --version
```

把同一条 `export` 加入用户自己的 shell profile 即可，不要用 `sudo` 运行 launcher。

`doctor`、`repair` 和 `release verify` 的默认 manifest 路径相对于当前目录。本文示例默认当前目录是源码 checkout；如果源码已经删除，可进入已安装运行时后再使用默认路径：

```bash
cd "$HOME/.local/share/strix-halo-rocm-toolkit/current"
```

## 安装模式与自动化

### 完整工作站模式

```bash
./install.sh --mode full \
  --project-dir "$HOME/ai-projects/video-lab" \
  --project-name video-lab
```

完整模式先展示固定 host plan。只有精确输入 `APPLY` 才会开始写入。主要动作包括：

- 备份待修改文件和主机状态到 `/var/backups/amd-ai/<UTC时间戳>/`；
- 安全禁用确认属于 ROCm 6.4 的源并清理确认的旧包；
- 安装 `linux-oem-24.04`、匹配 headers、firmware 和宿主工具；
- Docker 缺失时安装 Docker Engine、Buildx 和 Compose 插件；
- 根据真实 `/dev/kfd` 和 render node GID 配置目标用户设备组；
- 根据物理内存配置 AI Max 的 TTM 上限并更新 initramfs。

加入 `docker` 组需要单独授权，因为 Docker daemon 权限接近主机 root。拒绝授权时平台保留 `sudo docker` 工作方式。

需要重启时，安装器保存状态并以退出码 `1` 停止，**不会自动重启**：

```bash
sudo reboot
```

重启后进入同一个 checkout，再次执行原命令。安装器核对 boot ID 后从 host verify 阶段继续，不会重复已经完成的写入。

若新启动的 OEM patch kernel 高于最低版本，且 TTM、设备权限和内核日志检查通过，但内核尚未进入静态已测清单，`v0.2.1` 会显示 `WARN HOST_VERIFY unverified` 并继续部署。状态文件记录 `host_verification_status`、`host_kernel` 和诊断码；真正的 `change-required` 或 `blocked` 仍会停止。

后续 `IMAGE_VERIFY` 仍必须通过 exact stable 镜像的 `gfx1151` Torch runtime 探针。警告同时打印一条可选的完整硬件资格命令；将该内核用于新的正式 release 前，仍必须完成包含 300 秒压力测试和内核日志差分的全部门禁。

特权 helper 只用于读取内核日志等受限事实，并必须携带安装状态中的 `target_user`；设备组按目标用户解析，不会再把 root 的组误判为项目用户权限。

### 仅容器平台模式

适用于 Docker、OEM `amdgpu`、TTM/GTT 和设备权限已经准备好的宿主：

```bash
./install.sh --mode container \
  --project-dir "$HOME/ai-projects/video-lab" \
  --project-name video-lab
```

该模式仍要求以下条件全部通过：

- Docker daemon 可访问；
- `/dev/kfd` 和至少一个 DRM render node 存在；
- 当前目标用户具备实际设备 GID；
- GPU 身份为 `gfx1151`；
- stable release manifest 和镜像身份链完整。

### 无交互容器部署

```bash
./install.sh \
  --mode container \
  --non-interactive \
  --project-dir "$HOME/ai-projects/video-lab" \
  --project-name video-lab \
  --image-source pull
```

无交互完整模式还必须提供当前 host plan 的 64 位摘要。先用完全相同的目标用户、项目目录和项目名运行一次交互模式，审阅全部计划后在 `APPLY` 提示处输入其他内容以拒绝写入。此时退出码 `2` 是预期行为，已审阅计划的 digest 会保存在状态文件中：

```bash
STATE="$HOME/.local/state/strix-halo-rocm-toolkit/video-lab-install-state.json"

./install.sh \
  --mode full \
  --target-user "$USER" \
  --project-dir "$HOME/ai-projects/video-lab" \
  --project-name video-lab \
  --image-source pull \
  --state-path "$STATE"

PLAN_DIGEST="$(
  STATE="$STATE" python3.12 -c \
    'import json, os; print(json.load(open(os.environ["STATE"], encoding="utf-8"))["host_plan_digest"])'
)"
printf 'accepted host plan: %s\n' "$PLAN_DIGEST"
```

然后使用该摘要恢复同一个安装状态：

```bash
./install.sh \
  --mode full \
  --non-interactive \
  --target-user "$USER" \
  --project-dir "$HOME/ai-projects/video-lab" \
  --project-name video-lab \
  --image-source pull \
  --accept-host-plan-digest "$PLAN_DIGEST" \
  --accept-docker-group \
  --state-path "$STATE"
```

`--accept-docker-group` 可以省略，此时不授权 Docker 组。如果计划在两次运行之间发生任何变化，digest 校验会阻断应用并要求重新审阅。无交互模式不会自动接受 host plan 变化、digest 漂移、build fallback 或重启。

常用安装参数：

| 参数 | 作用 |
| --- | --- |
| `--mode full` | 完整工作站安装 |
| `--mode container` | 仅部署容器平台 |
| `--non-interactive` | 禁用全部提示，缺少确认即停止 |
| `--dry-run` | 安装用户 launcher 并显示阶段，不修改宿主、镜像或项目 |
| `--project-dir PATH` | 第一个项目目录 |
| `--project-name NAME` | 第一个项目名称 |
| `--image-source pull` | 只接受 stable exact digest 拉取 |
| `--image-source build` | 从当前干净 checkout 本地构建 |
| `--target-user USER` | 完整模式的目标用户和项目所有者 |
| `--state-path PATH` | 覆盖默认恢复状态文件 |

默认恢复状态文件为：

```text
~/.local/state/strix-halo-rocm-toolkit/install-state.json
```

损坏状态会被重命名为带 UTC 时间戳的证据文件；安装器不会猜测哪些动作已经成功。完整状态与退出码说明见[安装与恢复](docs/install.md)。

## 安装后验证

### 1. 平台检查

```bash
strix-halo-rocm doctor
```

### 2. 宿主和 GPU 探针

```bash
mkdir -p reports
sudo -v
strix-halo-rocm host-verify \
  --probe-image rocm-python:7.2.1-py3.12 \
  --json reports/host-verify.json
```

`sudo -v` 只刷新读取内核日志和必要 Docker 命令的凭据。`host-verify` 本身仍按目标用户运行，以免把 root 的设备权限误判为普通用户权限。

### 3. PyTorch GPU 运行检查

```bash
strix-halo-rocm container-check \
  --image rocm-pytorch:7.2.1-py3.12-torch2.9.1 \
  --mode torch \
  --runtime \
  --json reports/gpu-check.json
```

通过要求包括：Torch 识别 `gfx1151`、在 GPU 上完成同步 tensor 运算，并且没有 CPU fallback。

### 4. 验证公开 release

```bash
strix-halo-rocm release verify \
  --manifest profiles/releases/stable.json
```

当前 stable exact references：

```text
ghcr.io/hellcatjack/strix-halo-rocm-python@sha256:e9991f97f578156c8620fbb587d2d34504eb632f165cc5597deaadaa3e692a12
ghcr.io/hellcatjack/strix-halo-rocm-pytorch@sha256:dc0bb217474cfd4f602423bd3bf4fe8714b03e900cf3c6b4417b99e622ebcf8b
```

验证命令使用临时空 Docker 配置进行匿名检查，不会借用本机 GHCR 登录凭据。

## 创建和运行项目

安装器可以创建第一个项目。继续创建其他独立项目时：

```bash
mkdir -p "$HOME/ai-projects"
strix-halo-rocm project init video-lab \
  --directory "$HOME/ai-projects/video-lab"
```

目标目录必须不存在或为空。初始化结果包括：

```text
video-lab/
├── amd-ai-project.toml
├── Dockerfile
├── project-entrypoint
├── requirements.in
├── requirements.lock
├── torch-constraints.txt
└── .dockerignore
```

项目会把 stable 父镜像解析为本地不可变 `sha256` config ID，并同时记录到 `base_image` 和 `base_digest`。以后即使标签移动，也不会静默替换已有项目的父层。

先检查实际 Docker 命令、设备、GID、挂载和共享内存：

```bash
strix-halo-rocm project run "$HOME/ai-projects/video-lab" --dry-run
```

默认项目命令是 `bash`：

```bash
strix-halo-rocm project run "$HOME/ai-projects/video-lab"
```

每次运行都会检查项目上下文指纹、父镜像身份、Torch manifest 和 GPU runtime。项目镜像缺失或过期时自动构建；也可以显式控制：

```bash
strix-halo-rocm project run "$HOME/ai-projects/video-lab" --build
strix-halo-rocm project run "$HOME/ai-projects/video-lab" --no-build
strix-halo-rocm project run "$HOME/ai-projects/video-lab" --shm-size-gib 24
```

在生成的 `amd-ai-project.toml` 中只修改现有 `[project]` 表里的应用命令、调试和可选共享内存设置；不要替换初始化器写入的其他键：

```toml
command = ["python", "main.py"]
debug = false
# shm_size_gib = 16
```

不要手工修改 `base_image`、`base_digest` 或生成的 Torch 约束。共享内存默认根据宿主内存与 TTM 规划计算，范围为 4 至 16 GiB；项目配置或命令行可以设置 1 至 128 GiB。

## Python 依赖与受保护 pip

### 可复现的项目依赖

把项目正式依赖写入 `requirements.in`，不要加入 `torch`、`torchvision`、`torchaudio` 或 `triton`：

```text
safetensors==0.5.3
einops==0.8.1
```

然后生成哈希锁并重建：

```bash
strix-halo-rocm project lock "$HOME/ai-projects/video-lab"
strix-halo-rocm project run "$HOME/ai-projects/video-lab" --build
```

`project lock` 在已锁定父镜像内运行 `uv pip compile`，宿主不需要安装 `uv`。项目依赖成为派生镜像层，不会在每次容器启动时重新安装。

### 容器内直接 pip install

运行中的受管项目容器允许普通操作：

```bash
pip install transformers==4.53.0
pip install -r requirements-extra.txt
pip install ./dist/local_package.whl
pip uninstall transformers
pip list
pip check
```

`pip` 包装器先解析完整依赖，再创建新的项目私有 generation。只有安装和有效 Torch 检查全部通过后，`.amd-ai/current` 才原子切换。失败 generation 不会取代上一个可用环境。

以下操作会被拒绝：

- 安装、升级、降级或卸载 `torch`、`torchvision`、`torchaudio`、`triton`；
- `--user`、`--target`、`--prefix` 或 `--root`；
- editable、`--force-reinstall`、`--ignore-installed` 或 `--no-deps`；
- branch、tag、HEAD 等可变 VCS 引用；固定 Git 依赖必须使用完整 40 位 commit。

父镜像 `/opt/venv` 对项目用户不可写，因此即使某个工具尝试绕过 PATH 中的受保护 `pip`，也不能覆盖正式 Torch 文件。需要不同 Torch 版本时应创建完整的新父镜像 profile，不能在项目里局部替换。

容器内试装确认稳定后，将直接依赖提升到 `requirements.in`，运行 `project lock` 和 `project run --build`，再从 overlay 卸载对应普通 root。详细策略见[受保护 pip](docs/protected-pip.md)。

## 模型、缓存和数据挂载

平台默认不创建，也不共享以下内容：

- ComfyUI 模型目录；
- Hugging Face Hub 缓存；
- 输入、输出、报告目录；
- MIOpen、Triton、Torch Inductor 或其他编译缓存。

用户可以在每个项目的 `amd-ai-project.toml` 中自由决定私有或跨项目共享。以下 `/data` 只是示例；先创建仅归当前用户所有的宿主目录：

```bash
sudo install -d -m 0755 -o "$USER" -g "$(id -gn)" \
  /data/video-lab \
  /data/video-lab/models \
  /data/video-lab/output \
  /data/video-lab/huggingface
```

再显式配置：

```toml
[[mounts]]
source = "/data/video-lab/models"
target = "/workspace/models"
read_only = true

[[mounts]]
source = "/data/video-lab/output"
target = "/workspace/output"
read_only = false

[[mounts]]
source = "/data/video-lab/huggingface"
target = "/workspace/.cache/huggingface"
read_only = false

[environment]
HF_HOME = "/workspace/.cache/huggingface"
HF_HUB_CACHE = "/workspace/.cache/huggingface/hub"
```

工具不会自动创建配置中的宿主 source；路径不存在时会在 Docker 启动前停止。禁止用用户挂载覆盖 `/opt/venv`、`/opt/rocm`、`/opt/amd-ai`、`/usr/local/bin`、`/dev`、`/proc` 或 `/sys`。

把同一宿主目录配置到多个项目即可显式共享；给每个项目配置不同目录即可保持隔离。模型和输出不会被镜像清理命令删除。

## ComfyUI 与大型视频应用

本仓库**不预装 ComfyUI、custom nodes、workflow 或模型**。推荐流程是：

1. 创建独立项目容器。
2. 把经过审查并固定 revision 的 ComfyUI 和 custom node 源码放入项目目录。
3. 将非 Torch Python 依赖写入 `requirements.in` 并生成锁。
4. 在项目 TOML 中显式配置模型、输入、输出和可选 Hugging Face 缓存。
5. 根据工作流设置 `shm_size_gib`，先运行 `--dry-run` 检查设备和挂载。
6. 运行应用前后检查内核日志；首次编译和首次下载不能代替第二次稳定性测试。

把生成配置中现有的 `command` 改为类似下面的命令：

```toml
command = [
  "bash",
  "-lc",
  "cd ComfyUI && exec python main.py --listen 0.0.0.0"
]
```

当前项目运行器不提供通用端口发布配置；因此上例只描述容器内应用启动，不应视为已经完成对外 Web 服务暴露。需要 Web 访问的生产部署应在项目运行器加入并审计端口白名单能力后进行，不能通过随意改写受管 Docker 命令来绕过镜像、设备和 overlay 检查。

正式 GPU 基础门禁通过不代表任意 custom node 都已兼容。大型视频工作流至少应执行两次完整应用测试，记录峰值 GPU 内存、运行时间、输出正确性和新增 `amdgpu` 内核错误。详细门禁见 [GPU 资格与发布手册](docs/gpu-qualification.md)。

## 自选 PyTorch 版本

Stable 安装器只自动选择已经验证的 ROCm 7.2.1 / PyTorch 2.9.1 组合。用户可以构建其他完整组合，但它们必须保持 `experimental`，并同时指定兼容的 Torch、TorchVision、TorchAudio 和 Triton：

```bash
cp profiles/torch/custom.example.env /absolute/path/custom.env
```

在 `custom.env` 中设置唯一 `PROFILE_ID`、四个 wheel 的版本、HTTPS URL 和真实 SHA-256，然后运行：

```bash
./bin/image-build rocm-pytorch \
  --profile /absolute/path/custom.env \
  --allow-experimental
```

使用该 profile 创建项目：

```bash
strix-halo-rocm project init custom-torch-lab \
  --directory "$HOME/ai-projects/custom-torch-lab" \
  --base-profile <自定义PROFILE_ID>

ALLOW_UNVERIFIED=1 \
  strix-halo-rocm project run "$HOME/ai-projects/custom-torch-lab"
```

`ALLOW_UNVERIFIED=1` 只接受当前进程环境变量，不能写入项目 TOML 形成永久绕过。自定义 profile 不会被提升为 stable，也不会改写 `profiles/releases/stable.json`。完整构建与锁文件规则见 [ROCm 镜像构建手册](docs/image-build.md)。

## Doctor 与 Repair

### 只读检查

检查平台：

```bash
strix-halo-rocm doctor
```

检查具体项目并保存报告：

```bash
mkdir -p reports
strix-halo-rocm doctor "$HOME/ai-projects/video-lab" \
  --json reports/video-lab-doctor.json
```

### 精确修复

```bash
strix-halo-rocm repair "$HOME/ai-projects/video-lab"
```

命令先显示只包含 exact image ID、exact registry digest 和单个项目 generation 的计划。交互模式要求精确输入 `REPAIR`；自动化必须显式使用：

```bash
strix-halo-rocm repair "$HOME/ai-projects/video-lab" \
  --yes \
  --json reports/video-lab-repair.json
```

上例分别写入 `reports/video-lab-repair.pre.json` 和 `reports/video-lab-repair.post.json`，保留修复前后证据；已有同名证据时命令拒绝覆盖。

典型处理：

| 诊断码 | 行为 |
| --- | --- |
| `IMAGE.PARENT_MISSING` | 匿名拉取并验证 exact 父镜像 digest |
| `IMAGE.DIGEST_DRIFT` | 验证后恢复本地 stable 标签绑定 |
| `IMAGE.PROJECT_CHANGED` | 先构建并验证替代镜像，再删除记录的旧 exact ID |
| `TORCH.BASE_CHANGED` | 恢复不可变 PyTorch 父镜像 |
| `TORCH.SHADOWED` | 隔离损坏 overlay，并离线重放最后有效锁 |
| `OVERLAY.LOCK_INVALID` | 阻止使用篡改锁并保留证据 |
| `GPU.RUNTIME_FAILED` | 阻断自动修复，要求检查宿主 GPU |
| `KERNEL.LOG_FAILED` | 阻断自动修复，要求检查内核日志 |

如果容器内误执行了错误的 Torch 安装命令，正常情况下受保护 `pip` 会在写入前拒绝。若 overlay、标签或项目镜像被外部工具手工改坏，执行 `doctor` 后再运行 `repair`。损坏 generation 会移动到项目 `.amd-ai/quarantine`，成功修复不会删除证据。

Repair 不运行 `docker system prune`，不使用通配镜像删除，不 force-reinstall Torch，也不会清理模型或用户缓存。详细动作见 [Doctor 与 Repair](docs/doctor-repair.md)。

## 升级、磁盘与清理

### 升级工具包

每次升级使用明确的发布标签：

```bash
git fetch origin --tags
git switch --detach <新发布标签>
./install.sh
```

安装器为每个版本创建独立运行时，再原子切换 `current`。不要从含有未提交修改的 checkout 执行正式本地构建。

从 `v0.2.0-rc1` 停在 `HOST_VERIFY` 的安装可直接升级：

```bash
git fetch origin --tags
git switch --detach v0.2.1
./install.sh
```

再次选择相同模式和项目目录。`v0.2.1` 会验证旧 `BOOTSTRAP` 输入摘要、迁移 schema 1 状态并从 `HOST_VERIFY` 续跑；它不会重新执行 `HOST_APPLY`。只有同一 `0.2.x` 系列、宿主写入已经完成且其他引导输入完全一致时才允许该迁移。

### 查看共享层占用

```bash
sudo docker system df -v
sudo docker image inspect \
  --format '{{json .RootFS.Layers}}' \
  rocm-pytorch:7.2.1-py3.12-torch2.9.1
```

`docker image ls` 显示的每个项目镜像大小都包含父层的逻辑大小，不能相加作为实际磁盘占用。ROCm 基础层和 PyTorch 层在 Docker 内容存储中只保留一份。

### 安全清理镜像

先预览：

```bash
strix-halo-rocm image-build prune \
  --older-than-hours 168 \
  --project-root "$HOME/ai-projects"
```

确认精确 ID 后应用：

```bash
strix-halo-rocm image-build prune \
  --older-than-hours 168 \
  --project-root "$HOME/ai-projects" \
  --apply
```

清理器保护 stable 标签、运行中容器和项目配置记录的基础镜像。它不会删除容器、命名卷、模型目录、项目数据或 wheelhouse，也不会调用 `docker system prune`。

### 移除用户运行时

当前没有自动宿主回滚/卸载命令。只有确认安装器不处于待重启/待恢复状态，并且不再需要状态证据时，才移除用户 launcher 和运行时。该操作不会恢复内核、TTM 或用户组：

```bash
rm -f "$HOME/.local/bin/strix-halo-rocm"
rm -rf "$HOME/.local/share/strix-halo-rocm-toolkit"
rm -rf "$HOME/.local/state/strix-halo-rocm-toolkit"
```

项目目录、模型和 Docker 镜像需由用户分别确认后处理。宿主修改应根据 `/var/backups/amd-ai/` 的备份按[宿主运维手册](docs/host-operations.md)恢复；不要把删除 OEM 内核或 TTM 配置简化为通用卸载脚本。

## 常见故障

| 现象 | 原因与处理 |
| --- | --- |
| `python3.12 is required` | 在宿主安装 Python 3.12；ROCm 和 Torch 不需要装到宿主 Python |
| `interactive install requires a terminal` | 在真实交互终端运行，或提供完整 `--non-interactive` 参数 |
| 安装器显示 `REBOOT_PENDING` | 执行 `sudo reboot`，随后重新运行完全相同的安装命令 |
| Docker permission denied | 重新登录以刷新组成员关系，或运行 `sudo -v` 后让工具使用 `sudo docker` |
| 找不到 `/dev/kfd` | 直接在宿主运行 `host-preflight`；不要用未映射设备的普通容器报告判断宿主 |
| `HOST.UNSUPPORTED_OS` | 该发行版只支持只读采集，不能强制套用 Ubuntu 写入适配器 |
| `HOST.UPSTREAM_UNVERIFIED` | 新 OEM patch 尚未进入已测清单；`v0.2.1` 记录警告并继续，后续 GPU runtime 探针仍为强制项，正式发布前还需完整硬件门禁 |
| `GPU.RUNTIME_FAILED` | 运行 `host-verify`、检查设备 GID 和 `sudo dmesg`；CPU fallback 不算通过 |
| `IMAGE.DIGEST_DRIFT` | 运行项目级 `doctor`，确认计划后执行 `repair` |
| `TORCH.SHADOWED` | overlay 中出现受保护包或身份变化；执行 `doctor`/`repair`，不要 force-reinstall |
| `experimental ... requires ALLOW_UNVERIFIED=1` | 只对明确的自定义 profile 在单次命令前设置该变量 |
| 空间不足 | 用 `docker system df -v` 检查 Docker root；使用受管 prune 预览，不执行全局 prune |
| 无法读取 `dmesg` | 先运行 `sudo -v`；工具不能读取前后内核日志时不会把空结果当成通过 |

建议保留每次失败的 JSON 报告、`/proc/cmdline`、镜像 inspect、资格报告和对应 Git revision。不要在没有稳定复现的情况下直接套用 Reddit、issue 或论坛中的内核参数与 ROCm 补丁。

## 安全边界

- 安装器以普通用户运行；需要宿主写入时进入固定、受审计的 sudo helper。
- 完整 host plan 必须显式确认，Docker 组另行授权。
- 项目容器默认 `--read-only`、private IPC、有限共享内存和非 root UID/GID。
- 只映射 `/dev/kfd`、`/dev/dri` 和实际设备组，不使用 `--privileged` 或 `--ipc=host`。
- 调试模式只增加 `SYS_PTRACE` 和 `seccomp=unconfined`，不会变成 privileged 容器。
- 项目配置不能覆盖保留环境变量和系统路径。
- 日志会隐藏名称中包含 `TOKEN`、`SECRET`、`PASSWORD` 或 `KEY` 的环境值。
- Stable registry 拉取和发布验证使用匿名 Docker 配置，防止本地凭据掩盖公开权限错误。
- Stable manifest 同时绑定源码 revision、资格报告、SBOM、OCI manifest digest、Docker config ID 和内嵌锁摘要。

## 命令索引

安装后推荐统一使用 `strix-halo-rocm`：

| 命令 | 用途 |
| --- | --- |
| `strix-halo-rocm install` | 再次进入交互安装器 |
| `strix-halo-rocm host-preflight` | 只读采集宿主状态 |
| `strix-halo-rocm host-prepare plan` | 查看宿主写入计划 |
| `strix-halo-rocm host-prepare apply` | 确认后应用宿主计划 |
| `strix-halo-rocm host-verify` | 验证实时 TTM、设备权限、GPU 和内核日志 |
| `strix-halo-rocm image-build rocm-python` | 构建 ROCm/Python 基础镜像 |
| `strix-halo-rocm image-build rocm-pytorch` | 构建完整 Torch profile 镜像 |
| `strix-halo-rocm image-build prune` | 预览或精确清理受管旧镜像 |
| `strix-halo-rocm container-check` | 检查镜像 metadata 或 GPU runtime |
| `strix-halo-rocm project init` | 创建独立项目 |
| `strix-halo-rocm project lock` | 解析并哈希锁定项目依赖 |
| `strix-halo-rocm project run` | 验证、构建并运行项目容器 |
| `strix-halo-rocm doctor` | 只读检查平台或项目 |
| `strix-halo-rocm repair` | 按 exact target 事务修复项目 |
| `strix-halo-rocm release verify` | 匿名验证 stable manifest 中的公开镜像 |

查看任意子命令参数：

```bash
strix-halo-rocm --help
strix-halo-rocm project run --help
strix-halo-rocm repair --help
```

仓库中的 `bin/host-preflight`、`bin/host-prepare`、`bin/image-build`、`bin/project-*` 等兼容入口继续可用。

## 专项文档

- [安装、恢复、状态与 sudo 边界](docs/install.md)
- [Ryzen AI Max+ 395 宿主运维与备份恢复](docs/host-operations.md)
- [ROCm/PyTorch 镜像构建与自选 profile](docs/image-build.md)
- [独立项目容器与自选挂载](docs/project-workflow.md)
- [受保护 pip 与 overlay generation](docs/protected-pip.md)
- [Doctor 与精确 Repair](docs/doctor-repair.md)
- [Radeon 8060S GPU 资格与发布门禁](docs/gpu-qualification.md)
- [Stable release 身份链](docs/release-chain.md)

## 开发与验证

普通开发测试不需要 GPU：

```bash
uv sync --dev
uv run pytest -m "not hardware" -q
```

容器集成测试需要本地 Docker 和对应镜像。目标 Ryzen AI Max+ 395 主机的正式硬件门禁：

```bash
sudo -v
./bin/container-check \
  --suite stable \
  --profile profiles/qualification/stable.toml \
  --json reports/qualification.json
```

Stable profile 包含 `rocminfo`、FP16 GEMM/卷积、原生 HIP、PyTorch HIP extension、Triton JIT、重复初始化、300 秒压力测试和前后内核日志差分。任何 CPU fallback、无法读取内核日志或新增 GPU reset/MES timeout/page fault 都不算通过。
