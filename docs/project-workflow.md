# 独立项目容器工作流

本文说明如何从 ROCm 7.2.1 / PyTorch 2.9.1 正式父镜像创建独立项目。每个项目有自己的配置、源码、依赖锁、派生镜像和容器，但共同复用父镜像中的 `/opt/venv` 与完整 PyTorch 文件层。

## 1. 前置条件

先完成宿主准备、重启和两层正式镜像构建：

```bash
./bin/host-verify --probe-image rocm-python:7.2.1-py3.12
./bin/image-build rocm-pytorch \
  --profile profiles/torch/stable.env
```

项目工具自动尝试当前用户的 Docker；daemon 不可直接访问时使用 `sudo -n docker`。运行容器仍采用原宿主用户的 UID/GID，不会因为 Docker 命令使用 sudo 而默认变成 root。

## 2. 创建与运行

创建稳定基线项目：

```bash
./bin/project-init comfy-lab
./bin/project-run comfy-lab
```

指定其他目录：

```bash
./bin/project-init video-lab \
  --directory "$HOME/projects/video-lab"
./bin/project-run "$HOME/projects/video-lab"
```

初始化器将父镜像解析为本地不可变 `sha256` ID，并同时写入 `base_image` 与 `base_digest`。删除或替换父标签不会静默改变既有项目。项目目录已有内容时初始化器拒绝覆盖。生成的 `.dockerignore` 是构建策略的一部分，必须保留 `.git`、`.venv`、`.cache`、`.amd-ai`、模型/输入输出/报告和 Python 缓存排除规则；为防止大目录意外重新进入上下文，项目 `.dockerignore` 不允许 `!` negation 规则。

`project-run` 默认比较当前构建上下文指纹、父摘要与项目镜像标签：镜像缺失或过期时自动重建，其余情况直接复用。无论复用还是新建，运行前都会核对父层前缀、ROCm/Python/Torch profile 标签与环境、非 root UID、工作目录、策略入口点和 Torch 文件 manifest；项目 Dockerfile 不能把 experimental 状态伪装成 verified，也不能持久写入 `ALLOW_UNVERIFIED`。可显式控制：

```bash
./bin/project-run comfy-lab --build
./bin/project-run comfy-lab --no-build
./bin/project-run comfy-lab --dry-run
```

`--build` 强制重建；`--no-build` 在镜像缺失或过期时停止；`--dry-run` 完成设备、配置、镜像构建/复用和标签验证并打印脱敏后的最终命令，但不启动项目容器。

## 3. 项目依赖

把非 Torch Python 依赖写入项目的 `requirements.in`，然后显式更新哈希锁：

```bash
printf '%s\n' \
  'safetensors==0.5.3' \
  'einops==0.8.1' \
  > comfy-lab/requirements.in
./bin/project-lock comfy-lab
./bin/project-run comfy-lab
```

`project-lock` 在项目已锁定的父镜像中使用 `uv pip compile`，宿主机不需要单独安装 `uv`。该临时解析容器不映射 GPU、模型或持久缓存，只把当前项目目录挂入并生成带 SHA-256 的 `requirements.lock`。下一次运行把依赖安装为项目镜像的新层，不在项目目录创建 `.venv`，也不在容器启动时重复安装。

不要在 `requirements.in` 中改装 Torch、TorchVision、TorchAudio 或 Triton。项目约束和构建后的文件 manifest 会阻止它们被升级、降级、URL wheel 覆盖或部分替换。需要其他 PyTorch 组合时，应先按[镜像构建手册](image-build.md)构建完整 experimental profile，再创建项目：

```bash
./bin/project-init custom-torch-lab \
  --base-profile <自定义-PROFILE_ID>
ALLOW_UNVERIFIED=1 ./bin/project-run custom-torch-lab
```

experimental 或缺少状态标签的镜像只接受当前进程中精确的 `ALLOW_UNVERIFIED=1`。该变量不能写入项目 TOML 形成永久绕过。

## 4. 命令与运行资源

编辑 `amd-ai-project.toml` 中的数组来设置容器命令：

```toml
[project]
command = ["python", "main.py", "--listen", "0.0.0.0"]
debug = false
# shm_size_gib = 16
```

共享内存默认由宿主总内存和 TTM 规划动态计算，范围为 4 至 16 GiB。项目配置可固定 1 至 128 GiB，也可单次覆盖：

```bash
./bin/project-run video-lab --shm-size-gib 24
```

运行器使用 private IPC、有限 `--shm-size`、`/dev/kfd`、`/dev/dri` 和设备实际 GID，不使用 `--privileged` 或 host IPC。只有 stdin 与 stdout 同时为终端时才加入交互 TTY 参数。项目本地 HOME 位于 `.amd-ai/home`，权限为 `0700`，所有者与容器 UID/GID一致。

调试模式只增加 `SYS_PTRACE` 和 `seccomp=unconfined`：

```bash
./bin/project-run video-lab --debug
```

它不会追加其他 capability，也不会变成 privileged 容器。

## 5. 模型与缓存挂载

初始化结果没有 ComfyUI、模型目录、Hugging Face 缓存或任何共享卷。用户可以在每个项目的 `amd-ai-project.toml` 中独立决定是否共享、目标位置和读写权限。以下是一个完整的自选示例：

```toml
[[mounts]]
source = "/data/private/video-models"
target = "/workspace/models"
read_only = true

[[mounts]]
source = "/data/team/shared-models"
target = "/workspace/shared-models"
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

工具不会自动创建用户声明的 source，路径不存在时会在 Docker 启动前报错。相对 source 以项目目录为基准。模型目录可保持项目私有，也可由用户把同一宿主目录显式挂到多个项目；两种方式都不会被平台强制改变。

环境变量逐项传入容器。dry-run 和日志会隐藏名称中包含 `TOKEN`、`SECRET`、`PASSWORD` 或 `KEY` 的值。不要把长期凭据提交到项目配置；优先使用权限受控的外部凭据文件或临时注入机制。

## 6. PyTorch 层与磁盘占用

稳定项目镜像的层级为：

```text
Ubuntu/ROCm/Python -> PyTorch 组合 -> 项目依赖与源码
```

Docker 的内容寻址存储只保留一份相同父层。两个项目的 `docker image ls` 显示大小都包含约 25 GB 父镜像，这是逻辑总大小，不能相加视为实际新增占用；每个项目通常只增加自己的依赖和源码层。项目中没有第二个 Torch venv。

查看实际层与共享占用：

```bash
sudo docker system df -v
sudo docker image inspect \
  --format '{{json .RootFS.Layers}}' \
  rocm-pytorch:7.2.1-py3.12-torch2.9.1
```

安全清理先预览，并把外部项目根目录传给保护扫描：

```bash
./bin/image-build prune \
  --project-root "$HOME/projects"
./bin/image-build prune \
  --project-root "$HOME/projects" \
  --apply
```

清理器保护项目配置记录的父镜像、稳定标签和运行中容器，不删除命名卷、模型目录或 wheelhouse，也不调用 `docker system prune`。

## 7. ComfyUI 与大型视频应用

平台不预装 ComfyUI。把经过审查的 ComfyUI 版本及 custom nodes 放进自己的项目目录，并把它们的 Python 依赖写入 `requirements.in`。模型位置、输入输出目录和 Hugging Face 缓存均按上一节显式声明。

首次正式使用前，先用 `--dry-run` 检查挂载、设备 GID、共享内存和命令，再执行项目运行。GPU 正式资格测试、长时间压力测试和内核日志判定见后续 GPU 资格手册；CPU fallback 不算通过。
