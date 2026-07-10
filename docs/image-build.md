# ROCm 镜像构建手册

本文适用于 AMD Ryzen AI Max+ 395 / Radeon 8060S (`gfx1151`)。正式镜像组合固定为 Ubuntu 24.04、ROCm 7.2.1、Python 3.12、PyTorch 2.9.1、TorchVision 0.24.0、TorchAudio 2.9.0 和 Triton 3.5.1。

## 1. 镜像边界

`rocm-python:7.2.1-py3.12` 是不含 Torch 的开发父镜像，包含完整 HIP、ROCm ML SDK、C/C++、CMake、Ninja 和 Python headers。`rocm-pytorch:7.2.1-py3.12-torch2.9.1` 在其上增加完整且哈希锁定的 Torch 组合。

两个镜像都只有 `/opt/venv` 一个 Python 环境，默认用户为 `developer`。仓库不预装 ComfyUI，也不设置 Hugging Face、ComfyUI 模型、MIOpen、Triton 或 Torch Inductor 缓存目录。每个项目从 PyTorch 父镜像派生，Docker 复用大文件层；项目不再创建一份包含 Torch 的独立 venv。

## 2. 稳定构建

先完成宿主准备和重启后验证，再依次执行：

```bash
./bin/image-build rocm-python
./bin/host-verify --probe-image rocm-python:7.2.1-py3.12
./bin/image-build rocm-pytorch \
  --profile profiles/torch/stable.env
```

命令自动检测当前用户的 Docker 权限；直接访问 daemon 失败时尝试 `sudo -n docker`。构建前会验证：

- Ubuntu 与 `uv` OCI 摘要锁；
- ROCm 仓库 keyring SHA-256 和全部 Debian 包精确版本；
- profile 语法、状态、四个 AMD wheel SHA-256；
- wheelhouse manifest 中每个文件的大小和哈希；
- 本地 `rocm-python` 父镜像的完整配置 ID。

PyTorch 构建使用由父配置 ID 派生的本地不可变别名。wheelhouse 作为只读 BuildKit named context 挂载，不通过 `COPY` 形成额外下载层。构建后再次检查 OCI 标签、Torch 文件 manifest 和四组件导入。

查看最终 ID 和标签：

```bash
sudo docker image inspect --format \
  '{{.Id}} {{json .Config.Labels}}' \
  rocm-python:7.2.1-py3.12

sudo docker image inspect --format \
  '{{.Id}} {{json .Config.Labels}}' \
  rocm-pytorch:7.2.1-py3.12-torch2.9.1
```

## 3. 自选 PyTorch 组合

自选版本必须作为完整 profile 提供 Torch、TorchVision、TorchAudio 和 Triton，不能在项目镜像中单独替换 `torch`。从示例开始：

```bash
cp profiles/torch/custom.example.env /absolute/path/custom.env
```

编辑 `custom.env`：

1. 使用唯一的 `PROFILE_ID`。
2. 保持 `PROFILE_STATUS=experimental`。
3. 填写四个相互兼容的版本和 HTTPS wheel URL。
4. 将四个全零 SHA-256 替换为实际文件哈希。
5. 保持本镜像族要求的 `ROCM_VERSION=7.2.1`、`PYTHON_ABI=cp312` 和 `PLATFORM=linux/amd64`。

执行：

```bash
./bin/image-build rocm-pytorch \
  --profile /absolute/path/custom.env \
  --allow-experimental
```

工具先按用户给出的 SHA-256 下载四个主 wheel，再为该 profile 单独解析和锁定传递依赖。自定义构建不会借用稳定版 requirements lock。生成镜像带 `org.amd-ai.profile.status=experimental`；后续正式项目启动必须显式设置：

```bash
ALLOW_UNVERIFIED=1 ./bin/project-run <项目名>
```

只有仓库自带的 `profiles/torch/stable.env` 可以声明 `verified`。复制该文件到其他路径不能绕过 experimental 边界。

## 4. 锁文件刷新

稳定 profile 的官方 AMD wheel URL 为：

```text
https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/torch-2.9.1%2Brocm7.2.1.lw.gitff65f5bc-cp312-cp312-linux_x86_64.whl
https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/torchvision-0.24.0%2Brocm7.2.1.gitb919bd0c-cp312-cp312-linux_x86_64.whl
https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/torchaudio-2.9.0%2Brocm7.2.1.gite3c6ee2b-cp312-cp312-linux_x86_64.whl
https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/triton-3.5.1%2Brocm7.2.1.gita272dfa8-cp312-cp312-linux_x86_64.whl
```

ROCm APT 源固定为：

```text
https://repo.radeon.com/rocm/apt/7.2.1 noble main
https://repo.radeon.com/graphics/7.2.1/ubuntu noble main
```

维护者刷新 wheel 锁：

```bash
./tools/lock-wheels \
  --sources profiles/torch/stable.sources.env \
  --profile profiles/torch/stable.env \
  --wheelhouse .cache/wheels/rocm-7.2.1-py3.12-torch-2.9.1
```

维护者刷新 OCI、keyring 和 ROCm Debian 包锁：

```bash
./tools/lock-rocm-packages --rocm-version 7.2.1 --ubuntu noble
```

锁更新必须作为独立审查变更提交。不要手工删除本地 wheel 后继续使用旧 manifest，也不要把 `.cache/wheels` 加入 Git。

## 5. 磁盘与构建元数据

稳定 wheelhouse 当前约 2 GB，位于：

```text
.cache/wheels/rocm-7.2.1-py3.12-torch-2.9.1/
```

它是主机下载缓存，不进入最终镜像层，也不会被 `image-build prune` 删除。ROCm 基础镜像约 21 GB，PyTorch 镜像显示约 25 GB，但后者复用全部基础层；不能把两个显示大小简单相加作为实际增量。

支持 containerd image store 时，构建附加 max provenance 和 SBOM。经典 Docker image store 无法保存 image attestations，工具会明确告警、关闭不受支持的附加项，同时仍将 max build record 写到：

```text
.cache/build-metadata/<镜像标签>.json
```

工具不会自动切换 Docker 存储后端，因为切换会暂时隐藏旧后端中的现有镜像和容器。Docker 对该限制和 containerd store 的说明见 [Build attestations](https://docs.docker.com/build/metadata/attestations/) 与 [containerd image store](https://docs.docker.com/engine/storage/containerd/)。

## 6. 检查与清理

GPU 无关的镜像检查：

```bash
./bin/container-check \
  --image rocm-pytorch:7.2.1-py3.12-torch2.9.1 \
  --mode torch --metadata-only --json reports/image-check.json
```

宿主设备已准备好时执行运行检查：

```bash
./bin/container-check \
  --image rocm-pytorch:7.2.1-py3.12-torch2.9.1 \
  --mode torch --runtime --json reports/gpu-check.json
```

运行检查动态映射 `/dev/kfd`、`/dev/dri` 和实际 GID，使用 private IPC，不使用 `--privileged`。它要求 Torch 看见 `gfx1151` 并完成同步 GPU tensor 运算，不接受 CPU fallback。

清理默认只预览：

```bash
./bin/image-build prune --older-than-hours 168
```

工具保护当前稳定标签、运行中容器和 `--project-root` 下 `amd-ai-project.toml` 记录的基础 ID，只列出带 AMD AI profile/project 标签且超过年龄阈值的镜像。确认列表后才执行：

```bash
./bin/image-build prune --older-than-hours 168 --apply
```

应用模式只删除预览中的精确镜像 ID，并执行同年龄阈值的 `docker buildx prune`。它不运行 `docker system prune`，不删除容器、命名卷或 wheelhouse。

## 7. Linux 发行版边界

OCI 镜像可在满足 AMD GPU 内核驱动、`/dev/kfd`、render node 和 Docker/OCI runtime 条件的 Linux 主机上复用；容器内用户态保持 Ubuntu 24.04，不需要随宿主发行版复制一套镜像。

当前自动化宿主写入适配器只支持 Ubuntu 24.04.x AMD64。其他发行版可以运行只读预检，但 `host-prepare` 会拒绝写入；在新增并验证对应内核、包管理器、Docker 和设备权限适配器前，不应宣称正式支持。
