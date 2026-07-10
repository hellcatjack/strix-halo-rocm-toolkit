# Strix Halo ROCm Toolkit

面向 AMD Ryzen AI Max+ 395 / Radeon 8060S (`gfx1151`) 的可恢复 ROCm 容器开发平台。

正式使用应检出固定发布标签，再运行可审计的本地脚本：

```bash
git clone --branch v0.2.0 --depth 1 \
  https://github.com/hellcatjack/strix-halo-rocm-toolkit.git
cd strix-halo-rocm-toolkit
./install.sh
```

安装器提供“完整工作站安装”和“仅容器平台”两种模式。正式宿主写入支持 Ubuntu 24.04.x AMD64；其他 Linux 可以执行只读检查，但没有适配器时不会修改内核、APT、TTM 或用户组。容器基线固定为 ROCm 7.2.1、Python 3.12、PyTorch 2.9.1、TorchVision 0.24.0、TorchAudio 2.9.0 和 Triton 3.5.1。

每个项目使用独立镜像和容器。大型 Torch 文件由不可变父镜像层复用，不会在多个项目 venv 中重复安装。运行中允许普通 `pip install`，但受保护的 Torch 组合不能被覆盖；依赖写入项目私有、可事务恢复的 overlay。

仓库不预装 ComfyUI，也不默认创建或共享 ComfyUI、Hugging Face、模型、输入、输出或编译缓存目录。用户可在项目配置中显式选择任何宿主挂载。

## 快速安装

无交互容器模式示例：

```bash
./install.sh \
  --mode container \
  --non-interactive \
  --project-dir "$HOME/ai-projects/video-lab" \
  --project-name video-lab \
  --image-source pull
```

默认从公开 GHCR 匿名拉取 stable manifest 指定的 exact digest，并核对 OCI manifest、Docker config、镜像标签和内嵌锁。网络或镜像不存在时，交互模式才会询问是否本地构建；身份不匹配绝不回退。

完整模式会展示固定 host plan，要求精确输入 `APPLY`，并单独询问 Docker 组授权。需要重启时只写入 `REBOOT_PENDING` 并退出，不自动调用 reboot；重启后再次执行同一命令即可继续。

安装后统一命令为：

```bash
strix-halo-rocm install
strix-halo-rocm doctor [PROJECT]
strix-halo-rocm repair PROJECT
strix-halo-rocm project init NAME
strix-halo-rocm project lock PROJECT
strix-halo-rocm project run PROJECT
```

旧的 `bin/host-preflight`、`bin/host-prepare`、`bin/image-build`、`bin/project-*` 等命令继续兼容。

## 文档

- [安装、恢复与 sudo 边界](docs/install.md)
- [受保护 pip 与 Torch 修复](docs/protected-pip.md)
- [Doctor 与精确 Repair](docs/doctor-repair.md)
- [Stable release 身份链](docs/release-chain.md)
- [宿主运维](docs/host-operations.md)
- [项目容器与自选挂载](docs/project-workflow.md)
- [GPU 资格门禁](docs/gpu-qualification.md)

自定义完整 Torch profile 仍是显式 experimental 工作流：

```bash
bin/image-build rocm-pytorch \
  --profile profiles/torch/custom.example.env \
  --allow-experimental
```

安装器不会自动选择或提升 experimental 镜像。

## 开发验证

```bash
uv sync --dev
uv run pytest -m "not hardware" -q
```

目标主机的硬件发布门禁需单独运行 `tests/hardware/test_release.py`；任何 CPU fallback 都不算 GPU 通过。
