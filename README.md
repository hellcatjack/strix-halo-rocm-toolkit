# AMD AI Container Platform

面向 AMD Ryzen AI Max+ 395 / Radeon 8060S (`gfx1151`) 的可复现 ROCm 容器开发平台。

正式基线固定为：

| 层 | 版本 |
| --- | --- |
| 首发宿主机 | Ubuntu 24.04.x AMD64 + OEM 内核 |
| ROCm 用户态 | 7.2.1，位于容器内 |
| Python | 3.12，单一 `/opt/venv` |
| PyTorch 组合 | 2.9.1 / TorchVision 0.24.0 / TorchAudio 2.9.0 / Triton 3.5.1 |

每个项目使用独立镜像和容器。大体积 PyTorch 文件由不可变父镜像层复用，不通过多个项目 venv 重复安装。仓库不预装 ComfyUI，也不默认创建或共享 ComfyUI、Hugging Face、模型或编译缓存目录；挂载由项目用户显式选择。

## 主机工作流

```bash
./bin/host-preflight --json reports/preflight.json
sudo ./bin/host-prepare plan --target-user "$USER"
sudo ./bin/host-prepare apply --target-user "$USER"
sudo reboot
./bin/image-build rocm-python
./bin/host-verify \
  --probe-image rocm-python:7.2.1-py3.12 \
  --json reports/host-verify.json
```

主机只安装 OEM 内核、Linux firmware、内核自带 `amdgpu`、Docker 和必要诊断工具。ROCm、HIP SDK、Python、PyTorch 与编译工具链放在容器内。完整操作、恢复步骤和安全边界见 [主机运维手册](docs/host-operations.md)，镜像锁定与自定义 PyTorch 流程见 [镜像构建手册](docs/image-build.md)，独立项目、依赖和自选模型挂载见 [项目容器手册](docs/project-workflow.md)。

## 命令

- `bin/host-preflight`：只读采集系统、GPU、设备权限、旧 ROCm、DKMS、Docker、内存与 TTM 状态。
- `bin/host-prepare plan`：生成有序变更计划，不修改主机。
- `bin/host-prepare apply`：备份后按计划执行；要求 root 和精确确认。
- `bin/host-verify`：重启后检查 live TTM、内核 GPU 错误和容器内 `gfx1151`。
- `bin/image-build`：按固定锁构建 ROCm/Python、稳定或 experimental PyTorch 镜像，并提供安全清理预览。
- `bin/container-check`：在指定镜像中执行元数据或 GPU runtime 检查。
- `bin/container-check --suite stable`：执行完整 `gfx1151` 硬件资格门禁。
- `bin/gpu-release`：验证摘要与 SBOM 后创建不可变 verified 发布记录。
- `bin/project-init`：创建父镜像摘要锁定的独立项目。
- `bin/project-lock`：为项目依赖生成哈希锁，并保护父层 Torch 组合。
- `bin/project-run`：按指纹构建/复用项目镜像并安全映射 GPU 设备。

未知 Linux 发行版可以运行公共只读审计，但没有写入适配器，`host-prepare` 会拒绝执行。

## 开发验证

```bash
uv sync --dev
uv run pytest -m "not hardware" -q
bash -n bin/_dispatch bin/host-preflight bin/host-prepare bin/host-verify \
  bin/image-build bin/container-check bin/project-init bin/project-lock \
  bin/project-run bin/gpu-release
```

目标主机的 300 秒 GPU 门禁需显式运行：

```bash
sudo -v
uv run pytest tests/hardware/test_release.py -m hardware -v
```

检查顺序、失败证据、单变量 workaround 规则和发布步骤见 [GPU 资格与发布手册](docs/gpu-qualification.md)。

设计规格和分阶段实施计划位于 `docs/superpowers/`。
