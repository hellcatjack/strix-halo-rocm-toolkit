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

主机只安装 OEM 内核、Linux firmware、内核自带 `amdgpu`、Docker 和必要诊断工具。ROCm、HIP SDK、Python、PyTorch 与编译工具链放在容器内。完整操作、恢复步骤和安全边界见 [主机运维手册](docs/host-operations.md)。

## 命令

- `bin/host-preflight`：只读采集系统、GPU、设备权限、旧 ROCm、DKMS、Docker、内存与 TTM 状态。
- `bin/host-prepare plan`：生成有序变更计划，不修改主机。
- `bin/host-prepare apply`：备份后按计划执行；要求 root 和精确确认。
- `bin/host-verify`：重启后检查 live TTM、内核 GPU 错误和容器内 `gfx1151`。
- `bin/image-build`、`bin/project-init`、`bin/project-run`、`bin/container-check`：由后续镜像与项目运行层提供。

未知 Linux 发行版可以运行公共只读审计，但没有写入适配器，`host-prepare` 会拒绝执行。

## 开发验证

```bash
uv sync --dev
uv run pytest -q
bash -n bin/_dispatch bin/host-preflight bin/host-prepare bin/host-verify
```

设计规格和分阶段实施计划位于 `docs/superpowers/`。

