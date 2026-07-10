# 安装与恢复

## 支持范围

完整工作站模式正式支持 Ubuntu 24.04.x AMD64、Ryzen AI Max+ 395、Radeon 8060S 和 inbox `amdgpu`。容器模式不写入内核、APT、固件、TTM、用户组或 Docker 安装，但仍要求 Docker daemon、`/dev/kfd`、DRM render node、实际设备 GID 和目标 GPU 身份全部通过。

Python 安装器必须使用 3.12。ROCm 与 PyTorch 只存在于容器镜像中，不向宿主 Python 安装。

## 交互模式

```bash
./install.sh
```

首页可选择完整工作站、仅容器平台或 doctor/repair。完整模式会输出每一项 host plan，精确输入 `APPLY` 后才允许写入；Docker 组因具有等同 root 的 daemon 权限，使用单独的 yes/no 授权。主机写入由受审计的 sudo 边界执行，安装器本身和用户本地 launcher 保持目标用户所有权。

如果变更要求重启，安装器记录当前 boot ID 和 `REBOOT_PENDING` 后以状态 1 退出。它不会运行 `reboot`。手工重启后再次执行原命令；boot ID 未变化时不会重复 apply，变化后从 host verify 继续。

容器模式可直接指定：

```bash
./install.sh --mode container --project-dir "$HOME/ai-projects/demo"
```

默认优先匿名拉取 stable release 的 exact digest。只有获取失败时才提供本地 build 选择。已经拉取的 config、RepoDigest、label 或内嵌摘要不匹配时会直接阻断。

## 无交互模式

容器模式必须明确提供项目和镜像来源：

```bash
./install.sh \
  --mode container \
  --non-interactive \
  --project-dir /srv/comfy-lab \
  --project-name comfy-lab \
  --image-source pull
```

完整模式还要求当前 host plan 的 64 位摘要：

```bash
./install.sh \
  --mode full \
  --non-interactive \
  --project-dir /srv/video-lab \
  --project-name video-lab \
  --image-source pull \
  --accept-host-plan-digest <64-lowercase-hex> \
  --accept-docker-group
```

`--accept-docker-group` 可省略，此时保留 sudo Docker。无交互模式不会推断 build fallback、接受 digest 漂移、接受重启或模拟任何提示回答。

其他参数：

| 参数 | 作用 |
| --- | --- |
| `--dry-run` | 安装 launcher 并打印阶段，不修改宿主、镜像或项目 |
| `--target-user USER` | 完整模式计划和项目所有者 |
| `--manifest PATH` | 严格 stable release manifest |
| `--source-root PATH` | 本地 build 使用的固定源码 checkout |
| `--state-path PATH` | 覆盖默认恢复状态路径 |

## 状态与退出码

默认状态为：

```text
~/.local/state/strix-halo-rocm-toolkit/install-state.json
```

每个成功阶段保存其规范 JSON 输入的 SHA-256。状态采用 `0600` 临时文件、文件 `fsync`、`os.replace` 和目录 `fsync`。并发安装由非阻塞锁拒绝。损坏状态会保留为 `install-state.corrupt.<UTC>.json`，不会猜测已完成动作。

| 退出码 | 含义 |
| --- | --- |
| 0 | 完成，或 dry-run 成功 |
| 1 | 需要重启/继续，或用户中断 |
| 2 | 阻断、拒绝、输入漂移或动作失败 |

## 空间与本地构建

pull/build 前要求 Docker 文件系统可用空间大于缺失层估算再加 5 GiB。项目 generation 前要求可用空间大于已解析 wheel 字节的两倍再加 1 GiB。错误会报告 `required_bytes` 和 `available_bytes`，不会先写一半。

选择 `--image-source build` 时，必须保留原始源码 checkout。安装器要求 `git rev-parse HEAD` 等于记录的 `installer_source_revision`，`git status --porcelain` 为空，并复核全部构建文件。本地结果标记为 unqualified，不改写 stable manifest。

## 数据挂载

项目模板没有 ComfyUI、Hugging Face、模型、输入、输出或缓存挂载。按项目在 `amd-ai-project.toml` 中显式加入宿主目录，并自行决定是否跨项目共享。不要把 `/opt/venv`、`/opt/rocm`、`/opt/amd-ai`、`/dev`、`/proc` 或 `/sys` 替换为用户挂载。
