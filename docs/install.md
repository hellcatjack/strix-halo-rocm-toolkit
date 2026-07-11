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

Host verify 的 `pass` 正常继续。满足最低 OEM 版本，且 TTM、GPU 设备权限和内核日志检查通过，但内核 patch 尚未列入已测清单时，报告为 `unverified`；安装器输出 `WARN`、记录内核与诊断码并继续。后续 `IMAGE_VERIFY` 仍强制执行 exact stable 镜像的 `gfx1151` Torch runtime 探针。`change-required`、`reboot-required` 或 `blocked` 仍然阻断。该放宽只适用于普通部署，不改变 stable release 的完整硬件资格门禁。

sudo host verify helper 要求显式的 `target_user`。root 只承担受限事实采集，设备组始终按目标用户的 passwd/group 数据计算，避免把 root 缺少 render 组错误报告为 `GPU.PERMISSION`。

## 实时进度与私有日志

默认模式逐行显示 `PLAN/SKIP/START/DETAIL/WAIT/PASS/SUMMARY`。每个阶段使用当前工作流的 `[i/n]` 位置；container 模式是 8 个阶段，full 模式是 13 个阶段。只有状态检查点成功写入后才会出现 `PASS`。运行中的命令 15 秒没有新输出时显示 `WAIT`，任何新输出都会重新开始计时；等待用户输入期间暂停心跳。

```bash
# 显示命令、精确字节、阶段枚举与镜像 digest
./install.sh --verbose --mode container \
  --project-dir /app/test/video-lab

# 适合 CI：只显示警告、失败、恢复信息、摘要和最终日志路径
./install.sh --quiet --non-interactive \
  --mode container \
  --project-dir /app/test/video-lab \
  --image-source pull
```

`--verbose` 和 `--quiet` 互斥。所有模式都会创建完整日志，位置为：

```text
~/.local/state/strix-halo-rocm-toolkit/logs/<project-key>/install-<UTC>-<pid>.log
```

日志目录为 `0700`，文件为 `0600`。`v0.2.3` 不轮转或自动删除日志。Docker pull、BuildKit、uv/pip、wheel 下载和 sudo 宿主命令会实时刷新阶段活动；日志会脱敏常见凭据，但仍可能包含项目路径、镜像名和包名，分享前必须复核。

## 典型输出与恢复

下面省略了部分中间阶段和具体路径，实际输出始终是一行一个事件。

### 新安装

```text
PLAN     模式=container，项目=/app/test/video-lab，名称=video-lab
PLAN     状态=.../video-lab-<key>.json（per-project）
PLAN     镜像来源=pull，stable release=待解析
PLAN     共 8 个阶段，从 BOOTSTRAP 继续
LOG      .../install-<UTC>-<pid>.log
START    [1/8] 安装用户运行时
PASS     [1/8] 安装用户运行时，用时 00:01
START    [4/8] 获取或构建 stable 镜像
DETAIL   缺失层=10.0 GiB，需要=15.0 GiB，可用=100.0 GiB，...
WAIT     [4/8] 已运行 00:15，15 秒内没有新输出，仍在获取或构建 stable 镜像
PASS     [4/8] 获取或构建 stable 镜像，用时 08:42
SUMMARY  安装完成，...
LOG      .../install-<UTC>-<pid>.log
```

### 中断后恢复

重新执行完全相同的命令。输入摘要匹配的阶段只显示 `SKIP`，第一个未完成阶段才会启动：

```text
PLAN     共 8 个阶段，从 IMAGE_PULL_OR_BUILD 继续
SKIP     [1/8] 安装用户运行时：已有可信检查点
SKIP     [2/8] 验证容器宿主：已有可信检查点
SKIP     [3/8] 解析 stable release：已有可信检查点
START    [4/8] 获取或构建 stable 镜像
```

### 已经完成

```text
PLAN     共 8 个阶段，已经全部完成
SKIP     [1/8] 安装用户运行时：已有可信检查点
...
SKIP     [8/8] 完成安装：已有可信检查点
SUMMARY  安装完成，...
```

不会调用任何安装动作，也不会重建项目或重新拉取镜像；本次检查仍创建独立日志。

### 等待重启

```text
ACTION   [6/13] 检查重启状态（REBOOT_PENDING）：manual reboot is required
STATE    .../install-state.json
RESUME   sudo reboot；重启后重新执行同一条 install 命令
LOG      .../install-<UTC>-<pid>.log
```

执行 `sudo reboot`，启动后重新运行原命令。boot ID 未变化时不会继续，也不会重放 `HOST_APPLY`。

### 阶段失败或阻断

```text
FAIL     [6/8] 创建项目镜像与 Python 依赖（PROJECT_INIT），用时 03:12
CAUSE    PROJECT_INIT failed: uv download failed
STATE    .../install-state.json
RESUME   修复问题后重新执行同一条 install 命令；已完成阶段不会重放
LOG      .../install-<UTC>-<pid>.log
```

失败阶段没有检查点，不会错误显示 `PASS`。先检查 `CAUSE` 和日志中的命令尾部，修复网络、空间、权限或输入问题，再运行原命令。策略阻断使用 `BLOCKED`，恢复规则相同。不要以删除状态文件作为修复方式。

容器模式可直接指定：

```bash
./install.sh --mode container --project-dir "$HOME/ai-projects/demo"
```

完整模式成功准备宿主后，每个后续项目继续使用 container 模式即可。`v0.2.3` 会在没有 `--state-path` 时根据项目规范化绝对路径生成独立状态，不会复用其他项目的 full 状态：

```bash
./install.sh --mode container \
  --project-dir /app/test/video-lab \
  --project-name video-lab \
  --image-source pull
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
| `--verbose` | 显示命令、精确字节、阶段枚举和 debug 诊断 |
| `--quiet` | 终端只保留警告、失败、恢复、摘要和最终日志路径 |
| `--target-user USER` | 完整模式计划和项目所有者 |
| `--manifest PATH` | 严格 stable release manifest |
| `--source-root PATH` | 本地 build 使用的固定源码 checkout |
| `--state-path PATH` | 高级用法：显式覆盖项目恢复状态路径 |

## 状态与退出码

新项目的隐式状态目录为：

```text
~/.local/state/strix-halo-rocm-toolkit/projects/
```

文件名由清理后的项目目录名和规范化绝对路径 SHA-256 前缀组成。安装器在 `PLAN` 中显示 `状态=<绝对路径>（per-project|legacy|explicit）`。显式 `--state-path` 始终优先；若旧全局状态 `~/.local/state/strix-halo-rocm-toolkit/install-state.json` 的 `project_path` 与当前项目完全一致，则继续原地恢复。有效但属于其他项目的旧状态不会阻断新项目；无法安全识别的旧状态仍按原路径进入损坏状态处理，不能被自动绕过。同一项目更换模式仍会阻断。

每个成功阶段保存其规范 JSON 输入的 SHA-256。状态采用 `0600` 临时文件、文件 `fsync`、`os.replace` 和目录 `fsync`。固定的 toolkit 协调锁会串行化同一用户发起的全部安装流程，即使显式状态位于不同目录也不能并发执行宿主或项目动作；每个状态文件另有自己的非阻塞锁。缺少规范化 `project_path` 或其他必需身份的状态按损坏状态保留为 `install-state.corrupt.<UTC>.json`，不会猜测已完成动作，也不会要求删除旧状态。

`v0.2.1` 引入状态 schema 2，`v0.2.3` 保持该 schema。`v0.2.3` 可以接管已经完成 `BOOTSTRAP` 的 `v0.2.2` container 状态，包括已经全部完成的状态；它先用旧版本、旧源码 revision 和旧 source root 精确重建并验证原 `BOOTSTRAP` 摘要，然后只更新安装器元数据和该摘要，其他阶段摘要不变，也不重放任何动作。

已经到达 `HOST_VERIFY` 或更后阶段的 full 状态仍适用同一 `0.2.x` 补丁升级边界。摘要损坏、模式/项目/目标用户变化、其他引导输入变化或跨 minor 更新都会阻断。不要删除状态以绕过阻断；保留证据并按 `CAUSE` 修正输入。

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
