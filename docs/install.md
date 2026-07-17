# 安装与恢复

## 支持范围

完整工作站模式正式支持 Ubuntu 24.04.x AMD64、Ryzen AI Max+ 395、Radeon 8060S 和 inbox `amdgpu`。工具的所有模式都不会安装 `amd-debug-tools`、调用 `amd-ttm`、写入 `ttm.conf`、设置 `ttm.pages_limit` 或 `amdgpu.gttsize`，也不会要求 GTT/TTM 重启；观察值仅作为只读诊断事实。容器模式还不会写入内核、APT、固件、用户组或 Docker 安装，但仍要求 Docker daemon、`/dev/kfd`、DRM render node、实际设备 GID 和目标 GPU 身份全部通过。

Python 安装器必须使用 3.12。ROCm 与 PyTorch 只存在于容器镜像中，不向宿主 Python 安装。

## 交互模式

```bash
./install.sh
```

首页可选择完整工作站、仅容器平台或 doctor/repair。完整模式先输出内核计划，精确输入 `INSTALL-KERNEL` 后才允许内核写入；重启验证后再输出平台计划，精确输入 `APPLY` 后才准备 Docker、Buildx、诊断工具和设备组。Docker 组因具有等同 root 的 daemon 权限，使用单独的 yes/no 授权。主机写入由受审计的 sudo 边界执行，安装器本身和用户本地 launcher 保持目标用户所有权。

内核变更要求重启时，安装器记录当前 boot ID 和 `KERNEL_REBOOT_PENDING` 后以状态 1 退出。它不会运行 `reboot`。手工重启后再次执行原命令；boot ID 未变化时不会重复 apply，变化后先验证 OEM 6.17 内核、桌面和 GPU，再进入平台计划。平台应用后立即验证，不存在第二个重启检查点。

Host verify 的 `pass` 正常继续。满足最低 OEM 6.17 版本，且 GPU 设备权限和内核日志检查通过，但内核 patch 尚未列入已测清单时，报告为 `unverified`；安装器输出 `WARN`、记录内核与诊断码并继续。live TTM 值无论缺失或与其他机器不同都只记录为事实，不计算目标值或产生 mismatch。后续 `IMAGE_VERIFY` 仍强制执行 exact stable 镜像的 `gfx1151` Torch runtime 探针。`change-required`、`reboot-required` 或 `blocked` 仍然阻断。该放宽只适用于普通部署，不改变 stable release 的完整硬件资格门禁。

sudo host verify helper 要求显式的 `target_user`。root 只承担受限事实采集，设备组始终按目标用户的 passwd/group 数据计算，避免把 root 缺少 render 组错误报告为 `GPU.PERMISSION`。

## 实时进度与私有日志

默认模式逐行显示 `PLAN/SKIP/START/DETAIL/WAIT/PASS/SUMMARY`。每个阶段使用当前工作流的 `[i/n]` 位置；container 模式是 8 个阶段，full 模式是 17 个阶段。只有状态检查点成功写入后才会出现 `PASS`。运行中的命令 15 秒没有新输出时显示 `WAIT`，任何新输出都会重新开始计时；等待用户输入期间暂停心跳。

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

日志目录为 `0700`，文件为 `0600`。`v0.3.3` 不轮转或自动删除日志。Docker pull、BuildKit、uv/pip、wheel 下载和 sudo 宿主命令会实时刷新阶段活动；日志会脱敏常见凭据，但仍可能包含项目路径、镜像名和包名，分享前必须复核。

## 典型输出与恢复

下面省略了部分中间阶段和具体路径，实际输出始终是一行一个事件。

### 新安装

```text
PLAN     模式=container，项目=/app/test/video-lab，名称=video-lab
PLAN     状态=.../video-lab-<key>.json（per-project）
PLAN     镜像来源=pull，镜像仓库=auto（华为 SWR 优先，GHCR 回退），stable release=待解析
PLAN     共 8 个阶段，从 BOOTSTRAP 继续
LOG      .../install-<UTC>-<pid>.log
START    [1/8] 安装用户运行时
PASS     [1/8] 安装用户运行时，用时 00:01
START    [4/8] 获取或构建 stable 镜像
DETAIL   缺失层=10.0 GiB，需要=15.0 GiB，可用=100.0 GiB，...
DETAIL   当前仓库=华为 SWR，来源=公开匿名镜像
WAIT     [4/8] 已运行 00:15，15 秒内没有新输出，仍在获取或构建 stable 镜像
PASS     [4/8] 获取或构建 stable 镜像，用时 08:42
SUMMARY  安装完成，...
LOG      .../install-<UTC>-<pid>.log
```

### 中断后恢复

重新执行完全相同的命令。输入摘要匹配的写入阶段显示 `SKIP`，第一个未完成阶段才会启动。full 模式例外：已经完成的 `KERNEL_VERIFY` 和 `HOST_VERIFY` 仍会以 `START/PASS` 重新执行，确保当前启动内核、桌面、GPU 和日志仍然有效；它们不会写入宿主。

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
ACTION   [6/17] 等待内核重启（KERNEL_REBOOT_PENDING）：manual reboot is required
STATE    .../install-state.json
RESUME   sudo reboot；重启后重新执行同一条 install 命令
LOG      .../install-<UTC>-<pid>.log
```

执行 `sudo reboot`，启动后重新运行原命令。boot ID 未变化时不会继续，也不会重放 `KERNEL_APPLY`。内核验证通过后才生成平台计划，平台应用不会要求重启。

### 阶段失败或阻断

```text
FAIL     [6/8] 创建项目镜像与 Python 依赖（PROJECT_INIT），用时 03:12
CAUSE    PROJECT_INIT failed: uv download failed
STATE    .../install-state.json
RESUME   修复问题后重新执行同一条 install 命令；已完成写入阶段不会重放
LOG      .../install-<UTC>-<pid>.log
```

失败阶段没有检查点，不会错误显示 `PASS`。先检查 `CAUSE` 和日志中的命令尾部，修复网络、空间、权限或输入问题，再运行原命令。策略阻断使用 `BLOCKED`，恢复规则相同。不要以删除状态文件作为修复方式。

`v0.3.3` 会把验证报告中的 finding code、摘要和 remediation 直接写入 `CAUSE`，并把状态及 finding code 保存在项目状态中。例如：

```text
BLOCKED  [11/17] 验证宿主平台（HOST_VERIFY），用时 00:00
CAUSE    host-verify returned change-required; GPU.PERMISSION: The current user lacks one or more GPU device groups; action: Add the target user to the device groups and start a new login session.
STATE    .../install-state.json
RESUME   修复问题后重新执行同一条 install 命令；已完成阶段不会重放
```

该结果表示只读验证发现尚未满足的宿主条件，不表示前一阶段的 Docker 或 Buildx 安装失败。修复 `CAUSE` 指明的问题后，重新登录或调整 BIOS（按 finding 而定），再执行完全相同的安装命令。安装器会重验内核与宿主，但不会重放已经完成的写入阶段。

容器模式可直接指定：

```bash
./install.sh --mode container --project-dir "$HOME/ai-projects/demo"
```

完整模式成功准备宿主后，每个后续项目继续使用 container 模式即可。`v0.3.3` 会在没有 `--state-path` 时根据项目规范化绝对路径生成独立状态，不会复用其他项目的 full 状态：

```bash
./install.sh --mode container \
  --project-dir /app/test/video-lab \
  --project-name video-lab \
  --image-source pull
```

默认 `--registry auto` 先从公开华为云 SWR 匿名获取 stable release 的 exact
digest，仅在 manifest 查询或拉取等获取失败时回退 GHCR。两个 registry 都
获取失败后才提供本地 build 选择。已经拉取的 config、RepoDigest、label 或
内嵌摘要不匹配属于身份失败，会直接阻断，不会跨 registry 回退。用户不需要
SWR 或 GHCR 凭据。

可以强制单一来源：

```bash
./install.sh --mode container --project-dir "$HOME/ai-projects/demo" \
  --registry swr
./install.sh --mode container --project-dir "$HOME/ai-projects/demo" \
  --registry ghcr
```

公开 SWR exact references 为：

```text
swr.cn-east-3.myhuaweicloud.com/hellcat-home/strix-halo-rocm-python@sha256:e9991f97f578156c8620fbb587d2d34504eb632f165cc5597deaadaa3e692a12
swr.cn-east-3.myhuaweicloud.com/hellcat-home/strix-halo-rocm-pytorch@sha256:dc0bb217474cfd4f602423bd3bf4fe8714b03e900cf3c6b4417b99e622ebcf8b
```

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

完整模式使用两个独立的 64 位摘要。首次在 `INSTALL-KERNEL` 提示处拒绝，读取状态中的 `kernel_plan_digest`，然后运行：

```bash
./install.sh \
  --mode full \
  --non-interactive \
  --project-dir /srv/video-lab \
  --project-name video-lab \
  --image-source pull \
  --accept-kernel-plan-digest <64-lowercase-hex>
```

安装器应用内核后停在 `KERNEL_REBOOT_PENDING`。手工重启并原样重跑；通过内核检查后，它生成平台计划，并因缺少平台摘要而停止。审阅状态中的 `host_plan_digest` 后，同时提供两个摘要完成安装：

```bash
./install.sh \
  --mode full \
  --non-interactive \
  --project-dir /srv/video-lab \
  --project-name video-lab \
  --image-source pull \
  --accept-kernel-plan-digest <kernel-64-lowercase-hex> \
  --accept-host-plan-digest <host-64-lowercase-hex> \
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
| `--accept-kernel-plan-digest HEX` | 无交互审批内核计划 |
| `--accept-host-plan-digest HEX` | 无交互审批平台计划 |
| `--manifest PATH` | 严格 stable release manifest |
| `--registry auto` | 默认：华为 SWR 优先，获取失败时回退 GHCR |
| `--registry swr` | 只使用公开华为 SWR |
| `--registry ghcr` | 只使用公开 canonical GHCR |
| `--source-root PATH` | 本地 build 使用的固定源码 checkout |
| `--state-path PATH` | 高级用法：显式覆盖项目恢复状态路径 |

## 状态与退出码

新项目的隐式状态目录为：

```text
~/.local/state/strix-halo-rocm-toolkit/projects/
```

文件名由清理后的项目目录名和规范化绝对路径 SHA-256 前缀组成。安装器在 `PLAN` 中显示 `状态=<绝对路径>（per-project|legacy|explicit）`。显式 `--state-path` 始终优先；若旧全局状态 `~/.local/state/strix-halo-rocm-toolkit/install-state.json` 的 `project_path` 与当前项目完全一致，则继续原地恢复。有效但属于其他项目的旧状态不会阻断新项目；无法安全识别的旧状态仍按原路径进入损坏状态处理，不能被自动绕过。同一项目更换模式仍会阻断。

每个成功阶段保存其规范 JSON 输入的 SHA-256。状态采用 `0600` 临时文件、文件 `fsync`、`os.replace` 和目录 `fsync`。固定的 toolkit 协调锁会串行化同一用户发起的全部安装流程，即使显式状态位于不同目录也不能并发执行宿主或项目动作；每个状态文件另有自己的非阻塞锁。缺少规范化 `project_path` 或其他必需身份的状态按损坏状态保留为 `install-state.corrupt.<UTC>.json`，不会猜测已完成动作，也不会要求删除旧状态。

`v0.3.3` 继续使用状态 schema 3，记录独立的内核计划、内核 boot ID、恢复
内核、显示管理器状态、平台计划及阻断验证结果。已经完成的 `v0.3.2`
GHCR 镜像检查点升级后不会重放拉取，状态仍保留当时验证的 GHCR exact
reference；新拉取则保存实际验证成功的 SWR 或 GHCR reference。
`v0.3.1` full 状态从 `HOST_VERIFY` 起可以原地接管，因此停在
`IMAGE_PULL_OR_BUILD` 的下载后校验失败可直接恢复；`v0.3.0` full 状态从
`KERNEL_VERIFY` 起也可接管。container 状态在完成 `BOOTSTRAP` 后可以接管；
可重建旧 `BOOTSTRAP` 摘要的 `0.2.x` container 状态可以跨 minor 接管；
其他 container 状态会阻断。

`0.2.x` full 状态升级后从 `BOOTSTRAP` 重新评估，但保留不可变镜像引用，且不会继承旧 host plan 审批。新的内核与平台计划必须重新审阅。摘要损坏、模式/项目/目标用户变化或其他引导输入变化仍会阻断。不要删除状态以绕过阻断；保留证据并按 `CAUSE` 修正输入。

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
