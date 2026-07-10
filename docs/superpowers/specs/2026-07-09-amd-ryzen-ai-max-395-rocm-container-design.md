# AMD Ryzen AI Max+ 395 ROCm 容器开发环境设计

- 日期：2026-07-09
- 状态：书面规格已确认
- 首发平台：AMD64、Ubuntu 24.04.x Desktop、AMD Ryzen AI Max+ 395 / Radeon 8060S iGPU (`gfx1151`)
- 正式计算基线：ROCm 7.2.1、Python 3.12、PyTorch 2.9.1

## 1. 背景与决策摘要

目标是在 Ryzen AI Max+ 395 上提供一套可复现、可诊断、适合 ComfyUI 和视频生成类大规模 AI 项目的 GPU 开发环境。宿主机只负责内核、固件、`amdgpu` 驱动、设备权限和 Docker；ROCm 用户态、Python、PyTorch 与编译工具链均封装在容器镜像中。

正式基线固定为 ROCm 7.2.1 + Python 3.12 + PyTorch 2.9.1。每个项目使用独立容器和独立项目镜像，不再额外创建项目级 Python venv。PyTorch 放在可复用的父镜像层中，使多个项目镜像通过 Docker 内容寻址层共享同一份大体积文件。

系统同时提供不含 PyTorch 的基础镜像，允许用户安装自选 PyTorch 组合。自选组合必须整体声明 Torch、TorchVision、TorchAudio 和 Triton，且默认标记为未经验证；它不能通过卸载正式基线镜像中的 PyTorch 来实现，以免旧文件仍留在下层并造成空间翻倍。

首发正式支持 Ubuntu 24.04.x + `linux-oem-24.04` 内核线。容器镜像本身不依赖宿主发行版的用户态，但 GPU 驱动和主机修复高度依赖内核及发行版，因此其他 Linux 发行版只预留适配器接口，不纳入首发支持承诺。

## 2. 已观测的目标主机状态

设计阶段的本机快照如下：

| 项目 | 观测值 |
| --- | --- |
| 主机系统 | Ubuntu 24.04.3 Desktop AMD64 |
| 当前内核 | `6.17.0-1025-oem` |
| GPU PCI ID | `1002:1586`，已由内核 `amdgpu` 接管 |
| 物理内存 | `/proc/meminfo` 为 131,015,488 KiB，约 124.95 GiB，对应标称 128 GiB |
| 当前 TTM 上限 | `ttm.pages_limit=33554432`，即 128 GiB 的 4 KiB 页数 |
| 当前兼容参数 | `amdgpu.gttsize=131072`，即 128 GiB |
| 其他现有参数 | `amdgpu.mcbp=0 amdgpu.gpu_recovery=1 amdgpu.cwsr_enable=0` |
| Swap | 0 |

当前执行环境未映射 `/dev/kfd`，因此其中的 `rocminfo` 失败不能单独证明宿主驱动失败。正式诊断必须区分“宿主没有设备”“容器没有映射设备”和“ROCm 用户态不兼容”三类问题。

主机疑似残留 ROCm 6.4 用户态包、软件源或失败的 DKMS 安装。准备脚本必须先盘点再清理，不得假设所有名称包含 AMD 的包都可删除。

## 3. 目标与非目标

### 3.1 目标

1. 在 Radeon 8060S 上稳定暴露 `gfx1151`，运行 ROCm、PyTorch、HIP C++ 扩展和 Triton 内核。
2. 提供完整 Python 与 C++/HIP 开发工具链，适合构建项目原生扩展。
3. 用父镜像层复用 PyTorch，避免每个项目 venv 重复占用数 GB 至数十 GB。
4. 提供可审计、幂等、默认保守的一键主机准备和验证脚本。
5. 自动检测主机内存，并按 AI Max 策略配置尽可能大的 TTM/GTT 可分配上限。
6. 允许用户自由决定项目源码、模型目录、Hugging Face 缓存及生成结果如何挂载。
7. 将稳定基线与用户自选 PyTorch 清晰隔离，并在启动时验证实际运行栈。

### 3.2 非目标

1. 不预装 ComfyUI、视频模型或任何具体 AI 应用。
2. 不强制共享 ComfyUI 模型库、Hugging Face 缓存、MIOpen/Triton/Inductor 缓存。
3. 不在宿主机安装完整 ROCm 用户态开发栈。
4. 不自动修改 BIOS 的 UMA Frame Buffer；脚本只检测并给出明确操作要求。
5. 不承诺自动回滚已安装或切换的 Linux 内核。
6. 不首发承诺 Fedora、Arch、Debian 或其他 Ubuntu 大版本上的主机修复能力。
7. 不默认注入 `HSA_OVERRIDE_GFX_VERSION` 或未经 AMD 文档确认的社区补丁参数。

## 4. 支持边界

| 层级 | 首发状态 | 说明 |
| --- | --- | --- |
| Ubuntu 24.04.x Desktop + OEM 内核 | 正式支持 | 主机脚本可执行审计、清理、准备和验证 |
| Ubuntu 24.04.x Server + OEM 内核 | 条件支持 | 与 Desktop 使用同一适配器，但需通过全部验收测试 |
| 其他 Ubuntu 版本 | 实验性 | 只允许审计；不自动修改主机 |
| 其他 Linux 发行版 | 实验性 | 可尝试运行镜像；主机准备适配器暂不提供 |
| Windows/WSL | 不支持 | 不属于本方案范围 |
| 非 AMD64 架构 | 不支持 | 首发只构建 `linux/amd64` |

“正式支持”表示本项目会验证脚本、设备映射和固定软件栈；它不扩大 AMD 官方兼容矩阵。若实际 OEM 内核补丁版本不在 AMD 对 ROCm 7.2.1 的验证范围内，预检报告必须明确标为“项目可测试、上游未认证”，而不能静默显示为完全兼容。

## 5. 总体架构

系统分为四层：

1. **宿主层**：Ubuntu、OEM 内核、Linux 固件、内核自带 `amdgpu`、TTM 参数、Docker 和设备组权限。
2. **ROCm/Python 基础镜像**：固定 ROCm 7.2.1 用户态、HIP SDK、Python 3.12、`uv` 和完整编译工具链，不含 PyTorch。
3. **正式 PyTorch 父镜像**：在基础镜像上安装并验证 PyTorch 2.9.1 全套组合。
4. **项目镜像/容器**：只添加项目代码和项目依赖；每个项目独立构建、独立运行。

宿主内核驱动通过 `/dev/kfd` 和 `/dev/dri` 向容器提供设备接口。容器不安装 `amdgpu-dkms`，也不携带或替换宿主内核模块。

### 5.1 镜像层次与标签

| 镜像 | 父级 | 内容 |
| --- | --- | --- |
| `rocm-python:7.2.1-py3.12` | 固定摘要的 `ubuntu:24.04` | ROCm/HIP SDK、Python、`uv`、构建与诊断工具，无 Torch |
| `rocm-pytorch:7.2.1-py3.12-torch2.9.1` | `rocm-python` | 正式 PyTorch 组合和镜像内自检 |
| `project-name:runtime` | 上述任一镜像 | 单一项目代码、锁定依赖、入口命令 |

发布标签之外同时记录 OCI 镜像摘要。部署记录以摘要为准，避免同名标签后续漂移。

两个父镜像都直接从固定摘要的 Ubuntu 基础构建，不继承通用 `rocm/pytorch` 镜像。这样可以明确控制 ROCm 包、开发工具和 Python wheels，避免把当前场景不需要的 HPC 组件或另一套 Python 环境带入下层。

## 6. 固定软件栈

正式 profile 固定以下版本，并使用 AMD ROCm 7.2.1 官方 wheel 来源：

| 组件 | 版本 |
| --- | --- |
| Ubuntu userspace | 24.04 |
| ROCm userspace / HIP SDK | 7.2.1 |
| Python | 3.12 |
| PyTorch | 2.9.1 |
| TorchVision | 0.24.0 |
| TorchAudio | 2.9.0 |
| Triton | 3.5.1 |
| `uv` | 在构建锁文件中固定精确版本 |

基础 OS 摘要、APT 仓库签名密钥、ROCm Debian 包精确版本、Python wheels URL 与 SHA-256 均写入版本化锁清单。构建不得使用无版本范围的 `latest` 包，也不得依赖构建当天解析出的 PyPI 最新版本。

ROCm 7.2.1 是正式发布基线。ROCm 7.2.4 或后续版本只能作为新的候选 profile 进入完整验收，不能就地替换现有标签。该策略避免把 ComfyUI、Wan、LTX 等上层工作负载中尚未完成回归验证的行为变化带入稳定环境。

## 7. 基础镜像内容

`rocm-python` 至少包含：

- ROCm 7.2.1 HIP runtime、HIP SDK、`hipcc`、开发头文件、`rocminfo` 和必要诊断库。
- Python 3.12、开发头文件、`pip` 兼容入口及固定版本 `uv`。
- GCC、G++、Make、CMake、Ninja、`pkg-config`。
- Git、curl、wget、CA 证书、常用压缩/解压工具和进程/设备诊断工具。
- 非 root 开发用户所需的目录、入口脚本和权限模型。

镜像使用多阶段构建和 BuildKit 缓存挂载降低构建流量。APT 索引、wheel 下载缓存和编译临时文件不得进入最终运行层。最终镜像保留运行及编译 HIP/PyTorch 扩展真正需要的工具，不通过删除开发头文件换取表面上的小体积。

## 8. Python 环境与存储策略

### 8.1 单一环境

镜像只维护 `/opt/venv` 一个 Python 环境，并将其 `bin` 目录设为默认 `PATH`。项目容器以 `/opt/venv` 作为实际运行环境，不在项目目录创建 `.venv`，也不在容器启动时重复安装依赖。

项目隔离由独立镜像和独立容器完成。这样既保留依赖隔离，也让 Docker 的内容寻址层在所有项目镜像间只保存一份完全相同的 PyTorch 父层。

### 8.2 项目依赖保护

正式基线项目通过约束文件固定 Torch、TorchVision、TorchAudio 和 Triton。项目依赖安装前后都记录四个包的版本、来源和文件摘要；若安装过程试图升级、降级、卸载或覆盖其中任一包，构建立即失败。

项目模板采用“安装项目依赖到已有 `/opt/venv`”的模式，不使用会删除未列出父层包的同步命令。项目镜像完成后执行 `pip check` 等价检查和 GPU 栈元数据检查。

`uv` 的下载缓存只用于镜像构建加速，不作为多个运行中容器必须共享的持久卷。空间复用的主要机制是 Docker 镜像层，而不是跨 venv 的可变缓存或软链接。

### 8.3 垃圾回收

镜像维护命令先展示引用关系和预计释放空间，再允许删除未被项目引用的构建缓存和旧 profile。不得默认运行会删除正在使用镜像的全局 `docker system prune`。

## 9. 用户自选 PyTorch profile

自选 PyTorch 必须从 `rocm-python:7.2.1-py3.12` 构建，禁止从正式 `rocm-pytorch` 镜像中卸载或覆盖 Torch。原因是 Docker 下层不可变，覆盖只会新增一份文件而不会回收原版本。

每个 profile 必须整体声明：

- 唯一且不可变的 profile ID。
- Torch、TorchVision、TorchAudio、Triton 的精确版本。
- wheel 索引或每个 wheel 的 HTTPS URL。
- 每个下载项的 SHA-256。
- 目标 Python ABI、ROCm 版本和 `linux/amd64` 平台。
- profile 状态：`verified` 或 `experimental`。

官方基线 profile 是唯一默认 `verified` 的组合。自定义 profile 默认是 `experimental`，镜像带有对应 OCI 标签。正式启动器拒绝运行实验性镜像，除非用户显式设置 `ALLOW_UNVERIFIED=1`。

即使用户允许实验性组合，入口自检仍必须执行：导入四个组件、检查 `torch.version.hip`、确认 `torch.cuda.is_available()`、确认设备架构为 `gfx1151`，并完成一次小型 GPU 张量运算。版本选择自由不等于跳过可观测性或静默回退 CPU。

## 10. 宿主准备设计

宿主工具采用审计、应用、重启后验证三个阶段。所有修改操作需要 `sudo`，审计阶段不需要写系统文件。

### 10.1 `host-preflight`

只读采集并同时输出人类可读报告和 JSON 报告：

- Ubuntu 版本、架构、当前内核、已安装 OEM 内核。
- CPU/GPU 型号、PCI ID、使用中的内核驱动、固件与 `dmesg` 关键错误。
- `/dev/kfd`、`/dev/dri/render*` 的存在性、属主、组 GID 和当前用户访问权限。
- 已安装的 ROCm、HIP、HSA、AMD DKMS 包及其来源。
- ROCm/AMD APT 源和签名密钥。
- DKMS 状态，并识别是否存在与 GPU 无关的 DKMS 模块。
- Docker 版本、daemon 状态、当前用户权限和存储驱动。
- 物理内存、Swap、当前内核命令行、TTM 参数及 BIOS UMA 提示。

审计结果使用明确分级：通过、需要重启、需要修改、上游未认证、阻断。脚本不得把 `/dev/kfd` 未映射到当前容器误报为宿主没有 GPU。

### 10.2 `host-prepare`

`host-prepare` 默认只显示计划；执行 `apply` 后仍在修改前列出将移除、安装和修改的对象并要求确认。其顺序为：

1. 保存诊断快照、APT 源、GRUB 配置、已安装包清单、DKMS 状态、内核与设备信息。
2. 禁用并移除确认属于旧 ROCm 6.4 用户态的 APT 源、密钥和包。
3. 移除 AMD GPU 专用 `amdgpu-dkms`，保留内核自带 `amdgpu`。
4. 保留通用 `dkms` 框架及其他厂商模块；只有依赖分析证明无其他用途时才允许单独建议移除。
5. 确保 `linux-oem-24.04`、匹配 headers 和当前 Linux firmware 已安装。
6. 检测可用 Docker；若不存在，再从 Docker 官方 Ubuntu 仓库安装 Engine 与 BuildKit/Compose 插件。
7. 配置设备组访问和 TTM 参数，更新 GRUB/initramfs 中确有必要的部分。
8. 输出变更摘要和重启要求。只有显式传入 `--reboot` 才执行重启。

主机不安装完整 ROCm 用户态。允许保留宿主诊断所必需且与内核驱动不冲突的最小工具，但其版本不得参与容器内软件解析。

脚本必须幂等：第二次运行不重复添加仓库、参数、组成员或配置行。APT 安装失败、磁盘空间不足、Secure Boot/DKMS 状态异常或当前内核不满足条件时，停止后续变更并保留报告。

若用户选择通过 `docker` 组直接调用 daemon，脚本必须提示该组具备近似 root 的主机控制能力并单独确认；否则保留 `sudo docker` 或 rootless 模式，不静默放宽权限。

### 10.3 `host-verify`

重启后验证以下事实：

- 正在运行预期 OEM 内核，`amdgpu` 来自内核而非外部 DKMS。
- Radeon 8060S 和 `gfx1151` 设备节点存在，普通目标用户具备访问权限。
- TTM 实际参数与计算值一致。
- Docker 可运行设备探针容器，探针容器内 `rocminfo` 能看到 `gfx1151`。
- 内核日志没有初始化失败、持续 GPU reset、MES timeout 或 page fault 风暴。

### 10.4 其他发行版适配接口

公共层只负责不修改系统的硬件、内存、设备节点和容器 runtime 探测。任何会写宿主机的操作都必须由显式匹配的发行版适配器实现，适配器至少声明系统版本范围、内核策略、包管理器、驱动来源、回滚边界和验收 fixture。未知发行版只能运行公共审计与容器探针，`host-prepare apply` 必须拒绝执行。首发只实现 Ubuntu 24.04 适配器。

## 11. TTM/GTT 与 BIOS 策略

### 11.1 策略原则

Ryzen AI Max+ 395 使用统一物理内存。BIOS 中专用显存/UMA Frame Buffer 应按 AMD 建议保持最小可用值，目标为 0.5 GiB；其余内存由系统保留为普通 RAM，并允许 GPU 通过 TTM/GTT 按需映射。脚本无法可靠修改 BIOS，因此只报告当前可见信息和重启进入 BIOS 的操作要求。

TTM/GTT 数值表示 GPU 可映射上限，不是在开机时立即永久预留同等容量。正式默认策略名为 `ai-max`，不人为保留固定 16 GiB，也不把可分配上限压低到传统独显思路的数值。

### 11.2 自动计算

容量优先读取 DMI 内存设备总容量；DMI 不可用时，读取 `/proc/meminfo` 的 `MemTotal`。原始容量统一按 `ceil(memory_gib / 8) * 8` 向上归一到 8 GiB 档位。DMI 与 `MemTotal` 推导结果相差超过 8 GiB 时停止自动应用，要求用户通过显式容量参数确认。页大小从 `getconf PAGESIZE` 读取，随后计算：

```text
ttm.pages_limit = nominal_memory_gib * 1024^3 / page_size_bytes
legacy_gttsize_mib = nominal_memory_gib * 1024
```

对本机标称 128 GiB，结果为：

```text
ttm.pages_limit=33554432
amdgpu.gttsize=131072
```

新配置以当前内核的 TTM 参数和 AMD `amd-ttm` 方法为主。内核文档已将 `amdgpu.gttsize` 标为弃用，因此新主机不默认新增该参数；如果现有系统已用该值稳定工作则保留，只有兼容性测试证明需要时才作为回退写入。

现有 `amdgpu.cwsr_enable=0`、`amdgpu.mcbp=0` 等参数在升级时先保留并记录，避免一次变更多个变量。全新主机不默认加入这些社区常用规避项；发生可复现的 MES/CWSR 挂起时，再通过显式诊断 profile 单独测试。`amdgpu.gpu_recovery=1` 同样由审计报告说明，而非无条件复制到所有机器。

### 11.3 修改安全

写入前备份 `/etc/default/grub` 及相关 drop-in。生成新配置后先做语法和重复参数检查，再运行 `update-grub`。配置回退可以恢复备份；内核包安装本身不承诺自动回滚。任何 TTM 修改都只在重启后由 `host-verify` 判定生效。

## 12. 容器运行契约

`project-run` 根据设备文件实际 GID 构造运行参数，不硬编码只有 `video` 组：

- 映射 `/dev/kfd` 和 `/dev/dri`。
- 动态附加 `/dev/kfd` 及所有使用中 render node 的组 GID。
- 默认使用与宿主项目所有者一致的 UID/GID，以避免输出文件归 root。
- 默认使用私有 IPC namespace，不使用 `--ipc=host`。
- 共享内存默认值为 `max(4 GiB, min(16 GiB, nominal_memory_gib/8))`，其中 `nominal_memory_gib` 使用第 11.2 节的内存档位归一算法；本机按 128 GiB 档计算为 16 GiB，用户可显式覆盖。
- 正常模式不启用 `--privileged`，不增加额外 capability，也不放宽 seccomp。
- 调试模式仅按需增加 `SYS_PTRACE` 和调试所需 seccomp 设置，并在启动日志中醒目标记。

入口检查失败时必须以非零状态退出，不能自动切换 CPU。报告至少区分：设备节点缺失、设备未映射、组权限不足、宿主驱动异常、ROCm 用户态加载失败、软件版本不匹配和 GPU 运算失败。

## 13. 数据、模型和缓存

项目运行器不创建全局模型卷，也不自动设置 `HF_HOME`、`HF_HUB_CACHE`、ComfyUI 模型目录、MIOpen cache、Triton cache 或 Torch Inductor cache。

默认只挂载当前项目明确配置的工作目录。用户可在项目配置中添加任意 bind mount 或 Docker volume，并逐项选择只读或读写。由此支持以下两种同等合法的用法：

1. 每个项目完全私有的模型和缓存。
2. 用户主动将某个现有模型目录挂载给多个项目。

运行器只校验路径存在性、访问模式和容器目标路径冲突，不替用户决定目录结构，不扫描或重排模型文件，也不提供强制性的 ComfyUI/Hugging Face 共享策略。

## 14. 工具与仓库结构

实现阶段按以下职责划分：

| 工具 | 职责 |
| --- | --- |
| `bin/host-preflight` | 只读主机审计，输出文本和 JSON |
| `bin/host-prepare` | 备份、清理旧栈、准备 OEM 内核/Docker/TTM |
| `bin/host-verify` | 重启后宿主与探针容器验证 |
| `bin/image-build` | 按锁清单构建基础、正式或自定义 profile 镜像 |
| `bin/project-init` | 生成独立项目镜像模板和配置 |
| `bin/project-run` | 校验并启动项目容器，处理设备、GID、UID、IPC 和挂载 |
| `bin/container-check` | 在镜像或运行容器内执行 ROCm/PyTorch/HIP/Triton 检查 |

预计仓库边界如下：

```text
bin/                    用户入口
host/                   Ubuntu 24.04 主机适配器与共享审计逻辑
images/rocm-python/     无 Torch 基础镜像
images/rocm-pytorch/    正式 PyTorch 镜像
profiles/torch/         固定且可审计的完整 Torch profiles
templates/project/      项目镜像模板
tests/                  静态、容器和 GPU 验收测试
docs/                   运维、故障定位和发布文档
```

每个工具支持 `--help` 和 `--json` 或等价机器可读输出。主机工具保留执行日志；日志在写盘前过滤环境变量中的 token、仓库凭据和代理密码。

## 15. 标准工作流

新主机的顺序固定为：

```bash
./bin/host-preflight
sudo ./bin/host-prepare apply
# 按报告重启主机
./bin/image-build rocm-python
./bin/host-verify --probe-image rocm-python:7.2.1-py3.12
./bin/image-build rocm-pytorch --profile profiles/torch/stable.env
./bin/project-init demo --base stable
./bin/project-run demo
```

自选 PyTorch 使用无 Torch 基础镜像构建完整 profile：

```bash
./bin/image-build --profile-file profiles/torch/custom.env
./bin/project-init demo-custom --base-profile custom
ALLOW_UNVERIFIED=1 ./bin/project-run demo-custom
```

上述 `custom.env` 声明的 profile ID 为 `custom`；`project-init` 必须按该 ID 写入不可变父镜像摘要，而不是只记录可漂移标签。`project-init` 生成的项目文件归当前用户所有，未指定 `--base-profile` 时继承正式基线。所有步骤均可单独重跑，不要求通过一个不可观察的总脚本隐藏主机重启或高风险修改。

## 16. 错误处理与恢复边界

1. 审计失败不改主机，并保留部分报告。
2. 主机应用阶段每完成一个可恢复步骤就更新状态清单；重新运行时根据实际系统状态继续。
3. APT 或内核安装失败后不继续修改 GRUB 和设备权限。
4. GRUB 修改失败时恢复本次配置备份，但不宣称回滚已安装包。
5. 容器构建的任何版本、哈希或导入检查失败都会使镜像构建失败，不发布半成品标签。
6. GPU 自检失败不会自动启用 `HSA_OVERRIDE_GFX_VERSION`、关闭 CWSR 或下载社区补丁；报告提供证据后由单项诊断流程处理。
7. 上层项目需要补丁时，补丁保存在项目镜像及其锁文件中，不污染正式基础镜像。

## 17. 验收测试

### 17.1 无 GPU 的静态与构建测试

- Shell 脚本格式、静态分析和参数解析测试。
- Ubuntu 版本、包残留、DKMS、设备 GID、内存档位和 GRUB 参数的 fixture 测试。
- 两个基础镜像可从干净 BuildKit 缓存构建。
- 检查锁定版本、下载哈希、OCI 标签、非 root 默认用户和最终层中无 APT/wheel 临时缓存。
- 派生两个示例项目镜像，确认两者引用同一 PyTorch 父层，项目依赖安装不改动 Torch 组合。

### 17.2 目标硬件 GPU 测试

- `rocminfo` 能识别 `gfx1151`。
- ROCm 包清单和 `/opt/rocm` 版本元数据均为 7.2.1。
- PyTorch 的 `torch.version.hip` 与 profile 声明兼容，能识别 Radeon 8060S，架构字符串以 `gfx1151` 开头，且 `torch.cuda.is_available()` 为真。
- 执行 FP16 张量传输、矩阵乘法和卷积，并验证结果及同步无错误。
- 编译并运行最小 HIP 程序。
- 通过 `torch.utils.cpp_extension` 编译并运行最小 PyTorch C++/HIP 扩展。
- 编译并运行最小 Triton kernel。
- 进行持续 GPU 压力与反复启动测试，同时检查 `dmesg` 中的 page fault、MES timeout、GPU reset 和固件错误。
- 退出并再次启动容器，验证设备权限、缓存可选挂载和结果文件属主不漂移。

### 17.3 应用资格测试

ComfyUI、Wan、LTX 及具体视频工作流不进入基础镜像，但作为独立 qualification 项目执行长时测试。测试至少覆盖首次运行、第二次运行、模型切换、低显存/大 GTT 情况和容器重启。任何社区补丁只有在复现原问题、验证补丁来源并完成回归后，才可固定在该项目层；不能直接加入通用基线。

## 18. 发布门槛

只有同时满足以下条件的构建才能标记为正式：

1. 软件供应链锁定与哈希验证通过。
2. 静态、镜像层和无 GPU 测试通过。
3. Ryzen AI Max+ 395 实机上的全部 GPU 验收通过。
4. 至少一次持续压力测试后无未解释的 GPU reset、MES timeout 或 page fault 风暴。
5. 主机准备脚本在干净 Ubuntu 24.04.x 和带旧 ROCm/DKMS 残留的 fixture/测试机上均表现幂等。
6. 生成 SPDX 或等价 SBOM、镜像摘要、版本清单和验收报告。

升级 ROCm、OEM 内核、PyTorch 或 Triton 中任一核心组件都产生新的 profile 和镜像标签，并重新执行完整 GPU 验收；不得覆盖既有正式镜像。

## 19. 已知风险与控制

| 风险 | 控制措施 |
| --- | --- |
| Strix Halo 固件/MES/CWSR 相关挂起 | 固定已验证内核/固件组合，采集 `dmesg`，将规避参数隔离到诊断 profile |
| ComfyUI/Wan 第二次运行变慢或状态残留 | qualification 项目执行重复运行测试，不把应用补丁下沉到基础镜像 |
| 新 ROCm patch release 引入行为变化 | 7.2.1 标签不可变；7.2.4 及后续版本作为独立候选重新验收 |
| 项目依赖替换 PyTorch | 约束文件、安装前后摘要与构建失败策略 |
| 自定义 Torch 组合 ABI 不匹配 | 完整 profile、哈希、实验性标签、启动门禁和 GPU 冒烟测试 |
| 容器误判宿主 GPU 故障 | 分层诊断设备存在性、映射、GID、驱动和用户态加载 |
| TTM 上限过大被误解为立即占用 | 报告同时显示物理 RAM、上限语义、实际使用和 OOM 风险 |
| 一键脚本误删其他 DKMS/软件源 | 精确包归属、依赖检查、预览确认、备份和幂等执行 |

## 20. 依据

- [AMD Ryzen Linux 原生安装与 TTM 指南](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installryz/native_linux/install-ryzen.html)
- [AMD Ryzen Linux 兼容性矩阵](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityryz/native_linux/native_linux_compatibility.html)
- [ROCm 7.2.1 Ryzen PyTorch 安装说明](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2.1/docs/install/installryz/native_linux/install-pytorch.html)
- [ROCm 发布说明](https://rocm.docs.amd.com/en/latest/about/release-notes.html)
- [ROCm 容器设备映射说明](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/docker.html)
- [Linux 内核 AMDGPU 模块参数](https://docs.kernel.org/gpu/amdgpu/module-parameters.html)
- [Ubuntu OEM 内核 gfx1151/CWSR 相关变更](https://bugs.launchpad.net/ubuntu/%2Bsource/linux-oem-6.14/6.14.0-1018.18)
- [uv Docker 集成](https://docs.astral.sh/uv/guides/integration/docker/)
- [uv 缓存存储语义](https://docs.astral.sh/uv/reference/storage/)
- [ROCm issue 5724：Strix Halo 固件/MES 挂起案例](https://github.com/ROCm/ROCm/issues/5724)
- [ComfyUI issue 12672：Wan 第二次运行性能案例](https://github.com/Comfy-Org/ComfyUI/issues/12672)
