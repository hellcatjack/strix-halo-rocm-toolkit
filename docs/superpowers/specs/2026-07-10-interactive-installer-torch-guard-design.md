# 交互式一键安装与 Torch 防覆盖修复设计

- 日期：2026-07-10
- 状态：交互设计已确认，书面规格待复核
- 项目：`hellcatjack/strix-halo-rocm-toolkit`
- 正式平台：Ubuntu 24.04.x AMD64、Ryzen AI Max+ 395 / Radeon 8060S (`gfx1151`)
- 正式基线：ROCm 7.2.1、Python 3.12、PyTorch 2.9.1

## 1. 背景

当前平台已经用不可变父镜像、哈希锁、Torch 文件 manifest、项目镜像构建后检查和非 root 运行用户保护正式 Torch 栈。项目工具也禁止在 `requirements.in` 或派生镜像中升级、降级或用直接 URL 替换 `torch`、`torchvision`、`torchaudio` 和 `triton`。

仍有两个用户体验缺口：

1. 首次使用需要按顺序理解并执行宿主预检、主机准备、重启、镜像获取、GPU 验证和项目初始化，缺少一个可恢复的交互式编排入口。
2. 用户习惯在运行中的容器里执行 `pip install`。完全禁止该行为不符合实际开发流程；允许普通 pip 直接写 `/opt/venv` 又可能用 PyPI CPU/CUDA wheel 或错误 ROCm wheel 覆盖正式 `gfx1151` 栈。

本设计用双模式安装向导解决第一个问题，用“只读正式环境 + 项目持久 overlay + 事务化受保护 pip + 不可变重建修复”解决第二个问题。

## 2. 决策摘要

1. 同时提供“完整工作站安装”和“仅容器平台安装”。
2. 默认从 GHCR 匿名拉取 release manifest 指定的不可变 verified digest；没有匹配发布时才询问是否本地构建。
3. 托管项目容器始终使用只读根文件系统。应用需要写入其他位置时增加显式挂载，不提供 verified 模式下关闭只读根的快捷开关。
4. 允许用户在运行容器内直接执行常用 `pip install`。包安装到项目私有且持久的 overlay，不写 `/opt/venv`。
5. 四个受保护组件只能来自 verified 父镜像，不能进入 overlay。
6. pip 变更以新 generation 构建，验证通过后原子切换；失败不影响当前 generation。
7. 检测到 Torch 损坏或 shadow 时阻断启动并请求确认，不静默修复。
8. 修复只使用不可变重建：隔离受污染 overlay，重新拉取父 digest 或重建派生镜像，再从最后成功的 overlay 哈希锁恢复。

## 3. 目标与非目标

### 3.1 目标

1. 新用户可从单一入口完成宿主准备或容器平台部署。
2. 安装流程可审计、可中断、可跨重启恢复，并保留机器可读报告。
3. 现有 `bin/host-*`、`image-build`、`container-check` 和 `project-*` 继续作为底层稳定接口。
4. 常见 `pip install`、`pip install -r`、`pip list`、`pip show`、`pip check` 和 `pip freeze` 在运行容器中可用。
5. 运行时安装的非 Torch 包跨容器重启保留，但不跨项目共享。
6. 任何错误 Torch 安装都不能修改 `/opt/venv`，也不能在 overlay 中静默覆盖有效导入。
7. doctor 能区分父镜像漂移、派生镜像损坏、正式 Torch 文件损坏和 overlay shadow。
8. repair 在失败时保留旧环境与证据，在成功时恢复到同一 verified 发布链。

### 3.2 非目标

1. 不把 `/opt/venv` 改为跨项目共享卷。
2. 不在受污染容器内执行 `pip --force-reinstall` 修补 Torch。
3. 不保证保留用户绕过托管 pip 后产生的未声明容器内修改。
4. 不支持 verified 项目使用可写根文件系统。
5. 不自动接受宿主修改、Docker 组授权、重启或 experimental profile。
6. 不在首版自动恢复未固定 commit 的 VCS 依赖或 editable 安装。
7. 不预装 ComfyUI、模型、custom nodes 或共享模型缓存。

## 4. 用户入口

### 4.1 Bootstrap

仓库根目录增加：

```bash
./install.sh
```

`install.sh` 是短小、可审计的 Bash bootstrap。它只负责：

- 确认脚本从真实文件执行，交互模式要求 TTY；
- 定位 Python 3.12；
- 检查仓库结构和最低命令依赖；
- 调用 Python 标准库实现的安装向导；
- 透传明确的无交互参数。

它不解析远程 shell，不用 `eval`，不执行 `curl | sudo bash`，也不承载主机修改逻辑。正式文档优先使用固定 Git tag 的 clone 或 GitHub Release archive，再执行本地 `install.sh`。

### 4.2 统一命令

安装完成后在 `~/.local/bin` 提供：

```text
strix-halo-rocm
```

命令面如下：

```bash
strix-halo-rocm install
strix-halo-rocm doctor [PROJECT]
strix-halo-rocm repair PROJECT
strix-halo-rocm project init NAME
strix-halo-rocm project lock PROJECT
strix-halo-rocm project run PROJECT
```

现有独立脚本保持兼容。统一命令只编排已有模块和新 overlay 模块，不复制宿主、镜像或项目策略实现。

## 5. 安装模式与状态机

### 5.1 交互首页

```text
Strix Halo ROCm Toolkit

1. 完整工作站安装
2. 仅安装容器平台
3. 检查或修复已有安装
4. 退出
```

所有交互使用普通终端文本和编号选项，不依赖 `dialog`、`whiptail` 或第三方 TUI。显示状态统一为：

```text
PASS     已满足
WARN     可继续但需要注意
ACTION   将执行修改
BLOCKED  不允许继续
```

### 5.2 完整工作站模式

阶段顺序固定为：

```text
BOOTSTRAP
→ HOST_PREFLIGHT
→ HOST_PLAN
→ HOST_CONFIRM
→ HOST_APPLY
→ REBOOT_PENDING（需要时）
→ HOST_VERIFY
→ RELEASE_RESOLVE
→ IMAGE_PULL_OR_BUILD
→ IMAGE_VERIFY
→ PROJECT_INIT
→ PROJECT_VERIFY
→ COMPLETE
```

完整模式只在已有正式 HostAdapter 的发行版上开放主机写入。当前为 Ubuntu 24.04.x AMD64。其他发行版仍可运行公共只读预检，但不能通过安装向导绕过适配器。

向导展示现有 `host-prepare` 计划，并沿用精确 `APPLY`、Docker 组额外授权、备份和非自动重启边界。需要重启时写入 `REBOOT_PENDING` 后退出；用户重启并再次执行安装器时，从 host verify 继续。

### 5.3 仅容器平台模式

阶段顺序固定为：

```text
BOOTSTRAP
→ CONTAINER_HOST_CHECK
→ RELEASE_RESOLVE
→ IMAGE_PULL_OR_BUILD
→ IMAGE_VERIFY
→ PROJECT_INIT
→ PROJECT_VERIFY
→ COMPLETE
```

该模式不修改内核、APT、固件、TTM、用户组或 Docker 安装。它仍检查 Docker daemon、`/dev/kfd`、render node、实际 GID、宿主最低要求和 GPU probe。条件不满足时输出证据并阻断，不把 CPU fallback 当作成功。

### 5.4 无交互模式

示例：

```bash
./install.sh \
  --mode container \
  --non-interactive \
  --project-dir /srv/comfy-lab
```

无交互模式要求所有必要输入由参数或已验证状态提供。它不会隐式接受主机修改、重启、Docker 组授权、experimental 镜像或 digest 变化。需要人工授权时返回阻断退出码。

### 5.5 安装状态

状态原子写入：

```text
~/.local/state/strix-halo-rocm-toolkit/install-state.json
```

最低 schema：

```text
schema_version
installer_version
mode
target_user
release_id
source_revision
base_image_reference
base_manifest_digest
torch_image_reference
torch_manifest_digest
project_path
current_stage
completed_stage_input_digests
reboot_boot_id
created_at
updated_at
```

状态不保存 token、密码或完整环境变量。每个阶段记录其输入摘要；恢复时输入变化则停止自动续跑并重新展示计划。状态使用临时文件、`fsync` 和 `os.replace`，不能仅凭文件存在判定阶段完成。

## 6. Stable Release Manifest

仓库增加版本化 stable release manifest，例如：

```text
profiles/releases/stable.json
```

它区分 OCI manifest digest 与 Docker config/image ID，至少包含：

```text
schema_version
release_id
source_repository
source_revision
qualification_profile_digest
qualification_report_digest
sbom_digest
gpu_arch
supported_host_adapter_ids
rocm_version
python_version
torch_profile_id
torch_profile_digest
base.image
base.manifest_digest
base.config_digest
torch.image
torch.manifest_digest
torch.config_digest
published_at
```

镜像引用使用：

```text
ghcr.io/hellcatjack/strix-halo-rocm-python@sha256:<OCI manifest digest>
ghcr.io/hellcatjack/strix-halo-rocm-pytorch@sha256:<OCI manifest digest>
```

安装器不使用 `latest` 作为部署身份。友好标签只用于发现，最终 pull、状态和项目配置均绑定 digest。

发布 manifest 进入 stable 前必须满足：

- 源码 revision 已通过非硬件测试；
- 目标 `gfx1151` 主机已通过完整资格测试；
- 镜像标签、config digest、registry digest、profile 和锁摘要一致；
- SBOM 与资格报告摘要已记录；
- GHCR 中每个 OCI layer 满足 registry 限制并完成实际 pull 回归。

## 7. 托管项目运行边界

verified 项目运行参数固定增加：

```text
--read-only
--ipc=private
--tmpfs /tmp:rw,nosuid,nodev,size=<bounded>
--user <host uid>:<host gid>
--device /dev/kfd
--device /dev/dri
--group-add <actual gid>
```

继续使用有界 shm。默认 writable 路径只有：

- 项目 `/workspace` bind mount；
- 项目 `.amd-ai` 控制目录；
- 有界 `/tmp` tmpfs；
- 用户在项目配置中显式声明的挂载。

环境增加：

```text
PYTHONNOUSERSITE=1
PYTHONDONTWRITEBYTECODE=1
AMD_AI_OVERLAY=/workspace/.amd-ai/current/site-packages
PYTHONPATH=/workspace/.amd-ai/current/site-packages
```

`PYTHONPATH` 中 overlay 位于正式 site-packages 之前，因此任何受保护 distribution 出现在 overlay 都必须被有效栈检查阻断。应用需要写 MIOpen、Triton、Inductor、Hugging Face、模型或输出目录时，通过项目配置显式挂载，不关闭只读根。

## 8. 持久 Python Overlay

### 8.1 目录布局

每个项目独立维护：

```text
.amd-ai/
├── overlay.requirements.in
├── overlay.requirements.lock
├── overlay-state.json
├── transaction.lock
├── generations/
│   └── <transaction-id>/
│       └── site-packages/
├── current -> generations/<transaction-id>
└── quarantine/
```

`.amd-ai` 保持不进入 Docker build context，也不由镜像清理器删除。`current` 必须是项目控制目录内的相对 symlink，不能指向外部路径。

### 8.2 受保护 distributions

保护集合固定为：

```text
torch
torchvision
torchaudio
triton
```

正式约束记录完整 distribution version，包括 ROCm local version，例如 `2.9.1+rocm7.2.1...`。比较使用精确版本语义，不能只比较 `2.9.1` 公共版本。

overlay lock 明确禁止出现保护集合中的任何 distribution。解析器按规范化包名比较，覆盖大小写、`.`、`_`、`-`、extras、requirements include、direct URL 和 wheel 文件名变体。

直接或传递遇到受保护 requirement 时按以下规则处理：

- 没有版本条件，或其 specifier 接受父 profile 的完整 verified version：先验证父环境，再报告 `already satisfied by verified parent`，不下载、不复制，也不写入 overlay lock；
- specifier 不接受 verified version：在解析阶段以版本冲突阻断；
- direct URL、VCS 或本地 wheel 声明受保护名称：无论公开版本号是否相同都阻断，因为来源身份不等于批准 wheel；
- 卸载受保护名称：始终阻断。

因此 `pip install torch` 和常见项目 requirements 中的兼容 `torch>=...` 可以正常继续，同时 `torch==2.8.0`、CPU/CUDA wheel 或另一套 ROCm wheel 不能进入 overlay。

### 8.3 pip 命令

容器 PATH 最前方提供受保护的真实可执行入口 `pip` 和 `pip3`。首版支持：

```text
pip install <index requirement>
pip install -r <requirements file>
pip install <local source directory>
pip install <local wheel>
pip uninstall <top-level overlay requirement>
pip list
pip show
pip check
pip freeze
```

普通本地源码先构建 wheel、计算 SHA-256 后进入 lock。HTTPS direct wheel 必须解析名称、版本与下载哈希。以下输入首版拒绝并给出迁移提示：

- `--user`、`--target`、`--prefix`、`--root`；
- 未固定 commit 的 VCS requirement；
- editable (`-e`) 安装；
- 试图卸载保护集合；
- 无法形成可重放哈希锁的来源；
- 直接执行任意 pip 子命令逃逸到外部 Python 环境。

用户需要 editable 或复杂源码布局时，应把源码保留在 `/workspace`，或将安装步骤写入项目 Dockerfile 并通过项目镜像 manifest 门禁。

### 8.4 安装事务

每次变更执行：

1. 获取项目 `transaction.lock`；若另一个 install、run 或 repair 正在变更环境则停止。
2. 读取并验证当前 generation、top-level 输入与完整 hash lock。
3. 规范化新请求，并注入父 profile 的四组件精确约束。
4. 让解析器把保护集合视为由 verified 父环境外部满足。
5. 生成只包含非保护 distributions 的完整、精确、带哈希 overlay lock。
6. 在新 transaction ID 下从空 `site-packages` 构建 generation。
7. 执行 overlay lock 校验、依赖检查和有效 Torch 栈检查。
8. 原子写入新输入、lock 和 state。
9. 用临时 symlink 加 `os.replace` 原子切换 `current`。
10. 成功后保留前一 generation 到下次健康启动，随后按策略删除；失败则保留当前 generation 不变。

实现不得用 shell 字符串拼接需求参数。所有外部命令使用 argv 数组。pip/uv stdout 与 stderr 进入项目私有日志，凭据相关值按现有敏感名称规则脱敏。

## 9. Torch 有效栈验证

### 9.1 Base manifest

继续验证 `/opt/amd-ai/torch-manifest.json`：

- 四个 distribution 的版本；
- 所有记录文件存在；
- 文件大小和 SHA-256 一致。

manifest 本身还要与父镜像标签和 stable release manifest 中的 profile 摘要对应。

### 9.2 Effective import identity

仅验证 `/opt/venv` 文件不足以发现 overlay shadow。新增有效栈检查，在与项目相同的 `PYTHONPATH` 下检查：

- `importlib.metadata.distribution()` 返回的四个 distribution 全部位于 `/opt/venv`；
- `torch.__file__`、`torchvision.__file__`、`torchaudio.__file__` 和 Triton 模块路径位于 `/opt/venv`；
- 四组件完整版本等于 profile；
- `torch.version.hip` 等于正式 ROCm/HIP 版本族；
- overlay 中不存在保护集合的 package 目录、dist-info 或 egg-info；
- runtime 检查时 Torch 报告 `gfx1151` 并完成同步 GPU 运算。

每次 pip 事务前后执行静态有效栈检查。每次托管项目启动执行静态检查和现有 GPU runtime check。正式资格测试仍使用完整八项门禁。

### 9.3 绕过边界

`/opt/venv/bin/pip` 和 `python -m pip` 无法写只读根文件系统。`PYTHONNOUSERSITE=1` 防止 user site 自动进入导入路径。用户仍可主动绕过 PATH、环境或项目工具构造不受托管的容器；这类行为不宣称 verified，下一次托管启动会依据有效栈身份阻断，而不是静默接受。

## 10. Doctor

命令：

```bash
strix-halo-rocm doctor [PROJECT]
```

无项目参数时检查平台安装、release、镜像和 GPU；有项目参数时增加项目配置、派生镜像和 overlay。检查分类：

| 代码 | 含义 | 默认结果 |
| --- | --- | --- |
| `RELEASE.INVALID` | stable manifest schema 或摘要错误 | blocked |
| `IMAGE.PARENT_MISSING` | digest 指定的父镜像不存在 | repairable |
| `IMAGE.DIGEST_DRIFT` | 友好标签指向其他 image/config | repairable |
| `IMAGE.PROJECT_CHANGED` | 派生镜像父层、标签或 manifest 不符 | repairable |
| `TORCH.BASE_CHANGED` | `/opt/venv` 文件或版本变化 | blocked/repairable |
| `TORCH.SHADOWED` | overlay 或有效导入来自非正式路径 | blocked/repairable |
| `OVERLAY.LOCK_INVALID` | overlay 输入、lock 或 state 不一致 | blocked/repairable |
| `OVERLAY.TRANSACTION_INCOMPLETE` | 上次 generation 未完成 | warning/repairable |
| `GPU.RUNTIME_FAILED` | 实际 GPU 检查失败 | blocked |

`doctor --json PATH` 写完整证据。默认输出不得泄露 package index token、URL 凭据或环境 secret。

## 11. Repair

命令：

```bash
strix-halo-rocm repair PROJECT
```

repair 先运行 doctor，展示将隔离、重新拉取、删除和重建的精确对象，并要求确认。它不接受模糊 image pattern，也不调用 `docker system prune`。

### 11.1 Overlay 污染

1. 阻止新的项目启动并获取 transaction lock。
2. 将当前 generation 和相关 state 原子移动到：

```text
.amd-ai/quarantine/<UTC>-<reason>/
```

3. 从最后一份成功且通过摘要验证的 top-level 输入与 overlay hash lock 构建全新 generation。
4. 运行完整有效栈和依赖检查。
5. 只有成功才切换 `current`；失败时保持项目 blocked，保留 quarantine 和报告。

### 11.2 项目派生镜像损坏

1. 保留项目目录、`.amd-ai`、模型和全部显式挂载。
2. 删除或取消标记精确的受污染项目 image ID。
3. 从配置记录的 verified 父 digest、项目 `requirements.lock` 和 Docker build fingerprint 重建。
4. 重新验证父层前缀、OCI 标签、入口、UID、工作目录和 Torch manifest。

### 11.3 父镜像漂移或缺失

1. 从当前 stable release manifest 取得 GHCR exact manifest digest。
2. 匿名拉取 `image@sha256:...`，不回退到同名标签或 experimental profile。
3. 验证 RepoDigest、config/image ID、source revision、OCI 标签、嵌入 ROCm 锁、Torch profile 和 manifest。
4. 只在全部通过后重建本地友好标签和项目派生镜像。

### 11.4 修复完成条件

repair 只有在以下条件全部成立时返回 0：

- release chain 与 digest 一致；
- base manifest 与 effective import identity 通过；
- overlay lock 可重放且不含保护集合；
- 项目派生镜像 contract 通过；
- GPU runtime probe 报告 `gfx1151`；
- 没有新增阻断性内核日志。

## 12. 失败、并发与恢复

- 网络中断：保留 Docker 内容寻址层，允许重试，不更改已验证项目配置。
- Ctrl+C：当前阶段不标成功；pip generation 不切换。
- 磁盘不足：在 pull/build/generation 前按所需余量阻断。
- 重启：安装器按 boot ID 验证确实发生过重启，再进入 host verify。
- digest 更新：恢复时要求重新确认，不能把两个 release 的阶段拼接成一次安装。
- 并发：安装状态和每个项目 overlay 分别加文件锁；run 在切换 generation 时等待或停止。
- 损坏 state：保留原文件作为证据，从只读事实重新生成计划，不猜测完成状态。
- repair 失败：不启动项目，不删除 quarantine，不覆盖最后成功报告。

退出码沿用平台语义：`0` 成功，`1` 表示需操作/重启/明确 unverified，`2` 表示 blocked 或拒绝执行。

## 13. 模块边界

新增或扩展的逻辑按职责拆分：

```text
install.sh
bin/strix-halo-rocm
src/amd_ai/installer/
  models.py       安装和 release 数据模型
  state.py        原子状态、阶段摘要和恢复
  prompts.py      终端交互与无交互输入
  release.py      stable manifest 与镜像身份验证
  workflow.py     双模式状态机和现有命令编排
src/amd_ai/overlay/
  policy.py       pip 参数与保护集合策略
  resolver.py     外部满足 Torch 的结构化依赖解析
  lock.py         overlay 输入和哈希锁 schema
  transaction.py generation 构建与原子切换
  verify.py       base/effective Torch 身份检查
  repair.py       quarantine 与锁定重建
images/common/
  protected-pip   容器内 pip/pip3 入口
profiles/releases/
  stable.json
```

`cli.py` 只解析顶层参数并调用这些模块。installer 不直接复制 host/apply、image/build 或 project/run 的规则；overlay 不负责 Docker 拉取；repair 通过明确接口组合 release、image、project 和 overlay 检查。

## 14. 测试设计

### 14.1 单元测试

- 安装状态机每个合法/非法迁移；
- 重启 boot ID、阶段输入摘要和恢复拒绝；
- 交互确认、非 TTY、EOF 和无交互缺参；
- stable release manifest schema、digest 类型和 label 对应；
- pip 参数规范化与目标目录逃逸；
- 保护包大小写、分隔符、extras、URL、wheel 和 requirements include；
- overlay lock 禁止四组件且要求全部哈希；
- generation 临时目录、symlink 边界和原子切换；
- doctor 故障分类与 repair 计划只使用精确 ID；
- 日志和 dry-run 的 secret 脱敏。

### 14.2 容器集成测试

- verified 项目以只读根启动；
- 普通 `pip install` 写 overlay 并跨重启保留；
- `pip install -r` 的传递 Torch 依赖由父环境满足；
- 错误 Torch 版本、直接 URL、CPU wheel 和卸载请求被阻断；
- `--user`、`--target`、`--prefix` 无法绕过；
- 直接 base pip 无法修改 `/opt/venv`；
- 人工加入 overlay Torch 后入口检测为 `TORCH.SHADOWED`；
- pip/uv 中途失败不切换 current；
- quarantine 后可从最后成功 lock 重建；
- 项目镜像损坏后从父摘要和项目锁恢复；
- 父标签漂移不改变 digest 绑定。

### 14.3 宿主与硬件测试

- fixtures 覆盖完整模式、容器模式、重启和阻断主机；
- 目标主机从干净安装状态完成完整模式；
- overlay 安装常用 AI 依赖后运行 Torch FP16、HIP、Triton 和重复初始化；
- 人工制造 overlay shadow，确认阻断、repair 和恢复后 GPU probe；
- 正式发布前重新运行 300 秒压力测试与连续 dmesg 差分。

## 15. 验收标准

1. 新 Ubuntu 24.04.x 主机可通过完整模式完成重启前后流程，无需用户记忆底层命令顺序。
2. 已正确配置的主机可通过容器模式直接拉取 verified digest 并建立项目。
3. 安装器中断、重启或重复运行不会重复应用已确认动作，也不会跳过变化后的输入确认。
4. 用户可在运行容器内执行受支持的常规 `pip install`，非 Torch 包在该项目中跨重启保留。
5. 多个项目的 overlay 独立，不共享可写 Python 环境，也不复制正式 Torch。
6. 错误 Torch 请求在解析阶段失败；绕过产生的 shadow 在项目启动前失败。
7. `/opt/venv` 在托管 verified 容器中不可写，且 manifest 始终可验证。
8. overlay、项目镜像或父镜像任一层损坏时，doctor 给出明确分类和证据。
9. repair 只从已验证 digest 和哈希锁创建新环境；失败不切换、不删除 quarantine。
10. repair 成功后项目数据、模型和显式挂载不丢失，并重新通过 `gfx1151` GPU 检查。

## 16. 文档要求

实现同时更新：

- README 快速安装与安全说明；
- 完整模式和容器模式操作手册；
- 受保护 pip 支持/拒绝参数表；
- “试验依赖转为项目正式锁”的工作流；
- doctor 故障代码与 repair 证据保留说明；
- GHCR digest、release manifest、SBOM 与资格报告关系；
- 只读根下为应用增加显式可写挂载的方法。
