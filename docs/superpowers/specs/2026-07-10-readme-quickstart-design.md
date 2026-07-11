# README 快速开始重构设计

## 目标

让首次使用 Strix Halo ROCm Toolkit 的用户只阅读 README 顶部的一条连续路径，就能完成以下工作：

1. 确认宿主机满足最低条件；
2. 安装正式 ROCm/PyTorch 容器平台并创建第一个项目；
3. 在需要重启时安全恢复安装；
4. 启动项目 Docker 容器；
5. 确认 PyTorch 正在通过 ROCm 使用 Radeon 8060S；
6. 在不覆盖受保护 Torch 基线的前提下开始安装项目 Python 依赖；
7. 知道如何把试装依赖固化为可复现的项目镜像。

快速路径面向 Ubuntu 24.04.x、AMD Ryzen AI Max+ 395 / Radeon 8060S 的首次部署者。高级自动化、自选 Torch profile、修复和发布验证继续保留在后续章节。

## 信息架构

README 开头依次放置：

1. 项目名称、单句定位和当前正式基线；
2. `快速开始`；
3. 目录；
4. 项目原理、支持范围和完整参考内容。

现有分散在 `安装前准备`、`快速安装`、`安装后验证`、`创建和运行项目`、`Python 依赖与受保护 pip` 的详细内容继续保留，但快速开始不要求用户在这些章节之间跳转。后续章节负责解释和扩展，快速开始负责提供一条可执行主路径。

## 快速开始流程

### 1. 开始前确认

用一个短清单保留不可省略的限制：

- 宿主是 Ubuntu 24.04.x AMD64，硬件是 Ryzen AI Max+ 395 / Radeon 8060S；
- BIOS UMA Frame Buffer 使用主板允许的最小值，建议 512 MiB；
- 用户具有 `sudo`，互联网可用，Docker 数据目录建议至少有 40 GiB 可用空间。

Python 3.12、Git 等缺失依赖的安装命令保留在详细准备章节，快速路径提供链接，不展开故障矩阵。

### 2. 获取固定工具包版本

使用 HTTPS 匿名 clone 固定 `v0.2.3` tag，不使用 `curl | bash`：

```bash
git clone --branch v0.2.3 --depth 1 \
  https://github.com/hellcatjack/strix-halo-rocm-toolkit.git
cd strix-halo-rocm-toolkit
```

### 3. 定义项目并执行完整安装

项目路径只定义一次，后续命令复用同一个 shell 变量：

```bash
PROJECT="$HOME/ai-projects/video-lab"

./install.sh --mode full \
  --project-dir "$PROJECT" \
  --project-name video-lab \
  --image-source pull
```

显式命令是主入口，因为它可审阅、可复制，并且在重启或失败后能够原样恢复。`./install.sh` 交互菜单作为备选入口放在命令之后。

### 4. 处理重启或失败

当安装器显示 `ACTION` 和 `RESUME` 要求重启时，用户执行：

```bash
sudo reboot
```

重启后重新进入同一个 checkout，重新定义 `PROJECT`，原样执行完整安装命令。安装器根据项目状态跳过可信检查点。

普通失败只要求用户读取终端中的 `CAUSE`、`STATE`、`RESUME` 和 `LOG`，修复原因后原样重跑。快速开始不建议删除状态文件、不建议切换模式，也不建议创建新的状态路径。

### 5. 启动项目容器

安装成功后确保用户 launcher 可见：

```bash
export PATH="$HOME/.local/bin:$PATH"
strix-halo-rocm --version
```

launcher 始终按普通用户运行。若用户没有授权加入 Docker 组，只在启动前刷新 sudo 凭据：

```bash
sudo -v
strix-halo-rocm project run "$PROJECT"
```

若 Docker 可由当前用户直接访问，则不需要 `sudo -v`。文档不得建议执行 `sudo strix-halo-rocm`。

`project run` 在必要时构建项目镜像，然后启动映射 `/dev/kfd` 和 DRM render node 的 Docker 容器。项目 entrypoint 在交出 Bash 前自动执行 `container-check --mode torch --runtime`，检查 Torch manifest、有效 overlay、ROCm GPU 可用性和真实 GPU 运算；检查失败时不进入项目 shell。

### 6. 在容器内确认 PyTorch GPU

进入 Bash 后运行一个可复制的 Python 探针。探针必须：

- 打印 `torch.__version__` 和 `torch.version.hip`；
- 断言 `torch.cuda.is_available()`；
- 打印 `torch.cuda.get_device_name(0)`；
- 在 `cuda` 设备创建 tensor、执行运算并调用 `torch.cuda.synchronize()`；
- 打印结果所在设备，避免把仅能 import Torch 误认为 GPU 已可用。

ROCm 版 PyTorch 仍使用 `torch.cuda` API，README 在代码块前明确说明这一点。

### 7. 开始安装项目依赖

GPU 探针通过后，快速路径直接给出：

```bash
pip install transformers safetensors
pip check
```

随后说明：

- 直接 `pip install` 写入项目目录中的持久 overlay，退出临时容器后仍保留；
- `torch`、`torchvision`、`torchaudio` 和 `triton` 受到保护，不能被普通项目安装覆盖；
- 用户可以继续安装自己的普通依赖，不应创建另一个包含 Torch 的 venv；
- 试装稳定后，将直接依赖写入 `requirements.in`，退出容器并运行 `project lock` 与 `project run --build`，把环境固化为可复现项目镜像。

快速路径以一段结果清单结束，明确用户此时已经拥有可运行的 ROCm/PyTorch GPU 容器、独立项目目录和可继续使用的受保护 Python 环境。

## 错误引导

快速路径只覆盖最常见分支，并链接到完整章节：

| 场景 | 快速处理 |
| --- | --- |
| 安装器要求重启 | `sudo reboot` 后原样执行同一安装命令 |
| 安装阶段失败 | 按 `CAUSE` 修复，保留状态，按 `RESUME` 重跑 |
| Docker 权限不足 | 重新登录刷新组，或先执行 `sudo -v` |
| 项目启动 GPU 检查失败 | 不继续安装依赖，先运行 `doctor` 并查看 GPU 检查报告 |
| pip 请求替换 Torch | 保留正式基线；需要其他 Torch 时使用自定义完整 profile |

详细恢复、诊断和自选 Torch 流程仍由现有专项章节负责，快速开始不复制完整故障表。

## 一致性约束

- 快速路径使用正式 stable 名称和版本：ROCm 7.2.1、Python 3.12、PyTorch 2.9.1；
- 所有宿主路径都引用同一个 `PROJECT` 变量；
- 容器内命令与宿主命令必须明确分开，避免用户在错误环境执行；
- 不把 `container-check` 的自动检查描述成可选检查；
- 不声称 PyTorch 需要单独的 GPU “激活”步骤；成功标准是真实 GPU tensor 运算通过；
- 不引导用户在宿主 Python 安装 ROCm Torch；
- 不改变镜像、安装器、项目模板或依赖保护行为。

## 验证

实施后执行以下验证：

1. 对 README 运行项目现有 Markdown lint；
2. 检查目录锚点和文档链接；
3. 对照 CLI parser、项目模板和 entrypoint 复核每条命令与行为描述；
4. 检查代码块的宿主/容器上下文、引号和续行符；
5. 运行 README/安装器相关测试，确认版本和命令示例没有破坏既有契约；
6. 使用 `git diff --check` 排除空白错误。

## 非目标

- 不修改安装器交互、Docker 启动逻辑或 Torch 保护策略；
- 不预装 ComfyUI、Hugging Face 或其他 AI 应用；
- 不默认共享模型和缓存；
- 不发布新镜像、不改变 stable release digest；
- 不把其他 Linux 发行版描述为受支持的宿主自动安装目标。
