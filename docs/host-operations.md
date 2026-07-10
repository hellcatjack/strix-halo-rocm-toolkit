# Ryzen AI Max+ 395 主机运维手册

本文适用于 Ubuntu 24.04.x AMD64、Ryzen AI Max+ 395 / Radeon 8060S。正式容器基线为 ROCm 7.2.1；宿主机使用 Ubuntu OEM 内核自带的 `amdgpu`，不安装完整 ROCm 用户态，也不安装 `amdgpu-dkms`。

## 1. 安全边界

- `host-preflight` 只读运行；未知发行版也可用于采集报告，但会得到 `HOST.UNSUPPORTED_OS`，且不能执行主机写入。
- `host-prepare plan` 只生成动作，不修改主机。
- `host-prepare apply` 必须以 root 执行，并要求精确输入 `APPLY`。`--yes` 只确认动作计划。
- 加入 `docker` 组另需精确输入 `ADD_DOCKER_GROUP`。该组可控制 Docker daemon，权限近似主机 root；其他输入保留 `sudo docker`。
- 不传 `--reboot` 时，应用器不会执行重启。建议检查报告后自行运行 `sudo reboot`。
- 工具不会自动回滚已安装内核，也不会自动删除通用 `dkms`、`zfs-dkms`、`virtualbox-dkms` 或其他厂商模块。
- 旧包清理仅选择 `amdgpu-dkms`，以及版本以 6.4 开头、来源可确认是 `repo.radeon.com` 的 ROCm/HIP/HSA 白名单包；不执行 `apt autoremove`。

## 2. BIOS 与统一内存

进入 BIOS/UEFI，将 UMA Frame Buffer 或 Dedicated VRAM 设置为最小可用值 **512 MiB（0.5 GiB）**。脚本只能检查内核报告的专用 VRAM，不能修改 BIOS。

Ryzen AI Max+ 395 使用统一物理内存。512 MiB 是固定帧缓冲基线，其余内存保持为普通系统 RAM，并由 GPU 通过 TTM/GTT 按需映射。`ttm.pages_limit` 是可映射上限，不会在开机时立即、永久占用同等容量。

本机标称 128 GiB、4 KiB 页的目标为：

```text
ttm.pages_limit=33554432
amdgpu.gttsize=131072  # 仅保留已有兼容参数，不在新主机上主动新增
```

容量优先使用 DMI；DMI 不可用时使用 `/proc/meminfo`，并向上归一到 8 GiB 档。两者推导结果相差超过 8 GiB 时会停止自动应用，需使用 `--memory-gib` 明确确认。已有 `amdgpu.gttsize`、`amdgpu.cwsr_enable`、`amdgpu.mcbp` 和 `amdgpu.gpu_recovery` 参数会保留，不由本流程顺带改写。

## 3. 标准流程

### 3.1 只读预检

```bash
mkdir -p reports
./bin/host-preflight --json reports/preflight.json
```

退出码：`0` 表示通过或明确的 `unverified`，`1` 表示需要修改/重启，`2` 表示阻断。`6.17.0-1025-oem` 满足最低内核要求，但在纳入版本化已测内核清单前会报告 `HOST.UPSTREAM_UNVERIFIED`。

预检核对：

```bash
id
ls -l /dev/kfd /dev/dri/render* 2>/dev/null
lspci -Dnnk -d 1002:1586
cat /proc/cmdline
cat /sys/module/ttm/parameters/pages_limit 2>/dev/null || \
  cat /sys/module/amdttm/parameters/pages_limit
```

在未映射 `/dev/kfd` 的容器中运行预检，只能证明当前容器看不到设备，不能单独证明宿主驱动失败。正式宿主报告应直接在 Ubuntu 主机上生成。

### 3.2 查看准备计划

```bash
sudo ./bin/host-prepare plan --target-user "$USER" \
  --json reports/host-prepare-plan.json
```

确认输出中的包名、APT 源、目标用户、128 GiB TTM 计算和重启动作。若 DMI 信息异常，可在核实物理容量后显式使用：

```bash
sudo ./bin/host-prepare plan --target-user "$USER" --memory-gib 128
```

### 3.3 应用

```bash
sudo ./bin/host-prepare apply --target-user "$USER" \
  --json reports/host-prepare-apply.json
```

输入 `APPLY` 后才会执行。主要顺序为：

1. 创建私有备份和命令快照。
2. 禁用确认属于 ROCm 6.4 的源，清理确认的旧包和 `amdgpu-dkms`。
3. 安装 `linux-oem-24.04`、匹配 headers、firmware 和主机工具。
4. Docker 缺失时，校验官方密钥完整指纹后安装 Engine、Buildx 和 Compose 插件。
5. 补充实际 `/dev/kfd`、render node 所属设备组。
6. 校验固定 SHA-256 后安装 `amd-debug-tools==0.2.19`，设置 AI Max TTM 上限。

`amd-ttm` 的重启提示始终由包装器拒绝，重启策略由本工具统一控制。仅在该版本因“归一后的标称容量略高于可见 MemTotal”而拒绝时，才写入等价的 `ttm.conf`；其他错误立即停止。

自动化确认可使用 `--yes`，但它不会替用户授权 Docker 组，也不会自动重启：

```bash
sudo ./bin/host-prepare apply --target-user "$USER" --yes
```

### 3.4 重启

检查应用报告后执行：

```bash
sudo reboot
```

只有明确希望应用器在全部动作成功后立即重启时，才向 `host-prepare apply` 传入 `--reboot`。

### 3.5 构建探针镜像并验证

```bash
./bin/image-build rocm-python
./bin/host-verify \
  --probe-image rocm-python:7.2.1-py3.12 \
  --json reports/host-verify.json
```

验证要求：

- live TTM 页数等于按物理内存和实际页大小计算的目标；
- `/dev/kfd` 和 `/dev/dri/render*` 均存在，目标用户覆盖实际设备 GID；
- 探针使用 `--device /dev/kfd`、`--device /dev/dri` 和动态 `--group-add`，不使用 `--privileged` 或 `--ipc=host`；
- 容器输出包含 `gfx1151`；
- 当前启动的 `dmesg` 中没有 MES timeout、GPU reset、amdgpu page fault、firmware 加载失败或 ring timeout。

`host-verify` 只有在宿主检查和容器探针均正式通过时返回 0。满足最低要求但未登记的 OEM 内核保持 `unverified` 并返回 1；阻断错误返回 2。

## 4. 备份与恢复

每次 apply 首先创建：

```text
/var/backups/amd-ai/<UTC时间戳>/
```

目录权限为 `0700`，文件为 `0600`。`manifest.json` 记录已复制文件和 `dpkg-query`、DKMS、内核、GPU、Docker 命令输出。查找最近备份：

```bash
sudo ls -1dt /var/backups/amd-ai/*
sudo less /var/backups/amd-ai/<UTC时间戳>/manifest.json
```

恢复 TTM 配置前先停止 GPU 工作负载。若备份中原来存在该文件：

```bash
BACKUP=/var/backups/amd-ai/<UTC时间戳>
sudo install -m 0644 "$BACKUP/etc/modprobe.d/ttm.conf" \
  /etc/modprobe.d/ttm.conf
sudo update-initramfs -u
sudo reboot
```

若备份清单显示原来没有 `/etc/modprobe.d/ttm.conf`，而本次应用新建了它，可在核对内容后删除，再执行 `update-initramfs -u` 和重启。

恢复 GRUB 配置：

```bash
BACKUP=/var/backups/amd-ai/<UTC时间戳>
sudo install -m 0644 "$BACKUP/etc/default/grub" /etc/default/grub
sudo update-grub
sudo reboot
```

源文件备份位于相同相对路径。应用器将旧 ROCm 源重命名为 `.amd-ai-disabled`；恢复前需同时检查备份、禁用文件和当前发行版源，避免重新启用不兼容的 6.4 仓库。

内核包安装不自动回滚。内核无法启动时，从 GRUB 的 Advanced options 选择先前可启动内核，再根据 Ubuntu 包状态人工处理；不要在无法确认其他 DKMS 依赖时删除通用 `dkms`。

## 5. 不受支持的发行版

公共采集器可以只读检查其他 Linux：

```bash
./bin/host-preflight --json reports/unsupported-host.json
```

报告会保留硬件、内存、设备和容器 runtime 事实，同时返回 `HOST.UNSUPPORTED_OS` 或 `HOST.UNSUPPORTED_ARCH`。没有显式匹配的发行版适配器时，`host-prepare plan/apply` 均拒绝生成写入计划。不要通过修改 `/etc/os-release` 或绕过 CLI 强行套用 Ubuntu APT、内核及 systemd 动作。

