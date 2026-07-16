# Ryzen AI Max+ 395 主机运维手册

本文适用于 Ubuntu 24.04.x AMD64、Ryzen AI Max+ 395 / Radeon 8060S。正式容器基线为 ROCm 7.2.1；宿主机使用 Ubuntu OEM 内核自带的 `amdgpu`，不安装完整 ROCm 用户态，也不安装 `amdgpu-dkms`。

## 1. 安全边界

- `host-preflight` 只读运行；未知发行版也可用于采集报告，但会得到 `HOST.UNSUPPORTED_OS`，且不能执行主机写入。
- `host-prepare plan` 只生成动作，不修改主机。
- `host-prepare apply` 必须以 root 执行，并要求精确输入 `APPLY`。`--yes` 只确认动作计划。
- `HOST.UNSUPPORTED_OS`、`HOST.UNSUPPORTED_ARCH`、`GPU.NOT_FOUND` 或 `GPU.WRONG_DRIVER` 属于不可由准备动作修复的阻断项；`plan` 与 `apply` 都会在任何主机写入前停止。旧内核等可修复状态仍可生成计划。
- 加入 `docker` 组另需精确输入 `ADD_DOCKER_GROUP`。该组可控制 Docker daemon，权限近似主机 root；其他输入保留 `sudo docker`。
- 不传 `--reboot` 时，应用器不会执行重启。建议检查报告后自行运行 `sudo reboot`。
- 工具不会自动回滚已安装内核，也不会自动删除通用 `dkms`、`zfs-dkms`、`virtualbox-dkms` 或其他厂商模块。
- 旧包清理仅选择 `amdgpu-dkms`，以及版本以 6.4 开头、来源可确认是 `repo.radeon.com` 的 ROCm/HIP/HSA 白名单包；不执行 `apt autoremove`。
- 只有全部 active HTTP 源都明确属于 Radeon/ROCm 6.4 的单一 APT 文件才会整体禁用。一个文件同时包含 6.4 和其他仓库时，工具拒绝改名，操作员必须先人工删除对应行或 deb822 stanza，再重新运行预检和计划。

## 2. BIOS 与统一内存

进入 BIOS/UEFI，将 UMA Frame Buffer 或 Dedicated VRAM 设置为最小可用值 **512 MiB（0.5 GiB）**。脚本只能检查内核报告的专用 VRAM，不能修改 BIOS。

Ryzen AI Max+ 395 使用统一物理内存。512 MiB 是固定帧缓冲基线，其余内存保持为普通系统 RAM，并由 GPU 通过 TTM/GTT 按需映射。BIOS 设置属于人工固件操作，不是安装器的宿主调优步骤。

工具不会安装 `amd-debug-tools`、调用 `amd-ttm`、创建、替换或删除 `/etc/modprobe.d/ttm.conf`，也不会设置 `ttm.pages_limit`、`amdgpu.gttsize` 或因 GTT/TTM 执行 `update-initramfs`/重启。live TTM 页数、内核命令行和相关日志只作为诊断事实采集，不计算目标值，不参与安装通过判定。

## 3. 标准流程

### 3.1 只读预检

```bash
mkdir -p reports
./bin/host-preflight --json reports/preflight.json
```

退出码：`0` 表示通过或明确的 `unverified`，`1` 表示需要修改/重启，`2` 表示阻断。高于最低版本但未纳入版本化已测内核清单的 OEM patch kernel 会报告 `HOST.UPSTREAM_UNVERIFIED`。

只有当前内核低于最低要求或不是 OEM kernel 时，才应引导安装/启动 `linux-oem-6.17`。对更新但尚未登记的 OEM patch kernel，不自动升级、降级或固定旧包；先执行设备与容器探针，普通安装可带 `unverified` 记录继续，正式发布前再运行完整硬件资格测试。

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

确认输出中的阶段、包名、APT 源、目标用户和重启动作。旧于 6.17 的内核会生成内核阶段计划；启动 6.17 OEM 内核后再次运行，才会生成不重启的平台阶段计划。任何计划都不包含 GTT/TTM 写入。

### 3.3 应用

```bash
sudo ./bin/host-prepare apply --target-user "$USER" \
  --json reports/host-prepare-apply.json
```

输入 `APPLY` 后才会执行。完整安装器将以下动作拆为两个独立审批阶段：

1. 内核阶段创建私有备份，禁用确认属于 ROCm 6.4 的源，清理确认的旧包和 `amdgpu-dkms`，安装 OEM 6.17 内核、匹配 headers 与 firmware。
2. 一次手工重启后，严格验证当前内核、显示管理器、Radeon 8060S、KFD/render node 和当前启动日志。
3. 平台阶段创建新的快照，安装宿主诊断工具；Docker 缺失时安装 Engine、Buildx 和 Compose 插件，Buildx 缺失时使用与当前 Docker 发行方式匹配的包修复。
4. 按 `--target-user` 的 passwd/group 数据补充实际 `/dev/kfd`、render node 所属设备组，不以调用 sudo 的 root 进程组作为判断依据。

平台阶段不要求重启。工具既不调节 GTT/TTM，也不会为 Docker、Buildx 或组变更创建第二个重启检查点。

自动化确认可使用 `--yes`，但它不会替用户授权 Docker 组，也不会自动重启：

```bash
sudo ./bin/host-prepare apply --target-user "$USER" --yes
```

### 3.4 重启

检查应用报告后执行：

```bash
sudo reboot
```

只有直接使用低级 `host-prepare apply`，并明确希望内核阶段成功后立即重启时，才传入 `--reboot`。推荐的 `install.sh --mode full` 永远由操作员手工重启，以便保留桌面恢复窗口。

### 3.5 构建探针镜像并验证

```bash
./bin/image-build rocm-python
sudo -v
./bin/host-verify \
  --probe-image rocm-python:7.2.1-py3.12 \
  --json reports/host-verify.json
```

验证要求：

- `/dev/kfd` 和 `/dev/dri/render*` 均存在，目标用户覆盖实际设备 GID；
- 探针使用 `--device /dev/kfd`、`--device /dev/dri` 和动态 `--group-add`，不使用 `--privileged` 或 `--ipc=host`；
- 容器输出包含 `gfx1151`；
- 当前启动的 `dmesg` 中没有 MES timeout、GPU reset、amdgpu page fault、firmware 加载失败或 ring timeout。

报告继续包含 `ttm_pages_limit`、当前内核命令行和 TTM 日志事实，但这些只用于排障。缺失或不同的 live limit 不会产生 `HOST.TTM_MISMATCH` 或 `HOST.MEMORY_CONFLICT`。

`host-verify` 本身保持以普通目标用户运行，以便正确判断该用户的设备组；普通 `dmesg` 或 Docker daemon 不可访问时，只回退到固定的 `sudo -n dmesg` 与 `sudo -n docker` 命令。先运行 `sudo -v` 可刷新凭据，但不会让整条验证流程以 root 身份误判用户权限。两条路径都无法读取 dmesg 时会得到 `HOST.DMESG_UNAVAILABLE`，不会把空输出当成“没有 GPU 错误”。独立 `host-verify` 命令只有在宿主检查和容器探针均正式通过时返回 0；满足最低要求但未登记的 OEM 内核保持 `unverified` 并返回 1，阻断错误返回 2。交互安装器从 `v0.2.1` 起会明确记录该 `unverified` 状态并继续普通部署。

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

备份可能包含安装前已经存在的 `/etc/modprobe.d/ttm.conf`，仅用于保留诊断证据。工具从不创建、替换或删除该文件，因此没有由本工具产生的 TTM 回滚步骤。若操作员或其他软件修改过它，应按该修改来源的文档单独处理，不能归因于本工具。

恢复 GRUB 配置：

```bash
BACKUP=/var/backups/amd-ai/<UTC时间戳>
sudo install -m 0644 "$BACKUP/etc/default/grub" /etc/default/grub
sudo update-grub
sudo reboot
```

源文件备份位于相同相对路径。应用器只会把内容完全属于旧 ROCm 6.4 的源文件重命名为 `.amd-ai-disabled`；混合源文件必须人工按行或 stanza 清理。恢复前需同时检查备份、禁用文件和当前发行版源，避免重新启用不兼容的 6.4 仓库。

内核包安装不自动回滚。内核无法启动时，从 GRUB 的 Advanced options 选择先前可启动内核，再根据 Ubuntu 包状态人工处理；不要在无法确认其他 DKMS 依赖时删除通用 `dkms`。

## 5. 不受支持的发行版

公共采集器可以只读检查其他 Linux：

```bash
./bin/host-preflight --json reports/unsupported-host.json
```

报告会保留硬件、内存、设备和容器 runtime 事实，同时返回 `HOST.UNSUPPORTED_OS` 或 `HOST.UNSUPPORTED_ARCH`。没有显式匹配的发行版适配器时，`host-prepare plan/apply` 均拒绝生成写入计划。不要通过修改 `/etc/os-release` 或绕过 CLI 强行套用 Ubuntu APT、内核及 systemd 动作。
