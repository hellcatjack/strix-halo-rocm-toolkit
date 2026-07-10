# Radeon 8060S GPU 资格与发布手册

本文用于在目标 Ryzen AI Max+ 395 / Radeon 8060S (`gfx1151`) 主机上验证 ROCm 7.2.1、PyTorch 2.9.1 和 Triton 3.5.1 正式组合。测试不接受 CPU fallback，也不会因为 import 成功就判定 GPU 可用。

## 1. 正式门禁范围

`profiles/qualification/stable.toml` 固定八项必需检查：

1. `rocm`：设备映射、HIPCC 和 `rocminfo` 的 `gfx1151` agent。
2. `torch-fp16`：FP16 GEMM、FP16 卷积、同步与误差比较。
3. `hip`：HIPCC 原生编译并运行 `gfx1151` vector-add。
4. `torch-extension`：PyTorch C++/HIP 扩展 hipify、编译和零容差执行。
5. `triton`：Triton 首次 JIT 编译及 GPU add kernel。
6. `repeated-start`：五个全新 Python 进程分别初始化并执行 FP16 matmul。
7. `stress`：300 秒有界 GEMM/卷积持续负载。
8. `kernel-log`：前后 dmesg 差分，阻断新增 MES timeout、GPU reset、page fault、ring timeout 和 firmware 加载失败。

容器均使用 private IPC、16 GiB shm、`/dev/kfd`、`/dev/dri` 和设备实际 GID，不使用 privileged 或 host IPC。资格缓存仅位于 `reports/qualification-cache`；不会挂载模型、ComfyUI 或 Hugging Face 目录。

## 2. 执行正式资格测试

先确认工作树中的实现已经提交，重新构建最终镜像，并确保宿主已按主机手册重启验证。随后执行：

```bash
sudo -v
./bin/host-verify --json reports/host-verify.json
./bin/container-check --suite stable \
  --profile profiles/qualification/stable.toml \
  --json reports/qualification.json
```

`sudo -v` 是为了保证测试前后都能读取内核日志。无法读取 dmesg 会生成阻断项 `HOST.DMESG_UNAVAILABLE`，不会跳过 `kernel-log`。任一低成本检查失败后，编排器停止后续容器，不继续运行压力项，但仍采集测试后的 dmesg。

编排器在启动第一项检查前只解析一次配置中的镜像标签，将本地 `sha256` image ID 写入报告；八项检查随后都直接运行这个不可变 ID。测试期间即使标签被移动，也不会把不同镜像的结果拼成同一份资格报告。前后内核日志为空、无法读取或无法证明连续时分别阻断，不会把清空/轮转后的日志当成干净差分。

成功报告必须满足：

```text
status = pass
profile_id = stable-gfx1151
gpu_arch = gfx1151
image_id = sha256:<64 位摘要>
八项结果各出现一次且 passed = true
```

报告包含 profile SHA-256、每项耗时、最终 JSON、stdout/stderr 和新增 GPU 子系统内核行。`Freeing queue vital buffer`、workqueue latency 等未列入阻断模式的新增行仍保留为证据，不能据此自动添加 workaround。

## 3. 生成 verified 发布记录

资格报告通过后运行：

```bash
./bin/gpu-release \
  --qualification reports/qualification.json \
  --image rocm-pytorch:7.2.1-py3.12-torch2.9.1
```

发布门禁重新验证：

- stable qualification profile 与批准设计文档摘要；
- 八项资格结果及资格报告摘要；
- 镜像 `verified` profile、ROCm 7.2.1、Torch 2.9.1 标签；
- 当前标签解析出的本地不可变镜像 ID 必须与资格报告中的 ID 完全一致，以及存在时的 registry RepoDigest；
- 四个 AMD 主 wheel SHA-256；
- 镜像内 `/opt/amd-ai/locks/rocm-packages.lock` 与仓库批准锁文件的 SHA-256；
- 镜像内 `/opt/amd-ai/profile.env` 与稳定 Torch profile 的 SHA-256；
- 直接从该不可变 ID 生成的 SPDX 2.3 OS/Python 清单摘要；
- 当前 Git revision，且没有已跟踪或未跟踪文件。

所有条件满足后才创建：

```text
rocm-pytorch:7.2.1-py3.12-torch2.9.1-gfx1151-verified
```

工具会重新 inspect 此标签并要求它仍指向资格测试中的同一个本地 image ID。SPDX 和 JSON 先写入临时文件，verified 标签确认后才原子移动到 `reports/releases/`；标签后的产物写入若失败，工具会恢复原 verified 标签或删除新建标签。仅本地发布时 `repo_digest` 为 `null`；推送到 registry 后追加真实 RepoDigest，不得覆盖已经记录的本地 ID。

## 4. 失败留证

发生失败时先保留证据，不立即套用社区补丁。至少保存：

```bash
sudo dmesg --color=never > reports/failure-dmesg.txt
sudo docker image inspect \
  rocm-pytorch:7.2.1-py3.12-torch2.9.1 \
  > reports/failure-image-inspect.json
./bin/host-preflight --json reports/failure-host.json
git rev-parse HEAD > reports/failure-git-revision.txt
```

同时保留：

- `reports/qualification.json` 和失败项完整 stderr；
- `profiles/torch/stable.env`、`stable.requirements.lock` 与 wheelhouse manifest；
- `profiles/rocm/7.2.1-packages.lock`；
- `/proc/cmdline`、OEM kernel 版本和主机包快照；
- 首次出现错误的时间点及对应测试名称。

若第一次通过、第二次失败，应按重复初始化或状态泄漏回归处理，不能用第一次结果覆盖第二次。若第一次慢、第二次仅因编译缓存变快且两次内核日志干净，可记录为正常 warm-cache 差异。

## 5. 单变量 workaround 试验

只有基线报告稳定复现具体 CWSR 或 MES 错误时才试验内核参数：

1. 保存未修改基线报告与 `/proc/cmdline`。
2. 确认候选参数存在于当前内核的 `/sys/module/amdgpu/parameters/`。
3. 在 GRUB 启动项临时编辑中一次只加入一个候选，例如针对 CWSR 的 `amdgpu.cwsr_enable=0`，或针对明确 MES timeout 的 `amdgpu.mes=0`；不要同时加入两项。
4. 启动后记录实际 `/proc/cmdline`，连续运行两次完整资格测试。
5. 从 GRUB 正常启动以移除临时参数，再确认基线行为。

候选参数不存在、错误不能复现、或引入性能/功能回归时，试验无效。不得把 Reddit 或 issue 中的参数直接写入正式默认值。只有可复现故障、单变量改善、两次完整回归和明确适用内核范围都成立后，才单独审查持久化方案。

社区 userspace 补丁只能放入受影响的项目镜像，并在补丁前复现、补丁后回归；不得覆盖稳定 PyTorch 父层。内核补丁不能伪装成项目镜像修复，必须走独立宿主内核审查。

## 6. ComfyUI 视频应用手工资格

`profiles/qualification/comfy-video.example.toml` 只描述两次手工运行和必需观测，不安装 ComfyUI，也不包含仓库、workflow、模型 URL、模型挂载或缓存挂载。操作员先按[项目容器手册](project-workflow.md)建立自己的项目并明确配置这些资源。

两次运行至少记录：首轮/次轮耗时、峰值 GPU 分配、输出正确性和前后 kernel log。首次下载或编译不应被误判为 GPU 回归；第二次显著变慢、失败或出现新增 GPU 阻断日志则不能通过应用资格。

正式基础门禁通过不等于任意 ComfyUI custom node 已获支持。custom node、第三方 wheel 和社区 patch 都属于对应项目镜像的独立回归范围。
