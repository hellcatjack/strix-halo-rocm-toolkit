# Doctor 与 Repair

```bash
strix-halo-rocm doctor [PROJECT]
strix-halo-rocm repair PROJECT
```

Doctor 是只读检查。Repair 先生成只含精确目标的计划；交互模式要求输入 `REPAIR`，自动化必须显式使用 `--yes`。

## 稳定诊断码

| 代码 | 典型处置 |
| --- | --- |
| `PLATFORM.PASS` | pass |
| `RELEASE.INVALID` | blocked |
| `HOST.PREFLIGHT_FAILED` | blocked |
| `IMAGE.PARENT_MISSING` | repairable，拉取 exact digest |
| `IMAGE.DIGEST_DRIFT` | repairable，核对后重绑本地标签 |
| `IMAGE.PROJECT_CHANGED` | repairable，只移除记录的项目 image ID |
| `PROJECT.CONFIG_INVALID` | blocked |
| `TORCH.BASE_CHANGED` | repairable，恢复不可变父镜像 |
| `TORCH.SHADOWED` | repairable，隔离 overlay 后离线重放锁 |
| `OVERLAY.LOCK_INVALID` | repairable |
| `OVERLAY.TRANSACTION_INCOMPLETE` | warning |
| `GPU.RUNTIME_FAILED` | blocked |
| `KERNEL.LOG_FAILED` | blocked |

## 精确动作

允许的动作只针对一个项目 generation、一个 exact `sha256:` 本地 image ID 或一个 exact `image@sha256:` registry reference。损坏 overlay 会移动到项目 `.amd-ai/quarantine` 留证；可恢复时使用最后有效的 `overlay.requirements.lock` 和本地 artifact store，在 `--network none`、只读父镜像下重放。

父镜像变化时匿名重新拉取并验证 exact digest。项目镜像变化时先按当前指纹构建并验证替代镜像；只有成功取得不同的新 exact ID 后，才删除 doctor 记录的旧 exact ID。构建失败时保留旧 image 作为证据并维持项目 blocked。每次修复后重新运行受管项目启动和 effective Torch 检查。

Repair 永远不会运行 `docker system prune`、通配 image 删除、未限定 cache 清理或 force-reinstall Torch。证据目录不会被成功修复自动删除。
