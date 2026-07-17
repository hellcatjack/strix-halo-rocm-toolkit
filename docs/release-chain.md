# Stable Release 身份链

正式发布由 `profiles/releases/stable.json` 绑定以下身份：

| 身份 | 含义 |
| --- | --- |
| source revision | 经过资格测试的 40 位 Git commit |
| qualification profile digest | 测试参数文件 SHA-256 |
| qualification report digest | 完整 gfx1151 报告 SHA-256 |
| SBOM digest | SPDX JSON 摘要 |
| OCI manifest digest | registry 内容寻址入口，也称 manifest digest |
| Docker local image ID | 本机 Docker 后端用于寻址镜像的不可变 ID；可能是 config 或 manifest digest |
| OCI config digest | 解压后镜像配置身份，也称 config digest |
| parent manifest/config pair | 项目配置锁定的 Torch 父层 manifest 与 config digest |
| project image ID | 当前项目上下文派生的不可变本地 ID |

manifest digest 与 config digest 不是同一值，不能互换。安装器按 `image@sha256:...` 拉取 OCI manifest，并接受 Docker inspect 根据存储后端返回该 manifest digest 或对应 config digest；两种情况都会继续核对固定 manifest 的 config descriptor、RepoDigests、OCI labels 和内嵌 profile/lock/manifest 摘要。

默认 registry 是公开 GHCR。安装器使用临时空 Docker 配置执行 anonymous pull，证明部署不依赖维护者凭据；发布门禁还会在独立空认证目录中再次拉取两个 exact reference。

大镜像只从可信、已完成硬件资格测试的 gfx1151 主机发布。公开仓库不会让不受信任 pull request 在自托管 GPU runner 上执行；标准 GitHub-hosted runner 的磁盘容量也不作为大镜像发布前提。

本地 build 记录 `installer_source_revision`，但不会伪造资格报告、SBOM 或 registry manifest digest，也不会改写 stable release。`source_revision` 始终表示 stable manifest 绑定的已资格化源码，这两个 revision 不得混为一项身份。
