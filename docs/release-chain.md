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

`profiles/releases/stable.json` 中的 GHCR 路径是 canonical 发布身份。
`v0.3.3` 同时信任以下公开华为云 SWR 副本，仓库名不同，但 manifest、
config、artifact digest 和 OCI labels 必须与 canonical release 完全一致：

```text
swr.cn-east-3.myhuaweicloud.com/hellcat-home/strix-halo-rocm-python@sha256:e9991f97f578156c8620fbb587d2d34504eb632f165cc5597deaadaa3e692a12
swr.cn-east-3.myhuaweicloud.com/hellcat-home/strix-halo-rocm-pytorch@sha256:dc0bb217474cfd4f602423bd3bf4fe8714b03e900cf3c6b4417b99e622ebcf8b
```

默认 `--registry auto` 先匿名访问 SWR，仅在获取失败时回退 GHCR。
`--registry swr` 和 `--registry ghcr` 可以强制单一来源。用户不需要任何
registry 凭据。manifest 查询、连接或仓库不可用属于 acquisition failure，
允许按策略回退；digest、config ID、RepoDigest、label 或内嵌锁不一致属于
identity failure，必须立即阻断，不能切换副本掩盖问题。

安装状态保存实际验证成功的 exact reference，而不是强制改写为 canonical
路径。发布门禁仍以 canonical manifest 和资格证据为根，并在中国验收主机上
使用空 Docker 认证目录验证 SWR 匿名冷拉取与 GPU runtime。

大镜像只从可信、已完成硬件资格测试的 gfx1151 主机发布。公开仓库不会让不受信任 pull request 在自托管 GPU runner 上执行；标准 GitHub-hosted runner 的磁盘容量也不作为大镜像发布前提。

本地 build 记录 `installer_source_revision`，但不会伪造资格报告、SBOM 或 registry manifest digest，也不会改写 stable release。`source_revision` 始终表示 stable manifest 绑定的已资格化源码，这两个 revision 不得混为一项身份。
