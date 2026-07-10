# 受保护 pip

项目容器允许直接运行 `pip install`。包装器把普通依赖写入项目私有的事务 generation；父镜像中的 `torch`、`torchvision`、`torchaudio` 和 `triton` 仍由 manifest 保护，不会被多个 venv 重复复制或被临时安装覆盖。

## 支持命令

| 操作 | 示例 | 行为 |
| --- | --- | --- |
| 安装索引包 | `pip install transformers==4.53.0` | 解析后写入新 generation |
| requirements | `pip install -r requirements-extra.txt` | 检查全部直接和传递依赖 |
| 本地源码 | `pip install ./extension` | 构建 wheel 后锁定本地产物摘要 |
| 本地 wheel | `pip install ./dist/pkg.whl` | 复制到项目 artifact store 并哈希 |
| 固定 Git | `pip install 'name @ git+https://host/repo.git@<40-hex>'` | 只接受命名的 exact commit |
| 卸载普通包 | `pip uninstall transformers` | 创建不含该 root 的新 generation |
| 查询 | `pip list`, `pip show`, `pip check`, `pip freeze` | 对当前 effective 环境只读 |

如果请求与父层完全兼容，例如依赖声明已由正式 Torch 版本满足，报告会标记 parent-satisfied，不在 overlay 中再次安装 Torch。

## 拒绝操作

| 类别 | 拒绝内容 |
| --- | --- |
| 替代安装根 | `--user`、`--target`、`--prefix`、`--root` |
| 绕过事务 | editable/`-e`、`--force-reinstall`、`--ignore-installed`、`--no-deps` |
| 不可复现来源 | 未命名 direct URL、branch/tag/HEAD 等可变 VCS 引用 |
| 受保护卸载 | 卸载或替换 Torch、TorchVision、TorchAudio、Triton |

失败 generation 不会切换 `.amd-ai/current`。成功后保存输入、`overlay.requirements.lock`、本地 wheel SHA-256、父 config digest 和健康状态。日志会清除 URL userinfo 与 token/password/secret/key 环境值。

## 提升到项目镜像

overlay 适合容器内试装。确定依赖后：

1. 把经过验证的直接依赖写入项目 `requirements.in`，不要加入受保护 Torch 包。
2. 运行 `strix-halo-rocm project lock PROJECT` 生成项目哈希锁。
3. 运行 `strix-halo-rocm project run PROJECT --build` 重建派生镜像。
4. 验证新镜像后，再用 `pip uninstall` 从 overlay 删除已提升的普通 root。

这样可复现依赖进入项目镜像层，临时试验仍保留在项目私有 overlay 中。
